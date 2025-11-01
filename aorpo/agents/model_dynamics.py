# aorpo/agents/model_dynamics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from omegaconf import DictConfig


# -------------------------------
# Standardization helpers
# -------------------------------
@dataclass
class Standardizer:
    """Hold mean/std for obs, act, delta (next - obs)."""
    def __init__(self, obs_mean, obs_std, act_mean, act_std, delta_mean, delta_std, eps=1e-6):
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.act_mean = act_mean
        self.act_std = act_std
        self.delta_mean = delta_mean
        self.delta_std = delta_std
        self.eps = eps

    @classmethod
    def fit(cls, obs: jnp.ndarray, act: jnp.ndarray, next_obs: jnp.ndarray) -> "Standardizer":
        delta = next_obs - obs
        def _ms(x):
            return jnp.mean(x, axis=0), jnp.std(x, axis=0) + 1e-6
        om, os = _ms(obs)
        am, as_ = _ms(act)
        dm, ds = _ms(delta)
        return cls(om, os, am, as_, dm, ds)

    def norm_obs(self, x):  return (x - self.obs_mean) / (self.obs_std + self.eps)
    def denorm_obs(self, x): return x * (self.obs_std + self.eps) + self.obs_mean
    def norm_act(self, x):  return (x - self.act_mean) / (self.act_std + self.eps)
    def norm_delta(self, x): return (x - self.delta_mean) / (self.delta_std + self.eps)
    def denorm_delta(self, x): return x * (self.delta_std + self.eps) + self.delta_mean


# -------------------------------
# Networks
# -------------------------------
class SingleDynamics(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int  # = obs_dim (predict delta)
    min_logvar: float = -10.0
    max_logvar: float = 0.5

    @nn.compact
    def __call__(self, x):
        for d in self.hidden_dims:
            x = nn.relu(nn.Dense(d)(x))
        mu = nn.Dense(self.out_dim)(x)
        logvar = nn.Dense(self.out_dim)(x)
        logvar = jnp.clip(logvar, self.min_logvar, self.max_logvar)
        return mu, logvar



class EnsembleDynamics(nn.Module):
    num_members: int = 5
    hidden_dims: Sequence[int] = (256, 256)
    out_dim: int = 0  # must be set to obs_dim
    min_logvar: float = -10.0
    max_logvar: float = 0.5

    def setup(self):
        self.member = nn.vmap(
            SingleDynamics,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None, out_axes=0
        )(hidden_dims=self.hidden_dims, out_dim=self.out_dim,
          min_logvar=self.min_logvar, max_logvar=self.max_logvar)

    def __call__(self, x):
        mu, logvar = self.member(x)
        return mu, logvar


# -------------------------------
# Train utilities
# -------------------------------

def init_model(rng: jax.random.KeyArray, obs_dim:int, act_dim: int, cfg: DictConfig):
    model = EnsembleDynamics(
        num_members=cfg.num_members,
        hidden_dims=tuple(cfg.hidden_dims),
        out_dim=obs_dim,
        min_logvar=cfg.min_logvar,
        max_logvar=cfg.max_logvar,
    )

    dummy_in = jnp.zeros((1, cfg.obs_dim + cfg.act_dim), dtype=jnp.float32)
    params = model.init(rng, dummy_in)['params']
    tx = optax.adam(cfg.lr)
    state = TrainState(step=0, apply_fn=model.apply, params=params, tx=tx, opt_state=tx.init(params))
    return model, state

# Loss & Train
def _nll(mu, logvar, target):
    """Gaussian NLL per (E,B,D), reduce over D then mean over E,B."""
    inv_var = jnp.exp(-logvar)
    inv_var = jnp.clip(inv_var, 0, 1e3)
    mse = (mu - target) ** 2
    nll = 0.5 * (mse * inv_var + logvar)
    nll = jnp.mean(jnp.sum(nll, axis=-1))
    return nll


@jax.jit
def train_step(state: TrainState,
               batch: dict,
               std: Standardizer):
    """
    batch: dict with keys {obs, act, next_obs}, shapes (B, *)
    returns: new_state, metrics
    """
    def loss_fn(params):
        # standardize inputs/targets
        obs_n = std.norm_obs(batch['obs'])
        act_n = std.norm_act(batch['act'])
        delta = batch['next_obs'] - batch['obs']
        delta_n = std.norm_delta(delta)

        x = jnp.concatenate([obs_n, act_n], axis=-1)   # (B, obs+act)

        mu, logvar = state.apply_fn({'params': params}, x)  # (E,B,D)
        # Broadcast target to (E,B,D) for ensemble loss
        target = jnp.broadcast_to(delta_n, mu.shape)
        loss = _nll(mu, logvar, target)
        return loss, {'nll': loss}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state)
    return new_state, metrics


# Prediction & Evaluation
@jax.jit
def predict_next(state: TrainState,
                 std: Standardizer,
                 obs: jnp.ndarray,   # (B, obs_dim)
                 act: jnp.ndarray,   # (B, act_dim)
                 rng: Optional[jax.random.KeyArray] = None,
                 deterministic: bool = True,
                 member_idx: Optional[int] = None):
    """
    Return predicted next state s' (denormalized).
    - If member_idx is None: 随机选一个 ensemble 成员（需要 rng）
    - deterministic=True: 使用均值；False: 从 N(mu, var) 采样
    """
    obs_n = std.norm_obs(obs)
    act_n = std.norm_act(act)
    x = jnp.concatenate([obs_n, act_n], axis=-1)  # (B, in_dim)

    mu, logvar = state.apply_fn({'params': state.params}, x)  # (E,B,D)

    if member_idx is None:
        assert rng is not None, "predict_next: rng is required when member_idx is None."
        rng, sub = jax.random.split(rng)
        member_idx = jax.random.randint(sub, shape=(), minval=0, maxval=mu.shape[0])
    mu_m = mu[member_idx]       # (B,D)
    logvar_m = logvar[member_idx]

    if deterministic:
        delta_n = mu_m
    else:
        assert rng is not None, "predict_next: rng required for stochastic sampling."
        rng, sub = jax.random.split(rng)
        stddev = jnp.exp(0.5 * logvar_m)
        delta_n = mu_m + stddev * jax.random.normal(sub, mu_m.shape)

    delta = std.denorm_delta(delta_n)      # (B,D)
    next_obs = obs + delta
    return next_obs

def eval_error(state: TrainState,
               std: Standardizer,
               batch: dict,
               rng: Optional[jax.random.KeyArray] = None,
               deterministic: bool = True,
               member_idx: Optional[int] = None):
    """
    Evaluate model prediction error (MSE) on a given batch of real transitions.
    Used to measure model accuracy or adaptive rollout length in AORPO.
    """
    # 模型预测下一状态
    pred_next = predict_next(
        state=state,
        std=std,
        obs=batch['obs'],
        act=batch['act'],
        rng=rng,
        deterministic=deterministic,
        member_idx=member_idx
    )

    # 计算真实误差
    mse = jnp.mean((pred_next - batch['next_obs']) ** 2)

    return mse
