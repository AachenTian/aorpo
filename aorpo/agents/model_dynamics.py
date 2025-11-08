# aorpo/agents/model_dynamics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional, Any

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
    obs_mean: jnp.ndarray
    obs_std: jnp.ndarray
    a_ego_mean: jnp.ndarray
    a_ego_std: jnp.ndarray
    a_opp_mean: jnp.ndarray
    a_opp_std: jnp.ndarray
    delta_mean: jnp.ndarray
    delta_std: jnp.ndarray
    eps: float = 1e-6

    def __hash__(self)-> int:
        return id(self)

    @classmethod
    def fit(cls, obs: jnp.ndarray, a_ego: jnp.ndarray, a_opp:jnp.ndarray, next_obs: jnp.ndarray) -> "Standardizer":
        delta = next_obs - obs
        def _ms(x):
            return jnp.mean(x, axis=0), jnp.std(x, axis=0) + 1e-6

        om, os = _ms(obs)
        aem, aes = _ms(a_ego)
        aom, aos = _ms(a_opp)
        dm, ds = _ms(delta)
        return cls(om, os, aem, aes, aom, aos, dm, ds)

    def norm_obs(self, x):  return (x - self.obs_mean) / (self.obs_std + self.eps)
    def denorm_obs(self, x): return x * (self.obs_std + self.eps) + self.obs_mean
    def norm_a_ego(self, x):  return (x - self.a_ego_mean) / (self.a_ego_std + self.eps)
    def norm_a_opp(self, x):  return (x - self.a_opp_mean) / (self.a_opp_std + self.eps)
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
    def __call__(self, x: jnp.ndarray):
        h = x
        for d in self.hidden_dims:
            h = nn.relu(nn.Dense(d)(h))
        mu = nn.Dense(self.out_dim)(h)
        logvar = nn.Dense(self.out_dim)(h)
        logvar = jnp.clip(logvar, self.min_logvar, self.max_logvar)
        return mu, logvar



class EnsembleDynamics(nn.Module):
    num_members: int
    hidden_dims: Sequence[int]
    out_dim: int # must be set to obs_dim
    min_logvar: float = -10.0
    max_logvar: float = 0.5

    def setup(self):
        self.member = nn.vmap(
            SingleDynamics,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_members,
        )(hidden_dims=self.hidden_dims, out_dim=self.out_dim,
          min_logvar=self.min_logvar, max_logvar=self.max_logvar)

    def __call__(self, x: jnp.ndarray):
        mu, logvar = self.member(x, axis_name='ensemble')
        return mu, logvar


# -------------------------------
# Train utilities
# -------------------------------

def init_model(rng: Any, obs_dim:int, act_dim: int, opp_dim: int, cfg: DictConfig):
    model = EnsembleDynamics(
        num_members=cfg.num_members,
        hidden_dims=tuple(cfg.hidden_dims),
        out_dim=obs_dim,
        min_logvar=cfg.min_logvar,
        max_logvar=cfg.max_logvar,
    )
    rng, init_rng = jax.random.split(rng)
    dummy_in = jnp.zeros((1, obs_dim + act_dim + opp_dim), dtype=jnp.float32)
    params = model.init(init_rng, dummy_in)['params']
    tx = optax.adam(cfg.lr)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state

# Loss & Train
def _nll(mu, logvar, target):
    """Gaussian NLL per (E,B,D), reduce over D then mean over E,B."""
    inv_var = jnp.exp(-logvar)
    inv_var = jnp.clip(inv_var, 0, 1e3)
    mse = (mu - target) ** 2
    nll_dim = 0.5 * (mse * inv_var + logvar + jnp.log(2.0 * jnp.pi))
    nll_b = jnp.sum(nll_dim, axis=-1)
    nll = jnp.mean(nll_b, axis=-1)
    return jnp.mean(nll)



def train_step(state: TrainState,
               batch: dict,
               std: Standardizer):
    """
    :param state: dynamics TrainState
    :param batch: dict with keys {obs, act, next_obs}
    :param std:   Standardizer (静态)
    :return: new_state, metrics
    """
    def loss_fn(params):
        # standardize inputs/targets
        obs_n = std.norm_obs(batch['obs'])
        a_ego_n = std.norm_a_ego(batch['a_ego'])
        a_opp_n = std.norm_a_opp(batch['a_opp'])
        delta = batch['next_obs'] - batch['obs']
        delta_n = std.norm_delta(delta)

        x = jnp.concatenate([obs_n, a_ego_n, a_opp_n], axis=-1)   # (B, obs+act)

        mu, logvar = state.apply_fn({'params': params}, x)  # (E,B,D)
        target = jnp.broadcast_to(delta_n, mu.shape)
        loss = _nll(mu, logvar, target)
        return loss, {'nll': loss}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state)
    return new_state, metrics

train_step = jax.jit(train_step, static_argnums=(2,))



# Prediction & Evaluation
def predict_next(state: TrainState,
                 std: Standardizer,
                 obs: jnp.ndarray,   # (B, obs_dim)
                 a_ego: jnp.ndarray,   # (B, act_dim)
                 a_opp: jnp.ndarray,
                 rng: Optional[Any] = None,
                 deterministic: bool = True,
                 member_idx: Optional[int] = None)-> jnp.ndarray:
    """
    Return predicted next state s' (denormalized).
    - If member_idx is None: 随机选一个 ensemble 成员（需要 rng）
    - deterministic=True: 使用均值；False: 从 N(mu, var) 采样
    """
    a_ego = jnp.asarray(a_ego)
    a_opp = jnp.asarray(a_opp)
    obs = jnp.asarray(obs)
    obs_n = std.norm_obs(obs)
    a_ego_n = std.norm_a_ego(a_ego)
    a_opp_n = std.norm_a_opp(a_opp)
    x = jnp.concatenate([obs_n, a_ego_n, a_opp_n], axis=-1)  # (B, in_dim)

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

def eval_error(real_state:TrainState,###############
               opp_state: TrainState,###################
               std: Standardizer,
               batch: dict,
               rng: Optional[Any] = None,
               deterministic: bool = True,
               member_idx: Optional[int] = None)-> jnp.ndarray:
    """
    Evaluate model prediction error (MSE) on a given batch of real transitions.
    Used to measure model accuracy or adaptive rollout length in AORPO.
    """

    def kl_normal(mu_p, sigma_p, mu_q, sigma_q, eps=1e-6):
        sigma_p = jnp.clip(sigma_p, eps, 1e6)
        sigma_q = jnp.clip(sigma_q, eps, 1e6)
        return jnp.log(sigma_q / sigma_p) + ((sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2.0 * sigma_q ** 2)) - 0.5

    mu_real, std_real = real_state.apply_fn({"params": real_state.params}, batch["obs"])
    mu_opp, std_opp = opp_state.apply_fn({"params":opp_state.params}, batch["obs"])
    std_real = jnp.exp(jnp.clip(std_real, -10.0, 2.0))  # [exp(-10), exp(2)] ≈ [4.5e-5, 7.4]
    std_opp = jnp.exp(jnp.clip(std_opp, -10.0, 2.0))
    kl = kl_normal(mu_real, std_real, mu_opp, std_opp)
    kl = jnp.maximum(jnp.sum(kl, axis=-1), 0.0)
    tv = jnp.sqrt(0.5 * kl)
    tv = jnp.mean(tv)
    return tv # shape:[batch_size]
