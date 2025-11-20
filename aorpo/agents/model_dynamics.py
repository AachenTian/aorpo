# aorpo/agents/model_dynamics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional, Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from omegaconf import DictConfig
from aorpo.utils.replay import ReplayBuffer

# -------------------------------
# Standardization helpers
# -------------------------------
@dataclass
class Standardizer:
    """Hold mean/std for obs, act, delta (next - obs)."""
    state_mean: jnp.ndarray
    state_std: jnp.ndarray
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
    def fit(cls, state: jnp.ndarray, a_ego: jnp.ndarray, a_opp:jnp.ndarray, next_state: jnp.ndarray) -> "Standardizer":
        delta = next_state - state
        def _ms(x):
            return jnp.mean(x, axis=0), jnp.std(x, axis=0) + 1e-6

        sm, ss = _ms(state)
        aem, aes = _ms(a_ego)
        aom, aos = _ms(a_opp)
        dm, ds = _ms(delta)
        return cls(sm, ss, aem, aes, aom, aos, dm, ds)

    def norm_state(self, x):  return (x - self.state_mean) / (self.state_std + self.eps)
    def denorm_state(self, x): return x * (self.state_std + self.eps) + self.state_mean
    def norm_a_ego(self, x):  return (x - self.a_ego_mean) / (self.a_ego_std + self.eps)
    def norm_a_opp(self, x):  return (x - self.a_opp_mean) / (self.a_opp_std + self.eps)
    def norm_delta(self, x): return (x - self.delta_mean) / (self.delta_std + self.eps)
    def denorm_delta(self, x): return x * (self.delta_std + self.eps) + self.delta_mean


# -------------------------------
# State unflatten
# -------------------------------
@dataclass
class State:
    p_pos: jnp.ndarray
    p_vel: jnp.ndarray
    c: jnp.ndarray
    dones: jnp.ndarray
    step: jnp.ndarray

def manual_unflatten_state(flat_state: jnp.ndarray, num_agents: int = 3, num_land: int = 3):
    B = flat_state.shape[0]
    idx = 0
    num_object = num_agents +num_land
    # --- p_pos ---
    p_pos_dim = num_object * 2
    p_pos = flat_state[..., idx:idx + p_pos_dim].reshape(B, num_object, 2)
    idx += p_pos_dim

    # ---p_vel---
    p_vel_dim = num_object * 2
    p_vel = flat_state[..., idx: idx + p_vel_dim].reshape(B, num_object, 2)
    idx += p_vel_dim

    # --- c ---
    c_dim = num_agents * 2
    c = flat_state[..., idx : idx+c_dim].reshape(B, num_agents, 2)
    idx += c_dim

    # --- done ---
    dones = flat_state[..., idx : idx + num_agents].reshape(B, num_agents)
    idx += num_agents

    # --- step ---
    step = flat_state[..., idx:]

    restored_state = State(
        p_pos=p_pos,
        p_vel=p_vel,
        c=c,
        dones=dones,
        step=step
    )
    return restored_state

def unflatten_batch(flat_batch):
    states = []
    for i in range(flat_batch.shape[0]):
        s = manual_unflatten_state(flat_batch[i])   # 输入 shape (1,12) 或 (12,)
        states.append(s)
    # 把 256 个 State 各字段 stack 成 batched state
    return State(
        p_pos=jnp.squeeze(jnp.stack([s.p_pos for s in states]), axis=1),
        p_vel=jnp.squeeze(jnp.stack([s.p_vel for s in states]),axis=1),
        c=jnp.squeeze(jnp.stack([s.c for s in states]), axis=1),
        dones = jnp.squeeze(jnp.array([s.dones for s in states]), axis=1),
        step=jnp.squeeze(jnp.stack([s.step for s in states]),axis=1),
    )

def get_obs(state) -> Dict[str, jnp.ndarray]:
    """计算 batched 状态下每个智能体的观测"""
    num_agents = state.c.shape[1]                     # 第二维是 agent
    num_landmarks = state.p_pos.shape[1] - num_agents

    # === 拆分数据 ===
    agent_pos = state.p_pos[..., :num_agents, :]        # (B, num_agents, 2)
    agent_vel = state.p_vel[..., :num_agents, :]        # (B, num_agents, 2)
    landmark_pos = state.p_pos[..., num_agents:, :]     # (B, num_landmarks, 2)
    comm = state.c[..., :num_agents, :]                 # (B, num_agents, comm_dim)

    obs = {}

    # === 为每个智能体计算观测 ===
    for i in range(num_agents):
        self_pos = agent_pos[..., i, :]                 # (B, 2)
        self_vel = agent_vel[..., i, :]                 # (B, 2)

        # 相对 landmark 位置
        rel_landmark = landmark_pos - self_pos[..., None, :]  # (B, num_landmarks, 2)

        # 相对其他 agent 位置
        other_pos = jnp.concatenate(
            [agent_pos[..., :i, :], agent_pos[..., i + 1:, :]], axis=1
        )                                             # (B, num_agents - 1, 2)
        rel_others = other_pos - self_pos[..., None, :]  # (B, num_agents - 1, 2)

        # 其他 agent 的 communication
        other_comm = jnp.concatenate(
            [comm[..., :i, :], comm[..., i + 1:, :]], axis=1
        )                                             # (B, num_agents - 1, comm_dim)

        # 拼接观测
        obs_i = jnp.concatenate([
            self_vel,                                 # (B, 2)
            self_pos,                                 # (B, 2)
            rel_landmark.reshape(rel_landmark.shape[0], -1),
            rel_others.reshape(rel_others.shape[0], -1),
            other_comm.reshape(other_comm.shape[0], -1),
        ], axis=-1)                                   # (B, obs_dim)

        obs[f"agent_{i}"] = obs_i

    return obs


# -------------------------------
# Networks
# -------------------------------
class SingleDynamics(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int  # = state_dim (predict delta)
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
    out_dim: int # must be set to state_dim
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

def init_model(rng: Any, state_dim:int, num_agents:int, act_dim: int, opp_dim: int, cfg: DictConfig):
    model = EnsembleDynamics(
        num_members=cfg.num_members,
        hidden_dims=tuple(cfg.hidden_dims),
        out_dim=state_dim+num_agents,
        min_logvar=cfg.min_logvar,
        max_logvar=cfg.max_logvar,
    )
    rng, init_rng = jax.random.split(rng)
    dummy_in = jnp.zeros((1, state_dim + act_dim + opp_dim), dtype=jnp.float32)
    params = model.init(init_rng, dummy_in)['params']
    tx = optax.adam(cfg.lr)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state

# Loss & Train
def _nll(mu, logvar, target):
    inv_var = jnp.exp(-logvar)
    inv_var = jnp.clip(inv_var, 1e-6, 1e3)
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

    def agents_dict_to_array(agent_dict: dict) -> jnp.ndarray:
        # 以 agent_0, agent_1, ... 的顺序稳定堆叠
        keys = sorted(agent_dict.keys(), key=lambda s: int(s.split('_')[-1]))
        arrs = [jnp.asarray(agent_dict[k]).reshape(-1) for k in keys]  # 每个是 (B,)
        return jnp.stack(arrs, axis=-1)  # -> (B, num_agents)

    def loss_fn(params):
        # === standardize inputs/targets ===
        state_n = std.norm_state(batch['state'])
        a_ego_n = std.norm_a_ego(batch['a_ego'])
        a_opp_n = std.norm_a_opp(batch['a_opp'])

        delta = batch['next_state'] - batch['state']
        delta_n = std.norm_delta(delta)
        # === input ===
        x = jnp.concatenate([state_n, a_ego_n, a_opp_n], axis=-1)   # (B, obs+act)
        # === predict of model ===
        mu, logvar = state.apply_fn({'params': params}, x)  # (E,B,D)
        # E, B, D_out = mu.shape
        # === target ===
        rew_target = agents_dict_to_array(batch['rew']).astype(jnp.float32)  # (B, num_agents)
        target = jnp.concatenate([delta_n, rew_target], axis=-1)
        target = jnp.broadcast_to(target, mu.shape)
        loss = _nll(mu, logvar, target)
        mse = jnp.mean((mu - target) ** 2)
        logvar = jnp.mean(logvar)
        metrics = {"nll": loss, "mse": mse, "logvar": logvar}
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state)
    return new_state, metrics

train_step = jax.jit(train_step, static_argnums=(2,))



# Prediction & Evaluation
def predict_next(state: TrainState,
                 std: Standardizer,
                 state_agent: jnp.ndarray,   # (B, obs_dim)
                 a_ego: jnp.ndarray,   # (B, act_dim)
                 a_opp: jnp.ndarray,
                 rng: Optional[Any] = None,
                 deterministic: bool = True,
                 member_idx: Optional[int] = None)-> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], Dict[str,jnp.ndarray], Dict[str,jnp.ndarray]]:
    """
    Return predicted next state s' (denormalized).
    - If member_idx is None: 随机选一个 ensemble 成员（需要 rng）
    - deterministic=True: 使用均值；False: 从 N(mu, var) 采样
    """
    a_ego = jnp.asarray(a_ego)
    a_opp = jnp.asarray(a_opp)
    state_agent = jnp.asarray(state_agent)
    state_agent_n = std.norm_state(state_agent)
    a_ego_n = std.norm_a_ego(a_ego)
    a_opp_n = std.norm_a_opp(a_opp)
    x = jnp.concatenate([state_agent_n, a_ego_n, a_opp_n], axis=-1)  # (B, in_dim)

    mu, logvar = state.apply_fn({'params': state.params}, x)  # (E,B,D)

    if member_idx is None:
        assert rng is not None, "predict_next: rng is required when member_idx is None."
        rng, sub = jax.random.split(rng)
        member_idx = jax.random.randint(sub, shape=(), minval=0, maxval=mu.shape[0])
    mu_m = mu[member_idx]       # (B,D)
    logvar_m = logvar[member_idx]

    if deterministic:
        delta_and_rew = mu_m
    else:
        assert rng is not None, "predict_next: rng required for stochastic sampling."
        rng, sub = jax.random.split(rng)
        stddev = jnp.exp(0.5 * logvar_m)
        delta_and_rew = mu_m + stddev * jax.random.normal(sub, mu_m.shape)

    state_dim =state_agent.shape[-1]
    delta_n = delta_and_rew[..., :state_dim]
    reward_pred = delta_and_rew[..., state_dim:]
    reward_dict = {f"agent_{i}": reward_pred[..., i:i+1] for i in range(reward_pred.shape[-1])}

    delta = std.denorm_delta(delta_n)      # (B,D)
    next_state_agent = state_agent + delta

    # from state get obs and dones
    restored_state = manual_unflatten_state(next_state_agent)
    next_obs = get_obs(restored_state)
    dones_pred = next_state_agent[..., -4:-1]
    dones_bool = dones_pred > 0.0
    dones_dict = {f"agent_{i}": dones_bool[..., i:i+1] for i in range(dones_pred.shape[-1])}
    return next_state_agent, next_obs, reward_dict, dones_dict

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

    mu_real, std_real = real_state.apply_fn({"params": real_state.params}, batch["obs"]["agent_0"])
    mu_opp, std_opp = opp_state.apply_fn({"params":opp_state.params}, batch["obs"][f"agent_{member_idx+1}"])
    std_real = jnp.exp(jnp.clip(std_real, -10.0, 2.0))  # [exp(-10), exp(2)] ≈ [4.5e-5, 7.4]
    std_opp = jnp.exp(jnp.clip(std_opp, -10.0, 2.0))
    kl = kl_normal(mu_real, std_real, mu_opp, std_opp)
    kl = jnp.maximum(jnp.sum(kl, axis=-1), 0.0)
    tv = jnp.sqrt(0.5 * kl)
    tv = jnp.mean(tv)
    return tv # shape:[batch_size]
