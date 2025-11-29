# aorpo/agents/update_opponents_model.py
from __future__ import annotations
from typing import Dict, Any, List

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from aorpo.agents.policy import PolicyNet

def _bc_loss(params, apply_fn, obs, act_target):
    mu, log_std = apply_fn({"params": params}, obs)
    pred = jnp.tanh(mu)
    loss = jnp.mean((pred - act_target)**2)
    return loss


def update_opponent_model(
    opponent_states: List[TrainState],
    batch: Dict,
):
    opp_num = len(opponent_states)
    a_opp_all = batch["a_opp"]
    act_dim = a_opp_all.shape[-1] // opp_num

    # 将所有 params + opt_state 组合成 pytree 列表
    params_all = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[s.params for s in opponent_states])
    opt_all    = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[s.opt_state for s in opponent_states])

    # 每个 opponent 的 obs 和 target
    obs_list  = [batch["obs"][f"agent_{j+1}"] for j in range(opp_num)]
    obs_all   = jnp.stack(obs_list)  # (opp_num, B, obs_dim)

    target_list = [
        a_opp_all[:, j*act_dim:(j+1)*act_dim] for j in range(opp_num)
    ]
    target_all = jnp.stack(target_list)  # (opp_num, B, act_dim)

    apply_fn = opponent_states[0].apply_fn  # 所有 opponent 共享相同 architecture

    # --------------------------------------------------
    # vmap over opponents → 一次性计算所有 grad
    # --------------------------------------------------
    def loss_fn(params, obs, target):
        return _bc_loss(params, apply_fn, obs, target)

    loss_grad_fn = jax.vmap(jax.value_and_grad(loss_fn), in_axes=(0,0,0))
    losses, grads = loss_grad_fn(params_all, obs_all, target_all)

    # --------------------------------------------------
    # 更新：vmap optax.update
    # --------------------------------------------------
    updates, new_opt = jax.vmap(
        opponent_states[0].tx.update, in_axes=(0,0,0)
    )(grads, opt_all, params_all)

    new_params_all = jax.vmap(optax.apply_updates)(params_all, updates)

    # --------------------------------------------------
    # 解包：还原成 TrainState 列表
    # --------------------------------------------------
    new_states = []
    for j, s in enumerate(opponent_states):
        new_state = s.replace(
            params=jax.tree_util.tree_map(lambda x: x[j], new_params_all),
            opt_state=jax.tree_util.tree_map(lambda x: x[j], new_opt),
            step=s.step + 1,
        )
        new_states.append(new_state)

    metrics = {
        "opp_loss_mean": jnp.mean(losses),
        "opp_loss_each": losses,
    }

    return new_states, metrics


update_opponent_fast = jax.jit(update_opponent_model, static_argnums=0)