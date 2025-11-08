# aorpo/agents/update_policy.py
from __future__ import annotations
from typing import Dict, Any

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from omegaconf import DictConfig

from aorpo.agents.policy import PolicyNet


def update_policy(
    policy_state: TrainState,
    q_state: TrainState,
    batch: Dict[str, jnp.ndarray],
    cfg: DictConfig,
    rng: Any,
):
    """
    Update policy parameters using SAC/AORPO objective:
        J(π) = E_s[ α * log π(a|s) - Q(s,a) ]
    """

    def loss_fn(params):
        # --- 采样动作 & 对应 log π(a|s)
        action, log_prob, _ = PolicyNet.sample_action(
            params, policy_state.apply_fn, rng, batch["obs"]
        )

        # --- 计算 Q(s,a)
        q_value = q_state.apply_fn({"params": q_state.params}, batch["obs"], action)
        q_value = jax.lax.stop_gradient(q_value)  # ❗ policy 不能更新 Q 参数

        # --- 策略损失 Eq.(4)
        policy_loss = jnp.mean(cfg.alpha * log_prob - q_value)

        return policy_loss, {"policy_loss": policy_loss}

    grads, metrics = jax.grad(loss_fn, has_aux=True)(policy_state.params)

    # --- 更新参数
    updates, opt_state = policy_state.tx.update(grads, policy_state.opt_state, policy_state.params)
    new_params = optax.apply_updates(policy_state.params, updates)

    new_state = policy_state.replace(
        step=policy_state.step + 1,
        params=new_params,
        opt_state=opt_state,
    )

    return new_state, metrics


update_policy = jax.jit(update_policy, static_argnums=(3,))
