# aorpo/train_utils/update_q_function.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from aorpo.agents.q_function import QNet
from aorpo.agents.policy import PolicyNet


# --------------------------------
# Config
# --------------------------------
@dataclass
class QFunctionConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    alpha: float = 0.2  # entropy temperature


# --------------------------------
# Training Step
# --------------------------------
@jax.jit
def update_q_function(
    q_state: TrainState,
    target_q_state: TrainState,
    policy_state: TrainState,
    batch: Dict[str, jnp.ndarray],
    cfg: QFunctionConfig,
):
    """
    Update Q-function parameters.
    batch: dict with keys {obs, act, rew, next_obs, done}
    """

    def loss_fn(params):
        # Current Q(s,a)
        q_pred = q_state.apply_fn({'params': params}, batch["obs"], batch["act"])

        # Next action & log prob from policy
        rng = jax.random.PRNGKey(0)
        next_action, _ = PolicyNet.sample_action(policy_state.params, policy_state.apply_fn, rng, batch["next_obs"])
        log_prob = jnp.sum(-0.5 * next_action ** 2, axis=-1)  # 简化估计，可替换成真实 log π

        # Target Q
        q_target_next = target_q_state.apply_fn({'params': target_q_state.params},
                                                batch["next_obs"], next_action)
        target = batch["rew"] + cfg.gamma * (1.0 - batch["done"]) * (q_target_next - cfg.alpha * log_prob)

        # MSE loss
        loss = jnp.mean((q_pred - target) ** 2)
        return loss, {'q_loss': loss}

    grads, metrics = jax.grad(loss_fn, has_aux=True)(q_state.params)
    updates, opt_state = q_state.tx.update(grads, q_state.opt_state)
    new_params = optax.apply_updates(q_state.params, updates)
    new_state = q_state.replace(step=q_state.step + 1, params=new_params, opt_state=opt_state)
    return new_state, metrics
