# aorpo/agents/update_policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from aorpo.agents.policy import PolicyNet
from omegaconf import DictConfig


#Update step
@jax.jit
def update_policy(
        policy_state: TrainState,
        q_state: TrainState,
        batch: Dict[str, jnp.ndarray],
        cfg: DictConfig,
        rng: jax.random.KeyArray,
):
    """
    update policy parameters using the objective:
    L_pi = E_s [ alpha * log_pi(a|s) - Q(s,a) ]
    """
    def loss_fn(params):
        actions, log_probs = PolicyNet.sample_action(
            params, policy_state.apply_fn, rng, batch["obs"]
        )

        q_values = q_state.apply_fn({"params": q_state.params}, batch["obs"], actions)

        policy_loss = jnp.mean(cfg.alpha * log_probs - q_values)
        return policy_loss, {"policy_loss": policy_loss}

    grads, metrics = jax.grad(loss_fn, has_aux=True)(policy_state.params)
    updates, opt_state = policy_state.tx.update(grads, policy_state.opt_state)
    new_params = optax.apply_updates(policy_state.params, updates)
    new_state = policy_state.replace(step=policy_state.step + 1,
                                     params=new_params, opt_state=opt_state)
    return new_state, metrics