# aorpo/agents/update_policy.py
from __future__ import annotations
from typing import Dict, Any

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from aorpo.agents.policy import PolicyNet
from omegaconf import DictConfig


#Update step
def update_policy(
        policy_state: TrainState,
        q_state: TrainState,
        batch: Dict[str, jnp.ndarray],
        cfg: DictConfig,
        rng: Any,
):
    """
    update policy parameters using the objective:
    L_pi = E_s [ alpha * log_pi(a|s) - Q(s,a) ]
    """
    def loss_fn(params):
        actions, log_probs = policy_state.apply_fn(
            params,
            batch["obs"],
        )

        q_values = q_state.apply_fn({"params": q_state.params}, batch["obs"], actions)
        q_values = jnp.mean(q_values, axis=0)  # average in ensemble dimension E

        print("Log_probs mean:", jnp.mean(log_probs))
        print("Q_values mean:", jnp.mean(q_values))

        policy_loss = jnp.mean(cfg.alpha * log_probs - q_values)
        return policy_loss, {"policy_loss": policy_loss}

    grads, metrics = jax.grad(loss_fn, has_aux=True)(policy_state.params)
    print("Policy grad norm:", jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), grads))
    updates, opt_state = policy_state.tx.update(grads, policy_state.opt_state)
    new_params = optax.apply_updates(policy_state.params, updates)
    new_state = policy_state.replace(step=policy_state.step + 1,
                                     params=new_params, opt_state=opt_state)
    return new_state, metrics
update_policy = jax.jit(update_policy, static_argnums=(3,))