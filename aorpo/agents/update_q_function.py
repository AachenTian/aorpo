# aorpo/train_utils/update_q_function.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from omegaconf import DictConfig

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

def update_q_function(
        q1_state: TrainState,
        q2_state: TrainState,
        target_q1_state: TrainState,
        target_q2_state: TrainState,
        policy_state: TrainState,
        opponent_policies: List[Dict[str, Any]],
        batch: Dict[str, jnp.ndarray],
        cfg: DictConfig,
        rng: Any,
):
    """
    Update 2 Q-function parameters.
    AORPO Eq.(5):  J(Q) = E[(Q(s,a) - (r + γ(Q' - α log π)))^2]
    batch: dict with keys {obs, act, rew, next_obs, done}
    """
    def loss_fn(params1, params2, rng):
        ego_act = batch["a_ego"]
        opp_act = batch["a_opp"]
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        state = batch["state"]
        next_state = batch["next_state"]
        joint_act = jnp.concatenate([ego_act, opp_act],axis=-1)
        q1_pred = q1_state.apply_fn({"params":params1}, state, joint_act)
        q2_pred = q2_state.apply_fn({"params":params2}, state, joint_act)

        rng, subkey = jax.random.split(rng)
        a_i, log_prob, key = PolicyNet.sample_action(
            policy_state.params, policy_state.apply_fn, rng, next_obs['agent_0']
        )
        a_js = []
        for j, opp in enumerate(opponent_policies):
            # 使用 learned opponent policy
            rng, subkey = jax.random.split(rng)
            a_j, _, _ = PolicyNet.sample_action(
                opp["state"].params,
                opp["state"].apply_fn,
                subkey,
                next_obs[f"agent_{j+1}"],
            )
            a_js.append(a_j)

        next_action = jnp.concatenate([a_i] + a_js, axis=-1)
        q1_target_next = target_q1_state.apply_fn(
            {"params":target_q1_state.params}, next_state, next_action
        )
        q2_target_next = target_q2_state.apply_fn(
            {"params": target_q2_state.params}, next_state, next_action
        )
        q_target_next = jnp.minimum(q1_target_next, q2_target_next)

        rewards = jnp.stack(
            [batch["rew"][f"agent_{i}"] for i in range(cfg.agent_num)], axis=-1
        )  # (B, num_agents)
        reward = jnp.sum(rewards, axis=-1)  # (B,)
        target_q = reward.squeeze(-1) + cfg.gamma * (
            1.0 - batch["dones"]['agent_0'].squeeze(-1)
        ) * (q_target_next - cfg.alpha * log_prob)

        #Q1, Q2 MSE
        loss_q1 = jnp.mean((q1_pred - target_q) **2)
        loss_q2 = jnp.mean((q2_pred - target_q) **2)
        total_loss = 0.5 * (loss_q1 + loss_q2)
        q1_pred_mean = jnp.mean(q1_pred)
        q2_pred_mean = jnp.mean(q2_pred)

        return total_loss, {"q1_loss": loss_q1, "q2_loss": loss_q2, "q1_pred": q1_pred_mean, "q2_pred": q2_pred_mean}

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0,1), has_aux=True)
    (loss, metrics), (grads1, grads2) = grad_fn(q1_state.params, q2_state.params, rng)

    updates1, opt_state1 = q1_state.tx.update(grads1, q1_state.opt_state, q1_state.params)
    updates2, opt_state2 = q2_state.tx.update(grads2, q2_state.opt_state, q2_state.params)

    new_params1 = optax.apply_updates(q1_state.params, updates1)
    new_params2 = optax.apply_updates(q2_state.params, updates2)

    new_q1_state = q1_state.replace(
        step=q1_state.step + 1, params=new_params1, opt_state=opt_state1
    )
    new_q2_state = q2_state.replace(
        step=q2_state.step + 1, params=new_params2, opt_state=opt_state2
    )

    return new_q1_state, new_q2_state, metrics, rng

update_q_function = jax.jit(update_q_function, static_argnums=(7,))
