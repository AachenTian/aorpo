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

    obs_list  = [batch["obs"][f"adversary_{j+1}"] for j in range(opp_num)]
    obs_all   = jnp.stack(obs_list )   # (opp_num, B, obs_dim)

    target_all = jnp.stack([
        a_opp_all[:, j*act_dim:(j+1)*act_dim]
        for j in range(opp_num)
    ])                                   # (opp_num, B, act_dim)

    apply_fn = opponent_states[0].apply_fn

    def loss_fn(params, obs, target):
        return _bc_loss(params, apply_fn, obs, target)

    new_states = []
    losses = []

    for j, s in enumerate(opponent_states):
        loss, grads = jax.value_and_grad(loss_fn)(
            s.params, obs_all[j], target_all[j]
        )

        updates, new_opt = s.tx.update(grads, s.opt_state, s.params)
        new_params = optax.apply_updates(s.params, updates)

        new_states.append(
            s.replace(
                params=new_params,
                opt_state=new_opt,
                step=s.step + 1,
            )
        )
        losses.append(loss)

    return new_states, {
        "opp_loss_mean": jnp.mean(jnp.stack(losses)),
        "opp_loss_each": jnp.stack(losses),
    }
