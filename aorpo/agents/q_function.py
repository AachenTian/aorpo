import jax
from flax import linen as nn
import jax.numpy as jnp
import optax
from typing import Sequence
from flax.training.train_state import TrainState
from omegaconf import DictConfig


class QNet(nn.Module):
    hidden_dims: Sequence[int] = (256,256)

    @nn.compact
    def __call__(self, obs, act):
        """Forward pass for Q(s, a)."""

        x = jnp.concatenate([obs, act], axis=-1)

        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim)(x))
        q_value = nn.Dense(1)(x)
        return jnp.squeeze(q_value, -1)


def init_q_function(rng, obs_dim:int, act_dim: int, cfg: DictConfig):

    model = QNet(hidden_dims=tuple(cfg.hidden_dims))
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_act = jnp.zeros((1, act_dim))
    params = model.init(rng, dummy_obs, dummy_act)['params']
    tx = optax.adam(cfg.lr)
    state = TrainState.create(
        apply_fn = model.apply,
        params=params,
        tx=tx
    )
    return model, state