from flax import linen as nn
import jax.numpy as jnp
from typing import Sequence

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