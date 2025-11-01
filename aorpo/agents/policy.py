from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence

class PolicyNet(nn.Module):
    action_dim:int                              #action dimension
    hidden_dims:Sequence[int] = (256,256)       #hidden dimension
    min_logvar: float = -5.0
    max_logvar: float = 2.0


    @nn.compact
    def __call__(self,obs):
        x = obs
        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim)(x))
        mu = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.min_logvar, self.max_logvar)
        return mu, log_std

def sample_action(params, apply_fn, rng, obs):
    """Sample an action from the policy (Gaussian & tanh)."""
    mu, log_std = apply_fn(params, obs)
    std = jnp.exp(log_std)
    key, subkey = jax.random.split(rng)
    normal_sample = mu + std * jax.random.normal(subkey, mu.shape)
    action = jnp.tanh(normal_sample)
    return action, key