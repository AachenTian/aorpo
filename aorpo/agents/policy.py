from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Any
from flax.training.train_state import TrainState
import optax
from omegaconf import DictConfig


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

    @staticmethod
    def sample_action(params, apply_fn, rng, obs):
        """Sample an action from the policy (Gaussian & tanh)."""
        mu, log_std = apply_fn(params, obs)
        std = jnp.exp(log_std)
        key, subkey = jax.random.split(rng)
        normal_sample = mu + std * jax.random.normal(subkey, mu.shape)
        action = jnp.tanh(normal_sample)
        return action, key

def init_policy_model(rng: Any,
                      obs_dim: int,
                      act_dim: int,
                      cfg: DictConfig
                      ):
    model = PolicyNet(
        action_dim=act_dim,
        hidden_dims=tuple(cfg.hidden_dims),
        min_logvar=cfg.min_logvar,
        max_logvar=cfg.max_logvar
    )

    dummy_obs = jnp.zeros((1, obs_dim))
    params = model.init(rng, dummy_obs)
    tx = optax.adam(cfg.lr)

    state = TrainState.create(
        apply_fn = model.apply,
        params = params,
        tx = tx
    )
    return model, state