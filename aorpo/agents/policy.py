from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Any
from flax.training.train_state import TrainState
from flax import struct
import optax
from omegaconf import DictConfig


class TrainStateid(TrainState):
    agent_id: str = struct.field(pytree_node=False)  #

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
        mu, log_std = apply_fn({"params": params}, obs)
        std = jnp.exp(log_std)
        rng, subkey = jax.random.split(rng)
        normal_sample = mu + std * jax.random.normal(subkey, mu.shape)
        action = jnp.tanh(normal_sample)
        log_prob = -0.5 * (((normal_sample -mu) ** 2 / (std + 1e-6) ** 2) + 2 * log_std + jnp.log(2 * jnp.pi))
        log_prob = jnp.sum(log_prob, axis=-1)
        log_prob -=jnp.sum(jnp.log(1- action ** 2 + 1e-6), axis=-1)

        return action, log_prob, rng

def init_policy_model(rng: Any,
                      obs_dim: int,
                      act_dim: int,
                      cfg: DictConfig,
                      agent_id: str
                      ):
    model = PolicyNet(
        action_dim=act_dim,
        hidden_dims=tuple(cfg.hidden_dims),
        min_logvar=cfg.min_logvar,
        max_logvar=cfg.max_logvar
    )

    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = model.init(init_rng, dummy_obs)["params"]
    tx = optax.adam(cfg.lr)

    state = TrainStateid.create(
        apply_fn = model.apply,
        params = params,
        tx = tx,
        agent_id = agent_id
    )
    return model, state