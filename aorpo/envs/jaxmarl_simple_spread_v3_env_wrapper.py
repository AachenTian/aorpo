import jax
import jax.numpy as jnp
from jaxmarl import make
import os
import hydra
from omegaconf import DictConfig

def make_mpe_env(cfg: DictConfig):
    env = make(cfg.env.ENV_NAME)    #"MPE_simple_v3"
    return env

def env_reset(env, key):
    obs, state = env.reset(key)
    return state, obs, key

def env_step(env, state, a_ego, a_opps, key):
    actions = jnp.concatenate([a_ego, a_opps], axis=0)
    actions = {agent: actions[i] for i,agent in enumerate(env.agents)}
    obs, next_state, rewards, dones, infos = env.step(key, state, actions)
    return next_state, obs, rewards, dones, key


# @hydra.main(config_path="../configs", config_name="train", version_base=None)
# def main(cfg: DictConfig):
#     rng = jax.random.PRNGKey(cfg.seed)
#
#     env = make_mpe_env(cfg)
#     print(f"✅ Env initialized: {cfg.env.ENV_NAME}")
#
#     state, obs, key = env_reset(env, rng)
#     print("obs keys:", obs.keys())
#     print("state:", state)
#
#     a_ego = jnp.zeros((cfg.env.act_dim,))
#     a_opps = jnp.zeros((cfg.train.num_opponents, cfg.env.act_dim))
#     next_state, obs, rewards, dones, key = env_step(env, state, a_ego, a_opps, key)
#     print("next_state:",next_state)
#     print("obs:", obs)
#     print("rewards:", rewards)
#     print("dones:", dones)
#
# if __name__ == "__main__":
#     os.environ.setdefault("HYDRA_FULL_ERROR", "1")
#     main()
