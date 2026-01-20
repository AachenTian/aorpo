import jax
import jax.numpy as jnp
from jaxmarl import make
from jaxmarl.environments.mpe import MPEVisualizer
import os
import hydra
from omegaconf import DictConfig
from jaxmarl.environments.mpe.simple import State

def make_mpe_env(cfg: DictConfig):
    env = make(cfg.env.ENV_NAME)    #"MPE_simple_v3"
    return env

def env_reset(env, key):
    key_reset = jax.random.PRNGKey(40)
    obs, state = env.reset(key_reset)
    obs.pop("agent_0")
    return state, obs, key

def env_step(env, state, a_ego, a_opps, key):
    a_ego = a_ego.reshape(1,-1)
    a_opps = a_opps.reshape(2, -1)
    actions = jnp.concatenate([a_ego, a_opps], axis=0)
    actions = {agent: actions[i] for i,agent in enumerate(env.agents)}
    obs, next_state, rewards, dones, infos = env.step(key, state, actions)
    obs.pop("agent_0")
    rewards.pop("agent_0")
    dones.pop("agent_0")
    return next_state, obs, rewards, dones, key

def get_adversary_obs_batched(state, env):
    """
    state: batched State pytree
    env: environment (Python object)
    return: Dict[str, Array] (batched obs, adversaries only)
    """

    # 1️⃣ batched get_obs（无 Python loop）
    obs = jax.vmap(lambda s: env.get_obs(s))(state)

    # 2️⃣ 只保留 adversary（无 pop）
    obs = {
        k: v
        for k, v in obs.items()
        if k.startswith("adversary_")
    }

    return obs


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    max_steps = 25
    key = jax.random.PRNGKey(0)
    env = make_mpe_env(cfg)
    state, obs, key = env_reset(env, key)
    # Sample random actions
    key, key_a = jax.random.split(key, 2)
    key_a = jax.random.split(key_a, env.num_agents)
    actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}

    state_seq = []
    for i in range(max_steps):
        state_seq.append(state)
        key, key_s, key_a = jax.random.split(key, 3)
        key_a = jax.random.split(key_a, env.num_agents)
        actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}
        a_ego = actions['adversary_0']
        a_opp_1 = actions['adversary_1']
        a_opp_2 = actions['adversary_2']
        a_opps = (jnp.concatenate([a_opp_1, a_opp_2], axis=0)).reshape(2, -1)
        state, obs, rewards, dones, key = env_step(env, state, a_ego, a_opps, key)
        if i == 10:
            print("state:", state)
            print("actions:", actions)
            print("rewards:", rewards)
            print("dones:", dones)
            print("obs:", obs)

    viz = MPEVisualizer(env, state_seq)
    viz.animate(view=True)

if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()