import jax
import jax.numpy as jnp
from jaxmarl import make
import os
import hydra
from omegaconf import DictConfig

def make_mpe_env(cfg: DictConfig):
    env = make(cfg.env.ENV_NAME, action_type="Continuous")    #"MPE_simple_v3"
    return env

def env_reset(env, key):
    key = jax.random.PRNGKey(40)
    obs, state = env.reset(key)
    return state, obs, key

def env_step(env, state, a_ego, a_opps, key):
    a_ego = a_ego.reshape(1,-1)
    a_opps = a_opps.reshape(2, -1)
    actions = jnp.concatenate([a_ego, a_opps], axis=0)
    actions = {agent: actions[i] for i,agent in enumerate(env.agents)}
    obs, next_state, rewards, dones, infos = env.step(key, state, actions)
    return next_state, obs, rewards, dones, key


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    rng = jax.random.PRNGKey(cfg.seed)

    env = make_mpe_env(cfg)
    print(f"✅ Env initialized: {cfg.env.ENV_NAME}")
    rng, subrng = jax.random.split(rng,2)
    eval_rng = jax.random.PRNGKey(0)
    state, obs, key = env_reset(env, eval_rng)
    print("obs keys:", obs.keys())
    print("state:", state)

    # a_ego = jnp.ones((cfg.env.act_dim,))
    # a_opps = jnp.zeros((cfg.train.num_opponents * cfg.env.act_dim))
    # a_ego = jnp.array([-0.9009457, -0.19701889, 0.40034992, -0.6112185,  0.0424891 ]) ##(-0.6537076, -0.59736881)
    a_ego = jnp.array([-0.82266784, -0.18221606, 0.31996903, -1.6457553, 0.05335886]) ##(-0.69911416, -0.50218509)

    a_opps = jnp.array([
        -0.18004934, 0.10922899, 0.38247943, -0.97298497, -0.8712533,
        -0.2582537, 0.2940862, -0.98897564, -0.9850584, 0.343176
    ])

    for i in range(3):
        if i == 0:
            a_ego = a_ego
            a_opps = a_opps
        else:
            a_ego = jnp.array([-0.82266784, -10.18221606, 0.31996903, -10.6457553, 0.05335886])
            a_opps = jnp.array([
                -0.18004934, 0.10922899, 0.38247943, -0.97298497, -0.8712533,
                -0.2582537, 0.2940862, -0.98897564, -0.9850584, 0.343176
            ])
        state, obs, rewards, dones, key = env_step(env, state, a_ego, a_opps, key)
        # print("landmarks:", next_state.p_pos[3:])
        # print("episode done flags:", dones)
        # print("next_state:", next_state.step)
        # print("next_state:",next_state)
        # print("obs:", obs)
        print("rewards:", rewards)
        # print("dones:", dones)

if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
