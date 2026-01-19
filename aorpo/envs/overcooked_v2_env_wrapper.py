import time
import jax
import jax.numpy as jnp
from jaxmarl import make
import os
import hydra
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
from omegaconf import DictConfig

def make_overcooked_v2_env(cfg: DictConfig):
    env = make(
        cfg.env.ENV_NAME,
        layout = cfg.env.LAYOUT,
        max_steps = cfg.env.MAX_STEPS,
        # view_radius = cfg.env.VIEW_RADIUS,
    )    #"overcooked_v2"
    return env

def env_reset(env, key):
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset)
    return state, obs, key

def env_step(env, state, a_ego, a_opp, key):
    # 1️⃣ 确保 action 是 Python int
    a_ego = int(jnp.asarray(a_ego).item())
    a_opp = int(jnp.asarray(a_opp).item())

    # 2️⃣ 构造 JaxMARL 期望的 action dict
    actions = {
        agent: act
        for agent, act in zip(env.agents, [a_ego, a_opp])
    }

    # 3️⃣ step 环境
    key, subkey = jax.random.split(key)
    obs, next_state, rewards, dones, infos = env.step(
        subkey, state, actions
    )

    done = dones["__all__"]

    return next_state, obs, rewards, done, key


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    state_log = []
    rng = jax.random.PRNGKey(cfg.seed)
    env = make_overcooked_v2_env(cfg)
    print(f"✅ Env initialized: {cfg.env.ENV_NAME}")

    state, obs, rng = env_reset(env, rng)
    print(type(state))
    print(state)
    print(jax.tree_util.tree_structure(state))
    state_log.append(state)
    print("init_state:", state)
    a_ego = [2]
    a_opp = [3]
    for i in range(cfg.env.MAX_STEPS):
        rng, sub_rng_1, sub_rng_2 = jax.random.split(rng, 3)
        a_ego = env.action_space(env.agents[0]).sample(sub_rng_1)
        a_opp = env.action_space(env.agents[1]).sample(sub_rng_2)
        next_state, obs, reward, done, rng = env_step(env, state, a_ego, a_opp, rng)
        if i == 1:
            print("obs:", obs)
        state_log.append(next_state)
        state = next_state

    viz = OvercookedV2Visualizer()
    state_seq = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        *state_log
    )
    viz.animate(
        state_seq=state_seq,
        filename="overcooked_v2_env.gif",
        agent_view_size=cfg.env.agent_view_size,
    )
    # for s in state_log:
    #     viz.render(
    #         s,
    #         agent_view_size=cfg.env.agent_view_size,
    #     )
    #     time.sleep(0.2)


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()