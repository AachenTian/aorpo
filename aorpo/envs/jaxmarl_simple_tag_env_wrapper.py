import jax
import jax.numpy as jnp
from jaxmarl import make
from jaxmarl.environments.mpe import MPEVisualizer
import os
import hydra
from omegaconf import DictConfig
from dataclasses import dataclass
from typing import Any, Dict
from jaxmarl.environments.mpe.simple import State
from jaxmarl.environments.mpe.simple_facmac import SimpleFacmacMPE
from aorpo.agents.model_dynamics import facmac_get_obs_batched, FacmacObsConfig

class SimpleFacmacMPEFreezePrey(SimpleFacmacMPE):
    def _prey_policy(self, key, state, aidx):
        return jnp.zeros((self.dim_p,), dtype=jnp.float32)


def make_mpe_env(cfg: DictConfig):
    def frozen_prey_policy(key, state, aidx):
        # dim_p 通常是 2（x,y）
        return jnp.zeros((env.dim_p,), dtype=jnp.float32)
    env = make(cfg.env.ENV_NAME)    #"MPE_simple_v3"
    # env._prey_policy = frozen_prey_policy
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
    # =========================
    # reward shaping starts
    # =========================
    # adversaries: 0 .. num_adversaries-1
    adv_pos = next_state.p_pos[:env.num_adversaries]  # (A,2)
    prey_pos = next_state.p_pos[env.num_adversaries:env.num_agents]  # (G,2)

    diff = adv_pos[:, None, :] - prey_pos[None, :, :]  # (A,G,2)
    dist = jnp.linalg.norm(diff, axis=-1)  # (A,G)

    min_dist_adv = jnp.min(dist, axis=1)  # (A,)
    dist_reward = -jnp.mean(min_dist_adv)  # scalar

    alpha = 1.0  # 很小！
    for a in env.adversaries:
        rewards[a] = rewards[a] + alpha * dist_reward
    # =========================
    # reward shaping ends
    # =========================
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


@dataclass
class State:
    p_pos: jnp.ndarray
    p_vel: jnp.ndarray
    c: jnp.ndarray
    done: jnp.ndarray
    step: jnp.ndarray
    # goal: Any


# ===== 手动构造一个 SimpleTag 的 State =====
def make_manual_state() -> State:
    # SimpleTag: 4 agents + 2 landmarks
    num_agents = 4
    num_landmarks = 2
    num_entities = num_agents + num_landmarks
    dim_c = 2

    # --- p_pos: (6, 2) ---
    p_pos = jnp.array([
        [ 0.46320045,  0.03645253],
        [-0.7549167 , -0.8088202 ],
        [ 0.48811573, -0.6971592 ],
        [ 0.23211247,  0.6941235 ],
        [ 0.22214746,  0.96099734],
        [-0.8680732 ,  0.8911958 ],
    ], dtype=jnp.float32)

    # --- p_vel: (6, 2) ---
    p_vel = jnp.array([
        [-0.17443755,  0.07850721],
        [-0.15623096, -0.04373895],
        [ 0.09050206, -0.1870386 ],
        [-0.00526541, -0.00727525],
        [ 0.0        ,  0.0        ],
        [ 0.0        ,  0.0        ],
    ], dtype=jnp.float32)

    # --- c: (4, 2) ---
    c = jnp.array([
        [0.0, 0.1],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ], dtype=jnp.float32)

    # --- done: (4,) ---
    done = jnp.array([False, False, False, False], dtype=bool)

    # --- step: scalar ---
    step = jnp.array(11, dtype=jnp.int32)

    # --- goal: SimpleTag 不使用 ---
    goal = jnp.zeros((1,))

    return State(
        p_pos=p_pos,
        p_vel=p_vel,
        c=c,
        done=done,
        step=step,
        # goal=goal,
    )

def expand_state_to_batch2(state):
    return State(
        p_pos=jnp.broadcast_to(state.p_pos, (2,) + state.p_pos.shape),
        p_vel=jnp.broadcast_to(state.p_vel, (2,) + state.p_vel.shape),
        c=jnp.broadcast_to(state.c, (2,) + state.c.shape),
        done=jnp.broadcast_to(state.done, (2,) + state.done.shape),
        step=jnp.broadcast_to(state.step, (2,)),
        # goal=jnp.zeros((2,), dtype=jnp.int32),  # dummy goal，vmap 必须
    )


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    max_steps = 25
    key = jax.random.PRNGKey(0)
    env = make_mpe_env(cfg)
    print("env.num_adversaries:", env.num_adversaries)
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
            print("rewards:", rewards)
        # if i == 10:
        #     make_state = make_manual_state()
        #     # make_state = expand_state_to_batch2(make_state)
        #     make_state = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), state)
        #     make_state = expand_state_to_batch2(make_state)
        #     # print("step shape:", make_state.step.shape)
        #     # print("goal shape:", make_state.goal.shape)
        #     cfg_get_obs = FacmacObsConfig(
        #         num_agents=cfg.env.num_agents,
        #         num_adversaries=cfg.env.num_adversaries,
        #         num_landmarks=cfg.train.num_landmark,
        #         view_radius=jnp.full((cfg.env.num_agents,), cfg.env.view_radius),
        #     )
        #     obs_state = facmac_get_obs_batched(make_state, cfg_get_obs)
        #     print("make_state:", make_state)
        #     print("obs_state:",obs_state)
        #     print("state:", state)
        #     print("actions:", actions)
        #     print("rewards:", rewards)
        #     print("dones:", dones)
        #     print("obs:", obs)

    viz = MPEVisualizer(env, state_seq)
    viz.animate(view=True)

if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()