# aorpo/train.py
from __future__ import annotations
import os
import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb, random
import copy

# ===== 你项目里的模块 =====
from aorpo.utils.replay import ReplayBuffer, manual_flatten_dict
from aorpo.rollout.collect import collect_real_data, episode_reward, rollout_compare
from aorpo.rollout.rollout import rollout_model, compute_rollout_lengths


from aorpo.agents.policy import init_policy_model, PolicyNet
from aorpo.agents.q_function import init_q_function

from aorpo.agents.update_q_function import update_q_function, evaluate_fixed_q_loss
from aorpo.agents.update_policy import update_policy, update_opponent_policy
from aorpo.agents.update_opponents_model import update_opponent_model
from aorpo.visualiztion.make_animation import animate_episode

from aorpo.agents.model_dynamics import (
    init_model,
    train_transition_step,
    train_reward_step,
    StandardizerRS,
    Standardizer
)


wandb.login()
# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="yachen-tian-rwth-aachen-university",
    # Set the wandb project where this run will be logged.
    project="AORPO-dynamics model",
    # mode="offline",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 3e-4,
        "architecture": "AORPO",
        "Environment": "mpe_spread_v3",
        "epochs": 10,
    },
)

# -------------------------------------------------
# 辅助：软更新 target Q
# -------------------------------------------------
def soft_update(target_state, source_state, tau: float):
    new_params = jax.tree_util.tree_map(
        lambda t, s: (1.0 - tau) * t + tau * s, target_state.params, source_state.params
    )
    return target_state.replace(params=new_params)

# -------------------------------------------------
# 辅助：把一批 dict(jnp arrays) 加入 replay
# -------------------------------------------------
def add_batch_env_to_replay(replay: ReplayBuffer, batch: dict, cfg:DictConfig) -> ReplayBuffer:
    return replay.add_batch(batch, cfg)


# -------------------------------------------------
# JAX 风格 policy / opponent 的“可调用函数”（供 collect 使用）
#   collect_real_data(policy_fn, opp_fn, ...) 期望：
#   - policy_fn(s, key) -> ego 动作 a_i
#   - opp_fn(s, key)    -> 拼好的对手动作向量 a_-i
# -------------------------------------------------
def make_policy_fn(policy_state):
    def policy_fn(obs, key):
        act, _, new_key = PolicyNet.sample_action(
            policy_state.params,
            policy_state.apply_fn,
            key,
            obs["agent_0"],
        )
        return act, new_key
    return policy_fn



def make_opp_fn(opponent_states):
    def opp_fn(obs, key):
        acts = []
        key, sub = jax.random.split(key)
        for i, state in enumerate(opponent_states):
            a_j, _, sub = PolicyNet.sample_action(
                state.params,
                state.apply_fn,
                sub,
                obs[f"agent_{i+1}"]
            )
            acts.append(a_j)
        return jnp.concatenate(acts, -1), sub
    return opp_fn

# -------------------------------------------------
# 主训练流程（Hydra）
# -------------------------------------------------
@hydra.main(config_path="aorpo/configs", config_name="train", version_base=None)
def main(cfg: DictConfig):

    print("\n===== Config =====")
    print(OmegaConf.to_yaml(cfg))

    rng = jax.random.PRNGKey(cfg.seed)

    # 维度
    state_dim = cfg.env.state_dim
    num_opponents = cfg.train.num_opponents
    num_agents = num_opponents + 1
    obs_dim = cfg.env.obs_dim
    act_dim = cfg.env.act_dim
    opp_num = getattr(cfg.train, "num_opponents", 0)
    opp_dim = act_dim * opp_num  # 简单假设每个对手动作维度与 ego 相同



    # --- Replay Buffers
    replay_env = ReplayBuffer.create(cfg.replay.capacity, obs_dim, act_dim, opp_num, state_dim)
    replay_model = ReplayBuffer.create(cfg.replay.capacity, obs_dim, act_dim, opp_num, state_dim)
    replay_env_fix = ReplayBuffer.create(cfg.replay.capacity, obs_dim, act_dim, opp_num, state_dim)


    # --- 初始化网络
    rng, k1 = jax.random.split(rng)
    rng, k11 = jax.random.split(rng)
    _, policy_state = init_policy_model(k1, obs_dim, act_dim, cfg.policy, "agent_0")

    rng, kq1 = jax.random.split(rng)
    q1_net, q1_state = init_q_function(kq1, state_dim, act_dim, cfg.q_function)
    rng, kq2 = jax.random.split(rng)
    q2_net, q2_state = init_q_function(kq2, state_dim, act_dim, cfg.q_function)

    # target Q
    _, target_q1_state = init_q_function(kq1, state_dim, act_dim, cfg.q_function)
    _, target_q2_state = init_q_function(kq2, state_dim, act_dim, cfg.q_function)

    # dynamics model
    rng, km = jax.random.split(rng)
    transition_model, reward_model, transition_state, reward_state = init_model(
        km, num_agents, act_dim, opp_dim, cfg
    )

    # opponent
    opponent_states = []
    for i in range(opp_num):
        rng, ko = jax.random.split(rng)
        j = i+1
        _, opp_state = init_policy_model(ko, obs_dim, act_dim, cfg.policy, f"agent_{j}")
        opponent_states.append(opp_state)

    # real opponent
    real_opponent_states = []
    for i in range(opp_num):
        rng, ko = jax.random.split(rng)
        _, real_opp_state = init_policy_model(ko, obs_dim, act_dim, cfg.policy, f"agent_{i+1}")
        real_opponent_states.append(real_opp_state)

    # init std
    core_state_dim = cfg.train.core_state_dim
    std = StandardizerRS.create(
        core_state_dim=core_state_dim,
        act_dim_ego=act_dim,
        act_dim_opp=act_dim * opp_num,
    )

    # init animation parameters
    episode_reward_history = []
    epochs_to_render = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 130, 160, 200]

    # prepare fixed_batch
    rng, k_fix = jax.random.split(rng)
    policy_fn = make_policy_fn(policy_state)
    real_opp_fn = make_opp_fn(real_opponent_states)
    batch_env_fix, final_state_fix, rng = collect_real_data(
        policy_fn=policy_fn,
        opp_fn=real_opp_fn,
        obs_dim=obs_dim,
        act_dim=act_dim,
        opp_num=opp_num,
        opp_dim=act_dim,  # 如果每个对手与 ego 维度不同，这里改成对应维度
        key=k_fix,
        cfg=cfg
    )
    rng, k_fix_sample = jax.random.split(rng)
    replay_env_fix = add_batch_env_to_replay(replay_env_fix, batch_env_fix, cfg)
    batch_env_fix_sample = replay_env_fix.sample(k_fix_sample, batch_size=cfg.train.batch_size, opp_num=opp_num)

    print(batch_env_fix)

if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()

















