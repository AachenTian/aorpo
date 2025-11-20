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

from aorpo.agents.update_q_function import update_q_function
from aorpo.agents.update_policy import update_policy


from aorpo.agents.model_dynamics import (
    init_model,
    train_step as model_train_step,
    Standardizer,
)
wandb.login()
# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="yachen-tian-rwth-aachen-university",
    # Set the wandb project where this run will be logged.
    project="AORPO-dynamics model",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 3e-4,
        "architecture": "AORPO",
        "Environment": "mpe_spread_v3",
        "epochs": 10,
    },
)

# -------------------------------------------------
# 辅助：把一批 dict(jnp arrays) 加入 replay
# -------------------------------------------------
def add_batch_to_replay(replay: ReplayBuffer, batch: dict, cfg:DictConfig) -> ReplayBuffer:
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


    # --- 初始化网络
    rng, k1 = jax.random.split(rng)
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
    model_net, model_state = init_model(km, state_dim, num_agents, act_dim, opp_dim, cfg.model_dynamics)

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

    print("✅ Init done.")

    for epoch in tqdm(range(1, cfg.train.epochs + 1), desc="Training Epochs"):
        print(f"\n===== Epoch {epoch}/{cfg.train.epochs} =====")

        # -------------------------------------------------
        # 1) 真实环境采样 D_env
        # -------------------------------------------------
        policy_fn = make_policy_fn(policy_state)
        opp_fn = make_opp_fn(opponent_states)
        real_opp_fn = make_opp_fn(real_opponent_states)

        rng, kc = jax.random.split(rng)
        batch_env, final_state, rng = collect_real_data(
            policy_fn=policy_fn,
            opp_fn=real_opp_fn,
            obs_dim=obs_dim,
            act_dim=act_dim,
            opp_num=opp_num,
            opp_dim=act_dim,   # 如果每个对手与 ego 维度不同，这里改成对应维度
            key=kc,
            cfg=cfg
        )

        replay_env = add_batch_to_replay(replay_env, batch_env, cfg)
        # -------------------------------------------------
        # 2) 基于 D_env 拟合 Standardizer，并训练 dynamics
        # -------------------------------------------------
        # 用一批 env 数据估计均值方差
        rng, ks = jax.random.split(rng)
        boot = replay_env.sample(ks, batch_size=min(epoch * cfg.train.batch_size, len(replay_env)), opp_num=opp_num)
        std = Standardizer.fit(boot["state"], boot["a_ego"], boot["a_opp"], boot["next_state"])

        # 训练 dynamics
        for i in range(cfg.train.model_updates):
            rng, kb = jax.random.split(rng)
            b = replay_env.sample(kb, batch_size=cfg.train.batch_size, opp_num=opp_num)
            model_state, metrics_m = model_train_step(model_state, b, std)
            wandb.log({
                "dynamics_error": metrics_m["nll"],
                "dynamics_mse": metrics_m["mse"],
                "dynamics_logvar": metrics_m["logvar"],
                "model_step": i
            })
        print(f"Model NLL: {float(metrics_m['nll']):.4f}")
        print(f"[Epoch {epoch}] Model NLL: {float(metrics_m['nll']):.4f}")
    rng, compare_key = jax.random.split(rng, 2)
    state_env, state_dyna = rollout_compare(
        policy_fn=policy_fn,
        opp_fn=real_opp_fn,
        model_state=model_state,
        std=std,
        key=compare_key,
        horizon=15,
        cfg=cfg
    )
    T = state_dyna.shape[0]
    mse_list = []
    l2_list = []
    for t in range(T):
        env_state_t = {
            "p_pos": state_env.p_pos[t],
            "p_vel": state_env.p_vel[t],
            "c": state_env.c[t],
            "done": state_env.done[t],
            "step": state_env.step[t]
        }
        flat_env= manual_flatten_dict(env_state_t)
        flat_dyna = state_dyna[t]
        diff = flat_env - flat_dyna
        mse = jnp.mean(diff**2)
        l2 = jnp.linalg.norm(diff)

        mse_list.append(mse)
        l2_list.append(l2)
        wandb.log({
            "mse" : mse,
            "l2" : l2,
        })
    print("state_env:", state_env)
    print("state_dyna:", state_dyna)
    # print("state_env[0]:",state_env[0])
    wandb.finish()
    print("\n🎉 Training finished.")

if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()

















