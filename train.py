# aorpo/train.py
from __future__ import annotations
import os
import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# ===== 你项目里的模块 =====
from aorpo.utils.replay import ReplayBuffer
from aorpo.rollout.collect import collect_real_data
from aorpo.rollout.rollout import rollout_model, compute_rollout_lengths

from aorpo.agents.policy import init_policy_model, PolicyNet
from aorpo.agents.q_function import init_q_function   # 你在 q_function.py 里提供的初始化函数
from aorpo.agents.update_q_function import update_q_function
from aorpo.agents.update_policy import update_policy

from aorpo.agents.model_dynamics import (
    init_model,
    train_step as model_train_step,
    Standardizer,
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
def add_batch_to_replay(replay: ReplayBuffer, batch: dict) -> ReplayBuffer:
    return replay.add_batch(batch)


# -------------------------------------------------
# JAX 风格 policy / opponent 的“可调用函数”（供 collect 使用）
#   collect_real_data(policy_fn, opp_fn, ...) 期望：
#   - policy_fn(s, key) -> ego 动作 a_i
#   - opp_fn(s, key)    -> 拼好的对手动作向量 a_-i
# -------------------------------------------------
def make_policy_fn(policy_state):
    def policy_fn(obs, key):
        act, _, key = PolicyNet.sample_action(
            policy_state.params, policy_state.apply_fn, key, obs
        )
        return act
    return policy_fn


def make_opp_fn(opponent_states):
    """简单实现：把每个 opponent 的动作拼接起来。
       现在如果你还没真正训练 opponent，就先给 0 动作占位。"""
    if not opponent_states:
        def zeros_opp(obs, key):
            # 让对手维度为 0 时返回空张量（collect 会处理）
            return jnp.zeros((obs.shape[0], 0), dtype=obs.dtype)
        return zeros_opp

    def opp_fn(obs, key):
        acts = []
        k = key
        for state in opponent_states:
            a_j, _, k = PolicyNet.sample_action(state.params, state.apply_fn, k, obs)
            acts.append(a_j)
        return jnp.concatenate(acts, axis=-1)
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
    obs_dim = cfg.env.obs_dim
    act_dim = cfg.env.act_dim
    opp_num = getattr(cfg.train, "num_opponents", 0)
    opp_dim = act_dim * opp_num  # 简单假设每个对手动作维度与 ego 相同

    # --- Replay Buffers
    replay_env = ReplayBuffer.create(cfg.replay.capacity, obs_dim, act_dim, opp_dim)
    replay_model = ReplayBuffer.create(cfg.replay.capacity, obs_dim, act_dim, opp_dim)

    # --- 初始化网络
    rng, k1 = jax.random.split(rng)
    _, policy_state = init_policy_model(k1, obs_dim, act_dim, cfg.policy)

    rng, kq1 = jax.random.split(rng)
    q1_net, q1_state = init_q_function(kq1, obs_dim, act_dim, cfg.q_function)
    rng, kq2 = jax.random.split(rng)
    q2_net, q2_state = init_q_function(kq2, obs_dim, act_dim, cfg.q_function)

    # target Q
    _, target_q1_state = init_q_function(kq1, obs_dim, act_dim, cfg.q_function)
    _, target_q2_state = init_q_function(kq2, obs_dim, act_dim, cfg.q_function)

    # dynamics model
    rng, km = jax.random.split(rng)
    model_net, model_state = init_model(km, obs_dim, act_dim, opp_dim, cfg.model_dynamics)

    # opponent（可选，先不训练也可用）
    opponent_states = []
    for _ in range(opp_num):
        rng, ko = jax.random.split(rng)
        _, opp_state = init_policy_model(ko, obs_dim, act_dim, cfg.policy)
        opponent_states.append(opp_state)

    print("✅ Init done.")

    # ============= 训练循环 =============
    for epoch in tqdm(range(1, cfg.train.epochs + 1), desc="Training Epochs"):
        print(f"\n===== Epoch {epoch}/{cfg.train.epochs} =====")

        # -------------------------------------------------
        # 1) 真实环境采样 D_env
        # -------------------------------------------------
        policy_fn = make_policy_fn(policy_state)
        opp_fn = make_opp_fn(opponent_states)

        rng, kc = jax.random.split(rng)
        batch_env, final_state, rng = collect_real_data(
            policy_fn=policy_fn,
            opp_fn=opp_fn,
            obs_dim=obs_dim,
            act_dim=act_dim,
            opp_num=opp_num,
            opp_dim=act_dim,   # 如果每个对手与 ego 维度不同，这里改成对应维度
            steps=cfg.collect.steps_per_epoch,
            key=kc,
        )
        # print("Replay obs shape:", replay_env.obs.shape)
        # print("Replay ego act shape:", replay_env.a_ego.shape)
        # print("Replay opponent act shape:", replay_env.a_opp.shape)
        # print("Replay next_obs shape:", replay_env.next_obs.shape)
        # print("Replay rew shape:", replay_env.rew.shape)
        # print("Replay done shape:", replay_env.done.shape)
        # print("Replay current size:", replay_env.size)
        # print("batch obs shape:", batch_env["obs"].shape)
        # print("batch act shape:", batch_env["a_ego"].shape)
        # print("batch next_obs shape:", batch_env["next_obs"].shape)
        # print("batch rew shape:", batch_env["rew"].shape)
        # print("batch done shape:", batch_env.done.shape)
        # print("batch current size:", batch_env.size)
        replay_env = add_batch_to_replay(replay_env, batch_env)
        # print("Replay_env",replay_env)

        # -------------------------------------------------
        # 2) 基于 D_env 拟合 Standardizer，并训练 dynamics
        # -------------------------------------------------
        # 用一批 env 数据估计均值方差
        rng, ks = jax.random.split(rng)
        boot = replay_env.sample(ks, batch_size=min(cfg.train.batch_size, len(replay_env)))
        std = Standardizer.fit(boot["obs"], boot["a_ego"], boot["a_opp"], boot["next_obs"])

        # 训练 dynamics
        for _ in range(cfg.train.model_updates):
            rng, kb = jax.random.split(rng)
            b = replay_env.sample(kb, batch_size=cfg.train.batch_size)
            model_state, metrics_m = model_train_step(model_state, b, std)
        print(f"Model NLL: {float(metrics_m['nll']):.4f}")

        # -------------------------------------------------
        # 3) 模型 rollout 生成 D_model
        #    （rollout.py 内部已包含 adaptive n^j 逻辑）
        # -------------------------------------------------
        rng, kr = jax.random.split(rng)
        replay_model = rollout_model(
            rng=kr,
            model_state=model_state,
            std=std,
            policy_state=policy_state,
            opponent_policies=[{"state": s} for s in opponent_states],
            replay_env=replay_env,
            replay_model=replay_model,
            cfg=cfg,
        )

        # -------------------------------------------------
        # 4) 用 D_model 更新 Q & Policy
        # -------------------------------------------------

        for _ in range(cfg.train.gradient_updates):
            rng, ksamp = jax.random.split(rng)
            batch = replay_model.sample(ksamp, batch_size=cfg.train.batch_size)

            # 先更新两个 Q（update_q_function 里已做最小化目标）
            q1_state, q2_state, q_metrics, rng = update_q_function(
                q1_state=q1_state,
                q2_state=q2_state,
                target_q1_state=target_q1_state,
                target_q2_state=target_q2_state,
                policy_state=policy_state,
                batch=batch,
                cfg=cfg.q_function,
                rng=rng,
            )

            q1_pred = q1_state.apply_fn({"params": q1_state.params}, batch["obs"], batch["a_ego"])
            q2_pred = q2_state.apply_fn({"params": q2_state.params}, batch["obs"], batch["a_ego"])
            mean_q1 = jnp.mean(q1_pred)
            mean_q2 = jnp.mean(q2_pred)
            if mean_q1 < mean_q2:
                smaller_q_state = q1_state
                # print("Q1 is smaller, mean:", float(mean_q1))
            else:
                smaller_q_state = q2_state
                # print("Q2 is smaller, mean:", float(mean_q2))

            # 再更新 policy（用 Q 的最小值）
            policy_state, pi_metrics = update_policy(
                policy_state=policy_state,
                q_state=smaller_q_state,   # 如果你在 update_policy 里使用 min(Q1,Q2)，这里传个结构或改函数
                batch=batch,
                cfg=cfg.policy,
                rng=rng,
            )

            # 软更新 target Q
            target_q1_state = soft_update(target_q1_state, q1_state, cfg.q_function.tau)
            target_q2_state = soft_update(target_q2_state, q2_state, cfg.q_function.tau)

        # 简单日志
        q1l = float(q_metrics.get("q1_loss", 0.0))
        q2l = float(q_metrics.get("q2_loss", 0.0))
        pil = float(pi_metrics.get("policy_loss", 0.0))
        print(f"[Epoch {epoch}] Q1 {q1l:.4f} | Q2 {q2l:.4f} | Policy {pil:.4f}")

    print("\n🎉 Training finished.")


if __name__ == "__main__":
    # 避免 Hydra 改工作目录找不到相对路径
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
