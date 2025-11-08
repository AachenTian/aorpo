# aorpo/rollout/rollout.py
from __future__ import annotations
from typing import Dict, Any, List
import jax
import jax.numpy as jnp
from flax.nnx import TrainState

from aorpo.agents.model_dynamics import predict_next, eval_error
from aorpo.agents.policy import PolicyNet


# -----------------------------------------------------
# Compute adaptive rollout length for each opponent
# -----------------------------------------------------
def compute_rollout_lengths(errors: List[float], k: int) -> List[int]:
    """
    根据 AORPO 论文公式计算每个对手的 rollout 步数：
        n^j = floor(k * min_j'(ε̂_j') / ε̂_j)
    Args:
        errors: list of opponent model errors ε̂_j
        k: maximum rollout length (超参数)
    Returns:
        n_js: list[int], rollout steps for each opponent
    """
    min_err = min(errors)
    n_js = []
    for e in errors:
        print("errors",errors)
        print(e)
        ratio = k * min_err / float(e)
        print(ratio)
        n_js.append(max(1, int(ratio)))
    return n_js


# -----------------------------------------------------
# Rollout using learned dynamics + opponent models
# -----------------------------------------------------
def rollout_model(
    rng: Any,
    model_state: Any,
    std: Any,
    policy_state: Any,
    opponent_policies: List[Dict[str, Any]],
    replay_env: Any,
    replay_model: Any,
    cfg: Any,
):
    """
    Perform adaptive opponent-wise model rollouts.

    Args:
        rng: jax.random.PRNGKey
        model_state: trained dynamics model
        std: Standardizer (for model_dynamics normalization)
        policy_state: main agent policy (πζ)
        opponent_policies: list of opponent policy dicts [{state, model}, ...]
        replay_env: real environment replay buffer
        replay_model: model replay buffer (to store rollouts)
        cfg: rollout configuration (Hydra)
    """

    # 1️ 从真实经验池采样初始状态
    key, subkey = jax.random.split(rng)
    batch_env = replay_env.sample(subkey, cfg.rollout.batch_size)
    obs = batch_env["obs"]

    # 2️ 计算每个 opponent 模型误差 ε̂_j
    errors = []
    for opp in opponent_policies:
        eps_j = eval_error(
            real_state=policy_state,    #####################################
            opp_state=opp["state"],
            std=std,
            batch=batch_env,
            deterministic=True,
        )
        errors.append(float(eps_j))

    # 3️ 根据公式计算每个对手的 rollout 步数 n^j
    n_js = compute_rollout_lengths(errors, cfg.rollout.k)
    max_n = cfg.rollout.k  # 最大 rollout 步数上限

    print(f"Opponent model errors: {errors}")
    print(f"Adaptive rollout steps (n^j): {n_js}")

    # 4️ 模型rollout 循环
    for step in range(max_n):
        rng, subkey = jax.random.split(rng)

        # 主体动作 a_i
        a_i, _, subkey = PolicyNet.sample_action(
            policy_state.params,
            policy_state.apply_fn,
            subkey,
            obs,
        )

        # 每个对手动作 a_j
        a_js = []
        for j, opp in enumerate(opponent_policies):
            if step < n_js[j]:
                # 使用 learned opponent policy
                a_j, _, subkey = PolicyNet.sample_action(
                    opp["state"].params,
                    opp["state"].apply_fn,
                    subkey,
                    obs,
                )
            else:
                # 超出 n_j 时可模拟通信 (Comm)，此处简单保持上一步动作
                a_j = a_js[-1] if len(a_js) > 0 else jnp.zeros_like(a_i)
            a_js.append(a_j)

        # 联合动作
        joint_act = jnp.concatenate([a_i] + a_js, axis=-1)
        a_js = jnp.concatenate(a_js, axis=-1)  # 形状：(batch_size, opp_num * act_dim)
        # 预测下一状态（模型）
        next_obs = predict_next(
            state=model_state,
            std=std,
            obs=obs,
            a_ego=a_i,
            a_opp=a_js,
            rng=subkey,
            deterministic=False,
        )

        # 奖励可选，这里使用简单负距离奖励占位符
        reward = -jnp.linalg.norm(joint_act, axis=-1)

        # 存储到模型经验池
        replay_model.add_batch(dict(
            obs=obs,
            a_ego=a_i,
            a_opp=a_js,
            rew=reward,
            next_obs=next_obs,
            done=jnp.zeros_like(reward)
        ))

        obs = next_obs

    return replay_model
