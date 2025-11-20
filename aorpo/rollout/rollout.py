# aorpo/rollout/rollout.py
from __future__ import annotations
from typing import Dict, Any, List
import jax
import jax.numpy as jnp
from flax.nnx import TrainState
from omegaconf import DictConfig, OmegaConf
import wandb, random
from dataclasses import dataclass

from aorpo.agents.model_dynamics import predict_next, eval_error, unflatten_batch
from aorpo.agents.policy import PolicyNet
from aorpo.utils.replay import ReplayBuffer

@dataclass
class State:
    p_pos: jnp.ndarray
    p_vel: jnp.ndarray
    c: jnp.ndarray
    done: jnp.ndarray
    step: jnp.ndarray

def dict_to_state(d):
    return State(
        p_pos=d["p_pos"],
        p_vel=d["p_vel"],
        c=d["c"],
        done=d["done"],
        step=d["step"],
    )
# -----------------------------------------------------
# communication function
# -----------------------------------------------------
def Comm(policy_state, obs_j, rng):
    action, _, rng = PolicyNet.sample_action(
        policy_state.params,
        policy_state.apply_fn,
        rng,
        obs_j
    )
    return action, rng


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

def add_batch_model_to_replay(replay: ReplayBuffer, batch: dict, cfg:DictConfig) -> ReplayBuffer:
    return replay.add_batch_model(batch, cfg)

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
    batch_env = replay_env.sample(subkey, cfg.rollout.batch_size, cfg.train.num_opponents)
    obs = batch_env["obs"]
    state = batch_env["state"]

    # 2️ 计算每个 opponent 模型误差 ε̂_j
    errors = []
    for j, opp in enumerate(opponent_policies):
        eps_j = eval_error(
            real_state=policy_state,
            opp_state=opp["state"],
            std=std,
            batch=batch_env,
            deterministic=True,
            member_idx=j,
        )
        errors.append(float(eps_j))

    # 3️ 根据公式计算每个对手的 rollout 步数 n^j
    n_js = compute_rollout_lengths(errors, cfg.rollout.k)
    max_n = cfg.rollout.k  # 最大 rollout 步数上限

    print(f"Opponent model errors: {errors}")
    print(f"Adaptive rollout steps (n^j): {n_js}")

    reward_roll = 0
    # 4️ 模型rollout 循环
    for step in range(max_n):
        rng, subkey = jax.random.split(rng)

        # 主体动作 a_i
        a_i, _, subkey = PolicyNet.sample_action(
            policy_state.params,
            policy_state.apply_fn,
            subkey,
            obs["agent_0"],
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
                    obs[f"agent_{j+1}"],
                )
            else:
                # 超出 n_j 时Communicate
                a_j, subkey = Comm(policy_state, obs[f"agent_{j+1}"], subkey)
            a_js.append(a_j)

        # 联合动作
        joint_act = jnp.concatenate([a_i] + a_js, axis=-1)
        a_js = jnp.concatenate(a_js, axis=-1)  # 形状：(batch_size, opp_num * act_dim)
        # 预测下一状态（模型）
        next_state, next_obs, reward_dict, dones_dict= predict_next(
            state=model_state,
            std=std,
            state_agent=state,
            a_ego=a_i,
            a_opp=a_js,
            rng=subkey,
            deterministic=False,
        )

        # state = jax.tree_util.tree_map(lambda x: x[None], state)
        # next_state = jax.tree_util.tree_map(lambda x: x[None], next_state)

        reward_mean = jnp.mean(reward_dict["agent_0"])
        reward_roll += reward_mean
        batch_model = dict(
            state=state,
            obs=obs,
            a_ego=a_i,
            a_opp=a_js,
            next_state=next_state,
            next_obs=next_obs,
            rew=reward_dict,
            dones=dones_dict,
        )
        # print("batch_model_state.shape:", batch_model["state"].shape)
        # print("batch_model_obs.shape:", batch_model["obs"].shape)
        # # batch_model["state"] = dict_to_state(batch_model["state"])
        # # batch_model["next_state"] = dict_to_state(batch_model["next_state"])
        # batch_model["state"] = batch_model["state"][:, None, :]
        # batch_model["next_state"] = batch_model["next_state"][:, None, :]
        # print("batch_model_state.shape:", batch_model["state"].shape)
        # batch_model["state"] = unflatten_batch(batch_model["state"])
        # batch_model["next_state"] = unflatten_batch(batch_model["next_state"])
        # print("batch_model_state.shape:", batch_model["state"].p_pos.shape)
        # print("batch_model_state.shape:", batch_model["state"].p_vel.shape)
        # print("batch_model_state.shape:", batch_model["state"].c.shape)
        # print("batch_model_state.shape:", batch_model["state"].dones.shape)
        # print("batch_model_state.shape:", batch_model["state"].step.shape)
        # 存储到模型经验池
        replay_model = add_batch_model_to_replay(replay_model, batch_model, cfg)
        obs = next_obs
    wandb.log({
        "episode rewards": reward_roll
    })

    return replay_model
