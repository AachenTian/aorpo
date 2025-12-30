# aorpo/rollout/rollout.py
from __future__ import annotations
from typing import Dict, Any, List
import jax
import jax.numpy as jnp
from flax.nnx import TrainState
from omegaconf import DictConfig, OmegaConf
import wandb, random
from dataclasses import dataclass
from functools import partial

from aorpo.agents.model_dynamics import predict_next, eval_error, unflatten_batch
from aorpo.agents.policy import PolicyNet, EnsemblePolicyUtils
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
# uncertainty threshold of opponent model action
# -----------------------------------------------------
# 使用 partial 固定非 Array 参数，确保 jit 正常工作

def _compute_threshold_engine(opponent_states, obs_all, cfg):
    """
    内部核心引擎：对所有 opponent 计算动作熵的分位数阈值

    opponent_states: List[TrainState]，每个 state.apply_fn 是 EnsemblePolicyNet.apply
    obs_all: (num_opps, batch_size, obs_dim)
    return: (num_opps, act_dim)
    """
    num_opps = len(opponent_states)

    def get_single_opp_entropy(opp_state, single_opp_obs):
        """
        opp_state: 单个 opponent TrainState (ensemble policy)
        single_opp_obs: (B, obs_dim)
        return: entropies (B, act_dim)
        """
        # ✅ 关键：直接一次 apply，就得到 ensemble 输出
        # mu_ens/log_std_ens: (K, B, act_dim)
        mu_ens, log_std_ens = opp_state.apply_fn(
            {"params": opp_state.params},
            single_opp_obs
        )

        # ✅ 方差：sigma^2 = exp(2 * log_std)
        var_ale = jnp.exp(2.0 * log_std_ens)          # (K, B, act_dim)
        avg_ale_var = jnp.mean(var_ale, axis=0)       # (B, act_dim)

        avg_mu = jnp.mean(mu_ens, axis=0)             # (B, act_dim)
        epistemic_var = jnp.mean((mu_ens - avg_mu) ** 2, axis=0)  # (B, act_dim)

        total_var = avg_ale_var + epistemic_var       # (B, act_dim)

        # (B, act_dim) 维度熵
        ent = 0.5 * jnp.log(2 * jnp.pi * jnp.e * total_var + 1e-6)
        return ent

    # ✅ 外层不建议用 vmap 因为 opp_state 是 Python object（含 apply_fn method）
    # 用 for-loop 最稳（num_opps 很小，代价可忽略）
    all_entropies = []
    for j in range(num_opps):
        ent_j = get_single_opp_entropy(opponent_states[j], obs_all[j])  # (B, act_dim)
        all_entropies.append(ent_j)
    all_entropies = jnp.stack(all_entropies, axis=0)  # (num_opps, B, act_dim)

    # ✅ 对 batch 维度做分位数，得到每个 opponent 的阈值 lambda_j
    lambda_all = jnp.quantile(all_entropies, q=cfg.rollout.quantile_opp, axis=1)
    # shape: (num_opps, act_dim)
    return lambda_all


def compute_opponent_threshold(opponent_states, replay_env, cfg):
    """
    opponent_states: List[TrainState] (每个都是 ensemble policy state)
    return: (num_opps, act_dim)
    """
    batch_size = cfg.rollout.batch_size
    rng = jax.random.PRNGKey(0)

    batch = replay_env.sample(
        rng,
        batch_size=batch_size,
        opp_num=cfg.train.num_opponents
    )

    obs_list = [batch["obs"][f"agent_{j + 1}"] for j in range(cfg.train.num_opponents)]
    obs_all = jnp.stack(obs_list, axis=0)  # (num_opps, B, obs_dim)

    # ✅ 不再需要 combined_params（因为 apply_fn 已经处理 ensemble params）
    lambda_all = _compute_threshold_engine(
        opponent_states=opponent_states,
        obs_all=obs_all,
        cfg=cfg
    )

    return lambda_all
# -----------------------------------------------------
# Rollout using learned dynamics + opponent models
# -----------------------------------------------------
def rollout_model(
    rng: Any,
    transition_state: Any,
    reward_state: Any,
    std: Any,
    policy_state: Any,
    opponent_policies: List[Any],
    replay_env: Any,
    replay_model: Any,
    cfg: Any,
    epoch: Any,
):
    """
    Perform adaptive opponent-wise model rollouts.

    Args:
        rng: jax.random.PRNGKey
        transition_state: dynamics model (predict next_state)
        reward_state: reward model (predict reward)
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
    a_opp = batch_env["a_opp"]

    lambda_opp = compute_opponent_threshold(opponent_policies, replay_env, cfg)
    thresholds = lambda_opp[:, jnp.newaxis, :]
    # 2️ 计算每个 opponent 模型误差 ε̂_j
    errors = []
    for j, opp in enumerate(opponent_policies):
        target = a_opp[:, j * cfg.env.act_dim:(j + 1) * cfg.env.act_dim]
        pred, _ = opp.apply_fn({"params": opp.params}, obs[f"agent_{j + 1}"])
        pred = jnp.tanh(pred)
        eps_j = jnp.mean((pred - target) ** 2)
        # eps_j = eval_error(
        #     real_state=policy_state,
        #     opp_state=opp["state"],
        #     std=std,
        #     batch=batch_env,
        #     deterministic=True,
        #     member_idx=j,
        # )
        errors.append(float(eps_j))

    # 3️ 根据公式计算每个对手的 rollout 步数 n^j
    max_n = cfg.rollout.k  # 最大 rollout 步数上限
    def get_k(epoch, max_n):
        epoch = jnp.asarray(epoch)
        start_epoch = 15
        end_epoch = 100
        k_min = 1
        k_max = max_n

        # 线性插值 (浮点)
        k_float = k_min + (k_max - k_min) * ((epoch - start_epoch) / (end_epoch - start_epoch))

        # 四舍五入为整数
        k_int = jnp.round(k_float)

        # 三段逻辑：epoch < 15 → 1；epoch > 100 → max_n；中间 → 插值
        k = jnp.where(epoch < start_epoch, k_min,
                      jnp.where(epoch > end_epoch, k_max, k_int))

        return k.astype(int)
    max_n = get_k(epoch, max_n)
    n_js = compute_rollout_lengths(errors, int(max_n))

    print(f"Opponent model errors: {errors}")
    print(f"Adaptive rollout steps (n^j): {n_js}")

    reward_roll = 0
    #initialize a all True mask
    active_mask = jnp.ones((cfg.rollout.batch_size,), dtype=jnp.bool_)
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
        step_ood_any_opp = jnp.zeros((cfg.rollout.batch_size,), dtype=jnp.bool_)
        for j, opp in enumerate(opponent_policies):

            # 使用 learned opponent policy
            a_j_ens, mu_j, log_std_j, subkey = EnsemblePolicyUtils.sample_action_ensemble(
                opp.params,
                opp.apply_fn,
                subkey,
                obs[f"agent_{j+1}"],
            )
            current_entropy = EnsemblePolicyUtils.get_infoprop_entropy(mu_j, log_std_j)
            ood_mask_j = current_entropy > thresholds[j]
            opp_j_is_ood = jnp.any(ood_mask_j, axis=-1)
            step_ood_any_opp = jnp.logical_or(step_ood_any_opp, opp_j_is_ood)
            a_j_comm, subkey = Comm(policy_state, obs[f"agent_{j+1}"], subkey)
            actual_a_j = jnp.where(active_mask[:, jnp.newaxis], a_j_ens, a_j_comm)

            a_js.append(actual_a_j)

        # 联合动作
        joint_act = jnp.concatenate([a_i] + a_js, axis=-1)
        a_js = jnp.concatenate(a_js, axis=-1)  # 形状：(batch_size, opp_num * act_dim)
        # 预测下一状态（模型）
        next_state, next_obs, reward_dict, dones_dict= predict_next(
            transition_state=transition_state,
            reward_state=reward_state,
            std=std,
            state_agent=state,
            a_ego=a_i,
            a_opp=a_js,
            cfg=cfg,
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

        # 存储到模型经验池
        replay_model = add_batch_model_to_replay(replay_model, batch_model, cfg)
        obs = next_obs
        state = next_state
    wandb.log({
        "episode rewards": reward_roll
    })

    return replay_model
