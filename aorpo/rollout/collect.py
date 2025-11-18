# aorpo/rollout/collect.py
import jax
import jax.numpy as jnp
from aorpo.agents.policy import PolicyNet
from aorpo.envs.jaxmarl_simple_spread_v3_env_wrapper import make_mpe_env, env_step, env_reset
from aorpo.agents.model_dynamics import get_obs, predict_next
from omegaconf import DictConfig

def collect_real_data(policy_fn, opp_fn, obs_dim, act_dim, opp_num, opp_dim, key, cfg: DictConfig):
    """Collect real environment data using JAX scan."""
    def rollout(carry, _):
        state, obs, key = carry
        key, sub1, sub2 = jax.random.split(key, 3)
        a_ego, sub1 = policy_fn(obs, sub1)
        a_opps, sub2 = opp_fn(obs, sub2)
        state2, obs2, r, dones, key = env_step(env, state, a_ego, a_opps, key)
        # joint_act = jnp.concatenate([a_ego, a_opps], axis=-1)
        return (state2, obs2, key), (state2, obs2, state, obs, a_ego, a_opps, r, dones)

    # 初始化环境状态
    env = make_mpe_env(cfg)
    state, obs, key = env_reset(env, key)
    (final_state, final_obs, _), (next_state, next_obs, state, obs, a_ego, a_opp, rew, dones) = jax.lax.scan(
        rollout, (state, obs, key), None, length=cfg.collect.steps_per_epoch
    )
    batch = dict(state= state, obs=obs, a_ego=a_ego, a_opp=a_opp,  next_obs=next_obs, next_state=next_state, rew=rew, dones=dones)
    return batch, final_state, key

def episode_reward(policy_fn, opp_fn, num_agents, key, cfg):
    env = make_mpe_env(cfg)
    state, obs, key = env_reset(env, key)

    total_reward = 0.

    for t in range(25):
        key, sub1 = jax.random.split(key, 2)
        a_ego, sub1 = policy_fn(obs, sub1)

        key, sub2 = jax.random.split(key, 2)
        a_opp, sub2 = opp_fn(obs, sub2)

        key, sub3 = jax.random.split(key,2)
        state, obs, rewards, dones, sub3 = env_step(env, state, a_ego, a_opp, sub3)

        total_reward += sum(float(rewards[f"agent_{i}"]) for i in range(num_agents))

        if dones["agent_0"]:
            break

    return float(total_reward)

def rollout_env(policy_fn, opp_fn, init_state, init_obs, key, horizon, cfg: DictConfig):
    def rollout_scan(carry, _):
        state, obs, key_env = carry
        key_env, k1, k2, k3 = jax.random.split(key_env, 3)

        a_ego = policy_fn(obs, k1)
        a_opp = opp_fn(obs, k2)

        next_state, next_obs, _, _, k3 = env_step(env, state, a_ego, a_opp, k3)
        return (next_state, next_obs, key_env), next_state
    env = make_mpe_env(cfg)
    (final_state, final_obs, _), state = jax.lax.scan(rollout_scan, (init_state, init_obs, key), None, length=horizon)
    return state

def rollout_dynamics(policy_fn, opp_fn, model_state, std, init_state, init_obs, key, horizon):
    def rollout_scan_dyna(carry,_):
        state, obs, key_dyna = carry
        key_dyna, k1, k2, k3 = jax.random.split(key)

        a_ego = policy_fn(obs, k1)
        a_opp = opp_fn(obs, k2)

        next_state, next_obs, reward_dict, dones_dict = predict_next(
            state=model_state,
            std=std,
            state_agent=state,
            a_ego=a_ego,
            a_opp=a_opp,
            rng=k3,
            deterministic=False,
        )
        return (next_state, next_obs, key_dyna), next_state
    (final_state, final_obs, _), state = jax.lax.scan(rollout_scan_dyna, (init_state, init_obs, key), None, length=horizon)
    return state

def rollout_compare(policy_fn, opp_fn, model_state, std, key, horizon, cfg: DictConfig):
    key, k1, k2 = jax.random.split(key, 3)
    env = make_mpe_env(cfg)
    init_state, init_obs, key = env_reset(env, k1)
    state_env = rollout_env(policy_fn, opp_fn, init_state, init_obs, k2, horizon, cfg)
    state_dyna = rollout_dynamics(policy_fn, opp_fn, model_state, std, init_state, init_obs, k2, horizon)
    return state_env, state_dyna
























