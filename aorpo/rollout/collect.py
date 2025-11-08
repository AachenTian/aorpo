# aorpo/rollout/collect.py
import jax
import jax.numpy as jnp
from aorpo.envs.jax_toy_env import env_step, env_reset

def collect_real_data(policy_fn, opp_fn, obs_dim, act_dim, opp_num, opp_dim, steps, key):
    """Collect real environment data using JAX scan."""
    def rollout(carry, _):
        s, key = carry
        key, sub1, sub2 = jax.random.split(key, 3)
        a_ego = policy_fn(s, sub1)
        a_opps = opp_fn(s, sub2)
        s2, r, done, key = env_step(s, a_ego, a_opps, key)
        # joint_act = jnp.concatenate([a_ego, a_opps], axis=-1)
        return (s2, key), (s, a_ego, a_opps, s2, r)

    # 初始化环境状态
    state, key = env_reset(key, obs_dim)
    (final_state, _), (obs, a_ego, a_opp, next_obs, rew) = jax.lax.scan(
        rollout, (state, key), None, length=steps
    )
    batch = dict(obs=obs, a_ego=a_ego, a_opp=a_opp,  next_obs=next_obs, rew=rew)
    return batch, final_state, key
