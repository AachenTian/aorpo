# aorpo/utils/replay.py
import jax
import jax.numpy as jnp
from dataclasses import dataclass

@dataclass
class ReplayBuffer:
    obs: jnp.ndarray
    a_ego: jnp.ndarray
    a_opp: jnp.ndarray
    next_obs: jnp.ndarray
    rew: jnp.ndarray
    done: jnp.ndarray
    max_size: int
    size: int
    ptr: int

    @staticmethod
    def create(max_size, obs_dim, act_dim, opp_dim):
        """初始化空buffer"""
        obs = jnp.zeros((max_size, obs_dim), dtype=jnp.float32)
        a_ego = jnp.zeros((max_size, act_dim), dtype=jnp.float32)
        a_opp = jnp.zeros((max_size, opp_dim), dtype=jnp.float32)
        next_obs = jnp.zeros((max_size, obs_dim), dtype=jnp.float32)
        rew = jnp.zeros((max_size, 1), dtype=jnp.float32)
        done = jnp.zeros((max_size, 1), dtype=jnp.float32)
        return ReplayBuffer(obs, a_ego, a_opp, next_obs, rew, done, max_size, 0, 0)

    def add_batch(self, batch):
        """Add a batch (dict of jnp arrays) to replay buffer."""
        B = batch["obs"].shape[0]
        idx = (jnp.arange(B) + self.ptr) % self.max_size

        new_obs = self.obs.at[idx].set(batch["obs"])
        new_a_ego = self.a_ego.at[idx].set(batch["a_ego"])
        new_a_opp = self.a_opp.at[idx].set(batch["a_opp"])
        new_next = self.next_obs.at[idx].set(batch["next_obs"])

        # --- 确保 reward 和 done 的形状一致 ---
        rew = batch["rew"]
        if rew.ndim == 1:
            rew = rew.reshape(-1, 1)
        new_rew = self.rew.at[idx].set(rew)

        # --- 如果没有 done，就创建一个 zeros ---
        if "done" in batch:
            done = batch["done"]
            if done.ndim == 1:
                done = done.reshape(-1, 1)
        else:
            done = jnp.zeros_like(rew)
        new_done = self.done.at[idx].set(done)

        # --- 更新指针 ---
        new_ptr = int((self.ptr + B) % self.max_size)
        new_size = int(jnp.minimum(self.size + B, self.max_size))

        return ReplayBuffer(
            new_obs, new_a_ego, new_a_opp, new_next, new_rew, new_done,
            self.max_size, new_size, new_ptr
        )

    def sample(self, key, batch_size):
        """JAX 随机采样"""
        idx = jax.random.randint(key, (batch_size,), 0, self.size)
        batch = dict(
            obs=self.obs[idx],
            a_ego=self.a_ego[idx],
            a_opp=self.a_opp[idx],
            next_obs=self.next_obs[idx],
            rew=self.rew[idx],
            done=self.done[idx],
        )
        return batch

    def __len__(self):
        return int(self.size)
