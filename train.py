# train.py
import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig, OmegaConf

from aorpo.agents.policy import init_policy_model
from aorpo.agents.q_function import init_q_function
from aorpo.agents.model_dynamics import init_model, train_step, Standardizer
from aorpo.agents.update_policy import update_policy
from aorpo.agents.update_q_function import update_q_function  # 你之后可以加上这个

@hydra.main(config_path="../aorpo/aorpo/configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):
    print("✅ Loaded Hydra config:")
    print(OmegaConf.to_yaml(cfg))
    print("✅ Starting fake-data training test...")

    # --- RNG initialization ---
    rng = jax.random.PRNGKey(0)
    rng, policy_key, q_key, model_key, obs_key, act_key, next_key = jax.random.split(rng, 7)

    obs_dim, act_dim = cfg.env.obs_dim, cfg.env.act_dim

    # --- initialization of policy, q_function and model_dynamics ---
    policy_model, policy_state = init_policy_model(policy_key, obs_dim, act_dim, cfg.agents.policy)
    q_model, q_state = init_q_function(q_key, obs_dim, act_dim, cfg.agents.q_function)
    model, model_state = init_model(model_key, obs_dim, act_dim, cfg.agents.model_dynamics)

    # --- prepare fake data ---
    fake_obs = jax.random.normal(obs_key, (128, obs_dim))
    fake_act = jax.random.normal(act_key, (128, act_dim))
    fake_next_obs = fake_obs + 0.1 * jax.random.normal(rng, fake_obs.shape) # 假设下一状态略有变化
    std = Standardizer.fit(fake_obs, fake_act, fake_next_obs)

    # --- batch ---
    batch = {
        "obs": fake_obs[:32],
        "act": fake_act[:32],
        "next_obs": fake_next_obs[:32],
        "rew": jax.random.normal(rng, (32, 1)),  # 随机奖励
        "done": jnp.zeros((32, 1)),
    }

    # --- simulation training loop ---
    for step in range(1, 6):  # 只跑5步看看能否收敛
        # 1️ 更新模型（model dynamics）
        model_state, model_metrics = train_step(model_state, batch, std)

        # 2️ 更新 Q-function（这里暂时先跳过 / 之后加）
        q_state, q_metrics = update_q_function(
            q_state=q_state,
            target_q_state=q_state,  # 如果你没有单独的 target Q，可临时传自己
            policy_state=policy_state,  # 传入 policy 的 TrainState
            batch=batch,
            cfg=cfg.agents.q_function,  # 这里传 DictConfig
        )

        # 3️ 更新策略（policy）
        policy_state, policy_metrics = update_policy(
            policy_state=policy_state,
            q_state=q_state,
            batch=batch,
            cfg=cfg.agents.policy,
            rng=rng
        )

        print(f"Step {step}: "
              f"Model NLL={model_metrics['nll']:.4f}, "
              f"Policy Loss={policy_metrics['policy_loss']:.4f}")

    print("✅ Fake-data training completed successfully!")


if __name__ == "__main__":
    main()
