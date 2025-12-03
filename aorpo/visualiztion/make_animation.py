import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def animate_episode(traj, save_path="episode_animation.mp4"):
    agents = np.array(traj["agents"])        # (T, 3, 2)
    landmarks = np.array(traj["landmarks"])  # (T, 3, 2)
    rewards = np.array(traj["rewards"])      # (T,)
    T = agents.shape[0]

    fig, (ax_reward, ax_env) = plt.subplots(1, 2, figsize=(10, 5))

    # ---------- 左侧 reward ----------
    ax_reward.set_title("Reward per step")
    ax_reward.set_xlabel("Step")
    ax_reward.set_ylabel("Reward")
    line_reward, = ax_reward.plot([], [], lw=2)
    ax_reward.set_xlim(0, T)
    ax_reward.set_ylim(np.min(rewards) - 1, np.max(rewards) + 1)

    # ---------- 右侧环境图 ----------
    ax_env.set_title("Agents & Landmarks")
    ax_env.set_xlim(-1.5, 1.5)
    ax_env.set_ylim(-1.5, 1.5)

    # 初始化为单一颜色，不要 3 个
    scat_agents = ax_env.scatter([], [], s=80, color='blue', label="Agents")
    scat_landmarks = ax_env.scatter([], [], s=120, color='black', marker='X')

    def update(frame):
        # 左侧 reward
        line_reward.set_data(np.arange(frame), rewards[:frame])

        # 右侧位置
        scat_agents.set_offsets(agents[frame])
        scat_agents.set_color(['red', 'blue', 'green'])  # 3 agents → 3 colors

        scat_landmarks.set_offsets(landmarks[frame])
        scat_landmarks.set_color(['black', 'black', 'black'])

        return line_reward, scat_agents, scat_landmarks

    ani = animation.FuncAnimation(fig, update, frames=T, interval=150, blit=True)
    ani.save(save_path, fps=10, dpi=120)
    print(f"Animation saved → {save_path}")

