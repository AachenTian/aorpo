import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COMM_KEY = "total_comm_count"
REWARD_KEYS = {
    "Spread": "episode_reward_dyna",
    "FACMAC": "epi_reward_agent-0"
}

FILES = {
    "Spread": "mpe_simple_spread.csv",
    "FACMAC": "mpe_simple_tag.csv"
}

def interpolate_run(df, common_comm):

    x = df[COMM_KEY].to_numpy()
    y = df[reward_key].to_numpy()

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    y_interp = np.interp(common_comm, x, y, left=y[0], right=y[-1])
    return y_interp


def aggregate(df, run_ids, common_comm):

    curves = []

    for rid in run_ids:
        run_df = df[df["run_id"] == rid]
        curves.append(interpolate_run(run_df, common_comm))

    curves = np.stack(curves)

    mean = curves.mean(axis=0)
    std = curves.std(axis=0)

    return mean, std


fig, axes = plt.subplots(1,2, figsize=(10,4))

for i,(env, path) in enumerate(FILES.items()):

    reward_key = REWARD_KEYS[env]

    df = pd.read_csv(path)

    runs = df["run_id"].unique()

    # 前半 AORPO
    aorpo_runs = runs[:len(runs)//2]

    # 后半 ours
    ours_runs = runs[len(runs)//2:]

    max_comm = df[COMM_KEY].max()

    COMMON_COMM = np.linspace(0, max_comm, 40)

    mean_aorpo, std_aorpo = aggregate(df, aorpo_runs, COMMON_COMM)
    mean_ours, std_ours = aggregate(df, ours_runs, COMMON_COMM)

    ax = axes[i]

    ax.plot(COMMON_COMM, mean_ours, linewidth=2.5, label="Uncertainty-Aware")
    ax.fill_between(COMMON_COMM, mean_ours-std_ours, mean_ours+std_ours, alpha=0.2)

    ax.plot(COMMON_COMM, mean_aorpo, linewidth=2.5, label="AORPO")
    ax.fill_between(COMMON_COMM, mean_aorpo-std_aorpo, mean_aorpo+std_aorpo, alpha=0.2)

    ax.set_title(env)
    ax.set_xlabel("Communication Count")

    if i == 0:
        ax.set_ylabel("Episode Reward")

    ax.grid(alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=False
)

plt.tight_layout(rect=(0,0.08,1,1))

plt.savefig("reward_vs_comm.pdf", dpi=300)
plt.show()