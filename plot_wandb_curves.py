import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

# =========================
# 1. 基本配置
# =========================
ENTITY = "yachen-tian-rwth-aachen-university"
PROJECT = "AORPO-dynamics model"

METRIC_NAME = "episode_reward_env"
STEP_KEY = "_step"

OUT_DIR = "./figures"
os.makedirs(OUT_DIR, exist_ok=True)

# 统一插值到这些 step
COMMON_STEPS = np.arange(0, 30001, 1000)

# 你手动指定每条线要包含哪些 run.name
RUN_GROUPS = {
    "Ours": [
        "routing reset_seed = 40 xi = 50",
        "routing reset_seed =50 k = 25",
        "routing reset_seed = 30 k = 6 policy lr = 0.01",
    ],
    "AORPO": [
        "no routing reset_seed = 40 k =6",
        "no routing aorpo reset_seed = 50 k = 6",
        "no routing aorpo rest_seed = 30 k = 6",
    ],
}

TITLE = "Cooperative communication"
XLABEL = "dynamics interactions"
YLABEL = "episode rewards"

SHADE_MODE = "std"   # 可选 "std" 或 "sem"


# =========================
# 2. 工具函数
# =========================
def fetch_run_history(run, step_key, metric_name):
    hist = run.history(keys=[step_key, metric_name], pandas=True)
    if hist is None or len(hist) == 0:
        return None

    hist = hist[[step_key, metric_name]].dropna()
    if len(hist) < 2:
        return None

    hist = hist.sort_values(step_key)
    hist = hist.drop_duplicates(subset=[step_key], keep="last")
    return hist


def interpolate_series(df, x_key, y_key, common_x):
    x = df[x_key].to_numpy()
    y = df[y_key].to_numpy()

    if len(x) < 2:
        return None

    mask = (common_x >= x.min()) & (common_x <= x.max())
    if mask.sum() < 2:
        return None

    y_interp = np.full_like(common_x, fill_value=np.nan, dtype=np.float64)
    y_interp[mask] = np.interp(common_x[mask], x, y)
    return y_interp


def aggregate_runs(y_list, shade_mode="std"):
    ys = np.stack(y_list, axis=0)  # shape: (N_runs, T)

    # 哪些 step 至少有一个有效值
    valid_cols = np.any(~np.isnan(ys), axis=0)

    mean = np.full(ys.shape[1], np.nan, dtype=np.float64)
    spread = np.full(ys.shape[1], np.nan, dtype=np.float64)

    if valid_cols.any():
        mean[valid_cols] = np.nanmean(ys[:, valid_cols], axis=0)

        if shade_mode == "sem":
            counts = np.sum(~np.isnan(ys[:, valid_cols]), axis=0)
            std = np.nanstd(ys[:, valid_cols], axis=0)
            spread[valid_cols] = std / np.sqrt(np.maximum(counts, 1))
        else:
            spread[valid_cols] = np.nanstd(ys[:, valid_cols], axis=0)

    return mean, spread


# =========================
# 3. 主逻辑
# =========================
def main():
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    # 建一个字典：run.name -> run对象
    run_dict = {run.name: run for run in runs}

    plt.figure(figsize=(8, 6))

    for group_name, run_names in RUN_GROUPS.items():
        y_list = []

        print(f"\n=== Processing group: {group_name} ===")
        for run_name in run_names:
            if run_name not in run_dict:
                print(f"[WARN] Run not found: {run_name}")
                continue

            run = run_dict[run_name]
            hist = fetch_run_history(run, STEP_KEY, METRIC_NAME)
            if hist is None:
                print(f"[WARN] No valid history for: {run_name}")
                continue

            y_interp = interpolate_series(hist, STEP_KEY, METRIC_NAME, COMMON_STEPS)
            if y_interp is None:
                print(f"[WARN] Failed interpolation for: {run_name}")
                continue

            y_list.append(y_interp)
            print(f"[OK] Added run: {run_name}")

        if len(y_list) == 0:
            print(f"[WARN] No valid runs for group: {group_name}")
            continue

        mean, spread = aggregate_runs(y_list, shade_mode=SHADE_MODE)

        valid = ~np.isnan(mean)
        x = COMMON_STEPS[valid]
        m = mean[valid]
        s = spread[valid]

        plt.plot(x, m, linewidth=2, label=f"{group_name} (n={len(y_list)})")
        plt.fill_between(x, m - s, m + s, alpha=0.2)

    plt.title(TITLE, fontsize=18)
    plt.xlabel(XLABEL, fontsize=15)
    plt.ylabel(YLABEL, fontsize=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, "selected_runs_plot.png")
    out_pdf = os.path.join(OUT_DIR, "selected_runs_plot.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.show()

    print(f"\nSaved figure to:\n  {out_png}\n  {out_pdf}")


if __name__ == "__main__":
    main()