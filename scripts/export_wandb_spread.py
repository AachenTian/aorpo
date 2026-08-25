import wandb
import pandas as pd
from functools import reduce

# ====== 配置 ======
ENTITY = "yachen-tian-rwth-aachen-university"
PROJECT = "AORPO-dynamics model"

RUN_IDS = [
    "076aeh46", "yr331itg", "zs5yt6l6", "8ixhtw37", "djvdzln3", "zdcclxwl", "k217xrqp", "axoh0q85", "zcvrnrpi", "ybposc5q", "7m2xscks", "o6ic3h88"
]

METRIC_KEYS = ["total_comm_count", "episode_reward_dyna"]

OUTPUT_FILE = "results/mpe_simple_spread.csv"

# ====== 初始化 ======
api = wandb.Api()
all_runs = []

# ====== 主循环 ======
for rid in RUN_IDS:
    run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
    print(f"\n--- 处理 Run: {rid} ({run.name}) ---")

    metric_dfs = []

    for key in METRIC_KEYS:
        # 🔥 使用 scan_history 获取完整数据（关键！）
        rows = list(run.scan_history(keys=["_step", key]))

        if len(rows) == 0:
            print(f"❌ [{key}] 没有数据")
            continue

        df_temp = pd.DataFrame(rows)

        if key not in df_temp.columns:
            print(f"❌ [{key}] 不存在")
            continue

        # 去掉 NaN
        if key != "total_comm_count":
            df_temp = df_temp.dropna(subset=[key])

        # 去重（有些 run 会重复 step）
        df_temp = df_temp.drop_duplicates(subset=["_step"])

        print(f"✅ [{key}] 行数: {len(df_temp)}")

        metric_dfs.append(df_temp)

    # ====== 合并不同 metric ======
    if len(metric_dfs) != len(METRIC_KEYS):
        print(f"⚠️ 跳过 Run {rid}（指标不全）")
        continue

    # 🔥 正确 merge（outer + 排序）
    run_df = reduce(
        lambda left, right: pd.merge(left, right, on="_step", how="outer"),
        metric_dfs
    ).sort_values("_step")

    # 🔥 统一 forward fill（避免错位）
    run_df = run_df.ffill()

    # 再清理一下（防止开头 NaN）
    run_df = run_df.dropna(subset=METRIC_KEYS)

    # 加标识
    run_df["run_id"] = rid
    run_df["run_name"] = run.name

    print(f"🔗 合并后行数: {len(run_df)}")

    all_runs.append(run_df)

# ====== 合并所有 runs ======
if len(all_runs) > 0:
    final_df = pd.concat(all_runs, ignore_index=True)

    # 排序更干净
    final_df["run_id"] = pd.Categorical(
        final_df["run_id"],
        categories=RUN_IDS,
        ordered=True
    )

    final_df = final_df.sort_values(["run_id", "_step"])

    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✨ 已保存: {OUTPUT_FILE}")
    print(f"总行数: {len(final_df)}")
else:
    print("\n❌ 没有有效数据")



