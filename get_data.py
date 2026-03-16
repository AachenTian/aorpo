import wandb
import pandas as pd
from functools import reduce

ENTITY = "yachen-tian-rwth-aachen-university"
PROJECT = "AORPO-dynamics model"
# RUN_IDS = ["076aeh46", "yr331itg", "zs5yt6l6", "8ixhtw37", "djvdzln3", "zdcclxwl", "k217xrqp", "axoh0q85", "zcvrnrpi", "ybposc5q", "7m2xscks", "o6ic3h88"]
RUN_IDS = ["076aeh46", "yr331itg", "k217xrqp", "axoh0q85"]
RUN_NAMES = [""]
# 确保这里的 Key 名字和你打印出来的一模一样
METRIC_KEYS = ["total_comm_count", "episode_reward_dyna"]

api = wandb.Api()
all_runs_final = []

for rid in RUN_IDS:
    run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
    print(f"\n--- 正在处理 Run: {rid} ---")

    metric_dfs = []

    for key in METRIC_KEYS:
        # 关键修改：直接使用 history 并设置 samples 为一个极大的值
        # 这比 scan_history 在某些情况下更稳定，且能强行拉取所有点
        df_temp = run.history(keys=["_step", key], samples=100000)

        if not df_temp.empty and key in df_temp.columns:
            # 去掉该指标为空的行
            df_temp = df_temp.dropna(subset=[key])
            print(f"✅ 指标 [{key}] 提取成功: {len(df_temp)} 行")
            metric_dfs.append(df_temp)
        else:
            print(f"❌ 指标 [{key}] 提取失败：请检查该指标在 W&B 中是否真的有数据。")

    # 只有当两个指标都拿到数据时，才进行合并
    if len(metric_dfs) == len(METRIC_KEYS):
        # 使用 merge_ordered 对齐并向前填充
        run_df = reduce(lambda left, right: pd.merge_ordered(left, right, on="_step", fill_method="ffill"), metric_dfs)

        run_df["run_id"] = rid
        run_df["run_name"] = run.name
        all_runs_final.append(run_df)
        print(f"🔗 合并对齐完成，总行数: {len(run_df)}")
    else:
        print(f"⚠️ 跳过 Run {rid}，因为指标不全。")

# 合并所有实验
if all_runs_final:
    final_df = pd.concat(all_runs_final, ignore_index=True)
    final_df.to_csv("mpe_simple_spread.csv", index=False)
    print("\n✨ 最终文件已保存：mpe_simple_spread.csv")
else:
    print("\n❌ 没有任何数据被合并，请检查指标名。")


#
# import wandb
# import pandas as pd
# from functools import reduce
#
# ENTITY = "yachen-tian-rwth-aachen-university"
# PROJECT = "AORPO-simple_tag"
# RUN_IDS = ["yc5j624o", "86bwq0c8", "ftaey7xg", "k05ey7b6"]
# RUN_NAMES = [""]
# # 确保这里的 Key 名字和你打印出来的一模一样
# METRIC_KEYS = ["total_comm_count", "epi_reward_agent-0"]
#
# api = wandb.Api()
# all_runs_final = []
#
# for rid in RUN_IDS:
#     run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
#     print(f"\n--- 正在处理 Run: {rid} ---")
#
#     metric_dfs = []
#
#     for key in METRIC_KEYS:
#         # 关键修改：直接使用 history 并设置 samples 为一个极大的值
#         # 这比 scan_history 在某些情况下更稳定，且能强行拉取所有点
#         df_temp = run.history(keys=["_step", key], samples=100000)
#
#         if not df_temp.empty and key in df_temp.columns:
#             # 去掉该指标为空的行
#             df_temp = df_temp.dropna(subset=[key])
#             print(f"✅ 指标 [{key}] 提取成功: {len(df_temp)} 行")
#             metric_dfs.append(df_temp)
#         else:
#             print(f"❌ 指标 [{key}] 提取失败：请检查该指标在 W&B 中是否真的有数据。")
#
#     # 只有当两个指标都拿到数据时，才进行合并
#     if len(metric_dfs) == len(METRIC_KEYS):
#         # 使用 merge_ordered 对齐并向前填充
#         run_df = reduce(lambda left, right: pd.merge_ordered(left, right, on="_step", fill_method="ffill"), metric_dfs)
#
#         run_df["run_id"] = rid
#         run_df["run_name"] = run.name
#         all_runs_final.append(run_df)
#         print(f"🔗 合并对齐完成，总行数: {len(run_df)}")
#     else:
#         print(f"⚠️ 跳过 Run {rid}，因为指标不全。")
#
# # 合并所有实验
# if all_runs_final:
#     final_df = pd.concat(all_runs_final, ignore_index=True)
#     final_df.to_csv("mpe_simple_tag.csv", index=False)
#     print("\n✨ 最终文件已保存：mpe_simple_spread.csv")
# else:
#     print("\n❌ 没有任何数据被合并，请检查指标名。")
