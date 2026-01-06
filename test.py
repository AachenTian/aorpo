import jax.numpy as jnp

# 1️⃣ 输入数据（转成 jnp.array）
step_ood_any_opp = jnp.array([[False, False, True], [False, True, False], [True, False, False], [False, True, False], [False, False, True]])

a_j_ens = jnp.array([
    [10, 11, 12],  # sample 0
    [20, 21, 22],  # sample 1
    [30, 31, 32],  # sample 2
    [40, 41, 42],  # sample 3
    [50, 51, 52],  # sample 4
])

a_j_comm = jnp.array([
    [ 1,  2,  3],
    [ 4,  5,  6],
    [ 7,  8,  9],
    [10, 11, 12],
    [13, 14, 15],
])

# 2️⃣ where + broadcast
actual_a_j = jnp.where(
    step_ood_any_opp,
    a_j_ens,
    a_j_comm,
)

true_ratio = jnp.mean(step_ood_any_opp.astype(jnp.float32))
print("OOD ratio:", true_ratio)

# 3️⃣ 打印结果
print(actual_a_j)
