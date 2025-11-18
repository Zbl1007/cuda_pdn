
import yaml
import numpy as np
import torch
import torch.nn as nn
import re
import matplotlib.pyplot as plt
import time
import pickle
from OpAdjoint import OpAdjointFunction
from OpSimulation import OpSimulationScipy, OpSimulationPardiso
from Circuit import Circuit, BranchType
from math import exp, log, sqrt, pi
from build_ckt import build_op_ckt

# multigpu: 28 239.03s | 33 6.09s
# micro150: 23 39.36s | 24 4.13s
# ascend910: 13 47.69s | 14 4.74s
case = "multigpu"
file = "data/{}.yaml".format(case)
file_result = "data/2025_11/{}_result_tsv_10.yaml".format(case)
with open(file, "r") as f:
    design = yaml.load(f.read(), Loader=yaml.FullLoader)
result = dict()

interposer_w = design["interposer"]["w"] * 5
interposer_h = design["interposer"]["h"] * 5
vdd = design["vdd"]
vdd_threshold = vdd * 0.02
ckt = build_op_ckt(file)

# -------------------------
# make observe and candidate
# -------------------------
observe_branch = [
    "i_{}_{}".format(x, y) for x in range(interposer_w) for y in range(interposer_h)
]
candidate_branch = [
    "g_{}_{}".format(x, y) for x in range(interposer_w) for y in range(interposer_h)
]
candidate_tsv = [
    "tsv_{}_{}".format(x, y) for x in range(interposer_w) for y in range(interposer_h)
]
result_tsv = []

# -------------------------
# NN parameter
# -------------------------

# best: niters = 1000, lr = 0.3, tau_iter = 20, vio_coeff = 10, seed=42

niters = 1000
lr = 0.3
typ, u, v, val, index = ckt.prepare_sim("0", observe_branch + candidate_branch)
g_index = index[len(observe_branch) :]
r_index = torch.tensor([], dtype=torch.long)
i_index = index[: len(observe_branch)]
v_index = torch.tensor([], dtype=torch.long)
(all_exc_index,) = torch.where((typ == BranchType.I) | (typ == BranchType.V))
base_g_value = val[g_index]
r_value = torch.tensor([], dtype=torch.float64)
all_exc_value = val[all_exc_index]
sim = OpSimulationPardiso(typ, u, v, val)

temperature = 1
temperature_ratio = 0.9
temperature_update_iteration = niters // 20

tsv_count_coeff = 0.1
total_drop_violation_coeff = 1000

# -------------------------
# Train NN 3872 3733
# -------------------------
loss_history = []
count_loss_history = []
drop_loss_history = []

start = time.time()
torch.manual_seed(42)
theta = nn.Parameter(torch.clamp(torch.randn((g_index.size(0), 2)), -1.0, 1.0))
optimizer = torch.optim.Adam(nn.ParameterList([theta]), lr=lr)
for i in range(niters):
    # update gumbel softmax temperature
    if i != 0 and i % temperature_update_iteration == 0:
        temperature *= temperature_ratio

    optimizer.zero_grad()
    p = torch.nn.functional.gumbel_softmax(theta, tau=temperature, dim=1)
    g_value = p[:, 1] * base_g_value
    i_voltage, v_current = OpAdjointFunction.apply(
        g_index,
        r_index,
        i_index,
        v_index,
        all_exc_index,
        g_value,
        r_value,
        all_exc_value,
        sim,
    )

    ir_drop = vdd - i_voltage
    total_drop_violation = torch.nn.functional.relu(ir_drop - vdd_threshold).sum()
    tsv_count = p[:, 1].sum()

    loss = (
        total_drop_violation_coeff * total_drop_violation + tsv_count_coeff * tsv_count
    )
    loss_history.append(loss.detach().item())
    drop_loss_history.append(total_drop_violation.detach().item())
    count_loss_history.append(tsv_count.detach().item())
    print(
        "iter {}: loss {}, worst_drop {}, total_drop_violation {}, tsv_count {}".format(
            i, loss, torch.max(ir_drop), total_drop_violation, tsv_count
        )
    )

    loss.backward()
    optimizer.step()

with torch.no_grad():
    q = torch.softmax(theta, dim=1)[:, 1]
    order = q.argsort(descending=True)

    # top-k
    # find the top-k tsv to make worst ir-drop < vdd * vdd_threshold
    lower = 0
    upper = order.size(0)
    while lower < upper:
        n = (lower + upper) // 2
        select_tsv = order[:n]
        g_value = torch.zeros_like(base_g_value)
        g_value[select_tsv] = base_g_value[select_tsv]

        i_voltage, v_current = OpAdjointFunction.apply(
            g_index,
            r_index,
            i_index,
            v_index,
            all_exc_index,
            g_value,
            r_value,
            all_exc_value,
            sim,
        )
        worst_ir_drop = torch.max(vdd - i_voltage)
        if worst_ir_drop.item() > vdd_threshold * vdd:
            lower = n + 1
        else:
            upper = n
    select_tsv = order[:upper]
    for target in select_tsv:
        result_tsv.append(candidate_tsv[target.item()])

# with torch.no_grad():
#     q = torch.softmax(theta, dim=1)[:, 1]
#     order = q.argsort(descending=True)

#     for n in range(1, order.size(0) + 1):
#         print("n = {}".format(n))
#         select_tsv = order[:n]
#         g_value = torch.zeros_like(base_g_value)
#         g_value[select_tsv] = base_g_value[select_tsv]

#         i_voltage, v_current = OpAdjointFunction.apply(
#             g_index,
#             r_index,
#             i_index,
#             v_index,
#             all_exc_index,
#             g_value,
#             r_value,
#             all_exc_value,
#             sim,
#         )
#         worst_ir_drop = torch.max(vdd - i_voltage)
#         if worst_ir_drop.item() < vdd_threshold * vdd:
#             break
#     select_tsv = order[:n]
#     for target in select_tsv:
#         result_tsv.append(candidate_tsv[target.item()])


# with torch.no_grad():
#     # 1. 计算最终的概率 q 和全局排序 order
#     q = torch.softmax(theta, dim=1)[:, 1]
#     order = q.argsort(descending=True)

#     # 2. 找出所有概率极度接近1的 "高置信度" TSV 作为基础方案
#     high_confidence_mask = torch.isclose(q, torch.tensor(1.0), atol=1e-8)
#     base_indices = torch.where(high_confidence_mask)[0]
    
#     print("-" * 40)
#     print(f"找到 {len(base_indices)} 个高置信度 (p≈1.0) 的TSV作为基础方案。")
#     print("-" * 40)

#     # 3. 创建基础方案的 g_value
#     g_value = torch.zeros_like(base_g_value)
#     g_value[base_indices] = base_g_value[base_indices]

#     # 4. 检查基础方案是否已满足要求
#     i_voltage, _ = OpAdjointFunction.apply(
#         g_index, r_index, i_index, v_index, all_exc_index,
#         g_value, r_value, all_exc_value, sim,
#     )
#     worst_ir_drop = torch.max(vdd - i_voltage)

#     if worst_ir_drop.item() < vdd_threshold * vdd:
#         print("恭喜！仅使用高置信度TSV已满足IR Drop约束。")
#         select_tsv = base_indices
#         print(f"最终Worst IR Drop: {worst_ir_drop.item():.6f}")

#     else:
#         print("高置信度TSV方案不满足约束，需要对剩余候选进行二分搜索...")
#         print(f"当前Worst IR Drop: {worst_ir_drop.item():.6f}, 目标: < {vdd_threshold * vdd:.6f}")
        
#         # 5. 准备 "剩余候选" 列表
#         base_indices_set = set(base_indices.tolist())
#         remaining_indices_sorted = [idx for idx in order.tolist() if idx not in base_indices_set]
        
#         # 6. 对剩余候选列表进行二分搜索
#         lower = 0
#         upper = len(remaining_indices_sorted)
#         best_n_from_remaining = upper # 初始化为选择所有剩余TSV

#         while lower < upper:
#             n = (lower + upper) // 2
            
#             # 从剩余候选中选择 top-n
#             additional_indices = remaining_indices_sorted[:n]
            
#             # 合并基础方案和附加方案
#             current_selection_indices = torch.cat(
#                 (base_indices, torch.tensor(additional_indices, dtype=torch.long))
#             )
            
#             # 构建 g_value (这里直接重建更简单)
#             g_value_search = torch.zeros_like(base_g_value)
#             g_value_search[current_selection_indices] = base_g_value[current_selection_indices]
            
#             # 运行仿真
#             i_voltage_search, _ = OpAdjointFunction.apply(
#                 g_index, r_index, i_index, v_index, all_exc_index,
#                 g_value_search, r_value, all_exc_value, sim,
#             )
#             worst_ir_drop_search = torch.max(vdd - i_voltage_search)
            
#             print(f"二分搜索: 尝试添加 {n} 个额外TSV... Worst Drop: {worst_ir_drop_search.item():.6f}")

#             if worst_ir_drop_search.item() > vdd_threshold * vdd:
#                 # 压降仍然太大，需要更多TSV
#                 lower = n + 1
#             else:
#                 # 满足条件，尝试用更少的TSV，并记录当前可行的解
#                 best_n_from_remaining = n
#                 upper = n
        
#         # 7. 合并最终方案
#         print(f"\n二分搜索完成。需要从剩余候选中添加 {best_n_from_remaining} 个TSV。")
#         final_additional_indices = torch.tensor(remaining_indices_sorted[:best_n_from_remaining], dtype=torch.long)
#         select_tsv = torch.cat((base_indices, final_additional_indices))
#     for target in select_tsv:
#         result_tsv.append(candidate_tsv[target.item()])

end = time.time()
print("time = {}".format(end - start))

# print(result_tsv)
print(len(result_tsv))
with open("tsv_history.pkl", "wb") as f:
    pickle.dump((loss_history, drop_loss_history, count_loss_history), f)

np.savetxt("tsv_distrib.csv", q.numpy())
pattern = R"tsv_(\d+)_(\d+)"
result["tsvs"] = []
for tsv in result_tsv:
    match_result = re.match(R"tsv_(\d+)_(\d+)", tsv)
    if match_result:
        x = int(match_result.group(1))
        y = int(match_result.group(2))
        result["tsvs"].append((x, y))
with open(file_result, "w") as f:
    yaml.dump(result, f)

loss_history = np.array(loss_history)
plt.plot(loss_history)
plt.xlabel("#iter")
plt.ylabel("loss")
plt.tight_layout()
plt.savefig("tmp.png", dpi=600)
