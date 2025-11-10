
import yaml
import numpy as np
import torch
import torch.nn as nn
import re
import matplotlib.pyplot as plt
import time
import pickle
from AcAdjoint import AcAdjointFunction
from AcSimulation import AcSimulationScipy, AcSimulationPardiso, AcSimulationCuDSS
from Circuit import Circuit, BranchType
from math import exp, log, sqrt, pi
from build_ckt import build_ac_ckt
import csv
import os

from plt_impedance import plot_impedance_curve

# case1: 12 266.44s |
# micro150: 4 1944.01s | 7 712.79
# multigpu: 2 5539.12s | 2 1445.90
# ascend910: 1 1171.41s

# ---------------------------
# 从文件初始化 theta 的辅助函数
# ---------------------------
def initialize_theta_from_file(filepath, candidate_dtc):
    """
    从 YAML 文件中读取初始 DTC 位置，
    在这些位置设置高概率（logits偏向 [0,1]）。
    其余位置随机初始化。
    """
    num_locations = len(candidate_dtc)
    initial_theta = torch.clamp(torch.randn(num_locations, 2), -1.0, 1.0)

    if not os.path.exists(filepath):
        print(f"信息: 初始文件 '{filepath}' 未找到，将使用随机初始化。")
        return initial_theta

    print(f"信息: 从 '{filepath}' 读取初始 DTC 位置。")
    with open(filepath, "r") as f:
        initial_solution = yaml.load(f.read(), Loader=yaml.FullLoader)

    # 设定 logits 水平
    HIGH_LOGIT = 2.0
    LOW_LOGIT = -2.0

    if "dtcs" in initial_solution and initial_solution["dtcs"]:
        # 遍历 candidate_dtc 列表，找到匹配的 DTC 坐标
        for idx, name in enumerate(candidate_dtc):
            match = re.match(r"dtc_(\d+)_(\d+)", name)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                if (x, y) in initial_solution["dtcs"]:
                    # 设置为高概率选中（logits第二列高）
                    initial_theta[idx, :] = torch.tensor([LOW_LOGIT, HIGH_LOGIT], dtype=torch.float32)

    return initial_theta

case = "ascend910"
file = "data/{}.yaml".format(case)
file_result_tsv = "data/2025_11/{}_result_tsv.yaml".format(case)
file_initial = "data/2025_11/{}_result_dtc_ga_OptimalNd433.yaml".format(case)
with open(file_initial, "r") as f:
        initial_solution = yaml.load(f.read(), Loader=yaml.FullLoader)
file_result_dtc = "data/2025_11/{}__result_dtc_50_new.yaml".format(case)
with open(file, "r") as f:
    design = yaml.load(f.read(), Loader=yaml.FullLoader)
with open(file_result_tsv, "r") as f:
    result = yaml.load(f.read(), Loader=yaml.FullLoader)

interposer_w = design["interposer"]["w"] * 5
interposer_h = design["interposer"]["h"] * 5
vdd = design["vdd"]
# target_impedance = 0.1 * vdd * vdd / design["chiplets"][0]["power"]
target_impedance = 9999999
# target_impedancesss = 0.1 * vdd * vdd / design["chiplets"][0]["power"]
# print(target_impedancesss)
for chiplet in design["chiplets"]:
    target_impedance = min(target_impedance, 0.1 * vdd * vdd / chiplet['power'])
    # print(f"{target_impedance}")
print(f"min:{target_impedance}")
ckt = build_ac_ckt(file, file_result_tsv)

# ---------------------------
# make observe and candidate
# ---------------------------
observe_branch = ["id"]
candidate_branch = [
    "cd_{}_{}".format(x, y)
    for x in range(interposer_w)
    for y in range(interposer_h)
    if (x, y) not in result["tsvs"]
]
candidate_dtc = [
    "dtc_{}_{}".format(x, y)
    for x in range(interposer_w)
    for y in range(interposer_h)
    if (x, y) not in result["tsvs"]
]
result_dtc = []
print(len(candidate_dtc))

# ---------------------------
# frequency_points
# ---------------------------
frequency_points = np.geomspace(0.1e9, 10e9, 100)

# ---------------------------
# NN parameters
# ---------------------------
niters = 500
lr = 0.02
typ, u, v, val, index = ckt.prepare_sim("0", observe_branch + candidate_branch)
g_index = torch.tensor([], dtype=torch.long)
r_index = torch.tensor([], dtype=torch.long)
c_index = index[len(observe_branch) :]
l_index = torch.tensor([], dtype=torch.long)
xc_index = torch.tensor([], dtype=torch.long)
xl_index = torch.tensor([], dtype=torch.long)
i_index = index[: len(observe_branch)]
v_index = torch.tensor([], dtype=torch.long)
(all_exc_index,) = torch.where((typ == BranchType.I) | (typ == BranchType.V))
g_value = torch.tensor([], dtype=torch.float64)
r_value = torch.tensor([], dtype=torch.float64)
base_c_value = val[c_index]
l_value = torch.tensor([], dtype=torch.float64)
xc_value = torch.tensor([], dtype=torch.float64)
xl_value = torch.tensor([], dtype=torch.float64)
all_exc_value = val[all_exc_index]

temperature = 1
temperature_ratio = 0.9
temperature_update_iteration = niters // 20
total_impedance_violation_coeff = 100000
dtc_count_coeff = 1


# --------------------------------------
# Create simulation per frequency point
# --------------------------------------
sims = []
for freq in frequency_points:
    sims.append(AcSimulationCuDSS(typ, u, v, val, freq))

zs = []
for freq, sim in zip(frequency_points, sims):
    i_voltage, v_current = AcAdjointFunction.apply(
        g_index,
        r_index,
        c_index,
        l_index,
        xc_index,
        xl_index,
        i_index,
        v_index,
        all_exc_index,
        g_value,
        r_value,
        base_c_value, # 放满
        l_value,
        xc_value,
        xl_value,
        all_exc_value,
        freq,
        sim,
    )
    zs.append(i_voltage)
zs = torch.cat(zs)
impedances = zs.abs()
worst_impedance ,worst_impedance_index = torch.max(zs.abs(),  dim=0)
impedance_violation = torch.nn.functional.relu(impedances - target_impedance)
total_impedance_violation = impedance_violation.sum() / frequency_points.shape[0]


# 创建数据保存路径
output_file = '104impedance_data.csv'

# 打开CSV文件，准备写入
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # 写入文件的标题行
    writer.writerow(['Frequency (Hz)', 'Impedance (Ohms)', 'Impedance Violation (Ohms)'])
    
    # 写入频率点、阻抗值和违例阻抗
    for freq, impedance, violation in zip(frequency_points, impedances, impedance_violation):
        writer.writerow([freq.item(), impedance.item(), violation.item()])

print(f"Impedance data saved to {output_file}")


print(f"worst_impedance:{worst_impedance}")
print(f"worst_impedance_index:{worst_impedance_index}")

print(f"impedance_violation:{impedance_violation}")
print(f"impedance_violation.sum:{impedance_violation.sum()}")

print(f"total_impedance_violation:{total_impedance_violation}")



zs_no = []
c_value_no = torch.zeros_like(base_c_value)
for freq, sim in zip(frequency_points, sims):
    i_voltage, v_current = AcAdjointFunction.apply(
        g_index,
        r_index,
        c_index,
        l_index,
        xc_index,
        xl_index,
        i_index,
        v_index,
        all_exc_index,
        g_value,
        r_value,
        c_value_no, # 不放
        l_value,
        xc_value,
        xl_value,
        all_exc_value,
        freq,
        sim,
    )
    zs_no.append(i_voltage)
zs_no = torch.cat(zs_no)
impedances_no = zs_no.abs()
worst_impedance_no ,worst_impedance_index_no = torch.max(zs_no.abs(),  dim=0)
impedance_violation_no = torch.nn.functional.relu(impedances_no - target_impedance)
total_impedance_violation_no = impedance_violation_no.sum() / frequency_points.shape[0]




print(f"worst_impedance:{worst_impedance_no}")
print(f"worst_impedance_index:{worst_impedance_index_no}")

print(f"impedance_violation:{impedance_violation_no}")
print(f"impedance_violation.sum:{impedance_violation_no.sum()}")

print(f"total_impedance_violation:{total_impedance_violation_no}")



plt.plot(frequency_points, impedances.detach().numpy(), label='Simulated Impedance')

plt.plot(frequency_points, impedances_no.detach().numpy(), label='Simulated No DTC Impedance')

# --- 绘图设置 ---
plt.ylabel("Impedance[Ohm]")
plt.yscale("log")
plt.xlabel("Frequency[Hz]")
plt.xscale("log")
plt.grid(True, which="both", ls="--")

# ==========================================================
# --- 新增的代码在这里 ---
# 定义您要标记的值
target_value = target_impedance

# 在图上 y=target_value 的位置画一条红色的虚线
plt.axhline(y=target_value, color='r', linestyle='--', label=f'Reference Line')
# ==========================================================

# --- 新增：显示图例 (因为我们为两条线都添加了label) ---
plt.legend()

plt.tight_layout()
plt.savefig("2025_10_29.png")


# ---------------------------
# Traint NN
# ---------------------------
start = time.time()
loss_history = []
z_loss_history = []
count_loss_history = []
torch.manual_seed(42)
# theta = nn.Parameter(torch.rand((c_index.size(0), 2)))

initial_theta_values = initialize_theta_from_file(file_initial, candidate_dtc)
theta = nn.Parameter(initial_theta_values)


optimizer = torch.optim.Adam(nn.ParameterList([theta]), lr=lr)
# # --- 早停法 (Early Stopping) 参数 ---
# patience = 50      # 如果连续50次迭代loss都没有优化，则提前停止
# best_loss = float('inf')  # 初始化一个无穷大的最佳loss值
# patience_counter = 0      # 初始化耐心计数器
# ======== 生成冻结掩码（frozen_mask）========
frozen_mask = torch.zeros(len(candidate_dtc), dtype=torch.bool)
if initial_solution and "dtcs" in initial_solution and initial_solution["dtcs"]:
    for idx, name in enumerate(candidate_dtc):
        match = re.match(r"dtc_(\d+)_(\d+)", name)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            if (x, y) in initial_solution["dtcs"]:
                frozen_mask[idx] = True
print(f"冻结 DTC 数量: {frozen_mask.sum().item()} / {len(frozen_mask)}")
freeze_steps = 50  # 冻结前50轮（可调大/小）

# (可选) 用于保存效果最好的theta
best_theta = None
with torch.no_grad():
    probs = torch.softmax(theta, dim=1)[:, 1]
    print("前10个DTC概率：", probs[:10])
    print("最大概率:", probs.max().item(), "最小概率:", probs.min().item())

for i in range(niters):
    # update gumbel softmax temperature
    if i != 0 and i % temperature_update_iteration == 0:
        temperature *= temperature_ratio

    optimizer.zero_grad()
    p = torch.nn.functional.gumbel_softmax(theta, tau=temperature, dim=1)
    c_value = base_c_value * p[:, 1]
    zs = []
    for freq, sim in zip(frequency_points, sims):
        i_voltage, v_current = AcAdjointFunction.apply(
            g_index,
            r_index,
            c_index,
            l_index,
            xc_index,
            xl_index,
            i_index,
            v_index,
            all_exc_index,
            g_value,
            r_value,
            c_value,
            l_value,
            xc_value,
            xl_value,
            all_exc_value,
            freq,
            sim,
        )
        zs.append(i_voltage)
    zs = torch.cat(zs)
    impedances = zs.abs()
    impedance_violation = torch.nn.functional.relu(impedances - target_impedance)
    total_impedance_violation = impedance_violation.sum() / frequency_points.shape[0]
    dtc_count = p[:, 1].sum()

    loss = (
        total_impedance_violation_coeff * total_impedance_violation
        + dtc_count_coeff * dtc_count
    )
    # # --- 早停法判断逻辑 ---
    # # 我们使用 loss.item() 来获取loss的纯数值进行比较
    # if loss.item() < best_loss:
    #     # 如果当前loss比记录的最好loss还要低，说明有进步
    #     best_loss = loss.item()
    #     patience_counter = 0  # 重置耐心计数器
    #     # (可选) 保存当前效果最好的 theta
    #     best_theta = theta.clone().detach()
    # else:
    #     # 如果loss没有变得更好，增加耐心计数器
    #     patience_counter += 1

    # # 如果耐心耗尽，则打印信息并跳出循环
    # if patience_counter >= patience:
    #     print(f"\nEarly stopping triggered after {i+1} iterations as loss did not improve for {patience} steps.")
    #     break
    # # -----------------------
    
    loss_history.append(loss.detach().item())
    z_loss_history.append(total_impedance_violation.detach().item())
    count_loss_history.append(dtc_count.detach().item())
    print(
        "iter {}: loss {} total_impedance_violation {} dtc_count {}".format(
            i, loss, total_impedance_violation, dtc_count
        )
    )

    loss.backward()
    # ==========================================================
    # ======== 冻结阶段：阻止初始 DTC logits 被修改 ========
    # ==========================================================
    if i < freeze_steps:
        with torch.no_grad():
            # 手动将梯度置0，以防这些参数被Adam更新
            theta.grad[frozen_mask, :] = 0.0
            # 同时保证这些 logits 保持高低状态 [-2, 2]
            theta[frozen_mask, 0] = -2.0
            theta[frozen_mask, 1] = 2.0
        if i == 0:
            print(f"前 {freeze_steps} 轮冻结 {frozen_mask.sum().item()} 个 DTC 的 logits。")

    optimizer.step()

with open("dtc_history.pkl", "wb") as f:
    pickle.dump((loss_history, z_loss_history, count_loss_history), f)

# with torch.no_grad():
#     q = torch.softmax(theta, dim=1)[:, 1]
#     order = q.argsort(descending=True)

#     # top-k
#     lower = 0
#     upper = order.size(0)
    
#     # 用于给图片文件命名的计数器
#     plot_step_counter = 0 
    
#     while lower < upper:
#         n = (lower + upper) // 2
#         select_dtc = order[:n]
#         c_value = torch.zeros_like(base_c_value)
#         c_value[select_dtc] = base_c_value[select_dtc]
#         print(f"n:{n}")

#         zs = []
#         for freq, sim in zip(frequency_points, sims):
#             i_voltage, v_current = AcAdjointFunction.apply(
#                 g_index,
#                 r_index,
#                 c_index,
#                 l_index,
#                 xc_index,
#                 xl_index,
#                 i_index,
#                 v_index,
#                 all_exc_index,
#                 g_value,
#                 r_value,
#                 c_value,
#                 l_value,
#                 xc_value,
#                 xl_value,
#                 all_exc_value,
#                 freq,
#                 sim,
#             )
#             zs.append(i_voltage)
#         zs = torch.cat(zs)
#         # ==========================================================
#         # --- 在这里调用我们封装的绘图函数 ---
#         plot_step_counter += 1
#         plot_impedance_curve(frequency_points, zs, n, target_impedance, plot_step_counter)
#         # ==========================================================

#         worst_impedance = torch.max(zs.abs())
#         print(f"worst_impedance.item():{worst_impedance.item()}  ,  target_impedance:{target_impedance}")
#         if worst_impedance.item() > target_impedance:
#             lower = n + 1
#         else:
#             upper = n

#     select_dtc = order[:upper]
#     for target in select_dtc:
#         result_dtc.append(candidate_dtc[target.item()])
        
with torch.no_grad():
    q = torch.softmax(theta, dim=1)[:, 1]
    order = q.argsort(descending=True)
    for n in range(1, order.size(0) + 1,100):
        select_dtc = order[:n]
        c_value = torch.zeros_like(base_c_value)
        c_value[select_dtc] = base_c_value[select_dtc]
        zs = []
        for freq, sim in zip(frequency_points, sims):
            i_voltage, v_current = AcAdjointFunction.apply(
                g_index,
                r_index,
                c_index,
                l_index,
                xc_index,
                xl_index,
                i_index,
                v_index,
                all_exc_index,
                g_value,
                r_value,
                c_value,
                l_value,
                xc_value,
                xl_value,
                all_exc_value,
                freq,
                sim,
            )
            zs.append(i_voltage)
        zs = torch.cat(zs)
        # --- 在这里调用我们封装的绘图函数 ---
        plot_impedance_curve(frequency_points, zs, n, target_impedance, n)
        print(f"n:{n}  worst_impedance.item():{worst_impedance.item()}  ,  target_impedance:{target_impedance}")
        # ==========================================================
        worst_impedance = torch.max(zs.abs())
        if worst_impedance.item() < target_impedance:
            break
    select_dtc = order[:n]
    for target in select_dtc:
        result_dtc.append(candidate_dtc[target.item()])


end = time.time()
print("time = {}".format(end - start))


np.savetxt("dtc_distrib.csv", q.numpy())

# print(result_dtc)
print(len(result_dtc))
pattern = R"dtc_(\d+)_(\d+)"
result["dtcs"] = []
for dtc in result_dtc:
    match_result = re.match(R"dtc_(\d+)_(\d+)", dtc)
    if match_result:
        x = int(match_result.group(1))
        y = int(match_result.group(2))
        result["dtcs"].append((x, y))
with open(file_result_dtc, "w") as f:
    yaml.dump(result, f)

loss_history = np.array(loss_history)
plt.plot(loss_history)
plt.xlabel("#iter")
plt.ylabel("loss")
plt.tight_layout()
plt.savefig("tmp.png", dpi=600)
