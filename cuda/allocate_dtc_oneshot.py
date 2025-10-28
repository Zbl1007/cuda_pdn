import yaml
import numpy as np
import torch
import torch.nn as nn
import re
import matplotlib.pyplot as plt
import time
import pickle
from AcSimulation import AcSimulationCuDSS
import os
import sys
# 将上级目录加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from AcAdjoint import AcAdjointFunction
from Circuit import Circuit, BranchType
from math import exp, log, sqrt, pi
from build_ckt import build_ac_ckt

# case1: 12 266.44s |
# micro150: 4 1944.01s | 7 394.68  pardiso 7 585.10
# multigpu: 2 5539.12s | 2 1445.90
# ascend910: 1 515.15s   pardiso 1 800+

case = "ascend910"
file = "../data/{}.yaml".format(case)
file_result_tsv = "../data/{}_result_tsv.yaml".format(case)
file_result_dtc = "../data/{}_result_dtc.yaml".format(case)
with open(file, "r") as f:
    design = yaml.load(f.read(), Loader=yaml.FullLoader)
with open(file_result_tsv, "r") as f:
    result = yaml.load(f.read(), Loader=yaml.FullLoader)

interposer_w = design["interposer"]["w"] 
interposer_h = design["interposer"]["h"] 
vdd = design["vdd"]
target_impedance = 0.1 * vdd * vdd / design["chiplets"][0]["power"]
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

# ---------------------------
# frequency_points
# ---------------------------
frequency_points = np.geomspace(0.1e9, 10e9, 100)

# ---------------------------
# NN parameters
# ---------------------------
niters = 500
lr = 0.3
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
total_impedance_violation_coeff = 50000
dtc_count_coeff = 1

# --------------------------------------
# Create simulation per frequency point
# --------------------------------------
sims = []
for freq in frequency_points:
    sims.append(AcSimulationCuDSS(typ, u, v, val, freq))


# ---------------------------
# Traint NN
# ---------------------------
start = time.time()
loss_history = []
z_loss_history = []
count_loss_history = []
torch.manual_seed(42)
theta = nn.Parameter(torch.rand((c_index.size(0), 2)))
optimizer = torch.optim.Adam(nn.ParameterList([theta]), lr=lr)
for i in range(niters):
    # update gumbel softmax temperature
    if i != 0 and i % temperature_update_iteration == 0:
        temperature *= temperature_ratio

    optimizer.zero_grad()
    p = torch.nn.functional.gumbel_softmax(theta, tau=temperature, dim=1)
    # print(p)
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
    loss_history.append(loss.detach().item())
    z_loss_history.append(total_impedance_violation.detach().item())
    count_loss_history.append(dtc_count.detach().item())
    print(
        "iter {}: loss {} total_impedance_violation {} dtc_count {}".format(
            i, loss, total_impedance_violation, dtc_count
        )
    )

    loss.backward()
    optimizer.step()

with open("dtc_history.pkl", "wb") as f:
    pickle.dump((loss_history, z_loss_history, count_loss_history), f)

# with torch.no_grad():
#     q = torch.softmax(theta, dim=1)[:, 1]
#     order = q.argsort(descending=True)

#     # top-k
#     lower = 0
#     upper = order.size(0)
#     while lower < upper:
#         n = (lower + upper) // 2
#         select_dtc = order[:n]
#         c_value = torch.zeros_like(base_c_value)
#         c_value[select_dtc] = base_c_value[select_dtc]

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
#         worst_impedance = torch.max(zs.abs())
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
    for n in range(1, order.size(0) + 1):
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
        worst_impedance = torch.max(zs.abs())
        if worst_impedance.item() < target_impedance:
            break
    select_dtc = order[:n]
    for target in select_dtc:
        result_dtc.append(candidate_dtc[target.item()])

end = time.time()
print("time = {}".format(end - start))


np.savetxt("dtc_distrib.csv", q.numpy())

print(result_dtc)
print(len(result_dtc))
pattern = R"dtc_(\d+)_(\d+)"
result["dtcs"] = []
for tsv in result_dtc:
    match_result = re.match(R"dtc_(\d+)_(\d+)", tsv)
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




# 保存最终PDN阻抗数据
final_impedances = []
final_freqs = []
tsv_only_impedances = []  # 仅放置TSV的阻抗

# 首先计算只有TSV没有DTC的阻抗
c_value_tsv_only = torch.zeros_like(base_c_value)  # 全部DTC为0
for freq, sim in zip(frequency_points, sims):
    i_voltage, _ = AcAdjointFunction.apply(
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
        c_value_tsv_only,  # 无DTC
        l_value,
        xc_value,
        xl_value,
        all_exc_value,
        freq,
        sim,
    )
    if len(final_freqs) < len(frequency_points):
        final_freqs.append(freq)
    tsv_only_impedances.append(i_voltage.abs().item())

# 计算加入DTC后的最终阻抗
c_value = torch.zeros_like(base_c_value)
c_value[select_dtc] = base_c_value[select_dtc]

for freq, sim in zip(frequency_points, sims):
    i_voltage, _ = AcAdjointFunction.apply(
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
        c_value,  # 有DTC
        l_value,
        xc_value,
        xl_value,
        all_exc_value,
        freq,
        sim,
    )
    final_impedances.append(i_voltage.abs().item())

# 将数据保存为CSV文件
impedance_data = np.column_stack((final_freqs, tsv_only_impedances, final_impedances))
np.savetxt("./result/final_pdn_impedance.csv", impedance_data, delimiter=",", 
           header="Frequency(Hz),Impedance_TSV_Only(Ohm),Impedance_TSV_DTC(Ohm)", comments="")

# 获取最大阻抗值和对应频率点
tsv_only_max_impedance = max(tsv_only_impedances)
tsv_only_max_freq = final_freqs[tsv_only_impedances.index(tsv_only_max_impedance)]
final_max_impedance = max(final_impedances)
final_max_freq = final_freqs[final_impedances.index(final_max_impedance)]

# 绘制阻抗曲线并保存
plt.figure(figsize=(12, 7))
plt.loglog(final_freqs, tsv_only_impedances, 'b-', linewidth=2, label='TSV Only')
plt.loglog(final_freqs, final_impedances, 'g-', linewidth=2, label='TSV + DTC')
plt.axhline(y=target_impedance, color='r', linestyle='--', linewidth=2, 
            label=f'Target: {target_impedance:.6f} Ohm')

# 标记最大阻抗点
plt.plot(tsv_only_max_freq, tsv_only_max_impedance, 'bo', markersize=8)
plt.annotate(f'Max: {tsv_only_max_impedance:.6f} Ohm\n@{tsv_only_max_freq/1e9:.2f} GHz', 
             xy=(tsv_only_max_freq, tsv_only_max_impedance), xytext=(tsv_only_max_freq*1.2, tsv_only_max_impedance*1.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

plt.plot(final_max_freq, final_max_impedance, 'go', markersize=8)
plt.annotate(f'Max: {final_max_impedance:.6f} Ohm\n@{final_max_freq/1e9:.2f} GHz', 
             xy=(final_max_freq, final_max_impedance), xytext=(final_max_freq*0.8, final_max_impedance*1.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# 美化图表
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Impedance (Ohm)', fontsize=12)
plt.grid(True, which="both", ls="--")
plt.title(f'({case})PDN Impedance Profile Comparison\n(DTC count: {len(select_dtc)})', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"./result/{case}_pdn_impedance_comparison.png", dpi=600)

# 输出主要信息
print(f"Target impedance: {target_impedance:.6f} Ohm")
print(f"TSV only max impedance: {tsv_only_max_impedance:.6f} Ohm at {tsv_only_max_freq/1e9:.2f} GHz")
print(f"TSV+DTC max impedance: {final_max_impedance:.6f} Ohm at {final_max_freq/1e9:.2f} GHz")
print(f"Impedance reduction: {(tsv_only_max_impedance-final_max_impedance)/tsv_only_max_impedance*100:.2f}%")
