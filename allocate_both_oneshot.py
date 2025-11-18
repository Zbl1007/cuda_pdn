import yaml
import numpy as np
import torch
import torch.nn as nn
import re
import os
import time
import matplotlib.pyplot as plt
import pickle
import csv
from OpAdjoint import OpAdjointFunction
from OpSimulation import OpSimulationScipy, OpSimulationPardiso
from AcAdjoint import AcAdjointFunction
from AcSimulation import AcSimulationScipy, AcSimulationPardiso, AcSimulationCuDSS
from Circuit import Circuit, BranchType
from build_ckt import build_op_ckt, build_ac_ckt

from plt_impedance import plot_impedance_curve
    
# ---------------------------
# 从文件初始化 theta 的辅助函数
# ---------------------------
def initialize_theta_from_file(filepath, w, h):
    """
    从一个给定的 YAML 解决方案文件初始化 theta 张量。
    如果文件不存在，则回退到随机初始化。
    
    theta 张量的形状是 (w*h, 3)，其中每一列分别对应
    [空, tsv, dtc] 的 logits (原始概率值)。
    """
    num_locations = w * h
    if os.path.exists(filepath):
        print(f"信息: 在 '{filepath}' 找到初始解决方案。将从此文件初始化 theta。")
        with open(filepath, "r") as f:
            initial_solution = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        initial_theta = torch.zeros(num_locations, 3, dtype=torch.float32)
        
        # 使用一个较大的值来强烈建议某个选择
        # HIGH_LOGIT = 10.0
        # LOW_LOGIT = -10.0
        
        
        # 在给定初始解的情况下让概率不是确定的1和0，这样方便传递梯度
        HIGH_LOGIT = 2.0
        LOW_LOGIT = -2.0

        # 创建一个掩码来追踪哪些位置已经被分配
        assigned_mask = torch.zeros(num_locations, dtype=torch.bool)

        # 1. 为初始的 TSV 设置 logits
        if "tsvs" in initial_solution and initial_solution["tsvs"]:
            for x, y in initial_solution["tsvs"]:
                idx = x * h + y
                if 0 <= idx < num_locations:
                    initial_theta[idx, :] = torch.tensor([LOW_LOGIT, HIGH_LOGIT, LOW_LOGIT])
                    assigned_mask[idx] = True

        # 2. 为初始的 DTC 设置 logits
        if "dtcs" in initial_solution and initial_solution["dtcs"]:
            for x, y in initial_solution["dtcs"]:
                idx = x * h + y
                # 确保该位置未被分配为TSV
                if 0 <= idx < num_locations and not assigned_mask[idx]:
                    initial_theta[idx, :] = torch.tensor([LOW_LOGIT, LOW_LOGIT, HIGH_LOGIT])
                    assigned_mask[idx] = True
        
        # 3. 为所有未分配的位置显式地设置 "空" 的 logits
        #   这是最关键的修正步骤，确保了逻辑的清晰和正确性。
        unassigned_indices = ~assigned_mask
        initial_theta[unassigned_indices, :] = torch.tensor([HIGH_LOGIT, LOW_LOGIT, LOW_LOGIT])
        
        return initial_theta
    else:
        print(f"信息: 初始文件 '{filepath}' 未找到。将使用随机初始化。")
        # 回退到原始的随机初始化方法
        return torch.clamp(torch.randn(num_locations, 3), -1.0, 1.0)
    

# case = "micro150"

import sys
case = sys.argv[1]
file = "data/{}.yaml".format(case)
file_result = "data/2025_11/{}_both_result.yaml".format(case)
# file_initial = "data/2025_09/{}_result_dtc_dtconeshot_50.yaml".format(case)  # 初始解
with open(file, "r") as f:
    design = yaml.load(f.read(), Loader=yaml.FullLoader)

interposer_w = design["interposer"]["w"] * 5
interposer_h = design["interposer"]["h"] * 5
vdd = design["vdd"]
vdd_threshold = 0.02 * vdd
# target_impedance = 0.1 * vdd * vdd / design["chiplets"][0]["power"]
target_impedance = 99999
for chiplet in design["chiplets"]:
    target_impedance = min(target_impedance, 0.1 * vdd * vdd / chiplet["power"])
    # print(f"{target_impedance}")
print(f"min:{target_impedance}")
op_ckt = build_op_ckt(file)
ac_ckt = build_ac_ckt(file)


candidate_tsv = [
    "tsv_{}_{}".format(x, y) for x in range(interposer_w) for y in range(interposer_h)
]
result_tsv = []
candidate_dtc = [
    "dtc_{}_{}".format(x, y) for x in range(interposer_w) for y in range(interposer_h)
]
result_dtc = []

# --------------------------
# make op observe candidate
# --------------------------
op_observe_branch = [
    "i_{}_{}".format(x, y) for x in range(interposer_w) for y in range(interposer_h)
]
op_candidate_branch = [
    "g_{}_{}".format(x, y) for x in range(interposer_w) for y in range(interposer_h)
]

# --------------------------
# make ac observe candidate
# --------------------------
ac_observe_branch = ["id"]
ac_dtc_cap_branch = [
    "cd_{}_{}".format(x, y) for x in range(interposer_w) for y in range(interposer_h)
]
ac_tsv_con_branch = [
    "gt1_{}_{}".format(x, y) for x in range(interposer_w) for y in range(interposer_h)
] + ["gt2_{}_{}".format(x, y) for x in range(interposer_w) for y in range(interposer_h)]
ac_tsv_xind_branch = [
    "xlt1_{}_{}".format(x, y) for x in range(interposer_w) for y in range(interposer_h)
] + [
    "xlt2_{}_{}".format(x, y) for x in range(interposer_w) for y in range(interposer_h)
]
ac_tsv_cap_branch = [
    "ct_{}_{}".format(x, y) for x in range(interposer_w) for y in range(interposer_h)
]

# ---------------------------
# frequency_points
# ---------------------------
frequency_points = np.geomspace(0.1e9, 10e9, 100)

# ---------------------------
# op sim paramters
# ---------------------------
op_typ, op_u, op_v, op_val, op_index = op_ckt.prepare_sim(
    "0", op_observe_branch + op_candidate_branch
)
op_g_index = op_index[len(op_observe_branch) :]
op_r_index = torch.tensor([], dtype=torch.long)
op_i_index = op_index[: len(op_observe_branch)]
op_v_index = torch.tensor([], dtype=torch.long)
(op_all_exc_index,) = torch.where((op_typ == BranchType.I) | (op_typ == BranchType.V))
op_base_g_value = op_val[op_g_index]
op_r_value = torch.tensor([], dtype=torch.float64)
op_all_exc_value = op_val[op_all_exc_index]
op_sim = OpSimulationPardiso(op_typ, op_u, op_v, op_val)

# ---------------------------
# ac sim parameters
# ---------------------------
ac_typ, ac_u, ac_v, ac_val, ac_index = ac_ckt.prepare_sim(
    "0",
    ac_observe_branch
    + ac_dtc_cap_branch
    + ac_tsv_cap_branch
    + ac_tsv_con_branch
    + ac_tsv_xind_branch,
)
ac_observe_start = 0
ac_dtc_cap_start = len(ac_observe_branch)
ac_tsv_cap_start = ac_dtc_cap_start + len(ac_dtc_cap_branch)
ac_tsv_con_start = ac_tsv_cap_start + len(ac_tsv_cap_branch)
ac_tsv_xind_start = ac_tsv_con_start + len(ac_tsv_con_branch)
ac_g_index = ac_index[ac_tsv_con_start : ac_tsv_con_start + len(ac_tsv_con_branch)]
ac_r_index = torch.tensor([], dtype=torch.long)
ac_tsv_c_index = ac_index[ac_tsv_cap_start : ac_tsv_cap_start + len(ac_tsv_cap_branch)]
ac_dtc_c_index = ac_index[ac_dtc_cap_start : ac_dtc_cap_start + len(ac_dtc_cap_branch)]
ac_l_index = torch.tensor([], dtype=torch.long)
ac_xc_index = torch.tensor([], dtype=torch.long)
ac_xl_index = ac_index[ac_tsv_xind_start : ac_tsv_xind_start + len(ac_tsv_xind_branch)]
ac_i_index = ac_index[ac_observe_start : ac_observe_start + len(ac_observe_branch)]
ac_v_index = torch.tensor([], dtype=torch.long)
(ac_all_exc_index,) = torch.where((ac_typ == BranchType.I) | (ac_typ == BranchType.V))

ac_base_g_value = ac_val[ac_g_index]
ac_r_value = torch.tensor([], dtype=torch.float64)
ac_base_tsv_c_value = ac_val[ac_tsv_c_index]
ac_base_dtc_c_value = ac_val[ac_dtc_c_index]
ac_l_value = torch.tensor([], dtype=torch.float64)
ac_xc_value = torch.tensor([], dtype=torch.float64)
ac_base_xl_value = ac_val[ac_xl_index]
ac_all_exc_value = ac_val[ac_all_exc_index]

ac_sims = []
for freq in frequency_points:
    ac_sims.append(AcSimulationCuDSS(ac_typ, ac_u, ac_v, ac_val, freq))
    


# zs = []
# for freq, sim in zip(frequency_points, ac_sims):
#     i_voltage, v_current = AcAdjointFunction.apply(
#         ac_g_index,
#         ac_r_index,
#         torch.cat([ac_tsv_c_index, ac_dtc_c_index]),
#         ac_l_index,
#         ac_xc_index,
#         ac_xl_index,
#         ac_i_index,
#         ac_v_index,
#         ac_all_exc_index,
#         ac_base_g_value,
#         ac_r_value,
#         torch.cat([ac_base_tsv_c_value, ac_base_tsv_c_value]),# 放满
#         ac_l_value,
#         ac_xc_value,
#         ac_base_xl_value,
#         ac_all_exc_value,
#         freq,
#         sim,
#     )
#     zs.append(i_voltage)
# zs = torch.cat(zs)
# impedances = zs.abs()
# worst_impedance ,worst_impedance_index = torch.max(zs.abs(),  dim=0)
# impedance_violation = torch.nn.functional.relu(impedances - target_impedance)
# total_impedance_violation = impedance_violation.sum() / frequency_points.shape[0]


# # 创建数据保存路径
# output_file = '104impedance_data.csv'

# # 打开CSV文件，准备写入
# with open(output_file, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
    
#     # 写入文件的标题行
#     writer.writerow(['Frequency (Hz)', 'Impedance (Ohms)', 'Impedance Violation (Ohms)'])
    
#     # 写入频率点、阻抗值和违例阻抗
#     for freq, impedance, violation in zip(frequency_points, impedances, impedance_violation):
#         writer.writerow([freq.item(), impedance.item(), violation.item()])

# print(f"Impedance data saved to {output_file}")


# print(f"worst_impedance:{worst_impedance}")
# print(f"worst_impedance_index:{worst_impedance_index}")

# print(f"impedance_violation:{impedance_violation}")
# print(f"impedance_violation.sum:{impedance_violation.sum()}")

# print(f"total_impedance_violation:{total_impedance_violation}")



# zs_no = []
# tsv_c_value_no = torch.zeros_like(ac_base_tsv_c_value)
# ac_dtc_c_value_no = torch.zeros_like(ac_base_tsv_c_value)
# ac_g_value = torch.zeros_like(ac_base_g_value)
# ac_xl_value = torch.zeros_like(ac_base_xl_value)
# for freq, sim in zip(frequency_points, ac_sims):
#     i_voltage, v_current = AcAdjointFunction.apply(
#         ac_g_index,
#         ac_r_index,
#         torch.cat([ac_tsv_c_index, ac_dtc_c_index]),
#         ac_l_index,
#         ac_xc_index,
#         ac_xl_index,
#         ac_i_index,
#         ac_v_index,
#         ac_all_exc_index,
#         ac_g_value,
#         ac_r_value,
#         torch.cat([tsv_c_value_no, ac_dtc_c_value_no]),# 不放
#         ac_l_value,
#         ac_xc_value,
#         ac_xl_value,
#         ac_all_exc_value,
#         freq,
#         sim,
#     )
#     zs_no.append(i_voltage)
# zs_no = torch.cat(zs_no)
# impedances_no = zs_no.abs()
# worst_impedance_no ,worst_impedance_index_no = torch.max(zs_no.abs(),  dim=0)
# impedance_violation_no = torch.nn.functional.relu(impedances_no - target_impedance)
# total_impedance_violation_no = impedance_violation_no.sum() / frequency_points.shape[0]




# print(f"worst_impedance:{worst_impedance_no}")
# print(f"worst_impedance_index:{worst_impedance_index_no}")

# print(f"impedance_violation:{impedance_violation_no}")
# print(f"impedance_violation.sum:{impedance_violation_no.sum()}")

# print(f"total_impedance_violation:{total_impedance_violation_no}")



# plt.plot(frequency_points, impedances.detach().numpy(), label='Simulated Impedance')

# plt.plot(frequency_points, impedances_no.detach().numpy(), label='Simulated No DTC Impedance')

# # --- 绘图设置 ---
# plt.ylabel("Impedance[Ohm]")
# plt.yscale("log")
# plt.xlabel("Frequency[Hz]")
# plt.xscale("log")
# plt.grid(True, which="both", ls="--")

# # ==========================================================
# # --- 新增的代码在这里 ---
# # 定义您要标记的值
# target_value = target_impedance

# # 在图上 y=target_value 的位置画一条红色的虚线
# plt.axhline(y=target_value, color='r', linestyle='--', label=f'Reference Line')
# # ==========================================================

# # --- 新增：显示图例 (因为我们为两条线都添加了label) ---
# plt.legend()

# plt.tight_layout()
# plt.savefig(f"{case}2025_10_05_bothonshot.png")



# ---------------------------
# NN parameters
# ---------------------------
niters = 500
lr = 0.3
# lr = 0.05
temperature = 1
temperature_ratio = 0.9

# temperature_update_iteration = niters // 20
# total_drop_violation_coeff = 10
# total_impedance_violation_coeff = 50000
# 28, 9

# temperature_update_iteration = niters // 20
# total_drop_violation_coeff = 20
# total_impedance_violation_coeff = 50000
# 26, 9

# temperature_update_iteration = niters // 20
# total_drop_violation_coeff = 50
# total_impedance_violation_coeff = 50000
# 29, 9

# temperature_update_iteration = niters // 20
# total_drop_violation_coeff = 100
# total_impedance_violation_coeff = 50000
# 28, 9

# temperature_update_iteration = niters // 20
# total_drop_violation_coeff = 200
# total_impedance_violation_coeff = 50000
# 28, 8

# temperature_update_iteration = niters // 20
# total_drop_violation_coeff = 500
# total_impedance_violation_coeff = 50000
# 31, 8

# temperature_update_iteration = niters // 20
# total_drop_violation_coeff = 1000
# total_impedance_violation_coeff = 50000
# 33, 9

# temperature_update_iteration = niters // 20
# total_drop_violation_coeff = 2000
# total_impedance_violation_coeff = 50000
# 30, 8

# temperature_update_iteration = niters // 20
# total_drop_violation_coeff = 5000
# total_impedance_violation_coeff = 50000
# 42, 2

# temperature_update_iteration = niters // 20
# total_drop_violation_coeff = 10
# total_impedance_violation_coeff = 10000
# 28, 13

# temperature_update_iteration = niters // 20
# total_drop_violation_coeff = 20
# total_impedance_violation_coeff = 10000
# 30, 9

# temperature_update_iteration = niters // 20
# total_drop_violation_coeff = 50
# total_impedance_violation_coeff = 10000
# 29, 9

# temperature_update_iteration = niters // 20
# total_drop_violation_coeff = 100
# total_impedance_violation_coeff = 10000
# 25, 8

# temperature_update_iteration = niters // 20
# total_drop_violation_coeff = 200
# total_impedance_violation_coeff = 10000
# 28, 10

# temperature_update_iteration = niters // 20
# total_drop_violation_coeff = 500
# total_impedance_violation_coeff = 10000
# 27, 7

# temperature_update_iteration = niters // 20
# total_drop_violation_coeff = 1000
# total_impedance_violation_coeff = 10000
# 33, 6

# temperature_update_iteration = niters // 25
# total_drop_violation_coeff = 10
# total_impedance_violation_coeff = 50000
# 34, 9

# temperature_update_iteration = niters // 25
# total_drop_violation_coeff = 20
# total_impedance_violation_coeff = 50000
# 46, 13

# temperature_update_iteration = niters // 25
# total_drop_violation_coeff = 50
# total_impedance_violation_coeff = 50000
# 30, 11

# temperature_update_iteration = niters // 25
# total_drop_violation_coeff = 100
# total_impedance_violation_coeff = 50000
# 28, 9

# temperature_update_iteration = niters // 25
# total_drop_violation_coeff = 200
# total_impedance_violation_coeff = 50000
# 27, 10

# temperature_update_iteration = niters // 25
# total_drop_violation_coeff = 500
# total_impedance_violation_coeff = 50000
# 29, 9

# temperature_update_iteration = niters // 25
# total_drop_violation_coeff = 1000
# total_impedance_violation_coeff = 50000
# 30, 13

# temperature_update_iteration = niters // 25
# total_drop_violation_coeff = 2000
# total_impedance_violation_coeff = 50000
# 49, 10

# temperature_update_iteration = niters // 25
# total_drop_violation_coeff = 5000
# total_impedance_violation_coeff = 50000
# 32, 9

temperature_update_iteration = niters // 20
total_drop_violation_coeff = 1000
total_impedance_violation_coeff = 5000000
# 27, 4
# 32, 4
# 17, 1

# temperature_update_iteration = niters // 25
# total_drop_violation_coeff = 20
# total_impedance_violation_coeff = 10000
# 32, 14

# temperature_update_iteration = niters // 25
# total_drop_violation_coeff = 50
# total_impedance_violation_coeff = 10000
# 27, 10

# temperature_update_iteration = niters // 25
# total_drop_violation_coeff = 100
# total_impedance_violation_coeff = 10000
# 26, 7

# temperature_update_iteration = niters // 25
# total_drop_violation_coeff = 200
# total_impedance_violation_coeff = 10000
# 39, 9

# temperature_update_iteration = niters // 25
# total_drop_violation_coeff = 500
# total_impedance_violation_coeff = 10000
# 30,11

# temperature_update_iteration = niters // 25
# total_drop_violation_coeff = 1000
# total_impedance_violation_coeff = 10000
# 34, 9

# temperature_update_iteration = niters // 25
# total_drop_violation_coeff = 5000
# total_impedance_violation_coeff = 10000
# 34, 12

tsv_count_coeff = 0.12
dtc_count_coeff = 0.1

# ---------------------------
# Train NN
# ---------------------------
loss_history = []
ir_drop_loss_history = []
z_loss_history = []
tsv_count_loss_history = []
dtc_count_loss_history = []
torch.manual_seed(42)

theta = nn.Parameter(
    torch.clamp(torch.randn(interposer_w * interposer_h, 3), -1.0, 1.0)
)
# initial_theta_values = initialize_theta_from_file(
#     file_initial, interposer_w, interposer_h
# )
# theta = nn.Parameter(initial_theta_values)
optimizer = torch.optim.Adam(nn.ParameterList([theta]), lr=lr)
start = time.time()
# # --- 早停法 (Early Stopping) 参数 ---
# patience = 100      # 如果连续80次迭代loss都没有优化，则提前停止
# best_loss = float('inf')  # 初始化一个无穷大的最佳loss值
# patience_counter = 0      # 初始化耐心计数器

for i in range(niters):
    # update gumbel softmax temperature 不改变温度值，不需要动态缩小
    if i != 0 and i % temperature_update_iteration == 0:
        temperature *= temperature_ratio

    optimizer.zero_grad()
    p = torch.nn.functional.gumbel_softmax(theta, tau=temperature, dim=1)
    
        # ================================================================= #
    # =================== 在这里添加新代码来计算“硬”数量 ================== #
    # ================================================================= #
    with torch.no_grad():  # 我们不需要为这个计算梯度
        # 对于每个位置（每一行），找出概率最高的选择（0, 1, 或 2）
        hard_choices = torch.argmax(p, dim=1)
        
        # 统计选择为 TSV (1) 的位置数量
        hard_tsv_count = (hard_choices == 1).sum().item()
        
        # 统计选择为 DTC (2) 的位置数量
        hard_dtc_count = (hard_choices == 2).sum().item()
    # ================================================================= #
    # ========================= 新代码结束 ============================ #
    # ================================================================= #


    # update tsv
    tsv_prob = p[:, 1]
    op_g_value = op_base_g_value * tsv_prob
    op_i_voltage, op_v_current = OpAdjointFunction.apply(
        op_g_index,
        op_r_index,
        op_i_index,
        op_v_index,
        op_all_exc_index,
        op_g_value,
        op_r_value,
        op_all_exc_value,
        op_sim,
    )
    total_drop_violation = torch.nn.functional.relu(
        vdd - op_i_voltage - vdd_threshold
    ).sum()
    tsv_count = tsv_prob.sum()

    # update dtc
    dtc_prob = p[:, 2]
    ac_g_value = ac_base_g_value * tsv_prob.repeat(2)
    ac_tsv_c_value = ac_base_tsv_c_value * tsv_prob
    ac_dtc_c_value = ac_base_dtc_c_value * dtc_prob
    ac_xl_value = ac_base_xl_value * tsv_prob.repeat(2)
    ac_zs = []
    for freq, ac_sim in zip(frequency_points, ac_sims):
        i_voltage, v_current = AcAdjointFunction.apply(
            ac_g_index,
            ac_r_index,
            torch.cat([ac_tsv_c_index, ac_dtc_c_index]),
            ac_l_index,
            ac_xc_index,
            ac_xl_index,
            ac_i_index,
            ac_v_index,
            ac_all_exc_index,
            ac_g_value,
            ac_r_value,
            torch.cat([ac_tsv_c_value, ac_dtc_c_value]),
            ac_l_value,
            ac_xc_value,
            ac_xl_value,
            ac_all_exc_value,
            freq,
            ac_sim,
        )
        ac_zs.append(i_voltage)
    ac_zs = torch.cat(ac_zs)
    total_impedance_violation = (
        torch.nn.functional.relu(ac_zs.abs() - target_impedance).sum()
        / frequency_points.shape[0]
    )
    dtc_count = dtc_prob.sum()

    # loss
    loss = (
        total_drop_violation_coeff * total_drop_violation
        + total_impedance_violation_coeff * total_impedance_violation
        + tsv_count_coeff * tsv_count
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
    ir_drop_loss_history.append(total_drop_violation.detach().item())
    z_loss_history.append(total_impedance_violation.detach().item())
    tsv_count_loss_history.append(tsv_count.detach().item())
    dtc_count_loss_history.append(dtc_count.detach().item())
    # print(
    #     "iter {}: loss {:.5G} drop_violation {:.5G} impedance_violation {:.5G} tsv {:.5G} dtc {:.5G}".format(
    #         i,
    #         loss,
    #         total_drop_violation,
    #         total_impedance_violation,
    #         tsv_count,
    #         dtc_count,
    #     )
    # )
       # 修改 print 语句来同时显示“软”数量和“硬”数量
    print(
        "iter {}: loss {:.5G} drop_violation {:.5G} impedance_violation {:.5G} tsv(soft/hard) {:.5G}/{} dtc(soft/hard) {:.5G}/{}".format(
            i,
            loss,
            total_drop_violation,
            total_impedance_violation,
            tsv_count,       # 软数量
            hard_tsv_count,  # 硬数量
            dtc_count,       # 软数量
            hard_dtc_count,  # 硬数量
        )
    )
    loss.backward()
    optimizer.step()


with open("both_history.pkl", "wb") as f:
    pickle.dump(
        (
            loss_history,
            ir_drop_loss_history,
            z_loss_history,
            tsv_count_loss_history,
            dtc_count_loss_history,
        ),
        f,
    )

np.savetxt("both_distrib.csv", torch.softmax(theta, dim=1).detach().numpy())

# ---------------------------
# allocate tsvs
# ---------------------------

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
        op_g_value = torch.zeros_like(op_base_g_value)
        op_g_value[select_tsv] = op_base_g_value[select_tsv]

        i_voltage, v_current = OpAdjointFunction.apply(
            op_g_index,
            op_r_index,
            op_i_index,
            op_v_index,
            op_all_exc_index,
            op_g_value,
            op_r_value,
            op_all_exc_value,
            op_sim,
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
#     # 1. 计算TSV的概率q，并获取全局排序
#     q_tsv = torch.softmax(theta, dim=1)[:, 1]
#     order_tsv = q_tsv.argsort(descending=True)

#     # 2. 找出所有概率极度接近1的 "高置信度" TSV 作为基础方案
#     high_confidence_mask_tsv = torch.isclose(q_tsv, torch.tensor(1.0), atol=1e-8)
#     base_indices_tsv = torch.where(high_confidence_mask_tsv)[0]
    
#     print("-" * 40)
#     print(f"找到 {len(base_indices_tsv)} 个高置信度 (p≈1.0) 的TSV作为基础方案。")
#     print("-" * 40)

#     # 3. 检查基础方案的IR Drop
#     op_g_value = torch.zeros_like(op_base_g_value)
#     op_g_value[base_indices_tsv] = op_base_g_value[base_indices_tsv]
#     i_voltage, v_current = OpAdjointFunction.apply(
#         op_g_index, op_r_index, op_i_index, op_v_index,
#         op_all_exc_index, op_g_value, op_r_value, op_all_exc_value, op_sim,
#     )
#     worst_ir_drop = torch.max(vdd - i_voltage)

#     # 4. 判断基础方案是否已满足要求
#     if worst_ir_drop.item() < vdd_threshold:
#         print("恭喜！仅使用高置信度TSV已满足IR Drop约束。")
#         select_tsv = base_indices_tsv
#         print(f"最终Worst IR Drop: {worst_ir_drop.item():.6f}")
#     else:
#         print("高置信度TSV方案不满足约束，需要对剩余候选进行二分搜索...")
#         print(f"当前Worst IR Drop: {worst_ir_drop.item():.6f}, 目标: < {vdd_threshold:.6f}")

#         # 5. 准备 "剩余候选" 列表
#         base_indices_tsv_set = set(base_indices_tsv.tolist())
#         remaining_indices_tsv_sorted = [idx for idx in order_tsv.tolist() if idx not in base_indices_tsv_set]

#         # 6. 对剩余候选列表进行二分搜索
#         lower = 0
#         upper = len(remaining_indices_tsv_sorted)
#         best_n_from_remaining_tsv = upper

#         while lower < upper:
#             n = (lower + upper) // 2
            
#             additional_tsv_indices = remaining_indices_tsv_sorted[:n]
#             current_selection_indices = torch.cat(
#                 (base_indices_tsv, torch.tensor(additional_tsv_indices, dtype=torch.long))
#             )
            
#             op_g_value_search = torch.zeros_like(op_base_g_value)
#             op_g_value_search[current_selection_indices] = op_base_g_value[current_selection_indices]
            
#             i_voltage_search, _ = OpAdjointFunction.apply(
#                 op_g_index, op_r_index, op_i_index, op_v_index,
#                 op_all_exc_index, op_g_value_search, op_r_value, op_all_exc_value, op_sim,
#             )
#             worst_ir_drop_search = torch.max(vdd - i_voltage_search)
            
#             print(f"二分搜索: 尝试添加 {n} 个额外TSV... Worst Drop: {worst_ir_drop_search.item():.6f}")

#             if worst_ir_drop_search.item() > vdd_threshold:
#                 lower = n + 1
#             else:
#                 best_n_from_remaining_tsv = n
#                 upper = n

#         # 7. 合并最终方案
#         print(f"\n二分搜索完成。需要从剩余候选中添加 {best_n_from_remaining_tsv} 个TSV。")
#         final_additional_indices = torch.tensor(remaining_indices_tsv_sorted[:best_n_from_remaining_tsv], dtype=torch.long)
#         select_tsv = torch.cat((base_indices_tsv, final_additional_indices))

#     # 最终的 select_tsv 就是TSV的最终选择
#     for target in select_tsv:
#         result_tsv.append(candidate_tsv[target.item()])

print("num tsv:", len(result_tsv))

ac_g_value = torch.zeros_like(ac_base_g_value)
ac_g_value[select_tsv] = ac_base_g_value[select_tsv]
ac_g_value[select_tsv + interposer_w * interposer_h] = ac_base_g_value[
    select_tsv + interposer_w * interposer_h
]
ac_xl_value = torch.zeros_like(ac_base_xl_value)
ac_xl_value[select_tsv] = ac_base_xl_value[select_tsv]
ac_xl_value[select_tsv + interposer_w * interposer_h] = ac_base_xl_value[
    select_tsv + interposer_w * interposer_h
]
ac_tsv_c_value = torch.zeros_like(ac_base_tsv_c_value)
ac_tsv_c_value[select_tsv] = ac_base_tsv_c_value[select_tsv]

# with torch.no_grad():
#     # 1. 计算DTC的概率q，并获取完整排序
#     q_dtc = torch.softmax(theta, dim=1)[:, 2]
#     order_dtc_full = q_dtc.argsort(descending=True)

#     # 2. 【关键】预先过滤：从DTC候选列表里，排除所有已经被TSV占用的位置
#     order_dtc_valid = order_dtc_full[~torch.isin(order_dtc_full, select_tsv)]
#     print(f"总共有 {len(order_dtc_full)} 个DTC候选，排除掉TSV位置后剩余 {len(order_dtc_valid)} 个有效候选。")

#     # 3. 在有效的候选列表上，找出高置信度的DTC作为基础方案
#     q_dtc_valid = q_dtc[order_dtc_valid]
#     high_confidence_mask = torch.isclose(q_dtc_valid, torch.tensor(1.0), atol=1e-8)
#     base_dtc_indices_from_valid_list = torch.where(high_confidence_mask)[0]
#     base_dtc_indices = order_dtc_valid[base_dtc_indices_from_valid_list]
    
#     print("-" * 40)
#     print(f"找到 {len(base_dtc_indices)} 个高置信度 (p≈1.0) 的DTC作为基础方案。")
#     print("-" * 40)

#     # 4. 检查基础方案的阻抗
#     ac_dtc_c_value = torch.zeros_like(ac_base_dtc_c_value)
#     ac_dtc_c_value[base_dtc_indices] = ac_base_dtc_c_value[base_dtc_indices]
#     zs = []
#     for freq, ac_sim in zip(frequency_points, ac_sims):
#         i_voltage, v_current = AcAdjointFunction.apply(
#             ac_g_index, ac_r_index, torch.cat([ac_tsv_c_index, ac_dtc_c_index]),
#             ac_l_index, ac_xc_index, ac_xl_index, ac_i_index, ac_v_index,
#             ac_all_exc_index, ac_g_value, ac_r_value,
#             torch.cat([ac_tsv_c_value, ac_dtc_c_value]),
#             ac_l_value, ac_xc_value, ac_xl_value, ac_all_exc_value,
#             freq, ac_sim,
#         )
#         zs.append(i_voltage)
#     zs = torch.cat(zs)
#     worst_impedance = torch.max(zs.abs())

#     # 5. 判断基础方案是否已满足要求
#     if worst_impedance.item() < target_impedance:
#         print("恭喜！仅使用高置信度DTC已满足阻抗目标。")
#         select_dtc = base_dtc_indices
#         print(f"最终最差阻抗: {worst_impedance.item():.6f}")
#     else:
#         print("高置信度DTC方案不满足目标，需要对剩余候选进行二分搜索...")
#         print(f"当前最差阻抗: {worst_impedance.item():.6f}, 目标: < {target_impedance:.6f}")
        
#         remaining_indices_valid = order_dtc_valid[len(base_dtc_indices):]
        
#         lower = 0
#         upper = len(remaining_indices_valid)
#         best_n_from_remaining = upper
#         while lower < upper:
#             n = (lower + upper) // 2
#             additional_dtc_indices = remaining_indices_valid[:n]
#             current_selection_indices = torch.cat((base_dtc_indices, additional_dtc_indices))
            
#             ac_dtc_c_value_search = torch.zeros_like(ac_base_dtc_c_value)
#             ac_dtc_c_value_search[current_selection_indices] = ac_base_dtc_c_value[current_selection_indices]
#             zs_search = []
#             for freq, ac_sim in zip(frequency_points, ac_sims):
#                 i_voltage_search, _ = AcAdjointFunction.apply(
#                     ac_g_index, ac_r_index, torch.cat([ac_tsv_c_index, ac_dtc_c_index]),
#                     ac_l_index, ac_xc_index, ac_xl_index, ac_i_index, ac_v_index,
#                     ac_all_exc_index, ac_g_value, ac_r_value,
#                     torch.cat([ac_tsv_c_value, ac_dtc_c_value_search]),
#                     ac_l_value, ac_xc_value, ac_xl_value, ac_all_exc_value,
#                     freq, ac_sim,
#                 )
#                 zs_search.append(i_voltage_search)
#             zs_search = torch.cat(zs_search)
#             worst_impedance_search = torch.max(zs_search.abs())
            
#             print(f"二分搜索: 尝试添加 {n} 个额外DTC... 最差阻抗: {worst_impedance_search.item():.6f}")

#             if worst_impedance_search.item() > target_impedance:
#                 lower = n + 1
#             else:
#                 best_n_from_remaining = n
#                 upper = n
        
#         print(f"\n二分搜索完成。需要从剩余候选中添加 {best_n_from_remaining} 个DTC。")
#         final_additional_indices = remaining_indices_valid[:best_n_from_remaining]
#         select_dtc = torch.cat((base_dtc_indices, final_additional_indices))

#     for target in select_dtc:
#         result_dtc.append(candidate_dtc[target.item()])
        
with torch.no_grad():
    q = torch.softmax(theta, dim=1)[:, 2]
    order = q.argsort(descending=True)
    # top-k
    lower = 0
    upper = order.size(0)
    
    # 用于给图片文件命名的计数器
    plot_step_counter = 0 
    while lower < upper:
        n = (lower + upper) // 2
        select_dtc = order[:n]
        select_dtc = select_dtc[~torch.isin(select_dtc, select_tsv)]
        ac_dtc_c_value = torch.zeros_like(ac_base_dtc_c_value)
        ac_dtc_c_value[select_dtc] = ac_base_dtc_c_value[select_dtc]
        zs = []
        for freq, ac_sim in zip(frequency_points, ac_sims):
            i_voltage, v_current = AcAdjointFunction.apply(
                ac_g_index,
                ac_r_index,
                torch.cat([ac_tsv_c_index, ac_dtc_c_index]),
                ac_l_index,
                ac_xc_index,
                ac_xl_index,
                ac_i_index,
                ac_v_index,
                ac_all_exc_index,
                ac_g_value,
                ac_r_value,
                torch.cat([ac_tsv_c_value, ac_dtc_c_value]),
                ac_l_value,
                ac_xc_value,
                ac_xl_value,
                ac_all_exc_value,
                freq,
                ac_sim,
            )
            zs.append(i_voltage)
        zs = torch.cat(zs)
        # ==========================================================
        # --- 在这里调用我们封装的绘图函数 ---
        plot_step_counter += 1
        plot_impedance_curve(frequency_points, zs, n, target_impedance, plot_step_counter)
        # ==========================================================
        worst_impedance = torch.max(zs.abs())
        print(f"n:{n}")
        print(f"worst_impedance.item():{worst_impedance.item()}  ,  target_impedance:{target_impedance}")
        if worst_impedance.item() > target_impedance:
            lower = n + 1
        else:
            upper = n

    # # top-k
    # for n in range(1, 1 + order.size(0)):
    #     select_dtc = order[:n]
    #     select_dtc = select_dtc[~torch.isin(select_dtc, select_tsv)]
    #     ac_dtc_c_value = torch.zeros_like(ac_base_dtc_c_value)
    #     ac_dtc_c_value[select_dtc] = ac_base_dtc_c_value[select_dtc]
    #     zs = []
    #     for freq, ac_sim in zip(frequency_points, ac_sims):
    #         i_voltage, v_current = AcAdjointFunction.apply(
    #             ac_g_index,
    #             ac_r_index,
    #             torch.cat([ac_tsv_c_index, ac_dtc_c_index]),
    #             ac_l_index,
    #             ac_xc_index,
    #             ac_xl_index,
    #             ac_i_index,
    #             ac_v_index,
    #             ac_all_exc_index,
    #             ac_g_value,
    #             ac_r_value,
    #             torch.cat([ac_tsv_c_value, ac_dtc_c_value]),
    #             ac_l_value,
    #             ac_xc_value,
    #             ac_xl_value,
    #             ac_all_exc_value,
    #             freq,
    #             ac_sim,
    #         )
    #         zs.append(i_voltage)
    #     zs = torch.cat(zs)
    #     worst_impedance = torch.max(zs.abs())
    #     if worst_impedance.item() < target_impedance:
    #         break

    select_dtc = order[:n]
    select_dtc = select_dtc[~torch.isin(select_dtc, select_tsv)]
    for target in select_dtc:
        result_dtc.append(candidate_dtc[target.item()])

# with torch.no_grad():
#     # top-k
#     # q = torch.softmax(theta, tau=0.05, dim=1)[:, 1:]
#     q = torch.nn.functional.gumbel_softmax(theta, tau=0.05, dim=1)[:, 1:]
#     order, tsv_or_dtc = q.max(dim=1)
#     order = order.argsort()

#     # print(q[:, 1])
#     # print(q[:, 2])

#     lower = 0
#     upper = order.size(0)
#     op_sim.alter(op_all_exc_index, op_all_exc_value)
#     ac_sim = ac_sims[0]
#     ac_sim.alter(ac_all_exc_index, ac_all_exc_value)
#     while lower + 1 < upper:
#         n = (lower + upper) // 2
#         select = order[:n]
#         select_tsv = select[tsv_or_dtc[select] == 0]
#         select_dtc = select[tsv_or_dtc[select] == 1]

#         # check op
#         op_g_value = torch.zeros_like(op_base_g_value)
#         op_g_value[select_tsv] = op_base_g_value[select_tsv]
#         op_sim.alter(op_g_index, op_g_value)
#         op_sim.factorize()
#         op_sim.solve()
#         op_i_voltage = op_sim.branch_voltage(op_i_index)
#         worst_drop = (vdd - op_i_voltage).max().item()

#         # check ac
#         ac_g_value = torch.zeros_like(ac_base_g_value)
#         ac_g_value[select_tsv] = ac_base_g_value[select_tsv]
#         ac_g_value[interposer_w * interposer_h + select_tsv] = ac_base_g_value[
#             interposer_w * interposer_h + select_tsv
#         ]
#         ac_tsv_c_value = torch.zeros_like(ac_base_tsv_c_value)
#         ac_tsv_c_value[select_tsv] = ac_base_tsv_c_value[select_tsv]
#         ac_dtc_c_value = torch.zeros_like(ac_base_dtc_c_value)
#         ac_dtc_c_value[select_dtc] = ac_base_dtc_c_value[select_dtc]
#         ac_xl_value = torch.zeros_like(ac_base_xl_value)
#         ac_xl_value[select_tsv] = ac_base_xl_value[select_tsv]
#         ac_xl_value[interposer_w * interposer_h + select_tsv] = ac_base_xl_value[
#             interposer_w * interposer_h + select_tsv
#         ]
#         ac_zs = []
#         ac_sim.alter(ac_g_index, ac_g_value)
#         ac_sim.alter(ac_tsv_c_index, ac_tsv_c_value)
#         ac_sim.alter(ac_dtc_c_index, ac_dtc_c_value)
#         ac_sim.alter(ac_xl_index, ac_xl_value)
#         for freq in frequency_points:
#             ac_sim.set_freq(freq)
#             ac_sim.factorize()
#             ac_sim.solve()
#             ac_i_voltage = ac_sim.branch_voltage(ac_i_index)
#             ac_zs.append(ac_i_voltage.abs())
#         ac_zs = torch.cat(ac_zs)
#         worst_impedance = ac_zs.max()

#         if worst_drop > vdd_threshold or worst_impedance > target_impedance:
#             lower = n
#         else:
#             upper = n

#     select = order[:upper]
#     select_tsv = select[tsv_or_dtc[select] == 0]
#     select_dtc = select[tsv_or_dtc[select] == 1]
#     for target in select_tsv:
#         result_tsv.append(candidate_tsv[target.item()])
#     for target in select_dtc:
#         result_dtc.append(candidate_dtc[target.item()])

# print(result_tsv)
# print(result_dtc)
end = time.time()
print("time = {}".format(end - start))

print("num tsv:", len(result_tsv))
print("num dtc:", len(result_dtc))

# ------------------------
# write result to yaml
# ------------------------
result = dict()
pattern = R"tsv_(\d+)_(\d+)"
result["tsvs"] = []
for tsv in result_tsv:
    match_result = re.match(R"tsv_(\d+)_(\d+)", tsv)
    if match_result:
        x = int(match_result.group(1))
        y = int(match_result.group(2))
        result["tsvs"].append((x, y))
pattern = R"dtc_(\d+)_(\d+)"
result["dtcs"] = []
for dtc in result_dtc:
    match_result = re.match(R"dtc_(\d+)_(\d+)", dtc)
    if match_result:
        x = int(match_result.group(1))
        y = int(match_result.group(2))
        result["dtcs"].append((x, y))
with open(file_result, "w") as f:
    yaml.dump(result, f)

# ------------------------
# plot loss history
# ------------------------
loss_history = np.array(loss_history)
plt.plot(loss_history)
plt.xlabel("#iter")
plt.ylabel("loss")
plt.tight_layout()
plt.savefig("both_oneshot.png", dpi=600)
