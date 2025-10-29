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

case = "ascend910"
file = "data/{}.yaml".format(case)
file_result_tsv = "data/2025_11/{}_result_tsv_50.yaml".format(case)
file_result_dtc = "data/2025_11/{}_both_result_50.yaml".format(case)
with open(file, "r") as f:
    design = yaml.load(f.read(), Loader=yaml.FullLoader)
with open(file_result_tsv, "r") as f:
    result = yaml.load(f.read(), Loader=yaml.FullLoader)

interposer_w = design["interposer"]["w"] * 5
interposer_h = design["interposer"]["h"] * 5
vdd = design["vdd"]
# target_impedance = 0.1 * vdd * vdd / design["chiplets"][0]["power"]
target_impedance = 99999
for chiplet in design["chiplets"]:
    target_impedance = min(target_impedance, 0.1 * vdd * vdd / chiplet["power"])
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

tsv_locations = result["tsvs"]
# dtc_locations = result["dtcs"]

# for x, y in dtc_locations:
#     cap_name = "cd_{}_{}".format(x, y)
#     dtc_name = "dtc_{}_{}".format(x, y)
#     candidate_branch.append(cap_name)
#     candidate_dtc.append(dtc_name)



print(f"tsv_locations:{len(tsv_locations)}")
# print(f"dtc_locations:{len(dtc_locations)}")
print(f"candidate_branch:{len(candidate_branch)}")
print(f"candidate_dtc:{len(candidate_dtc)}")

# # exit()

# # print(candidate_branch)
# # print(candidate_dtc)

# # ---------------------------
# # frequency_points
# # ---------------------------
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
        base_c_value,
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
        c_value_no,
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


# ==============================================================================
# 新增代码部分：读取并解析LTspice的.txt文件
# ==============================================================================
# print("Parsing LTspice data file...")

# # 定义一个辅助函数，用于将dB值转换为线性值
# def db_to_linear(db_value):
#     return 10 ** (db_value / 20.0)

# # --- 文件解析逻辑 ---
# ltspice_data_file = f'data/2025_10/{case}_both.txt' # 请确保您的数据文件名是这个
# ltspice_data_file_nodtc = f'data/2025_10/{case}_both_nodtc.txt'
# lt_freqs = []
# lt_mags_linear = []
# lt_mags = []
# lt_phases = []
# lt_freqs_nodtc = []
# lt_mags_linear_nodtc = []
# lt_mags_nodtc = []
# lt_phases_nodtc = []
# # try:
# #     # 正则表达式，用于匹配每一行的数据
# #     # pattern = re.compile(r"^\s*([0-9.eE+-]+)\s+\(\s*([0-9.eE+-]+)dB\s*,\s*([0-9.eE+-]+)°.*$")
# #     pattern = re.compile(
# #     r"^\s*([0-9.eE+-]+)\s+"          # 频率
# #     r"\(\s*([0-9.eE+-]+)"           # 实部
# #     r"\s*\+\s*([0-9.eE+-]+)j\s*\)$" # 虚部
# # )
    
# #     with open(ltspice_data_file, 'r', encoding='latin-1') as f:
# #         next(f) # 跳过第一行标题
# #         for line in f:
# #             match = pattern.match(line)
# #             if match:
# #                 freq = float(match.group(1))
# #                 mag_db = float(match.group(2))
                
# #                 lt_freqs.append(freq)
# #                 # 直接转换为线性值并存储
# #                 lt_mags_linear.append(db_to_linear(mag_db))
    
# #     # 将列表转换为Numpy数组
# #     lt_freqs = np.array(lt_freqs)
# #     lt_mags_linear = np.array(lt_mags_linear)
# #     print(f"Successfully parsed {len(lt_freqs)} data points from {ltspice_data_file}")
# #     ltspice_data_found = True
# # except FileNotFoundError:
# #     print(f"WARNING: LTspice data file '{ltspice_data_file}' not found. Skipping LTspice plot.")
# #     ltspice_data_found = False
# # except Exception as e:
# #     print(f"An error occurred while parsing the LTspice file: {e}")
# #     ltspice_data_found = False

# try:
#     pattern = re.compile(
#         r"^\s*([0-9.eE+-]+)\s+"          # 频率
#         # r"\(\s*([0-9.eE+-]+)"           # 实部
#         r"\(\s*([+-]?[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)\s*"  # 实部，group(2)
#         r"([+-])\s*"                                    # 虚部符号 + 或 -，group(3)
#         r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)j\s*\)" # 虚部绝对值，group(4)
#         # r"\s*\+\s*([0-9.eE+-]+)j\s*\)$" # 虚部
#     )
#     with open(ltspice_data_file, 'r', encoding='latin-1') as f:
#         next(f)  # 跳过标题
#         for line in f:
#             match = pattern.match(line)
#             if match:
#                 freq = float(match.group(1))
#                 real = float(match.group(2))
#                 imag = float(match.group(4))

#                 mag = np.sqrt(real**2 + imag**2)
#                 phase = np.arctan2(imag, real)  # 弧度
#                 print(f"real:{real}  imag:{imag}  mag:{mag}")

#                 lt_freqs.append(freq)
#                 lt_mags.append(mag)
#                 lt_phases.append(phase)
#     with open(ltspice_data_file_nodtc, 'r', encoding='latin-1') as f:
#         next(f)  # 跳过标题
#         for line in f:
#             match = pattern.match(line)
#             if match:
#                 freq = float(match.group(1))
#                 real = float(match.group(2))
#                 imag = float(match.group(4))

#                 mag = np.sqrt(real**2 + imag**2)
#                 phase = np.arctan2(imag, real)  # 弧度
#                 print(f"real:{real}  imag:{imag}  mag:{mag}")

#                 lt_freqs_nodtc.append(freq)
#                 lt_mags_nodtc.append(mag)
#                 lt_phases_nodtc.append(phase)

#     lt_freqs = np.array(lt_freqs)
#     lt_mags = np.array(lt_mags)
#     lt_phases = np.array(lt_phases)
#     ltspice_data_found = True
    
#     print(f"Successfully parsed {len(lt_freqs)} data points from {ltspice_data_file}")
#     lt_freqs_nodtc = np.array(lt_freqs_nodtc)
#     lt_mags_nodtc = np.array(lt_mags_nodtc)
#     lt_phases_nodtc = np.array(lt_phases_nodtc)
#     ltspice_data_found_nodtc = True
#     print(f"Successfully parsed {len(lt_freqs_nodtc)} data points from {ltspice_data_file_nodtc}")


# except FileNotFoundError:
#     print(f"WARNING: LTspice data file '{ltspice_data_file}' not found. Skipping LTspice plot.")
#     ltspice_data_found = False
#     ltspice_data_found_nodtc = False
# except Exception as e:
#     print(f"An error occurred while parsing the LTspice file: {e}")
#     ltspice_data_found = False
#     ltspice_data_found_nodtc = False
    
# # ==============================================================================
# # 您的原始绘图代码部分 (已整合LTspice数据)
# # ==============================================================================
# print("Generating comparison plot...")

# 创建一个大一点的图，方便观察
plt.figure(figsize=(10, 6)) 

# 绘制您自己的 "有DTC" 仿真曲线
plt.plot(frequency_points, impedances.detach().numpy(), label='My Sim (With DTCs)', color='cyan', linewidth=2)

# 绘制您自己的 "无DTC" 仿真曲线
plt.plot(frequency_points, impedances_no.detach().numpy(), label='My Sim (No DTCs)', color='magenta', linestyle=':', linewidth=2)

# # --- 新增：如果成功读取了LTspice文件，就绘制它的曲线 ---
# if ltspice_data_found:
#     plt.plot(lt_freqs, lt_mags, label='Sim with DTC', color='red', linewidth=2)
# if ltspice_data_found_nodtc:
#     plt.plot(lt_freqs_nodtc, lt_mags_nodtc, label='Sim before DTC', color='green', linestyle='--', linewidth=2)
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.loglog(lt_freqs, lt_mags)

    # plt.subplot(2, 1, 2)
    # plt.semilogx(lt_freqs, np.degrees(lt_phases))

# --- 绘图设置 ---
plt.title("Impedance Comparison")
plt.ylabel("Impedance [Ohm]")
plt.yscale("log")
plt.xlabel("Frequency [Hz]")
plt.xscale("log")
plt.grid(True, which="both", ls="--")

# 定义参考线的值
target_value = target_impedance
plt.axhline(y=target_value, color='gray', linestyle='-.', label=f'Target Impedance')

# 显示图例
plt.legend()

plt.tight_layout()
plt.savefig(f"{case}result_both_comparison.png", dpi=300)

print(f"\nComparison plot saved as '{case}result_both_comparison.png'")



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
# plt.savefig("result_dtc.png")

# import re
# import numpy as np
# import matplotlib.pyplot as plt

# # ---------- 配置 ----------
# ltspice_data_file = 'data/2025_09/ltspice_output.txt'  # LTspice 输出文件路径

# # ---------- 数据存储 ----------
# lt_freqs = []
# lt_mags = []    # 幅值
# lt_phases = []  # 相位（度）

# # ---------- 正则匹配 AC 数据格式 ----------
# pattern = re.compile(
#     r"^\s*([0-9.eE+-]+)\s+"        # 频率
#     r"\(\s*([0-9.eE+-]+)"          # 实部
#     r"\s*\+\s*([0-9.eE+-]+)j\s*\)$"  # 虚部
# )

# # ---------- 解析文件 ----------
# try:
#     with open(ltspice_data_file, 'r', encoding='latin-1') as f:
#         next(f)  # 跳过标题
#         for line in f:
#             match = pattern.match(line)
#             if match:
#                 freq = float(match.group(1))
#                 real = float(match.group(2))
#                 imag = float(match.group(3))

#                 mag = np.sqrt(real**2 + imag**2)        # 幅值
#                 phase = np.arctan2(imag, real)          # 相位（弧度）

#                 lt_freqs.append(freq)
#                 lt_mags.append(mag)
#                 lt_phases.append(np.degrees(phase))     # 转换为度

#     lt_freqs = np.array(lt_freqs)
#     lt_mags = np.array(lt_mags)
#     lt_phases = np.array(lt_phases)

#     print(f"成功解析 {len(lt_freqs)} 个数据点")

#     ltspice_data_found = True

# except FileNotFoundError:
#     print(f"WARNING: 文件 '{ltspice_data_file}' 未找到。")
#     ltspice_data_found = False
# except Exception as e:
#     print(f"解析文件时出错: {e}")
#     ltspice_data_found = False


# # ---------- 绘制幅频与相频 ----------
# if ltspice_data_found:
#     plt.figure(figsize=(8, 6))

#     # 幅频
#     plt.subplot(2, 1, 1)
#     plt.loglog(lt_freqs, lt_mags, label="Magnitude")
#     plt.grid(True, which="both", ls="--")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude")
#     plt.title("LTspice AC Analysis - Magnitude")

#     # 相频
#     plt.subplot(2, 1, 2)
#     plt.semilogx(lt_freqs, lt_phases, label="Phase", color="orange")
#     plt.grid(True, which="both", ls="--")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Phase (degrees)")
#     plt.title("LTspice AC Analysis - Phase")

#     plt.tight_layout()
#     plt.savefig("Vdie0.png")
