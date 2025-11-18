from AcSimulation import AcSimulationScipy, AcSimulationCuDSS, AcSimulationPardiso
from OpSimulation import OpSimulationScipy, OpSimulationPardiso, OpSimulationCuDSS
from OpAdjoint import OpAdjointFunction
import os
import sys
import time
from Circuit import PG, TSV, MicroBump, BranchType, DTC, Circuit
import numpy as np
import matplotlib.pyplot as plt
import yaml
import numpy as np
import torch
import torch.nn as nn
import re
import matplotlib.pyplot as plt
import pickle
from AcAdjoint import AcAdjointFunction
from Circuit import Circuit, BranchType
from math import exp, log, sqrt, pi
from build_ckt import build_ac_ckt
interposer_w = 20
interposer_h = 15
vdd = 1

pg = PG(interposer_w, interposer_h)
# # # with open("test_ref.sp", "w") as f:
# # #     f.write(pg.__repr__())
# # # exit()
tsv = TSV()
dtc = DTC()
bump = MicroBump()
ckt = Circuit()
vdd_threshold = vdd * 0.02

# -------------------------
# construct PG
# -------------------------
s_uc = 100e-6
w_uc = 10e-6
t_uc = 0.7e-6
rho_uc = 1.7e-8
R_uc = (rho_uc * s_uc) / (t_uc * w_uc * 4)
# horizontal
for y in range(interposer_h):
    for x in range(interposer_w - 1):
        res_name = "r_{}_{}_{}_{}".format(x, y, x + 1, y)
        u_name = "c_{}_{}".format(x, y)
        v_name = "c_{}_{}".format(x + 1, y)
        ckt.make_branch(res_name, BranchType.R, u_name, v_name, 2 * R_uc)
# vertical
for y in range(interposer_h - 1):
    for x in range(interposer_w):
        res_name = "r_{}_{}_{}_{}".format(x, y, x, y + 1)
        u_name = "c_{}_{}".format(x, y)
        v_name = "c_{}_{}".format(x, y + 1)
        ckt.make_branch(res_name, BranchType.R, u_name, v_name, 2 * R_uc)


# ----------------
# construct load
# ----------------
load = np.zeros((interposer_w, interposer_h), dtype=np.float64)
chiplets = [
    {
        "x": 1,
        "y": 3,
        "w": 8,
        "h": 9,
        "power": 1.5,
        "frep": 5.0e+9
    },
    {
        "x": 10,
        "y": 3,
        "w": 9,
        "h": 9,
        "power": 0.2,
        "frep": 2.4e+9
    },
]

for chiplet in chiplets:
    i = chiplet["power"] / (chiplet["w"]) / (chiplet["h"]) / vdd
    for x in range(chiplet["x"], chiplet["x"] + chiplet["w"]):
        for y in range(chiplet["y"], chiplet["y"] + chiplet["h"]):
            load[x, y] += i
for y in range(interposer_h):
    for x in range(interposer_w):
        cur_name = "i_{}_{}".format(x, y)
        u_name = "c_{}_{}".format(x, y)
        ckt.make_branch(cur_name, BranchType.I, u_name, "0", load[x, y])

# -------------------------
# construct tsv
# -------------------------
d_tsv = 20e-6
h_tsv = 100e-6
rho_tsv = 1.7e-8
R_tsv = (rho_tsv * h_tsv) / (pi * (d_tsv / 2) ** 2)
G_tsv = 1 / R_tsv
# tsv location
tsv_locations = [(x, y) for x in range(interposer_w) for y in range(interposer_h)]

for x, y in tsv_locations:
    con_name = "g_{}_{}".format(x, y)
    u_name = "c_{}_{}".format(x, y)
    ckt.make_branch(con_name, BranchType.G, u_name, "vdd", G_tsv)


ckt.make_branch("vdd", BranchType.V, "vdd", "0", vdd)

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


niters = 200
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

tsv_count_coeff = 1
total_drop_violation_coeff = 10

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
    # 保存旧theta
    old_theta = theta.detach().clone()

    loss.backward()
    # 查看梯度信息
    print("θ.grad mean:", theta.grad.mean().item())
    print("θ.grad max :", theta.grad.max().item())
    print("θ.grad min :", theta.grad.min().item())

    zero_ratio = (theta.grad.abs() < 1e-12).float().mean().item()
    print("θ grad zero ratio:", zero_ratio)
    if torch.isnan(theta.grad).any():
        print("❌ NaN found in θ.grad!")
    if torch.isinf(theta.grad).any():
        print("❌ INF found in θ.grad!")
    
    if torch.isnan(theta.grad).any():
        print("❌ NaN found in θ.grad!")
    if torch.isinf(theta.grad).any():
        print("❌ INF found in θ.grad!")
    optimizer.step()
    theta_delta = (theta - old_theta).abs().mean().item()
    print("θ Δ:", theta_delta)

    if theta_delta < 1e-12:
        print("⚠️ WARNING: θ 没有更新（梯度太小或为 0？）")


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
print(len(result_tsv))
with open("tsv_history.pkl", "wb") as f:
    pickle.dump((loss_history, drop_loss_history, count_loss_history), f)

np.savetxt("tsv_distrib.csv", q.numpy())
pattern = R"tsv_(\d+)_(\d+)"
result = []
for tsv in result_tsv:
    match_result = re.match(R"tsv_(\d+)_(\d+)", tsv)
    if match_result:
        x = int(match_result.group(1))
        y = int(match_result.group(2))
        result.append((x, y))
with open("test.yaml", "w") as f:
    yaml.dump(result, f)

loss_history = np.array(loss_history)
plt.plot(loss_history)
plt.xlabel("#iter")
plt.ylabel("loss")
plt.tight_layout()
plt.savefig("tmp_test.png", dpi=600)
# # # add tsv
# # tsv_index = [(2, 4), (2, 9), (11, 11), (5, 10), (7, 7), (17, 10), (11, 6), (5, 0)]
# # for x, y in tsv_index:
# #     ckt.make_subcircuit(
# #         tsv,
# #         "tsv{}_{}".format(y + 1, x + 1),
# #         [("c{}_{}".format(y + 1, x + 1), "top"), ("0", "bottom")],
# #     )

# # dtc_index = []
# # for x, y in dtc_index:
# #     ckt.make_subcircuit(
# #         dtc,
# #         "dtc{}_{}".format(y + 1, x + 1),
# #         [("c{}_{}".format(y + 1, x + 1), "top"), ("0", "bottom")],
# #     )

# # chiplets = [(1, 3, 9, 12, 1.5, 3.8e9), (10, 3, 19, 12, 0.2, 2.4e9)]
# # for k, (lx, ly, hx, hy, power, freq) in enumerate(chiplets):
# #     for y in range(ly, hy):
# #         for x in range(lx, hx):
# #             ckt.make_subcircuit(
# #                 bump,
# #                 "bump{}_{}".format(y + 1, x + 1),
# #                 [
# #                     ("c{}_{}".format(y + 1, x + 1), "bottom"),
# #                     ("die{}_sink".format(k), "top"),
# #                 ],
# #             )
# #     C_ODC = 9 * power / vdd / vdd / freq
# #     ESR = 0.25e-9 / C_ODC
# #     ckt.make_branch(
# #         "die{}_ESR".format(k),
# #         BranchType.R,
# #         "die{}_sink".format(k),
# #         "die{}_mid".format(k),
# #         ESR,
# #     )
# #     ckt.make_branch(
# #         "die{}_ODC".format(k), BranchType.C, "die{}_mid".format(k), "0", C_ODC
# #     )

# # ckt.make_branch("die0_obs", BranchType.I, "die0_sink", "0", 1)
# # typ, u, v, val, index = ckt.prepare_sim("0", ["die0_obs"])



# # case1: 12 266.44s |
# # micro150: 4 1944.01s | 7 394.68  pardiso 7 585.10
# # multigpu: 2 5539.12s | 2 1445.90
# # ascend910: 1 515.15s   pardiso 1 800+

# case = "ascend910"
# file = "data/{}.yaml".format(case)
# file_result_tsv = "data/{}_result_tsv.yaml".format(case)
# file_result_dtc = "data/{}_result_dtc.yaml".format(case)
# with open(file, "r") as f:
#     design = yaml.load(f.read(), Loader=yaml.FullLoader)
# with open(file_result_tsv, "r") as f:
#     result = yaml.load(f.read(), Loader=yaml.FullLoader)

# interposer_w = design["interposer"]["w"] * 5
# interposer_h = design["interposer"]["h"] * 5
# vdd = design["vdd"]
# target_impedance = 0.1 * vdd * vdd / design["chiplets"][0]["power"]
# ckt = build_ac_ckt(file, file_result_tsv)

# # ---------------------------
# # make observe and candidate
# # ---------------------------
# observe_branch = ["id"]
# candidate_branch = [
#     "cd_{}_{}".format(x, y)
#     for x in range(interposer_w)
#     for y in range(interposer_h)
#     if (x, y) not in result["tsvs"]
# ]
# candidate_dtc = [
#     "dtc_{}_{}".format(x, y)
#     for x in range(interposer_w)
#     for y in range(interposer_h)
#     if (x, y) not in result["tsvs"]
# ]
# result_dtc = []

# # ---------------------------
# # frequency_points
# # ---------------------------
# frequency_points = np.geomspace(0.1e9, 10e9, 100)
# # ---------------------------
# # NN parameters
# # ---------------------------
# niters = 500
# lr = 0.3
# typ, u, v, val, index = ckt.prepare_sim("0", observe_branch + candidate_branch)
# g_index = torch.tensor([], dtype=torch.long)
# r_index = torch.tensor([], dtype=torch.long)
# c_index = index[len(observe_branch) :]
# l_index = torch.tensor([], dtype=torch.long)
# xc_index = torch.tensor([], dtype=torch.long)
# xl_index = torch.tensor([], dtype=torch.long)
# i_index = index[: len(observe_branch)]
# v_index = torch.tensor([], dtype=torch.long)
# (all_exc_index,) = torch.where((typ == BranchType.I) | (typ == BranchType.V))
# g_value = torch.tensor([], dtype=torch.float64)
# r_value = torch.tensor([], dtype=torch.float64)
# base_c_value = val[c_index]
# l_value = torch.tensor([], dtype=torch.float64)
# xc_value = torch.tensor([], dtype=torch.float64)
# xl_value = torch.tensor([], dtype=torch.float64)
# all_exc_value = val[all_exc_index]

# temperature = 1
# temperature_ratio = 0.9
# temperature_update_iteration = niters // 20
# total_impedance_violation_coeff = 50000
# dtc_count_coeff = 1

# # --------------------------------------
# # Create simulation per frequency point
# # --------------------------------------
# # sims = []
# # for freq in frequency_points:
# #     sims.append(AcSimulationPardiso(typ, u, v, val, freq))

# sims2 = []
# for freq in frequency_points:
#     sims2.append(AcSimulationCuDSS(typ, u, v, val, freq))


# # ---------------------------
# # Traint NN
# # ---------------------------
# start = time.time()
# loss_history = []
# z_loss_history = []
# count_loss_history = []
# torch.manual_seed(42)
# theta = nn.Parameter(torch.rand((c_index.size(0), 2)))
# optimizer = torch.optim.Adam(nn.ParameterList([theta]), lr=lr)
# for i in range(niters):
#     # update gumbel softmax temperature
#     if i != 0 and i % temperature_update_iteration == 0:
#         temperature *= temperature_ratio

#     optimizer.zero_grad()
#     p = torch.nn.functional.gumbel_softmax(theta, tau=temperature, dim=1)
#     # print(p)
#     c_value = base_c_value * p[:, 1]
#     start = time.time()
#     # zs = []
#     # for freq, sim in zip(frequency_points, sims):
#     #     i_voltage, v_current = AcAdjointFunction.apply(
#     #         g_index,
#     #         r_index,
#     #         c_index,
#     #         l_index,
#     #         xc_index,
#     #         xl_index,
#     #         i_index,
#     #         v_index,
#     #         all_exc_index,
#     #         g_value,
#     #         r_value,
#     #         c_value,
#     #         l_value,
#     #         xc_value,
#     #         xl_value,
#     #         all_exc_value,
#     #         freq,
#     #         sim,
#     #     )
#     #     zs.append(i_voltage)
#     # end = time.time()
#     # print(f"pardiso: {end - start}")

#     start = time.time()
#     zs = []
#     for freq, sim in zip(frequency_points, sims2):
#         i_voltage, v_current = AcAdjointFunction.apply(
#             g_index,
#             r_index,
#             c_index,
#             l_index,
#             xc_index,
#             xl_index,
#             i_index,
#             v_index,
#             all_exc_index,
#             g_value,
#             r_value,
#             c_value,
#             l_value,
#             xc_value,
#             xl_value,
#             all_exc_value,
#             freq,
#             sim,
#         )
#         zs.append(i_voltage)
#     end = time.time()

#     zs = torch.cat(zs)
#     impedances = zs.abs()
#     impedance_violation = torch.nn.functional.relu(impedances - target_impedance)
#     total_impedance_violation = impedance_violation.sum() / frequency_points.shape[0]
#     dtc_count = p[:, 1].sum()

#     loss = (
#         total_impedance_violation_coeff * total_impedance_violation
#         + dtc_count_coeff * dtc_count
#     )
#     loss_history.append(loss.detach().item())
#     z_loss_history.append(total_impedance_violation.detach().item())
#     count_loss_history.append(dtc_count.detach().item())
#     print(
#         "cudss:{}  iter {}: loss {} total_impedance_violation {} dtc_count {}".format(
#             end - start, i, loss, total_impedance_violation, dtc_count
#         )
#     )
#     print()

#     loss.backward()
#     optimizer.step()

# # sim = AcSimulationCuDSS(typ, u, v, val)
# # freqs = np.geomspace(0.1e9, 10e9, 200)
# # z = []
# # start_time = time.time()
# # for f in freqs:
# #     sim.set_freq(f)
# #     sim.factorize()
# #     sim.solve()
# #     z.append(abs(sim.branch_voltage(index)[0].item()))
# # z = np.array(z)
# # print(np.sum(z[z > 0.1] - 0.1))
# # # print(np.sum(z))
# # print("worst impedance:", np.max(z))
# # time_cost = time.time() - start_time
# # print("time cost:", time_cost)
# # plt.plot(freqs, z)
# # plt.ylabel("Impedance[Ohm]")
# # plt.yscale("log")
# # plt.yticks()
# # plt.xlabel("Frequency[Hz]")
# # plt.xscale("log")
# # plt.xticks()
# # plt.tight_layout()
# # plt.savefig("tmp.png")


# # ckt = Circuit()
# # ckt.make_branch("l0", BranchType.L, "n2", "n1", 0.5e-9)
# # ckt.make_branch("r0", BranchType.R, "n1", "0", 30e-3)
# # ckt.make_branch("c0", BranchType.C, "n2", "0", 100e-9)
# # ckt.make_branch("i0", BranchType.I, "0", "n2", 1)
# # fps = np.geomspace(100e3, 100e6, 300, True)
# # typ, u, v, val, index = ckt.prepare_sim("0", ["i0"])
# # start = time.time()
# # sim = AcSimulationScipy(typ, u, v, val)
# # zpkg = []
# # # print(index)
# # # freq = 100e3
# # # mat, rhs = sim.export()
# # # print(mat.to_dense())
# # # print(rhs)
# # # print(typ)
# # # print(u)
# # # print(v)
# # # print(val)
# # for freq in fps:
# #     sim.set_freq(freq)
# #     sim.factorize()
# #     sim.solve()
# #     zpkg.append(abs(sim.branch_voltage(index)[0].item()))


# # end = time.time()

# # print(f"scipy: {end - start}")



# # start = time.time()
# # sim = AcSimulationPardiso(typ, u, v, val)
# # zpkg = []
# # # print(index)
# # # freq = 100e3
# # # mat, rhs = sim.export()
# # # print(mat.to_dense())
# # # print(rhs)
# # # print(typ)
# # # print(u)
# # # print(v)
# # # print(val)
# # for freq in frequency_points:
# #     # print(freq)
# #     sim.set_freq(freq)
# #     sim.factorize()
# #     sim.solve()
# #     zpkg.append(abs(sim.branch_voltage(index)[0].item()))


# # end = time.time()
# # plt.plot(frequency_points, zpkg)
# # plt.ylabel("Impedance[Ohm]")
# # plt.yscale("log")
# # plt.yticks()
# # plt.xlabel("Frequency[Hz]")
# # plt.xscale("log")
# # plt.xticks()
# # plt.tight_layout()
# # plt.savefig("tmp_pardiso.png")

# # print(f"pardiso: {end - start}")


# # start = time.time()
# # sim = AcSimulationCuDSS(typ, u, v, val)
# # zpkg = []
# # # print(index)
# # # freq = 100e3
# # # mat, rhs = sim.export()
# # # print(mat.to_dense())
# # # print(rhs)
# # # print(typ)
# # # print(u)
# # # print(v)
# # # print(val)
# # for freq in frequency_points:
# #     sim.set_freq(freq)
# #     # print(freq)
# #     sim.factorize()
# #     sim.solve()
# #     zpkg.append(abs(sim.branch_voltage(index)[0].item()))


# # end = time.time()
# # plt.plot(frequency_points, zpkg)
# # plt.ylabel("Impedance[Ohm]")
# # plt.yscale("log")
# # plt.yticks()
# # plt.xlabel("Frequency[Hz]")
# # plt.xscale("log")
# # plt.xticks()
# # plt.tight_layout()
# # plt.savefig("tmp_cudss.png")
# # print(f"CuDSS: {end - start}")
# # print(zpkg)
# # plt.plot(fps, zpkg)
# # plt.ylabel("Impedance[Ohm]")
# # plt.yscale("log")
# # plt.yticks()
# # plt.xlabel("Frequency[Hz]")
# # plt.xscale("log")
# # plt.xticks()
# # plt.tight_layout()
# # plt.savefig("tmp_cudss.png")
# # print("done")



# from Circuit import PG_OP, TSV_OP, BranchType
# import torch
# import torch.nn as nn
# from OpSimulation import OpSimulationPardiso, OpSimulationCuDSS
# import random

# x = 10
# y = 10
# vdd = 1.8
# pg = PG_OP(x, y)
# tsv = TSV_OP()
# ckt = pg.clone()
# ckt.make_branch("vdd", BranchType.V, "vdd", "0", vdd)
# tsv_index = [
#     (11, 9),
#     (8, 9),
#     (5, 9),
#     (6, 8),
#     (12, 8),
#     (9, 8),
#     (6, 6),
#     (10, 7),
#     (11, 6),
#     (7, 5),
#     (8, 4),
#     (9, 3),
# ]
# tsv_index = random.sample([(i + 1, j + 1) for j in range(x) for i in range(y)], k=11)
# for ii, jj in tsv_index:
#     ckt.make_branch(
#         "tsv{}_{}".format(ii, jj),
#         BranchType.G,
#         "vdd",
#         "c{}_{}".format(ii, jj),
#         1 / 5.41e-3,
#     )

# chiplets = [(1, 3, 9, 12, 1.5, 3.8e9), (10, 3, 19, 12, 0.2, 2.4e9)]
# observe_load = []
# for lx, ly, hx, hy, power, freq in chiplets:
#     i_per_uc = power / (hx - lx) / (hy - ly) / vdd
#     for i in range(ly, hy):
#         for j in range(lx, hx):
#             ckt.make_branch(
#                 "load{}_{}".format(i + 1, j + 1),
#                 BranchType.I,
#                 "c{}_{}".format(i + 1, j + 1),
#                 "0",
#                 i_per_uc,
#             )
#             observe_load.append("load{}_{}".format(i + 1, j + 1))

# i_names = observe_load
# typ, u, v, val, index = ckt.prepare_sim("0", i_names)
# # sim = OpSimulationPardiso(typ, u, v, val)
# sim = OpSimulationCuDSS(typ, u, v, val)

# sim.factorize()
# sim.solve()
# i_voltage = sim.branch_voltage(index)
# ir_drop = vdd - i_voltage
# print(torch.sum(ir_drop))
# print(torch.max(ir_drop))
# print("drop = {:.5}% vdd".format((torch.max(ir_drop) / vdd).item() * 100))
