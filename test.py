from AcSimulation import AcSimulationScipy, AcSimulationCuDSS, AcSimulationPardiso
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
# interposer_w = 100
# interposer_h = 100
# vdd = 1

# pg = PG(interposer_w, interposer_h)
# # with open("test_ref.sp", "w") as f:
# #     f.write(pg.__repr__())
# # exit()
# tsv = TSV()
# dtc = DTC()
# bump = MicroBump()
# ckt = pg.clone()

# # add tsv
# tsv_index = [(2, 4), (2, 9), (11, 11), (5, 10), (7, 7), (17, 10), (11, 6), (5, 0)]
# for x, y in tsv_index:
#     ckt.make_subcircuit(
#         tsv,
#         "tsv{}_{}".format(y + 1, x + 1),
#         [("c{}_{}".format(y + 1, x + 1), "top"), ("0", "bottom")],
#     )

# dtc_index = []
# for x, y in dtc_index:
#     ckt.make_subcircuit(
#         dtc,
#         "dtc{}_{}".format(y + 1, x + 1),
#         [("c{}_{}".format(y + 1, x + 1), "top"), ("0", "bottom")],
#     )

# chiplets = [(1, 3, 9, 12, 1.5, 3.8e9), (10, 3, 19, 12, 0.2, 2.4e9)]
# for k, (lx, ly, hx, hy, power, freq) in enumerate(chiplets):
#     for y in range(ly, hy):
#         for x in range(lx, hx):
#             ckt.make_subcircuit(
#                 bump,
#                 "bump{}_{}".format(y + 1, x + 1),
#                 [
#                     ("c{}_{}".format(y + 1, x + 1), "bottom"),
#                     ("die{}_sink".format(k), "top"),
#                 ],
#             )
#     C_ODC = 9 * power / vdd / vdd / freq
#     ESR = 0.25e-9 / C_ODC
#     ckt.make_branch(
#         "die{}_ESR".format(k),
#         BranchType.R,
#         "die{}_sink".format(k),
#         "die{}_mid".format(k),
#         ESR,
#     )
#     ckt.make_branch(
#         "die{}_ODC".format(k), BranchType.C, "die{}_mid".format(k), "0", C_ODC
#     )

# ckt.make_branch("die0_obs", BranchType.I, "die0_sink", "0", 1)
# typ, u, v, val, index = ckt.prepare_sim("0", ["die0_obs"])



# case1: 12 266.44s |
# micro150: 4 1944.01s | 7 394.68  pardiso 7 585.10
# multigpu: 2 5539.12s | 2 1445.90
# ascend910: 1 515.15s   pardiso 1 800+

case = "ascend910"
file = "data/{}.yaml".format(case)
file_result_tsv = "data/{}_result_tsv.yaml".format(case)
file_result_dtc = "data/{}_result_dtc.yaml".format(case)
with open(file, "r") as f:
    design = yaml.load(f.read(), Loader=yaml.FullLoader)
with open(file_result_tsv, "r") as f:
    result = yaml.load(f.read(), Loader=yaml.FullLoader)

interposer_w = design["interposer"]["w"] * 5
interposer_h = design["interposer"]["h"] * 5
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
# sims = []
# for freq in frequency_points:
#     sims.append(AcSimulationPardiso(typ, u, v, val, freq))

sims2 = []
for freq in frequency_points:
    sims2.append(AcSimulationCuDSS(typ, u, v, val, freq))


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
    start = time.time()
    # zs = []
    # for freq, sim in zip(frequency_points, sims):
    #     i_voltage, v_current = AcAdjointFunction.apply(
    #         g_index,
    #         r_index,
    #         c_index,
    #         l_index,
    #         xc_index,
    #         xl_index,
    #         i_index,
    #         v_index,
    #         all_exc_index,
    #         g_value,
    #         r_value,
    #         c_value,
    #         l_value,
    #         xc_value,
    #         xl_value,
    #         all_exc_value,
    #         freq,
    #         sim,
    #     )
    #     zs.append(i_voltage)
    # end = time.time()
    # print(f"pardiso: {end - start}")

    start = time.time()
    zs = []
    for freq, sim in zip(frequency_points, sims2):
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
    end = time.time()

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
        "cudss:{}  iter {}: loss {} total_impedance_violation {} dtc_count {}".format(
            end - start, i, loss, total_impedance_violation, dtc_count
        )
    )
    print()

    loss.backward()
    optimizer.step()

# sim = AcSimulationCuDSS(typ, u, v, val)
# freqs = np.geomspace(0.1e9, 10e9, 200)
# z = []
# start_time = time.time()
# for f in freqs:
#     sim.set_freq(f)
#     sim.factorize()
#     sim.solve()
#     z.append(abs(sim.branch_voltage(index)[0].item()))
# z = np.array(z)
# print(np.sum(z[z > 0.1] - 0.1))
# # print(np.sum(z))
# print("worst impedance:", np.max(z))
# time_cost = time.time() - start_time
# print("time cost:", time_cost)
# plt.plot(freqs, z)
# plt.ylabel("Impedance[Ohm]")
# plt.yscale("log")
# plt.yticks()
# plt.xlabel("Frequency[Hz]")
# plt.xscale("log")
# plt.xticks()
# plt.tight_layout()
# plt.savefig("tmp.png")


# ckt = Circuit()
# ckt.make_branch("l0", BranchType.L, "n2", "n1", 0.5e-9)
# ckt.make_branch("r0", BranchType.R, "n1", "0", 30e-3)
# ckt.make_branch("c0", BranchType.C, "n2", "0", 100e-9)
# ckt.make_branch("i0", BranchType.I, "0", "n2", 1)
# fps = np.geomspace(100e3, 100e6, 300, True)
# typ, u, v, val, index = ckt.prepare_sim("0", ["i0"])
# start = time.time()
# sim = AcSimulationScipy(typ, u, v, val)
# zpkg = []
# # print(index)
# # freq = 100e3
# # mat, rhs = sim.export()
# # print(mat.to_dense())
# # print(rhs)
# # print(typ)
# # print(u)
# # print(v)
# # print(val)
# for freq in fps:
#     sim.set_freq(freq)
#     sim.factorize()
#     sim.solve()
#     zpkg.append(abs(sim.branch_voltage(index)[0].item()))


# end = time.time()

# print(f"scipy: {end - start}")



# start = time.time()
# sim = AcSimulationPardiso(typ, u, v, val)
# zpkg = []
# # print(index)
# # freq = 100e3
# # mat, rhs = sim.export()
# # print(mat.to_dense())
# # print(rhs)
# # print(typ)
# # print(u)
# # print(v)
# # print(val)
# for freq in frequency_points:
#     # print(freq)
#     sim.set_freq(freq)
#     sim.factorize()
#     sim.solve()
#     zpkg.append(abs(sim.branch_voltage(index)[0].item()))


# end = time.time()
# plt.plot(frequency_points, zpkg)
# plt.ylabel("Impedance[Ohm]")
# plt.yscale("log")
# plt.yticks()
# plt.xlabel("Frequency[Hz]")
# plt.xscale("log")
# plt.xticks()
# plt.tight_layout()
# plt.savefig("tmp_pardiso.png")

# print(f"pardiso: {end - start}")


# start = time.time()
# sim = AcSimulationCuDSS(typ, u, v, val)
# zpkg = []
# # print(index)
# # freq = 100e3
# # mat, rhs = sim.export()
# # print(mat.to_dense())
# # print(rhs)
# # print(typ)
# # print(u)
# # print(v)
# # print(val)
# for freq in frequency_points:
#     sim.set_freq(freq)
#     # print(freq)
#     sim.factorize()
#     sim.solve()
#     zpkg.append(abs(sim.branch_voltage(index)[0].item()))


# end = time.time()
# plt.plot(frequency_points, zpkg)
# plt.ylabel("Impedance[Ohm]")
# plt.yscale("log")
# plt.yticks()
# plt.xlabel("Frequency[Hz]")
# plt.xscale("log")
# plt.xticks()
# plt.tight_layout()
# plt.savefig("tmp_cudss.png")
# print(f"CuDSS: {end - start}")
# print(zpkg)
# plt.plot(fps, zpkg)
# plt.ylabel("Impedance[Ohm]")
# plt.yscale("log")
# plt.yticks()
# plt.xlabel("Frequency[Hz]")
# plt.xscale("log")
# plt.xticks()
# plt.tight_layout()
# plt.savefig("tmp_cudss.png")
# print("done")
