from AcSimulation import AcSimulationScipy, AcSimulationCuDSS, AcSimulationPardiso
import os
import sys
import time
# 将上级目录加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Circuit import PG, TSV, MicroBump, BranchType, DTC, Circuit
import numpy as np
import matplotlib.pyplot as plt

# interposer_w = 20
# interposer_h = 15
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


ckt = Circuit()
ckt.make_branch("l0", BranchType.L, "n2", "n1", 0.5e-9)
ckt.make_branch("r0", BranchType.R, "n1", "0", 30e-3)
ckt.make_branch("c0", BranchType.C, "n2", "0", 100e-9)
ckt.make_branch("i0", BranchType.I, "0", "n2", 1)
fps = np.geomspace(100e3, 100e6, 300, True)
typ, u, v, val, index = ckt.prepare_sim("0", ["i0"])
sim = AcSimulationPardiso(typ, u, v, val)
zpkg = []
print(index)
freq = 100e3
mat, rhs = sim.export()
print(mat.to_dense())
print(rhs)
print(typ)
print(u)
print(v)
print(val)
for freq in fps:
    sim.set_freq(freq)
    sim.factorize()
    sim.solve()
    zpkg.append(abs(sim.branch_voltage(index)[0].item()))

print(zpkg)
plt.plot(fps, zpkg)
plt.ylabel("Impedance[Ohm]")
plt.yscale("log")
plt.yticks()
plt.xlabel("Frequency[Hz]")
plt.xscale("log")
plt.xticks()
plt.tight_layout()
plt.savefig("tmp_pardiso.png")
print("done")
