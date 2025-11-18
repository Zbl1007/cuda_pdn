# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from AcSimulation import AcSimulationScipy, AcSimulationPardiso, AcSimulationCuDSS
# from AcAdjoint import AcAdjointFunction
# from Circuit import Circuit, BranchType

# R = 30e-3
# L = 0.5e-9
# C = 100e-9

# ckt = Circuit()
# ckt.make_branch("r0", BranchType.R, "n1", "0", R)
# ckt.make_branch("l0", BranchType.L, "n2", "n1", L)
# ckt.make_branch("c0", BranchType.C, "n2", "0", C)
# ckt.make_branch("i0", BranchType.I, "n2", "0", 1)

# fps = np.geomspace(100e3, 100e6, 300, True)
# typ, u, v, val, index = ckt.prepare_sim("0", ["i0", "r0", "l0", "c0"])
# freq = 100e6

# sim = AcSimulationPardiso(typ, u, v, val, freq)


# # test grad R
# sim.factorize()
# sim.solve()
# z1 = sim.branch_voltage(index[0:1]).abs()
# dR = 0.001 * R
# sim.alter(index[1:2], torch.tensor([R + dR], dtype=torch.complex128))
# sim.factorize()
# sim.solve()
# z2 = sim.branch_voltage(index[0:1]).abs()
# sim.alter(index[1:2], torch.tensor([R], dtype=torch.complex128))
# print("simple grad R:", (z2 - z1) / dR)

# # test grad L
# sim.factorize()
# sim.solve()
# z1 = sim.branch_voltage(index[0:1]).abs()
# dL = 0.001 * L
# sim.alter(index[2:3], torch.tensor([L + dL], dtype=torch.complex128))
# sim.factorize()
# sim.solve()
# z2 = sim.branch_voltage(index[0:1]).abs()
# sim.alter(index[2:3], torch.tensor([L], dtype=torch.complex128))
# print("simple grad L:", (z2 - z1) / dL)

# # test grad L
# sim.factorize()
# sim.solve()
# z1 = sim.branch_voltage(index[0:1]).abs()
# dC = 0.001 * C
# sim.alter(index[3:4], torch.tensor([C + dC], dtype=torch.complex128))
# sim.factorize()
# sim.solve()
# z2 = sim.branch_voltage(index[0:1]).abs()
# sim.alter(index[3:4], torch.tensor([C], dtype=torch.complex128))
# print("simple grad C:", (z2 - z1) / dC)

# r_index = index[1:2]
# r_value = val[r_index].requires_grad_(True)
# g_index = torch.zeros(0, dtype=torch.long)
# g_value = torch.zeros(0, dtype=torch.complex128)
# l_index = index[2:3]
# l_value = val[l_index].requires_grad_(True)
# c_index = index[3:4]
# c_value = val[c_index].requires_grad_(True)
# i_index = index[0:1]
# v_index = torch.zeros(0, dtype=torch.long)
# all_exc_index = index[0:1]
# all_exc_value = val[all_exc_index]
# xc_index = torch.tensor([], dtype=torch.long)
# xl_index = torch.tensor([], dtype=torch.long)
# xc_value = torch.tensor([], dtype=torch.float64)
# xl_value = torch.tensor([], dtype=torch.float64)
# i_voltage, v_current = AcAdjointFunction.apply(
#     g_index,
#     r_index,
#     c_index,
#     l_index,
#     xc_index,
#     xl_index,
#     i_index,
#     v_index,
#     all_exc_index,
#     g_value,
#     r_value,
#     c_value,
#     l_value,
#     xc_value,
#     xl_value,
#     all_exc_value,
#     freq,
#     sim,
# )
# loss = i_voltage.abs()
# loss.backward()
# print("adjoint grad R: ", r_value.grad)
# print("adjoint grad L: ", l_value.grad)
# print("adjoint grad C: ", c_value.grad)




# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from AcSimulation import AcSimulationScipy, AcSimulationPardiso
# from AcAdjoint import AcAdjointFunction
# from Circuit import Circuit, BranchType

# R = 30e-3
# L = 0.5e-9
# C = 100e-9
# delta1 = 0.1
# delta2 = 0.001
# dR1 = delta1 * R
# dL1 = delta1 * L
# dC1 = delta1 * C
# dR2 = delta2 * R
# dL2 = delta2 * L
# dC2 = delta2 * C

# ckt = Circuit()
# ckt.make_branch("r0", BranchType.R, "n1", "0", R)
# ckt.make_branch("l0", BranchType.L, "n2", "n1", L)
# ckt.make_branch("c0", BranchType.C, "n2", "0", C)
# ckt.make_branch("i0", BranchType.I, "n2", "0", 1)
# # fps = np.geomspace(100e3, 100e6, 300, True)
# fps = np.linspace(100e3, 100e6, 300, True)
# typ, u, v, val, index = ckt.prepare_sim("0", ["i0", "r0", "l0", "c0"])
# sim = AcSimulationPardiso(typ, u, v, val)

# z0_sim = []
# for freq in fps:
#     sim.set_freq(freq)
#     sim.factorize()
#     sim.solve()
#     z = sim.branch_voltage(index[0:1]).abs()
#     z0_sim.append(z[0].item())
# z0_sim = np.array(z0_sim)

# z1_sim = []
# sim.alter(index[1:4], torch.tensor([R + dR1, L + dL1, C + dC1], dtype=torch.complex128))
# for freq in fps:
#     sim.set_freq(freq)
#     sim.factorize()
#     sim.solve()
#     z = sim.branch_voltage(index[0:1]).abs()
#     z1_sim.append(z[0].item())
# z1_sim = np.array(z1_sim)

# z2_sim = []
# sim.alter(index[1:4], torch.tensor([R + dR2, L + dL2, C + dC2], dtype=torch.complex128))
# for freq in fps:
#     sim.set_freq(freq)
#     sim.factorize()
#     sim.solve()
#     z = sim.branch_voltage(index[0:1]).abs()
#     z2_sim.append(z[0].item())
# z2_sim = np.array(z2_sim)

# delta_z1_sim = z1_sim - z0_sim
# delta_z2_sim = z2_sim - z0_sim


# r_index = index[1:2]
# r_value = val[r_index].requires_grad_(True)
# g_index = torch.zeros(0, dtype=torch.long)
# g_value = torch.zeros(0, dtype=torch.float64)
# l_index = index[2:3]
# l_value = val[l_index].requires_grad_(True)
# c_index = index[3:4]
# c_value = val[c_index].requires_grad_(True)
# xc_index = torch.zeros(0, dtype=torch.long)
# xc_value = torch.zeros(0, dtype=torch.float64)
# xl_index = torch.zeros(0, dtype=torch.long)
# xl_value = torch.zeros(0, dtype=torch.float64)
# i_index = index[0:1]
# v_index = torch.zeros(0, dtype=torch.long)
# all_exc_index = index[0:1]
# all_exc_value = val[all_exc_index]

# adj_sims = []
# for freq in fps:
#     adj_sims.append(AcSimulationPardiso(typ, u, v, val, freq))

# delta_z1_adj = []
# delta_z2_adj = []
# for freq, ac_sim in zip(fps, adj_sims):
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
#         ac_sim,
#     )
#     loss = i_voltage.abs()
#     loss.backward()
#     delta_z1 = r_value.grad * dR1 + l_value.grad * dL1 + c_value.grad * dC1
#     delta_z2 = r_value.grad * dR2 + l_value.grad * dL2 + c_value.grad * dC2
#     delta_z1_adj.append(delta_z1[0].item())
#     delta_z2_adj.append(delta_z2[0].item())
#     r_value.grad.zero_()
#     l_value.grad.zero_()
#     c_value.grad.zero_()
# delta_z1_adj = np.array(delta_z1_adj)
# delta_z2_adj = np.array(delta_z2_adj)

# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
# ax1.plot(fps / 10e6, delta_z1_sim, color="blue", linestyle="-", label="sim")
# ax1.plot(fps / 10e6, delta_z1_adj, color="red", linestyle="--", label="adj")
# ax1.set_ylabel("Impedance ($\\Omega$)")
# # ax1.set_yticks()
# ax1.set_xlabel("Frequency (MHz)")
# # ax1.set_xticks()
# ax1.legend()
# ax1.set_title("$\\delta$={}".format(delta1))

# ax2.plot(fps / 10e6, delta_z2_sim, color="blue", linestyle="-", label="sim")
# ax2.plot(fps / 10e6, delta_z2_adj, color="red", linestyle="--", label="adj")
# ax2.set_ylabel("Impedance ($\\Omega$)")
# # ax2.set_yticks()
# ax2.set_xlabel("Frequency (MHz)")
# # ax2.set_xticks()
# ax2.legend()
# ax2.set_title("$\\delta$={}".format(delta2))

# plt.tight_layout()
# plt.savefig("tmp.png", dpi=600)



# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from AcSimulation import AcSimulationScipy, AcSimulationPardiso
# from AcAdjoint import AcAdjointFunction
# from Circuit import Circuit, BranchType

# R = 30e-3
# L = 0.5e-9
# C = 100e-9
# delta1 = 0.1
# delta2 = 0.01
# dC1 = delta1 * C
# dC2 = delta2 * C

# ckt = Circuit()
# ckt.make_branch("r0", BranchType.R, "n1", "0", R)
# ckt.make_branch("l0", BranchType.L, "n2", "n1", L)
# ckt.make_branch("c0", BranchType.C, "n2", "0", C)
# ckt.make_branch("i0", BranchType.I, "n2", "0", 1)
# # fps = np.geomspace(100e3, 100e6, 300, True)
# fps = np.linspace(100e3, 100e6, 300, True)
# typ, u, v, val, index = ckt.prepare_sim("0", ["i0", "r0", "l0", "c0"])
# sim = AcSimulationPardiso(typ, u, v, val)

# z0_sim = []
# for freq in fps:
#     sim.set_freq(freq)
#     sim.factorize()
#     sim.solve()
#     z = sim.branch_voltage(index[0:1]).abs()
#     z0_sim.append(z[0].item())
# z0_sim = np.array(z0_sim)

# z1_sim = []
# sim.alter(index[1:4], torch.tensor([R, L, C + dC1], dtype=torch.complex128))
# for freq in fps:
#     sim.set_freq(freq)
#     sim.factorize()
#     sim.solve()
#     z = sim.branch_voltage(index[0:1]).abs()
#     z1_sim.append(z[0].item())
# z1_sim = np.array(z1_sim)

# z2_sim = []
# sim.alter(index[1:4], torch.tensor([R, L, C + dC2], dtype=torch.complex128))
# for freq in fps:
#     sim.set_freq(freq)
#     sim.factorize()
#     sim.solve()
#     z = sim.branch_voltage(index[0:1]).abs()
#     z2_sim.append(z[0].item())
# z2_sim = np.array(z2_sim)

# delta_z1_sim = z1_sim - z0_sim
# delta_z2_sim = z2_sim - z0_sim
# grad_c_sim1 = delta_z1_sim / dC1
# grad_c_sim2 = delta_z2_sim / dC2


# r_index = index[1:2]
# r_value = val[r_index].requires_grad_(True)
# g_index = torch.zeros(0, dtype=torch.long)
# g_value = torch.zeros(0, dtype=torch.float64)
# l_index = index[2:3]
# l_value = val[l_index].requires_grad_(True)
# c_index = index[3:4]
# c_value = val[c_index].requires_grad_(True)
# xc_index = torch.zeros(0, dtype=torch.long)
# xc_value = torch.zeros(0, dtype=torch.float64)
# xl_index = torch.zeros(0, dtype=torch.long)
# xl_value = torch.zeros(0, dtype=torch.float64)
# i_index = index[0:1]
# v_index = torch.zeros(0, dtype=torch.long)
# all_exc_index = index[0:1]
# all_exc_value = val[all_exc_index]

# adj_sims = []
# for freq in fps:
#     adj_sims.append(AcSimulationPardiso(typ, u, v, val, freq))

# grad_c_adj = []
# for freq, ac_sim in zip(fps, adj_sims):
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
#         ac_sim,
#     )
#     loss = i_voltage.abs()
#     loss.backward()
#     grad_c_adj.append(c_value.grad[0].item())
#     r_value.grad.zero_()
#     l_value.grad.zero_()
#     c_value.grad.zero_()
# grad_c_adj = np.array(grad_c_adj)

# plt.plot(
#     fps / 10e6,
#     grad_c_adj,
#     color="red",
#     linestyle="-",
#     label="adjoint",
# )
# plt.plot(
#     fps / 10e6,
#     grad_c_sim1,
#     color="green",
#     linestyle="-.",
#     label="simulation($\\delta={}$)".format(delta1),
# )
# plt.plot(
#     fps / 10e6,
#     grad_c_sim2,
#     color="blue",
#     linestyle="--",
#     label="simulation($\\delta={}$)".format(delta2),
# )
# plt.ylabel("Gradient")
# plt.xlabel("Frequency (MHz)")
# # plt.xscale("log")
# plt.legend()
# plt.tight_layout()
# plt.savefig("tmp.png", dpi=600)



from Circuit import PG, TSV, MicroBump, BranchType, DTC
from AcSimulation import AcSimulationScipy, AcSimulationPardiso, AcSimulationCuDSS
from AcAdjoint import AcAdjointFunction
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

x = 20
y = 15
vdd = 1.8

pg = PG(x, y)
tsv = TSV()
dtc = DTC()
microbump = MicroBump()
ckt = pg.clone()

# add tsv
# tsv_index = [(5, 9), (10, 9), (7, 9), (7, 6), (5, 7), (11, 6), (5, 4), (8, 4), (11, 3)]
# tsv_index = [(10, 9), (6, 9), (10, 6), (10, 3)]
tsv_index = [
    (11, 9),
    (8, 9),
    (5, 9),
    (6, 8),
    (12, 8),
    (9, 8),
    (6, 6),
    (10, 7),
    (11, 6),
    (7, 5),
    (8, 4),
    (9, 3),
]
for ii, jj in tsv_index:
    ckt.make_subcircuit(
        tsv,
        "tsv{}_{}".format(ii, jj),
        [("c{}_{}".format(ii, jj), "top"), ("0", "bottom")],
    )

candidate_dtc = []
for ii, jj in [(i + 1, j + 1) for i in range(y) for j in range(x)]:
    if (ii, jj) in tsv_index:
        continue
    ckt.make_branch(
        "dtc{}_{}".format(ii, jj),
        BranchType.C,
        "c{}_{}".format(ii, jj),
        "0",
        0.245e-9,
    )
    candidate_dtc.append("dtc{}_{}".format(ii, jj))

chiplets = [(1, 3, 9, 12, 1.5, 3.8e9), (10, 3, 19, 12, 0.2, 2.4e9)]
for k, (lx, ly, hx, hy, power, freq) in enumerate(chiplets):
    for i in range(ly, hy):
        for j in range(lx, hx):
            ckt.make_branch(
                "bump{}_{}".format(i + 1, j + 1),
                BranchType.L,
                "c{}_{}".format(i + 1, j + 1),
                "die{}_sink".format(k),
                2.3e-12,
            )
    C_ODC = 9 * power / vdd / vdd / freq
    ESR = 0.25e-9 / C_ODC
    ckt.make_branch(
        "die{}_ESR".format(k),
        BranchType.R,
        "die{}_sink".format(k),
        "die{}_mid".format(k),
        ESR,
    )
    ckt.make_branch(
        "die{}_ODC".format(k), BranchType.C, "die{}_mid".format(k), "0", C_ODC
    )
ckt.make_branch("die0_obs", BranchType.I, "die0_sink", "0", 1)

observe_load = "die0_obs"

lr = 0.1
step = 1
niters = 500
frequency_points = torch.from_numpy(np.geomspace(0.1e9, 10e9, 200))
gamma = 0.01
target_impdeance = 0.1
result_dtc = []

ci_names = candidate_dtc + [observe_load]
typ, u, v, val, index = ckt.prepare_sim("0", ci_names)
c_index = index[: len(candidate_dtc)]
c_base_value = val[c_index]
r_index = torch.zeros(0, dtype=torch.long)
r_value = torch.zeros(0, dtype=torch.float64)
g_index = torch.zeros(0, dtype=torch.long)
g_value = torch.zeros(0, dtype=torch.float64)
l_index = torch.zeros(0, dtype=torch.long)
l_value = torch.zeros(0, dtype=torch.float64)
i_index = index[len(candidate_dtc) :]
v_index = torch.zeros(0, dtype=torch.long)
(all_exc_index,) = torch.where((typ == BranchType.I) | (typ == BranchType.V))
# all_exc_value = val[all_exc_index].repeat(frequency_points.size(0), 1)
all_exc_value = val[all_exc_index]
# sim = AcSimulationPardiso(typ, u, v, val)

xc_index = torch.tensor([], dtype=torch.long)
xl_index = torch.tensor([], dtype=torch.long)
xc_value = torch.tensor([], dtype=torch.float64)
xl_value = torch.tensor([], dtype=torch.float64)

sims = []
for freq in frequency_points:
    sims.append(AcSimulationCuDSS(typ, u, v, val, freq))

history = []


temperature = 1
temperature_ratio = 0.95
temperature_update_iteration = niters // 30
torch.manual_seed(42)
theta = nn.Parameter(torch.rand((c_index.size(0), 2)))


optimizer = torch.optim.Adam(nn.ParameterList([theta]), lr=lr)
loss_history = []
z_loss_history = []
count_loss_history = []

for i in range(niters):
    # update gumbel softmax temperature
    if i != 0 and i % temperature_update_iteration == 0:
        temperature *= temperature_ratio
    p = torch.nn.functional.gumbel_softmax(theta, tau=temperature, dim=1)
    c_value = c_base_value * p[:, 1]
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
    impedance_violation = torch.nn.functional.relu(impedances - target_impdeance)
    total_impedance_violation = impedance_violation.sum() / frequency_points.shape[0]
    dtc_count = p[:, 1].sum()
    loss = (
        50000 * (total_impedance_violation)
        + 1 * dtc_count
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
    optimizer.zero_grad()

print(theta.size())
print(theta)
p = torch.nn.functional.gumbel_softmax(theta, tau=temperature, dim=1)
print(p)
exit()

# epoch = 0
# while True:
#     epoch += 1
#     if epoch > 200:
#         break
#     theta = nn.Parameter(torch.rand_like(c_index, dtype=torch.float64))
#     # theta = nn.Parameter(
#     #     torch.ones_like(c_index, dtype=torch.float64) / c_index.size(0)
#     # )
#     param_list = nn.ParameterList([theta])
#     optimizer = torch.optim.Adam(param_list, lr=lr)

#     last_loss = None
#     for iter in range(niters):
#         optimizer.zero_grad()
#         # p = torch.softmax(theta, -1)
#         p = torch.nn.functional.gumbel_softmax(theta, tau=0.01, hard=False)
#         c_value = p * c_base_value

#         i_voltage, v_current = AcAdjointFunction.apply(
#             g_index,
#             r_index,
#             c_index,
#             l_index,
#             i_index,
#             v_index,
#             all_exc_index,
#             g_value,
#             r_value,
#             c_value,
#             l_value,
#             all_exc_value,
#             frequency_points,
#             sim,
#         )

#         z = i_voltage.abs() - target_impdeance * 0.9
#         expz = torch.exp(z / gamma)
#         loss = torch.sum(gamma * torch.log(1 + expz))
#         # loss = torch.sum(i_voltage.abs())
#         loss.backward()
#         optimizer.step()

#         if iter % step == 0:
#             print("iter {}: loss {}".format(iter, loss))
#         if last_loss is None:
#             last_loss = loss.detach()
#         else:
#             if (last_loss - loss.detach()).item() < 1e-3:
#                 break
#             last_loss = loss.detach()
#         history.append((iter, loss.detach().item()))

#     with torch.no_grad():
#         target = torch.argmax(theta)
#         c_value = torch.cat(
#             [
#                 torch.zeros(target, dtype=torch.float64),
#                 c_base_value[target].reshape(1),
#                 torch.zeros(c_index.size(0) - target - 1, dtype=torch.float64),
#             ]
#         )

#         i_voltage, v_current = AcAdjointFunction.apply(
#             g_index,
#             r_index,
#             c_index,
#             l_index,
#             # xc_index,
#             # xl_index
#             i_index,
#             v_index,
#             all_exc_index,
#             g_value,
#             r_value,
#             c_value,
#             l_value,
#             # xc_value,
#             # xl_value,
#             all_exc_value,
#             frequency_points,
#             sim,
#         )

#         worst_z = torch.max(i_voltage.abs())
#         c_index = torch.cat([c_index[:target], c_index[target + 1 :]])
#         c_base_value = torch.cat([c_base_value[:target], c_base_value[target + 1 :]])
#         result_dtc.append(candidate_dtc[target.item()])
#         print(candidate_dtc[target.item()], worst_z.item())
#         candidate_dtc.pop(target.item())
#         if worst_z.item() < target_impdeance:
#             break
#         gamma = (worst_z - target_impdeance) / 100
# print(result_dtc)

# import pickle

# with open("ac_history.pkl", "wb") as f:
#     pickle.dump(history, f)