from Circuit import PG_OP, TSV_OP, BranchType
import torch
import torch.nn as nn
from OpSimulation import OpSimulationPardiso, OpSimulationCuDSS
import random

x = 20
y = 15
vdd = 1.8
pg = PG_OP(x, y)
tsv = TSV_OP()
ckt = pg.clone()
ckt.make_branch("vdd", BranchType.V, "vdd", "0", vdd)
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
tsv_index = random.sample([(i + 1, j + 1) for j in range(x) for i in range(y)], k=11)
for ii, jj in tsv_index:
    ckt.make_branch(
        "tsv{}_{}".format(ii, jj),
        BranchType.G,
        "vdd",
        "c{}_{}".format(ii, jj),
        1 / 5.41e-3,
    )

chiplets = [(1, 3, 9, 12, 1.5, 3.8e9), (10, 3, 19, 12, 0.2, 2.4e9)]
observe_load = []
for lx, ly, hx, hy, power, freq in chiplets:
    i_per_uc = power / (hx - lx) / (hy - ly) / vdd
    for i in range(ly, hy):
        for j in range(lx, hx):
            ckt.make_branch(
                "load{}_{}".format(i + 1, j + 1),
                BranchType.I,
                "c{}_{}".format(i + 1, j + 1),
                "0",
                i_per_uc,
            )
            observe_load.append("load{}_{}".format(i + 1, j + 1))

i_names = observe_load
typ, u, v, val, index = ckt.prepare_sim("0", i_names)
# sim = OpSimulationPardiso(typ, u, v, val)
sim = OpSimulationCuDSS(typ, u, v, val)

sim.factorize()
sim.solve()
i_voltage = sim.branch_voltage(index)
ir_drop = vdd - i_voltage
print(torch.sum(ir_drop))
print(torch.max(ir_drop))
print("drop = {:.5}% vdd".format((torch.max(ir_drop) / vdd).item() * 100))
