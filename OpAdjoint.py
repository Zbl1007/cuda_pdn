import torch
from OpSimulation import OpSimulation, OpSimulationScipy, OpSimulationPardiso
from Circuit import BranchType


class OpAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        g_index: torch.Tensor,
        r_index: torch.Tensor,
        i_index: torch.Tensor,
        v_index: torch.Tensor,
        all_exc_index: torch.Tensor,
        g_value: torch.Tensor,
        r_value: torch.Tensor,
        all_exc_value: torch.Tensor,
        sim: OpSimulation,
    ):
        sim.alter(g_index, g_value)
        sim.alter(r_index, r_value)
        sim.alter(all_exc_index, all_exc_value)
        sim.factorize()
        sim.solve()

        g_voltage = sim.branch_voltage(g_index)
        i_voltage = sim.branch_voltage(i_index)
        r_current = sim.branch_voltage(r_index)
        v_current = sim.branch_voltage(v_index)

        ctx.sim = sim
        ctx.save_for_backward(
            g_index,
            r_index,
            i_index,
            v_index,
            all_exc_index,
            g_voltage,
            r_current,
        )
        return i_voltage, v_current

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_i_voltage: torch.Tensor,
        grad_v_current: torch.Tensor,
    ):
        g_index: torch.Tensor = None
        r_index: torch.Tensor = None
        i_index: torch.Tensor = None
        v_index: torch.Tensor = None
        all_exc_index: torch.Tensor = None
        g_voltage: torch.Tensor = None
        r_current: torch.Tensor = None
        g_index, r_index, i_index, v_index, all_exc_index, g_voltage, r_current = (
            ctx.saved_tensors
        )
        sim: OpSimulation = ctx.sim

        sim.alter(all_exc_index, torch.zeros_like(all_exc_index, dtype=torch.float64))
        sim.alter(i_index, grad_i_voltage)
        sim.alter(v_index, -grad_v_current)
        sim.solve()

        adj_g_voltage = sim.branch_voltage(g_index)
        adj_r_current = sim.branch_current(r_index)
        grad_g_value = g_voltage * adj_g_voltage
        grad_r_value = -r_current * adj_r_current

        return (None, None, None, None, None, grad_g_value, grad_r_value, None, None)


# class OpAdjointModule(torch.nn.Module):
#     def __init__(
#         self,
#         branch_typ: torch.Tensor,
#         branch_u: torch.Tensor,
#         branch_v: torch.Tensor,
#         branch_val: torch.Tensor,
#         candidate_branch_index: torch.Tensor,
#         observe_branch_index: torch.Tensor,
#         sim: OpSimulation = None,
#     ):
#         super().__init__()

#         if sim is None:
#             self.sim = OpSimulationPardiso(branch_typ, branch_u, branch_v, branch_v)
#         else:
#             self.sim = sim

#         candidate_branch_typ = branch_typ[candidate_branch_index]
#         self.g_mask = candidate_branch_typ == BranchType.G
#         self.g_index = candidate_branch_index[self.g_mask]
#         self.r_mask = candidate_branch_typ == BranchType.R
#         self.r_index = candidate_branch_index[self.r_mask]

#         observe_branch_typ = branch_typ[observe_branch_index]
#         self.i_mask = observe_branch_typ == BranchType.I
#         self.i_index = observe_branch_index[self.i_mask]
#         self.v_mask = observe_branch_typ == BranchType.V
#         self.v_index = observe_branch_index[self.v_mask]

#         (self.all_exc_index,) = torch.where(
#             (branch_typ == BranchType.I) | (branch_typ == BranchType.V)
#         )
#         self.all_exc_value = branch_val[self.all_exc_index]

#     def forward(self, candidate_branch_value: torch.Tensor) -> torch.Tensor:
#         g_value = candidate_branch_value[self.g_mask]
#         r_value = candidate_branch_value[self.r_mask]

#         i_voltage, v_current = OpAdjointFunction.apply(
#             self.g_index,
#             self.r_index,
#             self.i_index,
#             self.v_index,
#             self.all_exc_index,
#             g_value,
#             r_value,
#             self.all_exc_value,
#             self.sim,
#         )

#         result = torch.empty_like(self.i_mask, dtype=torch.float64)
#         result[self.i_mask] = i_voltage
#         result[self.v_mask] = v_current
#         return result
