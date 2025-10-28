import torch
from AcSimulation import AcSimulation
from Circuit import BranchType


class AcAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        g_index: torch.Tensor,
        r_index: torch.Tensor,
        c_index: torch.Tensor,
        l_index: torch.Tensor,
        xc_index: torch.Tensor,
        xl_index: torch.Tensor,
        i_index: torch.Tensor,
        v_index: torch.Tensor,
        all_exc_index: torch.Tensor,
        g_value: torch.Tensor,
        r_value: torch.Tensor,
        c_value: torch.Tensor,
        l_value: torch.Tensor,
        xc_value: torch.Tensor,
        xl_value: torch.Tensor,
        all_exc_value: torch.Tensor,
        freq: float,
        sim: AcSimulation,
    ):
        sim.alter(g_index, g_value)
        sim.alter(r_index, r_value)
        sim.alter(c_index, c_value)
        sim.alter(l_index, l_value)
        sim.alter(xc_index, xc_value)
        sim.alter(xl_index, xl_value)
        sim.alter(all_exc_index, all_exc_value)

        sim.set_freq(freq)
        sim.factorize()
        sim.solve()

        g_voltage = sim.branch_voltage(g_index)
        r_current = sim.branch_current(r_index)
        c_voltage = sim.branch_voltage(c_index)
        l_current = sim.branch_current(l_index)
        xc_current = sim.branch_current(xc_index)
        xl_voltage = sim.branch_voltage(xl_index)
        i_voltage = sim.branch_voltage(i_index)
        v_current = sim.branch_current(v_index)

        ctx.sim = sim
        ctx.freq = freq
        ctx.save_for_backward(
            g_index,
            r_index,
            c_index,
            l_index,
            xc_index,
            xl_index,
            i_index,
            v_index,
            all_exc_index,
            g_voltage,
            r_current,
            c_voltage,
            l_current,
            xc_current,
            xl_voltage,
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
        c_index: torch.Tensor = None
        l_index: torch.Tensor = None
        xc_index: torch.Tensor = None
        xl_index: torch.Tensor = None
        i_index: torch.Tensor = None
        v_index: torch.Tensor = None
        all_exc_index: torch.Tensor = None
        g_voltage: torch.Tensor = None
        r_current: torch.Tensor = None
        c_voltage: torch.Tensor = None
        l_current: torch.Tensor = None
        xc_current: torch.Tensor = None
        xl_voltage: torch.Tensor = None
        (
            g_index,
            r_index,
            c_index,
            l_index,
            xc_index,
            xl_index,
            i_index,
            v_index,
            all_exc_index,
            g_voltage,
            r_current,
            c_voltage,
            l_current,
            xc_current,
            xl_voltage,
        ) = ctx.saved_tensors
        sim: AcSimulation = ctx.sim
        freq: float = ctx.freq

        sim.alter(
            all_exc_index, torch.zeros(all_exc_index.size(0), dtype=torch.complex128)
        )
        sim.alter(i_index, grad_i_voltage.conj())
        sim.alter(v_index, -grad_v_current.conj())
        # sim.alter(i_index, grad_i_voltage)
        # sim.alter(v_index, -grad_v_current)
        # sim.set_freq(freq)
        # sim.factorize()
        sim.solve()

        adj_g_voltage = sim.branch_voltage(g_index)
        adj_r_current = sim.branch_current(r_index)
        adj_c_voltage = sim.branch_voltage(c_index)
        adj_l_current = sim.branch_current(l_index)
        adj_xc_current = sim.branch_current(xc_index)
        adj_xl_voltage = sim.branch_voltage(xl_index)

        jomega = 2j * torch.pi * freq
        grad_g_value = g_voltage * adj_g_voltage
        grad_r_value = -r_current * adj_r_current
        grad_c_value = jomega * c_voltage * adj_c_voltage
        grad_l_value = -jomega * l_current * adj_l_current
        grad_xc_value = -1.0 / jomega * xc_current * adj_xc_current
        grad_xl_value = 1.0 / jomega * xl_voltage * adj_xl_voltage

        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            torch.real(grad_g_value),
            torch.real(grad_r_value),
            torch.real(grad_c_value),
            torch.real(grad_l_value),
            torch.real(grad_xc_value),
            torch.real(grad_xl_value),
            None,
            None,
            None,
        )


# class AcAdjointModule(torch.nn.Module):
#     def __init__(
#         self,
#         branch_typ: torch.Tensor,
#         branch_u: torch.Tensor,
#         branch_v: torch.Tensor,
#         branch_val: torch.Tensor,
#         candidate_branch_index: torch.Tensor,
#         observe_branch_index: torch.Tensor,
#         freq: float,
#         sim: AcSimulation = None,
#     ):
#         super().__init__()

#         if sim is None:
#             self.sim = AcSimulationPardiso(
#                 branch_typ, branch_u, branch_v, branch_v, freq
#             )
#         else:
#             self.sim = sim

#         self.freq = freq

#         candidate_branch_typ = branch_typ[candidate_branch_index]
#         self.g_mask = candidate_branch_typ == BranchType.G
#         self.g_index = candidate_branch_index[self.g_mask]
#         self.r_mask = candidate_branch_typ == BranchType.R
#         self.r_index = candidate_branch_index[self.r_mask]
#         self.c_mask = candidate_branch_typ == BranchType.C
#         self.c_index = candidate_branch_index[self.c_mask]
#         self.l_mask = candidate_branch_typ == BranchType.L
#         self.l_index = candidate_branch_index[self.l_mask]

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
#         c_value = candidate_branch_value[self.c_mask]
#         l_value = candidate_branch_value[self.l_mask]

#         i_voltage, v_current = AcAdjointFunction.apply(
#             self.g_index,
#             self.r_index,
#             self.c_index,
#             self.l_index,
#             self.i_index,
#             self.v_index,
#             self.all_exc_index,
#             g_value,
#             r_value,
#             c_value,
#             l_value,
#             self.all_exc_value,
#             self.freq,
#             self.sim,
#         )

#         result = torch.empty_like(self.i_mask, dtype=torch.complex128)
#         result[self.i_mask] = i_voltage
#         result[self.v_mask] = v_current
#         return result
