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
        freq: float | torch.Tensor,
        sim: AcSimulation,
    ):
        # 更新电路参数 (假设参数在所有频率下共享)
        sim.alter(g_index, g_value)
        sim.alter(r_index, r_value)
        sim.alter(c_index, c_value)
        sim.alter(l_index, l_value)
        sim.alter(xc_index, xc_value)
        sim.alter(xl_index, xl_value)
        sim.alter(all_exc_index, all_exc_value)

        is_batch = isinstance(freq, torch.Tensor)
        
        if is_batch:
            # 批量模式
            # freq: [num_freqs]
            solutions = sim.solve_batch(freq) # [N, n]
            sim.V = solutions # 暂存解向量，供 branch_voltage/current 使用
            
            # 计算输出 (传入 freqs 以支持批量导纳计算)
            g_voltage = sim.branch_voltage(g_index)
            r_current = sim.branch_current(r_index, freqs=freq)
            c_voltage = sim.branch_voltage(c_index)
            l_current = sim.branch_current(l_index, freqs=freq)
            xc_current = sim.branch_current(xc_index, freqs=freq)
            xl_voltage = sim.branch_voltage(xl_index)
            i_voltage = sim.branch_voltage(i_index)
            v_current = sim.branch_current(v_index, freqs=freq)
            
        else:
            # 单频率模式 (Legacy)
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
        ctx.is_batch = is_batch
        
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
        freq = ctx.freq
        is_batch = ctx.is_batch

        if is_batch:
            # === 批量 Adjoint ===
            # freq: [N]
            # grad_i_voltage: [N, num_i]
            # grad_v_current: [N, num_v]
            
            num_freqs = freq.size(0)
            num_nodes = sim.n
            rhs_batch = torch.zeros((num_freqs, num_nodes), dtype=torch.complex128, device=sim.device)
            
            # 1. 处理 Current Source (i_index) 的 Adjoint 源: grad_i_voltage.conj()
            # 对应原电路中的电流源，Adjoint 中也是电流源
            # KCL: 注入节点 u, 流出节点 v
            if i_index.numel() > 0:
                u = sim.branch_u[i_index] - 1
                v = sim.branch_v[i_index] - 1
                val = grad_i_voltage.conj() # [N, num_i]
                
                # 广播 u, v 到 [N, num_i] (其实 u, v 是常数，不需要广播，直接用 index_add_)
                # rhs_batch[:, u] -= val  (注意符号，原代码 sim.alter 是正值，这里我们要看 KCL)
                # 原代码: sim.alter(i_index, grad_i_voltage.conj()) -> 设置电流源值
                # 电流源 I 从 u 流向 v。 KCL at u: ... + I = 0 -> ... = -I.  RHS at u is -I?
                # 通常 MNA: G x = b. 电流源 I (u->v) -> b[u] -= I, b[v] += I.
                # 让我们检查 update_rhs_kernel:
                # atomicAdd(&d_b[u].x, -val.x) -> b[u] -= val
                # atomicAdd(&d_b[v].x, val.x)  -> b[v] += val
                # 所以:
                
                # 批量 scatter_add
                # u, v 是 [num_i]
                # val 是 [N, num_i]
                # 我们需要对每一行进行操作。
                
                # 简便方法：循环 num_i (通常源数量不多) 或者 flatten
                # 为了效率，使用 index_add_ (dim=1)
                
                valid_u = u >= 0
                if valid_u.any():
                    # rhs_batch.index_add_(1, u[valid_u], -val[:, valid_u]) 
                    # index_add_ expect source to match dim. 
                    # rhs_batch: [N, n], val: [N, num_i]. 
                    # index_add_(dim, index, source). index is [num_i]. source must be [N, num_i].
                    # Yes, this works!
                    rhs_batch.index_add_(1, u[valid_u], -val[:, valid_u])
                    
                valid_v = v >= 0
                if valid_v.any():
                    rhs_batch.index_add_(1, v[valid_v], val[:, valid_v])

            # 2. 处理 Voltage Source (v_index) 的 Adjoint 源: -grad_v_current.conj()
            # 对应原电路中的电压源，Adjoint 中是电压源
            # MNA Aux Equation: V_u - V_v = E
            # RHS index: num_nonzero_nodes + branch_idx (relative index?)
            # No, V source row index is determined by its position in matrix.
            # In AcSimulationScipy: V_row = num_nonzero_nodes + V_index (relative to V_index array? No, V_index is global branch index)
            # Wait, V_row calculation in Scipy:
            # (V_index,) = np.where(branch_typ == BranchType.V) -> indices into branch array
            # V_row = self.num_nonzero_nodes + (0, 1, 2...) ?
            # No, MNA rows for V sources are appended after node equations.
            # Usually order matches the order of V branches.
            # Let's assume V rows correspond to v_index order if sorted?
            # Or does `AcSimulation` maintain a mapping?
            # In `ac_sim_cuda.cu`: `d_b[num_nonzero_nodes + branch_idx] = ...`
            # This implies the row index is `num_nonzero_nodes + branch_idx`.
            # BUT `branch_idx` is the global index of the branch.
            # This implies the matrix size is `num_nonzero_nodes + num_branches`? 
            # No, matrix size `n` is usually `num_nodes + num_v_sources`.
            # If `d_b` is indexed by `branch_idx`, then `d_b` must be huge?
            # Let's check `acCuDssHandleInit`.
            # `h.n = h.num_nonzero_nodes + h.num_v_sources`.
            # So `d_b` size is `n`.
            # If we access `d_b[num_nonzero_nodes + branch_idx]`, then `branch_idx` MUST be small (0..num_v_sources-1)?
            # NO. `branch_idx` is global.
            # So `update_rhs_kernel` logic `d_b[num_nonzero_nodes + branch_idx]` seems WRONG if `branch_idx` > `num_v_sources`.
            # UNLESS `branch_idx` in the kernel loop is re-mapped?
            # `update_rhs_kernel` iterates `idx` from 0 to `m` (num branches).
            # `if (type == BR_V) ... d_b[... + idx]`.
            # This suggests the matrix size `n` >= `num_nonzero_nodes + m`.
            # But `h.n` is `num_nonzero_nodes + num_v_sources`.
            # So `update_rhs_kernel` logic is suspicious OR I misunderstood `branch_idx`.
            # Wait, `update_rhs_kernel` logic:
            # `d_b[num_nonzero_nodes + branch_idx] = ...`
            # If `branch_idx` is large (e.g. last branch), this writes out of bounds!
            # UNLESS V-sources are ALWAYS at the beginning of branch list?
            # Or `branch_idx` is not global index?
            # In `update_rhs_kernel`: `int branch_idx = idx;` (global index).
            
            # **CRITICAL FINDING**: The CUDA kernel might be writing OOB if V-sources are not at start.
            # BUT, let's look at `AcSimulation.py` Scipy implementation:
            # `V_row = self.num_nonzero_nodes + V_index` (line 319).
            # `V_index` comes from `np.where(...)`. These are indices into `branch_typ`.
            # So `V_row` depends on global branch index.
            # This implies the matrix row indices for V-sources are NOT packed.
            # They are sparse?
            # `G = dok_matrix((self.n, self.n))`
            # `self.n` calculation?
            # I need to check `AcSimulation.__init__`.
            
            # If `self.n` is large enough, then it's fine.
            # Let's assume the user's code structure is correct for their MNA formulation.
            # So I should use `num_nonzero_nodes + v_index` as the row index.
            
            if v_index.numel() > 0:
                row_v = sim.num_nonzero_nodes + v_index
                val = -grad_v_current.conj() # [N, num_v]
                
                # rhs_batch[:, row_v] = val
                # row_v is [num_v]. val is [N, num_v].
                rhs_batch[:, row_v] = val

            # 3. 批量求解 Adjoint
            adj_solutions = sim.solve_batch_with_rhs(freq, rhs_batch)
            sim.V = adj_solutions # 更新 sim.V 以供 branch_voltage 使用
            
            # 4. 计算 Adjoint Branch Quantities
            adj_g_voltage = sim.branch_voltage(g_index)
            adj_r_current = sim.branch_current(r_index, freqs=freq)
            adj_c_voltage = sim.branch_voltage(c_index)
            adj_l_current = sim.branch_current(l_index, freqs=freq)
            adj_xc_current = sim.branch_current(xc_index, freqs=freq)
            adj_xl_voltage = sim.branch_voltage(xl_index)
            
            # 5. 计算梯度 (包含频率项)
            # jomega: [N]
            jomega = 2j * torch.pi * freq
            # Broadcasting jomega to [N, 1] for multiplication with [N, M]
            jomega = jomega.unsqueeze(1)
            
            grad_g_value = g_voltage * adj_g_voltage
            grad_r_value = -r_current * adj_r_current
            grad_c_value = jomega * c_voltage * adj_c_voltage
            grad_l_value = -jomega * l_current * adj_l_current
            grad_xc_value = -1.0 / jomega * xc_current * adj_xc_current
            grad_xl_value = 1.0 / jomega * xl_voltage * adj_xl_voltage
            
            # 6. 对频率维度求和，得到参数梯度 [M]
            grad_g_value = torch.sum(grad_g_value, dim=0)
            grad_r_value = torch.sum(grad_r_value, dim=0)
            grad_c_value = torch.sum(grad_c_value, dim=0)
            grad_l_value = torch.sum(grad_l_value, dim=0)
            grad_xc_value = torch.sum(grad_xc_value, dim=0)
            grad_xl_value = torch.sum(grad_xl_value, dim=0)
            
        else:
            # === 单频率 Adjoint (Legacy) ===
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
