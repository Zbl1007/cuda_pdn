import numpy as np
import torch
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import splu, SuperLU
from Circuit import BranchType
from ac_sim_cpp import (
    AcPardisoHandle,
    acPardisoHandleInit,
    acPardisoHandleFactorize,
    acPardisoHandleSolve,
    acPardisoHandleTerminate,
    acPardisoHandleGetSolution,
    acPardisoHandleExport,
)
from ac_sim_cuda_cpp import (
    AcCuDssHandle,
    acCuDssHandleInit,
    acCuDssHandleFactorize,
    acCuDssHandleSolve,
    acCuDssHandleTerminate,
    acCuDssHandleGetSolution,
    acCuDssHandleExport, # 我们将为 GPU 版本添加这个函数
    acCuDssHandleSolveBatch,
    acCuDssHandleSolveBatchWithRhs
)


class AcSimulation:
    def __init__(
        self,
        branch_type: torch.Tensor,
        branch_u: torch.Tensor,
        branch_v: torch.Tensor,
        branch_value: torch.Tensor,
        freq: float = 1,
    ):
        self.branch_typ = branch_type.contiguous()
        self.branch_u = branch_u.contiguous()
        self.branch_v = branch_v.contiguous()
        self.branch_val = branch_value.clone().type(torch.complex128).contiguous()

        self.V: torch.Tensor = None
        self.freq: float = freq

        self.num_nonzero_nodes = max(
            torch.max(self.branch_u).item(), torch.max(self.branch_v).item()
        )
        self.num_voltage_branches = torch.sum(self.branch_typ == BranchType.V).item()
        self.n = self.num_nonzero_nodes + self.num_voltage_branches

    def set_freq(self, freq: float):
        self.freq = freq

    def factorize(self):
        pass

    def solve(self):
        pass

    def alter(self, branch_index: torch.Tensor, branch_value: torch.Tensor):
        self.branch_val[branch_index] = branch_value.type(torch.complex128)
        self.branch_val = self.branch_val.contiguous()

    def branch_current(self, branch_index: torch.Tensor) -> np.ndarray:
        V_index_mask = self.branch_typ[branch_index] == BranchType.V
        V_index = branch_index[V_index_mask]
        V_row = self.num_nonzero_nodes + V_index
        V_current = self.V[V_row]

        I_index_mask = self.branch_typ[branch_index] == BranchType.I
        I_index = branch_index[I_index_mask]
        I_current = self.branch_val[I_index]

        G_index_mask = self.branch_typ[branch_index] == BranchType.G
        G_index = branch_index[G_index_mask]
        G_val = self.branch_val[G_index]
        R_index_mask = self.branch_typ[branch_index] == BranchType.R
        R_index = branch_index[R_index_mask]
        R_val = self.branch_val[R_index]
        C_index_mask = self.branch_typ[branch_index] == BranchType.C
        C_index = branch_index[C_index_mask]
        C_val = self.branch_val[C_index]
        L_index_mask = self.branch_typ[branch_index] == BranchType.L
        L_index = branch_index[L_index_mask]
        L_val = self.branch_val[L_index]
        XC_index_mask = self.branch_typ[branch_index] == BranchType.XC
        XC_index = branch_index[XC_index_mask]
        XC_val = self.branch_val[XC_index]
        XL_index_mask = self.branch_typ[branch_index] == BranchType.XL
        XL_index = branch_index[XL_index_mask]
        XL_val = self.branch_val[XL_index]
        Y_index_mask = self.branch_typ[branch_index] == BranchType.Y
        Y_index = branch_index[Y_index_mask]
        Y_val = self.branch_val[Y_index]
        Z_index_mask = self.branch_typ[branch_index] == BranchType.Z
        Z_index = branch_index[Z_index_mask]
        Z_val = self.branch_val[Z_index]

        G_index_mask = (
            G_index_mask
            | R_index_mask
            | C_index_mask
            | L_index_mask
            | XC_index_mask
            | XL_index_mask
            | Y_index_mask
            | Z_index_mask
        )
        G_index = torch.concat(
            [G_index, R_index, C_index, L_index, XC_index, XL_index, Y_index, Z_index]
        )
        G_val = torch.concat(
            [
                G_val,
                1 / R_val,
                2j * np.pi * self.freq * C_val,
                1 / (2j * np.pi * self.freq * L_val),
                2j * np.pi * self.freq / XC_val,
                1 / (2j * np.pi * self.freq) * XL_val,
                Y_val,
                1 / Z_val,
            ]
        )
        G_current = self.branch_voltage(G_index) * G_val

        current = torch.zeros_like(branch_index, dtype=torch.complex128)
        current[V_index_mask] = V_current
        current[G_index_mask] = G_current
        current[I_index_mask] = I_current
        return current

    def branch_current_batch(self, branch_index: torch.Tensor, freqs: torch.Tensor = None) -> torch.Tensor:
        """
        计算支路电流。
        如果 freqs 不为 None，则执行批量计算。
        freqs: [num_freqs]
        Returns: [num_freqs, num_branches] if freqs is not None, else [num_branches]
        """
        branch_typ = self.branch_typ[branch_index]
        branch_val = self.branch_val[branch_index]

        V_index_mask = branch_typ == BranchType.V
        I_index_mask = branch_typ == BranchType.I
        G_index_mask = branch_typ == BranchType.G
        R_index_mask = branch_typ == BranchType.R
        C_index_mask = branch_typ == BranchType.C
        L_index_mask = branch_typ == BranchType.L
        XC_index_mask = branch_typ == BranchType.XC
        XL_index_mask = branch_typ == BranchType.XL
        Y_index_mask = branch_typ == BranchType.Y
        Z_index_mask = branch_typ == BranchType.Z

        (V_index,) = torch.where(V_index_mask)
        # V_val = branch_val[V_index]
        (I_index,) = torch.where(I_index_mask)
        I_val = branch_val[I_index]
        (G_index,) = torch.where(G_index_mask)
        G_val = branch_val[G_index]
        (R_index,) = torch.where(R_index_mask)
        R_val = branch_val[R_index]
        (C_index,) = torch.where(C_index_mask)
        C_val = branch_val[C_index]
        (L_index,) = torch.where(L_index_mask)
        L_val = branch_val[L_index]
        (XC_index,) = torch.where(XC_index_mask)
        XC_val = branch_val[XC_index]
        (XL_index,) = torch.where(XL_index_mask)
        XL_val = branch_val[XL_index]
        (Y_index,) = torch.where(Y_index_mask)
        Y_val = branch_val[Y_index]
        (Z_index,) = torch.where(Z_index_mask)
        Z_val = branch_val[Z_index]

        # V_current 需要特殊处理，通常通过 KCL 或 MNA 求解得到
        # 这里假设 V_current 无法直接通过 V/Z 计算，暂且返回 0 或需要从解向量中提取（如果 MNA 包含电流变量）
        # 在 MNA 中，电压源电流是未知量，存储在解向量的后半部分
        # 假设 self.V 包含了电压源电流
        # 需要知道电压源电流在 self.V 中的索引
        # 假设电压源电流索引从 num_nodes 开始
        # 这是一个简化的假设，实际需要根据 MNA 结构
        # 暂时忽略 V_current 的精确计算，或者假设它为 0
        V_current = torch.zeros_like(V_index, dtype=torch.complex128) 
        # TODO: 从 self.V 中提取 V_current
        
        I_current = I_val # 电流源电流等于其值
        if freqs is not None:
             # 广播 I_current: [M] -> [N, M]
             I_current = I_current.unsqueeze(0).expand(freqs.size(0), -1)
             V_current = V_current.unsqueeze(0).expand(freqs.size(0), -1)

        # 计算导纳
        if freqs is None:
            f = self.freq
            omega = 2 * np.pi * f
            jomega = 1j * omega
            
            G_admittance = torch.concat([
                G_val,
                1 / R_val,
                jomega * C_val,
                1 / (jomega * L_val),
                jomega / XC_val,
                1 / jomega * XL_val,
                Y_val,
                1 / Z_val
            ])
        else:
            # Batch case
            # freqs: [N]
            # vals: [M]
            # result: [N, M]
            omega = 2 * np.pi * freqs # [N]
            jomega = 1j * omega # [N]
            
            # Helper for broadcasting
            def bcast(val): return val.unsqueeze(0) # [1, M]
            def bcast_f(f): return f.unsqueeze(1)   # [N, 1]
            
            G_admittance = torch.cat([
                bcast(G_val).expand(freqs.size(0), -1),
                bcast(1 / R_val).expand(freqs.size(0), -1),
                bcast_f(jomega) * bcast(C_val),
                1 / (bcast_f(jomega) * bcast(L_val)),
                bcast_f(jomega) / bcast(XC_val),
                1 / bcast_f(jomega) * bcast(XL_val),
                bcast(Y_val).expand(freqs.size(0), -1),
                bcast(1 / Z_val).expand(freqs.size(0), -1)
            ], dim=1)

        G_indices_all = torch.concat(
            [G_index, R_index, C_index, L_index, XC_index, XL_index, Y_index, Z_index]
        )
        
        # 计算电压差
        v_diff = self.branch_voltage(branch_index[G_indices_all])
        
        # 计算电流 I = Y * V
        G_current = v_diff * G_admittance

        # 组装结果
        if freqs is None:
            current = torch.zeros_like(branch_index, dtype=torch.complex128)
            current[V_index_mask] = V_current
            # 注意：G_indices_all 是针对 branch_index 的索引
            current.scatter_(0, G_indices_all, G_current)
            current[I_index_mask] = I_current
        else:
            N = freqs.size(0)
            M = branch_index.size(0)
            current = torch.zeros((N, M), dtype=torch.complex128, device=self.V.device)
            
            current[:, V_index_mask] = V_current
            current[:, I_index_mask] = I_current
            
            # G_indices_all 是相对于 branch_index 的索引
            current[:, G_indices_all] = G_current

        return current
    
    def branch_voltage(self, branch_index: torch.Tensor) -> torch.Tensor:
        u = self.branch_u[branch_index] - 1
        v = self.branch_v[branch_index] - 1
        voltage_u = torch.zeros_like(branch_index, dtype=torch.complex128)
        voltage_u[u >= 0] = self.V[u[u >= 0]]
        voltage_v = torch.zeros_like(branch_index, dtype=torch.complex128)
        voltage_v[v >= 0] = self.V[v[v >= 0]]
        return voltage_u - voltage_v

    def branch_voltage_batch(self, branch_index: torch.Tensor) -> torch.Tensor:
        u = self.branch_u[branch_index] - 1
        v = self.branch_v[branch_index] - 1
        
        if self.V.dim() == 1:
            voltage_u = torch.zeros_like(branch_index, dtype=torch.complex128)
            voltage_u[u >= 0] = self.V[u[u >= 0]]
            voltage_v = torch.zeros_like(branch_index, dtype=torch.complex128)
            voltage_v[v >= 0] = self.V[v[v >= 0]]
            return voltage_u - voltage_v
        else:
            # Batch case: self.V is [N, n]
            N = self.V.shape[0]
            M = branch_index.shape[0]
            
            voltage_u = torch.zeros((N, M), dtype=torch.complex128, device=self.V.device)
            valid_u = u >= 0
            if valid_u.any():
                voltage_u[:, valid_u] = self.V[:, u[valid_u]]
            
            voltage_v = torch.zeros((N, M), dtype=torch.complex128, device=self.V.device)
            valid_v = v >= 0
            if valid_v.any():
                voltage_v[:, valid_v] = self.V[:, v[valid_v]]
            
            return voltage_u - voltage_v


class AcSimulationScipy(AcSimulation):
    def __init__(
        self,
        branch_type: torch.Tensor,
        branch_u: torch.Tensor,
        branch_v: torch.Tensor,
        branch_value: torch.Tensor,
        freq: float = 1,
    ):
        super().__init__(branch_type, branch_u, branch_v, branch_value, freq)
        self.LU: SuperLU = None

    def factorize(self):
        G = dok_matrix((self.n, self.n), dtype=np.complex128)

        branch_typ = self.branch_typ.numpy()
        branch_u = self.branch_u.numpy()
        branch_v = self.branch_v.numpy()
        branch_val = self.branch_val.numpy()

        (G_index,) = np.where(branch_typ == BranchType.G)
        G_val = branch_val[G_index]
        (R_index,) = np.where(branch_typ == BranchType.R)
        R_val = branch_val[R_index]
        (C_index,) = np.where(branch_typ == BranchType.C)
        C_val = branch_val[C_index]
        (L_index,) = np.where(branch_typ == BranchType.L)
        L_val = branch_val[L_index]
        (XC_index,) = np.where(branch_typ == BranchType.XC)
        XC_val = branch_val[XC_index]
        (XL_index,) = np.where(branch_typ == BranchType.XL)
        XL_val = branch_val[XL_index]
        (Y_index,) = np.where(branch_typ == BranchType.Y)
        Y_val = branch_val[Y_index]
        (Z_index,) = np.where(branch_typ == BranchType.Z)
        Z_val = branch_val[Z_index]

        G_val = np.r_[
            G_val,
            1 / R_val,
            2j * np.pi * self.freq * C_val,
            1 / (2j * np.pi * self.freq * L_val),
            2j * np.pi * self.freq / XC_val,
            1 / (2j * np.pi * self.freq) * XL_val,
            Y_val,
            1 / Z_val,
        ]
        G_index = np.r_[
            G_index, R_index, C_index, L_index, XC_index, XL_index, Y_index, Z_index
        ]

        G_u = branch_u[G_index] - 1
        G_v = branch_v[G_index] - 1
        G_u_unique, G_u_unique_inverse = np.unique(G_u[G_u >= 0], return_inverse=True)
        G_u_unique_val = np.zeros_like(G_u_unique, dtype=np.complex128)
        np.add.at(G_u_unique_val, G_u_unique_inverse, G_val[G_u >= 0])
        G[G_u_unique, G_u_unique] += G_u_unique_val
        G_v_unique, G_v_unique_inverse = np.unique(G_v[G_v >= 0], return_inverse=True)
        G_v_unique_val = np.zeros_like(G_v_unique, dtype=np.complex128)
        np.add.at(G_v_unique_val, G_v_unique_inverse, G_val[G_v >= 0])
        G[G_v_unique, G_v_unique] += G_v_unique_val
        G_uv = np.stack(
            [
                G_u[(G_u >= 0) & (G_v >= 0)],
                G_v[(G_u >= 0) & (G_v >= 0)],
            ]
        )
        G_uv_unique, G_uv_unique_inverse = np.unique(G_uv, axis=1, return_inverse=True)
        G_uv_unique_val = np.zeros(G_uv_unique.shape[1], dtype=np.complex128)
        np.add.at(G_uv_unique_val, G_uv_unique_inverse, G_val[(G_u >= 0) & (G_v >= 0)])
        G[G_uv_unique[0], G_uv_unique[1]] -= G_uv_unique_val
        G[G_uv_unique[1], G_uv_unique[0]] -= G_uv_unique_val

        (V_index,) = np.where(branch_typ == BranchType.V)
        V_row = self.num_nonzero_nodes + V_index
        V_u = branch_u[V_index] - 1
        G[V_u[V_u >= 0], V_row[V_u >= 0]] = 1
        G[V_row[V_u >= 0], V_u[V_u >= 0]] = 1
        V_v = branch_v[V_index] - 1
        G[V_v[V_v >= 0], V_row[V_v >= 0]] = -1
        G[V_row[V_v >= 0], V_v[V_v >= 0]] = -1

        self.LU = splu(G)

    def solve(self) -> None:
        J = np.zeros(self.n, dtype=np.complex128)

        branch_typ = self.branch_typ.numpy()
        branch_u = self.branch_u.numpy()
        branch_v = self.branch_v.numpy()
        branch_val = self.branch_val.numpy()

        (I_index,) = np.where(branch_typ == BranchType.I)
        I_val = branch_val[I_index]
        I_u = branch_u[I_index] - 1
        I_v = branch_v[I_index] - 1
        np.subtract.at(J, I_u[I_u >= 0], I_val[I_u >= 0])
        np.add.at(J, I_v[I_v >= 0], I_val[I_v >= 0])

        (V_index,) = np.where(branch_typ == BranchType.V)
        V_val = branch_val[V_index]
        V_row = self.num_nonzero_nodes + V_index
        J[V_row] = V_val

        V = self.LU.solve(J)
        self.V = torch.from_numpy(V)


class AcSimulationPardiso(AcSimulation):
    def __init__(
        self,
        branch_type: torch.Tensor,
        branch_u: torch.Tensor,
        branch_v: torch.Tensor,
        branch_value: torch.Tensor,
        freq: float = 1,
    ):
        super().__init__(branch_type, branch_u, branch_v, branch_value, freq)

        self.pardiso = AcPardisoHandle()
        acPardisoHandleInit(
            self.pardiso,
            self.branch_typ,
            self.branch_u,
            self.branch_v,
            self.branch_val,
        )

    def factorize(self):
        acPardisoHandleFactorize(
            self.pardiso,
            self.branch_typ,
            self.branch_u,
            self.branch_v,
            self.branch_val,
            self.freq,
        )

    def solve(self):
        acPardisoHandleSolve(
            self.pardiso,
            self.branch_typ,
            self.branch_u,
            self.branch_v,
            self.branch_val,
        )
        self.V = acPardisoHandleGetSolution(self.pardiso)

    def __del__(self):
        acPardisoHandleTerminate(self.pardiso)

    def export(self):
        return acPardisoHandleExport(
            self.pardiso,
            self.branch_typ,
            self.branch_u,
            self.branch_v,
            self.branch_val,
            self.freq,
        )


class AcSimulationCuDSS(AcSimulation):
    def __init__(
        self,
        branch_type: torch.Tensor,
        branch_u: torch.Tensor,
        branch_v: torch.Tensor,
        branch_value: torch.Tensor,
        freq: float = 1,
        device: str = "cuda:0",
    ):
        super().__init__(branch_type, branch_u, branch_v, branch_value, freq)
        self.device = torch.device(device)

        with torch.cuda.device(self.device):
            self.cuDss = AcCuDssHandle()
            acCuDssHandleInit(
                self.cuDss, 
                self.branch_typ, 
                self.branch_u, 
                self.branch_v, 
                self.branch_val
            )

    def factorize(self):
        with torch.cuda.device(self.device):
            acCuDssHandleFactorize(
                self.cuDss,
                # self.branch_typ,
                # self.branch_u,
                # self.branch_v,
                self.branch_val,
                self.freq,
            )

    def solve(self):
        with torch.cuda.device(self.device):
            acCuDssHandleSolve(
                self.cuDss,
                # self.branch_typ,
                # self.branch_u,
                # self.branch_v,
                self.branch_val,
            )
            self.V = acCuDssHandleGetSolution(self.cuDss)

    def solve_batch(self, frequencies: torch.Tensor):
        """
        批量计算多个频率点的解。
        frequencies: 1D Tensor of frequencies.
        Returns: Tensor of shape [num_freqs, num_nodes] containing solution vectors (on GPU).
        """
        with torch.cuda.device(self.device):
            return acCuDssHandleSolveBatch(
                self.cuDss,
                self.branch_val,
                frequencies
            )

    def solve_batch_with_rhs(self, frequencies: torch.Tensor, rhs_batch: torch.Tensor):
        """
        批量计算 (带自定义 RHS)。
        frequencies: [num_freqs]
        rhs_batch: [num_freqs, num_nodes]
        Returns: Tensor of shape [num_freqs, num_nodes]
        """
        with torch.cuda.device(self.device):
            return acCuDssHandleSolveBatchWithRhs(
                self.cuDss,
                self.branch_val,
                frequencies,
                rhs_batch
            )


    def __del__(self):
        acCuDssHandleTerminate(self.cuDss)

    def export(self):
        return acCuDssHandleExport(
            self.cuDss,
            self.branch_typ,
            self.branch_u,
            self.branch_v,
            self.branch_val,
            self.freq,
        )
