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

    def branch_voltage(self, branch_index: torch.Tensor) -> torch.Tensor:
        u = self.branch_u[branch_index] - 1
        v = self.branch_v[branch_index] - 1
        voltage_u = torch.zeros_like(branch_index, dtype=torch.complex128)
        voltage_u[u >= 0] = self.V[u[u >= 0]]
        voltage_v = torch.zeros_like(branch_index, dtype=torch.complex128)
        voltage_v[v >= 0] = self.V[v[v >= 0]]
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
    ):
        super().__init__(branch_type, branch_u, branch_v, branch_value, freq)

        self.cuDss = AcCuDssHandle()
        acCuDssHandleInit(
            self.cuDss, 
            self.branch_typ, 
            self.branch_u, 
            self.branch_v, 
            self.branch_val
        )

    def factorize(self):
        acCuDssHandleFactorize(
            self.cuDss,
            # self.branch_typ,
            # self.branch_u,
            # self.branch_v,
            self.branch_val,
            self.freq,
        )

    def solve(self):
        acCuDssHandleSolve(
            self.cuDss,
            # self.branch_typ,
            # self.branch_u,
            # self.branch_v,
            self.branch_val,
        )
        self.V = acCuDssHandleGetSolution(self.cuDss)

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
