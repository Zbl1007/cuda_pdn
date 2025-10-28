import networkx as nx
import re
import torch
from enum import IntEnum, auto
from math import pi, log, sqrt, exp
from scipy.cluster.hierarchy import DisjointSet
from typing import Union


class BranchType(IntEnum):
    V = 1
    I = 2
    G = 3
    R = 4
    C = 5
    L = 6
    XC = 7
    XL = 8
    Y = 9
    Z = 10


class Circuit:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.branch_name_map: dict[
            str, tuple[BranchType, str, str, Union[float, complex]]
        ] = {}

    def __repr__(self):
        msg = ""
        # for name, (typ, u, v, val) in self.branch_name_map.items():
        #     msg += "{}_{} {} {} {:8e}\n".format(typ.name, name, u, v, val)
        for u, v, name in self.graph.edges(keys=True):
            typ, u, v, val = self.branch_name_map[name]
            msg += "{}_{} {} {} {:8e}\n".format(typ.name, name, u, v, val)
        return msg

    def get_branches(self) -> list[str]:
        return list(self.branch_name_map.keys())

    def get_branch_type(self, name: str) -> BranchType:
        assert name in self.branch_name_map
        typ, _, _, _ = self.branch_name_map[name]
        return typ

    def get_branch_uv(self, name: str) -> tuple[str, str]:
        assert name in self.branch_name_map
        _, u, v, _ = self.branch_name_map[name]
        return u, v

    def get_branch_value(self, name: str) -> Union[float, complex]:
        assert name in self.branch_name_map
        _, _, _, val = self.branch_name_map[name]
        return val

    def clone(self):
        other = Circuit()
        other.graph = self.graph.copy()
        other.branch_name_map = self.branch_name_map.copy()
        return other

    def make_branch(
        self, name: str, typ: BranchType, u: str, v: str, val: Union[float, complex]
    ):
        assert name not in self.branch_name_map
        self.graph.add_edge(u, v, key=name)
        self.branch_name_map[name] = (typ, u, v, val)

    def delete_branch(self, name: str):  # make branch open
        assert name in self.branch_name_map
        _, u, v, _ = self.branch_name_map.pop(name)
        self.graph.remove_edge(u, v, name)

    def merge_nodes(self, u: str, v: str):  # make branch short
        if u == v:
            return
        uv_branches = set()
        from_v_branches = set()
        to_v_branches = set()
        for _v, _u, name in self.graph.edges(v, keys=True):
            if _u == u:
                uv_branches.add(name)
            else:
                from_v_branches.add(name)
        for _u, _v, name in self.graph.in_edges(v, keys=True):
            if _u == u:
                uv_branches.add(name)
            else:
                to_v_branches.add(name)

        # from_v_branches -> from_u_branches
        for name in from_v_branches:
            typ, _v, _u, val = self.branch_name_map[name]
            self.graph.remove_edge(_v, _u, name)
            self.graph.add_edge(u, _u, key=name)
            self.branch_name_map[name] = (typ, u, _u, val)
        # to_v_branches -> to_u_branches
        for name in to_v_branches:
            typ, _u, _v, val = self.branch_name_map[name]
            self.graph.remove_edge(_u, _v, name)
            self.graph.add_edge(_u, u, key=name)
            self.branch_name_map[name] = (typ, _u, u, val)
        # delete uv_branches
        for name in uv_branches:
            self.branch_name_map.pop(name)
        self.graph.remove_node(v)

    def make_subcircuit(self, other, prefix: str, link: list[tuple[str, str]]):
        for name, (typ, u, v, val) in other.branch_name_map.items():
            self.make_branch(prefix + name, typ, prefix + u, prefix + v, val)
        for u, v in link:
            self.merge_nodes(u, prefix + v)

    def to_op(self):
        # make inductor short
        nodes_qset = DisjointSet(list(self.graph.nodes))
        for bname, (_, u, v, _) in self.branch_name_map.items():
            if self.get_branch_type(bname) == BranchType.L:
                nodes_qset.merge(u, v)
        for subset in nodes_qset.subsets():
            ss = sorted(list(subset))
            n1 = ss[0]
            for n2 in ss[1:]:
                self.merge_nodes(n1, n2)

        # make capacitors open
        branches_to_delete = []
        for bname in self.branch_name_map.keys():
            if self.get_branch_type(bname) == BranchType.C:
                branches_to_delete.append(bname)
        for bname in branches_to_delete:
            self.delete_branch(bname)

    def prepare_sim(
        self,
        gnd: str,
        branches_to_return_index: list[str] = [],
        is_complex: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_name_map = {gnd: 0}
        branch_typ = []
        branch_u = []
        branch_v = []
        branch_val = []
        index_map = {name: -1 for name in branches_to_return_index}
        for i, (name, (typ, u, v, val)) in enumerate(self.branch_name_map.items()):
            branch_typ.append(self.get_branch_type(name))
            if u not in node_name_map:
                node_name_map[u] = len(node_name_map)
            if v not in node_name_map:
                node_name_map[v] = len(node_name_map)
            branch_u.append(node_name_map[u])
            branch_v.append(node_name_map[v])
            branch_val.append(val)
            if name in index_map:
                index_map[name] = i
        index = []
        for name in branches_to_return_index:
            if not (index_map[name] >= 0):
                raise RuntimeError("can not find branch `{}`".format(name))
            index.append(index_map[name])

        branch_typ = torch.tensor(branch_typ, dtype=torch.int)
        branch_u = torch.tensor(branch_u, dtype=torch.int)
        branch_v = torch.tensor(branch_v, dtype=torch.int)
        if is_complex:
            branch_val = torch.tensor(branch_val, dtype=torch.complex128)
        else:
            branch_val = torch.tensor(branch_val, dtype=torch.float64)
        branch_index = torch.tensor(index, dtype=torch.long)

        # sort by branch type
        order = torch.argsort(branch_typ, stable=True)
        branch_typ = branch_typ[order]
        branch_u = branch_u[order]
        branch_v = branch_v[order]
        branch_val = branch_val[order]
        branch_perm = torch.empty_like(order, dtype=int)
        branch_perm[order] = torch.arange(branch_perm.shape[0])
        branch_index = branch_perm[branch_index]

        return branch_typ, branch_u, branch_v, branch_val, branch_index


class Spice(Circuit):
    def __init__(self, file: str):
        super().__init__()

        def name_to_typ(name: str):
            match name[0].upper():
                case "R":
                    return BranchType.R
                case "C":
                    return BranchType.C
                case "I":
                    return BranchType.I
                case "L":
                    return BranchType.L
                case "V":
                    return BranchType.V

        def str_to_float(value_str: str):
            unit_map = {
                "f": 1e-15,
                "p": 1e-12,
                "n": 1e-9,
                "u": 1e-6,
                "m": 1e-3,
                "k": 1e3,
                "meg": 1e6,
                "g": 1e9,
                "t": 1e12,
            }
            value = 1
            for k, v in unit_map.items():
                if value_str[-len(k) :].lower() == k:
                    value_str = value_str[: -len(k)]
                    value = v
                    break
            value *= float(value_str)
            return value

        VALUE = R"(-?\d+)(\.\d+)?((e(\+|-)?\d+)|(meg|k|m|u|n|p|f|t|g))?"
        patterns = [
            re.compile(R"^(R\w+) (\w+) (\w+) R=({})".format(VALUE), re.IGNORECASE),
            re.compile(R"^(R\w+) (\w+) (\w+) ({})".format(VALUE), re.IGNORECASE),
            re.compile(R"^(I\w+) (\w+) (\w+) DC ({})".format(VALUE), re.IGNORECASE),
            re.compile(R"^(I\w+) (\w+) (\w+) ({})".format(VALUE), re.IGNORECASE),
            re.compile(R"^(V\w+) (\w+) (\w+) DC ({})".format(VALUE), re.IGNORECASE),
            re.compile(R"^(V\w+) (\w+) (\w+) ({})".format(VALUE), re.IGNORECASE),
        ]
        with open(file, "r") as f:
            for line in f:
                for p in patterns:
                    match_result = p.match(line)
                    if match_result is None:
                        continue
                    name = match_result.group(1).lower()
                    u = match_result.group(2).lower()
                    v = match_result.group(3).lower()
                    val = str_to_float(match_result.group(4))
                    self.make_branch(name, name_to_typ(name), u, v, val)


class TSV(Circuit):
    def __init__(
        self,
        d: float = 20e-6,
        h: float = 100e-6,
        t: float = 0.5e-6,
        rho: float = 1.7e-8,
        dk: float = 4.1,
    ):
        """
        @param d diameter
        @param h height
        @param t thickness of the insulator surrounding conductor
        @param rho resistivy of the conductor
        @param dk relative permittivity of the insulator
        """

        super().__init__()

        R = (rho * h) / (pi * (d / 2) ** 2)
        u0 = 4 * pi * 1e-7
        L = (u0 / (4 * pi)) * (
            2 * h * log((2 * h + sqrt((d / 2) ** 2 + (2 * h) ** 2)) / (d / 2))
            + ((d / 2) - sqrt((d / 2) ** 2 + (2 * h) ** 2))
        )
        e0 = 8.854e-12
        C = 2 * pi * e0 * dk * h / log((d + 2 * t) / d)

        # R = 5.41m
        # L = 54.7p
        # C = 0.467p

        self.make_branch("r0", BranchType.R, "bottom", "n1", R / 2)
        self.make_branch("l0", BranchType.L, "n1", "n2", L / 2)
        self.make_branch("c0", BranchType.C, "n2", "0", C)
        self.make_branch("l1", BranchType.L, "n2", "n3", L / 2)
        self.make_branch("r1", BranchType.R, "n3", "top", R / 2)


class TSV_OP(Circuit):
    def __init__(
        self,
        d: float = 20e-6,
        h: float = 100e-6,
        t: float = 0.5e-6,
        rho: float = 1.7e-8,
        dk: float = 4.1,
    ):
        super().__init__()
        R = (rho * h) / (pi * (d / 2) ** 2)
        self.make_branch("r0", BranchType.R, "bottom", "top", R)


class UC(Circuit):
    def __init__(
        self,
        s: float = 100e-6,
        w: float = 10e-6,
        t: float = 0.7e-6,
        h: float = 0.8e-6,
        rho: float = 1.7e-8,
        dk: float = 4.1,
    ):
        """
        @param s pitch of adjacent power wire
        @param w width of power wire
        @param t thickness of wire
        @param h thickness of dielectric between P/G planes
        @param rho resistivy of metal
        @param dk relative permittivity of dielectric
        """
        super().__init__()

        R = (rho * s) / (t * w * 4)
        s_um, w_um, h_um = s * 1e6, w * 1e6, h * 1e6
        L_pH = s_um * (0.13 * exp(-s_um / 45) + 0.14 * log(s_um / w_um) + 0.07)
        L = L_pH * 1e-12
        Ci_fF = (dk * 1e-3) * (
            (44 - 28 * h_um) * w_um * w_um
            + (280 * h_um + 0.8 * s_um - 64) * w_um
            + 12 * s_um
            - 1500 * h_um
            + 1700
        )
        e0 = 8.854e-12
        Cf_fF = (e0 * dk * 1e9) * (
            (4 * s_um * w_um * (log(s_um / (s_um - 2 * w_um)) + exp(-1 / 3)))
            / (w_um * pi + 2 * h_um * (log(s_um / (s_um - 2 * w_um)) + exp(-1 / 3)))
            + 2 * s_um / pi * sqrt(2 * h_um / (s_um - 2 * w_um))
        )
        C = (Ci_fF + Cf_fF) * 1e-15

        # R = 60.7m
        # L = 40.6p
        # C = 30.1f

        self.make_branch("c0", BranchType.C, "c", "0", C)
        for i, n1 in enumerate(["w", "e", "n", "s"]):
            n2 = "c" + n1
            self.make_branch("l{}".format(i), BranchType.L, n1, n2, L)
            self.make_branch("r{}".format(i), BranchType.R, n2, "c", R)


class UC_OP(Circuit):
    def __init__(
        self,
        s: float = 100e-6,
        w: float = 10e-6,
        t: float = 0.7e-6,
        h: float = 0.8e-6,
        rho: float = 1.7e-8,
        dk: float = 4.1,
    ):
        """
        @param s pitch of adjacent power wire
        @param w width of power wire
        @param t thickness of wire
        @param h thickness of dielectric between P/G planes
        @param rho resistivy of metal
        @param dk relative permittivity of dielectric
        """
        super().__init__()

        R = (rho * s) / (t * w)

        for i, n1 in enumerate(["w", "e", "n", "s"]):
            self.make_branch("r{}".format(i), BranchType.R, "c", n1, R)


class PG(Circuit):
    def __init__(
        self,
        x: int = 1,
        y: int = 1,
        s: float = 100e-6,
        w: float = 10e-6,
        t: float = 0.7e-6,
        h: float = 0.8e-6,
        rho: float = 1.7e-8,
        dk: float = 4.1,
    ):
        """
        construct PG planes using x * y UCs
        """
        super().__init__()
        uc = UC(s, w, t, h, rho, dk)
        for i in range(y):
            for j in range(x):
                c = "c{}_{}".format(i + 1, j + 1)
                w = "h{}_{}".format(i + 1, j)
                e = "h{}_{}".format(i + 1, j + 1)
                n = "v{}_{}".format(i, j + 1)
                s = "v{}_{}".format(i + 1, j + 1)
                prefix = "uc{}_{}_".format(i + 1, j + 1)
                self.make_subcircuit(
                    uc,
                    prefix,
                    [("0", "0"), (c, "c"), (w, "w"), (e, "e"), (n, "n"), (s, "s")],
                )


class PG_OP(Circuit):
    def __init__(
        self,
        x: int = 1,
        y: int = 1,
        s: float = 100e-6,
        w: float = 10e-6,
        t: float = 0.7e-6,
        h: float = 0.8e-6,
        rho: float = 1.7e-8,
        dk: float = 4.1,
    ):
        """
        construct PG planes using x * y UCs
        """
        super().__init__()
        uc = UC_OP(s, w, t, h, rho, dk)
        for i in range(y):
            for j in range(x):
                c = "c{}_{}".format(i + 1, j + 1)
                w = "h{}_{}".format(i + 1, j)
                e = "h{}_{}".format(i + 1, j + 1)
                n = "v{}_{}".format(i, j + 1)
                s = "v{}_{}".format(i + 1, j + 1)
                prefix = "uc{}_{}_".format(i + 1, j + 1)
                self.make_subcircuit(
                    uc,
                    prefix,
                    [(c, "c"), (w, "w"), (e, "e"), (n, "n"), (s, "s")],
                )


class Pacakge(Circuit):
    def __init__(self):
        super().__init__()
        R = 30e-3
        L = 0.5e-9
        C = 100e-9
        self.make_branch("r0", BranchType.R, "bottom", "n1", R)
        self.make_branch("l0", BranchType.L, "n1", "top", L)
        self.make_branch("c0", BranchType.C, "top", "0", C)


class DTC(Circuit):
    def __init__(self, size: float = 15e-6 * 15e-6):
        super().__init__()
        C = 1.09 * size
        self.make_branch("c0", BranchType.C, "top", "0", C)


class MicroBump(Circuit):
    def __init__(
        self, height: float = 10e-6, diameter: float = 20e-6, pitch: float = 100e-6
    ):
        super().__init__()
        # L = 1/2 * u0/(2*pi) * height * ln(pitch/diameter)
        L = 1e-7 * height * log(pitch / diameter)
        self.make_branch("l0", BranchType.L, "top", "bottom", L)


if __name__ == "__main__":
    TSV()
    exit()
    # construct op
    x = 20
    y = 15
    vdd = 1.8
    pg_op = PG_OP(x, y)
    tsv_op = TSV_OP()
    ckt_op = pg_op.clone()
    ckt_op.make_branch("vdd", BranchType.V, "vdd", "0", vdd)
    for i in range(y):
        for j in range(x):
            ckt_op.make_subcircuit(
                tsv_op,
                "tsv{}_{}_".format(i + 1, j + 1),
                [("c{}_{}".format(i + 1, j + 1), "top"), ("vdd", "bottom")],
            )

    chiplets = [(1, 3, 9, 12, 1.5, 3.8e9), (10, 3, 19, 12, 0.2, 2.4e9)]
    for lx, ly, hx, hy, power, freq in chiplets:
        i_per_uc = power / (hx - lx) / (hy - ly) / vdd
        for i in range(ly, hy):
            for j in range(lx, hx):
                ckt_op.make_branch(
                    "iload{}_{}".format(i + 1, j + 1),
                    BranchType.I,
                    "c{}_{}".format(i + 1, j + 1),
                    "0",
                    i_per_uc,
                )

    # construct ac
    pg = PG(x, y)
    tsv = TSV()
    microbump = MicroBump()
    ckt = pg.clone()
    chiplets = [(1, 3, 9, 12, 1.5, 3.8e9), (10, 3, 19, 12, 0.2, 2.4e9)]
    for k, (lx, ly, hx, hy, power, freq) in enumerate(chiplets):
        for i in range(ly, hy):
            for j in range(lx, hx):
                ckt.make_subcircuit(
                    microbump,
                    "micro{}_{}".format(i + 1, j + 1),
                    [
                        ("c{}_{}".format(i + 1, j + 1), "bottom"),
                        ("die{}_sink".format(k), "top"),
                    ],
                )
        C_ODC = 9 * power / vdd / vdd / freq
        ESR = 0.25 / C_ODC
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
