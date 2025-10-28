import yaml
import numpy as np
from Circuit import Circuit, BranchType
from math import exp, log, sqrt, pi


def build_op_ckt(file: str, file_result: str = None):
    with open(file, "r") as f:
        design = yaml.load(f.read(), Loader=yaml.FullLoader)
    if file_result is not None:
        with open(file_result, "r") as f:
            result = yaml.load(f.read(), Loader=yaml.FullLoader)
    else:
        result = None
    interposer_w = design["interposer"]["w"]
    interposer_h = design["interposer"]["h"]

    vdd = design["vdd"]
    ckt = Circuit()

    # -------------------------
    # construct PG
    # -------------------------
    s_uc = 100e-6
    w_uc = 10e-6
    t_uc = 0.7e-6
    rho_uc = 1.7e-8
    R_uc = (rho_uc * s_uc) / (t_uc * w_uc * 4)
    # horizontal
    for y in range(interposer_h):
        for x in range(interposer_w - 1):
            res_name = "r_{}_{}_{}_{}".format(x, y, x + 1, y)
            u_name = "c_{}_{}".format(x, y)
            v_name = "c_{}_{}".format(x + 1, y)
            ckt.make_branch(res_name, BranchType.R, u_name, v_name, 2 * R_uc)
    # vertical
    for y in range(interposer_h - 1):
        for x in range(interposer_w):
            res_name = "r_{}_{}_{}_{}".format(x, y, x, y + 1)
            u_name = "c_{}_{}".format(x, y)
            v_name = "c_{}_{}".format(x, y + 1)
            ckt.make_branch(res_name, BranchType.R, u_name, v_name, 2 * R_uc)

    # ----------------
    # construct load
    # ----------------
    load = np.zeros((interposer_w, interposer_h), dtype=np.float64)
    for chiplet in design["chiplets"]:
        i = chiplet["power"] / chiplet["w"] / chiplet["h"] / vdd
        for x in range(chiplet["x"], chiplet["x"] + chiplet["w"]):
            for y in range(chiplet["y"], chiplet["y"] + chiplet["h"]):
                load[x, y] += i
    for y in range(interposer_h):
        for x in range(interposer_w):
            cur_name = "i_{}_{}".format(x, y)
            u_name = "c_{}_{}".format(x, y)
            ckt.make_branch(cur_name, BranchType.I, u_name, "0", load[x, y])

    # -------------------------
    # construct tsv
    # -------------------------
    d_tsv = 20e-6
    h_tsv = 100e-6
    rho_tsv = 1.7e-8
    R_tsv = (rho_tsv * h_tsv) / (pi * (d_tsv / 2) ** 2)
    G_tsv = 1 / R_tsv
    # tsv location
    tsv_locations = [(x, y) for x in range(interposer_w) for y in range(interposer_h)]
    if result is not None and "tsvs" in result:
        tsv_locations = result["tsvs"]
    for x, y in tsv_locations:
        con_name = "g_{}_{}".format(x, y)
        u_name = "c_{}_{}".format(x, y)
        ckt.make_branch(con_name, BranchType.G, u_name, "vdd", G_tsv)

    # -------------------------
    # construct voltage source
    # -------------------------
    ckt.make_branch("vdd", BranchType.V, "vdd", "0", vdd)

    return ckt


def build_ac_ckt(file: str, file_result: str = None):
    with open(file, "r") as f:
        design = yaml.load(f.read(), Loader=yaml.FullLoader)
    if file_result is not None:
        with open(file_result, "r") as f:
            result = yaml.load(f.read(), Loader=yaml.FullLoader)
    else:
        result = None
    interposer_w = design["interposer"]["w"]
    interposer_h = design["interposer"]["h"]

    vdd = 1
    ckt = Circuit()

    # ----------------
    # construct PG
    # ----------------
    s_uc = 100e-6
    w_uc = 10e-6
    t_uc = 0.7e-6
    h_uc = 0.8e-6
    rho_uc = 1.7e-8
    dk_uc = 4.1
    R_uc = (rho_uc * s_uc) / (t_uc * w_uc * 4)
    s_uc_um, w_uc_um, h_uc_um = s_uc * 1e6, w_uc * 1e6, h_uc * 1e6
    L_uc_pH = s_uc_um * (
        0.13 * exp(-s_uc_um / 45) + 0.14 * log(s_uc_um / w_uc_um) + 0.07
    )
    L_uc = L_uc_pH * 1e-12
    Ci_uc_fF = (dk_uc * 1e-3) * (
        (44 - 28 * h_uc_um) * w_uc_um * w_uc_um
        + (280 * h_uc_um + 0.8 * s_uc_um - 64) * w_uc_um
        + 12 * s_uc_um
        - 1500 * h_uc_um
        + 1700
    )
    Cf_uc_fF = (8.854e-12 * dk_uc * 1e9) * (
        (4 * s_uc_um * w_uc_um * (log(s_uc_um / (s_uc_um - 2 * w_uc_um)) + exp(-1 / 3)))
        / (
            w_uc_um * pi
            + 2 * h_uc_um * (log(s_uc_um / (s_uc_um - 2 * w_uc_um)) + exp(-1 / 3))
        )
        + 2 * s_uc_um / pi * sqrt(2 * h_uc_um / (s_uc_um - 2 * w_uc_um))
    )
    C_uc = (Ci_uc_fF + Cf_uc_fF) * 1e-15
    # horizontal
    for y in range(interposer_h):
        for x in range(interposer_w - 1):
            res1_name = "r1_{}_{}_{}_{}".format(x, y, x + 1, y)
            res2_name = "r2_{}_{}_{}_{}".format(x, y, x + 1, y)
            ind_name = "l_{}_{}_{}_{}".format(x, y, x + 1, y)

            u_name = "c_{}_{}".format(x, y)
            n1_name = "hl_{}_{}".format(x, y)
            n2_name = "hr_{}_{}".format(x, y)
            v_name = "c_{}_{}".format(x + 1, y)

            ckt.make_branch(res1_name, BranchType.R, u_name, n1_name, R_uc)
            ckt.make_branch(ind_name, BranchType.L, n1_name, n2_name, 2 * L_uc)
            ckt.make_branch(res2_name, BranchType.R, n2_name, v_name, R_uc)
    # vertical
    for y in range(interposer_h - 1):
        for x in range(interposer_w):
            res1_name = "r1_{}_{}_{}_{}".format(x, y, x, y + 1)
            res2_name = "r2_{}_{}_{}_{}".format(x, y, x, y + 1)
            ind_name = "l_{}_{}_{}_{}".format(x, y, x, y + 1)

            u_name = "c_{}_{}".format(x, y)
            n1_name = "vl_{}_{}".format(x, y)
            n2_name = "vr_{}_{}".format(x, y)
            v_name = "c_{}_{}".format(x, y + 1)

            ckt.make_branch(res1_name, BranchType.R, u_name, n1_name, R_uc)
            ckt.make_branch(ind_name, BranchType.L, n1_name, n2_name, 2 * L_uc)
            ckt.make_branch(res2_name, BranchType.R, n2_name, v_name, R_uc)
    # capacitor
    for y in range(interposer_h):
        for x in range(interposer_w):
            cap_name = "c_{}_{}".format(x, y)
            u_name = "c_{}_{}".format(x, y)
            ckt.make_branch(cap_name, BranchType.C, u_name, "0", C_uc)

    # -----------------------------------------
    # construct microbump and 1A current source
    # -----------------------------------------
    h_bump = 10e-6
    d_bump = 20e-6
    p_bump = 100e-6
    L_bump = 1e-7 * h_bump * log(p_bump / d_bump)
    # make bump and odc
    for i, chiplet in enumerate(design["chiplets"]):
        for y in range(chiplet["y"], chiplet["y"] + chiplet["h"]):
            for x in range(chiplet["x"], chiplet["x"] + chiplet["w"]):
                ckt.make_branch(
                    "lb_{}_{}".format(x, y),
                    BranchType.L,
                    "c_{}_{}".format(x, y),
                    "die{}".format(i),
                    L_bump,
                )
        # make on-die RC model
        C_die = 9 * chiplet["power"] / vdd / vdd / chiplet["freq"]
        R_die = 0.25e-9 / C_die
        ckt.make_branch(
            "rd{}".format(i),
            BranchType.R,
            "die{}".format(i),
            "die{}_tmp".format(i),
            R_die,
        )
        ckt.make_branch(
            "cd{}".format(i), BranchType.C, "die{}_tmp".format(i), "0", C_die
        )

    # observe the first chiplet
    ckt.make_branch("id", BranchType.I, "die0", "0", 1)

    # -----------------------------------------
    # construct placed tsv
    # -----------------------------------------
    d_tsv = 20e-6
    h_tsv = 100e-6
    t_tsv = 0.5e-6
    rho_tsv = 1.7e-8
    dk_tsv = 4.1
    R_tsv = (rho_tsv * h_tsv) / (pi * (d_tsv / 2) ** 2)
    L_tsv = (1e-7) * (
        2
        * h_tsv
        * log((2 * h_tsv + sqrt((d_tsv / 2) ** 2 + (2 * h_tsv) ** 2)) / (d_tsv / 2))
        + ((d_tsv / 2) - sqrt((d_tsv / 2) ** 2 + (2 * h_tsv) ** 2))
    )
    C_tsv = 2 * pi * 8.854e-12 * dk_tsv * h_tsv / log((d_tsv + 2 * t_tsv) / d_tsv)
    # tsv location
    tsv_locations = [(x, y) for x in range(interposer_w) for y in range(interposer_h)]
    if result is not None and "tsvs" in result:
        tsv_locations = result["tsvs"]
    for x, y in tsv_locations:
        u_name = "c_{}_{}".format(x, y)
        n1_name = "t1_{}_{}".format(x, y)
        n2_name = "t2_{}_{}".format(x, y)
        n3_name = "t3_{}_{}".format(x, y)

        con1_name = "gt1_{}_{}".format(x, y)
        con2_name = "gt2_{}_{}".format(x, y)
        xind1_name = "xlt1_{}_{}".format(x, y)
        xind2_name = "xlt2_{}_{}".format(x, y)
        cap_name = "ct_{}_{}".format(x, y)

        ckt.make_branch(con1_name, BranchType.G, u_name, n1_name, 2.0 / R_tsv)
        ckt.make_branch(xind1_name, BranchType.XL, n1_name, n2_name, 2.0 / L_tsv)
        ckt.make_branch(cap_name, BranchType.C, n2_name, "0", C_tsv)
        ckt.make_branch(xind2_name, BranchType.XL, n2_name, n3_name, 2.0 / L_tsv)
        ckt.make_branch(con2_name, BranchType.G, n3_name, "0", 2.0 / R_tsv)

    # ------------------------
    # construct candidate dtc
    # ------------------------
    w_dtc = 15e-6
    h_dtc = 15e-6
    density_dtc = 1.09
    C_dtc = w_dtc * h_dtc * density_dtc
    # dtc location
    dtc_locations = [(x, y) for x in range(interposer_w) for y in range(interposer_h)]
    if result is not None and "tsvs" in result:
        dtc_locations = [
            (x, y)
            for x in range(interposer_w)
            for y in range(interposer_h)
            if (x, y) not in tsv_locations
        ]
    if result is not None and "dtcs" in result:
        dtc_locations = result["dtcs"]
    for x, y in dtc_locations:
        cap_name = "cd_{}_{}".format(x, y)
        u_name = "c_{}_{}".format(x, y)
        ckt.make_branch(cap_name, BranchType.C, u_name, "0", C_dtc)

    return ckt
