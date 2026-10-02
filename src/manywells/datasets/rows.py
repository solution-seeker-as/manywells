"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The dataset row of a solved sample (specs/sampling.md, SMP-30; docs/datasets.md), and the features of a root, which
calibration predicts its observations from (specs/calibration.md, CAL-8).
"""

from manywells.datasets.schema import FEATURES, SECONDS_PER_HOUR
from manywells.units import P_REF, T_REF


def root_features(root, bc, rho_o, rho_w, rho_g, x_w) -> dict:  # spec: SMP-30
    """
    The features of a root (the operating point) at boundary conditions bc, but the inflow fractions.

    The rates are the reservoir's at standard conditions: the liquid w_res, split into oil and water by the water's
    mass fraction of the liquid x_w, and the reservoir gas, with the densities at standard conditions rho_o, rho_w and
    rho_g (kg/m³). Without dissolved gas they are v1.0.0's rates at the wellhead, to rounding; with it, the free gas
    at the wellhead also holds gas that left the oil, and the stock-tank rates are the reservoir's. TBH is the root's
    bottomhole temperature, T_r in v1.0.0 (THM-3).
    """
    X = root.state
    w_l = root.w_res
    w_g = root.w_g_res
    w_o, w_w = (1 - x_w) * w_l, x_w * w_l
    q_o, q_w = w_o / rho_o, w_w / rho_w
    q_g, q_lg = w_g / rho_g, bc.w_lg / rho_g
    h = SECONDS_PER_HOUR
    return {
        'CHK': bc.u, 'PBH': float(X[0, 0]), 'PWH': float(X[-1, 0]), 'PDC': bc.p_s,
        'TBH': float(X[0, 6]), 'TWH': float(X[-1, 6]),
        'WGL': bc.w_lg, 'WGAS': w_g, 'WLIQ': w_l, 'WOIL': w_o, 'WWAT': w_w, 'WTOT': w_g + bc.w_lg + w_l,
        'QGL': h * q_lg, 'QGAS': h * q_g, 'QLIQ': h * (q_o + q_w), 'QOIL': h * q_o, 'QWAT': h * q_w,
        'QTOT': h * (q_g + q_lg + q_o + q_w),
        'CHOKED': bool(root.choked), 'FRBH': root.flow_regime[0], 'FRWH': root.flow_regime[-1],
    }


def sample_row(root, draw, fractions, bc) -> dict:  # spec: SMP-30
    """
    The features of one sample: a root (the operating point) of the well of `draw` at the fractions and boundary
    conditions of the sample (root_features), with the fractions. The gas's density at standard conditions is
    PVT-GAS-2's, from the draw's specific gas constant.
    """
    f_g, f_o, f_w = fractions
    row = root_features(root, bc, draw.rho_o, draw.rho_w, P_REF / (draw.R_s * T_REF), f_w / (f_o + f_w))
    row |= {'FGAS': f_g, 'FOIL': f_o, 'FWAT': f_w}
    return {k: row[k] for k in FEATURES}
