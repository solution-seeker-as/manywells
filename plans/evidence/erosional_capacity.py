"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 05 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The most a tubing of D-25's list carries at the erosional velocity, for the sampler redesign's checkpoint (D1, D2;
plans/sampler-redesign/checkpoint.md). D-25 picks the smallest tubing whose no-slip mixture velocity at the design
rate stays below v_e = 1.22 C / sqrt(rho_m) m/s, with C from 100 to 150 (API RP 14E as quoted in
plans/sampler-redesign/sources-wells-production.md, Section 6). A design rate above the largest tubing's capacity
is then a well that no tubing on the list can carry.

Gas wells: single-phase gas at the wellhead, Z from Papay's explicit correlation with Sutton's pseudo-critical
properties, an estimate good to a few percent here. Oil wells: a no-slip mixture of density rho_m, as a volume rate
at the wellhead.

    .venv/bin/python plans/evidence/erosional_capacity.py
"""

import numpy as np

R, M_AIR, P_SC, T_SC = 8.314, 0.02897, 1.01325e5, 288.15
TUBING_ID = {'2-7/8': 62.0, '3-1/2': 76.0, '4-1/2': 100.5, '5-1/2': 124.3, '7': 157.1, '9-5/8': 220.5}  # mm, D-25
P_WH = (30, 50, 100, 150, 200, 300)  # bar


def z_papay(p_bar, T, gg):
    tpc = (169.2 + 349.5 * gg - 74.0 * gg**2) / 1.8  # K
    ppc = (756.8 - 131.0 * gg - 3.6 * gg**2) * 0.0689476  # bar
    ppr, tpr = p_bar / ppc, T / tpc
    return 1 - 3.52 * ppr / 10 ** (0.9813 * tpr) + 0.274 * ppr**2 / 10 ** (0.8157 * tpr)


def gas_capacity(d_mm, p_bar, T, gg, C):
    """Sm3/d of gas at the erosional velocity in a tubing of inner diameter d_mm."""
    M = gg * M_AIR
    rho = p_bar * 1e5 * M / (z_papay(p_bar, T, gg) * R * T)
    w = rho * 1.22 * C / np.sqrt(rho) * np.pi / 4 * (d_mm / 1000) ** 2
    return w / (P_SC * M / (R * T_SC)) * 86400


def mixture_capacity(d_mm, rho_m, C):
    """m3/d of no-slip mixture at the wellhead at the erosional velocity."""
    return 1.22 * C / np.sqrt(rho_m) * np.pi / 4 * (d_mm / 1000) ** 2 * 86400


if __name__ == '__main__':
    print('Gas, gas gravity 0.65, wellhead 330 K: million Sm3/d at the erosional velocity')
    for C in (100, 150):
        for name in ('7', '9-5/8'):
            caps = ', '.join(f'{p} bar {gas_capacity(TUBING_ID[name], p, 330.0, 0.65, C) / 1e6:.1f}' for p in P_WH)
            print(f'  C = {C}, {name} in ({TUBING_ID[name]} mm): {caps}')
    print('Oil wells, no-slip mixture at the wellhead, 7 in: m3/d of mixture at the erosional velocity')
    for C in (100, 150):
        caps = ', '.join(
            f'rho_m {rho} kg/m3 {mixture_capacity(TUBING_ID["7"], rho, C):,.0f}' for rho in (300, 500, 700)
        )
        print(f'  C = {C}: {caps}')
