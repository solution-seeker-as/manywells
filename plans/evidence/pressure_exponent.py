"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The exponent of the life's pressure path for oils under depletion (SMP-85; D-115 in plans/sampler-redesign/
decisions.md): p_r = p_i (r_1 + (r_0 - r_1)(1 - F)^gamma), with F the cumulative oil over the life's, fitted to the
material balance of each drawn oil from the life's start, r_0 p_i, to its end, r_1 p_i.

The material balance (Craft and Hawkins 1991, Applied petroleum reservoir engineering, 2nd ed., Ch. 9):
- Below the bubble point, Muskat's equation (Eq. 9.34) on the well's own black oil at T_r, with Fig. 9.16's k_g/k_o
  in the core (the tail's blend as a sensitivity), as solution_gas_drive.py integrates it: N_p/N_b = 1 - s B_ob/B_o.
- Above it, the undersaturated oil's expansion, N_p B_o = N (B_o - B_oi) (Eq. 9.6 with no water and a rigid pore
  space). The model's black oil has no compressibility above its bubble point (PVT-OIL-7 caps R_s, so B_o stays), so
  B_o there is Vazquez and Beggs's (1980, Journal of Petroleum Technology 32(6), 968-970), the same paper as the
  model's R_s and B_o: c_o = A/p, A = (5 R_sb + 17.2 T - 1180 gamma_gs + 12.61 API - 1433)/1e5 in field units, so
  B_o = B_ob (p_b/p)^A. A sensitivity adds the pore space's and the connate water's expansion,
  N_p B_o = N [B_o - B_oi + B_oi c_pw (p_i - p)], c_pw = (c_w S_w + c_f)/(1 - S_w), at c_f = 4e-6 and c_w = 3e-6 1/psi
  and S_w = 0.20, own assumptions of typical values, not from a source in hand.
- The two join at the bubble point: N_p/N = RF_b + (1 - RF_b) N_p/N_b below it.

The wells are the sampler's own (manywells.sampling.v2): the core black oils under depletion of the first wells of a
seed, each with its life's start and end (life.Life); their saturation pressure follows SMP-68's undersaturation, 0
with probability 0.20 and else LogN(0.10, 1.2). gamma is fitted by least squares in the pressure at 201 values of F
from 0 to 1.

    .venv/bin/python plans/evidence/pressure_exponent.py [N_WELLS] [SEED]
"""

import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from manywells.pvt.black_oil import BlackOil
from manywells.sampling.v2.life import Life
from manywells.units import CF_PSI, CF_RS
from solution_gas_drive import X_GRID, black_oil_props, kgko_blend, kgko_fig_9_16, muskat, pct

C_F, C_W, S_W = 4e-6, 3e-6, 0.20  # 1/psi, 1/psi, connate water: the sensitivity's own assumptions
F_FIT = np.linspace(0.0, 1.0, 201)


def drawn_oils(n, seed):
    """The core black oils under depletion among the first n wells of the seed, with their lives' start and end."""
    rows = []
    for k in range(n):
        life = Life(seed, k)
        w0 = life.w0
        if w0['coverage_tail'] == 'core' and w0['fluid_class'] == 'black_oil' and w0['drive'] == 'depletion':
            prm = life.parameters()
            rows.append(
                {key: w0[key] for key in ('well', 'rho_o', 'rho_g', 'p_sep', 'T_sep', 'T_r', 'p_i', 'p_sat', 'R_i')}
                | dict(api=w0['api'], r_0=prm['r_p_start'], r_1=prm['r_p_end'], gamma_drawn=prm['gamma_p'])
            )
    return pd.DataFrame(rows)


def vazquez_beggs_a(w):
    """Vazquez and Beggs's A of c_o = A/p above the bubble point, from the well's black oil."""
    oil = BlackOil(rho_o=w.rho_o, rho_g=w.rho_g, gas_law='dak', p_sep=w.p_sep, T_sep=w.T_sep)._black_oil
    T_F = (w.T_r - 273.15) * 1.8 + 32
    R_sb = w.R_i / CF_RS  # scf/STB
    return (5 * R_sb + 17.2 * T_F - 1180 * oil.sg_gas_corr + 12.61 * oil.api - 1433) / 1e5


def fit_gamma(p, F, p_0, p_1):
    """gamma of (p - p_1)/(p_0 - p_1) = (1 - F)^gamma, least squares at F_FIT; p and F along the path, F rising."""
    y = np.interp(F_FIT, F, (p - p_1) / (p_0 - p_1))
    res = minimize_scalar(lambda g: np.sum((y - (1 - F_FIT) ** g) ** 2), bounds=(0.05, 10.0), method='bounded')
    return res.x


def gammas(d, kgko, pore):
    """The fitted exponent of each well, with the k_g/k_o curve kgko, and the pore space's expansion or not."""
    props = black_oil_props(d)
    a = props
    _, _, rec_b = muskat(a['p'], a['Rs'], a['Bo'], a['Bg'], a['mu_r'], a['dRs'], a['dBo'], a['dBg'], kgko)
    out = []
    for j, w in enumerate(d.itertuples()):
        p_b, B_ob = w.p_sat, a['Bo'][j, 0]
        p_0, p_1 = w.r_0 * w.p_i, w.r_1 * w.p_i
        A = vazquez_beggs_a(w)

        def rf_above(p, w=w, A=A, B_ob=B_ob):  # N_p/N from p_i, above the bubble point
            B_oi, B_o = B_ob * (p_b / w.p_i) ** A, B_ob * (p_b / p) ** A
            extra = B_oi * (C_W * S_W + C_F) / (1 - S_W) * (w.p_i - p) * 1e5 / CF_PSI if pore else 0.0
            return (B_o - B_oi + extra) / B_o

        rf_b = rf_above(p_b)
        p_below = X_GRID[::2] * p_b
        rf_below = rf_b + (1 - rf_b) * rec_b[j]
        p_above = np.linspace(max(p_0, p_b), p_b, 200)[:-1] if p_0 > p_b else np.empty(0)
        p_path = np.concatenate([p_above, p_below])
        rf = np.concatenate([[rf_above(p) for p in p_above], rf_below])
        keep = (p_path <= p_0 + 1e-9) & (p_path >= p_1 - 1e-9)
        rf_0, rf_1 = np.interp(-p_0, -p_path, rf), np.interp(-p_1, -p_path, rf)
        p_k = np.concatenate([[p_0], p_path[keep], [p_1]])
        F = (np.concatenate([[rf_0], rf[keep], [rf_1]]) - rf_0) / (rf_1 - rf_0)
        F, idx = np.unique(F, return_index=True)
        out.append(fit_gamma(p_k[idx], F, p_0, p_1))
    return np.array(out)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    d = drawn_oils(n, seed)
    x = 1 - d.p_sat / d.p_i
    sat = x < 1e-12
    start_above = d.r_0 * d.p_i > d.p_sat
    print(
        f'{len(d)} core black oils under depletion of {n} wells (seed {seed}): {sat.sum()} saturated at discovery, '
        f'{(~sat).sum()} undersaturated (x 5/50/95%: {pct(x[~sat], fmt=".3f")}); the life starts above the bubble '
        f'point in {start_above.sum()}'
    )
    print('gamma by the material balance, 5th/50th/95th percentiles:')
    cases = (
        ("Fig. 9.16's k_g/k_o (the core), the oil's expansion above p_b", kgko_fig_9_16, False),
        ('the same, with the pore space and connate water (sensitivity)', kgko_fig_9_16, True),
        ("the tail's blend at w = 0.5, the oil's expansion (sensitivity)", kgko_blend(0.5), False),
    )
    for label, kgko, pore in cases:
        g = gammas(d, kgko, pore)
        print(f'  {label}:')
        print(f'    all {pct(g)}; saturated at discovery {pct(g[sat])}; undersaturated {pct(g[~sat])}; ')
        print(f'    life starting above p_b {pct(g[start_above])}, below it {pct(g[~start_above])}')
        if not pore and kgko is kgko_fig_9_16:
            core = g
    print(f'The core case: 5th and 95th percentiles {np.percentile(core, 5):.3f} and {np.percentile(core, 95):.3f}')


if __name__ == '__main__':
    main()
