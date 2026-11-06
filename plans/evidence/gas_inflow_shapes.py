"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The shapes of a gas well's inflow on the sampler redesign's gas and condensate wells (Step R5 of
plans/sampler-redesign-plan.md; the draft feature spec specs/features/022-gas-rate-inflow.md). It draws R3's wells
with plans/evidence/sampler_draft_draws.py, without solving them, and for each gas or condensate well compares the
separator-gas rate at a few bottomhole pressures p_wf from these inflows:

    pseudo     the reference: Forchheimer's law in the real-gas pseudo-pressure m(p) = 2 int p/(mu Z) dp
               (Golan and Whitson 1991, Eqs. 2.47 and 2.57; Economides et al. 1994, Eq. 4-54), with R3's k h, skin and
               non-Darcy coefficient, and Z and mu_g from the model's black-oil gas (Dranchuk-Abou-Kassem and
               Lee-Gonzalez-Eakin) at T_r. In R3's units it is (mu Z)_r (m(p_r) - m(p_wf)) = a q + b q^2.
    p2         R3's draw: p_r^2 - p_wf^2 = a q + b q^2 with mu Z at p_r (Golan and Whitson 1991, Eq. 2.51).
    p2_fit     the same form with a and b scaled by one factor so that it gives pseudo's rate at the design drawdown.
    p2_test    the same form with a and b from a four-point test: (p_r^2 - p_wf^2)/q against q, a straight line
               fitted by least squares to pseudo at p_wf/p_r = 0.95, 0.90, 0.85 and 0.80 (Golan and Whitson 1991,
               Fig. 2.13), and extrapolated.
    p2_ls      the same form with a and b fitted by least squares to pseudo at all the points below, the best that
               constant coefficients in p^2 can do over the whole curve.
    backpr     the backpressure law q = C (p_r^2 - p_wf^2)^n (Rawlins and Schellhardt 1936), C and n fitted on log-log
               axes to the same four points, as such a test would be read, and extrapolated.
    vogel      Vogel's curve, q = q_max (1 - 0.2 r - 0.8 r^2), r = p_wf/p_r, through pseudo's design point.
    interim    R3's interim mapping: a productivity index on the liquid, so the gas rate is linear in the drawdown,
               through p2's design point (sampling-draft.md, Inflow and Mapping).

Each is given as the ratio to pseudo, at p_wf/p_r = 0.95, 0.9, 0.8, 0.6, 0.4 and 0.2 and at p_wf = 1.01325 bar (the
absolute open flow, AOF); interim also against p2, which it stands in for. It also prints the reservoir pressures,
the share of the rate's pressure drop that is non-Darcy at the design point, the backpressure exponents a four-point
test would give, the factor between a condensate's wellstream and its separator gas, and R3's non-Darcy coefficient
next to Economides et al.'s estimate from k, h and mu (1994, Eq. 4-34).

From the repository root:

    (ulimit -v 4000000; OMP_NUM_THREADS=1 timeout 900 .venv/bin/python plans/evidence/gas_inflow_shapes.py 2000)

    --seed N     the dataset seed (default 2026)
    --csv OUT    write one row per gas well
"""

import argparse
import math

import numpy as np
import pandas as pd
from sampler_draft_draws import P_ATM, draw_well, gas_rate

from manywells.pvt.black_oil import BlackOil

R_POINTS = (0.95, 0.9, 0.8, 0.6, 0.4, 0.2)  # p_wf / p_r
TEST_POINTS = (0.95, 0.90, 0.85, 0.80)  # a four-point test's p_wf / p_r
CURVES = ('p2', 'p2_fit', 'p2_test', 'p2_ls', 'backpr', 'vogel', 'interim')
R_W_FT = 0.108 / 0.3048  # R3's wellbore radius (ft)
P2_LIMIT = 2000 * 0.0689476  # bar: 2000 psia, below which mu Z is about constant (Golan and Whitson 1991, p. 147)


def pseudo_pressure(bo, p_r, T_r, n=400):
    """m(p) = 2 int_1^p p/(mu Z) dp on a grid from 1 bar to p_r (bar2/(Pa s)), and mu Z at p_r (Pa s)."""
    p = np.linspace(1.0, p_r, n)
    muz = np.array([bo.gas_viscosity(x, T_r, bo.gas_density(x, T_r)) * bo.z_factor(x, T_r) for x in p])
    f = 2 * p / muz
    m = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(p))])
    return p, m, muz[-1]


def vogel(r):
    return 1 - 0.2 * r - 0.8 * r**2


def curves(w):
    """Rates (Sm3/d of separator gas) of each inflow at R_POINTS and at the AOF, for one of R3's gas wells."""
    p_r, T_r, a, b, dd = w['p_r'], w['T_r'], w['a_gas'], w['b_gas'], w['delta_d']
    bo = BlackOil(rho_o=w['rho_o'], rho_g=w['rho_g'], gas_law='dak', p_sep=w['p_sep'], T_sep=w['T_sep'])
    p, m, muz_r = pseudo_pressure(bo, p_r, T_r)
    p_d = (1 - dd) * p_r

    def q_pseudo(p_wf):
        return gas_rate(a, b, muz_r * (m[-1] - np.interp(p_wf, p, m)))

    q_d = q_pseudo(p_d)
    scale = (p_r**2 - p_d**2) / (muz_r * (m[-1] - np.interp(p_d, p, m)))  # p2_fit: a and b times this
    dp2_t = np.array([p_r**2 - (f * p_r) ** 2 for f in TEST_POINTS])
    q_t = np.array([q_pseudo(f * p_r) for f in TEST_POINTS])
    n_bp, ln_c = np.polyfit(np.log(dp2_t), np.log(q_t), 1)
    b_t, a_t = np.polyfit(q_t, dp2_t / q_t, 1)  # the four-point test's straight line: intercept a, slope b
    b_t = max(b_t, 1e-300)  # gas_rate divides by b
    dp2_all = np.array([p_r**2 - (f * p_r) ** 2 for f in R_POINTS] + [p_r**2 - P_ATM**2])
    q_all = np.array([q_pseudo(p_r * f) for f in R_POINTS] + [q_pseudo(P_ATM)])
    b_ls, a_ls = np.polyfit(q_all, dp2_all / q_all, 1)
    b_ls = max(b_ls, 1e-300)
    mu_cp = 1000 * bo.gas_viscosity(p_r, T_r, bo.gas_density(p_r, T_r))
    h_ft = w['h_net'] * w['f_geo'] / 0.3048
    d_eq = 6e-5 * w['sg_gas'] * w['perm'] ** -0.1 * h_ft / (mu_cp * R_W_FT * h_ft**2)  # d/Mscf, perforated over h
    q_p2_d = gas_rate(a, b, p_r**2 - p_d**2)
    out = {'n_backpr': n_bp, 'q_d': q_d, 'nondarcy_share': b * q_d / (a + b * q_d), 'muz_scale': scale}
    out |= {'b_test_negative': b_t <= 1e-300, 'D_r3': w['D_nondarcy'], 'D_eq': d_eq}
    points = [(f'{f:g}', f * p_r) for f in R_POINTS] + [('AOF', P_ATM)]
    for name, p_wf in points:
        dp2 = p_r**2 - p_wf**2
        ref = q_pseudo(p_wf)
        out[f'pseudo@{name}'] = ref
        out[f'p2@{name}'] = gas_rate(a, b, dp2) / ref
        out[f'p2_fit@{name}'] = gas_rate(a * scale, b * scale, dp2) / ref
        out[f'p2_test@{name}'] = gas_rate(a_t, b_t, dp2) / ref
        out[f'p2_ls@{name}'] = gas_rate(a_ls, b_ls, dp2) / ref
        out[f'backpr@{name}'] = math.exp(ln_c) * dp2**n_bp / ref
        out[f'vogel@{name}'] = q_d * vogel(p_wf / p_r) / vogel(1 - dd) / ref
        q_interim = q_p2_d * (p_r - p_wf) / (dd * p_r)
        out[f'interim@{name}'] = q_interim / ref
        out[f'interim_vs_p2@{name}'] = q_interim / gas_rate(a, b, dp2)
    return out


def wellstream_factor(w):
    """q_wellstream / q_separator gas = 1 + C_og / R_go, with the condensate's gas equivalent C_og = 133,000 g_o/M_o
    scf/STB and M_o = 6084/(API - 5.9) (Whitson and Brule 2000, Eqs. 3.73b, 3.73c), in Sm3/Sm3."""
    g_o = 141.5 / (131.5 + w['api'])
    c_og = 133_000 * g_o / (6084 / (w['api'] - 5.9)) * 0.178108
    return 1 + c_og / w['R_go']


def quantiles(s, ps=(5, 50, 95)):
    return ' / '.join(f'{v:.3g}' for v in np.percentile(s.dropna(), ps))


def report(df):
    print(f'{len(df)} gas and condensate wells ({df["tail"].sum()} in the tail); by class:')
    print(df.fluid_class.value_counts().to_string())
    print(
        f'\np_r (bar), P5 / P50 / P95: {quantiles(df.p_r)}; above {P2_LIMIT:.0f} bar (2000 psia): '
        f'{100 * (df.p_r > P2_LIMIT).mean():.1f}%'
    )
    print(f'design drawdown delta_d: {quantiles(df.delta_d)}')
    print(f'design rate, pseudo (1e6 Sm3/d): {quantiles(df.q_d / 1e6)}')
    print(f'AOF, pseudo (1e6 Sm3/d): {quantiles(df["pseudo@AOF"] / 1e6)}; R3 p2: {quantiles(df.AOF / 1e6)}')
    print(
        f'non-Darcy share b q/(a + b q) at the design rate: {quantiles(df.nondarcy_share)}; '
        f'above 0.5 in {100 * (df.nondarcy_share > 0.5).mean():.1f}%'
    )
    print(
        f'backpressure exponent n from a four-point test: {quantiles(df.n_backpr)}; '
        f'outside [0.5, 1] in {100 * ((df.n_backpr < 0.5) | (df.n_backpr > 1)).mean():.1f}%'
    )
    print(f'p2_fit scale on a and b: {quantiles(df.muz_scale)}')
    print(f'p2_test with a slope b <= 0: {100 * df.b_test_negative.mean():.1f}%')
    print(
        f'non-Darcy coefficient D (d/Mscf): R3 {quantiles(df.D_r3)}; Economides et al. Eq. 4-34, perforated over h: '
        f'{quantiles(df.D_eq)}; R3 / Eq. 4-34 {quantiles(df.D_r3 / df.D_eq)}'
    )
    cond = df.fluid_class == 'gas_condensate'
    print(
        f'wellstream / separator gas: condensates {quantiles(df.ws_factor[cond])}, '
        f'wet and dry gas {quantiles(df.ws_factor[~cond])}'
    )

    names = [f'{f:g}' for f in R_POINTS] + ['AOF']
    print('\nRate of pseudo, the reference (1e6 Sm3/d), median at each p_wf/p_r:')
    print('  ' + '  '.join(f'{n:>8}' for n in names))
    print('  ' + '  '.join(f'{df[f"pseudo@{n}"].median() / 1e6:8.3f}' for n in names))
    print('\nEach inflow over pseudo: P5 / P50 / P95 at each p_wf/p_r')
    rows = []
    for c in CURVES + ('interim_vs_p2',):
        rows.append({'inflow': c, **{n: quantiles(df[f'{c}@{n}']) for n in names}})
    print(pd.DataFrame(rows).to_string(index=False))
    for cls, sub in df.groupby('fluid_type'):
        print(f'\n{cls}: medians over pseudo')
        print(pd.DataFrame([{'inflow': c, **{n: f'{sub[f"{c}@{n}"].median():.3f}' for n in names}} for c in CURVES]))
    hi = df.p_r > P2_LIMIT
    for label, mask in ((f'p_r <= {P2_LIMIT:.0f} bar', ~hi), (f'p_r > {P2_LIMIT:.0f} bar', hi)):
        sub = df[mask]
        print(
            f'\n{label} ({len(sub)} wells): p2 over pseudo at 0.6 / 0.2 / AOF, medians '
            f'{sub["p2@0.6"].median():.3f} / {sub["p2@0.2"].median():.3f} / {sub["p2@AOF"].median():.3f}; '
            f'interim over p2 at 0.95 / AOF: {sub["interim_vs_p2@0.95"].median():.3f} / '
            f'{sub["interim_vs_p2@AOF"].median():.3f}'
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('n', type=int)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--csv')
    args = ap.parse_args()
    rows = []
    for k in range(args.n):
        w, _ = draw_well(args.seed, k)
        if w['fluid_type'] not in ('gas_condensate', 'gas'):
            continue
        keep = ('well', 'tail', 'fluid_type', 'fluid_class', 'p_r', 'T_r', 'api', 'sg_gas', 'R_go', 'a_gas', 'b_gas')
        keep += ('AOF', 'perm', 'h_net', 'f_geo', 'D_nondarcy')
        row = {key: w[key] for key in keep} | {'delta_d': w['delta_d'], 'ws_factor': wellstream_factor(w)}
        rows.append(row | curves(w))
    df = pd.DataFrame(rows)
    report(df)
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == '__main__':
    main()
