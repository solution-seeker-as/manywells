"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 29 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The inputs of each correlation along the operating points of wells drawn by develop's sampler, against the
correlation's published range (specs/model/validity.md). Each well (seed 2026) is solved at its nominal operating
point and at two sampled ones, in develop's default configuration, as in gor_cap_wells.py:

  run       solve the wells and write PREFIX_cases.csv, one row per case (the well's inputs, whether an operating
            point was found), and PREFIX_points.parquet, one row per grid point of each operating point (the state,
            and each correlation's inputs there: pressure and temperature in field units, the pseudo-reduced
            pressure and temperature, the uncapped and capped solution gas-oil ratio, B_o, the Reynolds number, the
            inclination)
  summary   for each row of RANGES, the sampler's range of the variable and the share of cases with a point (or a
            well input) outside the published range

From the repository root, one BLAS thread per process:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python plans/evidence/validity_ranges.py run 500 OUT --backend rust
    uv run python plans/evidence/validity_ranges.py summary OUT
"""

import argparse
import math
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from manywells.configurations import DEVELOP
from manywells.sampling.conditions import nominal_conditions, sample_conditions
from manywells.sampling.wells import rng_for, sample_well, well_properties
from manywells.simulator import SimError, SSDFSimulator
from manywells.units import CF_BAR, CF_PSI, CF_RS

SEED = 2026
STB_PER_M3 = 6.28981           # Barrels per cubic metre
PSI_PER_BAR = CF_BAR / CF_PSI


def fahrenheit(T):
    return (T - 273.15) * 1.8 + 32


def point_rows(wp, op):
    """Each correlation's inputs at the grid points of an operating point."""
    fl, geo = wp.fluid, wp.geometry
    ppc, tpc = fl.pseudo_critical
    cos = np.asarray(geo.cos_incl)
    rows = []
    for i, (p, v_g, v_l, alpha, rho_g, rho_l, T) in enumerate(op.state):
        rho_m = alpha * rho_g + (1 - alpha) * rho_l
        v_m = alpha * v_g + (1 - alpha) * v_l
        mu_m = float(fl.mixture_viscosity(p, T, alpha, rho_g, rho_l, v_g, v_l))
        rows.append(dict(
            i=i, md=geo.md[i], cos_incl=cos[max(i - 1, 0)], p=p, T=T, alpha=alpha, v_g=v_g, v_l=v_l, rho_g=rho_g,
            rho_l=rho_l, regime=op.flow_regime[i],
            p_psia=p * PSI_PER_BAR, T_F=fahrenheit(T), T_C=T - 273.15,
            ppr=p * CF_BAR / ppc, tpr=T / tpc,
            rs_vb=float(fl.pvt.rs(p, T)), rs=float(fl.rs(p, T)), bo=float(fl.bo(p, T)),
            Re=rho_m * abs(v_m) * geo.D / mu_m,
            vgs=alpha * v_g, vls=(1 - alpha) * v_l,
        ))
    return rows


def well_cases(well, backend):
    cases, points = [], []
    draw = sample_well(SEED, well)
    rng = rng_for(SEED, well, 'validity')
    for k in range(3):
        bc, fractions = (nominal_conditions(draw), draw.fractions) if k == 0 else sample_conditions(draw, rng)
        try:
            wp = well_properties(draw, fractions, DEVELOP)
        except ValueError:  # no oil to give a gas-oil ratio (SMP-43)
            continue
        fl, geo = wp.fluid, wp.geometry
        rho_l_sc = fl.rho_l
        kind = draw.trajectory[0]
        rec = dict(
            well=well, k=k, trajectory=kind,
            inclination=draw.trajectory[2] if kind == 'deviated' else (90.0 if kind == 'l_shaped' else 0.0),
            tvd=draw.L, md=geo.md[0], D=geo.D, eps_D=draw.roughness / geo.D, h=draw.h,
            api=fl.api, sg_gas=fl.sg_gas, gor=fl.gor, wlr=fl.wlr, f_g=fl.f_g,
            gor_scf=fl.gor / CF_RS, glr_scf=fl.glr / CF_RS,
            p_r=bc.p_r, p_s=bc.p_s, T_r=bc.T_r, T_s=bc.T_s, u=bc.u, w_lg=bc.w_lg,
        )
        t0 = time.time()
        try:
            op = SSDFSimulator(wp, backend=backend).simulate(bc)
            q_l = op.w_res / rho_l_sc                                   # Stock-tank liquid (m3/s)
            q_lg = bc.w_lg / fl.rho_g                                   # Lift gas at standard conditions (m3/s)
            rec.update(found=True, PBH=op.p_0, PWH=op.state[-1, 0], TWH=op.state[-1, 6], w_res=op.w_res,
                       q_l_stbd=q_l * 86400 * STB_PER_M3, bhp_psig=(op.p_0 - 1.01325) * PSI_PER_BAR,
                       glr_total_scf=(fl.glr + q_lg / q_l) / CF_RS if q_l > 0 else np.nan,
                       choked=op.choked)
            points += [dict(well=well, k=k, **r) for r in point_rows(wp, op)]
        except SimError as e:
            rec.update(found=False, err=type(e).__name__)
        rec['seconds'] = time.time() - t0
        cases.append(rec)
    return cases, points


def run(n_wells, prefix, backend, processes):
    with Pool(processes) as pool:
        results = pool.map(partial(well_cases, backend=backend), range(n_wells), chunksize=1)
    cases = pd.DataFrame([c for cs, _ in results for c in cs])
    points = pd.DataFrame([p for _, ps in results for p in ps])
    cases.to_csv(f'{prefix}_cases.csv', index=False)
    points.to_parquet(f'{prefix}_points.parquet', index=False)
    print(f'{len(cases)} cases from {n_wells} wells ({backend}), operating point in {cases.found.sum()}, '
          f'{len(points)} points, {cases.seconds.sum():.0f} s in total')


# The published ranges of specs/model/validity.md: (row, column, table, low, high). The table is 'cases' for a well
# input or an operating point's quantity, 'points' for a quantity at each grid point of the operating point. None is
# an open end.
RANGES = [
    ('Beggs and Robinson, Table 1', 'api', 'cases', 16, 58),
    ('Beggs and Robinson, Table 1', 'p_psia', 'points', None, 5250 + 14.696),
    ('Beggs and Robinson, Table 1', 'T_F', 'points', 70, 295),
    ('Beggs and Robinson, Table 1', 'rs', 'points', 20 * CF_RS, 2070 * CF_RS),
    ('BlackOil today', 'api', 'cases', 10, 40),
    ('Gas condensates and leaner (Whitson and Brule)', 'gor_scf', 'cases', None, 3000),
    ('Beyond gas condensates (Whitson and Brule)', 'gor_scf', 'cases', None, 150000),
    ('Valko and McCain, Table 2.1', 'gor', 'cases', 10 * CF_RS, 2216 * CF_RS),
    ('Valko and McCain, Table 2.1', 'api', 'cases', 6.0, 63.7),
    ('Valko and McCain, Table 2.1', 'sg_gas', 'cases', 0.555, 1.685),
    ('Valko and McCain, Table 2.1', 'T_r', 'cases', (60 - 32) / 1.8 + 273.15, (342 - 32) / 1.8 + 273.15),
    ('Abdul-Majeed, dead oil', 'T_C', 'points', 15.56, 54.44),
    ('Abdul-Majeed, dead oil', 'api', 'cases', 15, 50),
    ('Abdul-Majeed, live oil', 'p', 'points', 13.8, 206.8),
    ('Abdul-Majeed, live oil', 'rs', 'points', 3, 208),
    ('Sutton', 'sg_gas', 'cases', 0.571, 1.679),
    ('Dranchuk and Abou-Kassem', 'ppr', 'points', 0.2, 30),
    ('Dranchuk and Abou-Kassem', 'tpr', 'points', 1.0, 3.0),
    ('Papay, as gas.md states it', 'ppr', 'points', None, 6),
    ('Lee, Gonzalez and Eakin', 'p_psia', 'points', 100, 8000),
    ('Lee, Gonzalez and Eakin', 'T_F', 'points', 100, 340),
    ('Chen', 'Re', 'points', 4000, 4e8),
    ('Chen', 'eps_D', 'cases', 5e-7, 0.05),
    ('Haaland', 'Re', 'points', 4000, 1e8),
    ('Haaland', 'eps_D', 'cases', 0, 0.05),
    ('Hasan et al., Table 2', 'glr_scf', 'cases', 104, 5615),
    ('Hasan et al., Table 2', 'glr_total_scf', 'cases', 104, 5615),
    ('Hasan et al., Table 2', 'q_l_stbd', 'cases', 100, 27760),
    ('Hasan et al., Table 2', 'bhp_psig', 'cases', 108, 13755),
    ('Hasan et al., Table 2', 'wlr', 'cases', 0, 0.98),
    ('Hasan et al., Table 2', 'md', 'cases', 1635 * 0.3048, 30650 * 0.3048),
    ('Vertical and deviated cells', 'cos_incl', 'points', 0.05, 1),
]


def summary(prefix):
    cases = pd.read_csv(f'{prefix}_cases.csv')
    points = pd.read_parquet(f'{prefix}_points.parquet')
    solved = cases[cases.found]
    print(f'{len(cases)} cases of {cases.well.nunique()} wells; operating point in {len(solved)} '
          f'({solved.well.nunique()} wells); {len(points)} grid points\n')
    q = [0, 0.05, 0.5, 0.95, 1]
    rows = []
    for name, col, table, lo, hi in RANGES:
        df = solved if table == 'cases' else points
        x = df[col]
        out = pd.Series(False, index=df.index)
        if lo is not None:
            out |= x < lo
        if hi is not None:
            out |= x > hi
        if table == 'cases':
            share_cases = out.mean()
            share_points = math.nan
        else:
            share_cases = out.groupby([df.well, df.k]).any().mean()
            share_points = out.mean()
        rows.append(dict(row=name, variable=col, low=lo, high=hi, **{f'q{int(100 * v)}': x.quantile(v) for v in q},
                         cases_out=round(100 * share_cases, 1), points_out=round(100 * share_points, 1)))
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 20)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print('\nShares in %: of the cases with an operating point that have a point (or input) outside the range, and '
          'of all grid points.')
    print(f"Regimes at the points of horizontal cells (cos < 0.05): "
          f"{points[points.cos_incl < 0.05].regime.value_counts(normalize=True).round(3).to_dict()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    sub = parser.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('n_wells', type=int)
    r.add_argument('prefix')
    r.add_argument('--backend', choices=('casadi', 'rust'), default='rust')
    r.add_argument('--processes', type=int, default=8)
    s = sub.add_parser('summary')
    s.add_argument('prefix')
    args = parser.parse_args()
    if args.cmd == 'run':
        run(args.n_wells, args.prefix, args.backend, args.processes)
    else:
        summary(args.prefix)


if __name__ == '__main__':
    main()
