"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 29 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The implemented feature 021 (specs/features/021-lift-gas-injection-depth.md, Acceptance, item 7): the lift gas
injected at the bottomhole and at a valve at 75% and at 50% of the well's measured depth from the surface, on wells
drawn by develop's sampler (seed 2026), in develop's default configuration.

The cases are those of five_regime_default.py's draw (each well at its nominal operating point and two sampled ones,
with the generator 'jt'), from more wells, that carry lift gas (w_lg > 0). The nominal operating point has none
(SMP-28), so these are the sampled points of the wells with gas lift (SMP-17, SMP-19). The lift gas's temperature is
not given, so it is the ambient temperature at the injection point (THM-9).

  run       solve each case with the valve at each depth: the Rust core on every case, the CasADi backend on the first
            --casadi-cases; per case, backend and depth, the number of roots, PBH, PWH, TWH, the reservoir liquid rate,
            the search's time, and the core's work counters (march.rs, Counts). A case's depths run one after another
            in one process, so their times compare
  summary   the operating points found, the changes in PBH, PWH, TWH and the reservoir rate against the bottomhole,
            the share of cases that flow only with the lift gas at the bottomhole, and the times
  scan      for each case that loses its operating point with the valve, the core's shooting residual R(p_0) on
            --samples values from p_s to 1e-6 bar below p_r: a sign change is a root the search missed (a solver
            finding), R of one sign throughout means the model has no root there

From the repository root, one BLAS thread per process:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python plans/evidence/injection_depth_wells.py run 1000 OUT.csv
    uv run python plans/evidence/injection_depth_wells.py summary OUT.csv
    uv run python plans/evidence/injection_depth_wells.py scan OUT.csv
"""

import argparse
import time
from collections import Counter
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import pandas as pd

from manywells.configurations import DEVELOP
from manywells.sampling.conditions import nominal_conditions, sample_conditions
from manywells.sampling.wells import rng_for, sample_well, well_properties
from manywells.simulator import SimError, SSDFSimulator

SEED = 2026
VALVES = {'bh': None, '75': 0.75, '50': 0.5}  # The valve's measured depth as a share of the well's, or the bottomhole
COUNTS = ('marches', 'temperature_solves', 'chord_fallbacks', 'step_outs', 'lower_step_outs', 'temperature_failures')


def cases(n_wells):
    """(well, k, bc, wp) of the gas-lifted cases: each well's two sampled operating points with lift gas."""
    out = []
    for well in range(n_wells):
        draw = sample_well(SEED, well)
        rng = rng_for(SEED, well, 'jt')
        for k in range(3):
            bc, fractions = (nominal_conditions(draw), draw.fractions) if k == 0 else sample_conditions(draw, rng)
            try:
                wp = well_properties(draw, fractions, DEVELOP)
            except ValueError:  # no oil to give a gas-oil ratio (SMP-43)
                continue
            if bc.w_lg > 0:
                out.append((well, k, bc, wp))
    return out


def with_valve(wp, share):
    return replace(wp, injection_md=None if share is None else share * wp.geometry.L)


def solve(wp, bc, backend):
    t0 = time.perf_counter()
    try:
        sim = SSDFSimulator(wp, backend=backend)
        rs = sim.root_set(bc)
    except SimError:
        sim, rs = None, None
    out = {'seconds': time.perf_counter() - t0, 'roots': len(rs.roots) if rs else 0}
    op = rs.operating_point if rs else None
    if op is not None:
        X = op.state
        out.update(PBH=X[0, 0], PWH=X[-1, 0], TWH=X[-1, 6], w_res=op.w_res)
    if backend == 'rust' and sim is not None:
        out.update({c: sim._roots.counts[c] for c in COUNTS})
    return out


def case_rows(case):
    (well, k, bc, wp), casadi = case
    rows = []
    for backend in ('rust', 'casadi') if casadi else ('rust',):
        rec = dict(well=well, k=k, backend=backend, w_lg=bc.w_lg, p_r=bc.p_r, p_s=bc.p_s, u=bc.u, md=wp.geometry.L,
                   f_g=wp.fluid.f_g, gor=wp.fluid.gor, wlr=wp.fluid.wlr)
        for name, share in VALVES.items():
            v = with_valve(wp, share)
            rec[f'point_{name}'] = v.injection_point
            rec.update({f'{key}_{name}': val for key, val in solve(v, bc, backend).items()})
        rows.append(rec)
    return rows


def run(n_wells, out, casadi_cases, processes):
    todo = cases(n_wells)
    with Pool(processes) as pool:
        rows = [r for rs in pool.imap(case_rows, [(c, j < casadi_cases) for j, c in enumerate(todo)], chunksize=1)
                for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f'{len(todo)} gas-lifted cases from {df.well.nunique()} of {n_wells} wells; '
          f'{(df.backend == "casadi").sum()} on the CasADi backend too')


def summary(path):
    q = [0, 0.1, 0.5, 0.9, 1]
    pd.set_option('display.width', 250)
    df = pd.read_csv(path)
    for backend, d in df.groupby('backend', sort=False):
        print(f'== {backend}: {len(d)} cases from {d.well.nunique()} wells; MD {d.md.min():.0f} to {d.md.max():.0f} m, '
              f'w_lg {d.w_lg.min():.2f} to {d.w_lg.max():.2f} kg/s')
        for v in VALVES:
            s = d[f'seconds_{v}']
            print(f'  {v:>3}: operating point in {d[f"PBH_{v}"].notna().sum()}, roots per case '
                  f'{dict(sorted(Counter(d[f"roots_{v}"]).items()))}, {s.median():.2f} s median, {s.sum():.1f} s total')
        for v in ('75', '50'):
            found_bh, found_v = d.PBH_bh.notna(), d[f'PBH_{v}'].notna()
            lost, gained = d[found_bh & ~found_v], d[~found_bh & found_v]
            print(f'  valve at {v}% against the bottomhole: lost {len(lost)}, gained {len(gained)}; '
                  f'flow only with the lift gas at the bottomhole: {len(lost)} of {found_bh.sum()} '
                  f'({100 * len(lost) / max(found_bh.sum(), 1):.1f}%)')
            for name, rows in (('lost', lost), ('gained', gained)):
                if len(rows):
                    print(f'    {name}: ' + ', '.join(f'({r.well}, {r.k})' for r in rows.itertuples()))
            both = d[found_bh & found_v]
            changes = pd.DataFrame({
                'PBH (bar)': both[f'PBH_{v}'] - both.PBH_bh,
                'PWH (bar)': both[f'PWH_{v}'] - both.PWH_bh,
                'TWH (K)': both[f'TWH_{v}'] - both.TWH_bh,
                'w_res (%)': 100 * (both[f'w_res_{v}'] / both.w_res_bh - 1),
            })
            print(f'  {len(both)} cases with an operating point in both; quantiles of the change, valve minus '
                  'bottomhole:')
            print(changes.quantile(q).round(3).to_string())
        if backend == 'rust':
            for v in VALVES:
                print(f'  {v:>3}: ' + ', '.join(f'{c} {d[f"{c}_{v}"].sum():.0f}' for c in COUNTS))
    if set(df.backend) == {'rust', 'casadi'}:
        r, c = (df[df.backend == b].set_index(['well', 'k']) for b in ('rust', 'casadi'))
        r = r.loc[c.index]
        for v in VALVES:
            both = r[f'PBH_{v}'].notna() & c[f'PBH_{v}'].notna()
            differ = r[f'PBH_{v}'].notna() != c[f'PBH_{v}'].notna()
            print(f'backends, {v}: operating point in both {both.sum()}, in one only {differ.sum()} '
                  f'{list(r.index[differ])}; PBH within {np.max(np.abs(r[f"PBH_{v}"] - c[f"PBH_{v}"])[both]):.2e} bar')


def scan_case(args):
    from manywells.solvers.rust import core_well, operating_point
    (well, k, bc, wp), share, samples = args
    v = with_valve(wp, share)
    core, op = core_well(v), operating_point(v, bc)
    p = np.concatenate([np.linspace(bc.p_s + 1e-3, bc.p_r - 1e-3, samples - 20), bc.p_r - np.geomspace(1e-3, 1e-6, 20)])
    R = np.array([np.nan if (r := core.residual(op, float(x))) is None else r for x in p])
    finite = np.isfinite(R)
    signs = np.sign(R[finite])
    changes = int(np.sum(signs[1:] != signs[:-1]))
    return dict(well=well, k=k, valve=share, finite=int(finite.sum()), sign_changes=changes,
                R_min=float(np.min(R[finite])) if finite.any() else np.nan,
                p_at_min=float(p[finite][np.argmin(R[finite])]) if finite.any() else np.nan)


def scan(path, n_wells, samples, processes):
    df = pd.read_csv(path)
    d = df[df.backend == 'rust']
    todo = {(c[0], c[1]): c for c in cases(n_wells)}
    work = []
    for v, share in (('75', 0.75), ('50', 0.5)):
        for r in d[d.PBH_bh.notna() & d[f'PBH_{v}'].isna()].itertuples():
            work.append((todo[r.well, r.k], share, samples))
    with Pool(processes) as pool:
        out = pd.DataFrame(pool.map(scan_case, work, chunksize=1))
    pd.set_option('display.width', 250)
    print(f'{len(out)} lost (case, valve) pairs, R(p_0) on {samples} samples:')
    print(out.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    sub = parser.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('n_wells', type=int)
    r.add_argument('out')
    r.add_argument('--casadi-cases', type=int, default=90)
    r.add_argument('--processes', type=int, default=12)
    s = sub.add_parser('summary')
    s.add_argument('path')
    c = sub.add_parser('scan')
    c.add_argument('path')
    c.add_argument('--n-wells', type=int, default=1000)
    c.add_argument('--samples', type=int, default=4000)
    c.add_argument('--processes', type=int, default=12)
    args = parser.parse_args()
    if args.cmd == 'run':
        run(args.n_wells, args.out, args.casadi_cases, args.processes)
    elif args.cmd == 'summary':
        summary(args.path)
    else:
        scan(args.path, args.n_wells, args.samples, args.processes)


if __name__ == '__main__':
    main()
