"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 28 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The CasADi search missed operating points where the bottomhole is all liquid (issue #10; plans/improvements.md §1.7;
found with the prototype of feature 018, PVT tables), and the fix: the march retries point 0 from the no-slip void
fraction where it fails (solvers/march.py, no_slip_guess and Marcher._bottom_point).

Where all the reservoir gas is dissolved at the bottomhole, the smoothed phase rates of PVT-OIL-13 still give a gas
rate of sqrt(eps) / 2 = 5e-4 kg/s (SMO-1, SMO-2 with eps = 1e-6 (kg/s)^2), so point 0's void fraction is about 1e-4
to 1e-3. The march's guess for point 0 (discretization.py, bottom_guess) puts it at 0.5 whatever the phase rates are;
from there Newton fails and the Ipopt fallback reports the point infeasible, so every march from such a p_0 fails.

It imports the fix, so it runs on develop from the fix on. The variants of the search:

  rust    the Rust core
  casadi  the CasADi backend as it is: with the retry
  old     the CasADi backend without the retry, as develop was at ae754d8
  guess   the rejected alternative: point 0 always from the no-slip void fraction, at most 0.5, with no retry

Parts:

  repro     well 87 of seed 2026 at its nominal operating point, with develop's black oil at gor 40 (R_so at the
            bottomhole is 141 Sm3/Sm3): each variant's operating point, each start of the old search, and point 0
            at each start from void-fraction guesses of 0.5, 1e-2, 1e-3 and 2e-4
  cases     the 263 cases of jt_wells.py (88 wells, three operating points each) at the draw's GOR and at half and a
            quarter of it, which dissolves all the gas at the bottomhole more often: the operating point of each
            variant (all four, or those named in a fourth argument, such as rust,casadi), 14 processes
  attempts  one process: the first wells' cases at the draws' GOR, searched by the variants casadi, old and guess in
            turn; each start's outcome, Ipopt iterations and time. Iterations do not depend on the load
  verify    develop's verifier candidate (scripts/verification/develop_candidate.py) with the variant guess or old;
            the variant casadi is the candidate itself. Check it with manywells-verify as AGENTS.md describes

From the repository root, one BLAS thread per process:

    uv run python plans/evidence/all_liquid_bottomhole.py repro
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python plans/evidence/all_liquid_bottomhole.py cases 88 OUT.csv
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python plans/evidence/all_liquid_bottomhole.py attempts 20 OUT.csv
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. uv run python plans/evidence/all_liquid_bottomhole.py \\
        verify guess guess.parquet
    uv run manywells-verify guess.parquet --data verification/data --name "develop, no-slip bottom guess"
"""

import sys
import time
from dataclasses import replace
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from manywells.configurations import DEVELOP
from manywells.sampling.conditions import nominal_conditions, sample_conditions
from manywells.sampling.wells import rng_for, sample_well, well_properties
from manywells.simulator import SimError, SSDFSimulator
from manywells.solvers.march import Marcher, MarchFailed, no_slip_guess
from manywells.solvers.roots import RootFinder

SEED = 2026
GOR_FACTORS = (1.0, 0.5, 0.25)
VARIANTS = ('rust', 'casadi', 'old', 'guess')


def _no_retry(self, guess, parameters):
    """Point 0 from the bottom guess alone (the variant old)."""
    return self._bottom(guess[1:], parameters)


def _from_no_slip(self, guess, parameters):
    """Point 0 from the no-slip void fraction, at most 0.5, at every start, with no retry (the variant guess)."""
    retry = no_slip_guess(guess)
    return self._bottom((guess if retry is None else retry)[1:], parameters)


MARCHES = {'old': _no_retry, 'guess': _from_no_slip}


def simulator(wp, variant):
    sim = SSDFSimulator(wp, backend='rust' if variant == 'rust' else 'casadi')
    if variant in MARCHES:
        sim._roots.march.__class__ = type(f'{variant}Marcher', (Marcher,), {'_bottom_point': MARCHES[variant]})
    return sim


def solve(wp, bc, variant):
    t0 = time.perf_counter()
    try:
        op = simulator(wp, variant).simulate(bc)
        return dict(PBH=op.state[0, 0], PWH=op.state[-1, 0], alpha_0=op.state[0, 3], T_0=op.state[0, 6],
                    seconds=time.perf_counter() - t0)
    except SimError as e:
        return dict(err=type(e).__name__, seconds=time.perf_counter() - t0)


def all_liquid_well():
    draw = sample_well(SEED, 87)
    wp = well_properties(draw, draw.fractions, DEVELOP)
    return replace(wp, fluid=replace(wp.fluid, gor=40.0)), nominal_conditions(draw)


def repro():
    wp, bc = all_liquid_well()
    print(f'well 87, seed {SEED}, nominal point: p_r {bc.p_r:.2f}, p_s {bc.p_s:.2f} bar, T_r {bc.T_r:.2f} K; gor 40, '
          f'R_so at (p_r, T_r) {float(wp.fluid.rs(bc.p_r, bc.T_r)):.1f} Sm3/Sm3')
    core = SSDFSimulator(wp, backend='rust').simulate(bc)
    X = core.state
    print(f'Rust core: operating point at p_0 {X[0, 0]:.2f} bar, alpha_0 {X[0, 3]:.1e}, '
          f'max T - T_r {X[:, 6].max() - bc.T_r:.2f} K')
    for variant in ('old', 'casadi', 'guess'):
        out = solve(wp, bc, variant)
        print(f'CasADi backend, variant {variant}: ' +
              (f'operating point at p_0 {out["PBH"]:.2f} bar' if 'PBH' in out else out['err']))
    sim = simulator(wp, 'old')
    system, march = sim._roots.system, sim._roots.march
    params = system.params(bc)
    rows = np.asarray(system.residual(core.x, params)).ravel()
    print(f'The core\'s root in the CasADi rows: largest |row| {np.abs(rows).max():.1e}')
    print('The starts of the old search, and point 0 from four void-fraction guesses:')
    for name, p_0 in RootFinder.starts(bc):
        try:
            march(params, p_0)
            status = 'march reaches the wellhead'
        except MarchFailed as e:
            status = str(e).replace(f'march from p_0 = {p_0:.3f} bar ', '')
        g = np.asarray(system.bottom_guess(p_0, params), dtype=float).ravel()
        point_0 = []
        for alpha in (0.5, 1e-2, 1e-3, 2e-4):
            h = g.copy()
            h[1], h[2], h[3] = g[1] * g[3] / alpha, g[2] * (1 - g[3]) / (1 - alpha), alpha
            try:
                y = march._bottom(h[1:], np.concatenate([[p_0], params]))
                point_0.append(f'{alpha:g} solves (alpha {y[2]:.1e})')
            except MarchFailed:
                point_0.append(f'{alpha:g} fails')
        print(f'  {name:24s} p_0 {p_0:7.2f}: {status}; point 0 from {", ".join(point_0)}')


def sampled_cases(well, factors=GOR_FACTORS):
    """The cases of jt_wells.py for one well, at each GOR factor: (k, factor, WellProperties, BoundaryConditions)."""
    draw = sample_well(SEED, well)
    rng = rng_for(SEED, well, 'jt')
    for k in range(3):
        bc, fractions = (nominal_conditions(draw), draw.fractions) if k == 0 else sample_conditions(draw, rng)
        try:
            wp = well_properties(draw, fractions, DEVELOP)
        except ValueError:  # no oil to give a gas-oil ratio (SMP-43)
            continue
        for factor in factors:
            yield k, factor, replace(wp, fluid=replace(wp.fluid, gor=wp.fluid.gor * factor)), bc


def well_cases(well, variants=VARIANTS):
    rows = []
    for k, factor, w, bc in sampled_cases(well):
        rec = dict(well=well, k=k, gor_factor=factor, gor=w.fluid.gor, p_r=bc.p_r, T_r=bc.T_r)
        for name in variants:
            rec.update({f'{key}_{name}': v for key, v in solve(w, bc, name).items()})
        # All the gas dissolved at the bottomhole: R_so at the operating point's (p_0, T_0) at least the GOR
        for name in variants:
            if np.isfinite(rec.get(f'PBH_{name}', np.nan)):
                rec['all_liquid'] = float(w.fluid.rs(rec[f'PBH_{name}'], rec[f'T_0_{name}'])) >= w.fluid.gor
                break
        rows.append(rec)
    return rows


def summary(df):
    names = tuple(n for n in VARIANTS if f'PBH_{n}' in df)
    df['all_liquid'] = df['all_liquid'].astype('boolean')
    print('Operating points found, by GOR factor and by whether the bottomhole is all liquid at the operating point:')
    g = df.groupby(['gor_factor', 'all_liquid'], dropna=False)
    table = pd.DataFrame({'cases': g.size(), **{n: g[f'PBH_{n}'].apply(lambda s: s.notna().sum()) for n in names}})
    for n in names[1:]:
        table[f'rust, not {n}'] = g.apply(lambda d: (d.PBH_rust.notna() & d[f'PBH_{n}'].isna()).sum(),
                                          include_groups=False)
        table[f'{n}, not rust'] = g.apply(lambda d: (d[f'PBH_{n}'].notna() & d.PBH_rust.isna()).sum(),
                                          include_groups=False)
    print(table.to_string())
    for n in names[1:]:
        both = df.dropna(subset=['PBH_rust', f'PBH_{n}'])
        print(f'{n} against the core, {len(both)} cases with both: largest |dPBH| '
              f'{(both[f"PBH_{n}"] - both.PBH_rust).abs().max():.2e} bar')
    print('Time on one of 14 processes (s), median / total: ' + ', '.join(
        f'{n} {df[f"seconds_{n}"].median():.2f} / {df[f"seconds_{n}"].sum():.0f}' for n in names))


def cases(n_wells, out, variants=VARIANTS):
    with Pool(14) as pool:
        df = pd.DataFrame([r for rs in pool.map(partial(well_cases, variants=variants), range(n_wells), chunksize=1)
                           for r in rs])
    df.to_csv(out, index=False)
    pd.set_option('display.width', 220)
    summary(df)


def attempts(n_wells, out):
    rows = []
    for well in range(n_wells):
        for k, _, wp, bc in sampled_cases(well, factors=(1.0,)):
            for variant in ('casadi', 'old', 'guess'):
                for a in simulator(wp, variant).root_set(bc).search:
                    rows.append(dict(well=well, k=k, variant=variant, start=a.start,
                                     outcome=a.outcome.split(' at ')[0], iterations=a.iterations, seconds=a.seconds))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    pv = df.pivot_table(index=['well', 'k', 'start'], columns='variant', values=['outcome', 'iterations', 'seconds'],
                        aggfunc='first')
    pd.set_option('display.width', 220)
    print(f'{len(pv)} starts in {df.groupby(["well", "k"]).ngroups} cases')
    for v in ('casadi', 'guess'):
        changed = pv[pv[('outcome', v)] != pv[('outcome', 'old')]]
        print(f'\n{v} against old: outcomes changed at {len(changed)} starts:')
        print(pd.crosstab(changed[('outcome', 'old')], changed[('outcome', v)]).to_string())
        for c in ('iterations', 'seconds'):
            print(f'{c}: {v} {pv[(c, v)].sum():.1f}, old {pv[(c, "old")].sum():.1f}; at the starts whose outcome '
                  f'changed: {v} {changed[(c, v)].sum():.1f}, old {changed[(c, "old")].sum():.1f}')


def verify(variant, out):
    """develop's verifier candidate with a variant's march (forked workers inherit the patched class)."""
    import multiprocessing
    import scripts.verification.develop_candidate as candidate
    from manywells.solvers import march as march_module
    multiprocessing.set_start_method('fork', force=True)    # Python 3.14's forkserver would lose the patch
    march_module.Marcher._bottom_point = MARCHES[variant]
    sys.argv = ['develop_candidate', out]
    candidate.main()


if __name__ == '__main__':
    if sys.argv[1] == 'repro':
        repro()
    elif sys.argv[1] == 'cases':
        cases(int(sys.argv[2]), sys.argv[3], *([tuple(sys.argv[4].split(','))] if len(sys.argv) > 4 else []))
    elif sys.argv[1] == 'summary':
        pd.set_option('display.width', 220)
        summary(pd.read_csv(sys.argv[2]))
    elif sys.argv[1] == 'attempts':
        attempts(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == 'verify':
        verify(sys.argv[2], sys.argv[3])
