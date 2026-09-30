"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The validation case set (v2 plan, Step 2 item 3) and v1.0.0's roots on it (item 4). Runs in the
v1.0.0 environment, from the repository root, in two stages around the Rust roots:

    .worktrees/v1.0.0/.venv/bin/python verification/build/make_cases.py select --data-dir <dir>
    .worktrees/rust/.venv/bin/python   verification/build/rust_roots.py roots
    .worktrees/v1.0.0/.venv/bin/python verification/build/make_cases.py solve

<dir> holds manywells-sol-1_config.zip, manywells-nsol-1_config.zip and manywells-nsol-1.zip.
Every draw is seeded from the case's source, well and draw number (common.seed_for).

`select` solves every sol-1 well at its stored operating point (u = 0.5, v1's first solve per
well) to find the trickle-root wells, picks the cases and writes data/build/cases.json.

`solve` runs v1 on every case from: its default guess ("cold"); the start the dataset generator
used, where there is one ("warm": the well's u = 0.5 solution for drawn sol-1 cases; "dataset":
nsol-1's stored final state); the interpolated root one grid coarser ("interp", convergence
groups); cellwise guesses at p_s + f (p_r - p_s) for f in plans/evidence/root_sets.GUESSES
(method A); and each Rust root (method B). Every distinct solution is recorded with the starts
that reached it, its Ipopt status and its normalized stability slope from v1's Jacobian.
Writes data/build/v1_runs.json, v1_arrays.npz and coverage.md.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import Counter
from pathlib import Path

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')   # one thread per worker: the pool already uses every core

import casadi  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA, interpolate_state, params_of, quiet, read_json, seed_for, variant_of,  # noqa: E402
                    well_of, write_json)
from manywells.choke import BernoulliChokeModel  # noqa: E402  (v1.0.0 API)
from manywells.inflow import ProductivityIndex  # noqa: E402
from manywells.simulator import SimError  # noqa: E402
from manywells_verify.cases import Case  # noqa: E402
from manywells_verify.checks import state_distance  # noqa: E402
from root_sets import GUESSES  # noqa: E402  (plans/evidence)
from scripts.data_generation.well import sample_well  # noqa: E402
from scripts.load_well_from_dataset import load_well  # noqa: E402
from stability_label import Graph  # noqa: E402

GRAVITY = 9.81
SAME_ROOT = 1e-6          # scaled state distance below which two v1 solutions are one root
N_TRICKLE_U05, N_STABLE_U05 = 15, 3
QUOTA_DRAWN = 10          # sol-1 drawn operating points per (two-root, gas lift) stratum
N_FRESH = 12
N_NEAR_FOLD, NEAR_FOLD_DELTA = 10, (0.001, 0.01, 0.1, 1.0)     # bar above the Rust fold
N_PAST_FOLD, PAST_FOLD_DELTA = 6, (0.001, 0.1, 2.0)            # bar below it
N_LIFT, LIFT_RATES = 10, (0.05, 0.3, 1.0, 3.0)                 # kg/s, on two-root wells
N_NSOL = 20
N_SYNTH_WELLS = 2         # each with Bernoulli choke, productivity-index inflow, and both
N_CONV_GROUPS = 12        # drawn sol-1 cases at N, 2N and 4N

# Record the last Ipopt solver built, to read its return status and iteration count after a solve
_nlpsol = casadi.nlpsol
LAST_SOLVER = {}


def _recording_nlpsol(*args, **kwargs):
    LAST_SOLVER['solver'] = solver = _nlpsol(*args, **kwargs)
    return solver


casadi.nlpsol = _recording_nlpsol


def two_root(prm) -> bool:
    return prm['p_r'] - prm['p_s'] < prm['rho_l'] * GRAVITY * prm['L'] / 1e5


# ---------------------------------------------------------------------------------------------
# Solving one case (worker processes)

def stability_slope(sim, x) -> float:
    """
    d(choke row)/dp0 with every other row held, as plans/evidence/stability_label.py computes it
    (without its condition number), normalized by (p_r - p_s) / w_m. Positive is unstable.
    """
    J = np.array(sim.J_fun(x))
    c, n = sim.choke_row, J.shape[0]
    rows, cols = np.delete(np.arange(n), c), np.arange(1, n)
    dx = -np.linalg.solve(J[np.ix_(rows, cols)], J[rows, 0])
    slope = J[c, 0] + J[c, cols] @ dx
    p, v_g, v_l, alpha, rho_g, rho_l, T = x[-7:]
    w_m = sim.wp.A * (alpha * rho_g * v_g + (1 - alpha) * rho_l * v_l)
    return float(slope * (sim.bc.p_r - sim.bc.p_s) / max(w_m, 1e-3))


def solve(sim, x_guess):
    """v1's solve from x_guess (None: its default guess). Returns (x or None, outcome dict)."""
    sim.x_guess = x_guess
    LAST_SOLVER.clear()
    out = {'status': 'failed', 'return_status': None, 'iter_count': None, 'stage': None}
    x = None
    try:
        with quiet():
            x = np.array(sim.simulate())
        out['status'] = 'solved'
    except SimError as e:
        out['stage'] = str(e)
    solver = LAST_SOLVER.get('solver')
    if solver is not None and solver.size1_in(0) == 7 * (sim.n_cells + 1):   # the full solve ran
        stats = solver.stats()
        out['return_status'] = stats.get('return_status')
        out['iter_count'] = None if stats.get('iter_count') is None else int(stats['iter_count'])
        if x is None:
            out['stage'] = 'full solve'
    elif x is None:
        out['stage'] = f'initial guess ({out["stage"]})'
    if x is not None:
        df = sim.solution_as_df(x)
        out |= {'p0': float(x[0]), 'slope': stability_slope(sim, x),
                'regime_bh': df['flow-regime'].iloc[0], 'regime_wh': df['flow-regime'].iloc[-1],
                'choked': bool(sim.bc.p_s <= sim.wp.choke.cpr * x[-7])}
    return x, out


def run_case(task):
    """Solve one case from every start; group the solutions into distinct roots."""
    case, starts = task['case'], task.get('starts', {})
    wp, bc = well_of(case)
    sim = Graph(wp, bc, n_cells=case['n_cells']).build()
    case_obj = Case(case['case_id'], case['params'], case['n_cells'])
    runs, arrays, roots = {}, {}, []

    def record(tag, x, outcome):
        for r in roots:
            d = state_distance(x, arrays[r['key']], case_obj)
            if d < SAME_ROOT:
                r['starts'].append(tag)
                r['spread'] = max(r['spread'], d)      # how far apart v1's solves of one root land
                return
        key = f'R{len(roots)}'
        arrays[key] = x
        roots.append({'key': key, 'starts': [tag], 'spread': 0.0, 'p0': outcome['p0'], 'slope': outcome['slope'],
                      'return_status': outcome['return_status']})

    todo = [('cold', None)] + list(starts.items())
    if task.get('multistart', True):
        for f in GUESSES[1:]:
            try:
                with quiet():
                    todo.append((f'guess {f}', sim._simulate_cellwise(bc.p_s + f * (bc.p_r - bc.p_s), bc.T_r)))
            except SimError:
                pass
    for tag, x0 in todo:
        x, outcome = solve(sim, None if x0 is None else np.asarray(x0))
        if not tag.startswith(('guess', 'rust')):
            runs[tag] = outcome
            if x is not None:
                arrays[tag] = x
        if x is not None:
            record(tag, x, outcome)
    runs['roots'] = sorted(roots, key=lambda r: r['p0'])
    return case['case_id'], runs, arrays


def run_all(tasks, processes):
    with mp.get_context('fork').Pool(processes) as pool:
        return {cid: (runs, arrays) for cid, runs, arrays in pool.imap_unordered(run_case, tasks, chunksize=1)}


# ---------------------------------------------------------------------------------------------
# Case selection

def case_of(case_id, source, wp, bc, n_cells=100, group='', config_id=None, seed=None, note=''):
    return {'case_id': case_id, 'source': source, 'params': params_of(wp, bc), 'variant': variant_of(wp),
            'n_cells': n_cells, 'group': group, 'config_id': config_id, 'seed': seed, 'note': note}


def draw(well, *seed_parts):
    """v1's sample_new_conditions with a seed of its own; v1 draws from numpy's global generator."""
    seed = seed_for(*seed_parts)
    np.random.seed(seed)
    return well.sample_new_conditions(), seed


def select_drawn(df, rng):
    """sol-1 wells at operating points drawn as the sol-1 generator drew them, by (two-root, gas lift)."""
    strata = [(a, b) for a in (True, False) for b in (True, False)]
    quota, cases = Counter(), []
    for well_id in rng.permutation(df['ID'].to_numpy()):
        new, seed = draw(load_well(int(well_id), df), 'sol-1', well_id, 0)
        stratum = (two_root(params_of(new.wp, new.bc)), new.bc.w_lg > 0)
        if quota[stratum] < QUOTA_DRAWN:
            quota[stratum] += 1
            cases.append(case_of(f'sol1-{well_id:04d}-d0', 'sol-1 drawn', new.wp, new.bc, config_id=int(well_id),
                                 seed=seed, note=f'two-root {stratum[0]}, gas lift {stratum[1]}'))
        if all(quota[s] >= QUOTA_DRAWN for s in strata):
            break
    return cases


def select_fresh():
    cases = []
    for k in range(N_FRESH):
        seed = seed_for('fresh', k)
        np.random.seed(seed)
        new, op_seed = draw(sample_well(), 'fresh-op', k)
        cases.append(case_of(f'fresh-{k:03d}', 'fresh sample_well', new.wp, new.bc, seed=seed,
                             note=f'operating point seed {op_seed}'))
    return cases


def select_fold(df, fold):
    cases = []
    for k, row in enumerate(fold.itertuples()):
        well = load_well(int(row.config_id), df)
        if k < N_NEAR_FOLD:
            delta = NEAR_FOLD_DELTA[k % len(NEAR_FOLD_DELTA)]
            well.bc.p_r = row.p_r_fold + delta
            cases.append(case_of(f'fold-{row.config_id:04d}', 'near fold', well.wp, well.bc,
                                 config_id=int(row.config_id), note=f'{delta} bar above the Rust fold'))
        elif k < N_NEAR_FOLD + N_PAST_FOLD:
            delta = PAST_FOLD_DELTA[k % len(PAST_FOLD_DELTA)]
            well.bc.p_r = row.p_r_no_root - delta
            cases.append(case_of(f'nofold-{row.config_id:04d}', 'past fold', well.wp, well.bc,
                                 config_id=int(row.config_id), note=f'{delta} bar below the Rust fold'))
    return cases


def select_lift(df, rng):
    ids = df.loc[df['bc.p_r'] - df['bc.p_s'] < df['wp.rho_l'] * GRAVITY * df['wp.L'] / 1e5, 'ID'].to_numpy()
    cases = []
    for k, well_id in enumerate(rng.choice(ids, N_LIFT, replace=False)):
        well = load_well(int(well_id), df)
        well.bc.w_lg = LIFT_RATES[k % len(LIFT_RATES)]
        cases.append(case_of(f'lift-{well_id:04d}', 'gas lift on two-root well', well.wp, well.bc,
                             config_id=int(well_id), note=f'w_lg = {well.bc.w_lg} kg/s; two-root at w_lg = 0'))
    return cases


def select_nsol(df_cfg, data_dir, rng):
    """nsol-1 final-week states whose stored bc matches the dataset's last row for that well."""
    rows = pd.read_csv(data_dir / 'manywells-nsol-1.zip', compression='zip',
                       usecols=['ID', 'WEEKS', 'CHK', 'PDC', 'WGL', 'FGAS'])
    last = rows.sort_values('WEEKS').groupby('ID').tail(1).set_index('ID')
    cfg = df_cfg.set_index('ID')
    stored = ['bc.u', 'bc.p_s', 'bc.w_lg', 'fraction.gas']
    keep = [i for i in last.index if np.allclose(last.loc[i, ['CHK', 'PDC', 'WGL', 'FGAS']].to_numpy(dtype=float),
                                                 cfg.loc[i, stored].to_numpy(dtype=float), rtol=1e-6, atol=1e-9)]
    print(f'nsol-1: {len(keep)} of {len(last)} final states match their dataset row')
    cases, x_last = [], {}
    for well_id in rng.choice(sorted(keep), N_NSOL, replace=False):
        well = load_well(int(well_id), df_cfg)
        cid = f'nsol1-{well_id:04d}-last'
        cases.append(case_of(cid, 'nsol-1 final state', well.wp, well.bc, config_id=int(well_id)))
        x_last[cid] = np.array(json.loads(cfg.loc[well_id, 'x_last']))
        assert x_last[cid].size == 707
    return cases, x_last


def select_synthetic(df, rng):
    """Bernoulli choke and productivity-index inflow, which no v1 dataset uses."""
    cases = []
    for well_id in rng.choice(df['ID'].to_numpy(), N_SYNTH_WELLS, replace=False):
        base = load_well(int(well_id), df)
        for choke, inflow in (('bernoulli', 'vogel'), ('simpson', 'pi'), ('bernoulli', 'pi')):
            well = load_well(int(well_id), df)
            if choke == 'bernoulli':
                well.wp.choke = BernoulliChokeModel(K_c=base.wp.choke.K_c, chk_profile=base.wp.choke.chk_profile)
            if inflow == 'pi':
                well.wp.inflow = ProductivityIndex(k_l=base.wp.inflow.w_l_max / base.bc.p_r, f_g=base.wp.inflow.f_g)
            cases.append(case_of(f'synth-{well_id:04d}-{choke}-{inflow}', 'synthetic variant', well.wp, well.bc,
                                 config_id=int(well_id), note=f'{choke} choke, {inflow} inflow'))
    return cases


def convergence_groups(drawn, rng):
    """N, 2N and 4N members for drawn sol-1 cases, half of them two-root wells."""
    members = []
    for flag, n in ((True, (N_CONV_GROUPS + 1) // 2), (False, N_CONV_GROUPS // 2)):
        pool = [c for c in drawn if two_root(c['params']) == flag]
        for i in sorted(rng.choice(len(pool), min(n, len(pool)), replace=False)):
            base = pool[i]
            for factor in (1, 2, 4):
                N = base['n_cells'] * factor
                members.append(dict(base, case_id=f'{base["case_id"]}-N{N}', source='convergence', n_cells=N,
                                    group=base['case_id'], note=f'{factor}N of {base["case_id"]}'))
    return members


def select(args):
    df_sol = pd.read_csv(args.data_dir / 'manywells-sol-1_config.zip', compression='zip')
    df_nsol = pd.read_csv(args.data_dir / 'manywells-nsol-1_config.zip', compression='zip')
    fold = pd.read_csv(DATA / 'fold.csv')
    rng = np.random.default_rng(seed_for('case set'))

    # Every sol-1 well at its stored operating point (u = 0.5), cold only
    u05 = [case_of(f'sol1-{i:04d}-u05', 'sol-1 u = 0.5', w.wp, w.bc, config_id=int(i))
           for i in df_sol['ID'] for w in [load_well(int(i), df_sol)]]
    scan = run_all([{'case': c, 'multistart': False} for c in u05], args.processes)
    label = {c['case_id']: scan[c['case_id']][0]['cold'].get('slope') for c in u05}
    trickle = [c for c in u05 if label[c['case_id']] is not None and label[c['case_id']] > 0]
    stable = [c for c in u05 if label[c['case_id']] is not None and label[c['case_id']] < 0]
    print(f'u = 0.5 scan: {len(trickle)} trickle, {len(stable)} stable, {len(u05) - len(trickle) - len(stable)} failed')

    cases = [trickle[i] for i in sorted(rng.choice(len(trickle), min(N_TRICKLE_U05, len(trickle)), replace=False))]
    cases += [stable[i] for i in sorted(rng.choice(len(stable), N_STABLE_U05, replace=False))]
    drawn = select_drawn(df_sol, rng)
    nsol, x_last = select_nsol(df_nsol, args.data_dir, rng)
    cases += drawn + select_fresh() + select_fold(df_sol, fold) + select_lift(df_sol, rng) + nsol
    cases += select_synthetic(df_sol, rng) + convergence_groups(drawn, rng)
    assert len({c['case_id'] for c in cases}) == len(cases), 'duplicate case ids'

    write_json(cases, DATA / 'cases.json')
    starts = {f'{c["case_id"]}.warm': scan[f'sol1-{c["config_id"]:04d}-u05'][1]['cold']
              for c in cases if c['source'] in ('sol-1 drawn', 'convergence') and c['n_cells'] == 100
              and 'cold' in scan[f'sol1-{c["config_id"]:04d}-u05'][1]}
    starts |= {f'{cid}.dataset': x for cid, x in x_last.items()}
    np.savez_compressed(DATA / 'starts.npz', **starts)
    print(f'Wrote {len(cases)} cases: ' + ', '.join(f'{k} {v}' for k, v in Counter(c['source'] for c in cases).items()))


# ---------------------------------------------------------------------------------------------

def solve_all(args):
    cases = read_json(DATA / 'cases.json')
    rust = read_json(DATA / 'rust_roots.json')
    with np.load(DATA / 'starts.npz') as f:
        starts = dict(f)
    with np.load(DATA / 'rust_roots.npz') as f:
        rust_arrays = dict(f)

    def task(c, extra=None):
        s = {name: starts[f'{c["case_id"]}.{name}'] for name in ('warm', 'dataset') if f'{c["case_id"]}.{name}' in starts}
        s |= {f'rust {k}': rust_arrays[f'{c["case_id"]}.B{k}'] for k in range(len(rust.get(c['case_id'], [])))}
        return {'case': c, 'starts': s | (extra or {})}

    runs = run_all([task(c) for c in cases if c['n_cells'] == 100], args.processes)
    for factor in (2, 4):                       # convergence members, from the stable root one grid coarser
        tasks = []
        for c in cases:
            if c['source'] != 'convergence' or c['n_cells'] != 100 * factor:
                continue
            coarse = runs.get(f'{c["group"]}-N{c["n_cells"] // 2}')
            stable = [r for r in coarse[0]['roots'] if r['slope'] < 0] if coarse else []
            extra = {'interp': interpolate_state(coarse[1][stable[0]['key']], c['n_cells'] // 2, c['n_cells'])} if stable else {}
            tasks.append(task(c, extra))
        runs |= run_all(tasks, args.processes)

    write_json({cid: r for cid, (r, _) in runs.items()}, DATA / 'v1_runs.json')
    np.savez_compressed(DATA / 'v1_arrays.npz', **{f'{cid}.{k}': x for cid, (_, a) in runs.items() for k, x in a.items()})
    (DATA / 'coverage.md').write_text(coverage(cases, runs))
    print(f'Solved {len(runs)} cases')


def coverage(cases, runs) -> str:
    """Coverage from v1's cold outcome and the distinct roots found (before labelling)."""
    rows = []
    for c in cases:
        r = runs[c['case_id']][0]
        cold = r['cold']
        rows.append({'source': c['source'], 'two-root': two_root(c['params']), 'gas lift': c['params']['w_lg'] > 0,
                     'roots found': len(r['roots']),
                     'v1 cold': 'failed' if cold['status'] == 'failed' else ('unstable' if cold['slope'] > 0 else 'stable'),
                     'choked': cold.get('choked'), 'regime BH': cold.get('regime_bh'), 'regime WH': cold.get('regime_wh')})
    df = pd.DataFrame(rows)
    parts = [f'# Case set coverage (v1 cold outcome, before labelling)\n\n{len(df)} cases.\n']
    for col in ('source', 'two-root', 'gas lift', 'roots found', 'v1 cold', 'choked', 'regime BH', 'regime WH'):
        parts.append(f'\n## {col}\n\n' + '\n'.join(f'- {k}: {v}' for k, v in df[col].astype(str).value_counts().items()) + '\n')
    parts.append(f'\n## source by v1 cold outcome\n\n```\n{pd.crosstab(df["source"], df["v1 cold"]).to_string()}\n```\n')
    return ''.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    sub = parser.add_subparsers(dest='command', required=True)
    p_select = sub.add_parser('select', help='pick the cases')
    p_select.add_argument('--data-dir', type=Path, required=True)
    p_solve = sub.add_parser('solve', help='v1 roots on every case (after rust_roots.py roots)')
    for p in (p_select, p_solve):
        p.add_argument('--processes', type=int, default=min(22, mp.cpu_count()))
    args = parser.parse_args()
    {'select': select, 'solve': solve_all}[args.command](args)


if __name__ == '__main__':
    main()
