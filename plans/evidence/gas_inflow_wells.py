"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 06 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Feature 022's measurements on sampled wells (specs/features/022-gas-rate-inflow.md, Acceptance 8): the sampler
draft's gas and condensate wells, solved on both backends with the interim productivity index (the option off) and
with Forchheimer's gas inflow as R3's draws give its a and b (the option on).

The wells are R3's draws (plans/sampler-redesign/sampling-draft.md), made by sampler_draft_draws.py, and each well's
first sol-2 sample (R4's draws, sampler_prototype.sol2_inputs), mapped to v2.0.0 by the prototype's interim mapping
(sampler_prototype.Builder): the black oil for every fluid class, and a ProductivityIndex whose k_l gives the liquid
the gas carries at the design drawdown. The gas-rate inflow replaces it by Forchheimer.from_gas_rate(a_gas, b_gas,
rho_g): R3's coefficients in bar2 per Sm3/d of separator gas, with mu_g Z at p_r (not D-61's fit, which R9's sampler
makes). Nothing else of the well changes.

For each case and each (inflow, backend) it records the roots and their labels, the operating point's PBH, PWH, TWH
and rates, whether it is choked, the core's work (marches, non-finite samples of R, rejected sign changes), and the
solve's time on one process (each worker is one process, OMP_NUM_THREADS=1). Against the inflow: the gas rate over
the absolute open flow (Forchheimer at 1.01325 bar), and the wellhead's no-slip mixture velocity over the erosional
velocity 1.22 c_e / sqrt(rho_ns) (API RP 14E, with R3's c_e, as the draws size the tubing).

The draft's scripts live on the branch sampler-redesign until it is merged; --draws names the directory that holds
sampler_draft_draws.py, sampler_draft_samples.py and sampler_prototype.py (by default this directory). The
measurements of the feature spec used them as committed at 8e6c175. From the repository root, with the sandbox off
(the pool needs Unix sockets):

    (ulimit -v 3000000; OMP_NUM_THREADS=1 timeout 7200 .venv/bin/python plans/evidence/gas_inflow_wells.py 600 \\
        --draws DIR --processes 8 --out OUT.parquet)
    ... --report OUT.parquet     the report again, without solving
"""

import argparse
import math
import multiprocessing
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from manywells.inflow import Forchheimer
from manywells.sampling.wells import rng_for
from manywells.simulator import SSDFSimulator

P_ATM = 1.01325  # bar
INFLOWS = ('interim', 'forchheimer')
BACKENDS = ('rust', 'casadi')
SAME_P0 = 1e-3  # bar: the backends' operating points are the same root
ROOT_ROW, ROOT_CHOKE_REL = 1e-8, 1e-6  # tests/backend_cases.py: a core-only root zeroes the CasADi rows


def _draft(draws):
    """The draft's modules, from the directory draws."""
    sys.path.insert(0, str(Path(draws).resolve()))
    import sampler_draft_draws  # noqa: E402
    import sampler_prototype  # noqa: E402

    return sampler_draft_draws, sampler_prototype


def cases(n, seed, draws):
    """(k, w, stations, inputs) of the gas and condensate wells among the first n draws, with their first sample."""
    dd, proto = _draft(draws)
    out = []
    for k in range(n):
        w, stations = dd.draw_well(seed, k)
        if w['fluid_type'] not in ('gas_condensate', 'gas'):
            continue
        lifted = w['lift'] == 'gas_lift'
        inp = dict(proto.well_fixed(w), lift_on=lifted, **proto.sol2_inputs(rng_for(seed, k, 0, 'v2', 'sample'), w))
        out.append((k, w, stations, inp))
    return out


def wellhead(root, wp):
    """The operating point's no-slip mixture velocity (m/s) and density (kg/m3) at the wellhead."""
    p, v_g, v_l, alpha, rho_g, rho_l, T = root.state[-1]
    A = wp.geometry.A
    w_g, w_l = A * alpha * rho_g * v_g, A * (1 - alpha) * rho_l * v_l
    q = w_g / rho_g + w_l / rho_l
    return q / A, (w_g + w_l) / q


def solve(wp, bc, backend):
    """One backend's root set: its summary, and the simulator and root set (None after an error)."""
    out = dict(error=None, n_roots=0, n_stable=0, has_op=False, t=math.nan)
    t0 = time.perf_counter()
    try:
        sim = SSDFSimulator(wp, backend=backend)
        rs = sim.root_set(bc)
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        out.update(error=f'{type(e).__name__}: {e}'[:300], t=time.perf_counter() - t0)
        return out, None, None
    out['t'] = time.perf_counter() - t0
    out.update(n_roots=len(rs.roots), n_stable=sum(r.label == 'stable' for r in rs.roots))
    out['p0_roots'] = ';'.join(f'{r.p_0:.6f}{r.label[0] if r.label else "-"}' for r in rs.roots)
    if backend == 'rust':
        c = sim._roots.counts
        out.update(marches=c['marches'], non_finite=c['non_finite'], rejected=c['rejected'])
    op = rs.operating_point
    if op is not None:
        v_m, rho_ns = wellhead(op, wp)
        out.update(
            has_op=True,
            PBH=op.p_0,
            PWH=op.state[-1, 0],
            TWH=op.state[-1, 6],
            w_g_res=op.w_g_res,
            w_res=op.w_res,
            choked=bool(op.choked),
            v_m=v_m,
            rho_ns=rho_ns,
        )
    return out, sim, rs


def core_only(rust_rs, casadi_sim, casadi_rs, bc):
    """The core's roots that no CasADi root matches (p_0 within SAME_P0), as (p_0, label, zeroes): whether the state
    zeroes the CasADi rows, every row but the choke row to ROOT_ROW and the choke row to ROOT_CHOKE_REL of the rate
    (tests/backend_cases.py; its allowance for p_0's resolution at a trickle root left out)."""
    system = casadi_sim.system
    ids, params = np.asarray(system.row_ids), system.params(bc)
    out = []
    for r in rust_rs.roots:
        if any(abs(r.p_0 - c.p_0) <= SAME_P0 for c in casadi_rs.roots):
            continue
        rows = np.asarray(system.residual(r.x, params)).ravel()
        w_m = r.w_res + r.w_g_res + bc.w_lg
        zeroes = (
            np.max(np.abs(rows[ids != 'CHK-1'])) < ROOT_ROW and abs(rows[ids == 'CHK-1'][0]) <= ROOT_CHOKE_REL * w_m
        )
        out.append((r.p_0, r.label, bool(zeroes)))
    return out


def task(args):
    k, w, stations, inp, draws = args
    _, proto = _draft(draws)
    rows = []
    try:
        builder = proto.Builder(w, stations, design=w if inp['lift_on'] else None)
        wp_pi, bc = builder.case(inp)
        gas = Forchheimer.from_gas_rate(w['a_gas'], w['b_gas'], wp_pi.fluid.rho_g)
        wells = {'interim': wp_pi, 'forchheimer': replace(wp_pi, inflow=gas)}
    except Exception as e:  # noqa: BLE001
        return [dict(draw=k, inflow=i, backend=b, error=f'build: {e}'[:300]) for i in INFLOWS for b in BACKENDS]
    aof = gas.gas_mass_flow_rate(P_ATM, bc.p_r)
    base = dict(
        draw=k,
        fluid_class=w['fluid_class'],
        tail=bool(w['tail']),
        p_r=bc.p_r,
        p_s=bc.p_s,
        u=bc.u,
        w_lg=bc.w_lg,
        gor=inp['gor'],
        wlr=inp['wlr'],
        D=w['D'],
        c_e=w['c_e'],
        rho_g=wp_pi.fluid.rho_g,
        aof=aof,
        f_g=wp_pi.fluid.f_g,
    )
    for name, wp in wells.items():
        solved = {}
        for backend in BACKENDS:
            try:
                res, sim, rs = solve(wp, bc, backend)
            except Exception:  # noqa: BLE001
                res, sim, rs = dict(error=traceback.format_exc(limit=2)[-300:]), None, None
            solved[backend] = (res, sim, rs)
        (res_r, _, rs_r), (_, sim_c, rs_c) = solved['rust'], solved['casadi']
        if rs_r is not None and rs_c is not None:
            only = core_only(rs_r, sim_c, rs_c, bc)
            res_r['core_only'] = ';'.join(
                f'{p:.4f}{label[0] if label else "-"}{"z" if z else "!"}' for p, label, z in only
            )
        for backend in BACKENDS:
            rows.append(base | dict(inflow=name, backend=backend) | solved[backend][0])
    return rows


def run(n, seed, draws, processes, out):
    todo = [(k, w, st, inp, draws) for k, w, st, inp in cases(n, seed, draws)]
    print(f'{len(todo)} gas and condensate wells among the first {n} draws (seed {seed})', flush=True)
    rows = []
    with multiprocessing.get_context('forkserver').Pool(processes, maxtasksperchild=20) as pool:
        for i, r in enumerate(pool.imap_unordered(task, todo, chunksize=1)):
            rows += r
            if (i + 1) % 20 == 0:
                print(f'{i + 1} of {len(todo)} wells', flush=True)
    df = pd.DataFrame(rows).sort_values(['draw', 'inflow', 'backend'])
    df.to_parquet(out)
    return df


def q(s, ps=(5, 50, 95)):
    s = pd.Series(s).dropna()
    return ' / '.join(f'{v:.3g}' for v in np.percentile(s, ps)) + f' (n={len(s)})' if len(s) else '-'


def report(df):
    df = df.copy()
    df['q_g'] = df.w_g_res * 86400 / df.rho_g / 1e6  # 1e6 Sm3/d
    df['v_e'] = 1.22 * df.c_e / np.sqrt(df.rho_ns)
    n = df.draw.nunique()
    print(f'{n} cases (one sample of each gas or condensate well)\n')
    print('Operating points found, two roots (liquid loading), choked, errors; per inflow and backend:')
    for (inflow, backend), g in df.groupby(['inflow', 'backend']):
        ok = g[g.error.isna()]
        line = (
            f'  {inflow:11s} {backend:6s}: operating point {int(g.has_op.sum()):4d} of {len(g)} '
            f'({100 * g.has_op.mean():.1f}%), roots 0/1/2/3+ {[int((ok.n_roots == k).sum()) for k in (0, 1, 2)]}'
            f'/{int((ok.n_roots >= 3).sum())}, two roots {100 * (ok.n_roots == 2).mean():.1f}%, '
            f'choked {100 * g.choked.fillna(False).astype(bool).mean():.1f}%, errors {g.error.notna().sum()}'
        )
        if backend == 'rust':
            line += f', with non-finite R {100 * (ok.non_finite > 0).mean():.1f}%'
        print(line)
    print('\nAt the operating point, 5th / 50th / 95th percentiles:')
    for (inflow, backend), g in df[df.has_op.fillna(False).astype(bool)].groupby(['inflow', 'backend']):
        print(f'  {inflow} {backend}:')
        print(f'    PBH (bar) {q(g.PBH)}; PWH (bar) {q(g.PWH)}; TWH (K) {q(g.TWH)}')
        print(
            f'    gas (1e6 Sm3/d) {q(g.q_g)}; liquid (kg/s) {q(g.w_res)}; drawdown (p_r - PBH)/p_r {q(1 - g.PBH / g.p_r)}'
        )
        print(
            f'    gas rate / AOF {q(g.w_g_res / g.aof)}; wellhead v_m / v_e {q(g.v_m / g.v_e)}, above 1: '
            f'{100 * (g.v_m > g.v_e).mean():.1f}%'
        )
    print('\nThe options against each other (the same backend):')
    for backend in BACKENDS:
        a = df[(df.backend == backend) & (df.inflow == 'interim')].set_index('draw')
        b = df[(df.backend == backend) & (df.inflow == 'forchheimer')].set_index('draw')
        both = a.has_op.fillna(False).astype(bool) & b.has_op.fillna(False).astype(bool)
        lost = a.has_op.fillna(False).astype(bool) & ~b.has_op.fillna(False).astype(bool)
        won = ~a.has_op.fillna(False).astype(bool) & b.has_op.fillna(False).astype(bool)
        print(
            f'  {backend}: both {both.sum()}, interim only {lost.sum()}, forchheimer only {won.sum()}; '
            f'gas rate forchheimer / interim {q((b.w_g_res / a.w_g_res)[both])}'
        )
    print('\nThe backends against each other (the same inflow):')
    for inflow in INFLOWS:
        r = df[(df.inflow == inflow) & (df.backend == 'rust')].set_index('draw')
        c = df[(df.inflow == inflow) & (df.backend == 'casadi')].set_index('draw')
        ro, co = r.has_op.fillna(False).astype(bool), c.has_op.fillna(False).astype(bool)
        same = ro & co & ((r.PBH - c.PBH).abs() <= SAME_P0)
        print(
            f'  {inflow}: operating point in both {(ro & co).sum()}, the same root {same.sum()}, rust only '
            f'{(ro & ~co).sum()}, casadi only {(~ro & co).sum()}; root counts differ in {(r.n_roots != c.n_roots).sum()}'
        )
        only = r.core_only.fillna('')
        roots = [x for v in only for x in v.split(';') if x]
        print(
            f'    roots only the core finds: {len(roots)} in {(only != "").sum()} cases, '
            f'{sum(x.endswith("z") for x in roots)} of them roots of the CasADi rows'
        )
        for k in (ro & ~co)[lambda s: s].index.tolist() + (~ro & co)[lambda s: s].index.tolist():
            print(
                f'    draw {k}: rust {r.loc[k, "p0_roots"]} {r.loc[k, "error"]}; casadi {c.loc[k, "p0_roots"]} '
                f'{c.loc[k, "error"]}'
            )
    print('\nCost per case on one process (s), 5th / 50th / 95th percentiles, and the total:')
    for (inflow, backend), g in df.groupby(['inflow', 'backend']):
        print(f'  {inflow:11s} {backend:6s}: {q(g.t)}, total {g.t.sum():.0f} s')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('n', type=int, nargs='?', default=600)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--draws', default=str(Path(__file__).parent))
    ap.add_argument('--processes', type=int, default=8)
    ap.add_argument('--out', default='gas_inflow_wells.parquet')
    ap.add_argument('--report', help='report a saved run, without solving')
    args = ap.parse_args()
    if args.report:
        report(pd.read_parquet(args.report))
        return
    report(run(args.n, args.seed, args.draws, args.processes, args.out))


if __name__ == '__main__':
    main()
