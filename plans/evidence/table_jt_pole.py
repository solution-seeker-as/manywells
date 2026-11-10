"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 09 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Why the Rust core was slow on volatile oils solved on their PVT tables with Joule-Thomson cooling (THM-8), a finding
of the sampler redesign's R9 pilots (specs/features/018-pvt-tables.md, Findings), and the measurement of the fix: the
core's DAK Joule-Thomson factor is NaN where Z + rho_r Z_rho <= 0, beyond its pole (PVT-GAS-10, Safeguards).

Run each mode under the environment before the fix (a checkout without it) and after, from that checkout's root, so
that each loads its own manywells._core, and compare:

    synthetic
        A hot volatile-oil well whose PVT is a DAK black oil tabulated with the viscosity and surface-tension columns
        (tests/test_rust_backend.py), the same well without those columns, and on the black oil itself: each solve's
        time, roots and search counts, and how many of 39 marches over p_0 hold a point at the pole (|J| > 1e3).
    march P_0
        The synthetic table well's march from P_0: each point's T, T_pr, reduced density, J and DAK's
        Z + rho_r Z_rho, which is zero at the pole.
    capture EVIDENCE TABLES OUT [--wells K0 K1] [--attempts N] [--seed S] [--other] [--only K ...]
        The R9 tables' wells: the first N attempts' (wp, bc) of each table well that table_fluids.py built in TABLES
        (or, with --other, of the wells without a table among draw indices K0 to K1 - 1; with --only, of the draw
        indices given), from sampler_prototype.py's sol2_well in the directory EVIDENCE: the sampler-redesign branch's
        plans/evidence, which has --tables. With N large, each well runs to its end: its time and outcome. R9's slow
        wells, after the fix: --only 26 47 83 118 162 199 249 482 --attempts 100000.
    solve CASES OUT [--casadi]
        Every captured attempt solved by the Rust core (or the CasADi backend): time, roots as bytes, search counts.
    compare BEFORE AFTER
        Two solve outputs: roots bit for bit, operating points, time and counts per fluid class, and per well.

    (ulimit -v 3000000; OMP_NUM_THREADS=1 timeout 900 .venv/bin/python plans/evidence/table_jt_pole.py synthetic)
"""

import argparse
import collections
import json
import pickle
import sys
import time
from dataclasses import replace

import numpy as np

from manywells.choke import SimpsonChokeModel
from manywells.friction import RoughnessFriction
from manywells.geometry import WellGeometry
from manywells.inflow import Vogel
from manywells.pvt.black_oil import BlackOil
from manywells.pvt.fluid import FluidModel
from manywells.pvt.table import PVTTable
from manywells.simulator import BoundaryConditions, SSDFSimulator, WellProperties
from manywells.solvers.rust import core_well, operating_point
from manywells.thermal import ThermalModel

DAK_A = (0.3265, -1.0700, -0.5339, 0.01569, -0.05165, 0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210)
POLE_J = 1e3  # |J| beyond which a point counts as at the pole: J is at most 5 at T_pr >= 1.15 (PVT-GAS-10)


def synthetic_well(columns=('mu_o', 'mu_g', 'sigma'), w_l_max=1700.0, n_cells=20):
    """The well of tests/test_rust_backend.py: draw 162 of the sol pilot, rounded, with a DAK black oil as a table."""
    bo = BlackOil(rho_o=792.0, rho_g=1.02, gas_law='dak')
    fluid = FluidModel(pvt=bo, rho_w=1030.0, gor=466.0, wlr=0.3)
    p, T = np.r_[1.0, np.arange(10.0, 410.0, 10.0)], np.arange(260.0, 460.0, 10.0)
    P, TT = np.meshgrid(p, T, indexing='ij')
    funcs = {
        'R_so': fluid.rs,
        'B_o': fluid.bo,
        'Z': fluid.z_factor,
        'mu_o': bo.oil_viscosity,
        'mu_g': lambda p, T: bo.gas_viscosity(p, T, fluid.gas_density(p, T)),
        'sigma': bo.oil_surface_tension,
    }
    cols = {
        k: np.vectorize(lambda a, b, f=f: float(f(a, b)))(P, TT)
        for k, f in funcs.items()
        if k in ('R_so', 'B_o', 'Z') + tuple(columns)
    }
    table = PVTTable(p=p, T=T, rho_o=bo.rho_o, rho_g=bo.rho_g, gor=fluid.gor, **cols)
    wp = WellProperties(
        geometry=WellGeometry.vertical(3000.0, n_cells, D=0.1571),
        fluid=replace(fluid, pvt=table),
        friction=RoughnessFriction(roughness=2.8e-5),
        thermal=ThermalModel(h=23.0),
        inflow=Vogel(w_l_max=w_l_max),
        choke=SimpsonChokeModel(K_c=1e-3),
    )
    return wp, BoundaryConditions(p_r=345.0, p_s=1.5, T_r=447.0, T_s=279.0, u=1.0)


def dak_denominator(r, t):
    """Z + r Z_r of DAK's equation of state (PVT-GAS-10's denominator)."""
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = DAK_A
    c1 = a1 + a2 / t + a3 / t**3 + a4 / t**4 + a5 / t**5
    c2, c3, c4 = a6 + a7 / t + a8 / t**2, a9 * (a7 / t + a8 / t**2), a10 / t**3
    e = np.exp(-a11 * r * r)
    z = 1 + c1 * r + c2 * r * r - c3 * r**5 + c4 * r * r * (1 + a11 * r * r) * e
    z_r = c1 + 2 * c2 * r - 5 * c3 * r**4 + 2 * c4 * r * e * (1 + a11 * r * r - a11 * a11 * r**4)
    return z + r * z_r


def solve(wp, bc, backend='rust'):
    sim = SSDFSimulator(wp, backend=backend)
    t0 = time.perf_counter()
    try:
        rs, err = sim.root_set(bc), None
    except Exception as e:  # noqa: BLE001 - recorded, as the prototype records a solver failure
        rs, err = None, f'{type(e).__name__}: {e}'[:200]
    rec = dict(t=time.perf_counter() - t0, error=err)
    if rs is not None:
        rec['roots'] = [(np.asarray(r.x, dtype=float).tobytes(), r.label) for r in rs.roots]
        rec['op'] = None if rs.operating_point is None else np.asarray(rs.operating_point.x, dtype=float).tobytes()
        if backend == 'rust':
            rec['counts'] = dict(sim._roots.counts)
    return rec


def marches_at_the_pole(wp, bc, n=41):
    core, op = core_well(wp), operating_point(wp, bc)
    hits = 0
    for p_0 in np.linspace(bc.p_s, bc.p_r, n)[1:-1]:
        x = np.asarray(core.march(op, p_0)[0]).reshape(-1, 7)
        js = [core._component('jt_factor', [s[6], s[4]])[0] for s in x if np.isfinite(s[0])]
        hits += any(not abs(j) < POLE_J for j in js)
    return hits


def synthetic(_):
    wp, bc = synthetic_well()
    variants = {
        'table with every column': wp,
        'table without mu_o, mu_g, sigma': synthetic_well(columns=())[0],
        'black oil': replace(wp, fluid=replace(wp.fluid, pvt=BlackOil(rho_o=792.0, rho_g=1.02, gas_law='dak'))),
        'table with every column, THM-8 off': replace(wp, thermal=replace(wp.thermal, joule_thomson=False)),
    }
    for name, w in variants.items():
        r = solve(w, bc)
        c = r['counts']
        roots = [round(float(np.frombuffer(x)[0]), 3) for x, _ in r['roots']]
        print(
            f'{name}: {r["t"]:.3f} s, roots p_0 {roots}, marches {c["marches"]}, temperature solves '
            f'{c["temperature_solves"]} ({c["temperature_failures"]} unsolved), edge refinements '
            f'{c["edge_refinements"]}; marches at the pole {marches_at_the_pole(w, bc)} of 39'
        )


def march(args):
    wp, bc = synthetic_well()
    core, op = core_well(wp), operating_point(wp, bc)
    ppc, tpc = wp.fluid.pvt.pseudo_critical
    x = np.asarray(core.march(op, args.p_0)[0]).reshape(-1, 7)
    for i, s in enumerate(x):
        if not np.isfinite(s[0]):
            print(f'point {i}: no state')
            break
        r = float(wp.fluid.pvt.reduced_density(s[4]))
        j = core._component('jt_factor', [s[6], s[4]])[0]
        print(
            f'point {i}: p {s[0]:8.3f} T {s[6]:8.3f} T_pr {s[6] / tpc:.3f} rho_r {r:.4f} v_g {s[1]:8.1f} '
            f'J {j:11.4g} Z + rho_r Z_rho {dak_denominator(r, s[6] / tpc):+.3e}'
        )


def capture(args):
    sys.path.insert(0, args.evidence)
    import sampler_prototype as sp
    from sampler_draft_draws import draw_well

    table_wells = {w['well']: w['fluid_class'] for w in json.load(open(f'{args.tables}/wells.json'))['wells']}
    if args.only:
        wells = [(k, table_wells.get(k) or draw_well(args.seed, k)[0]['fluid_class']) for k in args.only]
    elif args.other:
        k0, k1 = args.wells
        wells = [(k, draw_well(args.seed, k)[0]['fluid_class']) for k in range(k0, k1) if k not in table_wells]
    else:
        wells = sorted(table_wells.items())
    cases, state, orig = [], {}, sp.solve

    class Stop(Exception):
        pass

    def record(wp, bc, backend='rust', x_guess=None):
        t0 = time.perf_counter()
        res = orig(wp, bc, backend, x_guess)
        t = time.perf_counter() - t0
        cases.append(dict(well=state['k'], cls=state['cls'], attempt=state['n'], wp=wp, bc=bc, t=t))
        state['n'] += 1
        if state['n'] >= args.attempts:
            raise Stop
        return res

    sp.solve = record
    for k, cls in wells:
        state.update(k=k, cls=cls, n=0)
        t0 = time.perf_counter()
        try:
            outcome = sp.sol2_well(args.seed, k, tables=args.tables)['info']['outcome']
        except Stop:
            outcome = f'stopped after {args.attempts} attempts'
        t_solves = sum(c['t'] for c in cases if c['well'] == k)
        print(
            f'well {k} {cls}: {time.perf_counter() - t0:.1f} s ({t_solves:.1f} s in solves), {state["n"]} attempts, '
            f'{outcome}',
            flush=True,
        )
    with open(args.out, 'wb') as f:
        pickle.dump(cases, f)
    print(f'{len(cases)} attempts of {len(wells)} wells')


def solve_all(args):
    cases = pickle.load(open(args.cases, 'rb'))
    out = [
        dict(well=c['well'], cls=c['cls'], attempt=c['attempt'], **solve(c['wp'], c['bc'], args.backend)) for c in cases
    ]
    with open(args.out, 'wb') as f:
        pickle.dump(out, f)
    print(f'{len(out)} attempts in {sum(r["t"] for r in out):.1f} s of solves')


def compare(args):
    a, b = pickle.load(open(args.before, 'rb')), pickle.load(open(args.after, 'rb'))
    by_cls = collections.defaultdict(list)
    for x, y in zip(a, b, strict=True):
        by_cls[x['cls']].append((x, y))
    for cls, pairs in by_cls.items():
        ta, tb = np.array([x['t'] for x, _ in pairs]), np.array([y['t'] for _, y in pairs])
        same = sum(x.get('roots') == y.get('roots') and x.get('op') == y.get('op') for x, y in pairs)
        work = sum(x.get('counts') == y.get('counts') for x, y in pairs)
        ops = [sum(r.get('op') is not None for r in rs) for rs in zip(*pairs)]
        print(
            f'{cls}: {len(pairs)} attempts, roots identical in {same}, search counts identical in {work}, '
            f'operating points {ops[0]} -> {ops[1]}; time {ta.sum():.1f} -> {tb.sum():.1f} s, median '
            f'{np.median(ta) * 1e3:.0f} -> {np.median(tb) * 1e3:.0f} ms, P95 {np.percentile(ta, 95):.2f} -> '
            f'{np.percentile(tb, 95):.2f} s, max {ta.max():.2f} -> {tb.max():.2f} s'
        )
    wells = collections.defaultdict(lambda: np.zeros(4))
    for x, y in zip(a, b):
        wells[(x['well'], x['cls'])] += [
            x['t'],
            y['t'],
            x.get('counts', {}).get('edge_refinements', 0),
            y.get('counts', {}).get('edge_refinements', 0),
        ]
    for (k, cls), (ta, tb, ea, eb) in sorted(wells.items()):
        if max(ta, tb) > 1.2 * min(ta, tb) and abs(ta - tb) > 0.1:
            print(f'  well {k} {cls}: {ta:.2f} -> {tb:.2f} s, edge refinements {ea:.0f} -> {eb:.0f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[1], formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    sub.add_parser('synthetic').set_defaults(func=synthetic)
    m = sub.add_parser('march')
    m.add_argument('p_0', type=float)
    m.set_defaults(func=march)
    c = sub.add_parser('capture')
    c.add_argument('evidence')
    c.add_argument('tables')
    c.add_argument('out')
    c.add_argument('--wells', type=int, nargs=2, default=(0, 560))
    c.add_argument('--attempts', type=int, default=4)
    c.add_argument('--seed', type=int, default=2026)
    c.add_argument('--other', action='store_true')
    c.add_argument('--only', type=int, nargs='+', help='these draw indices alone')
    c.set_defaults(func=capture)
    s = sub.add_parser('solve')
    s.add_argument('cases')
    s.add_argument('out')
    s.add_argument('--casadi', dest='backend', action='store_const', const='casadi', default='rust')
    s.set_defaults(func=solve_all)
    d = sub.add_parser('compare')
    d.add_argument('before')
    d.add_argument('after')
    d.set_defaults(func=compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
