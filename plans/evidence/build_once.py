"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The build-once measurement of specs/architecture.md (plans/improvements.md, 4.1), repeated on the verifier's case
set in Step 7: the time of one solve of a well, before Step 7 (develop at c5fedf2, which rebuilt the system, the NLP
and the march's rootfinders on every call) and after (built once per well, with the operating point as parameters).

For each case, both runs time a warm solve (from the case's v1.0.0 root, the generator's reuse of a well's root) and
a cold solve (the default march from p_0 = p_r - 0.05 (p_r - p_s), then Ipopt). The new run also times the build.
Before Step 7, develop had no v1.0.0 configuration, so the old run uses its nearest one (vertical, dead oil, ideal
gas, fixed f_D), which has the same size and nearly the same rows. From the project root:

    mkdir -p /tmp/c5fedf2 && git archive c5fedf2 src | tar -x -C /tmp/c5fedf2
    PYTHONPATH=/tmp/c5fedf2/src uv run python plans/evidence/build_once.py --old out_old.json
    uv run python plans/evidence/build_once.py out_new.json
    uv run python plans/evidence/build_once.py --compare out_old.json out_new.json
"""

import argparse
import contextlib
import io
import json
import multiprocessing
import time
import warnings
from pathlib import Path

import numpy as np

CASES = Path('verification/data/cases.parquet')
ROOTS = Path('verification/data/reference_roots.parquet')


def well_new(case):
    from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
    from manywells.configurations import v1_well
    from manywells.inflow import ProductivityIndex, Vogel
    from manywells.simulator import BoundaryConditions
    prm, v = case.params, case.variant
    inflow = Vogel(w_l_max=prm['w_l_max']) if v.inflow == 'vogel' else ProductivityIndex(k_l=prm['k_l'])
    choke = (SimpsonChokeModel if v.choke == 'simpson' else BernoulliChokeModel)(K_c=prm['K_c'], chk_profile=v.profile)
    wp = v1_well(L=prm['L'], D=prm['D'], rho_l=prm['rho_l'], R_s=prm['R_s'], cp_g=prm['cp_g'], cp_l=prm['cp_l'],
                 f_D=prm['f_D'], h=prm['h'], f_g=prm['f_g'], inflow=inflow, choke=choke, n_cells=case.n_cells)
    bc = BoundaryConditions(p_r=prm['p_r'], p_s=prm['p_s'], T_r=prm['T_r'], T_s=prm['T_s'], u=prm['u'], w_lg=prm['w_lg'])
    return wp, bc


def well_old(case):
    """develop at c5fedf2, in its nearest configuration to v1.0.0 (tests/test_spec_vectors.py at that commit)."""
    from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
    from manywells.geometry import WellGeometry
    from manywells.inflow import ProductivityIndex, Vogel
    from manywells.pvt.fluid import FluidModel
    from manywells.simulator import BoundaryConditions, WellProperties
    from manywells.units import P_REF, T_REF
    prm, v = case.params, case.variant
    rho_g = P_REF / (prm['R_s'] * T_REF)
    fluid = FluidModel(rho_o=prm['rho_l'], rho_g=rho_g, wlr=0.0, gor=prm['f_g'] * prm['rho_l'] / ((1 - prm['f_g']) * rho_g),
                       oil_model='dead_oil', ideal_gas=True, cp_g=prm['cp_g'], cp_o=prm['cp_l'])
    inflow = Vogel(prm['w_l_max']) if v.inflow == 'vogel' else ProductivityIndex(prm['k_l'])
    choke = (SimpsonChokeModel if v.choke == 'simpson' else BernoulliChokeModel)(K_c=prm['K_c'], chk_profile=v.profile)
    wp = WellProperties(geometry=WellGeometry.vertical(length=prm['L'], n_cells=case.n_cells, D=prm['D']), fluid=fluid,
                        f_D=prm['f_D'], h=prm['h'], inflow=inflow, choke=choke)
    bc = BoundaryConditions(p_r=prm['p_r'], p_s=prm['p_s'], T_r=prm['T_r'], T_s=prm['T_s'], u=prm['u'], w_lg=prm['w_lg'])
    return wp, bc


def measure_old(task):
    case, x_root = task
    from manywells.simulator import SimError, SSDFSimulator
    wp, bc = well_old(case)
    out = {}
    for name, guess in (('cold', None), ('warm', list(x_root))):
        sim = SSDFSimulator(wp, bc)
        sim.x_guess = guess
        t = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sim.simulate()
            ok = True
        except SimError:
            ok = False
        out[name] = (time.perf_counter() - t, ok)
    return case.case_id, out


def measure_new(task):
    case, x_root = task
    from manywells.discretization import build_system
    from manywells.solvers.ipopt import IpoptSolver
    from manywells.solvers.march import MarchFailed, Marcher
    from manywells.solvers.roots import DEFAULT_DRAWDOWN
    wp, bc = well_new(case)
    t = time.perf_counter()
    system = build_system(wp)
    solver, march = IpoptSolver(system), Marcher(system)
    out = {'build': (time.perf_counter() - t, True)}
    params, (lbx, ubx) = system.params(bc), system.bounds(bc)
    t = time.perf_counter()
    try:
        x0 = march(params, bc.p_r - DEFAULT_DRAWDOWN * (bc.p_r - bc.p_s))
        ok = solver.solve(x0, params, lbx, ubx).success
    except MarchFailed:
        ok = False
    out['cold'] = (time.perf_counter() - t, ok)
    t = time.perf_counter()
    ok = solver.solve(np.asarray(x_root), params, lbx, ubx).success
    out['warm'] = (time.perf_counter() - t, ok)
    return case.case_id, out


def compare(old_path, new_path):
    old, new = json.loads(Path(old_path).read_text()), json.loads(Path(new_path).read_text())
    ids = sorted(set(old) & set(new))
    for kind in ('warm', 'cold'):
        o = np.array([old[c][kind][0] for c in ids])
        n = np.array([new[c][kind][0] for c in ids])
        print(f'{kind}: before {np.median(o):.3f} s, after {np.median(n):.4f} s (medians); '
              f'speed-up median {np.median(o / n):.0f}x, total {o.sum() / n.sum():.0f}x over {len(ids)} cases')
    b = np.array([new[c]['build'][0] for c in ids])
    print(f'build, once per well: median {np.median(b):.2f} s, max {b.max():.2f} s')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    parser.add_argument('out', nargs='+')
    parser.add_argument('--old', action='store_true', help='run with develop at c5fedf2 on the path')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--processes', type=int, default=8)
    args = parser.parse_args()
    if args.compare:
        return compare(*args.out)
    from manywells_verify.cases import read_cases, read_roots
    cases, roots = read_cases(CASES), read_roots(ROOTS)
    tasks = [(c, roots[k][0].x) for k, c in cases.items() if roots.get(k)]
    warnings.simplefilter('ignore')
    with multiprocessing.Pool(args.processes) as pool:
        results = pool.map(measure_old if args.old else measure_new, tasks, chunksize=1)
    Path(args.out[0]).write_text(json.dumps(dict(results), indent=1))
    print(f'{len(results)} cases measured')


if __name__ == '__main__':
    main()
