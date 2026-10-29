"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 29 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Feature 021's acceptance item 2 (specs/features/021-lift-gas-injection-depth.md): with the lift gas at the bottomhole,
every row of develop's test wells is the same to the bit as before the feature, in both backends.

  dump     for every configuration of the backend comparison's matrix (tests/backend_cases.py, 10 cells): both
           backends' rows at the configuration's trial state, the system's parameters and bounds, and both backends'
           root sets (every root's state, label and slope); and for the comparison set's first two wells of the groups
           develop, develop+lift-gas temperature and v1.0.0+lift-gas temperature (100 cells), both root sets and both
           backends' rows at each root perturbed. MODE is 'none', the wells as they are, or 'bottom', the wells with
           injection_md at their measured depth, which feature 021 must take as the bottomhole
  compare  compare two dumps bit for bit: every row, parameter, bound and root

Run dump in the tree before the feature with MODE 'none' (MODE 'bottom' needs the feature), and in the tree with it
with both modes, from the repository root (a few minutes each):

    PYTHONPATH=. uv run python plans/evidence/injection_depth_rows.py dump before.pkl none
    PYTHONPATH=. uv run python plans/evidence/injection_depth_rows.py dump after_none.pkl none
    PYTHONPATH=. uv run python plans/evidence/injection_depth_rows.py dump after_bottom.pkl bottom
    uv run python plans/evidence/injection_depth_rows.py compare before.pkl after_none.pkl
"""

import argparse
import pickle
from dataclasses import replace

import numpy as np


def with_mode(wp, mode):
    return wp if mode == 'none' else replace(wp, injection_md=wp.geometry.L)


def roots_of(rs):
    return [(r.x.copy(), r.label, r.slope, r.choked, r.w_res, r.w_g_res, tuple(r.flow_regime)) for r in rs.roots]


def dump(out, mode):
    from manywells.discretization import build_system
    from manywells.simulator import SSDFSimulator
    from manywells.solvers.rust import RustRootFinder
    from tests import backend_cases as cases

    data = {}
    for conf in cases.matrix():
        wp, bc = conf.inputs(10)
        if '021' in cases.features(wp):  # the feature's own configurations, which the tree before it lacks
            continue
        wp = with_mode(wp, mode)
        X = cases.trial_state(wp, bc)
        system = build_system(wp)
        ids, r_casadi = cases.casadi_rows(system, bc, X)
        ids_rust, r_rust = RustRootFinder(wp).rows(bc, X)
        data[conf.name] = dict(
            ids=tuple(ids),
            casadi=np.asarray(r_casadi),
            ids_rust=tuple(ids_rust),
            rust=np.asarray(r_rust),
            params=np.asarray(system.params(bc)),
            bounds=system.bounds(bc),
            roots_casadi=roots_of(SSDFSimulator(wp).root_set(bc)),
            roots_rust=roots_of(SSDFSimulator(wp, backend='rust').root_set(bc)),
        )
    for case in cases.comparison_set(
        2, configurations=['develop', 'develop+lift-gas temperature', 'v1.0.0+lift-gas temperature']
    ):
        wp, bc = case.inputs()
        wp = with_mode(wp, mode)
        system = build_system(wp)
        casadi = SSDFSimulator(wp).root_set(bc)
        rust = SSDFSimulator(wp, backend='rust').root_set(bc)
        rows = {}
        for k, r in enumerate(list(casadi.roots) + list(rust.roots)):
            X = cases.perturb(r.state)
            rows[k] = (np.asarray(cases.casadi_rows(system, bc, X)[1]), RustRootFinder(wp).rows(bc, X)[1])
        data[case.name] = dict(
            roots_casadi=roots_of(casadi),
            roots_rust=roots_of(rust),
            rows=rows,
            params=np.asarray(system.params(bc)),
            bounds=system.bounds(bc),
        )
    with open(out, 'wb') as f:
        pickle.dump(data, f)
    print(f'{len(data)} wells written to {out}')


def same(a, b):
    """Equal to the bit: arrays and floats by their bytes, containers element by element."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, float) and isinstance(b, float):
        return np.float64(a).tobytes() == np.float64(b).tobytes()
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def compare(before, after):
    A, B = (pickle.load(open(f, 'rb')) for f in (before, after))
    assert A.keys() == B.keys(), set(A) ^ set(B)
    differ = [(name, key) for name in A for key in A[name] if not same(A[name][key], B[name][key])]
    rows = sum(np.size(A[n][k]) for n in A for k in ('casadi', 'rust') if k in A[n])
    rows += sum(np.size(r[0]) + np.size(r[1]) for n in A for r in A[n].get('rows', {}).values())
    roots = sum(len(A[n]['roots_casadi']) + len(A[n]['roots_rust']) for n in A)
    for name, key in differ:
        print('differs:', name, key)
    print(f'{len(A)} wells, {rows} row values, {roots} roots: {len(differ)} differences')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    sub = parser.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('dump')
    d.add_argument('out')
    d.add_argument('mode', choices=('none', 'bottom'))
    c = sub.add_parser('compare')
    c.add_argument('before')
    c.add_argument('after')
    args = parser.parse_args()
    if args.cmd == 'dump':
        dump(args.out, args.mode)
    else:
        compare(args.before, args.after)


if __name__ == '__main__':
    main()
