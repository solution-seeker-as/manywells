"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The measurements of specs/features/017-calibration.md that the twin study (scripts/calibration/twins.py) does not
make. Run from the project root:

    uv run python plans/evidence/calibration_measurements.py rows       # items 1 and 2: a row's cost, threads
    uv run python plans/evidence/calibration_measurements.py step       # item 3: the Jacobian's step
    uv run python plans/evidence/calibration_measurements.py valley     # item 7: the false minimum of well 3
    uv run python plans/evidence/calibration_measurements.py backends   # item 10: the backends' roots, wells 7 and 1
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from manywells.calibration import apply, synthetic_data
from manywells.calibration.fit import F_TOL, G_TOL, MAX_NFEV, X_TOL, priors_for, start
from manywells.calibration.objective import Objective
from manywells.configurations import DEVELOP
from manywells.sampling.conditions import nominal_conditions
from manywells.sampling.wells import rng_for, sample_well, well_properties
from manywells.simulator import NoOperatingPoint, SSDFSimulator

FREE = ['K_c', 'w_l_max', 'roughness', 'h']


def well(seed, k, n_cells):
    draw = sample_well(seed, k)
    return draw, well_properties(draw, configuration=DEVELOP, n_cells=n_cells), nominal_conditions(draw)


def rows():
    for k in range(4):
        for n_cells in (20, 50, 100):
            draw, wp, bc = well(1, k, n_cells)
            ms = []
            for u in (0.3, 0.6, 0.9):
                t = time.perf_counter()
                try:
                    SSDFSimulator(wp, backend='rust').simulate(replace(bc, u=u))
                except NoOperatingPoint:
                    pass
                ms.append(round(1e3 * (time.perf_counter() - t)))
            print(f'well {k} {draw.trajectory[0]:9s} {n_cells:3d} cells: ms per row at u = 0.3, 0.6, 0.9: {ms}')
    _, wp, bc = well(1, 1, 50)
    bcs = [replace(bc, u=u) for u in np.linspace(0.2, 1.0, 16)]
    for workers in (1, 4, 8, 16):
        t = time.perf_counter()
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda b: SSDFSimulator(wp, backend='rust').simulate(b), bcs))
        print(f'{workers:2d} threads: {time.perf_counter() - t:.2f} s for 16 rows')


def step():
    _, wp, bc = well(1, 1, 50)
    bc = replace(bc, u=0.6)

    def out(w):
        op = SSDFSimulator(w, backend='rust').simulate(bc)
        return np.array([op.p_0, op.state[-1, 0], op.state[-1, 6], np.log(op.w_res + op.w_g_res)])

    base = out(wp)
    for name in FREE:
        v0 = {'K_c': wp.choke.K_c, 'w_l_max': wp.inflow.w_l_max, 'roughness': wp.friction.roughness,
              'h': wp.thermal.h}[name]
        print(name)
        for h in 10.0 ** -np.arange(1, 9):
            d = (out(apply(wp, {name: v0 * np.exp(h)})) - base) / h
            print(f'  step {h:.0e}: d(PBH, PWH, TWH, log W)/d log {name} = {np.array2string(d, precision=5)}')


def valley():
    """Well 3 of seed 2026, periodic tests: the fit from the medians alone, with bounds, and from the start search."""
    seed, k = 2026, 3
    _, wp, bc = well(seed, k, 20)
    priors = priors_for(wp, FREE)
    z_true = np.clip(rng_for(seed, k, 'twin').normal(size=4), -2.5, 2.5)
    truth = {p.name: p.value(z) for p, z in zip(priors, z_true)}
    data = synthetic_data(wp, bc, truth, 20, seed=100 * k + 2, instrumentation='periodic_tests')

    def fit(z0, bound=np.inf):
        obj = Objective(wp, data, bc, priors)
        sol = least_squares(obj.residuals, z0, jac=obj.jacobian, method='trf', x_scale=1.0, xtol=X_TOL, ftol=F_TOL,
                            gtol=G_TOL, max_nfev=MAX_NFEV, bounds=(-bound, bound))
        return sol, obj

    obj = Objective(wp, data, bc, priors)
    print('z*', np.round(z_true, 3), '; cost at z*', round(0.5 * np.sum(obj.residuals(z_true) ** 2), 1),
          '; at the medians', round(0.5 * np.sum(obj.residuals(np.zeros(4)) ** 2), 1))
    for label, z0, bound in (('medians', np.zeros(4), np.inf), ('medians, |z| <= 4', np.zeros(4), 4.0),
                             ('start search', start(Objective(wp, data, bc, priors)), np.inf)):
        sol, o = fit(z0, bound)
        print(f'{label:18s}: start {np.round(z0, 2)} -> z {np.round(sol.x, 3)}, cost {sol.cost:.1f}, '
              f'{sol.message} ({o.n_solves} solves)')


def backends():
    for k in (7, 1):
        draw, small, bc = well(2026, k, 10)
        z_true = np.clip(rng_for(2026, k, 'twin').normal(size=4), -2.5, 2.5)
        sp = priors_for(small, ['K_c', 'h'])
        truth = apply(small, {p.name: p.value(z) for p, z in zip(sp, z_true[[0, 3]])})
        data = synthetic_data(truth, bc, {}, 4, seed=100 * k + 40, noisy=False)
        print(f'well {k} ({draw.trajectory[0]})')
        for _, row in data.rows.iterrows():
            b = replace(bc, u=float(row['CHK']), p_s=float(row['PDC']))
            sets = {name: SSDFSimulator(truth, backend=name).root_set(b) for name in ('rust', 'casadi')}
            print(f'  u = {b.u:.3f}: ' + '; '.join(f'{name} {[(round(r.p_0, 3), r.label) for r in rs.roots]}'
                                                  for name, rs in sets.items()))


if __name__ == '__main__':
    {'rows': rows, 'step': step, 'valley': valley, 'backends': backends}[sys.argv[1]]()
