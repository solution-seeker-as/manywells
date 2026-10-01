"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The initial-guess march: fix the bottomhole pressure p_0, solve point 0's rows for the other six unknowns, then
each point's rows in turn up to the wellhead, without the choke row. The solvers are built once per well.

Each point is solved by Newton's method from the previous point's state. Where Newton fails, the point is solved
by Ipopt on the same rows, with the unknowns bounded as v1.0.0's march bounded them (non-negative, alpha <= 1).
Newton fails at the slug-annular transition of the classifier (alpha near 0.7), where the state changes steeply
along the well. Measured on the verifier's case set (2026-10-01), the fallback raised the stable-root rate from
98.5% to 100% and completed every reference root set, for 13% more search time (plans/manywells-v2-plan.md, Step 7);
Bjarne accepted it, under principle 7, on 2026-10-01.
"""

import casadi as ca
import numpy as np

from manywells.discretization import DIM_X, PARAMS

QUIET = {'show_eval_warnings': False}  # A failed solve raises MarchFailed; CasADi need not warn as well
IPOPT = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0, 'show_eval_warnings': False, 'calc_lam_p': False}
LOWER = [0.0] * DIM_X                                          # v1.0.0's march: every unknown non-negative,
UPPER = [np.inf, np.inf, np.inf, 1.0, np.inf, np.inf, np.inf]  # and alpha <= 1


class MarchFailed(RuntimeError):
    """A point of the march could not be solved."""


class _PointSolver:
    """Newton on the rows of one point, with Ipopt as the fallback; F(unknowns, parameters) -> rows."""

    def __init__(self, name, unknowns, parameters, rows, lower, upper):
        self._newton = ca.rootfinder(f'{name}_newton', 'newton', ca.Function(f'F_{name}', [unknowns, parameters],
                                                                             [rows]), QUIET)
        self._ipopt = ca.nlpsol(f'{name}_ipopt', 'ipopt', {'x': unknowns, 'p': parameters, 'f': 0, 'g': rows}, IPOPT)
        self._lower, self._upper = lower, upper

    def __call__(self, guess, parameters) -> np.ndarray:
        try:
            return np.asarray(self._newton(guess, parameters), dtype=float).ravel()
        except RuntimeError:
            result = self._ipopt(x0=guess, p=parameters, lbx=self._lower, ubx=self._upper, lbg=0, ubg=0)
            if not self._ipopt.stats()['success']:
                raise MarchFailed(self._ipopt.stats()['return_status'])
            return np.asarray(result['x'], dtype=float).ravel()


class Marcher:
    """The march of a system, from a given p_0: march(params, p_0) -> x."""

    def __init__(self, system):
        self._system = system
        n_p = len(PARAMS)

        # Point 0: unknowns y = x_0 without p_0; parameters [p_0, params]
        y, q = ca.SX.sym('y', DIM_X - 1), ca.SX.sym('q', 1 + n_p)
        r_0 = system.bottom_rows(ca.vertcat(q[0], y), q[1:])
        self._bottom = _PointSolver('march_bottom', y, q, r_0, LOWER[1:], UPPER[1:])

        # Point i > 0: parameters [x_prev, w_res, params, delta_md, cos_incl, tvd_frac]
        x_i, q = ca.SX.sym('x_i', DIM_X), ca.SX.sym('q', DIM_X + 1 + n_p + 3)
        k = DIM_X + 1 + n_p
        r_i = system.point_rows(x_i, q[:DIM_X], q[DIM_X], q[DIM_X + 1:k], q[k], q[k + 1], q[k + 2])
        self._cell = _PointSolver('march_cell', x_i, q, r_i, LOWER, UPPER)

    def __call__(self, params, p_0) -> np.ndarray:
        system, geo = self._system, self._system.wp.geometry
        params = np.asarray(params, dtype=float)
        x = []
        try:
            guess = np.asarray(system.bottom_guess(p_0, params), dtype=float).ravel()
            x.append(np.concatenate([[p_0], self._bottom(guess[1:], np.concatenate([[p_0], params]))]))
            w_res = float(system.reservoir_rate(p_0, params))
            for i in range(1, system.n_cells + 1):
                q = np.concatenate([x[-1], [w_res], params, [geo.delta_md[i - 1], geo.cos_incl[i - 1], geo.tvd_frac[i]]])
                x.append(self._cell(x[-1], q))
        except MarchFailed as e:
            raise MarchFailed(f'march from p_0 = {p_0:.3f} bar failed at point {len(x)}: {e}') from e
        x = np.concatenate(x)
        if not np.all(np.isfinite(x)):
            raise MarchFailed(f'march from p_0 = {p_0:.3f} bar gave a non-finite state')
        return x
