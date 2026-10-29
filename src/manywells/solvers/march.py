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

Newton also fails where it returns a point whose rows are not zero. CasADi's Newton reports success where it steps
into a region in which the rows are NaN, and returns that point (issue #7): a march that runs out of pressure steps
to p < 0, where the black-oil properties are NaN. Its result is therefore accepted only if every row is finite and
at most NEWTON_ROW_TOL. On the verifier's case set, where no row is NaN, Newton's converged points leave at most
2.6e-9 and the check changes nothing. On the comparison set's 90 develop wells (tests/backend_cases.py, 6 per
configuration, 2026-10-16) it turned 101 marches with negative pressures into march failures, found the same 228
roots, and cut Ipopt's iterations in the search by 70% (specs/features/012-root-search.md); Bjarne accepted it,
under principle 7, on 2026-10-16.

Point 0 is retried where it fails from the bottom guess (issue #10). Where all the reservoir gas is dissolved at the
bottomhole, the smoothed phase rates of PVT-OIL-13 leave about 5e-4 kg/s of gas, and point 0's void fraction is 1e-4
to 1e-3. From the guess's 0.5, Newton fails there and the Ipopt fallback reports the point infeasible, so every
march near the root failed, and the search missed operating points that the Rust core finds. The retry starts point
0 from the phase rates' no-slip void fraction (no_slip_guess), and only where it fails, so every other start is
unchanged. On develop's sampled wells (263 cases at their GOR and at half and a quarter of it, 2026-10-28) the search
then finds all 116 operating points with an all-liquid bottomhole, where it missed 15, for 0.16% more Ipopt
iterations; on the verifier's case set it changes no root and no time (specs/features/012-root-search.md). Bjarne
chose it, on 2026-10-28, over guessing the no-slip void fraction at every start, which found the same operating
points for 11 to 20% more Ipopt iterations.
"""

import casadi as ca
import numpy as np

from manywells.discretization import DIM_X, PARAMS, lifts

QUIET = {'show_eval_warnings': False}  # A failed solve raises MarchFailed; CasADi need not warn as well
IPOPT = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0, 'show_eval_warnings': False, 'calc_lam_p': False}
LOWER = [0.0] * DIM_X                                          # v1.0.0's march: every unknown non-negative,
UPPER = [np.inf, np.inf, np.inf, 1.0, np.inf, np.inf, np.inf]  # and alpha <= 1
NEWTON_ROW_TOL = 1e-6  # Largest |row| of a point Newton solved (bar, K, m/s, kg/(m² s), kg/m³)
ALPHA_MIN = 1e-6       # Smallest void fraction of point 0's retry (issue #10)


class MarchFailed(RuntimeError):
    """A point of the march could not be solved."""


class _PointSolver:
    """Newton on the rows of one point, with Ipopt as the fallback; F(unknowns, parameters) -> rows."""

    def __init__(self, name, unknowns, parameters, rows, lower, upper):
        self._rows = ca.Function(f'F_{name}', [unknowns, parameters], [rows])
        self._newton = ca.rootfinder(f'{name}_newton', 'newton', self._rows, QUIET)
        self._ipopt = ca.nlpsol(f'{name}_ipopt', 'ipopt', {'x': unknowns, 'p': parameters, 'f': 0, 'g': rows}, IPOPT)
        self._lower, self._upper = lower, upper

    def __call__(self, guess, parameters) -> np.ndarray:
        try:
            x = np.asarray(self._newton(guess, parameters), dtype=float).ravel()
            r = np.asarray(self._rows(x, parameters), dtype=float)
            if np.all(np.isfinite(x)) and np.max(np.abs(r)) <= NEWTON_ROW_TOL:  # NaN rows fail this test
                return x
        except RuntimeError:
            pass
        result = self._ipopt(x0=guess, p=parameters, lbx=self._lower, ubx=self._upper, lbg=0, ubg=0)
        if not self._ipopt.stats()['success']:
            raise MarchFailed(self._ipopt.stats()['return_status'])
        return np.asarray(result['x'], dtype=float).ravel()


def no_slip_guess(guess):
    """
    Point 0's guess with the phase rates' no-slip void fraction, at most the guess's own, and the velocities that
    carry the same phase rates; None where that is the guess itself, or where nothing flows.
    """
    alpha, v_g, v_l = guess[3], guess[1], guess[2]
    q_g, q_l = alpha * v_g, (1 - alpha) * v_l       # Volume rates per unit area
    if not q_g + q_l > 0:
        return None
    no_slip = min(alpha, max(q_g / (q_g + q_l), ALPHA_MIN))
    if no_slip == alpha:
        return None
    retry = np.array(guess, dtype=float)
    retry[1], retry[2], retry[3] = q_g / no_slip, q_l / (1 - no_slip), no_slip
    return retry


class Marcher:
    """The march of a system, from a given p_0: march(params, p_0) -> x."""

    def __init__(self, system):
        self._system = system
        n_p = len(PARAMS)

        # Point 0: unknowns y = x_0 without p_0; parameters [p_0, params]
        y, q = ca.SX.sym('y', DIM_X - 1), ca.SX.sym('q', 1 + n_p)
        r_0 = system.bottom_rows(ca.vertcat(q[0], y), q[1:])
        self._bottom = _PointSolver('march_bottom', y, q, r_0, LOWER[1:], UPPER[1:])

        # Point i > 0: parameters [x_prev, w_res, params, delta_md, cos_incl, tvd_frac, lift, lift_prev]
        x_i, q = ca.SX.sym('x_i', DIM_X), ca.SX.sym('q', DIM_X + 1 + n_p + 5)
        k = DIM_X + 1 + n_p
        r_i = system.point_rows(x_i, q[:DIM_X], q[DIM_X], q[DIM_X + 1:k], q[k], q[k + 1], q[k + 2], q[k + 3], q[k + 4])
        self._cell = _PointSolver('march_cell', x_i, q, r_i, LOWER, UPPER)
        self._lift = lifts(system.wp)  # Whether the lift gas flows at each point (INF-9)

    def __call__(self, params, p_0) -> np.ndarray:
        system, geo = self._system, self._system.wp.geometry
        params = np.asarray(params, dtype=float)
        x = []
        try:
            guess = np.asarray(system.bottom_guess(p_0, params), dtype=float).ravel()
            x.append(np.concatenate([[p_0], self._bottom_point(guess, np.concatenate([[p_0], params]))]))
            w_res = float(system.reservoir_rate(p_0, params))
            for i in range(1, system.n_cells + 1):
                q = np.concatenate([x[-1], [w_res], params, [geo.delta_md[i - 1], geo.cos_incl[i - 1], geo.tvd_frac[i],
                                                            self._lift[i], self._lift[i - 1]]])
                x.append(self._cell(x[-1], q))
        except MarchFailed as e:
            raise MarchFailed(f'march from p_0 = {p_0:.3f} bar failed at point {len(x)}: {e}') from e
        x = np.concatenate(x)
        if not np.all(np.isfinite(x)):
            raise MarchFailed(f'march from p_0 = {p_0:.3f} bar gave a non-finite state')
        return x

    def _bottom_point(self, guess, parameters) -> np.ndarray:
        """Point 0's unknowns but p_0, from the guess, and where that fails, from its no-slip void fraction."""
        try:
            return self._bottom(guess[1:], parameters)
        except MarchFailed:
            retry = no_slip_guess(guess)
            if retry is None:
                raise
            return self._bottom(retry[1:], parameters)
