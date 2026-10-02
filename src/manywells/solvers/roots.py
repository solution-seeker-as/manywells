"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Multi-start root search with the stability label (specs/model/solution.md, SOL-1 to SOL-3).

The search solves the system from several starts: a given guess, the default march from
p_0 = p_r - 0.05 (p_r - p_s), and marches from p_0 = p_s + f (p_r - p_s) for f in START_FRACTIONS. It accepts a
solve that Ipopt reports as Solve_Succeeded and that is admissible (SOL-1), as the verifier's reference build does,
merges solutions within TOL_X of each other in the state distance, and labels each root by the sign of dR/dp_0
(SOL-3).

The starts are those of the verifier's reference search (method A: 0.5, 0.7, 0.85, 0.95, 0.995) and two more, 0.975
and 0.999. Most stable roots lie at f = 0.6 to 0.97 and most trickle roots at f > 0.995, and in some wells the
stable root lies just above f = 0.95, where a march from below cannot lift the rate. Measured on the verifier's case
set (2026-10-01, with the march's fallback): method A's starts give a stable-root rate of 97.1% and miss a root in 5
cases; with 0.975 and 0.999 the rate is 100% and every reference root set is complete, for twice the search time
(plans/manywells-v2-plan.md, Step 7). Bjarne accepted the two starts, under principle 7, on 2026-10-01.

The library cannot import the verifier, so the state distance, TOL_X and LABEL_MIN are copies of the verifier's
(manywells_verify.checks, specs/verification.md); tests/test_roots.py checks that they agree.
"""

import time
from dataclasses import dataclass

import casadi as ca
import numpy as np

from manywells.discretization import DIM_X
from manywells.solution import Root, RootSet
from manywells.solvers.ipopt import IpoptSolver
from manywells.solvers.march import MarchFailed, Marcher

TOL_X = 1e-4               # State distance within which two solutions are one root
LABEL_MIN = 1e-3           # |normalized dR/dp_0| at or below which a label is indeterminate
V_FLOOR = 1e-3             # m/s, floor on velocity scales in the state distance
DEFAULT_DRAWDOWN = 0.05    # The default start: p_0 = p_r - DEFAULT_DRAWDOWN (p_r - p_s)
START_FRACTIONS = (0.5, 0.7, 0.85, 0.975, 0.995, 0.999)  # Further starts: p_0 = p_s + f (p_r - p_s)


def state_distance(x, reference, bc) -> float:
    """
    Scaled ∞-norm distance between two states on the same grid: p by p_r - p_s, velocities and densities relative
    to the reference, alpha absolute, T by T_r - T_s.
    """
    X, Y = np.reshape(x, (-1, DIM_X)), np.reshape(reference, (-1, DIM_X))
    scale = np.column_stack([
        np.full(len(Y), max(bc.p_r - bc.p_s, 1.0)),
        np.maximum(np.abs(Y[:, 1]), V_FLOOR), np.maximum(np.abs(Y[:, 2]), V_FLOOR),
        np.ones(len(Y)),
        np.maximum(Y[:, 4], 1e-3), np.maximum(Y[:, 5], 1e-3),
        np.full(len(Y), max(bc.T_r - bc.T_s, 1.0)),
    ])
    return float(np.max(np.abs(X - Y) / scale))


def admissible(x, bc) -> bool:  # spec: SOL-1
    """p_s <= p <= p_r, 0 <= alpha <= 1, and positive velocities and densities at every point."""
    X = np.reshape(x, (-1, DIM_X))
    p, v_g, v_l, alpha, rho_g, rho_l, T = X.T
    return bool(np.all(np.isfinite(X)) and np.all((bc.p_s <= p) & (p <= bc.p_r)) and np.all((0 <= alpha) & (alpha <= 1))
                and np.all(v_g > 0) and np.all(v_l > 0) and np.all(rho_g > 0) and np.all(rho_l > 0))


def label_of(slope: float) -> str:
    return 'indeterminate' if abs(slope) <= LABEL_MIN else ('unstable' if slope > 0 else 'stable')


def normalized_slope(slope: float, x, p_r: float, p_s: float, A: float) -> float:
    """dR/dp_0 (kg/s per bar) at a root x, normalized by (p_r - p_s) / w_m with w_m the rate at the wellhead."""
    p, v_g, v_l, alpha, rho_g, rho_l, T = np.asarray(x)[-DIM_X:]
    w_m = A * (alpha * rho_g * v_g + (1 - alpha) * rho_l * v_l)
    return slope * (p_r - p_s) / max(w_m, 1e-3)


class StabilitySlope:  # spec: SOL-3
    """
    dR/dp_0 at a root (SOL-3): drop the choke row (CHK-1), treat p_0 as a parameter, and solve one linear system
    with the reduced Jacobian. Normalized by (p_r - p_s) / w_m, with w_m the rate at the wellhead.
    """

    def __init__(self, system):
        self._jac = system.residual.factory('jacobian', ['x', 'params'], ['jac:r:x'])
        self._choke = system.row_ids.index('CHK-1')
        self._A = system.wp.geometry.A
        n = system.n_x
        self._rows = [k for k in range(n) if k != self._choke]
        self._cols = list(range(1, n))  # p_0 is the first unknown

    def __call__(self, x, params) -> float:
        J = self._jac(x, params)
        c, rows, cols = self._choke, self._rows, self._cols
        dy = -ca.solve(J[rows, cols], J[rows, 0], 'csparse')
        slope = float(J[c, 0] + ca.mtimes(J[c, cols], dy))
        return normalized_slope(slope, x, params[0], params[1], self._A)


@dataclass(frozen=True)
class Attempt:
    """One start of a search and its outcome."""
    start: str          # 'x_guess', 'default' or 'p_s + f (p_r - p_s)' with f
    p_0: float          # p_0 the march started from (bar); NaN for a given guess
    outcome: str        # 'root at p_0 = ... bar', 'march failed', 'solve failed (<Ipopt status>)' or 'not admissible ...'
    seconds: float = 0.0  # Time taken by the march and the solve
    iterations: int = 0   # Ipopt iterations


class RootFinder:
    """The root search of one system, with its solver, march and stability slope built once."""

    def __init__(self, system, solver=None):
        self.system = system
        self.solver = solver or IpoptSolver(system)
        self.march = Marcher(system)
        self.slope = StabilitySlope(system)

    @staticmethod
    def starts(bc):
        """The march starts, as (name, p_0)."""
        dp = bc.p_r - bc.p_s
        return [('default', bc.p_r - DEFAULT_DRAWDOWN * dp)] + [(f'p_s + {f} (p_r - p_s)', bc.p_s + f * dp)
                                                                for f in START_FRACTIONS]

    def find(self, bc, x_guess=None) -> RootSet:  # spec: SOL-2
        """Every root found from the starts, labelled, with the operating point."""
        params = self.system.params(bc)
        lbx, ubx = self.system.bounds(bc)
        starts = ([('x_guess', None)] if x_guess is not None else []) + self.starts(bc)

        solutions, attempts = [], []
        for name, p_0 in starts:
            t0 = time.perf_counter()
            try:
                x0 = np.asarray(x_guess, dtype=float) if p_0 is None else self.march(params, p_0)
            except MarchFailed:
                attempts.append(Attempt(name, p_0, 'march failed', time.perf_counter() - t0))
                continue
            result = self.solver.solve(x0, params, lbx, ubx)
            if result.status != 'Solve_Succeeded':  # not Solved_To_Acceptable_Level, which CasADi reports as success
                attempts.append(Attempt(name, np.nan if p_0 is None else p_0, f'solve failed ({result.status})',
                                        time.perf_counter() - t0, result.stats.get('iter_count', 0)))
                continue
            if not admissible(result.x, bc):
                attempts.append(Attempt(name, np.nan if p_0 is None else p_0, 'not admissible', time.perf_counter() - t0,
                                        result.stats.get('iter_count', 0)))
                continue
            if not self.system.subsonic(result.x, params):  # spec: SOL-8
                attempts.append(Attempt(name, np.nan if p_0 is None else p_0, 'not admissible (a cell is supersonic)',
                                        time.perf_counter() - t0, result.stats.get('iter_count', 0)))
                continue
            same = [x for x in solutions if state_distance(result.x, x, bc) <= TOL_X]
            if not same:
                solutions.append(result.x)
            root = same[0] if same else result.x
            attempts.append(Attempt(name, np.nan if p_0 is None else p_0, f'root at p_0 = {root[0]:.4f} bar',
                                    time.perf_counter() - t0, result.stats.get('iter_count', 0)))

        roots = []
        for x in solutions:
            slope = self.slope(x, params)
            roots.append(Root(x=x, label=label_of(slope), slope=slope, **self.system.root_outputs(x, params)))
        return RootSet.of(roots, search=attempts)

    def flow_regimes(self, x) -> tuple:
        """The flow-regime label at each point of a state (SLIP-8)."""
        return self.system.flow_regimes(x)
