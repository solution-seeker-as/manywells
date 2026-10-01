"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Ipopt adapter: solves the system of a well as a feasibility NLP (objective zero), built once per well with the
operating point as NLP parameters.
"""

from dataclasses import dataclass

import casadi as ca
import numpy as np

from manywells.discretization import PARAMS


@dataclass(frozen=True, eq=False)
class SolveResult:
    x: np.ndarray       # The solver's last iterate
    success: bool       # Ipopt reported success
    status: str         # Ipopt's return status
    stats: dict         # CasADi's solver statistics


class IpoptSolver:
    """Ipopt on the feasibility NLP of a system: find x with residual(x, params) = 0 within the bounds."""

    # Quiet: no banner, no iteration log, no CasADi warnings about NaN in a trial step, and no multipliers of the
    # parameters, which a feasibility problem does not need
    OPTIONS = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0, 'show_eval_warnings': False,
               'calc_lam_p': False}

    def __init__(self, system, options=None):
        x = ca.SX.sym('x', system.n_x)
        p = ca.SX.sym('params', len(PARAMS))
        nlp = {'x': x, 'p': p, 'f': 0, 'g': system.residual(x, p)}
        self._nlp = ca.nlpsol('ssdf', 'ipopt', nlp, self.OPTIONS | (options or {}))

    def solve(self, x0, params, lbx, ubx) -> SolveResult:
        result = self._nlp(x0=x0, p=params, lbx=lbx, ubx=ubx, lbg=0, ubg=0)
        stats = self._nlp.stats()
        return SolveResult(x=np.asarray(result['x'], dtype=float).ravel(), success=bool(stats['success']),
                           status=str(stats['return_status']), stats=stats)
