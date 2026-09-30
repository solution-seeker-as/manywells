"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Stability label from v1.0.0's residual graph (v2 plan, "Multiple roots and stability").

Builds v1's residual g(x) exactly as SSDFSimulator.simulate() does, solves one well from
several initial guesses, and labels each root by the sign of d(choke row)/dp0: drop the
choke row, treat p0 as a parameter, and solve one linear system with the reduced Jacobian.
Positive is unstable (trickle root), negative is stable.

Runs against v1.0.0, from the root of a v1.0.0 checkout (or set MANYWELLS_V1 to it):

    python <repo>/plans/evidence/stability_label.py --config manywells-sol-1_config.zip 977
"""

import argparse
import contextlib
import io
import os
import sys

import casadi as ca
import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get('MANYWELLS_V1', os.getcwd()))
from manywells.simulator import SSDFSimulator, SimError  # noqa: E402  (v1.0.0 API)
from scripts.load_well_from_dataset import load_well  # noqa: E402


class Graph(SSDFSimulator):
    """v1.0.0 simulator that also exposes its residual g(x), Jacobian and choke row."""

    def build(self):
        x, g = [], []
        for i in range(self.n_cells + 1):
            x_i = self._create_variables(i)
            g_i = []
            if i == 0:
                g_i += self._left_boundary_eqs(x_i)
            else:
                g_i += self._differential_equations(x_i, x[self.dim_x * (i - 1):self.dim_x * i], i)
            if i == self.n_cells:
                self.choke_row = len(g) + len(g_i)
                g_i += self._right_boundary_eqs(x_i)
            g_i += self._closure_relations(x_i)
            x += x_i
            g += g_i
        xv, gv = ca.vertcat(*x), ca.vertcat(*g)
        self.g_fun = ca.Function('g', [xv], [gv])
        self.J_fun = ca.Function('J', [xv], [ca.jacobian(gv, xv)])
        return self

    def dR_dp0(self, xs):
        """d(choke row)/dp0 along the manifold where all other rows hold, and the reduced condition number."""
        J = np.array(self.J_fun(xs))
        c, k = self.choke_row, 0  # choke row; p0 is the first state variable
        rows = [r for r in range(J.shape[0]) if r != c]
        cols = [q for q in range(J.shape[1]) if q != k]
        dxr = -np.linalg.solve(J[np.ix_(rows, cols)], J[rows, k])
        return J[c, k] + J[c, cols] @ dxr, np.linalg.cond(J[np.ix_(rows, cols)])

    def R_of_p0(self, p0):
        """v1-only shooting residual: cellwise march at fixed p0, then the choke row (NaN if p_L < p_s)."""
        xs = self._simulate_cellwise(p0, self.bc.T_r)
        return float(self.g_fun(xs)[self.choke_row]), xs


def load_graph(config_path, well_id):
    df = pd.read_csv(config_path, compression='zip')
    well = load_well(well_id, df)
    return Graph(well.wp, well.bc).build()


def describe(sim, xs, tag):
    xs = np.asarray(xs)
    X = xs.reshape(-1, 7)  # [p, v_g, v_l, alpha, rho_g, rho_l, T] per grid point
    r = np.array(sim.g_fun(xs)).ravel()
    d, cond = sim.dR_dp0(xs)
    w_l = sim.wp.A * (1 - X[0, 3]) * X[0, 5] * X[0, 2]
    print(f'{tag}: p0={X[0, 0]:.4f} pL-ps={X[-1, 0] - sim.bc.p_s:.2e} bar TWH={X[-1, 6]:.1f} K '
          f'w_l={w_l:.3f} kg/s |g|inf={np.abs(r).max():.1e} min v={X[:, 1:3].min():.2e} '
          f'dR/dp0={d:+.3e} -> {"UNSTABLE" if d > 0 else "stable"}  cond={cond:.1e}')
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[2])
    parser.add_argument('--config', required=True, help='path to manywells-sol-1_config.zip')
    parser.add_argument('well_id', type=int, nargs='?', default=977)
    args = parser.parse_args()

    sim = load_graph(args.config, args.well_id)
    wp, bc = sim.wp, sim.bc
    print(f'well {args.well_id}: p_r={bc.p_r:.2f} p_s={bc.p_s:.2f} '
          f'rho_l g L/1e5={wp.rho_l * 9.81 * wp.L / 1e5:.2f} w_lg={bc.w_lg}')

    # v1 from its default guess (as the generators call it), then from cellwise guesses at
    # other p0 values, to reach the other root
    guesses = [None] + list(np.linspace(bc.p_s + 0.3 * (bc.p_r - bc.p_s), bc.p_r - 0.01, 4))
    for p0_guess in guesses:
        tag = 'v1 default guess ' if p0_guess is None else f'v1 guess p0={p0_guess:6.2f}'
        try:
            with contextlib.redirect_stdout(io.StringIO()):  # v1 prints the solver status on failure
                sim.x_guess = None if p0_guess is None else sim._simulate_cellwise(p0_guess, bc.T_r)
                xs = sim.simulate()
        except SimError as e:
            print(f'{tag}: failed ({e})')
            continue
        describe(sim, xs, tag)


if __name__ == '__main__':
    main()
