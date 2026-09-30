"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Frozen residual r(x; P, N) and its Jacobian, stacked from per-point blocks.

This is the only verifier module that imports CasADi. The blocks are v1.0.0's own per-point
equations, extracted by verification/build/build_graph.py and saved under residual_graph/.
They are stacked here in v1.0.0's row order, for any grid size N:

    point 0:          left boundary (3 rows), closure (3)
    point 0 < i < N:  cell (4), closure (3)
    point N:          cell (4), choke (1), closure (3)

The state x holds [p, v_g, v_l, alpha, rho_g, rho_l, T] at each of the N + 1 grid points,
so x and r both have length 7(N + 1), and the choke row is row 7N + 3.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np
import scipy.sparse as sp

DIM_X = 7
STATE = ('p', 'v_g', 'v_l', 'alpha', 'rho_g', 'rho_l', 'T')
GRAPH_ROOT = Path(__file__).resolve().parents[2] / 'residual_graph'

INFLOWS = ('vogel', 'pi')
CHOKES = ('simpson', 'bernoulli')
PROFILES = ('linear', 'sigmoid', 'convex', 'concave')


@dataclass(frozen=True)
class Variant:
    """The discrete model choices of a well. Each choice has its own block."""
    inflow: str = 'vogel'
    choke: str = 'simpson'
    profile: str = 'linear'

    def __post_init__(self):
        for value, allowed in ((self.inflow, INFLOWS), (self.choke, CHOKES), (self.profile, PROFILES)):
            if value not in allowed:
                raise ValueError(f'{value!r} is not one of {allowed}')


def choke_row(n_cells: int) -> int:
    """Index of the choke row in r."""
    return DIM_X * n_cells + 3


def row_layout(n_cells: int):
    """Row indices of each block in r: left (3,), closure (N+1, 3), cell (N, 4), choke (scalar)."""
    N = n_cells
    first = 6 + DIM_X * np.arange(N)            # first row of grid point i = 1..N
    closure_first = np.concatenate(([3], first + 4))
    closure_first[-1] += 1                      # at point N the choke row comes before the closure rows
    return (np.arange(3),
            closure_first[:, None] + np.arange(3),
            first[:, None] + np.arange(4),
            choke_row(N))


def _coo(rows, col0, blocks):
    """Triplets for dense blocks (m, k, 7) placed at rows (m, k) and state columns col0 + 0..6."""
    shape = blocks.shape
    r = np.broadcast_to(rows[:, :, None], shape)
    c = np.broadcast_to(np.asarray(col0)[:, None, None] + np.arange(DIM_X), shape)
    return r.ravel(), c.ravel(), blocks.ravel()


class ResidualGraph:
    """A frozen residual graph: the blocks of one model version, read from residual_graph/<version>/."""

    def __init__(self, version: str = 'v1.0.0', root: Path = GRAPH_ROOT):
        path = Path(root) / version
        self.manifest = json.loads((path / 'manifest.json').read_text())
        self.params = tuple(self.manifest['params'])
        self._blocks = {name: ca.Function.load(str(path / f'{name}.casadi')) for name in self.manifest['blocks']}
        self._maps = {}

    def param_vector(self, values) -> np.ndarray:
        """Parameter vector P in manifest order from a mapping of parameter names to values."""
        missing = [k for k in self.params if k not in values]
        if missing:
            raise KeyError(f'missing parameters: {missing}')
        return np.array([float(values[k]) for k in self.params])

    def row_names(self, n_cells: int) -> np.ndarray:
        """Name of each row of r, e.g. 'momentum' or 'choke', from the manifest."""
        blocks = self.manifest['blocks']
        rows_left, rows_clo, rows_cell, row_chk = row_layout(n_cells)
        names = np.empty(DIM_X * (n_cells + 1), dtype=object)
        names[rows_left] = blocks['left_vogel']['row_names']
        names[rows_clo] = blocks['closure']['row_names']
        names[rows_cell] = blocks['cell']['row_names']
        names[row_chk] = blocks['choke_simpson_linear']['row_names'][0]
        return names

    def _mapped(self, name: str, n: int) -> ca.Function:
        if (name, n) not in self._maps:
            self._maps[name, n] = self._blocks[name].map(n)
        return self._maps[name, n]

    def evaluate(self, x, P, n_cells: int, variant: Variant = Variant(), jacobian: bool = True):
        """
        Evaluate r(x; P, N) and, if jacobian is true, dr/dx as a sparse CSC matrix.

        :param x: State, length 7(N + 1), grid point by grid point
        :param P: Parameter vector in manifest order (see param_vector)
        :param n_cells: Number of cells N
        :param variant: Inflow, choke and choke-profile choice
        :return: r, or (r, J)
        """
        N = int(n_cells)
        n = DIM_X * (N + 1)
        x = np.asarray(x, dtype=float)
        if x.shape != (n,):
            raise ValueError(f'x has shape {x.shape}, expected ({n},) for N = {N}')
        X = x.reshape(N + 1, DIM_X).T            # column k is grid point k
        P = np.asarray(P, dtype=float).reshape(-1, 1)
        i = np.arange(1, N + 1, dtype=float)[None, :]

        g_left, J_left = self._blocks[f'left_{variant.inflow}'](X[:, 0], P)
        g_clo, J_clo = self._mapped('closure', N + 1)(X, P)
        g_cell, J_cell, J_cell_prev = self._mapped('cell', N)(X[:, 1:], X[:, :-1], i, N, P)
        g_chk, J_chk = self._blocks[f'choke_{variant.choke}_{variant.profile}'](X[:, N], P)

        rows_left, rows_clo, rows_cell, row_chk = row_layout(N)
        r = np.empty(n)
        r[rows_left] = np.asarray(g_left).ravel()
        r[rows_clo] = np.asarray(g_clo).T
        r[rows_cell] = np.asarray(g_cell).T
        r[row_chk] = float(g_chk)
        if not jacobian:
            return r

        def per_point(J, k):  # (k, 7m) horizontally stacked blocks -> (m, k, 7)
            return np.asarray(J).reshape(k, -1, DIM_X).transpose(1, 0, 2)

        points = np.arange(N + 1)
        triplets = [
            _coo(rows_left[None, :], [0], np.asarray(J_left)[None]),
            _coo(rows_clo, DIM_X * points, per_point(J_clo, 3)),
            _coo(rows_cell, DIM_X * points[1:], per_point(J_cell, 4)),
            _coo(rows_cell, DIM_X * points[:-1], per_point(J_cell_prev, 4)),
            _coo(np.array([[row_chk]]), [DIM_X * N], np.asarray(J_chk)[None]),
        ]
        rows, cols, vals = (np.concatenate(t) for t in zip(*triplets))
        return r, sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
