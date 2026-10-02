"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The objective of a calibration (specs/calibration.md, CAL-8 to CAL-10): each row's operating point under the
parameters, the observations predicted from it, and the residuals of the MAP estimate, scaled by the noise, with the
priors' residuals after them.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from manywells.calibration.data import OBSERVATIONS
from manywells.calibration.parameters import apply
from manywells.datasets.rows import root_features
from manywells.simulator import BoundaryConditions, SimError, SSDFSimulator

BARRIER = 1e3      # The scaled residual of every observation of a row with no operating point (CAL-9)
FD_STEP = 1e-6     # Forward-difference step of the Jacobian in z, the standardized log (CAL-10)
CACHE_SIZE = 64    # Evaluations kept, so that the fit reuses the start's and the residuals' solves at x


def row_conditions(data, bc: BoundaryConditions) -> list:  # spec: CAL-1
    """The boundary conditions of each row: bc with the row's inputs."""
    df = data.rows
    names = {'CHK': 'u', 'PDC': 'p_s', 'WGL': 'w_lg', 'p_r': 'p_r', 'T_r': 'T_r', 'T_s': 'T_s', 'T_lg': 'T_lg'}
    cols = [c for c in names if c in df]
    return [replace(bc, **{names[c]: float(row[c]) for c in cols}) for _, row in df.iterrows()]


def row_fluids(data, fluid) -> list:  # spec: CAL-1
    """The fluid of each row: the well's, with the row's gas-oil and water-liquid ratios."""
    df = data.rows
    cols = [c for c in ('gor', 'wlr') if c in df]
    if not cols:
        return [fluid] * len(df)
    cache, out = {}, []
    for _, row in df.iterrows():
        key = tuple(float(row[c]) for c in cols)
        if key not in cache:
            cache[key] = replace(fluid, **dict(zip(cols, key)))
        out.append(cache[key])
    return out


def failed_rows(pred) -> np.ndarray:
    """The rows of a prediction with no operating point: their predictions are all NaN."""
    return np.isnan(pred).all(axis=1)


def predicted(root, bc, fluid) -> np.ndarray:  # spec: CAL-8
    """The predicted observations of a root, in the order OBSERVATIONS: PBH, PWH, TWH (bar, K) and the reservoir
    mass rate WLIQ + WGAS (kg/s), with the definitions of the dataset features."""
    f = root_features(root, bc, fluid.rho_o, fluid.rho_w, fluid.rho_g, 1 - fluid.f_o_in_liquid)
    return np.array([f['PBH'], f['PWH'], f['TWH'], f['WLIQ'] + f['WGAS']])


class Objective:
    """
    The residuals of one well's calibration as a function of z, the free parameters' standardized logs.

    Each row is solved for its operating point (simulate) on its own well: the parameters applied, with the row's
    fluid. The rows of all the z an evaluation asks for are solved together in threads; the Rust core releases the
    GIL, so they run in parallel. The CasADi backend builds a system per well, and its rows run one by one.
    """

    def __init__(self, wp, data, bc, parameters, backend='rust', workers=None):
        self.wp, self.data, self.parameters, self.backend = wp, data, tuple(parameters), backend
        self.conditions = row_conditions(data, bc)
        self.fluids = row_fluids(data, wp.fluid)
        self.y = data.observations(self.fluids)
        self.sd = data.noise_sd()
        self.mask = ~np.isnan(self.y)
        self.workers = 1 if backend == 'casadi' else (workers or min(32, os.cpu_count() or 1))
        self.n_solves = 0
        self._guess = {}   # The last root of each row, as an extra start for the CasADi backend
        self._cache = {}

    @property
    def n_rows(self):
        return len(self.conditions)

    def values(self, z) -> dict:
        """The parameter values at z."""
        return {p.name: p.value(zi) for p, zi in zip(self.parameters, np.atleast_1d(z))}

    def wells(self, z) -> list:
        """The well of each row at z: the parameters applied, with the row's fluid."""
        base = apply(self.wp, self.values(z))
        cache = {}
        return [cache.setdefault(id(f), replace(base, fluid=f)) for f in self.fluids]

    def _solve(self, task):
        k, wp, bc = task
        try:
            root = SSDFSimulator(wp, backend=self.backend).simulate(bc, x_guess=self._guess.get(k))
        except (SimError, ValueError):
            return k, None, None
        return k, root, predicted(root, bc, wp.fluid)

    def predict(self, zs) -> list:
        """
        The predicted observations at each z in zs: for each, an (n, 4) array in the order OBSERVATIONS, with a row
        of NaN where the row has no operating point; and the roots.
        """
        tasks, out = [], []
        for z in zs:
            key = np.asarray(z, dtype=float).tobytes()
            if key in self._cache:
                out.append(self._cache[key])
                continue
            out.append(key)
            for k, (wp, bc) in enumerate(zip(self.wells(z), self.conditions)):
                tasks.append((len(out) - 1, k, wp, bc))
        if tasks:
            results = {}
            def run(t):
                i, k, wp, bc = t
                return i, self._solve((k, wp, bc))
            if self.workers > 1:
                with ThreadPoolExecutor(self.workers) as ex:
                    done = list(ex.map(run, tasks))
            else:
                done = [run(t) for t in tasks]
            self.n_solves += len(tasks)
            for i, (k, root, pred) in done:
                results.setdefault(i, {})[k] = (root, pred)
            for i, by_row in results.items():
                pred = np.full((self.n_rows, len(OBSERVATIONS)), np.nan)
                roots = [None] * self.n_rows
                for k, (root, p) in by_row.items():
                    if root is not None:
                        pred[k], roots[k] = p, root
                        if self.backend == 'casadi':
                            self._guess[k] = root.x
                self._cache[out[i]] = (pred, roots)
                out[i] = (pred, roots)
            while len(self._cache) > CACHE_SIZE:  # The oldest go first; the fit asks for residuals, then jacobian, at x
                self._cache.pop(next(iter(self._cache)))
        return out

    def scaled(self, pred) -> np.ndarray:  # spec: CAL-9, CAL-10
        """
        The observations' scaled residuals: (y - y_pred) / sd for pressures and temperature, (log y - log y_pred) / sd
        for the rate; BARRIER for every observation of a row with no operating point. Missing observations are left
        out, row by row in the order OBSERVATIONS.
        """
        r = np.empty_like(self.y)
        r[:, :3] = (self.y[:, :3] - pred[:, :3]) / self.sd[:, :3]
        with np.errstate(invalid='ignore', divide='ignore'):
            r[:, 3] = (np.log(self.y[:, 3]) - np.log(pred[:, 3])) / self.sd[:, 3]
        r[failed_rows(pred)] = BARRIER
        return r[self.mask]

    def residuals(self, z) -> np.ndarray:  # spec: CAL-10
        """The residual vector at z: the observations' scaled residuals, then the priors' (z itself)."""
        (pred, _), = self.predict([z])
        return np.concatenate([self.scaled(pred), np.asarray(z, dtype=float)])

    def jacobian(self, z) -> np.ndarray:  # spec: CAL-10
        """The Jacobian of residuals at z, by forward differences with step FD_STEP in z, all columns solved together."""
        z = np.asarray(z, dtype=float)
        zs = [z] + [z + FD_STEP * e for e in np.eye(len(z))]
        evaluations = self.predict(zs)
        r0 = self.scaled(evaluations[0][0])
        J = np.empty((len(r0) + len(z), len(z)))
        for j, (pred, _) in enumerate(evaluations[1:]):
            J[:len(r0), j] = (self.scaled(pred) - r0) / FD_STEP
        J[len(r0):] = np.eye(len(z))
        return J
