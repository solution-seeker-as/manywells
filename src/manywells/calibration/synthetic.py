"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Synthetic calibration data (specs/calibration.md, CAL-13): a well's rows simulated at known parameters, with seeded
measurement noise and the observations an instrumentation has. The recovery tests, the twin study and the example
calibrate wells made here, so that the parameters they should find are known.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from manywells.calibration.data import PHASE_RATES, PRESSURES_AND_TEMPERATURES, CalibrationData, Noise
from manywells.calibration.parameters import apply
from manywells.datasets.rows import root_features
from manywells.simulator import SimError, SSDFSimulator

# spec: CAL-13. The instrumentations of the recovery tests and the twin study
INSTRUMENTATIONS = ('full', 'no_downhole', 'periodic_tests', 'pressures_only', 'random_missing')
TEST_EVERY = 5          # periodic_tests: rates on one row in five
RATED_ROWS = 2          # pressures_only: rates on two rows
MISSING_SHARE = 0.2     # random_missing: each observation missing with this probability


def operating_points(bc, n, rng, u=(0.2, 1.0), w_lg=None, p_s_spread=0.1) -> list:  # spec: CAL-13
    """
    n operating points around bc, spread over the choke position (evenly over u, in random order), the lift-gas rate
    (uniform over w_lg, if given) and the downstream pressure (uniform within p_s_spread of bc.p_s).
    """
    us = rng.permutation(np.linspace(u[0], u[1], n))
    out = []
    for k in range(n):
        p_s = bc.p_s * rng.uniform(1 - p_s_spread, 1 + p_s_spread)
        lg = rng.uniform(*w_lg) if w_lg is not None else bc.w_lg
        out.append(replace(bc, u=float(us[k]), p_s=float(p_s), w_lg=float(lg)))
    return out


def simulate_rows(wp, conditions, backend='rust', workers=None) -> pd.DataFrame:  # spec: CAL-13
    """
    The true observations of the well at each operating point: the inputs (CHK, PDC, WGL, p_r, T_r, T_s) and PBH,
    PWH, TWH and the phase rates QOIL, QGAS, QWAT (Sm³/h). An operating point without an operating point of the well
    is left out.
    """
    fluid = wp.fluid

    def solve(bc):
        try:
            root = SSDFSimulator(wp, backend=backend).simulate(bc)
        except (SimError, ValueError):
            return None
        f = root_features(root, bc, fluid.rho_o, fluid.rho_w, fluid.rho_g, 1 - fluid.f_o_in_liquid)
        return {'CHK': bc.u, 'PDC': bc.p_s, 'WGL': bc.w_lg, 'p_r': bc.p_r, 'T_r': bc.T_r, 'T_s': bc.T_s,
                **{c: f[c] for c in PRESSURES_AND_TEMPERATURES + PHASE_RATES}}

    n = 1 if backend == 'casadi' else (workers or min(32, os.cpu_count() or 1))
    with ThreadPoolExecutor(n) as ex:
        rows = list(ex.map(solve, conditions))
    return pd.DataFrame([r for r in rows if r is not None])


def add_noise(rows: pd.DataFrame, noise: Noise, rng, rate_source='test') -> pd.DataFrame:  # spec: CAL-13
    """
    The rows with measurement noise at the standard deviations of noise: Gaussian on PBH, PWH and TWH, and one
    log-normal factor per row on the three phase rates together, so that the fractions stay the well's.
    """
    df = rows.copy()
    for col in PRESSURES_AND_TEMPERATURES:
        df[col] = df[col] + rng.normal(0.0, getattr(noise, col), len(df))
    sd = noise.rate_mpfm if rate_source == 'mpfm' else noise.rate_test
    factor = np.exp(rng.normal(0.0, sd, len(df)))
    for col in PHASE_RATES:
        df[col] = df[col] * factor
    df['RATE_SOURCE'] = rate_source
    return df


def instrument(rows: pd.DataFrame, instrumentation: str, rng) -> pd.DataFrame:  # spec: CAL-13
    """
    The rows with only the observations an instrumentation has (the others NaN):

        full             PBH, PWH, TWH and the rates on every row
        no_downhole      no PBH
        periodic_tests   PBH, PWH and TWH on every row, the rates on one row in five
        pressures_only   PBH, PWH and TWH, the rates on two rows
        random_missing   each observation missing with probability 0.2, the phase rates together
    """
    if instrumentation not in INSTRUMENTATIONS:
        raise ValueError(f'unknown instrumentation {instrumentation!r}; one of {INSTRUMENTATIONS}')
    df = rows.copy()
    rates = list(PHASE_RATES)
    n = len(df)
    if instrumentation == 'no_downhole':
        df['PBH'] = np.nan
    elif instrumentation == 'periodic_tests':
        df.loc[np.arange(n) % TEST_EVERY != 0, rates] = np.nan
    elif instrumentation == 'pressures_only':
        keep = rng.choice(n, size=min(RATED_ROWS, n), replace=False)
        df.loc[~np.isin(np.arange(n), keep), rates] = np.nan
    elif instrumentation == 'random_missing':
        for cols in [['PBH'], ['PWH'], ['TWH'], rates]:
            drop = rng.random(n) < MISSING_SHARE
            df.loc[drop, cols] = np.nan
        empty = df[list(PRESSURES_AND_TEMPERATURES) + rates].isna().all(axis=1)
        df.loc[empty, 'PWH'] = rows.loc[empty, 'PWH']   # Every row keeps at least one observation
    return df


def synthetic_data(wp, bc, values: dict, n_rows: int, seed: int, instrumentation='full', noise=None,
                   noisy=True, rate_source='test', backend='rust', workers=None, **spread) -> CalibrationData:
    """
    Calibration data for a well whose parameters are `values` ({name: value}): n_rows operating points around bc
    (operating_points, with `spread`), solved, with noise (unless noisy is False) and instrumented. Every draw comes
    from numpy.random.default_rng(seed).
    """
    noise = noise or Noise()
    rng = np.random.default_rng(seed)
    truth = apply(wp, values)
    rows = simulate_rows(truth, operating_points(bc, n_rows, rng, **spread), backend=backend, workers=workers)
    if len(rows) == 0:
        raise ValueError('the well has no operating point at any of the operating points drawn')
    if noisy:
        rows = add_noise(rows, noise, rng, rate_source=rate_source)
    return CalibrationData(instrument(rows, instrumentation, rng), noise=noise)
