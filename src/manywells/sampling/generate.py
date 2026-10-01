"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Dataset generation (specs/sampling.md): the solves, starts and acceptance rules of v1.0.0's generators, for the
stationary samples of sol-1 (SMP-28) and the weekly samples of nsol-1 (SMP-29), on develop's simulator.

The difference from v1.0.0 is the simulator's answer: every sample is the well's operating point, the stable root
(specs/model/solution.md), not whichever root Ipopt reached from the generator's start; the start is passed as an
extra start of the search (x_guess). A sample without an operating point is a failed solve.

Each sample has its own fractions (SMP-22, SMP-24), and with them its own fluid, so its well's system is built for
the sample.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from manywells.configurations import V1
from manywells.datasets.rows import sample_row
from manywells.datasets.schema import FEATURES
from manywells.sampling.conditions import NonStationaryBehavior, nominal_conditions, sample_conditions
from manywells.sampling.wells import N_CELLS, rng_for, well_properties
from manywells.simulator import SimError, SSDFSimulator

CHK_STD_MIN = np.sqrt((1 / 12) * (1 - 0.05) ** 2) / 5   # A fifth of the standard deviation of U(0.05, 1)


class WellDiscarded(Exception):
    """The generator discards the well (SMP-28, SMP-29)."""


@dataclass(frozen=True)
class Settings:
    """What a generator run needs besides the wells: the configuration, the grid and the dataset's seed (SMP-31)."""
    seed: int
    configuration: str = V1
    n_cells: int = N_CELLS


def solve(draw, fractions, bc, settings: Settings, x_guess=None):
    """The operating point of a draw's well at fractions and bc, or None if there is none (or the well is invalid)."""
    try:
        sim = SSDFSimulator(well_properties(draw, fractions, settings.configuration, settings.n_cells))
        return sim.simulate(bc, x_guess=x_guess)
    except (SimError, ValueError):
        return None


def _valid(row) -> bool:
    """Non-negative rates and fractions in [0, 1]; a failure otherwise (SMP-28, SMP-29)."""
    rates = row['WLIQ'] >= 0 and row['WGAS'] + row['WGL'] >= 0
    return rates and all(0 <= row[k] <= 1 for k in ('FGAS', 'FOIL', 'FWAT'))


def stationary_well(draw, well: int, n_samples: int, settings: Settings) -> pd.DataFrame:  # spec: SMP-28
    """
    The sol-1 samples of one well, or WellDiscarded.

    1. Solve at u = 0.5 with the nominal conditions; the operating point is the start of every sample.
    2. Draw samples (SMP-18 to SMP-22) and solve each. A failed solve or an invalid row is a failure; a sample with
       w_m < 0.1 kg/s is dropped without counting. Discard after 5 n_samples attempts or 100 failures.
    3. Discard if the choke positions vary too little, more than 80% of the samples are choked, or the coefficient of
       variation of QTOT is below 0.05.
    """
    first = solve(draw, draw.fractions, nominal_conditions(draw), settings)
    if first is None:
        raise WellDiscarded('initial solve failed')
    rows, attempts, failures = [], 0, 0
    while len(rows) < n_samples:
        if attempts >= 5 * n_samples:
            raise WellDiscarded('maximum number of attempts')
        if failures >= 100:
            raise WellDiscarded('too many failures')
        bc, fractions = sample_conditions(draw, rng_for(settings.seed, well, attempts, 'sample'))
        attempts += 1
        op = solve(draw, fractions, bc, settings, x_guess=first.x)
        if op is None:
            failures += 1
            continue
        row = sample_row(op, draw, fractions, bc)
        if not _valid(row):
            failures += 1
            continue
        if row['WTOT'] < 0.1:
            continue
        rows.append(row)
    df = pd.DataFrame(rows, columns=list(FEATURES))
    if df['CHK'].std() < CHK_STD_MIN:
        raise WellDiscarded('low choke variation')
    if df['CHOKED'].sum() > 0.8 * len(df):
        raise WellDiscarded('choked flow in more than 80% of the samples')
    if df['QTOT'].std() / df['QTOT'].mean() < 0.05:
        raise WellDiscarded('small variation in QTOT')
    return df


class StartPool:
    """
    The starts of nsol-1's samples (v1.0.0's `InitGuess`): roots of the well with w_m >= 3 kg/s that lie more than
    0.05 apart in the coordinates (f_g, p_r / 350 bar). Each sample starts from the closest.
    """

    def __init__(self, min_rate=3.0, min_distance=0.05, p_scale=350.0):
        self.min_rate, self.min_distance, self.p_scale = min_rate, min_distance, p_scale
        self.starts = []

    def _coord(self, f_g, p_r):
        return np.array([f_g, p_r / self.p_scale])

    def closest(self, f_g, p_r):
        if not self.starts:
            return None
        c = self._coord(f_g, p_r)
        return min(self.starts, key=lambda s: np.linalg.norm(s[0] - c))[1]

    def add(self, x, w_m, f_g, p_r):
        c = self._coord(f_g, p_r)
        if w_m >= self.min_rate and all(np.linalg.norm(s[0] - c) > self.min_distance for s in self.starts):
            self.starts.append((c, np.asarray(x)))


def nonstationary_well(draw, well: int, n_samples: int, settings: Settings) -> pd.DataFrame:  # spec: SMP-29
    """
    The nsol-1 samples of one well, or WellDiscarded, as v1.0.0's open-loop non-stationary generator.

    1. Solve at u = 0.1, 0.2, ..., 1.0 in turn, each operating point the start of the next; discard the well if one
       fails, or if w_m < 7 kg/s at u = 1.
    2. Keep a pool of starts (StartPool), beginning with the operating point at u = 1.
    3. Each attempt updates the conditions (SMP-24 to SMP-27) and solves from the pool's closest start. A failed
       solve is a failure and advances the week; every solved root joins the pool; an invalid row is a failure and
       w_m < 1 kg/s is rejected, and neither advances the week; an accepted sample advances it. Discard after 200
       failures or 50 failed solves in a row.
    4. Run while there are fewer than n_samples samples or the week equals the lifetime in whole weeks.
    5. Discard if there are not exactly n_samples samples, or the choke positions or QTOT vary too little.
    """
    rng = rng_for(settings.seed, well, 'nonstationary')
    behaviour = NonStationaryBehavior.draw(draw, rng)
    x = None
    for k in range(1, 11):
        op = solve(draw, draw.fractions, nominal_conditions(draw, u=0.1 * k), settings, x_guess=x)
        if op is None:
            raise WellDiscarded('initial solve failed')
        x = op.x
    A = np.pi * (draw.D / 2) ** 2
    w_m = A * (op.state[-1, 3] * op.state[-1, 4] * op.state[-1, 1] + (1 - op.state[-1, 3]) * op.state[-1, 5] * op.state[-1, 2])
    if w_m < 7:
        raise WellDiscarded('too little production at u = 1')
    pool = StartPool()
    pool.add(op.x, w_m, draw.fractions[0], draw.p_r)

    rows, failures, in_a_row = [], 0, 0
    week, week_prev = 0, 0
    while len(rows) < n_samples or week == int(behaviour.lifetime * 52):
        if failures >= 200:
            raise WellDiscarded('too many failures')
        if in_a_row >= 50:
            raise WellDiscarded('too many failed solves in a row')
        bc = behaviour.update(draw, week, week_prev, rng)
        fractions = behaviour.fractions
        op = solve(draw, fractions, bc, settings, x_guess=pool.closest(fractions[0], bc.p_r))
        if op is None:
            failures += 1
            in_a_row += 1
            week += 1
            continue
        in_a_row = 0
        row = sample_row(op, draw, fractions, bc)
        pool.add(op.x, row['WTOT'], fractions[0], bc.p_r)
        if not _valid(row):
            failures += 1
            continue
        if row['WTOT'] < 1.0:
            continue
        row['WEEKS'] = week
        rows.append(row)
        week_prev = week
        week += 1
    df = pd.DataFrame(rows, columns=list(FEATURES) + ['WEEKS'])
    if len(df) != n_samples:
        raise WellDiscarded('not exactly n_samples samples')
    if df['CHK'].std() < CHK_STD_MIN:
        raise WellDiscarded('low choke variation')
    if df['QTOT'].std() / df['QTOT'].mean() < 0.05:
        raise WellDiscarded('small variation in QTOT')
    return df


GENERATORS = {'sol': stationary_well, 'nsol': nonstationary_well}


def _attempt(task):
    """Draw well number `well` and generate its samples; (well, draw, rows or None, reason)."""
    from manywells.sampling.wells import sample_well
    kind, well, n_samples, settings = task
    draw = sample_well(settings.seed, well)
    try:
        return well, draw, GENERATORS[kind](draw, well, n_samples, settings), ''
    except WellDiscarded as e:
        return well, draw, None, str(e)


def generate(kind: str, n_wells: int, n_samples: int, settings: Settings, processes: int = 1):
    """
    Generate a dataset: draw wells 0, 1, 2, ... and keep the first n_wells that the generator accepts, so that the
    result does not depend on the number of processes.

    :param kind: 'sol' (stationary, SMP-28) or 'nsol' (non-stationary, SMP-29)
    :return: (rows with an ID column, {ID: WellDraw}, the number of wells drawn)
    """
    import multiprocessing
    accepted, drawn, chunk = {}, 0, max(1, processes)
    with multiprocessing.Pool(processes) if processes > 1 else _Serial() as pool:
        while len(accepted) < n_wells:
            tasks = [(kind, w, n_samples, settings) for w in range(drawn, drawn + chunk)]
            drawn += chunk
            for well, draw, rows, _ in sorted(pool.map(_attempt, tasks), key=lambda r: r[0]):
                if rows is not None and len(accepted) < n_wells:
                    accepted[well] = (draw, rows)
    ids = {well: i for i, well in enumerate(sorted(accepted))}
    rows = pd.concat([accepted[w][1].assign(ID=ids[w]) for w in sorted(accepted)], ignore_index=True)
    return rows, {ids[w]: accepted[w][0] for w in sorted(accepted)}, drawn


class _Serial:
    """A pool of one process, without multiprocessing."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @staticmethod
    def map(fn, tasks):
        return [fn(t) for t in tasks]
