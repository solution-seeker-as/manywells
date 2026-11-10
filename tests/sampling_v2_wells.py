"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Drawn wells for the tests of the v2 sampler (manywells.sampling.v2, specs/sampling.md, SMP-50 on): the wells and
their lives are drawn once per session, and none is solved here.

This module is a helper, not a test module: test_sampling_v2_draws.py, test_sampling_v2_mapping.py,
test_sampling_v2_life.py and test_sampling_v2_samples.py import it.
"""

import functools

import numpy as np

from manywells.sampling.v2.draws import black_oil, draw_well
from manywells.sampling.v2.inflow import inflow_at
from manywells.sampling.v2.life import Life, sol2_well
from manywells.sampling.v2.reservoir import OILS
from manywells.sampling.wells import rng_for

SEED = 11
N_WELLS = 300  # About 5 s of draws: enough for the bounds, and for the shares within wide sampling errors
N_MANY = 3000  # The slow tests' wells, for the shares and marginals within tighter ones
N_LIVES = 100  # A life takes its well at three stages: enough for every drive, class and lift rule
N_MANY_LIVES = N_WELLS  # The slow tests' lives


@functools.cache
def drawn_wells(n=N_WELLS) -> tuple:
    """The first n wells of the tests' seed, each as draw_well gives it: (draws, (md, tvd))."""
    return tuple(draw_well(SEED, k) for k in range(n))


@functools.cache
def drawn_lives(n=N_LIVES) -> tuple:
    """The lives of the first n wells of the tests' seed (life.Life)."""
    return tuple(Life(SEED, k) for k in range(n))


@functools.cache
def drawn_sol2_wells(n=N_LIVES) -> tuple:
    """The sol-2 wells of drawn_lives(n): each life at its date (life.sol2_well)."""
    return tuple(sol2_well(lf) for lf in drawn_lives(n))


def is_oil(w) -> bool:
    return w['fluid_type'] in OILS


class Counting:
    """A stand-in for a generator that records how many of its numbers each call takes (SMP-51). It has the methods
    that take a fixed count, one number per value, and no other: a draw by another method fails."""

    def __init__(self, rng, log):
        self._rng, self._log = rng, log

    def random(self, size=None):
        self._log.append(1 if size is None else int(np.prod(size)))
        return self._rng.random(size)

    def uniform(self, low=0.0, high=1.0, size=None):
        self._log.append(1 if size is None else int(np.prod(size)))
        return self._rng.uniform(low, high, size)

    def choice(self, a, p):
        self._log.append(1)  # One of a, with probabilities: one number
        return self._rng.choice(a, p=p)


def counting(monkeypatch, module) -> list:
    """Replace a module's rng_for by one whose generators are Counting ones. Returns the list that fills, in the
    order of the generators made, with (the generator's key after the seed, its calls' counts of numbers)."""
    made = []

    def counted(seed, *parts):
        made.append((parts, []))
        return Counting(rng_for(seed, *parts), made[-1][1])

    monkeypatch.setattr(module, 'rng_for', counted)
    return made


def nominal_inputs(w) -> dict:
    """An attempt's inputs (mapping.WellBuilder.case) at a drawn well's nominal state: its reservoir state and
    nominal downstream pressure, with the choke fully open and no lift gas."""
    coef = inflow_at(w, black_oil(w), w['p_r'], w['R_go'], w['wlr'])
    return dict(
        coef,
        p_r=w['p_r'],
        p_s=w['p_s'],
        u=1.0,
        w_lg=0.0,
        lift_on=w['lift'] == 'gas_lift',
        gor=w['R_go'],
        wlr=w['wlr'],
        h=w['h'],
        roughness=w['roughness'],
    )
