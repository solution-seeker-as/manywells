"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The distributions of the v2 draws (specs/sampling.md, v2: scope and conventions). Each takes the draw's generator and
uses one uniform number of it, so a truncated draw never loops, and unused takes the numbers that a well's branch
leaves unused (SMP-51).
"""

import math

import numpy as np
from scipy.special import ndtr, ndtri


def tnorm(rng, mu, sd, lo, hi) -> float:
    """Normal(mu, sd) truncated to [lo, hi], by the inverse CDF of one uniform draw."""
    a, b = ndtr((lo - mu) / sd), ndtr((hi - mu) / sd)
    u = min(max(rng.uniform(a, b), 1e-15), 1 - 1e-15)
    return float(min(max(mu + sd * ndtri(u), lo), hi))


def tnorm_at(v, mu, sd, lo, hi) -> float:
    """tnorm's value at the uniform number v of its draw: tnorm(rng, mu, sd, lo, hi) is tnorm_at(rng.random(), mu,
    sd, lo, hi) to the bit, so that one number gives the draw at several medians."""
    a, b = ndtr((lo - mu) / sd), ndtr((hi - mu) / sd)
    u = min(max(a + (b - a) * v, 1e-15), 1 - 1e-15)
    return float(min(max(mu + sd * ndtri(u), lo), hi))


def tlognorm(rng, median, s, lo, hi) -> float:
    """Log-normal with the median and the log standard deviation s, truncated to [lo, hi]."""
    return math.exp(tnorm(rng, math.log(median), s, math.log(lo), math.log(hi)))


def logu(rng, lo, hi) -> float:
    """Log-uniform over [lo, hi]."""
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def pick(rng, names, p) -> str:
    """One of names, with the probabilities p (normalised here)."""
    return str(names[rng.choice(len(names), p=np.asarray(p) / np.sum(p))])


def unused(rng, n=1) -> None:  # spec: SMP-51
    """Take n uniform numbers that the well's branch does not use, so that a block takes the same numbers in the
    same order for every well."""
    for _ in range(n):
        rng.random()


def logistic(x) -> float:
    return 1 / (1 + math.exp(-x))
