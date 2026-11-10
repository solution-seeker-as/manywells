"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Drawn wells for the tests of the v2 sampler (manywells.sampling.v2, specs/sampling.md, SMP-50 on): the wells are drawn
once per session, and none is solved here.

This module is a helper, not a test module: test_sampling_v2_draws.py and test_sampling_v2_mapping.py import it.
"""

import functools

from manywells.sampling.v2.draws import black_oil, draw_well
from manywells.sampling.v2.inflow import inflow_at
from manywells.sampling.v2.reservoir import OILS

SEED = 11
N_WELLS = 300  # About 5 s of draws: enough for the bounds, and for the shares within wide sampling errors
N_MANY = 3000  # The slow tests' wells, for the shares and marginals within tighter ones


@functools.cache
def drawn_wells(n=N_WELLS) -> tuple:
    """The first n wells of the tests' seed, each as draw_well gives it: (draws, (md, tvd))."""
    return tuple(draw_well(SEED, k) for k in range(n))


def is_oil(w) -> bool:
    return w['fluid_type'] in OILS


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
