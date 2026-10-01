"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests for the ported sampler (manywells.sampling, specs/sampling.md) and the dataset rows (manywells.datasets).
"""

import dataclasses

import numpy as np
import pytest

from manywells.configurations import DEVELOP, V1, differences
from manywells.datasets.rows import sample_row
from manywells.datasets.schema import FEATURES
from manywells.sampling.conditions import NonStationaryBehavior, nominal_conditions, sample_conditions
from manywells.sampling.generate import Settings, solve, stationary_well
from manywells.sampling.wells import (INCH, OUTER_DIAMETERS, liquid, rng_for, sample_well, well_from_config,
                                      well_properties)

SEED = 11
DRAWS = [sample_well(SEED, i) for i in range(3000)]


def v1_part(draw):
    return dataclasses.replace(draw, trajectory=('vertical',), roughness=4.5e-5)


def test_seeding_is_reproducible():
    """SMP-31: a rerun gives the same draws, whatever the order; another seed gives others."""
    assert sample_well(SEED, 17) == DRAWS[17]
    assert sample_well(SEED, 5) == DRAWS[5]
    assert sample_well(SEED + 1, 17) != DRAWS[17]
    a, b = rng_for(SEED, 3, 4, 'sample'), rng_for(SEED, 3, 4, 'sample')
    assert a.uniform() == b.uniform()
    assert rng_for(SEED, 3, 4, 'sample').uniform() != rng_for(SEED, 4, 3, 'sample').uniform()


def test_develop_draws_do_not_shift_the_v1_draws():
    """The develop draws come from their own generator, so both configurations share the v1 draws (SMP-40)."""
    from manywells.sampling import wells
    rng = rng_for(SEED, 17, 'well')
    draw = None
    while draw is None:
        draw = wells._v1_draws(rng)
    assert v1_part(draw) == v1_part(DRAWS[17])


def test_v1_draw_ranges():
    """SMP-1 to SMP-17: every draw in its range, and the derived values by their formulas."""
    diameters = {INCH * (od - 0.5) for od in OUTER_DIAMETERS}
    for d in DRAWS:
        A = np.pi * (d.D / 2) ** 2
        f_g, f_o, f_w = d.fractions
        assert 1500 <= d.L <= 4500 and d.D in diameters and 0.01 <= d.f_D <= 0.08 and 10 <= d.h <= 40
        assert 0.06 * A <= d.K_c <= 0.24 * A and d.chk_profile in ('linear', 'sigmoid', 'convex', 'concave')
        assert f_g <= 0.99 and abs(f_g + f_o + f_w - 1) < 1e-12
        assert 20 * (1 - f_g) <= d.w_l_max <= 200 * (1 - f_g)
        assert 825 <= d.rho_o <= 925 and 320 <= d.R_s <= 520 and 10 <= d.p_s <= 120
        assert d.p_r == pytest.approx(1012.05 * 9.80665 * d.L / 1e5 + 1, rel=1e-12)
        assert d.T_r == pytest.approx(333.15 + 0.03 * (d.L - 1500), rel=1e-12)
        assert not d.has_gas_lift or f_g <= 0.2


def test_v1_draw_distributions():
    """The marginals of the v1 draws against their distributions (SMP-1, SMP-7, SMP-16, SMP-17)."""
    L = np.array([d.L for d in DRAWS])
    assert abs(L.mean() - 3000) < 50 and abs(L.std() - 3000 / np.sqrt(12)) < 40
    f_g = np.array([d.fractions[0] for d in DRAWS])
    assert abs(f_g.mean() - 1 / 2.5) < 0.02  # Dir(1, 1, 1/2): E f_g = 1 / 2.5, before the f_g <= 0.99 cut
    low = f_g <= 0.2
    lift = np.array([d.has_gas_lift for d in DRAWS])
    assert abs(lift[low].mean() - 0.5) < 0.05 and not lift[~low].any()
    p_s = np.log([d.p_s for d in DRAWS])
    assert np.log(10) <= p_s.min() and p_s.max() <= np.log(120)


def test_develop_draw_distributions():
    """SMP-41, SMP-42: the trajectory mix and the roughness range (kept until the sampling redesign)."""
    kinds = np.array([d.trajectory[0] for d in DRAWS])
    assert abs((kinds == 'vertical').mean() - 0.5) < 0.04
    assert abs((kinds == 'deviated').mean() - 0.25) < 0.04 and abs((kinds == 'l_shaped').mean() - 0.25) < 0.04
    eps = np.log([d.roughness for d in DRAWS])
    assert np.log(1.5e-6) <= eps.min() and eps.max() <= np.log(1.5e-4)
    for d in DRAWS[:200]:
        geo = well_properties(d, configuration=DEVELOP).geometry
        assert geo.tvd[0] == pytest.approx(d.L)  # the bottomhole's true vertical depth is L
        if d.trajectory[0] == 'deviated':
            assert min(geo.cos_incl) == pytest.approx(np.cos(np.radians(d.trajectory[2])), rel=1e-9)


def test_v1_mapping_reproduces_v1_inputs():
    """SMP-40: the mapped well is in the v1.0.0 configuration, with v1.0.0's liquid, gas fraction and gas constant."""
    for d in DRAWS[:50]:
        wp = well_properties(d)
        mix = liquid(d)
        assert differences(wp, V1) == []
        assert wp.geometry.L == d.L and wp.geometry.D == d.D and wp.friction.f_D == d.f_D and wp.thermal.h == d.h
        assert wp.fluid.rho_l == mix.rho and wp.fluid.cp_l == mix.cp
        assert wp.fluid.f_g == pytest.approx(d.fractions[0], rel=1e-13)
        assert wp.fluid.R_s == pytest.approx(d.R_s, rel=1e-13)
        assert wp.inflow.w_l_max == d.w_l_max and wp.choke.K_c == d.K_c and wp.choke.chk_profile == d.chk_profile


def test_develop_mapping_shares_the_v1_inputs():
    """SMP-40, SMP-43: the develop well has the same liquid at standard conditions, gas fraction and gas constant."""
    for d in DRAWS[:50]:
        wp, mix = well_properties(d, configuration=DEVELOP), liquid(d)
        assert differences(wp, DEVELOP) == []
        assert wp.fluid.rho_l == pytest.approx(mix.rho, rel=1e-13) and wp.fluid.cp_l == pytest.approx(mix.cp, rel=1e-13)
        assert wp.fluid.f_g == pytest.approx(d.fractions[0], rel=1e-12)
        assert wp.fluid.R_s == pytest.approx(d.R_s, rel=1e-13)


def test_sample_conditions():
    """SMP-18 to SMP-22."""
    d = next(d for d in DRAWS if d.has_gas_lift)
    for k in range(200):
        bc, (f_g, f_o, f_w) = sample_conditions(d, rng_for(SEED, 0, k, 'sample'))
        assert 0.05 <= bc.u <= 1 and 0 <= bc.w_lg <= 5
        assert 0.9 * d.p_s <= bc.p_s <= 1.1 * d.p_s and 0.98 * d.p_r <= bc.p_r <= 1.02 * d.p_r
        assert 0.95 * d.fractions[0] <= f_g <= min(0.99, 1.05 * d.fractions[0])
        assert abs(f_g + f_o + f_w - 1) < 1e-12 and f_o >= 0 and f_w >= 0
    assert nominal_conditions(d).u == 0.5 and nominal_conditions(d).w_lg == 0.0


def test_nonstationary_behaviour():
    """SMP-23 to SMP-27: the walk keeps the fractions valid, and the reservoir pressure decays."""
    d = DRAWS[3]
    rng = rng_for(SEED, 3, 'nonstationary')
    b = NonStationaryBehavior.draw(d, rng)
    assert 10 <= b.lifetime <= 20 and 0.6 * d.p_r <= b.p_r_conv <= 0.8 * d.p_r
    p_r = []
    for week in range(0, 400, 4):
        bc = b.update(d, week, max(0, week - 4), rng)
        f_g, f_o, f_w = b.fractions
        assert 0.002 <= f_g <= 0.99 and 0.002 <= f_o <= 0.99 and f_g + f_o <= 0.999 and abs(f_g + f_o + f_w - 1) < 1e-12
        assert 0.1 <= b.decay_rate <= 0.9
        p_r.append(bc.p_r)
    assert p_r[-1] < p_r[0] and min(p_r) >= b.p_r_conv


def test_well_from_config():
    row = {'ID': 0, 'wp.L': 2668.0, 'wp.D': 0.0762, 'wp.f_D': 0.018, 'wp.h': 25.8, 'wp.inflow.class_name': 'Vogel',
           'wp.inflow.w_l_max': 33.2, 'wp.choke.class_name': 'SimpsonChokeModel', 'wp.choke.K_c': 0.00096,
           'wp.choke.chk_profile': 'convex', 'bc.p_r': 265.8, 'bc.p_s': 23.7, 'bc.T_r': 368.2, 'bc.T_s': 277.15,
           'gas.R_s': 472.2, 'gas.cp': 2225, 'oil.rho': 885.6, 'oil.cp': 2000, 'water.rho': 999.1, 'water.cp': 4184,
           'fraction.gas': 0.237, 'fraction.oil': 0.7345, 'fraction.water': 0.0285, 'has_gas_lift': False}
    d = well_from_config(row)
    assert d.L == 2668.0 and d.fractions == (0.237, 0.7345, 0.0285) and differences(well_properties(d), V1) == []


@pytest.mark.slow
def test_sample_row_in_the_v1_configuration():
    """SMP-30: in the v1.0.0 configuration the rows are v1.0.0's wellhead rates, to rounding."""
    d = DRAWS[0]
    bc, fractions = sample_conditions(d, rng_for(SEED, 0, 0, 'sample'))
    op = solve(d, fractions, bc, Settings(seed=SEED))
    assert op is not None
    row = sample_row(op, d, fractions, bc)
    assert list(row) == list(FEATURES)
    p, v_g, v_l, alpha, rho_g, rho_l, T = op.state[-1]
    A = np.pi * (d.D / 2) ** 2
    w_g, w_l = A * alpha * rho_g * v_g, A * (1 - alpha) * rho_l * v_l
    assert row['WLIQ'] == pytest.approx(w_l, rel=1e-6) and row['WGAS'] + row['WGL'] == pytest.approx(w_g, rel=1e-6)
    assert row['QLIQ'] == pytest.approx(3600 * w_l / liquid(d, fractions).rho, rel=1e-6)
    assert row['TBH'] == pytest.approx(d.T_r) and row['PDC'] == bc.p_s


@pytest.mark.slow
def test_stationary_generation():
    """SMP-28 on one well: samples at the stable root, or the well is discarded with a reason."""
    from manywells.sampling.generate import WellDiscarded
    for well in range(5):
        try:
            df = stationary_well(DRAWS[well], well, 8, Settings(seed=SEED))
        except WellDiscarded:
            continue
        assert len(df) == 8 and list(df.columns) == list(FEATURES)
        assert (df['PWH'] > df['PDC']).all() and (df['PBH'] > df['PWH']).all()
        return
    pytest.fail('every well was discarded')
