"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests of manywells.calibration that solve no well (specs/calibration.md): the data and its validation, the noise,
the parameters and their priors, the residuals, and the synthetic data's instrumentation. The recovery of known
parameters from simulated wells is in test_calibration_recovery.py.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from manywells.calibration import (CalibrationData, Noise, Parameter, apply, calibrate, darcy_productivity_index,
                                   default_prior, evaluate, vogel_maximum_rate)
from manywells.calibration.fit import priors_for
from manywells.calibration.objective import BARRIER, Objective, predicted, row_conditions, row_fluids
from manywells.calibration.parameters import MILLIDARCY, CENTIPOISE
from manywells.calibration.synthetic import INSTRUMENTATIONS, add_noise, instrument, operating_points
from manywells.datasets.rows import root_features
from manywells.friction import FixedFrictionFactor
from manywells.inflow import ProductivityIndex, Vogel
from manywells.simulator import BoundaryConditions, WellProperties
from manywells.solution import Root
from manywells.solvers.rust import _friction, _inflow


def rows(**extra):
    base = {'CHK': [0.5, 0.8], 'PDC': [20.0, 21.0], 'PBH': [150.0, 140.0], 'PWH': [40.0, 35.0],
            'TWH': [330.0, 332.0], 'QOIL': [30.0, 36.0], 'QGAS': [6000.0, 7200.0], 'QWAT': [3.0, 4.0]}
    return pd.DataFrame(base | extra)


WP = WellProperties(inflow=Vogel(w_l_max=50.0))


# --- Data (CAL-1 to CAL-4) ---------------------------------------------------------------------------------------

def test_data_keeps_a_copy():
    df = rows()
    data = CalibrationData(df)
    df.loc[0, 'PBH'] = 1.0
    assert data.rows.loc[0, 'PBH'] == 150.0


@pytest.mark.parametrize('change, match', [
    (lambda df: df.drop(columns='CHK'), 'input column CHK'),
    (lambda df: df.assign(CHK=[0.0, 0.5]), r'CHK must be in \(0, 1\]'),
    (lambda df: df.assign(PDC=[20.0, np.nan]), 'missing or non-numeric'),
    (lambda df: df.assign(p_r=[-1.0, 100.0]), 'p_r must be positive'),
    (lambda df: df.assign(WGL=[-0.1, 0.0]), 'WGL must be non-negative'),
    (lambda df: df.assign(wlr=[1.0, 0.1]), r'wlr must be in \[0, 1\)'),
    (lambda df: df.drop(columns='QWAT'), 'together'),
    (lambda df: df.assign(QWAT=[np.nan, 4.0]), 'some phase rates'),
    (lambda df: df.assign(QOIL=[-1.0, 36.0]), 'non-negative'),
    (lambda df: df.assign(PBH=[0.0, 140.0]), 'PBH must be positive'),
    (lambda df: df.assign(PBH=np.nan, PWH=np.nan, TWH=np.nan, QOIL=np.nan, QGAS=np.nan, QWAT=np.nan),
     'at least one observation'),
    (lambda df: df.assign(RATE_SOURCE=['test', 'meter']), 'RATE_SOURCE'),
    (lambda df: df.assign(PWH_SD=[0.0, 1.0]), 'PWH_SD must be positive'),
    (lambda df: df.assign(PWH=['40', 'x']), 'non-numeric'),
])
def test_data_rejects_malformed_rows(change, match):
    """CAL-1 to CAL-4: each malformed input is rejected with a ValueError that names it."""
    with pytest.raises(ValueError, match=match):
        CalibrationData(change(rows()))


def test_data_rejects_empty_table_and_bad_noise():
    with pytest.raises(ValueError, match='no rows'):
        CalibrationData(rows().iloc[:0])
    with pytest.raises(ValueError, match='noise'):
        CalibrationData(rows(), noise=0.3)
    with pytest.raises(ValueError, match='positive'):
        Noise(PBH=0.0)


def test_observed_marks_missing_values():
    """CAL-2: a NaN is a missing observation, and a row's rate exists when its three phase rates do."""
    df = rows(PBH=[np.nan, 140.0]).assign(QOIL=[np.nan, 36.0], QGAS=[np.nan, 7200.0], QWAT=[np.nan, 4.0])
    obs = CalibrationData(df).observed()
    assert obs.to_numpy().tolist() == [[False, True, True, False], [True, True, True, True]]
    only_wellhead = CalibrationData(pd.DataFrame({'CHK': [0.5], 'PDC': [20.0], 'PWH': [40.0]})).observed()
    assert only_wellhead.to_numpy().tolist() == [[False, True, False, False]]


def test_rate_observation_is_the_reservoir_mass_rate():
    """CAL-3: the phase rates, weighted by the densities at standard conditions, in kg/s."""
    data = CalibrationData(rows())
    f = WP.fluid
    y = data.observations([f, f])
    assert y[0, :3].tolist() == [150.0, 40.0, 330.0]
    assert y[0, 3] == pytest.approx((30.0 * f.rho_o + 6000.0 * f.rho_g + 3.0 * f.rho_w) / 3600)


def test_noise_defaults_and_overrides():
    """CAL-4: absolute for pressures and temperature, relative for the rate by its source; *_SD columns override."""
    noise = Noise()
    sd = CalibrationData(rows(RATE_SOURCE=['test', 'mpfm'], PWH_SD=[np.nan, 0.5], RATE_SD=[0.05, np.nan])).noise_sd()
    assert sd[0].tolist() == [noise.PBH, noise.PWH, noise.TWH, 0.05]
    assert sd[1].tolist() == [noise.PBH, 0.5, noise.TWH, noise.rate_mpfm]
    assert CalibrationData(rows()).noise_sd()[:, 3].tolist() == [noise.rate_test] * 2


def test_row_inputs():
    """CAL-1: each row's boundary conditions and fluid are the well's, with the row's inputs."""
    bc = BoundaryConditions(p_r=200.0, T_r=360.0, w_lg=1.0)
    data = CalibrationData(rows(WGL=[0.0, 2.0], p_r=[190.0, 185.0], gor=[150.0, 150.0]))
    conditions = row_conditions(data, bc)
    assert [(c.u, c.p_s, c.w_lg, c.p_r, c.T_r) for c in conditions] == [(0.5, 20.0, 0.0, 190.0, 360.0),
                                                                       (0.8, 21.0, 2.0, 185.0, 360.0)]
    fluids = row_fluids(data, WP.fluid)
    assert fluids[0] is fluids[1] and fluids[0].gor == 150.0 and fluids[0].wlr == WP.fluid.wlr
    assert row_fluids(CalibrationData(rows()), WP.fluid) == [WP.fluid, WP.fluid]


# --- Parameters and priors (CAL-5 to CAL-7) ----------------------------------------------------------------------

def test_parameter_transform():
    """CAL-5: z is the standardized log, so z = 0 is the median and z = 1 one prior standard deviation above it."""
    p = Parameter('h', 15.0, 0.5)
    assert p.value(0.0) == 15.0 and p.value(1.0) == pytest.approx(15.0 * np.exp(0.5))
    assert p.z(p.value(-1.3)) == pytest.approx(-1.3)
    for bad in (dict(name='U', median=1.0, log_sd=1.0), dict(name='h', median=0.0, log_sd=1.0),
                dict(name='h', median=1.0, log_sd=0.0)):
        with pytest.raises(ValueError):
            Parameter(**bad)


def test_apply_changes_only_the_named_fields():
    """CAL-5: applying values gives a new well with those fields changed; the caller's well is not."""
    wp2 = apply(WP, {'K_c': 0.003, 'w_l_max': 33.0, 'h': 25.0, 'roughness': 1e-4})
    assert (wp2.choke.K_c, wp2.inflow.w_l_max, wp2.thermal.h, wp2.friction.roughness) == (0.003, 33.0, 25.0, 1e-4)
    assert wp2.choke.chk_profile == WP.choke.chk_profile and wp2.thermal.joule_thomson == WP.thermal.joule_thomson
    assert wp2.fluid is WP.fluid and wp2.geometry is WP.geometry
    assert WP.inflow.w_l_max == 50.0 and WP.thermal.h == 20.0
    # The Rust core receives the values (solvers/rust.py, core_well)
    assert _inflow(wp2.inflow)['inflow_coefficient'] == 33.0 and _friction(wp2.friction)['roughness'] == 1e-4


def test_apply_checks_the_component_class():
    with pytest.raises(ValueError, match='ProductivityIndex'):
        apply(WP, {'k_l': 0.5})
    with pytest.raises(ValueError, match='FixedFrictionFactor'):
        apply(WP, {'f_D': 0.02})
    with pytest.raises(ValueError, match='unknown parameter'):
        apply(WP, {'U': 1.0})
    wp_pi = replace(WP, inflow=ProductivityIndex(k_l=0.5), friction=FixedFrictionFactor(f_D=0.03))
    assert apply(wp_pi, {'k_l': 0.7, 'f_D': 0.02}).friction.f_D == 0.02


def test_default_priors():
    """CAL-6: the table of specs/calibration.md."""
    A = WP.geometry.A
    k = default_prior('K_c', WP)
    assert (k.median, k.log_sd) == (pytest.approx(0.12 * A), 0.35)
    k = default_prior('K_c', WP, A_c=0.002)
    assert (k.median, k.log_sd) == (pytest.approx(0.0012), 0.2)
    assert (default_prior('w_l_max', WP).median, default_prior('w_l_max', WP).log_sd) == (50.0, 1.15)
    assert (default_prior('roughness', WP).median, default_prior('roughness', WP).log_sd) == (4.6e-5, 1.15)
    assert (default_prior('h', WP).median, default_prior('h', WP).log_sd) == (15.0, 0.5)
    wp_f = replace(WP, friction=FixedFrictionFactor(f_D=0.05), inflow=ProductivityIndex(k_l=0.8))
    assert (default_prior('f_D', wp_f).median, default_prior('f_D', wp_f).log_sd) == (0.02, 0.5)
    assert (default_prior('k_l', wp_f).median, default_prior('k_l', wp_f).log_sd) == (0.8, 1.15)
    with pytest.raises(ValueError, match='RoughnessFriction'):
        default_prior('roughness', wp_f)
    with pytest.raises(ValueError, match='A_c'):
        default_prior('K_c', WP, A_c=-1.0)


def test_a_well_the_backend_cannot_solve_is_refused_before_any_solve():
    """A component the Rust core lacks is a ValueError, not rows taken for ones that cannot flow."""
    class MyInflow(Vogel):
        pass

    wp = replace(WP, inflow=MyInflow(w_l_max=50.0))
    with pytest.raises(ValueError, match='Rust core cannot solve'):
        calibrate(wp, CalibrationData(rows()), BoundaryConditions(), ['K_c'])
    with pytest.raises(ValueError, match='Rust core cannot solve'):
        evaluate(wp, CalibrationData(rows()), BoundaryConditions())
    with pytest.raises(ValueError, match='backend must be'):
        calibrate(WP, CalibrationData(rows()), BoundaryConditions(), ['K_c'], backend='fortran')


def test_priors_for():
    ps = priors_for(WP, ['K_c', 'h'], priors={'h': Parameter('h', 30.0, 0.2)})
    assert [p.name for p in ps] == ['K_c', 'h'] and ps[1].median == 30.0
    for free, priors, match in ((['K_c', 'K_c'], None, 'once'), ([], None, 'at least one'),
                                (['K_c'], {'h': Parameter('h', 1.0, 1.0)}, 'not free'),
                                (['h'], {'h': Parameter('K_c', 1.0, 1.0)}, 'named')):
        with pytest.raises(ValueError, match=match):
            priors_for(WP, free, priors)


def test_darcy_productivity_index():
    """CAL-7: 100 mD, 20 m, 1 cP, B = 1.2, r_e = 500 m, r_w = 0.1 m, no skin, 850 kg/Sm³."""
    k_l = darcy_productivity_index(100 * MILLIDARCY, 20.0, 1 * CENTIPOISE, 1.2, 500.0, 0.1, 0.0, 850.0)
    expected = 850.0 * 2 * np.pi * 100 * 9.869233e-16 * 20.0 / (1e-3 * 1.2 * (np.log(5000.0) - 0.75)) * 1e5
    assert k_l == pytest.approx(expected) and 0.1 < k_l < 0.12   # About 11.5 Sm³/d per bar
    with pytest.raises(ValueError):
        darcy_productivity_index(100 * MILLIDARCY, 20.0, 1e-3, 1.2, 0.05, 0.1, 0.0, 850.0)


def test_vogel_maximum_rate_matches_the_productivity_index_at_p_r():
    """CAL-7: Vogel's slope at p_0 = p_r is the productivity index."""
    k_l, p_r = 0.8, 200.0
    v = Vogel(w_l_max=vogel_maximum_rate(k_l, p_r))
    h = 1e-4
    slope = (v.liquid_mass_flow_rate(p_r - h, p_r) - v.liquid_mass_flow_rate(p_r, p_r)) / h
    assert slope == pytest.approx(k_l, rel=1e-4)


# --- Predicted observations and residuals (CAL-8 to CAL-10) -----------------------------------------------------

def fake_root(n=4, w_res=10.0, w_g_res=1.5):
    X = np.tile([100.0, 2.0, 1.0, 0.3, 80.0, 800.0, 340.0], (n, 1))
    X[0, 0], X[-1, 0], X[0, 6], X[-1, 6] = 160.0, 40.0, 360.0, 330.0
    return Root(x=X.ravel(), label='stable', slope=-1.0, choked=False, flow_regime=('slug',) * n, w_res=w_res,
                w_g_res=w_g_res)


def test_predicted_observations_are_the_dataset_features():
    """CAL-8: PBH, PWH, TWH and WLIQ + WGAS, as root_features defines them."""
    root, bc, f = fake_root(), BoundaryConditions(w_lg=2.0), WP.fluid
    p = predicted(root, bc, f)
    features = root_features(root, bc, f.rho_o, f.rho_w, f.rho_g, 1 - f.f_o_in_liquid)
    assert p.tolist() == [160.0, 40.0, 330.0, features['WLIQ'] + features['WGAS']]
    assert p[3] == pytest.approx(11.5)


def test_scaled_residuals_masks_and_barrier():
    """CAL-9, CAL-10: noise-scaled residuals, log residuals for the rate, missing ones left out, and the barrier
    for every observation of a row with no operating point."""
    data = CalibrationData(rows(PBH=[np.nan, 140.0]))
    obj = Objective(WP, data, BoundaryConditions(), priors_for(WP, ['K_c']))
    y, sd = obj.y, obj.sd
    pred = y.copy()
    pred[0, 1] -= 0.6
    pred[0, 3] *= np.exp(-0.05)
    r = obj.scaled(pred)
    assert len(r) == 7
    assert r[0] == pytest.approx(0.6 / sd[0, 1]) and r[2] == pytest.approx(0.05 / sd[0, 3])
    assert np.allclose(r[3:], 0.0)
    pred[1] = np.nan
    assert obj.scaled(pred)[3:].tolist() == [BARRIER] * 4
    assert obj.values([1.0]) == {'K_c': pytest.approx(obj.parameters[0].median * np.exp(0.35))}


# --- Synthetic data (CAL-13) ------------------------------------------------------------------------------------

def table(n=10):
    rng = np.random.default_rng(0)
    return pd.DataFrame({'CHK': np.linspace(0.2, 1.0, n), 'PDC': 20.0, 'WGL': 0.0, 'PBH': 150.0 + rng.random(n),
                         'PWH': 40.0, 'TWH': 330.0, 'QOIL': 30.0, 'QGAS': 6000.0, 'QWAT': 3.0})


def test_operating_points_spread():
    bcs = operating_points(BoundaryConditions(p_s=20.0), 9, np.random.default_rng(1), u=(0.2, 1.0), w_lg=(0, 2))
    assert sorted(b.u for b in bcs) == pytest.approx(np.linspace(0.2, 1.0, 9).tolist())
    assert all(18.0 <= b.p_s <= 22.0 and 0 <= b.w_lg <= 2 for b in bcs)


def test_instrumentations():
    """CAL-13: each instrumentation keeps the observations it names."""
    df = table()
    rng = np.random.default_rng(3)
    assert instrument(df, 'full', rng).notna().all(axis=None)
    assert instrument(df, 'no_downhole', rng)['PBH'].isna().all()
    periodic = instrument(df, 'periodic_tests', rng)
    assert periodic['QOIL'].notna().sum() == 2 and periodic['PBH'].notna().all()
    assert instrument(df, 'pressures_only', rng)['QGAS'].notna().sum() == 2
    missing = instrument(df, 'random_missing', rng)
    assert (missing[['QOIL', 'QGAS', 'QWAT']].isna().nunique(axis=1) == 1).all()
    for name in INSTRUMENTATIONS:
        CalibrationData(instrument(df, name, np.random.default_rng(5)))  # every row keeps an observation
    with pytest.raises(ValueError, match='unknown instrumentation'):
        instrument(df, 'none', rng)


def test_noise_is_seeded_and_keeps_the_fractions():
    df = table()
    a, b = (add_noise(df, Noise(), np.random.default_rng(11)) for _ in range(2))
    pd.testing.assert_frame_equal(a, b)
    assert np.allclose(a['QGAS'] / a['QOIL'], 200.0) and np.allclose(a['QWAT'] / a['QOIL'], 0.1)
    assert not np.allclose(a['PBH'], df['PBH']) and (a['RATE_SOURCE'] == 'test').all()
