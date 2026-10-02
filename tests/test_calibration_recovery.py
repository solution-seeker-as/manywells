"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The calibration recovers known parameters (specs/calibration.md, Checks): wells drawn with manywells.sampling are
simulated at parameters a known distance from the priors' medians, and calibrated from the medians. With noise-free
rows and the noise scaled down, so that the priors' pull vanishes, the fit identifies every parameter; with noise, in
every instrumentation, it gets closer to the truth than it started, and close to the parameters the data determine.
Errors are in prior standard deviations, z.
"""

import numpy as np
import pytest

from manywells.calibration import CalibrationError, CalibrationData, Noise, calibrate, synthetic_data
from manywells.calibration.fit import priors_for
from manywells.calibration.synthetic import INSTRUMENTATIONS
from manywells.configurations import DEVELOP
from manywells.sampling.conditions import nominal_conditions
from manywells.sampling.wells import sample_well, well_properties

pytestmark = pytest.mark.slow

FREE = ['K_c', 'w_l_max', 'roughness', 'h']
Z_TRUE = np.array([1.5, -1.0, 1.0, -1.5])     # The truth, in prior standard deviations from the medians
SEED, WELLS, N_CELLS, N_ROWS = 1, (1, 3, 4), 20, 10
# The tolerances (specs/calibration.md, Checks; signed off by Bjarne, 2026-10-02)
IDENTIFIED = 1e-3       # Identification: every parameter within this many prior standard deviations
CLOSE = 0.5             # Getting close: each of DETERMINED within this many
DETERMINED = ('K_c', 'w_l_max', 'h')
SCALED_DOWN = Noise(PBH=3e-4, PWH=3e-4, TWH=1e-3, rate_test=2.5e-5, rate_mpfm=1e-4)   # The defaults / 1000


def case(well):
    draw = sample_well(SEED, well)
    wp = well_properties(draw, configuration=DEVELOP, n_cells=N_CELLS)
    bc = nominal_conditions(draw)
    truth = {p.name: p.value(z) for p, z in zip(priors_for(wp, FREE), Z_TRUE)}
    return wp, bc, truth


def z_error(result):
    return np.array([result.z[name] for name in FREE]) - Z_TRUE


@pytest.mark.parametrize('well', WELLS)
def test_identification(well):
    """Noise-free rows, full instrumentation, all four parameters free: the fit returns the truth."""
    wp, bc, truth = case(well)
    data = synthetic_data(wp, bc, truth, N_ROWS, seed=well, noisy=False, noise=SCALED_DOWN)
    result = calibrate(wp, data, bc, FREE)
    assert result.success
    assert np.abs(z_error(result)).max() < IDENTIFIED
    assert result.values == pytest.approx(truth, rel=1e-3)


@pytest.mark.parametrize('instrumentation', INSTRUMENTATIONS)
@pytest.mark.parametrize('well', WELLS)
def test_getting_close(well, instrumentation):
    """Noisy rows: the fit ends closer to the truth than the priors' medians are, and close to it in each parameter
    the data determine; the residuals are of the noise's size."""
    wp, bc, truth = case(well)
    data = synthetic_data(wp, bc, truth, N_ROWS, seed=well, instrumentation=instrumentation)
    result = calibrate(wp, data, bc, FREE)
    assert result.success
    err = z_error(result)
    assert np.linalg.norm(err) < np.linalg.norm(Z_TRUE)
    assert all(abs(err[FREE.index(name)]) < CLOSE for name in DETERMINED)
    assert all(rms < 2.0 for rms in result.rms().values())


def test_deterministic():
    wp, bc, truth = case(WELLS[0])
    data = synthetic_data(wp, bc, truth, N_ROWS, seed=5, instrumentation='random_missing')
    a, b = (calibrate(wp, data, bc, FREE) for _ in range(2))
    assert a.values == b.values and a.z == b.z


def test_result_and_diagnostics():
    """CAL-12: the calibrated well, the residual table and the flag on a parameter far from its prior."""
    wp, bc, truth = case(WELLS[0])
    truth = {'K_c': truth['K_c'], 'h': 15.0 * np.exp(0.5 * 3.0)}   # h three prior standard deviations above its median
    data = synthetic_data(wp, bc, truth, N_ROWS, seed=2, noisy=False, noise=SCALED_DOWN)
    result = calibrate(wp, data, bc, ['K_c', 'h'])
    assert result.well.thermal.h == pytest.approx(result.values['h']) and result.flagged() == ['h']
    table = result.residuals
    assert len(table) == len(data) and {'CHK', 'PWH_obs', 'PWH_pred', 'PWH_res', 'WRES_res'} <= set(table)
    assert 'far from its prior' in result.summary()


def test_start_off_the_medians():
    """CAL-9: where a row cannot flow at the priors' medians, the fit starts where every row flows, and still gets
    close. Well 1 of seed 2026 (L-shaped) does not flow at the medians at one of its choke openings."""
    draw = sample_well(2026, 1)
    wp = well_properties(draw, configuration=DEVELOP, n_cells=N_CELLS)
    bc = nominal_conditions(draw)
    z_true = np.array([-0.7, 0.1, -1.5, -0.4])
    truth = {p.name: p.value(z) for p, z in zip(priors_for(wp, FREE), z_true)}
    result = calibrate(wp, synthetic_data(wp, bc, truth, N_ROWS, seed=0), bc, FREE)
    assert any(z != 0 for z in result.start.values()) and result.success
    err = np.array([result.z[name] for name in FREE]) - z_true
    assert all(abs(err[FREE.index(name)]) < CLOSE for name in DETERMINED)


def test_no_operating_point_at_the_start():
    """CAL-9: a row the well cannot flow at, at the priors' medians or any start near them, stops the fit before
    it starts."""
    wp, bc, truth = case(WELLS[0])
    data = synthetic_data(wp, bc, truth, 4, seed=3)
    df = data.rows.copy()
    df.loc[0, 'PDC'] = bc.p_r + 10.0        # The separator above the reservoir: no flow
    with pytest.raises(CalibrationError, match='no operating point'):
        calibrate(wp, CalibrationData(df), bc, ['K_c'])
