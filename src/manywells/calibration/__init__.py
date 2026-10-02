"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 20 March 2025
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Calibration of a well to its production data (specs/calibration.md): the MAP estimate of a few free parameters
(the choke coefficient, the inflow productivity, friction and heat transfer), each with a physics-based prior, from
the rows a well's instruments give, with missing values.

    from manywells.calibration import CalibrationData, calibrate
    result = calibrate(wp, CalibrationData(df), bc, free=['K_c', 'w_l_max', 'roughness', 'h'])
    print(result.summary())
"""

from manywells.calibration.data import CalibrationData, Noise, OBSERVATIONS
from manywells.calibration.fit import CalibrationError, CalibrationResult, calibrate, evaluate
from manywells.calibration.parameters import (CENTIPOISE, FIELDS, MILLIDARCY, Parameter, apply,
                                              darcy_productivity_index, default_prior, vogel_maximum_rate)
from manywells.calibration.synthetic import INSTRUMENTATIONS, synthetic_data

__all__ = ['CalibrationData', 'Noise', 'OBSERVATIONS', 'CalibrationError', 'CalibrationResult', 'calibrate',
           'evaluate', 'FIELDS', 'MILLIDARCY', 'CENTIPOISE',
           'Parameter', 'apply', 'darcy_productivity_index', 'default_prior', 'vogel_maximum_rate',
           'INSTRUMENTATIONS', 'synthetic_data']
