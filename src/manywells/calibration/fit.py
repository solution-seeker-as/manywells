"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The fit of a calibration (specs/calibration.md, CAL-11 and CAL-12): the MAP estimate of the free parameters, by
least squares from the prior medians, and its result with diagnostics.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from manywells.calibration.data import OBSERVATIONS, CalibrationData
from manywells.calibration.objective import Objective, failed_rows
from manywells.calibration.parameters import Parameter, apply, check_applies, default_prior
from manywells.simulator import SimError

log = logging.getLogger('manywells')

# spec: CAL-11. Stopping rule of the least-squares solver
X_TOL = 1e-6        # Step in z (prior standard deviations)
F_TOL = 1e-10       # Relative change of the cost
G_TOL = 1e-10       # Gradient, scaled
MAX_NFEV = 100      # Evaluations of the residuals, Jacobians not counted
SHIFT_LIMIT = 2.0   # A parameter more than this many prior standard deviations from its median is flagged (CAL-12)
START_SHIFTS = (1.0, 2.0)   # The starts tried along each parameter, besides the medians (CAL-9)


class CalibrationError(SimError):
    """A calibration that cannot start or does not converge."""


@dataclass(frozen=True)
class CalibrationResult:  # spec: CAL-12
    """
    The MAP estimate of one well's free parameters, with diagnostics.

    values      the estimate, {name: value}
    z           each parameter's shift from its prior median, in prior standard deviations
    parameters  the free parameters with their priors
    well        the calibrated WellProperties
    residuals   one row per data row: the inputs and other columns of the data, and for each observation its
                observed and predicted value and its scaled residual (*_obs, *_pred, *_res), NaN where missing
    success     whether the solver met its stopping rule; message is its reason
    start       the z the fit started from (CAL-9)
    """
    values: dict
    z: dict
    parameters: tuple
    well: object
    residuals: pd.DataFrame
    success: bool
    message: str
    cost: float
    n_solves: int
    start: dict

    def flagged(self) -> list:
        """The parameters more than SHIFT_LIMIT prior standard deviations from their medians: a sign of model error
        rather than of a better estimate (CAL-12)."""
        return [name for name, z in self.z.items() if abs(z) > SHIFT_LIMIT]

    def rms(self) -> dict:
        """The root mean square of each observation's scaled residuals; near 1 where the model fits within the noise."""
        out = {}
        for obs in OBSERVATIONS:
            r = self.residuals[f'{obs}_res'].dropna()
            if len(r):
                out[obs] = float(np.sqrt(np.mean(r ** 2)))
        return out

    def summary(self) -> str:
        lines = [f'{"parameter":<10} {"value":>12} {"prior median":>13} {"shift (sd)":>11}']
        for p in self.parameters:
            flag = '  <- far from its prior' if p.name in self.flagged() else ''
            lines.append(f'{p.name:<10} {self.values[p.name]:>12.5g} {p.median:>13.5g} {self.z[p.name]:>11.2f}{flag}')
        lines.append('RMS of scaled residuals: ' + ', '.join(f'{k} {v:.2f}' for k, v in self.rms().items()))
        lines.append(f'{"converged" if self.success else "not converged"}: {self.message}')
        return '\n'.join(lines)


def priors_for(wp, free, priors=None, A_c=None) -> tuple:
    """The free parameters with their priors: priors[name] where given, else default_prior (CAL-6)."""
    priors = dict(priors or {})
    unknown = set(priors) - set(free)
    if unknown:
        raise ValueError(f'priors given for parameters that are not free: {", ".join(sorted(unknown))}')
    if len(set(free)) != len(free) or not free:
        raise ValueError('free must name each parameter once, and at least one')
    out = []
    for name in free:
        check_applies(wp, name)
        p = priors.get(name) or default_prior(name, wp, A_c=A_c)
        if not isinstance(p, Parameter) or p.name != name:
            raise ValueError(f'the prior of {name} must be a Parameter named {name!r}')
        out.append(p)
    return tuple(out)


def residual_table(objective, pred) -> pd.DataFrame:
    """The data's rows with each observation's observed and predicted value and scaled residual."""
    df = objective.data.rows.copy()
    for j, obs in enumerate(OBSERVATIONS):
        y, p, sd = objective.y[:, j], pred[:, j], objective.sd[:, j]
        df[f'{obs}_obs'] = y
        df[f'{obs}_pred'] = p
        df[f'{obs}_res'] = (np.log(y) - np.log(p)) / sd if obs == 'WRES' else (y - p) / sd
    return df


def start(objective) -> np.ndarray:  # spec: CAL-9
    """
    The start of the fit: of the prior medians, z = 0, and the starts START_SHIFTS prior standard deviations from them
    along each parameter in turn, the one with the lowest cost at which every row has an operating point; the
    medians if they tie. Raises CalibrationError if there is none.
    """
    p = len(objective.parameters)
    candidates = [np.zeros(p)] + [s * sign * e for s in START_SHIFTS for e in np.eye(p) for sign in (1, -1)]
    evaluations = objective.predict(candidates)
    feasible = [(0.5 * np.sum(np.concatenate([objective.scaled(pred), z]) ** 2), k, z)
                for k, (z, (pred, _)) in enumerate(zip(candidates, evaluations)) if not failed_rows(pred).any()]
    if feasible:
        _, k, z = min(feasible, key=lambda t: (t[0], t[1]))
        if k:
            log.info('starting at z = %s, the cheapest feasible start (CAL-9)', z)
        return z
    failed = np.flatnonzero(failed_rows(evaluations[0][0]))
    raise CalibrationError(f'{len(failed)} of {objective.n_rows} rows have no operating point at the prior medians '
                           f'(rows {", ".join(map(str, failed[:10]))}{", ..." if len(failed) > 10 else ""}), and no '
                           f'start {", ".join(map(str, START_SHIFTS))} prior standard deviations from them has one at '
                           'every row; check their inputs or the priors (CAL-9)')


def evaluate(wp, data: CalibrationData, bc, backend='rust', workers=None) -> pd.DataFrame:  # spec: CAL-12
    """
    The residual table of a well on data, as in CalibrationResult.residuals: for each row, each observation's
    observed and predicted value and its scaled residual. A row with no operating point has NaN predictions. Use it
    to check a calibrated well (result.well) on rows it was not calibrated on.
    """
    if not isinstance(data, CalibrationData):
        raise ValueError(f'data must be a CalibrationData, got {type(data).__name__}')
    objective = Objective(wp, data, bc, (), backend=backend, workers=workers)
    (pred, _), = objective.predict([np.zeros(0)])
    return residual_table(objective, pred)


def calibrate(wp, data: CalibrationData, bc, free, priors=None, A_c=None, backend='rust',
              workers=None) -> CalibrationResult:  # spec: CAL-11
    """
    Calibrate one well: the MAP estimate of the free parameters under their priors and the data's measurement noise.

        result = calibrate(wp, data, bc, free=['K_c', 'w_l_max', 'h'])
        result.values, result.well, result.summary()

    :param wp: The well (WellProperties). The parameters that are not free keep its values.
    :param data: The well's rows (CalibrationData)
    :param bc: The well's boundary conditions (BoundaryConditions); each row replaces the ones it has columns for
    :param free: The names of the free parameters (calibration.parameters.FIELDS)
    :param priors: {name: Parameter} replacing default priors
    :param A_c: The choke's full-open throat area (m²), for K_c's default prior
    :param backend: 'rust' (default) or 'casadi'
    :param workers: Threads for the Rust core's solves (default: the number of CPUs)
    :raises CalibrationError: if a row has no operating point at the prior medians
    """
    if not isinstance(data, CalibrationData):
        raise ValueError(f'data must be a CalibrationData, got {type(data).__name__}')
    parameters = priors_for(wp, list(free), priors, A_c)
    objective = Objective(wp, data, bc, parameters, backend=backend, workers=workers)
    z0 = start(objective)
    sol = least_squares(objective.residuals, z0, jac=objective.jacobian, method='trf', x_scale=1.0,
                        xtol=X_TOL, ftol=F_TOL, gtol=G_TOL, max_nfev=MAX_NFEV)
    (pred, _), = objective.predict([sol.x])
    if failed_rows(pred).any():
        raise CalibrationError('the fit ended at parameters under which a row has no operating point')
    values = objective.values(sol.x)
    if not sol.success:
        log.warning('calibration did not converge: %s', sol.message)
    return CalibrationResult(values=values, z={p.name: float(zi) for p, zi in zip(parameters, sol.x)},
                             parameters=parameters, well=apply(wp, values), residuals=residual_table(objective, pred),
                             success=bool(sol.success), message=str(sol.message), cost=float(sol.cost),
                             n_solves=objective.n_solves,
                             start={p.name: float(zi) for p, zi in zip(parameters, z0)})
