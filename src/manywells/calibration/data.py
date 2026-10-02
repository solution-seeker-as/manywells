"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The data of a calibration (specs/calibration.md, CAL-1 to CAL-4): one well's steady periods, one row each, with the
roles of their columns, the observations they hold and the noise of each.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from manywells.datasets.schema import SECONDS_PER_HOUR

# spec: CAL-1. Inputs: known for every row. CHK and PDC are required; the others default to the well's values.
REQUIRED_INPUTS = ('CHK', 'PDC')
OPTIONAL_INPUTS = ('WGL', 'p_r', 'T_r', 'T_s', 'T_lg', 'gor', 'wlr')
# spec: CAL-2. Observations, each may be missing (NaN). The phase rates are at standard conditions (Sm³/h).
PRESSURES_AND_TEMPERATURES = ('PBH', 'PWH', 'TWH')
PHASE_RATES = ('QOIL', 'QGAS', 'QWAT')
# The observations of the objective, in this order: PBH, PWH, TWH and the reservoir mass rate (CAL-3)
OBSERVATIONS = PRESSURES_AND_TEMPERATURES + ('WRES',)
RATE_SOURCES = ('test', 'mpfm')
SD_COLUMNS = {'PBH': 'PBH_SD', 'PWH': 'PWH_SD', 'TWH': 'TWH_SD', 'WRES': 'RATE_SD'}


@dataclass(frozen=True)
class Noise:  # spec: CAL-4
    """
    The default standard deviation of each observation's measurement noise: absolute for the pressures (bar) and the
    temperature (K), relative for the rate, by the source of the row's rates. A row's *_SD columns override them.
    """
    PBH: float = 0.3            # Downhole gauge (bar)
    PWH: float = 0.3            # Wellhead pressure transmitter (bar)
    TWH: float = 1.0            # Wellhead temperature (K), installed bias included
    rate_test: float = 0.025    # Test separator, relative
    rate_mpfm: float = 0.10     # Multiphase flow meter, relative

    def __post_init__(self):
        for name in ('PBH', 'PWH', 'TWH', 'rate_test', 'rate_mpfm'):
            if not getattr(self, name) > 0:
                raise ValueError(f'noise {name} must be positive')


@dataclass(frozen=True)
class CalibrationData:
    """
    One well's calibration data: a table with one row per steady period (a well test, an MPFM average, or a period
    with sensors only), in bar, K and Sm³/h.

    Inputs (CAL-1), with no missing values:
        CHK, PDC                    choke position in (0, 1] and the pressure downstream of the choke (bar)
        WGL                         lift-gas rate (kg/s), 0 if absent
        p_r, T_r, T_s, T_lg         reservoir pressure (bar) and temperatures (K), the well's if absent
        gor, wlr                    the fluid's gas-oil ratio (Sm³/Sm³) and water-liquid ratio, the well's if absent
    Observations (CAL-2), NaN where missing:
        PBH, PWH, TWH               bottomhole and wellhead pressure (bar), wellhead temperature (K)
        QOIL, QGAS, QWAT            phase rates at standard conditions (Sm³/h); all three or none
    Noise (CAL-4):
        RATE_SOURCE                 'test' or 'mpfm', the source of the row's rates ('test' if absent)
        PBH_SD, PWH_SD, TWH_SD      absolute standard deviations that override noise's
        RATE_SD                     relative standard deviation of the rate that overrides noise's

    Other columns, such as a time stamp, are kept and appear in the result's residual table.
    """
    rows: pd.DataFrame
    noise: Noise = field(default_factory=Noise)

    def __post_init__(self):
        df = pd.DataFrame(self.rows).reset_index(drop=True).copy()
        object.__setattr__(self, 'rows', df)
        if not isinstance(self.noise, Noise):
            raise ValueError(f'noise must be a Noise, got {type(self.noise).__name__}')
        if len(df) == 0:
            raise ValueError('the calibration data has no rows')
        for col in REQUIRED_INPUTS:
            if col not in df:
                raise ValueError(f'the calibration data needs the input column {col}')
        for col in REQUIRED_INPUTS + tuple(c for c in OPTIONAL_INPUTS if c in df):
            values = pd.to_numeric(df[col], errors='coerce')
            if values.isna().any():
                raise ValueError(f'input column {col} has missing or non-numeric values')
        if not ((df['CHK'] > 0) & (df['CHK'] <= 1)).all():
            raise ValueError('CHK must be in (0, 1]: every row is a flowing period')
        for col in ('PDC', 'p_r', 'T_r', 'T_s', 'T_lg'):
            if col in df and not (df[col] > 0).all():
                raise ValueError(f'input column {col} must be positive')
        if 'WGL' in df and not (df['WGL'] >= 0).all():
            raise ValueError('WGL must be non-negative')
        if 'gor' in df and not (df['gor'] > 0).all():
            raise ValueError('gor must be positive')
        if 'wlr' in df and not ((df['wlr'] >= 0) & (df['wlr'] < 1)).all():
            raise ValueError('wlr must be in [0, 1)')

        present = [c for c in PRESSURES_AND_TEMPERATURES + PHASE_RATES if c in df]
        for col in present:
            values = pd.to_numeric(df[col], errors='coerce')
            if (values.isna() & df[col].notna()).any():
                raise ValueError(f'observation column {col} has non-numeric values')
            df[col] = values
        for col in PRESSURES_AND_TEMPERATURES:
            if col in df and not (df[col].dropna() > 0).all():
                raise ValueError(f'observation {col} must be positive')
        rates = [c for c in PHASE_RATES if c in df]
        if rates:
            if len(rates) != len(PHASE_RATES):
                raise ValueError('the phase rates are QOIL, QGAS and QWAT together')
            given = df[list(PHASE_RATES)].notna()
            if not (given.all(axis=1) | ~given.any(axis=1)).all():
                raise ValueError('a row has some phase rates but not all of QOIL, QGAS and QWAT')
            q = df.loc[given.all(axis=1), list(PHASE_RATES)]
            if not ((q >= 0).all(axis=None) and (q.sum(axis=1) > 0).all()):
                raise ValueError('phase rates must be non-negative, with a positive total')
        if not self.observed().any(axis=1).all():
            raise ValueError('every row needs at least one observation (PBH, PWH, TWH or the phase rates)')

        if 'RATE_SOURCE' in df and not df['RATE_SOURCE'].fillna('test').isin(RATE_SOURCES).all():
            raise ValueError(f'RATE_SOURCE must be one of {RATE_SOURCES}')
        for col in SD_COLUMNS.values():
            if col in df and not (df[col].dropna() > 0).all():
                raise ValueError(f'{col} must be positive')

    def __len__(self):
        return len(self.rows)

    def observed(self) -> pd.DataFrame:
        """Which observations each row has, as booleans in the columns OBSERVATIONS."""
        df = self.rows
        out = pd.DataFrame({c: df[c].notna() if c in df else False for c in PRESSURES_AND_TEMPERATURES},
                           index=df.index)
        out['WRES'] = df[list(PHASE_RATES)].notna().all(axis=1) if PHASE_RATES[0] in df else False
        return out[list(OBSERVATIONS)]

    def observations(self, fluids) -> np.ndarray:  # spec: CAL-3
        """
        The observed values, an (n, 4) array in the order OBSERVATIONS, NaN where missing. The rate is the reservoir
        mass rate (kg/s), the phase rates weighted by each row's densities at standard conditions.

        :param fluids: The FluidModel of each row
        """
        df = self.rows
        y = np.full((len(df), len(OBSERVATIONS)), np.nan)
        for j, col in enumerate(PRESSURES_AND_TEMPERATURES):
            if col in df:
                y[:, j] = df[col].to_numpy(dtype=float)
        if PHASE_RATES[0] in df:
            q = df[list(PHASE_RATES)].to_numpy(dtype=float) / SECONDS_PER_HOUR
            rho = np.array([[f.rho_o, f.rho_g, f.rho_w] for f in fluids])
            y[:, 3] = (q * rho).sum(axis=1)
        return y

    def noise_sd(self) -> np.ndarray:  # spec: CAL-4
        """The standard deviation of each observation, an (n, 4) array in the order OBSERVATIONS."""
        df, noise = self.rows, self.noise
        source = df['RATE_SOURCE'].fillna('test') if 'RATE_SOURCE' in df else pd.Series('test', index=df.index)
        defaults = {'PBH': noise.PBH, 'PWH': noise.PWH, 'TWH': noise.TWH,
                    'WRES': np.where(source == 'mpfm', noise.rate_mpfm, noise.rate_test)}
        sd = np.empty((len(df), len(OBSERVATIONS)))
        for j, obs in enumerate(OBSERVATIONS):
            col = SD_COLUMNS[obs]
            sd[:, j] = defaults[obs]
            if col in df:
                sd[:, j] = df[col].fillna(pd.Series(sd[:, j], index=df.index)).to_numpy(dtype=float)
        return sd
