"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The parameters of a calibration and their physics-based priors (specs/calibration.md, CAL-5 to CAL-7).
"""

from dataclasses import dataclass, replace

import numpy as np

from manywells.friction import FixedFrictionFactor, RoughnessFriction
from manywells.inflow import ProductivityIndex, Vogel
from manywells.units import CF_BAR

# spec: CAL-5. Each parameter is a field of a component of WellProperties, and the class the component must be
FIELDS = {
    'K_c': ('choke', 'K_c', None),
    'k_l': ('inflow', 'k_l', ProductivityIndex),
    'w_l_max': ('inflow', 'w_l_max', Vogel),
    'f_D': ('friction', 'f_D', FixedFrictionFactor),
    'roughness': ('friction', 'roughness', RoughnessFriction),
    'h': ('thermal', 'h', None),
}

MILLIDARCY = 9.869233e-16   # m²
CENTIPOISE = 1e-3           # Pa s


@dataclass(frozen=True)
class Parameter:  # spec: CAL-5
    """
    A free parameter with a log-normal prior: log(value) ~ N(log(median), log_sd²). The fit works in the
    standardized log z = (log(value) - log(median)) / log_sd, so z = 0 is the prior median and z is the parameter's
    shift from it in prior standard deviations.
    """
    name: str           # One of FIELDS
    median: float
    log_sd: float
    basis: str = ''     # Where the prior comes from

    def __post_init__(self):
        if self.name not in FIELDS:
            raise ValueError(f'unknown parameter {self.name!r}; the parameters are {", ".join(FIELDS)}')
        if not (np.isfinite(self.median) and self.median > 0):
            raise ValueError(f'{self.name}: the prior median must be positive')
        if not (np.isfinite(self.log_sd) and self.log_sd > 0):
            raise ValueError(f'{self.name}: the prior log_sd must be positive')

    def value(self, z: float) -> float:
        return float(self.median * np.exp(self.log_sd * z))

    def z(self, value: float) -> float:
        return float((np.log(value) - np.log(self.median)) / self.log_sd)


def check_applies(wp, name: str):
    """Raise ValueError if parameter `name` is not a field of wp's component."""
    if name not in FIELDS:
        raise ValueError(f'unknown parameter {name!r}; the parameters are {", ".join(FIELDS)}')
    component, fld, cls = FIELDS[name]
    part = getattr(wp, component)
    if cls is not None and not isinstance(part, cls):
        raise ValueError(f'parameter {name} needs {component} to be a {cls.__name__}, '
                         f'but the well has a {type(part).__name__}')


def apply(wp, values: dict):  # spec: CAL-5
    """A copy of WellProperties wp with the parameters set to values ({name: value}); wp is not changed."""
    changes = {}
    for name, value in values.items():
        check_applies(wp, name)
        component, fld, _ = FIELDS[name]
        part = changes.get(component, getattr(wp, component))
        changes[component] = replace(part, **{fld: float(value)})
    return replace(wp, **changes)


def current_value(wp, name: str) -> float:
    """The value parameter `name` has in wp."""
    check_applies(wp, name)
    component, fld, _ = FIELDS[name]
    return float(getattr(getattr(wp, component), fld))


def default_prior(name: str, wp, A_c: float = None) -> Parameter:  # spec: CAL-6
    """
    The default prior of parameter `name` for well wp (specs/calibration.md, CAL-6).

    :param name: One of FIELDS
    :param wp: The well; its tubing area sets the choke's prior without A_c, and its inflow the productivity's median
    :param A_c: The choke's full-open throat area (m²), if known
    """
    check_applies(wp, name)
    if name == 'K_c':
        if A_c is not None:
            if not A_c > 0:
                raise ValueError('the throat area A_c must be positive')
            return Parameter('K_c', 0.6 * A_c, 0.2, 'C_D A_c with C_D = 0.6 (Haug 2012)')
        return Parameter('K_c', 0.12 * wp.geometry.A, 0.35, '0.12 A, the paper\'s reference (SMP-5)')
    if name in ('k_l', 'w_l_max'):
        return Parameter(name, current_value(wp, name), 1.15, "the well's value, a factor of 10 either way")
    if name == 'f_D':
        return Parameter('f_D', 0.02, 0.5, 'steel tubing (Moody), wider for holdup error')
    if name == 'roughness':
        return Parameter('roughness', 4.6e-5, 1.15, 'commercial steel (Moody 1944), drawn to corroded')
    return Parameter('h', 15.0, 0.5, 'effective coefficient on T - T_a (Hasan and Kabir 2012)')


def darcy_productivity_index(k, h_net, mu, B, r_e, r_w, skin, rho_sc) -> float:  # spec: CAL-7
    """
    The liquid productivity index k_l (kg/(s bar)) of pseudo-steady radial inflow, for a physics-based prior:
        k_l = rho_sc * 2 pi k h_net / (mu B (ln(r_e / r_w) - 3/4 + skin)) * 1e5.

    :param k: Permeability (m²); MILLIDARCY converts from mD
    :param h_net: Net pay (m)
    :param mu: Liquid viscosity at reservoir conditions (Pa s); CENTIPOISE converts from cP
    :param B: Formation volume factor (m³ at reservoir conditions per Sm³)
    :param r_e: Drainage radius (m)
    :param r_w: Wellbore radius (m)
    :param skin: Skin factor (dimensionless)
    :param rho_sc: Liquid density at standard conditions (kg/Sm³)
    """
    denominator = np.log(r_e / r_w) - 0.75 + skin
    if not (k > 0 and h_net > 0 and mu > 0 and B > 0 and r_e > r_w > 0 and rho_sc > 0 and denominator > 0):
        raise ValueError('darcy_productivity_index needs positive inputs, r_e > r_w and ln(r_e/r_w) - 3/4 + skin > 0')
    return float(rho_sc * 2 * np.pi * k * h_net / (mu * B * denominator) * CF_BAR)


def vogel_maximum_rate(k_l, p_r) -> float:  # spec: CAL-7
    """Vogel's maximum rate w_l_max (kg/s) whose slope at p_0 = p_r is the productivity index k_l: k_l p_r / 1.8."""
    return float(k_l * p_r / 1.8)
