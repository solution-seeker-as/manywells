"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Water thermophysical properties (CasADi-compatible).

Formation volume factor and viscosity.
"""

import casadi as ca

from manywells.units import P_REF


def water_fvf(p, T, p_ref=P_REF, c_w=4.5e-10):
    """
    Water formation volume factor with constant compressibility, to first order.

    Bw = 1 - c_w * (p - p_ref)

    Water shrinks under pressure, so Bw falls below 1.0 above p_ref.  It is
    nearly incompressible, so Bw stays very close to 1.0 under typical
    wellbore conditions.  Not used by the simulator.

    :param p: Pressure (Pa), may be CasADi symbolic
    :param T: Temperature (K), unused (included for API consistency)
    :param p_ref: Reference pressure (Pa), default ISO 13443
    :param c_w: Isothermal water compressibility (1/Pa), default 4.5e-10
    :return: Bw (dimensionless)
    """
    return 1.0 - c_w * (p - p_ref)  # spec: PVT-WAT-2


def water_viscosity(T):  # spec: PVT-WAT-3
    """
    Water viscosity using a Vogel-Fulcher-Tammann type correlation.
    CasADi-compatible.

    :param T: Temperature (K), may be a CasADi symbolic
    :return: Water viscosity (Pa·s)
    """
    return 2.414e-5 * ca.power(10, 247.8 / (T - 140))
