"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Gas formation volume factor and standard-condition density (CasADi-compatible).
"""

from manywells.pvt import P_REF, T_REF


def gas_fvf(p, T):
    """
    Gas formation volume factor using the ideal gas law.

    Bg = V_reservoir / V_standard = (p_ref / p) * (T / T_ref)

    Consistent with the gas equation of state used in the simulator
    (ideal gas: p = rho_g * R_s * T).

    :param p: Pressure (Pa), may be CasADi symbolic
    :param T: Temperature (K), may be CasADi symbolic
    :return: Bg (Sm3 at standard / m3 at reservoir conditions).
             Multiply standard-condition volume by Bg to get reservoir volume.
    """
    return (P_REF * T) / (T_REF * p)


def gas_density_std(R_s_gas):
    """
    Gas density at ISO 13443 standard conditions (15 degC, 101.325 kPa).

    :param R_s_gas: Specific gas constant (J/(kg K)), float
    :return: Gas density at standard conditions (kg/m3)
    """
    return P_REF / (R_s_gas * T_REF)
