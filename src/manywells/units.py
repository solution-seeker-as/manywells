"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Units, physical constants, and conversion helpers.

All public interfaces in the manywells package use SI units (Pa, K, kg/m3,
m, s, J, ...).  Some correlations accept field-unit quantities such as API
gravity where that is the natural parameterization; these cases are clearly
documented at the call site.

Standard conditions are defined per ISO 13443: 15 degC (288.15 K) and
1 atm (101 325 Pa).

Conversion factors in this module convert *from* the named unit *to* the
corresponding SI unit.  For example, multiplying a value in psi by CF_PSI
yields the equivalent value in Pa.
"""

################################################
# PHYSICAL CONSTANTS
################################################

STD_GRAVITY = 9.80665       # Standard acceleration of gravity (m/s2)
R_UNIVERSAL = 8314.46       # Universal gas constant (J/(kmol K))
M_AIR = 28.97               # Molecular weight of air (g/mol = kg/kmol)


################################################
# STANDARD REFERENCE CONDITIONS (ISO 13443)
################################################

P_REF = 101_325             # Reference pressure (Pa)
T_REF = 288.15              # Reference temperature (K), i.e. 15 degC


################################################
# PRESSURE CONVERSION FACTORS
################################################

CF_BAR = 1e5                # bar -> Pa
CF_PSI = 6894.76            # psi -> Pa


################################################
# VOLUME / RATIO CONVERSION FACTORS
################################################

CF_RS = 0.178108            # scf/STB -> Sm3/Sm3


################################################
# VISCOSITY CONVERSION FACTORS
################################################

CF_CP = 1e-3                # centipoise -> Pa s
CF_UP = 1e-7                # micropoise -> Pa s


################################################
# OTHER CONVERSION FACTORS
################################################

CF_DYNCM = 1e-3             # dyn/cm -> J/m2 (surface tension)
CF_KGM3_TO_GCC = 1e-3       # kg/m3 -> g/cm3


################################################
# TEMPERATURE CONVERSION HELPERS
################################################

def kelvin_to_fahrenheit(T):
    """Convert temperature from Kelvin to degrees Fahrenheit.

    Compatible with both float and CasADi symbolic types.
    """
    return 1.8 * (T - 273.15) + 32


def fahrenheit_to_kelvin(T_F):
    """Convert temperature from degrees Fahrenheit to Kelvin.

    Compatible with both float and CasADi symbolic types.
    """
    return (T_F - 32) / 1.8 + 273.15


def kelvin_to_celsius(T):
    """Convert temperature from Kelvin to degrees Celsius.

    Compatible with both float and CasADi symbolic types.
    """
    return T - 273.15


def kelvin_to_rankine(T):
    """Convert temperature from Kelvin to degrees Rankine.

    Compatible with both float and CasADi symbolic types.
    """
    return 1.8 * T
