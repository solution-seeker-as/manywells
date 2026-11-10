"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The inflow model's coefficients of a v2 well at a reservoir state (specs/sampling.md): an oil well's productivity
index (SMP-72) with the well's PVT model, and a gas or condensate well's Forchheimer coefficients fitted to its inflow
in the real-gas pseudo-pressure (SMP-117; Golan and Whitson 1991, Eq. 2.57).
"""

import math

import numpy as np

from manywells.inflow import Forchheimer
from manywells.pvt.black_oil import BlackOil
from manywells.pvt.table import PVTTable, PVTTableSet
from manywells.sampling.v2.draws import _P_ATM
from manywells.sampling.v2.reservoir import _OILS, _forchheimer, _gas_rate, _j_radial

_COEF_KEYS = ('k_l', 'w_l_max', 'forchheimer_a', 'forchheimer_b')  # The inflow's coefficients, NaN where unused
_FIT_POINTS = (0.95, 0.9, 0.8, 0.6, 0.4, 0.2)  # p_wf / p_r of the fit, with the absolute open flow  # spec: SMP-117
_N_PSEUDO = 200  # Pressures of the pseudo-pressure's integral, from 1 bar to p_r  # spec: SMP-117


def _inflow_at(w0, pvt, p_r, R_go, wlr) -> dict:  # spec: SMP-72, SMP-111
    """
    The coefficients of a well's inflow model at a reservoir state, as a dict with COEF_KEYS, NaN where the model
    does not use one: an oil well's `k_l` (kg/(s bar)), or Vogel's `w_l_max` = k_l p_r / 1.8 (kg/s), from the radial
    index at the PVT model's oil viscosity and B_o with the well's geometry factor; a gas or condensate well's
    `forchheimer_a` and `forchheimer_b` (_gas_inflow).

    :param w0: The well's draws (draws.draw_well) that fix its inflow: its initial-stage draw
    :param pvt: The well's PVT model: its black oil, or a table fluid's PVTTableSet
    :param p_r: Reservoir pressure (bar)
    :param R_go: Producing gas-oil ratio (Sm3/Sm3)
    :param wlr: Water-liquid ratio at standard conditions
    """
    out = dict.fromkeys(_COEF_KEYS, math.nan)
    if w0['fluid_type'] not in _OILS:
        return out | _gas_inflow(w0, pvt, p_r, R_go)
    T_r, ln_term, skin = w0['T_r'], w0['ln_re_rw'] - 0.75, w0['skin']
    rho_lsc = (1 - wlr) * pvt.rho_o + wlr * w0['rho_w']
    b = pvt.at_gor(R_go)
    J = _j_radial(w0['perm'], w0['h_net'], float(b.oil_viscosity(p_r, T_r)), float(b.bo(p_r, T_r)), ln_term, skin)
    k_l = J * w0['f_geo'] * rho_lsc / 86400.0
    return out | (dict(w_l_max=k_l * p_r / 1.8) if w0['inflow'] == 'vogel' else dict(k_l=k_l))


def _pseudo_pressure(gas, p_r, T_r):
    """m(p) = 2 int_1^p p/(mu Z) dp on a grid from 1 bar to p_r (bar2/(Pa s)), by the trapezoidal rule, and mu Z at
    p_r (Pa s). gas is any PVT model's gas: a black oil's, or a table's."""
    p = np.linspace(1.0, p_r, _N_PSEUDO)
    muz = np.array([float(gas.gas_viscosity(v, T_r, gas.gas_density(v, T_r)) * gas.z_factor(v, T_r)) for v in p])
    f = 2 * p / muz
    m = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(p))])
    return p, m, muz[-1]


def _fit_pressure_squared(q_of, p_r) -> tuple:  # spec: SMP-117
    """
    a and b >= 0 of p_r^2 - p_wf^2 = a q + b q^2 fitted to an inflow curve: least squares of (p_r^2 - p_wf^2)/q
    against q at p_wf/p_r = 0.95, 0.9, 0.8, 0.6, 0.4 and 0.2 and at the absolute open flow. Where the slope is
    negative, b = 0 and a is the mean of (p_r^2 - p_wf^2)/q.

    :param q_of: The rate (any unit) at a bottomhole flowing pressure p_wf (bar), a function
    :param p_r: Reservoir pressure (bar)
    """
    dp2 = np.array([p_r**2 - (f * p_r) ** 2 for f in _FIT_POINTS] + [p_r**2 - _P_ATM**2])
    q = np.array([q_of(p_r * f) for f in _FIT_POINTS] + [q_of(_P_ATM)])
    b, a = np.polyfit(q, dp2 / q, 1)
    if b < 0:
        return float(np.mean(dp2 / q)), 0.0
    return float(a), float(b)


def _wellstream_factor(api, R_go):
    """The wellstream's standard volume over the separator gas's, 1 + C_og / R_go, with the condensate's gas
    equivalent C_og = 133,000 gamma_o / M_o scf/STB and M_o = 6084 / (API - 5.9) (Whitson and Brule 2000, Eqs. 3.73b
    and 3.73c), in Sm3/Sm3."""
    g_o = 141.5 / (131.5 + api)
    c_og = 133_000 * g_o / (6084 / (api - 5.9)) * 0.178108
    return 1 + c_og / R_go


def _gas_inflow(w0, pvt, p_r, R_go) -> dict:  # spec: SMP-117
    """
    Forchheimer's a and b (INF-10) of a gas or condensate well at a reservoir state, in the model's units, as
    `forchheimer_a` (bar2 s/kg) and `forchheimer_b` (bar2 s2/kg2).

    The inflow in the pseudo-pressure, (mu_g Z)_r (m(p_r) - m(p_wf)) = a q + b q^2, has the radial coefficients at the
    state (SMP-73) with the well's geometry factors and non-Darcy coefficient, and mu_g Z of the reservoir gas at T_r:
    the black-oil gas at the wellstream's density for a wet or dry gas, the table's gas at R_go for a condensate. The
    pressure-squared form is fitted to it (_fit_pressure_squared), converted from the wellstream to the separator gas,
    a_q = c a_ws and b_q = c^2 b_ws, and to the gas's mass rate with the fluid's rho_g.

    c is 1 + C_og/R_go for the black-oil gas, whose standard volume is the wellstream's. A table's Z has the separator
    gas's gas constant (PVT-TAB-4), so with the table's gas the standard volume is the wellstream's mass over rho_g,
    and c is the mass ratio 1 + rho_o/(R_go rho_g).

    :param w0: The well's draws (draws.draw_well) that fix its inflow: its initial-stage draw
    :param pvt: The well's PVT model: its black oil, or a condensate's PVTTableSet
    :param p_r: Reservoir pressure (bar)
    :param R_go: Producing gas-oil ratio (Sm3/Sm3)
    """
    T_r = w0['T_r']
    if isinstance(pvt, (PVTTable, PVTTableSet)):
        gas = pvt.at_gor(R_go)
        c = 1 + gas.rho_o / (R_go * gas.rho_g)
    else:
        c = _wellstream_factor(w0['api'], R_go)
        rho_ws = (pvt.rho_g + pvt.rho_o / R_go) / c  # The wellstream as a gas at standard conditions
        gas = BlackOil(rho_o=pvt.rho_o, rho_g=rho_ws, gas_law='dak', p_sep=w0['p_sep'], T_sep=w0['T_sep'])
    p, m, muz_r = _pseudo_pressure(gas, p_r, T_r)
    Z, mu_g = float(gas.z_factor(p_r, T_r)), float(gas.gas_viscosity(p_r, T_r, gas.gas_density(p_r, T_r)))
    a, b = _forchheimer(w0['perm'], w0['h_net'], mu_g, Z, T_r, w0['ln_re_rw'] - 0.75, w0['skin'], w0['D_nondarcy'])
    a, b = a * w0['a_ratio'], b * w0['b_ratio']  # The geometry's factors

    def q_pseudo(p_wf):
        return _gas_rate(a, b, muz_r * (m[-1] - np.interp(p_wf, p, m)))

    a_ws, b_ws = _fit_pressure_squared(q_pseudo, p_r)
    inflow = Forchheimer.from_gas_rate(c * a_ws, c * c * b_ws, pvt.rho_g)
    return dict(forchheimer_a=inflow.a, forchheimer_b=inflow.b)
