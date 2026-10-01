"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests for manywells.thermal: the terms of the temperature gradient and the inflow temperature.
"""

import dataclasses

import pytest

from manywells.discretization import PointState
from manywells.pvt.fluid import FluidModel
from manywells.thermal import ThermalModel
from manywells.units import STD_GRAVITY

FLUID = FluidModel(oil_model='dead_oil', cp_g=2225.0, cp_o=4180.0)
V1 = ThermalModel(h=20.0, frictional_heating=False, gravity_term=False, lift_gas_mixing=False)
D, T_A = 0.15, 330.0


def state(alpha, T=350.0, rho_g=50.0, v_g=10.0, rho_l=850.0, v_l=2.0):
    return PointState(p=100.0, v_g=v_g, v_l=v_l, alpha=alpha, rho_g=rho_g, rho_l=rho_l, T=T)


def cp_flux(s):
    return FLUID.cp_g * s.alpha * s.rho_g * s.v_g + FLUID.cp_l * (1 - s.alpha) * s.rho_l * s.v_l


def gradient(model, s, F=0.0, cos_incl=1.0):
    return model.temperature_gradient(s, FLUID, T_A, F, dp_dmd=0.0, cos_incl=cos_incl, D=D)


def test_frozen_and_validated():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ThermalModel().h = 30.0
    with pytest.raises(ValueError, match="Heat transfer"):
        ThermalModel(h=-1.0)


def test_heat_loss_only():
    """With every option off, dT/dMD is minus the heat loss of THM-1."""
    s = state(0.4)
    assert gradient(V1, s, F=500.0) == pytest.approx(-4 * V1.h * (s.T - T_A) / (D * cp_flux(s)), rel=1e-15)


def test_no_heat_loss_at_ambient_temperature():
    assert gradient(V1, state(0.4, T=T_A)) == 0.0


def test_gravity_term_of_pure_gas_is_the_lapse_rate():
    """For pure gas (alpha=1), gravitational cooling rate equals g cos(theta) / cp_g."""
    m = ThermalModel(h=0.0, frictional_heating=False, gravity_term=True)
    for cos_incl in (1.0, 0.5):
        assert gradient(m, state(1.0), cos_incl=cos_incl) == pytest.approx(-STD_GRAVITY * cos_incl / FLUID.cp_g, rel=1e-12)


def test_gravity_term_of_pure_liquid_vanishes():
    m = ThermalModel(h=0.0, frictional_heating=False, gravity_term=True)
    assert gradient(m, state(0.0)) == pytest.approx(0.0, abs=1e-15)


def test_gravity_term_cools_a_mixture():
    m = ThermalModel(h=0.0, frictional_heating=False, gravity_term=True)
    assert gradient(m, state(0.5)) < 0


def test_frictional_heating_of_pure_liquid():
    """For pure liquid (alpha=0), friction heats the flow by F / (rho_l cp_l)."""
    m = ThermalModel(h=0.0, frictional_heating=True, gravity_term=False)
    s, F = state(0.0), 500.0
    assert gradient(m, s, F=F) == pytest.approx(F / (s.rho_l * FLUID.cp_l), rel=1e-12)


def test_frictional_heating_of_pure_gas_vanishes():
    m = ThermalModel(h=0.0, frictional_heating=True, gravity_term=False)
    assert gradient(m, state(1.0), F=500.0) == pytest.approx(0.0, abs=1e-15)


def test_ambient_temperature_is_linear_in_true_vertical_depth():
    assert ThermalModel.ambient_temperature(1.0, 370.0, 277.0) == 370.0
    assert ThermalModel.ambient_temperature(0.0, 370.0, 277.0) == 277.0
    assert ThermalModel.ambient_temperature(0.25, 370.0, 277.0) == pytest.approx(277.0 + 0.25 * 93.0)


def test_inflow_temperature():
    """Lift gas colder than the reservoir fluid lowers the inflow temperature, but not below T_lg."""
    m, T_r, T_lg, w_res = ThermalModel(), 373.15, 300.0, 10.0
    assert V1.inflow_temperature(w_res, 1.0, T_r, T_lg, FLUID) == T_r            # THM-3: the lift gas is ignored
    assert m.inflow_temperature(w_res, 1.0, T_r, T_r, FLUID) == T_r             # T_lg = T_r: exactly T_r
    assert m.inflow_temperature(w_res, 0.0, T_r, T_lg, FLUID) == T_r            # no lift gas: exactly T_r
    T_mix = m.inflow_temperature(w_res, 1.0, T_r, T_lg, FLUID)
    assert T_lg < T_mix < T_r
    assert m.inflow_temperature(w_res, 3.0, T_r, T_lg, FLUID) < T_mix          # more lift gas, colder
    H_res = w_res * FLUID.cp_l + FLUID.reservoir_gas_rate(w_res) * FLUID.cp_g
    assert T_mix == pytest.approx((H_res * T_r + 1.0 * FLUID.cp_g * T_lg) / (H_res + FLUID.cp_g), rel=1e-15)
