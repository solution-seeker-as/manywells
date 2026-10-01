"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests for manywells.configurations: the v1.0.0 configuration and the check that a well is in it.
"""

import dataclasses

import pytest

from manywells.choke import SimpsonChokeModel
from manywells.configurations import V1, DEVELOP, check, differences, v1_fluid, v1_well
from manywells.geometry import WellGeometry
from manywells.inflow import FixedFlowRate, Vogel
from manywells.pvt.dead_oil import dead_oil_surface_tension
from manywells.simulator import WellProperties
from manywells.thermal import ThermalModel

PARAMS = dict(L=2500.0, D=0.1397, rho_l=880.0, R_s=420.0, cp_g=2225.0, cp_l=3100.0, f_D=0.035, h=25.0, f_g=0.2)


def v1():
    return v1_well(**PARAMS, inflow=Vogel(w_l_max=60.0), choke=SimpsonChokeModel(K_c=0.002), n_cells=40)


def test_v1_fluid_gives_back_v1_parameters():
    """SMP-40: the mapped fluid has v1.0.0's f_g, rho_l, c_pl and R_s, to rounding."""
    fl = v1_fluid(PARAMS['rho_l'], PARAMS['R_s'], PARAMS['cp_g'], PARAMS['cp_l'], PARAMS['f_g'])
    assert fl.f_g == pytest.approx(PARAMS['f_g'], rel=1e-14)
    assert fl.rho_l == PARAMS['rho_l']
    assert fl.cp_l == PARAMS['cp_l']
    assert fl.R_s == pytest.approx(PARAMS['R_s'], rel=1e-14)
    assert float(fl.liquid_density(150.0, 350.0)) == PARAMS['rho_l']  # constant (PVT-MIX-1), exactly
    assert float(fl.surface_tension(150.0, 350.0, 800.0)) == float(dead_oil_surface_tension(800.0, 350.0))


def test_v1_well_is_in_the_v1_configuration():
    wp = v1()
    assert differences(wp, V1) == []
    assert wp.geometry.n_cells == 40 and wp.geometry.L == PARAMS['L']
    assert len(differences(wp, DEVELOP)) > 0


def test_develop_defaults_are_the_develop_configuration():
    assert differences(WellProperties(), DEVELOP) == []


def test_check_lists_every_difference():
    wp = dataclasses.replace(v1(), geometry=WellGeometry.from_survey([0, 1000, 2500], [0, 1000, 2200], n_cells=40),
                             friction=WellProperties().friction,
                             thermal=ThermalModel(h=25.0), inflow=FixedFlowRate(10.0))
    with pytest.raises(ValueError) as e:
        check(wp, V1)
    message = str(e.value)
    for part in ('not vertical', 'friction', 'frictional_heating', 'gravity_term', 'lift_gas_mixing', 'inflow'):
        assert part in message


def test_check_rejects_fluid_options():
    wp = dataclasses.replace(v1(), fluid=dataclasses.replace(v1().fluid, surface_tension_model='oil', ideal_gas=False))
    diff = differences(wp, V1)
    assert any('surface_tension_model' in d for d in diff) and any('ideal_gas' in d for d in diff)


def test_unknown_configuration():
    with pytest.raises(ValueError, match="unknown configuration"):
        differences(v1(), 'v0.9')
