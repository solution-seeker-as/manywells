"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests for manywells.configurations: the v1.0.0 and v2.0.0 configurations and the check that a well is in one.
"""

import dataclasses

import pytest

from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
from manywells.configurations import V1, V2, DEVELOP, check, differences, v1_fluid, v1_well
from manywells.geometry import WellGeometry
from manywells.inflow import FixedFlowRate, ProductivityIndex, Vogel
from manywells.pvt.black_oil import BlackOil
from manywells.pvt.dead_oil import DeadOil, dead_oil_surface_tension
from manywells.pvt.table import PVTTable
from manywells.simulator import WellProperties
from manywells.slip import SlipModel
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
    wp = dataclasses.replace(
        v1(),
        geometry=WellGeometry.from_survey([0, 1000, 2500], [0, 1000, 2200], n_cells=40),
        friction=WellProperties().friction,
        thermal=ThermalModel(h=25.0),
        inflow=FixedFlowRate(10.0),
    )
    with pytest.raises(ValueError) as e:
        check(wp, V1)
    message = str(e.value)
    for part in ('not vertical', 'friction', 'frictional_heating', 'gravity_term', 'lift_gas_mixing', 'inflow'):
        assert part in message


def test_check_rejects_fluid_options():
    fluid = v1().fluid
    fluid = dataclasses.replace(fluid, surface_tension_model='oil', pvt=dataclasses.replace(fluid.pvt, gas_law='dak'))
    diff = differences(dataclasses.replace(v1(), fluid=fluid), V1)
    assert any('surface_tension_model' in d for d in diff) and any('gas_law' in d for d in diff)
    diff = differences(dataclasses.replace(v1(), fluid=dataclasses.replace(fluid, pvt=BlackOil())), V1)
    assert any('pvt is BlackOil, not DeadOil' in d for d in diff)


def test_check_rejects_subclasses():
    """A subclass may change the physics, as a quadratic IPR in a ProductivityIndex would: each class is exact."""

    class QuadraticPI(ProductivityIndex):
        pass

    class MyThermal(ThermalModel):
        pass

    wp = dataclasses.replace(
        v1(),
        inflow=QuadraticPI(k_l=1.0),
        thermal=MyThermal(
            h=25.0, frictional_heating=False, gravity_term=False, lift_gas_mixing=False, joule_thomson=False
        ),
    )
    diff = differences(wp, V1)
    assert any('QuadraticPI, not Vogel or ProductivityIndex' in d for d in diff)
    assert any('thermal: MyThermal, not ThermalModel' in d for d in diff)
    with pytest.raises(ValueError):
        v1_well(**PARAMS, inflow=QuadraticPI(k_l=1.0), choke=SimpsonChokeModel(K_c=0.002), n_cells=40)
    assert differences(dataclasses.replace(WellProperties(), thermal=MyThermal()), DEVELOP) == [
        'thermal: MyThermal, not ThermalModel'
    ]


def test_check_rejects_an_injection_depth():
    """v1.0.0 injects the lift gas at the bottomhole (INF-5): any injection depth is a difference, even the
    bottomhole's; develop takes any."""
    for md in (1000.0, 2500.0):  # 2500 m is v1()'s bottomhole
        diff = differences(dataclasses.replace(v1(), injection_md=md), V1)
        assert diff == [f'injection_md is {md}, not None: v1.0.0 injects the lift gas at the bottomhole (INF-5)']
    assert differences(dataclasses.replace(WellProperties(), injection_md=1000.0), DEVELOP) == []


def v2():
    """develop's defaults with a choke: in v2.0.0 as long as develop's defaults are v2.0.0's."""
    return dataclasses.replace(WellProperties(), choke=SimpsonChokeModel(K_c=0.002))


def test_develop_defaults_are_v2():
    """develop's defaults are v2.0.0's on 2026-10-30. A default that changes takes develop out of v2.0.0, which this
    test reports: change the test, and the develop column of specs/model/README.md's table."""
    assert differences(v2(), V2) == []
    assert len(differences(v2(), V1)) > 0 and len(differences(v1(), V2)) > 0


def test_v2_takes_the_well_options():
    """v2.0.0 leaves the well's inputs free: the trajectory, the inflow and choke models, a PVT table instead of the
    black oil, an injection depth, the roughness and the profile scale (SLIP-18)."""
    table = PVTTable.read_csv('tests/data/pvt/synthetic_eos_oil.csv')
    for change in (
        dict(geometry=WellGeometry.from_survey([0, 1000, 2500], [0, 1000, 1500], n_cells=20)),
        dict(inflow=ProductivityIndex(k_l=0.5)),
        dict(inflow=FixedFlowRate(10.0)),
        dict(choke=BernoulliChokeModel(K_c=0.002, chk_profile='convex')),
        dict(fluid=dataclasses.replace(WellProperties().fluid, pvt=table, gor=table.gor)),
        dict(injection_md=1000.0),
        dict(friction=dataclasses.replace(WellProperties().friction, roughness=1e-4)),
        dict(slip=dataclasses.replace(WellProperties().slip, profile_scale=1.2)),
    ):
        assert differences(dataclasses.replace(v2(), **change), V2) == [], change


def test_v2_rejects_other_options():
    """v2.0.0 fixes the model's options and the slip law's constants, whatever develop's defaults become."""
    wp, fluid = v2(), v2().fluid
    for change, part in (
        (
            dict(fluid=dataclasses.replace(fluid, pvt=dataclasses.replace(fluid.pvt, gas_law='papay'))),
            "gas_law is 'papay'",
        ),
        (dict(fluid=dataclasses.replace(fluid, pvt=DeadOil())), 'pvt is DeadOil'),
        (dict(fluid=dataclasses.replace(fluid, surface_tension_model='liquid')), 'surface_tension_model'),
        (dict(friction=dataclasses.replace(wp.friction, correlation='haaland')), 'not RoughnessFriction with Chen'),
        (dict(thermal=dataclasses.replace(wp.thermal, joule_thomson=False)), 'joule_thomson is off'),
        (dict(slip=SlipModel(regime_model='fitted')), "not v2.0.0's five regimes"),
        (dict(slip=dataclasses.replace(wp.slip, C_0_slug=1.2)), "not v2.0.0's five regimes"),
    ):
        diff = differences(dataclasses.replace(wp, **change), V2)
        assert len(diff) == 1 and part in diff[0], (change, diff)


def test_unknown_configuration():
    with pytest.raises(ValueError, match='unknown configuration'):
        differences(v1(), 'v0.9')
