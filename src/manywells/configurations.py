"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Named configurations of the model (specs/model/README.md): a choice of one option per model part.

The options live in the component objects of WellProperties, so a configuration is not an object the simulator
takes. 'v1.0.0' reproduces ManyWells v1.0.0, which generated the published datasets; it is the regression anchor
that the verifier checks against reference roots computed from v1.0.0 (specs/verification.md). 'develop' is the
dataclasses' defaults.

    wp = v1_well(L=2000, D=0.1554, rho_l=850, R_s=450, cp_g=2225, cp_l=2000, f_D=0.03, h=20, f_g=0.1,
                 inflow=Vogel(w_l_max=50), choke=SimpsonChokeModel(K_c=0.002), n_cells=100)
    check(wp, 'v1.0.0')   # raises ValueError, listing every option that differs
"""

from dataclasses import fields

import numpy as np

from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
from manywells.friction import FixedFrictionFactor, RoughnessFriction
from manywells.geometry import WellGeometry
from manywells.inflow import ProductivityIndex, Vogel
from manywells.pvt.fluid import FluidModel
from manywells.simulator import WellProperties
from manywells.slip import SlipModel
from manywells.thermal import ThermalModel
from manywells.units import P_REF, T_REF

V1 = 'v1.0.0'
DEVELOP = 'develop'
CONFIGURATIONS = (V1, DEVELOP)


def v1_fluid(rho_l, R_s, cp_g, cp_l, f_g) -> FluidModel:  # spec: SMP-40
    """
    A fluid in the v1.0.0 configuration from v1.0.0's parameters: dead oil of the liquid's density and heat capacity
    (no water, so the liquid is the oil), an ideal gas with specific gas constant R_s, and a gas-oil ratio that gives
    the gas mass fraction f_g of the reservoir inflow.

    :param rho_l: Liquid density (kg/m³)
    :param R_s: Specific gas constant (J/(kg K))
    :param cp_g: Gas heat capacity (J/(kg K))
    :param cp_l: Liquid heat capacity (J/(kg K))
    :param f_g: Gas mass fraction of the reservoir inflow, in (0, 1)
    """
    rho_g_sc = P_REF / (R_s * T_REF)  # Gas density at standard conditions (PVT-GAS-2)
    gor = f_g * rho_l / ((1 - f_g) * rho_g_sc)
    return FluidModel(rho_o=rho_l, rho_g=rho_g_sc, wlr=0.0, gor=gor, oil_model='dead_oil', ideal_gas=True,
                      surface_tension_model='liquid', cp_g=cp_g, cp_o=cp_l)


def v1_well(*, L, D, rho_l, R_s, cp_g, cp_l, f_D, h, f_g, inflow, choke, n_cells) -> WellProperties:
    """
    A well in the v1.0.0 configuration from v1.0.0's parameters, those of a verifier case (specs/verification.md).

    :param L: Pipe length (m), vertical
    :param D: Inner pipe diameter (m)
    :param rho_l: Liquid density (kg/m³)
    :param R_s: Specific gas constant (J/(kg K))
    :param cp_g: Gas heat capacity (J/(kg K))
    :param cp_l: Liquid heat capacity (J/(kg K))
    :param f_D: Darcy friction factor
    :param h: Overall heat transfer coefficient (W/(m² K))
    :param f_g: Gas mass fraction of the reservoir inflow
    :param inflow: Vogel or ProductivityIndex
    :param choke: SimpsonChokeModel or BernoulliChokeModel
    :param n_cells: Number of cells N
    """
    wp = WellProperties(geometry=WellGeometry.vertical(length=L, n_cells=n_cells, D=D),
                        fluid=v1_fluid(rho_l, R_s, cp_g, cp_l, f_g),
                        friction=FixedFrictionFactor(f_D=f_D),
                        thermal=ThermalModel(h=h, frictional_heating=False, gravity_term=False, lift_gas_mixing=False),
                        slip=SlipModel(), inflow=inflow, choke=choke)
    check(wp, V1)
    return wp


def _thermal_options(thermal) -> dict:
    return {f.name: getattr(thermal, f.name) for f in fields(ThermalModel) if f.type in (bool, 'bool')}


def differences(wp: WellProperties, configuration: str) -> list:
    """Every option of wp that differs from the configuration, as readable strings (empty if wp is in it)."""
    if configuration not in CONFIGURATIONS:
        raise ValueError(f'unknown configuration {configuration!r}, expected one of {CONFIGURATIONS}')
    geo, fluid, out = wp.geometry, wp.fluid, []
    thermal = _thermal_options(wp.thermal)
    if configuration == V1:
        if not np.allclose(geo.cos_incl, 1.0, rtol=0, atol=1e-12):
            out.append('geometry: not vertical (GEO-1)')
        if not np.allclose(geo.delta_md, geo.L / geo.n_cells, rtol=1e-9, atol=0):
            out.append('geometry: grid not uniform (DISC-1)')
        expected = {'oil_model': 'dead_oil', 'ideal_gas': True, 'surface_tension_model': 'liquid'}
        out += [f'fluid: {k} is {getattr(fluid, k)!r}, not {v!r}' for k, v in expected.items() if getattr(fluid, k) != v]
        if not isinstance(wp.friction, FixedFrictionFactor):
            out.append(f'friction: {type(wp.friction).__name__}, not FixedFrictionFactor (FRIC-2)')
        out += [f'thermal: {k} is on' for k, v in thermal.items() if v]
        if wp.slip != SlipModel():
            out.append(f'slip: parameters differ from v1.0.0 ({wp.slip})')
        if not isinstance(wp.inflow, (Vogel, ProductivityIndex)):
            out.append(f'inflow: {type(wp.inflow).__name__}, not Vogel or ProductivityIndex (INF-1, INF-2)')
        if not isinstance(wp.choke, (SimpsonChokeModel, BernoulliChokeModel)):
            out.append(f'choke: {type(wp.choke).__name__}, not SimpsonChokeModel or BernoulliChokeModel')
    else:
        default = FluidModel()
        out += [f'fluid: {k} is {getattr(fluid, k)!r}, not {getattr(default, k)!r}'
                for k in ('oil_model', 'ideal_gas', 'surface_tension_model') if getattr(fluid, k) != getattr(default, k)]
        if not isinstance(wp.friction, RoughnessFriction) or wp.friction.correlation != RoughnessFriction().correlation:
            out.append(f'friction: {wp.friction}, not RoughnessFriction with the default correlation')
        out += [f'thermal: {k} is off' for k, v in thermal.items() if not v]
    return out


def check(wp: WellProperties, configuration: str = V1):
    """Raise ValueError, listing every difference, unless wp is in the configuration."""
    diff = differences(wp, configuration)
    if diff:
        raise ValueError(f'well is not in the {configuration} configuration: ' + '; '.join(diff))
