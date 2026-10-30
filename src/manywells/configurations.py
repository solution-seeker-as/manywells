"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Named configurations of the model (specs/model/README.md): a choice of one option per model part.

The options live in the component objects of WellProperties, so a configuration is not an object the simulator
takes. 'v1.0.0' reproduces ManyWells v1.0.0, which generated the published datasets; it is the regression anchor
that the verifier checks against reference roots computed from v1.0.0 (specs/verification.md). 'v2.0.0' is the model
that v2's datasets are drawn in, pinned by the row vectors of specs/model/vectors/v2_rows.json. 'develop' is the
dataclasses' defaults, which may move on without moving either.

    wp = v1_well(L=2000, D=0.1554, rho_l=850, R_s=450, cp_g=2225, cp_l=2000, f_D=0.03, h=20, f_g=0.1,
                 inflow=Vogel(w_l_max=50), choke=SimpsonChokeModel(K_c=0.002), n_cells=100)
    check(wp, 'v1.0.0')   # raises ValueError, listing every option that differs
"""

from dataclasses import fields, replace

import numpy as np

from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
from manywells.friction import FixedFrictionFactor, RoughnessFriction
from manywells.geometry import WellGeometry
from manywells.inflow import FixedFlowRate, ProductivityIndex, Vogel
from manywells.pvt.black_oil import BlackOil
from manywells.pvt.dead_oil import DeadOil
from manywells.pvt.fluid import FluidModel
from manywells.pvt.table import PVTTable, PVTTableSet
from manywells.simulator import WellProperties
from manywells.slip import SlipModel
from manywells.thermal import ThermalModel
from manywells.units import P_REF, T_REF

V1 = 'v1.0.0'
V2 = 'v2.0.0'
DEVELOP = 'develop'
CONFIGURATIONS = (V1, V2, DEVELOP)

# The v2.0.0 configuration's slip law: the five regimes after Hasan et al. with their constants, written out so that a
# changed default does not change it. profile_scale (SLIP-18) is a well's parameter, which calibration fits.
V2_SLIP = SlipModel(
    regime_model='hasan',
    C_0_annular=1.0,
    C_0_slug=1.175,
    C_0_bubbly=1.2,
    v_inf_annular=0.0,
    f_ms=0.02,
    transition_width=0.02,
    lambda_floor=0.1,
)
V2_THERMAL = ('frictional_heating', 'gravity_term', 'lift_gas_mixing', 'joule_thomson')  # on; any other term off


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
    return FluidModel(
        pvt=DeadOil(rho_o=rho_l, rho_g=rho_g_sc, gas_law='ideal'),
        wlr=0.0,
        gor=gor,
        surface_tension_model='liquid',
        cp_g=cp_g,
        cp_o=cp_l,
    )


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
    wp = WellProperties(
        geometry=WellGeometry.vertical(length=L, n_cells=n_cells, D=D),
        fluid=v1_fluid(rho_l, R_s, cp_g, cp_l, f_g),
        friction=FixedFrictionFactor(f_D=f_D),
        thermal=ThermalModel(
            h=h, frictional_heating=False, gravity_term=False, lift_gas_mixing=False, joule_thomson=False
        ),
        slip=SlipModel(regime_model='fitted'),
        inflow=inflow,
        choke=choke,
        injection_md=None,
    )  # The lift gas enters at the bottomhole (INF-5)
    check(wp, V1)
    return wp


def _thermal_options(thermal) -> dict:
    return {f.name: getattr(thermal, f.name) for f in fields(ThermalModel) if f.type in (bool, 'bool')}


def _fluid_options(fluid, pvt_class, gas_law, surface_tension_model) -> list:
    """The fluid's options that differ from a configuration's: its PVT model's class and gas law, and its surface
    tension model."""
    out = []
    if type(fluid.pvt) is not pvt_class:
        out.append(f'fluid: pvt is {type(fluid.pvt).__name__}, not {pvt_class.__name__}')
    if getattr(fluid.pvt, 'gas_law', None) != gas_law:
        out.append(f'fluid: gas_law is {getattr(fluid.pvt, "gas_law", None)!r}, not {gas_law!r}')
    if fluid.surface_tension_model != surface_tension_model:
        out.append(f'fluid: surface_tension_model is {fluid.surface_tension_model!r}, not {surface_tension_model!r}')
    return out


def differences(wp: WellProperties, configuration: str) -> list:
    """
    Every option of wp that differs from the configuration, as readable strings (empty if wp is in it). Each part's
    class is checked exactly, as the Rust adapter does: a subclass may override any method, and so change the physics.
    """
    if configuration not in CONFIGURATIONS:
        raise ValueError(f'unknown configuration {configuration!r}, expected one of {CONFIGURATIONS}')
    geo, fluid = wp.geometry, wp.fluid
    out = [
        f'{name}: {type(getattr(wp, name)).__name__}, not {cls.__name__}'
        for name, cls in (
            ('geometry', WellGeometry),
            ('fluid', FluidModel),
            ('thermal', ThermalModel),
            ('slip', SlipModel),
        )
        if type(getattr(wp, name)) is not cls
    ]
    thermal = _thermal_options(wp.thermal)
    if configuration == V1:
        if not np.allclose(geo.cos_incl, 1.0, rtol=0, atol=1e-12):
            out.append('geometry: not vertical (GEO-1)')
        if not np.allclose(geo.delta_md, geo.L / geo.n_cells, rtol=1e-9, atol=0):
            out.append('geometry: grid not uniform (DISC-1)')
        out += _fluid_options(fluid, DeadOil, 'ideal', 'liquid')
        if type(wp.friction) is not FixedFrictionFactor:
            out.append(f'friction: {type(wp.friction).__name__}, not FixedFrictionFactor (FRIC-2)')
        out += [f'thermal: {k} is on' for k, v in thermal.items() if v]
        if wp.slip != SlipModel(regime_model='fitted'):
            out.append(f"slip: not v1.0.0's fitted classifier with its parameters ({wp.slip})")
        if type(wp.inflow) not in (Vogel, ProductivityIndex):
            out.append(f'inflow: {type(wp.inflow).__name__}, not Vogel or ProductivityIndex (INF-1, INF-2)')
        if type(wp.choke) not in (SimpsonChokeModel, BernoulliChokeModel):
            out.append(f'choke: {type(wp.choke).__name__}, not SimpsonChokeModel or BernoulliChokeModel')
        if wp.injection_md is not None:
            out.append(
                f'injection_md is {wp.injection_md}, not None: v1.0.0 injects the lift gas at the bottomhole (INF-5)'
            )
    elif configuration == V2:
        if type(fluid.pvt) not in (BlackOil, PVTTable, PVTTableSet):
            out.append(f'fluid: pvt is {type(fluid.pvt).__name__}, not BlackOil, PVTTable or PVTTableSet')
        elif type(fluid.pvt) is BlackOil and fluid.pvt.gas_law != 'dak':
            out.append(f"fluid: gas_law is {fluid.pvt.gas_law!r}, not 'dak'")
        if fluid.surface_tension_model != 'oil':
            out.append(f"fluid: surface_tension_model is {fluid.surface_tension_model!r}, not 'oil'")
        if type(wp.friction) is not RoughnessFriction or wp.friction.correlation != 'chen':
            out.append(f'friction: {wp.friction}, not RoughnessFriction with Chen (FRIC-4)')
        out += [f'thermal: {k} is off' for k, v in thermal.items() if k in V2_THERMAL and not v]
        out += [f'thermal: {k} is on, not in v2.0.0' for k, v in thermal.items() if k not in V2_THERMAL and v]
        if replace(wp.slip, profile_scale=V2_SLIP.profile_scale) != V2_SLIP:
            out.append(f"slip: not v2.0.0's five regimes with their constants ({wp.slip})")
        if type(wp.inflow) not in (Vogel, ProductivityIndex, FixedFlowRate):
            out.append(f'inflow: {type(wp.inflow).__name__}, not Vogel, ProductivityIndex or FixedFlowRate')
        if type(wp.choke) not in (SimpsonChokeModel, BernoulliChokeModel):
            out.append(f'choke: {type(wp.choke).__name__}, not SimpsonChokeModel or BernoulliChokeModel')
    else:
        default = FluidModel()
        out += _fluid_options(fluid, type(default.pvt), default.pvt.gas_law, default.surface_tension_model)
        if type(wp.friction) is not RoughnessFriction or wp.friction.correlation != RoughnessFriction().correlation:
            out.append(f'friction: {wp.friction}, not RoughnessFriction with the default correlation')
        out += [f'thermal: {k} is off' for k, v in thermal.items() if not v]
        if wp.slip.regime_model != SlipModel().regime_model:
            out.append(f'slip: regime_model is {wp.slip.regime_model!r}, not {SlipModel().regime_model!r}')
    return out


def check(wp: WellProperties, configuration: str = V1):
    """Raise ValueError, listing every difference, unless wp is in the configuration."""
    diff = differences(wp, configuration)
    if diff:
        raise ValueError(f'well is not in the {configuration} configuration: ' + '; '.join(diff))
