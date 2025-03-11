"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 16 December 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no 

Methods to generate a random well
"""

import typing as ty
from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from manywells.simulator import WellProperties, BoundaryConditions, STD_GRAVITY, CF_PRES
from manywells.inflow import Vogel
from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
import manywells.pvt as pvt


@dataclass
class Well:
    wp: WellProperties
    bc: BoundaryConditions

    # Fluids are stored here since WellProperties only considers a liquid and gas phase
    gas: pvt.GasProperties          # Gas properties
    oil: pvt.LiquidProperties       # Oil properties
    water: pvt.LiquidProperties     # Water properties

    fractions: ty.Tuple[float, float, float]  # Tuple with (gas, oil, water) mass fraction (should sum to one)

    has_gas_lift: float  # Removed default value = False

    def copy(self):
        return deepcopy(self)

    def sample_new_conditions(self):

        # New conditions are stored in a new well object
        new_well = self.copy()

        # New choke opening
        new_well.bc.u = np.random.uniform(0.05, 1)

        # New gas lift setting
        if self.has_gas_lift:
            new_well.bc.w_lg = np.random.uniform(0, 5)  # Max 5 kg/s of lift gas

        # Update boundary pressures
        new_well.bc.p_s = np.random.uniform(0.9 * self.bc.p_s, 1.1 * self.bc.p_s)  # +/- 10% of initial pressure
        new_well.bc.p_r = np.random.uniform(0.98 * self.bc.p_r, 1.02 * self.bc.p_r)  # +/- 2% of initial pressure

        # Get initial fractions
        f_g, f_o, f_w = self.fractions

        # Sample new mass fractions
        new_f_g = np.random.uniform(0.95 * f_g, 1.05 * f_g)                 # +/- 5% of initial gas fraction
        new_f_g = min(0.99, new_f_g)
        wlf_init = f_w / (f_w + f_o)                                       # Initial water to liquid fraction
        new_wlf = np.random.uniform(0.95 * wlf_init, 1.05 * wlf_init)    # +/- 5% of initial water to liquid fraction
        new_wlf = min(1., new_wlf)
        new_f_w = (1 - new_f_g) * new_wlf
        new_f_o = 1 - new_f_g - new_f_w
        new_well.fractions = (new_f_g, new_f_o, new_f_w)
        # print('Fractions:', new_f_g, new_f_o, new_f_w, new_f_o / (new_f_o + new_f_w))

        # Update well properties using new fractions
        liquid_mix = pvt.liquid_mix(self.oil, self.water, new_f_o / (new_f_o + new_f_w))  # Mix liquid
        new_well.wp.rho_l = liquid_mix.rho
        new_well.wp.cp_l = liquid_mix.cp
        new_well.wp.inflow.f_g = new_f_g

        return new_well


def sample_well(alpha = (1.0, 1.0, 0.5)) -> Well:
    """
    Sample well

    :param alpha: Concentration parameters of Dirichlet distribution of fractions.
                  The default value, alpha = (1.0, 1.0, 0.5), puts more probability mass on gas and oil dominant wells.
    """

    # Create objects for well properties and boundary conditions
    wp = WellProperties()
    bc = BoundaryConditions()

    # Well depth
    wp.L = np.random.uniform(1500, 4500)

    # Well diameter
    # Common tubing diameters (we subtract 0.5 inch to get inner diameters)
    inch_to_meter = 0.0254  # Conversion factor from inch to meter
    outer_diameter = [3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]  # Outer diameter of pipe in inch
    well_diameters = [inch_to_meter * (od - 0.5) for od in outer_diameter]  # Inner diameter in meter
    wp.D = np.random.choice(well_diameters)  # Draw uniformly from list

    # Reservoir pressure
    p_0 = 1  # Pressure at L = 0 in bar (if L = 0 is above sea level, p_0 = 1 atm ~ 1 bar)
    rho_water = (pvt.SEAWATER.rho + pvt.WATER.rho) / 2
    bc.p_r = (1 / CF_PRES) * rho_water * STD_GRAVITY * wp.L + p_0  # Reservoir pressure in bar

    # Reservoir temperature
    T_0 = 273.15 + 60   # Reference reservoir temperature (K)
    L_0 = 1500          # Reference reservoir depth (m)
    dTdL = 0.03         # Temperature increase per meter (K / m)
    bc.T_r = T_0 + (wp.L - L_0) * dTdL  # Reservoir temperature as a function of depth

    # Downstream (surface) pressure
    # bc.p_s = np.random.uniform(10, 50)
    bc.p_s = np.random.lognormal(3, 1)
    if bc.p_s < 10 or bc.p_s > 120:
        raise ValueError('Downstream pressure not in [10, 120] - discarding well')

    # Heat transfer coefficient
    wp.h = np.random.uniform(10, 40)

    # Friction factor
    wp.f_D = np.random.uniform(0.01, 0.08)

    # Mass fractions (gas, oil, water)
    f_g, f_o, f_w = np.random.dirichlet(alpha=alpha)  # Draw gas, oil, and water mass fraction

    if f_g > 0.99:
        raise ValueError('Gas mass fraction > 0.99 - discarding well')

    # Fluids
    gas = pvt.GasProperties(
        name='gas',
        R_s=np.random.uniform(320, 520),   # Specific gas constant (520 is methane, 320 is EG gas)
        cp=pvt.METHANE.cp
    )

    oil = pvt.LiquidProperties(
        name='oil',
        rho=np.random.uniform(825, 925),  # API ranging from 40 (light) to 22 (intermediate-heavy)
        cp=pvt.NORTH_SEA_BRENT_CRUDE.cp
    )

    water = pvt.WATER

    liquid_mix = pvt.liquid_mix(oil, water, mass_fraction=f_o / (f_o + f_w))

    # Set fluid properties
    wp.rho_l = liquid_mix.rho
    wp.cp_l = liquid_mix.cp
    wp.R_s = gas.R_s
    wp.cp_g = gas.cp

    # Inflow model
    w_max = np.random.uniform(20, 200)  # Maximum mixture mass flow
    # f_g = np.random.uniform(0.0001, 0.99)   # Gas mass fraction
    w_l_max = (1 - f_g) * w_max         # Maximum liquid mass flow
    wp.inflow = Vogel(w_l_max, f_g)

    # Choke
    K_c_ref = 0.12 * wp.A  # Reference choke size
    K_c = np.random.uniform((1 / 2) * K_c_ref, 2 * K_c_ref)  # Sample choke size relative to reference size
    chk_profile = np.random.choice(wp.choke._chk_profiles)  # Pick choke characteristic uniformly
    wp.choke = SimpsonChokeModel(K_c=K_c, chk_profile=chk_profile)  # Init new choke model

    # On average, half of wells with f_g <= 0.1 have lift gas
    has_gas_lift = False
    if f_g <= 0.2 and np.random.choice([0, 1]) == 0:
        has_gas_lift = True

    # Create Well object
    well = Well(wp=wp, bc=bc, gas=gas, oil=oil, water=water, fractions=(f_g, f_o, f_w), has_gas_lift=has_gas_lift)

    return well

