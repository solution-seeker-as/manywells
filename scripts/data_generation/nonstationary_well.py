"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 8 August 2024
Erlend Lundby, erlend@solutionseeker.no
"""

from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from manywells.closed_loop.cl_simulator import SimError
import manywells.pvt as pvt

from scripts.data_generation.well import Well, sample_well



class NonStationaryBehavior:
    def __init__(self, pr_init, ps_init,  init_fractions):
        self.lifetime = np.random.uniform(10,20)
        self.pr_init = pr_init
        self.ps_init = ps_init
        self.pr_conv = pr_init - np.random.uniform(pr_init*0.2, pr_init*0.4)
        self.eps: float = 0.01
        self.decay_rate: float = 1 - np.exp(np.log(self.eps)/self.lifetime)  # Equals 1 - self.eps ** (1/self.lifetime)
        self.decay_rate_noise_factor = self.decay_rate/20

        # Fractions
        self.init_fractions = init_fractions   # Tuple with initial mass fractions (gas, oil, water)

        self.decay_g, self.decay_o = self.decay_fractions()

    def reservoir_pressure(self, i):
        years = i/52  # Sampling rate 1/week
        decay_noise = np.random.uniform(-self.decay_rate_noise_factor,self.decay_rate_noise_factor)
        self.decay_rate += decay_noise
        self.decay_rate = np.maximum(0.1,np.minimum(0.9,self.decay_rate))
        return (self.pr_init - self.pr_conv)*(1 - self.decay_rate)**years + self.pr_conv


    def decay_fractions(self):
        decay_g = np.random.uniform(self.init_fractions[0]/2,self.init_fractions[0])
        decay_g = decay_g/self.lifetime
        decay_o = np.random.uniform(self.init_fractions[1]/2,self.init_fractions[1])
        decay_o = decay_o/self.lifetime
        return decay_g, decay_o

    def sample_gas_oil(self, fractions):
        f_g = 1
        f_o = 1
        while f_g + f_o > 0.999:
            f_g = np.minimum(0.99, np.maximum(fractions[0] - self.decay_g / 52 + np.random.normal(0, 0.015), 0.002))
            f_o = np.minimum(0.99, np.maximum(fractions[1] - self.decay_o / 52 + np.random.normal(0, 0.015), 0.002))
        return f_g, f_o

    def mass_fractions(self, fractions, delta_i):

        if delta_i <= 1:
            f_g, f_o =self.sample_gas_oil(fractions)
            f_w = 1 - f_g - f_o
        else:
            frac = fractions
            for i in range(delta_i):
                f_g, f_o = self.sample_gas_oil(frac)
                f_w = 1 - f_g - f_o
                frac = (f_g, f_o, f_w)

        return f_g, f_o, f_w


@dataclass
class NonStationaryWell(Well):

    ns_bhv: NonStationaryBehavior
    feedback: bool


    def copy(self):
        return deepcopy(self)

    def update_conditions(self, i: int, i_prev: int):
        delta_i = i - i_prev
        f_g, f_o, f_w = self.ns_bhv.mass_fractions(self.fractions, delta_i)
        if not np.isclose(f_g + f_o + f_w, 1.0):
            print('Sum of fractions:', f_g + f_o + f_w)
            raise SimError('ValueError: sum of fractions not equal 1')
        self.fractions = (f_g, f_o, f_w)
        liquid_mix = pvt.liquid_mix(self.oil, self.water, f_o / (f_o + f_w))  # Mix liquid
        self.wp.rho_l = liquid_mix.rho
        self.wp.cp_l = liquid_mix.cp
        self.wp.inflow.f_g = f_g
        self.bc.p_r = self.ns_bhv.reservoir_pressure(i)
        self.bc.p_s = np.random.uniform(0.9 * self.ns_bhv.ps_init, 1.1 * self.ns_bhv.ps_init)

        if not self.feedback:
            self.bc.u = np.random.uniform(0.05, 1)
            if self.has_gas_lift:
                self.bc.w_lg = np.random.uniform(0, 5)


def sample_nonstationary_well(feedback: bool) -> NonStationaryWell:
    """
    Sample a nonstationary well
    """

    # Sample stationary well first
    # This ensures that we reuse the distribution for the well properties
    # Higher concentration of wells with low water fraction and high gas fraction
    well = sample_well(alpha = (1.0, 1.0, 0.5))

    wp = well.wp
    bc = well.bc
    f_g, f_o, f_w = well.fractions

    # pr_convergence = bc.p_r - np.random.uniform(9,15)
    ns_bhv = NonStationaryBehavior(pr_init=bc.p_r, ps_init=bc.p_s, init_fractions=(f_g, f_o, f_w))

    # Create Well object
    nonstationary_well = NonStationaryWell(wp=wp, bc=bc, ns_bhv=ns_bhv, gas=well.gas, oil=well.oil, water=well.water,
                                           fractions=(f_g, f_o, f_w), has_gas_lift=well.has_gas_lift, feedback=feedback)

    return nonstationary_well

