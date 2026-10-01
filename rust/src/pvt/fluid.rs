// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The fluid: the one interface to the fluid properties that the rest of the core calls, as FluidModel in
//! src/manywells/pvt/fluid.py. It is given by the densities of oil, gas and water at standard conditions, the
//! gas-oil and water-liquid ratios and the heat capacities, from which it derives the gas's specific gravity and gas
//! constant, the liquid at standard conditions and the inflow's gas mass fraction, with the same arithmetic as
//! FluidModel. The gas is ideal or real (Papay); so far the liquid is a dead oil mixed with water, incompressible.

use crate::pvt::{gas, mixture, oil};
use crate::units::{CF_BAR, M_AIR, P_REF, R_UNIVERSAL, T_REF};

/// The gas's compressibility factor
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GasLaw {
    /// Z = 1
    Ideal,
    /// Z from Papay's correlation, with the pseudo-critical pressure (Pa) and temperature (K) of the gas
    Papay { ppc: f64, tpc: f64 },
}

/// The fluid's fields, as FluidModel's
#[derive(Clone, Copy, Debug)]
pub struct FluidInputs {
    pub rho_o: f64, // Oil density at standard conditions (kg/m³)
    pub rho_g: f64, // Gas density at standard conditions (kg/m³)
    pub rho_w: f64, // Water density at standard conditions (kg/m³)
    pub gor: f64,   // Gas-oil ratio (Sm³/Sm³)
    pub wlr: f64,   // Water-liquid ratio, in [0, 1)
    pub cp_g: f64,  // Heat capacities (J/(kg K)) of gas, oil and water
    pub cp_o: f64,
    pub cp_w: f64,
    pub ideal_gas: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Fluid {
    pub inputs: FluidInputs,
    pub api: f64,    // Oil API gravity
    pub sg_gas: f64, // Gas specific gravity relative to air
    pub m_g: f64,    // Gas molecular weight (kg/kmol)
    pub r_s: f64,    // Specific gas constant (J/(kg K))
    pub rho_l: f64,  // Liquid density at standard conditions (kg/m³)
    pub cp_g: f64,   // Gas heat capacity (J/(kg K))
    pub cp_l: f64,   // Liquid heat capacity (J/(kg K)), volume-weighted
    pub f_g: f64,    // Gas mass fraction of the reservoir inflow at standard conditions
    pub x_o: f64,    // Oil mass fraction of the liquid at standard conditions
    pub gas_law: GasLaw,
}

impl Fluid {
    pub fn new(inputs: FluidInputs) -> Self {
        let FluidInputs { rho_o, rho_g, rho_w, gor, wlr, cp_g, cp_o, cp_w, ideal_gas } = inputs;
        let sg_gas = rho_g * R_UNIVERSAL * T_REF / (P_REF * M_AIR); // spec: PVT-GAS-6
        let rho_l = wlr * rho_w + (1.0 - wlr) * rho_o;               // spec: PVT-MIX-10
        Self {
            inputs,
            api: oil::api_from_density(rho_o),
            sg_gas,
            m_g: M_AIR * sg_gas,                 // spec: PVT-GAS-6
            r_s: R_UNIVERSAL / (M_AIR * sg_gas), // spec: PVT-GAS-6
            rho_l,
            cp_g,
            cp_l: wlr * cp_w + (1.0 - wlr) * cp_o,                                       // spec: PVT-MIX-10
            f_g: rho_g * gor / (rho_g * gor + rho_o + rho_w * wlr / (1.0 - wlr)),        // spec: PVT-MIX-10
            x_o: if rho_l == 0.0 { 0.0 } else { (1.0 - wlr) * rho_o / rho_l },           // spec: PVT-MIX-10
            gas_law: if ideal_gas {
                GasLaw::Ideal
            } else {
                let (ppc, tpc) = gas::sutton_pseudo_critical(sg_gas);
                GasLaw::Papay { ppc, tpc }
            },
        }
    }

    /// The gas's compressibility factor at p (bar) and T (K)
    pub fn z_factor(&self, p: f64, t: f64) -> f64 {
        match self.gas_law {
            GasLaw::Ideal => 1.0,
            GasLaw::Papay { ppc, tpc } => gas::papay_z_factor(p * CF_BAR, t, ppc, tpc),
        }
    }

    pub fn gas_density(&self, p: f64, t: f64) -> f64 {
        gas::gas_density(p, t, self.z_factor(p, t), self.r_s)
    }

    pub fn gas_law_row(&self, p: f64, t: f64, rho_g: f64) -> f64 {
        gas::gas_law_row(p, t, rho_g, self.z_factor(p, t), self.r_s)
    }

    pub fn liquid_density(&self) -> f64 {
        self.rho_l
    }

    pub fn liquid_density_row(&self, rho_l_state: f64) -> f64 {
        mixture::constant_liquid_density_row(rho_l_state, self.rho_l)
    }

    pub fn surface_tension(&self, rho_l: f64, t: f64) -> f64 {
        mixture::liquid_surface_tension(rho_l, t)
    }

    /// Whether gas dissolves into the liquid, so that the phase rates vary along the well
    pub fn has_mass_transfer(&self) -> bool {
        false
    }

    /// Gas mass rate from the reservoir (kg/s) for a reservoir liquid rate w_res (kg/s)
    pub fn reservoir_gas_rate(&self, w_res: f64) -> f64 {  // spec: INF-4
        (self.f_g / (1.0 - self.f_g)) * w_res
    }

    /// Gas and liquid mass rates (kg/s) at a point at pressure p (bar) and temperature t (K), for a reservoir liquid
    /// rate w_res and a lift gas rate w_lg (kg/s). Without mass transfer they are the same at every point.
    pub fn phase_rates(&self, _p: f64, _t: f64, w_res: f64, w_lg: f64) -> (f64, f64) {  // spec: INF-5
        (self.reservoir_gas_rate(w_res) + w_lg, w_res)
    }
}
