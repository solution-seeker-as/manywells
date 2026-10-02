// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The fluid: the one interface to the fluid properties that the rest of the core calls, as FluidModel in
//! src/manywells/pvt/fluid.py. It is given by the densities of oil, gas and water at standard conditions, the
//! gas-oil and water-liquid ratios and the heat capacities, from which it derives the gas's specific gravity and gas
//! constant, the liquid at standard conditions and the inflow's gas mass fraction, with the same arithmetic as
//! FluidModel. The gas is ideal or real (Dranchuk-Abou-Kassem or Papay); the oil is dead, or black oil into which
//! reservoir gas dissolves
//! (Vazquez-Beggs), mixed with incompressible water as one liquid.

use crate::pvt::oil::BlackOil;
use crate::pvt::{gas, mixture, oil, water};
use crate::smoothing::{max_approx, min_approx};
use crate::units::{CF_BAR, CF_RS, M_AIR, P_REF, R_UNIVERSAL, T_REF};

/// The density at which the dead-oil surface tension correlation is evaluated
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SurfaceTensionModel {
    /// The oil's at standard conditions, with the live-oil correction for black oil
    Oil,
    /// The local liquid density, as in v1.0.0
    Liquid,
}

/// The oil: dead, or black oil with gas dissolving into it
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OilModel {
    DeadOil,
    BlackOil(BlackOil),
}

/// The z-factor of a real gas, FluidModel.z_factor_model
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ZFactorModel {
    Dak,
    Papay,
}

/// The gas's compressibility factor
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GasLaw {
    /// Z = 1
    Ideal,
    /// Z from the Dranchuk-Abou-Kassem equation of state at the gas's density, with the pseudo-critical pressure
    /// (Pa) and temperature (K) of the gas
    Dak { ppc: f64, tpc: f64 },
    /// Z from Papay's correlation at the pressure, with the pseudo-critical pressure (Pa) and temperature (K)
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
    pub z_factor: ZFactorModel, // The real gas's z-factor, where ideal_gas is false
    pub black_oil: bool,
    pub p_sep: f64,             // Separator pressure (bar) and temperature (K) of the black-oil correlations
    pub t_sep: f64,
    pub p_bubble: Option<f64>,  // Bubble point pressure (bar), or none
    pub surface_tension: SurfaceTensionModel,
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
    pub oil: OilModel,
}

impl Fluid {
    pub fn new(inputs: FluidInputs) -> Self {
        let FluidInputs { rho_o, rho_g, rho_w, gor, wlr, cp_g, cp_o, cp_w, ideal_gas, z_factor, black_oil, p_sep, t_sep,
                          p_bubble, .. } = inputs;
        let sg_gas = rho_g * R_UNIVERSAL * T_REF / (P_REF * M_AIR); // spec: PVT-GAS-6
        let rho_l = wlr * rho_w + (1.0 - wlr) * rho_o;               // spec: PVT-MIX-10
        let api = oil::api_from_density(rho_o);
        Self {
            inputs,
            api,
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
                match z_factor {
                    ZFactorModel::Dak => GasLaw::Dak { ppc, tpc },
                    ZFactorModel::Papay => GasLaw::Papay { ppc, tpc },
                }
            },
            oil: if black_oil {
                OilModel::BlackOil(BlackOil::new(api, sg_gas, p_sep * CF_BAR, t_sep, p_bubble.map(|p_b| p_b * CF_BAR)))
            } else {
                OilModel::DeadOil
            },
        }
    }

    /// Solution gas-oil ratio (Sm³/Sm³) at p (bar) and T (K): none in a dead oil
    pub fn rs(&self, p: f64, t: f64) -> f64 {  // spec: PVT-OIL-4
        match self.oil {
            OilModel::DeadOil => 0.0,
            OilModel::BlackOil(b) => b.rs(p * CF_BAR, t),
        }
    }

    /// Oil formation volume factor at p (bar) and T (K): 1 for a dead oil
    pub fn bo(&self, p: f64, t: f64) -> f64 {  // spec: PVT-OIL-4
        match self.oil {
            OilModel::DeadOil => 1.0,
            OilModel::BlackOil(b) => b.bo(p * CF_BAR, t),
        }
    }

    /// The gas's compressibility factor at p (bar) and T (K): with DAK, at the density its gas law gives there
    pub fn z_factor(&self, p: f64, t: f64) -> f64 {
        match self.gas_law {
            GasLaw::Ideal => 1.0,
            GasLaw::Dak { ppc, tpc } => gas::dak_z_factor(gas::dak_reduced_density(p * CF_BAR / ppc, t / tpc), t / tpc),
            GasLaw::Papay { ppc, tpc } => gas::papay_z_factor(p * CF_BAR, t, ppc, tpc),
        }
    }

    /// Gas density (kg/m³) at p (bar) and T (K)
    pub fn gas_density(&self, p: f64, t: f64) -> f64 {
        match self.gas_law {
            GasLaw::Dak { ppc, tpc } => {  // spec: PVT-GAS-11
                gas::dak_reduced_density(p * CF_BAR / ppc, t / tpc) * ppc / (gas::DAK_ZC * self.r_s * tpc)
            }
            _ => gas::gas_density(p, t, self.z_factor(p, t), self.r_s),
        }
    }

    /// The gas-law row (bar), p - rho_g Z R_s T / c_bar; with DAK, Z is the equation of state's at rho_g and T
    pub fn gas_law_row(&self, p: f64, t: f64, rho_g: f64) -> f64 {
        let z = match self.gas_law {
            GasLaw::Dak { ppc, tpc } => {  // spec: PVT-GAS-11
                gas::dak_z_factor(gas::reduced_density(rho_g, self.r_s, ppc, tpc), t / tpc)
            }
            _ => self.z_factor(p, t),
        };
        gas::gas_law_row(p, t, rho_g, z, self.r_s)
    }

    /// The gas's Joule-Thomson factor J = T (d ln Z / dT)_p at T (K) and gas density rho_g (kg/m³): 0 for an ideal
    /// gas, and the DAK equation of state's for a real gas, whatever its z-factor
    pub fn jt_factor(&self, t: f64, rho_g: f64) -> f64 {
        match self.gas_law {
            GasLaw::Ideal => 0.0,
            GasLaw::Dak { ppc, tpc } | GasLaw::Papay { ppc, tpc } => {
                gas::dak_jt_factor(gas::reduced_density(rho_g, self.r_s, ppc, tpc), t / tpc) // spec: PVT-GAS-10
            }
        }
    }

    /// Liquid density (kg/m³) at p (bar) and T (K): the live oil's, with the dissolved gas and its formation volume
    /// factor, mixed with water. A dead oil's is the liquid's density at standard conditions, exactly.
    pub fn liquid_density(&self, p: f64, t: f64) -> f64 {  // spec: PVT-MIX-1, PVT-MIX-6, PVT-OIL-9
        let FluidInputs { rho_o, rho_g, rho_w, wlr, .. } = self.inputs;
        let rho_live_oil = (rho_o + self.rs(p, t) * rho_g) / self.bo(p, t);
        wlr * rho_w + (1.0 - wlr) * rho_live_oil
    }

    /// The liquid-density row (kg/m³)
    pub fn liquid_density_row(&self, p: f64, t: f64, rho_l_state: f64) -> f64 {
        mixture::liquid_density_row(rho_l_state, self.liquid_density(p, t))
    }

    /// Liquid viscosity (Pa s) at p (bar) and T (K): the dead oil's, corrected for the dissolved gas in black oil,
    /// mixed with water's by the water-liquid ratio
    pub fn liquid_viscosity(&self, p: f64, t: f64) -> f64 {  // spec: PVT-MIX-8
        let mut mu_o = oil::dead_oil_viscosity(self.api, t);
        if let OilModel::BlackOil(_) = self.oil {
            mu_o = oil::live_oil_viscosity(mu_o, self.rs(p, t) / CF_RS);
        }
        let wlr = self.inputs.wlr;
        wlr * water::water_viscosity(t) + (1.0 - wlr) * mu_o
    }

    /// Gas viscosity (Pa s) at T (K) and density rho_g (kg/m³)
    pub fn gas_viscosity(&self, t: f64, rho_g: f64) -> f64 {
        gas::gas_viscosity(t, rho_g, self.m_g)
    }

    /// Gas-liquid mixture viscosity (Pa s) at a point
    pub fn mixture_viscosity(&self, p: f64, t: f64, alpha: f64, rho_g: f64, rho_l: f64) -> f64 {
        mixture::mixture_viscosity(self.liquid_viscosity(p, t), self.gas_viscosity(t, rho_g), alpha, rho_l, rho_g)
    }

    /// Gas-liquid surface tension (J/m²) at p (bar), T (K) and the point's liquid density rho_l (kg/m³)
    pub fn surface_tension(&self, p: f64, t: f64, rho_l: f64) -> f64 {
        match self.inputs.surface_tension {
            SurfaceTensionModel::Liquid => mixture::liquid_surface_tension(rho_l, t),
            SurfaceTensionModel::Oil => {
                let sigma = oil::dead_oil_surface_tension(self.inputs.rho_o, t); // spec: PVT-MIX-7
                match self.oil {
                    OilModel::DeadOil => sigma,
                    OilModel::BlackOil(_) => oil::live_oil_surface_tension(sigma, self.rs(p, t) / CF_RS),
                }
            }
        }
    }

    /// Whether gas dissolves into the liquid, so that the phase rates vary along the well
    pub fn has_mass_transfer(&self) -> bool {
        matches!(self.oil, OilModel::BlackOil(_))
    }

    /// Gas mass rate from the reservoir (kg/s) for a reservoir liquid rate w_res (kg/s)
    pub fn reservoir_gas_rate(&self, w_res: f64) -> f64 {  // spec: INF-4
        (self.f_g / (1.0 - self.f_g)) * w_res
    }

    /// Gas and liquid mass rates (kg/s) at a point at pressure p (bar) and temperature t (K), for a reservoir liquid
    /// rate w_res and a lift gas rate w_lg (kg/s). Without mass transfer they are the same at every point; with black
    /// oil, reservoir gas (not lift gas) dissolves into the oil up to its solution gas-oil ratio at (p, T).
    pub fn phase_rates(&self, p: f64, t: f64, w_res: f64, w_lg: f64) -> (f64, f64) {
        let w_g_res = self.reservoir_gas_rate(w_res);
        if !self.has_mass_transfer() {
            return (w_g_res + w_lg, w_res); // spec: INF-5
        }
        let w_o = w_res * self.x_o;
        let w_d = min_approx(self.rs(p, t) * self.inputs.rho_g / self.inputs.rho_o * w_o, w_g_res, 1e-6);
        (max_approx(w_g_res + w_lg - w_d, 0.0, 1e-6), w_res + w_d) // spec: PVT-OIL-13
    }
}
