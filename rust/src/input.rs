// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The core's inputs: a well, built once from the Python dataclasses (src/manywells/solvers/rust.py), and an
//! operating point. Validation stays in the dataclasses; the core checks only what it needs to run.

use crate::choke::Choke;
use crate::friction::Friction;
use crate::geometry::Geometry;
use crate::inflow::Inflow;
use crate::pvt::fluid::Fluid;
use crate::slip::Slip;
use crate::thermal::Thermal;

/// A well: one part per model part, as WellProperties
#[derive(Clone, Debug)]
pub struct WellSpec {
    pub geometry: Geometry,
    pub fluid: Fluid,
    pub friction: Friction,
    pub thermal: Thermal,
    pub slip: Slip,
    pub inflow: Inflow,
    pub choke: Choke,
}

impl WellSpec {
    pub fn check(&self) -> Result<(), String> {
        if !(self.fluid.f_g > 0.0 && self.fluid.f_g < 1.0) {
            return Err("the gas mass fraction must be in (0, 1)".into());
        }
        Ok(())
    }

    /// Cross-section (m²)
    pub fn a(&self) -> f64 {
        self.geometry.a()
    }

    pub fn n_cells(&self) -> usize {
        self.geometry.n_cells()
    }

    /// Number of state values, 7 (N + 1)
    pub fn n_x(&self) -> usize {
        crate::discretization::DIM_X * (self.n_cells() + 1)
    }
}

/// The operating point: the boundary conditions and the controls, the parameters of the Python system
/// (discretization.PARAMS) in their order
#[derive(Clone, Copy, Debug)]
pub struct OperatingPoint {
    pub p_r: f64,  // Reservoir pressure (bar)
    pub p_s: f64,  // Separator pressure (bar)
    pub t_r: f64,  // Reservoir temperature (K)
    pub t_s: f64,  // Ambient temperature at the wellhead (K)
    pub t_lg: f64, // Lift gas temperature (K) at the bottomhole
    pub u: f64,    // Choke position in [0, 1]
    pub w_lg: f64, // Lift gas rate (kg/s), injected at the bottomhole
}

impl OperatingPoint {
    pub fn check(&self) -> Result<(), String> {
        if !(self.p_s > 0.0 && self.p_r > self.p_s) {
            return Err(format!("the reservoir pressure ({}) must exceed the separator pressure ({}) > 0", self.p_r,
                               self.p_s));
        }
        Ok(())
    }
}

/// Wells for the tests: W1 of specs/model/vectors/v1_rows.json (Vogel inflow, Simpson choke, sigmoid profile, lift
/// gas) and W2 (productivity index, Bernoulli choke, linear profile, choked at its root), on n cells
#[cfg(test)]
pub mod test_wells {
    use super::*;
    use crate::choke::{ChokeModel, Profile};
    use crate::geometry::tests::{survey, vertical};
    use crate::pvt::fluid::{FluidInputs, SurfaceTensionModel};
    use crate::units::{P_REF, T_REF};

    /// A fluid in the v1.0.0 configuration from v1.0.0's parameters, as configurations.v1_fluid: dead oil of the
    /// liquid's density and heat capacity, and the gas-oil ratio that gives the gas mass fraction f_g
    pub fn v1_fluid(rho_l: f64, r_s: f64, cp_g: f64, cp_l: f64, f_g: f64) -> Fluid {
        let rho_g = P_REF / (r_s * T_REF);
        let gor = f_g * rho_l / ((1.0 - f_g) * rho_g);
        Fluid::new(FluidInputs { rho_o: rho_l, rho_g, rho_w: 999.1, gor, wlr: 0.0, cp_g, cp_o: cp_l, cp_w: 4184.0,
                                 ideal_gas: true, black_oil: false, p_sep: P_REF / 1e5, t_sep: T_REF, p_bubble: None,
                                 surface_tension: SurfaceTensionModel::Liquid })
    }

    pub fn w1(n_cells: usize) -> (WellSpec, OperatingPoint) {
        let spec = WellSpec {
            geometry: vertical(2500.0, n_cells, 0.127),
            fluid: v1_fluid(900.0, 420.0, 2225.0, 3000.0, 0.15),
            friction: Friction::FixedFactor { f_d: 0.03 }, thermal: Thermal { h: 25.0, frictional_heating: false, gravity_term: false, lift_gas_mixing: false },
            slip: Slip::default(),
            inflow: Inflow::Vogel { w_l_max: 80.0 },
            choke: Choke::new(ChokeModel::Simpson, 0.0015201224372924933, Profile::Sigmoid),
        };
        (spec, OperatingPoint { p_r: 249.2, p_s: 30.0, t_r: 363.15, t_s: 277.15, t_lg: 363.15, u: 0.6, w_lg: 0.8 })
    }

    pub fn w2(n_cells: usize) -> (WellSpec, OperatingPoint) {
        let spec = WellSpec {
            geometry: vertical(1800.0, n_cells, 0.1524),
            fluid: v1_fluid(820.0, 500.0, 2225.0, 2200.0, 0.4),
            friction: Friction::FixedFactor { f_d: 0.05 }, thermal: Thermal { h: 15.0, frictional_heating: false, gravity_term: false, lift_gas_mixing: false },
            slip: Slip::default(),
            inflow: Inflow::ProductivityIndex { k_l: 0.6 },
            choke: Choke::new(ChokeModel::Bernoulli, 0.001824146924750992, Profile::Linear),
        };
        (spec, OperatingPoint { p_r: 150.0, p_s: 20.0, t_r: 345.0, t_s: 277.15, t_lg: 345.0, u: 0.8, w_lg: 0.0 })
    }

    /// W1 with frictional heating and the gravity term (009), so that the energy row depends on the pressure
    pub fn w1_thermal(n_cells: usize) -> (WellSpec, OperatingPoint) {
        let (mut spec, op) = w1(n_cells);
        spec.thermal = Thermal { h: 25.0, frictional_heating: true, gravity_term: true, lift_gas_mixing: false };
        (spec, op)
    }

    /// W1 with the energy terms, deviated: vertical to 750 m, then 45° to the same depth (001, 002, 009)
    pub fn w1_deviated(n_cells: usize) -> (WellSpec, OperatingPoint) {
        let (mut spec, op) = w1_thermal(n_cells);
        let md = 750.0 + 1750.0 * std::f64::consts::SQRT_2;
        spec.geometry = survey(&[0.0, 750.0, md], &[0.0, 750.0, 2500.0], n_cells, 0.127);
        (spec, op)
    }

    /// W2, L-shaped: vertical to 1800 m, then 600 m horizontal (001, 002)
    pub fn w2_l_shaped(n_cells: usize) -> (WellSpec, OperatingPoint) {
        let (mut spec, op) = w2(n_cells);
        spec.geometry = survey(&[0.0, 1800.0, 2400.0], &[0.0, 1800.0, 1800.0], n_cells, 0.1524);
        (spec, op)
    }

    /// W1 deviated with the energy terms, and a real gas (006)
    pub fn w1_real_gas(n_cells: usize) -> (WellSpec, OperatingPoint) {
        let (mut spec, op) = w1_deviated(n_cells);
        spec.fluid = Fluid::new(FluidInputs { ideal_gas: false, ..spec.fluid.inputs });
        (spec, op)
    }

    /// W1 deviated with the energy terms, a real gas, and black oil of API 25.7 with water and dissolved gas
    /// (007, 008)
    pub fn w1_black_oil(n_cells: usize) -> (WellSpec, OperatingPoint) {
        let (mut spec, op) = w1_real_gas(n_cells);
        spec.fluid = Fluid::new(FluidInputs { black_oil: true, wlr: 0.3, surface_tension: SurfaceTensionModel::Oil,
                                              ..spec.fluid.inputs });
        (spec, op)
    }

    /// W2 L-shaped with black oil of API 35 capped at a bubble point of 80 bar
    pub fn w2_bubble_point(n_cells: usize) -> (WellSpec, OperatingPoint) {
        let (mut spec, op) = w2_l_shaped(n_cells);
        let rho_o = 141.5 * 999.1 / (35.0 + 131.5);
        spec.fluid = Fluid::new(FluidInputs { black_oil: true, rho_o, p_bubble: Some(80.0), ..spec.fluid.inputs });
        (spec, op)
    }

    /// W1 deviated with develop's model: black oil, real gas, oil surface tension, the energy terms, and friction
    /// from roughness by Chen's correlation (004)
    pub fn w1_develop(n_cells: usize) -> (WellSpec, OperatingPoint) {
        let (mut spec, op) = w1_black_oil(n_cells);
        spec.friction = Friction::Roughness { roughness: 4.5e-5, correlation: crate::friction::Correlation::Chen };
        (spec, op)
    }

    /// W2 L-shaped at its bubble point, with friction from roughness by Haaland's correlation
    pub fn w2_haaland(n_cells: usize) -> (WellSpec, OperatingPoint) {
        let (mut spec, op) = w2_bubble_point(n_cells);
        spec.friction = Friction::Roughness { roughness: 1.5e-5, correlation: crate::friction::Correlation::Haaland };
        (spec, op)
    }

    /// W1 with develop's model and its lift gas (0.8 kg/s) injected at 300 K, mixing with the inflow (010)
    pub fn w1_cold_lift_gas(n_cells: usize) -> (WellSpec, OperatingPoint) {
        let (mut spec, mut op) = w1_develop(n_cells);
        spec.thermal.lift_gas_mixing = true;
        op.t_lg = 300.0;
        (spec, op)
    }

    /// W2 L-shaped at its bubble point with Haaland friction, at a fixed liquid rate of 25 kg/s (011)
    pub fn w2_fixed_rate(n_cells: usize) -> (WellSpec, OperatingPoint) {
        let (mut spec, op) = w2_haaland(n_cells);
        spec.inflow = Inflow::FixedRate { w_l: 25.0 };
        (spec, op)
    }

    /// Every test well
    pub fn all(n_cells: usize) -> Vec<(&'static str, WellSpec, OperatingPoint)> {
        let named = |name, (spec, op)| (name, spec, op);
        vec![named("w1", w1(n_cells)), named("w2", w2(n_cells)), named("w1_thermal", w1_thermal(n_cells)),
             named("w1_deviated", w1_deviated(n_cells)), named("w2_l_shaped", w2_l_shaped(n_cells)),
             named("w1_real_gas", w1_real_gas(n_cells)), named("w1_black_oil", w1_black_oil(n_cells)),
             named("w2_bubble_point", w2_bubble_point(n_cells)), named("w1_develop", w1_develop(n_cells)),
             named("w2_haaland", w2_haaland(n_cells)), named("w1_cold_lift_gas", w1_cold_lift_gas(n_cells)),
             named("w2_fixed_rate", w2_fixed_rate(n_cells))]
    }
}

#[cfg(test)]
mod tests {
    use super::test_wells::v1_fluid;
    use super::WellSpec;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn a_well_can_be_shared_between_threads() {
        assert_send_sync::<WellSpec>();  // The bindings lend it to the search with the GIL released
    }

    #[test]
    fn a_v1_fluid_gives_back_v1s_parameters() {
        let f = v1_fluid(900.0, 420.0, 2225.0, 3000.0, 0.15);
        assert_eq!(f.rho_l, 900.0);
        assert_eq!(f.cp_l, 3000.0);
        assert!((f.r_s - 420.0).abs() < 1e-12 * 420.0);
        assert!((f.f_g - 0.15).abs() < 1e-15);
        assert_eq!(f.x_o, 1.0);
    }
}
