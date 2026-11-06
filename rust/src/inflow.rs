// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 17 July 2026 (Rust port, rust_implementation), restructured 01 October 2026

//! The bottom boundary (specs/model/inflow.md): the reservoir liquid rate at the bottomhole pressure, or a fixed one.
//! The gas rate follows from the fluid (INF-4, pvt/fluid.rs). Forchheimer's gas inflow gives the gas rate instead
//! (INF-10), and the liquid that comes with it in the fluid's ratios (INF-11, pvt/fluid.rs).

use crate::pvt::fluid::Fluid;

#[derive(Clone, Copy, Debug)]
pub enum Inflow {
    /// Vogel's inflow performance relationship, with the liquid rate at zero bottomhole pressure (kg/s)
    Vogel { w_l_max: f64 },
    /// Linear inflow, with the liquid productivity index (kg/s/bar)
    ProductivityIndex { k_l: f64 },
    /// A fixed liquid rate (kg/s), whatever the bottomhole pressure
    FixedRate { w_l: f64 },
    /// Forchheimer's gas inflow in the pressure squared, with its Darcy coefficient a (bar² s/kg) and non-Darcy
    /// coefficient b (bar² s²/kg²)
    Forchheimer { a: f64, b: f64 },
}

/// The reservoir gas's mass rate at standard conditions (kg/s) of Forchheimer's law p_r² - p² = a w + b w², at
/// bottomhole pressure p (bar) and reservoir pressure p_r (bar): the root that is zero at p = p_r, and its odd
/// extension above p_r. p_r² - p² is factored, so that it keeps its precision near p_r.
pub fn forchheimer_gas_rate(a: f64, b: f64, p: f64, p_r: f64) -> f64 {
    let delta = (p_r - p) * (p_r + p);
    2.0 * delta / (a + (a * a + 4.0 * b * delta.abs()).sqrt()) // spec: INF-10
}

impl Inflow {
    /// Reservoir liquid mass rate (kg/s) at bottomhole pressure p (bar) and reservoir pressure p_r (bar); only the gas
    /// inflow uses the fluid, whose liquid comes with the gas
    pub fn liquid_rate(&self, p: f64, p_r: f64, fluid: &Fluid) -> f64 {
        match *self {
            Inflow::Vogel { w_l_max } => {
                let r = p / p_r;
                w_l_max * (1.0 - 0.2 * r - 0.8 * (r * r)) // spec: INF-1
            }
            Inflow::ProductivityIndex { k_l } => k_l * (p_r - p), // spec: INF-2
            Inflow::FixedRate { w_l } => w_l,                     // spec: INF-8
            Inflow::Forchheimer { a, b } => fluid.reservoir_liquid_rate(forchheimer_gas_rate(a, b, p, p_r)), // spec: INF-11
        }
    }

    /// The coefficients' checks of the Python classes: Forchheimer's a positive and b non-negative, both finite
    pub fn check(&self) -> Result<(), String> {
        if let Inflow::Forchheimer { a, b } = *self {
            if !(a.is_finite() && a > 0.0) {
                return Err(format!("Forchheimer: a must be positive and finite, got {a}"));
            }
            if !(b.is_finite() && b >= 0.0) {
                return Err(format!("Forchheimer: b must be non-negative and finite, got {b}"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::test_wells::v1_fluid;

    /// A fluid whose reservoir inflow has the gas mass fraction f_g
    fn fluid(f_g: f64) -> Fluid {
        v1_fluid(850.0, 420.0, 2225.0, 2000.0, f_g)
    }

    #[test]
    fn no_flow_at_reservoir_pressure() {
        assert_eq!(Inflow::Vogel { w_l_max: 30.0 }.liquid_rate(170.0, 170.0, &fluid(0.2)), 0.0);
        assert_eq!(Inflow::ProductivityIndex { k_l: 0.5 }.liquid_rate(170.0, 170.0, &fluid(0.2)), 0.0);
        assert_eq!(Inflow::Forchheimer { a: 2000.0, b: 50.0 }.liquid_rate(170.0, 170.0, &fluid(0.2)), 0.0);
    }

    #[test]
    fn a_fixed_rate_does_not_depend_on_the_pressure() {
        assert_eq!(Inflow::FixedRate { w_l: 12.0 }.liquid_rate(170.0, 170.0, &fluid(0.2)), 12.0);
    }

    #[test]
    fn vogel_gives_its_maximum_at_zero_pressure() {
        assert_eq!(Inflow::Vogel { w_l_max: 30.0 }.liquid_rate(0.0, 170.0, &fluid(0.2)), 30.0);
    }

    #[test]
    fn forchheimer_solves_its_law_and_is_odd_about_the_reservoir_pressure() {
        let (a, b, p_r) = (2000.0, 50.0, 170.0);
        for p in [1.01325, 60.0, 150.0, 169.9] {
            let w = forchheimer_gas_rate(a, b, p, p_r);
            let delta = p_r * p_r - p * p;
            assert!(w > 0.0 && (a * w + b * w * w - delta).abs() < 1e-10 * delta, "p = {p}");
            // Above p_r, the odd extension: the rate at the same |Δ|, negative
            let p_above = (2.0 * p_r * p_r - p * p).sqrt();
            assert!((forchheimer_gas_rate(a, b, p_above, p_r) + w).abs() < 1e-9 * w, "p = {p}");
        }
        // The linear limit, b = 0
        assert_eq!(forchheimer_gas_rate(a, 0.0, 100.0, p_r), (p_r * p_r - 100.0 * 100.0) / a);
    }

    #[test]
    fn forchheimer_gives_the_liquid_in_the_fluids_ratio() {
        let (a, b, p, p_r, fluid) = (2000.0, 50.0, 120.0, 170.0, fluid(0.8));
        let w_g = forchheimer_gas_rate(a, b, p, p_r);
        let w_l = Inflow::Forchheimer { a, b }.liquid_rate(p, p_r, &fluid);
        assert!((fluid.reservoir_gas_rate(w_l) - w_g).abs() < 1e-14 * w_g); // INF-4 gives back the gas rate
    }

    #[test]
    fn forchheimer_refuses_its_coefficients_out_of_range() {
        assert!(Inflow::Forchheimer { a: 2000.0, b: 0.0 }.check().is_ok());
        for (a, b) in [(0.0, 1.0), (-1.0, 1.0), (f64::NAN, 1.0), (f64::INFINITY, 1.0), (1.0, -1.0), (1.0, f64::NAN),
                       (1.0, f64::INFINITY)] {
            assert!(Inflow::Forchheimer { a, b }.check().is_err(), "a = {a}, b = {b}");
        }
        assert!(Inflow::Vogel { w_l_max: 30.0 }.check().is_ok());
    }
}
