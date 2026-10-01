// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 17 July 2026 (Rust port, rust_implementation), restructured 01 October 2026

//! Drift-flux closure and flow-regime classification (specs/model/slip.md), as SlipModel in src/manywells/slip.py:
//! v_g = C_0 v_m + v_inf, with C_0 and v_inf blended over three regimes (annular, slug/churn, bubbly) by a classifier,
//! at the inclination of the cell.

use crate::smoothing::softmax3;
use crate::units::STD_GRAVITY;

/// The profile parameter of each regime and the annular drift velocity (m/s), SlipModel's fields
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slip {
    pub c_0_annular: f64,
    pub c_0_slug: f64,
    pub c_0_bubbly: f64,
    pub v_inf_annular: f64,
}

impl Default for Slip {
    fn default() -> Self {
        Self { c_0_annular: 1.0, c_0_slug: 1.175, c_0_bubbly: 1.2, v_inf_annular: 0.0 }
    }
}

/// Bubble rise velocity (m/s), Harmathy, at gas-liquid surface tension sigma (J/m²)
pub fn harmathy_rise_velocity(rho_g: f64, rho_l: f64, sigma: f64) -> f64 {  // spec: SLIP-4
    1.53 * (STD_GRAVITY * sigma * (rho_l - rho_g) / (rho_l * rho_l)).powf(0.25)
}

/// Taylor-bubble rise velocity (m/s) in a pipe of inner diameter d (m)
pub fn taylor_rise_velocity(rho_g: f64, rho_l: f64, d: f64) -> f64 {  // spec: SLIP-5
    0.35 * (STD_GRAVITY * d * (1.0 - rho_g / rho_l)).sqrt()
}

/// The factor by which a cell's inclination scales the Taylor-bubble rise velocity, Eq. (A-10) of Hasan et al. (2010):
/// 1 in a vertical cell and 0 in a horizontal one
pub fn deviation_factor(cos_incl: f64) -> f64 {  // spec: SLIP-10
    let sin_incl = (1.0 - cos_incl * cos_incl).sqrt();
    cos_incl.sqrt() * (1.0 + sin_incl).powf(1.2)
}

/// The terms of the slip law at a point that do not depend on the void fraction: the classifier's two velocity
/// features, from the superficial velocities, and the rise velocities. With the phase mass fluxes fixed, as in the
/// march, the superficial velocities do not depend on alpha either, so these are computed once per pressure.
#[derive(Clone, Copy, Debug)]
pub struct SlipTerms {
    c_1: f64,
    c_3: f64,
    cos_incl: f64,
    v_inf_slug: f64,
    v_inf_bubbly: f64,
    slip: Slip,
}

impl SlipTerms {
    /// v_gs and v_ls are the superficial gas and liquid velocities (m/s), sigma the surface tension (J/m²), d the
    /// pipe's inner diameter (m) and cos_incl the cosine of the cell's inclination from vertical
    #[allow(clippy::too_many_arguments)]
    pub fn new(v_gs: f64, v_ls: f64, rho_g: f64, rho_l: f64, sigma: f64, d: f64, cos_incl: f64, slip: &Slip) -> Self {
        let annular_boundary = 3.1 * (STD_GRAVITY * sigma * (rho_l - rho_g) / (rho_g * rho_g)).powf(0.25);
        Self {
            c_1: (v_gs - annular_boundary).tanh(), // spec: SLIP-6
            c_3: (v_gs - 1.08 * v_ls).tanh(),      // spec: SLIP-6
            cos_incl,
            v_inf_slug: taylor_rise_velocity(rho_g, rho_l, d) * deviation_factor(cos_incl),
            v_inf_bubbly: harmathy_rise_velocity(rho_g, rho_l, sigma),
            slip: *slip,
        }
    }

    /// Regime probabilities [p_annular, p_slug, p_bubbly] at void fraction alpha
    pub fn probabilities(&self, alpha: f64) -> [f64; 3] {  // spec: SLIP-6, SLIP-7
        let c_2 = ((alpha - 0.7) * 2.0).tanh();
        let c_4 = ((alpha - 0.25 * self.cos_incl) * 2.0).tanh(); // spec: SLIP-11
        let (c_1, c_3) = (self.c_1, self.c_3);
        let y_1 = 3.17715258 * c_1 + 6.81938489 * c_2 + 0.30182974 * c_3 + 3.58362465 * c_4 - 3.92904391;
        let y_2 = -1.47973427 * c_1 - 4.34033317 * c_2 + 2.58200006 * c_3 + 3.49656911 * c_4 - 1.46509477;
        let y_3 = -1.6974183 * c_1 - 2.47905172 * c_2 - 2.8838298 * c_3 - 7.08019376 * c_4 + 5.39413869;
        softmax3([y_1, y_2, y_3])
    }

    /// The slip parameters (C_0, v_inf) at void fraction alpha
    pub fn parameters(&self, alpha: f64) -> (f64, f64) {  // spec: SLIP-2, SLIP-3
        let [p_annular, p_slug, p_bubbly] = self.probabilities(alpha);
        let s = &self.slip;
        let c_0 = p_annular * s.c_0_annular + p_slug * s.c_0_slug + p_bubbly * s.c_0_bubbly;
        let v_inf = p_annular * s.v_inf_annular + p_slug * self.v_inf_slug + p_bubbly * self.v_inf_bubbly;
        (c_0, v_inf)
    }
}

impl Slip {
    /// Regime probabilities [p_annular, p_slug, p_bubbly] at a point in a cell of inclination cos_incl
    #[allow(clippy::too_many_arguments)]
    pub fn classify(&self, v_g: f64, v_l: f64, alpha: f64, rho_g: f64, rho_l: f64, sigma: f64, d: f64,
                    cos_incl: f64) -> [f64; 3] {
        SlipTerms::new(alpha * v_g, (1.0 - alpha) * v_l, rho_g, rho_l, sigma, d, cos_incl, self).probabilities(alpha)
    }

    /// The slip parameters (C_0, v_inf) at a point in a cell of inclination cos_incl
    #[allow(clippy::too_many_arguments)]
    pub fn identify_parameters(&self, v_g: f64, v_l: f64, alpha: f64, rho_g: f64, rho_l: f64, sigma: f64, d: f64,
                               cos_incl: f64) -> (f64, f64) {
        SlipTerms::new(alpha * v_g, (1.0 - alpha) * v_l, rho_g, rho_l, sigma, d, cos_incl, self).parameters(alpha)
    }
}

/// The most probable regime; bubbly takes ties
pub fn regime_label(probs: [f64; 3]) -> &'static str {  // spec: SLIP-8
    let [p_annular, p_slug, p_bubbly] = probs;
    if p_annular > p_slug && p_annular > p_bubbly {
        "annular"
    } else if p_slug > p_annular && p_slug > p_bubbly {
        "slug-churn"
    } else {
        "bubbly"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIGMA: f64 = 0.02;

    #[test]
    fn probabilities_sum_to_one() {
        for cos_incl in [1.0, 0.5, 0.0] {
            let p = Slip::default().classify(20.0, 5.0, 0.5, 1.0, 900.0, SIGMA, 0.15, cos_incl);
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(p.iter().all(|&v| v > 0.0));
        }
    }

    #[test]
    fn parameters_are_blends_of_the_regime_values() {
        let s = Slip::default();
        for alpha in [0.0, 0.1, 0.3, 0.7, 0.95, 1.0] {
            for cos_incl in [1.0, 0.7, 0.0] {
                let (c_0, v_inf) = s.identify_parameters(5.0, 2.0, alpha, 50.0, 700.0, SIGMA, 0.15, cos_incl);
                assert!((s.c_0_annular - 1e-12..=s.c_0_bubbly + 1e-12).contains(&c_0));
                assert!(v_inf >= 0.0);
            }
        }
    }

    #[test]
    fn the_deviation_factor_is_one_vertical_and_zero_horizontal() {
        assert_eq!(deviation_factor(1.0), 1.0);
        assert_eq!(deviation_factor(0.0), 0.0);
    }

    #[test]
    fn label_is_a_regime() {
        let label = regime_label(Slip::default().classify(20.0, 2.0, 0.8, 1.0, 900.0, SIGMA, 0.15, 1.0));
        assert!(matches!(label, "annular" | "slug-churn" | "bubbly"));
        assert_eq!(regime_label([1.0 / 3.0; 3]), "bubbly");
    }
}
