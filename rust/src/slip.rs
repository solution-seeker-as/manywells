// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 17 July 2026 (Rust port, rust_implementation), restructured 01 October 2026

//! Drift-flux closure and flow-regime classification (specs/model/slip.md), as SlipModel in src/manywells/slip.py:
//! v_g = C_0 v_m + v_inf, with C_0 and v_inf blended over the regimes of a classifier, at the inclination of the cell.
//! The classifier is v1.0.0's fitted one (three regimes) or the rules of Hasan et al.'s hierarchy (five regimes).

use crate::smoothing::softmax3;
use crate::units::STD_GRAVITY;

/// The classifier of the slip law: SlipModel.regime_model
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegimeModel {
    /// Three regimes from v1.0.0's fitted classifier (SLIP-2 to SLIP-8)
    Fitted,
    /// Five regimes from the rules of Hasan, Kabir and Sayarpour's (2010) hierarchy (SLIP-12 to SLIP-16)
    Hasan,
}

/// SlipModel's fields: the classifier; the fitted classifier's profile parameter of each regime and annular drift
/// velocity (m/s); the five-regime classifier's friction factor in (A-17), logistic width and floor of the no-slip
/// gas fraction; and either classifier's scale of C_0 - 1 (SLIP-18)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slip {
    pub regime_model: RegimeModel,
    pub c_0_annular: f64,
    pub c_0_slug: f64,
    pub c_0_bubbly: f64,
    pub v_inf_annular: f64,
    pub f_ms: f64,
    pub transition_width: f64,
    pub lambda_floor: f64,
    pub profile_scale: f64,
}

impl Default for Slip {
    fn default() -> Self {
        Self { regime_model: RegimeModel::Hasan, c_0_annular: 1.0, c_0_slug: 1.175, c_0_bubbly: 1.2, v_inf_annular: 0.0,
               f_ms: 0.02, transition_width: 0.02, lambda_floor: 0.1, profile_scale: 1.0 }
    }
}

/// The regimes of each classifier, in the order of its probabilities
pub const FITTED_REGIMES: [&str; 3] = ["annular", "slug-churn", "bubbly"];
pub const HASAN_REGIMES: [&str; 5] = ["annular", "churn", "dispersed-bubbly", "slug", "bubbly"];

/// delta of SLIP-12: bounds the derivative of sqrt(lambda_g) where there is no free gas
const SQRT_GUARD: f64 = 1e-6;

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

/// The quantities the five-regime classifier's rules compare, from the superficial velocities v_gs and v_ls (m/s):
/// [lambda_g, alpha_b, alpha_t, v_gc (m/s), v_ms (m/s)]
#[allow(clippy::too_many_arguments)]
pub fn hasan_rule_quantities(v_gs: f64, v_ls: f64, rho_g: f64, rho_l: f64, sigma: f64, d: f64, cos_incl: f64,
                             slip: &Slip) -> [f64; 5] {  // spec: SLIP-12
    let v_m = v_gs + v_ls;
    let v_inf_b = harmathy_rise_velocity(rho_g, rho_l, sigma);
    let lambda_g = v_gs / (v_m + slip.lambda_floor * v_inf_b);
    let alpha_b = v_gs / (1.2 * v_m + v_inf_b);
    let alpha_t = cos_incl / (2.8 + 1.2 * cos_incl);
    let v_gc = 3.1 * (STD_GRAVITY * sigma * (rho_l - rho_g) / (rho_g * rho_g)).powf(0.25);
    let k = 2.0 * (slip.f_ms / (2.0 * d)).powf(0.4) * (rho_l / sigma).powf(0.6)
        * (0.4 * sigma / (STD_GRAVITY * (rho_l - rho_g))).sqrt();
    let v_ms = ((0.725 + 4.15 * (lambda_g + SQRT_GUARD).sqrt()) / k).powf(1.0 / 1.2);
    [lambda_g, alpha_b, alpha_t, v_gc, v_ms]
}

/// The five regimes' probabilities [p_annular, p_churn, p_dispersed, p_slug, p_bubbly] from the rules of Hasan et
/// al.'s hierarchy as a soft decision tree (hasan_probabilities in slip.py)
#[allow(clippy::too_many_arguments)]
pub fn hasan_probabilities(v_gs: f64, v_ls: f64, rho_g: f64, rho_l: f64, sigma: f64, d: f64, cos_incl: f64,
                           slip: &Slip) -> [f64; 5] {  // spec: SLIP-13, SLIP-14
    let [lambda_g, alpha_b, alpha_t, v_gc, v_ms] = hasan_rule_quantities(v_gs, v_ls, rho_g, rho_l, sigma, d, cos_incl,
                                                                         slip);
    let margins = [v_gs / v_gc - 1.0, lambda_g - 0.7, (v_gs + v_ls) / v_ms - 1.0, lambda_g - 1.08 / 2.08,
                   alpha_b - alpha_t];
    let [s_a1, s_a2, s_m, s_g, s_s] = margins.map(|z| 0.5 * (1.0 + (z / (2.0 * slip.transition_width)).tanh()));
    let a = s_a1 * s_a2;
    [a, (1.0 - a) * s_m * s_g, (1.0 - a) * s_m * (1.0 - s_g), (1.0 - a) * (1.0 - s_m) * s_s,
     (1.0 - a) * (1.0 - s_m) * (1.0 - s_s)]
}

/// The terms of the slip law at a point that do not depend on the void fraction: the fitted classifier's two
/// velocity features, or the five regimes' probabilities, from the superficial velocities, and the rise velocities.
/// With the phase mass fluxes fixed, as in the march, the superficial velocities do not depend on alpha either, so
/// these are computed once per pressure.
#[derive(Clone, Copy, Debug)]
pub struct SlipTerms {
    c_1: f64,
    c_3: f64,
    hasan: [f64; 5],
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
        let (c_1, c_3, hasan) = match slip.regime_model {
            RegimeModel::Fitted => {
                let annular_boundary = 3.1 * (STD_GRAVITY * sigma * (rho_l - rho_g) / (rho_g * rho_g)).powf(0.25);
                ((v_gs - annular_boundary).tanh(), // spec: SLIP-6
                 (v_gs - 1.08 * v_ls).tanh(),      // spec: SLIP-6
                 [0.0; 5])
            }
            RegimeModel::Hasan => (0.0, 0.0, hasan_probabilities(v_gs, v_ls, rho_g, rho_l, sigma, d, cos_incl, slip)),
        };
        Self {
            c_1,
            c_3,
            hasan,
            cos_incl,
            v_inf_slug: taylor_rise_velocity(rho_g, rho_l, d) * deviation_factor(cos_incl),
            v_inf_bubbly: harmathy_rise_velocity(rho_g, rho_l, sigma),
            slip: *slip,
        }
    }

    /// The fitted classifier's probabilities [p_annular, p_slug, p_bubbly] at void fraction alpha
    fn fitted_probabilities(&self, alpha: f64) -> [f64; 3] {  // spec: SLIP-6, SLIP-7
        let c_2 = ((alpha - 0.7) * 2.0).tanh();
        let c_4 = ((alpha - 0.25 * self.cos_incl) * 2.0).tanh(); // spec: SLIP-11
        let (c_1, c_3) = (self.c_1, self.c_3);
        let y_1 = 3.17715258 * c_1 + 6.81938489 * c_2 + 0.30182974 * c_3 + 3.58362465 * c_4 - 3.92904391;
        let y_2 = -1.47973427 * c_1 - 4.34033317 * c_2 + 2.58200006 * c_3 + 3.49656911 * c_4 - 1.46509477;
        let y_3 = -1.6974183 * c_1 - 2.47905172 * c_2 - 2.8838298 * c_3 - 7.08019376 * c_4 + 5.39413869;
        softmax3([y_1, y_2, y_3])
    }

    /// Regime probabilities at void fraction alpha, in the order of the classifier's regimes (Slip::regimes)
    pub fn probabilities(&self, alpha: f64) -> Vec<f64> {
        match self.slip.regime_model {
            RegimeModel::Fitted => self.fitted_probabilities(alpha).to_vec(),
            RegimeModel::Hasan => self.hasan.to_vec(),
        }
    }

    /// The slip parameters (C_0, v_inf) at void fraction alpha
    pub fn parameters(&self, alpha: f64) -> (f64, f64) {
        match self.slip.regime_model {
            RegimeModel::Fitted => {
                let [p_annular, p_slug, p_bubbly] = self.fitted_probabilities(alpha);
                let s = &self.slip;
                let c_0 = p_annular * s.c_0_annular + p_slug * s.c_0_slug + p_bubbly * s.c_0_bubbly; // spec: SLIP-2
                let v_inf = p_annular * s.v_inf_annular + p_slug * self.v_inf_slug + p_bubbly * self.v_inf_bubbly; // spec: SLIP-3
                (self.slip.scaled_profile(c_0), v_inf)
            }
            RegimeModel::Hasan => {
                // Table 1 of Hasan et al. (2010) for upward cocurrent flow; independent of alpha
                let [p_annular, p_churn, p_dispersed, p_slug, p_bubbly] = self.hasan;
                let c_0 = 1.0 * p_annular + 1.15 * p_churn + 1.2 * (p_dispersed + p_slug + p_bubbly); // spec: SLIP-15
                let v_inf = self.v_inf_bubbly * (p_bubbly + p_dispersed) + self.v_inf_slug * (p_slug + p_churn); // spec: SLIP-15
                (self.slip.scaled_profile(c_0), v_inf)
            }
        }
    }
}

impl Slip {
    /// v1.0.0's fitted classifier with its constants, the slip model of the v1.0.0 configuration
    pub fn fitted() -> Self {
        Self { regime_model: RegimeModel::Fitted, ..Self::default() }
    }

    /// C_0 scaled about 1 by profile_scale; at the default, 1, C_0 as it is, to the bit
    pub fn scaled_profile(&self, c_0: f64) -> f64 {  // spec: SLIP-18
        if self.profile_scale == 1.0 { c_0 } else { 1.0 + self.profile_scale * (c_0 - 1.0) }
    }

    /// The classifier's regimes, in the order of its probabilities
    pub fn regimes(&self) -> &'static [&'static str] {
        match self.regime_model {
            RegimeModel::Fitted => &FITTED_REGIMES,
            RegimeModel::Hasan => &HASAN_REGIMES,
        }
    }

    /// Regime probabilities at a point in a cell of inclination cos_incl, in the order of regimes()
    #[allow(clippy::too_many_arguments)]
    pub fn classify(&self, v_g: f64, v_l: f64, alpha: f64, rho_g: f64, rho_l: f64, sigma: f64, d: f64,
                    cos_incl: f64) -> Vec<f64> {
        SlipTerms::new(alpha * v_g, (1.0 - alpha) * v_l, rho_g, rho_l, sigma, d, cos_incl, self).probabilities(alpha)
    }

    /// The slip parameters (C_0, v_inf) at a point in a cell of inclination cos_incl
    #[allow(clippy::too_many_arguments)]
    pub fn identify_parameters(&self, v_g: f64, v_l: f64, alpha: f64, rho_g: f64, rho_l: f64, sigma: f64, d: f64,
                               cos_incl: f64) -> (f64, f64) {
        SlipTerms::new(alpha * v_g, (1.0 - alpha) * v_l, rho_g, rho_l, sigma, d, cos_incl, self).parameters(alpha)
    }

    /// The label of the most probable regime, from classify()'s probabilities (SLIP-8 or SLIP-16)
    pub fn label(&self, probs: &[f64]) -> &'static str {
        match self.regime_model {
            RegimeModel::Fitted => regime_label([probs[0], probs[1], probs[2]]),
            RegimeModel::Hasan => hasan_regime_label(probs),
        }
    }
}

/// The most probable regime of the fitted classifier; bubbly takes ties
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

/// The most probable of the five regimes; a tie goes to the regime later in HASAN_REGIMES
pub fn hasan_regime_label(probs: &[f64]) -> &'static str {  // spec: SLIP-16
    let mut best = 0;
    for k in 1..HASAN_REGIMES.len() {
        if probs[k] >= probs[best] {
            best = k;
        }
    }
    HASAN_REGIMES[best]
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIGMA: f64 = 0.02;

    fn both() -> [Slip; 2] {
        [Slip::fitted(), Slip::default()]
    }

    #[test]
    fn probabilities_sum_to_one() {
        for s in both() {
            for cos_incl in [1.0, 0.5, 0.0] {
                let p = s.classify(20.0, 5.0, 0.5, 1.0, 900.0, SIGMA, 0.15, cos_incl);
                assert_eq!(p.len(), s.regimes().len());
                assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                assert!(p.iter().all(|&v| (0.0..=1.0).contains(&v)));
            }
        }
    }

    #[test]
    fn profile_scale_moves_c_0_towards_one_and_leaves_v_inf() {  // SLIP-18
        for s in both() {
            for alpha in [0.05, 0.3, 0.7, 0.95] {
                for cos_incl in [1.0, 0.5, 0.0] {
                    let args = (5.0, 2.0, alpha, 50.0, 700.0, SIGMA, 0.15, cos_incl);
                    let at = |k: f64| Slip { profile_scale: k, ..s }
                        .identify_parameters(args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7);
                    let (c_0, v_inf) = at(1.0);
                    assert_eq!(at(1.0), s.identify_parameters(args.0, args.1, args.2, args.3, args.4, args.5, args.6,
                                                              args.7));
                    assert_eq!(at(0.0).0, 1.0);
                    let mut last = 1.0;
                    for k in [0.0, 0.25, 0.5, 1.0, 1.5] {
                        let (c, v) = at(k);
                        assert!((c - (1.0 + k * (c_0 - 1.0))).abs() < 1e-15);
                        assert!(c >= last - 1e-15 && c >= 1.0);
                        assert_eq!(v, v_inf);
                        last = c;
                    }
                }
            }
        }
    }

    #[test]
    fn parameters_are_blends_of_the_regime_values() {
        for s in both() {
            for alpha in [0.0, 0.1, 0.3, 0.7, 0.95, 1.0] {
                for cos_incl in [1.0, 0.7, 0.0] {
                    let (c_0, v_inf) = s.identify_parameters(5.0, 2.0, alpha, 50.0, 700.0, SIGMA, 0.15, cos_incl);
                    assert!((s.c_0_annular - 1e-12..=s.c_0_bubbly + 1e-12).contains(&c_0));
                    assert!(v_inf >= 0.0);
                }
            }
        }
    }

    #[test]
    fn the_five_regimes_do_not_depend_on_the_void_fraction_at_fixed_superficial_velocities() {
        let s = Slip::default();
        let (v_gs, v_ls) = (1.5, 0.8);
        let at = |alpha: f64| s.identify_parameters(v_gs / alpha, v_ls / (1.0 - alpha), alpha, 60.0, 780.0, SIGMA,
                                                    0.127, 0.6);
        let (c_0, v_inf) = at(0.5);
        for alpha in [0.05, 0.3, 0.7, 0.95] {
            let (c, v) = at(alpha);
            assert!((c - c_0).abs() < 1e-12 && (v - v_inf).abs() < 1e-12);
        }
    }

    #[test]
    fn the_deviation_factor_is_one_vertical_and_zero_horizontal() {
        assert_eq!(deviation_factor(1.0), 1.0);
        assert_eq!(deviation_factor(0.0), 0.0);
    }

    #[test]
    fn label_is_a_regime() {
        for s in both() {
            let label = s.label(&s.classify(20.0, 2.0, 0.8, 1.0, 900.0, SIGMA, 0.15, 1.0));
            assert!(s.regimes().contains(&label));
        }
        assert_eq!(regime_label([1.0 / 3.0; 3]), "bubbly");
        assert_eq!(hasan_regime_label(&[0.2; 5]), "bubbly");
        assert_eq!(hasan_regime_label(&[0.4, 0.4, 0.1, 0.05, 0.05]), "churn");
    }
}
