// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The gas phase (specs/model/pvt/gas.md), as src/manywells/pvt/gas.py: the real-gas law with a compressibility
//! factor Z, which is 1 for an ideal gas, and for a real one from the Dranchuk-Abou-Kassem equation of state or
//! Papay's correlation; and the gas's Joule-Thomson factor.

use crate::units::{CF_BAR, CF_PSI};

/// Gas density (kg/m³) at p (bar) and T (K), with compressibility factor z and specific gas constant r_s (J/(kg K))
pub fn gas_density(p: f64, t: f64, z: f64, r_s: f64) -> f64 {  // spec: PVT-GAS-1, PVT-GAS-3
    CF_BAR * p / (z * r_s * t)
}

/// The gas-law row (bar), p - rho_g Z R_s T / c_bar, zero where rho_g is the density at p and T
pub fn gas_law_row(p: f64, t: f64, rho_g: f64, z: f64, r_s: f64) -> f64 {  // spec: PVT-GAS-1, PVT-GAS-3
    p - rho_g * z * r_s * t / CF_BAR
}

/// Ideal-gas density (kg/m³) at p (bar) and T (K), with specific gas constant R_s (J/(kg K))
pub fn ideal_gas_density(p: f64, t: f64, r_s: f64) -> f64 {  // spec: PVT-GAS-1
    gas_density(p, t, 1.0, r_s)
}

/// Pseudo-critical pressure (Pa) and temperature (K) of a gas of specific gravity sg_gas, Sutton (1985)
pub fn sutton_pseudo_critical(sg_gas: f64) -> (f64, f64) {  // spec: PVT-GAS-5
    let ppc_psia = 756.8 - 131.07 * sg_gas - 3.6 * (sg_gas * sg_gas);
    let tpc_r = 169.2 + 349.5 * sg_gas - 74.0 * (sg_gas * sg_gas);
    (ppc_psia * CF_PSI, tpc_r / 1.8)
}

/// Compressibility factor at pressure p_pa (Pa) and temperature t (K), by Papay's (1968) correlation, for the
/// pseudo-critical pressure ppc (Pa) and temperature tpc (K)
pub fn papay_z_factor(p_pa: f64, t: f64, ppc: f64, tpc: f64) -> f64 {  // spec: PVT-GAS-4
    let ppr = p_pa / ppc;
    let tpr = t / tpc;
    1.0 - 3.52 * ppr * 10f64.powf(-0.9813 * tpr) + 0.274 * (ppr * ppr) * 10f64.powf(-0.8157 * tpr)
}

/// The coefficients A_1 to A_11 of the Dranchuk-Abou-Kassem equation of state (1975, Eq. 2)
const DAK_A: [f64; 11] = [0.3265, -1.0700, -0.5339, 0.01569, -0.05165, 0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210];

/// The critical compressibility factor of DAK's reduced density (1975, Eq. 3)
pub const DAK_ZC: f64 = 0.27;

/// Most Newton steps of dak_reduced_density; from the ideal-gas density it converges in at most 17 for
/// 1.05 <= T_pr <= 3 and p_pr <= 30 (specs/features/016-joule-thomson.md)
const DAK_MAXITER: usize = 50;

/// Relative step at which dak_reduced_density has converged
const DAK_RTOL: f64 = 1e-13;

/// DAK's Z, dZ/dr and t dZ/dt at reduced density r and pseudo-reduced temperature t
fn dak_terms(r: f64, t: f64) -> (f64, f64, f64) {
    let [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11] = DAK_A;
    let c1 = a1 + a2 / t + a3 / t.powi(3) + a4 / t.powi(4) + a5 / t.powi(5);
    let c2 = a6 + a7 / t + a8 / (t * t);
    let c3 = a9 * (a7 / t + a8 / (t * t));
    let c4 = a10 / t.powi(3);
    let e = (-a11 * r * r).exp();
    let z = 1.0 + c1 * r + c2 * r * r - c3 * r.powi(5) + c4 * r * r * (1.0 + a11 * r * r) * e;
    let z_r = c1 + 2.0 * c2 * r - 5.0 * c3 * r.powi(4) + 2.0 * c4 * r * e * (1.0 + a11 * r * r - a11 * a11 * r.powi(4));
    let tz_t = (-a2 / t - 3.0 * a3 / t.powi(3) - 4.0 * a4 / t.powi(4) - 5.0 * a5 / t.powi(5)) * r
        + (-a7 / t - 2.0 * a8 / (t * t)) * r * r
        - a9 * (-a7 / t - 2.0 * a8 / (t * t)) * r.powi(5)
        - 3.0 * a10 / t.powi(3) * r * r * (1.0 + a11 * r * r) * e;
    (z, z_r, tz_t)
}

/// Compressibility factor at reduced density r and pseudo-reduced temperature t, from the Dranchuk-Abou-Kassem
/// (1975) equation of state
pub fn dak_z_factor(r: f64, t: f64) -> f64 {  // spec: PVT-GAS-9
    dak_terms(r, t).0
}

/// The gas's Joule-Thomson factor J = T (d ln Z / dT)_p of the DAK equation of state at reduced density r and
/// pseudo-reduced temperature t: (t Z_t - r Z_r) / (Z + r Z_r)
pub fn dak_jt_factor(r: f64, t: f64) -> f64 {  // spec: PVT-GAS-10
    let (z, z_r, tz_t) = dak_terms(r, t);
    (tz_t - r * z_r) / (z + r * z_r)
}

/// DAK's reduced density Z_c rho_g R_s T_pc / p_pc of gas density rho_g (kg/m³), for the specific gas constant r_s
/// (J/(kg K)) and the pseudo-critical pressure ppc (Pa) and temperature tpc (K)
pub fn reduced_density(rho_g: f64, r_s: f64, ppc: f64, tpc: f64) -> f64 {  // spec: PVT-GAS-9
    DAK_ZC * rho_g * r_s * tpc / ppc
}

/// DAK's reduced density at pseudo-reduced pressure ppr and temperature t: the root of r t Z(r, t) / Z_c = p_pr, by
/// Newton's method from the ideal-gas density Z_c p_pr / t. NaN if it does not converge, which happens only outside
/// the equation's range (T_pr < 1.05, near the critical point).
pub fn dak_reduced_density(ppr: f64, t: f64) -> f64 {  // spec: PVT-GAS-11
    let mut r = DAK_ZC * ppr / t;
    for _ in 0..DAK_MAXITER {
        let (z, z_r, _) = dak_terms(r, t);
        let step = (r * t * z - DAK_ZC * ppr) / (t * (z + r * z_r));
        r -= step;
        if step.abs() <= DAK_RTOL * r.abs() {
            return r;
        }
    }
    f64::NAN
}

/// Gas viscosity (Pa s) at T (K) and density rho_g (kg/m³), for molecular weight m_g (kg/kmol), by the
/// Lee-Gonzalez-Eakin (1966) correlation
pub fn gas_viscosity(t: f64, rho_g: f64, m_g: f64) -> f64 {  // spec: PVT-GAS-7
    let t_r = 1.8 * t;
    let rho_gcc = rho_g * 1e-3;
    let k = (9.4 + 0.02 * m_g) * t_r.powf(1.5) / (209.0 + 19.0 * m_g + t_r);
    let x = 3.5 + 986.0 / t_r + 0.01 * m_g;
    let y = 2.4 - 0.2 * x;
    k * (x * rho_gcc.powf(y)).exp() * 1e-7
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_density_zeroes_the_row() {
        let (p, t, r_s) = (85.0, 330.0, 420.0);
        assert!(gas_law_row(p, t, ideal_gas_density(p, t, r_s), 1.0, r_s).abs() < 1e-12);
        assert!(gas_law_row(p, t, gas_density(p, t, 0.87, r_s), 0.87, r_s).abs() < 1e-12);
    }

    #[test]
    fn the_dak_density_solves_its_gas_law() {
        for t in [1.05, 1.2, 1.6, 2.2, 3.0] {
            for ppr in [1e-3, 0.2, 1.0, 3.0, 6.0, 10.0, 15.0, 30.0] {
                let r = dak_reduced_density(ppr, t);
                assert!((r * t * dak_z_factor(r, t) / DAK_ZC - ppr).abs() <= 1e-12 * ppr, "t = {t}, ppr = {ppr}");
            }
        }
    }

    #[test]
    fn the_jt_factor_is_t_dlnz_dt_along_an_isobar() {
        let lnz = |ppr: f64, t: f64| dak_z_factor(dak_reduced_density(ppr, t), t).ln();
        for (ppr, t) in [(0.5, 1.3), (2.0, 1.6), (4.5, 1.9), (9.0, 1.7)] {
            let h = 1e-5;
            let fd = t * (lnz(ppr, t + h) - lnz(ppr, t - h)) / (2.0 * h);
            let j = dak_jt_factor(dak_reduced_density(ppr, t), t);
            assert!((j - fd).abs() <= 1e-7 * (1.0 + j.abs()), "{j} vs {fd}");
        }
    }

    #[test]
    fn a_real_gas_is_ideal_at_low_pressure() {
        let (ppc, tpc) = sutton_pseudo_critical(0.65);
        assert!((papay_z_factor(1e3, 300.0, ppc, tpc) - 1.0).abs() < 1e-3);
        assert!(papay_z_factor(150e5, 350.0, ppc, tpc) < 0.95);
    }
}
