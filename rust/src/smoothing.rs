// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 17 July 2026 (Rust port, rust_implementation), restructured 01 October 2026

//! Smooth approximations that are part of the model (specs/model/smoothing.md).

/// Smooth max: (x + y + sqrt((x - y)² + eps)) / 2
pub fn max_approx(x: f64, y: f64, eps: f64) -> f64 {  // spec: SMO-1
    0.5 * (x + y + ((x - y) * (x - y) + eps).sqrt())
}

/// Smooth min: (x + y - sqrt((x - y)² + eps)) / 2
pub fn min_approx(x: f64, y: f64, eps: f64) -> f64 {  // spec: SMO-2
    0.5 * (x + y - ((x - y) * (x - y) + eps).sqrt())
}

/// Sigmoid in x, with inflection point a and rate k: 1 / (1 + exp(-k (x - a)))
pub fn sigmoid(x: f64, a: f64, k: f64) -> f64 {  // spec: SMO-4
    1.0 / (1.0 + (-k * (x - a)).exp())
}

/// Softmax of three values, without shifting them
pub fn softmax3(y: [f64; 3]) -> [f64; 3] {  // spec: SMO-3
    let e = [y[0].exp(), y[1].exp(), y[2].exp()];
    let s = e[0] + e[1] + e[2];
    [e[0] / s, e[1] / s, e[2] / s]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_approx_bounds_max_from_above() {
        let eps = 1e-6;
        for (x, y) in [(1.0, 2.0), (2.0, 1.0), (3.0, 3.0), (-1.0, 0.5)] {
            let m = max_approx(x, y, eps);
            assert!(m >= f64::max(x, y));
            assert!(m - f64::max(x, y) <= 0.5 * eps.sqrt() + 1e-15);
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax3([1.0, -2.0, 0.5]);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        assert!(p.iter().all(|&v| v > 0.0));
    }
}
