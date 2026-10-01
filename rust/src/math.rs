//! Smooth (differentiable) approximations of non-smooth functions.
//! Port from ca_functions.py.

/// Differentiable approximation of max(x, y):
///     max(x, y) ≃ (1/2) * (x + y + sqrt((x - y)² + eps))
pub fn max_approx(x: f64, y: f64) -> f64 {
    const EPS: f64 = 1e-6;
    0.5 * (x + y + ((x - y) * (x - y) + EPS).sqrt())
}

/// Softmax of a 3-vector 
pub fn softmax3(y: [f64; 3]) -> [f64; 3] {
    let e = [y[0].exp(), y[1].exp(), y[2].exp()];
    let s = e[0] + e[1] + e[2];
    [e[0] / s, e[1] / s, e[2] / s]
}
