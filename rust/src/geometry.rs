// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 01 October 2026

//! The well's geometry and grid (specs/model/geometry.md): a survey of measured and true vertical depth at the grid
//! points, from which each cell's length along the flow path, its inclination and each point's depth as a fraction of
//! the bottomhole's follow, as in WellGeometry (src/manywells/geometry.py). Interpolating a sparse survey onto the
//! grid (GEO-4) stays in Python.

use std::f64::consts::PI;

/// Cross-section (m²) of a circular pipe of inner diameter d (m)
pub fn cross_section(d: f64) -> f64 {  // spec: GEO-2
    PI * (d / 2.0) * (d / 2.0)
}

/// The grid, in simulator order: point 0 is the bottomhole and point N the wellhead; cell i lies between points
/// i - 1 and i, and its values are at index i - 1
#[derive(Clone, Debug)]
pub struct Geometry {
    pub d: f64,             // Inner pipe diameter (m)
    pub delta_md: Vec<f64>, // Length of each cell along the flow path (m)
    pub cos_incl: Vec<f64>, // Cosine of each cell's inclination from vertical, in [0, 1]
    pub tvd_frac: Vec<f64>, // True vertical depth of each point as a fraction of the bottomhole's
}

/// What the rows of cell i use of the geometry
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub delta_md: f64,
    pub cos_incl: f64,
    pub tvd_frac: f64, // of the cell's upper point, point i
}

impl Geometry {
    /// The grid of the measured and true vertical depths md and tvd (m) at the grid points, bottomhole first, as
    /// WellGeometry.md and WellGeometry.tvd, with WellGeometry's arithmetic
    pub fn from_grid(md: &[f64], tvd: &[f64], d: f64) -> Result<Self, String> {  // spec: GEO-3
        if md.len() != tvd.len() || md.len() < 2 {
            return Err("the grid needs the same number (at least two) of measured and vertical depths".into());
        }
        if d.is_nan() || d <= 0.0 {
            return Err("the pipe diameter must be positive".into());
        }
        let n = md.len() - 1;
        let delta_md: Vec<f64> = (0..n).map(|j| md[j] - md[j + 1]).collect();
        let cos_incl: Vec<f64> = (0..n).map(|j| (tvd[j] - tvd[j + 1]) / delta_md[j]).collect();
        if delta_md.iter().any(|&l| l.is_nan() || l <= 0.0) || cos_incl.iter().any(|&c| !(0.0..=1.0).contains(&c)) {
            return Err("the measured depth must fall towards the wellhead, and each cell's inclination lie in [0, 1]"
                .into());
        }
        let tvd_frac = tvd.iter().map(|&z| if tvd[0] > 0.0 { z / tvd[0] } else { 0.0 }).collect();
        Ok(Self { d, delta_md, cos_incl, tvd_frac })
    }

    pub fn n_cells(&self) -> usize {
        self.delta_md.len()
    }

    /// Cross-section (m²)
    pub fn a(&self) -> f64 {
        cross_section(self.d)
    }

    /// The geometry of cell i, 1 <= i <= N
    pub fn cell(&self, i: usize) -> Cell {
        Cell { delta_md: self.delta_md[i - 1], cos_incl: self.cos_incl[i - 1], tvd_frac: self.tvd_frac[i] }
    }

    /// The inclination of point i's closures: its cell's, and cell 1's at the bottomhole (DISC-11)
    pub fn point_cos(&self, i: usize) -> f64 {
        self.cos_incl[i.max(1) - 1]
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// A vertical pipe of length l on n uniform cells, bottomhole first
    pub fn vertical(l: f64, n: usize, d: f64) -> Geometry {
        let z: Vec<f64> = (0..=n).rev().map(|k| l * k as f64 / n as f64).collect();
        Geometry::from_grid(&z, &z, d).unwrap()
    }

    /// A survey of (MD, TVD) stations from the surface, interpolated linearly onto n uniform cells, as
    /// WellGeometry.from_survey does (GEO-4, which the core leaves to Python), bottomhole first
    pub fn survey(md: &[f64], tvd: &[f64], n: usize, d: f64) -> Geometry {
        let total = md[md.len() - 1];
        let at = |m: f64| {
            let k = md.windows(2).position(|w| m <= w[1]).unwrap_or(md.len() - 2);
            tvd[k] + (tvd[k + 1] - tvd[k]) * (m - md[k]) / (md[k + 1] - md[k])
        };
        let m: Vec<f64> = (0..=n).rev().map(|k| total * k as f64 / n as f64).collect();
        let z: Vec<f64> = m.iter().map(|&m| at(m)).collect();
        Geometry::from_grid(&m, &z, d).unwrap()
    }

    #[test]
    fn a_vertical_grid_has_unit_inclination() {
        let g = vertical(2000.0, 10, 0.15);
        assert!(g.cos_incl.iter().all(|&c| c == 1.0));
        assert_eq!((g.tvd_frac[0], g.tvd_frac[10]), (1.0, 0.0));
        assert!((g.delta_md.iter().sum::<f64>() - 2000.0).abs() < 1e-9);
    }

    #[test]
    fn a_horizontal_cell_has_zero_inclination() {
        // Bottomhole at the toe of a 500 m horizontal section at 1000 m depth
        let g = Geometry::from_grid(&[1500.0, 1000.0, 0.0], &[1000.0, 1000.0, 0.0], 0.1).unwrap();
        assert_eq!(g.cos_incl, vec![0.0, 1.0]);
        assert_eq!(g.tvd_frac, vec![1.0, 1.0, 0.0]);
        assert_eq!(g.point_cos(0), 0.0);
    }

    #[test]
    fn the_grid_must_rise_to_the_wellhead() {
        assert!(Geometry::from_grid(&[0.0, 100.0], &[0.0, 100.0], 0.1).is_err());
        assert!(Geometry::from_grid(&[100.0, 0.0], &[150.0, 0.0], 0.1).is_err());
    }
}
