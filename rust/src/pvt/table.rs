// Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the CC BY-NC 4.0 International Public License.
//
// Created 28 October 2026

//! A PVT table (specs/model/pvt/table.md), as PVTTable in src/manywells/pvt/table.py: R_so, B_o and Z, and
//! optionally mu_o, mu_g and sigma, on a (p, T) grid, interpolated bilinearly with p and T clamped to the grid
//! (PVT-TAB-2). The arithmetic is CasADi's linear interpolant's (casadi_interpn), corner by corner in its order, so
//! that the two backends agree to the bit at the same point. A GOR axis is read in Python (PVT-TAB-7); the core
//! receives the table at the fluid's GOR. Validation stays in Python; the core checks only the shapes.

/// A column of a table
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Column {
    RSo,   // Solution gas-oil ratio (Sm³/Sm³)
    BO,    // Oil formation volume factor (-)
    Z,     // The free gas's z-factor, with the gas constant of the gas at standard conditions (-)
    MuO,   // Oil viscosity (Pa s)
    MuG,   // Gas viscosity (Pa s)
    Sigma, // Gas-oil surface tension (J/m²)
}

/// A table: its grid, and each column's value at each node, the pressure varying fastest (CasADi's order)
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    p: Vec<f64>, // Pressures of the grid (bar), increasing, at least two
    t: Vec<f64>, // Temperatures (K), increasing, at least one: with one, the table does not depend on T
    r_so: Vec<f64>,
    b_o: Vec<f64>,
    z: Vec<f64>,
    mu_o: Option<Vec<f64>>,
    mu_g: Option<Vec<f64>>,
    sigma: Option<Vec<f64>>,
}

impl Table {
    /// A table from its grid and columns, each with one value per node, len(p) len(T) in all, the pressure fastest
    #[allow(clippy::too_many_arguments)]
    pub fn new(p: Vec<f64>, t: Vec<f64>, r_so: Vec<f64>, b_o: Vec<f64>, z: Vec<f64>, mu_o: Option<Vec<f64>>,
               mu_g: Option<Vec<f64>>, sigma: Option<Vec<f64>>) -> Result<Self, String> {
        if p.len() < 2 || t.is_empty() {
            return Err(format!("a PVT table needs at least two pressures and one temperature, got {} and {}", p.len(),
                               t.len()));
        }
        let n = p.len() * t.len();
        let columns = [("R_so", Some(&r_so)), ("B_o", Some(&b_o)), ("Z", Some(&z)), ("mu_o", mu_o.as_ref()),
                       ("mu_g", mu_g.as_ref()), ("sigma", sigma.as_ref())];
        if let Some((name, c)) = columns.iter().find_map(|&(name, c)| c.filter(|c| c.len() != n).map(|c| (name, c))) {
            return Err(format!("the column {name} has {} values, not one per node, {n}", c.len()));
        }
        Ok(Self { p, t, r_so, b_o, z, mu_o, mu_g, sigma })
    }

    /// The column's values, or none where the table does not have it
    fn values(&self, column: Column) -> Option<&[f64]> {
        match column {
            Column::RSo => Some(&self.r_so),
            Column::BO => Some(&self.b_o),
            Column::Z => Some(&self.z),
            Column::MuO => self.mu_o.as_deref(),
            Column::MuG => self.mu_g.as_deref(),
            Column::Sigma => self.sigma.as_deref(),
        }
    }

    /// Whether the table has the column
    pub fn has(&self, column: Column) -> bool {
        self.values(column).is_some()
    }

    /// The column at p (bar) and T (K), or none where the table does not have it
    pub fn get(&self, column: Column, p: f64, t: f64) -> Option<f64> {
        self.values(column).map(|y| self.interpolate(y, p, t))
    }

    /// A column the table always has (R_so, B_o or Z) at p (bar) and T (K)
    pub fn at(&self, column: Column, p: f64, t: f64) -> f64 {
        self.get(column, p, t).expect("R_so, B_o and Z are in every table")
    }

    /// The columns the table has, in the order of the file (R_so, B_o, Z, mu_o, mu_g, sigma), at p (bar) and T (K)
    pub fn columns(&self, p: f64, t: f64) -> Vec<f64> {
        [Column::RSo, Column::BO, Column::Z, Column::MuO, Column::MuG, Column::Sigma].iter()
            .filter_map(|&c| self.get(c, p, t)).collect()
    }

    /// Bilinear interpolation of the node values y at (p, T), each clamped to the grid; linear in p with one
    /// temperature
    fn interpolate(&self, y: &[f64], p: f64, t: f64) -> f64 {  // spec: PVT-TAB-2
        let (i, a) = cell(&self.p, p);
        if self.t.len() == 1 {
            return (1.0 - a) * y[i] + a * y[i + 1];
        }
        let (k, b) = cell(&self.t, t);
        let m = self.p.len();
        // casadi_interpn's sum over the corners, in its order (0, 0), (1, 0), (0, 1), (1, 1)
        (1.0 - a) * (1.0 - b) * y[i + m * k] + a * (1.0 - b) * y[i + 1 + m * k] + (1.0 - a) * b * y[i + m * (k + 1)]
            + a * b * y[i + 1 + m * (k + 1)]
    }
}

/// The cell of an increasing grid that x, clamped to the grid, lies in, as (its lower index, the fraction of the way
/// across it): casadi_low's linear lookup, the first i <= len - 2 with x < g[i + 1]
fn cell(g: &[f64], x: f64) -> (usize, f64) {
    let x = x.max(g[0]).min(g[g.len() - 1]);
    let i = g[1..g.len() - 1].partition_point(|&gi| gi <= x);
    (i, (x - g[i]) / (g[i + 1] - g[i]))
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// A 4 x 3 table of a made-up fluid, smooth in p and T, as tests/test_pvt_table.py's; with every column, or with
    /// R_so, B_o and Z only, at one temperature
    pub fn small(one_temperature: bool) -> Table {
        let p = vec![10.0, 100.0, 200.0, 300.0];
        let t = if one_temperature { vec![350.0] } else { vec![300.0, 350.0, 400.0] };
        let nodes: Vec<(f64, f64)> = t.iter().flat_map(|&t| p.iter().map(move |&p| (p, t))).collect();
        let col = |f: &dyn Fn(f64, f64) -> f64| nodes.iter().map(|&(p, t)| f(p, t)).collect::<Vec<f64>>();
        let r_so = |p: f64, t: f64| 100.0 * (p / 200.0).min(1.0).powf(0.8) * (1.0 + (t - 350.0) / 1000.0);
        let optional = |f: &dyn Fn(f64, f64) -> f64| if one_temperature { None } else { Some(col(f)) };
        Table::new(p.clone(), t.clone(), col(&r_so), col(&|p, t| 1.0 + 0.0025 * r_so(p, t) + 2e-4 * (t - 300.0)),
                   col(&|p, t| 1.0 - 0.3 * p / 300.0 * 350.0 / t),
                   optional(&|p, t| 3e-3 * (-(t - 300.0) / 50.0).exp() / (1.0 + r_so(p, t) / 100.0)),
                   optional(&|p, _| 1.2e-5 + 3e-8 * p), optional(&|p, _| 0.025 * (-p / 150.0).exp())).unwrap()
    }

    /// PVT-TAB-2's formula, written out
    fn bilinear(table: &Table, y: &[f64], p: f64, t: f64) -> f64 {
        let (pg, tg) = (&table.p, &table.t);
        let p = p.clamp(pg[0], pg[pg.len() - 1]);
        let j = (0..pg.len() - 1).rev().find(|&j| pg[j] <= p).unwrap().min(pg.len() - 2);
        let a = (p - pg[j]) / (pg[j + 1] - pg[j]);
        if tg.len() == 1 {
            return (1.0 - a) * y[j] + a * y[j + 1];
        }
        let t = t.clamp(tg[0], tg[tg.len() - 1]);
        let k = (0..tg.len() - 1).rev().find(|&k| tg[k] <= t).unwrap().min(tg.len() - 2);
        let b = (t - tg[k]) / (tg[k + 1] - tg[k]);
        let m = pg.len();
        let y = |j: usize, k: usize| y[j + m * k];
        (1.0 - a) * ((1.0 - b) * y(j, k) + b * y(j, k + 1)) + a * ((1.0 - b) * y(j + 1, k) + b * y(j + 1, k + 1))
    }

    const ALL: [Column; 6] = [Column::RSo, Column::BO, Column::Z, Column::MuO, Column::MuG, Column::Sigma];

    #[test]
    fn exact_at_the_nodes() {
        let table = small(false);
        for c in ALL {
            let y = table.values(c).unwrap();
            for (k, &t) in table.t.iter().enumerate() {
                for (j, &p) in table.p.iter().enumerate() {
                    assert_eq!(table.get(c, p, t).unwrap(), y[j + table.p.len() * k], "{c:?} at ({p}, {t})");
                }
            }
        }
    }

    #[test]
    fn the_formula_between_the_nodes() {
        let table = small(false);
        for k in 0..400 {
            let (p, t) = (10.0 + 290.0 * ((k as f64 * 0.618) % 1.0), 300.0 + 100.0 * ((k as f64 * 0.414) % 1.0));
            for c in ALL {
                let (got, want) = (table.get(c, p, t).unwrap(), bilinear(&table, table.values(c).unwrap(), p, t));
                assert!((got - want).abs() <= 1e-13 * want.abs(), "{c:?} at ({p}, {t}): {got} != {want}");
            }
        }
    }

    #[test]
    fn clamped_outside_the_grid() {
        let table = small(false);
        for ((p, t), (p_in, t_in)) in [((1.0, 320.0), (10.0, 320.0)), ((450.0, 320.0), (300.0, 320.0)),
                                       ((150.0, 250.0), (150.0, 300.0)), ((150.0, 500.0), (150.0, 400.0)),
                                       ((-50.0, 1000.0), (10.0, 400.0)), ((1e4, 1.0), (300.0, 300.0))] {
            assert_eq!(table.columns(p, t), table.columns(p_in, t_in), "({p}, {t})");
        }
    }

    #[test]
    fn one_temperature_is_linear_in_p_and_independent_of_t() {
        let table = small(true);
        assert_eq!(table.columns(55.0, 350.0).len(), 3);
        assert!(!table.has(Column::MuO) && table.get(Column::Sigma, 55.0, 350.0).is_none());
        for p in [1.0, 10.0, 55.0, 150.0, 300.0, 400.0] {
            for c in [Column::RSo, Column::BO, Column::Z] {
                let want = bilinear(&table, table.values(c).unwrap(), p, 350.0);
                assert!((table.at(c, p, 350.0) - want).abs() <= 1e-13 * want.abs());
                assert_eq!(table.at(c, p, 250.0), table.at(c, p, 350.0));
                assert_eq!(table.at(c, p, 500.0), table.at(c, p, 350.0));
            }
        }
    }

    #[test]
    fn the_shapes_are_checked() {
        let ok = |v: usize| vec![1.0; v];
        assert!(Table::new(vec![1.0], vec![300.0], ok(1), ok(1), ok(1), None, None, None).is_err());
        assert!(Table::new(vec![1.0, 2.0], vec![], ok(0), ok(0), ok(0), None, None, None).is_err());
        assert!(Table::new(vec![1.0, 2.0], vec![300.0], ok(2), ok(3), ok(2), None, None, None).is_err());
        assert!(Table::new(vec![1.0, 2.0], vec![300.0], ok(2), ok(2), ok(2), None, Some(ok(1)), None).is_err());
        assert!(Table::new(vec![1.0, 2.0], vec![300.0, 310.0], ok(4), ok(4), ok(4), Some(ok(4)), None, None).is_ok());
    }
}
