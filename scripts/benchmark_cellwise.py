"""
Benchmark: IPOPT cell-wise solve vs Newton rootfinder cell-wise solve.

Compares the initial-guess phase (_simulate_cellwise) and the full simulation
with both the old IPOPT approach and the new Newton rootfinder approach.

Usage:
    uv run python scripts/benchmark_cellwise.py
"""

import time

import casadi as ca
import numpy as np

from manywells.simulator import (
    WellProperties,
    BoundaryConditions,
    SSDFSimulator,
    SimError,
)
from manywells.units import CF_BAR


class SSDFSimulatorIPOPT(SSDFSimulator):
    """Simulator using the original IPOPT-based cell-wise solve for benchmarking."""

    def _compute_left_boundary_state(self, p_0, T_0):
        """Original IPOPT-based left boundary solve."""
        wp = self.wp
        bc = self.bc
        fl = wp.fluid
        A = self.geo.A
        D = self.geo.D

        Z_0 = float(fl.z_factor(p_0, T_0))
        rho_g = CF_BAR * p_0 / (Z_0 * fl.R_s * T_0)
        w_l_inflow = wp.inflow.liquid_mass_flow_rate(p_0, bc.p_r)
        w_g_inflow = fl.gas_mass_flow_rate(w_l_inflow)

        rho_l = float(fl.liquid_density(p_0, T_0))

        w_o = w_l_inflow * fl.f_o_in_liquid
        Rs_0 = float(fl.rs(p_0, T_0))
        r = Rs_0 * fl.rho_g / fl.rho_o
        w_dissolved = min(r * w_o, w_g_inflow)
        w_g = max(w_g_inflow + bc.w_lg - w_dissolved, 0.0)
        w_l = w_l_inflow + w_dissolved

        self._w_g_total = w_g_inflow
        self._w_o = w_o
        self._w_l_inflow = w_l_inflow

        v_g = ca.SX.sym('v_g_0')
        v_l = ca.SX.sym('v_l_0')
        alpha = ca.SX.sym('alpha_0')

        sigma = fl.surface_tension(p_0, T_0)
        C_0, v_inf = wp.slip.identify_parameters(v_g, v_l, alpha, rho_g, rho_l, sigma, D)
        v_m = w_g / (A * rho_g) + w_l / (A * rho_l)

        g0 = v_g - (C_0 * v_m + v_inf)
        g1 = (A * alpha * rho_g) * v_g - w_g
        g2 = (A * (1 - alpha) * rho_l) * v_l - w_l

        x_vec = ca.vertcat(*[v_g, v_l, alpha])
        g_vec = ca.vertcat(*[g0, g1, g2])

        nlp = {'x': x_vec, 'f': 0, 'g': g_vec}
        solver_config = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('S', 'ipopt', nlp, solver_config)
        alpha_guess = 0.5
        x_guess = [w_g / (A * alpha_guess * rho_g), w_l / (A * (1 - alpha_guess) * rho_l), alpha_guess]
        result = solver(x0=x_guess, lbx=[0, 0, 0], ubx=[ca.inf, ca.inf, 1], lbg=[0, 0, 0], ubg=[0, 0, 0])
        if not solver.stats()['success']:
            raise SimError('compute_left_boundary_state: Could not solve for alpha')

        v_g, v_l, alpha = result['x'].full().flatten().tolist()
        return [p_0, v_g, v_l, alpha, rho_g, rho_l, T_0]

    def _simulate_cellwise(self, p_0, T_0):
        """Original IPOPT-based cell-wise solve."""
        x = list()
        x_0 = self._compute_left_boundary_state(p_0, T_0)
        x += x_0

        for i in range(1, self.n_cells + 1):
            x_i = self._create_variables(i)
            x_i_prev = x[self.dim_x * (i - 1):self.dim_x * i]

            g_i = list()
            g_i += self._differential_equations(x_i, x_i_prev, i)
            g_i += self._closure_relations(x_i)

            x_vec = ca.vertcat(*x_i)
            g_vec = ca.vertcat(*g_i)

            lbx = [0] * len(x_i)
            ubx = [ca.inf] * len(x_i)
            ubx[3] = 1

            nlp = {'x': x_vec, 'f': 0, 'g': g_vec}
            solver = ca.nlpsol('S', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})
            result = solver(x0=x_i_prev, lbx=lbx, ubx=ubx,
                            lbg=[0] * len(g_i), ubg=[0] * len(g_i))
            if not solver.stats()['success']:
                raise SimError(f'IPOPT cell-wise solve failed at cell {i}')

            x += result['x'].full().flatten().tolist()

        return x


def compare_solutions(x_old, x_new, label=""):
    """Print per-variable comparison between two flat solution vectors."""
    var_names = ['p', 'v_g', 'v_l', 'alpha', 'rho_g', 'rho_l', 'T']
    dim_x = len(var_names)
    n_cells = len(x_old) // dim_x

    x_old_arr = np.array(x_old).reshape(n_cells, dim_x)
    x_new_arr = np.array(x_new).reshape(n_cells, dim_x)

    abs_diff = np.abs(x_old_arr - x_new_arr)
    rel_diff = abs_diff / (np.abs(x_old_arr) + 1e-15)

    print(f"\n  {label} per-variable max absolute / relative difference:")
    for j, name in enumerate(var_names):
        print(f"    {name:6s}  abs={abs_diff[:, j].max():.3e}  rel={rel_diff[:, j].max():.3e}")

    return abs_diff.max(), rel_diff.max()


def benchmark_cellwise(n_cells=100):
    """Benchmark the cell-wise initial-guess phase."""
    print(f"{'=' * 60}")
    print(f"Cell-wise solve benchmark (n_cells={n_cells}, dead oil)")
    print(f"{'=' * 60}")

    wp = WellProperties()
    bc = BoundaryConditions(u=0.5)

    p_0 = bc.p_r - (bc.p_r - bc.p_s) * 0.05
    T_0 = bc.T_r

    # --- IPOPT ---
    sim_ipopt = SSDFSimulatorIPOPT(wp, bc)
    t0 = time.perf_counter()
    x_ipopt = sim_ipopt._simulate_cellwise(p_0, T_0)
    t_ipopt = time.perf_counter() - t0

    # --- Newton rootfinder ---
    sim_rf = SSDFSimulator(wp, bc)
    t0 = time.perf_counter()
    x_rf = sim_rf._simulate_cellwise(p_0, T_0)
    t_rf = time.perf_counter() - t0

    print(f"\n  IPOPT:            {t_ipopt:8.3f} s")
    print(f"  Newton rootfinder:{t_rf:8.3f} s")
    speedup = t_ipopt / t_rf if t_rf > 0 else float('inf')
    print(f"  Speedup:          {speedup:8.1f}x")

    max_abs, max_rel = compare_solutions(x_ipopt, x_rf, "IPOPT vs rootfinder")
    print(f"\n  Overall max abs diff: {max_abs:.3e}")
    print(f"  Overall max rel diff: {max_rel:.3e}")

    return t_ipopt, t_rf


def benchmark_full_simulation(n_cells=100):
    """Benchmark the full simulate() call (initial guess + IPOPT solve)."""
    print(f"\n{'=' * 60}")
    print(f"Full simulation benchmark (n_cells={n_cells})")
    print(f"{'=' * 60}")

    wp = WellProperties()
    bc = BoundaryConditions(u=0.5)

    sim = SSDFSimulator(wp, bc)
    t0 = time.perf_counter()
    x = sim.simulate()
    t_total = time.perf_counter() - t0

    n_vars = (n_cells + 1) * sim.dim_x
    print(f"\n  Time:       {t_total:.3f} s")
    print(f"  Variables:  {n_vars}")
    print(f"  Converged:  yes")

    return t_total


if __name__ == '__main__':
    t_ipopt, t_rf = benchmark_cellwise(n_cells=100)
    t_full = benchmark_full_simulation(n_cells=100)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  Cell-wise IPOPT:       {t_ipopt:.3f} s")
    print(f"  Cell-wise rootfinder:  {t_rf:.3f} s")
    print(f"  Cell-wise speedup:     {t_ipopt / t_rf:.1f}x")
    print(f"  Full simulation:       {t_full:.3f} s")
