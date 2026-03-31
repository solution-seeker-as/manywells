"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 February 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Implementation of the steady-state drift flux model for two-phase flow in wellbores
"""

from dataclasses import dataclass, field

import casadi as ca
import numpy as np
import pandas as pd

from manywells.geometry import WellGeometry
from manywells.units import STD_GRAVITY, CF_BAR
from manywells.choke import ChokeModel, BernoulliChokeModel, SimpsonChokeModel
from manywells.friction import friction_factor
from manywells.inflow import InflowModel, ProductivityIndex, Vogel
import manywells.pvt as pvt
from manywells.pvt.fluid import FluidModel
from manywells.slip import SlipModel


class SimError(Exception):
    """ Exception caused by simulator """
    pass


@dataclass
class WellProperties:

    # Well geometry
    geometry: WellGeometry = field(default_factory=lambda: WellGeometry.vertical(length=2000, n_cells=100))

    # Fluid model
    fluid: FluidModel = field(default_factory=FluidModel)

    # Friction
    # Roughness of new/smooth Tubing: 0.0015 to 0.045 mm = 1.5e-6 to 4.5e-5 m
    # Roughness of commercial/welded steel: 0.045 mm = 4.5e-5 m
    roughness: float = 4.5e-5  # Pipe wall roughness (m). Default: commercial steel.
    f_D: float = None          # Fixed Darcy friction factor (overrides roughness-based calculation when set)

    # Heat transfer
    h: float = 20.0         # Heat transfer coefficient (W/m²/K)

    # Slip relation
    slip: SlipModel = field(default_factory=SlipModel)

    # Productivity
    inflow: InflowModel = field(default_factory=lambda: ProductivityIndex(k_l=0.5))

    # Choke model
    choke: ChokeModel = None

    def __post_init__(self):
        assert self.roughness > 0, 'Pipe roughness must be positive'
        if self.f_D is not None:
            assert self.f_D > 0, 'Friction factor must be positive'


@dataclass
class BoundaryConditions:
    # Pressures
    p_r: float = 170          # Upstream reservoir pressure (bar)
    p_s: float = 20           # Downstream separator pressure (bar)

    # Temperatures
    T_r: float = 373.15       # Reservoir temperature (K). Default value corresponds to 100 decC.
    T_s: float = 277.15       # Ambient temperature (K) at the surface z=L. Default value corresponds to 4 degC.

    # Controls
    u: float = 1.             # Choke position (dimensionless). Must be in [0,1].
    w_lg: float = 0.          # Lift gas mass flow rate (kg/s). Default value is 0. Assumed injected at z=0.

    def __post_init__(self):
        assert self.p_r > 0, 'Reservoir pressure must be positive'
        assert self.p_s > 0, 'Separator pressure must be positive'
        assert 0 <= self.u <= 1, 'Choke opening must be in [0, 1]'
        assert 0 <= self.w_lg, 'Gas lift rate must be non-negative'


class SSDFSimulator:
    """
    Implementation of the steady-state drift-flux model
    """

    def __init__(self, well_properties: WellProperties, boundary_conditions: BoundaryConditions):
        """
        Initialize steady-state drift-flux model simulator

        The pipe is discretized into n cells (see WellGeometry object)
        Variables x_i and equations g_i are indexed at grid points i = 0, ..., n.
        State x_0 represents the left boundary (bottom of the well).
        State x_n represents the right boundary (top of the well / surface).
        The state holds the following variables (in the given order)
            x = [p, v_g, v_l, alpha, rho_g, rho_l, T],
        where p is pressure, v_g and v_l are gas and liquid velocities, alpha is the volumetric fraction of gas,
        rho_g and rho_l are the gas and liquid densities, and T is temperature.

        :param well_properties: Well properties (object of type WellProperties)
        :param boundary_conditions: Boundary conditions (object of type BoundaryConditions)
        """
        self.wp = well_properties       # Well properties
        self.bc = boundary_conditions   # Boundary conditions
        self.geo = well_properties.geometry  # Convenience alias
        self.n_cells = self.geo.n_cells
        self.dim_x = 7                  # Dimension of state variables
        self.x_guess = None             # Initial guess on solution

        self.variable_names = ['p', 'v_g', 'v_l', 'alpha', 'rho_g', 'rho_l', 'T']  # Ordering is important

        # Initialize choke model if not provided
        if self.wp.choke is None:
            self.wp.choke = BernoulliChokeModel(K_c=0.1 * self.geo.A)

    @staticmethod
    def _create_variables(cell_index: int):
        """
        Create variables for cell i=cell_index

        :param cell_index: Cell index used in variable names
        :return: List of variables with order [p, v_g, v_l, alpha, rho_g, rho_l, T]
        """
        i = cell_index

        p = ca.SX.sym(f'p_{i}')
        v_g = ca.SX.sym(f'v_g_{i}')
        v_l = ca.SX.sym(f'v_l_{i}')
        alpha = ca.SX.sym(f'alpha_{i}')
        rho_g = ca.SX.sym(f'rho_g_{i}')
        rho_l = ca.SX.sym(f'rho_l_{i}')
        T = ca.SX.sym(f'T_{i}')

        return [p, v_g, v_l, alpha, rho_g, rho_l, T]

    def _closure_relations(self, x, cell_index: int):
        """
        Creates closure relation equations (constraints)

        :param x: Variables
        :param cell_index: Cell index (i)
        :return: list of equations
        """
        p, v_g, v_l, alpha, rho_g, rho_l, T = x
        wp = self.wp
        geo = self.geo
        fl = wp.fluid

        # Derivations for slip relation
        v_m = alpha * v_g + (1 - alpha) * v_l  # Mixture velocity
        sigma = fl.surface_tension(p, T)
        cos_incl = geo.cos_incl[min(cell_index, geo.n_cells - 1)]  # Clamp the cell index since we loop over n_cells + 1
        C_0, v_inf = wp.slip.identify_parameters(v_g, v_l, alpha, rho_g, rho_l, sigma, geo.D, cos_incl)

        # Closure relations
        g1 = v_g - C_0 * v_m - v_inf                # Slip relation
        Z = fl.z_factor(p, T)
        g2 = p - Z * rho_g * fl.R_s * T / CF_BAR    # Real gas equation of state
        g3 = rho_l - fl.liquid_density(p, T)        # Liquid density

        return [g1, g2, g3]

    def _left_boundary_eqs(self, x):
        """
        Equations (constraints) representing the left boundary conditions

        :param x: Variables, x(z=0)
        :return: list of equations
        """
        p, v_g, v_l, alpha, rho_g, rho_l, T = x
        wp = self.wp
        bc = self.bc
        A = self.geo.A

        fl = wp.fluid

        # Inflow from reservoir
        w_l = wp.inflow.liquid_mass_flow_rate(p, bc.p_r)
        w_g = fl.gas_mass_flow_rate(w_l)

        self._w_g_total = w_g
        self._w_o = w_l * fl.f_o_in_liquid
        self._w_l_inflow = w_l

        Rs_0 = fl.rs(p, T)
        w_g_free = fl.free_gas_flux(Rs_0, w_g, bc.w_lg, self._w_o)
        w_l_total = fl.liquid_flux(Rs_0, w_l, self._w_o, w_g)

        g1 = A * alpha * rho_g * v_g - w_g_free
        g2 = A * (1 - alpha) * rho_l * v_l - w_l_total
        g3 = T - bc.T_r  # Inflow fluid temperature (fixed)

        return [g1, g2, g3]

    def _right_boundary_eqs(self, x):
        """
        Equations (constraints) representing the right boundary conditions

        :param x: Variables, x(z=L)
        :return: list of equations
        """
        p, v_g, v_l, alpha, rho_g, rho_l, T = x
        wp = self.wp
        bc = self.bc
        A = self.geo.A

        w_g = A * alpha * rho_g * v_g  # Gas mass flow rate
        w_l = A * (1 - alpha) * rho_l * v_l  # Liquid mass flow rate
        w_m = w_g + w_l  # Mixture mass flow rate

        # Choke equation, where the upstream pressure is p_in = p(z=L) and the downstream pressure is p_out = p_s.
        if isinstance(wp.choke, BernoulliChokeModel):
            rho_m = alpha * rho_g + (1 - alpha) * rho_l  # Mixture density
            g1 = w_m - wp.choke.mass_flow_rate(bc.u, p, bc.p_s, rho_m)  # Used to generate dataset v6
        elif isinstance(wp.choke, SimpsonChokeModel):
            x_g = w_g / w_m  # Mass fraction of gas
            g1 = w_m - wp.choke.mass_flow_rate(bc.u, p, bc.p_s, x_g, rho_g, rho_l)  # With two-phase correction
        else:
            raise ValueError('Unsupported choke model')

        return [g1]

    def _differential_equations(self, x, x_prev, cell_index: int):
        """
        Create discretized differential equations for cell i

        :param x: Variables of cell i
        :param x_prev: Variables of cell i-1
        :param cell_index: Cell index (i), ranging from 1 to n_cells
        :return: List of equations
        """
        wp = self.wp
        bc = self.bc
        fl = wp.fluid
        geo = self.geo
        D = geo.D
        A = geo.A

        delta_md = geo.delta_md[cell_index - 1]
        cos_incl = geo.cos_incl[cell_index - 1]
        delta_tvd = delta_md * cos_incl

        # Get variables of current cell (i) and previous cell (i-1)
        p, v_g, v_l, alpha, rho_g, rho_l, T = x
        p_prev, v_g_prev, v_l_prev, alpha_prev, rho_g_prev, rho_l_prev, T_prev = x_prev

        # Helper derivations
        rho_m = alpha * rho_g + (1 - alpha) * rho_l  # Mixture density
        v_m = alpha * v_g + (1 - alpha) * v_l  # Mixture velocity

        if wp.f_D is not None:
            f_D = wp.f_D
        else:
            mu_l = fl.liquid_viscosity(p, T)
            mu_g = fl.gas_viscosity(T, rho_g)
            mu_m = pvt.mixture_viscosity(mu_l, mu_g, alpha, rho_l=rho_l, rho_g=rho_g)
            Re = rho_m * ca.fabs(v_m) * D / mu_m
            f_D = friction_factor(Re, wp.roughness / D)

        # Acceleration terms
        acc = alpha * rho_g * v_g ** 2 + (1 - alpha) * rho_l * v_l ** 2
        acc_prev = alpha_prev * rho_g_prev * v_g_prev ** 2 + (1 - alpha_prev) * rho_l_prev * v_l_prev ** 2

        # Frictional pressure drop (acts along the flow path)
        dp_f = delta_md * (f_D / D / 2) * rho_m * (v_m ** 2)

        # Gravitational pressure drop (only the vertical component contributes)
        dp_g = delta_tvd * STD_GRAVITY * rho_m

        # Ambient temperature: linear geothermal gradient based on TVD fraction
        tvd_frac_i = geo.tvd_frac[cell_index]
        T_a = bc.T_s + (bc.T_r - bc.T_s) * tvd_frac_i

        cp_flux = fl.cp_g * alpha * rho_g * v_g + fl.cp_l * (1 - alpha) * rho_l * v_l

        # Heat loss to surroundings
        dT_heat = delta_md * 4 * wp.h * (T - T_a) / (D * cp_flux)

        # Frictional dissipation heating of the liquid phase
        # For ideal gas, friction does not change enthalpy; for incompressible liquid it does
        F_fric = (f_D / D / 2) * rho_m * v_m ** 2
        dT_fric = delta_md * (1 - alpha) * v_l * F_fric / cp_flux

        # Gravitational cooling (adiabatic lapse rate effect)
        # Only the vertical component of the gravity vector contributes
        mass_flux = alpha * rho_g * v_g + (1 - alpha) * rho_l * v_l
        liq_flux = (1 - alpha) * v_l
        dT_grav = delta_tvd * STD_GRAVITY * (mass_flux - liq_flux * rho_m) / cp_flux

        dT = dT_heat - dT_fric + dT_grav

        # Discretized differential equations
        Rs_i = fl.rs(p, T)
        w_g_free_i = fl.free_gas_flux(Rs_i, self._w_g_total, bc.w_lg, self._w_o)
        w_l_total_i = fl.liquid_flux(Rs_i, self._w_l_inflow, self._w_o, self._w_g_total)
        g1 = A * alpha * rho_g * v_g - w_g_free_i
        g2 = A * (1 - alpha) * rho_l * v_l - w_l_total_i

        g3 = acc / CF_BAR + p - (acc_prev / CF_BAR + p_prev) + (dp_f + dp_g) / CF_BAR           # Momentum balance
        g4 = T - T_prev + dT                                                                    # Thermal energy balance

        return [g1, g2, g3, g4]

    def _compute_left_boundary_state(self, p_0, T_0):
        """
        Compute state of first cell given a pressure and temperature

        :param p_0: Pressure of cell in bar (must be specified)
        :param T_0: Temperature of cell in K (must be specified)
        :return:
        """
        wp = self.wp
        bc = self.bc
        geo = self.geo
        fl = wp.fluid
        A = geo.A
        D = geo.D

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

        """
        Solve the following equations for the velocities and void fraction (v_g, v_l, alpha).
        w_g = A * alpha * rho_g * v_g
        w_l = A * (1 - alpha) * rho_l * v_l
        v_g = C_0 * v_m + v_inf
        
        The parameters C_0 and v_inf are functions of v_g, v_l, and alpha. The mix velocity v_m is known since:
        v_m = alpha * v_g + (1 - alpha) * v_l
            = w_g / (A * rho_g) + w_l / (A * rho_l)
        """
        # Variables
        v_g = ca.SX.sym(f'v_g_0')
        v_l = ca.SX.sym(f'v_l_0')
        alpha = ca.SX.sym(f'alpha_0')

        # Slip model
        sigma = fl.surface_tension(p_0, T_0)
        cos_incl = geo.cos_incl[0]
        C_0, v_inf = wp.slip.identify_parameters(v_g, v_l, alpha, rho_g, rho_l, sigma, D, cos_incl)
        v_m = w_g / (A * rho_g) + w_l / (A * rho_l)  # Known

        # Equations
        g0 = v_g - (C_0 * v_m + v_inf)
        g1 = (A * alpha * rho_g) * v_g - w_g
        g2 = (A * (1 - alpha) * rho_l) * v_l - w_l

        # Create variable and constraint vectors
        x_vec = ca.vertcat(*[v_g, v_l, alpha])
        g_vec = ca.vertcat(*[g0, g1, g2])

        # Solve system of equations using Newton rootfinder
        F = ca.Function('F_bc', [x_vec], [g_vec])
        rf = ca.rootfinder('rf_bc', 'newton', F)
        alpha_guess = 0.5
        x_guess = [w_g / (A * alpha_guess * rho_g), w_l / (A * (1 - alpha_guess) * rho_l), alpha_guess]
        try:
            result = rf(x_guess)
        except RuntimeError as e:
            raise SimError(f'compute_left_boundary_state: rootfinder failed: {e}')

        v_g, v_l, alpha = result.full().flatten().tolist()

        x_0 = [p_0, v_g, v_l, alpha, rho_g, rho_l, T_0]  # Order is important here!

        return x_0

    def _simulate_cellwise(self, p_0, T_0):
        """
        Simulate cellwise given pressure and temperature at the left boundary.

        Builds a per-cell rootfinder so that _differential_equations receives a
        concrete cell_index and can look up geometry (cos_incl, tvd_frac) directly.

        :param p_0: Pressure at left boundary (bar)
        :param T_0: Temperature at left boundary (K)
        :return: Flat list of state variables for all cells
        """

        x = list()  # Variables

        # Compute state of first cell
        x_0 = self._compute_left_boundary_state(p_0, T_0)
        x += x_0

        for i in range(1, self.n_cells + 1):
            # Build rootfinder for this cell i
            x_i = self._create_variables(i)
            x_i_prev = self._create_variables(i - 1)            

            g_diff = self._differential_equations(x_i, x_i_prev, i)
            g_clos = self._closure_relations(x_i, i)

            # Pack symbolic variables (x_vec), equations (g_vec), and parameters
            # (p_vec) into CasADi vectors.  F maps (x, p) -> g(x; p) = 0, and
            # the rootfinder solves for x given p (the previous cell's state).
            x_vec = ca.vertcat(*x_i)
            g_vec = ca.vertcat(*(g_diff + g_clos))
            p_vec = ca.vertcat(*x_i_prev)

            F = ca.Function(f'F_{i}', [x_vec, p_vec], [g_vec])
            rf = ca.rootfinder(f'rf_{i}', 'newton', F)

            x_i_prev_values = x[self.dim_x * (i - 1):self.dim_x * i]

            try:
                result = rf(list(x_i_prev_values), list(x_i_prev_values))
            except RuntimeError:
                raise SimError(f'Cell-wise rootfinder failed at cell {i}')

            x_opt = result.full().flatten().tolist()
            x += x_opt

        return x

    def _initial_guess(self):
        """
        Provide an initial guess on the solution
        :return: Initial guess
        """
        bc = self.bc

        if self.x_guess is None:
            # Guess on pressure and temperature in first cell
            p_0 = bc.p_r - (bc.p_r - bc.p_s) * 0.05  # 5% of the total pressure drop occurs at the inflow
            T_0 = bc.T_r

            # Compute the other cell states
            x_guess = self._simulate_cellwise(p_0, T_0)
            return x_guess

        else:
            # Use provided initial guess
            return self.x_guess

    def simulate(self):
        """
        Simulate a well by simultaneously solving for all grid cells
        This leads to a large system of equations which we formulate as a non-linear programming problem
        The resulting NLP problem is solved using Ipopt

        :return: Simulation result (variable values stored in a flat list)
        """
        wp = self.wp
        bc = self.bc

        # We simulate for n cells, with variables at n+1 grid points: x_0, x_1, ..., x_n.
        # Each x_i is ordered as follows: x = (p, v_g, v_l, alpha, rho_g, rho_l)
        x = list()  # Variables
        g = list()  # Constraints (system of equations)

        # Add variables and equations
        for i in range(self.n_cells + 1):  # Loop over all grid points

            # Variables for cell i
            x_i = self._create_variables(i)

            # Equations / constraints for cell i
            g_i = list()

            if i == 0:  # Left boundary
                g_i += self._left_boundary_eqs(x_i)

            else:  # Cells 1,...,n

                # Get state in previous cell
                x_i_prev = x[self.dim_x*(i-1):self.dim_x*i]

                # Discretized differential equations
                g_i += self._differential_equations(x_i, x_i_prev, i)

            if i == self.n_cells:  # Right boundary
                g_i += self._right_boundary_eqs(x_i)

            # Closure relations
            g_i += self._closure_relations(x_i, i)

            # Add to variable and constraint lists
            x += x_i
            g += g_i

        # Create variable and constraint vectors
        x_vec = ca.vertcat(*x)
        g_vec = ca.vertcat(*g)

        # Initial guess on solution
        x_guess = self._initial_guess()

        # Variable bounds
        # All variables must be non-negative: x >= 0
        lbx = [0] * len(x)
        ubx = [ca.inf] * len(x)  # We use ca.inf for unbounded variables

        for i, x_i in enumerate(x):
            if x_i.name().split('_')[0] == 'p':
                lbx[i] = bc.p_s  # Lower bound on pressures
                ubx[i] = bc.p_r  # Upper bound on pressures

            if x_i.name().split('_')[0] == 'T':
                lbx[i] = bc.T_s      # Lower bound on temperatures
                ubx[i] = bc.T_r + 1  # Upper bound on temperatures (slacking bound by adding 1 K)

            if x_i.name().split('_')[0] == 'alpha':
                ubx[i] = 1  # Upper bound on alphas

        # Constraint bounds
        # Equality constraints, g(x) = 0, are implemented as: 0 <= g(x) <= 0
        lbg = [0] * len(g)
        ubg = [0] * len(g)

        # Solve system of equations using Ipopt
        f = 0  # We set the objective function, f, to zero to solve a feasibility problem
        nlp = {'x': x_vec, 'f': f, 'g': g_vec}
        solver_config = {'ipopt.print_level': 0, 'print_time': 0}
        # solver_config = {}
        solver = ca.nlpsol('S', 'ipopt', nlp, solver_config)
        result = solver(x0=x_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        # print(result)
        if not solver.stats()['success']:
            print('Simulation failed')
            print('Solver status:', solver.stats())
            raise SimError('Solve was not successful')

        x_opt = result['x']
        x_opt = x_opt.full().flatten().tolist()  # Convert solution to flat numpy array and then to a list

        return x_opt

    def solution_as_df(self, x):
        """
        Represent solution as a DataFrame

        :param x: Solution as flat list
        :return: Solution as DataFrame
        """
        x_list = list()
        for i in range(self.n_cells + 1):
            x_i = x[self.dim_x * i:self.dim_x * (i + 1)]
            x_list.append(np.array(x_i))

        cols = self.variable_names
        df = pd.DataFrame(x_list, columns=cols)

        # Add geometry columns (simulator order: bottom to top)
        geo = self.geo
        md = np.array(geo.md)    # simulator order (bottom → top)
        tvd = np.array(geo.tvd)  # simulator order (bottom → top)

        df.insert(loc=1, column='md', value=md)
        df.insert(loc=2, column='tvd', value=tvd)

        # Add flow regime identifier
        fl = self.wp.fluid
        flow_regime = list()
        for index, row in df.iterrows():
            sigma = float(fl.surface_tension(row['p'], row['T']))
            cos_incl = geo.cos_incl[min(index, geo.n_cells - 1)]
            fr = self.wp.slip.flow_regime(row['v_g'], row['v_l'], row['alpha'], row['rho_g'], row['rho_l'], sigma, cos_incl)
            flow_regime.append(fr)
        df['flow-regime'] = flow_regime

        return df
