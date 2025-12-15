"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 February 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Implementation of the steady-state drift flux model for two-phase flow in vertical wells
"""

from dataclasses import dataclass, field

import casadi as ca
import numpy as np
import pandas as pd

from manywells.constants import STD_GRAVITY, CF_PRES
from manywells.choke import ChokeModel, BernoulliChokeModel, SimpsonChokeModel
from manywells.inflow import InflowModel, ProductivityIndex, Vogel
from manywells.slip import SlipModel


class SimError(Exception):
    """ Exception caused by simulator """
    pass


@dataclass
class WellProperties:

    # Well geometry
    L: float = 2000         # Length of vertical pipe (m)
    D: float = 0.1554       # Inner diameter of pipe (m). Default value ≃ 6.1 inch.

    # Fluid properties
    rho_l: float = 850      # Liquid density (kg/m³)
    R_s: float = 518.3      # Specific gas constant (J/kg/K). Default value is for methane.
    cp_g: float = 2225      # Specific heat capacity of gas (J/kg/K). Default value is for methane.
    cp_l: float = 4180      # Specific heat capacity of liquid (J/kg/K). Default value is for water.

    # Friction
    f_D: float = 0.05       # Darcy friction factor (dimensionless)

    # Heat transfer
    h: float = 20.0         # Heat transfer coefficient (W/m²/K)

    # Slip relation
    slip: SlipModel = field(default_factory=SlipModel)

    # Productivity
    inflow: InflowModel = field(default_factory=lambda: ProductivityIndex(k_l=0.5, f_g=0.1379))

    # Choke model
    choke: ChokeModel = None

    @property
    def A(self) -> float:
        # Compute cross-sectional area of pipe (m²)
        return np.pi * (self.D / 2) ** 2

    def __post_init__(self):
        assert self.L > 0, 'Pipe length must be positive'
        assert self.D > 0, 'Pipe diameter must be positive'

        # Initialize choke model if not provided
        if self.choke is None:
            self.choke = BernoulliChokeModel(K_c=0.1*self.A)


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

    def __init__(self, well_properties: WellProperties, boundary_conditions: BoundaryConditions, n_cells: int = 100):
        """
        Initialize steady-state drift-flux model simulator

        The pipe is discretized into (n + 1) cells of length L / n, where n = n_cells.
        For each cell i in [0, ..., n] we create a state variable, x_i, and equations, g_i.
        State x_0 represents the left boundary at z=0.
        State x_n represents the right boundary at z=L.
        The state holds the following variables (in the given order)
            x = [p, v_g, v_l, alpha, rho_g, rho_l, T],
        where p is pressure, v_g and v_l are gas and liquid velocities, alpha is the volumetric fraction of gas,
        rho_g and rho_l are the gas and liquid densities, and T is temperature.

        :param well_properties: Well properties (object of type WellProperties)
        :param boundary_conditions: Boundary conditions (object of type BoundaryConditions)
        :param n_cells: Number of cells to use in discretization
        """
        self.wp = well_properties       # Well properties
        self.bc = boundary_conditions   # Boundary conditions
        self.n_cells = n_cells          # Dimension of state variables
        self.dim_x = 7                  # Dimension of state variables
        self.x_guess = None             # Initial guess on solution

        self.variable_names = ['p', 'v_g', 'v_l', 'alpha', 'rho_g', 'rho_l', 'T']  # Ordering is important

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

    def _closure_relations(self, x):
        """
        Creates closure relation equations (constraints)

        :param x: Variables
        :return: list of equations
        """
        p, v_g, v_l, alpha, rho_g, rho_l, T = x
        wp = self.wp

        # Derivations for slip relation
        v_m = alpha * v_g + (1 - alpha) * v_l  # Mixture velocity
        C_0, v_inf = wp.slip.identify_parameters(v_g, v_l, alpha, rho_g, rho_l, T, wp.D)  # Slip parameters

        # Closure relations
        g1 = v_g - C_0 * v_m - v_inf                # Slip relation
        g2 = p - rho_g * wp.R_s * T / CF_PRES       # Equation of state for gas density
        g3 = rho_l - wp.rho_l                       # Liquid density (fixed)

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

        # Inflow from reservoir
        w_l, w_g = wp.inflow.mass_flow_rates(p, bc.p_r)

        # Add lift gas flow rate
        w_g += bc.w_lg

        # Create system of equations, g(x) = 0
        g1 = wp.A * alpha * rho_g * v_g - w_g         # Constant flux of gas
        g2 = wp.A * (1 - alpha) * rho_l * v_l - w_l   # Constant flux of liquid
        g3 = T - bc.T_r                               # Inflow fluid temperature (fixed)

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

        w_g = wp.A * alpha * rho_g * v_g  # Gas mass flow rate
        w_l = wp.A * (1 - alpha) * rho_l * v_l  # Liquid mass flow rate
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
        Create discretized differential equations for cell i=cell_index

        :param x: Variables of cell i
        :param x_prev: Variables of cell i-1
        :param cell_index: Cell index (i)
        :return: List of equations
        """
        wp = self.wp
        bc = self.bc
        i = cell_index

        delta_z = wp.L / self.n_cells  # Length of cells in discretization

        # Get variables of current cell (i) and previous cell (i-1)
        p, v_g, v_l, alpha, rho_g, rho_l, T = x
        p_prev, v_g_prev, v_l_prev, alpha_prev, rho_g_prev, rho_l_prev, T_prev = x_prev

        # Helper derivations
        rho_m = alpha * rho_g + (1 - alpha) * rho_l  # Mixture density
        v_m = alpha * v_g + (1 - alpha) * v_l  # Mixture velocity

        acc = alpha * rho_g * v_g ** 2 + (1 - alpha) * rho_l * v_l ** 2
        acc_prev = alpha_prev * rho_g_prev * v_g_prev ** 2 + (1 - alpha_prev) * rho_l_prev * v_l_prev ** 2
        dp_f = delta_z * (wp.f_D / wp.D / 2) * rho_m * (v_m ** 2)
        dp_g = delta_z * STD_GRAVITY * rho_m

        T_a = bc.T_r - i * (bc.T_r - bc.T_s) / self.n_cells  # Linear profile for ambient temperature
        #cp_m = alpha * wp.cp_g + (1 - alpha) * wp.cp_l  # Mixture heat capacity, laminar-bubble-flow
        #dT = delta_z * 4 * wp.h * (T - T_a) / (wp.D * v_m * rho_m * cp_m)
        dT = delta_z * 4 * wp.h * (T - T_a) / (wp.D * (wp.cp_g * alpha * rho_g * v_g + wp.cp_l * (1 - alpha) * rho_l * v_l))

        # Discretized differential equations
        g1 = alpha * rho_g * v_g - alpha_prev * rho_g_prev * v_g_prev                       # Constant flux of gas
        g2 = (1 - alpha) * rho_l * v_l - (1 - alpha_prev) * rho_l_prev * v_l_prev           # Constant flux of liquid
        g3 = acc / CF_PRES + p - (acc_prev / CF_PRES + p_prev) + (dp_f + dp_g) / CF_PRES    # Momentum balance
        g4 = T - T_prev + dT                                                                # Energy balance

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

        rho_l = wp.rho_l
        rho_g = CF_PRES * p_0 / (wp.R_s * T_0)

        w_l, w_g = wp.inflow.mass_flow_rates(p_0, bc.p_r)
        w_g += bc.w_lg

        """
        Solve the following equations for the velocities and void fraction (v_g, v_l, alpha).
        w_g = wp.A * alpha * rho_g * v_g
        w_l = wp.A * (1 - alpha) * rho_l * v_l
        v_g = C_0 * v_m + v_inf
        
        The parameters C_0 and v_inf are functions of v_g, v_l, and alpha. The mix velocity v_m is known since:
        v_m = alpha * v_g + (1 - alpha) * v_l
            = w_g / (wp.A * rho_g) + w_l / (wp.A * rho_l)
        """
        # Variables
        v_g = ca.SX.sym(f'v_g_0')
        v_l = ca.SX.sym(f'v_l_0')
        alpha = ca.SX.sym(f'alpha_0')

        # Slip model
        C_0, v_inf = wp.slip.identify_parameters(v_g, v_l, alpha, rho_g, rho_l, T_0, wp.D)  # Slip parameters
        v_m = w_g / (wp.A * rho_g) + w_l / (wp.A * rho_l)  # Known

        # Equations
        g0 = v_g - (C_0 * v_m + v_inf)
        g1 = (wp.A * alpha * rho_g) * v_g - w_g
        g2 = (wp.A * (1 - alpha) * rho_l) * v_l - w_l

        # Create variable and constraint vectors
        x_vec = ca.vertcat(*[v_g, v_l, alpha])
        g_vec = ca.vertcat(*[g0, g1, g2])

        # Solve system of equations using Ipopt
        nlp = {'x': x_vec, 'f': 0, 'g': g_vec}
        solver_config = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('S', 'ipopt', nlp, solver_config)
        alpha_guess = 0.5
        x_guess = [w_g / (wp.A * alpha_guess * rho_g), w_l / (wp.A * (1 - alpha_guess) * rho_l), alpha_guess]
        result = solver(x0=x_guess, lbx=[0, 0, 0], ubx=[ca.inf, ca.inf, 1], lbg=[0, 0, 0], ubg=[0, 0, 0])
        if not solver.stats()['success']:
            print('Could not solve for alpha')
            print('Solver status:', solver.stats())
            raise SimError('compute_left_boundary_state: Could not solve for alpha')

        v_g, v_l, alpha = result['x'].full().flatten().tolist()

        x_0 = [p_0, v_g, v_l, alpha, rho_g, rho_l, T_0]  # Order is important here!

        return x_0

    def _simulate_cellwise(self, p_0, T_0):
        """
        Simulate cellwise given pressure and temperature at the left boundary

        :param p_0:
        :param T_0:
        :return:
        """

        x = list()  # Variables

        # Compute state of first cell
        x_0 = self._compute_left_boundary_state(p_0, T_0)
        x += x_0

        for i in range(1, self.n_cells + 1):
            # Variables for cell i
            x_i = self._create_variables(i)

            # Get state in previous cell
            x_i_prev = x[self.dim_x * (i - 1):self.dim_x * i]

            # Constraint list
            g_i = list()

            # Get discretized differential equations
            g_i += self._differential_equations(x_i, x_i_prev, i)

            # Closure relations
            g_i += self._closure_relations(x_i)

            # Create variable and constraint vectors
            x_vec = ca.vertcat(*x_i)
            g_vec = ca.vertcat(*g_i)

            # Initial guess on solution
            x_guess = x_i_prev

            # Variable bounds
            # All variables must be non-negative: x >= 0
            lbx = [0] * len(x_i)
            ubx = [ca.inf] * len(x_i)  # We use ca.inf for unbounded variables
            ubx[3] = 1  # alpha <= 1

            # Constraint bounds
            # Equality constraints, g(x) = 0, are implemented as: 0 <= g(x) <= 0
            lbg = [0] * len(g_i)
            ubg = [0] * len(g_i)

            # Solve system of equations using Ipopt
            f = 0  # We set the objective function, f, to zero to solve a feasibility problem
            nlp = {'x': x_vec, 'f': f, 'g': g_vec}
            solver_config = {'ipopt.print_level': 0, 'print_time': 0}
            # solver_config = {}
            solver = ca.nlpsol('S', 'ipopt', nlp, solver_config)
            result = solver(x0=x_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            # print(result)
            if not solver.stats()['success']:
                # print('Failed during cellwise solve')
                print('Solver status:', solver.stats())
                raise SimError('Solve was not successful')

            x_opt = result['x']
            x_opt = x_opt.full().flatten().tolist()  # Convert solution to flat numpy array and then to a list
            x += x_opt  # Add to list of results

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

        # Note that we simulate for n+1 cell states: x_0, x_1, ..., x_n.
        # The distance (in z) between x_0 and x_n is then equal to L
        # Each x_i is ordered as follows: x = (p, v_g, v_l, alpha, rho_g, rho_l)
        x = list()  # Variables
        g = list()  # Constraints (system of equations)

        # Add variables and equations for all cells
        for i in range(self.n_cells + 1):  # Note that we add one cell to get a total length of L

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
            g_i += self._closure_relations(x_i)

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

        # Add z variable
        delta_z = self.wp.L / self.n_cells
        z = np.array([i*delta_z for i in range(self.n_cells + 1)])
        df.insert(loc=0, column='z', value=z)

        # Add slug regime identifier
        #df['flow-regime'] = df['alpha'].map(lambda a: self.wp.slip.flow_regime(a))

        flow_regime = list()
        for index, row in df.iterrows():
            fr = self.wp.slip.flow_regime(row['v_g'], row['v_l'], row['alpha'], row['rho_g'], row['rho_l'], row['T'])
            flow_regime.append(fr)
        df['flow-regime'] = flow_regime

        return df


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ###########################################################################
    # Create a new well (using default values)
    ###########################################################################
    well_properties = WellProperties()
    boundary_conditions = BoundaryConditions(u=0.5)
    sim = SSDFSimulator(well_properties, boundary_conditions)

    ###########################################################################
    # Simulate
    ###########################################################################
    x = sim.simulate()

    # Convert solution to DataFrame
    df = sim.solution_as_df(x)

    ###########################################################################
    # Plot results
    ###########################################################################

    # Plot solution from cell-wise simulation
    df['w_g'] = well_properties.A * df['alpha'] * df['rho_g'] * df['v_g']  # Gas mass flow rates
    df['w_l'] = well_properties.A * (1 - df['alpha']) * df['rho_l'] * df['v_l']  # Liquid mass flow rates
    df['tvd'] = well_properties.L - df['z']  # True vertical depth
    pd.set_option('display.max_columns', None)
    print(df)

    df.plot()
    plt.xlabel('Cell index')
    plt.show()
