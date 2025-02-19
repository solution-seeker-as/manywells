"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 08 August 2024
Erlend Lundby, erlend@solutionseeker.no

Implementation of the steady-state drift flux model for two-phase flow in vertical wells

- Mainly based on simulator.py
- Making control inputs (choke opening and gas lift) symbolic variables   (decision variables)
"""

from dataclasses import dataclass

import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from manywells.constants import STD_GRAVITY, CF_PRES
from manywells.choke import ChokeModel, BernoulliChokeModel, SimpsonChokeModel
from manywells.simulator import SSDFSimulator, SimError, WellProperties, BoundaryConditions
from manywells.ca_functions import ca_min_approx, ca_max_approx

class ClosedLoopWellSimulator(SSDFSimulator):
    """
    Simulator for wells operating in closed loop
    """
    def __init__(self, well_properties: WellProperties, boundary_conditions: BoundaryConditions, feedback: bool = True,
                 n_cells: int = 100):
        super().__init__(well_properties, boundary_conditions, n_cells)
        self.feedback = feedback

    def _compute_left_boundary_state(self, p_0, T_0):
        """
        Compute state of first cell given a pressure and temperature

        :param p_0: Pressure of cell in bar (must be specified)
        :param T_0: Temperature of cell in K (must be specified)
        :return:
        """
        bc = self.bc

        # Temporarily set lift gas to 0
        temp_w_lg = bc.w_lg
        if isinstance(bc.w_lg, ca.SX):
            bc.w_lg = 0.0

        # Compute state at left boundary
        x_0 = super()._compute_left_boundary_state(p_0, T_0)

        # Set lift gas back to original variable
        bc.w_lg = temp_w_lg

        return x_0


    def _initial_guess(self):
        """
        Provide an initial guess on the solution
        :return: Initial guess
        """
        bc = self.bc

        #if self.x_guess is None:
        # Guess on pressure and temperature in first cell
        p_0 = bc.p_r - (bc.p_r - bc.p_s) * 0.01  # 5% of the total pressure drop occurs at the inflow
        T_0 = bc.T_r

        # Compute the other cell states and add noise
        x_guess = self._simulate_cellwise(p_0, T_0)
        for i in range(len(x_guess)):
            x_guess[i] = x_guess[i] + np.random.normal()*x_guess[i]*0.1
        return x_guess

    def simulate(self, wtot_ref=10, well=None, x_gu=None):
        """
        Simulate a well by simultaneously solving for all grid cells
        This leads to a large system of equations which we formulate as a non-linear programming problem
        The resulting NLP problem is solved using Ipopt

        :return: Simulation result (variable values stored in a flat list)
        """
        wp = self.wp
        bc = self.bc

        if self.feedback:
            controller_state = ca.SX.sym('controller_state')
            eps_approx = 1e-9
            self.bc.u = ca_min_approx(controller_state,1, eps_approx)

            self.bc.w_lg = ca_max_approx(controller_state-1, 0, eps_approx)

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
                x_i_prev = x[self.dim_x * (i - 1):self.dim_x * i]

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

        if x_gu is None:
            # Initial guess on solution
            x_guess = self._initial_guess()

        # Variable bounds
        # All variables must be non-negative: x >= 0
        lbx = [0] * len(x)
        ubx = [ca.inf] * len(x)  # We use ca.inf for unbounded variables

        if self.feedback:
            u = list()
            u += [controller_state]
            # Lift gas
            u_vec = ca.vertcat(*u)
            lbu = [0] * len(u)
            ubu = [0] * len(u)

            # Boundaries CHK + gas lift
            lbu[0] = 0.0  # CHK minimum 0.0
            ubu[0] = 6.0  # WGL max=5.0 (+CHK max =1)

            if well is not None:
                if well.has_gas_lift:
                    ubu[0] = 6.0
                else:
                    ubu[0] = 1.0

        for i, x_i in enumerate(x):
            if x_i.name().split('_')[0] == 'p':
                lbx[i] = bc.p_s  # Lower bound on pressures
                ubx[i] = bc.p_r  # Upper bound on pressures

            if x_i.name().split('_')[0] == 'T':
                lbx[i] = bc.T_a  # Lower bound on temperatures
                ubx[i] = bc.T_r + 1  # Upper bound on temperatures (slacking bound by adding 1 K)

            if x_i.name().split('_')[0] == 'alpha':
                ubx[i] = 1  # Upper bound on alphas

        # Constraint bounds
        # Equality constraints, g(x) = 0, are implemented as: 0 <= g(x) <= 0
        lbg = [0] * len(g)
        ubg = [0] * len(g)

        # Solve system of equations using Ipopt
        # f = 0  # We set the objective function, f, to zero to solve a feasibility problem
        if self.feedback:

            if isinstance(self.wp.choke, BernoulliChokeModel):
                rho_m = x[-4] * x[-3] + (1 - x[-4]) * x[-2]  # Mixture density alpha*rho_g + (1-alpha)*rho_l
                w_m = self.wp.choke.mass_flow_rate(self.bc.u, x[-7], bc.p_s, rho_m)  # Used to generate dataset v6
                w_g = self.wp.A * x[-4] * x[-3] * x[-6]  # A*alpha*rho_g*v_g
                w_l = w_m - w_g
            elif isinstance(self.wp.choke, SimpsonChokeModel):
                w_g = self.wp.A * x[-4] * x[-3] * x[-6]  # A*alpha*rho_g*v_g
                w_l = self.wp.A * (1 - x[-4]) * x[-2] * x[-5]  # A*(1-alpha)*rho_l*v_l
                x_g = w_g / (w_g + w_l)  # Mass fraction of gas wg/w,
                w_m = self.wp.choke.mass_flow_rate(self.bc.u, x[-7], self.bc.p_s, x_g, x[-3], x[-2])
                w_l = w_m - w_g
            else:
                raise ValueError('Unsupported choke model')

            # w_g = self.wp.A * x[-4] * x[-3] * x[-6]  # A*alpha*rho_g*v_g

            f = (w_m - wtot_ref) ** 2 + self.bc.w_lg ** 2
            if x_gu is None:
                x_0 = x_guess + [0.5]
            else:
                x_0 = x_gu
            x_final = ca.vertcat(x_vec, u_vec)
            nlp = {'x': x_final, 'f': f, 'g': g_vec}
            solver_config = {'ipopt.print_level': 0, 'print_time': 0,
                             'ipopt.max_iter': 500, 'verbose': False}  # 'ipopt.print_level': 1, 'print_time': 1,'ipopt.print_level': 0, 'print_time': 0,

            # solver = ca.nlpsol('S', 'ipopt', nlp, solver_config)
            solver = ca.nlpsol("nlpsol", "ipopt", nlp, solver_config)
            result = solver(x0=x_0, lbx=ca.vertcat(lbx, lbu), ubx=ca.vertcat(ubx, ubu), lbg=lbg, ubg=ubg)
            obj_res = result['f']
        else:
            if x_gu is None:
                x_0 = x_guess
            else:
                x_0 = x_gu
            f = 0
            nlp = {'x': x_vec, 'f': f, 'g': g_vec}
            solver_config = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 500}
            # solver_config ={}
            solver_config['verbose'] = False
            solver = ca.nlpsol('S', 'ipopt', nlp, solver_config)
            result = solver(x0=x_0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            obj_res = result['f']

        if not solver.stats()['success']:
            print('Simulation failed')
            raise SimError('Solve was not successful')


        x_opt = result['x']
        x_opt = x_opt.full().flatten().tolist()  # Convert solution to flat numpy array and then to a list

        # Update choke position and gas lift rate
        if self.feedback:
            t = x_opt[-1]
            self.bc.u = ca.fmin(1, t)
            self.bc.w_lg = ca.fmax(0, t - 1)

        return x_opt, obj_res


if __name__ == '__main__':

    ###########################################################################
    # Create a new well (using default values)
    ###########################################################################
    well_properties = WellProperties()
    well_properties.L = 2400
    boundary_conditions = BoundaryConditions(u=0.9)
    boundary_conditions.pr = (1 / CF_PRES) * 1000 * STD_GRAVITY * well_properties.L + 1
    boundary_conditions.T_r = 273.15+60 + (well_properties.L - 1500) * 0.03
    print(well_properties.L)
    print(boundary_conditions.pr)
    sim = ClosedLoopWellSimulator(well_properties, boundary_conditions, n_cells=int(well_properties.L/10))
    sim.feedback = True

    ###########################################################################
    # Simulate
    ###########################################################################
    x, _ = sim.simulate(wtot_ref=15.0)
    print('CHK')
    print(ca.fmin(x[-1],1))
    print('----')
    print('WLG')
    print(ca.fmax(0, x[-1]-1))
    print('---------------')
    #a =(1 - x[3])*x[5]*x[2]*well_properties.A
    #print(a)
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
    print('------')
    print(ca.repmat([1, 1000, 1e5],1,5))
    print('------')
    print(ca.vertcat(100,10,10,1,10,100))
    plt.show()
