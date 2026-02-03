"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 15 May 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no 

Tuning of inflow models
"""

from copy import deepcopy
import pandas as pd
import casadi as ca

from manywells.inflow import InflowModel, ProductivityIndex, Vogel


def calibrate_inflow_model(data: pd.DataFrame, inflow_model: InflowModel):
    """
    Calibrate inflow model using least squares

    TODO: Add calibration of f_g

    :param data: DataFrame with columns (p, v_g, v_l, alpha, rho_g, rho_l, T, u, p_s)
    :param inflow_model: Inflow model
    :return: list of equations
    """
    inflow_model = deepcopy(inflow_model)

    x = ca.SX.sym(f'x')  # Tuning factor
    x_vec = ca.vertcat(x)

    if isinstance(inflow_model, Vogel):
        x_guess = [inflow_model.w_l_max]
        lbx = [0]
        ubx = [200]
        inflow_model.w_l_max = x  # Set tuning factor to Casadi variable
    elif isinstance(inflow_model, ProductivityIndex):
        x_guess = [inflow_model.k_l]
        lbx = [0]
        ubx = [200]
        inflow_model.k_l = x  # Set tuning factor to Casadi variable
    else:
        raise ValueError('Inflow model type not supported')

    ls_objective = 0  # Least-squares objective

    for i in range(len(data)):
        data_i = data.iloc[i]
        p = float(data_i['p'])
        p_r = float(data_i['p_r'])
        w_l = float(data_i['w_l'])
        w_g = float(data_i['w_g'])

        w_l_sim, w_g_sim = inflow_model.mass_flow_rates(p, p_r)

        ls_objective += (w_l - w_l_sim)**2

    ls_objective /= len(data)

    # Solve system of equations using Ipopt
    f = ls_objective  # Least-squares objective
    nlp = {'x': x_vec, 'f': f, 'g': []}
    solver_config = {'ipopt.print_level': 0, 'print_time': 0}
    # solver_config = {}
    solver = ca.nlpsol('S', 'ipopt', nlp, solver_config)
    result = solver(x0=x_guess, lbx=lbx, ubx=ubx)
    # print(result)
    if not solver.stats()['success']:
        print('Solver status:', solver.stats())
        raise Exception('Solve was not successful')

    x_opt = result['x']
    x_opt = x_opt.full().flatten().tolist()  # Convert solution to flat numpy array and then to a list
    return x_opt[0]  # Optimal parameter

