"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 15 May 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no 

Scripts for calibrating choke models
"""

from copy import deepcopy
import pandas as pd
import casadi as ca

from manywells.choke import ChokeModel, BernoulliChokeModel, SimpsonChokeModel


def calibrate_bernoulli_choke_model(data: pd.DataFrame, choke: BernoulliChokeModel):
    """
    Calibration of choke model using least squares

    The tuning factor is the choke coefficient, K_c

    The choke model is calibrated using a dataset with features:
        u: choke position
        p: pressure upstream choke,
        p_s: pressure downstream choke,
        w_m: total (mixture) mass flow rate through choke,
        rho_m: mixture density (upstream choke),

    :param data: DataFrame with columns (u, p, p_s, w_m, rho_m)
    :param choke: Choke model
    :return: Optimal choke coefficient, K_c
    """
    assert isinstance(choke, BernoulliChokeModel), "Choke model is not of type BernoulliChokeModel"

    choke = deepcopy(choke)

    K_c_init = choke.K_c
    x_guess = [K_c_init]
    lbx = [0]
    ubx = [1]

    K_c = ca.SX.sym(f'K_c')  # Tuning factor
    choke.K_c = K_c  # Set choke coefficient to Casadi variable
    x_vec = ca.vertcat(K_c)

    ls_objective = 0  # Least-squares objective

    for i in range(len(data)):
        data_i = data.iloc[i]
        u = float(data_i['u'])
        p = float(data_i['p'])
        p_s = float(data_i['p_s'])
        w_m = float(data_i['w_m'])
        rho_m = float(data_i['rho_m'])

        g1 = w_m - choke.mass_flow_rate(u, p, p_s, rho_m)

        ls_objective += g1**2

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
    return x_opt[0]  # Optimal K_c


def calibrate_simpson_choke_model(data: pd.DataFrame, choke: SimpsonChokeModel):
    """
    Calibration of choke model using least squares

    The tuning factor is the choke coefficient, K_c

    The choke model is calibrated using a dataset with features:
        p: pressure upstream choke,
        p_s: pressure downstream choke,
        w_m: total (mixture) mass flow rate through choke,
        rho_m: mixture density (upstream choke),
        u: choke position

    :param data: DataFrame with columns (p, p_s, w_m, rho_m, u)
    :param choke: Choke model
    :return: list of equations
    """
    choke = deepcopy(choke)

    K_c_init = choke.K_c
    x_guess = [K_c_init]
    lbx = [0]
    ubx = [1]

    K_c = ca.SX.sym(f'K_c')  # Tuning factor
    choke.K_c = K_c  # Set choke coefficient to Casadi variable
    x_vec = ca.vertcat(K_c)

    ls_objective = 0  # Least-squares objective

    for i in range(len(data)):
        data_i = data.iloc[i]
        p = float(data_i['p'])
        w_m = float(data_i['w_m'])
        # rho_m = float(data_i['rho_m'])
        u = float(data_i['u'])
        p_s = float(data_i['p_s'])

        # alpha = float(data_i['alpha'])
        x_g = float(data_i['x_g'])
        rho_g = float(data_i['rho_g'])
        rho_l = float(data_i['rho_l'])

        # g1 = w_m ** 2 - choke.mass_flow_rate_squared(u, p, p_s, rho_m)
        g1 = w_m - choke.mass_flow_rate(u, p, p_s, x_g, rho_g, rho_l)  # Choke model with correction

        ls_objective += g1**2

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
    return x_opt[0]  # Optimal K_c


