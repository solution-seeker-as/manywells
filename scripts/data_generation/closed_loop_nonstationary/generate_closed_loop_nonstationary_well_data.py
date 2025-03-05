"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 8 August 2024
Erlend Lundby, erlend@solutionseeker.no
"""

import os
import time
import uuid
import pickle
# import random
import multiprocessing

import numpy as np
import pandas as pd

from manywells.closed_loop.cl_simulator import ClosedLoopWellSimulator, SimError
import manywells.pvt as pvt
from scripts.data_generation.nonstationary_well import NonStationaryWell, sample_nonstationary_well
from scripts.data_generation.init_utils import InitGuess


def save_object(obj, filename):
    with open(filename, 'wb') as f:  # Overwrites any existing file.
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_well_config_and_data(config, data, x_last, dataset_version):
    obj = {
        'config': config,
        'data': data,
        'last_x': x_last
    }

    script_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(script_path, '../../../data', dataset_version, 'dump')
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    fn = str(uuid.uuid4())  # uuid4 could not say if it is thread safe so running loop
    while os.path.isfile(os.path.join(data_path, fn)):
        fn = str(uuid.uuid4())

    save_object(obj, os.path.join(data_path, fn))



def simulate_closed_loop_nonstationary_well(well: NonStationaryWell, n_sim: int, dataset_version: str):
    """
    Simulate n_sim data points for a well
    :param well: NonStationaryWell object
    :param n_sim: Number of data points to simulate
    :return: None
    """

    sim = ClosedLoopWellSimulator(well.wp, well.bc, feedback=well.feedback, n_cells=int(well.wp.L/10))
    cols = ['CHK', 'PBH', 'PWH', 'PDC', 'TBH', 'TWH',
            'WGL', 'WGAS', 'WLIQ', 'WOIL', 'WWAT', 'WTOT',
            'QGL', 'QGAS', 'QLIQ', 'QOIL', 'QWAT', 'QTOT',
            'FGAS', 'FOIL', 'FWAT', 'CHOKED', 'FRBH', 'FRWH']

    cols += ['WEEKS']  # New column to track time
    well_data = pd.DataFrame(columns=cols, dtype=np.float32)
    well_data['CHOKED'] = well_data['CHOKED'].astype(bool)
    well_data['FRBH'] = well_data['FRBH'].astype(str)
    well_data['FRWH'] = well_data['FRWH'].astype(str)

    # Compute an initial gues
    x_guess = None
    for i in range(10):
        try:
            sim.feedback=False
            sim.bc.u = 0.1 + 0.1*i  # Start with half-open valve
            x,_ = sim.simulate(well=well, x_gu=x_guess)
            x_guess = x
            sim.x_guess = x

        except SimError as e:
            raise SimError('Initial simulation failed - discarding well')

    sim.bc.u = 1.0

    x,_ = sim.simulate(well=well, x_gu =x_guess)
    x_guess = x

    if well.feedback:
        x_guess += [1.0]

    df_x = sim.solution_as_df(x)
    wg_max = sim.wp.A * df_x['alpha'].iloc[-1] * df_x['rho_g'].iloc[-1] * df_x['v_g'].iloc[-1]
    wl_max = sim.wp.A * (1 - df_x['alpha'].iloc[-1]) * df_x['rho_l'].iloc[-1] * df_x['v_l'].iloc[-1]
    w_tot_max = wg_max + wl_max
    f_g, f_o, f_w = well.fractions
    wlf = f_w / (f_o + f_w)  # Water to liquid fraction
    w_w = wl_max * wlf
    w_o = wl_max * (1 - wlf)
    rho_g = pvt.gas_density(sim.wp.R_s)
    q_g = wg_max / rho_g  # Including lift gas

    q_l = wl_max / sim.wp.rho_l
    q_o = w_o / well.oil.rho
    q_w = w_w / well.water.rho
    q_tot = q_g + q_l
    q_tot *= 3600
    q_l *= 3600
    #print('MAX  QLIQ:', q_l)
    print('MAX WTOT', w_tot_max)

    if w_tot_max < 7.0:
        raise SimError('Too little production, discard well ')


    init_guess = InitGuess()
    init_guess.add_candidate(x_guess,sim,well)
    # wtot_max = wg_max + wl_max

    sim.feedback = well.feedback
    n_failed_sim = 0  # Number of failed simulations
    max_failures = 200  # *int(1 + (n_sim/10))  # Discard well if simulation has failed this many times
    max_consecutive_fails = 50
    count_consecutive = 0

    i = 0
    i_prev = 0
    pr = list()
    #x_guess = None
    while len(well_data) < n_sim or i == int(well.ns_bhv.lifetime*52):

        wtot_ref = 0.85*w_tot_max + np.random.choice([0, 1], p=[0.70, 0.30])*np.random.uniform(-0.1*w_tot_max, 0.1*w_tot_max)
        #print('-----')
        #print('WTOT_ref:',wtot_ref)
        if n_failed_sim >= max_failures:
            print('Cut of simulation after many fails, no datapts:', len(well_data))
            # Store generated data
            #save_well_config_and_data(config=well, data=well_data, dataset_version=dataset_version)
            #return len(well_data)
            raise SimError('Discarding well after too many failed simulation attempts')

        if count_consecutive>= max_consecutive_fails:
            print('many consecutive fails, cut of, no datapts:', len(well_data))
            # Store generated data
            #save_well_config_and_data(config=well, data=well_data, dataset_version=dataset_version)
            #return len(well_data)
            raise SimError('To many consecutive fails: ', count_consecutive)


        well.update_conditions(i, i_prev)
        sim.wp = well.wp
        sim.bc = well.bc

        try:
            x_guess = init_guess.closest_candidate(well)
            x, objective_sim = sim.simulate(wtot_ref=wtot_ref, well=well, x_gu=x_guess)

            # CC = sim.bc.u
            if objective_sim < 1e-2 or sim.bc.u>0.99:
                init_guess.add_candidate(x, sim, well)
            #x_guess = x
            # sim.x_guess = x

        except SimError as e:
            n_failed_sim += 1  # Count failure - discard simulation
            i += 1
            count_consecutive += 1
            continue
        count_consecutive = 0

        pr += [well.bc.p_r]

        # Prepare new data point
        df_x = sim.solution_as_df(x)

        df_x['w_g'] = sim.wp.A * df_x['alpha'] * df_x['rho_g'] * df_x['v_g']
        df_x['w_l'] = sim.wp.A * (1 - df_x['alpha']) * df_x['rho_l'] * df_x['v_l']

        pbh = float(df_x['p'].iloc[0])
        pwh = float(df_x['p'].iloc[-1])
        twh = float(df_x['T'].iloc[-1])
        w_g = float(df_x['w_g'].iloc[-1])  # Including lift gas
        w_l = float(df_x['w_l'].iloc[-1])
        w_tot = w_g + w_l
        w_lg = sim.bc.w_lg

        # Get fractions
        f_g, f_o, f_w = well.fractions

        wlf = f_w / (f_o + f_w)  # Water to liquid fraction
        w_w = w_l * wlf
        w_o = w_l * (1 - wlf)
        # Volumetric flow rates (at standard reference conditions) in Sm³/s
        rho_g = pvt.gas_density(sim.wp.R_s)
        q_g = w_g / rho_g  # Including lift gas
        q_lg = w_lg / rho_g
        q_l = w_l / sim.wp.rho_l
        q_o = w_o / well.oil.rho
        q_w = w_w / well.water.rho
        q_tot = q_g + q_l
        #print('WTOT:',w_tot)
        #print('-----')
        # assert abs(q_l - (q_o + q_w)) < 1e-5, f'Liquids do not sum: q_l = {q_l}, q_o + q_w = {q_o + q_w}'

        # Convert volumetric flow rates from Sm³/s to Sm³/h
        SECONDS_PER_HOUR = 3600
        q_g *= SECONDS_PER_HOUR
        q_lg *= SECONDS_PER_HOUR
        q_l *= SECONDS_PER_HOUR
        q_o *= SECONDS_PER_HOUR
        q_w *= SECONDS_PER_HOUR
        q_tot *= SECONDS_PER_HOUR

        # Choked flow?
        choked = sim.wp.choke.is_choked(pwh, sim.bc.p_s)

        # Flow regime at top and bottom of well
        regime_wh = str(df_x['flow-regime'].iloc[-1])
        regime_bh = str(df_x['flow-regime'].iloc[0])

        # Validate data before adding
        valid_rates = w_l >= 0 and w_g >= 0
        valid_fracs = (0 <= f_g <= 1) and (0 <= f_o <= 1) and (0 <= f_w <= 1)
        if not (valid_rates and valid_fracs):
            n_failed_sim += 1  # Count failure - discard simulation
            continue
        if w_tot < 1.0:
            continue # Do not store datapoint of low flow rate

        #print('CHK:', sim.bc.u)
        #print('W_GL:', sim.bc.w_lg)
        # Structure data point in dict
        dp = {
            'CHK': sim.bc.u,
            'PBH': pbh,
            'PWH': pwh,
            'PDC': sim.bc.p_s,
            'TBH': sim.bc.T_r,
            'TWH': twh,
            'WGL': w_lg,
            'WGAS': w_g - w_lg,  # Excluding lift gas
            'WLIQ': w_l,
            'WOIL': w_o,
            'WWAT': w_w,
            'WTOT': w_tot,  # Total mass flow, including lift gas
            'QGL': q_lg,
            'QGAS': q_g - q_lg,  # Excluding lift gas
            'QLIQ': q_l,
            'QOIL': q_o,
            'QWAT': q_w,
            'QTOT': q_tot,  # Total volumetric flow, including lift gas
            'FGAS': f_g,  # Inflow gas mass fraction (WGAS / (WTOT - WGL))
            'FOIL': f_o,  # Inflow oil mass fraction (WOIL / (WTOT - WGL))
            'FWAT': f_w,  # Inflow water mass fraction (WWAT / (WTOT - WGL))
            'CHOKED': choked,
            'FRBH': regime_bh,  # Flow regime at bottomhole
            'FRWH': regime_wh,  # Flow regime at wellhead
        }

        dp.update({'WEEKS':i})
        i_prev = i
        i += 1

        # Add new data point to dataset
        new_dp = pd.DataFrame(dp, index=[0])  # Passing index since values are scalar
        well_data = pd.concat([well_data, new_dp], ignore_index=True)

    if len(well_data) != n_sim and well.feedback==False:
        raise SimError('Could not simulate all data points - discarding well')

    # A well if the choke standard deviation is much lower than the expected standard deviation
    # The variance of U ~ Uniform(a, b) is Var(U) = (1 / 12) * (b - a)**2
    expected_chk_std = np.sqrt((1 / 12) * (1 - 0.05) ** 2)
    if well_data['CHK'].std() < expected_chk_std / 5 and well.feedback==False:
        raise SimError('Low choke variation detected - discarding well')

    # If a well has experienced choked flow for more than 80% of the samples, we discard it
    # well_data['critical_flow'] = well_data['PDC'] < 0.6 * well_data['PWH']
    # if well_data['CHOKED'].sum() > 0.8 * len(well_data):
    #     raise SimError('Choked flow for more than 80% of samples - discarding well')

    # If a well has very little variation in QTOT, we discard it
    qtot = well_data['QTOT'].values
    if qtot.std() / qtot.mean() < 0.05 and well.feedback==False:  # Cv less than 0.05
        raise SimError('Small variation in QTOT - discarding well')

    # Store generated data
    x_last = x
    save_well_config_and_data(config=well, data=well_data, x_last=x_last,  dataset_version=dataset_version)
    return len(well_data)


def simulate_cl_ns_many_wells(n_wells: int, n_sim_per_well: int, feedback: bool, dataset_version: str):
    """
    Simulate closed loop, nonstationary data a given number of wells

    :param n_wells: number of wells to simulate
    :param n_sim_per_well: number of data points to simulate per well
    :param feedback: whether to simulate with feedback
    :param dataset_version: dataset version
    """

    seed = os.getpid()  # Using process ID as a unique seed
    # random.seed(seed)  # TODO: Are we using the built-in 'random' package?
    np.random.seed(seed)

    # Start timer
    t0 = time.time()
    n_success = 0
    n_attempts = 0
    while n_success < n_wells:

        n_attempts += 1
        print(f'Simulation attempt {n_attempts}...')

        # Sample new non-stationary well
        try:
            well = sample_nonstationary_well(feedback=feedback)
        except ValueError as e:
            print(e)
            continue

        try:
            n_data_last = simulate_closed_loop_nonstationary_well(well, n_sim=n_sim_per_well, dataset_version=dataset_version)
        except SimError as e:
            print(e)
            continue

        n_success += 1  # Count success
        print('Successful wells simulated:', n_success)
        print('Datapoints in last well:', n_data_last)

    print('Number of wells attempted:', n_attempts)
    print('Successful simulations:', n_success)
    print('Failed simulations:', n_attempts - n_success)

    # Stop timer
    t1 = time.time()

    total_time = t1 - t0

    print('Time spent (sec):', total_time)


def multiprocessing_data_generation_cl(n_wells_tot, n_sim, n_processes, dataset_version):
    feedback = True
    t_tot_0 = time.time()

    wells_per_process = int(n_wells_tot / n_processes)
    rest_wells = n_wells_tot%n_processes
    # Simulate data
    pool = multiprocessing.Pool(processes=n_processes)

    async_results = []
    for i in range(n_processes):

        if i < n_processes-1:
            # Assign wells_per_process number of wells to cpu
            async_results.append(pool.apply_async(simulate_cl_ns_many_wells,
                                              args=(wells_per_process, n_sim, feedback, dataset_version)))
        else:
            # Assign wells_per_process + rest_wells number of wells to last cpu
            n_wells_last = wells_per_process + rest_wells
            async_results.append(pool.apply_async(simulate_cl_ns_many_wells,
                                                  args=(n_wells_last, n_sim, feedback, dataset_version)))

    results = [ar.get() for ar in async_results]
    t_tot_1 = time.time()
    tot_time = t_tot_1 - t_tot_0
    print('Total time spent:', tot_time)


if __name__ == '__main__':
    """
    Attempting to simulate wells
    """

    # Simulation settings
    n_processes = 10    # multiprocessing.cpu_count()
    n_wells = 2000  # Number of wells to simulate
    n_sim = 500         # Number of data points to simulate per well

    dataset_version = 'manywells-nscl-1'
    multiprocessing_data_generation_cl(n_wells, n_sim, n_processes, dataset_version)

