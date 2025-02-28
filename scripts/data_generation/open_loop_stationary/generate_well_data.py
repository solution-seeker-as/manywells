"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 26 February 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no
"""

import os
import uuid
import typing as ty
import pickle
import time
import multiprocessing

import numpy as np
import pandas as pd

from manywells.simulator import SSDFSimulator, SimError
import manywells.pvt as pvt
from scripts.data_generation.file_utils import save_well_config_and_data
from scripts.data_generation.well import Well, sample_well


def simulate_well(well: Well, n_sim: int, dataset_version: str = None):
    """
    Simulate n_sim data points for a well
    :param well: Well object
    :param n_sim: Number of data points to simulate
    :return: None
    """
    # Create simulator
    sim = SSDFSimulator(well.wp, well.bc)

    # Create dataframe to hold simulation results
    cols = ['CHK', 'PBH', 'PWH', 'PDC', 'TBH', 'TWH',
            'WGL', 'WGAS', 'WLIQ', 'WOIL', 'WWAT', 'WTOT',
            'QGL', 'QGAS', 'QLIQ', 'QOIL', 'QWAT', 'QTOT',
            'FGAS', 'FOIL', 'FWAT', 'CHOKED', 'FRBH', 'FRWH']
    well_data = pd.DataFrame(columns=cols, dtype=np.float32)
    well_data['CHOKED'] = well_data['CHOKED'].astype(bool)
    well_data['FRBH'] = well_data['FRBH'].astype(str)
    well_data['FRWH'] = well_data['FRWH'].astype(str)

    # Compute an initial guess
    try:
        sim.bc.u = 0.5  # Start with half-open valve
        x = sim.simulate()
        sim.x_guess = x

    except SimError as e:
        # print(sim.wp)
        # print(sim.bc)
        raise SimError('Initial simulation failed - discarding well')

    n_attempts = 0  # Number of simulation attempts
    max_attempts = 5 * n_sim  # Maximum number of attempts

    n_failed_sim = 0  # Number of failed simulations
    max_failures = 100  # Discard well if simulation has failed this many times

    while len(well_data) < n_sim:

        if n_attempts >= max_attempts:
            raise SimError('Discarding well: maximum number of simulation attempts reached')
        n_attempts += 1

        if n_failed_sim >= max_failures:
            raise SimError('Discarding well: too many simulation failures')

        # Sample new well conditions
        new_well = well.sample_new_conditions()
        sim.wp = new_well.wp
        sim.bc = new_well.bc

        try:
            x = sim.simulate()
            # sim.x_guess = x

        except SimError as e:
            n_failed_sim += 1  # Count failure - discard simulation
            continue

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

        # Get oil and water mass flow rate
        f_g, f_o, f_w = new_well.fractions
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

        # Discard simulation if total mass flow rate is less than 0.1 kg/s
        if w_l + w_g < 0.1:
            # Simulation did not fail, but solution is invalid (too low flow rate)
            # n_failed_sim += 1  # Count failure - discard simulation
            continue

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

        # Add new data point to dataset
        new_dp = pd.DataFrame(dp, index=[0])  # Passing index since values are scalar
        well_data = pd.concat([well_data, new_dp], ignore_index=True)

    if len(well_data) != n_sim:
        raise SimError('Discarding well: Could not simulate all data points')

    # A well if the choke standard deviation is much lower than the expected standard deviation
    # The variance of U ~ Uniform(a, b) is Var(U) = (1 / 12) * (b - a)**2
    expected_chk_std = np.sqrt((1 / 12) * (1 - 0.05) ** 2)
    if well_data['CHK'].std() < expected_chk_std / 5:
        raise SimError('Discarding well: Low choke variation detected')

    # If a well has experienced choked flow for more than 80% of the samples, we discard it
    # well_data['critical_flow'] = well_data['PDC'] < 0.6 * well_data['PWH']
    if well_data['CHOKED'].sum() > 0.8 * len(well_data):
        raise SimError('Discarding well: Choked flow for more than 80% of samples')

    # If a well has very little variation in QTOT, we discard it
    qtot = well_data['QTOT'].values
    if qtot.std() / qtot.mean() < 0.05:  # Cv less than 0.05
        raise SimError('Discarding well: Small variation in QTOT')

    # Store generated data
    if dataset_version is not None:
        save_well_config_and_data(config=well, data=well_data, dataset_version=dataset_version)

    return well_data, well


def simulate_many_wells(n_wells, n_sim, dataset_version):
    """
    Simulate many wells
    """

    # Set random seed based on PID and time (or else, the processes will use the same seed). The seed also based on
    # time since threads may share the same PID.
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    n_success = 0  # Number of wells successfully simulated
    n_attempts = 0  # Number of attempted simulations

    # Simulate data
    while n_success < n_wells:
        n_attempts += 1
        print(f'Simulation attempt {n_attempts}...')

        # Sample new well
        try:
            well = sample_well()
        except ValueError as e:
            print(e)
            continue

        try:
            simulate_well(well, n_sim=n_sim, dataset_version=dataset_version)
        except SimError as e:
            print(e)
            continue

        n_success += 1  # Count success

    return n_success, n_attempts


if __name__ == '__main__':

    """
    Attempting to simulate 2000 wells, each with 500 data points, for a total of 1M data points
    """
    dataset_version = 'manywells-sol-1'  # Update before running script to generate a new dataset

    # Simulation settings
    n_wells = 2000      # Number of wells to simulate
    n_sim = 500         # Number of data points to simulate per well  NOTE: sim does not terminate when setting n_sim = 1 (probably because the acceptance criteria for a well are never satisfied since they are based on variances)

    # Start timer
    t0 = time.time()

    # Create process pool
    n_processes = min(max(1, multiprocessing.cpu_count() - 2), n_wells)
    pool = multiprocessing.Pool(processes=n_processes)
    print(f'Created pool of {n_processes} processes')

    wells_per_process, remainder = divmod(n_wells, n_processes)
    wells_to_simulate = [wells_per_process] * n_processes
    for i in range(remainder):
        wells_to_simulate[i] += 1  # Distribute remainder
    print('Number of wells to simulate per process:', wells_to_simulate)

    async_results = [pool.apply_async(simulate_many_wells, args=(m, n_sim, dataset_version)) for m in wells_to_simulate]
    results = [ar.get() for ar in async_results]
    pool.close()

    # Print stats
    n_success = sum([s for s, a in results])
    n_attempts = sum([a for s, a in results])
    print('Number of wells attempted:', n_attempts)
    print('Successful simulations:', n_success)
    print('Failed simulations:', n_attempts - n_success)
    print('Success rate:', 100 * n_success / n_attempts, '%')

    # Stop timer
    t1 = time.time()

    total_time = t1 - t0

    print('Time spent (sec):', total_time)

    # time_per_data_point = len(data) / total_time
    # print('Time per successful data point (sec):', time_per_data_point)
    # print('Time to simulate 1M data points (min):', time_per_data_point * 1e6 / 60)
    # print('Time to simulate 1M data points (hour):', time_per_data_point * 1e6 / 3600)  # 532 - 1200 hours



