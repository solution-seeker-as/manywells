"""
Multi-solution variant of generate_open_loop_nonstationary_well_data.py.

This script keeps the *identical* data-generation logic of
``generate_open_loop_nonstationary_well_data.py`` (same non-stationary well
evolution over time, same derived quantities, same per-well acceptance
criteria), but swaps the CasADi/Ipopt simulator for the Rust simulator
(``manywells_rs``). The Rust ``simulate()`` returns *all* steady operating points
for a given set of boundary conditions (a well can have more than one), so every
accepted week now contributes one row per solution instead of a single row.

This mirrors ``open_loop_stationary/generate_well_data_multisolution.py``: the
solutions of a sample are sorted by ascending bottomhole pressure and tagged with
two extra columns:

    sample_id        Identifies the operating-condition sample (one per accepted
                     week). Ranges over [0, n_sim). Combine with the well ID
                     assigned by read_dump.py to get a globally unique key
                     (ID, sample_id). The week index is also stored in WEEKS.
    solution_number  Index of the solution within a sample, assigned after
                     sorting the sample's solutions by ascending bottomhole
                     pressure PBH = p(z=0). solution_number == 0 is therefore the
                     lowest-PBH operating point and there is exactly one such row
                     per sample.

Because ``n_sim`` counts *operating-condition samples* / accepted weeks (as in
the original, where each accepted week produced exactly one row), the per-well
acceptance criteria are evaluated on the primary-solution subset
(``solution_number == 0``), i.e. one row per sample -- this reproduces the
original single-solution-per-sample statistics exactly.

Two notes on faithfulness to the original:

1. The Rust solver is a global shooting method and *ignores* the warm-start
   ``x_guess`` (see manywells_rs docstring), so the original ``InitGuess``
   machinery (and the initial choke-ramp that only served to warm-start CasADi)
   has no effect on the output and is dropped. The max-production discard, which
   *is* a real acceptance criterion, is preserved.
2. Only the open-loop case (``feedback == False``) is generated, exactly as in
   the original ``multiprocessing_data_generation_ol``.
"""

import os
import sys
import time
import uuid
import pickle
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this file directly (e.g. `uv run ./scripts/.../generate_open_loop_nonstationary_well_data_multisolution.py`)
# by putting the project root on sys.path so the `scripts.*` imports below resolve.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Rust simulator. Drop-in for manywells.simulator.SSDFSimulator, except that
# simulate() returns a *list* of solutions (highest bottomhole pressure first).
# SimError is re-exported as the manywells SimError, so `except SimError` below
# keeps catching simulation failures unchanged.
from manywells_rs import SSDFSimulator, SimError
import manywells.pvt as pvt
from scripts.data_generation.nonstationary_well import NonStationaryWell, sample_nonstationary_well


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


def build_data_point(sim, well: NonStationaryWell, x):
    """
    Turn a single steady-state solution ``x`` into a data-point dict.

    Mirrors the per-data-point logic of generate_open_loop_nonstationary_well_data.py
    exactly (same derived quantities and same validity checks), operating on one
    solution. The mass fractions are read from ``well.fractions``, which
    update_conditions() has mutated for the current week.

    :param sim: The simulator (holds the current wp/bc for this week)
    :param well: The non-stationary well (holds the current fractions and the
        fixed oil/water densities)
    :param x: A single flat solution returned by sim.simulate()
    :return: Tuple (status, payload):
        - ('ok', (pbh, dp))  valid data point; pbh = p(z=0) used for ordering
        - ('invalid', None)  solution violates the rate/fraction validity checks
        - ('lowflow', None)  solution is valid but total mass flow < 1.0 kg/s
    """
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
        return 'invalid', None

    # Discard solution if total mass flow rate is less than 1.0 kg/s
    if w_tot < 1.0:
        # Simulation did not fail, but solution is invalid (too low flow rate)
        return 'lowflow', None

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

    return 'ok', (pbh, dp)


def simulate_open_loop_nonstationary_well(well: NonStationaryWell, n_sim: int, dataset_version: str):
    """
    Simulate n_sim operating-condition samples (accepted weeks) for a well, storing *all* solutions

    Each accepted week contributes one row per steady-state solution found by the
    Rust simulator. The solutions of a week are sorted by ascending bottomhole
    pressure and tagged with 'sample_id', 'solution_number' and 'WEEKS'.

    :param well: NonStationaryWell object
    :param n_sim: Number of operating-condition samples (accepted weeks) to simulate
    :return: Number of rows generated (may exceed n_sim due to multiple solutions)
    """
    # Create simulator
    sim = SSDFSimulator(well.wp, well.bc)

    # Create dataframe to hold simulation results
    cols = ['CHK', 'PBH', 'PWH', 'PDC', 'TBH', 'TWH',
            'WGL', 'WGAS', 'WLIQ', 'WOIL', 'WWAT', 'WTOT',
            'QGL', 'QGAS', 'QLIQ', 'QOIL', 'QWAT', 'QTOT',
            'FGAS', 'FOIL', 'FWAT', 'CHOKED', 'FRBH', 'FRWH',
            'WEEKS', 'sample_id', 'solution_number']
    well_data = pd.DataFrame(columns=cols, dtype=np.float32)
    well_data['CHOKED'] = well_data['CHOKED'].astype(bool)
    well_data['FRBH'] = well_data['FRBH'].astype(str)
    well_data['FRWH'] = well_data['FRWH'].astype(str)
    well_data['WEEKS'] = well_data['WEEKS'].astype(int)
    well_data['sample_id'] = well_data['sample_id'].astype(int)
    well_data['solution_number'] = well_data['solution_number'].astype(int)

    # Max-production check at a fully open choke (discard low-production wells).
    # The Rust solver ignores x_guess, so the original choke-ramp warm-start is
    # unnecessary; we simulate directly at u = 1.0. simulate()[0] is the highest-
    # PBH solution, i.e. the operating point the old (CasADi) simulator converged
    # to, so this matches the original w_tot_max computation.
    try:
        sim.bc.u = 1.0
        x_init = sim.simulate()[0]
    except SimError:
        raise SimError('Initial simulation failed - discarding well')

    df_x = sim.solution_as_df(x_init)
    wg_max = sim.wp.A * df_x['alpha'].iloc[-1] * df_x['rho_g'].iloc[-1] * df_x['v_g'].iloc[-1]
    wl_max = sim.wp.A * (1 - df_x['alpha'].iloc[-1]) * df_x['rho_l'].iloc[-1] * df_x['v_l'].iloc[-1]
    w_tot_max = wg_max + wl_max
    print('MAX WTOT', w_tot_max)
    if w_tot_max < 7:
        raise SimError('Too little production, discard well ')

    x_last = x_init  # Last successful solution, stored in the dump for parity

    n_failed_sim = 0  # Number of failed simulations
    max_failures = 200  # Discard well if simulation has failed this many times
    max_consecutive_fails = 50
    count_consecutive = 0

    n_accepted = 0  # Number of accepted operating-condition samples (weeks)

    i = 0  # Current week
    i_prev = 0  # Previous week that produced an accepted sample
    while n_accepted < n_sim:

        if n_failed_sim >= max_failures:
            print('Cut off simulation after many fails, no datapts:', len(well_data))
            raise SimError('Discarding well after too many failed simulation attempts')

        if count_consecutive >= max_consecutive_fails:
            print('many consecutive fails, cut off, no datapts:', len(well_data))
            raise SimError(f'Too many consecutive fails: {count_consecutive}')

        # Evolve the well to the current week and update the simulator conditions
        well.update_conditions(i, i_prev)
        sim.wp = well.wp
        sim.bc = well.bc

        try:
            solutions = sim.simulate()  # List of steady-state solutions (>= 1)

        except SimError:
            n_failed_sim += 1  # Count failure - discard simulation
            i += 1
            count_consecutive += 1
            continue
        count_consecutive = 0

        # Build a data point for each solution of this week
        points = []  # List of (pbh, dp) for the valid solutions
        had_invalid = False
        for x in solutions:
            status, payload = build_data_point(sim, well, x)
            if status == 'ok':
                points.append(payload)
            elif status == 'invalid':
                had_invalid = True
            # 'lowflow' solutions are discarded silently (as in the original)

        if not points:
            # No valid solution for this week. As in the original, count a failure
            # only if a solution was invalid (rate/fraction check), not for the
            # low-flow discard. Do not advance the week index (retry with new
            # random conditions), mirroring the original's continue-without-i++.
            if had_invalid:
                n_failed_sim += 1
            continue

        # Sort the week's solutions by bottomhole pressure and number them
        points.sort(key=lambda t: t[0])  # Ascending PBH = p(z=0)
        sample_id = n_accepted
        for solution_number, (pbh, dp) in enumerate(points):
            dp['WEEKS'] = i
            dp['sample_id'] = sample_id
            dp['solution_number'] = solution_number

            # Add new data point to dataset
            new_dp = pd.DataFrame(dp, index=[0])  # Passing index since values are scalar
            well_data = pd.concat([well_data, new_dp], ignore_index=True)

        x_last = solutions[0]  # Highest-PBH solution of the last accepted week
        i_prev = i
        i += 1
        n_accepted += 1  # Count accepted sample (week)

    if n_accepted != n_sim and well.feedback == False:
        raise SimError('Could not simulate all data points - discarding well')

    # The per-well acceptance criteria below are evaluated on the primary-solution
    # subset (one row per sample), which reproduces the original
    # single-solution-per-sample statistics.
    primary = well_data[well_data['solution_number'] == 0]

    # A well if the choke standard deviation is much lower than the expected standard deviation
    # The variance of U ~ Uniform(a, b) is Var(U) = (1 / 12) * (b - a)**2
    expected_chk_std = np.sqrt((1 / 12) * (1 - 0.05) ** 2)
    if primary['CHK'].std() < expected_chk_std / 5 and well.feedback == False:
        raise SimError('Low choke variation detected - discarding well')

    # If a well has experienced choked flow for more than 80% of the samples, we discard it
    # (Kept commented out to match the original nonstationary script.)
    # if primary['CHOKED'].sum() > 0.8 * len(primary):
    #     raise SimError('Choked flow for more than 80% of samples - discarding well')

    # If a well has very little variation in QTOT, we discard it
    qtot = primary['QTOT'].values
    if qtot.std() / qtot.mean() < 0.05 and well.feedback == False:  # Cv less than 0.05
        raise SimError('Small variation in QTOT - discarding well')

    # Store generated data
    save_well_config_and_data(config=well, data=well_data, x_last=x_last, dataset_version=dataset_version)
    return len(well_data)


def simulate_ol_ns_many_wells(n_wells: int, n_sim_per_well: int, feedback: bool, dataset_version: str):
    """
       Simulate open loop nonstationary data for a given number of wells

       :param n_wells: number of wells to simulate
       :param n_sim_per_well: number of data points to simulate per well
       :param feedback: whether to simulate with feedback
       :param dataset_version: dataset version
       """

    seed = os.getpid()  # Using process ID as a unique seed
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
            n_data_last = simulate_open_loop_nonstationary_well(well, n_sim=n_sim_per_well,
                                                                dataset_version=dataset_version)
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


def multiprocessing_data_generation_ol(n_wells_tot, n_sim, n_processes, dataset_version):
    feedback = False
    t_tot_0 = time.time()

    wells_per_process = int(n_wells_tot / n_processes)
    rest_wells = n_wells_tot % n_processes
    # Simulate data
    pool = multiprocessing.Pool(processes=n_processes)

    async_results = []
    for i in range(n_processes):

        if i < n_processes - 1:
            # Assign wells_per_process number of wells to cpu
            async_results.append(pool.apply_async(simulate_ol_ns_many_wells,
                                                  args=(wells_per_process, n_sim, feedback, dataset_version)))
        else:
            # Assign wells_per_process + rest_wells number of wells to last cpu
            n_wells_last = wells_per_process + rest_wells
            async_results.append(pool.apply_async(simulate_ol_ns_many_wells,
                                                  args=(n_wells_last, n_sim, feedback, dataset_version)))

    results = [ar.get() for ar in async_results]
    pool.close()
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
    n_sim = 500         # Number of operating-condition samples (accepted weeks) per well

    dataset_version = 'manywells_nsol_rs-1'  # Nonstationary, Rust solver (returns all solutions)
    multiprocessing_data_generation_ol(n_wells, n_sim, n_processes, dataset_version)
