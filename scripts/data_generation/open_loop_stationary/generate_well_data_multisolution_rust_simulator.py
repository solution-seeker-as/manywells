"""
Multi-solution variant of generate_well_data.py.

This script keeps the *identical* data-generation logic of
``generate_well_data.py`` (same well sampling, same derived quantities, same
per-well acceptance criteria), but swaps the CasADi/Ipopt simulator for the
Rust simulator (``manywells_rs``). The Rust ``simulate()`` returns *all* steady
operating points for a given set of boundary conditions (a well can have more
than one), so every accepted operating-condition sample now contributes one row
per solution instead of a single row.

To differentiate the solutions, two columns are added to the per-well data:

    sample_id        Identifies the operating-condition sample (shared inputs:
                     CHK, PDC, TBH, fractions, ...). Ranges over [0, n_sim).
                     Combine with the well ID assigned by read_dump.py to get a
                     globally unique key (ID, sample_id).
    solution_number  Index of the solution within a sample, assigned after
                     sorting the sample's solutions by ascending bottomhole
                     pressure PBH = p(z=0). solution_number == 0 is therefore the
                     lowest-PBH operating point and there is exactly one such row
                     per sample.

Because ``n_sim`` counts *operating-condition samples* (as in the original,
where each sample produced exactly one row), the per-well acceptance criteria
are evaluated on the primary-solution subset (``solution_number == 0``), i.e.
one row per sample -- this reproduces the original single-solution-per-sample
statistics exactly.
"""

import os
import sys
import uuid
import typing as ty
import pickle
import time
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this file directly (e.g. `uv run ./scripts/.../generate_well_data_multisolution.py`)
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
from scripts.data_generation.file_utils import save_well_config_and_data
from scripts.data_generation.well import Well, sample_well


def build_data_point(sim, well: Well, new_well: Well, x):
    """
    Turn a single steady-state solution ``x`` into a data-point dict.

    Mirrors the per-data-point logic of generate_well_data.py exactly (same
    derived quantities and same validity checks), operating on one solution.

    :param sim: The simulator (holds the current wp/bc for this sample)
    :param well: The base well (used for the fixed oil/water densities)
    :param new_well: The well with resampled conditions (used for the fractions)
    :param x: A single flat solution returned by sim.simulate()
    :return: Tuple (status, payload):
        - ('ok', (pbh, dp))  valid data point; pbh = p(z=0) used for ordering
        - ('invalid', None)  solution violates the rate/fraction validity checks
        - ('lowflow', None)  solution is valid but total mass flow < 0.1 kg/s
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

    # Discard solution if total mass flow rate is less than 0.1 kg/s
    if w_l + w_g < 0.1:
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


def simulate_well(well: Well, n_sim: int, dataset_version: str = None):
    """
    Simulate n_sim operating-condition samples for a well, storing *all* solutions

    Each accepted sample contributes one row per steady-state solution found by
    the Rust simulator. The solutions of a sample are sorted by ascending
    bottomhole pressure and tagged with 'sample_id' and 'solution_number'.

    :param well: Well object
    :param n_sim: Number of operating-condition samples to simulate
    :return: None
    """
    # Create simulator
    sim = SSDFSimulator(well.wp, well.bc)

    # Create dataframe to hold simulation results
    cols = ['CHK', 'PBH', 'PWH', 'PDC', 'TBH', 'TWH',
            'WGL', 'WGAS', 'WLIQ', 'WOIL', 'WWAT', 'WTOT',
            'QGL', 'QGAS', 'QLIQ', 'QOIL', 'QWAT', 'QTOT',
            'FGAS', 'FOIL', 'FWAT', 'CHOKED', 'FRBH', 'FRWH',
            'sample_id', 'solution_number']
    well_data = pd.DataFrame(columns=cols, dtype=np.float32)
    well_data['CHOKED'] = well_data['CHOKED'].astype(bool)
    well_data['FRBH'] = well_data['FRBH'].astype(str)
    well_data['FRWH'] = well_data['FRWH'].astype(str)
    well_data['sample_id'] = well_data['sample_id'].astype(int)
    well_data['solution_number'] = well_data['solution_number'].astype(int)

    # Compute an initial guess
    try:
        sim.bc.u = 0.5  # Start with half-open valve
        x = sim.simulate()[0]  # simulate() returns a list of solutions (highest PBH first)
        sim.x_guess = x

    except SimError as e:
        # print(sim.wp)
        # print(sim.bc)
        raise SimError('Initial simulation failed - discarding well')

    n_attempts = 0  # Number of simulation attempts
    max_attempts = 5 * n_sim  # Maximum number of attempts

    n_failed_sim = 0  # Number of failed simulations
    max_failures = 100  # Discard well if simulation has failed this many times

    n_accepted = 0  # Number of accepted operating-condition samples

    while n_accepted < n_sim:

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
            solutions = sim.simulate()  # List of steady-state solutions (>= 1)

        except SimError as e:
            n_failed_sim += 1  # Count failure - discard simulation
            continue

        # Build a data point for each solution of this sample
        points = []  # List of (pbh, dp) for the valid solutions
        had_invalid = False
        for x in solutions:
            status, payload = build_data_point(sim, well, new_well, x)
            if status == 'ok':
                points.append(payload)
            elif status == 'invalid':
                had_invalid = True
            # 'lowflow' solutions are discarded silently (as in the original)

        if not points:
            # No valid solution for this sample. Count as a failure only if a
            # solution was invalid (matching the original, which counted the
            # rate/fraction validity failure but not the low-flow discard).
            if had_invalid:
                n_failed_sim += 1
            continue

        # Sort the sample's solutions by bottomhole pressure and number them
        points.sort(key=lambda t: t[0])  # Ascending PBH = p(z=0)
        sample_id = n_accepted
        for solution_number, (pbh, dp) in enumerate(points):
            dp['sample_id'] = sample_id
            dp['solution_number'] = solution_number

            # Add new data point to dataset
            new_dp = pd.DataFrame(dp, index=[0])  # Passing index since values are scalar
            well_data = pd.concat([well_data, new_dp], ignore_index=True)

        n_accepted += 1  # Count accepted sample

    if n_accepted != n_sim:
        raise SimError('Discarding well: Could not simulate all data points')

    # The per-well acceptance criteria below are evaluated on the primary-solution
    # subset (one row per sample), which reproduces the original
    # single-solution-per-sample statistics.
    primary = well_data[well_data['solution_number'] == 0]

    # A well if the choke standard deviation is much lower than the expected standard deviation
    # The variance of U ~ Uniform(a, b) is Var(U) = (1 / 12) * (b - a)**2
    expected_chk_std = np.sqrt((1 / 12) * (1 - 0.05) ** 2)
    if primary['CHK'].std() < expected_chk_std / 5:
        raise SimError('Discarding well: Low choke variation detected')

    # If a well has experienced choked flow for more than 80% of the samples, we discard it
    # well_data['critical_flow'] = well_data['PDC'] < 0.6 * well_data['PWH']
    if primary['CHOKED'].sum() > 0.8 * len(primary):
        raise SimError('Discarding well: Choked flow for more than 80% of samples')

    # If a well has very little variation in QTOT, we discard it
    qtot = primary['QTOT'].values
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
    Attempting to simulate 2000 wells, each with 500 operating-condition samples.
    Note: with the Rust simulator each sample may yield more than one solution, so
    the total number of rows per well can exceed n_sim.
    """
    dataset_version = 'manywells-sol_rs-1'  # Update before running script to generate a new dataset

    # Simulation settings
    n_wells = 2000      # Number of wells to simulate
    n_sim = 500         # Number of operating-condition samples to simulate per well  NOTE: sim does not terminate when setting n_sim = 1 (probably because the acceptance criteria for a well are never satisfied since they are based on variances)

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
