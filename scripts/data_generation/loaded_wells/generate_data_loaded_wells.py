"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 18 August 2024
Erlend Lundby, erlend@solutionseeker.no
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import manywells.pvt as pvt
from manywells.inflow import InflowModel, ProductivityIndex, Vogel
from manywells.closed_loop.cl_simulator import WellProperties, BoundaryConditions, ClosedLoopWellSimulator, STD_GRAVITY, CF_PRES, SimError
from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
from manywells.slip import SlipModel
from scripts.data_generation.file_utils import save_well_config_and_data
from scripts.data_generation.nonstationary_well import NonStationaryBehavior, NonStationaryWell
import multiprocessing
import time
import random
import pickle
import uuid
import json


def load_data(dataset_name, version):
    df = pd.read_csv(os.path.join('../../data', version + '/' + dataset_name + '.csv'))
    return df


def load_simulator_from_config(config, df, feedback=False, stationary=True) -> list:
    """
    Load stored well parameters into well objects that can be simulated
    :param config: dataframe of configuration details from wells
    :param df: dataframe with simulated data from wells
    :param feedback: Boolean determining if feedback is used in coming data generation
    :param stationary: Boolean determining if wells are stationary in coming data generation
    :return: wells_list - List of well dicts, dict containing keys ID, well (object) and weeks_operated
    """
    unique_IDs = df['ID'].unique()
    wells_list = list()

    for i in range(len(unique_IDs)):
        c_i = config[config['ID'] == unique_IDs[i]]
        df_i = df[df['ID'] == unique_IDs[i]]

        f_g = c_i['fraction.gas'].values[0]
        f_o = c_i['fraction.oil'].values[0]
        f_w = c_i['fraction.water'].values[0]
        has_gas_lift = c_i['has_gas_lift'].values[0]

        # Create objects for well properties and boundary conditions
        R_s = c_i['wp.R_s'].values[0]
        cp_g = c_i['wp.cp_g'].values[0]
        rho_l = c_i['wp.rho_l'].values[0]

        # Fluids
        gas = pvt.GasProperties(
            name='gas',
            R_s=R_s,  # Specific gas constant (520 is methane, 320 is EG gas)
            cp=cp_g
        )

        water = pvt.LiquidProperties(name=c_i['water.rho'].values[0],
                                     rho=c_i['water.rho'].values[0],
                                     cp=c_i['water.cp'].values[0])  # pvt.WATER

        oil = pvt.LiquidProperties(
            name=c_i['oil.name'].values[0],
            rho=c_i['oil.rho'].values[0],  # API ranging from 40 (light) to 22 (intermediate-heavy)
            cp=c_i['oil.cp'].values[0]
        )

        wp = WellProperties(L=c_i['wp.L'].values[0], D=c_i['wp.D'].values[0], rho_l=rho_l, R_s=R_s,
                            cp_g=cp_g, cp_l=c_i['wp.cp_l'].values[0],
                            f_D=c_i['wp.f_D'].values[0], h=c_i['wp.h'].values[0])

        # Inflow model
        w_l_max = c_i['wp.inflow.w_l_max'].values[0]
        wp.inflow = Vogel(w_l_max, f_g)

        # liquid_mix = pvt.liquid_mix(oil, water, f_o / (f_o + f_w))
        # wp.rho_l = liquid_mix.rho
        # wp.cp_l = liquid_mix.cp
        # wp.inflow.f_g = f_g

        # Slip model
        slip_model = SlipModel(alpha_bubble_to_slug=c_i['wp.slip.alpha_bubble_to_slug'].values[0],
                               alpha_slug_to_annular=c_i['wp.slip.alpha_slug_to_annular'].values[0])
        wp.slip = slip_model

        # Choke model
        chk_profile = c_i['wp.choke.chk_profile'].values[0]
        K_c = c_i['wp.choke.K_c'].values[0]
        wp.choke = SimpsonChokeModel(K_c=K_c, chk_profile=chk_profile)

        bc = BoundaryConditions(T_s=c_i['bc.T_s'].values[0], T_r=c_i['bc.T_r'].values[0],
                                p_r=c_i['bc.p_r'].values[0], p_s=c_i['bc.p_s'].values[0],
                                u=c_i['bc.u'].values[0], w_lg=c_i['bc.w_lg'].values[0])

        init_frac_string = c_i['ns_bhv.init_fractions'].values[0]
        init_frac_string = init_frac_string.strip('()')
        init_frac_list = init_frac_string.split(',')
        init_fractions = tuple(float(num) for num in init_frac_list)
        ns_bhv = NonStationaryBehavior(pr_init=c_i['ns_bhv.pr_init'].values[0],
                                       ps_init=c_i['ns_bhv.ps_init'].values[0],
                                       init_fractions=init_fractions)
        ns_bhv.lifetime = c_i['ns_bhv.lifetime'].values[0]
        ns_bhv.pr_conv = c_i['ns_bhv.pr_conv'].values[0]
        ns_bhv.eps = c_i['ns_bhv.eps'].values[0]
        ns_bhv.decay_rate = c_i['ns_bhv.decay_rate'].values[0]
        ns_bhv.decay_rate_noise_factor = c_i['ns_bhv.decay_rate_noise_factor'].values[0]
        ns_bhv.init_fractions = c_i['ns_bhv.init_fractions'].values[0]
        ns_bhv.decay_g = c_i['ns_bhv.decay_g'].values[0]
        ns_bhv.decay_o = c_i['ns_bhv.decay_o'].values[0]

        well = NonStationaryWell(wp=wp, bc=bc, ns_bhv=ns_bhv, gas=gas, oil=oil, water=water, fractions=(f_g, f_o, f_w),
                                 has_gas_lift=has_gas_lift, feedback=feedback)
        a = df_i['WEEKS'].max()

        well_dict = {
            'ID': unique_IDs[i],
            'well': well,
            'weeks_operated': df_i['WEEKS'].max()
        }
        if 'x_last' in c_i:
            well_dict['x_last'] =  c_i['x_last']
        x_last = c_i['x_last']
        wells_list.append(well_dict)
        print(well_dict['x_last'])

    return wells_list


def simulate_loaded_many_wells(wells_list, n_data_per_well, dataset_version_store):
    """
    Simulate n_sim datapoints for len(well_list)
    :param wells_list:
    :param n_data_per_well:
    :param dataset_version_store:
    :return:
    """
    seed = os.getpid()  # Using process ID as a unique seed
    random.seed(seed)
    np.random.seed(seed)
    n_success = 0
    n_wells = len(wells_list)
    for i in range(n_wells):
        well_dict = wells_list[i]
        try:
            n_data_last = simulate_loaded_well(well_dict, n_data_per_well, dataset_version_store)
        except SimError as e:
            print(e)
            continue
        n_success += 1
        print('Successful wells simulated:', n_success)
        print('Datapoints in last well:', n_data_last)
    print('Successful simulations:', n_success)
    print('Failed simulations:', n_wells - n_success)


def simulate_loaded_well(well_dict, n_data, dataset_version_store, data_per_week=1):
    """
    Simulate n_data datapoints for single loaded well
    :param well_dict: keys: {ID, well (Well object), weeks_operated}
    :param n_data: Number of datapoints to simulate
    :param dataset_version_store: version to store
    :param data_per_week: Number of datapoints per week
    :return: Number of datapoints generated
    """
    well_id = int(well_dict['ID'])
    well = well_dict['well']
    week_0 = well_dict['weeks_operated']
    if 'x_last' in well_dict:
        x_guess = json.loads(well_dict['x_last'].values[0])
        if len(x_guess)%7 != 0:
            x_guess = x_guess[:-1]
    else:
        x_guess = None

    sim = ClosedLoopWellSimulator(well.wp, well.bc, feedback=well.feedback, n_cells=int(well.wp.L / 10))
    cols = ['CHK', 'PBH', 'PWH', 'PDC', 'TBH', 'TWH',
                'WGL', 'WGAS', 'WLIQ', 'WOIL', 'WWAT', 'WTOT',
                'QGL', 'QGAS', 'QLIQ', 'QOIL', 'QWAT', 'QTOT',
                'CHOKED']

    cols += ['WEEKS']
    cols += ['ID']

    well_data = pd.DataFrame(columns=cols, dtype=np.float32)
    well_data['CHOKED'] = well_data['CHOKED'].astype(bool)

    # well.update_conditions(week_0-1, week_0)
    sim.feedback = False
    sim.bc.u = 1.0
    try:
        x,_ = sim.simulate(well=well, x_gu=x_guess)
        x_guess = x
    except:
        pass

    if well.feedback:
        x_guess +=[1.0]
    df_x = sim.solution_as_df(x)
    wg_max = sim.wp.A * df_x['alpha'].iloc[-1] * df_x['rho_g'].iloc[-1] * df_x['v_g'].iloc[-1]
    wl_max = sim.wp.A * (1 - df_x['alpha'].iloc[-1]) * df_x['rho_l'].iloc[-1] * df_x['v_l'].iloc[-1]
    w_tot_max = wg_max + wl_max
    print('WTOT with CHK=1: ', w_tot_max)
    sim.feedback = well.feedback
    n_failed_sim = 0  # Number of failed simulations
    max_failures = 500  # *int(1 + (n_sim/10))  # Discard well if simulation has failed this many times
    max_consecutive_fails = 50
    count_consecutive = 0
    i = 0
    i_prev = 0
    week = week_0
    while len(well_data) < n_data or i == int(well.ns_bhv.lifetime*52):
        wtot_ref = 0.65*w_tot_max + np.random.choice([0, 1], p=[0.70, 0.30])*np.random.uniform(-0.1*w_tot_max, 0.1*w_tot_max)

        if n_failed_sim >= max_failures:
            print('Cut of simulation after many fails, no datapts:', len(well_data))
            raise SimError('Discarding well after too many failed simulation attempts')

        # Update well conditions
        well.update_conditions(i, i_prev)
        sim.wp = well.wp
        sim.bc = well.bc
        try:
            x, objective_sim = sim.simulate(wtot_ref=wtot_ref, well=well, x_gu=x_guess)
            # CC = sim.bc.u
            if objective_sim < 1e-2 or sim.bc.u>0.99:
                x_guess = x
            # sim.x_guess = x

        except SimError as e:
            n_failed_sim += 1  # Count failure - discard simulation
            i += 1
            count_consecutive += 1
            continue
        count_consecutive = 0
        i_prev = i
        i += 1

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

        # Validate data before adding
        valid_rates = w_l >= 0 and w_g >= 0
        valid_fracs = (0 <= f_g <= 1) and (0 <= f_o <= 1) and (0 <= f_w <= 1)
        if not (valid_rates and valid_fracs):
            n_failed_sim += 1  # Count failure - discard simulation
            continue

        # TODO: Check for low flow rates?
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
            'CHOKED': choked,
        }

        week += (i / data_per_week)

        dp.update({'WEEKS': week})
        dp.update({'ID': well_id})

        new_dp = pd.DataFrame(dp, index=[0])  # Passing index since values are scalar
        well_data = pd.concat([well_data, new_dp], ignore_index=True)
    save_well_config_and_data(config=well, data=well_data, dataset_version=dataset_version_store)  # TODO: Not storing x_last here. I guess that means that we can only load and generate more data once?
    return len(well_data)


def multiprocess_loaded_wells(dataset_version='v1_multiprocess_feedback', new_dataset_version_ending='test',
                              feedback=False, n_data_per_well=100, n_cpu=10):
    """
    Load wells and distribute over CPU's for simulation
    Send simulation tasks to simulate_loaded_many_wells(), which run in parallel on different CPU's.

    :param dataset_version: Dataset version in which well parameters are loaded from
    :param new_dataset_name_ending: Additional name for dataset to be generated
    :param feedback: Boolean determining if feedback is used in coming data generation
    :param n_cpu: Number of parallel cpu's running simulations
    :return:
    """
    dataset_name = 'synth_' + dataset_version
    new_dataset_version = dataset_version + '_' + new_dataset_version_ending
    config_name = dataset_name + '_config'
    df = load_data(dataset_name=dataset_name, version=dataset_version)
    config = load_data(config_name, version=dataset_version)
    wells_list = load_simulator_from_config(config, df, feedback=feedback)

    n_wells = len(wells_list)
    wells_per_cpu = int(n_wells/n_cpu)
    #rest = n_wells - (wells_per_cpu*n_cpu)
    t_tot_0 = time.time()
    #Multiprocessing
    pool = multiprocessing.Pool(processes=n_cpu)
    async_results = []
    for i in range(n_cpu):
        if i<n_cpu-1:
            wells_cpu = wells_list[i*wells_per_cpu:(i+1)*wells_per_cpu]
            async_results.append(pool.apply_async(simulate_loaded_many_wells,
                                                  args=(wells_cpu,n_data_per_well,new_dataset_version)))
        else:
            wells_cpu = wells_list[i*wells_per_cpu:]
            async_results.append(pool.apply_async(simulate_loaded_many_wells,
                                                  args=(wells_cpu, n_data_per_well, new_dataset_version)))

    results = [ar.get() for ar in async_results]
    t_tot_1 = time.time()
    tot_time = t_tot_1 - t_tot_0
    print('Total time spent:', tot_time)


def sim_loaded_wells_failed(dataset_version='v1_multiprocess_feedback', idx=[], feedback=False,
                            n_data_per_well=10, new_dataset_version_ending='test_former_failed_wells'):
    dataset_name = 'synth_' + dataset_version
    new_dataset_version = dataset_version + '_' + new_dataset_version_ending
    config_name = dataset_name + '_config'
    df = load_data(dataset_name=dataset_name, version=dataset_version)
    df = df[df['ID'].isin(idx)]
    config = load_data(config_name, version=dataset_version)
    config = config[config['ID'].isin(idx)]
    wells_list = load_simulator_from_config(config, df, feedback=feedback)
    simulate_loaded_many_wells(wells_list=wells_list, n_data_per_well=n_data_per_well, dataset_version_store=new_dataset_version)



if __name__ =='__main__':

    dataset_version = 'manywells-nscl-1'  # Folder
    dataset_name = 'manywells-nscl-1'  # File
    multiprocess_loaded_wells(dataset_version=dataset_version, feedback=False, n_data_per_well=50,
                              new_dataset_version_ending='testset', n_cpu=10)
