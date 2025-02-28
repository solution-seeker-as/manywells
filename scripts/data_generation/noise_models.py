"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 13 January 2025
Erlend Lundby, erlend@solutionseeker.no

Simple noise model that adds noise to data
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import manywells.pvt as pvt


def load_data(dataset_name, version):
    """
    Load synthetic dataset
    :param dataset_name:
    :param version:
    :return:
    """
    path = os.path.join('../../data/synth', version + '/' + dataset_name + '.csv')
    df = pd.read_csv(path)
    return df

def load_config(dataset_name, version):
    """
    Load config for dataset
    :param dataset_name:
    :param version:
    :return:
    """
    path = os.path.join('../../data/synth', version + '/' + dataset_name + '_config.csv')
    df = pd.read_csv(path)
    return df

cols = ['CHK', 'PBH', 'PWH', 'PDC', 'TBH', 'TWH',
        'WGL', 'WGAS', 'WLIQ', 'WOIL', 'WWAT', 'WTOT',
        'QGL', 'QGAS', 'QLIQ', 'QOIL', 'QWAT', 'QTOT',
        'CHOKED']


'''
rho_g = pvt.gas_density(sim.wp.R_s)
        q_g = w_g / rho_g  # Including lift gas
        q_lg = w_lg / rho_g
        q_l = w_l / sim.wp.rho_l
        q_o = w_o / well.oil.rho
        q_w = w_w / well.water.rho
        q_tot = q_g + q_l
'''


def add_noise(df, config, percentage_noise=0.1):
    """

    :param df:
    :param noise_cols:
    :param percentage_noise:
    :return:
    """
    noise_cols = ['PBH', 'PWH', 'PDC', 'TBH', 'TWH',
                  'WGL', 'WGAS', 'WLIQ', 'WOIL', 'WWAT', 'WTOT',
                  'QGL', 'QGAS', 'QLIQ', 'QOIL', 'QWAT', 'QTOT']

    derived_w_cols = ['WLIQ', 'WTOT']
    derived_q_cols = ['QGL', 'QGAS', 'QLIQ', 'QOIL', 'QWAT', 'QTOT']
    derived_cols = derived_w_cols + derived_q_cols

    noise_df = init_noise_df(df, noise_cols)
    independent_cols = list(set(noise_cols).difference(derived_cols))
    for col in independent_cols:
        mean_value = df[col].mean()
        std_dev = (percentage_noise / 100.0) * mean_value
        noise_df[col] = np.random.normal(0, std_dev, df[col].shape)

    for col in derived_w_cols:
        if col == 'WTOT':
            noise_df[col] = noise_df['WGL'] + noise_df['WGAS'] + noise_df['WOIL'] + noise_df['WWAT']
        elif col == 'WLIQ':
            noise_df[col] = noise_df['WOIL'] + noise_df['WWAT']

    for col in derived_q_cols:
        for id in df['ID'].unique():
            c_i = config[config['ID'] == id]
            #df_i = df[df['ID'] == id]
            rho_g = pvt.gas_density(c_i['wp.R_s'].values[0])
            rho_l = c_i['wp.rho_l'].values[0]
            rho_oil = c_i['oil.rho'].values[0]
            rho_water = c_i['water.rho'].values[0]
            mask = noise_df['ID'] == id
            if col == 'QGL':
                noise_df.loc[mask,col] = noise_df.loc[mask,'WGL'] / rho_g * 3600
            if col == 'QGAS':
                noise_df.loc[mask,col] = noise_df.loc[mask,'WGAS'] / rho_g * 3600
            if col == 'QLIQ':
                noise_df.loc[mask,col] = noise_df.loc[mask,'WLIQ'] / rho_l * 3600
            elif col == 'QOIL':
                noise_df.loc[mask,col] = noise_df.loc[mask,'WOIL'] / rho_oil * 3600
            elif col == 'QWAT':
                noise_df.loc[mask,col] = noise_df.loc[mask,'WWAT'] / rho_water * 3600
            elif col == 'QTOT':
                noise_df.loc[mask,col] = noise_df.loc[mask,'WTOT'] / rho_g * 3600


    df_copy = df.copy()

    df_copy[noise_cols] += noise_df[noise_cols]
    return df_copy, noise_df

def save_data(df, dataset_name, version):
    """
    Save synthetic dataset
    :param df:
    :param dataset_name:
    :param version:
    :return:
    """
    path = os.path.join('../../data/synth', version + '/' + dataset_name + '.csv')
    df.to_csv(path, index=False)

def add_noise_and_save(dataset_name, version):
    df = load_data(dataset_name, version)
    df_noisy = add_noise(df)
    noisy_name = dataset_name + '_noisy'
    save_data(df_noisy, noisy_name, version)

def init_noise_df(df, cols):
    id_col = df['ID'].copy()
    zero_df = pd.DataFrame(0, index=id_col.index,columns=cols)
    noise_df = pd.concat([id_col, zero_df], axis=1)
    return noise_df
#pvt.gas_density(sim.wp.R_s)

if __name__=='__main__':
    version = 'v9'
    dataset_name = 'synth_v9'
    # add_noise_and_save(dataset_name, version)
    config = load_config(dataset_name, version)
    print(config.loc[config['ID']==0,'wp.R_s'].iloc[0])
    df = load_data(dataset_name, version)
    df_new,noise_df = add_noise(df, config)
    w_zero = noise_df['WLIQ'] - noise_df['WWAT'] - noise_df['WOIL']
    q_zero = noise_df['QLIQ'] - noise_df['QWAT'] - noise_df['QOIL']
    df = pd.DataFrame()
    noise_df.drop(columns=['ID'], inplace=True)
    df__q = pd.DataFrame(q_zero, columns=['q_zero'])
    df__w = pd.DataFrame(w_zero, columns=['w_zero'])
    noise_df = pd.concat([noise_df, df__q], axis=1)
    noise_df = pd.concat([noise_df, df__w], axis=1)
    noise_df.plot(subplots=True, layout=(4, 5))

    plt.show()

