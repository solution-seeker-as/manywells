"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 26 January 2025
Erlend Lundby, erlend@solutionseeker.no
"""
import os
import pandas as pd


def load_data(version, dataset_name):
    df = pd.read_csv(os.path.join('../../data', version + '/' + dataset_name + '.csv'))
    return df


def load_config(version, dataset_name):
    df = pd.read_csv(os.path.join('../../data', version + '/' + dataset_name + '_config.csv'))
    return df
