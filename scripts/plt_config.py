"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 23 February 2023
Bjarne Grimstad, bjarne.grimstad@gmail.no

Default configuration of Matplotlib and Seaborn
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


default_mpl_config = {
    #'figure.figsize': (16, 9),
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 16,
}


def set_mpl_config():
    for k, v in default_mpl_config.items():
        mpl.rcParams[k] = v


def set_seaborn_config():
    sns.set_theme(
        style="whitegrid",  # white or whitegrid
        rc=default_mpl_config
    )
