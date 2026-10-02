"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 26 January 2025
Erlend Lundby, erlend@solutionseeker.no

Load a published ManyWells dataset and its config from Hugging Face (solution-seeker-as/manywells). The files are
downloaded on first use and cached by huggingface_hub.
"""
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = 'solution-seeker-as/manywells'


def _read_csv(filename):
    path = hf_hub_download(REPO_ID, 'data/' + filename, repo_type='dataset')
    return pd.read_csv(path, compression='zip')


def load_data(dataset_name):
    """The rows of a published dataset, such as 'manywells-sol-1'."""
    return _read_csv(dataset_name + '.zip')


def load_config(dataset_name):
    """The config of a published dataset: one row per well, with the parameters it was simulated with."""
    return _read_csv(dataset_name + '_config.zip')
