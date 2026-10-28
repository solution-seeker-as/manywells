"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 28 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests of the dataset files (manywells.datasets.io).
"""

import json

import numpy as np
import pandas as pd

from manywells.datasets.io import write_dataset
from manywells.sampling.wells import sample_well

TRAJECTORY_PARAMETERS = {'vertical': (), 'deviated': ('kickoff_fraction', 'inclination'),
                         'l_shaped': ('horizontal_length',)}


def test_a_dataset_round_trips_with_every_trajectory(tmp_path):
    """
    The config table holds each well's trajectory in typed columns. As a list that mixes a string and floats,
    pyarrow could not store it, and a dataset with one deviated or L-shaped well kept only its rows.
    """
    draws = {i: sample_well(1, i) for i in range(12)}
    assert {d.trajectory[0] for d in draws.values()} == set(TRAJECTORY_PARAMETERS)
    rows = pd.DataFrame({'ID': [0, 1], 'CHK': [0.5, 0.6]})
    write_dataset(tmp_path / 'ds', rows, draws, {'seed': 1})

    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / 'ds.parquet'), rows)
    assert json.loads((tmp_path / 'ds_meta.json').read_text())['seed'] == 1
    config = pd.read_parquet(tmp_path / 'ds_config.parquet').set_index('ID')
    for i, draw in draws.items():
        kind, *prm = draw.trajectory
        row = config.loc[i]
        assert row['trajectory'] == kind
        assert [row[k] for k in TRAJECTORY_PARAMETERS[kind]] == prm
        others = [k for k in ('kickoff_fraction', 'inclination', 'horizontal_length') if k not in TRAJECTORY_PARAMETERS[kind]]
        assert np.all(np.isnan(row[others].astype(float)))
        assert row['L'] == draw.L and list(row['fractions']) == list(draw.fractions)
