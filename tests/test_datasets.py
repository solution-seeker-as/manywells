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

TRAJECTORY_PARAMETERS = {
    'vertical': (),
    'deviated': ('kickoff_fraction', 'inclination'),
    'l_shaped': ('horizontal_length',),
}


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
        others = [
            k for k in ('kickoff_fraction', 'inclination', 'horizontal_length') if k not in TRAJECTORY_PARAMETERS[kind]
        ]
        assert np.all(np.isnan(row[others].astype(float)))
        assert row['L'] == draw.L and list(row['fractions']) == list(draw.fractions)


def test_a_dotted_name_keeps_its_dot(tmp_path):
    """The suffixes are appended: sol-v1.0 and sol-v1.1 are two datasets, not both sol-v1 (backlog §1.10)."""
    draws = {0: sample_well(1, 0)}
    for name, chk in (('sol-v1.0', 0.5), ('sol-v1.1', 0.6)):
        write_dataset(tmp_path / name, pd.DataFrame({'ID': [0], 'CHK': [chk]}), draws, {'name': name})
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        f'sol-v1.{k}{suffix}' for k in (0, 1) for suffix in ('.parquet', '_config.parquet', '_meta.json')
    )
    assert pd.read_parquet(tmp_path / 'sol-v1.1.parquet')['CHK'].item() == 0.6
    assert json.loads((tmp_path / 'sol-v1.0_meta.json').read_text())['name'] == 'sol-v1.0'
