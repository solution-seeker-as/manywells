"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests of the Distributions check, on synthetic data and on the committed reference.
"""

import json

import numpy as np
import pandas as pd
import pytest

from manywells_verify.distributions import (FEATURES, MAX_CDF_GAP, PERCENTILES, REFERENCE, compare, main, summarize,
                                            trickle_signature)


@pytest.fixture(scope='module')
def data():
    """Correlated synthetic rows, with a feature that is mostly zero (like WGL) to exercise ties."""
    rng = np.random.default_rng(0)
    n = 200_000
    base = rng.normal(size=n)
    df = pd.DataFrame({f: base * (k % 3) + rng.normal(size=n) for k, f in enumerate(FEATURES)})
    df['WGL'] = np.where(rng.random(n) < 0.8, 0.0, rng.uniform(0, 5, n))
    df['PDC'] = rng.uniform(10, 100, n)
    df['PWH'] = df['PDC'] + rng.uniform(0.5, 50, n)
    df['TWH'] = rng.normal(345, 10, n)
    return df


def test_same_data_and_subsample_pass(data):
    ref = summarize(data)
    assert compare(ref, data)['worst_cdf'][1] == pytest.approx(0.0, abs=1e-12)
    assert compare(ref, data.sample(20_000, random_state=1))['passed']


def test_trickle_spike_fails(data):
    ref = summarize(data)
    spiked = data.copy()
    rows = spiked.sample(frac=0.04, random_state=2).index
    spiked.loc[rows, 'TWH'] = 280.0                       # a low-temperature spike, as on the trickle root
    spiked.loc[rows, 'PWH'] = spiked.loc[rows, 'PDC'] + 1e-3
    result = compare(ref, spiked)
    assert not result['passed']
    assert result['worst_cdf'][0] in ('TWH', 'PWH')
    assert trickle_signature(spiked).sum() == len(rows)


def test_broken_correlation_fails(data):
    ref = summarize(data)
    shuffled = data.copy()
    shuffled['PBH'] = shuffled['PBH'].sample(frac=1.0, random_state=3).to_numpy()   # same marginal, no correlation
    result = compare(ref, shuffled)
    assert result['worst_cdf'][1] <= MAX_CDF_GAP
    assert not result['passed'] and 'PBH' in result['worst_corr'][:2]


def test_committed_reference():
    ref = json.loads(REFERENCE.read_text())
    assert set(ref['datasets']) == {'sol-1', 'nsol-1'}
    assert ref['trickle_signature']['sol-1']['rows'] == 35259
    for summary in ref['datasets'].values():
        assert summary['rows'] > 900_000
        assert np.array(summary['spearman']).shape == (len(FEATURES), len(FEATURES))
        for f in FEATURES:
            values, cdf = summary['features'][f]['values'], summary['features'][f]['cdf']
            assert len(values) == len(PERCENTILES) and np.all(np.diff(values) >= 0) and np.all(np.diff(cdf) >= 0)


def test_cli_exit_status(data, tmp_path):
    path = tmp_path / 'candidate.parquet'
    data.to_parquet(path)
    assert main([str(path), '--dataset', 'sol-1']) == 1     # synthetic rows are not ManyWells data
