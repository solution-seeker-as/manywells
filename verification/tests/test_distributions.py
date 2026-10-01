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

from manywells_verify.distributions import (FEATURES, MAX_CDF_GAP, MAX_CORR_GAP, PERCENTILES, REFERENCE, compare,
                                            main, summarize, trickle_signature, weighted_spearman, well_weights)


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
        assert sum(summary['well_rows'].values()) == summary['rows']
        assert np.array(summary['spearman']).shape == (len(FEATURES), len(FEATURES))
        for f in FEATURES:
            values, cdf = summary['features'][f]['values'], summary['features'][f]['cdf']
            assert len(values) == len(PERCENTILES) and np.all(np.diff(values) >= 0) and np.all(np.diff(cdf) >= 0)


def test_cli_exit_status(data, tmp_path):
    path = tmp_path / 'candidate.parquet'
    data.assign(ID=np.arange(len(data)) % 2000).to_parquet(path)
    assert main([str(path), '--dataset', 'sol-1']) == 1     # synthetic rows are not ManyWells data


def test_cli_needs_well_ids(data, tmp_path):
    path = tmp_path / 'candidate.parquet'
    data.to_parquet(path)
    with pytest.raises(ValueError, match='ID column'):
        main([str(path), '--dataset', 'sol-1'])


def test_equal_weights_are_the_unweighted_check(data):
    ref = summarize(data)
    sub = data.sample(20_000, random_state=4)
    a, b = compare(ref, sub), compare(ref, sub, np.ones(len(sub)))
    assert a['cdf_gaps'] == b['cdf_gaps'] and a['worst_corr'] == b['worst_corr']
    pandas = sub[list(FEATURES)].corr(method='spearman').fillna(0.0).to_numpy()
    assert np.abs(weighted_spearman(sub, np.ones(len(sub))) - pandas).max() < 1e-12


def test_weights_mix_the_wells_as_the_reference():
    """A reference that holds fewer rows of some wells, as removing the trickle rows did, against a candidate
    with the same number of rows per well: unweighted it fails, weighted to the reference's wells it passes."""
    rng = np.random.default_rng(5)
    n_wells, per_well = 400, 200
    level = rng.normal(size=n_wells) + np.where(np.arange(n_wells) < 80, 3.0, 0.0)   # 80 wells unlike the rest
    rows_kept = np.where(np.arange(n_wells) < 80, per_well // 10, per_well)           # which lost most of their rows

    def rows(counts, seed):
        r = np.random.default_rng(seed)
        ids = np.repeat(np.arange(n_wells), counts)
        base = level[ids] + r.normal(size=len(ids))
        return pd.DataFrame({f: base * (k % 3) + r.normal(size=len(ids)) for k, f in enumerate(FEATURES)}).assign(ID=ids)

    ref = summarize(rows(rows_kept, 6))
    candidate = rows(np.full(n_wells, 25), 7)
    assert not compare(ref, candidate)['passed']
    weighted = compare(ref, candidate, well_weights(ref, candidate))
    assert weighted['passed'], weighted['worst_cdf']
    assert weighted['worst_corr'][2] <= MAX_CORR_GAP
