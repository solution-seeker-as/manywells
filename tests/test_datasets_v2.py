"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests of the v2 datasets' files (manywells.datasets.v2; specs/sampling.md, SMP-108 and SMP-110): the columns' types
and the labels' codes, the records of the generation, a folder of tables with its metadata, and a table well's PVT
model rebuilt from its rows to the bit. The re-solve of a sample from its rows is tested on generated wells in
tests/test_sampling_v2_run.py.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manywells.datasets import v2
from manywells.datasets.v2 import _CODES, pvt_from_rows, read_dataset, _typed, _write_tables
from manywells.pvt.table import PVTTable, PVTTableSet
from manywells.sampling.v2.assemble import _pvt_rows

PVT_DATA = Path(__file__).parent / 'data' / 'pvt'


def test_labels_are_categories_in_the_order_of_their_codes():
    df = _typed(pd.DataFrame(dict(well_type=['subsea', 'onshore'], ID=[3.0, 4.0], n_cells=[100, 100])), 'wells')
    assert list(df['well_type'].cat.categories) == ['platform', 'subsea', 'onshore']
    assert list(df['well_type'].cat.codes) == [1, 2]
    assert (df['ID'].dtype, df['n_cells'].dtype) == (np.dtype('int32'), np.dtype('int16'))


def test_a_value_without_a_code_is_refused():
    with pytest.raises(ValueError, match='well_type: values without a code'):
        _typed(pd.DataFrame(dict(well_type=['platform', 'jackup'])), 'wells')
    with pytest.raises(ValueError, match='outcome'):
        _typed(pd.DataFrame(dict(outcome=['recovered'])), 'redraws')  # The CasADi retry's outcome is gone (D-122)


def test_the_logs_ids_are_nullable_and_their_outcomes_have_their_codes():
    drawn = _typed(pd.DataFrame(dict(ID=[0, None], outcome=['accepted', 'table_failed'])), 'drawn_wells')
    assert str(drawn['ID'].dtype) == 'Int32' and drawn['ID'].isna().tolist() == [False, True]
    assert list(drawn['outcome'].cat.categories) == list(v2._WELL_OUTCOMES)
    red = _typed(pd.DataFrame(dict(outcome=['solver_failure'], reason=['incomplete_scan'], n_roots=[0])), 'redraws')
    assert list(red['outcome'].cat.categories) == ['operating_point', 'no_operating_point', 'solver_failure']
    assert list(red['reason'].cat.categories) == list(v2._REASONS) and red['n_roots'].dtype == np.dtype('int8')


def test_the_codes_hold_the_spec_labels():
    """SMP-52's values, in datasets-draft.md's order with the spec's additions at the end (D-124)."""
    assert _CODES['lift_start_rule'][-1] == 'no_natural_flow' and _CODES['saturation_kind'][-1] == 'none'
    assert v2._WELL_OUTCOMES[-2:] == ('table_failed', 'crash')
    assert set(_CODES['FRBH']) == {'bubbly', 'dispersed-bubbly', 'slug', 'churn', 'annular'}


def test_a_folder_of_tables_and_its_metadata(tmp_path):
    tables = dict(wells=_typed(pd.DataFrame(dict(ID=[0, 1], z=[1000.0, 2000.0])), 'wells'))
    tables['geometry'] = _typed(
        pd.DataFrame(dict(ID=[0, 0], station=[0, 1], md=[0.0, 9.5], tvd=[0.0, 9.5])), 'geometry'
    )
    (tmp_path / 'metadata.json').write_text('{"stale": true}')
    meta = _write_tables(tmp_path, tables, dict(seed=7))
    assert meta['seed'] == 7 and set(meta['files']) == {'wells.parquet', 'geometry.parquet'}
    assert meta['files']['geometry.parquet']['rows'] == 2
    assert meta['files']['wells.parquet']['sha256'] == v2._sha256(tmp_path / 'wells.parquet')
    back = read_dataset(tmp_path)
    assert back['metadata'] == json.loads((tmp_path / 'metadata.json').read_text()) and 'stale' not in back['metadata']
    pd.testing.assert_frame_equal(back['geometry'], tables['geometry'])


def test_the_code_and_platform_records():
    code = v2.code_record()
    assert code['version'].startswith('2.')
    if code['commit'] is not None:
        assert len(code['commit']) == 40 and isinstance(code['dirty'], bool)
    lock = v2._lock_hash()
    assert lock is None or len(lock) == 64
    assert set(v2._platform_record()) == {'python', 'platform', 'machine'}


@pytest.mark.parametrize('name', ['synthetic_eos_oil_gor.csv', 'synthetic_eos_oil_360K.csv'])
def test_a_tables_rows_rebuild_its_table_to_the_bit(tmp_path, name):
    """SMP-110: a table's pvt rows, through parquet, rebuild the table that PVTTable.read_csv reads from its file."""
    rows = _pvt_rows(dict(tables_dir=str(PVT_DATA), csv=name), 'k000012')
    assert rows['pvt_table_id'].eq('k000012').all() and rows['rho_o'].nunique() == 1
    rows.to_parquet(tmp_path / 'pvt.parquet', index=False)
    back = pd.read_parquet(tmp_path / 'pvt.parquet')
    want, got = PVTTable.read_csv(PVT_DATA / name), pvt_from_rows(back)
    assert type(got) is type(want) and got == want
    tables = got.tables if isinstance(got, PVTTableSet) else (got,)
    for t, t_want in zip(tables, want.tables if isinstance(want, PVTTableSet) else (want,)):
        assert (t.name, t.source, t.separation) == (t_want.name, t_want.source, t_want.separation)
        assert all(np.array_equal(getattr(t, c), getattr(t_want, c)) for c in t.columns)


def test_a_table_without_an_optional_column():
    """A column that is null in every row of a table is one the table does not have."""
    rows = _pvt_rows(dict(tables_dir=str(PVT_DATA), csv='synthetic_eos_oil_gor.csv'), 'k000003')
    rows['mu_g'] = math.nan
    tables = pvt_from_rows(rows).tables
    assert all(t.mu_g is None and t.mu_o is not None for t in tables)
    single = pvt_from_rows(_pvt_rows(dict(tables_dir=str(PVT_DATA), csv='synthetic_eos_oil_360K.csv'), 'k000004'))
    assert single.columns == ('R_so', 'B_o', 'Z') and single.gor == 108.7299
