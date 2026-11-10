"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests of the v2 generator over many wells and of a dataset's assembly (manywells.sampling.v2.run, .assemble and
.table_wells; specs/sampling.md, SMP-108 to SMP-110, SMP-112 and SMP-116): the run in chunks until n wells are
accepted, a stopped run resumed, a crash logged and counted; the dataset's tables and logs, the split and the files;
the table wells and their GOR axes in the files that table_build.py reads; and a sample re-solved from its rows. The
fast tests generate wells with stand-in solves, or put a stand-in in place of a well's generation; the slow test runs
a small real pilot and re-solves its samples.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from manywells.datasets.v2 import (
    CODES,
    FEATURES,
    LOGS,
    REASONS,
    TABLES,
    WEEKS_PER_YEAR,
    read_dataset,
    resolve_sample,
    sample_case,
    sample_rows,
    sha256,
    write_tables,
)
from manywells.sampling.v2 import assemble as asm
from manywells.sampling.v2 import generate
from manywells.sampling.v2 import run as runner
from manywells.sampling.v2 import table_build as tb
from manywells.sampling.v2 import table_wells as tw
from manywells.sampling.v2.life import Life, sol2_well
from manywells.sampling.v2.mapping import WellBuilder
from manywells.sampling.v2.samples import nsol2_attempt, nsol2_plan, sol2_attempt
from manywells.sampling.v2.tables import TABLE_CLASSES, TableMismatch, read_table
from manywells.sampling.wells import rng_for

from .sampling_v2_wells import SEED, drawn_lives

N_TEST_SLOTS = 6  # The stand-in wells' slots
N_TEST_WELLS = 12  # The stand-in wells of each dataset: draw indices 0 to 11 of the tests' seed


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


# --- Stand-ins -------------------------------------------------------------------------------------------------------
def _solvable() -> list:
    return [lf.k for lf in drawn_lives() if lf.w0['fluid_class'] not in TABLE_CLASSES][:N_TEST_WELLS]


def stand_in_attempt(builder, inp) -> dict:
    """In place of generate._attempt: every best-rate and scan solve has an operating point. A slot's attempt has one
    too, but in three wells: one whose attempts at a choke below 0.4 have none (its slots are redrawn), one whose
    attempts fail in the solver (it reaches the solver-failure limit) and one whose attempts never flow (a slot
    reaches its limit)."""
    k, slot = builder.w['well'], 'test' in inp or 'p_s_nom' not in inp
    redrawn, solver, dead = _solvable()[1:4]
    outcome, reason = 'operating_point', None
    if slot and ((k == redrawn and inp['u'] < 0.4) or k == dead):
        outcome, reason = 'no_operating_point', 'no_root'
    elif slot and k == solver:
        outcome, reason = 'solver_failure', 'incomplete_scan'
    rec = dict(inp, outcome=outcome, reason=reason, error=None, n_roots=int(outcome != 'no_operating_point'))
    rec.update(
        n_stable=int(outcome == 'operating_point'), several_stable=outcome == 'operating_point' and inp['u'] > 0.9
    )
    rec.update(marches=7, non_finite=0, rejected=int(reason == 'incomplete_scan'), t_solve=0.0, t_build=0.0)
    if outcome == 'operating_point':
        rec.update({key: 1.0 + inp['u'] for key in FEATURES}, CHOKED=inp['u'] < 0.2, FRBH='bubbly', FRWH='annular')
        rec['q_oe'] = 100.0
    return rec


@pytest.fixture(scope='module')
def results() -> dict:
    """The results of draw indices 0 to N_TEST_WELLS - 1 of each dataset, generated with stand-in solves and
    N_TEST_SLOTS slots, without tables, and well CRASHED's generation crashed."""
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(generate, 'N_SLOTS', N_TEST_SLOTS)
        mp.setattr(generate, '_attempt', stand_in_attempt)
        for kind in ('sol', 'nsol'):
            out[kind] = [runner.run_well(kind, SEED, k) for k in range(N_TEST_WELLS)]
            with pytest.MonkeyPatch.context() as crash:
                crash.setitem(runner.GENERATE, kind, crashing(lambda k: True)[0])
                out[kind][CRASHED] = runner.run_well(kind, SEED, CRASHED)
    return out


CRASHED = 8  # A table well of the tests' seed, before its last accepted well: the crash is in the logs


def crashing(when):
    """A stand-in generation: a well that crashes where when(k), else an accepted well without solves."""
    calls = []

    def gen(seed, k, tables):
        calls.append(k)
        if when(k):
            raise ValueError(f"gor {k} is outside the PVT table's GOR axis")
        return dict(well={'well': k}, stations=None, table=None, nsol=None, samples=[], log=[], info=accepted(k))

    return gen, calls


def accepted(k) -> dict:
    return dict(draw_index=k, outcome='accepted' if k % 4 != 3 else 'not_producible', error=None)


class Stop(BaseException):
    """A run stopped from outside, as by an interrupt: not an exception that a well's generation catches."""


def stand_in_run(monkeypatch, outcome=accepted, raise_at=None):
    """A stand-in for both generations: well k's outcome is outcome(k)['outcome'], and at raise_at the run is stopped.
    Returns the draw indices generated."""
    calls = []

    def gen(seed, k, tables):
        if k == raise_at:
            raise Stop
        calls.append(k)
        return dict(well={'well': k}, stations=None, table=None, nsol=None, samples=[], log=[], info=outcome(k))

    monkeypatch.setitem(runner.GENERATE, 'sol', gen)
    monkeypatch.setitem(runner.GENERATE, 'nsol', gen)
    return calls


# --- SMP-110: the run in chunks ---------------------------------------------------------------------------------------
def first_accepted(n) -> list:
    return [k for k in range(10 * n) if k % 4 != 3][:n]


def test_a_run_until_n_wells_are_accepted(monkeypatch, tmp_path):
    """Draw indices 0, 1, 2, ... in chunks until the first n accepted wells are known; each chunk in one file."""
    calls = stand_in_run(monkeypatch)
    state = runner.run('sol', 25, tmp_path, SEED, processes=1, chunk_size=10, log=lambda line: None)
    last = first_accepted(25)[-1]
    assert (state['complete'], state['accepted'], state['drawn']) == (True, 25, last + 1)
    assert state['outcomes'] == {'accepted': 25, 'not_producible': last // 4}
    assert sorted(calls) == list(range(40)) and state['next_chunk'] == 4  # Chunks 0 to 3, each well once
    assert sorted(p.name for p in (tmp_path / 'chunks').iterdir()) == [f'chunk-{c:06d}.pkl' for c in range(4)]
    assert not list((tmp_path / 'wells').iterdir())
    assert [r['info']['draw_index'] for r in runner.read_results(tmp_path)] == list(range(40))
    headers = runner.chunk_headers(tmp_path)
    assert headers[2]['first'] == 20 and len(headers[2]['outcomes']) == 10
    assert headers[2]['settings'] == json.loads((tmp_path / 'settings.json').read_text())

    calls.clear()  # The same run again does nothing; a run of more wells resumes it
    assert runner.run('sol', 25, tmp_path, SEED, processes=1, chunk_size=10, log=lambda line: None)['complete']
    assert calls == []
    state = runner.run('sol', 31, tmp_path, SEED, processes=1, chunk_size=10, log=lambda line: None)
    assert state['accepted'] == 31 and sorted(calls) == list(range(40, 50))


def test_a_stopped_run_resumes(monkeypatch, tmp_path):
    """A run stopped within a chunk keeps the wells it solved: the rerun solves the rest, and gives the same
    results as a run that did not stop."""
    stand_in_run(monkeypatch, raise_at=15)
    with pytest.raises(Stop):
        runner.run('nsol', 12, tmp_path / 'a', SEED, processes=1, chunk_size=10, log=lambda line: None)
    assert [p.name for p in (tmp_path / 'a' / 'chunks').iterdir()] == ['chunk-000000.pkl']
    assert sorted(p.name for p in (tmp_path / 'a' / 'wells').iterdir()) == [f'k{k:06d}.pkl' for k in range(10, 15)]
    calls = stand_in_run(monkeypatch)
    state = runner.run('nsol', 12, tmp_path / 'a', SEED, processes=1, chunk_size=10, log=lambda line: None)
    assert state['complete'] and sorted(calls) == list(range(15, 20))
    runner.run('nsol', 12, tmp_path / 'b', SEED, processes=1, chunk_size=10, log=lambda line: None)
    a, b = list(runner.read_results(tmp_path / 'a')), list(runner.read_results(tmp_path / 'b'))
    assert [r['info'] for r in a] == [r['info'] for r in b]


def test_a_crash_is_logged_and_counted(monkeypatch, tmp_path):
    """A well whose generation raises is logged as `crash` with its traceback, and the run carries on."""
    gen, calls = crashing(lambda k: k in (2, 13))
    monkeypatch.setitem(runner.GENERATE, 'sol', gen)
    state = runner.run('sol', 12, tmp_path, SEED, processes=1, chunk_size=10, log=lambda line: None)
    assert state['complete'] and state['outcomes']['crash'] == 2 and sorted(calls) == list(range(20))
    crashed = [r for r in runner.read_results(tmp_path) if r['info']['outcome'] == 'crash']
    assert [r['info']['draw_index'] for r in crashed] == [2, 13]
    info = crashed[0]['info']
    assert info['error'].startswith('Traceback') and "ValueError: gor 2 is outside the PVT table's GOR" in info['error']
    assert crashed[0]['log'] == [] and crashed[0]['cpu_seconds'] >= 0 and crashed[0]['seconds'] >= 0


PanicException = type('PanicException', (BaseException,), {'__module__': 'pyo3_runtime'})  # PyO3's, as it raises it


def test_a_panic_outside_a_solve_is_a_crash(monkeypatch):
    def gen(seed, k, tables):
        raise PanicException('called `Option::unwrap()` on a `None` value')

    monkeypatch.setitem(runner.GENERATE, 'nsol', gen)
    res = runner.run_well('nsol', SEED, 3)
    assert res['info']['outcome'] == 'crash' and res['info']['error'].rstrip().endswith('on a `None` value')
    assert (res['info']['scan'], res['log'], res['samples']) == ([], [], [])


def out_of_memory(monkeypatch, fails):
    """A stand-in generation in which well k runs out of memory in its first fails(k) processes, and a stand-in for
    the fresh process (run._fresh), the same process here. Returns the processes each well was given."""
    tries = {}

    def gen(seed, k, tables):
        tries[k] = tries.get(k, 0) + 1
        if tries[k] <= fails(k):
            raise MemoryError
        return dict(well={'well': k}, stations=None, table=None, nsol=None, samples=[], log=[], info=accepted(k))

    fresh = []
    monkeypatch.setitem(runner.GENERATE, 'sol', gen)
    monkeypatch.setattr(runner, '_fresh', lambda task, cap: fresh.append(task[2]) or runner._task(task))
    return tries, fresh


def test_a_well_out_of_memory_is_solved_again_in_a_fresh_process(monkeypatch, tmp_path):
    """A MemoryError is the process's, not the well's: it is no outcome, and the well is solved again, after the other
    wells of its batch, in a fresh process; its result is then the one a run without the error gives."""
    tries, fresh = out_of_memory(monkeypatch, lambda k: 1 if k in (4, 12) else 0)
    lines = []
    state = runner.run('sol', 12, tmp_path / 'a', SEED, processes=1, chunk_size=10, log=lines.append)
    assert state['complete'] and 'crash' not in state['outcomes'] and fresh == [4, 12]
    assert tries[4] == tries[12] == 2 and all(n == 1 for k, n in tries.items() if k not in (4, 12))
    assert (
        lines.index('well 4: out of memory; solved again in a fresh process (1 of 2)')
        > [i for i, line in enumerate(lines) if line.startswith('well 9:')][0]
    )
    out_of_memory(monkeypatch, lambda k: 0)
    runner.run('sol', 12, tmp_path / 'b', SEED, processes=1, chunk_size=10, log=lambda x: None)
    a, b = list(runner.read_results(tmp_path / 'a')), list(runner.read_results(tmp_path / 'b'))
    assert [r['info'] for r in a] == [r['info'] for r in b]


def test_a_well_always_out_of_memory_stops_the_run_and_is_not_kept(monkeypatch, tmp_path):
    """After MEMORY_RETRIES fresh processes the run stops with a message that names the well; the batch's other
    wells are kept, the well is not, and a rerun solves it."""
    tries, fresh = out_of_memory(monkeypatch, lambda k: 99 if k == 6 else 0)
    with pytest.raises(
        runner.OutOfMemory, match='well 6 ran out of memory, with a cap of 2.50 GB, in the pool and in 2'
    ):
        runner.run('sol', 5, tmp_path, SEED, processes=1, chunk_size=10, memory_cap=2.5e9, log=lambda x: None)
    assert tries[6] == 1 + runner.MEMORY_RETRIES and fresh == [6, 6]
    assert sorted(p.name for p in (tmp_path / 'wells').iterdir()) == [f'k{k:06d}.pkl' for k in range(10) if k != 6]
    assert not list((tmp_path / 'chunks').iterdir())
    tries, fresh = out_of_memory(monkeypatch, lambda k: 0)
    state = runner.run('sol', 5, tmp_path, SEED, processes=1, chunk_size=10, log=lambda x: None)
    assert state['complete'] and tries == {6: 1} and fresh == []
    assert [r['info']['outcome'] for r in runner.read_results(tmp_path)][6] == accepted(6)['outcome']


def test_another_runs_settings_stop_the_run(monkeypatch, tmp_path):
    stand_in_run(monkeypatch)
    runner.run('sol', 5, tmp_path, SEED, processes=1, chunk_size=10, log=lambda line: None)
    for kind, seed, chunk_size in (('sol', SEED + 1, 10), ('sol', SEED, 20), ('nsol', SEED, 10)):
        with pytest.raises(runner.SettingsMismatch):
            runner.run(kind, 5, tmp_path, seed, processes=1, chunk_size=chunk_size, log=lambda line: None)
    settings = runner.run_settings('sol', SEED, None, 10)
    assert (settings['dataset'], settings['configuration'], settings['n_cells'], settings['backend']) == (
        'manywells-sol-2',
        'v2.0.0',
        100,
        'rust',
    )
    assert settings['limits'] == dict(slot=50, redraws=400, solver_failures=10) and settings['n_slots'] == 200
    assert set(settings['code']) == {'version', 'commit', 'dirty'}
    with pytest.raises(ValueError, match='kind'):
        runner.run_settings('closed', SEED)


def test_the_tables_range_limits_the_run(monkeypatch, tmp_path):
    """No chunk runs beyond the draw indices that the folder of tables covers: a table well there has no table."""
    stand_in_run(monkeypatch)
    tables = tmp_path / 'tables'
    tables.mkdir()
    (tables / 'wells.json').write_text(json.dumps(dict(kind='sol', seed=SEED, wells_range=[0, 25], wells=[])))
    state = runner.run('sol', 30, tmp_path / 'run', SEED, processes=1, tables=tables, chunk_size=10, log=lambda x: None)
    assert (state['complete'], state['limit'], state['next_chunk']) == (False, 25, 2)
    assert json.loads((tmp_path / 'run' / 'settings.json').read_text())['tables'] == str(tables.resolve())
    with pytest.raises(ValueError, match='not nsol'):
        runner.run('nsol', 3, tmp_path / 'other', SEED, processes=1, tables=tables, chunk_size=10, log=lambda x: None)


# --- SMP-108, SMP-110: the assembly -----------------------------------------------------------------------------------
@pytest.mark.parametrize('kind', ['sol', 'nsol'])
def test_the_assembled_tables(results, kind, tmp_path):
    """The tables and logs of the wells up to the n-th accepted: their columns and types, the IDs in draw order, every
    sample of an accepted well, and every slot's attempt of a drawn well in the samples or the redraw log."""
    res = results[kind]
    outcomes = [r['info']['outcome'] for r in res]
    assert {'accepted', 'table_failed', 'crash'} <= set(outcomes)
    assert {'solver_failure_limit', 'slot_limit' if kind == 'sol' else 'dead_date'} <= set(outcomes)
    n = outcomes.count('accepted')
    attempts = []
    out = asm.assemble(res, kind, SEED, n, attempts=attempts.append)
    meta = write_tables(tmp_path, {key: out[key] for key in TABLES + LOGS}, {})  # Every table writes as parquet
    assert {name: f['rows'] for name, f in meta['files'].items()} == {
        f'{k}.parquet': len(out[k]) for k in TABLES + LOGS
    }
    crash = out['drawn_wells'].loc[out['drawn_wells']['draw_index'] == CRASHED].iloc[0]
    assert crash['outcome'] == 'crash' and crash['error'].startswith('Traceback') and crash['n_attempts'] == 0
    wells, samples, drawn, red = out['wells'], out['samples'], out['drawn_wells'], out['redraws']
    assert list(wells.columns) == asm.well_columns(kind) and list(drawn.columns) == asm.drawn_well_columns(kind)
    assert list(samples.columns) == asm.sample_columns(kind) and list(red.columns) == asm.redraw_columns(kind)
    assert list(out['geometry'].columns) == ['ID', 'station', 'md', 'tvd'] and len(out['pvt']) == 0
    for df in (wells, drawn):
        for col in CODES:
            if col in df:
                assert isinstance(df[col].dtype, pd.CategoricalDtype)
    assert wells['ID'].tolist() == list(range(n)) and wells['ID'].dtype == np.dtype('int32')
    acc = [r['info']['draw_index'] for r in res if r['info']['outcome'] == 'accepted']
    assert wells['draw_index'].tolist() == acc and drawn['draw_index'].tolist() == list(range(acc[-1] + 1))
    assert drawn.loc[drawn['ID'].notna(), 'draw_index'].tolist() == acc
    assert drawn['outcome'].astype(str).tolist() == outcomes[: acc[-1] + 1]
    assert set(wells['split'].astype(str)) <= {'train', 'test'} and drawn.loc[drawn['ID'].isna(), 'split'].isna().all()
    # Every sample of every accepted well, in the order of its slots, with the record's inputs and features
    assert samples[['ID', 'slot']].values.tolist() == [[i, j] for i in range(n) for j in range(N_TEST_SLOTS)]
    by_k = {r['info']['draw_index']: r for r in res}
    for i, k in enumerate(acc):
        recs = by_k[k]['samples']
        s = samples[samples['ID'] == i]
        assert s['attempt'].tolist() == [rec['attempt'] for rec in recs]
        assert s['CHK'].tolist() == [rec['u'] for rec in recs] and s['PRES'].tolist() == [rec['p_r'] for rec in recs]
        assert s['PBH'].tolist() == [rec['PBH'] for rec in recs] and s['FRBH'].astype(str).eq('bubbly').all()
        if kind == 'nsol':
            assert s['WEEKS'].tolist() == [rec['t'] * WEEKS_PER_YEAR for rec in recs]
            assert s['STAGE'].astype(str).tolist() == [rec['life_stage'] for rec in recs]
            assert s['LIFT'].tolist() == [rec['lift_on'] for rec in recs] and s['TEST'].isna().all()
            assert all(same(a, rec['k_l']) for a, rec in zip(s['KL'], recs))
    # The redraw log: the attempts without an operating point of the accepted wells, every attempt of a dropped one
    for _, row in drawn.iterrows():
        k = row['draw_index']
        slots = [rec for rec in by_k[k]['log'] if rec['kind'] == 'slot']
        mine = red[red['draw_index'] == k]
        kept = [rec for rec in slots if rec['outcome'] != 'operating_point' or row['outcome'] != 'accepted']
        assert mine[['slot', 'attempt']].values.tolist() == [[rec['slot'], rec['attempt']] for rec in kept]
        assert mine['outcome'].astype(str).tolist() == [rec['outcome'] for rec in kept]
        assert mine['ID'].isna().tolist() == [row['outcome'] != 'accepted'] * len(mine)
        failed = [rec for rec in slots if rec['outcome'] != 'operating_point']
        assert row['n_attempts'] == len(slots) and row['n_redraws'] == sum(rec['attempt'] > 0 for rec in slots)
        assert row['n_redraws_no_op'] + row['n_redraws_solver'] == len(failed)
        assert sum(row[f'n_redraws_{reason}'] for reason in REASONS) == len(failed)
        if row['outcome'] == 'accepted':
            assert row['n_redraws'] == len(failed) == len(mine)  # Each attempt without an operating point is redrawn
    assert sum(len(a) for a in attempts) == sum(len(by_k[k]['log']) for k in drawn['draw_index'])
    counts = out['counts']
    assert counts['wells_drawn'] == len(drawn) and counts['wells_accepted'] == n
    assert sum(counts['outcomes'].values()) == len(drawn)
    assert sum(counts['attempts'].values()) == drawn['n_attempts'].sum()


@pytest.mark.parametrize('kind', ['sol', 'nsol'])
def test_a_wells_row(results, kind):
    """A well's row holds its draws, its labels and its generation's counts; a crash its draw index and error."""
    out = asm.assemble(results[kind], kind, SEED, 1)  # The wells up to the first accepted
    k = out['wells']['draw_index'][0]
    res = next(r for r in results[kind] if r['info']['draw_index'] == k)
    row, w = out['wells'].iloc[0], res['well']
    assert (row['z'], row['water_depth'], row['K_c'], row['R_si']) == (w['z'], w['d_w'], w['K_c'], w['R_i'])
    assert row['md_bottom'] == res['stations'][0][-1] and 0 <= row['max_inclination'] <= 75 + 1e-9
    assert row['floor'] == generate.FLOOR[w['well_type']] and row['best_rate'] == res['info']['best_rate']
    assert row['pvt_model'] == 'black_oil' and pd.isna(row['pvt_table_id']) and pd.isna(row['comp_w1'])
    if kind == 'sol':
        assert (row['p_r_nom'], row['gor_nom'], row['life_date']) == (w['p_r'], w['R_go'], w['date'])
        assert row['lift_start_rule'] == w['lift_start_rule'] or (
            pd.isna(row['lift_start_rule']) and w['lift'] == 'natural'
        )
    else:
        life = res['nsol']
        assert (row['T_life'], row['T_end'], row['p_change']) == (
            life['T_life'],
            res['info']['T_end'],
            res['info']['p_change'],
        )
        assert row['lift'] == ('natural' if life['lift_start'] is None else 'gas_lift')
        design = life['design']
        assert same(row['injection_md'], design['injection_md'] if design is not None else math.nan)
    crash = asm.well_row(results[kind][CRASHED], kind)
    assert (crash['outcome'], crash['n_attempts']) == ('crash', 0) and 'ValueError' in crash['error']
    with pytest.raises(ValueError, match='accepted wells'):
        asm.assemble(results[kind], kind, SEED, 100)


def test_the_split_by_stratum():
    """SMP-110: within each stratum, ranks 5, 10, 15, ... of the split's draw are `test`: a fifth of each stratum, to
    within one well; the same wells give the same split, and another seed another."""
    r = np.random.default_rng(3)
    n = 2000
    wells = pd.DataFrame(
        dict(
            ID=np.arange(n),
            fluid_type=r.choice(['black_oil', 'gas'], n),
            well_type=r.choice(['platform', 'subsea', 'onshore'], n, p=[0.5, 0.4, 0.1]),
            geometry_class=r.choice(['vertical', 'deviated'], n),
            coverage_tail=r.choice(['core', 'tail'], n, p=[0.8, 0.2]),
        )
    )
    s = asm.split(wells, 2026)
    strata = wells.assign(test=s == 'test').groupby(['fluid_type', 'well_type', 'geometry_class', 'coverage_tail'])
    for _, g in strata:
        assert g['test'].sum() == len(g) // 5
    assert (s == asm.split(wells, 2026)).all() and not (s == asm.split(wells, 2027)).all()
    key = rng_for(2026, 'v2', 'split').random(n)  # The split's own generator, one number per well in ID order
    g = wells.assign(key=key, test=s == 'test').query("fluid_type == 'gas' and well_type == 'onshore'")
    g = g.query("geometry_class == 'vertical' and coverage_tail == 'core'").sort_values('key')
    assert g['test'].tolist() == [(i + 1) % 5 == 0 for i in range(len(g))]


def test_a_samples_case_from_its_rows_is_the_generators(results):
    """SMP-110: a sample's row, its well's row and its survey rebuild the WellProperties and BoundaryConditions that
    the generator solved (mapping.WellBuilder.case of the sample's record)."""
    for kind in ('sol', 'nsol'):
        res = results[kind]
        n = sum(r['info']['outcome'] == 'accepted' for r in res)
        out = asm.assemble(res, kind, SEED, n)
        tables = {key: out[key] for key in ('wells', 'samples', 'geometry', 'pvt')}
        for r in [r for r in res if r['info']['outcome'] == 'accepted'][:3]:
            k = r['info']['draw_index']
            life = Life(SEED, k)
            builder = WellBuilder(life.w0, life.stations, design=life.design)
            ID = int(out['wells'].loc[out['wells']['draw_index'] == k, 'ID'].iloc[0])
            for rec in r['samples'][:3]:
                wp, bc = sample_case(*sample_rows(tables, ID, rec['slot']))
                wp_gen, bc_gen = builder.case(rec)
                assert bc == bc_gen and wp.fluid == wp_gen.fluid and wp.inflow == wp_gen.inflow
                assert (wp.friction, wp.thermal, wp.choke, wp.slip) == (
                    wp_gen.friction,
                    wp_gen.thermal,
                    wp_gen.choke,
                    wp_gen.slip,
                )
                assert wp.injection_md == wp_gen.injection_md
                assert np.array_equal(wp.geometry.md, wp_gen.geometry.md)
                assert np.array_equal(wp.geometry.tvd, wp_gen.geometry.tvd) and wp.geometry.D == wp_gen.geometry.D


def test_a_dataset_written_from_a_run(monkeypatch, tmp_path):
    """A run with stand-in solves, assembled: the dataset's folder and its companion folder of logs, each with its
    metadata, the run's attempt log, and no dataset from code with changes unless allowed."""
    monkeypatch.setattr(generate, 'N_SLOTS', N_TEST_SLOTS)
    monkeypatch.setattr(generate, '_attempt', stand_in_attempt)
    state = runner.run('sol', 3, tmp_path / 'run', SEED, processes=1, chunk_size=5, log=lambda x: None)
    assert state['complete']
    meta = asm.write_dataset(tmp_path / 'run', tmp_path / 'out', 3, allow_dirty=True)
    data = read_dataset(tmp_path / 'out' / 'manywells-sol-2')
    logs = read_dataset(tmp_path / 'out' / 'manywells-sol-2-logs')
    assert set(data) == {'wells', 'samples', 'geometry', 'pvt', 'metadata'} and data['metadata'] == meta
    assert set(logs) == {'drawn_wells', 'redraws', 'metadata'} and logs['metadata']['companion_of'] == 'manywells-sol-2'
    assert (meta['seed'], meta['n_wells'], meta['configuration'], meta['backend']) == (SEED, 3, 'v2.0.0', 'rust')
    assert meta['limits'] == dict(slot=50, redraws=400, solver_failures=10) and meta['n_slots'] == N_TEST_SLOTS
    assert meta['producibility']['floor'] == generate.FLOOR and meta['counts']['wells_accepted'] == 3
    assert meta['files']['samples.parquet']['rows'] == 3 * N_TEST_SLOTS == len(data['samples'])
    assert {'code', 'assembled_by', 'lock_sha256', 'python', 'platform', 'created'} <= set(meta)
    assert meta['tables'] is None and meta['table_build'] is None
    assert len(logs['drawn_wells']) == meta['counts']['wells_drawn']
    parts = sorted((tmp_path / 'run' / 'attempts').glob('part-*.parquet'))
    assert sum(len(pd.read_parquet(p)) for p in parts) == sum(
        len(r['log']) for r in list(runner.read_results(tmp_path / 'run'))[: meta['counts']['wells_drawn']]
    )
    dirty = dict(version='2.0.0.dev0 (abc-dirty)', commit='0' * 40, dirty=True)
    monkeypatch.setattr(asm, 'code_record', lambda: dirty)
    with pytest.raises(ValueError, match='has changes'):
        asm.write_dataset(tmp_path / 'run', tmp_path / 'out2', 3)


def test_the_table_builds_record(tmp_path):
    """SMP-110, SMP-116: a dataset records its folder of tables' build: the files' SHA-256, the draw indices and the
    table wells it covers, and NeqSim's version."""
    spec = dict(kind='sol', seed=SEED, wells_range=[0, 300], wells=[dict(well=3), dict(well=8)])
    tw.write_json(tmp_path / 'wells.json', spec)
    record = asm._table_build_record(tmp_path)
    assert record == dict(
        wells_sha256=sha256(tmp_path / 'wells.json'),
        axes_sha256=None,
        wells_range=[0, 300],
        table_wells=2,
        neqsim='3.4.0',
    )
    tw.write_json(tmp_path / 'axes.json', {'3': dict(gors=[1.0, 1.2])})
    assert asm._table_build_record(tmp_path)['axes_sha256'] == sha256(tmp_path / 'axes.json')
    assert asm._table_build_record(None) is None


# --- SMP-112, SMP-116: the table wells and their axes -----------------------------------------------------------------
HAND_MADE = dict(
    well=4,
    fluid_class='gas_condensate',
    coverage_tail='core',
    api=50.5,
    sg_gas=0.68,
    p_sat=402.25,
    p_i=455.0,
    p_r=430.125,
    T_r=401.3,
    T_s=279.9,
    p_sep=35.0,
    T_sep=310.0,
    R_i=1500.0,
)


def hand_made(k, fluid_class='gas_condensate'):
    return dict(HAND_MADE, well=k, fluid_class=fluid_class), ([0.0, 3000.0], [0.0, 3000.0])


def test_a_table_wells_record(monkeypatch):
    """SMP-112, SMP-116: a table well's record holds its draws, its grid, and the composition's seed and source
    from its `composition` generator; a well of another class has none."""
    monkeypatch.setattr(tw, 'draw_well', lambda seed, k: hand_made(k, 'black_oil' if k == 5 else 'gas_condensate'))
    rec = tw.table_well(SEED, 4)
    r = rng_for(SEED, 4, 'v2', 'composition')
    seed, u_source, u_key = int(r.integers(2**63)), r.random(), r.random()
    T_axis, p_top = tb.well_grid(279.9, 401.3, 455.0)
    assert rec == dict(
        {key: v for key, v in HAND_MADE.items() if key != 'R_i'},
        R_si=1500.0,
        composition_seed=seed,
        T_axis=[float(T) for T in T_axis],
        p_top=float(p_top),
        source=tb.fluid_source('gas_condensate', u_source, u_key),
    )
    assert rec['T_axis'][0] == 260.0 and rec['T_axis'][-1] == 410.0 and rec['p_top'] == 490.0
    assert tw.table_well(SEED, 5) is None
    assert [w['well'] for w in tw.table_wells('sol', SEED, 3, 7)] == [3, 4, 6]


def test_the_table_wells_file_is_extended(monkeypatch, tmp_path):
    monkeypatch.setattr(tw, 'draw_well', lambda seed, k: hand_made(k, ('gas_condensate', 'black_oil')[k % 2]))
    spec = tw.write_table_wells(tmp_path, 'nsol', SEED, 6)
    assert spec['wells_range'] == [0, 6] and [w['well'] for w in spec['wells']] == [0, 2, 4]
    assert json.loads((tmp_path / 'wells.json').read_text()) == json.loads(json.dumps(spec))
    monkeypatch.setattr(tw, 'draw_well', lambda seed, k: hand_made(k, 'volatile_oil'))
    spec = tw.write_table_wells(tmp_path, 'nsol', SEED, 9)  # The records kept, the new draw indices added
    assert spec['wells_range'] == [0, 9] and [w['well'] for w in spec['wells']] == [0, 2, 4, 6, 7, 8]
    assert spec['wells'][0]['fluid_class'] == 'gas_condensate'
    with pytest.raises(ValueError, match='not sol'):
        tw.write_table_wells(tmp_path, 'sol', SEED, 9)


def table_lives(n=2) -> list:
    return [lf for lf in drawn_lives() if lf.w0['fluid_class'] in TABLE_CLASSES][:n]


def own_hook(w) -> dict:
    """A table fluid's values as the draws give them: the hook that changes nothing."""
    return dict(p_sat=w['p_sat'], R_i=w['R_i'], rho_o=w['rho_o'], rho_g=w['rho_g'])


@pytest.mark.parametrize('kind', ['sol', 'nsol'])
def test_a_gor_axis_covers_the_samples_gors(kind):
    """SMP-116: GORs at most 1.25 apart over every GOR that the well's attempts reach, their drift included."""
    for lf in table_lives():
        axis = tw.gor_axis(kind, SEED, lf.k, own_hook(lf.w0))
        g = np.array(axis['gors'])
        assert np.all(np.diff(g) > 0) and np.all(g[1:] / g[:-1] <= tw.GOR_STEP * (1 + 1e-12))
        assert axis['R_i'] == lf.w0['R_i'] and axis['p_r_range'][0] < axis['p_r_range'][1]
        if kind == 'sol':
            w = sol2_well(lf)
            gors = [sol2_attempt(SEED, w, j, a)['gor'] for j in range(0, 200, 7) for a in (0, 1)]
            assert axis['p_r_range'] == pytest.approx([0.98 * w['p_r'], 1.02 * w['p_r']], rel=1e-14)
        else:
            plan, _ = nsol2_plan(lf, lf.T_life, lf.lift_start)
            gors = [nsol2_attempt(lf, plan, j)['gor'] for j in range(0, 200, 9)]
        assert g[0] < min(gors) and max(gors) < g[-1]
        assert g[0] > 0.95 * axis['gor_range'][0] * (1 - 2e-6) and g[-1] < 1.05 * axis['gor_range'][1] * (1 + 2e-6)


def test_the_axes_file_and_the_builds_tables(monkeypatch, tmp_path):
    """SMP-116, steps 1 to 4 with the flash replaced: the table wells, their fluids, the axes of those whose fluid was
    built, and table sets at the axes' GORs, which tables.read_table reads for the generation."""
    a, b = table_lives()
    tw.write_table_wells(tmp_path, 'sol', SEED, b.k + 1)
    (tmp_path / 'fluids').mkdir()
    sep = dict(rho_o=a.w0['rho_o'], rho_g=a.w0['rho_g'], gor=a.w0['R_i'], M_o=200.0, y=[1.0], x=[1.0])
    fluid = dict(well=a.k, status='ok', fluid_class=a.w0['fluid_class'], source='synthetic', composition={}, drawn={})
    fluid.update(hook=own_hook(a.w0), stream=dict(fractions=[], z=[1.0], sep=sep), seconds=dict(fluid=0.5))
    tw.write_json(tmp_path / 'fluids' / f'k{a.k:06d}.json', fluid)
    failed = dict(well=b.k, status='failed', step='fluid', error='x', seconds=dict(fluid=0.1))
    tw.write_json(tmp_path / 'fluids' / f'k{b.k:06d}.json', failed)
    axes = tw.write_axes(tmp_path)
    assert list(axes) == [str(a.k)] and axes[str(a.k)] == json.loads(
        json.dumps(tw.gor_axis('sol', SEED, a.k, own_hook(a.w0)))
    )
    axes[str(a.k)]['kept'] = True  # A well already in the file keeps its axis
    tw.write_json(tmp_path / 'axes.json', axes)
    assert tw.write_axes(tmp_path)[str(a.k)]['kept']

    class Flash:
        def __init__(self, fractions):
            pass

        def recombine(self, sep, gor):
            return gor

    def tabulate(fl, z, sep_gor, T_axis, p_top):
        p = np.arange(10.0, p_top + 1e-9, 10.0)
        P, T = np.meshgrid(p, T_axis, indexing='ij')
        df = pd.DataFrame(dict(p=P.ravel(), T=T.ravel()))
        return df.assign(R_so=0.5 * z, B_o=1.5, Z=0.9, mu_o=1e-3, mu_g=2e-5, sigma=0.01), {}

    monkeypatch.setattr(tb, '_Flash', Flash)
    monkeypatch.setattr(tb, '_tabulate', tabulate)
    spec = json.loads((tmp_path / 'wells.json').read_text())
    for w in spec['wells']:
        if w['well'] in (a.k, b.k):
            tb.build_task((str(tmp_path), w | dict(kind='sol', seed=SEED), axes.get(str(w['well']))))
    table = read_table(tmp_path, a.k, 'sol', SEED)
    assert table['status'] == 'ok' and table['gors'] == axes[str(a.k)]['gors']
    assert table['pvt'].gors == tuple(float(f'{g:.10g}') for g in axes[str(a.k)]['gors'])
    assert read_table(tmp_path, b.k, 'sol', SEED)['step'] == 'fluid'

    # The generation's checks of the folder against the run's draws (SMP-116): the record and the GOR axis
    assert generate._table(SEED, a.k, 'sol', tmp_path)['status'] == 'ok'
    manifest = tmp_path / f'k{a.k:06d}.json'
    man = json.loads(manifest.read_text())
    tw.write_json(manifest, dict(man, gors=man['gors'][:-1]))  # A table built on another axis
    with pytest.raises(TableMismatch, match='another GOR axis'):
        generate._table(SEED, a.k, 'sol', tmp_path)
    tw.write_json(manifest, man)

    # The run starts only where the build is complete (SMP-116), every table well with its manifest
    (tmp_path / f'k{b.k:06d}.json').rename(tmp_path / 'aside.json')
    with pytest.raises(ValueError, match=f'is not complete: 1 table wells have no manifest \\(draw indices {b.k}\\)'):
        runner.run('sol', 1, tmp_path / 'run', SEED, processes=1, tables=tmp_path, log=lambda x: None)


# --- SMP-110: re-solving a pilot ---------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize('kind', ['sol', 'nsol'])
def test_a_pilots_samples_resolve_to_their_features(monkeypatch, tmp_path, kind):
    """A small real run, assembled and read back: each sample, solved again from its rows alone, gives its features
    to the bit (SMP-110, with the same code and backend on the same platform)."""
    monkeypatch.setattr(generate, 'N_SLOTS', 8)
    state = runner.run(kind, 2, tmp_path / 'run', SEED, processes=1, chunk_size=4, log=lambda x: None)
    assert state['complete']
    asm.write_dataset(tmp_path / 'run', tmp_path, 2, allow_dirty=True, attempts=False)
    name = f'manywells-{kind}-2'
    tables = read_dataset(tmp_path / name)
    assert len(tables['samples']) == 16
    for _, s in tables['samples'].iterrows():
        features = resolve_sample(*sample_rows(tables, s['ID'], s['slot']))
        assert features is not None
        for key in FEATURES:
            want = s[key].item() if hasattr(s[key], 'item') else s[key]
            assert features[key] == want or (isinstance(want, float) and math.isnan(want) and math.isnan(features[key]))
