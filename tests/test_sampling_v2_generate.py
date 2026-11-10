"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests for the generation of one v2 well (manywells.sampling.v2.generate; specs/sampling.md, SMP-101 to SMP-106 and
the lift that a solve starts, SMP-90 and SMP-103): an attempt's outcome and reason, the best rate against the floor,
the producibility filter, the life scan, the slots' passes and the limits, and the wells that are dropped before a
solve. The fast tests put stand-ins in place of the solves (a root set, or an attempt's outcome), so that each rule
is tested on the case it covers; the slow tests generate a real well of each dataset end to end.
"""

import dataclasses
import json
import math
import types

import numpy as np
import pytest

from manywells.configurations import V2
from manywells.datasets import schema
from manywells.sampling.v2 import generate
from manywells.sampling.v2 import run as runner
from manywells.sampling.v2 import table_wells as tw
from manywells.sampling.v2.generate import (
    FEATURES,
    FLOOR,
    FLOOR_MARGIN,
    REDRAW_LIMIT,
    SLOT_LIMIT,
    SOLVER_FAILURE_LIMIT,
    generate_nsol2_well,
    generate_sol2_well,
)
from manywells.sampling.v2.inflow import COEF_KEYS
from manywells.sampling.v2.life import lift_on, sol2_well
from manywells.sampling.v2.mapping import WellBuilder
from manywells.sampling.v2.samples import N_SLOTS, nsol2_plan, sol2_attempt
from manywells.sampling.v2.tables import TABLE_CLASSES, TableMismatch
from manywells.solution import Root, RootSet

from .sampling_v2_wells import SEED, drawn_lives, drawn_sol2_wells

RECORD = (  # The keys of an attempt's record besides its inputs and labels (the module's docstring)
    'outcome reason error n_roots n_stable several_stable marches non_finite rejected t_solve t_build'.split()
)
INFO = 'draw_index outcome error best_rate best_rate_w_lg best_rate_outcomes not_producible_reason'.split()
RESULT = ['well', 'stations', 'table', 'nsol', 'info', 'samples', 'log', 'seconds']


def nan_or(a, b) -> bool:
    return a == b or (a != a and b != b)


# --- Stand-ins ---------------------------------------------------------------------------------------------------------
def a_sol2_well(*, oil=None, lifted=None, table=False):
    """The first life of the tests' seed, and its sol-2 well, with the properties asked for."""
    for lf, w in zip(drawn_lives(), drawn_sol2_wells()):
        if oil is not None and lf.oil != oil:
            continue
        if lifted is not None and (w['lift'] == 'gas_lift') != lifted:
            continue
        if (w['fluid_class'] in TABLE_CLASSES) != table:
            continue
        return lf, w
    raise LookupError('no such well among the drawn lives')


def solve_inputs(lf, w) -> tuple:
    """The WellProperties and BoundaryConditions of a sol-2 well's nominal state, with the choke open."""
    inp = dict(generate._sol2_nominal(w), u=1.0, w_lg=0.0, lift_on=False, p_s=w['p_s'])
    return WellBuilder(lf.w0, lf.stations, design=lf.design).case(inp)


def a_root(label='stable', w_res=5.0, w_g_res=0.5, p_0=200.0, n_points=101) -> Root:
    """A root of n_points points, the pressure falling from p_0 to a tenth of it."""
    X = np.tile([0.0, 3.0, 1.0, 0.3, 50.0, 800.0, 350.0], (n_points, 1))
    X[:, 0] = np.linspace(p_0, 0.1 * p_0, n_points)
    slope = -1.0 if label == 'stable' else 1.0
    regimes = ('bubbly',) * n_points
    return Root(x=X.ravel(), label=label, slope=slope, choked=False, flow_regime=regimes, w_res=w_res, w_g_res=w_g_res)


class StandInFinder:
    """In place of RustRootFinder: each search returns the roots and the counts given, or raises error."""

    def __init__(self, roots=(), rejected=0, non_finite=0, error=None):
        self.roots, self.error, self.wells = roots, error, []
        self.counts = dict(marches=7, states=700, rejected=rejected, non_finite=non_finite, edge_refinements=0)

    def __call__(self, wp):
        self.wells.append(wp)
        if self.error is not None:
            raise self.error
        return self

    def find(self, bc, x_guess=None):
        return RootSet.of(self.roots)


def stand_in_attempt(outcome_of):
    """In place of generate._attempt: outcome_of(inp) gives an attempt's outcome and its oil equivalent (Sm3/d)."""
    calls = []

    def attempt(builder, inp):
        calls.append(inp)
        outcome, q_oe = outcome_of(inp)
        rec = dict(inp, outcome=outcome, reason=None, error=None)
        if outcome == 'operating_point':
            rec['q_oe'] = q_oe
        return rec

    return attempt, calls


# --- SMP-104: an attempt's outcome ------------------------------------------------------------------------------------
def test_an_operating_point(monkeypatch):
    """The stable root of a valid row is the operating point: its features and the search's counts are recorded."""
    lf, w = a_sol2_well(oil=True)
    wp, bc = solve_inputs(lf, w)
    finder = StandInFinder([a_root('unstable', p_0=150.0), a_root(p_0=220.0)], non_finite=3)
    monkeypatch.setattr(generate, 'RustRootFinder', finder)
    out = generate._solve(wp, bc)
    assert (out['outcome'], out['reason'], out['error']) == ('operating_point', None, None)
    assert (out['n_roots'], out['n_stable'], out['several_stable']) == (2, 1, False)
    assert (out['marches'], out['non_finite'], out['rejected']) == (7, 3, 0)
    row = out['row']
    assert list(row) == list(FEATURES) and row['PBH'] == 220.0 and row['WLIQ'] == 5.0 and row['WGAS'] == 0.5
    assert math.isclose(row['FGAS'] + row['FOIL'] + row['FWAT'], 1.0, rel_tol=1e-12)
    assert finder.wells == [wp]


@pytest.mark.parametrize(
    'roots, rejected, outcome, reason',
    [
        ([], 0, 'no_operating_point', 'no_root'),
        ([], 2, 'solver_failure', 'incomplete_scan'),
        (['unstable'], 0, 'solver_failure', 'unstable_only'),
        (['unstable'], 1, 'solver_failure', 'unstable_only'),
        (['stable', 'stable'], 0, 'operating_point', None),
    ],
)
def test_the_outcome_of_a_root_set(monkeypatch, roots, rejected, outcome, reason):
    """SMP-104: no root is no operating point unless the search rejected a sign change, whatever its non-finite scan
    points; unstable roots alone are a missed root; of several stable roots, the one with the lowest p_0 (SOL-6)."""
    lf, w = a_sol2_well(oil=True)
    wp, bc = solve_inputs(lf, w)
    found = [a_root(label, p_0=200.0 + 10 * i) for i, label in enumerate(roots)]
    monkeypatch.setattr(generate, 'RustRootFinder', StandInFinder(found, rejected=rejected, non_finite=4))
    out = generate._solve(wp, bc)
    assert (out['outcome'], out['reason']) == (outcome, reason)
    assert (out['n_roots'], out['n_stable']) == (len(roots), roots.count('stable'))
    assert out['several_stable'] == (roots.count('stable') > 1)
    assert (out['row'] is not None) == (outcome == 'operating_point')
    if outcome == 'operating_point':
        assert out['row']['PBH'] == 200.0


@pytest.mark.parametrize(
    'root',
    [
        a_root(w_res=-1e-3),  # a negative liquid rate
        a_root(w_g_res=-1e-3),  # a negative gas rate
        a_root(p_0=math.nan),  # a non-finite value
    ],
)
def test_an_invalid_row_is_a_solver_failure(monkeypatch, root):
    lf, w = a_sol2_well(oil=True)
    wp, bc = solve_inputs(lf, w)
    monkeypatch.setattr(generate, 'RustRootFinder', StandInFinder([root]))
    out = generate._solve(wp, bc)
    assert (out['outcome'], out['reason'], out['row']) == ('solver_failure', 'invalid_row', None)
    assert (out['n_roots'], out['n_stable']) == (1, 1)


def test_a_row_with_a_fraction_outside_zero_to_one_is_invalid():
    row = {key: 0.5 for key in FEATURES} | dict(CHOKED=False, FRBH='bubbly', FRWH='annular')
    assert generate._valid_row(row)
    for key in ('FGAS', 'FOIL', 'FWAT'):
        assert not generate._valid_row(row | {key: 1.0 + 1e-12}) and not generate._valid_row(row | {key: -1e-12})
    assert generate._valid_row(row | dict(WLIQ=0.0, WGAS=0.0))
    assert not generate._valid_row(row | dict(QOIL=math.inf))


def test_an_exception_is_a_solver_failure(monkeypatch):
    lf, w = a_sol2_well(oil=True)
    wp, bc = solve_inputs(lf, w)
    monkeypatch.setattr(generate, 'RustRootFinder', StandInFinder(error=RuntimeError('the core stopped')))
    out = generate._solve(wp, bc)
    assert (out['outcome'], out['reason'], out['error']) == (
        'solver_failure',
        'exception',
        'RuntimeError: the core stopped',
    )
    assert out['n_roots'] == 0 and all(math.isnan(out[key]) for key in ('marches', 'non_finite', 'rejected'))


PanicException = type('PanicException', (BaseException,), {'__module__': 'pyo3_runtime'})  # PyO3's, as it raises it


def test_a_panic_of_the_core_is_a_solver_failure(monkeypatch):
    """A panic follows from the inputs, as an exception does: it is the attempt's solver failure, and reproducible."""
    lf, w = a_sol2_well(oil=True)
    wp, bc = solve_inputs(lf, w)
    monkeypatch.setattr(generate, 'RustRootFinder', StandInFinder(error=PanicException('index out of bounds')))
    out = generate._solve(wp, bc)
    assert (out['outcome'], out['reason'], out['error']) == (
        'solver_failure',
        'exception',
        'PanicException: index out of bounds',
    )


@pytest.mark.parametrize('error', [MemoryError(), KeyboardInterrupt(), type('Other', (BaseException,), {})()])
def test_a_memory_error_or_an_interrupt_is_not_an_outcome(monkeypatch, error):
    """A MemoryError is the process's cap, not the inputs': it goes on to the run (run.py), as an interrupt does."""
    lf, w = a_sol2_well(oil=True)
    wp, bc = solve_inputs(lf, w)
    monkeypatch.setattr(generate, 'RustRootFinder', StandInFinder(error=error))
    with pytest.raises(type(error)):
        generate._solve(wp, bc)


def test_no_solve_where_p_r_is_not_above_p_s(monkeypatch):
    """SMP-96 and SMP-104: an attempt with p_r <= p_s has no operating point, without a solve."""
    lf, w = a_sol2_well(oil=True)
    wp, bc = solve_inputs(lf, w)
    finder = StandInFinder([a_root()])
    monkeypatch.setattr(generate, 'RustRootFinder', finder)
    for p_s in (bc.p_r, bc.p_r + 1.0):
        out = generate._solve(wp, dataclasses.replace(bc, p_s=p_s))
        assert (out['outcome'], out['reason'], out['n_roots'], out['t_solve']) == (
            'no_operating_point',
            'p_r_le_p_s',
            0,
            0.0,
        )
    assert finder.wells == []


def test_an_attempts_record(monkeypatch):
    """The record holds the inputs and the solve's outcome, counts and times; an operating point's features and oil
    equivalent; and an input that the model refuses is a solver failure."""
    lf, w = a_sol2_well(oil=True)
    builder = WellBuilder(lf.w0, lf.stations, design=lf.design)
    inp = sol2_attempt(SEED, w, 3)
    monkeypatch.setattr(generate, 'RustRootFinder', StandInFinder([a_root()]))
    rec = generate._attempt(builder, inp)
    assert list(rec) == [*inp, *RECORD, *FEATURES, 'q_oe']
    assert all(nan_or(rec[key], v) for key, v in inp.items())
    assert rec['q_oe'] == 24 * (rec['QOIL'] + rec['QGAS'] / 1000)
    assert rec['t_build'] >= 0 and rec['t_solve'] >= 0

    refused = generate._attempt(builder, dict(inp, h=-1.0))
    assert list(refused) == [*inp, *RECORD]
    assert (refused['outcome'], refused['reason']) == ('solver_failure', 'exception')
    assert refused['error'].startswith('build: ') and refused['n_roots'] == 0 and math.isnan(refused['marches'])


def test_features_are_the_samples_operating_point():
    """The operating point's columns are SMP-30's features but the inputs CHK, PDC and WGL."""
    assert FEATURES == tuple(key for key in schema.FEATURES if key not in ('CHK', 'PDC', 'WGL'))


# --- SMP-102, SMP-103: the best rate ----------------------------------------------------------------------------------
def test_the_floor():
    assert FLOOR == {'platform': 1.6, 'onshore': 1.6, 'subsea': 20.0} and FLOOR_MARGIN == 0.5


def best_rate(monkeypatch, outcomes, lifted=True, fluid_type='black_oil', wlr=0.3, floor=1.6, w_lg_max=2.0):
    """The best rate with each solve's (outcome, q_oe) given in turn, and its log and the solves' inputs."""
    it = iter(outcomes)
    attempt, calls = stand_in_attempt(lambda inp: next(it))
    monkeypatch.setattr(generate, '_attempt', attempt)
    builder = types.SimpleNamespace(w=dict(fluid_type=fluid_type))
    state = dict(p_r=200.0, gor=100.0, wlr=wlr, wgr=math.nan, p_s_nom=20.0, h=10.0, roughness=1e-5)
    log = []
    out = generate._best_rate(builder, state, lifted, w_lg_max, floor, log, dict(draw_index=4, kind='best_rate'))
    return out, log, calls


def test_the_best_rate_at_three_lift_gas_rates(monkeypatch):
    """SMP-103: at a fully open choke and the nominal downstream pressure, the most oil equivalent over the lift gas
    at 0, 1/2 and 1 of the capacity; each solve is logged with its labels."""
    out, log, calls = best_rate(
        monkeypatch, [('no_operating_point', None), ('operating_point', 5.0), ('operating_point', 3.0)]
    )
    assert out == dict(
        best=5.0, best_w_lg=1.0, outcomes=['no_operating_point', 'operating_point', 'operating_point'], status='pass'
    )
    assert [(c['u'], c['p_s'], c['w_lg'], c['lift_on']) for c in calls] == [
        (1.0, 20.0, f * 2.0, True) for f in (0, 0.5, 1)
    ]
    assert [(r['draw_index'], r['kind'], r['attempt'], r['lift_fraction']) for r in log] == [
        (4, 'best_rate', i, f) for i, f in enumerate((0.0, 0.5, 1.0))
    ]


def test_a_natural_well_is_solved_without_lift_gas(monkeypatch):
    out, log, calls = best_rate(monkeypatch, [('operating_point', 2.0)], lifted=False)
    assert out['status'] == 'pass' and len(calls) == 1 and (calls[0]['w_lg'], calls[0]['lift_on']) == (0.0, False)


@pytest.mark.parametrize(
    'q_oe, floor, status',
    [(0.8, 1.6, 'pass'), (0.8 - 1e-12, 1.6, 'below_floor'), (10.0, 20.0, 'pass'), (9.99, 20.0, 'below_floor')],
)
def test_the_best_rate_against_half_the_floor(monkeypatch, q_oe, floor, status):
    """SMP-102: a well is produced down to half its floor."""
    out, _, _ = best_rate(monkeypatch, [('operating_point', q_oe)], lifted=False, floor=floor)
    assert (out['status'], out['best']) == (status, q_oe)


@pytest.mark.parametrize(
    'outcomes, status',
    [
        (['no_operating_point', 'solver_failure', 'no_operating_point'], 'solver_failure'),
        (['no_operating_point'] * 3, 'no_operating_point'),
        (['solver_failure', 'operating_point', 'solver_failure'], 'pass'),
    ],
)
def test_the_best_rates_status_without_an_operating_point(monkeypatch, outcomes, status):
    out, _, _ = best_rate(monkeypatch, [(o, 4.0) for o in outcomes])
    assert out['status'] == status and (status == 'pass') == (out['best'] == out['best'])


def test_an_oil_above_the_water_cut_is_not_produced(monkeypatch):
    """SMP-102: an oil whose WLR is above 0.98 is not produced, whatever its rate, and has no best rate; the cap is an
    oil's only."""
    out, _, _ = best_rate(monkeypatch, [('operating_point', 100.0)], lifted=False, wlr=0.981)
    assert out['status'] == 'water_cut' and math.isnan(out['best']) and out['outcomes'] == ['operating_point']
    out, _, _ = best_rate(monkeypatch, [('operating_point', 100.0)], lifted=False, wlr=0.98)
    assert out['status'] == 'pass'
    out, _, _ = best_rate(monkeypatch, [('operating_point', 100.0)], lifted=False, wlr=0.99, fluid_type='gas')
    assert out['status'] == 'pass'


# --- SMP-105, SMP-106: the slots' passes and the limits ---------------------------------------------------------------
def solve_slots(monkeypatch, fails, outcome='slot_limit'):
    """The passes over the slots with fails(j, a) the outcome of a failed attempt, or None for an operating point."""
    attempt, _ = stand_in_attempt(lambda inp: (fails(inp['j'], inp['a']) or 'operating_point', 1.0))
    monkeypatch.setattr(generate, '_attempt', attempt)
    log = []
    accepted, result = generate._solve_slots(3, lambda j, a: dict(j=j, a=a), None, log, outcome)
    assert all((r['draw_index'], r['kind'], r['slot'], r['attempt']) == (3, 'slot', r['j'], r['a']) for r in log)
    assert all(
        accepted[j] is next(r for r in log if r['slot'] == j and r['outcome'] == 'operating_point') for j in accepted
    )
    return accepted, result, [(r['slot'], r['attempt']) for r in log]


def test_every_slot_at_its_first_attempt(monkeypatch):
    accepted, result, order = solve_slots(monkeypatch, lambda j, a: None)
    assert (result, sorted(accepted), order) == ('accepted', list(range(N_SLOTS)), [(j, 0) for j in range(N_SLOTS)])


def test_the_passes(monkeypatch):
    """SMP-105: pass a solves attempt a of each slot still without an operating point, in the order of the slots."""
    needs = {5: 1, 17: 3, 2: 2}  # The attempts that slot j needs without an operating point
    accepted, result, order = solve_slots(
        monkeypatch, lambda j, a: 'no_operating_point' if a < needs.get(j, 0) else None
    )
    assert result == 'accepted' and sorted(accepted) == list(range(N_SLOTS))
    assert order == [(j, 0) for j in range(N_SLOTS)] + [(2, 1), (5, 1), (17, 1), (2, 2), (17, 2), (17, 3)]
    assert {j: accepted[j]['attempt'] for j in needs} == {5: 1, 17: 3, 2: 2}


def test_the_slot_limit(monkeypatch):
    """SMP-106: a slot without an operating point after 50 attempts drops the well, as `slot_limit` in sol-2 and as
    `dead_date` in nsol-2; every other slot keeps its operating point."""
    for name in ('slot_limit', 'dead_date'):
        accepted, result, order = solve_slots(monkeypatch, lambda j, a: 'no_operating_point' if j == 7 else None, name)
        assert result == name and SLOT_LIMIT == 50 and 7 not in accepted and len(accepted) == N_SLOTS - 1
        assert order[N_SLOTS:] == [(7, a) for a in range(1, SLOT_LIMIT)]


def test_the_redraw_limit(monkeypatch):
    """SMP-106: a well may redraw 400 times over its slots; a 401st redraw drops it."""
    assert REDRAW_LIMIT == 400
    accepted, result, order = solve_slots(monkeypatch, lambda j, a: 'no_operating_point' if a < 2 else None)
    assert result == 'accepted' and len(order) == N_SLOTS + REDRAW_LIMIT  # 2 redraws for each of 200 slots
    accepted, result, order = solve_slots(monkeypatch, lambda j, a: 'no_operating_point' if a < 2 or j == 0 else None)
    assert result == 'redraw_limit' and len(order) == N_SLOTS + REDRAW_LIMIT and order[-1] == (N_SLOTS - 1, 2)
    assert 0 not in accepted and len(accepted) == N_SLOTS - 1


def test_the_solver_failure_limit(monkeypatch):
    """SMP-106: 10 solver failures are allowed, the 11th drops the well; a missing operating point does not count."""
    assert SOLVER_FAILURE_LIMIT == 10
    failing = set(range(0, 200, 20))  # 10 slots

    def fails(j, a):
        if j in failing and a == 0:
            return 'solver_failure'
        return 'no_operating_point' if j % 2 and a == 0 else None

    accepted, result, order = solve_slots(monkeypatch, fails)
    assert result == 'accepted'
    accepted, result, order = solve_slots(monkeypatch, lambda j, a: 'solver_failure' if j in failing | {199} else None)
    assert result == 'solver_failure_limit' and order[-1] == (199, 0) and len(order) == N_SLOTS
    accepted, result, order = solve_slots(monkeypatch, lambda j, a: 'solver_failure' if j == 11 and a < 11 else None)
    assert result == 'solver_failure_limit' and order[-1] == (11, 10)


# --- SMP-101, SMP-90: the life scan -----------------------------------------------------------------------------------
def a_life(T_life=10.0, lift_start=None, rule=None, design=True):
    return types.SimpleNamespace(
        T_life=T_life, lift_start=lift_start, lift_start_rule=rule, design={} if design else None
    )


def scan(life, status_of):
    """The life scan with status_of(i, lifted) the status of the best rate at point i."""
    calls = []

    def best_rate_at(i, t, lifted):
        calls.append((i, t, lifted))
        status = status_of(i, lifted)
        return dict(status=status, best=float(i) if status in ('pass', 'below_floor') else math.nan, outcomes=[i])

    return (*generate._life_scan(life, best_rate_at), calls)


def test_a_life_that_passes_to_its_end():
    """SMP-101: 21 dates i T_life / 20, each at the lift the well has there; the life ends at T_life."""
    life = a_life(lift_start=4.0, rule='drawn_time')
    points, first, T_end, lift_start, rule, calls = scan(life, lambda i, lifted: 'pass')
    dates = [i * 10.0 / 20 for i in range(21)]
    assert [p['t'] for p in points] == [c[1] for c in calls] == pytest.approx(dates, rel=1e-15, abs=1e-15)
    assert [p['lift_on'] for p in points] == [c[2] for c in calls] == [t >= 4.0 for t in dates]
    assert (T_end, lift_start, rule, first['outcomes']) == (10.0, 4.0, 'drawn_time', [0])
    assert all(p['status'] == 'pass' and p['best'] == i for i, p in enumerate(points))


@pytest.mark.parametrize('status', ['below_floor', 'no_operating_point', 'water_cut'])
def test_the_life_ends_at_the_last_point_before_the_first_that_fails(status):
    life = a_life(lift_start=0.0, rule='from_start')
    points, first, T_end, lift_start, rule, calls = scan(life, lambda i, lifted: status if i == 9 else 'pass')
    assert len(points) == 10 and T_end == 8 * 10.0 / 20 and (lift_start, rule) == (0.0, 'from_start')


def test_a_point_with_a_solver_failure_is_skipped():
    """A skipped point neither passes nor ends the life."""
    life = a_life(lift_start=0.0, rule='from_start')
    statuses = {3: 'solver_failure', 7: 'solver_failure', 8: 'no_operating_point'}
    points, _, T_end, *_ = scan(life, lambda i, lifted: statuses.get(i, 'pass'))
    assert [p['status'] for p in points] == ['pass'] * 3 + ['solver_failure'] + ['pass'] * 3 + list(statuses.values())[
        1:
    ]
    assert T_end == 6 * 10.0 / 20
    points, _, T_end, *_ = scan(life, lambda i, lifted: 'solver_failure' if i == 20 else 'pass')
    assert len(points) == 21 and T_end == 10.0


@pytest.mark.parametrize('status', ['below_floor', 'no_operating_point', 'water_cut', 'solver_failure'])
def test_a_life_whose_first_point_does_not_pass(status):
    """SMP-101: the well is dropped at its start (generate._dropped_at_best_rate), and natural flow failing at the
    first point starts no lift."""
    life = a_life()
    points, first, T_end, lift_start, rule, calls = scan(life, lambda i, lifted: status)
    assert len(points) == 1 and T_end is None and (lift_start, rule) == (None, None) and first['status'] == status
    info = generate._info(4)
    assert generate._dropped_at_best_rate(info, first)
    reason = 'best_rate_solver_failure' if status == 'solver_failure' else 'not_producible'
    assert (info['outcome'], info['not_producible_reason']) == (reason, status)


@pytest.mark.parametrize('drawn', [(None, None), (7.5, 'drawn_time'), (9.0, 'static_check')])
@pytest.mark.parametrize('status', ['below_floor', 'no_operating_point'])
def test_rule_3_lifts_a_well_before_its_natural_flow_fails(drawn, status):
    """SMP-90, rule 3: where the natural well fails at a point after the first, lift starts at the point before,
    earlier than the draws' start, and the point is solved again with lift."""
    life = a_life(lift_start=drawn[0], rule=drawn[1])
    points, _, T_end, lift_start, rule, calls = scan(life, lambda i, lifted: 'pass' if lifted or i < 6 else status)
    assert (lift_start, rule, T_end) == (5 * 10.0 / 20, 'end_of_natural_flow', 10.0)
    assert calls[6:8] == [(6, 3.0, False), (6, 3.0, True)] and all(c[2] for c in calls[7:])
    assert len(points) == 21 and points[6] == dict(t=3.0, status='pass', best=6.0, lift_on=True)


def test_rule_3_where_the_lifted_well_fails_too():
    life = a_life()
    points, _, T_end, lift_start, rule, calls = scan(life, lambda i, lifted: 'pass' if i < 4 else 'no_operating_point')
    assert (lift_start, rule, T_end) == (1.5, 'end_of_natural_flow', 1.5) and len(points) == 5
    assert calls[-2:] == [(4, 2.0, False), (4, 2.0, True)] and points[-1]['lift_on']
    points, _, T_end, lift_start, rule, calls = scan(life, lambda i, lifted: 'pass' if i < 1 else 'below_floor')
    assert (lift_start, rule, T_end) == (0.0, 'end_of_natural_flow', 0.0)  # a short life


def test_no_rule_3_without_a_design_or_for_the_water_cut():
    """A well that no rule lifts (a gas well without a lift design) ends where its natural flow fails; a water cut
    ends the life of a natural well as of a lifted one."""
    points, _, T_end, lift_start, rule, calls = scan(
        a_life(design=False), lambda i, lifted: 'pass' if i < 6 else 'below_floor'
    )
    assert (lift_start, rule, T_end, len(calls)) == (None, None, 2.5, 7)
    points, _, T_end, lift_start, rule, calls = scan(a_life(), lambda i, lifted: 'pass' if i < 6 else 'water_cut')
    assert (lift_start, rule, T_end, len(calls)) == (None, None, 2.5, 7)


# --- A sol-2 well -----------------------------------------------------------------------------------------------------
def by_lift(natural, lifted):
    """An attempt's outcome by whether lift is on: (outcome, q_oe) each."""
    return lambda inp: lifted if inp['lift_on'] else natural


def test_a_sol2_wells_best_rate_and_slots(monkeypatch):
    """The best rate at the nominal state, then the slots' attempts, as samples.sol2_attempt draws them."""
    lf, w = a_sol2_well(oil=True, lifted=True)
    attempt, calls = stand_in_attempt(by_lift(None, ('operating_point', 50.0)))
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_sol2_well(SEED, lf.k)
    info = res['info']
    assert list(res) == RESULT and list(info) == INFO and res['nsol'] is None and res['table'] is None
    assert (info['outcome'], info['best_rate'], info['best_rate_w_lg']) == ('accepted', 50.0, 0.0)
    assert info['best_rate_outcomes'] == ['operating_point'] * 3
    assert all(nan_or(res['well'][key], v) for key, v in w.items()) and set(res['well']) == set(w)
    assert res['stations'] == lf.stations and all(r['draw_index'] == lf.k for r in res['log'])
    nominal = dict(p_r=w['p_r'], gor=w['R_go'], wlr=w['wlr'], p_s=w['p_s'], h=w['h'], roughness=w['roughness'])
    for c in calls[:3]:
        assert {key: c[key] for key in nominal} == nominal and (c['u'], c['lift_on']) == (1.0, True)
        assert all(nan_or(c[key], w[key]) for key in COEF_KEYS)
    assert [c['w_lg'] for c in calls[:3]] == [f * w['w_lg_max'] for f in (0.0, 0.5, 1.0)]
    for j, c in enumerate(calls[3:]):
        assert all(nan_or(c[key], v) for key, v in sol2_attempt(SEED, w, j).items())
    assert [r['slot'] for r in res['samples']] == list(range(N_SLOTS)) and len(res['log']) == 3 + N_SLOTS


def test_the_producibility_filter_lifts_a_natural_oil_well(monkeypatch):
    """SMP-103: a natural oil well without an operating point at its best rate takes its life's lift design, with
    `lift_start_rule` `no_natural_flow`, and its slots are drawn lifted."""
    lf, w = a_sol2_well(oil=True, lifted=False)
    attempt, calls = stand_in_attempt(by_lift(('no_operating_point', None), ('operating_point', 50.0)))
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_sol2_well(SEED, lf.k)
    lifted = sol2_well(lf, no_natural_flow=True)
    assert res['info']['outcome'] == 'accepted' and res['info']['best_rate_w_lg'] == 0.0
    assert (res['well']['lift'], res['well']['lift_start_rule']) == ('gas_lift', 'no_natural_flow')
    assert all(nan_or(res['well'][key], v) for key, v in lifted.items()) and set(res['well']) == set(lifted)
    kinds = [r['kind'] for r in res['log']]
    assert kinds == ['best_rate'] + ['best_rate_lifted'] * 3 + ['slot'] * N_SLOTS
    assert res['info']['best_rate_outcomes'] == ['operating_point'] * 3  # the lifted well's
    assert all(c['lift_on'] for c in calls[1:]) and calls[4]['w_lg'] == sol2_attempt(SEED, lifted, 0)['w_lg']


@pytest.mark.parametrize(
    'natural, lifted, outcome, reason',
    [
        (('no_operating_point', None), ('operating_point', 1e-3), 'not_producible', 'below_floor'),
        (('operating_point', 1e-3), ('no_operating_point', None), 'not_producible', 'no_operating_point'),
        (('no_operating_point', None), ('solver_failure', None), 'best_rate_solver_failure', 'solver_failure'),
        (('solver_failure', None), ('operating_point', 50.0), 'best_rate_solver_failure', 'solver_failure'),
    ],
)
def test_a_natural_oil_well_that_does_not_pass(monkeypatch, natural, lifted, outcome, reason):
    """SMP-103: a natural well's solver failure is not lifted; a lifted well that does not pass is dropped."""
    lf, w = a_sol2_well(oil=True, lifted=False)
    attempt, calls = stand_in_attempt(by_lift(natural, lifted))
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_sol2_well(SEED, lf.k)
    assert (res['info']['outcome'], res['info']['not_producible_reason']) == (outcome, reason)
    assert res['samples'] == [] and all(r['kind'] != 'slot' for r in res['log'])
    assert len(res['log']) == (1 if natural[0] == 'solver_failure' else 4)


def test_a_gas_well_is_not_lifted_by_the_filter(monkeypatch):
    lf, w = a_sol2_well(oil=False, lifted=False)
    attempt, calls = stand_in_attempt(by_lift(('no_operating_point', None), ('operating_point', 50.0)))
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_sol2_well(SEED, lf.k)
    assert (res['info']['outcome'], res['well']['lift'], len(res['log'])) == ('not_producible', 'natural', 1)


# --- An nsol-2 well ---------------------------------------------------------------------------------------------------
def an_nsol2_life():
    """The first black oil's life of the tests' seed that is natural at its start and whose draws start no lift
    before 0.6 of its life."""
    return next(
        lf
        for lf in drawn_lives()
        if lf.w0['fluid_class'] == 'black_oil'
        and lf.lift_start != 0.0
        and (lf.lift_start is None or lf.lift_start > 0.6 * lf.T_life)
    )


def test_an_nsol2_wells_scan_lift_and_slots(monkeypatch):
    """The life scan at the life's states, rule 3's lift from the scan, and the plan's slots with that lift."""
    lf = an_nsol2_life()
    T = lf.T_life
    flows = lambda inp: inp['lift_on'] or inp['t'] < 0.32 * T  # noqa: E731
    attempt, calls = stand_in_attempt(
        lambda inp: ('operating_point', 50.0) if flows(inp) else ('no_operating_point', None)
    )
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    info, nsol = res['info'], res['nsol']
    assert list(res) == RESULT and list(info) == INFO + ['scan', 'T_end', 'p_change']
    dates = np.linspace(0.0, T, 21)
    i_fail = 7  # The first date at or past 0.32 T_life
    assert (nsol['lift_start'], nsol['lift_start_rule']) == (float(dates[i_fail - 1]), 'end_of_natural_flow')
    assert {
        key: v for key, v in nsol.items() if key not in ('lift_start', 'lift_start_rule', 'design')
    } == lf.parameters()
    assert nsol['design'] == {key: lf.design[key] for key in nsol['design']} and 'w_lg_max' in nsol['design']
    assert (info['outcome'], info['T_end'], info['best_rate'], info['best_rate_w_lg']) == ('accepted', T, 50.0, 0.0)
    assert [p['t'] for p in info['scan']] == [float(t) for t in dates]
    assert [p['lift_on'] for p in info['scan']] == [i >= i_fail for i in range(21)]
    scan_calls = [c for c in calls if 'test' not in c]  # The slots' inputs hold their test period
    assert len(scan_calls) == i_fail + 1 + 3 * (21 - i_fail)  # A lifted point at the three lift-gas rates
    assert [c['w_lg'] for c in scan_calls[: i_fail + 1]] == [0.0] * (i_fail + 1)
    assert [c['w_lg'] / lf.design['w_lg_max'] for c in scan_calls[i_fail + 1 : i_fail + 4]] == [0.0, 0.5, 1.0]
    for c in scan_calls:
        want = dict(lf.state(c['t']), u=1.0, lift_on=c['lift_on'], p_s=c['p_s_nom'])
        assert set(c) == {*want, 'w_lg'} and all(nan_or(c[key], v) for key, v in want.items())
    plan, p_c = nsol2_plan(lf, T, nsol['lift_start'])
    assert info['p_change'] == p_c and len(res['samples']) == N_SLOTS
    for slot, rec in zip(plan, res['samples']):
        assert (rec['t'], rec['lift_on'], rec['u'], rec['p_s']) == (
            slot['t'],
            lift_on(slot['t'], nsol['lift_start']),
            slot['u'],
            slot['p_s'],
        )


def test_an_nsol2_well_whose_life_ends_before_its_lift_starts_is_natural(monkeypatch):
    """SMP-52, SMP-90 (D-131): where the draws start lift between two scan points and the lifted well fails at the
    later one, the life ends before lift starts; no date of it is lifted, so the well is natural, with no rule."""
    T = lambda lf: lf.T_life  # noqa: E731
    lf = next(
        lf
        for lf in drawn_lives()
        if lf.lift_start_rule in ('drawn_time', 'static_check')
        and (lf.lift_start * 20 / T(lf)) % 1 > 0.1
        and lf.w0['fluid_class'] not in TABLE_CLASSES
    )
    attempt, _ = stand_in_attempt(
        lambda inp: ('no_operating_point', None) if inp['lift_on'] else ('operating_point', 50.0)
    )
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    info, nsol = res['info'], res['nsol']
    dates = np.linspace(0.0, T(lf), 21)
    assert info['T_end'] == dates[dates < lf.lift_start][-1] and info['outcome'] == 'accepted'
    assert info['scan'][-1]['lift_on'] and info['scan'][-1]['status'] == 'no_operating_point'
    assert (nsol['lift_start'], nsol['lift_start_rule']) == (None, None)
    assert not any(rec['lift_on'] for rec in res['samples']) and len(res['samples']) == N_SLOTS
    plan, p_c = nsol2_plan(lf, info['T_end'], None)  # The plan of a natural well: the same draws
    assert info['p_change'] == p_c and [rec['u'] for rec in res['samples']] == [slot['u'] for slot in plan]


def test_an_nsol2_wells_short_life_and_dead_date(monkeypatch):
    """SMP-101: a life that ends at its first point is `short_life`, without slots; SMP-106: a slot that has no
    operating point at 50 attempts is a dead date."""
    lf = an_nsol2_life()
    attempt, _ = stand_in_attempt(
        lambda inp: ('operating_point', 50.0) if inp['t'] == 0.0 else ('no_operating_point', None)
    )
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    assert (res['info']['outcome'], res['info']['T_end'], res['nsol']['lift_start']) == ('short_life', 0.0, 0.0)
    assert res['samples'] == [] and math.isnan(res['info']['p_change']) and len(res['info']['scan']) == 2

    attempt, _ = stand_in_attempt(
        lambda inp: ('no_operating_point', None) if inp.get('test') == 8 else ('operating_point', 50.0)
    )
    monkeypatch.setattr(generate, '_attempt', attempt)
    res = generate_nsol2_well(SEED, lf.k)
    assert res['info']['outcome'] == 'dead_date' and len(res['samples']) == N_SLOTS - 5


# --- Wells dropped before a solve -------------------------------------------------------------------------------------
def no_solve(monkeypatch):
    def attempt(builder, inp):
        raise AssertionError('a well dropped before its solves was solved')

    monkeypatch.setattr(generate, '_attempt', attempt)


def a_table_well() -> int:
    return next(lf.k for lf in drawn_lives() if lf.w0['fluid_class'] in TABLE_CLASSES)


@pytest.mark.parametrize('kind', ['sol', 'nsol'])
def test_a_table_well_without_a_usable_table(monkeypatch, tmp_path, kind):
    """SMP-106, SMP-116: a volatile oil or condensate without a table, or whose table failed, is `table_failed`
    and not solved; its draws are made without a table."""
    no_solve(monkeypatch)
    k = a_table_well()
    run = generate_sol2_well if kind == 'sol' else generate_nsol2_well
    tw.write_table_wells(tmp_path, kind, SEED, k + 1)  # The build's first step, and no table yet
    for tables, step in ((None, 'missing'), (tmp_path, 'missing')):
        res = run(SEED, k, tables)
        assert (res['info']['outcome'], res['table']['status'], res['table']['step']) == (
            'table_failed',
            'failed',
            step,
        )
        assert res['info']['error'] == res['table']['error'] and res['log'] == res['samples'] == []
    (tmp_path / f'k{k:06d}.json').write_text(
        json.dumps(dict(kind=kind, seed=SEED, status='failed', step='tables', error='p/Z falls'))
    )
    res = run(SEED, k, tmp_path)
    assert (res['info']['outcome'], res['info']['error'], res['table']['step']) == (
        'table_failed',
        'p/Z falls',
        'tables',
    )
    assert 'pvt' not in res['table'] and res['well']['fluid_class'] in TABLE_CLASSES
    if kind == 'sol':
        drawn = sol2_well(drawn_lives()[k])
        assert set(res['well']) == set(drawn) and all(nan_or(res['well'][key], v) for key, v in drawn.items())
    else:
        assert math.isnan(res['info']['T_end']) and res['info']['scan'] == [] and res['nsol']['T_life'] > 0


def own_hook(w) -> dict:
    """A table fluid's values as the draws give them: the hook that changes nothing."""
    return dict(p_sat=w['p_sat'], R_i=w['R_i'], rho_o=w['rho_o'], rho_g=w['rho_g'])


@pytest.mark.parametrize('kind', ['sol', 'nsol'])
def test_a_folder_of_tables_built_from_other_draws_stops_the_run(monkeypatch, tmp_path, kind):
    """SMP-116: a folder whose record of a table well is not what the run's draws give, as after a rule changed
    since the build, or that records a well of another class, stops the run (TableMismatch): it is no outcome of the
    well. A draw index beyond the folder's range has no table."""
    no_solve(monkeypatch)
    k = a_table_well()
    other = next(lf.k for lf in drawn_lives() if lf.k > k and lf.w0['fluid_class'] not in TABLE_CLASSES)
    run = generate_sol2_well if kind == 'sol' else generate_nsol2_well
    spec = tw.write_table_wells(tmp_path, kind, SEED, other + 1)
    path = tmp_path / 'wells.json'
    rec = next(w for w in spec['wells'] if w['well'] == k)
    for key, value in (('p_i', rec['p_i'] + 1.0), ('T_axis', rec['T_axis'][:-1]), ('composition_seed', 7)):
        tw.write_json(path, dict(spec, wells=[dict(w, **{key: value}) if w is rec else w for w in spec['wells']]))
        with pytest.raises(TableMismatch, match=f'were not built from the draws of draw index {k}'):
            run(SEED, k, tmp_path)
        with pytest.raises(TableMismatch):
            runner.run_well(kind, SEED, k, tmp_path)  # Not a crash
    stray = dict(rec, well=other)  # A well that the draws give another class
    tw.write_json(path, dict(spec, wells=spec['wells'] + [stray]))
    with pytest.raises(TableMismatch, match=f'draw index {other}'):
        run(SEED, other, tmp_path)
    tw.write_json(path, dict(spec, wells_range=[0, k]))  # The folder covers draw indices below k only
    assert run(SEED, k, tmp_path)['table']['step'] == 'missing'


@pytest.mark.parametrize('kind', ['sol', 'nsol'])
def test_a_gor_off_the_tables_axis_at_the_nominal_state_is_invalid_input(monkeypatch, kind):
    """SMP-106: where the table refuses the GOR of the well's nominal state (nsol-2: its start), the well is
    `invalid_input`, not a crash; its draws are made without the table's coefficients."""
    no_solve(monkeypatch)
    lf = next(lf for lf in drawn_lives() if lf.w0['fluid_class'] == 'volatile_oil')

    def refuse(gor):
        raise ValueError(f"gor {gor} is outside the PVT table's GOR axis, 1.0 to 2.0 Sm3/Sm3")

    pvt = types.SimpleNamespace(rho_o=lf.w0['rho_o'], rho_g=lf.w0['rho_g'], at_gor=refuse)
    table = dict(status='ok', hook=own_hook(lf.w0), pvt=pvt)
    monkeypatch.setattr(generate, '_table', lambda seed, k, kind, tables: table)
    res = (generate_sol2_well if kind == 'sol' else generate_nsol2_well)(SEED, lf.k)
    assert res['info']['outcome'] == 'invalid_input' and "outside the PVT table's GOR axis" in res['info']['error']
    assert res['log'] == [] and res['well']['well'] == lf.k
    if kind == 'sol':
        assert all(math.isnan(res['well'][key]) for key in COEF_KEYS) and 'date' in res['well']


def test_a_well_of_another_class_takes_no_table(monkeypatch):
    lf, _ = a_sol2_well(oil=True, table=False)
    attempt, _ = stand_in_attempt(lambda inp: ('no_operating_point', None))
    monkeypatch.setattr(generate, '_attempt', attempt)
    assert generate_sol2_well(SEED, lf.k, tables=None)['table'] is None


@pytest.mark.parametrize('run', [generate_sol2_well, generate_nsol2_well])
def test_a_well_outside_the_configuration_is_invalid_input(monkeypatch, run):
    """SMP-106: a well that fails configurations.check at its nominal state (nsol-2: its start) is not solved."""
    no_solve(monkeypatch)

    def refuse(wp, configuration):
        assert configuration == V2
        raise ValueError('well is not in the v2.0.0 configuration: a stand-in difference')

    monkeypatch.setattr(generate, 'check', refuse)
    lf, _ = a_sol2_well(oil=True)
    res = run(SEED, lf.k)
    assert (res['info']['outcome'], res['log']) == ('invalid_input', [])
    assert res['info']['error'].endswith('a stand-in difference')


# --- Real wells, end to end -------------------------------------------------------------------------------------------
N_REAL_SLOTS = 12  # The slots of the slow tests' wells


def check_records(res):
    """What every record of a real well's log holds, by its outcome and reason (SMP-104)."""
    for rec in res['log']:
        outcome, reason = rec['outcome'], rec['reason']
        searched = reason not in ('p_r_le_p_s', 'exception')
        assert all(math.isfinite(rec[key]) == searched for key in ('marches', 'non_finite', 'rejected'))
        if outcome == 'operating_point':
            assert reason is None and rec['n_stable'] >= 1 and generate._valid_row({key: rec[key] for key in FEATURES})
        elif reason == 'no_root':
            assert outcome == 'no_operating_point' and rec['n_roots'] == 0 and rec['rejected'] == 0
        elif reason == 'p_r_le_p_s':
            assert outcome == 'no_operating_point' and rec['p_r'] <= rec['p_s']
        elif reason == 'incomplete_scan':
            assert outcome == 'solver_failure' and rec['n_roots'] == 0 and rec['rejected'] > 0
        elif reason == 'unstable_only':
            assert outcome == 'solver_failure' and rec['n_roots'] > 0 == rec['n_stable']
        else:
            assert (outcome, reason) in (('solver_failure', 'exception'), ('solver_failure', 'invalid_row'))
    kinds = {r['kind'] for r in res['log']}
    kind = 'scan' if 'scan' in kinds else ('best_rate_lifted' if 'best_rate_lifted' in kinds else 'best_rate')
    first = [
        r for r in res['log'] if r['kind'] == kind and r['slot'] == -1
    ]  # The best rate's solves (nsol-2: at t = 0)
    ops = [r['q_oe'] for r in first if r['outcome'] == 'operating_point']
    assert nan_or(res['info']['best_rate'], max(ops) if ops else math.nan)
    assert res['info']['best_rate_outcomes'] == [r['outcome'] for r in first]


def first_accepted(run, ks, monkeypatch):
    monkeypatch.setattr(generate, 'N_SLOTS', N_REAL_SLOTS)
    for k in ks:
        res = run(SEED, k)
        if res['info']['outcome'] == 'accepted':
            return k, res
    raise LookupError('no well was accepted')


@pytest.mark.slow
def test_a_real_sol2_well(monkeypatch):
    """A natural black oil and a lifted one, each solved end to end and again to the bit."""
    for lifted in (False, True):
        ks = [
            lf.k
            for lf, w in zip(drawn_lives(), drawn_sol2_wells())
            if w['fluid_class'] == 'black_oil' and (w['lift'] == 'gas_lift') == lifted
        ]
        k, res = first_accepted(generate_sol2_well, ks[:6], monkeypatch)
        check_records(res)
        assert [r['slot'] for r in res['samples']] == list(range(N_REAL_SLOTS))
        assert all(r['outcome'] == 'operating_point' and r['kind'] == 'slot' for r in res['samples'])
        again = generate_sol2_well(SEED, k)  # A rerun gives the same attempts and solves, to the bit (SMP-109)
        assert len(again['log']) == len(res['log'])
        for r1, r2 in zip(again['log'], res['log']):
            assert set(r1) == set(r2) and all(nan_or(r1[a], r2[a]) for a in r1 if a not in ('t_build', 't_solve'))


@pytest.mark.slow
def test_a_real_nsol2_well(monkeypatch):
    """An oil well's life scan and slots, solved end to end."""
    ks = [lf.k for lf in drawn_lives() if lf.w0['fluid_class'] == 'black_oil']
    k, res = first_accepted(generate_nsol2_well, ks[:6], monkeypatch)
    check_records(res)
    info, nsol = res['info'], res['nsol']
    scan_recs = [r for r in res['log'] if r['kind'] == 'scan']
    assert {r['slot'] for r in scan_recs} == {-1 - i for i in range(len(info['scan']))}
    assert 0 < info['T_end'] <= nsol['T_life'] and len(res['samples']) == N_REAL_SLOTS
    assert all(
        rec['t'] <= info['T_end'] and rec['lift_on'] == lift_on(rec['t'], nsol['lift_start']) for rec in res['samples']
    )
