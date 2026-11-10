"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests of the v2 datasets' table fluids (specs/sampling.md, SMP-112 to SMP-116): the build of
manywells.sampling.v2.table_build, as far as it needs no flash (the source, the heptanes-plus, the composition's
draws, the grid, the repair of p/Z and its limit, and the files of a well's steps, with the flash replaced), and
manywells.sampling.v2.tables, which reads a built table set. NeqSim runs in an environment of its own (the module's
docstring), so the one test with a flash is skipped where NeqSim is not installed, as in CI.
"""

import ast
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from manywells import units
from manywells.pvt.fluid import FluidModel
from manywells.pvt.table import PVTTable, PVTTableSet
from manywells.sampling.v2 import table_build as tb
from manywells.sampling.v2.tables import read_table

COLUMNS = ['p', 'T', 'R_so', 'B_o', 'Z', 'mu_o', 'mu_g', 'sigma']


# ---------------------------------------------------------------------------
# The module: what it may import, and its copies of ManyWells' constants
# ---------------------------------------------------------------------------


def test_the_module_imports_no_manywells_and_neqsim_only_where_it_flashes():
    """The build runs by its path in NeqSim's environment, which has the standard library, numpy and pandas."""
    tree = ast.parse(pathlib.Path(tb.__file__).read_text())

    def modules(nodes):
        for node in nodes:
            if isinstance(node, ast.Import):
                yield from (a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, 'a relative import needs the package'
                yield node.module.split('.')[0]

    top = set(modules(tree.body))
    assert top <= set(sys.stdlib_module_names) | {'numpy', 'pandas'}
    assert set(modules(ast.walk(tree))) - top == {'neqsim', 'jpype'}


def test_the_constants_are_manywells():
    assert (tb.R_UNIVERSAL, tb.M_AIR, tb.CF_BAR) == (units.R_UNIVERSAL, units.M_AIR, units.CF_BAR)
    assert (tb.P_REF, tb.T_REF) == (units.P_REF, units.T_REF)
    assert tb.P_SC == 1.01325


# ---------------------------------------------------------------------------
# SMP-112, SMP-114: the source and the published examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('cls', ['gas_condensate', 'volatile_oil'])
def test_the_source_is_an_example_of_the_class_with_probability_p_example(cls):
    keys = [name for name, ex in tb.EXAMPLES.items() if ex['cls'] == cls]
    assert len(keys) == 3 and tuple(keys) == tb.CLASSES[cls]['templates']
    assert tb.P_EXAMPLE == 0.2
    assert tb.fluid_source(cls, 0.2, 0.5) == 'synthetic' and tb.fluid_source(cls, 0.999, 0.0) == 'synthetic'
    assert [tb.fluid_source(cls, 0.1999, u) for u in (0.0, 0.3333, 0.3334, 0.6666, 0.6667, 0.9999)] == [
        keys[0],
        keys[0],
        keys[1],
        keys[1],
        keys[2],
        keys[2],
    ]
    u = np.random.default_rng(112).random((20000, 2))
    sources = pd.Series([tb.fluid_source(cls, a, b) for a, b in u]).value_counts(normalize=True)
    assert sources['synthetic'] == pytest.approx(0.8, abs=0.01)
    assert all(sources[k] == pytest.approx(0.2 / 3, abs=0.006) for k in keys)


@pytest.mark.parametrize('name', list(tb.EXAMPLES))
def test_an_examples_composition(name):
    """The published mole percents add to 100, and the stream keeps them; a lumped heptanes-plus is split in five
    with its molar mass and, by Soreide's Cf, its gravity."""
    ex = tb.EXAMPLES[name]
    heavy = [ex['c7'][0]] if 'c7' in ex else [x for x, _, _ in ex['tbp']]
    assert len(ex['light']) == len(tb.LIGHT) == 10
    assert sum(ex['light']) + sum(heavy) == pytest.approx(100.0, abs=0.011)
    fractions, z = tb._example_stream(name)
    assert z.sum() == pytest.approx(1.0, abs=1e-14) and len(z) == 10 + len(fractions)
    assert z[:10] == pytest.approx(np.array(ex['light']) / (sum(ex['light']) + sum(heavy)), rel=1e-12)
    assert [f[0] for f in fractions] == [f'F{i + 1}' for i in range(len(fractions))]
    M, g = np.array([f[1] for f in fractions]), np.array([f[2] for f in fractions])
    if 'tbp' in ex:
        assert [(m, s) for _, m, s in ex['tbp']] == list(zip(M, g))
        assert z[10:] == pytest.approx(np.array(heavy) / (sum(ex['light']) + sum(heavy)), rel=1e-12)
    else:
        x7, M7, g7 = ex['c7']
        zf = z[10:] / z[10:].sum()
        assert len(fractions) == 5 and z[10:].sum() == pytest.approx(x7 / (sum(ex['light']) + x7), rel=1e-12)
        assert np.dot(zf, M) == pytest.approx(M7, rel=1e-12)
        assert np.dot(zf, M) / np.dot(zf, M / g) == pytest.approx(g7, abs=1e-12)  # volume-additive


# ---------------------------------------------------------------------------
# SMP-113: the heptanes-plus and the composition's draws
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('M7, alpha', [(150.0, 0.5), (184.0, 1.0), (200.0, 1.7), (240.0, 2.5)])
def test_the_gamma_split_keeps_the_moles_and_the_molar_mass(M7, alpha):
    z, M = tb._gamma_split(M7, alpha)
    assert z.shape == M.shape == (5,) and np.all(z > 0) and np.all(np.diff(M) > 0)
    assert z.sum() == pytest.approx(1.0, abs=1e-14)
    assert np.dot(z, M) == pytest.approx(M7, rel=1e-13)
    # Each fraction's molar mass lies between its bounds, but for the rescaling to M7's mean, below 1%
    edges = np.array([110.0 * (1 - 1 / (1 + 4 / alpha**0.7)), *tb.GAMMA_BOUNDS, np.inf])
    assert np.all(M > 0.99 * edges[:-1]) and np.all(M < 1.01 * edges[1:])


def test_the_gamma_split_at_alpha_one_is_the_exponential_distribution():
    """With alpha = 1, eta = 88 g/mol and the density is exp(-(M - eta)/beta)/beta, whose integrals are known."""
    M7 = 200.0
    z, M = tb._gamma_split(M7, 1.0)
    eta, beta = 88.0, M7 - 88.0
    edges = np.array([eta, *tb.GAMMA_BOUNDS, np.inf])
    cdf = 1 - np.exp(-(edges - eta) / beta)
    assert z == pytest.approx(np.diff(cdf), rel=5e-3)
    lo, hi = edges[:-1], np.minimum(edges[1:], eta + 60 * beta)
    mean = ((lo + beta) * np.exp(-(lo - eta) / beta) - (hi + beta) * np.exp(-(hi - eta) / beta)) / np.diff(cdf)
    assert M == pytest.approx(mean, rel=5e-3)


def test_soreides_gravities():
    assert tb._soreide([67.0], 0.3) == pytest.approx([0.2855 + 0.3])
    g = tb._soreide(np.array([100.0, 150.0, 250.0, 500.0]), 0.29)
    assert g == pytest.approx([0.2855 + 0.29 * (m - 66.0) ** 0.13 for m in (100.0, 150.0, 250.0, 500.0)], rel=1e-15)
    assert np.all(np.diff(g) > 0) and 0.7 < g[0] < g[-1] < 0.95
    assert np.all(tb._soreide([150.0], 0.31) > tb._soreide([150.0], 0.29))


@pytest.mark.parametrize('cls', ['gas_condensate', 'volatile_oil'])
def test_a_compositions_draws(cls):
    """The weights, then z7, M7 and alpha, from the generator in this order; within the class's ranges."""
    c = tb.CLASSES[cls]
    for seed in range(20):
        d = tb._draw_composition(cls, np.random.default_rng(seed))
        r = np.random.default_rng(seed)
        w = r.dirichlet(np.ones(3))
        assert np.array_equal(d['w'], w) and w.sum() == pytest.approx(1.0)
        assert (d['z7'], d['M7'], d['alpha']) == (r.uniform(*c['z7']), r.uniform(*c['M7']), r.uniform(0.5, 2.5))
        assert c['z7'][0] <= d['z7'] < c['z7'][1] and c['M7'][0] <= d['M7'] < c['M7'][1] and 0.5 <= d['alpha'] < 2.5
        assert d['light'].shape == (10,) and d['light'].sum() == pytest.approx(1.0, abs=1e-14)
        lights = [np.array(tb.EXAMPLES[t]['light']) / sum(tb.EXAMPLES[t]['light']) for t in c['templates']]
        assert np.all(d['light'] >= np.min(lights, axis=0) - 1e-15) and np.all(
            d['light'] <= np.max(lights, axis=0) + 1e-15
        )
        zf, Ms = tb._gamma_split(d['M7'], d['alpha'])
        assert np.array_equal(d['zf'], zf) and np.array_equal(d['Ms'], Ms)


def test_a_composition_with_its_levers():
    """Cf sets the fractions' gravities; the factor scales C2 to C6 among the light ends, which keep their share."""
    d = tb._draw_composition('volatile_oil', np.random.default_rng(113))
    fractions, z = tb._compose(d, 0.29, 0.0)
    assert z.sum() == pytest.approx(1.0, abs=1e-14) and z[10:].sum() == pytest.approx(d['z7'], rel=1e-13)
    assert z[:10] == pytest.approx(d['light'] * (1 - d['z7']), rel=1e-13)
    assert z[10:] == pytest.approx(d['zf'] * d['z7'], rel=1e-13)
    assert [(n, m) for n, m, _ in fractions] == [(f'F{i + 1}', m) for i, m in enumerate(d['Ms'])]
    assert [g for _, _, g in fractions] == pytest.approx(tb._soreide(d['Ms'], 0.29))
    _, z2 = tb._compose(d, 0.29, math.log(2.0))
    assert z2.sum() == pytest.approx(1.0, abs=1e-14) and z2[10:] == pytest.approx(z[10:], rel=1e-13)
    ratio = z2[:10] / z[:10]
    assert ratio[3:] == pytest.approx(2 * ratio[0], rel=1e-12) and ratio[:3] == pytest.approx(ratio[0], rel=1e-12)
    assert ratio[0] < 1
    assert d['light'].sum() == pytest.approx(1.0, abs=1e-14)  # the draw is not changed


# ---------------------------------------------------------------------------
# SMP-116: the grid
# ---------------------------------------------------------------------------


def test_the_wells_grid():
    T, p_top = tb.well_grid(T_s=288.0, T_r=390.4, p_i=300.0)
    assert T.tolist() == [float(t) for t in range(270, 401, 10)] and p_top == 330
    T, p_top = tb.well_grid(T_s=279.67, T_r=315.99, p_i=286.73)
    assert T.tolist() == [260.0, 270.0, 280.0, 290.0, 300.0, 310.0, 320.0] and p_top == 320
    # Below T_s - 15 K and above T_r + 1 K, and above 1.05 p_i by at least a step
    rng = np.random.default_rng(116)
    for T_s, dT, p_i in zip(rng.uniform(275, 300, 200), rng.uniform(5, 150, 200), rng.uniform(30, 950, 200)):
        T, p_top = tb.well_grid(T_s, T_s + dT, p_i)
        assert T[0] <= T_s - 15 < T[0] + 10 and T[-1] - 10 < T_s + dT + 1 <= T[-1]
        assert np.allclose(np.diff(T), 10.0) and np.all(T % 10 == 0)
        assert 1.05 * p_i + 10 <= p_top < 1.05 * p_i + 20 and p_top % 10 == 0


def test_the_pressures_of_a_table():
    assert tb._pressure_axis(50.0).tolist() == [1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 30.0, 40.0, 50.0]
    p = tb._pressure_axis(50.0, [33.33349, 20.0004, 20.0, 46.0])
    assert p.tolist() == [1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 30.0, 33.333, 40.0, 46.0, 50.0]
    assert tb._pressure_axis(330).tolist()[-3:] == [310.0, 320.0, 330.0]


# ---------------------------------------------------------------------------
# SMP-116: the repair of p/Z
# ---------------------------------------------------------------------------


def frame(pz, p=None, T=(300.0, 350.0), gor=None):
    """A table's rows, with Z such that p/Z is pz: one list for every temperature, or one list per temperature."""
    rows = []
    for k, t in enumerate(T):
        col = pz[k] if isinstance(pz[0], (list, tuple)) else pz
        pressures = p or [10.0 * (i + 1) for i in range(len(col))]
        for x, y in zip(pressures, col):
            rows.append(
                dict(p=x, T=t, R_so=50.0 + 0.5 * x, B_o=1.1 + 1e-3 * x, Z=x / y, mu_o=1e-3, mu_g=1.5e-5, sigma=0.01)
            )
    df = pd.DataFrame(rows)
    return df if gor is None else df.assign(gor=gor)


RISING = [12.0, 25.0, 39.0, 54.0, 70.0, 87.0]
P40 = [5.0 * (i + 1) for i in range(40)]  # 40 pressures, of which the repair may drop two
RISING40 = [x * (1 + 1e-3 * x) for x in P40]


def changed(pz, **nodes):
    """pz with the values of some nodes replaced: changed(pz, i10=500.0) has 500 at index 10."""
    out = list(pz)
    for key, value in nodes.items():
        out[int(key[1:])] = value
    return out


def test_a_table_whose_pz_rises_is_kept():
    df = frame(RISING)
    assert tb._falling_pz_pressures(df) == []
    out, dropped = tb._repair(df, 100.0)
    assert out is df and dropped == []


def test_a_spike_and_a_dip_are_dropped_at_every_temperature():
    spike = [12.0, 25.0, 39.0, 95.0, 70.0, 87.0]  # at 40 bar, above the two nodes after it
    dip = [12.0, 25.0, 9.0, 54.0, 70.0, 87.0]  # at 30 bar, below the two nodes before it
    assert tb._falling_pz_pressures(frame(spike)) == [40.0]
    assert tb._falling_pz_pressures(frame(dip)) == [30.0]
    df = frame([spike, dip])
    assert tb._falling_pz_pressures(df) == [30.0, 40.0]
    assert tb._falling_pz_pressures(df.sample(frac=1.0, random_state=1)) == [30.0, 40.0]  # whatever the rows' order
    # The repair, on a table large enough to lose two pressures: a spike at 55 bar at 300 K, a dip at 130 bar at 350 K
    df = frame([changed(RISING40, i10=500.0), changed(RISING40, i25=1.0)], p=P40)
    out, dropped = tb._repair(df, 100.0)
    assert dropped == [55.0, 130.0]
    assert out.groupby('T').p.apply(list).tolist() == [[x for x in P40 if x not in (55.0, 130.0)]] * 2  # rectangular
    assert tb._falling_pz_pressures(out) == []
    assert out.equals(df[~df.p.isin([55.0, 130.0])])


def test_of_the_longest_runs_the_one_that_ends_highest_is_kept():
    """A spike at the last node but one: the runs with it and without it are as long, and the spike goes. A node
    equal to the one before it does not rise, and goes."""
    assert tb._falling_pz_pressures(frame([12.0, 25.0, 39.0, 95.0, 54.0])) == [40.0]
    assert tb._falling_pz_pressures(frame([12.0, 25.0, 25.0, 54.0])) == [30.0]
    assert tb._falling_pz_pressures(frame([12.0, 25.0, 39.0, 39.0])) == [30.0]


def test_a_table_that_loses_more_than_five_percent_of_its_pressures_fails():
    """2 of 40 pressures is 5%, which the repair may drop; 3 of 40 is more, as is 3 of the 50 of the pilot's well."""
    assert tb.MAX_DROPPED == 0.05
    out, dropped = tb._repair(frame(changed(RISING40, i10=500.0, i25=1.0), p=P40, T=(300.0,)), 123.456789)
    assert dropped == [55.0, 130.0] and out.p.nunique() == 38
    three = changed(RISING40, i10=500.0, i25=1.0, i30=2.0)
    with pytest.raises(ValueError, match=r'p/Z falls at 3 of 40 pressures at GOR 123\.457$'):
        tb._repair(frame(three, p=P40, T=(300.0,)), 123.456789)
    # The bad pressures of every temperature count together
    with pytest.raises(ValueError, match='p/Z falls at 3 of 40 pressures'):
        tb._repair(frame([changed(RISING40, i10=500.0, i25=1.0), changed(RISING40, i30=2.0)], p=P40), 100.0)
    p50 = [5.0 * (i + 1) for i in range(50)]
    three = changed([x * (1 + 1e-3 * x) for x in p50], i10=500.0, i25=1.0, i30=2.0)
    with pytest.raises(ValueError, match='p/Z falls at 3 of 50 pressures'):
        tb._repair(frame(three, p=p50, T=(300.0,)), 100.0)


# ---------------------------------------------------------------------------
# SMP-116: the files, with the flash replaced
# ---------------------------------------------------------------------------

WELL = dict(
    well=7,
    fluid_class='volatile_oil',
    source='synthetic',
    composition_seed=5143076675080566889,
    api=40.048926838002096,
    sg_gas=0.7677454721877202,
    R_si=441.8863083575895,
    p_sat=45.80543235648302,
    p_i=286.7283950239695,
    T_r=315.9930955013293,
    T_s=279.67362363570203,
    p_sep=21.625668673137067,
    T_sep=303.67930104225684,
    T_axis=[300.0, 350.0],
    p_top=60.0,
    kind='sol',
    seed=2026,
)
SEP = dict(
    rho_o=843.432774214697, rho_g=1.081093467944081, gor=507.81515609482847, M_o=201.3, y=[0.9, 0.1], x=[0.2, 0.8]
)
GORS = [450.123456789012, 562.6543209876543]


def fake_fluid(w):
    """A fluid's record as build_fluid returns it, without a flash."""
    hook = dict(p_sat=274.0072021484375, R_i=507.81515609482847, rho_o=SEP['rho_o'], rho_g=SEP['rho_g'])
    return dict(
        well=w['well'], status='ok', fluid_class=w['fluid_class'], source=w['source'], composition=dict(z7=0.14),
        saturation_kind='bubble', saturation_rule='matched', gor_moved=None, p_sat=hook['p_sat'], R_i=hook['R_i'],
        hook=hook, drawn={k: w[k] for k in ('api', 'sg_gas', 'R_si', 'p_sat')},
        stream=dict(fractions=[['F1', 111.9, 0.76]], z=[0.5, 0.5], sep=SEP), seconds=dict(match=0.25, saturation=0.05),
    )  # fmt: skip


class FakeFlash:
    """In place of the NeqSim system: the stream is its GOR."""

    def __init__(self, fractions):
        self.fractions = fractions

    def recombine(self, sep, gor):
        return gor


def fake_tabulate(pz_of_gor, p=None):
    def tabulate(fl, z, sep_gor, T_axis, p_top):
        assert sep_gor == (SEP['rho_o'], SEP['rho_g'], z) and p_top == WELL['p_top']
        return frame(pz_of_gor(z), p=p, T=tuple(T_axis)), {}

    return tabulate


@pytest.fixture
def no_flash(monkeypatch):
    monkeypatch.setattr(tb, 'build_fluid', fake_fluid)
    monkeypatch.setattr(tb, '_Flash', FakeFlash)
    monkeypatch.setattr(tb, '_tabulate', fake_tabulate(lambda gor: RISING))


def test_a_table_set_is_read_back_as_written(tmp_path):
    """The CSV is PVT-TAB-1's with a gor column: the densities to the bit, and the nodes to ten significant digits."""
    frames = [frame([x * (1 + 1e-3 * i) / 3 for x in RISING], gor=g) for i, g in enumerate(GORS)]
    path = tmp_path / 'k000007.csv'
    assert tb._write_tables(path, WELL, fake_fluid(WELL), frames) == 24
    head = path.read_text().splitlines()[:7]
    assert head == [
        '# ManyWells PVT table (specs/model/pvt/table.md)',
        '# name: k000007_volatile_oil',
        '# source: manywells.sampling.v2.table_build; NeqSim 3.4.0, Peng-Robinson with Peneloux; synthetic, draw index 7',
        '# separation: single stage at 1.01325 bar and 288.15 K',
        '# rho_o: 843.432774214697',
        '# rho_g: 1.081093467944081',
        'gor,p,T,R_so,B_o,Z,mu_o,mu_g,sigma',
    ]
    axis = PVTTable.read_csv(path)
    assert isinstance(axis, PVTTableSet) and (axis.rho_o, axis.rho_g) == (SEP['rho_o'], SEP['rho_g'])
    assert axis.gors == tuple(float(f'{g:.10g}') for g in GORS)
    for t, df in zip(axis.tables, frames):
        assert t.p == (10.0, 20.0, 30.0, 40.0, 50.0, 60.0) and t.T == (300.0, 350.0)
        assert t.name == 'k000007_volatile_oil' and t.separation == 'single stage at 1.01325 bar and 288.15 K'
        for c in t.columns:
            want = df.pivot(index='p', columns='T', values=c).map(lambda v: float(f'{v:.10g}')).to_numpy()
            assert np.array_equal(np.array(getattr(t, c)), want)
    example = fake_fluid(WELL | dict(source='pedersen2015_t1_4'))
    tb._write_tables(path, WELL, example, frames)
    assert path.read_text().splitlines()[2].endswith('Peng-Robinson with Peneloux; pedersen2015_t1_4')


def test_a_wells_steps_and_files(tmp_path, no_flash):
    """The fluid, then nothing until the well has an axis, then the table set and the manifest, then nothing."""
    (tmp_path / 'fluids').mkdir()
    task = (str(tmp_path), dict(WELL), None)
    k, step, status, seconds = tb.build_task(task)
    assert (k, step, status) == (7, 'fluid', 'ok') and seconds >= 0
    fluid = json.loads((tmp_path / 'fluids' / 'k000007.json').read_text())
    assert {k: v for k, v in fluid.items() if k != 'seconds'} == {
        k: v for k, v in fake_fluid(WELL).items() if k != 'seconds'
    }
    assert list(fluid['seconds']) == ['match', 'saturation', 'fluid']
    assert tb.build_task(task) == (7, 'skip', 'ok', 0.0)
    assert not (tmp_path / 'k000007.json').exists() and not list(tmp_path.glob('*.tmp'))
    assert read_table(tmp_path, 7, 'sol', 2026)['step'] == 'missing'

    with_axis = (str(tmp_path), dict(WELL), dict(gors=GORS, R_i=507.8))
    assert tb.build_task(with_axis)[:3] == (7, 'tables', 'ok')
    man = json.loads((tmp_path / 'k000007.json').read_text())
    assert 'stream' not in man and man['hook'] == fluid['hook'] and man['composition'] == fluid['composition']
    assert (man['kind'], man['seed'], man['csv'], man['gors']) == ('sol', 2026, 'k000007.csv', GORS)
    assert (man['nodes'], man['n_T'], man['p_top'], man['dropped_pressures']) == (24, 2, 60.0, {})
    assert list(man['seconds']) == ['match', 'saturation', 'fluid', 'tables']
    assert tb.build_task(with_axis) == (7, 'skip', 'ok', 0.0)

    table = read_table(tmp_path, 7, 'sol', 2026)
    assert table['status'] == 'ok' and table['tables_dir'] == str(tmp_path)
    assert isinstance(table['pvt'], PVTTableSet) and table['pvt'].gors == tuple(float(f'{g:.10g}') for g in GORS)
    assert {k: v for k, v in table.items() if k not in ('pvt', 'tables_dir')} == man
    for kind, seed in (('nsol', 2026), ('sol', 2027)):
        other = read_table(tmp_path, 7, kind, seed)
        assert (other['status'], other['step']) == ('failed', 'missing') and 'pvt' not in other
    assert read_table(tmp_path, 8, 'sol', 2026) == dict(
        status='failed', step='missing', error='no table built for draw index 8'
    )


def test_dropped_pressures_are_recorded_by_gor(tmp_path, no_flash, monkeypatch):
    dip = changed(RISING40, i25=1.0)  # at 130 bar
    monkeypatch.setattr(tb, '_tabulate', fake_tabulate(lambda gor: dip if gor == GORS[1] else RISING40, p=P40))
    (tmp_path / 'fluids').mkdir()
    tb.build_task((str(tmp_path), dict(WELL), None))
    assert tb.build_task((str(tmp_path), dict(WELL), dict(gors=GORS)))[:3] == (7, 'tables', 'ok')
    man = json.loads((tmp_path / 'k000007.json').read_text())
    assert man['dropped_pressures'] == {'562.654': [130.0]} and man['nodes'] == 80 + 78
    low, high = read_table(tmp_path, 7, 'sol', 2026)['pvt'].tables
    assert low.p == tuple(P40) and high.p == tuple(x for x in P40 if x != 130.0)  # each GOR keeps its own grid


def test_a_failed_step_is_recorded(tmp_path, no_flash, monkeypatch):
    """The fluid, the axis and the tables: each failure is written with its step, and the well has no table."""
    (tmp_path / 'fluids').mkdir()

    def fails(w):
        raise ValueError('no stock-tank oil: the stream is all gas at standard conditions')

    # The fluid: the record says so, and the next run writes the manifest without an axis
    monkeypatch.setattr(tb, 'build_fluid', fails)
    w = WELL | dict(well=1)
    assert tb.build_task((str(tmp_path), w, None))[:3] == (1, 'fluid', 'failed')
    rec = json.loads((tmp_path / 'fluids' / 'k000001.json').read_text())
    assert (rec['status'], rec['step']) == ('failed', 'fluid') and rec['error'].strip().endswith('standard conditions')
    assert tb.build_task((str(tmp_path), w, None))[:3] == (1, 'tables', 'failed')
    table = read_table(tmp_path, 1, 'sol', 2026)
    assert (table['status'], table['step']) == ('failed', 'fluid') and 'pvt' not in table

    # The axis: its error goes into the manifest
    monkeypatch.setattr(tb, 'build_fluid', fake_fluid)
    w = WELL | dict(well=2)
    tb.build_task((str(tmp_path), w, None))
    assert tb.build_task((str(tmp_path), w, dict(error='no life')))[:3] == (2, 'tables', 'failed')
    table = read_table(tmp_path, 2, 'sol', 2026)
    assert (table['status'], table['step'], table['error']) == ('failed', 'axis', 'no life')

    # The tables: one GOR over the repair's limit fails the set, and no CSV is written
    bad = [12.0, 25.0, 9.0, 54.0, 5.0, 87.0]
    monkeypatch.setattr(tb, '_tabulate', fake_tabulate(lambda gor: bad if gor == GORS[1] else RISING))
    w = WELL | dict(well=3)
    tb.build_task((str(tmp_path), w, None))
    assert tb.build_task((str(tmp_path), w, dict(gors=GORS)))[:3] == (3, 'tables', 'failed')
    table = read_table(tmp_path, 3, 'sol', 2026)
    assert (table['status'], table['step']) == ('failed', 'tables')
    assert table['error'].strip().endswith('ValueError: p/Z falls at 2 of 6 pressures at GOR 562.654')
    assert not (tmp_path / 'k000003.csv').exists() and 'csv' not in table and 'dropped_pressures' not in table


def test_a_table_that_pvt_tab_1_refuses_fails_when_it_is_read(tmp_path, no_flash, monkeypatch):
    """The repair mends p/Z only: a table with B_o = 0 at a node is built, and refused when ManyWells reads it."""

    def tabulate(fl, z, sep_gor, T_axis, p_top):
        df = frame(RISING, T=tuple(T_axis))
        df.loc[3, 'B_o'] = 0.0
        return df, {}

    monkeypatch.setattr(tb, '_tabulate', tabulate)
    (tmp_path / 'fluids').mkdir()
    tb.build_task((str(tmp_path), dict(WELL), None))
    assert tb.build_task((str(tmp_path), dict(WELL), dict(gors=GORS)))[:3] == (7, 'tables', 'ok')
    table = read_table(tmp_path, 7, 'sol', 2026)
    assert (table['status'], table['step']) == ('failed', 'check') and 'pvt' not in table
    assert table['error'] == 'PVT-TAB-1: B_o must be positive at every node'


# ---------------------------------------------------------------------------
# With NeqSim: one well's fluid and tables
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_well_built_with_neqsim(tmp_path):
    """
    A synthetic volatile oil (the draws of sol-2's draw index 83 at seed 2026, on a coarser grid): its fluid is
    matched in the field, saturated below p_i, and its table set at two GORs passes PVT-TAB-1 and gives the stream's
    phase split at its own GORs.
    """
    pytest.importorskip('neqsim')
    from importlib import metadata

    if metadata.version('neqsim') != tb.NEQSIM_VERSION:
        pytest.skip(f'the tables are made with NeqSim {tb.NEQSIM_VERSION}')
    w = WELL | dict(well=83, T_axis=[280.0, 300.0, 320.0], p_top=320.0)
    (tmp_path / 'fluids').mkdir()
    assert tb.build_task((str(tmp_path), w, None))[:3] == (83, 'fluid', 'ok')
    fluid = json.loads((tmp_path / 'fluids' / 'k000083.json').read_text())
    assert (fluid['saturation_kind'], fluid['saturation_rule']) == ('bubble', 'matched')
    assert fluid['p_sat'] == pytest.approx(274.007, abs=0.01) and fluid['p_sat'] <= w['p_i']
    assert fluid['R_i'] == pytest.approx(507.815, abs=0.01) == fluid['hook']['R_i']
    assert fluid['field']['gor'] == pytest.approx(w['R_si'], rel=0.02)
    assert fluid['field']['api'] == pytest.approx(w['api'], abs=1.0)
    assert fluid['composition']['api_error'] == pytest.approx(0.0, abs=0.5)
    assert 0.125 < fluid['c7_mol'] < 0.30  # McCain's limits for a volatile oil

    gors = [fluid['R_i'], 1.2 * fluid['R_i']]
    assert tb.build_task((str(tmp_path), w, dict(gors=gors)))[:3] == (83, 'tables', 'ok')
    table = read_table(tmp_path, 83, 'sol', 2026)
    assert table['status'] == 'ok' and table['n_T'] == 3
    low = table['pvt'].tables[0]
    assert low.T == (280.0, 300.0, 320.0) and low.p[0] == 1.0 and low.p[-1] == 320.0 and len(low.p) > 36
    assert (low.rho_o, low.rho_g) == (fluid['hook']['rho_o'], fluid['hook']['rho_g'])
    # Above its bubble point the stream is all liquid, and below it a part of its mass is gas
    fl = FluidModel(pvt=table['pvt'], gor=low.gor)
    w_g, w_l = (float(v) for v in fl.phase_rates(310.0, 320.0, 1.0, 0.0))
    assert w_g / (w_g + w_l) == pytest.approx(0.0, abs=1e-3)
    w_g, w_l = (float(v) for v in fl.phase_rates(50.0, 320.0, 1.0, 0.0))
    assert 0.2 < w_g / (w_g + w_l) < 0.6
