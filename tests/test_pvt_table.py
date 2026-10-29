"""
Tests of the PVT table (specs/model/pvt/table.md, feature 018): PVTTable and PVTTableSet, FluidModel with a table,
the example tables of tests/data/pvt/, and wells solved with a table on both backends. The vectors of PVT-TAB-2 to
PVT-TAB-5 are in tests/test_spec_vectors.py, for both backends, and tests/test_backend_comparison.py compares their
rows with a table.
"""

import dataclasses
import importlib.util
import logging
import pathlib

import casadi as ca
import numpy as np
import pytest

from manywells.geometry import WellGeometry
from manywells.pvt.black_oil import BlackOil, live_oil_surface_tension, live_oil_viscosity
from manywells.pvt.dead_oil import dead_oil_surface_tension, dead_oil_viscosity
from manywells.pvt.fluid import FluidModel
from manywells.pvt.table import COLUMNS, OPTIONAL_COLUMNS, PVTTable, PVTTableSet
from manywells.sampling.conditions import nominal_conditions
from manywells.sampling.wells import sample_well, well_properties
from manywells.simulator import BoundaryConditions, SSDFSimulator, WellProperties
from manywells.solvers.rust import core_well
from manywells.units import CF_BAR, CF_RS

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / 'tests' / 'data' / 'pvt'


def _make_pvt_tables():
    """specs/tools/make_pvt_tables.py, which writes vazquez_beggs.csv and synthetic_eos_oil_360K.csv."""
    spec = importlib.util.spec_from_file_location('make_pvt_tables', ROOT / 'specs' / 'tools' / 'make_pvt_tables.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAKE = _make_pvt_tables()


def num(x):
    return float(np.asarray(ca.DM(x)).item())


# A small table of a made-up fluid, smooth in p and T, so that the bilinear step is not exact between the nodes
P4 = (10.0, 100.0, 200.0, 300.0)
T3 = (300.0, 350.0, 400.0)


def table_kwargs(p=P4, T=T3, optional=True, gor=100.0, rho_o=850.0, rho_g=0.9):
    P, TT = np.meshgrid(p, T, indexing='ij')
    R_so = 100.0 * np.minimum(P / 200.0, 1.0) ** 0.8 * (1 + (TT - 350.0) / 1000.0)
    columns = dict(R_so=R_so, B_o=1 + 0.0025 * R_so + 2e-4 * (TT - 300.0), Z=1 - 0.3 * P / 300.0 * 350.0 / TT)
    if optional:
        columns.update(mu_o=3e-3 * np.exp(-(TT - 300.0) / 50.0) / (1 + R_so / 100.0), mu_g=1.2e-5 + 3e-8 * P,
                       sigma=0.025 * np.exp(-P / 150.0))
    return dict(p=p, T=T, rho_o=rho_o, rho_g=rho_g, gor=gor, **columns)


def small(**kw):
    return PVTTable(**table_kwargs(**kw))


def tabulate(pvt, p, T, gor):
    """A PVT model's R_so, B_o and Z at the nodes of a grid, as a table."""
    P, TT = np.meshgrid(p, T, indexing='ij')
    columns = {name: np.vectorize(lambda a, b, f=f: num(f(a, b)))(P, TT)
               for name, f in (('R_so', pvt.rs), ('B_o', pvt.bo), ('Z', pvt.z_factor))}
    return PVTTable(p=p, T=T, rho_o=pvt.rho_o, rho_g=pvt.rho_g, gor=gor, **columns)


def bilinear(table, column, p, T):
    """PVT-TAB-2's formula, in numpy."""
    y, P, TT = np.array(getattr(table, column)), np.array(table.p), np.array(table.T)
    p = min(max(p, P[0]), P[-1])
    j = min(int(np.searchsorted(P, p, side='right')) - 1, len(P) - 2)
    a = (p - P[j]) / (P[j + 1] - P[j])
    if len(TT) == 1:
        return (1 - a) * y[j, 0] + a * y[j + 1, 0]
    T = min(max(T, TT[0]), TT[-1])
    k = min(int(np.searchsorted(TT, T, side='right')) - 1, len(TT) - 2)
    b = (T - TT[k]) / (TT[k + 1] - TT[k])
    return (1 - a) * ((1 - b) * y[j, k] + b * y[j, k + 1]) + a * ((1 - b) * y[j + 1, k] + b * y[j + 1, k + 1])


def node(kw, column, value, j=1, k=1):
    a = np.array(kw[column], dtype=float)
    a[j, k] = value
    return {**kw, column: a}


# ---------------------------------------------------------------------------
# PVT-TAB-1: the checks
# ---------------------------------------------------------------------------

class TestChecks:

    @pytest.mark.parametrize('change', [
        pytest.param(lambda kw: {**kw, 'p': kw['p'][:1], 'R_so': kw['R_so'][:1], 'B_o': kw['B_o'][:1],
                                 'Z': kw['Z'][:1], 'mu_o': None, 'mu_g': None, 'sigma': None}, id='one pressure'),
        pytest.param(lambda kw: {**kw, 'p': (10.0, 10.0, 200.0, 300.0)}, id='p not increasing'),
        pytest.param(lambda kw: {**kw, 'p': (-10.0, 100.0, 200.0, 300.0)}, id='p not positive'),
        pytest.param(lambda kw: {**kw, 'T': (300.0, 290.0, 400.0)}, id='T not increasing'),
        pytest.param(lambda kw: node(kw, 'Z', np.nan), id='not finite'),
        pytest.param(lambda kw: node(kw, 'B_o', 0.0), id='B_o not positive'),
        pytest.param(lambda kw: node(kw, 'Z', -0.1), id='Z not positive'),
        pytest.param(lambda kw: node(kw, 'mu_g', 0.0), id='mu_g not positive'),
        pytest.param(lambda kw: node(kw, 'R_so', 1000.0), id='R_so falls as p rises'),
        pytest.param(lambda kw: {**node(kw, 'R_so', -1.0, j=0), 'mu_o': None}, id='R_so below 0 without mu_o'),
        pytest.param(lambda kw: {**node(kw, 'R_so', -1.0, j=0), 'sigma': None}, id='R_so below 0 without sigma'),
        pytest.param(lambda kw: {**kw, 'Z': np.ones((4, 2))}, id='a column of the wrong shape'),
        pytest.param(lambda kw: {**kw, 'Z': None}, id='a column missing'),
        pytest.param(lambda kw: {**kw, 'gor': -5.0}, id='gor negative'),
        pytest.param(lambda kw: {**kw, 'rho_g': 0.0}, id='rho_g not positive'),
    ])
    def test_each_check_raises(self, change):
        with pytest.raises(ValueError):
            PVTTable(**change(table_kwargs()))

    def test_R_so_below_0_with_the_oil_viscosity_and_the_surface_tension(self):
        """A flash's table (PVT-TAB-6): R_so < 0 is accepted where the table carries what the correlations cannot."""
        t = PVTTable(**node(table_kwargs(), 'R_so', -1.0, j=0))
        assert num(t.rs(10.0, 350.0)) == -1.0

    def test_a_column_of_one_temperature_may_be_flat(self):
        t = small(T=(350.0,))
        assert len(t.T) == 1 and np.array(t.R_so).shape == (4, 1)

    @pytest.mark.parametrize('tables', [
        pytest.param(lambda: (small(),), id='one table'),
        pytest.param(lambda: (small(gor=100.0), small(gor=100.0)), id='a GOR twice'),
        pytest.param(lambda: (small(gor=None), small(gor=100.0)), id='a table without a GOR'),
        pytest.param(lambda: (small(gor=80.0), small(gor=120.0, rho_o=860.0)), id='another stock-tank oil'),
        pytest.param(lambda: (small(gor=80.0), BlackOil()), id='not a table'),
    ])
    def test_each_check_of_a_gor_axis_raises(self, tables):
        with pytest.raises(ValueError):
            PVTTableSet(tables=tables())


# ---------------------------------------------------------------------------
# PVT-TAB-2: interpolation
# ---------------------------------------------------------------------------

class TestInterpolation:

    def test_exact_at_the_nodes(self):
        t = small()
        for c in t.columns:
            y = getattr(t, c)
            for j, p in enumerate(t.p):
                for k, T in enumerate(t.T):
                    assert num(t.interpolate(c, p, T)) == y[j][k]

    def test_the_formula_between_the_nodes(self):
        t, rng = small(), np.random.default_rng(18)
        for p, T in zip(rng.uniform(10.0, 300.0, 200), rng.uniform(300.0, 400.0, 200)):
            for c in t.columns:
                assert num(t.interpolate(c, p, T)) == pytest.approx(bilinear(t, c, p, T), rel=1e-13, abs=1e-18)

    @pytest.mark.parametrize('p, T, p_in, T_in', [
        (1.0, 320.0, 10.0, 320.0), (450.0, 320.0, 300.0, 320.0), (150.0, 250.0, 150.0, 300.0),
        (150.0, 500.0, 150.0, 400.0), (-50.0, 1000.0, 10.0, 400.0), (1e4, 1.0, 300.0, 300.0),
    ])
    def test_clamped_outside(self, p, T, p_in, T_in):
        """Off the grid, the value at the nearest point of the grid: finite and positive however far off."""
        t = small()
        for c in t.columns:
            assert num(t.interpolate(c, p, T)) == num(t.interpolate(c, p_in, T_in))
        assert num(t.bo(p, T)) > 0 and num(t.z_factor(p, T)) > 0

    def test_symbolic(self):
        t, p, T = small(), ca.SX.sym('p'), ca.SX.sym('T')
        f = ca.Function('f', [p, T], [ca.vertcat(*(t.interpolate(c, p, T) for c in t.columns))])
        J = ca.Function('J', [p, T], [ca.jacobian(f(p, T), ca.vertcat(p, T))])
        for point in ((55.0, 330.0), (250.0, 390.0), (500.0, 200.0)):
            assert np.array(f(*point)).ravel().tolist() == [num(t.interpolate(c, *point)) for c in t.columns]
            assert np.all(np.isfinite(np.array(J(*point))))

    def test_one_temperature(self):
        """With one temperature: linear in p, and independent of T."""
        t = small(T=(350.0,))
        for p in (1.0, 10.0, 55.0, 150.0, 300.0, 400.0):
            for c in t.columns:
                want = bilinear(t, c, p, 350.0)
                assert num(t.interpolate(c, p, 350.0)) == pytest.approx(want, rel=1e-13, abs=1e-18)
                assert num(t.interpolate(c, p, 250.0)) == num(t.interpolate(c, p, 500.0)) == num(
                    t.interpolate(c, p, 350.0))

    def test_covers(self):
        t, t1 = small(), small(T=(350.0,))
        assert t.covers(10.0, 300.0) and t.covers(300.0, 400.0) and t.covers(150.0, 375.0)
        assert not t.covers(5.0, 350.0) and not t.covers(150.0, 410.0)
        assert t1.covers(150.0, 410.0) and not t1.covers(301.0, 350.0)
        assert t.grid_range == '10.0 to 300.0 bar, 300.0 to 400.0 K'
        assert t1.grid_range == '10.0 to 300.0 bar, 350.0 K'


# ---------------------------------------------------------------------------
# PVT-TAB-3 to PVT-TAB-5, in FluidModel
# ---------------------------------------------------------------------------

POINTS = ((10.0, 300.0), (55.0, 330.0), (150.0, 375.0), (250.0, 390.0), (300.0, 400.0))


class TestProperties:

    def test_the_oil_is_the_tables(self):
        t = small()
        fl = FluidModel(pvt=t, gor=100.0)
        for p, T in POINTS:
            assert num(fl.rs(p, T)) == num(t.interpolate('R_so', p, T))
            assert num(fl.bo(p, T)) == num(t.interpolate('B_o', p, T))
            R_so, B_o = num(fl.rs(p, T)), num(fl.bo(p, T))
            assert num(fl.liquid_density(p, T)) == pytest.approx((fl.rho_o + R_so * fl.rho_g) / B_o, rel=1e-15)

    def test_the_gas_law(self):
        """PVT-TAB-4: the density is explicit in the table's Z, and the row is zero there, also symbolically."""
        fl = FluidModel(pvt=small(), gor=100.0)
        p, T, rho = ca.SX.sym('p'), ca.SX.sym('T'), ca.SX.sym('rho')
        row = ca.Function('row', [p, T, rho], [fl.gas_law_row(p, T, rho)])
        for p, T in POINTS:
            rho_g = num(fl.gas_density(p, T))
            assert rho_g == pytest.approx(CF_BAR * p / (num(fl.z_factor(p, T)) * fl.R_s * T), rel=1e-15)
            assert num(fl.gas_law_row(p, T, rho_g)) == pytest.approx(0.0, abs=1e-12 * p)
            assert num(row(p, T, rho_g)) == pytest.approx(0.0, abs=1e-12 * p)
            assert num(fl.z_factor(p, T)) == num(small().interpolate('Z', p, T))

    def test_the_joule_thomson_factor_is_daks(self):
        """PVT-TAB-4: as for every real gas, DAK's at the gravity of rho_g."""
        fl, bo = FluidModel(pvt=small(), gor=100.0), FluidModel(pvt=BlackOil(rho_o=850.0, rho_g=0.9), gor=100.0)
        for T, rho_g in ((320.0, 50.0), (380.0, 150.0)):
            assert num(fl.jt_factor(T, rho_g)) == num(bo.jt_factor(T, rho_g))

    def test_the_tables_viscosities_and_surface_tension(self):
        t = small()
        for p, T in POINTS:
            assert num(t.oil_viscosity(p, T)) == num(t.interpolate('mu_o', p, T))
            assert num(t.gas_viscosity(p, T, 80.0)) == num(t.interpolate('mu_g', p, T))
            assert num(t.oil_surface_tension(p, T)) == num(t.interpolate('sigma', p, T))

    def test_the_correlations_where_columns_are_missing(self):
        """PVT-TAB-5: Beggs-Robinson, Lee-Gonzalez-Eakin and the live-oil surface tension, at the table's R_so."""
        t = small(optional=False)
        bo = BlackOil(rho_o=t.rho_o, rho_g=t.rho_g)
        for p, T in POINTS:
            R_so = num(t.rs(p, T))
            assert num(t.oil_viscosity(p, T)) == num(live_oil_viscosity(dead_oil_viscosity(t.api, T), R_so / CF_RS))
            assert num(t.gas_viscosity(p, T, 80.0)) == num(bo.gas_viscosity(p, T, 80.0))
            assert num(t.oil_surface_tension(p, T)) == num(
                live_oil_surface_tension(dead_oil_surface_tension(t.rho_o, T), R_so / CF_RS))

    @pytest.mark.parametrize('wlr, w_lg', [(0.0, 0.0), (0.3, 0.7)])
    def test_mass_conservation(self, wlr, w_lg):
        """
        PVT-TAB-3 in PVT-OIL-13: the phase rates add up to the inflow, also where R_so < 0 or above the GOR, up to the
        smoothing of PVT-OIL-13's max, sqrt(eps)/2 = 5e-4 kg/s where all the gas dissolves without lift gas (SMO-1).
        """
        t = PVTTable(**node(table_kwargs(), 'R_so', -1.0, j=0, k=2))
        fl = FluidModel(pvt=t, gor=100.0, wlr=wlr)
        w_res = 12.0
        total = fl.reservoir_gas_rate(w_res) + w_lg + w_res
        assert num(fl.rs(10.0, 400.0)) < 0 < num(fl.rs(300.0, 400.0)) - fl.gor
        for p, T in POINTS:
            w_g, w_l = (num(v) for v in fl.phase_rates(p, T, w_res, w_lg))
            assert w_g + w_l == pytest.approx(total, rel=1e-9, abs=5e-4)

    def test_row_ids_name_the_tables_gas_law(self):
        wp = WellProperties(geometry=WellGeometry.vertical(100, 2, D=0.1), fluid=FluidModel(pvt=small(), gor=100.0))
        assert SSDFSimulator(wp).system.row_ids[3:6] == ('SLIP-1', 'PVT-TAB-4', 'PVT-MIX-6')


# ---------------------------------------------------------------------------
# Files and the example tables
# ---------------------------------------------------------------------------

class TestFiles:

    def test_csv_round_trip(self, tmp_path):
        t = PVTTable(**table_kwargs(), name='small', source='tests/test_pvt_table.py', separation='one stage')
        t.write_csv(tmp_path / 't.csv')
        back = PVTTable.read_csv(tmp_path / 't.csv')
        assert back == t  # every value to the bit
        assert (back.name, back.source, back.separation) == (t.name, t.source, t.separation)
        one = small(T=(350.0,), optional=False, gor=None)
        one.write_csv(tmp_path / 'one.csv')
        assert PVTTable.read_csv(tmp_path / 'one.csv') == one

    def test_csv_round_trip_of_a_gor_axis(self, tmp_path):
        axis = PVTTableSet(tables=(small(gor=120.0), small(gor=80.0, p=(1.0, 50.0, 150.0, 400.0))))
        axis.write_csv(tmp_path / 'axis.csv')
        assert PVTTable.read_csv(tmp_path / 'axis.csv') == axis
        assert axis.gors == (80.0, 120.0)

    @pytest.mark.parametrize('edit, match', [
        (lambda lines: [l.replace(',Z', ',W') for l in lines], 'missing'),
        (lambda lines: [l.replace(',sigma', ',tau') for l in lines], 'unknown'),
        (lambda lines: lines[:-1], 'rectangular'),
        (lambda lines: [l for l in lines if not l.startswith('# rho_o')], 'rho_o and rho_g'),
    ])
    def test_read_csv_refuses(self, tmp_path, edit, match):
        small().write_csv(tmp_path / 't.csv')
        lines = (tmp_path / 't.csv').read_text().splitlines()
        (tmp_path / 'bad.csv').write_text('\n'.join(edit(lines)) + '\n')
        with pytest.raises(ValueError, match=match):
            PVTTable.read_csv(tmp_path / 'bad.csv')

    @pytest.mark.parametrize('name, kind, columns, n_T', [
        ('vazquez_beggs', PVTTable, COLUMNS, 13),
        ('synthetic_eos_oil', PVTTable, COLUMNS + OPTIONAL_COLUMNS, 13),
        ('synthetic_eos_oil_360K', PVTTable, COLUMNS, 1),
        ('synthetic_eos_oil_gor', PVTTableSet, COLUMNS + OPTIONAL_COLUMNS, 7),
    ])
    def test_the_example_tables_load(self, name, kind, columns, n_T):
        t = PVTTable.read_csv(DATA / f'{name}.csv')
        assert type(t) is kind
        for table in (t.tables if kind is PVTTableSet else (t,)):
            assert table.columns == columns and len(table.T) == n_T and table.gor is not None

    def test_the_equation_of_state_tables_have_R_so_below_0(self):
        """The free gas carries components that are oil at standard conditions (PVT-TAB-6)."""
        eos, axis = PVTTable.read_csv(DATA / 'synthetic_eos_oil.csv'), PVTTable.read_csv(DATA / 'synthetic_eos_oil_gor.csv')
        assert np.min(eos.R_so) < 0 and all(np.min(t.R_so) < 0 for t in axis.tables)

    def test_the_tables_of_make_pvt_tables_are_fresh(self):
        """specs/tools/make_pvt_tables.py reproduces vazquez_beggs.csv from develop's black oil, and the 360 K table."""
        assert MAKE.vazquez_beggs() == PVTTable.read_csv(DATA / 'vazquez_beggs.csv')
        assert MAKE.eos_360K() == PVTTable.read_csv(DATA / 'synthetic_eos_oil_360K.csv')

    def test_vazquez_beggs_is_within_the_medium_grids_bounds(self):
        """
        At random points, the table of develop's black oil on a 10 bar by 10 K grid differs from the black oil by
        less than the largest error measured on such grids (specs/features/018-pvt-tables.md, Measurements). The
        black oil is its correlations, as tabulated, without the cap at a fluid's gas-oil ratio (PVT-OIL-15).
        """
        table = FluidModel(pvt=PVTTable.read_csv(DATA / 'vazquez_beggs.csv'), gor=MAKE.VB_GOR)
        vb = MAKE.VB
        black_oil = {'bo': vb.bo, 'gas_density': vb.gas_density,
                     'liquid_density': lambda p, T: (vb.rho_o + vb.rs(p, T) * vb.rho_g) / vb.bo(p, T)}  # PVT-OIL-9
        rng = np.random.default_rng(2026)
        for p, T in zip(rng.uniform(1.0, 400.0, 500), rng.uniform(280.0, 400.0, 500)):
            assert abs(num(table.rs(p, T)) - num(vb.rs(p, T))) < 0.45
            for f, bound in (('bo', 6e-4), ('liquid_density', 3e-4), ('gas_density', 2.1e-3)):
                want = num(black_oil[f](p, T))
                assert abs(num(getattr(table, f)(p, T)) / want - 1) < bound, (f, p, T)


# ---------------------------------------------------------------------------
# PVT-TAB-6: a table from a flash
# ---------------------------------------------------------------------------

class TestFromPhaseSplit:
    """A phase split made from develop's black oil, at its GOR: the flash's X, rho_L and rho_V at the nodes."""

    P = (1.0, 50.0, 100.0, 200.0, 235.688, 300.0)
    T = (300.0, 350.0, 400.0)

    def split(self):
        pvt, gor = MAKE.VB, MAKE.VB_GOR
        P, TT = np.meshgrid(self.P, self.T, indexing='ij')
        at = lambda f: np.vectorize(lambda a, b: num(f(a, b)))(P, TT)
        R_so, B_o, Z, rho_V = np.minimum(at(pvt.rs), gor), at(pvt.bo), at(pvt.z_factor), at(pvt.gas_density)
        m = pvt.rho_o + gor * pvt.rho_g
        X = (gor - R_so) * pvt.rho_g / m
        rho_L = (pvt.rho_o + R_so * pvt.rho_g) / B_o
        t = PVTTable.from_phase_split(self.P, self.T, X, rho_L, rho_V, rho_o=pvt.rho_o, rho_g=pvt.rho_g, gor=gor)
        return t, dict(R_so=R_so, B_o=B_o, Z=Z, X=X, rho_V=rho_V)

    def test_it_gives_back_the_black_oils_columns(self):
        t, want = self.split()
        for c in COLUMNS:
            np.testing.assert_allclose(np.array(getattr(t, c)), want[c], rtol=1e-13, atol=1e-12)

    def test_its_phase_rates_and_gas_density_are_the_flashs(self):
        """At a large rate, so that PVT-OIL-13's smoothing, about 1e-3 kg/s where the stream is all liquid, is small."""
        t, want = self.split()
        fl = FluidModel(pvt=t, gor=t.gor)
        for j, p in enumerate(self.P):
            for k, T in enumerate(self.T):
                w_g, w_l = (num(v) for v in fl.phase_rates(p, T, 1000.0, 0.0))
                assert w_g / (w_g + w_l) == pytest.approx(want['X'][j, k], abs=1e-6)
                assert num(fl.gas_density(p, T)) == pytest.approx(want['rho_V'][j, k], rel=1e-13)


# ---------------------------------------------------------------------------
# PVT-TAB-7: a GOR axis
# ---------------------------------------------------------------------------

class TestGorAxis:

    AXIS = PVTTable.read_csv(DATA / 'synthetic_eos_oil_gor.csv')

    def test_exact_at_the_axiss_gors(self):
        for t in self.AXIS.tables:
            assert self.AXIS.at_gor(t.gor) is t
            fl = FluidModel(pvt=self.AXIS, gor=t.gor)
            assert num(fl.rs(150.0, 350.0)) == num(t.rs(150.0, 350.0))
            assert (fl.rho_o, fl.rho_g) == (self.AXIS.rho_o, self.AXIS.rho_g)

    def test_linear_between_them(self):
        a, b = self.AXIS.tables[1:3]
        w = 0.3
        t = self.AXIS.at_gor((1 - w) * a.gor + w * b.gor)
        assert t.p == tuple(np.union1d(a.p, b.p)) and t.T == tuple(np.union1d(a.T, b.T))
        assert t.columns == a.columns
        for c in t.columns:
            y = np.array(getattr(t, c))
            for j, p in enumerate(t.p):
                for k, T in enumerate(t.T):
                    want = (1 - w) * num(a.interpolate(c, p, T)) + w * num(b.interpolate(c, p, T))
                    assert y[j, k] == pytest.approx(want, rel=1e-12, abs=1e-15)

    @pytest.mark.parametrize('k, step', [(0, -1.0), (-1, 1.0)])
    def test_refused_outside_them(self, k, step):
        with pytest.raises(ValueError, match='outside the PVT table\'s GOR axis'):
            FluidModel(pvt=self.AXIS, gor=self.AXIS.gors[k] + step)

    def test_replace_reads_the_axis_again(self):
        """Calibration replaces the GOR of each row's fluid."""
        fl = FluidModel(pvt=self.AXIS, gor=100.0)
        other = dataclasses.replace(fl, gor=130.0)
        assert num(other.rs(150.0, 350.0)) == num(self.AXIS.at_gor(130.0).rs(150.0, 350.0))
        assert num(other.rs(150.0, 350.0)) != num(fl.rs(150.0, 350.0))

    def test_an_axis_has_no_properties_of_its_own(self):
        with pytest.raises(TypeError, match='at_gor'):
            self.AXIS.rs(150.0, 350.0)

    def test_a_column_one_table_lacks_is_dropped(self):
        axis = PVTTableSet(tables=(small(gor=80.0), small(gor=120.0, optional=False)))
        assert axis.at_gor(100.0).columns == COLUMNS
        assert axis.at_gor(80.0).columns == COLUMNS + OPTIONAL_COLUMNS


# ---------------------------------------------------------------------------
# The simulator: the reservoir on the grid, and roots off it
# ---------------------------------------------------------------------------

def vertical_well(pvt, gor=100.0, L=1000.0, n_cells=5):
    return WellProperties(geometry=WellGeometry.vertical(L, n_cells, D=0.1), fluid=FluidModel(pvt=pvt, gor=gor))


class TestSimulator:

    @pytest.mark.parametrize('backend', ['casadi', 'rust'])
    @pytest.mark.parametrize('p_r, T_r', [(350.0, 360.0), (250.0, 420.0), (250.0, 290.0)])
    def test_the_reservoir_must_lie_on_the_grid(self, p_r, T_r, backend):
        sim = SSDFSimulator(vertical_well(small()), backend=backend)
        with pytest.raises(ValueError, match='does not cover the reservoir'):
            sim.simulate(BoundaryConditions(p_r=p_r, T_r=T_r))

    def test_a_table_of_one_temperature_covers_every_T_r(self):
        fl = FluidModel(pvt=small(T=(350.0,)), gor=100.0)
        assert fl.covers(250.0, 420.0) and fl.covers(250.0, 290.0) and not fl.covers(350.0, 350.0)

    @pytest.mark.parametrize('pvt', [
        pytest.param(lambda: small(), id='small'), pytest.param(lambda: small(T=(350.0,), optional=False), id='one T'),
        pytest.param(lambda: TestGorAxis.AXIS, id='a GOR axis between its GORs'),
    ])
    def test_the_core_reads_the_table_as_casadi_does(self, pvt):
        """
        The core's columns (rust/src/pvt/table.rs) are CasADi's interpolant's to the bit, on the grid and off it, with
        the GOR axis read in Python.
        """
        fl = FluidModel(pvt=pvt(), gor=100.0)
        table, core = fl.pvt.at_gor(fl.gor), core_well(vertical_well(pvt(), gor=100.0))
        rng = np.random.default_rng(18)
        for p, T in zip(rng.uniform(-20.0, 450.0, 300), rng.uniform(250.0, 450.0, 300)):
            want = [num(table.interpolate(c, p, T)) for c in table.columns]
            assert core._component('table_columns', [p, T]) == want

    def test_off_table(self):
        fl, fl1 = FluidModel(pvt=small(), gor=100.0), FluidModel(pvt=small(T=(350.0,)), gor=100.0)
        X = np.tile([150.0, 1.0, 1.0, 0.5, 50.0, 700.0, 350.0], (3, 1))
        assert not fl.off_table(X) and not fl1.off_table(X.ravel())
        for col, value in ((0, 5.0), (0, 301.0), (6, 299.0), (6, 401.0)):
            Y = X.copy()
            Y[1, col] = value
            assert fl.off_table(Y)
            assert fl1.off_table(Y) == (col == 0)
        assert not FluidModel(pvt=BlackOil(), gor=100.0).off_table(X)

    @pytest.mark.slow
    @pytest.mark.parametrize('backend', ['casadi', 'rust'])
    def test_a_root_flags_points_off_the_table(self, caplog, backend):
        """A wellhead below the grid's pressures: the root carries the flag, and simulate warns."""
        pvt, gor = MAKE.VB, MAKE.VB_GOR
        T = np.arange(270.0, 400.0 + 1e-9, 10.0)
        bc = BoundaryConditions(p_r=200.0, p_s=20.0, T_r=370.0, u=0.8)
        for p_min, off in ((1.0, False), (60.0, True)):
            p = np.unique(np.r_[p_min, np.arange(60.0, 400.0 + 1e-9, 10.0)])
            sim = SSDFSimulator(vertical_well(tabulate(pvt, p, T, gor), gor=gor, L=2000.0, n_cells=10),
                                backend=backend)
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger='manywells'):
                op = sim.simulate(bc)
            assert op.off_table is off and op.state[-1, 0] < 60.0
            assert any('off the fluid\'s PVT table' in r.message for r in caplog.records) is off
            assert all(r.off_table is off for r in sim.root_set(bc).roots if r.state[-1, 0] < 60.0)


# ---------------------------------------------------------------------------
# Solved wells
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('backend', ['casadi', 'rust'])
@pytest.mark.parametrize('well', [0, 5, 8])
def test_a_table_of_the_wells_black_oil_gives_its_operating_point(well, backend):
    """
    A sampled well (seed 2026, develop's configuration), with its own black oil, capped at its GOR (PVT-OIL-15),
    tabulated on a 10 bar by 10 K grid over its pressures and temperatures: the operating point within 0.05 bar at the
    bottomhole and the wellhead (specs/features/018-pvt-tables.md, acceptance 6), and on the grid. Wells 0 (deviated),
    5 and 8 flow at their nominal point with develop's five flow regimes (019); well 11, which did with the fitted
    three, does not.
    """
    draw = sample_well(2026, well)
    wp, bc = well_properties(draw, configuration='develop', n_cells=30), nominal_conditions(draw)
    p = np.unique(np.r_[1.0, np.arange(10.0, np.ceil(bc.p_r / 10) * 10 + 1e-9, 10.0)])
    T = np.arange(270.0, np.ceil(bc.T_r / 10) * 10 + 1e-9, 10.0)
    table = tabulate(wp.fluid, p, T, wp.fluid.gor)
    base = SSDFSimulator(wp, backend=backend).simulate(bc)
    op = SSDFSimulator(dataclasses.replace(wp, fluid=dataclasses.replace(wp.fluid, pvt=table)),
                       backend=backend).simulate(bc)
    assert abs(op.state[0, 0] - base.state[0, 0]) < 0.05
    assert abs(op.state[-1, 0] - base.state[-1, 0]) < 0.05
    assert not op.off_table


@pytest.mark.slow
def test_the_equation_of_state_table_solves():
    """
    synthetic_eos_oil.csv, with its negative R_so and its own viscosities and surface tension, in a vertical well:
    the same operating point on both backends.
    """
    pvt = PVTTable.read_csv(DATA / 'synthetic_eos_oil.csv')
    bc = BoundaryConditions(p_r=250.0, p_s=20.0, T_r=360.0, u=0.8)
    ops = [SSDFSimulator(vertical_well(pvt, gor=pvt.gor, L=2000.0, n_cells=20), backend=b).simulate(bc)
           for b in ('casadi', 'rust')]
    for op in ops:
        assert op.label == 'stable' and not op.off_table
        assert 20.0 < op.state[-1, 0] < op.state[0, 0] < 250.0
    np.testing.assert_allclose(ops[1].state, ops[0].state, rtol=1e-8)
