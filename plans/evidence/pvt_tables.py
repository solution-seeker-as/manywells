"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 28 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

PVT tables (feature spec specs/features/018-pvt-tables.md). A prototype: the table enters by a subclass of FluidModel,
so it runs on the CasADi backend only. Every table here is made from develop's own black oil (Vazquez-Beggs R_so and
B_o, the DAK z-factor) at its nodes, so that a table's results differ from the black oil's by its interpolation alone.

A table holds R_so, B_o and Z on a (p, T) grid, Z being the gas law's with the gas's R_s at standard conditions. It is
interpolated bilinearly, with p and T clamped to the grid (the proposed PVT-TAB-2), by CasADi's 'linear' interpolant;
the gas law row is p - rho_g Z R_s T / c_bar with the table's Z, and the gas density c_bar p / (Z R_s T). Everything
else is FluidModel's: the live-oil density, the phase rates, the viscosities and the surface tension, from the table's
R_so where they need it, and DAK's Joule-Thomson factor.

The grids (p from 1 bar to 470 bar, T from 270 K to 430 K; the sampler draws p_r up to 457 bar and temperatures from
277 K to 423 K):

  fine          2 bar, 5 K: an equation-of-state table as a PVT program writes it
  medium        10 bar, 10 K
  coarse        25 bar, 25 K: the spacing of a laboratory report's differential liberation
  coarse_pchip  the coarse table refined to 2 bar, 5 K by monotone cubic interpolation (Fritsch-Carlson) in p, then
                in T, before it is interpolated bilinearly
  isothermal    5 bar, at the well's reservoir temperature only, so that the properties do not depend on T

Parts:

  points  the error of each grid's R_so, B_o, live-oil density and gas density against the black oil's, at 2000
          random (p, T) of each of 40 sampled fluids, and of a bubble-point fluid with and without a node at p_b; and
          the error of interpolating the gas density instead of Z
  wells   88 wells of develop's sampler (seed 2026), each at its nominal operating point and two sampled ones (the
          cases of plans/evidence/jt_wells.py), with develop's default and with a table on each grid: PBH, PWH and TWH,
          the operating points found, and the time to build the well and find its roots
  interp  the cases of the first 12 wells on the medium grid, interpolated bilinearly or by CasADi's cubic B-spline
          (C2, with p and T clamped, since it is 0 off its grid): the operating points, Ipopt's iterations and the time
  write   the example tables vazquez_beggs.csv (develop's black oil) and synthetic_eos_oil_360K.csv (from
          synthetic_eos_oil.csv, which plans/evidence/pvt_table_eos.py writes) into a folder
  eos     the cases of the first 88 wells whose p_r and T_r the table covers, with the fluid replaced by
          synthetic_eos_oil.csv's, and by develop's black oil at the same stock-tank densities and GOR, uncapped and
          capped at the table's bubble point at T_r
  scan    each case of eos's CSV where the capped black oil has an operating point and the table has none: SOL-3's
          shooting residual (the CHK-1 row after a march) at 80 values of p_0, to tell a well that cannot flow from
          a root the search missed; and the table with 0.01 kg/s of lift gas, so that the bottomhole is never all liquid

From the repository root, one BLAS thread per process:

    uv run python plans/evidence/pvt_tables.py points
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python plans/evidence/pvt_tables.py wells 88 pvt_tables_wells.csv
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python plans/evidence/pvt_tables.py interp 12 pvt_tables_interp.csv
    uv run python plans/evidence/pvt_tables.py write tests/data/pvt
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python plans/evidence/pvt_tables.py eos 88 \\
        tests/data/pvt/synthetic_eos_oil.csv pvt_tables_eos.csv
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python plans/evidence/pvt_tables.py scan \\
        tests/data/pvt/synthetic_eos_oil.csv pvt_tables_eos.csv
"""

import sys
import time
from dataclasses import dataclass, replace
from multiprocessing import Pool

import casadi as ca
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from manywells.configurations import DEVELOP
from manywells.pvt import gas_density_from_sg, mixture_viscosity
from manywells.pvt.fluid import FluidModel
from manywells.pvt.water import water_viscosity
from manywells.sampling.conditions import nominal_conditions, sample_conditions
from manywells.sampling.wells import rng_for, sample_well, well_properties
from manywells.simulator import SimError, SSDFSimulator
from manywells.units import CF_BAR

SEED = 2026
P_MIN, P_MAX, T_MIN, T_MAX = 1.0, 470.0, 270.0, 430.0
GRIDS = {'fine': (2.0, 5.0), 'medium': (10.0, 10.0), 'coarse': (25.0, 25.0), 'coarse_pchip': (25.0, 25.0),
         'isothermal': (5.0, None)}
REFINED = (2.0, 5.0)    # The grid coarse_pchip is refined to


def axis(lo, hi, step, extra=()):
    """Nodes from lo by step, the last at or above hi, with the extra nodes added."""
    n = int(np.ceil((hi - lo) / step - 1e-9))
    return np.unique(np.concatenate([lo + step * np.arange(n + 1), np.asarray(extra, dtype=float)]))


class Table:
    """Columns on a (p, T) grid, or on p alone, interpolated bilinearly with p and T clamped to the grid."""

    def __init__(self, p, T, cols, method='linear'):
        self.p, self.T, self.cols = np.asarray(p, float), np.asarray(T, float), cols
        grid = [self.p] if len(self.T) == 1 else [self.p, self.T]
        self._f = {k: ca.interpolant(k, method, grid, np.asarray(v, float).ravel(order='F')) for k, v in cols.items()}

    def __call__(self, key, p, T):
        p = ca.fmin(ca.fmax(p, self.p[0]), self.p[-1])
        if len(self.T) == 1:
            return self._f[key](p)
        T = ca.fmin(ca.fmax(T, self.T[0]), self.T[-1])
        return self._f[key](ca.vertcat(p, T))


def black_oil_columns(fluid, p, T):
    """R_so, B_o and Z of a fluid at every node of the grid p x T, each of shape (len(p), len(T))."""
    ps, Ts = ca.SX.sym('p'), ca.SX.sym('T')
    f = ca.Function('pvt', [ps, Ts], [fluid.rs(ps, Ts), fluid.bo(ps, Ts), fluid.z_factor(ps, Ts)])
    P, TT = np.meshgrid(p, T, indexing='ij')
    out = f.map(P.size)(P.ravel(), TT.ravel())
    return {k: np.asarray(v, float).reshape(P.shape) for k, v in zip(('R_so', 'B_o', 'Z'), out)}


def refine(p, T, cols, dp, dT):
    """The columns refined to a dp x dT grid by monotone cubic (Fritsch-Carlson) interpolation in p, then in T."""
    p2, T2 = axis(p[0], p[-1], dp, p), axis(T[0], T[-1], dT, T)
    out = {}
    for k, v in cols.items():
        v = PchipInterpolator(p, v, axis=0)(p2)
        out[k] = PchipInterpolator(T, v, axis=1)(T2) if len(T) > 1 else v
    return p2, T2, out


def make_table(fluid, grid, T_r=None, p_extra=()):
    """A table of the fluid's black oil on one of GRIDS (the isothermal one at T_r)."""
    dp, dT = GRIDS[grid]
    p = axis(P_MIN, P_MAX, dp, p_extra)
    T = np.array([T_r]) if dT is None else axis(T_MIN, T_MAX, dT)
    cols = black_oil_columns(fluid, p, T)
    if grid == 'coarse_pchip':
        p, T, cols = refine(p, T, cols, *REFINED)
    return Table(p, T, cols)


@dataclass(frozen=True, eq=False)
class TableFluid(FluidModel):
    """
    develop's FluidModel with R_so, B_o and Z from a table (black oil, so that gas dissolves), and the oil and gas
    viscosities and the surface tension from it where it has them.
    """
    table: Table = None

    def rs(self, p, T):
        return self.table('R_so', p, T)

    def bo(self, p, T):
        return self.table('B_o', p, T)

    def z_factor(self, p, T):
        return self.table('Z', p, T)

    def gas_density(self, p, T):
        return CF_BAR * p / (self.z_factor(p, T) * self.R_s * T)

    def gas_law_row(self, p, T, rho_g):
        return p - rho_g * self.z_factor(p, T) * self.R_s * T / CF_BAR

    def liquid_viscosity(self, p, T):  # PVT-MIX-8 with the table's mu_o
        if 'mu_o' not in self.table.cols:
            return super().liquid_viscosity(p, T)
        # The in-situ water fraction since issue #8 (ae754d8); the standard-conditions one before it (8b4f1e6)
        lambda_w = self.wlr / self.liquid_fvf(p, T) if hasattr(self, 'liquid_fvf') else self.wlr
        return lambda_w * water_viscosity(T) + (1 - lambda_w) * self.table('mu_o', p, T)

    def gas_viscosity(self, T, rho_g):  # The table's mu_g is at (p, T): p from the gas law, at the state's rho_g
        if 'mu_g' not in self.table.cols:
            return super().gas_viscosity(T, rho_g)
        raise NotImplementedError('mu_g(p, T) needs p; see mixture_viscosity')

    def mixture_viscosity(self, p, T, alpha, rho_g, rho_l):
        if 'mu_g' not in self.table.cols:
            return super().mixture_viscosity(p, T, alpha, rho_g, rho_l)
        return mixture_viscosity(self.liquid_viscosity(p, T), self.table('mu_g', p, T), alpha, rho_l=rho_l,
                                 rho_g=rho_g)

    def surface_tension(self, p, T, rho_l):  # PVT-MIX-7 with the table's gas-oil surface tension
        if 'sigma' not in self.table.cols:
            return super().surface_tension(p, T, rho_l)
        return self.table('sigma', p, T)


def table_fluid(fluid, grid, T_r=None, p_extra=()):
    return TableFluid(**{f: getattr(fluid, f) for f in fluid.__dataclass_fields__},
                      table=make_table(fluid, grid, T_r, p_extra))


# ---------------------------------------------------------------------------------------------------------------------
# Part 1: the tables' error at random points
# ---------------------------------------------------------------------------------------------------------------------

def sampled_fluids(n):
    fluids = []
    for well in range(4 * n):
        draw = sample_well(SEED, well)
        try:
            fluids.append((well, well_properties(draw, draw.fractions, DEVELOP).fluid, draw.T_r))
        except ValueError:
            continue
        if len(fluids) == n:
            break
    return fluids


def evaluate(fluid, P, TT):
    ps, Ts = ca.SX.sym('p'), ca.SX.sym('T')
    f = ca.Function('f', [ps, Ts], [fluid.rs(ps, Ts), fluid.bo(ps, Ts), fluid.liquid_density(ps, Ts),
                                    fluid.gas_density(ps, Ts)])
    return [np.asarray(v, float).ravel() for v in f.map(P.size)(P, TT)]


def point_errors(fluid, T_r, rng, n=2000, p_extra=()):
    P, TT = rng.uniform(5.0, 460.0, n), rng.uniform(277.0, 423.0, n)
    ref = evaluate(fluid, P, TT)
    rows = {}
    for grid in GRIDS:
        got = evaluate(table_fluid(fluid, grid, T_r, p_extra), P, TT)
        rows[grid] = {'R_so (Sm3/Sm3)': np.max(np.abs(got[0] - ref[0])),
                      'B_o (%)': 100 * np.max(np.abs(got[1] / ref[1] - 1)),
                      'rho_l (%)': 100 * np.max(np.abs(got[2] / ref[2] - 1)),
                      'rho_g (%)': 100 * np.max(np.abs(got[3] / ref[3] - 1))}
    # The gas density interpolated directly, on the medium grid, instead of Z
    p, T = axis(P_MIN, P_MAX, 10.0), axis(T_MIN, T_MAX, 10.0)
    Pg, Tg = np.meshgrid(p, T, indexing='ij')
    rho = evaluate(fluid, Pg.ravel(), Tg.ravel())[3].reshape(Pg.shape)
    tab = Table(p, T, {'rho_g': rho})
    x = ca.SX.sym('x', 2)
    got = np.asarray(ca.Function('g', [x], [tab('rho_g', x[0], x[1])]).map(n)(np.vstack([P, TT])), float).ravel()
    rows['medium, rho_g interpolated'] = {'rho_g (%)': 100 * np.max(np.abs(got / ref[3] - 1))}
    return pd.DataFrame(rows).T


def points():
    rng = np.random.default_rng(SEED)
    frames = [point_errors(f, T_r, rng) for _, f, T_r in sampled_fluids(40)]
    pd.set_option('display.width', 200)
    print('Largest error over 40 sampled fluids (develop, seed 2026), 2000 random points each, 5-460 bar, 277-423 K:')
    print(pd.concat(frames).groupby(level=0, sort=False).max().round(3).to_string())
    print('\nMedian over the fluids of each fluid\'s largest error:')
    print(pd.concat(frames).groupby(level=0, sort=False).median().round(3).to_string())
    bp = FluidModel(p_bubble=150.0)
    print('\nA bubble-point fluid (FluidModel(p_bubble=150), API 35), without and with a node at p_b:')
    print(point_errors(bp, 350.0, np.random.default_rng(1)).round(3).to_string())
    print(point_errors(bp, 350.0, np.random.default_rng(1), p_extra=(150.0,)).round(3).to_string())


# ---------------------------------------------------------------------------------------------------------------------
# Part 2: solved wells
# ---------------------------------------------------------------------------------------------------------------------

def outputs(root):
    return dict(PBH=root.state[0, 0], PWH=root.state[-1, 0], TWH=root.state[-1, 6], p_min=root.state[:, 0].min(),
                T_min=root.state[:, 6].min())


def solve(wp, bc, x_guess=None):
    t0 = time.perf_counter()
    try:
        sim = SSDFSimulator(wp)
        t1 = time.perf_counter()
        op = sim.simulate(bc, x_guess=x_guess)
        return op, dict(outputs(op), build=t1 - t0, search=time.perf_counter() - t1)
    except SimError as e:
        return None, dict(err=type(e).__name__, build=np.nan, search=time.perf_counter() - t0)


def well_cases(well):
    rows = []
    draw = sample_well(SEED, well)
    rng = rng_for(SEED, well, 'jt')
    for k in range(3):
        bc, fractions = (nominal_conditions(draw), draw.fractions) if k == 0 else sample_conditions(draw, rng)
        try:
            wp = well_properties(draw, fractions, DEVELOP)
        except ValueError:  # no oil to give a gas-oil ratio (SMP-43)
            continue
        rec = dict(well=well, k=k, f_g=fractions[0], gor=wp.fluid.gor, api=wp.fluid.api, sg_gas=wp.fluid.sg_gas,
                   w_lg=bc.w_lg, p_r=bc.p_r, p_s=bc.p_s, T_r=bc.T_r, T_s=bc.T_s, u=bc.u)
        base, out = solve(wp, bc)
        rec.update({f'{key}_base': v for key, v in out.items()})
        for grid in GRIDS:
            t0 = time.perf_counter()
            fluid = table_fluid(wp.fluid, grid, bc.T_r)
            rec[f'table_{grid}'] = time.perf_counter() - t0
            # Cold, as the base: the table is judged on the search it gets, not on the base's root as a start
            _, out = solve(replace(wp, fluid=fluid), bc)
            rec.update({f'{key}_{grid}': v for key, v in out.items()})
        rows.append(rec)
    return rows


def summary(df):
    q = [0, 0.1, 0.5, 0.9, 1]
    base = df[df.PBH_base.notna()]
    print(f'{len(df)} cases, {len(base)} with a base operating point')
    for g in GRIDS:
        both = base[f'PBH_{g}'].notna()
        print(f'  {g}: operating point in {both.sum()} of them, and in {df[df.PBH_base.isna()][f"PBH_{g}"].notna().sum()}'
              f' of the {df.PBH_base.isna().sum()} without one')
    for g in GRIDS:
        ok = base[base[f'PBH_{g}'].notna()]
        d = pd.DataFrame({f'{x} (bar)' if x != 'TWH' else 'TWH (K)': ok[f'{x}_{g}'] - ok[f'{x}_base']
                          for x in ('PBH', 'PWH', 'TWH')})
        print(f'\n{g}: quantiles of the change against the black oil, over {len(ok)} cases')
        print(d.quantile(q).round(4).to_string())
        print('largest |change|: ' + ', '.join(f'{c} {d[c].abs().max():.4f}' for c in d))
    t = pd.DataFrame({g: base[f'build_{g}'] + base[f'search_{g}'] for g in ('base',) + tuple(GRIDS)})
    print('\nTime to build the well and find its roots, one process (s): median / total')
    print(pd.DataFrame({'median': t.median(), 'total': t.sum()}).round(2).to_string())
    print('Time to make a table (s), median:', {g: round(base[f'table_{g}'].median(), 3) for g in GRIDS})


def wells(n_wells, out):
    with Pool(14) as pool:
        rows = [r for rs in pool.map(well_cases, range(n_wells), chunksize=1) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    pd.set_option('display.width', 200)
    summary(df)


def counted_solve(wp, bc):
    """The operating point's PBH (NaN if none), the time, and Ipopt's iterations over the search's starts."""
    from manywells.solvers import ipopt
    iterations, solve_ = [], ipopt.IpoptSolver.solve

    def counting(self, *args, **kwargs):
        result = solve_(self, *args, **kwargs)
        iterations.append(result.stats['iter_count'])
        return result

    ipopt.IpoptSolver.solve = counting
    try:
        op, out = solve(wp, bc)
    finally:
        ipopt.IpoptSolver.solve = solve_
    return out.get('PBH', np.nan), out['build'] + out['search'] if op is not None else out['search'], sum(iterations)


def interp_cases(well):
    rows = []
    draw = sample_well(SEED, well)
    rng = rng_for(SEED, well, 'jt')
    for k in range(3):
        bc, fractions = (nominal_conditions(draw), draw.fractions) if k == 0 else sample_conditions(draw, rng)
        try:
            wp = well_properties(draw, fractions, DEVELOP)
        except ValueError:
            continue
        p, T = axis(P_MIN, P_MAX, 10.0), axis(T_MIN, T_MAX, 10.0)
        cols = black_oil_columns(wp.fluid, p, T)
        fields = {f: getattr(wp.fluid, f) for f in wp.fluid.__dataclass_fields__}
        rec = dict(well=well, k=k)
        for name, fluid in (('base', wp.fluid), ('linear', TableFluid(**fields, table=Table(p, T, cols))),
                            ('bspline', TableFluid(**fields, table=Table(p, T, cols, 'bspline')))):
            rec[f'PBH_{name}'], rec[f'seconds_{name}'], rec[f'iterations_{name}'] = counted_solve(
                replace(wp, fluid=fluid), bc)
        rows.append(rec)
    return rows


def interp(n_wells, out):
    with Pool(4) as pool:
        df = pd.DataFrame([r for rs in pool.map(interp_cases, range(n_wells), chunksize=1) for r in rs])
    df.to_csv(out, index=False)
    names = ('base', 'linear', 'bspline')
    print(f'{len(df)} cases; operating points: ' + ', '.join(f'{n} {df[f"PBH_{n}"].notna().sum()}' for n in names))
    print('Ipopt iterations, total: ' + ', '.join(f'{n} {df[f"iterations_{n}"].sum()}' for n in names))
    print('Time (s), median / total: ' + ', '.join(f'{n} {df[f"seconds_{n}"].median():.2f} / '
                                                  f'{df[f"seconds_{n}"].sum():.1f}' for n in names))
    ok = df[df.PBH_base.notna()]
    print('Largest |PBH - base| (bar): ' + ', '.join(f'{n} {(ok[f"PBH_{n}"] - ok.PBH_base).abs().max():.5f}'
                                                    for n in names[1:]))


# ---------------------------------------------------------------------------------------------------------------------
# Part 3: the example tables, and the equation-of-state table on sampled wells
# ---------------------------------------------------------------------------------------------------------------------

def read_table(path, columns=None):
    """A table in the proposed CSV format: '# key: value' lines, then one row per node. (Table, metadata)."""
    meta = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            key, _, value = line[1:].partition(':')
            meta[key.strip()] = value.strip()
    df = pd.read_csv(path, comment='#')
    p, T = np.unique(df.p), np.unique(df['T'])
    assert len(df) == len(p) * len(T), 'not a rectangular grid'
    df = df.sort_values(['T', 'p'])
    cols = {c: df[c].to_numpy().reshape(len(T), len(p)).T for c in df.columns if c not in ('p', 'T')
            and (columns is None or c in columns)}
    return Table(p, T, cols), meta


def bubble_points(meta):
    """{T (K): p_b (bar)} from a table's 'p_bubble: 280=109.270, ...  (T in K = p in bar)' line."""
    pairs = meta['p_bubble'].split('(')[0].split(',')
    return {float(t): float(p) for t, p in (kv.split('=') for kv in pairs)}


def write_table(path, table, header):
    P, TT = np.meshgrid(table.p, table.T, indexing='xy')
    df = pd.DataFrame({'p': P.ravel(), 'T': TT.ravel(), **{k: v.T.ravel() for k, v in table.cols.items()}})
    with open(path, 'w') as fh:
        fh.write('\n'.join('# ' + h for h in header) + '\n')
        df.to_csv(fh, index=False, float_format='%.10g')


VB_FLUID = dict(rho_o=850.0, rho_g=gas_density_from_sg(0.75), gor=120.0)
VB_T_RES = 360.0


def vb_example_fluid():
    """develop's black oil, API 35 and gas gravity 0.75, capped at the bubble point where R_so is the GOR at 360 K."""
    from scipy.optimize import brentq
    f = FluidModel(**VB_FLUID)
    p_b = brentq(lambda p: float(f.rs(p, VB_T_RES)) - f.gor, 10.0, 600.0, xtol=1e-10)
    return FluidModel(**VB_FLUID, p_bubble=round(p_b, 3))


def write(folder):
    f = vb_example_fluid()
    p, T = axis(P_MIN, 400.0, 10.0, (f.p_bubble,)), axis(280.0, 400.0, 10.0)
    p = p[p <= 400.0]
    write_table(f'{folder}/vazquez_beggs.csv', Table(p, T, black_oil_columns(f, p, T)), [
        'ManyWells PVT table (specs/features/018-pvt-tables.md, draft)',
        'name: vazquez_beggs',
        'source: plans/evidence/pvt_tables.py write; develop\'s black oil at 8b4f1e6: FluidModel(rho_o=850, '
        f'rho_g=gas_density_from_sg(0.75), gor=120, p_bubble={f.p_bubble}), R_so capped at the bubble point '
        f'(PVT-OIL-7), where R_so is the GOR at {VB_T_RES:.0f} K; Z from the DAK equation of state (PVT-GAS-11)',
        'separation: standard conditions (FluidModel\'s p_sep and T_sep)',
        f'rho_o: {f.rho_o}', f'rho_g: {f.rho_g:.10g}', f'gor: {f.gor}',
        'columns: p (bar), T (K), R_so (Sm3/Sm3), B_o (-), Z (-, with R_s at standard conditions)'])
    eos, meta = read_table(f'{folder}/synthetic_eos_oil.csv', ('R_so', 'B_o', 'Z'))
    k = int(np.argmin(np.abs(eos.T - 360.0)))
    p_b = bubble_points(meta)[360.0]
    lab = np.array([10, 30, 50, 70, 90, 110, 130, 150, 170, p_b, 200, 250, 300, 350, 400], float)
    idx = [int(np.argmin(np.abs(eos.p - x))) for x in lab]
    assert np.allclose(eos.p[idx], lab)
    write_table(f'{folder}/synthetic_eos_oil_360K.csv',
                Table(eos.p[idx], eos.T[k:k + 1], {c: v[idx, k:k + 1] for c, v in eos.cols.items()}), [
        'ManyWells PVT table (specs/features/018-pvt-tables.md, draft)',
        'name: synthetic_eos_oil_360K',
        'source: plans/evidence/pvt_tables.py write; synthetic_eos_oil.csv at 360 K only, on 15 of its pressures '
        '(a laboratory report\'s spacing), without the viscosities and the surface tension',
        *(f'{key}: {meta[key]}' for key in ('separation', 'rho_o', 'rho_g', 'gor')),
        'columns: p (bar), T (K), R_so (Sm3/Sm3), B_o (-), Z (-, with R_s at standard conditions)'])


def eos_cases(args):
    well, path = args
    eos, meta = read_table(path)
    p_b = bubble_points(meta)
    std = dict(rho_o=float(meta['rho_o']), rho_g=float(meta['rho_g']), gor=float(meta['gor']))
    rows = []
    draw = sample_well(SEED, well)
    rng = rng_for(SEED, well, 'jt')
    for k in range(3):
        bc, fractions = (nominal_conditions(draw), draw.fractions) if k == 0 else sample_conditions(draw, rng)
        try:
            wp = well_properties(draw, fractions, DEVELOP)
        except ValueError:
            continue
        if not (bc.p_r <= eos.p[-1] and bc.T_r <= eos.T[-1]):   # The bottom of the well on the grid; the rest is
            continue                                            # checked on the operating point (summary)
        vb = replace(wp.fluid, **std)
        T_b = np.array(sorted(p_b))
        cap = float(np.interp(bc.T_r, T_b, [p_b[t] for t in T_b]))
        variants = {'vb': vb, 'vb_cap': replace(vb, p_bubble=cap),
                    'eos': TableFluid(**{f: getattr(vb, f) for f in vb.__dataclass_fields__}, table=eos),
                    'eos_corr': TableFluid(**{f: getattr(vb, f) for f in vb.__dataclass_fields__},
                                           table=read_table(path, ('R_so', 'B_o', 'Z'))[0])}
        rec = dict(well=well, k=k, wlr=vb.wlr, p_r=bc.p_r, p_s=bc.p_s, T_r=bc.T_r, u=bc.u, w_lg=bc.w_lg, p_b=cap)
        for name, fluid in variants.items():
            _, out = solve(replace(wp, fluid=fluid), bc)
            rec.update({f'{key}_{name}': v for key, v in out.items()})
        rows.append(rec)
    return rows


def eos_summary(df):
    names = ('vb', 'vb_cap', 'eos', 'eos_corr')
    print(f'{len(df)} cases; operating points: ' + ', '.join(f'{n} {df[f"PBH_{n}"].notna().sum()}' for n in names))
    out = (df.p_min_eos < 1.0) | (df.T_min_eos < 280.0)
    print(f'eos operating points with a point off the grid (p < 1 bar or T < 280 K): {out.sum()}; '
          f'lowest T {df.T_min_eos.min():.1f} K, lowest p {df.p_min_eos.min():.1f} bar')
    for a, b in (('eos', 'vb'), ('eos', 'vb_cap'), ('eos', 'eos_corr')):
        ok = df.dropna(subset=[f'PBH_{a}', f'PBH_{b}'])
        d = pd.DataFrame({f'{x}': ok[f'{x}_{a}'] - ok[f'{x}_{b}'] for x in ('PBH', 'PWH', 'TWH')})
        print(f'\n{a} - {b}, {len(ok)} cases (bar, K):')
        print(d.quantile([0, 0.1, 0.5, 0.9, 1]).round(3).to_string())


def eos_wells(n_wells, path, out):
    with Pool(14) as pool:
        rows = [r for rs in pool.map(eos_cases, [(w, path) for w in range(n_wells)], chunksize=1) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    pd.set_option('display.width', 200)
    eos_summary(df)


def choke_scan(wp, bc, n=80):
    """
    SOL-3's shooting residual on a scan of p_0 over (p_s, p_r): the CHK-1 row after a march from each p_0, which
    zeroes every other row. A sign change brackets a root; a march that fails gives NaN.
    """
    from manywells.solvers.march import MarchFailed
    sim = SSDFSimulator(wp)
    system, march = sim._roots.system, sim._roots.march
    params, k = system.params(bc), system.row_ids.index('CHK-1')
    out = []
    for p_0 in np.linspace(bc.p_s, bc.p_r, n + 2)[1:-1]:
        try:
            out.append(float(np.asarray(system.residual(march(params, p_0), params)).ravel()[k]))
        except MarchFailed:
            out.append(np.nan)
    return np.array(out)


def scan_case(args):
    well, k, path = args
    eos, meta = read_table(path)
    std = dict(rho_o=float(meta['rho_o']), rho_g=float(meta['rho_g']), gor=float(meta['gor']))
    draw = sample_well(SEED, well)
    rng = rng_for(SEED, well, 'jt')
    for j in range(k + 1):
        bc, fractions = (nominal_conditions(draw), draw.fractions) if j == 0 else sample_conditions(draw, rng)
    wp = well_properties(draw, fractions, DEVELOP)
    vb = replace(wp.fluid, **std)
    rec = dict(well=well, k=k)
    T_b = sorted(bubble_points(meta))
    cap = float(np.interp(bc.T_r, T_b, [bubble_points(meta)[t] for t in T_b]))
    table = TableFluid(**{f: getattr(vb, f) for f in vb.__dataclass_fields__}, table=eos)
    for name, fluid in (('eos', table), ('vb_cap', replace(vb, p_bubble=cap))):
        r = choke_scan(replace(wp, fluid=fluid), bc)
        ok = r[np.isfinite(r)]
        rec.update({f'finite_{name}': len(ok), f'sign_changes_{name}': int(np.sum(np.diff(np.sign(ok)) != 0)),
                    f'min_{name}': ok.min() if len(ok) else np.nan, f'max_{name}': ok.max() if len(ok) else np.nan})
    # With a trace of lift gas, so that the bottomhole is never all liquid
    op, out = solve(replace(wp, fluid=table), replace(bc, w_lg=LIFT_GAS_TRACE))
    rec.update(p_b=cap, PBH_eos_trace=out.get('PBH', np.nan), PWH_eos_trace=out.get('PWH', np.nan))
    return rec


LIFT_GAS_TRACE = 0.01       # kg/s


def scan(path, cases_csv):
    """The scan of every case where the capped black oil has an operating point and the table has none."""
    df = pd.read_csv(cases_csv)
    lost = df[df.PBH_vb_cap.notna() & df.PBH_eos.isna()]
    with Pool(14) as pool:
        rows = pd.DataFrame(pool.map(scan_case, [(int(w), int(k), path) for w, k in zip(lost.well, lost.k)],
                                     chunksize=1))
    rows = rows.merge(lost[['well', 'k', 'PBH_vb_cap']], on=['well', 'k'])
    pd.set_option('display.width', 220)
    print(f'{len(rows)} cases where the black oil capped at p_b has an operating point and the table none; the CHK-1 '
          'row after a march from 80 values of p_0 in (p_s, p_r), with the table and with that black oil; and the '
          f'table with {LIFT_GAS_TRACE} kg/s of lift gas:')
    print(rows.round(3).to_string(index=False))
    print(f'The capped black oil\'s PBH above the table\'s bubble point at T_r in {(rows.PBH_vb_cap > rows.p_b).sum()}; '
          f'the table with the trace of lift gas has an operating point in {rows.PBH_eos_trace.notna().sum()}')


if __name__ == '__main__':
    if sys.argv[1] == 'points':
        points()
    elif sys.argv[1] == 'wells':
        wells(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == 'summary':
        pd.set_option('display.width', 200)
        summary(pd.read_csv(sys.argv[2]))
    elif sys.argv[1] == 'interp':
        interp(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == 'write':
        write(sys.argv[2])
    elif sys.argv[1] == 'eos':
        eos_wells(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'scan':
        scan(sys.argv[2], sys.argv[3])
