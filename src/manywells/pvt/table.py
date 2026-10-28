"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

A fluid's hydrocarbon PVT from a table (specs/model/pvt/table.md, feature 018): PVTTable, the columns R_so, B_o and Z,
and optionally mu_o, mu_g and sigma, on a (p, T) grid, interpolated bilinearly with p and T clamped to the grid;
PVTTableSet, the tables of one stream at several GORs, which FluidModel reads at its own GOR; a table from a flash's
phase split; and the CSV files of both.
"""

import math
from dataclasses import dataclass, field

import casadi as ca
import numpy as np
import pandas as pd

from manywells.pvt import sg_from_gas_density
from manywells.pvt.black_oil import live_oil_surface_tension, live_oil_viscosity
from manywells.pvt.dead_oil import dead_oil_surface_tension, dead_oil_viscosity
from manywells.pvt.model import PVTModel
from manywells.units import CF_BAR, CF_RS, M_AIR, R_UNIVERSAL

COLUMNS = ('R_so', 'B_o', 'Z')                  # Every table has them
OPTIONAL_COLUMNS = ('mu_o', 'mu_g', 'sigma')    # A table may have them; the correlations stand in where it does not
TEXT_KEYS = ('name', 'source', 'separation')    # Free text in a file's header


def _grid(values, n_p, n_T, name):
    """A column as a nested tuple of floats, one row per pressure: from an array of shape (n_p, n_T), or (n_p,)."""
    a = np.asarray(values, dtype=float)
    if a.ndim == 1 and n_T == 1:
        a = a.reshape(-1, 1)
    if a.shape != (n_p, n_T):
        raise ValueError(f'{name} must have shape ({n_p}, {n_T}), one value per (p, T) node, got {a.shape}')
    return tuple(tuple(float(v) for v in row) for row in a)


@dataclass(frozen=True)
class PVTTable(PVTModel):
    """
    The hydrocarbon PVT of a fluid from a table on a (p, T) grid (PVT-TAB-1 to PVT-TAB-5). Its gas law is the
    table's z-factor ('table'); the Joule-Thomson factor is DAK's at the gas's gravity, as for every real gas.

    :param p: Pressures of the grid (bar), increasing, at least two
    :param T: Temperatures of the grid (K), increasing, at least one: with one, the properties do not depend on T
    :param R_so: Solution gas-oil ratio (Sm3/Sm3), shape (len(p), len(T))
    :param B_o: Oil formation volume factor (-), the same shape
    :param Z: The free gas's z-factor with the gas constant of rho_g (-), the same shape
    :param rho_o: Stock-tank oil density (kg/m3) that R_so and B_o refer to
    :param rho_g: Gas density at standard conditions (kg/m3) that R_so and Z refer to
    :param mu_o: Oil viscosity (Pa s), or None for the Beggs-Robinson correlations
    :param mu_g: Gas viscosity (Pa s), or None for Lee-Gonzalez-Eakin
    :param sigma: Gas-oil surface tension (J/m2), or None for Abdul-Majeed and Abu Al-Soof
    :param gor: The GOR of the stream the table was made at (Sm3/Sm3), or None
    """
    p: tuple
    T: tuple
    R_so: tuple
    B_o: tuple
    Z: tuple
    rho_o: float
    rho_g: float
    mu_o: tuple = None
    mu_g: tuple = None
    sigma: tuple = None
    gor: float = None
    name: str = field(default='', compare=False)
    source: str = field(default='', compare=False)
    separation: str = field(default='', compare=False)

    gas_law = 'table'

    def __post_init__(self):  # spec: PVT-TAB-1
        p, T = (tuple(float(v) for v in np.atleast_1d(np.asarray(x, dtype=float))) for x in (self.p, self.T))
        object.__setattr__(self, 'p', p)
        object.__setattr__(self, 'T', T)
        for name in COLUMNS + OPTIONAL_COLUMNS:
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _grid(getattr(self, name), len(p), len(T), name))
            elif name in COLUMNS:
                raise ValueError(f'a PVT table needs the column {name}')
        self._validate()
        self._derive()
        grid = [np.array(p)] if len(T) == 1 else [np.array(p), np.array(T)]
        object.__setattr__(self, '_f', {c: ca.interpolant(f'pvt_table_{c}', 'linear', grid,
                                                          np.array(getattr(self, c)).ravel(order='F'))
                                        for c in self.columns})

    def _validate(self):  # spec: PVT-TAB-1
        super()._validate()
        p, T = np.array(self.p), np.array(self.T)
        if len(p) < 2 or len(T) < 1:
            raise ValueError(f'a PVT table needs at least two pressures and one temperature, got {len(p)} and {len(T)}')
        for name, axis in (('p', p), ('T', T)):
            if not (np.all(np.isfinite(axis)) and np.all(axis > 0) and np.all(np.diff(axis) > 0)):
                raise ValueError(f'the grid\'s {name} must be positive, finite and strictly increasing')
        for name in self.columns:
            a = np.array(getattr(self, name))
            if not np.all(np.isfinite(a)):
                raise ValueError(f'{name} must be finite at every node')
            if name != 'R_so' and not np.all(a > 0):
                raise ValueError(f'{name} must be positive at every node')
        R_so = np.array(self.R_so)
        if np.any(np.diff(R_so, axis=0) < -1e-12 * np.maximum(1.0, np.abs(R_so[1:]))):  # To rounding, as at_gor's
            raise ValueError('R_so must not fall as the pressure rises, at any temperature')
        if np.any(R_so < 0) and not (self.mu_o is not None and self.sigma is not None):
            raise ValueError('R_so below 0 needs the columns mu_o and sigma: the live-oil correlations take R_so >= 0')
        if self.gor is not None and not (math.isfinite(self.gor) and self.gor >= 0):
            raise ValueError(f'gor must be None, or non-negative and finite, got {self.gor}')

    @property
    def columns(self) -> tuple:
        """The columns the table has, in file order."""
        return tuple(c for c in COLUMNS + OPTIONAL_COLUMNS if getattr(self, c) is not None)

    # ------------------------------------------------------------------
    # PVT-TAB-2: a column at (p, T), may be CasADi symbolic
    # ------------------------------------------------------------------

    def interpolate(self, column, p, T):  # spec: PVT-TAB-2
        """
        A column at (p, T): bilinear on the grid, with p and T clamped to it; with one temperature, linear in p.

        :param column: One of the table's columns
        :param p: Pressure (bar), may be CasADi symbolic
        :param T: Temperature (K), may be CasADi symbolic
        """
        p = ca.fmin(ca.fmax(p, self.p[0]), self.p[-1])
        if len(self.T) == 1:
            return self._f[column](p)
        T = ca.fmin(ca.fmax(T, self.T[0]), self.T[-1])
        return self._f[column](ca.vertcat(p, T))

    def covers(self, p, T) -> bool:
        """Whether (p, T), floats, lies on the grid; with one temperature, whether p does."""
        return self.p[0] <= p <= self.p[-1] and (len(self.T) == 1 or self.T[0] <= T <= self.T[-1])

    @property
    def grid_range(self) -> str:
        """The grid's range, for messages."""
        T = f'{self.T[0]} K' if len(self.T) == 1 else f'{self.T[0]} to {self.T[-1]} K'
        return f'{self.p[0]} to {self.p[-1]} bar, {T}'

    # ------------------------------------------------------------------
    # PVT-TAB-3 to PVT-TAB-5
    # ------------------------------------------------------------------

    def rs(self, p, T):  # spec: PVT-TAB-3
        """Solution gas-oil ratio at (p, T) (Sm3/Sm3), from the table."""
        return self.interpolate('R_so', p, T)

    def bo(self, p, T):  # spec: PVT-TAB-3
        """Oil formation volume factor at (p, T) (-), from the table."""
        return self.interpolate('B_o', p, T)

    def z_factor(self, p, T):  # spec: PVT-TAB-4
        """The free gas's z-factor at (p, T), with the gas constant of rho_g, from the table."""
        return self.interpolate('Z', p, T)

    def gas_density(self, p, T):  # spec: PVT-TAB-4
        """Gas density at (p, T) (kg/m3): explicit, c_bar p / (Z R_s T)."""
        return CF_BAR * p / (self.z_factor(p, T) * self.R_s * T)

    def gas_law_row(self, p, T, rho_g):  # spec: PVT-TAB-4
        """The gas law as a row (bar), p - rho_g Z R_s T / c_bar with the table's Z, as PVT-GAS-3's."""
        return p - rho_g * self.z_factor(p, T) * self.R_s * T / CF_BAR

    def oil_viscosity(self, p, T):  # spec: PVT-TAB-5
        """The table's oil viscosity at (p, T) (Pa s), or Beggs-Robinson's with the table's R_so."""
        if self.mu_o is not None:
            return self.interpolate('mu_o', p, T)
        return live_oil_viscosity(dead_oil_viscosity(self._api, T), self.rs(p, T) / CF_RS)

    def gas_viscosity(self, p, T, rho_g):  # spec: PVT-TAB-5
        """The table's gas viscosity at (p, T) (Pa s), or Lee-Gonzalez-Eakin's at (T, rho_g)."""
        if self.mu_g is not None:
            return self.interpolate('mu_g', p, T)
        return super().gas_viscosity(p, T, rho_g)

    def oil_surface_tension(self, p, T):  # spec: PVT-TAB-5
        """The table's gas-oil surface tension at (p, T) (J/m2), or the live-oil correlation with the table's R_so."""
        if self.sigma is not None:
            return self.interpolate('sigma', p, T)
        return live_oil_surface_tension(dead_oil_surface_tension(self.rho_o, T), self.rs(p, T) / CF_RS)

    # ------------------------------------------------------------------
    # PVT-TAB-6: from a flash; and files
    # ------------------------------------------------------------------

    @classmethod
    def from_phase_split(cls, p, T, X, rho_L, rho_V, rho_o, rho_g, gor, mu_o=None, mu_g=None, sigma=None,
                         **text) -> 'PVTTable':  # spec: PVT-TAB-6
        """
        The table that reproduces a flash's phase split at the stream's GOR: at each node, the vapour mass fraction X
        of the hydrocarbon stream, and the liquid's and the vapour's densities, rho_L and rho_V (kg/m3), each of shape
        (len(p), len(T)). Where the stream is all liquid, X = 0, and rho_V is the density of the gas that would appear
        first. The stream's separation gives rho_o, rho_g and gor.

        :param p: Pressures (bar)
        :param T: Temperatures (K)
        :param rho_o: The separated stream's stock-tank oil density (kg/m3)
        :param rho_g: Its gas density at standard conditions (kg/m3)
        :param gor: Its gas-oil ratio (Sm3/Sm3)
        :param text: name, source, separation
        """
        p, T = np.asarray(p, dtype=float), np.asarray(T, dtype=float)
        X, rho_L, rho_V = (np.asarray(a, dtype=float).reshape(len(p), len(T)) for a in (X, rho_L, rho_V))
        m = rho_o + gor * rho_g                                       # The stream's mass per Sm3 of stock-tank oil
        R_s = R_UNIVERSAL / (M_AIR * sg_from_gas_density(rho_g))      # As PVTModel.R_s (PVT-GAS-6)
        return cls(p=p, T=T, R_so=gor - X * m / rho_g, B_o=(1 - X) * m / rho_L,
                   Z=CF_BAR * p[:, None] / (rho_V * R_s * T[None, :]), rho_o=rho_o, rho_g=rho_g, mu_o=mu_o,
                   mu_g=mu_g, sigma=sigma, gor=gor, **text)

    @staticmethod
    def read_csv(path) -> 'PVTTable | PVTTableSet':
        """
        A table from a CSV file: '# key: value' lines (rho_o and rho_g, and the free-text name, source, separation
        and gor), then the columns [gor,] p, T, R_so, B_o, Z[, mu_o, mu_g, sigma], one row per node. With a gor
        column, the tables of a GOR axis, each on its own grid (PVTTableSet).
        """
        meta = _read_header(path)
        df = pd.read_csv(path, comment='#', float_precision='round_trip')  # write_csv's values, to the bit
        missing = [c for c in ('p', 'T') + COLUMNS if c not in df]
        if missing:
            raise ValueError(f'{path}: columns {missing} missing')
        unknown = [c for c in df if c not in ('gor', 'p', 'T') + COLUMNS + OPTIONAL_COLUMNS]
        if unknown:
            raise ValueError(f'{path}: unknown columns {unknown}')
        common = dict(rho_o=float(meta['rho_o']), rho_g=float(meta['rho_g']),
                      **{k: meta[k] for k in TEXT_KEYS if k in meta})
        if 'gor' in df:
            return PVTTableSet(tables=tuple(_from_frame(d, path, gor=float(g), **common)
                                            for g, d in df.groupby('gor', sort=True)))
        gor = _number(meta.get('gor'))
        return _from_frame(df, path, gor=gor, **common)

    def frame(self) -> pd.DataFrame:
        """The table as one row per node, temperatures outer and pressures inner, as the files have it."""
        P, TT = np.meshgrid(self.p, self.T, indexing='xy')
        return pd.DataFrame({'p': P.ravel(), 'T': TT.ravel(),
                             **{c: np.array(getattr(self, c)).T.ravel() for c in self.columns}})

    def write_csv(self, path):
        """The table as a CSV file that read_csv reads back."""
        _write(path, self, self.frame())


@dataclass(frozen=True)
class PVTTableSet(PVTModel):
    """
    The tables of one stream at several GORs, a GOR axis (PVT-TAB-1), each on its own grid and with its gor, all
    with the same rho_o and rho_g. FluidModel reads it at its own GOR (at_gor, PVT-TAB-7) when it is built; it has
    no properties of its own at (p, T).
    """
    tables: tuple

    gas_law = 'table'

    def __post_init__(self):  # spec: PVT-TAB-1
        if len(self.tables) < 2 or not all(isinstance(t, PVTTable) for t in self.tables):
            raise ValueError('a GOR axis needs at least two PVTTables')
        if any(t.gor is None for t in self.tables) or len({t.gor for t in self.tables}) < len(self.tables):
            raise ValueError('each table of a GOR axis needs its own gor')
        tables = tuple(sorted(self.tables, key=lambda t: t.gor))
        object.__setattr__(self, 'tables', tables)
        if len({(t.rho_o, t.rho_g) for t in tables}) > 1:
            raise ValueError('the tables of a GOR axis must have the same rho_o and rho_g: one stream on one tie line')
        self._validate()
        self._derive()

    @property
    def rho_o(self) -> float:
        return self.tables[0].rho_o

    @property
    def rho_g(self) -> float:
        return self.tables[0].rho_g

    @property
    def gors(self) -> tuple:
        return tuple(t.gor for t in self.tables)

    def at_gor(self, gor) -> PVTTable:  # spec: PVT-TAB-7
        """
        The table at the GOR gor: linear in GOR between the two neighbouring GORs' tables, each interpolated onto the
        union of their grids (PVT-TAB-2), and exactly a table of the axis at its GOR. A column only one of the two has
        is dropped.

        :raises ValueError: for a GOR outside the axis
        """
        gors = self.gors
        if not gors[0] <= gor <= gors[-1]:
            raise ValueError(f'gor {gor} is outside the PVT table\'s GOR axis, {gors[0]} to {gors[-1]} Sm3/Sm3')
        if gor in gors:
            return self.tables[gors.index(gor)]
        i = int(np.searchsorted(gors, gor)) - 1
        a, b = self.tables[i], self.tables[i + 1]
        w = (gor - a.gor) / (b.gor - a.gor)
        p, T = np.union1d(a.p, b.p), np.union1d(a.T, b.T)
        P, TT = np.meshgrid(p, T, indexing='ij')
        columns = {c: (1 - w) * _on_grid(a, c, P, TT) + w * _on_grid(b, c, P, TT)
                   for c in a.columns if c in b.columns}
        return PVTTable(p=p, T=T, rho_o=a.rho_o, rho_g=a.rho_g, gor=gor, name=a.name, source=a.source,
                        separation=a.separation, **columns)

    def _no_gor(self, *args):
        raise TypeError('a GOR axis has no properties of its own: read it at a GOR with at_gor, as FluidModel does')

    rs = bo = z_factor = gas_density = gas_law_row = oil_viscosity = oil_surface_tension = _no_gor

    def write_csv(self, path):
        """The axis as one CSV file with a gor column, which PVTTable.read_csv reads back."""
        t = self.tables[0]
        _write(path, t, pd.concat([x.frame().assign(gor=x.gor) for x in self.tables])
               [['gor', 'p', 'T', *[c for c in t.columns]]], gor=False)


def _on_grid(table, column, P, TT):
    """A column of a table on the nodes P, TT (PVT-TAB-2), as floats."""
    out = np.empty(P.shape)
    for idx in np.ndindex(P.shape):
        out[idx] = float(table.interpolate(column, P[idx], TT[idx]))
    return out


def _from_frame(df, path, **kw) -> PVTTable:
    p, T = np.unique(df['p'].to_numpy()), np.unique(df['T'].to_numpy())
    if len(df) != len(p) * len(T) or df.duplicated(['p', 'T']).any():
        raise ValueError(f'{path}: not one row per node of a rectangular (p, T) grid')
    df = df.sort_values(['T', 'p'])
    columns = {c: df[c].to_numpy().reshape(len(T), len(p)).T for c in COLUMNS + OPTIONAL_COLUMNS if c in df}
    return PVTTable(p=p, T=T, **columns, **kw)


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_header(path) -> dict:
    meta = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            key, sep, value = line[1:].partition(':')
            if sep:
                meta[key.strip()] = value.strip()
    if 'rho_o' not in meta or 'rho_g' not in meta:
        raise ValueError(f'{path}: the header needs rho_o and rho_g, the densities at standard conditions')
    return meta


def _write(path, table, frame, gor=True):
    header = ['ManyWells PVT table (specs/model/pvt/table.md)']
    header += [f'{k}: {getattr(table, k)}' for k in TEXT_KEYS if getattr(table, k)]
    header += [f'rho_o: {table.rho_o!r}', f'rho_g: {table.rho_g!r}']
    if gor and table.gor is not None:
        header.append(f'gor: {table.gor!r}')
    with open(path, 'w') as fh:
        fh.write(''.join(f'# {h}\n' for h in header))
        frame.to_csv(fh, index=False, float_format='%.17g')
