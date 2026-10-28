"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 28 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

An example PVT table from an equation of state (feature spec specs/features/018-pvt-tables.md): a synthetic black
oil, not a real fluid, flashed by NeqSim's Peng-Robinson equation of state with the Peneloux volume correction. It
writes tests/data/pvt/synthetic_eos_oil.csv by the proposed PVT-TAB-6, which turns a flash's phase split into the
black-oil columns that reproduce it.

The composition (mole %): N2 0.5, CO2 2.0, C1 38.0, C2 6.5, C3 5.0, iC4 0.9, nC4 2.4, iC5 0.9, nC5 1.3, nC6 1.8, and
five TBP fractions (molar mass in kg/mol, density at standard conditions in g/cm3): C7 3.0 (0.096, 0.738), C8 3.3
(0.107, 0.765), C9 2.4 (0.121, 0.781), C10-C19 18.0 (0.190, 0.840) and C20+ 14.2 (0.400, 0.920). NeqSim
characterizes the TBP fractions; the binary interaction parameters are its defaults for the classic mixing rule.

The stream is separated in one stage at standard conditions (1.01325 bar, 288.15 K), which gives the stock-tank oil
density rho_o, the gas density at standard conditions rho_g = p_ref M_V / (R T_ref) (ideal, as PVT-GAS-2 has it) and
the gas-oil ratio gor. At each node (p, T), a TP flash of the stream gives the vapour mass fraction X and the phase
densities rho_V and rho_L, and per Sm3 of stock-tank oil, with m = rho_o + gor rho_g the mass of the stream,

    R_so = gor - X m / rho_g,    B_o = (1 - X) m / rho_L,    Z = c_bar p / (rho_V R_s T),    R_s = p_ref / (rho_g T_ref)

Above the bubble point there is no vapour (X = 0), and the gas columns hold the gas that would appear: the incipient
vapour at the bubble point, at its composition there, flashed at (p, T). Its viscosity follows it, and the surface
tension is held at its value at the bubble point. The bubble point of every temperature of the grid is a node of the
pressure axis, so that the bilinear interpolation keeps the kink of R_so and B_o there at that temperature.

With the argument gor, it measures how well one table serves other GORs. The stream is recombined from its own
separator gas and stock-tank oil at 0.6 to 2 times its GOR (the same tie line at standard conditions, so the same
stock-tank densities), and each recombined stream's table is compared, at the nodes from 10 to 400 bar and 300 to
390 K, with the base table read at that GOR by the black-oil rule: gas dissolves up to the smaller of R_so and the GOR
(PVT-OIL-13), and the oil keeps the table's B_o and the live-oil density of PVT-OIL-9 at the table's R_so (PVT-MIX-6).
It then measures a GOR axis: at each GOR but the outermost, the columns interpolated linearly in GOR between the two
neighbouring GORs' tables, against that GOR's own table. With the argument gor_table, it writes
tests/data/pvt/synthetic_eos_oil_gor.csv: the stream at 0.6, 0.8, 1, 1.25 and 1.5 times its GOR, every 20 K, each
GOR on its own grid (its own bubble points), with every column: R_so is negative at low pressure and high temperature,
so the table needs its own viscosities and surface tension (PVT-TAB-1).

NeqSim needs Java and is not a ManyWells dependency. From the repository root:

    uv run --no-project --python 3.12 --with neqsim==3.4.0 --with numpy --with pandas \\
        python plans/evidence/pvt_table_eos.py tests/data/pvt/synthetic_eos_oil.csv
    uv run --no-project --python 3.12 --with neqsim==3.4.0 --with numpy --with pandas \\
        python plans/evidence/pvt_table_eos.py gor_table tests/data/pvt/synthetic_eos_oil_gor.csv
    uv run --no-project --python 3.12 --with neqsim==3.4.0 --with numpy --with pandas \\
        python plans/evidence/pvt_table_eos.py gor
"""

import sys

import numpy as np
import pandas as pd
from neqsim import jneqsim

# ManyWells's constants (manywells.units), copied because this script runs without ManyWells
R_UNIVERSAL = 8314.46       # J/(kmol K)
P_REF, T_REF = 101_325.0, 288.15
CF_BAR = 1e5

DEFINED = [('nitrogen', 0.5), ('CO2', 2.0), ('methane', 38.0), ('ethane', 6.5), ('propane', 5.0), ('i-butane', 0.9),
           ('n-butane', 2.4), ('i-pentane', 0.9), ('n-pentane', 1.3), ('n-hexane', 1.8)]
TBP = [('C7', 3.0, 0.096, 0.738), ('C8', 3.3, 0.107, 0.765), ('C9', 2.4, 0.121, 0.781),
       ('C10_C19', 18.0, 0.190, 0.840), ('C20_plus', 14.2, 0.400, 0.920)]
P_AXIS = np.concatenate([[1.0], np.arange(10.0, 400.0 + 1e-9, 10.0)])     # bar
T_AXIS = np.arange(280.0, 400.0 + 1e-9, 10.0)                               # K


def system():
    s = jneqsim.thermo.system.SystemPrEos(T_REF, P_REF / CF_BAR)
    for name, x in DEFINED:
        s.addComponent(name, x)
    for name, x, M, rho in TBP:
        s.addTBPfraction(name, x, M, rho)
    s.setMixingRule('classic')
    s.useVolumeCorrection(True)
    s.setMultiPhaseCheck(False)
    return s


class Fluid:
    def __init__(self):
        self.s = system()
        self.ops = jneqsim.thermodynamicoperations.ThermodynamicOperations(self.s)
        self.s.init(0)
        self.n = self.s.getPhase(0).getNumberOfComponents()
        self.z = self.composition(self.s.getPhase(0), overall=True)
        assert abs(sum(self.z) - 1) < 1e-9, sum(self.z)

    def composition(self, phase, overall=False):
        return [float(phase.getComponent(i).getz() if overall else phase.getComponent(i).getx()) for i in range(self.n)]

    def set(self, z, p, T):
        self.s.setMolarComposition(_jdouble(z))
        self.s.setTemperature(float(T))
        self.s.setPressure(float(p))

    def flash(self, z, p, T):
        """TP flash: [(phase type, mass fraction, density kg/m3, molar mass kg/mol, viscosity Pa s)], IFT (N/m)."""
        self.set(z, p, T)
        self.ops.TPflash()
        self.s.initPhysicalProperties()
        n = self.s.getNumberOfPhases()
        phases = []
        for i in range(n):
            ph = self.s.getPhase(i)
            beta = float(ph.getBeta()) if n > 1 else 1.0
            phases.append((str(ph.getType()), beta * float(ph.getMolarMass()), float(ph.getDensity('kg/m3')),
                           float(ph.getMolarMass()), float(ph.getViscosity('kg/msec'))))
        total = sum(q[1] for q in phases)
        phases = [(t, m / total, rho, M, mu) for t, m, rho, M, mu in phases]
        ift = float(self.s.getInterfacialTension(0, 1)) if n > 1 else np.nan
        return phases, ift

    def bubble_point(self, z, T, p_guess=150.0):
        """Bubble-point pressure (bar) of the stream z at T, and the incipient vapour's composition."""
        self.set(z, p_guess, T)
        self.ops.bubblePointPressureFlash(False)
        return float(self.s.getPressure()), self.composition(self.s.getPhase(0))

    def separator_phases(self, z):
        """Mole fractions of vapour and liquid, and their compositions, at standard conditions."""
        self.set(z, P_REF / CF_BAR, T_REF)
        self.ops.TPflash()
        phases = sorted((self.s.getPhase(i) for i in range(2)), key=lambda ph: float(ph.getDensity('kg/m3')))
        return [(float(ph.getBeta()), self.composition(ph)) for ph in phases]


def _jdouble(z):
    from jpype import JArray, JDouble
    return JArray(JDouble)([float(v) for v in z])


def separation(f, z):
    """Single-stage separation at standard conditions: rho_o (kg/m3), rho_g (kg/Sm3) and gor (Sm3/Sm3)."""
    phases, _ = f.flash(z, P_REF / CF_BAR, T_REF)
    (_, X, _, M_V, _), (_, _, rho_L, _, _) = sorted(phases, key=lambda q: q[2])   # vapour first
    rho_g = P_REF * M_V * 1000 / (R_UNIVERSAL * T_REF)                              # M_V in kg/mol -> kg/kmol
    gor = (X / rho_g) / ((1 - X) / rho_L)
    return rho_L, rho_g, gor


def recombined(f, factor):
    """The stream with its separator gas scaled by factor against its stock-tank oil: its GOR times factor."""
    (b_V, y), (b_L, x) = f.separator_phases(f.z)
    z = np.array(y) * b_V * factor + np.array(x) * b_L
    return list(z / z.sum())


def tabulate(f, z, T_axis=T_AXIS):
    """The table of stream z (PVT-TAB-6) as a DataFrame, and its separation and bubble points."""
    rho_o, rho_g, gor = separation(f, z)
    m = rho_o + gor * rho_g
    R_s = P_REF / (rho_g * T_REF)
    p_b = {T: f.bubble_point(z, T) for T in T_axis}
    p_axis = np.unique(np.concatenate([P_AXIS, np.round([p for p, _ in p_b.values()], 3)]))
    rows, missed = [], 0
    for T in T_axis:
        pb, y_b = p_b[T]
        sigma_b = None
        for p in sorted(p_axis):
            phases, ift = f.flash(z, p, T)
            if p < pb - 1e-6 and len(phases) < 2:
                missed += 1                                     # A flash that missed the split: retry just below p_b
                phases, ift = f.flash(z, p * (1 - 1e-6), T)
            if len(phases) == 2:
                (_, X, rho_V, _, mu_g), (_, _, rho_L, _, mu_o) = sorted(phases, key=lambda q: q[2])
                sigma_b = ift
            else:                                               # Liquid only: the gas is the incipient vapour
                (_, _, rho_L, _, mu_o), X = phases[0], 0.0
                gas, _ = f.flash(y_b, p, T)
                rho_V = sum(q[1] for q in gas) / sum(q[1] / q[2] for q in gas)      # Its density, whole
                mu_g = min(gas, key=lambda q: q[2])[4]                  # The vapour's, where it splits at its dew point
            rows.append(dict(p=p, T=T, R_so=gor - X * m / rho_g, B_o=(1 - X) * m / rho_L,
                             Z=CF_BAR * p / (rho_V * R_s * T), mu_o=mu_o, mu_g=mu_g,
                             sigma=sigma_b if X == 0.0 else ift))
    df = pd.DataFrame(rows)
    # Above the bubble point, the surface tension is held at its value there
    df['sigma'] = df.groupby('T')['sigma'].transform(lambda s: s.ffill())
    assert df.notna().all().all(), df[df.isna().any(axis=1)]
    return df, dict(rho_o=rho_o, rho_g=rho_g, gor=gor, p_b=p_b, missed=missed, n_p=len(p_axis))


def main(out):
    f = Fluid()
    df, s = tabulate(f, f.z)
    header = [
        '# ManyWells PVT table (specs/model/pvt/table.md)',
        '# name: synthetic_eos_oil',
        '# source: plans/evidence/pvt_table_eos.py; NeqSim 3.4.0, Peng-Robinson with Peneloux volume correction; '
        'a synthetic composition, not a real fluid',
        '# separation: single stage at 1.01325 bar and 288.15 K',
        f'# rho_o: {s["rho_o"]:.4f}',
        f'# rho_g: {s["rho_g"]:.6f}',
        f'# gor: {s["gor"]:.4f}',
        '# p_bubble: ' + ', '.join(f'{T:.0f}={p:.3f}' for T, (p, _) in s['p_b'].items()) + '  (T in K = p in bar)',
        '# columns: p (bar), T (K), R_so (Sm3/Sm3), B_o (-), Z (-, with R_s at standard conditions), '
        'mu_o and mu_g (Pa s), sigma (N/m)',
    ]
    with open(out, 'w') as fh:
        fh.write('\n'.join(header) + '\n')
        df.to_csv(fh, index=False, float_format='%.10g')
    rho_air = P_REF * 28.97 / (R_UNIVERSAL * T_REF)
    print(f'rho_o {s["rho_o"]:.2f} kg/m3, rho_g {s["rho_g"]:.4f} kg/Sm3 (gravity {s["rho_g"] / rho_air:.3f}), '
          f'gor {s["gor"]:.2f} Sm3/Sm3')
    print('bubble points (bar):', {int(T): round(p, 2) for T, (p, _) in s['p_b'].items()})
    print(f'{len(df)} nodes ({s["n_p"]} x {len(T_AXIS)}), {s["missed"]} flashes missed the split and were retried')
    print(df.describe().T[['min', 'max']].to_string())


def split(df, s, gor=None):
    """Free-gas mass fraction X of the hydrocarbon stream, rho_L and rho_V at each node, by PVT-TAB-3 and PVT-TAB-4
    at the GOR gor (the table's own by default): the black-oil reading of a table, with R_so capped at the gas there."""
    gor = s['gor'] if gor is None else gor
    m = s['rho_o'] + gor * s['rho_g']
    R_d = np.minimum(df.R_so, gor)
    X = (gor - R_d) * s['rho_g'] / m
    rho_L = (s['rho_o'] + df.R_so * s['rho_g']) / df.B_o                               # PVT-OIL-9, as PVT-MIX-6 has it
    rho_V = CF_BAR * df.p / (df.Z * P_REF / (s['rho_g'] * T_REF) * df['T'])
    return X, rho_L, rho_V


def nodes(d):
    """The nodes every table shares (P_AXIS), from 10 bar and from 300 K to 390 K."""
    return d[d.p.isin(P_AXIS) & (d.p >= 10) & (d['T'] >= 300) & (d['T'] <= 390)].reset_index(drop=True)


def errors(got, want):
    """Errors of a reading (X, rho_L, rho_V) against the stream's own table, and of the no-slip density."""
    ns = lambda X, L, V: 1 / (X / V + (1 - X) / L)
    d_ns = ns(*got) / ns(*want) - 1
    return {'max |dX|': np.max(np.abs(got[0] - want[0])), 'max |d rho_L| (%)': 100 * np.max(np.abs(got[1] / want[1] - 1)),
            'max |d rho_V| (%)': 100 * np.max(np.abs(got[2] / want[2] - 1)),
            'median |d rho_ns| (%)': 100 * np.median(np.abs(d_ns)), 'max |d rho_ns| (%)': 100 * np.max(np.abs(d_ns))}


def gor_study():
    """
    One table read at other GORs by the black-oil rule, against the stream recombined at those GORs; and a GOR axis:
    the columns interpolated linearly in GOR between the two neighbouring GORs' tables.
    """
    f = Fluid()
    tables = {1.0: tabulate(f, f.z)}
    s0 = tables[1.0][1]
    rows = []
    for factor in (0.6, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0):
        df, s = tables[factor] = tabulate(f, recombined(f, factor))
        assert abs(s['rho_o'] / s0['rho_o'] - 1) < 1e-6 and abs(s['rho_g'] / s0['rho_g'] - 1) < 1e-6, (s, s0)
        rows.append({'GOR': round(s['gor'], 1), 'GOR / table\'s': factor, 'p_b at 360 K (bar)': round(s['p_b'][360.0][0], 1),
                     **errors(split(nodes(tables[1.0][0]), s0, s['gor']), split(nodes(df), s))})
    pd.set_option('display.width', 200)
    print(f"The table at GOR {s0['gor']:.1f}, read at the stream's GOR by the black-oil rule, against the stream's own "
          'table; nodes 10-400 bar, 300-390 K:')
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    rows = []
    factors = sorted(tables)
    for lo, mid, hi in zip(factors, factors[1:], factors[2:]):
        (a, s_a), (df, s), (b, s_b) = tables[lo], tables[mid], tables[hi]
        w = (s['gor'] - s_a['gor']) / (s_b['gor'] - s_a['gor'])
        a, b = nodes(a), nodes(b)
        mixed = a.copy()
        for c in ('R_so', 'B_o', 'Z'):
            mixed[c] = (1 - w) * a[c] + w * b[c]
        rows.append({'GOR': round(s['gor'], 1), 'between': f'{s_a["gor"]:.1f} and {s_b["gor"]:.1f}',
                     **errors(split(mixed, s), split(nodes(df), s))})
    print('\nA GOR axis: the columns interpolated linearly in GOR between the neighbouring tables, against the stream\'s '
          'own table, at the same nodes:')
    print(pd.DataFrame(rows).round(4).to_string(index=False))


GOR_FACTORS = (0.6, 0.8, 1.0, 1.25, 1.5)


def gor_table(out):
    """synthetic_eos_oil_gor.csv: the stream recombined at five GORs, each GOR with its own (p, T) grid."""
    f = Fluid()
    frames, gors = [], []
    for factor in GOR_FACTORS:
        df, s = tabulate(f, f.z if factor == 1.0 else recombined(f, factor), np.arange(280.0, 400.0 + 1e-9, 20.0))
        frames.append(df.assign(gor=round(s['gor'], 4)))
        gors.append(s)
    s0 = gors[GOR_FACTORS.index(1.0)]
    header = [
        '# ManyWells PVT table (specs/model/pvt/table.md)',
        '# name: synthetic_eos_oil_gor',
        '# source: plans/evidence/pvt_table_eos.py gor_table; synthetic_eos_oil.csv\'s stream recombined from its own '
        'separator gas and stock-tank oil at 0.6, 0.8, 1, 1.25 and 1.5 times its GOR, each GOR with its own grid',
        '# separation: single stage at 1.01325 bar and 288.15 K',
        f'# rho_o: {s0["rho_o"]:.4f}',
        f'# rho_g: {s0["rho_g"]:.6f}',
        '# p_bubble at 360 K: ' + ', '.join(f'{s["gor"]:.1f}={s["p_b"][360.0][0]:.3f}' for s in gors)
        + '  (GOR in Sm3/Sm3 = p in bar)',
        '# columns: gor (Sm3/Sm3), p (bar), T (K), R_so (Sm3/Sm3), B_o (-), Z (-, with R_s at standard conditions), '
        'mu_o and mu_g (Pa s), sigma (N/m)',
    ]
    df = pd.concat(frames)[['gor', 'p', 'T', 'R_so', 'B_o', 'Z', 'mu_o', 'mu_g', 'sigma']]
    with open(out, 'w') as fh:
        fh.write('\n'.join(header) + '\n')
        df.to_csv(fh, index=False, float_format='%.10g')
    print(f'{len(df)} rows; GORs', [round(s['gor'], 2) for s in gors])


if __name__ == '__main__':
    if sys.argv[1] == 'gor':
        gor_study()
    elif sys.argv[1] == 'gor_table':
        gor_table(sys.argv[2])
    else:
        main(sys.argv[1])
