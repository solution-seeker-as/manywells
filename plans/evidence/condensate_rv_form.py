"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

How a gas condensate's PVT-TAB-6 table (specs/model/pvt/table.md), with R_so < 0 where the gas carries oil, relates to
the modified black-oil form with R_v, the oil vaporised in the gas (Whitson and Brule 2000, Sec. 7.3), on Pedersen et
al.'s (2015) Table 3.8 condensate, the published example of plans/evidence/table_fluids.py
(plans/sampler-redesign/sources-condensate-tables.md). Three modes:

    make TABLE OUT.csv [--name pedersen2015_t3_8] [--factors 0.8 1.25]
        NeqSim's environment. At each node of table_fluids.py's table TABLE: the stream's TP flash (its vapour mass
        fraction X and phase densities), and each phase separated alone in one stage at standard conditions, as the
        Whitson-Torp method does on a constant-composition path: the liquid's R_s and B_o, the vapour's R_v and B_gd,
        and the four surface densities. Also the TP flash of the stream recombined along its tie line at each factor
        times its GOR.
    check OUT.csv
        Any environment with numpy and pandas. (a) The table's R_so against the one the modified black oil gives at the
        stream's GOR, with one stock-tank oil and one surface gas density (Eqs. 7.6 and 7.7), and the share of the
        stock-tank oil the gas carries where R_so > 0; (b) which columns fall as the pressure rises at each
        temperature; (c) each form read at another GOR (the black-oil rule on R_so, PVT-OIL-13; the modified black
        oil's phase split by Eq. 7.6) against the recombined stream's own flash.
    tables TABLE [TABLE ...]
        Any environment with numpy and pandas. On PVT-TAB-1 tables with rho_o, rho_g and gor in their headers: where
        R_so, the gas density p/Z and the liquid density fall as the pressure rises, as established tools check theirs.

From the repository root, with the JVM sized to fit the memory cap (TABLE from table_fluids.py's make):

    (ulimit -v 8000000; export MALLOC_ARENA_MAX=2 JAVA_TOOL_OPTIONS="-Xmx2g -XX:+UseSerialGC \\
        -XX:ActiveProcessorCount=2 -XX:ReservedCodeCacheSize=128m -XX:MaxMetaspaceSize=256m"; timeout 900 \\
        uv run --no-project --python 3.12 --with neqsim==3.4.0 --with numpy --with pandas \\
        python plans/evidence/condensate_rv_form.py make TABLES/pedersen2015_t3_8.csv RV.csv)
    (ulimit -v 4000000; OMP_NUM_THREADS=1 timeout 900 .venv/bin/python plans/evidence/condensate_rv_form.py check RV.csv)
    (ulimit -v 4000000; OMP_NUM_THREADS=1 timeout 900 .venv/bin/python plans/evidence/condensate_rv_form.py tables \\
        TABLES/*.csv tests/data/pvt/synthetic_eos_oil.csv tests/data/pvt/vazquez_beggs.csv \\
        tests/data/pvt/synthetic_eos_oil_360K.csv)
"""

import argparse

import numpy as np
import pandas as pd

DELTA = 1e-4  # table_fluids.py's liquid trace where the stream is all vapour


def read_header(path):
    meta = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            key, sep, value = line[1:].partition(':')
            if sep:
                meta[key.strip()] = value.strip()
    return meta


# --- NeqSim's side ------------------------------------------------------------------------------------------------------
def separate(tf, fl, x):
    """A phase of composition x separated alone in one stage at standard conditions: its surface gas and oil volumes
    per kg of the phase (Sm3/kg) and their densities (kg/m3), the gas ideal as table_fluids.py's separation has it."""
    phases, _ = fl.flash(x, tf.P_SC, tf.T_REF, properties=False)
    if len(phases) == 2:
        (w_g, _, M_g, _, _, _), (_, rho_os, _, _, _, _) = phases
    elif phases[0][1] < 100.0:  # one phase, a gas at standard conditions
        w_g, M_g, rho_os = 1.0, phases[0][2], np.nan
    else:
        w_g, M_g, rho_os = 0.0, np.nan, phases[0][1]
    rho_gs = tf.P_REF * M_g * 1000 / (tf.R_UNIVERSAL * tf.T_REF)
    V_g = w_g / rho_gs if w_g > 0 else 0.0
    V_o = (1 - w_g) / rho_os if w_g < 1 else 0.0
    return V_g, V_o, rho_gs, rho_os


def column(fl, z, ps, T):
    """The stream's TP flashes along the pressures ps at T: per node X, rho_L, rho_V and the phases' compositions x
    and y. A single phase is the vapour if the nearest two-phase node is mostly vapour by moles, so that its
    saturation point is a dew point, as table_fluids.py's saturation() tells the kind; otherwise the liquid."""
    flashes = [fl.flash(z, p, T, properties=False)[0] for p in ps]
    two = [k for k, ph in enumerate(flashes) if len(ph) == 2]
    out = []
    for k, ph in enumerate(flashes):
        if len(ph) == 2:
            (X, rho_V, _, _, _, y), (_, rho_L, _, _, _, x) = ph
            out.append((X, rho_L, rho_V, x, y))
            continue
        near = min(two, key=lambda j: abs(j - k)) if two else None
        vapour = flashes[near][0][4] > 0.5 if near is not None else False
        rho, comp = ph[0][1], ph[0][5]
        out.append((1.0, np.nan, rho, None, comp) if vapour else (0.0, rho, np.nan, comp, None))
    return out


def make(table, out, name, factors):
    import table_fluids as tf

    meta = read_header(table)
    rho_o, rho_g, gor = (float(meta[k]) for k in ('rho_o', 'rho_g', 'gor'))
    nodes = pd.read_csv(table, comment='#')
    fr, z = tf.example_stream(name)
    fl = tf.Flash(fr)
    sep = fl.separation(z)
    for k, v in (('rho_o', rho_o), ('rho_g', rho_g), ('gor', gor)):
        if abs(sep[k] - v) > 1e-4 * abs(v):
            raise ValueError(f'{name} separates to {k} = {sep[k]}, the table has {v}: not the same stream')
    streams = {f: fl.recombine(sep, f * gor) for f in factors}
    m = rho_o + gor * rho_g
    rows = []
    for T, g in nodes.groupby('T', sort=True):
        g = g.sort_values('p')
        ps = g.p.to_numpy()
        base = column(fl, z, ps, T)
        other = {f: column(fl, zf, ps, T) for f, zf in streams.items()}
        for k, (p, R_so) in enumerate(zip(ps, g.R_so)):
            X, rho_L, rho_V, x, y = base[k]
            row = dict(p=p, T=T, X_table=(gor - R_so) * rho_g / m, X=X, rho_L=rho_L, rho_V=rho_V)
            if x is not None:  # the liquid: R_s and B_o per Sm3 of its own stock-tank oil
                V_g, V_o, rho_go, rho_oo = separate(tf, fl, x)
                row |= dict(R_s=V_g / V_o, B_o=1 / rho_L / V_o, rho_oo=rho_oo, rho_go=rho_go)
            if y is not None:  # the vapour: R_v and B_gd per Sm3 of its own surface gas
                V_g, V_o, rho_gg, rho_og = separate(tf, fl, y)
                row |= dict(R_v=V_o / V_g, B_gd=1 / rho_V / V_g, rho_og=rho_og, rho_gg=rho_gg)
            for f, col in other.items():
                Xf, rLf, rVf, _, _ = col[k]
                row |= {f'X_{f:g}': Xf, f'rho_L_{f:g}': rLf, f'rho_V_{f:g}': rVf}
            rows.append(row)
    with open(out, 'w') as fh:
        fh.write(f'# name: {name}\n# rho_o: {rho_o!r}\n# rho_g: {rho_g!r}\n# gor: {gor!r}\n')
        fh.write(f'# factors: {" ".join(f"{f:g}" for f in factors)}\n')
        pd.DataFrame(rows).to_csv(fh, index=False, float_format='%.10g')
    print(f'{name}: {len(rows)} nodes written to {out}')


# --- The comparison -----------------------------------------------------------------------------------------------------
def falls(df, column, rel=0.0):
    """Pairs of neighbouring nodes, at one temperature, where the column falls as the pressure rises, by more than
    rel of its value."""
    n = 0
    for _, g in df.sort_values(['T', 'p']).groupby('T'):
        y = g[column].to_numpy()
        n += int(np.sum(np.diff(y) < -rel * np.abs(y[1:])))
    return n


def mbo_split(R_s, B_o, R_v, B_gd, rho_o, rho_g, R):
    """The modified black oil's vapour mass fraction and phase densities at GOR R, with one stock-tank oil and one
    surface gas density (Whitson and Brule 2000, Eqs. 7.6 to 7.8): F_oo, the share of the stock-tank oil from the
    liquid, clipped to [0, 1] where the stream is one phase."""
    F = np.clip((1 - R_v * R) / (1 - R_s * R_v), 0.0, 1.0)
    m = rho_o + R * rho_g
    X = np.clip(1 - F * (rho_o + R_s * rho_g) / m, 0.0, 1.0)
    return X, (rho_o + R_s * rho_g) / B_o, (rho_g + R_v * rho_o) / B_gd, F


def no_slip(X, rho_L, rho_V):
    """The no-slip density; a phase without mass needs no density."""
    X, rho_L, rho_V = (np.asarray(a, dtype=float) for a in (X, rho_L, rho_V))
    return 1 / (np.where(X > 0, X / np.where(X > 0, rho_V, 1.0), 0.0) + np.where(X < 1, (1 - X) / rho_L, 0.0))


def check(path):
    meta = read_header(path)
    rho_o, rho_g, gor = (float(meta[k]) for k in ('rho_o', 'rho_g', 'gor'))
    factors = [float(f) for f in meta['factors'].split()]
    df = pd.read_csv(path, comment='#')
    m = rho_o + gor * rho_g
    two = df[(df.X > 0) & (df.X < 1)].copy()
    two['rho_go'] = two.rho_go.where(two.R_s > 0, rho_g)  # no gas from the liquid: its density does not matter
    two['rho_og'] = two.rho_og.where(two.R_v > 0, rho_o)  # no oil from the vapour: likewise
    vap = df[df.X >= 1]
    print(f'{meta["name"]}: GOR {gor:.1f} Sm3/Sm3, rho_o {rho_o:.1f}, rho_g {rho_g:.4f}; {len(df)} nodes, '
          f'{len(two)} two-phase, {len(vap)} all vapour, {int(np.sum(df.X <= 0))} all liquid; all vapour in the '
          f'table: {int(np.sum(df.X_table >= 1 - 1.01 * DELTA))}')  # fmt: skip
    ok = two[two.X < 1 - DELTA]
    print(f'  the flash reproduces the table: largest |X - X_table| {np.max(np.abs(ok.X - ok.X_table)):.1e} on '
          f'{len(ok)} two-phase nodes with X < 1 - delta')  # fmt: skip

    # (a) The mapping at the stream's GOR
    R_so = gor - two.X * m / rho_g  # PVT-TAB-6
    X_m, rL_m, rV_m, F = mbo_split(two.R_s, two.B_o, two.R_v, two.B_gd, rho_o, rho_g, gor)
    R_so_m = F * two.R_s - (1 - F) * rho_o / rho_g
    print('  (a) the modified black oil at the stream GOR, one stock-tank oil and one surface gas density:')
    print(f'      R_so: largest |difference| {np.max(np.abs(R_so_m - R_so)):.1f} Sm3/Sm3 (R_so from '
          f'{R_so.min():.1f} to {R_so.max():.1f}); X: median |difference| {np.median(np.abs(X_m - two.X)):.4f}, '
          f'largest {np.max(np.abs(X_m - two.X)):.4f}')  # fmt: skip
    for col, mbo in (('rho_L', rL_m), ('rho_V', rV_m)):
        e = (mbo - two[col]) / two[col]
        print(f'      {col}: median {np.median(e):+.2%}, from {e.min():+.2%} to {e.max():+.2%}')
    for col, d in (('rho_oo', 'liquid'), ('rho_og', 'vapour'), ('rho_go', 'liquid'), ('rho_gg', 'vapour')):
        ref = rho_o if col.startswith('rho_o') else rho_g
        print(f'      the {d} phase\'s surface {"oil" if col[4] == "o" else "gas"} density: '
              f'{two[col].min():.3f} to {two[col].max():.3f} against the stream\'s {ref:.3f} kg/m3')  # fmt: skip
    # The stream's GOR from separating each phase alone, per kg of stream
    V_oo = 1 / (two.rho_oo + two.R_s * two.rho_go)  # Sm3 of stock-tank oil per kg of liquid
    V_gg = 1 / (two.rho_gg + two.R_v * two.rho_og)  # Sm3 of surface gas per kg of vapour
    R_p = ((1 - two.X) * two.R_s * V_oo + two.X * V_gg) / ((1 - two.X) * V_oo + two.X * two.R_v * V_gg)
    e = R_p / gor - 1
    print(f"      the GOR from separating each phase alone: {e.min():+.2%} to {e.max():+.2%} of the stream's")
    X_b, _, _, _ = mbo_split(two.R_s, two.B_o, two.R_v, two.B_gd, rho_o, rho_g, R_p)
    print(f'      with that GOR in place of the stream\'s, X: median |difference| {np.median(np.abs(X_b - two.X)):.4f}, '
          f'largest {np.max(np.abs(X_b - two.X)):.4f} (one surface density per component alone)')  # fmt: skip
    F_v = (1 - two.X) * V_oo / ((1 - two.X) * V_oo + two.X * two.R_v * V_gg)  # the share by volumes
    F_e = (1 - two.R_v * R_p) / (1 - two.R_s * two.R_v)  # Eq. 7.6 at the per-phase GOR
    print(f'      Eq. 7.6 at that GOR against the volumes: largest |difference| {np.max(np.abs(F_e - F_v)):.1e}')
    pos = (R_so > 0).to_numpy()
    share = 1 - F[pos]
    print(f'      R_so > 0 at {pos.sum()} two-phase nodes, where the gas carries {share.min():.1%} to {share.max():.1%}'
          f' of the stock-tank oil (R_v from {two.R_v[pos].min() * 1e6:.0f} to {two.R_v[pos].max() * 1e6:.0f} '
          f'Sm3 per 10^6 Sm3)')  # fmt: skip

    # (b) Monotonicity in the pressure, at each temperature
    two['R_so'] = R_so
    two['B_o_table'] = (1 - two.X) * m / two.rho_L
    print('  (b) pairs of neighbouring two-phase nodes where the column falls as the pressure rises:')
    print('      ' + ', '.join(f'{c} {falls(two, c)}' for c in ('R_so', 'B_o_table', 'R_s', 'B_o', 'R_v')))
    lows = []
    for T, g in two.sort_values('p').groupby('T'):
        i = int(np.argmin(g.R_v.to_numpy()))
        lows.append(f'{T:g} K: {g.p.iloc[i]:g} bar' + ('' if 0 < i < len(g) - 1 else ' (an end)'))
    print('      the smallest R_v along the two-phase nodes, at ' + '; '.join(lows))

    # (c) Each form read at another GOR, against the recombined stream's own flash
    print(
        "  (c) read at another GOR, on the base stream's two-phase nodes: |X - flash| median / largest; no-slip "
        'density median / largest |error|'
    )
    rL_t = (rho_o + R_so * rho_g) / two.B_o_table  # the table's liquid density, the flash's at the nodes
    two['X_1'], two['rho_L_1'], two['rho_V_1'] = two.X, two.rho_L, two.rho_V
    for f in [1.0] + factors:
        R = f * gor
        mf = rho_o + R * rho_g
        X_true = two[f'X_{f:g}']
        rho_true = no_slip(X_true, two[f'rho_L_{f:g}'], two[f'rho_V_{f:g}'])
        X_rule = (R - np.minimum(R_so, R)) * rho_g / mf  # PVT-OIL-13 on the table, unsmoothed
        X_mbo, rL, rV, _ = mbo_split(two.R_s, two.B_o, two.R_v, two.B_gd, rho_o, rho_g, R)
        one = int(np.sum((X_true >= 1) | (X_true <= 0)))
        for label, X, rho in (
            ('R_so, black-oil rule', X_rule, no_slip(X_rule, rL_t, two.rho_V)),
            ('R_s and R_v, Eq. 7.6', X_mbo, no_slip(X_mbo, rL, rV)),
        ):
            dX, dr = np.abs(X - X_true), np.abs(rho / rho_true - 1)
            print(f'      x{f:g} (GOR {R:.0f}; {one} nodes one phase in its flash), {label}: {np.median(dX):.4f} / '
                  f'{dX.max():.4f}; {np.median(dr):.2%} / {dr.max():.2%}')  # fmt: skip


def tables(paths):
    """Checks that established tools apply, on PVT-TAB-1 tables: the gas density p/Z rising with the pressure (as
    PVDG's B_g falling), the liquid density (rho_o + R_so rho_g)/B_o where the stream is all liquid (as PVTO's
    undersaturated B_o falling), and the bounds of the phase split."""
    print('table: pairs of neighbouring nodes where each falls as the pressure rises (R_so; gas density; liquid '
          'density, at all nodes and, by more than 1e-6 of it, where the stream is all liquid); the vapour mass '
          'fraction X')  # fmt: skip
    for path in paths:
        meta = read_header(path)
        rho_o, rho_g, gor = (float(meta[k]) for k in ('rho_o', 'rho_g', 'gor'))
        df = pd.read_csv(path, comment='#')
        df['X'] = (gor - df.R_so) * rho_g / (rho_o + gor * rho_g)
        df['rho_V'] = df.p / (df.Z * df['T'])  # proportional to the gas density
        df['rho_L'] = (rho_o + df.R_so * rho_g) / df.B_o
        liquid = df[df.X.abs() < 1e-6]  # all liquid, R_so the GOR to the rounding of R_so in the file
        print(f'  {meta["name"]}: {falls(df, "R_so")}; {falls(df, "rho_V")}; {falls(df, "rho_L")}, '
              f'{falls(liquid, "rho_L", 1e-6)} of {len(liquid)} all-liquid nodes; X from {df.X.min():.4f} to '
              f'{df.X.max():.6f}; liquid density {df.rho_L.min():.0f} to {df.rho_L.max():.0f} kg/m3')  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    tb = sub.add_parser('tables')
    tb.add_argument('paths', nargs='+')
    mk = sub.add_parser('make')
    mk.add_argument('table')
    mk.add_argument('out')
    mk.add_argument('--name', default='pedersen2015_t3_8')
    mk.add_argument('--factors', type=float, nargs='+', default=[0.8, 1.25])
    ck = sub.add_parser('check')
    ck.add_argument('out')
    a = ap.parse_args()
    if a.mode == 'make':
        make(a.table, a.out, a.name, a.factors)
    elif a.mode == 'check':
        check(a.out)
    else:
        tables(a.paths)


if __name__ == '__main__':
    main()
