"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 11 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The oils' water along a life under variants of SMP-86's breakthrough (G30 of plans/sampler-redesign/guesses.md):
the water cut of each well-year by years on production, the share of lives that start dry, and the recovery at
breakthrough of those that do. The lives are the package's (manywells.sampling.v2.life.Life), core oil wells of
the seed's draws, with their black oil; the water depends only on the depletion clock and the water's draws, so no
well is solved. Each variant reuses the life's own numbers: a later breakthrough maps the drawn F_bt's uniform
number onto the new range, and a wet start takes two numbers of a generator of its own.

    current        SMP-86 as specified
    late_bt        F_bt ~ U(0.1, 0.9) for both drives
    late_bt_wet30  late_bt, and a well under pressure support starts wet with probability 0.3: its WLR starts at
                   U(0.05, 0.9) and rises from the start, logit-linear in F, to the end's
    late_bt_wet50  the same with probability 0.5

    (ulimit -v 8000000; OMP_NUM_THREADS=1 .venv/bin/python plans/evidence/water_breakthrough.py [N_OILS] [SEED]
        [--reference DIR])

With --reference, DIR holds per-publisher folders of well statistics (publisher, group, basis, quantity, ...
p25, p50, p75 ...), the maintainer's analyses of open per-well data, kept outside the repo; the script prints the
span of the regions' well-year medians by years on production next to the variants'. Its output is not committed.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from manywells.sampling.v2.draws import draw_well
from manywells.sampling.v2.life import Life

BINS = ((0, 2), (2, 5), (5, 10), (10, 20), (20, 30))
RANGES = {'pressure_support': (0.05, 0.4), 'depletion': (0.2, 0.7)}  # SMP-86's F_bt
VARIANTS = ('current', 'late_bt', 'late_bt_wet30', 'late_bt_wet50')


def logit(x):
    return math.log(x / (1 - x))


def expit(x):
    return 1 / (1 + math.exp(-x))


def water(F, a, z, F_bt, bt):
    """SMP-86's oil water at the recovery F: the start's a until F_bt, then logit-linear in F from bt to the end's z."""
    if z <= a or F < F_bt:
        return a
    bt = min(bt, z)
    return max(a, expit(logit(bt) + (logit(z) - logit(bt)) * (F - F_bt) / (1 - F_bt)))


def lives(n_oils, seed):
    """The parameters of the first n_oils core oil lives of the seed's draws."""
    rows, k = [], 0
    while len(rows) < n_oils:
        w, _ = draw_well(seed, k, 'initial')
        if w['coverage_tail'] == 'core' and w['fluid_type'] in ('black_oil', 'volatile_oil'):
            life = Life(seed, k)
            years = np.arange(0.5, life.T_life, 1.0)
            rows.append(dict(k=k, drive=w['drive'], life=life, years=years, F=np.array([life.F(t) for t in years])))
        k += 1
    return rows


def well_years(rows):
    out = []
    for r in rows:
        life, drive = r['life'], r['drive']
        p = life.parameters()
        a, z, F_bt, bt = p['water_start'], p['water_end'], p['F_bt'], p['wlr_bt']
        lo, hi = RANGES[drive]
        F_late = 0.1 + 0.8 * (F_bt - lo) / (hi - lo)
        u_wet, u_w0 = np.random.default_rng([2026, 11, 11, r['k']]).random(2)
        a_wet = 0.05 + 0.85 * u_w0
        for y, F in zip(r['years'], r['F']):
            row = dict(k=r['k'], drive=drive, years=y)
            row['current'] = life.water(y)
            row['late_bt'] = water(F, a, z, F_late, bt)
            for name, p_wet in (('late_bt_wet30', 0.3), ('late_bt_wet50', 0.5)):
                wet = drive == 'pressure_support' and u_wet < p_wet
                row[name] = water(F, a_wet, z, 0.0, a_wet) if wet else row['late_bt']
            out.append(row)
    return pd.DataFrame(out)


def breakthrough(rows):
    """The share of lives whose first year is dry (water cut below 0.05), and the recovery at breakthrough of those."""
    out = {}
    for name in VARIANTS:
        dry, F_bts = 0, []
        for r in rows:
            life, drive = r['life'], r['drive']
            p = life.parameters()
            lo, hi = RANGES[drive]
            F_bt = {'current': p['F_bt']}.get(name, 0.1 + 0.8 * (p['F_bt'] - lo) / (hi - lo))
            u_wet = np.random.default_rng([2026, 11, 11, r['k']]).random(2)[0]
            p_wet = {'late_bt_wet30': 0.3, 'late_bt_wet50': 0.5}.get(name, 0.0)
            wet = drive == 'pressure_support' and u_wet < p_wet
            if not wet and p['water_start'] < 0.05:
                dry += 1
                if p['water_end'] > p['water_start']:
                    F_bts.append(F_bt)
        out[name] = (dry / len(rows), np.percentile(F_bts, (25, 50, 75)))
    return out


def reference(directory):
    """The span over the regions' groups of the oil wells' well-year water cut quantiles, by years on production."""
    frames = [pd.read_csv(f) for f in sorted(Path(directory).glob('*/well_statistics.csv'))]
    df = pd.concat(frames)
    df = df[(df.quantity == 'water_cut') & (df.basis == 'well-year') & df.group.str.contains('oil wells / ')]
    df = df.assign(bin=df.group.str.extract(r'oil wells / (.+) years on production')[0])
    return df


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    n_oils, seed = (int(args[0]) if args else 600), (int(args[1]) if len(args) > 1 else 2027)
    ref = sys.argv[sys.argv.index('--reference') + 1] if '--reference' in sys.argv else None
    rows = lives(n_oils, seed)
    wy = well_years(rows)
    print(f'{len(rows)} core oil lives of seed {seed}, {len(wy)} well-years')
    print('\nWater cut of the well-years by years on production, P25 / P50 / P75 (all core oils)')
    for lo, hi in BINS:
        m = (wy.years >= lo) & (wy.years < hi)
        cells = [' / '.join(f'{v:.2f}' for v in np.percentile(wy.loc[m, name], (25, 50, 75))) for name in VARIANTS]
        print(
            f'  {lo:>2} to {hi:<2} years (n={m.sum():>5}): ' + ' | '.join(f'{n} {c}' for n, c in zip(VARIANTS, cells))
        )
    print('\nMedians by drive')
    for drive, g in wy.groupby('drive'):
        for lo, hi in BINS:
            m = (g.years >= lo) & (g.years < hi)
            print(f'  {drive}, {lo} to {hi}: ' + ', '.join(f'{n} {g.loc[m, n].median():.2f}' for n in VARIANTS))
    print('\nFirst year dry (water cut below 0.05), and F_bt of the dry starts that break through, P25 / P50 / P75')
    for name, (dry, q) in breakthrough(rows).items():
        print(f'  {name}: dry {dry:.2f}; F_bt {" / ".join(f"{v:.2f}" for v in q)}')
    if ref:
        df = reference(ref)
        print(
            '\nThe regions (not for the repo): span over the groups of the well-year quantiles, by years on production'
        )
        for lo, hi in BINS:
            g = df[df.bin == f'{lo} to {hi}']
            if len(g):
                spans = [f'{q} {g[q].min():.2f} to {g[q].max():.2f}' for q in ('p25', 'p50', 'p75')]
                print(f'  {lo} to {hi} years ({len(g)} groups): ' + '; '.join(spans))
        for lo, hi in BINS:
            for _, row in df[df.bin == f'{lo} to {hi}'].iterrows():
                print(f'    {row.publisher} {row.group}: P25 {row.p25:.2f}, P50 {row.p50:.2f}, P75 {row.p75:.2f}')


if __name__ == '__main__':
    main()
