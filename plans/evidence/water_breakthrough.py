"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 11 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The oils' water along a life under variants of SMP-86's breakthrough, SMP-69's start and SMP-83's end (G30 of
plans/sampler-redesign/guesses.md; D-135, D-140 and D-141): the water cut of each well-year by years on production, the share
of lives that start dry, the recovery at breakthrough of those that do, and a proxy of the decline of the oils under
pressure support, whose liquid rate holds while their reservoir pressure does: the oil rate's 1 - WLR by fifth of the
life over its first year's. The lives are the package's (manywells.sampling.v2.life.Life), core oil wells of the
seed's draws with their black oil. The water depends only on the depletion clock and the water's draws, so no well is
solved. Each variant replays the well's own numbers, those of its `evolution` and `stage` blocks (SMP-51):

    r10      R10's rules: F_bt ~ U(0.05, 0.4) under pressure support and U(0.2, 0.7) under depletion, every start dry
    late_bt  D-135 alone: F_bt ~ U(0.1, 0.9) for both drives, every start dry
    d140     D-135 and D-140's wet starts, a third of the oils under pressure support
    current  the package: D-140, and D-141's end under pressure support, U(0.85, 0.98)
    wet50    the package's rules with wet starts at half of those oils

The variants before D-141 end a life under pressure support at a WLR of 0.98.

    (ulimit -v 8000000; OMP_NUM_THREADS=1 .venv/bin/python plans/evidence/water_breakthrough.py [N_OILS] [SEED]
        [--reference DIR])

With --reference, DIR holds per-publisher folders of well statistics (publisher, group, basis, quantity, ...
p25, p50, p75 ...), the maintainer's analyses of open per-well data, kept outside the repo; the script prints the
spans of the regions' medians next to the variants'. Its output is not committed.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from manywells.sampling.v2.draws import _P_WET_START, _WLR_WET, draw_well
from manywells.sampling.v2.life import _WLR_MAX, Life
from manywells.sampling.wells import rng_for

BINS = ((0, 2), (2, 5), (5, 10), (10, 20), (20, 30))
R10_RANGES = {'pressure_support': (0.05, 0.4), 'depletion': (0.2, 0.7)}  # SMP-86's F_bt before D-135
VARIANTS = ('r10', 'late_bt', 'd140', 'current', 'wet50')


def logit(x):
    return math.log(x / (1 - x))


def expit(x):
    return 1 / (1 + math.exp(-x))


def water(F, a, z, F_bt, bt):
    """SMP-86's oil water at the recovery F: the start's a until F_bt, then logit-linear in F from bt to the end's z."""
    if z <= a or F <= F_bt:
        return a
    bt = min(bt, z)
    return max(a, expit(logit(bt) + (logit(z) - logit(bt)) * (F - F_bt) / (1 - F_bt)))


def variants(seed, k, life, drive):
    """Each variant's water as a function of F: the well's evolution numbers (F_bt's is the sixth, the jump's the
    seventh) and its stage numbers (the dry start's third and fourth, the wet start's fifteenth and sixteenth)."""
    ev = rng_for(seed, k, 'v2', 'evolution').random(7)
    st = rng_for(seed, k, 'v2', 'stage').random(16)
    support = drive == 'pressure_support'
    z = life.parameters()['water_end']
    z98 = _WLR_MAX if support else z  # The end before D-141
    dry0 = (0.0 if st[2] < 0.5 else 0.1 * st[3]) if support else (0.0 if st[4] < 0.5 else 0.1 * st[5])
    bt = 0.05 + 0.25 * ev[6] if support else 0.02 + 0.08 * ev[6]
    lo, hi = R10_RANGES[drive]
    wet0 = _WLR_WET[0] + (_WLR_WET[1] - _WLR_WET[0]) * st[15]

    def late(end):
        return lambda F: water(F, dry0, end, 0.1 + 0.8 * ev[5], bt)

    def wet(end):
        return lambda F: water(F, wet0, end, 0.0, wet0)

    wet30, wet50 = support and st[14] < _P_WET_START, support and st[14] < 0.5
    assert ('wet_start' in life.w0['flags'].split(';')) == wet30
    return {
        'r10': lambda F: water(F, dry0, z98, lo + (hi - lo) * ev[5], bt),
        'late_bt': late(z98),
        'd140': wet(z98) if wet30 else late(z98),
        'current': None,  # Life.water
        'wet50': wet(z) if wet50 else late(z),
    }


def lives(n_oils, seed):
    """The first n_oils core oil lives of the seed's draws, with each variant's water at mid-year and on a grid."""
    rows, k = [], 0
    while len(rows) < n_oils:
        w, _ = draw_well(seed, k, 'initial')
        if w['coverage_tail'] == 'core' and w['fluid_type'] in ('black_oil', 'volatile_oil'):
            life = Life(seed, k)
            fs = variants(seed, k, life, w['drive'])
            years = np.arange(0.5, life.T_life, 1.0)
            grid = np.linspace(0, life.T_life, 201)[1:]
            rows.append(dict(k=k, drive=w['drive'], life=life, years=years, grid=grid))
            for name in VARIANTS:
                f = fs[name]
                rows[-1][name] = np.array([life.water(t) if f is None else f(life.F(t)) for t in years])
                rows[-1][name + '_grid'] = np.array([life.water(t) if f is None else f(life.F(t)) for t in grid])
        k += 1
    return rows


def well_years(rows):
    return pd.DataFrame(
        [
            dict(k=r['k'], drive=r['drive'], years=y, **{n: r[n][i] for n in VARIANTS})
            for r in rows
            for i, y in enumerate(r['years'])
        ]
    )


def decline(rows):
    """The oils under pressure support: 1 - WLR by fifth of the life over its first year's, medians."""
    out = {n: [] for n in VARIANTS}
    for r in rows:
        if r['drive'] != 'pressure_support':
            continue
        T, ts = r['life'].T_life, r['grid']
        for n in VARIANTS:
            o = 1 - r[n + '_grid']
            first = o[ts <= 1.0].mean()
            out[n].append([o[(ts > i * T / 5) & (ts <= (i + 1) * T / 5)].mean() / first for i in range(5)])
    return {n: np.median(np.array(v), axis=0) for n, v in out.items()}


def first_year_dry(rows):
    return {n: np.mean([r[n][0] < 0.05 for r in rows]) for n in VARIANTS}


def reference(directory):
    frames = [pd.read_csv(f) for f in sorted(Path(directory).glob('*/well_statistics.csv'))]
    return pd.concat(frames)


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
        cells = [' / '.join(f'{v:.2f}' for v in np.percentile(wy.loc[m, n], (25, 50, 75))) for n in VARIANTS]
        print(
            f'  {lo:>2} to {hi:<2} years (n={m.sum():>5}): ' + ' | '.join(f'{n} {c}' for n, c in zip(VARIANTS, cells))
        )
    print(
        '\nFirst year dry (water cut below 0.05): ' + ', '.join(f'{n} {v:.2f}' for n, v in first_year_dry(rows).items())
    )
    print('\nOils under pressure support, 1 - WLR by fifth of the life over the first year, medians')
    for n, v in decline(rows).items():
        print(f'  {n}: ' + ' '.join(f'{x:.2f}' for x in v))
    if ref:
        df = reference(ref)
        print('\nThe regions (not for the repo)')
        wc = df[(df.quantity == 'water_cut') & (df.basis == 'well-year') & df.group.str.contains('oil wells / ')]
        wc = wc.assign(bin=wc.group.str.extract(r'oil wells / (.+) years on production')[0])
        for lo, hi in BINS:
            g = wc[wc.bin == f'{lo} to {hi}']
            if len(g):
                print(f'  water cut, {lo} to {hi} years ({len(g)} groups): P50 {g.p50.min():.2f} to {g.p50.max():.2f}')
        d = df[(df.quantity == 'rate_over_first_year') & df.group.str.contains('oil wells / fifth')]
        d = d.assign(fifth=d.group.str.extract(r'fifth (\d)')[0].astype(int))
        spans = d.groupby('fifth').p50.agg(['min', 'max'])
        print(
            "  oil wells' rate over the first year by fifth, span of the medians: "
            + ', '.join(f'{f}: {a:.2f} to {b:.2f}' for f, (a, b) in spans.iterrows())
        )


if __name__ == '__main__':
    main()
