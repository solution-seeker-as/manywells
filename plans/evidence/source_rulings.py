"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 29 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The effect of the two rulings of 2026-10-29 that follow the sources (specs/model/README.md, Test vectors,
Independent checks): PVT-GAS-5's coefficient 131.0, as Sutton (1985) prints it, for 131.07, and PVT-MIX-9's flowing
mass fraction of gas, as Hasan, Kabir and Sayarpour (2010) define it, for the in-situ one. Both change develop's
default model on every well: its gas law is Dranchuk and Abou-Kassem's, with Sutton's pseudo-critical properties, and
its friction is from roughness, with the mixture viscosity in the Reynolds number.

The wells are those of gor_cap_wells.py: develop's sampler (seed 2026), each at its nominal operating point and at
two sampled ones, solved with the model of the tree the script runs in. Run gor_cap_wells.py before and after each
change, each in its own tree, and join the runs here:

  compare   join two runs on (well, k): the operating points found in each, and the quantiles of the changes in PBH,
            PWH, TWH and the reservoir liquid rate over the cases with an operating point in both
  backends  join a Rust core run and a CasADi backend run of the same tree: whether they find an operating point in
            the same cases, and how far apart their operating points are

From the repository root, one BLAS thread per process:

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    uv run python plans/evidence/gor_cap_wells.py run 500 rust.csv --backend rust
    uv run python plans/evidence/gor_cap_wells.py run 88 casadi.csv --backend casadi
    uv run python plans/evidence/source_rulings.py compare before.csv after.csv
    uv run python plans/evidence/source_rulings.py backends rust.csv casadi.csv
"""

import argparse

import pandas as pd

KEY = ['well', 'k']


def join(a, b, suffixes):
    return pd.read_csv(a).merge(pd.read_csv(b), on=KEY, suffixes=suffixes)


def compare(before, after):
    df = join(before, after, ('_b', '_a'))
    found_b, found_a = df.PBH_b.notna(), df.PBH_a.notna()
    print(f'{len(df)} cases; operating point before {found_b.sum()}, after {found_a.sum()}; '
          f'lost {(found_b & ~found_a).sum()}, gained {(~found_b & found_a).sum()}')
    for name, rows in (('lost', df[found_b & ~found_a]), ('gained', df[~found_b & found_a])):
        if len(rows):
            print(f'  {name}: ' + ', '.join(f'({r.well}, {r.k})' for r in rows.itertuples()))
    both = df[found_b & found_a]
    changes = pd.DataFrame({
        'PBH (bar)': both.PBH_a - both.PBH_b,
        'PWH (bar)': both.PWH_a - both.PWH_b,
        'TWH (K)': both.TWH_a - both.TWH_b,
        'w_res (%)': 100 * (both.w_res_a / both.w_res_b - 1),
    })
    print(f'\n{len(both)} cases with an operating point in both. Quantiles of the changes, after minus before:')
    print(changes.quantile([0, 0.1, 0.5, 0.9, 1]).to_string(float_format=lambda v: f'{v:.4g}'))
    print('\nLargest absolute change, and the median of the absolute changes:')
    print(pd.DataFrame({'max': changes.abs().max(), 'median': changes.abs().median()})
          .to_string(float_format=lambda v: f'{v:.4g}'))
    print(f'\nSeconds per case, median (total): before {df.seconds_b.median():.3f} ({df.seconds_b.sum():.0f}), '
          f'after {df.seconds_a.median():.3f} ({df.seconds_a.sum():.0f})')


def backends(rust, casadi):
    df = join(rust, casadi, ('_r', '_c'))
    found_r, found_c = df.PBH_r.notna(), df.PBH_c.notna()
    print(f'{len(df)} cases in both runs; operating point on the core {found_r.sum()}, on CasADi {found_c.sum()}; '
          f'the core only {(found_r & ~found_c).sum()}, CasADi only {(~found_r & found_c).sum()}')
    both = df[found_r & found_c]
    d_pbh, d_pwh = (both.PBH_r - both.PBH_c).abs().max(), (both.PWH_r - both.PWH_c).abs().max()
    print(f'{len(both)} cases solved by both: largest |PBH difference| {d_pbh:.3g} bar, |PWH| {d_pwh:.3g} bar')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    sub = parser.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('compare')
    c.add_argument('before')
    c.add_argument('after')
    b = sub.add_parser('backends')
    b.add_argument('rust')
    b.add_argument('casadi')
    args = parser.parse_args()
    pd.set_option('display.width', 200)
    if args.cmd == 'compare':
        compare(args.before, args.after)
    else:
        backends(args.rust, args.casadi)


if __name__ == '__main__':
    main()
