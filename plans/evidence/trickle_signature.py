"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Trickle-root signature in the published manywells-sol-1 data (v2 plan, "Known v1 defect").

Counts samples whose wellhead pressure is within a threshold of the pressure downstream of
the choke (PWH - PDC), the signature of the unstable trickle root, and shows how they
account for the low-temperature spike in the TWH histogram. This is a heuristic; the exact
count comes from relabelling the samples (Step 3). Needs only pandas and numpy:

    python plans/evidence/trickle_signature.py manywells-sol-1.zip
"""

import argparse

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[2])
    parser.add_argument('data', help='path to manywells-sol-1.zip')
    args = parser.parse_args()

    d = pd.read_csv(args.data, compression='zip')
    dp = d.PWH - d.PDC
    print(f'{len(d)} samples, {d.ID.nunique()} wells')
    for thr in [1e-3, 1e-2, 0.1, 1.0]:
        m = dp < thr
        print(f'PWH-PDC < {thr:g} bar: {m.mean() * 100:5.2f}% of samples, {d.loc[m, "ID"].nunique()} wells; '
              f'TWH median {d.loc[m, "TWH"].median():.1f} K, WLIQ median {d.loc[m, "WLIQ"].median():.3f} kg/s')
    rest = dp >= 1
    print(f'PWH-PDC >= 1 bar: TWH median {d.loc[rest, "TWH"].median():.1f} K, '
          f'WLIQ median {d.loc[rest, "WLIQ"].median():.3f} kg/s')

    edges = np.arange(270, 380, 5)
    h, _ = np.histogram(d.TWH, bins=edges)
    print('TWH histogram (5 K bins), with the samples that have PWH-PDC < 10 mbar:')
    for i in range(len(h)):
        in_bin = (d.TWH >= edges[i]) & (d.TWH < edges[i + 1])
        print(f'  {edges[i]:.0f}-{edges[i + 1]:.0f} K: {h[i]:7d}   signature: {(in_bin & (dp < 1e-2)).sum():6d}')


if __name__ == '__main__':
    main()
