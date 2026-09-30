"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The stable-root reference for the Distributions check (v2 plan, Step 3), written to
verification/data/distribution_reference.json. Runs in the develop environment:

    uv run python verification/build/distribution_reference.py --data-dir <dir>

<dir> holds manywells-sol-1.zip, manywells-nsol-1.zip and manywells-nscl-1.zip from the
solution-seeker-as/manywells dataset. For sol-1 and nsol-1, the rows on the trickle-root signature
(PWH within 10 mbar of PDC) are removed and the rest summarized. The signature catches most trickle
rows but not all (47 of the 64 unstable roots in the verifier's case set, and none of its 136 stable
roots), so the counts are lower bounds. The script also prints the counts for all three datasets,
for docs/corrigendum.md, and how the check's bounds compare with sampling noise and with the
trickle rows left in.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from manywells_verify.distributions import (FEATURES, MAX_CDF_GAP, MAX_CORR_GAP, REFERENCE, TRICKLE_DP, compare,
                                            summarize, trickle_signature)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    out = {'trickle_dp_bar': TRICKLE_DP, 'features': list(FEATURES), 'datasets': {}, 'trickle_signature': {}}
    rng = np.random.default_rng(20260930)

    for name in ('sol-1', 'nsol-1', 'nscl-1'):
        df = pd.read_csv(args.data_dir / f'manywells-{name}.zip', compression='zip')
        trickle = trickle_signature(df)
        out['trickle_signature'][name] = {'rows': int(trickle.sum()), 'share': float(trickle.mean()),
                                          'wells': int(df.loc[trickle, 'ID'].nunique()), 'of_rows': int(len(df)),
                                          'median_TWH_signature': float(df.loc[trickle, 'TWH'].median()),
                                          'median_TWH_rest': float(df.loc[~trickle, 'TWH'].median())}
        print(f'{name}: {trickle.sum()} signature rows ({100 * trickle.mean():.2f}%) in '
              f'{df.loc[trickle, "ID"].nunique()} wells')
        if name == 'nscl-1':
            continue
        stable = df[~trickle]
        ref = summarize(stable)
        out['datasets'][name] = ref

        # Calibration: a 50k-row subsample of the reference rows, and the published rows with trickle rows in
        noise = compare(ref, stable.sample(50_000, random_state=rng.integers(2**31)))
        spike = compare(ref, df)
        print(f'  50k-row subsample: largest CDF gap {noise["worst_cdf"][1]:.4f} ({noise["worst_cdf"][0]}), '
              f'rank correlation {noise["worst_corr"][2]:.4f}')
        print(f'  published rows incl. signature rows: largest CDF gap {spike["worst_cdf"][1]:.4f} '
              f'({spike["worst_cdf"][0]}), rank correlation {spike["worst_corr"][2]:.4f}; '
              f'bounds {MAX_CDF_GAP}, {MAX_CORR_GAP}')

    REFERENCE.write_text(json.dumps(out) + '\n')
    print(f'Wrote {REFERENCE} ({REFERENCE.stat().st_size / 1e3:.0f} kB)')


if __name__ == '__main__':
    main()
