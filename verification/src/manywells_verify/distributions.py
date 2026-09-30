"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The Distributions check (specs/verification.md): a dataset regenerated in the v1-compatibility
configuration must match the reference marginals and rank correlations of the published data.

The reference (verification/data/distribution_reference.json, built by
verification/build/distribution_reference.py) summarizes each published dataset with the rows on
the trickle-root signature removed: for every feature, its values at the percentiles 1..99 and the
reference's empirical CDF at those values; and the Spearman correlation matrix. A candidate passes
if, for every feature, its empirical CDF differs from the reference's by at most MAX_CDF_GAP at those
values (a Kolmogorov-Smirnov distance on a 1% grid), and no rank correlation differs by more than
MAX_CORR_GAP.

    manywells-verify-distributions CANDIDATE --dataset sol-1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ('CHK', 'PBH', 'PWH', 'PDC', 'TBH', 'TWH', 'WGL', 'WGAS', 'WLIQ', 'WOIL', 'WWAT', 'WTOT',
            'QGL', 'QGAS', 'QLIQ', 'QOIL', 'QWAT', 'QTOT', 'FGAS', 'FOIL', 'FWAT')
PERCENTILES = np.arange(1, 100)
TRICKLE_DP = 0.01        # bar: rows with PWH - PDC below this are on the trickle-root signature
MAX_CDF_GAP = 0.02       # largest allowed empirical-CDF difference per feature
MAX_CORR_GAP = 0.05      # largest allowed rank-correlation difference
REFERENCE = Path(__file__).resolve().parents[2] / 'data' / 'distribution_reference.json'


def trickle_signature(df: pd.DataFrame) -> pd.Series:
    """Rows whose wellhead pressure is within TRICKLE_DP of the downstream pressure."""
    return (df['PWH'] - df['PDC']) < TRICKLE_DP


def summarize(df: pd.DataFrame) -> dict:
    """Percentile values, the empirical CDF at them, and the Spearman correlation matrix."""
    features = {}
    for f in FEATURES:
        x = np.sort(df[f].to_numpy(dtype=float))
        q = np.percentile(x, PERCENTILES)
        features[f] = {'values': q.tolist(), 'cdf': (np.searchsorted(x, q, side='right') / len(x)).tolist()}
    corr = df[list(FEATURES)].corr(method='spearman').fillna(0.0)
    return {'rows': int(len(df)), 'features': features, 'spearman': corr.to_numpy().round(6).tolist()}


def compare(reference: dict, df: pd.DataFrame) -> dict:
    """CDF gap per feature and the largest rank-correlation gap of a candidate against a reference summary."""
    gaps = {}
    for f in FEATURES:
        x = np.sort(df[f].to_numpy(dtype=float))
        q, F_ref = np.array(reference['features'][f]['values']), np.array(reference['features'][f]['cdf'])
        gaps[f] = float(np.max(np.abs(np.searchsorted(x, q, side='right') / len(x) - F_ref)))
    corr = df[list(FEATURES)].corr(method='spearman').fillna(0.0).to_numpy()
    diff = np.abs(corr - np.array(reference['spearman']))
    i, j = np.unravel_index(np.argmax(diff), diff.shape)
    worst_cdf = max(gaps, key=gaps.get)
    passed = gaps[worst_cdf] <= MAX_CDF_GAP and diff[i, j] <= MAX_CORR_GAP
    return {'passed': bool(passed), 'cdf_gaps': gaps, 'worst_cdf': (worst_cdf, gaps[worst_cdf]),
            'worst_corr': (FEATURES[i], FEATURES[j], float(diff[i, j]))}


def report(result: dict, dataset: str, rows: int) -> str:
    f, gap = result['worst_cdf']
    a, b, cgap = result['worst_corr']
    lines = [f'## Distributions against {dataset} (stable-root reference)', '',
             f'**Verdict: {"PASS" if result["passed"] else "FAIL"}.** {rows} candidate rows.', '',
             f'- Largest CDF gap: {gap:.3f} ({f}); bound {MAX_CDF_GAP}.',
             f'- Largest rank-correlation gap: {cgap:.3f} ({a}, {b}); bound {MAX_CORR_GAP}.']
    over = [f'{k} {v:.3f}' for k, v in sorted(result['cdf_gaps'].items(), key=lambda kv: -kv[1]) if v > MAX_CDF_GAP]
    if over:
        lines.append('- Features over the bound: ' + ', '.join(over) + '.')
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='manywells-verify-distributions',
                                     description='Compare a regenerated dataset with the stable-root reference.')
    parser.add_argument('candidate', type=Path, help='dataset rows (parquet or csv) with the ManyWells features')
    parser.add_argument('--dataset', required=True, help='reference dataset, e.g. sol-1 or nsol-1')
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    args = parser.parse_args(argv)
    reference = json.loads(args.reference.read_text())['datasets'][args.dataset]
    df = pd.read_parquet(args.candidate) if args.candidate.suffix == '.parquet' else pd.read_csv(args.candidate)
    result = compare(reference, df)
    print(report(result, args.dataset, len(df)))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
