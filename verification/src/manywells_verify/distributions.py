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

The candidate's rows are weighted so that its wells mix as the reference's do (decided by Bjarne,
2026-10-01): a row of well k weighs n_ref(k) / n_cand(k), its well's rows in the reference over its
rows in the candidate, and a well with no reference rows weighs nothing. The reference weights each
well by its published rows, which the trickle signature reduced in some wells, while a regeneration
draws the same number of samples per well; unweighted, even the reference's own rows drawn that way
fail. The candidate therefore needs the reference's well IDs, in an ID column.

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
    """Percentile values, the empirical CDF at them, the Spearman correlation matrix, and the rows per well."""
    features = {}
    for f in FEATURES:
        x = np.sort(df[f].to_numpy(dtype=float))
        q = np.percentile(x, PERCENTILES)
        features[f] = {'values': q.tolist(), 'cdf': (np.searchsorted(x, q, side='right') / len(x)).tolist()}
    corr = df[list(FEATURES)].corr(method='spearman').fillna(0.0)
    out = {'rows': int(len(df)), 'features': features, 'spearman': corr.to_numpy().round(6).tolist()}
    if 'ID' in df:
        out['well_rows'] = {str(k): int(v) for k, v in df.groupby('ID').size().items()}
    return out


def well_weights(reference: dict, df: pd.DataFrame) -> np.ndarray:
    """Row weights that mix the candidate's wells as the reference's: n_ref(well) / n_cand(well) per row."""
    if 'ID' not in df:
        raise ValueError('the candidate rows need an ID column with the reference\'s well IDs')
    n_ref = df['ID'].map(lambda k: reference['well_rows'].get(str(int(k)), 0)).to_numpy(dtype=float)
    n_cand = df.groupby('ID')['ID'].transform('size').to_numpy(dtype=float)
    return n_ref / n_cand


def weighted_cdf(x, w, q) -> np.ndarray:
    """The weighted empirical CDF of x at the points q (with equal weights, the empirical CDF)."""
    order = np.argsort(x, kind='mergesort')
    xs, cw = x[order], np.cumsum(w[order])
    k = np.searchsorted(xs, q, side='right')
    return np.where(k > 0, cw[np.maximum(k - 1, 0)], 0.0) / cw[-1]


def weighted_ranks(x, w) -> np.ndarray:
    """Weighted mid-ranks: the weight below each value plus half the weight at it (ties share one rank)."""
    order = np.argsort(x, kind='mergesort')
    xs, ws = x[order], w[order]
    _, start, counts = np.unique(xs, return_index=True, return_counts=True)
    cw = np.concatenate([[0.0], np.cumsum(ws)])
    mid = cw[start] + (cw[start + counts] - cw[start]) / 2
    ranks = np.empty(len(x))
    ranks[order] = np.repeat(mid, counts)
    return ranks


def weighted_spearman(df: pd.DataFrame, w) -> np.ndarray:
    """Spearman's rank correlation with row weights: the weighted Pearson correlation of weighted mid-ranks."""
    R = np.column_stack([weighted_ranks(df[f].to_numpy(dtype=float), w) for f in FEATURES])
    R = R - (w @ R) / w.sum()
    C = (R * w[:, None]).T @ R
    sd = np.sqrt(np.diag(C))
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = C / np.outer(sd, sd)
    return np.nan_to_num(corr, nan=0.0)


def compare(reference: dict, df: pd.DataFrame, weights=None) -> dict:
    """
    CDF gap per feature and the largest rank-correlation gap of a candidate against a reference summary,
    with optional row weights (well_weights); without them every row weighs the same.
    """
    w = np.ones(len(df)) if weights is None else np.asarray(weights, dtype=float)
    keep = w > 0
    df, w = df[keep], w[keep]
    gaps = {}
    for f in FEATURES:
        q, F_ref = np.array(reference['features'][f]['values']), np.array(reference['features'][f]['cdf'])
        gaps[f] = float(np.max(np.abs(weighted_cdf(df[f].to_numpy(dtype=float), w, q) - F_ref)))
    corr = weighted_spearman(df, w)
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
    result = compare(reference, df, well_weights(reference, df))
    print(report(result, args.dataset, len(df)))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
