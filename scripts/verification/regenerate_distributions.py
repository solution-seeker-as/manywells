"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Samples regenerated at the stable root for the published sol-1 wells, in the v1.0.0 configuration, and the
verifier's Distributions check on them (specs/verification.md, Distributions; plans/manywells-v2-plan.md, Step 7
item 7). From the project root:

    uv run python -m scripts.verification.regenerate_distributions --config <manywells-sol-1_config.zip> \
        --samples 8 --out data/regenerated-sol-1.parquet [--backend rust]

The wells are the published ones (not newly sampled), because the distribution reference is the published data
(verification/build/distribution_reference.py). Each well's samples are drawn by the ported sampler (SMP-18 to
SMP-22, seeded by SMP-31) and solved by develop's simulator from the well's operating point at u = 0.5, as
SMP-28's generator started them; failed solves and rows with w_m < 0.1 kg/s are dropped, as there. The well-level
filters of SMP-28 step 3 are not applied: the published wells passed them. Without --config, the config is fetched
from the public Hugging Face dataset solution-seeker-as/manywells. The draws do not depend on the backend or on
the solves, so each row's ID and sample index k identify it across backends.
"""

import argparse
import multiprocessing
import time
from pathlib import Path

import pandas as pd

from manywells_verify.distributions import REFERENCE, compare, report, well_weights

from manywells.datasets.rows import sample_row
from manywells.datasets.schema import FEATURES
from manywells.sampling.conditions import nominal_conditions, sample_conditions
from manywells.sampling.generate import Settings, solve
from manywells.sampling.wells import rng_for, well_from_config

SEED = 20261001


def regenerate(task):
    """The regenerated rows of one published well, and its counts."""
    row, n_samples, settings = task
    draw = well_from_config(row)
    first = solve(draw, draw.fractions, nominal_conditions(draw), settings)
    rows, failed, dropped = [], 0, 0
    if first is None:
        return rows, {'ID': row['ID'], 'first': False, 'failed': n_samples, 'dropped': 0}
    for k in range(n_samples):
        bc, fractions = sample_conditions(draw, rng_for(settings.seed, row['ID'], k, 'sample'))
        op = solve(draw, fractions, bc, settings, x_guess=first.x)
        if op is None:
            failed += 1
            continue
        r = sample_row(op, draw, fractions, bc)
        if r['WTOT'] < 0.1:
            dropped += 1
            continue
        rows.append(r | {'ID': row['ID'], 'k': k})
    return rows, {'ID': row['ID'], 'first': True, 'failed': failed, 'dropped': dropped}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    parser.add_argument('--config', type=Path, help='manywells-sol-1_config.zip (default: from Hugging Face)')
    parser.add_argument('--samples', type=int, default=8, help='samples per well')
    parser.add_argument('--wells', type=int, help='only the first WELLS wells')
    parser.add_argument('--backend', choices=('casadi', 'rust'), default='casadi')
    parser.add_argument('--processes', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--out', type=Path, required=True, help='parquet file for the regenerated rows')
    args = parser.parse_args()

    config = args.config
    if config is None:
        from huggingface_hub import hf_hub_download
        config = hf_hub_download('solution-seeker-as/manywells', 'data/manywells-sol-1_config.zip', repo_type='dataset')
    wells = pd.read_csv(config, compression='zip').to_dict('records')[:args.wells]
    settings = Settings(seed=SEED, backend=args.backend)

    t0 = time.perf_counter()
    with multiprocessing.Pool(args.processes) as pool:
        results = pool.map(regenerate, [(w, args.samples, settings) for w in wells], chunksize=1)
    rows = pd.DataFrame([r for rs, _ in results for r in rs], columns=list(FEATURES) + ['ID', 'k'])
    counts = pd.DataFrame([c for _, c in results])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(args.out, index=False)
    print(f'{len(rows)} rows from {len(wells)} wells in {time.perf_counter() - t0:.0f} s; '
          f'{(~counts["first"]).sum()} wells without an operating point at u = 0.5, '
          f'{counts["failed"].sum()} failed and {counts["dropped"].sum()} dropped samples')

    import json
    reference = json.loads(REFERENCE.read_text())['datasets']['sol-1']
    print(report(compare(reference, rows, well_weights(reference, rows)), 'sol-1', len(rows)))


if __name__ == '__main__':
    main()
