"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 26 February 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Stationary open-loop data (the procedure of manywells-sol-1, specs/sampling.md SMP-28), with the ported sampler
(manywells.sampling) on develop's simulator. From the project root:

    uv run python -m scripts.data_generation.open_loop_stationary.generate_well_data --wells 2000 --samples 500 \
        --seed 1 --configuration v1.0.0 --out data/manywells-sol-v1cfg

The configuration is v1.0.0 (the published model) or develop. The samples are at the stable root, so a dataset in
the v1.0.0 configuration is the stable-root version of sol-1's procedure, not a copy of sol-1 (whose seeds were not
recorded, SMP-31).
"""

import argparse
import multiprocessing
import time
from pathlib import Path

from manywells.configurations import CONFIGURATIONS, V1
from manywells.datasets.io import write_dataset
from manywells.sampling.generate import Settings, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    parser.add_argument('--wells', type=int, default=2000)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--configuration', choices=CONFIGURATIONS, default=V1)
    parser.add_argument('--processes', type=int, default=max(1, multiprocessing.cpu_count() - 2))
    parser.add_argument('--out', type=Path, required=True, help='path of the dataset files, without suffix')
    args = parser.parse_args()

    settings = Settings(seed=args.seed, configuration=args.configuration)
    t0 = time.perf_counter()
    rows, draws, drawn = generate('sol', args.wells, args.samples, settings, args.processes)
    write_dataset(args.out, rows, draws, {'procedure': 'sol (SMP-28)', 'seed': args.seed,
                                          'configuration': args.configuration, 'n_cells': settings.n_cells,
                                          'wells': args.wells, 'samples': args.samples, 'wells_drawn': drawn})
    print(f'{len(draws)} wells of {drawn} drawn, {len(rows)} rows, in {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
