"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 8 August 2024
Erlend Lundby, erlend@solutionseeker.no

Non-stationary open-loop data (the procedure of manywells-nsol-1, specs/sampling.md SMP-29), with the ported
sampler (manywells.sampling) on develop's simulator. From the project root:

    uv run python -m scripts.data_generation.open_loop_nonstationary.generate_open_loop_nonstationary_well_data \
        --wells 2000 --samples 500 --seed 1 --configuration v1.0.0 --out data/manywells-nsol-v1cfg

The samples are at the stable root (`manywells.sampling.generate`).
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
    rows, draws, drawn = generate('nsol', args.wells, args.samples, settings, args.processes)
    write_dataset(args.out, rows, draws, {'procedure': 'nsol (SMP-29)', 'seed': args.seed,
                                          'configuration': args.configuration, 'n_cells': settings.n_cells,
                                          'wells': args.wells, 'samples': args.samples, 'wells_drawn': drawn})
    print(f'{len(draws)} wells of {drawn} drawn, {len(rows)} rows, in {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
