"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

A v2 dataset, manywells-sol-2 or manywells-nsol-2 (specs/sampling.md, SMP-50 to SMP-117), end to end: the table
fluids' build (manywells.sampling.v2.table_wells and table_build.py, SMP-112 to SMP-116), the run over the wells
(manywells.sampling.v2.run, SMP-109 and SMP-110) and the assembly of the dataset and its logs
(manywells.sampling.v2.assemble, SMP-108 and SMP-110). From the project root, with the sandbox off (the pools need
Unix sockets, and NeqSim's environment uv's cache), under a cap of 8 GB of address space, which NeqSim's JVMs take;
the run's processes cap themselves at --memory-cap:

    (ulimit -v 8000000; OMP_NUM_THREADS=1 uv run python -m scripts.data_generation.v2.generate_dataset sol \
        --n 100 --seed 2026 --processes 6 --out data/pilot-sol-2 --allow-dirty)
    (... nsol --n 100 --seed 2027 --processes 6 --out data/pilot-nsol-2 --allow-dirty)

OUT holds tables/ (the table fluids' build), run/ (the run's chunks and settings), and the dataset's folder
manywells-<kind>-2/ with its companion folder of logs manywells-<kind>-2-logs/. Every step resumes: a stopped run of
the script carries on where it stopped. The table wells are drawn for a first guess of the draw indices the run needs
(--draws, by default 1.5 n rounded up to the chunks); where the run needs more, the tables are extended and the run
carries on.

NeqSim's build runs in an environment of its own (table_build.py's docstring), twice: the fluids, and, once their GOR
axes are written, the tables. --neqsim run (the default) launches it, each JVM's process capped at 8 GB of address
space; --neqsim print prints the command and stops, so that it can be run by hand before this script is run again.
A dataset to publish is written from a clean working tree; --allow-dirty writes a pilot from any.
"""

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from manywells.sampling.v2 import table_build
from manywells.sampling.v2.assemble import write_dataset
from manywells.sampling.v2.run import CHUNK, OutOfMemory, run
from manywells.sampling.v2.table_wells import write_axes, write_table_wells
from manywells.sampling.v2.tables import TableMismatch

NEQSIM_CAP = 8_000_000_000  # bytes of address space per process of NeqSim's build
JAVA_OPTIONS = '-Xmx1g -XX:+UseSerialGC -XX:ActiveProcessorCount=2 -XX:ReservedCodeCacheSize=128m'
JAVA_OPTIONS += ' -XX:MaxMetaspaceSize=256m'


def neqsim_command(tables: Path, processes: int) -> list:
    """table_build.py's command in NeqSim's environment, by its path."""
    with_ = ['--with', f'neqsim=={table_build.NEQSIM_VERSION}', '--with', 'numpy', '--with', 'pandas']
    build = ['python', table_build.__file__, str(tables), '--processes', str(processes)]
    return ['uv', 'run', '--no-project', '--python', '3.12', *with_, *build]


def pending(tables: Path) -> str | None:
    """The build's step that NeqSim has still to make: 'fluids' for a table well without its fluid, 'tables' for one
    whose fluid failed or has an axis and has no manifest yet; None when every table well has its manifest."""
    spec = json.loads((tables / 'wells.json').read_text())
    axes = json.loads((tables / 'axes.json').read_text()) if (tables / 'axes.json').exists() else {}
    step = None
    for w in spec['wells']:
        k = w['well']
        fluid = tables / 'fluids' / f'k{k:06d}.json'
        if not fluid.exists():
            return 'fluids'
        if (tables / f'k{k:06d}.json').exists():
            continue
        if str(k) in axes or json.loads(fluid.read_text())['status'] != 'ok':
            step = 'tables'
    return step


def build_with_neqsim(tables: Path, step: str, mode: str, processes: int):
    """NeqSim's build of the table wells' fluids or tables: launched, or its command printed (and the script stops)."""
    cmd = neqsim_command(tables, processes)
    env = {key: v for key, v in os.environ.items() if key != 'VIRTUAL_ENV'}
    env.update(MALLOC_ARENA_MAX='2', JAVA_TOOL_OPTIONS=JAVA_OPTIONS)
    if mode == 'print':
        print(f'NeqSim has the {step} to build. Run, from the project root, then this script again:\n')
        print(f'    (ulimit -v {NEQSIM_CAP // 1000}; export MALLOC_ARENA_MAX=2 JAVA_TOOL_OPTIONS="{JAVA_OPTIONS}"; \\')
        print(f'        {" ".join(cmd)})')
        sys.exit(2)
    print(f'NeqSim: the {step} of {tables}', flush=True)

    def cap():
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = NEQSIM_CAP if hard == resource.RLIM_INFINITY else min(NEQSIM_CAP, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))

    subprocess.run(cmd, env=env, preexec_fn=cap, check=True)


def tables_for(kind, seed, tables: Path, n_draws: int, mode: str, neqsim_processes: int):
    """The four steps of the table fluids' build (table_wells.py's docstring) for draw indices 0 to n_draws - 1."""
    spec = write_table_wells(tables, kind, seed, n_draws)
    print(f'{len(spec["wells"])} table wells among draw indices 0 to {spec["wells_range"][1] - 1}', flush=True)
    if pending(tables) == 'fluids':
        build_with_neqsim(tables, 'fluids', mode, neqsim_processes)
    axes = write_axes(tables)
    print(f'{len(axes)} GOR axes, {sum("error" in a for a in axes.values())} failed', flush=True)
    if pending(tables) is not None:
        build_with_neqsim(tables, 'tables', mode, neqsim_processes)
    if pending(tables) is not None:
        raise RuntimeError(f'NeqSim left table wells of {tables} without their files')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[2])
    parser.add_argument('kind', choices=('sol', 'nsol'))
    parser.add_argument('--n', type=int, required=True, help='the accepted wells of the dataset')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--processes', type=int, default=6, help="the run's pool")
    parser.add_argument('--memory-cap', type=float, default=4.0, help='GB of address space per process of the run')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--draws', type=int, help='the draw indices to build tables for first')
    parser.add_argument('--neqsim', choices=('run', 'print'), default='run')
    parser.add_argument('--neqsim-processes', type=int, default=4, help='JVMs of the build, about 1 GB each')
    parser.add_argument('--allow-dirty', action='store_true', help='write the dataset from code with changes')
    args = parser.parse_args()
    # The run's processes import numpy after this: with a thread of OpenBLAS's per core a well takes about 1.9 GB
    # more address space, and a table well passes 3 GB (run.py's docstring)
    os.environ.setdefault('OMP_NUM_THREADS', '1')

    out, t0 = args.out, time.perf_counter()
    tables = out / 'tables'
    n_draws = args.draws or CHUNK * math.ceil(1.5 * args.n / CHUNK)
    if (tables / 'wells.json').exists():
        n_draws = max(n_draws, json.loads((tables / 'wells.json').read_text())['wells_range'][1])
    while True:
        tables_for(args.kind, args.seed, tables, n_draws, args.neqsim, args.neqsim_processes)
        cap = int(args.memory_cap * 1e9)
        try:
            state = run(args.kind, args.n, out / 'run', args.seed, args.processes, tables, memory_cap=cap)
        except (OutOfMemory, TableMismatch) as e:
            sys.exit(f'the run stopped: {e}')
        if state['complete']:
            break
        rate = max(state['accepted'], 1) / max(state['drawn'], 1)
        n_draws += CHUNK * math.ceil(1.1 * (args.n - state['accepted']) / rate / CHUNK)
        print(f'the run needs more draw indices: tables for 0 to {n_draws - 1}', flush=True)
    meta = write_dataset(out / 'run', out, args.n, allow_dirty=args.allow_dirty)
    counts = meta['counts']
    print(f'{counts["wells_accepted"]} wells of {counts["wells_drawn"]} drawn: {counts["outcomes"]}')
    print(f'{meta["files"]["samples.parquet"]["rows"]} samples in {out / meta["dataset"]}')
    print(f'{time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
