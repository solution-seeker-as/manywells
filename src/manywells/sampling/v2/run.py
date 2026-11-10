"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The run of the v2 generator over many wells (specs/sampling.md, SMP-109 and SMP-110): wells k = 0, 1, 2, ... are
generated (generate.generate_sol2_well or generate_nsol2_well), one task per well in a pool of processes, until the
first n wells it accepts are known. The same function runs a pilot of a hundred wells and a dataset of 10,000.

The draw indices are run in chunks of CHUNK. A well's result is kept in RUN/wells/kK.pkl as soon as it is solved, so
that a stopped run loses only the wells in progress; once every well of a chunk is solved, the chunk's results go to
one file, RUN/chunks/chunk-C.pkl, written to a temporary name and renamed, and the wells' files are removed. A chunk
file holds two pickles: its header (the run's settings, the chunk, its first draw index and size, each well's outcome
and the CPU time), then the list of its wells' results in draw order. RUN/settings.json holds the run's settings: the
dataset and seed, the configuration, the grid, the backend, the limits, the floor and the best rate's lift-gas rates,
scan points and water cut, the slots per well, the chunk size, the folder of tables, the lock file's hash and the
code's record (datasets.v2.code_record). A restart checks them, and stops on any difference, so that every chunk of a
run is made by the same code with the same settings; a well's own file holds their hash, so that a file kept from
another run's settings is refused too. The assembly takes the dataset's records from them (assemble.write_dataset).

A crash of a well's generation, an exception or a panic of the Rust core that the generation does not catch, is
logged as the well's outcome, `crash`, with its traceback, and the run carries on. Each result records its wall time
`seconds` and its CPU time `cpu_seconds`.

A pool's processes may each be capped (memory_cap, bytes of address space), below the cap of the process that runs
the pool. With OMP_NUM_THREADS=1 a well takes about 0.5 GB of address space at its peak and a table well 1.6 GB (on
one process, and the same in the pool; 2026-11-10); without it numpy's OpenBLAS starts a thread per core, which adds
about 1.9 GB more. A well that runs out of memory at its process's cap (a MemoryError) depends on the process, not
on its draws, so it is no outcome: it is not kept, and once the other wells of its batch are done it is solved again
in a fresh process of its own, up to MEMORY_RETRIES times; if it runs out of memory there too, the run stops with
OutOfMemory, which names the well, and a rerun resumes from it.

A table well takes its table from the folder that table_wells.py and table_build.py made for the dataset and seed;
the run runs no chunk beyond the draw indices that the folder covers (its wells.json's `wells_range`), since a table
well there would have no table, and returns incomplete, so that the tables are extended first. It starts only where
the build is complete, every table well of wells.json with its manifest, and it stops with TableMismatch where the
folder's record of a well or the well's GOR axis is not what the run's draws give (generate._table), as after a rule
changed since the build: such a folder is built again, not used. Without a folder of tables every table well is
`table_failed`.

From the repository root, with the sandbox off (the pool needs Unix sockets), each process capped:

    (ulimit -v 3000000; OMP_NUM_THREADS=1 .venv/bin/python -c "from manywells.sampling.v2.run import run; \\
        run('sol', 100, 'OUT/run', seed=2026, processes=6, tables='OUT/tables')")

scripts/data_generation/v2/generate_dataset.py runs the whole pipeline: the tables, the run and the assembly.
"""

import collections
import hashlib
import json
import math
import multiprocessing
import pickle
import resource
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from manywells.configurations import V2
from manywells.datasets.v2 import DATASETS, code_record, lock_hash
from manywells.sampling.v2 import generate
from manywells.sampling.v2.life import WLR_MAX
from manywells.sampling.v2.mapping import N_CELLS
from manywells.sampling.v2.tables import TableMismatch

CHUNK = 100  # Draw indices per chunk  # spec: SMP-110
GENERATE = {'sol': generate.generate_sol2_well, 'nsol': generate.generate_nsol2_well}
_FIRST_RATE = 1.0  # The share of wells accepted that a run assumes before its first chunk: it does not overshoot
_LEAST_RATE = 0.05  # The least share it plans with
_TASKS_PER_PROCESS = 100  # A pool's process is renewed after this many wells, so that no leak grows over a run
MEMORY_RETRIES = 2  # The fresh processes in which a well that ran out of memory is solved again


class SettingsMismatch(ValueError):
    """A run's folder holds the chunks of another run: other settings or another code."""


class OutOfMemory(RuntimeError):
    """A well ran out of memory at its process's cap, in the run's pool and in every fresh process it was given."""


# --- One well ---------------------------------------------------------------------------------------------------------
def run_well(kind: str, seed: int, k: int, tables=None) -> dict:  # spec: SMP-110
    """
    Well k's result (generate's module docstring), with `seconds` its wall time and `cpu_seconds` its CPU time. A
    crash of the generation, an exception or a panic of the Rust core, is the well's outcome, `crash`, with the
    traceback as its error. A MemoryError is not: it is raised, and the run solves the well again.

    :param kind: 'sol' or 'nsol'
    :param seed: The dataset's seed
    :param k: The well's draw index
    :param tables: The folder of the dataset's tables, or None
    """
    t0, cpu0 = time.perf_counter(), time.process_time()
    try:
        res = GENERATE[kind](seed, k, tables)
    except (MemoryError, TableMismatch):
        raise  # The process's, or the folder of tables': not the well's
    except BaseException as e:  # noqa: BLE001 - a crash is logged as the well's outcome, and the run carries on
        if not (isinstance(e, Exception) or generate._is_panic(e)):
            raise  # An interrupt or an exit stops the run
        info = generate._info(k, outcome='crash', error=_strip_home(traceback.format_exc())[-2000:])
        if kind == 'nsol':
            info.update(scan=[], T_end=math.nan, p_change=math.nan)
        res = dict(well=None, stations=None, table=None, nsol=None, info=info, samples=[], log=[])
    res['seconds'], res['cpu_seconds'] = time.perf_counter() - t0, time.process_time() - cpu0
    return res


def _strip_home(text):
    """text, a traceback say, with the home directory's path written as ~, so that no local path is published."""
    return text if text is None else text.replace(str(Path.home()), '~')


def _dump(obj, path, *more):
    """obj (and the objects more) pickled to path, through a temporary name, so that a stopped write leaves no file."""
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'wb') as f:
        for o in (obj, *more):
            pickle.dump(o, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(path)


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _well_path(run_dir, k) -> Path:
    return Path(run_dir) / 'wells' / f'k{k:06d}.pkl'


def _settings_hash(settings) -> str:
    """The SHA-256 of a run's settings, which each well's own file holds."""
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def _task(task) -> tuple:
    """A pool's task: well k's result, solved, or read from its file where a stopped run with the same settings left
    it. Returns (k, its outcome, its wall time (s), whether it was read); the outcome is None, and nothing is kept,
    where the well ran out of memory."""
    kind, seed, k, run_dir, tables, settings_hash = task
    path = _well_path(run_dir, k)
    if path.exists():
        res = _load(path)
        if res.get('settings_sha256') != settings_hash:
            raise SettingsMismatch(f"{path} was made with other settings than the run's: remove it, or run with them")
        return k, res['info']['outcome'], res['seconds'], True
    t0 = time.perf_counter()
    try:
        res = run_well(kind, seed, k, tables)
    except MemoryError:
        return k, None, time.perf_counter() - t0, False
    _dump(dict(res, settings_sha256=settings_hash), path)
    return k, res['info']['outcome'], res['seconds'], False


def _fresh(task, memory_cap) -> tuple:
    """_task in a fresh process of its own, capped at memory_cap bytes, which ends with it."""
    ctx = multiprocessing.get_context('forkserver')
    with ProcessPoolExecutor(1, mp_context=ctx, initializer=_cap, initargs=(memory_cap,), max_tasks_per_child=1) as p:
        return p.submit(_task, task).result()


def _retry(task, memory_cap, log) -> tuple:
    """A well that ran out of memory, solved again in fresh processes (_fresh), up to MEMORY_RETRIES times. Returns
    _task's result; raises OutOfMemory if the well runs out of memory every time."""
    k = task[2]
    for i in range(MEMORY_RETRIES):
        log(f'well {k}: out of memory; solved again in a fresh process ({i + 1} of {MEMORY_RETRIES})')
        out = _fresh(task, memory_cap)
        if out[1] is not None:
            return out
    cap = 'no cap' if memory_cap is None else f'a cap of {memory_cap / 1e9:.2f} GB'
    raise OutOfMemory(
        f'well {k} ran out of memory, with {cap}, in the pool and in {MEMORY_RETRIES} fresh processes: it is not '
        f'kept. Raise the cap, or free memory, and run again: the run resumes from it.'
    )


# --- Settings and chunks ----------------------------------------------------------------------------------------------
def run_settings(kind: str, seed: int, tables=None, chunk_size: int = CHUNK) -> dict:  # spec: SMP-110
    """The settings that a run's chunks share (the module's docstring), as they read back from JSON."""
    if kind not in GENERATE:
        raise ValueError(f"kind must be 'sol' or 'nsol', got {kind!r}")
    settings = dict(
        dataset=DATASETS[kind],
        kind=kind,
        seed=seed,
        configuration=V2,
        n_cells=N_CELLS,
        backend='rust',
        n_slots=generate.N_SLOTS,
        limits=dict(
            slot=generate.SLOT_LIMIT, redraws=generate.REDRAW_LIMIT, solver_failures=generate.SOLVER_FAILURE_LIMIT
        ),
        floor=generate.FLOOR,
        floor_margin=generate.FLOOR_MARGIN,
        lift_fractions=list(generate._BEST_RATE_LIFT),
        scan_points=generate._N_SCAN if kind == 'nsol' else None,
        wlr_max=WLR_MAX,
        chunk_size=chunk_size,
        tables=None if tables is None else str(Path(tables).resolve()),
        lock_sha256=lock_hash(),
        code=code_record(),
    )
    return json.loads(json.dumps(settings))


def _check_settings(run_dir, settings):
    """Writes RUN/settings.json, or checks that the run's folder holds a run with these settings."""
    path = Path(run_dir) / 'settings.json'
    if not path.exists():
        path.write_text(json.dumps(settings, indent=1))
        return
    old = json.loads(path.read_text())
    if old != settings:
        diff = sorted(key for key in set(old) | set(settings) if old.get(key) != settings.get(key))
        raise SettingsMismatch(f'{run_dir} holds a run with other settings: {", ".join(diff)} differ')


def _chunk_path(run_dir, c) -> Path:
    return Path(run_dir) / 'chunks' / f'chunk-{c:06d}.pkl'


def _pack(run_dir, settings, c):
    """Chunk c's wells' results into its file, and their own files removed. Returns the chunk's header."""
    size = settings['chunk_size']
    ks = range(c * size, (c + 1) * size)
    results = [_load(_well_path(run_dir, k)) for k in ks]
    for res in results:  # Each was checked against the settings when it was solved or kept
        res.pop('settings_sha256', None)
    header = dict(settings=settings, chunk=c, first=ks[0], size=size)
    header.update(outcomes=[r['info']['outcome'] for r in results], cpu_seconds=sum(r['cpu_seconds'] for r in results))
    _dump(header, _chunk_path(run_dir, c), results)
    for k in ks:
        _well_path(run_dir, k).unlink()
    return header


def chunk_headers(run_dir) -> dict:
    """The headers of a run's chunks, {chunk: header}, each checked against the run's settings."""
    run_dir = Path(run_dir)
    settings = json.loads((run_dir / 'settings.json').read_text())
    out = {}
    for path in sorted((run_dir / 'chunks').glob('chunk-*.pkl')):
        header = _load(path)
        if header['settings'] != settings:
            raise SettingsMismatch(f'{path} was made with other settings than {run_dir}/settings.json')
        out[header['chunk']] = header
    return out


def read_results(run_dir):
    """The results of a run's wells in draw order, chunk by chunk from chunk 0 up to the first chunk not yet made."""
    headers = chunk_headers(run_dir)
    c = 0
    while c in headers:
        with open(_chunk_path(run_dir, c), 'rb') as f:
            pickle.load(f)
            yield from pickle.load(f)
        c += 1


def _tables_limit(tables, kind, seed):  # spec: SMP-116
    """The draw indices that the folder of tables covers, 0 to the returned number; None without tables. The folder
    must be the dataset's and seed's, and its build complete: every table well of its wells.json with its manifest."""
    if tables is None:
        return None
    spec = json.loads((Path(tables) / 'wells.json').read_text())
    if (spec['kind'], spec['seed']) != (kind, seed):
        raise ValueError(f'{tables} holds the tables of {spec["kind"]}, seed {spec["seed"]}, not {kind}, seed {seed}')
    missing = [w['well'] for w in spec['wells'] if not (Path(tables) / f'k{w["well"]:06d}.json').exists()]
    if missing:
        raise ValueError(
            f'the build of {tables} is not complete: {len(missing)} table wells have no manifest (draw indices '
            f'{", ".join(map(str, missing[:5]))}{", ..." if len(missing) > 5 else ""}); run table_build.py'
        )
    return spec['wells_range'][1]


# --- The run ----------------------------------------------------------------------------------------------------------
def _cap(memory_cap):
    """A pool process's address space capped at memory_cap bytes, or at its hard limit if that is lower."""
    if memory_cap is not None:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        cap = memory_cap if hard == resource.RLIM_INFINITY else min(memory_cap, hard)
        resource.setrlimit(resource.RLIMIT_AS, (cap, hard))


def _imap(fn, tasks, processes, memory_cap=None):
    """fn's results over the tasks, as they finish: in a pool of processes, each capped at memory_cap bytes, or in
    this process for one. A worker that dies (a process killed at its memory cap, say) stops the run with
    BrokenProcessPool, and a rerun resumes it."""
    if processes <= 1:
        yield from map(fn, tasks)
        return
    ctx = multiprocessing.get_context('forkserver')
    pool = ProcessPoolExecutor(
        processes, mp_context=ctx, initializer=_cap, initargs=(memory_cap,), max_tasks_per_child=_TASKS_PER_PROCESS
    )
    try:
        for future in as_completed([pool.submit(fn, t) for t in tasks]):
            yield future.result()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()


def _summary(headers, n_wells) -> dict:
    """The run's state from its chunks' headers: the chunks made from 0 on without a gap, the wells they drew and
    accepted up to the n-th accepted, and whether the run is complete."""
    accepted = drawn = c = 0
    outcomes = collections.Counter()
    while c in headers and accepted < n_wells:
        for outcome in headers[c]['outcomes']:
            if accepted < n_wells:
                drawn += 1
                outcomes[outcome] += 1
                accepted += outcome == 'accepted'
        c += 1
    return dict(complete=accepted >= n_wells, accepted=accepted, drawn=drawn, next_chunk=c, outcomes=dict(outcomes))


def run(
    kind: str,
    n_wells: int,
    out,
    seed: int,
    processes: int = 6,
    tables=None,
    chunk_size: int = CHUNK,
    memory_cap=None,
    log=print,
) -> dict:  # spec: SMP-109, SMP-110
    """
    Generate wells k = 0, 1, 2, ... of a dataset in chunks of draw indices until its first n_wells accepted wells are
    known, each well's result kept on disk (the module's docstring). A stopped run resumes where it stopped; a run
    with another n_wells resumes the same chunks.

    :param kind: 'sol' or 'nsol'
    :param n_wells: The accepted wells the dataset needs
    :param out: The run's folder
    :param seed: The dataset's seed
    :param processes: The pool's processes; 1 runs the wells in this process
    :param tables: The folder of the dataset's tables (table_wells.py, table_build.py), or None
    :param chunk_size: Draw indices per chunk
    :param memory_cap: Each pool process's cap of address space (bytes), or None
    :param log: Called with each line of progress
    :return: A dict of `complete` (whether the first n_wells accepted wells are known), `accepted` and `drawn` (the
        wells up to the n-th accepted, or all of the chunks made), `next_chunk` (the first chunk not made, after those
        up to the n-th accepted), `outcomes` (the drawn wells' outcomes, counted), `limit` (the draw indices that the
        tables cover, or None) and `seconds` (the call's wall time)
    :raises OutOfMemory: where a well runs out of memory in the pool and in each of its fresh processes, once the
        batch's other wells are kept
    :raises TableMismatch: where the folder of tables was not built from the run's draws (generate._table)
    :raises ValueError: where the folder of tables is another dataset's, or its build is not complete
    """
    run_dir = Path(out)
    (run_dir / 'wells').mkdir(parents=True, exist_ok=True)
    (run_dir / 'chunks').mkdir(exist_ok=True)
    settings = run_settings(kind, seed, tables, chunk_size)
    limit = _tables_limit(tables, kind, seed)
    if tables is None:
        log('no tables: every volatile oil and gas condensate is table_failed')
    _check_settings(run_dir, settings)
    headers = chunk_headers(run_dir)
    t0 = time.perf_counter()
    while True:
        state = _summary(headers, n_wells)
        if state['complete']:
            break
        done = [h['outcomes'] for h in headers.values()]
        n_done = sum(len(o) for o in done)
        rate = sum(o.count('accepted') for o in done) / n_done if n_done else _FIRST_RATE
        n_more = math.ceil((n_wells - state['accepted']) / max(rate, _LEAST_RATE) / chunk_size)
        batch, c = [], state['next_chunk']
        while len(batch) < n_more:
            if c not in headers:
                batch.append(c)
            c += 1
        batch = [c for c in batch if limit is None or (c + 1) * chunk_size <= limit]
        if not batch:
            log(f'the tables cover draw indices below {limit}: chunk {state["next_chunk"]} needs more')
            break
        log(f'chunks {batch[0]} to {batch[-1]}: {len(batch) * chunk_size} wells')
        task = (kind, seed, str(run_dir), settings['tables'], _settings_hash(settings))
        tasks = {k: (*task[:2], k, *task[2:]) for c in batch for k in _ks(c, chunk_size)}
        pending = {c: set(_ks(c, chunk_size)) for c in batch}

        def done(k, outcome, sec, read):
            c = k // chunk_size
            pending[c].discard(k)
            log(f'well {k}: {outcome} in {sec:.1f} s' + (' (kept)' if read else ''))
            if not pending[c]:
                headers[c] = _pack(run_dir, settings, c)
                log(f'chunk {c} done: {dict(collections.Counter(headers[c]["outcomes"]))}')

        out_of_memory = []
        for k, outcome, sec, read in _imap(_task, list(tasks.values()), processes, memory_cap):
            if outcome is None:  # Solved again once the batch's other wells are done, so that they are kept
                out_of_memory.append(k)
            else:
                done(k, outcome, sec, read)
        for k in sorted(out_of_memory):
            done(*_retry(tasks[k], memory_cap, log))
    state = _summary(headers, n_wells)
    state.update(limit=limit, seconds=time.perf_counter() - t0)
    log(f'{state["accepted"]} wells accepted of {state["drawn"]} drawn; complete: {state["complete"]}')
    return state


def _ks(c, chunk_size):
    return range(c * chunk_size, (c + 1) * chunk_size)


def run_wells(kind: str, seed: int, ks, out, processes: int = 6, tables=None, memory_cap=None, log=print) -> None:
    """
    The given draw indices' wells only, each result kept in OUT/wells/kK.pkl as run keeps them, without chunks: a
    check of a few wells, or of the port against the prototype. load_wells reads them.
    """
    (Path(out) / 'wells').mkdir(parents=True, exist_ok=True)
    tables = None if tables is None else str(Path(tables).resolve())
    settings_hash = _settings_hash(run_settings(kind, seed, tables))
    tasks = {k: (kind, seed, k, str(out), tables, settings_hash) for k in ks}
    out_of_memory = []
    for k, outcome, sec, read in _imap(_task, list(tasks.values()), processes, memory_cap):
        if outcome is None:
            out_of_memory.append(k)
        else:
            log(f'well {k}: {outcome} in {sec:.1f} s' + (' (kept)' if read else ''))
    for k in sorted(out_of_memory):
        k, outcome, sec, _ = _retry(tasks[k], memory_cap, log)
        log(f'well {k}: {outcome} in {sec:.1f} s')


def load_wells(out, ks) -> list:
    """The results that run_wells kept for the draw indices ks, in their order."""
    return [_load(_well_path(out, k)) for k in ks]
