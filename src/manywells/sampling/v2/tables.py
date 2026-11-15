"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The ManyWells side of a table well's files (specs/sampling.md, SMP-116): the table set that table_build.py built for
the well, read as its PVTTableSet and so checked by PVT-TAB-1 (specs/model/pvt/table.md).
"""

import functools
import json
from pathlib import Path

from manywells.pvt.table import PVTTable

_TABLE_CLASSES = ('volatile_oil', 'gas_condensate')  # The fluid classes that take a PVT table  # spec: SMP-112


class TableMismatch(RuntimeError):
    """A folder of tables that was not built from the run's own draws: its record of a well, or the well's GOR axis,
    is not the one the run's draws give, as after a rule changed since the build. Not a well's outcome: the run
    stops, and the tables are built again."""


@functools.lru_cache(maxsize=4)
def _read_wells_file(path: str, stamp: tuple) -> tuple:
    """A folder's wells.json as (kind, seed, wells_range, {draw index: record}), read once per version of the file
    (stamp: its modification time and size)."""
    spec = json.loads(Path(path).read_text())
    return spec['kind'], spec['seed'], tuple(spec['wells_range']), {w['well']: w for w in spec['wells']}


def _wells_file(tables_dir) -> tuple:  # spec: SMP-116
    """The folder's wells.json (table_wells.py's docstring) as (kind, seed, wells_range, {draw index: record})."""
    path = Path(tables_dir) / 'wells.json'
    stat = path.stat()
    return _read_wells_file(str(path.resolve()), (stat.st_mtime_ns, stat.st_size))


def _read_table(tables_dir, k, kind, seed) -> dict:  # spec: SMP-116
    """
    Table well k's manifest from the folder that table_build.py built for the dataset kind ('sol' or 'nsol') and
    seed, with its PVTTableSet under 'pvt'. Its status is 'ok', or 'failed' with the step and the error: a step of the
    build failed ('fluid', 'axis' or 'tables'); none of the well's fluid draws gave a fluid of its class ('class',
    SMP-115); the folder holds no table for the well, or another dataset's ('missing'); or PVT-TAB-1's checks refuse
    the table ('check'). A well without a fluid of its class is not_producible, and another whose table failed is
    table_failed (SMP-106); neither is solved.

    :param tables_dir: The folder of the build, with the manifests kK.json and the tables kK.csv
    :param k: The well's draw index
    """
    tables_dir = Path(tables_dir)
    path = tables_dir / f'k{k:06d}.json'
    if not path.exists():
        return dict(status='failed', step='missing', error=f'no table built for draw index {k}')
    man = json.loads(path.read_text()) | dict(tables_dir=str(tables_dir))
    if (man.get('kind'), man.get('seed')) != (kind, seed):
        error = f'the table is for {man.get("kind")}, seed {man.get("seed")}, not {kind}, seed {seed}'
        return dict(man, status='failed', step='missing', error=error)
    if man['status'] == 'ok':
        try:
            man['pvt'] = PVTTable.read_csv(tables_dir / man['csv'])
        except ValueError as e:
            man.update(status='failed', step='check', error=f'PVT-TAB-1: {e}'[:500])
    return man
