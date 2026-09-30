"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Cases and roots, and their parquet files.

A case is one well at one operating point on one grid: the parameter vector of the residual
graph (one column per parameter), the model variant, the number of cells N and provenance.
A root is a full state x of length 7(N + 1). Reference root sets and candidate results use
the same root file format: one row per root, keyed by case_id. A case with no rows in a
candidate file means that the candidate reported no root for it.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from manywells_verify.residual import Variant

SCHEMA_VERSION = 1
LABELS = ('stable', 'unstable', 'indeterminate')
CASE_COLUMNS = ('schema_version', 'case_id', 'model', 'source', 'inflow', 'choke', 'profile', 'n_cells', 'group')
META_COLUMNS = ('config_id', 'seed', 'note')
ROOT_COLUMNS = ('case_id', 'root', 'x', 'label', 'operating_point', 'choked')


@dataclass(frozen=True)
class Case:
    case_id: str
    params: Mapping[str, float]
    n_cells: int
    variant: Variant = Variant()
    model: str = 'v1.0.0'
    source: str = ''
    group: str = ''                        # links the N, 2N and 4N cases of a convergence group
    meta: Mapping = field(default_factory=dict)


@dataclass
class Root:
    x: np.ndarray
    label: str | None = None               # 'stable', 'unstable' or 'indeterminate', if reported
    operating_point: bool = False
    choked: bool | None = None             # the candidate's CHOKED flag, if reported
    info: Mapping = field(default_factory=dict)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        if self.label is not None and self.label not in LABELS:
            raise ValueError(f'label {self.label!r} is not one of {LABELS}')


def write_cases(cases, path: Path):
    rows = []
    for c in cases:
        row = {'schema_version': SCHEMA_VERSION, 'case_id': c.case_id, 'model': c.model, 'source': c.source,
               'inflow': c.variant.inflow, 'choke': c.variant.choke, 'profile': c.variant.profile,
               'n_cells': int(c.n_cells), 'group': c.group}
        row |= {k: float(v) for k, v in c.params.items()}
        row |= {k: c.meta.get(k) for k in META_COLUMNS}
        rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def read_cases(path: Path) -> dict[str, Case]:
    df = pd.read_parquet(path)
    versions = set(df['schema_version'])
    if versions != {SCHEMA_VERSION}:
        raise ValueError(f'{path}: schema version {versions}, expected {SCHEMA_VERSION}')
    params = [k for k in df.columns if k not in CASE_COLUMNS + META_COLUMNS]
    cases = {}
    for row in df.to_dict('records'):
        cases[row['case_id']] = Case(
            case_id=row['case_id'], params={k: float(row[k]) for k in params}, n_cells=int(row['n_cells']),
            variant=Variant(row['inflow'], row['choke'], row['profile']), model=row['model'],
            source=row['source'], group=row['group'] if isinstance(row['group'], str) else '',
            meta={k: row.get(k) for k in META_COLUMNS})
    return cases


def write_roots(roots: Mapping[str, list[Root]], path: Path):
    rows = []
    for case_id, case_roots in roots.items():
        for k, r in enumerate(case_roots):
            rows.append({'case_id': case_id, 'root': k, 'x': r.x.tolist(), 'label': r.label or '',
                         'operating_point': bool(r.operating_point), 'choked': r.choked, **r.info})
    df = pd.DataFrame(rows, columns=list(ROOT_COLUMNS) if not rows else None)
    df['choked'] = df['choked'].astype('boolean')
    df.to_parquet(path, index=False)


def read_roots(path: Path) -> dict[str, list[Root]]:
    df = pd.read_parquet(path).sort_values(['case_id', 'root'])
    extra = [k for k in df.columns if k not in ROOT_COLUMNS]
    roots = {}
    for row in df.to_dict('records'):
        choked = row.get('choked')
        roots.setdefault(row['case_id'], []).append(Root(
            x=np.asarray(row['x'], dtype=float), label=row['label'] or None,
            operating_point=bool(row['operating_point']),
            choked=None if choked is None or pd.isna(choked) else bool(choked),
            info={k: row[k] for k in extra}))
    return roots
