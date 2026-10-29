"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Dataset files: the rows, the well configs and the generator's metadata (its seed, configuration, grid and code
version, constitution "Dataset reproducibility").
"""

import json
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd


def code_version() -> str:
    """The package version, and the git commit (with '-dirty' for local changes) where there is one."""
    v = version('manywells')
    try:
        root = Path(__file__).resolve().parents[3]
        commit = subprocess.run(['git', 'describe', '--always', '--dirty'], cwd=root, capture_output=True, text=True,
                                check=True).stdout.strip()
        return f'{v} ({commit})'
    except (OSError, subprocess.CalledProcessError):
        return v


def trajectory_columns(trajectory: tuple) -> dict:
    """
    A draw's trajectory as the config table's columns: its kind ('vertical', 'deviated' or 'l_shaped'), the kickoff
    depth as a fraction of the true vertical depth and the inclination (degrees) of a deviated well, and the horizontal
    length (m) of an L-shaped one, NaN where they do not apply.
    """
    kind, *prm = trajectory
    out = {'trajectory': kind, 'kickoff_fraction': np.nan, 'inclination': np.nan, 'horizontal_length': np.nan}
    if kind == 'deviated':
        out['kickoff_fraction'], out['inclination'] = prm
    elif kind == 'l_shaped':
        (out['horizontal_length'],) = prm
    elif kind != 'vertical':
        raise ValueError(f'unknown trajectory {kind!r}')
    return out


def config_row(well: int, draw) -> dict:
    """One well's row of the config table: its ID and its draws (WellDraw), with the trajectory as typed columns."""
    row = {'ID': well}
    for k, v in asdict(draw).items():
        row |= trajectory_columns(v) if k == 'trajectory' else {k: list(v) if isinstance(v, tuple) else v}
    return row


def write_dataset(path: Path, rows: pd.DataFrame, draws: dict, metadata: dict):
    """
    Write <path>_meta.json (the generator's metadata), <path>_config.parquet (one row per well: its draws) and
    <path>.parquet (the rows, with ID), in that order, so that no rows are written without their record. The suffixes
    are appended to path's name, which may itself hold a dot (data/sol-v1.0).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(path.name + '_meta.json').write_text(json.dumps(metadata | {'code_version': code_version()}, indent=1))
    configs = pd.DataFrame([config_row(i, d) for i, d in draws.items()])
    configs.to_parquet(path.with_name(path.name + '_config.parquet'), index=False)
    rows.to_parquet(path.with_name(path.name + '.parquet'), index=False)
