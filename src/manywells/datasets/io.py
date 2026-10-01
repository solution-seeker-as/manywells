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


def write_dataset(path: Path, rows: pd.DataFrame, draws: dict, metadata: dict):
    """
    Write <path>.parquet (the rows, with ID), <path>_config.parquet (one row per well: its draws) and
    <path>_meta.json (the generator's metadata).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path.with_suffix('.parquet'), index=False)
    configs = pd.DataFrame([{'ID': i} | {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(d).items()}
                            for i, d in draws.items()])
    configs.to_parquet(path.with_name(path.stem + '_config.parquet'), index=False)
    path.with_name(path.stem + '_meta.json').write_text(json.dumps(metadata | {'code_version': code_version()}, indent=1))
