"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Every example in scripts/sim_examples/ runs to the end, headless. The examples are the tutorial
for the public API, so a change that breaks one fails here (AGENTS.md, "Done means").

Each example runs in its own process, the way AGENTS.md says to run it: as a module
(`python -m scripts.sim_examples.<name>`), with the project root on the path so that imports of
`scripts.*` resolve. It runs in a temporary directory, so figures it saves stay out of the repo.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted(p.stem for p in (ROOT / 'scripts' / 'sim_examples').glob('*.py'))


@pytest.mark.slow
@pytest.mark.parametrize('name', EXAMPLES)
def test_example_runs(name, tmp_path):
    env = os.environ | {
        'MPLBACKEND': 'Agg',       # plt.show() returns at once
        'MPLCONFIGDIR': str(tmp_path),
        'PYTHONPATH': os.pathsep.join([str(ROOT), str(ROOT / 'src'), os.environ.get('PYTHONPATH', '')]),
    }
    result = subprocess.run([sys.executable, '-m', f'scripts.sim_examples.{name}'], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f'{name} failed:\n{result.stderr[-3000:]}'


def test_examples_found():
    assert EXAMPLES, 'no examples in scripts/sim_examples/'
