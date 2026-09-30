"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Helpers shared by the build scripts that run in the v1.0.0 and Rust environments. Both have
v1.0.0's `manywells` package; importing this module puts the v1.0.0 worktree (for its scripts),
verification/src and plans/evidence on sys.path.
"""

import contextlib
import io
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
V1 = Path(os.environ.get('MANYWELLS_V1', REPO / '.worktrees' / 'v1.0.0'))
os.environ['MANYWELLS_V1'] = str(V1)  # plans/evidence scripts put it first on sys.path (else the cwd)
DATA = REPO / 'verification' / 'data' / 'build'   # intermediate files, gitignored
for p in (str(REPO / 'plans' / 'evidence'), str(REPO / 'verification' / 'src'), str(V1)):
    if p not in sys.path:
        sys.path.insert(0, p)

from manywells.choke import BernoulliChokeModel, SimpsonChokeModel  # noqa: E402  (v1.0.0 API)
from manywells.inflow import ProductivityIndex, Vogel  # noqa: E402
from manywells.simulator import BoundaryConditions, WellProperties  # noqa: E402

DIM_X = 7
WELL_FIELDS = ('L', 'D', 'rho_l', 'R_s', 'cp_g', 'cp_l', 'f_D', 'h')
BC_FIELDS = ('p_r', 'p_s', 'T_r', 'T_s', 'u', 'w_lg')
INFLOW = {'Vogel': 'vogel', 'ProductivityIndex': 'pi'}
CHOKE = {'SimpsonChokeModel': 'simpson', 'BernoulliChokeModel': 'bernoulli'}


def params_of(wp, bc) -> dict:
    """The residual graph's parameter values for a v1.0.0 well."""
    inflow = wp.inflow
    values = {k: float(getattr(wp, k)) for k in WELL_FIELDS} | {k: float(getattr(bc, k)) for k in BC_FIELDS}
    values |= {'w_l_max': float(getattr(inflow, 'w_l_max', 0.0)), 'k_l': float(getattr(inflow, 'k_l', 0.0)),
               'f_g': float(inflow.f_g), 'K_c': float(wp.choke.K_c), 'cpr': float(wp.choke.cpr)}
    return values


def variant_of(wp) -> list:
    """[inflow, choke, profile] of a v1.0.0 well, as in manywells_verify.residual.Variant."""
    return [INFLOW[type(wp.inflow).__name__], CHOKE[type(wp.choke).__name__], wp.choke.chk_profile]


def well_of(case: dict):
    """v1.0.0 WellProperties and BoundaryConditions for a case (params, variant)."""
    prm, (inflow, choke, profile) = case['params'], case['variant']
    inflow_model = (Vogel(w_l_max=prm['w_l_max'], f_g=prm['f_g']) if inflow == 'vogel'
                    else ProductivityIndex(k_l=prm['k_l'], f_g=prm['f_g']))
    choke_model = (SimpsonChokeModel if choke == 'simpson' else BernoulliChokeModel)(K_c=prm['K_c'], chk_profile=profile)
    if abs(choke_model.cpr - prm['cpr']) > 1e-12:
        raise ValueError(f'{case["case_id"]}: cpr {prm["cpr"]} differs from v1.0.0\'s {choke_model.cpr}')
    wp = WellProperties(**{k: prm[k] for k in WELL_FIELDS}, inflow=inflow_model, choke=choke_model)
    bc = BoundaryConditions(**{k: prm[k] for k in BC_FIELDS})
    return wp, bc


def seed_for(*parts) -> int:
    """Deterministic 31-bit seed from a case's source, well and draw number."""
    return zlib.crc32(':'.join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def interpolate_state(x, n_from: int, n_to: int) -> np.ndarray:
    """Linear interpolation of a state along the well from N = n_from to N = n_to cells."""
    X = np.asarray(x, dtype=float).reshape(n_from + 1, DIM_X)
    z_from, z_to = np.linspace(0, 1, n_from + 1), np.linspace(0, 1, n_to + 1)
    return np.column_stack([np.interp(z_to, z_from, X[:, j]) for j in range(DIM_X)]).ravel()


@contextlib.contextmanager
def quiet():
    """Silence v1's prints (it prints the solver status on every failed solve)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(obj, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1) + '\n')
