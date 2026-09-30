"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Freeze v1.0.0's residual as per-point blocks (v2 plan, Step 2 item 1).

Builds each block from v1.0.0's own equation methods, with every well property and boundary
condition as a symbolic parameter, and writes residual_graph/v1.0.0/: one .casadi file per
block, generated C source, a manifest and golden vectors. It then checks that the stacked
blocks reproduce v1.0.0's full residual and Jacobian.

Runs in the v1.0.0 environment (see README.md in this folder), from the repository root:

    .worktrees/v1.0.0/.venv/bin/python verification/build/build_graph.py --data-dir <dir>

where <dir> holds manywells-sol-1_config.zip from the solution-seeker-as/manywells dataset.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import casadi as ca
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
V1 = Path(os.environ.get('MANYWELLS_V1', REPO / '.worktrees' / 'v1.0.0'))
sys.path[:0] = [str(V1), str(REPO / 'verification' / 'src'), str(REPO / 'plans' / 'evidence')]

from manywells.choke import BernoulliChokeModel, SimpsonChokeModel  # noqa: E402  (v1.0.0 API)
from manywells.inflow import ProductivityIndex, Vogel  # noqa: E402
from manywells.simulator import BoundaryConditions, SimError, SSDFSimulator, WellProperties  # noqa: E402
from scripts.load_well_from_dataset import load_well  # noqa: E402
from stability_label import Graph  # noqa: E402  (v1's full g and J, plans/evidence)
from manywells_verify.residual import (CHOKES, DIM_X, INFLOWS, PROFILES, STATE,  # noqa: E402
                                       ResidualGraph, Variant)

VERSION = 'v1.0.0'
OUT = REPO / 'verification' / 'residual_graph' / VERSION
C_FILE = 'manywells_v1_residual.c'

# Parameter vector P: every well property and boundary condition v1.0.0's equations use
PARAMS = {
    'L': 'm', 'D': 'm', 'rho_l': 'kg/m3', 'R_s': 'J/(kg K)', 'cp_g': 'J/(kg K)', 'cp_l': 'J/(kg K)',
    'f_D': '-', 'h': 'W/(m2 K)',
    'w_l_max': 'kg/s (Vogel)', 'k_l': 'kg/s/bar (productivity index)', 'f_g': '-',
    'K_c': 'm2', 'cpr': '-',
    'p_r': 'bar', 'p_s': 'bar', 'T_r': 'K', 'T_s': 'K', 'u': '-', 'w_lg': 'kg/s',
}
WELL_FIELDS = ('L', 'D', 'rho_l', 'R_s', 'cp_g', 'cp_l', 'f_D', 'h')
BC_FIELDS = ('p_r', 'p_s', 'T_r', 'T_s', 'u', 'w_lg')

# Row names per block, in v1.0.0's order (the Residuals check scales each row by its kind)
ROW_NAMES = {
    'left': ['gas inflow', 'liquid inflow', 'inflow temperature'],
    'closure': ['slip', 'gas EOS', 'liquid density'],
    'cell': ['gas mass', 'liquid mass', 'momentum', 'energy'],
    'choke': ['choke'],
}

CHECK_WELLS = (977, 1701, 1847, 0, 1, 2, 3)  # sol-1 config IDs; 977 and 1701 land on the trickle root
RTOL = 1e-13


def symbolic_simulator(inflow: str, choke: str, profile: str):
    """v1.0.0 simulator whose well properties, boundary conditions and N are CasADi symbols."""
    P = ca.SX.sym('P', len(PARAMS))
    p = dict(zip(PARAMS, ca.vertsplit(P)))
    N = ca.SX.sym('N')

    # Numeric placeholders pass v1's __post_init__ asserts; every value is then replaced by a symbol
    inflow_model = Vogel(w_l_max=1.0, f_g=0.1) if inflow == 'vogel' else ProductivityIndex(k_l=1.0, f_g=0.1)
    choke_class = SimpsonChokeModel if choke == 'simpson' else BernoulliChokeModel
    choke_model = choke_class(K_c=1e-3, chk_profile=profile)
    wp = WellProperties(inflow=inflow_model, choke=choke_model)
    bc = BoundaryConditions()
    sim = SSDFSimulator(wp, bc)

    for k in WELL_FIELDS:
        setattr(wp, k, p[k])
    if inflow == 'vogel':
        inflow_model.w_l_max = p['w_l_max']
    else:
        inflow_model.k_l = p['k_l']
    inflow_model.f_g = p['f_g']
    choke_model.K_c = p['K_c']
    choke_model.cpr = p['cpr']
    for k in BC_FIELDS:
        setattr(bc, k, p[k])
    sim.n_cells = N
    return sim, P, N


def build_blocks():
    """Blocks as CasADi Functions returning the rows and their Jacobian w.r.t. the state inputs."""
    x, x_prev, i = ca.SX.sym('x', DIM_X), ca.SX.sym('x_prev', DIM_X), ca.SX.sym('i')
    X, X_prev = ca.vertsplit(x), ca.vertsplit(x_prev)

    def block(name, inputs, in_names, g, wrt):
        g = ca.vertcat(*g)
        outs = [g] + [ca.densify(ca.jacobian(g, w)) for w in wrt]
        out_names = ['g'] + [f'J_{n}' for w, n in zip(wrt, ('x', 'x_prev'))]
        return ca.Function(f'mwv1_{name}', inputs, outs, in_names, out_names)

    blocks = {}
    for inflow in INFLOWS:
        sim, P, N = symbolic_simulator(inflow, 'simpson', 'linear')
        blocks[f'left_{inflow}'] = block(f'left_{inflow}', [x, P], ['x', 'P'], sim._left_boundary_eqs(X), [x])

    sim, P, N = symbolic_simulator('vogel', 'simpson', 'linear')
    blocks['closure'] = block('closure', [x, P], ['x', 'P'], sim._closure_relations(X), [x])
    blocks['cell'] = block('cell', [x, x_prev, i, N, P], ['x', 'x_prev', 'i', 'N', 'P'],
                           sim._differential_equations(X, X_prev, i), [x, x_prev])

    for choke in CHOKES:
        for profile in PROFILES:
            sim, P, N = symbolic_simulator('vogel', choke, profile)
            blocks[f'choke_{choke}_{profile}'] = block(f'choke_{choke}_{profile}', [x, P], ['x', 'P'],
                                                       sim._right_boundary_eqs(X), [x])
    return blocks


def params_used(f: ca.Function):
    """Names of the parameters a block depends on."""
    inputs = f.sx_in()
    g, P = f.call(inputs)[0], inputs[f.index_in('P')]
    return [k for j, k in enumerate(PARAMS) if ca.depends_on(g, P[j])]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(blocks):
    OUT.mkdir(parents=True, exist_ok=True)
    cg = ca.CodeGenerator(C_FILE, {'with_header': True})
    for name, f in blocks.items():
        f.save(str(OUT / f'{name}.casadi'))
        cg.add(f)
    cg.generate(str(OUT) + os.sep)

    used = {name: params_used(f) for name, f in blocks.items()}
    unused = [k for k in PARAMS if not any(k in u for u in used.values())]
    if unused:
        raise RuntimeError(f'parameters no block depends on: {unused}')

    v1_commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', f'{VERSION}^{{commit}}'],
                               capture_output=True, text=True, check=True).stdout.strip()
    manifest = {
        'model': f'ManyWells {VERSION}',
        'v1_commit': v1_commit,
        'built_with': {'casadi': ca.__version__, 'numpy': np.__version__, 'python': platform.python_version()},
        'state': list(STATE),
        'dim_x': DIM_X,
        'row_layout': ['point 0: left boundary (3), closure (3)',
                       'point 0 < i < N: cell (4), closure (3)',
                       'point N: cell (4), choke (1), closure (3)',
                       'len(r) = len(x) = 7(N + 1); choke row = 7N + 3'],
        'params': list(PARAMS),
        'param_units': PARAMS,
        'blocks': {name: {'c_name': f.name(),
                          'inputs': f.name_in(),
                          'outputs': f.name_out(),
                          'rows': f.size1_out(0),
                          'row_names': ROW_NAMES[name.split('_')[0]],
                          'params_used': used[name],
                          'sha256': sha256(OUT / f'{name}.casadi')}
                   for name, f in blocks.items()},
        'c_source': {'file': C_FILE, 'sha256': sha256(OUT / C_FILE),
                     'header': C_FILE.replace('.c', '.h'), 'header_sha256': sha256(OUT / C_FILE.replace('.c', '.h'))},
        'not_covered': ['FixedFlowRate inflow (not used by any v1.0.0 dataset)'],
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


# ---------------------------------------------------------------------------------------------
# Check against v1.0.0's full graph, and golden vectors

def params_of(wp, bc) -> dict:
    inflow = wp.inflow
    values = {k: getattr(wp, k) for k in WELL_FIELDS} | {k: getattr(bc, k) for k in BC_FIELDS}
    values |= {'w_l_max': getattr(inflow, 'w_l_max', 0.0), 'k_l': getattr(inflow, 'k_l', 0.0), 'f_g': inflow.f_g,
               'K_c': wp.choke.K_c, 'cpr': wp.choke.cpr}
    return values


def variant_of(wp) -> Variant:
    inflow = {'Vogel': 'vogel', 'ProductivityIndex': 'pi'}[type(wp.inflow).__name__]
    choke = {'SimpsonChokeModel': 'simpson', 'BernoulliChokeModel': 'bernoulli'}[type(wp.choke).__name__]
    return Variant(inflow, choke, wp.choke.chk_profile)


def solve(wp, bc, n_cells):
    """v1's solution from its default guess, or the initial guess if the solve fails."""
    sim = SSDFSimulator(wp, bc, n_cells=n_cells)
    with contextlib.redirect_stdout(io.StringIO()):  # v1 prints the solver status on failure
        try:
            return np.array(sim.simulate()), 'solution'
        except SimError:
            return np.array(sim._initial_guess()), 'initial guess'


def check_wells(data_dir: Path):
    """(tag, wp, bc, N) for the stacking check: sol-1 configs, gas lift, every choke variant, PI inflow."""
    df = pd.read_csv(data_dir / 'manywells-sol-1_config.zip', compression='zip')
    wells = []
    for well_id in CHECK_WELLS:
        well = load_well(well_id, df)
        wells.append((f'sol-1 {well_id}', well.wp, well.bc, 100))
    lift = load_well(int(df.loc[df['has_gas_lift'], 'ID'].iloc[0]), df)
    lift.bc.w_lg = 1.0
    wells.append(('sol-1 gas lift, w_lg = 1', lift.wp, lift.bc, 100))
    base = load_well(977, df)
    for choke in CHOKES:
        for profile in PROFILES:
            model = (SimpsonChokeModel if choke == 'simpson' else BernoulliChokeModel)(
                K_c=base.wp.choke.K_c, chk_profile=profile)
            wp = WellProperties(**{k: getattr(base.wp, k) for k in WELL_FIELDS}, inflow=base.wp.inflow, choke=model)
            wells.append((f'977 with {choke} {profile}', wp, base.bc, 100))
    default_wp = WellProperties()  # productivity index inflow, Bernoulli choke
    default_wp.choke = BernoulliChokeModel(K_c=0.1 * default_wp.A)
    wells.append(('default well, N = 200', default_wp, BoundaryConditions(), 200))
    wells.append(('default well, N = 37', default_wp, BoundaryConditions(), 37))
    return wells


def rel_err(a, b):
    """Largest |a - b| relative to the magnitude of the entries (at least 1); inf if NaNs differ."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    nan = np.isnan(a)
    if not np.array_equal(nan, np.isnan(b)):
        return np.inf
    a, b = a[~nan], b[~nan]
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.maximum(np.abs(a), np.abs(b))), initial=0.0))


def check_and_golden(data_dir: Path):
    graph = ResidualGraph(VERSION, OUT.parent)
    rng = np.random.default_rng(0)
    worst, lines, stack_golden = 0.0, [], {}
    golden = {name: [] for name in graph.manifest['blocks']}

    for tag, wp, bc, N in check_wells(data_dir):
        full = Graph(wp, bc, n_cells=N).build()
        variant, P = variant_of(wp), graph.param_vector(params_of(wp, bc))
        x_sol, kind = solve(wp, bc, N)
        states = [(kind, x_sol)] + [(f'perturbed {k}', x_sol * (1 + 0.02 * rng.standard_normal(x_sol.size)))
                                    for k in range(3)]
        for state_tag, x in states:
            r, J = graph.evaluate(x, P, N, variant)
            e_r = rel_err(r, np.array(full.g_fun(x)).ravel())
            e_J = rel_err(J.toarray(), np.array(full.J_fun(x)))
            worst = max(worst, e_r, e_J)
            lines.append(f'  {tag:34s} {state_tag:14s} r {e_r:.1e}  J {e_J:.1e}')

        # Golden vectors: a few inputs per block, evaluated here with this CasADi version
        X = x_sol.reshape(N + 1, DIM_X)
        for name in (f'left_{variant.inflow}', f'choke_{variant.choke}_{variant.profile}', 'closure', 'cell'):
            if len(golden[name]) >= 6:
                continue
            k = int(rng.integers(1, N + 1))
            inputs = {'left': [X[0], P], 'choke': [X[N], P], 'closure': [X[k], P],
                      'cell': [X[k], X[k - 1], float(k), float(N), P]}[name.split('_')[0]]
            golden[name].append(inputs)

        if tag in ('sol-1 977', 'default well, N = 200'):
            v = rng.standard_normal((x_sol.size, 2))
            stack_golden[tag] = {'x': x_sol, 'P': P, 'N': N, 'variant': variant, 'kind': kind,
                                 'r': np.array(full.g_fun(x_sol)).ravel(), 'v': v,
                                 'Jv': np.array(full.J_fun(x_sol)) @ v}

    # Every boundary block needs golden vectors, including those no check well uses directly
    blocks = {name: ca.Function.load(str(OUT / f'{name}.casadi')) for name in golden}
    reference = stack_golden['sol-1 977']
    X_ref = reference['x'].reshape(-1, DIM_X)
    for name, rows in golden.items():
        while len(rows) < 2:
            rows.append([X_ref[0] if name.startswith('left') else X_ref[-1], reference['P']])

    arrays = {}
    for name, rows in golden.items():
        f = blocks[name]
        outs = [f(*row) for row in rows]  # every block has at least two outputs: g and J_x
        for j in range(f.n_in()):
            arrays[f'{name}.in{j}'] = np.array([np.asarray(row[j], dtype=float) for row in rows])
        for j in range(f.n_out()):
            arrays[f'{name}.out{j}'] = np.array([np.array(o[j]) for o in outs])
    for k, (tag, s) in enumerate(stack_golden.items()):
        arrays |= {f'stack{k}.tag': np.array(tag), f'stack{k}.x': s['x'], f'stack{k}.P': s['P'],
                   f'stack{k}.N': np.array(s['N']), f'stack{k}.kind': np.array(s['kind']),
                   f'stack{k}.variant': np.array([s['variant'].inflow, s['variant'].choke, s['variant'].profile]),
                   f'stack{k}.r': s['r'], f'stack{k}.v': s['v'], f'stack{k}.Jv': s['Jv']}
    np.savez_compressed(OUT / 'golden.npz', **arrays)

    print('Stacked blocks against v1.0.0\'s full residual and Jacobian (largest relative difference):')
    print('\n'.join(lines))
    print(f'Worst: {worst:.1e} (tolerance {RTOL:.0e})')
    return worst <= RTOL


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    parser.add_argument('--data-dir', type=Path, required=True, help='folder with manywells-sol-1_config.zip')
    args = parser.parse_args()

    save(build_blocks())
    print(f'Wrote {len(list(OUT.glob("*.casadi")))} blocks, {C_FILE} and manifest.json to {OUT.relative_to(REPO)}')
    sys.exit(0 if check_and_golden(args.data_dir) else 1)


if __name__ == '__main__':
    main()
