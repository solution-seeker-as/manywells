"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Row vectors of the v2.0.0 configuration (specs/model/README.md, "Test vectors"): develop's residual rows in that
configuration, at perturbed roots of four wells, written to specs/model/vectors/v2_rows.json. They pin the
configuration as implemented, so that later work on develop cannot change it unnoticed; tests/test_spec_vectors.py
checks both backends against them. From the repository root:

    uv run python specs/tools/make_v2_vectors.py

Each well's inputs are written out in full, every option and constant included, so that the wells stay in v2.0.0
when develop's defaults move: v2_inputs builds them, here and in the test.
"""

import json
import math
import pathlib

import casadi as ca
import numpy as np

from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
from manywells.configurations import V2, check
from manywells.discretization import build_system
from manywells.friction import RoughnessFriction
from manywells.geometry import WellGeometry
from manywells.inflow import FixedFlowRate, ProductivityIndex, Vogel
from manywells.pvt.black_oil import BlackOil
from manywells.pvt.fluid import FluidModel
from manywells.pvt.table import PVTTable
from manywells.simulator import BoundaryConditions, SSDFSimulator, WellProperties
from manywells.slip import SlipModel
from manywells.thermal import ThermalModel

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'specs' / 'model' / 'vectors' / 'v2_rows.json'
STATE = ['p', 'v_g', 'v_l', 'alpha', 'rho_g', 'rho_l', 'T']
N_CELLS = 10

THERMAL = dict(frictional_heating=True, gravity_term=True, lift_gas_mixing=True, joule_thomson=True)
SLIP = dict(
    regime_model='hasan',
    C_0_annular=1.0,
    C_0_slug=1.175,
    C_0_bubbly=1.2,
    v_inf_annular=0.0,
    f_ms=0.02,
    transition_width=0.02,
    lambda_floor=0.1,
    profile_scale=1.0,
)
STREAM = dict(rho_w=1025.0, cp_g=2225.0, cp_o=2000.0, cp_w=4184.0, surface_tension_model='oil')


def _grid(md, tvd):
    """The grid of N_CELLS uniform cells on a survey (GEO-4), as its points' MD and TVD from the surface."""
    g = WellGeometry.from_survey(md, tvd, n_cells=N_CELLS)
    return dict(md=list(g.md_survey), tvd=list(g.tvd_survey))


def _non_uniform(depth):
    """A vertical grid whose cells grow towards the bottomhole."""
    s = np.linspace(0.0, 1.0, N_CELLS + 1) ** 1.3 * depth
    return dict(md=s.tolist(), tvd=s.tolist())


def _K_c(D, share):
    return share * math.pi * (D / 2) ** 2


WELLS = {
    # Deviated oil well with water, lift gas through a valve, Vogel inflow, Simpson choke with the sigmoid profile
    'V1': dict(
        grid=_grid([0.0, 600.0, 2900.0], [0.0, 600.0, 2400.0]),
        D=0.1397,
        fluid=dict(pvt='black_oil', rho_o=850.0, rho_g=0.80, p_sep=1.01325, T_sep=288.15, p_bubble=None)
        | dict(gor=120.0, wlr=0.3)
        | STREAM,
        friction=dict(roughness=3e-5, correlation='chen'),
        thermal=dict(h=25.0) | THERMAL,
        slip=SLIP,
        inflow=dict(model='vogel', w_l_max=60.0),
        choke=dict(model='simpson', K_c=_K_c(0.1397, 0.12), profile='sigmoid'),
        injection_md=1700.0,
        bc=dict(p_r=260.0, p_s=25.0, T_r=368.15, T_s=277.15, u=0.7, w_lg=1.5, T_lg=None),
    ),
    # Gas-rich well with a horizontal section and a bubble point, productivity index, Bernoulli choke, a profile scale
    'V2': dict(
        grid=_grid([0.0, 2200.0, 2800.0], [0.0, 2200.0, 2200.0]),
        D=0.1143,
        fluid=dict(pvt='black_oil', rho_o=780.0, rho_g=0.85, p_sep=20.0, T_sep=300.0, p_bubble=180.0)
        | dict(gor=1500.0, wlr=0.05)
        | STREAM,
        friction=dict(roughness=1e-4, correlation='chen'),
        thermal=dict(h=15.0) | THERMAL,
        slip=SLIP | dict(profile_scale=1.1),
        inflow=dict(model='pi', k_l=0.15),
        choke=dict(model='bernoulli', K_c=_K_c(0.1143, 0.1), profile='linear'),
        injection_md=None,
        bc=dict(p_r=230.0, p_s=30.0, T_r=370.0, T_s=280.0, u=0.5, w_lg=0.0, T_lg=None),
    ),
    # Vertical well with a table set at a gas-oil ratio between its tables, a fixed liquid rate, cold lift gas at the
    # bottomhole, Simpson choke with the convex profile
    'V3': dict(
        grid=_grid([0.0, 2000.0], [0.0, 2000.0]),
        D=0.1554,
        fluid=dict(pvt='table', path='tests/data/pvt/synthetic_eos_oil_gor.csv') | dict(gor=100.0, wlr=0.2) | STREAM,
        friction=dict(roughness=4.5e-5, correlation='chen'),
        thermal=dict(h=20.0) | THERMAL,
        slip=SLIP,
        inflow=dict(model='fixed', w_l_const=25.0),
        choke=dict(model='simpson', K_c=_K_c(0.1554, 0.15), profile='convex'),
        injection_md=None,
        bc=dict(p_r=240.0, p_s=20.0, T_r=360.0, T_s=277.15, u=0.8, w_lg=1.0, T_lg=300.0),
    ),
    # A non-uniform vertical grid with one table, lift gas through a valve, Vogel inflow, Bernoulli choke with the
    # concave profile
    'V4': dict(
        grid=_non_uniform(2300.0),
        D=0.1397,
        fluid=dict(pvt='table', path='tests/data/pvt/synthetic_eos_oil.csv') | dict(gor=108.7299, wlr=0.0) | STREAM,
        friction=dict(roughness=4.5e-5, correlation='chen'),
        thermal=dict(h=30.0) | THERMAL,
        slip=SLIP,
        inflow=dict(model='vogel', w_l_max=45.0),
        choke=dict(model='bernoulli', K_c=_K_c(0.1397, 0.12), profile='concave'),
        injection_md=1200.0,
        bc=dict(p_r=250.0, p_s=20.0, T_r=365.0, T_s=277.15, u=0.6, w_lg=1.2, T_lg=None),
    ),
}


def _pvt(f):
    if f['pvt'] == 'table':
        return PVTTable.read_csv(ROOT / f['path'])
    return BlackOil(
        rho_o=f['rho_o'], rho_g=f['rho_g'], gas_law='dak', p_sep=f['p_sep'], T_sep=f['T_sep'], p_bubble=f['p_bubble']
    )


def _inflow(w):
    if w['model'] == 'vogel':
        return Vogel(w_l_max=w['w_l_max'])
    if w['model'] == 'pi':
        return ProductivityIndex(k_l=w['k_l'])
    return FixedFlowRate(w_l_const=w['w_l_const'])


def v2_inputs(w):
    """The (wp, bc) of a well of v2_rows.json, every option set explicitly, checked to be in v2.0.0."""
    f = w['fluid']
    choke = SimpsonChokeModel if w['choke']['model'] == 'simpson' else BernoulliChokeModel
    wp = WellProperties(
        geometry=WellGeometry(md_survey=np.array(w['grid']['md']), tvd_survey=np.array(w['grid']['tvd']), D=w['D']),
        fluid=FluidModel(
            pvt=_pvt(f),
            **{k: f[k] for k in ('rho_w', 'gor', 'wlr', 'surface_tension_model', 'cp_g', 'cp_o', 'cp_w')},
        ),
        friction=RoughnessFriction(**w['friction']),
        thermal=ThermalModel(**w['thermal']),
        slip=SlipModel(**w['slip']),
        inflow=_inflow(w['inflow']),
        choke=choke(K_c=w['choke']['K_c'], chk_profile=w['choke']['profile']),
        injection_md=w['injection_md'],
    )
    check(wp, V2)
    return wp, BoundaryConditions(**w['bc'])


def perturb(X):
    """A few per cent off the root, deterministic, alpha kept inside (0, 1) (make_v1_vectors.perturb)."""
    s = np.sin(np.arange(X.size).reshape(X.shape) + 1.0)
    Y = X * (1 + 0.05 * s)
    Y[:, 3] = np.clip(X[:, 3] + 0.02 * s[:, 3], 0.01, 0.99)
    return Y


def num(x):
    return float(f'{float(x):.17g}')


def row_vectors():
    wells = []
    for name, w in WELLS.items():
        wp, bc = v2_inputs(w)
        root = SSDFSimulator(wp).simulate(bc)
        X = np.asarray(root.x).reshape(-1, len(STATE))
        Y = perturb(X)
        system = build_system(wp)
        r = np.asarray(system.residual(Y.ravel(), system.params(bc))).ravel()
        wells.append(
            dict(
                name=name,
                params=w,
                choked_at_root=bool(root.choked),
                root=X.tolist(),
                x=Y.tolist(),
                rows=[[eq_id, num(v)] for eq_id, v in zip(system.row_ids, r)],
            )
        )
        print(f'{name}: p_0 = {X[0, 0]:.3f} bar, p_N = {X[-1, 0]:.3f} bar, choked at the root: {root.choked}')
    return dict(
        generated_by='specs/tools/make_v2_vectors.py',
        source=f'develop in the {V2} configuration (casadi {ca.__version__}, numpy {np.__version__})',
        units='bar, K, kg/s, m, as in specs/model/nomenclature.md; row units in specs/model/discretization.md',
        state_order=STATE,
        note='x is the root perturbed by perturb(); rows are in the order of build_system(wp).row_ids (DISC-11)',
        wells=wells,
    )


def main():
    OUT.write_text(json.dumps(row_vectors(), indent=1) + '\n')
    print(f'wrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
