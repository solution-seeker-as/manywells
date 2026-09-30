"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Test vectors for specs/model/ from ManyWells v1.0.0 (specs/model/README.md, "Test vectors").

Writes the component vectors into the generated blocks of the spec files, between
<!-- vectors:begin --> and <!-- vectors:end -->, and the residual-row vectors to
specs/model/vectors/v1_rows.json. Runs against v1.0.0, from the repository root:

    .worktrees/v1.0.0/.venv/bin/python specs/tools/make_v1_vectors.py
"""

import json
import math
import pathlib
import sys
from importlib.metadata import version

import casadi as ca
import numpy as np

import manywells.pvt as pvt
from manywells.ca_functions import ca_max_approx, ca_min_approx, ca_softmax
from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
from manywells.inflow import FixedFlowRate, ProductivityIndex, Vogel
from manywells.simulator import BoundaryConditions, SSDFSimulator, WellProperties
from manywells.slip import SlipModel, classify_flow_regime

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = ROOT / 'specs' / 'model'
BEGIN, END = '<!-- vectors:begin -->', '<!-- vectors:end -->'
STATE = ['p', 'v_g', 'v_l', 'alpha', 'rho_g', 'rho_l', 'T']

if version('manywells') != '1.0.0' or not hasattr(WellProperties(), 'L'):
    sys.exit('Run this script with the v1.0.0 environment (see the module docstring).')


def num(x):
    """A float (or a 1x1 CasADi value) as a float."""
    return float(np.asarray(ca.DM(x)).item()) if isinstance(x, (ca.DM, ca.SX)) else float(x)


def cell(x):
    if isinstance(x, bool):
        return 'true' if x else 'false'
    if isinstance(x, str):
        return x
    return repr(num(x))


def table(heading, inputs, outputs, cases, fn):
    """A vector table: fn(**case) returns the outputs, in order."""
    rows = []
    for case in cases:
        out = fn(**case)
        out = out if isinstance(out, tuple) else (out,)
        assert len(out) == len(outputs), heading
        rows.append([cell(case[k]) for k in inputs] + [cell(v) for v in out])
    head = inputs + ['→ ' + o for o in outputs]
    lines = [f'### {heading}', '', '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    lines += ['| ' + ' | '.join(r) + ' |' for r in rows]
    return '\n'.join(lines)


def choke(model, K_c, profile):
    return model(K_c=K_c, chk_profile=profile)


def slip_cases():
    # (v_g, v_l, alpha, rho_g, rho_l, T, D): bubbly, slug/churn, annular, the v1 slip.py example, near alpha = 0.7
    raw = [(1.0, 1.5, 0.1, 100.0, 850.0, 360.0, 0.1554),
           (3.0, 1.0, 0.45, 50.0, 850.0, 340.0, 0.1016),
           (25.0, 3.0, 0.9, 20.0, 900.0, 320.0, 0.127),
           (20.0, 5.0, 0.5, 1.0, 900.0, 293.15, 0.1),
           (12.0, 2.5, 0.7, 30.0, 999.1, 300.0, 0.0762)]
    cases = []
    for v_g, v_l, alpha, rho_g, rho_l, T, D in raw:
        sigma = num(pvt.dead_oil_surface_tension(rho_l, T))
        cases.append(dict(v_g=v_g, v_l=v_l, alpha=alpha, rho_g=rho_g, rho_l=rho_l, T=T, sigma=sigma, D=D))
    return cases


def component_tables():
    """Component vectors per spec file, in the order they appear there."""
    t = {}

    pairs = [dict(x=x, y=y, eps=1e-6) for x, y in
             [(1.0, 2.0), (2.0, 1.0), (5.0, 5.0), (-3.0, 0.5), (30.0, 30.001), (0.0, 1e-4)]]
    t['smoothing.md'] = [
        table('SMO-1', ['x', 'y', 'eps'], ['max'], pairs, lambda x, y, eps: ca_max_approx(x, y, eps)),
        table('SMO-2', ['x', 'y', 'eps'], ['min'], pairs, lambda x, y, eps: ca_min_approx(x, y, eps)),
        table('SMO-3', ['y1', 'y2', 'y3'], ['p1', 'p2', 'p3'],
              [dict(y1=a, y2=b, y3=c) for a, b, c in
               [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (-3.9, -1.5, 5.4), (10.0, -10.0, 0.0)]],
              lambda y1, y2, y3: tuple(np.asarray(ca_softmax(ca.DM([y1, y2, y3]))).ravel())),
    ]

    us = [dict(u=u) for u in [0.0, 0.05, 0.3, 0.5, 0.8, 1.0]]
    op = lambda profile: lambda u: choke(SimpsonChokeModel, 1.0, profile).choke_opening(u)  # noqa: E731
    K = 0.12 * math.pi * (0.1254 / 2) ** 2
    eq_cases = [
        dict(K_c=K, profile='linear', u=0.5, p_in=50.0, p_out=40.0, rho=850.0, Phi=1.0),     # unchoked
        dict(K_c=K, profile='sigmoid', u=0.8, p_in=50.0, p_out=20.0, rho=850.0, Phi=2.5),    # choked
        dict(K_c=K, profile='convex', u=0.3, p_in=50.0, p_out=50.0 * SimpsonChokeModel.critical_pressure_ratio(),
             rho=300.0, Phi=1.0),                                                             # at the switch
        dict(K_c=K, profile='concave', u=1.0, p_in=20.01, p_out=20.0, rho=850.0, Phi=4.0),   # small drop
        dict(K_c=0.5 * K, profile='sigmoid', u=0.05, p_in=120.0, p_out=100.0, rho=600.0, Phi=1.3),
    ]
    simpson_cases = [dict(x_g=x, rho_g=g, rho_l=l) for x, g, l in
                     [(0.0, 50.0, 850.0), (0.1, 50.0, 850.0), (0.5, 20.0, 900.0), (1.0, 100.0, 800.0), (0.02, 150.0, 950.0)]]
    simpson_rate = [dict(K_c=K, profile='sigmoid', u=0.6, p_in=45.0, p_out=30.0, x_g=0.1, rho_g=35.0, rho_l=850.0),
                    dict(K_c=K, profile='linear', u=0.9, p_in=60.0, p_out=15.0, x_g=0.4, rho_g=45.0, rho_l=900.0),
                    dict(K_c=2 * K, profile='concave', u=0.2, p_in=80.0, p_out=70.0, x_g=0.02, rho_g=70.0, rho_l=820.0)]
    bernoulli = [dict(K_c=K, profile='linear', u=0.7, p_in=40.0, p_out=25.0, rho_m=400.0),
                 dict(K_c=K, profile='convex', u=1.0, p_in=60.0, p_out=20.0, rho_m=150.0)]
    t['choke.md'] = [
        table('CHK-2, CHK-3', ['K_c', 'profile', 'u', 'p_in', 'p_out', 'rho', 'Phi'], ['w'], eq_cases,
              lambda K_c, profile, u, p_in, p_out, rho, Phi:
              choke(SimpsonChokeModel, K_c, profile).choke_equation(u, p_in, p_out, rho, Phi)),
        table('CHK-4', ['gamma'], ['r_c'], [dict(gamma=g) for g in [1.307, 1.3, 1.4, 1.2]],
              lambda gamma: SimpsonChokeModel.critical_pressure_ratio(gamma)),
        table('CHK-5 (multiplier)', ['x_g', 'rho_g', 'rho_l'], ['Phi'], simpson_cases,
              SimpsonChokeModel.simpson_multiplier),
        table('CHK-5 (rate)', ['K_c', 'profile', 'u', 'p_in', 'p_out', 'x_g', 'rho_g', 'rho_l'], ['w'],
              simpson_rate,
              lambda K_c, profile, u, p_in, p_out, x_g, rho_g, rho_l:
              choke(SimpsonChokeModel, K_c, profile).mass_flow_rate(u, p_in, p_out, x_g, rho_g, rho_l)),
        table('CHK-6', ['K_c', 'profile', 'u', 'p_in', 'p_out', 'rho_m'], ['w'], bernoulli,
              lambda K_c, profile, u, p_in, p_out, rho_m:
              choke(BernoulliChokeModel, K_c, profile).mass_flow_rate(u, p_in, p_out, rho_m)),
        table('CHK-7', ['u'], ['sigma'], us, op('linear')),
        table('CHK-8', ['u'], ['sigma'], us, op('sigmoid')),
        table('CHK-9', ['u'], ['sigma'], us, op('convex')),
        table('CHK-10', ['u'], ['sigma'], us, op('concave')),
        table('CHK-12', ['p_in', 'p_out'], ['choked'],
              [dict(p_in=50.0, p_out=p) for p in [20.0, 27.2, 27.25, 30.0]],
              lambda p_in, p_out: bool(choke(SimpsonChokeModel, K, 'linear').is_choked(p_in, p_out))),
    ]

    vogel = [dict(w_l_max=100.0, p=p, p_r=200.0) for p in [0.0, 60.0, 160.0, 200.0]]
    vogel.append(dict(w_l_max=35.5, p=120.3, p_r=180.7))
    t['inflow.md'] = [
        table('INF-1', ['w_l_max', 'p', 'p_r'], ['w_l'], vogel,
              lambda w_l_max, p, p_r: Vogel(w_l_max, 0.5).mass_flow_rates(p, p_r)[0]),
        table('INF-2', ['k_l', 'p', 'p_r'], ['w_l'],
              [dict(k_l=0.5, p=150.0, p_r=170.0), dict(k_l=1.2, p=80.5, p_r=200.0), dict(k_l=0.5, p=170.0, p_r=170.0)],
              lambda k_l, p, p_r: ProductivityIndex(k_l, 0.5).mass_flow_rates(p, p_r)[0]),
        table('INF-3', ['w_l_const', 'w_g_const', 'p', 'p_r'], ['w_l', 'w_g'],
              [dict(w_l_const=20.0, w_g_const=3.0, p=100.0, p_r=200.0)],
              lambda w_l_const, w_g_const, p, p_r: FixedFlowRate(w_l_const, w_g_const).mass_flow_rates(p, p_r)),
        table('INF-4', ['f_g', 'w_l'], ['w_g'],
              [dict(f_g=f, w_l=Vogel(80.0, f).mass_flow_rates(90.0, 210.0)[0]) for f in [0.05, 0.5, 0.9]],
              lambda f_g, w_l: Vogel(80.0, f_g).mass_flow_rates(90.0, 210.0)[1]),
    ]

    sc = slip_cases()
    classify = lambda v_g, v_l, alpha, rho_g, rho_l, T, **_: tuple(  # noqa: E731
        np.asarray(classify_flow_regime(v_g, v_l, alpha, rho_g, rho_l, T)).ravel())
    t['slip.md'] = [
        table('SLIP-2, SLIP-3', ['v_g', 'v_l', 'alpha', 'rho_g', 'rho_l', 'T', 'sigma', 'D'], ['C_0', 'v_inf'], sc,
              lambda v_g, v_l, alpha, rho_g, rho_l, T, D, **_:
              SlipModel().identify_parameters(v_g, v_l, alpha, rho_g, rho_l, T, D)),
        table('SLIP-4', ['rho_g', 'rho_l', 'T', 'sigma'], ['v_inf_b'],
              [{k: c[k] for k in ('rho_g', 'rho_l', 'T', 'sigma')} for c in sc],
              lambda rho_g, rho_l, T, **_: SlipModel.harmathy_rise_velocity(rho_g, rho_l, T)),
        table('SLIP-5', ['rho_g', 'rho_l', 'D'], ['v_inf_T'],
              [{k: c[k] for k in ('rho_g', 'rho_l', 'D')} for c in sc],
              SlipModel.taylor_rise_velocity),
        table('SLIP-6, SLIP-7', ['v_g', 'v_l', 'alpha', 'rho_g', 'rho_l', 'T', 'sigma'], ['p_a', 'p_s', 'p_b'],
              sc, classify),
        table('SLIP-8', ['v_g', 'v_l', 'alpha', 'rho_g', 'rho_l', 'T', 'sigma'], ['regime'], sc,
              lambda v_g, v_l, alpha, rho_g, rho_l, T, **_: SlipModel().flow_regime(v_g, v_l, alpha, rho_g, rho_l, T)),
    ]

    t['pvt/gas.md'] = [
        table('PVT-GAS-1', ['p', 'T', 'R_s'], ['rho_g'],
              [dict(p=1.01325, T=288.15, R_s=518.3), dict(p=150.0, T=373.15, R_s=518.3), dict(p=35.0, T=330.0, R_s=320.0)],
              lambda p, T, R_s: pvt.gas_density(R_s, p * 1e5, T)),
        table('PVT-GAS-2', ['R_s'], ['rho_g_sc'], [dict(R_s=r) for r in [518.3, 320.0, 420.0]],
              lambda R_s: pvt.gas_density(R_s)),
    ]
    t['pvt/oil.md'] = [
        table('PVT-OIL-2', ['rho'], ['API'], [dict(rho=r) for r in [825.0, 850.0, 925.0, 999.1]],
              pvt.api_from_density),
        table('PVT-OIL-3', ['rho', 'T'], ['sigma'],
              [dict(rho=r, T=T) for r, T in [(850.0, 293.15), (825.0, 373.15), (925.0, 330.0), (999.1, 280.0)]],
              pvt.dead_oil_surface_tension),
    ]

    def mix(rho_o, cp_o, rho_w, cp_w, x_o):
        m = pvt.liquid_mix(pvt.LiquidProperties('oil', rho_o, cp_o), pvt.LiquidProperties('water', rho_w, cp_w), x_o)
        return m.rho, m.cp
    t['pvt/mixture.md'] = [
        table('PVT-MIX-2, PVT-MIX-3, PVT-MIX-4', ['rho_o', 'cp_o', 'rho_w', 'cp_w', 'x_o'], ['rho_l', 'cp_l'],
              [dict(rho_o=o, cp_o=2000.0, rho_w=999.1, cp_w=4184.0, x_o=x) for o, x in
               [(850.0, 0.5), (825.0, 0.9), (925.0, 0.1), (870.0, 1.0), (870.0, 0.0)]],
              mix),
    ]
    return t


def write_blocks(tables):
    header = ('Generated by `specs/tools/make_v1_vectors.py` from ManyWells v1.0.0 '
              f'(casadi {ca.__version__}). Do not edit by hand.')
    for rel, blocks in tables.items():
        path = SPEC / rel
        text = path.read_text()
        start, end = text.index(BEGIN) + len(BEGIN), text.index(END)
        body = '\n\n'.join([header] + blocks)
        path.write_text(text[:start] + '\n' + body + '\n' + text[end:])
        print(f'wrote {len(blocks)} tables to {path.relative_to(ROOT)}')


# Residual-row vectors (discretization.md): two wells, perturbed roots, every row at every point

WELLS = {
    'W1': dict(L=2500.0, D=0.127, rho_l=900.0, R_s=420.0, cp_g=2225.0, cp_l=3000.0, f_D=0.03, h=25.0,
               inflow='vogel', w_l_max=80.0, f_g=0.15, choke='simpson', K_c=0.12 * math.pi * (0.127 / 2) ** 2,
               profile='sigmoid', p_r=249.2, p_s=30.0, T_r=363.15, T_s=277.15, u=0.6, w_lg=0.8, n_cells=10),
    'W2': dict(L=1800.0, D=0.1524, rho_l=820.0, R_s=500.0, cp_g=2225.0, cp_l=2200.0, f_D=0.05, h=15.0,
               inflow='pi', k_l=0.6, f_g=0.4, choke='bernoulli', K_c=0.1 * math.pi * (0.1524 / 2) ** 2,
               profile='linear', p_r=150.0, p_s=20.0, T_r=345.0, T_s=277.15, u=0.8, w_lg=0.0, n_cells=10),
}
ROW_IDS = {'first': ['INF-6', 'INF-7', 'THM-3'], 'cell': ['DISC-2', 'DISC-3', 'DISC-4', 'DISC-5'],
           'last': ['CHK-1'], 'closure': ['SLIP-1', 'PVT-GAS-1', 'PVT-MIX-1']}


def v1_simulator(w):
    inflow = Vogel(w['w_l_max'], w['f_g']) if w['inflow'] == 'vogel' else ProductivityIndex(w['k_l'], w['f_g'])
    model = SimpsonChokeModel if w['choke'] == 'simpson' else BernoulliChokeModel
    wp = WellProperties(L=w['L'], D=w['D'], rho_l=w['rho_l'], R_s=w['R_s'], cp_g=w['cp_g'], cp_l=w['cp_l'],
                        f_D=w['f_D'], h=w['h'], inflow=inflow, choke=model(K_c=w['K_c'], chk_profile=w['profile']))
    bc = BoundaryConditions(p_r=w['p_r'], p_s=w['p_s'], T_r=w['T_r'], T_s=w['T_s'], u=w['u'], w_lg=w['w_lg'])
    return SSDFSimulator(wp, bc, n_cells=w['n_cells'])


def perturb(X):
    """A few per cent off the root, deterministic, alpha kept inside (0, 1)."""
    s = np.sin(np.arange(X.size).reshape(X.shape) + 1.0)
    Y = X * (1 + 0.05 * s)
    Y[:, 3] = np.clip(X[:, 3] + 0.02 * s[:, 3], 0.01, 0.99)
    return Y


def rows_at(sim, Y, i):
    n = sim.n_cells
    if i == 0:
        vals, ids = sim._left_boundary_eqs(list(Y[0])), list(ROW_IDS['first'])
    else:
        vals, ids = sim._differential_equations(list(Y[i]), list(Y[i - 1]), i), list(ROW_IDS['cell'])
        if i == n:
            vals, ids = vals + sim._right_boundary_eqs(list(Y[i])), ids + ROW_IDS['last']
    vals, ids = vals + sim._closure_relations(list(Y[i])), ids + ROW_IDS['closure']
    return [[k, num(v)] for k, v in zip(ids, vals)]


def row_vectors():
    wells = []
    for name, w in WELLS.items():
        sim = v1_simulator(w)
        X = np.array(sim.simulate()).reshape(-1, 7)
        Y = perturb(X)
        choked = bool(sim.wp.choke.is_choked(X[-1, 0], w['p_s']))
        wells.append(dict(name=name, params=w, choked_at_root=choked, root=X.tolist(), x=Y.tolist(),
                          rows=[rows_at(sim, Y, i) for i in range(w['n_cells'] + 1)]))
        print(f'{name}: p_0 = {X[0, 0]:.3f} bar, p_N = {X[-1, 0]:.3f} bar, choked at the root: {choked}')
    return dict(
        generated_by='specs/tools/make_v1_vectors.py',
        source=f'ManyWells v1.0.0 (casadi {ca.__version__}, numpy {np.__version__})',
        units='bar, K, kg/s, m, as in specs/model/nomenclature.md; row units in specs/model/discretization.md',
        state_order=STATE,
        note='x is the root perturbed by perturb(); rows are listed per grid point in DISC-6 order',
        wells=wells)


def main():
    write_blocks(component_tables())
    out = SPEC / 'vectors' / 'v1_rows.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(row_vectors(), indent=1) + '\n')
    print(f'wrote {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
