"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Test vectors for develop's options in specs/model/ (specs/model/README.md, "Test vectors").

They pin each option as specified and implemented, so that a later change or a port, such as the Rust core,
that alters a value is caught; they do not show that the implementation is right. Writes the tables into the
develop blocks of the spec files, between <!-- vectors:begin develop --> and <!-- vectors:end develop -->,
creating the block at the end of the file's "Test vectors" section if there is none. From the repository root:

    uv run python specs/tools/make_develop_vectors.py

TABLES is also the adapter that tests/test_spec_vectors.py checks the tables with: each entry's function is
the call of develop's code that the vectors pin.
"""

import pathlib
from dataclasses import dataclass

import casadi as ca
import numpy as np

from manywells.configurations import v1_fluid
from manywells.discretization import PointState
from manywells.friction import RoughnessFriction, chen_friction_factor, friction_factor, haaland_friction_factor
from manywells.pvt import density_from_api, gas_density_from_sg, mixture_viscosity
from manywells.pvt.black_oil import BlackOilPVT, live_oil_surface_tension, live_oil_viscosity
from manywells.pvt.dead_oil import dead_oil_viscosity
from manywells.pvt.fluid import FluidModel
from manywells.pvt.gas import gas_viscosity, sutton_pseudo_critical
from manywells.pvt.water import water_fvf, water_viscosity
from manywells.slip import SlipModel, classify_flow_regime
from manywells.thermal import ThermalModel
from manywells.units import CF_BAR, CF_RS, P_REF, T_REF

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = ROOT / 'specs' / 'model'
BEGIN, END = '<!-- vectors:begin develop -->', '<!-- vectors:end develop -->'


def num(x):
    return float(np.asarray(ca.DM(x)).item()) if isinstance(x, (ca.DM, ca.SX)) else float(x)


@dataclass(frozen=True)
class Table:
    file: str           # spec file, relative to specs/model
    heading: str        # the IDs it exercises
    inputs: tuple
    outputs: tuple
    cases: tuple        # input values, in the order of inputs
    fn: object          # fn(**inputs) -> outputs, in order

    def rows(self):
        for case in self.cases:
            kwargs = dict(zip(self.inputs, case))
            out = self.fn(**kwargs)
            yield kwargs, tuple(num(v) for v in (out if isinstance(out, tuple) else (out,)))


# ---------------------------------------------------------------------------------------------
# Develop's calls that the tables pin

def fluid(api=35.0, sg_gas=0.65, gor=150.0, wlr=0.0, **kw):
    return FluidModel(rho_o=density_from_api(api), rho_g=gas_density_from_sg(sg_gas), gor=gor, wlr=wlr, **kw)


def state(alpha, v_g, v_l, rho_g, rho_l, p=100.0, T=350.0):
    return PointState(p=p, v_g=v_g, v_l=v_l, alpha=alpha, rho_g=rho_g, rho_l=rho_l, T=T)


def thermal_term(alpha, rho_g, v_g, rho_l, v_l, F, cos_incl, cp_g, cp_l, frictional_heating, gravity_term):
    m = ThermalModel(h=0.0, frictional_heating=frictional_heating, gravity_term=gravity_term, lift_gas_mixing=False)
    fl = FluidModel(oil_model='dead_oil', cp_g=cp_g, cp_o=cp_l)
    return m.temperature_gradient(state(alpha, v_g, v_l, rho_g, rho_l), fl, T_a=350.0, F=F, dp_dmd=0.0,
                                  cos_incl=cos_incl, D=0.1)


def lift_gas_mixing(w_res, w_lg, T_r, T_lg, f_g, cp_g, cp_l):
    fl = v1_fluid(rho_l=850.0, R_s=420.0, cp_g=cp_g, cp_l=cp_l, f_g=f_g)
    return ThermalModel().inflow_temperature(w_res, w_lg, T_r, T_lg, fl)


def rough_pipe(p, T, alpha, v_g, v_l, rho_g, rho_l, D, roughness):
    s, fl = state(alpha, v_g, v_l, rho_g, rho_l, p, T), FluidModel()
    model = RoughnessFriction(roughness=roughness)
    return model.friction_factor(s, fl, D), model.pressure_gradient(s, fl, D)


def pseudo_critical_and_z(p, T, sg_gas):
    ppc, tpc = sutton_pseudo_critical(sg_gas)
    fl = FluidModel(rho_g=gas_density_from_sg(sg_gas))
    return ppc / CF_BAR, tpc, fl.z_factor(p, T)


def gas_parameters(rho_g_sc):
    fl = FluidModel(rho_g=rho_g_sc)
    return fl.sg_gas, fl.M_g, fl.R_s


def separator_gravity(api, sg_gas, p_sep, T_sep):
    return BlackOilPVT(api=api, sg_gas=sg_gas, p_sep=p_sep * CF_BAR, T_sep=T_sep).sg_gas_corr


def black_oil(api, sg_gas, p, T):
    fl = fluid(api, sg_gas)
    return fl.rs(p, T), fl.bo(p, T)


def fluid_parameters(rho_o, rho_g_sc, rho_w, gor, wlr, cp_o, cp_w):
    fl = FluidModel(rho_o=rho_o, rho_g=rho_g_sc, rho_w=rho_w, gor=gor, wlr=wlr, cp_o=cp_o, cp_w=cp_w)
    return fl.f_g, fl.rho_l, fl.cp_l, fl.f_o_in_liquid


# (v_g, v_l, alpha, rho_g, rho_l, sigma): bubbly, slug, annular, and near the bubbly-slug threshold
SLIP_STATES = ((1.2, 1.0, 0.15, 60.0, 780.0, 0.022), (6.0, 2.0, 0.4, 80.0, 800.0, 0.02),
               (25.0, 3.0, 0.85, 40.0, 820.0, 0.018), (3.0, 1.5, 0.27, 100.0, 750.0, 0.025))

TABLES = [
    # thermal.md
    Table('thermal.md', 'THM-4', ('tvd_frac', 'T_r', 'T_s'), ('T_a',),
          ((1.0, 370.0, 277.15), (0.0, 370.0, 277.15), (0.37, 360.0, 280.0), (0.9, 400.0, 270.0)),
          ThermalModel.ambient_temperature),
    Table('thermal.md', 'THM-5', ('w_res', 'w_lg', 'T_r', 'T_lg', 'f_g', 'cp_g', 'cp_l'), ('T_in',),
          ((10.0, 1.0, 373.15, 300.0, 0.1, 2225.0, 3000.0), (10.0, 0.0, 373.15, 300.0, 0.1, 2225.0, 3000.0),
           (2.0, 3.0, 380.0, 290.0, 0.3, 2225.0, 2500.0), (30.0, 0.5, 350.0, 360.0, 0.05, 2225.0, 4000.0)),
          lift_gas_mixing),
    Table('thermal.md', 'THM-6', ('alpha', 'rho_g', 'v_g', 'rho_l', 'v_l', 'F', 'cp_g', 'cp_l'), ('Phi_f',),
          ((0.0, 50.0, 10.0, 850.0, 2.0, 500.0, 2225.0, 4180.0), (0.4, 80.0, 6.0, 800.0, 2.5, 900.0, 2225.0, 3000.0),
           (0.9, 30.0, 20.0, 820.0, 4.0, 2000.0, 2225.0, 2500.0)),
          lambda F, **kw: thermal_term(F=F, cos_incl=1.0, frictional_heating=True, gravity_term=False, **kw)),
    Table('thermal.md', 'THM-7', ('alpha', 'rho_g', 'v_g', 'rho_l', 'v_l', 'cos_incl', 'cp_g', 'cp_l'), ('Phi_g',),
          ((1.0, 50.0, 10.0, 850.0, 2.0, 1.0, 2225.0, 4180.0), (0.4, 80.0, 6.0, 800.0, 2.5, 1.0, 2225.0, 3000.0),
           (0.6, 60.0, 8.0, 820.0, 3.0, 0.5, 2225.0, 2500.0), (0.5, 60.0, 8.0, 820.0, 3.0, 0.0, 2225.0, 2500.0)),
          lambda cos_incl, **kw: -thermal_term(F=0.0, cos_incl=cos_incl, frictional_heating=False, gravity_term=True,
                                               **kw)),
    # slip.md
    Table('slip.md', 'SLIP-10, SLIP-11', ('v_g', 'v_l', 'alpha', 'rho_g', 'rho_l', 'sigma', 'D', 'cos_incl'),
          ('C_0', 'v_inf'),
          tuple(s + (0.1524, c) for s in SLIP_STATES for c in (1.0, 0.7, 0.0)),
          lambda **kw: SlipModel().identify_parameters(**kw)),
    Table('slip.md', 'SLIP-11 (classifier)', ('v_g', 'v_l', 'alpha', 'rho_g', 'rho_l', 'sigma', 'cos_incl'),
          ('p_annular', 'p_slug', 'p_bubbly'),
          tuple(s + (c,) for s in SLIP_STATES for c in (0.7, 0.2)),
          lambda **kw: tuple(np.asarray(ca.DM(classify_flow_regime(**kw))).ravel())),
    # friction.md
    Table('friction.md', 'FRIC-3', ('p', 'T', 'alpha', 'v_g', 'v_l', 'rho_g', 'rho_l', 'D', 'roughness'),
          ('f_D', 'F'),
          ((150.0, 360.0, 0.2, 3.0, 2.0, 120.0, 780.0, 0.1524, 4.5e-5),
           (40.0, 320.0, 0.8, 15.0, 3.0, 30.0, 800.0, 0.1016, 1.5e-6),
           (200.0, 380.0, 0.05, 0.05, 0.02, 160.0, 760.0, 0.1524, 4.5e-5)),
          rough_pipe),
    Table('friction.md', 'FRIC-4', ('Re', 'eps_D'), ('f_t',),
          ((4.0e3, 1e-4), (1.0e5, 3e-4), (1.0e7, 1e-5), (5.0e5, 1e-2)), chen_friction_factor),
    Table('friction.md', 'FRIC-5', ('Re', 'eps_D'), ('f_t',),
          ((4.0e3, 1e-4), (1.0e5, 3e-4), (1.0e7, 1e-5), (5.0e5, 1e-2)), haaland_friction_factor),
    Table('friction.md', 'FRIC-6', ('Re', 'eps_D'), ('f_D',),
          ((0.5, 3e-4), (500.0, 3e-4), (2500.0, 3e-4), (3000.0, 3e-4), (4000.0, 3e-4), (1.0e5, 3e-4)),
          friction_factor),
    # pvt/gas.md
    Table('pvt/gas.md', 'PVT-GAS-3', ('p', 'T', 'rho_g_sc'), ('rho_g',),
          ((1.01325, 288.15, 0.8), (100.0, 350.0, 0.8), (250.0, 380.0, 0.68), (30.0, 290.0, 1.0)),
          lambda p, T, rho_g_sc: FluidModel(rho_g=rho_g_sc).gas_density(p, T)),
    Table('pvt/gas.md', 'PVT-GAS-4, PVT-GAS-5', ('p', 'T', 'sg_gas'), ('p_pc', 'T_pc', 'Z'),
          ((1.01325, 288.15, 0.554), (100.0, 350.0, 0.65), (250.0, 380.0, 0.75), (300.0, 330.0, 0.9)),
          pseudo_critical_and_z),
    Table('pvt/gas.md', 'PVT-GAS-6', ('rho_g_sc',), ('sg_gas', 'M_g', 'R_s'),
          ((0.6783,), (0.8,), (1.1,)), gas_parameters),
    Table('pvt/gas.md', 'PVT-GAS-7', ('T', 'rho_g', 'M_g'), ('mu_g',),
          ((288.15, 0.68, 16.04), (350.0, 80.0, 18.8), (380.0, 200.0, 22.0)), gas_viscosity),
    # pvt/oil.md
    Table('pvt/oil.md', 'PVT-OIL-5', ('api', 'sg_gas', 'p_sep', 'T_sep'), ('sg_gas_corr',),
          ((35.0, 0.65, P_REF / CF_BAR, T_REF), (25.0, 0.8, 7.0, 310.0), (40.0, 0.6, 20.0, 300.0)),
          separator_gravity),
    Table('pvt/oil.md', 'PVT-OIL-6, PVT-OIL-8', ('api', 'sg_gas', 'p', 'T'), ('R_so', 'B_o'),
          ((35.0, 0.65, 100.0, 350.0), (35.0, 0.65, 250.0, 380.0), (25.0, 0.8, 150.0, 340.0), (15.0, 0.7, 50.0, 320.0)),
          black_oil),
    Table('pvt/oil.md', 'PVT-OIL-7', ('api', 'sg_gas', 'p_bubble', 'p', 'T'), ('R_so',),
          ((35.0, 0.65, 150.0, 100.0, 350.0), (35.0, 0.65, 150.0, 150.0, 350.0), (35.0, 0.65, 150.0, 250.0, 350.0)),
          lambda api, sg_gas, p_bubble, p, T: fluid(api, sg_gas, p_bubble=p_bubble).rs(p, T)),
    Table('pvt/oil.md', 'PVT-OIL-9', ('api', 'sg_gas', 'p', 'T'), ('rho_lo',),
          ((35.0, 0.65, 100.0, 350.0), (25.0, 0.8, 250.0, 380.0)),
          lambda api, sg_gas, p, T: fluid(api, sg_gas).liquid_density(p, T)),
    Table('pvt/oil.md', 'PVT-OIL-10', ('api', 'T'), ('mu_od',),
          ((35.0, 320.0), (25.0, 350.0), (15.0, 380.0)), dead_oil_viscosity),
    Table('pvt/oil.md', 'PVT-OIL-11', ('mu_od', 'R_so_scf'), ('mu_o',),
          ((5e-3, 0.0), (5e-3, 500.0), (2e-2, 1000.0)), lambda mu_od, R_so_scf: live_oil_viscosity(mu_od, R_so_scf)),
    Table('pvt/oil.md', 'PVT-OIL-12', ('sigma_od', 'R_so_scf'), ('sigma_lo',),
          tuple((0.03, r / CF_RS) for r in (0.0, 20.0, 45.0, 50.0, 55.0, 150.0)),
          lambda sigma_od, R_so_scf: live_oil_surface_tension(sigma_od, R_so_scf)),
    Table('pvt/oil.md', 'PVT-OIL-13', ('api', 'sg_gas', 'gor', 'wlr', 'p', 'T', 'w_res', 'w_lg'), ('w_g', 'w_l'),
          ((35.0, 0.65, 150.0, 0.0, 100.0, 350.0, 20.0, 0.0), (35.0, 0.65, 150.0, 0.3, 250.0, 380.0, 20.0, 2.0),
           (35.0, 0.65, 20.0, 0.0, 250.0, 380.0, 20.0, 1.0), (25.0, 0.8, 300.0, 0.5, 30.0, 320.0, 5.0, 0.0)),
          lambda p, T, w_res, w_lg, **kw: fluid(**kw).phase_rates(p, T, w_res, w_lg)),
    # pvt/water.md
    Table('pvt/water.md', 'PVT-WAT-2', ('p_Pa', 'T'), ('B_w',),
          ((P_REF, 288.15), (200e5, 350.0)), lambda p_Pa, T: water_fvf(p_Pa, T)),
    Table('pvt/water.md', 'PVT-WAT-3', ('T',), ('mu_w',), ((293.15,), (350.0,), (400.0,)), water_viscosity),
    # pvt/mixture.md
    Table('pvt/mixture.md', 'PVT-MIX-6', ('api', 'sg_gas', 'gor', 'wlr', 'p', 'T'), ('rho_l',),
          ((35.0, 0.65, 150.0, 0.0, 100.0, 350.0), (35.0, 0.65, 150.0, 0.4, 250.0, 380.0)),
          lambda p, T, **kw: fluid(**kw).liquid_density(p, T)),
    Table('pvt/mixture.md', 'PVT-MIX-7', ('api', 'sg_gas', 'p', 'T', 'rho_l'), ('sigma',),
          ((35.0, 0.65, 100.0, 350.0, 700.0), (35.0, 0.65, 250.0, 380.0, 700.0), (25.0, 0.8, 30.0, 300.0, 900.0)),
          lambda api, sg_gas, p, T, rho_l: fluid(api, sg_gas).surface_tension(p, T, rho_l)),
    Table('pvt/mixture.md', 'PVT-MIX-8', ('api', 'sg_gas', 'wlr', 'p', 'T'), ('mu_l',),
          ((35.0, 0.65, 0.0, 100.0, 350.0), (35.0, 0.65, 0.5, 250.0, 380.0)),
          lambda api, sg_gas, wlr, p, T: fluid(api, sg_gas, wlr=wlr).liquid_viscosity(p, T)),
    Table('pvt/mixture.md', 'PVT-MIX-9', ('mu_l', 'mu_g', 'alpha', 'rho_l', 'rho_g'), ('mu_m',),
          ((1e-3, 1.5e-5, 0.0, 800.0, 50.0), (1e-3, 1.5e-5, 0.5, 800.0, 50.0), (2e-3, 2e-5, 0.9, 850.0, 100.0)),
          mixture_viscosity),
    Table('pvt/mixture.md', 'PVT-MIX-10', ('rho_o', 'rho_g_sc', 'rho_w', 'gor', 'wlr', 'cp_o', 'cp_w'),
          ('f_g', 'rho_l_sc', 'cp_l', 'x_o'),
          ((850.0, 0.8, 999.1, 150.0, 0.0, 2000.0, 4184.0), (870.0, 0.75, 1020.0, 300.0, 0.4, 2100.0, 4000.0)),
          fluid_parameters),
]


def markdown(table):
    cols = list(table.inputs) + [f'→ {o}' for o in table.outputs]
    lines = [f'### {table.heading}', '', '| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for kwargs, out in table.rows():
        lines.append('| ' + ' | '.join(repr(float(kwargs[k])) for k in table.inputs) + ' | '
                     + ' | '.join(repr(v) for v in out) + ' |')
    return '\n'.join(lines)


def main():
    header = ('Generated by `specs/tools/make_develop_vectors.py` from develop '
              f'(casadi {ca.__version__}): they pin develop\'s options. Do not edit by hand.')
    files = {}
    for t in TABLES:
        files.setdefault(t.file, []).append(markdown(t))
    for rel, blocks in files.items():
        path = SPEC / rel
        text = path.read_text()
        body = BEGIN + '\n' + '\n\n'.join([header] + blocks) + '\n' + END
        if BEGIN in text:
            start, end = text.index(BEGIN), text.index(END) + len(END)
            text = text[:start] + body + text[end:]
        else:
            k = text.index('\n## Coverage')
            text = text[:k].rstrip('\n') + '\n\n' + body + '\n' + text[k:]
        path.write_text(text)
        print(f'wrote {len(blocks)} tables to {path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
