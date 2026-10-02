"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 02 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

A Joule-Thomson term in the energy balance on wells drawn by develop's sampler (feature spec
specs/features/016-joule-thomson.md). A prototype: the term is added by a subclass of ThermalModel, so it runs on
the CasADi backend only. Each well (seed 2026) is solved at its nominal operating point and at two sampled ones,
in develop's default configuration and with each variant of the term:

  base       develop's default model, without the term
  dak_sub    Phi_JT = alpha v_g J (F + rho_m g cos) / C with DAK's J at the state (rho_g, T): the proposal
  pap_sub    the same with J from Papay's z-factor, the gas law's own
  pap_dp     alpha v_g J (-dp/dMD) / C with Papay's J: the cell's pressure gradient instead of F + rho_m g cos

C is the heat-capacity flux of THM-1. Each variant starts from the base operating point. Per case it records
PBH, PWH, TWH, the bottomhole's reduced pressure, the minimum over points i >= 1 of T_i - T_a,i (heat flows
outwards where it is positive) and min T - T_s (the CasADi backend's lower temperature bound is min(T_s, T_lg)).

From the repository root, one BLAS thread per process (about 20 minutes on 24 cores for 88 wells):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python plans/evidence/jt_wells.py 88 jt_wells.csv
"""

import sys
import time
from dataclasses import dataclass, replace
from multiprocessing import Pool

import casadi as ca
import numpy as np
import pandas as pd

from manywells.configurations import DEVELOP
from manywells.pvt.gas import sutton_pseudo_critical
from manywells.sampling.conditions import nominal_conditions, sample_conditions
from manywells.sampling.wells import rng_for, sample_well, well_properties
from manywells.simulator import SimError, SSDFSimulator
from manywells.thermal import ThermalModel
from manywells.units import CF_BAR, STD_GRAVITY

SEED = 2026
MODES = ('dak_sub', 'pap_sub', 'pap_dp')
A = (0.3265, -1.0700, -0.5339, 0.01569, -0.05165, 0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210)


def j_papay(fluid, s):
    """T (d ln Z / dT)_p of Papay's z-factor (PVT-GAS-4) at the state's p and T."""
    ppc, tpc = sutton_pseudo_critical(fluid.sg_gas)
    ppr, tpr = s.p * CF_BAR / ppc, s.T / tpc
    a = 3.52 * ppr * ca.power(10, -0.9813 * tpr)
    b = 0.274 * ppr ** 2 * ca.power(10, -0.8157 * tpr)
    return tpr * np.log(10) * (0.9813 * a - 0.8157 * b) / (1 - a + b)


def j_dak(fluid, s):
    """The proposed PVT-GAS-10: J = (t Z_t - r Z_r) / (Z + r Z_r) of DAK at the state's reduced density."""
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = A
    ppc, tpc = sutton_pseudo_critical(fluid.sg_gas)
    r = 0.27 * s.rho_g * fluid.R_s * tpc / ppc
    t = s.T / tpc
    c1 = a1 + a2 / t + a3 / t ** 3 + a4 / t ** 4 + a5 / t ** 5
    c2 = a6 + a7 / t + a8 / t ** 2
    c3 = a9 * (a7 / t + a8 / t ** 2)
    c4 = a10 / t ** 3
    e = ca.exp(-a11 * r ** 2)
    Z = 1 + c1 * r + c2 * r ** 2 - c3 * r ** 5 + c4 * r ** 2 * (1 + a11 * r ** 2) * e
    Z_r = c1 + 2 * c2 * r - 5 * c3 * r ** 4 + 2 * c4 * r * e * (1 + a11 * r ** 2 - a11 ** 2 * r ** 4)
    tZ_t = ((-a2 / t - 3 * a3 / t ** 3 - 4 * a4 / t ** 4 - 5 * a5 / t ** 5) * r
            + (-a7 / t - 2 * a8 / t ** 2) * r ** 2
            - a9 * (-a7 / t - 2 * a8 / t ** 2) * r ** 5
            - 3 * a10 / t ** 3 * r ** 2 * (1 + a11 * r ** 2) * e)
    return (tZ_t - r * Z_r) / (Z + r * Z_r)


@dataclass(frozen=True)
class JouleThomsonPrototype(ThermalModel):
    mode: str = 'dak_sub'

    def temperature_gradient(self, s, fluid, T_a, F, dp_dmd, cos_incl, D):
        dT = super().temperature_gradient(s, fluid, T_a, F, dp_dmd, cos_incl, D)
        C = fluid.cp_g * s.alpha * s.rho_g * s.v_g + fluid.cp_l * (1 - s.alpha) * s.rho_l * s.v_l
        if self.mode == 'dak_sub':
            return dT - s.alpha * s.v_g * j_dak(fluid, s) * (F + s.rho_m * STD_GRAVITY * cos_incl) / C
        if self.mode == 'pap_sub':
            return dT - s.alpha * s.v_g * j_papay(fluid, s) * (F + s.rho_m * STD_GRAVITY * cos_incl) / C
        if self.mode == 'pap_dp':
            return dT + s.alpha * s.v_g * j_papay(fluid, s) * dp_dmd / C
        raise ValueError(self.mode)


def outputs(sim, root, bc):
    T = root.state[:, 6]
    T_a = np.array([sim.wp.thermal.ambient_temperature(f, bc.T_r, bc.T_s) for f in sim.wp.geometry.tvd_frac])
    return dict(PBH=root.state[0, 0], PWH=root.state[-1, 0], TWH=T[-1], alpha_wh=root.state[-1, 3],
                min_T_minus_Ta=float(np.min(T[1:] - T_a[1:])), min_T_minus_Ts=float(np.min(T) - bc.T_s))


def well_cases(well):
    rows = []
    draw = sample_well(SEED, well)
    rng = rng_for(SEED, well, 'jt')
    for k in range(3):
        bc, fractions = (nominal_conditions(draw), draw.fractions) if k == 0 else sample_conditions(draw, rng)
        try:
            wp = well_properties(draw, fractions, DEVELOP)
        except ValueError:  # no oil to give a gas-oil ratio (SMP-43)
            continue
        ppc, _ = sutton_pseudo_critical(wp.fluid.sg_gas)
        rec = dict(well=well, k=k, trajectory=draw.trajectory[0], f_g=fractions[0], w_lg=bc.w_lg, p_r=bc.p_r,
                   p_s=bc.p_s, T_r=bc.T_r, u=bc.u, h=draw.h, sg_gas=wp.fluid.sg_gas, p_pc=ppc / CF_BAR)
        t0 = time.time()
        try:
            sim = SSDFSimulator(wp)
            base = sim.simulate(bc)
            rec.update({f'{key}_base': v for key, v in outputs(sim, base, bc).items()})
        except SimError as e:
            rec['err_base'] = type(e).__name__
            rows.append(rec)
            continue
        for mode in MODES:
            try:
                sim = SSDFSimulator(replace(wp, thermal=JouleThomsonPrototype(h=wp.thermal.h, mode=mode)))
                op = sim.simulate(bc, x_guess=base.x)
                rec.update({f'{key}_{mode}': v for key, v in outputs(sim, op, bc).items()})
            except SimError as e:
                rec[f'err_{mode}'] = type(e).__name__
        rec['seconds'] = time.time() - t0
        rows.append(rec)
    return rows


def summary(df):
    q = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
    base = df[df.TWH_base.notna()]
    print(f'{len(df)} cases, {len(base)} with a base operating point')
    for m in MODES:
        print(f'  {m}: operating point in {base[f"TWH_{m}"].notna().sum()} of them')
    ok = base.dropna(subset=[f'TWH_{m}' for m in MODES]).copy()
    print(f'{len(ok)} cases with an operating point in every variant')
    diff = pd.DataFrame({
        'TWH dak': ok.TWH_dak_sub - ok.TWH_base, 'PWH dak': ok.PWH_dak_sub - ok.PWH_base,
        'PBH dak': ok.PBH_dak_sub - ok.PBH_base,
        'TWH dak-pap': ok.TWH_dak_sub - ok.TWH_pap_sub, 'PWH dak-pap': ok.PWH_dak_sub - ok.PWH_pap_sub,
        'TWH sub-dp': ok.TWH_pap_sub - ok.TWH_pap_dp, 'PWH sub-dp': ok.PWH_pap_sub - ok.PWH_pap_dp})
    print('Quantiles of the differences (K, bar):')
    print(diff.quantile(q).round(2).to_string())
    print(f'Share with |TWH dak - pap| > 1 K: {(diff["TWH dak-pap"].abs() > 1).mean():.3f}')
    ok['ppr_bh'] = ok.PBH_base * CF_BAR / (ok.p_pc * CF_BAR)
    ok['f_g bin'] = pd.cut(ok.f_g, [0, 0.05, 0.2, 0.5, 1.0])
    ok['p_pr,bh bin'] = pd.cut(ok.ppr_bh, [0, 3, 5, 7, 12])
    ok['TWH dak'], ok['TWH dak-pap'] = diff['TWH dak'], diff['TWH dak-pap']
    print(ok.groupby('f_g bin', observed=True)['TWH dak'].describe()[['count', 'min', '50%', 'max']].round(2))
    print(ok.groupby('p_pr,bh bin', observed=True)['TWH dak-pap'].describe()[['count', 'min', '50%', 'max']].round(2))
    print(f'Bottomhole p_pr from {ok.ppr_bh.min():.2f} to {ok.ppr_bh.max():.2f}; above 6 in {(ok.ppr_bh > 6).mean():.3f}')
    for m in ('base',) + MODES:
        c = ok[f'min_T_minus_Ta_{m}']
        print(f'{m}: colder than ambient somewhere in {(c < 0).sum()} cases (min {c.min():.2f} K); '
              f'min T - T_s {ok[f"min_T_minus_Ts_{m}"].min():.2f} K')


def main(n_wells, out):
    with Pool(22) as pool:
        rows = [r for rs in pool.map(well_cases, range(n_wells), chunksize=1) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    pd.set_option('display.width', 200)
    summary(df)


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2])
