"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 06 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

A gas well with Forchheimer's gas inflow (INF-10 and INF-11 in specs/model/inflow.md; specs/features/022-gas-rate-inflow.md).

The inflow gives the gas rate, p_r^2 - p_wf^2 = a q + b q^2, and the condensate and water come with the gas in the
fluid's ratios. A well test gives a and b for the separator gas's standard volume rate q in Sm3/d, which
Forchheimer.from_gas_rate converts to the model's kg/s with the fluid's gas density. The fluid holds a gas well's
condensate-gas and water-gas ratios as gor = 1/CGR and wlr = WGR / (WGR + CGR).

The script solves the well at several choke openings and plots the operating points on the inflow's curve. Then it
solves the same well late in its life, at a lower reservoir pressure with more water, where the liquid held up in
the tubing gives it a second, unstable root at a lower rate (liquid loading), and where lift gas lets it flow when it
cannot flow on its own.
"""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from manywells.geometry import WellGeometry
from manywells.inflow import Forchheimer
from manywells.pvt import density_from_api, gas_density_from_sg
from manywells.pvt.black_oil import BlackOil
from manywells.pvt.fluid import FluidModel
from manywells.simulator import BoundaryConditions, SSDFSimulator, WellProperties
from scripts.plt_config import set_seaborn_config

set_seaborn_config()

# -- Well, fluid and inflow ---------------------------------------------
CGR, WGR = 50e-6, 30e-6  # Condensate-gas and water-gas ratios (Sm3/Sm3): 50 and 30 Sm3 per 1e6 Sm3 of gas
pvt = BlackOil(rho_o=density_from_api(55.0), rho_g=gas_density_from_sg(0.65))
fluid = FluidModel(pvt=pvt, gor=1 / CGR, wlr=WGR / (WGR + CGR))

# From a four-point test: p_r^2 - p_wf^2 = a_q q + b_q q^2, in bar2 per Sm3/d and per (Sm3/d)^2 of separator gas
a_q, b_q = 0.032, 7.0e-8
inflow = Forchheimer.from_gas_rate(a_q, b_q, fluid.rho_g)

geometry = WellGeometry.from_survey([0.0, 1000.0, 3700.0], [0.0, 1000.0, 3200.0], n_cells=80, D=0.1)
wp = WellProperties(geometry=geometry, fluid=fluid, inflow=inflow)
bc = BoundaryConditions(p_r=270.0, p_s=40.0, T_r=380.0, T_s=277.15)


def gas_rate_sm3d(w_g):
    """Standard volume rate (1e6 Sm3/d) of a gas mass rate (kg/s)."""
    return w_g * 86400 / fluid.rho_g / 1e6


# -- Operating points at several choke openings (the Rust core: the same model, faster) --------------------------
sim = SSDFSimulator(wp, backend='rust')
chokes = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
points = []
for u in chokes:
    op = sim.simulate(replace(bc, u=u))
    T_wh = op.state[-1, 6] - 273.15
    points.append((u, op.p_0, op.state[-1, 0], gas_rate_sm3d(op.w_g_res), op.w_res, T_wh, op.choked))
    print(
        f'u = {u:.1f}: PBH {op.p_0:6.1f} bar, PWH {op.state[-1, 0]:6.1f} bar, TWH {T_wh:5.1f} °C, '
        f'gas {gas_rate_sm3d(op.w_g_res):.3f} 1e6 Sm3/d, liquid {op.w_res:.3f} kg/s, choked: {op.choked}'
    )
points = np.array(points, dtype=float)

# -- Late in life: liquid loading, and lift gas -------------------------------------------------------------------
late = replace(wp, fluid=replace(fluid, wlr=400e-6 / (400e-6 + CGR)))  # WGR 400 Sm3 per 1e6 Sm3
sim_late = SSDFSimulator(late, backend='rust')
for p_r, w_lg in ((90.0, 0.0), (40.0, 0.0), (40.0, 0.3)):
    rs = sim_late.root_set(replace(bc, p_r=p_r, p_s=15.0, u=1.0, w_lg=w_lg))
    roots = ', '.join(f'{r.label} at {r.p_0:.1f} bar ({gas_rate_sm3d(r.w_g_res):.3f} 1e6 Sm3/d)' for r in rs.roots)
    print(f'late in life, p_r = {p_r:.0f} bar, lift gas {w_lg} kg/s: {roots or "no root: the well cannot flow"}')

# -- Plot -----------------------------------------------------------------------------------------------------------
p_wf = np.linspace(1.01325, bc.p_r, 200)
ipr = gas_rate_sm3d(np.array([inflow.gas_mass_flow_rate(p, bc.p_r) for p in p_wf]))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(ipr, p_wf, label="Forchheimer's inflow")
axes[0].plot(points[:, 3], points[:, 1], 'o', label='operating points')
for u, p_0, q in zip(points[:, 0], points[:, 1], points[:, 3]):
    axes[0].annotate(f'u = {u:.1f}', (q, p_0), textcoords='offset points', xytext=(6, 4), fontsize=9)
axes[0].set_xlabel('Gas rate (1e6 Sm³/d)')
axes[0].set_ylabel('Bottomhole pressure (bar)')
axes[0].legend()
axes[1].plot(points[:, 0], points[:, 2], 'o-', label='wellhead pressure (bar)')
axes[1].plot(points[:, 0], points[:, 5], 's-', label='wellhead temperature (°C)')
axes[1].set_xlabel('Choke opening u')
axes[1].legend()
for ax in axes:
    ax.grid(True, alpha=0.3)
fig.suptitle(f'A gas well with Forchheimer inflow ($p_r$ = {bc.p_r:.0f} bar, CGR 50, WGR 30 Sm³ per 10⁶ Sm³)')
fig.tight_layout()
plt.savefig('gas_well.png', dpi=150, bbox_inches='tight')
plt.show()
