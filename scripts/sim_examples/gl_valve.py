"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 29 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Gas lift through a valve: the same well and lift-gas rates with the lift gas injected at the bottomhole and at a
valve at 75% and 50% of the measured depth (WellProperties.injection_md, INF-9 in specs/model/inflow.md).

Below the valve the tubing carries the reservoir fluid alone, so the column there is heavier than with the lift gas at
the bottomhole, and the same lift-gas rate lifts less. The injection depth belongs to the well, so each valve depth
has its own simulator. The lift gas enters at its own temperature, here colder than the flow, and mixes with the flow
at the valve (THM-9).
"""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from manywells.geometry import WellGeometry
from manywells.inflow import ProductivityIndex
from manywells.pvt.fluid import FluidModel
from manywells.simulator import BoundaryConditions, SimError, SSDFSimulator, WellProperties
from scripts.plt_config import set_seaborn_config

set_seaborn_config()

# -- Well and operating point -----------------------------------------
L = 2500.0  # Measured depth (m), vertical
geometry = WellGeometry.vertical(length=L, n_cells=60)
fluid = FluidModel(wlr=0.5)
inflow = ProductivityIndex(k_l=1.0)
bc = BoundaryConditions(p_r=160.0, p_s=20.0, T_r=363.15, T_s=277.15, u=0.8, T_lg=300.0)

valves = {'bottomhole': None, '75% of MD': 0.75 * L, '50% of MD': 0.5 * L}
w_lg_values = np.linspace(0.5, 3.0, 6)  # Lift-gas rates (kg/s)

# -- Run simulations --------------------------------------------------
results = {}
for name, injection_md in valves.items():
    wp = WellProperties(geometry=geometry, fluid=fluid, inflow=inflow, injection_md=injection_md)
    sim = SSDFSimulator(wp)  # One simulator per valve depth: the injection point is part of the well's system
    rows = []
    for w_lg in w_lg_values:
        try:
            op = sim.simulate(replace(bc, w_lg=float(w_lg)))
        except SimError as e:
            print(f'{name}, w_lg = {w_lg:.2f} kg/s: {e}')
            continue
        df = sim.solution_as_df(op)
        rows.append((w_lg, op.p_0, op.w_res, df['T'].iloc[-1] - 273.15))
    results[name] = np.array(rows)
    print(f'{name}: injection point {wp.injection_point}, PBH at the largest rate {rows[-1][1]:.1f} bar')

# -- Plot results -----------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
for name, r in results.items():
    if len(r):
        axes[0].plot(r[:, 0], r[:, 1], 'o-', markersize=4, label=name)
        axes[1].plot(r[:, 0], r[:, 2], 'o-', markersize=4, label=name)
        axes[2].plot(r[:, 0], r[:, 3], 'o-', markersize=4, label=name)
axes[0].set_ylabel('Bottomhole pressure\n(bar)')
axes[1].set_ylabel('Reservoir liquid rate\n(kg/s)')
axes[2].set_ylabel('Wellhead temperature\n(°C)')
axes[2].set_xlabel('Lift-gas rate (kg/s)')
axes[0].legend(title='Lift gas injected at')
for ax in axes:
    ax.grid(True, alpha=0.3)
fig.suptitle(
    f'Gas lift through a valve (choke {bc.u * 100:.0f}%, $p_r$={bc.p_r:.0f} bar, $T_{{lg}}$={bc.T_lg - 273.15:.0f} °C)'
)
fig.tight_layout()
plt.savefig('gl_valve.png', dpi=150, bbox_inches='tight')
plt.show()
