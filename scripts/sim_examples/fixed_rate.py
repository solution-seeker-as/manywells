"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Simulate a vertical well with a fixed liquid flow rate (FixedFlowRate inflow model).

The FixedFlowRate model decouples inflow from drawdown, so the bottom-hole
pressure adjusts to sustain the prescribed liquid rate.  This is useful for
calibration or for studying wellbore hydraulics at a known production rate.
"""

import numpy as np
import matplotlib.pyplot as plt

from manywells.simulator import WellProperties, BoundaryConditions, SSDFSimulator
from manywells.inflow import FixedFlowRate

# -- Well configuration ---------------------------------------------------
W_L = 5.0  # Fixed liquid mass flow rate (kg/s)
wp = WellProperties(inflow=FixedFlowRate(w_l_const=W_L))
bc = BoundaryConditions(u=0.5)

# -- Simulate -------------------------------------------------------------
sim = SSDFSimulator(wp, bc)
x = sim.simulate()
df = sim.solution_as_df(x)

# -- Derived quantities ---------------------------------------------------
A = wp.geometry.A
df['w_g'] = A * df['alpha'] * df['rho_g'] * df['v_g']
df['w_l'] = A * (1 - df['alpha']) * df['rho_l'] * df['v_l']

print(f"Fixed liquid rate : {W_L:.2f} kg/s")
print(f"Bottom-hole pressure: {df['p'].iloc[0]:.2f} bar")
print(f"Wellhead pressure   : {df['p'].iloc[-1]:.2f} bar")
print(f"Wellhead temperature: {df['T'].iloc[-1] - 273.15:.1f} °C")

# -- Plot results ---------------------------------------------------------
panels = [
    ('p', 'Pressure (bar)'),
    ('w_g', 'Gas mass flow (kg/s)'),
    ('v_g', 'Gas velocity (m/s)'),
    ('rho_g', 'Gas density (kg/m³)'),
    ('alpha', 'Void fraction'),
    ('T', 'Temperature (K)'),
    ('w_l', 'Liquid mass flow (kg/s)'),
    ('v_l', 'Liquid velocity (m/s)'),
    ('rho_l', 'Liquid density (kg/m³)'),
]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
for ax, (col, ylabel) in zip(axes.flat, panels):
    ax.plot(df['md'], df[col], lw=2)
    ax.set_xlabel('MD (m)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    f'Fixed flow rate simulation  ($w_l$ = {W_L:.1f} kg/s)',
    fontsize=14, fontweight='bold',
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
