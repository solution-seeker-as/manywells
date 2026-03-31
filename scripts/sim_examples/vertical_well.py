"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Simulate a vertical well.
"""

import numpy as np
import matplotlib.pyplot as plt
from manywells.simulator import WellProperties, BoundaryConditions, SSDFSimulator


###########################################################################
# Create a new well (using default values)
###########################################################################
well_properties = WellProperties()
boundary_conditions = BoundaryConditions(u=0.5)
sim = SSDFSimulator(well_properties, boundary_conditions)

###########################################################################
# Simulate
###########################################################################
x = sim.simulate()

# Convert solution to DataFrame
df = sim.solution_as_df(x)

###########################################################################
# Plot results
###########################################################################

A = well_properties.geometry.A
df['w_g'] = A * df['alpha'] * df['rho_g'] * df['v_g']
df['w_l'] = A * (1 - df['alpha']) * df['rho_l'] * df['v_l']

panels = [
    ('tvd', 'Well trajectory: TVD (m)'),
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

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
for ax, (col, ylabel) in zip(axes.flat, panels):
    if col == 'tvd':
        md_arr = np.array(well_properties.geometry.md_survey)
        tvd_arr = np.array(well_properties.geometry.tvd_survey)
        d_h = np.sqrt(np.maximum(np.diff(md_arr)**2 - np.diff(tvd_arr)**2, 0.0))
        horiz = np.concatenate(([0.0], np.cumsum(d_h)))
        ax.plot(horiz, tvd_arr, 'k-', lw=2)
        ax.invert_yaxis()
        ax.set_xlabel('Horizontal displacement (m)')
    else:
        ax.plot(df['md'], df[col], lw=2)
        ax.set_xlabel('MD (m)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

fig.suptitle('Vertical well simulation', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
