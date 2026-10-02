"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Calibrate a well to its production data (specs/calibration.md, docs/calibration.md).

The "measured" data are synthetic: the well is simulated at parameters the calibration does not know, with
measurement noise, and with the instruments of a typical well: wellhead pressure and temperature on every row, a
downhole gauge, and rates from a well test on one row in five. The calibration starts from the priors' medians and
estimates the choke coefficient, Vogel's maximum rate, the pipe roughness and the heat-transfer coefficient; it then
predicts rows it has not seen.
"""

import matplotlib.pyplot as plt

from manywells.calibration import calibrate, evaluate, synthetic_data
from manywells.choke import SimpsonChokeModel
from manywells.geometry import WellGeometry
from manywells.inflow import Vogel
from manywells.simulator import BoundaryConditions, WellProperties

# -- The well as the user knows it ----------------------------------------
geometry = WellGeometry.vertical(length=2500, n_cells=50, D=0.1)
wp = WellProperties(geometry=geometry, inflow=Vogel(w_l_max=60.0),
                    choke=SimpsonChokeModel(K_c=0.12 * geometry.A, chk_profile='linear'))
bc = BoundaryConditions(p_r=250.0, p_s=20.0, T_r=360.0, T_s=280.0)

# -- The "measured" data: the true well differs from what the user knows --
truth = {'K_c': 0.15 * geometry.A, 'w_l_max': 45.0, 'roughness': 1.5e-4, 'h': 22.0}
data = synthetic_data(wp, bc, truth, n_rows=20, seed=1, instrumentation='periodic_tests')
print(data.rows[['CHK', 'PDC', 'PBH', 'PWH', 'TWH', 'QOIL', 'QGAS']].round(2).head(), '\n')

# -- Calibrate ------------------------------------------------------------
result = calibrate(wp, data, bc, free=['K_c', 'w_l_max', 'roughness', 'h'])
print(result.summary(), '\n')
for name, value in truth.items():
    print(f'{name:10s} true {value:10.4g}   calibrated {result.values[name]:10.4g}')

# -- Predict rows the calibration has not seen ----------------------------
new = synthetic_data(wp, bc, truth, n_rows=10, seed=2, instrumentation='full')
table = evaluate(result.well, new, bc)
before = evaluate(wp, new, bc)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (obs, label) in zip(axes, [('PWH', 'Wellhead pressure (bar)'), ('PBH', 'Bottomhole pressure (bar)'),
                                   ('WRES', 'Reservoir mass rate (kg/s)')]):
    ax.plot(table['CHK'], table[f'{obs}_obs'], 'ko', label='measured')
    ax.plot(before['CHK'], before[f'{obs}_pred'], 'x', color='tab:red', label='before calibration')
    ax.plot(table['CHK'], table[f'{obs}_pred'], '+', color='tab:blue', ms=10, label='calibrated')
    ax.set_xlabel('Choke position')
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
axes[0].legend()
fig.suptitle('Calibrated well on rows it has not seen', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
