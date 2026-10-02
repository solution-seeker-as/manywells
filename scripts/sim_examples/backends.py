"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Simulate a well with each backend and compare the answers.

SSDFSimulator(wp, backend=...) solves the same model with one of two backends:

- 'casadi' (the default): builds the well's system as a CasADi graph and searches for roots with Ipopt from
  several starts (manywells.solvers.roots).
- 'rust': the Rust core (manywells._core) marches the rows from the bottomhole to the wellhead and searches the
  bottomhole pressure for the roots of the choke row (manywells.solvers.rust). It is much faster.

Both take the same inputs and return the same types (Root, RootSet), so the rest of a script does not depend on
the backend. The Rust core has a closed set of component classes, so a well with a user's own component (here, a
subclass of an inflow model) needs the CasADi backend.

The operating point below has two roots, a stable one (the operating point) and an unstable one.
"""

import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from manywells.inflow import ProductivityIndex
from manywells.simulator import WellProperties, BoundaryConditions, SSDFSimulator, STATE
from manywells.solvers.rust import not_in_core

BACKENDS = ('casadi', 'rust')

# -- Well and operating point --------------------------------------------
wp = WellProperties()
bc = BoundaryConditions(p_r=130, u=0.3)

# -- Solve with each backend ----------------------------------------------
sims, root_sets = {}, {}
for backend in BACKENDS:
    t0 = time.perf_counter()
    sims[backend] = SSDFSimulator(wp, backend=backend)  # builds the well's system once
    t1 = time.perf_counter()
    root_sets[backend] = sims[backend].root_set(bc)     # every root, labelled; simulate(bc) returns the stable one
    t2 = time.perf_counter()

    rs = root_sets[backend]
    print(f'{backend}: build {t1 - t0:.2f} s, search {t2 - t1:.2f} s')
    for r in rs.roots:
        print(f'    p_0 = {r.p_0:7.2f} bar  {r.label:<13}  choked: {r.choked}')
    print(f'    operating point: p_0 = {rs.operating_point.p_0:.2f} bar')

# -- Compare the root sets ----------------------------------------------
# Both backends sort the roots by bottomhole pressure, so roots with the same index are the same root
rs_c, rs_r = root_sets['casadi'], root_sets['rust']
if len(rs_c.roots) != len(rs_r.roots):
    print(f'The backends found {len(rs_c.roots)} and {len(rs_r.roots)} roots')
for k, (r_c, r_r) in enumerate(zip(rs_c.roots, rs_r.roots)):
    diff = np.max(np.abs(r_c.state - r_r.state), axis=0)  # largest difference over the grid, per state variable
    print(f'Root {k} ({r_c.label} / {r_r.label}), largest difference:',
          ', '.join(f'{name} {d:.1e}' for name, d in zip(STATE, diff)))

# -- Plot every root from both backends ----------------------------------
panels = [
    ('p', 'Pressure (bar)'),
    ('alpha', 'Void fraction'),
    ('v_g', 'Gas velocity (m/s)'),
    ('T', 'Temperature (K)'),
]
colors = {'stable': 'tab:blue', 'unstable': 'tab:red', 'indeterminate': 'tab:gray'}

fig, axes = plt.subplots(1, len(panels), figsize=(18, 5))
for ax, (col, ylabel) in zip(axes, panels):
    for r in rs_c.roots:
        df = sims['casadi'].solution_as_df(r)
        ax.plot(df['md'], df[col], '-', lw=2, color=colors[r.label], label=f'casadi, {r.label}')
    for r in rs_r.roots:
        df = sims['rust'].solution_as_df(r)
        ax.plot(df['md'], df[col], 'o', ms=4, markevery=5, mfc='none', color=colors[r.label],
                label=f'rust, {r.label}')
    ax.set_xlabel('MD (m)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
axes[0].legend()

fig.suptitle('The same well solved by each backend: lines are CasADi, circles the Rust core',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])


# -- A well with a user's component --------------------------------------
@dataclass(frozen=True)
class DamagedProductivityIndex(ProductivityIndex):
    """A user's inflow model: the productivity index, reduced by near-wellbore damage."""
    damage: float = 0.2  # Fraction of the inflow lost (-)

    def liquid_mass_flow_rate(self, p, p_r):
        return (1 - self.damage) * super().liquid_mass_flow_rate(p, p_r)


wp_user = WellProperties(inflow=DamagedProductivityIndex(k_l=0.5, damage=0.2))
try:
    SSDFSimulator(wp_user, backend='rust')
except ValueError as err:
    print(f'Rust backend refused the well: {err}')

# not_in_core lists the parts of a well the Rust core cannot solve, so a script can pick the backend
backend = 'casadi' if not_in_core(wp_user) else 'rust'
op = SSDFSimulator(wp_user, backend=backend).simulate(bc)
print(f'Damaged well ({backend}): p_0 = {op.p_0:.2f} bar ({op.label}), '
      f'liquid rate {op.w_res:.2f} kg/s, against {rs_c.operating_point.w_res:.2f} kg/s without damage')

plt.show()
