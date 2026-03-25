"""
Visual tutorial: Solution Gas-Oil Ratio (Rs) and Oil Formation Volume Factor (Bo).

Demonstrates how Rs and Bo vary with pressure and temperature using the
Vazquez-Beggs correlations implemented in manywells, and shows their effect
on live-oil density.

Usage:
    uv run python scripts/fvf_learn.py
"""

import numpy as np
import matplotlib.pyplot as plt

from manywells.pvt import density_from_api, gas_density_from_sg
from manywells.pvt.fluid import FluidModel
from manywells.units import CF_BAR


# ── Fluid setup ──────────────────────────────────────────────────────────
#
# A typical North Sea oil: API 35, gas gravity 0.65, bubble point at 250 bar.
#
fl = FluidModel(
    rho_o=density_from_api(35),
    rho_g=gas_density_from_sg(0.65),
    gor=200.0,
    wlr=0.0,
    p_bubble=250e5,
)

print(f"Oil density (std):  {fl.rho_o:.1f} kg/m3  (API {fl.api:.1f})")
print(f"Gas density (std):  {fl.rho_g:.3f} kg/m3  (sg_gas {fl.sg_gas:.3f})")
print(f"Bubble point:       {fl.p_bubble / 1e5:.0f} bar")
print()


# ── 1. Rs and Bo vs pressure (at constant temperature) ───────────────────
#
# At low pressure, little gas is dissolved (low Rs) and the oil barely
# expands (Bo near 1).  As pressure rises, more gas dissolves and the
# oil swells.  Above the bubble point, Rs and Bo are capped -- all
# available gas is already in solution.

T_fixed = 273.15 + 80  # 80 degC
p_bar = np.linspace(1, 350, 500)

rs_vals = np.array([float(fl.rs(p, T_fixed)) for p in p_bar])
bo_vals = np.array([float(fl.bo(p, T_fixed)) for p in p_bar])
rho_l_vals = np.array([float(fl.liquid_density(p, T_fixed)) for p in p_bar])

p_bubble_bar = fl.p_bubble / CF_BAR

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

ax = axes[0]
ax.plot(p_bar, rs_vals, 'b-', lw=2)
ax.axvline(p_bubble_bar, color='grey', ls='--', lw=1, label=f'Bubble point ({p_bubble_bar:.0f} bar)')
ax.set_ylabel('Rs  (Sm$^3$ gas / Sm$^3$ oil)')
ax.set_title('Solution Gas-Oil Ratio vs Pressure')
ax.legend()
ax.annotate('Below bubble point:\nRs increases with pressure\n(more gas dissolves)',
            xy=(100, float(fl.rs(100, T_fixed))),
            xytext=(120, rs_vals.max() * 0.35),
            arrowprops=dict(arrowstyle='->', color='blue'),
            fontsize=9, color='blue')
ax.annotate('Above bubble point:\nRs is capped\n(all gas dissolved)',
            xy=(300, float(fl.rs(300, T_fixed))),
            xytext=(280, rs_vals.max() * 0.55),
            arrowprops=dict(arrowstyle='->', color='grey'),
            fontsize=9, color='grey')

ax = axes[1]
ax.plot(p_bar, bo_vals, 'r-', lw=2)
ax.axvline(p_bubble_bar, color='grey', ls='--', lw=1)
ax.axhline(1.0, color='black', ls=':', lw=0.8)
ax.set_ylabel('Bo  (dimensionless)')
ax.set_title('Oil Formation Volume Factor vs Pressure')
ax.annotate('Bo = 1 means no expansion\n(dead oil / surface conditions)',
            xy=(20, 1.0), xytext=(40, 1.05),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=9, color='black')
ax.annotate('Dissolved gas makes\nthe oil swell (Bo > 1)',
            xy=(180, float(fl.bo(180, T_fixed))),
            xytext=(100, bo_vals.max() * 0.97),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=9, color='red')

ax = axes[2]
ax.plot(p_bar, rho_l_vals, 'g-', lw=2)
ax.axvline(p_bubble_bar, color='grey', ls='--', lw=1)
ax.axhline(fl.rho_o, color='black', ls=':', lw=0.8, label=f'Dead oil density ({fl.rho_o:.0f} kg/m$^3$)')
ax.set_ylabel(r'$\rho_l$  (kg/m$^3$)')
ax.set_xlabel('Pressure (bar)')
ax.set_title('Live-Oil Density vs Pressure')
ax.legend(loc='lower right')
ax.annotate('Density = (rho_o + Rs * rho_g) / Bo\n'
            'Rs adds mass, Bo adds volume;\n'
            'the net effect depends on which dominates',
            xy=(150, float(fl.liquid_density(150, T_fixed))),
            xytext=(30, fl.rho_o - 40),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=9, color='green')

fig.suptitle(f'Black Oil PVT at T = {T_fixed - 273.15:.0f} $\\degree$C  '
             f'(API {fl.api:.0f}, $\\gamma_g$ = {fl.sg_gas:.2f})',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])


# ── 2. Temperature effect on Rs ──────────────────────────────────────────
#
# Hotter oil holds less dissolved gas at the same pressure.

fig2, ax2 = plt.subplots(figsize=(8, 5))
temperatures_C = [40, 60, 80, 100, 120]
cmap = plt.cm.coolwarm

for i, T_C in enumerate(temperatures_C):
    T_K = 273.15 + T_C
    rs = [float(fl.rs(p, T_K)) for p in p_bar]
    color = cmap(i / (len(temperatures_C) - 1))
    ax2.plot(p_bar, rs, lw=2, color=color, label=f'{T_C} $\\degree$C')

ax2.axvline(p_bubble_bar, color='grey', ls='--', lw=1, label='Bubble point')
ax2.set_xlabel('Pressure (bar)')
ax2.set_ylabel('Rs  (Sm$^3$ gas / Sm$^3$ oil)')
ax2.set_title('Temperature Effect on Rs: Hotter Oil Dissolves Less Gas',
              fontsize=12, fontweight='bold')
ax2.legend(title='Temperature')


# ── 3. Dead oil vs black oil comparison ──────────────────────────────────
#
# The dead oil model assumes Rs = 0, Bo = 1 everywhere.
# The black oil model captures the pressure-dependent phase behavior.

fl_dead = FluidModel(
    rho_o=density_from_api(35),
    rho_g=gas_density_from_sg(0.65),
    gor=200.0,
    wlr=0.0,
    oil_model='dead_oil',
)

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5.5))

rho_dead = [float(fl_dead.liquid_density(p, T_fixed)) for p in p_bar]
rho_live = [float(fl.liquid_density(p, T_fixed)) for p in p_bar]

ax3a.plot(p_bar, rho_dead, 'k--', lw=2, label='Dead oil (Rs=0, Bo=1)')
ax3a.plot(p_bar, rho_live, 'g-', lw=2, label='Black oil (Vazquez-Beggs)')
ax3a.axvline(p_bubble_bar, color='grey', ls='--', lw=0.8)
ax3a.set_xlabel('Pressure (bar)')
ax3a.set_ylabel(r'$\rho_l$  (kg/m$^3$)')
ax3a.set_title('Liquid Density')
ax3a.legend()

p_wellbore = np.linspace(250, 30, 100)  # bottom to top
rs_wellbore = [float(fl.rs(p, T_fixed)) for p in p_wellbore]

ax3b.plot(p_wellbore, rs_wellbore, 'b-', lw=2)
ax3b.set_xlabel('Pressure (bar)')
ax3b.set_ylabel('Rs  (Sm$^3$/Sm$^3$)')
ax3b.set_title('Gas Exsolution Along the Wellbore')
ax3b.annotate('Bottom of well\n(high pressure,\ngas dissolved)',
              xy=(p_wellbore[0], rs_wellbore[0]),
              xytext=(p_wellbore[0] - 30, rs_wellbore[0] * 0.7),
              arrowprops=dict(arrowstyle='->', color='blue'),
              fontsize=9, color='blue', ha='center')
ax3b.annotate('Top of well\n(low pressure,\ngas exsolved)',
              xy=(p_wellbore[-1], rs_wellbore[-1]),
              xytext=(p_wellbore[-1] + 40, rs_wellbore[-1] + rs_wellbore[0] * 0.3),
              arrowprops=dict(arrowstyle='->', color='blue'),
              fontsize=9, color='blue', ha='center')
ax3b.invert_xaxis()

fig3.suptitle('Dead Oil vs Black Oil Model', fontsize=13, fontweight='bold')
plt.tight_layout()


# ── Summary printout ─────────────────────────────────────────────────────

print("Key values at typical wellbore conditions:")
print(f"{'Pressure (bar)':>16} {'Rs (Sm3/Sm3)':>14} {'Bo':>8} {'rho_l (kg/m3)':>15}")
print("-" * 56)
for p in [30, 100, 150, 200, 250, 300]:
    rs = float(fl.rs(p, T_fixed))
    bo = float(fl.bo(p, T_fixed))
    rho = float(fl.liquid_density(p, T_fixed))
    marker = " <-- bubble point" if abs(p - p_bubble_bar) < 1 else ""
    print(f"{p:>16} {rs:>14.2f} {bo:>8.4f} {rho:>15.2f}{marker}")

# fig.savefig('scripts/fvf_learn_1_rs_bo_vs_pressure.png', dpi=150, bbox_inches='tight')
# fig2.savefig('scripts/fvf_learn_2_temperature_effect.png', dpi=150, bbox_inches='tight')
# fig3.savefig('scripts/fvf_learn_3_dead_vs_black_oil.png', dpi=150, bbox_inches='tight')
# print("\nFigures saved to scripts/fvf_learn_*.png")
plt.show()
