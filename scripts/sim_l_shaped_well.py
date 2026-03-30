"""
Simulate an L-shaped well: vertical section → build section → horizontal section.

Compares the L-shaped trajectory to a vertical well of the same TVD, showing
how deviation affects pressure, temperature, void fraction, and flow regime.

Usage:
    uv run python scripts/sim_l_shaped_well.py
"""

import numpy as np
import matplotlib.pyplot as plt

from manywells.geometry import WellGeometry
from manywells.simulator import WellProperties, BoundaryConditions, SSDFSimulator


# ── Well geometries ──────────────────────────────────────────────────────

TVD = 2000.0
N_CELLS = 100

# Vertical reference well
geo_vert = WellGeometry.vertical(TVD, n_cells=N_CELLS)

# L-shaped well: 1500 m vertical, 300 m build radius, 800 m horizontal
R = 300.0
md_survey = [0.0, 1500.0]
tvd_survey = [0.0, 1500.0]
for t in np.linspace(0, np.pi / 2, 20)[1:]:
    md_survey.append(1500.0 + R * t)
    tvd_survey.append(1500.0 + R * np.sin(t))
md_survey.append(md_survey[-1] + 800.0)
tvd_survey.append(tvd_survey[-1])
geo_l = WellGeometry.from_survey(md_survey, tvd_survey, n_cells=N_CELLS)


# ── Simulate both wells ─────────────────────────────────────────────────

bc = BoundaryConditions(u=0.5)

wp_vert = WellProperties(geometry=geo_vert)
wp_l = WellProperties(geometry=geo_l)

sim_vert = SSDFSimulator(wp_vert, bc)
sim_l = SSDFSimulator(wp_l, bc)

print("Simulating vertical well ...")
x_vert = sim_vert.simulate()
df_vert = sim_vert.solution_as_df(x_vert)

print("Simulating L-shaped well ...")
x_l = sim_l.simulate()
df_l = sim_l.solution_as_df(x_l)


# ── Plot results ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

datasets = [
    (df_vert, "Vertical", "C0", "-"),
    (df_l, "L-shaped", "C1", "--"),
]

for df, label, color, ls in datasets:
    md = df["md"]

    axes[0, 0].plot(md, df["p"], ls, color=color, lw=2, label=label)
    axes[0, 1].plot(md, df["T"] - 273.15, ls, color=color, lw=2, label=label)
    axes[0, 2].plot(md, df["alpha"], ls, color=color, lw=2, label=label)
    axes[1, 0].plot(md, df["v_g"], ls, color=color, lw=2, label=label)
    axes[1, 1].plot(md, df["v_l"], ls, color=color, lw=2, label=label)

    A = geo_vert.A if label == "Vertical" else geo_l.A
    w_g = A * df["alpha"] * df["rho_g"] * df["v_g"]
    w_l = A * (1 - df["alpha"]) * df["rho_l"] * df["v_l"]
    axes[1, 2].plot(md, w_g, ls, color=color, lw=2, label=f"{label} gas")
    axes[1, 2].plot(md, w_l, ":", color=color, lw=2, label=f"{label} liquid")

labels_units = [
    ("Pressure", "bar"),
    ("Temperature", "°C"),
    ("Void fraction α", "—"),
    ("Gas velocity", "m/s"),
    ("Liquid velocity", "m/s"),
    ("Mass flow rate", "kg/s"),
]

for ax, (name, unit) in zip(axes.flat, labels_units):
    ax.set_xlabel("Measured depth (m)")
    ax.set_ylabel(f"{name} ({unit})")
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    f"Vertical vs L-shaped well  (TVD ≈ {TVD:.0f} m, choke = {bc.u * 100:.0f}%)",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.96])


# ── Trajectory comparison ────────────────────────────────────────────────

fig2, (ax_traj, ax_cos) = plt.subplots(1, 2, figsize=(11, 5))

for geo, label, color, ls in [
    (geo_vert, "Vertical", "C0", "-"),
    (geo_l, "L-shaped", "C1", "--"),
]:
    md_s = np.array(geo.md_survey)
    tvd_s = np.array(geo.tvd_survey)
    d_md = np.diff(md_s)
    d_tvd = np.diff(tvd_s)
    horiz = np.concatenate(([0.0], np.cumsum(np.sqrt(np.maximum(d_md**2 - d_tvd**2, 0)))))

    ax_traj.plot(horiz, tvd_s, marker="o", markersize=2, color=color, ls=ls, lw=2, label=label)

    md_mid = 0.5 * (md_s[:-1] + md_s[1:])
    cos_i = np.array(geo.cos_incl[::-1])
    ax_cos.plot(md_mid, cos_i, marker="s", markersize=2, color=color, ls=ls, lw=2, label=label)

ax_traj.set_xlabel("Horizontal displacement (m)")
ax_traj.set_ylabel("TVD (m)")
ax_traj.set_title("Wellbore trajectory")
ax_traj.invert_yaxis()
ax_traj.set_aspect("equal")
ax_traj.legend()
ax_traj.grid(True, alpha=0.3)

ax_cos.set_xlabel("MD (m)")
ax_cos.set_ylabel("cos(inclination)")
ax_cos.set_title("Inclination profile")
ax_cos.set_ylim(-0.05, 1.1)
ax_cos.axhline(1.0, color="grey", ls=":", lw=0.8)
ax_cos.axhline(0.0, color="grey", ls=":", lw=0.8)
ax_cos.legend()
ax_cos.grid(True, alpha=0.3)

fig2.suptitle("Well geometry comparison", fontsize=13, fontweight="bold")
fig2.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()
