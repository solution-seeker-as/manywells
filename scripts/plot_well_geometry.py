"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Plot wellbore geometry from a WellGeometry object.

Produces two side-by-side panels:
  - Left:  wellbore trajectory (horizontal offset vs TVD, depth increasing downward)
  - Right: cos(inclination) vs measured depth
"""

import numpy as np
import matplotlib.pyplot as plt

from manywells.geometry import WellGeometry


def plot_well_geometry(geo: WellGeometry, ax_traj=None, ax_incl=None, label=None):
    """Plot a wellbore trajectory and inclination profile.

    Parameters
    ----------
    geo : WellGeometry
        Geometry to plot.
    ax_traj : matplotlib.axes.Axes, optional
        Axes for the trajectory panel. Created if *None*.
    ax_incl : matplotlib.axes.Axes, optional
        Axes for the inclination panel. Created if *None*.
    label : str, optional
        Legend label for this geometry.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_traj, ax_incl : matplotlib.axes.Axes
    """
    created_fig = ax_traj is None or ax_incl is None
    if created_fig:
        fig, (ax_traj, ax_incl) = plt.subplots(1, 2, figsize=(10, 6))
    else:
        fig = ax_traj.get_figure()

    md = np.array(geo.md_survey)
    tvd = np.array(geo.tvd_survey)

    # Horizontal displacement at each station (cumulative)
    d_md = np.diff(md)
    d_tvd = np.diff(tvd)
    d_horiz = np.sqrt(np.maximum(d_md**2 - d_tvd**2, 0.0))
    horiz = np.concatenate(([0.0], np.cumsum(d_horiz)))

    # -- trajectory --
    ax_traj.plot(horiz, tvd, 'o-', markersize=3, label=label)
    ax_traj.set_xlabel("Horizontal displacement (m)")
    ax_traj.set_ylabel("TVD (m)")
    ax_traj.set_title("Wellbore trajectory")
    if not ax_traj.yaxis_inverted():
        ax_traj.invert_yaxis()
    ax_traj.set_aspect("equal")
    if label:
        ax_traj.legend()

    # -- inclination --
    md_mid = 0.5 * (md[:-1] + md[1:])
    cos_incl = np.array(geo.cos_incl[::-1])  # back to survey order
    ax_incl.plot(md_mid, cos_incl, 's-', markersize=3, label=label)
    ax_incl.set_xlabel("MD (m)")
    ax_incl.set_ylabel("cos(inclination)")
    ax_incl.set_title("Inclination profile")
    ax_incl.set_ylim(-0.05, 1.1)
    ax_incl.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax_incl.axhline(0.0, color="grey", ls=":", lw=0.8)
    if label:
        ax_incl.legend()

    fig.tight_layout()
    return fig, ax_traj, ax_incl


if __name__ == "__main__":

    # Plot some example geometries

    # 1. Vertical well
    geo_vert = WellGeometry.vertical(2000, n_cells=50)

    # 2. L-shaped well: vertical → build section → horizontal
    R = 300.0
    md_survey = [0.0, 1800.0]
    tvd_survey = [0.0, 1800.0]
    for t in np.linspace(0, np.pi / 2, 20)[1:]:
        md_survey.append(1800.0 + R * t)
        tvd_survey.append(1800.0 + R * np.sin(t))
    md_survey.append(md_survey[-1] + 800.0)
    tvd_survey.append(tvd_survey[-1])
    geo_l = WellGeometry.from_survey(md_survey, tvd_survey, n_cells=80)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 7))
    plot_well_geometry(geo_vert, ax1, ax2, label="Vertical")
    plot_well_geometry(geo_l, ax1, ax2, label="L-shaped")

    fig.suptitle("Well geometry comparison", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
