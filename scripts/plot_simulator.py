"""
Plotting helpers for the steady-state drift-flux simulator.

These live outside the simulator class so they can be reused freely with the
shooting-method backend ``manywells_rs.SSDFSimulator`` (the Rust port), which
exposes the small, purely-computational interface these helpers rely on::

    sim.dim_x                    # state dimension (7)
    sim.delta_z                  # cell length (m)
    sim.bc.p_s, sim.bc.p_r       # separator / reservoir pressure (bar)
    sim._residual(p0)            # (R(p0), shady) for a trial bottomhole pressure, or None
    sim._right_boundary_eqs(cell)  # choke residual for a wellhead cell state

Usage (notebook)::

    import manywells_rs as mrs
    from scripts.plot_simulator import plot_solution, plot_residual

    sim = mrs.SSDFSimulator(wp, bc)          # wp/bc may be manywells or manywells_rs objects

    solutions = sim.simulate()               # list of solutions (>= 1), highest p0 first
    plot_solution(sim, solutions[0])         # displays inline
    plot_residual(sim, n_points=100)
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_solution(sim, x, tag='solution', label=None, axes=None, save=False):
    """
    Plot pressure, velocities v_g, v_l, and volume fraction alpha vs. z for the given
    flat solution list x -- works for a partial solution too (e.g. everything solved
    before a cell failure).

    By default the figure is neither saved nor closed, so it just displays inline in a
    notebook. Pass an existing array of (at least 4) matplotlib Axes via ``axes`` to draw
    into your own figure. Pass ``save=True`` to write '{tag}.pdf', or ``save='some/path.pdf'``
    to choose the path yourself.

    :param sim: A simulator instance (manywells_rs.SSDFSimulator)
    :param x: Solution as flat list
    :param tag: Title / default-filename tag
    :param axes: Optional array-like of >= 4 Axes to draw into (a new figure is created
        if None)
    :param save: If truthy, save the figure. True uses '{tag}.pdf'; a string is used as
        the file path.
    :return: The matplotlib Figure
    """
    dim_x = sim.dim_x
    delta_z = sim.wp.L / sim.n_cells

    n_done = len(x) // dim_x
    arr = np.array(x).reshape(n_done, dim_x)
    z_vals = np.array([j * delta_z for j in range(n_done)])

    if axes is None:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 10))
        created_fig = True
    else:
        axes = np.atleast_1d(axes).ravel()
        if len(axes) < 4:
            raise ValueError('plot_solution needs at least 4 axes')
        fig = axes[0].figure
        created_fig = False

    vars = ['p', 'v_g', 'v_l', 'alpha']
    for ax, col, var in zip(axes, range(4), vars):
        ax.plot(z_vals, arr[:, col], marker='.', label=label if label is not None else "")
        ax.set_ylabel(var)
        ax.grid(alpha=0.3)
    axes[3].set_xlabel('z (m)')

    if created_fig:
        fig.suptitle(tag)
        fig.tight_layout()

    if save:
        filename = save if isinstance(save, str) else f"{tag}.pdf"
        fig.savefig(filename)
        print(f"Saved plot to {filename}")

    return fig


def plot_residual(sim, p_min=None, p_max=None, n_points=50, p0s=None, tag='residual_scan', ax=None, save=False):
    """
    Debug plot of the outer shooting residual R(p_0). Each trial p_0 is evaluated via
    ``sim._residual(p0)``, which returns either ``(R, shady)`` or ``None``:

    - genuine solutions (``shady == False``) are drawn as a connected line;
    - "shady" points (``shady == True``) -- where R was computed from a *propagated*
      (non-physical) march, i.e. the integration hit a choked/infeasible cell and froze
      the last good state -- are drawn as orange open squares, since these are false
      solutions that ``simulate()`` would reject;
    - infeasible p_0 (``None``, e.g. the bottomhole cell is unsolvable) are drawn as
      red x-marks on the zero line.

    By default the p_0 grid is a uniform linspace over [p_min, p_max]. Pass an explicit
    array via ``p0s`` to use your own (possibly non-uniform) grid -- e.g. coarse at low
    pressure and fine at high pressure::

        p0s = np.concatenate([np.linspace(p_s, 0.8 * p_r, 20),
                              np.linspace(0.8 * p_r, p_r, 60)])
        plot_residual(sim, p0s=p0s)

    When ``p0s`` is given, ``p_min``/``p_max``/``n_points`` are ignored.

    By default the figure is neither saved nor closed, so it just displays inline in a
    notebook. Pass an existing ``ax`` to draw into your own figure, or ``save=True`` /
    ``save='some/path.pdf'`` to write a PDF.

    :param sim: A simulator instance (manywells_rs.SSDFSimulator)
    :param p_min: Lower bound of the p_0 sweep (defaults to p_s + 1e-3)
    :param p_max: Upper bound of the p_0 sweep (defaults to p_r - 1e-3)
    :param n_points: Number of p_0 samples (used only when ``p0s`` is None)
    :param p0s: Optional explicit array of p_0 values to evaluate; overrides
        p_min/p_max/n_points and allows a non-uniform grid
    :param tag: Default-filename tag
    :param ax: Optional Axes to draw into (a new figure is created if None)
    :param save: If truthy, save the figure. True uses '{tag}.svg'; a string is used as
        the file path.
    :return: (p0s, residuals, shady) -- residuals is NaN for infeasible points and
        shady is a boolean mask flagging propagated (false) solutions, so the data can
        be re-plotted or inspected in a notebook.
    """
    if p0s is None:
        p_min = sim.bc.p_s + 1e-3 if p_min is None else p_min
        p_max = sim.bc.p_r - 1e-3 if p_max is None else p_max
        p0s = np.linspace(p_min, p_max, n_points)
    else:
        p0s = np.asarray(p0s, dtype=float)

    residuals = np.full(len(p0s), np.nan)
    shady = np.zeros(len(p0s), dtype=bool)
    for j, p0 in enumerate(p0s):
        try:
            res = sim._residual(float(p0))  # (R, shady) or None
        except Exception as exc:
            print(f"\tp_0={p0:.4f} infeasible ({type(exc).__name__}: {exc})")
            continue
        if res is None:
            continue
        residuals[j], shady[j] = float(res[0]), bool(res[1])

    failed = np.isnan(residuals)                        # infeasible (None returned)
    genuine = np.isfinite(residuals) & ~shady           # real solutions
    propagated = np.isfinite(residuals) & shady         # false / "shady" solutions
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(p0s[genuine], residuals[genuine], '.-', label='R(p_0)')
    ax.plot(p0s[propagated], residuals[propagated], 's', mfc='none', color='tab:orange',
            label='R(p_0) propagated (shady)')
    ax.plot(p0s[failed], np.zeros(failed.sum()), 'rx', label='infeasible')
    ax.axhline(0.0, color='k', linewidth=0.5)

    # Bjarne's initial guess: 5% of the total pressure drop occurs at the inflow.
    p0_guess = sim.bc.p_r - (sim.bc.p_r - sim.bc.p_s) * 0.05
    ax.axvline(p0_guess, color='tab:green', linestyle='--', label=f'initial guess (p_0={p0_guess:.2f})')

    # Physical pressure bounds. Drawn subtly and pushed to the edges via an added
    # x-margin so points sampled right next to p_s / p_r stay easy to see.
    p_s, p_r = sim.bc.p_s, sim.bc.p_r
    ax.axvline(p_s, color='0.5', linestyle=':', linewidth=1, label=f'p_s = {p_s:.2f}')
    ax.axvline(p_r, color='0.5', linestyle=':', linewidth=1, label=f'p_r = {p_r:.2f}')
    ax.set_xlim(p_min, p_max)

    ax.set(xlabel='p_0 (bar)', ylabel='residual R(p_0)')
    ax.legend()
    ax.grid(alpha=0.3)

    if save:
        filename = save if isinstance(save, str) else f"{tag}.svg"
        ax.figure.savefig(filename)
        print(f"Saved plot to {filename}")

    return p0s, residuals, shady
