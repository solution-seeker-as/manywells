"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

The project's main figure (README.md): the well model, what the simulator returns and what it is used for, drawn
from the simulator's own results, in a light and a dark version.

    uv run python -m scripts.figures.main_figure                 # docs/manywells.svg and docs/manywells-dark.svg
    uv run python -m scripts.figures.main_figure --format pdf    # for print; png is written at 600 dpi

The panels:

    a  One well (example_well): its trajectory to scale, with the tubing coloured by the flow regimes' probabilities
       at each grid point, and the operating point's pressure, temperature and gas volume fraction against depth.
    b  The five flow regimes' probabilities (SLIP-13) in a vertical pipe, at the fluid properties of the well's middle
       grid point, with the well's superficial velocities from the bottomhole to the wellhead.
    c  The same well without lift gas: its inflow curve, and the bottomhole pressure that the tubing and the choke
       need at each liquid rate (solved with FixedFlowRate). The curves cross at the roots that root_set returns.
    d  The first wells that the v2 sampler draws (specs/sampling.md), each from its own wellhead, by fluid class.
    e  A calibration of the well to synthetic measurements, as in scripts/sim_examples/calibrate_well.py.
    f  The well's liquid rate against its lift-gas rate at three reservoir pressures.

The layout is in millimetres, for a figure of a journal's full page width, and the text is drawn as paths, so that
the files look the same wherever they are shown. The font is the first of FONTS that is installed.
"""

import argparse
import math
from dataclasses import replace
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

from manywells.calibration import apply, calibrate, synthetic_data
from manywells.geometry import WellGeometry
from manywells.inflow import FixedFlowRate, ProductivityIndex
from manywells.pvt.fluid import FluidModel
from manywells.sampling.v2.draws import draw_well
from manywells.simulator import BoundaryConditions, SimError, SSDFSimulator, WellProperties
from manywells.slip import HASAN_REGIMES, hasan_probabilities

# --- Size, type and colours -------------------------------------------------------------------------------------------
WIDTH, HEIGHT = 190.0, 147.0  # The figure (mm)
FONTS = ('Helvetica', 'Arial', 'Nimbus Sans', 'Liberation Sans', 'DejaVu Sans')
TITLE, LABEL, SMALL = 8.2, 7.0, 6.3  # Font sizes (pt)

# The regimes take five hues of one categorical palette, assigned so that the regimes that share a border on the map
# of panel b are pairs that are easy to tell apart, also with a colour-vision deficiency; each is named where it is
# drawn. The fluid classes are ordered, from black oil to dry gas, and take the steps of one hue, as do the reservoir
# pressures of panel f. The surfaces are GitHub's light and dark page backgrounds.
THEMES = {
    'light': dict(
        surface='#ffffff',
        ink='#1f2328',
        ink2='#59636e',
        muted='#818b98',
        grid='#e6e9ed',
        axis='#c2c9d1',
        accent='#2a78d6',
        ramp=('#184f95', '#2a78d6', '#6da7ec'),
        regimes=dict(zip(HASAN_REGIMES, ('#eda100', '#6250d6', '#008300', '#e87ba4', '#2a78d6'))),
        fluids=('#542c07', '#7d450f', '#a95f17', '#d47c2e', '#f5a15e'),
        wash=0.42,  # The regimes' share of their colour on the map, the rest the surface's
        reservoir='#f3efe6',
        grain='#cfc6b2',
    ),
    'dark': dict(
        surface='#0d1117',
        ink='#f0f6fc',
        ink2='#b7bfc8',
        muted='#8b949e',
        grid='#21262d',
        axis='#3d444d',
        accent='#3987e5',
        ramp=('#b7d3f6', '#5598e7', '#256abf'),
        regimes=dict(zip(HASAN_REGIMES, ('#c98500', '#9085e9', '#0ca30c', '#d55181', '#3987e5'))),
        fluids=('#905211', '#bb6e20', '#e38d3d', '#fcb67c', '#ffdfc0'),
        wash=0.5,
        reservoir='#1a1c1b',
        grain='#3b3a33',
    ),
}
FLUID_CLASSES = ('black_oil', 'volatile_oil', 'gas_condensate', 'wet_gas', 'dry_gas')

SEED = 1  # Of the sampler's wells and of the calibration's measurements
N_WELLS = 80  # Panel d: the sampler's first wells
DATASET_WELLS, DATASET_SAMPLES = 10_000, 200  # The v2 datasets' wells and samples per well (specs/goals.md)


# --- The example well and the data of each panel ---------------------------------------------------------------------
def example_well():
    """The well of panels a, b, c, e and f: a build-and-hold trajectory to 45 degrees, a black oil with water, and
    lift gas through a valve at 1700 m of measured depth."""
    kickoff, radius, inclination, hold = 600.0, 700.0, math.radians(45.0), 1500.0  # m, m, rad, m
    md, tvd = [0.0, kickoff], [0.0, kickoff]
    for theta in np.linspace(0.0, inclination, 20)[1:]:
        md.append(kickoff + radius * theta)
        tvd.append(kickoff + radius * math.sin(theta))
    md.append(md[-1] + hold)
    tvd.append(tvd[-1] + hold * math.cos(inclination))
    wp = WellProperties(
        geometry=WellGeometry.from_survey(md, tvd, n_cells=100, D=0.124),
        fluid=FluidModel(gor=80.0, wlr=0.3),
        inflow=ProductivityIndex(k_l=1.0),
        injection_md=1700.0,
    )
    bc = BoundaryConditions(p_r=160.0, p_s=15.0, T_r=368.15, T_s=280.15, u=0.7, w_lg=1.0, T_lg=310.0)
    return wp, bc


def regime_probabilities(wp, state) -> np.ndarray:
    """The regimes' probabilities at each grid point of a root's state, (N + 1, 5), in HASAN_REGIMES' order."""
    geo, out = wp.geometry, []
    for i, (p, v_g, v_l, alpha, rho_g, rho_l, T) in enumerate(state):
        sigma = wp.fluid.surface_tension(p, T, rho_l)
        probs = wp.slip.classify(v_g, v_l, alpha, rho_g, rho_l, sigma, geo.D, geo.cos_incl[max(i - 1, 0)])
        out.append(np.asarray(probs, dtype=float).ravel())
    return np.array(out)


def profile_data(wp, bc) -> dict:
    """Panel a: the operating point along the well, from the bottomhole."""
    op = SSDFSimulator(wp, backend='rust').simulate(bc)
    geo, state = wp.geometry, op.state
    md, tvd = np.array(geo.md), np.array(geo.tvd)
    step = np.sqrt(np.maximum(np.diff(md) ** 2 - np.diff(tvd) ** 2, 0.0))  # The horizontal step of each cell
    return dict(
        x=np.concatenate([np.cumsum(step[::-1])[::-1], [0.0]]),  # Horizontal distance from the wellhead (m)
        tvd=tvd,
        p=state[:, 0],
        T=state[:, 6] - 273.15,
        alpha=state[:, 3],
        v_gs=state[:, 3] * state[:, 1],
        v_ls=(1 - state[:, 3]) * state[:, 2],
        probs=regime_probabilities(wp, state),
        regime=op.flow_regime,
        valve=wp.injection_point,
        T_formation=bc.T_s - 273.15 + (bc.T_r - bc.T_s) * np.array(geo.tvd_frac),
        state_mid=state[len(state) // 2],
    )


def regime_map_data(wp, state) -> dict:
    """Panel b: the regimes' probabilities over the superficial velocities (as their base-10 logarithms), in a
    vertical pipe with the fluid properties of one state."""
    p, _v_g, _v_l, _alpha, rho_g, rho_l, T = state
    sigma, slip = float(wp.fluid.surface_tension(p, T, rho_l)), wp.slip
    lg_gs, lg_ls = np.linspace(-1.5, 1.8, 420), np.linspace(-1.5, 1.15, 340)
    v_gs, v_ls = np.meshgrid(10.0**lg_gs, 10.0**lg_ls)
    probs = hasan_probabilities(
        v_gs.ravel(),
        v_ls.ravel(),
        rho_g,
        rho_l,
        sigma,
        wp.geometry.D,
        1.0,
        slip.f_ms,
        slip.transition_width,
        slip.lambda_floor,
    )
    return dict(lg_gs=lg_gs, lg_ls=lg_ls, probs=np.asarray(probs, dtype=float).reshape(len(HASAN_REGIMES), *v_gs.shape))


def solutions_data(wp, bc) -> dict:
    """Panel c: the well without lift gas. The tubing curve is the bottomhole pressure at each fixed liquid rate."""
    bc = replace(bc, w_lg=0.0, T_lg=None)
    rates = np.concatenate([np.linspace(0.3, 3.0, 19), np.linspace(3.25, 21.0, 72)])
    p_0 = []
    for w_l in rates:
        fixed = replace(wp, inflow=FixedFlowRate(w_l_const=float(w_l)))
        # A reservoir pressure above every bottomhole pressure of the curve: the fixed rate does not depend on it
        p_0.append(SSDFSimulator(fixed, backend='rust').simulate(replace(bc, p_r=400.0)).p_0)
    roots = [(r.w_res, r.p_0, r.label) for r in SSDFSimulator(wp, backend='rust').root_set(bc).roots]
    return dict(rates=rates, p_0=np.array(p_0), roots=roots, p_r=bc.p_r, k_l=wp.inflow.k_l)


def sampled_wells() -> list:
    """Panel d: the fluid class and the trajectory, (x, tvd) from the wellhead in m, of the sampler's first wells."""
    wells = []
    for k in range(N_WELLS):
        w, (md, tvd) = draw_well(SEED, k)
        md, tvd = np.asarray(md), np.asarray(tvd)
        step = np.sqrt(np.maximum(np.diff(md) ** 2 - np.diff(tvd) ** 2, 0.0))
        wells.append(dict(fluid_class=w['fluid_class'], x=np.concatenate([[0.0], np.cumsum(step)]), tvd=tvd))
    return wells


def calibration_data(wp, bc) -> dict:
    """Panel e: the well as its user knows it (a first guess of four parameters), measurements simulated from the
    true well with noise, and the calibrated well, each swept over the choke position. Oil rates in Sm3/d."""
    guess = replace(wp, inflow=ProductivityIndex(k_l=1.5))
    truth = {'K_c': 0.8 * wp.choke.K_c, 'k_l': wp.inflow.k_l, 'roughness': 1.5e-4, 'h': 22.0}
    data = synthetic_data(guess, bc, truth, n_rows=12, seed=SEED, instrumentation='full', p_s_spread=0.0)
    result = calibrate(guess, data, bc, free=list(truth))
    to_sm3d = wp.fluid.f_o_in_liquid / wp.fluid.rho_o * 86400.0  # From the liquid's mass rate (kg/s)
    u = np.linspace(0.15, 1.0, 35)

    def sweep(well):
        sim, rates = SSDFSimulator(well, backend='rust'), []
        for u_k in u:
            try:
                rates.append(sim.simulate(replace(bc, u=float(u_k))).w_res * to_sm3d)
            except SimError:
                rates.append(np.nan)
        return np.array(rates)

    return dict(
        u=u,
        before=sweep(guess),
        after=sweep(result.well),
        true=sweep(apply(guess, truth)),
        u_measured=data.rows['CHK'].to_numpy(),
        q_measured=data.rows['QOIL'].to_numpy() * 24.0,
    )


def lift_data(wp, bc) -> dict:
    """Panel f: the liquid rate against the lift-gas rate at three reservoir pressures, NaN where the well has no
    operating point, and the point of panel a's well."""
    sim = SSDFSimulator(wp, backend='rust')
    w_lg = np.concatenate([np.linspace(0.0, 1.0, 41), np.linspace(1.1, 4.0, 30)])
    curves = {}
    for p_r in (160.0, 145.0, 130.0):
        rates = []
        for w in w_lg:
            try:
                rates.append(sim.simulate(replace(bc, p_r=p_r, w_lg=float(w))).w_res)
            except SimError:
                rates.append(np.nan)
        curves[p_r] = np.array(rates)
    return dict(w_lg=w_lg, curves=curves, point=(bc.w_lg, sim.simulate(bc).w_res))


def compute() -> dict:
    """The data of every panel."""
    wp, bc = example_well()
    profile = profile_data(wp, bc)
    return dict(
        profile=profile,
        regime_map=regime_map_data(wp, profile['state_mid']),
        solutions=solutions_data(wp, bc),
        wells=sampled_wells(),
        calibration=calibration_data(wp, bc),
        lift=lift_data(wp, bc),
    )


# --- Drawing helpers -------------------------------------------------------------------------------------------------
def font_family() -> str:
    """The first of FONTS that is installed."""
    for name in FONTS:
        try:
            font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            return name
        except ValueError:
            continue
    return 'sans-serif'


def rc(th) -> dict:
    """Matplotlib's settings for a theme."""
    family = font_family()
    return {
        'font.family': family,
        'font.size': LABEL,
        'mathtext.fontset': 'custom',
        'mathtext.rm': family,
        'mathtext.it': f'{family}:italic',
        'mathtext.bf': f'{family}:bold',
        'mathtext.cal': family,
        'mathtext.sf': family,
        'mathtext.tt': family,
        'svg.fonttype': 'path',
        'svg.hashsalt': 'manywells',
        'pdf.fonttype': 42,
        'text.color': th['ink'],
        'axes.edgecolor': th['axis'],
        'axes.labelcolor': th['ink2'],
        'axes.linewidth': 0.6,
        'axes.labelsize': LABEL,
        'axes.labelpad': 2.0,
        'axes.facecolor': 'none',
        'xtick.color': th['axis'],
        'ytick.color': th['axis'],
        'xtick.labelcolor': th['ink2'],
        'ytick.labelcolor': th['ink2'],
        'xtick.labelsize': SMALL,
        'ytick.labelsize': SMALL,
        'xtick.major.size': 2.0,
        'ytick.major.size': 2.0,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.pad': 1.5,
        'ytick.major.pad': 1.5,
        'grid.color': th['grid'],
        'grid.linewidth': 0.5,
        'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round',
        'legend.fontsize': SMALL,
        'legend.frameon': False,
    }


def axes_mm(fig, x, y, w, h):
    """Axes at (x, y) from the figure's top left corner, w wide and h high (mm)."""
    return fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])


def panel_title(fig, th, x, y, letter, title):
    """A panel's letter and title, with the baseline at (x, y) from the figure's top left corner (mm)."""
    kw = dict(fontweight='bold', color=th['ink'], va='baseline')
    fig.text(x / WIDTH, 1 - y / HEIGHT, letter, fontsize=TITLE + 1, **kw)
    fig.text((x + 4.2) / WIDTH, 1 - y / HEIGHT, title, fontsize=TITLE, **kw)


def plot_axes(ax):
    """A plot's frame: the left and bottom spines, and a hairline grid behind the data."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True)


def leader(th, **kw) -> dict:
    """The arrowprops of a label's leader line."""
    return dict(arrowstyle='-', color=th['muted'], lw=0.5, **kw)


def blend(probs, colors):
    """The colours (n, 3) of probability vectors (n, k) over k colours."""
    return np.clip(np.asarray(probs) @ np.array([to_rgb(c) for c in colors]), 0.0, 1.0)


def offset(path, d):
    """A path (n, 2) moved by d to the right of its direction, with y pointing down."""
    t = np.gradient(path, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return path + d * np.column_stack([-t[:, 1], t[:, 0]])


# --- Panel a: the well ------------------------------------------------------------------------------------------------
def draw_model(fig, th, d, x0, y0):
    """The schematic and the three tracks, in a block of 124 x 92 mm with its top left corner at (x0, y0)."""
    panel_title(fig, th, x0, y0 + 3.4, 'a', 'Steady-state drift-flux model of a well')
    ink, ink2, muted = th['ink'], th['ink2'], th['muted']

    # The schematic's axes are in mm from the wellhead, y down, with the well to scale: the ground is `above` mm
    # below the axes' top, the bottomhole `below` mm above its bottom and the wellhead `left` mm from its left edge
    w_ax, h_ax, above, below, left = 68.0, 85.0, 17.0, 5.0, 22.0
    depth = d['tvd'][0]  # m
    scale = (h_ax - above - below) / depth  # mm per m
    ax = axes_mm(fig, x0, y0 + 7.0, w_ax, h_ax)
    ax.set_xlim(-left, w_ax - left)
    ax.set_ylim(h_ax - above, -above)
    ax.axis('off')
    arrow = dict(arrowstyle='-|>', color=ink2, lw=0.8, mutation_scale=5.5, shrinkA=0, shrinkB=0)

    def text(x, y, s, **kw):
        return ax.text(x, y, s, **{'fontsize': SMALL, 'color': ink2, 'va': 'baseline', **kw})

    def callout(x, y, title, lines, target=None, anchor=None):
        """A bold title with lines of text under it, and a leader from `anchor` to `target`."""
        text(x, y, title, fontsize=LABEL, fontweight='bold', color=ink)
        for k, line in enumerate(lines):
            text(x, y + 2.75 * (k + 1), line)
        if target is not None:
            ax.plot([anchor[0], target[0]], [anchor[1], target[1]], color=muted, lw=0.5, zorder=4)
            ax.plot([target[0]], [target[1]], 'o', ms=1.7, color=muted, mew=0, zorder=4)

    # The reservoir: a band around the bottom of the well
    y_bottom = depth * scale
    top = y_bottom - 5.5
    ax.add_patch(Rectangle((-left, top), w_ax, below + 5.5, fc=th['reservoir'], ec='none', zorder=0))
    rng = np.random.default_rng(3)
    grains = np.column_stack([rng.uniform(-left, w_ax - left, 420), top + rng.uniform(0.3, below + 5.2, 420)])
    ax.scatter(grains[:, 0], grains[:, 1], s=0.5, color=th['grain'], linewidths=0, zorder=0.5)

    # The ground
    ax.plot([-left, w_ax - left], [0, 0], color=th['axis'], lw=0.8, zorder=1, solid_capstyle='butt')
    hatch = [[(x, 0.0), (x - 0.9, 0.9)] for x in np.arange(-left + 1.0, w_ax - left, 1.6)]
    ax.add_collection(LineCollection(hatch, colors=th['axis'], linewidths=0.5, zorder=1))

    # The tubing: the well's path from the bottomhole, then up through the wellhead and over to the choke
    hw, rise, bend, x_choke = 1.55, 5.2, 2.4, 8.2  # Half width, the flowline's height, its bend's radius, the choke
    arc = np.linspace(0.0, np.pi / 2, 12)[1:]
    head = [(0.0, bend - rise)] + [(bend * (1 - np.cos(a)), bend - rise - bend * np.sin(a)) for a in arc]
    head += [(x_choke, -rise)]
    n = len(d['x'])  # The well's grid points: path[:n]
    path = np.vstack([np.column_stack([d['x'], d['tvd']]) * scale, np.array(head)])
    cells = blend(0.5 * (d['probs'][1:] + d['probs'][:-1]), [th['regimes'][r] for r in HASAN_REGIMES])
    colors = np.vstack([cells, np.repeat(cells[-1:], len(head), axis=0)])
    right, lft = offset(path, hw), offset(path, -hw)
    out = (right - lft) / (2 * hw)  # At each point of the path: the unit vector across the tubing, to the right
    quads = [[right[i], right[i + 1], lft[i + 1], lft[i]] for i in range(len(path) - 1)]
    ax.add_collection(PolyCollection(quads, facecolors=colors, edgecolors=colors, linewidths=0.35, zorder=2))
    for wall in (right, lft):
        ax.plot(wall[:, 0], wall[:, 1], color=ink, lw=0.75, zorder=3, solid_capstyle='butt')
    ax.plot([right[0, 0], lft[0, 0]], [right[0, 1], lft[0, 1]], color=ink, lw=0.75, zorder=3)  # The bottom
    ax.plot([-hw - 0.9, hw + 0.9], [0, 0], color=ink, lw=1.8, zorder=3, solid_capstyle='butt')  # The wellhead

    # The choke, the flowline after it and the downstream pressure
    a, b = 2.1, 2.3  # Half the choke's length and height
    bowtie = [(x_choke, -rise - b), (x_choke + 2 * a, -rise + b), (x_choke + 2 * a, -rise - b), (x_choke, -rise + b)]
    ax.add_patch(Polygon(bowtie, closed=True, fc=ink, ec=ink, lw=0.6, joinstyle='miter', zorder=4))
    x_stem, y_handle = x_choke + a, -rise - 3.4
    ax.plot([x_stem, x_stem], [-rise, y_handle], color=ink, lw=0.75, zorder=3, solid_capstyle='butt')
    ax.plot([x_stem - 1.3, x_stem + 1.3], [y_handle, y_handle], color=ink, lw=1.4, solid_capstyle='butt')
    x_end = x_choke + 2 * a + 5.5
    for side in (-hw, hw):
        ax.plot([x_choke + 2 * a, x_end], [side - rise, side - rise], color=ink, lw=0.75, solid_capstyle='butt')
    ax.annotate('', (x_end + 5.0, -rise), (x_end - 2.5, -rise), arrowprops=arrow, zorder=4)
    text(x_end + 6.0, -rise + 0.85, r'$p_s$', fontsize=LABEL, color=ink)
    text(x_stem + 2.0, y_handle + 0.4, r'$u$', fontsize=LABEL, color=ink)
    callout(x_stem + 6.5, -rise - 9.0, 'Choke', ['Bernoulli or Simpson model,', 'subcritical or critical flow'])

    # The lift gas: down the outside of the tubing to the valve
    i_valve = d['valve']
    gas = offset(path, -(hw + 1.15))[i_valve:n]
    inlet = np.array([(gas[-1, 0] - 9.0, -2.6), (gas[-1, 0], -2.6)])
    line = np.vstack([inlet, gas[::-1]])
    ax.plot(line[:, 0], line[:, 1], color=ink2, lw=0.8, zorder=3, solid_joinstyle='miter', solid_capstyle='butt')
    ax.annotate('', lft[i_valve] + 0.3 * out[i_valve], gas[0] - 1.8 * out[i_valve], arrowprops=arrow, zorder=4)
    ax.annotate('', inlet[0] + (5.4, 0.0), inlet[0] + (1.0, 0.0), arrowprops=arrow, zorder=4)
    text(inlet[0, 0] - 0.2, -4.4, r'$w_{lg}$', fontsize=LABEL, color=ink)
    callout(-21.0, 33.0, 'Lift gas', ['through a valve', 'at any depth'], gas[0] + (-1.0, 0.2), (-6.6, 35.6))

    # Heat to the formation: wavy arrows from the lower side of the tubing, between the reservoir and the valve
    s = np.linspace(0.0, 1.0, 40)
    waves = (16, 23, 30)
    for i in waves:
        along = np.array([-out[i, 1], out[i, 0]])
        wave = lft[i] - np.outer(0.6 + 3.6 * s, out[i]) + np.outer(0.42 * np.sin(4 * np.pi * s), along)
        ax.plot(wave[:, 0], wave[:, 1], color=ink2, lw=0.65, zorder=3)
        ax.annotate('', wave[-1] - 1.5 * out[i], wave[-1], arrowprops=dict(arrow, mutation_scale=4.5, lw=0.65))
    lines = ['heat exchange with the', 'formation, and friction']
    callout(-21.0, 46.0, 'Energy balance', lines, lft[waves[1]] - 6.6 * out[waves[1]], (2.6, 50.0))

    # The inflow: arrows into the bottom of the well, within the reservoir
    for i in (1, 4):
        for wall, sign in ((right, 1.0), (lft, -1.0)):
            ax.annotate('', wall[i] + 0.3 * sign * out[i], wall[i] + 4.2 * sign * out[i], arrowprops=arrow, zorder=4)
    text(-21.0, y_bottom - 1.3, 'Inflow', fontsize=LABEL, fontweight='bold', color=ink)
    text(-12.7, y_bottom - 1.3, r'from the reservoir at $p_r$, $T_r$')
    text(-21.0, y_bottom + 1.45, 'productivity index, Vogel or Forchheimer')

    # The parts of the model that are not at one place
    lines = ['gas and liquid (oil and water),', 'slip by flow regime, wall friction']
    callout(11.0, 6.2, 'Tubing flow', lines, (hw, 5.3), (10.2, 5.3))
    callout(13.0, 17.4, 'Fluid', ['black oil with dissolved gas,', 'real gas and water,', 'or a PVT table'])
    i_bend = int(np.argmin(np.abs(d['tvd'] - 850.0)))  # In the build section
    callout(
        20.5, 32.6, 'Trajectory', ['vertical or deviated,', 'from a survey'], right[i_bend] + (0.2, 0.0), (19.7, 31.7)
    )

    # The regimes' names along the tubing, each at the middle of the points where it is the most probable
    regime = np.array(d['regime'])
    for name in dict.fromkeys(d['regime']):
        points = np.flatnonzero(regime == name)
        i = int(points[len(points) // 2])
        t = path[min(i + 1, n - 1)] - path[max(i - 1, 0)]
        angle = math.degrees(math.atan2(t[1], -t[0]))  # Along the tubing, read from the top of the well down
        at = right[i] + 1.5 * out[i]
        text(
            at[0],
            at[1],
            name.replace('-', ' '),
            rotation=angle,
            rotation_mode='anchor',
            ha='center',
            va='bottom',
            style='italic',
        )

    # The tracks: the operating point against depth (m), on the schematic's scale
    w_tr, gap, y_name = 13.5, 3.25, 1 + 5.4 / (depth * scale)  # y_name: the names' baseline, in the axes' heights
    tracks = [
        ('Pressure\n(bar)', d['p'], (0, 200), (0, 100, 200)),
        ('Temperature\n(°C)', d['T'], (0, 100), (0, 50, 100)),
        ('Gas volume\nfraction', d['alpha'], (0, 1), (0, 0.5, 1)),
    ]
    name_kw = dict(fontsize=LABEL, color=ink2, va='bottom', linespacing=1.15)
    for k, (name, values, lim, xticks) in enumerate(tracks):
        tr = axes_mm(fig, x0 + w_ax + 9.0 + k * (w_tr + gap), y0 + 7.0 + above, w_tr, depth * scale)
        tr.set_xlim(lim)
        tr.set_ylim(depth, 0.0)
        tr.set_xticks(xticks, [f'{v:g}' for v in xticks])
        tr.set_yticks(np.arange(0, depth, 500))
        tr.xaxis.tick_top()
        tr.tick_params(axis='y', labelleft=(k == 0), length=2.0 if k == 0 else 0.0)
        tr.grid(True)
        tr.set_axisbelow(True)
        ticks = tr.xaxis.get_major_ticks()
        ticks[0].label2.set_ha('left')
        ticks[-1].label2.set_ha('right')
        tr.axhline(d['tvd'][i_valve], color=muted, lw=0.6, dashes=(2.2, 1.8), zorder=1.5)
        tr.plot(values, d['tvd'], color=ink, lw=1.15, zorder=3)
        tr.text(0.5, y_name, name, transform=tr.transAxes, ha='center', **name_kw)
        if k == 0:
            tr.text(-1.6 / w_tr, y_name, 'Depth\n(m)', transform=tr.transAxes, ha='right', **name_kw)
            tr.text(5, d['tvd'][i_valve] - 28, 'valve', fontsize=SMALL, color=muted, va='bottom')
        if k == 1:
            tr.plot(d['T_formation'], d['tvd'], color=muted, lw=0.8, dashes=(2.2, 1.8), zorder=2)
            tr.text(
                27, 1120, 'formation', fontsize=SMALL, color=muted, rotation=-56, rotation_mode='anchor', ha='center'
            )
        if k == 2:
            tr.fill_betweenx(d['tvd'], 0, values, color=ink, alpha=0.08, lw=0, zorder=1)


# --- Panel b: the flow regimes ---------------------------------------------------------------------------------------
def draw_regimes(fig, th, d, profile, x0, y0):
    panel_title(fig, th, x0, y0 + 3.4, 'b', 'Flow regimes')
    ax = axes_mm(fig, x0 + 11.0, y0 + 6.5, 42.5, 27.0)
    surface = np.array(to_rgb(th['surface']))
    washes = [surface + th['wash'] * (np.array(to_rgb(th['regimes'][r])) - surface) for r in HASAN_REGIMES]
    image = np.clip(np.tensordot(d['probs'], np.array(washes), axes=(0, 0)), 0.0, 1.0)
    extent = (d['lg_gs'][0], d['lg_gs'][-1], d['lg_ls'][0], d['lg_ls'][-1])
    ax.imshow(image, origin='lower', extent=extent, aspect='auto', interpolation='bilinear', zorder=0)
    ax.set_xticks([-1, 0, 1], ['0.1', '1', '10'])
    ax.set_yticks([-1, 0, 1], ['0.1', '1', '10'])
    ax.set_xlabel('Superficial gas velocity (m/s)')
    ax.set_ylabel('Superficial liquid\nvelocity (m/s)', linespacing=1.15)
    for spine in ax.spines.values():
        spine.set_visible(False)

    kw = dict(fontsize=SMALL, color=th['ink'], ha='center', va='center')
    names = {'bubbly': (-0.95, -0.55), 'slug': (-0.22, -1.0), 'annular': (1.2, -0.6), 'dispersed bubbly': (-0.62, 0.84)}
    for name, (x, y) in names.items():
        ax.text(x, y, name, **kw)
    ax.text(1.0, 0.8, 'churn', rotation=40, rotation_mode='anchor', **kw)  # Along its band

    # The well of panel a, from the bottomhole to the wellhead
    x, y = np.log10(profile['v_gs']), np.log10(profile['v_ls'])
    ax.plot(x, y, color=th['surface'], lw=2.6, zorder=3)
    ax.plot(x, y, color=th['ink'], lw=1.15, zorder=4)
    ax.plot(x[0], y[0], 'o', ms=3.6, mfc=th['surface'], mec=th['ink'], mew=1.0, zorder=5)
    ax.plot(x[-1], y[-1], 'o', ms=3.6, mfc=th['ink'], mec=th['surface'], mew=0.6, zorder=5)
    kw = dict(fontsize=SMALL, color=th['ink2'], ha='center', va='top')
    ax.text(x[0], y[0] - 0.15, 'bottomhole', **kw)
    ax.text(x[-1] + 0.1, y[-1] - 0.15, 'wellhead', **kw)


# --- Panel c: the solutions ------------------------------------------------------------------------------------------
def draw_solutions(fig, th, d, x0, y0):
    panel_title(fig, th, x0, y0 + 3.4, 'c', 'Stable and unstable solutions')
    ax = axes_mm(fig, x0 + 11.0, y0 + 6.5, 42.5, 27.0)
    plot_axes(ax)
    ax.set_xlim(0, 20)
    ax.set_ylim(133, 182)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([140, 150, 160, 170, 180])
    ax.set_xlabel('Liquid rate (kg/s)')
    ax.set_ylabel('Bottomhole\npressure (bar)', linespacing=1.15)
    w = np.array([0.0, 20.0])
    ax.plot(w, d['p_r'] - w / d['k_l'], color=th['muted'], lw=1.3, zorder=2)
    ax.plot(d['rates'], d['p_0'], color=th['ink'], lw=1.3, zorder=3)
    kw = dict(fontsize=SMALL, color=th['ink2'], ha='center', va='bottom', rotation_mode='anchor')
    ax.text(17.6, 144.2, 'inflow', rotation=-16, **kw)
    ax.text(16.9, 158.3, 'tubing and choke', rotation=29, **kw)
    kw = dict(fontsize=SMALL, color=th['ink'])
    for w_res, p_0, label in d['roots']:
        if label == 'stable':
            ax.plot(w_res, p_0, 'o', ms=6.6, mfc=th['surface'], mec='none', zorder=4.5)
            ax.plot(w_res, p_0, 'o', ms=5.0, mfc=th['accent'], mec=th['surface'], mew=0.8, zorder=5)
            arrowprops = leader(th, shrinkA=0.5, shrinkB=3.2)
            ax.annotate(
                'stable: the operating point',
                (w_res, p_0),
                (w_res, 135.0),
                ha='center',
                va='bottom',
                arrowprops=arrowprops,
                **kw,
            )
        else:
            ax.plot(w_res, p_0, 'o', ms=5.0, mfc=th['surface'], mec=th['accent'], mew=1.3, zorder=5)
            arrowprops = leader(th, shrinkA=1.0, shrinkB=3.2)
            ax.annotate(label, (w_res, p_0), (w_res + 2.4, 168.5), ha='left', va='center', arrowprops=arrowprops, **kw)
    kw = dict(transform=ax.transAxes, fontsize=SMALL, color=th['muted'], ha='right', va='top', linespacing=1.15)
    ax.text(0.985, 0.97, 'the well in a\nwithout lift gas', **kw)


# --- Panel d: the datasets -------------------------------------------------------------------------------------------
def draw_wells(fig, th, wells, x0, y0):
    panel_title(fig, th, x0, y0 + 3.4, 'd', 'Datasets of thousands of sampled wells')
    note = f'{DATASET_WELLS:,} wells × {DATASET_SAMPLES} operating points per dataset; shown are {len(wells)} wells drawn by the sampler'
    fig.text((x0 + 4.2) / WIDTH, 1 - (y0 + 7.0) / HEIGHT, note, fontsize=SMALL, color=th['ink2'], va='baseline')

    # The axes are in km, to scale
    w_ax, h_ax, deepest = 84.0, 27.0, 7.0  # mm, mm, km
    km = deepest / h_ax  # km per mm
    ax = axes_mm(fig, x0 + 8.5, y0 + 9.5, w_ax, h_ax)
    ax.set_xlim(0.0, w_ax * km)
    ax.set_ylim(deepest, 0.0)
    for side in ('top', 'right', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([0, 2, 4, 6])
    ax.set_ylabel('Depth (km)')
    ax.set_axisbelow(True)
    ax.grid(True, axis='y')
    ax.axhline(0.0, color=th['axis'], lw=0.8, zorder=1)

    # Each well from its own wellhead, ordered by its reach, so that the trajectories do not cross much
    order = sorted(range(len(wells)), key=lambda k: wells[k]['x'][-1])
    reach = max(w['x'][-1] for w in wells) / 1000.0
    heads = np.linspace(1.2 * km, w_ax * km - reach - 0.6 * km, len(wells))
    for head, k in zip(heads, order):
        color = th['fluids'][FLUID_CLASSES.index(wells[k]['fluid_class'])]
        ax.plot(head + wells[k]['x'] / 1000.0, wells[k]['tvd'] / 1000.0, color=color, lw=0.9, zorder=3, clip_on=False)
    fig.legend(
        [Line2D([], [], color=c, lw=1.6) for c in th['fluids']],
        [c.replace('_', ' ') for c in FLUID_CLASSES],
        loc='upper left',
        bbox_to_anchor=((x0 + 7.5) / WIDTH, 1 - (y0 + 38.3) / HEIGHT),
        ncols=len(FLUID_CLASSES),
        handlelength=1.3,
        handletextpad=0.5,
        columnspacing=1.4,
        borderpad=0,
        borderaxespad=0,
        labelcolor=th['ink2'],
    )


# --- Panel e: calibration --------------------------------------------------------------------------------------------
def draw_calibration(fig, th, d, x0, y0):
    panel_title(fig, th, x0, y0 + 3.4, 'e', 'Calibration to data')
    ax = axes_mm(fig, x0 + 10.0, y0 + 9.5, 30.0, 27.0)
    plot_axes(ax)
    ax.set_xlim(0.1, 1.03)
    ax.set_ylim(0, 1500)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1'])
    ax.set_yticks([0, 500, 1000, 1500], ['0', '500', '1,000', '1,500'])
    ax.set_xlabel('Choke position')
    ax.set_ylabel('Oil rate (Sm³/d)')
    styles = {
        'measured': dict(marker='o', ls='none', ms=3.6, mfc=th['ink'], mec=th['surface'], mew=0.6, zorder=4),
        'first guess': dict(color=th['muted'], lw=1.1, zorder=2),
        'calibrated': dict(color=th['accent'], lw=1.4, zorder=3),
    }
    ax.plot(d['u'], d['before'], **styles['first guess'])
    ax.plot(d['u'], d['after'], **styles['calibrated'])
    ax.plot(d['u_measured'], d['q_measured'], **styles['measured'])
    ax.legend(
        [Line2D([], [], **style) for style in styles.values()],
        list(styles),
        loc='lower right',
        bbox_to_anchor=(1.03, -0.02),
        handlelength=1.2,
        handletextpad=0.5,
        labelspacing=0.25,
        borderpad=0,
        labelcolor=th['ink2'],
    )


# --- Panel f: sensitivities ------------------------------------------------------------------------------------------
def draw_sensitivity(fig, th, d, x0, y0):
    panel_title(fig, th, x0, y0 + 3.4, 'f', 'Sensitivity studies')
    ax = axes_mm(fig, x0 + 8.5, y0 + 9.5, 31.5, 27.0)
    plot_axes(ax)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 22)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_xlabel('Lift-gas rate (kg/s)')
    ax.set_ylabel('Liquid rate (kg/s)')
    for color, rates in zip(th['ramp'], d['curves'].values()):
        ax.plot(d['w_lg'], rates, color=color, lw=1.4, zorder=3)
    ax.plot(*d['point'], 'o', ms=6.4, mfc=th['surface'], mec='none', zorder=4)
    ax.plot(*d['point'], 'o', ms=4.6, mfc=th['ink'], mec=th['surface'], mew=0.6, zorder=5)
    kw = dict(fontsize=SMALL, color=th['muted'], ha='center', va='center')
    ax.annotate('the well in a', d['point'], (1.0, 20.6), arrowprops=leader(th, shrinkA=1.0, shrinkB=3.0), **kw)
    legend = ax.legend(
        [Line2D([], [], color=c, lw=1.4) for c in th['ramp']],
        [f'{p_r:.0f} bar' for p_r in d['curves']],
        title='Reservoir pressure',
        title_fontsize=SMALL,
        alignment='left',
        loc='lower right',
        bbox_to_anchor=(1.03, -0.02),
        handlelength=1.2,
        handletextpad=0.5,
        labelspacing=0.25,
        borderpad=0,
        labelcolor=th['ink2'],
    )
    legend.get_title().set_color(th['ink2'])


# --- The figure ------------------------------------------------------------------------------------------------------
def draw(data: dict, theme: str) -> Figure:
    """The figure in a theme of THEMES, from compute()'s data."""
    th = THEMES[theme]
    with mpl.rc_context(rc(th)):
        fig = Figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), facecolor=th['surface'])
        draw_model(fig, th, data['profile'], 3.0, 3.0)
        draw_regimes(fig, th, data['regime_map'], data['profile'], 133.0, 3.0)
        draw_solutions(fig, th, data['solutions'], 133.0, 50.5)
        draw_wells(fig, th, data['wells'], 3.0, 102.0)
        draw_calibration(fig, th, data['calibration'], 100.5, 102.0)
        draw_sensitivity(fig, th, data['lift'], 146.0, 102.0)
    return fig


def save(fig: Figure, path: Path, theme: str):
    """Writes a figure in its theme, with nothing in the file that changes from one run to the next."""
    metadata = {'.svg': {'Date': None}, '.pdf': {'CreationDate': None}}.get(path.suffix)
    with mpl.rc_context(rc(THEMES[theme])):
        # The dpi is the regime map's in an svg or a pdf
        fig.savefig(path, dpi=600 if path.suffix == '.png' else 300, facecolor=fig.get_facecolor(), metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description="Draw the project's main figure, in a light and a dark version.")
    parser.add_argument('--out', type=Path, default=Path('docs'), help='the folder of the files (default: docs)')
    parser.add_argument('--format', default='svg', choices=('svg', 'pdf', 'png'))
    args = parser.parse_args()
    print('Solving the wells of the panels (about a minute) ...')
    data = compute()
    for theme, name in (('light', 'manywells'), ('dark', 'manywells-dark')):
        path = args.out / f'{name}.{args.format}'
        save(draw(data, theme), path, theme)
        print(f'Wrote {path}')


if __name__ == '__main__':
    main()
