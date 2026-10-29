"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 28 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The prototype of specs/features/019-five-flow-regimes.md: a five-regime slip closure after Hasan, Kabir and Sayarpour
(2010), with regime probabilities from their hierarchy's rules (a soft decision tree, no fitted weights).

Every rule compares a dimensionless quantity with its threshold; its soft truth is (1 + tanh(z / 2w)) / 2 of its
margin z. The probabilities are the products along the hierarchy's decision tree:

    annular    = A                      A = [j_g > v_gc] and [lambda > 0.7]
    churn      = (1-A) * M * G          M = [j_m > v_ms]
    dispersed  = (1-A) * M * (1-G)      G = [j_g > 1.08 j_l], i.e. lambda > 1.08/2.08
    slug       = (1-A) * (1-M) * S      S = [alpha_b > alpha_t], i.e. j_g > v_gb (A-14)
    bubbly     = (1-A) * (1-M) * (1-S)

Every input is a superficial velocity or a fluid property, so at a point C_0 and v_inf do not depend on alpha, and
the slip law has one void fraction. Hasan5SlipModel is the closure as a SlipModel for the CasADi backend (a
prototype: it carries D as a field, as classify() has no D argument, and patch_closure_rows() passes the cell's
Darcy factor when f is None). five_regime_wells.py solves sampled wells with it.

From the repository root, the closure on a grid of superficial velocities against Hasan's model (slip_models.py)
and against develop's fitted classifier (a few minutes):

    uv run python plans/evidence/five_regime_slip.py
"""

from dataclasses import dataclass

import casadi as ca
import numpy as np

from manywells.slip import SlipModel
from manywells.units import STD_GRAVITY as g

LAMBDA_CHURN = 1.08 / 2.08  # j_g > 1.08 j_l  <=>  lambda > 0.519
REGIMES = ('annular', 'churn', 'dispersed', 'slug', 'bubbly')
C0 = {'annular': 1.0, 'churn': 1.15, 'dispersed': 1.2, 'slug': 1.2, 'bubbly': 1.2}  # Hasan et al., Table 1, upflow
MERGED = {'annular': 'annular', 'churn': 'slug-churn', 'slug': 'slug-churn', 'dispersed': 'bubbly', 'bubbly': 'bubbly'}


def terms(jg, jl, rho_g, rho_l, sigma, D, cos, f=0.02, floor=0.1, sqrt_eps=1e-6, m=np):
    """The rule quantities and thresholds, and the rise velocities; m is np or ca."""
    jm = jg + jl
    drho = rho_l - rho_g
    vb = 1.53 * (g * sigma * drho / rho_l**2) ** 0.25  # (A-8), SLIP-4
    lam = jg / (jm + floor * vb)  # no-slip gas fraction, floored
    sin = m.sqrt(1 - cos**2)
    vT = 0.35 * m.sqrt(g * D * drho / rho_l) * m.sqrt(cos) * (1 + sin) ** 1.2  # (A-9), (A-10), SLIP-5, SLIP-10
    vgc = 3.1 * (g * sigma * drho / rho_g**2) ** 0.25  # (A-19)
    k = 2 * (f / (2 * D)) ** 0.4 * (rho_l / sigma) ** 0.6 * m.sqrt(0.4 * sigma / (g * drho))
    vms = ((0.725 + 4.15 * m.sqrt(lam + sqrt_eps)) / k) ** (1 / 1.2)  # (A-17)
    alpha_b = jg / (1.2 * jm + vb)  # alpha with bubbly parameters
    alpha_t = cos / (2.8 + 1.2 * cos)  # (A-14) in alpha; 0.25 vertical
    return dict(jm=jm, lam=lam, vb=vb, vT=vT, vgc=vgc, vms=vms, alpha_b=alpha_b, alpha_t=alpha_t)


def margins(t, jg):
    """Signed dimensionless margin of each rule: positive where the rule holds."""
    return dict(
        A1=jg / t['vgc'] - 1,
        A2=t['lam'] - 0.7,
        M=t['jm'] / t['vms'] - 1,
        G=t['lam'] - LAMBDA_CHURN,
        S=t['alpha_b'] - t['alpha_t'],
    )


def probabilities(mg, w, m=np):
    """Regime probabilities (dict) from the rule margins; w = 0 gives the crisp hierarchy."""
    if w == 0:
        s = {k: (v > 0) * 1.0 for k, v in mg.items()}
    else:
        s = {k: 0.5 * (1 + m.tanh(v / (2 * w))) for k, v in mg.items()}
    A = s['A1'] * s['A2']
    M, G, S = s['M'], s['G'], s['S']
    return dict(
        annular=A,
        churn=(1 - A) * M * G,
        dispersed=(1 - A) * M * (1 - G),
        slug=(1 - A) * (1 - M) * S,
        bubbly=(1 - A) * (1 - M) * (1 - S),
    )


def parameters(p, t):
    """C_0 and v_inf, the regimes' Table 1 values weighted by their probabilities."""
    C_0 = sum(p[r] * C0[r] for r in REGIMES)
    v_inf = t['vb'] * (p['bubbly'] + p['dispersed']) + t['vT'] * (p['slug'] + p['churn'])
    return C_0, v_inf


def label(p):
    """The most probable regime at each point."""
    P = np.stack([np.asarray(p[r], dtype=float).ravel() for r in REGIMES])
    return np.array(REGIMES)[P.argmax(axis=0)]


def alpha_explicit(jg, jl, rho_g, rho_l, sigma, D, cos, w=0.02, **kw):
    """The void fraction (the slip law's one root), the probabilities, C_0 and v_inf at given j_g, j_l."""
    t = terms(jg, jl, rho_g, rho_l, sigma, D, cos, **kw)
    p = probabilities(margins(t, jg), w)
    C_0, v_inf = parameters(p, t)
    return jg / (C_0 * t['jm'] + v_inf), p, C_0, v_inf


@dataclass(frozen=True)
class Hasan5SlipModel(SlipModel):
    """The prototype as a SlipModel. classify() merges the five regimes into SLIP-8's three labels."""

    w: float = 0.02  # logistic width of every rule
    f: float = 0.02  # friction factor in (A-17); None: the cell's f_D, passed by patch_closure_rows()
    floor: float = 0.1  # lambda = j_g / (j_m + floor * v_inf_b)
    sqrt_eps: float = 1e-6  # sqrt(lambda + sqrt_eps) in (A-17)
    D: float = 0.1  # the well's diameter, for classify()

    def _probs(self, v_g, v_l, alpha, rho_g, rho_l, sigma, D, cos_incl, f_D=None):
        jg, jl = alpha * v_g, (1 - alpha) * v_l
        f = self.f if self.f is not None else (f_D if f_D is not None else 0.02)
        t = terms(jg, jl, rho_g, rho_l, sigma, D, cos_incl, f, self.floor, self.sqrt_eps, m=ca)
        return probabilities(margins(t, jg), self.w, m=ca), t

    def classify(self, v_g, v_l, alpha, rho_g, rho_l, sigma, cos_incl):
        # The root's labels only; with f = None they take f = 0.02 (five_regime_wells.py labels the points itself)
        p, _ = self._probs(v_g, v_l, alpha, rho_g, rho_l, sigma, self.D, cos_incl)
        return ca.vertcat(p['annular'], p['slug'] + p['churn'], p['bubbly'] + p['dispersed'])

    def identify_parameters(self, v_g, v_l, alpha, rho_g, rho_l, sigma, D, cos_incl, f_D=None):
        p, t = self._probs(v_g, v_l, alpha, rho_g, rho_l, sigma, D, cos_incl, f_D)
        return parameters(p, t)


def patch_closure_rows():
    """Make the discretization pass the cell's Darcy factor to a Hasan5SlipModel with f = None."""
    from manywells import discretization as disc

    original = disc.closure_rows

    def closure_rows(wp, s, cos_incl):
        if not (isinstance(wp.slip, Hasan5SlipModel) and wp.slip.f is None):
            return original(wp, s, cos_incl)
        fluid = wp.fluid
        sigma = fluid.surface_tension(s.p, s.T, s.rho_l)
        f_D = wp.friction.friction_factor(s, fluid, wp.geometry.D)
        C_0, v_inf = wp.slip.identify_parameters(
            s.v_g, s.v_l, s.alpha, s.rho_g, s.rho_l, sigma, wp.geometry.D, cos_incl, f_D=f_D
        )
        return [
            s.v_g - C_0 * s.v_m - v_inf,
            fluid.gas_law_row(s.p, s.T, s.rho_g),
            s.rho_l - fluid.liquid_density(s.p, s.T),
        ]

    disc.closure_rows = closure_rows


def grid():
    """The closure on a 60 x 60 grid of superficial velocities, against Hasan's model and develop's classifier."""
    from slip_models import FLUIDS, all_roots, hasan_alpha, mw_params

    JG, JL = [a.ravel() for a in np.meshgrid(np.geomspace(0.02, 30, 60), np.geomspace(0.01, 5, 60))]

    def q(x):
        return f'{np.mean(x):.3f} / {np.percentile(x, 95):.3f} / {np.max(x):.3f}'

    print('|d alpha| as mean / 95th percentile / max over the grid; "crisp" is the hierarchy (w = 0, no floor)')
    for fname, fl in FLUIDS.items():
        for cos in [1.0, 0.5, 0.0]:
            a_h = np.array([hasan_alpha(a, b, *fl, cos)[0] for a, b in zip(JG, JL)])
            a_c, p_c, *_ = alpha_explicit(JG, JL, *fl, cos, w=0, floor=0.0, sqrt_eps=0.0)
            lab_c = label(p_c)
            shares = ', '.join(f'{r} {np.mean(lab_c == r):.0%}' for r in REGIMES)
            print(f'== {fname}, cos {cos}: crisp shares {shares}')
            print(f'   crisp vs Hasan (Table 1 values, no blends A-13, A-20 to A-22): {q(abs(a_c - a_h))}')
            mw = [all_roots(mw_params, a, b, *fl, cos) for a, b in zip(JG, JL)]
            uniq = np.array([len(r) == 1 for r in mw])
            a_mw = np.array([r[0] if len(r) == 1 else np.nan for r in mw])
            print(
                f'   develop vs Hasan, {(~uniq).sum()} points with several roots left out: '
                f'{q(abs(a_mw[uniq] - a_h[uniq]))}'
            )
            for w in [0.01, 0.02, 0.05]:
                a_s, p_s, *_ = alpha_explicit(JG, JL, *fl, cos, w=w)
                print(
                    f'   w {w:<4}: labels as crisp {np.mean(label(p_s) == lab_c):.1%}; vs crisp {q(abs(a_s - a_c))}; '
                    f'vs Hasan {q(abs(a_s - a_h))}'
                )


def ranges(csv=None):
    """The sampler's gas-liquid ratios, and the liquid rates of five_regime_wells.py's proposal, against the ranges
    of Hasan et al.'s Table 2 (GLR 104 to 5615 scf/STB, liquid 100 to 27,760 STB/D)."""
    from manywells.sampling.wells import sample_well
    from manywells.units import P_REF, T_REF

    def liquid_density(d):
        f_g, f_o, f_w = d.fractions
        return (f_o + f_w) / (f_o / d.rho_o + f_w / d.rho_w)

    glr = []
    for well in range(2000):
        d = sample_well(2026, well)
        f_g, f_o, f_w = d.fractions
        rho_g = P_REF / (d.R_s * T_REF)
        glr.append((f_g / rho_g) / (f_o / d.rho_o + f_w / d.rho_w) * 5.615)  # Sm3/Sm3 to scf/STB
    glr = np.array(glr)
    print(
        f'formation GLR of 2000 sampled wells (scf/STB): 5th percentile {np.percentile(glr, 5):.0f}, median '
        f'{np.median(glr):.0f}, 95th {np.percentile(glr, 95):.0f}; inside Table 2 {np.mean((glr >= 104) & (glr <= 5615)):.0%}, '
        f'above {np.mean(glr > 5615):.0%}, below {np.mean(glr < 104):.0%}'
    )
    if csv:
        import pandas as pd

        df = pd.read_csv(csv)
        df = df[(df.variant == 'proposal') & df.op]
        q = [r.w_res / liquid_density(sample_well(2026, r.well)) * 86400 / 0.158987 for r in df.itertuples()]
        print(
            f"liquid rate at the proposal's {len(q)} operating points (STB/D): {min(q):.0f} to {max(q):.0f}; "
            f'bottomhole pressure {df.pbh.min():.0f} to {df.pbh.max():.0f} bar'
        )


if __name__ == '__main__':
    import sys

    if sys.argv[1:2] == ['ranges']:
        ranges(*sys.argv[2:3])
    else:
        grid()
