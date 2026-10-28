"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 28 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

develop's slip closure (SLIP-1 to SLIP-11) against Hasan, Kabir and Sayarpour (2010) and Shi et al. (2005,
two-phase), the comparison behind the motivation of specs/features/019-five-flow-regimes.md. For each model it
solves the slip law for the void fraction at given superficial velocities (j_g, j_l),
    h(alpha) = alpha * (C_0(alpha, ...) * j_m + v_d(alpha, ...)) - j_g = 0,
and reports every root, at two fluid states, four inclinations and a grid of superficial velocities. Hasan's model is
evaluated point by point (cocurrent upflow in tubing, no top-down hysteresis, f = 0.02 in (A-17)); Shi's Ku is read
from their Fig. 3. SI units. From the repository root (about a minute):

    uv run python plans/evidence/slip_models.py
"""
import numpy as np
from scipy.optimize import brentq

g = 9.80665

# ---------------------------------------------------------------- ManyWells (develop), numpy copy of slip.py
def mw_params(alpha, jg, jl, rho_g, rho_l, sigma, D, cos):
    vgs, vls = jg, jl
    ann = 3.1 * (g * sigma * (rho_l - rho_g) / rho_g**2) ** 0.25
    c1 = np.tanh(vgs - ann)
    c2 = np.tanh(2 * (alpha - 0.7))
    c3 = np.tanh(vgs - 1.08 * vls)
    c4 = np.tanh(2 * (alpha - 0.25 * cos))
    ya = 3.17715258 * c1 + 6.81938489 * c2 + 0.30182974 * c3 + 3.58362465 * c4 - 3.92904391
    ys = -1.47973427 * c1 - 4.34033317 * c2 + 2.58200006 * c3 + 3.49656911 * c4 - 1.46509477
    yb = -1.6974183 * c1 - 2.47905172 * c2 - 2.8838298 * c3 - 7.08019376 * c4 + 5.39413869
    e = np.exp(np.stack([ya, ys, yb]))
    pa, ps, pb = e / e.sum(axis=0)
    vb = 1.53 * (g * sigma * (rho_l - rho_g) / rho_l**2) ** 0.25
    sin = np.sqrt(1 - cos**2)
    vT = 0.35 * np.sqrt(g * D * (1 - rho_g / rho_l)) * np.sqrt(cos) * (1 + sin) ** 1.2
    C0 = 1.0 * pa + 1.175 * ps + 1.2 * pb
    vinf = ps * vT + pb * vb
    return C0, vinf


# ---------------------------------------------------------------- Hasan, Kabir and Sayarpour (2010), cocurrent upflow in tubing
def hasan_alpha(jg, jl, rho_g, rho_l, sigma, D, cos, f=0.02):
    """Explicit in (j_g, j_l): regime from superficial velocities, then (A-7). Returns alpha, C0, vinf, regime."""
    sin = np.sqrt(max(1 - cos**2, 0.0))
    drho = rho_l - rho_g
    jm = jg + jl
    vb = 1.53 * (g * drho * sigma / rho_l**2) ** 0.25                                 # (A-8)
    vT = 0.35 * np.sqrt(g * D * drho / rho_l) * np.sqrt(cos) * (1 + sin) ** 1.2       # (A-9), (A-10), F_a = 1
    vgb = (1.2 * jl + vb) / (4 - 1.2) * cos                                           # (A-14)
    lhs = 2 * (f / (2 * D)) ** 0.4 * (rho_l / sigma) ** 0.6 * np.sqrt(0.4 * sigma / (g * drho))
    vms = ((0.725 + 4.15 * np.sqrt(jg / jm)) / lhs) ** (1 / 1.2)                       # (A-17)
    vgc = 3.1 * (g * sigma * drho / rho_g**2) ** 0.25                                 # (A-19)

    def E(a, x):  # exp(-0.1 a / (x - a)) for x > a, else 0
        return np.exp(-0.1 * a / (x - a)) if x > a else 0.0

    def vbar():   # (A-13)
        e = E(vgb, jg)
        return vb * (1 - e) + vT * e

    if jg > vgc:
        e = E(vgc, jg)
        C0 = 1.15 * (1 - e) + 1.0 * e                                                  # (A-22)
        a = jg / (C0 * jm)
        if a > 0.7:
            return a, C0, 0.0, 'annular'
    if jm > vms and jg > 1.08 * jl:                                                   # churn
        e = E(vms, jm)
        C0 = 1.2 * (1 - e) + 1.15 * e                                                  # (A-21)
        v = vbar()
        return jg / (C0 * jm + v), C0, v, 'churn'
    if jm > vms:                                                                      # dispersed bubbly
        return jg / (1.2 * jm + vb), 1.2, vb, 'disp. bubbly'
    if jg > vgb:                                                                      # slug, (A-20) with C_ob = C_os = 1.2
        v = vbar()
        return jg / (1.2 * jm + v), 1.2, v, 'slug'
    return jg / (1.2 * jm + vb), 1.2, vb, 'bubbly'


# ---------------------------------------------------------------- Shi et al. (2005), ECLIPSE drift-flux
SHI_SETS = {
    'Shi opt': dict(A=1.0, B=0.3, a1=0.06, a2=0.21, m0=1.85, n1=0.21, n2=0.95),   # Table 1, Eq. 27
    'Shi A=1.2': dict(A=1.2, B=0.6, a1=0.06, a2=0.12, m0=1.27, n1=0.24, n2=1.08), # Eq. 28
    'ECLIPSE dflt': dict(A=1.2, B=0.3, a1=0.20, a2=0.40, m0=1.0, n1=0.5, n2=2.0), # Fig. 12, Eq. 22
}


def ku_of_dhat(dhat):
    # Fig. 3, read as the ECLIPSE table: 0 below 2, rising to 3.2 at 50 and above
    return np.interp(dhat, [2, 4, 10, 14, 20, 28, 50], [0, 1.0, 2.1, 2.5, 2.8, 3.0, 3.2])


def shi_params(alpha, jg, jl, rho_g, rho_l, sigma, D, cos, A, B, a1, a2, m0, n1, n2, Fv=1.0):
    sin = np.sqrt(np.maximum(1 - cos**2, 0.0))
    drho = rho_l - rho_g
    jm = jg + jl
    Vc = (sigma * g * drho / rho_l**2) ** 0.25                                         # (9)
    Ku = ku_of_dhat(np.sqrt(g * drho / sigma) * D)                                    # (11), Fig. 3
    Vsgf = Ku * np.sqrt(rho_l / rho_g) * Vc                                            # (10)
    beta = np.maximum(alpha, Fv * alpha * abs(jm) / Vsgf)                              # (6)
    gam = np.clip((beta - B) / (1 - B), 0, 1)                                          # (5)
    C0 = A / (1 + (A - 1) * gam**2)                                                    # (4)
    K = np.where(alpha <= a1, 1.53 / C0,
                 np.where(alpha >= a2, Ku, 1.53 / C0 + (Ku - 1.53 / C0) * (alpha - a1) / (a2 - a1)))
    Vd = (1 - alpha * C0) * C0 * K * Vc / (alpha * C0 * np.sqrt(rho_g / rho_l) + 1 - alpha * C0)  # (15)
    m = m0 * cos**n1 * (1 + sin) ** n2                                                 # (23)
    return C0, Vd * m


# ---------------------------------------------------------------- root finding
AG = np.linspace(1e-5, 1 - 1e-5, 20001)


def all_roots(params, jg, jl, *fluid):
    def h(a):
        C0, v = params(a, jg, jl, *fluid)
        return a * (C0 * (jg + jl) + v) - jg
    hv = h(AG)
    idx = np.nonzero(np.sign(hv[:-1]) * np.sign(hv[1:]) < 0)[0]
    return [brentq(lambda a: float(h(np.array(a))), AG[i], AG[i + 1], xtol=1e-12) for i in idx]


def models():
    yield 'ManyWells', mw_params
    for name, ps in SHI_SETS.items():
        yield name, (lambda ps: lambda a, jg, jl, *fl: shi_params(a, jg, jl, *fl, **ps))(ps)


FLUIDS = {
    # name: (rho_g, rho_l, sigma, D)
    'deep (rho_g=80)': (80.0, 780.0, 0.018, 0.127),
    'wellhead (rho_g=15)': (15.0, 820.0, 0.025, 0.127),
}


def main():
    # 0. check the numpy copy against the package
    from manywells.slip import SlipModel
    rng = np.random.default_rng(0)
    sm = SlipModel()
    err = 0.0
    for _ in range(200):
        a = rng.uniform(0.01, 0.99); vg = rng.uniform(0.1, 30); vl = rng.uniform(0.01, 5)
        rg = rng.uniform(1, 200); rl = rng.uniform(600, 1050); s = rng.uniform(0.005, 0.07)
        D = rng.uniform(0.07, 0.17); c = rng.uniform(0, 1)
        C0, v = sm.identify_parameters(vg, vl, a, rg, rl, s, D, c)
        C0n, vn = mw_params(a, a * vg, (1 - a) * vl, rg, rl, s, D, c)
        err = max(err, abs(float(C0) - C0n), abs(float(v) - vn))
    print(f'numpy copy of SlipModel: max abs diff {err:.1e} over 200 random points\n')

    for fname, fl in FLUIDS.items():
        rg, rl, s, D = fl
        dhat = np.sqrt(g * (rl - rg) / s) * D
        vgc = 3.1 * (g * s * (rl - rg) / rg**2) ** 0.25
        Vsgf = ku_of_dhat(dhat) * (s * g * (rl - rg) / rg**2) ** 0.25
        print(f'== {fname}: rho_l={rl}, sigma={s}, D={D} m; D_hat={dhat:.0f}, Ku={ku_of_dhat(dhat):.2f}; '
              f'Hasan v_gc={vgc:.2f} m/s, Shi V_sgf={Vsgf:.2f} m/s')
        for cos_name, cos in [('vertical', 1.0), ('45 deg', np.cos(np.pi / 4)), ('80 deg', np.cos(np.radians(80))),
                              ('horizontal', 0.0)]:
            print(f'-- {cos_name}')
            print(f'{"j_g":>5} {"j_l":>5} | {"Hasan":>22} | ' + ' | '.join(f'{n:>14}' for n, _ in models()))
            for jg, jl in [(0.05, 1.0), (0.3, 1.0), (0.6, 0.5), (1.0, 2.0), (2.0, 1.0), (5.0, 1.0), (10.0, 0.3),
                           (20.0, 0.05)]:
                ah, C0h, vh, reg = hasan_alpha(jg, jl, *fl, cos)
                cells = []
                for name, p in models():
                    r = all_roots(p, jg, jl, *fl, cos)
                    cells.append(','.join(f'{x:.3f}' for x in r) if r else 'none')
                print(f'{jg:5.2f} {jl:5.2f} | {ah:5.3f} {reg:>12} {C0h:4.2f} | ' + ' | '.join(f'{c:>14}' for c in cells))
            print()

    # 2. non-unique void fractions over a grid of superficial velocities
    print('== points with more than one root, over j_g in [0.02, 30], j_l in [0.01, 5] (40 x 40 log grid)')
    jgs = np.geomspace(0.02, 30, 40); jls = np.geomspace(0.01, 5, 40)
    for fname, fl in FLUIDS.items():
        for cos in [1.0, 0.5, 0.0]:
            counts = {}
            for name, p in models():
                n = sum(len(all_roots(p, jg, jl, *fl, cos)) > 1 for jg in jgs for jl in jls)
                counts[name] = n
            print(f'{fname:>22}, cos={cos:.1f}: ' + ', '.join(f'{k} {v}' for k, v in counts.items()) + ' of 1600')

    # 3. inclination factors on the drift velocity, relative and absolute
    print('\n== inclination factor on the drift velocity (theta from vertical)')
    print(f'{"theta":>5} | {"MW/Hasan (A-10)":>16} | {"Shi Eq.27":>16} | {"Shi Eq.28":>16} | {"ECLIPSE Eq.22":>16}')
    for th in [0, 15, 30, 45, 60, 70, 80, 85, 88, 90]:
        c, s_ = np.cos(np.radians(th)), np.sin(np.radians(th))
        f = [np.sqrt(c) * (1 + s_) ** 1.2, 1.85 * c**0.21 * (1 + s_) ** 0.95, 1.27 * c**0.24 * (1 + s_) ** 1.08,
             c**0.5 * (1 + s_) ** 2]
        f0 = [1.0, 1.85, 1.27, 1.0]
        print(f'{th:5d} | ' + ' | '.join(f'{x:6.2f} ({x / x0:4.2f}x)' for x, x0 in zip(f, f0)))

    # 4. bubbly-slug threshold in an inclined pipe: ManyWells 0.25 cos(theta) vs Hasan's alpha at v_sg = v_gb
    print('\n== bubbly-slug threshold in alpha: ManyWells 0.25 cos(theta); Hasan (A-14) gives cos/(4 - 1.2 (1 - cos))')
    for th in [0, 30, 45, 60, 80, 90]:
        c = np.cos(np.radians(th))
        print(f'{th:5d}: ManyWells {0.25 * c:.3f}, Hasan {c / (4 - 1.2 * (1 - c)):.3f}')

    # 5. Hasan's regime changes along j_g, vertical, deep fluid: the jump into annular flow at alpha = 0.7
    fl = FLUIDS['deep (rho_g=80)']
    for jl in [1.0, 0.2]:
        print(f'\n== Hasan, j_l = {jl}, vertical, deep fluid: regime changes along j_g')
        prev = None
        for jg in np.geomspace(0.05, 30, 4000):
            a, C0, v, reg = hasan_alpha(jg, jl, *fl, 1.0)
            if prev and reg != prev[3]:
                print(f'  j_g {prev[0]:.4f} -> {jg:.4f}: {prev[3]} -> {reg}, alpha {prev[1]:.3f} -> {a:.3f}, '
                      f'C0 {prev[2]:.3f} -> {C0:.3f}, v_inf {prev[4]:.3f} -> {v:.3f}')
            prev = (jg, a, C0, reg, v)

    # 6. ManyWells' points with several void fractions on the grid of 2., with Hasan's void fraction there
    print('\n== ManyWells points with several roots (grid of 2.), and Hasan at the same point')
    for fname, fl in FLUIDS.items():
        for cos in [1.0, 0.5, 0.0]:
            for jg in jgs:
                for jl in jls:
                    r = all_roots(mw_params, jg, jl, *fl, cos)
                    if len(r) > 1:
                        a, C0, v, reg = hasan_alpha(jg, jl, *fl, cos)
                        print(f'  {fname}, cos {cos}: j_g {jg:6.3f} j_l {jl:6.3f}: ManyWells '
                              f'{", ".join(f"{x:.3f}" for x in r)}; Hasan {a:.3f} {reg}')

    # 7. The fitted classifier's regime weights at the feature spec's example points, deep fluid
    from manywells.slip import classify_flow_regime
    print('\n== examples (deep fluid): every ManyWells root with its regime weights, and Hasan')
    fl = FLUIDS['deep (rho_g=80)']
    for jg, jl, cos in [(0.6, 0.5, 1.0), (0.6, 0.5, np.cos(np.pi / 4)), (0.042, 0.010, 0.0), (1.0, 2.0, 1.0)]:
        a_h, _, _, reg = hasan_alpha(jg, jl, *fl, cos)
        for a in all_roots(mw_params, jg, jl, *fl, cos):
            p = np.asarray(classify_flow_regime(jg / a, jl / (1 - a), a, fl[0], fl[1], fl[2], cos)).ravel()
            print(f'  j_g {jg} j_l {jl} cos {cos:.2f}: alpha {a:.3f}, weights annular {p[0]:.2f} slug-churn '
                  f'{p[1]:.2f} bubbly {p[2]:.2f}; Hasan {a_h:.3f} {reg}')


if __name__ == '__main__':
    main()
