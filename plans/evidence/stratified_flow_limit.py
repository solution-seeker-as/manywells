"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Where does the Taitel-Dukler (1976) model predict stratified flow in upward-inclined tubing?

R2 of the sampler redesign (wells and production), 2026-10-30. Evidence for the inclination
limit of specs/model/validity.md's decision 5 and D-10 (plans/sampler-redesign/decisions.md), and for
the scan in plans/sampler-redesign/sources-wells-production.md, Section 3. No data is read; the
inputs are ranges stated below.

Model: Taitel and Dukler (1976), AIChE J. 22(1), 47-55, as restated by Gomez et al. (1999,
SPE 56520, Eqs. 1-2) for the stratified to non-stratified transition of Barnea's (1987) unified
model, and by Hasan and Kabir (2018, Section 4.3.2) for the equilibrium liquid level:

- the stratified momentum balance in dimensionless form, with Blasius friction factors
  (C = 0.046, n = 0.2 turbulent; C = 16, n = 1 laminar) and the interfacial friction factor
  equal to the gas wall factor, solved for the liquid level h = h_L/D;
- the Kelvin-Helmholtz criterion F^2 [u_G^2 dA_L/dh / ((1 - h)^2 A_G)] >= 1 for the end of
  stratified flow, F = sqrt(rho_G/(rho_L - rho_G)) v_SG / sqrt(D g cos(beta)).

beta is the inclination above horizontal for upward flow (Taitel-Dukler's alpha = -beta).
Where the balance has several roots (upward flow), both the lowest and the highest liquid
level are tested; a point counts as stratified if either root is stable ("any") and,
separately, only with the lowest root ("low").

Run: (ulimit -v 3000000; OMP_NUM_THREADS=1 timeout 600 .venv/bin/python plans/evidence/stratified_flow_limit.py)
"""

import numpy as np

G = 9.81


def geometry(h):
    x = 2.0 * h - 1.0
    s = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    ac = np.arccos(x)
    AL = 0.25 * (np.pi - ac + x * s)
    AG = 0.25 * (ac - x * s)
    SL = np.pi - ac
    SG = ac
    Si = s
    return AL, AG, SL, SG, Si


def fric(re):
    """Return (C, n) per Reynolds number: laminar below 2000, Blasius above."""
    lam = re < 2000.0
    return np.where(lam, 16.0, 0.046), np.where(lam, 1.0, 0.2)


def residual(h, X2, Y, nL, nG):
    AL, AG, SL, SG, Si = geometry(h)
    A = np.pi / 4.0
    uL, uG = A / AL, A / AG
    DL = 4.0 * AL / SL
    DG = 4.0 * AG / (SG + Si)
    lhs = X2 * (uL * DL) ** (-nL) * uL**2 * SL / AL
    rhs = (uG * DG) ** (-nG) * uG**2 * (SG / AG + Si / AL + Si / AG)
    return lhs - rhs - 4.0 * Y


def kh(h, F2):
    AL, AG, SL, SG, Si = geometry(h)
    A = np.pi / 4.0
    uG = A / AG
    dAL = Si  # dA_L/dh in units of D^2, = sqrt(1 - (2h - 1)^2)
    return F2 * uG**2 * dAL / ((1.0 - h) ** 2 * AG)


HGRID = np.linspace(1e-4, 1.0 - 1e-4, 4000)


def stratified(vsl, vsg, beta_deg, D, rhoL, rhoG, muL, muG):
    """Return (any_root_stable, lowest_root_stable)."""
    b = np.radians(beta_deg)
    reL = rhoL * vsl * D / muL
    reG = rhoG * vsg * D / muG
    CL, nL = fric(np.array(reL))
    CG, nG = fric(np.array(reG))
    dpL = 4.0 * CL / D * reL ** (-nL) * rhoL * vsl**2 / 2.0
    dpG = 4.0 * CG / D * reG ** (-nG) * rhoG * vsg**2 / 2.0
    X2 = dpL / dpG
    Y = (rhoL - rhoG) * G * np.sin(-b) / dpG  # alpha = -beta for upward flow
    r = residual(HGRID, X2, Y, nL, nG)
    idx = np.nonzero(np.sign(r[:-1]) != np.sign(r[1:]))[0]
    if idx.size == 0:
        return False, False
    roots = []
    for i in idx:
        h0, h1, r0, r1 = HGRID[i], HGRID[i + 1], r[i], r[i + 1]
        roots.append(h0 - r0 * (h1 - h0) / (r1 - r0))
    F2 = rhoG / (rhoL - rhoG) * vsg**2 / (D * G * np.cos(b))
    stable = [kh(h, F2) < 1.0 for h in roots]
    return any(stable), stable[0]


def main():
    # In-situ cases: pressures of about 10 to 250 bar; oil and water.
    fluids = {
        'oil, 10 bar (rhoG 8)': (800.0, 8.0, 2e-3, 1.2e-5),
        'oil, 60 bar (rhoG 50)': (780.0, 50.0, 1.5e-3, 1.5e-5),
        'oil, 200 bar (rhoG 170)': (720.0, 170.0, 1.0e-3, 2.5e-5),
        'water, 60 bar (rhoG 50)': (1030.0, 50.0, 0.6e-3, 1.5e-5),
    }
    diams = {'2 7/8 in (ID 62 mm)': 0.0620, '4 1/2 in (ID 100 mm)': 0.1005, '7 in (ID 157 mm)': 0.1570}
    vsl_grid = np.logspace(np.log10(0.003), np.log10(5.0), 40)
    vsg_grid = np.logspace(np.log10(0.03), np.log10(30.0), 40)
    betas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 45.0]
    print('For each case and angle above horizontal: the largest v_SL (m/s) at which any v_SG in')
    print("0.03-30 m/s gives stratified flow, with any root ('any') and the lowest root ('low');")
    print("and the share of the (v_SL >= 0.05 m/s) grid that is stratified ('any').")
    for fname, (rhoL, rhoG, muL, muG) in fluids.items():
        for dname, D in diams.items():
            print(f'\n{fname}, {dname}')
            print('  beta  vSLmax_any  vSLmax_low  share(vSL>=0.05)')
            for beta in betas:
                vmax_any, vmax_low, n_strat, n_tot = 0.0, 0.0, 0, 0
                for vsl in vsl_grid:
                    for vsg in vsg_grid:
                        a, lo = stratified(vsl, vsg, beta, D, rhoL, rhoG, muL, muG)
                        if a:
                            vmax_any = max(vmax_any, vsl)
                        if lo:
                            vmax_low = max(vmax_low, vsl)
                        if vsl >= 0.05:
                            n_tot += 1
                            n_strat += int(a)
                share = n_strat / n_tot
                print(f'  {beta:4.1f}  {vmax_any:10.3f}  {vmax_low:10.3f}  {share:6.3f}')


if __name__ == '__main__':
    main()
