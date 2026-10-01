# Discretization

## Purpose

The grid, the implicit Euler discretization of the balances, and the system of residual rows that an implementation solves (paper §3). The roots of this system are the model's answer (`solution.md`).

DISC-1 to DISC-6 are v1.0.0's system, on its uniform grid of a vertical pipe with constant phase mass fluxes. DISC-7 to DISC-11 are `develop`'s: the same system on any survey grid (GEO-3), with phase mass rates that may vary with $(p, T)$. They reduce to DISC-2 to DISC-6 in the `v1.0.0` configuration, as functions of the state, so one implementation serves every configuration (`specs/architecture.md`, Discretization).

## Interface

Each row is a function of the state at one grid point, or at two neighbouring points, and of the case's parameters. It must accept CasADi symbols. The canonical form of each row, below, fixes its units and sign; this is the form of the row vectors. An implementation may use an equivalent form, such as the gas law solved for $\rho_g$, if it is the canonical row times a known nonzero factor; its test adapter divides the factor out. An implementation of the `v1.0.0` configuration must reproduce the canonical rows as functions of the state, not only their roots: substituting one closure into another, for example the constant $\rho_l$ for the state's $\rho_l$, keeps the roots but changes the rows away from them. The stability label (SOL-3) does not depend on the scaling of any row.

## Equations

### DISC-1 · Grid

The pipe of length $L$ is split into $N$ cells of length $\Delta z = L/N$. The grid points are $z_i = i\,\Delta z$, $i = 0, \dots, N$: point 0 is the bottomhole and point $N$ the wellhead. Cell $i$ lies between points $i-1$ and $i$. The paper speaks of "$N + 1$ cells"; it means these $N + 1$ grid points.

### DISC-2 · Gas mass row

For $i = 1, \dots, N$:

$$r = (\alpha\rho_g v_g)_i - (\alpha\rho_g v_g)_{i-1} \quad \text{[kg/(m² s)]}$$

This is BAL-1 with BAL-3, integrated exactly: the gas mass flux is the same at every point.

### DISC-3 · Liquid mass row

For $i = 1, \dots, N$:

$$r = ((1-\alpha)\rho_l v_l)_i - ((1-\alpha)\rho_l v_l)_{i-1} \quad \text{[kg/(m² s)]}$$

### DISC-4 · Momentum row

BAL-4 by implicit Euler. For $i = 1, \dots, N$, with pressures in bar:

$$r = \left(\frac{M_i}{c_\text{bar}} + p_i\right) - \left(\frac{M_{i-1}}{c_\text{bar}} + p_{i-1}\right) + \frac{\Delta z\,(F_i + G_i)}{c_\text{bar}} \quad \text{[bar]}$$

where $M = \alpha\rho_g v_g^2 + (1-\alpha)\rho_l v_l^2$ (Pa) is the momentum flux, and $F_i$ (FRIC-1) and $G_i$ (BAL-6) are evaluated at point $i$.

### DISC-5 · Energy row

BAL-5 by implicit Euler. For $i = 1, \dots, N$:

$$r = T_i - T_{i-1} + \Delta z\, H_i \quad \text{[K]}$$

where $H_i$ is THM-1 evaluated at point $i$, with the ambient temperature $T_{a,i} = T_a(z_i)$ of THM-2.

### DISC-6 · System and row order

The unknowns are the $7(N+1)$ state values of `nomenclature.md`. The rows are assembled point by point, from point 0 to point $N$; at each point in this order:

| Point | Rows, in order | Count |
|---|---|--:|
| $i = 0$ | INF-6, INF-7, THM-3, then the closures | 6 |
| $0 < i < N$ | DISC-2, DISC-3, DISC-4, DISC-5, then the closures | 7 |
| $i = N$ | DISC-2, DISC-3, DISC-4, DISC-5, CHK-1, then the closures | 8 |

The closures, in order, are SLIP-1, PVT-GAS-1 and PVT-MIX-1. That gives $6 + 7(N-1) + 8 = 7(N+1)$ rows. The paper counts $8(N+1)$ rows and unknowns because it keeps $\alpha_l$ and the row of BAL-9.

Canonical forms of the rows that are defined in other files:

| ID | Row | Unit |
|---|---|---|
| INF-6 | $A\alpha_0\rho_{g,0}v_{g,0} - w_g(z_0)$ | kg/s |
| INF-7 | $A(1-\alpha_0)\rho_{l,0}v_{l,0} - w_l(z_0)$ | kg/s |
| THM-3 | $T_0 - T_r$ | K |
| CHK-1 | $w_m(z_N) - w_c$ | kg/s |
| SLIP-1 | $v_g - C_0 v_m - v_\infty$ | m/s |
| PVT-GAS-1 | $p - \rho_g R_s T / c_\text{bar}$ | bar |
| PVT-MIX-1 | $\rho_l - \rho_{l,\text{const}}$ | kg/m³ |

The order is a convention: roots do not depend on it. It matters to code that works with the residual directly, such as the stability label (SOL-3), which drops the CHK-1 row.

### DISC-7 · Gas mass row with mass transfer

On the grid of GEO-3, for $i = 1, \dots, N$:

$$r = (\alpha\rho_g v_g)_i - (\alpha\rho_g v_g)_{i-1} - \frac{w_g(p_i, T_i) - w_g(p_{i-1}, T_{i-1})}{A} \quad \text{[kg/(m² s)]}$$

$w_g(p, T)$ is the gas mass rate of the fluid model at $(p, T)$, for the reservoir liquid rate $w_\text{res}$ of the case and its lift-gas rate (PVT-OIL-13; INF-5 for dead oil). This is BAL-1 with BAL-10, integrated exactly. Without mass transfer $w_g$ is the same at every point and the rate difference is identically zero, so this is DISC-2 as a function of the state. With mass transfer it has the same roots as the local rows $A(\alpha\rho_g v_g)_i - w_g(p_i, T_i)$, because INF-6 fixes point 0 (decided by Bjarne, 2026-09-30, `specs/architecture.md`, decision 4; this text signed off 2026-10-01).

### DISC-8 · Liquid mass row with mass transfer

For $i = 1, \dots, N$:

$$r = ((1-\alpha)\rho_l v_l)_i - ((1-\alpha)\rho_l v_l)_{i-1} - \frac{w_l(p_i, T_i) - w_l(p_{i-1}, T_{i-1})}{A} \quad \text{[kg/(m² s)]}$$

with $w_l(p, T)$ the fluid model's liquid mass rate. Without mass transfer it is DISC-3.

### DISC-9 · Momentum row along the flow path

BAL-11 by implicit Euler on the grid of GEO-3. For $i = 1, \dots, N$, with pressures in bar:

$$r = \left(\frac{M_i}{c_\text{bar}} + p_i\right) - \left(\frac{M_{i-1}}{c_\text{bar}} + p_{i-1}\right) + \frac{\Delta\text{MD}_i\,(F_i + G_i\cos\theta_i)}{c_\text{bar}} \quad \text{[bar]}$$

with $M$ as in DISC-4, $F_i$ the viscous pressure gradient of the friction model (FRIC-1) and $G_i$ (BAL-6) at point $i$. Friction acts along the cell's measured depth, gravity along its vertical depth $\Delta\text{MD}_i\cos\theta_i$. On GEO-1's grid it is DISC-4.

### DISC-10 · Energy row along the flow path

BAL-12 by implicit Euler. For $i = 1, \dots, N$:

$$r = T_i - T_{i-1} - \Delta\text{MD}_i \left(\frac{dT}{d\text{MD}}\right)_i \quad \text{[K]}$$

where $(dT/d\text{MD})_i = -H_i + \Phi_{f,i} - \Phi_{g,i}$ at point $i$: the heat loss of THM-1 to the ambient temperature of THM-4 at $f_i$, and the frictional-heating and gravity terms of THM-6 and THM-7 where the thermal model has them, with the cell's $\cos\theta_i$. With neither term, on GEO-1's grid, it is DISC-5.

### DISC-11 · System and row order on a survey grid

The unknowns and the order of the points are those of DISC-6. At each point, in this order:

| Point | Rows, in order | Count |
|---|---|--:|
| $i = 0$ | INF-6, INF-7, the inflow temperature row (THM-3 or THM-5), then the closures | 6 |
| $0 < i < N$ | DISC-7, DISC-8, DISC-9, DISC-10, then the closures | 7 |
| $i = N$ | DISC-7, DISC-8, DISC-9, DISC-10, CHK-1, then the closures | 8 |

The closures, in order, are SLIP-1, the gas law (PVT-GAS-1 or PVT-GAS-3) and the liquid density (PVT-MIX-1 or PVT-MIX-6). The closures at point $i > 0$ use the inclination of cell $i$, below the point; those at point 0 use cell 1's. The reservoir liquid rate $w_\text{res}$ is the inflow model's at $p_0$ (INF-1, INF-2 or INF-8), a function of the state that enters every point's mass rows; it is not hidden state.

In the `v1.0.0` configuration the rows are those of DISC-6, row for row: DISC-7 to DISC-10 are DISC-2 to DISC-5, the inflow temperature row is THM-3, and the closures are SLIP-1, PVT-GAS-1 and PVT-MIX-1. Canonical forms of the rows defined in other files, besides those of DISC-6:

| ID | Row | Unit |
|---|---|---|
| THM-5 | $T_0 - T_\text{in}$ | K |
| PVT-GAS-3 | $p - \rho_g Z R_s T / c_\text{bar}$ | bar |
| PVT-MIX-6 | $\rho_l - \rho_l(p, T)$ | kg/m³ |

## Options

| Option | IDs | Used by |
|---|---|---|
| Uniform grid on a vertical pipe, constant phase mass fluxes | DISC-1 to DISC-6 | `v1.0.0`, which `develop` implements through DISC-7 to DISC-11 |
| Survey grid; phase mass rates that vary with $(p, T)$ | DISC-7 to DISC-11 | `develop` |

Implicit Euler is the only integrator. Another scheme would be an option here, and would change the order of convergence the verifier's Convergence check expects (`specs/architecture.md`).

## Safeguards

None in the rows. v1.0.0's solver bounds the unknowns; those bounds define the admissible states (SOL-1).

## Sources

Paper §3.1–3.3, equations (16)–(19).

## Test vectors

`vectors/v1_rows.json` holds v1.0.0's rows at perturbed states of two wells with $N = 10$, generated by `specs/tools/make_v1_vectors.py`. `tests/test_spec_vectors.py` checks `develop`'s rows in the `v1.0.0` configuration against them, to a relative $10^{-10}$:

| Well | Inflow | Choke | Profile | Gas lift | At the root |
|---|---|---|---|---|---|
| W1 | Vogel (INF-1) | Simpson (CHK-5) | sigmoid (CHK-8) | 0.8 kg/s | $p_0$ = 205.3 bar, $p_N$ = 48.5 bar, not choked |
| W2 | productivity index (INF-2) | Bernoulli (CHK-6) | linear (CHK-7) | none | $p_0$ = 124.4 bar, $p_N$ = 48.7 bar, choked |

Each state is a v1.0.0 root perturbed by a few per cent, so that every row is well away from zero and has a sign. For each point the file lists the rows in DISC-6's order, each with its ID and value in the canonical units above. The row vectors exercise every equation that v1.0.0 has no separate function for: the balances, friction, heat loss, the ambient profile, the boundary rows and the closures.

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| DISC-1 | §3.1 | `simulator.py` `_differential_equations` (`delta_z`) | rows; verifier: Convergence; spec-only: implemented by GEO-1 through GEO-3 and GEO-4 |
| DISC-2 | (16) | `simulator.py` `_differential_equations` (`g1`) | rows; spec-only: implemented by DISC-7, which is DISC-2 without mass transfer |
| DISC-3 | (17) | `simulator.py` `_differential_equations` (`g2`) | rows; spec-only: implemented by DISC-8, which is DISC-3 without mass transfer |
| DISC-4 | (18) | `simulator.py` `_differential_equations` (`g3`) | rows; spec-only: implemented by DISC-9, which is DISC-4 on GEO-1's grid |
| DISC-5 | (19) | `simulator.py` `_differential_equations` (`g4`) | rows; spec-only: implemented by DISC-10, which is DISC-5 with v1.0.0's thermal option on GEO-1's grid |
| DISC-6 | §3.3 | `simulator.py` `simulate` | rows (the row order at each point); spec-only: implemented by DISC-11, which is DISC-6 in the v1.0.0 configuration |
| DISC-7 | — | — | rows (in the v1.0.0 configuration); property: tests/test_model_properties.py (total mass rate constant with dissolved gas) |
| DISC-8 | — | — | rows (in the v1.0.0 configuration); property: tests/test_model_properties.py |
| DISC-9 | — | — | rows (in the v1.0.0 configuration); property: tests/test_model_properties.py (deviated and L-shaped wells) |
| DISC-10 | — | — | rows (in the v1.0.0 configuration); property: tests/test_thermal.py (the terms) and tests/test_model_properties.py (heat flows outwards) |
| DISC-11 | — | — | rows (in the v1.0.0 configuration); property: tests/test_simulator.py (row order) |
