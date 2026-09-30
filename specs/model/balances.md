# Balances

## Purpose

The continuous model: steady-state conservation of gas mass, liquid mass, mixture momentum and energy along the pipe (paper §2, after Aarsnes et al., 2016). `discretization.md` turns these into residual rows; the closures come from the other component files.

## Interface

The balances are ordinary differential equations in $z$ for the state of `nomenclature.md`. They are not implemented as functions of their own: each is implemented through its discretized row (DISC-2 to DISC-5). Pressures are in Pa in this file.

## Equations

### BAL-1 · Gas mass

$$\frac{d}{dz}\left(\alpha \rho_g v_g\right) = \Gamma$$

$\Gamma$ (kg/(m³ s)) is the mass transfer from liquid to gas.

### BAL-2 · Liquid mass

$$\frac{d}{dz}\left((1 - \alpha) \rho_l v_l\right) = -\Gamma$$

### BAL-3 · No mass transfer

$$\Gamma = 0$$

With BAL-1 and BAL-2, both phase mass rates are constant along the pipe. Used by `v1.0.0`.

### BAL-4 · Momentum

$$\frac{d}{dz}\left(\alpha \rho_g v_g^2 + (1 - \alpha) \rho_l v_l^2 + p\right) = -F - G$$

$F$ is the viscous pressure gradient (FRIC-1) and $G$ the gravitational pressure gradient (BAL-6), both in Pa/m.

### BAL-5 · Energy

$$\frac{dT}{dz} = -H$$

$H$ (K/m) is the temperature loss to the surroundings (THM-1). Used by `v1.0.0`; `develop` adds frictional heating and a gravity term (Step 7).

### BAL-6 · Gravitational pressure gradient

$$G = \rho_m g$$

For the vertical pipe of GEO-1.

### BAL-7 · Mixture density

$$\rho_m = \alpha \rho_g + (1 - \alpha) \rho_l$$

### BAL-8 · Mixture velocity

$$v_m = \alpha v_g + (1 - \alpha) v_l$$

### BAL-9 · Volume fractions

$$\alpha_g + \alpha_l = 1$$

The state carries $\alpha = \alpha_g$ only, and every equation uses $1 - \alpha$ for $\alpha_l$, so this equation has no row.

## Options

| Option | IDs | Used by |
|---|---|---|
| No mass transfer | BAL-3 | `v1.0.0` |
| Gas dissolving into oil | Step 7 | `develop` |

The energy balance of `develop` (BAL-5 with extra terms) is added in Step 7.

## Safeguards

None in the continuous model. The admissible states are defined in `solution.md` (SOL-1).

## Sources

- Paper §2, equations (1)–(4), (7) and the source terms in §2.1.
- Aarsnes, Flåtten and Aamo (2016), "Review of two-phase flow models for control and estimation", *Annual Reviews in Control* 42, 50–62.

## Test vectors

The balances have no functions of their own. Their discretized rows are checked by the row vectors (`discretization.md`), and the verifier's Convergence check tests that roots converge at first order as the grid is refined, which ties the rows to the continuous equations.

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| BAL-1 | (1) | through DISC-2 | rows; verifier: Convergence; spec-only: continuous form, implemented as DISC-2 |
| BAL-2 | (2) | through DISC-3 | rows; verifier: Convergence; spec-only: continuous form, implemented as DISC-3 |
| BAL-3 | §2.1 | through DISC-2, DISC-3 | rows; verifier: Invariants (constant phase mass rates); spec-only: implemented by the constant-flux rows DISC-2 and DISC-3 |
| BAL-4 | (3) | through DISC-4 | rows; verifier: Convergence; spec-only: continuous form, implemented as DISC-4 |
| BAL-5 | (4) | through DISC-5 | rows; verifier: Convergence; spec-only: continuous form, implemented as DISC-5 |
| BAL-6 | §2.1 | `simulator.py` `_differential_equations` (`dp_g`) | rows |
| BAL-7 | §2.1 | `simulator.py` `_differential_equations`, `_right_boundary_eqs` (`rho_m`) | rows |
| BAL-8 | (8) | `simulator.py` `_closure_relations`, `_differential_equations` (`v_m`) | rows |
| BAL-9 | (7) | eliminated: the state carries `alpha` only | spec-only: eliminated by substitution, so there is no row to check |
