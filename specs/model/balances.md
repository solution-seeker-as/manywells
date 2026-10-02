# Balances

## Purpose

The continuous model: steady-state conservation of gas mass, liquid mass, mixture momentum and energy along the pipe (paper §2, after Aarsnes et al., 2016). `discretization.md` turns these into residual rows; the closures come from the other component files.

## Interface

The balances are ordinary differential equations in $z$ for the state of `nomenclature.md`; on a survey trajectory (GEO-3), $z$ is the measured depth from the bottomhole, $z = \text{MD}_\text{bh} - \text{MD}$, and $\theta(z)$ the inclination from vertical. They are not implemented as functions of their own: each is implemented through its discretized row (DISC-2 to DISC-5, or DISC-7 to DISC-10). Pressures are in Pa in this file.

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

$H$ (K/m) is the temperature loss to the surroundings (THM-1). Used by `v1.0.0`; BAL-12 adds frictional heating and a gravity term.

### BAL-6 · Gravitational pressure gradient

$$G = \rho_m g$$

the gradient per metre of vertical depth. In a vertical pipe (GEO-1) it acts along the whole pipe; in an inclined one, per metre of the flow path it is $G\cos\theta$ (BAL-11).

### BAL-7 · Mixture density

$$\rho_m = \alpha \rho_g + (1 - \alpha) \rho_l$$

### BAL-8 · Mixture velocity

$$v_m = \alpha v_g + (1 - \alpha) v_l$$

### BAL-9 · Volume fractions

$$\alpha_g + \alpha_l = 1$$

The state carries $\alpha = \alpha_g$ only, and every equation uses $1 - \alpha$ for $\alpha_l$, so this equation has no row.

### BAL-10 · Mass transfer from dissolved gas

$$\Gamma = \frac{1}{A}\,\frac{d\,w_g\big(p(z), T(z)\big)}{dz}$$

where $w_g(p, T)$ is the free-gas mass rate that the fluid model gives at the local pressure and temperature, for the case's reservoir and lift-gas rates (PVT-OIL-13). Gas leaves solution as the pressure falls, so $\Gamma > 0$ up the well. The total mass rate $w_g + w_l$ is the same at every point. Without dissolved gas $w_g$ is constant and this is BAL-3. Used by `develop` with black oil.

### BAL-11 · Momentum along an inclined path

$$\frac{d}{dz}\left(\alpha \rho_g v_g^2 + (1 - \alpha) \rho_l v_l^2 + p\right) = -F - G\cos\theta$$

Friction acts along the flow path, gravity along the vertical. With $\theta = 0$ it is BAL-4.

### BAL-12 · Energy with frictional heating and a gravity term

$$\frac{dT}{dz} = -H + \Phi_f - \Phi_g$$

with the heat loss $H$ (THM-1), the frictional heating $\Phi_f$ (THM-6) and the gravity term $\Phi_g$ (THM-7), each in K/m; a thermal model may leave either term out. The derivation from the total-energy balance, with its assumptions, is in `docs/thermal_energy_modeling.md`. With neither term it is BAL-5.

### BAL-13 · Energy with the Joule–Thomson term

$$\frac{dT}{dz} = -H + \Phi_f - \Phi_g - \Phi_{JT}$$

with the Joule–Thomson term $\Phi_{JT}$ of THM-8 (K/m), which a thermal model may leave out, as it may $\Phi_f$ and $\Phi_g$. It replaces BAL-12's ideal gas, whose enthalpy does not depend on pressure, by the real gas of the fluid model (`docs/thermal_energy_modeling.md`). With $\Phi_{JT} = 0$ it is BAL-12.

## Options

| Option | IDs | Used by |
|---|---|---|
| No mass transfer | BAL-3 | `v1.0.0`; `develop` with dead oil |
| Gas dissolving into oil | BAL-10 | `develop` with black oil |
| Vertical pipe | BAL-4, BAL-6 | `v1.0.0` |
| Inclined flow path | BAL-11, BAL-6 | `develop` |
| Heat loss only | BAL-5 | `v1.0.0` |
| Heat loss, frictional heating, gravity term | BAL-12 | `develop` without the Joule–Thomson term |
| Heat loss, frictional heating, gravity term, Joule–Thomson term | BAL-13 | `develop` |

## Safeguards

None in the continuous model. The admissible states are defined in `solution.md` (SOL-1).

## Sources

- Paper §2, equations (1)–(4), (7) and the source terms in §2.1.
- `docs/thermal_energy_modeling.md` for BAL-12 and BAL-13. Feature specs `specs/features/001-deviated-wells.md`, `008-dissolved-gas.md`, `009-energy-balance.md` and `016-joule-thomson.md`.
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
| BAL-10 | — | — | property: tests/test_model_properties.py (total mass rate constant, free gas increasing up the well); spec-only: continuous form, implemented as DISC-7 and DISC-8 |
| BAL-11 | — | — | property: tests/test_model_properties.py (deviated wells); spec-only: continuous form, implemented as DISC-9 |
| BAL-12 | — | — | property: tests/test_model_properties.py; spec-only: continuous form, implemented as DISC-10 |
| BAL-13 | — | — | property: tests/test_model_properties.py; spec-only: continuous form, implemented as DISC-10 |
