# Thermal

## Purpose

The heat exchanged with the surroundings, the ambient temperature profile, and the temperature of the fluid entering the well. With BAL-5 they give the temperature along the well.

## Interface

| Function | Inputs | Output | Symbolic |
|---|---|---|---|
| heat loss | $T$, $\alpha$, $\rho_g$, $v_g$, $\rho_l$, $v_l$ (state), $T_a$, $h$, $D$, $c_{pg}$, $c_{pl}$ | $H$ (K/m) | yes |
| ambient temperature | $z$ (m), $L$, $T_r$, $T_s$ | $T_a$ (K) | no |
| inflow temperature row | $T_0$ (state), $T_r$ | row (K) | yes |

## Equations

### THM-1 · Heat loss

After Zhang et al. (2006):

$$H = \frac{4h\,(T - T_a)}{D\,\big(c_{pg}\,\alpha\rho_g v_g + c_{pl}\,(1-\alpha)\rho_l v_l\big)}$$

$h$ is the overall heat transfer coefficient, and the heat capacities $c_{pg}$ and $c_{pl}$ are constants. The denominator is the heat-capacity flux of the flow.

### THM-2 · Ambient temperature

The ambient temperature falls linearly from the reservoir temperature at the bottomhole to the surface temperature at the wellhead:

$$T_a(z) = T_r - \frac{z}{L}\,(T_r - T_s), \qquad T_{a,i} = T_r - \frac{i}{N}\,(T_r - T_s).$$

Used by `v1.0.0`.

### THM-3 · Inflow temperature

The fluid entering the well is at the reservoir temperature:

$$T(z = 0) = T_r.$$

Its row in DISC-6 is $T_0 - T_r$ (K). The lift gas, injected at $z = 0$, is taken to be at $T_r$ too, so it does not change the inflow temperature. Used by `v1.0.0`.

## Options

| Option | IDs | Used by |
|---|---|---|
| Heat loss only, linear ambient profile in $z$, inflow at $T_r$ | THM-1 to THM-3 | `v1.0.0` |
| Frictional heating and a gravity term in the energy balance; ambient profile in true vertical depth; lift-gas mixing temperature at the bottomhole | Step 7 | `develop` |

The derivations of `develop`'s terms are in `docs/thermal_energy_modeling.md`.

## Safeguards

None. The denominator of THM-1 is positive at every admissible state (SOL-1).

## Sources

- Paper (6), (15) and §2.1.
- Zhang, Wang, Sarica and Brill (2006), "Unified model of heat transfer in gas–liquid pipe flow", *SPE Production & Operations* 21, 114–122.

## Test vectors

The row vectors (`discretization.md`) exercise THM-1 and THM-2 through the energy row DISC-5, and THM-3 at point 0.

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| THM-1 | (6) | `simulator.py` `_differential_equations` (`dT`) | rows |
| THM-2 | §2.1 | `simulator.py` `_differential_equations` (`T_a`) | rows |
| THM-3 | (15) | `simulator.py` `_left_boundary_eqs` (`g3`) | rows |
