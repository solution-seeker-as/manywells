# Thermal

## Purpose

The heat exchanged with the surroundings, the ambient temperature profile, the temperature of the fluid entering the well, and the other terms of the temperature gradient. With BAL-5, BAL-12 or BAL-13 they give the temperature along the well.

## Interface

| Function | Inputs | Output | Symbolic |
|---|---|---|---|
| heat loss | $T$, $\alpha$, $\rho_g$, $v_g$, $\rho_l$, $v_l$ (state), $T_a$, $h$, $D$, $c_{pg}$, $c_{pl}$ | $H$ (K/m) | yes |
| ambient temperature | $z$ (m), $L$, $T_r$, $T_s$; or $f_i$ (GEO-3), $T_r$, $T_s$ | $T_a$ (K) | no; yes in $T_r$, $T_s$ and $f_i$ in `develop` |
| inflow temperature | $w_\text{res}$, $w_{lg}$ (kg/s), $T_r$, $T_{lg}$ (K), the fluid's $c_{pg}$, $c_{pl}$, $f_g$ | $T_\text{in}$ (K) | yes |
| inflow temperature row | $T_0$ (state), $T_\text{in}$ | row (K) | yes |
| frictional heating, gravity term, Joule–Thomson term | the state, $F$ (Pa/m), $\cos\theta$, $c_{pg}$, $c_{pl}$; the gas's $J$ (PVT-GAS-10) | $\Phi_f$, $\Phi_g$, $\Phi_{JT}$ (K/m) | yes |

In `develop` the thermal model is a `ThermalModel` with $h$ and one switch per option: `frictional_heating` (THM-6), `gravity_term` (THM-7), `joule_thomson` (THM-8) and `lift_gas_mixing` (THM-5). Its `temperature_gradient` returns $dT/d\text{MD} = -H + \Phi_f - \Phi_g - \Phi_{JT}$ with the terms that are on (BAL-13, DISC-10); it also takes the cell's pressure gradient, which no option uses (`specs/architecture.md`). The `v1.0.0` configuration has every switch off.

## Equations

### THM-1 · Heat loss

After Zhang et al. (2006):

$$H = \frac{4h\thinspace(T - T_a)}{D\thinspace\big(c_{pg}\alpha\rho_g v_g + c_{pl}\thinspace(1-\alpha)\rho_l v_l\big)}$$

$h$ is the overall heat transfer coefficient, and the heat capacities $c_{pg}$ and $c_{pl}$ are constants. The denominator is the heat-capacity flux of the flow.

### THM-2 · Ambient temperature

The ambient temperature falls linearly from the reservoir temperature at the bottomhole to the surface temperature at the wellhead:

$$T_a(z) = T_r - \frac{z}{L}\thinspace(T_r - T_s), \qquad T_{a,i} = T_r - \frac{i}{N}\thinspace(T_r - T_s).$$

Used by `v1.0.0`.

### THM-3 · Inflow temperature

The fluid entering the well is at the reservoir temperature:

$$T(z = 0) = T_r.$$

Its row in DISC-6 is $T_0 - T_r$ (K). The lift gas, injected at $z = 0$, is taken to be at $T_r$ too, so it does not change the inflow temperature. Used by `v1.0.0`.

### THM-4 · Ambient temperature in true vertical depth

$$T_{a,i} = T_s + (T_r - T_s) f_i$$

with $f_i$ the TVD fraction of grid point $i$ (GEO-3): the ambient temperature falls linearly in true vertical depth, from $T_r$ at the bottomhole to $T_s$ at the surface. On GEO-1's grid $f_i = 1 - i/N$, and it is THM-2.

### THM-5 · Lift-gas mixing temperature

The reservoir fluid at $T_r$ and the lift gas at $T_{lg}$ mix at the bottomhole, weighted by their heat-capacity rates:

$$T_\text{in} = T_r + \frac{H_{lg}\thinspace(T_{lg} - T_r)}{H_\text{res} + H_{lg}}, \qquad H_\text{res} = c_{pl} w_\text{res} + c_{pg} w_{g,\text{res}}, \qquad H_{lg} = c_{pg} w_{lg},$$

with $w_{g,\text{res}}$ from INF-4. Its row is $T_0 - T_\text{in}$ (K). When $T_{lg}$ is not given it is $T_r$, and the row is exactly THM-3's; so it is without lift gas. Mixing at constant heat capacities, with no heat of solution; the lift gas is the same gas as the produced gas.

### THM-6 · Frictional heating

$$\Phi_f = \frac{(1-\alpha) v_l F}{C}, \qquad C = c_{pg}\alpha\rho_g v_g + c_{pl}\thinspace(1-\alpha)\rho_l v_l$$

in K/m, with $F$ the viscous pressure gradient (FRIC-1) and $C$ the heat-capacity flux of THM-1. The work against friction heats the liquid; an ideal gas's enthalpy does not change with it, so the gas phase gets none (`docs/thermal_energy_modeling.md`). For pure liquid, $\Phi_f = F/(\rho_l c_{pl})$.

### THM-7 · Gravity term

$$\Phi_g = \frac{g\cos\theta\thinspace\big(\alpha\rho_g v_g + (1-\alpha)\rho_l v_l - (1-\alpha) v_l\rho_m\big)}{C}$$

in K/m, with $\theta$ the cell's inclination and $C$ as in THM-6. Lifting the flow against gravity cools it: for pure gas $\Phi_g = g\cos\theta/c_{pg}$, the adiabatic lapse rate; for pure liquid it vanishes, because the liquid's gravitational work is in the pressure term.

### THM-8 · Joule–Thomson term

$$\Phi_{JT} = \frac{\alpha v_g J\thinspace (F + \rho_m g\cos\theta)}{C}$$

in K/m, with $J$ the gas's Joule–Thomson factor (PVT-GAS-10) at the point, $F$ the viscous pressure gradient (FRIC-1), $\theta$ the cell's inclination and $C$ the heat-capacity flux of THM-1. A real gas's enthalpy changes with pressure, $dh_g = c_{pg}\thinspace dT - c_{pg}\mu_{JT}\thinspace dp$, and the gas's share of the enthalpy flux, $\alpha\rho_g v_g c_{pg}\mu_{JT}\thinspace dp/dz = \alpha v_g J\thinspace dp/dz$, enters the energy balance as THM-6 enters for the liquid, with $dp/dz = -(F + \rho_m g\cos\theta)$, neglecting acceleration as THM-6 and THM-7 do (`docs/thermal_energy_modeling.md`). The pressure gradient is positive at every admissible state, so $\Phi_{JT}$ has the sign of $J$: it cools where $Z$ rises with $T$, everywhere in the sampled range below about 300 bar, and heats above the inversion pressure. It is zero for an ideal gas and for pure liquid; for pure gas, $\Phi_{JT} = \mu_{JT}(F + \rho_g g\cos\theta)$, the gas's Joule–Thomson cooling along its pressure drop. Near a gas well's choked wellhead the term can give a cell's energy row two roots in $T_i$; only the one where the row rises in $T_i$ is a root of the model (SOL-9).

## Options

| Option | IDs | Used by |
|---|---|---|
| Heat loss only, linear ambient profile in $z$, inflow at $T_r$ | THM-1 to THM-3 | `v1.0.0` |
| Ambient profile in true vertical depth | THM-4 | `develop`, every thermal option; it is THM-2 on a vertical grid |
| Frictional heating | THM-6 | `develop` default (`frictional_heating`) |
| Gravity term | THM-7 | `develop` default (`gravity_term`) |
| Joule–Thomson term | THM-8 | `develop` default (`joule_thomson`); zero with an ideal gas |
| Lift-gas mixing temperature at the bottomhole | THM-5 | `develop` default (`lift_gas_mixing`) |

The derivations of `develop`'s terms are in `docs/thermal_energy_modeling.md`.

## Safeguards

None. The denominator of THM-1, THM-6, THM-7 and THM-8 is positive at every admissible state (SOL-1); THM-8 has the safeguards of PVT-GAS-10. The denominator of THM-5 is positive wherever the reservoir delivers fluid or there is lift gas; at $p_0 = p_r$ without lift gas it is zero, which no root reaches.

## Sources

- Paper (6), (15) and §2.1.
- Zhang, Wang, Sarica and Brill (2006), "Unified model of heat transfer in gas–liquid pipe flow", *SPE Production & Operations* 21, 114–122.
- `docs/thermal_energy_modeling.md` for THM-6, THM-7 and THM-8. Feature specs `specs/features/009-energy-balance.md`, `010-lift-gas-temperature.md` and `016-joule-thomson.md`.
- Hasan and Kabir (2012), "Wellbore heat-transfer modeling and applications", *Journal of Petroleum Science and Engineering* 86–87, 127–136, Eq. (7), and Hasan and Kabir (2018), *Fluid flow and heat transfer in wellbores*, 2nd ed., §6.4.2, for THM-8.

## Test vectors

The row vectors (`discretization.md`) exercise THM-1 and THM-2 through the energy row DISC-5, and THM-3 at point 0. The vectors below pin `develop`'s options (`specs/model/README.md`, Test vectors); `tests/test_thermal.py` checks their limiting cases.

<!-- vectors:begin develop -->
Generated by `specs/tools/make_develop_vectors.py` from develop (casadi 3.8.1): they pin develop's options. Do not edit by hand.

### THM-4

| tvd_frac | T_r | T_s | → T_a |
|---|---|---|---|
| 1.0 | 370.0 | 277.15 | 370.0 |
| 0.0 | 370.0 | 277.15 | 277.15 |
| 0.37 | 360.0 | 280.0 | 309.6 |
| 0.9 | 400.0 | 270.0 | 387.0 |

### THM-5

| w_res | w_lg | T_r | T_lg | f_g | cp_g | cp_l | → T_in |
|---|---|---|---|---|---|---|---|
| 10.0 | 1.0 | 373.15 | 300.0 | 0.1 | 2225.0 | 3000.0 | 368.45917060283404 |
| 10.0 | 0.0 | 373.15 | 300.0 | 0.1 | 2225.0 | 3000.0 | 373.15 |
| 2.0 | 3.0 | 380.0 | 290.0 | 0.3 | 2225.0 | 2500.0 | 335.7691296344991 |
| 30.0 | 0.5 | 350.0 | 360.0 | 0.05 | 2225.0 | 4000.0 | 350.08926733216845 |

### THM-6

| alpha | rho_g | v_g | rho_l | v_l | F | cp_g | cp_l | → Phi_f |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 50.0 | 10.0 | 850.0 | 2.0 | 500.0 | 2225.0 | 4180.0 | 0.00014072614691809738 |
| 0.4 | 80.0 | 6.0 | 800.0 | 2.5 | 900.0 | 2225.0 | 3000.0 | 0.00033522050059594757 |
| 0.9 | 30.0 | 20.0 | 820.0 | 4.0 | 2000.0 | 2225.0 | 2500.0 | 0.0003957457333663121 |

### THM-7

| alpha | rho_g | v_g | rho_l | v_l | cos_incl | cp_g | cp_l | → Phi_g |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 50.0 | 10.0 | 850.0 | 2.0 | 1.0 | 2225.0 | 4180.0 | 0.004407483146067415 |
| 0.4 | 80.0 | 6.0 | 800.0 | 2.5 | 1.0 | 2225.0 | 3000.0 | 0.0015195047675804528 |
| 0.6 | 60.0 | 8.0 | 820.0 | 3.0 | 0.5 | 2225.0 | 2500.0 | 0.0013207098297213621 |
| 0.5 | 60.0 | 8.0 | 820.0 | 3.0 | 0.0 | 2225.0 | 2500.0 | 0.0 |

### THM-8

| alpha | rho_g | v_g | rho_l | v_l | F | cos_incl | T | sg_gas | cp_g | cp_l | → Phi_JT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 60.0 | 10.0 | 850.0 | 2.0 | 300.0 | 1.0 | 350.0 | 0.65 | 2225.0 | 4180.0 | 0.002819369193135331 |
| 0.5 | 80.0 | 6.0 | 800.0 | 2.5 | 900.0 | 0.7 | 330.0 | 0.7 | 2225.0 | 3000.0 | 0.002298249871027588 |
| 0.9 | 300.0 | 3.0 | 820.0 | 1.0 | 1500.0 | 1.0 | 300.0 | 0.65 | 2225.0 | 2500.0 | -5.637110958810489e-05 |
| 0.0 | 50.0 | 10.0 | 850.0 | 2.0 | 500.0 | 1.0 | 350.0 | 0.65 | 2225.0 | 4180.0 | 0.0 |
<!-- vectors:end develop -->

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| THM-1 | (6) | `simulator.py` `_differential_equations` (`dT`) | rows |
| THM-2 | §2.1 | `simulator.py` `_differential_equations` (`T_a`) | rows; spec-only: implemented by THM-4, which is THM-2 on GEO-1's grid |
| THM-3 | (15) | `simulator.py` `_left_boundary_eqs` (`g3`) | rows |
| THM-4 | — | — | vectors; rows (in the v1.0.0 configuration, as THM-2) |
| THM-5 | — | — | vectors; property: tests/test_thermal.py |
| THM-6 | — | — | vectors; property: tests/test_thermal.py |
| THM-7 | — | — | vectors; property: tests/test_thermal.py |
| THM-8 | — | — | vectors; property: tests/test_thermal.py |
