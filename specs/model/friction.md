# Friction

## Purpose

The viscous pressure gradient in the momentum balance, and the Darcy friction factor that scales it.

## Interface

| Function | Inputs | Output | Symbolic |
|---|---|---|---|
| viscous pressure gradient | $f_D$, $D$ (m), $\rho_m$ (kg/m³), $v_m$ (m/s) | $F$ (Pa/m) | yes |
| friction factor | per option; none for FRIC-2 | $f_D$ (–) | yes, for options that depend on the state |

## Equations

### FRIC-1 · Viscous pressure gradient

Darcy–Weisbach for the mixture:

$$F = \frac{f_D}{2D}\,\rho_m\, v_m \lvert v_m\rvert$$

v1.0.0 implements $\rho_m v_m^2$, which is the same for $v_m \ge 0$; every admissible state has $v_m > 0$ (SOL-1).

### FRIC-2 · Fixed friction factor

$f_D$ is a constant parameter of the well, the same in every cell. Used by `v1.0.0`.

## Options

| Option | IDs | Used by |
|---|---|---|
| Fixed $f_D$ | FRIC-2 | `v1.0.0`; `develop` when `f_D` is set |
| $f_D$ from the Reynolds number and pipe roughness: Chen (1979) or Haaland (1983), with a smooth laminar–turbulent blend; mixture viscosity | Step 7 | `develop` default |

## Safeguards

None for FRIC-2.

## Sources

- Paper (5) and §4.1 (the sampled range of $f_D$).
- Moody (1944), "Friction factors for pipe flow", *Transactions of the ASME* 66, 671–678.

## Test vectors

The row vectors (`discretization.md`) exercise FRIC-1 and FRIC-2 through the momentum row DISC-4.

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| FRIC-1 | (5) | `simulator.py` `_differential_equations` (`dp_f`) | rows |
| FRIC-2 | §2.1, §4.1 | `simulator.py` `WellProperties.f_D` | rows |
