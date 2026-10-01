# Nomenclature

Symbols, units, the state vector, parameters and constants, defined once for every file in `specs/model/`. Symbols follow the paper (Table 1) unless noted.

## Conventions

- **Coordinates.** $z$ is the distance along the pipe from the bottomhole ($z = 0$) to the wellhead ($z = L$). Flow is upwards, in the direction of increasing $z$. The paper's Table 1 calls $z$ "vertical depth"; it increases upwards.
- **Grid.** The pipe is split into $N$ cells. The state lives on the $N + 1$ grid points $z_0, \dots, z_N$, with $z_0$ at the bottomhole and $z_N$ at the wellhead (`discretization.md`, DISC-1). A subscript $i$ means evaluation at $z_i$.
- **Phases.** Subscripts $g$, $l$, $o$, $w$ and $m$ mean gas, liquid, oil, water and mixture. Oil and water form one liquid phase; there is no oil–water slip.
- **Units.** Equations are written in SI units. At interfaces (function arguments and returns, the state vector, test vectors), pressures are in **bar** and temperatures in **kelvin**; $c_\text{bar} = 10^5$ Pa/bar converts. Where an equation is evaluated in bar, it says so.
- **Symbolic.** Every function of the state must accept CasADi symbols as well as floats. It must not branch in Python on a value that may be symbolic; smooth approximations (`smoothing.md`) take the place of `max`, `min` and branches.

## State vector

Each grid point carries seven unknowns, in this order:

| Index | Symbol | Code | Unit | Meaning |
|--:|---|---|---|---|
| 0 | $p$ | `p` | bar | pressure |
| 1 | $v_g$ | `v_g` | m/s | gas velocity |
| 2 | $v_l$ | `v_l` | m/s | liquid velocity |
| 3 | $\alpha$ | `alpha` | – | gas volume fraction $\alpha_g$ (void fraction) |
| 4 | $\rho_g$ | `rho_g` | kg/m³ | gas density |
| 5 | $\rho_l$ | `rho_l` | kg/m³ | liquid density |
| 6 | $T$ | `T` | K | temperature |

The full state is $x = (x_0, \dots, x_N)$, $7(N + 1)$ values, point by point from the bottomhole. The liquid volume fraction is not a state: $\alpha_l = 1 - \alpha$ (BAL-9). The paper counts eight unknowns per point, with $\alpha_l$ as the eighth.

## Derived quantities

| Symbol | Definition | Unit | ID |
|---|---|---|---|
| $A$ | $\pi (D/2)^2$, cross-sectional area | m² | GEO-2 |
| $\rho_m$ | $\alpha \rho_g + (1 - \alpha)\rho_l$, mixture density | kg/m³ | BAL-7 |
| $v_m$ | $\alpha v_g + (1 - \alpha) v_l$, mixture velocity | m/s | BAL-8 |
| $v_{gs}$, $v_{ls}$ | $\alpha v_g$ and $(1 - \alpha) v_l$, superficial velocities | m/s | – |
| $w_g$, $w_l$ | $A\alpha\rho_g v_g$ and $A(1-\alpha)\rho_l v_l$, phase mass rates | kg/s | – |
| $w_m$ | $w_g + w_l$, total mass rate | kg/s | – |
| $x_g$ | $w_g / w_m$, gas mass fraction of the flow | – | – |

## Parameters

Well and fluid parameters of the `v1.0.0` configuration (v1.0.0's `WellProperties` and its inflow and choke models):

| Symbol | Code | Unit | Meaning | Valid range (enforced) |
|---|---|---|---|---|
| $L$ | `L` | m | pipe length (vertical depth) | $> 0$ |
| $D$ | `D` | m | inner pipe diameter | $> 0$ |
| $\rho_l$ | `rho_l` | kg/m³ | liquid density (constant) | – |
| $R_s$ | `R_s` | J/(kg K) | specific gas constant | – |
| $c_{pg}$, $c_{pl}$ | `cp_g`, `cp_l` | J/(kg K) | specific heat capacities | – |
| $f_D$ | `f_D` | – | Darcy friction factor | – |
| $h$ | `h` | W/(m² K) | overall heat transfer coefficient | – |
| $w_{l,\max}$ | `w_l_max` | kg/s | Vogel maximum liquid rate | $\ge 0$ |
| $k_l$ | `k_l` | kg/(s bar) | liquid productivity index | $\ge 0$ |
| $f_g$ | `f_g` | – | gas mass fraction of the reservoir inflow | $(0, 1)$ |
| $K_c$ | `K_c` | m² | choke coefficient | $> 0$ |
| $\sigma(\cdot)$ | `chk_profile` | – | choke profile | one of CHK-7 to CHK-10 |

Here $R_s$ is the specific gas constant, as in the paper. The black-oil correlations also use a solution gas–oil ratio; its symbol is $R_{so}$ (`rs` in the code) to avoid the clash.

Parameters and quantities of `develop`'s options, in addition (the dataclass fields of `FluidModel`, `WellGeometry`, `RoughnessFriction`, `ThermalModel` and `BoundaryConditions`):

| Symbol | Code | Unit | Meaning |
|---|---|---|---|
| $\text{MD}$, $\text{TVD}$ | `md`, `tvd` | m | measured and true vertical depth from the surface (GEO-3) |
| $\Delta\text{MD}_i$, $\cos\theta_i$ | `delta_md`, `cos_incl` | m, – | length and inclination from vertical of cell $i$ |
| $f_i$ | `tvd_frac` | – | TVD fraction of grid point $i$ (1 at the bottomhole) |
| $\rho_o$, $\rho_{g,\text{sc}}$, $\rho_w$ | `rho_o`, `rho_g`, `rho_w` | kg/m³ | oil, gas and water densities at standard conditions |
| $R_{go}$ | `gor` | Sm³/Sm³ | gas–oil ratio at standard conditions |
| $\alpha_{w,l}$ | `wlr` | – | water–liquid ratio (water cut) at standard conditions, in $[0, 1)$ |
| $c_{po}$, $c_{pw}$ | `cp_o`, `cp_w` | J/(kg K) | oil and water heat capacities |
| $\gamma_g$, $M_g$ | `sg_gas`, `M_g` | –, kg/kmol | gas specific gravity (air = 1) and molecular weight |
| $Z$ | `z_factor` | – | gas compressibility factor |
| $R_{so}$, $B_o$ | `rs`, `bo` | Sm³/Sm³, – | solution gas–oil ratio and oil formation volume factor |
| $p_\text{sep}$, $T_\text{sep}$, $p_b$ | `p_sep`, `T_sep`, `p_bubble` | bar, K, bar | separator conditions and bubble-point pressure |
| $\mu_g$, $\mu_o$, $\mu_w$, $\mu_l$, $\mu_m$ | `gas_viscosity`, …, `mixture_viscosity` | Pa s | viscosities |
| $\varepsilon$, Re | `roughness`, `Re` | m, – | pipe wall roughness and Reynolds number |
| $w_\text{res}$ | `w_res` | kg/s | liquid mass rate from the reservoir (the inflow model's) |
| $w_{g,\text{res}}$, $w_d$ | `w_g_res` | kg/s | reservoir gas rate, gas dissolved in the oil |
| $T_{lg}$ | `T_lg` | K | lift-gas temperature at injection; `None` means $T_r$ |
| $\Phi_f$, $\Phi_g$ | – | K/m | frictional heating and gravity terms of the temperature gradient |

Boundary conditions and controls (v1.0.0's `BoundaryConditions`):

| Symbol | Code | Unit | Meaning | Valid range (enforced) |
|---|---|---|---|---|
| $p_r$ | `p_r` | bar | reservoir pressure | $> 0$ |
| $p_s$ | `p_s` | bar | pressure downstream of the choke (separator) | $> 0$ |
| $T_r$ | `T_r` | K | reservoir temperature | – |
| $T_s$ | `T_s` | K | ambient temperature at the surface, $z = L$ | – |
| $u$ | `u` | – | choke position | $[0, 1]$ |
| $w_{lg}$ | `w_lg` | kg/s | lift-gas rate, injected at $z = 0$ | $\ge 0$ |

Input validation belongs in the dataclasses' `__post_init__`, not in the solver.

## Constants

| Symbol | Code | Value | Meaning |
|---|---|---|---|
| $g$ | `STD_GRAVITY` | 9.80665 m/s² | standard gravity |
| $c_\text{bar}$ | `CF_PRES` (v1.0.0), `CF_BAR` (`develop`) | $10^5$ Pa/bar | bar to Pa |
| $p_\text{ref}$, $T_\text{ref}$ | `P_REF`, `T_REF` | 101 325 Pa, 288.15 K | standard conditions, ISO 13443 (PVT-GAS-2) |
| $\rho_{w,\text{ref}}$ | `WATER.rho` | 999.1 kg/m³ | water density, the reference for specific gravity (PVT-OIL-2) |
| $\gamma$ | `gamma` | 1.307 | heat capacity ratio of the gas in the choke (CHK-4) |
| $\epsilon$ | `eps` | $10^{-6}$ | smoothing constant (SMO-1, SMO-2); its unit is the square of its arguments' unit |
| $R_u$ | `R_UNIVERSAL` | 8314.46 J/(kmol K) | universal gas constant (PVT-GAS-6) |
| $M_\text{air}$ | `M_AIR` | 28.97 kg/kmol | molecular weight of air (PVT-GAS-6) |
| – | `CF_PSI`, `CF_RS`, `CF_CP`, `CF_UP` | 6894.76 Pa/psi, 0.178108 (Sm³/Sm³)/(scf/STB), $10^{-3}$ Pa s/cP, $10^{-7}$ Pa s/µP | field-unit conversions of the correlations in `pvt/` |
