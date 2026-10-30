# Liquid mixture

## Purpose

The liquid phase is oil and water mixed, with no slip between them (paper §4). This file defines the liquid's density in the discretized system, the mixing rules that give the liquid's density and heat capacity from oil and water, and the gas–liquid surface tension that the slip model uses.

## Interface

| Function | Inputs | Output | Symbolic |
|---|---|---|---|
| liquid density row | $\rho_l$ (state), $\rho_{l,\text{const}}$ | row (kg/m³) | yes |
| liquid mixing | $\rho_o$, $c_{po}$, $\rho_w$, $c_{pw}$, $x_o$ | $\rho_l$ (kg/m³), $c_{pl}$ (J/(kg K)) | no |
| surface tension | $\rho_l$ (state), $T$ (K) | $\sigma$ (J/m²) | yes |

$x_o = f_o/(f_o + f_w)$ is the oil mass fraction of the liquid, from the mass fractions $f_o$ and $f_w$ of the reservoir inflow. In `v1.0.0` the mixing is applied by the sampler, before the simulator is built: the simulator takes $\rho_l$ and $c_{pl}$ as parameters.

In `develop` the fluid model (`FluidModel`) takes the phase densities at standard conditions $\rho_o$ and $\rho_{g,\text{sc}}$ (from its PVT model, `pvt`) and $\rho_w$, the gas–oil ratio $R_{go}$ and the water–liquid ratio $\alpha_{w,l}$ (both at standard conditions, `gor` and `wlr`), and the heat capacities $c_{pg}$, $c_{po}$ and $c_{pw}$; PVT-MIX-10 derives the rest. Its `surface_tension(p, T, rho_l)` takes the point's liquid density, which PVT-MIX-5 needs, and its `surface_tension_model` chooses PVT-MIX-5 (`'liquid'`) or PVT-MIX-7 (`'oil'`).

## Equations

### PVT-MIX-1 · Constant liquid density

$$\rho_l = \rho_{l,\text{const}}$$

The mixed liquid is incompressible. Its row in DISC-6 is $\rho_l - \rho_{l,\text{const}}$ (kg/m³). Used by `v1.0.0`.

### PVT-MIX-2 · Water volume fraction of the liquid

$$\alpha_{w,l} = \frac{(1 - x_o)/\rho_w}{(1 - x_o)/\rho_w + x_o/\rho_o} = \frac{1}{1 + (\rho_w/\rho_o)(f_o/f_w)}$$

The paper's (29) prints $1/\big(1 + (\rho_o/\rho_w)(f_w/f_o)\big)$, which is the oil volume fraction. v1.0.0's code weights each liquid by its own volume fraction, as here (`specs/discrepancies.md`, D-2).

### PVT-MIX-3 · Liquid density from mixing

$$\rho_l = \alpha_{w,l}\rho_w + (1 - \alpha_{w,l})\rho_o$$

Equivalently $1/\rho_l = x_o/\rho_o + (1 - x_o)/\rho_w$: the mixture conserves mass and volume.

### PVT-MIX-4 · Liquid heat capacity

$$c_{pl} = \alpha_{w,l}c_{pw} + (1 - \alpha_{w,l})c_{po}$$

A volume-weighted average, as the paper states.

### PVT-MIX-5 · Gas–liquid surface tension

$$\sigma = \sigma_{od}(\rho_l, T)$$

The dead-oil correlation PVT-OIL-3, evaluated at the liquid's density and the local temperature, whatever the water fraction. Used by the slip model (SLIP-4, SLIP-6). Used by `v1.0.0`.

### PVT-MIX-6 · Liquid density with black oil

$$\rho_l(p, T) = \frac{(1 - \alpha_{w,l})(\rho_o + R_{so}\rho_{g,\text{sc}}) + \alpha_{w,l}\rho_w}{B_l}, \qquad B_l = (1 - \alpha_{w,l})B_o + \alpha_{w,l}B_w,$$

with $R_{so}$ and $B_o$ at $(p, T)$ as in PVT-OIL-9, capped at the gas–oil ratio $R_{go}$ by PVT-OIL-15, the water–liquid ratio $\alpha_{w,l}$ at standard conditions, and $B_w = 1$: the water is incompressible (PVT-WAT-1). The numerator is the mass of one Sm³ of stock-tank liquid with the gas dissolved in its oil, and the liquid formation volume factor $B_l$ is its volume at $(p, T)$. With no slip between oil and water, this is the density that PVT-OIL-13's mass rates imply: since PVT-OIL-15 keeps $R_{so}$ at or below $R_{go}$, PVT-OIL-13 dissolves it in full, and $w_l/\rho_l = B_l w_\text{res}/\rho_{l,\text{sc}}$ up to the smoothing of PVT-OIL-13 and PVT-OIL-15, the in-situ volume rate of the oil and water. Equivalently, $\rho_l = \lambda_w\rho_w + (1 - \lambda_w)\rho_{lo}$ with the live-oil density $\rho_{lo}$ of PVT-OIL-9 and the in-situ water fraction of the liquid $\lambda_w = \alpha_{w,l}B_w/B_l$, which is below $\alpha_{w,l}$ where the oil swells. Its row is $\rho_l - \rho_l(p, T)$ (kg/m³). With dead oil, $R_{so} = 0$ and $B_l = 1$, and this is PVT-MIX-1 with PVT-MIX-3's constant, exactly.

Until 2026-10-28, PVT-MIX-6 mixed $\rho_{lo}$ and $\rho_w$ by $\alpha_{w,l}$, the water fraction at standard conditions. That gave the water too much weight wherever the oil swells, so the liquid was too heavy: by 3.2% at the second vector's point, and by up to 7% at the bottomhole of the wells `develop`'s sampler draws (issue #8; decided by Bjarne, 2026-10-28; `specs/features/007-black-oil.md`).

### PVT-MIX-7 · Surface tension from the oil

$$\sigma = \sigma_{lo}\big(\sigma_{od}(\rho_o, T),\ R_{so}(p, T)\big)$$

the dead-oil correlation at the oil's density at standard conditions, with the live-oil correction of PVT-OIL-12 for black oil (none for dead oil) at the $R_{so}$ of PVT-OIL-15, whatever the water fraction (`specs/discrepancies.md`, D-8). With dead oil and no water it equals PVT-MIX-5 at roots, where $\rho_l = \rho_o$, but not away from them. Used by `develop`'s default.

### PVT-MIX-8 · Liquid viscosity

$$\mu_l = \lambda_w\mu_w + (1 - \lambda_w)\mu_o$$

by volume in situ, with the in-situ water fraction $\lambda_w = \alpha_{w,l}B_w/B_l$ of PVT-MIX-6, so that the density and the viscosity agree on how much water there is; $\mu_w$ from PVT-WAT-3 and $\mu_o$ from PVT-OIL-10 or PVT-OIL-11. With dead oil, $\lambda_w = \alpha_{w,l}$. Until 2026-10-28 it weighted by $\alpha_{w,l}$ with black oil too (issue #8).

### PVT-MIX-9 · Mixture viscosity

Hasan, Kabir and Sayarpour (2010), Eq. (A-3), weighted by the gas mass fraction of the flow, the $x$ of their Nomenclature:

$$\mu_m = x_g\mu_g + (1 - x_g)\mu_l, \qquad x_g = \frac{w_g}{w_m} = \frac{\alpha\rho_g v_g}{\alpha\rho_g v_g + (1-\alpha)\rho_l v_l} = \frac{\rho_g v_{gs}}{\rho_g v_{gs} + \rho_l v_{ls}},$$

with the phase mass rates $w_g$ and $w_l$ of the point's state (`nomenclature.md`), and $\mu_g$ from PVT-GAS-7 at $(T, \rho_g)$. At a root the mass rows make $w_g$ and $w_l$ the phase rates of INF-5 or PVT-OIL-13, so $x_g$ is the gas's share of the flow there, as in CHK-5 at the wellhead. Used by FRIC-3.

Decided by Bjarne, 2026-10-29: the gas mass fraction of the flow, as the source defines it. Until then PVT-MIX-9 took the in-situ mass fraction $\alpha\rho_g/(\alpha\rho_g + (1-\alpha)\rho_l)$, for which no source and no decision was recorded (`specs/features/004-friction.md`, `plans/develop_model_changes.md`). The two are equal without slip. With slip the in-situ fraction is the smaller, so $\mu_m$ was larger and the Reynolds number smaller: in Hasan and Kabir's Field Example 3.1 (2018), 0.032 against 0.066, and 12.67 cP against the printed 12.2 (Test vectors, Independent checks). $x_g$ is taken from the point's state, not from the phase rates of PVT-OIL-13, so that FRIC-3 stays a function of the point's state. The two agree at every root, and at every state the Rust core's march computes, since it builds each state from the phase rates; they differ only at the CasADi backend's trial states. On the wells of `develop`'s sampler, 1492 cases on the Rust core and 263 on the CasADi backend, the change finds an operating point in the same cases. With less friction, PBH falls in every case, by $10^{-3}$ bar at the median and at most 0.19 bar, and PWH and the reservoir liquid rate rise, by at most 0.27 bar and 0.35% but in one case. That case (well 455, its second sampled point) produces 0.02 kg/s of a gas-rich stream (GOR 2372 Sm³/Sm³) through a nearly closed choke, of an oil of 21.8 °API whose Beggs–Robinson viscosity is 290 Pa s at its wellhead, at 274 K, below the correlation's range. Its friction is laminar, where $f_D$ is proportional to $\mu_m$, and its slip makes the flowing fraction 0.65 against an in-situ 0.04, so $\mu_m$ falls by a factor of 2.7: PWH rises by 13.9 bar and the rate doubles. Both backends find the same single stable root there (`plans/evidence/README.md`, Independent checks).

### PVT-MIX-10 · Fluid at standard conditions

From `develop`'s fluid parameters: the liquid density and heat capacity at standard conditions, by volume,

$$\rho_{l,\text{sc}} = \alpha_{w,l}\rho_w + (1-\alpha_{w,l})\rho_o, \qquad c_{pl} = \alpha_{w,l} c_{pw} + (1-\alpha_{w,l}) c_{po},$$

(PVT-MIX-3 and PVT-MIX-4), the oil mass fraction of the liquid $x_o = (1 - \alpha_{w,l})\rho_o/\rho_{l,\text{sc}}$, and the gas mass fraction of the reservoir inflow

$$f_g = \frac{\rho_{g,\text{sc}} R_{go}}{\rho_{g,\text{sc}} R_{go} + \rho_o + \rho_w\alpha_{w,l}/(1 - \alpha_{w,l})},$$

per unit volume of stock-tank oil, with $0 \le \alpha_{w,l} < 1$. The `v1.0.0` configuration maps v1.0.0's $\rho_l$, $c_{pl}$ and $f_g$ to these parameters (`specs/sampling.md`, SMP-40).

## Options

| Option | IDs | Used by |
|---|---|---|
| Constant liquid density | PVT-MIX-1 | `v1.0.0`; `develop` with dead oil |
| Liquid density with black oil or a PVT table, oil and water mixed by in-situ volume | PVT-MIX-6 | `develop` default, and with `pvt=PVTTable(...)` |
| Mixing of oil and water | PVT-MIX-2 to PVT-MIX-4 | the `v1.0.0` sampler; `develop`'s fluid model, by the water–liquid ratio at standard conditions (PVT-MIX-10) |
| Surface tension from the liquid density | PVT-MIX-5 | `v1.0.0`; `develop` with `surface_tension_model='liquid'` |
| Surface tension from the oil density, with a live-oil correction | PVT-MIX-7 | `develop` default |
| Liquid and mixture viscosity, for friction | PVT-MIX-8, PVT-MIX-9 | `develop`'s `RoughnessFriction` |

## Safeguards

`liquid_mix` returns the pure liquid for $x_o = 1$ and $x_o = 0$, and asserts $x_o \in [0, 1]$. The general formula would divide by zero at $x_o = 0$.

PVT-MIX-9 has none. Its $x_g$ is $0/0$ at a point through which nothing flows, $w_m = 0$, as at $v_g = v_l = 0$, a state the CasADi backend's bounds admit ($v_g, v_l \ge 0$; `System.bounds` in `discretization.py`), and there FRIC-3 and the momentum row are NaN. They were finite there with the in-situ fraction, but the residual was not: at such a point THM-1 divides by the flow's heat capacity $c_{pg}w_g + c_{pl}w_l$ per unit area, which is zero with $w_m$, so the energy row is not finite there, and at the wellhead neither is CHK-5's $x_g$, as in v1.0.0. So $x_g$ adds no state at which the residual or its Jacobian is not finite: on a sampled well, at $v_g = v_l = 0$, at $\alpha = 0$ with $v_l = 0$ and at $\alpha = 1$ with $v_g = 0$ at every point, DISC-10 and CHK-1 are NaN with either fraction, and DISC-9 with the flowing one. A guard on $x_g$ would make the momentum row finite there and leave the residual NaN; at `v1.0.0+chen#6` of the comparison set, where one start of the CasADi search now ends with Ipopt's `Invalid_Number_Detected` instead of at a state with a supersonic cell (SOL-8), it does so with a guard too. No search starts there: the CasADi backend marches from $p_0 < p_r$, where the liquid flows, and the Rust core builds each state from the phase rates at $p_0 < p_r$.

## Sources

- Paper §4.2, equations (28), (29) and (31); Appendix A.1 for the surface tension.
- Hasan, Kabir and Sayarpour (2010), Eq. (A-3) and the Nomenclature, for PVT-MIX-9.
- Hasan and Kabir (2018), *Fluid flow and heat transfer in wellbores*, 2nd ed., Society of Petroleum Engineers, §3.4, Field Example 3.1, a worked example of PVT-MIX-9.
- `fluids` 1.3.1 (Caleb Bell, MIT licence), for the independent checks of `plans/evidence/independent_checks.py`.
- Feature specs `specs/features/003-surface-tension.md` and `005-fluid-model.md`.

## Test vectors

**Independent checks**, on `develop` at `f7f1218` (2026-10-29; `README.md`, Test vectors), and rerun after Bjarne's ruling on PVT-MIX-9 the same day. PVT-MIX-9's mixing rule agrees with `fluids` 1.3.1 (`two_phase_voidage.gas_liquid_viscosity`, Cicchitti's rule) to rounding, given the gas mass fraction of the flow (`plans/evidence/independent_checks.py`). Both backends reproduce Hasan and Kabir's (2018) Field Example 3.1 at the precision printed, 12.2 cP, at the superficial velocities its solution carries (`tests/test_worked_examples.py`); until the ruling PVT-MIX-9 took the in-situ mass fraction and gave 12.67 cP, a finding of these checks (PVT-MIX-9). PVT-MIX-6, PVT-MIX-7, PVT-MIX-8 and PVT-MIX-10 are mixing rules and definitions, with no independent source.

<!-- vectors:begin -->
Generated by `specs/tools/make_v1_vectors.py` from ManyWells v1.0.0 (casadi 3.6.4). Do not edit by hand.

### PVT-MIX-2, PVT-MIX-3, PVT-MIX-4

| rho_o | cp_o | rho_w | cp_w | x_o | → rho_l | → cp_l |
|---|---|---|---|---|---|---|
| 850.0 | 2000.0 | 999.1 | 4184.0 | 0.5 | 918.5387485803905 | 3003.947866529663 |
| 825.0 | 2000.0 | 999.1 | 4184.0 | 0.9 | 839.6311462885433 | 2183.5406289154416 |
| 925.0 | 2000.0 | 999.1 | 4184.0 | 0.1 | 991.1600047189542 | 3949.9790864533848 |
| 870.0 | 2000.0 | 999.1 | 4184.0 | 1.0 | 870.0 | 2000.0 |
| 870.0 | 2000.0 | 999.1 | 4184.0 | 0.0 | 999.1 | 4184.0 |
<!-- vectors:end -->

<!-- vectors:begin develop -->
Generated by `specs/tools/make_develop_vectors.py` from develop (casadi 3.8.1): they pin develop's options. Do not edit by hand.

### PVT-MIX-6

| api | sg_gas | gor | wlr | p | T | → rho_l |
|---|---|---|---|---|---|---|
| 35.0 | 0.65 | 150.0 | 0.0 | 100.0 | 350.0 | 747.1961049371773 |
| 35.0 | 0.65 | 150.0 | 0.4 | 250.0 | 380.0 | 776.5691569777025 |
| 35.0 | 0.65 | 50.0 | 0.0 | 250.0 | 380.0 | 714.5180562374418 |
| 35.0 | 0.65 | 50.0 | 0.3 | 250.0 | 380.0 | 787.4352396651066 |

### PVT-MIX-7

| api | sg_gas | p | T | rho_l | → sigma |
|---|---|---|---|---|---|
| 35.0 | 0.65 | 100.0 | 350.0 | 700.0 | 0.012422595535172514 |
| 35.0 | 0.65 | 250.0 | 380.0 | 700.0 | 0.0037099021593460144 |
| 25.0 | 0.8 | 30.0 | 300.0 | 900.0 | 0.02478627716574904 |

### PVT-MIX-8

| api | sg_gas | wlr | p | T | → mu_l |
|---|---|---|---|---|---|
| 35.0 | 0.65 | 0.0 | 100.0 | 350.0 | 0.001058720284352048 |
| 35.0 | 0.65 | 0.5 | 250.0 | 380.0 | 0.0003764281352302541 |

### PVT-MIX-9

| mu_l | mu_g | alpha | rho_l | rho_g | v_g | v_l | → mu_m |
|---|---|---|---|---|---|---|---|
| 0.001 | 1.5e-05 | 0.0 | 800.0 | 50.0 | 0.0 | 2.0 | 0.001 |
| 0.001 | 1.5e-05 | 0.5 | 800.0 | 50.0 | 6.0 | 2.0 | 0.0008444736842105262 |
| 0.002 | 2e-05 | 0.9 | 850.0 | 100.0 | 20.0 | 3.0 | 0.0002656934306569343 |
| 0.001 | 1.5e-05 | 0.5 | 800.0 | 50.0 | 2.0 | 2.0 | 0.0009420588235294118 |

### PVT-MIX-10

| rho_o | rho_g_sc | rho_w | gor | wlr | cp_o | cp_w | → f_g | → rho_l_sc | → cp_l | → x_o |
|---|---|---|---|---|---|---|---|---|---|---|
| 850.0 | 0.8 | 999.1 | 150.0 | 0.0 | 2000.0 | 4184.0 | 0.12371134020618557 | 850.0 | 2000.0 | 1.0 |
| 870.0 | 0.75 | 1020.0 | 300.0 | 0.4 | 2100.0 | 4000.0 | 0.1267605633802817 | 930.0 | 2860.0 | 0.5612903225806452 |
<!-- vectors:end develop -->

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| PVT-MIX-1 | §2.2 | `simulator.py` `_closure_relations` (`g3`) | rows |
| PVT-MIX-2 | (29) | `pvt.py` `liquid_mix` (`vol_fraction`, the oil fraction) | vectors |
| PVT-MIX-3 | (28) | `pvt.py` `liquid_mix` | vectors |
| PVT-MIX-4 | (31) | `pvt.py` `liquid_mix` | vectors |
| PVT-MIX-5 | App. A.1 | `slip.py` `classify_flow_regime`, `SlipModel.harmathy_rise_velocity` | rows |
| PVT-MIX-6 | — | — | vectors; rows (in the v1.0.0 configuration, as PVT-MIX-1); property: tests/test_fluid.py (the in-situ volume rate, exact with dead oil) |
| PVT-MIX-7 | — | — | vectors; property: tests/test_fluid.py |
| PVT-MIX-8 | — | — | vectors; property: tests/test_fluid.py (the in-situ water fraction) |
| PVT-MIX-9 | — | — | vectors; property: tests/test_pvt.py |
| PVT-MIX-10 | — | — | vectors; property: tests/test_configurations.py (the v1.0.0 mapping) |
