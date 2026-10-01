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

In `develop` the fluid model (`FluidModel`) takes the phase densities at standard conditions $\rho_o$, $\rho_{g,\text{sc}}$ and $\rho_w$, the gas–oil ratio $R_{go}$ and the water–liquid ratio $\alpha_{w,l}$ (both at standard conditions, `gor` and `wlr`), and the heat capacities $c_{pg}$, $c_{po}$ and $c_{pw}$; PVT-MIX-10 derives the rest. Its `surface_tension(p, T, rho_l)` takes the point's liquid density, which PVT-MIX-5 needs, and its `surface_tension_model` chooses PVT-MIX-5 (`'liquid'`) or PVT-MIX-7 (`'oil'`).

## Equations

### PVT-MIX-1 · Constant liquid density

$$\rho_l = \rho_{l,\text{const}}$$

The mixed liquid is incompressible. Its row in DISC-6 is $\rho_l - \rho_{l,\text{const}}$ (kg/m³). Used by `v1.0.0`.

### PVT-MIX-2 · Water volume fraction of the liquid

$$\alpha_{w,l} = \frac{(1 - x_o)/\rho_w}{(1 - x_o)/\rho_w + x_o/\rho_o} = \frac{1}{1 + (\rho_w/\rho_o)\,(f_o/f_w)}$$

The paper's (29) prints $1/\big(1 + (\rho_o/\rho_w)(f_w/f_o)\big)$, which is the oil volume fraction. v1.0.0's code weights each liquid by its own volume fraction, as here (`specs/discrepancies.md`, D-2).

### PVT-MIX-3 · Liquid density from mixing

$$\rho_l = \alpha_{w,l}\,\rho_w + (1 - \alpha_{w,l})\,\rho_o$$

Equivalently $1/\rho_l = x_o/\rho_o + (1 - x_o)/\rho_w$: the mixture conserves mass and volume.

### PVT-MIX-4 · Liquid heat capacity

$$c_{pl} = \alpha_{w,l}\,c_{pw} + (1 - \alpha_{w,l})\,c_{po}$$

A volume-weighted average, as the paper states.

### PVT-MIX-5 · Gas–liquid surface tension

$$\sigma = \sigma_{od}(\rho_l, T)$$

The dead-oil correlation PVT-OIL-3, evaluated at the liquid's density and the local temperature, whatever the water fraction. Used by the slip model (SLIP-4, SLIP-6). Used by `v1.0.0`.

### PVT-MIX-6 · Liquid density with black oil

$$\rho_l(p, T) = \alpha_{w,l}\,\rho_w + (1 - \alpha_{w,l})\,\rho_{lo}(p, T)$$

with the live-oil density of PVT-OIL-9 and the water–liquid ratio $\alpha_{w,l}$ at standard conditions; the water is incompressible (PVT-WAT-1), and the in-situ water fraction is not recomputed as the oil swells. Its row is $\rho_l - \rho_l(p, T)$ (kg/m³). With dead oil, $\rho_{lo} = \rho_o$ and this is PVT-MIX-1 with PVT-MIX-3's constant, exactly.

### PVT-MIX-7 · Surface tension from the oil

$$\sigma = \sigma_{lo}\big(\sigma_{od}(\rho_o, T),\ R_{so}(p, T)\big)$$

the dead-oil correlation at the oil's density at standard conditions, with the live-oil correction of PVT-OIL-12 for black oil (none for dead oil), whatever the water fraction (`specs/discrepancies.md`, D-8). With dead oil and no water it equals PVT-MIX-5 at roots, where $\rho_l = \rho_o$, but not away from them. Used by `develop`'s default.

### PVT-MIX-8 · Liquid viscosity

$$\mu_l = \alpha_{w,l}\,\mu_w + (1 - \alpha_{w,l})\,\mu_o$$

by volume at standard conditions, with $\mu_w$ from PVT-WAT-3 and $\mu_o$ from PVT-OIL-10 or PVT-OIL-11.

### PVT-MIX-9 · Mixture viscosity

Hasan, Kabir and Sayarpour (2010), Eq. (A-3), weighted by the in-situ mass fraction of gas:

$$\mu_m = x\,\mu_g + (1 - x)\,\mu_l, \qquad x = \frac{\alpha\rho_g}{\alpha\rho_g + (1-\alpha)\rho_l},$$

with $\mu_g$ from PVT-GAS-7 at $(T, \rho_g)$. Used by FRIC-3.

### PVT-MIX-10 · Fluid at standard conditions

From `develop`'s fluid parameters: the liquid density and heat capacity at standard conditions, by volume,

$$\rho_{l,\text{sc}} = \alpha_{w,l}\rho_w + (1-\alpha_{w,l})\rho_o, \qquad c_{pl} = \alpha_{w,l} c_{pw} + (1-\alpha_{w,l}) c_{po},$$

(PVT-MIX-3 and PVT-MIX-4), the oil mass fraction of the liquid $x_o = (1 - \alpha_{w,l})\rho_o/\rho_{l,\text{sc}}$, and the gas mass fraction of the reservoir inflow

$$f_g = \frac{\rho_{g,\text{sc}} R_{go}}{\rho_{g,\text{sc}} R_{go} + \rho_o + \rho_w\,\alpha_{w,l}/(1 - \alpha_{w,l})},$$

per unit volume of stock-tank oil, with $0 \le \alpha_{w,l} < 1$. The `v1.0.0` configuration maps v1.0.0's $\rho_l$, $c_{pl}$ and $f_g$ to these parameters (`specs/sampling.md`, SMP-40).

## Options

| Option | IDs | Used by |
|---|---|---|
| Constant liquid density | PVT-MIX-1 | `v1.0.0`; `develop` with dead oil |
| Liquid density with black oil | PVT-MIX-6 | `develop` default |
| Mixing of oil and water | PVT-MIX-2 to PVT-MIX-4 | the `v1.0.0` sampler; `develop`'s fluid model, by the water–liquid ratio at standard conditions (PVT-MIX-10) |
| Surface tension from the liquid density | PVT-MIX-5 | `v1.0.0`; `develop` with `surface_tension_model='liquid'` |
| Surface tension from the oil density, with a live-oil correction | PVT-MIX-7 | `develop` default |
| Liquid and mixture viscosity, for friction | PVT-MIX-8, PVT-MIX-9 | `develop`'s `RoughnessFriction` |

## Safeguards

`liquid_mix` returns the pure liquid for $x_o = 1$ and $x_o = 0$, and asserts $x_o \in [0, 1]$. The general formula would divide by zero at $x_o = 0$.

## Sources

- Paper §4.2, equations (28), (29) and (31); Appendix A.1 for the surface tension.
- Hasan, Kabir and Sayarpour (2010), Eq. (A-3), for PVT-MIX-9.
- Feature specs `specs/features/003-surface-tension.md` and `005-fluid-model.md`.

## Test vectors

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
| 35.0 | 0.65 | 150.0 | 0.0 | 100.0 | 350.0 | 747.1961049371042 |
| 35.0 | 0.65 | 150.0 | 0.4 | 250.0 | 380.0 | 801.6325807183971 |

### PVT-MIX-7

| api | sg_gas | p | T | rho_l | → sigma |
|---|---|---|---|---|---|
| 35.0 | 0.65 | 100.0 | 350.0 | 700.0 | 0.01242259553516064 |
| 35.0 | 0.65 | 250.0 | 380.0 | 700.0 | 0.0037099021593389515 |
| 25.0 | 0.8 | 30.0 | 300.0 | 900.0 | 0.024786277165719954 |

### PVT-MIX-8

| api | sg_gas | wlr | p | T | → mu_l |
|---|---|---|---|---|---|
| 35.0 | 0.65 | 0.0 | 100.0 | 350.0 | 0.0010587202843511084 |
| 35.0 | 0.65 | 0.5 | 250.0 | 380.0 | 0.0003600589073794999 |

### PVT-MIX-9

| mu_l | mu_g | alpha | rho_l | rho_g | → mu_m |
|---|---|---|---|---|---|
| 0.001 | 1.5e-05 | 0.0 | 800.0 | 50.0 | 0.001 |
| 0.001 | 1.5e-05 | 0.5 | 800.0 | 50.0 | 0.0009420588235294118 |
| 0.002 | 2e-05 | 0.9 | 850.0 | 100.0 | 0.0009817142857142858 |

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
| PVT-MIX-6 | — | — | vectors; rows (in the v1.0.0 configuration, as PVT-MIX-1) |
| PVT-MIX-7 | — | — | vectors; property: tests/test_fluid.py |
| PVT-MIX-8 | — | — | vectors |
| PVT-MIX-9 | — | — | vectors; property: tests/test_pvt.py |
| PVT-MIX-10 | — | — | vectors; property: tests/test_configurations.py (the v1.0.0 mapping) |
