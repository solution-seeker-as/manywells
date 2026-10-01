# 005 · Unified fluid model

*Feature spec, written after the fact in Step 7 of `plans/manywells-v2-plan.md` (2026-10-01): change 5 of `plans/develop_model_changes.md`. Status: draft for Bjarne's sign-off.*

## Motivation

v1.0.0 takes the mixed liquid's density and heat capacity, the gas constant and the gas mass fraction as parameters, computed by the sampler. A fluid described by its stock-tank properties, the gas–oil ratio and the water cut is what an engineer has, and it is what the black-oil correlations need.

## Delta

- `pvt/mixture.md`: PVT-MIX-10, the fluid at standard conditions ($f_g$, $\rho_{l,\text{sc}}$, $c_{pl}$, $x_o$) from $\rho_o$, $\rho_{g,\text{sc}}$, $\rho_w$, the gas–oil ratio and the water–liquid ratio.
- `pvt/gas.md`: PVT-GAS-6, the gas gravity, molecular weight and gas constant from $\rho_{g,\text{sc}}$; PVT-GAS-8, the gas formation volume factor (not used by the simulator).
- `pvt/water.md`: PVT-WAT-2, the water formation volume factor, with its sign fixed in the first-order form $1 - c_w(p - p_\text{ref})$ (`plans/improvements.md` §1.2; not used by the simulator; signed off by Bjarne, 2026-10-01); PVT-WAT-3, the water viscosity.
- `FluidModel` is a frozen dataclass; it raises `ValueError` for an unknown oil or surface-tension model and a water–liquid ratio outside $[0, 1)$. `p_sep` and `p_bubble` are in bar, like every other interface (`plans/improvements.md` §1.6). It owns the reservoir gas–liquid split (INF-4) and the phase rates (INF-5, PVT-OIL-13), which moved from the simulator (`specs/architecture.md`).
- `nomenclature.md`: the new symbols.

## Off in the `v1.0.0` configuration

`configurations.v1_fluid` maps v1.0.0's $\rho_l$, $R_s$, $c_{pg}$, $c_{pl}$ and $f_g$ to a dead-oil, ideal-gas fluid with no water ($\rho_o = \rho_l$, $c_{po} = c_{pl}$, $\rho_{g,\text{sc}} = p_\text{ref}/(R_s T_\text{ref})$ and the gas–oil ratio that gives $f_g$), which gives them back to rounding (`specs/sampling.md`, SMP-40).

## Acceptance

`tests/test_configurations.py` (the mapping), `tests/test_fluid.py`, the develop vectors of PVT-MIX-10, PVT-GAS-6 and PVT-WAT-2/3, and the row vectors in the `v1.0.0` configuration.

## Breaking change

`p_sep` and `p_bubble` are in bar (were Pa). CHANGELOG, `specs/goals.md` (API).
