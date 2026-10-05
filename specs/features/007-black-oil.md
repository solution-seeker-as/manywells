# 007 · Black oil

*Feature spec, written after the fact in Step 7 of `plans/manywells-v2-plan.md` (2026-10-01): change 7 of `plans/develop_model_changes.md`. Status: draft for Bjarne's sign-off.*

## Motivation

v1.0.0's liquid is incompressible and dissolves no gas (paper §8, limitation 1). Oil at reservoir pressure holds dissolved gas, which comes out of solution as the pressure falls up the well, and the oil swells with it.

## Delta

- `pvt/oil.md`: PVT-OIL-4 (black oil), PVT-OIL-5 (separator gas-gravity correction), PVT-OIL-6 ($R_{so}$, Vazquez–Beggs), PVT-OIL-7 (bubble-point cap), PVT-OIL-8 ($B_o$, Vazquez–Beggs), PVT-OIL-9 (live-oil density), PVT-OIL-14 (Standing's bubble point; not used by the simulator).
- `pvt/mixture.md`: PVT-MIX-6, the liquid density with black oil, mixed with water by the water–liquid ratio at standard conditions.
- Code: `pvt.black_oil.BlackOilPVT`, `FluidModel(oil_model='black_oil')`.

## Off in the `v1.0.0` configuration

`FluidModel(oil_model='dead_oil')`: $R_{so} = 0$ and $B_o = 1$, so the liquid density is the constant of PVT-MIX-1, exactly.

## Acceptance

The PVT-MIX-1 row vectors in the `v1.0.0` configuration; the develop vectors of PVT-OIL-5 to PVT-OIL-9 and PVT-MIX-6; `tests/test_black_oil.py` and `tests/test_black_oil_consistency.py` (monotonicity, the bubble-point cap, consistency of $R_{so}$, $B_o$ and $B_g$).

## Rulings and open items

- **Separator correction (PVT-OIL-5).** The code took the natural log of $p_\text{sep}/114.7$; Vazquez and Beggs (1980) have $\log_{10}$. At standard separator conditions the code's corrected gas gravity was $0.75\gamma_g$ instead of $0.89\gamma_g$, so $R_{so}$ was 16% lower than the source's. Found in Step 7; Bjarne ruled on 2026-10-01 to follow the source, and the code now has $\log_{10}$, checked against hand values in `tests/test_black_oil.py`. A change of `develop`'s default model, not of the `v1.0.0` configuration.
- **Range (open).** API gravity is checked against 10 to 40; pressures and temperatures are not. Wells with a small oil fraction get gas–oil ratios above $10^4$ Sm³/Sm³ in the sampler (SMP-43).
