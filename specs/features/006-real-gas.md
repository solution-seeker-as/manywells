# 006 · Real gas

*Feature spec, written after the fact in Step 7 of `plans/manywells-v2-plan.md` (2026-10-01): change 6 of `plans/develop_model_changes.md`. Status: draft for Bjarne's sign-off.*

## Motivation

At the pressures of most wells the gas deviates from the ideal gas law by 10 to 20% (paper §8, part of limitation 1).

## Delta

- `pvt/gas.md`: PVT-GAS-3 (the real gas law, $c_\text{bar}p = Z\rho_g R_s T$, whose row is $p - \rho_g Z R_s T/c_\text{bar}$), PVT-GAS-4 (Papay's z-factor) and PVT-GAS-5 (Sutton's pseudo-critical properties from the gas gravity).
- The gas-law row is in its canonical form, in bar, for every gas option (`FluidModel.gas_law_row`). `develop` had the density form $\rho_g - c_\text{bar}p/(Z R_s T)$; with it, Ipopt stopped at states whose phase mass rates drifted along the well by up to $1.3\cdot10^{-6}$ of the total rate at $N = 400$, above the verifier's bound of $10^{-6}$. With the canonical form the search was also 25% faster on the case set (2026-10-01). It changes the scaling of a row, not the roots. Accepted by Bjarne under principle 7, 2026-10-01 (`012-root-search.md`).
- Code: `pvt.gas`, `FluidModel.z_factor`, `gas_density`, `gas_law_row`.

## Off in the `v1.0.0` configuration

`FluidModel(ideal_gas=True)`: $Z = 1$, and the row is PVT-GAS-1's exactly.

## Acceptance

The PVT-GAS-1 row vectors in the `v1.0.0` configuration (exact now, without the factor the test adapter divided out); the develop vectors of PVT-GAS-3 to PVT-GAS-5; `tests/test_pvt.py` (pseudo-critical properties of methane, $Z$ near 1 at standard conditions, falling with pressure).

## Open

Nothing checks Papay's range ($p_{pr} < 6$, $T_{pr} > 1.05$); `specs/sampling.md` (SMP-43) can draw gases and pressures outside it.
