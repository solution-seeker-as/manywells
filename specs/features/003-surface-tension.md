# 003 · Surface tension from the fluid model

*Feature spec, written after the fact in Step 7 of `plans/manywells-v2-plan.md` (2026-10-01): change 3 of `plans/develop_model_changes.md`. Status: draft for Bjarne's sign-off.*

## Motivation

v1.0.0 evaluates the dead-oil surface tension at the liquid's density, whatever the water fraction (PVT-MIX-5; `specs/discrepancies.md`, D-8). `develop` evaluates it at the oil's density and, for black oil, corrects it for the gas dissolved in the oil, which lowers the surface tension.

## Delta

- `pvt/mixture.md`: PVT-MIX-7, $\sigma = \sigma_{lo}(\sigma_{od}(\rho_o, T), R_{so})$, as an option of the fluid model (`surface_tension_model='oil'`, the default); PVT-MIX-5 is the option `'liquid'`. `FluidModel.surface_tension(p, T, rho_l)` takes the point's liquid density, which PVT-MIX-5 needs (`specs/architecture.md`).
- `pvt/oil.md`: PVT-OIL-12, the live-oil correction of Abdul-Majeed and Abu Al-Soof (2000), Eqs. (4) and (5), blended by a sigmoid at 50 Sm³/Sm³. The two branches do not meet there (0.425 against 0.375); the docstring that said they do is corrected.
- Code: `pvt.fluid.FluidModel.surface_tension`, `pvt.black_oil.live_oil_surface_tension`.

## Off in the `v1.0.0` configuration

`surface_tension_model='liquid'`: $\sigma_{od}$ at the state's $\rho_l$, so the SLIP-1 rows match v1.0.0's away from roots too (they differed by up to 13% before).

## Acceptance

The SLIP-1 row vectors in the `v1.0.0` configuration; the develop vectors of PVT-MIX-7 and PVT-OIL-12; `tests/test_fluid.py` (the `'liquid'` model uses the local density, the `'oil'` model ignores it).

## Open

The 12% step between the source's two branches is smoothed, not resolved; a source with a continuous correlation would remove it.
