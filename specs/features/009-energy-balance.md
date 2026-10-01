# 009 · Frictional heating and a gravity term in the energy balance

*Feature spec, written after the fact in Step 7 of `plans/manywells-v2-plan.md` (2026-10-01): change 9 of `plans/develop_model_changes.md`. Status: draft for Bjarne's sign-off.*

## Motivation

v1.0.0's energy balance has heat loss to the surroundings only (paper §8, limitation 4). The total-energy balance also has the work against friction, which heats the liquid, and the work against gravity, which cools the flow (`docs/thermal_energy_modeling.md`).

## Delta

- `thermal.md`: THM-6 (frictional heating, $\Phi_f = (1-\alpha)v_l F/C$) and THM-7 (gravity term, $\Phi_g = g\cos\theta\,(\alpha\rho_g v_g + (1-\alpha)\rho_l v_l - (1-\alpha)v_l\rho_m)/C$), with $C$ the heat-capacity flux.
- `balances.md`: BAL-12, $dT/dz = -H + \Phi_f - \Phi_g$.
- `discretization.md`: DISC-10, $T_i - T_{i-1} - \Delta\text{MD}_i\,(dT/d\text{MD})_i$.
- The terms move out of the simulator into `thermal.ThermalModel`, with a switch each (`frictional_heating`, `gravity_term`; `specs/architecture.md`). Before Step 7 they were always on.
- Test vectors pin the terms (`plans/improvements.md` §3).

## Off in the `v1.0.0` configuration

Both switches off: DISC-10 is DISC-5. Before Step 7, `develop`'s energy rows differed from v1.0.0's by up to 0.62 K per cell in the row vectors; now they match.

## Acceptance

The DISC-5 row vectors in the `v1.0.0` configuration; the develop vectors of THM-6 and THM-7; `tests/test_thermal.py` (lapse rate of pure gas, no gravity term for pure liquid, $F/(\rho_l c_{pl})$ for pure liquid); `tests/test_model_properties.py` (convergence, heat flowing outwards without lift gas).

## Out of scope

Joule–Thomson cooling (the thermal model receives the cell's pressure gradient for it, `specs/architecture.md`), heat capacities that vary with $(p, T)$, and a transient formation temperature.
