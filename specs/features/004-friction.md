# 004 · Friction from roughness and viscosity

*Feature spec, written after the fact in Step 7 of `plans/manywells-v2-plan.md` (2026-10-01): change 4 of `plans/develop_model_changes.md`. Status: draft for Bjarne's sign-off.*

## Motivation

v1.0.0 uses one Darcy friction factor for the whole well, drawn per well (paper §8, limitation 3). The friction factor depends on the Reynolds number and the pipe's roughness, so it changes along the well and with the rate.

## Delta

- `friction.md`: FRIC-3 (Reynolds number of the mixture), FRIC-4 (Chen), FRIC-5 (Haaland), FRIC-6 (laminar–turbulent blend by a sigmoid at Re = 3000, with smooth-max guards). Friction is a component of the well: `FrictionModel.pressure_gradient(s, fluid, D)`, with `FixedFrictionFactor(f_D)` (FRIC-2) and `RoughnessFriction(roughness, correlation)`; Haaland can now be chosen.
- Viscosities: PVT-MIX-8 (liquid, by volume), PVT-MIX-9 (mixture, mass-weighted), PVT-OIL-10 and PVT-OIL-11 (Beggs–Robinson, dead and live), PVT-WAT-3 (water), PVT-GAS-7 (Lee–Gonzalez–Eakin). The fluid model supplies them (`FluidModel.mixture_viscosity`), so friction imports nothing from `pvt/`.
- `smoothing.md`: SMO-4, the sigmoid.
- Code: `friction.py`, `pvt/`.

## Off in the `v1.0.0` configuration

`FixedFrictionFactor(f_D)`: FRIC-2, required by `configurations.check`.

## Acceptance

The DISC-4 row vectors in the `v1.0.0` configuration; the develop vectors of FRIC-3 to FRIC-6 and of the viscosities; `tests/test_friction.py` (Blasius and laminar limits, monotonicity, the components); `tests/test_model_properties.py` (non-negative friction).

## Open

Beggs–Robinson's stated range starts at 100 °F (38 °C); wellhead temperatures are often colder. Nothing checks the ranges.
