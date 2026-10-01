# 010 · Lift-gas temperature

*Feature spec, written after the fact in Step 7 of `plans/manywells-v2-plan.md` (2026-10-01): change 10 of `plans/develop_model_changes.md`. Status: draft for Bjarne's sign-off.*

## Motivation

v1.0.0 injects the lift gas at the reservoir temperature (THM-3). Lift gas comes from the surface and is often much colder than the reservoir, which cools the flow at the bottomhole.

## Delta

- `thermal.md`: THM-5, the heat-capacity-weighted mix of the reservoir fluid at $T_r$ and the lift gas at $T_{lg}$, $T_\text{in} = T_r + H_{lg}(T_{lg} - T_r)/(H_\text{res} + H_{lg})$; its row is $T_0 - T_\text{in}$.
- `BoundaryConditions.T_lg` (K); `None` means $T_r$. The solver's lower temperature bound is $\min(T_s, T_{lg})$.
- The mixing moves into `thermal.ThermalModel.inflow_temperature`, with a switch (`lift_gas_mixing`). It is written in the form above, which is exactly $T_r$ when $T_{lg} = T_r$ or $w_{lg} = 0$, so the operating point can be a parameter of the system (the simulator no longer branches on `w_lg > 0`).

## Off in the `v1.0.0` configuration

`lift_gas_mixing=False`: THM-3, $T_0 = T_r$, whatever $T_{lg}$.

## Acceptance

The THM-3 row vectors in the `v1.0.0` configuration; the develop vectors of THM-5; `tests/test_thermal.py` (bounded by $T_{lg}$ and $T_r$, colder with more lift gas, exactly $T_r$ without it); `tests/test_model_properties.py` (cold lift gas).

## Open

The upper temperature bound of the solver is $T_r + 1$ K, so lift gas hotter than that cannot be simulated; no sampler draws it (SMP-44).
