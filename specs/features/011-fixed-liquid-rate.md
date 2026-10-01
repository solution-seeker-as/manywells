# 011 · Inflow returns the liquid rate; fixed liquid rate

*Feature spec, written after the fact in Step 7 of `plans/manywells-v2-plan.md` (2026-10-01): change 11 of `plans/develop_model_changes.md`. Status: draft for Bjarne's sign-off.*

## Motivation

The gas–liquid split of the reservoir inflow is a property of the fluid, not of the inflow model; in v1.0.0 each inflow model carried its own $f_g$. With the fluid model of 005, the inflow model gives the liquid rate only.

## Delta

- `inflow.md`: inflow models return $w_\text{res}$ only (`liquid_mass_flow_rate(p, p_r)`); the fluid model gives the gas (INF-4). INF-8, a fixed liquid rate with the gas from INF-4, replaces INF-3, which fixed both rates; INF-3 is not implemented on `develop`.
- `ProductivityIndex(k_l=0.5)` is the default inflow.
- The inflow models are frozen dataclasses and raise `ValueError` for negative parameters.

## Off in the `v1.0.0` configuration

Vogel (INF-1) and the productivity index (INF-2) are unchanged; the `v1.0.0` configuration does not allow INF-8 (`configurations.check`).

## Acceptance

The INF-1, INF-2 and INF-4 vectors from v1.0.0; `tests/test_inflow.py`; `scripts/sim_examples/fixed_rate.py` (`tests/test_examples.py`).

## Breaking change

`Vogel(w_l_max, f_g)` and `ProductivityIndex(k_l, f_g)` lost their `f_g`; it is in `FluidModel`. CHANGELOG.
