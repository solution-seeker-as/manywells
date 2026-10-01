# 001 · Deviated and L-shaped wells

*Feature spec, written after the fact in Step 7 of `plans/manywells-v2-plan.md` (2026-10-01): change 1 of `plans/develop_model_changes.md`. Status: draft for Bjarne's sign-off.*

## Motivation

v1.0.0 models a vertical pipe only (paper §8, limitation 2). Most production wells are deviated, and many are L-shaped with a horizontal section. Gravity acts along the vertical, friction along the flow path, and the ambient temperature follows the true vertical depth, so a trajectory changes all three.

## Delta

- `geometry.md`: GEO-3 (a survey of (MD, TVD) stations is the grid; per cell $\Delta\text{MD}_i$ and $\cos\theta_i$, per point the TVD fraction $f_i$) and GEO-4 (a uniform grid in MD from a sparse survey). GEO-1 is GEO-4 with the survey $(0, 0), (L, L)$.
- `balances.md`: BAL-11, momentum along an inclined path, gravity weighted by $\cos\theta$.
- `discretization.md`: DISC-9, the momentum row with friction over $\Delta\text{MD}_i$ and gravity over $\Delta\text{MD}_i\cos\theta_i$; DISC-10, the energy row over $\Delta\text{MD}_i$; DISC-11, the system and row order, in which the closures of point $i$ use the inclination of cell $i$ (point 0 that of cell 1).
- `thermal.md`: THM-4, the ambient temperature linear in TVD.
- Code: `geometry.WellGeometry`, `discretization.cell_rows` and `closure_rows`, `thermal.ThermalModel.ambient_temperature`.

## Off in the `v1.0.0` configuration

`WellGeometry.vertical(L, N, D)`: $\Delta\text{MD}_i = L/N$, $\cos\theta_i = 1$, $f_i = 1 - i/N$, and DISC-9, DISC-10 and THM-4 reduce to DISC-4, DISC-5 and THM-2. `configurations.check` requires a vertical, uniform grid.

## Acceptance

- `tests/test_geometry.py`: vertical, deviated and L-shaped surveys, and the validation.
- The row vectors in the `v1.0.0` configuration (`tests/test_spec_vectors.py`), and the verifier on the case set (`specs/verification.md`).
- `tests/test_model_properties.py`: Invariants, spot checks, convergence at first order and the two-root stability property on deviated and L-shaped wells.

## Out of scope

Several flow paths (annulus, multilaterals; `specs/goals.md`, non-goals), wells that descend towards the wellhead ($\cos\theta < 0$), and a survey file reader.
