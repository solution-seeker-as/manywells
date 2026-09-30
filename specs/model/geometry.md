# Geometry

## Purpose

The shape of the flow path: the well trajectory and the pipe's cross-section. A well has one flow path from bottomhole to wellhead (`specs/goals.md`, non-goals).

## Interface

| Input | Unit | Meaning |
|---|---|---|
| $L$ | m | pipe length |
| $D$ | m | inner diameter, $> 0$ |

Outputs: the coordinate $z \in [0, L]$ of DISC-1, and the area $A$ (m²). Geometry is fixed per well, so these are floats, never CasADi symbols.

## Equations

### GEO-1 · Vertical pipe

The pipe is vertical, of length $L > 0$, from the bottomhole at $z = 0$ to the wellhead at $z = L$. The true vertical depth of the bottomhole equals $L$, and gravity acts along the full length of the pipe (BAL-6). Used by `v1.0.0`.

### GEO-2 · Circular cross-section

The pipe has a constant inner diameter $D > 0$ and cross-sectional area

$$A = \pi (D/2)^2.$$

## Options

| Option | IDs | Used by |
|---|---|---|
| Vertical pipe | GEO-1 | `v1.0.0` |
| Deviated and L-shaped wells from an (MD, TVD) survey | Step 7 | `develop` |

## Safeguards

None. v1.0.0 asserts $L > 0$ and $D > 0$.

## Sources

Paper §2 and Fig. 1.

## Test vectors

The row vectors (`discretization.md`) use GEO-1 and GEO-2 in every mass-rate and gravity term.

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| GEO-1 | §2, Fig. 1 | `simulator.py` `WellProperties.L`, `_differential_equations` | rows |
| GEO-2 | §2.3 | `simulator.py` `WellProperties.A` | rows |
