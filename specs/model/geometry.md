# Geometry

## Purpose

The shape of the flow path: the well trajectory and the pipe's cross-section. A well has one flow path from bottomhole to wellhead (`specs/goals.md`, non-goals).

## Interface

| Input | Unit | Meaning |
|---|---|---|
| $L$ | m | pipe length (GEO-1) |
| $(\text{MD}_ k, \text{TVD}_ k)$ | m | survey stations, from the surface (GEO-3) |
| $N$ | – | number of cells (GEO-4) |
| $D$ | m | inner diameter, $> 0$ |

Outputs, at the grid points $i = 0, \dots, N$ from the bottomhole and the cells $i = 1, \dots, N$ (cell $i$ between points $i - 1$ and $i$): the measured and true vertical depths $\text{MD}_ i$ and $\text{TVD}_ i$, the cell lengths $\Delta\text{MD}_ i$, the inclinations $\cos\theta_i$, the TVD fractions $f_i$, and the area $A$ (m²). Geometry is fixed per well, so these are floats, never CasADi symbols; the march's point function takes a cell's $\Delta\text{MD}_ i$, $\cos\theta_i$ and $f_i$ as inputs (`discretization.md`, DISC-11).

## Equations

### GEO-1 · Vertical pipe

The pipe is vertical, of length $L > 0$, from the bottomhole at $z = 0$ to the wellhead at $z = L$. The true vertical depth of the bottomhole equals $L$, and gravity acts along the full length of the pipe (BAL-6). Used by `v1.0.0`.

### GEO-2 · Circular cross-section

The pipe has a constant inner diameter $D > 0$ and cross-sectional area

$$A = \pi (D/2)^2.$$

### GEO-3 · Survey trajectory

A trajectory is given by stations $(\text{MD}_ k, \text{TVD}_ k)$, $k = 0, \dots, N$, from the surface: $\text{MD}_ 0 = \text{TVD}_ 0 = 0$, $\text{MD}$ strictly increasing, and $\text{MD}_ k \ge \text{TVD}_ k$. The stations are the grid points, numbered from the bottomhole: grid point $i$ is station $N - i$. For cell $i$, between grid points $i - 1$ and $i$,

$$\Delta\text{MD}_i = \text{MD}_{i-1} - \text{MD}_i, \qquad \cos\theta_i = \frac{\text{TVD}_{i-1} - \text{TVD}_i}{\Delta\text{MD}_i} \in [0, 1],$$

where $\theta_i$ is the inclination from vertical, and at grid point $i$ the TVD fraction is $f_i = \text{TVD}_ i/\text{TVD}_ 0$ (1 at the bottomhole, 0 at the wellhead). The cells need not have the same length. The flow path runs along $\text{MD}$; gravity acts along $\text{TVD}$ (DISC-9), and the ambient temperature follows $\text{TVD}$ (THM-4). The trajectory never descends towards the wellhead ($\cos\theta \ge 0$).

### GEO-4 · Uniform grid from a survey

$N$ cells of equal length $\text{MD}_ \text{end}/N$ in measured depth, with $\text{TVD}$ at each grid point by linear interpolation of a sparse survey in $\text{MD}$, then GEO-3. GEO-1 is GEO-4 with the survey $(0, 0), (L, L)$: $\Delta\text{MD}_ i = L/N$, $\cos\theta_i = 1$ and $f_i = 1 - i/N$, to rounding.

## Options

| Option | IDs | Used by |
|---|---|---|
| Vertical pipe | GEO-1 | `v1.0.0`; `develop`'s `WellGeometry.vertical` |
| Any trajectory from an (MD, TVD) survey, such as deviated and L-shaped wells | GEO-3, GEO-4 | `develop` |

GEO-2 applies to every option. The `v1.0.0` configuration requires a vertical, uniform grid (`configurations.check`).

## Safeguards

None in the equations. v1.0.0 asserts $L > 0$ and $D > 0$. `develop`'s `WellGeometry` raises `ValueError` unless the survey has at least two stations, starts at $(0, 0)$, has strictly increasing MD and $\text{MD} \ge \text{TVD}$, every $\cos\theta_i$ is in $[0, 1]$, and $D > 0$.

## Sources

Paper §2 and Fig. 1. Feature spec `specs/features/001-deviated-wells.md`.

## Test vectors

The row vectors (`discretization.md`) use GEO-1 and GEO-2 in every mass-rate and gravity term. GEO-3 and GEO-4 are checked by `tests/test_geometry.py` (vertical, deviated and L-shaped surveys, validation).

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| GEO-1 | §2, Fig. 1 | `simulator.py` `WellProperties.L`, `_differential_equations` | rows |
| GEO-2 | §2.3 | `simulator.py` `WellProperties.A` | rows |
| GEO-3 | — | — | property: tests/test_geometry.py |
| GEO-4 | — | — | property: tests/test_geometry.py |
