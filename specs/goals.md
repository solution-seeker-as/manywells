# ManyWells v2 goals

*2026-09-30. Owner: Bjarne Grimstad. Status: decided.*

This file records the goals of the v2 foundation plan (`plans/manywells-v2-plan.md`) and the direction for v2 that the plan prepares for. The v2 direction can be revised after the plan; any change needs Bjarne's sign-off. Terms such as *verifier*, *residual graph* and *v1-compatibility configuration* are defined in the plan.

## Purpose

ManyWells is a steady-state drift-flux simulator for gas–liquid flow in oil and gas wells. v2 serves two purposes of equal weight:

- **Datasets for machine-learning research**, as v1 did. This needs speed and robustness over thousands of wells.
- **A simulator as a tool**: flow prediction after calibration to a well's data, and sensitivity studies. This needs calibration and fidelity on one well.

Neither ranks above the other. Where they conflict, Bjarne decides case by case.

## Goals of the foundation plan

The plan ends when new equations are easy to add and test:

- `develop`'s model is specified in `specs/model/`, with v1.0.0 as a named configuration.
- The verifier checks v1.0.0, the Rust port against v1.0.0's model, `develop` in its v1-compatibility configuration, and `develop`'s full model.
- v1's sampling procedure, extended to `develop`'s new inputs (trajectory, black-oil parameters, pipe roughness), runs on `develop`.
- One new equation has gone through the whole loop (plan, Step 9).

The new flow-regime model and the port of `develop`'s model to Rust come after the plan.

## Direction for v2

- **Model scope.** v2 relaxes the five v1 limitations of paper §8:
  1. Incompressible liquid, no mass transfer: black-oil PVT with dissolved gas, and a real-gas z-factor (on `develop`).
  2. Vertical wells only: deviated and L-shaped wells (on `develop`).
  3. One friction factor for the whole well: friction from pipe roughness with the Chen or Haaland correlation, keeping a fixed `f_D` as an option (on `develop`).
  4. Simplified thermal model: frictional heating, a gravity term in the energy balance, and lift-gas temperature (on `develop`).
  5. Unrealistic sampled wells: a sampling redesign, after the plan.

  The other model changes on `develop` (fixed-rate inflow, inclination terms in the slip model) are in scope too. v2 also adds a flow-regime model with four regimes (bubbly, slug, churn, annular) after Hasan et al.; `develop`'s classifier merges slug and churn. It is a draft (`scripts/flow_regimes/new_flow_regime_model.md`), not yet on `develop`. Any other model change needs its own feature spec and Bjarne's approval. Which features ship in v2.0 and which wait for v2.x is decided later.
- **Solution.** The model's answer is a root set: every steady-state root, each labelled stable or unstable. The label is static (nodal-analysis) stability: at a stable root, a small increase in rate makes the flow fall back; at an unstable one it runs away. The unstable roots in v1's data are low-rate *trickle* roots. The operating point is the stable root; if there is none, the well cannot flow at those conditions. `simulate()` returns the operating point and raises if there is none; the full root set is available on request.
- **Implementation.** A Rust core with Python bindings, which are the public API. `develop`'s Python/CasADi simulator stays the working implementation until the Rust core covers the full model (after this plan, before v2.0.0), and is then retired. CasADi stays in the verifier, for the residual graph and the stability label. From then on each equation exists twice, in the Rust core and in the verifier's residual graph built from the spec, and the verifier checks one against the other. `closed_loop/` subclasses the Python simulator, so it is removed with it and can return in a later version on the Rust core. Calibration must work with the Rust core; its spec decides how.
- **Distribution.** Prebuilt wheels on PyPI, so `pip install manywells` needs no Rust toolchain.
- **Release.** Besides the model and the Rust core, v2.0.0 needs calibration, checked privately against real-well data, and new datasets. There is no closed-loop dataset. Whether there is a new paper is open.

## Compatibility

- **API.** v2.0.0 may break any part of the API, provided the CHANGELOG lists each break with an old→new snippet. The v1-compatibility configuration reproduces v1.0.0's physics, not its API.
- **Datasets.** v2 datasets are new datasets (`manywells-sol-2` and so on, after v1's `-sol-1`, `-nsol-1` and `-nscl-1`) with their own schema. The published v1 datasets are never modified. The v1 erratum, which labels published samples as stable-root or trickle-root, goes in the public dataset `solution-seeker-as/manywells-verification`. A note in `docs/corrigendum.md` and on the `manywells` dataset card points to it. The residual graph lives in this repo, next to the verifier.

## Non-goals

- Transient flow, including heading and other dynamic instabilities. The model is steady state.
- Competing with commercial simulators such as OLGA or LedaFlow.
- Closed-loop simulation in v2.
- Compositional or equation-of-state PVT. v2 stays with black oil and correlations.
- Anything downstream of the choke: flowlines, risers, manifolds, networks.
- Oil–water slip. Oil and water stay one mixed liquid phase.
- Complex completions: annulus flow, multilaterals, multiple tubing strings. A well has one flow path from bottomhole to wellhead.
- Publishing real-well data. It stays in the private `manywells-validation-data`.

## Decided elsewhere

| Decision | Where |
|---|---|
| Tolerances (`tol_r`, `tol_x`, convergence order) | Plan, Step 2 item 5 |
| Operating point when there are two stable roots | `specs/model/solution.md` (Step 4) |
| Rust performance target | Rust feature spec (Step 8) |
| Calibration method with the Rust core | Calibration spec, after the plan |
| v2.0 vs v2.x feature split; v2 dataset schema (every root or the operating point only); accuracy bound for the real-well check; license; wheel platforms | v2.0.0 release work |
