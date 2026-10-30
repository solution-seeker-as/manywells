# ManyWells v2 goals

*2026-09-30, revised the same day (lean verification), 2026-10-01 (the port of `develop`'s model to the Rust core moves into the plan; signed off by Bjarne) 2026-10-29 (v2.0's model scope, decided by Bjarne) and 2026-10-30 (the v2 datasets and their wells, decided by Bjarne in the sampler redesign's interview). Owner: Bjarne Grimstad. Status: decided.*

This file records the goals of the v2 foundation plan (`plans/manywells-v2-plan.md`) and the direction for v2 that the plan prepares for. The v2 direction can be revised after the plan; any change needs Bjarne's sign-off. Terms such as *verifier*, *reference root set* and *v1-compatibility configuration* are defined in the plan.

## Purpose

ManyWells is a steady-state drift-flux simulator for gas–liquid flow in oil and gas wells. v2 serves two purposes of equal weight:

- **Datasets for machine-learning research**, as v1 did. This needs speed and robustness over thousands of wells.
- **A simulator as a tool**: flow prediction after calibration to a well's data, and sensitivity studies. This needs calibration and fidelity on one well.

Neither ranks above the other. Where they conflict, Bjarne decides case by case.

## Goals of the foundation plan

The plan ends when new equations are easy to add and test:

- `develop`'s model is specified in `specs/model/`, with v1.0.0 as a named configuration.
- The verifier checks the Rust port and `develop` in its v1-compatibility configuration against reference root sets computed from v1.0.0. `develop`'s full model is checked by component test vectors, invariants, spot checks, convergence and stability property checks, and, from the plan's Step 9 on, by comparing the two backends; it has no reference implementation.
- v1's sampling procedure, extended to `develop`'s new inputs (trajectory, black-oil parameters, pipe roughness), runs on `develop`.
- The Rust core implements `develop`'s model, checked against the CasADi backend (plan, Step 9).
- One new equation has gone through the whole loop, in both backends (plan, Step 10).

The new flow-regime model comes after the plan; it is `specs/features/019-five-flow-regimes.md`.

## Direction for v2

- **Model scope.** v2 relaxes the five v1 limitations of paper §8:
  1. Incompressible liquid, no mass transfer: black-oil PVT with dissolved gas, and a real-gas z-factor (on `develop`).
  2. Vertical wells only: deviated and L-shaped wells (on `develop`).
  3. One friction factor for the whole well: friction from pipe roughness with the Chen or Haaland correlation, keeping a fixed `f_D` as an option (on `develop`).
  4. Simplified thermal model: frictional heating, a gravity term in the energy balance, and lift-gas temperature (on `develop`).
  5. Unrealistic sampled wells: a sampling redesign, after the plan (`plans/sampler-redesign-plan.md`).

  The other model changes on `develop` (fixed-rate inflow, inclination terms in the slip model) are in scope too. v2 also adds a flow-regime model with five regimes (bubbly, dispersed bubbly, slug, churn, annular) after Hasan et al., `develop`'s default (`specs/features/019-five-flow-regimes.md`); v1.0.0's fitted classifier, which merges slug and churn, stays as an option. Any other model change needs its own feature spec and Bjarne's approval.
- **v2.0's model** (Bjarne, 2026-10-29; `plans/model-lock-plan.md`, Step L1). Before the model is locked for the new sampler, v2.0 adds the lift gas's injection depth: today it enters at the bottomhole, where real gas-lift wells inject through a valve above the packer (feature spec 021). The `v2.0.0` configuration uses the five regimes, not the fitted classifier (Step L4). The black oil takes any gas–oil ratio, and a gas condensate whose dropout matters enters as a PVT table (`specs/model/validity.md`, decision 4). Left for v2.x, and documented in v2.0 as limits: a tubing diameter that changes along the well, heat capacities that depend on $(p, T)$, and a heat-transfer coefficient that varies with depth or time. The sampler redesign's interview decided the rest (Bjarne, 2026-10-30; `plans/sampler-redesign-plan.md`, Goals):
  - v2.0 adds a gas-rate inflow for gas wells, a well option that `v2.0.0`'s check accepts;
  - the datasets draw black oils, volatile oils, gas condensates and gas wells, the volatile oils and condensates as PVT tables;
  - they draw no horizontal sections: a well's bottom section is at most highly deviated, short of the inclinations where stratified flow sets in (`specs/model/validity.md`, decision 5).
- **Solution.** The model's answer is a root set: every steady-state root, each labelled stable or unstable. The label is static (nodal-analysis) stability: at a stable root, a small increase in rate makes the flow fall back; at an unstable one it runs away. The unstable roots in v1's data are low-rate *trickle* roots. The operating point is the stable root; if there is none, the well cannot flow at those conditions. `simulate()` returns the operating point and raises if there is none; the full root set is available on request.
- **Implementation.** Two backends behind one Python API, `SSDFSimulator(wp, backend=...)`: `develop`'s Python/CasADi simulator, and a Rust core with Python bindings. The Rust core covers the full model from the plan's Step 9 on.
  - **Developed together.** Both backends are kept and developed together (Bjarne, 2026-10-01). Each equation is implemented in both, under the same spec ID.
  - **Checked against each other.** The two backends are compared on the component vectors, on the rows at the same states, and on the operating point for the same cases. The verifier does not re-implement the model.
  - **`closed_loop/`** was retired on 2026-10-01 (it remains in v1.0.0); closed loop can return in a later version.
  - **Calibration** must work with the Rust core; its spec decides how.
- **Distribution.** Prebuilt wheels on PyPI, so `pip install manywells` needs no Rust toolchain.
- **Release.** Besides the model and the Rust core, v2.0.0 needs calibration, checked privately against real-well data, and new datasets. There is no closed-loop dataset. Whether there is a new paper is open.

## Compatibility

- **API.** v2.0.0 may break any part of the API, provided the CHANGELOG lists each break with an old→new snippet. The v1-compatibility configuration reproduces v1.0.0's physics, not its API.
- **Datasets.** v2 datasets are new datasets, after v1's `-sol-1`, `-nsol-1` and `-nscl-1`, with their own schema. There are two (Bjarne, 2026-10-30):
  - `manywells-sol-2`, stationary, and `manywells-nsol-2`, non-stationary;
  - each has 10,000 wells with 200 samples per well;
  - each has two tables: one of wells, with their inputs and labels, and one of samples, holding each sample's operating point only.

  The published v1 datasets are never modified. The v1 erratum is a note in `docs/corrigendum.md`: the ~280 K TWH spike in the published data comes from samples on the unstable trickle root. No labels marking those samples are published. There is no public verification dataset; the verifier's case set and reference root sets live in this repo.

## Non-goals

- Transient flow, including heading and other dynamic instabilities. The model is steady state.
- Competing with commercial simulators such as OLGA or LedaFlow.
- Closed-loop simulation in v2.
- Compositional PVT: cubic equations of state and flash calculations. v2 stays with black oil, correlations and tables; a fluid from an equation of state enters as a table (feature 018).
- Anything downstream of the choke: flowlines, risers, manifolds, networks.
- Oil–water slip. Oil and water stay one mixed liquid phase.
- Complex completions: annulus flow, multilaterals, multiple tubing strings. A well has one flow path from bottomhole to wellhead.
- Publishing real-well data (`specs/constitution.md`, Confidential data).

## Decided elsewhere

| Decision | Where |
|---|---|
| Tolerances (`tol_r`, `tol_x`, convergence order) | Plan, Step 2 item 5 |
| Operating point when there are two stable roots | `specs/model/solution.md` (Step 4): the one with the lowest p0 (SOL-6) |
| Rust performance target | none yet (Bjarne, 2026-10-02); `specs/features/015-rust-develop-model.md` records the measurements |
| Calibration method with the Rust core | Calibration spec, after the plan |
| v2.0's model scope | this file, Direction for v2 (2026-10-29) |
| v2 datasets: which, their size, two tables, the operating point only | this file, Compatibility (2026-10-30) |
| v2 datasets' columns and filters | the sampler redesign (`plans/sampler-redesign-plan.md`, R7) |
| Accuracy bound for the real-well check; license; wheel platforms | v2.0.0 release work |
