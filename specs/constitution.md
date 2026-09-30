# ManyWells constitution

*Step 5 of `plans/manywells-v2-plan.md`. Owner: Bjarne Grimstad. Status: decided, 2026-09-30. Any change to this file needs his sign-off.*

These are the rules every change to ManyWells follows, whether a person or an agent makes it. They are short and meant to stay stable; the files they point to hold the detail. `AGENTS.md` says how to apply them day to day: commands, contracts, what "done" means, and what needs sign-off.

## Purpose

ManyWells is a steady-state drift-flux simulator for gas–liquid flow in oil and gas wells. Oil and water are one mixed liquid phase. It serves two purposes of equal weight:

- **Datasets for machine-learning research.** This needs speed and robustness over thousands of wells.
- **A simulator as a tool:** flow prediction after calibration to a well's data, and sensitivity studies. This needs calibration and fidelity on one well.

Where the two conflict, Bjarne decides case by case. `specs/goals.md` has the goals and the direction for v2.

## Non-goals

- Transient flow, including heading and other dynamic instabilities. The model is steady state.
- Competing with commercial simulators such as OLGA or LedaFlow.
- Closed-loop simulation in v2.
- Compositional or equation-of-state PVT. ManyWells stays with black oil and correlations.
- Anything downstream of the choke: flowlines, risers, manifolds, networks.
- Oil–water slip, and complex completions (annulus flow, multilaterals, several tubing strings).
- Publishing real-well data.

Work towards a non-goal needs Bjarne's approval before it starts.

## Principles

1. **The spec is the source of truth; code is derived.** Every physics expression in the code traces to an equation ID in `specs/model/` through a `# spec: <ID>` tag, and `tests/test_spec_traceability.py` checks the tags. No physics change goes in without a spec change in the same PR.
2. **Anchor to v1.0.0's roots, not to one solver's choice of root.** The v1-compatibility configuration and every port of the solver are checked against reference root sets computed from ManyWells v1.0.0 (`specs/verification.md`). A new model option is off in that configuration, so the reference stays valid, and a change that alters the configuration's roots is a regression. The verifier does not re-implement the model.
3. **Harness before features.** A change merges only when the tests and the verifier pass. A feature is not done until something checks it: test vectors, the verifier, or a property check.
4. **Small specs, versioned with the code, reviewed like code.** One spec per feature (`specs/features/NNN-<name>.md`). The executable part (tests, tolerances, test vectors) is primary; prose is secondary.
5. **Humans decide physics, tolerances and scope. Agents draft, implement and run the loop.** `AGENTS.md` lists the changes that need Bjarne's sign-off. An agent may draft them, and says so when it hands them over.
6. **The model's answer is a root set, not a root.** A case has a set of steady-state roots, each labelled stable or unstable, and its operating point is the stable root (`specs/model/solution.md`). Which root a solver happens to reach is not part of the model.
7. **Transparent before marginally faster.** The code must be fast and robust, and a person must be able to read a solver routine next to its spec and follow it. Solver machinery (a new subroutine, a special-case path, a fallback, a tuning constant) has to pay for itself with a large, measured gain in speed or robustness on the verifier's case set. Prefer the simplest method that passes the verifier. Correctness and robustness are requirements, checked by the verifier and the stable-root rate; speed and readability are traded against each other. A change that adds machinery states its measured gain, and Bjarne decides whether it is large enough.

## Conventions

- **Units.** Equations are in SI units. At interfaces (function arguments and returns, the state vector, test vectors and datasets) pressure is in bar and temperature in kelvin, and `CF_BAR` converts bar to Pa where an equation needs it. `specs/model/nomenclature.md` states the rule.
- **Names.** Names in code follow `specs/model/nomenclature.md`, which follows the paper's nomenclature: `rho_l`, `v_m`, `w_g`, `f_D`, and `alpha` for the void fraction $\alpha_g$. A new quantity gets its symbol and code name there first, in the same PR.
- **Determinism.** Given its inputs and a seed, a run gives the same result every time in the same environment (`uv.lock`). Every random draw comes from a generator created from an explicit seed and passed in, such as `numpy.random.default_rng(seed)`, never from NumPy's global state or an unseeded source.
- **Dataset reproducibility.** Every dataset row carries the inputs needed to re-solve it: well, fluid, boundary conditions, grid and model configuration. A generator records its seeds and the code version it ran. A published dataset is never modified; a fix is a new dataset or a note in `docs/corrigendum.md`.
- **Confidential data.** Real-well data is confidential. Some real-well data for validation resides in the private Hugging Face dataset `solution-seeker-as/manywells-validation-data`. Real-well data never enters the public repo, the datasets, PRs, issues, logs or changelogs. Agents never access it. Bjarne runs the checks that use it, manually or on a private runner, and only aggregate results are made public.
- **License.** Code and datasets are under CC BY-NC 4.0, and every new source file carries the copyright header of the existing ones.

## Where things are decided

- `specs/` is decided: goals, this constitution, the model, sampling and verification. Where code and a spec disagree, one of them is wrong, and Bjarne decides which.
- `docs/` explains and derives, for users and contributors. It is not the spec.
- `plans/` holds the plan of work and the backlogs (`plans/README.md`). Plans are drafts, not specs: where a plan and a spec disagree, the spec holds.
