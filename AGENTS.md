# AGENTS.md

ManyWells is a steady-state drift-flux simulator for multiphase (gas + liquid) flow in oil and gas wells. Three-phase flow (gas, oil, water) is handled by treating oil and water as one mixed liquid phase.

`README.md` covers installation, the published datasets and how to cite the paper. This file covers what you need to change the code safely. `specs/constitution.md` holds the project's rules (purpose, non-goals, principles, conventions); read it before your first change. Longer explanations live in `docs/` and are pointed to below where they apply. `CLAUDE.md` imports this file (`@AGENTS.md`), so Claude Code reads the same instructions.

## Environment and commands

The environment is defined by `pyproject.toml` + `uv.lock`. Use [uv](https://docs.astral.sh/uv/):

```console
uv sync                                              # install the environment (the dev group adds pytest and the verifier)
uv run pytest                                        # full test suite: tests/ and verification/tests/
uv run pytest -m "not slow"                          # skip the slow tests (full solves, the examples)
uv run pytest tests/test_simulator.py::test_name     # one test
uv run python -m scripts.sim_examples.vertical_well  # run an example
```

Run the verifier from the project root (see The verifier below):

```console
uv run manywells-verify verification/data/v1_cold.parquet --data verification/data --expected-failures verification/expected_failures.csv
```

The layout is `src/`-based. Scripts under `scripts/` import both `manywells.*` and `scripts.*`, so run them as modules from the project root: `uv run python -m scripts.<folder>.<name>`. Running a script by its path (`python scripts/sim_examples/gl_temp.py`) puts the script's folder on `sys.path` instead of the root, and `import scripts.*` fails.

## Layout

`src/manywells/simulator.py` holds `SSDFSimulator` and its two inputs, `WellProperties` (well, fluid and physics models) and `BoundaryConditions` (the operating point). Everything else in the package is a pluggable model the simulator composes, each a dataclass or ABC whose methods work on CasADi symbols as well as floats:

- `geometry.py`: discretization and trajectory (MD/TVD, inclination).
- `pvt/`: fluid properties. `fluid.py` is the one interface the simulator calls; the other modules hold phase correlations.
- `slip.py`: drift-flux closure and flow-regime classification.
- `inflow.py`, `choke.py`: the bottom and top boundary models.
- `friction.py`, `ca_functions.py`, `units.py`: friction factor, smooth approximations for CasADi, constants and unit conversions.
- `calibration/`: fitting model parameters to data. `closed_loop/`: a simulator subclass for closed-loop control.

`scripts/` is research code, not library API. `sim_examples/` is the best reference for setting up and running a simulation; `data_generation/` produces the published datasets and is the slow integration path. Tests in `tests/` follow the module layout.

`specs/` holds the decided specifications: goals, the constitution, the model, sampling, verification and the architecture. `verification/` is the verifier, a separate package. `plans/` holds the plan of work and the backlog; plans are drafts, not specs.

## How a simulation runs

The pipe is discretized into `n_cells` cells, and each grid point carries the seven state variables listed under Contracts. `simulate()` assembles inflow equations at the bottom, discretized momentum and energy equations for each cell, the choke equation at the top and closure relations at every point into one CasADi system, solved as a feasibility NLP with Ipopt. A cell-by-cell Newton march supplies the initial guess. Failures raise `SimError`; `solution_as_df` turns the solution into a DataFrame with a flow regime per grid point.

## Contracts to preserve

- **State-vector order.** Each grid point holds `[p, v_g, v_l, alpha, rho_g, rho_l, T]` (pressure, gas and liquid velocity, void fraction, gas and liquid density, temperature). Variable creation, bounds and `solution_as_df` depend on this order; change it everywhere or not at all.
- **Units.** Method arguments and returns are in bar and Kelvin. `CF_BAR` converts to Pa where SI is needed internally.
- **CasADi-symbolic friendly.** Physics methods run both on symbols (building the NLP) and on floats. Do not branch in Python on a value that may be symbolic; use the smooth approximations in `ca_functions.py`.
- **Input validation** lives in the dataclasses' `__post_init__`, not in the solver.
- **License header.** New source files carry the same CC BY-NC 4.0 copyright header as the existing files. The license applies to code and datasets.

## The verifier

`verification/` is the package `manywells-verify`, a uv workspace member that `uv sync` installs. It checks a candidate's roots, case by case, against reference root sets computed from ManyWells v1.0.0. It holds no model: it imports neither `manywells` nor CasADi, and must stay that way. `specs/verification.md` defines the cases, the checks and the tolerances.

The command under Environment and commands checks v1.0.0's own solutions, the baseline, as the CI job `verify` does. Its report must say `Verdict: PASS` with 0 unexpected failures, 35 expected failures, 141 cases and a stable-root rate of 74.3%; it exits with status 1 on any unexpected failure. Always pass `--expected-failures`: without it, v1.0.0's 35 known defects count as failures and the verdict is FAIL. `--json OUT` writes every check of every root, and `--name` names the candidate in the report.

- **Candidates.** A candidate is a parquet file with one row per root: `case_id`, `root`, `x` (the 7(N + 1) state values, point by point from the bottomhole, in state-vector order), `label` (`stable`, `unstable` or empty), `operating_point` and `choked` (nullable). Read the cases with `manywells_verify.cases.read_cases('verification/data/cases.parquet')` and write the roots with `write_roots({case_id: [Root(x, label, operating_point, choked)], ...}, path)`. Every `case_id` must be in the case set; a case with no rows means the candidate found no root. The cases are v1.0.0 wells (`L`, `D`, `rho_l`, ...). Mapping them to `develop`'s `WellGeometry` and `FluidModel` in a v1-compatibility configuration is Step 7 of `plans/manywells-v2-plan.md`; until then there is no `develop` candidate.
- **Distributions.** `uv run manywells-verify-distributions ROWS --dataset sol-1` (or `nsol-1`) compares a regenerated dataset (parquet or CSV with the datasets' feature columns) with the stable-root reference of the published one.
- **Expected failures.** `verification/expected_failures.csv` lists v1.0.0's known defects on the case set: cases where its default guess reaches the trickle root or fails. `verification/build/expected_failures.py` writes it. Never add an entry to make a candidate pass.
- **Tests of the verifier itself:** `uv run pytest verification/tests`.

### Adding a case

Never edit the case set or the reference roots by hand. Every reference root is a v1.0.0 solution found by two independent searches, so a new case goes through the build:

1. Add a selector to `verification/build/make_cases.py` and add its cases in `select`, after the existing selectors. Give it its own generator, `np.random.default_rng(seed_for('<source>'))`, rather than the shared `rng`, so the existing cases keep their draws, and seed every other draw with `seed_for(source, well, draw)`.
2. Rebuild as `verification/build/README.md` describes, then run `uv run python verification/build/expected_failures.py`. The rebuild needs the v1.0.0 and Rust worktrees, the published config files and network access for the setup; its two v1 stages use every core for a few minutes.
3. Check that the new case is settled: a case the two searches disagree on is left out and listed in `verification/data/build/disagreements.md`. Check that no existing case or reference root changed (load the old and new parquet files with pandas; `git diff` only shows that they changed). Update the counts in `specs/verification.md` and in this file.

### Regenerating reference data

All reference data is built from v1.0.0, which is frozen. A rebuild reproduces the committed files unless a build script changed; any difference beyond rounding at the solver's tolerance is a finding.

| Data | Built by | Environment | Instructions |
|---|---|---|---|
| `verification/data/*.parquet`: cases, reference roots, v1.0.0's solutions | the stages in `verification/build/` | v1.0.0, Rust, develop | `verification/build/README.md` |
| `verification/expected_failures.csv` | `verification/build/expected_failures.py` | develop | after those stages |
| `verification/data/distribution_reference.json` | `verification/build/distribution_reference.py --data-dir <dir>` | develop | its docstring; needs the published dataset rows |
| test vectors in `specs/model/` and `specs/model/vectors/v1_rows.json` | `specs/tools/make_v1_vectors.py` | v1.0.0 | `specs/model/README.md`, Test vectors |
| `verification/tests/data/fixtures.npz` | `verification/build/make_test_fixtures.py` | v1.0.0 | `verification/build/README.md` |

The v1.0.0 environment is a git worktree of the tag in `.worktrees/v1.0.0`, set up as `verification/build/README.md` describes. Run its scripts from the project root with `.worktrees/v1.0.0/.venv/bin/python`, never with `uv run`, because `develop`'s package has the same name.

## Done means

- The relevant tests pass. `uv run pytest -m "not slow"` is fine while iterating; run the full suite before finishing changes to the solver, the physics or the public API, since only the slow tests run a full solve and the examples.
- New physics or a new model comes with a test, and with its spec change in `specs/model/` in the same commit: equation IDs, `# spec:` tags in the code and test vectors (`specs/model/README.md`). `tests/test_spec_traceability.py` and `tests/test_spec_vectors.py` pass.
- The examples in `scripts/sim_examples/` still run if you changed the public API. `tests/test_examples.py` (slow) runs each one headless.
- A change to `verification/` keeps `uv run pytest verification/tests` passing and the verifier's baseline report unchanged, unless changing it was the point.
- If the change adds a solver routine or path (a subroutine, a special-case path, a fallback, a tuning constant), state its measured gain in speed or robustness on the verifier's case set (constitution, principle 7). Bjarne decides whether the gain is large enough.
- Your summary or PR description names every change that needs sign-off (next section).
- Commits use a short imperative subject line, matching the existing history.

## Needs Bjarne's sign-off

Draft these when the task calls for it, but say that they need sign-off and don't present them as settled:

- Any change to what `specs/model/`, `specs/sampling.md`, `specs/goals.md` or `specs/constitution.md` says: physics, sampling, scope or rules.
- Any tolerance or bound, and any change to what a verifier check tests (`specs/verification.md`, `verification/src/manywells_verify/`).
- The case set and all reference data (the table above), and every new entry in `verification/expected_failures.csv`.
- Any change to a dataset schema. The published v1 datasets are never modified.
- New solver machinery, with its measured gain.
- A new model feature. Its feature spec (`specs/features/NNN-<name>.md`) is approved before the implementation starts.

## Private real-well data

Real-well data is confidential. Some real-well data for validation resides in the private Hugging Face dataset `solution-seeker-as/manywells-validation-data`. Never access real-well data: don't download, read or ask for it, even where credentials are available, and never put values derived from it in the repo, a PR or a log. Bjarne runs the checks that use it. Everything the tests and the verifier need is synthetic and in this repo.

## Where to look for more

- `specs/constitution.md` for the rules, and `specs/goals.md` for the goals and the direction for v2 (scope, non-goals, API and dataset compatibility).
- `specs/model/` before changing any physics: the model's equations with stable IDs, one file per module (`specs/model/README.md`). Code that implements an equation carries a `# spec: <ID>` tag; `tests/test_spec_traceability.py` checks the tags and `tests/test_spec_vectors.py` checks `develop` against test vectors from v1.0.0. `specs/discrepancies.md` lists where the paper and v1.0.0 differ, and `specs/sampling.md` specifies the dataset sampling.
- `specs/verification.md` for the verifier's case set, checks and tolerances, and `verification/build/README.md` for how its reference data is built.
- `specs/architecture.md` before moving code between modules, adding a module or a model option, or changing an interface: the target module layout (which Step 7 of the plan implements), the interface contracts with units, the extension points, the Rust core's design, and the module each planned v2 feature belongs to. The Layout section above describes the code as it is today.
- `docs/thermal_energy_modeling.md` when changing the energy equation; it derives the temperature terms and cites the sources.
- `docs/testing.md` for the test layout and pytest configuration.
- `docs/datasets.md` when touching data generation, for the dataset feature definitions and the relations between them.
- `docs/simulate.md` and `scripts/sim_examples/` for how a simulation is set up end to end.
- `plans/manywells-v2-plan.md` for the current plan of work (the v2 foundation) and `plans/improvements.md` for the backlog of code-level findings, each tagged with the plan step that handles it.
- The ManyWells paper (cited in `README.md`) documents release v1.0.0, which also generated the published HuggingFace datasets (`solution-seeker-as/manywells`). The code has changed since, so treat the paper as background rather than the spec for current behaviour; `git log v1.0.0..HEAD` shows what moved, and `docs/corrigendum.md` lists known typos in the paper.
