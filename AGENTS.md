# AGENTS.md

ManyWells is a steady-state drift-flux simulator for multiphase (gas + liquid) flow in oil and gas wells. Three-phase flow (gas, oil, water) is handled by treating oil and water as one mixed liquid phase.

`README.md` covers installation, the published datasets and how to cite the paper. This file covers what you need to change the code safely. `specs/constitution.md` holds the project's rules (purpose, non-goals, principles, conventions); read it before your first change. Longer explanations live in `docs/` and are pointed to below where they apply. `CLAUDE.md` imports this file (`@AGENTS.md`), so Claude Code reads the same instructions.

## Environment and commands

The environment is defined by `pyproject.toml` + `uv.lock`. Use [uv](https://docs.astral.sh/uv/). The package is built by maturin, which compiles the Rust core in `rust/` as the extension `manywells._core`, so `uv sync` needs a Rust toolchain (stable, at least 1.85, from [rustup](https://rustup.rs)). uv rebuilds the extension on the next `uv sync` or `uv run` after a file in `rust/src/` changes.

```console
uv sync                                              # install the environment (the dev group adds pytest and the verifier)
uv run pytest                                        # full test suite: tests/ and verification/tests/
uv run pytest -m "not slow"                          # skip the slow tests (full solves, the examples)
uv run pytest tests/test_simulator.py::test_name     # one test
uv run python -m scripts.sim_examples.vertical_well  # run an example
cargo test --manifest-path rust/Cargo.toml --no-default-features   # the Rust core's own tests, without Python
```

Run the verifier from the project root (see The verifier below):

```console
uv run manywells-verify verification/data/v1_cold.parquet --data verification/data --expected-failures verification/expected_failures.csv
```

The layout is `src/`-based. Scripts under `scripts/` import both `manywells.*` and `scripts.*`, so run them as modules from the project root: `uv run python -m scripts.<folder>.<name>`. Running a script by its path (`python scripts/sim_examples/gl_temp.py`) puts the script's folder on `sys.path` instead of the root, and `import scripts.*` fails.

## Layout

The modules follow `specs/architecture.md`. `src/manywells/simulator.py` holds `SSDFSimulator`, its inputs `WellProperties` (one component object per model part) and `BoundaryConditions` (the operating point), `SimError` and `NoOperatingPoint`. The components are frozen dataclasses or ABCs whose methods work on CasADi symbols as well as floats, and import only `units`, `ca_functions` and, within `pvt/`, each other:

- `geometry.py`: the trajectory and grid (MD/TVD, inclination, TVD fraction).
- `pvt/`: fluid properties. `fluid.py` (`FluidModel`) is the one interface the rest calls, including the phase rates with dissolved gas; the other modules hold phase correlations.
- `slip.py`: drift-flux closure and flow-regime classification.
- `friction.py`, `thermal.py`: the friction model (fixed `f_D` or from roughness) and the thermal model (heat loss, frictional heating, gravity term, Joule–Thomson cooling, inflow temperature).
- `inflow.py`, `choke.py`: the bottom and top boundary models.
- `ca_functions.py`, `units.py`: smooth approximations for CasADi, constants and unit conversions.

On top of them:

- `discretization.py`: `build_system(wp)`, the rows of every point (DISC-11), built once per well with the operating point as parameters.
- `solvers/`: the Ipopt adapter, the initial-guess march, and the multi-start root search with the stability label. `solution.py`: `Root`, `RootSet` and the operating point (SOL-4 to SOL-6).
- `configurations.py`: the `v1.0.0` configuration (`v1_well`) and the check that a well is in a configuration.
- `sampling/`, `datasets/`: the ported dataset sampler (`specs/sampling.md`) and the rows and files of a dataset.
- `calibration/`: fitting model parameters to data. Closed loop is out of v2; `closed_loop/` was retired (it remains in v1.0.0).

`rust/` is the Rust core (crate `manywells-core`, built as `manywells._core`): the same model parts, one module per spec file, with `// spec:` tags, the rows of each point defined once in `discretization.rs`, and a shooting search on the bottomhole pressure (`march.rs`, `shoot.rs`). It implements every option of `develop`'s model (`specs/features/014-rust-solver.md` for the `v1.0.0` configuration, `specs/features/015-rust-develop-model.md` for the rest). Both backends are kept and developed together, so a new equation goes into both, under the same spec ID (`specs/goals.md`), and its option goes into the comparison of the two (`tests/backend_cases.py`). `solvers/rust.py` converts a well to its inputs and builds the `RootSet` from its roots; the bindings in `lib.rs` sit behind the crate's `python` feature.

`scripts/` is research code, not library API. `sim_examples/` is the best reference for setting up and running a simulation; `data_generation/` holds the dataset generators, thin callers of `manywells.sampling`; `verification/` the scripts that run `develop` for the verifier. Tests in `tests/` follow the module layout.

`specs/` holds the decided specifications: goals, the constitution, the model, sampling, verification, the architecture and the feature specs (`specs/features/`). `verification/` is the verifier, a separate package. `plans/` holds the plan of work and the backlog; plans are drafts, not specs.

## How a simulation runs

The pipe is discretized into `n_cells` cells, and each grid point carries the seven state variables listed under Contracts. `SSDFSimulator(wp)` builds the well's system once (`discretization.build_system`): inflow rows at the bottom, mass, momentum and energy rows for each cell, the choke row at the top and closure relations at every point, with the boundary conditions as parameters, and an Ipopt feasibility NLP on it. `simulate(bc)` runs the root search (`solvers/roots.py`): marches from several bottomhole pressures (Newton per point, Ipopt where Newton fails) give starts for Ipopt, the solutions are merged into roots, and each root is labelled stable or unstable from the residual's Jacobian. It returns the operating point, the stable root, as a `Root`, and raises `NoOperatingPoint` (a `SimError`) if there is none; `root_set(bc)` returns every root found. `solution_as_df` turns a root into a DataFrame with a flow regime per grid point. The two-argument constructor `SSDFSimulator(wp, bc)` with `simulate()` still works, with a `DeprecationWarning`.

`SSDFSimulator(wp, backend='rust')` solves the same model with the Rust core instead, with the same inputs and outputs; a well with a component class the core does not have, such as a user's subclass of `InflowModel`, raises `ValueError`. It builds no CasADi system: given $p_0$ it marches the rows from the bottomhole to the wellhead, so the choke row is the only one left, and it scans that residual over $(p_s, p_r)$ for its roots. Where the energy row depends on the pressure, each cell's temperature is solved at each trial pressure. It is about 10 to 30 times faster per case than the CasADi backend (median and total, on one process).

## Contracts to preserve

- **State-vector order.** Each grid point holds `[p, v_g, v_l, alpha, rho_g, rho_l, T]` (pressure, gas and liquid velocity, void fraction, gas and liquid density, temperature). Variable creation, bounds and `solution_as_df` depend on this order; change it everywhere or not at all.
- **Units.** Method arguments and returns are in bar and Kelvin. `CF_BAR` converts to Pa where SI is needed internally.
- **CasADi-symbolic friendly.** Physics methods run both on symbols (building the NLP) and on floats. Do not branch in Python on a value that may be symbolic; use the smooth approximations in `ca_functions.py`.
- **Input validation** lives in the dataclasses' `__post_init__`, not in the solver.
- **License header.** New source files carry the same CC BY-NC 4.0 copyright header as the existing files. The license applies to code and datasets.

## The verifier

`verification/` is the package `manywells-verify`, a uv workspace member that `uv sync` installs. It checks a candidate's roots, case by case, against reference root sets computed from ManyWells v1.0.0. It holds no model: it imports neither `manywells` nor CasADi, and must stay that way. `specs/verification.md` defines the cases, the checks and the tolerances.

The command under Environment and commands checks v1.0.0's own solutions, the baseline, as the CI job `verify` does. Its report must say `Verdict: PASS` with 0 unexpected failures, 35 expected failures, 141 cases and a stable-root rate of 74.3%; it exits with status 1 on any unexpected failure. Always pass `--expected-failures`: without it, v1.0.0's 35 known defects count as failures and the verdict is FAIL. `--json OUT` writes every check of every root, and `--name` names the candidate in the report.

- **Candidates.** A candidate is a parquet file with one row per root: `case_id`, `root`, `x` (the 7(N + 1) state values, point by point from the bottomhole, in state-vector order), `label` (`stable`, `unstable` or empty), `operating_point` and `choked` (nullable). Read the cases with `manywells_verify.cases.read_cases('verification/data/cases.parquet')` and write the roots with `write_roots({case_id: [Root(x, label, operating_point, choked)], ...}, path)`. Every `case_id` must be in the case set; a case with no rows means the candidate found no root. The cases are v1.0.0 wells (`L`, `D`, `rho_l`, ...).
- **`develop`'s candidate.** `scripts/verification/develop_candidate.py` maps each case to `develop` in the `v1.0.0` configuration (`manywells.configurations.v1_well`) and writes every root it finds, about 40 s on 24 cores. The CI job `verify` runs it and the verifier, with no expected failures: the report must say `Verdict: PASS` with 0 failures and a stable-root rate of 100%.

  ```console
  uv run python -m scripts.verification.develop_candidate develop.parquet
  uv run manywells-verify develop.parquet --data verification/data --name "develop (v1.0.0 configuration)"
  ```

  With `--backend rust` it writes the Rust core's roots instead, in about 10 s on one core. The CI job runs that too, and the report must also say `Verdict: PASS` with 0 failures and a stable-root rate of 100%; its one finding, a second root at `fold-1503` that the reference lacks, is known (`specs/verification.md`).

  ```console
  uv run python -m scripts.verification.develop_candidate rust.parquet --backend rust
  uv run manywells-verify rust.parquet --data verification/data --name "Rust core (v1.0.0 configuration)"
  ```
- **Distributions.** `uv run manywells-verify-distributions ROWS --dataset sol-1` (or `nsol-1`) compares a regenerated dataset (parquet or CSV with the datasets' feature columns and the published wells' `ID`) with the stable-root reference of the published one, weighting the candidate's rows so that its wells mix as the reference's do. `scripts/verification/regenerate_distributions.py` regenerates samples at the stable root for the published `sol-1` wells and runs the check; at 5 samples per well it takes about 17 minutes on 24 cores.
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

The v1.0.0 environment is a git worktree of the tag in `.worktrees/v1.0.0`, set up as `verification/build/README.md` describes. The Rust environment is a worktree of the old port, `rust_implementation` at `0e9e98b`, in `.worktrees/rust`: the reference's second search uses that port, never the Rust core in `rust/`, so that the reference stays independent of the core it checks. Run its scripts from the project root with `.worktrees/v1.0.0/.venv/bin/python`, never with `uv run`, because `develop`'s package has the same name.

## Adding an equation

A new equation is a model option that is off in the `v1.0.0` configuration. These are the steps feature 016 (`specs/features/016-joule-thomson.md`) took, in order:

1. **Feature spec first.** Draft `specs/features/NNN-<name>.md`: motivation, the delta to `specs/model/` with new IDs (the next free number in each namespace), how the option is off in `v1.0.0`, the acceptance checks, and what is out of scope. Check the option's validity range against the ranges the sampler draws (`specs/sampling.md`): a correlation used outside its range is a finding. Put the scripts behind its measurements in `plans/evidence/`. Bjarne approves the spec before implementation starts.
2. **A branch in a worktree.** Other sessions share the `develop` checkout, so work in `git worktree add -b <branch> .worktrees/<name> develop` and run `uv sync` there.
3. **The spec files.** Define each ID by a `###` heading in its namespace file, and update the file's interface, options, safeguards, sources and coverage rows. Update DISC-11's row IDs and `specs/model/README.md`'s configuration table if a row or a default changes, `nomenclature.md` for new symbols and code names, and `docs/` where it derives the term.
4. **Python.** A field on the component's dataclass (a switch, or a model name validated in `__post_init__`), with `# spec:` tags. Functions take CasADi symbols: an iterative solve is unrolled with a fixed number of steps. `configurations.py` turns the option off in `v1_well` and checks it for `develop`; `discretization.row_ids` gives a changed row's ID.
5. **Rust.** The same option as an enum variant or a field, with `// spec:` tags, passed by `solvers/rust.py` and `Well::new` in `lib.rs`, and the component functions the vectors call added to `_component`. Add test wells with the option on to `input.rs`'s `all()`: the march's assumption tests run on every test well. A new energy term must keep the temperature solve's bracket valid (`specs/architecture.md`, Rust core, design point 4).
6. **Vectors.** A table per new ID in `specs/tools/make_develop_vectors.py`, then regenerate. Where a default changes, pin the old option in its own tables, so that their values do not change. Add the Rust adapters to `RUST_ADAPTERS` in `tests/test_spec_vectors.py`.
7. **Tests.** Unit and property tests in `tests/test_<module>.py`. Tests that assumed the old default pin it. In `tests/backend_cases.py`: the feature in `FEATURES` and `features()`, overlays that switch it on and off, matrix entries, and comparison-set groups. Properties of solved wells go in `tests/test_model_properties.py`.
8. **Run and measure.** `cargo test --no-default-features`, the full suite, both verifier candidates (PASS, 100%, no expected failures), and `scripts/verification/compare_backends.py` on the new groups. Measure the option's effect on sampled wells with each backend, with the option on and off: a fall in the number of operating points found is a solver finding, as it was for feature 016 in gas wells. State the cost and gain of any solver machinery, timed on one process (principle 7).

## Done means

- The relevant tests pass. `uv run pytest -m "not slow"` is fine while iterating; run the full suite before finishing changes to the solver, the physics or the public API, since only the slow tests run a full solve and the examples.
- New physics or a new model comes with a test, and with its spec change in `specs/model/` in the same commit: equation IDs, `# spec:` tags in the code and test vectors (`specs/model/README.md`; a new option's vectors come from `specs/tools/make_develop_vectors.py`). `tests/test_spec_traceability.py` and `tests/test_spec_vectors.py` pass.
- A change to the model keeps the `v1.0.0` configuration's rows exactly v1.0.0's (the row vectors) and its verifier report at PASS with no expected failures; a new option is off in that configuration.
- The examples in `scripts/sim_examples/` still run if you changed the public API. `tests/test_examples.py` (slow) runs each one headless.
- A change to the Rust core keeps `cargo test --no-default-features`, the Rust row and component vectors in `tests/test_spec_vectors.py`, `tests/test_rust_backend.py`, `tests/test_backend_comparison.py` (the core against the CasADi backend in every configuration; its slow test runs the comparison set) and the Rust candidate's verifier report (PASS, 100%, no expected failures) passing. Physics in `rust/` carries `// spec:` tags, as in Python. `uv run python -m scripts.verification.compare_backends OUT.json` compares the backends on more wells and times them.
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
- `specs/architecture.md` before moving code between modules, adding a module or a model option, or changing an interface: the module layout (implemented in Step 7 of the plan), the interface contracts with units, the extension points, the Rust core's design, and the module each planned v2 feature belongs to.
- `docs/thermal_energy_modeling.md` when changing the energy equation; it derives the temperature terms and cites the sources.
- `docs/testing.md` for the test layout and pytest configuration.
- `docs/datasets.md` when touching data generation, for the dataset feature definitions and the relations between them.
- `docs/simulate.md` and `scripts/sim_examples/` for how a simulation is set up end to end.
- `plans/manywells-v2-plan.md` for the current plan of work (the v2 foundation) and `plans/improvements.md` for the backlog of code-level findings, each tagged with the plan step that handles it.
- The ManyWells paper (cited in `README.md`) documents release v1.0.0, which also generated the published HuggingFace datasets (`solution-seeker-as/manywells`). The code has changed since, so treat the paper as background rather than the spec for current behaviour; `git log v1.0.0..HEAD` shows what moved, and `docs/corrigendum.md` lists known typos in the paper.
