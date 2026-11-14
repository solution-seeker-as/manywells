# AGENTS.md

ManyWells is a steady-state drift-flux simulator for multiphase (gas + liquid) flow in oil and gas wells. Three-phase flow (gas, oil, water) is handled by treating oil and water as one mixed liquid phase.

`README.md` covers installation, the published datasets and how to cite the paper. This file covers what you need to change the code safely. `specs/constitution.md` holds the project's rules (purpose, non-goals, principles, conventions); read it before your first change. Longer explanations live in `docs/` and are pointed to below where they apply. `CLAUDE.md` imports this file (`@AGENTS.md`), so Claude Code reads the same instructions.

## Environment and commands

The environment is defined by `pyproject.toml` + `uv.lock`. Use [uv](https://docs.astral.sh/uv/). The package is built by maturin, which compiles the Rust core in `rust/` as the extension `manywells._core`, so `uv sync` needs a Rust toolchain (stable, at least 1.85, from [rustup](https://rustup.rs)). uv rebuilds the extension on the next `uv sync` or `uv run` after a file in `rust/src/` changes.

```console
uv sync                                              # install the environment (the dev group adds pytest, the verifier and ruff)
uv run pytest                                        # full test suite: tests/ and verification/tests/
uv run pytest -m "not slow"                          # skip the slow tests (full solves, the examples)
uv run pytest tests/test_simulator.py::test_name     # one test
uv run python -m scripts.sim_examples.vertical_well  # run an example
cargo test --manifest-path rust/Cargo.toml --no-default-features   # the Rust core's own tests, without Python
uv run ruff check                                    # lint: pyflakes and pycodestyle's errors
uv run ruff format                                   # format every Python file; CI checks both
uvx pre-commit install                               # run both on the files of every commit
```

Run the verifier from the project root (see The verifier below):

```console
uv run manywells-verify verification/data/v1_cold.parquet --data verification/data --expected-failures verification/expected_failures.csv
```

The layout is `src/`-based. Scripts under `scripts/` import both `manywells.*` and `scripts.*`, so run them as modules from the project root: `uv run python -m scripts.<folder>.<name>`. Running a script by its path (`python scripts/sim_examples/gl_temp.py`) puts the script's folder on `sys.path` instead of the root, and `import scripts.*` fails.

## Layout

`specs/architecture.md` (Layers, Modules) gives each module's contents and the rules on what may import what. In short, under `src/manywells/`:

- `simulator.py`: `SSDFSimulator`, its inputs `WellProperties` (one component object per model part) and `BoundaryConditions` (the operating point), `SimError` and `NoOperatingPoint`.
- The components: `geometry.py`, `pvt/`, `slip.py`, `friction.py`, `thermal.py`, `inflow.py` and `choke.py`, with `ca_functions.py` (smooth approximations) and `units.py`. They are frozen dataclasses or ABCs whose methods work on CasADi symbols as well as floats, and import only `units`, `ca_functions` and, within `pvt/`, each other. `pvt/fluid.py`'s `FluidModel` is the one fluid interface the rest calls; its component `pvt`, a `PVTModel`, gives the hydrocarbon properties (`DeadOil`, `BlackOil`, `PVTTable` or `PVTTableSet`).
- `discretization.py` (the rows of every point, DISC-11), `solvers/` (the Ipopt adapter, the initial-guess march, the root search with the stability label) and `solution.py` (`Root`, `RootSet` and the operating point).
- `configurations.py`: the `v1.0.0` configuration (`v1_well`), the `v2.0.0` configuration, and the check that a well is in a configuration.
- On top of the simulator: `sampling/` (the ported sampler, and the v2 sampler in `sampling/v2/`; `specs/sampling.md`), `datasets/` (a dataset's rows and files) and `calibration/` (`specs/calibration.md`).

`rust/` is the Rust core (crate `manywells-core`, built as `manywells._core`): the same model parts, one module per spec file, with `// spec:` tags, implementing every option of `develop`'s model (`specs/architecture.md`, Rust core). Both backends are kept and developed together, so a new equation goes into both, under the same spec ID (`specs/goals.md`), and its option goes into the comparison of the two (`tests/backend_cases.py`). `solvers/rust.py` converts a well to the core's inputs and builds the `RootSet` from its roots; the bindings in `lib.rs` sit behind the crate's `python` feature.

`scripts/` is research code, not library API: `sim_examples/` is the best reference for setting up and running a simulation, `data_generation/` holds the dataset generators, `verification/` the scripts that run `develop` for the verifier, and `calibration/` the calibration's twin study. Tests in `tests/` follow the module layout (`docs/testing.md`). `specs/` holds the decided specifications, `verification/` the verifier, a separate package, and `plans/` the plans of work and the backlog, which are drafts, not specs.

## How a simulation runs

`SSDFSimulator(wp)` builds the well's system once (`discretization.build_system`): inflow rows at the bottom, mass, momentum and energy rows for each of the `n_cells` cells, the choke row at the top and closure relations at every point, with the boundary conditions as parameters. `simulate(bc)` runs the root search (`solvers/roots.py`): marches from several bottomhole pressures give starts for Ipopt, the solutions are merged into roots, and each root is labelled stable or unstable from the residual's Jacobian. It returns the operating point, the stable root, as a `Root`, and raises `NoOperatingPoint` (a `SimError`) if there is none; `root_set(bc)` returns every root found, and `solution_as_df` turns a root into a DataFrame with a flow regime per grid point. The two-argument constructor `SSDFSimulator(wp, bc)` with `simulate()` still works, with a `DeprecationWarning`.

`SSDFSimulator(wp, backend='rust')` solves the same model with the Rust core, with the same inputs and outputs, and is much faster. It builds no CasADi system: it marches from the bottomhole to the wellhead and searches the bottomhole pressure for the roots of the choke row, the only row left. A well with a component class the core does not have, such as a user's subclass of `InflowModel`, raises `ValueError`.

## Contracts to preserve

- **State-vector order.** Each grid point holds `[p, v_g, v_l, alpha, rho_g, rho_l, T]` (pressure, gas and liquid velocity, void fraction, gas and liquid density, temperature). Variable creation, bounds and `solution_as_df` depend on this order; change it everywhere or not at all.
- **Units.** Method arguments and returns are in bar and Kelvin. `CF_BAR` converts to Pa where SI is needed internally.
- **CasADi-symbolic friendly.** Physics methods run both on symbols (building the NLP) and on floats. Do not branch in Python on a value that may be symbolic; use the smooth approximations in `ca_functions.py`.
- **Input validation** lives in the dataclasses' `__post_init__`, not in the solver.
- **License header.** New source files carry the same CC BY-NC 4.0 copyright header as the existing files. The license applies to code and datasets.

## The verifier

`verification/` is the package `manywells-verify`, a uv workspace member that `uv sync` installs. It checks a candidate's roots, case by case, against reference root sets computed from ManyWells v1.0.0. It holds no model: it imports neither `manywells` nor CasADi, and must stay that way. `specs/verification.md` defines the cases, the checks, the tolerances and the files.

The command under Environment and commands checks v1.0.0's own solutions, the baseline, as the CI job `verify` does. Its report must say `Verdict: PASS` with 0 unexpected failures, 35 expected failures, 141 cases and a stable-root rate of 74.3%; it exits with status 1 on any unexpected failure. Always pass `--expected-failures` for the baseline: without it, v1.0.0's 35 known defects count as failures and the verdict is FAIL. `--json OUT` writes every check of every root, and `--name` names the candidate in the report.

- **Candidates.** A candidate is a parquet file with one row per root (`specs/verification.md`, Files). Read the cases with `manywells_verify.cases.read_cases('verification/data/cases.parquet')` and write the roots with `write_roots({case_id: [Root(x, label, operating_point, choked)], ...}, path)`, where `x` is the state point by point from the bottomhole, in state-vector order. Every `case_id` must be in the case set; a case with no rows means the candidate found no root. The cases are v1.0.0 wells (`L`, `D`, `rho_l`, ...).
- **`develop`'s candidates.** `scripts/verification/develop_candidate.py` maps each case to `develop` in the `v1.0.0` configuration (`manywells.configurations.v1_well`) and writes every root it finds; with `--backend rust` it writes the Rust core's. The CI job `verify` runs both. They are checked without `--expected-failures`, so every failure counts: each report must say `Verdict: PASS` with 0 failures and a stable-root rate of 100%. This is what "both candidates pass" means below. The Rust core's one finding, a second root at `fold-1503` that the reference lacks, is known (`specs/verification.md`).

  ```console
  uv run python -m scripts.verification.develop_candidate develop.parquet
  uv run manywells-verify develop.parquet --data verification/data --name "develop (v1.0.0 configuration)"
  uv run python -m scripts.verification.develop_candidate rust.parquet --backend rust
  uv run manywells-verify rust.parquet --data verification/data --name "Rust core (v1.0.0 configuration)"
  ```
- **Distributions.** `uv run manywells-verify-distributions ROWS --dataset sol-1` (or `nsol-1`) compares a regenerated dataset with the published one (`specs/verification.md`, Distributions). `scripts/verification/regenerate_distributions.py` regenerates samples for the published `sol-1` wells and runs the check; it is slow.
- **Expected failures.** `verification/expected_failures.csv` lists v1.0.0's known defects on the case set: cases where its default guess reaches the trickle root or fails. `verification/build/expected_failures.py` writes it. Never add an entry to make a candidate pass.
- **Cases.** Never edit the case set or the reference roots by hand: a new case goes through the build (`verification/build/README.md`, Adding a case).
- **Tests of the verifier itself:** `uv run pytest verification/tests`.

### Regenerating reference data

All reference data is built from v1.0.0, which is frozen. A rebuild reproduces the committed files unless a build script changed; any difference beyond rounding at the solver's tolerance is a finding.

| Data | Built by | Environment | Instructions |
|---|---|---|---|
| `verification/data/*.parquet`: cases, reference roots, v1.0.0's solutions | the stages in `verification/build/` | v1.0.0, Rust, develop | `verification/build/README.md` |
| `verification/expected_failures.csv` | `verification/build/expected_failures.py` | develop | after those stages |
| `verification/data/distribution_reference.json` | `verification/build/distribution_reference.py --data-dir <dir>` | develop | its docstring; needs the published dataset rows |
| test vectors in `specs/model/` and `specs/model/vectors/v1_rows.json` | `specs/tools/make_v1_vectors.py` | v1.0.0 | `specs/model/README.md`, Test vectors |
| `specs/model/vectors/v2_rows.json`, the `v2.0.0` configuration's row vectors | `specs/tools/make_v2_vectors.py` | develop | `specs/model/README.md`, Test vectors |
| `verification/tests/data/fixtures.npz` | `verification/build/make_test_fixtures.py` | v1.0.0 | `verification/build/README.md` |
| `tests/data/pvt/vazquez_beggs.csv`, `synthetic_eos_oil_360K.csv` | `specs/tools/make_pvt_tables.py` | develop | `tests/data/pvt/README.md` |
| `tests/data/pvt/synthetic_eos_oil.csv`, `synthetic_eos_oil_gor.csv` | `plans/evidence/pvt_table_eos.py` | NeqSim 3.4.0 and Java | its docstring |

The v1.0.0 and Rust environments are worktrees in `.worktrees/`, set up as `verification/build/README.md` describes. Run their scripts from the project root with their own `.venv/bin/python`, never with `uv run`, because `develop`'s package has the same name.

## Adding an equation

A new equation is a model option that is off in the `v1.0.0` configuration. These are the steps feature 016 (`specs/features/016-joule-thomson.md`) took, in order:

1. **Feature spec first.** Draft `specs/features/NNN-<name>.md`: motivation, the delta to `specs/model/` with new IDs (the next free number in each namespace), how the option is off in `v1.0.0`, the acceptance checks, and what is out of scope. Check the option's validity range against the ranges the sampler draws (`specs/sampling.md`): a correlation used outside its range is a finding. Add a row for each of its correlations to `specs/model/validity.md`, with a rule. Put the scripts behind its measurements in `plans/evidence/`. Bjarne approves the spec before implementation starts.
2. **A branch in a worktree.** Other sessions share the `develop` checkout, so work in `git worktree add -b <branch> .worktrees/<name> develop` and run `uv sync` there.
3. **The spec files.** Define each ID by a `###` heading in its namespace file, and update the file's interface, options, safeguards, sources and coverage rows. Update DISC-11's row IDs and `specs/model/README.md`'s configuration table if a row or a default changes, `nomenclature.md` for new symbols and code names, and `docs/` where it derives the term.
4. **Python.** A field on the component's dataclass (a switch, or a model name validated in `__post_init__`), with `# spec:` tags. Functions take CasADi symbols: an iterative solve is unrolled with a fixed number of steps. `configurations.py` turns the option off in `v1_well` and checks it for `develop`; `discretization.row_ids` gives a changed row's ID.
5. **Rust.** The same option as an enum variant or a field, with `// spec:` tags, passed by `solvers/rust.py` and `Well::new` in `lib.rs`, and the component functions the vectors call added to `_component`. Add test wells with the option on to `input.rs`'s `all()`: the march's assumption tests run on every test well. A new energy term must keep the temperature solve's bracket valid (`specs/architecture.md`, Rust core, design point 4).
6. **Vectors.** A table per new ID in `specs/tools/make_develop_vectors.py`, then regenerate. Where a default changes, pin the old option in its own tables, so that their values do not change. Add the Rust adapters to `RUST_ADAPTERS` in `tests/test_spec_vectors.py`.
7. **Tests.** Unit and property tests in `tests/test_<module>.py`. Tests that assumed the old default pin it. In `tests/backend_cases.py`: the feature in `FEATURES` and `features()`, overlays that switch it on and off, matrix entries, and comparison-set groups. Properties of solved wells go in `tests/test_model_properties.py`.
8. **Run and measure.** `cargo test --no-default-features`, the full suite, both verifier candidates (both pass), and `scripts/verification/compare_backends.py` on the new groups. Measure the option's effect on sampled wells with each backend, with the option on and off: a fall in the number of operating points found is a solver finding, as it was for feature 016 in gas wells. State the cost and gain of any solver machinery, timed on one process (principle 7).

## Math in markdown

Math in the markdown files must render the same on GitHub as in VS Code, and GitHub's markdown parser changes some math before MathJax sees it. `docs/markdown_math.md` gives the rules and `tests/test_markdown_math.py` checks them. The most common: no backslash before punctuation (`\thinspace` for `\,`, `\lbrace` for `\{`), `\ast` for `*`, a space after a `_` that follows a bracket in inline math (`(x)_ i`), and `\mathop{\mathrm{...}}` for `\operatorname`.

## Releases

The version (constitution, Versions) is set in `pyproject.toml`, `verification/pyproject.toml` and `rust/Cargo.toml`, where Cargo writes `2.0.0.dev0` as `2.0.0-dev.0`; `tests/test_versions.py` checks that the three agree. Run `uv lock` after changing it. A release raises:

- **MAJOR** for a break of the public API (any name in `manywells` without a leading `_`) or a configuration removed;
- **MINOR** for new API, a model option, a configuration or a changed default, whose change of results goes in the CHANGELOG;
- **PATCH** for anything else.

Between releases, `develop` carries the next version with `.dev0`; the change that first breaks the API raises it to the next major one. Tag a release with an annotated tag, `git tag -a vX.Y.Z`: `datasets.io.code_version` reads only annotated tags.

## Done means

- The relevant tests pass. `uv run pytest -m "not slow"` is fine while iterating; run the full suite before finishing changes to the solver, the physics or the public API, since only the slow tests run a full solve and the examples.
- New physics or a new model comes with a test, and with its spec change in `specs/model/` in the same commit: equation IDs, `# spec:` tags in the code and test vectors (`specs/model/README.md`; a new option's vectors come from `specs/tools/make_develop_vectors.py`). `tests/test_spec_traceability.py` and `tests/test_spec_vectors.py` pass.
- A change to the model keeps the `v1.0.0` configuration's rows exactly v1.0.0's (the row vectors), and both verifier candidates pass; a new option is off in that configuration.
- The examples in `scripts/sim_examples/` still run if you changed the public API. `tests/test_examples.py` (slow) runs each one headless.
- A change to the Rust core keeps `cargo test --no-default-features`, the Rust row and component vectors in `tests/test_spec_vectors.py`, `tests/test_rust_backend.py`, `tests/test_backend_comparison.py` (the core against the CasADi backend in every configuration; its slow test runs the comparison set) and the Rust candidate passing. Physics in `rust/` carries `// spec:` tags, as in Python. `uv run python -m scripts.verification.compare_backends OUT.json` compares the backends on more wells and times them.
- A change to `verification/` keeps `uv run pytest verification/tests` passing and the verifier's baseline report unchanged, unless changing it was the point.
- If the change adds a solver routine or path (a subroutine, a special-case path, a fallback, a tuning constant), state its measured gain in speed or robustness on the verifier's case set (constitution, principle 7). Bjarne decides whether the gain is large enough.
- Your summary or PR description names every change that needs sign-off (next section).
- `uv run ruff check` and `uv run ruff format --check` pass; the CI job `lint` runs both. The pre-commit hook fixes what it can and stops the commit, so stage its changes and commit again.
- Commits use a short imperative subject line, matching the existing history.
- Before pushing, merge `origin/develop` and rerun `uv run pytest -m "not slow"`: work merged meanwhile is what CI has caught that local runs had not. Don't wait for CI after pushing: it runs the full suite on every supported Python version and the verifier, which is slow. Carry on, and check the result (`gh pr checks`) before you finish; if it is still running, say so. Fix a red check with a follow-up commit, and never merge into `develop` or `main` on red.

## Needs Bjarne's sign-off

Draft these when the task calls for it, but say that they need sign-off and don't present them as settled:

- Any change to what `specs/model/`, `specs/sampling.md`, `specs/calibration.md`, `specs/goals.md` or `specs/constitution.md` says: physics, sampling, calibration (its priors and noise defaults included), scope or rules.
- Any tolerance or bound, and any change to what a verifier check tests (`specs/verification.md`, `verification/src/manywells_verify/`).
- The case set and all reference data (the table above), and every new entry in `verification/expected_failures.csv`.
- Any change to a dataset schema. The published v1 datasets are never modified.
- A release and its version number.
- New solver machinery, with its measured gain.
- A new model feature. Its feature spec (`specs/features/NNN-<name>.md`) is approved before the implementation starts.

## Data and sources

The maintainer (`pyproject.toml`, `maintainers`) holds the private data and decides which results from it become public. The constitution states the rules (Conventions, Confidential data and Sources); this section says how to apply them.

**Real-well data.**

- Real-well data is confidential. Some real-well data for validation resides in the private Hugging Face dataset `solution-seeker-as/manywells-validation-data`. It is named here and in the constitution so that you can avoid it.
- Never access real-well data: don't download, read or ask for it, even where credentials are available.
- Never put real-well data, or a value derived from it, in the repo, the datasets, a PR, an issue or a log. Only the maintainer decides to publish a result of a private check, aggregate or qualitative, and the file that gives it says so, as `specs/features/018-pvt-tables.md` does.
- The maintainer runs the checks that use it. Everything the tests and the verifier need is synthetic and in this repo.
- The maintainer may point you to a report of a private check's results. Read it only to frame questions for the maintainer; nothing from it goes into the repo.
- Refer to private material only in general terms, such as "a private comparison with field data" or "the local papers library". Name no field, well, operator, private dataset or file, and give no path, link or identifier.
- A decision that the maintainer bases on a private check is recorded as the maintainer's decision. It may say in general terms that a private check informed it, and gives nothing more from it.

**Open real-well data.**

- Statistics computed from real-well data that a publisher releases openly, such as production by well, well tests or surveys, are kept outside the repo by the maintainer, with their scripts and the publishers' licences (the maintainer's decision, 2026-11-06). No value computed from real-well data, open or confidential, is published in the repo.
- The repo may state what such an analysis shows, in words or as a comparison ("above the draft's range", "within it"), and a design value chosen with it. Cite it by its public source, as `[own analysis of BOEM's OGOR-A production]`. Figures that a publisher publishes itself, such as EIA's counts of wells by rate, are published work: cite them under Sources.

**Sources.**

- A spec's evidence is published work, cited by reference (authors, year, title, venue) with the page, table, figure or equation that a number comes from. The number may be committed with its citation. The documents are copyrighted: never commit, upload or publish them, and name no local path to them.
- Fetch only open-access copies: from the publisher, the authors' pages, an institutional repository or a preprint server, never from a shadow library or a file-sharing site.
- The maintainer obtains paywalled papers and books. Ask for one when it matters to a decision, and say which decision it bears on and what you assume without it.

## Where to look for more

- `specs/goals.md` for the goals and the direction for v2 (scope, non-goals, API and dataset compatibility).
- `specs/model/` before changing any physics: the equations with stable IDs, one file per module, the configurations and the test vectors (`specs/model/README.md`). `specs/discrepancies.md` lists where the paper and v1.0.0 differ.
- `specs/architecture.md` before moving code between modules, adding a module or a model option, or changing an interface.
- `specs/sampling.md` and `docs/datasets.md` when touching data generation; `specs/calibration.md` and `docs/calibration.md` for calibration.
- `docs/simulate.md` for how a simulation is set up end to end, `docs/testing.md` for the tests and CI, and `docs/thermal_energy_modeling.md` when changing the energy equation.
- `plans/` for the plans of work, and `plans/backlog.md` for the backlog of code-level findings.
- The ManyWells paper (cited in `README.md`) documents release v1.0.0, which also generated the published HuggingFace datasets (`solution-seeker-as/manywells`). The code has changed since, so treat it as background rather than the spec for current behaviour; `git log v1.0.0..HEAD` shows what moved, and `docs/corrigendum.md` lists known typos in the paper.
