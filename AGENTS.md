# AGENTS.md

ManyWells is a steady-state drift-flux simulator for multiphase (gas + liquid) flow in oil and gas wells. Three-phase flow (gas, oil, water) is handled by treating oil and water as one mixed liquid phase.

`README.md` covers installation, the published datasets and how to cite the paper. This file covers what you need to change the code safely. Longer explanations live in `docs/` and are pointed to below where they apply.

## Environment and commands

The environment is defined by `pyproject.toml` + `uv.lock`. Use [uv](https://docs.astral.sh/uv/):

```console
uv sync                                              # install the environment (the dev dependency group adds pytest)
uv run pytest                                        # full test suite
uv run pytest -m "not slow"                          # skip the slow full-solve tests
uv run pytest tests/test_simulator.py::test_name     # one test
uv run python scripts/sim_examples/vertical_well.py  # run an example
```

The layout is `src/`-based. Scripts under `scripts/` import both `manywells.*` and `scripts.*`, so run them from the project root.

## Layout

`src/manywells/simulator.py` holds `SSDFSimulator` and its two inputs, `WellProperties` (well, fluid and physics models) and `BoundaryConditions` (the operating point). Everything else in the package is a pluggable model the simulator composes, each a dataclass or ABC whose methods work on CasADi symbols as well as floats:

- `geometry.py`: discretization and trajectory (MD/TVD, inclination).
- `pvt/`: fluid properties. `fluid.py` is the one interface the simulator calls; the other modules hold phase correlations.
- `slip.py`: drift-flux closure and flow-regime classification.
- `inflow.py`, `choke.py`: the bottom and top boundary models.
- `friction.py`, `ca_functions.py`, `units.py`: friction factor, smooth approximations for CasADi, constants and unit conversions.
- `calibration/`: fitting model parameters to data. `closed_loop/`: a simulator subclass for closed-loop control.

`scripts/` is research code, not library API. `sim_examples/` is the best reference for setting up and running a simulation; `data_generation/` produces the published datasets and is the slow integration path. Tests in `tests/` follow the module layout.

## How a simulation runs

The pipe is discretized into `n_cells` cells, and each grid point carries the seven state variables listed under Contracts. `simulate()` assembles inflow equations at the bottom, discretized momentum and energy equations for each cell, the choke equation at the top and closure relations at every point into one CasADi system, solved as a feasibility NLP with Ipopt. A cell-by-cell Newton march supplies the initial guess. Failures raise `SimError`; `solution_as_df` turns the solution into a DataFrame with a flow regime per grid point.

## Contracts to preserve

- **State-vector order.** Each grid point holds `[p, v_g, v_l, alpha, rho_g, rho_l, T]` (pressure, gas and liquid velocity, void fraction, gas and liquid density, temperature). Variable creation, bounds and `solution_as_df` depend on this order; change it everywhere or not at all.
- **Units.** Method arguments and returns are in bar and Kelvin. `CF_BAR` converts to Pa where SI is needed internally.
- **CasADi-symbolic friendly.** Physics methods run both on symbols (building the NLP) and on floats. Do not branch in Python on a value that may be symbolic; use the smooth approximations in `ca_functions.py`.
- **Input validation** lives in the dataclasses' `__post_init__`, not in the solver.
- **License header.** New source files carry the same CC BY-NC 4.0 copyright header as the existing files. The license applies to code and datasets.

## Done means

- The relevant tests pass. `uv run pytest -m "not slow"` is fine while iterating; run the full suite before finishing changes to the solver or the physics, since only the slow tests run a full solve.
- New physics or a new model comes with a test.
- The examples in `scripts/sim_examples/` still run if you changed the public API.
- Commits use a short imperative subject line, matching the existing history.

## Where to look for more

- `docs/thermal_energy_modeling.md` when changing the energy equation; it derives the temperature terms and cites the sources.
- `docs/testing.md` for the test layout and pytest configuration.
- `docs/datasets.md` when touching data generation, for the dataset feature definitions and the relations between them.
- `docs/simulate.md` and `scripts/sim_examples/` for how a simulation is set up end to end.
- The ManyWells paper (cited in `README.md`) documents release v1.0.0, which also generated the published HuggingFace datasets (`solution-seeker-as/manywells`). The code has changed since, so treat the paper as background rather than the spec for current behaviour; `git log v1.0.0..HEAD` shows what moved, and `docs/corrigendum.md` lists known typos in the paper.
