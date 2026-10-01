# Testing

The project uses [pytest](https://pytest.org/) for testing. Tests live in the `tests/` directory.

## Setup

Install the environment. pytest comes from the `dev` dependency group, which `uv sync` installs by default:

```bash
uv sync
```

Or with pip (25.1 or newer, for dependency-group support):

```bash
pip install -e . --group dev
```

## Running tests

Run all tests:

```bash
uv run pytest tests/ -v
```

Skip slow tests (e.g. full simulator solve):

```bash
uv run pytest tests/ -v -m "not slow"
```

## Test layout

| File | Coverage |
|------|----------|
| **test_pvt.py** | Reference conditions, fluids, `specific_gas_constant`, `gas_density`, `liquid_mix`, `water_liquid_ratio`, API/density conversions, `dead_oil_surface_tension` |
| **test_ca_functions.py** | `ca_max_approx`, `ca_min_approx`, `ca_softmax`, `ca_sigmoid`, `ca_double_sigmoid` |
| **test_choke.py** | `ChokeModel` (critical pressure ratio, choke openings, invalid profile/K_c), `BernoulliChokeModel`, `SimpsonChokeModel`, `is_choked` |
| **test_inflow.py** | `ProductivityIndex`, `Vogel`, `FixedFlowRate` |
| **test_slip.py** | `classify_flow_regime`, `SlipModel` (Harmathy, Taylor, `identify_parameters`, `slip_equation`, `flow_regime`) |
| **test_simulator.py** | `WellProperties`, `BoundaryConditions` (frozen, validated), `SSDFSimulator` (construction, row order, the deprecated two-argument form, `solution_as_df`); `slow`: full solves, root sets, `NoOperatingPoint` |
| **test_thermal.py** | `ThermalModel`: heat loss, frictional heating, gravity term, ambient profile, inflow temperature |
| **test_configurations.py** | `manywells.configurations`: the `v1.0.0` configuration and its check |
| **test_roots.py** | The root search's copies of the verifier's state distance and thresholds, admissibility (SOL-1), labels, the operating point (SOL-4 to SOL-6), the starts |
| **test_model_properties.py** | Property and spot checks of develop's full model on deviated, L-shaped, black-oil and cold-lift-gas wells, with both backends: Invariants, inflow and choke rows, mass conservation, friction, heat flow, convergence, two-root stability, and the same roots from both (`slow`) |
| **test_rust_backend.py** | The Rust core as the simulator's backend on v1.0.0's wells: their roots, the CasADi rows at the core's roots, what the core refuses, the void-fraction bracket; `slow`: the same roots as the CasADi backend, the slope's sign on the case set |
| **test_backend_comparison.py** | The Rust core against the CasADi backend in every configuration of the matrix (`backend_cases.py`): rows at the same state, and the core's march zeroing the CasADi rows; `slow`: the root sets on the comparison set |
| **test_sampling.py** | The ported sampler: seeding, draw ranges and distributions, the map to each configuration, operating-point draws, the non-stationary walk, dataset rows; `slow`: one well's generation |
| **test_calibration.py** | `calibrate_bernoulli_choke_model`, `calibrate_inflow_model` (PI and Vogel), and error cases |
| **test_spec_vectors.py** | Both backends against the test vectors in `specs/model/` (component tables of v1.0.0 and develop, and v1.0.0's residual rows in the `v1.0.0` configuration); `spec_parse.py` reads the spec files |
| **test_spec_traceability.py** | Equation IDs, coverage tables and `# spec:` tags (`specs/model/README.md`) |
| **test_examples.py** | Every script in `scripts/sim_examples/` runs to the end headless (`slow`) |

`verification/tests/` tests the verifier (`specs/verification.md`); `uv run pytest` runs it with the rest.

The tests that run a full simulator solve, or the examples, are marked `slow` so they can be skipped for faster feedback with `-m "not slow"`.

## Configuration

Pytest is configured in `pyproject.toml` under `[tool.pytest.ini_options]`:

- **testpaths**: `["tests"]`
- **pythonpath**: `["src"]` so the `manywells` package is importable
- **markers**: `slow` — marks tests as slow (deselect with `-m "not slow"`)
