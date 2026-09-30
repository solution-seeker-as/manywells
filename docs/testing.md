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
| **test_simulator.py** | `WellProperties`, `BoundaryConditions`, `SSDFSimulator` (construction, variables, `solution_as_df`), optional `@pytest.mark.slow` full solve |
| **test_calibration.py** | `calibrate_bernoulli_choke_model`, `calibrate_inflow_model` (PI and Vogel), and error cases |
| **test_spec_vectors.py** | `develop` against the test vectors from v1.0.0 in `specs/model/` (component tables and residual rows); `spec_parse.py` reads the spec files |
| **test_spec_traceability.py** | Equation IDs, coverage tables and `# spec:` tags (`specs/model/README.md`) |
| **test_examples.py** | Every script in `scripts/sim_examples/` runs to the end headless (`slow`) |

`verification/tests/` tests the verifier (`specs/verification.md`); `uv run pytest` runs it with the rest.

The tests that run a full simulator solve, or the examples, are marked `slow` so they can be skipped for faster feedback with `-m "not slow"`.

## Configuration

Pytest is configured in `pyproject.toml` under `[tool.pytest.ini_options]`:

- **testpaths**: `["tests"]`
- **pythonpath**: `["src"]` so the `manywells` package is importable
- **markers**: `slow` — marks tests as slow (deselect with `-m "not slow"`)
