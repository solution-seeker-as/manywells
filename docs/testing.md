# Testing

The project uses [pytest](https://pytest.org/) for testing. Tests live in the `tests/` directory.

## Setup

Install the package with the optional dev dependency so pytest is available:

```bash
uv sync --extra dev
```

Or with pip:

```bash
pip install -e ".[dev]"
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
| **test_inflow.py** | `compute_gas_mass_fraction`, `ProductivityIndex`, `Vogel`, `FixedFlowRate` |
| **test_slip.py** | `classify_flow_regime`, `SlipModel` (Harmathy, Taylor, `identify_parameters`, `slip_equation`, `flow_regime`) |
| **test_simulator.py** | `WellProperties`, `BoundaryConditions`, `SSDFSimulator` (construction, variables, `solution_as_df`), optional `@pytest.mark.slow` full solve |
| **test_calibration.py** | `calibrate_bernoulli_choke_model`, `calibrate_inflow_model` (PI and Vogel), and error cases |

There are **65 tests** in total. The one that runs a small simulator solve is marked `slow` so it can be skipped for faster feedback with `-m "not slow"`.

## Configuration

Pytest is configured in `pyproject.toml` under `[tool.pytest.ini_options]`:

- **testpaths**: `["tests"]`
- **pythonpath**: `["."]` so the `manywells` package is importable
- **markers**: `slow` — marks tests as slow (deselect with `-m "not slow"`)
