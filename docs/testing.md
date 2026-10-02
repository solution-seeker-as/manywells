# Testing

The project uses [pytest](https://pytest.org/). `tests/` tests the `manywells` package and `verification/tests/`
tests the verifier (`specs/verification.md`). The Rust core in `rust/` has its own tests, which cargo runs.

## Setup

Install the environment from the project root:

```bash
uv sync
```

This installs the `dev` dependency group too, which holds pytest and the verifier (`manywells-verify`, the uv
workspace member in `verification/`). `uv sync` builds the Rust core, so it needs a Rust toolchain (stable, at least
1.85, from [rustup](https://rustup.rs)).

With pip, install the two packages and pytest:

```bash
pip install -e . -e ./verification pytest
```

`pip install --group dev` does not work here, because the verifier is not on PyPI.

## Running tests

```bash
uv run pytest                                        # everything: tests/ and verification/tests/
uv run pytest -m "not slow"                          # skip the slow tests
uv run pytest tests/test_thermal.py                  # one file
uv run pytest tests/test_simulator.py::test_row_order  # one test
cargo test --manifest-path rust/Cargo.toml --no-default-features   # the Rust core, without Python
```

The tests that run a full solve, and those that run the examples, are marked `slow`. Skipping them is fine while
iterating; run the full suite before finishing a change to the solver, the physics or the public API. CI runs all of
them.

`--no-default-features` leaves out the crate's `python` feature, the bindings in `rust/src/lib.rs`, so cargo builds
and tests the core alone.

## Layout

The test files follow the module layout: `test_<module>.py` tests `src/manywells/<module>.py`, and the tests of a
package are named after its modules (`test_pvt.py`, `test_fluid.py` and `test_black_oil.py` for `pvt/`, for
instance). `test_roots.py` tests the root search in `solvers/roots.py` and the operating point in `solution.py`.

These files cut across modules:

| File | What it checks |
|------|----------------|
| `test_spec_vectors.py` | Both backends against the test vectors in `specs/model/`: the component tables of v1.0.0 and `develop`, and v1.0.0's residual rows in the `v1.0.0` configuration. A vector whose equation `develop` does not implement is skipped with the reason; one that `develop` is known not to reproduce is a strict expected failure, so it fails once `develop` does. |
| `test_spec_traceability.py` | Equation IDs, coverage tables and `# spec:` tags (`specs/model/README.md`). `spec_parse.py` reads the spec files for both spec tests. |
| `test_model_properties.py` | `develop`'s full model, which has no reference root sets: invariants, inflow and choke rows, mass conservation, friction, heat flow, convergence and stability, on deviated, L-shaped, black-oil and cold-lift-gas wells, with both backends (`slow`). |
| `test_rust_backend.py` | The Rust core as the simulator's backend on v1.0.0's wells, and what it refuses. |
| `test_backend_comparison.py` | The Rust core against the CasADi backend in every configuration of the matrix in `backend_cases.py`: the rows at the same state, and, in the slow test, the root sets on the comparison set. |
| `test_examples.py` | Every script in `scripts/sim_examples/` runs to the end headless (`slow`). |

`verification/tests/` reads v1.0.0 roots from `verification/tests/data/fixtures.npz`, which
`verification/build/make_test_fixtures.py` writes from v1.0.0 (`verification/build/README.md`). Never edit it by
hand.

## Configuration

Pytest is configured in `pyproject.toml` under `[tool.pytest.ini_options]`:

- **testpaths**: `["tests", "verification/tests"]`
- **pythonpath**: `["src"]`, so the `manywells` package is importable
- **markers**: `slow`, for the tests that run a full solve or the examples (deselect with `-m "not slow"`)

## CI

`.github/workflows/tests.yml` runs on every push to `main` and `develop` and on every pull request. The `test` job
runs `uv run pytest` on Python 3.11 to 3.14. The `verify` job runs the Rust core's cargo tests and then the
verifier on v1.0.0's own solutions and on `develop`'s candidates with each backend (`AGENTS.md`, The verifier).
