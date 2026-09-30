# Verifier build scripts

These scripts produce the verifier's frozen inputs: the residual graph under `../residual_graph/` and, later, the case set and reference root sets. They run in three separate environments, because `develop`'s package and v1.0.0's are both called `manywells`:

| Environment | Where | Used for |
|---|---|---|
| v1.0.0 | `.worktrees/v1.0.0` (git worktree of the `v1.0.0` tag, Python 3.11, casadi 3.6.4) | building the residual graph; v1 solves |
| Rust | `.worktrees/rust` (worktree of `rust_implementation@0e9e98b`, with `manywells_rs` built into it) | independent roots for the reference root sets |
| develop | the repository root | the verifier itself, Newton refinement, labelling |

The environments exchange data through `.npz` and `.csv` files; the v1.0.0 environment has no pyarrow. `.worktrees/` is gitignored.

## Setup

From the repository root:

```console
git worktree add --detach .worktrees/v1.0.0 v1.0.0
git worktree add --detach .worktrees/rust 0e9e98b
(cd .worktrees/v1.0.0 && uv sync --frozen)
(cd .worktrees/rust && uv sync --frozen && VIRTUAL_ENV=$PWD/.venv uv pip install ./manywells_rs)
```

`--frozen` installs exactly from each worktree's lock file. The Rust build needs a Rust toolchain with edition 2024 (Rust 1.85 or later). In the Claude Code sandbox, the `uv` and `cargo` steps need network access, so they run with the sandbox off.

The scripts need the published configs, `manywells-sol-1_config.zip` and `manywells-nsol-1_config.zip`, from `data/` in the Hugging Face dataset `solution-seeker-as/manywells`. Pass their folder as `--data-dir`.

## Residual graph

```console
.worktrees/v1.0.0/.venv/bin/python verification/build/build_graph.py --data-dir <dir>
```

This writes `residual_graph/v1.0.0/`: one `.casadi` file per block, the generated C source and header, `manifest.json` and `golden.npz`. It exits non-zero unless the stacked blocks reproduce v1.0.0's full residual and Jacobian to a relative 1e-13. That check covers seven sol-1 wells, a gas-lift well, every choke model and profile, and productivity-index inflow at N = 37 and N = 200. It runs at v1's solutions and at perturbed states. `verification/tests/test_residual.py` then checks the files in the develop environment against `golden.npz`, through both CasADi and the compiled C.

Rebuild only when v1.0.0's graph needs a new block or output. The committed graph is the reference, and a rebuild must reproduce `golden.npz`.
