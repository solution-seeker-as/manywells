# Verifier build scripts

These scripts produce the verifier's reference data: the case set and its reference root sets, computed from ManyWells v1.0.0 (`specs/verification.md`). They run in three separate environments, because `develop`'s package and v1.0.0's are both called `manywells`:

| Environment | Where | Used for |
|---|---|---|
| v1.0.0 | `.worktrees/v1.0.0` (git worktree of the `v1.0.0` tag, Python 3.11, casadi 3.6.4) | selecting cases; every reference root is a v1 solution |
| Rust | `.worktrees/rust` (worktree of `rust_implementation@0e9e98b`, with `manywells_rs` built into it); the old port, never the Rust core in `rust/`, so that the reference stays independent of the core it checks | fold brackets; Rust roots as starting points for v1 |
| develop | the repository root | labelling and writing the committed files |

The environments exchange files through `verification/data/build/` (gitignored), using JSON and `.npz`, because the v1.0.0 environment has no pyarrow. The committed output goes to `verification/data/`.

## Setup

From the repository root:

```console
git worktree add --detach .worktrees/v1.0.0 v1.0.0
git worktree add --detach .worktrees/rust 0e9e98b
(cd .worktrees/v1.0.0 && uv sync --frozen)
(cd .worktrees/rust && uv sync --frozen && VIRTUAL_ENV=$PWD/.venv uv pip install ./manywells_rs)
```

`--frozen` installs exactly from each worktree's lock file. The Rust build needs a Rust toolchain with edition 2024 (Rust 1.85 or later). In the Claude Code sandbox, the `uv` and `cargo` steps need network access, so they run with the sandbox off.

The scripts need `manywells-sol-1_config.zip`, `manywells-nsol-1_config.zip` and `manywells-nsol-1.zip` from `data/` in the Hugging Face dataset `solution-seeker-as/manywells`. Pass their folder as `--data-dir`.

## Case set and reference root sets

```console
.worktrees/rust/.venv/bin/python   verification/build/rust_roots.py fold --data-dir <dir>   # fold.csv
.worktrees/v1.0.0/.venv/bin/python verification/build/make_cases.py select --data-dir <dir> # cases.json, starts.npz
.worktrees/rust/.venv/bin/python   verification/build/rust_roots.py roots                   # rust_roots.{json,npz}
.worktrees/v1.0.0/.venv/bin/python verification/build/make_cases.py solve                   # v1_runs.json, v1_arrays.npz, coverage.md
uv run python verification/build/label_cases.py                                             # verification/data/*.parquet, disagreements.md
```

- `rust_roots.py fold` picks two-root sol-1 wells and bisects p_r down to where the two roots merge.
- `make_cases.py select` solves every sol-1 well at its stored operating point to find the trickle-root wells, then picks about 150 cases.
- `rust_roots.py roots` solves every case with Rust.
- `make_cases.py solve` solves every case with v1 from its default guess, the dataset generator's start, cellwise guesses (method A) and each Rust root (method B).
- `label_cases.py` settles each case whose two methods agree and whose labels are clear; the rest go to `data/build/disagreements.md` for Bjarne.

Both v1 stages use one process per core with one thread each and take a few minutes. Every random draw is seeded from the case's source, well and draw number, so a rerun gives the same case set.

`tolerance_stats.py` (develop environment) prints the measurements behind the tolerances. `make_test_fixtures.py` (v1.0.0 environment) writes the small root set behind the unit tests, `verification/tests/data/fixtures.npz`.

## Adding a case

Never edit the case set or the reference roots by hand. Every reference root is a v1.0.0 solution found by two independent searches, so a new case goes through the build:

1. Add a selector to `make_cases.py` and add its cases in `select`, after the existing selectors. Give it its own generator, `np.random.default_rng(seed_for('<source>'))`, rather than the shared `rng`, so the existing cases keep their draws, and seed every other draw with `seed_for(source, well, draw)`.
2. Rebuild as above, then run `uv run python verification/build/expected_failures.py`. The rebuild needs the v1.0.0 and Rust worktrees, the published config files and network access for the setup.
3. Check that the new case is settled: a case the two searches disagree on is left out and listed in `verification/data/build/disagreements.md`. Check that no existing case or reference root changed (load the old and new parquet files with pandas; `git diff` only shows that they changed). Update the counts in `specs/verification.md` and in `AGENTS.md`.

The case set and the reference roots need Bjarne's sign-off (`AGENTS.md`).
