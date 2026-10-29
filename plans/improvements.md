# Improvements and corrections

The backlog of code-level findings on `develop`, from a review in June 2026 and
items found since. `manywells-v2-plan.md` decides scope and order. Specs, tests
and the plan cite items here by section number, so the numbers stay fixed, and a
done item keeps a short record of where it was handled.

Each open item is tagged **[after the plan]** (work for v2.0.0 that the
foundation plan leaves out) or **[any time]** (a small fix that needs no plan
step). Principle 7 of the plan (transparent before marginally faster) applies to
every performance item in section 4. Open items were checked against the code on
2026-10-28; line numbers are from that date.

## 1. Bugs and corrections

### 1.1 Gas-lift simulation crashed

**[done 2026-09-30 in `8b05b18`]** `FluidModel.gas_mass_flow_rate` had been
removed, so every simulation with `w_lg > 0` raised `AttributeError`. Fixed, with
tests of the lift-gas mixing temperature and of a full solve with gas lift.

### 1.2 `water_fvf` had the wrong sign

**[done 2026-10-01 in Step 7]** `Bw = 1 - c_w (p - p_ref)` (PVT-WAT-2, decided by
Bjarne on 2026-10-01), with the tests updated. The simulator does not use it.

### 1.3 Data-generation scripts were broken against the current API

**[done 2026-10-01 in Step 7]** `manywells.sampling` and `manywells.datasets` port
the procedure (`specs/sampling.md`), and the generators are thin callers. Frozen
dataclasses make an assignment to a misspelt field raise, where it used to create
a dead attribute and simulate the default well.

### 1.4 User-supplied `cpr` is silently overwritten

**[any time]** `ChokeModel` has `cpr` as a constructor field, but `__post_init__`
always sets it from CHK-4 (`choke.py:51,59`), so `ChokeModel(cpr=0.6)` gets 0.544
without a warning. The field's comment now says so, and `specs/discrepancies.md`
(D-21) rules to fix it on `develop` any time. Taking `cpr` out of the constructor
changes nothing else: the `v1.0.0` configuration and the Rust core both take the
ratio from CHK-4. The class docstring calls the same ratio `cpm`.

### 1.5 Choke docstring did not match the implementation

**[done 2026-09-30 in `1811ae2`]** The docstring has $\Phi$ inside the square root,
as the code does (D-13).

### 1.6 Smaller documentation and comment corrections

- **[any time]** `units.py:8` says that all public interfaces use SI units
  (Pa, ...), but the simulator and the components take bar (AGENTS.md,
  Contracts). Fix the module docstring.
- Done: `AGENTS.md` documents `uv run python -m scripts.<folder>.<name>` for the
  examples (Step 5, 2026-09-30); `FluidModel.p_sep` and `p_bubble` are in bar
  (Step 7, feature 005); the `gas_fvf` return doc, the "nubmer" typo and the
  "Used to generate dataset v6" comment went in `d8f015f`; the README's project
  tree is current (`661e36d`). `NOTES.md` is no longer in the repo.
- Dropped: `friction.py`'s 2024 copyright line is the header every file carries.

### 1.7 The CasADi search missed operating points where the bottomhole is all liquid

**[done 2026-10-28, issue #10]** Where all the reservoir gas is dissolved at the
bottomhole, point 0's void fraction is $10^{-4}$ to $10^{-3}$, and from the bottom
guess's 0.5 point 0 failed at every start near the root: at well 87 of seed 2026
with `gor=40` the CasADi search found no operating point, and the Rust core one at
304.40 bar. The march now retries point 0 from the phase rates' no-slip void
fraction where it fails. On develop's sampled wells the search finds all 116 such
operating points, where it missed 15, for 0.16% more Ipopt iterations, and the
verifier's root sets are unchanged (`specs/features/012-root-search.md`;
`plans/evidence/all_liquid_bottomhole.py`). Bjarne chose the retry over guessing
the no-slip void fraction at every start, which cost 11 to 20% more iterations.

## 2. Robustness and API quality

### 2.1 Replace `assert` validation with exceptions

**[done 2026-10-01 in Step 7]** for the inputs: `WellProperties`,
`BoundaryConditions`, the component models and `FluidModel` raise `ValueError` in
`__post_init__`.

**[any time]** Two asserts remain, in helpers rather than inputs: `liquid_mix`
(`pvt/__init__.py:77`) and `ca_double_sigmoid` (`ca_functions.py:93`).

### 2.2 Use `logging` and exception payloads instead of `print`

**[done]** The simulator never prints: failed starts go to
`logging.getLogger('manywells')`, and `NoOperatingPoint` carries the root set
(Step 7, 2026-10-01). Calibration's prints went with its old functions (§2.8).

### 2.3 `isinstance`-based choke dispatch blocked extension

**[done 2026-10-01 in Step 7]** `ChokeModel.mass_flow_rate(u, p_s, s, A)`, with
each model's `density_and_multiplier`, as settled in Step 6
(`specs/architecture.md`).

### 2.4 Hidden state `self._w_l_inflow`

**[done 2026-10-01 in Step 7]** `discretization.py` passes $w_\text{res}$ to every
point's rows.

### 2.5 The simulator mutated the user's objects

**[done 2026-10-01 in Step 7]** Frozen inputs, the default choke set in
`WellProperties.__post_init__`, and the operating point as parameters of the
system. The closed-loop half lapsed when `closed_loop/` was retired (2026-10-01).

### 2.6 `ClosedLoopWellSimulator` cleanups

**[out of v2]** Lapsed: `closed_loop/` was retired on 2026-10-01 (Bjarne). It
remains in v1.0.0.

### 2.7 `SlipModel` parameters were not configurable

**[done 2026-10-01 in Step 7]** The four regime constants are fields, as in
`SlipModel(C_0_slug=1.2)`, with v1.0.0's values as defaults
(`specs/model/slip.md`).

### 2.8 Calibration module consistency

**[done 2026-10-02 in `e96053b`, feature 017]** The old calibration functions,
which raised a bare `Exception` and printed, are retired. `calibrate` raises
`CalibrationError` (a `SimError`) and logs (`specs/calibration.md`).

### 2.9 The CasADi root search misses roots the Rust core finds

**[after the plan]** *Found in Step 8, 2026-10-01*
(`specs/features/014-rust-solver.md`, Findings 3). Both backends are kept
(`specs/goals.md`), so the CasADi backend's search stays in use, and its
completeness matters.

- In Step 8's regenerated `sol-1` samples (first 200 wells, 1,000 samples), the
  Rust core has an operating point in 3 samples where the multi-start search has
  none: IDs 16, 70 and 130. Each is a genuine root: every CasADi row is about
  1e-13 there, and Ipopt started there stays.
- At ID 67, k = 0, the core also finds an unstable root the CasADi search misses.
- The case set does not show this: both backends pass it at 100%.
- *Step 9, 2026-10-02* (`specs/features/015-rust-develop-model.md`): on the
  comparison set at 20 wells per configuration (520 cases, every option of
  `develop`'s model switched on and off), the core finds 21 roots the CasADi
  search misses, 20 of which zero every CasADi row. 17 of them are in cases where
  the slip law has one void fraction, mostly low-rate roots near $p_r$ (develop's
  well 15 in every develop configuration, L-shaped v1.0.0 wells). The 21st is the
  core's jump root at `v1.0.0+deviated#17` (015, Findings).
- *Feature 016* (`specs/features/016-joule-thomson.md`): the root only the core
  finds at `develop#15` zeroes the CasADi rows. The full suite's one warning is a
  core-only unstable root in `v1.0.0+L-shaped#0` at 433.06 bar.
- *Feature 017, 2026-10-02* (`specs/features/017-calibration.md`): the CasADi
  backend cannot calibrate well 1 (L-shaped), where its search finds no root at
  two of the four rows and the core finds a stable one. The core is calibration's
  backend.
- *Issue #10, 2026-10-28:* one cause found and fixed. An all-liquid bottomhole
  defeated the march's guess for point 0 (§1.7). The search still misses 9 of
  636 operating points with gas-rich bottomholes on develop's sampled wells.
- An option, in the `v1.0.0` configuration and later in every configuration the
  core covers: use the core's roots as extra starts for Ipopt, and record which
  roots only one backend finds. That is a comparison between the backends, not an
  independent search, so the verifier's reference stays as it is.

### 2.10 A calibration fit can end in a false minimum

**[after the plan]** *Found by feature 020, 2026-10-29*
(`specs/features/020-profile-scale.md`, Measurements, Twins). On the twin study's
well 1 (L-shaped), with `profile_scale` free, the fit stops at
$z_\text{profile\_scale} = -1.00$, one of CAL-9's starts, against a truth of
$+1.26$, with `h` 1.9 and `K_c` 0.8 from it. Along `profile_scale` alone the cost
is smooth with its minimum near the truth, so the minimum is in the joint space,
as feature 017's well 3 was for `roughness` before CAL-9's start search. It is
behind three of the four fits that miss 017's twin criteria. Options: more starts,
or starts along pairs of parameters, each measured under principle 7.

## 3. Testing

- Done: a gas-lift test (§1.1, `8b05b18`); `tests/test_examples.py` runs each
  example headless (Step 5, 2026-09-30); the water FVF tests (§1.2); test vectors
  for the energy balance, THM-4 to THM-7 in `specs/model/thermal.md` (Step 7).
- Dropped: `pytest --doctest-modules`. No docstring in `src/` has a `>>>` example.

## 4. Performance and architecture

### 4.1 Build the NLP once, parameterize the boundary conditions

**[done 2026-10-01 in Step 7]** `build_system`, `IpoptSolver` and the march's
rootfinders are built once per well. On the case set a warm solve went from
0.507 s to 3.5 ms and a cold solve from 0.719 s to 24.7 ms, at the median (plan,
Step 7 status; `plans/evidence/build_once.py`).

### 4.2 Further solver-level options

**[after the plan]** Each option adds machinery, so under principle 7 each needs
a large, measured gain of its own. A new adapter goes in `solvers/` with the same
`solve` contract (`specs/architecture.md`).

- **JIT compilation**: `ca.nlpsol(..., {'jit': True, 'compiler': 'shell'})`
  compiles the callbacks to C, which pays where one solver instance is reused
  many times, as it is since §4.1. Opt-in, since it needs a C compiler.
- **Dual warm starts**: pass the previous solution's duals (`lam_g0`, `lam_x0`)
  with `ipopt.warm_start_init_point: 'yes'` when sweeping operating points.
- **Newton on the full system**: the problem is square, so `ca.rootfinder` with
  `'newton'` or `'kinsol'` on the whole system, falling back to Ipopt, could
  speed up the easy solves. The march (`solvers/march.py`) already solves each
  point by Newton, so measure against it.

### 4.3 Library-level batch API

**[after the plan]** With the v2 dataset generation. Dataset generation has its
process pool in the library now (`manywells.sampling.generate.generate`). A batch
API for users is still missing: `specs/architecture.md` (Rust core, design
point 7) designs it as the core's `root_sets(well, operating_points)`, which
solves many operating points of one well in threads with the GIL released. It is
not built.

### 4.4 Vectorize `solution_as_df` regime classification

**[done 2026-10-01 in Step 7]** One classifier function mapped over the points
(`System.regime_probabilities`).

### 4.5 Make the fluid's fractions parameters of the CasADi system

**[after the plan]** It changes the contract of `System.params` in
`specs/architecture.md`, whose parameters are the operating point, so it needs
Bjarne's sign-off.

The fluid is a constant of a well's system. Each dataset sample redraws the
inflow's mass fractions (SMP-22, SMP-24), and calibration solves each row and
parameter value on its own fluid, so on the CasADi backend each builds a new
system and runs the full root search. Measured in Step 7 on 8 verifier wells at
N = 100, that is 0.87 s per sample (build 0.31 s, search 0.56 s, median), against
3 ms for a warm solve once a well is built.

The Rust core makes this unnecessary for dataset generation: the sampler runs on
it (`Settings(backend='rust')`, `2b0a50b`), and a core well costs almost nothing
to build. It matters only if datasets are generated, or wells calibrated, on the
CasADi backend (feature 017).

If it is needed: add the quantities that the fractions set to the system's
parameters, next to the operating point. These are $f_g$, $\rho_l$ and $c_{pl}$ in
the `v1.0.0` configuration (INF-4, PVT-MIX-1, THM-1), and the gas–oil and
water–liquid ratios in `develop`'s fluid model (PVT-MIX-10). `FluidModel`'s
derived quantities must then accept CasADi symbols, and its validation stays on
the well's base values. An early exit in the search, so that a sample costs about
a warm solve, is solver policy with its own measurement, because SOL-6 picks the
lowest-$p_0$ stable root of several.

## 5. Open-source friendliness

- Done: CI (`.github/workflows/tests.yml`, 2026-09-22); dependency ranges,
  `requires-python >= 3.11` and `dev` as a dependency group (2026-09-22);
  `CHANGELOG.md`, with an old→new snippet for every API break (the v2.0.0 tag is
  release work); a Documentation section in the README (2026-10-02).
- **[any time]** **Lint and format**: there is no ruff, black or mypy
  configuration. `ruff` (format and lint) with a pre-commit hook keeps
  contributions uniform at little cost.
- **[after the plan]** **Docs site**: mkdocs-material with mkdocstrings on GitHub
  Pages would render the guides in `docs/` and the docstrings.
- **[any time]** **Examples hygiene**: `slip.py` still ends with a `__main__` demo
  (line 247). Move it to `scripts/` or a test; library modules with executable
  tails evade testing.
- **[after the plan]** **License**: one of the decisions for v2.0.0
  (`specs/goals.md`). CC BY-NC 4.0 is unusual for code: Creative Commons
  recommends against CC licenses for software, mainly because they do not address
  source and binary distribution or patents. CC BY-NC for the datasets and a
  noncommercial software license (e.g. PolyForm Noncommercial) for the code would
  express the same intent with less ambiguity.
- **[any time]** **Type checking**: a `py.typed` marker, and filling the gaps in
  the annotations, lets users' type checkers see them.
