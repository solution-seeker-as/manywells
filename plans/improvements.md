# Improvements and corrections

The backlog of code-level findings on `develop`: from a review in June 2026, the
bug review of 2026-10-16 (at `396db80`), and items found since.
`manywells-v2-plan.md` decides scope and order. Specs, tests and the plans cite
items here by section number, so the numbers stay fixed: a done item is
removed, its number is not reused, and the table at the end says where it was
handled.

Each open item is tagged **[after the plan]** (work for v2.0.0 that the
foundation plan leaves out), **[any time]** (a small fix that needs no plan
step) or **[needs sign-off]** (it changes a spec, a bound or the sampling, which
Bjarne decides; `AGENTS.md`). Principle 7 of the plan (transparent before
marginally faster) applies to every performance item in section 4. Open items
were checked against the code on 2026-10-29; line numbers are from that date.
The review's items give the inputs that reproduce them, because its scripts are
not in the repo. Its suspicions that it could not confirm are in section 6.

## 1. Bugs and corrections

### 1.4 User-supplied `cpr` is silently overwritten

**[any time]** `ChokeModel` has `cpr` as a constructor field, but `__post_init__`
always sets it from CHK-4 (`choke.py:51,59`), so `ChokeModel(cpr=0.6)` gets 0.544
without a warning. The field's comment says so, and `specs/discrepancies.md`
(D-21) rules to fix it on `develop` any time. Taking `cpr` out of the constructor
changes nothing else: the `v1.0.0` configuration and the Rust core both take the
ratio from CHK-4. The class docstring calls the same ratio `cpm`.

### 1.6 Smaller documentation and comment corrections

**[any time]**

- `units.py:8` says that all public interfaces use SI units (Pa, ...), but the
  simulator and the components take bar (`AGENTS.md`, Contracts).
- `pvt/black_oil.py:205` says that at $R_{so}$ = 500 scf/STB the surface tension
  "drops to roughly 40%" of the dead-oil value; the code gives 19.5%.
- The flow-regime feature list in `slip.py:57` and `59` omits the factor 2 in
  `tanh(2 (alpha - 0.7))` and `tanh(2 (alpha - 0.25 cos))` that the code uses
  (lines 100 and 102; the factor itself is D-1).
- `scripts/sim_examples/L_shaped_well.py:8` and its suptitle (line 98) compare
  the L-shaped well with "a vertical well of the same TVD". The L-shaped well
  reaches 1800 m TVD and the vertical one 2000 m.
- `scripts/black_oil_tutorial.py:66` annotates "Rs is capped (all gas
  dissolved)". With `gor=200` and `p_bubble=250`, $R_{so}$ caps at
  115.05 Sm³/Sm³ (PVT-OIL-7), so 85 Sm³/Sm³ stays free.
- `verification/build/rust_roots.py:16-18` and `104` say it writes
  `data/full/...`; it writes `verification/data/build/`. They also say
  `label_cases.py` refines the roots with Newton; `make_cases.py solve` re-solves
  them with v1.0.0's Ipopt.
- Two scripts lack the CC BY-NC header: `scripts/black_oil_tutorial.py` and
  `scripts/flow_regimes/plot_flow_regime_map.py`.

### 1.8 `x_guess` in the search

**[any time]** *From the review of 2026-10-16.*

- The deprecated `simulate()` drops an `x_guess` argument: on the deprecated
  path `simulator.py:214` replaces it with `self.x_guess`, so
  `SSDFSimulator(wp, bc).simulate(x_guess=g)` searches without `g`. Fix:
  `x_guess = self.x_guess if x_guess is None else x_guess`.
- A `Root` as `x_guess` raises `TypeError`. The docstring at `simulator.py:179`
  suggests "the root of a nearby operating point", but
  `np.asarray(x_guess, dtype=float)` (`solvers/roots.py:139`) raises on a `Root`.
  Fix: accept a `Root` and use its `.x`, or document `op.x`.
- The `simulate()` docstring (`simulator.py:204`) says `x_guess` "makes the
  search faster, not the answer different". Neither half holds: `find()` runs
  every start anyway (7 against 8 attempts, 0.128 s either way), and an extra
  start can add a root, which through SOL-6 can change the operating point. Fix:
  reword it.

### 1.9 With the Rust backend, the debug log reports every search as failed

**[any time]** `simulator.py:192` filters on `outcome.startswith('root')`, but
the core's outcome reads `'2 roots'`. Fix: decide failure from the attempt's
kind, not its text.

### 1.10 `write_dataset` truncates a dotted `--out` name

**[any time]** `path.with_suffix` and `path.stem` in `write_dataset`
(`datasets/io.py:67-70`) make `--out data/sol-v1.0` and `--out data/sol-v1.1`
both write `data/sol-v1.parquet`, `sol-v1_config.parquet` and
`sol-v1_meta.json`, so the second run overwrites the first. Fix:
`path.parent / (path.name + '.parquet')`, and likewise for the other two files.

### 1.11 The QTOT variation filter uses a different standard deviation from v1.0.0

**[needs sign-off]**

- `sampling/generate.py:104` and `195` use pandas' `df['QTOT'].std()`, the
  sample standard deviation (ddof = 1).
- v1.0.0 used `well_data['QTOT'].values.std()`, numpy's population standard
  deviation (ddof = 0). The CHK filter used pandas in v1.0.0 too, so it matches.
- The port's coefficient of variation is therefore 1.001 times v1.0.0's, and a
  well with a population value in [0.04995, 0.05) is kept where v1.0.0 discarded
  it. `specs/sampling.md` (S-6) does not say which standard deviation applies.

Fix: `std(ddof=0)`, and say so in the spec; or settle it in the sampling
redesign.

### 1.12 Calibration's data handling

**[any time]**

- The docs say a missing `WGL` means 0, but the code uses `bc.w_lg`, as CAL-1
  says: `docs/calibration.md:96` and the `CalibrationData` docstring
  (`calibration/data.py:58`) are wrong. A user who passes `bc` with `w_lg = 2`
  and rows without `WGL` gets every row solved with 2 kg/s of lift gas. Fix:
  correct both texts.
- `CalibrationData` drops the index: `reset_index(drop=True)` (`data.py:75`)
  loses a time stamp held there. `docs/calibration.md:114` and `data.py:69`
  promise that other columns, such as a time stamp, are kept in the residual
  table, and CAL-12 suggests plotting residuals against time. Fix:
  `reset_index()` without `drop`, or carry the index into the residual table.
- `synthetic_data(noisy=False, rate_source='mpfm')` ignores the rate source:
  `RATE_SOURCE` is set only in `add_noise` (`synthetic.py:82`), which runs only
  if `noisy` (`synthetic.py:130`). The rows are then weighted with the well-test
  noise (2.5%) instead of the MPFM noise (10%). Fix: set `RATE_SOURCE` whatever
  `noisy` is.
- `synthetic_data` gives a misleading error for a well the core cannot solve.
  Each row's `ValueError` ("Use backend='casadi'") is swallowed
  (`synthetic.py:58`), and the user sees "the well has no operating point at any
  of the operating points drawn". `calibrate` and `evaluate` call
  `check_backend` (`fit.py:118`); `synthetic_data` does not. Fix: call it before
  solving.

### 1.13 The distribution check crashes when no candidate well is in the reference

**[any time]** For example, a CSV whose IDs are strings or come from another
dataset: `well_weights` gives every row weight 0, `compare`
(`verification/src/manywells_verify/distributions.py:106`) filters them all
out, and `weighted_cdf` indexes `cw[-1]` of an empty array (line 80), raising
`IndexError`. Fix: raise `ValueError('no candidate row belongs to a reference
well')` when nothing is kept.

### 1.14 `scripts/ml_examples/noise.py` scales the QTOT noise wrongly

**[any time]** Line 68 converts the WTOT noise at the gas density, but WTOT's
noise includes the oil and water noise. QTOT's noise comes out with a standard
deviation of 164.07 against 22.83 for the sum of its parts: 2.65% of QTOT
instead of the intended 0.5%. QTOT is neither an input nor an output of the
examples, but `add_noise_and_save` writes it. Fix: sum the QGL, QGAS, QOIL and
QWAT noise.

## 2. Robustness and API quality

### 2.1 Two asserts remain

**[any time]** The inputs raise `ValueError` in `__post_init__` since Step 7, but
two helpers still assert: `liquid_mix` (`pvt/__init__.py:77`) and
`ca_double_sigmoid` (`ca_functions.py:93`).

### 2.9 The root searches miss roots the other backend finds

**[after the plan]** *Found in Step 8, 2026-10-01*
(`specs/features/014-rust-solver.md`, Findings 3). Both backends are kept
(`specs/goals.md`), so each search's completeness matters. The verifier's case
set does not show this: both backends pass it at 100%.

- **Roots only the core finds.**
  - Step 8's regenerated `sol-1` samples (first 200 wells, 1,000 samples): an
    operating point in 3 samples where the CasADi search has none (IDs 16, 70
    and 130), and an unstable root it misses at ID 67, k = 0. Each zeroes every
    CasADi row (about 1e-13), and Ipopt started there stays.
  - Step 9's comparison set (`specs/features/015-rust-develop-model.md`, 520
    cases): 21 roots, 20 of which zero every CasADi row; 17 of them where the
    slip law has one void fraction, mostly low-rate roots near $p_r$ (develop's
    well 15 in every develop configuration, L-shaped v1.0.0 wells). The 21st is
    the core's jump root at `v1.0.0+deviated#17`.
  - The full suite's one warning: an unstable root in `v1.0.0+L-shaped#0` at
    433.06 bar (016).
  - Feature 017: the CasADi backend cannot calibrate well 1 (L-shaped), where it
    finds no root at two of the four rows and the core finds a stable one. The
    core is calibration's backend.
  - Feature 019, with the five regimes: an unstable root 0.003 bar below $p_r$ at
    case (9, 1), and a stable one within 0.7 bar of $p_r$ at (32, 1).
  - Issue #10 found one cause, an all-liquid bottomhole, and fixed it (PR #23).
    The CasADi search still misses 9 of 636 operating points with gas-rich
    bottomholes on develop's sampled wells.
- **Roots only the CasADi search finds.** Where the slip law has several void
  fractions, the core misses roots on other branches (`develop#17`, 015 Finding
  2). With the profile scale at $\kappa_0 \le 0.5$ (020), it misses a second
  stable root in 1 of 90 sampled cases, and the backends pick different
  operating points, 0.77 bar apart.
- An option, in the `v1.0.0` configuration and later in every configuration the
  core covers: use the core's roots as extra starts for Ipopt, and record which
  roots only one backend finds. That is a comparison between the backends, not
  an independent search, so the verifier's reference stays as it is.

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

### 2.11 The configuration check accepts subclasses that change the physics

**[any time]** `configurations.differences` checks the friction, inflow and
choke with `isinstance` (`configurations.py:116-128`). A `ProductivityIndex`
subclass with a quadratic IPR gives `differences(wp, 'v1.0.0') == []`, and
`v1_well` accepts it. The check of the PVT model (line 95) and the Rust adapter
use exact types for this reason. Fix: `type(x) in (...)`.

### 2.12 `NoOperatingPoint` cannot be unpickled

**[any time]** In `simulator.py:42-48`, `args` holds the message, so unpickling
calls `__init__(message)` and fails with
`AttributeError: 'str' object has no attribute 'roots'`. The repo's workers catch
it inside the worker, so only users' multiprocessing code is affected. Fix: a
`__reduce__` returning `(type(self), (self.root_set,))`.

### 2.13 A NaN stability slope is labelled stable, and a singular Jacobian loses the search

**[any time]** Latent: neither was reproduced at a real root.

- `label_of` (`solvers/roots.py:70-71`) returns `'stable'` for NaN. The Rust
  adapter guards against it (`solvers/rust.py:142`); the CasADi search does not.
- `ca.solve(..., 'csparse')` (`roots.py:98`) raises on a singular reduced
  Jacobian. `find()` does not catch it, so every root found so far is lost.

Fix: label a non-finite slope `indeterminate`, and catch the factorization error
as `indeterminate`.

### 2.14 Integer thermal switches crash the Rust backend

**[any time]** `solvers/rust.py:76-77` passes the four switches of
`ThermalModel` unconverted, and pyo3 rejects an `int` for a `bool`:
`ThermalModel(frictional_heating=0)` gives
`TypeError: 'int' object is not an instance of 'bool'` on the core, where the
CasADi backend solves it. The `TypeError` also escapes calibration's
`except (SimError, ValueError)`. Fix: pass `bool(...)`, or validate the switches
in `ThermalModel`.

### 2.15 Validation gaps in the components

**[any time]**

- NaN slip constants pass. `SlipModel.__post_init__` (`slip.py:205`) does not
  check the regime constants, and `not_in_core` (`solvers/rust.py:55`) tests
  `min(...) < 1`, which is False for NaN: `SlipModel(C_0_slug=nan)` gives
  `not_in_core(wp) == []`. Fix: validate the constants as finite with
  `C_0 >= 1` in `SlipModel`, and write the adapter's test as `not (C_0 >= 1)`.
- A NaN diameter is accepted in Python: `geometry.py:97` checks `self.D <= 0`,
  which lets NaN through, while the core's `from_grid` rejects it, so the
  backends disagree. Fix: `if not self.D > 0`.
- Unknown friction correlations fall back to Haaland: `friction_factor`
  (`friction.py:77`) uses Haaland for any name but `'chen'`, so
  `friction_factor(1e5, 1e-4, 'colebrook')` returns Haaland's 0.018265.
  `RoughnessFriction` validates the name, so only direct callers are exposed.
  Fix: `elif correlation == 'haaland'`, and raise otherwise.

## 4. Performance and architecture

### 4.2 Further solver-level options

**[after the plan]** Each option adds machinery, so under principle 7 each needs
a large, measured gain of its own. A new adapter goes in `solvers/` with the same
`solve` contract (`specs/architecture.md`).

- **JIT compilation**: `ca.nlpsol(..., {'jit': True, 'compiler': 'shell'})`
  compiles the callbacks to C, which pays where one solver instance is reused
  many times, as it is since the system is built once. Opt-in, since it needs a
  C compiler.
- **Dual warm starts**: pass the previous solution's duals (`lam_g0`, `lam_x0`)
  with `ipopt.warm_start_init_point: 'yes'` when sweeping operating points.
- **Newton on the full system**: the problem is square, so `ca.rootfinder` with
  `'newton'` or `'kinsol'` on the whole system, falling back to Ipopt, could
  speed up the easy solves. The march (`solvers/march.py`) already solves each
  point by Newton, so measure against it.

### 4.3 Library-level batch API

**[after the plan]** With the v2 dataset generation. Dataset generation has its
process pool in the library (`manywells.sampling.generate.generate`). A batch
API for users is still missing: `specs/architecture.md` (Rust core, design
point 7) designs it as the core's `root_sets(well, operating_points)`, which
solves many operating points of one well in threads with the GIL released. It is
not built.

### 4.5 Make the fluid's fractions parameters of the CasADi system

**[after the plan]** **[needs sign-off]** It changes the contract of
`System.params` in `specs/architecture.md`, whose parameters are the operating
point.

The fluid is a constant of a well's system. Each dataset sample redraws the
inflow's mass fractions (SMP-22, SMP-24), and calibration solves each row and
parameter value on its own fluid, so on the CasADi backend each builds a new
system and runs the full root search. Measured in Step 7 on 8 verifier wells at
N = 100, that is 0.87 s per sample (build 0.31 s, search 0.56 s, median),
against 3 ms for a warm solve once a well is built.

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

- **[any time]** **Lint and format**: there is no ruff, black or mypy
  configuration. `ruff` (format and lint) with a pre-commit hook keeps
  contributions uniform at little cost.
- **[after the plan]** **Docs site**: mkdocs-material with mkdocstrings on GitHub
  Pages would render the guides in `docs/` and the docstrings.
- **[any time]** **Examples hygiene**: `slip.py` still ends with a `__main__` demo
  (line 383). Move it to `scripts/` or a test; library modules with executable
  tails evade testing.
- **[after the plan]** **License**: one of the decisions for v2.0.0
  (`specs/goals.md`). CC BY-NC 4.0 is unusual for code: Creative Commons
  recommends against CC licenses for software, mainly because they do not address
  source and binary distribution or patents. CC BY-NC for the datasets and a
  noncommercial software license (e.g. PolyForm Noncommercial) for the code would
  express the same intent with less ambiguity.
- **[any time]** **Type checking**: a `py.typed` marker, and filling the gaps in
  the annotations, lets users' type checkers see them.

## 6. Suspicions not confirmed

From the review of 2026-10-16. Each needs a reproduction before a fix.

- **[needs sign-off]** **A labelling candidate's empty root set passes the
  verifier's Root set and Stability checks.** `checks.py:190` makes both n/a
  when a case has at most one unlabelled root, including none. A candidate that
  labels its roots but reports nothing for a case whose reference has only an
  unstable root passes every check. On the current case set every case with
  roots has a stable root, so the Operating point check catches it. The spec's
  wording ("for candidates that ...") leaves open whether such a set should be
  checked.
- **Expected-failure entries for unknown cases are ignored silently.** Entries
  whose `case_id` is not in the case set are neither counted nor listed as now
  passing (`verification/src/manywells_verify/report.py`).
- **`--json` can write non-standard JSON** (`NaN`, `-Infinity`) if an observed
  convergence order is the log of 0. Not reachable in practice.
- **The distribution reference is found only through the editable install.**
  `distributions.py:44` resolves it as `parents[2] / 'data'`, which works because
  the workspace installs the verifier editable; from a wheel it would not be
  found.
- **Calibration parameters are unbounded.** `Parameter.value` uses
  `exp(log_sd z)` and `least_squares` runs without bounds; feature 017 records a
  fit reaching z = 13 (a roughness of 143 m). A larger step could hand the core
  an infinite `K_c`, `w_l_max` or roughness. Not tried. Fix: guard the values
  passed to the backends.
- **Rust core:**
  - *Label of a root next to a non-finite region.* If R is not finite at
    `p_0 ± h`, the slope is NaN and the adapter labels the root `indeterminate`.
    It ignores the core's `rising` flag, which knows the sign. No case found.
  - *`cell_step` drift* (`march.rs:364`). Trial states between `p_s` and `p_prev`
    that fail count as infinite. The golden-section search can then drift
    towards `p_prev` and return `BelowSeparator` or `Unsolved`, although the row
    at `p_s` is finite.
  - *Step-outs into a NaN region.* The temperature bracket's step-outs
    (`march.rs`, around lines 190 to 240) keep doubling into a region where the
    row is NaN, since NaN is neither `>= 0` nor `<= 0`. They can step past a root
    that lies between the last finite end and that region.
  - *Undercounted temperature failures.* The `temperature_failures` counter
    misses several failure exits of `solve_temperature`. This affects only the
    counts reported in the feature specs.
  - *Narrow positive bumps.* The scan refines only local minima with R >= 0, so
    a narrow positive bump between negative samples, from two close roots, would
    be missed.
  - *Tiny drawdowns.* If `p_r - p_s < 1.001e-3` bar, `p_lo > p_hi`
    (`shoot.rs:65`) and the scan samples above `p_r`.
- **`solution_as_df` may alias `Root.x` under pandas older than 3.0.**
  `pyproject.toml` allows `pandas>=1.5`, and without copy-on-write an in-place
  edit of the frame could change the frozen root. Under the locked pandas there
  is no aliasing.
- **`regenerate_distributions.py` skips SMP-28's row check** (`_valid`,
  `sampling/generate.py:61`). It should not trigger at the stable root.
- **Numeric strings in `CHK` or `PDC` fail late.** `CalibrationData` checks them
  with `pd.to_numeric` (`data.py:85` and `102`) but never converts them, so they
  raise a `TypeError` later rather than a `ValueError` up front.

## Done and removed

Numbers are not reused. Each removed item, and where it was handled:

| § | Item | Handled |
|---|---|---|
| 1.1 | Gas-lift simulation crashed | `8b05b18`, with a gas-lift test |
| 1.2 | `water_fvf` had the wrong sign | Step 7 (PVT-WAT-2, Bjarne 2026-10-01) |
| 1.3 | Data-generation scripts broken against the API | Step 7: `manywells.sampling`, frozen inputs |
| 1.5 | Choke docstring did not match the code | `1811ae2` (D-13) |
| 1.6 | The examples' run command, `p_sep`/`p_bubble` in bar, typos, the README tree | Step 5, Step 7, `d8f015f`, `661e36d`; the open bullets stay above |
| 1.7 | The CasADi search missed all-liquid bottomholes | issue #10, PR #23 (`specs/features/012-root-search.md`) |
| 2.1 | `assert` validation of the inputs | Step 7; the open part stays above |
| 2.2 | `print` instead of `logging` and exception payloads | Step 7 |
| 2.3 | `isinstance` choke dispatch | Step 7 (`specs/architecture.md`) |
| 2.4 | Hidden state `self._w_l_inflow` | Step 7 (DISC-11) |
| 2.5 | The simulator mutated the user's objects | Step 7: frozen inputs |
| 2.6 | `ClosedLoopWellSimulator` cleanups | lapsed: `closed_loop/` retired 2026-10-01 |
| 2.7 | `SlipModel` parameters not configurable | Step 7 (`specs/model/slip.md`) |
| 2.8 | Calibration module consistency | feature 017, `e96053b` |
| 3 | Testing: the gas-lift, examples, water FVF and energy-vector tests; `--doctest-modules` dropped | Steps 5 and 7 |
| 4.1 | Build the NLP once | Step 7 (`plans/evidence/build_once.py`) |
| 4.4 | Vectorize the regime classification | Step 7 (`System.regime_probabilities`) |
| 5 | CI, dependency ranges, `CHANGELOG.md`, the README's Documentation section | 2026-09-22 to 2026-10-02 |
| — | The review of 2026-10-16: its High and Medium items | issues #11 to #22, all fixed; #13 and #16 by PR #29 |
| — | The review's suspicion that the lower temperature bound cuts off roots | confirmed below a geothermal gradient of about 5 K/km; kept as a known limit (`specs/features/012-root-search.md`) |
