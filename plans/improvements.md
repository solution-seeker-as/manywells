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

## 2. Robustness and API quality

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

### 2.16 No top boundary that fixes the wellhead pressure

**[after the plan]** *From the validation plan's Step V2, 2026-10-02.* ManyWells
can fix the liquid rate (`FixedFlowRate`, INF-8), but its top boundary is always
a choke. Running a well at a given rate and wellhead pressure, the
tubing-performance calculation of nodal analysis, needs the wellhead pressure
fixed instead. A feature spec, approved before implementation, would cover:

- **The boundary.** A top boundary whose row is $p_N - p_\text{wh}$, in place of
  the choke row: a new option in `choke.md`, or a top-boundary model beside
  `ChokeModel`, as `specs/architecture.md` decides. With INF-8 at the bottom,
  the case has one unknown left, $p_0$.
- **The operating point.** With both the rate and $p_N$ fixed, SOL-3's label
  (the sign of $dR/dp_0$ for the choke row) no longer describes stability. The
  spec says what the root set and the operating point are for this boundary.
- **The temperature.** Proposed: the energy balance as it is, from $T_r$ at the
  bottom. A prescribed temperature profile would be a further option.
- **The usual rules.** Off in the `v1.0.0` configuration; in both backends under
  the same IDs, and in `tests/backend_cases.py`.

Done when a synthetic round trip passes in both backends. A well solved with a
choke and a reservoir inflow, then solved again with `FixedFlowRate` at the
solved liquid rate and the wellhead pressure fixed at the solved $p_N$, gives the
same solution to the solver's tolerance.

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

- **[after the plan]** **Docs site**: mkdocs-material with mkdocstrings on GitHub
  Pages would render the guides in `docs/` and the docstrings.
- **[after the plan]** **License**: one of the decisions for v2.0.0
  (`specs/goals.md`). CC BY-NC 4.0 is unusual for code: Creative Commons
  recommends against CC licenses for software, mainly because they do not address
  source and binary distribution or patents. CC BY-NC for the datasets and a
  noncommercial software license (e.g. PolyForm Noncommercial) for the code would
  express the same intent with less ambiguity.
- **[any time]** **Type checking**: fill the gaps in the annotations, so that
  users' type checkers see them (the package has a `py.typed` marker). The
  physics methods take floats or CasADi symbols, so their annotations need a
  type for both.

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
- **`regenerate_distributions.py` skips SMP-28's row check** (`_valid`,
  `sampling/generate.py:61`). It should not trigger at the stable root.

## Done and removed

Numbers are not reused. Each removed item, and where it was handled:

| § | Item | Handled |
|---|---|---|
| 1.1 | Gas-lift simulation crashed | `8b05b18`, with a gas-lift test |
| 1.2 | `water_fvf` had the wrong sign | Step 7 (PVT-WAT-2, Bjarne 2026-10-01) |
| 1.3 | Data-generation scripts broken against the API | Step 7: `manywells.sampling`, frozen inputs |
| 1.5 | Choke docstring did not match the code | `1811ae2` (D-13) |
| 1.4 | User-supplied `cpr` silently overwritten | `cpr` is a property from CHK-4 (D-21), `a1e8da4` |
| 1.6 | The examples' run command, `p_sep`/`p_bubble` in bar, typos, the README tree; then the units docstring, the surface tension, the slip features, the L-shaped and black-oil examples, `rust_roots.py`, two license headers | Step 5, Step 7, `d8f015f`, `661e36d`; the rest `a1e8da4` |
| 1.7 | The CasADi search missed all-liquid bottomholes | issue #10, PR #23 (`specs/features/012-root-search.md`) |
| 1.8 | `x_guess`: dropped by the deprecated `simulate()`, a `Root` refused, the docstring's promise | `a1e8da4` |
| 1.9 | The Rust backend's debug log reported every search as failed | `Attempt.found`, `a1e8da4` |
| 1.10 | `write_dataset` truncated a dotted name | `a1e8da4` |
| 1.11 | The QTOT filter's standard deviation: pandas' sample one (ddof = 1) where v1.0.0 took numpy's population one (ddof = 0) | left to the sampling redesign (Bjarne, 2026-10-29); recorded in `specs/sampling.md`, SMP-28 |
| 1.12 | Calibration's data handling: `WGL`'s default, the index, the rate source without noise, the backend check | `a1e8da4` |
| 1.13 | The distribution check crashed with no candidate row in the reference | `a1e8da4` |
| 1.14 | `scripts/ml_examples/noise.py` scaled the QTOT noise wrongly | `a1e8da4` |
| 2.1 | `assert` validation of the inputs | Step 7; `liquid_mix` and `ca_double_sigmoid` `a1e8da4` |
| 2.2 | `print` instead of `logging` and exception payloads | Step 7 |
| 2.3 | `isinstance` choke dispatch | Step 7 (`specs/architecture.md`) |
| 2.4 | Hidden state `self._w_l_inflow` | Step 7 (DISC-11) |
| 2.5 | The simulator mutated the user's objects | Step 7: frozen inputs |
| 2.6 | `ClosedLoopWellSimulator` cleanups | lapsed: `closed_loop/` retired 2026-10-01 |
| 2.7 | `SlipModel` parameters not configurable | Step 7 (`specs/model/slip.md`) |
| 2.8 | Calibration module consistency | feature 017, `e96053b` |
| 2.11 | The configuration check accepted subclasses | exact classes for every part, `a1e8da4` |
| 2.12 | `NoOperatingPoint` could not be unpickled | `a1e8da4` |
| 2.13 | A NaN slope was labelled stable, and a singular Jacobian lost the search | both `indeterminate`, `a1e8da4` |
| 2.14 | Integer thermal switches crashed the Rust backend | `ThermalModel` stores bools, `a1e8da4` |
| 2.15 | NaN slip constants, NaN diameter, unknown friction correlations | `a1e8da4` |
| 3 | Testing: the gas-lift, examples, water FVF and energy-vector tests; `--doctest-modules` dropped | Steps 5 and 7 |
| 4.1 | Build the NLP once | Step 7 (`plans/evidence/build_once.py`) |
| 4.4 | Vectorize the regime classification | Step 7 (`System.regime_probabilities`) |
| 5 | CI, dependency ranges, `CHANGELOG.md`, the README's Documentation section | 2026-09-22 to 2026-10-02 |
| 5 | Examples hygiene (the `slip.py` demo, removed), the `py.typed` marker | `a1e8da4` |
| 5 | Lint and format: ruff 0.17.1, in CI (`lint`) and a pre-commit hook | `6594e7e` to `3a897df` (`docs/testing.md`) |
| 6 | Suspicions confirmed and fixed: `solution_as_df` aliased `Root.x` under pandas 2.2; numeric strings in `CHK` or `PDC` failed late | `a1e8da4` |
| — | The review of 2026-10-16: its High and Medium items | issues #11 to #22, all fixed; #13 and #16 by PR #29 |
| — | The review's suspicion that the lower temperature bound cuts off roots | confirmed below a geothermal gradient of about 5 K/km; kept as a known limit (`specs/features/012-root-search.md`) |
