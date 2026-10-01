# Improvements and corrections

A review of the project as of June 2026 (branch `develop`). Items are ordered by
priority within each section, with file/line references to the current code.
Verified findings (reproduced locally) are marked **[verified]**.

**Relation to the v2 plan** (added 2026-09-30). `manywells-v2-plan.md` decides
scope and order; this file is the backlog of code-level findings it draws on.
Each item is tagged with where the plan handles it: **[plan: Step N]**,
**[after the plan]** (work for v2.0.0 that the foundation plan leaves out),
**[out of v2]**, or **[any time]** for small fixes that need no plan step.
Principle 7 of the plan (transparent before marginally faster) applies to every
performance item in section 4.

## 1. Bugs and corrections

### 1.1 Gas-lift simulation crashes: `FluidModel.gas_mass_flow_rate` no longer exists **[verified]**

**[fixed 2026-09-30 in `8b05b18`]** with the one-liner below, the `inflow.py`
docstring updated, and two tests: the lift-gas mixing temperature, and a full
solve with gas lift that checks the lift gas ends up in the gas phase.

`src/manywells/simulator.py:218` calls `fl.gas_mass_flow_rate(w_l_inflow)` inside
`_left_boundary_eqs`, but this method was removed from `FluidModel` in the recent
refactor ("Remove flow rate computation from fluid model interface"). Any
simulation with `w_lg > 0` fails:

```
AttributeError: 'FluidModel' object has no attribute 'gas_mass_flow_rate'
```

This breaks the lift-gas mixing-temperature calculation and the
`scripts/sim_examples/gl_temp.py` example. The fix is a one-liner using the
existing gas-fraction property:

```python
w_g_res = (fl.f_g / (1 - fl.f_g)) * w_l_inflow
```

(the same expression already used in `_gas_and_liquid_flow_rate`,
`simulator.py:162`).

Related cleanups:
- `src/manywells/inflow.py:24` — docstring still references
  `FluidModel.gas_mass_flow_rate`.
- **Test gap:** no test exercises `w_lg > 0` (the only reference in
  `tests/test_simulator.py` is a negative-value validation check), which is why
  this slipped through. Add a gas-lift regression test, e.g. simulate with
  `w_lg=1.0` and assert the produced gas rate exceeds the no-lift case.

### 1.2 `water_fvf` has the wrong sign **[verified]**

**[done 2026-10-01 in Step 7]** `Bw = 1 - c_w (p - p_ref)` (PVT-WAT-2), with the
test updated; the form is a draft for Bjarne's sign-off.

**[plan: Step 7]** Fix it before its option is written into
`specs/model/pvt/water.md`, so the spec does not record the wrong sign.

`src/manywells/pvt/water.py:16`:

```python
Bw = 1 + c_w * (p - p_ref)
```

Compressibility means water *shrinks* with pressure, so Bw should *decrease*
with pressure: `Bw = 1 - c_w * (p - p_ref)` (or the more standard
`Bw = 1 / (1 + c_w (p - p_ref))`). The current tests codify the wrong sign
(`tests/test_black_oil.py:243-246`, "Water at 200 bar: Bw barely above 1.0") and
must be updated together with the fix. The function is currently unused by the
simulator, so the fix is safe.

### 1.3 Data-generation scripts are broken against the current API **[verified]**

**[done 2026-10-01 in Step 7]** `manywells.sampling` and `manywells.datasets`
port the procedure (`specs/sampling.md`), the generators are thin callers, and the
frozen dataclasses make a misspelt assignment raise.

**[plan: Step 7, item 7]** The sampler port. `scripts/data_generation_v2/` was
an untracked early draft and was deleted on 2026-09-30.

`scripts/data_generation_v2/well.py` (and `scripts/data_generation/well.py`)
still target the pre-refactor `WellProperties`:

- `Vogel(w_l_max, f_g)` → `TypeError` (Vogel now takes only `w_l_max`;
  the gas split moved to `FluidModel`).
- `wp.L`, `wp.D`, `wp.A` — geometry now lives in `wp.geometry`; setting these on
  the dataclass silently creates dead attributes (the well is simulated with the
  *default* 2000 m geometry, no error raised).
- `wp.rho_l`, `wp.cp_l`, `wp.R_s`, `wp.cp_g` — fluid properties now live in
  `wp.fluid` (`FluidModel`); these assignments are silently ignored too.
- `Well` still carries `pvt.GasProperties` / `pvt.LiquidProperties` objects that
  duplicate what `FluidModel` represents.

Since these scripts generated the published datasets, either (a) port them to
the new `FluidModel`/`WellGeometry` API, or (b) state clearly in their README /
module docstrings that they require the v1.0.0 tag. Option (a) is preferable for
an open-source project — the scripts are the reference implementation of the
dataset methodology. A cheap guard against future drift: add a smoke test that
constructs one sampled well and simulates a single point (marked `slow`).

The silent-attribute failure mode also suggests using `@dataclass(slots=True)`
(or `__slots__`) on `WellProperties`, `BoundaryConditions` and `FluidModel`, so
assigning to a nonexistent field raises immediately.

### 1.4 User-supplied `cpr` is silently overwritten

**[any time]**

`src/manywells/choke.py:49,58` — `ChokeModel` exposes `cpr` as a constructor
field, but `__post_init__` unconditionally does
`self.cpr = self.critical_pressure_ratio()`. A user passing
`ChokeModel(cpr=0.6)` gets 0.544 without warning. Either respect the given
value (`if self.cpr is None: ...`) or remove `cpr` from the constructor.

### 1.5 Choke docstring does not match the implementation

**[any time]**, or with `specs/model/choke.md` in Step 4, whichever comes first.

`src/manywells/choke.py:28` documents
`w = (K_c * sigma(u) / Phi) * sqrt(2 * rho * dp)` but the implementation
(`choke.py:112`) computes `K_c * chk * sqrt(2 * rho * dp / Phi)` — i.e. the
correction is `1/sqrt(Phi)`, not `1/Phi`. The implementation is the standard
form (Phi multiplies the *pressure-drop* term); fix the class docstring.

### 1.6 Smaller documentation/comment corrections

**[any time]**

- ~~`AGENTS.md` documents `uv run python scripts/sim_examples/<name>.py`, but that
  fails for examples that import `scripts.*`, such as `gl_temp.py`
  (`ModuleNotFoundError: No module named 'scripts'`), because running a file
  puts its own directory on `sys.path`, not the project root. `uv run python -m
  scripts.sim_examples.gl_temp` works. Either document `-m`, or drop the
  `scripts.*` import from the examples. *(Found 2026-09-30.)*~~ **[done
  2026-09-30 in Step 5]** `AGENTS.md` documents `-m`, and
  `tests/test_examples.py` runs the examples that way.
- `src/manywells/units.py:8` — module docstring claims "All public interfaces in
  the manywells package use SI units (Pa, ...)", but the simulator, choke,
  inflow and `FluidModel` methods all take **bar**. Meanwhile
  `FluidModel.p_sep`/`p_bubble` are in **Pa** (`fluid.py:55-57`) while every
  method argument on the same class is in bar — an easy footgun. At minimum fix
  the units.py docstring and call out the Pa fields prominently in `FluidModel`;
  better, accept `p_sep`/`p_bubble` in bar for consistency.
- `src/manywells/pvt/gas.py:64` — `gas_fvf` return doc says
  "(Sm3 at standard / m3 at reservoir conditions)" but the formula computes
  V_reservoir / V_standard (the first line of the docstring is correct).
- `NOTES.md` — the listed to-do is resolved: `_compute_left_boundary_state` now
  uses `fl.gas_density` (`simulator.py:354`) and `FluidModel` offers
  `gas_density`. Remove the note (or convert remaining ideas to GitHub issues).
- `src/manywells/geometry.py:97` — typo "nubmer".
- `src/manywells/simulator.py:250` and `cl_simulator.py:187` — stale internal
  comment "Used to generate dataset v6".
- `src/manywells/friction.py:6` — "Created 27 February 2026" with a 2024
  copyright line (check intent).
- `README.md` project tree is outdated: missing `src/manywells/pvt` and the
  root-level scripts.

## 2. Robustness and API quality

### 2.1 Replace `assert` validation with exceptions

**[done 2026-10-01 in Step 7]** for the inputs: `WellProperties`, `BoundaryConditions`,
the inflow, choke, friction and thermal models and `FluidModel` raise `ValueError`.
`liquid_mix` and `ca_double_sigmoid` still assert.

**[any time]**

`WellProperties`, `BoundaryConditions`, `ProductivityIndex`, `Vogel`,
`FixedFlowRate` and `ChokeModel` validate inputs with `assert`, which silently
disappears under `python -O`. `WellGeometry` and `BlackOilPVT` already raise
`ValueError` — make that the pattern everywhere.

### 2.2 Use `logging`/exception payloads instead of `print`

**[done 2026-10-01 in Step 7]** for the simulator: it never prints; failed starts go to
`logging.getLogger('manywells')`, and `NoOperatingPoint` carries the root set. The
calibration functions still print (§2.8).

**[any time]** The plan's evidence scripts and root-set search (Step 7, item 6)
currently have to capture stdout to silence failed starts.

`simulator.py:561-563` prints "Simulation failed" and the full solver stats to
stdout before raising `SimError`. For a library this should be a logger call
(or the stats attached to the exception: `raise SimError(..., stats=...)`),
so downstream users — e.g. the data-generation loop that *expects* failures —
can run quietly.

### 2.3 `isinstance`-based choke dispatch blocks extension

**[done 2026-10-01 in Step 7]** `ChokeModel.mass_flow_rate(u, p_s, s, A)`, with each
model's `density_and_multiplier`. Closed loop keeps its own dispatch, on its frozen base.

**[plan: Step 6]** Interface contracts. Settled in `specs/architecture.md`:
`ChokeModel.mass_flow_rate(u, p_s, s, A)` takes the wellhead state. Step 7
implements it.

`_right_boundary_eqs` (`simulator.py:248-255`) and
`cl_simulator.py:185-197` branch on `isinstance(wp.choke, ...)` and raise for
anything else, so a user cannot plug in their own `ChokeModel` subclass despite
the ABC inviting exactly that. Unify the interface — e.g.

```python
class ChokeModel(abc.ABC):
    @abc.abstractmethod
    def mass_flow_rate(self, u, p_in, p_out, alpha, rho_g, rho_l, x_g): ...
```

(or pass a small `ChokeState` value object) — and let each model pick what it
needs. The simulator then calls it polymorphically with no `isinstance`.

### 2.4 Hidden state `self._w_l_inflow`

**[done 2026-10-01 in Step 7]** `discretization.py` passes $w_\text{res}$ to every
point's rows.

**[plan: Step 6]** Interface contracts. Settled in `specs/architecture.md`: the
reservoir liquid rate is an explicit argument of every point's rows. Step 7
implements it.

`_differential_equations` depends on `self._w_l_inflow` being set as a side
effect of `_left_boundary_eqs` / `_compute_left_boundary_state`
(`simulator.py:210,360,329`). This call-order dependency is fragile (it is also
the kind of thing that breaks when methods are overridden, as in
`ClosedLoopWellSimulator`). Pass `w_l_inflow` explicitly, or compute it where
needed from `x_0` — the expression is cheap and symbolic anyway.

### 2.5 The simulator mutates the user's objects

**[done 2026-10-01 in Step 7]** for `SSDFSimulator`: frozen inputs, the default choke in
`WellProperties.__post_init__`, the operating point as parameters.

**[plan: Step 6]** The `SSDFSimulator` part, settled in `specs/architecture.md`:
frozen inputs, and the default choke set in `WellProperties.__post_init__`.
Step 7 implements it. The `ClosedLoopWellSimulator` part is **[out of v2]**.

- `SSDFSimulator.__init__` writes the default choke back into the *caller's*
  `WellProperties` (`simulator.py:126-127`).
- `ClosedLoopWellSimulator.simulate` overwrites `bc.u` and `bc.w_lg` with CasADi
  expressions (`cl_simulator.py:93-95,240-241`), bypassing
  `BoundaryConditions.__post_init__` validation and leaving the user's object
  holding symbolic values afterwards.

Keep defaults and decision variables internal to the simulator (e.g.
`self.choke = wp.choke or BernoulliChokeModel(...)`), and treat the input
dataclasses as immutable (consider `frozen=True` like `WellGeometry`).

### 2.6 `ClosedLoopWellSimulator` cleanups

**[out of v2]** Closed loop is out of scope for v2 (plan, Scope decisions).

- `import matplotlib.pyplot as plt` at module level (`cl_simulator.py:20`)
  drags a GUI dependency into library code and breaks headless use; it is only
  needed by the `__main__` demo.
- `_initial_guess` perturbs the guess with unseeded `np.random.normal()`
  (`cl_simulator.py:75`) — simulations are not reproducible. Accept an optional
  `numpy.random.Generator`/seed.
- Magic numbers: the combined control variable `t` packs choke (0–1) and gas
  lift (1–6) into one scalar with hardcoded bounds (`ubu[0] = 6.0`). Two named
  decision variables with explicit bounds would be far easier to follow.
- `simulate()` duplicates ~80 lines of the parent's NLP assembly. Extracting
  the shared "build variables + constraints" step in `SSDFSimulator` (e.g. a
  `_build_system()` returning `x, g`) would shrink the subclass to its actual
  delta: the objective, the extra decision variable, and the bounds.
- The feedback branch reads terminal-cell variables by negative indices
  (`x[-4]`, `x[-7]`, ...), which silently breaks if the state ordering or
  `dim_x` changes. Index via `self.dim_x` and the variable name list instead.

### 2.7 `SlipModel` parameters are not configurable

**[done 2026-10-01 in Step 7]** `SlipModel(C_0_slug=1.2)` works.

**[plan: Step 7]** Making them fields is part of specifying `slip.md`'s
interface. The calibration use case is **[after the plan]**.

`C_0_annular`, `C_0_slug`, `C_0_bubbly`, `v_inf_annular`
(`slip.py:122-127`) are class attributes without annotations, so the
`@dataclass` decorator ignores them — `SlipModel(C_0_slug=1.2)` raises.
Annotate them as fields so users can tune slip parameters per well (useful for
calibration, one of the stated use cases).

### 2.8 Calibration module consistency

**[after the plan]** Calibration is deferred (plan, Scope decisions).

`calibrate_inflow_model` / `calibrate_*_choke_model` raise bare `Exception`
on failure and `print` solver stats; align with `SimError`/logging (see 2.2).
They also rebuild a fresh Ipopt instance per call — fine for now, but if 3.x
becomes parametric (see 4.1), the calibration loops get the same benefit for
free by reusing a parameterized objective.

## 3. Testing

- ~~**Add a gas-lift test** (see 1.1) — the highest-value missing test.~~
  **[done 2026-09-30 in `8b05b18`]**
- ~~**[plan: Step 5]** **Smoke-test the examples**: a `slow`-marked test that runs each
  `scripts/sim_examples/*.py` headless (`matplotlib.use("Agg")`) would have
  caught both 1.1 and 1.3. The examples are the de-facto tutorial; broken
  examples are costly for an open-source project.~~ **[done 2026-09-30 in
  Step 5]** `tests/test_examples.py` runs each example in its own process with
  `MPLBACKEND=Agg`, in a temporary directory (`gl_temp.py` saves a figure to
  the working directory).
- **Fix the water FVF tests** along with 1.2.
- **[done 2026-10-01 in Step 7]** The develop vectors of THM-4 to THM-7 in
  `specs/model/thermal.md`, and `tests/test_thermal.py`.
  **[plan: Step 7]** **Energy-equation regression test** (as test vectors in
  `specs/model/thermal.md`): the thermal model
  (`docs/thermal_energy_modeling.md`) has heat loss, friction heating and
  adiabatic cooling terms; a test pinning wellhead temperature for a reference
  well would guard against sign/unit regressions in `_differential_equations`.
- Consider `pytest --doctest-modules` or a docs-examples check so docstring
  snippets (e.g. in `FluidModel`) stay valid.

## 4. Performance and architecture

### 4.1 Build the NLP once, parameterize the boundary conditions (largest win)

**[done 2026-10-01 in Step 7]** `build_system`, `IpoptSolver` and the march's
rootfinders are built once per well; the gain on the case set is in the plan's
Step 7 status.

**[plan: Step 6]** A structural change with an expected order-of-magnitude
gain, so it can pass principle 7; the gain is to be measured on the case set.
Settled in `specs/architecture.md` (`build_system` and `IpoptSolver` once per
well). Measured on two wells on 2026-09-30, the build is 97% of a warm
re-solve, a bound of about 40x; Step 7 implements it and repeats the
measurement on the case set.

`SSDFSimulator.simulate()` reconstructs the full symbolic system and a fresh
Ipopt instance on every call. For the package's main use cases — data
generation (thousands of solves per well with only `u`, `w_lg`, `p_r`, `p_s`,
`T_s` changing) and calibration (repeated solves in an outer loop) — almost all
of that work is identical between calls.

CasADi supports exactly this pattern: declare the boundary conditions as
parameters `p` of the NLP (`nlp = {'x': x, 'p': p, 'f': 0, 'g': g}`) and pass
values at solve time (`solver(x0=..., p=[u, w_lg, ...])`). Construction and
`nlpsol` instantiation then happen once per well instead of once per data
point. The same applies to `_simulate_cellwise`, which currently builds
`n_cells` separate `ca.Function` + `rootfinder` objects per call — one
parametric per-cell rootfinder (cell geometry `delta_md`, `cos_incl`,
`tvd_frac` as parameters) can be built once and reused, or rolled into a
`ca.Function.mapaccum` over cells.

Sketch of the resulting API:

```python
sim = SSDFSimulator(wp)              # builds parametric NLP once
x1 = sim.simulate(bc1)               # fast re-solve
x2 = sim.simulate(bc2, x_guess=x1)   # warm-started re-solve
```

This is a backwards-compatible refactor (keep the current constructor
signature working) and should give an order-of-magnitude speedup on dataset
generation, where construction time currently dominates.

### 4.2 Further solver-level options (after 4.1)

**[after the plan]** Each option adds machinery (a JIT flag, dual warm starts, a
Newton-then-Ipopt fallback), so under principle 7 each needs a large, measured
gain of its own.

- **JIT compilation**: `ca.nlpsol(..., {'jit': True, 'compiler': 'shell'})`
  compiles the constraint/Jacobian callbacks to C; worthwhile when one solver
  instance is reused many times (exactly the post-4.1 situation). Make it an
  opt-in flag since it needs a C compiler.
- **Warm starts**: with a parametric solver, pass the previous solution's
  primal *and dual* values (`lam_g0`, `lam_x0`) and set
  `ipopt.warm_start_init_point: 'yes'` when sweeping operating points (the
  data-generation scripts already reuse `x_guess`; duals are the missing half).
- **Pure rootfinding mode**: the problem is square (a feasibility NLP). When
  the variable bounds are not expected to be active, `ca.rootfinder` with
  `'newton'`/`'kinsol'` on the full system is much faster than Ipopt. Offering
  `method='newton'|'ipopt'` — try Newton first, fall back to Ipopt — would
  speed up the easy majority of solves while keeping robustness.
- The system Jacobian is block-tridiagonal; CasADi/Ipopt already exploit
  sparsity via MUMPS, so no action needed there, but it is worth a line in the
  docs since users may fear O(n³) scaling.

### 4.3 Library-level batch API

**[after the plan]** With the v2 dataset generation.

Dataset generation re-implements multiprocessing in scripts. A small
`manywells.batch` helper (simulate a list of `(WellProperties,
BoundaryConditions)` over a process pool, returning DataFrames and structured
failures) would make the published-dataset methodology reproducible from the
library itself and reduce script drift (see 1.3).

### 4.4 Vectorize `solution_as_df` regime classification

**[done 2026-10-01 in Step 7]** One classifier function mapped over the points
(`System.regime_probabilities`).

**[any time]**

`solution_as_df` (`simulator.py:594-601`) evaluates `flow_regime` per row in a
Python loop, building CasADi expressions each iteration. Build one
`ca.Function` for `classify_flow_regime` and `.map(n)` it over the grid. Minor,
but it is called after every simulation in the data-generation loop.

## 5. Open-source friendliness

- ~~**CI**: there is no `.github/workflows`. A minimal GitHub Actions job —
  `uv sync` + `uv run pytest -m "not slow"` on PRs, full suite (including
  `slow`) nightly or on release — protects contributors and would have flagged
  1.1/1.3 at the offending commit.~~ **[done 2026-09-22]**
  `.github/workflows/tests.yml` runs `uv sync --locked` and the full suite
  (it takes seconds, so no slow/nightly split) on Python 3.11–3.14 for pushes
  to `main`/`develop` and all pull requests.
- **Lint/format**: no ruff/black/mypy configuration. Adopting `ruff`
  (format + lint) with a pre-commit hook keeps external contributions uniform
  at near-zero maintenance cost.
- ~~**Loosen dependency pins**: `pyproject.toml` requires `numpy==1.24.3`,
  `pandas==1.5.3`, `python==3.11.*`. Exact pins belong in `uv.lock`;
  the package metadata should declare *compatible ranges* (e.g.
  `numpy>=1.24`, `python>=3.11`) so users can install manywells next to a
  modern stack. numpy 1.24/pandas 1.5 are EOL; testing against numpy 2.x and
  Python 3.12/3.13 is mostly a CI-matrix exercise once CI exists. Also consider
  moving `dev` from `[project.optional-dependencies]` to PEP 735
  `[dependency-groups]` so `uv sync` installs pytest by default.~~ **[done
  2026-09-22]** Ranges declared, `requires-python >= 3.11`, `dev` moved to
  `[dependency-groups]`, `uv.lock` upgraded (numpy 2.4/2.5, pandas 3.0, casadi
  3.8); full suite passes on Python 3.11 and 3.14.
- **[after the plan]** The plan makes v2.0.0 the major release with new
  datasets, so the API changes since v1.0.0 are released with it rather than as
  a separate tag. **Versioning/changelog**: the public API changed substantially since v1.0.0
  (`FluidModel`, `WellGeometry`, `Vogel` signature). Tag a v2.0.0 with a short
  CHANGELOG and a migration note (old → new construction snippets). This also
  cleanly resolves 1.3: "datasets were generated with v1.x; scripts in `main`
  target v2.x".
- **Docs**: `docs/` already contains good material (`simulate.md`,
  `thermal_energy_modeling.md`, `calibration.md`, ...) that the README never
  links to. Add a "Documentation" section to the README, and consider
  mkdocs-material + mkdocstrings on GitHub Pages to render both the guides and
  the (already thorough) docstrings.
- **Examples hygiene**: move the `__main__` demo blocks in `slip.py` and
  `cl_simulator.py` into `scripts/`/tests; library modules with executable
  tails confuse new readers and evade testing.
- **[after the plan]** One of the plan's decisions for v2.0.0.
  **License clarity**: CC-BY-NC 4.0 is an unusual choice for *code* (Creative
  Commons itself recommends against CC licenses for software, mainly because
  they do not address source/binary distinction or patents). Keeping CC-BY-NC
  for the datasets but adopting a software noncommercial license (e.g.
  PolyForm-Noncommercial) for the code would express the same intent with less
  legal ambiguity for users. Entirely optional — but worth a conscious
  decision.
- **Type-checking support**: the code is well annotated in places; adding a
  `py.typed` marker (and filling the gaps) lets users' type checkers see the
  annotations.
