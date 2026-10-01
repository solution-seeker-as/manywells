# ManyWells model specification

*Step 4 of `plans/manywells-v2-plan.md`. Owner: Bjarne Grimstad. Status: in use, 2026-09-30; `develop`'s options added in Step 7, 2026-10-01, a draft for Bjarne's sign-off. The `v1.0.0` configuration is complete, and Bjarne has ruled on every item of `specs/discrepancies.md`. `develop`'s model changes since v1.0.0 (`plans/develop_model_changes.md`) are options here, each with a feature spec in `specs/features/`.*

This directory specifies ManyWells' steady-state drift-flux model of gas–liquid flow in a well. It is the source of truth for the physics: every physics expression in the code traces to an equation ID here (principle 1), and no physics change goes in without a change to these files in the same PR. The spec says *what* the model is. Derivations and motivation stay in `docs/` and the paper, and the spec cites them.

The paper (Grimstad et al., Geoenergy Science and Engineering 257, 2026) documents v1.0.0. Where the paper and v1.0.0's code disagree, `specs/discrepancies.md` records the ruling, and these files follow it.

## Files

One file per model part, mirroring `src/manywells/`:

| File | Content |
|---|---|
| `nomenclature.md` | Symbols, units, the state vector, parameters and constants, for every file |
| `balances.md` | Mass, momentum and energy balances in continuous form |
| `discretization.md` | Grid, implicit Euler, the residual rows and their order |
| `solution.md` | Root set, stability label, operating point |
| `geometry.md` | Well trajectory and cross-section |
| `pvt/gas.md`, `pvt/oil.md`, `pvt/water.md`, `pvt/mixture.md` | Phase properties, liquid mixing, surface tension |
| `slip.md` | Drift-flux slip law, flow-regime classifier, rise velocities |
| `friction.md` | Viscous pressure gradient and friction factor |
| `thermal.md` | Heat loss, ambient temperature, inflow temperature |
| `inflow.md` | Inflow models and the bottomhole boundary rows |
| `choke.md` | Choke models, critical flow, the wellhead boundary row |
| `smoothing.md` | Smooth max, min and softmax, which are part of the model |
| `vectors/v1_rows.json` | Residual-row test vectors from v1.0.0 (`discretization.md`) |

Sampling of wells and operating points is not physics and has its own spec, `specs/sampling.md`.

## How the parts compose

A case is a well (parameters), an operating point (boundary conditions and controls) and a grid of N cells. The unknowns are the state at the N + 1 grid points (`nomenclature.md`). At each point the closure relations tie the state together: the gas law and the liquid density (`pvt/`) and the slip law (`slip.md`), which uses the flow-regime classifier, the rise velocities and the surface tension. Between neighbouring points, the discretized balances (`discretization.md`) carry mass, momentum and energy up the well, using friction (`friction.md`), gravity (`balances.md`) and heat loss (`thermal.md`). The bottomhole point takes its rates from the inflow model (`inflow.md`) and its temperature from the reservoir (`thermal.md`). The wellhead point must pass its rate through the choke (`choke.md`). Together these give 7(N + 1) equations in 7(N + 1) unknowns. Its roots, their stability labels and the operating point are defined in `solution.md`.

## Equation IDs

- **Form.** `<NAMESPACE>-<n>`, for example `CHK-2` or `PVT-OIL-3`. The namespace names the file: `BAL`, `DISC`, `SOL`, `GEO`, `PVT-GAS`, `PVT-OIL`, `PVT-WAT`, `PVT-MIX`, `SLIP`, `FRIC`, `THM`, `INF`, `CHK`, `SMO`, and `SMP` for `specs/sampling.md`.
- **Definition.** An ID is defined by a level-3 heading in its file, `### CHK-2 · Choke equation`, and nowhere else.
- **Stable.** IDs are never renumbered or reused. A new equation takes the next free number in its namespace. Removing an equation retires its ID: it moves to the list below and is never defined again.
- **Paper aliases.** Paper equation numbers, such as (11), are recorded in each file's coverage table. They are aliases, not IDs.
- **Code tags.** Code that implements an equation carries a comment `# spec: CHK-2` (Rust: `// spec: CHK-2`), listing several IDs with commas where one block implements several. A tag means the code implements the equation exactly as specified, in the configuration that uses it; code that generalizes an equation gets the ID of its own option.

Retired IDs: none.

## Options and configurations

Each component file lists its options: alternative equations for the same part of the model, each with its own ID. A **configuration** chooses one option per part and names a model version. There are two kinds of choice:

- **Model options** differ between configurations: the gas law, the oil model, friction, the energy terms, the geometry, mass transfer.
- **Well options** are inputs of every configuration, chosen per well: the inflow model, the choke model and the choke profile.

The `v1.0.0` configuration reproduces ManyWells v1.0.0, which generated the published datasets. It is the regression anchor: the verifier checks candidates in this configuration against reference root sets computed from v1.0.0 (`specs/verification.md`).

| Part | `v1.0.0` | `develop` default |
|---|---|---|
| Geometry | GEO-1 (vertical), GEO-2 | GEO-3, GEO-4 (any survey), GEO-2 |
| Mass transfer | BAL-3 (none) | BAL-10, PVT-OIL-13 (dissolved gas) |
| Gas | PVT-GAS-1 (ideal gas) | PVT-GAS-3 to PVT-GAS-6 (real gas, Papay) |
| Oil | PVT-OIL-1 (dead oil) | PVT-OIL-4 to PVT-OIL-9 (black oil, Vazquez–Beggs) |
| Liquid density | PVT-MIX-1 (constant) | PVT-MIX-6 (black oil and water) |
| Fluid parameters | $\rho_l$, $R_s$, $c_{pg}$, $c_{pl}$, $f_g$ | PVT-MIX-10 ($\rho_o$, $\rho_{g,\text{sc}}$, $\rho_w$, gas–oil and water–liquid ratios), PVT-GAS-6 |
| Surface tension | PVT-MIX-5 with PVT-OIL-3 | PVT-MIX-7 with PVT-OIL-3 and PVT-OIL-12 |
| Slip | SLIP-1 to SLIP-8 | SLIP-1 to SLIP-8 with SLIP-10, SLIP-11 (inclination) |
| Friction | FRIC-1 with FRIC-2 (fixed $f_D$) | FRIC-1, FRIC-3, FRIC-4, FRIC-6 (roughness, Chen), with PVT-MIX-8, PVT-MIX-9 |
| Energy | BAL-5 with THM-1, THM-2; THM-3 at the bottom | BAL-12 with THM-1, THM-4, THM-6, THM-7; THM-5 at the bottom |
| Momentum | BAL-4, BAL-6 | BAL-11, BAL-6 |
| Discretization | DISC-1 to DISC-6 | DISC-7 to DISC-11 |
| Choke | CHK-1 to CHK-4, CHK-11, CHK-12; CHK-11 extends v1.0.0, whose row is NaN where it applies, a region no root reaches | same |
| Solution | SOL-1 to SOL-6 | same |

`develop` implements the `v1.0.0` column through the general rows of the `develop` column, which reduce to it on a vertical uniform grid with dead oil, an ideal gas, a fixed $f_D$, the liquid-density surface tension and every thermal option off: DISC-11 gives DISC-6's rows as functions of the state, which the row vectors check (`discretization.md`). `manywells.configurations` builds a well in the `v1.0.0` configuration from v1.0.0's parameters (`v1_well`) and checks that a well is in a configuration (`check`); the `develop` configuration is the defaults of the dataclasses.

Well options in every configuration:

| Part | Options | Used by the published datasets |
|---|---|---|
| Inflow | INF-1 Vogel, INF-2 productivity index, INF-3 fixed rate (`v1.0.0` only), INF-8 fixed liquid rate (`develop` only); with INF-4 to INF-7 | INF-1 |
| Choke model | CHK-5 Simpson, CHK-6 Bernoulli | CHK-5 |
| Choke profile | CHK-7 linear, CHK-8 sigmoid, CHK-9 convex, CHK-10 concave | all four |

Step 7 added `develop`'s changes to the component files as options with new IDs; each has a feature spec in `specs/features/`.

## Component file template

Every component file has the same sections, in this order: purpose; interface (inputs, outputs, units, and whether the functions must accept CasADi symbols); equations, one `###` heading per ID; options, and which configuration uses each; safeguards (clipping, smoothing, bounds); sources; test vectors; coverage. `nomenclature.md`, `balances.md`, `discretization.md` and `solution.md` use the parts of the template that apply.

## Test vectors

Test vectors are inputs with expected outputs, checked against `develop` by `tests/test_spec_vectors.py`.

- **Component vectors** are tables in the component files. Each table has a `###` heading naming the IDs it exercises; output columns start with `→`. Values are in the units of the file's interface. A file has up to two blocks of them:
  - **From v1.0.0**, between `<!-- vectors:begin -->` and `<!-- vectors:end -->`, for the `v1.0.0` configuration. v1.0.0 is the reference, so they check that `develop` reproduces it.
  - **From `develop`**, between `<!-- vectors:begin develop -->` and `<!-- vectors:end develop -->`, for `develop`'s options. They pin each option as specified and implemented, so that a later change or a port (the Rust core, `specs/architecture.md`) that alters a value is caught. They do not show that the implementation is right; the sources and the property tests in the coverage tables do that.
- **Row vectors**, in `vectors/v1_rows.json`, give v1.0.0's residual rows at perturbed states of two wells, in v1.0.0's row order (`discretization.md`). `develop` must reproduce them in its `v1.0.0` configuration.
- **Regenerate** the v1.0.0 vectors from the v1.0.0 worktree (`verification/build/README.md` sets it up), from the repository root:

  ```console
  .worktrees/v1.0.0/.venv/bin/python specs/tools/make_v1_vectors.py
  ```

  The script rewrites the v1.0.0 blocks and the JSON file. v1.0.0 is frozen, so the output only changes if the script does.
- **Regenerate** the `develop` vectors with `uv run python specs/tools/make_develop_vectors.py`, which rewrites the `develop` blocks. Their values change only if an option changes, and such a change needs Bjarne's sign-off like any change to these files.

A vector whose equation has no implementation on `develop` is skipped with the reason. A vector that `develop` is known not to reproduce is a strict expected failure in `tests/test_spec_vectors.py`, with the reason.

## Coverage tables

Every component file ends with a coverage table, one row per ID defined in the file:

| Column | Content |
|---|---|
| ID | the equation ID |
| Paper | the paper's equation number or section, or "—" |
| v1.0.0 code | where v1.0.0 implements it (`manywells/<file>.py`, at the `v1.0.0` tag) |
| Checked by | one or more of `vectors` (a table in this file), `rows` (`vectors/v1_rows.json`), `verifier: <check>` (`specs/verification.md`), `property: <where>`, `spec-only: <reason>`, separated by semicolons; a semicolon never appears inside one of them |

`tests/test_spec_traceability.py` checks that every defined ID has exactly one coverage row with a valid "Checked by", that `vectors` and `rows` claims are backed by vectors, that every `# spec:` tag in `src/` names a defined ID, and that no retired ID is defined. Its last check is that every ID is tagged in code or marked spec-only; an ID of the `v1.0.0` configuration that `develop` implements through a more general equation is spec-only, with the general equation named.
