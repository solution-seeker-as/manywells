# ManyWells model specification

*Step 4 of `plans/manywells-v2-plan.md`. Owner: Bjarne Grimstad. Status: in use, 2026-09-30; `develop`'s options added in Step 7, 2026-10-01, and signed off by Bjarne on 2026-10-29. The `v1.0.0` configuration is complete, and Bjarne has ruled on every item of `specs/discrepancies.md`. `develop`'s model changes since v1.0.0 (`plans/develop_model_changes.md`) are options here, each with a feature spec in `specs/features/`.*

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
| `pvt/table.md` | The oil's and the gas's properties from a PVT table |
| `slip.md` | Drift-flux slip law, flow-regime classifier, rise velocities |
| `friction.md` | Viscous pressure gradient and friction factor |
| `thermal.md` | Heat loss, ambient temperature, inflow temperature |
| `inflow.md` | Inflow models and the bottomhole boundary rows |
| `choke.md` | Choke models, critical flow, the wellhead boundary row |
| `smoothing.md` | Smooth max, min and softmax, which are part of the model |
| `validity.md` | Each correlation's published range, the range the sampler reaches, and what the model does outside it |
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
| Gas | PVT-GAS-1 (ideal gas) | PVT-GAS-5, PVT-GAS-6, PVT-GAS-9, PVT-GAS-11 (real gas, Dranchuk–Abou-Kassem); Papay's PVT-GAS-3, PVT-GAS-4 as an option |
| Oil | PVT-OIL-1 (dead oil) | PVT-OIL-4 to PVT-OIL-9, PVT-OIL-15 (black oil, Vazquez–Beggs, capped at the gas–oil ratio) |
| Liquid density | PVT-MIX-1 (constant) | PVT-MIX-6 (black oil and water) |
| Fluid parameters | $\rho_l$, $R_s$, $c_{pg}$, $c_{pl}$, $f_g$ | PVT-MIX-10 ($\rho_o$, $\rho_{g,\text{sc}}$, $\rho_w$, gas–oil and water–liquid ratios), PVT-GAS-6 |
| Surface tension | PVT-MIX-5 with PVT-OIL-3 | PVT-MIX-7 with PVT-OIL-3 and PVT-OIL-12 |
| Slip | SLIP-1 to SLIP-8 | SLIP-1, SLIP-4, SLIP-5, SLIP-10, SLIP-12 to SLIP-16 (five regimes after Hasan et al.); the fitted classifier, SLIP-1 to SLIP-8 with SLIP-10, SLIP-11, as an option; with either, SLIP-18, the identity at its default |
| Friction | FRIC-1 with FRIC-2 (fixed $f_D$) | FRIC-1, FRIC-3, FRIC-4, FRIC-6 (roughness, Chen), with PVT-MIX-8, PVT-MIX-9 |
| Energy | BAL-5 with THM-1, THM-2; THM-3 at the bottom | BAL-13 with THM-1, THM-4, THM-6, THM-7, THM-8 (with PVT-GAS-10); THM-5 at the bottom |
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
| Lift-gas injection point | INF-5 at the bottomhole (`injection_md=None`); INF-9 at a depth, with THM-9 and DISC-12 (`develop` only; the `v1.0.0` configuration injects at the bottomhole) | the bottomhole |

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

### Independent checks

The `develop` vectors pin `develop`, and the Rust core implements the same spec, so neither shows that a correlation is implemented as published. The independent checks compare `develop`'s values with sources independent of it (2026-10-29, on `develop` at `f7f1218`, and rerun after the rulings below; `plans/model-lock-plan.md`, Step L3). Bjarne signed off the record of these checks, this section and the notes in the component files, on 2026-10-29.

- **Worked examples**, from the sources or from textbooks, compared at the precision printed: `tests/test_worked_examples.py`, on both backends, in every run of the test suite. A value that its own example's equation does not give at the example's inputs is the example's error; the test lists those and does not use them.
- **`fluids`** 1.3.1 (Caleb Bell, MIT licence), and **the sources' equations, transcribed**, where no worked example is in hand: `plans/evidence/independent_checks.py`, run by hand (`plans/evidence/README.md`, Independent checks). From then on the vectors carry the check: a change that alters a checked value fails `tests/test_spec_vectors.py`, and the vectors change only with Bjarne's sign-off, which reruns the script.

A value `develop` does not reproduce, after checking units and inputs, is a finding. It stays open until Bjarne rules on it, and is a strict expected failure in the test. The checks found two, and Bjarne ruled on both on 2026-10-29 to follow the source: PVT-GAS-5's coefficient of $\gamma_g$ in $p_{pc}$ is Sutton's 131.0, where `develop` had 131.07 (`pvt/gas.md`), and PVT-MIX-9 weights by the flowing mass fraction of gas, as Hasan et al. do, where `develop` took the in-situ one (`pvt/mixture.md`). Both backends now reproduce every example, and none is open. Bjarne signed off the two fixes the same day: the text and the measured effect of each ruling, PVT-MIX-9's fraction taken from the point's state with no guard (`pvt/mixture.md`, Safeguards), the regenerated vectors and table, and the re-pinned tests. Each component file's Test vectors section has the details.

Every ID of `develop`'s configuration, of the options Papay, Haaland and the fitted classifier, and of the well options, with its independent check:

| IDs | Independent check |
|---|---|
| GEO-2, GEO-3, GEO-4 | None: the trajectory and the cross-section, definitions |
| BAL-6 to BAL-11, BAL-13 | None: balance laws |
| DISC-7 to DISC-11 | None: ManyWells' discretization |
| SOL-1 to SOL-6 | None: ManyWells' definitions of the root set, the stability label and the operating point |
| SMO-1 to SMO-4 | None: ManyWells' smoothing |
| PVT-GAS-2, PVT-GAS-6 | None: definitions, of standard conditions and of the gas's gravity and gas constant |
| PVT-GAS-5 | Whitson and Brulé (2000), Appendix B, Problem 6, a worked example; Sutton (1985), Eqs. (14) and (15), transcribed. Ruled 2026-10-29: Sutton's 131.0 for 131.07 |
| PVT-GAS-7 | Ahmed (2006), Example 2-14, a worked example |
| PVT-GAS-9, PVT-GAS-11 | Dranchuk and Abou-Kassem (1975), Eqs. (2) and (3), transcribed; no worked example in hand |
| PVT-GAS-10 | A central difference of that transcription along an isobar |
| PVT-GAS-3, PVT-GAS-4 (Papay, an option) | None: Papay (1968) is not in hand, and no textbook in hand gives the correlation |
| PVT-OIL-2 | None: a definition |
| PVT-OIL-4, PVT-OIL-7, PVT-OIL-13, PVT-OIL-15 | None: the black oil's composition, its caps and its mass balance, ManyWells' own; the $B_o$ examples run through PVT-OIL-15's cap |
| PVT-OIL-5 | Ahmed (2006), Example 2-25, a worked example |
| PVT-OIL-6 | Ahmed (2006), Example 2-18, a worked example |
| PVT-OIL-8 | Ahmed (2006), Example 2-29, Method 2, a worked example |
| PVT-OIL-9 | None: the oil's mass over its volume |
| PVT-OIL-10, PVT-OIL-11 | Ahmed (2006), Example 2-35, a worked example |
| PVT-OIL-3, PVT-OIL-12 | Abdul-Majeed and Abu Al-Soof (2000), Eqs. (1) to (5), transcribed, and the data of their Table 1; no worked example in hand |
| PVT-WAT-1 | None: constants |
| PVT-WAT-3 | None: no source in hand (`validity.md`, note 5) |
| PVT-MIX-6, PVT-MIX-7, PVT-MIX-8, PVT-MIX-10 | None: mixing rules and definitions |
| PVT-MIX-9 | `fluids`' `gas_liquid_viscosity`, for the mixing rule; Hasan and Kabir (2018), Field Example 3.1, a worked example. Ruled 2026-10-29: the flowing mass fraction for the in-situ one |
| SLIP-1 | None: the drift-flux law |
| SLIP-4 | Hasan and Kabir (2018), Field Examples 3.1 and 4.1, worked examples |
| SLIP-5 | Hasan and Kabir (2018), Field Example 3.1, a worked example |
| SLIP-10 | None: the one example in hand, Hasan and Kabir's (2018) Field Example 4.1, prints a factor its own Eq. 4.33 does not give |
| SLIP-12 to SLIP-16 | Hasan and Kabir (2018), Field Examples 3.1 and 4.1 and Appendix 7-E, §7-E.2, worked examples: the flow regime, and $C_0$ in slug flow; not their void fractions (`slip.md`, Test vectors) |
| SLIP-18 | None: ManyWells' own (`specs/features/020-profile-scale.md`) |
| SLIP-2, SLIP-3, SLIP-6 to SLIP-8, SLIP-11 (the fitted classifier, an option) | None: v1.0.0's own fit to SLIP-9; the `v1.0.0` vectors check that `develop` reproduces it |
| FRIC-1, FRIC-3, FRIC-6 | None: Darcy–Weisbach, the Reynolds number and a blend |
| FRIC-4 | `fluids`' `Chen_1979`; Chen (1979), Table I, the largest deviation from his form of Colebrook's equation |
| FRIC-5 (Haaland, an option) | `fluids`' `Haaland`; Haaland (1983), the stated 1.5% from Colebrook–White |
| THM-1, THM-4 to THM-8 | None: the energy balance's terms and its bottom boundary, derived in `docs/thermal_energy_modeling.md`; THM-8's factor is PVT-GAS-10 |
| CHK-1 to CHK-3, CHK-6 to CHK-12 | None: ManyWells' wellhead boundary, paper (11) to (14) |
| CHK-4 | `fluids`' `P_critical_flow` |
| CHK-5 | None: Simpson, Rooney and Grattan (1983) is not in hand |
| INF-1 | Guo et al. (2007), Example Problem 3.2, a worked example |
| INF-2, INF-4 to INF-8 | None: definitions and balances |

## Coverage tables

Every component file ends with a coverage table, one row per ID defined in the file:

| Column | Content |
|---|---|
| ID | the equation ID |
| Paper | the paper's equation number or section, or "—" |
| v1.0.0 code | where v1.0.0 implements it (`manywells/<file>.py`, at the `v1.0.0` tag) |
| Checked by | one or more of `vectors` (a table in this file), `rows` (`vectors/v1_rows.json`), `verifier: <check>` (`specs/verification.md`), `property: <where>`, `spec-only: <reason>`, separated by semicolons; a semicolon never appears inside one of them |

`tests/test_spec_traceability.py` checks that every defined ID has exactly one coverage row with a valid "Checked by", that `vectors` and `rows` claims are backed by vectors, that every `# spec:` tag in `src/` names a defined ID, and that no retired ID is defined. Its last check is that every ID is tagged in code or marked spec-only; an ID of the `v1.0.0` configuration that `develop` implements through a more general equation is spec-only, with the general equation named.
