# Sampler redesign — proposed changes to decided specs

*Drafts for Bjarne's sign-off at the checkpoint (R8). Plans are drafts, not specs: nothing here changes a spec until he signs it off. The file belongs to R5; other steps add a section under the spec file they touch, marked with the step. R7 created the file on 2026-11-02 for its two sections, because it did not exist yet.*

Each section gives the spec file, the passage as it stands, the proposed text, and why.

*Applied (2026-11-06): feature 022's sections (`specs/model/inflow.md`, `solution.md`, `README.md`, `nomenclature.md`, the 022 parts of `validity.md`, `specs/architecture.md` and `specs/features/README.md`) went into `develop` with the feature (#38), and the `specs/model/pvt/table.md` section with K1's option A′ (#39); the specs as merged are the record. The sections for `specs/goals.md` and `specs/constitution.md` are still proposals, for R10.

*Applied (2026-11-10, R10): the sections for `specs/goals.md` (with I1's companion file for the logs), `specs/constitution.md` and the rest of `specs/model/validity.md` (the status line, decision 5, the inclination row, notes 9 and 10, items 1 and 3), with the second checkpoint's L3 in note 12 and feature 022, and D-107's water–gas ratio in INF-10's row. Decision 5's text leaves out the Gulf of Mexico's equivalent tangent, a value computed from real-well data, and points to `plans/evidence/stratified_flow_limit.py`. The specs as merged are the record.*

## `specs/goals.md`

### Compatibility → Datasets (R7)

*From the dataset design (`datasets-draft.md`; D-85, D-87) and Bjarne's ruling on D-52 (2026-11-02).*

**As it stands:**

> - `manywells-sol-2`, stationary, and `manywells-nsol-2`, non-stationary;
> - each has 10,000 wells with 200 samples per well;
> - each has two tables: one of wells, with their inputs and labels, and one of samples, holding each sample's operating point only.

**Proposed:**

> - `manywells-sol-2`, stationary, and `manywells-nsol-2`, non-stationary;
> - each has 10,000 wells with 200 samples per well, each sample an operating point: a draw without one is redrawn, and every redraw is logged with its inputs and its reason (Bjarne, 2026-11-02);
> - each has two main tables: one of wells, with their inputs and labels, and one of samples, holding each sample's inputs and its operating point only. The wells' surveys and PVT tables, which the wells table references, are tables of their own, and each dataset publishes its redraw log and a log of every drawn well (`docs/datasets.md`).

**Why.** A survey has up to about 150 stations and many wells may share a PVT table, so neither fits a well's row (D-87). The ruling asks for the redraws to be logged; publishing the log lets a user measure the bias the redraws cause (D-85). "Each sample's operating point only" stays: it means no unstable roots, and the samples also hold the inputs that vary within a well, as the Goals of the plan say.

**Also:** the header's list of dates gains "2026-11-02 (the samples are operating points, and redraws are logged, decided by Bjarne)", and the row of Decided elsewhere, "v2 datasets' columns and filters | the sampler redesign (`plans/sampler-redesign-plan.md`, R7)", becomes "v2 datasets' columns, filters and redraws | `docs/datasets.md` (the schema) and `specs/sampling.md` (the filters and redraws)" once R10 and R11 have written them.

### Direction for v2 → v2.0's model (R5)

*From Bjarne's rulings on horizontal drains (2026-10-30) and on the geometry (2026-11-02), and from feature 022 (D-60).*

**As it stands:**

> - v2.0 adds a gas-rate inflow for gas wells, a well option that `v2.0.0`'s check accepts;
> - the datasets draw black oils, volatile oils, gas condensates and gas wells, the volatile oils and condensates as PVT tables;
> - they draw no horizontal sections: a well's bottom section is at most highly deviated, short of the inclinations where stratified flow sets in (`specs/model/validity.md`, decision 5).

**Proposed:**

> - v2.0 adds a gas-rate inflow for gas wells, Forchheimer's law in the pressure squared (feature 022), a well option that `v2.0.0`'s check accepts;
> - the datasets draw black oils, volatile oils, gas condensates and gas wells, the volatile oils and condensates as PVT tables from a flash (feature 018), the wet and dry gases as black oil;
> - they draw no horizontal sections: every cell is at most 75° from vertical, and a well with a horizontal drain is modelled from its heel up, its inflow standing for the drain (`specs/model/validity.md`, decision 5).

**Why.** The first bullet names the form, once Bjarne approves feature 022. The third states the ruled limit and the heel approach in place of "highly deviated, short of the inclinations", which decision 5 now gives in numbers. The second says where the wet and dry gases go, which D-36 proposes and `validity.md`'s decision 4 already allows.

## `specs/constitution.md`

### Conventions → Dataset reproducibility (R7)

*From the dataset design (`datasets-draft.md`, The schema and Re-solving a sample; D-87, D-92).*

**As it stands:**

> Every dataset row carries the inputs needed to re-solve it: well, fluid, boundary conditions, grid and model configuration. Where a dataset keeps its wells in a table of their own, a row and its well's row carry them together. A generator records its seeds and the code version it ran.

**Proposed:**

> Every dataset row carries the inputs needed to re-solve it: well, fluid, boundary conditions, grid and model configuration. Where a dataset keeps its wells, or parts of them such as surveys and PVT tables, in tables of their own, a row and the rows it references carry them together. A generator records its seeds and the code version it ran.

**Why.** In the v2 datasets a sample's row, its well's row, the well's survey rows and, for a table fluid, its PVT table's rows carry the inputs together (`datasets-draft.md`, Re-solving a sample). The present text names only the well's row.

**Not proposed:** adding "and every draw it rejected, with the reason" to the constitution. The ruling on D-52 is a rule of the v2 datasets, so it belongs in `specs/goals.md` (above) and the v2 `specs/sampling.md`, not in a rule for every dataset.

## `specs/model/validity.md` (R5)

*From Bjarne's rulings of 2026-10-30 (D-10) and 2026-11-02 (D-9, D-57), and feature 022 (D-60, D-61). Each edit below is a change to a signed-off file.*

### The status line

**As it stands:**

> *Owner: Bjarne Grimstad. Status: signed off by Bjarne on 2026-10-29, the rules and decisions 1 to 4 (Decisions); decision 5, on horizontal sections, is open until the sampling redesign decides it with the sampling distribution. Any change to a rule or a range needs his sign-off.*

**Proposed:**

> *Owner: Bjarne Grimstad. Status: signed off by Bjarne on 2026-10-29, the rules and decisions 1 to 4 (Decisions); decision 5, on horizontal sections, ruled by him in the sampling redesign on 2026-10-30 and 2026-11-02. Any change to a rule or a range needs his sign-off.*

The Decisions section's opening, "Signed off by Bjarne, 2026-10-29, but for decision 5.", becomes "Signed off by Bjarne, 2026-10-29; decision 5 ruled in the sampling redesign, 2026-10-30 and 2026-11-02."

### Decision 5, closed

**As it stands:**

> 5. **Horizontal sections. Open.** Bjarne decides it with the sampling distribution, in the sampling redesign; until then the model extrapolates in them (note 9). The redesign either keeps drawing L-shaped wells with this known limit, or stops until a model with stratified flow exists, which would need its own feature spec.

**Proposed:**

> 5. **Horizontal sections** (Bjarne, 2026-10-30 and 2026-11-02, in the sampling redesign). The v2 sampler draws no horizontal cell. Every cell is at most 75° from vertical. A well with a horizontal drain is modelled from its heel up: the model's bottom is the heel, at the reservoir's true vertical depth, and the drain enters the inflow as a factor on the productivity, not as cells. By Taitel and Dukler's transition, scanned over oil and water at 10 to 200 bar in 62 to 157 mm tubing, 75° leaves stratified upward flow only below a superficial liquid velocity of 0.004 m/s, about 7 m³/d of liquid in 157 mm tubing; it admits the heel sections of the Gulf of Mexico's horizontal wells, whose equivalent tangent has a median of 67° (`plans/sampler-redesign/sources-wells-production.md`, §3). The model refuses no inclination: a horizontal cell that a user builds extrapolates as before (note 9). Stratified flow would need a feature spec of its own.

**Why.** It records the two rulings: the heel approach and the limit of about 75° (2026-10-30, D-10), and the geometry's classes, whose highly deviated class is half drains to the heel and half tangents capped at 75° (2026-11-02, D-9, D-57). The shares are sampling and go to the new `specs/sampling.md` (R10), not here. The rule stays Extrapolate, so that the `v1.0.0` configuration and a user's well refuse nothing they accept today (the constraints under Rules).

### The table's rows

- **Inclination** (Hasan et al.'s five regimes): the Rule column, "Extrapolate (note 9); horizontal sections are open (decision 5)", becomes "Extrapolate (note 9); the v2 sampler draws none beyond 75° (decision 5)". The Sampler and Outside columns stay the ported sampler's until R10 reruns `validity_ranges.py` on the new sampler.
- **Vogel inflow:** unchanged, but for its note (below).
- **New rows, from feature 022** (its Validity ranges): *Sampler* is the draft's (`gas_inflow_shapes.py`), to be rerun on the new sampler by R10.

  | Correlation | IDs | Input | Published range | Sampler | Outside | Rule |
  |---|---|---|---|---|---|---|
  | Forchheimer gas inflow, pressure-squared form | INF-10 | $p_r$, $p_0$ | $\mu_g Z$ nearly constant over $(p_0, p_r)$: $p_r$ below about 138 bar (2000 psia) (Golan and Whitson 1991, p. 147); above, the form is conservative at large drawdowns (their Example 2.6) | $p_r$ 60 / 278 / 652 bar (5th / 50th / 95th percentiles) | 79% of gas wells above 138 bar | Extrapolate (note 12) |
  | | | the reservoir | single-phase gas, stabilised flow; not a condensate bank or mobile water near the well (Golan and Whitson 1991, §2.4; Economides et al. 1994, §4-3) | condensates below the dew point in mid and late life; WGR up to 500 Sm³ per 10⁶ Sm³ | – | Extrapolate (note 12) |
  | Liquid with the gas | INF-11 | $f_g$ | a mass balance; $f_g > 0$ | – | – | Refuse $f_g = 0$ (`WellProperties`) |

  The non-Darcy coefficient $D$ is a parameter of the sampler's draw, not of the model, so it has no row; its range is the sampler's (`specs/sampling.md`, D-62).
- **PVT table:** unchanged. A condensate's table needs PVT-TAB-1's change (`specs/model/pvt/table.md`, below).

### Note 9

**As it stands (its last sentence):** "The model extrapolates; whether the redesign draws horizontal sections is decision 5, which is open."

**Proposed:** "The model extrapolates. The v2 sampler draws no horizontal cell (decision 5); the shares above are the ported sampler's."

### Note 10

**As it stands:**

> 10. **Vogel's inflow.** Vogel derived it by simulation for solution-gas drive reservoirs below the bubble point, with bubble points of 147 and 207 bar over 21 reservoir conditions. He gives an error of about 10% in most applications and at most about 20%, and warns that it does not hold where oil, gas and water all flow. Every sampled well uses it (SMP-8), wells with water and gas wells included. It is a well option, so the model extrapolates; which inflow a sampled fluid gets is the redesign's choice. The model has no inflow for gas wells.

**Proposed:**

> 10. **Vogel's inflow.** Vogel derived it by simulation for solution-gas drive reservoirs below the bubble point, with bubble points of 147 and 207 bar over 21 reservoir conditions. He gives an error of about 10% in most applications and at most about 20%, and warns that it does not hold where oil, gas and water all flow. It is a well option, so the model extrapolates. The ported sampler gives it to every well (SMP-8), wells with water and gas wells included. The v2 sampler gives it only to oil wells whose saturation pressure is at least 0.8 of $p_r$ and whose water cut is below 0.6, so that most of the drawdown lies below the bubble point; the other oil wells take a productivity index (INF-2), and gas and condensate wells Forchheimer's gas inflow (INF-10, feature 022).

**Why.** The last two sentences as they stand are overtaken: the redesign has chosen (D-22, D-60), and the model has a gas inflow once feature 022 is in. D-22's rule is sampling and is signed off with `specs/sampling.md`; the note follows it.

### Note 12, new

> 12. **Forchheimer's gas inflow.** In the pressure squared it holds where $\mu_g Z$ is about constant between $p_0$ and $p_r$, below about 138 bar. Above, the pseudo-pressure is the general form (Golan and Whitson 1991, Eq. 2.57), and the pressure-squared form with $\mu_g Z$ at $p_r$ gives 8.5% less rate at 40% drawdown and 14% less at the absolute open flow, at the median of the sampler draft's gas wells, and 28% less at the 5th percentile. The coefficients are a well's inputs, so the model extrapolates; the v2 sampler fits them to the pseudo-pressure over the well's drawdowns, which leaves 2% at the median and 6% at the 5th and 95th percentiles from 20% drawdown to the absolute open flow. Below the dew point a condensate drops liquid near the well, and a gas well with mobile water flows two phases there: neither is in the law, and the skin stands for both.

(Note 11 is the Reynolds number's; 12 is the next free.)

### Where the sampler leaves the ranges

- Item 1, gas–oil ratios: add at its end "The v2 sampler draws volatile oils and condensates as PVT tables, and wet and dry gases as black oil (decision 4)."
- Item 3, horizontal sections: "Horizontal sections in 24% of the cases (note 9)" gains "; the v2 sampler draws none (decision 5)".

## `specs/model/inflow.md` (R5)

*From feature 022 (D-60). The full delta, with the equations, is the feature spec's; here the edits in brief, for the checkpoint's list.*

- **New IDs:** INF-10 · Forchheimer gas inflow, and INF-11 · Reservoir liquid rate from the gas rate, as `specs/features/022-gas-rate-inflow.md`, Delta, gives them.
- **Interface:** the reservoir inflow's row gains INF-10 and INF-11; "In `develop` an inflow model returns the liquid rate only, `liquid_mass_flow_rate(p_0, p_r)`" becomes "`liquid_mass_flow_rate(p_0, p_r, fluid=None)`; the gas-rate inflow needs the fluid and returns INF-11 of its INF-10".
- **Options:** a row "Forchheimer gas rate | INF-10, INF-11 | `develop`, accepted by `v2.0.0`'s check; not in the `v1.0.0` configuration".
- **Safeguards:** the odd extension above $p_r$, and the refusals of $a \le 0$, $b < 0$, non-finite coefficients and a fluid with $f_g = 0$.
- **Sources, test vectors, coverage:** rows for INF-10 and INF-11 (the feature spec's Acceptance 3 and 4).

## `specs/model/solution.md` (R5)

*From feature 022.*

**As it stands (SOL-2):** "At every root $w_m > 0$, so the choke row (CHK-1) needs $p_N > p_c \ge p_s$, and the inflow (INF-1, INF-2) needs $p_0 < p_r$."

**Proposed:** "… and the inflow (INF-1, INF-2, INF-10) needs $p_0 < p_r$."

**Why.** INF-10 is zero at $p_r$ and negative above, as INF-1 and INF-2 are. SOL-3's argument for the label holds unchanged, since INF-10 is strictly decreasing in $p_0$.

## `specs/model/README.md` (R5)

*From feature 022.*

- **Well options in every configuration**, the Inflow row: "INF-1 Vogel, INF-2 productivity index, INF-3 fixed rate (`v1.0.0` only), INF-8 fixed liquid rate (`develop` only); with INF-4 to INF-7" becomes "INF-1 Vogel, INF-2 productivity index, INF-3 fixed rate (`v1.0.0` only), INF-8 fixed liquid rate and INF-10 Forchheimer gas rate (`develop`; INF-10 in `v2.0.0` too); with INF-4 to INF-7, and INF-11 for INF-10".
- **Independent checks:** "INF-10 | Golan and Whitson (1991), Example 2.4, and Economides et al. (1994), Example 4-7, worked examples" and "INF-11 | None: a mass balance"; the row "INF-2, INF-4 to INF-8" is unchanged.
- **Row vectors of `v2.0.0`:** "Between them the wells take every choice the configuration leaves to a well" gains "Forchheimer's gas inflow" in its list, from a fifth well in `vectors/v2_rows.json`; the four wells' values do not change. A change of the `v2.0.0` configuration's vectors needs Bjarne's sign-off, which the ruling in the Goals covers for this option.

## `specs/model/nomenclature.md` (R5)

*From feature 022.* New rows among the well's parameters: $a$ (`a`, bar² s/kg) and $b$ (`b`, bar² s²/kg²), Forchheimer's coefficients, $a > 0$, $b \ge 0$; and among the derived quantities $\Delta = p_r^2 - p_0^2$ (bar²). A sentence that a gas well's condensate–gas and water–gas ratios are held as `gor` $= 1/\text{CGR}$ and `wlr` $= \text{WGR}/(\text{WGR} + \text{CGR})$.

## `specs/model/pvt/table.md` (R5)

*From the table fluids' proof (`fluids-draft.md`, The table; D-64), with the two checks the condensate research proposes (`sources-condensate-tables.md`, option A′). Needs Bjarne's ruling at the checkpoint (`checkpoint.md`, K1). It changes feature 018's decided checks, so it may go through an amendment of 018 or a small feature spec of its own; the text is the same either way.*

### PVT-TAB-1, the checks

**As it stands:**

> - **Checks**, raising `ValueError` when the table is built: finite values; strictly increasing grids; $B_o > 0$, $Z > 0$, and $\mu_o, \mu_g, \sigma_{go} > 0$ where given; $R_{so}$ nondecreasing in $p$ at every $T$, to rounding (a fall of at most $10^{-12}\max(1, \lvert R_{so}\rvert)$, which PVT-TAB-7's arithmetic can make where $R_{so}$ is flat); and $R_{so} \ge 0$ unless the table has $\mu_o$ and $\sigma_{go}$, because the live-oil corrections PVT-OIL-11 and PVT-OIL-12 are undefined below $R_{so} = -17.8$ Sm³/Sm³ and untested below 0.

**Proposed:**

> - **Checks**, raising `ValueError` when the table is built: finite values; strictly increasing grids; $B_o > 0$, $Z > 0$, and $\mu_o, \mu_g, \sigma_{go} > 0$ where given; and $R_{so} \ge 0$ unless the table has $\mu_o$ and $\sigma_{go}$, because the live-oil corrections PVT-OIL-11 and PVT-OIL-12 are undefined below $R_{so} = -17.8$ Sm³/Sm³ and untested below 0; the oil phase's mass per Sm³ of stock-tank oil positive, $\rho_o + R_{so}\rho_{g,\text{sc}} > 0$, at every node; and the free gas's density, $p/Z$, strictly increasing in $p$ at every $T$. $R_{so}$ may fall as the pressure rises: below a gas condensate's dew point, liquid condenses as the pressure falls (retrograde condensation), so the oil phase holds more of the stream at the lower pressure.

**And in "Meaning, by the role in the balances"**, after "Above the fluid's bubble point …": "Above a dew point the stream is all vapour; a table from a flash keeps a trace of liquid there (PVT-TAB-6), so that $B_o > 0$."

### PVT-TAB-6, the vapour nodes

**Proposed addition** after "Where the stream is all liquid, the caller gives $\rho_V$ of the gas that would appear first (PVT-TAB-1).":

> Where the flash's liquid mass fraction $1 - X$ is below $\delta$, all vapour above a dew point or below a lower one, the caller sets $X = 1 - \delta$ and gives $\rho_L$ of the liquid at the nearest saturation point of the same temperature; `from_phase_split` takes $\delta$, $10^{-4}$ by default. The table then carries a liquid of mass $\delta$ times the stream's where the stream has none, which PVT-OIL-13 keeps as liquid: at $10^{-4}$, less than a water–gas ratio of 2 Sm³ per 10⁶ Sm³ carries.

**Why.** Two properties of every gas condensate that feature 018 did not meet, because its tables were oils. The proof's three condensate tables have $R_{so}$ falling with pressure at 176 to 316 pairs of neighbouring nodes, and $B_o = 0$ wherever the stream is all vapour without the trace. With both changes `FluidModel` builds on them, and the rows' expressions need nothing new: PVT-OIL-13 moves all but the trace of the oil into the gas, and PVT-MIX-6's liquid density is the water's and the trace's. The Rust core does not check the monotonicity today, so the change is in Python and the spec. The two new checks hold for every fluid: the oil phase cannot carry negative mass, and $p/Z$ rises with pressure at every node pair of all eight tables in hand, as Eclipse's PVDG requires of $B_g$ (`sources-condensate-tables.md`). No tool the research found checks the order of a net transfer such as $R_{so}$. **Alternatives** (D-64, `sources-condensate-tables.md`): $B_o = 0$ allowed with water required; the check kept for oils by a declared kind on the table; an $R_v$ form in the model; OLGA-style columns with $\rho_L$ in place of $B_o$.

**Also:** `tests/test_pvt_table.py`'s case that a falling $R_{so}$ raises becomes a case that a retrograde table builds; feature 018's ruling 10 (the check to rounding) lapses, and 018 gets a line pointing to the change.

## `specs/architecture.md` (R5)

*From feature 022.*

- **Components table**, the `InflowModel` row: "`liquid_mass_flow_rate(p_0, p_r)` | bar, bar | $w_\text{res}$, kg/s | Vogel, productivity index (INF-1, INF-2) · fixed rate" becomes "`liquid_mass_flow_rate(p_0, p_r, fluid=None)` | bar, bar, fluid | $w_\text{res}$, kg/s | Vogel, productivity index (INF-1, INF-2) · fixed rate · Forchheimer gas rate (INF-10, INF-11), which needs the fluid".
- **Discretization, No hidden state:** "$w_\text{res} = $ `inflow.liquid_mass_flow_rate(p_0, p_r)`" gains the fluid argument.
- **Rust core, design point 1:** the refusals gain "a gas-rate inflow on a fluid with no gas ($f_g = 0$, outside INF-11's range)".
- **Feature map:** a row "Gas-rate inflow (022) | `inflow`".

## `specs/features/README.md` (R5)

When Bjarne approves feature 022: a row "[022](022-gas-rate-inflow.md) | A gas-rate inflow for gas wells | inflow, solution, validity, README, nomenclature, architecture", and the paragraph's list of approved specs gains it.
