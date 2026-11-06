# 022 · A gas-rate inflow for gas wells

*Feature spec, draft of 2026-11-02 (Step R5 of `plans/sampler-redesign-plan.md`). Status: approved by Bjarne on 2026-11-05 at the checkpoint (R8, item J1), as drafted: Forchheimer's law in the pressure squared, with design choices 1 to 6 as proposed (`plans/sampler-redesign/decisions.md`, D-60); nothing is implemented. Decided before: Bjarne's Goals of 2026-10-30, that gas wells flow in through a new gas-rate inflow and that `v2.0.0`'s check accepts its option before the release. How the sampler fits $a$ and $b$ (D-61) and draws the non-Darcy coefficient (D-62) are sampling decisions, ruled separately. The measurements are draw-only, on the sampler draft's wells (`plans/evidence/gas_inflow_shapes.py`); no well was solved.*

## Motivation

The v2 datasets draw gas condensates and wet and dry gases, a third of the wells (the Goals; `plans/sampler-redesign/sampling-draft.md`, Fluid class). The model has no inflow for them:

- **Vogel's inflow (INF-1)** was derived for solution-gas drive oil reservoirs below the bubble point, not for gas reservoirs and not where oil, gas and water all flow (`specs/model/validity.md`, note 10). v1's sampler gives it to every well (SMP-8), gas wells included.
- **INF-4 ties the gas to the liquid.** Every inflow model returns the liquid rate, and the gas follows from the fluid's gas mass fraction. A gas well's rate is set by the gas, through a law in the pressure squared with a non-Darcy term (Golan and Whitson 1991, §2.4; Economides et al. 1994, §4-3 and §4-4); its liquid is a small share that comes with the gas.
- **The interim mapping errs by up to a factor 2.** Until this feature exists, the draft maps a gas well to a productivity index that gives the design rate's liquid at the design drawdown, so its gas rate is linear in the drawdown (`sampling-draft.md`, Mapping). On R3's 703 gas and condensate wells (seed 2026), against the reference below, it gives 0.57 times the rate at 5% drawdown and 2.05 times the absolute open flow (AOF), at the median; 0.38 and 3.07 at the 5th and 95th percentiles (Measurements).

## The form: Forchheimer's law

### The equation

Forchheimer's law for stabilised radial flow of a real gas (Golan and Whitson 1991, Eqs. 2.44 and 2.51 to 2.53; Economides et al. 1994, Eqs. 4-32 and 4-33):

$$p_r^2 - p_{wf}^2 = a\thinspace q + b\thinspace q^2,$$

with $a$ the Darcy term, from $kh$, the skin and $\mu_g Z$, and $b$ the non-Darcy term, from the non-Darcy coefficient $D$. The pressure-squared form holds where $\mu_g Z$ is nearly constant between $p_{wf}$ and $p_r$: below about 138 bar (2000 psia) (Golan and Whitson 1991, p. 147; Economides et al. 1994, Eq. 4-52). At higher pressures the general form is the real-gas pseudo-pressure $m(p) = 2\int_0^p p/(\mu_g Z)\thinspace dp$ (Al-Hussainy et al. 1966, as Golan and Whitson 1991, Eqs. 2.47 to 2.49 and 2.57, and Economides et al. 1994, Eqs. 4-51 and 4-54 give it), in which $a$ and $b$ are constants of the reservoir; and above about 207 bar (3000 psia) for both pressures, a form in $p$ (Golan and Whitson 1991, Eqs. 2.45, 2.46 and 2.54).

### Against the backpressure law

The backpressure law of Rawlins and Schellhardt (1936), $q = C\thinspace(p_r^2 - p_{wf}^2)^n$, was found from several hundred multirate tests; $n$ lies between 0.5 and 1 (Golan and Whitson 1991, p. 132; Economides et al. 1994, Eq. 4-26; Ahmed 2006, Eq. 8-20). The sources weigh them so:

- **Theory.** Forchheimer's law follows from Darcy's law with a velocity-squared term (Forchheimer 1901; Golan and Whitson 1991, Eqs. 2.36 and 2.37), and its $a$ and $b$ have expressions in the rock and fluid properties. The backpressure law came from field observation; why $n < 1$ was not known when it was proposed, and $C$ has no expression in the reservoir's properties for $n < 1$ (Golan and Whitson 1991, p. 132).
- **Shape.** Forchheimer's law has a local exponent $n = (a + bq)/(a + 2bq)$ that runs from 1 at low rates, where flow is laminar, to 0.5 at high rates. A backpressure line with one $n$ is straight on log–log paper only over part of a test: Golan and Whitson's Medicine Lodge well (Example 2.4, Fig. E2.4b, p. 140) gives $n$ = 0.766 from 5 MMscf/d to its AOF, and its curve bends below.
- **For the model.** The backpressure law's slope $dq/dp_{wf}$ is infinite at $p_{wf} = p_r$ for $n < 1$, and $(p_r^2 - p_{wf}^2)^n$ is not real above $p_r$. The CasADi rows would have an infinite derivative at zero drawdown, and the Rust core's search, which brackets roots up to $10^{-6}$ bar below $p_r$, would meet an inflow whose slope grows without bound there. It would also label a root very near $p_r$ stable by the inflow's slope alone, against the physics of laminar flow at low rates. Forchheimer's law has the finite slope $2p_r/a$ at $p_r$ and extends smoothly above it (INF-10).
- **The evidence script.** On R3's gas wells, both fitted to the same four-point test at 5% to 20% drawdown, the backpressure law extrapolates to the AOF within −14% to +11% (5th to 95th percentile) of the pseudo-pressure reference, Forchheimer's law in $p^2$ within −22% to 0%. The backpressure law follows the reference's high-pressure flattening better, because $n < 1$ bends its curve the same way; Forchheimer's law in $p^2$ is conservative there, as Golan and Whitson's Example 2.6 (p. 147) shows: 11.7 against 13.1 MMscf/d of AOF at 3150 psia.

**Proposed: Forchheimer's law in the pressure squared** (INF-10), with constant $a$ and $b$, the well's parameters like $k_l$. It is the physically based form, it has a finite slope at $p_r$, and R3's draws already give its $a$ and $b$ (D-23). Its error at high pressure is a matter of the coefficients, which a sampler or a user fits to the pressure range of the well (design choice 2), not of the law.

## Design choices

1. **The law.** *Proposed:* Forchheimer (above). *Alternative:* the backpressure law, which needs a safeguard at $p_r$ and a cap on its slope, both new model choices with no source.
2. **The pressure function.** *Proposed:* $p^2$, with constant $a$ and $b$. A sampler that wants the pseudo-pressure's shape fits $a$ and $b$ to it over the well's drawdowns; on R3's wells a least-squares fit over $p_{wf}/p_r$ from 0.2 to 0.95 and the AOF stays within 2% of the reference at the median and within 6% at the 5th and 95th percentiles, but for 5% drawdown, where it is 4% low at the median and 14% low at the 5th percentile (Measurements). *Alternatives:*
   - the form in $p$, $p_r - p_{wf} = aq + bq^2$, for wells whose pressures all lie above 207 bar: one more option for a range that the fitted $p^2$ form covers within a few per cent;
   - the pseudo-pressure, tabulated as $m(p)$ at $T_r$ on the inflow and interpolated linearly in both backends: exact, with $a$ and $b$ constants of the reservoir over a well's life, so that `nsol-2` would not recompute them as $p_r$ falls. It needs the fluid's $\mu_g$ and $Z$ at $T_r$, which the inflow does not have, so the table would be the maker's, like a PVT table. A later option if calibration needs it;
   - $m(p)$ from the fluid inside the rows: an integral of $\mu_g Z$ over $(p_0, p_r)$ at each trial state, a quadrature of DAK's Newton solve in the CasADi graph. Rejected: costly, and the inflow would depend on $T_r$, a boundary condition.
3. **The rate.** *Proposed:* $w_{g,\text{res}}$, the reservoir gas's mass rate at standard conditions in kg/s, the same quantity INF-4 gives for the liquid inflows. The liquid follows from it by INF-11. The model stays in kg/s and the inflow needs no fluid property to evaluate its law. *Alternatives:*
   - the standard volume rate in Sm³/d, which a test reports: the inflow would need $\rho_{g,\text{sc}}$, the fluid's, so the two would have to agree, and calibration replaces the fluid row by row;
   - the wellstream (all the hydrocarbon, condensate included) or the total mass with the water: physically the wellstream is what flows in the reservoir, but it needs a new fraction of the fluid, and water flows as a phase of its own, which Forchheimer's law does not describe. Units, below, gives the conversion from the wellstream.
4. **The interface.** *Proposed:* `InflowModel.liquid_mass_flow_rate(p_0, p_r, fluid=None)` gains the fluid as an optional argument. Vogel, the productivity index and the fixed rate ignore it; the gas-rate inflow requires it and returns INF-11 of INF-10. Every caller passes the well's fluid (`discretization.py`, `thermal.py` through $w_\text{res}$, `tests/backend_cases.py`'s overlays). The callers that pass two arguments keep working for the liquid inflows. *Alternatives:* a second method, `gas_mass_flow_rate`, and a branch in `discretization.py` on the inflow's kind, which puts a special case in the rows' assembly; or $\rho_{g,\text{sc}}$ and $f_g$ stored on the inflow, which duplicates the fluid.
5. **Above $p_r$.** *Proposed:* INF-10's odd extension, so that the inflow is defined, finite and strictly decreasing in $p_0$ everywhere (Safeguards). *Alternative:* none, as for INF-1 and INF-2; but INF-10's root is not real above $p_0^2 = p_r^2 + a^2/(4b)$, which on R3's wells lies 2.7 bar above $p_r$ at the median and 0.1 bar at the 5th percentile, and a trial state of Ipopt within its bound relaxation, or a central difference at a root next to $p_r$, could reach it.
6. **The name.** *Proposed:* `Forchheimer(a, b)`, after its law as `Vogel` is, in `inflow.py`; the Rust core's `Inflow::Forchheimer { a, b }`. The sampler's label `inflow = gas_rate` (`sampling-draft.md`, Labels) becomes `forchheimer`, as `vogel` names `Vogel`.

## Delta

### `specs/model/inflow.md`

The next free numbers are INF-10 and INF-11 (INF-1 to INF-9 are defined; none is retired; no branch or worktree defines more, checked on 2026-11-02).

- **INF-10 · Forchheimer gas inflow.** With $\Delta = p_r^2 - p_0^2$ (bar²), the reservoir gas's mass rate at standard conditions is the root of $\Delta = a\thinspace w_{g,\text{res}} + b\thinspace w_{g,\text{res}}^2$ that is zero at $\Delta = 0$:

  $$w_{g,\text{res}} = \frac{2\Delta}{a + \sqrt{a^2 + 4b\thinspace\lvert\Delta\rvert}}$$

  with $a > 0$ in bar² s/kg and $b \ge 0$ in bar² s²/kg². For $\Delta \ge 0$ this is the positive root of the quadratic, $(-a + \sqrt{a^2 + 4b\Delta})/(2b)$, written without the cancellation that form has at small $b\Delta$, and without its division by zero at $b = 0$, where it is $\Delta/a$. For $\Delta < 0$, above $p_r$, it is the odd extension (Safeguards).
- **INF-11 · Reservoir liquid rate from the gas rate.** For INF-10, the liquid comes with the gas in the fluid's ratios at standard conditions, INF-4 read backwards:

  $$w_\text{res} = \frac{1 - f_g}{f_g}\thinspace w_{g,\text{res}} = \frac{\rho_o + \rho_w\thinspace\text{WOR}}{\rho_{g,\text{sc}} R_{go}}\thinspace w_{g,\text{res}},$$

  with $f_g$ of PVT-MIX-10 and $\text{WOR} = \text{wlr}/(1 - \text{wlr})$. In a gas well's terms, with the condensate–gas ratio $\text{CGR} = 1/R_{go}$ and the water–gas ratio $\text{WGR} = \text{WOR}/R_{go}$ (both Sm³/Sm³), $w_\text{res} = (\rho_o\thinspace\text{CGR} + \rho_w\thinspace\text{WGR})\thinspace w_{g,\text{res}}/\rho_{g,\text{sc}}$. It needs $f_g > 0$, a fluid with gas. INF-4 then gives back $w_{g,\text{res}}$ exactly, and INF-5, PVT-OIL-13 and INF-9 split the stream at each point as for any inflow.
- **Interface.** The reservoir inflow's row gains INF-10 and INF-11; the inflow takes the fluid (design choice 4). The interface paragraph says that the gas-rate inflow returns $w_\text{res}$ by INF-11, so that every row downstream sees the same $w_\text{res}$.
- **Options.** A row: Forchheimer gas rate, INF-10, INF-11, used by `develop` and accepted by `v2.0.0`'s check; not in the `v1.0.0` configuration.
- **Safeguards.** Above $p_r$ ($\Delta < 0$) INF-10 is the odd extension, $-w_{g,\text{res}}(\lvert\Delta\rvert)$: defined, finite and strictly increasing in $\Delta$ everywhere, with the slope $1/a$ at $\Delta = 0$ from both sides. Its second derivative jumps at $p_0 = p_r$, where no root lies (SOL-2). `develop` raises `ValueError` for $a \le 0$, $b < 0$ or a non-finite coefficient (`Forchheimer.__post_init__`), and for a gas-rate inflow on a fluid with $f_g = 0$ (`WellProperties.__post_init__`, which holds both).
- **Sources.** Forchheimer (1901); Golan and Whitson (1991), §2.4, Eqs. 2.36, 2.37, 2.44 and 2.51 to 2.62, Examples 2.4 and 2.6; Economides et al. (1994), §4-3 and §4-4, Eqs. 4-23 to 4-34 and 4-51 to 4-54, Examples 4-7 and 4-8; Rawlins and Schellhardt (1936), for the backpressure law not taken.
- **Test vectors and coverage:** below.

### Units, and the conversion from a standard gas rate

The model's units are bar, K and kg/s. A test or a draw gives $a_q$ in bar² per (Sm³/d) and $b_q$ in bar² per (Sm³/d)² of separator gas, $q$ in Sm³/d. With $w_{g,\text{res}} = \rho_{g,\text{sc}}\thinspace q/86\thinspace 400$:

$$a = \frac{86\thinspace 400}{\rho_{g,\text{sc}}}\thinspace a_q, \qquad b = \left(\frac{86\thinspace 400}{\rho_{g,\text{sc}}}\right)^2 b_q .$$

From field units, 1 psi² = 4.7537·10⁻³ bar² and 1 Mscf ≈ 28.3 Sm³ (the 0.2% between 60 °F and 15 °C left out). `Forchheimer.from_gas_rate(a_q, b_q, rho_g)` does the conversion; the fluid's `rho_g` is the one to pass.

From reservoir properties, in SI with $p$ in Pa and $q_{sc}$ in Sm³/s, Darcy's law with Forchheimer's term in the pseudo-steady state gives

$$a_q = \frac{p_\text{sc} T_r\thinspace\mu_g Z\thinspace(\ln(r_e/r_w) - 0.75 + s)}{\pi k h\thinspace T_\text{sc}}, \qquad b_q = \frac{a_q D}{\ln(r_e/r_w) - 0.75 + s},$$

converted to bar² and Sm³/d (Golan and Whitson 1991, Eqs. 2.52 and 2.53, in field units; `plans/evidence/sampler_draft_draws.py`, `forchheimer`). $q$ there is the reservoir gas's standard volume, the wellstream. For a wet gas or a condensate the separator gas is less: $q_\text{ws} = c\thinspace q_g$ with $c = 1 + C_{og}/R_{go}$ and the condensate's gas equivalent $C_{og} = 133\thinspace 000\thinspace\gamma_o/M_o$ scf/STB, $M_o \approx 6084/(\gamma_\text{API} - 5.9)$ (Whitson and Brulé 2000, Eqs. 3.73b and 3.73c), so $a_q = c\thinspace a_\text{ws}$ and $b_q = c^2 b_\text{ws}$. On R3's draws $c$ is 1.01 / 1.05 / 1.14 for condensates (5th / 50th / 95th percentiles) and at most 1.01 for wet and dry gas. $\mu_g$ and $Z$ are the reservoir gas's, at the wellstream gravity (Whitson and Brulé 2000, after Eq. 3.35), or the table's gas above the dew point for a table fluid.

### The liquid with the gas

A gas well's condensate and water come in the fluid's ratios, as INF-4 has them for an oil well: `FluidModel(gor=1/CGR, wlr=WGR/(WGR + CGR))`, which the sampler draft already maps from its yield and WGR (`sampling-draft.md`, Reservoir state). INF-11 is INF-4 solved for the liquid, so a gas well and an oil well with the same fluid and the same $w_\text{res}$ have the same rows. Inside the well the black oil or the table splits the stream at each point: a wet gas's black oil has no oil in its gas, so all its condensate is liquid from the bottomhole up (decision 4 of `validity.md`); a condensate's table vaporises it above the dew point, by $R_{so} < 0$ (PVT-TAB-6), which today's PVT-TAB-1 refuses (`plans/sampler-redesign/fluids-draft.md`, The table).

### The CasADi form

INF-10 as written is smooth and symbolic-friendly in $p_0$: a square root whose argument is at least $a^2 > 0$, a division by $a + \sqrt{\cdot} \ge 2a > 0$, and `casadi.fabs`, which is no Python branch. Its derivative,

$$\frac{dw_{g,\text{res}}}{dp_0} = -\frac{2p_0}{\sqrt{a^2 + 4b\thinspace\lvert\Delta\rvert}},$$

is finite everywhere, $-2p_r/a$ at $p_0 = p_r$, and negative for every $p_0 > 0$. $f_g$ is a constant of the fluid, so INF-11 is linear. No smoothing function of `ca_functions.py` is needed. The Rust core evaluates the same expression on floats.

### `specs/model/solution.md` and the root search

- **SOL-2:** "the inflow (INF-1, INF-2) needs $p_0 < p_r$" becomes "(INF-1, INF-2, INF-10)".
- **SOL-3, the label.** Its argument, that a lower $p_0$ means a higher rate from the reservoir, holds for INF-10: $w_{g,\text{res}}$, and with it $w_\text{res}$ and the total rate, is strictly decreasing in $p_0$ on $(0, \infty)$. The label is computed as for any inflow.
- **The Rust core** (`specs/architecture.md`, Rust core, design point 3). Given $p_0$, INF-10 and INF-11 give $w_{g,\text{res}}$ and $w_\text{res}$, and the march is unchanged. The scan, the ladder of drawdowns down to $10^{-6}$ bar below $p_r$, Brent and the central difference for the slope need nothing new: the inflow is $C^\infty$ on $p_0 < p_r$ and $C^1$ through it.
- **Where $R(p_0)$ is not monotone.** The inflow side of $R = w_m - w_c$ falls monotonically with $p_0$. The tubing side does not in general, as for oil wells:
  - at high gas rates friction dominates, the wellhead pressure rises with $p_0$, $R$ falls, and the well has one stable root;
  - at low gas rates the liquid held up in the tubing grows as the rate falls, the wellhead pressure can fall as $p_0$ rises, and $R$ can rise again: liquid loading, with a stable root at the higher rate and an unstable one at the lower, the gas well's counterpart of SOL-7's two roots;
  - near a choked gas wellhead the Joule–Thomson term can leave a cell's energy row without a root at some trial pressures (`specs/features/016-joule-thomson.md`), where the core's scan already leaves out non-finite samples and refines the edges.

  Nothing in the scan assumes $R$ monotone; R6 measures the share of gas-well samples with two roots and with non-finite regions of $R$.

### `specs/model/README.md`

The Well options table's Inflow row: "INF-1 Vogel, INF-2 productivity index, INF-3 fixed rate (`v1.0.0` only), INF-8 fixed liquid rate and INF-10 Forchheimer gas rate (`develop` and `v2.0.0`); with INF-4 to INF-7, and INF-11 for INF-10". The independent-checks table: "INF-10 | Golan and Whitson (1991), Example 2.4, the Medicine Lodge well's IPR (Table E2.4), and Economides et al. (1994), Example 4-7, its AOF (Eq. 4-39), worked examples"; "INF-11 | None: a mass balance". `vectors/v2_rows.json` gains a fifth well with the gas-rate inflow; the four wells' values do not change.

### Other files

- `specs/architecture.md`: the Components table's `InflowModel` row, `liquid_mass_flow_rate(p_0, p_r, fluid=None)`, with Forchheimer (INF-10, INF-11) among the options; design point 1's list of refusals gains a gas-rate inflow on a fluid without gas; the feature map gains "Gas-rate inflow (022) | `inflow`".
- `specs/model/nomenclature.md`: $a$ (`a`, bar² s/kg) and $b$ (`b`, bar² s²/kg²), Forchheimer's coefficients; $\Delta$; CGR and WGR, which the fluid holds as `gor` and `wlr`.
- `specs/model/validity.md`: the rows below, and note 10 (`plans/sampler-redesign/spec-changes.md`).
- `specs/calibration.md`: unchanged; freeing $a$ and $b$ in a calibration is out of scope.
- `docs/simulate.md`: a gas well with the Forchheimer inflow, its conversion from a test in Sm³/d, and an example in `scripts/sim_examples/` (`gas_well.py`).

### Code

| Part | Python | Rust |
|---|---|---|
| the inflow | `inflow.py`: `Forchheimer(a, b)`, frozen, validated in `__post_init__`; `liquid_mass_flow_rate(p, p_r, fluid)` returns INF-11 of INF-10; `gas_mass_flow_rate(p, p_r)` is INF-10; `from_gas_rate` | `inflow.rs`: `Inflow::Forchheimer { a, b }`; `liquid_rate(p, p_r, f_g)` |
| the fluid argument | `InflowModel.liquid_mass_flow_rate(p_0, p_r, fluid=None)`; the callers in `discretization.py` pass `wp.fluid` | `discretization.rs`: `w_res` from the inflow with the fluid's $f_g$ |
| input | `WellProperties.__post_init__` refuses Forchheimer with $f_g = 0$ | `Well::new`'s inflow `'forchheimer'` with a second coefficient, passed by `solvers/rust.py`; `WellSpec::check` refuses what the dataclasses refuse |
| harness | `configurations.py`: `v1_well` unchanged and `check` flags Forchheimer in `v1.0.0`; `V2`'s accepted inflows gain it; `solvers/rust.py`'s component classes gain it | test wells with the gas-rate inflow in `input.rs`'s `all()`; `_component` gains `gas_rate` for the vectors |

## Off in the `v1.0.0` configuration

`v1_well` builds Vogel or a productivity index, as now. `configurations.check(wp, 'v1.0.0')` already accepts only `Vogel` and `ProductivityIndex` (`differences`, the `V1` branch) and so flags `Forchheimer`; nothing changes there. The row vectors, both verifier reports and every develop vector are unchanged, because no existing inflow's expression changes; the fluid argument is ignored by the liquid inflows.

## `v2.0.0`'s check

Decided by Bjarne in the Goals: it accepts the option. In `configurations.differences`, the `V2` branch's test becomes

```python
if type(wp.inflow) not in (Vogel, ProductivityIndex, FixedFlowRate, Forchheimer):
    out.append(f'inflow: {type(wp.inflow).__name__}, not Vogel, ProductivityIndex, FixedFlowRate or Forchheimer')
```

The class is checked exactly, as for every part. `tests/test_configurations.py` gains a well with the inflow that passes `v2.0.0` and fails `v1.0.0`. The `develop` configuration has no inflow check, and gains none.

## Validity ranges

Checked against the ranges the sampler draft draws (`sampling-draft.md`; `gas_inflow_shapes.py` on its 703 gas and condensate wells of 2000, seed 2026). Proposed rows for `specs/model/validity.md`:

| Correlation | IDs | Input | Published range | Sampler | Outside | Rule |
|---|---|---|---|---|---|---|
| Forchheimer gas inflow, pressure-squared form | INF-10 | $p_r$, $p_0$ | $\mu_g Z$ nearly constant over $(p_0, p_r)$: $p_r$ below about 138 bar (2000 psia) (Golan and Whitson 1991, p. 147); above, the form is conservative at large drawdowns (their Example 2.6) | the draft: $p_r$ 60 / 278 / 652 bar (5th / 50th / 95th percentiles) | 79% of gas wells above 138 bar | Extrapolate (note 12) |
| | | the reservoir | single-phase gas, stabilised flow; not a condensate bank or mobile water near the well (Golan and Whitson 1991, §2.4; Economides et al. 1994, §4-3) | condensates below the dew point in mid and late life; wet gases with WGR up to 500 Sm³ per 10⁶ Sm³ | – | Extrapolate (note 12) |
| | | $D$ | "frequently of the order of $10^{-3}$" d/Mscf (Economides et al. 1994, p. 71); Eq. 4-34 for an estimate | the draft: $2\cdot10^{-4}$ to $2\cdot10^{-3}$ d/Mscf, 12 times Eq. 4-34 at the median | – | – (a parameter, not a correlation) |
| Liquid with the gas | INF-11 | $f_g$ | a mass balance; $f_g > 0$ | – | – | Refuse $f_g = 0$ |

**Proposed note 12.** "Forchheimer's gas inflow in the pressure squared holds where $\mu_g Z$ is about constant between $p_0$ and $p_r$, below about 138 bar. Above, the pseudo-pressure is the general form (Golan and Whitson 1991, Eq. 2.57), and the pressure-squared form with $\mu_g Z$ at $p_r$ gives 8.5% less rate at 40% drawdown and 14% less at the AOF than it, at the median of the sampler draft's gas wells, 28% less at the 5th percentile. The coefficients are a well's inputs, so the model extrapolates; a sampler fits them to the pseudo-pressure over the well's drawdowns, which leaves 2% at the median and 6% at the 5th and 95th percentiles from 20% drawdown to the AOF. Below the dew point a condensate drops liquid near the well, and a gas well with mobile water flows two phases there: neither is in the law, and the skin stands for both."

## Measurements so far (draw-only)

`plans/evidence/gas_inflow_shapes.py 2000` (seed 2026, with the geometry ruled on 2026-11-02): R3's 703 gas and condensate wells (351 wet gas, 316 condensate, 36 dry gas; 151 in the coverage tail). No well is solved. The reference is Forchheimer's law in the pseudo-pressure with R3's $k h$, skin and $D$, and $\mu_g Z$ of the model's black-oil gas at $T_r$: what R3's $a$ and $b$ stand for. Each inflow's rate over the reference, 5th / 50th / 95th percentiles:

| Inflow | $p_{wf}/p_r$ = 0.95 | 0.8 | 0.6 | 0.2 | AOF |
|---|---|---|---|---|---|
| $p^2$, $\mu_g Z$ at $p_r$ (R3's $a$, $b$) | 0.98 / 0.99 / 1.00 | 0.92 / 0.95 / 1.00 | 0.85 / 0.92 / 1.00 | 0.74 / 0.87 / 1.00 | 0.72 / 0.86 / 1.00 |
| $p^2$, scaled to the design point | 1.00 / 1.04 / 1.12 | 0.98 / 1.00 / 1.05 | 0.91 / 0.98 / 1.00 | 0.81 / 0.92 / 1.00 | 0.79 / 0.92 / 1.00 |
| $p^2$, a four-point test at 5% to 20% drawdown | 0.99 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.94 / 0.97 / 1.00 | 0.81 / 0.93 / 1.00 | 0.78 / 0.92 / 1.00 |
| $p^2$, least squares over the curve | 0.86 / 0.96 / 1.00 | 1.00 / 1.02 / 1.06 | 1.00 / 1.02 / 1.06 | 0.98 / 0.99 / 1.00 | 0.95 / 0.99 / 1.00 |
| backpressure, the same four-point test | 1.00 / 1.00 / 1.01 | 1.00 / 1.00 / 1.01 | 0.96 / 1.01 / 1.05 | 0.87 / 1.00 / 1.11 | 0.86 / 0.99 / 1.11 |
| Vogel, through the design point | 0.48 / 0.65 / 0.99 | 0.82 / 0.96 / 1.19 | 1.00 / 1.12 / 1.49 | 1.05 / 1.31 / 1.78 | 1.08 / 1.36 / 1.87 |
| the interim productivity index | 0.38 / 0.57 / 0.89 | 0.69 / 0.89 / 1.20 | 0.92 / 1.16 / 1.67 | 1.30 / 1.70 / 2.54 | 1.54 / 2.05 / 3.07 |

- The reference's median rates are 0.31, 0.76, 1.13, 1.52 and 1.57 million Sm³/d at those pressures; the design rate 0.83 million Sm³/d at the median.
- 79% of the wells have $p_r$ above 138 bar. Below it, R3's $p^2$ form is within 2% of the reference at the AOF (median 0.98); above, 0.84.
- The four-point test gives backpressure exponents of 0.56 / 0.68 / 1.00, and 4.3% outside 0.5 to 1.
- The non-Darcy term is more than half of the pressure-squared drop at the design rate in 66% of the wells. R3's $D$ is 12 times Economides et al.'s Eq. 4-34 for a well perforated over its net pay, at the median (1.7 to 81 times, 5th to 95th percentiles): a finding for the sampler draft (`plans/sampler-redesign/decisions.md`, D-62).
- Against R3's own $p^2$ curve, which it stands in for, the interim index gives 0.58 at 5% drawdown and 2.37 at the AOF, at the median.

## Acceptance

After implementation, on a branch in its own worktree:

1. **The `v1.0.0` configuration is unchanged.** The row vectors pass; both verifier candidates report PASS with 0 failures, no expected failures and a stable-root rate of 100%.
2. **The liquid inflows are unchanged.** Every develop vector, the four wells of `v2_rows.json` and the backend comparison's existing groups pass without regenerating anything but the new tables, in both backends.
3. **Vectors** from `specs/tools/make_develop_vectors.py`, in both backends (`RUST_ADAPTERS` in `tests/test_spec_vectors.py`):
   - `INF-10`: columns `a`, `b`, `p`, `p_r` → `w_g`. Rows at $p_0 = p_r$ (zero), at a small drawdown, at a typical one, at $p_0$ = 1.01325 bar (the AOF), with $b = 0$ (the linear limit), with $b q \gg a$, and above $p_r$ (the odd extension); coefficients from the 5th, 50th and 95th percentiles above, in the model's units.
   - `INF-11`: columns `f_g`, `w_g` → `w_l`, as `INF-4`'s table; and the fluid's form, `gor`, `wlr`, `rho_o`, `rho_g`, `rho_w`, `w_g` → `w_l`, for a wet gas and a condensate.
   - The fifth well of `vectors/v2_rows.json`: a condensate-like black oil (GOR 3000, water–liquid ratio 0.3), the gas-rate inflow, a Simpson choke, deviated, with Joule–Thomson cooling, in both backends.
4. **`tests/test_inflow.py`:** the checks of $a$, $b$ and $f_g$ raise; the closed form equals the quadratic's root, and $\Delta/a$ at $b = 0$; zero at $p_r$; strictly decreasing in $p_0$, and finite with the slope $-2p_r/a$ at $p_r$; the odd extension above $p_r$; the same value on CasADi symbols and on floats, and the CasADi gradient against a central difference; `from_gas_rate` against a hand conversion; INF-4 of INF-11 gives back $w_{g,\text{res}}$. Two worked examples, to the precision printed (`tests/test_worked_examples.py`, both backends): Golan and Whitson's Example 2.4, $A$ = 0.1095 psia²/(scf/d) and $B$ = 3.58·10⁻⁹ psia²/(scf/d)² at $p_R$ = 1623 psia, the seven rates of Table E2.4 (15.8 MMscf/d at the AOF to 1.99 at 1550 psia); and Economides et al.'s Example 4-7, $a$ = 1.5·10⁴ and $b$ = 68.5 in psi² and Mscf/d at $\bar p^2$ = 2.13·10⁷ psi², its AOF of 460 Mscf/d. Their Example 2.6's printed pressure-squared rates are not what its own $A$ and $B$ give (11.85 against 11.7 MMscf/d at the AOF), so it is not used.
5. **Property tests** (`tests/test_model_properties.py`), on gas wells solved by both backends: the reservoir rates satisfy INF-10 and INF-11 at $p_0$ to rounding; the mass balance up the well; first-order convergence on 40, 80 and 160 cells; a liquid-loading well keeps a stable root at the higher rate and an unstable one at the lower; with lift gas, the stable root's rate rises.
6. **Backend comparison** (`tests/backend_cases.py`):
   - feature `022` in `FEATURES` and `features()` (`type(wp.inflow) is Forchheimer`);
   - overlays `forchheimer`, the inflow with the same gas rate as the well's at half the drawdown and half its drop non-Darcy there ($a = \Delta_h/(2w_h)$, $b = \Delta_h/(2w_h^2)$), and `gas well`, the fluid at GOR 3000 Sm³/Sm³ and water–liquid ratio 0.3 with `forchheimer`;
   - matrix entries `W1+forchheimer`, `W2+forchheimer`, `develop+forchheimer`, `develop+gas well`, `develop+gas well+joule-thomson`, `develop+gas well+valve at 50%`, `develop+gas well+pvt table`;
   - comparison-set groups `v1.0.0+forchheimer`, `develop+forchheimer` and `develop+gas well`.

   The rows agree in every configuration, and the comparison set passes.
7. **The Rust core.** `cargo test --no-default-features` passes, with the march's assumption checks on the new test wells: W2 with the gas-rate inflow, a deviated gas well with Joule–Thomson cooling, and one with lift gas at a valve.
8. **Measurements** (`plans/evidence/`; sampled wells only). The sampler draft's gas and condensate wells, solved on both backends with the interim productivity index and with the gas-rate inflow fitted as R3's draws give it:
   - the operating points found: a fall is a solver finding unless the core's scan shows that the model has no root;
   - PBH, PWH, TWH and the rates, against the AOF and the tubing's erosional rate;
   - the share of samples with two roots (liquid loading), choked and with non-finite regions of $R$;
   - the cost per case on one process.
9. **The suites.** The full test suite, ruff, and `tests/test_markdown_math.py`.

## Out of scope

- **The backpressure law** (design choice 1), and the form in $p$ and a tabulated pseudo-pressure (design choice 2), each a later option if a user needs it.
- **Two-phase flow in the reservoir:** a condensate bank below the dew point (Fevang and Whitson 1996, not in hand) and mobile water near a gas well. The sampler's skin stands for them.
- **Transient deliverability** (Economides et al. 1994, §4-5) and a skin that changes with time.
- **A horizontal gas well's own law** (Economides et al. 1994, §4-6): the sampler's factor on $kh$ stands for the drain (`sampling-draft.md`, Productivity).
- **A fixed gas rate,** INF-8's gas counterpart, and a composite oil IPR, straight above the bubble point and Vogel below.
- **Calibration of $a$ and $b$** (`specs/calibration.md`, CAL-5 and CAL-6).

## Sources

- Forchheimer, P. (1901), "Wasserbewegung durch Boden", *Zeitschrift des Vereines deutscher Ingenieure* 45, 1781–1788. Not in hand; as Golan and Whitson (1991, Eq. 2.36) give it.
- Rawlins, E.L. and Schellhardt, M.A. (1936), *Back-pressure data on natural-gas wells and their application to production practices*, US Bureau of Mines Monograph 7. Not in hand; as Golan and Whitson (1991, p. 132) and Ahmed (2006, Eq. 8-20) give it.
- Al-Hussainy, R., Ramey, H.J. and Crawford, P.B. (1966), "The flow of real gases through porous media", *Journal of Petroleum Technology* 18, 624–636. Not in hand; as Golan and Whitson (1991, Eq. 2.47) give it.
- Golan, M. and Whitson, C.H. (1991), *Well performance*, 2nd ed., Prentice Hall: §2.4, pp. 132–149, Eqs. 2.36 to 2.62, Example 2.4 (Table E2.4, Fig. E2.4b) and Example 2.6 (Table E2.6, Fig. E2.6).
- Economides, M.J., Hill, A.D. and Ehlig-Economides, C. (1994), *Petroleum production systems*, Prentice Hall: §4-3 to §4-5, pp. 68–77, Eqs. 4-23 to 4-34 and 4-51 to 4-54, Examples 4-6 to 4-9.
- Whitson, C.H. and Brulé, M.R. (2000), *Phase behavior*, SPE Monograph 20: Eq. 3.35 and the text after it (wellstream gravity), Eqs. 3.73b and 3.73c.
- Ahmed, T. (2006), *Reservoir engineering handbook*, 3rd ed., Gulf Professional Publishing: Eq. 8-20, p. 556.
- Fevang, Ø. and Whitson, C.H. (1996), "Modeling gas-condensate well deliverability", *SPE Reservoir Engineering* 11(4), 221–230. Not in hand.
- The measurements: `plans/evidence/gas_inflow_shapes.py` (`plans/evidence/README.md`, Sampler redesign: the model's gaps).
