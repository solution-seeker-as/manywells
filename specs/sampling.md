# Sampling

*Owner: Bjarne Grimstad. Status: two parts in use and one in draft. v1's procedure (SMP-1 to SMP-31) and its extension to `develop`'s inputs (SMP-40 to SMP-44) are in use since 2026-09-30 (Step 4 of `plans/manywells-v2-plan.md`): the v1 procedure records v1.0.0's code, with the rulings of `specs/discrepancies.md`; Step 7 (2026-10-01) implemented the sampler (`manywells.sampling`, `manywells.datasets`) and wrote SMP-31 and SMP-40; Bjarne kept the starting points of SMP-41 to SMP-43 until the sampling redesign (2026-10-01). The v2 procedure (SMP-50 to SMP-117) is a draft for Bjarne's sign-off, 2026-11-10 (Step R10 of `plans/sampler-redesign-plan.md`): every rule in it rests on a ruling, and the text as a whole waits for his sign-off. Any change to this file needs his sign-off.*

How ManyWells draws wells and operating points for its datasets. This is not physics, so it lives outside `specs/model/`. The file has three parts:

- **v1's procedure** (SMP-1 to SMP-31) records v1.0.0's, which generated `manywells-sol-1` and `manywells-nsol-1`. In the v1-compatibility configuration the sampler draws v1's inputs as before, so that the Distributions check (`specs/verification.md`, Step 3) can compare regenerated samples with the published datasets. Closed loop (`manywells-nscl-1`) is out of scope for v2 (`specs/goals.md`).
- **The extension to `develop`'s inputs** (SMP-40 to SMP-44) draws the inputs that `develop`'s model adds (trajectory, black-oil parameters and pipe roughness) as further independent draws next to v1's. `develop`'s tests and the comparison of the backends use it; the v2 datasets do not.
- **The v2 procedure** (SMP-50 to SMP-117) draws the wells and operating points of `manywells-sol-2` and `manywells-nsol-2` in the `v2.0.0` configuration (`specs/goals.md`, limitation 5: unrealistic sampled wells). It replaces v1's procedure for the v2 datasets only: its draws come from generators of their own, so the `v1.0.0` configuration's draws do not change.

**Numbering.** The IDs come in blocks with gaps, so that a part can grow without renumbering the next: v1's procedure 1 to 31, the extension 40 to 44, and the v2 procedure from 50. An ID keeps its meaning; a rule that is dropped retires its ID.

## Conventions

- $U(a, b)$ is the uniform distribution. Every draw is independent of the others unless its definition says otherwise.
- A **well** is drawn once (SMP-1 to SMP-17). A **sample** is one operating point of a well, solved once; `sol-1` draws 500 per well (SMP-18 to SMP-22), `nsol-1` evolves 500 weekly samples (SMP-23 to SMP-27).
- The published datasets store the samples in their rows (`docs/datasets.md`) and the wells in their config files. A `sol-1` config holds the well draws, with $u = 0.5$ from the generator's first solve (SMP-28). An `nsol-1` config holds the well's state at its last sample, the last row of its data, because the generator updates the well in place (SMP-24 to SMP-27) and saves it at the end: the fractions, $\rho_l$, $c_{pl}$, the inflow's $f_g$, $p_r$, $p_s$, $u$, $w_{lg}$ and the decay rate $\gamma_{pr}$ are final values. The draws survive in its `ns_bhv` fields `pr_init`, `ps_init` and `init_fractions`. Both kinds store $f_D$ wrongly for `nsol-1` (S-8).
- Source: `scripts/data_generation/` at the `v1.0.0` tag. Where the code and the paper differ, `specs/discrepancies.md` (S-1 to S-12) records it; this file follows the code.

## Well draws

`sample_well` in `well.py`.

### SMP-1 · Depth

$L \sim U(1500, 4500)$ m. Paper (20).

### SMP-2 · Tubing diameter

$D$ uniform from the inner diameters {3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5} inches, each 0.5 inch less than the outer diameters {3.5, …, 7}; 1 inch = 0.0254 m. Paper (21).

### SMP-3 · Friction factor

$f_D \sim U(0.01, 0.08)$ (FRIC-2). Paper (22).

### SMP-4 · Heat transfer coefficient

$h \sim U(10, 40)$ W/(m² K). Paper (23).

### SMP-5 · Choke

A Simpson choke (CHK-5) with $K_c \sim U(K_c^\text{ref}/2,\ 2K_c^\text{ref})$ and $K_c^\text{ref} = 0.12A$. Paper (24).

### SMP-6 · Choke profile

Uniform from the four profiles CHK-7 to CHK-10. Paper §4.1.

### SMP-7 · Mass fractions

$(f_g, f_o, f_w) \sim \text{Dir}(1, 1, 1/2)$, the gas, oil and water mass fractions of the reservoir inflow. The well is discarded and redrawn if $f_g > 0.99$. Paper (25).

### SMP-8 · Inflow

Vogel inflow (INF-1) with $w_{l,\max} \sim U\big(20(1 - f_g),\ 200(1 - f_g)\big)$ kg/s, using the $f_g$ of SMP-7, and the gas fraction $f_g$ (INF-4). Paper (26).

### SMP-9 · Oil

$\rho_o \sim U(825, 925)$ kg/m³, $c_{po} = 2000$ J/(kg K). Paper (27).

### SMP-10 · Water

$\rho_w = 999.1$ kg/m³, $c_{pw} = 4184$ J/(kg K) (PVT-WAT-1). Paper §4.2.

### SMP-11 · Gas

$R_s \sim U(320, 520)$ J/(kg K), $c_{pg} = 2225$ J/(kg K). Paper (30).

### SMP-12 · Liquid

$\rho_l$ and $c_{pl}$ from the mixing rules PVT-MIX-2 to PVT-MIX-4, with $x_o = f_o/(f_o + f_w)$. Recomputed whenever the fractions change (SMP-22, SMP-24). Paper (28)–(31).

### SMP-13 · Nominal reservoir pressure

$p_{r,\text{nom}} = \rho_{sw} g L / c_\text{bar} + 1$ bar, with $\rho_{sw} = 1012.05$ kg/m³, the mean of seawater (1025) and water (999.1). Paper (33).

### SMP-14 · Reservoir temperature

$T_r = 333.15 + 0.03(L - 1500)$ K, that is 60 °C at 1500 m and 150 °C at 4500 m. Paper (36).

### SMP-15 · Surface temperature

$T_s = 277.15$ K (4 °C). Paper §4.3.

### SMP-16 · Nominal separator pressure

$p_{s,\text{nom}} \sim \text{LogNormal}(3, 1)$ bar, the logarithm having mean 3 and standard deviation 1. The well is discarded and redrawn if $p_{s,\text{nom}} \notin [10, 120]$ bar. Paper (34).

### SMP-17 · Gas-lift availability

If $f_g \le 0.2$, the well has gas lift with probability 1/2; otherwise it has none. Paper §4.5.

## Stationary samples (`sol-1`)

`Well.sample_new_conditions` in `well.py`, drawn around the well's nominal values for each sample.

### SMP-18 · Choke position

$u \sim U(0.05, 1)$. Paper (44).

### SMP-19 · Lift-gas rate

$w_{lg} \sim U(0, 5)$ kg/s if the well has gas lift, otherwise 0. Paper (45).

### SMP-20 · Separator pressure

$p_s \sim U(0.9 p_{s,\text{nom}},\ 1.1 p_{s,\text{nom}})$. Paper (35).

### SMP-21 · Reservoir pressure

$p_r \sim U(0.98 p_{r,\text{nom}},\ 1.02 p_{r,\text{nom}})$. It is not stored in the datasets. Paper (32).

### SMP-22 · Fractions

$f_g \sim U(0.95 f_{g,\text{nom}},\ 1.05 f_{g,\text{nom}})$, capped at 0.99; the water–liquid mass fraction $f_{wl} \sim U(0.95 f_{wl,\text{nom}},\ 1.05 f_{wl,\text{nom}})$, capped at 1, with $f_{wl,\text{nom}} = f_w/(f_o + f_w)$; then $f_w = (1 - f_g) f_{wl}$ and $f_o = 1 - f_g - f_w$. The liquid (SMP-12) and the inflow's $f_g$ follow; $w_{l,\max}$ does not change. Paper (53)–(55).

## Non-stationary samples (`nsol-1`)

`NonStationaryBehavior` and `NonStationaryWell` in `nonstationary_well.py`. Time $t_i$ counts weeks $i$ from the first sample; the reservoir pressure starts at $p_r(t_0) = p_{r,\text{nom}}$, without SMP-21's redraw.

### SMP-23 · Lifetime

$t_\text{life}/52 \sim U(10, 20)$ years.

### SMP-24 · Fraction random walk

Weekly drifts $\gamma_g = U\big(f_g(t_0)/2,\ f_g(t_0)\big)/t_\text{life}$ and $\gamma_o = U\big(f_o(t_0)/2,\ f_o(t_0)\big)/t_\text{life}$, drawn once per well. Each week

$$f_g \leftarrow \min\lbrace 0.99, \max\lbrace f_g - \gamma_g + X_g, 0.002\rbrace\rbrace, \qquad f_o \leftarrow \min\lbrace 0.99, \max\lbrace f_o - \gamma_o + X_o, 0.002\rbrace\rbrace,$$

with $X_g, X_o \sim N(0, 0.015^2)$, both redrawn until $f_g + f_o \le 0.999$; then $f_w = 1 - f_g - f_o$, and the liquid and inflow follow (SMP-12). Paper (37)–(39).

The generator steps the walk on every attempt, not once per week (SMP-29). An attempt at week $i$, when the last accepted sample was at week $i'$, takes $\max(1, i - i')$ steps from the current fractions, which earlier attempts have already moved. The first attempt takes one step, so the fractions at $t_0$ are not the drawn ones. $k$ failed solves in a row, which advance the week, take $(k+1)(k+2)/2$ steps over $k + 1$ weeks, and a rejected sample, which does not advance it, takes as many steps again.

### SMP-25 · Reservoir pressure decay

$p_r(\infty) = p_r(t_0) - U\big(0.2 p_r(t_0),\ 0.4 p_r(t_0)\big)$ and $\gamma_{pr} = 1 - 0.01^{52/t_\text{life}}$ initially. At every attempt, $\gamma_{pr} \leftarrow \min\lbrace 0.9, \max\lbrace 0.1, \gamma_{pr} + \epsilon\rbrace\rbrace$ with $\epsilon \sim U(-\gamma_{pr}(t_0)/20,\ \gamma_{pr}(t_0)/20)$, and

$$p_r(t_i) = \big(p_r(t_0) - p_r(\infty)\big)(1 - \gamma_{pr})^{i/52} + p_r(\infty).$$

Paper (40)–(43).

### SMP-26 · Separator pressure

$p_s \sim U(0.9 p_{s,\text{nom}},\ 1.1 p_{s,\text{nom}})$ every week. Paper (35).

### SMP-27 · Controls

$u$ and $w_{lg}$ as in SMP-18 and SMP-19, every week (open loop). Paper §4.5.1.

## Generation

The solves, starts and acceptance rules. They are part of the procedure because they shape the datasets' distributions: a filter that discards wells, or a start that lands on the trickle root, changes what is published.

On `develop` (`sampling.generate`), every solve returns the well's operating point, the stable root (`specs/model/solution.md`), and the generator's start is an extra start of the root search (`x_guess`), not the only one; a solve without an operating point is a failed solve. Each sample's fractions change its fluid (SMP-22, SMP-24), so its well's system is built for the sample.

### SMP-28 · Stationary generation (`sol-1`)

`open_loop_stationary/generate_well_data.py`, 2,000 wells of 500 samples.

1. Solve the well at $u = 0.5$ with its nominal conditions; if that fails, discard the well. The root is the initial guess of every sample.
2. Draw samples (SMP-18 to SMP-22) and solve each. A failed solve, negative rates or fractions outside [0, 1] count as failures; a sample with $w_m < 0.1$ kg/s is dropped without counting. The well is discarded after $5 \cdot 500$ attempts or 100 failures.
3. Discard the well if the standard deviation of $u$ over its samples is below a fifth of $U(0.05, 1)$'s, $0.95/\sqrt{12}$; if more than 80% of its samples are choked (CHK-12); or if the coefficient of variation of its total standard volumetric rate (`QTOT`) is below 0.05. The port takes the sample standard deviation (ddof = 1) of `QTOT` where v1.0.0 took the population one (ddof = 0), so its coefficient of variation is 1.001 times v1.0.0's, and a well within 0.1% of the threshold is kept where v1.0.0 dropped it; left to the sampling redesign (Bjarne, 2026-10-29).

### SMP-29 · Non-stationary generation (`nsol-1`)

`open_loop_nonstationary/generate_open_loop_nonstationary_well_data.py`, 2,000 wells of 500 samples.

1. Solve the well at $u$ = 0.1, 0.2, …, 1.0 in turn, each root the initial guess of the next; if one fails, discard the well. If $w_m < 7$ kg/s at $u = 1$, discard the well.
2. Keep a pool of starts, beginning with the root at $u = 1$. A root joins it if $w_m \ge 3$ kg/s and it lies more than 0.05 from every root in the pool, in the coordinates $(f_g, p_r/350\ \text{bar})$. Each sample starts from the pool's closest root in those coordinates.
3. Each attempt updates the conditions (SMP-24 to SMP-27), with the week $i$ and the week $i'$ of the last accepted sample, and solves. A failed solve counts as a failure and advances $i$. Every solved root joins the pool (step 2) before the checks below. Negative rates or fractions outside [0, 1] count as failures, and a sample with $w_m < 1$ kg/s is rejected without counting; neither advances $i$. An accepted sample sets $i' = i$ and advances $i$. The well is discarded after 200 failures, or after 50 failed solves in a row; any successful solve resets that count, even if the sample is then rejected. There is no cap on attempts.
4. The loop runs while the well has fewer than 500 samples, or while $i$ equals the lifetime in whole weeks, $\lfloor t_\text{life} \rfloor$; the second condition can add a 501st sample, which then discards the well.
5. Discard the well if its choke positions vary too little or its `QTOT` too little, as in SMP-28 step 3; there is no filter on choked samples.

### SMP-30 · Dataset rows

Each accepted sample becomes one dataset row, computed from the root: the features of `docs/datasets.md`, with standard volumetric rates at PVT-GAS-2's standard conditions, `CHOKED` from CHK-12 and the regimes from SLIP-8. `TBH` is $T_r$, which is the root's $T_0$ (THM-3).

### SMP-31 · Seeding

Every draw comes from a NumPy generator seeded from the dataset's seed and the draw's place in the dataset: well $k$'s v1 draws from $(\text{seed}, k, \texttt{well})$, its `develop` draws from $(\text{seed}, k, \texttt{develop})$, the $j$-th stationary sample from $(\text{seed}, k, j, \texttt{sample})$, and a non-stationary well's evolution, which is sequential, from $(\text{seed}, k, \texttt{nonstationary})$; a name is hashed to an integer by CRC-32 (`sampling.wells.rng_for`). A run gives the same draws whatever the order of the wells or the number of processes, and the `develop` draws do not shift the v1 draws. The generator draws wells $0, 1, 2, \dots$ and keeps the first it accepts, so the dataset does not depend on the number of processes either. Each dataset records its seed, its configuration, its grid and the code version (`datasets.io`).

v1.0.0 seeded NumPy from process ID × time for `sol-1` and from the process ID alone for `nsol-1` (S-9), so its datasets cannot be regenerated; that is why the Distributions check compares distributions, not rows.

## Extension to `develop`'s inputs

`develop`'s model takes inputs that v1's does not. The sampler draws them as further independent draws, from their own generator (SMP-31), and draws v1's inputs exactly as above in both configurations, so that the two configurations share the v1 draws. The values of SMP-41 to SMP-43 are the starting points of Step 4, implemented as they stand; Bjarne kept them until the sampling redesign after the plan (2026-10-01).

### SMP-40 · Configurations and mapping

- **`v1.0.0` configuration.** Draw SMP-1 to SMP-29 and map them to `develop`'s inputs with `configurations.v1_well`: a vertical `WellGeometry` of length $L$ and diameter $D$ with $N = 100$ cells; the fixed $f_D$ of SMP-3 and v1.0.0's thermal model with $h$; Vogel inflow and the Simpson choke; and the fluid of `configurations.v1_fluid`, a dead oil with the mixed liquid's $\rho_l$ and $c_{pl}$ (SMP-12) and no water, an ideal gas with $\rho_{g,\text{sc}} = p_\text{ref}/(R_s T_\text{ref})$ (PVT-GAS-2), the gas–oil ratio that gives $f_g$, and the surface tension from $\rho_l$ (PVT-MIX-5). It gives back $\rho_l$, $c_{pl}$, $f_g$ and $R_s$, to rounding. The liquid is mixed before the mapping, as v1.0.0 mixed it, so a sample whose liquid is all water (SMP-22 caps the water–liquid fraction at 1) maps as well.
- **`develop` default.** The same draws, with the oil and the water by their own densities at standard conditions: $\rho_o$, $\rho_w$, the water–liquid ratio $\alpha_{w,l}$ of PVT-MIX-2 and the gas–oil ratio $(f_g/\rho_{g,\text{sc}})/(f_o/\rho_o)$, which give back $f_g$, $\rho_{l,\text{sc}}$ and $c_{pl}$ (PVT-MIX-10); black oil, real gas and the surface tension from the oil (PVT-MIX-7); SMP-41 for the trajectory, SMP-42 for friction, and `develop`'s thermal model with $h$. A sample without oil ($f_o = 0$) has no gas–oil ratio and is a failed solve.

### SMP-41 · Trajectory

The bottomhole's true vertical depth is drawn as SMP-1, and SMP-13 and SMP-14 use it, since pressure and temperature follow depth. The trajectory is vertical with probability 1/2, deviated with probability 1/4 and L-shaped with probability 1/4:

- **Deviated:** vertical down to a kickoff depth $U(0.1, 0.5)$ times the true vertical depth, then straight at an inclination $\theta \sim U(10°, 60°)$ to the bottomhole.
- **L-shaped:** vertical to the bottomhole's depth, then a horizontal section of length $U(500, 2000)$ m.

The grid is uniform in measured depth with $N = 100$ cells (`WellGeometry.from_survey`). Bjarne kept these probabilities and ranges until the redesign (2026-10-01).

### SMP-42 · Pipe roughness

$\varepsilon \sim \text{LogUniform}(1.5\cdot10^{-6},\ 1.5\cdot10^{-4})$ m, from new to corroded tubing, with $f_D$ from the Reynolds number (Chen) in place of SMP-3. For comparison, SMP-3's range of $f_D$ reaches 0.08, which fully rough turbulent flow gives only at a relative roughness above 0.05.

### SMP-43 · Black-oil parameters

No new draws: the API gravity from SMP-9 (21.3 to 39.9, inside the 16 to 58 that the black oil takes, `specs/model/validity.md`), the gas gravity from SMP-11 (0.55 to 0.90), and the gas–oil and water–liquid ratios from SMP-40's mapping. There is no bubble-point cap: gas dissolves up to the gas available. Separator conditions are standard conditions (`FluidModel`'s defaults). Wells with a small oil fraction get very high gas–oil ratios (above $10^4$ Sm³/Sm³), outside the correlations' range; Bjarne accepted them until the redesign, with no cap (2026-10-01).

### SMP-44 · Lift-gas temperature

Not drawn: the lift gas enters at $T_r$, as in v1 (THM-3).

## v2: scope and conventions

*A draft for Bjarne's sign-off (status line). The v2 procedure states the rules: distributions, bounds and the order of the draws. The reasons, the evidence and the measurements behind them stay in `plans/sampler-redesign/`: the decision log `decisions.md` (D-1 to D-107, each with its ruling), the drafts `sampling-draft.md`, `fluids-draft.md` and `datasets-draft.md`, the checkpoints `checkpoint.md` and `checkpoint-2.md`, and the pilots' measurements. Each rule ends with its decision entries and its sources.*

**Conventions** for SMP-50 on; v1's Conventions above hold for SMP-1 to SMP-44.

- **Units** are the model's: m, bar (absolute), K, kg/s, kg/m³, J/(kg K), W/(m² K), and Sm³/Sm³ at standard conditions (PVT-GAS-2). Liquid rates are Sm³/d of stock-tank liquid, gas rates Sm³/d of separator gas, and time $t$ is in years from the well's first production. $g$ = 9.80665 m/s² and $p_\text{atm}$ = 1.01325 bar.
- **Distributions.** $U(a, b)$ is uniform; $\text{LogU}(a, b)$ log-uniform; $N_{[a,b]}(\mu, \sigma)$ normal truncated to $[a, b]$; $\text{LogN}_ {[a,b]}(m, s)$ log-normal with median $m$ and log standard deviation $s$, truncated to $[a, b]$; $\text{Cat}$ categorical. A truncated draw is the inverse CDF of one uniform draw, so it never loops. $\text{logit}\thinspace x = \ln(x/(1 - x))$.
- **Outside a range** each rule says which it does: *truncated*, the distribution restricted to the bound; *redrawn*, the block drawn again, and the redraw counted; or *accepted*, a value outside a correlation's published range at which the model extrapolates (`specs/model/validity.md`).
- **Core and tail.** Each draw gives its core form and, where it differs, its tail form, which a tail well takes (SMP-53).
- **Depths.** $z$ is the reservoir's true vertical depth below the mudline (the seabed offshore, the ground onshore), $d_w$ the water depth, and $L$ the model's length: the true vertical depth of the bottom of the modelled flow path below the wellhead (GEO-3), the mid-perforation, or the heel of a well with a horizontal drain. A platform's wellhead is at sea level, a subsea well's at the seabed.
- **Water.** For oils the water is the water–liquid ratio at standard conditions (WLR); for gas classes the water–gas ratio (WGR, Sm³ per 10⁶ Sm³ of gas), and the WLR that the model takes is WGR/(WGR + $10^6/R_{go}$).
- **Oil equivalent.** $q_\text{oe} = q_o + q_g/1000$ in Sm³/d, with $q_g$ the formation gas, without the lift gas.
- **The model's correlations in the draws.** Where a draw links two inputs it evaluates the well's black oil on floats before the well is built: `BlackOil(rho_o, rho_g, gas_law='dak', p_sep, T_sep)` with the well's draws, that is Vazquez–Beggs's $R_{so}$ with PVT-OIL-5's separator correction (PVT-OIL-6), $B_o$ (PVT-OIL-8), Beggs–Robinson's viscosity (PVT-OIL-10, PVT-OIL-11), DAK's $Z$ (PVT-GAS-11) and Lee–Gonzalez–Eakin's gas viscosity (PVT-GAS-7). A table fluid's draws use the same black oil at its table's stock-tank densities, except for the inflow's coefficients, which take the table (SMP-72, SMP-117). A *no-slip stream* at $(p, T)$ is the stream's in-situ volume and mass per Sm³ of stock-tank liquid from that black oil, without slip: $\rho_m$ is their ratio.
- **Citations.** D-n is an entry of `plans/sampler-redesign/decisions.md`, whose Ruling line is the decision. [RF §n], [WP §n] and [GEO] are `sources-reservoir-fluids.md`, `sources-wells-production.md` and `sources-geometry.md` in the same folder, with the primary references they quote. [own analysis of …] is the maintainer's analysis of a publisher's open data, kept outside the repo (`specs/constitution.md`, Open real-well data). The references are under v2: references.

**Basis.** Every distribution, bound and rule of the procedure rests on data or on physics; a guess stands in only where both are missing, and a choice made for a goal is marked as a design choice (Bjarne, 2026-11-10: `plans/sampler-redesign-plan.md`, Goals, The yardstick, Data and physics decide). After its sources, each rule's **Basis** line says, part by part, which of four kinds it rests on: *data*, published statistics or ranges, or an analysis of open per-well data, whether the value is fitted to them, set against them on a pilot or chosen within them; *physics*, a derivation, a published correlation, equation or rule, or the model's own result or valid range; *design choice*, a choice made for a goal such as coverage, the Goals' shares, simplicity or reproducibility, Bjarne's rulings among options that the evidence leaves open and the generation's rules included; or *guess*, a value with neither data nor physics behind it, with what was looked for and what would replace it, as far as the plans record them. *Basis not recorded* marks a part whose basis the plans do not give; a draw's tail form is SMP-53's design choice; and every guess and design choice is listed under v2: guesses and design choices.

### SMP-50 · Datasets and configuration

- The v2 procedure draws two datasets, `manywells-sol-2` (stationary) and `manywells-nsol-2` (non-stationary), each of 10,000 wells with 200 samples per well. Every sample is an operating point.
- Every well is built in the `v2.0.0` configuration with $N$ = 100 cells (SMP-111) and solved by the Rust core, `SSDFSimulator(wp, backend='rust')`.
- The population is a realistic core, weighted towards offshore production wells on platforms and subsea that flow naturally or on gas lift and counted by wells, and a labelled coverage tail over the model's valid range (SMP-53). The draws first cover almost all real wells and hold no unrealistic ones; second, they weigh typical wells more than rare ones.
- Both datasets draw their wells by the same rules, from seeds of their own (SMP-109), and every well has a life (SMP-83). An `nsol-2` well's 200 samples span its life; a `sol-2` well's samples vary around one date of it (SMP-92).

From the Goals of `plans/sampler-redesign-plan.md` (Bjarne, 2026-10-30 and 2026-11-05), D-1 and D-52 (ruled 2026-11-02). Sources: `specs/goals.md`, Compatibility.

**Basis:** the two datasets, their size, a sample as an operating point and the population (a realistic core weighted towards offshore wells and counted by wells, with a labelled tail) are a design choice of the Goals (D-1, D-52); the configuration, the backend and $N$ = 100 cells, as in SMP-40, are a design choice of the generation.

### SMP-51 · Blocks and their order

A well's draws are made in blocks, in this order. Each block may use the blocks above it, and has a generator of its own (SMP-109).

| Block | Draws | Uses |
|---|---|---|
| `labels` | coverage tail, well type, fluid class, drive, life stage | – |
| `depth` | $z$, $d_w$, $L$ | labels |
| `geometry` | class, family, survey | $z$, $d_w$, $L$ |
| `water` | salinity, $\rho_w$ | – |
| `pressure` | $p_i$ | $z$, $d_w$, $\rho_w$ |
| `temperature` | $T_s$, $G$, $T_r$ | well type, $d_w$, $z$, $L$ |
| `separator` | $p_\text{sep}$, $T_\text{sep}$, $r_s$, $\Delta p_\text{fl}$, $p_{s,\text{nom}}$ | well type, stage, $d_w$ |
| `fluid` | API, $\gamma_g$, $p_\text{sat}$, $R_{si}$ | class, $p_i$, $T_r$, $z$, separator |
| `composition` | a table fluid's composition and source | class, fluid, separator |
| `stage` | $r_p$, the water, the GOR's parameters | drive, stage, $p_i$, fluid |
| `inflow`, `nondarcy` | $k$, $h_n$, $s$, drainage, $\delta_d$, the drain; $D$ | well type, geometry, fluid at the stage |
| `completion` | tubing, $\varepsilon$ | inflow, fluid, separator, stage |
| `choke` | bean, $C_D$, profile | tubing |
| `lift` | lift at the stage, injection depth, $T_{lg}$, $w_{lg,\max}$ | class, stage, pressures, rates, survey |
| `heat` | $h$, $c_{po}$, $c_{pw}$, $c_{pg}$ | tubing, stage, fluid, salinity, temperatures |

- Each block draws its numbers in a fixed order, and draws every one of them whatever branch the well takes, so that a change of one rule moves no other draw: a number that the well's branch does not use, such as a tail form in a core well, a subsea flowline in a platform well or a lifted well's design in a natural one, is drawn and left unused. Every draw takes a fixed count of numbers: a continuous draw one, by its inverse CDF; a survey six, whatever its family and whether it fits, and the `geometry` block all 20 surveys' numbers (SMP-60); and a Dirichlet's weights one each, as exponential draws over their sum (SMP-113).
- **The same well at another stage.** A well's draws at stage $s$ are its draws with the life stage set to $s$ after the `labels` block has drawn it, from the same generators, so that only the draws that depend on the stage change. The life (SMP-83) uses the well at its three stages.
- A table fluid's table (SMP-112 to SMP-116) is made after the `fluid` block, and its saturation pressure and initial GOR replace the `fluid` block's before the `stage` block; its tables are made at the GORs its life reaches.

From D-7 (ruled 2026-11-06, with the checkpoint's note that every number of a block is drawn whatever its branch), D-105 and D-108. Sources: `sampling-draft.md`, The order of the draws.

**Basis:** a design choice for reproducibility (D-7, D-108): one generator per block, and every number drawn whatever the branch, in a fixed count, so that a change of one rule moves no other draw. The order puts each block after the draws that condition it: the brine before the pressure, whose hydrostatic part it sets, and the separator before the fluid, whose $R_{so}$ it corrects (PVT-OIL-5).

### SMP-52 · Labels

Every categorical choice is stored with the well. All are drawn or set before the well is solved, except where a rule sets one after a solve (SMP-101, SMP-103).

| Label | Values | Set by |
|---|---|---|
| `coverage_tail` | `core`, `tail` | SMP-53 |
| `well_type` | `platform` (offshore, dry tree), `subsea`, `onshore` | SMP-54 |
| `fluid_type` | `black_oil`, `volatile_oil`, `gas_condensate`, `gas` | SMP-55 |
| `fluid_class` | the four, with `gas` split into `wet_gas` and `dry_gas` | SMP-55 |
| `class_rule` | `mccain_1990` | SMP-55 |
| `drive` | `depletion`, `pressure_support` | SMP-56 |
| `life_stage` | `initial`, `mid_life`, `late`: a `sol-2` well's, an `nsol-2` sample's | SMP-56, SMP-91 |
| `life_state` | `initial`, `mid_life`, `depleted` (late, depletion), `watered_out` (late, pressure support) | derived |
| `pressure_class` | `under`, `normal`, `over` | SMP-63 |
| `saturation` | `undersaturated`, `saturated`, `single_phase_gas` | SMP-68 |
| `geometry_class` | `vertical`, `slightly_deviated`, `deviated`, `highly_deviated` | SMP-59 |
| `geometry_family` | `vertical`, `build_and_hold`, `s_shaped`, `high_angle` | SMP-59 |
| `pvt_model` | `black_oil`, `table` | SMP-112 |
| `fluid_source` | `synthetic`, or a published example's key; table fluids only | SMP-112, SMP-114 |
| `saturation_kind` | `bubble`, `dew`, `none`; table fluids only | SMP-115 |
| `saturation_rule` | `matched`, `gor_moved`, `equilibrium_phase`; table fluids only | SMP-115 |
| `inflow` | `vogel`, `productivity_index`, `forchheimer` | SMP-74 |
| `tubing` | the nominal size, `2-7/8` to `9-5/8` | SMP-75 |
| `choke_model` | `simpson` | SMP-77 |
| `chk_profile` | `linear`, `sigmoid`, `convex`, `concave` | SMP-77 |
| `lift` | `natural`, `gas_lift`; in `nsol-2`, lifted at some date of the life | SMP-90 |
| `lift_start_rule` | `from_start`, `static_check`, `drawn_time`, `end_of_natural_flow` (`nsol-2`), `no_natural_flow` (`sol-2`); none for a natural well | SMP-90, SMP-101, SMP-103 |

The labels' codes and columns are the dataset schema's (`datasets-draft.md`, Section 4; R11).

From D-3, D-9, D-36, D-49, D-65, D-89, D-101 and D-105 (ruled 2026-11-02 to 2026-11-06). Sources: `sampling-draft.md`, Labels; `datasets-draft.md`, Section 4.

**Basis:** a design choice of the Goals: every categorical choice is stored with the well, so that a user can group, reweight or drop wells by it; the codes are the schema's (D-89).

### SMP-53 · Coverage tail

`coverage_tail` is `tail` with probability 0.20, else `core`. In a tail well every draw takes its tail form: the categorical shares of the well type, the fluid type and the life stage become equal, the other categorical draws take the tail shares their rules give, and the continuous draws become uniform or log-uniform over the ranges their rules give. The links to earlier draws stay: pressure still follows depth and brine, the bubble point the fluid, the tubing the rate. A tail well may take lift and any geometry whatever its stage and type.

From D-6 and D-103 (ruled 2026-11-06, checkpoint A2). Sources: the Goals' 10% to 20% of the accepted wells, of which 0.20 drawn gives about 15% (`r6-measurements.md`).

**Basis:** the tail's share is a design choice for coverage: the Goals ask for a labelled tail of 10% to 20% of the accepted wells, no public source sets a share, and 0.20 drawn gives about 15% accepted, because more tail wells are dropped (`r6-measurements.md`). Its form (flat, independent draws over each rule's range, equal shares, lift and geometry whatever the stage and type) is a design choice for coverage of the model's valid range (checkpoint A2); the Basis lines below mark a rule's tail form with (SMP-53).

### SMP-54 · Well type

$\text{Cat}$(platform 0.55, subsea 0.35, onshore 0.10); tail equal shares. It is no model input: it conditions the water depth, the surface temperature, the separator, the permeability, the injection pressure and the economic floor. The onshore share is a design choice.

From D-5 (ruled 2026-11-06, checkpoint A1). Sources: [WP §1]; Liu, Wang and Kaiser (2014); EIA (2025), p. 3; [own analysis of Sodir's FactMaps].

**Basis:** data for the split of the offshore wells between platforms and subsea: it is chosen, not fitted, between the Gulf of Mexico's shelf of platform wells, its deep water with 84% dry trees (Liu, Wang and Kaiser 2014) and Norway's large minority of subsea wellbores [own analysis of Sodir's FactMaps], and no global count of producing wells by type was found ([WP §1], Gaps). The onshore share of 0.10 is a design choice, the Goals' offshore weighting (checkpoint A1): counted by wells the world is onshore (EIA 2025, p. 3). The tail's equal shares are a design choice for coverage (SMP-53).

### SMP-55 · Fluid class

$\text{Cat}$(black oil 0.55, volatile oil 0.10, gas condensate 0.15, wet or dry gas 0.20); tail 0.25 each. A gas well is `dry_gas` with probability 0.07 (tail 0.25), else `wet_gas`. The classes are McCain's by the initial producing GOR, within which SMP-68 draws it: black oil up to 356 Sm³/Sm³ (2000 scf/STB), volatile oil from 356 to 588 Sm³/Sm³ (3300 scf/STB), gas condensate from 588 to 8900 Sm³/Sm³ (50,000 scf/STB), wet gas above 8900 Sm³/Sm³, and dry gas with a condensate yield below 2.8 Sm³ per 10⁶ Sm³ (0.5 STB/MMscf). Black oils and wet and dry gases take the black oil, volatile oils and condensates a PVT table (SMP-112).

From D-3, D-4 and D-36 (ruled 2026-11-06). Sources: [RF §5, §6]; McCain (1990), pp. 151–158; [own analysis of BOEM's sands atlas]; [own analysis of Sodir's FactMaps]; the Goals' floor of about 10% of the wells per type.

**Basis:** data for the black oils' 0.55 and the gases' 0.20, chosen, not fitted, between the Gulf of Mexico's and Norway's shares by wells [own analysis of BOEM's sands atlas; own analysis of Sodir's FactMaps], and for the dry gases' 0.07, between the Gulf's dry-gas shares by wells and by sands. The volatile oils' 0.10 and the condensates' 0.15 are a design choice for coverage, the Goals' floor of about 10% per type, which over-weights the volatile oils several times (D-4); the tail's shares are a design choice for coverage (SMP-53). The class limits are McCain's (1990) published rules, taken here as physics; choosing them among the published limits (D-3's alternatives), with tables for the volatile oils and condensates, is a design choice: the model's black oil carries McCain's black oils (D-3, D-36). For the dry gases' yield limit of 2.8 Sm³ per 10⁶ Sm³ (0.5 STB/MMscf), which stands for McCain's dry gas, with no liquid at the surface: basis not recorded.

### SMP-56 · Drive and life stage

`drive` is `pressure_support` (water drive or injection) with probability 0.7 for oil classes and 0.4 for gas classes, else `depletion`; tail 0.5. `life_stage` is $\text{Cat}$(initial 0.25, mid-life 0.40, late 0.35); tail equal shares. The stage places a `sol-2` well in its life (SMP-92); an `nsol-2` sample takes the stage of its date (SMP-91). The stage shares are a design choice: no public count of wells by stage is in hand.

From D-18 (ruled 2026-11-06, checkpoint C1). Sources: [RF §3, §11]; Miller, Sorrell and Speirs (2009), Table 4.4; [own analysis of BOEM's sands atlas]; [own analysis of Sodir's FactMaps].

**Basis:** data for the direction of the drive shares: most of the Gulf of Mexico's oil sands, by wells, have water drive [own analysis of BOEM's sands atlas], and Norway supports most fields' pressure by injection [own analysis of Sodir's FactMaps]. The values of 0.7 and 0.4 are a guess: no count of wells under pressure support was found, the atlas recording natural drive and not injection ([RF §3], Gaps), and one would replace them. The stage shares are a design choice (checkpoint C1), after fields' lives of 10 to 30 years with a plateau in the first third ([RF §11]; Miller, Sorrell and Speirs 2009, Table 4.4): no public count of wells by stage was found, and one would replace them. The tail's shares are a design choice for coverage (SMP-53).

## v2: well draws

### SMP-57 · Depth below the mudline

$z \sim \text{LogN}_ {[600, 6500]}(m, s)$ m by fluid class; tail $\text{LogU}(600, 7000)$ m. Truncated.

| Class | $m$ (m) | $s$ |
|---|---|---|
| black oil | 2674 | 0.317 |
| volatile oil | 3159 | 0.315 |
| gas condensate | 3424 | 0.278 |
| wet gas | 2419 | 0.422 |
| dry gas | 1141 | 0.586 |

From D-8 (ruled 2026-11-06). Sources: [RF §1]; $m$ and $s$ an own fit to the P5, P50 and P95 by wells and class [own analysis of BOEM's sands atlas]; [own analysis of Sodir's FactMaps]; Hasan et al.'s measured depths (`validity.md`).

**Basis:** data for $m$ and $s$: an own fit to the P5, P50 and P95 of the Gulf of Mexico's sands by wells and class [own analysis of BOEM's sands atlas], one region, with which Norway's oil wellbores agree [own analysis of Sodir's FactMaps]. The bounds of 600 and 6500 m are data as well, design values after that atlas and the published scatters of reservoir depths, with few reservoirs below 6 km ([RF §1]; Ehrenberg and Nadeau 2005; Nadeau, Sun and Ehrenberg 2023), not fitted; 600 m also lies within Hasan et al.'s measured depths (`validity.md`). The tail's log-uniform to 7000 m, past the core's bound, is a design choice for coverage (SMP-53).

### SMP-58 · Water depth and length

- platform: with probability 0.90 a shelf platform, $d_w \sim \text{LogN}_ {[5, 400]}(90, 0.6)$ m, else a deep-water dry tree, $\text{LogU}(400, 2000)$ m; tail $\text{LogU}(5, 2000)$ m;
- subsea: $d_w \sim \text{LogU}(80, 3000)$ m; tail $\text{LogU}(50, 3000)$ m;
- onshore: $d_w = 0$.

$L = z + d_w$ for a platform well, whose riser is part of the flow path, and $L = z$ for subsea and onshore wells. $d_\text{mud}$, the mudline's TVD in the modelled well, is $d_w$ for a platform well and 0 otherwise.

From D-8 (ruled 2026-11-06) and D-110. Sources: [WP §2]; Liu, Wang and Kaiser (2014); [own analysis of Sodir's FactMaps]; [own analysis of BOEM's borehole file]; [own analysis of BOEM's directional surveys and borehole file, Sodir's FactMaps, NSTA's wellbore layer and NOAA's World Ocean Atlas 2023], which separates the producing wells by type.

**Basis:** data for the spans of the water depths, those of Norway's platform and subsea wellbores [own analysis of Sodir's FactMaps] and of the Gulf of Mexico's development boreholes [own analysis of BOEM's borehole file], and for dry trees to about 2000 m (Liu, Wang and Kaiser 2014). Data for the subsea wells' range of 80 to 3000 m: the subsea wells of three regions together span it, one region's centred near 350 m and another's mostly in deep water [own analysis of BOEM's directional surveys and borehole file, Sodir's FactMaps, NSTA's wellbore layer and NOAA's World Ocean Atlas 2023]; its log-uniform shape is a design choice, D-104's rule of a uniform draw over the pooled range (D-110). Data also for the platforms: the deep-water dry trees' share of 0.10 lies between one region's present-day share and the other regions' none, a sizeable minority of one region's shelf wells stand in water shallower than 15 m, which the floor of 5 m covers, and the shelf's median of 90 m and log spread of 0.6 are chosen, not fitted, with the median between the Gulf's shelf wells' and Norway's and the UK's and the Gulf's spread wider [same analyses] (D-110). The lengths are physics, the model's flow path from the wellhead (GEO-3), with a platform's riser in it. The tail's log-uniforms, the subsea one from 50 m, are a design choice for coverage (SMP-53).

### SMP-59 · Geometry: classes and families

The class, then the family that draws the survey (SMP-60), with the same shares for every well type and fluid, core and tail:

| `geometry_class` | Share | Family and its inclination |
|---|---|---|
| `vertical` | 0.25 | `vertical` |
| `slightly_deviated` | 0.25 | `build_and_hold`, $\theta_h \sim U(5°, 20°)$ |
| `deviated` | 0.25 | with probability 2/3 `build_and_hold`, $\theta_h \sim U(20°, 60°)$; else `s_shaped` |
| `highly_deviated` | 0.25 | with probability 1/2 `high_angle`, a drain modelled to its heel, $\theta_\text{heel} \sim U(60°, 75°)$; else a high-angle tangent, `build_and_hold` with $\theta_h \sim U(60°, 75°)$ |

The even mix is a design choice for coverage.

From D-9 (ruled 2026-11-02) and D-57 (ruled 2026-11-06). Sources: [GEO]; [own analysis of BOEM's directional surveys, NLOG's deviation surveys, NSTA's wellbore layer and Sodir's FactMaps].

**Basis:** the even mix of the four classes is a design choice for coverage (Bjarne's ruling, D-9): the public surveys show few vertical offshore wells and most wells deviated [own analysis of BOEM's directional surveys, NLOG's deviation surveys, NSTA's wellbore layer and Sodir's FactMaps]. The classes' ranges of inclination, with the uniform angles within them and the highly deviated class's two halves, are a design choice too, the halves lying between regions where drains are a minority and a majority of such wells; 75° is SMP-61's. The S wells' third of the deviated class is data, chosen, not fitted, between their shares in the Gulf of Mexico's and the Dutch surveys [own analysis of BOEM's directional surveys and NLOG's deviation surveys] (D-57).

### SMP-60 · Geometry: the survey

A survey is a list of stations $(\text{MD}, \text{TVD})$ from the wellhead to the bottom at TVD $L$: straight sections, and circular builds and drops with one station per degree.

- `vertical`: straight down.
- `build_and_hold`: vertical to the kickoff $K$, a build at $b \sim U(0.3, 3.6)$ °/30 m to $\theta_h$ (SMP-59), then straight to $L$; the hold spans at least 50 m of TVD.
- `s_shaped`: $\theta_h \sim U(15°, 45°)$ and $b$ as above; after the hold, a drop at $d \sim U(0.2, 2.2)$ °/30 m to $\theta_e = \min(U(0°, 35°), \theta_h)$, then straight at $\theta_e$ over the last $\min(U(100, 400)\ \text{m}, L/4)$ of TVD; the hold spans at least 100 m of TVD.
- `high_angle`: a build at $b$ to the tangent $\theta_h = \min(U(10°, 65°), \theta_\text{heel})$, a hold, a landing build at $b_2 \sim U(1.5, 7)$ °/30 m to $\theta_\text{heel}$, and a tangent of $U(50, 300)$ m of MD at $\theta_\text{heel}$ to the heel at $L$; the hold spans at least 100 m of TVD.
- **Kickoff.** $K = d_\text{mud} + f z$ with $f \sim U(0.05, f_\text{max})$, $f_\text{max} = \min\big(0.75,\ (L - d_\text{mud} - \Delta\text{TVD}_ \text{below})/z\big)$, and $\Delta\text{TVD}_ \text{below}$ the TVD of the builds, the drop, the S well's end section and the drain's tangent, plus the hold's least TVD.
- **Fit.** Where the shape's sections do not fit below $d_\text{mud} + 0.05\thinspace z$, the build rates and the S well's drop rate are raised by steps of 20%, each to at most its range's top. A shape that still does not fit ($f_\text{max} < 0.05$), or whose measured depth exceeds 9342 m, is redrawn from the block's generator, at most 20 times; then the well is `vertical`. The angles are never lowered, so a well keeps its class.

From D-104 (ruled 2026-11-06, checkpoint B1), D-9 and D-35. Sources: each parameter uniform between the pooled 5th and 95th percentiles of [own analysis of BOEM's directional surveys, NLOG's deviation surveys, NSTA's wellbore layer and Sodir's FactMaps]. The limit of 20 redraws is the prototype's (`plans/evidence/sampler_draft_draws.py`), reached by none of its wells.

**Basis:** data for the ranges of the build, drop and landing rates, the S wells' bottom inclination, the drains' first tangent and the kickoff: each runs between the pooled 5th and 95th percentiles of the regions' surveys [own analysis of BOEM's directional surveys, NLOG's deviation surveys, NSTA's wellbore layer and Sodir's FactMaps] (D-104); a uniform draw over each range is a design choice for simplicity (checkpoint B1). The S wells' hold angle of 15° to 45° and end section of 100 to 400 m, the drain's tangent of 50 to 300 m at the heel and the holds' least 50 and 100 m of TVD are guesses, R3's values from before the surveys were read: the plans record no source for them and no replacement. The fit rule, with its steps of 20% and its 20 redraws, is a design choice of the generation that keeps a well in its class. The bound of 9342 m is physics, the edge of the model's valid range, Hasan et al.'s measured depths (`validity.md`), where D-35 stops the sampler.

### SMP-61 · Inclination limit and drains to the heel

No cell is more than 75° from vertical; no draw of SMP-59 or SMP-60 exceeds it. A `high_angle` well stands for a well with a horizontal drain: the model's bottom is the heel at the reservoir's true vertical depth, the drain has no cells, and its length enters the inflow (SMP-72, SMP-73). The v2 datasets therefore hold no horizontal cell, which settles `specs/model/validity.md`'s decision 5 for them.

From D-10 (ruled 2026-10-30) and D-9. Sources: [WP §3], Taitel and Dukler's stratified–non-stratified transition as scanned there; Brill and Mukherjee (1999), p. 44; [own analysis of BOEM's borehole file].

**Basis:** physics for where the limit can lie: Taitel and Dukler's transition leaves stratified upward flow only at very low liquid rates from 70° to 80° from vertical (`plans/evidence/stratified_flow_limit.py`; [WP §3]), and Brill and Mukherjee's (1999, p. 44) map has no stratified upward flow. Taking 75° within that band, and modelling a drain to its heel because no option models stratified flow, is a design choice (Bjarne's ruling, D-10); it admits the heel sections of the Gulf of Mexico's horizontal wells [own analysis of BOEM's borehole file].

### SMP-62 · Formation water

The salinity, the mass fraction of dissolved solids, is $w_s \sim \text{LogN}_ {[0.002, 0.25]}(0.06, 0.6)$; tail $\text{LogU}(0.002, 0.25)$. Truncated. The water's density at standard conditions,

$$\rho_w = 999.1 + 692\thinspace w_s + 311\thinspace w_s^2\ \text{kg/m}^3,$$

is the well's water (`FluidModel.rho_w`, PVT-WAT-1), and sets the hydrostatic pressure (SMP-63) and $c_{pw}$ (SMP-82).

From D-11 (ruled 2026-11-06, with the median and floor lowered). Sources: [RF §9]; McCain (1990), Example 16-6 (p. 449) and Fig. 16-8; Whitson and Brulé (2000), §9.2, Eq. 9.1; [own analysis of USGS's produced-water database].

**Basis:** data for the median of 0.06 and the floor of 0.002, set with the produced waters of US wells [own analysis of USGS's produced-water database] (D-11 as ruled); R3's log spread of 0.6 was left as it was, not fitted. The cap of 0.25 is physics, near halite's saturation and below 80% of the NaCl limit (Whitson and Brulé 2000, Eq. 9.1), and so is the density, a quadratic through three published points (McCain 1990, Example 16-6 and Fig. 16-8). The tail's log-uniform is a design choice for coverage (SMP-53).

### SMP-63 · Initial reservoir pressure

With the hydrostatic and lithostatic pressures in bar,

$$p_h = p_\text{atm} + \frac{g\thinspace(\rho_\text{sea} d_w + \rho_w z)}{10^5}, \qquad p_\text{lith} = p_\text{atm} + \frac{g\rho_\text{sea} d_w}{10^5} + 0.226\thinspace z,$$

with $\rho_\text{sea}$ = 1025 kg/m³ and 0.226 bar/m (1 psi/ft) below the mudline, the initial pressure is $p_i = p_h + \lambda\thinspace(p_\text{lith} - p_h)$, where

$$\text{logit}\thinspace\lambda \sim N_{[-12,\ \text{logit}\thinspace\lambda_\text{max}]}\big(\text{logit}\thinspace\lambda_{50}(z),\ 0.9\big), \qquad \lambda_{50}(z) = 0.02 + \frac{0.45}{1 + e^{-(z/1000\ \text{m} - 3.8)/0.9}},$$

and $\lambda_\text{max} = \max\big(10^{-3}, \min(0.85,\ (950 - p_h)/(p_\text{lith} - p_h))\big)$, so that $p_i \le 950$ bar. Truncated. Tail: $\lambda \sim U(0, \min(0.8, \lambda_\text{max}))$. With probability 0.04 (tail 0.10) the reservoir is initially underpressured instead: $p_i = p_h\thinspace U(0.8, 1.0)$. Where the hydrostatic pressure alone exceeds the cap, $p_h > 950$ bar, a deep reservoir under deep water, $p_i$ = 950 bar and the reservoir is underpressured, whatever its numbers. `pressure_class` is `under` for an underpressured reservoir, `over` where $\lambda \ge 0.25$ (about 0.6 psi/ft), and `normal` otherwise.

From D-12 and D-35 (ruled 2026-11-06) and D-109. Sources: [RF §2]; $\lambda_{50}$ and its spread an own fit to depth bins of [own analysis of BOEM's sands atlas]; Nadeau, Sun and Ehrenberg (2023), Fig. 20; Hasan et al.'s bottomhole pressures to 949 bar (`validity.md`). The lithostatic gradient has no source in hand.

**Basis:** physics for the hydrostatic pressure, from the well's own brine, and for the cap of 950 bar, the edge of the model's valid range, Hasan et al.'s bottomhole pressures (`validity.md`; D-35); holding $p_i$ at the cap where $p_h$ alone passes it is a design choice, the simplest rule that keeps the bound and moves no other draw (D-109). Data for $\lambda_{50}(z)$ and its spread of 0.9: an own fit to depth bins of the Gulf of Mexico's sands by wells [own analysis of BOEM's sands atlas], one region, checked against published counts of reservoirs (Nadeau, Sun and Ehrenberg 2023, Fig. 20); data also for the underpressured share of 0.04 at 0.8 to 1.0 of hydrostatic, chosen, not fitted, after that figure's 6.0% of reservoirs and the Gulf's few such wells. The lithostatic gradient of 0.226 bar/m (1 psi/ft) is a guess, a textbook rule with no source in hand: a published review of overpressure is on the resource list, not in hand (`checkpoint.md`, section 6), and would replace it; the caps on $\lambda$, 0.85 for a pressure below a fracture gradient near lithostatic and 0.8 in the tail, are guesses with no source and no replacement recorded. The tail's uniform $\lambda$ and its underpressured share of 0.10 are a design choice for coverage (SMP-53).

### SMP-64 · Surface temperature

$T_s$ is the top of THM-4's ambient profile:

- offshore (platform and subsea), core and tail, the seabed's temperature at the water depth: $T_s = T_\text{deep} + (T_\text{shelf} - T_\text{deep})\thinspace e^{-d_w/400\ \text{m}}$, with $T_\text{deep} \sim U(272, 278)$ K and $T_\text{shelf} \sim U(279, 297)$ K;
- onshore: $U(268, 303)$ K; tail $U(258, 308)$ K.

From D-14 (ruled 2026-11-06). Sources: [WP §2] and the seabed temperatures it quotes; Hasan and Kabir (2018), Fig. 7.17; [own analysis of NOAA's World Ocean Atlas 2023].

**Basis:** data for the offshore profile: its form and ranges go through published seabed temperatures ([WP §2]; Hasan and Kabir 2018, Fig. 7.17), and the seabed temperatures at the producing wells of three regions and on the sea floor of a fourth lie within its shelf and deep-water ranges [own analysis of NOAA's World Ocean Atlas 2023]. Taking the seabed's temperature for a platform well too is a design choice within THM-4's linear ambient profile, which cannot follow a water column. The onshore range of 268 to 303 K is a guess after the mean annual temperatures of producing regions ([WP §2]): no distribution over producing wells was found, and climate normals of producing regions would replace it. The onshore tail is a design choice for coverage (SMP-53).

### SMP-65 · Geothermal gradient and reservoir temperature

$G \sim \text{LogN}_ {[G_\text{lo}, G_\text{hi}]}(G_{50}, 0.22)$ K/km below the mudline, with the median $G_{50} = 20 + 10\thinspace e^{-\max(d_w - 100\ \text{m},\ 0)/400\ \text{m}}$ clamped to $[G_\text{lo}, G_\text{hi}]$; tail $U(G_\text{lo}, G_\text{hi})$. Then $T_r = T_s + G z$. With $L$ and $z$ in km and temperatures in K, $G_\text{lo} = \max(12,\ 10\thinspace L/z)$ keeps the model's gradient $(T_r - T_s)/L$ at 10 K/km or more, and $G_\text{hi} = \min\big(55,\ (450 - T_s)/z\big)$ keeps $T_r \le 450$ K; where $G_\text{lo} > G_\text{hi}$, $G_\text{lo}$ is set to $G_\text{hi}$. Truncated.

From D-13 and D-35 (ruled 2026-11-06). Sources: [RF §4]; Nadeau, Sun and Ehrenberg (2023), Fig. 7A and Table 1; [own analysis of BOEM's sands atlas]; [own analysis of Sodir's FactMaps]; the CasADi backend's limit near 5 K/km (`validity.md`).

**Basis:** data for the median and the spread: 30 K/km on shelves falling to 20 K/km in deep water, with a log spread of 0.22, are design values after the Gulf of Mexico's and Norway's gradients [own analysis of BOEM's sands atlas; own analysis of Sodir's FactMaps] and the published 14 to 36 K/km (Nadeau, Sun and Ehrenberg 2023, Fig. 7A), not a fit, and the form of the fall with water depth is an own one. For the bounds of 12 and 55 K/km: basis not recorded. The cap of 450 K is physics, near the end of the fluids behind the model's correlations (Valkó and McCain 2003, Table 2.1; `validity.md`), where D-35 stops the sampler. The least gradient of 10 K/km over the modelled well is a design choice, a margin above the CasADi backend's limit near 5 K/km (`validity.md`); the tail's uniform is a design choice for coverage (SMP-53).

### SMP-66 · Separator and the nominal downstream pressure

The host's first-stage separator at the start of the well's life:

- platform and subsea: $p_\text{sep} \sim \text{LogN}_ {[5, 85]}(35, 0.45)$ bar; $T_\text{sep} \sim U(300, 340)$ K on a platform, $U(290, 320)$ K for subsea;
- onshore: $p_\text{sep} \sim \text{LogN}_ {[2, 70]}(15, 0.6)$ bar, $T_\text{sep} \sim U(285, 320)$ K.

A subsea well's flowline and riser add $\Delta p_\text{fl} = U(5, 30) + U(0.03, 0.07)\thinspace d_w$ bar, with $d_w$ in m; other wells have $\Delta p_\text{fl} = 0$. At a stage, the nominal pressure downstream of the choke is

$$p_{s,\text{nom}} = \min\big(\max(r_s\thinspace p_\text{sep} + \Delta p_\text{fl},\ 1.5\ \text{bar}),\ p_\text{cap}\big),$$

with the stage's factor $r_s$ = 1 (initial), $U(0.5, 1)$ (mid-life) or $U(0.1, 0.6)$ (late), and $p_\text{cap}$ = 120 bar, or 250 bar for a subsea well. Tail: $p_{s,\text{nom}} = \min\big(\text{LogU}(1.5, 120) + \Delta p_\text{fl},\ p_\text{cap}\big)$. Over the life $r_s$ steps down (SMP-89). The separator is also the fluid's (`BlackOil(p_sep, T_sep)`, PVT-OIL-5): the GOR and the gas gravity are as reported at it.

From D-15 (ruled 2026-11-06, with the subsea flowline and its higher cap). Sources: [WP §7]; Guo, Lyons and Ghalambor (2007), §10.2.5 (p. 10/124) and Table 10.4; [own analysis of Petrobras' 3W Dataset].

**Basis:** data for the range of the separator pressures: published first stages at 41 to 83 bar and low-pressure vessels at 8.6 bar (Guo, Lyons and Ghalambor 2007, §10.2.5 and Table 10.4), with trade examples ([WP §7]); data also for the subsea flowline's loss and its cap of 250 bar, set on the rework's pilot against subsea tree pressures [own analysis of Petrobras' 3W Dataset], kept outside the repo, the term in $d_w$ being the static column of a riser's mixture at 300 to 700 kg/m³. The medians of 35 and 15 bar, the spreads and the bounds of $p_\text{sep}$ are guesses within the published range, and the separator temperatures are a guess with no source recorded: no public distribution of separator or manifold conditions over wells was found ([WP §7], Gaps), and one would replace them. The stage factors $r_s$ are a guess after two fields that moved to lower stages late in life (trade press, [WP §7]), with the same gap and replacement. For the floor of 1.5 bar and the cap of 120 bar, which is v1's (SMP-16): basis not recorded; the tail's log-uniform is a design choice for coverage (SMP-53).

### SMP-67 · Oil and gas gravities

| Class | API gravity | Gas gravity $\gamma_g$ (air = 1) | Tail |
|---|---|---|---|
| black oil | $N_{[16,45]}(33, 4.5)$ | $N_{[0.58,0.95]}(0.70, 0.07)$ | both uniform over the bounds |
| volatile oil | $U(40, 50)$ | $N_{[0.62,1.0]}(0.75, 0.06)$ | $\gamma_g$ uniform |
| gas condensate | $U(42, 58)$ | $N_{[0.60,0.90]}(0.72, 0.05)$ | $\gamma_g$ uniform |
| wet and dry gas | $U(45, 58)$ | $N_{[0.571,0.75]}(0.62, 0.035)$ | $\gamma_g$ uniform |

Independent draws; truncated. $\rho_o$ follows from the API gravity (PVT-OIL-2) and $\rho_{g,\text{sc}}$ from $\gamma_g$ (PVT-GAS-6). A table fluid's densities are its table's (SMP-113, SMP-114).

From D-16 (ruled 2026-11-06, the black oils' gas heavier and independent of the API) and D-35. Sources: [RF §7]; Valkó and McCain (2003), Table 2.1; McCain (1990), pp. 155–156; Whitson and Brulé (2000), Table 2.1 and §2.5; [own analysis of BOEM's sands atlas]; [own analysis of Sodir's FactPages]; the black oil's API range and Sutton's gas gravities from 0.571 (`validity.md`).

**Basis:** data for the black oils' API and gas gravity: the Gulf of Mexico's oils by wells [own analysis of BOEM's sands atlas], with the gas heavier as Norway's drill-stem tests show [own analysis of Sodir's FactPages] (D-16). Data, less directly, for the other classes: their API ranges are chosen within the published class ranges (McCain 1990, pp. 153–156; Whitson and Brulé 2000, §2.5), and their gas gravities after the Gulf's gas sands, Norway's tests and published example fluids (Whitson and Brulé 2000, Table 2.1), none stated as a fit. The bounds of 16 and 58 °API and of 0.571 are physics, the model's valid range: its black oil's and Sutton's gases' (`validity.md`). The independent draws are a design choice for simplicity, the tests' correlation of API and gas gravity changing $R_{si}$ little (D-16); the tail's uniforms are a design choice for coverage (SMP-53).

### SMP-68 · Saturation pressure and the initial GOR

- **Black oil.** $p_\text{sat} = p_i\thinspace(1 - x)$ at $T_r$, with $x = 0$ (saturated at discovery) with probability 0.20, else $x \sim \text{LogN}_ {[0.005, 0.85]}(0.10, 1.2)$; tail $x \sim U(0, 0.85)$. $R_{si} = R_{so}(p_\text{sat}, T_r)$ by the well's black oil (PVT-OIL-6). Where that exceeds 356 Sm³/Sm³, $p_\text{sat}$ is lowered by bisection to the pressure at which $R_{so}$ is 356 Sm³/Sm³, and $R_{si}$ = 356. The black oil's bubble point is then $p_\text{sat}$ at $T_r$ (PVT-OIL-15); no `p_bubble` is passed.
- **Volatile oil:** $R_{si} \sim \text{LogU}(356, 588)$ Sm³/Sm³.
- **Gas condensate:** $R_{si} \sim \text{LogN}_ {[588, 8900]}\big(2000\thinspace e^{-0.3\thinspace(z/1000\ \text{m} - 3.4)},\ 0.6\big)$ Sm³/Sm³; tail $\text{LogU}(588, 8900)$.
- **Wet gas:** a condensate yield of $\text{LogN}_ {[2.8, 112]}(30, 1.2)$ Sm³ per 10⁶ Sm³, tail $\text{LogU}(2.8, 112)$, and $R_{si} = 10^6$/yield. **Dry gas:** a yield of $\text{LogU}(0.5, 2.8)$.
- **Table fluids.** A volatile oil's or condensate's saturation pressure and initial GOR are its table's (SMP-115), and its drawn $R_{si}$ is the composition's target (SMP-113). The `fluid` block still draws $x$ for them as for a black oil, with the probability 0.40 (volatile oil) or 0.30 (condensate) of $x = 0$ and the median 0.05, so that its numbers do not depend on the table; $p_i\thinspace(1 - x)$ is kept only as a record.

`saturation` is `single_phase_gas` for wet and dry gas; otherwise `saturated` where $p_r \le p_\text{sat}$ at the well's state (`sol-2`: its date, SMP-92; `nsol-2`: its start) and `undersaturated` where $p_r > p_\text{sat}$.

From D-17 (ruled 2026-11-06; for table fluids, L2 of 2026-11-09) and D-65. Sources: [RF §7]; [own analysis of BOEM's sands atlas]; [own analysis of Sodir's FactMaps]; Whitson and Brulé (2000), Table 2.1; Beggs and Robinson's $R_{so}$ range (`validity.md`).

**Basis:** data for the black oils' undersaturation, the share of 0.20 saturated, the median of 0.10 and the long tail: the Gulf of Mexico's oil sands by wells [own analysis of BOEM's sands atlas], whose bubble points are Standing's estimates, in error by about as much as the margin ([RF §7]); physics for their $R_{si}$, the model's own $R_{so}$ at $p_\text{sat}$, capped at McCain's limit. The condensates' GOR with depth and the wet gases' yield are data too, own forms after the yields of the Gulf's gas sands [own analysis of BOEM's sands atlas], not stated as fits. For the volatile oils' log-uniform between McCain's limits, the dry gases' yield from 0.5 and the bounds of $x$: basis not recorded; the tail's uniforms are a design choice for coverage (SMP-53). A table fluid's $x$ is a record only (L2): its shares of 0.40 and 0.30 and its median of 0.05 were a guess, design values with no count of these classes by saturation behind them, and SMP-115's flash has replaced them.

### SMP-69 · Reservoir state at a stage

The `stage` block draws the well's reservoir state at its stage (SMP-51): the pressure ratio $r_p = p_r/p_i$ and the water.

| Stage | $r_p$, depletion | $r_p$, pressure support |
|---|---|---|
| initial | $U(0.85, 1.0)$ | $U(0.97, 1.0)$ |
| mid-life | $U(0.55, 0.92)$ | $U(0.85, 1.0)$ |
| late | oils $U(0.3, 0.6)$, gases $U(0.2, 0.55)$ | $U(0.70, 1.0)$ |

| Stage | Oils' WLR, depletion | Oils' WLR, support | Gases' WGR, depletion | Gases' WGR, support |
|---|---|---|---|---|
| initial | 0 with probability 0.5, else $U(0, 0.1)$ | as depletion | $\text{LogU}(2, 20)$ | as depletion |
| mid-life | $U(0, 0.3)$ | $U(0.1, 0.6)$ | $\text{LogU}(5, 50)$ | $\text{LogU}(10, 150)$ |
| late | $U(0.05, 0.4)$ | $U(0.6, 0.98)$ | $\text{LogU}(5, 80)$ | $\text{LogU}(50, 500)$ |

Tail, at every stage: $r_p \sim U(0.15, 1.0)$; oils' WLR $U(0, 0.98)$; gases' WGR $\text{LogU}(1, 10\thinspace 000)$ (D-107; D-105's tail was $\text{LogU}(1, 1000)$, from the same uniform number). The state's producing GOR is SMP-70's at $p_r$.

These states anchor the life (SMP-83): its start is the initial stage's, its end and its later lift (SMP-90) the late stage's, and its separator's middle step the mid-life stage's. Neither dataset samples them directly.

From D-18 (ruled 2026-11-06, checkpoint C1), D-105 (ruled 2026-11-06) and D-107 (ruled 2026-11-10, L4). Sources: [RF §3, §9, §11]; [own analysis of BOEM's sands atlas]; [own analysis of Sodir's FactMaps]; Hasan et al.'s water cuts to 0.98 (`validity.md`).

**Basis:** data for the gases' late range under depletion, after the $p/Z$ at which the Gulf of Mexico's depletion-drive gas sands end [own analysis of BOEM's sands atlas], and for the water: the oils' ranges by stage follow the water cut of Norway's oil fields by years on production, and the gases' the WGR of its gas fields [own analysis of Sodir's FactMaps], the upper tail to 0.98 is what real wells reach late in life (checkpoint C1), and the life built on these ranges lies within the regions' per-well medians [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well]; none is a fit. The other ranges of $r_p$ are guesses between that range, one textbook well (Golan and Whitson 1991, Example 4.11) and the rule that a supported reservoir stays near its initial pressure: no public statistics of producing wells' current against initial pressure were found ([RF §3], Gaps), and such statistics would replace them. The cap of 0.98 is physics, the edge of the model's valid range, Hasan et al.'s water cuts (`validity.md`). The tail's flat ranges, from its floor of 0.15 on $r_p$ to a WGR of 10,000 (D-107), are a design choice for coverage (SMP-53).

### SMP-70 · Producing GOR

- **Oils.** $R_{go} = R_{si}$ where $p_r \ge p_\text{sat}$; below, $R_{go} = R_{si}\thinspace\Phi(p_r/p_\text{sat})$ from the material balance of solution-gas drive: Muskat's equation (Craft and Hawkins 1991, Ch. 9, Eqs. 9.33 to 9.35) integrated by RK4 in pressure from $p_\text{sat}$ down to $0.05\thinspace p_\text{sat}$ on the well's black oil at $T_r$, whose $R_{so}$ at $p_\text{sat}$ is $R_{si}$, and interpolated linearly in $p_r/p_\text{sat}$. $k_g/k_o$ is Craft and Hawkins's Fig. 9.16 in the core; in the tail, the book's laboratory curves blended in $\log_{10} k_g/k_o$ by $w \sim U(0, 1)$, from Fig. 9.16 at $w = 0$ to Table 10.2 at $w = 1$. No cap. Just below the saturation pressure $\Phi$ dips under 1, while the freed gas is below its critical saturation, and then rises.
- **Gas condensates.** $R_{go} = R_{si}\big(1 + c_k\max(0,\ 1 - p_r/p_\text{sat})\big)$, $c_k \sim U(0.5, 3)$, as liquid drops out in the reservoir.
- **Wet and dry gas.** $R_{go} = R_{si}$.

$w$ and $c_k$ are drawn by the `stage` block; the life uses its start's (SMP-87). The construction of the curves (the connate water, the critical gas saturation and the blend) is `plans/evidence/solution_gas_drive.py`'s.

From D-19 and D-44 (ruled 2026-11-06, checkpoint C1). Sources: Craft and Hawkins (1991), Ch. 9 (Eqs. 9.33 to 9.35, Example 9.1, Fig. 9.16) and Ch. 10 (Table 10.2); Golan and Whitson (1991), Example 4.11 (p. 454); McCain (1990), pp. 151, 153. The condensates' factor is an own choice.

**Basis:** physics for the oils: the material balance of solution-gas drive (Muskat's equation; Craft and Hawkins 1991, Ch. 9) on the well's own black oil, which reproduces the book's Example 9.1 and its Ch. 10 history match (`plans/evidence/solution_gas_drive.py`), and for the wet and dry gases' constant GOR ([RF §5]). The core's $k_g/k_o$, Fig. 9.16, is data: of the book's laboratory curves it is the one that keeps the late wells' GOR within the regions' per-well GOR over their first year's [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well], kept outside the repo (checkpoint C1). The tail's blend up to Table 10.2 is a design choice for coverage of the book's range (SMP-53). The condensates' factor $c_k \sim U(0.5, 3)$ is a guess, an own choice (D-19): Norway's condensate fields' late GOR is the only check in hand [own analysis of Sodir's FactMaps], and the plans name no replacement.

### SMP-71 · Reservoir and completion parameters

Per well, independent of the fluid, from the `inflow` block:

- permeability by well type: platform $\text{LogN}_ {[0.5, 3000]}(15, 1.2)$ mD, subsea $\text{LogN}_ {[0.5, 3000]}(60, 1.2)$ mD, onshore $\text{LogN}_ {[0.05, 1000]}(3, 1.5)$ mD; tail log-uniform over the bounds. For gas classes the median and the bounds are multiplied by 1/14: an effective permeability that carries the operating limits the model lacks;
- net pay $h_n \sim \text{LogN}_ {[3, 100]}(15, 0.8)$ m; tail $\text{LogU}(3, 100)$ m;
- skin $s \sim N_{[-3, 15]}(2, 3)$; tail $U(-3, 15)$;
- drainage $\ln(r_e/r_w) \sim U(7, 9)$, and $U(8.5, 9.5)$ for a `high_angle` well, with $r_w$ = 0.108 m (an 8.5 in hole);
- the design drawdown $\delta_d \sim U(0.1, 0.4)$, a fraction of $p_r$;
- a drain's length $L_d \sim U(300, 2500)$ m and anisotropy $I_\text{ani} \sim \text{LogU}(1, 3)$, drawn for every well and used by `high_angle` wells.

For gas classes, the `nondarcy` block draws the non-Darcy coefficient from Economides et al.'s estimate (1994, Eq. 4-34) in field units (d/Mscf; $k$ in md, $h_n$, $r_w$ and $h_\text{perf}$ in ft, $\mu_g$ in cP),

$$D = 6\cdot10^{-5}\thinspace\frac{\gamma_g\thinspace k^{-0.1} h_n}{\mu_g r_w h_\text{perf}^2},$$

with the perforated interval $h_\text{perf} = U(0.3, 1)\thinspace h_n$ and $\mu_g$ at the start's $(p_r, T_r)$, times a log-normal error factor of median 1 and log standard deviation 0.7, truncated to $[10^{-5}, 2\cdot10^{-3}]$ d/Mscf, core and tail.

From D-20 and D-100 (ruled 2026-11-06, checkpoint D1; the platform oils' upper tail kept, L5 of 2026-11-09), D-62 (ruled 2026-11-06, D2), D-21 and D-104. Sources: [WP §4]; Ehrenberg and Nadeau (2005), Table 1; Economides, Hill and Ehlig-Economides (1994), §4-4 (p. 71) and Eq. 4-34; Golan and Whitson (1991), Examples 2.4 and 2.6; [own analysis of BSEE's well tests, ANP's production by well and NSTA's production by well], against which the medians were set on the rework's pilot.

**Basis:** data for the permeability's medians and spreads by well type and for the gas classes' factor of 1/14: both are set against per-well rates [own analysis of BSEE's well tests, ANP's production by well and NSTA's production by well], kept outside the repo, on the rework's pilot and with feature 022's inflow (`plans/evidence/gas_rates_d62.py`), so the permeability is an effective one; data also for the drainage's 7 to 9 (Economides et al. 1994, p. 71) and the drain's length, the pooled surveys' range (D-104). The non-Darcy coefficient is physics, Economides et al.'s estimate (1994, Eq. 4-34), with an error factor after the estimate's published error of up to 100% ([WP §4]) and an upper bound after the books' values. The permeability's wide bounds are a design choice, coverage first (D-100); the tail's forms are a design choice for coverage (SMP-53). The net pay, the skin, the high-angle wells' drainage, the hole of 8.5 in, the design drawdown, the anisotropy, the perforated share and the lower bound of $D$ are guesses: the plans give no source for them, no public distribution of wells' productivity was found ([WP §4], Gaps), one would replace them, and until then the per-well rates check only their joint effect with the permeability.

### SMP-72 · Oil wells' productivity index

At a reservoir state $(p_r, R_{go})$ the radial productivity index is, in SI units converted to Sm³/(d bar) of stock-tank liquid,

$$J_V = \frac{2\pi k h_n}{\mu_o B_o\thinspace(\ln(r_e/r_w) - 0.75 + s)},$$

with $\mu_o$ and $B_o$ of the well's PVT model at $(p_r, T_r)$ and $R_{go}$: its black oil, or for a volatile oil its table set at $R_{go}$. Then $J = f\thinspace J_V$, with the geometry's factor $f$ set once, from the well's initial-stage draw:

- `high_angle`: $f = \eta\thinspace J_H/J_V$, with Joshi's horizontal-well index $J_H$ (Economides et al. 1994, Eqs. 2-46 to 2-48) for the drain $L_d$ with the anisotropy $I_\text{ani}$, on the drainage length $r_{eH} = \max(r_e, L_d/1.6)$, and Dikken's friction efficiency along the liner, taken as the tubing, in closed form: $\eta = (1 + \Pi/1.5)^{-1/3}$, $\Pi = c_\text{lin} J_H^2 u L_d$, with $u = \delta_d p_r$ at a rate capped at the tubing's erosional rate (SMP-75), and $c_\text{lin}$ the liner's friction and acceleration at the heel's rate (`plans/evidence/drain_friction.py`). The liner carries the liquid of the state's stream, with $B_l = (1 - \text{WLR})\thinspace B_o + \text{WLR}$, the density $\big((1 - \text{WLR})(\rho_o + R_{go}\rho_{g,\text{sc}}) + \text{WLR}\thinspace\rho_w\big)/B_l$ kept within 400 to 1100 kg/m³ and the viscosity $(1 - \text{WLR})\thinspace\mu_o + \text{WLR}\cdot 0.5$ mPa s; its friction factor is Chen's (FRIC-4) at the tubing's roughness, times 1.5 for a perforated liner. No cap.
- `build_and_hold`, `s_shaped`: $f = (\cos\theta_\text{end})^{-1/2}$, with $\theta_\text{end}$ the last section's inclination; a placeholder until Cinco-Ley's slant skin (Economides et al. 1994, §5-4) is read.
- `vertical`: $f = 1$.

From D-20, D-21 (ruled 2026-11-06, checkpoints B1 and D) and D-45. Sources: [WP §4]; Economides et al. (1994), §2-7 (Eqs. 2-46 to 2-48, Example 2-10, Fig. 2-10); Joshi (1991), Ch. 3 and Ch. 10; Dikken (1990); Brekke and Lien (1994).

**Basis:** physics: the radial index is Darcy's law with the model's own $\mu_o$ and $B_o$, and a drain's is Joshi's index (Economides et al. 1994, Eqs. 2-46 to 2-48) times Dikken's friction efficiency in a closed form that stays within 2% of the coupled model, checked on Joshi's (1991) Table 3-2 and Example 10-5 and on Brekke and Lien's (1994) drain, with a factor of 1.5 for a perforated liner, within their 1.4 to 1.7 (`plans/evidence/drain_friction.py`); so friction, not a cap, limits a long drain. The drainage length that grows with the drain, Joshi's own form, and the liner taken as the tubing are a design choice among options that the evidence left open, kept simple (checkpoint B1, D-21). The slant factor $(\cos\theta_\text{end})^{-1/2}$ is a guess, a placeholder: Cinco-Ley's slant skin (Economides et al. 1994, §5-4), in hand but not yet read, would replace it. For the liner's liquid constants, the prototype's 0.5 mPa s for water and a density kept within 400 to 1100 kg/m³: basis not recorded.

### SMP-73 · Gas wells' radial coefficients

A gas or condensate well's Forchheimer coefficients of $p_r^2 - p_{wf}^2 = a\thinspace q + b\thinspace q^2$, with $q$ the standard gas rate (Economides et al. 1994, Eqs. 4-32 and 4-33), are for a vertical well

$$a_V = \frac{p_\text{sc} T_r\thinspace\mu_g Z\thinspace(\ln(r_e/r_w) - 0.75 + s)}{\pi k h_n T_\text{sc}}, \qquad b_V = \frac{a_V D}{\ln(r_e/r_w) - 0.75 + s},$$

in SI units, with $\mu_g$ and $Z$ of the reservoir gas at $(p_r, T_r)$, $(p_\text{sc}, T_\text{sc})$ the standard conditions (PVT-GAS-2) and $D$ of SMP-71 in s/Sm³ (1 d/Mscf = 3051 s/Sm³), so that the non-Darcy flow is a skin $D q$. The geometry's factors on them are set once, from the well's initial-stage draw:

- `high_angle`: $a = a_V/(J_H/J_V)$, with SMP-72's ratio; $b = b_V\thinspace(I_\text{ani} h_n/L_d)^2 + c_\text{lin} L_d/4$: the non-Darcy loss goes with the square of the rate per completed length, and the liner adds its friction to first order, with $c_\text{lin}$ at the heel's rate at the design drawdown, capped at the tubing's erosional rate, for the state's gas in a liner of the tubing's diameter, at most 157.1 mm (7 in), with SMP-72's friction factor (`drain_friction.py`);
- `build_and_hold`, `s_shaped`: $a = a_V/f$ and $b = b_V/f$, with SMP-72's $f$;
- `vertical`: $a = a_V$ and $b = b_V$.

The model's inflow is fitted to these at every reservoir state (SMP-117); they also give a gas well's design rates (SMP-75, SMP-80).

From D-23 (ruled 2026-11-06), D-62 (D2) and D-21. Sources: [WP §4]; Economides et al. (1994), §4-3 and §4-4 (Eqs. 4-32 to 4-34); Joshi (1991), Ch. 10.

**Basis:** physics: Forchheimer's coefficients with the non-Darcy flow as a rate-dependent skin (Economides et al. 1994, Eqs. 4-32 and 4-33), and for a drain the same derivation as SMP-72's, $a$ over Joshi's ratio and $b$ with the square of the rate per completed length plus the liner's friction to first order (`plans/evidence/drain_friction.py`, D-21). A slanted well's factor is SMP-72's guess, with the same replacement. For the liner's cap at 7 in, the prototype's: basis not recorded.

### SMP-74 · Inflow model

An oil well takes `vogel` (INF-1) if $p_\text{sat} \ge 0.8\thinspace p_r$ and its WLR is below 0.6, else `productivity_index` (INF-2); a gas or condensate well takes `forchheimer` (INF-10, feature 022). The rule is evaluated once, at the life's start (SMP-83), and the model is kept over the life in both datasets; its coefficients follow the reservoir's state (SMP-88).

From D-22 (ruled 2026-11-06), D-45 and D-60 (ruled 2026-11-05, checkpoint J1). Sources: Vogel (1968); `validity.md`, note 10. The 0.8 is an own choice.

**Basis:** physics for the models' ranges: Vogel's curve holds for solution-gas drive below the bubble point and not where water flows too (Vogel 1968; `validity.md`, note 10), and gas wells take Forchheimer's law (feature 022, checkpoint J1). The thresholds of 0.8 and 0.6 are guesses, own choices with no source: a composite inflow, straight above the bubble point and Vogel's below, would remove the need for them, and is not a model option. Keeping the model over the life is a design choice (D-45): a switch would step the rate by up to a factor of 1.8 with no change in the reservoir.

### SMP-75 · Tubing

The `completion` block takes the smallest inner diameter of {62.0, 76.0, 100.5, 124.3, 157.1} mm (2-7/8, 3-1/2, 4-1/2, 5-1/2 and 7 in tubing), and of 220.5 mm (9-5/8 in) too for gas and condensate wells, at which the no-slip mixture velocity at the design rate stays below the erosional velocity $v_e = 1.22\thinspace C/\sqrt{\rho_m}$ m/s, with $\rho_m$ in kg/m³ and $C \sim U(100, 150)$; with probability 0.3 (tail 0.5) it takes one size larger, where there is one.

- **The design point** is the initial state, when the tubing is run: $p_i$, $R_{si}$, no water for oils and a WGR of 10 for gases, the design drawdown $\delta_d p_i$, and the wellhead at $1.5\thinspace p_\text{sep} + 5\ \text{bar} + \Delta p_\text{fl}$ and $T_s + 0.6\thinspace(T_r - T_s)$. The design rate is an oil's $J\delta_d p_i$, and a gas well's SMP-73 rate at $p_i^2 - ((1 - \delta_d)\thinspace p_i)^2$ with its liquid, both with the geometry's factor before a drain's friction. A gas well's coefficients there take $Z$ at $p_i$, and $\mu_g$ and $D$ as at the draw's reservoir state (the start's, for a dataset's well).
- **Too small.** If no size fits, the well takes the largest and is flagged. The flag is a check: its share is measured, and a redraw is added only if it stays above a few percent.
- The tubing's erosional rate, the liquid rate at $v_e$ in the chosen tubing, caps the design rates of SMP-72, SMP-73 and SMP-80. The tubing is the well's for its life.

From D-25 (ruled 2026-11-06) and D-100 (checkpoint D1). Sources: [WP §6]; Guo, Lyons and Ghalambor (2007), Appendix B.

**Basis:** physics for the sizing: the erosional velocity with $C$ from 100 to 150 is API RP 14E's rule, as quoted in an open-access paper, the standard not being in hand ([WP §6]); it stands in for a distribution of tubing sizes, of which none is public (D-25). The sizes are data, standard tubing and casing-sized strings (Guo, Lyons and Ghalambor 2007, Appendix B); their inner diameters are the common weights' and are still to be checked against that appendix. The design point's constants (no water or a WGR of 10, a wellhead at $1.5\thinspace p_\text{sep} + 5$ bar and at 0.6 of the way from $T_s$ to $T_r$) and the chance of 0.3 of one size larger are guesses, own choices with no source and no replacement recorded; the tail's 0.5 is a design choice for coverage (SMP-53). Flagging a well too strong for the largest size, and not redrawing it, is a design choice for simplicity, with the flagged share measured (checkpoint D1, D-100).

### SMP-76 · Roughness

$\varepsilon \sim \text{LogU}$ by stage: initial $(1.5\cdot10^{-6},\ 3\cdot10^{-5})$ m, mid-life $(10^{-5},\ 6\cdot10^{-5})$ m, late $(2\cdot10^{-5},\ 1.5\cdot10^{-4})$ m; tail $\text{LogU}(1.5\cdot10^{-6},\ 1.5\cdot10^{-4})$ m, SMP-42's range. Friction from Chen's factor (FRIC-4). Over the life it ages from the initial draw's to the late draw's (SMP-88).

From D-26 (ruled 2026-11-06). Sources: [WP §6]. The split by stage is an own choice.

**Basis:** data for the whole range, SMP-42's, which brackets the published examples of 30 and 46 µm ([WP §6]). The split by stage is a guess, an own choice after the rule that roughness grows with corrosion and scale: measurements of new and used tubing (Farshad and Rieke 2006, on [WP]'s resource list, not in hand) would replace it. Chen's factor is physics (FRIC-4); the tail's log-uniform over the whole range is a design choice for coverage (SMP-53).

### SMP-77 · Choke

A Simpson choke (CHK-5) for every well, with the fully open bean $d = \min(\beta D,\ 101.6\ \text{mm})$, $\beta \sim U(0.15, 0.6)$ (tail $U(0.15, 0.75)$) of the tubing's diameter $D$, a discharge coefficient $C_D \sim U(0.75, 0.9)$, and $K_c = C_D\thinspace\pi d^2/4$. The profile, `chk_profile`, is one of CHK-7 to CHK-10 with equal shares. The Bernoulli choke (CHK-6) is not drawn.

From D-27 (ruled 2026-11-06, checkpoint E1). Sources: [WP §8]; Guo, Lyons and Ghalambor (2007), Eq. 5.4 (p. 5/60); [own analysis of BSEE's well tests] for the cap at 4 in.

**Basis:** data for the bean: the range of $\beta$ covers the textbook beans and v1's $K_c$ ([WP §8]), and the cap of 4 in is the largest choke tested in the Gulf of Mexico's well tests [own analysis of BSEE's well tests] (checkpoint E1); the discharge coefficient is data too, chosen within the published 0.62 to 0.90 (Guo, Lyons and Ghalambor 2007, Eq. 5.4). The lower end of 0.15 is a design choice for coverage of the textbook beans (E1); the tail's 0.75 is a design choice for coverage (SMP-53). The Simpson choke for every well and the profiles' equal shares, as in SMP-6, are a design choice: no public record of bean sizes or profiles over wells was found (D-27).

### SMP-78 · Gas lift at a stage

The `lift` block decides whether the well at its stage (SMP-51) is lifted:

- An oil well needs lift if its static column cannot reach the surface, $p_r < p_{s,\text{nom}} + g\rho_m L/10^5$, with $\rho_m$ the no-slip density of its stream at the mean of $p_r$ and $p_{s,\text{nom}}$ and the mean of $T_r$ and the design wellhead temperature $T_s + 0.6\thinspace(T_r - T_s)$. Such a well is lifted.
- Any other well is lifted with a probability by stage (initial, mid-life, late): oils offshore 0.20, 0.45 and 0.70; oils onshore 0.15, 0.25 and 0.35; gas and condensate wells 0, 0.05 and 0.15. Tail: 0.5 for oils and 0.25 for gas classes. No GOR limits lift.
- In the core, a well that this draw lifts, whose pressure-balance point $H_\text{bal}$ (SMP-79) lies above $0.3\thinspace L$, stays on natural flow: an operator does not lift a strong well.

A dataset's well takes its lift from these stage draws through its life (SMP-90).

From D-28 (ruled 2026-11-06, checkpoint F1: as drafted, no tuning by GOR quartile). Sources: [WP §1, §9]; Hasan and Kabir (2018), §12.2.1; von Flatern (2015); Goh et al. (2007). The stage shares are own choices.

**Basis:** physics for the static check, a no-slip column against the reservoir pressure. The stage shares are guesses, own choices: no public count of lifted wells exists ([WP §1]); they were later found within the Gulf of Mexico's lifted shares of platform oil wells by years on production and near its gas wells' [own analysis of BOEM's OGOR-A production], Bjarne ruled not to tune them to one region (checkpoint F1), and counts from more regions would replace them. The rule that leaves a strong well natural, with its 0.3 of $L$, is a guess with no source and no replacement recorded. Lift with no limit on the GOR is a design choice, Bjarne's goal 5; the tail's shares are a design choice for coverage (SMP-53).

### SMP-79 · Gas lift: injection depth and lift-gas temperature

A lifted well's operating valve lies at the true vertical depth $H$ below the wellhead:

- the packer at $L - \Delta z_\text{pk}$, $\Delta z_\text{pk} \sim U(30, 150)$ m;
- the injection pressure at the top of the modelled well, $p_\text{inj} \sim U(48, 85)$ bar at the surface for platform and onshore wells and $U(130, 210)$ bar at the tree for subsea wells; the valve's pressure drop $\Delta p_v \sim U(3.4, 13.8)$ bar; and the casing pressure at depth $H$ by Gilbert, $p_\text{inj}(1 + H/12\thinspace 192\ \text{m})$;
- the tubing's pressure below the valve at the lift's design point, $p_{wf} - G_l\thinspace(L - H)$, with $p_{wf} = p_r\thinspace(1 - U(0.3, 0.7))$ and $G_l$ the no-slip gradient of the stream at $(p_{wf}, T_r)$;
- $H_\text{bal}$, where the casing pressure less $\Delta p_v$ equals the tubing's pressure ($L$ where the two do not cross), and $H = \min\big(\max(H_\text{bal}, 0.3\thinspace L),\ L - \Delta z_\text{pk}\big)$.

The injection point's measured depth, `injection_md`, is the survey's MD at TVD $H$, interpolated linearly between stations. The lift gas's temperature is $T_{lg} = T_a(H) - \phi\thinspace\big(T_a(H) - T_s\big)$, $\phi \sim U(0, 0.3)$, with the ambient $T_a(H) = T_s + (T_r - T_s)\thinspace H/L$: from THM-9's default at the valve to 30% of the way towards the surface temperature.

From D-29 (ruled 2026-11-06, checkpoint F) and D-30 (ruled 2026-11-06). Sources: Golan and Whitson (1991), §5.3 (Eqs. 5.16 and 5.17, Example 5.7, pp. 528–536), with Gilbert's casing pressure; Guo, Lyons and Ghalambor (2007), Example 13.1; Hasan and Kabir (2018), §12.2.2; [own analysis of BSEE's well tests and Petrobras' 3W Dataset]. $\phi$ is an own choice.

**Basis:** physics for the method: the valve at the pressure balance of the casing's gas column, by Gilbert, against the flowing tubing (Golan and Whitson 1991, §5.3), with THM-9's ambient at the valve. The injection pressures are data: the published 48 to 76 bar (Golan and Whitson 1991, §5.3), widened to 85 bar with the casing pressures of the Gulf of Mexico's gas-lifted tests, and 130 to 210 bar at subsea trees [own analysis of BSEE's well tests and Petrobras' 3W Dataset], design values set with those analyses (D-29); the valve's drop is data too, the published 50 to 200 psi. The packer's 30 to 150 m is a guess after one example (Guo, Lyons and Ghalambor 2007, Example 13.1), and the lift's design drawdown of 0.3 to 0.7 and the least depth of 0.3 of $L$ are guesses with no source: no public distribution of injection depths was found ([WP §9], Gaps), and one would replace them. $\phi \sim U(0, 0.3)$ is a guess, an own choice (D-30): Hasan and Kabir (2018, §12.2.2) describe the annulus gas's heat exchange, no numbers were extracted, and their coupled balance would replace it.

### SMP-80 · Gas lift: capacity

The most lift gas a sample may inject is $w_{lg,\max} = k_q\thinspace\text{GLR}_ \text{cap}$ kg/s, with $k_q = \rho_{g,\text{sc}}\thinspace q_l/86\thinspace 400$ and $q_l$ the design liquid rate at the stage's state, $J\delta_d p_r$ for an oil and the liquid at SMP-73's gas rate at the design drawdown for a gas well, capped at the tubing's erosional rate. The injected gas–liquid ratio at capacity, $\text{GLR}_ \text{cap}$, is log-uniform over the part of $[30, 1000]$ Sm³/Sm³ that gives $w_{lg,\max}$ within $[0.1, 3.5]$ kg/s, core and tail. Where no part does, $w_{lg,\max}$ is 0.1 kg/s if $1000\thinspace k_q < 0.1$ kg/s, else 3.5 kg/s.

From D-31 (ruled 2026-11-06, checkpoint F2, truncated not clipped). Sources: [WP §9]; von Flatern (2015), Fig. 1; Guo, Lyons and Ghalambor (2007), Example 13.1; Golan and Whitson (1991), Example 5.8 (p. 536) and p. 538; Economides et al. (1994), Example 19-1 (p. 525); [own analysis of Petrobras' 3W Dataset].

**Basis:** data for the capacity's bounds of 0.1 and 3.5 kg/s, set with lifted subsea wells' injection rates [own analysis of Petrobras' 3W Dataset] (checkpoint F2), and for the range of the injected GLR, which widens the published examples of 27 to 77 Sm³/Sm³ and optima of 430 to 800 ([WP §9]) because lifted wells' rates spread wider. Its log-uniform shape is a guess: 3W gives no production rates, so its injected GLR is unknown, and injection and liquid rates of the same wells would replace it. Scaling the capacity to the well's design liquid rate is a design choice, Bjarne's goal 5; truncating the GLR, in place of clipping the capacity, is a design choice against point masses at the bounds (D-31).

### SMP-81 · Heat-transfer coefficient

$h = U_t k_e/(k_e + r_{ti} U_t T_D)$, the wellbore's overall coefficient in series with the formation's transient resistance (Hasan and Kabir 2012, Eq. 6), with $U_t \sim \text{LogU}(11, 85)$ W/(m² K), $k_e \sim U(1.5, 2.6)$ W/(m K), $r_{ti} = D/2$, and

$$T_D = \ln\big(e^{-0.2\thinspace t_D} + (1.5 - 0.3719\thinspace e^{-t_D})\sqrt{t_D}\big), \qquad t_D = \frac{\alpha\thinspace t_a}{r_h^2}$$

(Hasan and Kabir 2012, Eq. 2), with Ramey's diffusivity $\alpha = 1.03\cdot10^{-6}$ m²/s, the hole's radius $r_h$ = 0.108 m (0.156 m for tubing above 101 mm), and the well's age $t_a$ log-uniform over 30 to 365 days (initial), 1 to 8 years (mid-life) or 5 to 25 years (late). Tail: $h \sim U(2, 40)$ W/(m² K). Over the life the age grows (SMP-88).

From D-32 (ruled 2026-11-06). Sources: [WP §10]; Hasan and Kabir (2012), Eqs. 2 and 6; Ramey (1962), p. 431; Hasan and Kabir (2018), Table 6.1 and Fig. 7.39.

**Basis:** physics for the form: the wellbore's coefficient in series with the formation's transient resistance (Hasan and Kabir 2012, Eqs. 2 and 6), with Ramey's (1962, p. 431) diffusivity. Data for the middle of the ranges: $U_t$ of 36 to 57 W/(m² K) along two wells (Hasan and Kabir 2018, Table 6.1 and Fig. 7.39) and Ramey's $k_e$ of 2.42 W/(m K). The ends of both ranges and the two hole sizes are guesses: no distribution of $U_t$ over completions was found, and the papers on [WP]'s resource list (Willhite 1967; Alves et al. 1992) would replace them. The ages by stage are a design choice, the meaning of the stages (SMP-91); the tail's range, which keeps v1's 10 to 40 W/(m² K), is a design choice for coverage (SMP-53).

### SMP-82 · Heat capacities

From the well's initial-stage draw, constant along the well and over its life, each at the representative temperature $T_\text{rep} = T_s + 0.8\thinspace(T_r - T_s)$:

- oil, by Cragoe (1929, Eq. 5): $c_{po} = 4186.8\thinspace(0.388 + 0.00045\thinspace t_F)/\sqrt{\gamma_o}$ J/(kg K), with $t_F$ the temperature in °F and $\gamma_o = 141.5/(131.5 + \text{API})$;
- water, a quadratic through Sharqawy et al.'s (2010, Eq. 9) values at 20 °C: $c_{pw} = 4190 - 5.62\thinspace S + 0.0076\thinspace S^2$ J/(kg K), with $S = 1000\thinspace w_s$ in g/kg;
- gas: methane's $c_p$ at $\big((p_r + p_{s,\text{nom}})/2,\ T_\text{rep}\big)$, bilinear in the table below (linear in $p$ at each $T$, then in $T$) and clamped to it, times $(0.554/\gamma_g)^{0.4}$.

| $T$ (K) | 1 bar | 101 bar | 201 bar | 401 bar |
|---|---|---|---|---|
| 300 | 2236 | 3011 | 3549 | 3336 |
| 380 | 2466 | 2801 | 3079 | 3234 |

From D-33 (ruled 2026-11-06). Sources: [RF §10], with NIST's methane values in J/(kg K); Cragoe (1929), Eq. 5; Sharqawy, Lienhard and Zubair (2010), Eq. 9. The factor for heavier gases is an assumption.

**Basis:** physics for the oil (Cragoe 1929, Eq. 5) and the water (a quadratic within 5 J/(kg K) of Sharqawy et al.'s 2010 values); data for methane, NIST's values ([RF §10]). The factor $(0.554/\gamma_g)^{0.4}$ for heavier gases is a guess, an assumption: no open correlation for a natural gas's $c_p$ against gas gravity at pressure was found ([RF §10], Gaps), and one would replace it. The representative temperature at 0.8 of the way from $T_s$ to $T_r$ is a guess too, standing in for the $c_p(p, T)$ that the model does not have.

## v2: one life for both datasets

### SMP-83 · A well's life

Every well has one life, which both datasets use: an `nsol-2` well lives it, and a `sol-2` well is the same life at one date (SMP-92).

- **Start**, $t = 0$: the well's initial-stage draw, its $r_0 = p_r/p_i$ and its water (SMP-69).
- **End**, $t = T_\text{life}$, of a core well, its reservoir run towards exhausted: under depletion $r_1 = \min\big(U(0.15, 0.3), r_0\big)$ for oils and $\min\big(U(0.15, 0.2), r_0\big)$ for gas classes, with the late-stage draw's water; under pressure support $r_1 = \min(r_{p,L}, r_0)$, with $r_{p,L}$ the late-stage draw's, and the water at its cap: a WLR of 0.98 for oils, and a WGR of 10,000 Sm³ per 10⁶ Sm³ for gas classes, where the well waters out (D-107; D-105 had 1000).
- **A tail well** starts from a core initial state drawn for it, $r_0 \sim U(0.85, 1)$ under depletion and $U(0.97, 1)$ under pressure support, with a WLR of 0 with probability 0.5 and else $U(0, 0.1)$, or a WGR of $\text{LogU}(2, 20)$; it ends at its tail state, $r_1 = \min(r_{p,\text{tail}}, r_0)$, with its tail water.
- The life stops where the well stops paying: `nsol-2`'s life scan (SMP-101) and `sol-2`'s producibility filter (SMP-103) apply the economic floor along it.

The trends between the start and the end (SMP-84 to SMP-90) take their parameters from the `evolution` block.

From D-105 (ruled 2026-11-06: one life for both datasets), D-41, D-42 and D-107 (ruled 2026-11-10, L4). Sources: [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well]; [own analysis of BOEM's sands atlas].

**Basis:** one life for both datasets is a design choice for simplicity and consistency (Bjarne's ruling, D-105). How far a core life runs is data: the ends, an exhausted reservoir under depletion and a watered-out well under support, were set on the rework's pilots so that a life's last fifth falls as the regions' stopped wells' does [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well], kept outside the repo; the WGR of 10,000 at which a supported gas well waters out is a design value chosen with the upper tail of the Gulf of Mexico's gas wells' WGR in their later years, and with its gas sands by drive, which recover least under water drive [own analysis of BOEM's OGOR-A production; own analysis of BOEM's sands atlas] (D-107). The depletion ranges themselves reuse the tail's floor of 0.15 and the bottoms of SMP-69's late ranges, with no statistics of their own; the WLR of 0.98 is physics, the model's cap (`validity.md`). A tail well's start from a core state is a design choice for coverage (SMP-53).

### SMP-84 · Life span and the depletion clock

$T_\text{life} \sim \text{LogU}(3, 30)$ years. The trends follow the reservoir's cumulative production, normalised to the life span,

$$F(t) = \frac{N(t)}{N(T_\text{life})}, \qquad N(t) = \begin{cases} t, & t \le t_\text{pl} \\ t_\text{pl} + \dfrac{1 - (1 + b_\text{A} D_\text{A}\tau)^{1 - 1/b_\text{A}}}{(1 - b_\text{A})\thinspace D_\text{A}}, & \tau = t - t_\text{pl} > 0 \end{cases}$$

a plateau of unit rate and then Arps's hyperbolic decline, exponential, $N = t_\text{pl} + (1 - e^{-D_\text{A}\tau})/D_\text{A}$, where $b_\text{A} < 10^{-6}$. The plateau $t_\text{pl} = U(0, 0.25)\thinspace T_\text{life}$; $b_\text{A} \sim U(0, 0.4)$ under depletion and $U(0.3, 0.5)$ under pressure support; and $D_\text{A}$ from the decline's rate at the end of the life over the plateau's, $q_\text{end} \sim \text{LogU}(0.03, 0.3)$: $D_\text{A} = (q_\text{end}^{-b_\text{A}} - 1)/(b_\text{A}\tau_\text{end})$, or $-\ln q_\text{end}/\tau_\text{end}$ where $b_\text{A} < 10^{-6}$, with $\tau_\text{end} = T_\text{life} - t_\text{pl}$. The clock depends on no solve, so the controls stay open loop.

From D-41 (ruled 2026-11-06, with the steeper decline and lives shorter than 8 years). Sources: Ahmed (2006), Eqs. 16-1 to 16-4 (pp. 1238–1239); Golan and Whitson (1991), p. 385; Miller, Sorrell and Speirs (2009), Tables 4.2 to 4.4 [RF §11]; [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well].

**Basis:** data for the span, 3 to 30 years, whose lower end follows the many wells of the regions that stop before 8 years [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well] (D-41 as ruled), and for $q_\text{end}$, set on the rework's pilot against the same wells' decline, kept outside the repo; data also for $b_\text{A}$, the published 0 to 0.5, about 0.3 for solution-gas drive and 0.5 for water drive (Golan and Whitson 1991, p. 385). Arps's decline is physics, a published empirical form (Ahmed 2006, Eqs. 16-1 to 16-4). The plateau of up to a quarter of the life is a guess after fields' plateaus ([RF §11]; Miller, Sorrell and Speirs 2009, Table 4.4): plateau lengths per well are missing ([RF §11], Gaps) and would replace it. A clock that follows no solve is a design choice, the Goals' open loop.

### SMP-85 · Reservoir pressure along the life

- Oils, and every well under pressure support: $p_r(t) = p_i\big(r_1 + (r_0 - r_1)(1 - F)^\gamma\big)$, $\gamma \sim U(1, 2)$.
- Gas and condensate wells under depletion: $p/Z$ linear in $F$, $p_r/Z(p_r) = (1 - F)\thinspace p_0/Z_0 + F\thinspace p_1/Z_1$, with $p_0 = r_0 p_i$, $p_1 = r_1 p_i$ and the well's DAK $Z$ (PVT-GAS-11) at $T_r$, solved for $p_r$ by linear interpolation in $p/Z$ between 200 pressures from $0.5\thinspace p_1$ to $1.02\thinspace p_0$.

From D-42 (ruled 2026-11-06). Sources: Ahmed (2006), Eq. 13-8 (p. 861); Golan and Whitson (1991), Example 4.11 (Table E4.11a, p. 454).

**Basis:** physics for gas and condensate wells under depletion: the gas material balance, $p/Z$ linear in the cumulative production (Ahmed 2006, Eq. 13-8). The form for oils and for supported wells, with $\gamma \sim U(1, 2)$, is a guess after one textbook well, which it fits at $\gamma$ = 1.40 (Golan and Whitson 1991, Example 4.11): a material balance of solution-gas drive, D-42's alternative, would replace it for oils under depletion, and the plans name none for supported wells.

### SMP-86 · Water along the life

- **Oils.** The WLR stays at the start's until breakthrough at the recovery $F_\text{bt}$, jumps to $\text{WLR}_ \text{bt}$, and then rises with its logit linear in $F$ to the end's $\text{WLR}_ 1$:

  $$\text{logit}\thinspace\text{WLR}(F) = \text{logit}\thinspace\text{WLR}_ \text{bt} + \big(\text{logit}\thinspace\text{WLR}_ 1 - \text{logit}\thinspace\text{WLR}_ \text{bt}\big)\frac{F - F_\text{bt}}{1 - F_\text{bt}}, \qquad F \ge F_\text{bt},$$

  never below the start's WLR, with $\text{WLR}_ \text{bt}$ at most $\text{WLR}_ 1$; where $\text{WLR}_ 1$ is not above the start's, the WLR stays at the start's. Under pressure support $F_\text{bt} \sim U(0.05, 0.4)$ and $\text{WLR}_ \text{bt} \sim U(0.05, 0.3)$; under depletion $U(0.2, 0.7)$ and $U(0.02, 0.1)$.
- **Gas classes.** The WGR stays at the start's $\text{WGR}_ 0$ until $F_\text{bt}$ and then rises log-linearly in $F$ to the end's, $\text{WGR}_ 0\thinspace(\text{WGR}_ 1/\text{WGR}_ 0)^{(F - F_\text{bt})/(1 - F_\text{bt})}$; where the end's is not above the start's, it stays at the start's.

From D-43 (ruled 2026-11-06). Sources: Ahmed (2006), Eq. 5-28 (p. 309), Eqs. 14-10 and 14-22 and Fig. 14-12 (pp. 935, 941), pp. 970 and 983; [RF §9]; [own analysis of Sodir's FactMaps].

**Basis:** physics for the shape after breakthrough: with a semilog ratio of relative permeabilities, the logit of the fractional flow of water is linear in the water saturation (Ahmed 2006, Eqs. 5-28 and 14-22), taken as linear in $F$ by an own step. The breakthrough recoveries, the jumps and the gases' log-linear WGR are guesses, own choices: the front's jump (Ahmed 2006, p. 970) is taken as spread by a layered reservoir, by how much is not sourced; the water cut they give by years on production lies within the regions' per-well medians [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well], and per-well water-cut curves would replace them.

### SMP-87 · GOR along the life

$R_{go}(t)$ is SMP-70's at $p_r(t)$, with the start's $R_{si}$, $p_\text{sat}$, $k_g/k_o$ blend and condensate factor $c_k$: flat while the reservoir is above its saturation pressure, then rising, so that the pressure links the three trends. Wet and dry gas keep $R_{si}$.

From D-44 (ruled 2026-11-06 with checkpoint C1). Sources: McCain (1990), pp. 151, 153; Golan and Whitson (1991), Example 4.11 (p. 454); Craft and Hawkins (1991), Ch. 9.

**Basis:** physics: SMP-70's curves followed along $p_r(t)$, each part as classed there, so that the pressure links the three trends; leaving out the late fall is physics too, since the material balance puts it below every core state (D-44).

### SMP-88 · Well properties along the life

- **Heat transfer:** SMP-81's $h$, with the start's $U_t$, $k_e$ and age $t_{a,0}$, at the age $t_{a,0} + t$. A tail well keeps its tail $h$.
- **Roughness:** $\varepsilon(t) = \varepsilon_0\thinspace\big(\max(\varepsilon_L, \varepsilon_0)/\varepsilon_0\big)^{t/T_\text{life}}$, from the initial draw's $\varepsilon_0$ towards the late draw's $\varepsilon_L$.
- **Inflow coefficients:** at the trends' state at $t$, without the drift of SMP-96 or SMP-100: an oil's $J$ by SMP-72 at $\big(p_r(t), R_{go}(t)\big)$ and the WLR, with the start's geometry factor $f$; a gas well's coefficients by SMP-117 at $p_r(t)$ and $R_{go}(t)$.
- **Fixed for the life:** the inflow model (SMP-74), the survey, the tubing, the choke's bean and profile, $T_r$, $T_s$, the salinity, the gravities, $R_{si}$, $p_\text{sat}$, the non-Darcy coefficient, the geometry's factors and the heat capacities.

From D-45 (ruled 2026-11-06). Sources: Hasan and Kabir (2012), Eq. 2; [WP §6, §10].

**Basis:** physics for the heat transfer's fall with age (Hasan and Kabir 2012, Eq. 2) and for the inflow's coefficients at the trends' state (SMP-72, SMP-117). The roughness's log-linear path between its two stage draws is a guess, an own form after the rule that roughness grows with corrosion and scale ([WP §6]), with SMP-76's gap and replacement. What stays fixed is a design choice for simplicity (D-45): no workover is drawn, and the inflow model is kept so that no switch steps the rate.

### SMP-89 · Separator along the life

$p_{s,\text{nom}}(t)$ is SMP-66's with the factor $r_s(t)$: 1 until $t_1$, the mid-life draw's $r_s$ until $t_2$, and the smaller of it and the late draw's after, with $t_1 < t_2$ the sorted pair of two draws of $U(0.25, 0.9)\thinspace T_\text{life}$. A tail well scales its $p_{s,\text{nom}} - \Delta p_\text{fl}$ by the same factor, with $p_{s,\text{nom}}(t)$ at least 1.5 bar.

From D-46 (ruled 2026-11-06). Sources: [WP §7].

**Basis:** a guess: two step-downs at dates of $U(0.25, 0.9)\thinspace T_\text{life}$, after two fields that moved to lower stages late in life (trade press, [WP §7]); no public record of separator pressures over wells' lives was found, and one would replace it. The steps' sizes are SMP-66's stage factors.

### SMP-90 · Gas lift along the life

- A well that its initial-stage draw lifts (SMP-78) is lifted from the start, with that draw's design (SMP-79, SMP-80): `lift_start_rule` is `from_start`.
- For any other well lift starts at the earliest of:
  1. for an oil well, the first of 41 dates $t = i\thinspace T_\text{life}/40$ at which SMP-78's static check fails at the trends' state and $p_{s,\text{nom}}(t)$ (`static_check`);
  2. for a well that its late-stage draw lifts, a drawn date $U(0.2, 0.9)\thinspace T_\text{life}$ (`drawn_time`);
  3. in `nsol-2`, for an oil well or a well that its late-stage draw lifts, the last scan point before the natural well's best rate first fails (SMP-101; `end_of_natural_flow`).

  Its design is the late-stage draw's (SMP-79, SMP-80), made as if lifted where that draw leaves the well natural. A well that none of these start stays natural.
- Before lift starts the well has no injection point and $w_{lg} = 0$; from the start, its controls of SMP-94 or SMP-98.
- A `sol-2` oil well that is natural at its date and cannot flow at its best rate is lifted by SMP-103 (`no_natural_flow`).

From D-49 and D-84 (ruled 2026-11-06), D-101 (ruled 2026-11-06, checkpoint F2) and D-105. Sources: Golan and Whitson (1991), Example 4.11; Goh et al. (2007); [WP §9].

**Basis:** physics for the static check, SMP-78's, on a grid of 41 dates. Starting lift where natural flow ends, or where a `sol-2` well cannot flow, with the late-stage draw's design for a lift that starts later, is a design choice after operating practice (Golan and Whitson 1991, Example 4.11; [WP §9]; D-49; checkpoint F2). The drawn date of $U(0.2, 0.9)\thinspace T_\text{life}$ is a guess with no source and no replacement recorded.

### SMP-91 · Life stage of a date

A date $t$ is `initial` in the first year ($t < 1$), `late` from half the life span ($t \ge T_\text{life}/2$), and `mid_life` between. An `nsol-2` sample carries the stage of its date.

From D-41 (ruled 2026-11-06). Sources: `sampling-draft.md`, The well and its states.

**Basis:** a design choice: the stages are labels of this procedure, with the first year and half the life span as their limits, after SMP-81's ages (D-41).

## v2: `sol-2` samples

### SMP-92 · A `sol-2` well

A `sol-2` well is its life (SMP-83) at one date $t^\ast$, drawn from the date's generator (SMP-109) uniformly within its drawn stage (SMP-56): $[0, 1)$ for `initial`, $[1, T_\text{life}/2)$ for `mid_life` and $[T_\text{life}/2, T_\text{life}]$ for `late`, in years. A tail well's date is $T_\text{life}$, the end of its life, which holds its tail state. The well's nominal state is the life's at $t^\ast$ without drift: $p_{r,\text{nom}} = p_r(t^\ast)$, $R_{go,\text{nom}}$ and the water (SMP-85 to SMP-87), $p_{s,\text{nom}}(t^\ast)$, $h$, $\varepsilon$ and the inflow's coefficients (SMP-88, SMP-89), and the lift at $t^\ast$ (SMP-90). Its `saturation` label is at $p_{r,\text{nom}}$, and its `life_state` follows from its stage and drive.

From D-105 (ruled 2026-11-06). Sources: as SMP-83.

**Basis:** a design choice (D-105): a date drawn uniformly within the drawn stage makes the `sol-2` wells a well stock at a date, each at a point of its life that still pays; a tail well's date at the end of its life keeps its tail state.

### SMP-93 · Choke position

$u = 1$ with probability 0.05, else $u \sim U(0.05, 1)$, independent per slot, core and tail alike.

From D-37 (ruled 2026-11-06). Sources: the plan's Goals (goal 4: modest weight at a fully open choke, close to uniform); SMP-18 for the smallest opening; Guo, Lyons and Ghalambor (2007), §5.1 (p. 5/60); [own analysis of Petrobras' 3W Dataset]; [own analysis of NSTA's production by well].

**Basis:** a design choice, Bjarne's goal 4: a choke close to uniform, because a wide span serves the machine-learning tasks, with a modest mass at fully open; no public distribution of choke openings in operation was found ([WP §8]). The mass of 0.05 lies between the pooled and the typical well's share of time fully open in normal operation [own analysis of Petrobras' 3W Dataset], and the smallest opening is v1's (SMP-18).

### SMP-94 · Lift-gas rate

A lifted well has $w_{lg} = 0$ with probability 0.20, else $w_{lg} = v\thinspace w_{lg,\max}$ with $v \sim U(0, 1)$. A natural well has $w_{lg} = 0$.

From D-38 (ruled 2026-11-06, with 0.20). Sources: [WP §9]; von Flatern (2015), Fig. 1; Golan and Whitson (1991), §5.3 (p. 538); Asheim (1988), p. 1452.

**Basis:** a lift-gas rate uniform to the capacity is a design choice for coverage: some samples lie past the optimum, which a lift-optimisation task needs to see (D-38). The share of 0.20 with the lift gas off is a guess: no public share of time with the lift gas off was found, the draft's own choice was 0.10, and Bjarne's ruling set 0.20 (D-38); a record of lifted wells' injection over time would replace it.

### SMP-95 · Separator pressure

$p_s \sim U\big(\max(0.9\thinspace p_{s,\text{nom}},\ 1.5\ \text{bar},\ \Delta p_\text{fl}),\ 1.1\thinspace p_{s,\text{nom}}\big)$, or $1.1\thinspace p_{s,\text{nom}}$ where that interval is empty.

From D-39 (ruled 2026-11-06). Sources: SMP-20; [WP §7]; Guo, Lyons and Ghalambor (2007), §5.2 (p. 5/60).

**Basis:** a guess kept from v1 (SMP-20): ±10% around the nominal pressure; no public spread of separator or manifold pressures was found ([WP §7]), and one would replace it. The floors are SMP-66's.

### SMP-96 · Drift of the reservoir

Around the well's nominal state (SMP-92):

- $p_r = p_{r,\text{nom}}\thinspace U(0.98, 1.02)$;
- the producing GOR: SMP-70's at the drifted $p_r$, times $U(0.95, 1.05)$;
- the water: an oil's WLR $\min\big(0.98,\ \text{WLR}_ \text{nom}\thinspace U(0.95, 1.05)\big)$; a gas well's WGR $\text{WGR}_ \text{nom}\thinspace U(0.95, 1.05)$, and its WLR from it and the drifted GOR.

$T_r$, $T_s$, $T_{lg}$, $h$, $\varepsilon$ and the inflow's coefficients do not drift. An attempt with $p_r \le p_s$ has no operating point (SMP-104).

From D-40 (ruled 2026-11-06). Sources: the plan's Goals (±2% and ±5%); SMP-21 and SMP-22; McCain (1990), pp. 151, 153.

**Basis:** the sizes of the drift, ±2% and ±5%, are a design choice of the Goals, as in v1 (SMP-21, SMP-22), kept for variation with no public source; tying the GOR to the drifted pressure is physics, SMP-70's curve, and the cap of 0.98 is the model's (`validity.md`).

## v2: `nsol-2` samples

### SMP-97 · Dates and slots

A life of 200 slots spans $[0, T_\text{end}]$, the life scan's end (SMP-101): 160 operating samples at $t_j = j\thinspace T_\text{end}/159$, $j$ = 0 to 159, and 8 test periods of 5 samples at $t_m = (m - 1/2)\thinspace T_\text{end}/8$, $m$ = 1 to 8. The slots are numbered in the order of their dates. A sample's reservoir state is the trends' at its date (SMP-85 to SMP-89), with its date's drift (SMP-100); a test's samples share their date's state. The controls' generator (SMP-109) draws the plan of every slot's controls, separator pressure and drift, in slot order, before any slot is solved.

From D-48 and D-84 (ruled 2026-11-06). Sources: Golan and Whitson (1991), §5.3. The schedule of tests is a design choice for coverage.

**Basis:** a design choice: dates spread evenly so that a well's samples span its life (the Goals), and 8 tests of 5 samples, a fifth of the slots, for coverage and for the excitation that makes a well's response identifiable; multi-rate tests are practice (Golan and Whitson 1991, §5.3), and no public schedule of tests was found.

### SMP-98 · Control history

The choke and the lift gas follow set points $u_\text{sp}$ and $w_\text{sp}$, constant between changes and open loop:

- **Start:** $u_\text{sp} = 1$ with probability 0.05, else $U(0.05, 1)$. When lift starts, $w_\text{sp} = U(0.1, 1)\thinspace w_{lg,\max}$.
- **Changes:** at each operating sample after the first, each set point changes with probability $p_c \sim \text{LogU}(0.5, 0.95)$, drawn once per well; the lift gas's only while lift is installed.
- **A change** is, with probability 0.9, a step of $U(-0.1, 0.1)$, times $w_{lg,\max}$ for the lift gas, reflected into its range; else a new draw: the choke 1 with probability 0.05, else $U(0.05, 1)$; the lift gas 0 with probability 0.20, else $U(0.1, 1)\thinspace w_{lg,\max}$.
- **Ranges:** the choke's $[0.05, 1]$; the lift gas's $[0.1, 1]\thinspace w_{lg,\max}$ between changes, so that only a new draw shuts it in.
- The set points never follow a solve; a redraw (SMP-105) changes its own slot only.

From D-47 (ruled 2026-11-06, with the lift gas's floor of the same day). Sources: [own analysis of Petrobras' 3W Dataset]. The step's size and the probabilities are own choices, set on the rework's pilot.

**Basis:** data for how often a set point changes: operators move production chokes every day or so in normal operation [own analysis of Petrobras' 3W Dataset], so it changes at most samples (D-47 as ruled). The probability $p_c$, the step's size and the share of new draws are guesses, own choices set on the rework's pilot: the 3W windows last hours to weeks, so they do not show how far a set point travels over a life, and long histories of set points would replace them. Open-loop set points are a design choice of the Goals; the lift gas's floor of 0.1 of the capacity is a design choice from the rework's pilot, where a walk near zero left lifted wells at the redraw limit (D-47). The masses of 0.05 and 0.20 are SMP-93's and SMP-94's.

### SMP-99 · Test periods

A natural well sweeps its choke in every test. A well with lift installed sweeps its choke in tests 1, 3, 5 and 7 and its lift gas in tests 2, 4, 6 and 8:

- **choke sweep:** $u = 1$, then one draw in each quarter of $[0.05, 1)$ from the top down, $u = 0.05 + 0.95\thinspace x$ with $x \sim U(3/4, 1)$, $U(1/2, 3/4)$, $U(1/4, 1/2)$ and $U(0, 1/4)$ in turn, at the operating lift-gas rate;
- **lift sweep:** $w_{lg} = w_{lg,\max}$, then $x\thinspace w_{lg,\max}$ with $x$ drawn the same way, at the operating choke.

From D-48 (ruled 2026-11-06). Sources: Golan and Whitson (1991), §5.3 (the lift performance curve).

**Basis:** a design choice for coverage: a sweep from the top through each quarter of the range gives each well its choke and lift performance curves, as multi-rate tests do in practice (Golan and Whitson 1991, §5.3); the alternation of the two sweeps keeps a test at 5 samples (D-48).

### SMP-100 · Drift between dates

Each operating sample's GOR and water are their trends' values times $U(0.95, 1.05)$ each; a test's five samples share one such pair, drawn at its first sample. An oil's WLR is capped at 0.98, and a gas well's WLR follows from its WGR and GOR. $p_r$ follows its trend without noise. Each slot's separator pressure is SMP-95's around $p_{s,\text{nom}}(t)$.

From D-40 (ruled 2026-11-06). Sources: the plan's Goals.

**Basis:** a design choice of the Goals, as SMP-96's sizes; a reservoir pressure without noise follows from its smooth trend, and a test's shared state from its single date.

### SMP-101 · Life scan and the end of a life

Before its slots are drawn, the well's best rate (SMP-103) is solved at the 21 dates $t_i = i\thinspace T_\text{life}/20$, with the lift that the well has at each (SMP-90).

- A point fails where its status (SMP-103) is `below_floor`, `no_operating_point` or `water_cut`. A point with the status `solver_failure` is skipped.
- At a point $i > 0$ where the well is natural and its status is `below_floor` or `no_operating_point`, rule 3 of SMP-90 starts lift at $t_{i-1}$ where that is earlier than its start so far, and the point is solved again with lift.
- The life ends at $T_\text{end}$, the last point that passes before the first that fails, or at $T_\text{life}$ if none fails.
- A well whose first point fails is dropped as `not_producible`, or as `best_rate_solver_failure` (SMP-103); one whose life ends at $T_\text{end} = 0$ is dropped as `short_life`.

From D-50 and D-84 (ruled 2026-11-06) and D-49. Sources: D-34's floor; [WP §11]. The scan's step of 5% of the life is the ruled one; a finer scan is the named fix if dead dates (SMP-106) stay above their target.

**Basis:** a design choice of the generation: the scan's 21 points are the ruled resolution, and ending the life at the last point that passes gives every date an operating point (D-84); the floor it applies is SMP-102's.

## v2: producibility

### SMP-102 · Economic floor

The floor in oil equivalent is 1.6 Sm³/d (10 BOE/d) for platform and onshore wells and 20 Sm³/d for subsea wells. A well whose best rate is below half its floor is not produced, and neither is an oil well whose WLR is above 0.98.

From D-34 and D-50 (ruled 2026-11-06, checkpoint D3). Sources: [WP §11]; EIA (2025), p. 10 and Table B32; [own analysis of BOEM's OGOR-A production and ANP's production by well] for the subsea floor.

**Basis:** data for the floors: 1.6 Sm³/d is the stripper-well limit of 10 bbl/d (EIA 2025, p. 10), down to which, with the margin, stopped platform wells' final rates reach, and 20 Sm³/d is a design value set with the final rates of stopped subsea wells [own analysis of BOEM's OGOR-A production and ANP's production by well], kept outside the repo (checkpoint D3). The margin of half the floor is a design choice of the Goals, so that marginally economic wells stay in; the limit of 0.98 is physics, the edge of the model's valid range (`validity.md`).

### SMP-103 · Best rate and the producibility filter

A well's best rate at a state is the largest oil equivalent among its operating points at a fully open choke ($u = 1$), $p_s = p_{s,\text{nom}}$ and the state's $p_r$, GOR and water without drift, solved at $w_{lg} \in \lbrace 0,\ 1/2,\ 1\rbrace\thinspace w_{lg,\max}$ for a lifted well and at $w_{lg} = 0$ for a natural one. Its status is, in this order: `water_cut` for an oil whose WLR is above 0.98; `pass` if it has an operating point and the best rate is at least half the floor; `below_floor` if it has one; `solver_failure` if no solve gave an operating point and one was a solver failure (SMP-104); else `no_operating_point`.

- **`sol-2`:** at the well's nominal state (SMP-92), before its slots. A natural oil well whose status is `below_floor` or `no_operating_point` takes its life's lift design (SMP-90), with `lift_start_rule` `no_natural_flow`, and is solved again. A well that then does not pass is dropped: as `best_rate_solver_failure` for the status `solver_failure`, else as `not_producible`.
- **`nsol-2`:** at the start, the life scan's first point, with the same outcomes, and then along the life by SMP-101.

From D-50 (ruled 2026-11-06, checkpoint D3), D-55, D-101 (checkpoint F2) and D-84. Sources: D-34; Golan and Whitson (1991), §5.3, whose lift performance curve is flat near its optimum.

**Basis:** a design choice of the generation: the best rate at a fully open choke and three lift-gas rates stands for the well's potential, three points lying close to the optimum because the lift performance curve is flat there (Golan and Whitson 1991, §5.3); lifting a natural oil well that cannot flow, before dropping it, follows operating practice (D-101, checkpoint F2).

## v2: generation

### SMP-104 · Attempts and outcomes

An attempt is one draw of a slot's inputs, solved once by the Rust core. Its outcome is one of:

- `operating_point`: the root set's operating point, its stable root (SOL-4, SOL-6), whose row is valid; the slot takes it;
- `no_operating_point`, the well cannot flow at these inputs (SOL-5): $p_r \le p_s$, without a solve (reason `p_r_le_p_s`); or a search that found no root and rejected no sign change, whatever its non-finite scan points (`no_root`);
- `solver_failure`: an exception, a refusal of the inputs by the model included (`exception`); a row with a non-finite value, a negative liquid or gas rate, or a mass fraction outside $[0, 1]$ (`invalid_row`); a search with a rejected sign change and no root (`incomplete_scan`); or roots with no stable one (`unstable_only`), which a continuous $R(p_0)$ cannot give.

No other backend retries a solver failure: it is redrawn and logged as a missing operating point is, and counted apart.

From D-53 as revised by D-102 (ruled 2026-11-06, checkpoint H2), D-54 and D-82 (checkpoint H1: the CasADi retry dropped). Sources: `specs/model/solution.md` (SOL-2, SOL-3, SOL-5, SOL-7); the Rust core's search counts (`rust/src/march.rs`).

**Basis:** physics for the two outcomes: a root needs $p_0 < p_r$ and $p_N > p_s$ (SOL-2), and a continuous $R(p_0)$ has at least as many stable roots as unstable ones, so a set of unstable roots alone has missed one (SOL-3, SOL-5). Counting a scan with non-finite points, no root and no rejected sign change as no operating point is a design choice resting on the pilots, where the CasADi backend found no root in 70 of 72 such attempts (D-102); so is dropping the retry, which had no measured gain (constitution, principle 7; D-54).

### SMP-105 · Slots and redraws

Each well has 200 slots, $j$ = 0 to 199, and each slot ends with an operating point. An attempt without one is redrawn as attempt $a + 1$ of the same slot.

- **Passes.** Pass 0 solves attempt 0 of every slot, and pass $a$ attempt $a$ of each slot still without an operating point, in the order of $j$. A slot's draws do not depend on the other slots' outcomes.
- **`sol-2`:** a redraw draws the slot's whole draw again (SMP-93 to SMP-96), so the samples follow the drawn distribution conditioned on an operating point.
- **`nsol-2`:** a redraw keeps the slot's date and everything that follows from it: the reservoir's state with the date's drift, $h$, $\varepsilon$, the inflow's coefficients, $p_{s,\text{nom}}$ and whether lift is installed. It draws the separator pressure again (SMP-95), and both controls from the slot's planned values (in a test, the sweep's point): each, with probability 0.6, a step of $U(-0.2, 0.2)$, times $w_{lg,\max}$ for the lift gas, reflected into its range (the lift gas's $[0.1, 1]\thinspace w_{lg,\max}$), else a new draw as in SMP-98; the lift gas only where lift is installed. The redrawn controls hold for that slot only.

From D-52 (ruled 2026-11-02), D-80 and D-83 (ruled 2026-11-06, with R4's step kept for redraws). Sources: `sampling-draft.md`, Slots and outcomes and Redraws in `nsol-2`.

**Basis:** a design choice: 200 operating points per well, each failed draw redrawn and logged (Bjarne's ruling, D-52), from generators that make a run reproducible and a slot independent of the others (D-80); an `nsol-2` redraw's step of 0.2 with probability 0.6 is R4's change, kept after smaller steps left wells at the redraw limit in the rework's pilot (D-83).

### SMP-106 · Limits and dropped wells

A well is dropped, and the next drawn well takes its place (SMP-110), when it reaches a limit: 50 attempts in a slot, 400 redraws, or 10 solver failures. Its outcome is one of:

| Outcome | When |
|---|---|
| `accepted` | 200 operating points |
| `invalid_input` | the well fails `configurations.check(wp, 'v2.0.0')` at its nominal state (`nsol-2`: its start), or its table does not cover its $p_r$, $T_r$ or GORs |
| `table_failed` | its table could not be built, or is refused (SMP-116) |
| `not_producible`, `best_rate_solver_failure` | SMP-101, SMP-103 |
| `slot_limit` | a `sol-2` slot without an operating point after 50 attempts |
| `dead_date` | an `nsol-2` slot without an operating point after 50 attempts: a date at which the well flowed at none of its redrawn controls |
| `redraw_limit` | a 401st redraw would be needed |
| `solver_failure_limit` | an 11th solver failure |
| `short_life` | `nsol-2` only, SMP-101 |

**Targets**, judged on R11's pilot: at most 1% of a class's drawn wells dropped at the limits; at most 0.5% of the `nsol-2` wells as `dead_date`; solver failures in at most 0.5% of the attempts and of the wells of every class. A share above its target is a finding to fix before the full generation, in the draws, the limits, the life scan or the solver, not a filter to keep.

Nothing else drops a well or a sample: SMP-28's three filters (the choke's spread, the share of choked samples and the variation of `QTOT`), v1's floors on a sample's rate and its counts of failures do not apply to v2, and a small rate stays a sample.

From D-51 (ruled 2026-11-06), D-81, D-82 and D-83 (ruled 2026-11-06, checkpoint H3: kept, judged on R11's pilot), D-55, and D-106 (ruled 2026-11-09, L1: `table_failed` wells count with the drops). Sources: `sampling-draft.md`, Slots and outcomes, and How the rules meet acceptance criterion 4.

**Basis:** the limits (400 redraws let a kept well flow in a third of its draws) and the targets are a design choice, tolerances of the generation, kept and to be judged on R11's pilot (checkpoint H3); dropping v1's filters is a design choice against distributions shaped by filters (acceptance criterion 4, D-51).

### SMP-107 · Starts

The Rust core takes no start. Where a run solves with the CasADi backend, as a comparison, its extra start (`x_guess`) is the operating point of the same well nearest in $(u,\ w_{lg}/w_{lg,\max},\ p_s/p_{s,\text{nom}})$ among its best-rate solves and the slots solved so far (`sol-2`), or the one nearest in time to the slot's date (`nsol-2`). A start only adds roots to the set.

From D-56 (ruled 2026-11-06). Sources: SMP-28, SMP-29; `specs/model/solution.md`.

**Basis:** a design choice of the generation, for comparison runs only since the retry was dropped; that a start only adds roots is the root search's (`specs/model/solution.md`).

### SMP-108 · Generation logs

Each dataset has two logs, published as a companion file next to it, not in it:

- **the well log** (`drawn_wells`): every drawn well, accepted or not, with its draw index, every drawn input and label, its outcome, its best rate with its lift-gas rate and the outcomes of its solves, the counts of its attempts by outcome and reason, and in `nsol-2` $T_\text{life}$, the life scan's best rates and outcomes, $T_\text{end}$, and the lift's start and rule;
- **the redraw log** (`redraws`): every attempt that was redrawn and every attempt of a dropped well, with its inputs, its outcome and reason, and the sizes of its root set.

With the samples, the redraw log holds every draw of the accepted wells, so the shift that the redraws cause can be measured input by input and split into `no_operating_point` and `solver_failure`. The search counts and times of the attempts that gave a sample at once stay with the run. The columns are the schema's (`datasets-draft.md`, `redraws` and `drawn_wells`; R11).

From D-85 (ruled 2026-11-06, checkpoint I1) and D-52. Sources: `datasets-draft.md`, Section 2.

**Basis:** a design choice: the logs let a user measure and undo the redraws' conditioning (acceptance criterion 4), in a companion file by Bjarne's ruling (checkpoint I1, D-85).

### SMP-109 · Seeding

As in SMP-31, every draw comes from `rng_for(seed, …)` (`sampling.wells.rng_for`), seeded from the dataset's seed and the draw's place in it. Every v2 key holds `v2`, so v2's generators never coincide with v1's:

| Draws | Generator |
|---|---|
| block `b` of well $k$ (SMP-51) | `rng_for(seed, k, 'v2', b)` |
| the life's parameters and a tail well's start (SMP-83 to SMP-90) | `rng_for(seed, k, 'v2', 'evolution')` |
| a `sol-2` well's date (SMP-92) | `rng_for(seed, k, 'v2', 'date')` |
| attempt 0 of `sol-2` slot $j$ (SMP-93 to SMP-96) | `rng_for(seed, k, j, 'v2', 'sample')` |
| an `nsol-2` well's plan (SMP-97 to SMP-100) | `rng_for(seed, k, 'v2', 'controls')` |
| attempt $a \ge 1$ of slot $j$ (SMP-105) | `rng_for(seed, k, j, a, 'v2', 'redraw')` |
| the split (SMP-110) | `rng_for(seed, 'v2', 'split')` |

Each dataset has a seed of its own, so the wells of `sol-2` and `nsol-2` are independent draws. A run gives the same draws whatever the order of the wells or the number of processes.

From D-7, D-80 and D-91 (ruled 2026-11-06). Sources: SMP-31.

**Basis:** a design choice for reproducibility, SMP-31's scheme extended (D-7, D-80, D-91).

### SMP-110 · The datasets' wells and tables

- **Wells.** The generator draws wells $k$ = 0, 1, 2, … and keeps the first 10,000 it accepts. `ID` numbers them from 0 in draw order, and `draw_index` keeps $k$.
- **Split.** 20% of the wells are `test` and 80% `train`, by well, stratified by `fluid_type` × `well_type` × `geometry_class` × `coverage_tail`: each well, in `ID` order, draws one number from the split's generator; within each stratum the wells are ranked by it, and ranks 5, 10, 15, … are `test`.
- **Tables.** Each dataset holds `wells` (every input that is constant over a well's samples, and every label), `samples` (each sample's inputs that vary and its operating point: SMP-30's features with SLIP-16's five regimes, and the v2 columns), `geometry` (each well's survey stations) and `pvt` (the table fluids' tables), with SMP-108's logs beside it. A row of `samples`, its well's rows and, for a table fluid, its table's rows carry every input of its `WellProperties` and `BoundaryConditions`. The columns are `datasets-draft.md`'s, Section 2, which R11 writes into `docs/datasets.md`.
- **Records.** Each dataset records its seed, the configuration, $N$, the backend, the limits and the floor, the code version (`datasets.io.code_version`, a full commit with a clean working tree), the lock file's hash and the platform. The generator works in resumable chunks of 100 draw indices.
- **Re-solving.** A sample re-solved from its rows gives the same features with the recorded code version and backend on the generator's platform, and elsewhere features within the verifier's `tol_x` applied to them (`datasets-draft.md`, Section 3).

From D-86 to D-93 (ruled 2026-11-06; the logs beside the dataset, I1; the re-solve tolerance, I2). Sources: `specs/goals.md`, Compatibility; `datasets-draft.md`, Sections 1 to 3 and 6.

**Basis:** a design choice: the datasets' layout, split, records and re-solve tolerance are the dataset design's (D-86 to D-93), the tolerance being the verifier's `tol_x` (checkpoint I2).

## v2: mapping to the model's inputs

### SMP-111 · A well in `v2.0.0`

An attempt's well and boundary conditions are:

| Input | From |
|---|---|
| `geometry` | `WellGeometry.from_survey(md, tvd, n_cells=100, D)` (GEO-3, GEO-4) |
| `fluid.pvt` | black oil, wet and dry gas: `BlackOil(rho_o, rho_g, gas_law='dak', p_sep, T_sep)`; volatile oil and gas condensate: the well's `PVTTableSet` (SMP-116) |
| `fluid` | `FluidModel(pvt, rho_w, gor, wlr, cp_g, cp_o, cp_w)`, the surface tension from the oil (the default) |
| `friction` | `RoughnessFriction(roughness, correlation='chen')` |
| `thermal` | `ThermalModel(h)`, every `v2.0.0` term on (the defaults) |
| `slip` | `V2_SLIP`; `profile_scale` is not drawn (1) |
| `inflow` | `Vogel(w_l_max)` with $w_{l,\max} = k_l p_r/1.8$, so that its slope at $p_0 = p_r$ is $J$; `ProductivityIndex(k_l)` with $k_l = J\rho_{l,\text{sc}}/86\thinspace 400$ kg/(s bar) and $\rho_{l,\text{sc}} = (1 - \text{WLR})\thinspace\rho_o + \text{WLR}\thinspace\rho_w$; `Forchheimer(a, b)` for gas and condensate wells (SMP-117) |
| `choke` | `SimpsonChokeModel(K_c, chk_profile)` |
| `injection_md` | the valve's MD (SMP-79) while lift is installed, else `None` |
| `BoundaryConditions` | the attempt's $p_r$, $p_s$, $u$ and $w_{lg}$; the well's $T_r$ and $T_s$; $T_{lg}$ while lift is installed, else `None` |

Every well passes `configurations.check(wp, 'v2.0.0')`, or is `invalid_input` (SMP-106).

From D-36, D-60 (ruled 2026-11-05, J1) and the draws' entries. Sources: `sampling-draft.md`, Mapping to a well in `v2.0.0`; `specs/model/README.md`, the `v2.0.0` configuration.

**Basis:** physics: the mapping is the model's own inputs in the `v2.0.0` configuration, and Vogel's $w_{l,\max} = k_l p_r/1.8$ follows from INF-1's slope at $p_0 = p_r$; $N$ = 100 cells and an undrawn `profile_scale` are a design choice of the generation.

### SMP-112 · Table fluids and their source

Volatile oils and gas condensates (`pvt_model` `table`) take a PVT table set of their own; black oils and wet and dry gases take the black oil (`pvt_model` `black_oil`). NeqSim 3.4.0 makes the tables (Peng–Robinson with the Peneloux volume correction, its characterisation and its default interaction parameters, untuned) in a process environment of the generator's own, outside ManyWells' dependencies; the dataset carries the tables, so that a user re-solves without NeqSim. The `composition` block draws the synthetic composition's parameters first, then the source: with probability 0.2 one of the class's three published examples with equal shares (SMP-114; `fluid_source` its key), else the synthetic composition (SMP-113; `synthetic`).

From D-36, D-66 and D-68 (ruled 2026-11-06). Sources: `fluids-draft.md`, Which wells take a table; `specs/features/018-pvt-tables.md`.

**Basis:** a design choice of the Goals: volatile oils and condensates as tables from a flash, with synthetic compositions and published examples; the tool and its untuned equation of state are a design choice of the build (D-68), which published tests support: an untuned Peng–Robinson predicts 340 condensates' dew points within about 9% on average (Elsharkawy 2002, Table 3). The examples' share of 0.2, with equal shares among them, is a design choice too (D-66), for whose value the log gives no reason.

### SMP-113 · Synthetic composition

- **Components.** N₂, CO₂, C₁, C₂, C₃, i-C₄, n-C₄, i-C₅, n-C₅ and C₆ (as n-hexane), and the heptanes-plus as five fractions.
- **Light ends.** A mixture of the class's three published examples (SMP-114), normalised over N₂ to C₆, with weights $w \sim \text{Dirichlet}(1, 1, 1)$; then C₂ to C₆ are scaled by one factor $f_\text{int}$, and the light ends renormalised.
- **Heptanes-plus.** The mole fraction before the GOR is set, $z_{7+} \sim U(0.04, 0.10)$ for condensates and $U(0.10, 0.22)$ for volatile oils; the molar mass $M_{7+} \sim U(140, 200)$ and $U(190, 240)$ g/mol; split into five fractions at 135, 205, 275 and 415 g/mol by Whitson's gamma distribution with the shape $\alpha \sim U(0.5, 2.5)$; each fraction's specific gravity by Søreide, $\gamma_i = 0.2855 + C_f\thinspace(M_i - 66)^{0.13}$.
- **Separation.** One stage at standard conditions (1.01325 bar, 288.15 K), so that every GOR of the well's table set lies on one tie line. The reservoir fluid is the separator gas and the stock-tank oil recombined at the GOR.
- **Matching.** Newton's method on $(C_f, \ln f_\text{int})$ matches the stock-tank oil's API gravity and the separator gas's gravity to 0.5 °API and 0.005, and the recombination the GOR exactly. The targets are the drawn, field-reported values (SMP-67, SMP-68) shifted by the fluid's own offsets: the matched stream is flashed through the well's separator at $(p_\text{sep}, T_\text{sep})$, its liquid through a stock tank at standard conditions, and through one stage at standard conditions; the single-stage targets are moved by the difference and matched again, in two passes. The offsets are recorded with the fluid.

From D-63 (ruled 2026-11-06, checkpoint J2: per-fluid offsets). Sources: Whitson and Brulé (2000), Table 2.1, §5.3.2, Eqs. 5.13 to 5.15 and 5.44; Pedersen, Christensen and Shaikh (2015), Tables 1.3, 1.4 and 3.8; McCain (1990), pp. 151–156 [RF §8].

**Basis:** physics for the method: Whitson's gamma distribution and Søreide's gravities (Whitson and Brulé 2000, Eqs. 5.13 to 5.15 and 5.44), a flash, and each fluid's offsets by its own separator flash (checkpoint J2). Data, thinly, for the ranges: the light ends mix three published examples per class, the heptanes-plus ranges follow the span of about a dozen published fluids within McCain's class limits ([RF §8]), and $\alpha$ is the published 0.5 to 2.5 (Whitson and Brulé 2000, §5.3.2). The Dirichlet and uniform shapes are guesses: no statistics of compositions per class were found ([RF §8], Gaps), and a published set of measured compositions would replace them. One stage of separation with two matched descriptors is a design choice for the single tie line that the GOR axis needs (PVT-TAB-7), and the fractions' boundaries and the tolerances of 0.5 °API and 0.005 are a design choice of the build.

### SMP-114 · Published example fluids

A table well that takes an example uses its published composition, recombined along its single-stage tie line to the drawn GOR. Its stock-tank oil and separator gas are the example's, so its API and gas gravity, reported through the well's separator, replace the drawn ones, which are kept as records. A lumped heptanes-plus is split as a synthetic one's, with $\alpha = 1$ and $C_f$ matched to its gravity.

| Key | Class | Reference |
|---|---|---|
| `wb2000_gas_condensate` | gas condensate | Whitson and Brulé (2000), Table 2.1 |
| `pedersen2015_t1_3` | gas condensate | Pedersen et al. (2015), Table 1.3 |
| `pedersen2015_t3_8` | gas condensate | Pedersen et al. (2015), Table 3.8 |
| `wb2000_near_critical_oil` | volatile oil | Whitson and Brulé (2000), Table 2.1 |
| `wb2000_volatile_oil` | volatile oil | Whitson and Brulé (2000), Table 2.1 |
| `pedersen2015_t1_4` | volatile oil | Pedersen et al. (2015), Table 1.4 |

From D-66 (ruled 2026-11-06). Sources: [RF §8a]; the table's references.

**Basis:** data: six published compositions, each cited. Recombining an example to the drawn GOR is a design choice that puts it into the class it serves (D-66). For $\alpha = 1$ in a lumped heptanes-plus: basis not recorded.

### SMP-115 · Saturation pressure of a table fluid

The fluid's saturation pressure at $T_r$ is its flash's: TP flashes on steps of 10 bar up to $1.3\thinspace p_i$, with NeqSim's check for more phases on, then bisection to $10^{-3}$ bar on the edge of the two-phase region. Its kind, from the incipient phase, is `saturation_kind`, `bubble` or `dew`; a fluid with two phases nowhere from 1 bar to $1.3\thinspace p_i$ at $T_r$ takes $p_\text{sat}$ = 1 bar and `none`.

- Where $p_\text{sat}(T_r) \le p_i$, `saturation_rule` is `matched`.
- Where $p_\text{sat}(T_r) > p_i$, the fluid could not be one phase at $p_i$: its GOR moves along its tie line until $p_\text{sat}(T_r) = p_i$, by bisection on $\ln R$ within a factor of 4 (`gor_moved`); where no GOR on the line does it, the reservoir fluid is the class's phase in equilibrium at $(p_i, T_r)$, the vapour for a condensate and the liquid for an oil, separated again (`equilibrium_phase`).

The table's $p_\text{sat}$ and initial GOR replace SMP-68's in every later block; SMP-68's undersaturation does not apply. A fluid keeps its class (SMP-55) whatever its kind.

From D-65 (ruled 2026-11-06) and L2 (ruled 2026-11-09). Sources: McCain (1990), pp. 151–156; Pedersen et al. (2015), §1.4; Elsharkawy (2002), Table 3.

**Basis:** physics: a fluid's saturation pressure follows from its composition, so the table's own flash sets it (D-65, L2), and published tests of an untuned Peng–Robinson put its dew-point errors near the measurements' own (Elsharkawy 2002, Table 3). Moving the GOR, or taking the equilibrium phase, where the flash puts the fluid in two phases at $p_i$ is a design choice, taken over redrawing the composition, which would shape the compositions by a filter; the search's constants (steps of 10 bar to $1.3\thinspace p_i$, a factor of 4) are a design choice of the build.

### SMP-116 · The table set

- **Build order.** The table wells and their draws (ManyWells); each fluid, its saturation pressure and SMP-115's rule (NeqSim); each well's GOR axis from its life with the table's values in it (ManyWells); the tables at the axis's GORs (NeqSim).
- **Grid,** the well's own: temperatures every 10 K from $10\thinspace\lfloor(T_s - 15\ \text{K})/10\rfloor$ to $10\thinspace\lceil(T_r + 1\ \text{K})/10\rceil$; pressures 1, 2.5, 5 and 7.5 bar, then every 10 bar to $10\thinspace\lceil 1.05\thinspace p_i/10\rceil + 10$ bar, with each temperature's saturation pressure a node.
- **GOR axis** (PVT-TAB-7): GORs at most a factor 1.25 apart over every GOR the well's samples reach, their drift included: `sol-2` over $p_r$ within ±2% of its date's, `nsol-2` over the life's trend from $p_r(0)$ to $p_r(T_\text{life})$, each times the ends of $U(0.95, 1.05)$.
- **Columns.** $R_{so}$, $B_o$, $Z$, $\mu_o$, $\mu_g$ and $\sigma_{go}$ by PVT-TAB-6 from the flash, with its liquid trace $\delta = 10^{-4}$ where the stream is all vapour; where the stream is all liquid, the gas columns are the incipient vapour's.
- **Repair.** At each GOR, the pressures at which $p/Z$ does not rise strictly with $p$ at some temperature, the nodes off the longest run along which it rises (of the longest, the one ending highest), are dropped at every temperature of that GOR's table, so that the grid stays rectangular; the dropped pressures are recorded. A table that would lose more than 5% of its pressures fails.
- **Failure.** A build step that fails, a table that fails the repair, or a table that PVT-TAB-1's checks refuse makes the well `table_failed` (SMP-106); it is not solved.

From D-67 and D-68 (ruled 2026-11-06), D-64 (checkpoint K1) and D-106 (ruled 2026-11-09, L1). Sources: `specs/features/018-pvt-tables.md`; `specs/model/pvt/table.md`; `fluids-draft.md`, The table.

**Basis:** a design choice of the build, with a measurement behind each part: the grid's steps and the GOR axis's factor of 1.25 from feature 018's measured interpolation errors, the liquid trace from checkpoint K1, as other tools fill an absent phase (`sources-condensate-tables.md`), and the repair with its limit of 5% from the flash artefacts that R9's check run found (D-106, L1). The margin of 15 K below $T_s$ is a guess for Joule–Thomson cooling near a gas well's wellhead: the share of `off_table` roots on a pilot decides whether it widens (`fluids-draft.md`, Validity).

### SMP-117 · Forchheimer coefficients fitted to the pseudo-pressure

At each reservoir state $(p_r, R_{go})$, a gas or condensate well's `Forchheimer(a, b)` (INF-10) is fitted to its inflow in the pseudo-pressure,

$$(\mu_g Z)_ r\thinspace\big(m(p_r) - m(p_{wf})\big) = a\thinspace q + b\thinspace q^2, \qquad m(p) = 2\int_1^p \frac{p'}{\mu_g Z}\thinspace dp',$$

with $a$ and $b$ SMP-73's at the state, from $\mu_g Z$ of the reservoir gas at $T_r$ and the start's geometry factors and $D$. The reservoir gas is the black-oil gas at the wellstream's density, $(\rho_{g,\text{sc}} + \rho_o/R_{go})/c$, for a wet or dry gas, and the table's gas at $R_{go}$ for a condensate. $m$ is integrated by the trapezoidal rule on 200 pressures from 1 bar to $p_r$.

- **The fit.** Least squares of $(p_r^2 - p_{wf}^2)/q$ against $q$ at $p_{wf}/p_r$ = 0.95, 0.9, 0.8, 0.6, 0.4 and 0.2 and at the absolute open flow ($p_{wf}$ = 1.01325 bar) give the pressure-squared form's $a_\text{ws}$ and $b_\text{ws}$. Where the slope is negative, $b_\text{ws} = 0$ and $a_\text{ws}$ is the mean of $(p_r^2 - p_{wf}^2)/q$ over the points: the form cannot follow the pseudo-pressure's curvature there, a known limit at high pressure.
- **To the separator gas:** $a_q = c\thinspace a_\text{ws}$ and $b_q = c^2 b_\text{ws}$, with $c = 1 + C_{og}/R_{go}$, $C_{og} = 0.178108 \times 133\thinspace 000\thinspace\gamma_o/M_o$ Sm³/Sm³, $\gamma_o$ as in SMP-82 and $M_o = 6084/(\text{API} - 5.9)$ (Whitson and Brulé 2000, Eqs. 3.73b and 3.73c), for the black oil; and $c = 1 + \rho_o/(R_{go}\rho_{g,\text{sc}})$ for a table, whose $Z$ has the separator gas's gas constant (PVT-TAB-4). Then to the model's units by `Forchheimer.from_gas_rate(a_q, b_q, rho_g)`.

From D-61 (ruled 2026-11-06; $b = 0$ kept, L3 of 2026-11-09) and D-60. Sources: Golan and Whitson (1991), Eq. 2.57; Whitson and Brulé (2000), Eqs. 3.73b and 3.73c; `specs/features/022-gas-rate-inflow.md`, Units; `validity.md`, note 12.

**Basis:** physics: the pseudo-pressure's inflow with the model's own $\mu_g Z$, and the wellstream's conversion (Whitson and Brulé 2000, Eqs. 3.73b and 3.73c); from 20% drawdown to the open flow the fit's largest error is 0.7% at the median where $b > 0$ and 5.4% where $b = 0$ (R9's check run, `checkpoint-2.md`, L3). The fit's points and equal weights are a design choice of the build, and so is keeping $b = 0$ where the slope is negative, a known limit of the pressure-squared form at high pressure (L3).

## v2: guesses and design choices

*What the rules' Basis lines class as a guess or as a design choice, in one place (v2: scope and conventions, Basis). A guess is where the procedure is weakest: it stands in for data or physics that was not found. A design choice is deliberate, and serves a goal that data would not settle. A row covers the values that share its reasons; D-n and the sources are as in the rules.*

**Guesses.** Each with what was looked for and what would replace it, as far as the plans record them; *not recorded* means that the plans give none.

| Rule | Value | Looked for | Would replace it |
|---|---|---|---|
| SMP-56 | Pressure support with probability 0.7 for oils and 0.4 for gas classes | A count of wells by drive: the Gulf of Mexico's atlas records natural drive, not injection ([RF §3], Gaps) | A count of producing wells under pressure support |
| SMP-60 | The S wells' hold angle of 15° to 45° and end section of 100 to 400 m; the drain's tangent of 50 to 300 m at the heel; the holds' least 50 and 100 m of TVD | Not recorded: R3's values, from before the surveys were read | Not recorded |
| SMP-63 | The lithostatic gradient of 0.226 bar/m (1 psi/ft) | A published review of overpressure: on the resource list, not in hand (`checkpoint.md`, section 6) | A sourced lithostatic gradient offshore |
| SMP-63 | The caps on $\lambda$: 0.85, and 0.8 in the tail | Not recorded | Not recorded |
| SMP-64 | The onshore surface temperature of 268 to 303 K | A distribution of mean annual surface temperatures over producing wells ([WP §2], Gaps) | Climate normals of producing regions |
| SMP-66 | The medians (35 and 15 bar), spreads and bounds of $p_\text{sep}$ | A public distribution of separator or manifold pressures over wells ([WP §7], Gaps) | Such a distribution |
| SMP-66 | The separator temperatures: 300 to 340 K (platform), 290 to 320 K (subsea) and 285 to 320 K (onshore) | Not recorded | Not recorded |
| SMP-66 | The stage factors $r_s$: $U(0.5, 1)$ in mid-life and $U(0.1, 0.6)$ late | Separator pressures over wells' lives: two trade-press examples only ([WP §7]) | A record of separator pressures over wells' lives |
| SMP-68 | A table fluid's $x$, saturated with probability 0.40 or 0.30 and a median of 0.05; a record only | A count of volatile oils and condensates by saturation (`checkpoint-2.md`, L2) | Replaced already: the table's flash sets the saturation pressure (SMP-115) |
| SMP-69 | The ranges of $r_p$ by stage and drive, but for the gases' late range under depletion | Statistics of producing wells' current against initial pressure, and abandonment pressures of oil reservoirs ([RF §3], Gaps) | Such statistics |
| SMP-70 | The condensates' factor $c_k \sim U(0.5, 3)$ | Not recorded: Norway's condensate fields' late GOR is the one check in hand | Not recorded |
| SMP-71 | The net pay (median 15 m), the skin (mean 2), the high-angle wells' drainage of 8.5 to 9.5, the hole of 8.5 in, the design drawdown of 0.1 to 0.4, the anisotropy of 1 to 3, the perforated share of 0.3 to 1 and the lower bound of $D$ | A public distribution of wells' productivity, by well type ([WP §4], Gaps) | Such a distribution; until then the per-well rates check only the joint effect with the permeability |
| SMP-72 | The slant factor $(\cos\theta_\text{end})^{-1/2}$ on $J$ | Economides et al. (1994), §5-4: in hand, not yet read | Cinco-Ley's slant skin |
| SMP-73 | The same slant factor on $a$ and $b$ | As SMP-72 | As SMP-72 |
| SMP-74 | Vogel's thresholds: a saturation pressure of at least 0.8 of $p_r$, and a WLR below 0.6 | Not recorded | A composite inflow, straight above the bubble point and Vogel's below, which is not a model option |
| SMP-75 | The design point's constants (no water or a WGR of 10; a wellhead at $1.5\thinspace p_\text{sep} + 5$ bar and at 0.6 of the way from $T_s$ to $T_r$), and one size larger with probability 0.3 | A distribution of tubing sizes by well type or rate: none public ([WP §6], Gaps) | Not recorded |
| SMP-76 | The roughness's ranges by stage | Measurements of new and used tubing: Farshad and Rieke (2006), on [WP]'s resource list, not in hand | Those measurements |
| SMP-78 | The lift shares by stage: 0.20, 0.45 and 0.70 (oils offshore), 0.15, 0.25 and 0.35 (oils onshore), 0, 0.05 and 0.15 (gas classes) | A count of lifted wells: no register exists ([WP §1]); later found within one region's shares | Counts of lifted wells by stage from more regions |
| SMP-78 | A drawn lift left off where the balance point lies above 0.3 of $L$ | Not recorded | Not recorded |
| SMP-79 | The packer 30 to 150 m above the bottom | A distribution of injection depths over wells ([WP §9], Gaps): one textbook example | Such a distribution |
| SMP-79 | The lift's design drawdown of 0.3 to 0.7, and the least valve depth of 0.3 of $L$ | As the row above | As the row above |
| SMP-79 | $\phi \sim U(0, 0.3)$, the lift gas's cooling towards $T_s$ | Numbers on the annulus gas's heat exchange: none extracted (Hasan and Kabir 2018, §12.2.2) | Their coupled tubing and annulus balance |
| SMP-80 | The log-uniform shape of the injected GLR at capacity | Injected GLRs of lifted wells: 3W has no production rates | Injection and liquid rates of the same wells |
| SMP-81 | The ends of the ranges of $U_t$ (11 and 85 W/(m² K)) and $k_e$ (1.5 and 2.6 W/(m K)), and the two hole sizes | A distribution of the overall coefficient over completions ([WP §10], Gaps) | Willhite (1967) or Alves et al. (1992), on [WP]'s resource list, not in hand |
| SMP-82 | The factor $(0.554/\gamma_g)^{0.4}$ on methane's $c_p$ | An open correlation of a natural gas's $c_p$ against gas gravity at pressure ([RF §10], Gaps) | Such a correlation |
| SMP-82 | The representative temperature at 0.8 of the way from $T_s$ to $T_r$ | Not recorded | $c_p(p, T)$ along the well, which the model does not have |
| SMP-84 | The plateau of $U(0, 0.25)\thinspace T_\text{life}$ | Plateau lengths per well ([RF §11], Gaps) | Per-well plateau lengths |
| SMP-85 | The pressure's form for oils and supported wells, with $\gamma \sim U(1, 2)$ | Not recorded: one textbook well | A material balance of solution-gas drive for oils under depletion (D-42's alternative); none named for supported wells |
| SMP-86 | The breakthrough recoveries and jumps, and the gases' log-linear WGR | Per-well water-cut curves offshore, and a text on fractional flow ([RF §9], Gaps) | Per-well water-cut curves |
| SMP-88 | The roughness's log-linear path from its initial to its late draw | As SMP-76 | As SMP-76 |
| SMP-89 | Two step-downs of the separator, at dates of $U(0.25, 0.9)\thinspace T_\text{life}$ | Separator pressures over wells' lives: two trade-press examples only ([WP §7]) | A record of separator pressures over wells' lives |
| SMP-90 | Lift's drawn start at $U(0.2, 0.9)\thinspace T_\text{life}$ | Not recorded | Not recorded |
| SMP-94 | The lift gas off in 0.20 of a lifted well's slots | A share of time with the lift gas off: none public (D-38) | A record of lifted wells' injection over time |
| SMP-95 | The separator pressure within ±10% of its nominal | A spread of separator or manifold pressures: none public ([WP §7]) | Such a spread |
| SMP-98 | The change probability $p_c \sim \text{LogU}(0.5, 0.95)$, the step of ±0.1 of the range and the share of 0.1 of new draws | Histories of choke and lift-gas set points: 3W's windows last hours to weeks | Histories of set points over wells' lives |
| SMP-113 | The Dirichlet weights of the light ends, and the uniform shapes of the heptanes-plus ranges | Statistics of compositions per fluid class ([RF §8], Gaps) | A published set of measured condensate and volatile-oil compositions |
| SMP-116 | The table's margin of 15 K below $T_s$ | Not recorded | The share of `off_table` roots on a pilot (`fluids-draft.md`, Validity) |

**Design choices.** Each with the goal it serves. The tail's forms, which the rules' Basis lines mark with (SMP-53), are SMP-53's second row.

| Rule | Choice | Goal |
|---|---|---|
| SMP-50 | Two datasets of 10,000 wells with 200 operating points each; a realistic core weighted towards offshore wells and counted by wells, with a labelled tail | The Goals: four machine-learning tasks, realism first and coverage second |
| SMP-50 | The `v2.0.0` configuration, the Rust core and $N$ = 100 cells | The generation: the released model on its faster backend, with v1's grid |
| SMP-51 | One generator per block, every number drawn whatever the branch, and the blocks' order | Reproducibility: a change of one rule moves no other draw |
| SMP-52 | Every categorical choice stored as a label | The Goals: a user groups, reweights or drops wells |
| SMP-53 | 0.20 of the drawn wells are tail wells | Coverage: about 15% of the accepted wells, the middle of the Goals' 10% to 20% |
| SMP-53 | The tail's form, in every rule that gives one (SMP-54 to SMP-83): flat draws over the rule's range, equal or the rule's tail shares, and some ranges past the core's bounds, such as depths to 7000 m and a WGR to 10,000 | Coverage of the model's valid range |
| SMP-54 | An onshore share of 0.10 | The Goals' offshore weighting, with some land wells |
| SMP-55 | Volatile oils at 0.10 and condensates at 0.15 | Coverage: the Goals' floor of about 10% of the wells per fluid type |
| SMP-55 | McCain's class limits among the published ones, with tables for volatile oils and condensates | The model's validity: its black oil carries McCain's black oils |
| SMP-56 | Stage shares of 0.25, 0.40 and 0.35 | Wells at every stage of life, with depleted and watered-out ones (Bjarne's goal 1) |
| SMP-58 | The subsea wells' water depth log-uniform over the pooled range | Simplicity: D-104's uniform draw over the pooled range (D-110) |
| SMP-59 | An even mix of the four geometry classes | Coverage of inclinations, where real offshore wells are mostly deviated |
| SMP-59 | The classes' ranges of inclination, uniform angles within them, and the highly deviated class's two halves | Coverage, with a class label that names one range |
| SMP-60 | Each shape parameter uniform over its pooled range | Simplicity |
| SMP-60 | The fit rule: rates raised by steps of 20%, at most 20 redraws, the angles never lowered | The generation: a well keeps its class |
| SMP-61 | 75° within the band of 70° to 80°, and a drain modelled to its heel | The model's validity: no cell where stratified flow can occur |
| SMP-64 | The seabed's temperature at a platform's wellhead too | Consistency with THM-4's linear ambient profile |
| SMP-63 | $p_i$ held at 950 bar, underpressured, where $p_h$ alone passes it | The model's valid range, by the simplest rule that moves no other draw (D-109) |
| SMP-65 | A gradient of at least 10 K/km over the modelled well | A margin above the CasADi backend's limit |
| SMP-67 | API and gas gravity drawn independently | Simplicity |
| SMP-71 | The permeability's wide bounds | Coverage first: a wide support |
| SMP-72 | A drainage length that grows with the drain, and the liner taken as the tubing | Simplicity, among options that the evidence left open |
| SMP-74 | The inflow model kept over the life | No step in the rate without a change in the reservoir |
| SMP-75 | A well too strong for the largest tubing is flagged, not redrawn | Simplicity |
| SMP-77 | The lower end of $\beta$ at 0.15 | Coverage of the textbook beans |
| SMP-77 | A Simpson choke for every well, and equal shares of the four profiles | Simplicity and coverage, as in v1 |
| SMP-78 | Lift with no limit on the GOR | Bjarne's goal 5: gas lift in gassier wells too |
| SMP-80 | A capacity scaled to the well's design liquid rate | Bjarne's goal 5: a lift-gas rate scaled to the well |
| SMP-80 | The GLR truncated, in place of the capacity clipped | No point masses at the capacity's bounds |
| SMP-81 | The well's age by stage | The stages' meaning (SMP-91) |
| SMP-83 | One life for both datasets | Simplicity and consistency: a stage means the same state in both |
| SMP-84 | A clock that follows no solve | The Goals: open-loop controls and trends |
| SMP-88 | The inflow model, the completion and the fluid's descriptors fixed for the life | Simplicity |
| SMP-90 | Lift starts where natural flow ends, with the late-stage draw's design | Acceptance criterion 2: a well that an operator would lift is not dropped |
| SMP-91 | The stages' limits: the first year, and half the life span | Labels that mean the same in both datasets |
| SMP-92 | A `sol-2` well's date uniform within its drawn stage; a tail well's at the end of its life | One life for both datasets, with the drawn stage shares |
| SMP-93 | $u = 1$ with probability 0.05, else uniform from 0.05 | Bjarne's goal 4: close to uniform, with a modest weight at fully open |
| SMP-94 | A lift-gas rate uniform to the capacity | Coverage of the lift performance curve, past its optimum |
| SMP-96 | A drift of ±2% in $p_r$ and ±5% in the GOR and the water | The Goals: variation between a well's samples, as in v1 |
| SMP-97 | 160 operating samples at even dates and 8 tests of 5 samples | The Goals: samples that span the life, and regular tests for coverage |
| SMP-98 | Open-loop set points | The Goals |
| SMP-98 | The lift gas's set point kept at 0.1 of the capacity or more between changes | The generation: lifted wells keep flowing between changes |
| SMP-99 | Sweeps from the top through each quarter of the range, the choke's and the lift gas's in turn | Coverage: each well's response to its controls |
| SMP-100 | The same drift between dates, and none in $p_r$ | The Goals |
| SMP-101 | A life scan of 21 points, ending the life at the last that passes | The generation: every date has an operating point |
| SMP-102 | A margin of half the floor | The Goals: marginally economic wells stay in |
| SMP-103 | The best rate from a fully open choke and three lift-gas rates; lift before a well is dropped | The generation, and acceptance criterion 2 |
| SMP-104 | A scan with non-finite points, no root and no rejected sign change counts as no operating point; no retry on the other backend | Acceptance criterion 4, and the constitution's principle 7 |
| SMP-105 | 200 operating points per well, with redraws from each slot's own generator in passes | The Goals (D-52) and reproducibility |
| SMP-106 | The limits of 50 attempts, 400 redraws and 10 solver failures, and the targets | Tolerances of the generation, judged on R11's pilot |
| SMP-106 | None of v1's filters | Acceptance criterion 4: no distributions shaped by filters |
| SMP-107 | The extra start of a comparison run | The generation |
| SMP-108 | Two published logs, in a companion file | Acceptance criterion 4: a user can measure the redraws' conditioning |
| SMP-109 | The seeding | Reproducibility |
| SMP-110 | The datasets' layout, split, records and re-solve tolerance | Dataset reproducibility and use |
| SMP-111 | $N$ = 100 cells, and `profile_scale` not drawn | The generation |
| SMP-112 | Volatile oils and condensates as tables from a flash, with synthetic compositions and published examples | The Goals |
| SMP-112 | NeqSim, with an untuned equation of state, outside ManyWells' dependencies | The build: no equation of state in ManyWells |
| SMP-112 | A published example in 0.2 of the table wells, with equal shares | The Goals: published examples beside the synthetic fluids |
| SMP-113 | One stage of separation, with the API and gas gravity matched | The single tie line that the table set's GOR axis needs |
| SMP-113 | The five fractions' boundaries, and the tolerances of 0.5 °API and 0.005 | The build |
| SMP-114 | An example recombined to the drawn GOR | The example serves the class it is drawn for |
| SMP-115 | The GOR moved, or the equilibrium phase taken, where the fluid is in two phases at $p_i$ | Compositions not shaped by a filter |
| SMP-115 | The search's constants: steps of 10 bar to $1.3\thinspace p_i$, and a factor of 4 | The build |
| SMP-116 | The table's grid, GOR axis, liquid trace and repair | The build: tables that the model accepts, at a measured interpolation error |
| SMP-117 | The fit's points and equal weights, and $b = 0$ where the slope is negative | The build: the model's inflow form, with its known limit |

**Basis not recorded.** The parts for which the plans give no basis: the dry gases' yield limit of 2.8 Sm³ per 10⁶ Sm³ (SMP-55); the gradient's bounds of 12 and 55 K/km (SMP-65); the floor of 1.5 bar and the cap of 120 bar on the nominal pressure downstream of the choke (SMP-66); the volatile oils' log-uniform GOR, the dry gases' yield from 0.5 and the bounds of $x$ (SMP-68); the drain's liner constants (SMP-72, SMP-73); and $\alpha = 1$ for a lumped heptanes-plus (SMP-114).

## v2: checks

What the tests check on drawn wells, core and tail apart, without solving them (`sampling-draft.md`, Checks a test can run; `fluids-draft.md`, Checks a test can run):

- **Bounds**, broken by no draw: every cell within 75° of vertical; MD at most 9342 m; $p_i \le 950$ bar and $\lambda \le 0.85$; $T_r \le 450$ K; $(T_r - T_s)/L \ge 10$ K/km; $p_\text{sat} \le p_i$ (a table's to $10^{-3}$ bar); the black oils' $R_{si}$ at most 356 Sm³/Sm³; API from 16 to 58; gas gravity at least 0.571; an oil's WLR at most 0.98; $w_{lg,\max}$ in $[0.1, 3.5]$ kg/s; every well passes `configurations.check(wp, 'v2.0.0')`.
- **Shares and marginals:** each label's share, and each draw's distribution, within its sampling error of the rule's.
- **Reproducibility:** a rerun gives the same draws, whatever the order of the wells or the number of processes (SMP-109); a change of one block's distribution leaves the other blocks' draws as they were; the `v1.0.0` configuration's draws (SMP-1 to SMP-31) do not change.
- **Tables:** every table passes PVT-TAB-1 and builds a `FluidModel`; a `matched` table's GOR is its target to rounding, and its single-stage API and gas gravity are within 0.5 °API and 0.005 of their shifted targets; the heptanes-plus content is within McCain's limits for the class; the grid covers $p_r$ and $T_r$, and the GOR axis every sample's GOR; a rerun gives the same tables on the same NeqSim version.
- **Solved rates**, on R11's pilot: the core offshore wells' rates against the Gulf of Mexico's brackets (EIA 2025, Table B32), a check and not a draw, measured by well type; the reservoirs' temperatures and overpressures against Nadeau, Sun and Ehrenberg's (2023) counts of reservoirs (D-2, D-24, L5).
- **Rank correlations** of core black oils (depth with $p_i$ and $T_r$; $R_{si}$ with $p_i$, $T_r$, depth and API) are reported against the Gulf of Mexico's by wells [own analysis of BOEM's sands atlas]: a check of realism, not a bound.

## v2: references

- Ahmed, T. (2006), *Reservoir engineering handbook*, 3rd ed., Gulf Professional Publishing.
- Asheim, H. (1988), "Criteria for gas-lift stability", *Journal of Petroleum Technology* 40(11), 1452–1456.
- Brekke, K. and Lien, S.C. (1994), "New, simple completion methods for horizontal wells improve production performance in high-permeability thin oil zones", *SPE Drilling & Completion* 9(3), 205–209.
- Brill, J.P. and Mukherjee, H. (1999), *Multiphase flow in wells*, SPE Monograph 17.
- Craft, B.C. and Hawkins, M.F. (1991), *Applied petroleum reservoir engineering*, 2nd ed., revised by R.E. Terry, Prentice Hall.
- Cragoe, C.S. (1929), *Thermal properties of petroleum products*, US Bureau of Standards Miscellaneous Publication 97.
- Dikken, B.J. (1990), "Pressure drop in horizontal wells and its effect on production performance", *Journal of Petroleum Technology* 42(11), 1426–1433.
- Economides, M.J., Hill, A.D. and Ehlig-Economides, C. (1994), *Petroleum production systems*, Prentice Hall.
- EIA (2025), *The distribution of U.S. oil and natural gas wells by production rate*, US Energy Information Administration.
- Ehrenberg, S.N. and Nadeau, P.H. (2005), *AAPG Bulletin* 89(4), 435–445.
- Elsharkawy, A.M. (2002), *Fluid Phase Equilibria* 193, 147–165.
- Goh, K.-C., Moncur, C.E., Van Overschee, P. and Briers, J. (2007), "Production surveillance and optimization with data driven models", IPTC 11647.
- Golan, M. and Whitson, C.H. (1991), *Well performance*, 2nd ed., Prentice Hall.
- Guo, B., Lyons, W.C. and Ghalambor, A. (2007), *Petroleum production engineering*, Gulf Professional Publishing.
- Hasan, A.R. and Kabir, C.S. (2012), *Journal of Petroleum Science and Engineering* 86–87, 127–136; Hasan, A.R. and Kabir, C.S. (2018), *Fluid flow and heat transfer in wellbores*, 2nd ed., SPE.
- Joshi, S.D. (1991), *Horizontal well technology*, PennWell.
- Liu, M., Wang, Z. and Kaiser, M.J. (2014), *Oil & Gas Journal*, 3 March 2014.
- McCain, W.D. (1990), *The properties of petroleum fluids*, 2nd ed., PennWell.
- Miller, R., Sorrell, S. and Speirs, J. (2009), *UKERC review of evidence for global oil depletion, Technical report 4: Decline rates and depletion rates*, UK Energy Research Centre.
- Nadeau, P.H., Sun, S. and Ehrenberg, S.N. (2023), *Marine and Petroleum Geology* 158, 106507.
- Pedersen, K.S., Christensen, P.L. and Shaikh, J.A. (2015), *Phase behavior of petroleum reservoir fluids*, 2nd ed., CRC Press.
- Ramey, H.J. (1962), *Journal of Petroleum Technology* 14(4), 427–435.
- Sharqawy, M.H., Lienhard, J.H. and Zubair, S.M. (2010), *Desalination and Water Treatment* 16, 354–380.
- Valkó, P.P. and McCain, W.D. (2003), *Journal of Petroleum Science and Engineering* 37, 153–169.
- Vogel, J.V. (1968), *Journal of Petroleum Technology* 20(1), 83–92.
- von Flatern, R. (2015), "The defining series: gas lift", *Oilfield Review*.
- Whitson, C.H. and Brulé, M.R. (2000), *Phase behavior*, SPE Monograph 20.
- NeqSim 3.4.0, an open-source Java library.
- The public databases behind [RF], [WP] and [GEO], and behind the checkpoints' evidence: BOEM's sands atlas, borehole file, directional surveys and OGOR-A production; BSEE's well tests; Sodir's FactMaps and FactPages; NSTA's wellbore layer and production by well; ANP's production by well; NLOG's deviation surveys; Petrobras' 3W Dataset; NOAA's World Ocean Atlas 2023; and USGS's produced-water database. The maintainer's analyses of them are kept outside the repo and cited as [own analysis of …].

## Coverage

A v2 row (SMP-50 to SMP-117) is spec-only until R10's code carries its `# spec:` tag: its code column then names where the code is planned, and its planned check is in brackets. It becomes `property:` when its test exists.

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| SMP-1 | (20) | `well.py` `sample_well` | verifier: Distributions |
| SMP-2 | (21) | `well.py` `sample_well` | verifier: Distributions |
| SMP-3 | (22) | `well.py` `sample_well` | verifier: Distributions |
| SMP-4 | (23) | `well.py` `sample_well` | verifier: Distributions |
| SMP-5 | (24) | `well.py` `sample_well` | verifier: Distributions |
| SMP-6 | §4.1 | `well.py` `sample_well` | verifier: Distributions |
| SMP-7 | (25) | `well.py` `sample_well` | verifier: Distributions |
| SMP-8 | (26) | `well.py` `sample_well` | verifier: Distributions |
| SMP-9 | (27) | `well.py` `sample_well` | verifier: Distributions |
| SMP-10 | §4.2 | `well.py` `sample_well` (`pvt.WATER`) | verifier: Distributions |
| SMP-11 | (30) | `well.py` `sample_well` | verifier: Distributions |
| SMP-12 | (28)–(31) | `well.py` (`pvt.liquid_mix`) | verifier: Distributions |
| SMP-13 | (33) | `well.py` `sample_well` | verifier: Distributions |
| SMP-14 | (36) | `well.py` `sample_well` | verifier: Distributions |
| SMP-15 | §4.3 | `simulator.py` `BoundaryConditions.T_s` | verifier: Distributions |
| SMP-16 | (34) | `well.py` `sample_well` | verifier: Distributions |
| SMP-17 | §4.5 | `well.py` `sample_well` | verifier: Distributions |
| SMP-18 | (44) | `well.py` `Well.sample_new_conditions` | verifier: Distributions |
| SMP-19 | (45) | `well.py` `Well.sample_new_conditions` | verifier: Distributions |
| SMP-20 | (35) | `well.py` `Well.sample_new_conditions` | verifier: Distributions |
| SMP-21 | (32) | `well.py` `Well.sample_new_conditions` | verifier: Distributions |
| SMP-22 | (53)–(55) | `well.py` `Well.sample_new_conditions` | verifier: Distributions |
| SMP-23 | §4.4.1 | `nonstationary_well.py` `NonStationaryBehavior` | verifier: Distributions |
| SMP-24 | (37)–(39) | `nonstationary_well.py` `NonStationaryBehavior.mass_fractions` | verifier: Distributions |
| SMP-25 | (40)–(43) | `nonstationary_well.py` `NonStationaryBehavior.reservoir_pressure` | verifier: Distributions |
| SMP-26 | (35) | `nonstationary_well.py` `NonStationaryWell.update_conditions` | verifier: Distributions |
| SMP-27 | §4.5.1 | `nonstationary_well.py` `NonStationaryWell.update_conditions` | verifier: Distributions |
| SMP-28 | §5.1 | `open_loop_stationary/generate_well_data.py` | verifier: Distributions |
| SMP-29 | §5.2 | `open_loop_nonstationary/generate_open_loop_nonstationary_well_data.py`, `init_utils.py` | verifier: Distributions |
| SMP-30 | Table 3 | `open_loop_stationary/generate_well_data.py` `simulate_well` | verifier: Distributions |
| SMP-31 | — | none (S-9) | property: tests/test_sampling.py (a rerun gives the same draws) |
| SMP-40 | — | — | property: tests/test_sampling.py (the mapped well reproduces v1's inputs) |
| SMP-41 | — | — | property: tests/test_sampling.py (the drawn distributions) |
| SMP-42 | — | — | property: tests/test_sampling.py (the drawn distributions) |
| SMP-43 | — | — | property: tests/test_sampling.py (the mapped fluid) |
| SMP-44 | — | — | spec-only: a rule that nothing is drawn |
| SMP-50 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the datasets' counts and configuration) |
| SMP-51 | — | — (`sampling/v2/draws.py` `draw_well`; `sampling/v2/survey.py` `survey`; `sampling/v2/_distributions.py` `unused`) | property: tests/test_sampling_v2_draws.py (the blocks' generators in their order, a block drawn from another seed moves no other block's numbers, a stage override changes only what depends on the stage, every block takes the same numbers for every well, a branch's constant moves no later draw); property: tests/test_sampling_v2_tables.py (a composition's six numbers) |
| SMP-52 | — | — (`sampling/v2/draws.py`, the labels that the well draws set; `sampling/v2/life.py`, a date's `life_stage`, and `lift` and `lift_start_rule` by the draws; the tables' and the generation's labels: planned, R10) | property: tests/test_sampling_v2_draws.py (every label of the well draws within its values); property: tests/test_sampling_v2_life.py (the stage of a date, a `sol-2` well's stage and lift labels) |
| SMP-53 | — | — (`sampling/v2/draws.py` `_labels`, and each block's tail form) | property: tests/test_sampling_v2_draws.py (the tail's share, the tail wells' bounds, shares and marginals) |
| SMP-54 | — | — (`sampling/v2/draws.py` `_labels`) | property: tests/test_sampling_v2_draws.py (the shares, core and tail) |
| SMP-55 | — | — (`sampling/v2/draws.py` `_labels`) | property: tests/test_sampling_v2_draws.py (the shares, core and tail, each class's range of the initial GOR) |
| SMP-56 | — | — (`sampling/v2/draws.py` `_labels`) | property: tests/test_sampling_v2_draws.py (the shares, core and tail) |
| SMP-57 | — | — (`sampling/v2/draws.py` `_depth`) | property: tests/test_sampling_v2_draws.py (the bounds and the marginals, core and tail) |
| SMP-58 | — | — (`sampling/v2/draws.py` `_depth`) | property: tests/test_sampling_v2_draws.py (the bounds, the share of deep-water platforms, the subsea and shelf marginals) |
| SMP-59 | — | — (`sampling/v2/draws.py` `_geometry`) | property: tests/test_sampling_v2_draws.py (the shares of the classes and of their families, the class's inclination) |
| SMP-60 | — | — (`sampling/v2/survey.py` `survey`; `sampling/v2/draws.py` `_geometry`) | property: tests/test_sampling_v2_draws.py (each family's sections, ranges, kickoff and least hold, MD at most 9342 m) |
| SMP-61 | — | — (`sampling/v2/draws.py` `_geometry`) | property: tests/test_sampling_v2_draws.py (every cell within 75° of vertical) |
| SMP-62 | — | — (`sampling/v2/draws.py` `_water`) | property: tests/test_sampling_v2_draws.py (the bounds, the density and the marginals) |
| SMP-63 | — | — (`sampling/v2/draws.py` `_pressure`) | property: tests/test_sampling_v2_draws.py (the bounds, the truncations and the shares of underpressure, $p_i$ at 950 bar where the hydrostatic pressure alone exceeds it) |
| SMP-64 | — | — (`sampling/v2/draws.py` `_temperature`) | property: tests/test_sampling_v2_draws.py (the bounds) |
| SMP-65 | — | — (`sampling/v2/draws.py` `_temperature`) | property: tests/test_sampling_v2_draws.py (the bounds and the truncations: $T_r$ at most 450 K, at least 10 K/km over the well) |
| SMP-66 | — | — (`sampling/v2/draws.py` `_separator`) | property: tests/test_sampling_v2_draws.py (the bounds, the nominal pressure's formula and a marginal) |
| SMP-67 | — | — (`sampling/v2/draws.py` `_fluid`) | property: tests/test_sampling_v2_draws.py (the bounds and the marginals) |
| SMP-68 | — | — (`sampling/v2/draws.py` `_fluid`, and `_stage` for `saturation`) | property: tests/test_sampling_v2_draws.py ($p_\text{sat}$ at most $p_i$, a black oil's $R_{si}$ on its correlation and at most 356 Sm³/Sm³, the label) |
| SMP-69 | — | — (`sampling/v2/draws.py` `_stage`) | property: tests/test_sampling_v2_draws.py (the bounds by stage and drive, the tail's marginal) |
| SMP-70 | — | — (`sampling/v2/reservoir.py` `producing_gor`; `sampling/v2/draws.py` `_stage`) | property: tests/test_sampling_v2_mapping.py (the curve below the saturation pressure, its dip and rise, the condensates' factor) |
| SMP-71 | — | — (`sampling/v2/draws.py` `_inflow`; `sampling/v2/reservoir.py` `nondarcy`) | property: tests/test_sampling_v2_draws.py (the bounds and the marginals); property: tests/test_sampling_v2_mapping.py (Eq. 4-34 in field units) |
| SMP-72 | — | — (`sampling/v2/reservoir.py` `j_radial`, `joshi_ratio`, `liner_liquid`, `drain_index`; `sampling/v2/draws.py` `_inflow`, `_inflow_at_stage`; `sampling/v2/inflow.py` `inflow_at`) | property: tests/test_sampling_v2_mapping.py (the radial index in field units, Joshi's index against Economides et al.'s Example 2-10, the closed form's efficiency, the index at a state); property: tests/test_sampling_v2_draws.py (the geometry's factor of every well) |
| SMP-73 | — | — (`sampling/v2/reservoir.py` `forchheimer`, `liner_gas`; `sampling/v2/draws.py` `_on_geometry`, `_inflow_at_stage`) | property: tests/test_sampling_v2_mapping.py (the coefficients in field units, the liner's friction); property: tests/test_sampling_v2_draws.py (the geometry's factors of every gas well) |
| SMP-74 | — | — (`sampling/v2/draws.py` `_inflow_at_stage`) | property: tests/test_sampling_v2_draws.py (the rule on every oil well, gas classes take `forchheimer`) |
| SMP-75 | — | — (`sampling/v2/draws.py` `_completion`) | property: tests/test_sampling_v2_draws.py (the sizes, 9-5/8 in for gas classes only, the marginal of $C$) |
| SMP-76 | — | — (`sampling/v2/draws.py` `_completion`) | property: tests/test_sampling_v2_draws.py (the ranges by stage, and the tail's) |
| SMP-77 | — | — (`sampling/v2/draws.py` `_choke`) | property: tests/test_sampling_v2_draws.py (the bounds, $K_c$ and the profiles' shares) |
| SMP-78 | — | — (`sampling/v2/draws.py` `_lift`, `_needs_lift`) | property: tests/test_sampling_v2_draws.py (a well that needs lift is lifted, no core gas well is lifted at its start) |
| SMP-79 | — | — (`sampling/v2/draws.py` `_lift`) | property: tests/test_sampling_v2_draws.py (the valve's depth within its bounds, its MD on the survey, the lift gas's temperature) |
| SMP-80 | — | — (`sampling/v2/draws.py` `_lift`) | property: tests/test_sampling_v2_draws.py ($w_{lg,\max}$ within 0.1 to 3.5 kg/s) |
| SMP-81 | — | — (`sampling/v2/draws.py` `_heat`, `_heat_transfer`) | property: tests/test_sampling_v2_draws.py (the bounds, the tail's marginal) |
| SMP-82 | — | — (`sampling/v2/draws.py` `_heat`) | property: tests/test_sampling_v2_draws.py ($c_{po}$ and $c_{pw}$ by their formulas, $c_{pg}$ within the table's range) |
| SMP-83 | — | — (`sampling/v2/life.py` `Life`) | property: tests/test_sampling_v2_life.py (the start and the end by drive and class, core and tail, a `sol-2` well is its life at its date) |
| SMP-84 | — | — (`sampling/v2/life.py` `Life`, `Life.F`, `_cumulative`) | property: tests/test_sampling_v2_life.py (the parameters' ranges, the clock from 0 to 1 with a plateau and a rate that never rises, Arps's cumulative production against its rate) |
| SMP-85 | — | — (`sampling/v2/life.py` `Life.p_r`) | property: tests/test_sampling_v2_life.py (the pressure falls from the start's to the end's, the form with $\gamma$, a gas well's $p/Z$ linear in $F$) |
| SMP-86 | — | — (`sampling/v2/life.py` `Life.water`) | property: tests/test_sampling_v2_life.py (the water never falls, the start's until breakthrough, the jump, the logit and the logarithm linear in $F$, the end's) |
| SMP-87 | — | — (`sampling/v2/life.py` `Life.state`) | property: tests/test_sampling_v2_life.py (the start's GOR above the saturation pressure, SMP-70's at the trend's pressure, a condensate's never falls) |
| SMP-88 | — | — (`sampling/v2/life.py` `Life.h`, `Life.roughness`, `Life.state`) | property: tests/test_sampling_v2_life.py (the roughness log-linear in time, $h$ falls with age and a tail well keeps its own, the coefficients at the trends' state whatever the drift, what a `sol-2` well keeps of its initial draw) |
| SMP-89 | — | — (`sampling/v2/life.py` `Life.r_s`, `Life.p_s_nom`) | property: tests/test_sampling_v2_life.py (the factor's two steps, the nominal pressure's formula, core and tail) |
| SMP-90 | — | — (`sampling/v2/life.py` `Life`, `lift_on`, `sol2_well`; `sampling/v2/samples.py` `nsol2_plan`, with the scan's start; the life scan's rule 3: planned, R10) | property: tests/test_sampling_v2_life.py (the rule of every start, the static check's first failure on its 41 dates, the late draw's design, the well that the filter lifts); property: tests/test_sampling_v2_samples.py (no lift gas before the start, a plan with the scan's earlier start) |
| SMP-91 | — | — (`sampling/v2/life.py` `Life.stage`) | property: tests/test_sampling_v2_life.py (the stage at the end of the first year and at half the life span) |
| SMP-92 | — | — (`sampling/v2/life.py` `sol2_well`) | property: tests/test_sampling_v2_life.py (the date within the drawn stage and uniform in it, a tail well's at the end of its life, the state, the labels and the lift at the date, the well passes `configurations.check`) |
| SMP-93 | — | — (`sampling/v2/samples.py` `sol2_attempt`) | property: tests/test_sampling_v2_samples.py (the bounds, the share fully open and the marginal) |
| SMP-94 | — | — (`sampling/v2/samples.py` `sol2_attempt`) | property: tests/test_sampling_v2_samples.py (a natural well's lift gas is off, the share off of a lifted well's and the marginal) |
| SMP-95 | — | — (`sampling/v2/samples.py` `_separator_pressure`) | property: tests/test_sampling_v2_samples.py (the bounds and the marginal, in `sol-2` slots, plans and redraws) |
| SMP-96 | — | — (`sampling/v2/samples.py` `sol2_attempt`) | property: tests/test_sampling_v2_samples.py (the bounds and the marginals of the pressure, of the GOR at the drifted pressure and of the water, the well's own inputs do not drift) |
| SMP-97 | — | — (`sampling/v2/samples.py` `nsol2_plan`, `nsol2_attempt`) | property: tests/test_sampling_v2_samples.py (200 slots in the order of their dates, 160 operating samples and 8 tests of 5, the dates to the end of the life, an attempt holds the state of its date) |
| SMP-98 | — | — (`sampling/v2/samples.py` `nsol2_plan`) | property: tests/test_sampling_v2_samples.py (the set points at the start, the changes' share by the well's probability, the steps' share and reflection, the lift gas's floor and its shut-in, the history whatever the end of the life) |
| SMP-99 | — | — (`sampling/v2/samples.py` `nsol2_plan`) | property: tests/test_sampling_v2_samples.py (the sweep by test and lift, the quarters from the top down, the other control at its set point) |
| SMP-100 | — | — (`sampling/v2/samples.py` `nsol2_plan`; `sampling/v2/life.py` `Life.state`) | property: tests/test_sampling_v2_samples.py (the drift's bounds and marginal, one drift per test, the pressure without noise); property: tests/test_sampling_v2_life.py (the drift on the GOR and the water only, an oil's WLR capped) |
| SMP-101 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: a pilot well's life scan and end) |
| SMP-102 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the best rate against the floor, on a pilot) |
| SMP-103 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the best rate against the floor, on a pilot) |
| SMP-104 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: each outcome and reason, on solved cases) |
| SMP-105 | — | — (`sampling/v2/samples.py` `sol2_attempt`, `nsol2_attempt`, an attempt's and a redraw's draws; the passes over the slots: planned, R10) | property: tests/test_sampling_v2_samples.py (a slot depends on no other, a `sol-2` redraw draws the slot again, an `nsol-2` redraw keeps its date's state and changes the controls of its own slot by SMP-105's numbers. Planned: a pilot well's passes) |
| SMP-106 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the drops against their targets, on R11's pilot) |
| SMP-107 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: a comparison run's starts) |
| SMP-108 | — | — (planned: `manywells.datasets`, R10) | spec-only: until R10 implements it (planned property: the logs hold every draw of a pilot) |
| SMP-109 | — | — (`sampling/v2/draws.py` `draw_well`, the blocks' generators; `sampling/v2/life.py`, the life's and the date's; `sampling/v2/samples.py`, the samples', the plan's and the redraws'; the split's: planned, R10) | property: tests/test_sampling_v2_draws.py (a rerun gives the same draws, whatever was drawn before and in a process of its own, every key holds `v2`); property: tests/test_sampling_v2_life.py (the life's and the date's generators, a rerun gives the same life); property: tests/test_sampling_v2_samples.py (each generator's key, a rerun gives the same plan and attempts) |
| SMP-110 | — | — (planned: `manywells.datasets`, R10) | spec-only: until R10 implements it (planned property: the counts, the split, and a re-solved row) |
| SMP-111 | — | — (`sampling/v2/mapping.py` `WellBuilder`) | property: tests/test_sampling_v2_mapping.py (every drawn well at its nominal state passes `configurations.check`, with each input from its draws) |
| SMP-112 | — | — (`sampling/v2/table_build.py` `fluid_source`, `NEQSIM_VERSION` and the flash; the choice of the table wells: planned, R10) | property: tests/test_sampling_v2_tables.py (the source's shares and keys, the build refuses another NeqSim, the module imports nothing of ManyWells) |
| SMP-113 | — | — (`sampling/v2/table_build.py`: the composition's draws, the gamma split, Søreide's gravities, the match) | property: tests/test_sampling_v2_tables.py (the gamma split and Søreide's gravities against their closed forms, the composition's draws and levers, with NeqSim, a matched fluid's GOR and gravities) |
| SMP-114 | — | — (`sampling/v2/table_build.py` `EXAMPLES`, the example's stream and its match) | property: tests/test_sampling_v2_tables.py (the examples' compositions sum to one and name their class) |
| SMP-115 | — | — (`sampling/v2/table_build.py` `build_fluid`, the saturation pressure and its rule; `sampling/v2/draws.py`, the `fluid` hook) | property: tests/test_sampling_v2_tables.py (a well's steps and files with the flash replaced, each rule and each failure), tests/test_sampling_v2_draws.py (the hook replaces the fluid block's values and moves no draw) |
| SMP-116 | — | — (`sampling/v2/table_build.py` `well_grid`, `build_tables`, `build`; `sampling/v2/tables.py` `read_table`; the GOR axis: planned, R10) | property: tests/test_sampling_v2_tables.py (the grid, the repair on a spike, a dip and ties, the 5% limit, the files' round trip, each failure gives `table_failed`'s status) |
| SMP-117 | — | — (`sampling/v2/inflow.py` `gas_inflow`, `fit_pressure_squared`) | property: tests/test_sampling_v2_mapping.py (the fit reproduces a pressure-squared law, $b = 0$ where the slope is negative, the fit against the pseudo-pressure) |
