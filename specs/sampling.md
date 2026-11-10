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

### SMP-50 · Datasets and configuration

- The v2 procedure draws two datasets, `manywells-sol-2` (stationary) and `manywells-nsol-2` (non-stationary), each of 10,000 wells with 200 samples per well. Every sample is an operating point.
- Every well is built in the `v2.0.0` configuration with $N$ = 100 cells (SMP-111) and solved by the Rust core, `SSDFSimulator(wp, backend='rust')`.
- The population is a realistic core, weighted towards offshore production wells on platforms and subsea that flow naturally or on gas lift and counted by wells, and a labelled coverage tail over the model's valid range (SMP-53). The draws first cover almost all real wells and hold no unrealistic ones; second, they weigh typical wells more than rare ones.
- Both datasets draw their wells by the same rules, from seeds of their own (SMP-109), and every well has a life (SMP-83). An `nsol-2` well's 200 samples span its life; a `sol-2` well's samples vary around one date of it (SMP-92).

From the Goals of `plans/sampler-redesign-plan.md` (Bjarne, 2026-10-30 and 2026-11-05), D-1 and D-52 (ruled 2026-11-02). Sources: `specs/goals.md`, Compatibility.

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

- Each block draws its numbers in a fixed order, and draws every one of them whatever branch the well takes, so that a change of one rule moves no other draw.
- **The same well at another stage.** A well's draws at stage $s$ are its draws with the life stage set to $s$ after the `labels` block has drawn it, from the same generators, so that only the draws that depend on the stage change. The life (SMP-83) uses the well at its three stages.
- A table fluid's table (SMP-112 to SMP-116) is made after the `fluid` block, and its saturation pressure and initial GOR replace the `fluid` block's before the `stage` block; its tables are made at the GORs its life reaches.

From D-7 (ruled 2026-11-06, with the checkpoint's note that every number of a block is drawn whatever its branch) and D-105. Sources: `sampling-draft.md`, The order of the draws.

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

### SMP-53 · Coverage tail

`coverage_tail` is `tail` with probability 0.20, else `core`. In a tail well every draw takes its tail form: the categorical shares of the well type, the fluid type and the life stage become equal, the other categorical draws take the tail shares their rules give, and the continuous draws become uniform or log-uniform over the ranges their rules give. The links to earlier draws stay: pressure still follows depth and brine, the bubble point the fluid, the tubing the rate. A tail well may take lift and any geometry whatever its stage and type.

From D-6 and D-103 (ruled 2026-11-06, checkpoint A2). Sources: the Goals' 10% to 20% of the accepted wells, of which 0.20 drawn gives about 15% (`r6-measurements.md`).

### SMP-54 · Well type

$\text{Cat}$(platform 0.55, subsea 0.35, onshore 0.10); tail equal shares. It is no model input: it conditions the water depth, the surface temperature, the separator, the permeability, the injection pressure and the economic floor. The onshore share is a design choice.

From D-5 (ruled 2026-11-06, checkpoint A1). Sources: [WP §1]; Liu, Wang and Kaiser (2014); EIA (2025), p. 3; [own analysis of Sodir's FactMaps].

### SMP-55 · Fluid class

$\text{Cat}$(black oil 0.55, volatile oil 0.10, gas condensate 0.15, wet or dry gas 0.20); tail 0.25 each. A gas well is `dry_gas` with probability 0.07 (tail 0.25), else `wet_gas`. The classes are McCain's by the initial producing GOR, within which SMP-68 draws it: black oil up to 356 Sm³/Sm³ (2000 scf/STB), volatile oil from 356 to 588 Sm³/Sm³ (3300 scf/STB), gas condensate from 588 to 8900 Sm³/Sm³ (50,000 scf/STB), wet gas above 8900 Sm³/Sm³, and dry gas with a condensate yield below 2.8 Sm³ per 10⁶ Sm³ (0.5 STB/MMscf). Black oils and wet and dry gases take the black oil, volatile oils and condensates a PVT table (SMP-112).

From D-3, D-4 and D-36 (ruled 2026-11-06). Sources: [RF §5, §6]; McCain (1990), pp. 151–158; [own analysis of BOEM's sands atlas]; [own analysis of Sodir's FactMaps]; the Goals' floor of about 10% of the wells per type.

### SMP-56 · Drive and life stage

`drive` is `pressure_support` (water drive or injection) with probability 0.7 for oil classes and 0.4 for gas classes, else `depletion`; tail 0.5. `life_stage` is $\text{Cat}$(initial 0.25, mid-life 0.40, late 0.35); tail equal shares. The stage places a `sol-2` well in its life (SMP-92); an `nsol-2` sample takes the stage of its date (SMP-91). The stage shares are a design choice: no public count of wells by stage is in hand.

From D-18 (ruled 2026-11-06, checkpoint C1). Sources: [RF §3, §11]; Miller, Sorrell and Speirs (2009), Table 4.4; [own analysis of BOEM's sands atlas]; [own analysis of Sodir's FactMaps].

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

### SMP-58 · Water depth and length

- platform: with probability 0.85 a shelf platform, $d_w \sim \text{LogN}_ {[15, 400]}(90, 0.6)$ m, else a deep-water dry tree, $\text{LogU}(400, 2000)$ m; tail $\text{LogU}(15, 2000)$ m;
- subsea: $d_w \sim \text{LogN}_ {[80, 3000]}(350, 0.8)$ m; tail $\text{LogU}(50, 3000)$ m;
- onshore: $d_w = 0$.

$L = z + d_w$ for a platform well, whose riser is part of the flow path, and $L = z$ for subsea and onshore wells. $d_\text{mud}$, the mudline's TVD in the modelled well, is $d_w$ for a platform well and 0 otherwise.

From D-8 (ruled 2026-11-06). Sources: [WP §2]; Liu, Wang and Kaiser (2014); [own analysis of Sodir's FactMaps]; [own analysis of BOEM's borehole file].

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

### SMP-60 · Geometry: the survey

A survey is a list of stations $(\text{MD}, \text{TVD})$ from the wellhead to the bottom at TVD $L$: straight sections, and circular builds and drops with one station per degree.

- `vertical`: straight down.
- `build_and_hold`: vertical to the kickoff $K$, a build at $b \sim U(0.3, 3.6)$ °/30 m to $\theta_h$ (SMP-59), then straight to $L$; the hold spans at least 50 m of TVD.
- `s_shaped`: $\theta_h \sim U(15°, 45°)$ and $b$ as above; after the hold, a drop at $d \sim U(0.2, 2.2)$ °/30 m to $\theta_e = \min(U(0°, 35°), \theta_h)$, then straight at $\theta_e$ over the last $\min(U(100, 400)\ \text{m}, L/4)$ of TVD; the hold spans at least 100 m of TVD.
- `high_angle`: a build at $b$ to the tangent $\theta_h = \min(U(10°, 65°), \theta_\text{heel})$, a hold, a landing build at $b_2 \sim U(1.5, 7)$ °/30 m to $\theta_\text{heel}$, and a tangent of $U(50, 300)$ m of MD at $\theta_\text{heel}$ to the heel at $L$; the hold spans at least 100 m of TVD.
- **Kickoff.** $K = d_\text{mud} + f z$ with $f \sim U(0.05, f_\text{max})$, $f_\text{max} = \min\big(0.75,\ (L - d_\text{mud} - \Delta\text{TVD}_ \text{below})/z\big)$, and $\Delta\text{TVD}_ \text{below}$ the TVD of the builds, the drop, the S well's end section and the drain's tangent, plus the hold's least TVD.
- **Fit.** Where the shape's sections do not fit below $d_\text{mud} + 0.05\thinspace z$, the build rates and the S well's drop rate are raised by steps of 20%, each to at most its range's top. A shape that still does not fit ($f_\text{max} < 0.05$), or whose measured depth exceeds 9342 m, is redrawn from the block's generator, at most 20 times; then the well is `vertical`. The angles are never lowered, so a well keeps its class.

From D-104 (ruled 2026-11-06, checkpoint B1), D-9 and D-35. Sources: each parameter uniform between the pooled 5th and 95th percentiles of [own analysis of BOEM's directional surveys, NLOG's deviation surveys, NSTA's wellbore layer and Sodir's FactMaps]. The limit of 20 redraws is the prototype's (`plans/evidence/sampler_draft_draws.py`), reached by none of its wells.

### SMP-61 · Inclination limit and drains to the heel

No cell is more than 75° from vertical; no draw of SMP-59 or SMP-60 exceeds it. A `high_angle` well stands for a well with a horizontal drain: the model's bottom is the heel at the reservoir's true vertical depth, the drain has no cells, and its length enters the inflow (SMP-72, SMP-73). The v2 datasets therefore hold no horizontal cell, which settles `specs/model/validity.md`'s decision 5 for them.

From D-10 (ruled 2026-10-30) and D-9. Sources: [WP §3], Taitel and Dukler's stratified–non-stratified transition as scanned there; Brill and Mukherjee (1999), p. 44; [own analysis of BOEM's borehole file].

### SMP-62 · Formation water

The salinity, the mass fraction of dissolved solids, is $w_s \sim \text{LogN}_ {[0.002, 0.25]}(0.06, 0.6)$; tail $\text{LogU}(0.002, 0.25)$. Truncated. The water's density at standard conditions,

$$\rho_w = 999.1 + 692\thinspace w_s + 311\thinspace w_s^2\ \text{kg/m}^3,$$

is the well's water (`FluidModel.rho_w`, PVT-WAT-1), and sets the hydrostatic pressure (SMP-63) and $c_{pw}$ (SMP-82).

From D-11 (ruled 2026-11-06, with the median and floor lowered). Sources: [RF §9]; McCain (1990), Example 16-6 (p. 449) and Fig. 16-8; Whitson and Brulé (2000), §9.2, Eq. 9.1; [own analysis of USGS's produced-water database].

### SMP-63 · Initial reservoir pressure

With the hydrostatic and lithostatic pressures in bar,

$$p_h = p_\text{atm} + \frac{g\thinspace(\rho_\text{sea} d_w + \rho_w z)}{10^5}, \qquad p_\text{lith} = p_\text{atm} + \frac{g\rho_\text{sea} d_w}{10^5} + 0.226\thinspace z,$$

with $\rho_\text{sea}$ = 1025 kg/m³ and 0.226 bar/m (1 psi/ft) below the mudline, the initial pressure is $p_i = p_h + \lambda\thinspace(p_\text{lith} - p_h)$, where

$$\text{logit}\thinspace\lambda \sim N_{[-12,\ \text{logit}\thinspace\lambda_\text{max}]}\big(\text{logit}\thinspace\lambda_{50}(z),\ 0.9\big), \qquad \lambda_{50}(z) = 0.02 + \frac{0.45}{1 + e^{-(z/1000\ \text{m} - 3.8)/0.9}},$$

and $\lambda_\text{max} = \max\big(10^{-3}, \min(0.85,\ (950 - p_h)/(p_\text{lith} - p_h))\big)$, so that $p_i \le 950$ bar. Truncated. Tail: $\lambda \sim U(0, \min(0.8, \lambda_\text{max}))$. With probability 0.04 (tail 0.10) the reservoir is initially underpressured instead: $p_i = p_h\thinspace U(0.8, 1.0)$. `pressure_class` is `under` for an underpressured reservoir, `over` where $\lambda \ge 0.25$ (about 0.6 psi/ft), and `normal` otherwise.

From D-12 and D-35 (ruled 2026-11-06). Sources: [RF §2]; $\lambda_{50}$ and its spread an own fit to depth bins of [own analysis of BOEM's sands atlas]; Nadeau, Sun and Ehrenberg (2023), Fig. 20; Hasan et al.'s bottomhole pressures to 949 bar (`validity.md`). The lithostatic gradient has no source in hand.

### SMP-64 · Surface temperature

$T_s$ is the top of THM-4's ambient profile:

- offshore (platform and subsea), core and tail, the seabed's temperature at the water depth: $T_s = T_\text{deep} + (T_\text{shelf} - T_\text{deep})\thinspace e^{-d_w/400\ \text{m}}$, with $T_\text{deep} \sim U(272, 278)$ K and $T_\text{shelf} \sim U(279, 297)$ K;
- onshore: $U(268, 303)$ K; tail $U(258, 308)$ K.

From D-14 (ruled 2026-11-06). Sources: [WP §2] and the seabed temperatures it quotes; Hasan and Kabir (2018), Fig. 7.17; [own analysis of NOAA's World Ocean Atlas 2023].

### SMP-65 · Geothermal gradient and reservoir temperature

$G \sim \text{LogN}_ {[G_\text{lo}, G_\text{hi}]}(G_{50}, 0.22)$ K/km below the mudline, with the median $G_{50} = 20 + 10\thinspace e^{-\max(d_w - 100\ \text{m},\ 0)/400\ \text{m}}$ clamped to $[G_\text{lo}, G_\text{hi}]$; tail $U(G_\text{lo}, G_\text{hi})$. Then $T_r = T_s + G z$. With $L$ and $z$ in km and temperatures in K, $G_\text{lo} = \max(12,\ 10\thinspace L/z)$ keeps the model's gradient $(T_r - T_s)/L$ at 10 K/km or more, and $G_\text{hi} = \min\big(55,\ (450 - T_s)/z\big)$ keeps $T_r \le 450$ K; where $G_\text{lo} > G_\text{hi}$, $G_\text{lo}$ is set to $G_\text{hi}$. Truncated.

From D-13 and D-35 (ruled 2026-11-06). Sources: [RF §4]; Nadeau, Sun and Ehrenberg (2023), Fig. 7A and Table 1; [own analysis of BOEM's sands atlas]; [own analysis of Sodir's FactMaps]; the CasADi backend's limit near 5 K/km (`validity.md`).

### SMP-66 · Separator and the nominal downstream pressure

The host's first-stage separator at the start of the well's life:

- platform and subsea: $p_\text{sep} \sim \text{LogN}_ {[5, 85]}(35, 0.45)$ bar; $T_\text{sep} \sim U(300, 340)$ K on a platform, $U(290, 320)$ K for subsea;
- onshore: $p_\text{sep} \sim \text{LogN}_ {[2, 70]}(15, 0.6)$ bar, $T_\text{sep} \sim U(285, 320)$ K.

A subsea well's flowline and riser add $\Delta p_\text{fl} = U(5, 30) + U(0.03, 0.07)\thinspace d_w$ bar, with $d_w$ in m; other wells have $\Delta p_\text{fl} = 0$. At a stage, the nominal pressure downstream of the choke is

$$p_{s,\text{nom}} = \min\big(\max(r_s\thinspace p_\text{sep} + \Delta p_\text{fl},\ 1.5\ \text{bar}),\ p_\text{cap}\big),$$

with the stage's factor $r_s$ = 1 (initial), $U(0.5, 1)$ (mid-life) or $U(0.1, 0.6)$ (late), and $p_\text{cap}$ = 120 bar, or 250 bar for a subsea well. Tail: $p_{s,\text{nom}} = \min\big(\text{LogU}(1.5, 120) + \Delta p_\text{fl},\ p_\text{cap}\big)$. Over the life $r_s$ steps down (SMP-89). The separator is also the fluid's (`BlackOil(p_sep, T_sep)`, PVT-OIL-5): the GOR and the gas gravity are as reported at it.

From D-15 (ruled 2026-11-06, with the subsea flowline and its higher cap). Sources: [WP §7]; Guo, Lyons and Ghalambor (2007), §10.2.5 (p. 10/124) and Table 10.4; [own analysis of Petrobras' 3W Dataset].

### SMP-67 · Oil and gas gravities

| Class | API gravity | Gas gravity $\gamma_g$ (air = 1) | Tail |
|---|---|---|---|
| black oil | $N_{[16,45]}(33, 4.5)$ | $N_{[0.58,0.95]}(0.70, 0.07)$ | both uniform over the bounds |
| volatile oil | $U(40, 50)$ | $N_{[0.62,1.0]}(0.75, 0.06)$ | $\gamma_g$ uniform |
| gas condensate | $U(42, 58)$ | $N_{[0.60,0.90]}(0.72, 0.05)$ | $\gamma_g$ uniform |
| wet and dry gas | $U(45, 58)$ | $N_{[0.571,0.75]}(0.62, 0.035)$ | $\gamma_g$ uniform |

Independent draws; truncated. $\rho_o$ follows from the API gravity (PVT-OIL-2) and $\rho_{g,\text{sc}}$ from $\gamma_g$ (PVT-GAS-6). A table fluid's densities are its table's (SMP-113, SMP-114).

From D-16 (ruled 2026-11-06, the black oils' gas heavier and independent of the API) and D-35. Sources: [RF §7]; Valkó and McCain (2003), Table 2.1; McCain (1990), pp. 155–156; Whitson and Brulé (2000), Table 2.1 and §2.5; [own analysis of BOEM's sands atlas]; [own analysis of Sodir's FactPages]; the black oil's API range and Sutton's gas gravities from 0.571 (`validity.md`).

### SMP-68 · Saturation pressure and the initial GOR

- **Black oil.** $p_\text{sat} = p_i\thinspace(1 - x)$ at $T_r$, with $x = 0$ (saturated at discovery) with probability 0.20, else $x \sim \text{LogN}_ {[0.005, 0.85]}(0.10, 1.2)$; tail $x \sim U(0, 0.85)$. $R_{si} = R_{so}(p_\text{sat}, T_r)$ by the well's black oil (PVT-OIL-6). Where that exceeds 356 Sm³/Sm³, $p_\text{sat}$ is lowered by bisection to the pressure at which $R_{so}$ is 356 Sm³/Sm³, and $R_{si}$ = 356. The black oil's bubble point is then $p_\text{sat}$ at $T_r$ (PVT-OIL-15); no `p_bubble` is passed.
- **Volatile oil:** $R_{si} \sim \text{LogU}(356, 588)$ Sm³/Sm³.
- **Gas condensate:** $R_{si} \sim \text{LogN}_ {[588, 8900]}\big(2000\thinspace e^{-0.3\thinspace(z/1000\ \text{m} - 3.4)},\ 0.6\big)$ Sm³/Sm³; tail $\text{LogU}(588, 8900)$.
- **Wet gas:** a condensate yield of $\text{LogN}_ {[2.8, 112]}(30, 1.2)$ Sm³ per 10⁶ Sm³, tail $\text{LogU}(2.8, 112)$, and $R_{si} = 10^6$/yield. **Dry gas:** a yield of $\text{LogU}(0.5, 2.8)$.
- **Table fluids.** A volatile oil's or condensate's saturation pressure and initial GOR are its table's (SMP-115), and its drawn $R_{si}$ is the composition's target (SMP-113). The `fluid` block still draws $x$ for them as for a black oil, with the probability 0.40 (volatile oil) or 0.30 (condensate) of $x = 0$ and the median 0.05, so that its numbers do not depend on the table; $p_i\thinspace(1 - x)$ is kept only as a record.

`saturation` is `single_phase_gas` for wet and dry gas; otherwise `saturated` where $p_r \le p_\text{sat}$ at the well's state (`sol-2`: its date, SMP-92; `nsol-2`: its start) and `undersaturated` where $p_r > p_\text{sat}$.

From D-17 (ruled 2026-11-06; for table fluids, L2 of 2026-11-09) and D-65. Sources: [RF §7]; [own analysis of BOEM's sands atlas]; [own analysis of Sodir's FactMaps]; Whitson and Brulé (2000), Table 2.1; Beggs and Robinson's $R_{so}$ range (`validity.md`).

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

### SMP-70 · Producing GOR

- **Oils.** $R_{go} = R_{si}$ where $p_r \ge p_\text{sat}$; below, $R_{go} = R_{si}\thinspace\Phi(p_r/p_\text{sat})$ from the material balance of solution-gas drive: Muskat's equation (Craft and Hawkins 1991, Ch. 9, Eqs. 9.33 to 9.35) integrated by RK4 in pressure from $p_\text{sat}$ down to $0.05\thinspace p_\text{sat}$ on the well's black oil at $T_r$, whose $R_{so}$ at $p_\text{sat}$ is $R_{si}$, and interpolated linearly in $p_r/p_\text{sat}$. $k_g/k_o$ is Craft and Hawkins's Fig. 9.16 in the core; in the tail, the book's laboratory curves blended in $\log_{10} k_g/k_o$ by $w \sim U(0, 1)$, from Fig. 9.16 at $w = 0$ to Table 10.2 at $w = 1$. No cap. Just below the saturation pressure $\Phi$ dips under 1, while the freed gas is below its critical saturation, and then rises.
- **Gas condensates.** $R_{go} = R_{si}\big(1 + c_k\max(0,\ 1 - p_r/p_\text{sat})\big)$, $c_k \sim U(0.5, 3)$, as liquid drops out in the reservoir.
- **Wet and dry gas.** $R_{go} = R_{si}$.

$w$ and $c_k$ are drawn by the `stage` block; the life uses its start's (SMP-87). The construction of the curves (the connate water, the critical gas saturation and the blend) is `plans/evidence/solution_gas_drive.py`'s.

From D-19 and D-44 (ruled 2026-11-06, checkpoint C1). Sources: Craft and Hawkins (1991), Ch. 9 (Eqs. 9.33 to 9.35, Example 9.1, Fig. 9.16) and Ch. 10 (Table 10.2); Golan and Whitson (1991), Example 4.11 (p. 454); McCain (1990), pp. 151, 153. The condensates' factor is an own choice.

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

### SMP-72 · Oil wells' productivity index

At a reservoir state $(p_r, R_{go})$ the radial productivity index is, in SI units converted to Sm³/(d bar) of stock-tank liquid,

$$J_V = \frac{2\pi k h_n}{\mu_o B_o\thinspace(\ln(r_e/r_w) - 0.75 + s)},$$

with $\mu_o$ and $B_o$ of the well's PVT model at $(p_r, T_r)$ and $R_{go}$: its black oil, or for a volatile oil its table set at $R_{go}$. Then $J = f\thinspace J_V$, with the geometry's factor $f$ set once, from the well's initial-stage draw:

- `high_angle`: $f = \eta\thinspace J_H/J_V$, with Joshi's horizontal-well index $J_H$ (Economides et al. 1994, Eqs. 2-46 to 2-48) for the drain $L_d$ with the anisotropy $I_\text{ani}$, on the drainage length $r_{eH} = \max(r_e, L_d/1.6)$, and Dikken's friction efficiency along the liner, taken as the tubing, in closed form: $\eta = (1 + \Pi/1.5)^{-1/3}$, $\Pi = c_\text{lin} J_H^2 u L_d$, with $u = \delta_d p_r$ at a rate capped at the tubing's erosional rate (SMP-75), and $c_\text{lin}$ the liner's friction and acceleration at the heel's rate (`plans/evidence/drain_friction.py`). The liner carries the liquid of the state's stream, with $B_l = (1 - \text{WLR})\thinspace B_o + \text{WLR}$, the density $\big((1 - \text{WLR})(\rho_o + R_{go}\rho_{g,\text{sc}}) + \text{WLR}\thinspace\rho_w\big)/B_l$ kept within 400 to 1100 kg/m³ and the viscosity $(1 - \text{WLR})\thinspace\mu_o + \text{WLR}\cdot 0.5$ mPa s; its friction factor is Chen's (FRIC-4) at the tubing's roughness, times 1.5 for a perforated liner. No cap.
- `build_and_hold`, `s_shaped`: $f = (\cos\theta_\text{end})^{-1/2}$, with $\theta_\text{end}$ the last section's inclination; a placeholder until Cinco-Ley's slant skin (Economides et al. 1994, §5-4) is read.
- `vertical`: $f = 1$.

From D-20, D-21 (ruled 2026-11-06, checkpoints B1 and D) and D-45. Sources: [WP §4]; Economides et al. (1994), §2-7 (Eqs. 2-46 to 2-48, Example 2-10, Fig. 2-10); Joshi (1991), Ch. 3 and Ch. 10; Dikken (1990); Brekke and Lien (1994).

### SMP-73 · Gas wells' radial coefficients

A gas or condensate well's Forchheimer coefficients of $p_r^2 - p_{wf}^2 = a\thinspace q + b\thinspace q^2$, with $q$ the standard gas rate (Economides et al. 1994, Eqs. 4-32 and 4-33), are for a vertical well

$$a_V = \frac{p_\text{sc} T_r\thinspace\mu_g Z\thinspace(\ln(r_e/r_w) - 0.75 + s)}{\pi k h_n T_\text{sc}}, \qquad b_V = \frac{a_V D}{\ln(r_e/r_w) - 0.75 + s},$$

in SI units, with $\mu_g$ and $Z$ of the reservoir gas at $(p_r, T_r)$, $(p_\text{sc}, T_\text{sc})$ the standard conditions (PVT-GAS-2) and $D$ of SMP-71 in s/Sm³ (1 d/Mscf = 3051 s/Sm³), so that the non-Darcy flow is a skin $D q$. The geometry's factors on them are set once, from the well's initial-stage draw:

- `high_angle`: $a = a_V/(J_H/J_V)$, with SMP-72's ratio; $b = b_V\thinspace(I_\text{ani} h_n/L_d)^2 + c_\text{lin} L_d/4$: the non-Darcy loss goes with the square of the rate per completed length, and the liner adds its friction to first order, with $c_\text{lin}$ at the heel's rate at the design drawdown, capped at the tubing's erosional rate, for the state's gas in a liner of the tubing's diameter, at most 157.1 mm (7 in), with SMP-72's friction factor (`drain_friction.py`);
- `build_and_hold`, `s_shaped`: $a = a_V/f$ and $b = b_V/f$, with SMP-72's $f$;
- `vertical`: $a = a_V$ and $b = b_V$.

The model's inflow is fitted to these at every reservoir state (SMP-117); they also give a gas well's design rates (SMP-75, SMP-80).

From D-23 (ruled 2026-11-06), D-62 (D2) and D-21. Sources: [WP §4]; Economides et al. (1994), §4-3 and §4-4 (Eqs. 4-32 to 4-34); Joshi (1991), Ch. 10.

### SMP-74 · Inflow model

An oil well takes `vogel` (INF-1) if $p_\text{sat} \ge 0.8\thinspace p_r$ and its WLR is below 0.6, else `productivity_index` (INF-2); a gas or condensate well takes `forchheimer` (INF-10, feature 022). The rule is evaluated once, at the life's start (SMP-83), and the model is kept over the life in both datasets; its coefficients follow the reservoir's state (SMP-88).

From D-22 (ruled 2026-11-06), D-45 and D-60 (ruled 2026-11-05, checkpoint J1). Sources: Vogel (1968); `validity.md`, note 10. The 0.8 is an own choice.

### SMP-75 · Tubing

The `completion` block takes the smallest inner diameter of {62.0, 76.0, 100.5, 124.3, 157.1} mm (2-7/8, 3-1/2, 4-1/2, 5-1/2 and 7 in tubing), and of 220.5 mm (9-5/8 in) too for gas and condensate wells, at which the no-slip mixture velocity at the design rate stays below the erosional velocity $v_e = 1.22\thinspace C/\sqrt{\rho_m}$ m/s, with $\rho_m$ in kg/m³ and $C \sim U(100, 150)$; with probability 0.3 (tail 0.5) it takes one size larger, where there is one.

- **The design point** is the initial state, when the tubing is run: $p_i$, $R_{si}$, no water for oils and a WGR of 10 for gases, the design drawdown $\delta_d p_i$, and the wellhead at $1.5\thinspace p_\text{sep} + 5\ \text{bar} + \Delta p_\text{fl}$ and $T_s + 0.6\thinspace(T_r - T_s)$. The design rate is an oil's $J\delta_d p_i$, and a gas well's SMP-73 rate at $p_i^2 - ((1 - \delta_d)\thinspace p_i)^2$ with its liquid, both with the geometry's factor before a drain's friction. A gas well's coefficients there take $Z$ at $p_i$, and $\mu_g$ and $D$ as at the draw's reservoir state (the start's, for a dataset's well).
- **Too small.** If no size fits, the well takes the largest and is flagged. The flag is a check: its share is measured, and a redraw is added only if it stays above a few percent.
- The tubing's erosional rate, the liquid rate at $v_e$ in the chosen tubing, caps the design rates of SMP-72, SMP-73 and SMP-80. The tubing is the well's for its life.

From D-25 (ruled 2026-11-06) and D-100 (checkpoint D1). Sources: [WP §6]; Guo, Lyons and Ghalambor (2007), Appendix B.

### SMP-76 · Roughness

$\varepsilon \sim \text{LogU}$ by stage: initial $(1.5\cdot10^{-6},\ 3\cdot10^{-5})$ m, mid-life $(10^{-5},\ 6\cdot10^{-5})$ m, late $(2\cdot10^{-5},\ 1.5\cdot10^{-4})$ m; tail $\text{LogU}(1.5\cdot10^{-6},\ 1.5\cdot10^{-4})$ m, SMP-42's range. Friction from Chen's factor (FRIC-4). Over the life it ages from the initial draw's to the late draw's (SMP-88).

From D-26 (ruled 2026-11-06). Sources: [WP §6]. The split by stage is an own choice.

### SMP-77 · Choke

A Simpson choke (CHK-5) for every well, with the fully open bean $d = \min(\beta D,\ 101.6\ \text{mm})$, $\beta \sim U(0.15, 0.6)$ (tail $U(0.15, 0.75)$) of the tubing's diameter $D$, a discharge coefficient $C_D \sim U(0.75, 0.9)$, and $K_c = C_D\thinspace\pi d^2/4$. The profile, `chk_profile`, is one of CHK-7 to CHK-10 with equal shares. The Bernoulli choke (CHK-6) is not drawn.

From D-27 (ruled 2026-11-06, checkpoint E1). Sources: [WP §8]; Guo, Lyons and Ghalambor (2007), Eq. 5.4 (p. 5/60); [own analysis of BSEE's well tests] for the cap at 4 in.

### SMP-78 · Gas lift at a stage

The `lift` block decides whether the well at its stage (SMP-51) is lifted:

- An oil well needs lift if its static column cannot reach the surface, $p_r < p_{s,\text{nom}} + g\rho_m L/10^5$, with $\rho_m$ the no-slip density of its stream at the mean of $p_r$ and $p_{s,\text{nom}}$ and the mean of $T_r$ and the design wellhead temperature $T_s + 0.6\thinspace(T_r - T_s)$. Such a well is lifted.
- Any other well is lifted with a probability by stage (initial, mid-life, late): oils offshore 0.20, 0.45 and 0.70; oils onshore 0.15, 0.25 and 0.35; gas and condensate wells 0, 0.05 and 0.15. Tail: 0.5 for oils and 0.25 for gas classes. No GOR limits lift.
- In the core, a well that this draw lifts, whose pressure-balance point $H_\text{bal}$ (SMP-79) lies above $0.3\thinspace L$, stays on natural flow: an operator does not lift a strong well.

A dataset's well takes its lift from these stage draws through its life (SMP-90).

From D-28 (ruled 2026-11-06, checkpoint F1: as drafted, no tuning by GOR quartile). Sources: [WP §1, §9]; Hasan and Kabir (2018), §12.2.1; von Flatern (2015); Goh et al. (2007). The stage shares are own choices.

### SMP-79 · Gas lift: injection depth and lift-gas temperature

A lifted well's operating valve lies at the true vertical depth $H$ below the wellhead:

- the packer at $L - \Delta z_\text{pk}$, $\Delta z_\text{pk} \sim U(30, 150)$ m;
- the injection pressure at the top of the modelled well, $p_\text{inj} \sim U(48, 85)$ bar at the surface for platform and onshore wells and $U(130, 210)$ bar at the tree for subsea wells; the valve's pressure drop $\Delta p_v \sim U(3.4, 13.8)$ bar; and the casing pressure at depth $H$ by Gilbert, $p_\text{inj}(1 + H/12\thinspace 192\ \text{m})$;
- the tubing's pressure below the valve at the lift's design point, $p_{wf} - G_l\thinspace(L - H)$, with $p_{wf} = p_r\thinspace(1 - U(0.3, 0.7))$ and $G_l$ the no-slip gradient of the stream at $(p_{wf}, T_r)$;
- $H_\text{bal}$, where the casing pressure less $\Delta p_v$ equals the tubing's pressure ($L$ where the two do not cross), and $H = \min\big(\max(H_\text{bal}, 0.3\thinspace L),\ L - \Delta z_\text{pk}\big)$.

The injection point's measured depth, `injection_md`, is the survey's MD at TVD $H$, interpolated linearly between stations. The lift gas's temperature is $T_{lg} = T_a(H) - \phi\thinspace\big(T_a(H) - T_s\big)$, $\phi \sim U(0, 0.3)$, with the ambient $T_a(H) = T_s + (T_r - T_s)\thinspace H/L$: from THM-9's default at the valve to 30% of the way towards the surface temperature.

From D-29 (ruled 2026-11-06, checkpoint F) and D-30 (ruled 2026-11-06). Sources: Golan and Whitson (1991), §5.3 (Eqs. 5.16 and 5.17, Example 5.7, pp. 528–536), with Gilbert's casing pressure; Guo, Lyons and Ghalambor (2007), Example 13.1; Hasan and Kabir (2018), §12.2.2; [own analysis of BSEE's well tests and Petrobras' 3W Dataset]. $\phi$ is an own choice.

### SMP-80 · Gas lift: capacity

The most lift gas a sample may inject is $w_{lg,\max} = k_q\thinspace\text{GLR}_ \text{cap}$ kg/s, with $k_q = \rho_{g,\text{sc}}\thinspace q_l/86\thinspace 400$ and $q_l$ the design liquid rate at the stage's state, $J\delta_d p_r$ for an oil and the liquid at SMP-73's gas rate at the design drawdown for a gas well, capped at the tubing's erosional rate. The injected gas–liquid ratio at capacity, $\text{GLR}_ \text{cap}$, is log-uniform over the part of $[30, 1000]$ Sm³/Sm³ that gives $w_{lg,\max}$ within $[0.1, 3.5]$ kg/s, core and tail. Where no part does, $w_{lg,\max}$ is 0.1 kg/s if $1000\thinspace k_q < 0.1$ kg/s, else 3.5 kg/s.

From D-31 (ruled 2026-11-06, checkpoint F2, truncated not clipped). Sources: [WP §9]; von Flatern (2015), Fig. 1; Guo, Lyons and Ghalambor (2007), Example 13.1; Golan and Whitson (1991), Example 5.8 (p. 536) and p. 538; Economides et al. (1994), Example 19-1 (p. 525); [own analysis of Petrobras' 3W Dataset].

### SMP-81 · Heat-transfer coefficient

$h = U_t k_e/(k_e + r_{ti} U_t T_D)$, the wellbore's overall coefficient in series with the formation's transient resistance (Hasan and Kabir 2012, Eq. 6), with $U_t \sim \text{LogU}(11, 85)$ W/(m² K), $k_e \sim U(1.5, 2.6)$ W/(m K), $r_{ti} = D/2$, and

$$T_D = \ln\big(e^{-0.2\thinspace t_D} + (1.5 - 0.3719\thinspace e^{-t_D})\sqrt{t_D}\big), \qquad t_D = \frac{\alpha\thinspace t_a}{r_h^2}$$

(Hasan and Kabir 2012, Eq. 2), with Ramey's diffusivity $\alpha = 1.03\cdot10^{-6}$ m²/s, the hole's radius $r_h$ = 0.108 m (0.156 m for tubing above 101 mm), and the well's age $t_a$ log-uniform over 30 to 365 days (initial), 1 to 8 years (mid-life) or 5 to 25 years (late). Tail: $h \sim U(2, 40)$ W/(m² K). Over the life the age grows (SMP-88).

From D-32 (ruled 2026-11-06). Sources: [WP §10]; Hasan and Kabir (2012), Eqs. 2 and 6; Ramey (1962), p. 431; Hasan and Kabir (2018), Table 6.1 and Fig. 7.39.

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

## v2: one life for both datasets

### SMP-83 · A well's life

Every well has one life, which both datasets use: an `nsol-2` well lives it, and a `sol-2` well is the same life at one date (SMP-92).

- **Start**, $t = 0$: the well's initial-stage draw, its $r_0 = p_r/p_i$ and its water (SMP-69).
- **End**, $t = T_\text{life}$, of a core well, its reservoir run towards exhausted: under depletion $r_1 = \min\big(U(0.15, 0.3), r_0\big)$ for oils and $\min\big(U(0.15, 0.2), r_0\big)$ for gas classes, with the late-stage draw's water; under pressure support $r_1 = \min(r_{p,L}, r_0)$, with $r_{p,L}$ the late-stage draw's, and the water at its cap: a WLR of 0.98 for oils, and a WGR of 10,000 Sm³ per 10⁶ Sm³ for gas classes, where the well waters out (D-107; D-105 had 1000).
- **A tail well** starts from a core initial state drawn for it, $r_0 \sim U(0.85, 1)$ under depletion and $U(0.97, 1)$ under pressure support, with a WLR of 0 with probability 0.5 and else $U(0, 0.1)$, or a WGR of $\text{LogU}(2, 20)$; it ends at its tail state, $r_1 = \min(r_{p,\text{tail}}, r_0)$, with its tail water.
- The life stops where the well stops paying: `nsol-2`'s life scan (SMP-101) and `sol-2`'s producibility filter (SMP-103) apply the economic floor along it.

The trends between the start and the end (SMP-84 to SMP-90) take their parameters from the `evolution` block.

From D-105 (ruled 2026-11-06: one life for both datasets), D-41, D-42 and D-107 (ruled 2026-11-10, L4). Sources: [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well]; [own analysis of BOEM's sands atlas].

### SMP-84 · Life span and the depletion clock

$T_\text{life} \sim \text{LogU}(3, 30)$ years. The trends follow the reservoir's cumulative production, normalised to the life span,

$$F(t) = \frac{N(t)}{N(T_\text{life})}, \qquad N(t) = \begin{cases} t, & t \le t_\text{pl} \\ t_\text{pl} + \dfrac{1 - (1 + b_\text{A} D_\text{A}\tau)^{1 - 1/b_\text{A}}}{(1 - b_\text{A})\thinspace D_\text{A}}, & \tau = t - t_\text{pl} > 0 \end{cases}$$

a plateau of unit rate and then Arps's hyperbolic decline, exponential, $N = t_\text{pl} + (1 - e^{-D_\text{A}\tau})/D_\text{A}$, where $b_\text{A} < 10^{-6}$. The plateau $t_\text{pl} = U(0, 0.25)\thinspace T_\text{life}$; $b_\text{A} \sim U(0, 0.4)$ under depletion and $U(0.3, 0.5)$ under pressure support; and $D_\text{A}$ from the decline's rate at the end of the life over the plateau's, $q_\text{end} \sim \text{LogU}(0.03, 0.3)$: $D_\text{A} = (q_\text{end}^{-b_\text{A}} - 1)/(b_\text{A}\tau_\text{end})$, or $-\ln q_\text{end}/\tau_\text{end}$ where $b_\text{A} < 10^{-6}$, with $\tau_\text{end} = T_\text{life} - t_\text{pl}$. The clock depends on no solve, so the controls stay open loop.

From D-41 (ruled 2026-11-06, with the steeper decline and lives shorter than 8 years). Sources: Ahmed (2006), Eqs. 16-1 to 16-4 (pp. 1238–1239); Golan and Whitson (1991), p. 385; Miller, Sorrell and Speirs (2009), Tables 4.2 to 4.4 [RF §11]; [own analysis of BOEM's OGOR-A production, ANP's production by well and NSTA's production by well].

### SMP-85 · Reservoir pressure along the life

- Oils, and every well under pressure support: $p_r(t) = p_i\big(r_1 + (r_0 - r_1)(1 - F)^\gamma\big)$, $\gamma \sim U(1, 2)$.
- Gas and condensate wells under depletion: $p/Z$ linear in $F$, $p_r/Z(p_r) = (1 - F)\thinspace p_0/Z_0 + F\thinspace p_1/Z_1$, with $p_0 = r_0 p_i$, $p_1 = r_1 p_i$ and the well's DAK $Z$ (PVT-GAS-11) at $T_r$, solved for $p_r$.

From D-42 (ruled 2026-11-06). Sources: Ahmed (2006), Eq. 13-8 (p. 861); Golan and Whitson (1991), Example 4.11 (Table E4.11a, p. 454).

### SMP-86 · Water along the life

- **Oils.** The WLR stays at the start's until breakthrough at the recovery $F_\text{bt}$, jumps to $\text{WLR}_ \text{bt}$, and then rises with its logit linear in $F$ to the end's $\text{WLR}_ 1$:

  $$\text{logit}\thinspace\text{WLR}(F) = \text{logit}\thinspace\text{WLR}_ \text{bt} + \big(\text{logit}\thinspace\text{WLR}_ 1 - \text{logit}\thinspace\text{WLR}_ \text{bt}\big)\frac{F - F_\text{bt}}{1 - F_\text{bt}}, \qquad F \ge F_\text{bt},$$

  never below the start's WLR, with $\text{WLR}_ \text{bt}$ at most $\text{WLR}_ 1$; where $\text{WLR}_ 1$ is not above the start's, the WLR stays at the start's. Under pressure support $F_\text{bt} \sim U(0.05, 0.4)$ and $\text{WLR}_ \text{bt} \sim U(0.05, 0.3)$; under depletion $U(0.2, 0.7)$ and $U(0.02, 0.1)$.
- **Gas classes.** The WGR stays at the start's $\text{WGR}_ 0$ until $F_\text{bt}$ and then rises log-linearly in $F$ to the end's, $\text{WGR}_ 0\thinspace(\text{WGR}_ 1/\text{WGR}_ 0)^{(F - F_\text{bt})/(1 - F_\text{bt})}$; where the end's is not above the start's, it stays at the start's.

From D-43 (ruled 2026-11-06). Sources: Ahmed (2006), Eq. 5-28 (p. 309), Eqs. 14-10 and 14-22 and Fig. 14-12 (pp. 935, 941), pp. 970 and 983; [RF §9]; [own analysis of Sodir's FactMaps].

### SMP-87 · GOR along the life

$R_{go}(t)$ is SMP-70's at $p_r(t)$, with the start's $R_{si}$, $p_\text{sat}$, $k_g/k_o$ blend and condensate factor $c_k$: flat while the reservoir is above its saturation pressure, then rising, so that the pressure links the three trends. Wet and dry gas keep $R_{si}$.

From D-44 (ruled 2026-11-06 with checkpoint C1). Sources: McCain (1990), pp. 151, 153; Golan and Whitson (1991), Example 4.11 (p. 454); Craft and Hawkins (1991), Ch. 9.

### SMP-88 · Well properties along the life

- **Heat transfer:** SMP-81's $h$, with the start's $U_t$, $k_e$ and age $t_{a,0}$, at the age $t_{a,0} + t$. A tail well keeps its tail $h$.
- **Roughness:** $\varepsilon(t) = \varepsilon_0\thinspace\big(\max(\varepsilon_L, \varepsilon_0)/\varepsilon_0\big)^{t/T_\text{life}}$, from the initial draw's $\varepsilon_0$ towards the late draw's $\varepsilon_L$.
- **Inflow coefficients:** at the trends' state at $t$, without the drift of SMP-96 or SMP-100: an oil's $J$ by SMP-72 at $\big(p_r(t), R_{go}(t)\big)$ and the WLR, with the start's geometry factor $f$; a gas well's coefficients by SMP-117 at $p_r(t)$ and $R_{go}(t)$.
- **Fixed for the life:** the inflow model (SMP-74), the survey, the tubing, the choke's bean and profile, $T_r$, $T_s$, the salinity, the gravities, $R_{si}$, $p_\text{sat}$, the non-Darcy coefficient, the geometry's factors and the heat capacities.

From D-45 (ruled 2026-11-06). Sources: Hasan and Kabir (2012), Eq. 2; [WP §6, §10].

### SMP-89 · Separator along the life

$p_{s,\text{nom}}(t)$ is SMP-66's with the factor $r_s(t)$: 1 until $t_1$, the mid-life draw's $r_s$ until $t_2$, and the smaller of it and the late draw's after, with $t_1 < t_2$ the sorted pair of two draws of $U(0.25, 0.9)\thinspace T_\text{life}$. A tail well scales its $p_{s,\text{nom}} - \Delta p_\text{fl}$ by the same factor, with $p_{s,\text{nom}}(t)$ at least 1.5 bar.

From D-46 (ruled 2026-11-06). Sources: [WP §7].

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

### SMP-91 · Life stage of a date

A date $t$ is `initial` in the first year ($t < 1$), `late` from half the life span ($t \ge T_\text{life}/2$), and `mid_life` between. An `nsol-2` sample carries the stage of its date.

From D-41 (ruled 2026-11-06). Sources: `sampling-draft.md`, The well and its states.

## v2: `sol-2` samples

### SMP-92 · A `sol-2` well

A `sol-2` well is its life (SMP-83) at one date $t^\ast$, drawn from the date's generator (SMP-109) uniformly within its drawn stage (SMP-56): $[0, 1)$ for `initial`, $[1, T_\text{life}/2)$ for `mid_life` and $[T_\text{life}/2, T_\text{life}]$ for `late`, in years. A tail well's date is $T_\text{life}$, the end of its life, which holds its tail state. The well's nominal state is the life's at $t^\ast$ without drift: $p_{r,\text{nom}} = p_r(t^\ast)$, $R_{go,\text{nom}}$ and the water (SMP-85 to SMP-87), $p_{s,\text{nom}}(t^\ast)$, $h$, $\varepsilon$ and the inflow's coefficients (SMP-88, SMP-89), and the lift at $t^\ast$ (SMP-90). Its `saturation` label is at $p_{r,\text{nom}}$, and its `life_state` follows from its stage and drive.

From D-105 (ruled 2026-11-06). Sources: as SMP-83.

### SMP-93 · Choke position

$u = 1$ with probability 0.05, else $u \sim U(0.05, 1)$, independent per slot, core and tail alike.

From D-37 (ruled 2026-11-06). Sources: the plan's Goals (goal 4: modest weight at a fully open choke, close to uniform); SMP-18 for the smallest opening; Guo, Lyons and Ghalambor (2007), §5.1 (p. 5/60); [own analysis of Petrobras' 3W Dataset]; [own analysis of NSTA's production by well].

### SMP-94 · Lift-gas rate

A lifted well has $w_{lg} = 0$ with probability 0.20, else $w_{lg} = v\thinspace w_{lg,\max}$ with $v \sim U(0, 1)$. A natural well has $w_{lg} = 0$.

From D-38 (ruled 2026-11-06, with 0.20). Sources: [WP §9]; von Flatern (2015), Fig. 1; Golan and Whitson (1991), §5.3 (p. 538); Asheim (1988), p. 1452.

### SMP-95 · Separator pressure

$p_s \sim U\big(\max(0.9\thinspace p_{s,\text{nom}},\ 1.5\ \text{bar},\ \Delta p_\text{fl}),\ 1.1\thinspace p_{s,\text{nom}}\big)$, or $1.1\thinspace p_{s,\text{nom}}$ where that interval is empty.

From D-39 (ruled 2026-11-06). Sources: SMP-20; [WP §7]; Guo, Lyons and Ghalambor (2007), §5.2 (p. 5/60).

### SMP-96 · Drift of the reservoir

Around the well's nominal state (SMP-92):

- $p_r = p_{r,\text{nom}}\thinspace U(0.98, 1.02)$;
- the producing GOR: SMP-70's at the drifted $p_r$, times $U(0.95, 1.05)$;
- the water: an oil's WLR $\min\big(0.98,\ \text{WLR}_ \text{nom}\thinspace U(0.95, 1.05)\big)$; a gas well's WGR $\text{WGR}_ \text{nom}\thinspace U(0.95, 1.05)$, and its WLR from it and the drifted GOR.

$T_r$, $T_s$, $T_{lg}$, $h$, $\varepsilon$ and the inflow's coefficients do not drift. An attempt with $p_r \le p_s$ has no operating point (SMP-104).

From D-40 (ruled 2026-11-06). Sources: the plan's Goals (±2% and ±5%); SMP-21 and SMP-22; McCain (1990), pp. 151, 153.

## v2: `nsol-2` samples

### SMP-97 · Dates and slots

A life of 200 slots spans $[0, T_\text{end}]$, the life scan's end (SMP-101): 160 operating samples at $t_j = j\thinspace T_\text{end}/159$, $j$ = 0 to 159, and 8 test periods of 5 samples at $t_m = (m - 1/2)\thinspace T_\text{end}/8$, $m$ = 1 to 8. The slots are numbered in the order of their dates. A sample's reservoir state is the trends' at its date (SMP-85 to SMP-89), with its date's drift (SMP-100); a test's samples share their date's state. The controls' generator (SMP-109) draws the plan of every slot's controls, separator pressure and drift, in slot order, before any slot is solved.

From D-48 and D-84 (ruled 2026-11-06). Sources: Golan and Whitson (1991), §5.3. The schedule of tests is a design choice for coverage.

### SMP-98 · Control history

The choke and the lift gas follow set points $u_\text{sp}$ and $w_\text{sp}$, constant between changes and open loop:

- **Start:** $u_\text{sp} = 1$ with probability 0.05, else $U(0.05, 1)$. When lift starts, $w_\text{sp} = U(0.1, 1)\thinspace w_{lg,\max}$.
- **Changes:** at each operating sample after the first, each set point changes with probability $p_c \sim \text{LogU}(0.5, 0.95)$, drawn once per well; the lift gas's only while lift is installed.
- **A change** is, with probability 0.9, a step of $U(-0.1, 0.1)$ times its range's width ($w_{lg,\max}$ for the lift gas), reflected into its range; else a new draw: the choke 1 with probability 0.05, else $U(0.05, 1)$; the lift gas 0 with probability 0.20, else $U(0.1, 1)\thinspace w_{lg,\max}$.
- **Ranges:** the choke's $[0.05, 1]$; the lift gas's $[0.1, 1]\thinspace w_{lg,\max}$ between changes, so that only a new draw shuts it in.
- The set points never follow a solve; a redraw (SMP-105) changes its own slot only.

From D-47 (ruled 2026-11-06, with the lift gas's floor of the same day). Sources: [own analysis of Petrobras' 3W Dataset]. The step's size and the probabilities are own choices, set on the rework's pilot.

### SMP-99 · Test periods

A natural well sweeps its choke in every test. A well with lift installed sweeps its choke in tests 1, 3, 5 and 7 and its lift gas in tests 2, 4, 6 and 8:

- **choke sweep:** $u = 1$, then one draw in each quarter of $[0.05, 1)$ from the top down, $u = 0.05 + 0.95\thinspace x$ with $x \sim U(3/4, 1)$, $U(1/2, 3/4)$, $U(1/4, 1/2)$ and $U(0, 1/4)$ in turn, at the operating lift-gas rate;
- **lift sweep:** $w_{lg} = w_{lg,\max}$, then $x\thinspace w_{lg,\max}$ with $x$ drawn the same way, at the operating choke.

From D-48 (ruled 2026-11-06). Sources: Golan and Whitson (1991), §5.3 (the lift performance curve).

### SMP-100 · Drift between dates

Each operating sample's GOR and water are their trends' values times $U(0.95, 1.05)$ each; a test's five samples share one such pair, drawn at its first sample. An oil's WLR is capped at 0.98, and a gas well's WLR follows from its WGR and GOR. $p_r$ follows its trend without noise. Each slot's separator pressure is SMP-95's around $p_{s,\text{nom}}(t)$.

From D-40 (ruled 2026-11-06). Sources: the plan's Goals.

### SMP-101 · Life scan and the end of a life

Before its slots are drawn, the well's best rate (SMP-103) is solved at the 21 dates $t_i = i\thinspace T_\text{life}/20$, with the lift that the well has at each (SMP-90).

- A point fails where its status (SMP-103) is `below_floor`, `no_operating_point` or `water_cut`. A point with the status `solver_failure` is skipped.
- At a point $i > 0$ where the well is natural and its status is `below_floor` or `no_operating_point`, rule 3 of SMP-90 starts lift at $t_{i-1}$ where that is earlier than its start so far, and the point is solved again with lift.
- The life ends at $T_\text{end}$, the last point that passes before the first that fails, or at $T_\text{life}$ if none fails.
- A well whose first point fails is dropped as `not_producible`, or as `best_rate_solver_failure` (SMP-103); one whose life ends at $T_\text{end} = 0$ is dropped as `short_life`.

From D-50 and D-84 (ruled 2026-11-06) and D-49. Sources: D-34's floor; [WP §11]. The scan's step of 5% of the life is the ruled one; a finer scan is the named fix if dead dates (SMP-106) stay above their target.

## v2: producibility

### SMP-102 · Economic floor

The floor in oil equivalent is 1.6 Sm³/d (10 BOE/d) for platform and onshore wells and 20 Sm³/d for subsea wells. A well whose best rate is below half its floor is not produced, and neither is an oil well whose WLR is above 0.98.

From D-34 and D-50 (ruled 2026-11-06, checkpoint D3). Sources: [WP §11]; EIA (2025), p. 10 and Table B32; [own analysis of BOEM's OGOR-A production and ANP's production by well] for the subsea floor.

### SMP-103 · Best rate and the producibility filter

A well's best rate at a state is the largest oil equivalent among its operating points at a fully open choke ($u = 1$), $p_s = p_{s,\text{nom}}$ and the state's $p_r$, GOR and water without drift, solved at $w_{lg} \in \lbrace 0,\ 1/2,\ 1\rbrace\thinspace w_{lg,\max}$ for a lifted well and at $w_{lg} = 0$ for a natural one. Its status is, in this order: `water_cut` for an oil whose WLR is above 0.98; `pass` if it has an operating point and the best rate is at least half the floor; `below_floor` if it has one; `solver_failure` if no solve gave an operating point and one was a solver failure (SMP-104); else `no_operating_point`.

- **`sol-2`:** at the well's nominal state (SMP-92), before its slots. A natural oil well whose status is `below_floor` or `no_operating_point` takes its life's lift design (SMP-90), with `lift_start_rule` `no_natural_flow`, and is solved again. A well that then does not pass is dropped: as `best_rate_solver_failure` for the status `solver_failure`, else as `not_producible`.
- **`nsol-2`:** at the start, the life scan's first point, with the same outcomes, and then along the life by SMP-101.

From D-50 (ruled 2026-11-06, checkpoint D3), D-55, D-101 (checkpoint F2) and D-84. Sources: D-34; Golan and Whitson (1991), §5.3, whose lift performance curve is flat near its optimum.

## v2: generation

### SMP-104 · Attempts and outcomes

An attempt is one draw of a slot's inputs, solved once by the Rust core. Its outcome is one of:

- `operating_point`: the root set's operating point, its stable root (SOL-4, SOL-6), whose row is valid; the slot takes it;
- `no_operating_point`, the well cannot flow at these inputs (SOL-5): $p_r \le p_s$, without a solve (reason `p_r_le_p_s`); or a search that found no root and rejected no sign change, whatever its non-finite scan points (`no_root`);
- `solver_failure`: an exception, a refusal of the inputs by the model included (`exception`); a row with a non-finite value, a negative liquid or gas rate, or a mass fraction outside $[0, 1]$ (`invalid_row`); a search with a rejected sign change and no root (`incomplete_scan`); or roots with no stable one (`unstable_only`), which a continuous $R(p_0)$ cannot give.

No other backend retries a solver failure: it is redrawn and logged as a missing operating point is, and counted apart.

From D-53 as revised by D-102 (ruled 2026-11-06, checkpoint H2), D-54 and D-82 (checkpoint H1: the CasADi retry dropped). Sources: `specs/model/solution.md` (SOL-2, SOL-3, SOL-5, SOL-7); the Rust core's search counts (`rust/src/march.rs`).

### SMP-105 · Slots and redraws

Each well has 200 slots, $j$ = 0 to 199, and each slot ends with an operating point. An attempt without one is redrawn as attempt $a + 1$ of the same slot.

- **Passes.** Pass 0 solves attempt 0 of every slot, and pass $a$ attempt $a$ of each slot still without an operating point, in the order of $j$. A slot's draws do not depend on the other slots' outcomes.
- **`sol-2`:** a redraw draws the slot's whole draw again (SMP-93 to SMP-96), so the samples follow the drawn distribution conditioned on an operating point.
- **`nsol-2`:** a redraw keeps the slot's date and everything that follows from it: the reservoir's state with the date's drift, $h$, $\varepsilon$, the inflow's coefficients, $p_{s,\text{nom}}$ and whether lift is installed. It draws the separator pressure again (SMP-95), and both controls from the slot's planned values (in a test, the sweep's point): each, with probability 0.6, a step of $U(-0.2, 0.2)$ times its range's width, reflected into its range (the lift gas's $[0.1, 1]\thinspace w_{lg,\max}$), else a new draw as in SMP-98; the lift gas only where lift is installed. The redrawn controls hold for that slot only.

From D-52 (ruled 2026-11-02), D-80 and D-83 (ruled 2026-11-06, with R4's step kept for redraws). Sources: `sampling-draft.md`, Slots and outcomes and Redraws in `nsol-2`.

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

### SMP-107 · Starts

The Rust core takes no start. Where a run solves with the CasADi backend, as a comparison, its extra start (`x_guess`) is the operating point of the same well nearest in $(u,\ w_{lg}/w_{lg,\max},\ p_s/p_{s,\text{nom}})$ among its best-rate solves and the slots solved so far (`sol-2`), or the one nearest in time to the slot's date (`nsol-2`). A start only adds roots to the set.

From D-56 (ruled 2026-11-06). Sources: SMP-28, SMP-29; `specs/model/solution.md`.

### SMP-108 · Generation logs

Each dataset has two logs, published as a companion file next to it, not in it:

- **the well log** (`drawn_wells`): every drawn well, accepted or not, with its draw index, every drawn input and label, its outcome, its best rate with its lift-gas rate and the outcomes of its solves, the counts of its attempts by outcome and reason, and in `nsol-2` $T_\text{life}$, the life scan's best rates and outcomes, $T_\text{end}$, and the lift's start and rule;
- **the redraw log** (`redraws`): every attempt that was redrawn and every attempt of a dropped well, with its inputs, its outcome and reason, and the sizes of its root set.

With the samples, the redraw log holds every draw of the accepted wells, so the shift that the redraws cause can be measured input by input and split into `no_operating_point` and `solver_failure`. The search counts and times of the attempts that gave a sample at once stay with the run. The columns are the schema's (`datasets-draft.md`, `redraws` and `drawn_wells`; R11).

From D-85 (ruled 2026-11-06, checkpoint I1) and D-52. Sources: `datasets-draft.md`, Section 2.

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

### SMP-110 · The datasets' wells and tables

- **Wells.** The generator draws wells $k$ = 0, 1, 2, … and keeps the first 10,000 it accepts. `ID` numbers them from 0 in draw order, and `draw_index` keeps $k$.
- **Split.** 20% of the wells are `test` and 80% `train`, by well, stratified by `fluid_type` × `well_type` × `geometry_class` × `coverage_tail`: each well, in `ID` order, draws one number from the split's generator; within each stratum the wells are ranked by it, and ranks 5, 10, 15, … are `test`.
- **Tables.** Each dataset holds `wells` (every input that is constant over a well's samples, and every label), `samples` (each sample's inputs that vary and its operating point: SMP-30's features with SLIP-16's five regimes, and the v2 columns), `geometry` (each well's survey stations) and `pvt` (the table fluids' tables), with SMP-108's logs beside it. A row of `samples`, its well's rows and, for a table fluid, its table's rows carry every input of its `WellProperties` and `BoundaryConditions`. The columns are `datasets-draft.md`'s, Section 2, which R11 writes into `docs/datasets.md`.
- **Records.** Each dataset records its seed, the configuration, $N$, the backend, the limits and the floor, the code version (`datasets.io.code_version`, a full commit with a clean working tree), the lock file's hash and the platform. The generator works in resumable chunks of 100 draw indices.
- **Re-solving.** A sample re-solved from its rows gives the same features with the recorded code version and backend on the generator's platform, and elsewhere features within the verifier's `tol_x` applied to them (`datasets-draft.md`, Section 3).

From D-86 to D-93 (ruled 2026-11-06; the logs beside the dataset, I1; the re-solve tolerance, I2). Sources: `specs/goals.md`, Compatibility; `datasets-draft.md`, Sections 1 to 3 and 6.

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

### SMP-112 · Table fluids and their source

Volatile oils and gas condensates (`pvt_model` `table`) take a PVT table set of their own; black oils and wet and dry gases take the black oil (`pvt_model` `black_oil`). NeqSim 3.4.0 makes the tables (Peng–Robinson with the Peneloux volume correction, its characterisation and its default interaction parameters, untuned) in a process environment of the generator's own, outside ManyWells' dependencies; the dataset carries the tables, so that a user re-solves without NeqSim. The `composition` block draws the synthetic composition's parameters first, then the source: with probability 0.2 one of the class's three published examples with equal shares (SMP-114; `fluid_source` its key), else the synthetic composition (SMP-113; `synthetic`).

From D-36, D-66 and D-68 (ruled 2026-11-06). Sources: `fluids-draft.md`, Which wells take a table; `specs/features/018-pvt-tables.md`.

### SMP-113 · Synthetic composition

- **Components.** N₂, CO₂, C₁, C₂, C₃, i-C₄, n-C₄, i-C₅, n-C₅ and C₆ (as n-hexane), and the heptanes-plus as five fractions.
- **Light ends.** A mixture of the class's three published examples (SMP-114), normalised over N₂ to C₆, with weights $w \sim \text{Dirichlet}(1, 1, 1)$; then C₂ to C₆ are scaled by one factor $f_\text{int}$, and the light ends renormalised.
- **Heptanes-plus.** The mole fraction before the GOR is set, $z_{7+} \sim U(0.04, 0.10)$ for condensates and $U(0.10, 0.22)$ for volatile oils; the molar mass $M_{7+} \sim U(140, 200)$ and $U(190, 240)$ g/mol; split into five fractions at 135, 205, 275 and 415 g/mol by Whitson's gamma distribution with the shape $\alpha \sim U(0.5, 2.5)$; each fraction's specific gravity by Søreide, $\gamma_i = 0.2855 + C_f\thinspace(M_i - 66)^{0.13}$.
- **Separation.** One stage at standard conditions (1.01325 bar, 288.15 K), so that every GOR of the well's table set lies on one tie line. The reservoir fluid is the separator gas and the stock-tank oil recombined at the GOR.
- **Matching.** Newton's method on $(C_f, \ln f_\text{int})$ matches the stock-tank oil's API gravity and the separator gas's gravity to 0.5 °API and 0.005, and the recombination the GOR exactly. The targets are the drawn, field-reported values (SMP-67, SMP-68) shifted by the fluid's own offsets: the matched stream is flashed through the well's separator at $(p_\text{sep}, T_\text{sep})$, its liquid through a stock tank at standard conditions, and through one stage at standard conditions; the single-stage targets are moved by the difference and matched again, in two passes. The offsets are recorded with the fluid.

From D-63 (ruled 2026-11-06, checkpoint J2: per-fluid offsets). Sources: Whitson and Brulé (2000), Table 2.1, §5.3.2, Eqs. 5.13 to 5.15 and 5.44; Pedersen, Christensen and Shaikh (2015), Tables 1.3, 1.4 and 3.8; McCain (1990), pp. 151–156 [RF §8].

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

### SMP-115 · Saturation pressure of a table fluid

The fluid's saturation pressure at $T_r$ is its flash's: TP flashes on steps of 10 bar up to $1.3\thinspace p_i$, with NeqSim's check for more phases on, then bisection to $10^{-3}$ bar on the edge of the two-phase region. Its kind, from the incipient phase, is `saturation_kind`, `bubble` or `dew`; a fluid with two phases nowhere from 1 bar to $1.3\thinspace p_i$ at $T_r$ takes $p_\text{sat}$ = 1 bar and `none`.

- Where $p_\text{sat}(T_r) \le p_i$, `saturation_rule` is `matched`.
- Where $p_\text{sat}(T_r) > p_i$, the fluid could not be one phase at $p_i$: its GOR moves along its tie line until $p_\text{sat}(T_r) = p_i$, by bisection on $\ln R$ within a factor of 4 (`gor_moved`); where no GOR on the line does it, the reservoir fluid is the class's phase in equilibrium at $(p_i, T_r)$, the vapour for a condensate and the liquid for an oil, separated again (`equilibrium_phase`).

The table's $p_\text{sat}$ and initial GOR replace SMP-68's in every later block; SMP-68's undersaturation does not apply. A fluid keeps its class (SMP-55) whatever its kind.

From D-65 (ruled 2026-11-06) and L2 (ruled 2026-11-09). Sources: McCain (1990), pp. 151–156; Pedersen et al. (2015), §1.4; Elsharkawy (2002), Table 3.

### SMP-116 · The table set

- **Build order.** The table wells and their draws (ManyWells); each fluid, its saturation pressure and SMP-115's rule (NeqSim); each well's GOR axis from its life with the table's values in it (ManyWells); the tables at the axis's GORs (NeqSim).
- **Grid,** the well's own: temperatures every 10 K from $10\thinspace\lfloor(T_s - 15\ \text{K})/10\rfloor$ to $10\thinspace\lceil(T_r + 1\ \text{K})/10\rceil$; pressures 1, 2.5, 5 and 7.5 bar, then every 10 bar to $10\thinspace\lceil 1.05\thinspace p_i/10\rceil + 10$ bar, with each temperature's saturation pressure a node.
- **GOR axis** (PVT-TAB-7): GORs at most a factor 1.25 apart over every GOR the well's samples reach, their drift included: `sol-2` over $p_r$ within ±2% of its date's, `nsol-2` over the life's trend from $p_r(0)$ to $p_r(T_\text{life})$, each times the ends of $U(0.95, 1.05)$.
- **Columns.** $R_{so}$, $B_o$, $Z$, $\mu_o$, $\mu_g$ and $\sigma_{go}$ by PVT-TAB-6 from the flash, with its liquid trace $\delta = 10^{-4}$ where the stream is all vapour; where the stream is all liquid, the gas columns are the incipient vapour's.
- **Repair.** At each GOR, the pressures at which $p/Z$ does not rise strictly with $p$ at some temperature, the nodes off the longest run along which it rises (of the longest, the one ending highest), are dropped at every temperature of that GOR's table, so that the grid stays rectangular; the dropped pressures are recorded. A table that would lose more than 5% of its pressures fails.
- **Failure.** A build step that fails, a table that fails the repair, or a table that PVT-TAB-1's checks refuse makes the well `table_failed` (SMP-106); it is not solved.

From D-67 and D-68 (ruled 2026-11-06), D-64 (checkpoint K1) and D-106 (ruled 2026-11-09, L1). Sources: `specs/features/018-pvt-tables.md`; `specs/model/pvt/table.md`; `fluids-draft.md`, The table.

### SMP-117 · Forchheimer coefficients fitted to the pseudo-pressure

At each reservoir state $(p_r, R_{go})$, a gas or condensate well's `Forchheimer(a, b)` (INF-10) is fitted to its inflow in the pseudo-pressure,

$$(\mu_g Z)_ r\thinspace\big(m(p_r) - m(p_{wf})\big) = a\thinspace q + b\thinspace q^2, \qquad m(p) = 2\int_1^p \frac{p'}{\mu_g Z}\thinspace dp',$$

with $a$ and $b$ SMP-73's at the state, from $\mu_g Z$ of the reservoir gas at $T_r$ and the start's geometry factors and $D$. The reservoir gas is the black-oil gas at the wellstream's density, $(\rho_{g,\text{sc}} + \rho_o/R_{go})/c$, for a wet or dry gas, and the table's gas at $R_{go}$ for a condensate. $m$ is integrated by the trapezoidal rule on 200 pressures from 1 bar to $p_r$.

- **The fit.** Least squares of $(p_r^2 - p_{wf}^2)/q$ against $q$ at $p_{wf}/p_r$ = 0.95, 0.9, 0.8, 0.6, 0.4 and 0.2 and at the absolute open flow ($p_{wf}$ = 1.01325 bar) give the pressure-squared form's $a_\text{ws}$ and $b_\text{ws}$. Where the slope is negative, $b_\text{ws} = 0$ and $a_\text{ws}$ is the mean of $(p_r^2 - p_{wf}^2)/q$ over the points: the form cannot follow the pseudo-pressure's curvature there, a known limit at high pressure.
- **To the separator gas:** $a_q = c\thinspace a_\text{ws}$ and $b_q = c^2 b_\text{ws}$, with $c = 1 + C_{og}/R_{go}$, $C_{og} = 0.178108 \times 133\thinspace 000\thinspace\gamma_o/M_o$ Sm³/Sm³, $\gamma_o$ as in SMP-82 and $M_o = 6084/(\text{API} - 5.9)$ (Whitson and Brulé 2000, Eqs. 3.73b and 3.73c), for the black oil; and $c = 1 + \rho_o/(R_{go}\rho_{g,\text{sc}})$ for a table, whose $Z$ has the separator gas's gas constant (PVT-TAB-4). Then to the model's units by `Forchheimer.from_gas_rate(a_q, b_q, rho_g)`.

From D-61 (ruled 2026-11-06; $b = 0$ kept, L3 of 2026-11-09) and D-60. Sources: Golan and Whitson (1991), Eq. 2.57; Whitson and Brulé (2000), Eqs. 3.73b and 3.73c; `specs/features/022-gas-rate-inflow.md`, Units; `validity.md`, note 12.

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

The v2 rows (SMP-50 to SMP-117) are spec-only until R10's code carries their `# spec:` tags: the code column names where the code is planned, and the planned checks are in brackets. Each row becomes `property:` when its test exists.

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
| SMP-51 | — | — (`sampling/v2/draws.py` `draw_well`) | property: tests/test_sampling_v2_draws.py (the blocks' generators in their order, a block drawn from another seed moves no other block's numbers, a stage override changes only what depends on the stage. Not implemented yet: every number of a block drawn whatever its branch, since the port keeps the prototype's draws) |
| SMP-52 | — | — (`sampling/v2/draws.py`, the labels that the well draws set; the tables', the life's and the generation's labels: planned, R10) | property: tests/test_sampling_v2_draws.py (every label of the well draws within its values) |
| SMP-53 | — | — (`sampling/v2/draws.py` `_labels`, and each block's tail form) | property: tests/test_sampling_v2_draws.py (the tail's share, the tail wells' bounds, shares and marginals) |
| SMP-54 | — | — (`sampling/v2/draws.py` `_labels`) | property: tests/test_sampling_v2_draws.py (the shares, core and tail) |
| SMP-55 | — | — (`sampling/v2/draws.py` `_labels`) | property: tests/test_sampling_v2_draws.py (the shares, core and tail, each class's range of the initial GOR) |
| SMP-56 | — | — (`sampling/v2/draws.py` `_labels`) | property: tests/test_sampling_v2_draws.py (the shares, core and tail) |
| SMP-57 | — | — (`sampling/v2/draws.py` `_depth`) | property: tests/test_sampling_v2_draws.py (the bounds and the marginals, core and tail) |
| SMP-58 | — | — (`sampling/v2/draws.py` `_depth`) | property: tests/test_sampling_v2_draws.py (the bounds, the share of deep-water platforms) |
| SMP-59 | — | — (`sampling/v2/draws.py` `_geometry`) | property: tests/test_sampling_v2_draws.py (the shares of the classes and of their families, the class's inclination) |
| SMP-60 | — | — (`sampling/v2/survey.py` `survey`; `sampling/v2/draws.py` `_geometry`) | property: tests/test_sampling_v2_draws.py (each family's sections, ranges, kickoff and least hold, MD at most 9342 m) |
| SMP-61 | — | — (`sampling/v2/draws.py` `_geometry`) | property: tests/test_sampling_v2_draws.py (every cell within 75° of vertical) |
| SMP-62 | — | — (`sampling/v2/draws.py` `_water`) | property: tests/test_sampling_v2_draws.py (the bounds, the density and the marginals) |
| SMP-63 | — | — (`sampling/v2/draws.py` `_pressure`) | property: tests/test_sampling_v2_draws.py (the bounds, the truncations and the shares of underpressure. An open finding, kept as a strict expected failure: $p_i$ exceeds 950 bar where the hydrostatic pressure alone does) |
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
| SMP-78 | — | — (`sampling/v2/draws.py` `_lift`) | property: tests/test_sampling_v2_draws.py (a well that needs lift is lifted, no core gas well is lifted at its start) |
| SMP-79 | — | — (`sampling/v2/draws.py` `_lift`) | property: tests/test_sampling_v2_draws.py (the valve's depth within its bounds, its MD on the survey, the lift gas's temperature) |
| SMP-80 | — | — (`sampling/v2/draws.py` `_lift`) | property: tests/test_sampling_v2_draws.py ($w_{lg,\max}$ within 0.1 to 3.5 kg/s) |
| SMP-81 | — | — (`sampling/v2/draws.py` `_heat`) | property: tests/test_sampling_v2_draws.py (the bounds, the tail's marginal) |
| SMP-82 | — | — (`sampling/v2/draws.py` `_heat`) | property: tests/test_sampling_v2_draws.py ($c_{po}$ and $c_{pw}$ by their formulas, $c_{pg}$ within the table's range) |
| SMP-83 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the life's trends, and the two datasets' states by stage) |
| SMP-84 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the life's trends, and the two datasets' states by stage) |
| SMP-85 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the life's trends, and the two datasets' states by stage) |
| SMP-86 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the life's trends, and the two datasets' states by stage) |
| SMP-87 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the life's trends, and the two datasets' states by stage) |
| SMP-88 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the life's trends, and the two datasets' states by stage) |
| SMP-89 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the life's trends, and the two datasets' states by stage) |
| SMP-90 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the lift's start by rule, on a pilot) |
| SMP-91 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the life's trends, and the two datasets' states by stage) |
| SMP-92 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the draws' bounds, shares and marginals, v2: checks) |
| SMP-93 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the draws' bounds, shares and marginals, v2: checks) |
| SMP-94 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the draws' bounds, shares and marginals, v2: checks) |
| SMP-95 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the draws' bounds, shares and marginals, v2: checks) |
| SMP-96 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the draws' bounds, shares and marginals, v2: checks) |
| SMP-97 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the draws' bounds, shares and marginals, v2: checks) |
| SMP-98 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the draws' bounds, shares and marginals, v2: checks) |
| SMP-99 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the draws' bounds, shares and marginals, v2: checks) |
| SMP-100 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the draws' bounds, shares and marginals, v2: checks) |
| SMP-101 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: a pilot well's life scan and end) |
| SMP-102 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the best rate against the floor, on a pilot) |
| SMP-103 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the best rate against the floor, on a pilot) |
| SMP-104 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: each outcome and reason, on solved cases) |
| SMP-105 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: a pilot well's redraws, slot by slot) |
| SMP-106 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: the drops against their targets, on R11's pilot) |
| SMP-107 | — | — (planned: `manywells.sampling`, R10) | spec-only: until R10 implements it (planned property: a comparison run's starts) |
| SMP-108 | — | — (planned: `manywells.datasets`, R10) | spec-only: until R10 implements it (planned property: the logs hold every draw of a pilot) |
| SMP-109 | — | — (`sampling/v2/draws.py` `draw_well`, the blocks' generators; the life's, the samples' and the split's: planned, R10) | property: tests/test_sampling_v2_draws.py (a rerun gives the same draws, whatever was drawn before and in a process of its own, every key holds `v2`) |
| SMP-110 | — | — (planned: `manywells.datasets`, R10) | spec-only: until R10 implements it (planned property: the counts, the split, and a re-solved row) |
| SMP-111 | — | — (`sampling/v2/mapping.py` `WellBuilder`) | property: tests/test_sampling_v2_mapping.py (every drawn well at its nominal state passes `configurations.check`, with each input from its draws) |
| SMP-112 | — | — (planned: the generator's NeqSim step, R10) | spec-only: until R10 implements it (planned property: the table checks, v2: checks) |
| SMP-113 | — | — (planned: the generator's NeqSim step, R10) | spec-only: until R10 implements it (planned property: the table checks, v2: checks) |
| SMP-114 | — | — (planned: the generator's NeqSim step, R10) | spec-only: until R10 implements it (planned property: the table checks, v2: checks) |
| SMP-115 | — | — (planned: the generator's NeqSim step, R10) | spec-only: until R10 implements it (planned property: the table checks, v2: checks) |
| SMP-116 | — | — (planned: the generator's NeqSim step, R10) | spec-only: until R10 implements it (planned property: the table checks, v2: checks) |
| SMP-117 | — | — (`sampling/v2/inflow.py` `gas_inflow`, `fit_pressure_squared`) | property: tests/test_sampling_v2_mapping.py (the fit reproduces a pressure-squared law, $b = 0$ where the slope is negative, the fit against the pseudo-pressure) |
