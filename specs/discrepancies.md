# Discrepancies between the paper and v1.0.0

*Step 4 of `plans/manywells-v2-plan.md`. Owner: Bjarne Grimstad. Status: ruled by Bjarne on 2026-09-30, who accepted every proposed ruling; D-9 checked against the source the same day.*

Where the paper (Grimstad et al., Geoenergy Science and Engineering 257, 2026) and v1.0.0's code (`manywells/*.py` and `scripts/data_generation/` at the `v1.0.0` tag) disagree, or where the code has something the paper does not state. For each item, Bjarne ruled that the paper is right, the code is right, or both change. The `v1.0.0` configuration in `specs/model/` follows the rulings; a new ruling that changes it changes the spec in the same PR. Items marked *corrigendum* are in `docs/corrigendum.md`. Line numbers are at `v1.0.0`.

Model changes made on `develop` since v1.0.0 are not discrepancies; they are listed in `plans/develop_model_changes.md`.

## Summary

| Item | Spec IDs | Kind | Ruling |
|---|---|---|---|
| D-1 | SLIP-6 | paper typo | code is right; corrigendum |
| D-2 | PVT-MIX-2 | paper typo | code is right; corrigendum |
| D-3 | SLIP-3 | paper typo | code is right; already in the corrigendum |
| D-4 | DISC-1, nomenclature | paper wording | code is right; corrigendum |
| D-5 | SLIP-7 | constants not in the paper | code is right |
| D-6 | CHK-3, SMO-1 | smoothing constant not in the paper | code is right |
| D-7 | CHK-4 | constant not in the paper | code is right |
| D-8 | PVT-MIX-5 | choice not in the paper | code is right for `v1.0.0` |
| D-9 | PVT-OIL-3 | formula not in the paper | code is right: it matches the source exactly |
| D-10 | SLIP-8 | rule not in the paper | code is right |
| D-11 | CHK-12, CHK-3 | smoothing | code is right |
| D-12 | FRIC-1 | equivalent forms | both are right |
| D-13 | CHK-2, CHK-5 | equivalent forms; code docstring | both are right; docstring fixed on `develop` |
| D-14 | BAL-9, DISC-6 | equivalent forms | both are right |
| D-15 | CHK-6, INF-2, INF-3 | options not in the paper | code is right |
| D-16 | INF-4 | validity range | code is right |
| D-17 | SOL-1 | bounds not in the paper | code is right, with the temperature bounds as solver safeguards |
| D-18 | — (solver) | initial guess | not part of the model |
| D-19 | CHK-11 | undefined residual | no flow where $\Delta p \le 0$; any row with the same sign |
| D-20 | SOL-6 | undefined operating point | the stable root with the lowest $p_0$ |
| D-21 | CHK-4, INF-3, PVT-MIX-2 | code defects | code is right as a model; fix the defects on `develop` |
| D-22 | PVT-MIX-4, CHK-4, SLIP-6, THM-3 | paper and code agree | observations for v2, no ruling needed |
| D-23 | CHK-5 | paper notation | code is right; corrigendum |
| D-24 | CHK-1 | paper typo | code is right; corrigendum |
| S-1 to S-12 | `specs/sampling.md` | sampling and generation | code is right; S-8 and S-11 are dataset errata; S-10 and S-12 are corrigendum entries |
| C-1 | — | closed loop, out of scope | the error in (52) is a corrigendum entry; the rest is recorded only |

## Model

### D-1 · Classifier features on α are scaled by 2

- **Paper.** (A.8): $\tanh(\alpha_g - 0.25)$ and $\tanh(\alpha_g - 0.7)$.
- **v1.0.0.** $\tanh(2(\alpha - 0.25))$ and $\tanh(2(\alpha - 0.7))$ (`slip.py:90,92`), "multiplied by 2 to increase sensitivity". The docstring lists the fitted weights next to features without the factor, so the code does not record whether $A$ and $b$ were fitted with it.
- **Effect.** Steeper regime transitions in $\alpha$. Every published dataset was generated with the factor.
- **Ruling.** The code is right: it is v1.0.0's model. Corrigendum for (A.8). `plans/solver_description.md` §9 found this first.

### D-2 · Equation (29) gives the oil volume fraction

- **Paper.** (29): $\alpha_{w,l} = 1/\big(1 + (\rho_o/\rho_w)(f_w/f_o)\big)$, used as the water fraction in (28) and (31).
- **v1.0.0.** `pvt.liquid_mix(oil, water, f_o/(f_o + f_w))` computes $1/\big(1 + (\rho_o/\rho_w)(f_w/f_o)\big)$ and uses it as the *oil* fraction (`pvt.py:114-116`), so its $\rho_l$ and $c_{pl}$ are the correct volume-weighted averages. The comment there calls it the water fraction.
- **Effect.** None on the data; (28) with the printed (29) would weight the densities the wrong way round (918.5 against 930.6 kg/m³ for $\rho_o = 850$ and equal masses).
- **Ruling.** The code is right. Corrigendum for (29): $\alpha_{w,l} = 1/\big(1 + (\rho_w/\rho_o)(f_o/f_w)\big)$. Fix the comment on `develop`.

### D-3 · Rise velocities swapped in (A.10)

- **Paper.** (A.10) weights the Harmathy velocity by $p_\text{slug-churn}$ and the Taylor velocity by $p_\text{bubbly}$.
- **v1.0.0.** The other way round (`slip.py:163-170`).
- **Ruling.** The code is right. Already in `docs/corrigendum.md`; decided.

### D-4 · Wording of $z$ and of the grid

- **Paper.** Table 1 calls $z$ "vertical depth", although it increases upwards from the bottomhole. §3.1 discretizes "into $N + 1$ cells", each of length $L/N$.
- **v1.0.0.** The docstring repeats "(n + 1) cells of length L / n" (`simulator.py:101`); the code has $N$ cells and $N + 1$ grid points.
- **Ruling.** The spec says $N$ cells and $N + 1$ grid points (DISC-1) and defines $z$ as the distance from the bottomhole. Corrigendum, although no equation is wrong.

### D-5 · The classifier's weights and ordering

- **Paper.** (A.7) with "learned parameters" $A$ and $b$, not listed; features ordered $(\alpha - 0.25, \dots)$ and outputs (bubbly, slug/churn, annular).
- **v1.0.0.** The twelve weights and three biases in `slip.py:95-97`, with the features and outputs in the reverse order.
- **Ruling.** The code is right. SLIP-7 records the values, which are part of the model.

### D-6 · Smoothing constant of the critical pressure

- **Paper.** (14) with an exact max, smoothed by (B.1) with "$\epsilon \ll 1$".
- **v1.0.0.** $\epsilon = 10^{-6}$ on pressures in bar (`ca_functions.py:15`, `choke.py:110`), so $\epsilon$ is $10^{-6}$ bar² and the smooth max exceeds the max by up to $5\cdot10^{-4}$ bar.
- **Ruling.** The code is right. CHK-3 and SMO-1 record $\epsilon$ and its unit.

### D-7 · Heat capacity ratio in the critical pressure ratio

- **Paper.** (13) with $\gamma \approx 1.3$ for methane at 20 °C, giving $r_c \approx 0.544$.
- **v1.0.0.** $\gamma = 1.307$, $r_c = 0.5445$ (`choke.py:83`), for every well.
- **Ruling.** The code is right. CHK-4 records the value.

### D-8 · Density at which the surface tension is evaluated

- **Paper.** App. A.1: $\sigma$ "is determined by the correlation found in Abdul-Majeed and Abu Al-Soof (2000)", a dead-oil correlation in the API gravity and temperature, without saying which density gives the API gravity.
- **v1.0.0.** The liquid's density, oil and water mixed, and the local temperature: `dead_oil_surface_tension(rho_l, T)` with the state $\rho_l$ (`slip.py:85,135`). For a water-rich liquid this is the dead-oil correlation at an API gravity near 10.
- **Ruling.** The code is right for `v1.0.0`, which the datasets need. `develop` evaluates it at the oil density instead (`plans/develop_model_changes.md`); which rule the `develop` default uses is a Step 7 question.

### D-9 · Coefficients of the dead-oil surface tension

- **Paper.** Cites the correlation without the formula.
- **v1.0.0.** $10^{-3}(1.11591 - 0.00305 T_C)(38.085 - 0.259\thinspace\text{API})$ (`pvt.py:176`).
- **Source.** Abdul-Majeed and Abu Al-Soof (2000), Eqs. (1)–(3): $\sigma_{do} = A\thinspace(38.085 - 0.259\thinspace\text{API})$ dyn/cm with $A = 1.11591 - 0.00305 T$ and $T$ in °C, fitted at 15.6, 37.8 and 54.4 °C and API 15 to 50. An earlier draft of this item gave a form in °F from memory; the source has no such form.
- **Ruling.** The code is right: it matches the source exactly. PVT-OIL-3 records the source's data range; ManyWells evaluates the correlation outside it, up to 150 °C.

### D-10 · Regime label and ties

- **Paper.** Table 3 has the features `FRBH` and `FRWH`; the rule is not stated.
- **v1.0.0.** The most probable regime, with ties going to bubbly (`slip.py:183-194`).
- **Ruling.** The code is right. SLIP-8 records the rule.

### D-11 · The choked flag is exact, the choke row is smooth

- **Paper.** (14): the flow is choked when $p_s \le r_c p_u$.
- **v1.0.0.** The `CHOKED` feature uses the exact comparison (`choke.py:129`), while the choke row uses the smooth max. Within about $5\cdot10^{-4}$ bar of the switch the two can disagree.
- **Ruling.** The code is right. CHK-12 records the exact rule; the verifier's `choke_band` allows for the difference.

### D-12 · Friction with $v_m^2$

- **Paper.** (5): $\rho_m v_m \lvert v_m\rvert$.
- **v1.0.0.** $\rho_m v_m^2$ (`simulator.py:239`).
- **Ruling.** Both are right: they agree for $v_m \ge 0$, and every admissible state has $v_m > 0$. FRIC-1 states the paper's form.

### D-13 · Choke density and multiplier

- **Paper.** (11)–(12) with the momentum density $\rho_e$.
- **v1.0.0.** $\rho_l$ with Simpson's multiplier $\Phi$ inside the square root (`choke.py:112,184`), which equals $\rho_e = \rho_l/\Phi$ (paper, footnote 1). The class docstring has $\Phi$ outside the square root (`choke.py:28,99`).
- **Ruling.** Both are right. CHK-2 and CHK-5 give both forms. The docstring is fixed on `develop` in this step (`plans/improvements.md` §1.5).

### D-14 · Seven unknowns per point, not eight

- **Paper.** §3.3: $8(N+1)$ unknowns and rows, keeping $\alpha_l$ and (7).
- **v1.0.0.** $7(N+1)$: $\alpha_l = 1 - \alpha$ is substituted, and (7) has no row.
- **Ruling.** Both are right. BAL-9 and DISC-6 record the substitution.

### D-15 · Options that the paper does not describe

- **Paper.** Vogel inflow (10) and the Simpson choke (11)–(12) only.
- **v1.0.0.** Also a Bernoulli choke with the mixture density and no multiplier, a productivity-index inflow and a fixed-rate inflow (`choke.py:132`, `inflow.py:55,130`). The Bernoulli choke with $K_c = 0.1A$ and the productivity index with $k_l = 0.5$, $f_g = 0.1379$ are `WellProperties`' defaults. No dataset uses them; the verifier's case set has Bernoulli and productivity-index cases.
- **Ruling.** The code is right. CHK-6, INF-2 and INF-3 record them as well options.

### D-16 · Range of the gas mass fraction

- **Paper.** (10): $f_g \in [0, 1)$.
- **v1.0.0.** Asserts $0 < f_g < 1$ (`inflow.py:76,113`).
- **Ruling.** The code is right: the spec takes $(0, 1)$. No dataset has $f_g = 0$. No corrigendum; the model itself would be defined at $f_g = 0$.

### D-17 · Bounds on the unknowns

- **Paper.** Does not mention bounds.
- **v1.0.0.** Ipopt bounds $p \in [p_s, p_r]$, $\alpha \in [0, 1]$, $T \in [T_s, T_r + 1]$ and every other unknown $\ge 0$ (`simulator.py:465-475`).
- **Ruling.** The code is right. SOL-1 takes the pressure and volume-fraction bounds as the admissible set, with strict positivity of velocities and densities, as the verifier's Invariants have it. The temperature bounds are solver safeguards: they are never active at a root, so they are not part of the definition.

### D-18 · Initial guess

- **Paper.** §3.3: an approximate problem solved cell by cell, or earlier solutions as starts.
- **v1.0.0.** $p_0 = p_r - 0.05(p_r - p_s)$, $T_0 = T_r$, the bottom point from $\alpha = 0.5$, and a march by Ipopt (`simulator.py:255-404`); the generators' warm starts (`specs/sampling.md`).
- **Ruling.** Not part of the model: which root a solver reaches is not (principle 6). Recorded in `solution.md` as informative, because the verifier records v1.0.0's outcome from its default guess.

### D-19 · Choke row where $p_N \le p_c$

- **Paper.** (11) has no real value when $p_u \le p_{sc}$.
- **v1.0.0.** NaN, the square root of a negative number. The Rust port squares the equation.
- **Ruling.** The choke passes no flow there ($\max(\Delta p, 0)$ under the root), and an implementation may use any row with the same sign, such as the squared form (CHK-11). The root set and the labels are the same under every choice; it matters to solvers near $p_N = p_s$. Not chosen: leaving the row undefined there, or making the squared form canonical.

### D-20 · More than one stable root

- **Paper.** Says the system has "multiple solutions, some which may be unphysical", without a rule.
- **v1.0.0.** Returns whichever root Ipopt reaches.
- **Ruling.** The stable root with the lowest $p_0$, with the case flagged (SOL-6); the verifier checks it. Step 2 found two such cases near the fold (`fold-1505`, `fold-0485`). Not chosen: reporting no unique operating point, or the stable root with the highest $p_0$.

### D-21 · Defects in the code that do not change the model

- `ChokeModel.cpr` is a constructor argument that `__post_init__` always overwrites with CHK-4 (`choke.py:58`; `plans/improvements.md` §1.4). The `v1.0.0` configuration takes $r_c$ from CHK-4 always.
- `FixedFlowRate.__post_init__` checks `w_l_const >= 0` twice and never checks `w_g_const` (`inflow.py:143-144`).
- The comment in `liquid_mix` calls the oil volume fraction the water fraction (D-2).
- **Ruling.** None changes the `v1.0.0` model. Fix them on `develop` any time (`plans/improvements.md` §1.4 for the first).

### D-22 · Observations where the paper and the code agree

No ruling is needed for these; they are questions for v2's model, recorded here because the cross-check found them.

- The liquid heat capacity is a volume-weighted average (PVT-MIX-4); specific heat usually mixes by mass.
- $\gamma = 1.307$ is methane's for every well, whatever its specific gas constant (CHK-4).
- The classifier's features $c_1$ and $c_3$ take velocities in m/s as arguments of $\tanh$, so their sharpness depends on the unit (SLIP-6).
- The lift gas enters at $T_r$, whatever its temperature (THM-3); `develop` adds a lift-gas temperature.

### D-23 · The gas fraction in Simpson's multiplier includes the lift gas

- **Paper.** (12) uses $f_g$ and $f_l$, "the gas and liquid mass fraction", the same symbol as the reservoir inflow's gas fraction in (10), and Table 3's `FGAS` excludes the lift gas.
- **v1.0.0.** $x_g = w_g/w_m$ at the wellhead, where $w_g$ includes the lift gas (`simulator.py:198-207`).
- **Effect.** The two differ for every gas-lifted well.
- **Ruling.** The code is right: the choke passes the whole flow. CHK-5 uses $x_g$. Corrigendum clarifying that (12)'s fraction is that of the flow through the choke, lift gas included.

### D-24 · "Left boundary" for the choke

- **Paper.** §2.3, after (11): "With $u_c$ and $p_s$ given, this equation imposes a condition on the pressure at the left boundary." The choke is at the right boundary, $z = L$, as the paragraph before (11) says.
- **v1.0.0.** The choke row is at point $N$ (`simulator.py:187-212`).
- **Ruling.** A typo for "right boundary". Corrigendum.

## Sampling and data generation

Paper §4–5 against `scripts/data_generation/` at `v1.0.0` (`well.py`, `nonstationary_well.py`, `open_loop_stationary/generate_well_data.py`, `open_loop_nonstationary/generate_open_loop_nonstationary_well_data.py`). `specs/sampling.md` records the code's procedure.

- **S-1.** A well whose gas mass fraction is above 0.99 is discarded (`well.py:128`). Not in the paper.
- **S-2.** The truncation of $p_{s,\text{nom}}$ to [10, 120] bar (34) is done by discarding the whole well draw (`well.py:116`). Since the draws are independent, the distribution is the paper's.
- **S-3.** Gas lift is available with probability 0.5 when $f_g \le 0.2$ (`well.py:168`); the paper says $f_g < 0.2$, and the code's comment says 0.1. The same distribution.
- **S-4.** The nominal reservoir pressure (33) uses $\rho_w = 1012.05$ kg/m³, the mean of seawater (1025) and water (999.1) (`well.py:104`); the paper says "salt water with a low salinity" without a value.
- **S-5.** The per-sample redraws cap $f_g$ at 0.99 and the water–liquid fraction at 1 (`well.py:62,65`); (53)–(54) do not.
- **S-6.** `sol-1`: the first solve at $u = 0.5$ is the initial guess for every sample; samples with $w_m < 0.1$ kg/s are dropped without counting as failures; a well is discarded after $5 n$ attempts or 100 failures, or if its choke positions have a standard deviation below a fifth of U(0.05, 1)'s, more than 80% of its samples are choked, or its QTOT has a coefficient of variation below 0.05. §5.1 states the filters without their thresholds.
- **S-7.** `nsol-1`: warm-up solves at $u$ = 0.1, 0.2, …, 1.0; the well is discarded if its rate at $u = 1$ is below 7 kg/s; each start is the closest earlier solution (`InitGuess`); samples with $w_m < 1$ kg/s are redrawn; a well is discarded after 200 failures or 50 failed solves in a row, with no cap on attempts. The random walk (37) and the decay rate's noise (42) step on every attempt, not every week: an attempt takes $\max(1, i - i')$ steps of (37), where $i'$ is the week of the last accepted sample, so failed and rejected attempts add steps (`nonstationary_well.py:60-72,85-91`; `generate_open_loop_nonstationary_well_data.py:137,147-151,206-210`). None of this is in §5.2.
- **S-8.** The `nsol-1` and `nscl-1` configs store $f_D = 0.05$ for every well, although the generator draws it from U(0.01, 0.08) (found in Step 2, `specs/verification.md`). A data erratum; its corrigendum note is Step 3's.
- **S-9.** The `sol-1` generator seeds NumPy from process ID × time (`generate_well_data.py:210`), the `nsol-1` generator from the process ID alone (`generate_open_loop_nonstationary_well_data.py:285`), so no dataset can be regenerated. Not in the paper. The port seeds every draw (`specs/sampling.md`).

- **S-10.** The initial reservoir pressure of `nsol-1`: §5.2 says it is sampled by (32), with the ±2% draw; §4.4.2 says it is found by (33), the nominal value. The code uses (33) (`nonstationary_well.py:120`).
- **S-11.** An `nsol-1` or `nscl-1` config holds the well's state at its last sample, not its draws, because the generator updates the well in place and saves it at the end (`nonstationary_well.py:85-102`, `generate_open_loop_nonstationary_well_data.py:271`). The draws survive as `ns_bhv.pr_init`, `ps_init` and `init_fractions`. A `sol-1` config holds its draws, with $u = 0.5$ from the first solve. Not a paper matter; it is in the dataset errata of `docs/corrigendum.md`.
- **S-12.** Two typos in §4: before (34), $p_s$ is called "the upstream pressure", although it is downstream of the choke; and (22) reads $f_D = \text{Uniform}(0.01, 0.08)$, with $=$ for $\sim$.

**Ruling** for S-1 to S-7, S-9 and S-10: the code is right, and `specs/sampling.md` records it; corrigendum entries for S-10 (§5.2 should cite (33)) and S-12. S-8 (from Step 3) and S-11 are dataset errata in `docs/corrigendum.md`.

## Closed loop

### C-1 · Differences in the closed-loop generator

Closed loop is out of scope for v2 (`specs/goals.md`). `manywells-nscl-1` is published, though, so the error in (52) is in `docs/corrigendum.md`; the other two are recorded here for its users.

- (52) gives $P(k = 0) = 0.8$; the generator uses 0.7 (`closed_loop_nonstationary/generate_closed_loop_nonstationary_well_data.py:132`).
- (46) leaves the weight $c$ unstated; the controller uses $c = 1$ (`closed_loop/cl_simulator.py:200`).
- (47)–(48) are smoothed with $\epsilon = 10^{-9}$, not SMO-1's $10^{-6}$ (`closed_loop/cl_simulator.py:91-94`).
