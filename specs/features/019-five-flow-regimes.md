# 019 · Five flow regimes after Hasan, Kabir and Sayarpour (2010)

*Feature spec, 2026-10-28. Status: approved by Bjarne on 2026-10-28 (Approval, at the end); implemented on the branch `five-flow-regimes` (Implementation); the implemented feature signed off by Bjarne on 2026-10-29 (Sign-off). Bjarne made the decisions below while it was drafted.*

## Decisions (Bjarne, 2026-10-28)

- **Five regimes**, bubbly, dispersed bubbly, slug, churn and annular, with probabilities set by the rules of Hasan et al.'s hierarchy, not fitted.
- **The bubbly–slug threshold** is Hasan et al.'s (A-14), not SLIP-11's $0.25\cos\theta$.
- **The friction factor in (A-17)** is a constant, $f_{ms} = 0.02$, a field of the slip model. The cell's Darcy factor, which Hasan and Kabir use, was measured against it first (design choice 5).
- **The constants:** the logistic width $w = 0.02$ and the floor $\epsilon_\lambda = 0.1$ of the no-slip gas fraction, as measured.
- **The default.** The five-regime model is `develop`'s default. The fitted classifier stays as an option, and the `v1.0.0` configuration uses it.
- **The labels.** With this option a root's flow regime takes the five labels.

## Motivation

`plans/evidence/slip_models.py` compares `develop`'s slip closure with Hasan et al.'s model, point by point, at two fluid states and on a grid of superficial velocities.

- **The fitted classifier's hierarchy is not Hasan et al.'s.** SLIP-9, the target of SLIP-7's fit, puts a point in slug/churn flow only if $v_{gs} \ge 1.08 v_{ls}$. In Hasan et al. that is a condition for churn flow (A-18), together with $v_m > v_{ms}$, and slug flow needs only $\alpha > 0.25$. SLIP-9 also has no dispersed bubbly flow, and merges slug and churn flow with $C_0 = 1.175$. Where only the 1.08 condition differs, the void fractions are close, as Hasan et al.'s (A-13) keeps the bubble rise velocity just past the transition: at $v_{gs}$ = 1.0 and $v_{ls}$ = 2.0 m/s ($\rho_g$ = 80 kg/m³), `develop` has bubbly flow and $\alpha$ = 0.264, Hasan et al. slug flow and 0.260.
- **Its transitions are wide.** Its velocity features are the tanh of differences in m/s (SLIP-6), and its weights a regularized fit, so near the boundaries it is far from the hierarchy it was fitted to. At $v_{gs}$ = 0.6 and $v_{ls}$ = 0.5 m/s, where SLIP-9 and Hasan et al. both have slug flow, it gives bubbly flow, with Harmathy's rise velocity, 93% of the weight in a vertical cell and 82% at 45°; $\alpha$ is 0.396 against Hasan et al.'s 0.358, and 0.382 against 0.317.
- **Several void fractions.** The classifier sees $\alpha$ through $c_2$ and $c_4$, and the slip law can have three void fractions (`slip.md`, Open question). They lie near $\alpha = 0.7$, where Hasan et al.'s model jumps: annular flow needs $\alpha > 0.7$, and at $\rho_g$ = 80 kg/m³ and $v_{ls}$ = 1 m/s the void fraction jumps from 0.543 to 0.700 as $v_{gs}$ crosses that boundary. A smooth classifier that depends on $\alpha$ turns the jump into a fold. On a 40 × 40 grid of superficial velocities at that density, 11 of the 1600 points have three void fractions in a vertical cell and 19 at 60°, the lowest between 0.65 and 0.72 and the highest between 0.85 and 0.99; Hasan et al.'s void fraction is within 0.04 of the highest.
- **Horizontal cells.** At $\cos\theta = 0$ Hasan et al. rule out bubbly flow, since (A-14) gives $v_{gb} = 0$, and the Taylor bubble's drift vanishes (SLIP-10). The fitted classifier still mixes in bubbly flow, with Harmathy's drift: at $v_{gs}$ = 0.042 and $v_{ls}$ = 0.010 m/s ($\rho_g$ = 80 kg/m³) the slip law has three void fractions, 0.20, 0.40 and 0.48, with bubbly flow weighted 80%, 24% and 15%, against Hasan et al.'s 0.67 without drift. A quarter of the sampler's wells are L-shaped, with a horizontal section (SMP-41).
- **Distance to Hasan et al.** Over the grid, at $\cos\theta$ = 1, 0.5 and 0, `develop`'s void fraction differs from Hasan et al.'s by 0.016 to 0.035 at the mean and by 0.087 to 0.172 at the 95th percentile, leaving out the points with several roots; this model's differs by 0.005 to 0.007 and 0.024 to 0.034 (Measurements).
- **Simplicity.** Bjarne asked for a model closer to Hasan et al., with all five regimes, a classifier whose parameters follow the hierarchy's logic rather than a fit, and no great complication of the model (2026-10-28).
- **Goals.** `specs/goals.md` plans a flow-regime model after Hasan et al. that separates slug and churn flow. This spec makes it five regimes, with dispersed bubbly flow.

## Delta

### `specs/model/slip.md`

The new IDs follow SLIP-11. SLIP-1, SLIP-4, SLIP-5 and SLIP-10 are unchanged and used by both options.

**SLIP-12 · Rule quantities.** At a point, with $v_{gs} = \alpha v_g$, $v_{ls} = (1 - \alpha) v_l$ and $v_m = v_{gs} + v_{ls}$ (BAL-8):

$$
\lambda_g = \frac{v_{gs}}{v_m + \epsilon_\lambda v_{\infty b}}, \qquad
\alpha_b = \frac{v_{gs}}{1.2 v_m + v_{\infty b}}, \qquad
\alpha_t = \frac{\cos\theta}{2.8 + 1.2\cos\theta},
$$

$$
v_{gc} = 3.1\left(\frac{g\sigma\thinspace(\rho_l - \rho_g)}{\rho_g^2}\right)^{1/4}, \qquad
v_{ms} = \left(\frac{0.725 + 4.15\sqrt{\lambda_g + \delta}}{2\thinspace(f_{ms}/2D)^{0.4}(\rho_l/\sigma)^{0.6}\sqrt{0.4\sigma/\big(g\thinspace(\rho_l - \rho_g)\big)}}\right)^{1/1.2},
$$

with $v_{\infty b}$ from SLIP-4, $\epsilon_\lambda = 0.1$, $\delta = 10^{-6}$ and $f_{ms} = 0.02$.

- $\lambda_g$ is the no-slip gas fraction, floored by $\epsilon_\lambda v_{\infty b}$ so that it stays smooth where the flow stops (design choice 6).
- $\alpha_b$ is the void fraction SLIP-1 gives with bubbly flow's parameters, and $\alpha_b > \alpha_t$ is Hasan et al.'s (A-14), $v_{gs} > v_{gb} = (1.2 v_{ls} + v_{\infty b})\cos\theta/2.8$. In a vertical cell it is $\alpha_b > 0.25$.
- $v_{gc}$ is (A-19), the gas velocity for annular flow, as in SLIP-6's $c_1$.
- $v_{ms}$ is (A-17) solved for the mixture velocity above which the flow is dispersed, with $\sqrt{v_{gs}/v_m}$ as $\sqrt{\lambda_g + \delta}$.

**SLIP-13 · Rules.** Each rule has a margin $z$ that is positive where it holds, and the soft truth $s = \tfrac{1}{2}\big(1 + \tanh(z/2w)\big)$, with $w = 0.02$:

| Rule | Hasan et al. | Margin |
|---|---|---|
| $A_1$: the gas is fast enough for annular flow | $v_{gs} > v_{gc}$ (A-19) | $z_{A1} = v_{gs}/v_{gc} - 1$ |
| $A_2$: there is enough gas for a core | $\alpha > 0.7$ with annular flow's parameters | $z_{A2} = \lambda_g - 0.7$ |
| $M$: the mixture is fast enough to disperse the bubbles | $v_m > v_{ms}$ (A-17) | $z_M = v_m/v_{ms} - 1$ |
| $G$: the gas dominates | $v_{gs} > 1.08 v_{ls}$ (A-18) | $z_G = \lambda_g - 1.08/2.08$ |
| $S$: past the bubbly–slug transition | $v_{gs} > v_{gb}$ (A-14) | $z_S = \alpha_b - \alpha_t$ |

**SLIP-14 · Regime probabilities.** Hasan et al.'s hierarchy as a decision tree, with $a = s_{A1}s_{A2}$:

$$
\begin{aligned}
p_a &= a, & p_c &= (1 - a) s_M s_G, & p_d &= (1 - a) s_M (1 - s_G),\\
p_s &= (1 - a)(1 - s_M) s_S, & p_b &= (1 - a)(1 - s_M)(1 - s_S),
\end{aligned}
$$

the probabilities of annular, churn, dispersed bubbly, slug and bubbly flow. They lie in $[0, 1]$ and sum to 1. As $w$ and $\epsilon_\lambda$ go to 0 they become SLIP-17's regimes. No weight is fitted.

**SLIP-15 · Profile parameter and drift velocity.** Each regime takes its value for upward cocurrent flow from Table 1 of Hasan et al.:

$$C_0 = 1.0 p_a + 1.15 p_c + 1.2 (p_d + p_s + p_b), \qquad v_\infty = v_{\infty b} (p_b + p_d) + v_{\infty T} (p_s + p_c),$$

with $v_{\infty T}$ from SLIP-5 times SLIP-10's deviation factor. So $1 \le C_0 \le 1.2$ and $v_\infty \ge 0$. Bubbly and dispersed bubbly flow have the same values; they differ in the rules that lead to them. Hasan et al.'s blends within a regime, (A-13) and (A-20) to (A-22), are left out (design choice 3).

**SLIP-16 · Regime label.** The regime with the highest probability: `annular`, `churn`, `dispersed-bubbly`, `slug` or `bubbly`. A tie goes to the regime later in that order, as SLIP-8's ties go to bubbly. It is computed after the solve, from the root, and is the `FRBH` and `FRWH` feature of a dataset made with this option.

**SLIP-17 · Reference hierarchy.** The limit of SLIP-13 and SLIP-14 as $w$ and $\epsilon_\lambda$ go to 0: Hasan et al.'s pattern-transition criteria at a point. The flow is annular if $v_{gs} > v_{gc}$ and $v_{gs}/v_m > 0.7$; else churn if $v_m > v_{ms}$ and $v_{gs} > 1.08 v_{ls}$; else dispersed bubbly if $v_m > v_{ms}$; else slug if $v_{gs} > v_{gb}$; else bubbly. The model never evaluates it. It is the reference of the tests, as SLIP-9 is the target of SLIP-7's fit.

The rest of `slip.md`:

- **Interface.** `classify` takes $D$, which $v_{ms}$ needs: `classify(v_g, v_l, alpha, rho_g, rho_l, sigma, D, cos_incl)` in both options (the fitted classifier ignores it). It returns the probabilities in the order of `SlipModel.regimes`: `('annular', 'slug-churn', 'bubbly')` for the fitted classifier, `('annular', 'churn', 'dispersed-bubbly', 'slug', 'bubbly')` for this one. `identify_parameters` is unchanged. The Rust core's `Slip::classify` already takes $d$.
- **Options.** The table gets a row for this option, `regime_model='hasan'` (SLIP-1, SLIP-4, SLIP-5, SLIP-10, SLIP-12 to SLIP-16), `develop`'s default; the fitted classifier's rows say `regime_model='fitted'`. The row "Four regimes (after the plan)" goes.
- **Safeguards.** The floor $\epsilon_\lambda$ and the guard $\delta$ in SLIP-12 (design choice 6). As in SLIP-6, nothing guards the fourth roots against $\rho_g > \rho_l$; $v_{gc}$ grows without bound as $\rho_g \to 0$, where $z_{A1} \to -1$.
- **Open question.** With this option the probabilities, $C_0$ and $v_\infty$ depend on the state only through $v_{gs}$, $v_{ls}$ and the fluid's properties, which the mass rows fix at a point. SLIP-1 is then linear in $\alpha$ there, with the one root $\alpha = v_{gs}/(C_0 v_m + v_\infty)$. The question stays open for the fitted classifier.
- **Sources and coverage** as below; SLIP-17 is spec-only, like SLIP-9.

### Other files

- `specs/model/solution.md`: the informative paragraph on several void fractions says that they arise with the fitted classifier only.
- `specs/model/README.md`: the configuration table's slip row gives `develop` SLIP-1, SLIP-4, SLIP-5, SLIP-10 and SLIP-12 to SLIP-16.
- `specs/model/nomenclature.md`: $\lambda_g$, $\alpha_b$, $\alpha_t$, $v_{gc}$, $v_{ms}$, $w$ (`transition_width`), $\epsilon_\lambda$ (`lambda_floor`), $f_{ms}$ (`f_ms`), `regime_model`, `regimes`, and the five labels.
- `specs/model/discretization.md`: unchanged; the slip closure's row is SLIP-1 in both options.
- `specs/architecture.md`: the `SlipModel` row of the interface table (`classify`'s $D$, `regimes`, the option), and the two rows that plan a four-regime model.
- `specs/goals.md`: the flow-regime model has five regimes, with dispersed bubbly flow, and is specified here, not in the draft it cites.
- `docs/datasets.md`: `FRBH` and `FRWH` take the five labels in datasets made with this option; the v1 datasets keep three.

### Code

| Where | Python | Rust |
|---|---|---|
| slip | `SlipModel.regime_model`, `'hasan'` (default) or `'fitted'`, validated in `__post_init__`; the fields `f_ms = 0.02`, `transition_width = 0.02` and `lambda_floor = 0.1`; `regimes`; SLIP-12 to SLIP-15 beside the classifier; `classify(..., D, cos_incl)`; `regime_label(probs, regimes)` | `RegimeModel::{Fitted, Hasan}` and the three fields in `Slip`; `SlipTerms::new` computes the five probabilities once, as they do not depend on $\alpha$, and `parameters(alpha)` blends them; `regime_label` for the five labels |
| harness | `configurations.v1_well` sets `regime_model='fitted'`, and `check` flags any other value; `discretization` passes $D$ to `classify`; `solvers/rust.py` passes the option and the fields, and checks the fitted classifier's constants for `'fitted'` only | `lib.rs`: `Well::new`'s arguments, `regime_index` for the five labels, the components SLIP-12 to SLIP-16 for the vectors; test wells with each option in `input.rs` |

Table 1's values are constants in the code, like the rise velocities' coefficients; the fitted classifier's four fields (§2.7) apply to it only. The Rust core's void-fraction solve, Brent on $[0, 1]$, is unchanged: with this option its function is linear in $\alpha$, with one root. No solver machinery is added.

## Design choices

1. **A decision tree, not a softmax.** Each probability is the product of its rules' soft truths along the hierarchy, so the five sum to 1 by construction and become the hierarchy as $w \to 0$, with one constant and nothing fitted. A softmax of logits built from the margins would need weights to set the regimes' logits against each other, which is what the fit chose for SLIP-7.
2. **Rules on the superficial velocities.** Hasan et al.'s two void-fraction conditions are written with the regime's own parameters, as Hasan et al. derive them: (A-14) is $\alpha = 0.25$ with bubbly flow's $C_0 = 1.2$ and $v_{\infty b}$, and their check $f_g > 0.7$ for annular flow uses annular flow's parameters, where without slip $\alpha = \lambda_g$. Every rule then depends on $v_{gs}$, $v_{ls}$ and the fluid, and the slip law has one void fraction. The fitted classifier's $c_2$ and $c_4$ use the state's $\alpha$.
3. **Table 1's values, without the blends.** Hasan et al.'s (A-13) averages the bubble and Taylor-bubble rise velocities in slug and churn flow by how far $v_{gs}$ is past $v_{gb}$, and (A-20) to (A-22) move $C_0$ from one regime's value to the next; the tree's smooth transitions take their place. On the closure grid, the crisp tree with Table 1's values differs from Hasan et al.'s model with the blends by 0.004 to 0.006 in $\alpha$ at the mean and 0.019 to 0.031 at the 95th percentile. The largest differences, 0.13 to 0.28, lie at the annular boundary, where Hasan et al. check $\alpha > 0.7$ with (A-22)'s $C_0 = 1.15$ and the tree with $C_0 = 1$.
4. **Dispersed bubbly flow's limit.** Hasan and Kabir (2018, Ch. 3) note, after Taitel et al. (1980), that dispersed bubbly flow holds a void fraction of at most 0.52. The tree needs no rule for it: a point is dispersed only where $G$ fails, $\lambda_g < 0.519$, and dispersed flow has $C_0 = 1.2$ and $v_\infty > 0$, so $\alpha < v_{gs}/(1.2 v_m) < 0.44$. Where $v_m > v_{ms}$, which is metres per second, the floor changes $\lambda_g$ by about 1%.
5. **A constant friction factor in (A-17).** Hasan and Kabir's worked examples (2018, Ch. 3 and 4) take the flow's Moody friction factor, from the mixture Reynolds number, which is the cell's $f_D$ in `develop`. With it, the slip closure would take the friction model's factor as an input in both backends, and $\alpha$ would enter $C_0$ and $v_\infty$ through the mixture's density and viscosity, so that the void fraction is unique by measurement rather than by construction. On the 39 sampled wells (Measurements) the two give the same roots: the bottomhole pressures agree within 0.013 bar and the rates within 0.08%, 3 of 3939 points change label (slug to churn), and with $f_D(\alpha)$ no point has more than one void fraction. The cell's factor costs 20% more time in total (188 against 157 s). Along the operating points $f_D$ runs from 0.011 to 0.022 (5th to 95th percentile, median 0.016); since $v_{ms} \propto f_{ms}^{-1/3}$, it moves $v_{ms}$ by a factor of 0.96 to 1.21, and few points lie near rule $M$'s boundary.
6. **The floor on $\lambda_g$ and the guard $\delta$.** With $\lambda_g = v_{gs}/v_m$ exactly, which is 0/0 where the flow stops, the CasADi search's starts near zero drawdown run Ipopt to its 3000-iteration limit instead of failing within a second, as the fitted classifier's do (the search records of wells 15 and 18, `RootSet.search`). On the 39 sampled wells the search without the floor takes 690 s in total against 157 s, with 11 wells over 20 s against 1. It finds the same operating points; the floor moves them by at most 0.46 bar in PBH (median 0.004 bar) and 0.58% in the rate. The guard $\delta$ bounds the derivative of $\sqrt{\lambda_g}$ where there is almost no free gas (PVT-OIL-13 floors the free gas rate smoothly at 0): without it the search takes 1107 s, with 14 wells over 20 s, and its operating points are the same to $10^{-3}$ bar. Both are safeguards under principle 7, with that gain.
7. **The width $w$.** On the closure grid, the labels agree with SLIP-17's at 99.9% of the points for $w$ = 0.01 and 0.02, and at 99.7% to 99.8% for 0.05. The mean difference in $\alpha$ from the crisp tree is 0.000 to 0.002, 0.001 to 0.004 and 0.006 to 0.009. At $w = 0.05$ bubbly flow leaks into horizontal cells at low gas fractions, up to 0.23 to 0.26 in $\alpha$. With $w = 0.02$ every sampled well solves (Measurements).
8. **The bubble rise velocity** is not corrected for inclination: Hasan et al. use Harmathy's "in all flow directions and well deviations" (A-8). Their remark that bubbly flow cannot exist where $v_{\infty b} > v_{\infty T}$, in small pipes, is not a rule. In a vertical cell it is Taitel et al.'s small-pipe criterion, $D\sqrt{g\thinspace(\rho_l - \rho_g)/\sigma} < 19$, and the sampler's narrowest pipe (3 inches, SMP-2) gives about 27 even with water's surface tension; in horizontal cells rule $S$ already rules bubbly flow out.
9. **Five labels.** With this option a root carries the five labels; the `v1.0.0` configuration keeps SLIP-8's three. The v2 datasets have their own schema (`specs/goals.md`, Compatibility), and the published v1 datasets do not change.

## Off in the `v1.0.0` configuration

- `v1_well` sets `regime_model='fitted'`, with the fitted classifier's default constants, and `configurations.check` flags any other slip model.
- The row vectors and both verifier reports are unchanged.

## Validity ranges

Hasan et al. tested their model on 212 field test points (their Table 2): liquid rates from 100 to 27,760 STB/D, gas–liquid ratios from 104 to 5615 scf/STB, water cuts from 0 to 98%, measured depths from 1635 to 30,650 ft and bottomhole pressures from 108 to 13,755 psig. The transition criteria (A-17) to (A-19) come from Taitel et al. (1980), for vertical pipes. Against the sampler (`five_regime_slip.py ranges`):

- **Rates and pressures** are inside: the liquid rates at the 39 operating points of the Measurements run from 310 to 23,493 STB/D, and the bottomhole pressures from 126 to 420 bar (1830 to 6090 psi).
- **Gas–liquid ratios.** The formation GLR of 2000 sampled wells has a median of 3529 scf/STB and a 95th percentile of 38,829; 38% of the wells lie above Table 2's range, 3% below. This is a finding. These wells flow mostly in annular flow, which Hasan et al. model without slip, after Kabir and Hasan's (2006) work on gas–condensate wells. The fitted classifier has the same range, as it was fitted to a hierarchy from the same source.
- **Inclination.** The deviated wells' 10° to 60° (SMP-41) lie within Hasan et al.'s deviated-well model. The horizontal sections of L-shaped wells do not: neither Hasan et al. nor this model, nor the fitted classifier, has stratified flow.

## Measurements

The scripts are in `plans/evidence/` (`README.md` there): `slip_models.py` compares the models (Motivation), `five_regime_slip.py` is the prototype and its closure grid, and `five_regime_wells.py` solves sampled wells with it. The prototype is a subclass of `SlipModel` on the CasADi backend.

**The closure** (`five_regime_slip.py`). A 60 × 60 grid of superficial velocities ($v_{gs}$ from 0.02 to 30 m/s, $v_{ls}$ from 0.01 to 5 m/s), at two fluid states ($\rho_g$ = 80 and 15 kg/m³) and $\cos\theta$ = 1, 0.5 and 0. Differences in $\alpha$ from Hasan et al.'s model, which is evaluated point by point with $f = 0.02$:

| Model | Labels as SLIP-17's | Mean | 95th percentile | Largest |
|---|---|---|---|---|
| this model, $w = 0.02$ | 99.9% | 0.005 to 0.007 | 0.024 to 0.034 | 0.10 to 0.22 |
| SLIP-17 (crisp, Table 1's values) | – | 0.004 to 0.006 | 0.019 to 0.031 | 0.13 to 0.28 |
| `develop`'s fitted classifier, points with one root | – | 0.016 to 0.035 | 0.087 to 0.172 | 0.12 to 0.50 |

**Sampled wells** (`five_regime_wells.py 40`). 39 wells of `develop`'s sampler (seed 2026; one of the 40 has no oil), each at one sampled operating point, in `develop`'s default but for the slip model, on the CasADi backend and one process. A well's variants run one after another, so their times compare, but the machine was loaded at the start of the run.

| Variant | Operating points | Wells with 1 / 2 roots | Time in total (median) | Wells over 20 s |
|---|---|---|---|---|
| the fitted classifier (`develop` today) | 38 | 31 / 7, one with none | 191 s (3.5 s) | 2 |
| this model | 39 | 31 / 8 | 157 s (3.1 s) | 1 |
| without the floor $\epsilon_\lambda$ | 39 | 31 / 8 | 690 s (3.1 s) | 11 |
| without the guard $\delta$ | 39 | 31 / 8 | 1107 s (3.7 s) | 14 |
| with the cell's $f_D$ in (A-17) | 39 | 31 / 8 | 188 s (3.7 s) | 1 |

- **The well without an operating point** under the fitted classifier, well 10 (L-shaped), has a stable and an unstable root with this model.
- **Against the fitted classifier**, in the 38 wells both solve: PBH changes by −0.73 bar at the median, 1.6 bar on average in size and up to 14.0 bar, and falls in 34 of them; the reservoir liquid rate rises by 2.6% at the median and by up to 19.8%. The change is larger than 5 bar in 3 wells.
- **Regimes.** Along this model's operating points (3838 points), 69.2% are annular, 14.0% churn, 6.5% dispersed bubbly, 6.1% slug and 4.2% bubbly. At 22.0% of the points the label differs from the fitted classifier's at the same point of its own operating point, with slug and churn merged and dispersed bubbly counted as bubbly.

**The implemented feature** (`five_regime_default.py`): the 263 cases of `dak_jt_default.py` (88 wells of `develop`'s sampler, seed 2026, at their nominal operating point and two sampled ones), with the fitted classifier (`develop`'s default before this feature) and the five regimes, on the Rust core; the CasADi backend solves the 90 cases of the first 30 wells. One process; a case's variants run one after another.

| | Rust core, fitted | Rust core, five regimes | CasADi, fitted | CasADi, five regimes |
|---|---|---|---|---|
| Operating points | 260 of 263 | 257 of 263 | 89 of 90 | 88 of 90 |
| Cases with 0 / 1 / 2 roots | 3 / 199 / 61 | 6 / 199 / 57, and one with 4 | 1 / 75 / 14 | 2 / 72 / 16 |
| Time, median (total) | 0.31 s (121 s) | 0.20 s (78 s) | 3.86 s (463 s) | 3.62 s (415 s) |

- **The lost operating points are the model's, not the solver's.** The five regimes have no operating point in 4 of the 263 cases where the fitted classifier had one, cases (11, 2), (75, 0), (75, 1) and (75, 2) as (well, operating point); they gain one in case (82, 2). In each lost case both backends find no root, and the core's residual $R(p_0)$ stays positive on 4000 samples from $p_s$ to $10^{-6}$ bar below $p_r$, down to the no-flow limit at $p_r$. Wells 11 and 75 carry little gas ($f_g$ 0.04 to 0.09) and are deviated at about 55°; under the fitted classifier each of these cases has a stable and an unstable root, the stable one 9 to 20 bar below $p_r$.
- **Against the fitted classifier**, in the 256 cases both solve on the core: PBH changes by −0.56 bar at the median, from −14.6 to +4.0 bar, and falls in 225 of them; it changes by more than 5 bar in 22. PWH changes by +0.66 bar at the median, from −23.0 to +30.6 bar; TWH by +0.25 K, from −6.7 to +5.3 K; the reservoir liquid rate by +1.2%, from −26% to +75%.
- **Regimes** along the core's operating points: 67.2% annular, 13.8% slug, 8.8% churn, 6.3% dispersed bubbly and 3.8% bubbly, against the fitted classifier's 51.8% annular, 28.8% slug/churn and 19.4% bubbly.
- **The backends agree better.** With the five regimes, the two backends' operating points agree within $9 \cdot 10^{-9}$ bar in the 88 cases both solve, and their root counts differ in one case, (9, 1), where the CasADi search misses an unstable root 0.003 bar below $p_r$. With the fitted classifier the operating points differ by up to 0.059 bar, and the root counts in 6 cases. At case (32, 1), an L-shaped well, the core finds 4 roots with the five regimes: the operating point at 224.35 bar, and within 0.7 bar of $p_r$ an unstable, a stable and an unstable root, of which the CasADi search misses the stable one; the operating point is the same.

**The backends** (`scripts/verification/compare_backends.py`, 20 sampled wells for each group this feature touches, 12 processes): all 60 cases pass Step 9's rule.

| Group | Cases | CasADi roots | Core roots | Missed | Several $\alpha$ | Largest row difference |
|---|---|---|---|---|---|---|
| `v1.0.0+five regimes` | 20 | 28 | 28 | 0 | 0 | $1.3 \cdot 10^{-13}$ |
| `develop` (the five regimes) | 20 | 22 | 22 | 0 | 0 | $4.9 \cdot 10^{-12}$ |
| `develop+fitted` | 20 | 24 | 22 | 2 | 1 | $4.9 \cdot 10^{-12}$ |

The core is 28× faster than the CasADi backend at the median and in total.

## Implementation (2026-10-28)

On the branch `five-flow-regimes`, as the Delta's code table, with these differences:

- **Labels.** `SlipModel.label(probs)` calls SLIP-8's `regime_label` or SLIP-16's `hasan_regime_label`, in place of `regime_label(probs, regimes)`; the discretization takes the labels from it. The Rust core has `Slip::label`, `Slip::regimes` and `Slip::fitted()`.
- **Defaults.** The Rust core's `Slip::default()` is the five regimes, as `SlipModel()` is. Its test wells W1 and W2, and those built from them, pin the fitted classifier; `w1_five_regimes` and `w2_five_regimes`, W1 with Joule–Thomson cooling and W2 L-shaped with it, switch the five regimes on. The components `hasan_rule_quantities` and `hasan_probabilities` serve the vectors. $\delta$ is the constant `SQRT_GUARD`.
- **Tests that assumed the old default** pin `regime_model='fitted'`: the v1.0.0 vector adapters, the tables of SLIP-10 and SLIP-11, the `slip constants` overlay, the gas wells of 016 in `tests/test_rust_backend.py`, issue #7's default well in `tests/test_march.py`, issue #10's all-liquid well in `tests/test_roots.py`, and the calibration twin `slip C_0 10% low` (`scripts/calibration/twins.py`), whose $C_0$ are the fitted classifier's.
- **A finding.** With the five regimes, point 0 of issue #10's well (well 87 with `gor = 40`) solves from the bottom guess's $\alpha = 0.5$, without the march's retry from the no-slip void fraction (`solvers/march.py`); the retry's test pins the fitted classifier.
- **No solver machinery** beyond the floor $\epsilon_\lambda$ and the guard $\delta$ of the approved spec. The core's void-fraction solve is unchanged.

## Acceptance

Done on the branch, 2026-10-28:

1. **The `v1.0.0` configuration** is unchanged: the row vectors pass, and both verifier candidates report PASS with 0 failures, 0 expected failures, 141 cases and a stable-root rate of 100%; the Rust candidate's one finding is the known second root at `fold-1503`.
2. **Component vectors.** Develop vectors for SLIP-12 (18 rows), SLIP-13 with SLIP-14 (18) and SLIP-15 with SLIP-16 (24), with states in each of the five regimes at $\cos\theta$ = 1, 0.5 and 0 (there is no bubbly flow at 0), checked in both backends by `tests/test_spec_vectors.py`. The tables of SLIP-10 and SLIP-11 pin `regime_model='fitted'`; no value changed.
3. **`tests/test_slip.py`** and the core's `slip.rs`: on 4000 random states across the sampler's range, the probabilities lie in $[0, 1]$ and sum to 1; where every margin exceeds $5w$ in size, the label is SLIP-17's, in each of the five regimes; $1 \le C_0 \le 1.2$ and $v_\infty \ge 0$; at fixed $v_{gs}$ and $v_{ls}$, $C_0$ and $v_\infty$ do not depend on $\alpha$; in a horizontal cell with $\alpha_b > 5w$ bubbly flow has at most $1/(1 + e^5)$ and is never the label; dispersed bubbly flow has $\alpha < 0.52$. `SlipModel` validates its new fields.
4. **`tests/test_model_properties.py`**, whose wells get the new default: every check passes on both backends.
5. **Backend comparison** (`tests/backend_cases.py`): feature `019`; overlays `five regimes` and `fitted`; the matrix's new entries `W1+deviated+five regimes` and `W2+L-shaped+five regimes`, and `develop` and `develop+fitted`; comparison-set groups `v1.0.0+five regimes` and `develop+fitted`. The rows agree in every configuration of the matrix, and the comparison set passes (Measurements).
6. **The Rust core.** `cargo test --no-default-features` passes (44 tests), with the march's assumption checks on the two new test wells.
7. **Sampled wells.** The option's effect on both backends (Measurements): 3 fewer operating points of 263 on the core and 1 fewer of 90 on the CasADi backend, each a case where the model has no root; no solver finding.
8. **The suites.** The full test suite passes: 822 passed and 1 skipped, after merging `develop` with feature 018's phase A; both verifier candidates as in item 1 after the merge too. Its one warning, a root only the core finds in `v1.0.0+L-shaped#0` at 433.06 bar, is 016's (`plans/improvements.md` §2.9).

## Out of scope

- Countercurrent and downward flow (Table 1's other columns), the annulus factor (A-11), and oil–water slip, (A-23) to (A-27), a non-goal (`specs/goals.md`).
- Hasan et al.'s top-down assignment of regimes, which keeps annular flow below the wellhead until its criteria fail. A closure at a point cannot hold it, and the tree is continuous without it.
- The blends (A-13) and (A-20) to (A-22) (design choice 3).
- Stratified flow in horizontal and near-horizontal sections.
- The cell's friction factor in (A-17), measured above; a later option.
- Retiring the fitted classifier, which stays as an option (Decisions).
- Shi et al.'s (2005) drift-flux closure, which `slip_models.py` compares; it has no flow regimes, and its parameters were fitted to near-atmospheric water–gas flow in one 15.2 cm pipe.

## Approval (Bjarne, 2026-10-28)

- **The spec**, as written, before the implementation starts. Approved.
- **The edits to `specs/model/`** (SLIP-12 to SLIP-17; `slip.md`'s interface, options, safeguards and open question; `solution.md`; the configuration table) and to `specs/goals.md`; **the changed default of `develop`**, which changes its results (Measurements). Signed off.
- **The constants** $w = 0.02$, $\epsilon_\lambda = 0.1$, $\delta = 10^{-6}$ and $f_{ms} = 0.02$, with their measured cost and gain (design choices 5 to 7). Signed off.
- **The five labels** in the v2 datasets' `FRBH` and `FRWH` (`docs/datasets.md`). Signed off.

The implemented feature's measurements (Acceptance, item 7), and any solver machinery the implementation finds it needs, come back for sign-off, as in 016.

## Sign-off (Bjarne, 2026-10-29)

On the implemented feature:

- **The changed results of `develop`** on sampled wells (Measurements, the implemented feature), including the 4 cases of 263 that lose their operating point, where the model has no root, and the 1 that gains one. Signed off.
- **The implementation's differences from the Delta** (Implementation): `SlipModel.label` in place of `regime_label(probs, regimes)`, and the tests and the calibration twin that now pin the fitted classifier. Signed off.

## Sources

- Hasan, Kabir and Sayarpour (2010), "Simplified two-phase flow modeling in wellbores", *Journal of Petroleum Science and Engineering* 72, 42–49: Table 1, Eqs. (A-6) to (A-10), (A-13), (A-14) and (A-17) to (A-22), and Table 2 for the ranges.
- Hasan and Kabir (2018), *Fluid flow and heat transfer in wellbores*, 2nd ed., Society of Petroleum Engineers: Ch. 3, dispersed bubbly flow, its limit of 0.52 and the criterion (A-17) as Eq. 3.10, with the worked examples' friction factor; Ch. 4, the example in a deviated well.
- Taitel, Barnea and Dukler (1980), "Modelling flow pattern transitions for steady upward gas–liquid flow in vertical tubes", *AIChE Journal* 26, 345–354: the source of (A-17) to (A-19). Not consulted.
- Kabir and Hasan (2006), "Simplified wellbore flow modeling in gas/condensate systems", *SPE Production & Operations* 21, 89–97: annular flow without slip. Not consulted.
- Shi, Holmes, Durlofsky, Aziz, Diaz, Alkaya and Oddie (2005), "Drift-flux modeling of two-phase flow in wellbores", *SPE Journal* 10, 24–33: compared, not adopted.
- Harmathy (1960), for SLIP-4, as before.
