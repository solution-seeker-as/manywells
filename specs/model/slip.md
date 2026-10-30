# Slip

## Purpose

The drift-flux slip law that sets the gas velocity relative to the mixture, with parameters that depend on the flow regime. A smooth classifier gives the probability of each regime, and the slip parameters are the probability-weighted averages of each regime's values, after Hasan, Kabir and Sayarpour (2010). There are two classifiers: v1.0.0's, fitted to a hierarchy of three regimes (bubbly, slug/churn, annular; paper §2.2 and Appendix A; SLIP-2 to SLIP-11), and `develop`'s default, five regimes (bubbly, dispersed bubbly, slug, churn, annular) from the rules of Hasan et al.'s hierarchy (SLIP-12 to SLIP-17; `specs/features/019-five-flow-regimes.md`). With either, a scale on the profile parameter can move $C_0$ towards no slip, for calibration (SLIP-18; `specs/features/020-profile-scale.md`).

## Interface

| Function | Inputs | Outputs | Symbolic |
|---|---|---|---|
| slip row | $v_g, v_l, \alpha, \rho_g, \rho_l$ (state), $\sigma$, $D$ | row (m/s) | yes |
| slip parameters | $v_g, v_l, \alpha, \rho_g, \rho_l, \sigma, D$ | $C_0$ (–), $v_\infty$ (m/s) | yes |
| classifier | $v_g, v_l, \alpha, \rho_g, \rho_l, \sigma$, $D$ | the probabilities of `SlipModel.regimes` (–) | yes |
| regime label | the classifier's inputs, as floats | one of `SlipModel.regimes` | no |

With the fitted classifier, `regime_model='fitted'`, the regimes are `('annular', 'slug-churn', 'bubbly')`, and $p_a$, $p_s$ and $p_b$ are their probabilities; it does not use $D$. With the five regimes, `regime_model='hasan'`, they are `('annular', 'churn', 'dispersed-bubbly', 'slug', 'bubbly')`, with the probabilities $p_a$, $p_c$, $p_d$, $p_s$ and $p_b$. $\sigma$ comes from PVT-MIX-5 (or PVT-MIX-7 in `develop`'s default) and is evaluated at the same point. v1.0.0's functions take $T$ and compute $\sigma$ themselves; `develop`'s take $\sigma$ and the inclination $\cos\theta$ of the point's cell (DISC-11). The vectors below give $\sigma$.

In `develop` the four regime constants of SLIP-2 and SLIP-3 (1.0, 1.175 and 1.2 for $C_0$, and 0 for the annular drift velocity) are parameters of the fitted classifier, fields of `SlipModel` with these values as defaults (`plans/backlog.md` §2.7). The `v1.0.0` configuration uses the fitted classifier with the defaults. The five regimes' constants $f_{ms}$, $w$ and $\epsilon_\lambda$ of SLIP-12 and SLIP-13 are fields too (`f_ms`, `transition_width`, `lambda_floor`); Table 1's values in SLIP-15 are constants. The scale $\kappa_0$ of SLIP-18 is the field `profile_scale` of either classifier, 1 by default, which leaves $C_0$ as it is; it is a free parameter of a calibration (CAL-5).

## Equations

### SLIP-1 · Slip law

$$v_g = C_0 v_m + v_\infty$$

with $v_m$ from BAL-8. Its row in DISC-6 is $v_g - C_0 v_m - v_\infty$ (m/s), at every grid point.

### SLIP-2 · Profile parameter

$$C_0 = 1.0 p_a + 1.175 p_s + 1.2 p_b$$

The regime values are those of Table A.6: 1.2 for bubbly flow, 1.0 for annular flow, and for slug/churn flow the average of the values for slug (1.2) and churn (1.15) flow.

### SLIP-3 · Drift velocity

$$v_\infty = 0 \cdot p_a + v_{\infty T} p_s + v_{\infty b} p_b$$

This is (A.10) as corrected in `docs/corrigendum.md`: the paper swaps the two rise velocities, and the code does not.

### SLIP-4 · Bubble rise velocity (Harmathy)

$$v_{\infty b} = 1.53 \left(\frac{g\sigma\thinspace(\rho_l - \rho_g)}{\rho_l^2}\right)^{1/4}$$

### SLIP-5 · Taylor-bubble rise velocity

$$v_{\infty T} = 0.35\sqrt{g D\thinspace(1 - \rho_g/\rho_l)}$$

for a vertical pipe; SLIP-10 corrects it for inclination.

### SLIP-6 · Classifier features

$$
c_1 = \tanh\Big(v_{gs} - 3.1\big(g\sigma\thinspace(\rho_l - \rho_g)/\rho_g^2\big)^{1/4}\Big), \quad
c_2 = \tanh\big(2(\alpha - 0.7)\big), \quad
c_3 = \tanh\big(v_{gs} - 1.08 v_{ls}\big), \quad
c_4 = \tanh\big(2(\alpha - 0.25)\big)
$$

with $v_{gs} = \alpha v_g$ and $v_{ls} = (1 - \alpha) v_l$ in m/s. The arguments of $c_1$ and $c_3$ are velocities in m/s, not dimensionless. The factor 2 in $c_2$ and $c_4$ is in v1.0.0's code and not in the paper's (A.8) (`specs/discrepancies.md`, D-1). The paper lists the features in the reverse order.

### SLIP-7 · Classifier output

$$
\begin{aligned}
y_a &= \phantom{-}3.17715258 c_1 + 6.81938489 c_2 + 0.30182974 c_3 + 3.58362465 c_4 - 3.92904391\\
y_s &= -1.47973427 c_1 - 4.34033317 c_2 + 2.58200006 c_3 + 3.49656911 c_4 - 1.46509477\\
y_b &= -1.6974183 c_1 - 2.47905172 c_2 - 2.8838298 c_3 - 7.08019376 c_4 + 5.39413869
\end{aligned}
$$

$$(p_a, p_s, p_b) = \mathop{\mathrm{softmax}}(y_a, y_s, y_b)$$

with the softmax of SMO-3. This is (A.7) with v1.0.0's weights $A$ and $b$, which the paper does not list; the paper orders the outputs (bubbly, slug/churn, annular).

### SLIP-8 · Regime label

The label of a point is the regime with the highest probability: `annular` if $p_a > p_s$ and $p_a > p_b$; else `slug-churn` if $p_s > p_a$ and $p_s > p_b$; else `bubbly`, which also takes ties. It is computed after the solve, from the root, and is the `FRBH` and `FRWH` feature of the datasets (at points 0 and $N$). It is not part of the discretized system.

### SLIP-9 · Reference regime hierarchy

The hierarchy that the classifier approximates, from (A.3)–(A.6): the flow is annular if $\alpha \ge 0.7$ and $v_{gs} \ge 3.1\big(g\sigma(\rho_l - \rho_g)/\rho_g^2\big)^{1/4}$; else slug/churn if $\alpha \ge 0.25$ and $v_{gs} \ge 1.08 v_{ls}$; else bubbly. The weights of SLIP-7 were fitted to it by multinomial logistic regression on sampled points (scikit-learn, L2 penalty, $C = 0.01$). The model never evaluates the hierarchy itself.

### SLIP-10 · Deviation factor of the Taylor-bubble rise velocity

In an inclined cell, the Taylor-bubble rise velocity of SLIP-5 is multiplied by

$$\sqrt{\cos\theta}\thinspace(1 + \sin\theta)^{1.2}, \qquad \sin\theta = \sqrt{1 - \cos^2\theta},$$

Hasan, Kabir and Sayarpour (2010), Eq. (A-10), with $\theta$ the inclination from vertical. The factor is exactly 1 in a vertical cell, and 0 in a horizontal one. `develop` had $\sqrt{\cos\theta + 10^{-9}}$, a guard against a negative argument that GEO-3 already rules out ($\cos\theta \ge 0$); Step 7 dropped it, so the vertical case is exact (`plans/develop_model_changes.md`, change 2; decided by Bjarne, 2026-10-01).

### SLIP-11 · Bubbly–slug threshold in an inclined cell

The fourth classifier feature of SLIP-6 becomes

$$c_4 = \tanh\big(2(\alpha - 0.25\cos\theta)\big),$$

so the transition from bubbly to slug flow comes at a lower void fraction in an inclined cell. The weights of SLIP-7 are unchanged, and were fitted for the vertical threshold. At $\cos\theta = 1$ it is SLIP-6's $c_4$.

### SLIP-12 · Rule quantities of the five regimes

At a point, with $v_{gs} = \alpha v_g$, $v_{ls} = (1 - \alpha) v_l$ and $v_m = v_{gs} + v_{ls}$ (BAL-8):

$$
\lambda_g = \frac{v_{gs}}{v_m + \epsilon_\lambda v_{\infty b}}, \qquad
\alpha_b = \frac{v_{gs}}{1.2 v_m + v_{\infty b}}, \qquad
\alpha_t = \frac{\cos\theta}{2.8 + 1.2\cos\theta},
$$

$$
v_{gc} = 3.1\left(\frac{g\sigma\thinspace(\rho_l - \rho_g)}{\rho_g^2}\right)^{1/4}, \qquad
v_{ms} = \left(\frac{0.725 + 4.15\sqrt{\lambda_g + \delta}}{2\thinspace(f_{ms}/2D)^{0.4}(\rho_l/\sigma)^{0.6}\sqrt{0.4\sigma/\big(g\thinspace(\rho_l - \rho_g)\big)}}\right)^{1/1.2},
$$

with $v_{\infty b}$ from SLIP-4, $\epsilon_\lambda = 0.1$, $\delta = 10^{-6}$ and $f_{ms} = 0.02$. $\lambda_g$ is the no-slip gas fraction, floored so that it is smooth where the flow stops. $\alpha_b$ is the void fraction SLIP-1 gives with bubbly flow's parameters, and $\alpha_b > \alpha_t$ is Hasan et al.'s (A-14), $v_{gs} > v_{gb} = (1.2 v_{ls} + v_{\infty b})\cos\theta/2.8$; in a vertical cell, $\alpha_b > 0.25$. $v_{gc}$ is (A-19), the gas velocity for annular flow, as in SLIP-6's $c_1$. $v_{ms}$ is (A-17) solved for the mixture velocity above which the flow is dispersed, with $\sqrt{v_{gs}/v_m}$ as $\sqrt{\lambda_g + \delta}$ and $f_{ms}$ for the flow's friction factor.

### SLIP-13 · Rules of the five regimes

Each rule has a margin $z$ that is positive where it holds, and the soft truth $s = \tfrac{1}{2}\big(1 + \tanh(z/2w)\big)$, with $w = 0.02$:

| Rule | Hasan et al. | Margin |
|---|---|---|
| $A_1$: the gas is fast enough for annular flow | $v_{gs} > v_{gc}$ (A-19) | $z_{A1} = v_{gs}/v_{gc} - 1$ |
| $A_2$: there is enough gas for a core | $\alpha > 0.7$ with annular flow's parameters | $z_{A2} = \lambda_g - 0.7$ |
| $M$: the mixture is fast enough to disperse the bubbles | $v_m > v_{ms}$ (A-17) | $z_M = v_m/v_{ms} - 1$ |
| $G$: the gas dominates | $v_{gs} > 1.08 v_{ls}$ (A-18) | $z_G = \lambda_g - 1.08/2.08$ |
| $S$: past the bubbly–slug transition | $v_{gs} > v_{gb}$ (A-14) | $z_S = \alpha_b - \alpha_t$ |

### SLIP-14 · Probabilities of the five regimes

Hasan et al.'s hierarchy as a decision tree, with $a = s_{A1}s_{A2}$:

$$
\begin{aligned}
p_a &= a, & p_c &= (1 - a) s_M s_G, & p_d &= (1 - a) s_M (1 - s_G),\\
p_s &= (1 - a)(1 - s_M) s_S, & p_b &= (1 - a)(1 - s_M)(1 - s_S),
\end{aligned}
$$

the probabilities of annular, churn, dispersed bubbly, slug and bubbly flow. They lie in $[0, 1]$ and sum to 1, and as $w$ and $\epsilon_\lambda$ go to 0 they become SLIP-17's regimes. No weight is fitted.

### SLIP-15 · Profile parameter and drift velocity of the five regimes

Each regime takes its value for upward cocurrent flow from Table 1 of Hasan et al.:

$$C_0 = 1.0 p_a + 1.15 p_c + 1.2 (p_d + p_s + p_b), \qquad v_\infty = v_{\infty b} (p_b + p_d) + v_{\infty T} (p_s + p_c),$$

with $v_{\infty T}$ from SLIP-5 times SLIP-10's deviation factor. So $1 \le C_0 \le 1.2$ and $v_\infty \ge 0$. Bubbly and dispersed bubbly flow have the same values; they differ in the rules that lead to them. Hasan et al.'s blends within a regime, (A-13) and (A-20) to (A-22), are left out.

### SLIP-16 · Regime label of the five regimes

The regime with the highest probability: `annular`, `churn`, `dispersed-bubbly`, `slug` or `bubbly`. A tie goes to the regime later in that order, as SLIP-8's ties go to bubbly. It is computed after the solve, from the root, and is the `FRBH` and `FRWH` feature of a dataset made with the five regimes.

### SLIP-17 · Reference hierarchy of the five regimes

The limit of SLIP-13 and SLIP-14 as $w$ and $\epsilon_\lambda$ go to 0: Hasan et al.'s pattern-transition criteria at a point. The flow is annular if $v_{gs} > v_{gc}$ and $v_{gs}/v_m > 0.7$; else churn if $v_m > v_{ms}$ and $v_{gs} > 1.08 v_{ls}$; else dispersed bubbly if $v_m > v_{ms}$; else slug if $v_{gs} > v_{gb}$; else bubbly. The model never evaluates it. It is the reference of the tests, as SLIP-9 is the target of SLIP-7's fit.

### SLIP-18 · Scaled profile parameter

The profile parameter $C_0$ of SLIP-2 or SLIP-15 is scaled about 1:

$$C_0^\ast = 1 + \kappa_0\thinspace(C_0 - 1), \qquad \kappa_0 \ge 0,$$

and $C_0^\ast$ takes $C_0$'s place in SLIP-1; $v_\infty$ is unchanged.
- **The ends.** $\kappa_0 = 1$, the default, is the classifier's own $C_0$, exactly. $\kappa_0 = 0$ gives $C_0^\ast = 1$, no slip in the profile.
- **Between them.** For $0 \le \kappa_0 \le 1$, $C_0^\ast$ lies between 1 and $C_0$. Since the probabilities sum to 1, it scales every regime's $C_0 - 1$ by the same $\kappa_0$, and annular flow, with $C_0 = 1$ in both classifiers, does not move.
- **The bound.** $C_0 \ge 1$ for every mix of the regimes, so $C_0^\ast \ge 1$ for every $\kappa_0 \ge 0$: the scale never takes the flow past no slip.
- **One scale.** One scale for every regime, rather than one per regime, is a decision (`specs/features/020-profile-scale.md`, design choice).

## Options

| Option | IDs | Used by |
|---|---|---|
| Three regimes, fitted, vertical pipe: `regime_model='fitted'` | SLIP-1 to SLIP-8 | `v1.0.0` |
| Three regimes, fitted, with inclination: bubbly–slug threshold $0.25\cos\theta$, deviation factor on $v_{\infty T}$ | SLIP-10, SLIP-11, with SLIP-1 to SLIP-8 | `regime_model='fitted'` on any grid; on a vertical grid it is the `v1.0.0` option exactly |
| Five regimes from Hasan et al.'s hierarchy: `regime_model='hasan'` | SLIP-1, SLIP-4, SLIP-5, SLIP-10, SLIP-12 to SLIP-16 | `develop`'s default |
| Scaled profile parameter: `profile_scale` | SLIP-18, with either classifier | `develop` with $\kappa_0 \ne 1$; a free parameter of a calibration (CAL-5). At the default $\kappa_0 = 1$ it is the identity, and the `v1.0.0` configuration keeps that |

## Safeguards

- None in the equations: no clipping, and nothing guards the fourth roots and the square root against negative arguments, which occur only if $\rho_g > \rho_l$.
- The factor 2 in $c_2$ and $c_4$ steepens the transitions in $\alpha$ ("to increase sensitivity", v1.0.0's comment).
- The softmax is evaluated without shifting the logits. $|c_k| \le 1$ bounds the logits to $|y| < 22$, so it cannot overflow.
- With the five regimes: the floor $\epsilon_\lambda v_{\infty b}$ keeps $\lambda_g$ smooth where $v_m \to 0$, and the guard $\delta$ bounds the derivative of $\sqrt{\lambda_g}$ where there is almost no free gas (PVT-OIL-13 floors the free gas rate smoothly at 0). Without them the CasADi search's starts near zero drawdown run Ipopt to its iteration limit (`specs/features/019-five-flow-regimes.md`, design choice 6). As in SLIP-6, nothing guards the fourth roots against $\rho_g > \rho_l$; $v_{gc}$ grows without bound as $\rho_g \to 0$, where $z_{A1} \to -1$.

## Open question: several void fractions

At fixed superficial velocities $j_g = \alpha v_g$ and $j_l = (1 - \alpha) v_l$, which the mass rows fix at a point, SLIP-1 is one equation in $\alpha$: $h(\alpha) = \alpha\thinspace(C_0 j_m + v_\infty) - j_g = 0$, which is $-\alpha$ times its row. The classifier sees $\alpha$ only through $c_2$ and $c_4$, and $C_0 \ge 1$ and $v_\infty \ge 0$ for every mix of the regimes, scaled or not (SLIP-18). So $h(0) < 0 < h(1)$, and a root always exists, but it need not be unique.
- **Where it occurs.** Near the slug–annular transition with little liquid there can be three roots. In Step 8's regenerated `sol-1` samples, at well 44 (k = 4: $u$ = 0.056, 3.8 kg/s of lift gas, $j_l$ = 0.016 m/s), point 95 has $\alpha$ = 0.662, 0.915 and 0.920.
- **Consequence for the root set.** SOL-2 then holds roots that differ only in the branch at some points, and which branch is physical is not specified (`solution.md`, informative section).
- **The case set.** At every point of every reference root, the root is unique (`tests/test_rust_backend.py`).
- **Consequence for the Rust core.** Where the void fraction a march takes switches branch between neighbouring $p_0$, $R(p_0)$ jumps across zero, and the core can accept the jump as a root: at `v1.0.0+deviated#17` of Step 9's comparison set, its choke row is $-7.5 \cdot 10^{-4}$ kg/s, under the acceptance bound of $10^{-3} w_m$. A known finding, not changed (Bjarne, 2026-10-02, `specs/features/015-rust-develop-model.md`, Finding 3).
- **The five regimes.** With `regime_model='hasan'` the probabilities, $C_0$ and $v_\infty$ (SLIP-12 to SLIP-15) depend on the state only through $j_g$, $j_l$ and the fluid's properties, which the mass rows fix at a point. $h$ is then linear in $\alpha$, with the one root $\alpha = j_g/(C_0 j_m + v_\infty)$.
- **Status.** Settled for `develop`'s default, the five regimes (`specs/features/019-five-flow-regimes.md`); open for the fitted classifier (Bjarne, 2026-10-01), which the `v1.0.0` configuration keeps. The new sampler uses `develop`'s default only, and the `v2.0.0` configuration leaves the fitted classifier out, so the question stays open for the fitted classifier alone and is not ruled for v2 (Bjarne, 2026-10-29; `plans/model-lock-plan.md`, Step L4).

## Sources

- Paper (8), Appendix A.1–A.2, Table A.6, and `docs/corrigendum.md` for (A.10).
- Hasan, Kabir and Sayarpour (2010), "Simplified two-phase flow modeling in wellbores", *Journal of Petroleum Science and Engineering* 72, 42–49: Eqs. (A-18) and (A-19) for the transitions, (A-10) for SLIP-10.
- Hasan, Kabir and Sayarpour (2010): Table 1, Eqs. (A-8) to (A-10), (A-14) and (A-17) to (A-19), for SLIP-12 to SLIP-17.
- Hasan and Kabir (2018), *Fluid flow and heat transfer in wellbores*, 2nd ed., Society of Petroleum Engineers, Ch. 3: dispersed bubbly flow, its limit of $\alpha = 0.52$ (which SLIP-14 meets without a rule, as dispersed flow has $\lambda_g < 0.519$), and the criterion (A-17) as Eq. 3.10.
- Taitel, Barnea and Dukler (1980), "Modelling flow pattern transitions for steady upward gas–liquid flow in vertical tubes", *AIChE Journal* 26, 345–354: the source of (A-17) to (A-19). Not consulted.
- Feature specs `specs/features/002-slip-inclination.md`, `specs/features/019-five-flow-regimes.md` and `specs/features/020-profile-scale.md` (SLIP-18, with the published range of $C_0$ that motivates it).
- Harmathy (1960), "Velocity of large drops and bubbles in media of infinite or restricted extent", *AIChE Journal* 6, 281–288.
- Hasan and Kabir (2018): Field Examples 3.1 (§3.4) and 4.1 (§4.2.4) and Appendix 7-E, §7-E.2, worked examples of SLIP-4, SLIP-5 and SLIP-12 to SLIP-16.

## Test vectors

The $\sigma$ column is $\sigma_{od}(\rho_l, T)$ (PVT-MIX-5), and the `T` column records the temperature it was computed at; the functions under test take $\sigma$, not $T$.

**Independent checks**, on `develop` at `f7f1218` (2026-10-29; `README.md`, Test vectors). Both backends reproduce Hasan and Kabir's field examples (2018) at the precision printed (`tests/test_worked_examples.py`):

- the five regimes give each example's flow regime: slug flow in Field Example 3.1 (§3.4), churn flow in Field Example 4.1 (§4.2.4) and annular flow in Appendix 7-E, §7-E.2 (SLIP-12 to SLIP-14, SLIP-16), and $C_0 = 1.2$ in slug flow (SLIP-15);
- SLIP-4 gives their bubble-rise velocities, 0.680 and 0.675 ft/s, and SLIP-5 the Taylor-bubble rise velocity of Example 3.1, 0.971 ft/s.

Their void fractions are not checked. The examples blend the rise velocities within slug and churn flow, (A-13) of Hasan et al. (2010), which SLIP-15 leaves out, and the points of Examples 4.1 and 7-E.2 lie within a few percent of a transition, which SLIP-13 smooths: $\alpha$ is 0.451 (at Example 3.1's $v_{sg}$ of 2.845 ft/s in its solution, 2.784 in its data), 0.626 and 0.815, against the printed 0.457, 0.583 and 0.927. Example 4.1 prints a deviation factor of 1.208, which its own Eq. 4.33 does not give (1.339, SLIP-10's), so no example checks SLIP-10. The fitted classifier's weights (SLIP-7) are v1.0.0's own fit to SLIP-9, with no source outside ManyWells; its vectors from v1.0.0 check that `develop` reproduces them.

<!-- vectors:begin -->
Generated by `specs/tools/make_v1_vectors.py` from ManyWells v1.0.0 (casadi 3.6.4). Do not edit by hand.

### SLIP-2, SLIP-3

| v_g | v_l | alpha | rho_g | rho_l | T | sigma | D | → C_0 | → v_inf |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.5 | 0.1 | 100.0 | 850.0 | 360.0 | 0.02473603373432514 | 0.1554 | 1.1999999600030595 | 0.1927350661121966 |
| 3.0 | 1.0 | 0.45 | 50.0 | 850.0 | 340.0 | 0.026509085472736903 | 0.1016 | 1.1797679661213705 | 0.31226792987525964 |
| 25.0 | 3.0 | 0.9 | 20.0 | 900.0 | 320.0 | 0.03061073447782097 | 0.127 | 1.002517373617638 | 0.005553518471674428 |
| 20.0 | 5.0 | 0.5 | 1.0 | 900.0 | 293.15 | 0.03318703919302389 | 0.1 | 1.175455953500943 | 0.343934716387237 |
| 12.0 | 2.5 | 0.7 | 30.0 | 999.1 | 300.0 | 0.036702451162500004 | 0.0762 | 1.0886845097297446 | 0.15089606087155932 |

### SLIP-4

| rho_g | rho_l | T | sigma | → v_inf_b |
|---|---|---|---|---|
| 100.0 | 850.0 | 360.0 | 0.02473603373432514 | 0.19273472514415424 |
| 50.0 | 850.0 | 340.0 | 0.026509085472736903 | 0.19928899633437236 |
| 20.0 | 900.0 | 320.0 | 0.03061073447782097 | 0.20560773581541836 |
| 1.0 | 900.0 | 293.15 | 0.03318703919302389 | 0.21092710647153223 |
| 30.0 | 999.1 | 300.0 | 0.036702451162500004 | 0.20918620251431558 |

### SLIP-5

| rho_g | rho_l | D | → v_inf_T |
|---|---|---|---|
| 100.0 | 850.0 | 0.1554 | 0.40585888527584674 |
| 50.0 | 850.0 | 0.1016 | 0.3389305893195103 |
| 20.0 | 900.0 | 0.127 | 0.386233841790753 |
| 1.0 | 900.0 | 0.1 | 0.34640725035313885 |
| 30.0 | 999.1 | 0.0762 | 0.29797901835718316 |

### SLIP-6, SLIP-7

| v_g | v_l | alpha | rho_g | rho_l | T | sigma | → p_a | → p_s | → p_b |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.5 | 0.1 | 100.0 | 850.0 | 360.0 | 0.02473603373432514 | 2.387446864072425e-12 | 1.5998585157312343e-06 | 0.9999984001390968 |
| 3.0 | 1.0 | 0.45 | 50.0 | 850.0 | 340.0 | 0.026509085472736903 | 2.3095991071330792e-05 | 0.8090965872166183 | 0.19088031679231052 |
| 25.0 | 3.0 | 0.9 | 20.0 | 900.0 | 320.0 | 0.03061073447782097 | 0.9856164935813294 | 0.014373106643840013 | 1.039977483057048e-05 |
| 20.0 | 5.0 | 0.5 | 1.0 | 900.0 | 293.15 | 0.03318703919302389 | 1.257467851682411e-06 | 0.9817518002194707 | 0.018246942312677598 |
| 12.0 | 2.5 | 0.7 | 30.0 | 999.1 | 300.0 | 0.036702451162500004 | 0.49335138786821975 | 0.505808507864451 | 0.0008401042673291876 |

### SLIP-8

| v_g | v_l | alpha | rho_g | rho_l | T | sigma | → regime |
|---|---|---|---|---|---|---|---|
| 1.0 | 1.5 | 0.1 | 100.0 | 850.0 | 360.0 | 0.02473603373432514 | bubbly |
| 3.0 | 1.0 | 0.45 | 50.0 | 850.0 | 340.0 | 0.026509085472736903 | slug-churn |
| 25.0 | 3.0 | 0.9 | 20.0 | 900.0 | 320.0 | 0.03061073447782097 | annular |
| 20.0 | 5.0 | 0.5 | 1.0 | 900.0 | 293.15 | 0.03318703919302389 | slug-churn |
| 12.0 | 2.5 | 0.7 | 30.0 | 999.1 | 300.0 | 0.036702451162500004 | slug-churn |
<!-- vectors:end -->

<!-- vectors:begin develop -->
Generated by `specs/tools/make_develop_vectors.py` from develop (casadi 3.8.1): they pin develop's options. Do not edit by hand.

### SLIP-10, SLIP-11

| v_g | v_l | alpha | rho_g | rho_l | sigma | D | cos_incl | → C_0 | → v_inf |
|---|---|---|---|---|---|---|---|---|---|
| 1.2 | 1.0 | 0.15 | 60.0 | 780.0 | 0.022 | 0.1524 | 1.0 | 1.1999996117219047 | 0.19340633028659143 |
| 1.2 | 1.0 | 0.15 | 60.0 | 780.0 | 0.022 | 0.1524 | 0.7 | 1.199998153849992 | 0.19343715928453148 |
| 1.2 | 1.0 | 0.15 | 60.0 | 780.0 | 0.022 | 0.1524 | 0.0 | 1.1999319669403223 | 0.19287664106739574 |
| 6.0 | 2.0 | 0.4 | 80.0 | 800.0 | 0.02 | 0.1524 | 1.0 | 1.1783261142814245 | 0.37388375219412906 |
| 6.0 | 2.0 | 0.4 | 80.0 | 800.0 | 0.02 | 0.1524 | 0.7 | 1.1757181956143978 | 0.6287015801239744 |
| 6.0 | 2.0 | 0.4 | 80.0 | 800.0 | 0.02 | 0.1524 | 0.0 | 1.1747731631080285 | 0.0006032154445075432 |
| 25.0 | 3.0 | 0.85 | 40.0 | 820.0 | 0.018 | 0.1524 | 1.0 | 1.006624889739434 | 0.01578881358848783 |
| 25.0 | 3.0 | 0.85 | 40.0 | 820.0 | 0.018 | 0.1524 | 0.7 | 1.006600416097813 | 0.025130258112749354 |
| 25.0 | 3.0 | 0.85 | 40.0 | 820.0 | 0.018 | 0.1524 | 0.0 | 1.0065647093032866 | 1.931513694337049e-06 |
| 3.0 | 1.5 | 0.27 | 100.0 | 750.0 | 0.025 | 0.1524 | 1.0 | 1.1999803793302484 | 0.19865397178970357 |
| 3.0 | 1.5 | 0.27 | 100.0 | 750.0 | 0.025 | 0.1524 | 0.7 | 1.1999066335700104 | 0.2001318527525124 |
| 3.0 | 1.5 | 0.27 | 100.0 | 750.0 | 0.025 | 0.1524 | 0.0 | 1.1978389412546517 | 0.18134131490030822 |

### SLIP-11 (classifier)

| v_g | v_l | alpha | rho_g | rho_l | sigma | cos_incl | → p_annular | → p_slug | → p_bubbly |
|---|---|---|---|---|---|---|---|---|---|
| 1.2 | 1.0 | 0.15 | 60.0 | 780.0 | 0.022 | 0.7 | 6.803120123988049e-11 | 7.384545606512948e-05 | 0.9999261544759036 |
| 1.2 | 1.0 | 0.15 | 60.0 | 780.0 | 0.022 | 0.2 | 9.500855899419222e-10 | 0.0010093161920342241 | 0.9989906828578802 |
| 6.0 | 2.0 | 0.4 | 80.0 | 800.0 | 0.02 | 0.7 | 0.001657642975866415 | 0.9580110316171563 | 0.04033132540697738 |
| 6.0 | 2.0 | 0.4 | 80.0 | 800.0 | 0.02 | 0.2 | 0.0017442756503283504 | 0.9921924056294804 | 0.006063318720191246 |
| 25.0 | 3.0 | 0.85 | 40.0 | 820.0 | 0.018 | 0.7 | 0.9622862362971848 | 0.03769346571 | 2.0297992815283018e-05 |
| 25.0 | 3.0 | 0.85 | 40.0 | 820.0 | 0.018 | 0.2 | 0.9624440813144379 | 0.037543700499933524 | 1.2218185628515325e-05 |
| 3.0 | 1.5 | 0.27 | 100.0 | 750.0 | 0.025 | 0.7 | 8.208132685382527e-08 | 0.0037340005489674205 | 0.9962659173697057 |
| 3.0 | 1.5 | 0.27 | 100.0 | 750.0 | 0.025 | 0.2 | 8.803572550226391e-07 | 0.03926885649938972 | 0.9607302631433552 |

### SLIP-12

| v_gs | v_ls | rho_g | rho_l | sigma | D | cos_incl | → lambda_g | → alpha_b | → alpha_t | → v_gc | → v_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 1.0 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 0.046804870886254735 | 0.03465846876679883 | 0.25 | 1.1555520317835992 | 2.919200259736721 |
| 0.05 | 1.0 | 80.0 | 780.0 | 0.018 | 0.127 | 0.5 | 0.046804870886254735 | 0.03465846876679883 | 0.14705882352941177 | 1.1555520317835992 | 2.919200259736721 |
| 0.05 | 1.0 | 80.0 | 780.0 | 0.018 | 0.127 | 0.0 | 0.046804870886254735 | 0.03465846876679883 | 0.0 | 1.1555520317835992 | 2.919200259736721 |
| 0.6 | 0.5 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 0.5365455074349632 | 0.39929488599940255 | 0.25 | 1.1555520317835992 | 5.886049833528683 |
| 0.6 | 0.5 | 80.0 | 780.0 | 0.018 | 0.127 | 0.5 | 0.5365455074349632 | 0.39929488599940255 | 0.14705882352941177 | 1.1555520317835992 | 5.886049833528683 |
| 0.6 | 0.5 | 80.0 | 780.0 | 0.018 | 0.127 | 0.0 | 0.5365455074349632 | 0.39929488599940255 | 0.0 | 1.1555520317835992 | 5.886049833528683 |
| 1.0 | 4.0 | 15.0 | 820.0 | 0.025 | 0.1 | 1.0 | 0.19920214741350614 | 0.16128350589230608 | 0.25 | 3.000061830165984 | 4.2109659409943 |
| 1.0 | 4.0 | 15.0 | 820.0 | 0.025 | 0.1 | 0.5 | 0.19920214741350614 | 0.16128350589230608 | 0.14705882352941177 | 3.000061830165984 | 4.2109659409943 |
| 1.0 | 4.0 | 15.0 | 820.0 | 0.025 | 0.1 | 0.0 | 0.19920214741350614 | 0.16128350589230608 | 0.0 | 3.000061830165984 | 4.2109659409943 |
| 4.0 | 3.0 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 0.5699414407581949 | 0.46605658362315494 | 0.25 | 1.1555520317835992 | 6.007196043596891 |
| 4.0 | 3.0 | 80.0 | 780.0 | 0.018 | 0.127 | 0.5 | 0.5699414407581949 | 0.46605658362315494 | 0.14705882352941177 | 1.1555520317835992 | 6.007196043596891 |
| 4.0 | 3.0 | 80.0 | 780.0 | 0.018 | 0.127 | 0.0 | 0.5699414407581949 | 0.46605658362315494 | 0.0 | 1.1555520317835992 | 6.007196043596891 |
| 10.0 | 0.3 | 40.0 | 820.0 | 0.02 | 0.15 | 1.0 | 0.9691057964673919 | 0.7969455792418021 | 0.25 | 1.7238243721388375 | 7.855060827319523 |
| 10.0 | 0.3 | 40.0 | 820.0 | 0.02 | 0.15 | 0.5 | 0.9691057964673919 | 0.7969455792418021 | 0.14705882352941177 | 1.7238243721388375 | 7.855060827319523 |
| 10.0 | 0.3 | 40.0 | 820.0 | 0.02 | 0.15 | 0.0 | 0.9691057964673919 | 0.7969455792418021 | 0.0 | 1.7238243721388375 | 7.855060827319523 |
| 2.45 | 1.0 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 0.7064050992237105 | 0.5667821020866667 | 0.25 | 1.1555520317835992 | 6.464374907964801 |
| 2.45 | 1.0 | 80.0 | 780.0 | 0.018 | 0.127 | 0.5 | 0.7064050992237105 | 0.5667821020866667 | 0.14705882352941177 | 1.1555520317835992 | 6.464374907964801 |
| 2.45 | 1.0 | 80.0 | 780.0 | 0.018 | 0.127 | 0.0 | 0.7064050992237105 | 0.5667821020866667 | 0.0 | 1.1555520317835992 | 6.464374907964801 |

### SLIP-13, SLIP-14

| v_gs | v_ls | rho_g | rho_l | sigma | D | cos_incl | → p_annular | → p_churn | → p_dispersed | → p_slug | → p_bubbly |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 1.0 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 0.0 | 6.8859653009302565e-25 | 1.2490009026344414e-14 | 2.108185913873824e-05 | 0.9999789181408488 |
| 0.05 | 1.0 | 80.0 | 780.0 | 0.018 | 0.127 | 0.5 | 0.0 | 6.8859653009302565e-25 | 1.2490009026344414e-14 | 0.0036114867042020375 | 0.9963885132957855 |
| 0.05 | 1.0 | 80.0 | 780.0 | 0.018 | 0.127 | 0.0 | 0.0 | 6.8859653009302565e-25 | 1.2490009026344414e-14 | 0.8497859782102325 | 0.1502140217897551 |
| 0.6 | 0.5 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.0251126022540714e-14 | 0.0 | 0.0 | 0.9994273965083569 | 0.0005726034916329001 |
| 0.6 | 0.5 | 80.0 | 780.0 | 0.018 | 0.127 | 0.5 | 1.0251126022540714e-14 | 0.0 | 0.0 | 0.9999966675622105 | 3.332437779279795e-06 |
| 0.6 | 0.5 | 80.0 | 780.0 | 0.018 | 0.127 | 0.0 | 1.0251126022540714e-14 | 0.0 | 0.0 | 0.9999999978648726 | 2.1351171941574657e-09 |
| 1.0 | 4.0 | 15.0 | 820.0 | 0.025 | 0.1 | 1.0 | 4.444719673927172e-26 | 1.1236463936817087e-07 | 0.999914549282979 | 9.990240555628451e-07 | 8.433932832610691e-05 |
| 1.0 | 4.0 | 15.0 | 820.0 | 0.025 | 0.1 | 0.5 | 4.444719673927172e-26 | 1.1236463936817087e-07 | 0.999914549282979 | 5.7234196858776535e-05 | 2.8104155522893223e-05 |
| 1.0 | 4.0 | 15.0 | 820.0 | 0.025 | 0.1 | 0.0 | 4.444719673927172e-26 | 1.1236463936817087e-07 | 0.999914549282979 | 8.531151248681982e-05 | 2.683989484992962e-08 |
| 4.0 | 3.0 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 0.0014967998499312296 | 0.9249702610102476 | 0.0732756245817744 | 0.00025730932389370136 | 5.234153095637069e-09 |
| 4.0 | 3.0 | 80.0 | 780.0 | 0.018 | 0.127 | 0.5 | 0.0014967998499312296 | 0.9249702610102476 | 0.0732756245817744 | 0.0002573145276017984 | 3.044499863110672e-11 |
| 4.0 | 3.0 | 80.0 | 780.0 | 0.018 | 0.127 | 0.0 | 0.0014967998499312296 | 0.9249702610102476 | 0.0732756245817744 | 0.00025731455802729074 | 1.9506280299889485e-14 |
| 10.0 | 0.3 | 40.0 | 820.0 | 0.02 | 0.15 | 1.0 | 0.9999985663562205 | 1.4336435294260745e-06 | 2.440781812875244e-16 | 2.497991542287602e-13 | 3.3174546253271742e-25 |
| 10.0 | 0.3 | 40.0 | 820.0 | 0.02 | 0.15 | 0.5 | 0.9999985663562205 | 1.4336435294260745e-06 | 2.440781812875244e-16 | 2.4979915422909005e-13 | 1.941329407899199e-27 |
| 10.0 | 0.3 | 40.0 | 820.0 | 0.02 | 0.15 | 0.0 | 0.9999985663562205 | 1.4336435294260745e-06 | 2.440781812875244e-16 | 2.4979915422909197e-13 | 0.0 |
| 2.45 | 1.0 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 0.5793863868864486 | 3.148806585333301e-11 | 2.714607543071048e-15 | 0.42061355748557727 | 5.559648334635926e-08 |
| 2.45 | 1.0 | 80.0 | 780.0 | 0.018 | 0.127 | 0.5 | 0.5793863868864486 | 3.148806585333301e-11 | 2.714607543071048e-15 | 0.42061361275868436 | 3.2337625462546083e-10 |
| 2.45 | 1.0 | 80.0 | 780.0 | 0.018 | 0.127 | 0.0 | 0.5793863868864486 | 3.148806585333301e-11 | 2.714607543071048e-15 | 0.4206136130818534 | 2.0719677098991507e-13 |

### SLIP-15, SLIP-16

| v_g | v_l | alpha | rho_g | rho_l | sigma | D | cos_incl | → C_0 | → v_inf | → regime |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 1.25 | 0.2 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.2 | 0.18265279710346274 | bubbly |
| 0.25 | 1.25 | 0.2 | 80.0 | 780.0 | 0.018 | 0.127 | 0.0 | 1.2 | 0.027436417860562515 | slug |
| 0.1 | 2.0 | 0.5 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.2 | 0.18265279710346274 | bubbly |
| 0.1 | 2.0 | 0.5 | 80.0 | 780.0 | 0.018 | 0.127 | 0.0 | 1.2 | 0.027436417860562515 | slug |
| 2.9999999999999996 | 0.625 | 0.2 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.199999999999998 | 0.3699184690097599 | slug |
| 2.9999999999999996 | 0.625 | 0.2 | 80.0 | 780.0 | 0.018 | 0.127 | 0.0 | 1.199999999999998 | 3.899766934019099e-10 | slug |
| 1.2 | 1.0 | 0.5 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.199999999999998 | 0.3699184690097599 | slug |
| 1.2 | 1.0 | 0.5 | 80.0 | 780.0 | 0.018 | 0.127 | 0.0 | 1.199999999999998 | 3.899766934019099e-10 | slug |
| 5.0 | 5.0 | 0.2 | 15.0 | 820.0 | 0.025 | 0.1 | 1.0 | 1.1999999943817679 | 0.2002622036733616 | dispersed-bubbly |
| 5.0 | 5.0 | 0.2 | 15.0 | 820.0 | 0.025 | 0.1 | 0.0 | 1.1999999943817679 | 0.20024493741437754 | dispersed-bubbly |
| 2.0 | 8.0 | 0.5 | 15.0 | 820.0 | 0.025 | 0.1 | 1.0 | 1.1999999943817679 | 0.2002622036733616 | dispersed-bubbly |
| 2.0 | 8.0 | 0.5 | 15.0 | 820.0 | 0.025 | 0.1 | 0.0 | 1.1999999943817679 | 0.20024493741437754 | dispersed-bubbly |
| 20.0 | 3.75 | 0.2 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.153452126979501 | 0.35574174573333633 | churn |
| 20.0 | 3.75 | 0.2 | 80.0 | 780.0 | 0.018 | 0.127 | 0.0 | 1.153452126979501 | 0.013383708332058998 | churn |
| 8.0 | 6.0 | 0.5 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.153452126979501 | 0.35574174573333633 | churn |
| 8.0 | 6.0 | 0.5 | 80.0 | 780.0 | 0.018 | 0.127 | 0.0 | 1.153452126979501 | 0.013383708332058998 | churn |
| 50.0 | 0.37499999999999994 | 0.2 | 40.0 | 820.0 | 0.02 | 0.15 | 1.0 | 1.0000002150465794 | 5.935477205094346e-07 | annular |
| 50.0 | 0.37499999999999994 | 0.2 | 40.0 | 820.0 | 0.02 | 0.15 | 0.0 | 1.0000002150465794 | 4.586430128444612e-17 | annular |
| 20.0 | 0.6 | 0.5 | 40.0 | 820.0 | 0.02 | 0.15 | 1.0 | 1.0000002150465794 | 5.935477205094346e-07 | annular |
| 20.0 | 0.6 | 0.5 | 40.0 | 820.0 | 0.02 | 0.15 | 0.0 | 1.0000002150465794 | 4.586430128444612e-17 | annular |
| 12.25 | 1.25 | 0.2 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.0841227226211358 | 0.15563786215011302 | annular |
| 12.25 | 1.25 | 0.2 | 80.0 | 780.0 | 0.018 | 0.127 | 0.0 | 1.0841227226211358 | 3.834007122968883e-14 | annular |
| 4.9 | 2.0 | 0.5 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.0841227226211358 | 0.15563786215011302 | annular |
| 4.9 | 2.0 | 0.5 | 80.0 | 780.0 | 0.018 | 0.127 | 0.0 | 1.0841227226211358 | 3.834007122968883e-14 | annular |

### SLIP-18

| profile_scale | v_g | v_l | alpha | rho_g | rho_l | sigma | D | cos_incl | → C_0 | → v_inf |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.16666666666666669 | 1.4285714285714286 | 0.3 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.0 | 0.18265279710346274 |
| 0.5 | 0.16666666666666669 | 1.4285714285714286 | 0.3 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.1 | 0.18265279710346274 |
| 1.0 | 0.16666666666666669 | 1.4285714285714286 | 0.3 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.2 | 0.18265279710346274 |
| 1.5 | 0.16666666666666669 | 1.4285714285714286 | 0.3 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.2999999999999998 | 0.18265279710346274 |
| 0.0 | 2.0 | 0.7142857142857143 | 0.3 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.0 | 0.3699184690097599 |
| 0.5 | 2.0 | 0.7142857142857143 | 0.3 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.099999999999999 | 0.3699184690097599 |
| 1.0 | 2.0 | 0.7142857142857143 | 0.3 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.199999999999998 | 0.3699184690097599 |
| 1.5 | 2.0 | 0.7142857142857143 | 0.3 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.299999999999997 | 0.3699184690097599 |
| 0.0 | 3.3333333333333335 | 5.714285714285714 | 0.3 | 15.0 | 820.0 | 0.025 | 0.1 | 1.0 | 1.0 | 0.2002622036733616 |
| 0.5 | 3.3333333333333335 | 5.714285714285714 | 0.3 | 15.0 | 820.0 | 0.025 | 0.1 | 1.0 | 1.099999997190884 | 0.2002622036733616 |
| 1.0 | 3.3333333333333335 | 5.714285714285714 | 0.3 | 15.0 | 820.0 | 0.025 | 0.1 | 1.0 | 1.1999999943817679 | 0.2002622036733616 |
| 1.5 | 3.3333333333333335 | 5.714285714285714 | 0.3 | 15.0 | 820.0 | 0.025 | 0.1 | 1.0 | 1.2999999915726517 | 0.2002622036733616 |
| 0.0 | 13.333333333333334 | 4.285714285714286 | 0.3 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.0 | 0.35574174573333633 |
| 0.5 | 13.333333333333334 | 4.285714285714286 | 0.3 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.0767260634897506 | 0.35574174573333633 |
| 1.0 | 13.333333333333334 | 4.285714285714286 | 0.3 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.153452126979501 | 0.35574174573333633 |
| 1.5 | 13.333333333333334 | 4.285714285714286 | 0.3 | 80.0 | 780.0 | 0.018 | 0.127 | 1.0 | 1.2301781904692515 | 0.35574174573333633 |
| 0.0 | 33.333333333333336 | 0.4285714285714286 | 0.3 | 40.0 | 820.0 | 0.02 | 0.15 | 1.0 | 1.0 | 5.935477205094346e-07 |
| 0.5 | 33.333333333333336 | 0.4285714285714286 | 0.3 | 40.0 | 820.0 | 0.02 | 0.15 | 1.0 | 1.0000001075232898 | 5.935477205094346e-07 |
| 1.0 | 33.333333333333336 | 0.4285714285714286 | 0.3 | 40.0 | 820.0 | 0.02 | 0.15 | 1.0 | 1.0000002150465794 | 5.935477205094346e-07 |
| 1.5 | 33.333333333333336 | 0.4285714285714286 | 0.3 | 40.0 | 820.0 | 0.02 | 0.15 | 1.0 | 1.000000322569869 | 5.935477205094346e-07 |

### SLIP-18 (fitted)

| profile_scale | v_g | v_l | alpha | rho_g | rho_l | sigma | D | cos_incl | → C_0 | → v_inf |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.2 | 1.0 | 0.15 | 60.0 | 780.0 | 0.022 | 0.1524 | 1.0 | 1.0 | 0.19340633028659143 |
| 0.5 | 1.2 | 1.0 | 0.15 | 60.0 | 780.0 | 0.022 | 0.1524 | 1.0 | 1.0999998058609524 | 0.19340633028659143 |
| 1.0 | 1.2 | 1.0 | 0.15 | 60.0 | 780.0 | 0.022 | 0.1524 | 1.0 | 1.1999996117219047 | 0.19340633028659143 |
| 1.5 | 1.2 | 1.0 | 0.15 | 60.0 | 780.0 | 0.022 | 0.1524 | 1.0 | 1.299999417582857 | 0.19340633028659143 |
| 0.0 | 6.0 | 2.0 | 0.4 | 80.0 | 800.0 | 0.02 | 0.1524 | 1.0 | 1.0 | 0.37388375219412906 |
| 0.5 | 6.0 | 2.0 | 0.4 | 80.0 | 800.0 | 0.02 | 0.1524 | 1.0 | 1.0891630571407123 | 0.37388375219412906 |
| 1.0 | 6.0 | 2.0 | 0.4 | 80.0 | 800.0 | 0.02 | 0.1524 | 1.0 | 1.1783261142814245 | 0.37388375219412906 |
| 1.5 | 6.0 | 2.0 | 0.4 | 80.0 | 800.0 | 0.02 | 0.1524 | 1.0 | 1.2674891714221368 | 0.37388375219412906 |
| 0.0 | 25.0 | 3.0 | 0.85 | 40.0 | 820.0 | 0.018 | 0.1524 | 1.0 | 1.0 | 0.01578881358848783 |
| 0.5 | 25.0 | 3.0 | 0.85 | 40.0 | 820.0 | 0.018 | 0.1524 | 1.0 | 1.003312444869717 | 0.01578881358848783 |
| 1.0 | 25.0 | 3.0 | 0.85 | 40.0 | 820.0 | 0.018 | 0.1524 | 1.0 | 1.006624889739434 | 0.01578881358848783 |
| 1.5 | 25.0 | 3.0 | 0.85 | 40.0 | 820.0 | 0.018 | 0.1524 | 1.0 | 1.0099373346091511 | 0.01578881358848783 |
<!-- vectors:end develop -->

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| SLIP-1 | (8) | `simulator.py` `_closure_relations` (`g1`) | rows |
| SLIP-2 | (A.9), Table A.6 | `slip.py` `SlipModel.identify_parameters` | vectors; rows |
| SLIP-3 | (A.10), corrected | `slip.py` `SlipModel.identify_parameters` | vectors; rows |
| SLIP-4 | (A.1) | `slip.py` `SlipModel.harmathy_rise_velocity` | vectors |
| SLIP-5 | (A.2) | `slip.py` `SlipModel.taylor_rise_velocity` | vectors |
| SLIP-6 | (A.8) | `slip.py` `classify_flow_regime` | vectors |
| SLIP-7 | (A.7) | `slip.py` `classify_flow_regime` | vectors |
| SLIP-8 | Table 3 (`FRBH`, `FRWH`) | `slip.py` `SlipModel.flow_regime` | vectors |
| SLIP-9 | (A.3)–(A.6) | `slip.py` `classify_flow_regime` (docstring) | spec-only: the fitting target of SLIP-7's weights, which the model never evaluates |
| SLIP-10 | — | — | vectors; property: tests/test_slip.py |
| SLIP-11 | — | — | vectors; property: tests/test_slip.py |
| SLIP-12 | — | — | vectors |
| SLIP-13 | — | — | vectors; property: tests/test_slip.py |
| SLIP-14 | — | — | vectors; property: tests/test_slip.py |
| SLIP-15 | — | — | vectors; property: tests/test_slip.py |
| SLIP-16 | — | — | vectors |
| SLIP-17 | — | — | spec-only: the crisp limit of SLIP-13 and SLIP-14, which the tests compare the labels with and the model never evaluates |
| SLIP-18 | — | — | vectors; property: tests/test_slip.py |
