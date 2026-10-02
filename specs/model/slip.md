# Slip

## Purpose

The drift-flux slip law that sets the gas velocity relative to the mixture, with parameters that depend on the flow regime. A smooth classifier gives the probability of each of three regimes (bubbly, slug/churn, annular), and the slip parameters are the probability-weighted averages of each regime's values (paper §2.2 and Appendix A, after Hasan, Kabir and Sayarpour, 2010).

## Interface

| Function | Inputs | Outputs | Symbolic |
|---|---|---|---|
| slip row | $v_g, v_l, \alpha, \rho_g, \rho_l$ (state), $\sigma$, $D$ | row (m/s) | yes |
| slip parameters | $v_g, v_l, \alpha, \rho_g, \rho_l, \sigma, D$ | $C_0$ (–), $v_\infty$ (m/s) | yes |
| classifier | $v_g, v_l, \alpha, \rho_g, \rho_l, \sigma$ | $(p_a, p_s, p_b)$ (–) | yes |
| regime label | the classifier's inputs, as floats | `annular`, `slug-churn` or `bubbly` | no |

$p_a$, $p_s$ and $p_b$ are the probabilities of annular, slug/churn and bubbly flow. $\sigma$ comes from PVT-MIX-5 (or PVT-MIX-7 in `develop`'s default) and is evaluated at the same point. v1.0.0's functions take $T$ and compute $\sigma$ themselves; `develop`'s take $\sigma$ and the inclination $\cos\theta$ of the point's cell (DISC-11). The vectors below give $\sigma$.

In `develop` the four regime constants of SLIP-2 and SLIP-3 (1.0, 1.175 and 1.2 for $C_0$, and 0 for the annular drift velocity) are parameters of the slip model, fields of `SlipModel` with these values as defaults (`plans/improvements.md` §2.7). The `v1.0.0` configuration uses the defaults.

## Equations

### SLIP-1 · Slip law

$$v_g = C_0\, v_m + v_\infty$$

with $v_m$ from BAL-8. Its row in DISC-6 is $v_g - C_0 v_m - v_\infty$ (m/s), at every grid point.

### SLIP-2 · Profile parameter

$$C_0 = 1.0\, p_a + 1.175\, p_s + 1.2\, p_b$$

The regime values are those of Table A.6: 1.2 for bubbly flow, 1.0 for annular flow, and for slug/churn flow the average of the values for slug (1.2) and churn (1.15) flow.

### SLIP-3 · Drift velocity

$$v_\infty = 0 \cdot p_a + v_{\infty T}\, p_s + v_{\infty b}\, p_b$$

This is (A.10) as corrected in `docs/corrigendum.md`: the paper swaps the two rise velocities, and the code does not.

### SLIP-4 · Bubble rise velocity (Harmathy)

$$v_{\infty b} = 1.53 \left(\frac{g\,\sigma\,(\rho_l - \rho_g)}{\rho_l^2}\right)^{1/4}$$

### SLIP-5 · Taylor-bubble rise velocity

$$v_{\infty T} = 0.35\,\sqrt{g\,D\,(1 - \rho_g/\rho_l)}$$

for a vertical pipe; SLIP-10 corrects it for inclination.

### SLIP-6 · Classifier features

$$
c_1 = \tanh\!\Big(v_{gs} - 3.1\,\big(g\,\sigma\,(\rho_l - \rho_g)/\rho_g^2\big)^{1/4}\Big), \quad
c_2 = \tanh\!\big(2(\alpha - 0.7)\big), \quad
c_3 = \tanh\!\big(v_{gs} - 1.08\, v_{ls}\big), \quad
c_4 = \tanh\!\big(2(\alpha - 0.25)\big)
$$

with $v_{gs} = \alpha v_g$ and $v_{ls} = (1 - \alpha) v_l$ in m/s. The arguments of $c_1$ and $c_3$ are velocities in m/s, not dimensionless. The factor 2 in $c_2$ and $c_4$ is in v1.0.0's code and not in the paper's (A.8) (`specs/discrepancies.md`, D-1). The paper lists the features in the reverse order.

### SLIP-7 · Classifier output

$$
\begin{aligned}
y_a &= \phantom{-}3.17715258\,c_1 + 6.81938489\,c_2 + 0.30182974\,c_3 + 3.58362465\,c_4 - 3.92904391\\
y_s &= -1.47973427\,c_1 - 4.34033317\,c_2 + 2.58200006\,c_3 + 3.49656911\,c_4 - 1.46509477\\
y_b &= -1.6974183\,c_1 - 2.47905172\,c_2 - 2.8838298\,c_3 - 7.08019376\,c_4 + 5.39413869
\end{aligned}
$$

$$(p_a, p_s, p_b) = \operatorname{softmax}(y_a, y_s, y_b)$$

with the softmax of SMO-3. This is (A.7) with v1.0.0's weights $A$ and $b$, which the paper does not list; the paper orders the outputs (bubbly, slug/churn, annular).

### SLIP-8 · Regime label

The label of a point is the regime with the highest probability: `annular` if $p_a > p_s$ and $p_a > p_b$; else `slug-churn` if $p_s > p_a$ and $p_s > p_b$; else `bubbly`, which also takes ties. It is computed after the solve, from the root, and is the `FRBH` and `FRWH` feature of the datasets (at points 0 and $N$). It is not part of the discretized system.

### SLIP-9 · Reference regime hierarchy

The hierarchy that the classifier approximates, from (A.3)–(A.6): the flow is annular if $\alpha \ge 0.7$ and $v_{gs} \ge 3.1\,\big(g\sigma(\rho_l - \rho_g)/\rho_g^2\big)^{1/4}$; else slug/churn if $\alpha \ge 0.25$ and $v_{gs} \ge 1.08\,v_{ls}$; else bubbly. The weights of SLIP-7 were fitted to it by multinomial logistic regression on sampled points (scikit-learn, L2 penalty, $C = 0.01$). The model never evaluates the hierarchy itself.

### SLIP-10 · Deviation factor of the Taylor-bubble rise velocity

In an inclined cell, the Taylor-bubble rise velocity of SLIP-5 is multiplied by

$$\sqrt{\cos\theta}\,(1 + \sin\theta)^{1.2}, \qquad \sin\theta = \sqrt{1 - \cos^2\theta},$$

Hasan, Kabir and Sayarpour (2010), Eq. (A-10), with $\theta$ the inclination from vertical. The factor is exactly 1 in a vertical cell, and 0 in a horizontal one. `develop` had $\sqrt{\cos\theta + 10^{-9}}$, a guard against a negative argument that GEO-3 already rules out ($\cos\theta \ge 0$); Step 7 dropped it, so the vertical case is exact (`plans/develop_model_changes.md`, change 2; decided by Bjarne, 2026-10-01).

### SLIP-11 · Bubbly–slug threshold in an inclined cell

The fourth classifier feature of SLIP-6 becomes

$$c_4 = \tanh\!\big(2(\alpha - 0.25\cos\theta)\big),$$

so the transition from bubbly to slug flow comes at a lower void fraction in an inclined cell. The weights of SLIP-7 are unchanged, and were fitted for the vertical threshold. At $\cos\theta = 1$ it is SLIP-6's $c_4$.

## Options

| Option | IDs | Used by |
|---|---|---|
| Three regimes, vertical pipe | SLIP-1 to SLIP-8 | `v1.0.0` |
| Inclination: bubbly–slug threshold $0.25\cos\theta$, deviation factor on $v_{\infty T}$ | SLIP-10, SLIP-11, with SLIP-1 to SLIP-8 | `develop`; on a vertical grid it is the `v1.0.0` option exactly |
| Four regimes (bubbly, slug, churn, annular) | after the plan | — |

## Safeguards

- None in the equations: no clipping, and nothing guards the fourth roots and the square root against negative arguments, which occur only if $\rho_g > \rho_l$.
- The factor 2 in $c_2$ and $c_4$ steepens the transitions in $\alpha$ ("to increase sensitivity", v1.0.0's comment).
- The softmax is evaluated without shifting the logits. $|c_k| \le 1$ bounds the logits to $|y| < 22$, so it cannot overflow.

## Open question: several void fractions

At fixed superficial velocities $j_g = \alpha v_g$ and $j_l = (1 - \alpha) v_l$, which the mass rows fix at a point, SLIP-1 is one equation in $\alpha$: $h(\alpha) = \alpha\,(C_0 j_m + v_\infty) - j_g = 0$, which is $-\alpha$ times its row. The classifier sees $\alpha$ only through $c_2$ and $c_4$, and $C_0 \ge 1$ and $v_\infty \ge 0$ for every mix of the regimes. So $h(0) < 0 < h(1)$, and a root always exists, but it need not be unique.
- **Where it occurs.** Near the slug–annular transition with little liquid there can be three roots. In Step 8's regenerated `sol-1` samples, at well 44 (k = 4: $u$ = 0.056, 3.8 kg/s of lift gas, $j_l$ = 0.016 m/s), point 95 has $\alpha$ = 0.662, 0.915 and 0.920.
- **Consequence for the root set.** SOL-2 then holds roots that differ only in the branch at some points, and which branch is physical is not specified (`solution.md`, informative section).
- **The case set.** At every point of every reference root, the root is unique (`tests/test_rust_backend.py`).
- **Consequence for the Rust core.** Where the void fraction a march takes switches branch between neighbouring $p_0$, $R(p_0)$ jumps across zero, and the core can accept the jump as a root: at `v1.0.0+deviated#17` of Step 9's comparison set, its choke row is $-7.5 \cdot 10^{-4}$ kg/s, under the acceptance bound of $10^{-3} w_m$. A known finding, not changed (Bjarne, 2026-10-02, `specs/features/015-rust-develop-model.md`, Finding 3).
- **Status.** Open (Bjarne, 2026-10-01). It is to be ruled after the plan, with the new flow-regime model of `specs/goals.md`, which replaces this classifier.

## Sources

- Paper (8), Appendix A.1–A.2, Table A.6, and `docs/corrigendum.md` for (A.10).
- Hasan, Kabir and Sayarpour (2010), "Simplified two-phase flow modeling in wellbores", *Journal of Petroleum Science and Engineering* 72, 42–49: Eqs. (A-18) and (A-19) for the transitions, (A-10) for SLIP-10.
- Feature spec `specs/features/002-slip-inclination.md`.
- Harmathy (1960), "Velocity of large drops and bubbles in media of infinite or restricted extent", *AIChE Journal* 6, 281–288.

## Test vectors

The $\sigma$ column is $\sigma_{od}(\rho_l, T)$ (PVT-MIX-5), and the `T` column records the temperature it was computed at; the functions under test take $\sigma$, not $T$.

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
