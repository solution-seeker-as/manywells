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

$p_a$, $p_s$ and $p_b$ are the probabilities of annular, slug/churn and bubbly flow. $\sigma$ comes from PVT-MIX-5 and is evaluated at the same point. v1.0.0's functions take $T$ and compute $\sigma$ themselves; `develop`'s take $\sigma$. The vectors below give $\sigma$.

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

for a vertical pipe.

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

## Options

| Option | IDs | Used by |
|---|---|---|
| Three regimes, vertical pipe | SLIP-1 to SLIP-8 | `v1.0.0` |
| Inclination: bubbly–slug threshold $0.25\cos\theta$, deviation factor on $v_{\infty T}$ | Step 7 | `develop` |
| Four regimes (bubbly, slug, churn, annular) | after the plan | — |

## Safeguards

- None in the equations: no clipping, and nothing guards the fourth roots and the square root against negative arguments, which occur only if $\rho_g > \rho_l$.
- The factor 2 in $c_2$ and $c_4$ steepens the transitions in $\alpha$ ("to increase sensitivity", v1.0.0's comment).
- The softmax is evaluated without shifting the logits. $|c_k| \le 1$ bounds the logits to $|y| < 22$, so it cannot overflow.

## Sources

- Paper (8), Appendix A.1–A.2, Table A.6, and `docs/corrigendum.md` for (A.10).
- Hasan, Kabir and Sayarpour (2010), "Simplified two-phase flow modeling in wellbores", *Journal of Petroleum Science and Engineering* 72, 42–49: Eqs. (A-18) and (A-19) for the transitions.
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
