# Choke

## Purpose

The production choke at the wellhead, which sets the top boundary condition: the rate leaving the tubing must pass through the choke, from the wellhead pressure $p_N$ to the downstream pressure $p_s$. Includes critical (choked) flow and the choke profiles (paper §2.3).

## Interface

| Function | Inputs | Output | Symbolic |
|---|---|---|---|
| choke rate | $u$, $p_N$ (bar), $p_s$ (bar), the model's density and multiplier; $K_c$ (m²), profile | $w_c$ (kg/s) | yes, in $p_N$ and the state |
| choke row | state at point $N$, $u$, $p_s$ | row (kg/s) | yes |
| choke profile | $u \in [0, 1]$ | $\sigma(u) \in [0, 1]$ | no |
| choked flag | $p_N$, $p_s$ (bar) | boolean | no |

## Equations

### CHK-1 · Choke row

$$r = w_m(z_N) - w_c \quad \text{[kg/s]}$$

where $w_m(z_N) = A\alpha_N\rho_{g,N}v_{g,N} + A(1-\alpha_N)\rho_{l,N}v_{l,N}$ is the rate leaving the tubing and $w_c$ the rate the choke passes (CHK-2). This is the row that SOL-3 removes to define the shooting residual.

### CHK-2 · Choke equation

$$w_c = K_c\,\sigma(u)\,\sqrt{\frac{2\rho\,\Delta p}{\Phi}}, \qquad \Delta p = c_\text{bar}\,(p_N - p_c)\ \text{[Pa]}$$

$K_c > 0$ is the choke coefficient, $\sigma$ the choke profile (CHK-7 to CHK-10), $p_c$ the effective downstream pressure (CHK-3), and $\rho$ and $\Phi$ the density and two-phase multiplier of the choke model (CHK-5, CHK-6). The paper's (11) writes $\sqrt{2\rho_e\Delta p}$ with $\rho_e = \rho/\Phi$. v1.0.0's class docstring puts $\Phi$ outside the square root; the code and this equation have it inside (the docstring is fixed on `develop`).

### CHK-3 · Critical downstream pressure

$$p_c = \operatorname{smax}(r_c\, p_N,\ p_s)$$

with the smooth max of SMO-1 on pressures in bar, $\epsilon = 10^{-6}$ bar². When $p_s \le r_c p_N$ the flow is critical and the rate no longer depends on $p_s$ (paper (14)). The smooth max exceeds the exact max by at most $\sqrt{\epsilon}/2 = 5\cdot10^{-4}$ bar, at $r_c p_N = p_s$.

### CHK-4 · Critical pressure ratio

$$r_c = \left(\frac{2}{\gamma + 1}\right)^{\gamma/(\gamma - 1)}, \qquad \gamma = 1.307,$$

which gives $r_c = 0.5445$. $\gamma$ is the heat capacity ratio of methane at 20 °C, the same for every well.

### CHK-5 · Simpson choke

$$\rho = \rho_l, \qquad \Phi = \big(1 + x_g(k - 1)\big)\big(1 + x_g(k^5 - 1)\big), \qquad k = (\rho_l/\rho_g)^{1/6}$$

evaluated at point $N$, with $x_g = w_g/w_m$ there; $w_g$ includes the lift gas (`specs/discrepancies.md`, D-23). This is the paper's (12): with Simpson's slip factor $k$, $\rho_l/\Phi = \rho_e$, where $1/\rho_e = (x_g/\rho_g + k\,x_l/\rho_l)(x_g + x_l/k)$ and $x_l = 1 - x_g$. Used by the published datasets.

### CHK-6 · Bernoulli choke

$$\rho = \rho_m, \qquad \Phi = 1$$

evaluated at point $N$. Not in the paper. v1.0.0's default choke, with $K_c = 0.1A$.

### CHK-7 · Linear profile

$$\sigma_l(u) = u$$

### CHK-8 · Sigmoid profile

$$\sigma_s(u) = \frac{u^b}{u^b + (1 - u)^b}, \qquad b = 3/2$$

### CHK-9 · Convex profile

$$\sigma_c(u) = b\,u + (1 - b)\,u^2, \qquad b = 1/4$$

### CHK-10 · Concave profile

$$\sigma_q(u) = u^b, \qquad b = 3/4$$

A quick-opening valve. Every profile has $\sigma(0) = 0$ and $\sigma(1) = 1$.

### CHK-11 · Choke row where $\Delta p \le 0$

Decided by Bjarne, 2026-09-30 (`specs/discrepancies.md`, D-19). Where $p_N \le p_c$ the choke passes no flow from the well:

$$w_c = K_c\,\sigma(u)\,\sqrt{\frac{2\rho\,\max(\Delta p, 0)}{\Phi}},$$

so the CHK-1 row is $w_m(z_N) > 0$ there. The row is then defined for every admissible state, continuous at $\Delta p = 0$, and positive wherever $\Delta p \le 0$. An implementation may use any row with the sign of $w_m - w_c$ everywhere; the Rust port's squared row $w_m^2 - (K_c\sigma(u))^2\, 2\rho\,\Delta p/\Phi$ qualifies.

The root set does not depend on this choice: a root has $w_m > 0$, so its $\Delta p > 0$ (SOL-2). The stability label does not either (SOL-3). The choice matters to solvers that evaluate the row near $p_N = p_s$, such as a shooting method that brackets the trickle root. v1.0.0's row is NaN there, the square root of a negative number, so the `v1.0.0` configuration extends v1.0.0 here; no root reaches this region, so its roots are unchanged.

### CHK-12 · Choked flag

The flow is choked if

$$p_s \le r_c\, p_N,$$

evaluated exactly, without smoothing, after the solve. It is the `CHOKED` feature of the datasets, and not part of the discretized system.

## Options

| Option | IDs | Used by |
|---|---|---|
| Simpson | CHK-5 | the published datasets |
| Bernoulli | CHK-6 | v1.0.0's default |
| Profiles | CHK-7, CHK-8, CHK-9, CHK-10 | all four in the datasets, drawn uniformly |

CHK-1 to CHK-4, CHK-11 and CHK-12 apply to every option, in every configuration.

## Safeguards

- The smooth max in CHK-3 (SMO-1), $\epsilon = 10^{-6}$ bar².
- Nothing guards the square root in CHK-2 against $\Delta p < 0$ in v1.0.0 (CHK-11).
- v1.0.0 asserts $K_c > 0$ and that the profile is one of the four. Its `cpr` constructor argument is ignored: `__post_init__` always sets it from CHK-4 (`plans/improvements.md` §1.4).

## Sources

- Paper (11)–(14), Fig. 2 and §2.3.
- Simpson, Rooney and Grattan (1983), "Two-phase flow through gate valves and orifice plates", International Conference on the Physical Modelling of Multi-Phase Flow, Coventry, 57–76.
- Haug (2012), *Multiphase flow through chokes*, Master's thesis, NTNU.

## Test vectors

<!-- vectors:begin -->
Generated by `specs/tools/make_v1_vectors.py` from ManyWells v1.0.0 (casadi 3.6.4). Do not edit by hand.

### CHK-2, CHK-3

| K_c | profile | u | p_in | p_out | rho | Phi | → w |
|---|---|---|---|---|---|---|---|
| 0.0014820614139757222 | linear | 0.5 | 50.0 | 40.0 | 850.0 | 1.0 | 30.553478737479036 |
| 0.0014820614139757222 | sigmoid | 0.8 | 50.0 | 20.0 | 850.0 | 2.5 | 51.845800733672505 |
| 0.0014820614139757222 | convex | 0.3 | 50.0 | 27.2232945913927 | 300.0 | 1.0 | 7.80723721877817 |
| 0.0014820614139757222 | concave | 1.0 | 20.01 | 20.0 | 850.0 | 4.0 | 0.9661845070490904 |
| 0.0007410307069878611 | sigmoid | 0.05 | 120.0 | 100.0 | 600.0 | 1.3 | 0.3798632328763746 |

### CHK-4

| gamma | → r_c |
|---|---|
| 1.307 | 0.544465891827854 |
| 1.3 | 0.545727733814065 |
| 1.4 | 0.5282817877171742 |
| 1.2 | 0.5644739300537772 |

### CHK-5 (multiplier)

| x_g | rho_g | rho_l | → Phi |
|---|---|---|---|
| 0.0 | 50.0 | 850.0 | 1.0 |
| 0.1 | 50.0 | 850.0 | 2.078466849170238 |
| 0.5 | 20.0 | 900.0 | 17.936584606679865 |
| 1.0 | 100.0 | 800.0 | 7.9999999999999964 |
| 0.02 | 150.0 | 950.0 | 1.0808539525881824 |

### CHK-5 (rate)

| K_c | profile | u | p_in | p_out | x_g | rho_g | rho_l | → w |
|---|---|---|---|---|---|---|---|---|
| 0.0014820614139757222 | sigmoid | 0.6 | 45.0 | 30.0 | 0.1 | 35.0 | 850.0 | 30.708587878237445 |
| 0.0014820614139757222 | linear | 0.9 | 60.0 | 15.0 | 0.4 | 45.0 | 900.0 | 35.69765097976452 |
| 0.0029641228279514444 | concave | 0.2 | 80.0 | 70.0 | 0.02 | 70.0 | 820.0 | 33.52071590505092 |

### CHK-6

| K_c | profile | u | p_in | p_out | rho_m | → w |
|---|---|---|---|---|---|---|
| 0.0014820614139757222 | linear | 0.7 | 40.0 | 25.0 | 400.0 | 35.93807927223845 |
| 0.0014820614139757222 | convex | 1.0 | 60.0 | 20.0 | 150.0 | 42.438781208692795 |

### CHK-7

| u | → sigma |
|---|---|
| 0.0 | 0.0 |
| 0.05 | 0.05 |
| 0.3 | 0.3 |
| 0.5 | 0.5 |
| 0.8 | 0.8 |
| 1.0 | 1.0 |

### CHK-8

| u | → sigma |
|---|---|
| 0.0 | 0.0 |
| 0.05 | 0.011930457848829514 |
| 0.3 | 0.2190952202344071 |
| 0.5 | 0.5 |
| 0.8 | 0.888888888888889 |
| 1.0 | 1.0 |

### CHK-9

| u | → sigma |
|---|---|
| 0.0 | 0.0 |
| 0.05 | 0.014375 |
| 0.3 | 0.14250000000000002 |
| 0.5 | 0.3125 |
| 0.8 | 0.6800000000000002 |
| 1.0 | 1.0 |

### CHK-10

| u | → sigma |
|---|---|
| 0.0 | 0.0 |
| 0.05 | 0.10573712634405642 |
| 0.3 | 0.4053600464421103 |
| 0.5 | 0.5946035575013605 |
| 0.8 | 0.8458970107524514 |
| 1.0 | 1.0 |

### CHK-12

| p_in | p_out | → choked |
|---|---|---|
| 50.0 | 20.0 | true |
| 50.0 | 27.2 | true |
| 50.0 | 27.25 | false |
| 50.0 | 30.0 | false |
<!-- vectors:end -->

## Coverage

| ID | Paper | v1.0.0 code | Checked by |
|---|---|---|---|
| CHK-1 | (11) | `simulator.py` `_right_boundary_eqs` (`g1`) | rows |
| CHK-2 | (11) | `choke.py` `ChokeModel.choke_equation` | vectors; rows |
| CHK-3 | (14), (B.1) | `choke.py` `ChokeModel.choke_equation` (`p_c`) | vectors; rows |
| CHK-4 | (13) | `choke.py` `ChokeModel.critical_pressure_ratio` | vectors |
| CHK-5 | (12) | `choke.py` `SimpsonChokeModel` | vectors; rows |
| CHK-6 | — | `choke.py` `BernoulliChokeModel` | vectors; rows |
| CHK-7 | §2.3 | `choke.py` `ChokeModel.choke_opening` | vectors; rows |
| CHK-8 | §2.3 | `choke.py` `ChokeModel.choke_opening` | vectors; rows |
| CHK-9 | §2.3 | `choke.py` `ChokeModel.choke_opening` | vectors |
| CHK-10 | §2.3 | `choke.py` `ChokeModel.choke_opening` | vectors |
| CHK-11 | — | none: the row is NaN for $\Delta p < 0$ | property: Step 7 checks the row's sign where $\Delta p \le 0$ on the implementation; spec-only: v1.0.0 has no value there to take vectors from, and no root reaches it |
| CHK-12 | (14), Table 3 (`CHOKED`) | `choke.py` `ChokeModel.is_choked` | vectors |
