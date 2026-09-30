# manywells_rs solver description

*Draft 2026-09-29. Status: draft. Describes the code on branch `rust_implementation` @ 0e9e98b; it is not a specification.*

This document describes how the Rust simulator (`manywells_rs`) solves the steady-state drift-flux model: the problem it solves, how the problem is reduced to one scalar equation, and each numerical subroutine with its stopping rules and failure behaviour. It is written as input to `manywells-v2-plan.md`:

- **Step 4** (model spec and discrepancy list): section 9 lists where the code departs from the paper.
- **Steps 2 and 8** (verifier and Rust pilot): section 8 states how closely a Rust solution satisfies each row of the v1 residual.
- **Step 6** (architecture): sections 3 and 4 give the call structure and the interfaces between the subroutines.

Known problems and proposed changes are in `solver_improvements.md`. Equation numbers in parentheses, such as (18), refer to the ManyWells paper (Grimstad et al., Geoenergy Science and Engineering 257, 2026). The physics files on `rust_implementation` are unchanged since v1.0.0, so "v1" below means both the tag and `manywells/simulator.py` on that branch; `develop` has changed them since (see `manywells-v2-plan.md`, Starting point).

## 1. Problem

### Unknowns

The well is discretized into grid points $z_i = i\,\Delta z$, $i = 0, \dots, N$, with $\Delta z = L/N$ and $N$ = `n_cells` (default 100). Each point carries the state of the paper's §3.1, $x(z_i) = (\alpha_g, \alpha_l, \rho_g, \rho_l, v_g, v_l, p, T)(z_i)$. The code stores seven numbers per point, in the order fixed by `DIM_X` (`simulator.rs:18`):

```
[p, v_g, v_l, alpha, rho_g, rho_l, T]      with alpha = α_g and α_l = 1 − alpha
```

A solution is a flat vector of length $7(N+1)$, ordered from the bottom ($z_0 = 0$) to the wellhead ($z_N = L$).

### Notation and units

| Symbol | Code | Unit | Meaning |
|---|---|---|---|
| $p$, $p_0$, $p_L$ | `p`, `p0` | bar | pressure; bottomhole pressure $p(z_0)$; wellhead pressure $p(z_N)$ |
| $\alpha$ | `alpha` | – | gas volume fraction $\alpha_g$ |
| $\rho_g$, $\rho_l$ | `rho_g`, `rho_l` | kg/m³ | gas and liquid density |
| $v_g$, $v_l$ | `v_g`, `v_l` | m/s | gas and liquid velocity |
| $T$ | `T` | K | temperature |
| $w_g$, $w_l$ | `w_g`, `w_l` | kg/s | gas and liquid mass rate, constant along the well |
| $v_m$ | `vm` | m/s | mixture velocity, $v_m = w_g/(A\rho_g) + w_l/(A\rho_l) = \alpha v_g + (1-\alpha) v_l$ |
| $A$ | `a` | m² | flow area $\pi D^2/4$ |
| $K_c$, $\sigma(u)$, $c_{pr}$ | `k_c`, `choke_opening`, `cpr` | m², –, – | choke coefficient, choke opening profile, critical pressure ratio |

Pressures are in bar in every function signature. `CF_PRES` = 10⁵ Pa/bar converts where SI is needed, for example in the kinetic term of the momentum flux and in the choke pressure drop.

### Inputs and supported models

| Input | Fields | Notes |
|---|---|---|
| `WellProperties` | `L, D, rho_l, R_s, cp_g, cp_l, f_D, h, inflow, choke` | `slip` is accepted for API parity and ignored; the slip model is built in. Defaults as in v1. |
| `BoundaryConditions` | `p_r, p_s, T_r, T_s, u, w_lg` | validated in the constructor (`simulator.rs:123`) |
| Inflow | `Vogel` (10), `ProductivityIndex` $w_l = k_l(p_r - p_0)$ | `FixedFlowRate` is not ported |
| Choke | `SimpsonChokeModel` (11)–(12), `BernoulliChokeModel` ($\rho = \rho_m$) | profiles linear, sigmoid, convex, concave |
| Fluid | ideal gas (9), constant $\rho_l$, dead-oil surface tension | surface tension uses the mixed-liquid density $\rho_l$ |
| Geometry | vertical well, constant $D$ and $f_D$, linear ambient temperature | no trajectory or inclination |

The Python wrapper (`python/manywells_rs/__init__.py`) converts `manywells` objects to the Rust classes on every `simulate()` call. It accepts `x_guess` for API parity but ignores it, and re-raises failures as `manywells.simulator.SimError` when `manywells` is installed.

### Output

`simulate()` returns a list of solutions, highest $p_0$ first, and raises `SimError("No valid bottomhole pressure found")` if the list would be empty. Each solution satisfies the equations in section 2 to the accuracy stated in section 8. `solution_as_df` adds $z$ and a flow-regime label per grid point. The Rust call releases the Python GIL while solving.

## 2. Reduction to one unknown

v1 solves all $7(N+1)$ equations at once with Ipopt. The Rust solver fixes the bottomhole pressure $p_0$ first. Once $p_0$ is fixed, every state at a grid point follows from the pressure there, the momentum balance links neighbouring pressures, and only the choke equation is left to determine $p_0$.

### Equations and how each is handled

| Equation | Rust handling | Code |
|---|---|---|
| Inflow (10), or PI | explicit, once per trial $p_0$ | `inflow.rs:18` |
| Mass balances (16)–(17) | exact by construction: $w_g$, $w_l$ fixed; $v_g = w_g/(A\alpha\rho_g)$, $v_l = w_l/(A(1-\alpha)\rho_l)$ | `simulator.rs:184-190` |
| Volume fractions (7) | exact: $\alpha_l = 1 - \alpha$ | implicit in the state layout |
| Gas law (9), constant $\rho_l$ | explicit: $\rho_g = 10^5 p/(R_s T)$ | `simulator.rs:192` |
| Slip law (8) with (A.1)–(A.10) | fixed-point iteration for $\alpha$ (section 4.7) | `simulator.rs:199` |
| Energy (4), (6), (15) | exact solution of the ODE, in place of the discretization (19) | `simulator.rs:178` |
| Momentum (18) | one scalar root per cell (section 4.5) | `simulator.rs:273` |
| Choke (11), (12), (14) | squared, and used as the shooting residual (section 4.3) | `simulator.rs:406` |

### State at a point

Given the rates $(w_g, w_l)$ from $p_0$ and a pressure $p$ at height $z$, `compute_cell_state` (`simulator.rs:252`) returns the full state:

1. $T = T(z)$ from the temperature solution below.
2. $\rho_g = 10^5\,p/(R_s T)$ and $\rho_l$ = constant.
3. $\alpha$ from `solve_alpha`.
4. $v_g$ and $v_l$ from the mass rates.

Write this map as $X(p, z)$. It fails, returning `None`, only when `solve_alpha` fails.

### Temperature

With $\Delta z \to 0$, (4) with (6) and the linear ambient profile $T_a(z) = T_r - (T_r - T_s)\,z/L$ reads $dT/dz = -k\,(T - T_a(z))$, where

$$
k = \frac{\pi h D}{c_{p,g}\,w_g + c_{p,l}\,w_l}.
$$

The denominator of (6) is $D(c_{p,g}\,\alpha\rho_g v_g + c_{p,l}(1-\alpha)\rho_l v_l) = D\,(c_{p,g} w_g + c_{p,l} w_l)/A$, so $k$ depends on $p_0$ only through the rates. With $T(0) = T_r$ (15), `Core::temp` uses the exact solution

$$
T(z) = T_r - \frac{z}{L}(T_r - T_s) + \frac{T_r - T_s}{kL}\left(1 - e^{-kz}\right).
$$

Temperature therefore does not depend on pressure, and the energy balance decouples from the momentum balance. v1 instead solves (19), $T_{i+1} = T_i - \Delta z\,k\,(T_{i+1} - T_a(z_{i+1}))$. That recursion is also explicit, $T_{i+1} = (T_i + \Delta z\,k\,T_a(z_{i+1}))/(1 + \Delta z\,k)$, but the Rust code does not use it (section 8).

### Momentum

For cell $i$, the momentum balance (18) becomes one equation in the outlet pressure $p_{i+1}$:

$$
f_i(p_{i+1}) = M\big(X(p_{i+1}, z_{i+1})\big) - M\big(X(p_i, z_i)\big) + \frac{\Delta z}{10^5}\,(F + G)\big(X(p_{i+1}, z_{i+1})\big) = 0,
$$

$$
M = p + \frac{\alpha\rho_g v_g^2 + (1-\alpha)\rho_l v_l^2}{10^5}, \qquad
F = \frac{f_D}{2D}\,\rho_m v_m |v_m|, \qquad
G = \rho_m\,g,
$$

with $\rho_m = \alpha\rho_g + (1-\alpha)\rho_l$ and $v_m = \alpha v_g + (1-\alpha)v_l$ (`simulator.rs:234-249`). Friction and gravity are evaluated at the outlet, which is the implicit-Euler form of (18).

### Choke residual

With the wellhead state $X(p_N, z_N)$ from the march, the shooting residual (`simulator.rs:406`) is

$$
R(p_0) = w_m^2 - \big(K_c\,\sigma(u)\big)^2\,\frac{2\rho\,\Delta p}{\Phi}, \qquad
\Delta p = 10^5\,\big(p_L - \mathrm{smax}(c_{pr}\,p_L,\ p_s)\big),
$$

where $w_m = w_g + w_l$ is recomputed from the top state. For the Simpson choke, $\rho = \rho_l$ and $\Phi = (1 + x_g(s-1))(1 + x_g(s^5 - 1))$ with $s = (\rho_l/\rho_g)^{1/6}$ and $x_g = w_g/w_m$. This is (11)–(12) squared, since $\rho_e = \rho_l/\Phi$ (paper footnote 1). For the Bernoulli choke, $\rho = \rho_m$ and $\Phi = 1$. The critical downstream pressure (14) is the smooth max of section 4.10.

Squaring keeps $R$ defined when $\Delta p < 0$ and adds no spurious roots, because $w_m \ge 0$ and a negative $\Delta p$ makes $R > 0$. The sign has a physical meaning:

- $R > 0$: the tubing delivers too little wellhead pressure for the choke to pass this rate.
- $R < 0$: the tubing delivers more than the choke needs.

## 3. Algorithm overview

```
simulate()                                   simulator.rs:566
└─ shoot()                                   simulator.rs:441   scan + Brent on p0
   └─ residual(p0)                           simulator.rs:433   R(p0) and the failed flag
      ├─ simulate_inner(p0)                  simulator.rs:341   march over N cells
      │  └─ solve_cell(i, p_i)               simulator.rs:273   p_{i+1} from f_i = 0
      │     ├─ compute_cell_state(z, p)      simulator.rs:252   X(p, z)
      │     │  ├─ temp(z)                    simulator.rs:178   analytic T(z)
      │     │  └─ solve_alpha(...)           simulator.rs:199   fixed-point iteration
      │     ├─ brentq(f_i, ...)              brentq.rs:40       fast and slow path
      │     └─ minimize(f_i, ...)            brentq.rs:167      golden section for p*
      └─ right_boundary(top state)           simulator.rs:406   squared choke equation
```

Each evaluation of $R$ is one march. Each march solves $N$ cell equations. Each evaluation of a cell equation runs one $\alpha$ solve. `brentq` is used at two levels, on $R(p_0)$ and on $f_i(p_{i+1})$.

## 4. Subroutines

### 4.1 `simulate` (entry point)

*`simulator.rs:566`*

1. Snapshot `wp` and `bc` into a plain-Rust `Core` (`simulator.rs:500`). Inflow and choke parameters are re-read on every call, so Python-side mutations take effect.
2. `roots = shoot()`.
3. For each root, march again with `simulate_inner(root)` and keep the state vector if the march is not flagged `failed`. `shoot` already filters on the same flag, so this is a guard.
4. Raise `SimError` if no solution remains; otherwise return the solutions in `shoot` order, highest $p_0$ first.

The solver is deterministic and uses no initial guess.

### 4.2 `shoot` (scan and bracket)

*`simulator.rs:441`*

**Purpose.** Find the roots of $R(p_0)$ on $(p_s, p_r)$.

**Assumption.** As $p_0$ decreases from $p_r$ to $p_s$, $R$ changes sign in the pattern $+\,-\,+$ or $-\,+$, with at most one negative region (section 7).

**Algorithm.**

1. Set $p_{lo} = p_s + 10^{-3}$, $p_{hi} = p_r - 10^{-6}$ and step $= (p_{hi} - p_{lo})/100$.
2. For $k = 0, \dots, 100$, evaluate $R$ at $p = p_{hi} - k\cdot$step, walking downward:
   - If `residual(p)` is `None` (non-finite $R$), forget the previous sample and continue. No bracket is formed across such a hole.
   - If $R \ge 0$, remember $p$ as the previous sample $p_{prev}$ and continue. The sample's `failed` flag is ignored here.
   - If $R < 0$, this is the first negative sample $p_{neg}$:
     1. If $p_{prev}$ exists, run `brentq(R, p_neg, p_prev)`. The result is the high-$p_0$ root.
     2. Run `brentq(R, p_lo, p_neg)`. The result is the low-$p_0$ root. This assumes $R(p_{lo}) > 0$.
     3. Keep each root whose march is not flagged `failed`, then return. The high-$p_0$ root comes first.
3. If no sample is negative, return an empty list.

**Stopping and tolerances.** Brent with xtol = 10⁻⁶ bar, rtol = 4ε and at most 100 iterations. An error inside a Brent call aborts only that bracket; `residual` returning `None` inside Brent is mapped to an error. $|R|$ at the returned point is not checked.

**Properties.**

- Finds at most two roots.
- A negative region narrower than one step, or a third root after the first negative region, is missed.
- The scan starts at $p_r - 10^{-6}$, so a root with less than 10⁻⁶ bar of drawdown is never bracketed.
- Cost: 9 to 33 evaluations of $R$ per well, median 21 (sol-1 configs), because the scan stops at the first negative sample.

### 4.3 `residual` and `right_boundary`

*`simulator.rs:433`, `simulator.rs:406`*

`residual(p0)` runs `simulate_inner(p0)` and applies `right_boundary` to the last grid point. It returns `Some((R, failed))` if $R$ is finite and `None` otherwise. `right_boundary` evaluates the squared choke residual of section 2. Python exposes them as `_residual(p0)` and `_right_boundary_eqs(x)`.

### 4.4 `simulate_inner` (the march)

*`simulator.rs:341`*

**Purpose.** Given $p_0$, compute the state at all $N+1$ grid points and report whether the march is a genuine solution of the cell equations.

**Algorithm.**

1. $(w_l, w_g) = \text{inflow}(p_0, p_r)$, then $w_g \mathrel{+}= w_{lg}$. Both rates stay fixed for the march.
2. Set $p = p_0$. For $i = 0, \dots, N-1$, call `solve_cell(i, p)`:
   - **Feasible**`(x_i, p_next)`: store $x_i$ and set $p = p_{next}$.
   - **Choked**`(x_i, p*)`: store $x_i$, set $p = p^*$ and flag the march `failed`. This continuation keeps $R$ continuous in $p_0$ across choked marches, so the outer Brent can bracket across them.
   - **None** (the state at $p_i$ cannot be computed): flag `failed` and stop solving. The last computable state is copied to grid point $i$ and every point above it, or NaN if there is none.
3. If the march did not stop, the top state is `compute_cell_state(z_N, p)`; otherwise it is the copied state.
4. Return `(x, failed)`, where `failed` is also set if no state could be computed at all.

**Properties.**

- A stopped march makes $R$ jump at the $p_0$ where stopping starts. A choked march does not.
- A march is accepted as a solution only if no cell choked or stopped.
- A trial march can pass below $p_s$; a solution cannot (section 8).

### 4.5 `solve_cell` (one cell's momentum balance)

*`simulator.rs:273`*

**Purpose.** Solve $f_i(p_{i+1}) = 0$ for the physical root.

**Shape of $f_i$.** As $p_{i+1}$ decreases, gas expands and the kinetic part of $M$ grows roughly like $1/p$. So $f_i$ is U-shaped with its minimum at $p^*$, the discrete choking point where $dM/dp \approx 0$ (roughly where the mixture velocity reaches the two-phase sound speed). $f_i(p_i) > 0$, since the cell loses pressure to friction and gravity. If $f_i(p^*) < 0$ there are two roots:

- the physical subsonic root on $(p^*, p_i)$, typically about 1 bar below $p_i$ for $N = 100$;
- a spurious supersonic root below $p^*$, typically below 1 bar.

**Algorithm.**

1. $x_i = X(p_i, z_i)$. If this fails, return `None`.
2. Set up $f_i$. Each evaluation at a trial $p_{i+1}$ computes $\rho_g$ at $T(z_{i+1})$, calls `solve_alpha`, and returns an error if $\alpha$ fails.
3. **Fast path.** With $P_{min} = 10^{-3}$ and $lo = \max(p_i - 0.1\,(p_i - p_s),\ P_{min})$, if $lo < p_i$ run `brentq(f_i, lo, p_i)`. On success return **Feasible**. Because the upper end $p_i$ is above $p^*$, a sign change on $[lo, p_i]$ brackets the subsonic root, whether or not $lo < p^*$.
4. **Slow path.** Run `minimize` on $[P_{min}, p_i]$, with errors and non-finite values mapped to $+\infty$, to get $p^*$. Then run `brentq(f_i, p*, p_i)`. On success return **Feasible**.
5. Otherwise return **Choked**`(x_i, p*)`. This covers both "no subsonic root" and "an $\alpha$ failure inside Brent on $[p^*, p_i]$".

**Stopping and tolerances.** Brent: xtol = 10⁻⁶ bar, rtol = 4ε, at most 100 iterations. Golden section: interval ≤ 10⁻² bar or 200 iterations.

**Properties.** The fast path fails in two situations:

- the pressure drop exceeds $0.1(p_i - p_s)$, which happens near the wellhead as $p_i$ approaches $p_s$;
- the march is already below $p_s$, where $lo \ge p_i$ and the fast path is skipped.

Over all shoots on the sol-1 configs: 85.8% of cell solves take the fast path, 5.6% the slow path and 8.6% choke. There are 7.8 evaluations of $f_i$ per cell on average.

### 4.6 `compute_cell_state` and `temp`

*`simulator.rs:252`, `simulator.rs:178`*

`compute_cell_state(z, p, w_l, w_g)` implements the map $X(p, z)$ of section 2 and returns `None` only if `solve_alpha` does. `temp(z, w_g, w_l)` is the closed-form $T(z)$. It needs $c_{p,g} w_g + c_{p,l} w_l > 0$, which the scan guarantees by staying $10^{-6}$ bar below $p_r$.

### 4.7 `solve_alpha` (fixed-point iteration)

*`simulator.rs:199`*

**Purpose.** Find the void fraction that satisfies the slip law (8) at a point where $w_g$, $w_l$, $\rho_g$, $\rho_l$ and $T$ are known.

**Equation.** Substituting $v_g = w_g/(A\alpha\rho_g)$ into (8) gives a fixed-point equation $\alpha = S(\alpha)$:

$$
S(\alpha) = \frac{w_g}{A\,\rho_g\,\big(C_0(\alpha)\,v_m + v_\infty(\alpha) + 10^{-6}\big)}, \qquad
v_m = \frac{w_g}{A\rho_g} + \frac{w_l}{A\rho_l}.
$$

$C_0$ (A.9) and $v_\infty$ (A.10, corrected) are blends of the regime values weighted by the classifier probabilities (A.7)–(A.8). The superficial velocities $\alpha v_g = w_g/(A\rho_g)$ and $(1-\alpha)v_l = w_l/(A\rho_l)$ do not depend on $\alpha$. So $S$ depends on $\alpha$ only through the classifier's two $\alpha$ features, $\tanh(2(\alpha - 0.25))$ and $\tanh(2(\alpha - 0.7))$. Raising $\alpha$ moves probability toward annular flow, which lowers $C_0$ and $v_\infty$, so $S$ is increasing.

**Algorithm.**

1. If $\rho_g \le 0$, replace it by $10^{-3}$. If $w_g \le 0$, return $\alpha = 10^{-6}$.
2. Start from $\alpha_0 = \mathrm{clamp}\big(w_g / (A\rho_g(1.1\,v_m + 0.5 + 10^{-6}))\big)$, i.e. $S$ with $C_0 = 1.1$ and $v_\infty = 0.5$ m/s.
3. Repeat $\alpha_{k+1} = \mathrm{clamp}(S(\alpha_k))$, where clamp means $[10^{-6},\ 1 - 10^{-6}]$, until $|\alpha_{k+1} - \alpha_k| < 10^{-3}$ or 100 iterations.
4. Return $\alpha_{k+1}$ if the loop converged and the value is finite; otherwise `None`.

**Properties.**

- The iteration converges linearly with ratio $q = S'(\alpha^*)$, provided $q < 1$. That was observed in every sampled state but is not guaranteed.
- It averages 3.0 iterations per call.
- Near $\alpha \approx 0.7$, $q$ reaches 0.78–0.96. The stop tests the step size, not the error, and the remaining error is about $q/(1-q)$ times the last step, so the returned $\alpha$ can be off by several times 10⁻³ there.
- All sampled states had a single fixed point.

See item 2 of `solver_improvements.md`.

### 4.8 `brentq` (Brent's method)

*`brentq.rs:40`*

**Purpose.** Find a root of a continuous scalar function on an interval with a sign change. The code is a transcription of SciPy's `brentq.c`, with the same step-acceptance logic and convergence test.

**Signature.** `brentq(f, xa, xb, xtol, rtol, maxiter) -> Result<f64, RootError>`. `f` returns `Result<f64, RootError>`; any error from `f` aborts the call and is returned.

**Algorithm.** The routine tracks three points:

- `xcur`, the current best estimate (the one with the smaller $|f|$);
- `xblk`, the other end of the current sign-change interval;
- `xpre`, the previous iterate.

It also tracks the last two step lengths, `spre` and `scur`.

1. Evaluate $f(x_a)$ and $f(x_b)$. Return an endpoint whose value is exactly 0. Return `NoSignChange` if both values have the same sign.
2. Repeat up to `maxiter` times:
   1. If $f(x_{pre})$ and $f(x_{cur})$ have opposite signs, set `xblk = xpre`, so the sign change lies between `xcur` and `xblk`.
   2. If $|f(x_{blk})| < |f(x_{cur})|$, swap the roles so `xcur` holds the smaller $|f|$.
   3. Let $\delta = (\text{xtol} + \text{rtol}\,|x_{cur}|)/2$ and let $s_{bis} = (x_{blk} - x_{cur})/2$ be the bisection step. If $f(x_{cur}) = 0$ or $|s_{bis}| < \delta$, return `xcur`.
   4. If the previous step was larger than $\delta$ and $|f|$ decreased, try an interpolation step. Use a secant step if only two distinct points are available (`xpre == xblk`), otherwise inverse quadratic interpolation through `xpre`, `xcur` and `xblk`. Accept it if $2|s_{try}| < \min(|s_{pre}|,\ 3|s_{bis}| - \delta)$; otherwise bisect.
   5. Move `xcur` by the chosen step, or by $\pm\delta$ toward `xblk` if the step is smaller than $\delta$. Evaluate $f$ there.
3. After `maxiter` iterations, return `MaxIter`.

**Properties.**

- It always keeps a sign-change interval, so it converges for any continuous $f$ with a sign change. The worst case is close to bisection.
- Near a simple root it converges superlinearly (order about 1.8 with inverse quadratic interpolation).
- It costs 2 evaluations for the endpoints plus one per iteration.
- The stopping test is on $x$ only: the returned point is within about xtol + rtol·|x| of a sign change. If $f$ jumps instead of crossing zero, Brent converges onto the jump.

**Uses.**

| Call | Function | Interval | xtol | rtol | maxiter |
|---|---|---|---|---|---|
| Cell, fast path (`simulator.rs:311`) | $f_i$ | $[lo, p_i]$ | 10⁻⁶ bar | 4ε | 100 |
| Cell, slow path (`simulator.rs:334`) | $f_i$ | $[p^*, p_i]$ | 10⁻⁶ bar | 4ε | 100 |
| Shoot, high-$p_0$ root (`simulator.rs:464`) | $R$ | $[p_{neg}, p_{prev}]$ | 10⁻⁶ bar | 4ε | 100 |
| Shoot, low-$p_0$ root (`simulator.rs:472`) | $R$ | $[p_s + 10^{-3}, p_{neg}]$ | 10⁻⁶ bar | 4ε | 100 |

`RTOL` = 4ε ≈ 8.9·10⁻¹⁶ is SciPy's default (`brentq.rs:17`).

### 4.9 `minimize` (golden-section search)

*`brentq.rs:167`*

**Purpose.** Find the minimum of a function on $[a, b]$ without derivatives. The solver uses it only to locate $p^*$ in the slow path of `solve_cell`.

**Signature.** `minimize(f, a, b, xtol, maxiter) -> (x_min, f_min)`. `f` returns a plain `f64`; the caller maps errors and non-finite values to $+\infty$.

**Algorithm.** With $\varphi^{-1} = 0.618\ldots$:

1. Place two interior points, $c = b - \varphi^{-1}(b - a)$ and $d = a + \varphi^{-1}(b - a)$, and evaluate $f$ at both.
2. Repeat until $b - a \le$ xtol or `maxiter` iterations:
   - If $f(c) < f(d)$, the minimum lies in $[a, d]$: set $b = d$, move $c$ to $d$, and place and evaluate a new $c$.
   - Otherwise it lies in $[c, b]$: set $a = c$, move $d$ to $c$, and place and evaluate a new $d$.
3. Return the midpoint $x = (a+b)/2$ and $f(x)$.

**Properties.**

- Each iteration shrinks the interval by 0.618 and costs one new evaluation, because the golden ratio lets one interior point be reused.
- On $[10^{-3}, p_i]$ with $p_i \approx 100$ bar and xtol = 10⁻² bar, that is 20 iterations and 23 evaluations of $f_i$, each including an $\alpha$ solve.
- It assumes $f$ has a single minimum on the interval and does not check it. With several minima, or large $+\infty$ regions, it can return a point that is not the minimum.
- The result feeds only the lower end of the slow-path bracket and the continuation pressure of a choked march, so the loose 10⁻² bar tolerance does not affect accepted roots.

Parameters in `solve_cell`: interval $[10^{-3}, p_i]$, xtol = 10⁻² bar, maxiter = 200.

### 4.10 Smoothing functions

These are part of the equations, not solvers. They are listed because they shape the functions the solvers see.

| Function | Definition | Used in | Paper |
|---|---|---|---|
| `max_approx` (`math.rs:6`) | $\mathrm{smax}(x, y) = \tfrac12\big(x + y + \sqrt{(x-y)^2 + \epsilon}\big)$, $\epsilon = 10^{-6}$ bar² | critical pressure in the choke residual, $\mathrm{smax}(c_{pr} p_L, p_s)$ | (B.1), (14) |
| `softmax3` (`math.rs:12`) | $p_j = e^{y_j} / \sum_k e^{y_k}$ | regime probabilities from the classifier logits | (A.7) |
| `classify_flow_regime` (`slip.rs:17`) | affine map of four tanh features, then softmax | $C_0$, $v_\infty$ and the regime labels | (A.7)–(A.8) |
| `identify_parameters` (`slip.rs:51`) | $C_0 = 1.0\,p_a + 1.175\,p_s + 1.2\,p_b$; $v_\infty = v_{\infty T}\,p_s + v_{\infty b}\,p_b$ | `solve_alpha` | (A.9), (A.10) corrected |

$v_{\infty b}$ is Harmathy's bubble rise velocity (A.1) and $v_{\infty T}$ the Taylor-bubble rise velocity (A.2); $p_a$, $p_s$ and $p_b$ are the annular, slug/churn and bubbly probabilities. `smax` overestimates $\max(x, y)$ by at most 5·10⁻⁴ bar, at $x = y$. The critical pressure ratio is $c_{pr} = (2/(\gamma+1))^{\gamma/(\gamma-1)}$ with $\gamma = 1.307$, i.e. $c_{pr} \approx 0.5445$ (`choke.rs:15`).

## 5. Tolerances and constants

| Constant | Value | Location | Role |
|---|---|---|---|
| $N$ | 100 (default) | `SSDFSimulator::new` | number of cells |
| Scan intervals | 100, i.e. 101 samples | `shoot` | bracketing grid for $p_0$ |
| $p_{hi}$, $p_{lo}$ | $p_r - 10^{-6}$ bar, $p_s + 10^{-3}$ bar | `shoot` | scan range |
| Outer Brent | xtol 10⁻⁶ bar, rtol 4ε, 100 iterations | `shoot` | roots of $R$ |
| $P_{min}$ | 10⁻³ bar | `solve_cell` | lowest trial outlet pressure |
| Fast bracket | $0.1\,(p_i - p_s)$ below $p_i$ | `solve_cell` | first bracket for $f_i$ |
| Cell Brent | xtol 10⁻⁶ bar, rtol 4ε, 100 iterations | `solve_cell` | roots of $f_i$ |
| Golden section | interval 10⁻² bar, 200 iterations | `solve_cell` | $p^*$ |
| $\alpha$ stop | $\lvert\Delta\alpha\rvert < 10^{-3}$, 100 iterations | `solve_alpha` | fixed-point termination |
| $\alpha$ bounds | $[10^{-6},\ 1 - 10^{-6}]$ | `solve_alpha` | clamp on every iterate |
| $\alpha_0$ | $C_0 = 1.1$, $v_\infty = 0.5$ m/s | `solve_alpha` | starting guess |
| Slip guard | $+10^{-6}$ m/s in the denominator of $S$ | `solve_alpha` | avoids division by zero |
| $\rho_g$ floor | $10^{-3}$ kg/m³ if $\rho_g \le 0$ | `solve_alpha` | guard |
| Smooth-max $\epsilon$ | 10⁻⁶ bar² | `max_approx` | smoothing of (14) |
| $\gamma$ | 1.307 | `critical_pressure_ratio` | $c_{pr} \approx 0.5445$ |
| `CF_PRES`, $g$ | 10⁵ Pa/bar, 9.80665 m/s² | `constants.rs` | unit conversion, gravity |

## 6. Failure handling

| Where | Event | Result | Effect higher up |
|---|---|---|---|
| `solve_alpha` | no convergence in 100 iterations, or non-finite value | `None` | an error inside $f_i$ or `compute_cell_state` |
| `brentq` in the fast path | no sign change, `MaxIter`, or $\alpha$ error | error | falls through to the slow path |
| `brentq` in the slow path | same | error | cell returns **Choked** at $p^*$ |
| `compute_cell_state` at $p_i$ | $\alpha$ fails | `None` | march **stops**; last state copied upward |
| `simulate_inner` | any Choked or stopped cell | `failed = true` | $R$ is still returned, but a root there is rejected |
| `residual` | $R$ not finite | `None` | a hole in the scan, or an error inside the outer Brent |
| `shoot` | no negative sample, or Brent errors on both brackets | empty list | `simulate` raises `SimError` |

## 7. The solution set

On the sol-1 configs:

- **Number of roots.** $R$ follows $+\,-\,+$ (two roots) or $-\,+$ (one root). Two roots occur exactly when $p_r - p_s < \rho_l\,g\,L/10^5$, i.e. when a static liquid column cannot reach the separator, so $R > 0$ at zero rate. This criterion split all 2,000 configs without exception (1,254 two-root, 746 one-root). If the negative region never forms, there is no root and the well cannot flow.
- **Stability.** The root from the right bracket has $dR/dp_0 > 0$ and is statically unstable; the root from the left bracket has $dR/dp_0 < 0$ and is stable. The label follows from the bracket a root came from, given one crossing per bracket. The unstable root is usually a near-dead trickle, with a median drawdown of 0.13 bar and the wellhead within about a millibar of $p_s$.
- **Order.** `simulate()` returns the unstable root first. It is not the root v1 usually converges to: on 151 sampled two-root wells, v1 found the stable root in 139.

## 8. Conformance with the v1 residual

A Rust solution is a state vector on the same grid as v1. Evaluated with v1's equations, as the verifier in Step 2 of the v2 plan does, its rows behave as follows:

| v1 rows | Satisfied | Size of the residual |
|---|---|---|
| Inflow (10), $T(0) = T_r$ (15) | exactly | rounding |
| Mass balances (16)–(17) | exactly | rounding |
| (7), gas law (9), constant $\rho_l$ | exactly | rounding |
| Slip law (8) | to the $\alpha$ stopping rule | $\alpha$ errors up to several 10⁻³ near $\alpha \approx 0.7$; observed 5.2·10⁻³ at well 6, cell 70 |
| Momentum (18) | to cell Brent's xtol | about 10⁻⁶ bar per cell, since $f_i' = O(1)$ near the subsonic root |
| Energy (19) | no | Rust uses the exact ODE solution, so each row is the implicit-Euler truncation error, which grows with $\Delta z\,k$ and is largest at low rates. Well 977: 0.031 K in the first cell of the high-$p_0$ root, 2.3·10⁻³ K at most for the low-$p_0$ root. Largest row per solution over sol-1: median 5·10⁻³ K, 95th percentile 0.30 K, maximum 0.43 K |
| Choke (11)/(14) | to outer Brent's xtol | small for flowing roots. Poorly resolved for trickle roots with tiny drawdown: at well 87, $\lvert R\rvert \approx 9\cdot10^4\,w_m^2$ |
| v1 bounds $p \in [p_s, p_r]$, $T \in [T_s, T_r + 1]$, $\alpha \le 1$ | yes, at accepted roots | $p$ decreases up the well and $p_L > p_s$ at a root; $T$ lies between $T_a(z)$ and $T_r$ |

To make the energy rows exact, use the explicit recursion of section 2 instead of the analytic solution. It costs the same and changes $T$ by O($\Delta z$).

## 9. Differences from the paper

Items for the discrepancy list of Step 4. "v1" means `manywells/*.py` at v1.0.0.

| Item | Paper | v1 code | Rust code |
|---|---|---|---|
| Classifier $\alpha$ features (A.8) | $\tanh(\alpha_g - 0.25)$, $\tanh(\alpha_g - 0.7)$ | $\tanh(2(\alpha - 0.25))$, $\tanh(2(\alpha - 0.7))$ (`slip.py:90-92`) | same as v1 (`slip.rs:27-29`) |
| Rise-velocity weights (A.10) | typo, listed in `docs/corrigendum.md` on `develop` | corrected form | corrected form |
| Critical pressure (14) | exact max | smooth max (B.1), $\epsilon = 10^{-6}$ | same as v1 |
| Energy (19) | implicit Euler | implicit Euler | exact solution of (4) |
| Friction (5) | $\rho_m v_m\lvert v_m\rvert$ | $\rho_m v_m^2$ | $\rho_m v_m\lvert v_m\rvert$ (identical for upward flow) |
| Choke (11)–(12) | $\rho_e$ with Simpson slip factor | $\rho_l$ with Simpson multiplier (equivalent) | same as v1, squared |
| $\gamma$ in (13) | ≈ 1.3, $r_c \approx 0.544$ | 1.307, 0.5445 | same as v1 |
| Slip-law safeguards | none | none (solved inside the NLP) | $+10^{-6}$ in the denominator, $\alpha$ clamp, $\rho_g$ floor |
| Inflow | Vogel (10) | Vogel, PI, fixed rate | Vogel, PI |

The first item is not in the corrigendum.

## 10. Cost profile

Measured on the 2,000 sol-1 configs with a JavaScript port of this code, which reproduces the Rust roots to 1e-13 bar:

| Quantity | Value |
|---|---|
| Evaluations of $R$ (marches) per well | min 9, median 21, max 33 |
| Cell solves by path | fast 85.8%, slow 5.6%, choked 8.6% |
| Evaluations of $f_i$ per cell | 7.8 |
| Golden-section searches | 14.2% of cell solves |
| Fixed-point iterations per $\alpha$ solve | 3.0 |

Choked cells occur only in trial marches away from a root, but each costs a golden-section search, so they account for about a third of all evaluations of $f_i$. `docs/simulator_in_rust.md` reports a mean of 5.3 ms per well for the Rust solver on sol-1, against 0.90 s for v1 on the same machine.

## 11. Code map

| Component | Location |
|---|---|
| Python classes and entry points: `WellProperties`, `BoundaryConditions`, `SSDFSimulator` | `simulator.rs:25`, `:104`, `:488`; `simulate` `:566` |
| Debug entry points: `_residual`, `_right_boundary_eqs`, `solution_as_df` | `simulator.rs:589`, `:595`, `:604` |
| `Core` snapshot | `simulator.rs:154`, built at `:500` |
| `shoot`, `residual`, `right_boundary` | `simulator.rs:441`, `:433`, `:406` |
| `simulate_inner`, `solve_cell`, `CellStep` | `simulator.rs:341`, `:273`, `:141` |
| `compute_cell_state`, `temp`, `rho_gas`, `v_g`, `v_l` | `simulator.rs:252`, `:178`, `:192`, `:184`, `:188` |
| `mom`, `f_fric`, `g_grav` | `simulator.rs:234`, `:239`, `:246` |
| `solve_alpha` | `simulator.rs:199` |
| `brentq`, `minimize`, `RTOL` | `brentq.rs:40`, `:167`, `:17` |
| Slip model: `classify_flow_regime`, `identify_parameters`, `harmathy_rise_velocity`, `taylor_rise_velocity` | `slip.rs:17`, `:51`, `:40`, `:46` |
| `dead_oil_surface_tension` | `pvt.rs:17` |
| `max_approx`, `softmax3` | `math.rs:6`, `:12` |
| Choke profiles, critical pressure ratio | `choke.rs:49`, `:15` |
| Inflow rates | `inflow.rs:18` |
| Python wrapper | `python/manywells_rs/__init__.py` |
