### Corrigendum to ManyWells: Simulation of multiphase flow in thousands of wells

```
@article{Grimstad2026,
	title = {{ManyWells: Simulation of multiphase flow in thousands of wells}},
	author = {Bjarne Grimstad and Erlend Lundby and Henrik Andersson},
	journal = {Geoenergy Science and Engineering},
	volume = {257},
	pages = {214226},
	year = {2026},
	issn = {2949-8910},
	doi = {https://doi.org/10.1016/j.geoen.2025.214226},
}
```

List of errors, in the order they appear in the paper:
- In Table 1, $z$ is the distance along the well from the bottomhole ($z = 0$) up to the wellhead ($z = L$), not a vertical depth.
- In Section 2.3, the sentence after Equation (11) should read: "With $u_c$ and $p_s$ given, this equation imposes a condition on the pressure at the right boundary." The choke is at $z = L$.
- In Equation (12), $f_g$ and $f_l$ are the gas and liquid mass fractions of the flow through the choke, which includes the lift gas; they are not the inflow fractions of Equation (10). The implementation uses the fractions of the flow through the choke.
- In Section 3.1, the flow path is discretized into $N$ cells of length $\Delta z = L/N$, bounded by the $N + 1$ grid points $z_0, \dots, z_N$, rather than into $N + 1$ cells.
- Equation (22) should read $f_D \sim \text{Uniform}(0.01, 0.08)$.
- Equation (29) gives the oil volume fraction of the liquid. The water volume fraction used in Equations (28) and (31) is $\alpha_{w,l} = 1/\big(1 + (\rho_w/\rho_o)\cdot(f_o/f_w)\big)$. The implementation uses the correct fractions, and this typo had no effect on the results presented in the paper.
- In Section 4.3, $p_{s,\text{nom}}$ and $p_s$ are the downstream (separator) pressure, not the upstream pressure.
- In Section 5.2, the initial reservoir pressure of `manywells-nsol-1` is the nominal value of Equation (33), as Section 4.4.2 states, not a sample from Equation (32). The same holds for `manywells-nscl-1`.
- In Equation (52) and the sentence after it, the generator of `manywells-nscl-1` draws $k = 1$ with probability 0.3, so the term $kq$ is zero with probability 0.7, not 0.8.
- In Equation (A.8), the implementation that generated the datasets scales the void-fraction differences by 2: the first and third features are $\tanh\big(2(\alpha_g - 0.25)\big)$ and $\tanh\big(2(\alpha_g - 0.7)\big)$.
- In Equation (A.10), the Harmathy and Taylor bubble rise velocities should be swapped; that is, the equation should be $v_{\infty} = 0 \cdot p_{\text{annular}} + v_{\infty T} \cdot p_{\text{slug-churn}} + v_{\infty b} \cdot p_{\text{bubbly}}$. The equation was correctly implemented and this typo had no effect on the results presented in the paper.

### Errata in the published datasets

These concern the datasets `manywells-sol-1`, `manywells-nsol-1` and `manywells-nscl-1` on Hugging Face (`solution-seeker-as/manywells`), generated with ManyWells v1.0.0. The datasets themselves are unchanged.

**Samples on the unstable trickle root.** For many wells the steady-state equations have two roots: a stable operating point and a statically unstable "trickle" root with a very small rate. At the trickle root the wellhead pressure is within millibars of the downstream pressure and the wellhead is cold. v1.0.0 returns whichever root its solver reaches from its initial guess. The generators solve each well at a choke opening of 0.5 first and reuse that solution as the initial guess for the well's samples, so when that first solve lands on the trickle root, many of the well's samples do too. This is the source of the low-temperature spike near 280 K in the TWH histogram of `manywells-sol-1`.

The signature PWH − PDC < 0.01 bar marks trickle-root samples. On 200 roots verified for the ManyWells verifier, it caught 47 of the 64 unstable roots and none of the 136 stable ones. It misses some trickle-root samples, so the counts below are lower bounds.

| Dataset | Rows with PWH − PDC < 0.01 bar | Wells | Median TWH of those rows (other rows) |
|---|--:|--:|---|
| `manywells-sol-1` | 35,259 (3.53%) | 131 | 279 K (348 K) |
| `manywells-nsol-1` | 1,464 (0.15%) | 164 | 292 K (349 K) |
| `manywells-nscl-1` | 67 (0.01%) | 26 | 300 K (350 K) |

To leave these samples out, drop the rows with `PWH - PDC < 0.01`.

**Friction factor in the `nsol-1` and `nscl-1` configs.** The config files of `manywells-nsol-1` and `manywells-nscl-1` store `wp.f_D = 0.05` for every well. The wells were simulated with a friction factor drawn from U(0.01, 0.08), as in `manywells-sol-1`, whose configs store it correctly. The samples are therefore consistent, but re-simulating a well from these configs uses the wrong friction factor. For `manywells-nsol-1`, the friction factor a well was simulated with can be recovered from its stored final state `x_last`: with it, the state solves v1.0.0's equations to 1e-7 or better (`verification/build/make_cases.py`, `recover_f_D`).

**Configs of `manywells-nsol-1` and `manywells-nscl-1` hold each well's final state.** The generators update a well in place as its conditions evolve and save it after its last sample. In these configs the mass fractions, the liquid density and heat capacity, the inflow gas fraction, the reservoir and separator pressures and the reservoir-pressure decay rate are those of the well's last sample, the last row of its data, not the initial draws; in `manywells-nsol-1` the choke position and lift-gas rate are too. The initial reservoir pressure, separator pressure and mass fractions are stored in the well's `ns_bhv` (`pr_init`, `ps_init`, `init_fractions`). The `manywells-sol-1` configs hold the drawn values, with the choke position set to 0.5 by the generator's first simulation.
