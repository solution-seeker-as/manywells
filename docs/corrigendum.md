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

List of errors:
- In Equation (A.10), the Harmathy and Taylor bubble rise velocities should be swapped; that is, the equation should be $v_{\infty} = 0 \cdot p_{\text{annular}} + v_{\infty T} \cdot p_{\text{slug-churn}} + v_{\infty b} \cdot p_{\text{bubbly}}$. The equation was correctly implemented and this typo had no effect on the results presented in the paper.

