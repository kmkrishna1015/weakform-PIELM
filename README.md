# Weak-Form PIELM

Code, data and figure scripts for the paper

> K. K. Mishra, H. Sonavane, R. Mallick, D. K. Harursampath, **Weak-Form PIELM for Differential Eigenvalue Problems: A Frozen-Basis Variational Framework and Structural Comparison with Strong-Form Eig-PIELM**, submitted to *Computer Methods in Applied Mechanics and Engineering*, 2026.

Weak-Form PIELM keeps the frozen trial basis, the exact null-space enforcement of the essential conditions and the single reduced generalized eigenproblem of Eig-PIELM, but lets natural, corner and concentrated mass/stiffness contributions enter through variational bilinear forms. For self-adjoint problems with exact integration this gives the ordering `exact <= weak <= strong` of the eigenvalues. The repository contains both formulations, every experiment of the paper, and the figure scripts.

## Quick start
```bash
git clone https://github.com/kmkrishna1015/weakform-PIELM.git
cd weakform-PIELM
pip install -r requirements.txt      # numpy, scipy, pandas, matplotlib, mpmath, sympy
python3 tests.py                     # a few seconds, sanity checks
python3 run_all.py                   # regenerate everything (tens of minutes, one core)
```
On Windows use `py -m pip install -r requirements.txt` and `py run_all.py` (or `run_all.bat`); on Linux/macOS `./run_all.sh` is an equivalent wrapper.
Options: `--figures-only`, `--skip-tests`, `--only e5_basis_comparison e19_fem_richardson`.
No GPU is needed. `torch` is optional and only used for the autodiff timing rows of `e2`.

## Minimal example
```python
from experiments._common import *            # problems(), rel_err_pct, ...
P = problems(); p = P["beam_cf"]               # cantilever Euler-Bernoulli beam
from pielm.bases import Bern1D
r = p.weak(Bern1D(p.L, 23), 70, rcond=1e-12, symmetric=True)   # one reduced symmetric eigenproblem
print(r.omega[:3], rel_err_pct(r.omega, p.exact(r.omega.size))[:3])
```

## Formulations and eigensolver
Two formulations are implemented; three timing columns are reported.

| column (`method=`) | formulation | eigensolver |
|---|---|---|
| `strong` | Eig-PIELM as published: PDE residual at uniform interior points (unit weights); essential, natural and corner/interface rows projected out | `scipy.linalg.eig(S, G)`, general pencil (not symmetric) |
| `weak` | Weak-Form PIELM: Gauss-Legendre Rayleigh-Ritz, essential rows only | `scipy.linalg.eig(K, M)`, same solver as `strong` |
| `weak_sym` | Weak-Form PIELM, identical matrices | `eigh` on the Löwdin-reduced pencil (`rcond=1e-12`) |

All three go through `pielm.numerics.solve_pencil(A, B, rcond=None, symmetric=False)`. `rcond=None` means no truncation (used for `strong` and `weak`); `symmetric=True` is used only for the weak form.
A plain Cholesky `eigh(K, M)` is not used for `weak_sym`: the Gram matrix of a polynomial basis is nearly singular (cond ~ 1e16) and the beams lose about five digits (5e-8 to 9e-8 %), whereas the Löwdin-reduced solve keeps ~1e-12 %.
Two controls are also provided: the strong form with exact Gauss weights (`pielm/strong_exact.py`, i.e. Rayleigh-Ritz on the space V_s) and a bordered one-shot strong pencil for the concentrated-mass beams (`e13_additional_validation.py`, `e18_bordered_checks.py`). They are not part of the three timing columns.

## Manuscript map
| manuscript item | script | output in `results/` / `figures/` |
|---|---|---|
| Table 4 (dim V_w, dim V_s) | `e13_additional_validation.py`, `e6_kirchhoff.py` | `additional_validation.json`, `P_plate.csv` |
| Table 5 (seven standard benchmarks) | `e1_benchmarks.py` | `benchmarks.json` |
| Section 4.2 timings and cost ratios | `e2_timing.py` | `timing.json` |
| Section 4.2 exact-weight control | `e13_additional_validation.py` | `additional_validation.json` |
| Section 3.4 ordering check | `e8_ordering.py` | `ordering_check.csv` |
| Table 6 (concentrated masses), solve counts | `e4_pointmass.py`, `e13_additional_validation.py` | `hard.json`, `additional_validation.json` |
| Section 4.3 timing ratios (Table 6) and mode-count study | `e4_pointmass.py` | `hard.json`, `H_nmodes.csv` |
| Section 4.3 bordered-pencil checks | `e18_bordered_checks.py` | `bordered_checks.json` |
| Section 4.3 bordered-pencil diagnostics (eigenvalue filter, fixed-point residual, 50-digit solve) and Brent failure at the tenth mode | `e23_bordered_diagnostics.py` | printed to the console |
| Table 8(a) (Kirchhoff plates, eigenvalues) | `e6_kirchhoff.py` | `P_plate.csv`, `P_refs.json`, `P_fairness.json`, `P_young_check.json` |
| Table 8(a), exact-weight strong-form columns and dimensions of V_s | `e9_vs_convergence.py` | `V_convergence.csv` |
| Table 8(b) (eigenfunction errors, n = 16) | `e14_eigenfunction_error.py` | `eigenfunction_error.json` |
| Free-edge residuals (Section 4.4) | `e20_plate_residuals.py` | `P_residuals.csv`, `P_residuals_summary.json` |
| Appendix C, Fig. C.1 | `e3_sensitivity.py S2`, `f2_appendix_figures.py` | `S2_collocation.csv`, `fig_S2_collocation` |
| Appendix D: 60-digit Ritz, mass-free limit, Leissa | `e21_highprecision_checks.py` | `HP_bernstein_ritz.csv`, `HP_checks_summary.json` |
| Appendix D: BFS finite-element check | `e19_fem_richardson.py` | `P_fem_richardson.csv`, `P_fem_richardson_summary.json` |
| Appendix D (Tables D.1, D.2): reference values and convergence n = 22..38 | `e13_additional_validation.py` | `additional_validation.json` |
| Appendix E, Table E.1 (basis comparison) | `e5_basis_comparison.py` | `B_basis_1d.csv`, `B_basis_2d.csv`, `B_basis_summary.json` |
| Figs. 2, 3, 4 | `f1_main_figures.py` | `fig_modes_1d`, `fig_modes_helmholtz`, `fig_modes_plate` |
| Fig. 7 | `f4_pointmass_figures.py` | `fig_mass_modeshapes` |
| Fig. 8 (SSSS/SSFF, role of the natural rows) | `e22_natural_row_ablation.py`, `fig_plate_natural_rows.py` | `P_natural_row_ablation.json`, `fig_plate_modes` |
| Fig. 9 (free-plate modes) | `f3_kirchhoff_figures.py` | `fig_plate_freemodes` |

Figs. 1, 5 and 6 are drawings and are not generated by code. In `P_plate.csv` the plate with two opposite simply supported and two free edges is stored under the legacy label `SFSF`; it is the SSFF plate of the paper (simply supported at x = 0, 1, free at y = 0, 1). Other files in `results/` and `figures/` (error budget `V_convergence.csv`, plate convergence and Pareto plots, degree and timing-scaling studies, scattered-point integration rule, conditioning study, autodiff timings, Q4-FEM baseline, Pareto plots) are supplementary outputs of the scripts and are not quoted in the paper. This also holds for `cost_equivalence.json` and `mass_gauss_timing.json` (`e12`, `e17`), which time the concentrated-mass beams with other assembly choices than the stored Table 5 run.

## Reproducibility notes
* `results/` and `figures/` hold the output of one complete run (Windows 64-bit, Intel CPU, Python 3.14.0, NumPy 2.5.3, SciPy 1.18.1, single BLAS thread). The paper quotes this run.
* Deterministic quantities (space dimensions, constraint counts, errors well above the round-off floor, plate, eigenfunction and error-budget values, references) are reproduced to the quoted digits on other platforms. A Linux run (Python 3.12, NumPy 2.4, SciPy 1.17) reproduced Tables 3 and 7 and the eigenfunction errors of Table 7(b).
* Errors near the double-precision floor (below about 1e-9 %) change by up to a factor of a few between CPUs and BLAS builds. The iteration counts of the Brent root search for the concentrated-mass beams also vary (79-111 solves here, 73-110 on Linux).
* Wall-clock times depend on the machine and BLAS build, and so do the cost ratios (for the 1D benchmarks the ratio of strong to weak time was about 78-150 in the stored run and about 40-50 on the Linux run). The qualitative conclusions do not change. Quote timings from your own run.
* Fig. 8 (`e22`): the strong form without the natural rows is a non-symmetric pencil with mostly complex eigenvalues. `fig_plate_modes` shows its three lowest real eigenpairs at one fixed Gauss order (n + 12 points); which real eigenvalues appear depends on that order (see `sensitivity_to_ng` in `P_natural_row_ablation.json`). The figure demonstrates that the ablated frequencies and shapes are wrong, not specific values.
* The un-truncated plate solves at n >= 20 (columns `weak`, `strong` of `P_plate.csv`) are ill-conditioned and their errors fluctuate between runs (order 1e-1 to 1).
* In 2D the strong form uses tensor-product uniform grids (5000 x 3000 for the Helmholtz cavity, 2000 x 1200 for the in-plane plate) whose Gram matrices are assembled separably (Kronecker structure), so assembly costs O(Nx + Ny) and is included in `timing.json`.
* Concentrated-mass beams: the bordered one-shot pencil is limited by the scaling of its blocks (see `e23`); the iterated strong form (Brent root search) works up to nine modes and fails from the tenth, where the bracket [0, mu_j(0)] has no sign change.
* All BLAS libraries are limited to one thread (`pielm/numerics.py`); timings are medians after two discarded warm-up calls.
* Random-feature results of `e5_basis_comparison.py` use seeds 0..9 and the scale parameters listed in the script header.

## Layout
```
pielm/             numerics (Gauss-Legendre, null space, common eigensolver), bases (Bernstein, Legendre, Chebyshev, monomial,
                   tanh-ELM, Gaussian RBF), problems (bars, beams, Helmholtz cavity, in-plane plate), pointmass (concentrated
                   masses/springs), kirchhoff + levy (plates), ablation (Fig. 8), fem_plate_* (BFS finite-element check), strong_exact, scattered
experiments/       e1 ... e23: each writes results/*.csv|json
figure_scripts/    f1 ... f5 (read results/) and fig_plate_natural_rows (Fig. 8, computes its own data); write figures/*.pdf|png
results/, figures/ stored outputs
tests.py           sanity checks
run_all.py         full reproduction (+ run_all.sh / run_all.bat wrappers)
```

## Citation
Please cite the paper and this repository (see `CITATION.cff`).
```bibtex
@article{mishra2026weakpielm,
  author  = {Mishra, Krishna Kant and Sonavane, Harsh and Mallick, Rajnish and Harursampath, Dineshkumar},
  title   = {Weak-Form {PIELM} for Differential Eigenvalue Problems: A Frozen-Basis Variational Framework and Structural Comparison with Strong-Form {Eig-PIELM}},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  year    = {2026},
  note    = {submitted}
}
```

## License
MIT, see `LICENSE`.
