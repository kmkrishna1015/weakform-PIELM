# Weak-Form PIELM for Differential Eigenvalue Problems

This repository contains the numerical implementation and reproducibility data for the paper:

> **Weak-Form PIELM for Differential Eigenvalue Problems: A Frozen-Basis Variational Framework and Structural Comparison with Strong-Form Eig-PIELM**

The code implements a frozen-basis Physics-Informed Extreme Learning Machine (PIELM) framework for differential eigenvalue problems and provides numerical experiments comparing weak-form and strong-form formulations.

Repository: https://github.com/kmkrishna1015/weakform-PIELM

---

## 1. Overview

The weak-form formulation uses a fixed randomly generated basis and assembles the eigenvalue problem from variational integrals. Essential boundary conditions are imposed exactly through null-space reduction.

The repository includes:

- weak-form PIELM eigenvalue solvers;
- strong-form PIELM / collocation comparisons;
- exact-weight strong-form controls;
- concentrated-mass and eigenvalue-dependent interface problems;
- Kirchhoff plate problems with natural and corner boundary conditions;
- convergence and sensitivity studies;
- eigenfunction-error calculations;
- high-precision reference calculations;
- FEM/Richardson reference calculations;
- basis-comparison studies;
- scripts for reproducing the paper figures and numerical tables.

The stored numerical results are included so that the reported results can be checked without rerunning every experiment.

---
## 2. Repository structure

```text
weakform-PIELM/
│
├── pielm/
│   ├── basis.py
│   ├── weak_form.py
│   ├── strong_form.py
│   ├── constraints.py
│   ├── quadrature.py
│   └── ...
│
├── experiments/
│   ├── e1_benchmarks.py
│   ├── e2_timing.py
│   ├── e3_sensitivity.py
│   ├── e4_pointmass.py
│   ├── e5_basis_comparison.py
│   ├── e6_kirchhoff.py
│   ├── e7_fem_plate.py
│   ├── e8_ordering.py
│   ├── e9_vs_convergence.py
│   ├── e10_quadrature_free.py
│   ├── e11_scattered_main.py
│   ├── e12_cost_and_equivalence.py
│   ├── e13_additional_validation.py
│   ├── e14_eigenfunction_error.py
│   ├── e15_conditioning.py
│   ├── e16_integration_checks.py
│   ├── e17_mass_gauss_timing.py
│   ├── e18_bordered_checks.py
│   ├── e19_fem_richardson.py
│   ├── e20_plate_residuals.py
│   └── e21_highprecision_checks.py
│
├── figure_scripts/
│   ├── fig.py
│   ├── f1_main_figures.py
│   ├── f2_appendix_figures.py
│   ├── f3_kirchhoff_figures.py
│   ├── f4_pointmass_figures.py
│   └── f5_additional_figures.py
│
├── figures/
├── results/
│   ├── benchmarks.json
│   ├── cost_equivalence.json
│   ├── bordered_checks.json
│   ├── additional_validation.json
│   ├── eigenfunction_error.json
│   ├── P_plate.csv
│   ├── P_refs.json
│   ├── P_young_check.json
│   ├── V_convergence.csv
│   ├── ordering_check.csv
│   ├── S2_collocation.csv
│   ├── HP_bernstein_ritz.csv
│   ├── HP_checks_summary.json
│   ├── P_fem_richardson.csv
│   └── ...
│
├── tests.py
├── run_all.py
├── run_all.bat
├── run_all.sh
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
