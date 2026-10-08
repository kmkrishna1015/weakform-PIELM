#!/usr/bin/env python3
"""Cross-platform reproduction driver (Windows, macOS, Linux).

    python run_all.py                       # tests -> experiments e1..e21 -> figures f1..f5
    python run_all.py --figures-only        # only the figure scripts (uses the stored results/)
    python run_all.py --only e5_basis_comparison e19_fem_richardson     # selected scripts (name or name.py), run in the given order
    python run_all.py --skip-tests

Run on an otherwise idle machine if you care about timings.  BLAS libraries are limited to one thread (set here and inside the scripts).
"""
import argparse, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = ["e1_benchmarks", "e2_timing", "e3_sensitivity", "e4_pointmass", "e5_basis_comparison", "e6_kirchhoff", "e7_fem_plate", "e8_ordering",
               "e9_vs_convergence", "e10_quadrature_free", "e11_scattered_main", "e12_cost_and_equivalence", "e13_additional_validation", "e14_eigenfunction_error",
               "e15_conditioning", "e16_integration_checks", "e17_mass_gauss_timing", "e18_bordered_checks", "e19_fem_richardson", "e20_plate_residuals",
               "e21_highprecision_checks"]
FIGURES = ["f1_main_figures", "f2_appendix_figures", "f3_kirchhoff_figures", "f4_pointmass_figures", "f5_additional_figures"]


def run(script, cwd):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time(); print(f"=== {script}", flush=True)
    rc = subprocess.call([sys.executable, script if script.endswith(".py") else script + ".py"], cwd=cwd, env=env)
    if rc != 0:
        sys.exit(f"!!! {script} failed with exit code {rc}")
    print(f"--- {script} finished in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures-only", action="store_true"); ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="SCRIPT"); a = ap.parse_args()
    ex_dir, fig_dir = os.path.join(ROOT, "experiments"), os.path.join(ROOT, "figure_scripts")
    if a.only:
        for s in a.only:
            name = s[:-3] if s.endswith(".py") else s
            if name in EXPERIMENTS: run(name, ex_dir)
            elif name in FIGURES: run(name, fig_dir)
            else: sys.exit(f"unknown script: {s}")
        return
    if not a.skip_tests: run("tests.py", ROOT)
    if not a.figures_only:
        for s in EXPERIMENTS: run(s, ex_dir)
    for s in FIGURES: run(s, fig_dir)
    print("done: results/ and figures/ regenerated")


if __name__ == "__main__":
    main()
