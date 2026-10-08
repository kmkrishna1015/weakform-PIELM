"""Additional figures for the numerical study.

The script generates the plate error budget, quadrature comparison,
and basis-conditioning figures from the stored numerical results."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
def save(name): plt.tight_layout(); plt.savefig(os.path.join(FIG, name + ".pdf")); plt.savefig(os.path.join(FIG, name + ".png"), dpi=130); plt.close()
# 1. plate error budget (e9_vs_convergence.py)
d = pd.read_csv(os.path.join(RES, "V_convergence.csv")); fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for a, c, t in zip(ax, ["FFFF", "CFFF"], ["(a) fully free (FFFF)", "(b) cantilever (CFFF)"]):
    x = d[d.case == c]
    a.semilogy(x.n, x.err_strong_uniform, "s--", c="tab:red", label="published strong form (uniform points, W = I)")
    a.semilogy(x.n, x.err_strong_gauss, "o-", c="tab:orange", label="strong form, Gauss weights")
    a.semilogy(x.n, x.err_RitzVs, "x", c="k", ms=8, label="Ritz on $V_s$ (exact quadrature)")
    a.semilogy(x.n, x.err_weak, "^-", c="tab:green", label="weak form ($V_w$)"); a.set_title(t); a.set_xlabel("Bernstein degree $n$"); a.grid(alpha=.3, which="both")
ax[0].set_ylabel("max. rel. error, 6 lowest modes"); ax[0].legend(fontsize=7, loc="lower left"); save("fig_plate_error_budget")
# 2. equal vs moment-fitted weights (e10_quadrature_free.py)
d = pd.read_csv(os.path.join(RES, "quadrature_free.csv")); fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
sty = {"grid": ("s--", "tab:red", "uniform grid, equal weights"), "halton_eq": ("o--", "tab:orange", "scattered (Halton), equal weights"), "halton_mf": ("^-", "tab:green", "scattered (Halton), moment-fitted weights")}
for a, c, t in zip(ax, ["Helmholtz", "Plate FFFF", "Plate CFFF"], ["(a) Helmholtz cavity, n=13", "(b) FFFF plate, n=12", "(c) CFFF plate, n=12"]):
    for k, (m, col, l) in sty.items():
        x = d[(d.case == c) & (d.method == k)]; a.loglog(x.N, x.dev_from_gauss.clip(lower=1e-15), m, c=col, label=l)
    a.set_title(t); a.set_xlabel("number of integration points N"); a.grid(alpha=.3, which="both")
ax[0].set_ylabel("max. rel. deviation from Gauss weak form"); ax[0].legend(fontsize=7, loc="lower left"); save("fig_quadrature_free")
# 3. basis conditioning (e15_conditioning.py)
d = pd.read_csv(os.path.join(RES, "E_conditioning.csv")); fig, ax = plt.subplots(1, 3, figsize=(12, 3.7))
sty = {"bernstein": ("o-", "tab:green", "Bernstein"), "legendre": ("s-", "tab:blue", "Legendre"), "chebyshev": ("^-", "tab:orange", "Chebyshev"), "monomial": ("x--", "0.5", "monomial")}
for k, (m, c, l) in sty.items():
    x = d[(d.case == "beam_cf") & (d.basis == k)]
    ax[0].semilogy(x.n, x.kappa_M, m, c=c, label=l); ax[1].semilogy(x.n, x.kappa_K, m, c=c, label=l); ax[2].semilogy(x.n, x.err_pct.clip(lower=1e-14), m, c=c, label=l)
for a, t in zip(ax, ["(a) $\\kappa(M_{red})$", "(b) $\\kappa(K_{red})$", "(c) max. rel. error of 5 lowest modes (%)"]):
    a.set_title(t); a.set_xlabel("polynomial degree $n$"); a.grid(alpha=.3, which="both")
ax[0].legend(fontsize=8); save("fig_basis_conditioning")
print("additional figures written to", FIG)
