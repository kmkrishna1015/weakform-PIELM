"""Kirchhoff-plate figures: fig_plate_convergence, fig_pareto_plate, fig_plate_freemodes.  Reads results/P_plate.csv, P_fem.csv."""
from _fig import *
from pielm.kirchhoff import Plate, NU

d = load_csv("P_plate"); F = load_csv("P_fem"); FL = 1e-13
STY = {"weak": ("#2ca02c", "o-", "Weak-form PIELM (essential BCs only)"),
       "strong": ("#7f7f7f", "v--", "Eig-PIELM (published strong form: moment, shear, corner BCs)")}
TT = {"SSSS": "(a) simply supported [control]\nexact", "SFSF": "(b) two free edges\nexact Lévy solution",
      "FFFF": "(c) fully free plate\nself-converged ref. (= Leissa)", "CFFF": "(d) cantilever plate\nself-converged ref."}

def fig_plate_convergence():
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.7), sharey=True)
    for a, k in zip(axs, TT):
        for m, (col, st, lab) in STY.items():
            y = d[(d.case == k) & (d.method == m)].sort_values("nphi")
            a.plot(y.nphi, np.maximum(y.err, FL), st, color=col, ms=4, lw=1.1, label=lab)
        a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1e-13, 3); a.set_xlabel("$N_\\phi=(n+1)^2$"); a.set_title(TT[k], fontsize=9)
    axs[0].set_ylabel("max. error over first 6 modes (rel.)"); h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.08)); fig.tight_layout(); save(fig, "fig_plate_convergence")

def fig_pareto_plate():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, case, ttl in zip(axes, ("FFFF", "CFFF"), ("(a) Fully free plate", "(b) Cantilever plate")):
        w = d[(d.case == case) & (d.method == "weak") & (d.n >= 10)].sort_values("n")
        e = d[(d.case == case) & (d.method == "strong") & (d.n >= 10)].sort_values("n")
        f = F[F.case == case].sort_values("dof")
        ax.loglog(w.time * 1e3, np.maximum(w.err, 1e-9), "o-", color="tab:green", ms=6, label="Weak-form PIELM ($n=10$–16)")
        ax.loglog(e.time * 1e3, e.err, "v--", color="tab:gray", ms=6, label="Eig-PIELM (published, $n=10$–16)")
        ax.loglog(f.time_ms, f.err6, "s-", color="k", ms=6, label="FEM (BFS)")
        ax.set_xlabel("wall time (ms)"); ax.set_ylabel("max. relative error, 6 modes"); ax.set_title(ttl); ax.grid(True, which="both", alpha=0.3)
    h, l = axes[1].get_legend_handles_labels(); plt.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(h, l, ncol=3, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 0.0)); save(fig, "fig_pareto_plate")

def fig_plate_freemodes():
    P = Plate("FFFF"); n = 16; r = P.weak(n, 40, vectors=True); b = Bern1D(1.0, n); g = np.linspace(0, 1, 121); B0 = [b.d(g, k) for k in range(4)]
    ev = lambda beta, kx, ky: np.einsum("pi,qj,ij->pq", B0[kx], B0[ky], beta.reshape(n + 1, n + 1))
    fig, axs = plt.subplots(2, 3, figsize=(9, 5.6)); res = []
    for j, a in enumerate(axs.ravel()):
        w = ev(r.beta[:, j], 0, 0); w = w / np.abs(w).max(); a.grid(False)
        a.contourf(g, g, w.T, levels=np.linspace(-1, 1, 21), cmap="RdBu_r"); a.contour(g, g, w.T, levels=[0], colors="k", linewidths=0.8)
        a.set_aspect("equal"); a.set_title(f"mode {j+1}: $\\Omega={r.omega[j]:.4f}$", fontsize=9); a.set_xticks([0, 1]); a.set_yticks([0, 1])
        bt = r.beta[:, j]; Mx = ev(bt, 2, 0) + NU * ev(bt, 0, 2); Vx = ev(bt, 3, 0) + (2 - NU) * ev(bt, 1, 2)
        res.append((r.omega[j], np.abs(Mx[0, :]).max() / np.abs(Mx).max(), np.abs(Vx[0, :]).max() / np.abs(Vx).max()))
    fig.tight_layout(); save(fig, "fig_plate_freemodes")
    print("natural-BC residuals on edge x=0 of the weak solution (relative to interior max), n=16:")
    for om, m, v in res: print(f"  Omega={om:8.4f}  M_n {m:.1e}   V_n {v:.1e}")

if __name__ == "__main__":
    for f in (fig_plate_convergence, fig_pareto_plate, fig_plate_freemodes): f()
