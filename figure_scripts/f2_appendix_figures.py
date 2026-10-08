"""Appendix figures (weak vs published strong form, same eigensolver).  Reads results/*.csv."""
from _fig import *
BM = load_json("benchmarks")
CASES = (("bar_fx", "Bar, fixed–free"), ("beam_cf", "Beam, cantilever"), ("helmholtz", "2D Helmholtz"), ("plate", "2D plate"))

def fig_S1():   # error vs number of basis functions: form x solver (plain general solve / identical truncation)
    d = load_csv("S1_degree")
    sty = {"strong_bern_plain": ("Eig-PIELM, plain solve", COL["strong"], "v-"),
           "strong_bern_trunc": ("Eig-PIELM, truncation", COL["strong"], "v:"),
           "weak_bern_plain": ("Weak, plain solve", COL["weak"], "o-"),
           "weak_bern_trunc": ("Weak, truncation", COL["weak"], "o:"),
           "weak_lege_plain": ("Weak, Legendre, plain solve", "#9467bd", "d-"),
           "weak_lege_trunc": ("Weak, Legendre, truncation", "#9467bd", "d:")}
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.3))
    for a, (c, t) in zip(axs, CASES):
        for m, (lab, col, st) in sty.items():
            y = d[(d.case == c) & (d.method == m)]; a.plot(y.nphi, np.maximum(y.max_err_pct, FLOOR), st, color=col, ms=3.5, lw=1, label=lab)
        a.set_yscale("log"); a.set_title(t, fontsize=9); a.set_xlabel("$N_\\phi$"); a.set_ylim(1e-14, 1e3)
    axs[0].set_ylabel("max. eigenfrequency error (%)"); h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.16)); fig.tight_layout(); save(fig, "fig_S1_degree")

def fig_S2():   # error vs number of collocation points (published Eig-PIELM); weak form shown as a level (independent of N)
    d = load_csv("S2_collocation"); fig, axs = plt.subplots(1, 4, figsize=(13, 3.3))
    for a, (c, t) in zip(axs, CASES):
        y = d[(d.case == c) & (d.method == "strong")].sort_values("N")
        a.plot(y.N, np.maximum(y.max_err_pct, FLOOR), "o-", color=COL["strong"], ms=3.5, label="Eig-PIELM (published; general solve)")
        a.axhline(max(BM[c]["methods"]["weak"]["max_err_pct"], FLOOR), color=COL["weak"], ls="--", lw=1.2, label="Weak form (Gauss quadrature; independent of $N$)")
        a.set_xscale("log"); a.set_yscale("log"); a.set_title(t, fontsize=9); a.set_xlabel("number of sample points $N$"); a.set_ylim(1e-14, 1e3)
    axs[0].set_ylabel("max. eigenfrequency error (%)"); h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.1)); fig.tight_layout(); save(fig, "fig_S2_collocation")

def fig_S3():   # truncation threshold (both forms), Gram conditioning, cantilever-beam degree sweep
    r = load_csv("S3_rcond"); c = load_csv("S3_cond"); fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    for key, ls in (("beam_cf", "-"), ("beam_ff", "--"), ("bar_fx", ":")):
        for m in ("strong", "weak"):
            y = r[(r.case == key) & (r.method == m)].sort_values("rcond")
            axs[0].plot(y.rcond, np.maximum(y.max_err_pct, FLOOR), ls, color=COL[m], marker="o" if m == "weak" else "v", ms=3, label=f"{key}, {m}")
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("truncation threshold rcond"); axs[0].set_ylabel("max. error (%)"); axs[0].axvline(1e-12, color="k", lw=0.7, ls=":"); axs[0].legend(fontsize=5.5, ncol=2)
    for b, col in (("bernstein", "#2ca02c"), ("legendre", "#9467bd")):
        y = c[c.basis == b]; axs[1].plot(y.deg + 1, y.cond_gram, "o-", color=col, label=b.capitalize())
    axs[1].axhline(1e16, color="k", ls=":", lw=0.7); axs[1].text(8, 2e16, "1/$\\epsilon_{mach}$", fontsize=7); axs[1].set_yscale("log"); axs[1].set_xlabel("$N_\\phi$"); axs[1].set_ylabel("cond(Gram matrix)"); axs[1].legend(fontsize=7)
    s = load_csv("S1_degree"); y1 = s[(s.case == "beam_cf")]
    for m, lab, col, st in (("strong_bern_plain", "Eig-PIELM, plain", COL["strong"], "v-"), ("strong_bern_trunc", "Eig-PIELM, truncation", COL["strong"], "v:"),
                            ("weak_bern_plain", "Weak, plain", COL["weak"], "o-"), ("weak_bern_trunc", "Weak, truncation", COL["weak"], "o:"),
                            ("weak_lege_plain", "Weak Legendre, plain", "#9467bd", "d-")):
        y = y1[y1.method == m]; axs[2].plot(y.nphi, np.maximum(y.max_err_pct, FLOOR), st, color=col, ms=3.5, label=lab)
    axs[2].set_yscale("log"); axs[2].set_xlabel("$N_\\phi$"); axs[2].set_ylabel("max. error (%)"); axs[2].legend(fontsize=6)
    for a, t in zip(axs, ("(a) truncation threshold, both forms", "(b) Gram conditioning (1D)", "(c) cantilever beam, plain vs truncated solve")): a.set_title(t, loc="left", fontsize=8)
    fig.tight_layout(); save(fig, "fig_S3_conditioning")

def fig_S5():   # timing vs basis size: published Eig-PIELM vs weak form
    d = load_csv("S5_timing_scaling"); fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    nm = {"strong": "Eig-PIELM (published)", "weak": "Weak, same general solver", "weak_sym": "Weak, symmetric solver"}
    cl = {"strong": COL["strong"], "weak": COL["weak"], "weak_sym": "#17becf"}
    for a, (c, t) in zip(axs, (("beam_cf", "Beam, cantilever"), ("helmholtz", "2D Helmholtz"))):
        for m in ("strong", "weak", "weak_sym"):
            for dv, ls in (("analytic", "-"), ("autodiff", "--")):
                y = d[(d.case == c) & (d.method == m) & (d.deriv == dv)].sort_values("nphi")
                if len(y): a.plot(y.nphi, y["median"] * 1e3, ls, marker="o", ms=3, color=cl[m], label=f"{nm[m]}, {dv}")
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("$N_\\phi$"); a.set_title(t, fontsize=9)
    axs[0].set_ylabel("time (ms)"); hl = {}
    for a_ in axs:
        for h_, l_ in zip(*a_.get_legend_handles_labels()): hl.setdefault(l_, h_)
    h, l = list(hl.values()), list(hl.keys()); fig.legend(h, l, ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.12)); fig.tight_layout(); save(fig, "fig_S5_timing_scaling")

if __name__ == "__main__":
    for f in (fig_S1, fig_S2, fig_S3, fig_S5): f()
