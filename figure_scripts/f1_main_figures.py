from _fig import *
from matplotlib.lines import Line2D

NAME = {"bar_ff": "Bar\nfixed–fixed", "bar_fx": "Bar\nfixed–free", "beam_ss": "Beam\nsimply supp.", "beam_ff": "Beam\nfixed–fixed",
        "beam_cf": "Beam\ncantilever", "helmholtz": "2D Helmholtz\ncavity", "plate": "2D plate\n(in-plane)"}
P = problems(); BM = load_json("benchmarks"); TM = load_json("timing")

# ---------------------------------------------------------------- Fig. modes 1D
def fig_modes_1d():
    keys = ["bar_ff", "bar_fx", "beam_ss", "beam_ff", "beam_cf"]
    fig, axs = plt.subplots(5, 5, figsize=(11, 8), sharex="col")
    for c, key in enumerate(keys):
        p = P[key]; B = bases(p); x = np.linspace(0, p.L, 401); xs = np.linspace(0, p.L, 17)
        rw = run_method(p, "weak", vectors=True); rs = run_method(p, "strong", vectors=True)
        for j in range(5):
            ue = p.shape_fn(j, x); ue = ue / np.abs(ue).max()
            ax = axs[j, c]; ax.plot(x / p.L, ue, "k-", lw=1.6, label="Exact")
            for r, mk, col, lab in ((rw, "o", COL["weak"], "Weak"), (rs, "x", COL["strong"], "Eig-PIELM")):
                u = p.eval_shape(B, r.beta[:, j], x); s = np.dot(u, ue) / np.dot(u, u); ax.plot(xs / p.L, (s * p.eval_shape(B, r.beta[:, j], xs)), mk, ms=4.5 if mk == "o" else 5, mfc="none" if mk == "o" else col, color=col, label=lab)
            ax.set_yticks([-1, 0, 1]); ax.set_ylim(-1.25, 1.25)
            if c == 0: ax.set_ylabel(f"mode {j+1}")
            if j == 0: ax.set_title(NAME[key].replace("\n", " "), fontsize=9)
            if j == 4: ax.set_xlabel("$x/L$")
    h = [Line2D([], [], color="k", lw=1.6), Line2D([], [], color=COL["weak"], marker="o", mfc="none", ls=""), Line2D([], [], color=COL["strong"], marker="x", ls="")]
    fig.legend(h, ["Exact", "Weak form (present study)", "Eig-PIELM (published)"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.02, 1, 1)); save(fig, "fig_modes_1d")


def _grid2d(nrows, figsize):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, 5, width_ratios=[1, 1, 0.05, 1, 0.05], wspace=0.75, hspace=0.12)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in (0, 1, 3)] for i in range(nrows)])
    for i in range(nrows):
        for j in (1, 2): axs[i, j].sharey(axs[i, 0])
        for j in (0, 1, 2): axs[i, j].sharex(axs[nrows - 1, j])
    for i in range(nrows):
        for j in (1, 2): axs[i, j].tick_params(labelleft=False)
        for j in range(3):
            if i < nrows - 1: axs[i, j].tick_params(labelbottom=False)
    cax1 = fig.add_subplot(gs[:, 2]); cax2 = fig.add_subplot(gs[:, 4])
    return fig, axs, cax1, cax2

def fig_modes_helmholtz():
    p = P["helmholtz"]; B = bases(p); r = run_method(p, "weak", vectors=True)
    X, Y = np.linspace(0, p.Lx, 121), np.linspace(0, p.Ly, 73)
    fig, axs, cax1, cax2 = _grid2d(5, (8.6, 8.4))
    for j in range(5):
        ue = p.exact_shape(j, X, Y); un = p.eval_shape(*B, r.beta[:, j], X, Y); s = np.sum(un * ue) / np.sum(un * un); un = s * un
        for c, (f, ttl) in enumerate(((ue, "Exact"), (un, "Weak form (present study)"), (np.abs(un - ue), "|difference|"))):
            ax = axs[j, c]; ax.grid(False)
            if c < 2: im = ax.contourf(X, Y, f.T, levels=np.linspace(-1, 1, 21), cmap="RdBu_r", extend="both")
            else: imd = ax.contourf(X, Y, np.maximum(f.T, 1e-14), levels=np.logspace(-13, -6, 15), norm=matplotlib.colors.LogNorm(1e-13, 1e-6), cmap="viridis", extend="both")
            ax.set_aspect("equal")
            if j == 0: ax.set_title(ttl)
            if c == 0: ax.set_ylabel(f"mode {j+1}\n$y$ (m)")
            if j == 4: ax.set_xlabel("$x$ (m)")
    fig.colorbar(im, cax=cax1, label="$P(x,y)$"); cax1.yaxis.set_label_position("left"); cax1.yaxis.labelpad = 4; cax2.yaxis.labelpad = 6; fig.colorbar(imd, cax=cax2, label="|error|")
    save(fig, "fig_modes_helmholtz")

def fig_modes_plate():
    p = P["plate"]; B = bases(p); r = run_method(p, "weak", vectors=True)
    X, Y = np.linspace(0, p.Lx, 101), np.linspace(0, p.Ly, 61)
    fig, axs, cax1, cax2 = _grid2d(6, (8.6, 9.4))
    for j in range(6):
        ex, ey = p.exact_shape(j, X, Y); ux, uy = p.eval_field(*B, r.beta[:, j], X, Y)
        s = (np.sum(ux * ex) + np.sum(uy * ey)) / (np.sum(ux ** 2) + np.sum(uy ** 2)); ux, uy = s * ux, s * uy
        mag = lambda a, b: np.sqrt(a ** 2 + b ** 2); nm = mag(ex, ey).max()
        fields = [mag(ex, ey) / nm, mag(ux, uy) / nm, mag(ux - ex, uy - ey) / nm]
        for c in range(3):
            ax = axs[j, c]; ax.grid(False); ax.set_aspect("equal")
            if c < 2: im = ax.contourf(X, Y, fields[c].T, levels=np.linspace(0, 1, 21), cmap="magma")
            else: imd = ax.contourf(X, Y, fields[c].T, levels=np.logspace(-13, -5, 17), norm=matplotlib.colors.LogNorm(1e-13, 1e-5), cmap="viridis", extend="both")
            if j == 0: ax.set_title(["Exact $|\\mathbf{u}|$", "Weak form (present study) $|\\mathbf{u}|$", "|difference|"][c])
            if c == 0: ax.set_ylabel(f"mode {j+1}\n$y$ (m)")
            if j == 5: ax.set_xlabel("$x$ (m)")
    fig.colorbar(im, cax=cax1, label="normalized $|\\mathbf{u}|$"); cax1.yaxis.set_label_position("left"); fig.colorbar(imd, cax=cax2, label="|error|")
    save(fig, "fig_modes_plate")

def fig_pareto():
    fig, ax = plt.subplots(figsize=(5.4, 4)); h = "helmholtz"
    for m, mk, col, lab in (("strong", "v", COL["strong"], "Eig-PIELM (published, general solver)"), ("weak", "o", COL["weak"], "Weak form, same general solver"),
                            ("weak_sym", "D", "#17becf", "Weak form, symmetric solver")):
        ax.plot(TM["cases"][h][f"{m}|analytic"]["median"], max(BM[h]["methods"][m]["max_err_pct"], FLOOR), mk, ms=8, color=col, label=lab)
    fem = BM["fem_helmholtz"]
    ax.plot([v["time"] for v in fem.values()], [v["max_err_pct"] for v in fem.values()], "s-", color="k", ms=6, label="Q4 FEM (196, 1e4, 4e4 dof)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("wall time (s)"); ax.set_ylabel("max. error, first 5 modes (%)"); ax.legend(fontsize=6.5, loc="upper right"); ax.set_title("2D Helmholtz cavity: accuracy vs cost", fontsize=9)
    fig.tight_layout(); save(fig, "fig_pareto_helmholtz")


if __name__ == "__main__":
    for f in (fig_modes_1d, fig_modes_helmholtz, fig_modes_plate, fig_pareto): f()
