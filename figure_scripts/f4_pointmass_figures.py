"""Concentrated-mass figures: fig_H_pointmass_summary (accuracy, cost, cost vs modes) and fig_mass_modeshapes (mode 2 and shear, H1-H4)."""
from _fig import *
from pielm.pointmass import *

T = load_json("hard")["table"]; N_STRONG = load_json("hard")["N_strong_per_element"]
GREEN, BLUE, GREY, RED = "#2ca02c", "#1f77b4", "#7f7f7f", "#d62728"
KEYS = ["H1", "H2", "H3", "H4"]

def fig_summary():
    x = np.arange(4); w = 0.26; fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for i, (m, lab, col) in enumerate((("weak", "Weak form (1 solve)", GREEN), ("strong_brent", "Eig-PIELM + Brent iteration", BLUE), ("strong_onepass", "Eig-PIELM, single pass ($\\lambda^*=0$)", RED))):
        axs[0].bar(x + (i - 1) * w, [max(T[k][m]["max_err"], FLOOR) for k in KEYS], w, color=col, label=lab)
    axs[0].set_yscale("log"); axs[0].set_ylabel("max. eigenfrequency error, 5 modes (%)"); axs[0].set_ylim(1e-14, 1e4); axs[0].legend(fontsize=7, loc="upper left"); axs[0].set_title("(a) accuracy", loc="left", fontsize=9)
    for i, (m, col) in enumerate((("weak", GREEN), ("strong_brent", BLUE))):
        t = np.array([T[k][m]["time"] for k in KEYS]) * 1e3; lo = t - np.array([T[k][m]["q25"] for k in KEYS]) * 1e3; hi = np.array([T[k][m]["q75"] for k in KEYS]) * 1e3 - t
        axs[1].bar(x + (i - 0.5) * 0.36, t, 0.36, color=col, yerr=[lo, hi], error_kw=dict(lw=0.7, capsize=2))
        for xi, tt, k in zip(x + (i - 0.5) * 0.36, t, KEYS): axs[1].text(xi, tt * 1.15, f"{T[k][m]['solves']} solve" + ("s" if T[k][m]["solves"] > 1 else ""), ha="center", fontsize=6.5)
    axs[1].set_yscale("log"); axs[1].set_ylabel("wall time (ms)"); axs[1].set_ylim(0.5, 1000); axs[1].set_title("(b) cost, first 5 modes", loc="left", fontsize=9)
    for a in axs[:2]: a.set_xticks(x); a.set_xticklabels(["H1", "H2", "H3", "H4"], fontsize=8)
    dd = load_csv("H_nmodes")
    axs[2].plot(dd.nmodes, dd.t_weak * 1e3, "o-", color=GREEN, label="weak (one solve)"); axs[2].plot(dd.nmodes, dd.t_strong * 1e3, "s-", color=BLUE, label="Eig-PIELM + Brent iteration")
    for n_, t_, s_ in zip(dd.nmodes, dd.t_strong * 1e3, dd.solves_strong):
        if s_ > 0: axs[2].text(n_, t_ * 1.25, str(s_), fontsize=6.5, color=BLUE, ha="center")
    axs[2].set_yscale("log"); axs[2].set_xlabel("number of requested modes"); axs[2].set_ylabel("time (ms)"); axs[2].legend(fontsize=7, loc="center right"); axs[2].set_title("(c) cost vs modes (H2; numbers = solves)", loc="left", fontsize=9)
    fig.tight_layout(); save(fig, "fig_H_pointmass_summary")

CASEDEF = {"H1": lambda: PointMassBeam("C", "F", None, 0, 0, m_tip=1.0 * rA * Lb), "H2": lambda: PointMassBeam("S", "S", 0.3 * Lb, 1.0 * rA * Lb),
           "H3": lambda: PointMassBeam("C", "F", 0.6 * Lb, 2.0 * rA * Lb), "H4": lambda: PointMassBeam("C", "F", 0.4 * Lb, 1.0 * rA * Lb, k=100 * EI / Lb ** 3)}

def fig_modeshapes(mode=1):
    """Mode `mode`+1 (0-based) and shear w''' : reference (2-element weak, n=40), weak n=23, strong Brent n=23, strong single pass."""
    n = 23; x = np.linspace(0, Lb, 1201); fig, axs = plt.subplots(2, 4, figsize=(13, 5.2), sharex=True)
    for c, key in enumerate(KEYS):
        b = CASEDEF[key](); ref = b.weak(40, 60, vectors=True, rcond=1e-12); ws = b.weak(n, 40, vectors=True)
        st = b.strong(n, N=N_STRONG, mode="brent", vectors=True); sp = b.strong(n, N=N_STRONG, mode="onepass", vectors=True)
        j = mode; u0 = b.field(ref.beta[:, j], 40, x, 0); sc = np.abs(u0).max()
        for res, lst, col, lw, lab in ((ref, "-", "k", 1.5, "reference (2-element weak, $n=40$, truncated)"), (ws, "--", GREEN, 1.2, "weak, $n=23$"),
                                       (st, ":", BLUE, 1.8, "Eig-PIELM, Brent-iterated"), (sp, "-.", RED, 1.0, "Eig-PIELM, single pass")):
            nn = 40 if res is ref else n; u = b.field(res.beta[:, j], nn, x, 0); s = np.dot(u, u0) / np.dot(u, u)
            axs[0, c].plot(x / Lb, s * u / sc, lst, color=col, lw=lw, label=lab)
            axs[1, c].plot(x / Lb, s * b.field(res.beta[:, j], nn, x, 3) / sc * Lb ** 3, lst, color=col, lw=lw)
        if b.x0: [a.axvline(b.x0 / Lb, color=GREY, ls=":", lw=0.8) for a in axs[:, c]]
        axs[0, c].set_title(key, fontsize=9); axs[1, c].set_xlabel("$x/L$")
        lim = np.abs(b.field(ref.beta[:, j], 40, x, 3) / sc * Lb ** 3).max(); axs[1, c].set_ylim(-1.4 * lim, 1.4 * lim)
    axs[0, 0].set_ylabel("$w(x)$ (normalized)"); axs[1, 0].set_ylabel("$L^3 w'''(x)$ (normalized)"); axs[0, 0].legend(fontsize=6.5, loc="lower center")
    fig.tight_layout(); save(fig, "fig_mass_modeshapes")

if __name__ == "__main__":
    fig_summary(); fig_modeshapes()
