"""Paper Fig. 4: role of the natural rows on the SSSS and SSFF plates (weak form / strong form all rows / strong form without natural rows).
Computes everything itself through pielm.ablation (same code as experiments/e15_natural_row_ablation.py).  Output: figures/fig_plate_modes.{pdf,png}"""
from _fig import *
from pielm.ablation import ablation, GRID
N = 16
ROWS = (("weak", "weak form\n(essential rows only)"), ("strong_all", "strong form,\nall rows (exact weights)"),
        ("strong_noNat", "strong form, no {nat}\n(exact weights; 3 lowest\nreal eigenpairs)"))
NAT = {"SSSS": "$M_n=0$", "SSFF": "$V_n=0$"}
fig, axs = plt.subplots(6, 3, figsize=(6.3, 9.8)); kw = dict(levels=np.linspace(-1, 1, 21), cmap="RdBu_r")
from pielm.ablation import _field
for c, case in enumerate(("SSSS", "SSFF")):
    r = ablation(case, N); mask_x = True          # x = 0, 1 are simply supported in both cases (edge string x0,x1,y0,y1)
    for i, (key, lab) in enumerate(ROWS):
        for j in range(3):
            ax = axs[3 * c + i, j]; om, beta, _ = r[key][j]; F = _field(beta, N); F = F / np.abs(F).max()
            F = F if F.max() >= -F.min() else -F; ax.grid(False); ax.set_aspect("equal"); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.contourf(GRID, GRID, F.T, **kw)
            Fm = np.ma.array(F.copy()); Fm[0, :] = np.ma.masked; Fm[-1, :] = np.ma.masked      # u = 0 on x = 0, 1: no round-off zero contours there
            if case == "SSSS": Fm[:, 0] = np.ma.masked; Fm[:, -1] = np.ma.masked
            ax.contour(GRID, GRID, Fm.T, levels=[0], colors="k", linewidths=0.7)
            ax.set_title(r"$\Omega_f=%.3f$" % om + (r" (exact %.3f)" % r["exact"][j] if key != "strong_noNat" else ""), fontsize=7, pad=2)
            if j == 0: ax.set_ylabel(f"{case}: " + lab.format(nat=NAT[case]), fontsize=6.5)
fig.tight_layout(h_pad=0.6, w_pad=0.4); save(fig, "fig_plate_modes")
