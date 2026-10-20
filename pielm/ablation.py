"""Natural-row ablation on the SSSS and SSFF Kirchhoff plates (paper Fig. 8): shared by experiments/e15_natural_row_ablation.py and figure_scripts/fig_plate_natural_rows.py.

Rows of the figure, all on the same frozen Bernstein space (degree n) and the same Gauss weights (ng points):
  weak          Ritz on V_w, essential rows only
  strong_all    exact-weight strong pencil with all rows (Ritz on V_s)
  strong_noNat  exact-weight strong pencil, natural rows removed (M_n = 0 for SSSS, V_n = 0 for SSFF).  ABLATION, not a method.

Selection rules (fixed, no tuning):
  weak, strong_all  : the non-degenerate target modes -- SSSS: (m,m) = (1,1),(2,2),(3,3); SSFF: modes 1,2,3 of the weak form -- chosen as the real eigenpair
                      with the largest normalised L2 overlap with the target.
  strong_noNat      : the three LOWEST REAL eigenpairs (lambda > 1).  The ablated pencil is not symmetric and most of its eigenvalues are complex.
"""
import numpy as np
from .kirchhoff import Plate
from .bases import Bern1D

GRID = np.linspace(0, 1, 101)


def _field(beta, n):
    B0 = Bern1D(1.0, n).d(GRID, 0); return B0 @ beta.reshape(n + 1, n + 1) @ B0.T


def real_pairs(lam, beta):
    ok = np.isfinite(lam) & (np.abs(lam.imag) <= 1e-8 * np.abs(lam.real) + 1e-300) & (lam.real > 1.0)
    i = np.flatnonzero(ok); i = i[np.argsort(lam.real[i])]; return np.sqrt(lam.real[i]), beta[:, i].real


def _match(om, B, target, n):
    t = target.ravel() / np.linalg.norm(target)
    ov = np.array([abs(_field(B[:, k], n).ravel() @ t) / np.linalg.norm(_field(B[:, k], n)) for k in range(B.shape[1])])
    k = int(np.argmax(ov)); return k, float(ov[k])


def ablation(case, n=16, ng=None):
    """Returns dict with, for each method, lists of (omega, beta, overlap or None) plus diagnostics."""
    ng = ng or n + 12; P = Plate(case); P.nmodes = 24
    wk = P.weak(n, 40, rcond=1e-12, vectors=True, symmetric=True)
    la, Ba, _ = P.strong_gauss(n, ng, natural=True); oa, Bar = real_pairs(la, Ba)
    ln, Bn, info = P.strong_gauss(n, ng, natural=False); on, Bnr = real_pairs(ln, Bn)
    if case == "SSSS":
        m = (1, 2, 3); tg = [np.outer(np.sin(k * np.pi * GRID), np.sin(k * np.pi * GRID)) for k in m]; ex = [2 * np.pi ** 2 * k * k for k in m]
    else:
        tg = [_field(wk.beta[:, j], n) for j in range(3)]; ex = [float(wk.omega[j]) for j in range(3)]
    out = dict(case=case, n=n, ng=ng, exact=ex, n_real_noNat=int(on.size), n_total=int(ln.size), weak=[], strong_all=[], strong_noNat=[])
    for t in tg:
        k, ov = _match(wk.omega, wk.beta, t, n); out["weak"].append((float(wk.omega[k]), wk.beta[:, k], ov))
        k, ov = _match(oa, Bar, t, n); out["strong_all"].append((float(oa[k]), Bar[:, k], ov))
    for k in range(3):
        _, ov_best = _match(on[k:k + 1], Bnr[:, k:k + 1], tg[k], n)
        out["strong_noNat"].append((float(on[k]), Bnr[:, k], ov_best))
    # diagnostic: best-overlap real eigenpair of the ablated pencil for each target
    out["noNat_best_match"] = [dict(omega=float(on[(r := _match(on, Bnr, t, n))[0]]), overlap=r[1]) for t in tg]
    return out
