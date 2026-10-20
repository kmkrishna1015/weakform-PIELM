"""Section 4.3 support: robustness checks of the bordered one-shot strong pencil (row scaling, exact Gauss weights in A and M, Loewdin truncation)."""
from _common import *
from scipy.linalg import eig, eigh
from pielm.pointmass import PointMassBeam, rA, Lb, EI
def setup(b, n=23, N=5000):
    Bs, G, S4, tot, Nb = b._strong_matrices(n, N)
    def rowsL(ls):
        rows = []
        if b.x0 is not None:
            r = np.zeros(tot); r[:Nb] = -EI * Bs[0].d([Bs[0].L], 3)[0]; r[Nb:] = EI * Bs[1].d([0.0], 3)[0]; rows.append(r - (b.m * ls - b.k) * b._phi0(Bs, tot, Nb, "x0"))
        if b.mt and b.bcR == "F":
            r = np.zeros(tot); B = Bs[-1]; r[(len(Bs) - 1) * Nb:] = EI * B.d([B.L], 3)[0] + b.mt * ls * B.d([B.L], 0)[0]; rows.append(r)
        return np.vstack(rows)
    Zf, _ = null_space_svd(np.vstack(b._ess(Bs, Nb, tot, True)))
    return Bs, G, S4, tot, Nb, Zf, rowsL(0.0) @ Zf, (rowsL(0.0) - rowsL(1.0)) @ Zf
def solve(A, M, c0, c1, ex, s=1.0):
    nc = c0.shape[0]; Kb = np.block([[A, s * c0.T], [s * c0, np.zeros((nc, nc))]]); Mb = np.block([[M, s * c1.T], [s * c1, np.zeros((nc, nc))]])
    lam = eig(Kb, Mb, right=False); ok = np.isfinite(lam) & (np.abs(lam.imag) <= 1e-8 * np.abs(lam.real) + 1e-300) & (lam.real > 1e-3); l = np.sort(lam.real[ok])[:5]
    return float(np.max(np.abs(np.sqrt(l) / ex - 1)) * 100) if len(l) == 5 else float("nan")
CASES = {"H1": lambda: PointMassBeam("C", "F", None, 0, 0, m_tip=1.0*rA*Lb), "H2": lambda: PointMassBeam("S", "S", 0.3*Lb, 1.0*rA*Lb),
         "H3": lambda: PointMassBeam("C", "F", 0.6*Lb, 2.0*rA*Lb), "H4": lambda: PointMassBeam("C", "F", 0.4*Lb, 1.0*rA*Lb, k=100*EI/Lb**3)}
out = {}
for key, f in CASES.items():
    b = f(); ex = b.exact(); Bs, G, S4, tot, Nb, Zf, c0, c1 = setup(b); A, M = Zf.T @ S4 @ Zf, Zf.T @ G @ Zf; o = {}
    o["plain"] = solve(A, M, c0, c1, ex)
    o["scale_norm_c0"] = solve(A, M, c0, c1, ex, 1 / np.linalg.norm(c0)); o["scale_normA_over_c0"] = solve(A, M, c0, c1, ex, np.linalg.norm(A) / np.linalg.norm(c0))
    Gx, Sx = np.zeros_like(G), np.zeros_like(S4)                                       # exact Gauss weights per element
    for i, B in enumerate(Bs):
        sl = slice(i * Nb, (i + 1) * Nb); x, w = gl(0, B.L, 40); P = B.d(x, 0); Gx[sl, sl] = P.T @ (w[:, None] * P); Sx[sl, sl] = (EI / rA) * P.T @ (w[:, None] * B.d(x, 4))
    o["exact_gauss_weights"] = solve(Zf.T @ Sx @ Zf, Zf.T @ Gx @ Zf, c0, c1, ex)
    Mh = 0.5 * (M + M.T); w_, U = eigh(Mh)
    for tol in (1e-10, 1e-12, 1e-14):
        keep = w_ > tol * w_.max(); Uk = U[:, keep] / np.sqrt(w_[keep]); o[f"loewdin_{tol:g}"] = solve(Uk.T @ A @ Uk, np.eye(Uk.shape[1]), c0 @ Uk, c1 @ Uk, ex)
    out[key] = o; print(key, o, flush=True)
save_json(out, "bordered_checks")
