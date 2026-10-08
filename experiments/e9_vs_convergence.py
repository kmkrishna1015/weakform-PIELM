"""Does the strong-form admissible space V_s (essential+natural+corner rows) still converge under EXACT integration?
Compares, per degree n, the max rel. error over 6 modes of: weak (Ritz on V_w), Ritz on V_s, strong pencil with Gauss weights,
and the published strong pencil (uniform points, unit weights).  All with identical truncation rcond."""
from _common import *
from pielm.kirchhoff import Plate
RC = 1e-12
def strong_gauss(P, n, ng):
    b = Bern1D(1.0, n); x, w = gl(0, 1.0, ng)
    A = {c: P._ax(b, x, w, 0, c) for c in (0, 2, 4)}
    S = np.kron(A[4], A[0]) + 2*np.kron(A[2], A[2]) + np.kron(A[0], A[4]); G = np.kron(A[0], A[0])
    Z, rk = null_space_svd(P._rows(b, n+1, strong=True))
    lam, V, kept = solve_pencil(Z.T@S@Z, Z.T@G@Z, RC); idx, om = lowest(lam, 6, 1.0); return om, Z.shape[1]
def ritz(K, M, Z):
    lam, V, kept = solve_pencil(Z.T@K@Z, Z.T@M@Z, RC, True); idx, om = lowest(lam, 6, 1.0); return om
rows = []
for code in ("FFFF", "CFFF"):
    P = Plate(code); ref = P.weak(n=30, ng=40, rcond=RC, symmetric=True).omega[:6]
    for n in (8, 10, 12, 14, 16, 18, 20, 24):
        K, M, Zw, Zs = P.ritz_matrices(n, ng=n+12)
        ow, os_ = ritz(K, M, Zw), ritz(K, M, Zs)
        og, dim_s = strong_gauss(P, n, n+12)
        ou = P.strong(n=n, npts=300, rcond=RC).omega
        e = lambda o: float(np.max(np.abs(o[:6]-ref)/ref)) if len(o) >= 6 else np.nan
        rows.append(dict(case=code, n=n, nphi=(n+1)**2, dim_Vw=Zw.shape[1], dim_Vs=Zs.shape[1],
                         err_weak=e(ow), err_RitzVs=e(os_), err_strong_gauss=e(og), err_strong_uniform=e(ou)))
        print(rows[-1], flush=True)
save_csv(pd.DataFrame(rows), "V_convergence")
