"""Weak form WITHOUT a designed quadrature rule: all integrals are replaced by sums over scattered points.
  gauss     : tensor Gauss-Legendre (reference)
  grid      : uniform midpoint grid, equal weights |Omega|/N          (Riemann sum)
  halton_eq : scattered Halton points, equal weights |Omega|/N        (quasi-Monte-Carlo)
  halton_mf : same scattered points, moment-fitted weights (min-norm LS so all products of degree<=2n are integrated exactly)
Error is the max rel. deviation of the 5/6 lowest eigenfrequencies from the Gauss weak form at the same degree (isolates the
quadrature effect); Gauss-vs-exact is reported separately."""
from _common import *
from pielm.kirchhoff import Plate
from scipy.stats import qmc
from numpy.polynomial import legendre as Lg
RC = 1e-12

def moment_weights(X, Lx, Ly, q):
    s = 2*X[:, 0]/Lx - 1; t = 2*X[:, 1]/Ly - 1
    Ps, Pt = Lg.legvander(s, q), Lg.legvander(t, q)
    A = (Ps[:, :, None]*Pt[:, None, :]).reshape(len(X), -1).T          # (q+1)^2 x N
    m = np.zeros((q+1, q+1)); m[0, 0] = Lx*Ly
    w, *_ = np.linalg.lstsq(A, m.ravel(), rcond=None)
    return w, float(np.linalg.norm(A@w - m.ravel()))

def points(kind, N, Lx, Ly, n):
    if kind == "gauss":
        g = n + 10; x, wx = gl(0, Lx, g); y, wy = gl(0, Ly, g); X, Y = np.meshgrid(x, y, indexing="ij")
        return np.c_[X.ravel(), Y.ravel()], np.outer(wx, wy).ravel()
    if kind == "grid":
        k = int(round(np.sqrt(N))); x = (np.arange(k)+.5)*Lx/k; y = (np.arange(k)+.5)*Ly/k; X, Y = np.meshgrid(x, y, indexing="ij")
        return np.c_[X.ravel(), Y.ravel()], np.full(k*k, Lx*Ly/(k*k))
    P = qmc.Halton(2, scramble=False).random(N)*np.array([Lx, Ly])
    if kind == "halton_eq": return P, np.full(N, Lx*Ly/N)
    w, res = moment_weights(P, Lx, Ly, 2*n); return P, w

def tens(bx, by, X, a, c):          # rows: d^a/dx^a d^c/dy^c phi_ij at points, column i*(ny+1)+j
    return np.einsum('pi,pj->pij', bx.d(X[:, 0], a), by.d(X[:, 1], c)).reshape(len(X), -1)

def helm_KM(n, Lx, Ly, X, w):
    bx, by = Bern1D(Lx, n), Bern1D(Ly, n)
    F, Fx, Fy = tens(bx, by, X, 0, 0), tens(bx, by, X, 1, 0), tens(bx, by, X, 0, 1)
    return Fx.T@(w[:, None]*Fx) + Fy.T@(w[:, None]*Fy), F.T@(w[:, None]*F), None

def plate_KM(n, X, w, nu=0.3):
    b = Bern1D(1.0, n); F = tens(b, b, X, 0, 0); Fxx, Fyy, Fxy = tens(b, b, X, 2, 0), tens(b, b, X, 0, 2), tens(b, b, X, 1, 1)
    W = w[:, None]
    K = Fxx.T@(W*Fxx) + Fyy.T@(W*Fyy) + nu*(Fxx.T@(W*Fyy) + Fyy.T@(W*Fxx)) + 2*(1-nu)*Fxy.T@(W*Fxy)
    return K, F.T@(W*F)

def eig(K, M, Z, nm, floor):
    lam, V, kept = solve_pencil(Z.T@K@Z, Z.T@M@Z, RC, True); idx, om = lowest(lam, nm, floor); return om

rows = []
# ---------------- Helmholtz cavity (Neumann, no essential rows), n=13
Lx, Ly, n = 5.0, 3.0, 13; ph = problems()["helmholtz"]; ex = ph.exact(5)
Pg, wg = points("gauss", 0, Lx, Ly, n); K, M, _ = helm_KM(n, Lx, Ly, Pg, wg); Z = np.eye(K.shape[0])
og = eig(K, M, Z, 5, 1e-6*ph.k2[0][0]) ; ref = og*340.0/340.0
exw = np.sqrt(np.array([t[0] for t in ph.k2[:5]]))        # sqrt(k^2)
gauss_err = float(np.max(np.abs(og-exw)/exw)); print("Helmholtz Gauss vs exact", gauss_err)
for kind in ("grid", "halton_eq", "halton_mf"):
    for N in (1000, 2000, 5000, 10000, 20000):
        P, w = points(kind, N, Lx, Ly, n)
        if kind == "halton_mf" and N < 3*(2*n+1)**2: continue
        K, M, _ = helm_KM(n, Lx, Ly, P, w); o = eig(K, M, Z, 5, 1e-6*ph.k2[0][0])
        e = float(np.max(np.abs(o[:5]-og)/og)) if len(o) >= 5 else np.nan
        rows.append(dict(case="Helmholtz", n=n, method=kind, N=len(P), dev_from_gauss=e, gauss_vs_exact=gauss_err, min_w=float(w.min())))
        print(rows[-1], flush=True)
# ---------------- Kirchhoff FFFF, CFFF, n=12 (unit square)
n = 12
for code in ("FFFF", "CFFF"):
    Pl = Plate(code); Zw, _ = null_space_svd(Pl._rows(Bern1D(1.0, n), n+1, strong=False))
    Pg, wg = points("gauss", 0, 1.0, 1.0, n); K, M = plate_KM(n, Pg, wg); og = eig(K, M, Zw, 6, 1.0)
    for kind in ("grid", "halton_eq", "halton_mf"):
        for N in (2000, 5000, 10000, 20000, 40000):
            P, w = points(kind, N, 1.0, 1.0, n)
            K, M = plate_KM(n, P, w); o = eig(K, M, Zw, 6, 1.0)
            e = float(np.max(np.abs(o[:6]-og)/og)) if len(o) >= 6 else np.nan
            rows.append(dict(case="Plate "+code, n=n, method=kind, N=len(P), dev_from_gauss=e, gauss_vs_exact=np.nan, min_w=float(w.min())))
            print(rows[-1], flush=True)
save_csv(pd.DataFrame(rows), "quadrature_free")
