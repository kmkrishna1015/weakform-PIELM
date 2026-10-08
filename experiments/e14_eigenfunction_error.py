"""Eigenfunction errors (L2 and energy/H2-seminorm) of the Kirchhoff free-plate modes, n=16, vs the n=30 weak-form reference modes.
Degenerate clusters are compared as subspaces (Procrustes rotation in L2)."""
from _common import *
from pielm.kirchhoff import Plate
RC = 1e-12; nu = 0.3
def grid(m=60):
    x, w = gl(0, 1.0, m); X, Y = np.meshgrid(x, x, indexing="ij"); return x, np.outer(w, w).ravel()
def fields(n, x):
    b = Bern1D(1.0, n); f = lambda a, c: np.einsum('pi,qj->pqij', b.d(x, a), b.d(x, c)).reshape(len(x)**2, -1)
    return f(0, 0), f(2, 0), f(0, 2), f(1, 1)
def evalmodes(beta, n, x):
    F, Fxx, Fyy, Fxy = fields(n, x); return F @ beta, Fxx @ beta, Fyy @ beta, Fxy @ beta
def energy(u, w):
    _, uxx, uyy, uxy = u; return float(np.sum(w[:, None] * (uxx**2 + uyy**2 + 2*nu*uxx*uyy + 2*(1-nu)*uxy**2)))
x, w = grid()
out = {}
for code in ("FFFF", "CFFF"):
    P = Plate(code); P.nmodes = 9; ref = P.weak(30, 44, rcond=RC, symmetric=True, vectors=True); oref = ref.omega
    U_ref = evalmodes(ref.beta, 30, x)
    cand = {"weak": P.weak(16, 40, rcond=RC, symmetric=True, vectors=True)}
    K, M, Zw, Zs = P.ritz_matrices(16, 40)
    from pielm.numerics import solve_pencil, lowest
    lam, V, kept = solve_pencil(Zs.T@K@Zs, Zs.T@M@Zs, RC, True); idx, om = lowest(lam, 9, 1.0); cand["strong_exact"] = type("R", (), dict(omega=om, beta=(Zs@V[:, idx]).real))
    cand["strong_published"] = P.strong(16, 300, vectors=True)
    # clusters of the reference spectrum
    cl, cur = [], [0]
    for i in range(1, 9):
        if abs(oref[i]/oref[i-1]-1) < 1e-4: cur.append(i)
        else: cl.append(cur); cur = [i]
    cl.append(cur); cl = [c for c in cl if c[0] < 6]; out[code] = {}
    for name, r in cand.items():
        Uc = evalmodes(r.beta, 16, x); e2, eE = [], []
        for c in cl:
            Ur = U_ref[0][:, c]; Vc = Uc[0][:, c]
            # L2-orthonormalise both bases of the cluster, keeping the map to the other fields
            def orth(Ufld, c):
                A = Ufld[0][:, c]; G = A.T @ (w[:, None] * A); L = np.linalg.cholesky(G); T = np.linalg.inv(L).T; return [f[:, c] @ T for f in Ufld]
            Rr, Vv = orth(U_ref, c), orth(Uc, c)
            s, sg, tt = np.linalg.svd(Rr[0].T @ (w[:, None] * Vv[0])); Rot = (tt.T @ s.T)         # V Rot ~ R
            dif = [Vv[i] @ Rot - Rr[i] for i in range(4)]
            e2.append(float(np.sqrt(np.sum(w[:, None] * dif[0]**2) / np.sum(w[:, None] * Rr[0]**2))))
            eE.append(float(np.sqrt(sum(energy(tuple(d[:, k:k+1] for d in dif), w) for k in range(len(c))) / sum(energy(tuple(f[:, k:k+1] for f in Rr), w) for k in range(len(c))))))
        out[code][name] = dict(L2_max=max(e2), E_max=max(eE), L2_mode1=e2[0], E_mode1=eE[0], clusters=len(cl)); print(code, name, out[code][name], flush=True)
save_json(out, "eigenfunction_error")
