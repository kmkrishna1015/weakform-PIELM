"""Appendix E: basis-equivalence control with conditioning evidence.
For Bernstein / Legendre / Chebyshev / monomial bases of the SAME degree n (same polynomial space), record
  kappa(M_red), kappa(K_red)  (2-norm condition numbers of the reduced matrices Z'MZ, Z'KZ exactly as used in Algorithm 2),
  the same after symmetric Jacobi scaling D^-1/2 A D^-1/2 (removes pure column scaling),
  the number of directions kept by the 1e-12 truncation of M_red (rcond relative to the largest eigenvalue of the UNSCALED M_red),
  the maximum relative eigenfrequency error (%) of the five lowest modes (weak form, Loewdin-reduced eigh, rcond = 1e-12)."""
from _common import *
from pielm.bases import make_extra
P = problems(); RC = 1e-12; NG = 70
def cond(A):
    s = np.linalg.svd(0.5 * (A + A.T), compute_uv=False); return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
def jac(A):
    d = np.sqrt(np.abs(np.diag(A))); return A / np.outer(d, d)
rows = []
for key in ("bar_fx", "beam_ss", "beam_cf"):
    p = P[key]; ex = p.exact(p.nmodes)
    for kind in ("bernstein", "legendre", "chebyshev", "monomial"):
        for n in (8, 12, 16, 20, 23, 28, 32, 36):
            b = make_extra(kind, p.L, n + 1); x, w = gl(0, p.L, NG); Phi, Dm = b.d(x, 0), b.d(x, p.m)
            K = p.kc * (Dm.T @ (w[:, None] * Dm)); M = p.mc * (Phi.T @ (w[:, None] * Phi))
            Z, rk = null_space_svd(np.vstack(p._bcrows(b, p.ess))); Kr, Mr = Z.T @ K @ Z, Z.T @ M @ Z
            r = p.weak(b, NG, rcond=RC, symmetric=True)
            err = float(np.max(rel_err_pct(r.omega, ex))) if len(r.omega) == p.nmodes else float("nan")
            rng = np.random.default_rng(0); l0 = None; ch = []
            def lam5(Kf, Mf):
                l, V, k = solve_pencil(Z.T @ Kf @ Z, Z.T @ Mf @ Z, RC, True); return np.sort(l.real)[:5]
            l0 = lam5(K, M)
            for _ in range(20):
                Kp = K * (1 + 1e-16 * rng.standard_normal(K.shape)); Kp = 0.5 * (Kp + Kp.T)
                Mp = M * (1 + 1e-16 * rng.standard_normal(M.shape)); Mp = 0.5 * (Mp + Mp.T)
                ch.append(float(np.max(np.abs(lam5(Kp, Mp) / l0 - 1))))
            rows.append(dict(case=key, basis=kind, n=n, dim=Z.shape[1], kappa_M=cond(Mr), kappa_K=cond(Kr), kappa_M_jac=cond(jac(Mr)), kappa_K_jac=cond(jac(Kr)),
                             kept=int(r.info["kept"]), err_pct=err, normK=float(np.linalg.norm(Kr, 2)), normM=float(np.linalg.norm(Mr, 2)), sens=float(np.median(ch))))
            
save_csv(pd.DataFrame(rows), "E_conditioning")
# ---- tensor-product identity for the Helmholtz cavity (Z = I): kappa(M) = kappa(Mx) kappa(My); Section E, item (iv)
kh = {}
for kind in ("bernstein", "legendre", "chebyshev"):
    ks = []
    for L in (5.0, 3.0):
        b = make_extra(kind, L, 14); x, w = gl(0, L, 30); Pm = b.d(x, 0); s = np.linalg.svd(Pm.T @ (w[:, None] * Pm), compute_uv=False); ks.append(float(s[0] / s[-1]))
    kh[kind] = ks[0] * ks[1]
print(kh); save_json(kh, "E_helmholtz_kappa")
