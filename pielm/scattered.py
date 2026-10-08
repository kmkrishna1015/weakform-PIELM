"""Grid-free integration for the weak form: scattered points + moment-fitted weights.
Weights w solve  sum_p w_p P_k(x_p) = int_Omega P_k  for all Legendre products of degree <= q (minimum-norm least squares),
so every integral of a product of polynomials of degree <= q is exact, with no grid, no cells, no connectivity."""
import time, numpy as np
from numpy.polynomial import legendre as Lg
from scipy.stats import qmc
from .numerics import null_space_svd, solve_pencil, lowest, Res, finish

def rule_1d(a, b, q, N=None):
    N = N or 4 * (q + 1)
    x = a + (b - a) * qmc.Halton(1, scramble=False).random(N + 1)[1:, 0]
    A = Lg.legvander(2 * (x - a) / (b - a) - 1, q).T; m = np.zeros(q + 1); m[0] = b - a
    return x, np.linalg.lstsq(A, m, rcond=None)[0]

def make_gl_sc(q, factor=8):             # drop-in for numerics.gl(a, b, ng): scattered points, exact for degree <= q = 2n (n = basis degree)
    return lambda a, b, ng=None: rule_1d(a, b, q, N=factor * (q + 1))

def rule_2d(Lx, Ly, qx, qy, N=None, factor=6):
    N = N or int(factor * (qx + 1) * (qy + 1))
    P = qmc.Halton(2, scramble=False).random(N + 1)[1:] * np.array([Lx, Ly])
    Ps, Pt = Lg.legvander(2 * P[:, 0] / Lx - 1, qx), Lg.legvander(2 * P[:, 1] / Ly - 1, qy)
    A = (Ps[:, :, None] * Pt[:, None, :]).reshape(N, -1).T; m = np.zeros((qx + 1, qy + 1)); m[0, 0] = Lx * Ly
    return P, np.linalg.lstsq(A, m.ravel(), rcond=None)[0]

def _tens(bx, by, X, a, c):
    return np.einsum('pi,pj->pij', bx.d(X[:, 0], a), by.d(X[:, 1], c)).reshape(len(X), -1)

def helmholtz_weak(p, bx, by, rcond=1e-12, symmetric=True, factor=6):
    t0 = time.perf_counter(); X, w = rule_2d(p.Lx, p.Ly, 2 * (bx.nphi - 1), 2 * (by.nphi - 1), factor=factor); t1 = time.perf_counter()
    W = w[:, None]; F, Fx, Fy = _tens(bx, by, X, 0, 0), _tens(bx, by, X, 1, 0), _tens(bx, by, X, 0, 1)
    K = Fx.T @ (W * Fx) + Fy.T @ (W * Fy); M = F.T @ (W * F); t2 = time.perf_counter()
    lam, V, kept = solve_pencil(K, M, rcond, symmetric); idx, om = lowest(lam, p.nmodes, p._floor()); t3 = time.perf_counter()
    return Res(p.c * om, None, dict(red_dim=K.shape[0], kept=kept, stages=(t1 - t0, t2 - t1, t3 - t2), npts=len(X)))

def plate2d_weak(p, bx, by, nb=(18, 14), rcond=1e-12, symmetric=True, factor=6):
    t0 = time.perf_counter(); X, w = rule_2d(p.Lx, p.Ly, 2 * (bx.nphi - 1), 2 * (by.nphi - 1), factor=factor); t1 = time.perf_counter()
    W = w[:, None]; F, Fx, Fy = _tens(bx, by, X, 0, 0), _tens(bx, by, X, 1, 0), _tens(bx, by, X, 0, 1)
    c11, c12, c66, rho = p.c11, p.c12, p.c66, p.rho
    xx, yy, yx, xy, ff = Fx.T @ (W * Fx), Fy.T @ (W * Fy), Fy.T @ (W * Fx), Fx.T @ (W * Fy), F.T @ (W * F)
    Kxx = c11 * xx + c66 * yy; Kyy = c66 * xx + c11 * yy; Kxy = c12 * yx + c66 * xy
    K = np.block([[Kxx, Kxy], [Kxy.T, Kyy]]); Z0 = np.zeros_like(ff); M = np.block([[rho * ff, Z0], [Z0, rho * ff]]); t2 = time.perf_counter()
    ess, nat = p._bc_rows(bx, by, nb); Z, rk = null_space_svd(ess)
    lam, V, kept = solve_pencil(Z.T @ K @ Z, Z.T @ M @ Z, rcond, symmetric); idx, om = lowest(lam, p.nmodes, p._floor()); t3 = time.perf_counter()
    return Res(om, None, dict(red_dim=Z.shape[1], kept=kept, stages=(t1 - t0, t2 - t1, t3 - t2), npts=len(X)))

def kirchhoff_weak(P, n=16, rcond=1e-12, symmetric=True, factor=6):
    from .bases import Bern1D
    t0 = time.perf_counter(); b = Bern1D(1.0, n); X, w = rule_2d(1.0, 1.0, 2 * n, 2 * n, factor=factor); t1 = time.perf_counter()
    W = w[:, None]; nu = P.nu; F = _tens(b, b, X, 0, 0); Fxx, Fyy, Fxy = _tens(b, b, X, 2, 0), _tens(b, b, X, 0, 2), _tens(b, b, X, 1, 1)
    K = Fxx.T @ (W * Fxx) + Fyy.T @ (W * Fyy) + nu * (Fxx.T @ (W * Fyy) + Fyy.T @ (W * Fxx)) + 2 * (1 - nu) * Fxy.T @ (W * Fxy)
    M = F.T @ (W * F); t2 = time.perf_counter()
    Z, rk = null_space_svd(P._rows(b, n + 1, strong=False))
    lam, V, kept = solve_pencil(Z.T @ K @ Z, Z.T @ M @ Z, rcond, symmetric); idx, om = lowest(lam, P.nmodes, 1.0); t3 = time.perf_counter()
    return Res(om, None, dict(red=Z.shape[1], kept=kept, stages=(t1 - t0, t2 - t1, t3 - t2), npts=len(X)))
