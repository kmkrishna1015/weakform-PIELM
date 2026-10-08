"""Kirchhoff plate free vibration on the unit square, D = 1, rho*h = 1  ->  Omega = omega a^2 sqrt(rho h / D) = sqrt(lambda).
Edges: S (simply supported), C (clamped), F (free); edge order (x=0, x=1, y=0, y=1).

  weak   : a(w,v) = int[w_xx v_xx + w_yy v_yy + nu (w_xx v_yy + w_yy v_xx) + 2(1-nu) w_xy v_xy], Gauss-Legendre quadrature,
           only ESSENTIAL conditions (w = 0, dw/dn = 0) projected out.
  strong : published Eig-PIELM  D nabla^4 w = lambda w  at uniform interior points (unit weights); boundary rows are
           S: w = 0, M_n = 0 | C: w = 0, w_n = 0 | F: M_n = 0, V_n = 0 (effective shear) plus w_xy = 0 at free-free corners.
Both reduced pencils are solved by `solve_pencil` (dense general eigensolver unless a truncation rcond is passed to both).
"""
import time
import numpy as np
from .numerics import gl, uniform_interior, null_space_svd, solve_pencil, lowest, Res
from .bases import Bern1D

NU = 0.3


class Plate:
    nmodes = 6
    def __init__(self, edges, nu=NU): self.e, self.nu = edges, nu
    def _ax(self, b, x, w, a, c): return b.d(x, a).T @ (w[:, None] * b.d(x, c))

    def weak(self, n=16, ng=30, rcond=None, vectors=False, bx=None, by=None, admissible=False, symmetric=False):
        t0 = time.perf_counter(); bx = bx or Bern1D(1.0, n); by = by or bx; x, w = gl(0, 1.0, ng); nu = self.nu
        X = {(a, c): self._ax(bx, x, w, a, c) for a in range(3) for c in range(3)}
        Y = {(a, c): self._ax(by, x, w, a, c) for a in range(3) for c in range(3)}
        K = (np.kron(X[2, 2], Y[0, 0]) + np.kron(X[0, 0], Y[2, 2]) + nu * (np.kron(X[2, 0], Y[0, 2]) + np.kron(X[0, 2], Y[2, 0]))
             + 2 * (1 - nu) * np.kron(X[1, 1], Y[1, 1]))
        M = np.kron(X[0, 0], Y[0, 0]); t1 = time.perf_counter()
        Z, rk = (np.eye(K.shape[0]), 0) if admissible else null_space_svd(self._rows(bx, max(bx.nphi, by.nphi), strong=False, by=by))
        t2 = time.perf_counter()
        lam, V, kept = solve_pencil(Z.T @ K @ Z, Z.T @ M @ Z, rcond, symmetric); idx, om = lowest(lam, self.nmodes, 1.0)
        return Res(om, (Z @ V[:, idx]).real if vectors else None,
                   dict(time=time.perf_counter() - t0, red=Z.shape[1], rank=rk, kept=kept, stages=(t1 - t0, t2 - t1, time.perf_counter() - t2)))

    def strong(self, n=16, npts=300, nb=None, rcond=None, vectors=False, corners=True):
        t0 = time.perf_counter(); b = Bern1D(1.0, n)
        x = uniform_interior(1.0, npts); w = np.ones(npts)
        A = {c: self._ax(b, x, w, 0, c) for c in (0, 2, 4)}
        S = np.kron(A[4], A[0]) + 2 * np.kron(A[2], A[2]) + np.kron(A[0], A[4]); G = np.kron(A[0], A[0]); t1 = time.perf_counter()
        Z, rk = null_space_svd(self._rows(b, nb or n + 1, strong=True, corners=corners)); t2 = time.perf_counter()
        lam, V, kept = solve_pencil(Z.T @ S @ Z, Z.T @ G @ Z, rcond); idx, om = lowest(lam, self.nmodes, 1.0)
        return Res(om, (Z @ V[:, idx]).real if vectors else None,
                   dict(time=time.perf_counter() - t0, red=Z.shape[1], rank=rk, kept=kept, stages=(t1 - t0, t2 - t1, time.perf_counter() - t2)))

    def ritz_matrices(self, n=12, ng=30):
        """Weak-form K, M (Gauss-Legendre) and the two nested admissible subspaces for the min-max ordering check:
        Zw = essential rows only (weak form), Zs = essential + natural + corner rows (strong form)."""
        b = Bern1D(1.0, n); x, w = gl(0, 1.0, ng); nu = self.nu
        X = {(a, c): self._ax(b, x, w, a, c) for a in range(3) for c in range(3)}
        K = (np.kron(X[2, 2], X[0, 0]) + np.kron(X[0, 0], X[2, 2]) + nu * (np.kron(X[2, 0], X[0, 2]) + np.kron(X[0, 2], X[2, 0]))
             + 2 * (1 - nu) * np.kron(X[1, 1], X[1, 1]))
        M = np.kron(X[0, 0], X[0, 0])
        Zw, _ = null_space_svd(self._rows(b, n + 1, strong=False)); Zs, _ = null_space_svd(self._rows(b, n + 1, strong=True))
        return K, M, Zw, Zs

    def _rows(self, b, nb, strong, by=None, corners=True):
        """Boundary rows on the tensor basis (column i*(n+1)+j : x slow, y fast)."""
        by = by or b; nu = self.nu; t = np.linspace(0, 1, nb); rows = []
        bxf = lambda x, k: b.d(x, k); byf = lambda x, k: by.d(x, k)
        def edge(axis, pos, code):
            def op(terms):
                r = 0
                for kx, ky, cf in terms:
                    r = r + cf * (np.kron(bxf([pos], kx), byf(t, ky)) if axis == "x" else np.kron(bxf(t, kx), byf([pos], ky)))
                return r
            sw = (lambda p, q: (p, q)) if axis == "x" else (lambda p, q: (q, p))
            T = lambda pn, pt, c: (*sw(pn, pt), c)
            w0 = op([T(0, 0, 1)]); wn = op([T(1, 0, 1)])
            Mn = op([T(2, 0, 1), T(0, 2, nu)]); Vn = op([T(3, 0, 1), T(1, 2, 2 - nu)])
            if code == "S": return [w0] + ([Mn] if strong else [])
            if code == "C": return [w0, wn]
            if code == "F": return ([Mn, Vn] if strong else [])
        for axis, pos, code in (("x", 0.0, self.e[0]), ("x", 1.0, self.e[1]), ("y", 0.0, self.e[2]), ("y", 1.0, self.e[3])):
            rows += edge(axis, pos, code)
        if strong and corners:      # free-free corners: w_xy = 0
            for cx, cy, (c1, c2) in ((0., 0., (self.e[0], self.e[2])), (0., 1., (self.e[0], self.e[3])),
                                     (1., 0., (self.e[1], self.e[2])), (1., 1., (self.e[1], self.e[3]))):
                if c1 == "F" and c2 == "F": rows.append(np.kron(bxf([cx], 1), byf([cy], 1)))
        return np.vstack(rows) if rows else np.zeros((0, b.nphi ** 2))


def ssss_exact(n=6):
    """Exact Omega = pi^2 (m^2 + q^2) of the simply supported unit square plate (Omega = sqrt(lambda))."""
    return np.sort([np.pi ** 2 * (m * m + q * q) for m in range(1, 8) for q in range(1, 8)])[:n]
