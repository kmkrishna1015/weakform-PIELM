"""Benchmark problems (bars, beams, 2-D Helmholtz cavity, in-plane plate) for the two formulations compared in the paper.

  strong : published Eig-PIELM -- pointwise residual at UNIFORM interior points with unit weights,
           S = Phi^T Psi, G = Phi^T Phi, all boundary/natural rows projected out by an exact null space.
  weak   : Rayleigh-Ritz/Galerkin -- K = int (D^m phi)(D^m phi), M = int phi phi with Gauss-Legendre quadrature,
           only ESSENTIAL rows projected out.
Both reduced pencils are solved by the same routine `solve_pencil` (dense general eigensolver by default).

Two-D operators use Kronecker structure; index convention: column = i*(ny+1)+j (x slow, y fast).
"""
import math, time
import numpy as np
import scipy.linalg as sla
from scipy.optimize import brentq
from .numerics import (gl, uniform_interior, null_space_svd, solve_pencil, lowest, rel_err_pct, shape_err, Res, finish)
from .bases import make_basis

E_STEEL, RHO_STEEL = 210e9, 7800.0
METHODS = ("strong", "weak")              # like-for-like: both use the general eigensolver
METHODS3 = ("strong", "weak", "weak_sym")  # + weak form with the symmetric (eigh) solver


# ------------------------------------------------------------------------------------------------ 1-D problems
def beam_roots(kind, n=6):
    f = {"ff": lambda z: np.cos(z) * np.cosh(z) - 1, "cf": lambda z: np.cos(z) * np.cosh(z) + 1}[kind]
    g = {"ff": [4.73004074486, 7.85320462409, 10.9956078380, 14.1371654912, 17.2787596574],
         "cf": [1.87510406871, 4.69409113297, 7.85475743823, 10.9955407349, 14.1371683910]}[kind]
    out = []
    for k in range(n):
        gk = g[k] if k < len(g) else (np.pi * (k + 0.5) if kind == "cf" else np.pi * (k + 1.5))
        out.append(brentq(f, gk - 0.3, gk + 0.3, xtol=1e-15, rtol=1e-15))
    return np.array(out)


class Problem1D:
    dim = 1
    def __init__(self, key, title, m, a_s, kc, mc, L, ess, nat, exact_fn, shape_fn):
        self.key, self.title, self.m, self.a_s, self.kc, self.mc, self.L = key, title, m, a_s, kc, mc, L
        self.ess, self.nat, self.exact_fn, self.shape_fn = ess, nat, exact_fn, shape_fn
        self.nmodes = 5
        self.default = dict(n=23, ng=40, N=50000)
    def exact(self, k=10): return self.exact_fn(k)
    def _floor(self): return 1e-6 * self.exact(1)[0] ** 2
    def _bcrows(self, basis, lst): return [basis.d([xb], k) for xb, k in lst]

    def strong(self, basis, N=50000, vectors=False, rcond=None):
        t0 = time.perf_counter(); x = uniform_interior(self.L, N)
        Phi = basis.d(x, 0); Psi = self.a_s * basis.d(x, 2 * self.m)
        G = Phi.T @ Phi; S = Phi.T @ Psi
        t1 = time.perf_counter()
        Z, rk = null_space_svd(np.vstack(self._bcrows(basis, self.ess + self.nat)))
        Sr, Gr = Z.T @ S @ Z, Z.T @ G @ Z
        t2 = time.perf_counter()
        lam, V, kept = solve_pencil(Sr, Gr, rcond)
        om, beta = finish(lam, V, Z, self.nmodes, self._floor(), vectors)
        t3 = time.perf_counter()
        return Res(om, beta, dict(red_dim=Z.shape[1], kept=kept, asym=float(np.linalg.norm(Sr - Sr.T) / np.linalg.norm(Sr)),
                                  stages=(t1 - t0, t2 - t1, t3 - t2)))

    def weak(self, basis, ng=40, vectors=False, rcond=None, symmetric=False):
        t0 = time.perf_counter(); x, w = gl(0, self.L, ng)
        Phi = basis.d(x, 0); Dm = basis.d(x, self.m)
        K = self.kc * (Dm.T @ (w[:, None] * Dm)); M = self.mc * (Phi.T @ (w[:, None] * Phi))
        t1 = time.perf_counter()
        Z, rk = null_space_svd(np.vstack(self._bcrows(basis, self.ess)))
        Kr, Mr = Z.T @ K @ Z, Z.T @ M @ Z
        t2 = time.perf_counter()
        lam, V, kept = solve_pencil(Kr, Mr, rcond, symmetric)
        om, beta = finish(lam, V, Z, self.nmodes, self._floor(), vectors)
        t3 = time.perf_counter()
        return Res(om, beta, dict(red_dim=Z.shape[1], kept=kept, stages=(t1 - t0, t2 - t1, t3 - t2), lam=lam))

    def ritz_matrices(self, basis, ng=40):
        """Weak-form matrices and the two admissible subspaces (used by the min-max ordering check)."""
        x, w = gl(0, self.L, ng); Phi = basis.d(x, 0); Dm = basis.d(x, self.m)
        K = self.kc * (Dm.T @ (w[:, None] * Dm)); M = self.mc * (Phi.T @ (w[:, None] * Phi))
        Zw, _ = null_space_svd(np.vstack(self._bcrows(basis, self.ess)))
        Zs, _ = null_space_svd(np.vstack(self._bcrows(basis, self.ess + self.nat)))
        return K, M, Zw, Zs

    def eval_shape(self, basis, beta, x): return basis.d(x, 0) @ beta


def make_1d_problems():
    E, rho = E_STEEL, RHO_STEEL
    Lb, b, h = 0.12, 0.003, 0.002
    A, I = b * h, b * h ** 3 / 12; EI, rA = E * I, rho * A
    k = np.arange(1, 11)
    P = {}
    P["bar_ff"] = Problem1D("bar_ff", "Bar, fixed-fixed", 1, -E / rho, E, rho, 1.0, [(0, 0), (1.0, 0)], [],
                            lambda n: k[:n] * np.pi * np.sqrt(E / rho), lambda j, x: np.sin((j + 1) * np.pi * x))
    P["bar_fx"] = Problem1D("bar_fx", "Bar, fixed-free", 1, -E / rho, E, rho, 1.0, [(0, 0)], [(1.0, 1)],
                            lambda n: (2 * k[:n] - 1) * np.pi / 2 * np.sqrt(E / rho), lambda j, x: np.sin((2 * j + 1) * np.pi * x / 2))
    P["beam_ss"] = Problem1D("beam_ss", "Beam, simply supported", 2, EI / rA, EI, rA, Lb, [(0, 0), (Lb, 0)], [(0, 2), (Lb, 2)],
                             lambda n: (k[:n] * np.pi / Lb) ** 2 * np.sqrt(EI / rA), lambda j, x: np.sin((j + 1) * np.pi * x / Lb))
    bff, bcf = beam_roots("ff"), beam_roots("cf")
    def sh_ff(j, x):
        b_ = bff[j]; s = (np.cos(b_) - np.cosh(b_)) / (np.sin(b_) - np.sinh(b_))
        return np.cos(b_ * x / Lb) - np.cosh(b_ * x / Lb) - s * (np.sin(b_ * x / Lb) - np.sinh(b_ * x / Lb))
    def sh_cf(j, x):
        b_ = bcf[j]; s = (np.cos(b_) + np.cosh(b_)) / (np.sin(b_) + np.sinh(b_))
        return np.cos(b_ * x / Lb) - np.cosh(b_ * x / Lb) - s * (np.sin(b_ * x / Lb) - np.sinh(b_ * x / Lb))
    P["beam_ff"] = Problem1D("beam_ff", "Beam, fixed-fixed", 2, EI / rA, EI, rA, Lb, [(0, 0), (0, 1), (Lb, 0), (Lb, 1)], [],
                             lambda n: (bff[:n] / Lb) ** 2 * np.sqrt(EI / rA), sh_ff)
    P["beam_cf"] = Problem1D("beam_cf", "Beam, cantilever", 2, EI / rA, EI, rA, Lb, [(0, 0), (0, 1)], [(Lb, 2), (Lb, 3)],
                             lambda n: (bcf[:n] / Lb) ** 2 * np.sqrt(EI / rA), sh_cf)
    return P


# ------------------------------------------------------------------------------------------------ 2-D Helmholtz (Neumann cavity)
class Helmholtz2D:
    key, title, dim = "helmholtz", "Helmholtz cavity (Neumann)", 2
    def __init__(self, Lx=5.0, Ly=3.0, c=340.0):
        self.Lx, self.Ly, self.c, self.nmodes = Lx, Ly, c, 5
        self.default = dict(n=(13, 13), ng=(20, 20), N=(5000, 3000), nb=None)
        self.k2 = sorted([((m * np.pi / Lx) ** 2 + (q * np.pi / Ly) ** 2, m, q) for m in range(40) for q in range(40) if (m, q) != (0, 0)])
    def exact(self, k=10): return self.c * np.sqrt([t[0] for t in self.k2[:k]])
    def _floor(self): return 1e-6 * self.k2[0][0]

    def strong(self, bx, by, N=(5000, 3000), nb=None, vectors=False, rcond=None):
        t0 = time.perf_counter()
        x, y = uniform_interior(self.Lx, N[0]), uniform_interior(self.Ly, N[1])
        Px, Py = bx.d(x, 0), by.d(y, 0)
        Gx, Gy = Px.T @ Px, Py.T @ Py; Ax, Ay = Px.T @ bx.d(x, 2), Py.T @ by.d(y, 2)
        G = np.kron(Gx, Gy); S = -(np.kron(Ax, Gy) + np.kron(Gx, Ay))
        t1 = time.perf_counter()
        nbx, nby = (bx.nphi, by.nphi) if nb is None else nb
        xb, yb = np.linspace(0, self.Lx, nbx), np.linspace(0, self.Ly, nby)
        Bx_b, By_b = bx.d(xb, 0), by.d(yb, 0)
        rows = [np.kron(bx.d([0.0], 1), By_b), np.kron(bx.d([self.Lx], 1), By_b),
                np.kron(Bx_b, by.d([0.0], 1)), np.kron(Bx_b, by.d([self.Ly], 1))]
        Z, rk = null_space_svd(np.vstack(rows))
        Sr, Gr = Z.T @ S @ Z, Z.T @ G @ Z
        t2 = time.perf_counter()
        lam, V, kept = solve_pencil(Sr, Gr, rcond)
        idx, om = lowest(lam, self.nmodes, self._floor())
        beta = (Z @ V[:, idx]).real if vectors and len(idx) else None
        t3 = time.perf_counter()
        return Res(self.c * om if len(idx) else om, beta, dict(red_dim=Z.shape[1], rank=rk, kept=kept, stages=(t1 - t0, t2 - t1, t3 - t2)))

    def weak(self, bx, by, ng=(20, 20), vectors=False, rcond=None, symmetric=False):
        t0 = time.perf_counter()
        (x, wx), (y, wy) = gl(0, self.Lx, ng[0]), gl(0, self.Ly, ng[1])
        Px, Py, Dx, Dy = bx.d(x, 0), by.d(y, 0), bx.d(x, 1), by.d(y, 1)
        Mx, My = Px.T @ (wx[:, None] * Px), Py.T @ (wy[:, None] * Py)
        Kx, Ky = Dx.T @ (wx[:, None] * Dx), Dy.T @ (wy[:, None] * Dy)
        K = np.kron(Kx, My) + np.kron(Mx, Ky); M = np.kron(Mx, My)
        t1 = time.perf_counter()
        lam, V, kept = solve_pencil(K, M, rcond, symmetric)         # Neumann cavity: no essential rows, Z = I
        idx, om = lowest(lam, self.nmodes, self._floor())
        beta = V[:, idx].real if vectors and len(idx) else None
        t2 = time.perf_counter()
        return Res(self.c * om, beta, dict(red_dim=K.shape[0], kept=kept, stages=(t1 - t0, 0.0, t2 - t1)))

    def eval_shape(self, bx, by, beta, X, Y):
        return np.einsum("pi,qj,ij->pq", bx.d(X, 0), by.d(Y, 0), beta.reshape(bx.nphi, by.nphi))
    def exact_shape(self, j, X, Y):
        _, m, q = self.k2[j]
        return np.cos(m * np.pi * X / self.Lx)[:, None] * np.cos(q * np.pi * Y / self.Ly)[None, :]


# ------------------------------------------------------------------------------------------------ 2-D plane-stress plate (diaphragm supports)
class Plate2D:
    key, title, dim = "plate", "Plate in-plane (diaphragm)", 2
    def __init__(self, Lx=1.0, Ly=0.6, E=210e9, nu=0.3, rho=7800.0):
        self.Lx, self.Ly, self.E, self.nu, self.rho, self.nmodes = Lx, Ly, E, nu, rho, 6
        self.c11 = E / (1 - nu ** 2); self.c12 = nu * E / (1 - nu ** 2); self.c66 = E / (2 * (1 + nu))
        self.default = dict(n=(14, 10), ng=(26, 20), N=(2000, 1200), nb=(18, 14))
        self._modes = self._exact_modes(12)
    def _exact_modes(self, nkeep, pmax=10, qmax=10):
        c11, c12, c66, rho = self.c11, self.c12, self.c66, self.rho
        out = []
        for p in range(1, pmax + 1):
            for q in range(1, qmax + 1):
                kx, ky = p * np.pi / self.Lx, q * np.pi / self.Ly
                M = np.array([[c11 * kx ** 2 + c66 * ky ** 2, (c12 + c66) * kx * ky], [(c12 + c66) * kx * ky, c66 * kx ** 2 + c11 * ky ** 2]])
                w2, V = np.linalg.eigh(M)
                for b in range(2): out.append((w2[b] / rho, ("pq", p, q, V[:, b])))
        for q in range(1, qmax + 1): out.append((c66 * (q * np.pi / self.Ly) ** 2 / rho, ("p0", 0, q, None)))
        for p in range(1, pmax + 1): out.append((c66 * (p * np.pi / self.Lx) ** 2 / rho, ("q0", p, 0, None)))
        out.sort(key=lambda t: t[0]); return out[:nkeep]
    def exact(self, k=10): return np.sqrt([m[0] for m in self._modes[:k]])
    def _floor(self): return 1e-6 * self._modes[0][0]
    def exact_shape(self, j, X, Y):
        kind, p, q, v = self._modes[j][1]
        Xg, Yg = np.meshgrid(X, Y, indexing="ij"); kx, ky = p * np.pi / self.Lx, q * np.pi / self.Ly
        if kind == "pq": return v[0] * np.cos(kx * Xg) * np.sin(ky * Yg), v[1] * np.sin(kx * Xg) * np.cos(ky * Yg)
        if kind == "p0": return np.sin(ky * Yg), np.zeros_like(Xg)
        return np.zeros_like(Xg), np.sin(kx * Xg)
    def _blocks(self, bx, by, x, wx, y, wy):
        Px, Py = bx.d(x, 0), by.d(y, 0)
        f = lambda P, Q, w: P.T @ (w[:, None] * Q)
        return dict(Px=Px, Py=Py, Mx=f(Px, Px, wx), My=f(Py, Py, wy),
                    Cx=f(bx.d(x, 1), Px, wx), Cy=f(by.d(y, 1), Py, wy),
                    Kx=f(bx.d(x, 1), bx.d(x, 1), wx), Ky=f(by.d(y, 1), by.d(y, 1), wy),
                    A2x=f(Px, bx.d(x, 2), wx), A2y=f(Py, by.d(y, 2), wy),
                    A1x=f(Px, bx.d(x, 1), wx), A1y=f(Py, by.d(y, 1), wy))
    def _bc_rows(self, bx, by, nb):
        nbx, nby = nb; xb, yb = np.linspace(0, self.Lx, nbx), np.linspace(0, self.Ly, nby)
        Bxb, Byb, dBxb, dByb = bx.d(xb, 0), by.d(yb, 0), bx.d(xb, 1), by.d(yb, 1)
        n1 = bx.nphi * by.nphi; Z0 = lambda m: np.zeros((m, n1)); r = self.c12 / self.c11
        ess, nat = [], []
        for xe in (0.0, self.Lx):
            b0, d0 = bx.d([xe], 0), bx.d([xe], 1)
            ess.append(np.hstack([Z0(nby), np.kron(b0, Byb)]))                                  # u_y = 0
            nat.append(np.hstack([np.kron(d0, Byb), r * np.kron(b0, dByb)]))                    # sigma_xx = 0
        for ye in (0.0, self.Ly):
            b0, d0 = by.d([ye], 0), by.d([ye], 1)
            ess.append(np.hstack([np.kron(Bxb, b0), Z0(nbx)]))                                  # u_x = 0
            nat.append(np.hstack([r * np.kron(dBxb, b0), np.kron(Bxb, d0)]))                    # sigma_yy = 0
        return np.vstack(ess), np.vstack(nat)

    def strong(self, bx, by, N=(2000, 1200), nb=(18, 14), vectors=False, rcond=None):
        t0 = time.perf_counter(); c11, c12, c66, rho = self.c11, self.c12, self.c66, self.rho
        x, y = uniform_interior(self.Lx, N[0]), uniform_interior(self.Ly, N[1])
        B = self._blocks(bx, by, x, np.ones(len(x)), y, np.ones(len(y)))
        G0 = np.kron(B["Mx"], B["My"]); Z0 = np.zeros_like(G0)
        S11 = -(c11 * np.kron(B["A2x"], B["My"]) + c66 * np.kron(B["Mx"], B["A2y"])) / rho
        S22 = -(c66 * np.kron(B["A2x"], B["My"]) + c11 * np.kron(B["Mx"], B["A2y"])) / rho
        S12 = -(c12 + c66) * np.kron(B["A1x"], B["A1y"]) / rho
        G = np.block([[G0, Z0], [Z0, G0]]); S = np.block([[S11, S12], [S12, S22]])
        t1 = time.perf_counter()
        ess, nat = self._bc_rows(bx, by, nb)
        Z, rk = null_space_svd(np.vstack([ess, nat]))
        Sr, Gr = Z.T @ S @ Z, Z.T @ G @ Z
        t2 = time.perf_counter()
        lam, V, kept = solve_pencil(Sr, Gr, rcond)
        idx, om = lowest(lam, self.nmodes, self._floor())
        beta = (Z @ V[:, idx]).real if vectors and len(idx) else None
        t3 = time.perf_counter()
        return Res(om, beta, dict(red_dim=Z.shape[1], rank=rk, kept=kept, stages=(t1 - t0, t2 - t1, t3 - t2)))

    def weak(self, bx, by, ng=(26, 20), nb=(18, 14), vectors=False, rcond=None, symmetric=False):
        t0 = time.perf_counter(); c11, c12, c66, rho = self.c11, self.c12, self.c66, self.rho
        (x, wx), (y, wy) = gl(0, self.Lx, ng[0]), gl(0, self.Ly, ng[1])
        B = self._blocks(bx, by, x, wx, y, wy)
        Kxx = c11 * np.kron(B["Kx"], B["My"]) + c66 * np.kron(B["Mx"], B["Ky"])
        Kyy = c66 * np.kron(B["Kx"], B["My"]) + c11 * np.kron(B["Mx"], B["Ky"])
        Kxy = c12 * np.kron(B["Cx"].T, B["Cy"]) + c66 * np.kron(B["Cx"], B["Cy"].T)
        M0 = rho * np.kron(B["Mx"], B["My"]); Zm = np.zeros_like(M0)
        K = np.block([[Kxx, Kxy], [Kxy.T, Kyy]]); M = np.block([[M0, Zm], [Zm, M0]])
        t1 = time.perf_counter()
        ess, nat = self._bc_rows(bx, by, nb)
        Z, rk = null_space_svd(ess)
        Kr, Mr = Z.T @ K @ Z, Z.T @ M @ Z
        t2 = time.perf_counter()
        lam, V, kept = solve_pencil(Kr, Mr, rcond, symmetric)
        idx, om = lowest(lam, self.nmodes, self._floor())
        beta = (Z @ V[:, idx]).real if vectors and len(idx) else None
        t3 = time.perf_counter()
        return Res(om, beta, dict(red_dim=Z.shape[1], rank=rk, kept=kept, stages=(t1 - t0, t2 - t1, t3 - t2)))

    def eval_field(self, bx, by, beta, X, Y):
        n1 = bx.nphi * by.nphi
        f = lambda b: np.einsum("pi,qj,ij->pq", bx.d(X, 0), by.d(Y, 0), b.reshape(bx.nphi, by.nphi))
        return f(beta[:n1]), f(beta[n1:])


# ------------------------------------------------------------------------------------------------ FEM baseline (Q4, Helmholtz cavity)
def fem_helmholtz(Lx, Ly, c, nnx, nny, nmodes=5):
    import scipy.sparse as sp, scipy.sparse.linalg as spl
    def one(L, nn):
        h = L / (nn - 1); e = np.ones(nn)
        K = sp.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1], format="lil") / h; K[0, 0] = K[-1, -1] = 1 / h
        M = sp.diags([e[:-1], 4 * e, e[:-1]], [-1, 0, 1], format="lil") * h / 6; M[0, 0] = M[-1, -1] = h / 3
        return K.tocsc(), M.tocsc()
    (Kx, Mx), (Ky, My) = one(Lx, nnx), one(Ly, nny)
    K = (sp.kron(Kx, My) + sp.kron(Mx, Ky)).tocsc(); M = sp.kron(Mx, My).tocsc()
    if K.shape[0] <= 1500: lam = sla.eigh(K.toarray(), M.toarray(), eigvals_only=True)
    else: lam = spl.eigsh(K, k=nmodes + 2, M=M, sigma=-1.0, which="LM", return_eigenvectors=False)
    lam = np.sort(lam); lam = lam[lam > 1e-6][:nmodes]
    return c * np.sqrt(lam)


# ------------------------------------------------------------------------------------------------ dispatch helpers
ORDER = ["bar_ff", "bar_fx", "beam_ss", "beam_ff", "beam_cf", "helmholtz", "plate"]
_CACHE = {}
def problems():
    if not _CACHE:
        P = make_1d_problems(); P["helmholtz"] = Helmholtz2D(); P["plate"] = Plate2D(); _CACHE.update(P)
    return dict(_CACHE)

def bases(prob, kind="bernstein", n=None):
    d = prob.default; n = d["n"] if n is None else n
    if prob.dim == 1: return make_basis(kind, prob.L, n, 4)
    return make_basis(kind, prob.Lx, n[0], 2), make_basis(kind, prob.Ly, n[1], 2)

def run_method(prob, method, kind="bernstein", n=None, vectors=False, B=None, rcond=None, **ov):
    """method in {'strong', 'weak', 'weak_sym'}.  'weak_sym' = weak form solved with the symmetric eigensolver (eigh).
    Overrides: n (degree), ng (Gauss points, weak), N (uniform points, strong), nb (boundary sampling)."""
    assert method in METHODS3, method
    sym = method == "weak_sym"; method = "weak" if sym else method
    if sym and rcond is None: rcond = 1e-12      # symmetric solve = Loewdin-reduced eigh with the 1e-12 truncation used throughout
                                                 # (plain Cholesky eigh(K, M) loses ~5 digits on the beams: Gram matrix cond ~ 1e16)
    d = dict(prob.default); d.update(ov)
    nn = d["n"] if n is None else n
    if B is None: B = bases(prob, kind, nn)
    if prob.dim == 1:
        if method == "strong": return prob.strong(B, d["N"], vectors, rcond)
        return prob.weak(B, d["ng"], vectors, rcond, sym)
    if prob.key == "helmholtz":
        if method == "strong": return prob.strong(*B, N=d["N"], nb=d.get("nb"), vectors=vectors, rcond=rcond)
        return prob.weak(*B, ng=d["ng"], vectors=vectors, rcond=rcond, symmetric=sym)
    if method == "strong": return prob.strong(*B, N=d["N"], nb=d["nb"], vectors=vectors, rcond=rcond)
    return prob.weak(*B, ng=d["ng"], nb=d["nb"], vectors=vectors, rcond=rcond, symmetric=sym)

def max_err_pct(r, p):
    e = rel_err_pct(r.omega, p.exact(p.nmodes))
    return float(np.max(e)) if len(r.omega) == p.nmodes else float("nan")
