"""
Hard problem 1: Euler-Bernoulli beam with concentrated attachments (point mass m, spring k at x0, tip mass).
  weak form   : K += k phi0 phi0^T ,  M += m phi0 phi0^T                (rank-one, one line, one solve)
  strong form : EI [w'''](x0) = (m*omega^2 - k) w(x0)     -> interface constraint DEPENDS ON THE EIGENVALUE
                 => cannot live in a fixed null-space projection; needs per-mode fixed-point iteration.
Exact reference: transcendental determinant, 70-digit mpmath.
Strong form = published Eig-PIELM construction on each element (uniform interior points, unit weights); weak and
strong reduced pencils are solved by the same routine `solve_pencil`.
"""
import time, numpy as np, mpmath as mp
from scipy.optimize import brentq
from .numerics import gl, uniform_interior, null_space_svd, solve_pencil, lowest, Res
from .bases import Bern1D
E_STEEL, RHO_STEEL = 210e9, 7800.0

E, RHO = E_STEEL, RHO_STEEL
Lb, bb, hh = 0.12, 0.003, 0.002
A_, I_ = bb * hh, bb * hh ** 3 / 12
EI, rA = E * I_, RHO * A_

# ------------------------------------------------------------------ exact (mpmath)
def exact_freqs(bcL, bcR, L, x0=None, m=0.0, k=0.0, m_tip=0.0, nmodes=5, bmax=40.0):
    mp.mp.dps = 70; bmax = max(bmax, 3.6 * nmodes + 12)
    L, EIm, rAm = mp.mpf(L), mp.mpf(EI), mp.mpf(rA)
    mu, mut, kap = mp.mpf(m) / rAm, mp.mpf(m_tip) / rAm, mp.mpf(k) / EIm
    segs = [(0.0, L)] if x0 is None else [(0.0, mp.mpf(x0)), (mp.mpf(x0), L)]
    def rows_end(bc, xi, beta, tip=0.0, left=True):
        c, s, ch, sh = mp.cos(beta * xi), mp.sin(beta * xi), mp.cosh(beta * xi), mp.sinh(beta * xi)
        w = [c, s, ch, sh]; w1 = [-beta * s, beta * c, beta * sh, beta * ch]
        w2 = [-beta ** 2 * c, -beta ** 2 * s, beta ** 2 * ch, beta ** 2 * sh]; w3 = [beta ** 3 * s, -beta ** 3 * c, beta ** 3 * sh, beta ** 3 * ch]
        if bc == "C": return [w, w1]
        if bc == "S": return [w, w2]
        if bc == "F":
            if tip: return [w2, [a + tip * beta ** 4 * b for a, b in zip(w3, w)]]
            return [w2, w3]
    def det(z):
        beta = mp.mpf(z) / L; nseg = len(segs); n = 4 * nseg; M = mp.zeros(n, n); r = 0
        for j, row in enumerate(rows_end(bcL, 0, beta)):
            for c in range(4): M[r, c] = row[c]
            r += 1
        last = 4 * (nseg - 1); ell = segs[-1][1] - segs[-1][0]
        for row in rows_end(bcR, ell, beta, tip=mut):
            for c in range(4): M[r, last + c] = row[c]
            r += 1
        if nseg == 2:
            l1 = segs[0][1] - segs[0][0]
            def ev(xi):
                c, s, ch, sh = mp.cos(beta * xi), mp.sin(beta * xi), mp.cosh(beta * xi), mp.sinh(beta * xi)
                return ([c, s, ch, sh], [-beta * s, beta * c, beta * sh, beta * ch], [-beta ** 2 * c, -beta ** 2 * s, beta ** 2 * ch, beta ** 2 * sh], [beta ** 3 * s, -beta ** 3 * c, beta ** 3 * sh, beta ** 3 * ch])
            L0, L1 = ev(l1), ev(0)
            for q in range(3):                     # continuity of w, w', w''
                for c in range(4): M[r, c] = L0[q][c]; M[r, 4 + c] = -L1[q][c]
                r += 1
            for c in range(4):                     # [w'''] = (mu beta^4 - kappa) w
                M[r, c] = -L0[3][c] - (mu * beta ** 4 - kap) * L0[0][c]; M[r, 4 + c] = L1[3][c]
            r += 1
        return mp.det(M)
    roots, prev, bprev, b = [], None, None, mp.mpf("0.05")
    while len(roots) < nmodes and b < bmax:
        d = det(b)
        if prev is not None and d * prev < 0:
            roots.append(mp.findroot(det, (bprev, b), solver="illinois", tol=1e-26, verify=False, maxsteps=300))
        prev, bprev, b = d, b, b + mp.mpf("0.03")
    return np.array([float((r / L) ** 2 * mp.sqrt(EIm / rAm)) for r in roots])

# ------------------------------------------------------------------ discretisations
def _bc_rows(basis, bc, x, kind):   # essential rows (kind='e') and natural rows (kind='n'), at local x of `basis`
    if bc == "C": return [basis.d([x], 0), basis.d([x], 1)] if kind == "e" else []
    if bc == "S": return [basis.d([x], 0)] if kind == "e" else [basis.d([x], 2)]
    if bc == "F": return [] if kind == "e" else [basis.d([x], 2), basis.d([x], 3)]

class PointMassBeam:
    def __init__(self, bcL, bcR, x0=None, m=0.0, k=0.0, m_tip=0.0, L=Lb):
        self.bcL, self.bcR, self.x0, self.m, self.k, self.mt, self.L = bcL, bcR, x0, m, k, m_tip, L
        self.nmodes = 5
    def exact(self): return exact_freqs(self.bcL, self.bcR, self.L, self.x0, self.m, self.k, self.mt, self.nmodes)
    def _bases(self, n):
        if self.x0 is None: return [Bern1D(self.L, n)], [0.0, self.L]
        return [Bern1D(self.x0, n), Bern1D(self.L - self.x0, n)], [0.0, self.x0, self.L]
    def _phi0(self, Bs, tot, N, where):     # row picking w at attachment point / right end
        r = np.zeros(tot)
        if where == "x0": r[:N] = Bs[0].d([Bs[0].L], 0)[0]
        else:             r[(len(Bs) - 1) * N:] = Bs[-1].d([Bs[-1].L], 0)[0]
        return r
    def _ess(self, Bs, N, tot, strong):
        rows = []; ns = len(Bs)
        def put(seg, row):
            r = np.zeros(tot); r[seg * N:(seg + 1) * N] = row; return r
        for row in _bc_rows(Bs[0], self.bcL, 0.0, "e"): rows.append(put(0, row[0]))
        for row in _bc_rows(Bs[-1], self.bcR, Bs[-1].L, "e"): rows.append(put(ns - 1, row[0]))
        if strong:
            for row in _bc_rows(Bs[0], self.bcL, 0.0, "n"): rows.append(put(0, row[0]))
            nat_r = _bc_rows(Bs[-1], self.bcR, Bs[-1].L, "n")
            if self.mt and self.bcR == "F": nat_r = nat_r[:1]          # keep w''=0 ; w''' row becomes lambda-dependent
            for row in nat_r: rows.append(put(ns - 1, row[0]))
        if ns == 2:
            for q in range(3 if strong else 2):    # continuity of w, w' (+ w'' for the strong form)
                r = np.zeros(tot); r[:N] = Bs[0].d([Bs[0].L], q)[0]; r[N:] = -Bs[1].d([0.0], q)[0]; rows.append(r)
        return rows
    # ---------------- weak form: one solve, all modes
    def _weak_matrices(self, n, ng):
        Bs, xs = self._bases(n); Nb = n + 1; tot = len(Bs) * Nb
        K = np.zeros((tot, tot)); M = np.zeros((tot, tot))
        for i, B in enumerate(Bs):
            ell = xs[i + 1] - xs[i]; sl = slice(i * Nb, (i + 1) * Nb)
            x, w = gl(0, ell, ng); P, D2 = B.d(x, 0), B.d(x, 2)
            K[sl, sl] = EI * D2.T @ (w[:, None] * D2); M[sl, sl] = rA * P.T @ (w[:, None] * P)
        return Bs, K, M, tot, Nb
    def weak(self, n=23, ng=40, vectors=False, rcond=None, symmetric=False):
        t0 = time.perf_counter(); Bs, K, M, tot, N = self._weak_matrices(n, ng)
        if self.x0 is not None:
            p0 = self._phi0(Bs, tot, N, "x0"); K = K + self.k * np.outer(p0, p0); M = M + self.m * np.outer(p0, p0)
        if self.mt:
            pt = self._phi0(Bs, tot, N, "end"); M = M + self.mt * np.outer(pt, pt)
        Z, _ = null_space_svd(np.vstack(self._ess(Bs, N, tot, False)))
        lam, V, kept = solve_pencil(Z.T @ K @ Z, Z.T @ M @ Z, rcond, symmetric)
        lr = np.real(lam[np.isfinite(lam)]); floor = 1e-6 * lr[lr > 0].min() if (lr > 0).any() else 0
        idx, om = lowest(lam, self.nmodes, floor)
        return Res(om, (Z @ V[:, idx]).real if vectors else None, dict(time=time.perf_counter() - t0, solves=1, red=Z.shape[1]))
    # ---------------- strong form (published Eig-PIELM on each element): eigenvalue-dependent constraint frozen at lam_star
    def _strong_matrices(self, n, N):
        Bs, xs = self._bases(n); Nb = n + 1; tot = len(Bs) * Nb
        G = np.zeros((tot, tot)); S4 = np.zeros((tot, tot))
        for i, B in enumerate(Bs):
            ell = xs[i + 1] - xs[i]; sl = slice(i * Nb, (i + 1) * Nb)
            u = uniform_interior(ell, N); Pu = B.d(u, 0)
            G[sl, sl] = Pu.T @ Pu; S4[sl, sl] = (EI / rA) * Pu.T @ B.d(u, 4)
        return Bs, G, S4, tot, Nb
    def _strong_pass(self, Bs, G, S4, tot, N, lam_star, return_vec=False):
        rows = self._ess(Bs, N, tot, True)
        if self.x0 is not None:                     # EI [w3] = (m lam - k) w(x0)   (w3 = third derivative jump)
            r = np.zeros(tot); r[:N] = -EI * Bs[0].d([Bs[0].L], 3)[0]; r[N:] = EI * Bs[1].d([0.0], 3)[0]
            rows.append(r - (self.m * lam_star - self.k) * self._phi0(Bs, tot, N, "x0"))
        if self.mt and self.bcR == "F":             # EI w3(L) = -mt lam w(L)
            r = np.zeros(tot); B = Bs[-1]
            r[(len(Bs) - 1) * N:] = EI * B.d([B.L], 3)[0] + self.mt * lam_star * B.d([B.L], 0)[0]; rows.append(r)
        Z, _ = null_space_svd(np.vstack(rows))
        lam, V, _ = solve_pencil(Z.T @ S4 @ Z, Z.T @ G @ Z)
        ok = np.isfinite(lam) & (np.abs(lam.imag) <= 1e-8 * np.abs(lam.real) + 1e-300)
        order = np.flatnonzero(ok)[np.argsort(lam.real[ok])]      # real eigenvalues ascending; negatives kept so the branch index cannot shift
        if return_vec: return lam.real[order], Z, V[:, order]
        return lam.real[order]
    def strong(self, n=23, N=5000, mode="brent", tol=1e-14, vectors=False):
        """mode='onepass': lam*=0 (ignores the eigenvalue dependence -> solves the wrong problem).
           mode='brent'  : per mode, root-find h(lam*) = mu_j(lam*) - lam* = 0 (h monotone, bracketed by [0, mu_j(0)]);
                           every function evaluation is a full constrained eigensolve.
           vectors=True  : one extra constrained solve at the converged lam* per mode; beta = Z y_j (Section 3.2 recovery)."""
        t0 = time.perf_counter(); Bs, G, S4, tot, Nb = self._strong_matrices(n, N)
        nsolve = [0]
        def mu(lam_star, j):
            nsolve[0] += 1; return self._strong_pass(Bs, G, S4, tot, Nb, lam_star)[j]
        if mode == "onepass" or (self.m == 0 and self.mt == 0):      # no lambda-dependence -> a single pass is exact
            if vectors:
                lam, Z, V = self._strong_pass(Bs, G, S4, tot, Nb, 0.0, return_vec=True)
                pos = np.flatnonzero(lam > 1e-6 * lam[lam > 0][0])[:self.nmodes]
                return Res(np.sqrt(lam[pos]), (Z @ V[:, pos]).real, dict(time=time.perf_counter() - t0, solves=1, per_mode=[1] * self.nmodes))
            lam = self._strong_pass(Bs, G, S4, tot, Nb, 0.0)[:self.nmodes]
            return Res(np.sqrt(lam), None, dict(time=time.perf_counter() - t0, solves=1, per_mode=[1] * self.nmodes))
        oms, per, betas = [], [], []
        for j in range(self.nmodes):
            n0 = nsolve[0]; hi = mu(0.0, j)
            root = brentq(lambda z: mu(z, j) - z, 0.0, hi, xtol=tol * hi, rtol=1e-15, maxiter=200)
            oms.append(np.sqrt(root)); per.append(nsolve[0] - n0)
            if vectors:
                lam, Z, V = self._strong_pass(Bs, G, S4, tot, Nb, root, return_vec=True)
                betas.append((Z @ V[:, j]).real); nsolve[0] += 1
        beta_out = np.column_stack(betas) if vectors else None
        return Res(np.array(oms), beta_out, dict(time=time.perf_counter() - t0, solves=nsolve[0], per_mode=per))

# ------------------------------------------------------------------ extras: single-domain weak form + field evaluation
def weak_global(self, n=23, ng=None, vectors=False, rcond=None):
    """One global Bernstein polynomial on [0,L]; attachments enter as rank-one terms phi(x0)phi(x0)^T (no splitting)."""
    t0 = time.perf_counter(); ng = ng or max(40, n + 2); B = Bern1D(self.L, n); x, w = gl(0, self.L, ng)
    P, D2 = B.d(x, 0), B.d(x, 2); K = EI * D2.T @ (w[:, None] * D2); M = rA * P.T @ (w[:, None] * P)
    if self.x0 is not None:
        p0 = B.d([self.x0], 0)[0]; K = K + self.k * np.outer(p0, p0); M = M + self.m * np.outer(p0, p0)
    if self.mt:
        pt = B.d([self.L], 0)[0]; M = M + self.mt * np.outer(pt, pt)
    rows = []
    for row in _bc_rows(B, self.bcL, 0.0, "e"): rows.append(row[0])
    for row in _bc_rows(B, self.bcR, self.L, "e"): rows.append(row[0])
    Z, _ = null_space_svd(np.vstack(rows)); lam, V, _ = solve_pencil(Z.T @ K @ Z, Z.T @ M @ Z, rcond)
    lr = np.real(lam[np.isfinite(lam)]); idx, om = lowest(lam, self.nmodes, 1e-6 * lr[lr > 0].min())
    return Res(om, (Z @ V[:, idx]).real if vectors else None, dict(time=time.perf_counter() - t0, solves=1))
PointMassBeam.weak_global = weak_global

def field(self, beta, n, x, k, split=True):
    """k-th derivative of a mode (beta = coefficient vector) on points x"""
    if not split or self.x0 is None:
        return Bern1D(self.L, n).d(x, k) @ beta
    B1, B2 = Bern1D(self.x0, n), Bern1D(self.L - self.x0, n); out = np.zeros(len(x)); N = n + 1
    l = x <= self.x0
    out[l] = B1.d(x[l], k) @ beta[:N]; out[~l] = B2.d(x[~l] - self.x0, k) @ beta[N:]
    return out
PointMassBeam.field = field
