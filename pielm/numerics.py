"""Shared numerics.  Both formulations use exactly the same eigensolver (`solve_pencil`)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single BLAS thread for fair timings
import time, warnings
import numpy as np
import scipy.linalg as sla
warnings.filterwarnings("ignore")


def gl(a, b, n):
    """Gauss-Legendre points and weights on [a, b]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * (x + 1) + a, 0.5 * (b - a) * w


def uniform_interior(L, N):
    """N equispaced interior points of (0, L) -- the sampling of the published Eig-PIELM."""
    return np.linspace(0.0, L, N + 2)[1:-1]


def null_space_svd(B, rtol=1e-12):
    """Orthonormal basis Z of ker(B).  Rows are normalised first (scale-invariant rank decision)."""
    if B.shape[0] == 0:
        return np.eye(B.shape[1]), 0
    nr = np.linalg.norm(B, axis=1, keepdims=True); nr[nr == 0] = 1.0
    _, s, Vh = np.linalg.svd(B / nr, full_matrices=True)
    rank = int(np.sum(s > rtol * s[0]))
    return Vh[rank:].T, rank


def solve_pencil(A, B, rcond=None, symmetric=False):
    """Eigenpairs of the reduced pencil  A y = lambda B y  -- the single eigensolver entry point of the package.

    symmetric=False (default): dense general solver scipy.linalg.eig(A, B); no symmetrisation.  This is the solver of the
                          published Eig-PIELM (its pencil S, G is not symmetric) and, for a like-for-like comparison, is
                          also used for the weak form.
    symmetric=True      : for self-adjoint weak forms K y = lambda M y with K symmetric, M symmetric positive definite:
                          scipy.linalg.eigh(K, M) (Cholesky-based, real eigenvalues by construction).  Exploits the
                          structure of the weak form; never used for the strong form.  Because the Gram matrix of a
                          polynomial basis is nearly singular, the package calls it with rcond=1e-12 (Loewdin-reduced
                          eigh), see run_method('weak_sym').
    rcond=float         : identical Gram/mass-matrix truncation for either solver.  Directions of the symmetrised B with
                          eigenvalue <= rcond*max are discarded and the projected standard problem is solved by eig (or
                          eigh if symmetric=True).  Used only in the conditioning studies.
    Returns (lam, V, number_of_retained_directions); failures (e.g. B not numerically positive definite) return NaNs.
    """
    n = A.shape[0]
    if symmetric:
        A = 0.5 * (A + A.T); B = 0.5 * (B + B.T)
    if rcond is None:
        try:
            lam, V = sla.eigh(A, B, check_finite=False) if symmetric else sla.eig(A, B, check_finite=False)
        except Exception:
            return np.full(n, np.nan), np.full((n, n), np.nan), 0
        return lam, V, n
    eb, Vb = np.linalg.eigh(0.5 * (B + B.T))
    keep = eb > rcond * eb[-1]
    if not keep.any():
        return np.zeros(0), np.zeros((n, 0)), 0
    T = Vb[:, keep] / np.sqrt(eb[keep])
    At = T.T @ A @ T
    if symmetric:
        lam, W = np.linalg.eigh(0.5 * (At + At.T))
    else:
        lam, W = sla.eig(At, check_finite=False)
    return lam, T @ W, int(keep.sum())


def lowest(lam, n, floor):
    """First n positive real eigenvalues above `floor`; returns (indices, sqrt values)."""
    lam = np.asarray(lam)
    ok = np.isfinite(lam) & (np.abs(lam.imag) <= 1e-8 * np.abs(lam.real) + 1e-300) & (lam.real > floor)
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(lam.real[idx])][:n]
    return idx, np.sqrt(lam.real[idx])


def rel_err_pct(num, ex):
    num = np.asarray(num, float); ex = np.asarray(ex, float)[:len(num)]
    return np.abs(num - ex) / np.abs(ex) * 100.0


def shape_err(u_num, u_ex, w=None):
    """Sign/scale-invariant mode-shape error  || u_ex - (u_ex.uh) uh || / ||u_ex||  (= sqrt(1-MAC))."""
    u_num = u_num.ravel(); u_ex = u_ex.ravel()
    w = np.ones_like(u_num) if w is None else np.repeat(w.ravel(), u_num.size // w.size)
    un = u_num / np.sqrt(np.sum(w * u_num ** 2)); ue = u_ex / np.sqrt(np.sum(w * u_ex ** 2))
    r = ue - np.sum(w * ue * un) * un
    return float(np.sqrt(np.sum(w * r ** 2)))


class Res:
    def __init__(self, omega, beta=None, info=None):
        self.omega, self.beta, self.info = omega, beta, info or {}


def finish(lam, V, Z, nmodes, floor, vectors):
    idx, om = lowest(lam, nmodes, floor)
    beta = (Z @ V[:, idx]).real if vectors and len(idx) else None
    return om, beta
