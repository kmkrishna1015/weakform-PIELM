"""Strong-form (collocation) pencils with EXACT Gauss weights W = diag(w): Z'Phi'W Psi Z y = lambda Z'Phi'W Phi Z y  (= Ritz on V_s)."""
import numpy as np
from .numerics import gl, null_space_svd, solve_pencil, lowest, Res

def strong1d(p, basis, ng=40, rcond=1e-12):
    x, w = gl(0, p.L, ng); Phi = basis.d(x, 0); Psi = p.a_s * basis.d(x, 2 * p.m)
    G = Phi.T @ (w[:, None] * Phi); S = Phi.T @ (w[:, None] * Psi)
    Z, rk = null_space_svd(np.vstack(p._bcrows(basis, p.ess + p.nat)))
    lam, V, kept = solve_pencil(Z.T @ S @ Z, Z.T @ G @ Z, rcond); idx, om = lowest(lam, p.nmodes, p._floor())
    return Res(om, None, dict(red_dim=Z.shape[1]))

def helmholtz(p, bx, by, ng=(20, 20), rcond=1e-12):
    (x, wx), (y, wy) = gl(0, p.Lx, ng[0]), gl(0, p.Ly, ng[1]); Px, Py = bx.d(x, 0), by.d(y, 0)
    Gx, Gy = Px.T @ (wx[:, None] * Px), Py.T @ (wy[:, None] * Py)
    Ax, Ay = Px.T @ (wx[:, None] * bx.d(x, 2)), Py.T @ (wy[:, None] * by.d(y, 2))
    G = np.kron(Gx, Gy); S = -(np.kron(Ax, Gy) + np.kron(Gx, Ay))
    xb, yb = np.linspace(0, p.Lx, bx.nphi), np.linspace(0, p.Ly, by.nphi); Bx_b, By_b = bx.d(xb, 0), by.d(yb, 0)
    rows = [np.kron(bx.d([0.0], 1), By_b), np.kron(bx.d([p.Lx], 1), By_b), np.kron(Bx_b, by.d([0.0], 1)), np.kron(Bx_b, by.d([p.Ly], 1))]
    Z, rk = null_space_svd(np.vstack(rows))
    lam, V, kept = solve_pencil(Z.T @ S @ Z, Z.T @ G @ Z, rcond); idx, om = lowest(lam, p.nmodes, p._floor())
    return Res(p.c * om, None, dict(red_dim=Z.shape[1]))

def plate2d(p, bx, by, ng=(26, 20), nb=(18, 14), rcond=1e-12):
    c11, c12, c66, rho = p.c11, p.c12, p.c66, p.rho
    (x, wx), (y, wy) = gl(0, p.Lx, ng[0]), gl(0, p.Ly, ng[1]); B = p._blocks(bx, by, x, wx, y, wy)
    G0 = np.kron(B["Mx"], B["My"]); Z0 = np.zeros_like(G0)
    S11 = -(c11 * np.kron(B["A2x"], B["My"]) + c66 * np.kron(B["Mx"], B["A2y"])) / rho
    S22 = -(c66 * np.kron(B["A2x"], B["My"]) + c11 * np.kron(B["Mx"], B["A2y"])) / rho
    S12 = -(c12 + c66) * np.kron(B["A1x"], B["A1y"]) / rho
    G = np.block([[G0, Z0], [Z0, G0]]); S = np.block([[S11, S12], [S12, S22]])
    ess, nat = p._bc_rows(bx, by, nb); Z, rk = null_space_svd(np.vstack([ess, nat]))
    lam, V, kept = solve_pencil(Z.T @ S @ Z, Z.T @ G @ Z, rcond); idx, om = lowest(lam, p.nmodes, p._floor())
    return Res(om, None, dict(red_dim=Z.shape[1]))
