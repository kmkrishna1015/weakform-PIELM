"""
BFS (Bogner-Fox-Schmit) C1 rectangular plate bending element via the
tensor-product-Hermite shortcut: a structured mesh of BFS elements on a
rectangle is exactly the Kronecker product of two 1D cubic-Hermite FE
spaces, so we can reuse the SAME bilinear-form assembly as the paper's
own Appendix B:
    K = D[ X22⊗Y00 + X00⊗Y22 + nu(X20⊗Y02 + X02⊗Y20) + 2(1-nu)X11⊗Y11 ]
    M = rho*h * X00⊗Y00
where X^{ab}, Y^{ab} are now *global FEM* matrices (banded, from element
assembly) instead of the paper's global Bernstein Gram matrices.

Unit square, D=1, rho*h=1  =>  Omega = sqrt(lambda) directly.
"""
import numpy as np
import sympy as sp
from scipy.linalg import eigh

# ---------------------------------------------------------------------
# 1. Symbolic 1D element matrices (same derivation as hermite_elem.py)
# ---------------------------------------------------------------------
x, L = sp.symbols('x L', positive=True)
N1 = 1 - 3*(x/L)**2 + 2*(x/L)**3
N2 = x - 2*x**2/L + x**3/L**2
N3 = 3*(x/L)**2 - 2*(x/L)**3
N4 = -x**2/L + x**3/L**2
N = [N1, N2, N3, N4]

def deriv(f, order):
    g = f
    for _ in range(order):
        g = sp.diff(g, x)
    return g

def elem_matrix(a, b):
    Da = [deriv(f, a) for f in N]
    Db = [deriv(f, b) for f in N]
    M = sp.zeros(4, 4)
    for i in range(4):
        for k in range(4):
            M[i, k] = sp.integrate(Da[i]*Db[k], (x, 0, L))
    return sp.lambdify(L, M, 'numpy')

I00_f = elem_matrix(0, 0)
I11_f = elem_matrix(1, 1)
I22_f = elem_matrix(2, 2)
I20_f = elem_matrix(2, 0)   # I02 = I20^T

# ---------------------------------------------------------------------
# 2. 1D global assembly (cubic-Hermite FE space on [0, length])
# ---------------------------------------------------------------------
def assemble_1d(n_el, length):
    h = length / n_el
    ndof = 2 * (n_el + 1)
    G00 = np.zeros((ndof, ndof))
    G11 = np.zeros((ndof, ndof))
    G22 = np.zeros((ndof, ndof))
    G20 = np.zeros((ndof, ndof))
    e00, e11, e22, e20 = I00_f(h), I11_f(h), I22_f(h), I20_f(h)
    for e in range(n_el):
        idx = [2*e, 2*e+1, 2*e+2, 2*e+3]
        for i in range(4):
            for k in range(4):
                G00[idx[i], idx[k]] += e00[i, k]
                G11[idx[i], idx[k]] += e11[i, k]
                G22[idx[i], idx[k]] += e22[i, k]
                G20[idx[i], idx[k]] += e20[i, k]
    return G00, G11, G22, G20

# ---------------------------------------------------------------------
# 3. Assemble the 2D plate K, M via Kronecker products (Appendix B)
# ---------------------------------------------------------------------
def plate_KM(n_el_x, n_el_y, nu=0.3):
    X00, X11, X22, X20 = assemble_1d(n_el_x, 1.0)
    Y00, Y11, Y22, Y20 = assemble_1d(n_el_y, 1.0)
    X02, Y02 = X20.T, Y20.T
    K = (np.kron(X22, Y00) + np.kron(X00, Y22)
         + nu*(np.kron(X20, Y02) + np.kron(X02, Y20))
         + 2*(1-nu)*np.kron(X11, Y11))
    M = np.kron(X00, Y00)
    K = 0.5*(K+K.T)   # symmetrize away round-off
    M = 0.5*(M+M.T)
    ndof_x = 2*(n_el_x+1)
    ndof_y = 2*(n_el_y+1)
    return K, M, ndof_x, ndof_y

# ---------------------------------------------------------------------
# 4. dof indexing: global dof (ix, jy) tensor index, ix in [0,ndof_x),
#    jy in [0,ndof_y); flattened index = ix*ndof_y + jy (row-major, x slow)
#    Node (p,q) with p in [0,n_el_x], q in [0,n_el_y] owns local dof
#    (w,w_x) at position 2p in x-block and (w,w_y) at position 2q in y-block.
#    So global dof (2p,2q)=w_{pq}, (2p+1,2q)=w_x, (2p,2q+1)=w_y, (2p+1,2q+1)=w_xy
# ---------------------------------------------------------------------
def clamp_edge_x0(n_el_x, n_el_y):
    """Return list of flattened dof indices to REMOVE for a clamped edge at x=0:
       w=0 and w_x=0 for every node along x=0 (i.e. ix in {0,1}, all jy)."""
    ndof_y = 2*(n_el_y+1)
    remove = []
    for ix in (0, 1):
        for jy in range(ndof_y):
            remove.append(ix*ndof_y + jy)
    return remove

def solve_modes(n_el_x, n_el_y, nu=0.3, clamp='none', n_modes=6, drop_rigid=False):
    K, M, ndof_x, ndof_y = plate_KM(n_el_x, n_el_y, nu)
    ntot = ndof_x*ndof_y
    all_dof = np.arange(ntot)
    if clamp == 'x0':
        remove = clamp_edge_x0(n_el_x, n_el_y)
        keep = np.setdiff1d(all_dof, remove)
    elif clamp == 'none':
        keep = all_dof
    else:
        raise ValueError(clamp)
    Kc = K[np.ix_(keep, keep)]
    Mc = M[np.ix_(keep, keep)]
    w, v = eigh(Kc, Mc)
    w = np.clip(w, 0, None)
    omega = np.sqrt(w)
    if drop_rigid:
        omega = omega[3:]   # discard 3 (near-)zero rigid-body modes
    return omega[:n_modes], keep.size

if __name__ == '__main__':
    print("=== sanity: simply-supported analog not directly testable here")
    print("    (BFS needs w=0 AND M_n=0; M_n=0 is natural for a straight edge")
    print("     under this bilinear form only if wnn-type dof are free, which")
    print("     they are for w=0-only clamp -- so 'w=0 only' IS simple support")
    print("     for this element). Quick S-S-S-S check vs exact pi^2(m^2+n^2):")

    # Simply supported on all 4 edges: constrain w=0 (NOT w_x/w_y) at boundary nodes
    def ss_all_edges(n_el_x, n_el_y, nu=0.3, n_modes=6):
        K, M, ndof_x, ndof_y = plate_KM(n_el_x, n_el_y, nu)
        remove = []
        for p in range(n_el_x+1):
            for q in range(n_el_y+1):
                if p in (0, n_el_x) or q in (0, n_el_y):
                    ix, jy = 2*p, 2*q
                    remove.append(ix*ndof_y+jy)   # w = 0 only
        ntot = ndof_x*ndof_y
        keep = np.setdiff1d(np.arange(ntot), remove)
        Kc = K[np.ix_(keep, keep)]
        Mc = M[np.ix_(keep, keep)]
        w, v = eigh(Kc, Mc)
        return np.sqrt(np.clip(w, 0, None))[:n_modes], keep.size

    for n in [6, 10, 14]:
        om, ndof = ss_all_edges(n, n)
        exact = [np.pi**2*(m**2+q**2) for m in [1,1,2,2,1] for q in [1,2,1,2,3]]
        exact = sorted(set([np.pi**2*(1+1), np.pi**2*(1+4), np.pi**2*(4+1),
                             np.pi**2*(4+4), np.pi**2*(1+9)]))
        print(f"n_el={n:2d} dof={ndof:4d}  Omega_FEM = {np.round(om,4)}")
    print("exact SSSS lowest few (pi^2(m^2+q^2)):", np.round(exact,4))
