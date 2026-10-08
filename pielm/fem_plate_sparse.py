import numpy as np
import sympy as sp
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh, splu

x, L = sp.symbols('x L', positive=True)
N1 = 1 - 3*(x/L)**2 + 2*(x/L)**3
N2 = x - 2*x**2/L + x**3/L**2
N3 = 3*(x/L)**2 - 2*(x/L)**3
N4 = -x**2/L + x**3/L**2
N = [N1, N2, N3, N4]
def deriv(f, order):
    g = f
    for _ in range(order): g = sp.diff(g, x)
    return g
def elem_matrix(a, b):
    Da = [deriv(f, a) for f in N]; Db = [deriv(f, b) for f in N]
    M = sp.zeros(4, 4)
    for i in range(4):
        for k in range(4):
            M[i, k] = sp.integrate(Da[i]*Db[k], (x, 0, L))
    return sp.lambdify(L, M, 'numpy')
I00_f = elem_matrix(0,0); I11_f = elem_matrix(1,1)
I22_f = elem_matrix(2,2); I20_f = elem_matrix(2,0)

def assemble_1d_sparse(n_el, length):
    h = length/n_el
    ndof = 2*(n_el+1)
    e00, e11, e22, e20 = I00_f(h), I11_f(h), I22_f(h), I20_f(h)
    rows, cols = [], []
    d00, d11, d22, d20 = [], [], [], []
    for e in range(n_el):
        idx = [2*e, 2*e+1, 2*e+2, 2*e+3]
        for i in range(4):
            for k in range(4):
                rows.append(idx[i]); cols.append(idx[k])
                d00.append(e00[i,k]); d11.append(e11[i,k])
                d22.append(e22[i,k]); d20.append(e20[i,k])
    G00 = sps.coo_matrix((d00,(rows,cols)), shape=(ndof,ndof)).tocsr()
    G11 = sps.coo_matrix((d11,(rows,cols)), shape=(ndof,ndof)).tocsr()
    G22 = sps.coo_matrix((d22,(rows,cols)), shape=(ndof,ndof)).tocsr()
    G20 = sps.coo_matrix((d20,(rows,cols)), shape=(ndof,ndof)).tocsr()
    return G00, G11, G22, G20

def plate_KM_sparse(n_el_x, n_el_y, nu=0.3):
    X00,X11,X22,X20 = assemble_1d_sparse(n_el_x, 1.0)
    Y00,Y11,Y22,Y20 = assemble_1d_sparse(n_el_y, 1.0)
    X02, Y02 = X20.T.tocsr(), Y20.T.tocsr()
    K = (sps.kron(X22,Y00) + sps.kron(X00,Y22)
         + nu*(sps.kron(X20,Y02) + sps.kron(X02,Y20))
         + 2*(1-nu)*sps.kron(X11,Y11)).tocsr()
    M = sps.kron(X00,Y00).tocsr()
    K = 0.5*(K+K.T); M = 0.5*(M+M.T)
    ndof_x, ndof_y = 2*(n_el_x+1), 2*(n_el_y+1)
    return K, M, ndof_x, ndof_y

def clamp_edge_x0(n_el_x, n_el_y):
    ndof_y = 2*(n_el_y+1)
    return [ix*ndof_y+jy for ix in (0,1) for jy in range(ndof_y)]

def solve_sparse(n_el_x, n_el_y, clamp='none', n_modes=9, sigma=1e-6, reps=3):
    import time
    times=[]
    for _ in range(reps):
        t0=time.perf_counter()
        K, M, ndof_x, ndof_y = plate_KM_sparse(n_el_x, n_el_y)
        ntot = ndof_x*ndof_y
        if clamp=='x0':
            remove = clamp_edge_x0(n_el_x, n_el_y)
            keep = np.setdiff1d(np.arange(ntot), remove)
            Kc = K[keep,:][:,keep]; Mc = M[keep,:][:,keep]
            s = 1e-8  # K nonsingular, but tiny shift is harmless
        else:
            keep = np.arange(ntot)
            Kc, Mc = K, M
            s = sigma  # K singular (rigid modes) -> need small positive shift
        w, v = eigsh(Kc, k=n_modes, M=Mc, sigma=s, which='LM')
        t1=time.perf_counter()
        times.append(t1-t0)
    order = np.argsort(w)
    w = np.clip(w[order],0,None)
    return np.sqrt(w), keep.size, np.median(times)*1000

if __name__=='__main__':
    ref_FFFF = np.array([13.4682,19.5961,24.2702,34.8009,34.8009,61.0932])
    print("FFFF sparse shift-invert sweep (dropping 3 rigid modes):")
    for n in [16,20,24,30,36,44]:
        om, dof, tms = solve_sparse(n, n, clamp='none', n_modes=9, sigma=1e-6, reps=3)
        om6 = om[3:9]
        err = np.max(np.abs(om6/ref_FFFF-1))*100
        print(f"n_el={n:3d} dof={dof:5d} time={tms:9.2f} ms   Omega={np.round(om6,4)}  max err={err:.3e} %")

    print("\nCFFF sparse shift-invert sweep:")
    for n in [16,20,24,30,36,44]:
        om, dof, tms = solve_sparse(n, n, clamp='x0', n_modes=6, sigma=1e-8, reps=3)
        print(f"n_el={n:3d} dof={dof:5d} time={tms:9.2f} ms   Omega={np.round(om,4)}")

def resweep_fixed():
    ref_FFFF = np.array([13.4682,19.5961,24.2702,34.8009,34.8009,61.0932])
    print("FFFF sparse, sigma=1.0 (away from the exact rigid-body null space):")
    for n in [16,20,24,30,36,44,50]:
        om, dof, tms = solve_sparse(n, n, clamp='none', n_modes=9, sigma=1.0, reps=3)
        om6 = om[3:9]
        err = np.max(np.abs(om6/ref_FFFF-1))*100
        print(f"n_el={n:3d} dof={dof:5d} time={tms:9.2f} ms   Omega={np.round(om6,4)}  max err={err:.3e} %")
