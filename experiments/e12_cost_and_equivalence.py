"""(1) strong form with the SAME scattered points and moment-fitted weights as the weak form (exact-rule strong form = Ritz on V_s);
(2) single-machine timings: mass beams (weak grid-free vs iterated strong), mode-count sweep, Q4 FEM."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
from _common import *
import pielm.pointmass as PM
from pielm import scattered as SC
from pielm.kirchhoff import Plate
from pielm.pointmass import PointMassBeam, rA, Lb, EI, exact_freqs
R = {}
# ---- (1)
refs = {k: np.array(v) for k, v in load_json("P_refs").items()}
for key, code in (("FFFF", "FFFF"), ("CFFF", "CFFF")):
    P = Plate(code); n = 16; b = Bern1D(1.0, n); X, w = SC.rule_2d(1.0, 1.0, 2*n, 2*n); W = w[:, None]
    F = SC._tens(b, b, X, 0, 0); L4 = SC._tens(b, b, X, 4, 0) + 2*SC._tens(b, b, X, 2, 2) + SC._tens(b, b, X, 0, 4)
    S, G = F.T @ (W*L4), F.T @ (W*F)
    Z, rk = null_space_svd(P._rows(b, n+1, strong=True))
    lam, V, kept = solve_pencil(Z.T@S@Z, Z.T@G@Z, 1e-12); idx, om = lowest(lam, 6, 1.0)
    # compare to Ritz on V_s with Gauss quadrature
    K, M, Zw, Zs = P.ritz_matrices(n, 40); l2, V2, k2 = solve_pencil(Zs.T@K@Zs, Zs.T@M@Zs, 1e-12, True); i2, o2 = lowest(l2, 6, 1.0)
    R["strong_mf_"+key] = dict(err_strong_mf=float(np.abs(om/refs[key]-1).max()), err_ritzVs_gauss=float(np.abs(o2/refs[key]-1).max()),
                               dev=float(np.max(np.abs(om/o2-1))), dim_Vs=int(Z.shape[1]))
    print(key, R["strong_mf_"+key], flush=True)
# ---- (2) timings
def med(fn, reps):
    fn(); ts = []
    for _ in range(reps): t = time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return float(np.median(ts))
g0 = PM.gl
CASES = {"H1": lambda: PointMassBeam("C", "F", None, 0, 0, m_tip=1.0*rA*Lb), "H2": lambda: PointMassBeam("S", "S", 0.3*Lb, 1.0*rA*Lb),
         "H3": lambda: PointMassBeam("C", "F", 0.6*Lb, 2.0*rA*Lb), "H4": lambda: PointMassBeam("C", "F", 0.4*Lb, 1.0*rA*Lb, k=100*EI/Lb**3)}
R["mass"] = {}
for key, f in CASES.items():
    b = f(); PM.gl = SC.make_gl_sc(46)
    try: tw = med(lambda: b.weak(23), 15)
    finally: PM.gl = g0
    rs = b.strong(23, N=5000, mode="brent"); ts = med(lambda: b.strong(23, N=5000, mode="brent"), 3)
    R["mass"][key] = dict(t_weak=tw, t_strong=ts, solves=int(rs.info["solves"]), ratio=ts/tw); print(key, R["mass"][key], flush=True)
b = CASES["H2"](); R["nmodes"] = []
for nm in (1, 2, 5, 10, 20):
    b.nmodes = nm; PM.gl = SC.make_gl_sc(46)
    try: tw = med(lambda: b.weak(23), 9)
    finally: PM.gl = g0
    try: rs = b.strong(23, N=5000, mode="brent"); ts = med(lambda: b.strong(23, N=5000, mode="brent"), 2); so = int(rs.info["solves"]); ok = len(rs.omega) == nm
    except Exception: ts, so, ok = float("nan"), -1, False
    R["nmodes"].append(dict(nmodes=nm, t_weak=tw, t_strong=ts, solves=so, ok=ok)); print(R["nmodes"][-1], flush=True)
h = problems()["helmholtz"]; ex = h.exact(5); R["fem"] = {}
for nn in (14, 200):
    t = med(lambda: fem_helmholtz(h.Lx, h.Ly, h.c, nn, nn), 3 if nn < 200 else 1); om = fem_helmholtz(h.Lx, h.Ly, h.c, nn, nn)
    R["fem"][str(nn)] = dict(time=t, err_pct=float(rel_err_pct(om, ex).max()), ndof=nn*nn)
print(R["fem"]); save_json(R, "cost_equivalence")
