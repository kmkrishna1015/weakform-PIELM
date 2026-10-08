"""Main-method results with the grid-free (scattered points + moment-fitted weights) integration rule.
Recomputes the weak-form columns of Tables 3, 4, 6 and the stage costs; Gauss values are recomputed alongside for a like-for-like check."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
from _common import *
import pielm.problems as PR, pielm.pointmass as PM
from pielm import scattered as SC
from pielm.kirchhoff import Plate
RC = 1e-12; P = problems(); out = {"table3": {}, "table4": {}, "table6": {}, "timing": {}}

def med(fn, reps=7):
    fn(); ts = []; st = []
    for _ in range(reps):
        t = time.perf_counter(); r = fn(); ts.append(time.perf_counter()-t); st.append(r.info.get("stages", (0, 0, 0)))
    return float(np.median(ts)), np.median(np.array(st), axis=0).tolist()

# ---- Table 3 -----------------------------------------------------------------------------------------------
g0 = PR.gl
def run_sc(p):
    B = bases(p)
    if p.dim == 1:
        PR.gl = SC.make_gl_sc(2 * 23)
        try: return p.weak(B, 40, False, RC, True)
        finally: PR.gl = g0
    if p.key == "helmholtz": return SC.helmholtz_weak(p, *B)
    return SC.plate2d_weak(p, *B)
for key in ORDER:
    p = P[key]; ex = p.exact(p.nmodes)
    rg = run_method(p, "weak_sym"); rs = run_sc(p)
    eg, es = rel_err_pct(rg.omega, ex), rel_err_pct(rs.omega, ex)
    out["table3"][key] = dict(omega1=float(rs.omega[0]), exact1=float(ex[0]), err_gauss=float(eg.max()), err_scattered=float(es.max()),
                              dev_gauss_scattered=float(np.max(np.abs(rs.omega/rg.omega-1))), nmodes=len(rs.omega))
    tg = med(lambda: run_method(p, "weak_sym")); ts = med(lambda: run_sc(p))
    out["timing"][key] = dict(gauss=tg[0], scattered=ts[0], stages_scattered=ts[1], strong=med(lambda: run_method(p, "strong"), 3)[0])
    print(key, out["table3"][key], out["timing"][key], flush=True)

# ---- Table 4 (general eigensolver, as in the paper) -------------------------------------------------------

from pielm.pointmass import PointMassBeam, rA, Lb, EI
CASES = {"H1": lambda: PointMassBeam("C", "F", None, 0, 0, m_tip=1.0*rA*Lb), "H2": lambda: PointMassBeam("S", "S", 0.3*Lb, 1.0*rA*Lb),
         "H3": lambda: PointMassBeam("C", "F", 0.6*Lb, 2.0*rA*Lb), "H4": lambda: PointMassBeam("C", "F", 0.4*Lb, 1.0*rA*Lb, k=100*EI/Lb**3)}
for key, f in CASES.items():
    b = f(); ex = b.exact(); rg = b.weak(23)
    PM.gl = SC.make_gl_sc(2 * 23)
    try: rs = b.weak(23); tsc = med(lambda: b.weak(23))[0]
    finally: PM.gl = g0
    eg, es = np.abs(rg.omega/ex-1)*100, np.abs(rs.omega/ex-1)*100
    out["table4"][key] = dict(omega1=float(rs.omega[0]), exact1=float(ex[0]), err_gauss=float(eg.max()), err_scattered=float(es.max()), t_scattered=tsc)
    print(key, out["table4"][key], flush=True)

# ---- Table 6 (Kirchhoff, n=16, symmetric truncated solver) ------------------------------------------------
from pielm.kirchhoff import ssss_exact
refs = {k: np.array(v) for k, v in load_json("P_refs").items()}
CODE = {"SSSS": "SSSS", "SFSF": "SSFF", "FFFF": "FFFF", "CFFF": "CFFF"}
for key, code in CODE.items():
    Pl = Plate(code); rg = Pl.weak(16, 40, rcond=RC, symmetric=True); rs = SC.kirchhoff_weak(Pl, 16)
    ex = refs[key]; eg, es = np.abs(rg.omega/ex-1).max(), np.abs(rs.omega/ex-1).max()
    out["table6"][key] = dict(omega1=float(rs.omega[0]), err_gauss=float(eg), err_scattered=float(es), dev=float(np.max(np.abs(rs.omega/rg.omega-1))), red=int(rs.info["red"]),
                              stages=rs.info["stages"], npts=int(rs.info["npts"]))
    print(key, out["table6"][key], flush=True)
save_json(out, "scattered_main")
