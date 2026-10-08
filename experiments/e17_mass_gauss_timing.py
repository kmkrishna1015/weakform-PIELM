"""Mass-beam weak form with Gauss quadrature, timed on the same machine/protocol as e12 (run e12 first: needs cost_equivalence.json)."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
from _common import *
from pielm.pointmass import PointMassBeam, rA, Lb, EI
def med(fn, reps):
    fn(); ts = []
    for _ in range(reps): t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))
C = load_json("cost_equivalence"); out = {"mass": {}, "nmodes": []}
CASES = {"H1": lambda: PointMassBeam("C", "F", None, 0, 0, m_tip=1.0*rA*Lb), "H2": lambda: PointMassBeam("S", "S", 0.3*Lb, 1.0*rA*Lb),
         "H3": lambda: PointMassBeam("C", "F", 0.6*Lb, 2.0*rA*Lb), "H4": lambda: PointMassBeam("C", "F", 0.4*Lb, 1.0*rA*Lb, k=100*EI/Lb**3)}
for k, f in CASES.items():
    b = f(); tg = med(lambda: b.weak(23), 15)
    out["mass"][k] = dict(t_weak_gauss=tg, t_strong=C["mass"][k]["t_strong"], ratio_gauss=C["mass"][k]["t_strong"] / tg); print(k, out["mass"][k])
b = CASES["H2"]()
for r in C["nmodes"]:
    b.nmodes = r["nmodes"]; out["nmodes"].append(dict(nmodes=r["nmodes"], t_weak_gauss=med(lambda: b.weak(23), 9))); print(out["nmodes"][-1])
save_json(out, "mass_gauss_timing")
