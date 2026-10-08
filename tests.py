import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
"""Quick self-checks (about 10 s).  Run:  python tests.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pielm.numerics as nm
import pielm.problems as pr
from pielm.bases import Bern1D, Leg1D
from pielm.kirchhoff import Plate, ssss_exact

# 1. analytic derivatives of the polynomial bases against central differences
for B in (Bern1D(0.12, 23), Leg1D(0.12, 23)):
    x = np.linspace(0.02, 0.10, 7); h = 1e-6
    for k in (0, 1, 2):
        fd = (B.d(x + h, k) - B.d(x - h, k)) / (2 * h); an = B.d(x, k + 1)
        assert np.max(np.abs(fd - an)) / np.max(np.abs(an)) < 1e-4, (B.name, k)
# 2. Gauss-Legendre integrates a degree-2n polynomial Gram matrix exactly
B = Bern1D(1.0, 10); x, w = nm.gl(0, 1, 12); M = B.d(x, 0).T @ (w[:, None] * B.d(x, 0))
assert abs(M.sum() - 1.0) < 1e-13                                   # partition of unity: sum_ij int B_i B_j = 1
# 3. all formulations go through the very same eigensolver entry point (strong and weak: general; weak_sym: symmetric)
calls = []; orig = nm.solve_pencil
def spy(A, Bm, rcond=None, symmetric=False): calls.append(("solve_pencil", symmetric)); return orig(A, Bm, rcond, symmetric)
pr.solve_pencil = spy
P = pr.problems(); p = P["beam_cf"]
pr.run_method(p, "strong"); pr.run_method(p, "weak"); pr.run_method(p, "weak_sym")
assert calls == [("solve_pencil", False), ("solve_pencil", False), ("solve_pencil", True)], calls   # one entry point; strong never symmetric
pr.solve_pencil = orig
# 4. accuracy of both forms on all standard cases (loose thresholds; see results/benchmarks.json for the exact values)
for key in pr.ORDER:
    p = P[key]
    for m in pr.METHODS3:
        e = pr.max_err_pct(pr.run_method(p, m), p); assert e < 1e-5, (key, m, e)
# 5. weak-form Kirchhoff plate, simply supported, against the exact Omega = pi^2 (m^2 + q^2)
r = Plate("SSSS").weak(16, 30); assert np.abs(r.omega / ssss_exact() - 1).max() < 1e-10
print("all self-checks passed")
