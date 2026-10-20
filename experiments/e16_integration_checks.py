"""Supplementary study (not quoted in the paper): (a) scattered-point rule vs Gauss at higher plate degree (n=20, 24) incl. weight-fit time,
(b) relative moment residual ||A w - b|| / ||b|| of the minimum-norm moment fit for the cases of Table D.1."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
from _common import *
from numpy.polynomial import legendre as Lg
from scipy.stats import qmc
from pielm import scattered as SC
from pielm.kirchhoff import Plate
out = {}
for code in ("FFFF", "CFFF"):
    P = Plate(code)
    for n in (20, 24):
        g = P.weak(n, n + 12, rcond=1e-12, symmetric=True).omega; s = SC.kirchhoff_weak(P, n)
        out[f"{code}_{n}"] = dict(dev=float(np.max(np.abs(s.omega / g - 1))), fit=s.info["stages"][0], npts=int(s.info["npts"])); print(code, n, out[f"{code}_{n}"], flush=True)
save_json(out, "scattered_highdeg")
def res1d(q, N, L=1.0):
    x = qmc.Halton(1, scramble=False).random(N + 1)[1:, 0] * L; A = Lg.legvander(2 * x / L - 1, q).T; b = np.zeros(q + 1); b[0] = L
    w = np.linalg.lstsq(A, b, rcond=None)[0]; return float(np.linalg.norm(A @ w - b) / np.linalg.norm(b))
def res2d(Lx, Ly, qx, qy, N):
    P = qmc.Halton(2, scramble=False).random(N + 1)[1:] * np.array([Lx, Ly])
    Ps, Pt = Lg.legvander(2 * P[:, 0] / Lx - 1, qx), Lg.legvander(2 * P[:, 1] / Ly - 1, qy)
    A = (Ps[:, :, None] * Pt[:, None, :]).reshape(N, -1).T; b = np.zeros((qx + 1, qy + 1)); b[0, 0] = Lx * Ly; b = b.ravel()
    w = np.linalg.lstsq(A, b, rcond=None)[0]; return float(np.linalg.norm(A @ w - b) / np.linalg.norm(b))
R = {"beam_cf": res1d(46, 376), "helmholtz": res2d(5.0, 3.0, 26, 26, 6 * 27 * 27), "plate": res2d(1.0, 0.6, 28, 20, 6 * 29 * 21), "kirchhoff_n16": res2d(1.0, 1.0, 32, 32, 6 * 33 * 33)}
print(R); save_json(R, "moment_residual")
