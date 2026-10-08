"""Numerical check of the min-max ordering  lambda_exact <= lambda_weak <= lambda_strong  (Section 3).
Exact-integration Rayleigh-Ritz on the two nested admissible spaces V_s (essential + natural rows projected, as the strong form
does) and V_w (essential rows only), with Gauss-Legendre quadrature and the SAME eigensolver.  No separate strong-form variant
is involved: the strong-form space is obtained from the weak-form bilinear forms by projecting the natural rows as well."""
from _common import *

P = problems(); rows = []; worst = 0.0
for key in ("bar_fx", "beam_ss", "beam_cf"):          # the 1-D benchmarks with natural conditions (V_s strictly smaller than V_w)
    p = P[key]
    for n in (6, 8, 10, 12, 14, 16):
        B = make_basis("bernstein", p.L, n); K, M, Zw, Zs = p.ritz_matrices(B, ng=max(40, n + 10)); ex = p.exact(p.nmodes) ** 2
        lw, _, _ = solve_pencil(Zw.T @ K @ Zw, Zw.T @ M @ Zw); ls, _, _ = solve_pencil(Zs.T @ K @ Zs, Zs.T @ M @ Zs)
        lw = np.sort(lw.real[np.isfinite(lw)]); ls = np.sort(ls.real[np.isfinite(ls)])
        lw = lw[lw > 1e-6 * ex[0]][:p.nmodes]; ls = ls[ls > 1e-6 * ex[0]][:p.nmodes]
        for k in range(min(len(lw), len(ls), p.nmodes)):
            v1 = (ex[k] - lw[k]) / ex[k]; v2 = (lw[k] - ls[k]) / ex[k]        # positive = violation
            worst = max(worst, v1, v2)
            rows.append(dict(case=key, n=n, mode=k + 1, exact=ex[k], weak=lw[k], strong_space=ls[k], viol_exact_weak=v1, viol_weak_strong=v2))
# Kirchhoff plates with natural / corner conditions: weak-form space V_w contains the strong-form space V_s
from pielm.kirchhoff import Plate
for code in ("SSFF", "FFFF", "CFFF"):
    P_ = Plate(code)
    for n in (6, 8, 10, 12):
        K, M, Zw, Zs = P_.ritz_matrices(n, ng=n + 12)
        lw, _, _ = solve_pencil(Zw.T @ K @ Zw, Zw.T @ M @ Zw); ls, _, _ = solve_pencil(Zs.T @ K @ Zs, Zs.T @ M @ Zs)
        lw = np.sort(lw.real[np.isfinite(lw)]); ls = np.sort(ls.real[np.isfinite(ls)])
        lw = lw[lw > 1.0][:6]; ls = ls[ls > 1.0][:6]
        for k in range(min(len(lw), len(ls))):
            v2 = (lw[k] - ls[k]) / lw[k]; worst = max(worst, v2)
            rows.append(dict(case="plate_" + code, n=n, mode=k + 1, exact=np.nan, weak=lw[k], strong_space=ls[k], viol_exact_weak=np.nan, viol_weak_strong=v2))
df = pd.DataFrame(rows); save_csv(df, "ordering_check")
print(f"modes checked: {len(df)};  largest relative ordering violation: {worst:.2e}  (round-off level expected)")
