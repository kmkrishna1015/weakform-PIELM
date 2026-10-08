"""Appendix C: independent finite-element check of the self-consistent plate references.
Bogner-Fox-Schmit (C1 bicubic Hermite) elements, unit square, D = 1, rho h = 1, nu = 0.3, five uniform meshes (16, 24, 32, 48, 64 elements per side),
sparse shift-invert eigensolver.  Per mode, Omega(h) = Omega_inf + c h^p is fitted (least squares over the five meshes, p free) and
the extrapolated Omega_inf is compared with the weak-form reference (Bernstein n = 38).  The observed order is also reported from the geometric triple
16-32-64 (p = log2[(O16-O32)/(O32-O64)]).  CFFF: the clamped-free corner singularity limits the FEM order (expected ~2).
Output: results/P_fem_richardson.csv, results/P_fem_richardson_summary.json.  Errors are relative (x100 for percent)."""
from _common import *
from scipy.optimize import least_squares
from pielm.fem_plate_sparse import solve_sparse
from pielm.kirchhoff import Plate

MESHES = (16, 24, 32, 48, 64)
ref = {c: np.asarray(Plate(c).weak(38, 60, rcond=1e-12, symmetric=True).omega) for c in ("FFFF", "CFFF")}
print({c: v.round(6).tolist() for c, v in ref.items()}, flush=True)
om = {c: [] for c in ref}
for n in MESHES:
    o, dof, _ = solve_sparse(n, n, clamp="none", n_modes=9, sigma=1.0, reps=1); om["FFFF"].append(o[3:9])                  # drop 3 rigid modes
    o, dof, _ = solve_sparse(n, n, clamp="x0", n_modes=6, sigma=1e-8, reps=1); om["CFFF"].append(np.sort(o))
    print("mesh", n, "dof", dof, flush=True)
rows = []; S = {}
h = 1.0 / np.array(MESHES, float)
for c in ref:
    A = np.array(om[c])                                     # (mesh, mode)
    orders, errs_ex, errs_fin = [], [], []
    for j in range(A.shape[1]):
        y = A[:, j]
        p3 = float(np.log2((y[0] - y[2]) / (y[2] - y[4])))                                       # meshes 16, 32, 64
        f = lambda q: q[0] + q[1] * h ** q[2] - y
        sol = least_squares(f, [y[-1], y[0] - y[-1], max(p3, 1.0)], x_scale=[1, 1, 1], xtol=1e-14, ftol=1e-14, gtol=1e-14)
        inf, _, pfit = sol.x
        e_ex = abs(inf / ref[c][j] - 1); e_fin = abs(y[-1] / ref[c][j] - 1)
        rows.append(dict(case=c, mode=j + 1, order_triple_16_32_64=p3, order_fit=float(pfit), omega_ref=float(ref[c][j]), omega_64=float(y[-1]), omega_extrap=float(inf), err_finest=float(e_fin), err_extrap=float(e_ex)))
        orders.append(p3); errs_ex.append(e_ex); errs_fin.append(e_fin)
    S[c] = dict(order_triple_min=float(min(orders)), order_triple_max=float(max(orders)), max_err_extrapolated=float(max(errs_ex)), max_err_finest_mesh=float(max(errs_fin)),
                err_finest_mode1=float(errs_fin[0]), max_err_extrapolated_percent=100 * float(max(errs_ex)))
save_csv(pd.DataFrame(rows), "P_fem_richardson"); save_json(S, "P_fem_richardson_summary"); print(json.dumps(S, indent=1))
