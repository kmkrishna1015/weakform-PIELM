"""BFS (Bogner-Fox-Schmit, C1 bicubic Hermite) finite-element baseline for the free and cantilever Kirchhoff plates
(unit square, D = 1, rho*h = 1), sparse shift-invert eigensolver.  Errors are measured against the same references as e6."""
from _common import *
from pielm.fem_plate_sparse import solve_sparse

ref = load_json("P_refs"); ref = {k: np.array(v) for k, v in ref.items()}
rows = []
for n_el in (4, 8, 16, 24, 32, 44):
    om, dof, tms = solve_sparse(n_el, n_el, clamp="none", n_modes=9, sigma=1.0, reps=3); om6 = om[3:9]          # 3 rigid modes dropped
    rows.append(dict(case="FFFF", n_el=n_el, dof=dof, time_ms=tms, err6=float(np.abs(om6 / ref["FFFF"] - 1).max()), err1=float(abs(om6[0] / ref["FFFF"][0] - 1))))
    om, dof, tms = solve_sparse(n_el, n_el, clamp="x0", n_modes=6, sigma=1e-8, reps=3)
    rows.append(dict(case="CFFF", n_el=n_el, dof=dof, time_ms=tms, err6=float(np.abs(om / ref["CFFF"] - 1).max()), err1=float(abs(om[0] / ref["CFFF"][0] - 1))))
    print(rows[-2], rows[-1], flush=True)
save_csv(pd.DataFrame(rows), "P_fem")
