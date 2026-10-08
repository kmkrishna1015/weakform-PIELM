"""Section 'Free-edge Kirchhoff plates': pointwise natural-condition residuals of the weak-form modes.
For every free edge, the pointwise bending-moment residual M_n = w_nn + nu w_tt and effective-shear residual V_n = w_nnn + (2-nu) w_ntt
is evaluated along the edge and divided by the maximum of the same quantity over the whole plate (161 x 161 grid).  The weak form
(Bernstein basis of degree n, Gauss quadrature 40 pts, Loewdin truncation rcond = 1e-12, symmetric eigensolver) satisfies these
conditions only variationally, so the residuals do not vanish.  Reported: worst case over the six modes and over all free edges, for
n = 8, 16, 24; plus the per-mode values at n = 16.  Output: results/P_residuals.csv, results/P_residuals_summary.json."""
from _common import *
from pielm.kirchhoff import Plate, NU

FREE = {"FFFF": ("x0", "x1", "y0", "y1"), "CFFF": ("x1", "y0", "y1")}
g = np.linspace(0.0, 1.0, 161)
rows = []
for code in ("FFFF", "CFFF"):
    for n in (8, 12, 16, 20, 24):
        r = Plate(code).weak(n, 40, rcond=1e-12, vectors=True, symmetric=True); b = Bern1D(1.0, n); B = [b.d(g, k) for k in range(4)]
        ev = lambda bt, kx, ky: np.einsum("pi,qj,ij->pq", B[kx], B[ky], bt.reshape(n + 1, n + 1))
        for j in range(len(r.omega)):
            bt = r.beta[:, j]
            Mx = ev(bt, 2, 0) + NU * ev(bt, 0, 2); Vx = ev(bt, 3, 0) + (2 - NU) * ev(bt, 1, 2)     # normal = x
            My = ev(bt, 0, 2) + NU * ev(bt, 2, 0); Vy = ev(bt, 0, 3) + (2 - NU) * ev(bt, 2, 1)     # normal = y
            for edge in FREE[code]:
                M, V = (Mx, Vx) if edge[0] == "x" else (My, Vy); sl = (0 if edge[1] == "0" else -1)
                em = (np.abs(M[sl, :]) if edge[0] == "x" else np.abs(M[:, sl])).max() / np.abs(M).max()
                ev_ = (np.abs(V[sl, :]) if edge[0] == "x" else np.abs(V[:, sl])).max() / np.abs(V).max()
                rows.append(dict(case=code, n=n, mode=j + 1, omega=float(r.omega[j]), edge=edge, M_res=float(em), V_res=float(ev_)))
    print("done", code, flush=True)
df = pd.DataFrame(rows); save_csv(df, "P_residuals")
S = {}
for code in ("FFFF", "CFFF"):
    S[code] = {int(n): dict(M_worst=float(d.M_res.max()), V_worst=float(d.V_res.max()), M_mode1=float(d[d["mode"] == 1].M_res.max()), V_mode1=float(d[d["mode"] == 1].V_res.max()))
               for n, d in df[df.case == code].groupby("n")}
save_json(S, "P_residuals_summary"); print(json.dumps(S, indent=1))
