"""Accuracy (eigenvalues + mode shapes) of the two formulations on the 7 standard cases, and the Q4-FEM baseline."""
from _common import *

P = problems(); out = {}

def shapes(p, r):
    B = bases(p); errs = []
    for j in range(len(r.omega)):
        if p.dim == 1:
            x = np.linspace(0, p.L, 401); errs.append(shape_err(p.eval_shape(B, r.beta[:, j], x), p.shape_fn(j, x)))
        elif p.key == "helmholtz":
            X, Y = np.linspace(0, p.Lx, 81), np.linspace(0, p.Ly, 49)
            errs.append(shape_err(p.eval_shape(*B, r.beta[:, j], X, Y), p.exact_shape(j, X, Y)))
        else:
            X, Y = np.linspace(0, p.Lx, 61), np.linspace(0, p.Ly, 37)
            ux, uy = p.eval_field(*B, r.beta[:, j], X, Y); ex, ey = p.exact_shape(j, X, Y)
            errs.append(shape_err(np.concatenate([ux.ravel(), uy.ravel()]), np.concatenate([ex.ravel(), ey.ravel()])))
    return errs

for key in ORDER:
    p = P[key]; ex = p.exact(p.nmodes); out[key] = dict(title=p.title, exact=ex.tolist(), methods={})
    for m in METHODS3:
        r = run_method(p, m, vectors=True); e = rel_err_pct(r.omega, ex)
        out[key]["methods"][m] = dict(omega=r.omega.tolist(), err_pct=e.tolist(), max_err_pct=float(np.max(e)) if len(e) == p.nmodes else None,
                                      shape_err=shapes(p, r), red_dim=int(r.info["red_dim"]), kept=int(r.info["kept"]))
    print(key, {m: f"{out[key]['methods'][m]['max_err_pct']:.2e}" for m in METHODS3}, flush=True)

h = P["helmholtz"]; ex = h.exact(5); fem = {}
for label, nn in (("Q4 14x14 nodes (196 dof)", 14), ("Q4 100x100 nodes (1e4 dof)", 100), ("Q4 200x200 nodes (4e4 dof)", 200)):
    ts = []
    for _ in range(3 if nn < 200 else 1):
        t = time.perf_counter(); om = fem_helmholtz(h.Lx, h.Ly, h.c, nn, nn); ts.append(time.perf_counter() - t)
    fem[label] = dict(omega=om.tolist(), err_abs=np.abs(om - ex).tolist(), max_err_pct=float(np.max(rel_err_pct(om, ex))), time=float(np.median(ts)), ndof=nn * nn)
    print("FEM", label, f"{fem[label]['max_err_pct']:.2e}%  {fem[label]['time']:.3f}s")
out["fem_helmholtz"] = fem
save_json(out, "benchmarks")
