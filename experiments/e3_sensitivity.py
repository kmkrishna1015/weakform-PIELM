"""Sensitivity / robustness studies S1..S7.  Usage: python e3_sensitivity.py [S1 S2 ...]
Both formulations always use the same eigensolver; 'plain' = dense general solve, 'trunc' = identical rcond=1e-12 truncation."""
from _common import *

P = problems(); RC = 1e-12
merr = max_err_pct
def signed1(r, p):
    return float((r.omega[0] - p.exact(1)[0]) / p.exact(1)[0]) if len(r.omega) else np.nan

# ---- S1 degree convergence: form x basis x solver
def S1():
    rows = []
    deg1 = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 34, 38, 44]
    deg_h = [(n, n) for n in (4, 6, 8, 10, 12, 13, 14, 16, 18, 20)]
    deg_p = [(4, 3), (6, 4), (8, 6), (10, 7), (12, 8), (14, 10), (16, 11), (18, 13), (20, 14)]
    combos = (("strong", "bernstein", "plain"), ("strong", "bernstein", "trunc"), ("weak", "bernstein", "plain"),
              ("weak", "bernstein", "trunc"), ("weak", "legendre", "plain"), ("weak", "legendre", "trunc"))
    for key, degs in (("bar_fx", deg1), ("beam_cf", deg1), ("helmholtz", deg_h), ("plate", deg_p)):
        p = P[key]
        for n in degs:
            nphi = (n + 1) if p.dim == 1 else (n[0] + 1) * (n[1] + 1)
            ov = dict(ng=max(40, n + 2)) if p.dim == 1 else dict(ng=(max(20, n[0] + 4), max(20, n[1] + 4)))
            if key == "helmholtz": ov["N"] = (2000, 1200)
            if key == "plate": ov["N"] = (2000, 1200); ov["nb"] = (n[0] + 1, n[1] + 1)
            for form, kind, sol in combos:
                try:
                    r = run_method(p, form, kind=kind, n=n, rcond=(RC if sol == "trunc" else None), **ov)
                    rows.append(dict(case=key, method=f"{form}_{kind[:4]}_{sol}", deg=str(n), nphi=nphi, max_err_pct=merr(r, p),
                                     signed_rel_mode1=signed1(r, p), red_dim=r.info["red_dim"], kept=r.info["kept"]))
                except Exception:
                    rows.append(dict(case=key, method=f"{form}_{kind[:4]}_{sol}", deg=str(n), nphi=nphi, max_err_pct=np.nan,
                                     signed_rel_mode1=np.nan, red_dim=-1, kept=-1))
        print("S1", key, flush=True)
    save_csv(pd.DataFrame(rows), "S1_degree")

# ---- S2 number of sample points of the published strong form (weak form shown for reference)
def S2():
    rows = []
    for key in ("bar_fx", "beam_cf", "beam_ss"):
        p = P[key]
        for N in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10000, 20000, 50000]:
            r = run_method(p, "strong", N=N); rows.append(dict(case=key, method="strong", N=N, max_err_pct=merr(r, p), asym=r.info.get("asym", np.nan)))
    for key in ("helmholtz", "plate"):
        p = P[key]
        for k in [1, 2, 4, 8, 16, 32, 64]:
            N = (50 * k, 30 * k); r = run_method(p, "strong", N=N)
            rows.append(dict(case=key, method="strong", N=N[0] * N[1], max_err_pct=merr(r, p), asym=np.nan))
    save_csv(pd.DataFrame(rows), "S2_collocation")

# ---- S3 truncation threshold (both forms) + Gram conditioning
def S3():
    rows = []
    for key in ("beam_ff", "beam_cf", "bar_fx"):
        p = P[key]; B = bases(p)
        for rc in [1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4]:
            for form in METHODS:
                r = run_method(p, form, B=B, rcond=rc); rows.append(dict(case=key, method=form, rcond=rc, max_err_pct=merr(r, p), kept=r.info["kept"], red_dim=r.info["red_dim"]))
    save_csv(pd.DataFrame(rows), "S3_rcond")
    rows = []
    for n in (4, 8, 12, 16, 20, 24, 28, 32, 40):
        for kind in ("bernstein", "legendre"):
            b = make_basis(kind, 1.0, n); x, w = gl(0, 1.0, n + 5); Ph = b.d(x, 0); M = Ph.T @ (w[:, None] * Ph)
            rows.append(dict(deg=n, basis=kind, cond_gram=float(np.linalg.cond(M))))
    save_csv(pd.DataFrame(rows), "S3_cond")

# ---- S4 Gauss order of the weak form
def S4():
    rows = []
    for key in ("bar_fx", "beam_cf"):
        p = P[key]
        for ng in [4, 8, 12, 16, 18, 20, 22, 23, 24, 26, 30, 40, 60]:
            rows.append(dict(case=key, deg=23, ng=ng, max_err_pct=merr(run_method(p, "weak_sym", ng=ng), P[key])))
    p = P["helmholtz"]
    for g in [6, 8, 10, 12, 13, 14, 16, 20, 30]: rows.append(dict(case="helmholtz", deg=13, ng=g, max_err_pct=merr(run_method(p, "weak_sym", ng=(g, g)), p)))
    p = P["plate"]
    for g in [8, 12, 15, 16, 20, 24, 30]: rows.append(dict(case="plate", deg=14, ng=g, max_err_pct=merr(run_method(p, "weak_sym", ng=(g + 6, g)), p)))
    save_csv(pd.DataFrame(rows), "S4_gauss_order")

# ---- S5 timing scaling with basis size (analytic; autodiff if torch is available)
def S5():
    rows = []; kinds = ["bernstein"]
    try:
        import torch; kinds.append("ad")
    except ImportError:
        print("torch not installed: autodiff rows skipped")
    for key, degs in (("beam_cf", [8, 12, 16, 20, 24, 32, 40]), ("helmholtz", [(n, n) for n in (4, 6, 8, 10, 13, 16, 20, 24)])):
        p = P[key]
        for n in degs:
            nphi = (n + 1) if p.dim == 1 else (n[0] + 1) * (n[1] + 1)
            ov = dict(ng=max(40, n + 2)) if p.dim == 1 else dict(ng=(max(20, n[0] + 4),) * 2)
            for kind in kinds:
                B = bases(p, kind, n)
                for m in METHODS3:
                    if kind == "ad" and m == "strong" and p.dim == 1: continue
                    med, q1, q3 = median_time(lambda: run_method(p, m, B=B, n=n, **ov), 20 if kind == "bernstein" else 6)
                    rows.append(dict(case=key, nphi=nphi, method=m, deriv="analytic" if kind == "bernstein" else "autodiff", median=med, q25=q1, q75=q3))
        print("S5", key, flush=True)
    save_csv(pd.DataFrame(rows), "S5_timing_scaling")

# ---- S6 boundary-sampling density of the strong form
def S6():
    rows = []
    for key in ("helmholtz", "plate"):
        p = P[key]; n = p.default["n"]
        for f in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]:
            nb = (max(2, int(math.ceil(f * (n[0] + 1)))), max(2, int(math.ceil(f * (n[1] + 1)))))
            r = run_method(p, "strong", nb=nb); rows.append(dict(case=key, method="strong", nb_factor=f, nb=str(nb), max_err_pct=merr(r, p), red_dim=r.info["red_dim"]))
    save_csv(pd.DataFrame(rows), "S6_bc_sampling")

# ---- S7 unit-system invariance
def S7():
    rows = []
    systems = {"SI (m,kg,s)": dict(E=210e9, rho=7800.0, Ls=1.0, Lb=0.12, bb=0.003, hh=0.002),
               "mm-kgf-s": dict(E=2.1e4, rho=8.0e-10, Ls=1000.0, Lb=120.0, bb=3.0, hh=2.0),
               "mm-N-s (tonne)": dict(E=2.1e5, rho=7.8e-9, Ls=1000.0, Lb=120.0, bb=3.0, hh=2.0),
               "cm-g-s (dyn)": dict(E=2.1e12, rho=7.8, Ls=100.0, Lb=12.0, bb=0.3, hh=0.2)}
    for name, s in systems.items():
        E, rho = s["E"], s["rho"]; A = s["bb"] * s["hh"]; I = s["bb"] * s["hh"] ** 3 / 12; EI, rA = E * I, rho * A
        bcf = beam_roots("cf"); Lb = s["Lb"]
        probs = {"bar_fx": Problem1D("bar_fx", "", 1, -E / rho, E, rho, s["Ls"], [(0, 0)], [(s["Ls"], 1)],
                                     lambda n, E=E, rho=rho, Ls=s["Ls"]: (2 * np.arange(1, n + 1) - 1) * np.pi / (2 * Ls) * np.sqrt(E / rho), None),
                 "beam_cf": Problem1D("beam_cf", "", 2, EI / rA, EI, rA, Lb, [(0, 0), (0, 1)], [(Lb, 2), (Lb, 3)],
                                      lambda n, Lb=Lb, EI=EI, rA=rA: (bcf[:n] / Lb) ** 2 * np.sqrt(EI / rA), None)}
        for key, p in probs.items():
            for m in METHODS:
                r = run_method(p, m); rows.append(dict(units=name, case=key, method=m, max_err_pct=merr(r, p), omega1=r.omega[0]))
    save_csv(pd.DataFrame(rows), "S7_units")

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]):
        t = time.time(); globals()[w](); print(w, f"done in {time.time()-t:.1f}s", flush=True)
