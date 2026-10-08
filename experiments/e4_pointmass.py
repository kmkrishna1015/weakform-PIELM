"""Beams with concentrated masses / springs (eigenvalue-dependent interface conditions).
weak   : one solve (rank-one updates of K and M).
strong : published Eig-PIELM on each element (uniform points), constraint frozen at lambda*, Brent root search per mode."""
from _common import *
from pielm.pointmass import *

N_STRONG = 5000          # uniform interior points per element for the strong form
CASES = {"H1": ("Cantilever + tip mass ($m=\\rho AL$)", lambda: PointMassBeam("C", "F", None, 0, 0, m_tip=1.0 * rA * Lb)),
         "H2": ("Simply supp. + mass at $0.3L$ ($m=\\rho AL$)", lambda: PointMassBeam("S", "S", 0.3 * Lb, 1.0 * rA * Lb)),
         "H3": ("Cantilever + heavy mass at $0.6L$ ($m=2\\rho AL$)", lambda: PointMassBeam("C", "F", 0.6 * Lb, 2.0 * rA * Lb)),
         "H4": ("Cantilever + mass $\\rho AL$ and spring $100EI/L^3$ at $0.4L$", lambda: PointMassBeam("C", "F", 0.4 * Lb, 1.0 * rA * Lb, k=100 * EI / Lb ** 3))}
R = {}

# (a) headline table: n = 23 per element
tab = {}
for key, (title, f) in CASES.items():
    b = f(); ex = b.exact(); row = dict(title=title, exact=ex.tolist())
    for name, fn, reps in (("weak", lambda: b.weak(23), 15), ("strong_brent", lambda: b.strong(23, N=N_STRONG, mode="brent"), 5),
                           ("strong_onepass", lambda: b.strong(23, N=N_STRONG, mode="onepass"), 15)):
        r = fn(); e = np.abs(r.omega / ex - 1) * 100; t = median_time(fn, reps, warm=1)
        row[name] = dict(err_pct=e.tolist(), max_err=float(e.max()), solves=int(r.info["solves"]), per_mode=r.info.get("per_mode"),
                         time=t[0], q25=t[1], q75=t[2], omega=r.omega.tolist())
    tab[key] = row
    print(key, {n: (f"{row[n]['max_err']:.1e}", row[n]["solves"], f"{row[n]['time']*1e3:.1f}ms") for n in ("weak", "strong_brent", "strong_onepass")}, flush=True)
R["table"] = tab; R["N_strong_per_element"] = N_STRONG

# (b) degree sweep (weak two-element and single polynomial with the symmetric truncated solver, strong Brent two-element with the plain general solver)
rows = []
for key in CASES:
    if key == "H4": continue
    b = CASES[key][1](); ex = b.exact()
    for n in (4, 6, 8, 10, 12, 14, 16, 20, 24, 30, 36, 44):
        for name, fn in (("weak (2-element)" if b.x0 else "weak", lambda: b.weak(n, max(40, n + 2), rcond=1e-12, symmetric=True)),
                         ("weak (single polynomial)", lambda: b.weak_global(n, rcond=1e-12)),
                         ("strong (Brent, 2-element)", lambda: b.strong(n, N=N_STRONG))):
            if b.x0 is None and name.startswith("weak (single"): continue
            try:
                r = fn(); e = np.abs(r.omega / ex - 1) * 100 if len(r.omega) == 5 else np.full(5, np.nan)
            except Exception: e = np.full(5, np.nan)
            rows.append(dict(case=key, method=name, n=n, nphi=(n + 1) * (2 if b.x0 else 1), max_err=float(np.max(e)), err1=float(e[0])))
save_csv(pd.DataFrame(rows), "H_degree")

# (c) cost versus number of requested modes (H2)
rows = []; b = CASES["H2"][1]()
for nm in (1, 2, 5, 10, 15, 20):
    b.nmodes = nm; ex = exact_freqs("S", "S", Lb, 0.3 * Lb, 1.0 * rA * Lb, 0, 0, nm)
    fw = lambda: b.weak(23, 40); fs = lambda: b.strong(23, N=N_STRONG, mode="brent")
    tw = median_time(fw, 9, warm=1); rw = fw()
    try:                                   # the untruncated published solver can fail for the higher modes of a 46-dof space
        ts = median_time(fs, 3, warm=1); rs = fs()
        t_s, so, es = ts[0], rs.info["solves"], float(np.max(np.abs(rs.omega / ex - 1)) * 100)
    except Exception as exn:
        t_s, so, es = np.nan, -1, np.nan
    rows.append(dict(nmodes=nm, t_weak=tw[0], t_strong=t_s, solves_strong=so,
                     err_weak=float(np.max(np.abs(rw.omega / ex - 1)) * 100), err_strong=es))
    print("nmodes", nm, rows[-1], flush=True)
save_csv(pd.DataFrame(rows), "H_nmodes")

# (d) mass-ratio sweep (H2 geometry)
rows = []
for mu in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0):
    b = PointMassBeam("S", "S", 0.3 * Lb, mu * rA * Lb); ex = b.exact(); rw = b.weak(23)
    try: rs = b.strong(23, N=N_STRONG, mode="brent"); es, so = float(np.max(np.abs(rs.omega / ex - 1)) * 100), rs.info["solves"]
    except Exception: es, so = np.nan, -1
    rows.append(dict(mass_ratio=mu, err_weak=float(np.max(np.abs(rw.omega / ex - 1)) * 100), err_strong=es, solves_strong=so))
save_csv(pd.DataFrame(rows), "H_massratio")
save_json(R, "hard")
