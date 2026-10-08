"""Appendix D: the weak form with six frozen bases (Bernstein, Legendre, Chebyshev, monomial, random tanh features, Gaussian RBF).

Identical weak-form solver for all bases:  K y = lambda M y,  Gauss-Legendre quadrature (NG points per direction),
essential conditions removed by a null-space projection, Loewdin-reduced symmetric eigensolver.

Protocol (the one stated in the caption of Table 'basis' of the paper):
  * truncation threshold rcond of the Gram (mass) matrix chosen a posteriori from {1e-10, ..., 1e-16} for EVERY basis ('best');
    'default' is the fixed value rcond = 1e-12;
  * polynomial bases (same polynomial space):      essential-condition projection first, then truncation of Z'MZ;
  * random-feature bases (tanh-ELM, Gaussian RBF): Gram matrix orthonormalised and truncated FIRST, the essential conditions are
    then imposed on the orthonormal coordinates (the reverse order; see the discussion in Appendix D);
  * tanh-ELM: 10 random seeds, hidden-layer scale R in {10, 30} (1-D) or {3, 10} (2-D, genuine 2-D hidden layer with N_phi features); entry = median over seeds of the best of the two scales;
    Gaussian RBF: deterministic, overlap factor in {2, 3}; entry = best of the two;  'lo'/'hi' = 25/75 percentiles over seeds;
    'default' = default scale (R = 20 in 1-D, 5 in 2-D; overlap 2) at rcond = 1e-12.
  * 'cond' = 2-norm condition number of the reduced mass matrix Z'MZ (polynomial bases) or of the full Gram matrix (random bases).
Output: results/B_basis_1d.csv, results/B_basis_2d.csv, results/B_basis_summary.json (numbers quoted in the text of Appendix D).
Error = maximum relative eigenfrequency error (%) over the five lowest modes.
"""
from _common import *
from pielm.bases import make_extra

P = problems(); NG = 70
RCS = [1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16]
SEEDS = range(10)
SCALES = {"tanh-ELM": ("R", (10.0, 30.0), 20.0), "gaussian-RBF": ("ovl", (2.0, 3.0), 2.0)}      # name, the two scales, default scale
POLY = ("bernstein", "legendre", "chebyshev", "monomial"); RAND = ("tanh-ELM", "gaussian-RBF")
SCALES_2D_TANH = ("R", (3.0, 10.0), 5.0)       # the 2-D hidden layer needs smaller weights: two scales per dimension, default scale 5
KINDS = POLY + RAND
NPHI_1D = (6, 8, 12, 16, 20, 24, 32, 40, 60)
NPHI_2D = (4, 6, 8, 10, 12, 14, 16, 20)                       # per direction -> 16, 36, ..., 196, 256, 400 functions


def cond(A):
    s = np.linalg.svd(0.5 * (A + A.T), compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")


def err_of(om, ex):
    return float(np.max(rel_err_pct(om, ex))) if om is not None and len(om) == len(ex) else np.nan


def ortho_first(p, b, rc):
    """Random-feature ordering: truncate/orthonormalise the Gram matrix first, then impose the essential conditions."""
    x, w = gl(0, p.L, NG); Phi = b.d(x, 0); Dm = b.d(x, p.m)
    K = p.kc * (Dm.T @ (w[:, None] * Dm)); M = p.mc * (Phi.T @ (w[:, None] * Phi))
    eb, Vb = np.linalg.eigh(0.5 * (M + M.T)); keep = eb > rc * eb[-1]
    if not keep.any(): return None, cond(M)
    T = Vb[:, keep] / np.sqrt(eb[keep]); Kt = T.T @ K @ T
    Zt, _ = null_space_svd(np.vstack(p._bcrows(b, p.ess)) @ T)
    Kz = Zt.T @ Kt @ Zt
    idx, om = lowest(np.linalg.eigvalsh(0.5 * (Kz + Kz.T)), p.nmodes, p._floor())
    return om, cond(M)


def poly_err(p, b, rc, ex):
    return err_of(p.weak(b, NG, rcond=rc, symmetric=True).omega, ex)


def best_over_rc(fn):
    e = np.array([fn(rc) for rc in RCS], float)
    return float(np.nanmin(e)) if np.isfinite(e).any() else np.nan


rows = []


class RandTanh2D:
    """phi_j(x,y) = tanh( w_j . (xi - c_j) ),  xi = (x/Lx, y/Ly),  w_j ~ U(-R,R)^2, c_j ~ U(0,1)^2  (a genuine 2-D ELM hidden layer)."""
    def __init__(self, Lx, Ly, nphi, R, seed):
        r = np.random.default_rng(seed); self.w = r.uniform(-R, R, (nphi, 2)); self.c = r.uniform(0, 1, (nphi, 2)); self.L = (Lx, Ly)
    def eval(self, X, Y):
        z = self.w[:, 0] * (X[:, None] / self.L[0] - self.c[:, 0]) + self.w[:, 1] * (Y[:, None] / self.L[1] - self.c[:, 1])
        t = np.tanh(z); u = 1 - t * t
        return t, u * self.w[:, 0] / self.L[0], u * self.w[:, 1] / self.L[1]


def helm_tanh2d_err(H, exh, nphi, R, seed, rc, NQ=60):
    """Weak form of the Neumann cavity with the 2-D random-feature basis (no essential rows; Gram truncation only)."""
    (x, wx), (y, wy) = gl(0, H.Lx, NQ), gl(0, H.Ly, NQ)
    X, Y = np.meshgrid(x, y, indexing="ij"); W = np.outer(wx, wy).ravel()
    b = RandTanh2D(H.Lx, H.Ly, nphi, R, seed); f, fx, fy = b.eval(X.ravel(), Y.ravel())
    K = fx.T @ (W[:, None] * fx) + fy.T @ (W[:, None] * fy); M = f.T @ (W[:, None] * f)
    lam, V, kept = solve_pencil(K, M, rc, True)
    idx, om = lowest(lam, H.nmodes, H._floor())
    return err_of(H.c * om if len(idx) else om, exh)


# ------------------------------------------------------------------------------------------------ 1-D
for case in ("bar_fx", "beam_ss", "beam_cf"):
    p = P[case]; ex = p.exact(p.nmodes)
    for kind in KINDS:
        for nphi in NPHI_1D:
            if kind in POLY:
                b = make_extra(kind, p.L, nphi)
                best = best_over_rc(lambda rc: poly_err(p, b, rc, ex)); dflt = poly_err(p, b, 1e-12, ex)
                Z, _ = null_space_svd(np.vstack(p._bcrows(b, p.ess)))
                x, w = gl(0, p.L, NG); Phi = b.d(x, 0); k_ = cond(Z.T @ (p.mc * (Phi.T @ (w[:, None] * Phi))) @ Z)
                lo = hi = best
            else:
                key, scales, sdef = SCALES[kind]; per_seed = []; dd = []
                for sd in (SEEDS if kind == "tanh-ELM" else [0]):
                    cand = []
                    for sc in scales:
                        b = make_extra(kind, p.L, nphi, seed=sd, **{key: sc})
                        cand.append(best_over_rc(lambda rc: err_of(ortho_first(p, b, rc)[0], ex)))
                    per_seed.append(np.nanmin(cand) if np.isfinite(cand).any() else np.nan)
                    b = make_extra(kind, p.L, nphi, seed=sd, **{key: sdef}); om, k_ = ortho_first(p, b, 1e-12); dd.append(err_of(om, ex))
                best = float(np.nanmedian(per_seed)); lo, hi = (float(v) for v in np.nanpercentile(per_seed, (25, 75))); dflt = float(np.nanmedian(dd))
            bmin = float(np.nanmin(per_seed)) if kind in RAND and np.isfinite(per_seed).any() else best
            rows.append(dict(dim=1, case=case, basis=kind, method="weak", nphi=nphi, best=best, lo=lo, hi=hi, default=dflt, cond=k_, best_single_seed=bmin))
    print("1-D done:", case, flush=True)
save_csv(pd.DataFrame(rows), "B_basis_1d")

# ------------------------------------------------------------------------------------------------ 2-D Helmholtz cavity (Neumann: no essential rows)
H = Helmholtz2D(); exh = H.exact(H.nmodes); rows2 = []
def h_err(bx, by, rc): return err_of(H.weak(bx, by, ng=(20, 20), rcond=rc, symmetric=True).omega, exh)
for kind in KINDS:
    for nx in NPHI_2D:
        if kind in POLY:
            bx, by = make_extra(kind, H.Lx, nx), make_extra(kind, H.Ly, nx)
            best = best_over_rc(lambda rc: h_err(bx, by, rc)); dflt = h_err(bx, by, 1e-12); lo = hi = best
        elif kind == "tanh-ELM":                       # 2-D random hidden layer with nx*nx features (same number of functions as the tensor bases)
            key, scales, sdef = SCALES_2D_TANH; per_seed = []; dd = []
            for sd in SEEDS:
                cand = [best_over_rc(lambda rc: helm_tanh2d_err(H, exh, nx * nx, sc, sd, rc)) for sc in scales]
                per_seed.append(np.nanmin(cand) if np.isfinite(cand).any() else np.nan)
                dd.append(helm_tanh2d_err(H, exh, nx * nx, sdef, sd, 1e-12))
        else:                                          # Gaussian RBF: tensor product of 1-D Gaussians
            key, scales, sdef = SCALES[kind]; per_seed = []; dd = []
            for sd in [0]:
                cand = []
                for sc in scales:
                    bx, by = make_extra(kind, H.Lx, nx, **{key: sc}), make_extra(kind, H.Ly, nx, **{key: sc})
                    cand.append(best_over_rc(lambda rc: h_err(bx, by, rc)))
                per_seed.append(np.nanmin(cand) if np.isfinite(cand).any() else np.nan)
                bx, by = make_extra(kind, H.Lx, nx, **{key: sdef}), make_extra(kind, H.Ly, nx, **{key: sdef}); dd.append(h_err(bx, by, 1e-12))
        if kind in RAND:
            best = float(np.nanmedian(per_seed)); lo, hi = (float(v) for v in np.nanpercentile(per_seed, (25, 75))); dflt = float(np.nanmedian(dd))
            bmin = float(np.nanmin(per_seed))
        else:
            bmin = best
        rows2.append(dict(dim=2, case="helmholtz", basis=kind, method="weak", nphi=nx * nx, best=best, lo=lo, hi=hi, default=dflt, cond=np.nan, best_single_seed=bmin))
    print("2-D done:", kind, flush=True)
save_csv(pd.DataFrame(rows2), "B_basis_2d")

# ------------------------------------------------------------------------------------------------ numbers quoted in Appendix D
d1, d2 = pd.DataFrame(rows), pd.DataFrame(rows2); S = {"table_N24_N196": {}}
for kind in KINDS:
    S["table_N24_N196"][kind] = dict(
        bar_fx=float(d1[(d1.case == "bar_fx") & (d1.basis == kind) & (d1.nphi == 24)].best.iloc[0]),
        beam_cf=float(d1[(d1.case == "beam_cf") & (d1.basis == kind) & (d1.nphi == 24)].best.iloc[0]),
        helmholtz=float(d2[(d2.basis == kind) & (d2.nphi == 196)].best.iloc[0]))
t = d1[(d1.basis == "tanh-ELM") & (d1.nphi <= 60)]
S["best_tanh_nphi_le_60_single_seed"] = {c: float(t[t.case == c].best_single_seed.min()) for c in ("bar_fx", "beam_ss", "beam_cf")}
S["best_tanh_nphi_le_60_median_seed"] = {c: float(t[t.case == c].best.min()) for c in ("bar_fx", "beam_ss", "beam_cf")}
S["tanh_seed_spread_N24"] = {c: [float(d1[(d1.case == c) & (d1.basis == "tanh-ELM") & (d1.nphi == 24)][k].iloc[0]) for k in ("lo", "hi")] for c in ("bar_fx", "beam_cf")}
g = d1[(d1.nphi == 24) & d1.basis.isin(("legendre", "chebyshev"))]
S["gram_cond_legendre_chebyshev_N24"] = dict(min=float(g.cond.min()), max=float(g.cond.max()))
S["gram_cond_monomial_N24"] = {c: float(d1[(d1.case == c) & (d1.basis == "monomial") & (d1.nphi == 24)].cond.iloc[0]) for c in ("bar_fx", "beam_ss", "beam_cf")}
S["ratio_leg_cheb_over_bern_beam_cf_N24"] = {k: S["table_N24_N196"][k]["beam_cf"] / S["table_N24_N196"]["bernstein"]["beam_cf"] for k in ("legendre", "chebyshev")}
save_json(S, "B_basis_summary"); print(json.dumps(S, indent=1))
