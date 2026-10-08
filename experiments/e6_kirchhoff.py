"""Kirchhoff plates: SSSS (exact), SFSF (Levy, exact), FFFF and CFFF (self-converged Ritz references), weak vs published strong form.
Also: independent Young beam-function Ritz check of the cantilever plate, and fairness checks of the strong form."""
from _common import *
from scipy.optimize import brentq
from pielm.kirchhoff import Plate, ssss_exact
from pielm import levy

CASES = {"SSSS": "SSSS", "SFSF": "SSFF", "FFFF": "FFFF", "CFFF": "CFFF"}
RC = 1e-12

def references():
    ref = {"SSSS": ssss_exact(), "SFSF": np.array([a for a, m in levy.levy_sfsf(6)])}
    for k in ("FFFF", "CFFF"):      # weak-form Ritz at two high degrees, truncated (reference computation only)
        a = Plate(CASES[k]).weak(26, 40, rcond=RC).omega; b = Plate(CASES[k]).weak(30, 44, rcond=RC).omega
        ref[k] = b; print(k, "reference n=26 vs 30 agree to", np.abs(a / b - 1).max(), b, flush=True)
    save_json({k: v.tolist() for k, v in ref.items()}, "P_refs"); return ref

def sweep(ref):
    rows = []
    for key, code in CASES.items():
        P = Plate(code); ex = ref[key]
        for n in (6, 8, 10, 12, 14, 16, 20, 24):
            ng = max(30, n + 10)
            for form in ("weak", "strong"):
                for sol, rc in (("", None), ("_trunc", RC)):
                    fn = (lambda: P.weak(n, ng, rcond=rc)) if form == "weak" else (lambda: P.strong(n, 300, rcond=rc))
                    try:
                        r = fn(); e = float(np.abs(r.omega / ex - 1).max()) if len(r.omega) == 6 else np.nan; info = r.info
                        t = np.median([fn().info["time"] for _ in range(3)])
                    except Exception: e, info, t = np.nan, dict(red=-1, rank=-1, kept=-1), np.nan
                    rows.append(dict(case=key, n=n, nphi=(n + 1) ** 2, method=form + sol, err=e, red=info["red"], rank=info["rank"], time=float(t)))
        print(key, "done", flush=True)
    save_csv(pd.DataFrame(rows), "P_plate")

class BeamFun:
    """Clamped-free (cf) or free-free (ff, incl. rigid modes) beam eigenfunctions as a Ritz basis (Young 1950)."""
    def __init__(self, kind, nf):
        self.L, self.kind, self.nphi = 1.0, kind, nf
        cf = lambda z: np.cos(z) * np.cosh(z) + (1 if kind == "cf" else -1)
        self.b = []; z = 0.5
        while len(self.b) < (nf if kind == "cf" else nf - 2):
            zz = np.linspace(z, z + 0.05, 2)
            if cf(zz[0]) * cf(zz[1]) < 0: self.b.append(brentq(cf, zz[0], zz[1], xtol=1e-15))
            z += 0.05
        self.b = np.array(self.b)
        self.sig = ((np.cosh(self.b) + np.cos(self.b)) / (np.sinh(self.b) + np.sin(self.b)) if kind == "cf"
                    else (np.cosh(self.b) - np.cos(self.b)) / (np.sinh(self.b) - np.sin(self.b)))
    def d(self, x, k=0):
        x = np.atleast_1d(np.asarray(x, float))[:, None]; b = self.b[None, :]; z = b * x; ph = k * np.pi / 2
        ch = b ** k * (np.cosh(z) if k % 2 == 0 else np.sinh(z)); sh = b ** k * (np.sinh(z) if k % 2 == 0 else np.cosh(z))
        co = b ** k * np.cos(z + ph); si = b ** k * np.sin(z + ph)
        F = ch - co - self.sig * (sh - si) if self.kind == "cf" else ch + co - self.sig * (sh + si)
        if self.kind == "cf": return F
        r = np.zeros((x.shape[0], 2))
        if k == 0: r[:, 0] = 1; r[:, 1] = np.sqrt(3) * (1 - 2 * x[:, 0])
        if k == 1: r[:, 1] = -2 * np.sqrt(3)
        return np.hstack([r, F])

def young_check(ref):
    out = {}
    for nf in (6, 8, 10, 12):
        bx, by = BeamFun("cf", nf), BeamFun("ff", nf + 2)
        r = Plate("CFFF").weak(bx=bx, by=by, ng=60, admissible=True, rcond=RC); out[f"nf={nf}"] = r.omega[:4].tolist()
        print("Young Ritz nf =", nf, "->", r.omega[:4])
    out["bernstein_n20"] = Plate("CFFF").weak(20, rcond=RC).omega[:4].tolist(); print("Bernstein weak n=20:", out["bernstein_n20"])
    save_json(out, "P_young_check")

def fairness(ref):
    """Published strong form: (i) boundary oversampling nb = (n+1), 3(n+1), 6(n+1); (ii) removal of the free-free corner rows."""
    out = {}
    for key in ("FFFF", "CFFF"):
        P = Plate(key); ex = ref[key]
        for n in (16, 24):
            base = P.strong(n, 300).omega; d = {"baseline_err": float(np.abs(base / ex - 1).max())}
            for f in (3, 6):
                o = P.strong(n, 300, nb=f * (n + 1)).omega
                d[f"nb={f}(n+1)"] = dict(dev_vs_baseline=float(np.abs(o / base - 1).max()), err=float(np.abs(o / ex - 1).max()))
            try:
                nc = P.strong(n, 300, corners=False).omega; d["no_corner_rows_err"] = float(np.abs(nc / ex - 1).max()) if len(nc) == 6 else None
            except Exception as e: d["no_corner_rows_err"] = None
            out[f"{key} n={n}"] = d; print(key, n, d, flush=True)
    save_json(out, "P_fairness")

if __name__ == "__main__":
    ref = references(); sweep(ref); young_check(ref); fairness(ref)
