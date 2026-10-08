"""Appendix C reference checks.
(a) 60-digit Bernstein-Ritz computation (degree n = 23) for the five standard 1-D benchmarks, in exact rational arithmetic for K and M and 60-digit
    mpmath for the eigenproblem.  The discretisation error of the first five modes (against closed-form / 60-digit characteristic-equation roots) is
    reported; anything in the double-precision tables below it is round-off.  Essential conditions are exact coefficient constraints of the Bernstein form
    (c_0 = 0, c_1 = c_0, c_n = 0, c_{n-1} = c_n).
(b) Mass-free limit of the 70-digit determinant used for the concentrated-mass beams: with m = k = 0 it must reproduce the classical beam frequencies.
(c) Leissa's tabulated free-plate values (nu = 0.3, Omega = omega a^2 sqrt(rho h / D)): 13.468, 19.596, 24.270, 34.801, 34.801, 61.093 against the weak form (n = 30).
Output: results/HP_bernstein_ritz.csv, results/HP_checks_summary.json."""
from _common import *
from fractions import Fraction
from math import comb, factorial
import mpmath as mp
from pielm.pointmass import exact_freqs
from pielm.kirchhoff import Plate
P = problems()

# ---- (b) mass-free limit of the 70-digit determinant (done first: exact_freqs sets mp.dps = 70)
mf = {}
for key, (a, b) in {"beam_ss": ("S", "S"), "beam_cf": ("C", "F"), "beam_ff": ("C", "C")}.items():
    p = P[key]; ex = exact_freqs(a, b, p.L, nmodes=5); ref = p.exact(5)
    mf[key] = float(np.max(np.abs(ex / ref - 1)))
print("mass-free limit, max relative difference to classical beam frequencies:", mf, flush=True)

# ---- (a) 60-digit Bernstein-Ritz
mp.mp.dps = 60; N = 23
def gram(p):                                   # exact Bernstein Gram matrix of degree p on [0,1]
    return [[Fraction(comb(p, a) * comb(p, b) * factorial(a + b) * factorial(2 * p - a - b), factorial(2 * p + 1)) for b in range(p + 1)] for a in range(p + 1)]
def mpm(F): return mp.matrix([[mp.mpf(x.numerator) / x.denominator for x in row] for row in F])
def diff_matrix(n, m):                          # coefficients of u^(m) in the degree n-m Bernstein basis: n!/(n-m)! * Delta^m c
    D = [[Fraction(0)] * (n + 1) for _ in range(n - m + 1)]
    for k in range(n - m + 1):
        for j in range(m + 1): D[k][k + j] = Fraction(factorial(n), factorial(n - m)) * (-1) ** (m - j) * comb(m, j)
    return D
def matmul(A, B): return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
def transpose(A): return [list(r) for r in zip(*A)]
CONS = {"bar_ff": ([0, N], 1), "bar_fx": ([0], 1), "beam_ss": ([0, N], 2), "beam_ff": ([0, 1, N - 1, N], 2), "beam_cf": ([0, 1], 2)}   # fixed coefficients, derivative order m
def exact_dimensionless(key, k=5):
    if key == "bar_ff": return [(mp.pi * j) ** 2 for j in range(1, k + 1)]
    if key == "bar_fx": return [(mp.pi * (2 * j - 1) / 2) ** 2 for j in range(1, k + 1)]
    if key == "beam_ss": return [(mp.pi * j) ** 4 for j in range(1, k + 1)]
    g = {"beam_ff": ([4.73004074486, 7.85320462409, 10.9956078380, 14.1371654912, 17.2787596574], lambda z: mp.cos(z) * mp.cosh(z) - 1),
         "beam_cf": ([1.87510406871, 4.69409113297, 7.85475743823, 10.9955407349, 14.1371683910], lambda z: mp.cos(z) * mp.cosh(z) + 1)}[key]
    return [mp.findroot(g[1], mp.mpf(z0)) ** 4 for z0 in g[0][:k]]
rows = []
for key, (fixed, m) in CONS.items():
    D = diff_matrix(N, m); Kf = matmul(transpose(D), matmul(gram(N - m), D)); Mf = gram(N)
    keep = [i for i in range(N + 1) if i not in fixed]
    K = mpm([[Kf[i][j] for j in keep] for i in keep]); M = mpm([[Mf[i][j] for j in keep] for i in keep])
    L = mp.cholesky(M); Li = L ** -1; A = Li * K * Li.T; A = (A + A.T) / 2
    E, _ = mp.eigsy(A); lam = sorted(E)[:5]; ex = exact_dimensionless(key)
    for j in range(5):
        rows.append(dict(case=key, mode=j + 1, err_pct=float(100 * (mp.sqrt(lam[j] / ex[j]) - 1)), err_pct_abs=float(100 * abs(mp.sqrt(lam[j] / ex[j]) - 1))))
    print(key, "max |err| %.2e %%" % max(r["err_pct_abs"] for r in rows if r["case"] == key), flush=True)
hp = pd.DataFrame(rows); save_csv(hp, "HP_bernstein_ritz")
worst = hp.loc[hp.err_pct_abs.idxmax()]

# ---- (c) Leissa
leissa = np.array([13.468, 19.596, 24.270, 34.801, 34.801, 61.093]); om = Plate("FFFF").weak(30, 50, rcond=1e-12, symmetric=True).omega
S = dict(mass_free_limit_max_rel_diff=mf, hp_ritz_max_err_pct=float(hp.err_pct_abs.max()), hp_ritz_worst_case=dict(case=str(worst.case), mode=int(worst["mode"]), err_pct=float(worst.err_pct_abs)),
         hp_ritz_max_err_pct_by_case={c: float(d.err_pct_abs.max()) for c, d in hp.groupby("case")},
         leissa_free_plate=dict(weak_n30=[float(v) for v in om], leissa=leissa.tolist(), max_abs_diff=float(np.max(np.abs(om - leissa))), max_rel_diff_percent=float(100 * np.max(np.abs(om / leissa - 1)))))
save_json(S, "HP_checks_summary"); print(json.dumps(S, indent=1))
