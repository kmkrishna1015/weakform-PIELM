"""Additional validation checks for the numerical study.

The script evaluates:
(T1) the dimensions of the strong- and weak-form reduced spaces,
(T2) the strong form with exact Gauss weights on the seven standard
     benchmarks,
(T3) reference-solution convergence for the fully free and cantilever
     Kirchhoff plates, and
(T4) the bordered one-shot strong formulation for the concentrated-mass
     beam problems.
"""

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

from _common import *
from scipy.linalg import eig

from pielm import strong_exact as SE
from pielm.pointmass import PointMassBeam, rA, Lb, EI
from pielm.kirchhoff import Plate


P = problems()

R = {
    "space_dimensions": {},
    "exact_weight_strong_form": {},
    "bordered_strong_form": {},
    "plate_reference_convergence": {},
}


# ---------------------------------------------------------------------------
# Reduced-space dimensions and exact-weight strong-form comparison
# ---------------------------------------------------------------------------

for key in ORDER:
    p = P[key]
    ex = p.exact(p.nmodes)
    B = bases(p)

    rw = run_method(p, "weak_sym")
    rs = run_method(p, "strong")

    if p.dim == 1:
        re_ = SE.strong1d(p, B)
    elif key == "helmholtz":
        re_ = SE.helmholtz(p, *B)
    else:
        re_ = SE.plate2d(p, *B)

    def max_rel_error_percent(result):
        if len(result.omega) == p.nmodes:
            return float(np.max(rel_err_pct(result.omega, ex)))
        return float("nan")

    R["space_dimensions"][key] = {
        "dim_Vw": int(rw.info["red_dim"]),
        "dim_Vs": int(rs.info["red_dim"]),
        "nphi": int(
            B.nphi
            if p.dim == 1
            else B[0].nphi * B[1].nphi
            * (2 if key == "plate" else 1)
        ),
    }

    R["exact_weight_strong_form"][key] = {
        "weak": max_rel_error_percent(rw),
        "strong_published": max_rel_error_percent(rs),
        "strong_exact": max_rel_error_percent(re_),
    }

    print(
        key,
        R["space_dimensions"][key],
        R["exact_weight_strong_form"][key],
        flush=True,
    )


# Kirchhoff plate reduced-space dimensions from stored results
d = pd.read_csv(os.path.join(RES, "P_plate.csv"))
d = d[(d.n == 16) & (d.method.isin(["weak", "strong"]))]

for case in ("SSSS", "SSFF", "FFFF", "CFFF"):
    x = (
        d[d.case == case]
        if case in set(d.case)
        else d[d.case == {"SSFF": "SFSF"}.get(case, case)]
    )

    R["space_dimensions"]["K_" + case] = {
        "dim_Vw": int(x[x.method == "weak"].red.iloc[0]),
        "dim_Vs": int(x[x.method == "strong"].red.iloc[0]),
        "nphi": 289,
    }

    print(
        case,
        R["space_dimensions"]["K_" + case],
        flush=True,
    )


# ---------------------------------------------------------------------------
# Bordered strong-form pencil for concentrated-mass beams
# ---------------------------------------------------------------------------

CASES = {
    "H1": lambda: PointMassBeam(
        "C", "F", None, 0, 0, m_tip=1.0 * rA * Lb
    ),
    "H2": lambda: PointMassBeam(
        "S", "S", 0.3 * Lb, 1.0 * rA * Lb
    ),
    "H3": lambda: PointMassBeam(
        "C", "F", 0.6 * Lb, 2.0 * rA * Lb
    ),
    "H4": lambda: PointMassBeam(
        "C", "F", 0.4 * Lb, 1.0 * rA * Lb,
        k=100 * EI / Lb**3,
    ),
}


def bordered(b, n=23, N=5000):
    """Solve the bordered one-shot strong-form eigenproblem."""

    t0 = time.perf_counter()
    Bs, G, S4, tot, Nb = b._strong_matrices(n, N)

    b._strong_rows = None

    def lam_rows(lam):
        """Return the lambda-dependent strong-form boundary/interface rows."""

        rows = []

        if b.x0 is not None:
            r = np.zeros(tot)
            r[:Nb] = -EI * Bs[0].d([Bs[0].L], 3)[0]
            r[Nb:] = EI * Bs[1].d([0.0], 3)[0]
            rows.append(r - (b.m * lam - b.k) * b._phi0(Bs, tot, Nb, "x0"))

        if b.mt and b.bcR == "F":
            r = np.zeros(tot)
            B = Bs[-1]
            r[(len(Bs) - 1) * Nb:] = (
                EI * B.d([B.L], 3)[0]
                + b.mt * lam * B.d([B.L], 0)[0]
            )
            rows.append(r)

        return np.vstack(rows)

    # Fixed rows are all essential conditions. The lambda-dependent
    # interface/boundary rows are appended separately below.
    fixed = np.vstack(b._ess(Bs, Nb, tot, True))
    Zf, _ = null_space_svd(fixed)

    c0 = lam_rows(0.0) @ Zf
    c1 = (lam_rows(0.0) - lam_rows(1.0)) @ Zf

    # C(lambda) = C0 - lambda*C1
    A = Zf.T @ S4 @ Zf
    Mm = Zf.T @ G @ Zf
    nc = c0.shape[0]

    Kb = np.block(
        [[A, c0.T], [c0, np.zeros((nc, nc))]]
    )
    Mb = np.block(
        [[Mm, c1.T], [c1, np.zeros((nc, nc))]]
    )

    lam = eig(Kb, Mb, right=False)
    ok = (
        np.isfinite(lam)
        & (np.abs(lam.imag) <= 1e-8 * np.abs(lam.real) + 1e-300)
        & (lam.real > 1e-3)
    )

    lam = np.sort(lam.real[ok])
    return np.sqrt(lam[:b.nmodes]), time.perf_counter() - t0


for key, make_problem in CASES.items():
    b = make_problem()
    ex = b.exact()
    omega_bordered, _ = bordered(b)

    timings = []
    for _ in range(9):
        timings.append(bordered(b)[1])

    strong_iterative = b.strong(23, N=5000, mode="brent")
    weak_result = b.weak(23)

    R["bordered_strong_form"][key] = {
        "err_bordered": (
            float(np.max(np.abs(omega_bordered / ex - 1)) * 100)
            if len(omega_bordered) == 5
            else float("nan")
        ),
        "nfound": len(omega_bordered),
        "t_bordered": float(np.median(timings)),
        "err_iter": float(
            np.max(np.abs(strong_iterative.omega / ex - 1)) * 100
        ),
        "solves_iter": int(strong_iterative.info["solves"]),
        "err_weak": float(
            np.max(np.abs(weak_result.omega / ex - 1)) * 100
        ),
        "om1_bordered": (
            float(omega_bordered[0]) if len(omega_bordered) else None
        ),
    }

    print(key, R["bordered_strong_form"][key], flush=True)


# ---------------------------------------------------------------------------
# Reference-solution convergence for Kirchhoff plates
# ---------------------------------------------------------------------------

for case in ("FFFF", "CFFF"):
    plate = Plate(case)
    R["plate_reference_convergence"][case] = {}

    for n in (22, 26, 30, 34, 38):
        omega = plate.weak(
            n, n + 14, rcond=1e-12, symmetric=True
        ).omega
        R["plate_reference_convergence"][case][n] = omega.tolist()
        print(case, n, omega[:3], flush=True)


save_json(R, "additional_validation")
