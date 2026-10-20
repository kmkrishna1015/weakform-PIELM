"""Role of the natural rows (paper Fig. 8): SSSS and SSFF Kirchhoff plates, n = 16, Gauss weights with ng = n + 12 = 28.
See pielm/ablation.py for the three rows and the (fixed) selection rules.  The ablated pencil (natural rows removed, pointwise residual kept) is NOT
symmetric: most of its eigenvalues are complex and its real eigenvalues depend on the quadrature order, so the sensitivity to ng is stored as well."""
from _common import *
from pielm.ablation import ablation, real_pairs
from pielm.kirchhoff import Plate
N = 16; out = {}
for case in ("SSSS", "SSFF"):
    r = ablation(case, N); P = Plate(case); sens = {}
    for ng in (24, 26, 28, 30, 32, 40):
        l, B, _ = P.strong_gauss(N, ng, natural=False); o, _ = real_pairs(l, B)
        sens[ng] = dict(n_real=int(o.size), n_total=int(l.size), lowest3=[round(float(v), 3) for v in o[:3]])
    out[case] = dict(n=N, ng=r["ng"], exact=r["exact"], n_real_noNat=r["n_real_noNat"], n_total=r["n_total"],
                     weak=[dict(omega=o, overlap=v) for o, _, v in r["weak"]], strong_all=[dict(omega=o, overlap=v) for o, _, v in r["strong_all"]],
                     strong_noNat_lowest3=[dict(omega=o) for o, _, _ in r["strong_noNat"]], noNat_best_match=r["noNat_best_match"], sensitivity_to_ng=sens)
    print(case, "exact", np.round(r["exact"], 3), "| weak", [round(o, 3) for o, _, _ in r["weak"]], "| all", [round(o, 3) for o, _, _ in r["strong_all"]],
          "| noNat lowest3", [round(o, 3) for o, _, _ in r["strong_noNat"]], f"| real {r['n_real_noNat']}/{r['n_total']}", flush=True)
save_json(out, "P_natural_row_ablation")
