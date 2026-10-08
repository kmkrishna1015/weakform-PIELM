"""Wall time of the full pipeline (basis evaluation -> eigenpairs), single BLAS thread, warm caches.
Analytic derivatives always; nested forward-mode autodiff derivatives if `torch` is installed."""
from _common import *
import platform

REPS = dict(analytic=40, ad=15)
P = problems()
out = dict(env=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=__import__("scipy").__version__,
                    machine=platform.machine(), processor=platform.processor(), threads=1), cases={})
kinds = ["bernstein"]
try:
    import torch; kinds.append("ad")
except ImportError:
    print("torch not installed: autodiff timings skipped")
for key in ORDER:
    p = P[key]; out["cases"][key] = {}
    for kind in kinds:
        B = bases(p, kind)
        for m in METHODS3:
            reps = REPS["analytic"] if kind == "bernstein" else REPS["ad"]
            if kind == "ad" and m == "strong" and p.dim == 1: reps = 5
            run_method(p, m, B=B); run_method(p, m, B=B)
            ts, st = [], []
            for _ in range(reps):
                t = time.perf_counter(); r = run_method(p, m, B=B); ts.append(time.perf_counter() - t); st.append(r.info["stages"])
            ts = np.array(ts)
            out["cases"][key][f"{m}|{'analytic' if kind == 'bernstein' else 'ad'}"] = dict(
                median=float(np.median(ts)), q25=float(np.percentile(ts, 25)), q75=float(np.percentile(ts, 75)),
                min=float(ts.min()), reps=reps, stages=np.median(np.array(st), axis=0).tolist())
    print(key, "done", flush=True)
save_json(out, "timing")
print(f"{'case':10s}{'strong':>10s}{'weak(eig)':>11s}{'weak(sym)':>11s}   ms, analytic derivatives")
for key in ORDER:
    c = out["cases"][key]; g = lambda m: c[m + "|analytic"]["median"] * 1e3
    print(f"{key:10s}{g('strong'):10.2f}{g('weak'):11.2f}{g('weak_sym'):11.2f}   strong/weak {g('strong')/g('weak'):5.1f}x   strong/weak_sym {g('strong')/g('weak_sym'):5.1f}x   weak/weak_sym {g('weak')/g('weak_sym'):4.1f}x")
