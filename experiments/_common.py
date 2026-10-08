"""Common setup for experiment / figure scripts: repo root on sys.path, results/ and figures/ helpers."""
import os, sys, time, json, math
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single BLAS thread for fair timings; must happen before numpy is imported
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from pielm.numerics import *
from pielm.bases import *
from pielm.problems import *
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

def save_csv(df, name):
    df.to_csv(os.path.join(RES, name + ".csv"), index=False); print(f"saved {name}.csv ({len(df)} rows)", flush=True)
def save_json(obj, name):
    with open(os.path.join(RES, name + ".json"), "w", encoding="utf-8") as fh: json.dump(obj, fh, indent=1)
    print(f"saved {name}.json", flush=True)
def load_json(name):
    with open(os.path.join(RES, name + ".json"), encoding="utf-8") as fh: return json.load(fh)
def load_csv(name): return pd.read_csv(os.path.join(RES, name + ".csv"))

def median_time(fn, reps, warm=2):
    """Median / quartiles of wall time of fn() after `warm` discarded calls."""
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ts = np.array(ts)
    return float(np.median(ts)), float(np.percentile(ts, 25)), float(np.percentile(ts, 75))
