"""Shared figure setup: repo root on sys.path, cwd = repo root (so 'results/' and 'figures/' resolve), house style."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments")); os.chdir(ROOT)
os.makedirs("figures", exist_ok=True)
from _common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "figure.dpi": 110, "legend.frameon": False, "mathtext.fontset": "cm"})
FLOOR = 1e-14
COL = {"strong": "#7f7f7f", "weak": "#2ca02c"}
LAB = {"strong": "Eig-PIELM (published strong form)", "weak": "Weak-form PIELM (proposed)"}
def save(fig, name):
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight"); fig.savefig(f"figures/{name}.png", bbox_inches="tight"); plt.close(fig); print("fig", name)
