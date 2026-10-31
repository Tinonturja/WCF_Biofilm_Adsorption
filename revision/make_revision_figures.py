"""Figures for the revision, built only from the JSON results written by the other
revision scripts (no refitting here). Output: figures/rev_*.png"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from wcf_biofilm.data import t_kin, q_kin       # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
J = lambda f: json.load(open(os.path.join(RES, f)))
cm = J("revision_classical_ml.json")
p1 = J("revision_pinn1d_summary.json")
us = J("revision_unified_surface.json")
p3 = J("revision_pinn3d_summary.json")

# palette: categorical slots 1-3 (validated) + neutral grey; text stays in ink colours
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.linewidth": 0.8, "savefig.dpi": 600,
                     "savefig.bbox": "tight", "legend.frameon": False})


def clean(ax, grid="x"):
    ax.grid(axis=grid, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------- Fig R1: kinetics benchmark
kin = cm["kinetics"]
models = [("PSO", "PSO (2 parameters)", BLUE, kin["PSO"]["LOO_MAE"], kin["PSO"]["extrap_MAE"]),
          ("PFO", "PFO (2 parameters)", BLUE, kin["PFO"]["LOO_MAE"], kin["PFO"]["extrap_MAE"]),
          ("nd", "PINN, corrected (λ=1)", AQUA, p1["nd_l1"]["LOO_MAE_mean_sd_min_max"][0], p1["nd_l1"]["extrap_MAE_mean_sd_min_max"][0]),
          ("raw", "PINN, as submitted", GREY, p1["raw_l1"]["LOO_MAE_mean_sd_min_max"][0], p1["raw_l1"]["extrap_MAE_mean_sd_min_max"][0])]
for k in ("GPR (Matern 5/2)", "Polynomial (deg 2)", "Polynomial (deg 3)",
          "Random forest (100 trees)", "XGBoost (100 rounds, depth 3)"):
    models.append((k, k.replace(" (100 rounds, depth 3)", "").replace(" (Matern 5/2)", ""),
                   ORANGE, kin[k]["LOO_MAE"], kin[k]["extrap_MAE"]))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
y = np.arange(len(models))[::-1]
for ax, j, title in ((axes[0], 3, "Leave-one-out MAE (interpolation)"),
                     (axes[1], 4, "Forward-extrapolation MAE\n(train t ≤ 110 min, predict 140 and 170 min)")):
    vals = [m[j] for m in models]
    ax.barh(y, vals, height=0.62, color=[m[2] for m in models], zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v + max(vals) * 0.015, yy, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("mg g$^{-1}$")
    ax.set_xlim(0, max(vals) * 1.22)
    clean(ax)
axes[0].set_yticks(y)
axes[0].set_yticklabels([m[1] for m in models])
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=BLUE, label="Classical rate law"), Patch(color=AQUA, label="PINN (corrected)"),
                    Patch(color=GREY, label="PINN (as submitted)"), Patch(color=ORANGE, label="Data-driven")],
           loc="lower center", ncol=4, bbox_to_anchor=(0.55, -0.04), fontsize=9)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(FIG, "rev_fig1_kinetics_benchmark.png"))
plt.close(fig)

# ------------------------------------------------------------- Fig R2: PINN ablation
cfgs = [("raw_l0", "λ=0"), ("raw_l1", "λ=1"),
        ("nd_l0", "λ=0"), ("nd_l0.1", "λ=0.1"), ("nd_l1", "λ=1"),
        ("nd_l10", "λ=10"), ("nd_l100", "λ=100")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
x = np.arange(len(cfgs))
for ax, key, lab in ((axes[0], "LOO_MAE", "Leave-one-out MAE (mg g$^{-1}$)"),
                     (axes[1], "extrap_MAE", "Extrapolation MAE (mg g$^{-1}$)")):
    for i, (c, _) in enumerate(cfgs):
        vals = [s[key] for s in p1[c]["per_seed"]]
        col = GREY if c.startswith("raw") else AQUA
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, s=34,
                   color=col, edgecolor="white", linewidth=1.5, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [np.mean(vals)] * 2, color=INK, lw=1.4, zorder=4)
    ax.axhline(kin["PSO"][key], color=BLUE, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(1.42, kin["PSO"][key] * 1.04, "PSO equation ", color=INK2, fontsize=8.5, va="bottom", ha="right", zorder=6, bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in cfgs], fontsize=8.5)
    ax.axvline(1.5, color=GRID, lw=1)
    for xc, grp in ((0.5, "as submitted"), (4.0, "corrected")):
        ax.text(xc, -0.16, grp, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, color=INK)
    ax.set_ylabel(lab)
    ax.set_yscale("log")
    clean(ax, "y")
axes[0].set_title("Each dot is one random seed (5 per setting); bar = mean", fontsize=9.5, loc="left", color=INK2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "rev_fig2_pinn_ablation.png"))
plt.close(fig)

# ------------------------------------------------------------- Fig R3: learned kinetic curves
rows = [json.loads(l) for l in open(os.path.join(HERE, "res1d.jsonl"))] if os.path.exists(
    os.path.join(HERE, "res1d.jsonl")) else []
from wcf_biofilm.classical import pso
fig, ax = plt.subplots(figsize=(6.4, 4.2))
tf = np.linspace(0, 180, 181)
qe, k2 = kin["PSO"]["params"]
for r in rows:
    if r["split"] == "full" and r["cfg"] in ("raw_l1", "nd_l1"):
        ax.plot(r["curve_t"], r["curve_q"], color=GREY if r["cfg"] == "raw_l1" else AQUA, lw=1.2, alpha=0.8, zorder=2)
ax.plot(tf, pso(tf, qe, k2), color=BLUE, lw=2, ls=(0, (4, 3)), zorder=3)
ax.scatter(t_kin, q_kin, s=40, color=INK, edgecolor="white", linewidth=1.5, zorder=5)
ax.axhline(qe, color=INK2, lw=0.8, ls=":", zorder=1)
ax.text(2, qe + 0.012, f"fitted PSO asymptote q$_e$ = {qe:.3f}", fontsize=8.5, color=INK2)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=BLUE, lw=2, ls=(0, (4, 3)), label="PSO equation (NLS)"),
                   Line2D([], [], color=AQUA, lw=1.2, label="PINN corrected, 5 seeds"),
                   Line2D([], [], color=GREY, lw=1.2, label="PINN as submitted, 5 seeds"),
                   Line2D([], [], marker="o", ls="", color=INK, label="Measured")],
          loc="lower right", fontsize=8.5)
ax.set_xlabel("Contact time (min)")
ax.set_ylabel("q$_t$ (mg g$^{-1}$)")
clean(ax, "both")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "rev_fig3_kinetic_curves.png"))
plt.close(fig)

# ------------------------------------------------------------- Fig R4: untested conditions
tests = [("t60_C30_10gL", "t = 60 min, C$_0$ = 30 mg L$^{-1}$, 10 g L$^{-1}$\n(inside the sampled range)"),
         ("t170_C40_5gL", "t = 170 min, C$_0$ = 40 mg L$^{-1}$, 5 g L$^{-1}$\n(untested dosage: extrapolation)")]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(wspace=0.75))
for ax, (k, title) in zip(axes, tests):
    rowsp = [("Sequential surface\n(as submitted)", BLUE, [us["sequential"]["test"][k]], us["sequential"]["bootstrap"]["test_CI"][k]),
             ("Joint 3-parameter fit", BLUE, [us["joint"]["test"][k]], us["joint"]["bootstrap"]["test_CI"][k]),
             ("3-input PINN, corrected", AQUA, p3["nondim"]["test_values"][k], None),
             ("3-input PINN, as submitted", GREY, p3["raw"]["test_values"][k], None)]
    for i, (lab, col, vals, ci) in enumerate(rowsp[::-1]):
        if ci:
            ax.plot(ci, [i, i], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(vals, np.full(len(vals), i), s=40, color=col, edgecolor="white", linewidth=1.5, zorder=3)
    ax.set_yticks(range(len(rowsp)))
    ax.set_yticklabels([r[0] for r in rowsp[::-1]], fontsize=8.5)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_xlabel("Predicted q (mg g$^{-1}$)")
    clean(ax)
axes[0].text(0.0, -0.32, "Lines: 95% bootstrap interval (parameter uncertainty). Dots for PINNs: one per random seed.",
             transform=axes[0].transAxes, fontsize=8, color=INK2)
fig.savefig(os.path.join(FIG, "rev_fig4_untested_conditions.png"))
plt.close(fig)
print("saved rev_fig1..4")
