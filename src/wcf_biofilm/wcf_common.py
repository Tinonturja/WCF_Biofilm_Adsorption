import os
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa: F401  (re-exported for convenience)

# ------------------------------------------------------------------
# 1. PATHS — resolved relative to this file (repo_root/code/wcf_common.py)
# ------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
#print(f"Dirname: {_HERE}")
ROOT_DIR = os.path.dirname(os.path.dirname(_HERE))
#print(f"Root dir: {ROOT_DIR}")
FIG_DIR = os.path.join(ROOT_DIR, "figures")
RES_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 2. PUBLICATION STYLE
# ------------------------------------------------------------------
mpl.rcParams.update({
    # fonts
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    # colours / weights (muted, print-friendly)
    "axes.edgecolor": "#4a4a4a",
    "axes.labelcolor": "#2b2b2b",
    "xtick.color": "#4a4a4a",
    "ytick.color": "#4a4a4a",
    "text.color": "#2b2b2b",
    "axes.linewidth": 0.9,
    # output quality
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    # embed editable TrueType text in vector output (journal requirement)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# shared muted palette (slate / teal / clay / ochre / sage / plum)
PAL = {
    "data":  "#2b2b2b",
    "gpr":   "#3d6b7d",   # muted teal-blue
    "poly":  "#a6772f",   # ochre
    "poly2": "#a6772f",
    "poly3": "#c99b3f",   # light ochre
    "rf":    "#7a8b5a",   # sage
    "xgb":   "#8a5a6d",   # dusty plum
    "pinn":  "#b5613f",   # clay / terracotta
    "frnd":  "#b5613f",
    "kin":   "#a6772f",
    "iso":   "#b5613f",
    "alt":   "#7a8b5a",
    "alt2":  "#8a5a6d",
    "band":  "#c9d5da",   # pale slate for CI
    "grid":  "#e7e7e2",
    "bg":    "#fbfbf9",
    "surf":  "#3d6b7d",
}


# ------------------------------------------------------------------
# 3. SAVE HELPER — PNG (300 dpi) + PDF (vector) into figures/
# ------------------------------------------------------------------
def save(fig, stem, dpi=600):
    """Save `fig` as <stem>.png (raster, dpi) and <stem>.pdf (vector)."""
    png = os.path.join(FIG_DIR, stem + ".png")
    fig.savefig(png, dpi=dpi)
    return png
