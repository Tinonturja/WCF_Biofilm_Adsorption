import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

from .wcf_common import RES_DIR, save   # shared paths + 300 DPI PNG / vector PDF saver

# ---------------- shared styling (matches the other paper figures) -------------
mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "axes.edgecolor": "#4a4a4a",
    "axes.labelcolor": "#2b2b2b",
    "xtick.color": "#4a4a4a",
    "ytick.color": "#4a4a4a",
    "text.color": "#2b2b2b",
    "axes.linewidth": 0.9,
})
PAL = {
    "data":  "#2b2b2b",
    "pinn":  "#3d6b7d",   # muted teal-blue
    "iso":   "#b5613f",   # clay
    "kin":   "#a6772f",   # ochre
    "alt":   "#7a8b5a",   # sage
    "alt2":  "#8a5a6d",   # dusty plum
    "band":  "#c9d5da",
    "grid":  "#e7e7e2",
    "bg":    "#fbfbf9",
}
CMAP = LinearSegmentedColormap.from_list(
    "muted", ["#dce4e7", "#9fb8c4", "#6f93a5", "#b5613f"])


def _ax(ax, grid_axis="both"):
    ax.set_facecolor(PAL["bg"])
    ax.grid(axis=grid_axis, color=PAL["grid"], linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def r2(y, yh):
    y, yh = np.asarray(y), np.asarray(yh)
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


D = np.load(f"{RES_DIR}/pinn_surface_results.npz", allow_pickle=True)
k2_, Kf_, n_, n_inv_ = D["constants"]
VM_KIN, VM_ISO = float(D["vm_kin"]), float(D["vm_iso"])

# ======================================================================
# 1. VALIDATION — both arms
# ======================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5))

_ax(a1)
a1.plot(D["t_fine"], D["kin_curve"], color=PAL["pinn"], lw=2,
        label=f"PINN surface (R²={r2(D['q_kin'], D['pred_kin']):.3f})", zorder=4)
a1.scatter(D["t_kin"], D["q_kin"], color=PAL["data"], s=52, zorder=6,
           edgecolors="white", lw=0.9, label=f"Kinetics data (V/m={VM_KIN:.4f})")
a1.set_xlabel("Contact time (min)")
a1.set_ylabel("$q_t$ (mg g$^{-1}$)")
a1.set_title("(a) Kinetics arm", fontsize=11)
a1.legend(fontsize=8.5, frameon=False)

_ax(a2)
a2.plot(D["c_fine"], D["iso_curve"], color=PAL["iso"], lw=2,
        label=f"PINN surface (R²={r2(D['q_iso'], D['pred_iso']):.3f})", zorder=4)
a2.scatter(D["c_iso"], D["q_iso"], color=PAL["data"], s=52, zorder=6,
           edgecolors="white", lw=0.9, label=f"Isotherm data (V/m={VM_ISO:.4f})")
a2.set_xlabel("Initial concentration $C_0$ (ppm)")
a2.set_ylabel("$q_e$ (mg g$^{-1}$)")
a2.set_title("(b) Isotherm arm", fontsize=11)
a2.legend(fontsize=8.5, frameon=False)

fig.suptitle("PINN surface validated against both experimental arms",
             fontsize=12.5, y=1.00)
fig.tight_layout()
save(fig, "fig_pinn_validation")
plt.close()

# ======================================================================
# 2. 3-D SURFACES at the two dosages
# ======================================================================
fig = plt.figure(figsize=(13.5, 6))
T, C = np.meshgrid(D["t_grid"], D["c_grid"], indexing="ij")

for k, (surf, vmv, ttl, pts) in enumerate([
    (D["surf_iso"], VM_ISO, f"(a) V/m = {VM_ISO:.4f} L g$^{{-1}}$ (isotherm)", "iso"),
    (D["surf_kin"], VM_KIN, f"(b) V/m = {VM_KIN:.4f} L g$^{{-1}}$ (kinetics)", "kin"),
]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.plot_surface(T, C, surf, cmap=CMAP, alpha=0.85,
                    linewidth=0, antialiased=True, rstride=2, cstride=2)
    if pts == "iso":
        ax.scatter(np.full_like(D["c_iso"], 170.0), D["c_iso"], D["q_iso"],
                   color=PAL["data"], s=45, depthshade=False, label="measured")
    else:
        ax.scatter(D["t_kin"], np.full_like(D["t_kin"], 40.0), D["q_kin"],
                   color=PAL["data"], s=45, depthshade=False, label="measured")
    ax.set_xlabel("t (min)", labelpad=8)
    ax.set_ylabel("$C_0$ (ppm)", labelpad=8)
    ax.set_zlabel("q (mg g$^{-1}$)", labelpad=6)
    ax.set_title(ttl, fontsize=10.5, pad=12)
    ax.view_init(elev=26, azim=-58)
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle("PINN-learned adsorption surface $q(t,\\,C_0)$ at two dosages",
             fontsize=12.5, y=0.97)
fig.tight_layout()
save(fig, "fig_pinn_surface3d")
plt.close()

# ======================================================================
# 3. DOSAGE EFFECT
# ======================================================================
fig, ax = plt.subplots(figsize=(7.5, 5))
_ax(ax)
keys = [str(k) for k in D["dosage_keys"]]
vals = D["dosage_vals"]
colors = [PAL["kin"], PAL["alt"], PAL["pinn"], PAL["alt2"]]
styles = [(0, (5, 1.5)), (0, (3, 1.5)), "-", (0, (1, 1.2))]
for i, (k, v) in enumerate(zip(keys, vals)):
    ax.plot(D["t_fine"], v, color=colors[i % len(colors)],
            ls=styles[i % len(styles)], lw=2,
            label=f"V/m = {float(k):.4f} L g$^{{-1}}$")
ax.scatter(D["t_kin"], D["q_kin"], color=PAL["data"], s=48, zorder=6,
           edgecolors="white", lw=0.9, label="measured (kinetics arm)")
ax.set_xlabel("Contact time (min)")
ax.set_ylabel("$q_t$ (mg g$^{-1}$)")
ax.set_title("Predicted effect of dosage on uptake ($C_0$ = 40 ppm)", fontsize=11.5)
ax.legend(fontsize=8.5, frameon=False)
fig.tight_layout()
save(fig, "fig_pinn_dosage")
plt.close()

# ======================================================================
# 4. TRAINING CONVERGENCE
# ======================================================================
H = D["history"]
fig, ax = plt.subplots(figsize=(7.5, 5))
_ax(ax)
for col, lab, colr in [(1, "data", PAL["pinn"]), (2, "PSO residual", PAL["iso"]),
                       (3, "isotherm", PAL["alt"]), (4, "initial cond.", PAL["alt2"])]:
    ax.semilogy(H[:, 0], np.maximum(H[:, col], 1e-16), lw=1.8, color=colr, label=lab)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss component (log scale)")
ax.set_title("Convergence of the four loss terms", fontsize=11.5)
ax.legend(fontsize=8.5, frameon=False)
fig.tight_layout()
save(fig, "fig_pinn_training")
plt.close()

# ======================================================================
# 5. CONTOUR MAP with the measured "L" overlaid
# ======================================================================
fig, ax = plt.subplots(figsize=(8, 5.6))
ax.set_facecolor(PAL["bg"])
cf = ax.contourf(T, C, D["surf_iso"], levels=18, cmap=CMAP, alpha=0.92)
cl = ax.contour(T, C, D["surf_iso"], levels=9, colors="#5a5a5a",
                linewidths=0.6, alpha=0.55)
ax.clabel(cl, inline=True, fontsize=7, fmt="%.2f")
cb = fig.colorbar(cf, ax=ax, pad=0.02)
cb.set_label("q (mg g$^{-1}$)")

# the measured L: kinetics arm (horizontal) + isotherm arm (vertical)
ax.plot(D["t_kin"], np.full_like(D["t_kin"], 40.0), "o-", color=PAL["data"],
        ms=5, lw=1.4, label="kinetics arm (measured)")
ax.plot(np.full_like(D["c_iso"], 170.0), D["c_iso"], "s-", color=PAL["iso"],
        ms=5, lw=1.4, label="isotherm arm (measured)")
ax.set_xlabel("Contact time (min)")
ax.set_ylabel("Initial concentration $C_0$ (ppm)")
ax.set_title(f"Predicted surface at V/m = {VM_ISO:.4f} L g$^{{-1}}$\n"
             "(interior filled by the physics constraints, not by data)",
             fontsize=11)
ax.legend(fontsize=8.5, frameon=False, loc="lower left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
save(fig, "fig_pinn_contour")
plt.close()

# ======================================================================
print("Recovered constants:")
print(f"  k2  = {k2_:.5f} g mg^-1 min^-1")
print(f"  Kf  = {Kf_:.5f}")
print(f"  n   = {n_:.4f}   (1/n = {n_inv_:.4f})")
print("\nSaved:")
for f in ["fig_pinn_validation.png", "fig_pinn_surface3d.png",
          "fig_pinn_dosage.png", "fig_pinn_training.png",
          "fig_pinn_contour.png"]:
    print(f"  {f}")
