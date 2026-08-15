import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import brentq
from sklearn.metrics import r2_score, mean_absolute_error

from .wcf_common import save   # shared paths + 600 DPI PNG / vector PDF saver

mpl.rcParams.update({
    "font.family": "DejaVu Serif", "axes.edgecolor": "#4a4a4a",
    "axes.labelcolor": "#2b2b2b", "xtick.color": "#4a4a4a",
    "ytick.color": "#4a4a4a", "text.color": "#2b2b2b", "axes.linewidth": 0.9,
})
PAL = {"gpr": "#3d6b7d", "frnd": "#b5613f", "kin": "#a6772f",
       "data": "#2b2b2b", "band": "#c9d5da", "grid": "#e7e7e2", "surf": "#3d6b7d"}

# ==================================================================
# REAL DATA
# ==================================================================
# Kinetics arm: m = 1.2 g, V = 0.05 L, C0 = 40 ppm
t_kin  = np.array([0, 10, 20, 30, 50, 80, 110, 140, 170], dtype=float)
qt_kin = np.array([0.0, 0.16306, 0.31258, 0.39029, 0.4958,
                    0.61295, 0.6918, 0.72048, 0.73692])
C0_kin, dosage_kin = 40.0, 24.0  # g/L

# Isotherm arm: m = 0.5 g, V = 0.05 L, t = 170 min (equilibrium)
Co_iso = np.array([20, 30, 40, 50, 60], dtype=float)
qe_iso = np.array([0.68856, 1.16535, 1.67641, 2.43122, 2.82825])
dosage_iso = 10.0  # g/L

# ==================================================================
# MODEL PARAMETERS (from our refits on real data)
# ==================================================================
# Freundlich (refit to isotherm): qe = Kf * Ce^(1/n)
Kf, n_f = 0.01240, 0.6320

# PSO k2 (refit to kinetics arm): we use the value from our independent
# nonlinear least-squares fit on the 9-point kinetics data.
# qe for the kinetics arm is solved implicitly (not the Table 5 value).
k2 = 0.02521  # g/(mg·min) — sourced by 6_pso_classical_fit.py (non-linear PSO
              # fit: qe=0.93601, k2=0.02521, R2=0.9983; see results/wcf_pso_classical_fit.csv)


# ==================================================================
# CORE: solve for qe at given (C0, dosage)
# ==================================================================
def solve_qe(C0, dosage, Kf=Kf, n=n_f):
    """
    Solve the coupled Freundlich + mass-balance system:
        qe = Kf * Ce^(1/n)
        Ce = C0 - qe * dosage
    Returns qe (mg/g). If no positive root exists, returns 0.
    """
    def residual(qe_):
        Ce_ = C0 - qe_ * dosage
        if Ce_ <= 0:
            return -1e6
        return qe_ - Kf * Ce_ ** (1 / n)
    # qe must be in (0, C0/dosage) so Ce stays positive
    qe_max = C0 / dosage * 0.999
    if qe_max <= 0:
        return 0.0
    try:
        return brentq(residual, 1e-12, qe_max, xtol=1e-10)
    except ValueError:
        return 0.0


def predict_q(t, C0, dosage, Kf=Kf, n=n_f, k2=k2):
    """
    Full prediction:  q(t, C0, dosage).
    Works for scalar or array t; scalar C0 and dosage.
    """
    qe = solve_qe(C0, dosage, Kf, n)
    t = np.asarray(t, dtype=float)
    return (qe**2 * k2 * t) / (1 + qe * k2 * t)


# ==================================================================
# VALIDATION: predict both arms from the unified model
# ==================================================================
print("=" * 65)
print("UNIFIED MODEL VALIDATION")
print("=" * 65)

# --- Kinetics arm ---
pred_kin = predict_q(t_kin, C0_kin, dosage_kin)
qe_kin_solved = solve_qe(C0_kin, dosage_kin)
r2_kin = r2_score(qt_kin, pred_kin)
mae_kin = mean_absolute_error(qt_kin, pred_kin)
print(f"\nKinetics arm (C0=40, dosage=24 g/L, n=9):")
print(f"  solved qe = {qe_kin_solved:.5f} mg/g  (last measured: {qt_kin[-1]:.5f})")
print(f"  R² = {r2_kin:.4f},  MAE = {mae_kin:.5f} mg/g")
print(f"  {'t':>6} {'actual':>8} {'predicted':>10}")
for ti, a, p in zip(t_kin, qt_kin, pred_kin):
    print(f"  {ti:6.0f} {a:8.5f} {p:10.5f}")

# --- Isotherm arm (t→equilibrium, different dosage) ---
pred_iso = np.array([solve_qe(c0, dosage_iso) for c0 in Co_iso])
r2_iso = r2_score(qe_iso, pred_iso)
mae_iso = mean_absolute_error(qe_iso, pred_iso)
print(f"\nIsotherm arm (t=equil, dosage=10 g/L, n=5):")
print(f"  R² = {r2_iso:.4f},  MAE = {mae_iso:.5f} mg/g")
print(f"  {'C0':>6} {'actual qe':>10} {'predicted qe':>13}")
for c0, a, p in zip(Co_iso, qe_iso, pred_iso):
    print(f"  {c0:6.0f} {a:10.5f} {p:13.5f}")

# --- Cross-arm check: predict kinetics at ISOTHERM dosage ---
pred_kin_at_iso_dosage = predict_q(170, C0_kin, dosage_iso)
print(f"\nCross-check: predict q at (t=170, C0=40, dosage=10) = {pred_kin_at_iso_dosage:.5f}")
print(f"  actual isotherm qe at C0=40 = {qe_iso[2]:.5f}")
print(f"  match: {abs(pred_kin_at_iso_dosage - qe_iso[2]) / qe_iso[2] * 100:.2f}% error")


# ==================================================================
# FIGURE 1: validation panels
# ==================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5))

# (a) kinetics arm
a1.set_facecolor("#fbfbf9"); a1.grid(color=PAL["grid"], lw=0.8, zorder=0)
t_f = np.linspace(0, 175, 200)
a1.plot(t_f, predict_q(t_f, C0_kin, dosage_kin), color=PAL["gpr"], lw=2,
        label=f"Unified model (R²={r2_kin:.3f})", zorder=4)
a1.scatter(t_kin, qt_kin, color=PAL["data"], s=50, zorder=6,
           edgecolors="white", lw=0.9, label="Kinetics data (24 g/L)")
a1.set_xlabel("Contact time (min)"); a1.set_ylabel("$q_t$ (mg g$^{-1}$)")
a1.set_title("Kinetics arm validation", fontsize=11)
for sp in ["top", "right"]: a1.spines[sp].set_visible(False)
a1.legend(fontsize=9, frameon=False)

# (b) isotherm arm
a2.set_facecolor("#fbfbf9"); a2.grid(color=PAL["grid"], lw=0.8, zorder=0)
co_f = np.linspace(15, 65, 200)
qe_f = np.array([solve_qe(c, dosage_iso) for c in co_f])
a2.plot(co_f, qe_f, color=PAL["frnd"], lw=2,
        label=f"Unified model (R²={r2_iso:.3f})", zorder=4)
a2.scatter(Co_iso, qe_iso, color=PAL["data"], s=50, zorder=6,
           edgecolors="white", lw=0.9, label="Isotherm data (10 g/L)")
a2.set_xlabel("Initial concentration $C_0$ (ppm)")
a2.set_ylabel("$q_e$ (mg g$^{-1}$)")
a2.set_title("Isotherm arm validation", fontsize=11)
for sp in ["top", "right"]: a2.spines[sp].set_visible(False)
a2.legend(fontsize=9, frameon=False)

fig.tight_layout()
save(fig, "wcf_fig_unified_validation")
plt.close()


# ==================================================================
# FIGURE 2: 3-D predictive surface  q(t, C0) at fixed dosage
# ==================================================================
fig2 = plt.figure(figsize=(8, 6))
ax = fig2.add_subplot(111, projection="3d")

# surface at dosage = 10 g/L (isotherm basis, so isotherm points sit ON the surface)
t_grid = np.linspace(0, 180, 60)
c_grid = np.linspace(15, 65, 50)
T, C = np.meshgrid(t_grid, c_grid)
Q = np.zeros_like(T)
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        Q[i, j] = predict_q(T[i, j], C[i, j], dosage_iso)

ax.plot_surface(T, C, Q, alpha=0.55, cmap="coolwarm", edgecolor="none", zorder=1)

# isotherm data points (on the surface)
ax.scatter(np.full_like(Co_iso, 170), Co_iso, qe_iso,
           color=PAL["data"], s=50, zorder=5, depthshade=False,
           label="Isotherm data (10 g/L)")

# kinetics data points (different dosage, NOT on this surface — shown for context)
# convert kinetics q to what it would be at 10 g/L dosage for fair comparison:
# don't convert — show them at their real q, with a note
ax.scatter(t_kin, np.full_like(t_kin, C0_kin), qt_kin,
           color=PAL["kin"], s=35, marker="^", zorder=5, depthshade=False,
           label="Kinetics data (24 g/L, different dosage)")

ax.set_xlabel("Time (min)", labelpad=10)
ax.set_ylabel("$C_0$ (ppm)", labelpad=10)
ax.set_zlabel("$q$ (mg g$^{-1}$)", labelpad=8)
ax.set_title("Predictive surface: $q(t, C_0)$ at 10 g/L dosage", fontsize=12, pad=15)
ax.legend(fontsize=8, loc="upper left")
ax.view_init(elev=28, azim=-55)

fig2.tight_layout()
save(fig2, "wcf_fig_3d_surface")
plt.close()


# ==================================================================
# FIGURE 3: dosage effect — same C0, two dosages
# ==================================================================
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.set_facecolor("#fbfbf9"); ax3.grid(color=PAL["grid"], lw=0.8, zorder=0)

for dos, color, ls, label in [
    (10, PAL["gpr"], "-", "10 g/L (isotherm dosage)"),
    (24, PAL["frnd"], (0, (5, 1.5)), "24 g/L (kinetics dosage)"),
    (5,  PAL["kin"], "--", "5 g/L (hypothetical lower)"),
]:
    t_f = np.linspace(0, 180, 200)
    q_f = predict_q(t_f, 40, dos)
    qe_label = solve_qe(40, dos)
    ax3.plot(t_f, q_f, color=color, lw=2, ls=ls,
             label=f"{label}, $q_e$={qe_label:.3f}")

ax3.scatter(t_kin, qt_kin, color=PAL["data"], s=45, zorder=6,
            edgecolors="white", lw=0.9, label="Kinetics data (24 g/L)")
ax3.set_xlabel("Contact time (min)"); ax3.set_ylabel("$q_t$ (mg g$^{-1}$)")
ax3.set_title("Effect of adsorbent dosage on uptake curve ($C_0$ = 40 ppm)", fontsize=11)
for sp in ["top", "right"]: ax3.spines[sp].set_visible(False)
ax3.legend(fontsize=8, frameon=False)
fig3.tight_layout()
save(fig3, "wcf_fig_dosage_effect")
plt.close()


print("\n✓ Saved wcf_fig_unified_validation.png")
print("✓ Saved wcf_fig_3d_surface.png")
print("✓ Saved wcf_fig_dosage_effect.png")
print("\n" + "=" * 65)
print("USAGE EXAMPLE")
print("=" * 65)
print(">>> predict_q(t=60, C0=35, dosage=10)")
print(f"    = {predict_q(60, 35, 10):.5f} mg/g")
print(">>> predict_q(t=120, C0=50, dosage=15)")
print(f"    = {predict_q(120, 50, 15):.5f} mg/g")
