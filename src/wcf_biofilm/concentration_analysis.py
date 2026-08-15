import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

from .wcf_common import RES_DIR, save   # shared paths + 300 DPI PNG / vector PDF saver

# ---------------- styling (matches the kinetics figures) ----------------
mpl.rcParams.update({
    "font.family": "DejaVu Serif", "axes.edgecolor": "#4a4a4a",
    "axes.labelcolor": "#2b2b2b", "xtick.color": "#4a4a4a",
    "ytick.color": "#4a4a4a", "text.color": "#2b2b2b", "axes.linewidth": 0.9,
})
PAL = {"data": "#2b2b2b", "gpr": "#3d6b7d", "poly": "#a6772f",
       "rf": "#7a8b5a", "frnd": "#b5613f", "band": "#c9d5da", "grid": "#e7e7e2"}

# ---------------- REAL WCF DATA (data_of_biofilm.xlsx) ----------------
Co      = np.array([20.0, 30.0, 40.0, 50.0, 60.0])          # initial conc, ppm
Ce      = np.array([13.11445, 18.3465, 23.23591, 25.68784, 31.71751])  # mg/L
qe      = np.array([0.68856, 1.16535, 1.67641, 2.43122, 2.82825])      # mg/g
removal = np.array([34.42776, 38.845, 41.91021, 48.62431, 47.13748])   # %

Kf_TABLE6, n_TABLE6 = 0.0093, 0.598   # manuscript Table 6


def freundlich(Ce_, Kf, n):
    return Kf * Ce_ ** (1.0 / n)


def gpr_factory():
    k = 1.0 * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.01)
    return GaussianProcessRegressor(kernel=k, n_restarts_optimizer=15,
                                    normalize_y=True, random_state=42)


MODELS = {
    "GPR (Matern 5/2)":   gpr_factory,
    "Polynomial (deg 2)": lambda: Pipeline([("p", PolynomialFeatures(2)),
                                            ("l", LinearRegression())]),
    "Random Forest":      lambda: RandomForestRegressor(n_estimators=100, random_state=42),
}


def evaluate(x, y, label):
    """Train + LOO scores for each model on a 1-D input."""
    out = []
    for name, factory in MODELS.items():
        # train
        s = StandardScaler(); X = s.fit_transform(x.reshape(-1, 1))
        m = factory(); m.fit(X, y)
        tr = r2_score(y, m.predict(X))
        # LOO
        preds = np.zeros_like(y, dtype=float)
        for tr_i, te_i in LeaveOneOut().split(x):
            ss = StandardScaler()
            mm = factory()
            mm.fit(ss.fit_transform(x[tr_i].reshape(-1, 1)), y[tr_i])
            preds[te_i] = mm.predict(ss.transform(x[te_i].reshape(-1, 1)))
        out.append({"Response": label, "Model": name,
                    "Train_R2": round(tr, 4),
                    "LOO_R2": round(r2_score(y, preds), 4),
                    "LOO_MAE": round(mean_absolute_error(y, preds), 4)})
    return out


rows = []
rows += evaluate(Ce, qe, "q_e vs C_e")
rows += evaluate(Co, removal, "removal % vs C_0")

# ---- Freundlich refit directly to the same isotherm points ----
popt, _ = curve_fit(freundlich, Ce, qe, p0=(0.01, 0.6), maxfev=20000,
                    bounds=([1e-6, 0.05], [10, 10]))
frnd_refit_r2 = r2_score(qe, freundlich(Ce, *popt))
frnd_table6_r2 = r2_score(qe, freundlich(Ce, Kf_TABLE6, n_TABLE6))

# Freundlich LOO
frnd_loo = np.zeros_like(qe)
for tr_i, te_i in LeaveOneOut().split(Ce):
    p, _ = curve_fit(freundlich, Ce[tr_i], qe[tr_i], p0=(0.01, 0.6),
                     maxfev=20000, bounds=([1e-6, 0.05], [10, 10]))
    frnd_loo[te_i] = freundlich(Ce[te_i], *p)
rows.append({"Response": "q_e vs C_e", "Model": "Freundlich (refit)",
             "Train_R2": round(frnd_refit_r2, 4),
             "LOO_R2": round(r2_score(qe, frnd_loo), 4),
             "LOO_MAE": round(mean_absolute_error(qe, frnd_loo), 4)})

res = pd.DataFrame(rows)
print("=== WCF concentration-domain model comparison (LOO-CV, n=5) ===")
print(res.to_string(index=False))
print(f"\nFreundlich refit to raw data : Kf = {popt[0]:.5f}, n = {popt[1]:.4f}, R2 = {frnd_refit_r2:.4f}")
print(f"Freundlich Table 6 params    : Kf = {Kf_TABLE6}, n = {n_TABLE6}, R2 vs raw = {frnd_table6_r2:.4f}")

# ---------------- DRIVING-FORCE / CAPACITY-EFFICIENCY DIAGNOSTIC ----------------
qe_gain      = (qe[-1] - qe[0]) / qe[0] * 100.0          # % rise in uptake
removal_gain = (removal[-1] - removal[0]) / removal[0] * 100.0
print("\n--- capacity vs efficiency across 20 -> 60 ppm ---")
print(f"q_e     rose from {qe[0]:.3f} to {qe[-1]:.3f} mg/g   (+{qe_gain:.1f}%)")
print(f"removal rose from {removal[0]:.2f} to {removal[-1]:.2f} %      (+{removal_gain:.1f}%)")
print(f"ratio (capacity gain / efficiency gain) = {qe_gain/removal_gain:.2f}")
peak_i = int(np.argmax(removal))
print(f"peak removal {removal[peak_i]:.2f}% at C0 = {Co[peak_i]:.0f} ppm "
      f"(declines at {Co[-1]:.0f} ppm -> {removal[-1]:.2f}%)")

res.to_csv(f"{RES_DIR}/wcf_concentration_model_comparison.csv", index=False)

# ==================================================================
# FIGURE: two panels — (a) isotherm, (b) removal %
# ==================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5))

# ---- (a) q_e vs C_e ----
a1.set_facecolor("#fbfbf9"); a1.grid(color=PAL["grid"], lw=0.8, zorder=0)
s = StandardScaler(); X = s.fit_transform(Ce.reshape(-1, 1))
g = gpr_factory(); g.fit(X, qe)
ce_f = np.linspace(Ce.min() * 0.85, Ce.max() * 1.12, 200)
mu, sd = g.predict(s.transform(ce_f.reshape(-1, 1)), return_std=True)
a1.fill_between(ce_f, mu - 2 * sd, mu + 2 * sd, color=PAL["band"], alpha=0.85,
                label="95% confidence interval", zorder=1)
a1.plot(ce_f, mu, color=PAL["gpr"], lw=2, label="GPR mean", zorder=4)
a1.plot(ce_f, freundlich(ce_f, *popt), ls=(0, (5, 1.5)), color=PAL["frnd"], lw=1.8,
        label=f"Freundlich refit (R²={frnd_refit_r2:.3f})", zorder=3)
a1.scatter(Ce, qe, color=PAL["data"], s=45, zorder=6, edgecolors="white", lw=0.9,
           label="Experimental data")
a1.set_xlabel("Equilibrium concentration $C_e$ (mg L$^{-1}$)")
a1.set_ylabel("$q_e$ (mg g$^{-1}$)")
a1.set_title("Equilibrium uptake rises with concentration", fontsize=11)
for sp in ["top", "right"]: a1.spines[sp].set_visible(False)
a1.legend(fontsize=8, frameon=False)

# ---- (b) removal % vs C0 ----
a2.set_facecolor("#fbfbf9"); a2.grid(color=PAL["grid"], lw=0.8, zorder=0)
s2 = StandardScaler(); X2 = s2.fit_transform(Co.reshape(-1, 1))
g2 = gpr_factory(); g2.fit(X2, removal)
co_f = np.linspace(Co.min() * 0.9, Co.max() * 1.06, 200)
mu2, sd2 = g2.predict(s2.transform(co_f.reshape(-1, 1)), return_std=True)
a2.fill_between(co_f, mu2 - 2 * sd2, mu2 + 2 * sd2, color=PAL["band"], alpha=0.85,
                label="95% confidence interval", zorder=1)
a2.plot(co_f, mu2, color=PAL["gpr"], lw=2, label="GPR mean", zorder=4)
a2.scatter(Co, removal, color=PAL["data"], s=45, zorder=6, edgecolors="white", lw=0.9,
           label="Experimental data")
a2.axvline(Co[peak_i], color=PAL["poly"], lw=1.2, ls=":", zorder=2,
           label=f"peak removal ({Co[peak_i]:.0f} ppm)")
a2.set_xlabel("Initial dye concentration $C_0$ (ppm)")
a2.set_ylabel("Dye removal (%)")
a2.set_title("Removal efficiency plateaus and turns over", fontsize=11)
for sp in ["top", "right"]: a2.spines[sp].set_visible(False)
a2.legend(fontsize=8, frameon=False)

fig.tight_layout()
save(fig, "wcf_fig_concentration")
plt.close()
print("\nSaved wcf_fig_concentration.png")
