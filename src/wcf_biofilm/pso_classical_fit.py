import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error

from .wcf_common import RES_DIR

# ---------------- REAL WCF kinetics data (data_of_biofilm.xlsx) --------------
t  = np.array([0, 10, 20, 30, 50, 80, 110, 140, 170], dtype=float)   # min
qt = np.array([0.0, 0.16306, 0.31258, 0.39029, 0.4958,
               0.61295, 0.6918, 0.72048, 0.73692])                   # mg/g


# ---------------- (A) non-linear PSO fit -------------------------------------
def pso_integrated(t_, qe, k2):
    return (qe**2 * k2 * t_) / (1.0 + qe * k2 * t_)

# fit only on t>0 (t=0,q=0 is exact and carries no information for the ratio)
popt, pcov = curve_fit(pso_integrated, t, qt, p0=(0.75, 0.02),
                       bounds=([0.1, 1e-4], [5.0, 5.0]), maxfev=50000)
qe_nl, k2_nl = popt
perr = np.sqrt(np.diag(pcov))            # 1-sigma parameter std-errors
q_pred_nl = pso_integrated(t, *popt)
r2_nl = r2_score(qt, q_pred_nl)
mae_nl = mean_absolute_error(qt, q_pred_nl)

# ---------------- (B) Ho & McKay linear form ---------------------------------
mask = t > 0
tq = t[mask] / qt[mask]                    # t/q
A = np.vstack([t[mask], np.ones(mask.sum())]).T
slope, intercept = np.linalg.lstsq(A, tq, rcond=None)[0]
qe_lin = 1.0 / slope
k2_lin = 1.0 / (intercept * qe_lin**2)
# R2 of the linear regression itself (t/q vs t)
tq_hat = slope * t[mask] + intercept
r2_lin = r2_score(tq, tq_hat)

# ---------------- report -----------------------------------------------------
print("=" * 60)
print("CLASSICAL PSEUDO-SECOND-ORDER FIT — WCF kinetics arm (n=9)")
print("=" * 60)
print("\n(A) Non-linear least squares on q(t) = qe^2 k2 t / (1 + qe k2 t)")
print(f"    qe = {qe_nl:.5f} +/- {perr[0]:.5f} mg/g")
print(f"    k2 = {k2_nl:.5f} +/- {perr[1]:.5f} g mg^-1 min^-1")
print(f"    R2 = {r2_nl:.4f},  MAE = {mae_nl:.5f} mg/g")
print("\n(B) Ho & McKay linear form  t/q = 1/(k2 qe^2) + t/qe")
print(f"    qe = {qe_lin:.5f} mg/g")
print(f"    k2 = {k2_lin:.5f} g mg^-1 min^-1")
print(f"    R2 (linearised) = {r2_lin:.4f}")

df = pd.DataFrame([
    {"method": "non-linear LS", "qe_mg_g": round(qe_nl, 5),
     "k2_g_mg_min": round(k2_nl, 5), "qe_std": round(perr[0], 5),
     "k2_std": round(perr[1], 5), "R2": round(r2_nl, 4), "MAE": round(mae_nl, 5)},
    {"method": "Ho-McKay linear", "qe_mg_g": round(qe_lin, 5),
     "k2_g_mg_min": round(k2_lin, 5), "qe_std": np.nan, "k2_std": np.nan,
     "R2": round(r2_lin, 4), "MAE": np.nan},
])
out = f"{RES_DIR}/wcf_pso_classical_fit.csv"
df.to_csv(out, index=False)
print(f"\nSaved -> {out}")
print("\nUse the non-linear qe, k2 above as the sourced classical PSO constants.")
# ---------------- sanity check: fitted qe vs last observed point -----------
last_measured = qt[-1]
gap_pct = (qe_nl - last_measured) / last_measured * 100
print("\n" + "=" * 60)
print("SANITY CHECK — fitted qe vs. last raw data point")
print("=" * 60)
print(f"  last MEASURED qt at t=170 min : {last_measured:.5f} mg/g")
print(f"  FITTED asymptotic qe (non-lin): {qe_nl:.5f} mg/g")
print(f"  gap                           : {gap_pct:.2f}%")
