"""Deterministic part of the benchmark: classical equations and data-driven models.

Kinetics arm (n = 9) and isotherm arm (n = 5), each evaluated by
  * training fit (all points)                 -> Train R2, AICc (classical only)
  * leave-one-out CV                          -> LOO R2 / MAE / RMSE
  * forward extrapolation (kinetics only)     -> train t <= 110 min, predict 140 and 170 min
All preprocessing (standardisation) is refit inside each training fold.
Writes results/revision_classical_ml.json
"""
import os, sys, json, warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
from xgboost import XGBRegressor
from wcf_biofilm import classical as C
from wcf_biofilm.data import t_kin, q_kin, Ce_iso, q_iso, C0_iso, removal_iso

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                   "revision_classical_ml.json")


def gpr():
    k = 1.0 * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.01)
    return GaussianProcessRegressor(kernel=k, n_restarts_optimizer=5, normalize_y=True,
                                    random_state=42)


ML = {
    "GPR (Matern 5/2)": gpr,
    "Polynomial (deg 2)": lambda: Pipeline([("p", PolynomialFeatures(2)), ("l", LinearRegression())]),
    "Polynomial (deg 3)": lambda: Pipeline([("p", PolynomialFeatures(3)), ("l", LinearRegression())]),
    "Random forest (50 trees)": lambda: RandomForestRegressor(n_estimators=50, random_state=42),
    "Random forest (100 trees)": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "Random forest (500 trees)": lambda: RandomForestRegressor(n_estimators=500, random_state=42),
    "XGBoost (100 rounds, depth 3)": lambda: XGBRegressor(n_estimators=100, max_depth=3,
                                                          random_state=42, verbosity=0),
}
ML_ISO = {
    "GPR (Matern 5/2)": gpr,
    "Polynomial (deg 2)": ML["Polynomial (deg 2)"],
    "Random forest (100 trees, depth 2)": lambda: RandomForestRegressor(
        n_estimators=100, max_depth=2, random_state=42),
}


def ml_predict(factory, xtr, ytr, xte):
    s = StandardScaler()
    m = factory()
    m.fit(s.fit_transform(np.reshape(xtr, (-1, 1))), ytr)
    return m.predict(s.transform(np.reshape(xte, (-1, 1))))


def cls_predict(kind, name, xtr, ytr, xte):
    if kind == "kin":
        p, _ = C.fit_kinetic(name, xtr, ytr)
        return C.KINETIC_MODELS[name][0](xte, *p)
    p, _ = C.fit_isotherm(name, xtr, ytr)
    return C.ISOTHERM_MODELS[name][0](xte, *p)


def evaluate(predict, x, y, extrap=False):
    n = len(x)
    fit = predict(x, y, x)
    loo = np.array([predict(np.delete(x, i), np.delete(y, i), x[i:i + 1])[0] for i in range(n)])
    r = dict(train_R2=C.r2(y, fit), train_RMSE=C.rmse(y, fit),
             LOO_R2=C.r2(y, loo), LOO_MAE=C.mae(y, loo), LOO_RMSE=C.rmse(y, loo),
             LOO_pred=loo.tolist(), fit_pred=np.asarray(fit).tolist())
    if extrap:
        ex = predict(x[:-2], y[:-2], x[-2:])
        r.update(extrap_pred=np.asarray(ex).tolist(),
                 extrap_abs_err=np.abs(np.asarray(ex) - y[-2:]).tolist(),
                 extrap_MAE=C.mae(y[-2:], ex))
    return r


res = {"kinetics": {}, "isotherm_qe_vs_Ce": {}, "removal_vs_C0": {}}

# ---------------- kinetics
for name in ("PSO", "PFO"):
    r = evaluate(lambda a, b, c, nm=name: cls_predict("kin", nm, a, b, c), t_kin, q_kin, True)
    p, se = C.fit_kinetic(name, t_kin, q_kin)
    r.update(params=p.tolist(), param_se=se.tolist(), n_params=2,
             AICc=C.aicc(q_kin, C.KINETIC_MODELS[name][0](t_kin, *p), 2))
    res["kinetics"][name] = r
for name, f in ML.items():
    res["kinetics"][name] = evaluate(lambda a, b, c, f=f: ml_predict(f, a, b, c), t_kin, q_kin, True)

# ---------------- isotherm arm (q at 170 min vs Ce)
for name in ("Freundlich", "Langmuir"):
    r = evaluate(lambda a, b, c, nm=name: cls_predict("iso", nm, a, b, c), Ce_iso, q_iso)
    p, se = C.fit_isotherm(name, Ce_iso, q_iso)
    r.update(params=p.tolist(), param_se=se.tolist(), n_params=2,
             AICc=C.aicc(q_iso, C.ISOTHERM_MODELS[name][0](Ce_iso, *p), 2))
    res["isotherm_qe_vs_Ce"][name] = r
for name, f in ML_ISO.items():
    res["isotherm_qe_vs_Ce"][name] = evaluate(lambda a, b, c, f=f: ml_predict(f, a, b, c), Ce_iso, q_iso)
for name, f in ML_ISO.items():
    res["removal_vs_C0"][name] = evaluate(lambda a, b, c, f=f: ml_predict(f, a, b, c), C0_iso, removal_iso)

# linearised Ho-McKay PSO and ln-ln Freundlich, as quoted in the manuscript
m = t_kin > 0
sl, ic = np.polyfit(t_kin[m], t_kin[m] / q_kin[m], 1)
res["linearised"] = {"PSO_HoMcKay": dict(qe=1 / sl, k2=1 / (ic * (1 / sl) ** 2),
                                         R2_linear=C.r2(t_kin[m] / q_kin[m], sl * t_kin[m] + ic))}
b1, b0 = np.polyfit(np.log(Ce_iso), np.log(q_iso), 1)
res["linearised"]["Freundlich_lnln"] = dict(Kf=float(np.exp(b0)), n=float(1 / b1),
                                            R2_linear=C.r2(np.log(q_iso), b1 * np.log(Ce_iso) + b0))
# percentage gap between last measurement and fitted PSO asymptote (Reviewer 2, #8)
qe_fit = res["kinetics"]["PSO"]["params"][0]
res["gap"] = dict(q170=float(q_kin[-1]), qe_fit=qe_fit,
                  measured_below_fit_pct=100 * (qe_fit - q_kin[-1]) / qe_fit,
                  fit_above_measured_pct=100 * (qe_fit - q_kin[-1]) / q_kin[-1])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
for arm in ("kinetics", "isotherm_qe_vs_Ce", "removal_vs_C0"):
    print(f"\n== {arm}")
    for k, v in res[arm].items():
        s = f"  {k:32s} train R2={v['train_R2']:.4f}  LOO R2={v['LOO_R2']:.4f}  LOO MAE={v['LOO_MAE']:.4f}"
        if "extrap_MAE" in v:
            s += f"  extrap MAE={v['extrap_MAE']:.4f}"
        if "AICc" in v:
            s += f"  AICc={v['AICc']:.2f}  params={np.round(v['params'], 5).tolist()}"
        print(s)
print("\nlinearised:", json.dumps(res["linearised"], indent=0))
print("gap:", res["gap"])
