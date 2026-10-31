"""Unified surface q(t, C0, V/m) = PSO approach to a Freundlich + mass-balance equilibrium.

Governing equations (all symbols as in the manuscript; V/m in L/g):
    q_e   = K_f * C_e^(1/n)                     (Freundlich)
    C_e   = C0 - q_e / (V/m)                    (mass balance)
    q(t)  = k2 q_e^2 t / (1 + k2 q_e t)         (integrated PSO)
Fitted parameters: K_f, n, k2 (three).  Inputs: t, C0, V/m.  q_e is solved, not fitted.

Two estimation routes are compared:
  sequential (as submitted): K_f, n from the isotherm arm treated as equilibrium;
             k2 from the kinetics arm alone (its own q_e discarded);
             isotherm points predicted as q_e (t -> infinity).
  joint     : K_f, n, k2 fitted together to all 14 points, every point modelled
             at its actual contact time (isotherm arm at t = 170 min).
Both are scored by in-sample R2 and by 14-fold leave-one-out CV, and their parameter
and prediction uncertainty is estimated by an arm-stratified residual bootstrap.
Writes results/revision_unified_surface.json
"""
import os, sys, json, warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from wcf_biofilm import classical as C
from wcf_biofilm.data import (t_kin, q_kin, C0_KIN, VM_KIN, C0_iso, q_iso, Ce_iso, VM_ISO,
                              T_ALL, C_ALL, VM_ALL, Q_ALL, ARM_ALL, T_ISO)

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                   "revision_unified_surface.json")
KIN = ARM_ALL == "kinetics"
ISO = ~KIN
# operating points never measured, used to compare predictive spread
TEST_POINTS = {"t60_C30_10gL": (60.0, 30.0, 0.1),
               "t170_C40_5gL": (170.0, 40.0, 0.2),
               "t60_C40_24gL": (60.0, 40.0, VM_KIN)}


# ---------------------------------------------------------------- sequential
def fit_sequential(t, C0, vm, q, arm):
    k, i = arm == "kinetics", arm == "isotherm"
    Ce = C0[i] - q[i] / vm[i]
    (Kf, n), _ = C.fit_isotherm("Freundlich", Ce, q[i])
    (_, k2), _ = C.fit_kinetic("PSO", t[k], q[k])
    return np.array([Kf, n, k2])


def predict_sequential(p, t, C0, vm, arm):
    Kf, n, k2 = p
    out = np.empty(len(t))
    for j in range(len(t)):
        qe = C.solve_qe(C0[j], vm[j], Kf, n)
        out[j] = qe if arm[j] == "isotherm" else C.pso(t[j], qe, k2)
    return out


# ---------------------------------------------------------------- joint
def fit_joint(t, C0, vm, q, arm=None, p0=(0.0124, 0.632, 0.0252)):
    p, _ = C.fit_surface_joint(t, C0, vm, q, p0=p0)
    return p


def predict_joint(p, t, C0, vm, arm=None):
    return C.surface(t, C0, vm, *p)


ROUTES = {"sequential": (fit_sequential, predict_sequential),
          "joint": (fit_joint, predict_joint)}


def scores(y, yh):
    return dict(R2_all=C.r2(y, yh), R2_kin=C.r2(y[KIN], yh[KIN]), R2_iso=C.r2(y[ISO], yh[ISO]),
                MAE_all=C.mae(y, yh), MAE_kin=C.mae(y[KIN], yh[KIN]), MAE_iso=C.mae(y[ISO], yh[ISO]))


def test_preds(p, route):
    _, pr = ROUTES[route]
    out = {}
    for key, (t, c, v) in TEST_POINTS.items():
        out[key] = float(pr(p, np.array([t]), np.array([c]), np.array([v]),
                            np.array(["kinetics"]))[0])
    out["qe_C40_24gL"] = C.solve_qe(C0_KIN, VM_KIN, p[0], p[1])
    return out


res = {}
rng = np.random.default_rng(2026)
for route, (fit, pred) in ROUTES.items():
    p = fit(T_ALL, C_ALL, VM_ALL, Q_ALL, ARM_ALL)
    yh = pred(p, T_ALL, C_ALL, VM_ALL, ARM_ALL)
    r = dict(params=dict(Kf=p[0], n=p[1], one_over_n=1 / p[1], k2=p[2]),
             in_sample=scores(Q_ALL, yh), fit_pred=yh.tolist(), test=test_preds(p, route))
    # leave-one-out over the 14 points
    loo = np.empty(len(Q_ALL))
    for i in range(len(Q_ALL)):
        m = np.arange(len(Q_ALL)) != i
        pi = fit(T_ALL[m], C_ALL[m], VM_ALL[m], Q_ALL[m], ARM_ALL[m])
        loo[i] = pred(pi, T_ALL[i:i + 1], C_ALL[i:i + 1], VM_ALL[i:i + 1], ARM_ALL[i:i + 1])[0]
    r["LOO"] = scores(Q_ALL, loo)
    r["LOO_pred"] = loo.tolist()
    # arm-stratified residual bootstrap
    resid = Q_ALL - yh
    B, boot_p, boot_t = 1000, [], {k: [] for k in r["test"]}
    for b in range(B):
        e = np.empty_like(resid)
        for msk in (KIN, ISO):
            e[msk] = rng.choice(resid[msk], size=msk.sum(), replace=True)
        e[0] = 0.0                       # q(t=0) = 0 is exact, not a measurement error
        try:
            pb = fit(T_ALL, C_ALL, VM_ALL, np.clip(yh + e, 0, None), ARM_ALL, ) if route == "sequential" \
                else fit(T_ALL, C_ALL, VM_ALL, np.clip(yh + e, 0, None), ARM_ALL, p0=tuple(p))
        except Exception:
            continue
        boot_p.append(pb)
        for k, v in test_preds(pb, route).items():
            boot_t[k].append(v)
    boot_p = np.array(boot_p)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    r["bootstrap"] = dict(B_ok=len(boot_p),
                          Kf_CI=ci(boot_p[:, 0]), n_CI=ci(boot_p[:, 1]), k2_CI=ci(boot_p[:, 2]),
                          param_corr=np.corrcoef(np.log(boot_p).T).round(3).tolist(),
                          test_CI={k: ci(v) for k, v in boot_t.items()})
    res[route] = r

json.dump(res, open(OUT, "w"), indent=1, default=float)
for route, r in res.items():
    print(f"\n== {route}")
    print("  params", {k: round(v, 5) for k, v in r["params"].items()})
    print("  in-sample", {k: round(v, 4) for k, v in r["in_sample"].items()})
    print("  LOO      ", {k: round(v, 4) for k, v in r["LOO"].items()})
    print("  test     ", {k: round(v, 4) for k, v in r["test"].items()})
    print("  boot CI  ", {k: np.round(v, 4).tolist() for k, v in r["bootstrap"].items() if k.endswith("CI") and k != "test_CI"})
    print("  test CI  ", {k: np.round(v, 4).tolist() for k, v in r["bootstrap"]["test_CI"].items()})
    print("  corr(logKf,logn,logk2)", r["bootstrap"]["param_corr"])
