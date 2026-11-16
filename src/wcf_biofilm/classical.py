"""Classical kinetic / isotherm equations, fitting helpers and information criteria.

Units: t [min], q [mg/g], C [mg/L], k1 [1/min], k2 [g/(mg min)], V/m [L/g],
K_f [(mg/g)(L/mg)^(1/n)], n [dimensionless], q_max [mg/g], K_L [L/mg].
Freundlich is written q_e = K_f * C_e**(1/n), as in the manuscript.
The isotherm-arm data are uptakes at t = 170 min, so fitted isotherm constants are
apparent values at 170 min, not verified equilibrium constants.
"""
import numpy as np
from scipy.optimize import curve_fit, brentq, least_squares


# ---------------------------------------------------------------- kinetics
def pso(t, qe, k2):
    t = np.asarray(t, float)
    return qe ** 2 * k2 * t / (1.0 + qe * k2 * t)


def pfo(t, qe, k1):
    return qe * (1.0 - np.exp(-k1 * np.asarray(t, float)))


KINETIC_MODELS = {"PSO": (pso, (0.9, 0.02)), "PFO": (pfo, (0.75, 0.02))}


def fit_kinetic(name, t, q):
    f, p0 = KINETIC_MODELS[name]
    p, cov = curve_fit(f, t, q, p0=p0, bounds=([1e-6, 1e-6], [50.0, 10.0]),
                       maxfev=100000)
    return p, np.sqrt(np.diag(cov))


# ---------------------------------------------------------------- isotherms
def freundlich(Ce, Kf, n):
    return Kf * np.asarray(Ce, float) ** (1.0 / n)


def langmuir(Ce, qmax, KL):
    Ce = np.asarray(Ce, float)
    return qmax * KL * Ce / (1.0 + KL * Ce)


ISOTHERM_MODELS = {"Freundlich": (freundlich, (0.01, 0.6), ([1e-8, 0.05], [10.0, 20.0])),
                   "Langmuir": (langmuir, (5.0, 0.01), ([1e-6, 1e-8], [1e4, 100.0]))}


def fit_isotherm(name, Ce, q):
    f, p0, b = ISOTHERM_MODELS[name]
    p, cov = curve_fit(f, Ce, q, p0=p0, bounds=b, maxfev=200000)
    return p, np.sqrt(np.diag(cov))


# ------------------------------------------- Freundlich + mass balance + PSO
def solve_qe(C0, vm, Kf, n):
    """q_e such that q_e = Kf*Ce^(1/n) and Ce = C0 - q_e/vm (vm = V/m in L/g)."""
    qmax = C0 * vm * (1 - 1e-12)

    def g(qe):
        return qe - Kf * max(C0 - qe / vm, 0.0) ** (1.0 / n)
    if g(qmax) < 0:          # adsorbent would take up everything: no interior root
        return qmax
    return brentq(g, 0.0, qmax, xtol=1e-12)


def surface(t, C0, vm, Kf, n, k2):
    """q(t, C0, V/m): PSO approach to the mass-balance Freundlich equilibrium."""
    t, C0, vm = np.broadcast_arrays(np.asarray(t, float), np.asarray(C0, float),
                                    np.asarray(vm, float))
    out = np.empty(t.shape)
    for idx in np.ndindex(t.shape):
        qe = solve_qe(C0[idx], vm[idx], Kf, n)
        out[idx] = pso(t[idx], qe, k2)
    return out


def fit_surface_joint(t, C0, vm, q, p0=(0.0124, 0.632, 0.0252)):
    """Fit (Kf, n, k2) jointly to all points, every point modelled as q(t)."""
    def res(lp):
        Kf, n, k2 = np.exp(lp)
        return surface(t, C0, vm, Kf, n, k2) - q
    r = least_squares(res, np.log(p0), method="lm", max_nfev=20000)
    return np.exp(r.x), r


# ---------------------------------------------------------------- metrics
def r2(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    return 1.0 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def mae(y, yh):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yh))))


def rmse(y, yh):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yh)) ** 2)))


def aicc(y, yh, k):
    """Small-sample corrected AIC for least squares with k fitted parameters."""
    n = len(y)
    rss = float(((np.asarray(y) - np.asarray(yh)) ** 2).sum())
    aic = n * np.log(rss / n) + 2 * k
    return aic + 2 * k * (k + 1) / (n - k - 1)
