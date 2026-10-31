"""One 1-D PINN fit = one job. Usage: python run_pinn1d_job.py CFG SEED SPLIT FOLD
CFG   : key of CONFIGS below
SPLIT : loo     (FOLD = index of held-out point, 0..8)
        extrap  (train on t <= 110 min, predict t = 140, 170; FOLD ignored)
        full    (train on all 9 points; FOLD ignored)
Prints one JSON line.
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from wcf_biofilm.data import t_kin, q_kin          # noqa: E402
from wcf_biofilm.pinn_torch_1d import PSO_PINN      # noqa: E402

torch.set_num_threads(1)

CONFIGS = {
    "raw_l1":      dict(scaling="raw", lam=1.0),      # as submitted
    "raw_l0":      dict(scaling="raw", lam=0.0),      # same network, physics off
    "nd_l0":       dict(scaling="nondim", lam=0.0),   # fixed architecture, physics off
    "nd_l0.1":     dict(scaling="nondim", lam=0.1),
    "nd_l1":       dict(scaling="nondim", lam=1.0),
    "nd_l10":      dict(scaling="nondim", lam=10.0),
    "nd_l100":     dict(scaling="nondim", lam=100.0),
}

cfg, seed, split, fold = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
n = len(t_kin)
if split == "loo":
    tr, te = [j for j in range(n) if j != fold], [fold]
elif split == "extrap":
    tr, te = list(range(n - 2)), [n - 2, n - 1]
else:
    tr, te = list(range(n)), list(range(n))

t0 = time.time()
m = PSO_PINN(seed=seed, epochs=8000, **CONFIGS[cfg]).fit(t_kin[tr], q_kin[tr])
out = dict(cfg=cfg, seed=seed, split=split, fold=fold, test_idx=te,
           pred=m.predict(t_kin[te]), qe=m.qe_, k2=m.k2_,
           data_loss=m.final_data_loss_, phys_loss=m.final_phys_loss_,
           sec=round(time.time() - t0, 1))
if split == "full":
    tf = np.linspace(0, 180, 181)
    out["curve_t"] = tf.tolist()
    out["curve_q"] = m.predict(tf)
print(json.dumps(out))
