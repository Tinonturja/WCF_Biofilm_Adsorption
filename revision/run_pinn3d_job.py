"""One 3-input PINN fit = one job (resumable). Usage:
    WCF_WALL_BUDGET=150 python run_pinn3d_job.py CFG SEED SPLIT FOLD
CFG   : raw | nondim
SPLIT : full (all 14 points) | loo (FOLD = held-out index 0..13)
Prints one JSON line when training is complete; exits 75 if checkpointed early.
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from wcf_biofilm.pinn_surface_3d import SurfacePINN          # noqa: E402
from wcf_biofilm.data import T_ALL, C_ALL, VM_ALL, Q_ALL, VM_KIN  # noqa: E402

torch.set_num_threads(1)
cfg, seed, split, fold = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
idx = np.arange(len(Q_ALL))
tr = idx if split == "full" else idx[idx != fold]
os.makedirs(os.path.join(HERE, "ckpt3d"), exist_ok=True)
ck = os.path.join(HERE, "ckpt3d", f"{cfg}_{seed}_{split}_{fold}.ckpt")

m = SurfacePINN(n_hidden=32, n_layers=3, epochs=15000, seed=seed, scaling=cfg, ckpt_path=ck)
m.fit(T_ALL[tr], C_ALL[tr], VM_ALL[tr], Q_ALL[tr], verbose=False)
if not m.completed:
    sys.exit(75)
TEST = {"t60_C30_10gL": (60.0, 30.0, 0.1), "t170_C40_5gL": (170.0, 40.0, 0.2),
        "t60_C40_24gL": (60.0, 40.0, VM_KIN)}
out = dict(cfg=cfg, seed=seed, split=split, fold=fold, k2=m.k2_, Kf=m.Kf_, n=m.n_,
           final_losses=m.history[-1][1:].tolist())
if split == "full":
    out["fit_pred"] = m.predict(T_ALL, C_ALL, VM_ALL).tolist()
    out["test"] = {k: float(m.predict([a], [b], [c])[0]) for k, (a, b, c) in TEST.items()}
else:
    out["pred"] = float(m.predict(T_ALL[fold:fold + 1], C_ALL[fold:fold + 1],
                                  VM_ALL[fold:fold + 1])[0])
print(json.dumps(out))
