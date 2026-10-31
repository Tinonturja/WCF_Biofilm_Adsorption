"""Summarise the 1-D PINN job results (res1d.jsonl) -> results/revision_pinn1d_summary.json"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from wcf_biofilm.data import q_kin            # noqa: E402
from wcf_biofilm import classical as C         # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "res1d.jsonl")
rows = [json.loads(l) for l in open(src)]
CFGS = ["raw_l1", "raw_l0", "nd_l0", "nd_l0.1", "nd_l1", "nd_l10", "nd_l100"]
summary = {}
for c in CFGS:
    per_seed = []
    for s in range(5):
        R = [r for r in rows if r["cfg"] == c and r["seed"] == s]
        loo = sorted([r for r in R if r["split"] == "loo"], key=lambda r: r["fold"])
        if len(loo) != 9:
            continue
        p = np.array([r["pred"][0] for r in loo])
        ex = [r for r in R if r["split"] == "extrap"][0]
        full = [r for r in R if r["split"] == "full"][0]
        per_seed.append(dict(seed=s, LOO_R2=C.r2(q_kin, p), LOO_MAE=C.mae(q_kin, p),
                             LOO_pred=p.tolist(),
                             extrap_MAE=C.mae(q_kin[-2:], ex["pred"]), extrap_pred=ex["pred"],
                             qe=full["qe"], k2=full["k2"],
                             data_loss=full["data_loss"], phys_loss=full["phys_loss"],
                             fold_qe=[r["qe"] for r in loo], fold_k2=[r["k2"] for r in loo]))
    agg = lambda k: (float(np.mean([d[k] for d in per_seed])), float(np.std([d[k] for d in per_seed], ddof=1)),
                     float(np.min([d[k] for d in per_seed])), float(np.max([d[k] for d in per_seed])))
    summary[c] = dict(n_seeds=len(per_seed), per_seed=per_seed,
                      **{f"{k}_mean_sd_min_max": agg(k) for k in
                         ("LOO_R2", "LOO_MAE", "extrap_MAE", "qe", "k2", "data_loss", "phys_loss")})
    # ensemble (mean over seeds) prediction
    ens = np.mean([d["LOO_pred"] for d in per_seed], axis=0)
    summary[c]["ensemble_LOO_R2"] = C.r2(q_kin, ens)
    summary[c]["ensemble_LOO_MAE"] = C.mae(q_kin, ens)

out = os.path.join(HERE, "..", "results", "revision_pinn1d_summary.json")
json.dump(summary, open(out, "w"), indent=1)
print(f"{'config':9s} {'LOO R2 mean±sd [min,max]':34s} {'LOO MAE mean±sd':18s} {'extrap MAE':16s} {'qe (full)':18s} {'k2 (full)':20s} data/phys loss")
for c in CFGS:
    s = summary[c]
    f = lambda k, d=4: "{:.{d}f}±{:.{d}f}".format(*s[k + "_mean_sd_min_max"][:2], d=d)
    mm = s["LOO_R2_mean_sd_min_max"]
    print(f"{c:9s} {f('LOO_R2')} [{mm[2]:.4f},{mm[3]:.4f}]   {f('LOO_MAE')}   {f('extrap_MAE')}   {f('qe')}   {f('k2',5)}   "
          f"{s['data_loss_mean_sd_min_max'][0]:.2e}/{s['phys_loss_mean_sd_min_max'][0]:.2e}  ens R2={s['ensemble_LOO_R2']:.4f}")
