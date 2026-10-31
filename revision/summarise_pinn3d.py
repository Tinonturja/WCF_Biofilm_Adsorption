"""Summarise 3-input PINN jobs (res3d.jsonl) -> results/revision_pinn3d_summary.json"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from wcf_biofilm.data import Q_ALL, ARM_ALL     # noqa: E402
from wcf_biofilm import classical as C          # noqa: E402

KIN = ARM_ALL == "kinetics"
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "res3d.jsonl")
rows = [json.loads(l) for l in open(src)]
out = {}
for cfg in ("raw", "nondim"):
    full = sorted([r for r in rows if r["cfg"] == cfg and r["split"] == "full"], key=lambda r: r["seed"])
    seeds = []
    for r in full:
        p = np.array(r["fit_pred"])
        seeds.append(dict(seed=r["seed"], k2=r["k2"], Kf=r["Kf"], n=r["n"],
                          R2_all=C.r2(Q_ALL, p), R2_kin=C.r2(Q_ALL[KIN], p[KIN]),
                          R2_iso=C.r2(Q_ALL[~KIN], p[~KIN]), test=r["test"],
                          final_losses=r["final_losses"]))
    rng = lambda k: [float(min(s[k] for s in seeds)), float(max(s[k] for s in seeds))]
    tests = {k: [s["test"][k] for s in seeds] for k in seeds[0]["test"]}
    loo = {}
    for s in sorted({r["seed"] for r in rows if r["cfg"] == cfg and r["split"] == "loo"}):
        L = sorted([r for r in rows if r["cfg"] == cfg and r["split"] == "loo" and r["seed"] == s],
                   key=lambda r: r["fold"])
        if len(L) == 14:
            p = np.array([r["pred"] for r in L])
            loo[s] = dict(R2_all=C.r2(Q_ALL, p), R2_kin=C.r2(Q_ALL[KIN], p[KIN]),
                          R2_iso=C.r2(Q_ALL[~KIN], p[~KIN]), MAE_all=C.mae(Q_ALL, p),
                          MAE_kin=C.mae(Q_ALL[KIN], p[KIN]), MAE_iso=C.mae(Q_ALL[~KIN], p[~KIN]),
                          pred=p.tolist())
    out[cfg] = dict(seeds=seeds, n_seeds=len(seeds),
                    k2_range=rng("k2"), Kf_range=rng("Kf"), n_range=rng("n"),
                    R2_all_range=rng("R2_all"), R2_kin_range=rng("R2_kin"), R2_iso_range=rng("R2_iso"),
                    test_values=tests,
                    test_range={k: [min(v), max(v)] for k, v in tests.items()},
                    test_rel_spread_pct={k: 100 * (max(v) - min(v)) / np.mean(v) for k, v in tests.items()},
                    LOO=loo)
json.dump(out, open(os.path.join(HERE, "..", "results", "revision_pinn3d_summary.json"), "w"), indent=1)
for cfg, o in out.items():
    print(f"\n== {cfg}  ({o['n_seeds']} seeds)")
    print("  k2", np.round(o["k2_range"], 5), " Kf", np.round(o["Kf_range"], 5), " n", np.round(o["n_range"], 3))
    print("  in-sample R2 all/kin/iso", np.round(o["R2_all_range"], 4), np.round(o["R2_kin_range"], 4), np.round(o["R2_iso_range"], 4))
    print("  test ranges", {k: np.round(v, 3).tolist() for k, v in o["test_range"].items()})
    print("  test spread %", {k: round(v, 1) for k, v in o["test_rel_spread_pct"].items()})
    for s, l in o["LOO"].items():
        print(f"  LOO seed {s}: R2 all={l['R2_all']:.4f} kin={l['R2_kin']:.4f} iso={l['R2_iso']:.4f}  MAE all={l['MAE_all']:.4f} kin={l['MAE_kin']:.4f} iso={l['MAE_iso']:.4f}")
