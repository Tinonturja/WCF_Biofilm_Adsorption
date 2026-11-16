"""Compare a re-run of the revision analysis with the committed results.

Usage: python compare_reproduction.py RERUN_RESULTS_DIR RERUN_REVISION_DIR
Writes a Markdown report to stdout. For every summary file it prints the largest
absolute difference; for the PINN grids it compares each fit (job by job) and the
headline statistics quoted in the manuscript.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "results")
new_res, new_rev = sys.argv[1], sys.argv[2]


def numbers(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict):
        for k in a:
            if k in b:
                numbers(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            numbers(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        out.append((abs(a - b), path))
    return out


def jobs(p):
    return {(d["cfg"], d["seed"], d["split"], d["fold"]): d for d in map(json.loads, open(p))}


print("# Reproduction check\n")
print("| File | Numbers compared | Largest difference | Where |\n| --- | --- | --- | --- |")
for f in ("revision_classical_ml.json", "revision_unified_surface.json"):
    d = numbers(json.load(open(os.path.join(REF, f))), json.load(open(os.path.join(new_res, f))))
    m = max(d)
    print(f"| {f} | {len(d)} | {m[0]:.3g} | {m[1]} |")

for name, ref, new, fields in (
        ("1-D PINN", "revision_pinn1d_jobs.jsonl", "res1d.jsonl", ("pred", "qe", "k2")),
        ("3-input PINN", "revision_pinn3d_jobs.jsonl", "res3d.jsonl", ("pred", "k2", "Kf", "n", "test"))):
    a, b = jobs(os.path.join(REF, ref)), jobs(os.path.join(new_rev, new))
    keys = sorted(set(a) & set(b))
    print(f"\n## {name}: {len(keys)} of {len(a)} fits compared\n")
    print("| Configuration | Quantity | Largest difference |\n| --- | --- | --- |")
    worst = {}
    for k in keys:
        for f in fields:
            if f not in a[k]:
                continue
            x, y = a[k][f], b[k][f]
            if isinstance(x, dict):
                x, y = list(x.values()), list(y.values())
            x = np.atleast_1d(x); y = np.atleast_1d(y)
            worst[(k[0], f)] = max(worst.get((k[0], f), 0.0), float(np.max(np.abs(x - y))))
    for (c, f), v in sorted(worst.items()):
        print(f"| {c} | {f} | {v:.3g} |")
