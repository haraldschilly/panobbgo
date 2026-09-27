"""Per-family AOCC means of the wide smoke run (core + trq units)."""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

src = sys.argv[1]
SHOW = [
    "RoundRobin_CMAES",
    "Blocks_warm_CMAES_JSO",
    "RoundRobin_TRQ",
    "Blocks_warm_CMAES_JSO_TRQ",
    "RoundRobin_COBYQA",
    "Baseline_pycma_IPOP",
    "Baseline_NGOpt",
    "Baseline_Optuna_CmaEs",
    "Baseline_Optuna_TPE",
    "Baseline_PyBOBYQA",
]
SHORT = {
    "RoundRobin_CMAES": "RR_CMA",
    "Blocks_warm_CMAES_JSO": "Blocks",
    "RoundRobin_TRQ": "RR_TRQ",
    "Blocks_warm_CMAES_JSO_TRQ": "Bl3_TRQ",
    "RoundRobin_COBYQA": "COBYQA",
    "Baseline_pycma_IPOP": "IPOP",
    "Baseline_NGOpt": "NGOpt",
    "Baseline_Optuna_CmaEs": "OptCMA",
    "Baseline_Optuna_TPE": "TPE",
    "Baseline_PyBOBYQA": "PyBOBYQA",
}
# (dim, q, family, strategy) -> list of aocc (seed x instance)
data = defaultdict(list)
fams = []
override = sys.argv[2] if len(sys.argv) > 2 else None
paths = sorted(glob.glob(f"{src}/*.wide.*.json")) + (sorted(glob.glob(f"{override}/*.wide.*.json")) if override else [])
over_fams = set()
if override:
    for path in glob.glob(f"{override}/*.wide.*.json"):
        over_fams |= {r["problem_kind"] for r in json.load(open(path))["result"]["runs"]}
for path in paths:
    d = json.load(open(path))
    metric = "aocc" if d["q"] == 1 else "aocc_time"
    from_override = override is not None and path.startswith(override)
    for r in d["result"]["runs"]:
        if r["problem_kind"] in over_fams and not from_override:
            continue
        v = r.get(metric)
        v = 0.0 if v is None or r.get("error") and "EndedEarly" not in str(r.get("error")) else v
        data[(d["dim"], d["q"], r["problem_kind"], r["strategy_name"])].append(v)
        if r["problem_kind"] not in fams:
            fams.append(r["problem_kind"])
from panobbgo.harness_families import WIDE_FAMILIES

fams = [c.name() for c in WIDE_FAMILIES if c.name() in fams]
strats = sorted({k[3] for k in data})
missing = [s for s in SHOW if s not in strats]
if missing:
    print("missing:", missing)
cols = [s for s in SHOW if s in strats]
for dim in (2, 5):
    for q in (1, 4):
        metric = "AOCC" if q == 1 else "aocc_time"
        print(f"\n#### d = {dim}, 100·d, q = {q} ({metric}, mean over 5 seeds x 3 instances)\n")
        print("| family | " + " | ".join(SHORT[s] for s in cols) + " | best | spread |")
        print("|---|" + "---|" * (len(cols) + 2))
        means_all = defaultdict(list)
        for fam in fams:
            row = []
            for s in cols:
                vals = data.get((dim, q, fam, s))
                m = float(np.mean(vals)) if vals else float("nan")
                row.append(m)
                means_all[s].append(m)
            arr = np.array(row)
            best = cols[int(np.nanargmax(arr))]
            cells = []
            for s, m in zip(cols, row):
                txt = f"{m:.3f}"
                cells.append(f"**{txt}**" if s == best else txt)
            print(f"| {fam} | " + " | ".join(cells) + f" | {SHORT[best]} | {np.nanmax(arr) - np.nanmin(arr):.2f} |")
        mean = [float(np.mean(means_all[s])) for s in cols]
        print("| **mean** | " + " | ".join(f"{m:.3f}" for m in mean) + f" | {SHORT[cols[int(np.argmax(mean))]]} | |")
        exl = [float(np.mean([v for f, v in zip(fams, means_all[s]) if f != "ellipsoid"])) for s in cols]
        print("| mean ex-ellipsoid | " + " | ".join(f"{m:.3f}" for m in exl) + f" | {SHORT[cols[int(np.argmax(exl))]]} | |")
        # how many families does each arm win
        wins = defaultdict(int)
        for i, fam in enumerate(fams):
            vals = [means_all[s][i] for s in cols]
            wins[cols[int(np.nanargmax(vals))]] += 1
        print("\nfamilies won: " + ", ".join(f"{SHORT[s]} {n}" for s, n in sorted(wins.items(), key=lambda t: -t[1])))
