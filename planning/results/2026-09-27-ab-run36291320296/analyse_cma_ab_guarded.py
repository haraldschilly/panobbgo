"""§60 analysis: paired per-seed comparisons of the CMA-ES variants (paired_seed_stats)."""

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

from panobbgo.harness_ioh import IOHMultiSeedResult, bbob_class_of, paired_seed_stats

REF = Path(sys.argv[1])
RR = "RoundRobin_CMAES"
OPT = "Baseline_Optuna_CmaEs"
CLIP = "Baseline_Optuna_CmaEs_clip"
BIPOP = "Baseline_pycma_BIPOP"
VARIANTS = [
    "RoundRobin_CMAES_resample",
    "RoundRobin_CMAES_reflect",
    "RoundRobin_CMAES_randstart",
    "RoundRobin_CMAES_active",
    "RoundRobin_CMAES_active_guarded",
    "RoundRobin_CMAES_resample_randstart_active",
]
ACT = "RoundRobin_CMAES_active"
GRD = "RoundRobin_CMAES_active_guarded"
ALL = [RR, *VARIANTS, OPT, CLIP, BIPOP]
SHORT = {RR: "RR", OPT: "Optuna", CLIP: "Optuna_clip", BIPOP: "BIPOP"}
SHORT.update({v: v.replace("RoundRobin_CMAES_", "") for v in VARIANTS})


def load(name):
    return IOHMultiSeedResult.from_dict(json.loads((REF / name).read_text()))


def subset(res, keep, label, cell=lambda r: True):
    """A copy of ``res`` with only ``keep``'s runs on the cells ``cell`` accepts, relabelled ``label``."""
    results = []
    for one in res.results:
        runs = [dataclasses.replace(r, strategy_name=label) for r in one.runs if r.strategy_name == keep and cell(r)]
        results.append(dataclasses.replace(one, runs=runs))
    return dataclasses.replace(res, results=results)


def pair(res, a, b, cell=lambda r: True):
    """``b − a`` per seed over the cells: (mean delta, ci_low, ci_high, wins of b, n)."""
    st = paired_seed_stats(subset(res, a, "X", cell), subset(res, b, "X", cell))["X"]
    wins = sum(1 for d in st["per_seed_delta"] if d > 0)
    return st["mean_delta"], st["ci_low"], st["ci_high"], wins, st["n"], st["before_mean"], st["after_mean"]


def fmt(t):
    d, lo, hi, w, n, _, _ = t
    star = "**" if (lo > 0 or hi < 0) else ""
    return f"{star}{d:+.4f} [{lo:+.4f}, {hi:+.4f}], {w}/{n}{star}"


def means(res, cell=lambda r: True):
    out = {}
    for s in ALL:
        vals = [r.aocc for one in res.results for r in one.runs if r.strategy_name == s and cell(r)]
        out[s] = float(np.mean(vals)) if vals else float("nan")
    return out


def table(res, cuts, title):
    print(f"\n### {title}")
    for cut_name, cell in cuts:
        m = means(res, cell)
        print(f"\n{cut_name}: mean AOCC " + ", ".join(f"{SHORT[s]} {m[s]:.4f}" for s in ALL))
        print("| spec | − RR | − Optuna | − active (unguarded) |")
        print("|---|---|---|---|")
        for s in [*VARIANTS, OPT, CLIP, BIPOP]:
            vs_rr = fmt(pair(res, RR, s, cell)) if s != RR else ""
            vs_opt = fmt(pair(res, OPT, s, cell)) if s != OPT else ""
            vs_act = fmt(pair(res, ACT, s, cell)) if s == GRD else ""
            print(f"| {SHORT[s]} | {vs_rr} | {vs_opt} | {vs_act} |")


def reach(res, dim, inst, targets=(1e-1, 1e-8), by=911):
    print(f"\n### reach on MA-BBOB cell ({dim}, {inst})")
    for s in ALL:
        runs = [r for one in res.results for r in one.runs if r.strategy_name == s and r.dim == dim and r.instance == inst]
        row = []
        for t in targets:
            end = sum(1 for r in runs if r.best_fx - r.f_opt <= t)
            early = 0
            for r in runs:
                ev = [(e, f) for e, f in zip(r.trace_evals, r.trace_fx) if e <= by]
                if ev and ev[-1][1] - r.f_opt <= t:
                    early += 1
            row.append(f"{t:g}: by {by} {early}/{len(runs)}, by end {end}/{len(runs)}")
        aocc = np.mean([r.aocc for r in runs])
        print(f"{SHORT[s]:28s} AOCC {aocc:.3f}  " + " | ".join(row))


def health(res, name):
    errs = [(r.strategy_name, r.error) for one in res.results for r in one.runs if r.error]
    n = sum(len(one.runs) for one in res.results)
    print(f"{name}: seeds {res.base_seeds}, {n} runs, {len(errs)} with error, fp_env_id {res.fp_env_id}")
    for e in errs[:5]:
        print("   ", e)


if (REF / "ref_ioh_standard_cma_ab.json").exists():
    ioh = load("ref_ioh_standard_cma_ab.json")
    health(ioh, "ioh standard")
    table(
        ioh,
        [
            ("all", lambda r: True),
            ("d=2", lambda r: r.dim == 2),
            ("d=5", lambda r: r.dim == 5),
            ("cell (5,2)", lambda r: r.dim == 5 and r.instance == 2),
            ("without (5,2)", lambda r: not (r.dim == 5 and r.instance == 2)),
        ],
    "IOH standard (MA-BBOB, 500·d)",
    )
    reach(ioh, 5, 2)
    reach(ioh, 2, 2)

for bm in (200, 500):
    f = f"ref_ioh_bbob_cma_ab_b{bm}.json"
    if not (REF / f).exists():
        print("missing", f)
        continue
    res = load(f)
    health(res, f)
    cuts = [("all", lambda r: True), ("all without f5", lambda r: r.fid != 5)]
    cuts += [(f"d={d}", (lambda d: lambda r: r.dim == d)(d)) for d in (2, 5, 10)]
    cuts += [(f"d={d} without f5", (lambda d: lambda r: r.dim == d and r.fid != 5)(d)) for d in (2, 5, 10)]
    cuts += [("separable without f5", lambda r: bbob_class_of(r.fid) == "separable" and r.fid != 5)]
    cuts += [(f"f5 d={d}", (lambda d: lambda r: r.fid == 5 and r.dim == d)(d)) for d in (2, 5, 10)]
    classes = sorted({bbob_class_of(r.fid) for one in res.results for r in one.runs})
    cuts += [(f"class {c}", (lambda c: lambda r: bbob_class_of(r.fid) == c)(c)) for c in classes]
    table(res, cuts, f"BBOB 24 fids, d 2/5/10, inst 0/1, {bm}·d")
    # per-fid for the headline variants vs RR, to see where effects sit
    print(f"\nper fid (b{bm}), delta vs RR, all dims pooled:")
    fids = sorted({r.fid for one in res.results for r in one.runs})
    for s in [ACT, GRD, OPT]:
        cells = []
        for fid in fids:
            d, lo, hi, w, n, _, _ = pair(res, RR, s, (lambda f: lambda r: r.fid == f)(fid))
            mark = "+" if lo > 0 else ("-" if hi < 0 else " ")
            cells.append(f"f{fid}:{d:+.3f}{mark}")
        print(f"  {SHORT[s]:28s} " + " ".join(cells))
