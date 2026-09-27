# -*- coding: utf8 -*-
# Copyright 2012-2026 Panobbgo Contributors
"""Counterfactual labels and probe features for a learned arm selector (roadmap §4 A step 1, DISCOVERY §70).

For every task — (instance, seed, q) of a family preset — a shared probe of
``probe·d`` Latin-hypercube points is evaluated once, its features computed,
and every arm of the menu (``panobbgo.selector_data.ARM_MENU_V0``) continued
from that probe archive to the budget on the virtual clock.  The label is each
arm's regret against the best arm of the task (``selector_data.add_labels``).
No selector is trained here.

Usage::

    uv run python benchmarks/selector_labels.py run OUT.csv.gz [preset=wide] [dims=2,5] [bm=100] \\
        [qs=1,4] [seeds=3] [probe=10] [ninst=3] [diag=1] [jobs=4]
    uv run python benchmarks/selector_labels.py analyze IN.csv.gz [OUT.md]

``run``
    One row per task (``selector_data.run_task`` + labels), written as a gzipped
    CSV, rewritten every 20 finished tasks.  ``seeds`` is a count of the decision
    roster (42, 7, 1234, ...) or an explicit comma list; ``diag=1`` also runs the
    cold forms of RoundRobin_CMAES / RoundRobin_COBYQA (``DIAGNOSTIC_ARMS``,
    unlabelled).  ``jobs`` worker processes (niced; at most half the effective
    CPU limit on a shared machine, ``doc/dev/environment.md``); the rows do not
    depend on it.
``analyze``
    The sanity analysis of §70: the label distribution per arm, the oracle vs
    single-best-arm gap (the headroom a perfect selector has), and which probe
    features go with which arm winning.  Markdown to stdout (and ``OUT.md``).
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import panobbgo.fp_pin  # noqa: F401  # first: pins the OpenBLAS / numpy kernels before numpy loads

import numpy as np

ROSTER = (42, 7, 1234, 2025, 3, 11, 99, 123, 777, 2024, 31337, 555)


def _parse(argv: Sequence[str]) -> Dict[str, str]:
    opts = {}
    for a in argv:
        if "=" not in a:
            raise SystemExit(f"expected key=value, got {a!r}\n{__doc__}")
        k, v = a.split("=", 1)
        opts[k] = v
    return opts


def _seeds(spec: str) -> List[int]:
    spec = spec.strip()
    if spec.isdigit():
        return list(ROSTER[: int(spec)])
    return [int(s) for s in spec.split(",") if s.strip()]


def _write(rows: List[Dict[str, Any]], path: str) -> None:
    import pandas as pd

    df = pd.DataFrame([r for r in rows if r is not None])
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, compression="gzip", float_format="%.6g")
    import os

    os.replace(tmp, path)


def cmd_run(out: str, opts: Dict[str, str]) -> None:
    from panobbgo import local_run
    from panobbgo.harness_families import make_wide_battery, make_families_battery, make_shapes_battery
    from panobbgo.selector_data import ARM_MENU_V0, DIAGNOSTIC_ARMS, PROBE_PER_DIM, add_labels, run_task

    local_run.be_nice()
    preset = opts.get("preset", "wide")
    make = {"wide": make_wide_battery, "free": make_families_battery, "shapes": make_shapes_battery}[preset]
    dims = tuple(int(d) for d in opts.get("dims", "2,5").split(","))
    bm = int(opts.get("bm", "100"))
    qs = [int(q) for q in opts.get("qs", "1,4").split(",")]
    seeds = _seeds(opts.get("seeds", "3"))
    probe = int(opts.get("probe", str(PROBE_PER_DIM)))
    ninst = int(opts.get("ninst", "3"))
    jobs = int(opts.get("jobs", "4"))
    arms = list(ARM_MENU_V0) + (list(DIAGNOSTIC_ARMS) if opts.get("diag", "1") == "1" else [])
    instances = list(make(dims=dims, n_instances=ninst))
    tasks = [
        dict(problem=p, base_seed=s, budget_multiplier=bm, q=q, probe_per_dim=probe, arms=arms)
        for _n, p in instances
        for s in seeds
        for q in qs
    ]
    print(
        f"{len(tasks)} tasks ({len(instances)} instances x {len(seeds)} seeds x {len(qs)} q), "
        f"{len(arms)} arms each, preset={preset} dims={dims} bm={bm} probe={probe}*d jobs={jobs}",
        flush=True,
    )
    rows: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
    t0 = time.time()
    done = [0]

    def on_done(i: int, row: Dict[str, Any]) -> None:
        rows[i] = add_labels(row)
        done[0] += 1
        n = done[0]
        el = time.time() - t0
        print(
            f"  [{n:>4d}/{len(tasks)}] {row['family']:<20s} d{row['dim']} i{row['instance']} s{row['seed']:<5d} "
            f"q{row['q']} best={row['best_arm']}  {el / 60:.1f} min, eta {el / n * (len(tasks) - n) / 60:.1f} min",
            flush=True,
        )
        if n % 20 == 0:
            _write([r for r in rows if r is not None], out)

    with local_run.TaskPool(jobs) as pool:
        pool.map(run_task, tasks, on_done=on_done)
    _write([r for r in rows if r is not None], out)
    print(f"wrote {out} ({len(tasks)} rows) in {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def _fmt(v: float, nd: int = 3) -> str:
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def _cluster_boot(values: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple:
    """95 % percentile CI of the mean, resampling whole clusters (instances)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    groups = [values[clusters == c] for c in uniq]
    sums = np.array([g.sum() for g in groups])
    counts = np.array([g.size for g in groups])
    idx = rng.integers(0, uniq.size, size=(n_boot, uniq.size))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze(df: Any, arms: Sequence[str]) -> str:
    """The §70 sanity analysis as markdown."""
    import pandas as pd
    from scipy.stats import spearmanr

    out: List[str] = []
    p = out.append
    df = df[df["best_arm"].notna()].copy()
    df["inst_key"] = df["family"] + "_d" + df["dim"].astype(str) + "_i" + df["instance"].astype(str)
    df["cell"] = "d" + df["dim"].astype(str) + " q" + df["q"].astype(str)
    S = df[[f"{a}:score" for a in arms]].to_numpy(dtype=float)
    R = df[[f"{a}:regret" for a in arms]].to_numpy(dtype=float)
    short = {a: a.replace("Blocks_warm_CMAES_JSO", "Blocks").replace("RoundRobin_", "RR_") for a in arms}
    p(f"{len(df)} labelled tasks, {df['inst_key'].nunique()} instances, cells: {sorted(df['cell'].unique())}\n")

    # 1. label distribution
    p("### Label distribution (regret vs the best arm of the task; score: AOCC at q = 1, aocc_time at q = 4)\n")
    p("| arm | mean score | mean regret | median | p90 | max | wins (regret 0) | regret < 0.01 | EndedEarly |")
    p("|---|---|---|---|---|---|---|---|---|")
    for j, a in enumerate(arms):
        r = R[:, j]
        wins = (df["best_arm"] == a).mean()
        early = df[f"{a}:error"].astype(str).str.startswith("EndedEarly").mean()
        p(
            f"| {short[a]} | {_fmt(S[:, j].mean())} | {_fmt(r.mean())} | {_fmt(np.median(r))} | "
            f"{_fmt(np.percentile(r, 90))} | {_fmt(r.max())} | {wins:.0%} | {(r < 0.01).mean():.0%} | {early:.0%} |"
        )
    p("")
    p("Per cell, mean regret (wins):\n")
    p("| cell | " + " | ".join(short[a] for a in arms) + " |")
    p("|---" * (len(arms) + 1) + "|")
    for cell, g in df.groupby("cell"):
        cells = [f"{g[f'{a}:regret'].mean():.3f} ({(g['best_arm'] == a).mean():.0%})" for a in arms]
        p(f"| {cell} | " + " | ".join(cells) + " |")
    p("")

    # 2. headroom
    p("### Headroom: oracle vs single best arm\n")
    p(
        "SBS = the arm with the best mean score over the tasks in scope (chosen in hindsight on "
        "the same tasks); gap = oracle mean − SBS mean = the SBS's mean regret.  CIs: cluster "
        "bootstrap over instances (2000 resamples).\n"
    )
    p(
        "| scope | tasks | SBS | SBS mean | task oracle | gap (task oracle) [CI] | instance oracle gap | family oracle gap |"
    )
    p("|---|---|---|---|---|---|---|---|")

    def headroom(g: Any, label: str) -> None:
        s = g[[f"{a}:score" for a in arms]].to_numpy(dtype=float)
        sbs = int(np.argmax(s.mean(axis=0)))
        oracle = s.max(axis=1)
        gap = oracle - s[:, sbs]
        lo, hi = _cluster_boot(gap, g["inst_key"].to_numpy())
        # instance oracle: the arm best on average over the seeds of an instance (a selector cannot see seed luck)
        inst_means = g.groupby(["inst_key", "q"])[[f"{a}:score" for a in arms]].transform("mean").to_numpy()
        inst_pick = s[np.arange(len(g)), inst_means.argmax(axis=1)]
        fam_means = g.groupby(["family", "dim", "q"])[[f"{a}:score" for a in arms]].transform("mean").to_numpy()
        fam_pick = s[np.arange(len(g)), fam_means.argmax(axis=1)]
        p(
            f"| {label} | {len(g)} | {short[arms[sbs]]} | {_fmt(s[:, sbs].mean())} | {_fmt(oracle.mean())} | "
            f"**{_fmt(gap.mean())}** [{_fmt(lo)}, {_fmt(hi)}] | {_fmt((inst_pick - s[:, sbs]).mean())} | "
            f"{_fmt((fam_pick - s[:, sbs]).mean())} |"
        )

    headroom(df, "all")
    headroom(df[df["family"] != "ellipsoid"], "all, ex-ellipsoid")
    for cell, g in df.groupby("cell"):
        headroom(g, cell)
    # a context-only selector: the per-cell SBS (knows d and q, nothing else)
    s_all = df[[f"{a}:score" for a in arms]].to_numpy(dtype=float)
    cell_means = df.groupby("cell")[[f"{a}:score" for a in arms]].transform("mean").to_numpy()
    ctx_pick = s_all[np.arange(len(df)), cell_means.argmax(axis=1)]
    sbs = int(np.argmax(s_all.mean(axis=0)))
    p("")
    p(
        f"Context-only selector (the best arm per (d, q) cell, in hindsight): mean score {_fmt(ctx_pick.mean())}, "
        f"{_fmt(ctx_pick.mean() - s_all[:, sbs].mean())} above the global SBS; the task oracle is "
        f"{_fmt(s_all.max(axis=1).mean() - ctx_pick.mean())} above it.\n"
    )

    # per family: who wins
    p("### Best arm per family (mean score over dims, seeds, q; wins = share of tasks)\n")
    p("| family | best arm (mean score) | runner-up | spread best − worst | wins |")
    p("|---|---|---|---|---|")
    for fam, g in df.groupby("family"):
        m = g[[f"{a}:score" for a in arms]].mean()
        order = np.argsort(-m.to_numpy())
        wins = g["best_arm"].value_counts(normalize=True)
        wtxt = ", ".join(f"{short[a]} {v:.0%}" for a, v in wins.items())
        p(
            f"| {fam} | {short[arms[order[0]]]} ({m.iloc[order[0]]:.3f}) | {short[arms[order[1]]]} "
            f"({m.iloc[order[1]]:.3f}) | {m.max() - m.min():.3f} | {wtxt} |"
        )
    p("")

    # 3. features vs winners
    feats = [c for c in df.columns if c.startswith("f_") and df[c].nunique(dropna=True) > 1]
    p("### Features vs arms\n")
    p(
        "Spearman ρ between a probe feature and an arm's regret over all tasks (negative = the arm "
        "does better where the feature is high); the three strongest |ρ| per arm, and per feature "
        "the arm whose regret it tracks most.\n"
    )
    p("| arm | strongest features (ρ with regret) |")
    p("|---|---|")
    rho: Dict[str, Dict[str, float]] = {}
    for a in arms:
        rho[a] = {}
        for f in feats:
            x = df[f].to_numpy(dtype=float)
            ok = np.isfinite(x)
            if ok.sum() > 10:
                rho[a][f] = float(spearmanr(x[ok], df.loc[ok, f"{a}:regret"].to_numpy(dtype=float)).statistic)
        top = sorted(rho[a].items(), key=lambda kv: -abs(kv[1]))[:3]
        p(f"| {short[a]} | " + ", ".join(f"{k[2:]} {v:+.2f}" for k, v in top) + " |")
    p("")
    p("| feature | defined | " + " | ".join(short[a] for a in arms) + " |")
    p("|---" * (len(arms) + 2) + "|")
    for f in feats:
        defined = df[f].notna().mean()
        p(f"| {f[2:]} | {defined:.0%} | " + " | ".join(_fmt(rho[a].get(f), 2) for a in arms) + " |")
    p("")
    # winners: mean feature by winning arm
    key = [
        "f_flog_quad_gap",
        "f_fr2_quad",
        "f_r2_quad",
        "f_fdc",
        "f_nbc_nb_fitness_cor",
        "f_disp_10",
        "f_y_ties",
        "f_fsep_ratio",
        "f_y_skew",
    ]
    key = [k for k in key if k in df.columns]
    p("Mean feature value by winning arm (tasks it wins):\n")
    p("| winner | tasks | " + " | ".join(k[2:] for k in key) + " |")
    p("|---" * (len(key) + 2) + "|")
    for a in arms:
        g = df[df["best_arm"] == a]
        if len(g):
            p(f"| {short[a]} | {len(g)} | " + " | ".join(_fmt(g[k].astype(float).mean(), 2) for k in key) + " |")
    p("")

    # 4. diagnostics: warm vs cold
    diag = [("RoundRobin_CMAES", "RoundRobin_CMAES_cold"), ("RoundRobin_COBYQA", "RoundRobin_COBYQA_cold")]
    if all(f"{c}:aocc" in df.columns for _w, c in diag):
        p("### Warm continuation vs the registry's cold start (diagnostic arms, not labelled)\n")
        p("| arm | cell | warm | cold | warm − cold [CI] | warm better |")
        p("|---|---|---|---|---|---|")
        for w, c in diag:
            for cell, g in df.groupby("cell"):
                key_s = "aocc" if cell.endswith("q1") else "aocc_time"
                dw = g[f"{w}:{key_s}"].to_numpy(dtype=float)
                dc = g[f"{c}:{key_s}"].to_numpy(dtype=float)
                d = dw - dc
                lo, hi = _cluster_boot(d, g["inst_key"].to_numpy())
                p(
                    f"| {short[w]} | {cell} | {_fmt(dw.mean())} | {_fmt(dc.mean())} | "
                    f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {(d > 0).mean():.0%} |"
                )
        p("")
    _ = pd
    return "\n".join(out)


def cmd_analyze(path: str, out_md: Optional[str]) -> None:
    import pandas as pd

    from panobbgo.selector_data import ARM_MENU_V0

    df = pd.read_csv(path)
    text = analyze(df, ARM_MENU_V0)
    print(text)
    if out_md:
        with open(out_md, "w") as fh:
            fh.write(text + "\n")


def main(argv: Sequence[str]) -> None:
    if len(argv) < 2 or argv[0] not in ("run", "analyze"):
        raise SystemExit(__doc__)
    if argv[0] == "run":
        cmd_run(argv[1], _parse(argv[2:]))
    else:
        cmd_analyze(argv[1], argv[2] if len(argv) > 2 else None)


if __name__ == "__main__":
    main(sys.argv[1:])
