"""Phase A: tune one optimizer ALONE on the standard IOH battery.

Each arm of a portfolio can only be as useful as it is on its own (see
``planning/DISCOVERY_2026-09-09.md`` §9 and §14), so this sweeps one
heuristic's own hyper-parameters with that heuristic as the *only* arm,
and reports each variant paired per seed against that arm's current
default.

Usage::

    uv run python benchmarks/arm_sweep.py ARM OUT.json SEED [SEED ...]

    ARM   one of: cmaes, lbc, jso, lshade, pso
    SEED  base seeds; three is enough to see a large effect, twelve is
          the canonical decision roster

Results are written to ``OUT.json`` after every seed, so interrupting the
run never loses finished work.  Runs are niced by the caller; progress is
measured in evaluations, not wall time.
"""

import sys
import json
import dataclasses
import statistics as st
import time
from collections import defaultdict
from panobbgo.harness_ioh import make_ioh_strategies, make_standard_battery, run_ioh_harness
from panobbgo.heuristics import CMAES, JSO, LSHADE, NLSHADE_LBC, PSO
from panobbgo.strategies import StrategyRoundRobin

BASE = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"][0]
AUTO = {"NP_init": "auto"}

ARMS = {
    "cmaes": (
        CMAES,
        {},
        {
            "sigma0_0.2": {"sigma0": 0.2},
            "sigma0_0.4": {"sigma0": 0.4},
            "sigma0_0.5": {"sigma0": 0.5},
            "bipop": {"restart_mode": "bipop"},
            "ipop_3": {"ipop_factor": 3.0},
            "ipop_1.5": {"ipop_factor": 1.5},
        },
    ),
    "lbc": (
        NLSHADE_LBC,
        {**AUTO, "k_rank": 3.0},
        {
            "k_rank_1": {"k_rank": 1.0},
            "k_rank_6": {"k_rank": 6.0},
            "H_10": {"H": 10},
            "H_20": {"H": 20},
            "arch_1": {"archive_factor": 1.0},
            "arch_3": {"archive_factor": 3.0},
            "np_fixed30": {"NP_init": 30},
        },
    ),
    "jso": (
        JSO,
        dict(AUTO),
        {
            "H_10": {"H": 10},
            "H_20": {"H": 20},
            "arch_1": {"archive_factor": 1.0},
            "arch_3": {"archive_factor": 3.0},
            "np_fixed30": {"NP_init": 30},
            "np_min_8": {"NP_min": 8},
        },
    ),
    "lshade": (
        LSHADE,
        dict(AUTO),
        {
            "H_10": {"H": 10},
            "H_20": {"H": 20},
            "arch_1": {"archive_factor": 1.0},
            "arch_3": {"archive_factor": 3.0},
            "f_sched_jso": {"F_schedule": "jso"},
            "np_fixed30": {"NP_init": 30},
        },
    ),
    "pso": (
        PSO,
        {},
        {
            "NP_40": {"NP": 40},
            "NP_10": {"NP": 10},
            "w_decay": {"w": 0.9, "w_end": 0.4},
            "lbest": {"topology": "lbest"},
            "c_2.0": {"c1": 2.0, "c2": 2.0},
            "vmax_0.2": {"v_max_frac": 0.2},
        },
    ),
}

arm, out = sys.argv[1], sys.argv[2]
seeds = [int(x) for x in sys.argv[3:]]
cls, base_kw, variants = ARMS[arm]


def solo(name, kw):
    return dataclasses.replace(BASE, name=name, strategy_class=StrategyRoundRobin, heuristics=[(cls, kw)], analyzers=[])


specs = [solo("default", dict(base_kw))] + [solo(n, {**base_kw, **kw}) for n, kw in variants.items()]
rows, t0 = [], time.perf_counter()
for seed in seeds:
    r = run_ioh_harness(specs, make_standard_battery(), base_seed=seed, progress=False, sync_eval=True)
    rows += [
        {"seed": seed, "s": x.strategy_name, "dim": x.dim, "inst": x.instance, "aocc": x.aocc, "err": x.error}
        for x in r.runs
    ]
    json.dump(rows, open(out, "w"))
    print(f"seed {seed} done ({time.perf_counter() - t0:.0f}s)", flush=True)

tot, by, errs = defaultdict(list), defaultdict(dict), defaultdict(int)
for r in rows:
    tot[r["s"]].append(r["aocc"])
    by[(r["seed"], r["dim"], r["inst"])][r["s"]] = r["aocc"]
    if r["err"]:
        errs[r["s"]] += 1
n = len(seeds)
tc = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(n, 2.5)
print(f"\n=== {arm} ===  ({n} seeds)")
print(f"{'variant':16s} {'mean':>7s}   delta vs default")
for s in sorted(tot, key=lambda k: -st.mean(tot[k])):
    line = f"{s:16s} {st.mean(tot[s]):7.4f}"
    if s == "default":
        print(line + "   (reference)" + (f"  errors={errs[s]}" if errs[s] else ""))
        continue
    ps = defaultdict(list)
    for k, v in by.items():
        if s in v and "default" in v:
            ps[k[0]].append(v[s] - v["default"])
    ds = [st.mean(x) for x in ps.values()]
    if len(ds) < 2:
        continue
    m, sd = st.mean(ds), st.stdev(ds)
    h = tc * sd / len(ds) ** 0.5
    flag = "  <-- CI excludes 0" if (m - h > 0 or m + h < 0) else ""
    print(
        line
        + f"   {m:+.4f} [{m - h:+.4f},{m + h:+.4f}] {sum(d > 0 for d in ds)}/{len(ds)}{flag}"
        + (f"  errors={errs[s]}" if errs[s] else "")
    )
