"""Phase A: tune one optimizer ALONE on the standard IOH battery.

Each arm of a portfolio can only be as useful as it is on its own (see
``planning/DISCOVERY_2026-09-09.md`` §9 and §14), so this sweeps one
heuristic's own hyper-parameters with that heuristic as the *only* arm,
and reports each variant paired per seed against that arm's current
default.

Usage::

    uv run python benchmarks/arm_sweep.py ARM OUT.json SEED [SEED ...] \
        [dims=2,5] [bm=500]

    ARM   one of: cmaes, lbc, jso, lshade, pso
    SEED  base seeds; three is enough to see a large effect, twelve is
          the canonical decision roster
    dims  battery dimensions (default 2,5 — the standard battery)
    grid  NP_init grid for the DE arms (default 4,6,8,10,12,15,20)
    bm    budget multiplier; the budget per run is ``bm * dim``

The report breaks every delta down **per dimension** as well as overall.
That is not cosmetic: the standard battery's budget is ``500 * dim``, so
dimension and budget move together and a rule of the form ``NP = c * dim``
cannot be told apart from ``NP = budget / k`` by the pooled mean alone.
Reading a fixed-value grid separately at each dimension does separate
them — if the optimum moves with ``dim``, the law is dimensional.

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

_opts = {k: v for k, _, v in (a.partition("=") for a in sys.argv[3:] if "=" in a)}
# ``grid=20,30,45`` overrides the NP_init grid, e.g. to probe a higher dimension.
NP_GRID = {f"np_{n}": {"NP_init": int(n)} for n in _opts.get("grid", "4,6,8,10,12,15,20").split(",")}

ARMS = {
    "cmaes": (
        CMAES,
        {},
        {
            # 1.5 is an INTERIOR optimum: 1.2 +0.017, 1.35 +0.018, 1.5 +0.089,
            # 1.75 +0.061, 2.0 (default) 0.  Kept here only to re-confirm it
            # on the 12-seed roster before it becomes the default.
            "ipop_1.5": {"ipop_factor": 1.5},
            "ipop_1.75": {"ipop_factor": 1.75},
        },
    ),
    "lbc": (
        NLSHADE_LBC,
        {**AUTO, "k_rank": 3.0},
        {
            **NP_GRID,
            "np_30_H_20": {"NP_init": 30, "H": 20},
            "np_30_H_20_k_6": {"NP_init": 30, "H": 20, "k_rank": 6.0},
        },
    ),
    "jso": (
        JSO,
        dict(AUTO),
        {**NP_GRID, "np_30_H_20": {"NP_init": 30, "H": 20}},
    ),
    "lshade": (
        LSHADE,
        dict(AUTO),
        {**NP_GRID, "np_30_f_jso": {"NP_init": 30, "F_schedule": "jso"}},
    ),
    "pso": (
        PSO,
        {},
        {
            "NP_3": {"NP": 3},
            "NP_4": {"NP": 4},
            "NP_5": {"NP": 5},
            "NP_6": {"NP": 6},
            "NP_8": {"NP": 8},
            "NP_6_vmax_0.2": {"NP": 6, "v_max_frac": 0.2},
            "NP_6_vmax_0.1_lbest": {"NP": 6, "v_max_frac": 0.1, "topology": "lbest"},
        },
    ),
}

arm, out = sys.argv[1], sys.argv[2]
opts = _opts
seeds = [int(x) for x in sys.argv[3:] if "=" not in x]
cls, base_kw, variants = ARMS[arm]

battery = make_standard_battery()
if "dims" in opts or "bm" in opts:
    battery = dataclasses.replace(
        battery,
        dims=tuple(int(d) for d in opts.get("dims", "2,5").split(",")),
        budget_multiplier=int(opts.get("bm", battery.budget_multiplier)),
    )


def solo(name, kw):
    # ``seed_name`` is the *arm*, constant across "default" and every variant,
    # so all of them run the identical RNG stream on each (dim, inst, rep) cell.
    # Without it the seed is hashed from the display name and a variant that
    # changes nothing still shows a nonzero delta (proved on CMA-ES's
    # ``ipop_factor``): the paired delta would carry the full run-to-run
    # variance rather than the parameter's own effect.
    return dataclasses.replace(
        BASE,
        name=name,
        seed_name=arm,
        strategy_class=StrategyRoundRobin,
        heuristics=[(cls, kw)],
        analyzers=[],
    )


specs = [solo("default", dict(base_kw))] + [solo(n, {**base_kw, **kw}) for n, kw in variants.items()]
rows, t0 = [], time.perf_counter()
for seed in seeds:
    r = run_ioh_harness(specs, battery, base_seed=seed, progress=False, sync_eval=True)
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
dims = sorted({r["dim"] for r in rows})
tc = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(n, 2.5)


def paired(name, dim=None):
    """Per-seed mean deltas of ``name`` against ``default``, optionally one dimension."""
    ps = defaultdict(list)
    for (seed, d, _), v in by.items():
        if name in v and "default" in v and (dim is None or d == dim):
            ps[seed].append(v[name] - v["default"])
    return [st.mean(x) for x in ps.values()]


print(f"\n=== {arm} ===  ({n} seeds, dims {dims}, budget {battery.budget_multiplier}*d)")
head = f"{'variant':22s} {'mean':>7s}   {'delta vs default':>28s}"
print(head + "".join(f"   {'d=' + str(d):>8s}" for d in dims))
for s in sorted(tot, key=lambda k: -st.mean(tot[k])):
    line = f"{s:22s} {st.mean(tot[s]):7.4f}"
    err = f"  errors={errs[s]}" if errs[s] else ""
    if s == "default":
        per = "".join(
            f"   {st.mean([v[s] for (_, d, _), v in by.items() if s in v and d == dim]):8.4f}" for dim in dims
        )
        print(line + f"{'(reference)':>31s}" + per + err)
        continue
    ds = paired(s)
    if len(ds) < 2:
        continue
    m, sd = st.mean(ds), st.stdev(ds)
    h = tc * sd / len(ds) ** 0.5
    flag = " <--" if (m - h > 0 or m + h < 0) else "   "
    per = "".join(f"   {st.mean(paired(s, dim)):+8.4f}" for dim in dims)
    print(line + f"   {m:+.4f} [{m - h:+.4f},{m + h:+.4f}] {sum(d > 0 for d in ds)}/{len(ds)}{flag}" + per + err)
print("\nPer-dimension columns are mean AOCC for `default`, mean delta for the rest.")
print("`<--` marks a 95% t-CI on the pooled delta that excludes zero.")
