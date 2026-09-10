"""Acceptance run for the new ``NP_init="auto"`` sizing rule.

``planning/DISCOVERY_2026-09-09.md`` §17 located the population law at
``NP ≈ 3·dim`` at the reference budget of ``500·dim`` evaluations, against
the ``18·dim`` the library shipped.  This script decides whether the new
rule may become the default: it runs, per arm, three solo-arm variants
paired on the same RNG stream and reports the paired per-seed deltas.

    auto_old   the *shipped* rule, ``clip(round(min(18·dim, budget/12)), 6, 400)``,
               passed as an explicit int (that is what ``"auto"`` used to give)
    auto_new   ``NP_init="auto"`` — i.e. whatever the rule in
               :mod:`panobbgo.heuristics.lshade` currently resolves to
    fixed_4d   ``NP_init = 3·dim``, the plain dimensional rule with no budget
               term — at ``bm=500`` this is ``auto_new`` by construction, so a
               non-zero delta here would mean the harness is not pairing

Because ``NP_init`` must be a concrete ``int`` per spec and the two explicit
variants depend on the dimension, each dimension of the battery is run as its
own battery (``dataclasses.replace(battery, dims=(d,))``) with per-dimension
specs.  Everything else follows ``benchmarks/arm_sweep.py``: solo arms via
``StrategyRoundRobin`` with no analyzers, a constant ``seed_name`` so all
variants share the RNG stream, ``sync_eval=True``, rows flushed after every
seed, and paired-per-seed t-CIs reported overall and per dimension.

Usage::

    uv run python benchmarks/np_accept.py ARM OUT.json SEED [SEED ...] \
        [dims=2,5] [bm=500]

    ARM   one of: lshade, jso, lbc
    SEED  base seeds; the canonical decision roster is the 12 seeds in
          ``panobbgo.harness_ioh.DEFAULT_DECISION_SEEDS``
"""

import sys
import json
import dataclasses
import statistics as st
import time
from collections import defaultdict
from panobbgo.harness_ioh import make_ioh_strategies, make_standard_battery, run_ioh_harness
from panobbgo.heuristics import JSO, LSHADE, NLSHADE_LBC
from panobbgo.strategies import StrategyRoundRobin

BASE = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"][0]

# The rule as it shipped before ``planning/DISCOVERY_2026-09-09.md`` §17:
# ``min(18·dim, budget/12)`` clipped into ``[6, 400]``.  Reproduced here (not
# imported) so the reference stays the *old* behaviour even after the module
# constants change.
_OLD_DIM_COEF, _OLD_GEN_TARGET, _OLD_MIN_NP, _OLD_MAX_NP = 18, 12.0, 6, 400


def old_auto(dim: int, budget: float) -> int:
    """What ``NP_init="auto"`` resolved to before the new rule."""
    raw = min(float(_OLD_DIM_COEF * dim), budget / _OLD_GEN_TARGET)
    return int(min(max(round(raw), _OLD_MIN_NP), _OLD_MAX_NP))


ARMS = {
    "lshade": (LSHADE, {}),
    "jso": (JSO, {}),
    "lbc": (NLSHADE_LBC, {"k_rank": 3.0}),
}

arm, out = sys.argv[1], sys.argv[2]
_opts = {k: v for k, _, v in (a.partition("=") for a in sys.argv[3:] if "=" in a)}
seeds = [int(x) for x in sys.argv[3:] if "=" not in x]
cls, base_kw = ARMS[arm]

battery = make_standard_battery()
if "dims" in _opts or "bm" in _opts:
    battery = dataclasses.replace(
        battery,
        dims=tuple(int(d) for d in _opts.get("dims", "2,5").split(",")),
        budget_multiplier=int(_opts.get("bm", battery.budget_multiplier)),
    )


def solo(name, kw):
    # ``seed_name`` is the *arm*, constant across all three variants, so each
    # of them runs the identical RNG stream on every (dim, inst, rep) cell and
    # the paired delta carries the parameter's effect, not run-to-run variance.
    return dataclasses.replace(
        BASE,
        name=name,
        seed_name=arm,
        strategy_class=StrategyRoundRobin,
        heuristics=[(cls, {**base_kw, **kw})],
        analyzers=[],
    )


def specs_for(dim: int):
    budget = battery.budget_for(dim)
    return [
        solo("auto_old", {"NP_init": old_auto(dim, budget)}),
        solo("auto_new", {"NP_init": "auto"}),
        solo("fixed_4d", {"NP_init": max(6, round(4 * dim))}),
    ]


rows, t0 = [], time.perf_counter()
for seed in seeds:
    for d in battery.dims:
        one_dim = dataclasses.replace(battery, dims=(d,))
        r = run_ioh_harness(specs_for(d), one_dim, base_seed=seed, progress=False, sync_eval=True)
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
tc = {
    2: 12.71,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
    11: 2.228,
    12: 2.201,
}.get(n, 2.0)


def paired_by_seed(name, ref, dim=None):
    """Per-seed mean deltas of ``name`` against ``ref``, optionally one dimension."""
    ps = defaultdict(list)
    for (seed, d, _), v in by.items():
        if name in v and ref in v and (dim is None or d == dim):
            ps[seed].append(v[name] - v[ref])
    return {s: st.mean(x) for s, x in ps.items()}


def ci(ds):
    m, sd = st.mean(ds), st.stdev(ds)
    h = tc * sd / len(ds) ** 0.5
    return m, m - h, m + h


def report(name, ref):
    ds = list(paired_by_seed(name, ref).values())
    m, lo, hi = ci(ds)
    pos = sum(d > 0 for d in ds)
    flag = " <--" if (lo > 0 or hi < 0) else "   "
    print(f"\n{name} vs {ref}:  {m:+.4f} [{lo:+.4f},{hi:+.4f}]  {pos}/{len(ds)} seeds positive{flag}")
    for dim in dims:
        dd = list(paired_by_seed(name, ref, dim).values())
        dm, dlo, dhi = ci(dd)
        dflag = " <--" if (dlo > 0 or dhi < 0) else ""
        print(f"    d={dim:<3d} {dm:+.4f} [{dlo:+.4f},{dhi:+.4f}]  {sum(x > 0 for x in dd)}/{len(dd)}{dflag}")
    per_seed = paired_by_seed(name, ref)
    print("    per seed: " + "  ".join(f"{s}:{v:+.4f}" for s, v in sorted(per_seed.items())))


print(f"\n=== {arm} ===  ({n} seeds, dims {dims}, budget {battery.budget_multiplier}*d)")
old_np = {d: old_auto(d, battery.budget_for(d)) for d in dims}
new_np = {d: max(6, 4 * d) for d in dims}
print(f"NP_init: auto_old={old_np}  fixed_4d={new_np}")
head = f"{'variant':12s} {'mean':>7s}"
print(head + "".join(f"   {'d=' + str(d):>8s}" for d in dims))
for s in sorted(tot, key=lambda k: -st.mean(tot[k])):
    per = "".join(f"   {st.mean([v[s] for (_, d, _), v in by.items() if s in v and d == dim]):8.4f}" for dim in dims)
    err = f"  errors={errs[s]}" if errs[s] else ""
    print(f"{s:12s} {st.mean(tot[s]):7.4f}" + per + err)

report("auto_new", "auto_old")
report("fixed_4d", "auto_new")
print("\n`<--` marks a 95% t-CI on the paired delta that excludes zero.")
