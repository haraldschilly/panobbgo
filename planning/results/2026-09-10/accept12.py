"""12-seed acceptance for the CMA-ES σ-divergence default (worktree @ 22326eb).

Three solo-CMAES specs sharing one ``seed_name`` (same RNG stream per cell):

    old      CMAES(self_restart=False)      — what shipped before 2026-09-10
    new      CMAES()                        — shipped defaults (σ-divergence on)
    new_best CMAES(restart_from="best")

Each spec is run in its own ``run_ioh_harness`` call — the per-cell seed derives
from ``rng_identity``/dim/instance/rep only, never from the spec list — so each
run can record its own restart count and reasons.

Usage (from the worktree, so the pinned checkout is what gets measured)::

    cd <worktree> && nice -n 15 uv run python <this file> OUT.json
"""

import sys
import json
import dataclasses
import statistics as st
import time
from collections import defaultdict

from panobbgo.harness_ioh import make_ioh_strategies, make_standard_battery, run_ioh_harness
from panobbgo.heuristics import CMAES
from panobbgo.strategies import StrategyRoundRobin

SEEDS = [42, 7, 1234, 2025, 3, 11, 99, 123, 777, 2024, 31337, 555]
VARIANTS = {
    "old": {"self_restart": False},
    "new": {},
    "new_best": {"restart_from": "best"},
}
# two-sided 95% t quantiles by sample size
TC = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 8: 2.365, 10: 2.262, 12: 2.201}

BASE = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"][0]

# --- per-run restart instrumentation -------------------------------------
RUNS: list[list[str]] = []
_orig_start = CMAES.on_start
_orig_self = CMAES._self_restart_now


def _start(self):
    RUNS.append([])
    _orig_start(self)


def _self(self, reason):
    if RUNS:
        RUNS[-1].append(reason)
    _orig_self(self, reason)


CMAES.on_start = _start
CMAES._self_restart_now = _self


def solo(name, kw):
    return dataclasses.replace(
        BASE,
        name=name,
        strategy_class=StrategyRoundRobin,
        heuristics=[(CMAES, kw)],
        analyzers=[],
        seed_name="cmaes_solo",
    )


def measure(out):
    battery = make_standard_battery()
    rows, t0 = [], time.perf_counter()
    for seed in SEEDS:
        for name, kw in VARIANTS.items():
            RUNS.clear()
            r = run_ioh_harness([solo(name, kw)], battery, base_seed=seed, progress=False, sync_eval=True)
            reasons = RUNS if len(RUNS) == len(r.runs) else [[] for _ in r.runs]
            rows += [
                {
                    "seed": seed,
                    "s": x.strategy_name,
                    "dim": x.dim,
                    "inst": x.instance,
                    "aocc": x.aocc,
                    "err": x.error,
                    "best_fx": x.best_fx,
                    "restarts": len(rs),
                    "reasons": rs,
                }
                for x, rs in zip(r.runs, reasons)
            ]
            json.dump(rows, open(out, "w"))
        print(f"seed {seed} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    return rows


def ci(deltas):
    """Mean and 95% t half-width of a list of per-seed paired deltas."""
    m = st.mean(deltas)
    h = TC.get(len(deltas), 2.2) * st.stdev(deltas) / len(deltas) ** 0.5
    return m, m - h, m + h


def report(rows):
    by, nrest, why, errs = defaultdict(dict), defaultdict(list), defaultdict(lambda: defaultdict(int)), defaultdict(int)
    for r in rows:
        by[(r["seed"], r["dim"], r["inst"])][r["s"]] = r
        nrest[r["s"]].append(r["restarts"])
        for reason in r["reasons"]:
            why[r["s"]][reason] += 1
        if r["err"]:
            errs[r["s"]] += 1
    dims = sorted({k[1] for k in by})

    def paired(name, dim=None):
        ps = defaultdict(list)
        for (seed, d, _), v in by.items():
            if name in v and "old" in v and (dim is None or d == dim):
                ps[seed].append(v[name]["aocc"] - v["old"]["aocc"])
        return [st.mean(x) for x in ps.values()]

    print(f"\n=== 12-seed acceptance: CMA-ES σ-divergence default ===  ({len(SEEDS)} seeds, dims {dims})")
    print(f"{'variant':10s} {'mean AOCC':>9s}   {'paired delta vs old [95% t-CI]':>34s}  {'+seeds':>7s}  {'restarts/run':>12s}")
    for name in VARIANTS:
        mean_a = st.mean(v[name]["aocc"] for v in by.values() if name in v)
        rs = f"{st.mean(nrest[name]):.2f} (max {max(nrest[name])})"
        if name == "old":
            print(f"{name:10s} {mean_a:9.4f}   {'(reference)':>34s}  {'':>7s}  {rs:>12s}")
            continue
        ds = paired(name)
        m, lo, hi = ci(ds)
        print(
            f"{name:10s} {mean_a:9.4f}   {m:+.4f} [{lo:+.4f}, {hi:+.4f}]{'':>10s}"
            f"  {sum(d > 0 for d in ds):>4d}/{len(ds)}  {rs:>12s}"
            + (f"  errors={errs[name]}" if errs[name] else "")
        )

    print("\nper dimension (paired delta vs old, 95% t-CI over seeds):")
    for name in VARIANTS:
        if name == "old":
            for d in dims:
                mm = st.mean(v["old"]["aocc"] for k, v in by.items() if k[1] == d)
                print(f"  {'old':10s} d={d}  mean AOCC {mm:.4f}")
            continue
        for d in dims:
            ds = paired(name, d)
            m, lo, hi = ci(ds)
            print(f"  {name:10s} d={d}  {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  {sum(x > 0 for x in ds)}/{len(ds)} seeds positive")

    print("\nrestart reasons (count over all runs):")
    for name in VARIANTS:
        w = why[name]
        print(f"  {name:10s} " + ("  ".join(f"{k}={v}" for k, v in sorted(w.items(), key=lambda kv: -kv[1])) or "(none)"))

    print("\ncells where sigma_divergence fired:")
    for name in ("new", "new_best"):
        fired = [k for k, v in by.items() if name in v and "sigma_divergence" in v[name]["reasons"]]
        if not fired:
            print(f"  {name:10s} (none)")
            continue
        dl = [by[k][name]["aocc"] - by[k]["old"]["aocc"] for k in fired]
        m, lo, hi = st.mean(dl), min(dl), max(dl)
        print(
            f"  {name:10s} {len(fired)}/{len(by)} cells   mean delta {m:+.4f} "
            f"[min {lo:+.4f}, max {hi:+.4f}]   {sum(x > 0 for x in dl)}/{len(dl)} positive   "
            f"mean 'old' AOCC on those cells {st.mean(by[k]['old']['aocc'] for k in fired):.4f} "
            f"(battery {st.mean(v['old']['aocc'] for v in by.values()):.4f})"
        )
        for k in sorted(fired):
            print(
                f"      seed {k[0]:<6d} d={k[1]} inst={k[2]}  "
                f"{by[k]['old']['aocc']:.4f} -> {by[k][name]['aocc']:.4f}  "
                f"{by[k][name]['aocc'] - by[k]['old']['aocc']:+.4f}"
            )

    print("\nacceptance rule: CI excludes zero, >=9/12 seeds positive, no dimension negative-excluding")
    for name in ("new", "new_best"):
        ds = paired(name)
        m, lo, hi = ci(ds)
        pos = sum(d > 0 for d in ds)
        g1, g2 = lo > 0, pos >= 9
        g3 = all(ci(paired(name, d))[2] > 0 for d in dims)
        verdict = "ACCEPT" if (g1 and g2 and g3) else "REJECT"
        print(
            f"  {name:10s} CI>0: {str(g1):5s}   seeds {pos}/12 >= 9: {str(g2):5s}   "
            f"no dim negative-excluding: {str(g3):5s}   ->  {verdict}"
        )


if __name__ == "__main__":
    out = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--report-only":
        report(json.load(open(out)))
    else:
        report(measure(out))
