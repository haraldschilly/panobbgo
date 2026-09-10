"""Does a *meta level* — one analysis at a chosen moment — beat the streams?

``planning/DESIGN_meta_level_2026-09-10.md`` §4.  Same conventions as
``benchmarks/portfolio_screen.py``: every spec on one ``seed_name="screen"``
stream so a paired delta carries only the strategy's own effect,
``sync_eval=True``, rows JSON rewritten after every seed, paired per
``(seed, dim, inst)`` cell, three seeds to screen and the 12-seed roster to
decide (§30: the maximum of a screen is a *candidate*, its screen CI carries
no weight).

The reference is ``Blocks_uniform_cj_warm2`` — CMA-ES + jSO, both
``warm_start="archive"``, rotation, both warm-start guards off, the shared
``Archive`` analyzer — 0.6854 on the roster, +0.019 over CMA-ES alone (§27).

| spec | what it isolates |
|---|---|
| ``CMAES_alone`` | the single-arm bar |
| ``Blocks_uniform_cj_warm2`` | the reference portfolio |
| ``Meta_never`` | reference + ``MetaAnalyst(trigger="never")`` — **the null**, which must be *bit-identical* to the reference |
| ``Meta_b25`` | the leaf scan, ``budget_fraction(0.25)``, ``k = 0.02*max_eval`` points |
| ``Meta_stag`` | the same scan on a ``stagnation(0.10)`` trigger |
| ``Meta_random_b25`` | ``k`` **uniform random** points at the identical firing times — the falsifier: if it ties ``Meta_b25``, the analysis is decoration and only the jolt of exploration mattered |
| ``Meta_region_b25`` | the region hand-off, **no** points emitted: the box is handed to an arm and applied at its next block open |
| ``RegionUCB_arm`` | reference + ``RegionUCB`` as a third arm — a *stream* at the same total cost, the attribution control for the concentration claim (and a measurement of the third-arm tax, §26.2: 0.685 → 0.648) |

**Battery: ``dims=5 bm=500`` only.**  Not the standard *d* ∈ {2,5}: at
*d* = 2 the ``Splitter``'s ``limit`` is 250, so the root cannot split before
evaluation 250 and a 25 %-of-budget trigger fires into a tree with one leaf
(design §0.2); §27/§30 also record that *d* = 2 has nothing to gain.

**Stated before running** (design §4): a meta level that repairs only the
cells where the portfolio's CMA-ES arm is badly wrong is chasing
**≤ +0.017**, which is *below* the ±0.05 three-seed null floor.  So this
screen can realistically only (a) confirm the null is exact, (b) rule a
mechanism *out*, or (c) produce a candidate for the roster.  It cannot
produce evidence of a gain on its own.

Pre-registered drop rules:

* ``Meta_random_b25`` within 0.01 of ``Meta_b25`` ⇒ drop the scan as a point
  emitter;
* ``Meta_b25`` ≤ ``RegionUCB_arm`` ⇒ drop the concentration claim;
* ``Meta_region_b25`` ≤ the reference at *d* = 5 on the roster ⇒ drop the
  design;
* any spec below 98 % of budget ⇒ investigate the ``Splitter`` live-lock
  before reading any AOCC number.

Usage::

    uv run python benchmarks/meta_screen.py OUT.json SEED [SEED ...] \
        [dims=5] [bm=500] [insts=0,1,2] [specs=name,name]
    uv run python benchmarks/meta_screen.py from=OUT.json [specs=a,b]
"""

import dataclasses
import json
import statistics as st
import sys
import time
from collections import defaultdict

from panobbgo.analyzers import Archive
from panobbgo.harness_ioh import make_ioh_strategies, make_standard_battery, run_ioh_harness
from panobbgo.heuristics import CMAES, JSO, MetaAnalyst, RegionUCB
from panobbgo.heuristics.meta import budget_fraction, stagnation
from panobbgo.strategies import StrategyBlockBandit, StrategyRoundRobin

BASE = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"][0]

#: The two tuned arms of the reference portfolio, both sharing evaluations.
CJ = [(CMAES, {"warm_start": "archive"}), (JSO, {"NP_init": "auto", "warm_start": "archive"})]

#: The reference portfolio's strategy kwargs.  Both warm-start guards are
#: pinned explicitly: ``only_if_foreign`` measured better off (§21) and
#: ``only_if_better`` measured −0.007 on the roster (§30).
UNIFORM_WARM = {
    "policy": "uniform",
    "warm_start_on_resume": True,
    "warm_start_only_if_foreign": False,
    "warm_start_only_if_better": False,
}

#: Analyzer list of every portfolio spec — ``Archive`` is the opt-in one and
#: must be present or ``archive_seed`` silently falls back to the Splitter.
ARCHIVE = [(Archive, {})]


def meta(**kw):
    """The meta arm, **last** in a heuristics list.

    Order matters and is the whole reason ``Meta_never`` can be a null:
    modules draw their RNG stream from ``spawn_rng`` in construction order
    (``benchmark.py`` adds heuristics in list order), and the name sorts
    after ``CMAES``/``JSO``, so the event-bus registration order of the arms
    is preserved too.  Move it anywhere but last and every number below
    becomes a stream shift rather than a mechanism (design §5, last row).
    """
    return (MetaAnalyst, kw)


#: ``k = 0.02 * max_eval`` (design §4) — 50 points at d=5, bm=500 — under the
#: hard ``meta_frac = 0.05`` cap, i.e. one firing may spend at most 2 % of the
#: budget and the module at most 5 % over the whole run.
K = {"k_frac": 0.02, "meta_frac": 0.05}

SPECS = {
    # -- references -------------------------------------------------------
    "CMAES_alone": (StrategyRoundRobin, [(CMAES, {})], {}, []),
    "Blocks_uniform_cj_warm2": (StrategyBlockBandit, CJ, dict(UNIFORM_WARM), ARCHIVE),
    # -- step 0: the null --------------------------------------------------
    "Meta_never": (
        StrategyBlockBandit,
        CJ + [meta(trigger="never")],
        dict(UNIFORM_WARM),
        ARCHIVE,
    ),
    # -- steps 1+2: the leaf scan as a point emitter ------------------------
    "Meta_b25": (
        StrategyBlockBandit,
        CJ + [meta(trigger=budget_fraction(0.25), mode="scan", **K)],
        dict(UNIFORM_WARM),
        ARCHIVE,
    ),
    "Meta_stag": (
        StrategyBlockBandit,
        CJ + [meta(trigger=stagnation(0.10), mode="scan", **K)],
        dict(UNIFORM_WARM),
        ARCHIVE,
    ),
    # The falsifier: identical trigger, identical k, no analysis at all.
    "Meta_random_b25": (
        StrategyBlockBandit,
        CJ + [meta(trigger=budget_fraction(0.25), mode="random", **K)],
        dict(UNIFORM_WARM),
        ARCHIVE,
    ),
    # -- step 3: the region hand-off (zero evaluations of its own) ---------
    "Meta_region_b25": (
        StrategyBlockBandit,
        CJ + [meta(trigger=budget_fraction(0.25), mode="none", region=True)],
        dict(UNIFORM_WARM),
        ARCHIVE,
    ),
    # -- the stream-at-the-same-cost control -------------------------------
    "RegionUCB_arm": (
        StrategyBlockBandit,
        CJ + [(RegionUCB, {})],
        dict(UNIFORM_WARM),
        ARCHIVE,
    ),
}

#: The bar every spec is measured against.
REF = "Blocks_uniform_cj_warm2"
#: The single-arm reference.
SINGLE = "CMAES_alone"

# --- argv: `key=value` options, then positionals ---------------------------
opts, pos = {}, []
for a in sys.argv[1:]:
    if "=" in a:
        k, _, v = a.partition("=")
        opts[k] = v
    else:
        pos.append(a)

src = opts.get("from")
names = [n for n in opts["specs"].split(",") if n] if opts.get("specs") else list(SPECS)
unknown = [n for n in names if n not in SPECS]
if unknown:
    sys.exit(f"unknown spec(s): {','.join(unknown)}  (known: {','.join(SPECS)})")

#: d=5 only, per design §4 — d=2 cannot resolve a kd-tree coverage statistic.
battery = make_standard_battery()
battery = dataclasses.replace(
    battery,
    dims=tuple(int(d) for d in opts.get("dims", "5").split(",")),
    budget_multiplier=int(opts.get("bm", 500)),
    instances=tuple(int(i) for i in opts.get("insts", ",".join(str(i) for i in battery.instances)).split(",")),
)


def spec(name):
    cls, heuristics, kw, analyzers = SPECS[name]
    return dataclasses.replace(
        BASE,
        name=name,
        # One stream for the whole screen: every spec sees the identical
        # (dim, inst, rep) seeds, so the comparison is paired on the stream.
        seed_name="screen",
        strategy_class=cls,
        heuristics=[(c, dict(k)) for c, k in heuristics],
        analyzers=[(c, dict(k)) for c, k in analyzers],
        config_overrides=dict(kw),
    )


if src:
    rows = json.load(open(src))
    seeds = sorted({r["seed"] for r in rows})
    have = {r["s"] for r in rows}
    names = [n for n in names if n in have] if opts.get("specs") else [n for n in SPECS if n in have]
    print(f"read {len(rows)} rows from {src}")
else:
    if len(pos) < 2:
        sys.exit(__doc__)
    out = pos[0]
    seeds = [int(x) for x in pos[1:]]
    specs = [spec(n) for n in names]
    rows, t0 = [], time.perf_counter()
    for seed in seeds:
        r = run_ioh_harness(specs, battery, base_seed=seed, progress=False, sync_eval=True)
        rows += [
            {
                "seed": seed,
                "s": x.strategy_name,
                "dim": x.dim,
                "inst": x.instance,
                "aocc": x.aocc,
                "evals": x.n_evals,
                "budget": x.budget,
                "err": x.error,
            }
            for x in r.runs
        ]
        json.dump(rows, open(out, "w"))
        print(f"seed {seed} done ({time.perf_counter() - t0:.0f}s)", flush=True)

# --- fold rows into cells: (seed, dim, inst) -> {spec: mean AOCC over reps} --
raw = defaultdict(lambda: defaultdict(list))
errs, short = defaultdict(list), defaultdict(list)
for r in rows:
    if r["s"] not in names:
        continue
    raw[(r["seed"], r["dim"], r["inst"])][r["s"]].append(r["aocc"])
    if r["err"]:
        errs[r["s"]].append(r["err"])
    if r.get("budget") and r.get("evals", 0) < 0.98 * r["budget"]:
        short[r["s"]].append((r["dim"], r["inst"], r["evals"], r["budget"]))
cells = {k: {s: st.mean(v) for s, v in d.items()} for k, d in raw.items()}
dims = sorted({d for _, d, _ in cells})
insts = sorted({i for _, _, i in cells})
n = len(seeds)
tc = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}.get(n, 2.26 if n > 8 else 2.5)

if not cells:
    sys.exit("no rows for the selected specs — nothing to compare")


def mean_of(name, dim=None, inst=None):
    vals = [
        v[name]
        for (_, d, i), v in cells.items()
        if name in v and (dim is None or d == dim) and (inst is None or i == inst)
    ]
    return st.mean(vals) if vals else float("nan")


def paired(a, b, dim=None, inst=None):
    """Per-seed mean of ``a - b`` over the cells where both have a result."""
    ps = defaultdict(list)
    for (seed, d, i), v in cells.items():
        if a in v and b in v and (dim is None or d == dim) and (inst is None or i == inst):
            ps[seed].append(v[a] - v[b])
    return [st.mean(x) for x in ps.values()]


def ci(ds):
    """``(mean, halfwidth)`` of a 95% t-CI over the per-seed deltas."""
    if len(ds) < 2:
        return (st.mean(ds) if ds else float("nan")), float("nan")
    m = st.mean(ds)
    return m, tc * st.stdev(ds) / len(ds) ** 0.5


def delta(a, b):
    ds = paired(a, b)
    return st.mean(ds) if ds else float("nan")


def identical(a, b):
    """Are the two specs equal in *every* cell?  The null's own test."""
    both = [(v[a], v[b]) for v in cells.values() if a in v and b in v]
    return bool(both) and all(x == y for x, y in both), len(both)


setup = f"from {src}" if src else f"budget {battery.budget_multiplier}*d"
print(f"\n=== meta-level screen ===  ({n} seeds, dims {dims}, {setup})")
print(f"specs: {', '.join(names)}   cells: {len(cells)}")

# (a) means, overall and per dimension.
print(f"\n{'spec':28s} {'mean':>7s} " + "".join(f"  {'d=' + str(d):>8s}" for d in dims))
order = sorted(names, key=lambda s: -mean_of(s))
for s in order:
    per = "".join(f"  {mean_of(s, d):8.4f}" for d in dims)
    tail = f"  errors={len(errs[s])}" if errs[s] else ""
    tail += f"  short={len(short[s])}" if short[s] else ""
    print(f"{s:28s} {mean_of(s):7.4f} " + per + tail)

# (b) paired deltas against the reference portfolio and the single arm.
for ref in (REF, SINGLE):
    if ref not in names:
        continue
    print(f"\ndelta vs {ref} (paired per cell, t-CI over per-seed means)")
    print(f"{'spec':28s} {'delta':>8s} {'95% CI':>21s} {'seeds':>7s} " + "".join(f"  {'d=' + str(d):>8s}" for d in dims))
    for s in order:
        if s == ref:
            continue
        ds = paired(s, ref)
        if not ds:
            continue
        m, h = ci(ds)
        band = f"[{m - h:+.4f},{m + h:+.4f}]" if h == h else "        (n<2)"
        flag = " <--" if h == h and (m - h > 0 or m + h < 0) else ""
        per = "".join(f"  {st.mean(paired(s, ref, d) or [float('nan')]):+8.4f}" for d in dims)
        print(f"{s:28s} {m:+8.4f} {band:>21s} {sum(d > 0 for d in ds):3d}/{len(ds):<3d} " + per + flag)

# (c) the per-instance breakdown §4 asks for: the cells where the
# portfolio's CMA-ES arm is badly wrong are a *minority of instances*
# (§28), so a battery mean can hide the whole effect.
if REF in names and len(insts) > 1:
    print(f"\nper-instance delta vs {REF} (d={dims[-1]})")
    print(f"{'spec':28s} " + "".join(f"  {'i' + str(i):>8s}" for i in insts))
    for s in order:
        if s == REF:
            continue
        row = "".join(f"  {st.mean(paired(s, REF, dims[-1], i) or [float('nan')]):+8.4f}" for i in insts)
        print(f"{s:28s} " + row)

# (d) the pre-registered verdicts.
print("\n--- pre-registered rules (design §4) ---")
if "Meta_never" in names and REF in names:
    same, k = identical("Meta_never", REF)
    verdict = "PASS" if same else "FAIL"
    print(f"  NULL {verdict}  Meta_never == {REF} in all {k} shared cells")
    if not same:
        print("       the module is in the wrong place in the heuristics list;")
        print("       every number above is a stream shift, not a mechanism.")
if "Meta_random_b25" in names and "Meta_b25" in names:
    d = delta("Meta_b25", "Meta_random_b25")
    print(f"  R1   Meta_b25 - Meta_random_b25 = {d:+.4f}   (|d| <= 0.01 ⇒ drop the scan as a point emitter)")
if "RegionUCB_arm" in names and "Meta_b25" in names:
    d = delta("Meta_b25", "RegionUCB_arm")
    print(f"  R2   Meta_b25 - RegionUCB_arm   = {d:+.4f}   (<= 0 ⇒ drop the concentration claim)")
if "Meta_region_b25" in names and REF in names:
    d = delta("Meta_region_b25", REF)
    print(f"  R3   Meta_region_b25 - {REF[:12]} = {d:+.4f}   (<= 0 on the roster ⇒ drop the design)")

# (e) what the harness itself reported about the runs.
print("\n--- run health ---")
if not any(errs.values()) and not any(short.values()):
    print("  no errored runs, every run spent its full budget")
for s in names:
    if errs[s]:
        seen = sorted(set(errs[s]))[:3]
        print(f"  {s}: {len(errs[s])} errored run(s); first messages: {seen}")
    if short[s]:
        ex = ", ".join(f"d{d}i{i} {e}/{b}" for d, i, e, b in short[s][:4])
        print(f"  {s}: {len(short[s])} run(s) below budget: {ex}")

print(
    "\nSize estimate, stated before the run (design §4): a meta level that repairs\n"
    "only the cells where the portfolio's CMA-ES arm is badly wrong is chasing\n"
    "<= +0.017 -- below the +-0.05 three-seed null floor.  So a positive delta here\n"
    "is a candidate for the 12-seed roster and nothing more; a *negative* one, or a\n"
    "falsifier that ties, is the only evidence three seeds can actually deliver."
)
