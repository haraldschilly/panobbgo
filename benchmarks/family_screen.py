"""Does the sharing portfolio hold up *outside* the battery of record?

``planning/GOAL.md`` §2 measures one regime — noiseless, unconstrained
MA-BBOB at ``d <= 5`` — and in that regime a two-arm sharing portfolio is
level with the best single arm (``DISCOVERY_2026-09-09.md`` §27/§31).
"Level on one problem class" is the weakest possible verdict on a
portfolio, because robustness *across* classes is the thing a portfolio
is for.  This screen runs the same four specs over
:mod:`panobbgo.harness_families`:

* the **free** preset — five parametrised families (ill-conditioned
  ellipsoid, Rosenbrock valley, Rastrigin, Ackley, sharp ridge), shifted,
  rotated, at ``d`` = 2, 5 **and 10**;
* the **constrained** preset — four families with 1 to 3 constraints that
  are *active at the optimum*, at ``d`` = 2 and 5.  Panobbgo's constraint
  handling (``lib/constraints.py``) has never been measured on AOCC at
  all; this is its first number.

The specs are copied verbatim from ``benchmarks/portfolio_screen.py`` so
the two screens are the same comparison on different problems:
``CMAES_alone``, ``JSO_alone``, ``LSHADE_alone`` — the three single arms
that sit within 0.014 of each other on MA-BBOB (§22) — and
``Blocks_uniform_cj_warm2``, the best portfolio found so far (CMA-ES +
jSO, blocked, no learning rule, both arms re-seeded from the shared
``Archive``).

Every spec shares ``seed_name="screen"``, so all four run the identical
RNG stream on each (family, dim, instance) cell and a delta carries only
the strategy's own effect (``StrategySpec.seed_name``, §18).

Usage::

    uv run python benchmarks/family_screen.py OUT.json SEED [SEED ...] \
        [preset=free|constrained] [dims=2,5,10] [bm=500] [ninst=3] \
        [specs=name,name]

    OUT.json  rows file, rewritten after every seed
    SEED      base seeds; three screens, twelve decides
    preset    which battery (default: free)
    dims      override the preset's dimensions
    bm        budget multiplier; the budget per run is ``bm * dim``
    ninst     instances per (family, dim)

Re-analysis of a finished run is free::

    uv run python benchmarks/family_screen.py from=OUT.json [specs=a,b]

Read every number against the measured null floor: on three seeds a
CMA-ES-containing spec drifts by up to +-0.05 for no reason at all
(§18a), so only a larger delta, or a CI that excludes zero, is evidence —
and even then the 12-seed roster is what decides.
"""

import dataclasses
import json
import statistics as st
import sys
import time
from collections import defaultdict

from panobbgo.analyzers import Archive
from panobbgo.harness_families import make_constrained_battery, make_families_battery, run_family_harness
from panobbgo.harness_ioh import make_ioh_strategies
from panobbgo.heuristics import CMAES, JSO, LSHADE
from panobbgo.strategies import StrategyBlockBandit, StrategyRoundRobin

BASE = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"][0]

#: The tuned Phase A arms, verbatim from ``portfolio_screen.py``.
#: ``NP_init="auto"`` is the accepted default (12/12), so the DE arms run
#: on the shipped rule rather than a hand-picked constant.
ARM = {
    "cmaes": (CMAES, {}),
    "lshade": (LSHADE, {"NP_init": "auto"}),
    "jso": (JSO, {"NP_init": "auto"}),
}


def arms(*keys):
    return [ARM[k] for k in keys]


def warm(key, mode):
    """The tuned arm ``key``, re-seeding from the shared ``Archive`` on re-acquisition."""
    cls, kw = ARM[key]
    return (cls, {**kw, "warm_start": mode})


#: ``warm_start_only_if_foreign=False``: re-seed on *every* re-acquisition
#: (§21).  ``warm_start_only_if_better`` is left at its default, which is
#: ``False`` since §31 measured the guard at -0.073.
WARM_ON = {"warm_start_on_resume": True, "warm_start_only_if_foreign": False}

#: ``Archive`` is the opt-in analyzer; without it ``archive_seed``
#: silently falls back to the Splitter root.  ``Splitter`` is not listed —
#: ``StrategyBase.initialize`` always installs it.
ARCHIVE = [(Archive, {})]

#: name -> (strategy_class, heuristics, strategy kwargs, analyzers).
SPECS = {
    "CMAES_alone": (StrategyRoundRobin, arms("cmaes"), {}, []),
    "JSO_alone": (StrategyRoundRobin, arms("jso"), {}, []),
    "LSHADE_alone": (StrategyRoundRobin, arms("lshade"), {}, []),
    # The best portfolio found so far (§27/§30): CMA-ES + jSO, blocked,
    # *no* learning rule, both arms warm.
    "Blocks_uniform_cj_warm2": (
        StrategyBlockBandit,
        [warm("cmaes", "archive"), warm("jso", "archive")],
        {"policy": "uniform", **WARM_ON},
        ARCHIVE,
    ),
}

#: Every single-arm reference.  "The bar" is the best of them *in this
#: run*, computed rather than named — §22 killed the idea of one champion.
REFS = tuple(n for n in SPECS if n.endswith("_alone"))

PRESETS = {"free": make_families_battery, "constrained": make_constrained_battery}

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

preset = opts.get("preset", "free")
if preset not in PRESETS:
    sys.exit(f"unknown preset {preset!r} (known: {', '.join(PRESETS)})")
bm = int(opts.get("bm", 500))
kwargs = {}
if "dims" in opts:
    kwargs["dims"] = tuple(int(d) for d in opts["dims"].split(","))
if "ninst" in opts:
    kwargs["n_instances"] = int(opts["ninst"])


def spec(name):
    cls, heuristics, kw, analyzers = SPECS[name]
    return dataclasses.replace(
        BASE,
        name=name,
        # One stream for the whole screen: every spec sees the identical
        # (family, dim, instance) seeds, so the comparison is paired.
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
    preset = rows[0].get("preset", preset) if rows else preset
    bm = rows[0].get("bm", bm) if rows else bm
    print(f"read {len(rows)} rows from {src}")
else:
    if len(pos) < 2:
        sys.exit(__doc__)
    out = pos[0]
    seeds = [int(x) for x in pos[1:]]
    instances = PRESETS[preset](**kwargs)
    specs = [spec(n) for n in names]
    print(
        f"battery {preset}: {len(instances)} instances "
        f"({len({p.family for _n, p in instances})} families, dims "
        f"{sorted({p.dim for _n, p in instances})}), budget {bm}*d, "
        f"{len(specs)} specs, {len(seeds)} seeds",
        flush=True,
    )
    rows, t0 = [], time.perf_counter()
    for seed in seeds:
        r = run_family_harness(specs, instances, budget_multiplier=bm, base_seed=seed, progress=False, sync_eval=True)
        rows += [
            {
                "seed": seed,
                "preset": preset,
                "bm": bm,
                "s": x.strategy_name,
                "fam": x.problem_kind,
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

# --- fold rows into cells: (seed, family, dim, inst) -> {spec: mean AOCC} ---
raw = defaultdict(lambda: defaultdict(list))
errs, short = defaultdict(list), defaultdict(list)
for r in rows:
    if r["s"] not in names:
        continue
    raw[(r["seed"], r["fam"], r["dim"], r["inst"])][r["s"]].append(r["aocc"])
    if r["err"]:
        errs[r["s"]].append(r["err"])
    # A run that stops short of its budget did not spend what it was given.
    if r.get("budget") and r.get("evals", 0) < 0.98 * r["budget"]:
        short[r["s"]].append((r["fam"], r["dim"], r["evals"], r["budget"]))
cells = {k: {s: st.mean(v) for s, v in d.items()} for k, d in raw.items()}
if not cells:
    sys.exit("no rows for the selected specs — nothing to compare")

dims = sorted({d for _, _, d, _ in cells})
fams = sorted({f for _, f, _, _ in cells})
n = len(seeds)
tc = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}.get(n, 2.26 if n > 8 else 2.5)


def mean_of(name, dim=None, fam=None):
    vals = [
        v[name]
        for (_, f, d, _), v in cells.items()
        if name in v and (dim is None or d == dim) and (fam is None or f == fam)
    ]
    return st.mean(vals) if vals else float("nan")


def paired(a, b, dim=None, fam=None):
    """Per-seed mean of ``a - b`` over the cells where both have a result."""
    ps = defaultdict(list)
    for (seed, f, d, _), v in cells.items():
        if a in v and b in v and (dim is None or d == dim) and (fam is None or f == fam):
            ps[seed].append(v[a] - v[b])
    return [st.mean(x) for x in ps.values()]


def ci(ds):
    """``(mean, halfwidth)`` of a 95% t-CI over the per-seed deltas."""
    if len(ds) < 2:
        return (st.mean(ds) if ds else float("nan")), float("nan")
    m = st.mean(ds)
    return m, tc * st.stdev(ds) / len(ds) ** 0.5


print(f"\n=== family screen ===  ({n} seeds, preset {preset}, budget {bm}*d, dims {dims})")
print(f"specs: {', '.join(names)}   cells: {len(cells)}   families: {', '.join(fams)}")

# (a) means, overall and per dimension.
print(f"\n{'spec':28s} {'mean':>7s} " + "".join(f"  {'d=' + str(d):>8s}" for d in dims))
order = sorted(names, key=lambda s: -mean_of(s))
for s in order:
    per = "".join(f"  {mean_of(s, d):8.4f}" for d in dims)
    tail = f"  errors={len(errs[s])}" if errs[s] else ""
    tail += f"  short={len(short[s])}" if short[s] else ""
    print(f"{s:28s} {mean_of(s):7.4f} " + per + tail)

# (b) means per family — the whole point of a multi-class battery.
print(f"\n{'spec':28s} " + "".join(f"  {f[:14]:>15s}" for f in fams))
for s in order:
    print(f"{s:28s} " + "".join(f"  {mean_of(s, fam=f):15.4f}" for f in fams))

# (c) paired deltas against each reference, overall CI + per-dimension means.
for ref in REFS:
    if ref not in names:
        continue
    print(f"\ndelta vs {ref} (paired per cell, t-CI over per-seed means)")
    print(
        f"{'spec':28s} {'delta':>8s} {'95% CI':>21s} {'seeds':>7s} " + "".join(f"  {'d=' + str(d):>8s}" for d in dims)
    )
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

# (d) the portfolio against the best single arm *in this run*, per family.
best_ref = max((r for r in REFS if r in names), key=mean_of, default=None)
port = [s for s in names if s.startswith("Blocks")]
if best_ref and port:
    print(f"\nper-family delta of the portfolio vs the best single arm ({best_ref})")
    print(f"{'family':18s} " + "".join(f"  {p[:20]:>21s}" for p in port))
    for f in fams:
        print(f"{f:18s} " + "".join(f"  {st.mean(paired(p, best_ref, fam=f) or [float('nan')]):+21.4f}" for p in port))

# (e) what the harness itself reported about the runs.
print("\n--- run health ---")
if not any(errs.values()) and not any(short.values()):
    print("  no errored runs, every run spent its full budget")
for s in names:
    if errs[s]:
        seen = sorted(set(errs[s]))[:3]
        print(f"  {s}: {len(errs[s])} errored run(s); first messages: {seen}")
    if short[s]:
        ex = ", ".join(f"{f} d{d} {e}/{b}" for f, d, e, b in short[s][:4])
        print(f"  {s}: {len(short[s])} run(s) below budget: {ex}")

print(
    "\nNull floor: on 3 seeds a CMA-ES-containing spec drifts by up to +-0.05 for no\n"
    "reason at all, so a delta inside that band is a direction, not evidence.  Only a\n"
    "CI that excludes zero (marked `<--`) counts, and the 12-seed roster decides."
)
