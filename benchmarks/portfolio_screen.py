"""Step 2 of the block-bandit experiment plan: does a *portfolio* beat one arm?

``planning/DESIGN_block_bandit_2026-09-10.md`` §6 asks for a three-seed
screen before any twelve-seed roster is spent: run the tuned Phase A arms
under :class:`~panobbgo.strategies.StrategyBlockBandit` and compare them,
paired per (seed, dim, instance) cell, against the two references that a
portfolio has to beat — CMA-ES alone (the current competition candidate,
"the bar") and L-SHADE alone (the best single arm in the provisional
oracle).

The screen separates the two things a bandit portfolio does:

* ``Blocks_uniform_2`` blocks the budget but *learns nothing* (round-robin
  over blocks).  Its delta to the references is the price of blocking —
  the transient every switch costs, and the budget CMA-ES no longer has
  for its covariance adaptation.
* ``Blocks_ducb_2`` adds the discounted-UCB rule on the same arms.  Its
  delta to ``Blocks_uniform_2`` is what the AOCC-area reward *learns*.
  If that is zero, the reward carries no signal and the policy is noise.
* ``Rewarding_ema_2`` is the interleaving control: the same two arms under
  the existing probability-matching strategy, which hands every ready arm
  points on every pass instead of giving one arm a contiguous block.
* The ``*_warm`` specs turn the shared :class:`~panobbgo.analyzers.Archive`
  on and let L-SHADE re-seed from it whenever the scheduler hands it a
  block back (``warm_start_on_resume=True``).  Their delta to the cold
  specs is the direct test of the thesis a portfolio stands on: arms are
  only worth their switching transients if they *share* the evaluations
  they paid for.  CMA-ES has no warm-start hook yet, so the sharing is
  one-directional — CMA-ES simply resumes its own paused state.

Every spec shares ``seed_name="screen"``, so all of them run the identical
RNG stream on each (dim, instance, rep) cell and a delta carries only the
strategy's own effect, not the run-to-run variance (see
:attr:`panobbgo.benchmark.StrategySpec.seed_name`).

Usage::

    uv run python benchmarks/portfolio_screen.py OUT.json SEED [SEED ...] \
        [dims=2,5] [bm=500] [specs=name,name]

    OUT.json  rows file, rewritten after every seed
    SEED      base seeds; three screens, twelve decides
    dims      battery dimensions (default 2,5 — the standard battery)
    bm        budget multiplier; the budget per run is ``bm * dim``
    specs     subset of ``SPECS`` to run (default: all of them)

Re-analysis of a finished run is free::

    uv run python benchmarks/portfolio_screen.py from=OUT.json [specs=a,b]

The screening gates of §6 are printed at the end with PASS/FAIL.  Read
them against the measured null floor: on three seeds a CMA-ES-containing
spec moves by up to ±0.05 for no reason at all, so only deltas larger
than that are evidence.
"""

import dataclasses
import json
import statistics as st
import sys
import time
from collections import defaultdict

from panobbgo.analyzers import Archive
from panobbgo.harness_ioh import make_ioh_strategies, make_standard_battery, run_ioh_harness
from panobbgo.heuristics import CMAES, JSO, LSHADE, NLSHADE_LBC, PSO
from panobbgo.strategies import StrategyBlockBandit, StrategyRewarding, StrategyRoundRobin

BASE = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"][0]

#: The tuned Phase A arms.  One dict, so "which arm with which knobs" is a
#: single obvious edit and every spec below quotes the same settings.
ARM = {
    "cmaes": (CMAES, {}),
    "lshade": (LSHADE, {"NP_init": 10}),
    "jso": (JSO, {"NP_init": 15}),
    "lbc": (NLSHADE_LBC, {"NP_init": 15, "k_rank": 3.0}),
    # PSO won zero cells in the provisional oracle, so it only appears in
    # the deliberately over-armed five-arm spec.
    "pso": (PSO, {"NP": 6}),
}


def arms(*keys):
    return [ARM[k] for k in keys]


def warm_lshade(mode):
    """The L-SHADE arm, re-seeding from the shared ``Archive`` on re-acquisition."""
    cls, kw = ARM["lshade"]
    return (cls, {**kw, "warm_start": mode})


#: Analyzer list of every warm spec.  ``Splitter`` is **not** listed: it is
#: one of the four analyzers ``StrategyBase.initialize`` always installs
#: (``core.py:1327``), so ``archive_leaf`` finds it without help.  ``Archive``
#: is the opt-in one, and it must be present or ``archive_seed`` silently
#: falls back to the Splitter root.
ARCHIVE = [(Archive, {})]

#: name -> (strategy_class, heuristics, strategy kwargs, analyzers).  Strategy
#: kwargs travel through ``config_overrides``: ``create_strategy`` passes them
#: to the constructor for a ``StrategyBase`` subclass (benchmark.py:204).
SPECS = {
    # -- references ------------------------------------------------------
    "CMAES_alone": (StrategyRoundRobin, arms("cmaes"), {}, []),
    "LSHADE_alone": (StrategyRoundRobin, arms("lshade"), {}, []),
    # -- blocking, with and without learning ------------------------------
    "Blocks_uniform_2": (StrategyBlockBandit, arms("cmaes", "lshade"), {"policy": "uniform"}, []),
    "Blocks_ducb_2": (StrategyBlockBandit, arms("cmaes", "lshade"), {"policy": "ducb"}, []),
    "Blocks_ducb_4": (StrategyBlockBandit, arms("cmaes", "lshade", "jso", "lbc"), {"policy": "ducb"}, []),
    "Blocks_ducb_5": (StrategyBlockBandit, arms("cmaes", "lshade", "jso", "lbc", "pso"), {"policy": "ducb"}, []),
    # -- the interleaving control -----------------------------------------
    "Rewarding_ema_2": (StrategyRewarding, arms("cmaes", "lshade"), {}, []),
    # -- the same portfolios, but the arms now SHARE their evaluations ----
    #
    # The thesis a portfolio stands or falls on: an arm that resumes from
    # the best points the *other* arm paid for does not re-buy them.  Only
    # L-SHADE warm-starts here — CMA-ES has no ``warm_start`` hook yet, so
    # it simply resumes its own paused state (its covariance, step size and
    # mean survive the pause untouched).  The asymmetry is the honest
    # measurement of what exists today, not a handicap: the point is whether
    # sharing in *one* direction already moves the number.
    "Blocks_uniform_2_warm": (
        StrategyBlockBandit,
        [ARM["cmaes"], warm_lshade("archive")],
        {"policy": "uniform", "warm_start_on_resume": True},
        ARCHIVE,
    ),
    "Blocks_ducb_2_warm": (
        StrategyBlockBandit,
        [ARM["cmaes"], warm_lshade("archive")],
        {"policy": "ducb", "warm_start_on_resume": True},
        ARCHIVE,
    ),
    # ``archive_leaf`` takes the best point of each of the k best Splitter
    # leaves: k *different basins* rather than k neighbours of one incumbent.
    "Blocks_ducb_2_warm_leaf": (
        StrategyBlockBandit,
        [ARM["cmaes"], warm_lshade("archive_leaf")],
        {"policy": "ducb", "warm_start_on_resume": True},
        ARCHIVE,
    ),
    "Blocks_ducb_2_warm_div": (
        StrategyBlockBandit,
        [ARM["cmaes"], warm_lshade("archive_diverse")],
        {"policy": "ducb", "warm_start_on_resume": True},
        ARCHIVE,
    ),
    # ``warm_start_only_if_foreign=False``: re-seed on *every* re-acquisition,
    # even when the top-k are all the arm's own points.  The default skips
    # that case because re-seeding an arm from itself is a no-op that still
    # throws away its live generation; these two specs measure whether the
    # skip is worth its condition or is just suppressing warm starts.
    "Blocks_uniform_2_warm_any": (
        StrategyBlockBandit,
        [ARM["cmaes"], warm_lshade("archive")],
        {"policy": "uniform", "warm_start_on_resume": True, "warm_start_only_if_foreign": False},
        ARCHIVE,
    ),
    "Blocks_ducb_2_warm_any": (
        StrategyBlockBandit,
        [ARM["cmaes"], warm_lshade("archive")],
        {"policy": "ducb", "warm_start_on_resume": True, "warm_start_only_if_foreign": False},
        ARCHIVE,
    ),
    # No ``Phased_cma60_lshade_warm``: ``StrategyPhased`` never calls
    # ``warm_start_now`` at a phase boundary (the §12 defect), and the arm's
    # own ``on_start`` warm path runs at t = 0 against an empty archive.  The
    # spec would therefore be a *cold* hand-off wearing a warm label, which
    # is worse than not measuring it.  Adding the boundary call means editing
    # ``phased.py``, which this screen does not own.
}

REFS = ("CMAES_alone", "LSHADE_alone")
#: Specs whose arms share evaluations, in the order the gates prefer them.
WARM = [n for n in SPECS if "_warm" in n]

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

battery = make_standard_battery()
if "dims" in opts or "bm" in opts:
    battery = dataclasses.replace(
        battery,
        dims=tuple(int(d) for d in opts.get("dims", ",".join(str(d) for d in battery.dims)).split(",")),
        budget_multiplier=int(opts.get("bm", battery.budget_multiplier)),
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
    # A run that stops short of its budget did not spend what it was given
    # -- a stall, an exhausted arm, or the strategy returning no points.
    if r.get("budget") and r.get("evals", 0) < 0.98 * r["budget"]:
        short[r["s"]].append((r["dim"], r["inst"], r["evals"], r["budget"]))
cells = {k: {s: st.mean(v) for s, v in d.items()} for k, d in raw.items()}
dims = sorted({d for _, d, _ in cells})
n = len(seeds)
tc = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}.get(n, 2.26 if n > 8 else 2.5)

if not cells:
    sys.exit("no rows for the selected specs — nothing to compare")


def mean_of(name, dim=None):
    vals = [v[name] for (_, d, _), v in cells.items() if name in v and (dim is None or d == dim)]
    return st.mean(vals) if vals else float("nan")


def paired(a, b, dim=None):
    """Per-seed mean of ``a - b`` over the cells where both have a result."""
    ps = defaultdict(list)
    for (seed, d, _), v in cells.items():
        if a in v and b in v and (dim is None or d == dim):
            ps[seed].append(v[a] - v[b])
    return [st.mean(x) for x in ps.values()]


def ci(ds):
    """``(mean, halfwidth)`` of a 95% t-CI over the per-seed deltas."""
    if len(ds) < 2:
        return (st.mean(ds) if ds else float("nan")), float("nan")
    m = st.mean(ds)
    return m, tc * st.stdev(ds) / len(ds) ** 0.5


def delta(a, b):
    """Mean paired delta of ``a`` over ``b``, or ``nan`` if uncomparable."""
    ds = paired(a, b)
    return st.mean(ds) if ds else float("nan")


setup = f"from {src}" if src else f"budget {battery.budget_multiplier}*d"
print(f"\n=== portfolio screen ===  ({n} seeds, dims {dims}, {setup})")
print(f"specs: {', '.join(names)}   cells: {len(cells)}")

# (a) means, overall and per dimension.
print(f"\n{'spec':24s} {'mean':>7s} " + "".join(f"  {'d=' + str(d):>8s}" for d in dims))
order = sorted(names, key=lambda s: -mean_of(s))
for s in order:
    per = "".join(f"  {mean_of(s, d):8.4f}" for d in dims)
    tail = f"  errors={len(errs[s])}" if errs[s] else ""
    tail += f"  short={len(short[s])}" if short[s] else ""
    print(f"{s:24s} {mean_of(s):7.4f} " + per + tail)

# (b) paired deltas against each reference, overall CI + per-dimension means.
for ref in REFS:
    if ref not in names:
        continue
    print(f"\ndelta vs {ref} (paired per cell, t-CI over per-seed means)")
    print(
        f"{'spec':24s} {'delta':>8s} {'95% CI':>21s} {'seeds':>7s} " + "".join(f"  {'d=' + str(d):>8s}" for d in dims)
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
        print(f"{s:24s} {m:+8.4f} {band:>21s} {sum(d > 0 for d in ds):3d}/{len(ds):<3d} " + per + flag)

# (c) the §6 screening gates.
best_ref = max((r for r in REFS if r in names), key=mean_of, default=None)
print("\n--- screening gates (design §6) ---")
gates = []
if "Blocks_uniform_2" in names and best_ref:
    d1 = delta("Blocks_uniform_2", best_ref)
    gates.append(("G1", f"Blocks_uniform_2 - max({', '.join(REFS)}) = {best_ref}", d1, -0.02, ">="))
if "Blocks_ducb_2" in names and "Blocks_uniform_2" in names:
    gates.append(("G2", "Blocks_ducb_2 - Blocks_uniform_2", delta("Blocks_ducb_2", "Blocks_uniform_2"), 0.005, ">="))
if "Blocks_ducb_2" in names and best_ref:
    gates.append(("G3", f"Blocks_ducb_2 - best single ({best_ref})", delta("Blocks_ducb_2", best_ref), -0.01, ">="))
# G4/G5 test the sharing thesis: a portfolio is only worth its transients if
# the arms hand each other the evaluations they already paid for.  G4 asks
# whether sharing moves the number *at all* beyond the +-0.05 null floor;
# G5 asks the only question that decides the phase — does it rescue the
# portfolio past the bar.
if "Blocks_ducb_2_warm" in names and "Blocks_ducb_2" in names:
    gates.append(("G4", "Blocks_ducb_2_warm - Blocks_ducb_2", delta("Blocks_ducb_2_warm", "Blocks_ducb_2"), 0.03, ">="))
warm_here = [s for s in WARM if s in names]
if warm_here and "CMAES_alone" in names:
    best_warm = max(warm_here, key=mean_of)
    gates.append(("G5", f"best warm spec ({best_warm}) - CMAES_alone", delta(best_warm, "CMAES_alone"), 0.0, ">="))
if not gates:
    print("  (no gate is computable from the selected specs)")
for tag, what, val, thr, _ in gates:
    ok = val == val and val >= thr
    print(f"  {tag} {'PASS' if ok else 'FAIL'}  {what:52s} {val:+.4f}  (need >= {thr:+.3f})")

# (d) what the harness itself reported about the runs.
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
    "\nNull floor: on 3 seeds a CMA-ES-containing spec drifts by up to +-0.05 for no\n"
    "reason at all, so a gate verdict inside that band is a direction, not evidence.\n"
    "Only deltas larger than +-0.05 (or a CI that excludes zero, marked `<--`) count;\n"
    "the 12-seed roster of §6 step 3 is what decides."
)
