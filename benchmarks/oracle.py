"""How much is there to gain from *choosing* an arm at all?

Run every arm solo on the same battery, then ask, per (seed, dim,
instance) cell, what a clairvoyant selector would have scored: the
maximum AOCC over the arms on that very cell.  The mean of those maxima
is the **oracle**.  Its distance to the best *single* arm is the
**headroom** — everything a portfolio, a bandit, or any selection policy
could ever win here.  If the headroom is small, no amount of bandit
cleverness pays for itself and the effort belongs in the arms instead
(see ``planning/DISCOVERY_2026-09-09.md`` §9/§14 and the Phase A sweeps).

The oracle is an **upper** bound, and a loose one.  It picks the winner
per cell *for free and after the fact*: no evaluations are spent finding
out which arm is best, and it may switch arms between two instances of
the same function, which a real policy learning online cannot do.  A
real policy pays for discovery out of the same budget, so it lands below
the oracle — often far below.  The realistic target sits between the
best single arm and the oracle; the **top-2 oracle** (best of just two
arms) is the more honest goal, since a two-arm portfolio is what a
simple selector can actually carry.

Usage::

    uv run python benchmarks/oracle.py OUT.json SEED [SEED ...] \
        [dims=2,5] [bm=500] [arms=cmaes,lbc,jso,lshade,pso] [--defaults] [jobs=N]

    OUT.json  rows file, rewritten after every seed
    SEED      base seeds; three shows the shape, twelve decides
    dims      battery dimensions (default 2,5 — the standard battery)
    bm        budget multiplier; the budget per run is ``bm * dim``
    arms      subset of ``ARMS`` to compare (default: all of them)
    jobs      run the (seed, cell) runs in N worker processes; results do not depend on N
    --defaults  run every arm with an EMPTY kwargs dict instead of the
                tuned settings in ``ARMS``, to see whether tuning the
                arms moved the oracle or only moved the arms

Re-analysis of a finished run is free::

    uv run python benchmarks/oracle.py --from OUT.json [arms=cmaes,pso]
    uv run python benchmarks/oracle.py from=OUT.json

Only cells where *every* selected arm has a result enter the oracle,
win, regret and pair statistics — a maximum over a different arm set per
cell would not be comparable.  Incomplete cells are reported and skipped.
"""

import sys
import json
import dataclasses
import statistics as st
from collections import defaultdict
from itertools import combinations
from panobbgo.harness_ioh import make_standard_battery, run_ioh_harness, t_ci
from panobbgo.local_run import screen_jobs
from panobbgo.heuristics import CMAES, JSO, LSHADE, NLSHADE_LBC, PSO

from _screen import IOH_FIELDS, base_spec, csv_of, fold, int_tuple, parse_argv, run_seeds, runs_to_rows, solo_spec


def main():
    BASE = base_spec()

    # The currently tuned per-arm settings (Phase A sweeps).  One dict, so
    # that "which arms, with which knobs" is a single obvious edit.
    ARMS = {
        "cmaes": (CMAES, {}),  # sigma-divergence restart is the default now
        "lbc": (NLSHADE_LBC, {"NP_init": "auto"}),  # auto = 4*dim*(budget/500dim)^0.25 (LBC's AUTO_DIM_COEF)
        "jso": (JSO, {"NP_init": "auto"}),
        "lshade": (LSHADE, {"NP_init": "auto"}),
        "pso": (PSO, {"NP": 6}),
    }

    # --- argv: `key=value` options, `--flag`s, `--from PATH`, then positionals ---
    opts, flags, pos = parse_argv(sys.argv[1:], value_flags=("--from", "--arms", "--dims", "--bm"))

    JOBS = screen_jobs(opts) if not opts.get("from") else 1
    defaults = "defaults" in flags or opts.get("defaults", "0") not in ("0", "", "no", "false")
    src = opts.get("from")
    arms = [a for a in opts["arms"].split(",") if a] if opts.get("arms") else list(ARMS)

    battery = make_standard_battery()
    if "dims" in opts or "bm" in opts:
        battery = dataclasses.replace(
            battery,
            dims=int_tuple(opts.get("dims", csv_of(battery.dims))),
            budget_multiplier=int(opts.get("bm", battery.budget_multiplier)),
        )

    #: One RNG stream for EVERY arm.  The oracle compares arms per cell, so the
    #: arms must be paired on the same stream — pinning the stream per arm (the
    #: arm_sweep convention, right for variants of one arm) makes the per-cell
    #: max a max over unpaired draws (REGIME_TABLE_2026-09-11.md §2: one cell
    #: differed by 0.52 between the per-arm and the shared stream).
    ORACLE_SEED_NAME = "oracle"

    def solo(name, cls, kw, seed_name=None):
        # ``seed_name`` pins the RNG stream to the *arm key*, not the display name,
        # so a tuned run and its ``--defaults`` counterpart (and any future run that
        # relabels an arm) share the stream per (dim, inst, rep) cell and their
        # difference is the settings, not the run-to-run variance.
        return solo_spec(BASE, name, seed_name or ORACLE_SEED_NAME, (cls, kw))

    if src:
        rows = json.load(open(src))
        seeds = sorted({r["seed"] for r in rows})
        have = {r["arm"] for r in rows}
        arms = [a for a in arms if a in have] if opts.get("arms") else sorted(have)
        print(f"read {len(rows)} rows from {src}")
    else:
        out = pos[0]
        seeds = [int(x) for x in pos[1:]]
        unknown = [a for a in arms if a not in ARMS]
        if unknown:
            sys.exit(f"unknown arm(s): {','.join(unknown)}  (known: {','.join(ARMS)})")
        specs = [solo(a, ARMS[a][0], {} if defaults else dict(ARMS[a][1])) for a in arms]
        #: The display name is the arm key.
        fields = (("arm", "strategy_name"),) + IOH_FIELDS[1:]

        def batches(seed):
            r = run_ioh_harness(specs, battery, base_seed=seed, progress=False, sync_eval=True, jobs=JOBS)
            yield runs_to_rows(seed, r.runs, fields)

        rows = run_seeds(seeds, out, batches)

    # --- fold rows into cells: (seed, fid, dim, inst) -> {arm: mean AOCC over reps} ---
    # ``fid`` is None without a function axis (and in files written before it).
    cells, errs, _ = fold(rows, arms, key="arm")
    full = {k: v for k, v in cells.items() if len(v) == len(arms)}
    dims = sorted({d for _, _, d, _ in full})
    n = len(seeds)

    if not full:
        sys.exit("no cell has a result for every selected arm — nothing to compare")

    def mean_of(f, dim=None):
        """Mean of ``f(cell)`` over the complete cells, optionally one dimension."""
        return st.mean([f(v) for (_, _, d, _), v in full.items() if dim is None or d == dim])

    # In ``--from`` mode the battery is whatever produced the file, not the one
    # built above, so do not claim a budget or a kwargs flavour we cannot know.
    setup = (
        f"from {src}" if src else f"budget {battery.budget_multiplier}*d, {'defaults' if defaults else 'tuned'} kwargs"
    )
    print(f"\n=== oracle ===  ({n} seeds, dims {dims}, {setup})")
    gone = len(cells) - len(full)
    print(f"arms: {', '.join(arms)}   cells: {len(full)} complete" + (f", {gone} INCOMPLETE (skipped)" if gone else ""))

    # (a) per-arm means, overall and per dimension, plus (b) the oracle.
    means = {a: mean_of(lambda v, a=a: v[a]) for a in arms}
    oracle = mean_of(lambda v: max(v.values()))
    best = max(arms, key=lambda a: means[a])
    print(
        f"\n{'arm':10s} {'mean':>7s} "
        + "".join(f"  {'d=' + str(d):>8s}" for d in dims)
        + f"  {'regret':>7s}  {'wins':>5s}"
    )
    wins = defaultdict(int)
    for v in full.values():
        wins[max(arms, key=lambda a: v[a])] += 1
    for a in sorted(arms, key=lambda a: -means[a]):
        per = "".join(f"  {mean_of(lambda v, a=a: v[a], d):8.4f}" for d in dims)
        err = f"  errors={len(errs[a])}" if errs[a] else ""
        mark = " <-- best single" if a == best else ""
        print(f"{a:10s} {means[a]:7.4f} " + per + f"  {oracle - means[a]:7.4f}  {wins[a]:5d}" + mark + err)
    per = "".join(f"  {mean_of(lambda v: max(v.values()), d):8.4f}" for d in dims)
    print(f"{'ORACLE':10s} {oracle:7.4f} " + per + f"  {0.0:7.4f}  {len(full):5d}")

    # (c) headroom with a per-seed paired t-CI: per-seed oracle minus per-seed best arm.
    ds = []
    for seed in seeds:
        cs = [v for (s, _, _, _), v in full.items() if s == seed]
        if cs:
            ds.append(st.mean([max(v.values()) for v in cs]) - st.mean([v[best] for v in cs]))
    line = f"\nheadroom (oracle - {best}): {oracle - means[best]:+.4f}"
    if len(ds) >= 2:
        m, h = t_ci(ds)
        sig = " <--" if m - h > 0 else ""
        line += f"   paired per seed {m:+.4f} [{m - h:+.4f},{m + h:+.4f}] {sum(d > 1e-12 for d in ds)}/{len(ds)}{sig}"
    line += f"   = {100 * (oracle - means[best]) / max(means[best], 1e-12):.1f}% of {best}"
    print(line)
    for d in dims:
        o, b = mean_of(lambda v: max(v.values()), d), mean_of(lambda v: v[best], d)
        bd = max(arms, key=lambda a: mean_of(lambda v, a=a: v[a], d))
        print(
            f"  d={d}: oracle {o:.4f}  best {bd} {mean_of(lambda v: v[bd], d):.4f}  headroom {o - b:+.4f} (vs {best})"
        )

    # (d) win counts per (fid, dim, inst): majority over seeds, and summed over seeds.
    seedwins = defaultdict(lambda: defaultdict(int))
    for (_, f, d, i), v in full.items():
        seedwins[(f, d, i)][max(arms, key=lambda a: v[a])] += 1
    # Majority owner of a (fid, dim, inst) cell: most seed wins, ties to the better arm overall.
    owner = {k: max(arms, key=lambda a: (w[a], means[a])) for k, w in seedwins.items()}
    cellwins = {a: sum(o == a for o in owner.values()) for a in arms}
    allwins = {a: sum(w[a] for w in seedwins.values()) for a in arms}

    def cell_label(f, d, i):
        return (f"f{f}" if f is not None else "") + f"d{d}i{i}"

    print(f"\nwin counts over {len(seedwins)} (fid, dim, inst) cells")
    print(f"{'arm':10s} {'majority':>9s} {'seed-wins':>10s}   per-cell majorities")
    for a in sorted(arms, key=lambda a: (-cellwins[a], -allwins[a])):
        ordered = sorted(owner.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1], kv[0][2]))
        owned = " ".join(cell_label(*k) for k, o in ordered if o == a)
        print(f"{a:10s} {cellwins[a]:9d} {allwins[a]:10d}   {owned}")

    # (f) top-2 oracle: how much of the headroom a two-arm portfolio could capture.
    if len(arms) >= 2:
        gap = oracle - means[best]
        print("\ntop-2 oracle (best of two arms only)")
        pairs = sorted(
            ((mean_of(lambda v, p=p: max(v[p[0]], v[p[1]])), p) for p in combinations(arms, 2)), key=lambda t: -t[0]
        )
        for val, (a, b) in pairs:
            frac = f"{100 * (val - means[best]) / gap:5.1f}%" if gap > 1e-12 else "    -"
            print(f"  {a + '+' + b:22s} {val:7.4f}   {val - means[best]:+.4f} over {best}   {frac} of the headroom")

    print(
        "\nThe oracle picks the per-cell winner for free and after the fact, so it is an\n"
        "UPPER bound: a real policy spends part of the same budget discovering which arm\n"
        "wins and cannot switch per instance, so it lands below — the top-2 row is the\n"
        "more realistic target.  `regret` is what you lose by always picking that one arm."
    )


if __name__ == "__main__":
    main()
