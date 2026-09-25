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
        [dims=5] [bm=500] [insts=0,1,2] [specs=name,name] [jobs=N]
    uv run python benchmarks/meta_screen.py from=OUT.json [specs=a,b]
"""

import dataclasses
import sys

from panobbgo.analyzers import Archive
from panobbgo.harness_ioh import make_standard_battery, run_ioh_harness
from panobbgo.local_run import screen_jobs
from panobbgo.heuristics import CMAES, JSO, MetaAnalyst, RegionUCB
from panobbgo.heuristics.meta import budget_fraction, stagnation
from panobbgo.strategies import StrategyBlockBandit, StrategyRoundRobin

if __package__:  # ``python -m benchmarks.<screen>``: make ``_screen`` importable by name
    import sys

    from . import _screen as _screen_module

    sys.modules.setdefault("_screen", _screen_module)
from _screen import IOH_FIELDS_BUDGET, base_spec, csv_of, delta, fold, int_tuple, load_rows, match, mean_of
from _screen import mean_or_nan, paired, parse_argv, print_delta_table, print_means, print_run_health, run_seeds
from _screen import runs_to_rows, select_names, table_spec


def main():
    BASE = base_spec()

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
        module RNG streams are keyed by name (``spawn_rng``), and the name sorts
        after ``CMAES``/``JSO``, so the event-bus registration order of the arms
        is preserved too.  Move it anywhere but last and every number below
        becomes an ordering effect rather than a mechanism (design §5, last row).
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
    opts, _, pos = parse_argv(sys.argv[1:])

    src = opts.get("from")
    JOBS = 1 if src else screen_jobs(opts)
    names = select_names(opts, SPECS)

    #: d=5 only, per design §4 — d=2 cannot resolve a kd-tree coverage statistic.
    battery = make_standard_battery()
    battery = dataclasses.replace(
        battery,
        dims=int_tuple(opts.get("dims", "5")),
        budget_multiplier=int(opts.get("bm", 500)),
        instances=int_tuple(opts.get("insts", csv_of(battery.instances))),
    )

    if src:
        rows, seeds, names = load_rows(src, opts, names, SPECS)
    else:
        if len(pos) < 2:
            sys.exit(__doc__)
        out = pos[0]
        seeds = [int(x) for x in pos[1:]]
        # One stream for the whole screen: every spec sees the identical
        # (dim, inst, rep) seeds, so the comparison is paired on the stream.
        specs = [table_spec(BASE, n, SPECS[n]) for n in names]

        def batches(seed):
            r = run_ioh_harness(specs, battery, base_seed=seed, progress=False, sync_eval=True, jobs=JOBS)
            yield runs_to_rows(seed, r.runs, IOH_FIELDS_BUDGET)

        rows = run_seeds(seeds, out, batches)

    # --- fold rows into cells: (seed, fid, dim, inst) -> {spec: mean AOCC over reps}
    # ``fid`` is None without a function axis (and in files written before it).
    cells, errs, short = fold(rows, names, short_label=lambda r: f"d{r['dim']}i{r['inst']}")
    dims = sorted({d for _, _, d, _ in cells})
    insts = sorted({i for _, _, _, i in cells})
    n = len(seeds)

    if not cells:
        sys.exit("no rows for the selected specs — nothing to compare")

    def identical(a, b):
        """Are the two specs equal in *every* cell?  The null's own test."""
        both = [(v[a], v[b]) for v in cells.values() if a in v and b in v]
        return bool(both) and all(x == y for x, y in both), len(both)

    setup = f"from {src}" if src else f"budget {battery.budget_multiplier}*d"
    print(f"\n=== meta-level screen ===  ({n} seeds, dims {dims}, {setup})")
    print(f"specs: {', '.join(names)}   cells: {len(cells)}")

    # (a) means, overall and per dimension.
    order = sorted(names, key=lambda s: -mean_of(cells, s))
    print_means(cells, order, dims, errs, short, width=28)

    # (b) paired deltas against the reference portfolio and the single arm.
    for ref in (REF, SINGLE):
        if ref in names:
            print_delta_table(cells, order, ref, dims, width=28)

    # (c) the per-instance breakdown §4 asks for: the cells where the
    # portfolio's CMA-ES arm is badly wrong are a *minority of instances*
    # (§28), so a battery mean can hide the whole effect.
    if REF in names and len(insts) > 1:
        print(f"\nper-instance delta vs {REF} (d={dims[-1]})")
        print(f"{'spec':28s} " + "".join(f"  {'i' + str(i):>8s}" for i in insts))
        for s in order:
            if s == REF:
                continue
            row = "".join(f"  {mean_or_nan(paired(cells, s, REF, match(dim=dims[-1], inst=i))):+8.4f}" for i in insts)
            print(f"{s:28s} " + row)

    # (d) the pre-registered verdicts.
    print("\n--- pre-registered rules (design §4) ---")
    if "Meta_never" in names and REF in names:
        same, k = identical("Meta_never", REF)
        verdict = "PASS" if same else "FAIL"
        print(f"  NULL {verdict}  Meta_never == {REF} in all {k} shared cells")
        if not same:
            print("       the module is in the wrong place in the heuristics list;")
            print("       every number above is an ordering effect, not a mechanism.")
    if "Meta_random_b25" in names and "Meta_b25" in names:
        d = delta(cells, "Meta_b25", "Meta_random_b25")
        print(f"  R1   Meta_b25 - Meta_random_b25 = {d:+.4f}   (|d| <= 0.01 ⇒ drop the scan as a point emitter)")
    if "RegionUCB_arm" in names and "Meta_b25" in names:
        d = delta(cells, "Meta_b25", "RegionUCB_arm")
        print(f"  R2   Meta_b25 - RegionUCB_arm   = {d:+.4f}   (<= 0 ⇒ drop the concentration claim)")
    if "Meta_region_b25" in names and REF in names:
        d = delta(cells, "Meta_region_b25", REF)
        print(f"  R3   Meta_region_b25 - {REF[:12]} = {d:+.4f}   (<= 0 on the roster ⇒ drop the design)")

    # (e) what the harness itself reported about the runs.
    print_run_health(names, errs, short)

    print(
        "\nSize estimate, stated before the run (design §4): a meta level that repairs\n"
        "only the cells where the portfolio's CMA-ES arm is badly wrong is chasing\n"
        "<= +0.017 -- below the +-0.05 three-seed null floor.  So a positive delta here\n"
        "is a candidate for the 12-seed roster and nothing more; a *negative* one, or a\n"
        "falsifier that ties, is the only evidence three seeds can actually deliver."
    )


if __name__ == "__main__":
    main()
