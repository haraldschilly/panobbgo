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
  all; this is its first number;
* the **shapes** preset — the five BBOB shapes the free preset lacks
  (Lunacek double funnel, Gallagher peaks, attractive sector, step
  ellipsoid, bent cigar) at ``d`` = 2, 5 and 10;
* the **failure** preset — four families with a region where evaluations
  crash or time out (half-space with the optimum on its boundary, ball,
  random boxes), at ``d`` = 2 and 5.  Failed calls are spent budget.

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
        [preset=free|constrained|shapes|failure] [dims=2,5,10] [bm=500] [ninst=3] \
        [specs=name,name] [timeout=SECONDS] [jobs=N]

``timeout`` is a per-run wall-clock deadline (default: none); a run past it
is stopped, scored so far and counted under ``errors``.

    OUT.json  rows file, rewritten after every seed
    SEED      base seeds; three screens, twelve decides
    preset    which battery: free (default), constrained, shapes, failure
    dims      override the preset's dimensions
    bm        budget multiplier; the budget per run is ``bm * dim``
    ninst     instances per (family, dim)
    jobs      run the (seed, cell) runs in N worker processes; results do not depend on N

Re-analysis of a finished run is free::

    uv run python benchmarks/family_screen.py from=OUT.json [specs=a,b]

Read every number against the measured null floor: on three seeds a
CMA-ES-containing spec drifts by up to +-0.05 for no reason at all
(§18a), so only a larger delta, or a CI that excludes zero, is evidence —
and even then the 12-seed roster is what decides.
"""

import sys

from panobbgo.analyzers import Archive
from panobbgo.harness_families import (
    make_constrained_battery,
    make_failure_battery,
    make_families_battery,
    make_shapes_battery,
    run_family_harness,
)
from panobbgo.local_run import screen_jobs
from panobbgo.heuristics import CMAES, JSO, LSHADE
from panobbgo.strategies import StrategyBlockBandit, StrategyRoundRobin

if __package__:  # ``python -m benchmarks.<screen>``: make ``_screen`` importable by name
    import sys

    from . import _screen as _screen_module

    sys.modules.setdefault("_screen", _screen_module)
from _screen import base_spec, fold, load_rows, match, mean_of, mean_or_nan, paired, parse_argv, print_delta_table
from _screen import print_means, print_run_health, run_seeds, runs_to_rows, select_names, table_spec


def main():
    BASE = base_spec()

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

    PRESETS = {
        "free": make_families_battery,
        "constrained": make_constrained_battery,
        "shapes": make_shapes_battery,
        "failure": make_failure_battery,
    }

    # --- argv: `key=value` options, then positionals ---------------------------
    opts, _, pos = parse_argv(sys.argv[1:])

    src = opts.get("from")
    JOBS = 1 if src else screen_jobs(opts)
    names = select_names(opts, SPECS)

    preset = opts.get("preset", "free")
    if preset not in PRESETS:
        sys.exit(f"unknown preset {preset!r} (known: {', '.join(PRESETS)})")
    bm = int(opts.get("bm", 500))
    timeout_s = float(opts["timeout"]) if opts.get("timeout") else None
    kwargs = {}
    if "dims" in opts:
        kwargs["dims"] = tuple(int(d) for d in opts["dims"].split(","))
    if "ninst" in opts:
        kwargs["n_instances"] = int(opts["ninst"])

    if src:
        rows, seeds, names = load_rows(src, opts, names, SPECS)
        preset = rows[0].get("preset", preset) if rows else preset
        bm = rows[0].get("bm", bm) if rows else bm
    else:
        if len(pos) < 2:
            sys.exit(__doc__)
        out = pos[0]
        seeds = [int(x) for x in pos[1:]]
        instances = PRESETS[preset](**kwargs)
        # One stream for the whole screen: every spec sees the identical
        # (family, dim, instance) seeds, so the comparison is paired.
        specs = [table_spec(BASE, n, SPECS[n]) for n in names]
        print(
            f"battery {preset}: {len(instances)} instances "
            f"({len({p.family for _n, p in instances})} families, dims "
            f"{sorted({p.dim for _n, p in instances})}), budget {bm}*d, "
            f"{len(specs)} specs, {len(seeds)} seeds",
            flush=True,
        )
        fields = (
            ("s", "strategy_name"),
            ("fam", "problem_kind"),
            ("dim", "dim"),
            ("inst", "instance"),
            ("aocc", "aocc"),
            ("evals", "n_evals"),
            ("budget", "budget"),
            ("err", "error"),
        )

        def batches(seed):
            r = run_family_harness(
                specs,
                instances,
                budget_multiplier=bm,
                base_seed=seed,
                progress=False,
                sync_eval=True,
                timeout_s=timeout_s,
                jobs=JOBS,
            )
            yield runs_to_rows(seed, r.runs, fields, extra={"preset": preset, "bm": bm})

        rows = run_seeds(seeds, out, batches)

    # --- fold rows into cells: (seed, family, dim, inst) -> {spec: mean AOCC} ---
    cells, errs, short = fold(
        rows,
        names,
        cell=lambda r: (r["seed"], r["fam"], r["dim"], r["inst"]),
        short_label=lambda r: f"{r['fam']} d{r['dim']}",
    )
    if not cells:
        sys.exit("no rows for the selected specs — nothing to compare")

    dims = sorted({d for _, _, d, _ in cells})
    fams = sorted({f for _, f, _, _ in cells})
    n = len(seeds)

    def score(name, dim=None, fam=None):
        return mean_of(cells, name, match(dim=dim, group=fam))

    print(f"\n=== family screen ===  ({n} seeds, preset {preset}, budget {bm}*d, dims {dims})")
    print(f"specs: {', '.join(names)}   cells: {len(cells)}   families: {', '.join(fams)}")

    # (a) means, overall and per dimension.
    order = sorted(names, key=lambda s: -score(s))
    print_means(cells, order, dims, errs, short, width=28)

    # (b) means per family — the whole point of a multi-class battery.
    # Full family labels: a truncation would merge ``..._crash`` / ``..._tmo``.
    fw = max([15] + [len(f) for f in fams])
    print(f"\n{'spec':28s} " + "".join(f"  {f:>{fw}s}" for f in fams))
    for s in order:
        print(f"{s:28s} " + "".join(f"  {score(s, fam=f):{fw}.4f}" for f in fams))

    # (c) paired deltas against each reference, overall CI + per-dimension means.
    for ref in REFS:
        if ref in names:
            print_delta_table(cells, order, ref, dims, width=28)

    # (d) the portfolio against the best single arm *in this run*, per family.
    best_ref = max((r for r in REFS if r in names), key=score, default=None)
    port = [s for s in names if s.startswith("Blocks")]
    if best_ref and port:
        print(f"\nper-family delta of the portfolio vs the best single arm ({best_ref})")
        print(f"{'family':{fw}s} " + "".join(f"  {p[:20]:>21s}" for p in port))
        for f in fams:
            print(
                f"{f:{fw}s} "
                + "".join(f"  {mean_or_nan(paired(cells, p, best_ref, match(group=f))):+21.4f}" for p in port)
            )

    # (e) what the harness itself reported about the runs.
    print_run_health(names, errs, short)

    print(
        "\nNull floor: on 3 seeds a CMA-ES-containing spec drifts by up to +-0.05 for no\n"
        "reason at all, so a delta inside that band is a direction, not evidence.  Only a\n"
        "CI that excludes zero (marked `<--`) counts, and the 12-seed roster decides."
    )


if __name__ == "__main__":
    main()
