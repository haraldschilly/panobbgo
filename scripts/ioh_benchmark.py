#!/usr/bin/env python
# -*- coding: utf8 -*-
"""IOH (MA-BBOB anytime) benchmark CLI.

Companion to ``benchmark_harness.py`` but scoring on the IOHprofiler
MA-BBOB suite with the AOCC metric used by the MA-BBOB Anytime
competition.

Examples::

    # Quick run — ~10 sec, 3 instances at dim 2, default Panobbgo strategies
    uv run python scripts/ioh_benchmark.py run --quick

    # Standard run with random + scipy DE baselines for context
    uv run python scripts/ioh_benchmark.py run --standard --baselines

    # Save & diff (single seed; runs are synchronous by default, so a
    # seeded run is reproducible — prefer the paired multi-seed form below
    # for decisions)
    uv run python scripts/ioh_benchmark.py run --quick --output before.json
    # ... make changes ...
    uv run python scripts/ioh_benchmark.py run --quick --output after.json
    uv run python scripts/ioh_benchmark.py compare before.json after.json

    # Paired multi-seed decision A/B (the 2026-08-03 protocol: N >= 12
    # paired quick seeds, mean/sd/CI95 per strategy, flat-control check).
    # Synchronous evaluation is the default (reproducible seeded runs);
    # --no-sync-eval opts into the threaded evaluator — same mode on both sides.
    uv run python scripts/ioh_benchmark.py run --quick --decision-seeds --output before.json
    # ... make changes ...
    uv run python scripts/ioh_benchmark.py run --quick --decision-seeds --output after.json
    uv run python scripts/ioh_benchmark.py compare before.json after.json

    # Standard battery with 5 replicates per (dim, instance) pair
    uv run python scripts/ioh_benchmark.py run --standard --reps 5 --output before.json

    # Generated problem families instead of MA-BBOB — many cheap classes,
    # including constrained ones, which no IOH battery covers.  Same AOCC,
    # same records, same --output format (panobbgo.harness_families).
    uv run python scripts/ioh_benchmark.py run --families
    uv run python scripts/ioh_benchmark.py run --families-constrained
    uv run python scripts/ioh_benchmark.py run --families-quick   # smoke test

    # The two regimes GOAL.md 2c asks for: noise, and dimension.  AOCC on a
    # noisy battery is scored on the TRUE value (panobbgo.lib.noise).
    uv run python scripts/ioh_benchmark.py run --noisy gauss
    uv run python scripts/ioh_benchmark.py run --noisy cauchy --noisy-severe
    uv run python scripts/ioh_benchmark.py run --highdim          # d = 10, 20; slow
    uv run python scripts/ioh_benchmark.py run --noisy-highdim gauss

    # Dimensions 30/40 (opt-in; the presets above keep their dims), and a
    # bbob-largescale-style slice (plain BBOB, 5 functions at d = 80, 160).
    uv run python scripts/ioh_benchmark.py run --large
    uv run python scripts/ioh_benchmark.py run --families-large
    uv run python scripts/ioh_benchmark.py run --largescale

    # The SEALED test set: only to report a result or back a claim, never
    # for tuning or screening (doc/dev/benchmarking.md, "The sealed test set").
    uv run python scripts/ioh_benchmark.py run --sealed --decision-seeds --output claim.json
    uv run python scripts/ioh_benchmark.py run --families-sealed --output claim_families.json

    # Parallel behaviour on a virtual clock (deterministic, no waiting):
    # q simulated workers, AOCC over evaluations AND over virtual time.
    for q in 1 4 16 64; do
      uv run python scripts/ioh_benchmark.py run --families --virtual-workers $q --duration lognormal --output virtual_q$q.json
    done
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from panobbgo.benchmark import StrategySpec
from panobbgo import local_run
from panobbgo.harness import _make_quick_strategies, _make_standard_strategies
from panobbgo.harness_baselines import check_baseline_selection, make_baseline_strategies
from panobbgo.harness_families import (
    FamilyInstances,
    make_constrained_battery,
    make_families_battery,
    make_large_families_battery,
    make_sealed_families_battery,
    run_family_harness,
)
from panobbgo.harness_ioh import (
    AOCC_LOG_HI,
    AOCC_LOG_LO,
    DEFAULT_DECISION_SEEDS,
    IOHBatterySpec,
    IOHHarnessResult,
    IOHMultiSeedResult,
    make_full_battery,
    make_highdim_battery,
    make_ioh_strategies,
    make_large_battery,
    make_largescale_battery,
    make_noisy_battery,
    make_noisy_highdim_battery,
    make_quick_battery,
    make_sealed_battery,
    make_standard_battery,
    paired_seed_stats,
    run_ioh_harness,
    run_ioh_harness_multi_seed,
    t_ci,
)
from panobbgo.virtual_clock import VirtualSpec


def _resolve_battery(args: argparse.Namespace) -> IOHBatterySpec:
    if args.full:
        battery = make_full_battery()
    elif args.standard:
        battery = make_standard_battery()
    elif getattr(args, "noisy", None):
        # Standard cube, BBOB-style noise on the objective, AOCC scored on
        # the TRUE value (panobbgo.lib.noise).  --noisy-severe switches the
        # BBOB f101-f106 "moderate" parameters for the f107-f130 ones.
        battery = make_noisy_battery(args.noisy, level="severe" if args.noisy_severe else "moderate")
    elif getattr(args, "noisy_highdim", None):
        battery = make_noisy_highdim_battery(args.noisy_highdim, level="severe" if args.noisy_severe else "moderate")
    elif getattr(args, "highdim", False):
        battery = make_highdim_battery()
    elif getattr(args, "large", False):
        battery = make_large_battery()
    elif getattr(args, "largescale", False):
        battery = make_largescale_battery()
    elif getattr(args, "sealed", False):
        battery = make_sealed_battery()
    else:
        battery = make_quick_battery()
    if args.reps is not None:
        if args.reps < 1:
            raise SystemExit("--reps must be >= 1")
        battery = dataclasses.replace(battery, reps=args.reps)
    bm = getattr(args, "budget_multiplier", None)
    if bm is not None:
        battery = dataclasses.replace(battery, budget_multiplier=bm, name=f"{battery.name}-b{bm}")
    return battery


#: Budget multiplier of the family batteries — ``budget = FAMILY_BUDGET_MULTIPLIER * dim``,
#: matching the standard IOH battery so the two tracks' budgets are comparable.
FAMILY_BUDGET_MULTIPLIER: int = 500


def _resolve_family_battery(args: argparse.Namespace) -> Optional[Tuple[str, FamilyInstances, int]]:
    """``(battery name, instances, budget multiplier)`` for the family track, else ``None``.

    The family track (:mod:`panobbgo.harness_families`) scores generated
    problem *families* — including constrained ones, which no IOH battery
    covers — with the same AOCC machinery.  ``None`` means no family flag
    was given and the caller should take the MA-BBOB path.
    """
    family = _family_battery(args)
    bm = getattr(args, "budget_multiplier", None)
    if family is None or bm is None:
        return family
    name, instances, _ = family
    return f"{name}-b{bm}", instances, bm


def _family_battery(args: argparse.Namespace) -> Optional[Tuple[str, FamilyInstances, int]]:
    """The family battery the flags name, at its own budget multiplier."""
    if args.families_quick:
        # Two instances at dim 2, 100 evaluations each: a smoke test, not a
        # measurement.  One unconstrained (Rosenbrock) and one constrained
        # (sphere, one active linear constraint) so the run exercises both
        # the plain and the penalty-tracker path.  Deliberately not a preset
        # in ``harness_families`` — the presets there are contracts.
        quick = make_families_battery(dims=(2,), n_instances=1)[1:2]
        quick += make_constrained_battery(dims=(2,), n_instances=1)[:1]
        return "families-quick", quick, 50
    if args.families_constrained:
        return "families-constrained", make_constrained_battery(), FAMILY_BUDGET_MULTIPLIER
    if getattr(args, "families_large", False):
        return "families-large", make_large_families_battery(), FAMILY_BUDGET_MULTIPLIER
    if getattr(args, "families_sealed", False):
        return "sealed-families", make_sealed_families_battery(), FAMILY_BUDGET_MULTIPLIER
    if args.families:
        return "families", make_families_battery(), FAMILY_BUDGET_MULTIPLIER
    return None


def _resolve_strategies(args: argparse.Namespace) -> List[StrategySpec]:
    if args.legacy:
        # Fall back to the composite-score harness's strategy registry —
        # useful for diffing IOH-tuned specs against the same strategies
        # used by the legacy benchmark_harness.
        strats = list(_make_standard_strategies() if (args.standard or args.full) else _make_quick_strategies())
    else:
        strats = list(make_ioh_strategies())
    try:
        check_baseline_selection(args.strategies, args.baselines)
        if args.baselines:
            # External baselines (pycma, Nevergrad, Optuna) join only when --strategies names them.
            strats.extend(make_baseline_strategies(args.strategies))
    except (ValueError, ImportError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.strategies:
        wanted = set(args.strategies)
        strats = [s for s in strats if s.name in wanted]
        missing = wanted - {s.name for s in strats}
        if missing:
            print(f"warning: unknown strategy names ignored: {sorted(missing)}", file=sys.stderr)
    return strats


def _resolve_seeds(args: argparse.Namespace) -> Optional[List[int]]:
    """Return the multi-seed roster, or ``None`` for a single-seed run."""
    if args.decision_seeds:
        return list(DEFAULT_DECISION_SEEDS)
    if args.seeds:
        return [int(s) for s in args.seeds]
    return None


def _print_eval_mode(sync_eval: bool, virtual: Optional[VirtualSpec] = None) -> None:
    if virtual is not None:
        print(f"Eval mode: virtual clock {virtual.to_dict()} (deterministic; no real waiting)")
    elif sync_eval:
        print("Eval mode: sync (deterministic result batches; reproducible seeded runs)")
    else:
        print("Eval mode: async (--no-sync-eval; trajectories depend on thread scheduling)")


def _resolve_virtual(args: argparse.Namespace) -> Optional[VirtualSpec]:
    q = getattr(args, "virtual_workers", None)
    if q is None:
        return None
    return VirtualSpec(
        workers=int(q),
        duration=args.duration or "constant",
        sigma=0.5 if args.duration_sigma is None else float(args.duration_sigma),
        policy=args.virtual_policy or "async",
    )


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {text!r}") from None
    if v < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {v}")
    return v


def _non_negative_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number >= 0, got {text!r}") from None
    if not (v >= 0 and np.isfinite(v)):
        raise argparse.ArgumentTypeError(f"expected a finite number >= 0, got {v}")
    return v


#: Flags whose batteries refuse ``--legacy`` (the composite registry's GP /
#: quadratic heuristics do not scale to these dims) and, for the sealed ones,
#: ``--reps``.
_LARGE_FLAGS = ("large", "largescale", "families_large", "sealed", "families_sealed")
_SEALED_FLAGS = ("sealed", "families_sealed")


def _check_battery_options(args: argparse.Namespace) -> None:
    """Refuse the option combinations the large and sealed batteries do not take."""
    chosen = [f for f in _LARGE_FLAGS if getattr(args, f, False)]
    if not chosen:
        return
    flag = "--" + chosen[0].replace("_", "-")
    if getattr(args, "legacy", False):
        raise SystemExit(
            f"error: {flag} does not take --legacy: the composite registry's GP / QuadraticWLS / "
            "Nearby(quadratic) heuristics are not for d >= 30 (a Nearby quadratic fit takes seconds and "
            "up to 1 GB per new best at d = 160)"
        )
    if chosen[0] in _SEALED_FLAGS and getattr(args, "reps", None) is not None:
        raise SystemExit(f"error: {flag} is fixed: no --reps (panobbgo.sealed)")


def cmd_run(args: argparse.Namespace) -> int:
    _check_battery_options(args)
    strategies = _resolve_strategies(args)
    if not strategies:
        print("No strategies selected.", file=sys.stderr)
        return 2
    virtual = _resolve_virtual(args)
    result: Any
    family = _resolve_family_battery(args)
    if family is not None:
        name, instances, budget_multiplier = family
        dims = sorted({p.dim for _n, p in instances})
        print(
            f"Battery: {name}  instances={len(instances)}  dims={dims}  families={len({p.family for _n, p in instances})}"
        )
        print(f"Strategies: {[s.name for s in strategies]}")
        print(f"Per-run budget: {[budget_multiplier * d for d in dims]} (dim={dims})")
        _print_eval_mode(args.sync_eval, virtual)
        if _resolve_seeds(args) is not None:
            # The multi-seed roster is an IOHBatterySpec construction;
            # ``benchmarks/family_screen.py`` is the multi-seed instrument
            # for this track.  Say so rather than silently running one seed.
            print(
                "warning: --seeds/--decision-seeds are not wired for the family track; "
                f"running the single --seed {args.seed}.  Use benchmarks/family_screen.py "
                "for a paired multi-seed screen.",
                file=sys.stderr,
            )
        result = run_family_harness(
            strategies,
            instances,
            budget_multiplier=budget_multiplier,
            base_seed=args.seed,
            sync_eval=args.sync_eval,
            reps=args.reps or 1,
            progress=not args.quiet,
            battery_name=name,
            timeout_s=args.timeout,
            jobs=args.jobs,
            virtual=virtual,
        )
    else:
        battery = _resolve_battery(args)
        seeds = _resolve_seeds(args)
        print(f"Battery: {battery.name}  dims={battery.dims}  instances={battery.instances}  reps={battery.reps}")
        print(f"Strategies: {[s.name for s in strategies]}")
        print(f"Per-run budget: {[battery.budget_for(d) for d in battery.dims]} (dim={battery.dims})")
        _print_eval_mode(args.sync_eval, virtual)
        if seeds is not None:
            print(f"Seeds ({len(seeds)}): {seeds}")
            result = run_ioh_harness_multi_seed(
                strategies,
                battery,
                seeds,
                progress=not args.quiet,
                sync_eval=args.sync_eval,
                timeout_s=args.timeout,
                jobs=args.jobs,
                virtual=virtual,
            )
        else:
            result = run_ioh_harness(
                strategies,
                battery,
                base_seed=args.seed,
                progress=not args.quiet,
                sync_eval=args.sync_eval,
                timeout_s=args.timeout,
                jobs=args.jobs,
                virtual=virtual,
            )
    result.print_summary()
    if args.output:
        Path(args.output).write_text(result.to_json())
        print(f"\nSaved: {args.output}")
    return 0


def _compare_single(before: IOHHarnessResult, after: IOHHarnessResult, fail_on_regression: bool) -> int:
    delta = after.mean_aocc - before.mean_aocc
    print(f"mean AOCC:  before={before.mean_aocc:.4f}  after={after.mean_aocc:.4f}  delta={delta:+.4f}")
    _print_time_delta(before, after)
    print("\n  per strategy (after - before):")
    p_before = before.per_strategy_aocc()
    p_after = after.per_strategy_aocc()
    names = sorted(set(p_before) | set(p_after))
    for name in names:
        b = p_before.get(name, float("nan"))
        a = p_after.get(name, float("nan"))
        marker = ""
        if not (b != b or a != a):  # both not NaN
            d = a - b
            marker = "  +" if d > 0 else ("  -" if d < 0 else "  =")
        print(f"    {name:32s}  {b:.4f} -> {a:.4f}{marker}")
    for side, res in (("before", before), ("after", after)):
        bad = {
            k: sum(getattr(r, k) for r in res.runs)
            for k in ("crashed", "timed_out", "ended_early")
            if any(getattr(r, k) for r in res.runs)
        }
        if bad:
            print(f"\n  {side}: " + ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in bad.items()) + " run(s)")
    common, gate_delta = _common_cell_delta(before, after)
    if not common == len(before.runs) == len(after.runs):
        print(
            f"\n  common (strategy, cell) runs: {common} of {len(before.runs)} before / {len(after.runs)} after; "
            f"delta over them = {gate_delta:+.4f}"
        )
    if fail_on_regression:
        if common == 0:
            print("No (strategy, cell) run appears in both results; cannot gate.", file=sys.stderr)
            return 2
        if gate_delta < 0:
            return 2
    return 0


def _run_key(r: Any) -> Tuple[Any, ...]:
    return (r.strategy_name, r.problem_kind, r.fid, r.dim, r.instance, r.rep)


def _common_cell_delta(before: IOHHarnessResult, after: IOHHarnessResult) -> Tuple[int, float]:
    """``(n, mean AOCC delta)`` over the (strategy, cell, rep) runs present on both sides.

    The regression gate compares like with like: a strategy (e.g. a
    baseline) or a cell that exists on one side only would otherwise move
    the overall mean by its own level, not by a change.
    """
    b = {_run_key(r): r.aocc for r in before.runs}
    a = {_run_key(r): r.aocc for r in after.runs}
    keys = [k for k in b if k in a]
    if not keys:
        return 0, float("nan")
    return len(keys), float(sum(a[k] - b[k] for k in keys) / len(keys))


def _compare_multi(before: IOHMultiSeedResult, after: IOHMultiSeedResult, fail_on_regression: bool) -> int:
    stats = paired_seed_stats(before, after)
    if not stats:
        print("No strategy appears in both results; nothing to compare.", file=sys.stderr)
        return 2
    any_stat = next(iter(stats.values()))
    common_seeds = any_stat["seeds"]
    print(f"Paired multi-seed compare  (battery {before.battery_name}, {len(common_seeds)} common seed(s))")
    print(f"  seeds: {', '.join(str(s) for s in common_seeds)}")
    delta = after.mean_aocc - before.mean_aocc
    print(f"mean AOCC:  before={before.mean_aocc:.4f}  after={after.mean_aocc:.4f}  delta={delta:+.4f}")
    _print_time_delta(before, after)
    print("\n  per strategy (paired by seed; CI95 via t-dist on the per-seed deltas):")
    print(f"    {'strategy':32s}  {'before':>7s}  {'after':>7s}  {'Δmean':>8s}  {'sd':>7s}  {'CI95':>19s}")
    for name, st in stats.items():
        if st["n"] >= 2:
            ci = f"[{st['ci_low']:+.4f},{st['ci_high']:+.4f}]"
            if st["ci_low"] > 0:
                verdict = "+ improved"
            elif st["ci_high"] < 0:
                verdict = "- regressed"
            else:
                verdict = "~ noise"
        else:
            ci = "[n/a]"
            verdict = "? n<2"
        print(
            f"    {name:32s}  {st['before_mean']:7.4f}  {st['after_mean']:7.4f}  "
            f"{st['mean_delta']:+8.4f}  {st['sd']:7.4f}  {ci:>19s}  {verdict}"
        )
    print("\n  per-seed deltas:")
    for name, st in stats.items():
        deltas = "  ".join(f"{s}:{d:+.4f}" for s, d in zip(st["seeds"], st["per_seed_delta"]))
        print(f"    {name:32s}  {deltas}")
    # The gate: per common seed, the mean delta over the strategies present
    # on both sides; regression iff the t-CI95 of those per-seed deltas lies
    # below zero (the mean delta alone with a single common seed).
    per_seed = [float(np.mean([st["per_seed_delta"][i] for st in stats.values()])) for i in range(len(common_seeds))]
    pooled, half = t_ci(per_seed)
    if len(per_seed) >= 2:
        regressed = pooled + half < 0
        print(
            f"\n  pooled over common strategies: Δmean={pooled:+.4f}  CI95=[{pooled - half:+.4f},{pooled + half:+.4f}]"
        )
    else:
        regressed = pooled < 0
        print(f"\n  pooled over common strategies: Δmean={pooled:+.4f}  (one common seed: no CI)")
    if fail_on_regression and regressed:
        return 2
    return 0


def _virtual_of(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The virtual-clock settings of a result file (a multi-seed file: its first seed's)."""
    if d.get("virtual") is not None:
        return d["virtual"]
    results = d.get("results") or []
    return results[0].get("virtual") if d.get("multi_seed") and results else None


def _print_time_delta(before: Any, after: Any) -> None:
    b, a = before.mean_aocc_time, after.mean_aocc_time
    if b is not None and a is not None:
        print(f"AOCC (time): before={b:.4f}  after={a:.4f}  delta={a - b:+.4f}")


def cmd_compare(args: argparse.Namespace) -> int:
    d_before: Dict[str, Any] = json.loads(Path(args.before).read_text())
    d_after: Dict[str, Any] = json.loads(Path(args.after).read_text())
    b_sync = bool(d_before.get("sync_eval"))
    a_sync = bool(d_after.get("sync_eval"))
    if b_sync != a_sync:
        print(
            f"warning: evaluation-mode mismatch ({args.before} sync_eval={b_sync}, "
            f"{args.after} sync_eval={a_sync}) — the two sides were measured under "
            "different scheduling regimes; deltas are not decision-grade.",
            file=sys.stderr,
        )
    b_virtual, a_virtual = _virtual_of(d_before), _virtual_of(d_after)
    if b_virtual != a_virtual:
        print(
            f"warning: virtual-clock mismatch ({args.before} virtual={b_virtual}, {args.after} "
            f"virtual={a_virtual}) — different worker counts, duration models or policies are not "
            "comparable" + ("; --fail-on-regression refuses to gate." if args.fail_on_regression else "."),
            file=sys.stderr,
        )
        if args.fail_on_regression:
            return 2
    bounds = [(float(d.get("log_lo", AOCC_LOG_LO)), float(d.get("log_hi", AOCC_LOG_HI))) for d in (d_before, d_after)]
    if bounds[0] != bounds[1]:
        # AOCC over different target ranges is a different number (the
        # largescale slice is scored up to 1e6, not 1e2): refuse outright.
        print(
            f"Cannot compare AOCC over different target ranges ({args.before} log bounds {bounds[0]}, "
            f"{args.after} {bounds[1]}).",
            file=sys.stderr,
        )
        return 2
    for key, what in (("sealed", "one side is the sealed test set"), ("blas_threads", "BLAS thread counts differ")):
        if d_before.get(key) != d_after.get(key):
            print(
                f"warning: {key} mismatch ({args.before} {key}={d_before.get(key)}, {args.after} "
                f"{key}={d_after.get(key)}): {what}.",
                file=sys.stderr,
            )
    b_multi = bool(d_before.get("multi_seed"))
    a_multi = bool(d_after.get("multi_seed"))
    if b_multi != a_multi:
        print(
            "Cannot compare a multi-seed result with a single-seed result "
            f"({args.before} multi_seed={b_multi}, {args.after} multi_seed={a_multi}). "
            "Re-run the missing side with the same --seeds / --decision-seeds roster.",
            file=sys.stderr,
        )
        return 2
    if b_multi:
        return _compare_multi(
            IOHMultiSeedResult.from_dict(d_before),
            IOHMultiSeedResult.from_dict(d_after),
            args.fail_on_regression,
        )
    return _compare_single(
        IOHHarnessResult.from_dict(d_before),
        IOHHarnessResult.from_dict(d_after),
        args.fail_on_regression,
    )


def main(argv: Optional[List[str]] = None, apply_hygiene: bool = False) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a battery and report mean AOCC.")
    grp = run_p.add_mutually_exclusive_group()
    grp.add_argument("--quick", action="store_true", help="Small battery (default).")
    grp.add_argument("--standard", action="store_true", help="Mid-sized battery.")
    grp.add_argument("--full", action="store_true", help="Competition-budget battery.")
    grp.add_argument(
        "--families",
        action="store_true",
        help="Generated problem families instead of MA-BBOB: 5 families x dims (2, 5, 10) x 3 instances, "
        "shifted/rotated, known optimum (panobbgo.harness_families).",
    )
    grp.add_argument(
        "--families-constrained",
        action="store_true",
        help="Constrained family battery: 4 families x dims (2, 5) x 3 instances, k=1..3 constraints "
        "active at the optimum.  AOCC is scored on the penalty value f + 100*cv.",
    )
    grp.add_argument(
        "--families-quick",
        action="store_true",
        help="Smoke test of the family track: 2 instances at dim 2 (one unconstrained, one "
        "constrained), 100 evaluations each. Not a measurement.",
    )
    grp.add_argument(
        "--noisy",
        choices=("gauss", "unif", "cauchy"),
        help="Standard MA-BBOB cube with BBOB-style noise on the objective. AOCC is scored on "
        "the TRUE (noise-free) value, as the BBOB noisy suite does; what the optimizer observed "
        "is reported alongside (panobbgo.lib.noise).",
    )
    grp.add_argument(
        "--highdim",
        action="store_true",
        help="Noiseless MA-BBOB at dims (10, 20), instances 0-2, budget 2000*d. Expensive: "
        "~18 s per run at d=10 and ~55 s at d=20.",
    )
    grp.add_argument(
        "--noisy-highdim",
        choices=("gauss", "unif", "cauchy"),
        help="Both regimes crossed cheaply: noise at dim 10, instances 0-2, budget 500*d.",
    )
    grp.add_argument(
        "--large",
        action="store_true",
        help="Noiseless MA-BBOB at dims (30, 40), instances 0-2, budget 500*d: ~7-9 s per run at d=40.",
    )
    grp.add_argument(
        "--largescale",
        action="store_true",
        help="A bbob-largescale-style slice: plain BBOB f2/f8/f10/f15/f21 at dims (80, 160), instances 0-2, "
        "budget 500*d, AOCC targets up to 1e6 (full rotations, not COCO's block rotations).",
    )
    grp.add_argument(
        "--families-large",
        action="store_true",
        help="The free and shapes families at dims (30, 40), 3 instances, budget 500*d.",
    )
    grp.add_argument(
        "--sealed",
        action="store_true",
        help="SEALED MA-BBOB test set (20 fresh instances, dims 2-40, 500*d): only to report a result or "
        "back a claim, never for tuning or screening (doc/dev/benchmarking.md).  No --reps / --legacy.",
    )
    grp.add_argument(
        "--families-sealed",
        action="store_true",
        help="SEALED family test set (fresh instances of every family class, dims 2-40, 500*d): claims "
        "only, never for tuning or screening (doc/dev/benchmarking.md).  No --reps / --legacy.",
    )
    run_p.add_argument(
        "--noisy-severe",
        action="store_true",
        help="With --noisy / --noisy-highdim: the BBOB *severe* noise parameters (f107-f130) "
        "instead of the moderate ones (f101-f106).",
    )
    run_p.add_argument(
        "--baselines",
        action="store_true",
        help=(
            "Include external baselines (Random, scipy DE, scipy dual annealing); pycma / Nevergrad / Optuna"
            " ones (Baseline_NGOpt, ...; needs --extra baselines) join when --strategies names them."
        ),
    )
    run_p.add_argument(
        "--legacy",
        action="store_true",
        help="Use the legacy composite-score strategy registry instead of make_ioh_strategies().",
    )
    run_p.add_argument("--strategies", nargs="+", help="Restrict to these strategy names.")
    run_p.add_argument("--seed", type=int, default=42)
    seed_grp = run_p.add_mutually_exclusive_group()
    seed_grp.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Run the battery once per seed and save a paired multi-seed result (see compare).",
    )
    seed_grp.add_argument(
        "--decision-seeds",
        action="store_true",
        help=f"Shorthand for --seeds {' '.join(str(s) for s in DEFAULT_DECISION_SEEDS)} "
        "(the canonical 12-seed decision roster).",
    )
    run_p.add_argument(
        "--budget-multiplier",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Override the battery's budget: N*dim evaluations per run (IOH and family batteries; the "
        "battery name gets a -bN suffix).  E.g. 20 or 100 for the expensive-track baselines "
        "(Baseline_BoTorch_qLogEI, ...), whose GP fits cannot afford 500*dim.  Default: the battery's own.",
    )
    run_p.add_argument(
        "--reps",
        type=int,
        default=None,
        help="Override battery repetitions per (dim, instance, strategy) — e.g. 5 for standard-battery decisions.",
    )
    run_p.add_argument(
        "--sync-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronous-harvest evaluation (default: on, for every battery): "
        "deterministic result batches, so a seeded run is reproducible.  "
        "--no-sync-eval opts into the threaded evaluator, whose trajectory depends on "
        "thread scheduling.  Use the same mode on BOTH sides of an A/B (compare warns "
        "on a mismatch; result files from before 2026-09-25 default to async).",
    )
    run_p.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Per-run wall-clock deadline (default: none).  A run past it is stopped, "
        "scored on its trajectory so far and marked with a TimeoutError.  A wedged IOH "
        "worker is bounded separately by IOHProblem.call_timeout (300 s per round-trip).",
    )
    run_p.add_argument(
        "--virtual-workers",
        type=_positive_int,
        default=None,
        metavar="Q",
        help="Simulate Q parallel workers on a virtual clock (panobbgo.virtual_clock): deterministic, "
        "no real waiting.  Also scores AOCC over virtual time (aocc_time).  Default: off.",
    )
    run_p.add_argument(
        "--duration",
        choices=("constant", "lognormal"),
        default=None,
        help="Duration model of the virtual clock (mean 1): constant (default), or lognormal with "
        "--duration-sigma.  Needs --virtual-workers.",
    )
    run_p.add_argument(
        "--duration-sigma",
        type=_non_negative_float,
        default=None,
        help="Log-space standard deviation of --duration lognormal (default 0.5).  Needs --virtual-workers.",
    )
    run_p.add_argument(
        "--virtual-policy",
        choices=("async", "sync"),
        default=None,
        help="Dispatch policy of the virtual clock: async (default; a decision at every completion, "
        "candidates only for the free workers) or sync (the synchronous batch policy, a regression mode).  "
        "Needs --virtual-workers.",
    )
    run_p.add_argument("--output", help="Save full result as JSON.")
    run_p.add_argument("--quiet", action="store_true", help="Suppress per-run progress lines.")
    local_run.add_jobs_argument(run_p)
    run_p.set_defaults(func=cmd_run)

    cmp_p = sub.add_parser("compare", help="Compare two saved IOH harness results.")
    cmp_p.add_argument("before")
    cmp_p.add_argument("after")
    cmp_p.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit 2 on a regression, judged on the strategies (and, single-seed, the cells) present on "
        "both sides only.  Multi-seed: the t-CI95 of the per-seed mean delta lies below 0.  "
        "Single-seed: the mean delta over the common runs is negative.",
    )
    cmp_p.set_defaults(func=cmd_compare)

    local_run.add_arguments(p)
    args = p.parse_args(argv)
    if getattr(args, "cmd", None) == "run" and args.virtual_workers is None:
        given = [
            flag
            for flag, v in (
                ("--duration", args.duration),
                ("--duration-sigma", args.duration_sigma),
                ("--virtual-policy", args.virtual_policy),
            )
            if v is not None
        ]
        if given:
            p.error(f"{', '.join(given)} only apply to the virtual clock: add --virtual-workers Q")
    if apply_hygiene:
        local_run.apply(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(apply_hygiene=True))
