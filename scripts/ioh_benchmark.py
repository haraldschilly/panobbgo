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

    # Save & diff (single seed — subject to the threaded-evaluation noise
    # floor of ~0.015 sd per seed; prefer the paired multi-seed form below
    # for decisions)
    uv run python scripts/ioh_benchmark.py run --quick --output before.json
    # ... make changes ...
    uv run python scripts/ioh_benchmark.py run --quick --output after.json
    uv run python scripts/ioh_benchmark.py compare before.json after.json

    # Paired multi-seed decision A/B (the 2026-08-03 protocol: N >= 12
    # paired quick seeds, mean/sd/CI95 per strategy, flat-control check).
    # --sync-eval halves the scheduling-noise floor (2026-08-09 finding);
    # use it on both sides.
    uv run python scripts/ioh_benchmark.py run --quick --decision-seeds --sync-eval --output before.json
    # ... make changes ...
    uv run python scripts/ioh_benchmark.py run --quick --decision-seeds --sync-eval --output after.json
    uv run python scripts/ioh_benchmark.py compare before.json after.json

    # Standard battery with 5 replicates per (dim, instance) pair
    uv run python scripts/ioh_benchmark.py run --standard --reps 5 --output before.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from panobbgo.benchmark import StrategySpec
from panobbgo.harness import _make_quick_strategies, _make_standard_strategies
from panobbgo.harness_baselines import make_baseline_strategies
from panobbgo.harness_ioh import (
    DEFAULT_DECISION_SEEDS,
    IOHBatterySpec,
    IOHHarnessResult,
    IOHMultiSeedResult,
    make_full_battery,
    make_ioh_strategies,
    make_quick_battery,
    make_standard_battery,
    paired_seed_stats,
    run_ioh_harness,
    run_ioh_harness_multi_seed,
)


def _resolve_battery(args: argparse.Namespace) -> IOHBatterySpec:
    if args.full:
        battery = make_full_battery()
    elif args.standard:
        battery = make_standard_battery()
    else:
        battery = make_quick_battery()
    if args.reps is not None:
        if args.reps < 1:
            raise SystemExit("--reps must be >= 1")
        battery = dataclasses.replace(battery, reps=args.reps)
    return battery


def _resolve_strategies(args: argparse.Namespace) -> List[StrategySpec]:
    if args.legacy:
        # Fall back to the composite-score harness's strategy registry —
        # useful for diffing IOH-tuned specs against the same strategies
        # used by the legacy benchmark_harness.
        strats = list(_make_standard_strategies() if (args.standard or args.full) else _make_quick_strategies())
    else:
        strats = list(make_ioh_strategies())
    if args.baselines:
        strats.extend(make_baseline_strategies())
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


def cmd_run(args: argparse.Namespace) -> int:
    battery = _resolve_battery(args)
    strategies = _resolve_strategies(args)
    if not strategies:
        print("No strategies selected.", file=sys.stderr)
        return 2
    seeds = _resolve_seeds(args)
    print(f"Battery: {battery.name}  dims={battery.dims}  instances={battery.instances}  reps={battery.reps}")
    print(f"Strategies: {[s.name for s in strategies]}")
    print(f"Per-run budget: {[battery.budget_for(d) for d in battery.dims]} (dim={battery.dims})")
    if args.sync_eval:
        print("Eval mode: sync (deterministic result batches; ~2x lower run-to-run noise)")
    result: Any
    if seeds is not None:
        print(f"Seeds ({len(seeds)}): {seeds}")
        result = run_ioh_harness_multi_seed(
            strategies, battery, seeds, progress=not args.quiet, sync_eval=args.sync_eval
        )
    else:
        result = run_ioh_harness(
            strategies, battery, base_seed=args.seed, progress=not args.quiet, sync_eval=args.sync_eval
        )
    result.print_summary()
    if args.output:
        Path(args.output).write_text(result.to_json())
        print(f"\nSaved: {args.output}")
    return 0


def _compare_single(before: IOHHarnessResult, after: IOHHarnessResult, fail_on_regression: bool) -> int:
    delta = after.mean_aocc - before.mean_aocc
    print(f"mean AOCC:  before={before.mean_aocc:.4f}  after={after.mean_aocc:.4f}  delta={delta:+.4f}")
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
    if fail_on_regression and delta < 0:
        return 2
    return 0


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
    if fail_on_regression and delta < 0:
        return 2
    return 0


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


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a battery and report mean AOCC.")
    grp = run_p.add_mutually_exclusive_group()
    grp.add_argument("--quick", action="store_true", help="Small battery (default).")
    grp.add_argument("--standard", action="store_true", help="Mid-sized battery.")
    grp.add_argument("--full", action="store_true", help="Competition-budget battery.")
    run_p.add_argument("--baselines", action="store_true", help="Include external baselines (Random, scipy DE, ...).")
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
        "--reps",
        type=int,
        default=None,
        help="Override battery repetitions per (dim, instance, strategy) — e.g. 5 for standard-battery decisions.",
    )
    run_p.add_argument(
        "--sync-eval",
        action="store_true",
        help="Synchronous-harvest evaluation: deterministic result batches, ~2x lower "
        "run-to-run noise for adaptive strategies. Use on BOTH sides of an A/B "
        "(compare warns on a mode mismatch).",
    )
    run_p.add_argument("--output", help="Save full result as JSON.")
    run_p.add_argument("--quiet", action="store_true", help="Suppress per-run progress lines.")
    run_p.set_defaults(func=cmd_run)

    cmp_p = sub.add_parser("compare", help="Compare two saved IOH harness results.")
    cmp_p.add_argument("before")
    cmp_p.add_argument("after")
    cmp_p.add_argument("--fail-on-regression", action="store_true")
    cmp_p.set_defaults(func=cmd_compare)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
