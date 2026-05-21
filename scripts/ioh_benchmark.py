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

    # Save & diff
    uv run python scripts/ioh_benchmark.py run --quick --output before.json
    # ... make changes ...
    uv run python scripts/ioh_benchmark.py run --quick --output after.json
    uv run python scripts/ioh_benchmark.py compare before.json after.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from panobbgo.benchmark import StrategySpec
from panobbgo.harness import _make_quick_strategies, _make_standard_strategies
from panobbgo.harness_baselines import make_baseline_strategies
from panobbgo.harness_ioh import (
    IOHBatterySpec,
    IOHHarnessResult,
    make_full_battery,
    make_ioh_strategies,
    make_quick_battery,
    make_standard_battery,
    run_ioh_harness,
)


def _resolve_battery(args: argparse.Namespace) -> IOHBatterySpec:
    if args.full:
        return make_full_battery()
    if args.standard:
        return make_standard_battery()
    return make_quick_battery()


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


def cmd_run(args: argparse.Namespace) -> int:
    battery = _resolve_battery(args)
    strategies = _resolve_strategies(args)
    if not strategies:
        print("No strategies selected.", file=sys.stderr)
        return 2
    print(f"Battery: {battery.name}  dims={battery.dims}  instances={battery.instances}  reps={battery.reps}")
    print(f"Strategies: {[s.name for s in strategies]}")
    print(f"Per-run budget: {[battery.budget_for(d) for d in battery.dims]} (dim={battery.dims})")
    result = run_ioh_harness(strategies, battery, base_seed=args.seed, progress=not args.quiet)
    result.print_summary()
    if args.output:
        Path(args.output).write_text(result.to_json())
        print(f"\nSaved: {args.output}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    before = IOHHarnessResult.from_dict(json.loads(Path(args.before).read_text()))
    after = IOHHarnessResult.from_dict(json.loads(Path(args.after).read_text()))
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
    if args.fail_on_regression and delta < 0:
        return 2
    return 0


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
