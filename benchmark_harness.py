#!/usr/bin/env python
# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Benchmark Harness CLI
=====================

Command-line interface for the panobbgo benchmark harness, designed as the
primary tool for agent-driven unsupervised improvement cycles.

Subcommands
-----------

``run``
    Execute benchmarks and write a JSON result file.  Use this before and
    after each code change to capture performance snapshots::

        uv run python benchmark_harness.py run --quick
        uv run python benchmark_harness.py run --standard --output after.json

``score``
    Print the composite score and per-pair breakdown for a result file::

        uv run python benchmark_harness.py score results.json

``compare``
    Compare two result files and show what improved or degraded::

        uv run python benchmark_harness.py compare before.json after.json

Typical agent loop
------------------

1. Run baseline::

       uv run python benchmark_harness.py run --quick --output baseline.json

2. Make a change (edit a heuristic, tune a parameter, …).

3. Run again::

       uv run python benchmark_harness.py run --quick --output candidate.json

4. Compare::

       uv run python benchmark_harness.py compare baseline.json candidate.json

5. Keep changes if ``composite_score`` improved; revert otherwise.

6. Repeat.

Exit codes
----------
- ``0`` – success
- ``1`` – argument/file error
- ``2`` – candidate is worse than baseline (useful for scripted gating)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime


def _default_output_path(mode: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"harness_{mode}_{ts}.json"


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    """Execute benchmark runs and save the result."""
    from panobbgo.harness import BenchmarkHarness, HarnessConfig

    mode = args.mode
    config = HarnessConfig(
        mode=mode,
        budget=args.budget,
        reps=args.reps,
        seed=args.seed,
        problems=args.problems or None,
        strategies=args.strategies or None,
        timeout_per_run=args.timeout,
    )

    harness = BenchmarkHarness(config)
    result = harness.run(verbose=not args.quiet)

    output = args.output or _default_output_path(mode)
    result.save(output)

    print(f"\nComposite Score: {result.composite_score:.4f}")
    print(f"Saved to: {output}")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: score
# ---------------------------------------------------------------------------


def cmd_score(args: argparse.Namespace) -> int:
    """Print score breakdown for a saved result file."""
    from panobbgo.harness import HarnessResult

    path = args.result_file
    if not pathlib.Path(path).exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    result = HarnessResult.load(path)
    result.print_summary()

    # Machine-readable JSON output for agent consumption
    if args.json:
        summary = {
            "composite_score": result.composite_score,
            "mode": result.config.mode,
            "budget": result.config.effective_budget(),
            "reps": result.config.effective_reps(),
            "total_runs": result.total_runs,
            "total_duration": result.total_duration,
            "timestamp": result.timestamp,
            "per_pair": [
                {
                    "problem": psr.problem_name,
                    "strategy": psr.strategy_name,
                    "score": psr.score,
                    "success_rate": psr.success_rate,
                    "ert": psr.ert if psr.ert < float("inf") else None,
                    "best_func_distance": (
                        psr.best_func_distance
                        if psr.best_func_distance < float("inf")
                        else None
                    ),
                }
                for psr in result.problem_strategy_results
            ],
        }
        print(json.dumps(summary, indent=2))

    return 0


# ---------------------------------------------------------------------------
# Subcommand: compare
# ---------------------------------------------------------------------------


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two result files and report improvements/regressions."""
    from panobbgo.harness import HarnessResult, compare as harness_compare

    for path in (args.before, args.after):
        if not pathlib.Path(path).exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            return 1

    before = HarnessResult.load(args.before)
    after = HarnessResult.load(args.after)

    comparison = harness_compare(
        before,
        after,
        eps=args.eps,
        label_before=args.before,
        label_after=args.after,
    )
    comparison.print_summary()

    if args.json:
        out = {
            "score_before": comparison.score_before,
            "score_after": comparison.score_after,
            "delta": comparison.delta,
            "relative_delta_pct": comparison.relative_delta,
            "improved": [
                {"problem": p, "strategy": s, "before": b, "after": a}
                for p, s, b, a in comparison.improved
            ],
            "degraded": [
                {"problem": p, "strategy": s, "before": b, "after": a}
                for p, s, b, a in comparison.degraded
            ],
        }
        print(json.dumps(out, indent=2))

    # Non-zero exit if candidate is strictly worse (useful for CI gating)
    if args.fail_on_regression and comparison.delta < -args.eps:
        print(
            f"\nFAIL: composite score decreased by {comparison.delta:.4f}",
            file=sys.stderr,
        )
        return 2

    return 0


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    """List available problems and strategies for a given mode."""
    from panobbgo.harness import BenchmarkHarness, HarnessConfig

    config = HarnessConfig(mode=args.mode)
    harness = BenchmarkHarness(config)

    problems = harness.get_problems()
    strategies = harness.get_strategies()

    print(f"\nProblems ({len(problems)}) for mode={args.mode!r}:")
    for p in problems:
        print(
            f"  {p.name}  dim={p.dims}  tol={p.tolerance}  budget={p.max_evaluations}"
        )

    print(f"\nStrategies ({len(strategies)}) for mode={args.mode!r}:")
    for s in strategies:
        heuristics = ", ".join(h.__name__ for h, _ in s.heuristics)
        print(f"  {s.name}  [{heuristics}]")

    print()
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="benchmark_harness.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ---- run ----------------------------------------------------------------
    run_p = sub.add_parser("run", help="Run benchmarks and save results")
    _add_mode_group(run_p)
    run_p.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Output JSON file (default: harness_<mode>_<timestamp>.json)",
    )
    run_p.add_argument(
        "--budget", type=int, metavar="N", help="Override evaluations per run"
    )
    run_p.add_argument(
        "--reps",
        type=int,
        metavar="N",
        help="Override repetitions per (problem, strategy)",
    )
    run_p.add_argument(
        "--seed", type=int, default=42, help="Base random seed (default: 42)"
    )
    run_p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        metavar="SECS",
        help="Per-run timeout in seconds (default: 120)",
    )
    run_p.add_argument(
        "--problems",
        nargs="+",
        metavar="NAME",
        help="Restrict to specific problem names",
    )
    run_p.add_argument(
        "--strategies",
        nargs="+",
        metavar="NAME",
        help="Restrict to specific strategy names",
    )
    run_p.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress per-run output"
    )
    run_p.set_defaults(func=cmd_run)

    # ---- score --------------------------------------------------------------
    score_p = sub.add_parser("score", help="Print score summary for a result file")
    score_p.add_argument("result_file", metavar="FILE", help="JSON result file")
    score_p.add_argument(
        "--json", action="store_true", help="Also print machine-readable JSON summary"
    )
    score_p.set_defaults(func=cmd_score)

    # ---- compare ------------------------------------------------------------
    cmp_p = sub.add_parser("compare", help="Compare two result files")
    cmp_p.add_argument(
        "before", metavar="BEFORE_FILE", help="Baseline JSON result file"
    )
    cmp_p.add_argument("after", metavar="AFTER_FILE", help="Candidate JSON result file")
    cmp_p.add_argument(
        "--eps",
        type=float,
        default=0.01,
        help="Min absolute score change considered significant (default: 0.01)",
    )
    cmp_p.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 2 if composite score decreased",
    )
    cmp_p.add_argument(
        "--json", action="store_true", help="Also print machine-readable JSON diff"
    )
    cmp_p.set_defaults(func=cmd_compare)

    # ---- list ---------------------------------------------------------------
    list_p = sub.add_parser("list", help="List available problems and strategies")
    _add_mode_group(list_p)
    list_p.set_defaults(func=cmd_list)

    return parser


def _add_mode_group(parser: argparse.ArgumentParser) -> None:
    """Add mutually exclusive --quick / --standard / --full / --mode flags."""
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--quick",
        dest="mode",
        action="store_const",
        const="quick",
        help="Quick mode: 3 problems, 2 strategies, 75 evals (default)",
    )
    grp.add_argument(
        "--standard",
        dest="mode",
        action="store_const",
        const="standard",
        help="Standard mode: 8 problems, 3 strategies, 200 evals",
    )
    grp.add_argument(
        "--full",
        dest="mode",
        action="store_const",
        const="full",
        help="Full mode: all problems and strategies, 500 evals",
    )
    grp.add_argument(
        "--mode",
        dest="mode",
        metavar="MODE",
        choices=["quick", "standard", "full"],
        help="Explicit mode selection",
    )
    parser.set_defaults(mode="quick")


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Override ``sys.argv[1:]`` (used in tests).

    Returns:
        Exit code (0 = success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
