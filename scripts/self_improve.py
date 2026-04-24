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
Self-Improvement Loop CLI
=========================

Thin command-line wrapper around :class:`panobbgo.self_improve.SelfImprover`
(Phase 5 of ``planning/SELF_IMPROVEMENT_LOOP.md``).

Two subcommands:

``run``
    Run the loop for ``N`` iterations.  Each iteration samples a
    hyperparameter mutation, runs the randomized harness for both the
    baseline and the candidate on the same sampled instances, and decides
    via the bootstrap-CI statistical rule from Phase 4.  Accepts are
    promoted for the next iteration; rejects keep the current specs.  Every
    iteration writes one JSONL line to the ledger::

        uv run python scripts/self_improve.py run --iterations 10

``summary``
    Pretty-print the ledger: totals, acceptance rate, best delta::

        uv run python scripts/self_improve.py summary

Stop the loop early by ``touch STOP_SELF_IMPROVE`` (configurable via
``--stop-sentinel``); the current iteration will finish, then the loop
exits and the ledger is preserved.

Exit codes
----------
- ``0`` — loop completed (or stopped via sentinel).
- ``1`` — argument error.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="self_improve.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    run_p = sub.add_parser("run", help="Run the self-improvement loop")
    run_p.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of loop iterations (default: 5)",
    )
    run_p.add_argument(
        "--mode",
        choices=["quick", "standard", "full"],
        default="quick",
        help="Harness mode per iteration (default: quick)",
    )
    run_p.add_argument("--reps", type=int, default=None, help="Override reps per (problem, strategy)")
    run_p.add_argument("--budget", type=int, default=None, help="Override evaluation budget per run")
    run_p.add_argument(
        "--base-seed",
        dest="base_seed",
        type=int,
        default=42,
        help="Base seed for the harness (default: 42)",
    )
    run_p.add_argument(
        "--mutation-seed",
        dest="mutation_seed",
        type=int,
        default=0,
        help="RNG seed for the mutation sampler (default: 0)",
    )
    run_p.add_argument(
        "--stat-seed",
        dest="stat_seed",
        type=int,
        default=42,
        help="Base RNG seed for the bootstrap (default: 42)",
    )
    run_p.add_argument(
        "--eps-accept",
        dest="eps_accept",
        type=float,
        default=0.005,
        help="Minimum composite delta to accept (default: 0.005)",
    )
    run_p.add_argument(
        "--eps-regress",
        dest="eps_regress",
        type=float,
        default=0.05,
        help="Maximum tolerated per-pair regression (default: 0.05)",
    )
    run_p.add_argument(
        "--n-boot",
        dest="n_boot",
        type=int,
        default=2000,
        help="Bootstrap resamples per iteration (default: 2000)",
    )
    run_p.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for the bootstrap CI (default: 0.95)",
    )
    run_p.add_argument(
        "--strategies",
        nargs="+",
        metavar="NAME",
        default=None,
        help="Restrict to these strategy names",
    )
    run_p.add_argument(
        "--ledger",
        dest="ledger",
        default="planning/self_improve_ledger.jsonl",
        help="Path to append-only JSONL ledger",
    )
    run_p.add_argument(
        "--stop-sentinel",
        dest="stop_sentinel",
        default="STOP_SELF_IMPROVE",
        help="Stop loop if this file exists (empty string disables)",
    )
    run_p.add_argument(
        "--no-randomize",
        dest="randomize",
        action="store_false",
        help="Disable parametric randomization (use fixed instances)",
    )
    run_p.set_defaults(randomize=True)
    run_p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        metavar="SECS",
        help="Per-run timeout in seconds (default: 120)",
    )
    run_p.add_argument("--quiet", "-q", action="store_true", help="Suppress per-iteration output")
    run_p.set_defaults(func=_cmd_run)

    sum_p = sub.add_parser("summary", help="Summarise a JSONL ledger file")
    sum_p.add_argument(
        "ledger",
        nargs="?",
        default="planning/self_improve_ledger.jsonl",
        help="Path to the ledger (default: planning/self_improve_ledger.jsonl)",
    )
    sum_p.set_defaults(func=_cmd_summary)

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    from panobbgo.self_improve import LoopConfig, SelfImprover

    cfg = LoopConfig(
        iterations=args.iterations,
        base_seed=args.base_seed,
        mode=args.mode,
        reps=args.reps,
        budget=args.budget,
        eps_accept=args.eps_accept,
        eps_regress=args.eps_regress,
        n_boot=args.n_boot,
        confidence=args.confidence,
        stat_seed=args.stat_seed,
        mutation_seed=args.mutation_seed,
        strategy_names=args.strategies,
        ledger_path=args.ledger,
        stop_sentinel_path=args.stop_sentinel,
        timeout_per_run=args.timeout,
        randomize=args.randomize,
    )
    records = SelfImprover(cfg).run(verbose=not args.quiet)

    n_accepts = sum(1 for r in records if r.accepted)
    n_skips = sum(1 for r in records if r.proposal is None)
    n_total = len(records)
    print()
    print(f"[self_improve] completed: {n_total} iter, {n_accepts} accept, {n_skips} skip, ledger={cfg.ledger_path}")
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    from panobbgo.self_improve import load_ledger

    path = pathlib.Path(args.ledger)
    if not path.exists():
        print(f"Error: ledger not found: {path}", file=sys.stderr)
        return 1

    records = load_ledger(str(path))
    if not records:
        print(f"(empty ledger: {path})")
        return 0

    n = len(records)
    accepted = [r for r in records if r.get("accepted")]
    skipped = [r for r in records if r.get("proposal") is None]
    decided = [r for r in records if r.get("proposal") is not None]
    accept_rate = (len(accepted) / len(decided)) if decided else 0.0
    best_delta = max((r.get("delta", 0.0) for r in decided), default=0.0)

    print(f"Ledger:        {path}")
    print(f"Iterations:    {n}  (decided={len(decided)}, skipped={len(skipped)})")
    print(f"Accepts:       {len(accepted)}  ({accept_rate:.1%} of decided)")
    if decided:
        print(f"Best Δ seen:   {best_delta:+.4f}")

    if accepted:
        print("Accepted changes:")
        for r in accepted:
            p = r.get("proposal") or {}
            print(
                f"  iter={r.get('iteration')}  Δ={r.get('delta'):+.4f}  "
                f"{p.get('strategy_name')}/{p.get('class_name')}."
                f"{p.get('param_name')}: "
                f"{p.get('old_value')!r} -> {p.get('new_value')!r}"
            )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
