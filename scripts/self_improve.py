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

``--adaptive``
    Enable Thompson-sampling adaptive mutation sampler (§10 of the
    plan).  The loop maintains a Beta posterior per rule and biases
    future samples toward rules with positive accept history.  Cold-start
    equivalent to uniform sampling.  Add ``--adaptive-prime-from-ledger``
    to seed history from a prior ledger when resuming a long run::

        uv run python scripts/self_improve.py run --iterations 50 \\
            --adaptive --adaptive-prime-from-ledger

``--structural``
    Use the structural mutation catalog (§7.2): the default
    hyperparameter rules **plus** ``add_heuristic`` /
    ``drop_heuristic`` ops over a curated heuristic pool
    (Random, Nearby, NelderMead, Center, LatinHypercube, Sobol,
    Extremal).  Drops respect a ``min_heuristics=2`` safety floor and
    adds skip classes already present in the strategy.  Off by default
    so existing CLI invocations are byte-identical::

        uv run python scripts/self_improve.py run --iterations 50 \\
            --structural --adaptive

``--holdout-base-seed``
    Enable end-of-loop hold-out validation against an independent
    ``base_seed`` family.  After the main loop finishes, the seed and
    final-top of the ladder are re-measured on instances drawn from
    the hold-out base seed; an overfit ladder (gap shrinks more than
    ``--holdout-eps-overfit`` on hold-out) is logged and, when
    ``--fail-on-overfit`` is set, exits the CLI with code ``3``::

        uv run python scripts/self_improve.py run --iterations 50 \\
            --holdout-base-seed 1234 --fail-on-overfit

``--holdout-base-seeds``
    Multi-seed variant of ``--holdout-base-seed``.  Pass a comma-separated
    list of integers; one :class:`LoopHoldoutRecord` is written per seed
    and the CLI prints a single aggregated verdict (``OVERFIT`` if any
    seed flagged overfit; worst drift across seeds).  Strictly more
    robust than the single-seed check at the cost of one extra
    ``2 × holdout_iterations`` runs per added seed::

        uv run python scripts/self_improve.py run --iterations 50 \\
            --holdout-base-seeds 1234,5678,9012 --fail-on-overfit

``--fail-on-overfit-ci``
    Stricter sibling of ``--fail-on-overfit``.  Computes a bootstrap
    confidence interval on the pooled per-iteration hold-out drift
    (across all configured hold-out seeds and iterations) and exits
    with code ``3`` iff the upper bound of the CI falls below
    ``-holdout-eps-overfit`` — i.e. the bootstrap *rules out* a drift
    better than the tolerance at the configured confidence level.
    Less reactive than ``--fail-on-overfit`` (single-seed noise can no
    longer trip it) and pairs naturally with
    :func:`panobbgo.harness.statistical_accept`-style decision rules
    used elsewhere in the loop::

        uv run python scripts/self_improve.py run --iterations 50 \\
            --holdout-base-seeds 1234,5678,9012 \\
            --fail-on-overfit-ci --holdout-ci-confidence 0.95

Stop the loop early by ``touch STOP_SELF_IMPROVE`` (configurable via
``--stop-sentinel``); the current iteration will finish, then the loop
exits and the ledger is preserved.

Exit codes
----------
- ``0`` — loop completed (or stopped via sentinel).
- ``1`` — argument error.
- ``3`` — ``--fail-on-overfit`` and the hold-out flagged overfit, or
  ``--fail-on-overfit-ci`` and the aggregated drift CI's upper bound
  fell below ``-holdout_eps_overfit``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple


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
        "--metric",
        choices=["composite", "aocc"],
        default="composite",
        help=(
            "Which metric drives accept/reject. 'composite' (default) "
            "uses Panobbgo's internal problem battery; 'aocc' uses the "
            "IOH/MA-BBOB anytime metric. With --metric=aocc, --mode "
            "selects the IOH battery preset (quick/standard/full)."
        ),
    )
    run_p.add_argument(
        "--registry",
        choices=["default", "loop"],
        default="default",
        help=(
            "Named strategy registry. 'default' (historical) selects the "
            "quick/standard/full battery from --mode; 'loop' selects the "
            "catalog-exercising loop registry (the quick specs plus one "
            "compact spec per rule-bearing family — DE / PSO / RegionUCB "
            "/ LBFGSB+COBYQA / Restart) so the dormant catalog mutation "
            "rules actually fire on the seed specs. See §9.1 of "
            "planning/SELF_IMPROVEMENT_LOOP.md (V2 plan)."
        ),
    )
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
    run_p.add_argument(
        "--guard-interval",
        dest="guard_interval",
        type=int,
        default=0,
        help=("Run the anti-cherry-pick guard every N iterations (0 = disabled, default; typical values are 5 or 10)"),
    )
    run_p.add_argument(
        "--guard-eps-ladder",
        dest="guard_eps_ladder",
        type=float,
        default=0.02,
        help="Tolerance for ladder drift detected by the guard (default: 0.02)",
    )
    run_p.add_argument(
        "--guard-iteration-offset",
        dest="guard_iteration_offset",
        type=int,
        default=1_000_000,
        help="Iteration-id offset for the guard's fresh seed (default: 1_000_000)",
    )
    run_p.add_argument(
        "--adaptive",
        dest="adaptive_sampling",
        action="store_true",
        help=(
            "Enable Thompson-sampling adaptive mutation sampler (§10): "
            "the loop biases future iterations toward rules with positive accept history. "
            "Cold-start equivalent to uniform sampling."
        ),
    )
    run_p.set_defaults(adaptive_sampling=False)
    run_p.add_argument(
        "--adaptive-prior-alpha",
        dest="adaptive_prior_alpha",
        type=float,
        default=1.0,
        help="Beta prior alpha for the adaptive sampler (default: 1.0)",
    )
    run_p.add_argument(
        "--adaptive-prior-beta",
        dest="adaptive_prior_beta",
        type=float,
        default=1.0,
        help="Beta prior beta for the adaptive sampler (default: 1.0)",
    )
    run_p.add_argument(
        "--adaptive-prime-from-ledger",
        dest="adaptive_prime_from_ledger",
        action="store_true",
        help="Seed adaptive sampler history from the existing ledger before running",
    )
    run_p.set_defaults(adaptive_prime_from_ledger=False)
    run_p.add_argument(
        "--structural-per-class-arms",
        dest="structural_per_class_arms",
        action="store_true",
        help=(
            "Split structural ops (add_heuristic / drop_heuristic) into "
            "per-target-class bandit arms in the adaptive sampler.  Lets "
            "the loop distinguish 'add Sobol' from 'add Random' at the "
            "cost of sparser per-arm data.  Only effective with --adaptive."
        ),
    )
    run_p.set_defaults(structural_per_class_arms=False)
    run_p.add_argument(
        "--bandit-reward",
        dest="bandit_reward_shaping",
        choices=["binary", "graded"],
        default="binary",
        help=(
            "Bandit reward shaping policy.  'binary' (default) — accept "
            "contributes +1, reject contributes 0 to the Beta posterior "
            "(the historical behaviour).  'graded' — implements §7.4 of "
            "planning/SELF_IMPROVEMENT_LOOP.md: accept contributes "
            "0.5 + clip(ci_low/(4·eps_accept), 0, 0.5) (between 0.5 and 1), "
            "reject contributes clip(0.5 + delta/(4·eps_accept), 0, 0.5) "
            "(between 0 and 0.5).  Converts every informative iteration "
            "into evidence on the chosen arm, distinguishing 'honest near "
            "miss' rejections from clearly-harmful rejections so "
            "small-positive arms become identifiable at realistic "
            "per-night iteration counts.  Only takes effect with --adaptive."
        ),
    )
    run_p.add_argument(
        "--structural-borrow-alpha",
        dest="structural_borrow_alpha",
        type=float,
        default=0.0,
        help=(
            "Hierarchical 'borrow' coefficient kappa >= 0 for per-class "
            "structural arms.  When > 0, each per-class arm's Beta "
            "posterior borrows kappa * (n_other_class_accepts, ...) from "
            "the op-level aggregate (sum across sibling per-class arms), "
            "so a fresh candidate class warms with the op's empirical "
            "accept rate instead of the symmetric Beta(1, 1) prior.  "
            "0.0 (default) keeps the pure per-class semantics.  Only "
            "effective with both --adaptive and --structural-per-class-arms."
        ),
    )
    run_p.add_argument(
        "--structural",
        dest="structural",
        action="store_true",
        help=(
            "Use the structural catalog (§7.2): kwarg perturbations + "
            "add_heuristic / drop_heuristic ops over a curated heuristic pool. "
            "Off by default to keep existing CLI invocations byte-identical."
        ),
    )
    run_p.set_defaults(structural=False)
    run_p.add_argument(
        "--holdout-base-seed",
        dest="holdout_base_seed",
        type=int,
        default=0,
        help=(
            "Independent base_seed for end-of-loop hold-out validation."
            " Default 0 disables hold-out.  Must differ from --base-seed."
            " Ignored when --holdout-base-seeds is set."
        ),
    )
    run_p.add_argument(
        "--holdout-base-seeds",
        dest="holdout_base_seeds",
        type=str,
        default="",
        metavar="S1,S2,...",
        help=(
            "Comma-separated list of independent base_seeds for multi-seed"
            " hold-out validation.  When set, supersedes --holdout-base-seed."
            " Reduction across seeds: worst (most negative) drift, any overfit."
            " Empty (default) disables multi-seed hold-out."
            " Each seed must be non-zero, distinct, and differ from --base-seed."
        ),
    )
    run_p.add_argument(
        "--holdout-iterations",
        dest="holdout_iterations",
        type=int,
        default=5,
        help="Number of distinct iteration_ids to average for the hold-out (default: 5)",
    )
    run_p.add_argument(
        "--holdout-iteration-offset",
        dest="holdout_iteration_offset",
        type=int,
        default=0,
        help="Starting iteration_id for the hold-out sweep (default: 0)",
    )
    run_p.add_argument(
        "--holdout-eps-overfit",
        dest="holdout_eps_overfit",
        type=float,
        default=0.05,
        help="Drift tolerance on top-vs-seed gap; below -eps flags overfit (default: 0.05)",
    )
    run_p.add_argument(
        "--fail-on-overfit",
        dest="fail_on_overfit",
        action="store_true",
        help="Exit with code 3 if the hold-out flags an overfit ladder",
    )
    run_p.set_defaults(fail_on_overfit=False)
    run_p.add_argument(
        "--fail-on-overfit-ci",
        dest="fail_on_overfit_ci",
        action="store_true",
        help=(
            "Exit with code 3 if the bootstrap CI on the aggregated"
            " hold-out drift is statistically significant for overfit"
            " (upper bound of the CI < -eps_overfit).  Stricter than"
            " --fail-on-overfit: ignores single-seed noise and only"
            " fires when the pooled per-iteration drift CI rules out"
            " no-drift at the configured confidence level."
        ),
    )
    run_p.set_defaults(fail_on_overfit_ci=False)
    run_p.add_argument(
        "--holdout-ci-confidence",
        dest="holdout_ci_confidence",
        type=float,
        default=0.95,
        help=(
            "Confidence level for the bootstrap CI on the aggregated"
            " hold-out drift (default: 0.95).  Only affects the"
            " printed aggregate line and --fail-on-overfit-ci; the"
            " per-record overfit flag is unchanged."
        ),
    )
    run_p.add_argument(
        "--holdout-ci-n-boot",
        dest="holdout_ci_n_boot",
        type=int,
        default=10_000,
        help="Bootstrap resamples for the hold-out drift CI (default: 10000)",
    )
    run_p.add_argument(
        "--inactivity-relax-after",
        dest="inactivity_relax_after",
        type=int,
        default=0,
        help=(
            "Relax eps_accept after N consecutive non-accept iterations to"
            " break out of long droughts (0 = disabled, default).  Each"
            " additional block of N non-accepts halves the threshold by"
            " --inactivity-relax-factor, floored at"
            " --inactivity-min-eps-accept.  Resets on the next accept."
            " Typical unattended value: 10."
        ),
    )
    run_p.add_argument(
        "--inactivity-relax-factor",
        dest="inactivity_relax_factor",
        type=float,
        default=0.5,
        help=(
            "Multiplicative factor applied to eps_accept per relaxation"
            " step (default: 0.5).  Must be in (0, 1).  Ignored when"
            " --inactivity-relax-after is 0."
        ),
    )
    run_p.add_argument(
        "--inactivity-min-eps-accept",
        dest="inactivity_min_eps_accept",
        type=float,
        default=0.001,
        help=(
            "Floor on the relaxed eps_accept (default: 0.001 — matches"
            " the bootstrap CI noise floor at typical quick-mode rep"
            " counts).  Must be <= --eps-accept.  Ignored when"
            " --inactivity-relax-after is 0."
        ),
    )
    paired_grp = run_p.add_mutually_exclusive_group()
    paired_grp.add_argument(
        "--paired",
        dest="paired",
        action="store_const",
        const=True,
        help=(
            "Force paired bootstrap in statistical_accept (default: auto"
            " — paired when reps are instance-aligned, which is the case"
            " under the randomized harness)."
        ),
    )
    paired_grp.add_argument(
        "--unpaired",
        dest="paired",
        action="store_const",
        const=False,
        help="Force the historical unpaired (independent) bootstrap.",
    )
    run_p.set_defaults(paired=None)
    run_p.add_argument("--quiet", "-q", action="store_true", help="Suppress per-iteration output")
    run_p.set_defaults(func=_cmd_run)

    sum_p = sub.add_parser("summary", help="Summarise a JSONL ledger file")
    sum_p.add_argument(
        "ledger",
        nargs="?",
        default="planning/self_improve_ledger.jsonl",
        help="Path to the ledger (default: planning/self_improve_ledger.jsonl)",
    )
    sum_p.add_argument(
        "--top-n",
        type=int,
        default=10,
        help=(
            "Number of top-ranked mutation-rule posteriors to surface in "
            "the bandit trend block (default: 10).  Ranking is by mean "
            "graded reward (binary path: == accept rate); ties broken by "
            "n_attempts."
        ),
    )
    sum_p.add_argument(
        "--bottom-n",
        type=int,
        default=5,
        help=(
            "Number of bottom-ranked mutation-rule posteriors to surface "
            "alongside the top-N (default: 5).  Use 0 to hide the bottom "
            "list entirely."
        ),
    )
    sum_p.add_argument(
        "--min-attempts",
        type=int,
        default=3,
        help=(
            "Hide rules with fewer than N informative attempts (no-op "
            "iterations excluded) from the top / bottom lists (default: "
            "3).  Prevents one-shot rules from dominating the leaderboard."
        ),
    )
    sum_p.set_defaults(func=_cmd_summary)

    return parser


def _parse_seed_list(raw: str) -> tuple:
    """Parse a comma-separated seed list (e.g. ``"1234,5678,9012"``).

    Empty / blank → empty tuple.  Whitespace around entries is tolerated
    so command-line callers can write ``"1234, 5678"`` without quoting
    surprises.  Non-integer entries raise ``ValueError`` with the
    offending token for ergonomic error messages.
    """
    s = (raw or "").strip()
    if not s:
        return ()
    out = []
    for piece in s.split(","):
        token = piece.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError as e:
            raise ValueError(f"--holdout-base-seeds: invalid integer {token!r}") from e
    return tuple(out)


def _cmd_run(args: argparse.Namespace) -> int:
    from panobbgo.self_improve import LoopConfig, SelfImprover, default_catalog, default_structural_catalog

    catalog = default_structural_catalog() if args.structural else default_catalog()
    try:
        holdout_seeds = _parse_seed_list(args.holdout_base_seeds)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    try:
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
            guard_interval=args.guard_interval,
            guard_eps_ladder=args.guard_eps_ladder,
            guard_iteration_offset=args.guard_iteration_offset,
            adaptive_sampling=args.adaptive_sampling,
            adaptive_prior_alpha=args.adaptive_prior_alpha,
            adaptive_prior_beta=args.adaptive_prior_beta,
            adaptive_prime_from_ledger=args.adaptive_prime_from_ledger,
            structural_per_class_arms=args.structural_per_class_arms,
            structural_borrow_alpha=args.structural_borrow_alpha,
            holdout_base_seed=args.holdout_base_seed,
            holdout_base_seeds=holdout_seeds,
            holdout_iterations=args.holdout_iterations,
            holdout_iteration_offset=args.holdout_iteration_offset,
            holdout_eps_overfit=args.holdout_eps_overfit,
            paired=args.paired,
            metric=args.metric,
            registry=args.registry,
            inactivity_relax_after=args.inactivity_relax_after,
            inactivity_relax_factor=args.inactivity_relax_factor,
            inactivity_min_eps_accept=args.inactivity_min_eps_accept,
            bandit_reward_shaping=args.bandit_reward_shaping,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    improver = SelfImprover(cfg, catalog=catalog)
    records, _, holdout_records = improver.run_full(verbose=not args.quiet)

    n_accepts = sum(1 for r in records if r.accepted)
    n_skips = sum(1 for r in records if r.proposal is None)
    # §12.4 no-op detection (post-measure): proposals whose candidate
    # per-pair scores were bit-identical to baseline carry zero
    # information and are not pulled on the bandit.  Surface the count
    # so an operator can see at a glance whether the loop is starving
    # itself on dormant rules.
    n_no_op = sum(1 for r in records if r.no_op)
    n_total = len(records)
    print()
    print(
        f"[self_improve] completed: {n_total} iter, {n_accepts} accept, "
        f"{n_skips} skip, {n_no_op} no-op, ledger={cfg.ledger_path}"
    )
    if improver.sampler is not None:
        snap = improver.sampler.stats_snapshot()
        if snap:
            print("[self_improve] adaptive sampler stats (class.param[kind] -> n_accepts/n_attempts):")
            for s in snap:
                cls, param, kind = s.rule_key
                print(f"  {cls}.{param}[{kind}] -> {s.n_accepts}/{s.n_attempts} ({s.accept_rate:.0%})")
    if holdout_records:
        from panobbgo.self_improve import aggregate_holdout_drift

        any_overfit = any(r.overfit for r in holdout_records)
        # Worst-case generalisation: the most negative drift across seeds
        # is the one a reviewer cares about — a positive aggregate hides
        # a single bad seed.  Match the planning doc's `min` reduction.
        worst = min(holdout_records, key=lambda r: r.drift)
        # Vacuous hold-outs (ladder kept only the seed entry — no
        # accepted mutations) historically rendered as ``OK
        # drift=+0.0000`` even though no generalisation signal exists.
        # V2 §6.4 / §12.4 of `planning/SELF_IMPROVEMENT_LOOP.md` demands
        # they surface as VACUOUS instead; use
        # :meth:`LoopHoldoutRecord.effective_status` so legacy ledger
        # records (no explicit status field) still classify correctly.
        n_vacuous = sum(1 for r in holdout_records if r.effective_status() == "vacuous")
        all_vacuous = n_vacuous == len(holdout_records)
        if len(holdout_records) == 1:
            ho = holdout_records[0]
            verdict = ho.effective_status().upper()
            print(
                f"[self_improve] hold-out: {verdict}  drift={ho.drift:+.4f}  "
                f"holdout_gap={ho.holdout_delta:+.4f}  training_gap={ho.training_delta:+.4f}  "
                f"(base_seed={ho.holdout_base_seed}, n={ho.holdout_iterations})"
            )
        else:
            if all_vacuous:
                verdict = "VACUOUS"
            elif any_overfit:
                verdict = "OVERFIT"
            else:
                verdict = "OK"
            seeds = ",".join(str(r.holdout_base_seed) for r in holdout_records)
            n_overfit = sum(1 for r in holdout_records if r.overfit)
            print(
                f"[self_improve] hold-out aggregate: {verdict}  worst_drift={worst.drift:+.4f}  "
                f"overfit={n_overfit}/{len(holdout_records)}  vacuous={n_vacuous}/{len(holdout_records)}  "
                f"worst_seed={worst.holdout_base_seed}  "
                f"(seeds=[{seeds}], n={worst.holdout_iterations})"
            )

        # Bootstrap CI on the pooled per-iteration drift across all
        # hold-out records.  Stable across single-vs-multi-seed: with a
        # single seed it is a CI on the per-iteration drifts inside
        # that one record (still a real distribution because each
        # holdout_iterations is paired); with multiple seeds it pools
        # across (seed, iter).  See aggregate_holdout_drift docstring
        # for the per-iteration vs per-record fallback.
        agg = aggregate_holdout_drift(
            holdout_records,
            n_boot=int(args.holdout_ci_n_boot),
            confidence=float(args.holdout_ci_confidence),
            seed=int(args.stat_seed),
        )
        # Match the single-record verdict semantics: when every
        # contributing record was vacuous the CI carries no signal and
        # must not be reported as ``OK_CI`` — V2 §6.4 / §12.4.
        if agg.all_vacuous:
            ci_verdict = "VACUOUS_CI"
        elif agg.statistically_overfit:
            ci_verdict = "OVERFIT_CI"
        else:
            ci_verdict = "OK_CI"
        print(
            f"[self_improve] hold-out drift CI: {ci_verdict}  "
            f"mean={agg.mean_drift:+.4f}  "
            f"CI{int(agg.confidence * 100)}%=[{agg.ci_low:+.4f}, {agg.ci_high:+.4f}]  "
            f"n_samples={agg.n_samples}  n_records={agg.n_records}  "
            f"vacuous={agg.vacuous_count}/{agg.n_records}"
        )
        if any_overfit and args.fail_on_overfit:
            return 3
        if agg.statistically_overfit and args.fail_on_overfit_ci:
            return 3
    return 0


def _group_runs(iter_records: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Partition iteration records into per-run buckets.

    The ledger is append-only and concatenates the iteration records of
    every nightly run end-to-end.  Each call to
    :meth:`panobbgo.self_improve.SelfImprover.run` restarts the iteration
    counter at ``0``, so a new run begins wherever the current record's
    ``iteration`` is **less than or equal to** the previous one's (the
    common case is ``0`` after the previous run finished at ``N-1``;
    pathological ``--start-iteration`` overrides still trigger correctly).
    The very first record starts the first run.

    Returns a list of runs in ledger order; each run is a list of
    iteration records in the order they appeared.  Empty input → empty
    list.  Records without an ``iteration`` field default to ``0`` so a
    legacy / partial record never silently joins an unrelated run.
    """
    runs: List[List[Dict[str, Any]]] = []
    prev_iter = None
    for rec in iter_records:
        cur = int(rec.get("iteration", 0))
        if prev_iter is None or cur <= prev_iter:
            runs.append([])
        runs[-1].append(rec)
        prev_iter = cur
    return runs


def _print_trend_block(iter_records: List[Dict[str, Any]]) -> None:
    """Per-run trend table — V2 §12.4 "Summary trend block".

    One row per nightly run with: date, base_seed, mode, iter count,
    decided (non-skip) count, accepts, no-op count, best Δ seen.  This is
    the at-a-glance signal the §12.3 daily routine reads: "did last night
    accept anything? was the no-op rate sane? is the per-night seed
    score holding?".
    """
    runs = _group_runs(iter_records)
    if not runs:
        return
    print()
    print("Trend (one row per loop run, oldest first):")
    print(
        f"  {'date':<19}  {'seed':>5}  {'mode':<8}  {'iters':>5}  "
        f"{'dec':>4}  {'acc':>4}  {'nop':>4}  {'best_Δ':>8}  {'seed_score':>10}"
    )
    for run in runs:
        first = run[0]
        ts = str(first.get("timestamp", ""))[:19].replace("T", " ")
        base_seed = first.get("base_seed", "?")
        mode = str(first.get("mode", "?"))[:8]
        n_iters = len(run)
        decided = [r for r in run if r.get("proposal") is not None]
        n_decided = len(decided)
        n_accepts = sum(1 for r in run if r.get("accepted"))
        n_no_op = sum(1 for r in decided if r.get("no_op"))
        best = max((float(r.get("delta", 0.0)) for r in decided), default=0.0)
        # Seed score for this run = baseline_score on the first decided
        # iteration (the seed-spec measurement, before any accept can have
        # taken effect).  Skip records carry the same baseline; the
        # critical thing is to source it from a real measurement so the
        # column tracks per-night signal, not a recomputed average.
        seed_score = float(first.get("baseline_score", 0.0))
        print(
            f"  {ts:<19}  {base_seed!s:>5}  {mode:<8}  {n_iters:>5}  "
            f"{n_decided:>4}  {n_accepts:>4}  {n_no_op:>4}  {best:>+8.4f}  {seed_score:>10.4f}"
        )


def _replay_bandit_posteriors(
    iter_records: List[Dict[str, Any]],
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Reconstruct per-rule bandit stats by replaying iteration records.

    Mirrors :meth:`panobbgo.self_improve.AdaptiveMutationSampler.prime_from_ledger`
    on the default key layout (``per_class_structural=False``) so the
    summary's posterior view matches what a freshly-primed nightly bandit
    would carry into the next run.  No-op iterations and skip / guard /
    hold-out records are excluded — exactly the same filter applied to
    live bandit pulls per V2 §12.4.

    Returns a dict keyed on the rule's ``(class_name, param_name,
    rule_kind)`` tuple — or the structural collapse ``("*", op,
    "structural")`` for ``add_/drop_`` ops — with the cumulative
    ``n_attempts``, ``n_accepts``, ``reward_sum`` and the resulting
    ``mean_reward`` / ``accept_rate``.  Legacy records (no
    ``bandit_reward``) fall back to the binary ``1.0`` per accept /
    ``0.0`` per reject, matching :meth:`prime_from_ledger`.
    """
    from panobbgo.self_improve import _proposal_rule_key

    stats: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for rec in iter_records:
        if rec.get("record_type", "iteration") != "iteration":
            continue
        proposal = rec.get("proposal")
        if proposal is None:
            continue
        if rec.get("no_op"):
            continue
        key = _proposal_rule_key(
            str(proposal.get("class_name", "")),
            str(proposal.get("param_name", "")),
            str(proposal.get("rule_kind", "")),
            per_class_structural=False,
        )
        bucket = stats.setdefault(
            key,
            {"n_attempts": 0, "n_accepts": 0, "reward_sum": 0.0},
        )
        accepted = bool(rec.get("accepted"))
        bucket["n_attempts"] += 1
        if accepted:
            bucket["n_accepts"] += 1
        reward = rec.get("bandit_reward")
        if reward is None:
            graded = 1.0 if accepted else 0.0
        else:
            graded = float(reward)
            if graded < 0.0:
                graded = 0.0
            elif graded > 1.0:
                graded = 1.0
        bucket["reward_sum"] += graded
    # Derive convenience fields for sorting / rendering.
    for bucket in stats.values():
        attempts = bucket["n_attempts"]
        bucket["mean_reward"] = (bucket["reward_sum"] / attempts) if attempts else 0.0
        bucket["accept_rate"] = (bucket["n_accepts"] / attempts) if attempts else 0.0
    return stats


def _print_bandit_block(
    iter_records: List[Dict[str, Any]],
    top_n: int,
    bottom_n: int,
    min_attempts: int,
) -> None:
    """Top-N / bottom-N mutation-rule posteriors — V2 §12.4 trend block.

    Rank rules by graded ``mean_reward`` so the §7.4 reward shaping
    (barely-confirmed accepts at ``~0.5``, honest near-miss rejects at
    ``~0.5``, clearly-harmful rejects at ``~0``) shows through.  On
    legacy binary-reward ledgers the rank collapses to ``accept_rate``
    so pre-2026-06-13 evidence is rendered without distortion.

    Filters by ``min_attempts`` so one-shot rules cannot dominate the
    leaderboard; ties are broken by ``n_attempts`` so a high-mean rule
    with sparse data does not edge out a slightly-lower-mean rule with
    much more evidence.
    """
    stats = _replay_bandit_posteriors(iter_records)
    if not stats:
        return
    eligible = [(k, v) for k, v in stats.items() if v["n_attempts"] >= min_attempts]
    if not eligible:
        print()
        print(f"Bandit posteriors: (no rules with >= {min_attempts} informative attempts)")
        return
    # Sort by mean reward descending (tie-break by n_attempts so denser
    # evidence beats sparse evidence at the same mean).
    eligible.sort(key=lambda kv: (kv[1]["mean_reward"], kv[1]["n_attempts"]), reverse=True)

    def _render(label: str, items: List[Tuple[Tuple[str, str, str], Dict[str, Any]]]) -> None:
        if not items:
            return
        print(f"{label}:")
        print(f"  {'class':<22}  {'param':<22}  {'kind':<18}  {'att':>4}  {'acc':>4}  {'mean_r':>7}  {'acc_rate':>8}")
        for key, bucket in items:
            cls, param, kind = key
            print(
                f"  {cls[:22]:<22}  {param[:22]:<22}  {kind[:18]:<18}  "
                f"{bucket['n_attempts']:>4}  {bucket['n_accepts']:>4}  "
                f"{bucket['mean_reward']:>7.3f}  {bucket['accept_rate']:>8.1%}"
            )

    top_count = max(0, top_n)
    bottom_count = max(0, bottom_n)
    print()
    print(f"Bandit posteriors (n_attempts >= {min_attempts}, {len(eligible)} eligible rules):")
    if top_count:
        _render(f"  Top {min(top_count, len(eligible))} (highest mean reward)", eligible[:top_count])
    if bottom_count and len(eligible) > top_count:
        # Reverse the bottom slice so the worst rule prints last — easier
        # for an operator to scan the "should I deprioritize this rule?"
        # block from top to bottom.
        worst = sorted(eligible[-bottom_count:], key=lambda kv: kv[1]["mean_reward"])
        _render(f"  Bottom {len(worst)} (lowest mean reward)", worst)


def _print_inactivity_block(iter_records: List[Dict[str, Any]]) -> None:
    """Inactivity-relax telemetry — backlog "Inactivity-relax telemetry".

    Surfaces the longest accept drought, the count of accepts that fired
    on a *relaxed* threshold (``effective_eps_accept < eps_accept_base``),
    and the mean decay factor at those accepts.  The base
    ``eps_accept`` is inferred from the maximum observed
    ``effective_eps_accept`` across the ledger — relaxation only
    *decreases* the threshold (it is re-tightened back to the base on
    every accept), so the maximum is the configured base.

    Silently no-ops on ledgers whose iteration records carry no
    ``effective_eps_accept`` / ``iters_since_accept`` fields (pre
    2026-05-30 ledgers).  Operators reading the §12.3 daily routine see
    nothing — preserving the existing summary semantics — until at
    least one record exposes the relax telemetry.
    """
    relax_records = [
        r for r in iter_records if r.get("effective_eps_accept") is not None or r.get("iters_since_accept") is not None
    ]
    if not relax_records:
        return
    effective_values = [
        float(r["effective_eps_accept"]) for r in relax_records if r.get("effective_eps_accept") is not None
    ]
    if not effective_values:
        return
    eps_base = max(effective_values)
    streaks = [int(r["iters_since_accept"]) for r in relax_records if r.get("iters_since_accept") is not None]
    longest_drought = max(streaks) if streaks else 0
    accepts = [r for r in relax_records if r.get("accepted")]
    relaxed_accepts = [
        r
        for r in accepts
        if r.get("effective_eps_accept") is not None and float(r["effective_eps_accept"]) + 1e-12 < eps_base
    ]
    if accepts:
        decays = [float(r["effective_eps_accept"]) / eps_base for r in relaxed_accepts]
        mean_decay = (sum(decays) / len(decays)) if decays else 1.0
    else:
        mean_decay = 1.0
    print()
    print("Inactivity:")
    print(f"  eps_accept_base={eps_base:.4f}  longest_drought={longest_drought} iters")
    print(
        f"  relaxed_accepts={len(relaxed_accepts)}/{len(accepts)}"
        + (f"  mean_decay_at_accept={mean_decay:.3f}" if relaxed_accepts else "")
    )


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

    iter_records = [r for r in records if r.get("record_type", "iteration") == "iteration"]
    guard_records = [r for r in records if r.get("record_type") == "guard"]
    holdout_records = [r for r in records if r.get("record_type") == "holdout"]
    n = len(iter_records)
    accepted = [r for r in iter_records if r.get("accepted")]
    skipped = [r for r in iter_records if r.get("proposal") is None]
    decided = [r for r in iter_records if r.get("proposal") is not None]
    # §12.4 no-op telemetry: proposals whose candidate per-pair scores
    # were bit-identical to baseline carry zero information and are
    # excluded from the accept rate denominator.  Legacy records (pre
    # 2026-06-12) have no ``no_op`` key and default to False here.
    no_op = [r for r in decided if r.get("no_op")]
    informative = [r for r in decided if not r.get("no_op")]
    accept_rate = (len(accepted) / len(informative)) if informative else 0.0
    best_delta = max((r.get("delta", 0.0) for r in decided), default=0.0)
    rolled_back = [r for r in guard_records if r.get("rolled_back")]
    overfits = [r for r in holdout_records if r.get("overfit")]
    # Vacuous hold-outs (V2 §6.4 / §12.4): records whose ladder kept
    # only the seed entry — historically reported as ``OK drift=+0.0000``
    # because no mutation was accepted to validate.  Compute the count
    # from the legacy-aware predicate ``top_iteration < 0 and
    # ladder_size <= 1`` so summaries of pre-2026-06-11 ledgers
    # (no ``status`` field) classify correctly too.
    vacuous_holdouts = [
        r
        for r in holdout_records
        if r.get("status") == "vacuous"
        or (
            r.get("status") in (None, "ok")
            and int(r.get("top_iteration", -1)) < 0
            and int(r.get("ladder_size", 0)) <= 1
        )
    ]

    print(f"Ledger:        {path}")
    print(f"Iterations:    {n}  (decided={len(decided)}, skipped={len(skipped)}, no-op={len(no_op)})")
    print(f"Accepts:       {len(accepted)}  ({accept_rate:.1%} of informative)")
    if guard_records:
        total_pops = sum(int(r.get("pops", 0)) for r in guard_records)
        print(f"Guards:        {len(guard_records)}  (rollbacks={len(rolled_back)}, total pops={total_pops})")
    if holdout_records:
        print(f"Hold-outs:     {len(holdout_records)}  (overfit={len(overfits)}, vacuous={len(vacuous_holdouts)})")
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

    if rolled_back:
        print("Guard rollbacks:")
        for r in rolled_back:
            print(
                f"  iter={r.get('iteration')}  pops={r.get('pops')}  "
                f"score={r.get('guard_score'):.4f} vs stored "
                f"{r.get('pre_guard_top_score'):.4f}; "
                f"new top iter={r.get('rolled_back_to_iteration')}"
            )
    if holdout_records:
        # Multi-seed hold-out lands as multiple records back-to-back per
        # loop run (one per seed).  Reduce to the worst (most negative)
        # drift so the summary surfaces the failure mode a single-seed
        # check would have missed.
        print("Hold-out validation:")

        def _legacy_aware_status(rec: Dict[str, Any]) -> str:
            """Pre-2026-06-11 records have no ``status`` field — derive it."""
            explicit = rec.get("status")
            if explicit in ("ok", "overfit", "vacuous"):
                return explicit
            if int(rec.get("top_iteration", -1)) < 0 and int(rec.get("ladder_size", 0)) <= 1:
                return "vacuous"
            if rec.get("overfit"):
                return "overfit"
            return "ok"

        for r in holdout_records:
            verdict = _legacy_aware_status(r).upper()
            print(
                f"  {verdict}  drift={r.get('drift'):+.4f}  "
                f"holdout_gap={r.get('holdout_delta'):+.4f} "
                f"training_gap={r.get('training_delta'):+.4f}  "
                f"top_iter={r.get('top_iteration')}  "
                f"(base_seed={r.get('holdout_base_seed')}, n={r.get('holdout_iterations')})"
            )
        if len(holdout_records) > 1:
            n_vacuous = sum(1 for r in holdout_records if _legacy_aware_status(r) == "vacuous")
            all_vacuous = n_vacuous == len(holdout_records)
            worst = min(holdout_records, key=lambda r: float(r.get("drift", 0.0)))
            n_overfit = sum(1 for r in holdout_records if r.get("overfit"))
            if all_vacuous:
                agg = "VACUOUS"
            elif n_overfit:
                agg = "OVERFIT"
            else:
                agg = "OK"
            print(
                f"  --> aggregate: {agg}  worst_drift={float(worst.get('drift', 0.0)):+.4f}  "
                f"overfit={n_overfit}/{len(holdout_records)}  "
                f"vacuous={n_vacuous}/{len(holdout_records)}  "
                f"worst_seed={worst.get('holdout_base_seed')}"
            )
        # Bootstrap-CI aggregation across all hold-out records.  Rebuilds
        # :class:`LoopHoldoutRecord` instances from the JSONL payload so
        # the same helper used at loop end is reused unchanged; legacy
        # records (missing the per-iteration score lists) fall back to
        # one-sample-per-record automatically inside the helper.
        from panobbgo.self_improve import LoopHoldoutRecord, aggregate_holdout_drift

        rebuilt = [
            LoopHoldoutRecord(
                timestamp=str(r.get("timestamp", "")),
                duration_seconds=float(r.get("duration_seconds", 0.0)),
                holdout_base_seed=int(r.get("holdout_base_seed", 0)),
                holdout_iterations=int(r.get("holdout_iterations", 0)),
                holdout_iteration_offset=int(r.get("holdout_iteration_offset", 0)),
                seed_holdout_score=float(r.get("seed_holdout_score", 0.0)),
                top_holdout_score=float(r.get("top_holdout_score", 0.0)),
                seed_training_score=float(r.get("seed_training_score", 0.0)),
                top_training_score=float(r.get("top_training_score", 0.0)),
                holdout_delta=float(r.get("holdout_delta", 0.0)),
                training_delta=float(r.get("training_delta", 0.0)),
                drift=float(r.get("drift", 0.0)),
                overfit=bool(r.get("overfit", False)),
                eps_overfit=float(r.get("eps_overfit", 0.05)),
                top_iteration=int(r.get("top_iteration", -1)),
                ladder_size=int(r.get("ladder_size", 0)),
                base_seed=int(r.get("base_seed", 42)),
                mode=str(r.get("mode", "quick")),
                reasons=list(r.get("reasons", [])),
                seed_iteration_scores=[float(x) for x in r.get("seed_iteration_scores", [])],
                top_iteration_scores=[float(x) for x in r.get("top_iteration_scores", [])],
                # Preserve the explicit status when the ledger carries
                # one (post-2026-06-11 records); legacy records default
                # to ``"ok"`` and downstream code falls back to
                # :meth:`effective_status` for vacuous detection.
                status=str(r.get("status", "ok")),
            )
            for r in holdout_records
        ]
        agg = aggregate_holdout_drift(rebuilt)
        # Match _cmd_run: surface VACUOUS_CI when every contributing
        # record was vacuous so the summary cannot mistake "no signal"
        # for "no drift" — V2 §6.4 / §12.4.
        if agg.all_vacuous:
            ci_verdict = "VACUOUS_CI"
        elif agg.statistically_overfit:
            ci_verdict = "OVERFIT_CI"
        else:
            ci_verdict = "OK_CI"
        print(
            f"  --> drift CI: {ci_verdict}  mean={agg.mean_drift:+.4f}  "
            f"CI{int(agg.confidence * 100)}%=[{agg.ci_low:+.4f}, {agg.ci_high:+.4f}]  "
            f"n_samples={agg.n_samples}  n_records={agg.n_records}  "
            f"vacuous={agg.vacuous_count}/{agg.n_records}"
        )

    # V2 §12.4 trend block + backlog "Inactivity-relax telemetry in the
    # summary view".  Three additive sub-blocks rendered after the
    # existing per-record sections so the at-a-glance daily-routine
    # signal is visible without scrolling past the legacy detail.
    _print_trend_block(iter_records)
    _print_bandit_block(
        iter_records,
        top_n=int(getattr(args, "top_n", 10)),
        bottom_n=int(getattr(args, "bottom_n", 5)),
        min_attempts=int(getattr(args, "min_attempts", 3)),
    )
    _print_inactivity_block(iter_records)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
