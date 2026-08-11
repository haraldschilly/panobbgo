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

``codify-scan``
    Scan the live ledger + ``planning/done/`` archives for accepted
    mutations that fire directionally on at least ``--min-nights``
    distinct dates (default ``2``).  Surfaces the candidate set the
    daily routine should consider codifying into seed defaults
    (V2 §9.3 / §9.5 step 4 of the plan)::

        uv run python scripts/self_improve.py codify-scan
        # JSON output for an external tool / dashboard:
        uv run python scripts/self_improve.py codify-scan --json
        # Strict mode: only ``confirmed=True`` records (post V2 §6.4 ship):
        uv run python scripts/self_improve.py codify-scan --confirmed-only
        # Audit mode: include candidates whose implied edit is already
        # live in the seed-spec factories (default suppresses them):
        uv run python scripts/self_improve.py codify-scan --include-already-codified
        # Apply the top actionable kwarg candidate to panobbgo/harness.py
        # in place (shipped 2026-06-30 — V2 §9.5 step 4 plumbing).  Preview
        # with --apply-dry-run, then run pytest + commit + open a draft PR
        # manually.  The driver skips structural and bidirectional
        # candidates by default; override the bidirectional skip with
        # --apply-include-bidirectional (rare).
        uv run python scripts/self_improve.py codify-scan --apply-top --apply-dry-run
        uv run python scripts/self_improve.py codify-scan --apply-top
        # Open a draft PR for the top actionable candidate (implies
        # --apply-top).  Dedups against ``gh pr list --state open`` using
        # the codify-slot marker embedded in every codify PR body.  Pair
        # with --apply-dry-run to preview the git / gh command sequence
        # without side effects.  Requires the ``gh`` CLI on PATH.
        uv run python scripts/self_improve.py codify-scan --open-pr --apply-dry-run
        uv run python scripts/self_improve.py codify-scan --open-pr

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
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
        "--sync-eval",
        dest="sync_eval",
        action="store_true",
        help=(
            "Measure with the synchronous-harvest evaluation mode "
            "(config.sync_evaluation).  Cuts the AOCC quick-battery "
            "single-measurement noise sd from ~0.0101 to ~0.0063 (1.6x) "
            "by removing scheduling nondeterminism, at some wall-clock "
            "cost.  Inert under --metric composite.  Recorded per "
            "iteration in the ledger as 'sync_eval' — never pool "
            "cross-night evidence across the boundary, the two modes "
            "have different noise floors."
        ),
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
        default=None,
        help=(
            "Path to append-only JSONL ledger.  When omitted, defaults to "
            "the metric-specific ledger for --metric (composite → "
            "planning/self_improve_ledger.jsonl; aocc → "
            "planning/self_improve_ledger_aocc.jsonl) so the two scales "
            "never share a ledger (§12.1)."
        ),
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
        "--extra-highdim",
        dest="extra_highdim",
        action="store_true",
        help=(
            "Append the opt-in rotated higher-dimensional families "
            "(Rosenbrock_HighDim, dim_choices=(2, 5)) to the randomized "
            "battery so the loop / guard / hold-out measure a regime the "
            "frozen 2-D default battery cannot reach.  Composite-metric "
            "path only (inert on --metric aocc, whose battery lives in "
            "panobbgo.harness_ioh)."
        ),
    )
    run_p.set_defaults(extra_highdim=False)
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
        "--prime-include-archives",
        dest="adaptive_prime_include_archives",
        action="store_true",
        help=(
            "Also seed the adaptive sampler from archived ledgers in "
            "<dirname(ledger_path)>/done/ matching self_improve_ledger_*.jsonl "
            "before the live ledger.  Closes the V2 §2.6 'archives in "
            "planning/done/ are invisible' diagnosis so the bandit posterior "
            "accumulates evidence across every retained nightly run.  Only "
            "effective with --adaptive --adaptive-prime-from-ledger.  Silent "
            "no-op when the archive directory is missing or empty."
        ),
    )
    run_p.set_defaults(adaptive_prime_include_archives=False)
    run_p.add_argument(
        "--prime-archive-dir",
        dest="adaptive_prime_archive_dir",
        type=str,
        default=None,
        help=(
            "Override the archive directory scanned when "
            "--prime-include-archives is set (default: "
            "<dirname(ledger_path)>/done)."
        ),
    )
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
        "--confirm-accepts",
        dest="confirm_accepts",
        action="store_true",
        help=(
            "Enable the §6.4 same-night confirmation gate.  Every "
            "screening-accepted candidate is re-measured on a fresh "
            "randomize_iteration (and, when --holdout-base-seed / "
            "--holdout-base-seeds is set, on the *first* hold-out "
            "base_seed too); promotion happens only when the pooled "
            "(screen + confirm) bootstrap CI still clears eps_accept.  "
            "Failed confirmations are recorded as LoopConfirmRecord "
            "(record_type='confirm_reject') in the ledger and count as "
            "bandit reward 0 (binary mode) or graded-rejection reward "
            "from the pooled delta (graded mode).  Off by default to "
            "keep existing CLI invocations byte-identical."
        ),
    )
    run_p.set_defaults(confirm_accepts=False)
    run_p.add_argument(
        "--confirm-iteration-offset",
        dest="confirm_iteration_offset",
        type=int,
        default=500_000,
        help=(
            "randomize_iteration offset used by the §6.4 confirmation "
            "gate (default: 500_000).  Sits between the regular "
            "iteration stream (0..N) and the guard's offset (1_000_000) "
            "so the three streams never collide at realistic iteration "
            "counts.  Inert when --confirm-accepts is not set."
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
        "--structural-borrow-horizon",
        dest="structural_borrow_horizon",
        type=float,
        default=0.0,
        help=(
            "Auto-tune horizon h >= 0 for the hierarchical borrow "
            "coefficient kappa.  When > 0, the effective borrow per "
            "per-class arm is kappa / (1 + n_class_attempts / h) — full "
            "kappa at a cold arm, halved at h attempts, vanishing as "
            "evidence accumulates.  Borrow heavily early, trust the "
            "leaf signal once the arm has data.  0.0 (default) disables "
            "annealing (every arm always borrows the full kappa).  Inert "
            "when --structural-borrow-alpha=0 or when "
            "--structural-per-class-arms is off."
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
        default=None,
        help=(
            "Path to the ledger.  When omitted, defaults to the "
            "metric-specific ledger for --metric (composite → "
            "planning/self_improve_ledger.jsonl; aocc → "
            "planning/self_improve_ledger_aocc.jsonl)."
        ),
    )
    sum_p.add_argument(
        "--metric",
        choices=["composite", "aocc"],
        default="composite",
        help=(
            "Resolve the default ledger path for this metric when no ledger "
            "argument is given (default: composite, preserving the historical "
            "path).  Under the AOCC nightly default (2026-07-09) pass "
            "--metric aocc to point the daily routine at the active ledger."
        ),
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

    scan_p = sub.add_parser(
        "codify-scan",
        help="Scan ledger + archives for cross-night codify candidates",
    )
    scan_p.add_argument(
        "--ledger",
        default=None,
        help=(
            "Path to the live ledger.  When omitted, defaults to the "
            "metric-specific ledger for --metric (composite → "
            "planning/self_improve_ledger.jsonl; aocc → "
            "planning/self_improve_ledger_aocc.jsonl)."
        ),
    )
    scan_p.add_argument(
        "--metric",
        choices=["composite", "aocc"],
        default="composite",
        help=(
            "Resolve the default ledger path for this metric when --ledger is "
            "not given (default: composite, preserving the historical path).  "
            "Under the AOCC nightly default (2026-07-09) pass --metric aocc so "
            "the codify scan reads the active ledger; archive pooling is "
            "scoped to the same metric either way (§12.1)."
        ),
    )
    scan_p.add_argument(
        "--archive-dir",
        default=None,
        help=("Directory of rotated ledger archives (default: <ledger parent>/done — typically planning/done/)"),
    )
    scan_p.add_argument(
        "--no-include-archives",
        dest="include_archives",
        action="store_false",
        help=(
            "Skip the archive directory entirely (live ledger only).  By "
            "default the scan pools archive evidence in chronological order "
            "before the live ledger so cross-night signal accumulates "
            "across nightly rotations."
        ),
    )
    scan_p.set_defaults(include_archives=True)
    scan_p.add_argument(
        "--min-nights",
        type=int,
        default=2,
        help=(
            "Minimum number of distinct accept dates a candidate must "
            "carry to be surfaced (default: 2, matching §9.3 'k>=2 "
            "confirmed accepts on distinct nights')."
        ),
    )
    scan_p.add_argument(
        "--no-require-positive-min-ci",
        dest="require_positive_min_ci",
        action="store_false",
        help=(
            "Surface candidates even when at least one contributing "
            "record's ci_low fell <= 0.  Default behaviour requires every "
            "contributing accept to have cleared its own per-record "
            "statistical-accept gate (the screening bootstrap CI > 0)."
        ),
    )
    scan_p.set_defaults(require_positive_min_ci=True)
    scan_p.add_argument(
        "--confirmed-only",
        action="store_true",
        help=(
            "Restrict the input to iteration records carrying "
            "confirmed=True (V2 §6.4 confirmation gate, PR #255).  "
            "Default off so scans against pre-§6.4 ledgers still "
            "produce evidence."
        ),
    )
    scan_p.add_argument(
        "--pooled-ci-n-boot",
        type=int,
        default=2000,
        help="Bootstrap resamples for the pooled per-record delta CI (default: 2000).",
    )
    scan_p.add_argument(
        "--pooled-ci-confidence",
        type=float,
        default=0.95,
        help="Two-sided confidence level for the pooled CI (default: 0.95).",
    )
    scan_p.add_argument(
        "--pooled-ci-seed",
        type=int,
        default=42,
        help="RNG seed for the pooled bootstrap CI (default: 42).",
    )
    scan_p.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit one JSON object per candidate instead of the text report.",
    )
    scan_p.add_argument(
        "--top",
        type=int,
        default=0,
        help=(
            "Limit the report to the top N candidates by "
            "(n_distinct_nights, mean_delta, n_accepts).  0 (default) "
            "prints every candidate that clears the gates."
        ),
    )
    scan_p.add_argument(
        "--include-already-codified",
        action="store_true",
        help=(
            "Include candidates whose implied source edit is already "
            "live in the seed-spec factories (quick + loop registries).  "
            "Off by default so the operator's attention stays on "
            "actionable evidence; an already-codified candidate is "
            "tagged ``[already codified]`` in the report when this flag "
            "is set and ``already_codified=true`` in the JSON output."
        ),
    )
    scan_p.add_argument(
        "--no-suppress-codified",
        dest="suppress_codified",
        action="store_false",
        help=(
            "Alias for --include-already-codified that reads more "
            "naturally when paired with --json (the JSON consumer can "
            "filter on the already_codified field itself)."
        ),
    )
    scan_p.set_defaults(suppress_codified=True)
    scan_p.add_argument(
        "--rejections",
        default=None,
        help=(
            "Path to the codify-rejections JSON file (default: the "
            "metric-specific path — composite → "
            "planning/self_improve_rejections.json; aocc → "
            "planning/self_improve_rejections_aocc.json).  A missing "
            "file is an empty rejection memory.  Candidates whose slot "
            "was rejected by a recorded operator decision are hidden "
            "from the report (and skipped by --apply-top) until the "
            "post-rejection evidence alone reaches --min-fresh-nights "
            "distinct nights, at which point the slot resurrects with "
            "a rejection-history tag.  Record decisions with the "
            "codify-reject subcommand."
        ),
    )
    scan_p.add_argument(
        "--min-fresh-nights",
        dest="min_fresh_nights",
        type=int,
        default=None,
        help=(
            "Distinct post-rejection evidence nights required before a "
            "rejected slot resurrects (default: 2, matching the "
            "--min-nights actionability bar — evidence newer than an "
            "operator rejection must clear the same k>=2 gate as a "
            "brand-new candidate; pre-rejection nights were already "
            "adjudicated by the rejecting A/B).  1 restores the "
            "pre-2026-08-08 'any single fresh night resurrects' "
            "behaviour, which had a measured 0/3 hit rate against "
            "12-seed paired A/Bs."
        ),
    )
    scan_p.add_argument(
        "--include-rejected",
        action="store_true",
        help=(
            "Include candidates suppressed by the rejection memory.  "
            "Off by default so the operator's attention stays on "
            "actionable evidence; a rejected candidate is tagged "
            "``[rejected YYYY-MM-DD: reason]`` in the report when this "
            "flag is set and ``rejected=true`` in the JSON output."
        ),
    )
    scan_p.add_argument(
        "--widen-bounds",
        action="store_true",
        help=(
            "Append a 'Bound-widening candidates' section that pairs "
            "bidirectional codify candidates (same (class, param) slot "
            "with accepts in both 'up' and 'down' directions) into "
            "proposed catalog bound updates.  See the *Mutation-bound "
            "widening rule* idea under planning/SELF_IMPROVEMENT_LOG.md. "
            "In --json mode, every widening candidate is emitted on its "
            'own line tagged with `"_type": "widening_candidate"` so '
            "the JSON stream remains line-delimited and a consumer can "
            "filter by type."
        ),
    )
    scan_p.add_argument(
        "--widen-factor",
        type=float,
        default=1.5,
        help=(
            "Multiplicative widening applied to the observed min/max of "
            "every accepted new_value to produce the proposed bound "
            "(default: 1.5).  Symmetric in log space for "
            "log_uniform_perturb; rounded outward for integer_add; "
            "linear-multiplicative for float_uniform.  Must be > 1.0.  "
            "When --widen-auto-tune is set this value is used as the "
            "fallback for slots whose current_bounds are None (no rule)."
        ),
    )
    scan_p.add_argument(
        "--widen-auto-tune",
        action="store_true",
        help=(
            "Size the widen factor per-candidate from the observed "
            "spread relative to the catalog bound.  Narrow observed "
            "spread (high agreement) → larger factor for more "
            "exploration headroom; wide spread (low agreement) → "
            "smaller factor focused on the consensus.  See "
            "panobbgo.self_improve._auto_tune_widen_factor for the "
            "rule.  Falls back to --widen-factor when no catalog rule "
            "targets the slot."
        ),
    )
    scan_p.add_argument(
        "--widen-factor-min",
        type=float,
        default=1.1,
        dest="widen_factor_min",
        help=(
            "Auto-tuned widen factor at the wide-spread end (observed "
            "range covers the whole catalog range).  Must be > 1.0.  "
            "Only consulted when --widen-auto-tune is set.  Default: 1.1."
        ),
    )
    scan_p.add_argument(
        "--widen-factor-max",
        type=float,
        default=2.5,
        dest="widen_factor_max",
        help=(
            "Auto-tuned widen factor at the narrow-spread end (observed "
            "range is a tight band inside the catalog).  Must be >= "
            "--widen-factor-min.  Only consulted when --widen-auto-tune "
            "is set.  Default: 2.5."
        ),
    )
    scan_p.add_argument(
        "--open-pr",
        action="store_true",
        help=(
            "After applying the top actionable candidate (implies "
            "--apply-top when not set explicitly), create a git branch, "
            "commit the diff, push it, and open a draft PR via ``gh pr "
            "create``.  Dedups against ``gh pr list --state open`` "
            "using the codify-slot marker embedded in every codify PR "
            "body — an existing open PR for the same (class, param) "
            "slot skips the open-PR step (with a note) rather than "
            "producing a duplicate.  Requires the ``gh`` and ``git`` "
            "binaries on PATH (defaults; override with --pr-gh-bin / "
            "--pr-git-bin).  Inert when combined with --apply-dry-run "
            "(the dry-run just prints the commands the driver would "
            "run instead of executing them).  See V2 §9.5 step 4 in "
            "planning/SELF_IMPROVEMENT_LOOP.md."
        ),
    )
    scan_p.add_argument(
        "--pr-branch-prefix",
        default="claude/codify",
        help=(
            "Prefix for the codify PR branch name (default "
            "'claude/codify').  Full branch name is "
            "'<prefix>-<class_snake>-<param_snake>-<direction>'.  The "
            "'claude/' family matches the watcher-infrastructure naming "
            "convention (see V2 §9.5 step 4 follow-up idea in "
            "planning/SELF_IMPROVEMENT_LOG.md)."
        ),
    )
    scan_p.add_argument(
        "--pr-base",
        default="master",
        help=(
            "Base branch the codify PR targets (default 'master').  "
            "Surfaced verbatim in ``gh pr create --base`` and in the "
            "test-plan snippet of the PR body."
        ),
    )
    scan_p.add_argument(
        "--pr-gh-bin",
        default="gh",
        help="Path to the ``gh`` binary (default 'gh').",
    )
    scan_p.add_argument(
        "--pr-git-bin",
        default="git",
        help="Path to the ``git`` binary (default 'git').",
    )
    scan_p.add_argument(
        "--apply-top",
        action="store_true",
        help=(
            "After printing the candidate report, take the top "
            "actionable candidate and apply its implied source edits "
            "to panobbgo/harness.py in place.  Kwarg candidates: every "
            "(ClassName, {param_name: value, ...}) heuristic / analyzer "
            "literal across the four registry factories (quick / "
            "standard / full / loop) is updated to the candidate's "
            "proposed_codify_value; sites already at-or-beyond the "
            "proposal are left alone (deliberately-tighter sibling "
            "specs preserved).  Structural candidates (add_/drop_"
            "heuristic, add_/drop_analyzer — shipped 2026-07-01) "
            "insert or remove a tuple entry in the target bucket, "
            "scoped to the specs listed in the candidate's "
            "strategy_names; drop safety guards preserve buckets with "
            "one entry and skip specs where the class is already "
            "absent, add safety guards skip specs where the class is "
            "already present.  The operator runs tests + commits + "
            "opens the PR manually — this driver does NOT touch git or "
            "the working-tree commit state.  Combine with "
            "--apply-dry-run to preview the edits without writing.  "
            "See the *codify-scan --apply-top driver* entry under "
            "planning/SELF_IMPROVEMENT_LOG.md for the design and the "
            "2026-06-30 *Follow-up ideas* seed that motivates the "
            "structural extension shipped 2026-07-01."
        ),
    )
    scan_p.add_argument(
        "--apply-dry-run",
        action="store_true",
        help=(
            "With --apply-top, print the edits the driver would apply "
            "but do not write them to disk.  Useful for previewing "
            "what the apply would do before committing to the in-place "
            "rewrite.  Inert without --apply-top."
        ),
    )
    scan_p.add_argument(
        "--apply-include-bidirectional",
        action="store_true",
        help=(
            "With --apply-top, do NOT skip candidates whose (class, "
            "param) slot also appears with the opposite direction in "
            "the visible candidate list.  Default behaviour skips "
            "bidirectional slots because the right action for those is "
            "usually a catalog bound update (see --widen-bounds) rather "
            "than a default shift — applying either direction's edit "
            "would be a guess against contradictory ledger evidence.  "
            "Override only when the operator has a specific reason to "
            "force a default shift on a bidirectional slot."
        ),
    )
    scan_p.add_argument(
        "--apply-format",
        action="store_true",
        help=(
            "After --apply-top writes edits to disk, run "
            "``uv run ruff format`` on the modified files so the "
            "operator does not have to remember.  Inert with "
            "--apply-dry-run (nothing to format) and inert when no "
            "site needed editing.  Non-zero rc from the formatter "
            "propagates so a CI wrapper surfaces the failure."
        ),
    )
    scan_p.add_argument(
        "--apply-run-tests",
        action="store_true",
        help=(
            "After --apply-top writes edits to disk (and after "
            "--apply-format if requested), run ``uv run pytest "
            "tests/test_self_improve.py`` so the operator gets "
            "immediate feedback that the codify edit did not break "
            "the codify plumbing.  Inert with --apply-dry-run and "
            "inert when no site needed editing.  Non-zero rc from "
            "pytest propagates so a CI wrapper surfaces the failure."
        ),
    )
    scan_p.set_defaults(func=_cmd_codify_scan)

    rej_p = sub.add_parser(
        "codify-reject",
        help="Record a codify-slot rejection so codify-scan stops re-surfacing it",
        description=(
            "Append one rejection record to the metric's codify-rejections "
            "JSON file (the codify scan's rejection memory).  Use after an "
            "A/B on the current spec rejects a scan candidate, or when a "
            "candidate is moot (e.g. it tunes a class that was since "
            "dropped).  The scan suppresses a matching candidate until its "
            "post---date ledger evidence alone reaches the scan's "
            "--min-fresh-nights bar (default 2 distinct nights), at which "
            "point the slot resurrects automatically.  Always pair the "
            "record with a dated entry in "
            "planning/SELF_IMPROVEMENT_LOG.md carrying the full numbers "
            "(--log-ref should point at it)."
        ),
    )
    rej_p.add_argument(
        "--metric",
        choices=["composite", "aocc"],
        default="composite",
        help=(
            "Resolve the default rejections path for this metric when "
            "--rejections is not given (default: composite).  Under the "
            "AOCC nightly default pass --metric aocc."
        ),
    )
    rej_p.add_argument(
        "--rejections",
        default=None,
        help=(
            "Rejections file to append to (default: the metric-specific "
            "path, e.g. planning/self_improve_rejections_aocc.json).  "
            "Created if missing."
        ),
    )
    rej_p.add_argument(
        "--class-name",
        required=True,
        help="Heuristic / analyzer class of the rejected slot (e.g. LBFGSB).",
    )
    rej_p.add_argument(
        "--param",
        default="",
        help="Kwarg name of the rejected slot; omit for structural ops.",
    )
    rej_p.add_argument(
        "--op",
        default=None,
        choices=["add_heuristic", "drop_heuristic", "add_analyzer", "drop_analyzer"],
        help="Structural op of the rejected slot; omit for kwarg slots.",
    )
    rej_p.add_argument(
        "--direction",
        default=None,
        help=(
            "Optional direction restriction ('up' / 'down' / a "
            "categorical repr).  Omit to reject the slot in every "
            "direction.  Restrict when only one direction was tested — "
            "e.g. rejecting update_interval 'down' leaves a future 'up' "
            "signal actionable."
        ),
    )
    rej_p.add_argument(
        "--date",
        default=None,
        help=(
            "Rejection decision date, YYYY-MM-DD (default: today, UTC).  "
            "This is the A/B session date, not the evidence nights — "
            "suppression covers evidence up to and including this date."
        ),
    )
    rej_p.add_argument(
        "--reason",
        required=True,
        help=(
            "One-line why, with the decisive numbers (e.g. '12-seed "
            "paired quick A/B mean Δ -0.0012, CI [-0.0097,+0.0072]').  "
            "Surfaced verbatim by codify-scan --include-rejected."
        ),
    )
    rej_p.add_argument(
        "--log-ref",
        default="",
        help=(
            "Pointer to the full write-up, conventionally the dated "
            "SELF_IMPROVEMENT_LOG.md heading (e.g. "
            "'planning/SELF_IMPROVEMENT_LOG.md 2026-08-03')."
        ),
    )
    rej_p.set_defaults(func=_cmd_codify_reject)

    return parser


# Overridable subprocess runner so tests can intercept the
# ``uv run ruff format`` / ``uv run pytest`` invocations without
# shelling out to the real binaries.  Signature matches
# :func:`subprocess.run`'s minimal shape (list of args, returns an
# object with a ``.returncode`` int attribute).
def _run_subprocess(cmd: Sequence[str]) -> "subprocess.CompletedProcess[Any]":
    return subprocess.run(list(cmd), check=False)


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
    from panobbgo.self_improve import (
        LoopConfig,
        SelfImprover,
        default_catalog,
        default_structural_catalog,
        ledger_path_for_metric,
    )

    ledger = args.ledger if args.ledger is not None else ledger_path_for_metric(args.metric)
    catalog = default_structural_catalog() if args.structural else default_catalog()
    try:
        holdout_seeds = _parse_seed_list(args.holdout_base_seeds)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    try:
        extra_families = None
        if getattr(args, "extra_highdim", False):
            from panobbgo.harness_randomized import make_highdim_families

            extra_families = make_highdim_families()
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
            ledger_path=ledger,
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
            adaptive_prime_include_archives=args.adaptive_prime_include_archives,
            adaptive_prime_archive_dir=args.adaptive_prime_archive_dir,
            structural_per_class_arms=args.structural_per_class_arms,
            structural_borrow_alpha=args.structural_borrow_alpha,
            structural_borrow_horizon=args.structural_borrow_horizon,
            holdout_base_seed=args.holdout_base_seed,
            holdout_base_seeds=holdout_seeds,
            holdout_iterations=args.holdout_iterations,
            holdout_iteration_offset=args.holdout_iteration_offset,
            holdout_eps_overfit=args.holdout_eps_overfit,
            paired=args.paired,
            metric=args.metric,
            registry=args.registry,
            extra_families=extra_families,
            inactivity_relax_after=args.inactivity_relax_after,
            inactivity_relax_factor=args.inactivity_relax_factor,
            inactivity_min_eps_accept=args.inactivity_min_eps_accept,
            bandit_reward_shaping=args.bandit_reward_shaping,
            confirm_accepts=args.confirm_accepts,
            confirm_iteration_offset=args.confirm_iteration_offset,
            sync_eval=args.sync_eval,
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
    # §6.4 same-night confirmation gate: when --confirm-accepts is on,
    # a screening accept may be overturned at the post-confirmation
    # gate.  Surface the count of these "confirm rejects" alongside
    # accepts / no-ops so an operator can see at a glance how often
    # the gate is catching screening noise spikes.
    n_confirm_reject = sum(1 for r in records if r.confirmed is False)
    n_total = len(records)
    print()
    confirm_blurb = f", {n_confirm_reject} confirm-reject" if cfg.confirm_accepts else ""
    print(
        f"[self_improve] completed: {n_total} iter, {n_accepts} accept, "
        f"{n_skips} skip, {n_no_op} no-op{confirm_blurb}, ledger={cfg.ledger_path}"
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
    from panobbgo.self_improve import ledger_path_for_metric, load_ledger

    ledger = args.ledger if args.ledger is not None else ledger_path_for_metric(args.metric)
    path = pathlib.Path(ledger)
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
    # §6.4 same-night confirmation gate: failed confirmations land as
    # ``record_type="confirm_reject"`` next to the iteration record
    # they refer to.  Surface their count so summary readers can see
    # how often the gate fires.
    confirm_reject_records = [r for r in records if r.get("record_type") == "confirm_reject"]
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
    if confirm_reject_records:
        # §6.4: every confirm_reject record corresponds to a screening
        # accept the gate overturned; the screening_accepts denominator
        # is the gate's "attempts" count (final accepts + confirm
        # rejects).  Surface both so the operator can see the gate's
        # rejection rate at a glance.
        n_screen_attempts = len(accepted) + len(confirm_reject_records)
        gate_rate = (len(confirm_reject_records) / n_screen_attempts) if n_screen_attempts else 0.0
        print(f"Confirm-rej:   {len(confirm_reject_records)}  ({gate_rate:.1%} of screening accepts overturned)")
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
    if confirm_reject_records:
        # §6.4 confirmation gate: surface the screen/confirm/pooled
        # deltas for each overturned screening accept so an operator
        # can see whether the gate is catching noise spikes (screen Δ
        # ≫ confirm Δ) or systematic regressions (screen Δ ≈ confirm
        # Δ but ci_low ≤ 0).
        print("Confirm rejects (§6.4 same-night gate):")
        for r in confirm_reject_records:
            print(
                f"  iter={r.get('iteration')}  "
                f"screen_Δ={r.get('screen_delta', 0.0):+.4f}  "
                f"confirm_Δ={r.get('confirm_delta', 0.0):+.4f}  "
                f"pooled_Δ={r.get('pooled_delta', 0.0):+.4f}  "
                f"CI=[{r.get('pooled_ci_low', 0.0):+.4f},"
                f"{r.get('pooled_ci_high', 0.0):+.4f}]"
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


def _format_old_new(old: Any, new: Any) -> str:
    """Compact ``old -> new`` representation for the codify-scan report.

    Floats round to 6 significant digits so ``log_uniform_perturb``
    micro-perturbations (e.g. ``0.10088523662787297``) don't dominate
    the line.  Booleans / strings round-trip via ``repr`` so the report
    reads ``True -> False`` rather than ``True -> False``.
    """

    def _fmt(v: Any) -> str:
        if isinstance(v, bool):
            return repr(v)
        if isinstance(v, float):
            return f"{v:.6g}"
        return repr(v)

    return f"{_fmt(old)} -> {_fmt(new)}"


def _print_codify_candidate(
    cand: Any,
    *,
    pooled_ci_n_boot: int,
    pooled_ci_confidence: float,
    pooled_ci_seed: int,
) -> None:
    """Render one :class:`CodifyCandidate` as a human-readable block.

    Format chosen so an operator reading
    ``planning/self_improve_summary.txt`` after a nightly run can
    triage candidates without reaching for the ledger: lead with the
    slot identifier, pooled stats next, then a compact evidence list.
    """
    op_label = f" op={cand.op}" if cand.op else ""
    slot = f"{cand.class_name}.{cand.param_name}" if cand.param_name else cand.class_name
    codified_tag = " [already codified]" if cand.already_codified else ""
    rejected_tag = ""
    if getattr(cand, "rejected", False):
        n_fresh = sum(1 for d in cand.distinct_dates if d > cand.rejected_on)
        if n_fresh:
            # Post-rejection evidence is accruing but still below the
            # resurrection bar — surface the count so an operator
            # auditing with --include-rejected sees the progress.
            rejected_tag = f" [rejected {cand.rejected_on}; fresh nights since: {n_fresh}, below resurrection bar]"
        else:
            rejected_tag = f" [rejected {cand.rejected_on}]"
    elif getattr(cand, "rejected_on", ""):
        # A rejection matches the slot but enough newer evidence has
        # accrued since — actionable again, yet the operator should
        # re-verify rather than trust pooled stats straddling the
        # spec change.
        rejected_tag = f" [fresh evidence since rejection {cand.rejected_on}]"
    print(f"- {slot} [{cand.rule_kind}{op_label}] direction={cand.direction}{codified_tag}{rejected_tag}")
    ci_low, ci_high = cand.pooled_bootstrap_ci(
        n_boot=pooled_ci_n_boot,
        confidence=pooled_ci_confidence,
        seed=pooled_ci_seed,
    )
    n_confirmed = sum(1 for f in cand.confirmed_flags if f is True)
    n_unconfirmed = sum(1 for f in cand.confirmed_flags if f is None)
    conf_note = ""
    if n_confirmed:
        conf_note = f"  confirmed={n_confirmed}/{cand.n_accepts}"
    elif n_unconfirmed == cand.n_accepts:
        conf_note = "  (legacy: no confirmation gate)"
    print(
        f"    n_accepts={cand.n_accepts}  n_nights={cand.n_distinct_nights}  "
        f"mean_Δ={cand.mean_delta:+.4f}  pooled_CI{int(pooled_ci_confidence * 100)}%="
        f"[{ci_low:+.4f},{ci_high:+.4f}]  min_record_ci_low={cand.min_ci_low:+.4f}" + conf_note
    )
    dates_str = ", ".join(cand.distinct_dates)
    print(f"    nights: {dates_str}")
    if getattr(cand, "rejected_on", ""):
        # Surface the rejection record driving the tag so the operator
        # need not open the rejections file / log to see why.
        reason = cand.rejection_reason or "(no reason recorded)"
        print(f"    rejection: {cand.rejected_on}  {reason}")
    if cand.live_codified_values:
        # Surface the live values driving the already-codified verdict
        # so the operator can confirm the suppression rule's reasoning.
        live_repr = ", ".join(repr(v) for v in cand.live_codified_values)
        print(f"    live seed value(s): {live_repr}")
    # Surface the value a codify edit would shift the seed-spec to (the
    # median of new_values, rounded outward).  Saves the operator from
    # hand-computing it; the queued ``--open-pr`` driver will consume
    # the same field.
    proposed = cand.proposed_codify_value()
    if proposed is not None:
        if isinstance(proposed, bool):
            proposed_repr = repr(proposed)
        elif isinstance(proposed, float):
            proposed_repr = f"{proposed:.6g}"
        else:
            proposed_repr = repr(proposed)
        print(f"    proposed codify value: {proposed_repr}")
    strategies = sorted(set(cand.strategy_names))
    if strategies:
        print(f"    strategies: {', '.join(strategies)}")
    print("    evidence:")
    for i in range(cand.n_accepts):
        ts = cand.timestamps[i][:19].replace("T", " ")
        old_new = _format_old_new(cand.old_values[i], cand.new_values[i])
        confirmed_marker = ""
        if cand.confirmed_flags[i] is True:
            confirmed_marker = "  [confirmed]"
        elif cand.confirmed_flags[i] is False:
            confirmed_marker = "  [confirm-rejected]"
        print(
            f"      {ts}  Δ={cand.deltas[i]:+.4f}  CI=[{cand.ci_lows[i]:+.4f},"
            f"{cand.ci_highs[i]:+.4f}]  {cand.strategy_names[i]}/"
            f"{cand.class_name}.{cand.param_name}: {old_new}{confirmed_marker}"
        )


def _format_bound(value: Any, *, integer: bool) -> str:
    """Compact representation of one widening-bound numeric value."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return repr(value)
    if integer:
        return str(int(round(f)))
    if f == 0.0:
        return "0"
    return f"{f:.6g}"


def _print_widening_candidate(cand: Any) -> None:
    """Render one :class:`WideningCandidate` as a human-readable block.

    The output keeps the codify-candidate visual style — slot header,
    one stats line, observed / current / proposed bounds, then the
    contributing per-direction evidence summaries — so an operator can
    triage the widening section the same way they triage the codify
    section above it.
    """
    integer = cand.rule_kind == "integer_add"
    obs = f"[{_format_bound(cand.observed_lo, integer=integer)}, {_format_bound(cand.observed_hi, integer=integer)}]"
    if cand.current_bounds is None:
        cur = "(no rule)"
    else:
        cur_lo, cur_hi = cand.current_bounds
        cur = f"[{_format_bound(cur_lo, integer=integer)}, {_format_bound(cur_hi, integer=integer)}]"
    prop = f"[{_format_bound(cand.proposed_lo, integer=integer)}, {_format_bound(cand.proposed_hi, integer=integer)}]"
    tag = ""
    if cand.proposal_is_wider:
        tag = " [widens current]"
    elif cand.proposal_is_tighter:
        tag = " [tightens current — focuses bandit on observed range]"
    elif cand.current_bounds is not None:
        tag = " [partial overlap]"
    print(f"- {cand.class_name}.{cand.param_name} [{cand.rule_kind}] bidirectional{tag}")
    print(
        f"    n_accepts={cand.n_accepts}  n_nights={cand.n_distinct_nights}  "
        f"observed={obs}  current={cur}  proposed={prop}  widen_factor={cand.widen_factor}"
    )
    nights = ", ".join(cand.distinct_dates)
    print(f"    nights: {nights}")
    print(
        f"    up:   n_accepts={cand.up_candidate.n_accepts}  "
        f"mean_Δ={cand.up_candidate.mean_delta:+.4f}  "
        f"n_nights={cand.up_candidate.n_distinct_nights}"
    )
    print(
        f"    down: n_accepts={cand.down_candidate.n_accepts}  "
        f"mean_Δ={cand.down_candidate.mean_delta:+.4f}  "
        f"n_nights={cand.down_candidate.n_distinct_nights}"
    )


def _cmd_codify_scan(args: argparse.Namespace) -> int:
    """V2 §9.3 / §9.5 step 4 — scan ledger + archives for codify candidates.

    Reads ``--ledger`` plus (by default) every archive under
    ``--archive-dir``, groups accepted iteration records by
    ``(class, param, direction)`` (or ``(op, class)`` for structural
    ops), and surfaces every group with at least ``--min-nights``
    distinct accept dates.  Each candidate is a *suggestion* — the
    operator (or a future ``--open-pr`` follow-up) translates "the
    bandit raised ``Nearby.radius`` on N distinct nights" into a
    concrete source edit.

    When ``--widen-bounds`` is set, an extra *Bound-widening candidates*
    section pairs every bidirectional ``(class_name, param_name)`` slot
    (same slot with accepts in both ``"up"`` and ``"down"`` directions)
    into a proposed catalog ``MutationRule.bounds`` update — see the
    *Mutation-bound widening rule* idea under
    ``planning/SELF_IMPROVEMENT_LOG.md``.
    """
    import json as _json

    from panobbgo.self_improve import (
        DEFAULT_RESURRECT_MIN_FRESH_NIGHTS,
        aggregate_codify_candidates,
        annotate_codified_status,
        annotate_rejected_status,
        default_codify_apply_sources,
        default_codify_registries,
        detect_widening_candidates,
        ledger_path_for_metric,
        load_codify_rejections,
        load_ledgers_for_codify_scan,
        rejections_path_for_metric,
    )

    if args.min_nights < 1:
        print(f"Error: --min-nights must be >= 1, got {args.min_nights}", file=sys.stderr)
        return 1
    min_fresh_arg = getattr(args, "min_fresh_nights", None)
    min_fresh_nights = DEFAULT_RESURRECT_MIN_FRESH_NIGHTS if min_fresh_arg is None else int(min_fresh_arg)
    if min_fresh_nights < 1:
        print(f"Error: --min-fresh-nights must be >= 1, got {min_fresh_nights}", file=sys.stderr)
        return 1

    ledger = args.ledger if args.ledger is not None else ledger_path_for_metric(args.metric)
    ledger_path = pathlib.Path(ledger)
    if not ledger_path.exists():
        # Match _cmd_summary's behaviour: missing live ledger is an
        # error, but the scanner can still draw on archives — surface
        # both signals so the operator can correct the path.
        print(f"Note: live ledger not found: {ledger_path}", file=sys.stderr)

    records = load_ledgers_for_codify_scan(
        str(ledger_path),
        include_archives=args.include_archives,
        archive_dir=args.archive_dir,
    )
    if not records:
        print(
            f"(no records found — checked ledger={ledger_path}"
            + (f", archive_dir={args.archive_dir or 'default'}" if args.include_archives else "")
            + ")"
        )
        return 0

    candidates = aggregate_codify_candidates(
        records,
        min_nights=args.min_nights,
        require_positive_min_ci=args.require_positive_min_ci,
        confirmed_only=args.confirmed_only,
    )

    # Annotate already-codified status against the live seed-spec
    # factories of the selected metric's regime (composite → quick +
    # loop registries; aocc → make_ioh_strategies) so the report can
    # suppress candidates whose implied source edit would be a no-op.
    metric = str(getattr(args, "metric", "composite") or "composite")
    annotate_codified_status(candidates, registries=default_codify_registries(metric=metric))

    # Annotate rejection status against the metric's rejection memory
    # (the "scan hygiene" fix from the 2026-08-02/03 log entries) so the
    # report — and --apply-top — skip slots an operator A/B already
    # rejected on the current spec, until fresh evidence accrues.
    rejections_arg = getattr(args, "rejections", None)
    rejections_path = rejections_arg if rejections_arg is not None else rejections_path_for_metric(metric)
    try:
        rejections = load_codify_rejections(rejections_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    annotate_rejected_status(candidates, rejections, min_fresh_nights=min_fresh_nights)

    n_total_candidates = len(candidates)
    suppress_codified = getattr(args, "suppress_codified", True)
    include_already_codified = getattr(args, "include_already_codified", False)
    suppress = suppress_codified and not include_already_codified
    suppress_rejected = not getattr(args, "include_rejected", False)
    visible_candidates = []
    n_suppressed = 0
    n_rejected_hidden = 0
    for c in candidates:
        if suppress and c.already_codified:
            n_suppressed += 1
        elif suppress_rejected and c.rejected:
            n_rejected_hidden += 1
        else:
            visible_candidates.append(c)

    if args.top > 0:
        visible_candidates = visible_candidates[: args.top]

    widen_bounds = bool(getattr(args, "widen_bounds", False))
    widen_factor = float(getattr(args, "widen_factor", 1.5))
    widen_auto_tune = bool(getattr(args, "widen_auto_tune", False))
    widen_factor_min = float(getattr(args, "widen_factor_min", 1.1))
    widen_factor_max = float(getattr(args, "widen_factor_max", 2.5))
    widening: List[Any] = []
    if widen_bounds:
        widening = detect_widening_candidates(
            candidates,
            widen_factor=widen_factor,
            auto_tune=widen_auto_tune,
            auto_tune_min_factor=widen_factor_min,
            auto_tune_max_factor=widen_factor_max,
        )

    if args.as_json:
        # JSON output always emits every candidate (including
        # already-codified) — the consumer can filter on the new
        # ``already_codified`` field itself.  The ``--top`` truncation
        # still applies for parity with the human-readable mode.
        if args.top > 0:
            json_candidates = candidates[: args.top]
        else:
            json_candidates = candidates
        for cand in json_candidates:
            d = cand.to_dict()
            ci_low, ci_high = cand.pooled_bootstrap_ci(
                n_boot=args.pooled_ci_n_boot,
                confidence=args.pooled_ci_confidence,
                seed=args.pooled_ci_seed,
            )
            d["pooled_ci_low"] = ci_low
            d["pooled_ci_high"] = ci_high
            d["pooled_ci_confidence"] = args.pooled_ci_confidence
            d["_type"] = "codify_candidate"
            print(_json.dumps(d, sort_keys=True))
        # Widening candidates ride the same line-delimited JSON stream
        # tagged with `_type` so a JSON consumer can filter by type and
        # the existing codify-candidate JSON shape stays byte-identical.
        for w in widening:
            wd = w.to_dict()
            wd["_type"] = "widening_candidate"
            print(_json.dumps(wd, sort_keys=True))
        return 0

    # Human-readable report.
    src_note = f"live={ledger_path}"
    if args.include_archives:
        src_note += f"  archives={args.archive_dir or '<ledger parent>/done'}"
    gates: List[str] = [f"min_nights>={args.min_nights}"]
    if args.require_positive_min_ci:
        gates.append("all_record_ci_low>0")
    else:
        gates.append("any_record_ci_low")
    if args.confirmed_only:
        gates.append("confirmed_only")
    if suppress:
        gates.append("hide_already_codified")
    if suppress_rejected:
        gates.append(f"hide_rejected(resurrect_fresh_nights>={min_fresh_nights})")
    print(f"Codify scan ({src_note})")
    print(f"  gates: {', '.join(gates)}")
    print(f"  records scanned: {len(records)}")
    rejected_note = f", {n_rejected_hidden} rejected" if n_rejected_hidden else ""
    if suppress or n_rejected_hidden:
        print(
            f"  candidates surfaced: {len(visible_candidates)} "
            f"(of {n_total_candidates}; {n_suppressed} already codified{rejected_note}, hidden)"
        )
    else:
        print(f"  candidates surfaced: {len(visible_candidates)}")
    if not visible_candidates:
        if n_rejected_hidden and (n_suppressed + n_rejected_hidden):
            print(
                "  (every candidate is already codified or rejected — pass "
                "--include-already-codified / --include-rejected to audit them; "
                f"rejection memory: {rejections_path})"
            )
        elif suppress and n_suppressed:
            print(
                "  (every candidate is already codified in the seed factories — "
                "pass --include-already-codified to audit them)"
            )
        else:
            print("  (no group cleared the gates)")
        return 0
    print()
    for cand in visible_candidates:
        _print_codify_candidate(
            cand,
            pooled_ci_n_boot=args.pooled_ci_n_boot,
            pooled_ci_confidence=args.pooled_ci_confidence,
            pooled_ci_seed=args.pooled_ci_seed,
        )
        print()
    if widen_bounds:
        if widen_auto_tune:
            factor_label = f"widen_factor=auto-tune [{widen_factor_min}, {widen_factor_max}] (fallback={widen_factor})"
        else:
            factor_label = f"widen_factor={widen_factor}"
        print(f"Bound-widening candidates ({factor_label}):")
        print(f"  bidirectional pairs surfaced: {len(widening)}")
        if not widening:
            print("  (no bidirectional pattern surfaced — every numeric candidate fired in only one direction)")
        else:
            print()
            for w in widening:
                _print_widening_candidate(w)
                print()

    apply_top = bool(getattr(args, "apply_top", False))
    open_pr = bool(getattr(args, "open_pr", False))
    # --open-pr implies --apply-top: opening a PR without an apply would
    # produce an empty commit.  Explicit here rather than at parse time so
    # the flag description reads cleanly.
    if open_pr and not apply_top:
        apply_top = True
    if apply_top:
        rc = _apply_top_codify_candidate(
            visible_candidates,
            all_candidates=candidates,
            sources=default_codify_apply_sources(metric=metric),
            dry_run=bool(getattr(args, "apply_dry_run", False)),
            include_bidirectional=bool(getattr(args, "apply_include_bidirectional", False)),
            open_pr=open_pr,
            pr_branch_prefix=str(getattr(args, "pr_branch_prefix", "claude/codify")),
            pr_base=str(getattr(args, "pr_base", "master")),
            pr_gh_bin=str(getattr(args, "pr_gh_bin", "gh")),
            pr_git_bin=str(getattr(args, "pr_git_bin", "git")),
            run_format=bool(getattr(args, "apply_format", False)),
            run_tests=bool(getattr(args, "apply_run_tests", False)),
        )
        if rc != 0:
            return rc
    return 0


def _cmd_codify_reject(args: argparse.Namespace) -> int:
    """Append one rejection record to the metric's codify-rejections file.

    The write side of the codify-scan rejection memory: reads the
    existing file (missing → empty), refuses an exact-duplicate slot
    (same class / param / op / direction — re-rejecting after fresh
    evidence should *update* the date, so the newer record replaces the
    older one for the same slot), appends, and rewrites the file
    pretty-printed (indent=2, trailing newline) so the diff the operator
    commits is one readable hunk.
    """
    import json as _json
    from datetime import date as _date

    from panobbgo.self_improve import (
        CodifyRejection,
        load_codify_rejections,
        rejections_path_for_metric,
    )

    rejections_arg = getattr(args, "rejections", None)
    metric = str(getattr(args, "metric", "composite") or "composite")
    path = pathlib.Path(rejections_arg if rejections_arg is not None else rejections_path_for_metric(metric))

    rejected_on = args.date or _date.today().isoformat()
    entry = CodifyRejection(
        class_name=args.class_name,
        param_name=args.param or "",
        op=args.op or None,
        direction=args.direction or None,
        rejected_on=rejected_on,
        reason=args.reason,
        log_ref=args.log_ref or "",
    )
    from datetime import datetime as _datetime

    try:
        _datetime.strptime(rejected_on, "%Y-%m-%d")
    except ValueError:
        print(f"Error: --date must be YYYY-MM-DD, got {rejected_on!r}", file=sys.stderr)
        return 1
    try:
        # Validates the existing file before any write.
        existing = load_codify_rejections(path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    slot = (entry.class_name, entry.param_name, entry.op, entry.direction)
    kept: List[Any] = []
    replaced = None
    for r in existing:
        if (r.class_name, r.param_name, r.op, r.direction) == slot:
            if r.rejected_on >= entry.rejected_on:
                print(
                    f"Error: an equal-or-newer rejection for this slot already exists (rejected_on={r.rejected_on}); nothing to do.",
                    file=sys.stderr,
                )
                return 1
            replaced = r
            continue
        kept.append(r)
    kept.append(entry)
    kept.sort(key=lambda r: (r.rejected_on, r.class_name, r.param_name, r.op or "", r.direction or ""))

    payload = {
        "_comment": (
            "Codify-scan rejection memory. Each entry suppresses the matching "
            "(class_name, param_name, op[, direction]) candidate until the ledger "
            "evidence nights strictly after rejected_on alone reach the scan's "
            "--min-fresh-nights bar (default 2 distinct nights, same k>=2 gate as a "
            "fresh candidate); pre-rejection nights never count, they were "
            "adjudicated by the rejecting A/B. Append via "
            "'scripts/self_improve.py codify-reject'; pair every entry with a dated "
            "planning/SELF_IMPROVEMENT_LOG.md write-up (log_ref)."
        ),
        "rejections": [r.to_dict() for r in kept],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(payload, indent=2, sort_keys=False) + "\n")

    slot_repr = entry.class_name + (f".{entry.param_name}" if entry.param_name else "")
    op_repr = f" op={entry.op}" if entry.op else ""
    dir_repr = f" direction={entry.direction}" if entry.direction else ""
    action = "Updated" if replaced is not None else "Recorded"
    print(f"{action} rejection: {slot_repr}{op_repr}{dir_repr}  rejected_on={rejected_on}")
    if replaced is not None:
        print(f"  (superseded the {replaced.rejected_on} record for the same slot)")
    print(f"  file: {path}  ({len(kept)} record(s))")
    return 0


def _apply_top_codify_candidate(
    visible_candidates: Sequence[Any],
    *,
    all_candidates: Sequence[Any],
    sources: Optional[Sequence[Tuple[str, Sequence[str]]]] = None,
    dry_run: bool,
    include_bidirectional: bool = False,
    open_pr: bool = False,
    pr_branch_prefix: str = "claude/codify",
    pr_base: str = "master",
    pr_gh_bin: str = "gh",
    pr_git_bin: str = "git",
    run_format: bool = False,
    run_tests: bool = False,
) -> int:
    """Apply the top actionable kwarg codify candidate to ``panobbgo/harness.py``.

    Picks the *first* visible candidate that is a kwarg edit (skips
    structural candidates whose ``op is not None`` — the queued
    ``--open-pr`` driver will handle those once structural source-edit
    primitives ship; see the V2 §9.5 step 4 *Next iteration ideas*
    entry in :doc:`/planning/SELF_IMPROVEMENT_LOG.md`).  Calls
    :func:`panobbgo.self_improve.apply_codify_candidate`, prints the
    derived edits, and writes them to disk unless ``dry_run`` is set.

    Operator-friendly output: one block per edit showing the
    ``factory/spec`` location, the ``old -> new`` value transition, and
    the source line.  At the end, a one-line "Wrote N file(s)" /
    "Would write" summary so the operator can re-run with
    ``--apply-dry-run`` off / on without re-reading the candidate list.

    When ``run_format`` is set and edits were written, follows the write
    with ``uv run ruff format`` on the modified files.  When
    ``run_tests`` is set, follows the (optional) format step with
    ``uv run pytest tests/test_self_improve.py``.  Both flags are
    inert under ``dry_run`` (nothing to format / test) and inert when
    no site needed editing.  Non-zero rc from either subprocess is
    propagated to the caller.

    Returns the process exit code: ``0`` on success (including the
    no-op case where no actionable candidate exists or no site needs
    editing), non-zero only if an actual error occurred.  Matches the
    convention of every other ``_cmd_*`` helper in this module — exit
    codes signal subprocess failure, not "nothing to do".
    """
    from panobbgo.self_improve import apply_codify_candidate

    print()
    print("Apply-top:")
    if not visible_candidates:
        print("  (no actionable candidates to apply)")
        return 0
    # Identify bidirectional (class, param) slots — same slot with both
    # an "up" and a "down" candidate anywhere in the full candidate
    # list, *including* already-codified ones.  Walking the full list
    # rather than just ``visible_candidates`` catches the post-codify
    # case where one direction (e.g. Nearby.radius "up") was just
    # codified into the source (so the up-candidate is now suppressed
    # as already_codified) but the opposite-direction evidence (down)
    # is still live — naive selection of the down-direction candidate
    # would un-codify the up signal and oscillate the bandit.  The
    # widening detector handles bidirectional slots cleanly via
    # --widen-bounds; the apply driver defers to it.
    direction_by_slot: Dict[Tuple[str, str], set] = {}
    for cand in all_candidates:
        if cand.op is not None or cand.direction not in ("up", "down"):
            continue
        direction_by_slot.setdefault((cand.class_name, cand.param_name), set()).add(cand.direction)
    bidirectional_slots = {slot for slot, dirs in direction_by_slot.items() if dirs == {"up", "down"}}
    # Pick the first non-bidirectional candidate.  Structural candidates
    # are now handled by
    # :func:`panobbgo.self_improve._scan_source_for_structural_edits`
    # (shipped 2026-07-01) — add / drop of a heuristic or analyzer
    # tuple in the target spec's bucket, scoped to the specs in the
    # candidate's ``strategy_names``.  Bidirectional-slot skip still
    # applies as before.
    chosen = None
    skipped_bidirectional = 0
    for cand in visible_candidates:
        slot = (cand.class_name, cand.param_name)
        if cand.op is None and not include_bidirectional and slot in bidirectional_slots:
            skipped_bidirectional += 1
            continue
        chosen = cand
        break
    if skipped_bidirectional:
        print(
            f"  skipped {skipped_bidirectional} bidirectional candidate(s) "
            "— same (class, param) slot fired in both 'up' and 'down' "
            "directions.  Use --widen-bounds for catalog bound updates "
            "(the recommended action), or pass "
            "--apply-include-bidirectional to override."
        )
    if chosen is None:
        if skipped_bidirectional:
            print("  (every visible candidate was skipped — nothing to apply)")
        else:
            print("  (no actionable candidates to apply)")
        return 0

    if chosen.op is not None:
        # Structural candidate — the "slot" is the (class, op) pair; there
        # is no param_name / proposed_codify_value to print.
        slot = f"{chosen.class_name} [{chosen.op}]"
        print(f"  selected: {slot} direction={chosen.direction}")
        target_spec_names = sorted({n for n in chosen.strategy_names if n})
        if target_spec_names:
            print(f"  target spec(s): {', '.join(target_spec_names)}")
    else:
        slot = f"{chosen.class_name}.{chosen.param_name}" if chosen.param_name else chosen.class_name
        proposed = chosen.proposed_codify_value()
        print(f"  selected: {slot} [{chosen.rule_kind}] direction={chosen.direction}")
        print(f"  proposed codify value: {proposed!r}")

    edits, modified_files = apply_codify_candidate(chosen, sources=sources, dry_run=dry_run)
    if not edits:
        if chosen.op is not None:
            print(
                "  (no source site needed editing — either every target "
                "spec already reflects the structural op, or the safety "
                "guards suppressed every match — e.g. dropping the last "
                "entry in a bucket)"
            )
        else:
            print(
                "  (no source site needed editing — every matching "
                "(class, param) literal already sits at-or-beyond the "
                "proposal in the candidate's direction)"
            )
        return 0
    print(f"  derived {len(edits)} edit(s):")
    for edit in edits:
        if edit.rule_kind == "structural":
            # Structural edits don't have a ``old_value → new_value``
            # story — they add / remove a whole tuple entry.  Print a
            # compact "op class in factory/spec" line instead.
            action_word = "drop" if edit.direction.startswith("drop_") else "add"
            print(
                f"    {edit.source_path}:{edit.lineno} "
                f"{edit.factory_name}/{edit.spec_name}: "
                f"{action_word} {edit.class_name}"
            )
        else:
            print(
                f"    {edit.source_path}:{edit.lineno} "
                f"{edit.factory_name}/{edit.spec_name}: "
                f"{edit.class_name}.{edit.param_name} = "
                f"{edit.old_source} -> {edit.new_source}"
            )
    action = "Would write" if dry_run else "Wrote"
    print(f"  {action} {len(modified_files)} file(s): {', '.join(sorted(modified_files))}")
    if dry_run:
        # --apply-format / --apply-run-tests are inert under dry-run: no
        # edits landed, so nothing to format or test.  Report the skip so
        # the operator knows the flags were parsed but no-oped.
        if run_format or run_tests:
            skipped = []
            if run_format:
                skipped.append("--apply-format")
            if run_tests:
                skipped.append("--apply-run-tests")
            print(f"  (inert under --apply-dry-run: {', '.join(skipped)} skipped)")
        # --open-pr still runs under dry-run: _open_pr_for_candidate prints
        # the git / gh command sequence it *would* execute without invoking
        # any subprocess (the hygiene flags above are the only dry-run no-op).
        if open_pr:
            return _open_pr_for_candidate(
                chosen,
                edits=edits,
                dry_run=dry_run,
                branch_prefix=pr_branch_prefix,
                base_branch=pr_base,
                gh_bin=pr_gh_bin,
                git_bin=pr_git_bin,
            )
        return 0

    files = sorted(modified_files)
    if run_format:
        cmd = ["uv", "run", "ruff", "format", *files]
        print(f"  Formatting: {' '.join(cmd)}")
        result = _run_subprocess(cmd)
        if result.returncode != 0:
            print(f"  ruff format failed (rc={result.returncode})")
            return int(result.returncode)
    if run_tests:
        cmd = ["uv", "run", "pytest", "tests/test_self_improve.py"]
        print(f"  Running tests: {' '.join(cmd)}")
        result = _run_subprocess(cmd)
        if result.returncode != 0:
            print(f"  pytest failed (rc={result.returncode})")
            return int(result.returncode)
        print("  Next: commit and open a draft PR with the codify evidence in the body.")
    else:
        print(
            "  Next: run the test suite (uv run pytest), then commit and "
            "open a draft PR with the codify evidence in the body."
        )
    if open_pr:
        return _open_pr_for_candidate(
            chosen,
            edits=edits,
            dry_run=dry_run,
            branch_prefix=pr_branch_prefix,
            base_branch=pr_base,
            gh_bin=pr_gh_bin,
            git_bin=pr_git_bin,
        )
    return 0


def _open_pr_for_candidate(
    chosen: Any,
    *,
    edits: Sequence[Any],
    dry_run: bool,
    branch_prefix: str,
    base_branch: str,
    gh_bin: str,
    git_bin: str,
    runner: Optional[Any] = None,
) -> int:
    """Open a draft codify PR for ``chosen`` via ``gh pr create``.

    The final layer of the V2 §9.5 step 4 codify pipeline (detection →
    value derivation → source edit → **PR**).  Called from
    :func:`_apply_top_codify_candidate` after the source edit has been
    applied to the working tree (or after ``--apply-dry-run`` printed
    the would-be diff).

    Flow:

    1. Dedup — run ``gh pr list --state open --json
       number,title,body,headRefName`` and skip if a PR carrying
       :func:`codify_pr_marker` for the same slot already exists.
       (§12.3 step 0 lesson: open PRs are the source of truth for
       in-flight work.)
    2. Branch — ``git checkout -b <branch_prefix>-<slot_key>-<direction>``.
    3. Commit — ``git add <edited files>`` +
       ``git commit -m <title>`` where title = :func:`codify_pr_title`.
    4. Push — ``git push -u origin <branch>`` (retried on network
       failures via the standard cron retry loop; the driver here
       shells out once and lets the caller handle retries).
    5. PR — ``gh pr create --draft --base <base_branch> --head <branch>
       --title <title> --body-file <tmpfile>`` where body =
       :func:`codify_pr_body`.  The marker embedded in the body's HTML
       comment is what step 1 matches against on the next run so a
       failed / re-run does not stack duplicates.

    ``runner`` is a subprocess launcher (defaults to
    :func:`subprocess.run` with ``check=False``, ``capture_output=True``,
    ``text=True``) — dependency-injected for the test suite so the
    hermetic tests never shell out.  A ``dry_run=True`` invocation
    prints every command the driver *would* run but does not invoke the
    launcher, mirroring the ``--apply-dry-run`` shape one layer up.

    Returns the process exit code: ``0`` on success (including the
    dedup / dry-run cases), non-zero on subprocess failure.
    """
    import json as _json
    import shutil
    import subprocess
    import tempfile

    from panobbgo.self_improve import (
        codify_pr_body,
        codify_pr_branch_name,
        codify_pr_marker,
        codify_pr_title,
        find_open_pr_for_slot,
    )

    def _default_runner(cmd, cwd=None):
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)

    run_cmd = _default_runner if runner is None else runner

    print()
    print("Open-PR:")
    title = codify_pr_title(chosen)
    branch = codify_pr_branch_name(chosen, prefix=branch_prefix)
    body = codify_pr_body(chosen, edits=edits, base_branch=base_branch)
    marker = codify_pr_marker(chosen)

    if dry_run:
        # Dry-run mode: print the commands the driver would run, then exit.
        # Every command is quoted with :func:`shlex.join` so an operator can
        # copy-paste it verbatim into a shell.
        import shlex

        dry_cmds = [
            [gh_bin, "pr", "list", "--state", "open", "--json", "number,title,body,headRefName"],
            [git_bin, "checkout", "-b", branch],
        ]
        # git add for each modified source file (deduplicated).
        edited_paths = sorted({e.source_path for e in edits})
        if edited_paths:
            dry_cmds.append([git_bin, "add", *edited_paths])
        dry_cmds.append([git_bin, "commit", "-m", title])
        dry_cmds.append([git_bin, "push", "-u", "origin", branch])
        dry_cmds.append(
            [
                gh_bin,
                "pr",
                "create",
                "--draft",
                "--base",
                base_branch,
                "--head",
                branch,
                "--title",
                title,
                "--body-file",
                "<pr_body.md>",
            ]
        )
        print(f"  slot: {marker}")
        print(f"  branch: {branch}")
        print(f"  title: {title}")
        print("  would run:")
        for cmd in dry_cmds:
            print(f"    {shlex.join(cmd)}")
        return 0

    # gh presence check: dedup step needs it.  If the binary is missing,
    # error clearly so the operator knows to install it rather than get a
    # cryptic FileNotFoundError later.  ``shutil.which`` also honours the
    # PATH the workflow runner sets up, matching the environment in which
    # the cron would actually invoke this driver.
    if shutil.which(gh_bin) is None:
        print(
            f"  ERROR: gh binary '{gh_bin}' not found on PATH; install "
            "https://cli.github.com/ or override with --pr-gh-bin."
        )
        return 4
    if shutil.which(git_bin) is None:
        print(f"  ERROR: git binary '{git_bin}' not found on PATH; install git or override with --pr-git-bin.")
        return 4

    # Dedup — one shot ``gh pr list``.
    print(f"  slot: {marker}")
    dedup_cmd = [gh_bin, "pr", "list", "--state", "open", "--json", "number,title,body,headRefName"]
    proc = run_cmd(dedup_cmd)
    if proc.returncode != 0:
        print(f"  ERROR: `{' '.join(dedup_cmd)}` failed (rc={proc.returncode}):")
        if proc.stderr:
            print(f"    stderr: {proc.stderr.strip()}")
        return proc.returncode or 5
    try:
        open_prs = _json.loads(proc.stdout or "[]")
    except _json.JSONDecodeError as exc:
        print(f"  ERROR: failed to parse `gh pr list` output as JSON: {exc}")
        return 5
    existing = find_open_pr_for_slot(chosen, open_prs)
    if existing is not None:
        print(
            f"  dedup: PR #{existing.get('number', '?')} "
            f"({existing.get('headRefName', '<unknown branch>')}) "
            "already covers this slot — skipping.  "
            "Close / merge the existing PR before re-running."
        )
        return 0

    # Branch + commit + push + create PR.
    edited_paths = sorted({e.source_path for e in edits})

    def _run(cmd: List[str], step: str) -> Optional[int]:
        print(f"  $ {' '.join(cmd)}")
        rc_proc = run_cmd(cmd)
        if rc_proc.returncode != 0:
            print(f"  ERROR: {step} failed (rc={rc_proc.returncode}):")
            if getattr(rc_proc, "stderr", None):
                print(f"    stderr: {rc_proc.stderr.strip()}")
            return rc_proc.returncode or 5
        return None

    steps: List[Tuple[List[str], str]] = [
        ([git_bin, "checkout", "-b", branch], "git checkout -b"),
    ]
    if edited_paths:
        steps.append(([git_bin, "add", *edited_paths], "git add"))
    steps.extend(
        [
            ([git_bin, "commit", "-m", title], "git commit"),
            ([git_bin, "push", "-u", "origin", branch], "git push"),
        ]
    )
    for cmd, step in steps:
        rc = _run(cmd, step)
        if rc is not None:
            return rc

    # Write the body to a temp file so ``gh pr create --body-file`` sees
    # a clean argument (bash arg length limits are not a concern in
    # practice, but a temp file avoids quoting hazards regardless).
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        body_path = f.name
        f.write(body)
    try:
        cmd = [
            gh_bin,
            "pr",
            "create",
            "--draft",
            "--base",
            base_branch,
            "--head",
            branch,
            "--title",
            title,
            "--body-file",
            body_path,
        ]
        rc = _run(cmd, "gh pr create")
        if rc is not None:
            return rc
    finally:
        try:
            pathlib.Path(body_path).unlink()
        except OSError:
            pass

    print(f"  opened draft PR for slot {marker} on branch {branch}.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
