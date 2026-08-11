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
Self-Improvement Loop Driver (Phase 5 MVP)
==========================================

Phase 5 of :doc:`../planning/SELF_IMPROVEMENT_LOOP.md`.  A small autonomous
driver that closes the measure → propose → apply → measure → accept/revert
cycle on top of the existing harness machinery.

The driver operates entirely **in-memory**: mutations perturb a copy of
the current :class:`~panobbgo.benchmark.StrategySpec` list; no source files
are rewritten and no git commits happen.  That is the MVP scope — it covers
the hyperparameter-retune category from §7 item 1 of the plan.  Source-level
mutations (heuristic code edits, new heuristics) remain future work.

Workflow
--------

.. code-block:: text

    for iter in 0 .. N-1:
        1. sample a mutation from the catalog                 (rng_mutation)
        2. apply it to the current StrategySpec list          (apply_mutation)
        3. run the randomized harness with the current list   (baseline)
        4. run the randomized harness with the mutated list   (candidate)
        5. decide via statistical_accept(before, after)       (bootstrap CI)
        6. if accept: adopt the mutated list; else: keep current
        7. append one JSONL line to the ledger

Reproducibility
---------------

Within one iteration, the baseline and the candidate are measured against
**the same sampled problem instances** via ``HarnessConfig.randomize_iteration
= iteration``.  This makes before/after apples-to-apples at the same compute
cost, and it plugs directly into the Phase 3 randomized battery
(:mod:`panobbgo.harness_randomized`).  Across iterations the instances
intentionally differ — so persistent improvements cannot be explained by
instance cherry-picking.

The mutation sampler uses its own seeded RNG
(``LoopConfig.mutation_seed``) that is independent of the harness seeds.
A loop of ``N`` iterations is fully replayable from
``(base_seed, mutation_seed, stat_seed)``.

Adaptive mutation sampler (§10)
-------------------------------

By default the loop draws mutations uniformly from the applicable rules
of the catalog.  When ``LoopConfig.adaptive_sampling = True``, the loop
substitutes :class:`AdaptiveMutationSampler` — a Thompson-sampling
bandit over per-rule Beta posteriors that biases future iterations
toward rules with positive accept history while still exploring
under-tried rules.  Cold-start (no history, default symmetric prior) is
statistically identical to uniform sampling, so flipping the flag is
safe on a fresh ledger.  Set
``LoopConfig.adaptive_prime_from_ledger = True`` to seed the bandit's
history from a prior JSONL ledger when resuming a long run.

Anti-cherry-pick guard (§6.3)
-----------------------------

Even with a randomized battery, a sequence of "lucky" instance draws can
inflate per-iteration ``after`` scores enough to clear the bootstrap CI.
The guard mitigates this by periodically re-measuring the **top of the
accepted ladder** on a *fresh* iteration seed (``iteration +
guard_iteration_offset``).  If the re-measured composite drops more than
``guard_eps_ladder`` below the score that originally got it accepted, the
loop pops that ladder entry and retries with the previous one — until a
stable entry is found or the seed strategies are reached.  Set
``LoopConfig.guard_interval`` to a positive integer (typically ``5`` or
``10``) to enable; ``0`` (default) disables the guard so existing
configurations behave identically.

Hold-out validation set
-----------------------

The guard catches drift inside the *training* base-seed family — it uses
the same :attr:`HarnessConfig.seed` and only varies
:attr:`HarnessConfig.randomize_iteration`.  A mutation that overfits to
peculiarities of the training base-seed family will *not* be caught: the
guard's "fresh" instances are drawn from the same SHA-256 stream.

The hold-out validation closes that gap.  At the **end** of the loop run
(once :attr:`LoopConfig.iterations` are exhausted), if at least one
hold-out base seed is configured (either the scalar
:attr:`LoopConfig.holdout_base_seed` or the list-typed
:attr:`LoopConfig.holdout_base_seeds`) and
:attr:`LoopConfig.holdout_iterations` is positive, the loop re-measures
both the **seed** ladder entry and the **final top** entry on each
completely independent ``base_seed`` family.  Per seed, the two scores
are averaged over ``holdout_iterations`` distinct instances and compared
to the training-time scores recorded on the ladder.  If the
"top minus seed" gap on the hold-out is smaller than the training gap
by more than :attr:`LoopConfig.holdout_eps_overfit`, the loop emits an
``overfit=True`` :class:`LoopHoldoutRecord` and exits non-zero from the
CLI when ``--fail-on-overfit`` is set.  Otherwise the record is written
informationally so an auditor can inspect the generalisation drift.

When multiple seeds are configured via :attr:`LoopConfig.holdout_base_seeds`,
one :class:`LoopHoldoutRecord` is written per seed and the CLI aggregates
them into a single verdict line — ``overfit`` if **any** seed flagged
overfit, and the **worst** (most negative) drift across seeds.  This
makes the hold-out a more robust drift estimator: a single hold-out seed
reduces the entire generalisation question to one independent draw, but
several seeds catch overfits that escape any one draw while remaining
cheap relative to the loop's training cost.

Hold-out is one-shot at the end of the loop and uses fresh randomized
instances, so its compute cost is fixed per seed:
``2 × holdout_iterations`` harness runs per seed — typically a small
fraction of the loop's total budget.  Disabled by default
(``holdout_base_seed = 0`` and ``holdout_base_seeds = ()``) so existing
configurations behave identically.

Safety rails (§8 in the plan)
-----------------------------

* **In-memory only** — no git state is touched.  Accept = "promote the
  mutated spec list for the next iteration"; reject = "discard it".
* **STOP sentinel** — if ``LoopConfig.stop_sentinel_path`` exists, the
  loop halts *before* the next iteration and returns the ledger.  This
  is the human escape hatch.
* **Ledger** — append-only JSONL at ``LoopConfig.ledger_path``; each line
  is one fully-specified iteration record (proposal, decision, scores,
  CIs, reasons, seeds).  Humans can audit or replay.
* **Bounded perturbations** — every mutation rule declares ``bounds`` so
  the search cannot run away (e.g., ``Nearby.radius`` stays in
  ``[0.005, 0.5]``).
* **Anti-cherry-pick** — the periodic guard described above catches
  drifts caused by accidental instance cherry-picking.

See also
--------

* :mod:`panobbgo.harness` — :class:`BenchmarkHarness`,
  :func:`statistical_accept`.
* :mod:`panobbgo.harness_randomized` — parametric problem battery
  (Phase 3).
* ``planning/SELF_IMPROVEMENT_LOOP.md`` — full design document.
"""

from __future__ import annotations

import ast
import bisect
import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from panobbgo.benchmark import StrategySpec

if TYPE_CHECKING:
    from panobbgo.harness_randomized import ProblemFamily
from panobbgo.harness import (
    BenchmarkHarness,
    HarnessConfig,
    HarnessResult,
    ProblemStrategyResult,
    statistical_accept,
)


# ---------------------------------------------------------------------------
# Mutation catalog
# ---------------------------------------------------------------------------


@dataclass
class MutationRule:
    """Describes one kind of mutation the catalog may sample.

    Args:
        strategy_pattern: Substring matched against
            :attr:`StrategySpec.name`.  Empty string matches every
            strategy — useful for a knob that should apply wherever the
            target class appears.
        class_name: ``__name__`` of a heuristic or analyzer class carrying
            the target kwarg (e.g., ``"Nearby"``, ``"CMAES"``).
        param_name: Keyword argument on the target class (e.g.,
            ``"radius"``, ``"sigma0"``).
        kind: One of
            * ``"log_uniform_perturb"`` — multiplicative perturbation,
              ``new = old * 10^u`` where ``u ~ U[-log_step, +log_step]``.
              Requires a positive ``old``.
            * ``"integer_add"`` — additive integer step from
              :attr:`delta_choices`.  Result is clamped to
              ``int(bounds[0]) .. int(bounds[1])``.
            * ``"float_uniform"`` — uniform sample from
              ``[low, high]``, then clamped to :attr:`bounds`.
            * ``"categorical_choice"`` — discrete choice from
              :attr:`choices`.  Always proposes a *different* value from
              the current one; the rule therefore never produces a
              no-op mutation on its own.  ``bounds`` is ignored.
        bounds: Inclusive ``(low, high)`` clamp applied after sampling.
            Always interpreted as floats for ``log_uniform_perturb`` and
            ``float_uniform``; as ints for ``integer_add``.  Ignored for
            ``categorical_choice`` — pass any placeholder (the default
            ``(0.0, 0.0)`` is fine).
        log_step: Half-width of the log-uniform perturbation (decades).
            Default ``0.15`` ≈ ±41 %.
        delta_choices: Integer deltas for ``integer_add``.
        low: Lower bound for ``float_uniform``.
        high: Upper bound for ``float_uniform``.
        choices: Discrete options for ``categorical_choice``.  Must
            contain at least two entries; any hashable Python value works
            (strings, bools, numerics — anything JSON-serialisable
            without a custom encoder).  Ignored for the numeric kinds.
        probability: Relative weight used when the catalog picks among
            multiple applicable rules; normalised automatically.
    """

    strategy_pattern: str
    class_name: str
    param_name: str
    kind: str
    bounds: Tuple[float, float] = (0.0, 0.0)
    log_step: float = 0.15
    delta_choices: Tuple[int, ...] = (-1, 1)
    low: float = 0.0
    high: float = 1.0
    choices: Tuple[Any, ...] = ()
    probability: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {
            "log_uniform_perturb",
            "integer_add",
            "float_uniform",
            "categorical_choice",
        }:
            raise ValueError(f"Unknown mutation kind: {self.kind!r}")
        if self.kind == "categorical_choice":
            if len(self.choices) < 2:
                raise ValueError(f"categorical_choice requires at least 2 distinct choices, got {len(self.choices)}")
            if len(set(self.choices)) != len(self.choices):
                raise ValueError(f"categorical_choice: duplicate entries in choices={self.choices!r}")
        else:
            lo, hi = self.bounds
            if not lo <= hi:
                raise ValueError(f"bounds not ordered: {self.bounds}")
        if self.probability <= 0:
            raise ValueError(f"probability must be > 0, got {self.probability}")

    def rule_key(self) -> "RuleKey":
        """Return the bandit-arm key for this rule.

        Centralising this on the rule object lets the catalog and the
        adaptive sampler treat :class:`MutationRule` and
        :class:`StructuralMutationRule` uniformly without knowing each
        other's field layout.
        """
        return (self.class_name, self.param_name, self.kind)


# Op values accepted by :class:`StructuralMutationRule`.  Kept as a tuple
# rather than a Literal for runtime introspection (``op in _STRUCTURAL_OPS``).
# Heuristic ops shipped 2026-05-03; analyzer ops shipped 2026-06-02 — the
# four cover the §7.2 portfolio-composition mutation class fully.
_STRUCTURAL_HEURISTIC_OPS: Tuple[str, ...] = ("add_heuristic", "drop_heuristic")
_STRUCTURAL_ANALYZER_OPS: Tuple[str, ...] = ("add_analyzer", "drop_analyzer")
_STRUCTURAL_OPS: Tuple[str, ...] = _STRUCTURAL_HEURISTIC_OPS + _STRUCTURAL_ANALYZER_OPS


def _is_analyzer_op(op: Optional[str]) -> bool:
    """True for analyzer-bucket structural ops; False otherwise (heuristic or kwarg)."""
    return op in _STRUCTURAL_ANALYZER_OPS


@dataclass
class StructuralMutationRule:
    """Add or drop a heuristic or analyzer from a strategy's portfolio.

    Implements the *Strategy portfolio composition* mutation class from
    §7.2 of ``planning/SELF_IMPROVEMENT_LOOP.md``.  Where
    :class:`MutationRule` only retunes existing kwargs, this rule changes
    the *shape* of a :class:`StrategySpec`'s ``heuristics`` or
    ``analyzers`` list — the loop can therefore discover whether
    dropping an existing entry, or adding a fresh one, generalises
    better than the seed composition.

    Heuristic ops (``add_heuristic`` / ``drop_heuristic``) shipped
    2026-05-03; analyzer ops (``add_analyzer`` / ``drop_analyzer``)
    shipped 2026-06-02 and mirror the heuristic semantics — same
    ``candidate_classes`` / ``droppable_classes`` / ``avoid_duplicates``
    fields, the analyzer-specific safety floor lives on the sibling
    :attr:`min_analyzers`.

    Args:
        strategy_pattern: Substring matched against
            :attr:`StrategySpec.name`; empty string matches every
            strategy.  Same semantics as :class:`MutationRule`.
        op: One of

            * ``"add_heuristic"`` — append a heuristic from
              :attr:`candidate_classes` to a target strategy.  When
              :attr:`avoid_duplicates` is ``True`` (default) candidate
              classes already present in the strategy are skipped, which
              keeps the catalog from cluttering a portfolio with
              redundant copies.
            * ``"drop_heuristic"`` — remove an existing heuristic from a
              target strategy.  The :attr:`min_heuristics` safety guard
              forbids dropping below this many heuristics so the strategy
              always has *something* to emit points.  When
              :attr:`droppable_classes` is non-empty, only heuristics
              whose ``__name__`` is in the set are eligible.
            * ``"add_analyzer"`` — append an analyzer from
              :attr:`candidate_classes` to a target strategy.  Same
              duplicate-avoidance semantics as ``add_heuristic``.
              Useful for letting the loop discover whether adding
              ``Restart`` (warm restarts) or ``Sensitivity`` (adaptive
              tracking) helps a given seed composition.
            * ``"drop_analyzer"`` — remove an existing analyzer from a
              target strategy.  Subject to the :attr:`min_analyzers`
              floor (default ``0`` — analyzers are non-essential, unlike
              heuristics, so an empty analyzer list is a valid spec).
        candidate_classes: Sequence of ``(Class, default_kwargs)``
            pairs the rule may pull from for ``add_heuristic`` /
            ``add_analyzer``.  Ignored for drop ops.  Each tuple's
            ``default_kwargs`` is shallow-copied into the new spec so
            subsequent kwarg-tune mutations can perturb it independently.
        droppable_classes: Optional restriction for ``drop_heuristic`` /
            ``drop_analyzer``.  When provided (a tuple of class
            ``__name__``s), only matching entries may be dropped.  Empty
            tuple means "any entry in the strategy is eligible (subject
            to :attr:`min_heuristics` / :attr:`min_analyzers`)".
        min_heuristics: Lower bound on the size of the heuristics list
            after a drop.  Default ``2`` keeps every strategy with at
            least one diversity slot beyond the bare minimum.  ``1`` is
            the absolute floor; ``0`` is rejected.  Only consulted for
            ``drop_heuristic`` ops.
        min_analyzers: Lower bound on the size of the analyzers list
            after a drop.  Default ``0`` because analyzers are
            non-essential — most strategies in the default battery do
            ship one (typically :class:`Sensitivity`) but stripping it
            yields a valid, runnable spec.  Set to ``1`` if the rule
            should preserve at least one analyzer.  Only consulted for
            ``drop_analyzer`` ops.
        avoid_duplicates: For ``add_heuristic`` / ``add_analyzer``, skip
            candidates whose class is already present in the matching
            bucket.  Default ``True``.  Set to ``False`` when intentional
            duplicates are desirable.
        probability: Relative weight when the catalog picks among
            multiple applicable rules; normalised automatically.  Same
            semantics as :class:`MutationRule`.

    Raises:
        ValueError: If ``op`` is not one of :data:`_STRUCTURAL_OPS`,
            ``min_heuristics`` is below ``1``, ``min_analyzers`` is
            below ``0``, ``probability`` is non-positive, or an add op
            is paired with an empty :attr:`candidate_classes`.
    """

    strategy_pattern: str
    op: str
    candidate_classes: Tuple[Tuple[type, Dict[str, Any]], ...] = ()
    droppable_classes: Tuple[str, ...] = ()
    min_heuristics: int = 2
    min_analyzers: int = 0
    avoid_duplicates: bool = True
    probability: float = 1.0

    def __post_init__(self) -> None:
        if self.op not in _STRUCTURAL_OPS:
            raise ValueError(f"Unknown structural op: {self.op!r}; expected one of {_STRUCTURAL_OPS}")
        if self.min_heuristics < 1:
            raise ValueError(f"min_heuristics must be >= 1, got {self.min_heuristics}")
        if self.min_analyzers < 0:
            raise ValueError(f"min_analyzers must be >= 0, got {self.min_analyzers}")
        if self.probability <= 0:
            raise ValueError(f"probability must be > 0, got {self.probability}")
        if self.op in ("add_heuristic", "add_analyzer") and not self.candidate_classes:
            raise ValueError(f"{self.op} requires at least one entry in candidate_classes")

    def rule_key(self) -> "RuleKey":
        """Return the bandit-arm key for this rule.

        All structural rules with the same ``op`` share one arm by
        default — this keeps the bandit space small and matches the
        coarsest reasonable taxonomy ("does adding heuristics help?",
        "does dropping help?", and the symmetric questions about
        analyzers).  Per-class arms split each op into one arm per
        candidate class — see
        :attr:`AdaptiveMutationSampler.per_class_structural`.
        """
        return ("*", self.op, "structural")


@dataclass
class MutationProposal:
    """A concrete mutation produced by :meth:`MutationCatalog.sample`.

    Two flavours share this type:

    * **Hyperparameter retune** — the canonical case.  ``op`` is ``None``;
      ``class_name`` / ``param_name`` identify the kwarg slot;
      ``old_value`` / ``new_value`` are scalar values.

    * **Structural mutation** (:class:`StructuralMutationRule`) — the
      candidate adds or drops a heuristic from a strategy's portfolio.
      ``op`` is ``"add_heuristic"`` or ``"drop_heuristic"``; ``class_name``
      identifies the heuristic class added or dropped; ``param_name`` is
      the empty string; ``structural_kwargs`` carries the heuristic's
      kwargs (the kwargs about to be added, or the kwargs that were on the
      dropped heuristic).  ``old_value`` and ``new_value`` are unused for
      structural ops and serialised as ``None``.

    The proposal is the universal currency the ledger and
    :func:`apply_mutation` consume — both flavours round-trip through
    :meth:`to_dict` so a JSONL ledger can be replayed losslessly.
    """

    strategy_name: str
    class_name: str
    param_name: str
    old_value: Any
    new_value: Any
    rule_kind: str
    rationale: str
    op: Optional[str] = None
    structural_kwargs: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "strategy_name": self.strategy_name,
            "class_name": self.class_name,
            "param_name": self.param_name,
            "old_value": _to_plain(self.old_value),
            "new_value": _to_plain(self.new_value),
            "rule_kind": self.rule_kind,
            "rationale": self.rationale,
        }
        if self.op is not None:
            d["op"] = self.op
            d["structural_kwargs"] = (
                {k: _to_plain(v) for k, v in self.structural_kwargs.items()}
                if self.structural_kwargs is not None
                else None
            )
        return d


def _to_plain(val: Any) -> Any:
    """Best-effort coerce numpy scalars / arrays to JSON-friendly types."""
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


#: Mutation kinds whose value arithmetic requires a real number — used by
#: :func:`_find_targets` to skip string sentinels (e.g. ``NP_init="auto"``)
#: that a categorical rule could legitimately carry but a numeric rule cannot
#: perturb without a ``TypeError`` / ``int("auto")`` crash.  (A frozenset for
#: O(1) membership; the codify-scan layer keeps its own ordered tuple of the
#: same kinds under the name ``_NUMERIC_RULE_KINDS`` further down the module.)
_NUMERIC_MUTATION_KINDS: frozenset = frozenset({"integer_add", "float_uniform", "log_uniform_perturb"})


def _is_numeric_value(value: Any) -> bool:
    """True for real int / float values (``bool`` excluded — it is an int subclass)."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _find_targets(
    specs: Sequence[StrategySpec],
    strategy_pattern: str,
    class_name: str,
    param_name: str,
    rule_kind: Optional[str] = None,
) -> List[Tuple[int, str, int, Any]]:
    """Locate every ``(spec, bucket, entry, current_value)`` that matches.

    A hit is produced iff the spec name contains ``strategy_pattern`` (or
    the pattern is empty), the heuristic / analyzer class name equals
    ``class_name``, *and* ``param_name`` is already present in the kwargs
    dict **with a non-``None`` value**.  Locations where the class matches
    but the kwarg is missing are intentionally skipped — we only tune
    existing parameters.

    Kwargs explicitly set to ``None`` are also skipped: ``None`` is the
    sentinel a number of heuristics use to mean "use the heuristic-internal
    auto-default" (e.g. :class:`~panobbgo.analyzers.restart.Restart`
    resolves ``patience=None`` to ``5 * dim`` at ``__start__`` time,
    :class:`~panobbgo.heuristics.lbfgsb.LBFGSB` resolves ``max_starts=None``
    to "unlimited until budget").  Numeric mutation kinds (``integer_add``
    / ``float_uniform`` / ``log_uniform_perturb``) cannot meaningfully
    perturb the ``None`` sentinel, and categorical rules typically do not
    list ``None`` as a value either — skipping at ``_find_targets`` keeps
    the predicate uniform across kinds and lets the catalog include
    ``patience``- / ``max_starts``-style rules without crashing on specs
    that opted into the auto-default sentinel.

    When ``rule_kind`` names a numeric mutation kind (``integer_add`` /
    ``float_uniform`` / ``log_uniform_perturb``), *non-numeric* values are
    skipped too: a heuristic may carry a string sentinel such as
    ``NP_init="auto"`` (budget-adaptive DE sizing) which a numeric rule
    cannot perturb — matching it would crash ``int("auto")`` in
    :meth:`MutationRule.apply`.  Categorical rules (``rule_kind=None`` or
    ``"categorical_choice"``) still see string values so e.g. a
    ``F_schedule`` regime flip keeps working.
    """
    numeric_only = rule_kind in _NUMERIC_MUTATION_KINDS
    hits: List[Tuple[int, str, int, Any]] = []
    for si, spec in enumerate(specs):
        if strategy_pattern and strategy_pattern not in spec.name:
            continue
        for bucket_name, entries in (
            ("heuristics", spec.heuristics),
            ("analyzers", spec.analyzers),
        ):
            for ei, (cls, kwargs) in enumerate(entries):
                if cls.__name__ != class_name:
                    continue
                if param_name not in kwargs:
                    continue
                value = kwargs[param_name]
                if value is None:
                    continue
                if numeric_only and not _is_numeric_value(value):
                    continue
                hits.append((si, bucket_name, ei, value))
    return hits


# Shape of one structural hit: (spec_index, candidate_class, candidate_kwargs).
# For ``add_heuristic`` ``candidate_class`` is the class about to be added and
# ``candidate_kwargs`` are its default kwargs.  For ``drop_heuristic`` the
# triple is (spec_index, dropped_class, dropped_kwargs) and the heuristic is
# located by class name (the strategy carries at most one entry per class
# under :attr:`StructuralMutationRule.avoid_duplicates = True`, which the
# default — duplicates are allowed but a drop targets *one* of them).
_StructuralHit = Tuple[int, type, Dict[str, Any]]


def _find_structural_hits(
    specs: Sequence[StrategySpec],
    rule: "StructuralMutationRule",
) -> List[_StructuralHit]:
    """Enumerate every (spec, candidate) site to which ``rule`` may be applied.

    The set differs by ``op``:

    * ``add_heuristic`` / ``add_analyzer`` — Cartesian product of
      matching strategies and :attr:`candidate_classes`, optionally
      pruned by :attr:`avoid_duplicates`.  Each candidate's
      ``default_kwargs`` is shallow-copied so two hits never share a
      mutable dict.  The bucket (``heuristics`` vs ``analyzers``)
      determines which list ``avoid_duplicates`` compares against.
    * ``drop_heuristic`` / ``drop_analyzer`` — every existing
      ``(spec, entry)`` pair whose strategy matches
      :attr:`strategy_pattern`, the entry's class is in
      :attr:`droppable_classes` (or any, if empty), and the strategy
      currently has more than :attr:`min_heuristics` /
      :attr:`min_analyzers` entries in the matching bucket so removal
      would not violate the safety floor.

    Returning an empty list signals "rule not applicable to current
    specs"; the catalog uses that to skip the rule entirely (and so the
    bandit does not waste an iteration on a no-op).
    """
    hits: List[_StructuralHit] = []
    analyzer_op = _is_analyzer_op(rule.op)
    floor = rule.min_analyzers if analyzer_op else rule.min_heuristics
    if rule.op in ("add_heuristic", "add_analyzer"):
        for si, spec in enumerate(specs):
            if rule.strategy_pattern and rule.strategy_pattern not in spec.name:
                continue
            bucket = spec.analyzers if analyzer_op else spec.heuristics
            present = {cls.__name__ for cls, _ in bucket}
            for cls, default_kwargs in rule.candidate_classes:
                if rule.avoid_duplicates and cls.__name__ in present:
                    continue
                hits.append((si, cls, dict(default_kwargs)))
    elif rule.op in ("drop_heuristic", "drop_analyzer"):
        droppable = set(rule.droppable_classes)
        for si, spec in enumerate(specs):
            if rule.strategy_pattern and rule.strategy_pattern not in spec.name:
                continue
            bucket = spec.analyzers if analyzer_op else spec.heuristics
            if len(bucket) <= floor:
                # Dropping any one would breach the safety floor.
                continue
            for cls, kwargs in bucket:
                if droppable and cls.__name__ not in droppable:
                    continue
                hits.append((si, cls, dict(kwargs)))
    return hits


# A catalog entry is either a kwarg perturbation rule or a structural
# rule.  Both expose ``rule_key()`` and ``probability``; the catalog
# branches on type when it needs the kind-specific machinery.
CatalogRule = Any  # Union[MutationRule, StructuralMutationRule] — kept loose for static checkers


class MutationCatalog:
    """A weighted pool of :class:`MutationRule` and :class:`StructuralMutationRule` instances.

    :meth:`sample` returns one applicable :class:`MutationProposal`, or
    ``None`` when no rule can be applied to the input spec list.  An
    "applicable" rule is one whose target class + kwarg exists somewhere
    in the input specs (kwarg rules), or whose op has at least one valid
    site under the current portfolio shape (structural rules).
    """

    def __init__(self, rules: Sequence[CatalogRule]) -> None:
        if not rules:
            raise ValueError("MutationCatalog requires at least one rule")
        self.rules: List[CatalogRule] = list(rules)

    def applicable_rules(self, specs: Sequence[StrategySpec]) -> List[Tuple[CatalogRule, List[Any]]]:
        """Return ``[(rule, hits), …]`` for rules with ≥1 target in ``specs``.

        The shape of each ``hits`` entry depends on the rule type:

        * :class:`MutationRule` → ``(spec_index, bucket, entry_index, current_value)``
          (the existing layout, unchanged)
        * :class:`StructuralMutationRule` → ``(spec_index, class, kwargs_dict)``
          (the new layout — see :func:`_find_structural_hits`)

        Callers must dispatch on ``isinstance(rule, ...)`` before
        unpacking — :meth:`sample` does so internally.
        """
        out: List[Tuple[CatalogRule, List[Any]]] = []
        for rule in self.rules:
            if isinstance(rule, StructuralMutationRule):
                s_hits = _find_structural_hits(specs, rule)
                if s_hits:
                    out.append((rule, list(s_hits)))
            else:
                k_hits = _find_targets(
                    specs, rule.strategy_pattern, rule.class_name, rule.param_name, rule_kind=rule.kind
                )
                if k_hits:
                    out.append((rule, list(k_hits)))
        return out

    def sample(
        self,
        rng: np.random.Generator,
        specs: Sequence[StrategySpec],
    ) -> Optional[MutationProposal]:
        """Draw one applicable mutation.

        Returns ``None`` iff no rule matches ``specs`` — callers should
        treat this as "nothing to do this iteration".
        """
        applicable = self.applicable_rules(specs)
        if not applicable:
            return None

        weights = np.array([rule.probability for rule, _ in applicable], dtype=np.float64)
        weights = weights / weights.sum()
        chosen_idx = int(rng.choice(len(applicable), p=weights))
        rule, hits = applicable[chosen_idx]

        hit_idx = int(rng.integers(0, len(hits)))

        if isinstance(rule, StructuralMutationRule):
            return _make_structural_proposal(rule, hits[hit_idx], specs)

        # Kwarg perturbation.
        si, _, _, old_value = hits[hit_idx]
        strategy_name = specs[si].name
        new_value = self._mutate_value(rule, old_value, rng)
        rationale = (
            f"{rule.kind} on {rule.class_name}.{rule.param_name} in {strategy_name}: {old_value!r} -> {new_value!r}"
        )
        return MutationProposal(
            strategy_name=strategy_name,
            class_name=rule.class_name,
            param_name=rule.param_name,
            old_value=_to_plain(old_value),
            new_value=_to_plain(new_value),
            rule_kind=rule.kind,
            rationale=rationale,
        )

    @staticmethod
    def _mutate_value(rule: MutationRule, old: Any, rng: np.random.Generator) -> Any:
        lo, hi = rule.bounds
        if rule.kind == "log_uniform_perturb":
            if float(old) <= 0.0:
                raise ValueError(f"log_uniform_perturb requires positive value, got {old!r}")
            exponent = float(rng.uniform(-rule.log_step, rule.log_step))
            candidate = float(old) * (10.0**exponent)
            return float(min(hi, max(lo, candidate)))
        if rule.kind == "integer_add":
            delta = int(rng.choice(np.asarray(rule.delta_choices)))
            candidate_int = int(old) + delta
            return int(min(int(hi), max(int(lo), candidate_int)))
        if rule.kind == "float_uniform":
            candidate_f = float(rng.uniform(rule.low, rule.high))
            return float(min(hi, max(lo, candidate_f)))
        if rule.kind == "categorical_choice":
            # Always propose a value different from ``old`` so the
            # mutation does something observable.  When ``old`` is not
            # in ``choices`` (spec drift) every entry is a valid
            # candidate.  ``MutationRule.__post_init__`` guarantees
            # ``len(choices) >= 2`` so the alternative pool is never
            # empty in the well-formed case.
            alternatives = [c for c in rule.choices if c != old]
            if not alternatives:
                alternatives = list(rule.choices)
            idx = int(rng.integers(0, len(alternatives)))
            return alternatives[idx]
        # Unreachable — validated in MutationRule.__post_init__
        raise ValueError(f"Unknown mutation kind: {rule.kind!r}")


def _make_structural_proposal(
    rule: StructuralMutationRule,
    hit: _StructuralHit,
    specs: Sequence[StrategySpec],
) -> MutationProposal:
    """Convert one structural hit into a :class:`MutationProposal`.

    The ``op`` and ``structural_kwargs`` fields carry the per-op
    information :func:`apply_mutation` needs; the legacy
    ``class_name`` / ``rule_kind`` fields stay populated so existing
    ledger consumers (and the bandit's :func:`_proposal_rule_key`)
    continue to work without special-casing.  Heuristic and analyzer
    ops share the same proposal shape — they differ only in which
    bucket :func:`apply_mutation` mutates.
    """
    si, cls, kwargs = hit
    strategy_name = specs[si].name
    if rule.op in ("add_heuristic", "add_analyzer"):
        rationale = f"{rule.op} {cls.__name__}({kwargs!r}) to {strategy_name}"
    else:
        # drop_heuristic / drop_analyzer
        rationale = f"{rule.op} {cls.__name__}({kwargs!r}) from {strategy_name}"
    return MutationProposal(
        strategy_name=strategy_name,
        class_name=cls.__name__,
        param_name="",
        old_value=None,
        new_value=None,
        rule_kind=rule.op,
        rationale=rationale,
        op=rule.op,
        structural_kwargs=dict(kwargs),
    )


# ---------------------------------------------------------------------------
# Adaptive (Thompson-sampling) mutation sampler
# ---------------------------------------------------------------------------


# Identifier used to bucket accept/attempt history.  Keeping it a small tuple
# of native strings keeps stats reproducible across processes and trivially
# JSON-serialisable.  ``(class_name, param_name, rule_kind)`` is what every
# ledger record exposes via ``MutationProposal.to_dict()``, so the sampler's
# history can be replayed from a prior run without ambiguity.
RuleKey = Tuple[str, str, str]


@dataclass
class MutationRuleStats:
    """Per-rule accept/attempt history maintained by :class:`AdaptiveMutationSampler`.

    Attributes:
        rule_key: ``(class_name, param_name, rule_kind)`` identifying the
            mutation type.  Two :class:`MutationRule` instances that
            differ only in ``strategy_pattern`` *share* one stats bucket
            — they are conceptually the same dial and the bandit treats
            them as one arm.
        n_attempts: Number of times :meth:`AdaptiveMutationSampler.sample`
            picked a rule with this key *and* the iteration produced a
            decision (accept or reject).  Skip records do not count.
        n_accepts: Number of those attempts that the loop accepted.
        reward_sum: Sum of per-iteration bandit rewards in ``[0, 1]``.
            Under the historical binary-reward path (:meth:`record_outcome`
            called without an explicit ``reward``) the reward is
            ``1.0`` per accept / ``0.0`` per reject, so ``reward_sum`` is
            byte-identical to :attr:`n_accepts` and the Thompson
            posterior is unchanged.  Under the graded-reward path
            (``LoopConfig.bandit_reward_shaping = "graded"`` — §7.4 of
            ``planning/SELF_IMPROVEMENT_LOOP.md``) each iteration
            contributes a continuous reward derived from the bootstrap
            CI / point delta, so a barely-confirmed accept and a
            clearly-winning accept can be distinguished.  The Thompson
            posterior consumes ``reward_sum`` directly — see
            :meth:`AdaptiveMutationSampler.sample`.
    """

    rule_key: RuleKey
    n_attempts: int = 0
    n_accepts: int = 0
    reward_sum: float = 0.0

    def __post_init__(self) -> None:
        # Backwards compat for direct construction (tests, hand-built
        # priming fixtures, and pre-2026-06-13 callers that only knew
        # about the binary ``n_accepts`` counter): when ``reward_sum``
        # is its default and ``n_accepts`` is non-zero, mirror
        # ``n_accepts`` into ``reward_sum`` so the Thompson posterior is
        # byte-identical to the historical Beta(α₀ + n_accepts, …)
        # parameterisation.  Graded-reward callers always populate
        # ``reward_sum`` via :meth:`record_outcome` so this branch only
        # fires on the binary path.
        if self.reward_sum == 0.0 and self.n_accepts > 0:
            self.reward_sum = float(self.n_accepts)

    @property
    def accept_rate(self) -> float:
        """Empirical accept rate, or 0.0 with no attempts."""
        if self.n_attempts == 0:
            return 0.0
        return self.n_accepts / self.n_attempts

    @property
    def mean_reward(self) -> float:
        """Mean bandit reward across attempts (graded scale), or 0.0 with no attempts.

        Under the binary-reward path this equals :attr:`accept_rate`
        (every accept contributes ``1.0``, every reject ``0.0``).  Under
        the graded-reward path of §7.4 this carries strictly more signal
        than :attr:`accept_rate`: an arm with many barely-rejected
        proposals (Δ ≈ 0) ends up at ``mean_reward ≈ 0.5`` even when
        ``accept_rate == 0``, distinguishing it from an arm that
        consistently produces clearly-harmful proposals
        (``mean_reward ≈ 0``).
        """
        if self.n_attempts == 0:
            return 0.0
        return float(self.reward_sum) / float(self.n_attempts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.rule_key[0],
            "param_name": self.rule_key[1],
            "rule_kind": self.rule_key[2],
            "n_attempts": int(self.n_attempts),
            "n_accepts": int(self.n_accepts),
            "accept_rate": float(self.accept_rate),
            "reward_sum": float(self.reward_sum),
            "mean_reward": float(self.mean_reward),
        }


def _proposal_rule_key(
    class_name: str,
    param_name: str,
    rule_kind: str,
    per_class_structural: bool = False,
) -> RuleKey:
    """Map a proposal triple to the bandit's arm key.

    Structural ops (``add_heuristic`` / ``drop_heuristic`` /
    ``add_analyzer`` / ``drop_analyzer``) collapse onto a single arm
    per op type by default — see :meth:`StructuralMutationRule.rule_key`.
    Kwarg perturbations keep the natural one-arm-per-(class, param,
    kind) granularity.  The two paths must agree because
    :meth:`AdaptiveMutationSampler.prime_from_ledger` rebuilds the
    bandit history from JSONL records that only carry the proposal
    triple — never the original rule object.

    When ``per_class_structural`` is ``True``, structural ops are
    further split by the target class's name so each (op, class) pair
    gets its own bandit arm.  For example, adding ``Sobol`` lives on
    ``("Sobol", "add_heuristic", "structural")`` while adding
    ``Random`` lives on ``("Random", "add_heuristic", "structural")``,
    and adding ``Restart`` lives on ``("Restart", "add_analyzer",
    "structural")``.  This trades sparser data per arm for the ability
    to distinguish which class wins / loses inside a structural op.
    The flag must be passed consistently with
    :class:`AdaptiveMutationSampler.per_class_structural` so live
    sampling and ledger replay use the same key layout.
    """
    if rule_kind in _STRUCTURAL_OPS:
        if per_class_structural:
            return (str(class_name), str(rule_kind), "structural")
        return ("*", str(rule_kind), "structural")
    return (str(class_name), str(param_name), str(rule_kind))


class AdaptiveMutationSampler:
    """Thompson-sampling wrapper over :class:`MutationCatalog`.

    Closes the *Adaptive mutation sampler* item from §10 of
    ``planning/SELF_IMPROVEMENT_LOOP.md``.  Each mutation rule is treated as
    one arm of a Bernoulli bandit whose reward is "this iteration was
    accepted".  Per-rule history is summarised by a Beta posterior::

        Beta(prior_alpha + n_accepts, prior_beta + n_attempts - n_accepts)

    On every :meth:`sample` call the sampler draws one variate from each
    applicable rule's posterior and picks the arg-max — the canonical
    Thompson-sampling rule.  Within the chosen rule, the concrete hit
    (which strategy spec / which slot) is selected uniformly, exactly as
    :meth:`MutationCatalog.sample` does today.

    The accept history is updated by :meth:`record_outcome`, which the
    :class:`SelfImprover` invokes after every iteration's
    statistical-acceptance decision (skip iterations are no-ops).  The
    sampler can also be primed from a prior JSONL ledger via
    :meth:`prime_from_ledger`, so the loop carries learning across
    restarts of the driver.

    Cold-start equivalence to uniform sampling.  With the defaults
    ``prior_alpha = prior_beta = 1`` and zero history, every Beta
    posterior is :math:`\\mathrm{U}(0, 1)`.  The arg-max of i.i.d.
    uniforms is itself uniform — so the very first sample is statistically
    indistinguishable from :meth:`MutationCatalog.sample`.

    Args:
        catalog: The underlying :class:`MutationCatalog`.  The adaptive
            sampler does not modify it; it only re-weights how rules are
            selected.
        prior_alpha: Pseudo-count of "successes" in the Beta prior.  Larger
            values flatten the posterior and slow learning; smaller values
            (e.g. ``0.5``) make the sampler greedier earlier.  Must be > 0.
        prior_beta: Pseudo-count of "failures" in the Beta prior.  Same
            shape as ``prior_alpha``; defaults to a symmetric ``Beta(1, 1)``.
            Must be > 0.
        per_class_structural: When ``True``, structural ops
            (``add_heuristic`` / ``drop_heuristic``) are split into
            per-candidate-class bandit arms — e.g. adding ``Sobol`` is a
            distinct arm from adding ``Random``.  Each
            :class:`StructuralMutationRule` is expanded into one
            arm-per-class at :meth:`sample` time, and the per-class hits
            are Thompson-sampled directly so the bandit can learn that
            ``add Sobol`` wins while ``add Random`` loses (instead of
            pooling them).  Default ``False`` keeps the coarse one-arm-
            per-op semantics that have been the published behaviour
            since 2026-05-03.  Pairs naturally with
            :attr:`structural_borrow_alpha`: per-class arms are the leaf
            nodes a hierarchical posterior shares strength across.  Must
            match :func:`_proposal_rule_key`'s ``per_class_structural``
            flag for ledger priming to recover the same arms.
        structural_borrow_alpha: Hierarchical "borrow" coefficient
            ``κ ≥ 0`` for per-class structural arms.  When > 0 (and
            :attr:`per_class_structural` is also ``True``) each per-class
            arm's Beta posterior is built as::

                Beta(prior_alpha + n_class_accepts + κ_eff · n_op_accepts,
                     prior_beta  + n_class_failures + κ_eff · n_op_failures)

            where ``(n_op_accepts, n_op_failures)`` is the aggregate
            over every per-class arm sharing the same structural op.  A
            fresh candidate class therefore starts with the op's
            empirical accept rate rather than the symmetric
            :math:`\\mathrm{Beta}(1, 1)` prior — closing the
            sample-efficiency gap that per-class arms introduce when the
            candidate pool is large.  ``κ = 0`` (default) recovers the
            pure per-class semantics shipped 2026-05-18; ``κ = 1``
            weights every accept across the op equally with the
            class's own accepts.  Inert when
            :attr:`per_class_structural` is ``False`` or when the rule
            being sampled is a kwarg perturbation (kwarg arms are not
            grouped by an "op" so there is nothing to borrow from).
            See :attr:`structural_borrow_horizon` for adaptive annealing
            that reduces the effective ``κ`` as per-class evidence
            accumulates.
        structural_borrow_horizon: Optional adaptive annealing horizon
            ``h > 0`` for the hierarchical borrow coefficient.  When
            ``> 0`` (and :attr:`structural_borrow_alpha` is also ``> 0``
            and :attr:`per_class_structural` is ``True``), each
            per-class arm's effective borrow shrinks toward zero as
            its own evidence accumulates::

                κ_eff = κ / (1 + n_class_attempts / h)

            So a brand-new arm (zero attempts) borrows the full ``κ``
            from the op aggregate — same behaviour as the fixed-``κ``
            path — and a saturated arm (``n_class_attempts >> h``)
            effectively stops borrowing and trusts its own per-class
            posterior.  At ``n_class_attempts == h`` the effective
            borrow is exactly ``κ / 2``.  Default ``0.0`` disables the
            annealing — every arm always borrows the full ``κ``,
            byte-identical to the 2026-06-01 ship.  Recommended values
            for a typical cron: ``5`` to ``10`` (the per-arm posteriors
            warm up within a couple of nights, beyond which the bandit
            should trust the leaf-level signal rather than continue
            being pulled toward the op-level mean).  Inert when any of
            the three preconditions above is missing.  See the
            *Auto-tune κ* idea under :attr:`structural_borrow_alpha`'s
            follow-up backlog in
            ``planning/SELF_IMPROVEMENT_LOG.md``.

    Raises:
        ValueError: If either prior is non-positive, if
            ``structural_borrow_alpha`` is negative, or if
            ``structural_borrow_horizon`` is negative / non-finite.
    """

    def __init__(
        self,
        catalog: MutationCatalog,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        per_class_structural: bool = False,
        structural_borrow_alpha: float = 0.0,
        structural_borrow_horizon: float = 0.0,
    ) -> None:
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError(f"prior_alpha and prior_beta must be > 0, got {prior_alpha!r}, {prior_beta!r}")
        if structural_borrow_alpha < 0 or not np.isfinite(structural_borrow_alpha):
            raise ValueError(f"structural_borrow_alpha must be >= 0 and finite, got {structural_borrow_alpha!r}")
        if structural_borrow_horizon < 0 or not np.isfinite(structural_borrow_horizon):
            raise ValueError(f"structural_borrow_horizon must be >= 0 and finite, got {structural_borrow_horizon!r}")
        self.catalog = catalog
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.per_class_structural = bool(per_class_structural)
        self.structural_borrow_alpha = float(structural_borrow_alpha)
        self.structural_borrow_horizon = float(structural_borrow_horizon)
        self._stats: Dict[RuleKey, MutationRuleStats] = {}
        self._last_rule_key: Optional[RuleKey] = None

    @staticmethod
    def _rule_key(rule: CatalogRule) -> RuleKey:
        # Both :class:`MutationRule` and :class:`StructuralMutationRule`
        # expose ``rule_key()``; delegating keeps the sampler agnostic to
        # which kind of rule the catalog holds.
        return rule.rule_key()

    def _structural_arm_key(self, op: str, class_name: str) -> RuleKey:
        """Return the bandit arm key for a structural hit.

        Centralises the per-class vs collapsed decision so :meth:`sample`
        and :meth:`prime_from_ledger` cannot drift out of sync.
        """
        if self.per_class_structural:
            return (str(class_name), str(op), "structural")
        return ("*", str(op), "structural")

    def _effective_borrow(self, n_class_attempts: int) -> float:
        """Return the per-arm effective hierarchical borrow coefficient.

        Closes the *Auto-tune ``κ``* follow-up below the 2026-06-01
        hierarchical-borrow ship: when
        :attr:`structural_borrow_horizon` is ``> 0``, anneal the
        configured :attr:`structural_borrow_alpha` toward zero as
        per-class evidence accumulates::

            κ_eff = κ / (1 + n_class_attempts / h)

        So a cold arm (``n_class_attempts == 0``) borrows the full
        configured ``κ``, and a saturated arm
        (``n_class_attempts >> h``) effectively stops borrowing — the
        leaf posterior trusts its own per-class evidence rather than
        being indefinitely pulled toward the op-level aggregate.  At
        ``n_class_attempts == h`` the borrow is halved exactly.

        Returns the configured :attr:`structural_borrow_alpha`
        unchanged when annealing is disabled (``horizon == 0``) or
        when the arm has no attempts (``n_class_attempts == 0``) — the
        cold-start case where the borrow is most valuable.
        """
        kappa = self.structural_borrow_alpha
        h = self.structural_borrow_horizon
        if kappa == 0.0 or h == 0.0 or n_class_attempts <= 0:
            return kappa
        return kappa / (1.0 + float(n_class_attempts) / h)

    def get_stats(self, rule: MutationRule) -> MutationRuleStats:
        """Return (creating if needed) the stats bucket for ``rule``."""
        key = self._rule_key(rule)
        if key not in self._stats:
            self._stats[key] = MutationRuleStats(rule_key=key)
        return self._stats[key]

    def stats_snapshot(self) -> List[MutationRuleStats]:
        """Return all rule stats sorted by key (stable across calls)."""
        return [self._stats[k] for k in sorted(self._stats.keys())]

    @property
    def last_rule_key(self) -> Optional[RuleKey]:
        """Rule key of the most recent :meth:`sample` call, or ``None``."""
        return self._last_rule_key

    def sample(
        self,
        rng: np.random.Generator,
        specs: Sequence[StrategySpec],
    ) -> Optional[MutationProposal]:
        """Draw one applicable mutation, biased by Beta posteriors.

        Returns ``None`` iff no rule matches ``specs`` — same contract as
        :meth:`MutationCatalog.sample`.  When that happens,
        :attr:`last_rule_key` is reset to ``None`` so a subsequent
        :meth:`record_outcome` is a safe no-op.

        With ``per_class_structural=True`` each
        :class:`StructuralMutationRule` is expanded into one arm per
        candidate class — every (op, class) pair gets its own Beta
        posterior.  The expansion is done locally in :meth:`sample` so
        the catalog stays the canonical declaration of *what is
        sampleable*, while the bandit decides *how* to slice it.
        """
        applicable = self.catalog.applicable_rules(specs)
        if not applicable:
            self._last_rule_key = None
            return None

        # Build the per-arm view.  For kwarg rules each rule is one arm and
        # the hits stay pooled (the rule-level Beta picks; the hit is then
        # chosen uniformly).  For structural rules we either keep the
        # coarse one-arm-per-op key (legacy behaviour) or split into
        # per-class arms when ``per_class_structural`` is on.
        arms: List[Tuple[CatalogRule, RuleKey, List[Any]]] = []
        for rule, hits in applicable:
            if isinstance(rule, StructuralMutationRule):
                if self.per_class_structural:
                    by_class: Dict[str, List[Any]] = {}
                    for hit in hits:
                        _si, cls, _kwargs = hit
                        by_class.setdefault(cls.__name__, []).append(hit)
                    # Sort by class name so the arm order is stable per
                    # applicable rule and tests can rely on a fixed
                    # enumeration when the rng is seeded.
                    for class_name in sorted(by_class):
                        arms.append(
                            (
                                rule,
                                self._structural_arm_key(rule.op, class_name),
                                list(by_class[class_name]),
                            )
                        )
                else:
                    arms.append((rule, self._rule_key(rule), list(hits)))
            else:
                arms.append((rule, self._rule_key(rule), list(hits)))

        # Op-level aggregates are needed for hierarchical borrowing.  Build
        # them once per :meth:`sample` call rather than per arm so the
        # cost stays linear in the number of stored stats rather than
        # quadratic in the arm count.  Only used when both
        # ``per_class_structural`` and a positive borrow coefficient
        # opt-in to the hierarchy.
        borrow_enabled = self.per_class_structural and self.structural_borrow_alpha > 0
        op_aggregate: Dict[str, Tuple[float, int]] = {}
        if borrow_enabled:
            for k, stats in self._stats.items():
                if k[2] != "structural" or k[0] == "*":
                    continue
                op_name = k[1]
                r, n_a = op_aggregate.get(op_name, (0.0, 0))
                op_aggregate[op_name] = (r + float(stats.reward_sum), n_a + stats.n_attempts)

        # Thompson: one Beta draw per arm, pick the arg-max.  The Beta
        # parameters use ``reward_sum`` (graded-reward path) rather than
        # ``n_accepts`` (binary path), but the historical binary callers
        # set ``reward_sum == n_accepts`` so behaviour is byte-identical
        # whenever :meth:`record_outcome` is called without an explicit
        # ``reward``.  See the ``MutationRuleStats.reward_sum`` docstring.
        n = len(arms)
        sampled = np.empty(n, dtype=np.float64)
        alpha_eff_arr = np.empty(n, dtype=np.float64)
        beta_eff_arr = np.empty(n, dtype=np.float64)
        for i, (rule, key, _) in enumerate(arms):
            stats = self._stats.setdefault(key, MutationRuleStats(rule_key=key))
            reward_sum = float(stats.reward_sum)
            alpha = self.prior_alpha + reward_sum
            beta_param = self.prior_beta + (stats.n_attempts - reward_sum)
            if borrow_enabled and isinstance(rule, StructuralMutationRule):
                op_reward, op_attempts = op_aggregate.get(rule.op, (0.0, 0))
                # Exclude this arm's own contribution so the borrow is
                # over *other* classes' evidence — the leaf posterior is
                # ``Beta(α₀ + r_class + κ_eff·r_other_class, ...)``.
                # Otherwise an arm with lots of evidence would borrow from
                # itself and the hierarchy would collapse to a κ-amplified
                # version of the same per-class posterior.
                other_reward = op_reward - reward_sum
                other_failures = (op_attempts - stats.n_attempts) - other_reward
                kappa_eff = self._effective_borrow(stats.n_attempts)
                alpha += kappa_eff * other_reward
                beta_param += kappa_eff * other_failures
            alpha_eff_arr[i] = alpha
            beta_eff_arr[i] = beta_param
            sampled[i] = float(rng.beta(alpha, beta_param))
        chosen_idx = int(np.argmax(sampled))
        rule, chosen_key, hits = arms[chosen_idx]
        chosen_stats = self._stats[chosen_key]
        alpha_eff = float(alpha_eff_arr[chosen_idx])
        beta_eff = float(beta_eff_arr[chosen_idx])
        thompson_tag = (
            f"[Thompson Beta({alpha_eff:.1f}, {beta_eff:.1f}); "
            f"draw={sampled[chosen_idx]:.3f}; "
            f"history {chosen_stats.n_accepts}/{chosen_stats.n_attempts}]"
        )
        self._last_rule_key = chosen_key

        hit_idx = int(rng.integers(0, len(hits)))

        if isinstance(rule, StructuralMutationRule):
            base = _make_structural_proposal(rule, hits[hit_idx], specs)
            base.rationale = f"{base.rationale} {thompson_tag}"
            return base

        # Kwarg perturbation path — same formatting as the uniform sampler.
        si, _, _, old_value = hits[hit_idx]
        strategy_name = specs[si].name
        new_value = MutationCatalog._mutate_value(rule, old_value, rng)
        rationale = (
            f"{rule.kind} on {rule.class_name}.{rule.param_name} in {strategy_name}: "
            f"{old_value!r} -> {new_value!r} {thompson_tag}"
        )
        return MutationProposal(
            strategy_name=strategy_name,
            class_name=rule.class_name,
            param_name=rule.param_name,
            old_value=_to_plain(old_value),
            new_value=_to_plain(new_value),
            rule_kind=rule.kind,
            rationale=rationale,
        )

    def record_outcome(self, accepted: bool, reward: Optional[float] = None) -> None:
        """Update the bandit with the most recent iteration's verdict.

        No-op when :attr:`last_rule_key` is ``None`` — i.e. when the
        previous iteration was a skip or :meth:`sample` was never called
        — so the driver can call this unconditionally on every
        iteration.

        Args:
            accepted: Whether the loop accepted this iteration.  Updates
                the binary :attr:`MutationRuleStats.n_accepts` counter
                unchanged from historical semantics so :attr:`accept_rate`
                still reports the fraction of accepted proposals (used by
                the summary view and the §12.3 daily routine).
            reward: Graded reward in ``[0, 1]`` accumulated into
                :attr:`MutationRuleStats.reward_sum`.  ``None`` (default)
                falls back to the binary reward ``1.0 if accepted else
                0.0`` — under this default, ``reward_sum`` tracks
                ``n_accepts`` exactly and the Thompson posterior is
                byte-identical to the historical behaviour.  Explicit
                values implement the §7.4 graded-reward shaping
                (``LoopConfig.bandit_reward_shaping = "graded"``): a
                barely-confirmed accept contributes ``~0.5``, a
                clearly-winning accept contributes ``1.0``; an honest
                reject with a positive but sub-eps delta contributes
                ``~0.5`` (still informative), a clearly-harmful reject
                contributes ``~0``.  Values are clamped to ``[0, 1]`` so
                a numeric over/underflow never corrupts the posterior.
        """
        if self._last_rule_key is None:
            return
        stats = self._stats.setdefault(
            self._last_rule_key,
            MutationRuleStats(rule_key=self._last_rule_key),
        )
        if reward is None:
            graded = 1.0 if accepted else 0.0
        else:
            # Clamp defensively — the Beta posterior is undefined for
            # negative reward_sum, and rewards > 1 would silently let an
            # arm dominate by accumulating more than ``n_attempts`` worth
            # of α per pull.  The driver only ever passes values in
            # [0, 1] (§7.4 formula); this clamp is a safety net.
            graded = float(reward)
            if graded < 0.0:
                graded = 0.0
            elif graded > 1.0:
                graded = 1.0
        stats.n_attempts += 1
        if accepted:
            stats.n_accepts += 1
        stats.reward_sum += graded
        self._last_rule_key = None

    def discard_outcome(self) -> None:
        """Forget the most recent proposal *without* updating the bandit.

        Used by the loop driver when an iteration carries zero
        information about whether the rule helps or hurts — most
        importantly on §12.4 no-op iterations where the candidate's
        per-pair scores were bit-identical to baseline.  Pulling the
        arm on a no-op would mis-train the Beta posterior toward the
        symmetric ``Beta(1, 1) → Beta(1, 2)`` direction even though the
        rule's value is undetermined.

        Equivalent to clearing :attr:`last_rule_key` so the next
        iteration's :meth:`record_outcome` call is a no-op.  Safe to
        call when no proposal is pending.
        """
        self._last_rule_key = None

    def _consume_record(self, rec: Dict[str, Any]) -> bool:
        """Apply one ledger record to the bandit posterior.

        Returns ``True`` if the record contributed an ``n_attempts += 1``
        / ``reward_sum += r`` update, ``False`` if it was filtered out
        (non-iteration record, skip, no-op, or null proposal).  Used by
        both :meth:`prime_from_ledger` and :meth:`prime_from_archives`
        so the priming semantics are byte-identical regardless of which
        file the record originated from.
        """
        if rec.get("record_type", "iteration") != "iteration":
            return False
        proposal = rec.get("proposal")
        if proposal is None:
            return False
        # §12.4 no-op iterations carry zero information about whether
        # the rule helps or hurts (per-pair scores were bit-identical
        # to baseline at measure time).  Replaying them as
        # ``n_attempts += 1`` would mis-train the posterior the same
        # way :meth:`record_outcome` is bypassed during the live
        # run.  Legacy records (pre-2026-06-12) carry no ``no_op``
        # key and default to ``False`` here — preserving the
        # historical priming semantics exactly.
        if rec.get("no_op"):
            return False
        key = _proposal_rule_key(
            proposal.get("class_name", ""),
            proposal.get("param_name", ""),
            proposal.get("rule_kind", ""),
            per_class_structural=self.per_class_structural,
        )
        stats = self._stats.setdefault(key, MutationRuleStats(rule_key=key))
        accepted = bool(rec.get("accepted"))
        stats.n_attempts += 1
        if accepted:
            stats.n_accepts += 1
        reward = rec.get("bandit_reward")
        if reward is None:
            graded = 1.0 if accepted else 0.0
        else:
            graded = float(reward)
            if graded < 0.0:
                graded = 0.0
            elif graded > 1.0:
                graded = 1.0
        stats.reward_sum += graded
        return True

    def prime_from_ledger(self, ledger_path: str) -> int:
        """Seed the bandit's history from a prior JSONL ledger.

        Replays every iteration record with a non-null proposal: each
        contributes ``n_attempts += 1`` and, if accepted, also ``n_accepts
        += 1``.  Skip records and guard records are ignored.  Returns the
        number of records consumed.

        Graded-reward records carry an explicit ``bandit_reward`` field
        (added 2026-06-13 with the §7.4 graded reward shipping) and the
        replay accumulates that value into
        :attr:`MutationRuleStats.reward_sum`.  Legacy records (no
        ``bandit_reward`` key) fall back to the binary reward
        ``1.0 if accepted else 0.0`` — the same value the historical
        binary path produced, so ledgers from before the graded ship
        replay byte-identically.

        Useful for resuming a long unattended loop run without losing
        the meta-knowledge of which mutation rules tend to succeed.
        """
        consumed = 0
        for rec in load_ledger(ledger_path):
            if self._consume_record(rec):
                consumed += 1
        return consumed

    def prime_from_archives(self, archive_dir: str, *, ledger_path: Optional[str] = None) -> int:
        """Seed the bandit's history from archived JSONL ledgers.

        Discovers files matching the rotation glob
        ``self_improve_ledger_*.jsonl`` under ``archive_dir`` and replays
        every iteration record across them, oldest first by filename
        (the nightly rotation pattern is
        ``self_improve_ledger_YYYY-MM-DD.jsonl`` so a plain
        :py:func:`sorted` is chronological).  Returns the total number
        of records consumed across all archives — a missing directory,
        an empty directory, or a directory with no matching files all
        return ``0`` and leave the posterior untouched.

        When ``ledger_path`` is given, archive selection is *scoped to that
        ledger's metric* via :func:`iter_metric_archives` so an aocc run
        warms only from aocc archives (and a composite run only from
        composite archives) — the two live on ~100×-different delta scales
        and their graded rewards must not mix (see the AOCC-regime
        follow-ups in ``planning/SELF_IMPROVEMENT_LOG.md``).  When ``None``
        (the default) every matching archive is replayed regardless of
        metric — the historical single-metric behaviour, byte-identical for
        pre-flip archive sets.

        Per-record semantics are byte-identical to
        :meth:`prime_from_ledger` (same :meth:`_consume_record` helper).
        Combined with :meth:`prime_from_ledger` on the live ledger, this
        closes the §2.6 "archives in ``planning/done/`` are invisible"
        diagnosis: the bandit posterior now accumulates evidence across
        every retained nightly run rather than just the current one.
        """
        consumed = 0
        archive_path = pathlib.Path(archive_dir)
        if not archive_path.is_dir():
            return 0
        # Sort by filename for deterministic, chronological replay.  The
        # rotation convention ``self_improve_ledger_YYYY-MM-DD.jsonl``
        # makes lexicographic order equal to chronological order.
        if ledger_path is None:
            archive_files = sorted(archive_path.glob("self_improve_ledger_*.jsonl"))
        else:
            archive_files = iter_metric_archives(archive_dir, ledger_path)
        for ledger_file in archive_files:
            for rec in load_ledger(str(ledger_file)):
                if self._consume_record(rec):
                    consumed += 1
        return consumed


def default_catalog() -> MutationCatalog:
    """Return the built-in hyperparameter mutation catalog.

    Covers the most impactful dials on the harness strategies:

    * ``Nearby.radius`` — local-search step magnitude.
    * ``CMAES.sigma0`` — CMA-ES initial step-size fraction.
    * ``Sensitivity.update_interval`` — importance-recomputation cadence.
    * ``LatinHypercube.div`` — initial-sample coarseness.
    * ``Sobol.n`` — Sobol' initial-design sample count (powers of two).
    * ``Restart.max_restarts`` — restart budget.
    * ``Restart.patience`` — consecutive non-improvement evaluations
      before a restart fires.  Only fires when a spec sets ``patience``
      to a concrete integer; the ``None`` auto-default (``5 * dim``) is
      skipped by :func:`_find_targets`.
    * ``LBFGSB.max_starts`` — multi-start L-BFGS-B restart budget cap.
      Only fires when a spec sets ``max_starts`` to a concrete integer;
      the ``None`` auto-default (unlimited until budget) is skipped by
      :func:`_find_targets`.
    * ``PSO.NP`` / ``PSO.w`` / ``PSO.w_end`` — swarm size, initial /
      terminal inertia (Clerc-Kennedy and Shi-Eberhart parameters).
    * ``PSO.stagnation_threshold`` — stochastic-K stagnation rebuild
      cadence for the ``random`` topology (Clerc 2007 / SPSO 2011).
    * ``LSHADE.NP_init`` / ``LSHADE.H`` / ``LSHADE.p_best`` /
      ``LSHADE.p_best_end`` — L-SHADE population, success-history
      memory size, initial and (optional, iLSHADE / jSO) terminal
      pbest greediness.
    * ``JSO.NP_init`` / ``JSO.H`` / ``JSO.p_best_max`` — jSO
      population, success-history memory size, and upper bound of the
      linear ``p_best`` schedule.  Brest et al. (2017) report
      ``H = 5`` as best for the CEC battery.
    * ``NLSHADE_RSP.NP_init`` / ``NLSHADE_RSP.H`` /
      ``NLSHADE_RSP.k_rank`` — NL-SHADE-RSP population,
      success-history memory size (inherits the ``H >= 2`` anchor-bin
      constraint from jSO), and rank-based selective-pressure
      coefficient.
    * ``COBYQA.initial_tr_radius`` / ``COBYQA.final_tr_radius`` —
      Powell-family trust-region radii.
    * ``RegionUCB.ucb_c`` / ``RegionUCB.gauss_fraction`` /
      ``RegionUCB.gauss_scale`` — UCB1 exploration weight, fraction
      of in-leaf draws taken as Gaussian-around-leaf-best instead of
      uniform, and the Gaussian's relative std-dev.  Closes the
      *Follow-ups: tune ``ucb_c`` / ``gauss_fraction`` via the
      self-improvement catalog* note in the RegionUCB ship entry.
    * Categorical toggles — ``PSO.topology``
      (``gbest`` ↔ ``lbest`` ↔ ``vonneumann``), ``Sobol.scramble``
      (``True`` ↔ ``False``), ``LSHADE.archive_factor``
      (``0.0`` / ``1.0`` / ``2.6``), ``LSHADE.F_schedule``
      (``"off"`` / ``"jso"`` / ``"early"`` / ``"strict"`` — four
      asymmetric F-cap regimes shipped 2026-06-23, with the bool
      inputs accepted as backwards-compat synonyms for ``"jso"`` /
      ``"off"``), ``JSO.p_best_max``
      (``0.15`` / ``0.25`` / ``0.4`` — L-SHADE-like / jSO default /
      iLSHADE-like greediness regimes, alongside the continuous
      ``float_uniform`` rule), ``NLSHADE_RSP.adaptive_archive``
      (``True`` ↔ ``False``), ``NLSHADE_RSP.k_rank``
      (``0.0`` / ``3.0`` / ``5.0`` — RSP-off / default / aggressive
      regimes, alongside the continuous ``float_uniform`` rule),
      ``COBYQA.scale`` (``True`` ↔ ``False``),
      ``NLSHADE_LBC.lbc_regime``
      (``"cec2022"`` / ``"lshade"`` / ``"flat"`` / ``"aggressive"``
      — one composite categorical arm over the five LBC fields
      (``p_F_init`` / ``p_F_final`` / ``p_CR_init`` /
      ``p_CR_final`` / ``m_lbc``); shipped 2026-06-24, replaced the
      five per-field ``float_uniform`` rules previously on the
      catalog with a single literature-motivated joint search), and
      ``Restart.restart_strategy`` (``"random"`` / ``"diverse"`` /
      ``"sphere"`` — uniform-in-box / max-min-distance / Gaussian-
      around-centre center-selection regimes).  These use the
      ``categorical_choice`` mutation kind so the loop can flip
      discrete design knobs the same way it tunes numeric ones.

    Bounds are chosen so a single accept keeps the value in a sensible
    range (never zero, never pathologically large).
    """
    return MutationCatalog(
        [
            # Nearby.radius — tightened 2026-06-26 from (0.005, 0.5) to
            # (0.032, 0.313) based on cross-night ledger evidence: 13
            # accepts across 9 nights cluster in the observed window
            # [0.073, 0.135] with the bandit consistently rejecting
            # proposals well outside that window.  The auto-tuned widening
            # detector (shipped 2026-06-22) sizes the catalog bound to a
            # ~2.31× headroom factor around the observed range — wide
            # enough to keep exploration headroom on either side, narrow
            # enough that every per-iteration pull lands in the
            # productive region.  See the 2026-06-26 entry in
            # ``planning/SELF_IMPROVEMENT_LOG.md``.
            MutationRule(
                strategy_pattern="",
                class_name="Nearby",
                param_name="radius",
                kind="log_uniform_perturb",
                bounds=(0.032, 0.313),
                log_step=0.15,
                probability=1.0,
            ),
            MutationRule(
                strategy_pattern="",
                class_name="CMAES",
                param_name="sigma0",
                kind="log_uniform_perturb",
                bounds=(0.05, 1.0),
                log_step=0.15,
                probability=1.0,
            ),
            MutationRule(
                strategy_pattern="",
                class_name="Sensitivity",
                param_name="update_interval",
                kind="integer_add",
                bounds=(5, 60),
                delta_choices=(-5, -2, 2, 5),
                probability=0.5,
            ),
            MutationRule(
                strategy_pattern="",
                class_name="LatinHypercube",
                param_name="div",
                kind="integer_add",
                bounds=(2, 8),
                delta_choices=(-1, 1),
                probability=0.5,
            ),
            # Sobol' is a power-of-two-friendly low-discrepancy sequence; we
            # double / halve the sample count to stay on 2^k boundaries while
            # respecting the 4..64 envelope.
            MutationRule(
                strategy_pattern="",
                class_name="Sobol",
                param_name="n",
                kind="integer_add",
                bounds=(4, 64),
                delta_choices=(-8, -4, 4, 8),
                probability=0.5,
            ),
            MutationRule(
                strategy_pattern="",
                class_name="Restart",
                param_name="max_restarts",
                kind="integer_add",
                bounds=(1, 20),
                delta_choices=(-2, -1, 1, 2),
                probability=0.5,
            ),
            # Restart patience — the more impactful of the two Restart
            # dials.  Counts the number of consecutive non-improvement
            # evaluations before a restart is triggered.  The analyzer's
            # default is ``5 * dim`` (auto-derived at ``__start__``);
            # the built-in factories deliberately ship ``patience=None``
            # to opt into the auto-default, so this rule only fires when
            # a spec sets ``patience`` to a concrete integer
            # (the ``None``-skip in :func:`_find_targets`).  Bounds
            # ``[3, 200]`` bracket the practical range: 3 is the
            # smallest useful value (trigger restarts as soon as
            # stagnation is detected on tiny problems), 200 keeps the
            # restart cadence above the per-evaluation tick rate even
            # on the longest-budget runs.  Delta choices are
            # asymmetric-by-magnitude so the bandit can probe both
            # nearby and farther-away cadences.
            MutationRule(
                strategy_pattern="",
                class_name="Restart",
                param_name="patience",
                kind="integer_add",
                bounds=(3, 200),
                delta_choices=(-20, -10, -5, 5, 10, 20),
                probability=0.5,
            ),
            # Restart center-picking regime (categorical).  Flips an
            # existing :class:`~panobbgo.analyzers.restart.Restart`
            # instance between the three supported center-selection
            # policies: ``"random"`` (uniform draw inside the box),
            # ``"diverse"`` (max-min distance from previous restart
            # centers — strongest coverage signal after several
            # restarts), and ``"sphere"`` (Gaussian around the box
            # centre with ``std = ranges / 6`` — biases the restart
            # cloud toward the centroid, useful when the optimum lies
            # in the box interior rather than near its boundary).
            # Only fires when the spec sets ``restart_strategy``
            # explicitly — the analyzer's default is ``"random"`` and
            # specs that omit the kwarg are skipped.  The structural
            # catalog's ``add_analyzer`` candidate, ``IPOP_CMAES`` /
            # ``BIPOP_CMAES`` in :mod:`panobbgo.harness`, and the
            # ``Sensitivity_Aggressive`` spec in
            # :mod:`panobbgo.harness_ioh` all ship
            # ``restart_strategy="diverse"`` so this rule fires
            # out-of-the-box on every batter spec that uses
            # :class:`Restart`.
            MutationRule(
                strategy_pattern="",
                class_name="Restart",
                param_name="restart_strategy",
                kind="categorical_choice",
                choices=("random", "diverse", "sphere"),
                probability=0.3,
            ),
            # LBFGSB multi-start budget cap.  ``max_starts`` defaults to
            # ``None`` (= unlimited until the strategy budget is
            # exhausted) and is auto-resolved on the heuristic; this
            # rule only fires when a spec sets ``max_starts`` to a
            # concrete positive integer (the ``None``-skip in
            # :func:`_find_targets`).  Bounds ``[1, 50]`` bracket the
            # exploration / exploitation trade-off: 1 = pure box-centre
            # descent (no random restarts; useful on smooth unimodal
            # problems), 50 = highly aggressive multi-start (useful on
            # multi-modal problems where a single basin is unlikely to
            # win the run).  Step sizes mirror ``Restart.max_restarts``
            # — small enough that one accept does not catapult the dial
            # across regimes, large enough that the bandit can climb the
            # surface in a handful of accepts.
            MutationRule(
                strategy_pattern="",
                class_name="LBFGSB",
                param_name="max_starts",
                kind="integer_add",
                bounds=(1, 50),
                delta_choices=(-5, -2, -1, 1, 2, 5),
                probability=0.5,
            ),
            # PSO swarm size — too small starves the social attraction
            # term, too large wastes evaluations on a fixed budget.
            # Step in increments of 4 so the swarm does not jitter around
            # noise-level changes.
            MutationRule(
                strategy_pattern="",
                class_name="PSO",
                param_name="NP",
                kind="integer_add",
                bounds=(8, 60),
                delta_choices=(-8, -4, 4, 8),
                probability=0.5,
            ),
            # PSO inertia (initial value when ``w_end`` is set, constant
            # otherwise).  Bounds bracket the literature: 0.4 is the
            # lower end of the Shi-Eberhart schedule, 0.95 the upper end
            # of "still convergent in practice".  Only fires when a
            # spec explicitly sets ``w`` — it is otherwise a default
            # kwarg on :class:`PSO` and not present in the spec dict.
            MutationRule(
                strategy_pattern="",
                class_name="PSO",
                param_name="w",
                kind="float_uniform",
                bounds=(0.4, 0.95),
                low=0.4,
                high=0.95,
                probability=0.5,
            ),
            # PSO terminal inertia for the linearly-decreasing schedule
            # (Shi-Eberhart 1998).  When the spec sets ``w_end`` the
            # heuristic anneals from ``w`` to ``w_end`` over the budget;
            # without ``w_end`` the inertia is constant.  Bounds chosen
            # to keep the late-search inertia in the "exploit" regime.
            MutationRule(
                strategy_pattern="",
                class_name="PSO",
                param_name="w_end",
                kind="float_uniform",
                bounds=(0.2, 0.6),
                low=0.2,
                high=0.6,
                probability=0.5,
            ),
            # PSO stochastic-K stagnation-rebuild threshold (Clerc 2007
            # / SPSO 2011).  Re-samples the random adjacency when the
            # swarm fails to lift its global best for ``N`` consecutive
            # incoming results.  Bounds bracket the literature: ``NP``
            # (one swarm cycle) is the SPSO 2011 default at NP=20, so
            # the [5, 60] range covers half-cycle through triple-cycle
            # for typical swarm sizes.  Only fires when the spec sets
            # ``stagnation_threshold`` explicitly — the default kwarg
            # on :class:`PSO` is ``None`` and not present in the spec
            # dict.  Only meaningful with ``topology="random"`` (the
            # heuristic ignores the kwarg under the three geometric
            # topologies whose adjacency is deterministic), so the
            # rule pairs naturally with the structural-catalog
            # ``add_heuristic`` candidate that ships
            # ``topology="random"``.
            MutationRule(
                strategy_pattern="",
                class_name="PSO",
                param_name="stagnation_threshold",
                kind="integer_add",
                bounds=(5, 60),
                delta_choices=(-10, -5, 5, 10),
                probability=0.5,
            ),
            # L-SHADE (Tanabe-Fukunaga 2014) initial population size.
            # The literature setting is ``18 · d`` which is well above
            # Panobbgo's typical budget; bracket the practical range
            # ``[10, 60]`` and step in 5/10-individual increments so the
            # swarm size moves meaningfully but not catastrophically.
            # ``NP_min = 4`` is a hard constraint inside the heuristic
            # so the lower bound here stays well clear of it.
            MutationRule(
                strategy_pattern="",
                class_name="LSHADE",
                param_name="NP_init",
                kind="integer_add",
                bounds=(10, 60),
                delta_choices=(-10, -5, 5, 10),
                probability=0.5,
            ),
            # L-SHADE history memory size H.  The SHADE / L-SHADE papers
            # both use H = 6.  Probing 4 .. 12 lets the loop adapt the
            # update smoothness without straying outside the regime
            # where the algorithm is well-behaved.
            MutationRule(
                strategy_pattern="",
                class_name="LSHADE",
                param_name="H",
                kind="integer_add",
                bounds=(4, 12),
                delta_choices=(-2, -1, 1, 2),
                probability=0.5,
            ),
            # L-SHADE pbest greediness.  ``p_best`` controls how greedy
            # the ``current-to-pbest/1`` mutation is — lower values
            # (0.05) pull toward the very best individual, higher values
            # (0.2) sample from a broader top slice.  Tanabe-Fukunaga
            # report 0.11 as a robust default.
            MutationRule(
                strategy_pattern="",
                class_name="LSHADE",
                param_name="p_best",
                kind="float_uniform",
                bounds=(0.05, 0.25),
                low=0.05,
                high=0.25,
                probability=0.5,
            ),
            # L-SHADE terminal pbest for the optional iLSHADE / jSO
            # linearly-decreasing schedule (Brest et al. 2016 / 2017).
            # When the spec sets ``p_best_end`` the heuristic anneals
            # ``p_best`` from its initial value down to ``p_best_end``
            # over the strategy budget; without ``p_best_end`` the
            # greediness is constant.  The jSO setting is half the
            # initial ``p_best`` (e.g. 0.125 when starting from 0.25),
            # so bracket the practical range ``[0.025, 0.15]``.  Only
            # fires when a spec sets ``p_best_end`` explicitly — the
            # default kwarg on :class:`LSHADE` is ``None`` and not
            # present in the spec dict.
            MutationRule(
                strategy_pattern="",
                class_name="LSHADE",
                param_name="p_best_end",
                kind="float_uniform",
                bounds=(0.025, 0.15),
                low=0.025,
                high=0.15,
                probability=0.5,
            ),
            # PSO topology toggle (categorical).  Flips an existing PSO
            # heuristic between the canonical Kennedy-Eberhart ``gbest``
            # (fully-connected swarm, instantaneous diffusion), the
            # Kennedy-Mendes ``lbest`` (ring with one-hop diffusion,
            # better on multimodal landscapes), ``vonneumann`` (a
            # 4-connected 2-D toroidal grid that sits between the two
            # extremes — Kennedy & Mendes 2003, Mendes 2004), and
            # ``random`` (the Mendes 2004 / Clerc 2007 / SPSO 2011
            # stochastic-informer graph — structure-free middle ground
            # whose diffusion speed depends on the realised graph).
            # Only fires when the spec sets ``topology`` explicitly —
            # the default PSO constructor leaves it implicit at
            # ``"gbest"``.  The structural catalog ships lbest,
            # vonneumann, and random variants, so this rule is
            # immediately useful for any portfolio that has gained an
            # explicit-topology PSO via ``add_heuristic``.
            MutationRule(
                strategy_pattern="",
                class_name="PSO",
                param_name="topology",
                kind="categorical_choice",
                choices=("gbest", "lbest", "vonneumann", "random"),
                probability=0.3,
            ),
            # Sobol' scrambling toggle (categorical).  ``scramble=True``
            # (Owen-style) keeps draws low-discrepancy but breaks the
            # exact deterministic grid; ``scramble=False`` reproduces
            # the classic Sobol' sequence verbatim.  ``Rewarding_Diverse``
            # was codified to ``scramble=False`` on 2026-05-31 after
            # three independent self-improvement loop accepts (see the
            # archived ledger in ``planning/done/``); ``BayesOpt_Sobol``
            # still ships ``scramble=True`` so this rule fires
            # out-of-the-box on both the quick and standard battery and
            # the bandit is free to flip either spec.
            MutationRule(
                strategy_pattern="",
                class_name="Sobol",
                param_name="scramble",
                kind="categorical_choice",
                choices=(True, False),
                probability=0.3,
            ),
            # L-SHADE archive factor toggle (categorical).  ``0.0``
            # disables the external archive entirely (classic
            # current-to-pbest/1 with no replaced-parent memory);
            # ``1.0`` and ``2.6`` are the Tanabe-Fukunaga 2014 default
            # and the L-SHADE-RSP enlarged-archive setting
            # respectively.  Discrete switch because the
            # archive-on/archive-off boundary is qualitatively
            # different from "tune the archive size".  Only fires
            # when a spec sets ``archive_factor`` explicitly.
            MutationRule(
                strategy_pattern="",
                class_name="LSHADE",
                param_name="archive_factor",
                kind="categorical_choice",
                choices=(0.0, 1.0, 2.6),
                probability=0.3,
            ),
            # L-SHADE asymmetric F-cap regime (categorical).  Each named
            # regime is a 3-phase asymmetric cap on the drawn ``F``:
            #
            # * ``"off"`` — no cap (byte-identical Tanabe-Fukunaga 2014).
            # * ``"jso"`` — Brest et al. 2017: clamp at 0.7 in the first
            #   60% of the budget, at 0.8 in the next 30%, unclamped in
            #   the final 10%.
            # * ``"early"`` — earlier kick-in: clamp at 0.6 in the first
            #   40%, 0.8 in the next 30%, unclamped in the final 30%.
            # * ``"strict"`` — aggressive throughout: clamp at 0.5 in the
            #   first 50%, 0.7 in the next 35%, unclamped in the final 15%.
            #
            # See :data:`panobbgo.heuristics.lshade._F_SCHEDULE_REGIMES`
            # for the per-regime (phase1_bound, phase2_bound, phase1_cap,
            # phase2_cap) tuples.  Only fires when a spec sets
            # ``F_schedule`` explicitly (the default kwarg is ``None``);
            # the constructor accepts the bool synonyms ``True`` / ``False``
            # for backwards compatibility with the binary ledger entries
            # shipped between 2026-05-21 and this rule.
            MutationRule(
                strategy_pattern="",
                class_name="LSHADE",
                param_name="F_schedule",
                kind="categorical_choice",
                choices=("off", "jso", "early", "strict"),
                probability=0.3,
            ),
            # jSO (Brest, Maučec & Bošković 2017) initial population size.
            # Same ``[10, 60]`` bracket as L-SHADE — they share the
            # ``current-to-pbest/1`` mutation skeleton and benefit from the
            # same population sizing trade-off.
            MutationRule(
                strategy_pattern="",
                class_name="JSO",
                param_name="NP_init",
                kind="integer_add",
                bounds=(10, 60),
                delta_choices=(-10, -5, 5, 10),
                probability=0.5,
            ),
            # jSO upper bound on the linear ``p_best`` schedule.  Brest et al.
            # report ``0.25`` as the default; bracket the practical range
            # ``[0.15, 0.4]`` so the loop can probe a slightly greedier or
            # broader pbest pool without going below the implicit floor of
            # ``p_best_min`` (the constructor enforces ``p_best_min <= p_best_max``).
            MutationRule(
                strategy_pattern="",
                class_name="JSO",
                param_name="p_best_max",
                kind="float_uniform",
                bounds=(0.15, 0.4),
                low=0.15,
                high=0.4,
                probability=0.5,
            ),
            # jSO ``p_best_max`` regime toggle (categorical).  Three
            # literature-canonical settings collapsed onto one bandit
            # arm: ``0.15`` (close to the Tanabe-Fukunaga L-SHADE
            # ``0.11`` setting, raised above jSO's default
            # ``p_best_min = 0.125`` so the constructor's
            # ``p_best_min <= p_best_max`` check passes), ``0.25``
            # (the Brest et al. 2017 jSO default), and ``0.4`` (the
            # iLSHADE / Brest et al. 2016 broader pool — useful on
            # highly multi-modal landscapes where a narrow ``pbest``
            # slice can lock onto the wrong basin).  Sits alongside
            # the ``float_uniform`` rule above so the bandit can either
            # continuously walk ``p_best_max`` or jump between the
            # qualitatively distinct regimes; the two rules occupy
            # distinct ``(class, param, rule_kind)`` arms.  Fires only
            # when a spec sets ``p_best_max`` explicitly — the
            # constructor default ``0.25`` is filtered out by the
            # established opt-in predicate so this rule is dormant
            # on specs that omit the kwarg.
            MutationRule(
                strategy_pattern="",
                class_name="JSO",
                param_name="p_best_max",
                kind="categorical_choice",
                choices=(0.15, 0.25, 0.4),
                probability=0.3,
            ),
            # jSO history memory size H.  Brest et al. report ``H = 5`` as
            # best for the CEC battery (vs L-SHADE's H = 6).  The constructor
            # enforces ``H >= 2`` (anchor bin requires at least one writable
            # bin); bracket ``[4, 12]`` so the loop probes the smoothness
            # of the success-history update without straying outside the
            # well-behaved regime.  Mirrors the ``LSHADE.H`` rule for the
            # subclass.
            MutationRule(
                strategy_pattern="",
                class_name="JSO",
                param_name="H",
                kind="integer_add",
                bounds=(4, 12),
                delta_choices=(-2, -1, 1, 2),
                probability=0.5,
            ),
            # NL-SHADE-RSP (Stanovov et al. 2021) initial population size.
            # Same ``[10, 60]`` bracket as L-SHADE / jSO — they share the
            # ``current-to-pbest-w/1`` mutation skeleton and benefit from
            # the same population sizing trade-off.
            MutationRule(
                strategy_pattern="",
                class_name="NLSHADE_RSP",
                param_name="NP_init",
                kind="integer_add",
                bounds=(10, 60),
                delta_choices=(-10, -5, 5, 10),
                probability=0.5,
            ),
            # NL-SHADE-RSP rank-based selective-pressure coefficient.  The
            # literature default is ``k_rank = 3``; bracket ``[1, 5]`` so
            # the loop can probe weaker (closer to uniform ``r1`` selection)
            # or stronger (greedier toward the leading basin) pressure.
            # Fires whenever a spec sets ``k_rank`` explicitly — the
            # structural-catalog candidate does, so this dial is live
            # out-of-the-box once NL-SHADE-RSP is added to a portfolio.
            MutationRule(
                strategy_pattern="",
                class_name="NLSHADE_RSP",
                param_name="k_rank",
                kind="float_uniform",
                bounds=(1.0, 5.0),
                low=1.0,
                high=5.0,
                probability=0.5,
            ),
            # NL-SHADE-RSP ``k_rank`` regime toggle (categorical).  Three
            # literature-canonical settings: ``0.0`` (uniform ``r1``
            # selection, recovers jSO behaviour and lets the bandit flip
            # an RSP-on instance to RSP-off without dropping the heuristic),
            # ``3.0`` (the Stanovov et al. 2018 / 2021 RSP default), and
            # ``5.0`` (more aggressive rank pressure for highly multi-modal
            # landscapes).  Sits alongside the ``float_uniform`` rule so
            # the bandit can either continuously walk ``k_rank`` or jump
            # between qualitatively distinct regimes.  Fires whenever a
            # spec sets ``k_rank`` explicitly — the structural-catalog
            # candidate does.
            MutationRule(
                strategy_pattern="",
                class_name="NLSHADE_RSP",
                param_name="k_rank",
                kind="categorical_choice",
                choices=(0.0, 3.0, 5.0),
                probability=0.3,
            ),
            # NL-SHADE-RSP history memory size H.  Symmetric with the
            # ``LSHADE.H`` / ``JSO.H`` rules; bracket ``[4, 12]``.  Fires
            # whenever a spec sets ``H`` explicitly.
            MutationRule(
                strategy_pattern="",
                class_name="NLSHADE_RSP",
                param_name="H",
                kind="integer_add",
                bounds=(4, 12),
                delta_choices=(-2, -1, 1, 2),
                probability=0.5,
            ),
            # NL-SHADE-RSP randomised-archive toggle (categorical).  When
            # ``True`` (the CEC-2021 behaviour) the archive cap is resampled
            # per generation in ``[0, A_max]``; ``False`` recovers jSO's
            # fixed cap.  Only fires when a spec sets ``adaptive_archive``
            # explicitly (the constructor default is ``True``).
            MutationRule(
                strategy_pattern="",
                class_name="NLSHADE_RSP",
                param_name="adaptive_archive",
                kind="categorical_choice",
                choices=(True, False),
                probability=0.3,
            ),
            # NL-SHADE-LBC initial population size — same ``[10, 60]``
            # bracket as the L-SHADE / jSO / NL-SHADE-RSP family.
            MutationRule(
                strategy_pattern="",
                class_name="NLSHADE_LBC",
                param_name="NP_init",
                kind="integer_add",
                bounds=(10, 60),
                delta_choices=(-10, -5, 5, 10),
                probability=0.5,
            ),
            # NL-SHADE-LBC named bias-change regime (categorical).  Flips
            # an existing :class:`~panobbgo.heuristics.nl_shade_lbc.NLSHADE_LBC`
            # instance between four literature-motivated joint
            # configurations of the five LBC fields
            # (``p_F_init`` / ``p_F_final`` / ``p_CR_init`` /
            # ``p_CR_final`` / ``m_lbc``) as a single discrete bandit
            # arm: ``"cec2022"`` (Stanovov et al. 2022 defaults — the
            # CEC-2022 winning configuration), ``"lshade"`` (recovers
            # the standard L-SHADE / jSO / NL-SHADE-RSP Lehmer mean at
            # ``p = 2, m = 1`` — turns the LBC mechanism off without
            # dropping the heuristic), ``"flat"`` (pure arithmetic
            # mean throughout, default spread) and ``"aggressive"``
            # (strong bias throughout, default spread).  Only fires
            # when the spec sets ``lbc_regime`` explicitly — the
            # constructor's default is ``None`` (the five individual
            # float kwargs apply, all at their byte-identical CEC 2022
            # defaults).  Sits alongside the five per-field
            # ``float_uniform`` rules above on the same heuristic — the
            # categorical rule operates on the *joint* regime, the
            # per-field rules on individual dials.  Distinct bandit arm
            # keys by construction (different ``rule_kind``).  Mirrors
            # the 2026-06-23 :class:`LSHADE`.``F_schedule`` regime
            # broadening: one well-curated discrete arm for joint
            # exploration of a high-dimensional dial group instead of
            # five independent cold-started float arms.
            MutationRule(
                strategy_pattern="",
                class_name="NLSHADE_LBC",
                param_name="lbc_regime",
                kind="categorical_choice",
                choices=("cec2022", "lshade", "flat", "aggressive"),
                probability=0.3,
            ),
            # LSHADE-EpSin (Awad, Ali, Suganthan 2016) initial population
            # size — same ``[10, 60]`` bracket as L-SHADE / jSO / NL-SHADE-RSP.
            # The EpSin sinusoidal-ensemble adaptation operates on top of
            # the same ``current-to-pbest/1`` mutation skeleton, so the
            # NP_init sweet spot is dictated by the same exploration /
            # LPSR trade-off rather than the F-adaptation mechanism.
            MutationRule(
                strategy_pattern="",
                class_name="LSHADE_EpSin",
                param_name="NP_init",
                kind="integer_add",
                bounds=(10, 60),
                delta_choices=(-10, -5, 5, 10),
                probability=0.5,
            ),
            # LSHADE-EpSin initial mean frequency for Sinusoid 2 (variable-
            # frequency, increasing-envelope ``F`` sampler).  Awad et al.
            # report ``mu_freq_init = 0.5`` as the default; bracket
            # ``[0.1, 0.9]`` so the loop can probe slower oscillations
            # (closer to 0.1, ``F`` shifts slowly with the generation
            # count) or faster ones (closer to 0.9, more variation per
            # generation).  Stays well clear of ``0`` and ``1`` where
            # the sinusoidal envelope degenerates.
            MutationRule(
                strategy_pattern="",
                class_name="LSHADE_EpSin",
                param_name="mu_freq_init",
                kind="float_uniform",
                bounds=(0.1, 0.9),
                low=0.1,
                high=0.9,
                probability=0.5,
            ),
            # COBYQA (Ragonneau-Zhang 2023) initial trust-region radius —
            # log-uniform around the literature default (0.1).  Only fires
            # when a spec explicitly sets ``initial_tr_radius`` (the
            # heuristic auto-derives it from the box width otherwise);
            # values are in *absolute* units, suitable for boxes scaled to
            # ``[-1, 1]`` via ``scale=True`` (the default).
            MutationRule(
                strategy_pattern="",
                class_name="COBYQA",
                param_name="initial_tr_radius",
                kind="log_uniform_perturb",
                bounds=(0.01, 1.0),
                log_step=0.15,
                probability=0.5,
            ),
            # COBYQA final trust-region radius — convergence threshold.
            # Bracket the literature range; tighter (1e-8) gives more
            # accurate final values at the cost of more evaluations,
            # looser (1e-4) stops earlier and frees budget for other
            # heuristics in the strategy.
            MutationRule(
                strategy_pattern="",
                class_name="COBYQA",
                param_name="final_tr_radius",
                kind="log_uniform_perturb",
                bounds=(1e-8, 1e-4),
                log_step=0.25,
                probability=0.5,
            ),
            # COBYQA box-rescaling toggle (categorical).  ``True`` (the
            # default) rescales the variables to ``[-1, 1]`` based on the
            # box bounds so the interpolation geometry stays
            # well-conditioned on boxes whose axes span very different
            # magnitudes; ``False`` runs COBYQA on the raw box.  Useful
            # when the problem's box is already isotropic and the rescale
            # adds rounding noise that hurts the quadratic-model fit.
            # Only fires when a spec sets ``scale`` explicitly (the
            # default kwarg is ``True``).
            MutationRule(
                strategy_pattern="",
                class_name="COBYQA",
                param_name="scale",
                kind="categorical_choice",
                choices=(True, False),
                probability=0.3,
            ),
            # RegionUCB UCB1 exploration weight — controls the
            # exploration / exploitation balance of the leaf-bandit
            # score ``quality + ucb_c * sqrt(log(N) / n_leaf)``.
            # Bounds bracket the literature range: ``0.1`` strongly
            # favours exploitation of the currently-best leaf, ``4.0``
            # favours uniform-ish allocation across leaves.  The
            # heuristic default of ``1.0`` matches Auer et al. 2002's
            # canonical UCB1 setting and lives near the centre of the
            # log-uniform window.  Only fires when a spec sets
            # ``ucb_c`` explicitly.
            MutationRule(
                strategy_pattern="",
                class_name="RegionUCB",
                param_name="ucb_c",
                kind="log_uniform_perturb",
                bounds=(0.1, 4.0),
                log_step=0.15,
                probability=0.5,
            ),
            # RegionUCB fraction of in-leaf draws taken as Gaussian
            # around the leaf's best point instead of uniform over the
            # leaf box.  ``0.0`` reduces RegionUCB to a pure
            # in-leaf uniform sampler (LA-MCTS style); ``1.0`` makes
            # every draw a local refinement around the leaf best (no
            # in-leaf exploration).  The constructor default of
            # ``0.5`` balances both modes.  Only fires when a spec
            # sets ``gauss_fraction`` explicitly.
            MutationRule(
                strategy_pattern="",
                class_name="RegionUCB",
                param_name="gauss_fraction",
                kind="float_uniform",
                bounds=(0.0, 1.0),
                low=0.0,
                high=1.0,
                probability=0.5,
            ),
            # RegionUCB Gaussian-around-best std-dev, expressed as a
            # fraction of the leaf's per-axis ranges.  Smaller values
            # produce tighter local refinement (close to a Nearby-style
            # neighbourhood), larger values approach the uniform-leaf
            # baseline.  The constructor default of ``0.25`` sits
            # near the geometric middle of the log-uniform window so a
            # symmetric perturbation can both shrink and widen the
            # Gaussian.  Only fires when a spec sets ``gauss_scale``
            # explicitly.
            MutationRule(
                strategy_pattern="",
                class_name="RegionUCB",
                param_name="gauss_scale",
                kind="log_uniform_perturb",
                bounds=(0.05, 0.5),
                log_step=0.15,
                probability=0.5,
            ),
        ]
    )


def default_structural_catalog() -> MutationCatalog:
    """Return :func:`default_catalog` extended with portfolio-shape rules.

    Implements the *Strategy portfolio composition* mutation class from
    §7.2 of ``planning/SELF_IMPROVEMENT_LOOP.md``.  Four structural rules
    sit alongside the existing kwarg perturbations:

    * ``add_heuristic`` from a curated pool of unconditionally-safe
      generators (``Random``, ``Nearby``, ``NelderMead``, ``Center``,
      ``LatinHypercube``, ``Sobol``, ``Extremal``, plus the strong
      population variants PSO ``gbest`` / ``lbest`` / ``vonneumann``,
      LSHADE, jSO, NLSHADE_RSP, COBYQA).  ``avoid_duplicates=True`` so
      the catalog never proposes a duplicate of a class already in the
      strategy.
    * ``drop_heuristic`` with ``min_heuristics=2`` so no strategy is
      ever stripped down past the diversity floor.
    * ``add_analyzer`` from a curated pool of analyzers (``Sensitivity``,
      ``Restart``).  Shipped 2026-06-02 — extends the loop's reach
      beyond heuristics so the bandit can also discover whether the
      ``Restart`` analyzer (warm restarts) or the ``Sensitivity``
      analyzer (adaptive tracking) helps a given seed composition.
      ``avoid_duplicates=True`` skips analyzer classes already attached.
    * ``drop_analyzer`` with ``min_analyzers=0`` because analyzers are
      non-essential — stripping :class:`Sensitivity` from a Rewarding
      strategy yields a valid, slightly faster spec.

    All four rules carry a low probability (``0.3``) relative to the
    kwarg rules — structural changes are higher-variance than retunes,
    so the loop should sample them sparingly.  The overall acceptance
    rate stays in the same neighbourhood as :func:`default_catalog`'s
    while expanding the search space the loop can explore.

    Opt-in via :class:`SelfImprover`'s ``catalog=`` argument or
    ``scripts/self_improve.py run --structural``.  The default
    :class:`SelfImprover` instance still uses :func:`default_catalog`,
    so existing CLI invocations are byte-identical.
    """
    from panobbgo.heuristics import (  # local import to avoid heuristics-package cycles
        Center,
        Extremal,
        LatinHypercube,
        Nearby,
        NelderMead,
        Random,
        Sobol,
    )
    from panobbgo.analyzers import Restart, Sensitivity

    base_rules = list(default_catalog().rules)
    # PSO and L-SHADE are loaded lazily because they use a slightly
    # heavier set of numpy / RNG primitives than the simpler heuristics
    # above, and ``default_structural_catalog`` may be called from
    # environments (e.g. minimal CI) that import :mod:`panobbgo.self_improve`
    # without the full heuristics package.  The local imports keep the
    # cost of the catalog factory unchanged when these classes are not
    # actually selected.
    from panobbgo.heuristics.pso import PSO
    from panobbgo.heuristics.lshade import LSHADE
    from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin
    from panobbgo.heuristics.jso import JSO
    from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP
    from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC
    from panobbgo.heuristics.cobyqa import COBYQA
    from panobbgo.heuristics.lbfgsb import LBFGSB
    from panobbgo.heuristics.cma_es import CMAES

    # Four PSO entries cover the canonical ``gbest`` (default
    # Kennedy-Eberhart 1995 swarm), the ``lbest`` ring topology (Kennedy &
    # Mendes 2002), ``vonneumann`` 2-D toroidal grid (Kennedy & Mendes
    # 2003, Mendes 2004), and ``random`` stochastic-informer graph
    # (Mendes 2004; Clerc 2007 / SPSO 2011) — four complementary
    # information-diffusion regimes (instantaneous, one-hop linear,
    # two-hop planar, stochastic asymmetric).
    # ``avoid_duplicates=True`` ensures only one PSO variant ends up in any
    # given strategy — the catalog picks one of the four uniformly when
    # PSO is not yet present, after which subsequent samples skip them all.
    # L-SHADE (Tanabe-Fukunaga 2014) is the literature-best adaptive
    # Differential Evolution variant; the default ``NP_init=30`` matches
    # Panobbgo's typical max-eval budgets.  ``NP_min`` stays at the
    # heuristic's default of 4 (the floor required by current-to-pbest/1).
    # jSO (Brest, Maučec & Bošković 2017) is the CEC-2017 winner, a
    # direct refinement of L-SHADE that adds a weighted ``current-to-pbest-w/1``
    # mutation, a linear ``p_best`` schedule, Cauchy-F clamping, and a
    # frozen-anchor memory bin.  L-SHADE and jSO are listed as *separate*
    # candidate classes — ``avoid_duplicates`` (the default) only filters
    # exact-class matches, so a portfolio can end up with both arms.  The
    # bandit then weighs whichever DE-family variant wins on the current
    # battery via the per-heuristic reward signal.
    # COBYQA (Ragonneau-Zhang 2023) is Powell's BOBYQA / NEWUOA successor —
    # a derivative-free trust-region method with quadratic interpolation
    # models, dominant on smooth / near-smooth local refinement.  Defaults
    # to ``scale=True`` and the heuristic's auto-derived initial TR radius.
    # NL-SHADE-RSP (Stanovov, Akhmedova & Semenkin 2021) is the CEC-2021
    # winner, a direct refinement of jSO that adds non-linear population
    # reduction, rank-based selective pressure (``k_rank``) on the
    # differential ``r1`` draw, and a randomised adaptive archive.  Listed
    # as a separate candidate class from L-SHADE / jSO so the bandit can
    # weigh whichever DE-family arm wins on the current battery.
    # NL-SHADE-LBC (Stanovov, Akhmedova & Semenkin 2022) is the CEC-2022
    # winner, a direct refinement of NL-SHADE-RSP that adds **Linear Bias
    # Change** in the success-history memory update: the Lehmer-mean
    # exponents for F and CR follow a budget-progress schedule.  Listed
    # as a separate candidate class so the bandit can weigh whichever
    # DE-family arm wins on the current battery.
    # LBFGSB (Zhu-Byrd-Lu-Nocedal 1997) is the only *gradient-based*
    # (finite-difference quasi-Newton) arm — a multi-start bound-constrained
    # local optimizer.  It complements the derivative-free generators above
    # on smooth ill-conditioned valleys (e.g. Rosenbrock), where a curvature
    # estimate converges in a fraction of the evaluations a population method
    # needs.  ``avoid_duplicates`` keeps at most one LBFGSB per strategy.
    # LSHADE-EpSin (Awad, Ali & Suganthan 2016) is the direct precursor of
    # the CEC-2017 co-winner LSHADE-cnEpSin.  It replaces the SHADE
    # Cauchy-from-memory ``F`` sampling with an ensemble of two sinusoidal
    # candidates during the first half of the search (Sinusoid 1 = fixed
    # frequency, decreasing envelope; Sinusoid 2 = variable frequency from
    # an adaptive Cauchy mean, increasing envelope), reverting to
    # SHADE Cauchy in the second half.  Different *branch* of the DE
    # family tree from jSO / NL-SHADE-RSP — all current arms adapt ``F``
    # via the SHADE Cauchy memory; EpSin's deterministic-amplitude
    # sinusoid is algorithmically distinct.  Listed as a separate
    # candidate class so the bandit can weigh whichever ``F``-adaptation
    # mechanism wins on the current battery.
    candidates: Tuple[Tuple[type, Dict[str, Any]], ...] = (
        (Random, {}),
        (Nearby, {"radius": 0.1, "axes": "all", "new": 3}),
        (NelderMead, {}),
        (Center, {}),
        (LatinHypercube, {"div": 4}),
        (Sobol, {"n": 16, "scramble": True}),
        (Extremal, {}),
        (PSO, {"NP": 20}),  # canonical Clerc-Kennedy global-best swarm
        (PSO, {"NP": 20, "topology": "lbest", "k_neighbors": 2}),  # ring topology
        (PSO, {"NP": 20, "topology": "vonneumann"}),  # 4-connected 2-D grid (Mendes 2004)
        (PSO, {"NP": 20, "topology": "random", "k_neighbors": 3}),  # Clerc 2007 / SPSO 2011 (K=3)
        # ``NP_init="auto"`` sizes each DE population from the strategy budget
        # (see :class:`~panobbgo.heuristics.lshade.LSHADE`).  A fixed ``NP_init=30``
        # is far too large for the tight budgets the loop runs (measured ~2× worse
        # than budget-adaptive sizing at budget 200, worse still at the quick-mode
        # budget 75), so a structurally-added DE arm was hobbled before it started;
        # ``"auto"`` lets the bandit see these strong optimizers at a fair size.
        (LSHADE, {"NP_init": "auto"}),  # adaptive DE w/ linear pop reduction
        (JSO, {"NP_init": "auto"}),  # CEC-2017 winner, weighted current-to-pbest-w/1
        (NLSHADE_RSP, {"NP_init": "auto", "k_rank": 3.0}),  # CEC-2021 winner, NLPSR + RSP
        (NLSHADE_LBC, {"NP_init": "auto", "k_rank": 3.0}),  # CEC-2022 winner, NLPSR + RSP + LBC
        (LSHADE_EpSin, {"NP_init": "auto", "mu_freq_init": 0.5}),  # ensemble-sinusoid F
        (COBYQA, {}),  # Powell-family derivative-free trust-region local optimizer
        # ``warm_start=True`` makes each restart (after the first box-centre
        # descent) polish a perturbation of the strategy's best incumbent
        # instead of a fresh uniform-random point — the memetic recipe scipy
        # ``dual_annealing`` owes its Rosenbrock win to.  A *structurally-added*
        # local optimizer sits in a portfolio whose other arms are actively
        # discovering good basins, so the uniform-restart geometry was exactly
        # wrong: the 2026-07-06 A/B measured that bolting *cold* LBFGSB onto
        # ``Rewarding_Diverse`` *regressed* the composite even though it halved
        # the Rosenbrock best-distance.  Warm restarts fix the geometry; a lean
        # 3-seed portfolio A/B on the curved-valley battery measured warm 0.198
        # vs cold 0.156.  See the module docstring of
        # :mod:`panobbgo.heuristics.lbfgsb` and the dated entry in
        # ``planning/SELF_IMPROVEMENT_LOG.md``.
        (LBFGSB, {"warm_start": True}),  # multi-start warm-started quasi-Newton local optimizer
        # CMA-ES (Hansen 2016, arXiv:1604.00772) is the only *covariance-
        # adapting* arm — it learns the full mutation covariance N(m, σ²C)
        # from successful steps and is invariant under rotations of the
        # search space, the exact regime (rotated ill-conditioned valleys,
        # 5-D MA-BBOB families) where the DE/PSO/pattern arms above are
        # weakest.  Default population λ = 4 + ⌊3·ln n⌋ (7 at n=2) is
        # already budget-appropriate for the quick battery, so no NP-style
        # sizing override is needed.  ``sigma0`` is set explicitly to the
        # heuristic default so the ``CMAES.sigma0`` kwarg rule in
        # :func:`default_catalog` fires on any spec the bandit builds.
        # Without a :class:`~panobbgo.analyzers.restart.Restart` analyzer
        # in the spec this runs as plain CMA-ES; if the bandit also adds
        # the Restart analyzer it becomes IPOP-CMA-ES (the heuristic's
        # ``on_restart`` doubles λ).  ``avoid_duplicates`` keeps at most
        # one CMAES per portfolio.
        (CMAES, {"sigma0": 0.3}),
    )
    # Analyzer candidate pool — narrowly curated.  ``Sensitivity``
    # (cheap adaptive tracking) and ``Restart`` (warm restarts on
    # stagnation) are the two analyzers most strategies in the default
    # battery already use; adding them to bare strategies is the
    # natural shipping target.  The remaining analyzers
    # (``Best`` / ``Convergence`` are auto-attached by the strategy
    # base class; ``Splitter`` / ``Grid`` / ``Dedensifyer`` are
    # research-grade special-purpose tools) are intentionally
    # excluded — the bandit should not add experimental analyzers
    # to a working spec without explicit opt-in.  ``Restart``'s
    # default kwargs match the canonical IPOP-CMA-ES wiring in
    # :func:`_make_standard_strategies`.  ``Sensitivity``'s
    # ``update_interval`` matches the standard-mode default.
    analyzer_candidates: Tuple[Tuple[type, Dict[str, Any]], ...] = (
        (Sensitivity, {"update_interval": 20}),
        (Restart, {"patience": None, "restart_strategy": "diverse", "max_restarts": 5}),
    )
    structural_rules: List[CatalogRule] = [
        StructuralMutationRule(
            strategy_pattern="",
            op="add_heuristic",
            candidate_classes=candidates,
            probability=0.3,
        ),
        StructuralMutationRule(
            strategy_pattern="",
            op="drop_heuristic",
            min_heuristics=2,
            probability=0.3,
        ),
        StructuralMutationRule(
            strategy_pattern="",
            op="add_analyzer",
            candidate_classes=analyzer_candidates,
            probability=0.3,
        ),
        StructuralMutationRule(
            strategy_pattern="",
            op="drop_analyzer",
            # Analyzers are non-essential — an empty list is valid.
            min_analyzers=0,
            probability=0.3,
        ),
    ]
    return MutationCatalog(base_rules + structural_rules)


# ---------------------------------------------------------------------------
# Applying a mutation
# ---------------------------------------------------------------------------


def apply_mutation(
    specs: Sequence[StrategySpec],
    proposal: MutationProposal,
) -> List[StrategySpec]:
    """Return a new list with ``proposal`` applied to the matching spec.

    Only the :class:`StrategySpec` whose name equals
    :attr:`MutationProposal.strategy_name` is rewritten.  Inside that
    spec, every ``kwargs`` dict is shallow-copied before mutation so the
    input list is untouched — this lets the loop keep the prior spec
    list around as the "fallback" when a proposal is rejected.

    Five proposal flavours are supported:

    * Hyperparameter retune (``proposal.op is None``) — overwrite the
      existing kwarg value at ``(class_name, param_name)``.
    * ``proposal.op == "add_heuristic"`` — append
      ``(proposal.class_name's class object, proposal.structural_kwargs)``
      to the strategy's heuristics list.  The class object is recovered
      by name from the existing heuristics or from ``structural_kwargs``
      that the catalog kept alive on the proposal.  See
      :func:`_make_structural_proposal`.
    * ``proposal.op == "drop_heuristic"`` — remove the first heuristic
      whose ``__name__`` equals ``proposal.class_name``.  The
      :attr:`StructuralMutationRule.min_heuristics` floor is enforced at
      *sample* time (in :func:`_find_structural_hits`); apply trusts the
      catalog and only re-checks the trivial "≥ 1 entry remains" floor
      so this function still refuses to produce an empty strategy.
    * ``proposal.op == "add_analyzer"`` — same as ``add_heuristic`` but
      targets the analyzers bucket.  The class object is resolved from
      :mod:`panobbgo.analyzers` via :func:`_resolve_analyzer_class`.
    * ``proposal.op == "drop_analyzer"`` — same as ``drop_heuristic``
      but targets the analyzers bucket.  The analyzers list is allowed
      to become empty (unlike heuristics) — strategies with no
      analyzers still run cleanly.

    Raises:
        ValueError: If the target strategy is absent, the target class
            cannot be located inside it (drop / kwarg paths), or the
            structural rule kept on the proposal is unrecoverable
            (add path with no class object reachable).
    """
    out: List[StrategySpec] = []
    applied = False
    for spec in specs:
        if spec.name != proposal.strategy_name:
            out.append(spec)
            continue

        new_heuristics = [(cls, dict(kw)) for cls, kw in spec.heuristics]
        new_analyzers = [(cls, dict(kw)) for cls, kw in spec.analyzers]

        if proposal.op == "add_heuristic":
            cls_obj = _resolve_heuristic_class(proposal, spec)
            new_kwargs = dict(proposal.structural_kwargs or {})
            new_heuristics.append((cls_obj, new_kwargs))
        elif proposal.op == "drop_heuristic":
            drop_idx = next(
                (i for i, (cls, _) in enumerate(new_heuristics) if cls.__name__ == proposal.class_name),
                None,
            )
            if drop_idx is None:
                raise ValueError(
                    f"drop_heuristic proposal targets class {proposal.class_name!r}"
                    f" but no heuristic of that name exists in {spec.name!r}"
                )
            if len(new_heuristics) <= 1:
                raise ValueError(f"drop_heuristic on {spec.name!r} would leave the strategy with no heuristics")
            new_heuristics.pop(drop_idx)
        elif proposal.op == "add_analyzer":
            cls_obj = _resolve_analyzer_class(proposal, spec)
            new_kwargs = dict(proposal.structural_kwargs or {})
            new_analyzers.append((cls_obj, new_kwargs))
        elif proposal.op == "drop_analyzer":
            drop_idx = next(
                (i for i, (cls, _) in enumerate(new_analyzers) if cls.__name__ == proposal.class_name),
                None,
            )
            if drop_idx is None:
                raise ValueError(
                    f"drop_analyzer proposal targets class {proposal.class_name!r}"
                    f" but no analyzer of that name exists in {spec.name!r}"
                )
            # Unlike heuristics, an empty analyzers list is a valid spec
            # — Sensitivity etc. are non-essential to running the strategy.
            new_analyzers.pop(drop_idx)
        else:
            hit = False
            for cls, kw in new_heuristics:
                if cls.__name__ == proposal.class_name and proposal.param_name in kw:
                    kw[proposal.param_name] = proposal.new_value
                    hit = True
                    break
            if not hit:
                for cls, kw in new_analyzers:
                    if cls.__name__ == proposal.class_name and proposal.param_name in kw:
                        kw[proposal.param_name] = proposal.new_value
                        hit = True
                        break
            if not hit:
                raise ValueError(
                    f"proposal target {proposal.class_name}.{proposal.param_name} not found in strategy {spec.name!r}"
                )

        applied = True
        out.append(
            StrategySpec(
                name=spec.name,
                strategy_class=spec.strategy_class,
                heuristics=new_heuristics,
                analyzers=new_analyzers,
                config_overrides=dict(spec.config_overrides),
            )
        )

    if not applied:
        raise ValueError(f"proposal refers to strategy {proposal.strategy_name!r} which is not in the input spec list")
    return out


def _is_no_op(baseline_result: HarnessResult, candidate_result: HarnessResult) -> bool:
    """Return True iff the candidate's per-pair scores are bit-identical to baseline.

    §12.4 of ``planning/SELF_IMPROVEMENT_LOOP.md``: an iteration that
    measured exact equality on every ``(problem, strategy)`` pair carries
    zero information about whether the mutation rule helps or hurts —
    pulling the bandit arm on the outcome would mis-train the posterior.
    The §2.1 V2 diagnosis ("34% of mutations measure Δ = exactly
    0.0000") makes this the dominant failure mode of the V1 loop:
    proposals targeting kwargs whose effect is invisible at the
    quick-mode budget produce a zero delta that the bandit currently
    counts as a real rejection.

    Bit-identical here means ``a == b`` in IEEE 754 — composite_score is
    a mean of solve-fractions so two runs that share their deterministic
    seed stream produce truly equal floats (no rounding hazard).  We
    compare per-pair scores rather than the composite to detect the
    case where the composite happens to round to the same value despite
    a real per-pair difference; that's a separate (and rare) coincidence
    that should still count as a real bandit pull.

    Pair identity is keyed on ``(problem_name, problem_dim,
    strategy_name)`` — the harness ships the same set of pairs in both
    measurements when invoked back-to-back with the same spec list, so
    the maps are guaranteed equal-keyed in the live loop path.  When
    keys mismatch (e.g. a structural proposal renamed a strategy), we
    conservatively return ``False`` — the iteration carries real
    information about whether the rename helps.
    """
    before = {
        (psr.problem_name, psr.problem_dim, psr.strategy_name): float(psr.score)
        for psr in baseline_result.problem_strategy_results
    }
    after = {
        (psr.problem_name, psr.problem_dim, psr.strategy_name): float(psr.score)
        for psr in candidate_result.problem_strategy_results
    }
    if not before or before.keys() != after.keys():
        return False
    return all(before[k] == after[k] for k in before)


def _pool_harness_results(*results: HarnessResult) -> HarnessResult:
    """Concatenate per-(problem, strategy) runs across two or more results.

    The §6.4 same-night confirmation gate re-measures a screening-accepted
    candidate on a fresh ``randomize_iteration`` (and, optionally, a
    hold-out ``base_seed``) and then re-runs
    :func:`~panobbgo.harness.statistical_accept` on the *pooled* sample.
    Pooling at the run level is the natural shape for the paired bootstrap:
    every concatenated rep is still an instance-aligned ``(problem,
    strategy)`` measurement, so a paired sampler with shared resample
    indices preserves the within-rep correlation it relies on for narrow
    CIs.  We keep the metric recomputation centralised on
    :meth:`~panobbgo.harness.ProblemStrategyResult.compute_metrics` so the
    pooled ``score`` / ``success_rate`` / ``ert`` are derived from the
    concatenated runs by the same formula the live harness uses.

    The composite ``composite_score`` is recomputed as the mean of the
    pooled per-pair ``score`` values — same definition the harness uses
    for the live result, so the pooled result is interchangeable with a
    fresh :class:`HarnessResult` everywhere the loop already consumes
    one (statistical_accept, _is_no_op, the ledger writer).

    Pairs are matched by ``(problem_name, problem_dim, strategy_name)``.
    Pairs present in only one input are kept with their original runs;
    this is conservative and matches what the live harness would produce
    if the missing side simply timed out on that pair.  An empty
    ``results`` tuple raises ``ValueError`` because the consumer needs a
    well-formed result back, not an opaque empty one.

    Args:
        *results: One or more :class:`HarnessResult` instances to pool.
            The first result's ``config`` is reused for the pooled
            result; ``total_runs`` / ``total_duration`` sum across inputs.

    Returns:
        A populated :class:`HarnessResult` whose per-pair runs are the
        concatenation of the inputs' runs and whose composite score is
        the mean of the recomputed per-pair scores.
    """
    if not results:
        raise ValueError("_pool_harness_results requires at least one result")
    if len(results) == 1:
        # Identity case — preserves byte-identical behaviour when the
        # confirmation step's caller passes a single result.
        return results[0]
    head = results[0]
    pooled_runs: Dict[Tuple[str, int, str], Tuple[ProblemStrategyResult, list]] = {}
    pair_order: List[Tuple[str, int, str]] = []
    for res in results:
        for psr in res.problem_strategy_results:
            key = (psr.problem_name, psr.problem_dim, psr.strategy_name)
            if key not in pooled_runs:
                # Reuse the first-seen pair's metadata (f_opt, tolerance,
                # budget) — these are properties of the (problem, strategy)
                # pair, not of any individual measurement.
                pooled_runs[key] = (psr, list(psr.runs))
                pair_order.append(key)
            else:
                pooled_runs[key][1].extend(psr.runs)

    pooled_psrs: List[ProblemStrategyResult] = []
    for key in pair_order:
        template, runs = pooled_runs[key]
        pooled_psr = ProblemStrategyResult(
            problem_name=template.problem_name,
            problem_dim=template.problem_dim,
            strategy_name=template.strategy_name,
            f_opt=template.f_opt,
            tolerance=template.tolerance,
            budget=template.budget,
            runs=runs,
        )
        pooled_psr.compute_metrics()
        pooled_psrs.append(pooled_psr)

    composite = float(np.mean([p.score for p in pooled_psrs])) if pooled_psrs else 0.0
    total_runs = sum(int(r.total_runs) for r in results)
    total_duration = float(sum(float(r.total_duration) for r in results))
    return HarnessResult(
        config=head.config,
        timestamp=head.timestamp,
        total_runs=total_runs,
        total_duration=total_duration,
        problem_strategy_results=pooled_psrs,
        composite_score=composite,
    )


def _compute_graded_reward(
    *,
    accepted: bool,
    delta: float,
    ci_low: float,
    eps_accept: float,
) -> float:
    """Return the graded bandit reward in ``[0, 1]`` per §7.4.

    Implements the formula spelt out in §7.4 of
    ``planning/SELF_IMPROVEMENT_LOOP.md``:

    * ``accepted`` → ``0.5 + clip(ci_low / eps_scale, 0, 0.5)``
      — barely confirmed accepts contribute ``~0.5``; clearly winning
      accepts (lower CI bound well above zero) contribute up to ``1.0``.
    * rejected → ``clip(0.5 + delta / eps_scale, 0, 0.5)`` — a positive
      sub-eps delta contributes ``~0.5`` ("honest near miss"), a delta
      at zero contributes ``0.5``, a clearly harmful delta contributes
      down to ``0``.

    where ``eps_scale = 4 · eps_accept`` (planning doc default).  The
    function never returns negative or >1 values — the clamps in the
    formula and a defensive sanitiser on ``eps_accept`` (any
    non-positive ``eps_accept`` collapses to ``1e-12`` so the divide is
    finite and the clamps still pin the output).

    Designed for the *informative* iteration path only: no-op iterations
    and skips are gated upstream by
    :meth:`AdaptiveMutationSampler.discard_outcome`, so this helper
    does not branch on those cases.
    """
    eps_scale = 4.0 * max(float(eps_accept), 1e-12)
    if accepted:
        bonus = float(ci_low) / eps_scale
        if bonus < 0.0:
            bonus = 0.0
        elif bonus > 0.5:
            bonus = 0.5
        return 0.5 + bonus
    value = 0.5 + float(delta) / eps_scale
    if value < 0.0:
        return 0.0
    if value > 0.5:
        return 0.5
    return value


def _resolve_heuristic_class(proposal: MutationProposal, spec: StrategySpec) -> type:
    """Recover the actual class object for an ``add_heuristic`` proposal.

    The catalog's :func:`_make_structural_proposal` records the
    heuristic's ``__name__``; the proposal carries the *string*, not the
    object.  We look it up against the spec's existing classes first
    (which covers the ``avoid_duplicates=False`` case where the same
    class is already present) and, failing that, walk the strategy's
    sibling specs the loop has just sampled.  In the common path the
    catalog hands us a class that is *not* yet in the spec — so we fall
    back to importing it via its registered location in
    :mod:`panobbgo.heuristics`.
    """
    name = proposal.class_name
    for cls, _ in spec.heuristics:
        if cls.__name__ == name:
            return cls
    # Fallback: import from the heuristics package by name.  Restrict to
    # names actually registered there to avoid eval-style class lookup.
    import panobbgo.heuristics as _h

    if hasattr(_h, name):
        candidate = getattr(_h, name)
        if isinstance(candidate, type):
            return candidate
    raise ValueError(
        f"add_heuristic proposal references class {name!r} which is not present in"
        f" strategy {spec.name!r} and not exported from panobbgo.heuristics"
    )


def _resolve_analyzer_class(proposal: MutationProposal, spec: StrategySpec) -> type:
    """Recover the actual class object for an ``add_analyzer`` proposal.

    Mirror of :func:`_resolve_heuristic_class` for the analyzers bucket.
    Looks up the class by name on the spec's existing analyzers first
    (covers the ``avoid_duplicates=False`` case), then falls back to
    :mod:`panobbgo.analyzers`'s registered re-exports.
    """
    name = proposal.class_name
    for cls, _ in spec.analyzers:
        if cls.__name__ == name:
            return cls
    # Fallback: import from the analyzers package by name.  Restrict to
    # names actually registered there to avoid eval-style class lookup.
    import panobbgo.analyzers as _a

    if hasattr(_a, name):
        candidate = getattr(_a, name)
        if isinstance(candidate, type):
            return candidate
    raise ValueError(
        f"add_analyzer proposal references class {name!r} which is not present in"
        f" strategy {spec.name!r} and not exported from panobbgo.analyzers"
    )


# ---------------------------------------------------------------------------
# Loop config and record
# ---------------------------------------------------------------------------


@dataclass
class LoopConfig:
    """Configuration for :class:`SelfImprover`.

    The defaults target the ``quick`` harness mode so the MVP loop is
    usable during local development.  Bump ``mode`` to ``"standard"`` and
    increase ``iterations`` for an overnight loop.

    Args:
        iterations: Number of loop passes.
        base_seed: Base seed forwarded to the harness for strategy
            reproducibility.
        mode: One of ``"quick"``, ``"standard"``, ``"full"`` — controls
            default problems, reps, and budget.
        reps: Override repetitions per ``(problem, strategy)``; ``None``
            uses the mode default.
        budget: Override evaluations per run; ``None`` uses the mode default.
        eps_accept: Minimum composite delta required to accept
            (see :func:`panobbgo.harness.statistical_accept` §6.2).
        eps_regress: Maximum tolerated per-pair regression.
        n_boot: Bootstrap resamples for the CI.  Fewer is faster; 2000 is
            already noise-limited under ``quick`` reps.
        confidence: Two-sided confidence level (e.g., ``0.95``).
        stat_seed: Base RNG seed for the bootstrap; the per-iteration
            seed is ``stat_seed + iteration`` so every iteration uses an
            independent stream.
        mutation_seed: RNG seed for the mutation sampler.
        strategy_names: Restrict to a subset of the mode's strategies.
            ``None`` uses all of them.
        ledger_path: Where the JSONL ledger is appended.  Parent dirs are
            created on demand.
        stop_sentinel_path: Before each iteration, the loop checks for
            this file.  If it exists, the loop stops gracefully.  Set to
            an empty string to disable.
        timeout_per_run: Per-run wall-clock cap passed to the harness.
        randomize: If ``True`` (default), the randomized problem battery
            is used so instances vary by iteration.  Set to ``False`` to
            run against the fixed modes (useful for debugging).
        guard_interval: How often (in completed iterations) to run the
            anti-cherry-pick guard from §6.3 of the plan.  ``0`` (default)
            disables the guard.  A positive value, typically ``5`` or
            ``10``, instructs the loop to re-measure the top of the
            accepted ladder on a *fresh* seed every ``guard_interval``
            iterations and roll the ladder back if the re-measurement
            drops more than :attr:`guard_eps_ladder` below the score that
            originally got the entry accepted.
        guard_eps_ladder: Tolerance for ladder drift detected by the
            guard.  If the re-measured composite is lower than the
            stored ``last_validated_score`` by more than this amount,
            the entry is rolled back.  Default ``0.02`` follows the plan.
        guard_iteration_offset: Offset added to the regular iteration id
            when deriving the guard's randomized seed.  Picking a large
            constant (default ``1_000_000``) keeps the guard's instance
            stream independent from the regular iteration stream so a
            mutation cannot accidentally tune itself to the guard's
            seeds.
        adaptive_sampling: If ``True``, the loop replaces uniform
            mutation sampling with :class:`AdaptiveMutationSampler` —
            Thompson sampling over a Beta posterior per rule.  Defaults
            to ``False`` so existing CLI invocations behave identically.
            With this flag, the loop biases future iterations toward
            rules that have produced accepts in the past while still
            exploring less-tried rules.
        adaptive_prior_alpha: Pseudo-count of "successes" in the Beta
            prior used by the adaptive sampler.  Has no effect unless
            :attr:`adaptive_sampling` is ``True``.
        adaptive_prior_beta: Pseudo-count of "failures" in the Beta
            prior; symmetric default ``1.0`` ⇒ Beta(1, 1) ≡ U(0, 1) so
            cold-start behaviour matches uniform sampling.
        adaptive_prime_from_ledger: When ``True``, the adaptive sampler
            seeds its bandit history from any existing
            :attr:`ledger_path` before the first iteration.  Useful when
            resuming a long unattended run.
        adaptive_prime_include_archives: When ``True`` *and*
            :attr:`adaptive_prime_from_ledger` is also ``True``, the
            sampler additionally primes from archived ledgers in
            :attr:`adaptive_prime_archive_dir` (default
            ``<dirname(ledger_path)>/done``) before the live ledger.
            Closes the §2.6 "archives in ``planning/done/`` are
            invisible" diagnosis: the bandit posterior accumulates
            evidence across every retained nightly run rather than
            only the current one.  Default ``False`` keeps existing
            CLI invocations byte-identical.  Only takes effect when
            :attr:`adaptive_sampling` and
            :attr:`adaptive_prime_from_ledger` are both ``True``.
        adaptive_prime_archive_dir: Directory to scan for archived
            JSONL ledgers when
            :attr:`adaptive_prime_include_archives` is ``True``.
            Files matching ``self_improve_ledger_*.jsonl`` are
            consumed in chronological (lexicographic) order.  ``None``
            (default) derives the directory from
            :attr:`ledger_path` as ``<parent>/done``.  A missing
            directory is a silent no-op so the flag is safe to enable
            on first-night runs.
        structural_per_class_arms: When ``True``, structural ops in the
            adaptive sampler are split into per-target-class bandit
            arms (e.g. adding ``Sobol`` lives on
            ``("Sobol", "add_heuristic", "structural")`` rather than
            collapsing into ``("*", "add_heuristic", "structural")``).
            Gives the loop sharper signal about *which* class to add /
            drop at the cost of sparser per-arm data.  Default
            ``False`` keeps the published 2026-05-03 semantics.
            Only takes effect when :attr:`adaptive_sampling` is also
            ``True``; the uniform-sampler path is unchanged.
        structural_borrow_alpha: Hierarchical "borrow" coefficient
            ``κ ≥ 0`` for per-class structural arms.  When > 0 and
            :attr:`structural_per_class_arms` is also ``True``, each
            per-class arm's Beta posterior borrows
            ``κ · (n_other_class_accepts, n_other_class_failures)``
            from the op-level aggregate (sum over all sibling per-class
            arms with the same op).  This closes the sample-efficiency
            gap that per-class arms introduce: a fresh candidate class
            starts with the op's empirical accept rate rather than the
            symmetric :math:`\\mathrm{Beta}(1, 1)` prior.  ``0.0``
            (default) keeps the pure per-class semantics shipped
            2026-05-18; ``0.5`` weights each sibling accept at half a
            local accept; ``1.0`` weights them equally.  Inert when
            :attr:`structural_per_class_arms` is ``False`` (no per-class
            arms exist to borrow from each other) or when
            :attr:`adaptive_sampling` is ``False``.  See
            :attr:`structural_borrow_horizon` for the auto-tune knob
            that anneals ``κ`` down as per-class evidence accumulates.
        structural_borrow_horizon: Optional adaptive annealing horizon
            ``h > 0`` for the hierarchical borrow coefficient.  When
            ``> 0`` (and the two preconditions for borrow itself —
            :attr:`structural_borrow_alpha` ``> 0`` and
            :attr:`structural_per_class_arms` ``= True`` — are also
            met), each per-class arm's effective borrow shrinks toward
            zero as its own attempts accumulate::

                κ_eff = κ / (1 + n_class_attempts / h)

            So a cold arm borrows the full ``κ`` (same as the
            non-annealed path); a saturated arm
            (``n_class_attempts >> h``) effectively trusts its own
            per-class posterior.  Closes the *Auto-tune κ* follow-up:
            "borrow heavily early, vanish as evidence grows" so the
            hierarchy stops dragging well-evidenced arms back toward
            the op-level mean indefinitely.  Default ``0.0`` disables
            annealing (every arm always borrows the full ``κ``),
            byte-identical to the 2026-06-01 ship.  Recommended values
            for an unattended cron: ``5`` to ``10`` (the per-arm
            posteriors warm up over a couple of nights).  Inert when
            :attr:`structural_borrow_alpha` ``= 0`` (no borrow to
            anneal),  :attr:`structural_per_class_arms` ``= False``,
            or :attr:`adaptive_sampling` ``= False``.
        holdout_base_seed: Independent ``base_seed`` used for the
            end-of-loop hold-out validation.  ``0`` (default) disables
            hold-out entirely (unless :attr:`holdout_base_seeds` is set).
            Should differ from :attr:`base_seed` — using the same value
            collapses the hold-out check to the anti-cherry-pick guard
            with offset ``0`` and is rejected at validation time.  See
            "Hold-out validation set" in the module docstring for the full
            rationale.  This scalar knob is retained for back-compat;
            multi-seed callers should prefer :attr:`holdout_base_seeds`.
        holdout_base_seeds: Multi-seed hold-out validation.  Each entry is
            an independent ``base_seed`` value the loop will re-measure
            the ladder against at the end of the run.  Empty tuple
            (default) means "fall back to the scalar
            :attr:`holdout_base_seed`".  When non-empty, the scalar is
            ignored.  Reducing across the per-seed records uses ``min``
            on drift (worst-case generalisation) and ``any`` on overfit
            (one bad seed flags the ladder).  All entries must differ
            from :attr:`base_seed` and from one another; ``0`` is not a
            valid entry (use the empty tuple to disable).
        holdout_iterations: Number of distinct ``randomize_iteration``
            values to average over when computing the hold-out composite
            for both the seed and the top ladder entries.  Must be
            non-negative.  ``0`` (with ``holdout_base_seed != 0``) is a
            valid no-op.
        holdout_iteration_offset: Starting ``randomize_iteration`` index
            for the hold-out sweep.  Almost always ``0`` is fine because
            ``holdout_base_seed`` already gives an independent SHA-256
            stream, but the knob exists for users who want to walk a
            specific window of instances (e.g. for replication studies).
        holdout_eps_overfit: Drift tolerance on the
            ``(top − seed)`` gap.  When the hold-out gap is smaller than
            the training-time gap by more than this amount, the
            :class:`LoopHoldoutRecord` is flagged ``overfit=True``.
            Default ``0.05`` matches the per-pair regression bound used
            by the statistical-acceptance rule.
        paired: Bootstrap scheme passed to
            :func:`panobbgo.harness.statistical_accept`.  ``None``
            (default) uses auto-detection: when the randomized harness
            keeps reps instance-aligned by index (the common case under
            :attr:`randomize` ``= True``), the paired scheme preserves
            the strong within-rep correlation between baseline and
            candidate and shrinks the CI substantially compared to the
            independent-resample scheme.  Set to ``False`` to force the
            historical unpaired sampler.
        inactivity_relax_after: When positive, the loop temporarily
            relaxes :attr:`eps_accept` after this many consecutive
            non-accept iterations to break out of long droughts.  The
            effective threshold decays geometrically by
            :attr:`inactivity_relax_factor` for every additional
            :attr:`inactivity_relax_after` non-accepts, floored at
            :attr:`inactivity_min_eps_accept`.  The decay resets to the
            configured :attr:`eps_accept` on the next accept.  ``0``
            (default) disables relaxation, preserving the historical
            behaviour byte-for-byte.  Skip-iterations (no applicable
            mutation) count toward the inactivity streak.  See §6.2 /
            §10 "inactivity-guarded loop productivity" in the planning
            doc for the rationale: the bandit's posterior only updates
            on observed accepts/rejects, so very long droughts leave
            the sampler effectively uninformed.  Recommended values for
            an unattended cron: ``inactivity_relax_after=10`` (start
            relaxing after a typical iteration window worth of misses),
            ``inactivity_relax_factor=0.5`` (halve each step), and
            ``inactivity_min_eps_accept=0.001`` (don't drop below the
            statistical-accept noise floor).
        inactivity_relax_factor: Multiplicative factor applied to
            :attr:`eps_accept` for each :attr:`inactivity_relax_after`
            block of consecutive non-accepts.  Must satisfy
            ``0 < factor < 1`` — values outside this range either don't
            relax at all (``1.0``) or amplify the threshold (``> 1``)
            which would be the opposite of what this knob is for.
            Ignored when :attr:`inactivity_relax_after` ``= 0``.
        inactivity_min_eps_accept: Lower bound on the relaxed
            :attr:`eps_accept`.  The geometric decay never drops the
            effective threshold below this floor, so reviewers can be
            sure a relaxed accept still beats a baseline-grade signal.
            Must be non-negative and ``<= eps_accept``.  Default
            ``0.001`` matches the noise floor the bootstrap CI can
            reliably resolve at typical quick-mode rep counts.  Ignored
            when :attr:`inactivity_relax_after` ``= 0``.
    """

    iterations: int = 5
    base_seed: int = 42
    mode: str = "quick"
    reps: Optional[int] = None
    budget: Optional[int] = None
    eps_accept: float = 0.005
    eps_regress: float = 0.05
    n_boot: int = 2000
    confidence: float = 0.95
    stat_seed: int = 42
    mutation_seed: int = 0
    strategy_names: Optional[List[str]] = None
    ledger_path: str = "planning/self_improve_ledger.jsonl"
    stop_sentinel_path: str = "STOP_SELF_IMPROVE"
    timeout_per_run: Optional[float] = 120.0
    randomize: bool = True
    guard_interval: int = 0
    guard_eps_ladder: float = 0.02
    guard_iteration_offset: int = 1_000_000
    adaptive_sampling: bool = False
    adaptive_prior_alpha: float = 1.0
    adaptive_prior_beta: float = 1.0
    adaptive_prime_from_ledger: bool = False
    adaptive_prime_include_archives: bool = False
    adaptive_prime_archive_dir: Optional[str] = None
    structural_per_class_arms: bool = False
    structural_borrow_alpha: float = 0.0
    structural_borrow_horizon: float = 0.0
    holdout_base_seed: int = 0
    holdout_base_seeds: Tuple[int, ...] = ()
    holdout_iterations: int = 5
    holdout_iteration_offset: int = 0
    holdout_eps_overfit: float = 0.05
    paired: Optional[bool] = None
    inactivity_relax_after: int = 0
    inactivity_relax_factor: float = 0.5
    inactivity_min_eps_accept: float = 0.001
    #: Which scoring metric drives accept/reject decisions.
    #:
    #: ``"composite"`` (default) — score on Panobbgo's own problem
    #: battery using :class:`~panobbgo.harness.BenchmarkHarness`.  This
    #: is the historical behaviour; ledger ``baseline_score`` /
    #: ``candidate_score`` carry ``composite_score`` values.
    #:
    #: ``"aocc"`` — score on the IOH/MA-BBOB battery using
    #: :func:`~panobbgo.harness_ioh.run_ioh_harness`.  Per-iteration
    #: measurement runs :func:`~panobbgo.harness_ioh.make_quick_battery`
    #: (mode-dependent — quick/standard/full map to the IOH battery of
    #: the same name) and the result is adapted via
    #: :func:`~panobbgo.harness_ioh.aocc_to_harness_result` so the
    #: existing bootstrap CI / ledger / guard machinery works
    #: unchanged.  Ledger scores carry **mean AOCC** in the same fields.
    #:
    #: Use ``"aocc"`` when the loop should optimise for the MA-BBOB
    #: anytime competition rather than panobbgo's internal score.
    metric: str = "composite"
    #: Named seed-strategy registry.
    #:
    #: ``"default"`` (historical) selects the strategy battery from the
    #: harness mode — quick / standard / full mapping to the matching
    #: :func:`_make_*_strategies` factory in :mod:`panobbgo.harness`.
    #:
    #: ``"loop"`` selects :func:`~panobbgo.harness._make_loop_strategies`
    #: regardless of :attr:`mode`.  The loop registry ships the quick
    #: specs plus one compact spec per rule-bearing family (DE / PSO /
    #: RegionUCB / LBFGSB+COBYQA / Restart analyzer) with every tunable
    #: kwarg explicit at the constructor default, so the ~30 mutation
    #: rules in :func:`default_catalog` actually fire on the seed instead
    #: of staying dormant.  See §9.1 of
    #: ``planning/SELF_IMPROVEMENT_LOOP.md`` (V2 plan).
    #:
    #: Ignored on the AOCC metric path — the IOH battery has its own
    #: registry (:func:`panobbgo.harness_ioh.make_ioh_strategies`).
    registry: str = "default"
    #: Opt-in extra randomized families appended to the default 2-D battery
    #: on the **composite** metric path (see
    #: :attr:`panobbgo.harness.HarnessConfig.extra_families`).  ``None``
    #: (default) keeps the historical 2-D-only battery.  Set to
    #: :func:`~panobbgo.harness_randomized.make_highdim_families` (CLI
    #: ``--extra-highdim``) so the loop, guard, and hold-out all measure a
    #: rotated higher-dimensional regime the default battery cannot reach —
    #: e.g. to confirm/reward a change (such as the 2026-07-12
    #: diagonal-plus-low-rank ``Nearby`` Hessian) whose benefit only
    #: manifests above dim 2.  Inert on the AOCC metric path (that battery
    #: is defined by :mod:`panobbgo.harness_ioh`, not by randomized
    #: families).
    extra_families: Optional[List["ProblemFamily"]] = None
    #: Bandit reward shaping policy.
    #:
    #: ``"binary"`` (default) — the bandit's Beta posterior updates with
    #: ``+1`` on accept and ``+0`` on reject (the historical behaviour).
    #: At realistic per-night iteration counts (20-40), this delivers a
    #: ~2.5% base accept rate so most arms accumulate no positive
    #: evidence and the posterior stays close to the prior — the §2.6
    #: V2 diagnosis identified this as the "bandit starved" failure
    #: mode.  Preserved as the default so existing CLI invocations are
    #: byte-identical.
    #:
    #: ``"graded"`` — implements §7.4 of
    #: ``planning/SELF_IMPROVEMENT_LOOP.md``.  Each iteration's reward
    #: is a continuous function of the bootstrap CI and point delta:
    #:
    #: * ``no-op`` → no posterior update (already gated by
    #:   :meth:`AdaptiveMutationSampler.discard_outcome` since
    #:   2026-06-12).
    #: * accepted → ``0.5 + clip(ci_low / (4·eps_accept), 0, 0.5)`` —
    #:   between ``0.5`` (barely confirmed) and ``1.0`` (clearly
    #:   winning).
    #: * rejected → ``clip(0.5 + Δ / (4·eps_accept), 0, 0.5)`` —
    #:   between ``0`` (clearly harmful) and ``0.5`` (a positive but
    #:   sub-eps delta, "honest near miss").
    #:
    #: This converts every iteration into informative evidence on the
    #: chosen arm, so arms that consistently produce small-positive
    #: deltas become distinguishable from arms that produce harmful
    #: deltas — a property the binary path cannot achieve at realistic
    #: per-night iteration counts.
    #:
    #: Only takes effect when :attr:`adaptive_sampling` is also
    #: ``True``; the uniform-sampler path does not pull arms.
    bandit_reward_shaping: str = "binary"
    #: Same-night confirmation gate (§6.4 of
    #: ``planning/SELF_IMPROVEMENT_LOOP.md``).
    #:
    #: ``False`` (default) — the loop promotes a screening-accepted
    #: candidate straight onto the ladder; the anti-cherry-pick guard is
    #: the only downstream check.  This is the V1 behaviour and the
    #: §2.2 V2 diagnosis identified it as the dominant accept-rollback
    #: source: with a ~2.5% accept rate, 15/16 accepts roll back on the
    #: next guard pass because the screening CI was driven by an
    #: upward-noise spike on the one instance batch the iteration
    #: happened to draw.
    #:
    #: ``True`` — every screening-accepted candidate is re-measured on
    #: a fresh ``randomize_iteration`` (``iteration +
    #: :attr:`confirm_iteration_offset```) and the
    #: :func:`~panobbgo.harness.statistical_accept` rule is re-run on
    #: the *pooled* (screen + confirm) sample.  Promotion happens only
    #: when the pooled paired CI stays above ``eps_accept`` — a noise
    #: spike on the screening batch can no longer drive an accept
    #: because the confirmation batch is independent.  When the pooled
    #: CI fails the gate the iteration is recorded with ``confirmed =
    #: False`` and an extra ``LoopConfirmRecord`` (``record_type =
    #: "confirm_reject"``) carries the screen + confirm scores so an
    #: auditor can trace why the promotion was held back.  The bandit's
    #: reward path uses the *post-confirmation* delta / CI so an
    #: arm that consistently produces noise-spike accepts gets the
    #: weak-signal reward it actually deserves rather than the
    #: full-accept reward the screening would have given it.
    #:
    #: When :attr:`holdout_base_seed` / :attr:`holdout_base_seeds` are
    #: also configured, the confirmation step additionally re-measures
    #: on the *first* hold-out base_seed at ``randomize_iteration =
    #: iteration + :attr:`confirm_iteration_offset``` and pools that
    #: too — the planning doc's "fresh ``randomize_iteration`` *and*
    #: hold-out base_seed" prescription.  Only the first seed is used
    #: per iteration to keep the per-iteration compute cost bounded at
    #: ``≤ 3×`` the screening cost regardless of how many hold-out
    #: seeds are configured; the end-of-loop hold-out continues to
    #: walk every configured seed.
    confirm_accepts: bool = False
    #: ``randomize_iteration`` offset used by the §6.4 confirmation
    #: gate.  The confirmation batch sees instances drawn from
    #: ``iteration + confirm_iteration_offset`` so the screening and
    #: confirmation batches are independent SHA-256 streams.  Default
    #: ``500_000`` sits between the regular iteration stream (``0..N``)
    #: and the guard's offset (``1_000_000``) so the three streams
    #: never collide at realistic iteration counts.  Inert when
    #: :attr:`confirm_accepts` is ``False``.
    confirm_iteration_offset: int = 500_000
    #: Run every AOCC measurement with the synchronous-harvest
    #: evaluation mode (``config.sync_evaluation``, shipped 2026-08-09).
    #: Blocking until every in-flight future is harvested removes the
    #: scheduling nondeterminism that dominates the loop's noise floor:
    #: the measured single-measurement AOCC sd on the quick battery
    #: drops from ~0.0101 to ~0.0063 (1.6x).  Costs a little wall-clock
    #: per run because the strategy can no longer overlap generation
    #: with evaluation.
    #:
    #: Inert under ``metric="composite"`` — the composite harness has
    #: its own evaluation path and is not plumbed for this flag.
    #:
    #: Defaults to ``False`` so existing invocations stay byte-identical;
    #: ``scripts/self_improve.py run --sync-eval`` and the nightly
    #: workflow turn it on.  **Never compare a sync-eval measurement
    #: against a non-sync-eval one** — the noise floors differ, so a
    #: mixed-mode A/B reads the mode change as a spec effect.
    sync_eval: bool = False
    #: Extra dimensions appended to the AOCC battery for *every*
    #: measurement the loop makes — screening, confirm, guard, hold-out.
    #:
    #: The mode presets are frozen contracts, so this composes a widened
    #: battery via :func:`~panobbgo.harness_ioh.with_extra_dims` instead
    #: of editing them.  ``(5,)`` on the quick preset turns the nightly's
    #: ``dims=(2,)`` regime into ``dims=(2, 5)``.
    #:
    #: Motivation: the loop samples quick-2-D, but the two sharpest
    #: measured results of 2026-08 both lived at d5 — the JSO d5 add
    #: (2026-08-02) and the NLSHADE_LBC per-dim split (2026-08-11, d2
    #: −0.0241 against d5 +0.0080, both CIs excluding zero).  A regime
    #: the loop cannot see is a regime it cannot optimise, and worse, a
    #: change that helps there reads as noise or as a loss in the
    #: aggregate.
    #:
    #: Cost scales super-linearly: ``budget_for`` is
    #: ``budget_multiplier * dim``, so adding dim 5 to the quick battery
    #: doubles the run count *and* the added runs are 2.5x longer.
    #:
    #: Empty (default) leaves every existing invocation byte-identical.
    #: Inert under ``metric="composite"``, which reaches higher dims
    #: through ``--extra-highdim`` / :attr:`extra_families` instead.
    aocc_extra_dims: Tuple[int, ...] = ()
    #: Which statistic gates the accept decision — forwarded to
    #: :func:`panobbgo.harness.statistical_accept` at both the screening
    #: and the §6.4 confirmation call.
    #:
    #: ``"mean"`` (default, historical) uses the bootstrap CI on the mean
    #: per-pair delta.  ``"rank"`` uses a one-sided Wilcoxon signed-rank
    #: test on the per-pair deltas shifted by ``eps_accept``
    #: (``GOAL.md`` §5.3).
    #:
    #: The motivation is specific: a mean over pairs lets one lucky pair
    #: carry the composite past the bar, and AOCC deltas on a small
    #: battery are exactly the kind of heavy-tailed sample where that
    #: happens.  The rank test asks whether the change wins *typically*.
    #: Ledger ``delta`` stays the mean under both modes so the series
    #: remains comparable across the switch.
    accept_stat: str = "mean"

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {self.iterations}")
        if self.mode not in {"quick", "standard", "full"}:
            raise ValueError(f"Unknown mode {self.mode!r}")
        if self.metric not in {"composite", "aocc"}:
            raise ValueError(f"metric must be 'composite' or 'aocc', got {self.metric!r}")
        if self.registry not in {"default", "loop"}:
            raise ValueError(f"registry must be 'default' or 'loop', got {self.registry!r}")
        if self.accept_stat not in {"mean", "rank"}:
            raise ValueError(f"accept_stat must be 'mean' or 'rank', got {self.accept_stat!r}")
        if self.bandit_reward_shaping not in {"binary", "graded"}:
            raise ValueError(f"bandit_reward_shaping must be 'binary' or 'graded', got {self.bandit_reward_shaping!r}")
        if self.confirm_iteration_offset <= 0:
            raise ValueError(f"confirm_iteration_offset must be > 0, got {self.confirm_iteration_offset}")
        # The confirmation gate (§6.4) and the anti-cherry-pick guard
        # (§6.3) both derive a "fresh" ``randomize_iteration`` from the
        # current iteration plus an offset.  If the two offsets collide
        # (or land within a window the regular iteration stream walks
        # through), the confirm batch and the guard batch would see
        # bit-identical instances, defeating the point of the second
        # check.  ``500_000`` and ``1_000_000`` are the planning-doc
        # defaults; the validation enforces the offsets stay distinct.
        if self.confirm_accepts and self.confirm_iteration_offset == self.guard_iteration_offset:
            raise ValueError(
                "confirm_iteration_offset must differ from guard_iteration_offset"
                f" (both={self.confirm_iteration_offset});"
                " the confirmation and guard fresh seeds are meant to draw from independent streams"
            )
        if self.guard_interval < 0:
            raise ValueError(f"guard_interval must be >= 0, got {self.guard_interval}")
        if self.guard_eps_ladder < 0:
            raise ValueError(f"guard_eps_ladder must be >= 0, got {self.guard_eps_ladder}")
        if self.adaptive_prior_alpha <= 0:
            raise ValueError(f"adaptive_prior_alpha must be > 0, got {self.adaptive_prior_alpha}")
        if self.adaptive_prior_beta <= 0:
            raise ValueError(f"adaptive_prior_beta must be > 0, got {self.adaptive_prior_beta}")
        if self.structural_borrow_alpha < 0:
            raise ValueError(f"structural_borrow_alpha must be >= 0, got {self.structural_borrow_alpha}")
        if self.structural_borrow_horizon < 0:
            raise ValueError(f"structural_borrow_horizon must be >= 0, got {self.structural_borrow_horizon}")
        if self.holdout_iterations < 0:
            raise ValueError(f"holdout_iterations must be >= 0, got {self.holdout_iterations}")
        if self.holdout_eps_overfit < 0:
            raise ValueError(f"holdout_eps_overfit must be >= 0, got {self.holdout_eps_overfit}")
        # An independent base_seed is the entire point — silently treating
        # equal values as "ok" would let the loop ship a hold-out that
        # collapses to a glorified guard check, which is exactly the
        # measurement gap this feature exists to close.
        if self.holdout_base_seed != 0 and self.holdout_base_seed == self.base_seed:
            raise ValueError(
                f"holdout_base_seed must differ from base_seed (both={self.base_seed});"
                " hold-out is meant to draw from a *different* SHA-256 stream"
            )
        # Normalize a list/sequence argument into a tuple so callers may pass
        # either a tuple or a list and the dataclass stays hashable-safe.
        # ``Sequence[int]`` would type the field correctly but a tuple is
        # the simplest concrete shape the rest of the code relies on.
        self.holdout_base_seeds = tuple(int(s) for s in self.holdout_base_seeds)
        if self.holdout_base_seeds:
            # Per-entry constraints: no 0 sentinel, no collision with the
            # training base_seed, and no duplicates within the list.  Each
            # rule has a distinct failure mode the user should hear about.
            if any(s == 0 for s in self.holdout_base_seeds):
                raise ValueError(
                    "holdout_base_seeds entries must be non-zero; pass an empty tuple to disable multi-seed hold-out"
                )
            collisions = [s for s in self.holdout_base_seeds if s == self.base_seed]
            if collisions:
                raise ValueError(
                    f"holdout_base_seeds must differ from base_seed (overlap={collisions});"
                    " hold-out is meant to draw from *different* SHA-256 streams"
                )
            if len(set(self.holdout_base_seeds)) != len(self.holdout_base_seeds):
                raise ValueError(
                    f"holdout_base_seeds must be distinct, got {list(self.holdout_base_seeds)};"
                    " duplicates would just re-measure the same SHA-256 stream"
                )
        # Inactivity-relax knobs.  The three parameters interact, so we
        # validate them together: ``inactivity_relax_after = 0`` disables
        # the feature and the other two are unused; once enabled we
        # require ``0 < factor < 1`` (anything else either doesn't relax
        # or amplifies, both pointless) and a non-negative floor that
        # doesn't already exceed the configured threshold.
        if self.inactivity_relax_after < 0:
            raise ValueError(f"inactivity_relax_after must be >= 0, got {self.inactivity_relax_after}")
        if self.inactivity_relax_after > 0:
            if not (0.0 < self.inactivity_relax_factor < 1.0):
                raise ValueError(
                    "inactivity_relax_factor must be in (0, 1) when relaxation is enabled,"
                    f" got {self.inactivity_relax_factor}"
                )
            if self.inactivity_min_eps_accept < 0:
                raise ValueError(f"inactivity_min_eps_accept must be >= 0, got {self.inactivity_min_eps_accept}")
            if self.inactivity_min_eps_accept > self.eps_accept:
                raise ValueError(
                    "inactivity_min_eps_accept must be <= eps_accept"
                    f" (floor={self.inactivity_min_eps_accept} > eps={self.eps_accept});"
                    " the floor exists so a relaxed accept still beats a baseline-grade signal"
                )

    def effective_eps_accept(self, iters_since_accept: int) -> float:
        """Return the eps_accept that the loop should use right now.

        ``iters_since_accept`` is the number of consecutive iterations
        without an accept *before* the current one (so on the first
        iteration the counter is 0 and the full :attr:`eps_accept` is
        used).  Every full :attr:`inactivity_relax_after` block of
        non-accepts halves the threshold by
        :attr:`inactivity_relax_factor`, floored at
        :attr:`inactivity_min_eps_accept`.  When
        :attr:`inactivity_relax_after` ``= 0`` this is a constant
        :attr:`eps_accept`, which preserves the historical behaviour
        byte-for-byte.
        """
        if self.inactivity_relax_after <= 0:
            return float(self.eps_accept)
        if iters_since_accept < self.inactivity_relax_after:
            return float(self.eps_accept)
        steps = iters_since_accept // self.inactivity_relax_after
        relaxed = self.eps_accept * (self.inactivity_relax_factor**steps)
        return float(max(relaxed, self.inactivity_min_eps_accept))

    def harness_config(
        self,
        strategies_override: List[StrategySpec],
        iteration_id: int,
    ) -> HarnessConfig:
        """Build the :class:`HarnessConfig` used to measure one spec list."""
        return HarnessConfig(
            mode=self.mode,
            budget=self.budget,
            reps=self.reps,
            seed=self.base_seed,
            strategies=self.strategy_names,
            timeout_per_run=self.timeout_per_run,
            randomize=self.randomize,
            randomize_iteration=iteration_id,
            strategies_override=strategies_override,
            extra_families=self.extra_families,
        )

    def holdout_harness_config(
        self,
        strategies_override: List[StrategySpec],
        iteration_id: int,
        base_seed: Optional[int] = None,
    ) -> HarnessConfig:
        """Build the :class:`HarnessConfig` used by hold-out validation.

        Identical to :meth:`harness_config` except that ``seed`` is taken
        either from the explicit ``base_seed`` argument (used by the
        multi-seed path) or from :attr:`holdout_base_seed` when not
        provided.  This swaps the SHA-256 instance stream wholesale,
        giving the hold-out a genuinely independent set of randomized
        problems.
        """
        return HarnessConfig(
            mode=self.mode,
            budget=self.budget,
            reps=self.reps,
            seed=self.holdout_base_seed if base_seed is None else int(base_seed),
            strategies=self.strategy_names,
            timeout_per_run=self.timeout_per_run,
            randomize=self.randomize,
            randomize_iteration=iteration_id,
            strategies_override=strategies_override,
            extra_families=self.extra_families,
        )

    def resolved_holdout_seeds(self) -> Tuple[int, ...]:
        """Effective tuple of hold-out base seeds.

        Multi-seed (:attr:`holdout_base_seeds`) wins when set so callers
        opting into the list don't have to also clear the scalar.  Otherwise
        the scalar :attr:`holdout_base_seed` is promoted to a 1-tuple when
        non-zero, or the empty tuple when both knobs are at their defaults
        (= hold-out disabled).
        """
        if self.holdout_base_seeds:
            return self.holdout_base_seeds
        if self.holdout_base_seed != 0:
            return (int(self.holdout_base_seed),)
        return ()


@dataclass
class LoopIterationRecord:
    """One ledger line — the full trace of a single loop iteration."""

    iteration: int
    timestamp: str
    duration_seconds: float
    proposal: Optional[Dict[str, Any]]
    accepted: bool
    baseline_score: float
    candidate_score: float
    delta: float
    ci_low: float
    ci_high: float
    worst_pair_regression: float
    worst_pair: Optional[Tuple[str, str]]
    reasons: List[str] = field(default_factory=list)
    base_seed: int = 42
    randomize_iteration: int = 0
    mode: str = "quick"
    reason_skipped: Optional[str] = None
    record_type: str = "iteration"
    #: Effective ``eps_accept`` used for this iteration's accept gate.
    #:
    #: ``None`` (default) on records produced before the
    #: 2026-05-30 inactivity-relax ship — the threshold was always
    #: :attr:`LoopConfig.eps_accept` so the field was implicit.  On
    #: newer records this carries the actual value
    #: :func:`panobbgo.harness.statistical_accept` saw, which can be
    #: lower than ``LoopConfig.eps_accept`` when relaxation kicked in
    #: after :attr:`LoopConfig.inactivity_relax_after` consecutive
    #: non-accepts.  Persisted so an auditor can replay the loop with
    #: the same effective rule.
    effective_eps_accept: Optional[float] = None
    #: Consecutive non-accept iterations seen *before* this iteration
    #: started (i.e. the streak that the relax rule consulted to compute
    #: :attr:`effective_eps_accept`).  ``0`` on the very first iteration
    #: of a run.  ``None`` on legacy records.
    iters_since_accept: Optional[int] = None
    #: True iff this iteration's per-(problem, strategy) candidate scores
    #: were bit-identical to baseline — the proposal touched a kwarg that
    #: produced no measurable behavioural difference at the current
    #: budget.  Recorded so the bandit does not pull an arm on a
    #: zero-information event and the summary can distinguish these from
    #: genuine rejections.  ``False`` on legacy records and on records
    #: written before the 2026-06-12 ship; the §12.4 telemetry rule from
    #: ``planning/SELF_IMPROVEMENT_LOOP.md`` defines the post-measure
    #: detection.
    no_op: bool = False
    #: Graded bandit reward in ``[0, 1]`` accumulated into the adaptive
    #: sampler's :attr:`MutationRuleStats.reward_sum` for this
    #: iteration.  ``None`` on (a) skip / no-op iterations where the
    #: bandit is not pulled, (b) iterations from runs using the legacy
    #: binary-reward path (``LoopConfig.bandit_reward_shaping =
    #: "binary"`` — the default), and (c) legacy ledger records written
    #: before the 2026-06-13 §7.4 ship.  When non-``None`` carries the
    #: actual reward the bandit received so
    #: :meth:`AdaptiveMutationSampler.prime_from_ledger` can replay the
    #: graded posterior bit-exactly.  See §7.4 of
    #: ``planning/SELF_IMPROVEMENT_LOOP.md`` for the formula.
    bandit_reward: Optional[float] = None
    #: §6.4 same-night confirmation outcome.
    #:
    #: * ``None`` — no confirmation step was run for this iteration:
    #:   either ``LoopConfig.confirm_accepts`` was ``False`` (the
    #:   default, historical behaviour), or the iteration was a skip /
    #:   no-op, or the screening decision was already a reject (the
    #:   gate only runs after a screening-accept).  Legacy ledger
    #:   records written before the 2026-06-14 ship default to ``None``
    #:   so they replay byte-identically.
    #: * ``True`` — the confirmation step ran and the pooled (screen +
    #:   confirm) bootstrap CI cleared the accept gate; the candidate
    #:   was promoted to the ladder.  :attr:`accepted` is ``True`` in
    #:   this case.
    #: * ``False`` — the confirmation step ran and the pooled CI failed
    #:   the gate; the candidate was *not* promoted.  :attr:`accepted`
    #:   is ``False`` and the same-night ledger carries a companion
    #:   :class:`LoopConfirmRecord` (``record_type="confirm_reject"``)
    #:   that records the screen + confirm scores so the failure is
    #:   auditable.
    confirmed: Optional[bool] = None
    #: Whether this iteration was measured with the synchronous-harvest
    #: evaluation mode (:attr:`LoopConfig.sync_eval`).  ``False`` on
    #: legacy records, which were all measured asynchronously.
    #:
    #: Recorded because sync and async measurements have *different
    #: noise floors* (sd ~0.0063 vs ~0.0101 on the AOCC quick battery),
    #: so pooling nights across the boundary — as cross-night codify
    #: evidence does — mixes two sampling distributions and produces a
    #: pooled CI narrower than either mode justifies.  Consumers that
    #: aggregate across nights should group by this field.
    sync_eval: bool = False
    #: Dimensions appended to the mode's AOCC battery for this iteration
    #: (:attr:`LoopConfig.aocc_extra_dims`).  Empty on legacy records and
    #: on composite runs.
    #:
    #: Recorded for the same reason as :attr:`sync_eval`: widening the
    #: battery moves the *level* of the score, not just its noise —
    #: measured on the quick preset, d2 alone reads 0.3685 while
    #: (d2, d5) reads ~0.31 — so a night before the widening and a night
    #: after are not on one scale.  Cross-night consumers must group by
    #: this field as well.
    aocc_extra_dims: Tuple[int, ...] = ()
    #: Which statistic gated this iteration's verdict — ``"mean"`` or
    #: ``"rank"`` (:attr:`LoopConfig.accept_stat`).  ``"mean"`` on legacy
    #: records, which predate the rank rule.
    accept_stat: str = "mean"
    #: One-sided Wilcoxon p-value behind a ``"rank"`` verdict; ``None``
    #: under the mean rule.  Persisted so a rank-gated accept can be
    #: audited without re-running the measurement.
    rank_p: Optional[float] = None
    #: Hodges-Lehmann per-pair delta behind a ``"rank"`` verdict;
    #: ``None`` under the mean rule.  Note this is *not* what
    #: :attr:`delta` carries — ``delta`` stays the mean under both rules
    #: so the ledger series is continuous across a rule switch.
    rank_delta: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "record_type": self.record_type,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "proposal": self.proposal,
            "accepted": self.accepted,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "worst_pair_regression": self.worst_pair_regression,
            "worst_pair": (list(self.worst_pair) if self.worst_pair is not None else None),
            "reasons": list(self.reasons),
            "base_seed": self.base_seed,
            "randomize_iteration": self.randomize_iteration,
            "mode": self.mode,
            "reason_skipped": self.reason_skipped,
            "effective_eps_accept": self.effective_eps_accept,
            "iters_since_accept": self.iters_since_accept,
            "no_op": self.no_op,
            "bandit_reward": self.bandit_reward,
            "confirmed": self.confirmed,
            "sync_eval": self.sync_eval,
            "aocc_extra_dims": list(self.aocc_extra_dims),
            "accept_stat": self.accept_stat,
            "rank_p": self.rank_p,
            "rank_delta": self.rank_delta,
        }
        return d


@dataclass
class LadderEntry:
    """An entry in the accepted-mutation ladder maintained by :class:`SelfImprover`.

    The ladder is the running record of strategy spec lists the loop has
    promoted.  Entry ``-1`` is the seed (the spec list the loop started
    with); subsequent entries record each accepted mutation.  The
    anti-cherry-pick guard from §6.3 of the plan operates on this list:
    it re-measures the top entry on a fresh randomized seed and pops it
    if the score has drifted below :attr:`last_validated_score` by more
    than :attr:`LoopConfig.guard_eps_ladder`.

    Attributes:
        iteration: Iteration index that produced this entry, or ``-1``
            for the seed.
        specs: The :class:`StrategySpec` list snapshot.
        last_validated_score: The composite score most recently observed
            for this entry.  Refreshed each time the guard validates the
            entry, so it tracks the *current* expected performance, not
            just the historical one.
        proposal: The mutation that produced this entry, or ``None`` for
            the seed.  Used for ledger output.
    """

    iteration: int
    specs: List[StrategySpec]
    last_validated_score: float
    proposal: Optional[Dict[str, Any]] = None


@dataclass
class LoopGuardRecord:
    """One ledger line — outcome of a single anti-cherry-pick guard check.

    The guard is invoked every :attr:`LoopConfig.guard_interval`
    iterations.  It re-measures the top of the ladder on a *fresh* seed
    (``iteration + LoopConfig.guard_iteration_offset``) and rolls the
    ladder back if the re-measured score has drifted too far below the
    stored :attr:`LadderEntry.last_validated_score`.

    Attributes:
        iteration: Iteration index after which the guard ran (the
            iteration that *triggered* it).
        timestamp: ISO-8601 UTC timestamp.
        duration_seconds: Wall-clock cost of the guard, including any
            re-measurements performed during rollback.
        guard_score: Re-measured composite score for the top entry of
            the ladder before any rollback.
        pre_guard_top_score: ``last_validated_score`` of the top entry
            before the guard ran.
        pre_guard_top_iteration: ``iteration`` of the top entry before
            the guard ran (``-1`` for the seed).
        rolled_back: Whether the guard popped at least one ladder entry.
        rolled_back_to_iteration: Iteration index of the new top after
            rollback (``-1`` for the seed), or ``None`` if no rollback
            happened.
        pops: Number of ladder entries popped during rollback.
        ladder_size_before: Length of the ladder before the guard.
        ladder_size_after: Length of the ladder after the guard.
        guard_iteration_id: ``randomize_iteration`` used by the harness
            for the guard re-measurement (i.e. the "fresh" seed).
        reasons: Human-readable bullet points.
        base_seed: Loop's base seed (for ledger search).
        mode: Harness mode used for the guard.
        record_type: Always ``"guard"``; lets ledger consumers
            distinguish guard records from regular iteration records.
    """

    iteration: int
    timestamp: str
    duration_seconds: float
    guard_score: float
    pre_guard_top_score: float
    pre_guard_top_iteration: int
    rolled_back: bool
    rolled_back_to_iteration: Optional[int]
    pops: int
    ladder_size_before: int
    ladder_size_after: int
    guard_iteration_id: int
    reasons: List[str] = field(default_factory=list)
    base_seed: int = 42
    mode: str = "quick"
    record_type: str = "guard"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_type": self.record_type,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "guard_score": self.guard_score,
            "pre_guard_top_score": self.pre_guard_top_score,
            "pre_guard_top_iteration": self.pre_guard_top_iteration,
            "rolled_back": self.rolled_back,
            "rolled_back_to_iteration": self.rolled_back_to_iteration,
            "pops": self.pops,
            "ladder_size_before": self.ladder_size_before,
            "ladder_size_after": self.ladder_size_after,
            "guard_iteration_id": self.guard_iteration_id,
            "reasons": list(self.reasons),
            "base_seed": self.base_seed,
            "mode": self.mode,
        }


@dataclass
class LoopConfirmRecord:
    """One ledger line — outcome of a §6.4 confirmation gate rejection.

    Only written when :attr:`LoopConfig.confirm_accepts` is ``True`` *and*
    the same-night confirmation step rejected a screening-accepted
    candidate.  Successful confirmations leave the surrounding
    :class:`LoopIterationRecord` carrying ``accepted=True`` /
    ``confirmed=True`` and need no companion record; failed confirmations
    additionally append this record so the screen + confirm scores and
    the pooled CI are auditable from the ledger alone.

    Attributes:
        iteration: Iteration index whose screening accept the gate
            overturned.
        timestamp: ISO-8601 UTC timestamp recorded when the confirmation
            decision landed.
        duration_seconds: Wall-clock cost of the confirmation
            measurements (baseline + candidate, summed across the
            randomize-iteration and any hold-out re-measurement).
        proposal: Serialised :class:`MutationProposal` that the gate
            rejected, identical to the surrounding iteration record's
            ``proposal`` field so a JSONL consumer can match the two on
            ``(iteration, proposal["rule_key"])``.
        screen_baseline_score: ``HarnessResult.composite_score`` of the
            *baseline* (pre-mutation) measurement on the screening
            randomize_iteration.
        screen_candidate_score: ``HarnessResult.composite_score`` of the
            *candidate* (post-mutation) measurement on the screening
            randomize_iteration — same value as
            :attr:`LoopIterationRecord.candidate_score`.
        screen_delta: ``screen_candidate_score − screen_baseline_score``,
            cached for ledger consumers.
        confirm_baseline_score: ``HarnessResult.composite_score`` of the
            *baseline* measurement on the confirmation
            randomize_iteration (``iteration +
            LoopConfig.confirm_iteration_offset``).
        confirm_candidate_score: ``HarnessResult.composite_score`` of the
            *candidate* measurement on the confirmation iteration.
        confirm_delta: ``confirm_candidate_score −
            confirm_baseline_score``.
        pooled_delta: Composite delta of
            :func:`~panobbgo.harness.statistical_accept` run on the
            *pooled* (screen + confirm) sample.  This is the value the
            gate evaluated against ``eps_accept`` to decide promotion.
        pooled_ci_low: Lower bound of the pooled bootstrap CI; the gate
            requires ``pooled_ci_low > 0`` to promote.
        pooled_ci_high: Upper bound of the pooled bootstrap CI.
        pooled_worst_pair_regression: Most-negative per-pair delta on
            the pooled sample.  When this dips below
            ``-LoopConfig.eps_regress`` the gate rejects even if the
            mean CI clears.
        pooled_worst_pair: ``(problem, strategy)`` pair carrying
            :attr:`pooled_worst_pair_regression`, or ``None`` when no
            pair regressed.
        confirm_iteration_id: ``randomize_iteration`` used for the
            confirmation re-measurement (i.e., ``iteration +
            LoopConfig.confirm_iteration_offset``).
        confirm_holdout_seed: ``base_seed`` of the optional hold-out
            confirmation re-measurement, or ``None`` when no hold-out
            seed was configured / used.
        confirm_holdout_baseline_score: Mean composite of the *baseline*
            measurement on the hold-out base_seed, or ``None`` when no
            hold-out confirmation ran.
        confirm_holdout_candidate_score: Mean composite of the
            *candidate* measurement on the hold-out base_seed, or
            ``None`` when no hold-out confirmation ran.
        reasons: Human-readable bullet points (mirrors the
            :class:`~panobbgo.harness.StatisticalDecision` reasons list).
        base_seed: Loop's training base_seed (for ledger search).
        mode: Harness mode used for the confirmation measurements.
        record_type: Always ``"confirm_reject"``; lets ledger consumers
            distinguish confirmation rejections from iteration / guard /
            hold-out records.
    """

    iteration: int
    timestamp: str
    duration_seconds: float
    proposal: Optional[Dict[str, Any]]
    screen_baseline_score: float
    screen_candidate_score: float
    screen_delta: float
    confirm_baseline_score: float
    confirm_candidate_score: float
    confirm_delta: float
    pooled_delta: float
    pooled_ci_low: float
    pooled_ci_high: float
    pooled_worst_pair_regression: float
    pooled_worst_pair: Optional[Tuple[str, str]]
    confirm_iteration_id: int
    confirm_holdout_seed: Optional[int] = None
    confirm_holdout_baseline_score: Optional[float] = None
    confirm_holdout_candidate_score: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    base_seed: int = 42
    mode: str = "quick"
    record_type: str = "confirm_reject"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_type": self.record_type,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "proposal": self.proposal,
            "screen_baseline_score": float(self.screen_baseline_score),
            "screen_candidate_score": float(self.screen_candidate_score),
            "screen_delta": float(self.screen_delta),
            "confirm_baseline_score": float(self.confirm_baseline_score),
            "confirm_candidate_score": float(self.confirm_candidate_score),
            "confirm_delta": float(self.confirm_delta),
            "pooled_delta": float(self.pooled_delta),
            "pooled_ci_low": float(self.pooled_ci_low),
            "pooled_ci_high": float(self.pooled_ci_high),
            "pooled_worst_pair_regression": float(self.pooled_worst_pair_regression),
            "pooled_worst_pair": (list(self.pooled_worst_pair) if self.pooled_worst_pair is not None else None),
            "confirm_iteration_id": int(self.confirm_iteration_id),
            "confirm_holdout_seed": (int(self.confirm_holdout_seed) if self.confirm_holdout_seed is not None else None),
            "confirm_holdout_baseline_score": (
                float(self.confirm_holdout_baseline_score) if self.confirm_holdout_baseline_score is not None else None
            ),
            "confirm_holdout_candidate_score": (
                float(self.confirm_holdout_candidate_score)
                if self.confirm_holdout_candidate_score is not None
                else None
            ),
            "reasons": list(self.reasons),
            "base_seed": int(self.base_seed),
            "mode": self.mode,
        }


@dataclass
class LoopHoldoutRecord:
    """One ledger line — outcome of the end-of-loop hold-out validation.

    Run once at the end of :meth:`SelfImprover.run` when
    :attr:`LoopConfig.holdout_base_seed` is non-zero and
    :attr:`LoopConfig.holdout_iterations` is positive.  The hold-out
    re-measures both the **seed** ladder entry and the **final top**
    entry on instances drawn from a completely independent
    ``base_seed`` SHA-256 stream, then compares the on-hold-out gap
    ``top − seed`` to the on-training gap.  A drop of more than
    :attr:`LoopConfig.holdout_eps_overfit` is flagged as overfitting.

    Unlike the anti-cherry-pick guard, the hold-out:

    * runs **once** at the end of the loop (cheap), and
    * uses an **independent base_seed** rather than the training
      base_seed with an iteration offset, so it catches overfit to the
      base_seed family that the guard cannot see.

    Attributes:
        timestamp: ISO-8601 UTC timestamp.
        duration_seconds: Wall-clock cost of all hold-out re-measurements.
        holdout_base_seed: The independent ``base_seed`` used for the
            hold-out.
        holdout_iterations: Number of distinct ``randomize_iteration``
            values averaged for both seed and top.
        holdout_iteration_offset: Starting iteration_id for the sweep
            (passed through from :attr:`LoopConfig`).
        seed_holdout_score: Mean composite of the **seed** ladder entry
            evaluated on the hold-out instances.
        top_holdout_score: Mean composite of the **top** ladder entry
            evaluated on the hold-out instances.  Equals
            :attr:`seed_holdout_score` when the ladder has only the seed
            (no accepted mutations).
        seed_training_score: ``last_validated_score`` of the seed entry
            recorded during the training loop.  ``NaN`` when the seed
            never got a baseline measurement (e.g. a zero-iteration
            run).
        top_training_score: ``last_validated_score`` of the top ladder
            entry at the moment the hold-out runs.  Equals
            :attr:`seed_training_score` when the ladder has only the
            seed.
        holdout_delta: ``top_holdout_score − seed_holdout_score``.  This
            is the *generalisation* effect size of all accepted
            mutations on a held-out instance family.
        training_delta: ``top_training_score − seed_training_score``.
            This is the *training* effect size; the loop's claimed
            improvement.
        drift: ``holdout_delta − training_delta``.  Negative values
            mean the gain shrank on hold-out (overfit); zero means it
            generalised exactly; positive means the hold-out happened
            to like the mutation even more than the training set
            (within noise, treated as fine).
        overfit: ``True`` iff ``drift < -eps_overfit``.
        eps_overfit: Tolerance from :attr:`LoopConfig.holdout_eps_overfit`.
        top_iteration: Iteration index of the final ladder top, ``-1``
            for the seed.
        ladder_size: Length of the ladder at the moment hold-out ran.
        base_seed: The loop's *training* base_seed (for cross-reference
            against the iteration ledger).
        mode: Harness mode used for the hold-out (same as the loop).
        reasons: Human-readable bullet points: skipped, overfit, or
            generalised.
        seed_iteration_scores: Per-iteration composite scores of the
            **seed** ladder entry on the hold-out instances.  Length
            equals :attr:`holdout_iterations` when the record was
            written by the current code path; an empty list signals a
            legacy record that only carries the aggregate
            :attr:`seed_holdout_score` (so :func:`aggregate_holdout_drift`
            falls back to per-record point estimates).
        top_iteration_scores: Per-iteration composite scores of the
            **top** ladder entry, paired index-by-index with
            :attr:`seed_iteration_scores`.  When the ladder has only
            the seed entry both lists carry the same values (the seed
            measurement is reused for top).
        record_type: Always ``"holdout"``; lets ledger consumers
            distinguish hold-out records from iteration and guard
            records.
        status: One of ``"ok"`` (the drift stayed within
            ``eps_overfit`` — improvement appears to generalise),
            ``"overfit"`` (drift below ``-eps_overfit`` — the ladder
            appears to have overfit the training base_seed family) or
            ``"vacuous"`` (the ladder had only the seed entry, so no
            accepted mutations existed to validate — ``holdout_delta``,
            ``training_delta`` and ``drift`` are all ``0.0`` by
            construction and the record carries **no** generalisation
            signal).  Introduced 2026-06-11 per
            `planning/SELF_IMPROVEMENT_LOOP.md` §6.4 / §12.4 so vacuous
            records are no longer reported as ``OK drift=+0.0000`` —
            see the dated entry in `planning/SELF_IMPROVEMENT_LOG.md`.
            Legacy records (written before this field existed) carry
            ``status="ok"`` by the dataclass default; downstream
            consumers that need the legacy-aware verdict should use
            :meth:`effective_status` instead.
    """

    #: Set of permissible :attr:`status` values.  Constructor validates
    #: against this set so a typo in a downstream caller fails loudly
    #: rather than silently producing an unrecognised verdict.
    SUPPORTED_STATUSES: ClassVar[Tuple[str, ...]] = ("ok", "overfit", "vacuous")

    timestamp: str
    duration_seconds: float
    holdout_base_seed: int
    holdout_iterations: int
    holdout_iteration_offset: int
    seed_holdout_score: float
    top_holdout_score: float
    seed_training_score: float
    top_training_score: float
    holdout_delta: float
    training_delta: float
    drift: float
    overfit: bool
    eps_overfit: float
    top_iteration: int
    ladder_size: int
    base_seed: int = 42
    mode: str = "quick"
    reasons: List[str] = field(default_factory=list)
    seed_iteration_scores: List[float] = field(default_factory=list)
    top_iteration_scores: List[float] = field(default_factory=list)
    record_type: str = "holdout"
    status: str = "ok"

    def __post_init__(self) -> None:
        if self.status not in self.SUPPORTED_STATUSES:
            raise ValueError(f"status must be one of {self.SUPPORTED_STATUSES}, got {self.status!r}")

    def effective_status(self) -> str:
        """Status with legacy fallback for records written before §12.4.

        Records written before the :attr:`status` field shipped
        (2026-06-11) all carry the dataclass default ``"ok"``, even when
        they were vacuous (empty ladder) or overfit.  This helper
        derives the right verdict from the other fields when the
        explicit status is ``"ok"`` but the structural conditions for
        a non-``"ok"`` verdict are present:

        * ``ladder_size <= 1`` and ``top_iteration == -1`` → ``"vacuous"``
          (the loop ran but never accepted a mutation, so the hold-out
          measured the seed against itself).
        * ``overfit=True`` → ``"overfit"`` (mirrors the boolean flag,
          which legacy records always have correctly).
        * otherwise ``"ok"``.

        New records carry an explicit non-``"ok"`` status so this helper
        is a no-op on them — kept for ledger-replay paths that read
        old JSONL lines.
        """
        if self.status != "ok":
            return self.status
        if self.ladder_size <= 1 and self.top_iteration < 0:
            return "vacuous"
        if self.overfit:
            return "overfit"
        return "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_type": self.record_type,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "holdout_base_seed": int(self.holdout_base_seed),
            "holdout_iterations": int(self.holdout_iterations),
            "holdout_iteration_offset": int(self.holdout_iteration_offset),
            "seed_holdout_score": float(self.seed_holdout_score),
            "top_holdout_score": float(self.top_holdout_score),
            "seed_training_score": float(self.seed_training_score),
            "top_training_score": float(self.top_training_score),
            "holdout_delta": float(self.holdout_delta),
            "training_delta": float(self.training_delta),
            "drift": float(self.drift),
            "overfit": bool(self.overfit),
            "eps_overfit": float(self.eps_overfit),
            "top_iteration": int(self.top_iteration),
            "ladder_size": int(self.ladder_size),
            "base_seed": int(self.base_seed),
            "mode": self.mode,
            "reasons": list(self.reasons),
            "seed_iteration_scores": [float(x) for x in self.seed_iteration_scores],
            "top_iteration_scores": [float(x) for x in self.top_iteration_scores],
            "status": str(self.status),
        }


# ---------------------------------------------------------------------------
# Bootstrap-CI aggregation across multi-seed hold-out records
# (planning/SELF_IMPROVEMENT_LOOP.md §13 — *Bootstrap CI on the drift
# estimate* follow-up to the multi-seed hold-out shipped 2026-05-16).
# ---------------------------------------------------------------------------


@dataclass
class HoldoutDriftAggregate:
    """Bootstrap CI on the aggregated hold-out drift across seeds.

    Produced by :func:`aggregate_holdout_drift`.  The aggregation pools
    per-iteration paired drift samples (one sample per ``(record, k)``
    where ``k`` indexes the hold-out iteration) and bootstrap-resamples
    the mean.  This turns the single-seed point check from 2026-05-08
    and the worst-case reduction from 2026-05-16 into a *statistical
    test* — pairs naturally with the existing :func:`statistical_accept`
    rule in :mod:`panobbgo.harness`.

    Attributes:
        mean_drift: Mean of all pooled per-iteration drift values
            (records × iterations).  When records lack per-iteration
            scores (legacy ledger lines), the per-record point drift is
            used as a single sample contribution.
        ci_low: Lower bound of the bootstrap CI on the mean drift.
        ci_high: Upper bound of the bootstrap CI on the mean drift.
        worst_drift: Most negative per-record point drift across the
            input records — same reduction the CLI already prints.
            Preserved here so callers can show point + CI side-by-side.
        worst_seed: ``holdout_base_seed`` of the worst-drift record.
        any_overfit: ``True`` iff at least one input record is flagged
            ``overfit=True``.  Mirrors the existing CLI semantics.
        overfit_count: Number of input records flagged ``overfit=True``.
        n_samples: Total pooled drift samples used for the bootstrap
            (records × iterations when per-iteration scores are
            present; otherwise records).
        n_records: Number of input :class:`LoopHoldoutRecord` instances.
        confidence: Confidence level used for the bootstrap CI.
        eps_overfit: Tolerance from
            :attr:`LoopConfig.holdout_eps_overfit` for "statistically
            significant overfit" downstream — the aggregate is flagged
            iff ``ci_high < -eps_overfit`` (the upper bound of the CI
            falls below the negative tolerance, i.e. even the
            *optimistic* end of the CI says we drifted).
        statistically_overfit: ``ci_high < -eps_overfit`` — a stronger
            verdict than per-record ``any_overfit``; tripping this means
            the bootstrap CI rules out drift better than ``-eps_overfit``
            at the configured confidence level.
        vacuous_count: Number of input records whose effective status
            is ``"vacuous"`` (empty-ladder hold-outs).  These are
            *excluded* from the bootstrap and the worst-drift reduction
            because their drift is ``0.0`` by construction and would
            otherwise pull the CI toward zero and mask a single
            negative-drift seed.  ``vacuous_count == n_records`` means
            no informative records exist; the aggregate is degenerate.
        all_vacuous: ``True`` iff every input record was vacuous (so
            the bootstrap had nothing to sample).  Callers should treat
            this the same way they treat empty input: ``mean_drift``
            and CI are both ``0.0`` but carry no signal.
    """

    mean_drift: float
    ci_low: float
    ci_high: float
    worst_drift: float
    worst_seed: int
    any_overfit: bool
    overfit_count: int
    n_samples: int
    n_records: int
    confidence: float
    eps_overfit: float
    statistically_overfit: bool
    vacuous_count: int = 0
    all_vacuous: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_drift": float(self.mean_drift),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "worst_drift": float(self.worst_drift),
            "worst_seed": int(self.worst_seed),
            "any_overfit": bool(self.any_overfit),
            "overfit_count": int(self.overfit_count),
            "n_samples": int(self.n_samples),
            "n_records": int(self.n_records),
            "confidence": float(self.confidence),
            "eps_overfit": float(self.eps_overfit),
            "statistically_overfit": bool(self.statistically_overfit),
            "vacuous_count": int(self.vacuous_count),
            "all_vacuous": bool(self.all_vacuous),
        }


def aggregate_holdout_drift(
    records: Sequence["LoopHoldoutRecord"],
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
    eps_overfit: Optional[float] = None,
) -> HoldoutDriftAggregate:
    """Bootstrap-CI aggregation across multi-seed hold-out records.

    Each record contributes per-iteration paired drift samples::

        drift_{r, k} = (top_holdout_{r, k} - seed_holdout_{r, k}) - training_delta_r

    where ``training_delta_r`` is the *per-record* training-time
    improvement gap (so the comparison is correctly paired against the
    training-time baseline each seed actually saw).  The pooled samples
    are bootstrap-resampled to produce a CI on the mean drift.

    Why "per-iteration" not "per-record"?  With 3 seeds the per-record
    sample size is 3 — bootstrap CIs on three points are nearly
    point-estimate-shaped and carry no real information.  Each record's
    per-iteration scores already exist (``holdout_iterations`` of them,
    default 5), so pooling exposes ``3 × 5 = 15`` paired drift samples
    to the bootstrap — enough that the CI quantiles are meaningful
    without changing the loop's measurement cost.

    Legacy records (written before the per-iteration scores were
    persisted) fall back to a *single* point-drift contribution per
    record.  Mixed inputs work too: any record with non-empty per-iter
    score lists contributes them; the rest contribute one point each.

    Args:
        records: Sequence of :class:`LoopHoldoutRecord` from a single
            loop run.  Empty input is allowed and produces a degenerate
            zero-drift aggregate with no CI.
        n_boot: Number of bootstrap resamples.  Default ``10_000``
            matches :func:`statistical_accept`.
        confidence: Two-sided confidence level for the CI quantiles.
            Default ``0.95``.
        seed: Base RNG seed for the bootstrap (reproducibility).
        eps_overfit: Tolerance for the statistical-overfit verdict.
            ``None`` (default) reads it from the first input record's
            :attr:`LoopHoldoutRecord.eps_overfit`; pass an explicit
            value to override.

    Returns:
        A populated :class:`HoldoutDriftAggregate`.  When ``records`` is
        empty, returns an all-zero aggregate that is safe to print but
        carries no information.
    """
    if not records:
        return HoldoutDriftAggregate(
            mean_drift=0.0,
            ci_low=0.0,
            ci_high=0.0,
            worst_drift=0.0,
            worst_seed=0,
            any_overfit=False,
            overfit_count=0,
            n_samples=0,
            n_records=0,
            confidence=float(confidence),
            eps_overfit=float(eps_overfit if eps_overfit is not None else 0.0),
            statistically_overfit=False,
            vacuous_count=0,
            all_vacuous=False,
        )

    # Vacuous records (empty-ladder hold-outs) contribute ``drift = 0.0``
    # by construction — pooling them would pull the CI toward zero and
    # mask a single negative-drift seed.  Filter them out of the
    # bootstrap and the worst-drift reduction but keep the count for the
    # caller so the aggregate stays auditable.  See V2 §6.4 / §12.4 of
    # `planning/SELF_IMPROVEMENT_LOOP.md`.
    vacuous_count = sum(1 for r in records if r.effective_status() == "vacuous")
    informative = [r for r in records if r.effective_status() != "vacuous"]
    all_vacuous = vacuous_count == len(records)
    eps = float(eps_overfit) if eps_overfit is not None else float(records[0].eps_overfit)

    if not informative:
        # Every record was vacuous — degenerate aggregate with no signal,
        # mirroring the empty-input case but recording the vacuous count
        # so the operator can see the loop ran without accepting any
        # mutation.
        return HoldoutDriftAggregate(
            mean_drift=0.0,
            ci_low=0.0,
            ci_high=0.0,
            worst_drift=0.0,
            worst_seed=int(records[0].holdout_base_seed),
            any_overfit=False,
            overfit_count=0,
            n_samples=0,
            n_records=len(records),
            confidence=float(confidence),
            eps_overfit=eps,
            statistically_overfit=False,
            vacuous_count=vacuous_count,
            all_vacuous=True,
        )

    # Pool per-iteration drifts.  When a record carries paired iter
    # scores we use them (the "high-resolution" contribution); otherwise
    # fall back to the cached point drift (legacy record).
    samples: List[float] = []
    for rec in informative:
        n_pair = min(len(rec.seed_iteration_scores), len(rec.top_iteration_scores))
        if n_pair > 0:
            seed_arr = np.asarray(rec.seed_iteration_scores[:n_pair], dtype=np.float64)
            top_arr = np.asarray(rec.top_iteration_scores[:n_pair], dtype=np.float64)
            iter_drifts = (top_arr - seed_arr) - float(rec.training_delta)
            samples.extend(float(x) for x in iter_drifts)
        else:
            # Legacy record — one point per record.  Better than dropping
            # it entirely; documents the limitation in the n_samples count.
            samples.append(float(rec.drift))

    arr = np.asarray(samples, dtype=np.float64)
    mean_drift = float(arr.mean()) if arr.size else 0.0
    n_samples = int(arr.size)

    if n_samples >= 2 and n_boot > 0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n_samples, size=(int(n_boot), n_samples))
        boots = arr[idx].mean(axis=1)
        alpha = (1.0 - float(confidence)) / 2.0
        ci_low = float(np.quantile(boots, alpha))
        ci_high = float(np.quantile(boots, 1.0 - alpha))
    else:
        # n_samples == 1 — the bootstrap would just resample the same
        # value n_boot times.  Skip the work and return a degenerate
        # CI equal to the point estimate.
        ci_low = mean_drift
        ci_high = mean_drift

    worst_rec = min(informative, key=lambda r: float(r.drift))
    any_overfit = any(bool(r.overfit) for r in informative)
    overfit_count = sum(1 for r in informative if bool(r.overfit))
    # "Statistically significant overfit": the *upper* end of the CI
    # is still below the negative tolerance — i.e. even the optimistic
    # bootstrap resample says we drifted worse than -eps_overfit.  The
    # point check (any_overfit) fires on a single bad seed; the CI
    # check is stricter and only fires when the aggregate is bad on
    # the optimistic end of the noise envelope.
    statistically_overfit = bool(ci_high < -eps)

    return HoldoutDriftAggregate(
        mean_drift=mean_drift,
        ci_low=ci_low,
        ci_high=ci_high,
        worst_drift=float(worst_rec.drift),
        worst_seed=int(worst_rec.holdout_base_seed),
        any_overfit=any_overfit,
        overfit_count=overfit_count,
        n_samples=n_samples,
        n_records=len(records),
        confidence=float(confidence),
        eps_overfit=eps,
        statistically_overfit=statistically_overfit,
        vacuous_count=vacuous_count,
        all_vacuous=all_vacuous,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class _ConfirmationOutcome:
    """Internal return type of :meth:`SelfImprover._run_confirmation`.

    Bundles the pooled
    :class:`~panobbgo.harness.StatisticalDecision`, the verdict, the
    fresh-iteration / hold-out-seed identifiers used by the
    confirmation step, and (when the gate rejected) the
    :class:`LoopConfirmRecord` to write to the ledger.  Keeping the
    return shape in a dataclass instead of a tuple keeps the caller in
    :meth:`SelfImprover._run_loop` readable and lets the test suite
    introspect the confirmation step directly.
    """

    confirmed: bool
    pooled_decision: Any  # StatisticalDecision; Any avoids a forward-ref cycle.
    confirm_iteration_id: int
    confirm_holdout_seed: Optional[int]
    record: Optional[LoopConfirmRecord]


class SelfImprover:
    """The loop driver.

    Usage::

        from panobbgo.self_improve import SelfImprover, LoopConfig
        records = SelfImprover(LoopConfig(iterations=10)).run()

    Args:
        config: :class:`LoopConfig`.  Defaults to a ``quick`` 5-iteration
            loop with the randomized battery enabled.
        catalog: :class:`MutationCatalog`.  Defaults to
            :func:`default_catalog`.
        seed_strategies: Initial strategy list.  ``None`` (default) pulls
            the mode's default specs from
            :meth:`BenchmarkHarness.get_strategies`, respecting
            ``LoopConfig.strategy_names``.
    """

    def __init__(
        self,
        config: Optional[LoopConfig] = None,
        catalog: Optional[MutationCatalog] = None,
        seed_strategies: Optional[Sequence[StrategySpec]] = None,
        sampler: Optional[AdaptiveMutationSampler] = None,
    ) -> None:
        self.config = config or LoopConfig()
        self.catalog = catalog or default_catalog()
        self._seed_strategies: Optional[List[StrategySpec]] = (
            list(seed_strategies) if seed_strategies is not None else None
        )
        # The adaptive sampler is constructed lazily when requested.  An
        # explicit instance always wins so tests / callers can pass a
        # pre-primed sampler.
        if sampler is not None:
            self.sampler: Optional[AdaptiveMutationSampler] = sampler
        elif self.config.adaptive_sampling:
            self.sampler = AdaptiveMutationSampler(
                self.catalog,
                prior_alpha=self.config.adaptive_prior_alpha,
                prior_beta=self.config.adaptive_prior_beta,
                per_class_structural=self.config.structural_per_class_arms,
                structural_borrow_alpha=self.config.structural_borrow_alpha,
                structural_borrow_horizon=self.config.structural_borrow_horizon,
            )
            if self.config.adaptive_prime_from_ledger:
                if self.config.adaptive_prime_include_archives:
                    archive_dir = self.config.adaptive_prime_archive_dir
                    if archive_dir is None:
                        archive_dir = str(pathlib.Path(self.config.ledger_path).parent / "done")
                    # Scope archive priming to the active metric so an aocc
                    # run warms only from aocc archives (§12.1 routing).
                    self.sampler.prime_from_archives(archive_dir, ledger_path=self.config.ledger_path)
                self.sampler.prime_from_ledger(self.config.ledger_path)
        else:
            self.sampler = None
        # Late-bound so tests can swap a fake harness in.
        self._harness_factory = BenchmarkHarness

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, verbose: bool = False) -> List[LoopIterationRecord]:
        """Run the loop and return the per-iteration records.

        Guard and hold-out records (:class:`LoopGuardRecord`,
        :class:`LoopHoldoutRecord`) are written to the ledger alongside
        iteration records but are not returned here — the contract of
        :meth:`run` is unchanged for backward compatibility.  Use
        :meth:`run_with_guard_records` for ``(iter, guard)`` or
        :meth:`run_full` for ``(iter, guard, holdout)`` when those are
        wanted in-process.  Reading the ledger recovers all three.
        """
        records, _, _ = self._run_internal(verbose=verbose)
        return records

    def run_with_guard_records(self, verbose: bool = False) -> Tuple[List[LoopIterationRecord], List[LoopGuardRecord]]:
        """Run the loop and return ``(iteration_records, guard_records)``.

        Hold-out records, when produced, are persisted to the ledger
        but not returned by this method.  Use :meth:`run_full` to also
        receive the hold-out record in-process.
        """
        records, guards, _ = self._run_internal(verbose=verbose)
        return records, guards

    def run_full(
        self, verbose: bool = False
    ) -> Tuple[List[LoopIterationRecord], List[LoopGuardRecord], List["LoopHoldoutRecord"]]:
        """Run the loop and return all three record streams.

        Returns ``(iteration_records, guard_records, holdout_records)``.
        The hold-out list is empty when hold-out is disabled
        (``LoopConfig.holdout_base_seed == 0``) or when the loop ran
        zero iterations.
        """
        return self._run_internal(verbose=verbose)

    def _run_internal(
        self, verbose: bool = False
    ) -> Tuple[List[LoopIterationRecord], List[LoopGuardRecord], List["LoopHoldoutRecord"]]:
        current = self._load_seed_strategies()
        rng = np.random.default_rng(self.config.mutation_seed)
        records: List[LoopIterationRecord] = []
        guard_records: List[LoopGuardRecord] = []
        holdout_records: List["LoopHoldoutRecord"] = []
        ladder: List[LadderEntry] = [
            LadderEntry(iteration=-1, specs=list(current), last_validated_score=float("nan"), proposal=None)
        ]
        ledger = _LedgerWriter(self.config.ledger_path)
        # Inactivity-relax bookkeeping: count consecutive non-accept
        # iterations (both skip and reject contribute) so the effective
        # eps_accept can decay during droughts.  Reset to 0 on every
        # accept — the planning doc requires the loop to re-tighten the
        # threshold as soon as a real improvement lands so the relaxation
        # is genuinely temporary.
        iters_since_accept = 0

        for iteration in range(self.config.iterations):
            if self._stop_requested():
                if verbose:
                    print(
                        f"[self_improve] STOP sentinel {self.config.stop_sentinel_path!r}"
                        f" present — halting at iter {iteration}"
                    )
                break

            start = time.time()
            # Compute the eps_accept this iteration will see *now*, before
            # any side-effect (skip-record, statistical_accept call).  The
            # counter snapshot is what we persist alongside the record so
            # an auditor can replay the relax rule deterministically.
            eps_for_iter = self.config.effective_eps_accept(iters_since_accept)
            streak_for_iter = iters_since_accept

            proposal = self._sample_proposal(rng, current)
            if proposal is None:
                rec = self._skip_record(
                    iteration,
                    start,
                    "no applicable mutations for current specs",
                    effective_eps_accept=eps_for_iter,
                    iters_since_accept=streak_for_iter,
                )
                records.append(rec)
                ledger.write(rec)
                if verbose:
                    self._print_iteration(rec)
                # Skip-iterations count toward the inactivity streak —
                # they're observationally indistinguishable from "no
                # candidate worth proposing", which is exactly the
                # signal the relax rule exists to break out of.
                iters_since_accept += 1
                # Guard still runs on skip iterations — it validates the
                # ladder, which is independent of whether this iteration
                # produced a proposal.
                if self._guard_due(iteration):
                    guard_record = self._run_guard(ladder, iteration, verbose)
                    guard_records.append(guard_record)
                    ledger.write(guard_record)
                    if guard_record.rolled_back:
                        current = list(ladder[-1].specs)
                continue

            candidate = apply_mutation(current, proposal)

            baseline_result = self._measure(current, iteration, "baseline", verbose)
            candidate_result = self._measure(candidate, iteration, "candidate", verbose)

            # §12.4 no-op detection: if the candidate's per-(problem,
            # strategy) scores are bit-identical to baseline the proposal
            # produced zero measurable difference at this budget — pulling
            # the bandit arm on the outcome would mis-train it because the
            # iteration carries no information about whether the rule
            # helps or hurts.  Skip the bandit pull and tag the record so
            # the summary view and codify-scan can filter these out.
            no_op = _is_no_op(baseline_result, candidate_result)

            decision = statistical_accept(
                baseline_result,
                candidate_result,
                eps_accept=eps_for_iter,
                eps_regress=self.config.eps_regress,
                n_boot=self.config.n_boot,
                confidence=self.config.confidence,
                seed=self.config.stat_seed + iteration,
                paired=self.config.paired,
                accept_stat=self.config.accept_stat,
            )

            reasons = list(decision.reasons)
            if no_op:
                reasons.append("no-op: per-pair scores bit-identical to baseline")

            screen_accept = bool(decision.accept) and not no_op

            # §6.4 same-night confirmation gate.  Runs only after a
            # screening accept (the gate cannot promote what screening
            # already rejected) and only when the loop is configured for
            # it — otherwise the V1 promote-on-screening behaviour is
            # preserved byte-for-byte.
            confirmed_flag: Optional[bool] = None
            confirm_record: Optional[LoopConfirmRecord] = None
            final_decision = decision
            if screen_accept and self.config.confirm_accepts:
                confirmation = self._run_confirmation(
                    iteration=iteration,
                    current=current,
                    candidate=candidate,
                    proposal=proposal,
                    screen_baseline=baseline_result,
                    screen_candidate=candidate_result,
                    eps_for_iter=eps_for_iter,
                    streak_for_iter=streak_for_iter,
                    verbose=verbose,
                )
                final_decision = confirmation.pooled_decision
                if confirmation.confirmed:
                    confirmed_flag = True
                    reasons.append(
                        f"confirmed: pooled CI cleared eps_accept on fresh"
                        f" randomize_iteration={confirmation.confirm_iteration_id}"
                        + (
                            f" + holdout_base_seed={confirmation.confirm_holdout_seed}"
                            if confirmation.confirm_holdout_seed is not None
                            else ""
                        )
                    )
                else:
                    confirmed_flag = False
                    reasons.append(
                        "confirm_reject: pooled CI did not clear eps_accept on"
                        " same-night re-measurement; screening was a noise spike"
                    )
                    confirm_record = confirmation.record

            accepted_flag = screen_accept and (confirmed_flag is not False)
            # Compute the graded bandit reward up front when the loop is
            # configured for §7.4 reward shaping so the persisted record
            # carries the exact value the bandit consumed.  Skip /
            # no-op iterations leave ``bandit_reward = None`` because the
            # bandit's posterior is not pulled on them (the upstream
            # ``discard_outcome`` path).  When the confirmation gate ran,
            # the reward is computed from the *post-confirmation* pooled
            # decision so an arm that produced a screening noise-spike no
            # longer collects a full-accept reward for what the gate
            # subsequently demoted to a reject.
            bandit_reward: Optional[float] = None
            if not no_op and self.config.bandit_reward_shaping == "graded":
                bandit_reward = _compute_graded_reward(
                    accepted=accepted_flag,
                    delta=float(final_decision.delta),
                    ci_low=float(final_decision.ci_low),
                    eps_accept=eps_for_iter,
                )

            rec = LoopIterationRecord(
                iteration=iteration,
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                duration_seconds=time.time() - start,
                proposal=proposal.to_dict(),
                accepted=accepted_flag,
                baseline_score=float(baseline_result.composite_score),
                candidate_score=float(candidate_result.composite_score),
                delta=float(decision.delta),
                ci_low=float(decision.ci_low),
                ci_high=float(decision.ci_high),
                worst_pair_regression=float(decision.worst_pair_regression),
                worst_pair=(
                    (str(decision.worst_pair[0]), str(decision.worst_pair[1]))
                    if decision.worst_pair is not None
                    else None
                ),
                reasons=reasons,
                base_seed=self.config.base_seed,
                randomize_iteration=iteration,
                mode=self.config.mode,
                reason_skipped="no_op" if no_op else None,
                effective_eps_accept=eps_for_iter,
                iters_since_accept=streak_for_iter,
                no_op=no_op,
                bandit_reward=bandit_reward,
                confirmed=confirmed_flag,
                sync_eval=bool(self.config.sync_eval),
                aocc_extra_dims=tuple(self.config.aocc_extra_dims),
                accept_stat=self.config.accept_stat,
                rank_p=decision.rank_p,
                rank_delta=decision.rank_delta,
            )
            records.append(rec)
            ledger.write(rec)
            if verbose:
                self._print_iteration(rec)
            if confirm_record is not None:
                ledger.write(confirm_record)

            # Refresh the seed entry's validated score the first time we
            # measure with it so the guard has a baseline to compare
            # against (the seed itself never gets accepted, but it can
            # still be the rollback target).
            if np.isnan(ladder[0].last_validated_score):
                ladder[0].last_validated_score = float(baseline_result.composite_score)

            # Update the adaptive bandit *before* swapping the ladder so
            # the rule key recorded by `_sample_proposal` still matches
            # this iteration's outcome.  Uniform-sampler runs do nothing.
            # No-op iterations carry zero information about the rule's
            # value, so we deliberately do not pull the arm — the
            # sampler's :attr:`last_rule_key` is cleared without an
            # update so the next iteration starts from a clean slate.
            if self.sampler is not None:
                if no_op:
                    self.sampler.discard_outcome()
                else:
                    self.sampler.record_outcome(accepted_flag, reward=bandit_reward)

            if rec.accepted:
                current = candidate
                ladder.append(
                    LadderEntry(
                        iteration=iteration,
                        specs=list(candidate),
                        last_validated_score=float(candidate_result.composite_score),
                        proposal=proposal.to_dict(),
                    )
                )
                # Real accept ends the drought; re-tighten the threshold
                # on the next iteration.  Done after the ladder append so
                # the snapshot recorded above still reflects the relax
                # rule that produced this accept.
                iters_since_accept = 0
            else:
                # No-op iterations still count toward the inactivity
                # streak: from the relax rule's perspective they are
                # observationally a non-accept, and the streak exists to
                # break out of long droughts regardless of why each
                # iteration failed to accept.
                iters_since_accept += 1

            # Anti-cherry-pick guard (§6.3 of the plan).  Run after the
            # iteration so a freshly accepted entry can be challenged.
            if self._guard_due(iteration):
                guard_record = self._run_guard(ladder, iteration, verbose)
                guard_records.append(guard_record)
                ledger.write(guard_record)
                if guard_record.rolled_back:
                    current = list(ladder[-1].specs)

        # End-of-loop hold-out validation.  Skipped when disabled, when
        # the loop never produced a baseline measurement, or when the
        # battery is non-randomized (in which case a different base_seed
        # would not change the instances and the check is meaningless).
        # When multiple hold-out seeds are configured we write one
        # :class:`LoopHoldoutRecord` per seed so an auditor can inspect
        # per-seed generalisation; aggregation across seeds is left to
        # the CLI summary path.
        if self._holdout_enabled() and len(records) > 0:
            for ho_seed in self.config.resolved_holdout_seeds():
                holdout_record = self._run_holdout(ladder, ho_seed, verbose)
                if holdout_record is not None:
                    holdout_records.append(holdout_record)
                    ledger.write(holdout_record)
                    if verbose:
                        self._print_holdout(holdout_record)

        return records, guard_records, holdout_records

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_proposal(
        self,
        rng: np.random.Generator,
        specs: List[StrategySpec],
    ) -> Optional[MutationProposal]:
        """Delegate to the adaptive sampler if configured, else the catalog.

        Centralising the call site lets :meth:`_run_internal` stay
        agnostic to which sampler is in use, and ensures the bandit's
        ``last_rule_key`` is set the same way an explicit caller would.
        """
        if self.sampler is not None:
            return self.sampler.sample(rng, specs)
        return self.catalog.sample(rng, specs)

    def _load_seed_strategies(self) -> List[StrategySpec]:
        if self._seed_strategies is not None:
            return list(self._seed_strategies)
        if self.config.metric == "aocc":
            from panobbgo.harness_ioh import make_ioh_strategies

            strats = make_ioh_strategies()
        else:
            cfg = HarnessConfig(
                mode=self.config.mode,
                strategies=self.config.strategy_names,
                registry=self.config.registry,
            )
            strats = self._harness_factory(cfg).get_strategies()
        if self.config.strategy_names:
            wanted = set(self.config.strategy_names)
            strats = [s for s in strats if s.name in wanted]
        return strats

    def _measure(
        self,
        specs: List[StrategySpec],
        iteration: int,
        label: str,
        verbose: bool,
    ) -> HarnessResult:
        if self.config.metric == "aocc":
            return self._measure_aocc(specs, iteration, label, verbose)
        hc = self.config.harness_config(specs, iteration)
        if verbose:
            print(f"[self_improve] iter={iteration} measuring {label}")
        return self._harness_factory(hc).run(verbose=False)

    def _measure_aocc(
        self,
        specs: List[StrategySpec],
        iteration: int,
        label: str,
        verbose: bool,
        base_seed_override: Optional[int] = None,
    ) -> HarnessResult:
        """AOCC-metric variant of :meth:`_measure`.

        Runs the IOH harness on a mode-mapped battery and adapts the
        result so the rest of the loop (statistical_accept, ledger
        writer, guard, hold-out) sees a :class:`HarnessResult` whose
        ``composite_score`` is mean AOCC and whose per-pair ``score``
        values are per-instance AOCC.  The bootstrap CI on the
        composite delta then operates directly on AOCC values.

        ``base_seed_override`` swaps the instance stream wholesale, the
        AOCC counterpart of :meth:`LoopConfig.holdout_harness_config`'s
        ``seed`` swap.  :meth:`_measure_holdout` passes the hold-out
        base seed through it so a hold-out measurement under
        ``metric="aocc"`` stays on the AOCC scale instead of silently
        falling back to ``composite_score`` (fixed 2026-08-11 — see the
        dated log entry; the mismatch is what produced the phantom
        "0.33 training vs 0.04 hold-out" generalization gap).
        """
        from panobbgo.harness_ioh import (
            aocc_to_harness_result,
            make_full_battery,
            make_quick_battery,
            make_standard_battery,
            run_ioh_harness,
            with_extra_dims,
        )

        battery_factories = {
            "quick": make_quick_battery,
            "standard": make_standard_battery,
            "full": make_full_battery,
        }
        battery = battery_factories[self.config.mode]()
        if self.config.aocc_extra_dims:
            # Widen *every* measurement identically — screening,
            # confirm, guard and hold-out all route through here, so
            # the loop can never compare a 2-D baseline against a
            # (2, 5)-D candidate.
            battery = with_extra_dims(battery, self.config.aocc_extra_dims)
        if verbose:
            print(f"[self_improve] iter={iteration} measuring {label} (AOCC, battery={battery.name})")
        # Mix the iteration into the base seed so each iteration draws
        # fresh-but-reproducible instance RNG seeds, matching the
        # randomized composite-score path.
        root_seed = self.config.base_seed if base_seed_override is None else int(base_seed_override)
        base_seed = root_seed + iteration if self.config.randomize else root_seed
        ioh_result = run_ioh_harness(
            specs,
            battery,
            base_seed=base_seed,
            progress=False,
            sync_eval=bool(self.config.sync_eval),
        )
        return aocc_to_harness_result(ioh_result, mode=self.config.mode, base_seed=root_seed)

    def _skip_record(
        self,
        iteration: int,
        start: float,
        reason: str,
        effective_eps_accept: Optional[float] = None,
        iters_since_accept: Optional[int] = None,
    ) -> LoopIterationRecord:
        return LoopIterationRecord(
            iteration=iteration,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            duration_seconds=time.time() - start,
            proposal=None,
            accepted=False,
            baseline_score=0.0,
            candidate_score=0.0,
            delta=0.0,
            ci_low=0.0,
            ci_high=0.0,
            worst_pair_regression=0.0,
            worst_pair=None,
            reasons=[reason],
            base_seed=self.config.base_seed,
            randomize_iteration=iteration,
            mode=self.config.mode,
            reason_skipped=reason,
            effective_eps_accept=effective_eps_accept,
            iters_since_accept=iters_since_accept,
            sync_eval=bool(self.config.sync_eval),
            aocc_extra_dims=tuple(self.config.aocc_extra_dims),
            accept_stat=self.config.accept_stat,
        )

    def _stop_requested(self) -> bool:
        if not self.config.stop_sentinel_path:
            return False
        return pathlib.Path(self.config.stop_sentinel_path).exists()

    def _guard_due(self, iteration: int) -> bool:
        """Return True if the guard should run after this iteration."""
        if self.config.guard_interval <= 0:
            return False
        # Run on every multiple of guard_interval (1-indexed): after
        # iteration 0 if interval == 1, after iteration K-1 if K > 1, etc.
        return ((iteration + 1) % self.config.guard_interval) == 0

    def _guard_iteration_id(self, iteration: int) -> int:
        """Translate a regular iteration index into the guard's seed.

        We add a large offset rather than reuse the regular iteration
        stream so the guard's instances are independent — this prevents
        a mutation from accidentally tuning itself to the seeds the
        guard would reuse.
        """
        return int(iteration) + int(self.config.guard_iteration_offset)

    def _run_guard(
        self,
        ladder: List["LadderEntry"],
        iteration: int,
        verbose: bool,
    ) -> "LoopGuardRecord":
        """Re-measure the top of the ladder on a fresh seed; roll back if drifted.

        Implements §6.3 of ``planning/SELF_IMPROVEMENT_LOOP.md``.  The
        method is deliberately simple: it walks down the ladder one
        entry at a time, re-measuring each on the same fresh
        ``randomize_iteration`` seed, and stops at the first entry whose
        score is within :attr:`LoopConfig.guard_eps_ladder` of its
        stored ``last_validated_score``.  If the seed entry is reached,
        no further pops happen — the seed strategies are by definition
        the safe fallback.
        """
        start = time.time()
        guard_iter_id = self._guard_iteration_id(iteration)
        size_before = len(ladder)
        reasons: List[str] = []

        # Re-measure the current top.
        top = ladder[-1]
        top_result = self._measure(top.specs, guard_iter_id, "guard", verbose)
        guard_score = float(top_result.composite_score)
        pre_guard_top_score = float(top.last_validated_score)
        pre_guard_top_iteration = int(top.iteration)

        # Within tolerance?  Refresh the validated score and return.
        if not self._guard_drifted(guard_score, pre_guard_top_score):
            top.last_validated_score = guard_score
            reasons.append(
                f"guard re-measure score {guard_score:.4f} within tolerance of "
                f"{pre_guard_top_score:.4f} (eps_ladder={self.config.guard_eps_ladder:.4f})"
            )
            return LoopGuardRecord(
                iteration=iteration,
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                duration_seconds=time.time() - start,
                guard_score=guard_score,
                pre_guard_top_score=pre_guard_top_score,
                pre_guard_top_iteration=pre_guard_top_iteration,
                rolled_back=False,
                rolled_back_to_iteration=None,
                pops=0,
                ladder_size_before=size_before,
                ladder_size_after=len(ladder),
                guard_iteration_id=guard_iter_id,
                reasons=reasons,
                base_seed=self.config.base_seed,
                mode=self.config.mode,
            )

        # Drift detected — pop and walk down the ladder.
        reasons.append(
            f"guard re-measure score {guard_score:.4f} dropped > eps_ladder "
            f"({self.config.guard_eps_ladder:.4f}) below stored {pre_guard_top_score:.4f}"
            f" — rolling back from iter {pre_guard_top_iteration}"
        )
        pops = 0
        # Always keep the seed entry (index 0).  Pop while drift persists.
        while len(ladder) > 1:
            ladder.pop()
            pops += 1
            new_top = ladder[-1]
            if new_top.iteration < 0 or np.isnan(new_top.last_validated_score):
                # Reached the seed (or an entry without a stored score)
                # — accept it without re-measurement; it is by definition
                # the trusted fallback.
                reasons.append(f"reached seed/anchor entry (iter={new_top.iteration}); stopping rollback")
                break
            new_result = self._measure(new_top.specs, guard_iter_id, "guard-rollback", verbose)
            new_score = float(new_result.composite_score)
            if not self._guard_drifted(new_score, float(new_top.last_validated_score)):
                new_top.last_validated_score = new_score
                reasons.append(
                    f"rollback target iter={new_top.iteration} stable: "
                    f"score {new_score:.4f} vs stored {new_top.last_validated_score:.4f}"
                )
                break
            reasons.append(
                f"rollback candidate iter={new_top.iteration} also drifted "
                f"({new_score:.4f} vs {new_top.last_validated_score:.4f}); continuing"
            )

        return LoopGuardRecord(
            iteration=iteration,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            duration_seconds=time.time() - start,
            guard_score=guard_score,
            pre_guard_top_score=pre_guard_top_score,
            pre_guard_top_iteration=pre_guard_top_iteration,
            rolled_back=True,
            rolled_back_to_iteration=int(ladder[-1].iteration),
            pops=pops,
            ladder_size_before=size_before,
            ladder_size_after=len(ladder),
            guard_iteration_id=guard_iter_id,
            reasons=reasons,
            base_seed=self.config.base_seed,
            mode=self.config.mode,
        )

    def _guard_drifted(self, guard_score: float, stored_score: float) -> bool:
        """Return True if ``guard_score`` is more than ``eps_ladder`` below stored."""
        if np.isnan(stored_score):
            # Nothing to compare against — treat as not drifted.
            return False
        return guard_score < stored_score - self.config.guard_eps_ladder

    # ------------------------------------------------------------------
    # Hold-out validation
    # ------------------------------------------------------------------

    def _holdout_enabled(self) -> bool:
        """Return True iff the hold-out validation is configured to run.

        Three knobs gate this: at least one independent ``base_seed``
        (via the scalar :attr:`LoopConfig.holdout_base_seed` or the
        list-typed :attr:`LoopConfig.holdout_base_seeds`), a positive
        iteration count, and the randomized battery itself — without
        randomization a different ``base_seed`` does not produce
        different instances and the check would be vacuous.
        """
        return (
            len(self.config.resolved_holdout_seeds()) > 0
            and int(self.config.holdout_iterations) > 0
            and bool(self.config.randomize)
        )

    def _measure_holdout(
        self,
        specs: List[StrategySpec],
        iteration_id: int,
        base_seed: int,
        label: str,
        verbose: bool,
    ) -> HarnessResult:
        """Single hold-out measurement on an independent ``base_seed``.

        Routes through the same metric as :meth:`_measure`.  Under
        ``metric="aocc"`` this used to fall through to the composite
        harness unconditionally, so an AOCC run's hold-out records
        carried ``composite_score`` values while its training records
        carried mean AOCC — two different scales compared as if they
        were one, which is where the phantom "0.33 training vs 0.04
        hold-out" gap came from (fixed 2026-08-11).
        """
        if verbose:
            print(f"[self_improve] hold-out measuring {label} at iter_id={iteration_id} base_seed={base_seed}")
        if self.config.metric == "aocc":
            return self._measure_aocc(
                specs,
                iteration_id,
                label,
                verbose=False,
                base_seed_override=int(base_seed),
            )
        hc = self.config.holdout_harness_config(specs, iteration_id, base_seed=base_seed)
        return self._harness_factory(hc).run(verbose=False)

    def _run_holdout(
        self,
        ladder: List[LadderEntry],
        base_seed: int,
        verbose: bool,
    ) -> Optional["LoopHoldoutRecord"]:
        """Re-measure seed and top of the ladder on an independent base_seed.

        Returns ``None`` only if the ladder is empty (defensive — by the
        time this is called the ladder always has at least the seed
        entry).  Otherwise produces a :class:`LoopHoldoutRecord` whose
        ``overfit`` flag is set when the on-hold-out improvement gap
        falls more than :attr:`LoopConfig.holdout_eps_overfit` short of
        the on-training gap.

        ``base_seed`` is the per-call hold-out seed — when only the
        scalar :attr:`LoopConfig.holdout_base_seed` is set this is that
        value, otherwise it cycles through
        :attr:`LoopConfig.holdout_base_seeds`.
        """
        if not ladder:
            return None

        start = time.time()
        seed_entry = ladder[0]
        top_entry = ladder[-1]
        seed_only = top_entry is seed_entry  # ladder never accepted anything

        n_iters = int(self.config.holdout_iterations)
        offset = int(self.config.holdout_iteration_offset)
        seed_scores: List[float] = []
        top_scores: List[float] = []

        for k in range(n_iters):
            iter_id = offset + k
            seed_result = self._measure_holdout(seed_entry.specs, iter_id, base_seed, "seed", verbose)
            seed_scores.append(float(seed_result.composite_score))
            if seed_only:
                top_scores.append(seed_scores[-1])
            else:
                top_result = self._measure_holdout(top_entry.specs, iter_id, base_seed, "top", verbose)
                top_scores.append(float(top_result.composite_score))

        seed_holdout = float(np.mean(seed_scores)) if seed_scores else 0.0
        top_holdout = float(np.mean(top_scores)) if top_scores else seed_holdout

        # ``last_validated_score`` is the most recent training-time
        # measurement.  When the seed never recorded a baseline (e.g. a
        # zero-iteration loop, which we filter out before calling this),
        # NaN is mapped to 0.0 so the delta stays well-defined.
        seed_training = float(seed_entry.last_validated_score) if not np.isnan(seed_entry.last_validated_score) else 0.0
        top_training = (
            float(top_entry.last_validated_score) if not np.isnan(top_entry.last_validated_score) else seed_training
        )

        holdout_delta = top_holdout - seed_holdout
        training_delta = top_training - seed_training
        drift = holdout_delta - training_delta
        # ``seed_only`` records are vacuous by construction: ``holdout_delta``,
        # ``training_delta`` and ``drift`` are forced to ``0.0`` because the
        # "top" we are validating *is* the seed.  Reporting ``overfit=False``
        # alongside ``drift=+0.0000`` historically masqueraded as an
        # honest "OK" verdict in the loop output even though no
        # generalisation signal exists — see V2 §6.4 / §12.4 of
        # `planning/SELF_IMPROVEMENT_LOOP.md`.  Setting
        # ``status="vacuous"`` here keeps ``overfit=False`` (vacuous is
        # not overfit) while letting downstream consumers (printer,
        # aggregator, summary) distinguish "ladder produced no
        # mutations to validate" from "ladder generalised cleanly".
        overfit = (not seed_only) and drift < -float(self.config.holdout_eps_overfit)
        if seed_only:
            status = "vacuous"
        elif overfit:
            status = "overfit"
        else:
            status = "ok"

        reasons: List[str] = []
        if seed_only:
            reasons.append(
                "ladder has only the seed entry — no accepted mutations to validate; "
                "hold-out is VACUOUS: scores recorded for reference but drift is 0 by construction"
            )
        elif overfit:
            reasons.append(
                f"hold-out drift {drift:+.4f} below -eps_overfit "
                f"({-float(self.config.holdout_eps_overfit):.4f}); "
                f"on-hold-out gap {holdout_delta:+.4f} vs on-training {training_delta:+.4f} "
                "— the ladder appears to overfit the training base_seed family"
            )
        else:
            reasons.append(
                f"hold-out drift {drift:+.4f} within tolerance "
                f"(>= {-float(self.config.holdout_eps_overfit):.4f}); "
                f"on-hold-out gap {holdout_delta:+.4f} vs on-training {training_delta:+.4f} "
                "— improvement appears to generalise"
            )

        return LoopHoldoutRecord(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            duration_seconds=time.time() - start,
            holdout_base_seed=int(base_seed),
            holdout_iterations=n_iters,
            holdout_iteration_offset=offset,
            seed_holdout_score=seed_holdout,
            top_holdout_score=top_holdout,
            seed_training_score=seed_training,
            top_training_score=top_training,
            holdout_delta=float(holdout_delta),
            training_delta=float(training_delta),
            drift=float(drift),
            overfit=bool(overfit),
            eps_overfit=float(self.config.holdout_eps_overfit),
            top_iteration=int(top_entry.iteration),
            ladder_size=len(ladder),
            base_seed=int(self.config.base_seed),
            mode=self.config.mode,
            reasons=reasons,
            # Per-iteration paired scores enable bootstrap-CI aggregation
            # via :func:`aggregate_holdout_drift`.  Stored alongside the
            # aggregate scores so consumers can either use the cached
            # means or re-derive a CI on the pooled drift sample.
            seed_iteration_scores=list(seed_scores),
            top_iteration_scores=list(top_scores),
            status=status,
        )

    def _confirm_iteration_id(self, iteration: int) -> int:
        """Translate a regular iteration index into the confirmation seed.

        §6.4 of ``planning/SELF_IMPROVEMENT_LOOP.md``: the same-night
        confirmation gate re-measures a screening-accepted candidate on
        a fresh ``randomize_iteration`` to break the noise-spike
        correlation that V1 promote-on-screening was vulnerable to.
        The offset is large and distinct from the guard's offset so the
        three iteration streams (regular / confirm / guard) never
        collide at realistic iteration counts.
        """
        return int(iteration) + int(self.config.confirm_iteration_offset)

    def _run_confirmation(
        self,
        *,
        iteration: int,
        current: List[StrategySpec],
        candidate: List[StrategySpec],
        proposal: MutationProposal,
        screen_baseline: HarnessResult,
        screen_candidate: HarnessResult,
        eps_for_iter: float,
        streak_for_iter: int,
        verbose: bool,
    ) -> "_ConfirmationOutcome":
        """Re-measure a screening accept on independent instances; gate promotion.

        Implements §6.4 of ``planning/SELF_IMPROVEMENT_LOOP.md``.  The
        screening measurement (``screen_baseline``, ``screen_candidate``)
        is paired with one — or two, when a hold-out seed is configured
        — additional measurements drawn from independent SHA-256
        streams, then :func:`~panobbgo.harness.statistical_accept` is
        re-run on the pooled sample.  Promotion happens only when the
        pooled CI still clears ``eps_accept``.

        The hold-out re-measurement uses the *first* configured
        hold-out base_seed to keep per-iteration compute bounded at
        ``≤ 3×`` the screening cost regardless of how many hold-out
        seeds are configured for the end-of-loop drift check.  The
        end-of-loop hold-out continues to walk every configured seed.

        Args:
            iteration: Regular iteration index — used to derive the
                fresh ``randomize_iteration`` for the confirmation step.
            current: Pre-mutation spec list (paired baseline measurement).
            candidate: Post-mutation spec list (paired candidate
                measurement).
            proposal: The :class:`MutationProposal` whose screening
                accept the gate is about to confirm or reject.
            screen_baseline: Screening baseline measurement (re-used
                without re-measuring).
            screen_candidate: Screening candidate measurement (re-used
                without re-measuring).
            eps_for_iter: Effective ``eps_accept`` for the iteration —
                consults the inactivity-relax rule so a relaxed
                screening threshold gets a relaxed confirmation
                threshold too.
            streak_for_iter: Inactivity streak snapshot (forwarded into
                the confirm record so an auditor can replay the relax
                rule that produced the screening accept).
            verbose: If ``True``, print one ``[self_improve]`` line per
                confirmation measurement.

        Returns:
            A populated :class:`_ConfirmationOutcome` carrying the
            pooled :class:`~panobbgo.harness.StatisticalDecision`, the
            verdict, the fresh-iteration id, and (when ``confirmed`` is
            ``False``) the :class:`LoopConfirmRecord` to append to the
            ledger.
        """
        start = time.time()
        confirm_iter_id = self._confirm_iteration_id(iteration)
        confirm_baseline = self._measure(current, confirm_iter_id, "confirm-baseline", verbose)
        confirm_candidate = self._measure(candidate, confirm_iter_id, "confirm-candidate", verbose)

        pooled_baseline = _pool_harness_results(screen_baseline, confirm_baseline)
        pooled_candidate = _pool_harness_results(screen_candidate, confirm_candidate)

        # Optional hold-out leg: when the loop is also configured with
        # a hold-out base_seed (single- or multi-seed), confirm on the
        # *first* hold-out seed too.  The planning doc's "fresh
        # randomize_iteration *and* hold-out base_seed" wording.  Only
        # the first seed is used so the per-iteration confirmation cost
        # stays bounded regardless of how many hold-out seeds the
        # end-of-loop drift check walks.
        # The ``metric != "aocc"`` exclusion that used to sit on this
        # branch existed only because :meth:`_measure_holdout` could not
        # produce AOCC-scale results; with that fixed the AOCC nightly
        # gets the cross-base-seed leg too.  This is the only place in
        # the accept path that crosses an instance-family boundary, so
        # without it every accept in an AOCC run was decided on a single
        # base seed (all 952 records of the 2026-07..08 ledger were
        # base_seed=42) and "k>=2 distinct nights" of codify evidence
        # meant one instance draw re-measured k times.
        ho_seed: Optional[int] = None
        ho_baseline_score: Optional[float] = None
        ho_candidate_score: Optional[float] = None
        resolved_holdout_seeds = self.config.resolved_holdout_seeds()
        if resolved_holdout_seeds and bool(self.config.randomize):
            ho_seed = int(resolved_holdout_seeds[0])
            ho_baseline = self._measure_holdout(current, confirm_iter_id, ho_seed, "confirm-ho-baseline", verbose)
            ho_candidate = self._measure_holdout(candidate, confirm_iter_id, ho_seed, "confirm-ho-candidate", verbose)
            pooled_baseline = _pool_harness_results(pooled_baseline, ho_baseline)
            pooled_candidate = _pool_harness_results(pooled_candidate, ho_candidate)
            ho_baseline_score = float(ho_baseline.composite_score)
            ho_candidate_score = float(ho_candidate.composite_score)

        pooled_decision = statistical_accept(
            pooled_baseline,
            pooled_candidate,
            eps_accept=eps_for_iter,
            eps_regress=self.config.eps_regress,
            n_boot=self.config.n_boot,
            confidence=self.config.confidence,
            # Distinct seed offset so the pooled bootstrap draws are not
            # bit-identical to the screening bootstrap (paranoia — the
            # rep arrays are larger so the resample index space is
            # different anyway, but we want the two CIs to be
            # statistically independent at every layer).
            seed=self.config.stat_seed + iteration + int(self.config.confirm_iteration_offset),
            paired=self.config.paired,
            accept_stat=self.config.accept_stat,
        )
        confirmed = bool(pooled_decision.accept)

        confirm_record: Optional[LoopConfirmRecord] = None
        if not confirmed:
            confirm_record = LoopConfirmRecord(
                iteration=iteration,
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                duration_seconds=time.time() - start,
                proposal=proposal.to_dict(),
                screen_baseline_score=float(screen_baseline.composite_score),
                screen_candidate_score=float(screen_candidate.composite_score),
                screen_delta=float(screen_candidate.composite_score - screen_baseline.composite_score),
                confirm_baseline_score=float(confirm_baseline.composite_score),
                confirm_candidate_score=float(confirm_candidate.composite_score),
                confirm_delta=float(confirm_candidate.composite_score - confirm_baseline.composite_score),
                pooled_delta=float(pooled_decision.delta),
                pooled_ci_low=float(pooled_decision.ci_low),
                pooled_ci_high=float(pooled_decision.ci_high),
                pooled_worst_pair_regression=float(pooled_decision.worst_pair_regression),
                pooled_worst_pair=(
                    (str(pooled_decision.worst_pair[0]), str(pooled_decision.worst_pair[1]))
                    if pooled_decision.worst_pair is not None
                    else None
                ),
                confirm_iteration_id=confirm_iter_id,
                confirm_holdout_seed=ho_seed,
                confirm_holdout_baseline_score=ho_baseline_score,
                confirm_holdout_candidate_score=ho_candidate_score,
                reasons=list(pooled_decision.reasons),
                base_seed=int(self.config.base_seed),
                mode=self.config.mode,
            )

        return _ConfirmationOutcome(
            confirmed=confirmed,
            pooled_decision=pooled_decision,
            confirm_iteration_id=confirm_iter_id,
            confirm_holdout_seed=ho_seed,
            record=confirm_record,
        )

    @staticmethod
    def _print_holdout(rec: "LoopHoldoutRecord") -> None:
        # Use ``effective_status`` so legacy ledger lines (no status
        # field, all defaulted to ``"ok"``) still surface the vacuous
        # verdict — V2 §6.4 / §12.4 of
        # `planning/SELF_IMPROVEMENT_LOOP.md`.
        verdict = rec.effective_status().upper()
        print(
            f"[hold-out] {verdict}  drift={rec.drift:+.4f}  "
            f"holdout_gap={rec.holdout_delta:+.4f}  training_gap={rec.training_delta:+.4f}  "
            f"top_iter={rec.top_iteration}"
        )

    @staticmethod
    def _print_iteration(rec: LoopIterationRecord) -> None:
        if rec.proposal is None:
            print(f"[iter {rec.iteration}] SKIP: {rec.reason_skipped}")
            return
        verdict = "ACCEPT" if rec.accepted else "REJECT"
        p = rec.proposal
        print(
            f"[iter {rec.iteration}] {verdict}  Δ={rec.delta:+.4f}  "
            f"CI=[{rec.ci_low:+.4f},{rec.ci_high:+.4f}]  "
            f"{p.get('strategy_name')}/{p.get('class_name')}.{p.get('param_name')}: "
            f"{p.get('old_value')!r} -> {p.get('new_value')!r}"
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class _LedgerWriter:
    """Append-only JSONL ledger.  Creates parent directories on demand.

    Writes :class:`LoopIterationRecord`, :class:`LoopGuardRecord`,
    :class:`LoopConfirmRecord`, and :class:`LoopHoldoutRecord`
    instances; the ``record_type`` field distinguishes them on read.
    """

    def __init__(self, path: str) -> None:
        self.path = pathlib.Path(path)
        parent = self.path.parent
        if str(parent) and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Any) -> None:
        line = json.dumps(record.to_dict(), default=_json_default)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)!r} not JSON-serialisable")


def load_ledger(path: str) -> List[Dict[str, Any]]:
    """Parse a JSONL ledger file back into a list of record dicts.

    Useful for analysis / dashboards; the result preserves the order of
    iterations as written.
    """
    p = pathlib.Path(path)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Per-metric ledger routing (V2 §12.1 — after the 2026-07-09 AOCC flip)
# ---------------------------------------------------------------------------

# Canonical active-ledger basename stems keyed by accept/reject metric.  The
# nightly cron writes a *metric-specific* ledger (§12.1) because composite
# deltas (~1e-3 scale) and aocc deltas (~0.3 scale) live on ~100×-different
# scales and must never pool in codify-scan's bootstrap CI or the graded
# bandit reward.  Note the composite stem is a strict *prefix* of the aocc
# stem, so :func:`metric_for_ledger_path` matches the longest stem first.
LEDGER_STEM_BY_METRIC: Dict[str, str] = {
    "composite": "self_improve_ledger",
    "aocc": "self_improve_ledger_aocc",
}

# Directory the canonical active ledgers live in (§12.1).
DEFAULT_LEDGER_DIR = "planning"


def ledger_path_for_metric(metric: str, ledger_dir: str = DEFAULT_LEDGER_DIR) -> str:
    """Return the canonical active-ledger path for ``metric``.

    ``composite`` → ``<ledger_dir>/self_improve_ledger.jsonl`` (the frozen
    historical ledger); ``aocc`` → ``<ledger_dir>/self_improve_ledger_aocc.jsonl``
    (the active regime since the 2026-07-09 flip).  Raises ``ValueError`` for
    an unknown metric so a mistyped ``--metric`` fails loudly instead of
    silently scanning the wrong ledger.
    """
    try:
        stem = LEDGER_STEM_BY_METRIC[metric]
    except KeyError:
        known = ", ".join(sorted(LEDGER_STEM_BY_METRIC))
        raise ValueError(f"unknown metric {metric!r} (known: {known})") from None
    return str(pathlib.Path(ledger_dir) / f"{stem}.jsonl")


def metric_for_ledger_path(path: Any) -> str:
    """Infer which metric a ledger / archive file belongs to from its name.

    Recognises both the live name ``<stem>.jsonl`` and the rotated-archive
    name ``<stem>_<suffix>.jsonl`` (e.g. ``self_improve_ledger_2026-05-31.jsonl``
    or ``self_improve_ledger_aocc_2026-07-10.jsonl``).  Because the composite
    stem is a prefix of the aocc stem, the *longest* matching stem wins so an
    ``aocc`` archive is never misclassified as ``composite``.  Names matching
    no known stem fall back to ``composite`` — the historical single-metric
    regime, so pre-flip archives and test fixtures classify unchanged.
    """
    name = pathlib.Path(path).name
    best_metric, best_len = "composite", -1
    for metric, stem in LEDGER_STEM_BY_METRIC.items():
        if (name == f"{stem}.jsonl" or name.startswith(f"{stem}_")) and len(stem) > best_len:
            best_metric, best_len = metric, len(stem)
    return best_metric


def iter_metric_archives(archive_dir: str, ledger_path: str) -> List[pathlib.Path]:
    """Archive ledger files under ``archive_dir`` sharing ``ledger_path``'s metric.

    Returns the rotated archives (``self_improve_ledger_*.jsonl``) whose
    inferred metric matches ``ledger_path``'s, in chronological (lexicographic
    by filename) order — the rotation convention
    ``self_improve_ledger[_aocc]_YYYY-MM-DD.jsonl`` makes a plain sort
    chronological.  A missing directory returns ``[]``.  This is what scopes
    archive priming / codify-scan to a single metric so an aocc run warms only
    from aocc archives (and vice versa) — see the AOCC-regime follow-ups in
    ``planning/SELF_IMPROVEMENT_LOG.md``.
    """
    archive_path = pathlib.Path(archive_dir)
    if not archive_path.is_dir():
        return []
    metric = metric_for_ledger_path(ledger_path)
    return [f for f in sorted(archive_path.glob("self_improve_ledger_*.jsonl")) if metric_for_ledger_path(f) == metric]


# ---------------------------------------------------------------------------
# Codify-scan (§9.3 / §9.5 step 4)
# ---------------------------------------------------------------------------


def _direction_key(proposal: Dict[str, Any]) -> Optional[str]:
    """Compute the *direction* of an accepted mutation proposal.

    The direction collapses every accepted iteration record into a stable
    bucket identifier so the scanner can group "the same change" across
    nights.  For numeric kwarg rules this is just the sign of
    ``(new - old)`` — ``"up"`` if the bandit raised the value,
    ``"down"`` if it lowered it.  For categorical rules each *chosen
    value* gets its own bucket (so ``Sobol.scramble=False`` is a distinct
    candidate from ``Sobol.scramble=True``).  For structural ops the
    operation itself is the direction (``"add_heuristic"`` /
    ``"drop_heuristic"`` / ``"add_analyzer"`` / ``"drop_analyzer"``).

    Returns ``None`` when the proposal carries no informative direction
    (delta exactly zero on a numeric rule — rare with current catalogs
    but possible on pathological no-op proposals; the post-2026-06-12
    no-op detector filters these out earlier anyway).
    """
    op = proposal.get("op")
    if op:
        return str(op)
    rule_kind = str(proposal.get("rule_kind", ""))
    new_value = proposal.get("new_value")
    if rule_kind == "categorical_choice":
        # ``repr`` so booleans and strings each get their own bucket and
        # ``True`` / ``False`` cannot collide with ``"True"`` / ``"False"``.
        return repr(new_value)
    old_value = proposal.get("old_value")
    try:
        old_f = float(old_value) if old_value is not None else None
        new_f = float(new_value) if new_value is not None else None
    except (TypeError, ValueError):
        return None
    if old_f is None or new_f is None:
        return None
    if new_f > old_f:
        return "up"
    if new_f < old_f:
        return "down"
    return None


def _date_from_timestamp(ts: Any) -> str:
    """Extract the ``YYYY-MM-DD`` date prefix from an ISO 8601 timestamp.

    The ledger writes ISO 8601 timestamps with a UTC offset
    (``"2026-06-06T06:26:46.485238+00:00"``); the first 10 characters
    are the date.  Missing / malformed timestamps collapse to the empty
    string so the caller can treat them as "unknown date".
    """
    s = str(ts) if ts is not None else ""
    return s[:10] if len(s) >= 10 else ""


def _round_outward_to_significant(value: float, direction: str, n_sig: int = 3) -> float:
    """Round ``value`` outward (away from the median) to ``n_sig`` significant digits.

    "Outward" is the linear direction of the codify signal: ``direction="up"``
    rounds toward a larger value, ``direction="down"`` rounds toward a smaller
    value.  The result satisfies the
    :func:`_candidate_already_codified` predicate on the next scan (i.e.
    ``rounded_up >= value`` for ``direction="up"`` and ``rounded_down <= value``
    for ``direction="down"``), which is the self-stability property the
    manual codify entries (PR #271's :class:`~panobbgo.heuristics.nearby.Nearby`
    ``radius`` shift, 2026-06-28; the 2026-06-26 ``Nearby.radius`` catalog
    tightening) relied on so the source edit cleanly suppresses the candidate
    on the following nightly run.

    For ``n_sig=3`` and ``direction="up"`` the value ``0.123105`` rounds to
    ``0.124`` (the value PR #271 shipped after hand-computing the median of
    the accepted ``new_value`` distribution).

    Args:
        value: The numeric value to round, typically the median of accepted
            ``new_value`` entries on a codify candidate.  ``0.0`` and
            non-finite inputs are returned unchanged.
        direction: ``"up"`` (round up / toward larger magnitude for positive
            values) or ``"down"`` (round down / toward smaller magnitude).
            For negative values the abs-rounding direction is inverted so the
            result still moves in the *linear* direction (``"up"`` on
            ``-0.5`` returns a less-negative value).
        n_sig: Number of significant digits to preserve.  Default ``3`` is
            the precision used by the manual codify entries (PR #271 shipped
            ``0.124`` after rounding ``0.123105`` to 3 sig figs).

    Returns:
        The rounded value.  Always returns a ``float`` to avoid surprise
        promotion of ``int`` inputs — callers handling integer rules
        (``"integer_add"``) should wrap with ``int(...)`` and use
        :func:`math.ceil` / :func:`math.floor` directly instead.
    """
    if not math.isfinite(value) or value == 0.0:
        return value
    sign = 1.0 if value > 0 else -1.0
    abs_value = abs(value)
    exp = math.floor(math.log10(abs_value))
    scale = 10.0 ** (exp - n_sig + 1)
    scaled = abs_value / scale
    # For negative values, swap direction so "up" still means "larger in linear sense":
    # rounding abs DOWN on a negative value moves the linear value UP toward zero.
    abs_direction = direction
    if sign < 0:
        abs_direction = "down" if direction == "up" else "up"
    if abs_direction == "up":
        rounded_scaled = math.ceil(scaled)
    elif abs_direction == "down":
        rounded_scaled = math.floor(scaled)
    else:
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    return sign * rounded_scaled * scale


def _percentile_bootstrap_ci(
    samples: Sequence[float],
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Two-sided percentile bootstrap CI on the mean.

    Returns ``(ci_low, ci_high)``.  With fewer than two samples the CI
    collapses to ``(mean, mean)`` — the bootstrap has nothing to draw
    over and the caller should treat the result as a degenerate point
    estimate.  Matches the simple non-paired bootstrap used elsewhere
    in the module for parity (see :func:`aggregate_holdout_drift`).
    """
    arr = np.asarray(list(samples), dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = arr.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(arr[idx].mean())
    half = (1.0 - confidence) / 2.0
    lo = float(np.quantile(means, half))
    hi = float(np.quantile(means, 1.0 - half))
    return (lo, hi)


@dataclass
class CodifyCandidate:
    """One directionally-consistent group of accepted mutations.

    Produced by :func:`aggregate_codify_candidates`.  Carries the raw
    per-record evidence (deltas, CIs, timestamps, old / new values) plus
    a small set of pooled statistics so the CLI report and any future
    ``--open-pr`` workflow can show "why" without re-aggregating the
    ledger.

    A "candidate" is a *suggestion* — the operator (or a future codify
    PR routine) still has to translate "rule X has fired up in direction
    Y on N distinct nights" into a concrete source edit.  For numeric
    rules the natural translation is the median of :attr:`new_values`
    (the new default to ship); for categorical rules the translation is
    the unique chosen value; for structural ops it is "add this class
    to the seed pool" or "drop this class".

    Attributes:
        class_name: Heuristic / analyzer class the proposals target
            (``"Sobol"``, ``"Restart"`` …).
        param_name: Kwarg slot the proposals perturbed.  Empty string
            for structural ops.
        rule_kind: ``"integer_add"`` / ``"float_uniform"`` /
            ``"log_uniform_perturb"`` / ``"categorical_choice"`` /
            ``"structural"`` (for ``add_/drop_`` ops).
        op: ``None`` for kwarg rules; one of ``add_heuristic`` /
            ``drop_heuristic`` / ``add_analyzer`` / ``drop_analyzer``
            for structural ops.  Used by :attr:`slot_key` so dedup
            against open PRs distinguishes structural changes from
            kwarg tunes on the same class.
        direction: Stable bucket identifier — ``"up"`` / ``"down"`` for
            numeric rules; ``repr(value)`` for categorical; the op name
            for structural.  See :func:`_direction_key`.
        n_accepts: Number of accepted iteration records contributing.
        distinct_dates: Sorted tuple of distinct ``YYYY-MM-DD`` dates
            on which an accept fired.  ``len(distinct_dates)`` is the
            k≥2 threshold the §9.3 spec gates on.
        deltas: Per-record composite deltas (the same ``delta`` field
            the iteration record carries).
        ci_lows: Per-record paired-bootstrap CI lower bounds.
        ci_highs: Per-record paired-bootstrap CI upper bounds.
        old_values: Per-record ``proposal.old_value`` (raw, JSON-typed).
        new_values: Per-record ``proposal.new_value`` (raw, JSON-typed).
        timestamps: Per-record ISO 8601 timestamps in ledger order.
        strategy_names: Per-record ``proposal.strategy_name``.  A
            candidate that fires across multiple seed strategies
            (e.g. both ``Rewarding_Diverse`` and ``Loop_Sobol``) is
            stronger evidence than one that fires on a single strategy.
        confirmed_flags: Per-record value of the
            :attr:`LoopIterationRecord.confirmed` field — ``True`` /
            ``False`` for records written after V2 §6.4 ships,
            ``None`` for legacy records.  Surfaced verbatim so the CLI
            report can flag candidates with no confirmation gate
            coverage as soft evidence.
        already_codified: ``True`` when the candidate's direction is
            already reflected in at least one seed-spec factory — i.e.
            shipping the implied source edit would be a no-op.  Set by
            :func:`annotate_codified_status` (default ``False`` so a
            candidate produced by :func:`aggregate_codify_candidates`
            without the annotation pass is treated as actionable).
        live_codified_values: Tuple of values the
            ``(class_name, param_name)`` slot currently carries in the
            scanned seed-spec factories.  Empty when no factory sets
            the kwarg explicitly (the constructor default applies).
            Populated by :func:`annotate_codified_status` alongside
            :attr:`already_codified`.
        rejected: ``True`` when the candidate's slot was rejected by a
            recorded operator decision (an A/B negative result or a
            moot verdict in the rejections file) *and* the evidence
            nights post-dating that rejection number fewer than the
            resurrection bar
            (:data:`DEFAULT_RESURRECT_MIN_FRESH_NIGHTS`) — i.e.
            re-applying the edit would re-litigate a decided question
            without materially new information.  Set by
            :func:`annotate_rejected_status` (default ``False`` so an
            un-annotated candidate is treated as actionable).
        rejected_on: ``YYYY-MM-DD`` date of the matching rejection
            record.  Populated whenever a rejection matches the slot,
            *even when* fresh post-rejection evidence keeps
            :attr:`rejected` ``False`` — the CLI uses the combination
            (``rejected=False``, ``rejected_on`` set) to tag a
            resurfaced candidate with its rejection history.
        rejection_reason: Free-text reason from the matching rejection
            record (empty when no rejection matches).
    """

    class_name: str
    param_name: str
    rule_kind: str
    op: Optional[str]
    direction: str
    n_accepts: int
    distinct_dates: Tuple[str, ...]
    deltas: Tuple[float, ...]
    ci_lows: Tuple[float, ...]
    ci_highs: Tuple[float, ...]
    old_values: Tuple[Any, ...]
    new_values: Tuple[Any, ...]
    timestamps: Tuple[str, ...]
    strategy_names: Tuple[str, ...]
    confirmed_flags: Tuple[Optional[bool], ...]
    already_codified: bool = False
    live_codified_values: Tuple[Any, ...] = ()
    #: Per-record ``proposal.structural_kwargs`` (dict or ``None``) in
    #: ledger order.  Only meaningful for structural ``add_*`` ops —
    #: the kwargs the measured arm constructed the added class with
    #: (e.g. ``{"warm_start": True}`` for the LBFGSB catalog
    #: candidate).  Kwarg / categorical candidates carry an empty
    #: tuple.  Consumed by :meth:`consensus_structural_kwargs`.
    structural_kwargs_list: Tuple[Optional[Dict[str, Any]], ...] = ()
    rejected: bool = False
    rejected_on: str = ""
    rejection_reason: str = ""

    @property
    def n_distinct_nights(self) -> int:
        return len(self.distinct_dates)

    @property
    def mean_delta(self) -> float:
        if not self.deltas:
            return 0.0
        return float(np.mean(self.deltas))

    @property
    def min_ci_low(self) -> float:
        if not self.ci_lows:
            return 0.0
        return float(min(self.ci_lows))

    @property
    def max_ci_high(self) -> float:
        if not self.ci_highs:
            return 0.0
        return float(max(self.ci_highs))

    @property
    def slot_key(self) -> Tuple[str, str, Optional[str]]:
        """Identifier for dedup against open PRs — ``(class, param, op)``.

        Used by the CLI's ``--open-pr`` follow-up (queued under V2 §9.5
        step 4) to skip slots where a codify PR already exists.  The
        direction is intentionally *not* part of the key: a single open
        PR per slot is enough (a same-slot opposite-direction signal
        would supersede the open PR, not duplicate it).
        """
        return (self.class_name, self.param_name, self.op)

    def pooled_bootstrap_ci(
        self,
        *,
        n_boot: int = 2000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> Tuple[float, float]:
        """Percentile bootstrap CI on the pooled per-record deltas.

        Each accept contributes one sample (its post-paired-bootstrap
        point delta).  This is *coarser* than re-pooling the underlying
        per-(problem, strategy) bootstrap samples — those would need
        the original :class:`HarnessResult` objects, which are not in
        the ledger — but it captures the cross-night dispersion the
        §9.3 "pooled CI > 0" rule actually wants to test.  Two or
        fewer accepts produce a degenerate point CI; the caller should
        treat the result as suggestive only.
        """
        return _percentile_bootstrap_ci(self.deltas, n_boot=n_boot, confidence=confidence, seed=seed)

    def consensus_structural_kwargs(self) -> Optional[Dict[str, Any]]:
        """Kwargs a structural ``add_*`` codify edit should construct with.

        The nightly loop measured the added class with the *catalog
        candidate's* kwargs (recorded per-accept in
        ``proposal.structural_kwargs``), so a codify edit that writes
        ``(Class, {})`` would ship something the ledger never measured
        — for ``LBFGSB`` the difference is stark: the catalog arm is
        ``{"warm_start": True}`` precisely because the cold variant was
        a measured regression (2026-07-06 negative result).

        Returns the most common kwargs dict among contributing records
        (ties broken toward the most recent), or ``None`` when no
        record carries kwargs (legacy ledgers, drop ops, kwarg rules).
        """
        counted: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}
        for idx, kw in enumerate(self.structural_kwargs_list):
            if not isinstance(kw, dict):
                continue
            key = json.dumps({k: kw[k] for k in sorted(kw)}, sort_keys=True, default=str)
            count, _last_idx, _ = counted.get(key, (0, -1, kw))
            counted[key] = (count + 1, idx, kw)
        if not counted:
            return None
        _, _, best = max(counted.values(), key=lambda t: (t[0], t[1]))
        return dict(best)

    def proposed_codify_value(self, *, n_sig: int = 3) -> Any:
        """Compute the new seed value a codify edit would apply for this candidate.

        Surfaces the value the operator (or a future
        ``codify-scan --open-pr`` driver) would shift the seed-spec
        factory to.  Centralises the rounding policy used by the manual
        codify PRs (PR #257 / #271 / 2026-06-26 catalog tightening) so
        the report and any downstream automation share one source of
        truth.

        The rule depends on the candidate's :attr:`rule_kind` and
        :attr:`direction`:

        * Structural ops (``op is not None``): returns ``None`` —
          structural codification adds / drops a class, it has no kwarg
          value.  Callers should consult :attr:`class_name` and
          :attr:`op` directly.
        * Categorical (``rule_kind == "categorical_choice"``): returns
          the chosen value.  By construction every record in the
          candidate's bucket carries the same ``new_value`` (the
          :attr:`direction` is ``repr(new_value)``), so the most recent
          ``new_value`` is the canonical choice.
        * Numeric (``rule_kind in {"integer_add", "float_uniform",
          "log_uniform_perturb"}``) with ``direction in {"up", "down"}``:
          returns the median of :attr:`new_values`, rounded *outward*
          (in :attr:`direction`) to ``n_sig`` significant digits.  For
          ``integer_add`` the median is rounded to the nearest integer
          via :func:`math.ceil` (``"up"``) / :func:`math.floor`
          (``"down"``).  The outward rounding ensures the codified
          value satisfies :func:`_candidate_already_codified` on the
          next scan (``max(live) >= median`` for ``"up"``,
          ``min(live) <= median`` for ``"down"``) — i.e. the source
          edit cleanly suppresses the candidate on the following night.
        * Numeric with ``direction not in {"up", "down"}``: returns
          ``None`` (the candidate is not directionally consistent;
          rare with current catalogs since the no-op detector filters
          zero-delta records).
        * Empty :attr:`new_values` (defensive): returns ``None``.
        * Non-numeric :attr:`new_values` for a numeric rule kind
          (defensive): returns ``None``.

        Args:
            n_sig: Number of significant digits to round to for
                float-valued numeric rules.  Default ``3`` matches the
                precision the manual codify entries (PR #271's
                ``Nearby.radius: 0.123105 -> 0.124``) used.  Ignored
                for ``integer_add`` (always integer-rounded) and for
                categorical / structural candidates.
        """
        if self.op is not None:
            return None
        if not self.new_values:
            return None
        if self.rule_kind == "categorical_choice":
            # Every record in a categorical bucket shares the same new_value
            # (the bucket key is repr(new_value)); take the most recent.
            return self.new_values[-1]
        if self.direction not in ("up", "down"):
            return None
        try:
            new_floats = [float(v) for v in self.new_values]
        except (TypeError, ValueError):
            return None
        if not new_floats:
            return None
        target = float(np.median(new_floats))
        if self.rule_kind == "integer_add":
            if self.direction == "up":
                return int(math.ceil(target))
            return int(math.floor(target))
        return _round_outward_to_significant(target, self.direction, n_sig=n_sig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "param_name": self.param_name,
            "rule_kind": self.rule_kind,
            "op": self.op,
            "direction": self.direction,
            "n_accepts": int(self.n_accepts),
            "n_distinct_nights": int(self.n_distinct_nights),
            "distinct_dates": list(self.distinct_dates),
            "deltas": [float(d) for d in self.deltas],
            "ci_lows": [float(c) for c in self.ci_lows],
            "ci_highs": [float(c) for c in self.ci_highs],
            "old_values": [_to_plain(v) for v in self.old_values],
            "new_values": [_to_plain(v) for v in self.new_values],
            "timestamps": list(self.timestamps),
            "strategy_names": list(self.strategy_names),
            "confirmed_flags": list(self.confirmed_flags),
            "mean_delta": float(self.mean_delta),
            "min_ci_low": float(self.min_ci_low),
            "max_ci_high": float(self.max_ci_high),
            "already_codified": bool(self.already_codified),
            "live_codified_values": [_to_plain(v) for v in self.live_codified_values],
            "rejected": bool(self.rejected),
            "rejected_on": self.rejected_on,
            "rejection_reason": self.rejection_reason,
            "proposed_codify_value": _to_plain(self.proposed_codify_value()),
            "structural_kwargs": self.consensus_structural_kwargs(),
        }


def aggregate_codify_candidates(
    records: Sequence[Dict[str, Any]],
    *,
    min_nights: int = 2,
    require_positive_min_ci: bool = True,
    confirmed_only: bool = False,
) -> List[CodifyCandidate]:
    """Scan ledger records for directionally-consistent accepted patterns.

    The cross-night codification stage of V2 §9.3 / §9.5 step 4: pool
    every accepted mutation iteration across the live ledger and the
    archives, group by ``(class, param, direction)`` (or ``(op, class)``
    for structural ops), and surface every group with at least
    ``min_nights`` distinct accept dates.  These are the candidates a
    daily routine (or a future ``--open-pr`` driver) can codify into
    constructor defaults.

    Args:
        records: Concatenation of every JSONL record relevant for
            scanning — typically ``load_ledger(live)`` plus
            ``load_ledger(each archive)``.  Non-iteration records and
            non-accepted iterations are silently dropped.  Records
            without a proposal (skip rows) are skipped.  No-op
            iterations (``no_op == True``) are skipped — by
            construction they contributed zero behavioural information
            even though they may be flagged as accepted in pathological
            ledgers.
        min_nights: Minimum number of distinct accept dates a candidate
            must have to be surfaced.  Default ``2`` matches the §9.3
            "``k ≥ 2`` confirmed accepts on distinct nights" rule.
        require_positive_min_ci: When ``True`` (default) only emit
            candidates whose *least confident* contributing record's
            ``ci_low`` is still strictly positive — every accept in
            the group cleared its own per-record statistical-accept
            gate, so the pooled signal cannot be a single lucky-CI
            spike.  Setting ``False`` keeps the rule's coverage
            broader (useful for an exploratory operator who wants to
            see weakly-suggestive evidence too).
        confirmed_only: When ``True``, restricts the input to records
            with ``confirmed == True`` — the post-V2-§6.4 ledger
            field that records whether the screening accept survived
            the same-night confirmation gate.  Default ``False`` so
            scans against pre-§6.4 ledgers (the current state of the
            archive) still produce evidence.

    Returns:
        Sorted list of :class:`CodifyCandidate` instances, ordered by
        ``(n_distinct_nights desc, mean_delta desc)`` so the strongest
        and most-replicated evidence is surfaced first.  Empty list
        when no group clears the gates.
    """
    if min_nights < 1:
        raise ValueError(f"min_nights must be >= 1, got {min_nights}")

    buckets: Dict[Tuple[str, str, str, Optional[str], str], Dict[str, Any]] = {}

    for rec in records:
        if rec.get("record_type", "iteration") != "iteration":
            continue
        if not rec.get("accepted"):
            continue
        if rec.get("no_op"):
            continue
        proposal = rec.get("proposal")
        if proposal is None:
            continue
        if confirmed_only and not rec.get("confirmed"):
            continue

        direction = _direction_key(proposal)
        if direction is None:
            continue

        class_name = str(proposal.get("class_name", ""))
        param_name = str(proposal.get("param_name", ""))
        rule_kind = str(proposal.get("rule_kind", ""))
        op = proposal.get("op")
        op_key: Optional[str] = str(op) if op else None
        # Structural ops collapse onto ``rule_kind == "structural"`` so
        # the bucket key stays uniform.  The op survives via ``op_key``.
        if op_key is not None:
            rule_kind = "structural"

        bucket_key = (class_name, param_name, rule_kind, op_key, direction)
        bucket = buckets.setdefault(
            bucket_key,
            {
                "deltas": [],
                "ci_lows": [],
                "ci_highs": [],
                "old_values": [],
                "new_values": [],
                "timestamps": [],
                "strategy_names": [],
                "confirmed_flags": [],
                "structural_kwargs_list": [],
                "dates": set(),
            },
        )
        bucket["deltas"].append(float(rec.get("delta", 0.0)))
        bucket["ci_lows"].append(float(rec.get("ci_low", 0.0)))
        bucket["ci_highs"].append(float(rec.get("ci_high", 0.0)))
        bucket["old_values"].append(proposal.get("old_value"))
        bucket["new_values"].append(proposal.get("new_value"))
        bucket["timestamps"].append(str(rec.get("timestamp", "")))
        bucket["strategy_names"].append(str(proposal.get("strategy_name", "")))
        skw = proposal.get("structural_kwargs")
        bucket["structural_kwargs_list"].append(dict(skw) if isinstance(skw, dict) else None)
        confirmed = rec.get("confirmed")
        bucket["confirmed_flags"].append(None if confirmed is None else bool(confirmed))
        date_str = _date_from_timestamp(rec.get("timestamp"))
        if date_str:
            bucket["dates"].add(date_str)

    candidates: List[CodifyCandidate] = []
    for (class_name, param_name, rule_kind, op_key, direction), data in buckets.items():
        distinct_dates: Tuple[str, ...] = tuple(sorted(data["dates"]))
        if len(distinct_dates) < min_nights:
            continue
        ci_lows = tuple(data["ci_lows"])
        if require_positive_min_ci and ci_lows and min(ci_lows) <= 0.0:
            continue
        cand = CodifyCandidate(
            class_name=class_name,
            param_name=param_name,
            rule_kind=rule_kind,
            op=op_key,
            direction=direction,
            n_accepts=len(data["deltas"]),
            distinct_dates=distinct_dates,
            deltas=tuple(data["deltas"]),
            ci_lows=ci_lows,
            ci_highs=tuple(data["ci_highs"]),
            old_values=tuple(data["old_values"]),
            new_values=tuple(data["new_values"]),
            timestamps=tuple(data["timestamps"]),
            strategy_names=tuple(data["strategy_names"]),
            confirmed_flags=tuple(data["confirmed_flags"]),
            structural_kwargs_list=tuple(data["structural_kwargs_list"]),
        )
        candidates.append(cand)

    candidates.sort(
        key=lambda c: (c.n_distinct_nights, c.mean_delta, c.n_accepts),
        reverse=True,
    )
    return candidates


def load_ledgers_for_codify_scan(
    ledger_path: str,
    *,
    include_archives: bool = True,
    archive_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read the live ledger + (optionally) every rotated archive.

    Convenience wrapper around :func:`load_ledger` that mirrors the
    :meth:`AdaptiveMutationSampler.prime_from_archives` semantics:
    archives in ``planning/done/`` are scanned in chronological
    (lexicographic) order and prepended *before* the live ledger so the
    aggregator sees evidence in time order.  Missing files / missing
    archive directories are silently no-ops so the helper is safe to
    call on a fresh checkout.

    Archive selection is *scoped to ``ledger_path``'s metric* via
    :func:`iter_metric_archives`: scanning the composite ledger pools only
    composite archives, scanning the aocc ledger pools only aocc archives.
    This prevents composite deltas (~1e-3 scale) and aocc deltas (~0.3
    scale) from mixing in the codify aggregator's bootstrap CI after the
    2026-07-09 metric flip (§12.1).  Ledger names matching no known metric
    stem classify as ``composite``, so pre-flip archive sets and test
    fixtures pool exactly as before.

    Args:
        ledger_path: Path to the live ledger (typically
            ``planning/self_improve_ledger.jsonl`` for composite or
            ``planning/self_improve_ledger_aocc.jsonl`` for aocc).
        include_archives: When ``True`` (default), also scan the
            archive directory.  Set to ``False`` for a live-only scan.
        archive_dir: Path to the archive directory.  When ``None`` the
            default is ``<ledger parent>/done`` to match the rotated
            archive convention from §12.1 (live ledger lives in
            ``planning/``, archives in ``planning/done/``).

    Returns:
        Concatenation of same-metric archive records (chronological by
        archive filename) followed by the live ledger records, in the
        order :func:`aggregate_codify_candidates` should consume them.
    """
    records: List[Dict[str, Any]] = []
    if include_archives:
        if archive_dir is None:
            archive_dir = str(pathlib.Path(ledger_path).parent / "done")
        for archive_file in iter_metric_archives(archive_dir, ledger_path):
            records.extend(load_ledger(str(archive_file)))
    records.extend(load_ledger(ledger_path))
    return records


def _live_kwarg_values(
    class_name: str,
    param_name: str,
    factories: Sequence[Callable[[], Sequence[StrategySpec]]],
) -> List[Any]:
    """Collect every value the ``(class_name, param_name)`` slot is explicitly set to.

    Walks every :class:`~panobbgo.benchmark.StrategySpec` returned by each
    factory and inspects both ``heuristics`` and ``analyzers`` entries.
    A match requires the entry's class ``__name__`` to equal
    ``class_name`` *and* the entry's kwargs dict to contain
    ``param_name`` — kwargs left at the constructor default are
    intentionally ignored, mirroring :func:`_find_targets`'s
    "param already in kwargs" predicate the catalog uses.

    Factories that raise are silently skipped — the helper is a
    best-effort scan, and a caller-supplied factory that throws should
    not break the whole codify-scan run.  Order in the returned list
    matches factory order followed by spec order; duplicates are
    intentionally preserved so the caller can count multi-spec
    coverage if needed.
    """
    values: List[Any] = []
    for factory in factories:
        try:
            specs = factory()
        except Exception:
            continue
        for spec in specs:
            for entry in getattr(spec, "heuristics", None) or []:
                try:
                    cls, kwargs = entry
                except (TypeError, ValueError):
                    continue
                if getattr(cls, "__name__", "") == class_name and param_name in kwargs:
                    values.append(kwargs[param_name])
            for entry in getattr(spec, "analyzers", None) or []:
                try:
                    cls, kwargs = entry
                except (TypeError, ValueError):
                    continue
                if getattr(cls, "__name__", "") == class_name and param_name in kwargs:
                    values.append(kwargs[param_name])
    return values


def _candidate_already_codified(
    candidate: CodifyCandidate,
    live_values: Sequence[Any],
) -> bool:
    """Decide whether the candidate's implied source edit is a no-op.

    The rule depends on the rule kind:

    * ``categorical_choice``: the candidate's direction is
      ``repr(new_value)``; the cross-check is exact ``repr`` equality
      against any live value (so ``False`` and ``"False"`` do not
      collide).
    * Numeric (``integer_add`` / ``float_uniform`` /
      ``log_uniform_perturb``): the direction is ``"up"`` or
      ``"down"`` and the candidate proposes shifting the default in
      that direction.  We compare the median of
      :attr:`candidate.new_values` against the live values: if any
      live value already meets or exceeds the proposal in the
      candidate's direction, the codify edit would not move the
      default further.  Specifically:

      - ``"up"`` is codified iff ``max(live) >= median(new_values)``
      - ``"down"`` is codified iff ``min(live) <= median(new_values)``

    * Structural ops (``op is not None``): handled by
      :func:`_structural_already_codified` against a class-membership
      dict produced by :func:`_live_class_membership` — see
      :func:`annotate_codified_status`, which branches on the candidate
      shape and calls the appropriate helper directly.  This function
      assumes the caller already filtered structural candidates out;
      it conservatively returns ``False`` on them so a mis-routed call
      cannot raise.

    When ``live_values`` is empty the candidate is *not* codified —
    the seed factory does not set the kwarg explicitly, so any codify
    PR would still be actionable (it would change the constructor
    default rather than a factory override).
    """
    if not live_values:
        return False
    if candidate.op is not None:
        return False
    if candidate.rule_kind == "categorical_choice":
        return any(repr(v) == candidate.direction for v in live_values)
    if candidate.direction not in ("up", "down"):
        return False
    try:
        new_floats = [float(v) for v in candidate.new_values]
    except (TypeError, ValueError):
        return False
    if not new_floats:
        return False
    try:
        live_floats = [float(v) for v in live_values]
    except (TypeError, ValueError):
        return False
    if not live_floats:
        return False
    target = float(np.median(new_floats))
    if candidate.direction == "up":
        return max(live_floats) >= target
    return min(live_floats) <= target


def _live_class_membership(
    class_name: str,
    factories: Sequence[Callable[[], Sequence[StrategySpec]]],
) -> Dict[str, Tuple[str, ...]]:
    """Find which seed specs already contain ``class_name`` as a heuristic / analyzer.

    Returns ``{"heuristics": (spec_name, ...), "analyzers": (spec_name, ...)}``
    listing every :class:`~panobbgo.benchmark.StrategySpec` whose
    ``heuristics`` / ``analyzers`` tuple includes a class whose
    ``__name__`` equals ``class_name``.  Spec names are recorded in
    factory-then-spec order with each spec appearing at most once per
    bucket; if the same spec ships the class in both buckets the spec
    name appears under both keys.

    Factories that raise are silently skipped — mirrors the resilience
    of :func:`_live_kwarg_values` so a misbehaving caller-supplied
    factory cannot break the whole codify-scan run.  Used by
    :func:`_structural_already_codified` to decide whether
    ``add_/drop_heuristic`` / ``add_/drop_analyzer`` candidates would
    be no-op edits.
    """
    heuristic_specs: List[str] = []
    analyzer_specs: List[str] = []
    for factory in factories:
        try:
            specs = factory()
        except Exception:
            continue
        for spec in specs:
            in_heuristics = False
            for entry in getattr(spec, "heuristics", None) or []:
                try:
                    cls, _ = entry
                except (TypeError, ValueError):
                    continue
                if getattr(cls, "__name__", "") == class_name:
                    in_heuristics = True
                    break
            if in_heuristics:
                heuristic_specs.append(spec.name)
            in_analyzers = False
            for entry in getattr(spec, "analyzers", None) or []:
                try:
                    cls, _ = entry
                except (TypeError, ValueError):
                    continue
                if getattr(cls, "__name__", "") == class_name:
                    in_analyzers = True
                    break
            if in_analyzers:
                analyzer_specs.append(spec.name)
    return {
        "heuristics": tuple(heuristic_specs),
        "analyzers": tuple(analyzer_specs),
    }


def _structural_already_codified(
    candidate: CodifyCandidate,
    membership: Mapping[str, Sequence[str]],
) -> bool:
    """Decide whether the structural candidate's implied source edit is a no-op.

    Structural ops do not target a kwarg; their "live state" is membership
    of the candidate's :attr:`~CodifyCandidate.class_name` in the seed
    spec's ``heuristics`` / ``analyzers`` tuple.  Rules — symmetric to
    :func:`_candidate_already_codified`'s ``max(live) >= median`` /
    ``min(live) <= median`` shape:

    * ``add_heuristic`` of class ``X``: codified iff at least one seed
      spec already lists ``X`` under ``heuristics``.  The codify edit
      "append ``X`` to the seed pool" would be partially redundant
      because at least one spec already carries the heuristic.
    * ``drop_heuristic`` of class ``X``: codified iff no seed spec
      lists ``X``.  The codify edit "remove ``X`` from the seed pool"
      cannot remove anything that is not already there.
    * ``add_analyzer`` / ``drop_analyzer``: same shape, against the
      ``analyzers`` bucket.

    ``membership`` is the dict :func:`_live_class_membership` returns;
    only the keys named in the candidate's op are consulted.  An
    unknown op (defensive — the catalog ships exactly the four ops
    above) classifies as not-codified so the candidate continues to
    surface.
    """
    op = candidate.direction
    if op == "add_heuristic":
        return len(membership.get("heuristics", ())) > 0
    if op == "drop_heuristic":
        return len(membership.get("heuristics", ())) == 0
    if op == "add_analyzer":
        return len(membership.get("analyzers", ())) > 0
    if op == "drop_analyzer":
        return len(membership.get("analyzers", ())) == 0
    return False


def default_codify_registries(
    metric: str = "composite",
) -> List[Callable[[], List[StrategySpec]]]:
    """Default seed-spec factories the codify-scan suppression check uses.

    The factories depend on which metric's ledger is being scanned,
    because each metric measures a different seed registry:

    * ``"composite"`` — the two factories the composite cron exercises:
      the ``quick`` registry (the historical default mode) and the
      ``loop`` registry (the catalog-exercising registry shipped
      2026-06-10).  The ``loop`` registry already includes the
      ``quick`` specs, but listing both makes the predicate behaviour
      the same whether the run was configured with ``--registry
      quick`` or ``--registry loop``.
    * ``"aocc"`` — :func:`panobbgo.harness_ioh.make_ioh_strategies`,
      the IOH-tuned registry every ``--metric aocc`` iteration
      measures (the nightly default since 2026-07-09).  Checking the
      composite registries here would compare candidates against specs
      the aocc loop never runs, so suppression would key on the wrong
      source of truth.

    Standard / full registries are intentionally excluded: their seed
    specs target the manual benchmark battery (200 / 500 evals), not
    the cron, and surfacing "already codified" candidates whose
    codification only lives in those registries would mis-direct the
    operator away from actionable evidence on the cron's regime.
    """
    if metric == "aocc":
        from panobbgo.harness_ioh import make_ioh_strategies

        return [make_ioh_strategies]
    from panobbgo.harness import _make_loop_strategies, _make_quick_strategies

    return [_make_quick_strategies, _make_loop_strategies]


def annotate_codified_status(
    candidates: Sequence[CodifyCandidate],
    *,
    registries: Optional[Sequence[Callable[[], Sequence[StrategySpec]]]] = None,
) -> None:
    """Mark each candidate's ``already_codified`` flag in-place.

    Walks every seed-spec factory in ``registries`` (default:
    :func:`default_codify_registries`), collects the live kwarg
    values for each candidate's ``(class_name, param_name)`` slot via
    :func:`_live_kwarg_values`, and sets both
    :attr:`CodifyCandidate.already_codified` and
    :attr:`CodifyCandidate.live_codified_values` accordingly.

    The result lets the CLI report suppress candidates whose implied
    source edit would be a no-op — the
    ``Sobol.scramble=False`` candidate that surfaces from the
    pre-codification archive even though
    :func:`panobbgo.harness._make_quick_strategies` already ships
    ``scramble=False`` is the motivating example.  See V2 §9.3 / the
    *Next iteration ideas* "Suppress already-codified candidates"
    entry in :doc:`/planning/SELF_IMPROVEMENT_LOG.md`.

    The annotation pass is intentionally *separate* from
    :func:`aggregate_codify_candidates` so unit tests can verify the
    raw scanner output without importing the factories and so the
    library is robust to a caller that ships a non-standard
    registry.

    Mutates ``candidates`` in place; returns ``None``.
    """
    factories: Sequence[Callable[[], Sequence[StrategySpec]]]
    if registries is None:
        factories = default_codify_registries()
    else:
        factories = registries
    for cand in candidates:
        if cand.op is not None:
            # Structural candidate — class-membership check rather than
            # kwarg-value comparison.  The relevant live data is the
            # subset of spec names that already include the candidate's
            # class in the matching bucket; surfaced in
            # ``live_codified_values`` so the CLI can print the spec
            # names alongside the ``[already codified]`` tag.
            membership = _live_class_membership(cand.class_name, factories)
            if "analyzer" in (cand.op or ""):
                relevant = membership.get("analyzers", ())
            else:
                relevant = membership.get("heuristics", ())
            cand.live_codified_values = tuple(relevant)
            cand.already_codified = _structural_already_codified(cand, membership)
        else:
            live_values = _live_kwarg_values(cand.class_name, cand.param_name, factories)
            cand.live_codified_values = tuple(live_values)
            cand.already_codified = _candidate_already_codified(cand, live_values)


# Rejections file next to each metric's ledger (§ scan hygiene, the
# 2026-08-02 / 2026-08-03 log entries): ``composite`` keeps the
# historical bare stem, ``aocc`` gets the metric-suffixed one — the same
# naming convention as :data:`LEDGER_STEM_BY_METRIC`.
REJECTIONS_STEM_BY_METRIC: Dict[str, str] = {
    "composite": "self_improve_rejections",
    "aocc": "self_improve_rejections_aocc",
}


def rejections_path_for_metric(metric: str, ledger_dir: str = DEFAULT_LEDGER_DIR) -> str:
    """Return the canonical codify-rejections path for ``metric``.

    ``composite`` → ``<ledger_dir>/self_improve_rejections.json``;
    ``aocc`` → ``<ledger_dir>/self_improve_rejections_aocc.json``.
    Raises ``ValueError`` for an unknown metric so a mistyped
    ``--metric`` fails loudly instead of silently consulting the wrong
    rejection memory (mirrors :func:`ledger_path_for_metric`).
    """
    try:
        stem = REJECTIONS_STEM_BY_METRIC[metric]
    except KeyError:
        known = ", ".join(sorted(REJECTIONS_STEM_BY_METRIC))
        raise ValueError(f"unknown metric {metric!r} (known: {known})") from None
    return str(pathlib.Path(ledger_dir) / f"{stem}.json")


#: Distinct post-rejection evidence nights required before a rejected
#: codify slot resurrects (see :meth:`CodifyRejection.suppresses`).
#: Matches the §9.3 ``min_nights`` actionability default: evidence that
#: post-dates an operator rejection must clear the same k≥2 bar as a
#: brand-new candidate, because the pre-rejection nights were already
#: adjudicated by the rejecting A/B.
DEFAULT_RESURRECT_MIN_FRESH_NIGHTS = 2


@dataclass(frozen=True)
class CodifyRejection:
    """One recorded operator rejection of a codify slot.

    The codify scan's rejection memory (the "scan hygiene" gap named in
    the 2026-08-02 and 2026-08-03 log entries): when a session A/B-tests
    a scan candidate against the *current* spec and rejects it, the
    ledger evidence that produced the candidate does not disappear — the
    scan would re-surface the identical slot every night and every
    future session would have to re-derive the rejection from the dated
    log entries by hand.  A :class:`CodifyRejection` records the
    decision next to the ledger so :func:`annotate_rejected_status` can
    suppress the slot automatically.

    Suppression is *evidence-scoped*, not permanent: post-rejection
    evidence nights can resurrect the slot.  Resurrection is gated,
    though — the post-rejection nights *alone* must reach
    ``min_fresh_nights`` distinct dates (default
    :data:`DEFAULT_RESURRECT_MIN_FRESH_NIGHTS`), the same k≥2 bar the
    §9.3 aggregation applies to a brand-new candidate.  The pre-gate
    "any single fresh night resurrects" semantics had a measured 0/3
    hit rate: three consecutive resurrected slots
    (``Sensitivity.update_interval`` and ``drop_analyzer Sensitivity``
    on 2026-08-03, ``drop_heuristic NelderMead`` on 2026-08-07) were
    each re-rejected flat by 12-seed paired A/Bs after a single fresh
    seed-42 accept night re-surfaced them — exactly the
    training-battery artifact class the original rejections named.
    Evidence that predates the rejection was already adjudicated by the
    operator's A/B and never counts toward resurrection.  A resurrected
    slot is tagged with its rejection history so the operator re-verifies
    rather than trusting pooled stats that straddle the decision.

    Attributes:
        class_name: Heuristic / analyzer class the rejected slot
            targets (matches :attr:`CodifyCandidate.class_name`).
        param_name: Kwarg slot; empty string for structural ops
            (matches :attr:`CodifyCandidate.param_name`).
        op: ``None`` for kwarg rules; the ``add_/drop_`` op name for
            structural slots (matches :attr:`CodifyCandidate.op`).
        direction: Optional direction restriction.  ``None`` (the
            default) rejects the slot in every direction; a concrete
            value (``"up"`` / ``"down"`` / a categorical ``repr`` /
            an op name) rejects only candidates with that
            :attr:`CodifyCandidate.direction`, so e.g. rejecting
            ``Sensitivity.update_interval`` *down* leaves a future
            *up* signal actionable.
        rejected_on: ``YYYY-MM-DD`` date of the rejection decision
            (the A/B session date, not the evidence nights).
        reason: One-line human-readable why — surfaced verbatim in the
            scan report so the operator need not open the log.
        log_ref: Optional pointer to the full write-up (conventionally
            the dated ``planning/SELF_IMPROVEMENT_LOG.md`` heading).
    """

    class_name: str
    param_name: str = ""
    op: Optional[str] = None
    direction: Optional[str] = None
    rejected_on: str = ""
    reason: str = ""
    log_ref: str = ""

    def matches(self, candidate: "CodifyCandidate") -> bool:
        """Slot-key equality (+ optional direction restriction)."""
        if (candidate.class_name, candidate.param_name, candidate.op) != (
            self.class_name,
            self.param_name,
            self.op,
        ):
            return False
        if self.direction is not None and candidate.direction != self.direction:
            return False
        return True

    def suppresses(
        self,
        candidate: "CodifyCandidate",
        *,
        min_fresh_nights: int = DEFAULT_RESURRECT_MIN_FRESH_NIGHTS,
    ) -> bool:
        """True while the post-rejection evidence stays under the resurrection bar.

        Counts the candidate's evidence nights strictly *after*
        :attr:`rejected_on` (``distinct_dates`` are ``YYYY-MM-DD``
        strings, so lexicographic comparison is chronological) and
        suppresses while that count is below ``min_fresh_nights``.
        Nights on or before the rejection date were covered by the
        operator's A/B and never count.  ``min_fresh_nights=1``
        restores the pre-2026-08-08 "any single fresh night
        resurrects" semantics; values below 1 are clamped to 1 so the
        predicate cannot suppress forever.  An empty ``rejected_on``
        never suppresses (a date-less rejection record is malformed
        input and :func:`load_codify_rejections` refuses it; the guard
        here keeps the predicate total for hand-built instances).
        """
        if not self.matches(candidate) or not self.rejected_on:
            return False
        n_fresh = sum(1 for d in candidate.distinct_dates if d > self.rejected_on)
        return n_fresh < max(1, int(min_fresh_nights))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "param_name": self.param_name,
            "op": self.op,
            "direction": self.direction,
            "rejected_on": self.rejected_on,
            "reason": self.reason,
            "log_ref": self.log_ref,
        }


def load_codify_rejections(path: Any) -> List[CodifyRejection]:
    """Load the codify-rejections file at ``path``.

    Accepts the canonical shape ``{"rejections": [ {...}, ... ]}`` (a
    top-level object so the file can carry a ``_comment`` key) or a
    bare top-level list.  A missing file is an empty rejection memory
    (returns ``[]``) — the feature is opt-in per metric.  Anything else
    malformed — unparseable JSON, an entry without ``class_name`` or
    without a ``rejected_on`` date — raises ``ValueError`` naming the
    offending entry: the file gates automated suppression, so a typo
    must fail the scan loudly rather than silently widen or narrow the
    memory.
    """
    p = pathlib.Path(path)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"rejections file {p} is not valid JSON: {exc}") from exc
    if isinstance(raw, dict):
        entries = raw.get("rejections", [])
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(
            f"rejections file {p}: expected an object with a 'rejections' list or a bare list, got {type(raw).__name__}"
        )
    if not isinstance(entries, list):
        raise ValueError(f"rejections file {p}: 'rejections' must be a list, got {type(entries).__name__}")
    out: List[CodifyRejection] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"rejections file {p}: entry #{i} is not an object")
        class_name = entry.get("class_name")
        rejected_on = entry.get("rejected_on")
        if not class_name or not isinstance(class_name, str):
            raise ValueError(f"rejections file {p}: entry #{i} is missing 'class_name'")
        if not rejected_on or not isinstance(rejected_on, str):
            raise ValueError(f"rejections file {p}: entry #{i} ({class_name}) is missing 'rejected_on' (YYYY-MM-DD)")
        try:
            datetime.strptime(rejected_on, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"rejections file {p}: entry #{i} ({class_name}) has malformed rejected_on={rejected_on!r} (want YYYY-MM-DD)"
            ) from None
        out.append(
            CodifyRejection(
                class_name=class_name,
                param_name=str(entry.get("param_name") or ""),
                op=entry.get("op") or None,
                direction=entry.get("direction") or None,
                rejected_on=rejected_on,
                reason=str(entry.get("reason") or ""),
                log_ref=str(entry.get("log_ref") or ""),
            )
        )
    return out


def annotate_rejected_status(
    candidates: Sequence[CodifyCandidate],
    rejections: Sequence[CodifyRejection],
    *,
    min_fresh_nights: int = DEFAULT_RESURRECT_MIN_FRESH_NIGHTS,
) -> None:
    """Mark each candidate's rejection status in-place.

    For every candidate, finds the matching rejection records (slot key
    + optional direction, per :meth:`CodifyRejection.matches`) and sets:

    * ``rejected=True`` (+ ``rejected_on`` / ``rejection_reason`` from
      the *most recent* matching rejection) when at least one matching
      rejection :meth:`~CodifyRejection.suppresses` it — i.e. fewer
      than ``min_fresh_nights`` distinct evidence nights post-date that
      rejection (see :data:`DEFAULT_RESURRECT_MIN_FRESH_NIGHTS`);
    * ``rejected=False`` but ``rejected_on`` / ``rejection_reason``
      still populated when a rejection matches yet the candidate
      carries enough fresher evidence — the CLI renders this as a
      "fresh evidence since rejection" tag so the operator re-verifies
      instead of trusting pooled stats that straddle the spec change;
    * all three fields untouched (defaults) when nothing matches.

    Kept separate from :func:`annotate_codified_status` for the same
    reason that pass is separate from the aggregator: unit tests can
    exercise the raw scanner without a rejections file, and callers
    with a non-standard memory can supply their own list.

    Mutates ``candidates`` in place; returns ``None``.
    """
    for cand in candidates:
        matching = [r for r in rejections if r.matches(cand)]
        if not matching:
            continue
        newest = max(matching, key=lambda r: r.rejected_on)
        cand.rejected_on = newest.rejected_on
        cand.rejection_reason = newest.reason
        cand.rejected = any(r.suppresses(cand, min_fresh_nights=min_fresh_nights) for r in matching)


# Default source file + factory function names the ``--apply-top`` driver
# scans for kwarg-default edits.  Broader than
# :func:`default_codify_registries` (quick + loop only): the apply driver
# also edits ``standard`` / ``full`` so a single codify run covers every
# sibling spec sharing the same heuristic mix — matching the 2026-06-28
# manual codify pattern (the ``Nearby.radius = 0.124`` shift updated
# ``Rewarding_Diverse`` plus five sibling specs across the four registry
# tiers in one PR).  Listed as a top-level constant rather than a function
# so callers can mutate-in-place for tests.
_DEFAULT_APPLY_SOURCES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "panobbgo/harness.py",
        (
            "_make_quick_strategies",
            "_make_standard_strategies",
            "_make_full_strategies",
            "_make_loop_strategies",
        ),
    ),
)

# The aocc regime measures the IOH-tuned registry, so its codify edits
# must land in panobbgo/harness_ioh.py — the composite factories above
# never run under --metric aocc and editing them from aocc evidence
# would cross-contaminate the two ~100×-different delta scales.
_DEFAULT_APPLY_SOURCES_AOCC: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "panobbgo/harness_ioh.py",
        ("make_ioh_strategies",),
    ),
)


def default_codify_apply_sources(
    metric: str = "composite",
) -> List[Tuple[str, Tuple[str, ...]]]:
    """Source files + factory function names the ``--apply-top`` driver scans.

    Returns ``[(source_path, (factory_name, ...))]``.  Each ``source_path``
    is repo-relative; the driver resolves it against the current working
    directory.  Each ``factory_name`` is the name of a top-level function
    in the source file whose body builds and returns a list of
    :class:`~panobbgo.benchmark.StrategySpec` literals — the driver walks
    those function bodies to find every ``(ClassName, {param_name:
    value, ...})`` heuristic / analyzer entry that the codify candidate
    targets.

    The source set depends on which metric's evidence is being applied,
    mirroring :func:`default_codify_registries`:

    * ``"composite"`` — ``panobbgo/harness.py``, all four registry
      tiers.  Broader than :func:`default_codify_registries` (which
      returns only the ``quick`` + ``loop`` registries, the regime the
      nightly cron measures): the apply driver covers ``standard`` /
      ``full`` too so a single codify run updates every sibling spec
      sharing the same heuristic mix.  This matches the 2026-06-28
      manual codify pattern (``Nearby.radius = 0.124`` across
      ``Rewarding_Diverse`` plus five sibling specs across all four
      registry tiers in one PR).
    * ``"aocc"`` — ``panobbgo/harness_ioh.py``'s
      ``make_ioh_strategies``, the registry every ``--metric aocc``
      iteration measures.  Without this routing, aocc evidence (which
      names IOH specs like ``Rewarding_Restart``) can never land as a
      source edit — the ``--apply-top`` driver would scan
      ``harness.py``, find no matching spec, and silently no-op.

    Returns a fresh ``list`` so callers can mutate without affecting the
    module-level constant.
    """
    table = _DEFAULT_APPLY_SOURCES_AOCC if metric == "aocc" else _DEFAULT_APPLY_SOURCES
    return [(path, names) for (path, names) in table]


def _format_value_repr(value: Any) -> str:
    """Source-text representation of ``value`` matching project style.

    Uses :func:`repr` for booleans / strings / ``None`` (so ``False``
    renders as ``False`` and ``"all"`` renders as ``'all'``), ``str()``
    for integers (no quotes / trailing ``L``), and :func:`repr` for
    floats (Python's ``repr`` chooses the shortest round-trippable
    representation, so ``repr(0.124) == '0.124'``).  Used by the
    ``--apply-top`` driver to produce the replacement source segment.

    The deliberate ``repr`` use guarantees a round-trip safe literal —
    ``ast.literal_eval(_format_value_repr(v)) == v`` for every value the
    codify pipeline can produce (numeric, boolean, string, ``None``).
    """
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return repr(value)


@dataclass(frozen=True)
class CodifyEdit:
    """One concrete source-file edit derived from a :class:`CodifyCandidate`.

    Produced by :func:`derive_codify_edits` (which walks the seed-spec
    factory functions and finds every site where the candidate's slot is
    set explicitly) and consumed by :func:`apply_codify_edits` (which
    rewrites the source file).

    The edit carries both the AST coordinates (``lineno`` / ``col_offset``
    / ``end_lineno`` / ``end_col_offset``) and the exact textual
    replacement so the apply step is a pure string operation — no AST
    round-trip that would reformat the surrounding code.  Repeatedly
    applying the same edit list is idempotent: after the first apply the
    old source segment no longer matches, so a second
    :func:`apply_codify_edits` pass is a no-op against the now-codified
    file (the per-site current-direction guard in
    :func:`_scan_source_for_kwarg_edits` ensures sites that already
    satisfy the proposal are not re-listed).

    Attributes:
        source_path: Repo-relative path of the source file the edit
            applies to (e.g. ``"panobbgo/harness.py"``).
        factory_name: Name of the function in ``source_path`` whose body
            contains the edited :class:`StrategySpec` literal.
        spec_name: ``StrategySpec.name`` extracted from the surrounding
            constructor call.  ``"<unknown>"`` when the call uses a
            non-literal name (rare; the project convention is a string
            literal).
        class_name: Heuristic / analyzer class targeted by the edit
            (e.g. ``"Nearby"``).
        param_name: Kwarg name on ``class_name`` being shifted
            (e.g. ``"radius"``).
        rule_kind: ``"log_uniform_perturb"`` / ``"integer_add"`` /
            ``"float_uniform"`` / ``"categorical_choice"`` — copied from
            the originating candidate so consumers can disambiguate
            without re-aggregating.
        direction: ``"up"`` / ``"down"`` / ``repr(value)`` from the
            originating candidate.  Used by
            :func:`_scan_source_for_kwarg_edits` to filter sites that
            already satisfy the proposal in the candidate's direction.
        old_value: Literal value present in the source before the edit
            (parsed via :func:`ast.literal_eval`).
        new_value: Value the edit replaces ``old_value`` with — always
            equal to :meth:`CodifyCandidate.proposed_codify_value`.
        lineno: 1-indexed line of the value literal's start.
        col_offset: 0-indexed column of the value literal's start.
        end_lineno: 1-indexed line of the value literal's end.
        end_col_offset: 0-indexed column of the value literal's end.
        old_source: Exact text segment being replaced (slice of the
            source file at the AST coordinates).
        new_source: Replacement text segment (the new value formatted
            via :func:`_format_value_repr`).
    """

    source_path: str
    factory_name: str
    spec_name: str
    class_name: str
    param_name: str
    rule_kind: str
    direction: str
    old_value: Any
    new_value: Any
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int
    old_source: str
    new_source: str

    def to_dict(self) -> Dict[str, Any]:
        """Plain-Python dict for JSON serialisation / diagnostic prints.

        Mirrors :meth:`CodifyCandidate.to_dict` — every value is JSON-safe
        via :func:`_to_plain` so the dict round-trips through
        :func:`json.dumps` / :func:`json.loads` without precision loss.
        """
        return {
            "source_path": self.source_path,
            "factory_name": self.factory_name,
            "spec_name": self.spec_name,
            "class_name": self.class_name,
            "param_name": self.param_name,
            "rule_kind": self.rule_kind,
            "direction": self.direction,
            "old_value": _to_plain(self.old_value),
            "new_value": _to_plain(self.new_value),
            "lineno": int(self.lineno),
            "col_offset": int(self.col_offset),
            "end_lineno": int(self.end_lineno),
            "end_col_offset": int(self.end_col_offset),
            "old_source": self.old_source,
            "new_source": self.new_source,
        }


def _extract_keyword_value(call: ast.Call, name: str) -> Optional[ast.AST]:
    """Return the AST node bound to keyword ``name`` on ``call`` (or ``None``)."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _safe_literal_eval(node: ast.AST) -> Tuple[bool, Any]:
    """Best-effort :func:`ast.literal_eval` returning ``(ok, value)``.

    Returns ``(False, None)`` when the node isn't a literal — protects
    callers that need to filter out computed expressions (e.g. ``5 * dim``)
    without raising.
    """
    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return False, None


def _should_apply_at_site(
    *,
    direction: str,
    rule_kind: str,
    current_value: Any,
    new_value: Any,
) -> bool:
    """Decide whether a kwarg site should be edited given the candidate direction.

    Conservative policy that respects deliberately-different settings on
    sibling specs:

    * ``categorical_choice`` (``direction = repr(new_value)``): edit iff
      the current value isn't already at the target.
    * Numeric ``"up"``: edit iff ``current_value < new_value`` — the
      site is still below the consensus; updating it advances it toward
      the proposal.  Sites already at or above the proposal are left
      alone (they were probably deliberately set there).
    * Numeric ``"down"``: edit iff ``current_value > new_value``.
    * Numeric without a direction (``new_value is None`` or
      ``direction not in ("up", "down")``): never edit (defensive — the
      caller should have filtered the candidate out before reaching
      here).

    The matching ``current_value < new_value`` predicate for ``"up"`` is
    *strict* so re-applying the same edit list is idempotent: after the
    first apply the site sits at ``new_value`` and the predicate
    evaluates ``False``.  Same shape as
    :func:`_candidate_already_codified`'s ``max(live) >= median`` rule
    but applied per-site rather than across the full ``live_values``
    set.
    """
    if rule_kind == "categorical_choice":
        return repr(current_value) != repr(new_value)
    if direction == "up":
        try:
            return float(current_value) < float(new_value)
        except (TypeError, ValueError):
            return False
    if direction == "down":
        try:
            return float(current_value) > float(new_value)
        except (TypeError, ValueError):
            return False
    return False


def _scan_source_for_kwarg_edits(
    source_path: str,
    *,
    factory_names: Sequence[str],
    class_name: str,
    param_name: str,
    rule_kind: str,
    direction: str,
    new_value: Any,
) -> List[CodifyEdit]:
    """Find every ``(class_name, {param_name: literal, ...})`` site in named factories.

    AST-based: parses ``source_path`` with :func:`ast.parse`, walks every
    top-level function whose name is in ``factory_names``, and within
    those functions finds every :class:`StrategySpec` literal — i.e.
    every ``StrategySpec(name=..., heuristics=[...], analyzers=[...])``
    call.  Within each spec's ``heuristics`` / ``analyzers`` list,
    finds every ``(ClassName, {…})`` tuple where ``ClassName`` matches
    ``class_name`` and ``param_name`` appears in the dict literal.

    For each site, the per-site direction guard
    (:func:`_should_apply_at_site`) decides whether the edit should fire
    — sites already at-or-beyond the proposal are skipped so
    deliberately-different sibling specs (e.g. ``BayesOpt_GP`` shipping
    ``Nearby(radius=0.05)`` while everything else uses ``0.124``) are
    left alone.  Sites whose value isn't a Python literal (e.g.
    ``patience=5 * dim``) are also skipped — the AST can't safely
    rewrite a computed expression to a literal.

    Returns the list of :class:`CodifyEdit` objects with source spans
    already populated; pair with :func:`apply_codify_edits` to commit
    the changes.  Returns an empty list when the source file doesn't
    exist or when no site matches (so the caller's loop over multiple
    source files is silent on miss).
    """
    path = pathlib.Path(source_path)
    if not path.exists():
        return []
    text = path.read_text()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # The source file doesn't parse — surface no edits rather than
        # raising, so a misbehaving source file doesn't break the apply
        # driver mid-run.  The CLI prints a warning when the returned
        # edit list is empty.
        return []
    edits: List[CodifyEdit] = []
    factory_set = set(factory_names)
    for top in ast.walk(tree):
        if not isinstance(top, ast.FunctionDef):
            continue
        if top.name not in factory_set:
            continue
        # Within the factory, walk every nested Call to find StrategySpec(…)
        # literals.  The convention is StrategySpec(name=…, heuristics=[…],
        # analyzers=[…]) so we only inspect calls whose func.id is
        # "StrategySpec" — drops nested helper / wrapper calls.
        for inner in ast.walk(top):
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            if isinstance(func, ast.Attribute):
                func_name = func.attr
            elif isinstance(func, ast.Name):
                func_name = func.id
            else:
                continue
            if func_name != "StrategySpec":
                continue
            # Spec name (best-effort): only string literals are extracted;
            # a non-literal name (e.g. f-string) reports "<unknown>" rather
            # than crashing.
            spec_name_node = _extract_keyword_value(inner, "name")
            spec_name = "<unknown>"
            if isinstance(spec_name_node, ast.Constant) and isinstance(spec_name_node.value, str):
                spec_name = spec_name_node.value
            # Inspect heuristics + analyzers buckets symmetrically.
            for bucket_name in ("heuristics", "analyzers"):
                bucket = _extract_keyword_value(inner, bucket_name)
                if bucket is None or not isinstance(bucket, (ast.List, ast.Tuple)):
                    continue
                for entry in bucket.elts:
                    if not isinstance(entry, ast.Tuple) or len(entry.elts) != 2:
                        continue
                    cls_node, dict_node = entry.elts
                    if not isinstance(cls_node, ast.Name):
                        continue
                    if cls_node.id != class_name:
                        continue
                    if not isinstance(dict_node, ast.Dict):
                        continue
                    for key_node, value_node in zip(dict_node.keys, dict_node.values):
                        if not isinstance(key_node, ast.Constant):
                            continue
                        if key_node.value != param_name:
                            continue
                        ok, current_value = _safe_literal_eval(value_node)
                        if not ok:
                            continue
                        if not _should_apply_at_site(
                            direction=direction,
                            rule_kind=rule_kind,
                            current_value=current_value,
                            new_value=new_value,
                        ):
                            continue
                        # ast nodes carry start + end coordinates since
                        # Python 3.8.  ``end_lineno`` / ``end_col_offset``
                        # are 1-indexed / 0-indexed respectively.
                        if value_node.end_lineno is None or value_node.end_col_offset is None:
                            continue
                        old_source = ast.get_source_segment(text, value_node) or ""
                        edits.append(
                            CodifyEdit(
                                source_path=source_path,
                                factory_name=top.name,
                                spec_name=spec_name,
                                class_name=class_name,
                                param_name=param_name,
                                rule_kind=rule_kind,
                                direction=direction,
                                old_value=current_value,
                                new_value=new_value,
                                lineno=value_node.lineno,
                                col_offset=value_node.col_offset,
                                end_lineno=value_node.end_lineno,
                                end_col_offset=value_node.end_col_offset,
                                old_source=old_source,
                                new_source=_format_value_repr(new_value),
                            )
                        )
                        # First matching key in the dict is the binding —
                        # don't keep walking the dict (Python dict literals
                        # with duplicate keys are pathological).
                        break
    return edits


#: Structural op → bucket name it targets on :class:`StrategySpec`.
#: ``add_/drop_heuristic`` target ``heuristics``; ``add_/drop_analyzer``
#: target ``analyzers``.  Unknown ops are left out — callers that see an
#: unrecognised op skip the candidate rather than raising.
_STRUCTURAL_OPS_TO_BUCKET: Dict[str, str] = {
    "add_heuristic": "heuristics",
    "drop_heuristic": "heuristics",
    "add_analyzer": "analyzers",
    "drop_analyzer": "analyzers",
}


def _byte_to_lineno_col(byte_offset: int, line_starts: Sequence[int]) -> Tuple[int, int]:
    """Convert a byte offset back to (1-indexed lineno, 0-indexed col_offset).

    ``line_starts`` is the same ascending list of newline-start offsets
    that :func:`_apply_edits_to_text` computes — line ``k`` starts at
    ``line_starts[k - 1]``.  Uses :func:`bisect.bisect_right` so a byte
    at ``line_starts[k]`` (i.e. the start of line ``k + 1``) resolves to
    ``(k + 1, 0)`` rather than ``(k, len(line k))``, matching the
    inverse of ``line_starts[lineno - 1] + col_offset`` used by
    :func:`_apply_edits_to_text`.
    """
    lineno = bisect.bisect_right(list(line_starts), byte_offset)
    if lineno < 1:
        lineno = 1
    col_offset = byte_offset - line_starts[lineno - 1]
    return lineno, col_offset


def _format_structural_kwargs(kwargs: Optional[Dict[str, Any]]) -> str:
    """Render an ``add_*`` entry's kwargs dict as project-style source.

    Double-quoted keys + :func:`_format_value_repr` values, insertion
    order preserved — matches the ``(ClassName, {"param": value})``
    convention used across the seed-spec factories.  ``None`` / empty
    renders as ``{}`` (constructor defaults).
    """
    if not kwargs:
        return "{}"
    inner = ", ".join(f'"{k}": {_format_value_repr(v)}' for k, v in kwargs.items())
    return "{" + inner + "}"


_STRUCTURAL_BUCKET_TO_IMPORT_MODULE: Dict[str, str] = {
    "heuristics": "panobbgo.heuristics",
    "analyzers": "panobbgo.analyzers",
}


def _derive_import_edit(
    text: str,
    line_starts: List[int],
    factory: ast.FunctionDef,
    *,
    module: str,
    class_name: str,
    source_path: str,
    op: str,
) -> Optional[CodifyEdit]:
    """Rewrite the factory's ``from <module> import ...`` to bind ``class_name``.

    Seed-spec factories import their heuristic / analyzer classes
    inside the function body (e.g. ``from panobbgo.heuristics import
    Center, Random``), so a structural ``add_*`` entry edit for a class
    the factory has never used would raise ``NameError`` at run time.
    This helper finds the factory's matching :class:`ast.ImportFrom`
    and produces a :class:`CodifyEdit` replacing it with the same
    import plus ``class_name`` inserted in sorted position (aliases are
    preserved verbatim).

    Returns ``None`` when the class is already bound by the import,
    or when the factory contains no import from ``module`` — in the
    latter case the caller ships the entry edit alone and the operator
    resolves the binding manually (the post-apply parse validation in
    :func:`apply_codify_edits` only guards *syntax*, not name
    resolution).
    """
    for node in ast.walk(factory):
        if not isinstance(node, ast.ImportFrom) or node.module != module:
            continue
        if any(alias.name == class_name for alias in node.names):
            return None
        if node.end_lineno is None or node.end_col_offset is None:
            return None
        rendered = [
            (alias.name, f"{alias.name} as {alias.asname}" if alias.asname else alias.name) for alias in node.names
        ]
        rendered.append((class_name, class_name))
        rendered.sort(key=lambda pair: pair[0])
        new_source = f"from {module} import " + ", ".join(src for _name, src in rendered)
        start_byte = line_starts[node.lineno - 1] + node.col_offset
        end_byte = line_starts[node.end_lineno - 1] + node.end_col_offset
        return CodifyEdit(
            source_path=source_path,
            factory_name=factory.name,
            spec_name="<import>",
            class_name=class_name,
            param_name="",
            rule_kind="structural",
            direction=op,
            old_value=None,
            new_value=None,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            old_source=text[start_byte:end_byte],
            new_source=new_source,
        )
    return None


def _scan_source_for_structural_edits(
    source_path: str,
    *,
    factory_names: Sequence[str],
    class_name: str,
    op: str,
    target_spec_names: Optional[Set[str]] = None,
    add_kwargs: Optional[Dict[str, Any]] = None,
) -> List[CodifyEdit]:
    """Find every :class:`StrategySpec` site to structurally mutate for ``class_name``.

    The structural sibling of :func:`_scan_source_for_kwarg_edits` — where
    that function edits *values* inside ``(ClassName, {param: value, ...})``
    dict literals, this one adds or removes the surrounding tuple entry
    from the spec's ``heuristics`` / ``analyzers`` list literal.

    Behaviour by ``op``:

    * ``drop_heuristic`` / ``drop_analyzer``: emit one
      :class:`CodifyEdit` per matching ``(ClassName, {...})`` tuple in
      the target bucket.  The removal span covers the tuple *plus* the
      trailing comma and the whitespace up to the start of the next
      entry (or ``]`` when this was the last entry), so the surviving
      source is well-formatted rather than left with a stray comma or
      trailing blank line.
    * ``add_heuristic`` / ``add_analyzer``: emit a single zero-width
      insertion :class:`CodifyEdit` at the position just *before* the
      closing ``]`` of the target bucket.  The inserted text is
      ``(ClassName, <kwargs>),\\n<indent>`` where ``<indent>`` matches
      the column offset of the bucket's first existing entry (falling
      back to 12 spaces — the ``StrategySpec(...)`` convention used
      across ``_make_*_strategies``) and ``<kwargs>`` renders
      ``add_kwargs`` via :func:`_format_structural_kwargs` (``{}`` =
      constructor defaults when the candidate carries none).  When the
      last existing entry has no trailing comma (compact single-line
      buckets like ``heuristics=[(Random, {})]``), a ``,`` is inserted
      first so the resulting list literal stays syntactically valid.
      When ``class_name`` is not already bound by the factory's
      ``from panobbgo.heuristics import ...`` (or ``.analyzers``)
      statement, an additional :class:`CodifyEdit` rewrites that import
      with the class inserted in sorted position — one import edit per
      (factory, module) even when several specs in the factory receive
      the class.  A factory with no matching import statement gets no
      import edit (the entry edit still lands; the operator resolves
      the binding manually).

    Safety guards keep the primitive conservative:

    * ``drop_*`` skips specs whose bucket has only one entry — dropping
      the last heuristic / analyzer would leave the spec unable to
      generate points / observe events.
    * ``drop_*`` skips specs whose bucket does not contain ``class_name``
      (nothing to drop).
    * ``add_*`` skips specs whose bucket already contains ``class_name``
      (matches :func:`_structural_already_codified`, so the primitive
      is idempotent under re-runs).
    * When ``target_spec_names`` is given, only specs whose
      :attr:`~panobbgo.benchmark.StrategySpec.name` appears in that
      set are edited.  The ``--apply-top`` CLI passes the candidate's
      :attr:`~CodifyCandidate.strategy_names` here so structural edits
      only touch the specs the ledger actually accumulated evidence
      against (unlike kwarg edits, which safely propagate across every
      matching spec).

    Args:
        source_path: Repo-relative path of a source file whose top-level
            functions build seed specs.
        factory_names: Names of the top-level factory functions in
            ``source_path`` whose bodies :class:`StrategySpec` literals
            live in.
        class_name: Heuristic / analyzer class ``__name__`` the op
            targets (e.g. ``"LatinHypercube"``).
        op: One of ``add_heuristic`` / ``drop_heuristic`` /
            ``add_analyzer`` / ``drop_analyzer``.  Unknown ops return
            an empty list rather than raising.
        target_spec_names: Optional filter — restrict edits to specs
            whose ``name`` is in this set.  ``None`` (default) allows
            every scanned spec through.  An empty set disables all
            edits (defensive).

    Returns:
        Zero or more :class:`CodifyEdit` objects.  Empty list on
        missing file, syntax error, unknown op, or when no
        surviving spec matches the safety guards above.
    """
    bucket_name = _STRUCTURAL_OPS_TO_BUCKET.get(op)
    if bucket_name is None:
        return []
    if target_spec_names is not None and not target_spec_names:
        return []
    path = pathlib.Path(source_path)
    if not path.exists():
        return []
    text = path.read_text()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    edits: List[CodifyEdit] = []
    import_edited: Set[Tuple[str, str]] = set()
    factory_set = set(factory_names)
    # Precompute line starts for byte-offset arithmetic — matches the
    # convention in :func:`_apply_edits_to_text` so the resulting
    # ``CodifyEdit`` coordinates apply cleanly.
    line_starts: List[int] = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(i + 1)

    def byte_offset(lineno: int, col_offset: int) -> int:
        if lineno <= 0 or lineno > len(line_starts):
            return -1
        return line_starts[lineno - 1] + col_offset

    for top in ast.walk(tree):
        if not isinstance(top, ast.FunctionDef):
            continue
        if top.name not in factory_set:
            continue
        for inner in ast.walk(top):
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            if isinstance(func, ast.Attribute):
                func_name = func.attr
            elif isinstance(func, ast.Name):
                func_name = func.id
            else:
                continue
            if func_name != "StrategySpec":
                continue
            spec_name_node = _extract_keyword_value(inner, "name")
            spec_name = "<unknown>"
            if isinstance(spec_name_node, ast.Constant) and isinstance(spec_name_node.value, str):
                spec_name = spec_name_node.value
            if target_spec_names is not None and spec_name not in target_spec_names:
                continue
            bucket = _extract_keyword_value(inner, bucket_name)
            if bucket is None or not isinstance(bucket, ast.List):
                continue
            # Collect every existing ``(ClassName, {...})`` entry so the
            # add / drop decisions can inspect current membership.
            existing_entries: List[ast.Tuple] = []
            for entry in bucket.elts:
                if not isinstance(entry, ast.Tuple) or len(entry.elts) != 2:
                    continue
                cls_node = entry.elts[0]
                if isinstance(cls_node, ast.Name) and cls_node.id == class_name:
                    existing_entries.append(entry)

            if op.startswith("drop_"):
                if not existing_entries:
                    # Nothing to drop — leave the spec untouched.
                    continue
                # Safety: never drop the last entry in a bucket.  The
                # spec's downstream consumers (strategy factory, analyzer
                # pipeline) expect at least one heuristic; dropping the
                # last one would silently produce an unusable spec.
                if len(bucket.elts) <= 1:
                    continue
                for entry in existing_entries:
                    if entry.end_lineno is None or entry.end_col_offset is None:
                        continue
                    start_byte = byte_offset(entry.lineno, entry.col_offset)
                    end_byte = byte_offset(entry.end_lineno, entry.end_col_offset)
                    if start_byte < 0 or end_byte < 0 or end_byte > len(text):
                        continue
                    # Extend the removal span to consume the trailing
                    # comma + any inter-entry whitespace so the surviving
                    # list literal is well-formatted.  Two cases:
                    #
                    # * **Middle entry** (next non-whitespace is another
                    #   tuple): consume the trailing ``,``, the newline
                    #   after it, and the indent on the next line so
                    #   the sibling below shifts up cleanly.
                    # * **Last entry** (next non-whitespace is ``]``):
                    #   consuming forward would eat the ``]``'s leading
                    #   indent and leave the closing bracket at the
                    #   wrong column.  Instead extend the removal
                    #   *backwards* through the entry's own leading
                    #   newline + indent so the whole line the entry
                    #   sat on disappears cleanly.
                    #
                    # Idempotent under re-runs — after the removal the
                    # ``existing_entries`` scan comes up empty on the
                    # now-codified source, so :func:`derive_codify_edits`
                    # returns ``[]``.
                    scan = end_byte
                    if scan < len(text) and text[scan] == ",":
                        scan += 1
                    peek = scan
                    while peek < len(text) and text[peek] in " \t\n":
                        peek += 1
                    is_last_entry = peek < len(text) and text[peek] == "]"
                    if is_last_entry:
                        # Extend backwards through the entry's leading
                        # whitespace + newline so the closing ``]``
                        # inherits the pre-entry indentation.
                        back = start_byte - 1
                        while back >= 0 and text[back] in " \t":
                            back -= 1
                        if back >= 0 and text[back] == "\n":
                            start_byte_final = back
                        else:
                            # No leading newline (compact single-line
                            # bucket) — fall back to the original start.
                            start_byte_final = start_byte
                        end_byte_expanded = scan
                    else:
                        start_byte_final = start_byte
                        tmp = scan
                        while tmp < len(text) and text[tmp] in " \t":
                            tmp += 1
                        saw_newline = False
                        if tmp < len(text) and text[tmp] == "\n":
                            tmp += 1
                            saw_newline = True
                            while tmp < len(text) and text[tmp] in " \t":
                                tmp += 1
                        end_byte_expanded = tmp if saw_newline else scan
                    start_lineno, start_col_offset = _byte_to_lineno_col(start_byte_final, line_starts)
                    end_lineno, end_col_offset = _byte_to_lineno_col(end_byte_expanded, line_starts)
                    old_source = text[start_byte_final:end_byte_expanded]
                    edits.append(
                        CodifyEdit(
                            source_path=source_path,
                            factory_name=top.name,
                            spec_name=spec_name,
                            class_name=class_name,
                            param_name="",
                            rule_kind="structural",
                            direction=op,
                            old_value=None,
                            new_value=None,
                            lineno=start_lineno,
                            col_offset=start_col_offset,
                            end_lineno=end_lineno,
                            end_col_offset=end_col_offset,
                            old_source=old_source,
                            new_source="",
                        )
                    )
            elif op.startswith("add_"):
                if existing_entries:
                    # Already codified — matches
                    # ``_structural_already_codified``'s add-branch rule.
                    continue
                if bucket.end_lineno is None or bucket.end_col_offset is None:
                    continue
                # Insertion strategy varies by bucket population:
                #
                # * Non-empty bucket (the common case) — insert AFTER
                #   the last entry's trailing comma so the new entry
                #   lands on its own line at the same indent as its
                #   siblings.  Sample layout::
                #
                #       heuristics=[
                #           (COBYQA, {"scale": True}),  <-- last entry
                #           (NewClass, {}),             <-- inserted
                #       ],
                #
                # * Empty bucket — insert *inside* the ``[]`` with a
                #   newline on each side so the new entry sits on its
                #   own line matching the enclosing ``[`` indent.
                #
                # Both variants preserve the source's existing
                # indentation convention rather than guessing.
                kwargs_source = _format_structural_kwargs(add_kwargs)
                if bucket.elts:
                    last_entry = bucket.elts[-1]
                    if last_entry.end_lineno is None or last_entry.end_col_offset is None:
                        continue
                    last_end = byte_offset(last_entry.end_lineno, last_entry.end_col_offset)
                    if last_end < 0 or last_end > len(text):
                        continue
                    # Consume a trailing comma if present so the new
                    # entry lands after it (not before).  When the last
                    # entry has NO trailing comma (compact single-line
                    # buckets like ``heuristics=[(Random, {})]``), the
                    # insertion must supply one itself — inserting the
                    # new tuple directly after ``(Random, {})`` would
                    # produce the *call expression* ``(Random,
                    # {})(NewClass, {})``, which is syntactically valid
                    # to the eye but crashes at import time.
                    insert_byte = last_end
                    had_trailing_comma = insert_byte < len(text) and text[insert_byte] == ","
                    if had_trailing_comma:
                        insert_byte += 1
                    # Match the last entry's indent (its column offset
                    # is the number of leading spaces before it on its
                    # line) so the new entry aligns visually.
                    indent = " " * last_entry.col_offset
                    insert_lineno, insert_col = _byte_to_lineno_col(insert_byte, line_starts)
                    comma_prefix = "" if had_trailing_comma else ","
                    new_entry_source = f"{comma_prefix}\n{indent}({class_name}, {kwargs_source}),"
                else:
                    # Empty bucket (e.g. ``analyzers=[]``): insert
                    # inline between ``[`` and ``]`` since there's no
                    # existing entry style to match.  Result:
                    # ``analyzers=[(NewClass, {})]``.  A future manual
                    # tune can reformat to multi-line if the bucket
                    # gains more entries.
                    open_byte = byte_offset(bucket.lineno, bucket.col_offset + 1)
                    if open_byte < 0 or open_byte > len(text):
                        continue
                    insert_byte = open_byte
                    insert_lineno, insert_col = _byte_to_lineno_col(insert_byte, line_starts)
                    new_entry_source = f"({class_name}, {kwargs_source})"
                edits.append(
                    CodifyEdit(
                        source_path=source_path,
                        factory_name=top.name,
                        spec_name=spec_name,
                        class_name=class_name,
                        param_name="",
                        rule_kind="structural",
                        direction=op,
                        old_value=None,
                        new_value=None,
                        lineno=insert_lineno,
                        col_offset=insert_col,
                        end_lineno=insert_lineno,
                        end_col_offset=insert_col,
                        old_source="",
                        new_source=new_entry_source,
                    )
                )
                # Make sure the factory can actually resolve the class
                # it now constructs — one import rewrite per (factory,
                # module) even when several specs in the factory
                # receive the class.
                module = _STRUCTURAL_BUCKET_TO_IMPORT_MODULE.get(bucket_name)
                if module is not None and (top.name, module) not in import_edited:
                    import_edit = _derive_import_edit(
                        text,
                        line_starts,
                        top,
                        module=module,
                        class_name=class_name,
                        source_path=source_path,
                        op=op,
                    )
                    import_edited.add((top.name, module))
                    if import_edit is not None:
                        edits.append(import_edit)
    return edits


def derive_codify_edits(
    candidate: CodifyCandidate,
    *,
    sources: Optional[Sequence[Tuple[str, Sequence[str]]]] = None,
) -> List[CodifyEdit]:
    """Compute every source edit that codifies ``candidate`` into the seed-spec factories.

    Kwarg candidates (``op is None``): walks every named factory function
    in ``sources``, finds every ``(class_name, {param_name: literal,
    ...})`` heuristic / analyzer entry, and produces a
    :class:`CodifyEdit` that replaces the literal with the candidate's
    :meth:`~CodifyCandidate.proposed_codify_value`.  Sites already at or
    beyond the proposal in the candidate's direction are skipped (so
    deliberately-tighter sibling specs are left alone).

    Structural candidates (``op is not None``): dispatches to
    :func:`_scan_source_for_structural_edits` which produces list-entry
    insertions (``add_heuristic`` / ``add_analyzer``) or removals
    (``drop_heuristic`` / ``drop_analyzer``) in the target bucket.
    The ledger's :attr:`~CodifyCandidate.strategy_names` set narrows the
    edit scope to the specs the ledger accumulated evidence against
    (unlike kwarg edits, which safely propagate across every matching
    spec).  Empty when no strategy_names are recorded — defensive: a
    structural candidate without a recorded strategy_name cannot be
    routed to a specific spec.

    Args:
        candidate: The codify candidate produced by
            :func:`aggregate_codify_candidates` (typically the top of a
            human-readable scan).  For kwarg candidates the
            :meth:`~CodifyCandidate.proposed_codify_value` is consulted
            for the value the edits ship.
        sources: Source files + factory function names to scan.
            Defaults to :func:`default_codify_apply_sources` (which
            spans all four registry tiers in ``panobbgo/harness.py``).
            Tests pass a custom list pointing at a synthetic source
            file to keep the suite hermetic.

    Returns:
        List of :class:`CodifyEdit` objects, one per matching site, in
        the order encountered by :func:`ast.walk` (factory-then-spec
        order).  Empty when
        :meth:`~CodifyCandidate.proposed_codify_value` returns ``None``,
        when a structural candidate carries no recorded strategy names,
        or when no source site clears the primitive's safety guards.
    """
    if sources is None:
        sources = default_codify_apply_sources()
    if candidate.op is not None:
        target_spec_names: Optional[Set[str]] = set(name for name in candidate.strategy_names if name)
        # A structural candidate that never recorded a strategy_name is
        # not routable — refuse to guess.  Kwarg candidates fall back to
        # "every spec" but structural ops can't (adding a heuristic to
        # every spec that lacks it is a much bigger edit than the
        # ledger's evidence supports).
        if not target_spec_names:
            return []
        edits: List[CodifyEdit] = []
        for source_path, factory_names in sources:
            edits.extend(
                _scan_source_for_structural_edits(
                    source_path,
                    factory_names=tuple(factory_names),
                    class_name=candidate.class_name,
                    op=candidate.op,
                    target_spec_names=target_spec_names,
                    add_kwargs=candidate.consensus_structural_kwargs(),
                )
            )
        return edits
    proposed = candidate.proposed_codify_value()
    if proposed is None:
        return []
    edits = []
    for source_path, factory_names in sources:
        edits.extend(
            _scan_source_for_kwarg_edits(
                source_path,
                factory_names=tuple(factory_names),
                class_name=candidate.class_name,
                param_name=candidate.param_name,
                rule_kind=candidate.rule_kind,
                direction=candidate.direction,
                new_value=proposed,
            )
        )
    return edits


def _apply_edits_to_text(text: str, edits: Sequence[CodifyEdit]) -> str:
    """Apply ``edits`` to ``text`` in reverse byte-offset order.

    Reverse order ensures earlier edits don't invalidate the line/col
    coordinates of later ones.  Edits referencing line / column pairs
    outside the text bounds are silently skipped (defensive — the AST
    coordinates are authoritative for the parsed snapshot, so an
    out-of-bound edit signals that ``text`` is not what was parsed).
    """
    if not edits:
        return text
    line_starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(i + 1)

    def start_offset(edit: CodifyEdit) -> int:
        if edit.lineno <= 0 or edit.lineno > len(line_starts):
            return -1
        return line_starts[edit.lineno - 1] + edit.col_offset

    sorted_edits = sorted(edits, key=start_offset, reverse=True)
    result = text
    for edit in sorted_edits:
        if edit.lineno <= 0 or edit.lineno > len(line_starts):
            continue
        if edit.end_lineno <= 0 or edit.end_lineno > len(line_starts):
            continue
        start = line_starts[edit.lineno - 1] + edit.col_offset
        end = line_starts[edit.end_lineno - 1] + edit.end_col_offset
        if start < 0 or end > len(result) or start > end:
            continue
        result = result[:start] + edit.new_source + result[end:]
    return result


def apply_codify_edits(
    edits: Sequence[CodifyEdit],
    *,
    dry_run: bool = False,
) -> Dict[str, str]:
    """Apply :class:`CodifyEdit` objects to disk (or simulate via ``dry_run``).

    Edits are grouped by :attr:`CodifyEdit.source_path`; each source
    file is read once, all its edits applied via
    :func:`_apply_edits_to_text` (in reverse byte-offset order so
    earlier edits don't invalidate later coordinates), and written back
    to disk unless ``dry_run`` is set.

    Args:
        edits: Edits to apply.  Typically the output of
            :func:`derive_codify_edits` for a single
            :class:`CodifyCandidate`, but ``apply_codify_edits`` doesn't
            enforce same-candidate provenance — a caller can mix edits
            from multiple candidates if they don't overlap.
        dry_run: When ``True``, compute the new file contents but do
            **not** write them to disk.  The returned dict still maps
            each touched path to its new contents so a CLI consumer can
            print a preview / diff before the operator commits the
            apply.

    Returns:
        Dict mapping each touched ``source_path`` to its new contents
        (after every edit applies).  Empty when ``edits`` is empty.
    """
    by_path: Dict[str, List[CodifyEdit]] = {}
    for edit in edits:
        by_path.setdefault(edit.source_path, []).append(edit)
    out: Dict[str, str] = {}
    for source_path, file_edits in by_path.items():
        path = pathlib.Path(source_path)
        if not path.exists():
            continue
        text = path.read_text()
        new_text = _apply_edits_to_text(text, file_edits)
        # Safety net: never land a syntactically-broken source file.  A
        # coordinate bug in an edit primitive (e.g. the pre-2026-07-30
        # missing-comma insertion on single-line buckets) must surface
        # as a loud skip, not a silently corrupted registry that every
        # later harness run crashes on.
        if source_path.endswith(".py"):
            try:
                ast.parse(new_text)
            except SyntaxError as exc:
                print(
                    f"apply_codify_edits: refusing to write {source_path} — "
                    f"edited result does not parse ({exc}).  "
                    "This is a bug in the edit primitive; nothing was written.",
                    file=sys.stderr,
                )
                continue
        out[source_path] = new_text
        if not dry_run and new_text != text:
            path.write_text(new_text)
    return out


def apply_codify_candidate(
    candidate: CodifyCandidate,
    *,
    sources: Optional[Sequence[Tuple[str, Sequence[str]]]] = None,
    dry_run: bool = False,
) -> Tuple[List[CodifyEdit], Dict[str, str]]:
    """Convenience wrapper: derive edits for ``candidate``, then apply them.

    Combines :func:`derive_codify_edits` and :func:`apply_codify_edits`
    into the single call the ``--apply-top`` CLI driver makes per
    invocation.  Equivalent to::

        edits = derive_codify_edits(candidate, sources=sources)
        modified_files = apply_codify_edits(edits, dry_run=dry_run)

    Returns ``(edits, modified_files)`` so the caller can inspect both
    the per-site decisions and the final file contents.  ``edits`` is
    empty for structural candidates and for candidates whose
    :meth:`~CodifyCandidate.proposed_codify_value` is ``None``;
    ``modified_files`` is empty when ``edits`` is empty or when no edit
    actually changed its source file.
    """
    edits = derive_codify_edits(candidate, sources=sources)
    modified = apply_codify_edits(edits, dry_run=dry_run)
    return edits, modified


def _slot_key_string(slot_key: Tuple[str, str, Optional[str]]) -> str:
    """Human-readable form of a :attr:`CodifyCandidate.slot_key` tuple.

    Used both as the *machine-readable dedup marker* embedded in every
    codify PR body (see :func:`codify_pr_marker`) and as a hopefully-stable
    identifier the operator can grep for.  Kwarg slots render as
    ``ClassName.param_name``; structural slots render as
    ``ClassName::structural::op_name`` — the double-colon separator keeps
    the two spaces disjoint (no kwarg could ever produce a slot string
    containing ``::``).
    """
    class_name, param_name, op = slot_key
    if op is None:
        return f"{class_name}.{param_name}"
    return f"{class_name}::structural::{op}"


def codify_pr_marker(candidate: CodifyCandidate) -> str:
    """Machine-readable marker embedded in every codify PR body.

    The queued ``codify-scan --open-pr`` driver (V2 §9.5 step 4) uses
    this string to *dedup* against ``gh pr list --state open``: an open
    PR carrying the marker for a given
    :attr:`CodifyCandidate.slot_key` means the codify work has already
    been proposed and the driver should skip re-opening a second PR for
    the same slot.  The direction is intentionally excluded from the
    marker so a same-slot opposite-direction signal is treated as
    "an existing PR already covers this slot, supersede it in review"
    rather than "open a duplicate" — matches the §12.3 step 0 lesson
    (``gh pr list --state open`` first; open PRs are the source of
    truth for in-flight work) and the docstring on
    :attr:`CodifyCandidate.slot_key`.

    Format: ``codify-slot: <slot_key_string>`` where
    :func:`_slot_key_string` renders ``(class, param, op)`` in the
    unambiguous form documented on that helper.  The literal
    ``codify-slot:`` prefix makes the marker greppable both in PR bodies
    and in local diffs / commit messages if a downstream driver decides
    to embed it there too.
    """
    return f"codify-slot: {_slot_key_string(candidate.slot_key)}"


def find_open_pr_for_slot(
    candidate: CodifyCandidate,
    open_prs: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the first open PR whose body / title carries ``candidate``'s marker.

    ``open_prs`` is the parsed JSON output of ``gh pr list --state open
    --json number,title,body,headRefName`` — a list of dicts with at
    least ``"title"`` / ``"body"`` keys.  Missing keys are treated as
    empty strings so a partial JSON payload doesn't raise.  Returns
    ``None`` when no PR matches the marker (the driver should proceed
    to open a fresh PR).

    Matching is a plain substring check on the concatenation of title +
    body so the marker survives GitHub's Markdown rendering (backticks /
    HTML comments don't rewrite text).  Case-sensitive: the marker is
    machine-generated so a case-insensitive check would only produce
    false positives from operator-written text.
    """
    marker = codify_pr_marker(candidate)
    for pr in open_prs:
        title = str(pr.get("title") or "")
        body = str(pr.get("body") or "")
        if marker in title or marker in body:
            return pr
    return None


def codify_pr_title(candidate: CodifyCandidate) -> str:
    """One-line PR title summarising the codify shift.

    Format for kwarg candidates: ``codify(<Class>.<param>): shift default
    <old_repr> -> <new_repr> (<direction>, ledger evidence)``.  Uses
    :func:`_format_value_repr` for both values so the title reads
    naturally for ints (``5 -> 7``), floats (``0.1 -> 0.124``) and
    booleans (``True -> False``).  Falls back to a slot-only title for
    structural candidates (``codify(<Class>): <op_name>``).

    Deliberately short (< 80 chars for the vast majority of slots) so
    the PR list rendering in ``gh pr list`` fits one line.  The
    :func:`codify_pr_marker` string lives in the *body*, not the title —
    keeping the title human-readable while the dedup marker stays
    stable across title edits.
    """
    if candidate.op is not None:
        return f"codify({candidate.class_name}): {candidate.op} (ledger evidence)"
    proposed = candidate.proposed_codify_value()
    if proposed is None:
        return f"codify({candidate.class_name}.{candidate.param_name}): {candidate.direction} (ledger evidence)"
    old_repr = _format_value_repr(candidate.old_values[-1]) if candidate.old_values else "?"
    new_repr = _format_value_repr(proposed)
    return (
        f"codify({candidate.class_name}.{candidate.param_name}): "
        f"shift default {old_repr} -> {new_repr} "
        f"({candidate.direction}, ledger evidence)"
    )


def codify_pr_branch_name(
    candidate: CodifyCandidate,
    *,
    prefix: str = "claude/codify",
) -> str:
    """Stable branch name for a codify PR.

    Format: ``<prefix>-<class_snake>-<param_snake>-<direction>`` for
    kwarg candidates; ``<prefix>-<class_snake>-<op_name>`` for
    structural candidates.  The class / param are lower-cased so the
    branch matches the common ``feat/`` / ``fix/`` casing on the repo.
    Non-alphanumerics in the direction (e.g. ``repr(False) == "False"``,
    ``repr("all") == "'all'"``) are collapsed to ``_`` so the branch is
    always a valid git ref.

    Deliberately *does not* embed a timestamp: the dedup path is the
    marker in :func:`codify_pr_marker`, and a fixed branch name makes
    "close-and-reopen" of the same slot (rare but possible) fall onto
    the same branch instead of accumulating branch litter.  For a
    version-bumped codify (same slot, different proposal) the operator
    can pass ``--pr-branch-suffix`` on the CLI to disambiguate; the
    default is stable.

    Args:
        candidate: The chosen kwarg / structural candidate.
        prefix: Branch prefix.  Default ``"claude/codify"`` matches the
            existing bot-branch naming convention (the watcher
            infrastructure keys on the ``claude/`` prefix; see V2 §9.5
            step 4 "branch naming convention" in the follow-up idea).

    Returns:
        Sanitised branch name.  Guaranteed to satisfy git's ref-format
        rules for the identifiers Panobbgo ships (ASCII heuristic /
        analyzer class names, snake_case kwarg names, up / down /
        repr-form directions).
    """
    import re as _re

    def _sanitize(s: str) -> str:
        s = s.strip().lower()
        s = _re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_") or "x"

    parts: List[str] = [prefix.rstrip("/-")]
    parts.append(_sanitize(candidate.class_name))
    if candidate.op is not None:
        parts.append(_sanitize(candidate.op))
    else:
        parts.append(_sanitize(candidate.param_name or "kwarg"))
        parts.append(_sanitize(candidate.direction))
    return "-".join(parts)


def codify_pr_body(
    candidate: CodifyCandidate,
    edits: Sequence[CodifyEdit] = (),
    *,
    marker: Optional[str] = None,
    base_branch: str = "master",
) -> str:
    """Draft PR body citing the ledger evidence behind ``candidate``.

    Assembled as plain Markdown so ``gh pr create --body`` renders it on
    GitHub without post-processing.  Contains four sections:

    * **Codify slot** — the machine-readable marker (see
      :func:`codify_pr_marker`) plus the human-readable slot label.
      The marker line is what :func:`find_open_pr_for_slot` matches
      against so an operator (or a follow-up run of the same driver)
      does not open a duplicate PR against the same slot.
    * **Ledger evidence** — one row per accepted iteration: date,
      Δ, CI, strategy, old → new.  Copied straight from the candidate's
      per-record fields; no re-aggregation, no bootstrap re-runs — the
      body reflects exactly what the operator sees in the codify-scan
      report.
    * **Proposed source edit** — one line per :class:`CodifyEdit`
      (``source_path:lineno  factory/spec: old -> new``) so a reviewer
      can pattern-match the diff without opening the file.  Empty when
      no edit list is passed (e.g. an early-exit dry-run).
    * **Test plan** — the ``benchmark_harness.py compare --statistical``
      invocation that reproduces the ledger evidence, mirroring the
      wording of the ``--fix`` PR template.

    Args:
        candidate: Codify candidate the PR embodies.
        edits: Source edits the PR ships (typically the output of
            :func:`derive_codify_edits`).  When empty the "Proposed
            source edit" section renders a warning line rather than an
            empty bullet list.
        marker: Machine-readable dedup marker.  Defaults to
            :func:`codify_pr_marker` — override only in tests that want
            to check body content without re-computing the marker.
        base_branch: Branch the PR merges into.  Surfaced in the test-
            plan snippet as the ``compare --statistical --base`` flag
            reference.  Default ``"master"``.

    Returns:
        Full PR body as a single string.  Trailing newline included so
        ``gh pr create --body-file`` sees a proper terminator.
    """
    if marker is None:
        marker = codify_pr_marker(candidate)
    slot = _slot_key_string(candidate.slot_key)
    lines: List[str] = []
    lines.append(f"<!-- {marker} -->")
    lines.append("")
    lines.append("## Codify slot")
    lines.append("")
    lines.append(f"- **Slot**: `{slot}`")
    lines.append(f"- **Direction**: `{candidate.direction}`")
    lines.append(f"- **Rule kind**: `{candidate.rule_kind}`")
    if candidate.op is not None:
        lines.append(f"- **Structural op**: `{candidate.op}`")
    proposed = candidate.proposed_codify_value()
    if proposed is not None:
        lines.append(f"- **Proposed value**: `{_format_value_repr(proposed)}`")
    if candidate.live_codified_values:
        live_repr = ", ".join(_format_value_repr(v) for v in candidate.live_codified_values)
        lines.append(f"- **Live seed value(s) before edit**: {live_repr}")
    lines.append("")
    lines.append("## Ledger evidence")
    lines.append("")
    lines.append(
        f"- {candidate.n_accepts} accept(s) across {candidate.n_distinct_nights} distinct night(s) "
        f"(dates: {', '.join(candidate.distinct_dates)})."
    )
    lines.append(f"- Mean Δ = `{candidate.mean_delta:+.4f}`; min per-record `ci_low` = `{candidate.min_ci_low:+.4f}`.")
    n_confirmed = sum(1 for f in candidate.confirmed_flags if f is True)
    if n_confirmed:
        lines.append(f"- Confirmed by same-night gate: {n_confirmed}/{candidate.n_accepts}.")
    strategies = sorted({s for s in candidate.strategy_names if s})
    if strategies:
        lines.append(f"- Strategies seen: {', '.join(f'`{s}`' for s in strategies)}.")
    lines.append("")
    lines.append("| Date | Strategy | Δ | CI | Old → New |")
    lines.append("|---|---|---|---|---|")
    for i in range(candidate.n_accepts):
        ts = candidate.timestamps[i][:10] if candidate.timestamps[i] else "?"
        strat = candidate.strategy_names[i] or "?"
        old_repr = _format_value_repr(candidate.old_values[i])
        new_repr = _format_value_repr(candidate.new_values[i])
        confirmed_tag = ""
        if candidate.confirmed_flags[i] is True:
            confirmed_tag = " ✓"
        elif candidate.confirmed_flags[i] is False:
            confirmed_tag = " ✗"
        lines.append(
            f"| {ts} | `{strat}` | `{candidate.deltas[i]:+.4f}` | "
            f"`[{candidate.ci_lows[i]:+.4f}, {candidate.ci_highs[i]:+.4f}]` | "
            f"`{old_repr}` → `{new_repr}`{confirmed_tag} |"
        )
    lines.append("")
    lines.append("## Proposed source edit")
    lines.append("")
    if edits:
        for edit in edits:
            lines.append(
                f"- `{edit.source_path}:{edit.lineno}` "
                f"`{edit.factory_name}/{edit.spec_name}`: "
                f"`{edit.class_name}.{edit.param_name} = {edit.old_source} -> {edit.new_source}`"
            )
    else:
        lines.append("- (no source edits derived — see `codify-scan --apply-top --apply-dry-run` output)")
    lines.append("")
    lines.append("## Test plan")
    lines.append("")
    lines.append("- [ ] `uv run pytest tests/test_self_improve.py`")
    lines.append(
        "- [ ] `uv run python benchmark_harness.py compare --statistical "
        f"--randomize --base {base_branch}` (reproduces the paired-bootstrap "
        "verdict the ledger evidence rests on)"
    )
    lines.append("")
    lines.append(
        "*Auto-drafted by `codify-scan --open-pr` (V2 §9.5 step 4).  "
        "The `codify-slot` marker in the HTML comment above is used by "
        "the driver to dedup against open PRs — do not remove it.*"
    )
    lines.append("")
    return "\n".join(lines)


# Numeric mutation kinds the widening detector reasons about.  Categorical
# rules and structural ops have no meaningful "wider bound" — they live on
# discrete choice sets or op names.
_NUMERIC_RULE_KINDS: Tuple[str, ...] = (
    "log_uniform_perturb",
    "integer_add",
    "float_uniform",
)


@dataclass
class WideningCandidate:
    """A pair of bidirectional codify candidates proposing a wider catalog bound.

    The 2026-06-17 :func:`aggregate_codify_candidates` scanner surfaces
    bidirectional patterns — same ``(class_name, param_name)`` slot
    accumulating accepts in both ``direction="up"`` and
    ``direction="down"`` across multiple nights.  Both directions are
    legitimate signal: the bandit genuinely finds value moving the kwarg
    up *and* moving it down, depending on the instance.  The right
    action for these is rarely a default shift (which direction?) but a
    *catalog bound update* so the bandit's exploration focuses where the
    observed accepts live and gets some headroom outside that range.
    See the *Mutation-bound widening rule* idea under
    :doc:`/planning/SELF_IMPROVEMENT_LOG.md` "Next iteration ideas".

    The proposed bound is the observed range
    (``min`` / ``max`` of every accepted ``new_value`` across both
    directions) widened by :attr:`widen_factor`.  For
    ``log_uniform_perturb`` and ``float_uniform`` the widening is
    multiplicative; ``integer_add`` uses the same rule but rounds the
    proposed bounds outward to integers so the catalog's ``bounds`` tuple
    stays integer-typed.  A proposed bound that's *tighter* than the
    current one is also a useful signal — it focuses bandit draws on
    where the evidence supports them — so callers should treat
    "wider vs tighter" as informational, not an accept gate.

    Attributes:
        class_name: Heuristic / analyzer class targeted by both
            directions.
        param_name: Kwarg slot.
        rule_kind: One of the entries of :data:`_NUMERIC_RULE_KINDS`.
            Categorical / structural candidates never make it into
            widening — the detector skips them.
        current_bounds: Bounds the catalog rule currently advertises for
            this slot (``rule.bounds``).  ``None`` when no catalog rule
            targets the slot (the operator can still act on the proposed
            bound by adding a new rule).
        observed_lo: Minimum ``new_value`` across both directions.
        observed_hi: Maximum ``new_value`` across both directions.
        proposed_lo: Observed minimum widened in the safe direction
            (downward for log / float kinds; rounded toward zero for
            integers).  See :func:`_widen_numeric_bounds` for the rule.
        proposed_hi: Observed maximum widened upward (multiplicative for
            log / float; rounded outward for integers).
        widen_factor: The multiplicative widening factor used.
        up_candidate: The :class:`CodifyCandidate` carrying the
            ``direction="up"`` evidence.
        down_candidate: The :class:`CodifyCandidate` carrying the
            ``direction="down"`` evidence.
        n_accepts: Combined accepts across both directions.
        distinct_dates: Sorted union of the contributing dates.
    """

    class_name: str
    param_name: str
    rule_kind: str
    current_bounds: Optional[Tuple[float, float]]
    observed_lo: float
    observed_hi: float
    proposed_lo: float
    proposed_hi: float
    widen_factor: float
    up_candidate: "CodifyCandidate"
    down_candidate: "CodifyCandidate"

    @property
    def n_accepts(self) -> int:
        return self.up_candidate.n_accepts + self.down_candidate.n_accepts

    @property
    def distinct_dates(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.up_candidate.distinct_dates) | set(self.down_candidate.distinct_dates)))

    @property
    def n_distinct_nights(self) -> int:
        return len(self.distinct_dates)

    @property
    def slot_key(self) -> Tuple[str, str, Optional[str]]:
        """Slot identifier mirroring :attr:`CodifyCandidate.slot_key`.

        Widening candidates always target a kwarg rule (``op is None``)
        so a future ``--open-pr`` driver can dedup against open codify
        PRs using the same key shape both candidate types produce.
        """
        return (self.class_name, self.param_name, None)

    @property
    def proposal_is_wider(self) -> bool:
        """``True`` when the proposed bound exceeds the current one in either direction."""
        if self.current_bounds is None:
            return True
        cur_lo, cur_hi = self.current_bounds
        return self.proposed_lo < cur_lo or self.proposed_hi > cur_hi

    @property
    def proposal_is_tighter(self) -> bool:
        """``True`` when the proposed bound is strictly inside the current one in both directions.

        A tighter proposal is still actionable evidence — it concentrates
        bandit draws on the observed range so dormant edges of the
        catalog stop wasting effort.  The CLI surfaces both flags so the
        operator can prioritise.
        """
        if self.current_bounds is None:
            return False
        cur_lo, cur_hi = self.current_bounds
        return self.proposed_lo > cur_lo and self.proposed_hi < cur_hi

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "param_name": self.param_name,
            "rule_kind": self.rule_kind,
            "current_bounds": (
                None
                if self.current_bounds is None
                else [_to_plain(self.current_bounds[0]), _to_plain(self.current_bounds[1])]
            ),
            "observed_lo": _to_plain(self.observed_lo),
            "observed_hi": _to_plain(self.observed_hi),
            "proposed_lo": _to_plain(self.proposed_lo),
            "proposed_hi": _to_plain(self.proposed_hi),
            "widen_factor": float(self.widen_factor),
            "proposal_is_wider": bool(self.proposal_is_wider),
            "proposal_is_tighter": bool(self.proposal_is_tighter),
            "n_accepts": int(self.n_accepts),
            "n_distinct_nights": int(self.n_distinct_nights),
            "distinct_dates": list(self.distinct_dates),
            "up_candidate": self.up_candidate.to_dict(),
            "down_candidate": self.down_candidate.to_dict(),
        }


def _widen_numeric_bounds(
    observed_lo: float,
    observed_hi: float,
    rule_kind: str,
    *,
    widen_factor: float,
) -> Tuple[float, float]:
    """Compute the widened ``(lo, hi)`` for a numeric rule kind.

    Rule per ``rule_kind``:

    * ``log_uniform_perturb`` — multiplicative on both sides: the
      observed range is a log-scale window, so dividing the lower end
      by ``widen_factor`` and multiplying the upper end is the symmetric
      operation in log space.  Lower bound is floored at a tiny positive
      value (``1e-12``) because :class:`MutationRule` rejects
      non-positive ``log_uniform_perturb`` values.
    * ``integer_add`` — same multiplicative rule but rounded *outward*
      (lower bound via :func:`math.floor`, upper via
      :func:`math.ceil`) so the proposed window is at least as wide as
      the multiplicative one.  Lower bound is clipped to ``1`` when the
      observed minimum is positive — most integer-typed catalog kwargs
      are pool sizes / iteration counts where zero would be a degenerate
      configuration.  When the observed minimum is itself ``<= 0`` we
      leave the sign untouched so a future negative-int kwarg would
      survive widening.
    * ``float_uniform`` — multiplicative on the absolute values; we keep
      the sign of the bound (so a negative-valued knob widens away from
      zero on both sides).  When ``observed_lo`` is exactly zero we
      leave it at zero (the operator likely wants the bound to start at
      zero).

    The widen factor is intentionally fixed by the caller — different
    rule kinds want different defaults (log rules want a larger factor
    because the observed range is itself logarithmic) but enforcing the
    caller's choice keeps the function unsurprising.
    """
    import math as _math

    if widen_factor <= 1.0:
        raise ValueError(f"widen_factor must be > 1.0, got {widen_factor}")
    if rule_kind == "log_uniform_perturb":
        # Both ends are positive (rule construction enforces).  Divide
        # the floor, multiply the ceiling — symmetric in log space.
        new_lo = max(1e-12, observed_lo / widen_factor)
        new_hi = observed_hi * widen_factor
        return (float(new_lo), float(new_hi))
    if rule_kind == "integer_add":
        if observed_lo > 0:
            new_lo_f = max(1.0, observed_lo / widen_factor)
        elif observed_lo == 0:
            new_lo_f = 0.0
        else:
            new_lo_f = observed_lo * widen_factor
        new_hi_f = observed_hi * widen_factor if observed_hi >= 0 else observed_hi / widen_factor
        return (float(_math.floor(new_lo_f)), float(_math.ceil(new_hi_f)))
    if rule_kind == "float_uniform":
        if observed_lo > 0:
            new_lo = observed_lo / widen_factor
        elif observed_lo == 0:
            new_lo = 0.0
        else:
            new_lo = observed_lo * widen_factor
        if observed_hi >= 0:
            new_hi = observed_hi * widen_factor
        else:
            new_hi = observed_hi / widen_factor
        return (float(new_lo), float(new_hi))
    raise ValueError(f"Unsupported rule_kind for widening: {rule_kind!r}")


def _auto_tune_widen_factor(
    observed_lo: float,
    observed_hi: float,
    current_bounds: Optional[Tuple[float, float]],
    rule_kind: str,
    *,
    min_factor: float = 1.1,
    max_factor: float = 2.5,
    fallback: float = 1.5,
) -> float:
    """Size a widen factor from observed spread relative to the catalog bound.

    Intuition (closes the *Auto-tune widen factor from observed spread*
    follow-up seeded under the 2026-06-19 widening-detector ship): when
    the bandit's accepts cluster in a narrow window inside a wide catalog
    range, the observed range is high-agreement evidence — a *larger*
    widen factor is appropriate because the bandit has already converged
    and a small headroom would barely give it room to discover the next
    win.  When accepts span most of the catalog range, agreement is low
    — a *smaller* factor focuses on the consensus rather than ballooning
    the bounds further.

    Spread is measured in the rule's natural scale:

    * ``log_uniform_perturb`` — log-space ratio
      ``log(observed_hi / observed_lo) / log(current_hi / current_lo)``.
      Log because the rule samples log-uniform; a linear ratio on a
      log-distributed quantity would mis-rank pairs that span the same
      number of decades.
    * ``integer_add`` / ``float_uniform`` — linear ratio
      ``(observed_hi - observed_lo) / (current_hi - current_lo)``.

    ``ratio`` is clipped to ``[0.0, 1.0]`` and linearly interpolated:

    .. math::

       \\mathrm{factor} = \\mathrm{max\\_factor}
         - (\\mathrm{max\\_factor} - \\mathrm{min\\_factor}) \\cdot
           \\mathrm{ratio}

    so ``ratio = 0`` (perfect agreement) returns ``max_factor`` and
    ``ratio = 1`` (observed spans the catalog) returns ``min_factor``.

    When ``current_bounds`` is ``None`` (no rule targets the slot) or
    the catalog span is degenerate (``cur_hi <= cur_lo``), the relative-
    spread signal is unavailable — return ``fallback``.  Callers can
    pass the same fixed ``widen_factor`` they would have used pre-
    auto-tune as ``fallback`` so the no-rule case stays compatible.

    Args:
        observed_lo: Minimum ``new_value`` across both directions.
        observed_hi: Maximum ``new_value`` across both directions.
        current_bounds: The catalog rule's bounds, or ``None``.
        rule_kind: One of :data:`_NUMERIC_RULE_KINDS`.
        min_factor: Returned at ratio = 1.  Must be ``> 1.0``.
        max_factor: Returned at ratio = 0.  Must be ``>= min_factor``.
        fallback: Returned when the spread signal is unavailable.

    Returns:
        A widen factor strictly ``> 1.0`` (validated against
        ``min_factor`` and ``fallback`` constraints) that
        :func:`_widen_numeric_bounds` accepts.
    """
    import math as _math

    if min_factor <= 1.0:
        raise ValueError(f"min_factor must be > 1.0, got {min_factor}")
    if max_factor < min_factor:
        raise ValueError(f"max_factor must be >= min_factor; got {max_factor} < {min_factor}")
    if fallback <= 1.0:
        raise ValueError(f"fallback must be > 1.0, got {fallback}")
    if current_bounds is None:
        return float(fallback)
    cur_lo, cur_hi = current_bounds
    if rule_kind == "log_uniform_perturb":
        # Log-space ratio.  Defensive against non-positive bounds (the
        # rule construction enforces positivity, but a ledger may carry
        # values that don't match — fall back rather than NaN).
        if observed_lo <= 0 or observed_hi <= 0 or cur_lo <= 0 or cur_hi <= 0:
            return float(fallback)
        catalog_span = _math.log(cur_hi / cur_lo)
        if catalog_span <= 0:
            return float(fallback)
        observed_span = max(0.0, _math.log(observed_hi / observed_lo))
        ratio = observed_span / catalog_span
    elif rule_kind in ("integer_add", "float_uniform"):
        catalog_span = cur_hi - cur_lo
        if catalog_span <= 0:
            return float(fallback)
        observed_span = max(0.0, observed_hi - observed_lo)
        ratio = observed_span / catalog_span
    else:
        return float(fallback)

    ratio = max(0.0, min(1.0, ratio))
    factor = max_factor - (max_factor - min_factor) * ratio
    # Guard against floating-point drift just below min_factor.
    factor = max(min_factor, factor)
    return float(factor)


def _catalog_numeric_bounds(
    catalog: "MutationCatalog",
    class_name: str,
    param_name: str,
    rule_kind: str,
) -> Optional[Tuple[float, float]]:
    """Look up the bounds of the matching numeric rule in ``catalog``.

    Returns ``None`` when no rule targets the
    ``(class_name, param_name, rule_kind)`` slot.  Multiple rules on the
    same slot (e.g. the ``NLSHADE_RSP.k_rank`` ``float_uniform`` plus
    ``categorical_choice`` pair shipped 2026-06-04) survive as separate
    bandit arms — the widening detector matches the *numeric* rule for
    the bidirectional pattern.  Structural rules are ignored here; only
    :class:`MutationRule` instances carry numeric bounds.
    """
    if rule_kind not in _NUMERIC_RULE_KINDS:
        return None
    for rule in catalog.rules:
        if not isinstance(rule, MutationRule):
            continue
        if rule.kind != rule_kind:
            continue
        if rule.class_name != class_name:
            continue
        if rule.param_name != param_name:
            continue
        return (float(rule.bounds[0]), float(rule.bounds[1]))
    return None


def detect_widening_candidates(
    candidates: Sequence["CodifyCandidate"],
    *,
    catalog: Optional["MutationCatalog"] = None,
    widen_factor: float = 1.5,
    auto_tune: bool = False,
    auto_tune_min_factor: float = 1.1,
    auto_tune_max_factor: float = 2.5,
) -> List[WideningCandidate]:
    """Pair bidirectional codify candidates into bound-widening proposals.

    Walks ``candidates`` and pairs each ``(class_name, param_name,
    rule_kind)`` slot whose direction set contains both ``"up"`` and
    ``"down"``.  Each pair becomes one :class:`WideningCandidate`
    carrying:

    * the observed ``new_value`` range (min across the down accepts to
      max across the up accepts),
    * the catalog rule's current bounds (looked up by class / param /
      rule_kind via :func:`_catalog_numeric_bounds`),
    * the proposed bounds (observed range widened by ``widen_factor``
      via :func:`_widen_numeric_bounds`).

    Only numeric rule kinds (``log_uniform_perturb`` / ``integer_add`` /
    ``float_uniform``) are considered — categorical and structural
    candidates don't carry a meaningful "wider bound".  Candidates with
    ``op is not None`` (structural) are skipped, matching the planning
    doc's "op == None — only kwarg candidates" precondition.

    Args:
        candidates: Output of :func:`aggregate_codify_candidates`.
            Typically the scanner's `min_nights >= 2` filtering is
            already applied; the widening detector applies no further
            gates beyond the bidirectional requirement.
        catalog: Catalog to consult for the current bounds.  Defaults to
            :func:`default_catalog`.  Pass an explicit catalog when the
            loop runs against a non-default rule set.
        widen_factor: Multiplicative widening factor applied to the
            observed range.  Default ``1.5`` matches the *Mutation-bound
            widening* idea sketch in :doc:`/planning/SELF_IMPROVEMENT_LOG.md`.
            When ``auto_tune=True`` this value is the *fallback* the
            detector uses for slots whose ``current_bounds`` are
            ``None`` (no catalog rule targets the slot, so the relative-
            spread signal is unavailable).
        auto_tune: When ``True``, per-candidate widen factor is sized to
            the observed spread relative to the catalog bound via
            :func:`_auto_tune_widen_factor`.  Narrow observed spreads
            (high agreement) produce larger factors so the proposed
            bound has headroom; wide spreads (low agreement) produce
            smaller factors so the proposed bound focuses on the
            consensus.  When ``False`` (default), every candidate uses
            the fixed ``widen_factor`` — byte-identical to the
            pre-2026-06-22 behaviour.
        auto_tune_min_factor: Returned at observed-spread / catalog-
            span ratio = 1.  Must be ``> 1.0``.  Only consulted when
            ``auto_tune=True``.  Default ``1.1`` keeps the proposed
            bound strictly wider than the observed range while
            preserving most of the consensus.
        auto_tune_max_factor: Returned at observed-spread / catalog-
            span ratio = 0.  Must be ``>= auto_tune_min_factor``.  Only
            consulted when ``auto_tune=True``.  Default ``2.5`` gives a
            tightly-clustered observed range generous headroom outside
            the consensus window.

    Returns:
        Sorted list of :class:`WideningCandidate` instances, ordered by
        ``(n_distinct_nights desc, n_accepts desc, class_name asc)`` so
        the strongest bidirectional evidence surfaces first.  Empty list
        when no slot carries both directions.
    """
    if catalog is None:
        catalog = default_catalog()
    by_slot: Dict[Tuple[str, str, str], Dict[str, "CodifyCandidate"]] = {}
    for cand in candidates:
        if cand.op is not None:
            continue
        if cand.rule_kind not in _NUMERIC_RULE_KINDS:
            continue
        if cand.direction not in ("up", "down"):
            continue
        slot = (cand.class_name, cand.param_name, cand.rule_kind)
        bucket = by_slot.setdefault(slot, {})
        # A second candidate on the same (slot, direction) shouldn't happen
        # in well-formed scanner output (aggregate_codify_candidates groups
        # by direction first), but if it does, keep the stronger one.
        prev = bucket.get(cand.direction)
        if prev is None or cand.n_distinct_nights > prev.n_distinct_nights:
            bucket[cand.direction] = cand

    out: List[WideningCandidate] = []
    for (class_name, param_name, rule_kind), bucket in by_slot.items():
        if "up" not in bucket or "down" not in bucket:
            continue
        up_cand = bucket["up"]
        down_cand = bucket["down"]
        # Pool every observed new_value from both directions; coerce
        # gracefully — non-numeric entries shouldn't appear in numeric
        # rule kinds, but be defensive about ledger drift.
        observed: List[float] = []
        for v in up_cand.new_values:
            try:
                observed.append(float(v))
            except (TypeError, ValueError):
                pass
        for v in down_cand.new_values:
            try:
                observed.append(float(v))
            except (TypeError, ValueError):
                pass
        if not observed:
            continue
        observed_lo = min(observed)
        observed_hi = max(observed)
        current = _catalog_numeric_bounds(catalog, class_name, param_name, rule_kind)
        if auto_tune:
            effective_factor = _auto_tune_widen_factor(
                observed_lo,
                observed_hi,
                current,
                rule_kind,
                min_factor=auto_tune_min_factor,
                max_factor=auto_tune_max_factor,
                fallback=widen_factor,
            )
        else:
            effective_factor = widen_factor
        proposed_lo, proposed_hi = _widen_numeric_bounds(
            observed_lo,
            observed_hi,
            rule_kind,
            widen_factor=effective_factor,
        )
        out.append(
            WideningCandidate(
                class_name=class_name,
                param_name=param_name,
                rule_kind=rule_kind,
                current_bounds=current,
                observed_lo=float(observed_lo),
                observed_hi=float(observed_hi),
                proposed_lo=float(proposed_lo),
                proposed_hi=float(proposed_hi),
                widen_factor=float(effective_factor),
                up_candidate=up_cand,
                down_candidate=down_cand,
            )
        )

    # Sort by strongest evidence first: most distinct nights desc, then
    # combined accept count desc, then class_name asc (deterministic
    # tie-break so the report order is stable across runs).
    out.sort(key=lambda w: (-w.n_distinct_nights, -w.n_accepts, w.class_name, w.param_name))
    return out


__all__ = [
    "MutationRule",
    "StructuralMutationRule",
    "MutationProposal",
    "MutationCatalog",
    "MutationRuleStats",
    "AdaptiveMutationSampler",
    "RuleKey",
    "default_catalog",
    "default_structural_catalog",
    "apply_mutation",
    "LoopConfig",
    "LoopIterationRecord",
    "LoopGuardRecord",
    "LoopHoldoutRecord",
    "HoldoutDriftAggregate",
    "aggregate_holdout_drift",
    "LadderEntry",
    "SelfImprover",
    "load_ledger",
    "CodifyCandidate",
    "CodifyEdit",
    "CodifyRejection",
    "DEFAULT_RESURRECT_MIN_FRESH_NIGHTS",
    "aggregate_codify_candidates",
    "annotate_codified_status",
    "annotate_rejected_status",
    "load_codify_rejections",
    "rejections_path_for_metric",
    "default_codify_registries",
    "default_codify_apply_sources",
    "derive_codify_edits",
    "apply_codify_edits",
    "apply_codify_candidate",
    "codify_pr_marker",
    "codify_pr_title",
    "codify_pr_body",
    "codify_pr_branch_name",
    "find_open_pr_for_slot",
    "load_ledgers_for_codify_scan",
    "WideningCandidate",
    "detect_widening_candidates",
]
