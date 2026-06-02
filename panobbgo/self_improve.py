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

import json
import pathlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from panobbgo.benchmark import StrategySpec
from panobbgo.harness import (
    BenchmarkHarness,
    HarnessConfig,
    HarnessResult,
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


def _find_targets(
    specs: Sequence[StrategySpec],
    strategy_pattern: str,
    class_name: str,
    param_name: str,
) -> List[Tuple[int, str, int, Any]]:
    """Locate every ``(spec, bucket, entry, current_value)`` that matches.

    A hit is produced iff the spec name contains ``strategy_pattern`` (or
    the pattern is empty), the heuristic / analyzer class name equals
    ``class_name``, *and* ``param_name`` is already present in the kwargs
    dict.  Locations where the class matches but the kwarg is missing are
    intentionally skipped — we only tune existing parameters.
    """
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
                if param_name in kwargs:
                    hits.append((si, bucket_name, ei, kwargs[param_name]))
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
                k_hits = _find_targets(specs, rule.strategy_pattern, rule.class_name, rule.param_name)
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
    """

    rule_key: RuleKey
    n_attempts: int = 0
    n_accepts: int = 0

    @property
    def accept_rate(self) -> float:
        """Empirical accept rate, or 0.0 with no attempts."""
        if self.n_attempts == 0:
            return 0.0
        return self.n_accepts / self.n_attempts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.rule_key[0],
            "param_name": self.rule_key[1],
            "rule_kind": self.rule_key[2],
            "n_attempts": int(self.n_attempts),
            "n_accepts": int(self.n_accepts),
            "accept_rate": float(self.accept_rate),
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
            since 2026-05-03.  Pairs naturally with the next-iteration
            hierarchical-bandit idea: per-class arms are the leaf nodes
            a hierarchical posterior would share strength across.  Must
            match :func:`_proposal_rule_key`'s ``per_class_structural``
            flag for ledger priming to recover the same arms.

    Raises:
        ValueError: If either prior is non-positive.
    """

    def __init__(
        self,
        catalog: MutationCatalog,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        per_class_structural: bool = False,
    ) -> None:
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError(f"prior_alpha and prior_beta must be > 0, got {prior_alpha!r}, {prior_beta!r}")
        self.catalog = catalog
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.per_class_structural = bool(per_class_structural)
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

        # Thompson: one Beta draw per arm, pick the arg-max.
        n = len(arms)
        sampled = np.empty(n, dtype=np.float64)
        for i, (_, key, _) in enumerate(arms):
            stats = self._stats.setdefault(key, MutationRuleStats(rule_key=key))
            alpha = self.prior_alpha + stats.n_accepts
            beta_param = self.prior_beta + (stats.n_attempts - stats.n_accepts)
            sampled[i] = float(rng.beta(alpha, beta_param))
        chosen_idx = int(np.argmax(sampled))
        rule, chosen_key, hits = arms[chosen_idx]
        chosen_stats = self._stats[chosen_key]
        alpha_eff = self.prior_alpha + chosen_stats.n_accepts
        beta_eff = self.prior_beta + (chosen_stats.n_attempts - chosen_stats.n_accepts)
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

    def record_outcome(self, accepted: bool) -> None:
        """Update the bandit with the most recent iteration's verdict.

        No-op when :attr:`last_rule_key` is ``None`` — i.e. when the
        previous iteration was a skip or :meth:`sample` was never called
        — so the driver can call this unconditionally on every
        iteration.
        """
        if self._last_rule_key is None:
            return
        stats = self._stats.setdefault(
            self._last_rule_key,
            MutationRuleStats(rule_key=self._last_rule_key),
        )
        stats.n_attempts += 1
        if accepted:
            stats.n_accepts += 1
        self._last_rule_key = None

    def prime_from_ledger(self, ledger_path: str) -> int:
        """Seed the bandit's history from a prior JSONL ledger.

        Replays every iteration record with a non-null proposal: each
        contributes ``n_attempts += 1`` and, if accepted, also ``n_accepts
        += 1``.  Skip records and guard records are ignored.  Returns the
        number of records consumed.

        Useful for resuming a long unattended loop run without losing
        the meta-knowledge of which mutation rules tend to succeed.
        """
        consumed = 0
        for rec in load_ledger(ledger_path):
            if rec.get("record_type", "iteration") != "iteration":
                continue
            proposal = rec.get("proposal")
            if proposal is None:
                continue
            key = _proposal_rule_key(
                proposal.get("class_name", ""),
                proposal.get("param_name", ""),
                proposal.get("rule_kind", ""),
                per_class_structural=self.per_class_structural,
            )
            stats = self._stats.setdefault(key, MutationRuleStats(rule_key=key))
            stats.n_attempts += 1
            if rec.get("accepted"):
                stats.n_accepts += 1
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
    * Categorical toggles — ``PSO.topology``
      (``gbest`` ↔ ``lbest`` ↔ ``vonneumann``), ``Sobol.scramble``
      (``True`` ↔ ``False``), ``LSHADE.archive_factor``
      (``0.0`` / ``1.0`` / ``2.6``), ``LSHADE.F_schedule``
      (``True`` ↔ ``False``), ``NLSHADE_RSP.adaptive_archive``
      (``True`` ↔ ``False``), ``NLSHADE_RSP.k_rank``
      (``0.0`` / ``3.0`` / ``5.0`` — RSP-off / default / aggressive
      regimes, alongside the continuous ``float_uniform`` rule),
      and ``COBYQA.scale`` (``True`` ↔ ``False``).  These use the
      ``categorical_choice`` mutation kind so the loop can flip
      discrete design knobs the same way it tunes numeric ones.

    Bounds are chosen so a single accept keeps the value in a sensible
    range (never zero, never pathologically large).
    """
    return MutationCatalog(
        [
            MutationRule(
                strategy_pattern="",
                class_name="Nearby",
                param_name="radius",
                kind="log_uniform_perturb",
                bounds=(0.005, 0.5),
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
            # the classic Sobol' sequence verbatim.  Different problems
            # respond differently — scramble helps when the true
            # optimum is *not* axis-aligned with the box, hurts when
            # it is.  ``BayesOpt_Sobol`` sets ``scramble=True``
            # explicitly so this rule fires out-of-the-box on the
            # standard battery.
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
            # L-SHADE asymmetric F-cap toggle (categorical).  When ``True``
            # the heuristic clamps drawn ``F`` to 0.7 in the first 60% of
            # the budget, 0.8 in the next 30%, and leaves it unclamped in
            # the final 10% — the jSO (Brest et al. 2017) refinement of
            # the Cauchy F-sampler.  ``False`` reproduces the byte-identical
            # Tanabe-Fukunaga 2014 L-SHADE.  Only fires when a spec sets
            # ``F_schedule`` explicitly (the default kwarg is ``None``).
            # Gives the loop a discrete way to opt L-SHADE into the
            # literature-best mutation magnitude schedule without dropping
            # and re-adding the heuristic.
            MutationRule(
                strategy_pattern="",
                class_name="LSHADE",
                param_name="F_schedule",
                kind="categorical_choice",
                choices=(True, False),
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
            # NL-SHADE-LBC F-memory initial Lehmer exponent.  Literature
            # default ``3.5``; bracket ``[1.5, 5.0]`` so the loop can probe
            # weaker (closer to the L-SHADE-style bias) or stronger
            # (heavily weighting the largest successful F's at the start
            # of the search) initial bias.  ``p_F_init`` only takes effect
            # via the schedule when the strategy budget is known.
            MutationRule(
                strategy_pattern="",
                class_name="NLSHADE_LBC",
                param_name="p_F_init",
                kind="float_uniform",
                bounds=(1.5, 5.0),
                low=1.5,
                high=5.0,
                probability=0.5,
            ),
            # NL-SHADE-LBC F-memory final Lehmer exponent.  Literature
            # default ``1.5``; bracket ``[1.0, 3.0]`` so the loop can probe
            # values bracketing the standard L-SHADE order-2 exponent.
            MutationRule(
                strategy_pattern="",
                class_name="NLSHADE_LBC",
                param_name="p_F_final",
                kind="float_uniform",
                bounds=(1.0, 3.0),
                low=1.0,
                high=3.0,
                probability=0.5,
            ),
            # NL-SHADE-LBC CR-memory schedule (initial / final).  CR
            # literature defaults are ``1.0 → 1.5``; bracket ``[0.5, 2.5]``
            # so the loop can probe pure-arithmetic-mean-like behaviour
            # (low exponent) as well as more biased regimes.
            MutationRule(
                strategy_pattern="",
                class_name="NLSHADE_LBC",
                param_name="p_CR_init",
                kind="float_uniform",
                bounds=(0.5, 2.5),
                low=0.5,
                high=2.5,
                probability=0.5,
            ),
            MutationRule(
                strategy_pattern="",
                class_name="NLSHADE_LBC",
                param_name="p_CR_final",
                kind="float_uniform",
                bounds=(0.5, 2.5),
                low=0.5,
                high=2.5,
                probability=0.5,
            ),
            # NL-SHADE-LBC Lehmer spread.  Default ``1.5`` (CEC-2022);
            # ``1.0`` recovers the standard L-SHADE Lehmer-mean spread.
            # Bracket ``[1.0, 2.0]`` so the loop can flip between them.
            MutationRule(
                strategy_pattern="",
                class_name="NLSHADE_LBC",
                param_name="m_lbc",
                kind="float_uniform",
                bounds=(1.0, 2.0),
                low=1.0,
                high=2.0,
                probability=0.5,
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
        (LSHADE, {"NP_init": 30}),  # adaptive DE w/ linear pop reduction
        (JSO, {"NP_init": 30}),  # CEC-2017 winner, weighted current-to-pbest-w/1
        (NLSHADE_RSP, {"NP_init": 30, "k_rank": 3.0}),  # CEC-2021 winner, NLPSR + RSP
        (NLSHADE_LBC, {"NP_init": 30, "k_rank": 3.0}),  # CEC-2022 winner, NLPSR + RSP + LBC
        (LSHADE_EpSin, {"NP_init": 30, "mu_freq_init": 0.5}),  # ensemble-sinusoid F
        (COBYQA, {}),  # Powell-family derivative-free trust-region local optimizer
        (LBFGSB, {}),  # multi-start gradient-based (quasi-Newton) local optimizer
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
    structural_per_class_arms: bool = False
    holdout_base_seed: int = 0
    holdout_base_seeds: Tuple[int, ...] = ()
    holdout_iterations: int = 5
    holdout_iteration_offset: int = 0
    holdout_eps_overfit: float = 0.05
    paired: Optional[bool] = None
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

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {self.iterations}")
        if self.mode not in {"quick", "standard", "full"}:
            raise ValueError(f"Unknown mode {self.mode!r}")
        if self.metric not in {"composite", "aocc"}:
            raise ValueError(f"metric must be 'composite' or 'aocc', got {self.metric!r}")
        if self.guard_interval < 0:
            raise ValueError(f"guard_interval must be >= 0, got {self.guard_interval}")
        if self.guard_eps_ladder < 0:
            raise ValueError(f"guard_eps_ladder must be >= 0, got {self.guard_eps_ladder}")
        if self.adaptive_prior_alpha <= 0:
            raise ValueError(f"adaptive_prior_alpha must be > 0, got {self.adaptive_prior_alpha}")
        if self.adaptive_prior_beta <= 0:
            raise ValueError(f"adaptive_prior_beta must be > 0, got {self.adaptive_prior_beta}")
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
    """

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
        )

    # Pool per-iteration drifts.  When a record carries paired iter
    # scores we use them (the "high-resolution" contribution); otherwise
    # fall back to the cached point drift (legacy record).
    samples: List[float] = []
    for rec in records:
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

    worst_rec = min(records, key=lambda r: float(r.drift))
    any_overfit = any(bool(r.overfit) for r in records)
    overfit_count = sum(1 for r in records if bool(r.overfit))
    eps = float(eps_overfit) if eps_overfit is not None else float(records[0].eps_overfit)
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
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


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
            )
            if self.config.adaptive_prime_from_ledger:
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

        for iteration in range(self.config.iterations):
            if self._stop_requested():
                if verbose:
                    print(
                        f"[self_improve] STOP sentinel {self.config.stop_sentinel_path!r}"
                        f" present — halting at iter {iteration}"
                    )
                break

            start = time.time()

            proposal = self._sample_proposal(rng, current)
            if proposal is None:
                rec = self._skip_record(iteration, start, "no applicable mutations for current specs")
                records.append(rec)
                ledger.write(rec)
                if verbose:
                    self._print_iteration(rec)
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

            decision = statistical_accept(
                baseline_result,
                candidate_result,
                eps_accept=self.config.eps_accept,
                eps_regress=self.config.eps_regress,
                n_boot=self.config.n_boot,
                confidence=self.config.confidence,
                seed=self.config.stat_seed + iteration,
                paired=self.config.paired,
            )

            rec = LoopIterationRecord(
                iteration=iteration,
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                duration_seconds=time.time() - start,
                proposal=proposal.to_dict(),
                accepted=bool(decision.accept),
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
                reasons=list(decision.reasons),
                base_seed=self.config.base_seed,
                randomize_iteration=iteration,
                mode=self.config.mode,
            )
            records.append(rec)
            ledger.write(rec)
            if verbose:
                self._print_iteration(rec)

            # Refresh the seed entry's validated score the first time we
            # measure with it so the guard has a baseline to compare
            # against (the seed itself never gets accepted, but it can
            # still be the rollback target).
            if np.isnan(ladder[0].last_validated_score):
                ladder[0].last_validated_score = float(baseline_result.composite_score)

            # Update the adaptive bandit *before* swapping the ladder so
            # the rule key recorded by `_sample_proposal` still matches
            # this iteration's outcome.  Uniform-sampler runs do nothing.
            if self.sampler is not None:
                self.sampler.record_outcome(bool(decision.accept))

            if decision.accept:
                current = candidate
                ladder.append(
                    LadderEntry(
                        iteration=iteration,
                        specs=list(candidate),
                        last_validated_score=float(candidate_result.composite_score),
                        proposal=proposal.to_dict(),
                    )
                )

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
            cfg = HarnessConfig(mode=self.config.mode, strategies=self.config.strategy_names)
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
    ) -> HarnessResult:
        """AOCC-metric variant of :meth:`_measure`.

        Runs the IOH harness on a mode-mapped battery and adapts the
        result so the rest of the loop (statistical_accept, ledger
        writer, guard, hold-out) sees a :class:`HarnessResult` whose
        ``composite_score`` is mean AOCC and whose per-pair ``score``
        values are per-instance AOCC.  The bootstrap CI on the
        composite delta then operates directly on AOCC values.
        """
        from panobbgo.harness_ioh import (
            aocc_to_harness_result,
            make_full_battery,
            make_quick_battery,
            make_standard_battery,
            run_ioh_harness,
        )

        battery_factories = {
            "quick": make_quick_battery,
            "standard": make_standard_battery,
            "full": make_full_battery,
        }
        battery = battery_factories[self.config.mode]()
        if verbose:
            print(f"[self_improve] iter={iteration} measuring {label} (AOCC, battery={battery.name})")
        # Mix the iteration into the base seed so each iteration draws
        # fresh-but-reproducible instance RNG seeds, matching the
        # randomized composite-score path.
        base_seed = self.config.base_seed + iteration if self.config.randomize else self.config.base_seed
        ioh_result = run_ioh_harness(specs, battery, base_seed=base_seed, progress=False)
        return aocc_to_harness_result(ioh_result, mode=self.config.mode, base_seed=self.config.base_seed)

    def _skip_record(self, iteration: int, start: float, reason: str) -> LoopIterationRecord:
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
        """Single hold-out measurement on an independent ``base_seed``."""
        hc = self.config.holdout_harness_config(specs, iteration_id, base_seed=base_seed)
        if verbose:
            print(f"[self_improve] hold-out measuring {label} at iter_id={iteration_id} base_seed={base_seed}")
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
        overfit = drift < -float(self.config.holdout_eps_overfit)

        reasons: List[str] = []
        if seed_only:
            reasons.append(
                "ladder has only the seed entry — no accepted mutations to validate; "
                "hold-out scores recorded for reference but drift is 0 by construction"
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
        )

    @staticmethod
    def _print_holdout(rec: "LoopHoldoutRecord") -> None:
        verdict = "OVERFIT" if rec.overfit else "OK"
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

    Writes :class:`LoopIterationRecord`, :class:`LoopGuardRecord`, and
    :class:`LoopHoldoutRecord` instances; the ``record_type`` field
    distinguishes them on read.
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
]
