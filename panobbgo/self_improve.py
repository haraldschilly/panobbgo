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
        bounds: Inclusive ``(low, high)`` clamp applied after sampling.
            Always interpreted as floats for ``log_uniform_perturb`` and
            ``float_uniform``; as ints for ``integer_add``.
        log_step: Half-width of the log-uniform perturbation (decades).
            Default ``0.15`` ≈ ±41 %.
        delta_choices: Integer deltas for ``integer_add``.
        low: Lower bound for ``float_uniform``.
        high: Upper bound for ``float_uniform``.
        probability: Relative weight used when the catalog picks among
            multiple applicable rules; normalised automatically.
    """

    strategy_pattern: str
    class_name: str
    param_name: str
    kind: str
    bounds: Tuple[float, float]
    log_step: float = 0.15
    delta_choices: Tuple[int, ...] = (-1, 1)
    low: float = 0.0
    high: float = 1.0
    probability: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"log_uniform_perturb", "integer_add", "float_uniform"}:
            raise ValueError(f"Unknown mutation kind: {self.kind!r}")
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
_STRUCTURAL_OPS: Tuple[str, ...] = ("add_heuristic", "drop_heuristic")


@dataclass
class StructuralMutationRule:
    """Add or drop a heuristic from a strategy's portfolio.

    Implements the *Strategy portfolio composition* mutation class from
    §7.2 of ``planning/SELF_IMPROVEMENT_LOOP.md``.  Where
    :class:`MutationRule` only retunes existing kwargs, this rule changes
    the *shape* of a :class:`StrategySpec`'s ``heuristics`` list — the
    loop can therefore discover whether dropping a heuristic, or adding a
    fresh one, generalises better than the seed composition.

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
        candidate_classes: Sequence of ``(HeuristicClass, default_kwargs)``
            pairs the rule may pull from for ``add_heuristic``.  Ignored
            for ``drop_heuristic``.  Each tuple's ``default_kwargs`` is
            shallow-copied into the new spec so subsequent kwarg-tune
            mutations can perturb it independently.
        droppable_classes: Optional restriction for ``drop_heuristic``.
            When provided (a tuple of class ``__name__``s), only matching
            heuristics may be dropped.  Empty tuple means "any heuristic
            in the strategy is eligible (subject to :attr:`min_heuristics`)".
        min_heuristics: Lower bound on the size of the heuristics list
            after a drop.  Default ``2`` keeps every strategy with at
            least one diversity slot beyond the bare minimum.  ``1`` is
            the absolute floor; ``0`` is rejected.
        avoid_duplicates: For ``add_heuristic``, skip candidates whose
            class is already present in the strategy.  Default ``True``.
            Set to ``False`` when intentional duplicates are desirable
            (e.g. two :class:`Nearby` instances at different radii).
        probability: Relative weight when the catalog picks among
            multiple applicable rules; normalised automatically.  Same
            semantics as :class:`MutationRule`.

    Raises:
        ValueError: If ``op`` is not one of :data:`_STRUCTURAL_OPS`,
            ``min_heuristics`` is below ``1``, ``probability`` is
            non-positive, or ``op == "add_heuristic"`` is paired with an
            empty :attr:`candidate_classes`.
    """

    strategy_pattern: str
    op: str
    candidate_classes: Tuple[Tuple[type, Dict[str, Any]], ...] = ()
    droppable_classes: Tuple[str, ...] = ()
    min_heuristics: int = 2
    avoid_duplicates: bool = True
    probability: float = 1.0

    def __post_init__(self) -> None:
        if self.op not in _STRUCTURAL_OPS:
            raise ValueError(f"Unknown structural op: {self.op!r}; expected one of {_STRUCTURAL_OPS}")
        if self.min_heuristics < 1:
            raise ValueError(f"min_heuristics must be >= 1, got {self.min_heuristics}")
        if self.probability <= 0:
            raise ValueError(f"probability must be > 0, got {self.probability}")
        if self.op == "add_heuristic" and not self.candidate_classes:
            raise ValueError("add_heuristic requires at least one entry in candidate_classes")

    def rule_key(self) -> "RuleKey":
        """Return the bandit-arm key for this rule.

        All structural rules with the same ``op`` share one arm by
        default — this keeps the bandit space small and matches the
        coarsest reasonable taxonomy ("does adding heuristics help?",
        "does dropping help?").  Per-class arms are a future refinement
        (see §10 ledger note in the plan).
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

    * ``add_heuristic`` — Cartesian product of matching strategies and
      :attr:`candidate_classes`, optionally pruned by
      :attr:`avoid_duplicates`.  Each candidate's ``default_kwargs`` is
      shallow-copied so two hits never share a mutable dict.
    * ``drop_heuristic`` — every existing ``(spec, heuristic)`` pair
      whose strategy matches :attr:`strategy_pattern`, the heuristic's
      class is in :attr:`droppable_classes` (or any, if empty), and the
      strategy currently has more than :attr:`min_heuristics` heuristics
      so removal would not violate the safety floor.

    Returning an empty list signals "rule not applicable to current
    specs"; the catalog uses that to skip the rule entirely (and so the
    bandit does not waste an iteration on a no-op).
    """
    hits: List[_StructuralHit] = []
    if rule.op == "add_heuristic":
        for si, spec in enumerate(specs):
            if rule.strategy_pattern and rule.strategy_pattern not in spec.name:
                continue
            present = {cls.__name__ for cls, _ in spec.heuristics}
            for cls, default_kwargs in rule.candidate_classes:
                if rule.avoid_duplicates and cls.__name__ in present:
                    continue
                hits.append((si, cls, dict(default_kwargs)))
    elif rule.op == "drop_heuristic":
        droppable = set(rule.droppable_classes)
        for si, spec in enumerate(specs):
            if rule.strategy_pattern and rule.strategy_pattern not in spec.name:
                continue
            if len(spec.heuristics) <= rule.min_heuristics:
                # Dropping any one would breach the safety floor.
                continue
            for cls, kwargs in spec.heuristics:
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
    continue to work without special-casing.
    """
    si, cls, kwargs = hit
    strategy_name = specs[si].name
    if rule.op == "add_heuristic":
        rationale = f"add_heuristic {cls.__name__}({kwargs!r}) to {strategy_name}"
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
    # drop_heuristic
    rationale = f"drop_heuristic {cls.__name__}({kwargs!r}) from {strategy_name}"
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


def _proposal_rule_key(class_name: str, param_name: str, rule_kind: str) -> RuleKey:
    """Map a proposal triple to the bandit's arm key.

    Structural ops (``add_heuristic``, ``drop_heuristic``) collapse
    onto a single arm per op type — see
    :meth:`StructuralMutationRule.rule_key`.  Kwarg perturbations keep
    the natural one-arm-per-(class, param, kind) granularity.  The two
    paths must agree because :meth:`AdaptiveMutationSampler.prime_from_ledger`
    rebuilds the bandit history from JSONL records that only carry the
    proposal triple — never the original rule object.
    """
    if rule_kind in _STRUCTURAL_OPS:
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

    Raises:
        ValueError: If either prior is non-positive.
    """

    def __init__(
        self,
        catalog: MutationCatalog,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError(f"prior_alpha and prior_beta must be > 0, got {prior_alpha!r}, {prior_beta!r}")
        self.catalog = catalog
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self._stats: Dict[RuleKey, MutationRuleStats] = {}
        self._last_rule_key: Optional[RuleKey] = None

    @staticmethod
    def _rule_key(rule: CatalogRule) -> RuleKey:
        # Both :class:`MutationRule` and :class:`StructuralMutationRule`
        # expose ``rule_key()``; delegating keeps the sampler agnostic to
        # which kind of rule the catalog holds.
        return rule.rule_key()

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
        """
        applicable = self.catalog.applicable_rules(specs)
        if not applicable:
            self._last_rule_key = None
            return None

        # Thompson: one Beta draw per applicable rule, pick the arg-max.
        n = len(applicable)
        sampled = np.empty(n, dtype=np.float64)
        for i, (rule, _) in enumerate(applicable):
            stats = self.get_stats(rule)
            alpha = self.prior_alpha + stats.n_accepts
            beta_param = self.prior_beta + (stats.n_attempts - stats.n_accepts)
            sampled[i] = float(rng.beta(alpha, beta_param))
        chosen_idx = int(np.argmax(sampled))
        rule, hits = applicable[chosen_idx]
        chosen_stats = self.get_stats(rule)
        alpha_eff = self.prior_alpha + chosen_stats.n_accepts
        beta_eff = self.prior_beta + (chosen_stats.n_attempts - chosen_stats.n_accepts)
        thompson_tag = (
            f"[Thompson Beta({alpha_eff:.1f}, {beta_eff:.1f}); "
            f"draw={sampled[chosen_idx]:.3f}; "
            f"history {chosen_stats.n_accepts}/{chosen_stats.n_attempts}]"
        )
        self._last_rule_key = self._rule_key(rule)

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
            # L-SHADE memory length.  Tanabe-Fukunaga (2014) recommend
            # ``H = 6``; the loop may explore a tighter window
            # ``[4, 12]`` where the algorithm is known to remain
            # competitive.  Only fires when a spec explicitly sets ``H``
            # (it is otherwise a default kwarg on :class:`LSHADE`).
            MutationRule(
                strategy_pattern="",
                class_name="LSHADE",
                param_name="H",
                kind="integer_add",
                bounds=(4, 12),
                delta_choices=(-2, -1, 1, 2),
                probability=0.5,
            ),
        ]
    )


def default_structural_catalog() -> MutationCatalog:
    """Return :func:`default_catalog` extended with portfolio-shape rules.

    Implements the *Strategy portfolio composition* mutation class from
    §7.2 of ``planning/SELF_IMPROVEMENT_LOOP.md``.  Two structural rules
    sit alongside the existing kwarg perturbations:

    * ``add_heuristic`` from a curated pool of unconditionally-safe
      generators (``Random``, ``Nearby``, ``NelderMead``, ``Center``,
      ``LatinHypercube``, ``Sobol``, ``Extremal``).  ``avoid_duplicates=True``
      so the catalog never proposes a duplicate of a class already in the
      strategy.
    * ``drop_heuristic`` with ``min_heuristics=2`` so no strategy is
      ever stripped down past the diversity floor.

    Both rules carry a low probability (``0.3``) relative to the kwarg
    rules — structural changes are higher-variance than retunes, so the
    loop should sample them sparingly.  The overall acceptance rate stays
    in the same neighbourhood as :func:`default_catalog`'s while
    expanding the search space the loop can explore.

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

    base_rules = list(default_catalog().rules)
    # PSO and LSHADE are loaded lazily because they use a slightly
    # heavier set of numpy / RNG primitives than the simpler heuristics
    # above, and ``default_structural_catalog`` may be called from
    # environments (e.g. minimal CI) that import :mod:`panobbgo.self_improve`
    # without the full heuristics package.  The local imports keep the
    # cost of the catalog factory unchanged when they are not selected.
    from panobbgo.heuristics.pso import PSO
    from panobbgo.heuristics.lshade import LSHADE

    # Two PSO entries cover the canonical ``gbest`` (default Kennedy-Eberhart
    # 1995 swarm) and the ``lbest`` ring topology (Kennedy & Mendes 2002).
    # ``avoid_duplicates=True`` ensures only one PSO variant ends up in any
    # given strategy — the catalog picks gbest or lbest uniformly when PSO
    # is not yet present, after which subsequent samples skip both.
    #
    # LSHADE (Tanabe-Fukunaga 2014) — the CEC-2014 winner — sits next to
    # PSO as a strong population-based candidate.  Its ``NP_init=None``
    # default lets the heuristic auto-size from the problem dimension at
    # on_start time (``18·dim`` capped at ``NP_init_cap``).
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
        (LSHADE, {}),  # Tanabe-Fukunaga 2014 — CEC-2014 winner
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

    Three proposal flavours are supported:

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

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {self.iterations}")
        if self.mode not in {"quick", "standard", "full"}:
            raise ValueError(f"Unknown mode {self.mode!r}")
        if self.guard_interval < 0:
            raise ValueError(f"guard_interval must be >= 0, got {self.guard_interval}")
        if self.guard_eps_ladder < 0:
            raise ValueError(f"guard_eps_ladder must be >= 0, got {self.guard_eps_ladder}")
        if self.adaptive_prior_alpha <= 0:
            raise ValueError(f"adaptive_prior_alpha must be > 0, got {self.adaptive_prior_alpha}")
        if self.adaptive_prior_beta <= 0:
            raise ValueError(f"adaptive_prior_beta must be > 0, got {self.adaptive_prior_beta}")

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

        Guard records (:class:`LoopGuardRecord`) are written to the
        ledger alongside iteration records but are not returned here —
        the contract of :meth:`run` is unchanged for backward
        compatibility.  Use :meth:`run_with_guard_records` when both
        are wanted in-process, or read the ledger to recover them.
        """
        records, _ = self._run_internal(verbose=verbose)
        return records

    def run_with_guard_records(self, verbose: bool = False) -> Tuple[List[LoopIterationRecord], List[LoopGuardRecord]]:
        """Run the loop and return ``(iteration_records, guard_records)``.

        The two lists are returned separately so existing callers of
        :meth:`run` keep their type contract.  Both lists are also
        persisted to the ledger.
        """
        return self._run_internal(verbose=verbose)

    def _run_internal(self, verbose: bool = False) -> Tuple[List[LoopIterationRecord], List[LoopGuardRecord]]:
        current = self._load_seed_strategies()
        rng = np.random.default_rng(self.config.mutation_seed)
        records: List[LoopIterationRecord] = []
        guard_records: List[LoopGuardRecord] = []
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

        return records, guard_records

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
        cfg = HarnessConfig(mode=self.config.mode, strategies=self.config.strategy_names)
        return self._harness_factory(cfg).get_strategies()

    def _measure(
        self,
        specs: List[StrategySpec],
        iteration: int,
        label: str,
        verbose: bool,
    ) -> HarnessResult:
        hc = self.config.harness_config(specs, iteration)
        if verbose:
            print(f"[self_improve] iter={iteration} measuring {label}")
        return self._harness_factory(hc).run(verbose=False)

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

    Writes both :class:`LoopIterationRecord` and :class:`LoopGuardRecord`
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
    "LadderEntry",
    "SelfImprover",
    "load_ledger",
]
