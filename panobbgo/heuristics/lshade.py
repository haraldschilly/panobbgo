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
L-SHADE Heuristic
=================

Linear Population Size Reduction Success-History Adaptive Differential
Evolution (L-SHADE), Tanabe & Fukunaga (CEC 2014).

The basic Differential Evolution heuristic shipped in
:mod:`panobbgo.heuristics.differential_evolution` uses fixed
``F = 0.8`` / ``CR = 0.9`` and a fixed population size — a simple,
robust baseline.  L-SHADE adds two literature-best tricks that won
CEC-2014 and have been the high-water mark for single-population
black-box solvers ever since:

1. **Success-History parameter Adaptation (SHADE / Zhang-Sanderson 2009)**.
   Instead of fixed scalars, each trial draws its own ``(F_i, CR_i)``
   from per-bin Cauchy / Normal distributions.  After every "generation"
   of ``NP_current`` trials, the bins that produced *successful*
   replacements update their centres via the **weighted Lehmer mean**
   of the F/CR values that worked, weighted by the magnitude of the
   improvement they produced.  The memory bins rotate cyclically so
   recent successes dominate.
2. **Linear Population Size Reduction (LPSR)**.  The population shrinks
   linearly from ``NP_init`` (typically large, 18·d in the paper) down
   to ``NP_min = 4`` over the strategy's evaluation budget.  Larger
   populations explore broadly early; smaller ones exploit the leading
   basin late.

Mutation uses ``current-to-pbest/1`` (Zhang-Sanderson 2009)::

    v_i = x_i + F_i · (x_pbest − x_i) + F_i · (x_r1 − x_r2)

where ``x_pbest`` is drawn from the top ``p_best · |population|`` by
fitness, ``x_r1`` from the population, and ``x_r2`` from the union of
the population and an *external archive* of recently-replaced parents
(at most ``archive_factor · NP_current`` entries).  The crossover is
binomial with rate ``CR_i``.  Out-of-bounds components are repaired by
midpoint reflection (Tanabe-Fukunaga §III-A): ``v[j] = (lb[j] + x_i[j]) / 2``
when ``v[j] < lb[j]``, symmetric for ``ub[j]``.

Optionally pass ``p_best_end`` to enable the iLSHADE / jSO adaptive
``p_best`` schedule (Brest et al. 2016 / 2017).  The effective
greediness at evaluation count ``e`` (out of
``E = strategy.config.max_eval``) becomes
``p_eff(e) = p_best − (p_best − p_best_end) · min(e/E, 1)``,
shrinking the pool of ``pbest`` candidates as the population shrinks
under LPSR.  The canonical jSO setting is
``p_best = 0.25, p_best_end = 0.125``.  ``p_best_end=None`` (the
default) preserves the constant-``p_best`` L-SHADE behaviour.

Optionally pass ``F_schedule`` to enable an asymmetric F-cap schedule.
Three named regimes ship out-of-the-box:

* ``"jso"`` (Brest et al. 2017) — ``F ≤ 0.7`` while ``progress < 0.6``,
  ``F ≤ 0.8`` while ``progress < 0.9``, unclamped in the final 10%.
* ``"early"`` — kicks in earlier with a tighter first cap: ``F ≤ 0.6``
  while ``progress < 0.4``, ``F ≤ 0.8`` while ``progress < 0.7``,
  unclamped after that.  Useful when the basin sits near the box centre
  and small initial steps help converge faster.
* ``"strict"`` — aggressive throughout: ``F ≤ 0.5`` while
  ``progress < 0.5``, ``F ≤ 0.7`` while ``progress < 0.85``, unclamped
  in the final 15%.  Useful when large F is harmful (ill-conditioned
  basins, narrow valleys).

``F_schedule="off"`` (explicit) and ``F_schedule=None`` (default) both
keep the unclamped Tanabe-Fukunaga behaviour.  For backwards
compatibility ``True`` is accepted as a synonym for ``"jso"`` and
``False`` as a synonym for ``"off"``; jSO opts into ``"jso"`` by
construction.

Asynchronous execution
----------------------

Like Panobbgo's other population heuristics, L-SHADE here runs
asynchronously inside the event loop:

1. ``on_start()`` emits ``NP_init`` random initial positions.
2. ``on_new_results()`` matches incoming results back to their slot via
   the ``who`` tag, fills the slot on the first arrival, and on later
   arrivals competes the trial against the slot's incumbent — the loser
   is discarded, the winner stays in the population, and the loser (if
   it was the parent) is pushed onto the archive.  After every
   ``NP_current`` evolutionary trials complete, the heuristic updates
   the memory bins with the successful F/CR triples and applies LPSR.
3. ``on_restart(center, reason)`` drops in-flight trials, clears the
   archive, resets the memory bins, and reseeds the population around
   the suggested center — matching the warm-restart behaviour of
   :class:`~panobbgo.heuristics.pso.PSO` and
   :class:`~panobbgo.heuristics.cma_es.CMAES`.

Notes on the async / sync gap
------------------------------

Synchronous L-SHADE applies parameter adaptation only at the *end* of
each generation, after every individual has been re-evaluated.  In the
async port we batch by *count*: every ``NP_current`` completed
evolutionary trials forms one "generation".  This keeps the same total
update cadence; the only difference is that within one async generation
a slot may have been touched twice (or zero times) rather than exactly
once.  In practice the SHADE memory adaptation is robust to this
because the weighted Lehmer mean is invariant under the order of its
contributing samples.

Constraint handling delegates to ``strategy.constraint_handler`` exactly
like :class:`~panobbgo.heuristics.differential_evolution.DifferentialEvolution`:
``is_better`` for trial-vs-target competition, ``get_penalty_value`` for
the scalar fitness used to rank pbest candidates and to weight memory
updates by improvement magnitude.

References
----------

* J. Zhang & A. Sanderson (2009). "JADE: Adaptive Differential Evolution
  with Optional External Archive." *IEEE Transactions on Evolutionary
  Computation*, 13(5):945-958.
* R. Tanabe & A. Fukunaga (2013). "Success-History Based Parameter
  Adaptation for Differential Evolution." *Proceedings of CEC 2013*.
* R. Tanabe & A. Fukunaga (2014). "Improving the Search Performance of
  SHADE Using Linear Population Size Reduction." *Proceedings of
  CEC 2014*.  Winner of the CEC-2014 single-objective competition.
* J. Brest, M. S. Maučec & B. Bošković (2016). "iL-SHADE: Improved
  L-SHADE Algorithm for Single Objective Real-Parameter Optimization."
  *Proceedings of CEC 2016*.  Introduces the linearly-decreasing
  ``p_best`` schedule.
* J. Brest, M. S. Maučec & B. Bošković (2017). "Single Objective
  Real-Parameter Optimization: Algorithm jSO." *Proceedings of CEC
  2017*.  Winner of the CEC-2017 single-objective competition.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from panobbgo.core import Heuristic
from panobbgo.lib import Point, Result


# Default tuning constants — match the values from Tanabe & Fukunaga
# (2014, Algorithm 1).  ``_DEFAULT_NP_INIT`` is no longer the constructor
# default (that is ``"auto"`` since the §17 measurement); it is the fallback
# :func:`_resolve_auto_np_init` returns when the evaluation budget is unknown
# and no dimensional rule can be evaluated.
_DEFAULT_NP_INIT: int = 30
_DEFAULT_NP_MIN: int = 4

# Budget-adaptive ("auto") NP_init sizing.  A single fixed ``NP_init`` is
# mistuned across budgets and dimensions: too large a population at a tight
# evaluation budget spends most of the budget on the initial random fill and
# never runs enough generations for the SHADE success-history adaptation to
# pay off, while too small a population collapses ``current-to-pbest/1`` into
# a near-degenerate local search.  ``NP_init="auto"`` sizes the population
# from the strategy budget and the problem dimension::
#
#     NP = clip( round( _AUTO_DIM_COEF · dim
#                       · (budget / (_AUTO_REF_BUDGET_PER_DIM · dim)) ** _AUTO_BUDGET_EXP ),
#                max(NP_min, _AUTO_MIN_NP), _AUTO_MAX_NP )
#
# Both the coefficient and the shape are measured, not inherited from the
# literature (``planning/DISCOVERY_2026-09-09.md`` §17).  Fixed-``NP_init``
# grid sweeps of the three L-SHADE-lineage arms (L-SHADE, jSO, NLSHADE_LBC)
# on the MA-BBOB standard battery (budget ``500·dim``, three seeds) put the
# AOCC-optimal size at 6-8 for ``d=2``, 10-20 for ``d=5`` and 20-30 for
# ``d=10``: the optimum tracks ``dim`` — the ``d=5 : d=2`` ratio is the
# dimension ratio — which fixes ``_AUTO_DIM_COEF ≈ 3``.  The same sweep at
# four times the budget (``2000·dim``) moves the optimum only from 6-8 to 10
# at ``d=2`` and from 15 to 15-20 at ``d=5``, i.e. roughly the fourth root of
# the budget ratio, hence ``_AUTO_BUDGET_EXP = 0.25`` anchored at the
# reference ``_AUTO_REF_BUDGET_PER_DIM = 500`` evaluations per dimension.
# Getting this right is worth +0.10 to +0.22 AOCC on those arms — by far the
# largest single effect measured on this codebase.  (The previous rule used
# the CEC-2014 upper bound ``18·d`` capped by ``budget/12``; on the IOH
# batteries the cap never binds, so it shipped populations ~5× too large.)
# ``_AUTO_MIN_NP`` floors the size at 6 so ``current-to-pbest/1`` (which needs
# ≥ 4 distinct individuals) keeps working headroom — ``NP_init=4`` collapses
# on the same battery (−0.25 AOCC for NLSHADE_LBC).
_AUTO_DIM_COEF: float = 3.0
_AUTO_REF_BUDGET_PER_DIM: float = 500.0
_AUTO_BUDGET_EXP: float = 0.25
_AUTO_MIN_NP: int = 6
_AUTO_MAX_NP: int = 400
_DEFAULT_H: int = 6
_DEFAULT_P_BEST: float = 0.11
_DEFAULT_ARCHIVE_FACTOR: float = 1.0
# Cauchy/Normal scale used by SHADE for F/CR sampling — fixed to 0.1 in
# all published variants of the algorithm.
_PARAM_SCALE: float = 0.1
# Maximum number of Cauchy redraws when sampling F.  A failed F sample
# is one with F <= 0; the redraw distribution is heavy-tailed so 100
# attempts give an effectively zero failure probability.
_F_MAX_REDRAWS: int = 100
# Sentinel emitted into M_CR when an entire generation produced only
# CR = 0 successes; subsequent draws then deterministically return 0
# (per Tanabe-Fukunaga §III-B).
_CR_TERMINAL: float = -1.0

# Asymmetric F-cap schedule.  Each regime is a 4-tuple
# ``(phase1_bound, phase2_bound, phase1_cap, phase2_cap)`` keyed on
# ``progress = len(results) / max_eval``.  When ``F_schedule`` resolves
# to a named regime, sampled ``F`` is clamped via::
#
#   progress < phase1_bound                    → F ≤ phase1_cap
#   phase1_bound ≤ progress < phase2_bound     → F ≤ phase2_cap
#   progress ≥ phase2_bound                    → F unclamped (just F ≤ 1)
#
# The ``"jso"`` regime is the canonical Brest, Maučec & Bošković (2017,
# §III-D) settings; ``"early"`` and ``"strict"`` extend the literature
# regime with tighter or earlier-kicking caps so the bandit can search
# the broader cap geometry without changing the heuristic at all.
# ``"off"`` / ``None`` / ``False`` all bypass the cap entirely
# (byte-identical Tanabe-Fukunaga behaviour shipped 2026-05-10).
_F_SCHEDULE_REGIMES: Dict[str, Tuple[float, float, float, float]] = {
    "jso": (0.6, 0.9, 0.7, 0.8),
    "early": (0.4, 0.7, 0.6, 0.8),
    "strict": (0.5, 0.85, 0.5, 0.7),
}
# Backwards-compatibility aliases for the original boolean toggle
# (shipped 2026-05-21 as ``F_schedule: bool``).  The literature canonical
# breakpoints / caps live on the ``"jso"`` regime above; these constants
# stay so module-level introspection code (and existing tests pinning
# the canonical Brest 2017 values) keeps working.
_F_SCHEDULE_PHASE1_BOUND: float = _F_SCHEDULE_REGIMES["jso"][0]
_F_SCHEDULE_PHASE2_BOUND: float = _F_SCHEDULE_REGIMES["jso"][1]
_F_SCHEDULE_PHASE1_CAP: float = _F_SCHEDULE_REGIMES["jso"][2]
_F_SCHEDULE_PHASE2_CAP: float = _F_SCHEDULE_REGIMES["jso"][3]


def _resolve_auto_np_init(strategy, NP_min: int, dim_coef: Optional[float] = None) -> int:
    """Resolve ``NP_init="auto"`` to a concrete budget-adaptive population size.

    Uses the owning strategy's evaluation budget (``strategy.config.max_eval``)
    and the problem dimension (``strategy.problem.dim``).  Returns the fixed
    :data:`_DEFAULT_NP_INIT` fallback when the budget is unknown (no
    ``max_eval``, zero, or non-numeric) or the dimension is unavailable, so the
    caller degrades to the literature default rather than guessing a horizon.
    See the :data:`_AUTO_DIM_COEF` / :data:`_AUTO_BUDGET_EXP` comment for the
    sizing formula and the measured motivation.

    ``dim_coef`` overrides the per-dimension coefficient; the constructors pass
    their class's :attr:`LSHADE.AUTO_DIM_COEF`, which a subclass may raise when
    the measured optimum for that variant is a bigger swarm (NL-SHADE-LBC).
    ``None`` uses the module default :data:`_AUTO_DIM_COEF`.
    """
    try:
        budget = float(strategy.config.max_eval)
    except Exception:
        budget = float("nan")
    try:
        dim = int(strategy.problem.dim)
    except Exception:
        dim = 0
    if not np.isfinite(budget) or budget <= 0.0 or dim <= 0:
        return _DEFAULT_NP_INIT
    try:
        np_min_i = int(NP_min)
    except Exception:
        np_min_i = _DEFAULT_NP_MIN
    try:
        coef = _AUTO_DIM_COEF if dim_coef is None else float(dim_coef)
    except Exception:
        coef = _AUTO_DIM_COEF
    if not np.isfinite(coef) or coef <= 0.0:
        coef = _AUTO_DIM_COEF
    ref = _AUTO_REF_BUDGET_PER_DIM * dim
    raw = coef * dim * (budget / ref) ** _AUTO_BUDGET_EXP
    lo = max(np_min_i, _AUTO_MIN_NP)
    hi = max(lo, _AUTO_MAX_NP)
    return int(np.clip(int(round(raw)), lo, hi))


def _normalize_F_schedule(value: Optional[Union[bool, str]]) -> Optional[str]:
    """Map the constructor's ``F_schedule`` argument onto a regime name.

    Returns ``None`` for the cap-disabled regimes (``None`` / ``False`` /
    ``"off"``) so :meth:`LSHADE._apply_F_cap` can branch on a single
    ``is None`` check.  Returns a key into :data:`_F_SCHEDULE_REGIMES`
    for the active regimes.  Raises :class:`ValueError` for any other
    input — bools other than ``True`` / ``False``, unknown strings, etc.
    """
    if value is None or value is False:
        return None
    if value is True:
        return "jso"
    if isinstance(value, str):
        if value == "off":
            return None
        if value in _F_SCHEDULE_REGIMES:
            return value
        valid = ("off",) + tuple(sorted(_F_SCHEDULE_REGIMES))
        raise ValueError(f"LSHADE: F_schedule must be one of {valid} (or None / True / False), got {value!r}")
    raise ValueError(f"LSHADE: F_schedule must be a string regime name, bool, or None, got {value!r}")


class _Dropped:
    """Sentinel used to mark population slots removed by LPSR."""


_DROPPED = _Dropped()


class LSHADE(Heuristic):
    """L-SHADE: linear-population-reduction SHADE adaptive DE.

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        NP_init: Initial population size, or the string ``"auto"`` for
            budget-adaptive sizing.  Default ``"auto"``; the literature
            default of ``30`` is the fallback when the budget is unknown.
            A fixed population is mistuned by construction — the CEC-2014
            paper's ``18 · d`` is far heavier than Panobbgo's typical
            budget can support, and even the milder constant ``30`` costs
            ~0.10 AOCC at ``d=2`` and ~0.05 at ``d=5`` against the
            measured rule (``planning/DISCOVERY_2026-09-09.md`` §17), so
            the default sizes itself from the strategy's evaluation budget
            and the problem dimension via
            ``clip(round(3·dim · (budget / (500·dim))**0.25), max(NP_min, 6), 400)``
            — the measured AOCC optimum on the MA-BBOB battery is
            ``≈ 3·dim`` at the reference budget of ``500·dim``
            evaluations, with a mild (fourth-root) budget dependence
            (``planning/DISCOVERY_2026-09-09.md`` §17): 6 at ``d=2``,
            15 at ``d=5``, 30 at ``d=10``, rising to 21 for ``d=5`` at
            four times the budget.  The coefficient ``3`` is the class
            attribute :attr:`AUTO_DIM_COEF`, so a subclass whose measured
            optimum is a bigger swarm can raise it —
            :class:`~panobbgo.heuristics.nl_shade_lbc.NLSHADE_LBC` uses
            ``4``.  Pass an explicit ``int`` to pin a fixed population.
            See :func:`_resolve_auto_np_init`.
        NP_min: Minimum population size after LPSR shrinking.  Default
            ``4`` — required by ``current-to-pbest/1`` (mutation needs
            at least four distinct individuals).  Must satisfy
            ``NP_min <= NP_init``.
        H: History memory size — number of (M_F, M_CR) bins.  Default
            ``6`` — the value used by both SHADE and L-SHADE.  Larger
            values smooth memory updates but slow adaptation; smaller
            values track recent successes more tightly at the cost of
            more variance.
        p_best: Greediness factor for ``current-to-pbest/1``.  Each
            trial picks its ``pbest`` uniformly from the top
            ``ceil(p_best · |population|)`` by fitness.  Default
            ``0.11`` per Tanabe-Fukunaga §III-A.  Must lie in ``(0, 1]``.
            When ``p_best_end`` is set, this is the *initial* value of
            a linearly-annealed schedule (iLSHADE / jSO).
        p_best_end: Optional terminal greediness for the iLSHADE
            (Brest et al. 2016) / jSO (Brest et al. 2017) adaptive
            schedule.  When set, the effective ``p_best`` at evaluation
            count ``e`` (out of ``E = strategy.config.max_eval``) is
            ``p_eff(e) = p_best − (p_best − p_best_end) · min(e/E, 1)``.
            The canonical jSO setting is
            ``p_best = 0.25, p_best_end = 0.125`` — greediness halves
            as the population shrinks under LPSR so the late-search
            mutation pulls toward a narrower, more tightly-chosen
            ``pbest`` slice.  Must lie in ``(0, 1]`` when set;
            ``None`` (the default) keeps ``p_best`` constant for
            byte-identical L-SHADE behaviour.  Falls back to constant
            ``p_best`` whenever the strategy budget is unknown.
        archive_factor: Multiplier for the external archive size; the
            archive is capped at ``ceil(archive_factor · NP_current)``.
            Default ``1.0``.  Setting it to ``0`` disables the archive
            (``r2`` is then drawn only from the live population).
        F_schedule: Optional asymmetric F-cap regime.  Accepts one of
            the named regimes ``"off"`` / ``"jso"`` / ``"early"`` /
            ``"strict"`` (see the module docstring for the per-regime
            breakpoints and caps), or ``None`` (default — same as
            ``"off"``).  ``True`` / ``False`` are accepted as
            backwards-compatible synonyms for ``"jso"`` / ``"off"``.
            The constructor normalizes the value so
            :attr:`F_schedule` is ``None`` for the disabled regimes and
            a string regime name otherwise.  Falls back to the
            unclamped behaviour when the strategy budget is unknown.
            jSO opts into ``"jso"`` by construction.
        warm_start: Optional seeding of the initial population from the
            *shared* archive instead of uniform random points
            (``planning/DESIGN_warm_start_2026-09-10.md`` §2).  One of
            ``"archive"`` (the k best results in the run), ``"archive_diverse"``
            (k well-separated good results) or ``"archive_leaf"`` (the best
            point of each of the k best Splitter leaves — k different
            basins); ``None`` (default) is the cold start, statement for
            statement the behaviour shipped before.  Seeds are placed into
            the population **directly, as evaluated results**, so warm
            starting costs zero evaluations; any shortfall is filled by the
            cold random path, and the next good points seed the external
            archive.  Needs points to exist: at ``t = 0`` the archive is
            empty and the heuristic silently cold-starts.
        seed: Optional seed for the per-instance RNG.  ``None`` (default)
            uses the module's strategy-derived ``self.rng`` stream.
        name: Override the heuristic's display name.

    Notes:
        - All numeric arguments are validated; bad values raise
          :class:`ValueError`.
        - Like every Panobbgo heuristic, all state is per-instance.
          Multiple ``LSHADE`` heuristics in one strategy run
          independently and never share memory bins.
        - LPSR scales the population by progress
          ``len(strategy.results) / strategy.config.max_eval``.  When
          the budget is unknown (no ``max_eval``, zero, or non-numeric)
          the heuristic falls back to a *constant* population at
          ``NP_init``.
    """

    #: Per-dimension coefficient of the ``NP_init="auto"`` rule, overridable
    #: per subclass: the measured optimum is not the same swarm size for every
    #: variant of the algorithm (see :data:`_AUTO_DIM_COEF` and
    #: :class:`~panobbgo.heuristics.nl_shade_lbc.NLSHADE_LBC`).
    AUTO_DIM_COEF: float = _AUTO_DIM_COEF

    def __init__(
        self,
        strategy,
        NP_init: Union[int, str] = "auto",
        NP_min: int = _DEFAULT_NP_MIN,
        H: int = _DEFAULT_H,
        p_best: float = _DEFAULT_P_BEST,
        p_best_end: Optional[float] = None,
        archive_factor: float = _DEFAULT_ARCHIVE_FACTOR,
        F_schedule: Optional[Union[bool, str]] = None,
        warm_start: Optional[str] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        # ``NP_init="auto"`` resolves to a concrete budget-adaptive size here,
        # so every downstream code path (validation, ``on_start``, LPSR, and all
        # subclasses) sees a normal ``int`` and needs no further branching.
        if isinstance(NP_init, str):
            if NP_init != "auto":
                raise ValueError(f"LSHADE: NP_init string must be 'auto', got {NP_init!r}")
            NP_init = _resolve_auto_np_init(strategy, NP_min, type(self).AUTO_DIM_COEF)
        # A bool is an ``int`` subclass — reject it explicitly so ``True`` / ``False``
        # don't silently become populations of size 1 / 0.
        if isinstance(NP_init, bool) or not isinstance(NP_init, int):
            raise ValueError(f"LSHADE: NP_init must be an integer or 'auto', got {NP_init!r}")
        if NP_init < 4:
            raise ValueError(f"LSHADE: NP_init must be >= 4, got {NP_init}")
        if not isinstance(NP_min, int):
            raise ValueError(f"LSHADE: NP_min must be an integer, got {NP_min!r}")
        if NP_min < 4:
            raise ValueError(f"LSHADE: NP_min must be >= 4, got {NP_min}")
        if NP_min > NP_init:
            raise ValueError(f"LSHADE: NP_min ({NP_min}) must be <= NP_init ({NP_init})")
        if not isinstance(H, int):
            raise ValueError(f"LSHADE: H must be an integer, got {H!r}")
        if H < 1:
            raise ValueError(f"LSHADE: H must be >= 1, got {H}")
        if not np.isfinite(p_best) or not (0.0 < p_best <= 1.0):
            raise ValueError(f"LSHADE: p_best must be in (0, 1], got {p_best}")
        if p_best_end is not None and (not np.isfinite(p_best_end) or not (0.0 < p_best_end <= 1.0)):
            raise ValueError(f"LSHADE: p_best_end must be in (0, 1] when set, got {p_best_end}")
        if not np.isfinite(archive_factor) or archive_factor < 0.0:
            raise ValueError(f"LSHADE: archive_factor must be a non-negative finite float, got {archive_factor}")
        if warm_start is not None and warm_start not in Heuristic.WARM_START_MODES:
            raise ValueError(
                f"LSHADE: warm_start must be None or one of {Heuristic.WARM_START_MODES}, got {warm_start!r}"
            )
        # ``_normalize_F_schedule`` validates and maps the input onto a
        # regime name; backward-compat bool inputs collapse onto the
        # canonical string regimes.
        normalized_F_schedule = _normalize_F_schedule(F_schedule)

        super().__init__(strategy, name=name or "LSHADE")
        self.NP_init: int = NP_init
        self.NP_min: int = NP_min
        self.H: int = H
        self.p_best: float = float(p_best)
        self.p_best_end: Optional[float] = None if p_best_end is None else float(p_best_end)
        self.archive_factor: float = float(archive_factor)
        self.F_schedule: Optional[str] = normalized_F_schedule
        #: Archive selector for :meth:`_warm_start_population`, or ``None``
        #: for the cold start.  A *string*, deliberately not a callable: the
        #: trigger is :meth:`warm_start_now`.
        self.warm_start: Optional[str] = warm_start
        #: Region the next warm start is restricted to, or ``None`` for the
        #: whole archive.  Written by
        #: :meth:`panobbgo.strategies.blocks.StrategyBlockBandit._apply_pending_region`
        #: on the main thread just before a block opens, and cleared again
        #: right after — a *one-shot* hand-off from
        #: :class:`~panobbgo.heuristics.meta.MetaAnalyst`
        #: (``planning/DESIGN_meta_level_2026-09-10.md`` §2).  Anything
        #: :meth:`~panobbgo.core.Heuristic.archive_seed` accepts as ``box``:
        #: a :class:`~panobbgo.analyzers.splitter.Splitter.Box` or a
        #: ``(dim, 2)`` bounds array.
        self.warm_start_box: Any = None
        self._rng: np.random.Generator = self.derive_rng(seed)
        # Ranking-key memo, see :meth:`_fx_of`.
        self._fx_cache: Dict[int, float] = {}
        self._fx_keep: List[Result] = []

        # Success-history memory.  Initial value 0.5 per the SHADE paper.
        self._M_F: np.ndarray = np.full(H, 0.5, dtype=float)
        self._M_CR: np.ndarray = np.full(H, 0.5, dtype=float)
        self._mem_ptr: int = 0

        # Population bookkeeping.  ``_population[i]`` is one of:
        #   * ``None``                — slot still pending its initial fill
        #   * a :class:`Result`       — currently-occupied slot
        #   * :data:`_DROPPED`        — slot removed by LPSR; future returns
        #                               for this slot are silently dropped.
        self._population: List[Optional[Result]] = []
        self._NP_current: int = NP_init

        # External archive of replaced parents.  Stored as raw position
        # vectors so we don't carry full :class:`Result` overhead.
        self._archive: List[np.ndarray] = []

        # Pending trials: req_id -> (slot_idx, F, CR).  Initial random
        # trials use F=NaN, CR=NaN so they don't contribute to the
        # success memory.
        self._pending: Dict[str, "_TrialMeta"] = {}

        # Current-generation success buffer.  At end of generation we
        # update memory and clear.
        self._gen_completed: int = 0  # evolutionary trials finished this gen
        self._success_F: List[float] = []
        self._success_CR: List[float] = []
        self._success_delta: List[float] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _max_eval(self) -> Optional[float]:
        """Return the strategy's evaluation budget, or ``None`` if unknown."""
        try:
            v = float(self.strategy.config.max_eval)  # type: ignore[union-attr]
        except Exception:
            return None
        if not np.isfinite(v) or v <= 0.0:
            return None
        return v

    def _progress(self) -> Optional[float]:
        """Return ``len(strategy.results) / max_eval`` clipped to ``[0, 1]``.

        Returns ``None`` when the budget is unknown so callers can
        distinguish "early phase" (progress = 0.0) from "no budget at
        all" and pick a safe fallback for each schedule.
        """
        max_eval = self._max_eval()
        if max_eval is None:
            return None
        try:
            current = float(len(self.strategy.results))
        except Exception:
            return None
        return float(np.clip(current / max_eval, 0.0, 1.0))

    def _current_p_best(self) -> float:
        """Return the ``p_best`` value to use for the next trial.

        When ``p_best_end`` is ``None`` this is the constant ``self.p_best``,
        i.e. the byte-identical L-SHADE behaviour shipped 2026-05-10.
        Otherwise it is a linearly-annealed schedule paced by
        ``len(strategy.results) / strategy.config.max_eval`` — the
        iLSHADE (Brest et al. 2016) / jSO (Brest et al. 2017) move that
        shrinks greediness as the population shrinks under LPSR.  When
        the budget is unknown (no ``max_eval``, zero, or non-numeric)
        the heuristic falls back to constant ``self.p_best`` rather than
        guessing a horizon.
        """
        if self.p_best_end is None:
            return self.p_best
        progress = self._progress()
        if progress is None:
            return self.p_best
        return self.p_best - (self.p_best - self.p_best_end) * progress

    def _apply_F_cap(self, F: float) -> float:
        """Apply the configured asymmetric F-cap regime, when opted in.

        ``F_schedule`` resolves to ``None`` (default / ``"off"`` /
        ``False``) → ``F`` is returned unchanged, byte-identical to the
        unclamped Tanabe-Fukunaga L-SHADE.  For an active regime name,
        the three-phase cap is keyed on
        ``progress = len(strategy.results) / strategy.config.max_eval``
        with the regime's ``(phase1_bound, phase2_bound, phase1_cap,
        phase2_cap)`` 4-tuple from :data:`_F_SCHEDULE_REGIMES`:

        * ``progress < phase1_bound``: ``F`` clamped at ``phase1_cap``.
        * ``phase1_bound ≤ progress < phase2_bound``: ``F`` clamped at
          ``phase2_cap``.
        * ``progress ≥ phase2_bound``: ``F`` left unclamped (still
          ``≤ 1.0`` from the sampler).

        When the strategy budget is unknown the cap is bypassed — the
        same safe fallback used by :meth:`_current_p_best` and
        :meth:`_apply_lpsr`.
        """
        if self.F_schedule is None:
            return F
        progress = self._progress()
        if progress is None:
            return F
        bound1, bound2, cap1, cap2 = _F_SCHEDULE_REGIMES[self.F_schedule]
        if progress < bound1:
            return min(F, cap1)
        if progress < bound2:
            return min(F, cap2)
        return F

    def _make_trial_meta(self, slot_idx: int, F: float, CR: float) -> "_TrialMeta":
        """Construct the per-trial bookkeeping record for ``_pending``.

        Default returns a plain :class:`_TrialMeta`.  Subclasses (e.g.
        :class:`~panobbgo.heuristics.lshade_ep_sin.LSHADE_EpSin`) override
        this to attach per-trial metadata (e.g. which sinusoid produced
        ``F``) needed by their adaptation rules.
        """
        return _TrialMeta(slot_idx=slot_idx, F=F, CR=CR)

    def _record_success(self, meta: "_TrialMeta", delta: float) -> None:
        """Hook called for every successful competitive trial.

        Default is a no-op.  Subclasses that adapt per-trial metadata
        (e.g. :class:`~panobbgo.heuristics.lshade_ep_sin.LSHADE_EpSin`'s
        sinusoidal-ensemble success counters and frequency memory) extend
        this without having to override the whole :meth:`on_new_results`
        loop.  ``meta`` is the trial's ``_pending`` record (post-pop),
        ``delta`` is the absolute fitness improvement that survived the
        constraint handler.
        """
        return

    def _emit_trial(self, x: np.ndarray, slot_idx: int, F: float, CR: float) -> bool:
        """Project, queue, and book-keep one candidate point."""
        if self._stopped:
            return False
        try:
            x_proj = self.problem.project(x)
        except Exception as exc:
            self.logger.debug(f"LSHADE: projection failed: {exc}")
            return False

        # Request id drawn from the instance RNG (not ``uuid4``/OS entropy) so
        # ``Result.who`` tags are reproducible under a fixed seed.
        who = self.new_who(self._rng)
        req_id = who.split(":", 1)[1]
        self._put(Point(x_proj, who))
        self._pending[req_id] = self._make_trial_meta(slot_idx, F, CR)
        return True

    def _live_indices(self) -> List[int]:
        """Indices of currently-filled, non-dropped population slots."""
        out: List[int] = []
        for i, slot in enumerate(self._population):
            if isinstance(slot, Result):
                out.append(i)
        return out

    def _fx_of(self, r: Result) -> float:
        """Scalar fitness for ranking — falls back to ``r.fx`` if no handler.

        Memoised per :class:`~panobbgo.lib.Result`: the population is ranked
        on every trial generation, so the same handful of results would
        otherwise be re-penalised thousands of times per run.  Results are
        immutable, so the cached value cannot go stale.
        """
        cached = self._fx_cache.get(id(r))
        if cached is not None:
            return cached
        handler = getattr(self.strategy, "constraint_handler", None)
        if handler is None:
            value = float(r.fx) if r.fx is not None else float("inf")
        else:
            value = handler.get_penalty_value(r)
        self._fx_cache[id(r)] = value
        self._fx_keep.append(r)  # keep alive so ``id`` cannot be reused
        if len(self._fx_keep) > 4 * max(self.NP_init, 1):
            self._fx_cache.clear()
            del self._fx_keep[:]
        return value

    def _archive_cap(self) -> int:
        """Maximum number of replaced parents the external archive retains.

        Default is the fixed ``archive_factor · NP_current`` cap from
        Tanabe-Fukunaga (2014).  Subclasses (e.g.
        :class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP`) override
        this to randomise the cap per generation.
        """
        return max(int(round(self.archive_factor * self._NP_current)), 0)

    def _trim_archive(self) -> None:
        """Cap the archive at :meth:`_archive_cap` (drop random entries)."""
        cap = self._archive_cap()
        while len(self._archive) > cap:
            j = int(self._rng.integers(0, len(self._archive)))
            self._archive.pop(j)

    def _select_r1(self, live: List[int], target_idx: int) -> Optional[int]:
        """Pick the index ``r1`` for the differential ``F · (x_r1 − x_r2)`` term.

        Default: uniform over live slots excluding the target — the
        Tanabe-Fukunaga / jSO behaviour.  Subclasses (e.g.
        :class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP`) override
        this to bias the choice toward higher-ranked individuals
        (rank-based selective pressure).  Returns ``None`` when no
        candidate is available so the caller can abort the trial.
        """
        r1_pool = [i for i in live if i != target_idx]
        if not r1_pool:
            return None
        return int(self._rng.choice(np.asarray(r1_pool)))

    def _sample_F_CR(self) -> tuple[float, float]:
        """Draw one ``(F, CR)`` pair from a random history bin."""
        r = int(self._rng.integers(0, self.H))
        # CR sampling: Normal(M_CR[r], 0.1), clamped to [0, 1].  The
        # M_CR = -1 sentinel collapses to deterministic CR = 0.
        m_cr = float(self._M_CR[r])
        if m_cr < 0:
            CR = 0.0
        else:
            CR = float(self._rng.normal(m_cr, _PARAM_SCALE))
            CR = float(np.clip(CR, 0.0, 1.0))

        # F sampling: Cauchy(M_F[r], 0.1), regenerate while F <= 0,
        # clip at 1.  Bounded redraws to prevent worst-case loops.
        m_f = float(self._M_F[r])
        F = 0.5
        for _ in range(_F_MAX_REDRAWS):
            f = m_f + _PARAM_SCALE * float(self._rng.standard_cauchy())
            if f > 0.0:
                F = float(min(f, 1.0))
                break
        return self._apply_F_cap(F), CR

    def _reflect_bounds(self, v: np.ndarray, x_target: np.ndarray) -> np.ndarray:
        """Midpoint reflection bounds repair (Tanabe-Fukunaga §III-A)."""
        lb = self.problem.box[:, 0]
        ub = self.problem.box[:, 1]
        out = v.copy()
        below = out < lb
        if np.any(below):
            out[below] = (lb[below] + x_target[below]) / 2.0
        above = out > ub
        if np.any(above):
            out[above] = (ub[above] + x_target[above]) / 2.0
        return out

    def _generate_trial(self, target_idx: int) -> None:
        """Build and emit one ``current-to-pbest/1`` trial vector."""
        live = self._live_indices()
        if len(live) < 4 or target_idx not in live:
            return
        slot = self._population[target_idx]
        if not isinstance(slot, Result):
            return

        F, CR = self._sample_F_CR()
        x_target = np.asarray(slot.x, dtype=float)

        # pbest: top p% of live population by fitness (ascending — best first).
        # ``_current_p_best`` honours the optional iLSHADE / jSO linearly-
        # decreasing schedule when ``p_best_end`` is set; otherwise it is
        # the constant ``self.p_best``.
        sorted_live = sorted(live, key=lambda i: self._fx_of(self._population[i]))  # type: ignore[arg-type]
        p_eff = self._current_p_best()
        p_count = max(int(np.ceil(p_eff * len(sorted_live))), 1)
        pbest_pool = sorted_live[:p_count]
        pbest_idx = int(self._rng.choice(np.asarray(pbest_pool)))
        pbest_slot = self._population[pbest_idx]
        if not isinstance(pbest_slot, Result):
            return
        x_pbest = np.asarray(pbest_slot.x, dtype=float)

        # r1 from live population, distinct from target.
        r1 = self._select_r1(live, target_idx)
        if r1 is None:
            return
        r1_slot = self._population[r1]
        if not isinstance(r1_slot, Result):
            return
        x_r1 = np.asarray(r1_slot.x, dtype=float)

        # r2 from (live ∪ archive) \ {target, r1}.
        union: List[np.ndarray] = []
        for i in live:
            if i == target_idx or i == r1:
                continue
            slot_i = self._population[i]
            if isinstance(slot_i, Result):
                union.append(np.asarray(slot_i.x, dtype=float))
        union.extend(self._archive)
        if not union:
            return
        x_r2 = union[int(self._rng.integers(0, len(union)))]

        # Mutation: current-to-pbest/1.
        v = x_target + F * (x_pbest - x_target) + F * (x_r1 - x_r2)
        v = self._reflect_bounds(v, x_target)

        # Binomial crossover with at least one component swapped.
        dim = self.problem.dim
        cross = self._rng.random(dim) < CR
        j_rand = int(self._rng.integers(0, dim))
        cross[j_rand] = True
        u = np.where(cross, v, x_target)

        self._emit_trial(u, target_idx, F, CR)

    def _update_memory(self) -> None:
        """Apply the weighted Lehmer-mean memory update for one generation."""
        if not self._success_F:
            return  # no successes — leave memory untouched
        F_arr = np.asarray(self._success_F, dtype=float)
        CR_arr = np.asarray(self._success_CR, dtype=float)
        delta_arr = np.asarray(self._success_delta, dtype=float)
        total = float(delta_arr.sum())
        if total > 0.0:
            w = delta_arr / total
        else:
            w = np.full_like(delta_arr, 1.0 / len(delta_arr))

        # F: weighted Lehmer mean; F is always > 0 by construction.
        F_num = float(np.sum(w * F_arr * F_arr))
        F_den = float(np.sum(w * F_arr))
        if F_den > 0.0:
            self._M_F[self._mem_ptr] = float(np.clip(F_num / F_den, 0.0, 1.0))

        # CR: if all successes had CR = 0 OR the bin is already terminal,
        # plant the terminal sentinel (-1).  Otherwise weighted Lehmer mean.
        cr_max = float(CR_arr.max())
        if cr_max <= 0.0 or self._M_CR[self._mem_ptr] < 0.0:
            self._M_CR[self._mem_ptr] = _CR_TERMINAL
        else:
            CR_num = float(np.sum(w * CR_arr * CR_arr))
            CR_den = float(np.sum(w * CR_arr))
            if CR_den > 0.0:
                self._M_CR[self._mem_ptr] = float(np.clip(CR_num / CR_den, 0.0, 1.0))

        self._mem_ptr = (self._mem_ptr + 1) % self.H

    def _lpsr_target(self, progress: float) -> int:
        """Target population size at ``progress`` (Tanabe-Fukunaga 2014, linear).

        Linear interpolation from ``NP_init`` (progress 0) down to
        ``NP_min`` (progress 1).  Subclasses (e.g.
        :class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP`) override
        this with a non-linear schedule.
        """
        return int(round(self.NP_init - (self.NP_init - self.NP_min) * progress))

    def _apply_lpsr(self) -> None:
        """Shrink the population to ``NP_target`` based on budget progress."""
        progress = self._progress()
        if progress is None:
            return
        target = self._lpsr_target(progress)
        target = max(target, self.NP_min)
        target = min(target, self._NP_current)
        if target >= self._NP_current:
            return

        # Drop the worst (NP_current - target) live slots by fitness.
        live = self._live_indices()
        if len(live) <= target:
            self._NP_current = target
            self._trim_archive()
            return
        sorted_live = sorted(live, key=lambda i: self._fx_of(self._population[i]))  # type: ignore[arg-type]
        n_drop = self._NP_current - target
        for j in sorted_live[-n_drop:]:  # worst n_drop
            self._population[j] = _DROPPED  # type: ignore[assignment]
        self._NP_current = target
        self._trim_archive()

    def _end_of_generation(self) -> None:
        """Run memory + LPSR updates and reset the success buffer."""
        self._update_memory()
        self._apply_lpsr()
        self._gen_completed = 0
        self._success_F.clear()
        self._success_CR.clear()
        self._success_delta.clear()

    def _wake_idle_slots(self) -> None:
        """Make sure every live slot has at most one pending trial.

        Slots that finished their initial random fill before the
        population reached the four-individual threshold for
        ``current-to-pbest/1`` end up *idle* — filled, but with no
        in-flight trial.  Once the threshold is met we kick them off
        with their first evolutionary trial so the swarm gets back to
        full async throughput.
        """
        live = set(self._live_indices())
        if len(live) < 4:
            return
        active = {meta.slot_idx for meta in self._pending.values()}
        for slot_idx in sorted(live - active):
            self._generate_trial(slot_idx)

    def _init_memory(self) -> None:
        """(Re-)plant the initial success-history memory.

        A hook rather than two assignments so that a subclass with its own
        initial values (jSO) has them in place *before* the first trial is
        generated — which, on the warm-start path, happens inside
        :meth:`on_start` itself rather than after the first batch of results.
        """
        self._M_F[:] = 0.5
        self._M_CR[:] = 0.5

    def _warm_start_population(self) -> bool:
        """Fill open population slots from the shared archive.  ``True`` iff seeded.

        The seeds are inserted as :class:`~panobbgo.lib.Result` objects —
        the slot type is ``Optional[Result]`` either way — so a warm start
        costs **zero evaluations**: the points were already paid for by
        whoever produced them.  Slots the archive cannot cover keep the cold
        path (a uniform random trial for an empty slot, the incumbent for a
        live one), the *next* good points seed the external archive of
        replaced parents (preferring points from *other* heuristics, which is
        the whole point of a shared archive), and :meth:`_wake_idle_slots`
        starts the first generation of real trials.
        """
        if not self.warm_start:
            return False
        if not self._population:
            self._population = [None] * self.NP_init
            self._NP_current = self.NP_init

        slots = [i for i, slot in enumerate(self._population) if not isinstance(slot, _Dropped)]

        # Find out whether the archive has anything at all *before* asking for
        # the archive cap.  :meth:`_archive_cap` is not a pure accessor in
        # every subclass: :meth:`NLSHADE_RSP._archive_cap
        # <panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP._archive_cap>` draws
        # the per-generation cap from ``self._rng`` when
        # ``adaptive_archive=True`` (the default).  Computing it on a path
        # that then bails out consumed an RNG draw, so on NL-SHADE-RSP and
        # NL-SHADE-LBC merely *passing* ``warm_start="archive"`` shifted the
        # whole initial population even though the empty archive meant the
        # warm start never happened — which turns every paired "warm vs cold"
        # A/B on those arms into a comparison of two different RNG streams
        # (``DISCOVERY_2026-09-09.md`` §18).  ``archive_seed`` is a pure
        # query with no randomness of its own and returns a prefix of the
        # same ranking for every ``k``, so a one-element probe is empty
        # exactly when the full request would be.
        if not self.archive_seed(1, mode=self.warm_start, box=self.warm_start_box):
            return False  # empty archive: the caller falls back to the cold path

        cap = self._archive_cap()
        pool = self.archive_seed(len(slots) + cap, mode=self.warm_start, box=self.warm_start_box)
        if not pool:
            return False  # nothing requestable (every slot dropped and cap 0)

        seeds = pool[: len(slots)]
        for i, r in zip(slots, seeds):
            self._population[i] = r
        for i in slots[len(seeds) :]:
            if self._population[i] is None:
                # shortfall: the existing random path, unchanged
                x = self.problem.random_point(rng=self._rng)
                self._emit_trial(x, i, F=float("nan"), CR=float("nan"))

        self._seed_archive(pool[len(seeds) :], cap)
        self._wake_idle_slots()
        return True

    def _seed_archive(self, rest: List[Result], cap: int) -> None:
        """Prime the external archive with good points, foreign ones first."""
        if cap <= 0 or not rest:
            return
        prefix = f"{self.name}:"
        foreign = [r for r in rest if not (getattr(r, "who", "") or "").startswith(prefix)]
        own = [r for r in rest if (getattr(r, "who", "") or "").startswith(prefix)]
        for r in (foreign + own)[:cap]:
            self._archive.append(np.asarray(r.x, dtype=float))
        # ``on_start`` cleared the archive first, but a mid-run re-seed
        # (:meth:`warm_start_now`) appends on top of a live one.
        self._trim_archive()

    # ------------------------------------------------------------------
    # Heuristic interface
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Allocate state and emit ``NP_init`` random initial trials.

        With ``warm_start`` set and a non-empty archive the initial trials
        are replaced by seeds taken straight from the shared archive; with
        ``warm_start=None`` — or an archive that has nothing to give — this
        is the cold start, statement for statement as before.
        """
        self._population = [None] * self.NP_init
        self._NP_current = self.NP_init
        self._archive.clear()
        self._pending.clear()
        self._init_memory()
        self._mem_ptr = 0
        self._gen_completed = 0
        self._success_F.clear()
        self._success_CR.clear()
        self._success_delta.clear()

        if self.warm_start and self._warm_start_population():
            return

        for i in range(self.NP_init):
            x = self.problem.random_point(rng=self._rng)
            self._emit_trial(x, i, F=float("nan"), CR=float("nan"))

    def warm_start_now(self) -> bool:
        """Re-seed the population from the shared archive on re-acquisition.

        The direct-call hook of
        :class:`~panobbgo.strategies.blocks.StrategyBlockBandit`: an arm that
        is handed a fresh block with an empty queue drops its stale in-flight
        trials and restarts from the best points the *portfolio* has found,
        without spending an evaluation on any of them.  Adapted state (the
        success-history memory, the LPSR population size) is deliberately
        kept — this is a re-seeding, not a restart.
        """
        if self._stopped or not self.warm_start:
            return False
        self.clear_output()
        self._pending.clear()
        return self._warm_start_population()

    def on_new_results(self, results) -> None:
        """Process incoming evaluations and dispatch follow-up trials."""
        if not self._population:
            return  # not started yet

        prefix = f"{self.name}:"
        handler = self.strategy.constraint_handler
        for r in results:
            who: str = getattr(r, "who", "") or ""
            if not who.startswith(prefix):
                continue
            req_id = who[len(prefix) :]
            meta = self._pending.pop(req_id, None)
            if meta is None:
                continue  # stale or unknown trial id
            slot_idx = meta.slot_idx

            # Slot dropped by LPSR after we issued this trial — discard.
            if slot_idx >= len(self._population):
                continue
            slot = self._population[slot_idx]
            if isinstance(slot, _Dropped):
                continue

            if slot is None:
                # Initial random fill — just store.  No success counted.
                self._population[slot_idx] = r
            else:
                # Competitive trial.  Compete; loser may go to archive.
                target = slot
                if handler.is_better(target, r):
                    delta = abs(self._fx_of(target) - self._fx_of(r))
                    self._archive.append(np.asarray(target.x, dtype=float))
                    self._trim_archive()
                    self._population[slot_idx] = r
                    if not np.isnan(meta.F) and not np.isnan(meta.CR):
                        self._success_F.append(meta.F)
                        self._success_CR.append(meta.CR)
                        # Floor delta so an unweighted-but-real success
                        # still influences the Lehmer mean.
                        self._success_delta.append(max(float(delta), 1e-30))
                        self._record_success(meta, float(delta))
                self._gen_completed += 1

                if self._gen_completed >= max(self._NP_current, 1):
                    self._end_of_generation()

            # Emit a follow-up trial for this slot if it survived.
            if slot_idx < len(self._population) and isinstance(self._population[slot_idx], Result):
                self._generate_trial(slot_idx)

            # Wake up any idle live slots (filled, but no pending trial).
            self._wake_idle_slots()

    def on_restart(self, center, reason: str = "") -> None:
        """Drop in-flight state and reseed the population around ``center``.

        Mirrors the warm-restart pattern used by
        :class:`~panobbgo.heuristics.pso.PSO` and the IPOP/BIPOP CMA-ES
        variants: archive cleared, memory bins reset to 0.5, slots
        re-randomised in a small ball around ``center`` (or random in
        the box if ``center`` is ``None``), and a fresh round of
        initial-random trials emitted.
        """
        if self._stopped:
            return
        self.clear_output()
        self._pending.clear()
        if not self._population:
            return  # not started yet — nothing to reset

        self._archive.clear()
        self._init_memory()
        self._mem_ptr = 0
        self._gen_completed = 0
        self._success_F.clear()
        self._success_CR.clear()
        self._success_delta.clear()
        # Restore full-size population; LPSR will shrink it again from scratch.
        self._population = [None] * self.NP_init
        self._NP_current = self.NP_init

        # A warm-started arm re-seeds from the shared archive instead of
        # re-evaluating ``NP_init`` fresh points — the restart's ``center``
        # is the incumbent, and the archive's best points are around it
        # anyway.  Design §2: the random re-emission below is pure waste.
        if self.warm_start and self._warm_start_population():
            return

        ranges = self.problem.box[:, 1] - self.problem.box[:, 0]
        ball = 0.1 * ranges  # small reseed ball; conservative
        if center is None:
            base = None
        else:
            base = np.asarray(center, dtype=float)
        for i in range(self.NP_init):
            if base is None:
                x = self.problem.random_point(rng=self._rng)
            else:
                offset = self._rng.uniform(-ball, ball)
                x = self.problem.project(base + offset)
            self._emit_trial(x, i, F=float("nan"), CR=float("nan"))


class _TrialMeta:
    """Per-trial bookkeeping used to identify which slot/F/CR a result came from."""

    __slots__ = ("slot_idx", "F", "CR")

    def __init__(self, slot_idx: int, F: float, CR: float) -> None:
        self.slot_idx = slot_idx
        self.F = F
        self.CR = CR
