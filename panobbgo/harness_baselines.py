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
External baseline strategies for the benchmark harness
======================================================

Without external reference solvers, a harness score only answers *"is
Panobbgo better than its previous self"*.  This module plugs well-known
external optimizers into the same :class:`~panobbgo.benchmark.StrategySpec`
interface the harnesses (:mod:`panobbgo.harness`, :mod:`panobbgo.harness_ioh`)
already use, so their scores sit next to Panobbgo's at equal budget.

Always available (``numpy`` / ``scipy`` only) — the ``--baselines`` set:

- :class:`RandomSearchStrategy` — pure uniform random search (score floor).
- :class:`SciPyDEStrategy` — ``scipy.optimize.differential_evolution``
  (competitive, general-purpose global optimizer).
- :class:`SciPyAnnealStrategy` — ``scipy.optimize.dual_annealing``
  (competitive simulated-annealing hybrid).

External, from the optional ``baselines`` extra (``uv sync --extra
baselines``) — opt-in by name, see :data:`EXTERNAL_BASELINE_NAMES`:

- :class:`PycmaIPOPStrategy`, :class:`PycmaBIPOPStrategy` — pycma's
  CMA-ES with IPOP / BIPOP restarts (the restart schedule of ``cma.fmin2``
  re-implemented over ask/tell, since ``fmin2`` owns the evaluation loop).
- :class:`NevergradNGOptStrategy` — Nevergrad's ``NGOpt`` meta-optimizer,
  plus :class:`NevergradCMAStrategy` and :class:`NevergradTwoPointsDEStrategy`
  as cross-checks.
- :class:`OptunaCmaEsStrategy`, :class:`OptunaTPEStrategy` — Optuna's
  ``CmaEsSampler`` (no restarts) and ``TPESampler``.

External, expensive track, from the optional ``baselines-bo`` extra —
BoTorch qLogEI, TuRBO-1, SMAC3's BlackBox facade, Py-BOBYQA: see
:mod:`panobbgo.harness_baselines_bo`; registered here by name
(:data:`BO_BASELINE_NAMES`, and with the cheap track in
:data:`ALL_EXTERNAL_BASELINE_NAMES`).

Every cheap-track external baseline starts from a uniform random point of
the box (per seed), treats a NaN value as the worst (``+inf``), and is
deterministic for a fixed seed; see the section comment above
:class:`AskTellAdapter`.  The expensive-track ones have their own start
and failed-value rules (:mod:`panobbgo.harness_baselines_bo`).  Optuna's wall time grows quadratically with the
budget, so measure Optuna baselines with ``--no-timeout`` in
``benchmark_harness.py`` (its default per-run timeout is 120 s).

All adapters respect a strict evaluation budget (``config.max_eval``),
record every evaluation into a MultiIndex results DataFrame compatible
with :class:`panobbgo.harness.BenchmarkHarness`, and honour the harness
per-run seed.

Design notes
------------

* Baselines do **not** subclass :class:`~panobbgo.core.StrategyBase`.
  Instead they implement the subset of the strategy protocol the harness
  actually uses (``config.max_eval``, ``config.evaluation_method``,
  ``start()``, ``best``, ``results.results``) as a small duck-typed shim.
  This keeps the wrappers small, avoids spinning up the Dask/event-bus
  infrastructure which is meaningless for non-portfolio solvers, and makes
  budget enforcement deterministic.
* Budget enforcement is **hard**: the objective wrapper raises
  :class:`_BudgetExhausted` as soon as ``max_eval`` is reached.  The
  external solver's own stopping criterion may terminate earlier, but it
  can never overshoot, matching the harness contract.
* The results DataFrame uses the same MultiIndex columns
  ``(("fx", 0), ("who", 0), ("x", j), ...)`` that Panobbgo's own strategies
  emit, so :meth:`panobbgo.harness.BenchmarkHarness._get_column` and the
  convergence extractor work unchanged.
* The external baselines are **batch-capable**: each wraps its library in
  an :class:`AskTellAdapter` (``ask(n)`` returns up to ``n`` keyed
  candidates, ``tell(key, fx)`` reports one result, in any order).
  :class:`AskTellBaselineStrategy` drives it synchronously with batch size
  ``q = config.batch_size`` (default 1): ask ``q`` points, evaluate them
  in dispatch order, tell them.  With ``q > 1`` the results frame is in
  dispatch order.  On the virtual clock (a harness ``virtual=VirtualSpec``)
  the same adapter is driven asynchronously with ``q`` simulated workers
  (:func:`panobbgo.virtual_clock.run_ask_tell`): ask for up to the free
  workers, evaluate at dispatch, tell at completion time — so these
  baselines get ``aocc_time`` too.

Usage
-----

The ``--baselines`` CLI flag (``benchmark_harness.py``,
``scripts/ioh_benchmark.py``) or ``HarnessConfig(include_baselines=True)``
appends Random / SciPy DE / SciPy dual annealing.  The external baselines
are selected by name on top of that, so the default ``--baselines`` set
does not change::

    uv run python benchmark_harness.py run --standard --baselines
    uv run python scripts/ioh_benchmark.py run --standard --baselines \
        --strategies RoundRobin_CMAES Baseline_NGOpt Baseline_pycma_BIPOP

With baselines in the picture, the harness output lets the user read
Panobbgo's score side-by-side with the external references at equal
budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import functools
import importlib
import importlib.util
import math
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from panobbgo.benchmark import StrategySpec
from panobbgo.ioh_runner import _BudgetExhausted
from panobbgo.lib import EvaluationFailed, Point, Problem, Result

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: hard-budget objective wrapper
# ---------------------------------------------------------------------------


# ``_BudgetExhausted`` (imported above) is the one hard-stop signal, shared
# with ``IOHTracker(hard=True)``: an IOH tracker that raises it inside a
# baseline's objective ends the baseline cleanly instead of as a crash.


@dataclass
class _EvaluationLog:
    """Accumulates every ``(x, fx)`` pair the external solver evaluates.

    Used to build the harness-compatible results DataFrame and to compute
    the best-so-far record without leaning on the solver's own bookkeeping
    (which is implementation-specific and not always accessible).
    """

    who: str
    max_eval: int
    xs: List[np.ndarray] = field(default_factory=list)
    fxs: List[float] = field(default_factory=list)
    best_fx: float = float("inf")
    best_x: Optional[np.ndarray] = None
    #: Set by :meth:`BaselineStrategy.request_stop` (from another thread):
    #: the next objective call raises :class:`_BudgetExhausted` instead of
    #: evaluating, which ends the external solver the same way the budget does.
    stop_requested: bool = False

    def record(self, x: np.ndarray, fx: float) -> None:
        if len(self.fxs) >= self.max_eval:
            raise _BudgetExhausted()
        x_copy = np.asarray(x, dtype=np.float64).copy()
        fx_val = float(fx)
        self.xs.append(x_copy)
        self.fxs.append(fx_val)
        if np.isfinite(fx_val) and fx_val < self.best_fx:
            self.best_fx = fx_val
            self.best_x = x_copy


#: Penalty coefficient a baseline's objective applies on a constrained problem:
#: the ``rho`` of panobbgo's :class:`~panobbgo.lib.constraints.DefaultConstraintHandler`
#: (and of the constrained families' AOCC, ``harness_families.PENALTY_RHO``).
BASELINE_PENALTY_RHO: float = 100.0


def penalized_value(fx: float, cv_vec: Optional[np.ndarray], rho: float = BASELINE_PENALTY_RHO) -> float:
    r"""``fx + rho * cv`` with :math:`\mathrm{cv} = \lVert \max(g, 0) \rVert_2` — ``fx`` unchanged without constraints.

    The same ``cv`` as :attr:`Result.cv <panobbgo.lib.lib.Result.cv>` (a
    ``NaN`` entry is an unknown, i.e. infinite, violation), so a baseline
    minimises exactly what panobbgo's default constraint handler scalarises
    (:meth:`DefaultConstraintHandler.get_penalty_value
    <panobbgo.lib.constraints.DefaultConstraintHandler.get_penalty_value>`).
    """
    if cv_vec is None:
        return fx
    cv = Result(None, fx, cv_vec=np.asarray(cv_vec, dtype=np.float64)).cv
    return float(fx + rho * cv) if cv > 0.0 else fx


def _make_objective(problem: Problem, log: _EvaluationLog, nan_is_worst: bool = False) -> Callable[[np.ndarray], float]:
    """Return a scalar objective that projects into the box, evaluates the
    problem, logs the result, and enforces the evaluation budget.

    A simulated failure (:class:`~panobbgo.lib.lib.EvaluationFailed`, a
    family's failure region) is logged as NaN.  ``nan_is_worst`` hands the
    solver ``+inf`` for it instead (:func:`_nan_is_worst`) — for the SciPy
    solvers, which call the objective directly; the ask/tell adapters apply
    the same rule in their ``tell``.

    **Constraints.**  The baselines have no constraint handling of their own,
    so on a constrained problem (``eval_constraints`` returns a vector) the
    objective is the penalty value :func:`penalized_value` — ``f + 100·cv``,
    what panobbgo's default constraint handler minimises — instead of the
    bare ``f``, which would let a baseline score by luck on whichever side of
    the constraints it happens to land.  On an unconstrained problem the
    value is ``f``, bit for bit.
    """

    def objective(x: np.ndarray) -> float:
        if log.stop_requested:
            raise _BudgetExhausted()
        x_arr = np.asarray(x, dtype=np.float64)
        # Respect the problem's box — scipy solvers already pass candidates
        # inside bounds, but project defensively to cover edge cases with
        # solvers that occasionally probe slightly outside (``basinhopping``
        # has historically done this at tolerance boundaries).
        x_proj = problem.project(x_arr)
        try:
            fx = float(problem.eval(x_proj))
        except EvaluationFailed as exc:
            # A simulated crash / timeout (a family's failure region): the
            # call was made and paid for (the AOCC tracker has counted it),
            # but it has no value.  NaN is the "no value" answer; the solver
            # must not abort on the first failure.
            # DEBUG: a failure preset has hundreds of these per run.
            _logger.debug("%s: evaluation failed (%s); answering NaN", log.who, exc)
            fx = float("nan")
        else:
            fx = penalized_value(fx, problem.eval_constraints(x_proj))
        log.record(x_proj, fx)
        return _nan_is_worst(fx) if nan_is_worst else fx

    return objective


# ---------------------------------------------------------------------------
# Minimal strategy-like adapter for the benchmark harness
# ---------------------------------------------------------------------------


class _BaselineConfig:
    """The subset of :class:`panobbgo.config.Config` the harness touches.

    ``max_eval`` is respected as a hard budget.  ``evaluation_method`` is
    accepted (to match the harness contract) but ignored: baselines run
    the objective synchronously inside ``start()``.  ``batch_size`` is the
    number of candidates an :class:`AskTellBaselineStrategy` asks for
    before telling results (``q``); the SciPy baselines ignore it.  Set it
    through ``StrategySpec(config_overrides={"batch_size": q})``.
    """

    def __init__(self, max_eval: int = 1000) -> None:
        self.max_eval: int = max_eval
        self.evaluation_method: str = "threaded"
        self.batch_size: int = 1


class _BaselineResults:
    """Exposes ``.results`` as a pandas DataFrame with the MultiIndex
    columns the harness' ``_get_column`` expects."""

    def __init__(self) -> None:
        self.results: Optional[pd.DataFrame] = None

    def build(self, log: _EvaluationLog) -> None:
        """Materialize the MultiIndex DataFrame from accumulated evaluations.

        Columns follow the Panobbgo convention so that
        :meth:`panobbgo.harness.BenchmarkHarness._get_column` and the
        convergence extractor work without special-casing.
        """
        n = len(log.fxs)
        if n == 0:
            self.results = None
            return

        dim = log.xs[0].size
        data: Dict[Any, Any] = {}
        x_stack = np.vstack(log.xs)
        for j in range(dim):
            data[("x", j)] = x_stack[:, j]
        data[("fx", 0)] = np.asarray(log.fxs, dtype=np.float64)
        data[("cv", 0)] = np.zeros(n, dtype=np.float64)
        data[("who", 0)] = np.asarray([log.who] * n, dtype=object)
        data[("error", 0)] = np.zeros(n, dtype=np.float64)

        midx = pd.MultiIndex.from_tuples(list(data.keys()))
        self.results = pd.DataFrame(data, columns=midx)


class BaselineStrategy:
    """Minimal adapter that presents the strategy surface the benchmark
    harness uses.

    Subclasses override :meth:`_optimize` to implement the actual search
    algorithm, leaning on the provided :class:`_EvaluationLog` for
    bookkeeping and :func:`_make_objective` for the budget-safe wrapper.
    """

    #: Short identifier written to the ``who`` column of the results
    #: DataFrame.  Subclasses override.
    who: str = "Baseline"

    def __init__(self, problem: Problem, parse_args: bool = False, seed: Optional[int] = None) -> None:
        # ``parse_args`` accepted for API compatibility with
        # ``StrategyBase.__init__`` — baselines never parse CLI.
        del parse_args
        self.problem: Problem = problem
        #: Run seed handed in by the harness (``None`` → numpy global state).
        self.seed: Optional[int] = None if seed is None else int(seed)
        self.config: _BaselineConfig = _BaselineConfig()
        self.results: _BaselineResults = _BaselineResults()
        self._best: Optional[Result] = None
        self._stop_requested: bool = False
        self._log: Optional[_EvaluationLog] = None

    def _run_seed(self) -> int:
        """Integer seed for this run: the harness-provided ``seed`` if any,
        else a draw from numpy's global state (which the harness seeds)."""
        if self.seed is not None:
            return int(self.seed)
        return int(np.random.randint(0, 2**31 - 1))

    # -- harness compatibility shims ---------------------------------------

    def add(self, heuristic_class: type, **kwargs: Any) -> None:
        """No-op: baselines ignore Panobbgo heuristics entirely.

        Accepted so :meth:`panobbgo.benchmark.StrategySpec.create_strategy`
        can iterate through a (possibly empty) ``heuristics`` list without
        special-casing baselines.
        """
        del heuristic_class, kwargs

    def add_analyzer(self, analyzer: Any) -> None:
        """No-op: baselines have no event bus for analyzers to subscribe to."""
        del analyzer

    # -- run ---------------------------------------------------------------

    def start(self) -> None:
        """Execute the wrapped solver under the harness budget cap.

        Catches :class:`_BudgetExhausted` so that exceeding the budget from
        inside the external solver is a clean termination, not a crash.
        """
        log = _EvaluationLog(who=self.who, max_eval=max(1, int(self.config.max_eval)))
        # Publish the log before reading the flag: a request_stop() racing
        # with this start either sees the log or has already set the flag.
        self._log = log
        log.stop_requested = log.stop_requested or self._stop_requested

        try:
            self._optimize(log)
        except _BudgetExhausted:
            # Reached the hard budget mid-optimization — this is expected
            # and is exactly how the harness' own strategies terminate.
            pass
        except Exception:
            # Surface unexpected errors so the harness records them in the
            # RunRecord.error field.
            self.results.build(log)
            self._best = self._build_best_result(log)
            raise

        self.results.build(log)
        self._best = self._build_best_result(log)

    def request_stop(self) -> None:
        """End the run at the next objective call (same protocol as
        :meth:`panobbgo.core.StrategyBase.request_stop`).

        Thread-safe flag writes: the harness calls it from its timeout path
        while :meth:`start` runs the external solver on another thread.
        """
        self._stop_requested = True
        if self._log is not None:
            self._log.stop_requested = True

    def _optimize(self, log: _EvaluationLog) -> None:
        """Subclass hook: run the actual optimizer, calling the objective
        returned by :func:`_make_objective` up to ``log.max_eval`` times."""
        raise NotImplementedError

    # -- results -----------------------------------------------------------

    @property
    def best(self) -> Optional[Result]:
        """Return the best :class:`~panobbgo.lib.Result` seen, or ``None``
        if the solver produced no finite evaluation (e.g. immediate error)."""
        return self._best

    def _build_best_result(self, log: _EvaluationLog) -> Optional[Result]:
        if log.best_x is None:
            return None
        point = Point(log.best_x, self.who)
        return Result(point, log.best_fx)


# ---------------------------------------------------------------------------
# Concrete baselines
# ---------------------------------------------------------------------------


class RandomSearchStrategy(BaselineStrategy):
    """Pure uniform random search over the problem box.

    This is the **score floor**: any serious optimizer should beat it on
    most problems.  Its composite score on the standard battery gives us a
    lower reference point for what "doing nothing adaptive" achieves under
    the harness formula.
    """

    who: str = "Random"

    def _optimize(self, log: _EvaluationLog) -> None:
        objective = _make_objective(self.problem, log)
        lo = self.problem.box[:, 0]
        hi = self.problem.box[:, 1]
        rng = np.random.default_rng(self._run_seed())
        for _ in range(log.max_eval):
            x = rng.uniform(lo, hi, size=self.problem.dim)
            objective(x)


class SciPyDEStrategy(BaselineStrategy):
    """Adapter for :func:`scipy.optimize.differential_evolution`.

    Differential Evolution is a strong population-based global optimizer
    and a standard baseline in the black-box optimization literature.  We
    configure it with a small population so that the harness' 75-to-500
    evaluation budgets produce meaningful generations.
    """

    who: str = "SciPyDE"

    def __init__(
        self,
        problem: Problem,
        parse_args: bool = False,
        popsize: int = 10,
        tol: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(problem, parse_args=parse_args, seed=seed)
        self._popsize = popsize
        self._tol = tol

    def _optimize(self, log: _EvaluationLog) -> None:
        from scipy.optimize import differential_evolution

        objective = _make_objective(self.problem, log, nan_is_worst=True)
        bounds = [(float(lo), float(hi)) for lo, hi in self.problem.box]

        # DE uses popsize * dim candidates per generation.  With a hard
        # evaluation cap, set ``maxiter`` generously — the _BudgetExhausted
        # exception inside the objective is the real stopping criterion.
        #
        # ``tol=0`` forces DE to use the full budget; otherwise it will
        # terminate early on convergence of the population standard
        # deviation, wasting evaluations we could spend exploring.
        #
        # Seed derivation: the harness calls ``np.random.seed(seed)`` just
        # before this runs, so ``int(np.random.randint(...))`` gives us a
        # deterministic integer we can pass to scipy's RNG.
        de_seed = self._run_seed()
        max_generations = max(1, log.max_eval // max(1, self._popsize * self.problem.dim))

        # ``differential_evolution`` uses ``rng`` in scipy >= 1.15 (``seed``
        # is retained as a deprecated alias).  Prefer ``rng``.
        # A _BudgetExhausted from the objective ends the run; start() catches it.
        differential_evolution(
            objective,
            bounds=bounds,
            popsize=self._popsize,
            maxiter=max_generations,
            tol=self._tol,
            rng=de_seed,
            polish=False,  # we enforce budget ourselves; polish would exceed
            init="sobol",
            updating="deferred",
        )


class SciPyAnnealStrategy(BaselineStrategy):
    """Adapter for :func:`scipy.optimize.dual_annealing`.

    Dual annealing combines generalized simulated annealing with a local
    search step — a competitive global optimizer on multimodal problems.
    We pair it with ``no_local_search=False`` so it can exploit the
    built-in L-BFGS-B polish, which is fair because CMA-ES and BayesOpt
    harness strategies also include a local-refinement heuristic.
    """

    who: str = "SciPyAnneal"

    def _optimize(self, log: _EvaluationLog) -> None:
        from scipy.optimize import dual_annealing

        objective = _make_objective(self.problem, log, nan_is_worst=True)
        bounds = [(float(lo), float(hi)) for lo, hi in self.problem.box]

        da_seed = self._run_seed()

        # ``dual_annealing`` uses ``rng`` in scipy >= 1.15.
        dual_annealing(
            objective,
            bounds=bounds,
            maxfun=log.max_eval,
            # ``maxiter`` is the cap on the outer SA loop; we let the
            # hard ``maxfun`` + our _BudgetExhausted guard terminate it.
            maxiter=10_000,
            rng=da_seed,
            no_local_search=False,
        )


# ---------------------------------------------------------------------------
# Batch-capable ask/tell baselines (optional ``baselines`` extra)
# ---------------------------------------------------------------------------
#
# Conventions shared by every external baseline:
#
# * **Start point.**  Each run starts from a point drawn uniformly from the
#   box with the run seed (pycma: ``x0``; Nevergrad: the parametrization's
#   ``init``; Optuna CMA-ES: ``x0``; Optuna TPE samples its startup trials
#   at random anyway).  No library starts at the box centre, where several
#   classic test functions have their optimum.
# * **Failed values.**  See :meth:`AskTellAdapter.tell`.
# * **Determinism.**  Every random source is derived from the run seed, and
#   numpy's global RNG — which pycma and Nevergrad's recast CMA fall back
#   to — is seeded from it for the run and restored afterwards.


def _extra_hint(extra: str = "baselines") -> str:
    """The install hint for an optional extra (``baselines`` or ``baselines-bo``).

    Names only that extra (``baselines`` never pulls torch).  ``uv sync`` is
    exact — it removes what the command does not name — so the hint says to
    add the other extras in use.
    """
    return (
        f"install the optional extra `{extra}`: `uv sync --extra dev --extra {extra}` (uv sync is exact:"
        f" add every other extra you use) or `pip install 'panobbgo[{extra}]'`"
    )


_EXTRA_HINT = _extra_hint()


def _require(module: str, extra: str = "baselines") -> Any:
    """Import ``module`` or fail with a hint at the optional ``extra``."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"{module!r} is not installed; {_extra_hint(extra)}") from exc


def _seed32(seed: int) -> int:
    """Fold a run seed into ``[0, 2**32)``, the range every library accepts."""
    return int(seed) % (2**32)


def _nan_is_worst(fx: float) -> float:
    """The shared failed-value rule: NaN becomes ``+inf``, everything else passes."""
    fx = float(fx)
    return float("inf") if math.isnan(fx) else fx


class AskTellAdapter:
    """Keyed ask/tell interface over one external optimizer.

    The unit a synchronous driver (:class:`AskTellBaselineStrategy`) or an
    asynchronous one (a virtual-clock simulator with ``q`` workers) steps.
    Coordinates are in the problem's box.

    * ``ask(n)`` returns up to ``n`` new candidates as ``(key, x)`` pairs;
      keys are unique ints.  It may return fewer than ``n``: a generational
      optimizer (CMA-ES) hands out the rest of its current generation and
      cannot sample the next one before the current one is told.  It
      returns ``[]`` only while every proposable point waits on a result.
    * ``tell(key, fx)`` reports the value of one asked candidate, in any
      order.
    * ``close()`` releases what the library holds (threads).
    """

    #: Modules the adapter imports; checked when a spec is built, so a
    #: missing extra fails before any run.
    requires: Tuple[str, ...] = ()

    def ask(self, n: int) -> List[Tuple[int, np.ndarray]]:
        """Return up to ``n`` new ``(key, x)`` candidates."""
        raise NotImplementedError

    def tell(self, key: int, fx: float) -> None:
        """Report the objective value of the candidate ``key``.

        One rule for every adapter: a NaN value (a failed or timed-out
        evaluation) is the worst value and is told as ``+inf`` — to Optuna
        as a COMPLETE trial with value ``inf``, not a FAIL, so its model
        learns that the region is bad.  ``±inf`` values the objective itself
        returns pass through unchanged (Nevergrad clips them internally).
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release resources (threads) the library holds; the default does nothing."""


class AskTellBaselineStrategy(BaselineStrategy):
    """Baseline driven through an :class:`AskTellAdapter` with batch size ``q``.

    ``q`` is ``config.batch_size`` (default 1; the ``batch_size`` argument or
    ``StrategySpec(config_overrides={"batch_size": q})`` set it).  Each step
    asks for ``min(q, remaining budget)`` candidates, evaluates them in
    dispatch order and tells them, so the results frame is in dispatch
    order and the budget cap is exact.  ``q`` is also passed to the adapter:
    Nevergrad takes it as ``num_workers`` (which changes what NGOpt picks),
    pycma raises its population to at least ``q``.
    """

    #: Modules the strategy's adapter imports (see :func:`make_baseline_strategies`).
    requires: Tuple[str, ...] = ()
    #: The optional extra that provides :attr:`requires` (named in the install hint).
    extra: str = "baselines"

    def __init__(
        self,
        problem: Problem,
        parse_args: bool = False,
        seed: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        super().__init__(problem, parse_args=parse_args, seed=seed)
        if batch_size is not None:
            self.config.batch_size = int(batch_size)

    def make_adapter(self, seed: int, budget: int, batch_size: int) -> AskTellAdapter:
        """Build a fresh adapter for one run.

        Args:
            seed: Integer run seed; the adapter is deterministic given it.
            budget: Evaluation budget of the run (some libraries size
                themselves by it).
            batch_size: Number of candidates in flight at once (``q``).
        """
        raise NotImplementedError

    #: Evaluation observer of a virtual-clock run (an IOH tracker), set by
    #: :meth:`VirtualSpec.apply <panobbgo.virtual_clock.VirtualSpec.apply>`.
    _virtual_observer: Any = None

    @property
    def _virtual(self) -> bool:
        """Run on the virtual clock (``VirtualSpec.apply`` set ``evaluation_method = "virtual"``)?"""
        return getattr(self.config, "evaluation_method", None) == "virtual"

    def _optimize(self, log: _EvaluationLog) -> None:
        # On the virtual clock ``q`` is the number of simulated workers.
        q = max(1, int(getattr(self.config, "virtual_workers", 1) if self._virtual else self.config.batch_size))
        seed = self._run_seed()
        # pycma and Nevergrad's recast CMA draw from numpy's global RNG (the
        # latter in a background thread, seeded from the clock unless the
        # hook below injects a seed).  Seed it from the run seed for the run
        # and hand the caller's state back afterwards.
        global_state = np.random.get_state()
        np.random.seed(_seed32(seed) ^ 0x5EED)
        adapter: Optional[AskTellAdapter] = None
        try:
            adapter = self.make_adapter(seed, log.max_eval, q)
            if self._virtual:
                self._drive_virtual(adapter, log, q, seed)
            else:
                self._drive(adapter, log, q)
        finally:
            if adapter is not None:
                adapter.close()
            np.random.set_state(global_state)

    def _drive_virtual(self, adapter: AskTellAdapter, log: _EvaluationLog, q: int, seed: int) -> None:
        """Drive the adapter on the virtual clock with ``q`` simulated workers.

        :func:`panobbgo.virtual_clock.run_ask_tell`: under the default async
        policy it asks for up to the free workers at every completion
        instant, evaluates at dispatch and tells at completion time, and
        feeds the tracker (``_virtual_observer``) in completion order, so
        the baseline gets ``aocc_time`` like a panobbgo strategy.  The
        duration stream is the one panobbgo strategies use
        (:data:`~panobbgo.virtual_clock.RNG_STREAM_KEY` of the run seed).
        The results frame stays in dispatch order.  A signalled timeout (a
        family's ``failure_at``) is not evaluated on this path, so it is not
        written to the results frame; the tracker counts it as a spent
        evaluation all the same.
        """
        from panobbgo.core import keyed_rng
        from panobbgo.virtual_clock import RNG_STREAM_KEY, _model_of, failure_mode, run_ask_tell

        cfg = self.config
        objective = _make_objective(self.problem, log)
        timeout = getattr(cfg, "evaluation_timeout", None)

        def ask(n: int) -> List[Tuple[int, np.ndarray]]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return adapter.ask(n)

        def tell(key: int, fx: float) -> None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adapter.tell(key, fx)

        run_ask_tell(
            ask,
            tell,
            objective,
            workers=q,
            model=_model_of(cfg),
            rng=keyed_rng(seed, RNG_STREAM_KEY),
            budget=log.max_eval,
            policy=getattr(cfg, "virtual_policy", "async"),
            timeout=float(timeout) if timeout else None,
            observer=self._virtual_observer,
            reraise=(_BudgetExhausted,),
            failure_at=lambda x: failure_mode(self.problem, self.problem.project(np.asarray(x, dtype=np.float64))),
        )

    def _drive(self, adapter: AskTellAdapter, log: _EvaluationLog, q: int) -> None:
        objective = _make_objective(self.problem, log)
        while len(log.fxs) < log.max_eval:
            n = min(q, log.max_eval - len(log.fxs))
            with warnings.catch_warnings():
                # Nevergrad warns on every clipped non-finite loss and on its
                # scipy sub-optimizers' settings; none of it is actionable here.
                warnings.simplefilter("ignore")
                batch = adapter.ask(n)
            if not batch:
                raise RuntimeError(f"{self.who}: ask() proposed nothing with no evaluation pending")
            fxs = [objective(x) for _key, x in batch]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for (key, _x), fx in zip(batch, fxs):
                    adapter.tell(key, fx)


# -- pycma ---------------------------------------------------------------------


class _PycmaRestartAdapter(AskTellAdapter):
    """IPOP- or BIPOP-CMA-ES over pycma's ``CMAEvolutionStrategy`` ask/tell.

    The restart schedule is that of ``cma.fmin2(..., restarts=, bipop=)``,
    re-implemented here because ``fmin2`` owns the evaluation loop and cannot
    be driven by batches:

    * base population ``popsize0 = 4 + 3 ln N`` kept as a float and
      truncated at each use (8, 17, 35, 70, ... at N = 5), raised to ``q``
      when ``q`` is larger so that one generation fills the workers;
    * IPOP doubles it per restart (``incpopsize=2``) and keeps the first
      run's ``maxiter``;
    * BIPOP interleaves small-population runs (``sigma0 * 0.01**U``,
      ``popsize0 * m**(U**2)``, an iteration cap of half the large runs'
      budget) while their budget is below the large runs'; the first run
      counts as small, as in pycma.

    The search runs in the unit cube (pycma ``bounds=[0, 1]``,
    ``sigma0 = 0.25``) mapped affinely onto the box; every run starts at a
    uniform random ``x0``.  All sampling uses the adapter's own generator
    (pycma's ``randn`` option), not numpy's global RNG.
    """

    requires = ("cma",)

    def __init__(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        seed: int,
        bipop: bool,
        batch_size: int = 1,
        sigma0: float = 0.25,
    ) -> None:
        self._cma = _require("cma")
        self._lo = np.asarray(lo, dtype=np.float64)
        self._span = np.asarray(hi, dtype=np.float64) - self._lo
        self._dim = int(self._lo.size)
        self._rng = np.random.default_rng(_seed32(seed))
        self._bipop = bool(bipop)
        self._sigma0 = float(sigma0)
        self._incpopsize = 2
        self._popsize0 = max(4.0 + 3.0 * math.log(self._dim), float(batch_size))
        # pycma's default ``100 + 150 * (N+3)**2 // popsize**0.5``, evaluated as
        # fmin2 does with the float base population (3330 at N=5, not 3494).
        self._maxiter0 = float(100 + 150 * (self._dim + 3) ** 2 // math.sqrt(self._popsize0))
        self._irun = 0
        self._runs_with_small = 0
        self._poptype = "small"
        self._evals_small = 0
        self._evals_large = 0
        self._run_evals = 0
        self._next_key = 0
        self._gen_x: List[np.ndarray] = []
        self._gen_keys: List[int] = []
        self._gen_f: Dict[int, float] = {}
        self._undispatched: List[int] = []
        self._es: Any = None
        self._start_run(self._popsize0, self._sigma0, self._maxiter0)

    @property
    def restarts(self) -> int:
        """Number of restarts so far."""
        return self._irun

    @property
    def popsize(self) -> int:
        """Population size of the current run."""
        return int(self._es.popsize)

    def _randn(self, *shape: int) -> np.ndarray:
        return self._rng.standard_normal(shape)

    def _start_run(self, popsize: float, sigma: float, maxiter: Optional[float]) -> None:
        opts: Dict[str, Any] = {
            "bounds": [0.0, 1.0],
            "popsize": max(2, int(popsize)),
            "randn": self._randn,
            # NaN: pycma neither seeds nor warns when ``randn`` is its own.
            "seed": float("nan"),
            "verbose": -9,
            "verb_log": 0,
            "verb_disp": 0,
        }
        if maxiter is not None:
            opts["maxiter"] = maxiter
        x0 = self._rng.uniform(0.0, 1.0, size=self._dim)
        self._es = self._cma.CMAEvolutionStrategy(x0, sigma, opts)
        self._run_evals = 0

    def _restart(self) -> None:
        if self._poptype == "small":
            self._evals_small += self._run_evals
        else:
            self._evals_large += self._run_evals
        self._irun += 1
        if not self._bipop:
            self._start_run(self._popsize0 * self._incpopsize**self._irun, self._sigma0, self._maxiter0)
        elif self._evals_small < max(1, self._evals_large):
            self._poptype = "small"
            self._runs_with_small += 1
            sigma_factor = 0.01 ** self._rng.uniform()
            multiplier = self._incpopsize ** (self._irun - self._runs_with_small)
            popsize = int(self._popsize0 * multiplier ** (self._rng.uniform() ** 2))
            maxiter = min(self._maxiter0, 0.5 * self._evals_large / max(2, popsize))
            self._start_run(popsize, self._sigma0 * sigma_factor, maxiter)
        else:
            self._poptype = "large"
            multiplier = self._incpopsize ** (self._irun - self._runs_with_small)
            self._start_run(self._popsize0 * multiplier, self._sigma0, self._maxiter0)

    def _new_generation(self) -> None:
        if self._es.stop():
            self._restart()
        self._gen_x = [np.asarray(u, dtype=np.float64) for u in self._es.ask()]
        self._gen_keys = list(range(self._next_key, self._next_key + len(self._gen_x)))
        self._next_key += len(self._gen_x)
        self._gen_f = {}
        self._undispatched = list(self._gen_keys)

    def ask(self, n: int) -> List[Tuple[int, np.ndarray]]:
        out: List[Tuple[int, np.ndarray]] = []
        while len(out) < n:
            if not self._undispatched:
                if self._gen_keys:
                    break  # the current generation still waits on results
                self._new_generation()
            key = self._undispatched.pop(0)
            u = self._gen_x[key - self._gen_keys[0]]
            out.append((key, self._lo + np.clip(u, 0.0, 1.0) * self._span))
        return out

    def tell(self, key: int, fx: float) -> None:
        if not self._gen_keys or not (self._gen_keys[0] <= key <= self._gen_keys[-1]):
            raise KeyError(f"candidate {key} is not in the current generation")
        self._gen_f[key] = _nan_is_worst(fx)
        if len(self._gen_f) == len(self._gen_keys):
            self._es.tell(self._gen_x, [self._gen_f[k] for k in self._gen_keys])
            self._run_evals += len(self._gen_keys)
            self._gen_keys = []
            self._gen_x = []


class PycmaIPOPStrategy(AskTellBaselineStrategy):
    """pycma CMA-ES with IPOP restarts (population doubles on each restart)."""

    who: str = "pycma_IPOP"
    requires = _PycmaRestartAdapter.requires
    _bipop: bool = False

    def make_adapter(self, seed: int, budget: int, batch_size: int) -> AskTellAdapter:
        del budget
        box = self.problem.box
        return _PycmaRestartAdapter(box[:, 0], box[:, 1], seed, bipop=self._bipop, batch_size=batch_size)


class PycmaBIPOPStrategy(PycmaIPOPStrategy):
    """pycma CMA-ES with BIPOP restarts (IPOP interleaved with small local runs)."""

    who: str = "pycma_BIPOP"
    _bipop: bool = True


# -- Nevergrad -------------------------------------------------------------------


def _size1_float(value: Any) -> float:
    """``float()`` that also takes a size-1 array, as numpy < 2.5 did."""
    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.reshape(-1)[0])
    return float(value)


def _patch_nevergrad_metamodel() -> None:
    """Make Nevergrad 1.0.12's meta-model work under numpy >= 2.5.

    ``nevergrad.optimization.metamodel.learn_on_k_best`` calls
    ``float(model.predict(x))`` on a shape-``(1,)`` array, which numpy 2.5
    turned from a deprecation warning into a ``TypeError``.  NGOpt picks
    ``MetaModel`` in low dimension with parallel workers, so without this
    those runs crash.  Shadowing ``float`` in that one module with a
    version that accepts size-1 arrays restores the behaviour the library
    was written against; drop it once a Nevergrad release fixes the call.
    """
    metamodel = _require("nevergrad.optimization.metamodel")
    if getattr(metamodel, "float", None) is not _size1_float:
        setattr(metamodel, "float", _size1_float)


#: Seed source for ``cma.fmin`` calls made by Nevergrad while a Nevergrad
#: adapter is live (``None`` otherwise).  See :func:`_hook_cma_fmin_seed`.
#: Owned by the adapter that set it (identity check on release), so an
#: adapter can only unset its own generator.  One process-wide slot:
#: determinism holds for one live Nevergrad adapter per process at a time
#: (the harnesses run one run per process or thread at a time).
_CMA_FMIN_SEEDS: Optional[np.random.Generator] = None


def _hook_cma_fmin_seed() -> None:
    """Make the ``cma.fmin`` calls of Nevergrad's recast CMA reproducible.

    NGOpt's ``MetaModel(CmaFmin2)`` (e.g. dim 5, budget 300) runs
    ``cma.fmin`` in a background thread (``nevergrad/optimization/
    recastlib.py``) without a ``seed`` option, so pycma seeds numpy's global
    RNG from the clock and samples from it.  Seeding alone is not enough:
    the thread resumes after each ``tell`` while the main thread may draw
    from the same global RNG (the meta-model's surrogate search creates
    parametrizations whose ``random_state`` comes from it), so the two race.
    The wrapper therefore gives each such call its own generator through
    pycma's ``randn`` option (with ``seed=nan``: no global seeding), drawn
    from :data:`_CMA_FMIN_SEEDS`.

    recastlib imports ``cma`` inside the function and calls ``cma.fmin``, so
    the only place to intercept it is the module attribute.  The wrapper
    acts only while a Nevergrad adapter has set :data:`_CMA_FMIN_SEEDS` and
    only when the caller passed ``options`` without ``seed``/``randn``;
    every other ``cma.fmin`` call passes through unchanged.
    """
    cma = _require("cma")
    if getattr(cma.fmin, "_panobbgo_seed_hook", False):
        return
    original = cma.fmin

    @functools.wraps(original)
    def fmin(*args: Any, **kwargs: Any) -> Any:
        seeds = _CMA_FMIN_SEEDS
        options = kwargs.get("options")
        if seeds is not None and isinstance(options, dict) and not ({"seed", "randn"} & set(options)):
            rng = np.random.default_rng(seeds.integers(0, 2**32))

            def randn(*shape: int) -> np.ndarray:
                return rng.standard_normal(shape)

            kwargs["options"] = {**options, "seed": float("nan"), "randn": randn}
        return original(*args, **kwargs)

    setattr(fmin, "_panobbgo_seed_hook", True)
    cma.fmin = fmin


class _NevergradAdapter(AskTellAdapter):
    """One Nevergrad optimizer from ``ng.optimizers.registry``.

    The parametrization is ``ng.p.Array(init=x0, lower=lo, upper=hi)`` with
    ``x0`` uniform in the box: Nevergrad sets ``sigma = (hi - lo) / 6`` per
    coordinate and keeps candidates inside the box by bouncing.  ``budget``
    and ``num_workers = q`` go to the constructor; NGOpt chooses its
    sub-optimizer from them and the dimension.  Seeded through the
    parametrization's ``random_state`` and the ``cma.fmin`` hook.
    """

    requires = ("nevergrad", "cma")

    def __init__(self, name: str, lo: np.ndarray, hi: np.ndarray, seed: int, budget: int, num_workers: int) -> None:
        global _CMA_FMIN_SEEDS
        ng = _require("nevergrad")
        _patch_nevergrad_metamodel()
        _hook_cma_fmin_seed()
        rng = np.random.default_rng(_seed32(seed))
        self._fmin_seeds = np.random.default_rng(rng.integers(0, 2**32))
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        param = ng.p.Array(init=rng.uniform(lo, hi), lower=lo, upper=hi)
        param.random_state = np.random.RandomState(_seed32(seed))
        self._optimizer_base = ng.optimizers.base.Optimizer
        self._opt = ng.optimizers.registry[name](
            parametrization=param, budget=int(budget), num_workers=int(num_workers)
        )
        self._pending: Dict[int, Any] = {}
        self._next_key = 0
        # Last, once nothing above can raise: a failed constructor never
        # takes the slot, so it has nothing to release.  No ``cma.fmin`` runs
        # before the first ``ask`` (the recast thread starts there).
        _CMA_FMIN_SEEDS = self._fmin_seeds

    def ask(self, n: int) -> List[Tuple[int, np.ndarray]]:
        out: List[Tuple[int, np.ndarray]] = []
        for _ in range(n):
            cand = self._opt.ask()
            key = self._next_key
            self._next_key += 1
            self._pending[key] = cand
            out.append((key, np.asarray(cand.value, dtype=np.float64).copy()))
        return out

    def tell(self, key: int, fx: float) -> None:
        self._opt.tell(self._pending.pop(key), _nan_is_worst(fx))

    def _optimizers(self) -> List[Any]:
        """Every optimizer in the tree under ``self._opt``.

        Walks instance attributes only (``vars()``), never properties: NGOpt's
        ``optim`` property would *create* a sub-optimizer.  Finds
        ``NGOpt._optim``, ``_MetaModel._optim``, ``_Chain.optimizers``,
        ``Portfolio.optims`` and the like.
        """
        found: List[Any] = []
        stack: List[Any] = [self._opt]
        seen: set = set()
        while stack:
            opt = stack.pop()
            if id(opt) in seen:
                continue
            seen.add(id(opt))
            found.append(opt)
            for value in vars(opt).values():
                items = value if isinstance(value, (list, tuple)) else [value]
                stack.extend(v for v in items if isinstance(v, self._optimizer_base))
        return found

    def close(self) -> None:
        """Stop the worker threads of "recast" sub-optimizers.

        NGOpt may pick a SciPy method (Cobyla) or ``CmaFmin2``, which
        Nevergrad runs in a non-daemon thread blocked on the next ``tell``.
        Nevergrad stops them only when the optimizer is garbage-collected,
        which a traceback holding the run's frames can postpone until
        interpreter exit — a hang.
        """
        global _CMA_FMIN_SEEDS
        for opt in self._optimizers():
            thread = vars(opt).get("_messaging_thread")
            if thread is not None:
                thread.stop()
        if _CMA_FMIN_SEEDS is self._fmin_seeds:
            _CMA_FMIN_SEEDS = None


class NevergradStrategy(AskTellBaselineStrategy):
    """A Nevergrad optimizer, named by :attr:`optimizer_name` (set by subclasses)."""

    who: str = "Nevergrad"
    optimizer_name: str = ""
    requires = _NevergradAdapter.requires

    def make_adapter(self, seed: int, budget: int, batch_size: int) -> AskTellAdapter:
        if not self.optimizer_name:
            raise TypeError("NevergradStrategy is abstract: subclass it and set optimizer_name")
        box = self.problem.box
        return _NevergradAdapter(self.optimizer_name, box[:, 0], box[:, 1], seed, budget, batch_size)


class NevergradNGOptStrategy(NevergradStrategy):
    """Nevergrad ``NGOpt``: a hand-ruled selector over a portfolio."""

    who: str = "NGOpt"
    optimizer_name: str = "NGOpt"


class NevergradCMAStrategy(NevergradStrategy):
    """Nevergrad ``CMA`` (cross-check against pycma)."""

    who: str = "NG_CMA"
    optimizer_name: str = "CMA"


class NevergradTwoPointsDEStrategy(NevergradStrategy):
    """Nevergrad ``TwoPointsDE`` (cross-check against SciPy DE)."""

    who: str = "NG_TwoPointsDE"
    optimizer_name: str = "TwoPointsDE"


# -- Optuna ----------------------------------------------------------------------


_OPTUNA_QUIETED = False


def _quiet_optuna(optuna: Any) -> None:
    """Lower Optuna's log level to WARNING, once per process.

    ``optuna.logging.set_verbosity`` is process-wide; it is set on the first
    Optuna baseline run (without it every trial logs an INFO line) and not
    touched again, so a caller that changes it afterwards keeps its setting.
    """
    global _OPTUNA_QUIETED
    if not _OPTUNA_QUIETED:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        _OPTUNA_QUIETED = True


class _OptunaAdapter(AskTellAdapter):
    """An in-memory Optuna study with one ``FloatDistribution`` per coordinate.

    Keys are trial numbers.  Several asked-but-untold trials are pending
    trials to Optuna (TPE's default ``constant_liar=True`` accounts for
    them).  Per-trial cost grows with the number of trials, so wall time is
    roughly quadratic in the budget (TPE: ~21 s for 2000 evaluations at
    dim 10).  The composite harness' default 120 s per-run timeout cuts such
    runs short: measure Optuna baselines with ``--no-timeout``.
    """

    def __init__(self, sampler: str, lo: np.ndarray, hi: np.ndarray, seed: int) -> None:
        optuna = _require("optuna")
        _quiet_optuna(optuna)
        names = [f"x{j}" for j in range(len(lo))]
        if sampler == "cmaes":
            _require("cmaes")  # CmaEsSampler's backend, not an Optuna dependency
            rng = np.random.default_rng(_seed32(seed))
            x0 = {k: float(v) for k, v in zip(names, rng.uniform(lo, hi))}
            with warnings.catch_warnings():
                # ``x0`` is deprecated since Optuna 4.9 (removal planned for
                # 6.0); without it the sampler starts at the box centre.
                warnings.simplefilter("ignore", FutureWarning)
                smp = optuna.samplers.CmaEsSampler(x0=x0, seed=_seed32(seed), warn_independent_sampling=False)
        elif sampler == "tpe":
            smp = optuna.samplers.TPESampler(seed=_seed32(seed))
        else:
            raise ValueError(f"unknown Optuna sampler {sampler!r}")
        self._study = optuna.create_study(direction="minimize", sampler=smp)
        self._dists = {
            k: optuna.distributions.FloatDistribution(float(lo_j), float(hi_j)) for k, lo_j, hi_j in zip(names, lo, hi)
        }

    def ask(self, n: int) -> List[Tuple[int, np.ndarray]]:
        out: List[Tuple[int, np.ndarray]] = []
        for _ in range(n):
            trial = self._study.ask(self._dists)
            out.append((trial.number, np.array([trial.params[k] for k in self._dists], dtype=np.float64)))
        return out

    def tell(self, key: int, fx: float) -> None:
        # A COMPLETE trial even for +inf: a FAIL would hide the region from TPE.
        self._study.tell(key, _nan_is_worst(fx))


class OptunaCmaEsStrategy(AskTellBaselineStrategy):
    """Optuna ``CmaEsSampler``, default settings, **no restarts**.

    ``CmaEsSampler(restart_strategy=...)`` is deprecated since Optuna 4.4,
    so a converged run spends the rest of its budget near its optimum; the
    restart variants are the pycma baselines.  Starts from a uniform random
    ``x0``.  See :class:`_OptunaAdapter` on wall time.
    """

    who: str = "Optuna_CmaEs"
    requires: Tuple[str, ...] = ("optuna", "cmaes")
    _sampler: str = "cmaes"

    def make_adapter(self, seed: int, budget: int, batch_size: int) -> AskTellAdapter:
        del budget, batch_size
        box = self.problem.box
        return _OptunaAdapter(self._sampler, box[:, 0], box[:, 1], seed)


class OptunaTPEStrategy(OptunaCmaEsStrategy):
    """Optuna ``TPESampler``, default settings.

    Wall time is roughly quadratic in the budget; run with ``--no-timeout``
    (see :class:`_OptunaAdapter`).
    """

    who: str = "Optuna_TPE"
    requires: Tuple[str, ...] = ("optuna",)
    _sampler: str = "tpe"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


#: The external baseline classes (the ``baselines`` extra), in registry order.
_EXTERNAL_BASELINE_CLASSES: Tuple[type, ...] = (
    PycmaIPOPStrategy,
    PycmaBIPOPStrategy,
    NevergradNGOptStrategy,
    NevergradCMAStrategy,
    NevergradTwoPointsDEStrategy,
    OptunaCmaEsStrategy,
    OptunaTPEStrategy,
)

#: Spec names of the expensive-track baselines (the ``baselines-bo`` extra,
#: :mod:`panobbgo.harness_baselines_bo`), in registry order.  Spelled out
#: here because that module imports this one; a test checks them against
#: its classes.
BO_BASELINE_NAMES: Tuple[str, ...] = (
    "Baseline_BoTorch_qLogEI",
    "Baseline_TuRBO1",
    "Baseline_SMAC_BB",
    "Baseline_PyBOBYQA",
)

#: Spec names of the cheap-track external baselines (the ``baselines``
#: extra), in registry order.  Not part of the default ``--baselines`` set:
#: name them in the harness' strategy filter (``--baselines --strategies
#: Baseline_NGOpt``) to select them.  Iterating over it never needs torch;
#: the expensive track is :data:`BO_BASELINE_NAMES`.
EXTERNAL_BASELINE_NAMES: Tuple[str, ...] = tuple(f"Baseline_{cls.who}" for cls in _EXTERNAL_BASELINE_CLASSES)

#: Every opt-in external baseline: :data:`EXTERNAL_BASELINE_NAMES`, then
#: :data:`BO_BASELINE_NAMES` — the order of
#: :func:`make_external_baseline_strategies`.
ALL_EXTERNAL_BASELINE_NAMES: Tuple[str, ...] = EXTERNAL_BASELINE_NAMES + BO_BASELINE_NAMES

#: Spec names of the default ``--baselines`` set.
DEFAULT_BASELINE_NAMES: Tuple[str, ...] = ("Baseline_Random", "Baseline_SciPyDE", "Baseline_SciPyAnneal")


def make_external_baseline_strategies() -> List[StrategySpec]:
    """Return the :class:`StrategySpec` list for every opt-in external baseline.

    Both tracks, in :data:`ALL_EXTERNAL_BASELINE_NAMES` order; each class
    names its extra (``extra`` attribute).  Needs no optional import;
    :func:`make_baseline_strategies` checks the extra for the ones a run
    selects.
    """
    from panobbgo.harness_baselines_bo import BO_BASELINE_CLASSES

    return [
        StrategySpec(name=name, strategy_class=cls, heuristics=[])
        for name, cls in zip(ALL_EXTERNAL_BASELINE_NAMES, _EXTERNAL_BASELINE_CLASSES + BO_BASELINE_CLASSES)
    ]


def check_baseline_selection(names: Optional[Iterable[str]], include_baselines: bool) -> None:
    """Raise :class:`ValueError` when ``names`` asks for a baseline without ``--baselines``.

    The name filter of both harness CLIs only sees the baselines when
    ``--baselines`` is given; without it a baseline name would be reported
    as unknown, with no hint why.
    """
    if include_baselines or not names:
        return
    asked = sorted(set(names) & set(DEFAULT_BASELINE_NAMES + ALL_EXTERNAL_BASELINE_NAMES))
    if asked:
        raise ValueError(
            f"{asked} are baseline strategies: add --baselines (HarnessConfig(include_baselines=True)) to select them"
        )


def make_baseline_strategies(extra: Optional[Iterable[str]] = None) -> List[StrategySpec]:
    """Return the :class:`StrategySpec` list of the baseline solvers.

    These are plugged into the harness when a run is requested with
    baselines enabled.  All are budget-agnostic — they inherit
    ``config.max_eval`` from the harness — so they can be dropped into any
    of the ``quick`` / ``standard`` / ``full`` modes.

    Args:
        extra: Strategy names (e.g. the harness' ``--strategies`` filter).
            The external baselines named in it (see
            :data:`ALL_EXTERNAL_BASELINE_NAMES`) are appended; other names are
            ignored.  ``None`` returns the default set: Random, SciPy DE,
            SciPy dual annealing.

    Raises:
        ImportError: A named external baseline needs a module that is not
            installed (the ``baselines`` or ``baselines-bo`` extra) — raised
            here, before any run, rather than as a per-run error that
            scores 0.
    """
    wanted = set(extra or ())
    external = [spec for spec in make_external_baseline_strategies() if spec.name in wanted]
    for spec in external:
        missing = [m for m in getattr(spec.strategy_class, "requires", ()) if importlib.util.find_spec(m) is None]
        if missing:
            hint = _extra_hint(getattr(spec.strategy_class, "extra", "baselines"))
            raise ImportError(f"{spec.name} needs {', '.join(missing)}; {hint}")
    return [
        StrategySpec(
            name="Baseline_Random",
            strategy_class=RandomSearchStrategy,
            heuristics=[],
        ),
        StrategySpec(
            name="Baseline_SciPyDE",
            strategy_class=SciPyDEStrategy,
            heuristics=[],
        ),
        StrategySpec(
            name="Baseline_SciPyAnneal",
            strategy_class=SciPyAnnealStrategy,
            heuristics=[],
        ),
    ] + external


__all__ = [
    "ALL_EXTERNAL_BASELINE_NAMES",
    "BO_BASELINE_NAMES",
    "DEFAULT_BASELINE_NAMES",
    "EXTERNAL_BASELINE_NAMES",
    "AskTellAdapter",
    "AskTellBaselineStrategy",
    "BaselineStrategy",
    "NevergradCMAStrategy",
    "NevergradNGOptStrategy",
    "NevergradStrategy",
    "NevergradTwoPointsDEStrategy",
    "OptunaCmaEsStrategy",
    "OptunaTPEStrategy",
    "PycmaBIPOPStrategy",
    "PycmaIPOPStrategy",
    "RandomSearchStrategy",
    "SciPyDEStrategy",
    "SciPyAnnealStrategy",
    "check_baseline_selection",
    "make_baseline_strategies",
    "make_external_baseline_strategies",
]
