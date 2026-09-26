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
  CMA-ES with IPOP / BIPOP restarts (the restart schedule of ``cma.fmin``
  re-implemented over ask/tell).
- :class:`NevergradNGOptStrategy` — Nevergrad's ``NGOpt`` meta-optimizer,
  plus :class:`NevergradCMAStrategy` and :class:`NevergradTwoPointsDEStrategy`
  as cross-checks.
- :class:`OptunaCmaEsStrategy`, :class:`OptunaTPEStrategy` — Optuna's
  ``CmaEsSampler`` and ``TPESampler``.

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
  dispatch order.  A virtual-clock parallel simulator can drive the same
  adapter asynchronously via :meth:`AskTellBaselineStrategy.make_adapter`.

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

from dataclasses import dataclass, field
import math
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from panobbgo.benchmark import StrategySpec
from panobbgo.ioh_runner import _BudgetExhausted
from panobbgo.lib import Point, Problem, Result


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


def _make_objective(problem: Problem, log: _EvaluationLog) -> Callable[[np.ndarray], float]:
    """Return a scalar objective that projects into the box, evaluates the
    problem, logs the result, and enforces the evaluation budget."""

    def objective(x: np.ndarray) -> float:
        if log.stop_requested:
            raise _BudgetExhausted()
        x_arr = np.asarray(x, dtype=np.float64)
        # Respect the problem's box — scipy solvers already pass candidates
        # inside bounds, but project defensively to cover edge cases with
        # solvers that occasionally probe slightly outside (``basinhopping``
        # has historically done this at tolerance boundaries).
        x_proj = problem.project(x_arr)
        fx = float(problem.eval(x_proj))
        log.record(x_proj, fx)
        return fx

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

        objective = _make_objective(self.problem, log)
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

        objective = _make_objective(self.problem, log)
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

_EXTRA_HINT = "install the optional extra: `uv sync --extra baselines` (or `pip install 'panobbgo[baselines]'`)"


def _require(module: str) -> Any:
    """Import ``module`` or fail with a hint at the ``baselines`` extra."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"{module!r} is not installed; {_EXTRA_HINT}") from exc


def _seed32(seed: int) -> int:
    """Fold a run seed into ``[0, 2**32)``, the range every library accepts."""
    return int(seed) % (2**32)


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
      order.  Non-finite values are allowed; each adapter maps them to what
      its library accepts.
    """

    def ask(self, n: int) -> List[Tuple[int, np.ndarray]]:
        """Return up to ``n`` new ``(key, x)`` candidates."""
        raise NotImplementedError

    def tell(self, key: int, fx: float) -> None:
        """Report the objective value of the candidate ``key``."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources (threads) the library holds; the default does nothing."""


class AskTellBaselineStrategy(BaselineStrategy):
    """Baseline driven through an :class:`AskTellAdapter` with batch size ``q``.

    ``q`` is ``config.batch_size`` (default 1; the ``batch_size`` argument or
    ``StrategySpec(config_overrides={"batch_size": q})`` set it).  Each step
    asks for ``min(q, remaining budget)`` candidates, evaluates them in
    dispatch order and tells them, so the results frame is in dispatch
    order and the budget cap is exact.  ``q`` is also passed to the adapter
    (Nevergrad's ``num_workers``), since some libraries choose their
    algorithm by it.
    """

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

    def _optimize(self, log: _EvaluationLog) -> None:
        q = max(1, int(self.config.batch_size))
        adapter = self.make_adapter(self._run_seed(), log.max_eval, q)
        try:
            self._drive(adapter, log, q)
        finally:
            adapter.close()

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

    The restart schedule is that of ``cma.fmin2(..., restarts=, bipop=)``
    (``incpopsize=2``; BIPOP interleaves small-population runs with a random
    ``sigma0`` factor ``0.01**U`` and an iteration cap while their budget
    is below the large runs'), re-implemented here because ``fmin2`` owns
    the evaluation loop and cannot be driven by batches.  The search runs in
    the unit cube (pycma ``bounds=[0, 1]``, ``sigma0 = 0.25``) mapped
    affinely onto the box; every run starts at a uniform random ``x0``.
    """

    def __init__(self, lo: np.ndarray, hi: np.ndarray, seed: int, bipop: bool, sigma0: float = 0.25) -> None:
        self._cma = _require("cma")
        self._lo = np.asarray(lo, dtype=np.float64)
        self._span = np.asarray(hi, dtype=np.float64) - self._lo
        self._dim = int(self._lo.size)
        self._rng = np.random.default_rng(_seed32(seed))
        self._bipop = bool(bipop)
        self._sigma0 = float(sigma0)
        self._incpopsize = 2
        self._popsize0 = 4 + int(3 * math.log(self._dim))
        self._maxiter0: Optional[float] = None
        self._irun = 0
        self._runs_with_small = 0
        # pycma counts the first run as "small" (a Matlab carry-over it keeps).
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
        self._start_run(self._popsize0, self._sigma0, None)

    @property
    def restarts(self) -> int:
        """Number of restarts so far."""
        return self._irun

    def _start_run(self, popsize: int, sigma: float, maxiter: Optional[float]) -> None:
        opts: Dict[str, Any] = {
            "bounds": [0.0, 1.0],
            "popsize": int(max(2, popsize)),
            # pycma seeds numpy's global RNG from this at construction; it is
            # drawn from the run's own generator, so the run is deterministic.
            "seed": int(self._rng.integers(1, 2**31 - 1)),
            "verbose": -9,
            "verb_log": 0,
            "verb_disp": 0,
        }
        if maxiter is not None:
            opts["maxiter"] = maxiter
        x0 = self._rng.uniform(0.0, 1.0, size=self._dim)
        self._es = self._cma.CMAEvolutionStrategy(x0, sigma, opts)
        if self._maxiter0 is None:
            self._maxiter0 = float(self._es.opts["maxiter"])
        self._run_evals = 0

    def _restart(self) -> None:
        if self._poptype == "small":
            self._evals_small += self._run_evals
        else:
            self._evals_large += self._run_evals
        self._irun += 1
        assert self._maxiter0 is not None
        if not self._bipop:
            self._start_run(self._popsize0 * self._incpopsize**self._irun, self._sigma0, None)
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
        self._gen_f[key] = float(fx) if np.isfinite(fx) else float("inf")
        if len(self._gen_f) == len(self._gen_keys):
            self._es.tell(self._gen_x, [self._gen_f[k] for k in self._gen_keys])
            self._run_evals += len(self._gen_keys)
            self._gen_keys = []
            self._gen_x = []


class PycmaIPOPStrategy(AskTellBaselineStrategy):
    """pycma CMA-ES with IPOP restarts (population doubles on each restart)."""

    who: str = "pycma_IPOP"
    _bipop: bool = False

    def make_adapter(self, seed: int, budget: int, batch_size: int) -> AskTellAdapter:
        del budget, batch_size
        box = self.problem.box
        return _PycmaRestartAdapter(box[:, 0], box[:, 1], seed, bipop=self._bipop)


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


class _NevergradAdapter(AskTellAdapter):
    """One Nevergrad optimizer from ``ng.optimizers.registry``.

    The parametrization is ``ng.p.Array(init=centre, lower=lo, upper=hi)``:
    Nevergrad then sets ``sigma = (hi - lo) / 6`` per coordinate and keeps
    candidates inside the box by bouncing.  ``budget`` and ``num_workers =
    q`` go to the constructor; NGOpt chooses its sub-optimizer from them
    and the dimension.  Seeded through the parametrization's
    ``random_state``.
    """

    def __init__(self, name: str, lo: np.ndarray, hi: np.ndarray, seed: int, budget: int, num_workers: int) -> None:
        ng = _require("nevergrad")
        _patch_nevergrad_metamodel()
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        param = ng.p.Array(init=(lo + hi) / 2.0, lower=lo, upper=hi)
        param.random_state = np.random.RandomState(_seed32(seed))
        self._opt = ng.optimizers.registry[name](
            parametrization=param, budget=int(budget), num_workers=int(num_workers)
        )
        self._pending: Dict[int, Any] = {}
        self._next_key = 0

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
        # Nevergrad clips non-finite losses itself (with a warning).
        self._opt.tell(self._pending.pop(key), float(fx) if np.isfinite(fx) else float("inf"))

    def close(self) -> None:
        """Stop the worker threads of "recast" sub-optimizers (NGOpt may pick
        a SciPy method such as Cobyla, which Nevergrad runs in a non-daemon
        thread blocked on the next ``tell``).  Nevergrad stops them only when
        the optimizer is garbage-collected, which a traceback holding the
        run's frames can postpone until interpreter exit — a hang."""
        stack: List[Any] = [self._opt]
        seen: set = set()
        while stack:
            opt = stack.pop()
            if opt is None or id(opt) in seen:
                continue
            seen.add(id(opt))
            thread = getattr(opt, "_messaging_thread", None)
            if thread is not None:
                thread.stop()
            stack.append(getattr(opt, "optim", None))
            stack.extend(getattr(opt, "optims", None) or [])


class NevergradStrategy(AskTellBaselineStrategy):
    """A Nevergrad optimizer, named by :attr:`optimizer_name`."""

    who: str = "NGOpt"
    optimizer_name: str = "NGOpt"

    def make_adapter(self, seed: int, budget: int, batch_size: int) -> AskTellAdapter:
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


class _OptunaAdapter(AskTellAdapter):
    """An in-memory Optuna study with one ``FloatDistribution`` per coordinate.

    Keys are trial numbers; a non-finite value is told as a failed trial.
    Several asked-but-untold trials are pending trials to Optuna (TPE
    handles them with its constant liar).
    """

    def __init__(self, sampler: str, lo: np.ndarray, hi: np.ndarray, seed: int) -> None:
        optuna = _require("optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if sampler == "cmaes":
            _require("cmaes")  # CmaEsSampler's backend, not an Optuna dependency
            smp = optuna.samplers.CmaEsSampler(seed=_seed32(seed), warn_independent_sampling=False)
        elif sampler == "tpe":
            smp = optuna.samplers.TPESampler(seed=_seed32(seed))
        else:
            raise ValueError(f"unknown Optuna sampler {sampler!r}")
        self._state_fail = optuna.trial.TrialState.FAIL
        self._study = optuna.create_study(direction="minimize", sampler=smp)
        self._dists = {
            f"x{j}": optuna.distributions.FloatDistribution(float(lo_j), float(hi_j))
            for j, (lo_j, hi_j) in enumerate(zip(lo, hi))
        }

    def ask(self, n: int) -> List[Tuple[int, np.ndarray]]:
        out: List[Tuple[int, np.ndarray]] = []
        for _ in range(n):
            trial = self._study.ask(self._dists)
            out.append((trial.number, np.array([trial.params[k] for k in self._dists], dtype=np.float64)))
        return out

    def tell(self, key: int, fx: float) -> None:
        if np.isfinite(fx):
            self._study.tell(key, float(fx))
        else:
            self._study.tell(key, state=self._state_fail)


class OptunaCmaEsStrategy(AskTellBaselineStrategy):
    """Optuna ``CmaEsSampler`` (default settings: no restarts)."""

    who: str = "Optuna_CmaEs"
    _sampler: str = "cmaes"

    def make_adapter(self, seed: int, budget: int, batch_size: int) -> AskTellAdapter:
        del budget, batch_size
        box = self.problem.box
        return _OptunaAdapter(self._sampler, box[:, 0], box[:, 1], seed)


class OptunaTPEStrategy(OptunaCmaEsStrategy):
    """Optuna ``TPESampler`` (default settings)."""

    who: str = "Optuna_TPE"
    _sampler: str = "tpe"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


#: Spec names of the external baselines (the ``baselines`` extra), in
#: registry order.  Not part of the default ``--baselines`` set: name them in
#: the harness' strategy filter to select them.
EXTERNAL_BASELINE_NAMES: Tuple[str, ...] = (
    "Baseline_pycma_IPOP",
    "Baseline_pycma_BIPOP",
    "Baseline_NGOpt",
    "Baseline_NG_CMA",
    "Baseline_NG_TwoPointsDE",
    "Baseline_Optuna_CmaEs",
    "Baseline_Optuna_TPE",
)


def make_external_baseline_strategies() -> List[StrategySpec]:
    """Return the :class:`StrategySpec` list for the external baselines.

    Building the specs needs no optional import; running one without the
    ``baselines`` extra raises an :class:`ImportError` naming the extra.
    """
    classes: List[type] = [
        PycmaIPOPStrategy,
        PycmaBIPOPStrategy,
        NevergradNGOptStrategy,
        NevergradCMAStrategy,
        NevergradTwoPointsDEStrategy,
        OptunaCmaEsStrategy,
        OptunaTPEStrategy,
    ]
    specs = [StrategySpec(name=f"Baseline_{cls.who}", strategy_class=cls, heuristics=[]) for cls in classes]
    assert tuple(s.name for s in specs) == EXTERNAL_BASELINE_NAMES
    return specs


def make_baseline_strategies(extra: Optional[Iterable[str]] = None) -> List[StrategySpec]:
    """Return the :class:`StrategySpec` list of the baseline solvers.

    These are plugged into the harness when a run is requested with
    baselines enabled.  All are budget-agnostic — they inherit
    ``config.max_eval`` from the harness — so they can be dropped into any
    of the ``quick`` / ``standard`` / ``full`` modes.

    Args:
        extra: Strategy names (e.g. the harness' ``--strategies`` filter).
            The external baselines named in it (see
            :data:`EXTERNAL_BASELINE_NAMES`) are appended; other names are
            ignored.  ``None`` returns the default set: Random, SciPy DE,
            SciPy dual annealing.
    """
    wanted = set(extra or ())
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
    ] + [spec for spec in make_external_baseline_strategies() if spec.name in wanted]


__all__ = [
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
    "make_baseline_strategies",
    "make_external_baseline_strategies",
]
