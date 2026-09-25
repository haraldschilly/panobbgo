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

r"""
Core
====

This is the core part. It contains the essential components
and base-classes for the modules:

- :class:`.Results`: Database of all results, with some rudimentary queries and statistics.
- :class:`.EventBus`: This is the backbone for communicating between the strategy,
  the heuristics and the analyzers.
- "abstract" base-classes for the modules

  - :mod:`.heuristics`
  - :mod:`.analyzers`.

- and most importantly, the :class:`.StrategyBase` which holds everything together and
  subclasses in :mod:`.strategies` implement the actual strategies.

.. inheritance-diagram:: panobbgo.core

.. codeauthor:: Harald Schilly <harald.schilly@gmail.com>
"""

from .config import Config
from panobbgo.lib import Result, Point
from panobbgo.lib.constraints import (
    DefaultConstraintHandler,
    PenaltyConstraintHandler,
    DynamicPenaltyConstraintHandler,
    AugmentedLagrangianConstraintHandler,
    EpsilonConstraintHandler,
    FilterConstraintHandler,
)
from .logging import PanobbgoLogger
import time as time_module
import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex, concat
import inspect
import logging
import uuid
from queue import Empty, Queue
from threading import Condition, RLock, Thread
import multiprocessing
import collections
import heapq
import zlib
import re
import threading
from .logging.progress import ProgressContext
from typing import TYPE_CHECKING, Any, Callable, cast, Optional, List, Dict, Union, Tuple

if TYPE_CHECKING:
    from .lib import Problem, Result
    from .core import EventBus


class _RankTracker:
    """Running minimum and ``int(frac * n)``-th smallest value of a stream.

    Two heaps: ``_low`` (a max-heap, negated) holds the ``int(frac * n) + 1``
    smallest values, ``_high`` the rest, so :meth:`threshold` is ``O(1)`` and
    :meth:`push` ``O(log n)``.  NaNs are ignored.
    """

    def __init__(self, frac: float = 0.1) -> None:
        self.frac = frac
        self.n = 0
        self.min = float("inf")
        self._low: List[float] = []
        self._high: List[float] = []

    def push(self, v: float) -> None:
        if v != v:  # NaN
            return
        self.n += 1
        if v < self.min:
            self.min = v
        if self._low and v < -self._low[0]:
            heapq.heappush(self._low, -v)
        else:
            heapq.heappush(self._high, v)
        target = int(self.n * self.frac) + 1
        while len(self._low) > target:
            heapq.heappush(self._high, -heapq.heappop(self._low))
        while len(self._low) < target and self._high:
            heapq.heappush(self._low, -heapq.heappop(self._high))

    def threshold(self) -> float:
        """The ``int(frac * n)``-th smallest value (0-based) pushed so far."""
        return -self._low[0]


class Results:
    """
    A very simple database of results with a notification for new results.
    The new results are fed directly by the :class:`.StrategyBase`, outside of the
    :class:`.EventBus`.

    .. Note::

      Later on, maybe this will be a cool actual database which allows to
      persistently store past evaluations for a given problem.
      This would allow resuming and further a-posteriory analysis.
      In the meantime, this is a pandas DataFrame.
    """

    def __init__(self, strategy: "StrategyBase") -> None:
        self.logger = strategy.config.get_logger("RSLTS")
        self.strategy: "StrategyBase" = strategy
        self.eventbus: "EventBus" = strategy.eventbus
        self.problem: "Problem" = strategy.problem
        self._results_df: Optional["DataFrame"] = None
        self._unmerged_dfs: List["DataFrame"] = []
        self._buffer: List["Result"] = []
        self._last_nb: int = 0  # for logging
        # Order statistics of past fx for the progress reporter, built lazily
        # the first time a batch arrives with the reporter on; ``_fed`` is the
        # ``len(self)`` they cover, so a gap (reporter toggled, frame replaced)
        # triggers a rebuild instead of silently going stale.
        self._progress_ranks: Optional[_RankTracker] = None
        self._progress_ranks_fed: int = -1
        self._lock: RLock = RLock()

        # Initialize storage backend if configured
        self.backend: Any = None
        if hasattr(strategy.config, "storage_backend") and strategy.config.storage_backend == "sqlite":
            from .storage import SQLiteStorage, problem_fingerprint

            # The fingerprint keeps a run from resuming another problem's
            # results (the default URI is a shared ``panobbgo.db`` in the cwd).
            self.backend = SQLiteStorage(
                strategy.config.storage_uri,
                fingerprint=problem_fingerprint(self.problem),
                adopt_legacy=bool(getattr(strategy.config, "storage_adopt_legacy", False)),
            )
            self.logger.info(f"Using SQLite storage backend: {strategy.config.storage_uri}")

    def load_from_storage(self) -> int:
        """
        Load results from storage backend and populate the database.
        """
        if self.backend:
            loaded_results = self.backend.load()
            if loaded_results:
                self.logger.info(f"Loaded {len(loaded_results)} results from storage.")
                self.add_results(loaded_results, save_to_storage=False)
                return len(loaded_results)
        return 0

    @property
    def results(self) -> Optional["DataFrame"]:
        with self._lock:
            self._flush_buffer()
            if self._unmerged_dfs:
                if self._results_df is None or self._results_df.empty:
                    if len(self._unmerged_dfs) == 1:
                        self._results_df = self._unmerged_dfs[0]
                    else:
                        self._results_df = concat(self._unmerged_dfs, ignore_index=True)
                else:
                    self._results_df = concat([self._results_df] + self._unmerged_dfs, ignore_index=True)
                self._unmerged_dfs = []
            return self._results_df

    @results.setter
    def results(self, value: Optional["DataFrame"]) -> None:
        with self._lock:
            self._results_df = value
            self._unmerged_dfs = []
            self._buffer = []
            # Derived state describes the old frame: drop it.
            self._progress_ranks = None
            self._progress_ranks_fed = -1
            self._last_nb = 0 if value is None else len(value)

    def _flush_buffer(self) -> None:
        """
        Flush buffered results to the DataFrame.
        """
        with self._lock:
            if not self._buffer:
                return

            # Initialize DataFrame if needed
            if self._results_df is None:
                r = self._buffer[0]
                midx_x = [("x", _) for _ in range(r.x.size if hasattr(r.x, "size") else len(r.x))]  # pyright: ignore
                cv_vec = r.cv_vec
                len_cv_vec = 0 if cv_vec is None else np.atleast_1d(cv_vec).size
                midx_cv = [("cv_vec", _) for _ in range(len_cv_vec)]
                midx = MultiIndex.from_tuples(midx_x + [("fx", 0)] + midx_cv + [("cv", 0), ("who", 0), ("error", 0)])
                self._results_df = DataFrame(columns=midx)

            # Build data with explicit types to avoid mixed-type array issues
            # (np.r_ with strings causes all values to become object dtype)
            n_results = len(self._buffer)
            if n_results == 0:
                return  # Should not happen given check above, but for safety

            r0 = self._buffer[0]
            dim_x = r0.x.size if hasattr(r0.x, "size") else len(r0.x)  # pyright: ignore
            r0_cv_vec = r0.cv_vec
            len_cv_vec = 0 if r0_cv_vec is None else np.atleast_1d(r0_cv_vec).size

            # Pre-allocate typed arrays
            x_data = np.empty((n_results, dim_x), dtype=np.float64)
            fx_data = np.empty(n_results, dtype=np.float64)
            cv_vec_data = np.empty((n_results, len_cv_vec), dtype=np.float64) if len_cv_vec > 0 else None
            cv_data = np.empty(n_results, dtype=np.float64)
            who_data = np.empty(n_results, dtype=object)  # strings
            error_data = np.empty(n_results, dtype=np.float64)

            for i, r in enumerate(self._buffer):
                x_data[i, :] = r.x
                fx_data[i] = r.fx if r.fx is not None else np.nan
                if cv_vec_data is not None:
                    cv_vec_data[i, :] = r.cv_vec if r.cv_vec is not None else 0.0
                cv_data[i] = r.cv if r.cv is not None else 0.0
                who_data[i] = r.who
                error_data[i] = r.error if r.error is not None else 0.0

            # Build DataFrame with proper column types
            data_dict = {}
            for j in range(dim_x):
                data_dict[("x", j)] = x_data[:, j]
            data_dict[("fx", 0)] = fx_data
            if cv_vec_data is not None:
                for j in range(len_cv_vec):
                    data_dict[("cv_vec", j)] = cv_vec_data[:, j]
            data_dict[("cv", 0)] = cv_data
            data_dict[("who", 0)] = who_data
            data_dict[("error", 0)] = error_data

            results_new = DataFrame(
                data_dict, columns=self._results_df.columns if self._results_df is not None else midx
            )

            if self._results_df is None and not self._unmerged_dfs:
                self._results_df = results_new
            else:
                self._unmerged_dfs.append(results_new)

            self._buffer = []

    def add_results(self, new_results: List["Result"], save_to_storage: bool = True) -> None:
        """
        Add one single or a list of new @Result objects.
        Then, publish a ``new_result`` event.
        """
        # Persist to storage backend if enabled
        if self.backend and save_to_storage:
            self.backend.save(new_results)

        if not new_results:
            return

        assert all([isinstance(_, Result) for _ in new_results])

        # Progress stats are computed *before* the batch lands in the buffer
        # below, so "previous best" keeps its meaning.  All of this is skipped
        # when the progress reporter is off (the default without a TTY).
        reporting = self._progress_reporting()
        progress_stats: Dict[str, float] = {}
        if reporting:
            progress_stats = self._progress_stats()

        # The batch must be *in* the store before anyone is told about it.
        # ``publish`` hands the event to the bus thread, which runs the
        # subscribers' handlers concurrently with whatever this (main) thread
        # does next.  Several handlers pace themselves on
        # ``len(strategy.results)`` — the DE family's LPSR population
        # schedule, its F-schedule and its ``p_best`` annealing all read
        # :meth:`panobbgo.heuristics.lshade.LSHADE._progress`, and the
        # constraint handlers count evaluations the same way.  Publishing
        # first raced this ``_buffer.extend``: whether a handler counted its
        # own batch depended on thread scheduling, so a run was reproducible
        # only by luck (an LPSR step could land one batch early or late, which
        # shifts the RNG stream and every point drawn after it).  Landing the
        # results first, and publishing *last*, makes the count a pure
        # function of the evaluation sequence.
        with self._lock:
            n_before = len(self)
            self._buffer.extend(new_results)

            if reporting:
                # The tracker covers ``_progress_ranks_fed`` results.  Extend
                # it only if that is exactly the history this batch lands on;
                # otherwise (a concurrent ``add_results`` got in between the
                # stats and the extend) leave it marked stale, and the next
                # ``_progress_stats`` rebuilds it from the store.
                ranks = self._progress_ranks
                if ranks is not None and self._progress_ranks_fed == n_before:
                    for r in new_results:
                        if r.fx is not None:
                            ranks.push(float(r.fx))
                    self._progress_ranks_fed = len(self)
                else:
                    self._progress_ranks_fed = -1

        if reporting:
            for result in new_results:
                self._report_evaluation_progress(result, stats=progress_stats)

        if len(self) // 100 > self._last_nb // 100:
            self.info()
            self._last_nb = len(self)

        # notification for all received results at once — last, so no
        # main-thread bookkeeping overlaps the handler cascade.
        self.eventbus.publish("new_results", results=new_results)

    def close(self) -> None:
        """
        Close the storage backend if it exists.
        """
        if self.backend and hasattr(self.backend, "close"):
            self.backend.close()

    def info(self) -> None:
        self.logger.info("%d results in DB" % len(self))
        # ``self.results`` concatenates every pending frame, so only touch it
        # when the line will actually be emitted.  ``logger.isEnabledFor`` is
        # no help here: :func:`panobbgo.utils.create_logger` puts the level on
        # the handler and leaves the logger itself at DEBUG.
        if self.strategy.config.loglevel <= logging.DEBUG and self.results is not None:
            self.logger.debug("Dataframe Results:\n%s" % self.results.tail(3))

    def __iadd__(self, results: List["Result"]) -> "Results":
        self.add_results(results)
        return self

    def __len__(self) -> int:
        with self._lock:
            count = len(self._results_df) if self._results_df is not None else 0
            count += sum(len(df) for df in self._unmerged_dfs)
            count += len(self._buffer)
            return count

    def get_history(self, n: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve the last n results as numpy arrays.

        Args:
            n (int, optional): Number of recent results to retrieve. If None, returns all.

        Returns:
            dict: Dictionary with keys 'x', 'fx', 'cv', 'cv_vec', 'who', 'error'.
                  Values are numpy arrays (or list/array of objects for 'who').
        """
        if self.results is None or self.results.empty:
            return {
                "x": np.array([]),
                "fx": np.array([]),
                "cv": np.array([]),
                "cv_vec": np.array([]),
                "who": np.array([]),
                "error": np.array([]),
            }

        df = self.results
        if n is not None:
            df = df.iloc[-n:]

        # Extract data using column names
        # 'x' is a top-level column with sub-columns 0, 1, ...
        # df['x'] returns a DataFrame. .values converts to numpy array (N, dim)
        x = df["x"].values.astype(float)

        # 'fx' is ('fx', 0)
        fx = df[("fx", 0)].values.astype(float)

        # 'cv' is ('cv', 0)
        cv = df[("cv", 0)].values.astype(float)

        # 'cv_vec' might not exist or might have multiple columns
        if "cv_vec" in df.columns.get_level_values(0):
            cv_vec = df["cv_vec"].values.astype(float)
        else:
            # If no constraint vector, return empty array with shape (N, 0)
            cv_vec = np.zeros((len(df), 0))

        # 'who' is ('who', 0)
        who = df[("who", 0)].values.astype(str)

        # 'error' is ('error', 0)
        error = df[("error", 0)].values.astype(float)

        return {
            "x": x,
            "fx": fx,
            "cv": cv,
            "cv_vec": cv_vec,
            "who": who,
            "error": error,
        }

    def _progress_reporting(self) -> bool:
        """Whether per-evaluation progress output is on for this strategy."""
        logger = getattr(self.strategy, "panobbgo_logger", None)
        return logger is not None and bool(logger.progress_reporter.enabled)

    def _progress_stats(self) -> Dict[str, float]:
        """Statistics of the results stored so far, for :meth:`_report_evaluation_progress`.

        ``current_best_fx`` is the best fx so far;
        ``threshold`` is the ``int(0.1 * n)``-th smallest of the ``n`` finite
        fx so far (only once ``n > 10``).  O(log n) per result via
        :class:`_RankTracker`; a full rebuild happens only when the tracker does
        not cover the current history.
        """
        with self._lock:
            if self._progress_ranks is None or self._progress_ranks_fed != len(self):
                ranks = _RankTracker()
                try:
                    for fx in self._all_fx():
                        ranks.push(fx)
                except Exception:
                    ranks = _RankTracker()
                self._progress_ranks = ranks
                self._progress_ranks_fed = len(self)
            ranks = self._progress_ranks
        stats: Dict[str, float] = {}
        if len(self) > 0:
            stats["current_best_fx"] = ranks.min
        if ranks.n > 10:
            stats["threshold"] = ranks.threshold()
        return stats

    def _all_fx(self) -> List[float]:
        """Every stored fx, oldest first, without concatenating the frames."""
        with self._lock:
            fx_values: List[float] = []
            frames = ([self._results_df] if self._results_df is not None else []) + list(self._unmerged_dfs)
            for df in frames:
                if not df.empty:
                    fx_values.extend(df.xs(0, level=1, axis=1)["fx"].dropna().tolist())
            fx_values.extend(float(r.fx) for r in self._buffer if r.fx is not None)
            return fx_values

    def _report_evaluation_progress(self, result: "Result", stats: Optional[Dict[str, float]] = None) -> None:
        """
        Report progress for a single evaluation result.

        Args:
            result: The evaluation result to report
            stats: Optional dictionary of pre-calculated statistics
        """
        # Get strategy's logger for progress reporting
        if not hasattr(self.strategy, "panobbgo_logger"):
            return  # No logger configured, skip progress reporting

        progress_reporter = self.strategy.panobbgo_logger.progress_reporter

        # Determine progress context
        context = ProgressContext()

        # Only analyze if we have valid fx
        if result.fx is not None:
            try:
                # Stats describe the history *before* this batch.
                stats = stats or {}
                if "current_best_fx" in stats and result.fx < stats["current_best_fx"]:
                    context.is_global_best = True
                # Significant: within the top 10 % of the results so far.
                if "threshold" in stats and result.fx < stats["threshold"]:
                    context.is_significant_improvement = True
            except (ValueError, TypeError, KeyError):
                # If we can't determine improvement status, skip it
                pass

        # Check for failure (fx is None or has error)
        if result.fx is None or (result.error and result.error > 0):
            context.evaluation_failed = True

        # Check for warnings (constraint violations, etc.)
        if result.cv is not None and result.cv > 0:
            context.has_warnings = True

        # Report the evaluation
        progress_reporter.report_evaluation(result, context)


def known_budget(max_eval: Any) -> Optional[float]:
    """``max_eval`` as a positive finite float, or ``None`` when the budget is unknown.

    Unknown means missing, non-numeric, non-finite, zero or negative.  The
    one parser behind :meth:`Module.budget_progress` / :meth:`Module.max_eval_or`
    and their :class:`StrategyBase` twins.
    """
    try:
        v = float(max_eval)
    except Exception:
        return None
    if not np.isfinite(v) or v <= 0.0:
        return None
    return v


def budget_progress_of(max_eval: Any, n_results: Callable[[], int]) -> Optional[float]:
    """``n_results() / max_eval`` clipped to ``[0, 1]``; ``None`` when the budget is unknown.

    *n_results* is a callable so a strategy stand-in without ``results``
    degrades to ``None`` instead of raising.
    """
    budget = known_budget(max_eval)
    if budget is None:
        return None
    try:
        current = float(n_results())
    except Exception:
        return None
    return float(np.clip(current / budget, 0.0, 1.0))


def max_eval_or_default(max_eval: Any, default: int) -> int:
    """``int(max_eval)`` when the budget is known (see :func:`known_budget`), else *default*."""
    budget = known_budget(max_eval)
    return default if budget is None else int(budget)


#: Stream key of the strategy-level generator (:attr:`StrategyBase.rng`).
#: A one-element ``spawn_key`` cannot collide with a module stream, whose
#: key has two elements.
_STRATEGY_STREAM_KEY: Tuple[int, ...] = (zlib.crc32(b"panobbgo.strategy"),)


def rng_stream_key(name: str, occurrence: int) -> Tuple[int, int]:
    """The ``SeedSequence`` ``spawn_key`` of the ``occurrence``-th module called ``name``."""
    return (zlib.crc32(name.encode("utf-8")), int(occurrence))


def keyed_rng(seed: int, spawn_key: Tuple[int, ...]) -> np.random.Generator:
    """A generator derived from the master ``seed`` and a stable ``spawn_key``.

    ``SeedSequence(seed, spawn_key=...)`` streams are independent for
    distinct keys and depend on nothing else — not on how many other
    streams were drawn before, nor in which order.
    """
    return np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=tuple(int(k) for k in spawn_key)))


def _module_rng(strategy: Any, key: str) -> np.random.Generator:
    """The generator a :class:`Module` named ``key`` owned by ``strategy`` draws from.

    A stream from :meth:`StrategyBase.spawn_rng`, keyed by the module's
    name, which makes the module's randomness a function of the strategy's
    seed and of its own name only.  A strategy stand-in must provide
    ``spawn_rng(key)`` too (the test suite's
    ``tests.support.StrategyDouble`` / ``attach_spawn_rng`` do): there is no
    unseeded fallback, which would silently make the module irreproducible.
    """
    spawn = getattr(strategy, "spawn_rng", None)
    rng = spawn(key) if callable(spawn) else None
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "%s.spawn_rng() must return a numpy Generator (got %r); modules draw their "
            "randomness from the strategy's seed" % (type(strategy).__name__, type(rng).__name__)
        )
    return rng


class Module:
    """
    "Abstract" parent class for various panobbgo modules, e.g.
    :class:`.Heuristic` and :class:`.Analyzer`.
    """

    #: Names of the analyzers this module reads (``strategy.analyzer(name)``)
    #: or whose events it subscribes to, e.g. ``("Splitter",)``.
    #: :meth:`StrategyBase.initialize` installs a default analyzer that is
    #: only needed on demand (the ``Splitter``) exactly when some module
    #: declares it here; see :meth:`required_analyzers`.
    requires_analyzers: Tuple[str, ...] = ()

    def __init__(self, strategy: "StrategyBase", name: Optional[str] = None) -> None:
        """
        :param StrategyBase strategy:
        :param str name:
        @type strategy: StrategyBase
        """
        name = name if name else self.__class__.__name__
        self._strategy: "StrategyBase" = strategy
        self.config = strategy.config
        self._name: str = name
        #: Per-module random generator, keyed by the master seed and this
        #: module's name (see :meth:`StrategyBase.spawn_rng`): adding,
        #: removing or reordering *other* modules does not change it.
        #: Modules must draw randomness from here, never from the global
        #: ``np.random`` state.
        self.rng: np.random.Generator = _module_rng(strategy, name)
        self._threads: List[Any] = []
        # implicit dependency check (only class references)
        self._depends_on: List[Any] = []
        self._stopped: bool = False
        # Ensure logger name is at most 5 characters to satisfy config.get_logger assertion
        log_name = self._name[:5].upper()
        self.logger = self.config.get_logger(log_name)

    def budget_progress(self) -> Optional[float]:
        """Fraction of the evaluation budget used, ``len(strategy.results) / max_eval`` in ``[0, 1]``.

        ``None`` when the budget is unknown (no ``max_eval``, zero, negative
        or non-numeric), so a schedule can fall back to its constant setting
        instead of guessing a horizon.
        """
        return budget_progress_of(self.config.max_eval, lambda: len(self._strategy.results))

    def max_eval_or(self, default: int) -> int:
        """The evaluation budget as an ``int``, or *default* when it is unknown."""
        return max_eval_or_default(self.config.max_eval, default)

    def derive_rng(self, seed: Optional[int]) -> np.random.Generator:
        """Generator for a sub-component: ``seed`` if given, else this module's own.

        Heuristics that expose a ``seed`` argument use this so an explicit
        seed pins that component while ``None`` keeps it inside the
        strategy's reproducible stream.
        """
        return self.rng if seed is None else np.random.default_rng(seed)

    def derive_seed(self, seed: Optional[int]) -> int:
        """Integer seed for a component that needs one (e.g. ``scipy.stats.qmc``)."""
        return int(seed) if seed is not None else int(self.rng.integers(2**31))

    def spawn_thread(self, target: Callable[[], None], name: Optional[str] = None) -> "Thread":
        """Run ``target`` on a daemon thread owned by this module.

        The event bus delivers handlers serially, so anything that blocks —
        a pipe pump bridging a subprocess solver, say — must not run inside
        a handler.  Threads registered here are joined by :meth:`__stop__`.
        """
        t = Thread(target=target, name=name or ("%s-thread" % self._name), daemon=True)
        self._threads.append(t)
        t.start()
        return t

    @property
    def name(self) -> str:
        """
        The module's name.

        .. Note::

          It should be unique, which is important for
          parameterized heuristics or analyzers!
        """
        return self._name

    @property
    def strategy(self) -> "StrategyBase":
        return self._strategy

    @property
    def eventbus(self) -> "EventBus":
        return self._strategy.eventbus

    @property
    def problem(self) -> "Problem":
        return self._strategy.problem

    @property
    def results(self) -> Any:
        return self._strategy.results

    def check_dependencies(self, analyzers: List["Analyzer"], heuristics: List["Heuristic"]) -> bool:
        """
        This method is called by the core initialization to assess,
        if the dependencies for the given module are met.

        By default, it returns true. Return false if there is
        a problem.

        The arguments are the list of pre-initialized analyzers
        and heuristics.
        """
        return True

    def required_analyzers(self) -> Tuple[str, ...]:
        """Analyzer names this instance needs; defaults to :attr:`requires_analyzers`.

        Override when the need depends on constructor arguments.
        """
        return tuple(self.requires_analyzers)

    def __start__(self) -> None:
        """
        This method should be overwritten by the respective subclass.
        It is called in the 2nd initialization phase, inside :meth:`._init_module`.
        Now, the strategy and all its components (e.g. :class:`panobbgo.lib.Problem`, ...)
        are available.
        """
        pass

    def __stop__(self) -> None:
        """
        Called right at the end after the strategy has finished.
        """
        self._stopped = True
        self._strategy.eventbus.unsubscribe(None, self)
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=0.5)
                if t.is_alive():
                    self.logger.debug("Thread %s did not terminate gracefully." % t.name)

    def _init_plot(self) -> Tuple[Any, Any]:
        """
        This plot initializer is called right after the :meth:`._init` method.
        It could be used to tell the (optionally enabled) :module:`user interface <.ui>` that
        this module wants to have a tab for displaying and visualizing some data.

        It has to return a tuple consisting of a string as the label of the tab,
        and a gtk container (e.g. :class:`gtk.VBox`)

        To trigger a redraw after an update, call the ``.draw_idle()`` method
        of the :class:`~matplotlib.backends.backend_gtkagg.FigureCnavasGTKAgg`.
        """
        return None, None

    def __repr__(self) -> str:
        return "Module %s" % self.name


class StopHeuristic(Exception):
    """
    Indicates the heuristic has finished and should be ignored/removed.
    """

    def __init__(self, msg: str = "stopped") -> None:
        """
        Args:

        - ``msg``: a custom message, will be visible in the log. (default: "stopped")
        """
        Exception.__init__(self, msg)


class Heuristic(Module):
    """
    This is the "abstract" parent class for all types of point generating classes,
    which we call collectively ":mod:`Heuristics <.heuristics>`".

    Such a heuristic is capable of the following:

    #. They can be parameterized by passing in optional arguments in the constructor.
       This should be reflected in the :attr:`~.Module.name`!
    #. The :class:`.EventBus` spawns a thread for each ``on_*`` method
       and calls them when a corresponding :class:`.Event` occurs.
    #. Of course, they are capable of storing their state in the instance.
       This is also the way of how information is shared between those threads.
    #. The `main purpose` of a heuristic is to emit new search points
       by calling either :meth:`.emit` or returning a list of points.
       The datatype must be :class:`numpy.ndarray` of
       `floats <http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html>`_.
    #. Additionally, the can get hold of other heuristics or anayzers via the strategy instance.
    #. The :class:`.EventBus` inside this strategy instance allows them to publish their
       own events, too. This can be used to signal related heuristics something
       or to queue up tasks for itself.
    """

    def __init__(self, strategy: "StrategyBase", name: Optional[str] = None, cap: Optional[int] = None) -> None:
        Module.__init__(self, strategy, name)
        self.cap: int = cap if cap is not None else self.config.capacity

        self._output: Queue = Queue(self.cap)

        # statistics; performance
        self.performance: float = 0.0

    def clear_output(self) -> None:
        q = self._output
        with q.not_full:
            # with q.mutex:
            # del self._output.queue[:]  # LifoQueue
            q.queue.clear()  # Queue
            q.not_full.notify()  # to wakeup "put()"

    def ensure_output_capacity(self, n: int) -> None:
        """
        Grow the output queue so that at least ``n`` *additional* points fit.

        Heuristics that emit points in batches whose size can exceed the
        configured queue capacity (e.g. CMA-ES after IPOP restarts double λ)
        must call this before emitting, otherwise ``put_nowait`` silently
        drops the overflow and the heuristic deadlocks waiting for results
        that will never arrive.
        """
        q = self._output
        with q.mutex:
            needed = len(q.queue) + n
            if 0 < q.maxsize < needed:
                q.maxsize = needed
                q.not_full.notify_all()

    def _put(self, point: "Point") -> None:
        """Queue one :class:`Point`, growing the queue if it is full.

        ``cap`` is the *fill level* reactive heuristics top up to, not a
        hard limit: a population heuristic that emits a whole generation
        at once must never lose part of it.  (Until 2026-09 the overflow
        was dropped silently, so ``NP_init=90`` ran as a 20-member
        population.)
        """
        self.ensure_output_capacity(1)
        self._output.put_nowait(point)

    def new_who(self, rng: Optional[np.random.Generator] = None) -> str:
        """A ``who`` tag carrying a unique request id: ``"<name>:<hex>"``.

        Heuristics that must match a result back to the trial that produced
        it (the DE family, PSO) tag each point this way.  The id is drawn
        from the module's own generator, so it is part of the reproducible
        stream.
        """
        return "%s:%s" % (self.name, uuid.UUID(bytes=(rng or self.rng).bytes(16)).hex)

    def fill_queue(self, draw: Callable[[], np.ndarray]) -> None:
        """Top the output queue up to :attr:`cap` with points from ``draw``.

        The refill contract of every *reactive* heuristic: fill on ``start``
        (or whatever event defines its search region) and top up on each
        result batch, so the strategy always finds points without the
        heuristic having to poll.
        """
        free = self.cap - self._output.qsize()
        if free > 0:
            self.emit([draw() for _ in range(free)])

    #: Selector names :meth:`archive_seed` understands.  ``None`` (the
    #: default of every ``warm_start=`` constructor argument) means "cold
    #: start", i.e. do not call :meth:`archive_seed` at all.
    WARM_START_MODES: Tuple[str, ...] = ("archive", "archive_diverse", "archive_leaf")

    def required_analyzers(self) -> Tuple[str, ...]:
        """Also the ``Splitter`` when this heuristic warm-starts.

        :meth:`archive_seed` reads ``Splitter`` leaves for ``"archive_leaf"``
        and falls back to the ``Splitter`` root without an ``Archive``; a
        ``warm_start_box`` is a ``Splitter`` box.  Every heuristic with a
        warm start stores its mode in ``self.warm_start``.
        """
        req = Module.required_analyzers(self)
        if getattr(self, "warm_start", None) or getattr(self, "warm_start_box", None) is not None:
            req += ("Splitter",)
        return req

    def archive_seed(self, k: int, *, mode: Optional[str] = None, box: Any = None) -> List["Result"]:
        """Up to ``k`` good points from the *shared* archive, best first.

        The query layer of ``planning/DESIGN_warm_start_2026-09-10.md`` §2:
        a heuristic that wants to start from points it did not pay for asks
        here instead of reaching into another module's state.  Three
        selectors, all of them unfiltered by ``who``:

        * ``"archive"`` (default) — the ``k`` best results;
        * ``"archive_diverse"`` — ``k`` well-separated good results;
        * ``"archive_leaf"`` — the best point of each of the ``k`` best
          :class:`~panobbgo.analyzers.splitter.Splitter` leaves, i.e. ``k``
          *different basins*.

        Source is the opt-in :class:`~panobbgo.analyzers.archive.Archive`
        analyzer when the strategy has one; otherwise the ``Splitter``'s
        root box, which holds every result in :class:`~panobbgo.lib.Result`
        form, sorted by penalty value.  With neither analyzer — or with
        nothing evaluated yet — the answer is ``[]``, which every caller has
        to read as "use the cold path".

        Never raises: a missing analyzer, an unstarted one or a broken
        constraint handler all degrade to ``[]``.
        """
        if k <= 0:
            return []
        archive = self._archive_analyzer()
        seeds: Any = []
        if archive is not None:
            try:
                if mode == "archive_diverse":
                    seeds = archive.diverse_k(k, box=box)
                elif mode == "archive_leaf":
                    seeds = archive.per_leaf_best(k)
                else:
                    seeds = archive.top_k(k, box=box)
            except Exception:
                seeds = []
        if not seeds:
            seeds = self._splitter_seed(k, box=box)
        if not isinstance(seeds, (list, tuple)):
            return []
        return [r for r in seeds if isinstance(r, Result)][:k]

    def _archive_analyzer(self) -> Any:
        """The :class:`Archive` analyzer, or ``None`` if the strategy has none."""
        try:
            from panobbgo.analyzers.archive import Archive

            analyzer = self._strategy.analyzer("Archive")
        except Exception:
            return None
        return analyzer if isinstance(analyzer, Archive) else None

    def _splitter_seed(self, k: int, *, box: Any = None) -> List["Result"]:
        """Fallback for :meth:`archive_seed`: sort the ``Splitter``'s root box."""
        try:
            splitter = self._strategy.analyzer("Splitter")
            region = box if box is not None else getattr(splitter, "root", None)
            pool = list(getattr(region, "results", []))
        except Exception:
            return []
        from panobbgo.lib.constraints import result_key

        handler = getattr(self._strategy, "constraint_handler", None)

        def key(r: "Result") -> tuple:
            # The handler's ordering (what ``Best`` and ``Archive`` rank by).
            try:
                return tuple(float(v) for v in result_key(handler, r))
            except (TypeError, ValueError):
                return (float("inf"),)

        return sorted(pool, key=key)[:k]

    def warm_start_now(self) -> bool:
        """Re-seed this heuristic from the shared archive, right now.

        Called by a scheduler that just handed this heuristic a fresh block
        of evaluations (see
        :class:`~panobbgo.strategies.blocks.StrategyBlockBandit`), by direct
        method call — the event bus only broadcasts.  Returns ``True`` iff a
        seed was actually used.  The default is ``False``: a heuristic that
        does not opt in is left exactly as it was.
        """
        return False

    def emit(self, points: Union[np.ndarray, List[np.ndarray], "Point", List["Point"], List[Any]]) -> None:
        """
        This is used to send out new search points for evaluation.
        Args:

        - ``points``: Either a :class:`numpy.ndarray` of ``float64`` or preferrably a list of them.

        Raises ``TypeError`` for anything that is not an ndarray — a wrong
        return type is a bug in the heuristic, not something to skip quietly.
        """
        if self._stopped:
            raise StopHeuristic()
        if points is None:
            self._stopped = True
            self.logger.info("'%s' heuristic stopped." % self.name)
            return
        if not isinstance(points, (list, tuple)):
            points = [points]
        self.ensure_output_capacity(len(points))
        for point in points:
            if not isinstance(point, np.ndarray):
                raise TypeError("%s emitted %r, expected a numpy ndarray" % (self.name, type(point).__name__))
            self._output.put_nowait(Point(self.problem.project(point), self.name))

    def get_points(self, limit: Optional[int] = None) -> List["Point"]:
        """
        Drain the output queue until ``limit`` points are removed or it is empty.
        """
        new_points = []
        try:
            while limit is None or len(new_points) < limit:
                new_points.append(self._output.get(block=False))
        except Empty:
            pass
        return new_points

    #: ``True`` iff this heuristic produces points **on demand**, on the
    #: caller's thread, rather than reactively topping its queue up from
    #: event handlers.  The solver bridges
    #: (:class:`~panobbgo.heuristics.lbfgsb.LBFGSB`,
    #: :class:`~panobbgo.heuristics.cobyqa.COBYQA`) set it: a sequential
    #: SciPy optimizer cannot queue a point ahead, because its next iterate
    #: is a function of ``f`` at the current one.  A scheduler must call
    #: :meth:`produce` — never :meth:`get_points` — or such an arm is
    #: starved by any competitor that keeps a stocked queue
    #: (``planning/DESIGN_pump_and_stall_2026-09-11.md`` §1).
    on_demand: bool = False

    def produce(self, limit: Optional[int] = None, timeout: Optional[float] = None) -> List["Point"]:
        """The scheduler's point-acquisition call.

        For a reactive heuristic this is exactly :meth:`get_points` — the
        queue is whatever its event handlers put there.  An
        :attr:`on_demand` heuristic overrides it and *produces* the next
        point synchronously, on the calling thread.

        ``timeout`` is a deadlock backstop for the on-demand case, never a
        scheduling parameter: a healthy producer answers immediately, so
        the value cannot influence *which* points a run evaluates — only
        whether the run reports a wedged worker.
        """
        return self.get_points(limit)

    @property
    def has_points(self) -> bool:
        """``True`` iff this heuristic can hand out a point right now.

        Selection strategies use this to skip heuristics that would
        return an empty list, so the probability mass they would have
        received is not wasted.
        """
        return self._output.qsize() > 0

    @property
    def can_produce(self) -> bool:
        """``True`` iff :meth:`produce` would hand out a point right now.

        The same question as :attr:`has_points` for a reactive heuristic,
        and the *correct* one for an :attr:`on_demand` one, whose queue is
        empty by construction between round trips.  This is the predicate
        schedulers gate on and the one
        :meth:`StrategyBase._alive` sums over.
        """
        return self.has_points

    @property
    def active(self) -> bool:
        """
        This is queried by the strategy to determine, if it should still consider it.
        This is the case, iff there is still something in its output queue
        or if there is a chance that there will be something in the future
        (it still listens to at least one event).
        """
        if self._output.qsize() > 0:
            return True
        if self._stopped:
            return False
        if any(t.is_alive() for t in self._threads):
            return True  # a background pump (subprocess bridge) is still running
        bus = getattr(self._strategy, "eventbus", None)
        return bool(bus.is_subscribed(self)) if bus is not None else True


def pipe_objective(pipe: Any) -> Callable[[np.ndarray], float]:
    """Worker side of a :class:`PipeBridgeHeuristic`: ``f(x)`` as a pipe round trip.

    Sends ``x`` (as a float array), blocks for the value and maps ``None`` or
    a non-finite value to ``inf``.  A closed pipe raises ``SystemExit`` so the
    worker shuts down cleanly when the parent terminates it.
    """

    def f(x: np.ndarray) -> float:
        pipe.send(np.asarray(x, dtype=float))
        try:
            fx = pipe.recv()
        except (EOFError, OSError):
            raise SystemExit(0)
        if fx is None or not np.isfinite(fx):
            return float("inf")
        return float(fx)

    return f


def safe_send(output: Any, payload: Any) -> None:
    """Worker side: send ``payload`` over the status pipe, ignoring a parent that already went away."""
    try:
        output.send(payload)
    except Exception:
        pass


def terminate_process(proc: Any, timeout: float = 1.0) -> None:
    """Terminate a worker process if it is alive; kill it if it ignores that."""
    if proc is None or not proc.is_alive():
        return
    proc.terminate()
    proc.join(timeout)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout)


class HeuristicSubprocess(Heuristic):
    r"""
    This Heuristic is a subclass of :class:`.Heuristic`, which is additionally starting
    a subprocess, which communicates with the main thread via a pipe in a blocking
    communication scheme.
    """

    def __init__(self, strategy: "StrategyBase", name: Optional[str] = None, cap: Optional[int] = None) -> None:
        Heuristic.__init__(self, strategy, name=name, cap=cap)

        # Use 'spawn' context to avoid DeprecationWarning: use of fork() may lead to deadlocks
        # in multi-threaded process. 'spawn' is safer and cross-platform.
        ctx = multiprocessing.get_context("spawn")

        # a pipe has two ends, parent and child.
        self.pipe, self.pipe_child = ctx.Pipe()
        self.__subprocess = ctx.Process(
            target=self.subprocess,
            args=(self.pipe_child,),
            name="%s-subprocess" % (self.name),
        )
        self.__subprocess.daemon = True
        self.__subprocess.start()

    def __stop__(self) -> None:
        """Close both pipe ends and terminate the worker process, then stop as usual.

        ``daemon=True`` only reaps the worker when the *interpreter* exits; a
        long-lived process running many strategies leaked one per run.
        Unsubscribes first, so no handler can touch the pipe while it closes.
        """
        super().__stop__()
        for end in (self.pipe, self.pipe_child):
            try:
                end.close()
            except Exception:
                pass
        terminate_process(self.__subprocess)

    @staticmethod
    def subprocess(pipe: Any) -> None:
        """
        overwrite this pipe.recv() & pipe.send() loop and
        compute something in between.
        """
        while True:
            payload = pipe.recv()
            pipe.send("subprocess received: %s" % payload)


class PipeBridgeHeuristic(Heuristic):
    r"""A sequential solver in a subprocess, **pulled** from the caller's thread.

    SciPy's local optimizers (``fmin_l_bfgs_b``, ``minimize(method="COBYQA")``)
    are synchronous: they call ``f(x)`` and block for the return value, and
    they cannot be suspended mid-step to yield an evaluation request.  So the
    solver runs in its own subprocess and each ``f(x)`` is a round trip over a
    pipe.  The protocol is strictly one outstanding evaluation:

    .. code-block:: none

        worker: f(x) -> pipe.send(x), blocks on recv
        parent: produce()  -> recv(x), emit -> the strategy evaluates it
        parent: on_new_results -> stores the penalty value
        parent: produce()  -> sends the value, recv's the next x

    **Every pipe operation happens on the thread that calls** :meth:`produce`,
    i.e. the strategy's main loop.  Until 2026-09 a daemon "pump" thread did
    the ``recv``/``emit`` half and the event-bus thread did the ``send`` half.
    That had three consequences, all removed here:

    #. the arm's output queue was empty whenever the scheduler looked, so any
       competitor with a stocked queue starved it completely — under
       :class:`~panobbgo.strategies.round_robin.StrategyRoundRobin` such an arm
       contributed **0** points beside ``Random``;
    #. :attr:`Heuristic.active` and
       :meth:`StrategyBase._can_still_produce` answered "yes" from the pump
       thread's liveness, which never ended, so they were constants;
    #. emissions were timing-dependent, which excluded these heuristics from
       the reproducible ``sync_evaluation`` mode.

    Now a bridge arm's contribution is a pure function of the seed: the point
    it hands out depends only on the values it was given and its worker's
    seed.  See ``planning/DESIGN_pump_and_stall_2026-09-11.md`` §1.

    Subclasses provide the subprocess and the pipes (``p1`` = parent end of
    the request pipe, ``out1`` = parent end of the status pipe), and may
    intercept control messages by overriding :meth:`_bridge_control`.

    **Restarts** follow the same rule.  An ``on_restart`` handler runs on the
    event-bus thread, so it only *records* the request
    (:meth:`_request_restart`); the next :meth:`produce` replaces the worker
    (:meth:`_bridge_respawn`) on the main loop's thread.  Respawning from the
    bus thread used to rebind ``p1`` under a ``produce`` that was mid-``recv``.
    A worker that *finished* (converged, hit ``max_starts``) is only
    ``_bridge_done``, not stopped: a later restart revives it.

    Only the value of the point we are *currently* waiting on is handed to the
    worker: :meth:`on_new_results` matches the result's ``x`` against it.  A
    point emitted by a replaced worker (or an aborted descent) may still be in
    flight, and its value must not answer the new worker's first ``f(x)``.
    """

    on_demand = True

    #: Parent end of the request pipe (``x`` in, ``f(x)`` out) and of the
    #: worker's status pipe.  Created by the subclass when it spawns.
    p1: Any = None
    out1: Any = None

    #: Deadlock backstop for :meth:`produce`, in seconds.  **Not** a
    #: scheduling parameter: a live worker is waited for however long it
    #: needs, and this can only expire when the worker is wedged — an error,
    #: logged as one.  It therefore never influences *which* points a run
    #: evaluates.
    bridge_timeout: float = 60.0

    #: Granularity of the wait.  Only affects how quickly a *dead* worker is
    #: noticed; a live worker's data is returned as soon as it arrives.
    _bridge_poll_slice: float = 0.05

    def __init__(self, strategy: "StrategyBase", name: Optional[str] = None, cap: Optional[int] = None) -> None:
        Heuristic.__init__(self, strategy, name=name, cap=cap)
        #: Penalty values handed over from the event-bus thread.  A queue,
        #: not an attribute, so the hand-off needs no lock and no GIL
        #: assumption.
        self._fx_inbox: Queue = Queue()
        #: ``True`` while the worker is blocked waiting for the value of a
        #: point we already emitted.  The structural deadlock guard: while it
        #: is set and the inbox is empty, :meth:`produce` returns ``[]``
        #: *immediately* instead of waiting for an arm that is waiting for us.
        self._outstanding: bool = False
        #: The worker finished (converged, hit its cap, or died).  Not
        #: ``_stopped``: a restart can still respawn it.
        self._bridge_done: bool = False
        #: The (projected) point whose value the worker is waiting for; only a
        #: result at this ``x`` answers it.
        self._outstanding_x: Optional[np.ndarray] = None
        #: ``(center,)`` recorded by :meth:`_request_restart` on the event-bus
        #: thread and applied by :meth:`produce` on the main thread.
        self._pending_restart: Optional[tuple] = None
        self._restart_lock = threading.Lock()
        #: Number of restart *requests* so far (counted on the event bus, so a
        #: pure function of the event sequence), and the number of the request
        #: the current worker was spawned for (``0``: the initial worker).
        #: Requests that arrive before the next ``produce`` coalesce, so
        #: respawns can skip numbers; the number itself never depends on timing.
        self._restart_requests: int = 0
        self._restart_index: int = 0
        #: Guards the hand-off of a value: the bus thread's "is this the
        #: outstanding point? then queue its value" and the main thread's
        #: reset / new outstanding point are atomic with respect to each other.
        #: Without it a reset could land between the check and the ``put`` and
        #: the old point's value would answer the new worker.
        self._handoff_lock = threading.Lock()

    # -- subclass hooks ----------------------------------------------------

    def _bridge_process(self) -> Any:
        """The worker :class:`multiprocessing.Process`, or ``None``."""
        raise NotImplementedError

    def _bridge_control(self, msg: Any) -> bool:
        """Handle a non-point message from the worker.

        Return ``True`` if ``msg`` was consumed (the loop then asks for the
        next message), ``False`` if it is a search point to emit.
        """
        return False

    def _bridge_point(self, msg: Any) -> Any:
        """The search point carried by ``msg``.

        The default is the message itself — L-BFGS-B and COBYQA send a bare
        ``ndarray``.  A worker with a richer protocol (a tagged dict, say)
        unwraps it here.
        """
        return msg

    def _bridge_send_fx(self, fx: float) -> None:
        """Hand the value of our outstanding point back to the worker.

        The default is a bare float, which is what
        :func:`panobbgo.core.pipe_objective` expects.  A
        worker with a tagged protocol wraps it here.
        """
        self.p1.send(fx)

    def _bridge_pending_request(self) -> bool:
        """``True`` while the worker is expected to send a point.

        The default is "always, as long as it lives": L-BFGS-B and COBYQA run
        one long solve and ask for an evaluation whenever they are not waiting
        on us.  A worker that *idles between searches* — accepting a "start"
        command, running a descent, reporting "done" and then waiting —
        overrides this, otherwise :meth:`produce` would sit on the pipe until
        the deadlock backstop every time the search is between runs.
        """
        return True

    def _bridge_respawn(self, center: Any) -> None:
        """Replace the worker with a fresh one started at ``center``.

        Called from :meth:`produce` (main thread) for a restart recorded by
        :meth:`_request_restart`.  Terminate the old process, spawn the new
        one and rebind the pipes; the base class resets the round-trip state
        afterwards.  Only arms that restart need to implement it.
        """
        raise NotImplementedError

    # -- worker plumbing shared by the subclasses ---------------------------

    def _bridge_spawn(self, target: Callable[..., None], args: Tuple[Any, ...], name: str) -> Any:
        """Create fresh pipes and start ``target(p2, out2, *args)`` in a ``"spawn"`` subprocess.

        Binds ``p1``/``p2`` (request pipe) and ``out1``/``out2`` (status
        pipe, one-way) and returns the started daemon process; the subclass
        keeps it and returns it from :meth:`_bridge_process`.  ``"spawn"``,
        not ``fork``: forking a multi-threaded process can deadlock.
        """
        ctx = multiprocessing.get_context("spawn")
        self.p1, self.p2 = ctx.Pipe()
        self.out1, self.out2 = ctx.Pipe(False)
        proc = ctx.Process(target=target, args=(self.p2, self.out2) + tuple(args), name=name)
        proc.daemon = True
        proc.start()
        # The child has its own copies now.  Holding ours kept the pipes open
        # after the worker died, so ``p1`` never saw EOF and a dead worker was
        # noticed only by the ``is_alive()`` poll.
        self.p2.close()
        self.out2.close()
        return proc

    def _bridge_stop_worker(self) -> None:
        """Terminate the current worker and close our pipe ends before a respawn.

        A failed teardown is only logged.  Closing here, not at garbage
        collection, keeps a run with many restarts from piling up open pipes.
        """
        try:
            terminate_process(self._bridge_process())
        except Exception as exc:
            self.logger.debug("%s: subprocess teardown on restart failed: %s" % (self.name, exc))
        for end in (getattr(self, "p1", None), getattr(self, "out1", None)):
            if end is None:
                continue
            try:
                end.close()
            except Exception:
                pass

    def _bridge_box_bounds(self) -> List[Tuple[float, float]]:
        """The feasible box as a list of ``(low, high)`` tuples (SciPy's ``bounds``)."""
        return [tuple(row) for row in self.problem.box.box]

    @staticmethod
    def _bridge_x0(center: Any, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Start point of a (re)spawned worker: ``center`` clipped into the box, else the box centre.

        Clipped because a restart centre may sit on or past the boundary,
        which the solvers tolerate only as equality.
        """
        if center is None:
            return np.array([(low + high) / 2.0 for low, high in bounds], dtype=float)
        lo = np.asarray([b[0] for b in bounds], dtype=float)
        hi = np.asarray([b[1] for b in bounds], dtype=float)
        return np.clip(np.asarray(center, dtype=float), lo, hi)

    def _request_restart(self, center: Any) -> None:
        """Record a restart; safe to call from an event handler.

        The last request before the next :meth:`produce` wins.
        """
        if self._stopped:
            return
        with self._restart_lock:
            self._restart_requests += 1
            self._pending_restart = (center, self._restart_requests)

    def _apply_pending_restart(self) -> None:
        """Perform a recorded restart.  Main thread only (see :meth:`produce`)."""
        with self._restart_lock:
            pending, self._pending_restart = self._pending_restart, None
        if pending is None or self._stopped:
            return
        self.clear_output()
        self._restart_index = pending[1]
        try:
            self._bridge_respawn(pending[0])
        except Exception as exc:
            self.logger.warning("%s: worker restart failed: %s" % (self.name, exc))
            return
        # A fresh worker owes us nothing and we owe it nothing: drop any value
        # the old one's last point produced.
        self._bridge_reset()

    def _bridge_alive(self) -> bool:
        proc = self._bridge_process()
        return proc is not None and proc.is_alive()

    def _bridge_reset(self) -> None:
        """Forget the in-flight round trip; call after respawning a worker."""
        with self._handoff_lock:
            self._outstanding_x = None
            while True:
                try:
                    self._fx_inbox.get_nowait()
                except Empty:
                    break
            self._outstanding = False
        self._bridge_done = False

    # -- the Heuristic contract -------------------------------------------

    @property
    def active(self) -> bool:
        """A finished worker drops out of the rotation until a restart revives it."""
        if self._output.qsize() > 0:
            return True
        if self._stopped:
            return False
        if self._pending_restart is not None:
            return True
        if self._bridge_done:
            return False
        return super().active

    @property
    def can_produce(self) -> bool:
        if self.has_points:
            return True
        if self._stopped:
            return False
        if self._pending_restart is not None:
            return True  # produce() will respawn the worker
        if self._bridge_done:
            return False
        if not self._bridge_alive():
            return False
        if self._outstanding:
            # We owe the worker a value; it can only move once we have one.
            return not self._fx_inbox.empty()
        return self._bridge_pending_request()

    def produce(self, limit: Optional[int] = None, timeout: Optional[float] = None) -> List["Point"]:
        self._apply_pending_restart()
        if self.has_points:
            return self.get_points(limit)
        if self._stopped or self._bridge_done:
            return []
        if not self._bridge_alive():
            # Never spawned, or the solver converged / hit its cap / died.
            self._bridge_finished("worker is not running")
            return []
        if self._outstanding:
            try:
                fx = self._fx_inbox.get_nowait()
            except Empty:
                return []  # the evaluation of our last point has not landed yet
            try:
                self._bridge_send_fx(fx)
            except (EOFError, OSError):
                self._bridge_finished("pipe closed while answering f(x)")
                return []
            self._outstanding = False
        return self._bridge_next_point(self.bridge_timeout if timeout is None else timeout, limit)

    def on_new_results(self, results: List["Result"]) -> None:
        """Store the penalty value of *our* point; never touch the pipe here.

        This runs on the event-bus dispatcher thread, which
        :class:`EventBus` delivers serially — so it must return promptly and
        must not do I/O.  :meth:`produce` sends the value on, from the main
        thread.
        """
        if not self._outstanding:
            return
        x_out = self._outstanding_x
        for result in results:
            if result.who != self.name or x_out is None:
                continue
            # ``equal_nan``: a NaN coordinate must still match, or the bridge
            # would wait forever for a value that already arrived.
            if not np.array_equal(np.asarray(result.x, dtype=float), x_out, equal_nan=True):
                # Ours by name, but not the point the worker is waiting on: a
                # leftover of a replaced worker or an aborted descent.
                continue
            value = self.strategy.constraint_handler.get_penalty_value(result)
            with self._handoff_lock:
                # Re-check under the lock: a reset (restart, abort) since the
                # match above makes this value stale.
                if self._outstanding_x is not x_out:
                    return
                self._outstanding_x = None  # answer it exactly once
                self._fx_inbox.put(value)
            return

    def on_failed_evaluations(self, points: List["Point"]) -> None:
        """Answer our outstanding point with ``inf`` when its evaluation failed.

        Without an answer the worker waits for a value that never comes:
        ``can_produce`` stays ``False`` and the arm is dead for the rest of
        the run, silently.  ``inf`` is what :func:`pipe_objective` makes of
        any non-finite value.  Same thread and locking rules as
        :meth:`on_new_results`.
        """
        if not self._outstanding:
            return
        x_out = self._outstanding_x
        for point in points:
            if point.who != self.name or x_out is None:
                continue
            if not np.array_equal(np.asarray(point.x, dtype=float), x_out, equal_nan=True):
                continue
            self.logger.warning("%s: the evaluation of its point failed; answering inf" % self.name)
            with self._handoff_lock:
                if self._outstanding_x is not x_out:
                    return
                self._outstanding_x = None
                self._fx_inbox.put(float("inf"))
            return

    # -- internals ---------------------------------------------------------

    def _bridge_next_point(self, timeout: float, limit: Optional[int]) -> List["Point"]:
        waited = 0.0
        slice_ = self._bridge_poll_slice
        while True:
            self._bridge_drain_status()
            if not self._bridge_pending_request():
                return []  # the worker is idle; there is nothing to wait for
            try:
                ready = self.p1.poll(slice_)
            except (EOFError, OSError):
                self._bridge_finished("request pipe closed")
                return []
            if not ready:
                if not self._bridge_alive():
                    # The worker exited.  Everything it wrote before exiting is
                    # already in the pipe buffer, so one non-blocking poll is
                    # conclusive: nothing waiting means it converged, hit its
                    # evaluation cap or died.  A deterministic end, not a
                    # timeout.
                    try:
                        leftover = self.p1.poll(0)
                    except (EOFError, OSError):
                        leftover = False
                    if not leftover:
                        self._bridge_finished("worker finished")
                        return []
                    continue
                waited += slice_
                if waited >= timeout:
                    self.logger.error(
                        "%s: no request from the worker for %.0fs; treating it as wedged. "
                        "This is a bug, not a slow problem." % (self.name, waited)
                    )
                    self._bridge_finished("worker wedged")
                    return []
                continue
            try:
                msg = self.p1.recv()
            except (EOFError, OSError):
                self._bridge_finished("worker closed the request pipe")
                return []
            if self._bridge_control(msg):
                continue
            x = self._bridge_point(msg)
            # ``emit`` projects the point; the result will carry that ``x``.
            # Recorded before the point can be dispatched for evaluation.
            with self._handoff_lock:
                self._outstanding_x = np.asarray(self.problem.project(x), dtype=float)
            self.emit(x)
            self._outstanding = True
            return self.get_points(limit)

    def _bridge_drain_status(self) -> None:
        out = getattr(self, "out1", None)
        if out is None:
            return
        try:
            while out.poll(0):
                self.logger.info(out.recv())
        except (EOFError, OSError):
            pass

    def _bridge_finished(self, reason: str) -> None:
        # Only ``_bridge_done``: ``_stopped`` is the run's end, and a finished
        # worker (a converged COBYQA, an L-BFGS-B at ``max_starts``) must
        # still honour a later restart.
        self._bridge_done = True
        self.logger.debug("%s: bridge finished (%s)" % (self.name, reason))


#
# Analyzer
#
class Analyzer(Module):
    """
    Abstract parent class for all types of analyzers.
    """

    def __init__(self, strategy: "StrategyBase", name: Optional[str] = None) -> None:
        Module.__init__(self, strategy, name)


#
# EventBus
#


class Event:
    """
    This class holds the data for one single :class:`~.EventBus` event.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._when: float = time_module.time()
        self.terminate: bool = False
        self._kwargs: Dict[str, Any] = kwargs
        for k, v in list(kwargs.items()):
            setattr(self, k, v)

    def __repr__(self) -> str:
        return "Event[%s]" % self._kwargs


class EventBus:
    """
    Publish/subscribe bus connecting the strategy, its analyzers and its
    heuristics.  E.g. the "new best point" information is sent to every
    subscribing heuristic.

    Delivery is **serial and ordered**: one dispatcher thread pops events
    from a single FIFO and calls the subscribers' ``on_<key>`` handlers one
    after the other, in subscription order.  Events published *by* a handler
    are appended to the same FIFO.  Consequently

    * a module never has two handlers running at the same time, so module
      state needs no locking;
    * the sequence of handler invocations is a pure function of the
      sequence of ``publish`` calls — the property the seeded, reproducible
      runs rely on (see :meth:`wait_idle`);
    * handlers must return promptly.  A module *reacts* to events; it never
      sleeps, polls or waits for other events inside a handler.
    """

    # pattern for a valid key
    _re_key = re.compile(r"^[a-z_]+$")

    def __init__(self, config: Any) -> None:
        self._subs: Dict[str, List[Any]] = {}
        self.config = config
        self.logger = config.get_logger("EVBUS")
        self._queue: "collections.deque[Tuple[str, Any, Event]]" = collections.deque()
        self._cv = Condition()
        # queued + currently running events; ``wait_idle`` waits for zero.
        self._inflight: int = 0
        self._running: bool = True
        #: The dispatcher starts with the first published event, so a bus
        #: (i.e. a strategy) that is constructed but never started owns no
        #: thread — it used to leak one per unstarted strategy.
        self._thread: Optional[Thread] = None

    @property
    def keys(self) -> List[str]:
        """
        List of all keys where you can send an :class:`Event` to.
        """
        return list(self._subs.keys())

    # -- registration ------------------------------------------------------

    def register(self, target: Any) -> None:
        """
        Registers a given ``target`` for this EventBus instance: every
        ``on_<key>`` method it has becomes a subscription to ``key``.

        :param Module target:
        """
        for name, _ in inspect.getmembers(target, predicate=inspect.ismethod):
            if name.startswith("on_"):
                self.subscribe(self._check_key(name[3:]), target)

    @staticmethod
    def _check_key(key: str) -> str:
        if not EventBus._re_key.match(key):
            raise ValueError('"%s" key not allowed' % key)
        return key

    def subscribe(self, key: str, target: Any) -> None:
        """
        Called by :meth:`.register`.

        .. Note:: counterpart is :func:`unsubscribe`.
        """
        self._check_key(key)
        with self._cv:
            if key not in self._subs:
                self._subs[key] = []
            assert target not in self._subs[key]
            self._subs[key].append(target)

    def unsubscribe(self, key: Optional[str], target: Any) -> None:
        """
        Args:

        - if ``key`` is ``None``, the target is removed from all keys.

        """
        if key is None:
            with self._cv:
                for v in list(self._subs.values()):
                    if target in v:
                        v.remove(target)
            return

        self._check_key(key)
        with self._cv:
            if key not in self._subs:
                self.logger.critical("cannot unsubscribe unknown key '%s'" % key)
                return
            if target in self._subs[key]:
                self._subs[key].remove(target)

    def is_subscribed(self, target: Any) -> bool:
        """``True`` iff ``target`` still listens to at least one key."""
        with self._cv:
            return any(target in v for v in self._subs.values())

    # -- publishing / dispatching ----------------------------------------------

    def publish(self, key: str, event: Optional[Event] = None, terminate: bool = False, **kwargs: Any) -> None:
        """
        Publishes a new :class:`.Event` to all subscribers,
        who listen to the given ``key``.
        It is either possible to send an existing event or to create an event
        object on the fly with the given ``**kwargs``.  Every subscriber gets
        its own :class:`Event` instance.

        Args:

        - ``event``: if set, this given :class:`.Event`'s payload is sent (and not a new one created).
        - ``terminate``: if True, the subscription of ``key`` ends after this event
                         (use it for ``on_start`` and similar one-shot lifecycle events).
        - ``**kwargs``: any additional keyword arguments are stored inside the Event
                        if ``event`` is ``None``.
        """
        if key not in self._subs:
            if self.config.debug:
                self.logger.warning("key '%s' unknown." % key)
            return
        payload = dict(kwargs) if event is None else dict(event._kwargs)
        with self._cv:
            if self._thread is None and self._running:
                self._thread = Thread(target=self._loop, name="EventBus", daemon=True)
                self._thread.start()
            for target in list(self._subs[key]):
                ev = Event(**payload)
                ev.terminate = terminate
                self._queue.append((key, target, ev))
                self._inflight += 1
            self._cv.notify_all()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._queue and self._running:
                    self._cv.wait()
                if not self._queue:
                    return  # shut down and drained
                key, target, event = self._queue.popleft()
            try:
                self._dispatch(key, target, event)
            finally:
                with self._cv:
                    self._inflight -= 1
                    if self._inflight == 0:
                        self._cv.notify_all()

    def _dispatch(self, key: str, target: Any, event: Event) -> None:
        with self._cv:
            subscribed = target in self._subs.get(key, [])
        if not subscribed:
            return  # unsubscribed after the event was queued
        try:
            handler = getattr(target, "on_%s" % key)
            if event.terminate:
                # A one-shot lifecycle event (``start``, ``finished``) goes to
                # every module that has the handler, whatever its signature;
                # one that cannot take this payload is skipped.  Checked
                # *before* the call, so a TypeError raised inside the handler
                # body is reported like any other exception instead of being
                # mistaken for a signature mismatch and swallowed.
                try:
                    inspect.signature(handler).bind(**event._kwargs)
                except TypeError:
                    raise StopHeuristic("signature does not accept %s" % sorted(event._kwargs))
                except ValueError:
                    pass  # no introspectable signature: just call it
            new_points = handler(**event._kwargs)
            # heuristics might call self.emit and/or return a list
            if new_points is not None:
                target.emit(new_points)
            if event.terminate:
                raise StopHeuristic("%s terminated" % target.name)
        except StopHeuristic as e:
            self.logger.debug("'%s/on_%s' %s -> unsubscribing." % (target.name, key, str(e)))
            self.unsubscribe(key, target)
        except Exception as e:
            # A failing handler must not take the dispatcher down; report
            # loudly and keep serving the other modules.
            self.logger.critical("Exception in %s/on_%s: %r" % (target, key, e), exc_info=True)
            if event.terminate:
                self.unsubscribe(key, target)  # a one-shot subscription ends either way

    @property
    def inflight(self) -> int:
        """Events queued plus the one currently being dispatched.

        A non-zero value means a handler is about to run (or is running),
        i.e. a heuristic may still refill its queue without any new
        evaluation.  :meth:`StrategyBase._alive` needs exactly this: the
        difference between "every queue is empty" and "nothing can ever
        produce again".
        """
        with self._cv:
            return self._inflight

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """Block until every published event has been handled.

        Handlers may publish further events (``new_results`` → ``new_best`` →
        ``on_new_best`` ...); the whole cascade counts.  Returns ``True`` when
        the bus is idle, ``False`` on timeout.  The synchronous evaluation mode
        calls this before every draw so each main-loop pass sees the fully
        updated state of all analyzers and heuristics.
        """
        with self._cv:
            return self._cv.wait_for(lambda: self._inflight == 0, timeout=timeout)

    def shutdown(self, timeout: float = 2.0) -> None:
        """Deliver what is still queued (bounded by ``timeout``), then stop the
        dispatcher thread and drop all subscriptions."""
        self.wait_idle(timeout=timeout)
        with self._cv:
            self._running = False
            # Events dropped on a timeout will never run: take them out of
            # the in-flight count, so ``inflight``/``wait_idle`` do not report
            # (or wait for) work that no longer exists.
            self._inflight -= len(self._queue)
            self._queue.clear()
            self._subs.clear()
            self._cv.notify_all()
        t = self._thread
        if t is not None and t.is_alive() and threading.current_thread() is not t:
            t.join(timeout=timeout)


#: The default analyzers :meth:`StrategyBase.initialize` installs, in
#: construction order.  Their RNG streams are keyed by name
#: (:meth:`StrategyBase.spawn_rng`), so which of them is built — the
#: ``Splitter`` only on demand — changes no other module's randomness.
_DEFAULT_ANALYZER_ORDER: Tuple[str, ...] = ("Best", "Splitter", "Convergence")

#: The analyzers of :data:`_DEFAULT_ANALYZER_ORDER`, as a set.
_DEFAULT_ANALYZERS = frozenset(_DEFAULT_ANALYZER_ORDER)


def _default_analyzer_classes() -> Dict[str, "type[Analyzer]"]:
    """The classes behind :data:`_DEFAULT_ANALYZERS`, imported by module path.

    Resolved explicitly rather than by ``getattr(panobbgo.analyzers, name)``,
    which silently built nothing once a class was renamed or dropped from
    the package namespace.  Imported lazily: the analyzers import this module.
    """
    from .analyzers.best import Best
    from .analyzers.convergence import Convergence
    from .analyzers.splitter import Splitter

    classes: Dict[str, "type[Analyzer]"] = {"Best": Best, "Splitter": Splitter, "Convergence": Convergence}
    assert set(classes) == _DEFAULT_ANALYZERS
    assert all(cls.__name__ == name for name, cls in classes.items())
    return classes


def _config_keys(config: Any) -> set:
    """Public, non-callable attributes of a :class:`Config` — the settable keys."""
    return {k for k, v in vars(config).items() if not k.startswith("_") and not callable(v)}


class _DirectEvaluators:
    """View of the in-process evaluation backend (threads or subprocesses).

    Mirrors the interface of :class:`panobbgo.dask_evaluation.DaskEvaluators`
    so strategies can ask "how many workers, how many tasks in flight" without
    knowing which backend they run on.
    """

    def __init__(self, strategy: "StrategyBase") -> None:
        self.strategy = strategy

    @property
    def outstanding(self) -> List[Any]:
        """Keys of the tasks currently in flight."""
        return list(self.strategy.pending.keys())

    def __len__(self) -> int:
        """Number of workers."""
        return getattr(self.strategy, "_n_processes", 1)


class StrategyBase:
    """
    This abstract BaseStrategy is the parent class of all Strategies.

    Use it this way:

    #. Subclass it, write your optional initializer, *afterwards* call the initializer
       of this class (it will start its the main loop).

    #. Overwrite the :meth:`.execute`, which returns a list of new search points
       (by requesting them from the :mod:`~panobbgo.heuristics` via the
       :meth:`~panobbgo.core.Heuristic.get_points` method) and might
       also emit :class:`Events <panobbgo.core.Event>`.

    This ``execute`` method will be called repeatedly as long as there are less than the
    given maximum number of search points evaluated.
    """

    # constant reference id for sending the evaluation code to workers
    PROBLEM_KEY = "problem"

    # Dask backend state — assigned lazily by panobbgo.dask_evaluation when
    # evaluation_method == "dask" (and directly by some tests); declared here
    # for static checkers only.
    _client: Any
    _cluster: Any
    _problem_future: Any

    def __init__(self, problem, parse_args=False, testing_mode=False, **kwargs):
        """


        @type problem: panobbgo.lib.Problem
        @param problem:
        @param parse_args:
        @param testing_mode:
        @param kwargs: Additional configuration parameters (e.g. max_eval, max_evaluations)
        """
        self._name = name = self.__class__.__name__
        self.config = config = Config(parse_args, testing_mode=testing_mode)

        # Handle configuration overrides from kwargs
        if "max_evaluations" in kwargs:
            self.config.max_eval = kwargs.pop("max_evaluations")
        if "max_eval" in kwargs:
            self.config.max_eval = kwargs.pop("max_eval")

        # Master seed.  Precedence: explicit ``seed=`` kwarg, then
        # ``config.seed``, then a draw from numpy's global RNG (so a
        # preceding ``np.random.seed(s)`` still makes the run reproducible).
        seed = kwargs.pop("seed", None)
        if seed is None:  # not ``or``: seed=0 is a valid seed
            seed = self.config.seed
        self.seed: int = int(seed) if seed is not None else int(np.random.randint(0, 2**31 - 1))
        #: Strategy-level generator (Thompson sampling, phase draws, ...):
        #: its own keyed stream of :attr:`seed`, independent of the module
        #: streams :meth:`spawn_rng` hands out, so neither shifts the other.
        self.rng: np.random.Generator = keyed_rng(self.seed, _STRATEGY_STREAM_KEY)
        #: How many module streams each name has received (:meth:`spawn_rng`).
        self._rng_key_counts: Dict[str, int] = {}

        # Remaining kwargs override config attributes; anything else is a
        # typo (``max_evals=``) that used to be dropped silently.
        valid = _config_keys(self.config)
        unknown = sorted(k for k in kwargs if k not in valid)
        if unknown:
            raise TypeError(
                "%s got unexpected keyword argument(s) %s. Valid: max_eval, max_evaluations, seed, "
                "the strategy's own parameters, or a config attribute: %s"
                % (name, ", ".join(unknown), ", ".join(sorted(valid)))
            )
        for k, v in kwargs.items():
            setattr(self.config, k, v)

        self.logger = logger = config.get_logger("STRAT")
        self.slogger = config.get_logger("STATS")
        logger.info("Init of '%s'" % (name))
        logger.info("%s" % problem)

        # aux configuration
        # determine width based on console info
        pd.set_option("display.width", None)
        pd.set_option("display.precision", 2)  # default 7

        # statistics
        self.show_last = 0.0  # for throttling the info line (see dask_evaluation._add_tasks)
        self._last_status_update = 0  # for throttling _update_progress_status
        self.time_start = time_module.time()
        # Running count and sum of task walltimes (see avg_time_per_task).
        self._walltime_n = 0
        self._walltime_sum = 0.0

        # task accounting (tasks != points !!!)
        self.jobs_per_client = 1  # number of tasks per client in 'chunksize'
        self.pending = {}  # dict mapping future id to future object
        self.new_finished = []
        self.n_finished = 0  # number of finished tasks

        # init & start everything
        self._setup_cluster(problem)
        self._threads = []
        self._hs = []

        self._heuristics = collections.OrderedDict()
        self._analyzers = collections.OrderedDict()
        self.problem = problem
        self._stop_requested = False
        self._dispatched = 0  # evaluations charged against max_eval (see _clamp_to_budget)

        # Configure Constraint Handler.  A setting left unset (``None``) is
        # not passed, so each handler keeps its own class default (see the
        # ``constraints.*`` keys in panobbgo/config.py).
        def _set(**kw: Any) -> Dict[str, Any]:
            return {k: float(v) for k, v in kw.items() if v is not None}

        rho = getattr(config, "rho", None)
        exponent = getattr(config, "constraint_exponent", None)
        handler_name = getattr(config, "constraint_handler", "DefaultConstraintHandler")

        if handler_name == "PenaltyConstraintHandler":
            self.constraint_handler = PenaltyConstraintHandler(strategy=self, **_set(rho=rho, exponent=exponent))
        elif handler_name == "DynamicPenaltyConstraintHandler":
            self.constraint_handler = DynamicPenaltyConstraintHandler(
                strategy=self,
                **_set(rho_start=rho, rate=getattr(config, "dynamic_penalty_rate", None), exponent=exponent),
            )
        elif handler_name == "AugmentedLagrangianConstraintHandler":
            self.constraint_handler = AugmentedLagrangianConstraintHandler(
                strategy=self, **_set(rho=rho, rate=getattr(config, "alm_rate", None))
            )
        elif handler_name == "EpsilonConstraintHandler":
            epsilon_start = float(config.epsilon_start) if hasattr(config, "epsilon_start") else 1.0
            epsilon_cp = float(config.epsilon_cp) if hasattr(config, "epsilon_cp") else 5.0
            epsilon_cutoff = int(config.epsilon_cutoff) if hasattr(config, "epsilon_cutoff") else 100

            self.constraint_handler = EpsilonConstraintHandler(
                strategy=self,
                epsilon_start=epsilon_start,
                cp=epsilon_cp,
                cutoff=epsilon_cutoff,
                **_set(rho=rho),
            )
        elif handler_name == "FilterConstraintHandler":
            self.constraint_handler = FilterConstraintHandler(strategy=self)
        else:
            self.constraint_handler = DefaultConstraintHandler(strategy=self, **_set(rho=rho))

        self.eventbus = EventBus(config)
        self.eventbus.register(self.constraint_handler)
        self.results = Results(self)

        # Initialize new logging system
        self.panobbgo_logger = PanobbgoLogger(
            cast(dict[str, Any], config.logging) if hasattr(config, "logging") else {}
        )

    def __enter__(self):
        """
        Context manager entry. Allows using the strategy in a 'with' block.
        Example:
            with StrategyRewarding(problem) as strategy:
                strategy.start()
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit. Ensures resource cleanup even if exceptions occur.
        """
        self._cleanup()

    def add(self, Heur, **kwargs):
        self.logger.debug("init: %s %s" % (Heur.__name__, kwargs))
        self._hs.append(Heur(self, **kwargs))

    def initialize(self):
        """
        Initialize the strategy components (heuristics, analyzers) without running the optimization loop.
        Useful for manual control or custom runners (e.g. benchmarks).
        """
        # heuristics
        for h in sorted(self._hs, key=lambda h: h.name):
            self.add_heuristic(h)

        # Default analyzers: ``Best`` and ``Convergence`` always, the
        # ``Splitter`` only when a module declares it (``requires_analyzers``).
        # Module RNG streams are keyed by name, so building one or not
        # changes no other module's randomness.
        default_classes = _default_analyzer_classes()
        needed = set(self._required_analyzers())
        new_analyzers = []
        for name in _DEFAULT_ANALYZER_ORDER:
            if name not in self._analyzers and (name != "Splitter" or name in needed):
                new_analyzers.append(default_classes[name](self))
        for a in new_analyzers:
            self.add_analyzer(a)

        self.check_dependencies()

        # Validate framework setup before starting optimization
        self.validate_setup()

        self._ensure_cluster()

        # Load previous results if storage is enabled
        if hasattr(self.results, "load_from_storage"):
            self.results.load_from_storage()

        self.logger.debug("EventBus keys: %s" % self.eventbus.keys)

        # Prepare for execution
        self.eventbus.publish("start", terminate=True)
        # Let the heuristics process the start event and emit their initial
        # points.  Under ``sync_evaluation`` this must not be bounded by a
        # clock: a slow ``on_start`` (a GP prior, a large LHS design) that
        # timed out here silently changed which points the run began with.
        self.eventbus.wait_idle(timeout=None if self.config.sync_evaluation else 5.0)
        self._start = time_module.time()
        self.eventbus.register(self)
        self.logger.info("Strategy '%s' initialized" % self._name)

    def start(self):
        try:
            # Inside the ``try``: a failing initialize() (validate_setup, a
            # dependency check) must still stop what the constructor and
            # add() started — e.g. a HeuristicSubprocess worker.
            self.initialize()
            if isinstance(self, threading.Thread):
                raise Exception("change run() to start()")
            self._run()
        except KeyboardInterrupt:
            self.logger.critical("KeyboardInterrupt received, e.g. via Ctrl-C")
        finally:
            # Every exit path cleans up.  Before this ``finally`` only
            # ``KeyboardInterrupt`` did, so any other exception escaping
            # ``_run`` — the ``ZeroDivisionError`` of F1, a heuristic raising
            # inside ``execute`` — leaked the event-bus dispatcher thread, the
            # evaluator pool, the results store and every heuristic's
            # subprocess.  ``_cleanup`` is idempotent, so the normal path
            # (which calls it at the end of ``_run``) is unaffected.
            self._cleanup()

    @property
    def heuristics(self):
        return [h for h in list(self._heuristics.values()) if h.active]

    @property
    def analyzers(self):
        return list(self._analyzers.values())

    def budget_progress(self) -> Optional[float]:
        """``len(self.results) / max_eval`` in ``[0, 1]``, ``None`` if the budget is unknown (:meth:`Module.budget_progress`)."""
        return budget_progress_of(self.config.max_eval, lambda: len(self.results))

    def max_eval_or(self, default: int) -> int:
        """The evaluation budget as an ``int``, or *default* when it is unknown (:meth:`Module.max_eval_or`)."""
        return max_eval_or_default(self.config.max_eval, default)

    def spawn_rng(self, key: str = "") -> np.random.Generator:
        """Return the next generator for ``key`` derived from the master seed.

        The stream of the *n*-th call with a given ``key`` is
        ``SeedSequence(seed, spawn_key=(crc32(key), n))`` — a function of
        :attr:`seed`, ``key`` and ``n`` only.  :class:`Module` passes its
        name, so a module's randomness does not change when an unrelated
        module is added, removed or reordered, and two instances of one class
        (same name) still get distinct streams.  It never draws from
        :attr:`rng`, the strategy-level stream.
        """
        # ``setdefault``: subclasses that skip ``__init__`` (test doubles) still work.
        counts: Dict[str, int] = self.__dict__.setdefault("_rng_key_counts", {})
        n = counts.get(key, 0)
        counts[key] = n + 1
        return keyed_rng(self.seed, rng_stream_key(key, n))

    def heuristic(self, who):
        """Look up a heuristic by its *who* tag.

        Some heuristics (e.g. :class:`~panobbgo.heuristics.CMAES`,
        :class:`~panobbgo.heuristics.DifferentialEvolution`) embed extra
        context in the ``who`` field (e.g. ``"CMAES:g3:i0"``).  We first try
        the full string, then fall back to the prefix before the first ``:``.
        """
        if who in self._heuristics:
            return self._heuristics[who]
        base = who.split(":")[0]
        if base in self._heuristics:
            return self._heuristics[base]
        raise KeyError(who)

    def analyzer(self, who):
        try:
            return self._analyzers[who]
        except KeyError:
            raise KeyError(
                f"analyzer {who!r} is not installed; a module that reads it must list it in "
                f"its ``requires_analyzers`` (the Splitter is only installed on demand)"
            ) from None

    def add_heuristic(self, h):
        """
        Add a heuristic to the strategy.

        :param Heuristic h: The heuristic instance to add
        """
        name = h.name
        if name in self._heuristics:
            existing_names = list(self._heuristics.keys())
            raise ValueError(
                f"Heuristic name '{name}' is already used by another heuristic. "
                f"Each heuristic must have a unique name. "
                f"Existing heuristics: {existing_names}"
            )
        self._heuristics[name] = h
        try:
            self.init_module(h)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize heuristic '{name}' ({h.__class__.__name__}). "
                f"This usually indicates an error in the heuristic's __start__() method. "
                f"Original error: {e}"
            ) from e

    def add_analyzer(self, a):
        """
        Add an analyzer to the strategy.

        :param Analyzer a: The analyzer instance to add
        """
        name = a.name
        if name in self._analyzers:
            existing_names = list(self._analyzers.keys())
            raise ValueError(
                f"Analyzer name '{name}' is already used by another analyzer. "
                f"Each analyzer must have a unique name. "
                f"Existing analyzers: {existing_names}"
            )
        self._analyzers[name] = a
        try:
            self.init_module(a)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize analyzer '{name}' ({a.__class__.__name__}). "
                f"This usually indicates an error in the analyzer's __start__() method. "
                f"Original error: {e}"
            ) from e

    #: Analyzer names the strategy itself reads (see :attr:`Module.requires_analyzers`).
    requires_analyzers: Tuple[str, ...] = ()

    def _required_analyzers(self) -> List[str]:
        """Union of the analyzer names declared by the strategy and every registered module."""
        names: List[str] = list(self.requires_analyzers)
        for m in list(self._heuristics.values()) + list(self._analyzers.values()):
            req = getattr(m, "required_analyzers", None)
            if callable(req):
                for n in req():
                    if n not in names:
                        names.append(n)
        return names

    def check_dependencies(self):
        """
        This method is called in self.start() (and only there)
        for checking all the dependencies of all modules.
        """
        heuristics = list(self._heuristics.values())
        analyzers = list(self._analyzers.values())
        all_mods = [m.__class__ for m in heuristics]
        all_mods.extend([m.__class__ for m in analyzers])
        all_mods = set(all_mods)
        for module in analyzers + heuristics:
            # explicit
            if not module.check_dependencies(analyzers, heuristics):
                raise Exception("%s does not satisfy dependencies. #1" % module)
            # implicit (just list of respective classes)
            for mod_class in module._depends_on:
                if mod_class not in all_mods:
                    raise Exception("%s depends on %s, but missing." % (module, mod_class))

    def validate_setup(self):
        """
        Validate that the framework is properly set up before starting optimization.

        This method checks for common configuration errors and missing components
        that would cause the optimization to fail or behave unexpectedly.

        Raises:
            ValueError: If the setup is invalid with descriptive error messages.
        """
        errors = []

        # Check for at least one heuristic
        active_heuristics = [h for h in self._heuristics.values() if h.active]
        if len(active_heuristics) == 0:
            errors.append(
                "No active heuristics found. You must add at least one heuristic before starting optimization.\n"
                "Example: strategy.add(Random)\n"
                "Available heuristics: Center, Zero, Random, Extremal, LatinHypercube, Nearby, WeightedAverage, NelderMead, LBFGSB, QuadraticWlsModel, FeasibleSearch, ConstraintGradient, LocalPenaltySearch, GaussianProcessHeuristic"
            )

        # Check for required analyzers
        required_analyzers = ["Best", "Convergence"] + sorted(set(self._required_analyzers()) - {"Best", "Convergence"})
        missing_analyzers = []
        for analyzer_name in required_analyzers:
            if analyzer_name not in self._analyzers:
                missing_analyzers.append(analyzer_name)

        if missing_analyzers:
            errors.append(
                f"Missing required analyzers: {', '.join(missing_analyzers)}.\n"
                "Best, Convergence and (when a module declares it in ``requires_analyzers``) Splitter "
                "are added by StrategyBase.initialize() or StrategyBase.start(); any other required "
                "analyzer has to be added with strategy.add_analyzer() before starting."
            )

        # Check configuration validity
        config_errors = self._validate_config()
        errors.extend(config_errors)

        # If any errors found, raise ValueError with all issues
        if errors:
            error_msg = "Framework setup validation failed:\n\n"
            error_msg += "\n\n".join(f"• {error}" for error in errors)
            error_msg += "\n\nPlease fix these issues before starting optimization."
            raise ValueError(error_msg)

    def _validate_config(self):
        """
        Validate configuration parameters.

        Returns:
            List of error messages (empty if no errors).
        """
        errors = []

        # Check max_eval is reasonable
        try:
            max_eval = int(self.config.max_eval)
            if max_eval <= 0:
                errors.append(f"max_eval must be positive, got {max_eval}")
            elif max_eval > 100000:
                # Unusual for an expensive black box, but not invalid.
                self.logger.warning(f"max_eval ({max_eval}) is very large for expensive black-box optimisation.")
        except (ValueError, TypeError):
            errors.append(f"max_eval must be a valid integer, got {self.config.max_eval}")

        # Check discount factor (used by rewarding strategy)
        try:
            discount = float(self.config.discount)
            if not (0 < discount <= 1):
                errors.append(f"discount must be between 0 and 1, got {discount}")
        except (ValueError, TypeError):
            errors.append(f"discount must be a valid float between 0 and 1, got {self.config.discount}")

        # Check smoothing parameter
        try:
            smooth = float(self.config.smooth)
            if smooth < 0:
                errors.append(f"smooth must be non-negative, got {smooth}")
        except (ValueError, TypeError):
            errors.append(f"smooth must be a valid float, got {self.config.smooth}")

        # Check evaluation method
        valid_methods = ["threaded", "processes", "dask"]
        if self.config.evaluation_method not in valid_methods:
            errors.append(f"evaluation_method must be one of {valid_methods}, got '{self.config.evaluation_method}'")

        return errors

    def init_module(self, module):
        """
        :class:`.StrategyBase` calls this method.

        :param Module module:
        """
        module.__start__()
        # only after _init_ it is ready to receive events
        module.eventbus.register(module)

    def _setup_cluster(self, problem):
        """
        Set up evaluation infrastructure based on configuration.

        Evaluation methods:
        - 'threaded': Thread pool for fast testing with pure Python functions (default for tests)
        - 'processes': Spawned worker-process pool for isolation (picklable problem)
        - 'dask': Distributed evaluation for heavy workloads
        """
        if self.config.evaluation_method == "dask":
            # Dask is fully isolated in its own module and imported lazily —
            # see panobbgo/dask_evaluation.py
            from . import dask_evaluation

            dask_evaluation.setup_cluster(self, problem)
        elif self.config.evaluation_method == "processes":
            self._setup_process_evaluation(problem)
        elif self.config.evaluation_method == "threaded":
            self._setup_threaded_evaluation(problem)
        else:
            raise ValueError(f"Unknown evaluation method: {self.config.evaluation_method}")
        self._pool_method = self.config.evaluation_method

    def _ensure_cluster(self):
        """Rebuild the evaluation backend if ``evaluation_method`` changed after construction.

        ``__init__`` sets the backend up from the config it was given; callers
        (tests, the harness) often set ``config.evaluation_method`` on the
        constructed strategy.  Without this the old backend kept evaluating
        while the main loop routed by the new name.
        """
        old = getattr(self, "_pool_method", None)
        if old == self.config.evaluation_method:
            return
        if old == "dask":
            from . import dask_evaluation

            dask_evaluation.close(self)
        elif getattr(self, "_pool", None) is not None:
            self._pool.close(time_module.time())
        self._setup_cluster(self.problem)

    def _setup_process_evaluation(self, problem):
        """Set up a spawn-context process pool (``evaluation_method="processes"``).

        Each worker is a fresh interpreter (``sys.executable``, ``spawn``
        start method) that receives the pickled problem once.  Submission,
        harvesting and budget accounting are those of the threaded path; see
        :mod:`panobbgo.local_pool` for timeouts (killed), crashing workers
        (the task fails, the run goes on) and shutdown.  The problem must be
        picklable, a script using this mode needs the usual
        ``if __name__ == "__main__":`` guard, and state the problem object
        accumulates while evaluating lives in the worker copies.
        """
        from .local_pool import LocalPool

        self._problem = problem
        self._n_processes = int(self.config.dask_n_workers)
        self._pool = LocalPool(problem, self._n_processes, processes=True, logger=self.logger)
        self.logger.info("Process evaluation ready with %d worker processes" % self._n_processes)

    def _setup_threaded_evaluation(self, problem):
        """
        Set up thread pool evaluation for fast testing with pure Python functions.

        This is ideal for:
        - Unit tests and CI
        - Pure Python objective functions
        - Quick prototyping

        NOT suitable for:
        - External process calls
        - Functions that release the GIL
        - CPU-intensive parallel work (use Dask instead)
        """
        from .local_pool import LocalPool

        # Store problem directly (shared memory, no pickling needed)
        self._problem = problem
        self._n_processes = int(self.config.dask_n_workers)
        self._pool = LocalPool(problem, self._n_processes, processes=False, logger=self.logger)
        self.logger.info("Threaded evaluation ready with %d threads" % self._n_processes)

    @property
    def best(self):
        best_analyzer = self._analyzers.get("Best")
        return best_analyzer.best if best_analyzer else None

    @property
    def name(self):
        return self._name

    @property
    def evaluators(self):
        """
        Compatibility property for legacy code that references evaluators.
        Returns a mock object with attributes needed by strategies.
        """
        if self.config.evaluation_method == "dask":
            from . import dask_evaluation

            return dask_evaluation.DaskEvaluators(self)
        return _DirectEvaluators(self)  # process or threaded evaluation

    def _run(self):
        # Initialization logic moved to self.initialize() which is called by self.start()
        # We assume self.initialize() has been called if we are here via self.start()
        # If _run is called directly (legacy), we might miss init, but that's discouraged.

        self.logger.info("Strategy '%s' started main loop" % self._name)
        self.loops = 0
        self._last_results_count = 0
        self._dead_loops = 0
        #: Consecutive dead passes before a run is declared finished.  Under
        #: ``sync_evaluation`` one is conclusive (see :meth:`_alive`); the
        #: asynchronous path allows a small margin for the window in
        #: :meth:`_run_threaded_evaluation` where a task is submitted but not
        #: yet in ``pending``.
        self._max_dead_loops = 1 if self.config.sync_evaluation else 3
        self._last_progress_at: Optional[float] = None
        #: Wall-clock **backstop for a bug**, not a scheduling parameter: it
        #: fires only when there is nothing to wait for — no evaluation
        #: outstanding anywhere (see :meth:`_pool_progressed`) — yet
        #: :meth:`_alive` says the run can go on and nothing has changed for
        #: this long: a heuristic that claims a point but never gives one, a
        #: handler that never returns.  It never cuts a running evaluation;
        #: per-evaluation run time is limited only by ``evaluation.timeout``.
        #: See ``planning/DESIGN_pump_and_stall_2026-09-11.md`` §2.3.
        self._deadlock_seconds = float(getattr(self.config, "deadlock_seconds", 600.0))
        sync = bool(self.config.sync_evaluation)
        self._last_n_finished = self.n_finished
        #: Evaluations charged against ``max_eval``: incremented when a point
        #: is *dispatched*, so a failing evaluation (no result) is charged
        #: too.  Results restored from storage count as dispatched.
        self._dispatched = len(self.results)

        while True:
            self.loops += 1

            # No cap on the number of passes: an asynchronous run makes an
            # idle ~1 ms pass per millisecond while evaluations run, so a cap
            # of ``max_eval * 10000`` passes ended runs with slow evaluations
            # after ~10 s per budgeted evaluation, silently (a WARNING under
            # the default loglevel).  The run ends by the budget, the liveness
            # predicate or the deadlock backstop below.

            # execute the actual strategy
            # Once the whole budget is dispatched, only in-flight results
            # are awaited; asking the strategy for more points would just
            # hand them back to the heuristics every pass.
            if self.config.max_eval and self._dispatched >= self.config.max_eval:
                points = []
            else:
                points = self._clamp_to_budget(self.execute())

            # Update progress status
            self._update_progress_status()

            if self.config.evaluation_method == "dask":
                from . import dask_evaluation

                self.results += dask_evaluation.run_evaluation(self, points)
            else:  # "threaded" or "processes": same bookkeeping, different pool
                self._run_threaded_evaluation(points)

            if sync:
                # Let every analyzer / heuristic finish reacting to this batch
                # before the next draw, so the run is a deterministic function
                # of the seed rather than of thread scheduling, and size the
                # next batch without consulting wall-clock task timings.
                #
                # No timeout: a slow handler (a GP fit takes ~0.4 s) must delay
                # the run, never shorten it.  Waiting a bounded number of
                # *seconds* here was the primary half of F4 — the next pass then
                # found every queue empty, because the refills were still
                # undelivered, and the stall guard below ended the run at
                # whatever evaluation count the machine's speed happened to
                # produce.  Progress is counted in evaluations (AGENTS.md
                # "Local runs"); the deadlock backstop is the only clock left,
                # and it is an error path: a handler that has not returned
                # after ``deadlock_seconds`` never will.
                if not self.eventbus.wait_idle(timeout=self._deadlock_seconds):
                    self.logger.error(
                        "Deadlock backstop: an event handler has not returned for %.0fs (core.deadlock_seconds; "
                        "bus=%d). Ending the run; results so far: %d/%s."
                        % (self._deadlock_seconds, self.eventbus.inflight, len(self.results), self.config.max_eval)
                    )
                    break
                self.jobs_per_client = max(1, int(self.config.max_eval / 50.0))
            else:
                per_client = self.config.max_eval / 50.0
                avg = self.avg_time_per_task
                if avg > 0:  # NaN (no task timed yet) compares False
                    per_client = min(per_client, 1.0 / avg)
                self.jobs_per_client = max(1, int(per_client))

            # show heuristic performances after each round
            # logger.info('  '.join(('%s:%.3f' % (h, h.performance) for h in
            # heurs)))

            # Progress / liveness check.  A pass that produced nothing is not
            # a stall: the question is whether anything *can* still produce.
            # ``_alive`` answers that from state alone — no wall clock.  An
            # outstanding evaluation (running, queued for live workers, any
            # dask future) counts as progress for as long as it takes.
            current_results_count = len(self.results)
            progressed = (
                len(points) > 0
                or current_results_count != self._last_results_count
                or self.n_finished != self._last_n_finished  # a failed evaluation is progress too
                or self._pool_progressed()
            )
            now = time_module.time()
            if progressed:
                self._dead_loops = 0
                self._last_progress_at = now
                self._last_results_count = current_results_count
                self._last_n_finished = self.n_finished
            elif not self._alive():
                self._dead_loops += 1
                if self._dead_loops >= self._max_dead_loops:
                    # Not an incident: every queue is empty, nothing is in
                    # flight and the event bus is drained, so nothing in this
                    # process can change any of those three.  The run is over.
                    self.logger.info(
                        "No heuristic can produce a point (queues empty, bus idle, nothing in "
                        "flight); ending run at %d/%s evaluations." % (len(self.results), self.config.max_eval)
                    )
                    break
            else:
                # Nothing to wait for — no evaluation outstanding — yet the run
                # claims it can go on.  Only a genuine wedge stays here for
                # ``deadlock_seconds``.
                self._dead_loops = 0
                if self._last_progress_at is None:
                    self._last_progress_at = now
                elif now - self._last_progress_at > self._deadlock_seconds:
                    self.logger.error(
                        "Deadlock backstop: %.0fs with no evaluation outstanding and nothing new while the "
                        "run still reports itself alive (pending=%d, bus=%d, ready=%s): a heuristic that "
                        "claims a point but never gives one, a handler that never returns, or queued "
                        "evaluations without a live worker (core.deadlock_seconds). Results so far: %d/%s."
                        % (
                            now - self._last_progress_at,
                            len(self.pending),
                            self.eventbus.inflight,
                            [h.name for h in self.heuristics if h.can_produce],
                            len(self.results),
                            self.config.max_eval,
                        )
                    )
                    break

            # stopping criteria
            if len(self.results) >= self.config.max_eval:
                break
            # The whole budget is dispatched and nothing is in flight: no
            # further evaluation can happen, even if some of them failed
            # and left no result behind.
            if self.config.max_eval and self._dispatched >= int(self.config.max_eval) and not self.pending:
                break

            if self._stop_requested:
                self.logger.info("Stop requested via flag (e.g. convergence).")
                break

            if not sync or self.pending:
                # Limit loop speed while evaluations are in flight.  Under
                # sync threaded/processes evaluation nothing is in flight here
                # (the sleep would be pure latency); dask ignores
                # ``evaluation.sync`` and keeps ``pending`` filled, and without
                # the sleep that loop spun at full CPU.
                time_module.sleep(1e-3)

        # Final forced update to ensure UI shows 100% or final results
        self._update_progress_status(force=True)
        self._cleanup()

    def _pool_progressed(self) -> bool:
        """Is an evaluation outstanding, or did the pool move since the last pass?

        Evaluations can legitimately take hours and run remotely; the
        machinery then sits idle, waiting.  So *any* outstanding evaluation
        counts as progress, indefinitely:

        * dask: any outstanding future, queued or running — the client
          cannot reliably tell the two apart, so every one is "waiting",
          never a stall;
        * threads / processes: a task a worker has started, or a task
          queued on a pool whose workers are alive
          (:meth:`~panobbgo.local_pool.LocalPool.waiting`) — which covers
          queued tasks while worker processes spawn.

        The deadlock backstop therefore never cuts a running evaluation;
        per-evaluation run time is limited only by the opt-in
        ``evaluation.timeout``.  A pool event (a start or finish since the
        last pass) is progress too.
        """
        if self.config.evaluation_method == "dask":
            return bool(self.pending)
        pool = getattr(self, "_pool", None)
        if pool is None:
            return False
        events, last = pool.events, getattr(self, "_last_pool_events", None)
        self._last_pool_events = events
        if last is not None and events != last:
            return True
        return bool(pool.waiting())

    def _clamp_to_budget(self, points):
        """Cut a batch from :meth:`execute` to the evaluations the budget still allows.

        ``max_eval`` is a hard cap (AGENTS.md "Domain context"), and this is
        the one place every evaluation mode passes through, so no strategy
        has to size its batches against the budget itself.  The budget is
        charged at *dispatch* (``_dispatched``), not when a result arrives:
        a failing evaluation leaves no result but still cost a call, and an
        asynchronous run has evaluations in flight.  ``len(results) +
        len(pending)`` is a floor for the case where results arrive by
        another path.

        Surplus points go back to the queue of the heuristic that emitted
        them (front of the queue, original order) rather than being
        dropped.  The clamp can only bite on the batch that exhausts the
        budget — after it nothing is ever dispatched again — so this does
        not change which points a run evaluates; it keeps each heuristic's
        queue honest about what was never evaluated.  Sizing the request
        itself would mean threading ``room`` through every strategy's
        ``execute`` and selector; the bandits' pull counts on that last
        batch are the only state it would change.
        """
        if not self.config.max_eval:
            self._dispatched += len(points)
            return points
        used = max(self._dispatched, len(self.results) + len(self.pending))
        room = max(0, int(self.config.max_eval) - used)
        if len(points) > room:
            self.logger.debug("Budget clamp: dispatching %d of %d points." % (room, len(points)))
            self._return_to_queues(list(points[room:]))
            points = list(points[:room])
        self._dispatched = used + len(points)
        return points

    def _return_to_queues(self, points):
        """Put undispatched points back at the front of their heuristic's queue."""
        by_owner: Dict[Any, List[Any]] = collections.OrderedDict()
        for p in points:
            try:
                h = self.heuristic(p.who)
            except (KeyError, AttributeError):
                continue  # owner unknown (e.g. a test double): nothing to return to
            by_owner.setdefault(h, []).append(p)
        for h, pts in by_owner.items():
            q = h._output
            with q.mutex:
                q.queue.extendleft(reversed(pts))
                if 0 < q.maxsize < len(q.queue):
                    q.maxsize = len(q.queue)
                q.not_empty.notify_all()

    def request_stop(self):
        """Ask the main loop to end after the current pass.

        Thread-safe (a single flag write): a harness enforcing a wall-clock
        timeout calls it from another thread, then joins the runner thread.
        In-flight evaluations of an asynchronous run are abandoned.
        """
        self._stop_requested = True

    def _eval_timeout(self) -> Optional[float]:
        """Per-evaluation timeout in seconds of running time (``evaluation.timeout``); ``None`` = none."""
        t = getattr(self.config, "evaluation_timeout", None)
        return float(t) if t else None

    def _harvest(self, outcomes, new_results, failed):
        """Book :class:`~panobbgo.local_pool.Outcome`\\ s: results, failures, walltimes, ``pending``.

        The points of failed evaluations are appended to ``failed`` (see
        :meth:`_publish_failures`).
        """
        for o in outcomes:
            self.pending.pop(o.task_id, None)
            self.new_finished.append(o.task_id)
            self.n_finished += 1
            if o.started is not None:
                self.record_walltime(o.finished - o.started)
            if not o.ok:
                self.logger.error("Evaluation failed: %s" % o.error)
                if o.point is not None:
                    failed.append(o.point)
            elif isinstance(o.result, list):
                new_results.extend(o.result)
            else:
                new_results.append(o.result)

    def _publish_failures(self, points):
        """Publish ``failed_evaluations`` (``points``: the :class:`~panobbgo.lib.Point`\\ s that left no result).

        A failure (the objective raised, ``evaluation.timeout`` fired, the
        worker process died) used to be only logged.  A module waiting for
        the result of its own point -- a solver bridge's outstanding round
        trip -- then waited forever.  Published only when someone listens,
        so runs without failures see no extra event.
        """
        if points and "failed_evaluations" in self.eventbus.keys:
            self.eventbus.publish("failed_evaluations", points=list(points))

    def _run_threaded_evaluation(self, points):
        """Evaluate on the local pool — threads or (``"processes"``) spawned workers.

        An evaluation that raises, whose worker process dies, or that runs
        longer than ``evaluation.timeout`` is logged and leaves ``pending``
        without a result; it stays charged against the budget (see
        :meth:`_clamp_to_budget`).  See :mod:`panobbgo.local_pool`.
        """
        self.new_finished = []
        new_results = []
        failed = []
        timeout = self._eval_timeout()
        pool = self._pool

        if self.config.sync_evaluation:
            # Reproducible mode: results are booked in submission order.
            # A pool would return results in completion order, which makes
            # the evaluation trajectory (and every anytime metric computed
            # from it) depend on thread scheduling.  Threads evaluate on this
            # thread (for cheap objectives faster than the pool round trip);
            # processes run in parallel and are harvested in order.
            ids = [f"sync_task_{self.loops}_{i}" for i in range(len(points))]
            if self.config.evaluation_method == "processes":
                for tid, point in zip(ids, points):
                    pool.submit(tid, point)
                outcomes = {}
                # The deadlock backstop of ``_run`` never sees this loop, so it
                # applies its own — with the same rule: a running evaluation,
                # or one queued for live workers, is waited for as long as it
                # takes (only ``evaluation.timeout`` limits it).  Only tasks
                # that no worker can pick up for ``deadlock_seconds`` end the
                # run instead of blocking forever.
                deadlock = float(getattr(self, "_deadlock_seconds", self.config.deadlock_seconds))
                events, moved_at = pool.events, time_module.time()
                while len(pool):
                    for o in pool.poll(timeout):
                        outcomes[o.task_id] = o
                    now = time_module.time()
                    if pool.events != events or pool.waiting():  # moving, running, or queued for live workers
                        events, moved_at = pool.events, now
                    elif now - moved_at > deadlock:
                        self.logger.error(
                            "Deadlock backstop: %.0fs with %d evaluation(s) queued and no live worker to "
                            "run them (core.deadlock_seconds). Ending the run; results so far: %d/%s."
                            % (
                                now - moved_at,
                                len(pool),
                                len(self.results),
                                self.config.max_eval,
                            )
                        )
                        self._stop_requested = True
                        break
                    if len(pool):
                        pool.wait()
                self._harvest([outcomes[t] for t in ids if t in outcomes], new_results, failed)
            else:
                from .local_pool import Outcome

                if timeout is not None and not getattr(self, "_warned_sync_timeout", False):
                    self._warned_sync_timeout = True
                    self.logger.warning(
                        "evaluation.timeout is ignored for threaded evaluation with evaluation.sync: "
                        "evaluations run inline on the main thread and cannot be interrupted. "
                        "Use evaluation.method 'processes' to enforce it."
                    )
                for tid, point in zip(ids, points):
                    t0 = time_module.time()
                    try:
                        o = Outcome(tid, True, result=self._problem(point), started=t0, point=point)
                    except Exception as e:
                        o = Outcome(tid, False, error=repr(e), started=t0, point=point)
                    self._harvest([o], new_results, failed)
            self._publish_failures(failed)
            self.results += new_results
            return

        for i, point in enumerate(points):
            task_id = f"thread_task_{self.loops}_{i}"
            pool.submit(task_id, point)
            self.pending[task_id] = task_id
        self._harvest(pool.poll(timeout), new_results, failed)
        self._publish_failures(failed)
        self.results += new_results

    def _update_progress_status(self, force=False):
        """
        Update the progress status line with current optimization information.

        Args:
            force (bool): If True, force an update regardless of throttling.
        """
        if not hasattr(self, "panobbgo_logger"):
            return

        progress_reporter = self.panobbgo_logger.progress_reporter
        if not progress_reporter.status_enabled:
            return

        # Throttle updates to avoid excessive UI processing
        now = time_module.time()
        # 0.1 seconds (10 Hz) is a good balance for responsive UI without high CPU overhead
        if not force and now - self._last_status_update < 0.1:
            return

        self._last_status_update = now

        # Gather status information
        current_evals = len(self.results)
        max_evals = self.max_eval_or(1000)
        budget_pct = (current_evals / max_evals) * 100 if max_evals > 0 else 0

        # Calculate ETA
        if hasattr(self, "_start") and self._start is not None:
            elapsed_time = time_module.time() - self._start
            if current_evals > 0:
                avg_time_per_eval = elapsed_time / current_evals
                remaining_evals = max(0, max_evals - current_evals)
                eta_seconds = int(avg_time_per_eval * remaining_evals)
            else:
                eta_seconds = 0
        else:
            eta_seconds = 0

        # Get convergence estimate (simplified: distance to best known)
        convergence = 0.0
        if hasattr(self, "best") and self.best is not None:
            # Simple convergence measure: could be improved
            convergence = min(100.0, (current_evals / max_evals) * 100)

        # Get best function value
        best_value = None
        if hasattr(self, "best") and self.best is not None:
            best_value = self.best.fx

        # Get strategy-specific information (for bandit strategies)
        extra_fields = {}
        if hasattr(self, "_get_status_info"):
            extra_fields = getattr(self, "_get_status_info")()

        # Update status
        progress_reporter.update_status(
            budget_pct=budget_pct,
            eta_seconds=eta_seconds,
            convergence=convergence,
            best_value=float(best_value) if best_value is not None else 0.0,
            current_evals=current_evals,
            max_evals=max_evals,
            extra_fields=extra_fields,
        )

    def _alive(self) -> bool:
        """``True`` iff this run can still produce a point.

        The liveness predicate that replaced the wall-clock stall guard
        (``planning/DESIGN_pump_and_stall_2026-09-11.md`` §2).  Three terms,
        all read from existing state, none of them a clock:

        * ``pending`` — an evaluation is in flight, and its result will wake
          every reactive heuristic;
        * ``eventbus.inflight`` — a handler is queued or running, i.e. a
          heuristic is *about to* refill its queue.  This is the term the old
          guard lacked: it read "every queue is empty" and concluded "nothing
          can produce", when the refills were merely still on the bus;
        * ``h.can_produce`` — a queued point, or an on-demand arm ready to be
          asked.

        When all three are false the state is closed: no handler can run
        (the bus is drained), no result can arrive (nothing is in flight) and
        no queue holds a point, so nothing in this process can change any of
        them.  Stopping is then a fact, not a timeout.
        """
        if self.pending:
            return True
        if self.eventbus.inflight > 0:
            return True
        return any(h.can_produce for h in self.heuristics)

    def _can_still_produce(self) -> bool:
        """``True`` while an evaluation is in flight, i.e. a later result batch may wake a starved arm.

        Narrower than :meth:`_alive`: the block scheduler
        (:class:`~panobbgo.strategies.blocks.StrategyBlockBandit`) asks it
        whether an arm that cannot produce now may produce again.  It is
        *not* the right question inside one ``execute()``: pending results
        are harvested by the main loop only after ``execute`` returns, so
        they cannot refill a queue mid-pass (see
        :meth:`_collect_points_safely`).

        Until the pull bridge of ``DESIGN_pump_and_stall_2026-09-11.md`` §1
        this also counted a live pump thread — which never exits, so for any
        strategy carrying a solver-bridge arm the answer was unconditionally
        ``True``.  An on-demand arm now answers synchronously in
        :meth:`Heuristic.produce`, so there is nothing left to wait for.
        """
        return bool(self.pending)

    def _collect_points_safely(self, target, selector, until=None):
        """
        Safely collect points from heuristics with timeout protection.
        """

        points = []
        attempts = 0
        max_attempts = 20  # Approx 2 seconds

        while True:
            # Check termination
            if until:
                if until(points, target):
                    break
            elif len(points) >= target:
                break

            # Try to get points
            initial_count = len(points)
            new_points = selector()

            if new_points is None:
                # Selector signalled to stop (e.g. no active heuristics)
                break

            if new_points:
                points.extend(new_points)

            # Check progress
            if len(points) == initial_count:
                # Only a handler still on the event bus can refill a queue
                # within this pass.  In-flight evaluations cannot: the main
                # loop harvests them after ``execute`` returns.  (Waiting on
                # them cost 20 x 10 ms per empty draw -- 9.6 s of a 9.7 s
                # async StrategyRewarding run on Rosenbrock(2) -- and made
                # point collection depend on timing.)
                if self.eventbus.inflight <= 0:
                    break  # nothing will arrive until the next result batch
                attempts += 1
                if attempts >= max_attempts:
                    self.logger.warning(f"{self.name}: Timed out waiting for points.")
                    break
                time_module.sleep(0.01)
            else:
                attempts = 0

        return points

    def execute(self):
        """
        Overwrite this method when you extend this base strategy.
        """
        raise Exception("You need to extend the class StrategyBase and overwrite this execute method.")

    def _cleanup(self):
        """
        cleanup + shutdown

        Idempotent: :meth:`start` calls it from a ``finally`` so that an
        exception escaping the main loop cannot leak threads, and ``_run``
        calls it on the normal path.  The second call is a no-op.
        """
        if getattr(self, "_cleaned_up", False):
            return
        self._cleaned_up = True
        self.logger.info("Cleaning up strategy...")
        # One deadline for the whole cleanup: the pool, the bus drain and the
        # dispatcher join share ``shutdown_grace_seconds`` (the harness joins a
        # timed-out run for deadlock_seconds + shutdown_grace_seconds).
        deadline = time_module.time() + float(self.config.shutdown_grace_seconds)

        def remaining() -> float:
            return max(0.0, deadline - time_module.time())

        # Signal termination to all event bus subscribers
        self.eventbus.publish("finished", terminate=True)
        self._end = time_module.time()

        # The backend that was actually set up, not the configured name: a
        # caller may change ``config.evaluation_method`` after construction,
        # and if ``initialize()`` then fails before ``_ensure_cluster``, the
        # old backend (a LocalCluster, a LocalPool) is still the live one.
        if getattr(self, "_pool_method", self.config.evaluation_method) == "dask":
            from . import dask_evaluation

            # Cancel outstanding futures, close client + cluster
            dask_evaluation.close(self)
        else:  # "threaded" or "processes"
            # Drop queued evaluations and wait (bounded) for the ones already
            # running, so nothing of this run is still calling the objective
            # when the caller starts the next one; worker processes still
            # running at the deadline are killed.
            if getattr(self, "_pool", None) is not None:
                still = self._pool.close(deadline)
                if still:
                    self.logger.error(
                        "%d evaluation(s) still running at the shutdown deadline; %s."
                        % (still, "killed" if self._pool.processes else "abandoning them")
                    )

        # Finalize progress reporting
        if hasattr(self, "panobbgo_logger"):
            self.panobbgo_logger.progress_reporter.finalize()

        duration = self._end - self._start if hasattr(self, "_start") else 0.0
        loops = self.loops if hasattr(self, "loops") else 0
        self.logger.info("Strategy '%s' finished after %.3f [s] and %d loops." % (self._name, duration, loops))

        self.info()
        self.results.info()
        # Let ``on_finished`` reach every subscriber before the ``__stop__``
        # loop below unsubscribes them: a handler still queued behind a slow
        # one was silently dropped.  Bounded — cleanup must not hang on a
        # wedged handler — and skipped on the bus thread, which cannot wait
        # for itself.
        if threading.current_thread() is not self.eventbus._thread:
            self.eventbus.wait_idle(timeout=remaining())
        # *Every* module, not ``self.heuristics`` — that property filters on
        # ``active``, so a heuristic that had already exhausted itself (the F1
        # shape) never got its ``__stop__`` and kept its subprocess alive.
        # Plus heuristics from add() that initialize() never registered (it raised).
        modules = list(self._analyzers.values()) + list(self._heuristics.values())
        seen = {id(m) for m in modules}
        modules += [h for h in self._hs if id(h) not in seen]
        for m in modules:
            try:
                m.__stop__()
            except Exception as exc:
                self.logger.debug("%s.__stop__() failed: %r" % (getattr(m, "name", m), exc))
        # Deliver what is still queued (e.g. on_finished), then stop the dispatcher.
        self.eventbus.shutdown(timeout=min(2.0, remaining()))
        self.results.close()
        # (The Dask client and cluster were closed by dask_evaluation.close above.)

    def on_converged(self, reason, stats):
        """
        Called when the Convergence analyzer detects convergence.
        """
        stop_on_conv = getattr(self.config, "stop_on_convergence", True)
        if stop_on_conv:
            self.logger.info(f"Convergence detected: {reason}. Stopping strategy.")
            self._stop_requested = True

    def info(self):
        """ """
        avg = self.avg_time_per_task
        pend = len(self.pending)
        fini = self.n_finished
        peval = len(self.results)
        s = (
            "{0:4d} pnts | Tasks: {1:3d} pend, {2:3d} finished | "
            "{3:6.3f} [s] cpu, {4:6.3f} [s] wall, {5:6.3f} [s/task]".format(
                peval, pend, fini, self.time_cpu, self.time_wall, avg
            )
        )
        self.slogger.info(s)

    def record_walltime(self, seconds: float) -> None:
        """Book the walltime of one finished task (for :attr:`avg_time_per_task`)."""
        self._walltime_n += 1
        self._walltime_sum += float(seconds)

    @property
    def avg_time_per_task(self) -> float:
        """
        :return float: average walltime per finished task, ``NaN`` before the first one.

        O(1): a running count and sum, since the main loop reads it every pass.
        """
        if self._walltime_n > 0:
            return self._walltime_sum / self._walltime_n
        return float("nan")

    @property
    def time_wall(self):
        """
        wall time in seconds
        """
        return time_module.time() - self.time_start

    @property
    def time_cpu(self):
        """
        effective cpu time in seconds
        """
        return time_module.process_time()

    @property
    def time_start_str(self):
        return time_module.ctime(self.time_start)
