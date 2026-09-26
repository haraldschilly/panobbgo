# -*- coding: utf8 -*-
# Copyright 2012-2026 Panobbgo Contributors
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
Virtual-clock parallel evaluation
=================================

``evaluation.method = "virtual"``: a deterministic discrete-event simulation
of ``q`` parallel workers (``evaluation.virtual_workers``), so that
asynchronous parallel behaviour can be benchmarked reproducibly on any
machine, without real waiting.

The objective is evaluated **synchronously, in-process**, at the moment a
simulated worker picks a candidate up.  Only the *durations*, and therefore
the order in which results reach the strategy and their timestamps, are
simulated.  The same seed gives the same evaluation sequence on every host,
whatever its speed.  The main loop runs synchronously (every event handler
drains before the next decision): optimizer time is taken to be negligible
next to an expensive evaluation.

Model
-----

* **Workers.**  ``q`` simulated workers.
* **Durations.**  Each call gets a duration from a :class:`DurationModel`
  (``evaluation.virtual_duration``): ``"constant"``, ``"lognormal"`` (log-space
  standard deviation ``evaluation.virtual_duration_sigma``, drawn from a
  keyed RNG stream, see :data:`RNG_STREAM_KEY`), a number (a
  constant duration) or x-dependent (a :class:`CallableDuration`, which needs
  an explicit nominal ``mean``).  The built-in models have **mean 1** unless
  ``evaluation.virtual_duration_mean`` says otherwise; the virtual-time
  metric measures time in units of the model's mean.
* **Common random numbers.**  The duration stream is keyed on
  ``evaluation.virtual_duration_seed`` when it is set (the harnesses set it
  per *cell* — base seed, problem, dimension, instance, rep — through
  :attr:`VirtualSpec.duration_seed`, never per strategy), else on the
  strategy's own seed.  One duration is drawn per dispatch, in dispatch
  order, by :class:`VirtualClock` and :func:`run_ask_tell` alike, so on one
  cell the i-th dispatched call takes the same time for every strategy and
  baseline: a paired comparison is paired in its durations too.  (An
  x-dependent model still sees each strategy's own points.)
* **Events.**  A call dispatched at virtual time ``t`` with duration ``d``
  completes at ``t + d``.  Completions form a queue ordered by ``(time,
  sequence)``, the sequence number being the dispatch order, so ties are
  broken deterministically.
* **Policy** (``evaluation.virtual_policy``):

  - ``"async"`` (default) — the asynchronous expensive-evaluation policy.
    There is a decision point at every completion instant: every call
    completing at that instant is delivered (one ``new_results`` batch), the
    handlers drain, and the strategy is asked for candidates for the free
    workers only.  The main loop sets the strategy's request cap
    (:attr:`StrategyBase.request_cap <panobbgo.core.StrategyBase.request_cap>`)
    to the free workers within the budget, and ``jobs_per_client = 1``, so a
    bandit pulls, and a block counts, exactly what is dispatched.  Nothing is
    queued beyond the free workers; :meth:`VirtualClock.admit` is a safety
    net that hands any surplus back to its heuristic's queue.  If the
    strategy fills fewer workers than are free, it is asked again at the
    same instant; if it has nothing, the clock moves to the next completion.
    With ``q = 1`` the run is strictly one call at a time, each candidate
    chosen after the previous result arrived.  This is an *idealized*
    pull-when-free loop: the real threaded / processes / dask loop does not
    implement it yet (see TODO.md).  Under a capped bandit a generational
    arm's queued generation drains at the bandit's share of the free
    workers, so its candidates can get stale while other arms run (a
    TODO.md follow-up).
  - ``"sync"`` — a regression mode that keeps the synchronous batch policy:
    a strategy batch (sized by the synchronous rule, ``max_eval / 50`` per
    worker) is queued first in, first out and the results are delivered when
    the queue has drained.  With ``q = 1``, a constant duration **and**
    ``dask.local.n_workers = 1`` (which sizes the synchronous batches) it
    reproduces the ``evaluation.sync`` run exactly
    (``tests/test_virtual_clock.py``).

* **Timeouts.**  With ``evaluation.timeout`` set, a call whose simulated
  duration exceeds it is not evaluated: it completes at ``dispatch +
  timeout`` as the usual ``NaN`` placeholder
  (:attr:`~panobbgo.lib.Result.timed_out`).  The timeout is in virtual time
  units here.  A timeout the problem *signals* (``lib.EvaluationTimedOut``,
  a family's failure region, known in advance through ``failure_at``) is
  treated the same way: not evaluated, completing after ``evaluation.timeout``
  (or its drawn duration when no timeout is set).  Crashes are evaluated
  (and fail) at dispatch.  Either failure is one spent, non-improving
  evaluation for an observer, counted once at completion.
* **Recording.**  Every result carries :attr:`~panobbgo.lib.Result.t_dispatch`
  and :attr:`~panobbgo.lib.Result.t_complete`, its virtual dispatch and
  completion times (in memory only: the sqlite storage backend does not
  store them).

Observers
---------

An evaluation counter such as :class:`~panobbgo.ioh_runner.IOHTracker` can
follow the clock (``strategy._virtual_observer``, set by
:meth:`VirtualSpec.apply`): the clock calls ``begin_call(seq)`` /
``end_call()`` around each objective call and ``complete_call(seq, t, ok)``
when the call completes, in completion order.  The tracker then records its
trace in completion order, with a failed or timed-out call as a spent,
non-improving evaluation — the input of
:func:`~panobbgo.ioh_runner.aocc_virtual_time`.
"""

from __future__ import annotations

import heapq
import math
import warnings
import zlib
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np

if TYPE_CHECKING:
    from .core import StrategyBase

#: ``SeedSequence`` spawn key of the duration stream: a stream of its own,
#: derived from the master seed, so drawing durations moves no module's RNG.
RNG_STREAM_KEY: Tuple[int, ...] = (zlib.crc32(b"panobbgo.virtual_clock"),)

#: ``evaluation.virtual_policy`` values (see the module docstring).
POLICIES: Tuple[str, ...] = ("async", "sync")

#: An x-dependent model warns when the observed mean duration of a run is
#: further than this (relative) from its nominal ``mean``: the virtual-time
#: metric's unit would then be off.
MEAN_TOLERANCE: float = 0.2


def _positive(name: str, value: float) -> float:
    v = float(value)
    if not (v > 0 and math.isfinite(v)):
        raise ValueError("%s must be finite and > 0, got %r" % (name, value))
    return v


class DurationModel:
    """Simulated duration of one evaluation call, in virtual time units.

    Subclasses implement :meth:`__call__`.  :attr:`mean` is the model's
    nominal mean duration; the anytime metric over virtual time measures
    time in units of it.
    """

    #: Nominal mean duration (the unit of the virtual-time metric).
    mean: float = 1.0

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> float:
        """Return the duration of evaluating ``x``; ``rng`` is the run's duration stream."""
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        """A JSON-able description (recorded with harness results)."""
        return {"model": type(self).__name__, "mean": self.mean}


class ConstantDuration(DurationModel):
    """Every call takes ``value`` time units (default 1, must be > 0)."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = self.mean = _positive("constant duration", value)

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> float:
        return self.value

    def describe(self) -> Dict[str, Any]:
        return {"model": "constant", "mean": self.mean}


class LogNormalDuration(DurationModel):
    """Log-normal durations with the given ``mean`` (default 1) and log-space ``sigma``.

    ``log d ~ N(mu, sigma^2)`` with ``mu = log(mean) - sigma^2 / 2``, so
    ``E[d] = mean``.  ``sigma = 0.5`` gives a coefficient of variation of
    about 0.53; ``sigma = 1`` about 1.3 (a heavy right tail).
    """

    def __init__(self, sigma: float = 0.5, mean: float = 1.0) -> None:
        if not (sigma >= 0 and math.isfinite(sigma)):
            raise ValueError("lognormal sigma must be finite and >= 0, got %r" % (sigma,))
        self.sigma = float(sigma)
        self.mean = _positive("lognormal mean", mean)
        self.mu = math.log(self.mean) - 0.5 * self.sigma**2

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> float:
        return float(rng.lognormal(self.mu, self.sigma))

    def describe(self) -> Dict[str, Any]:
        return {"model": "lognormal", "mean": self.mean, "sigma": self.sigma}


class CallableDuration(DurationModel):
    """x-dependent durations: ``fn(x)`` for the point's coordinates ``x``.

    ``mean`` (required) is the nominal mean duration the virtual-time metric
    uses as its unit; it is *not* estimated from the calls, so two strategies
    that visit different regions are scored on the same time axis.  A run
    whose observed mean is more than :data:`MEAN_TOLERANCE` off warns.
    ``fn`` must be deterministic in ``x`` (it does not see the RNG), and
    picklable (a module-level function, not a lambda) if the run goes to a
    worker process (harness ``jobs > 1``).
    """

    def __init__(self, fn: Callable[[np.ndarray], float], mean: float) -> None:
        if not callable(fn):
            raise TypeError("CallableDuration needs a callable, got %r" % (fn,))
        self.fn = fn
        self.mean = _positive("mean", mean)

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> float:
        return float(self.fn(x))

    def describe(self) -> Dict[str, Any]:
        return {"model": "callable", "mean": self.mean, "fn": getattr(self.fn, "__name__", repr(self.fn))}


DurationSpec = Union[str, float, int, DurationModel, Callable[[np.ndarray], float]]


def make_duration_model(
    spec: DurationSpec = "constant", sigma: float = 0.5, mean: Optional[float] = None
) -> DurationModel:
    """Build a :class:`DurationModel` from a config value.

    ``spec`` is a :class:`DurationModel` (returned as is), ``"constant"``
    (duration ``mean``, default 1), ``"lognormal"`` (mean ``mean``, default 1,
    log-space ``sigma``), a number or numeric string (a constant duration,
    e.g. from YAML), or a callable ``f(x)`` — which needs an explicit
    ``mean`` (a :class:`CallableDuration`).
    """
    if isinstance(spec, DurationModel):
        return spec
    if isinstance(spec, bool):
        raise TypeError("virtual_duration must be a model name, a number or a callable, got %r" % (spec,))
    if isinstance(spec, (int, float)):
        return ConstantDuration(float(spec))
    if isinstance(spec, str):
        name = spec.strip().lower()
        try:
            return ConstantDuration(float(name))
        except ValueError as e:
            if "must be finite" in str(e):
                raise
        if name == "constant":
            return ConstantDuration(1.0 if mean is None else mean)
        if name in ("lognormal", "log-normal"):
            return LogNormalDuration(sigma=float(sigma), mean=1.0 if mean is None else mean)
        raise ValueError("unknown virtual_duration %r: use 'constant', 'lognormal', a number or a callable" % (spec,))
    if callable(spec):
        if mean is None:
            raise ValueError(
                "an x-dependent duration model needs an explicit nominal mean "
                "(virtual_duration_mean / VirtualSpec(mean=...) / CallableDuration(fn, mean))"
            )
        return CallableDuration(spec, mean)
    raise TypeError("virtual_duration must be a model name, a number or a callable, got %r" % (spec,))


@dataclass(frozen=True)
class VirtualSpec:
    """Settings of a virtual-clock run, as a harness passes them.

    :meth:`apply` switches a constructed strategy to
    ``evaluation.method = "virtual"`` with these settings.  A spec travels to
    worker processes with the harness's ``jobs > 1``: an x-dependent
    ``duration`` must then be picklable (a module-level function or a
    :class:`CallableDuration` of one), not a lambda.
    """

    #: Number of simulated workers ``q``.
    workers: int = 4
    #: ``"constant"``, ``"lognormal"``, a number, a :class:`DurationModel` or a callable ``f(x)``.
    duration: Any = "constant"
    #: Log-space standard deviation of the ``"lognormal"`` model.
    sigma: float = 0.5
    #: Nominal mean duration; required for a callable ``duration``, default 1 otherwise.
    mean: Optional[float] = None
    #: ``"async"`` (default) or ``"sync"`` (see the module docstring).
    policy: str = "async"
    #: Seed of the duration stream (common random numbers).  The harnesses set
    #: it per cell, the same for every strategy on the cell
    #: (:func:`cell_duration_seed`); ``None``: the strategy's own seed.
    duration_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.workers) < 1:
            raise ValueError("virtual workers must be >= 1, got %r" % (self.workers,))
        if self.policy not in POLICIES:
            raise ValueError("virtual policy must be one of %s, got %r" % (POLICIES, self.policy))
        self.model()  # validate early

    def model(self) -> DurationModel:
        """The duration model these settings describe."""
        return make_duration_model(self.duration, self.sigma, self.mean)

    def apply(self, strategy: "StrategyBase", observer: Any = None) -> None:
        """Configure ``strategy`` (not yet started) for a virtual-clock run; ``observer``: see the module docstring."""
        cfg = strategy.config
        cfg.evaluation_method = "virtual"
        cfg.sync_evaluation = True  # the clock runs the synchronous main loop
        cfg.virtual_workers = int(self.workers)
        cfg.virtual_duration = self.duration
        cfg.virtual_duration_sigma = float(self.sigma)
        cfg.virtual_duration_mean = self.mean
        cfg.virtual_policy = self.policy
        cfg.virtual_duration_seed = self.duration_seed
        strategy._virtual_observer = observer

    def with_cell(self, seed: int) -> "VirtualSpec":
        """This spec with the duration stream of one cell (``seed``: :func:`cell_duration_seed`)."""
        return replace(self, duration_seed=int(seed))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able description: ``{"workers": q, "policy": ..., **model.describe(), "durations": "crn"}``.

        ``durations: "crn"`` records that durations are common random numbers
        per cell (results from before 2026-09-26 lack it: their streams were
        keyed per strategy).  The per-cell seed itself is not recorded.
        """
        return {"workers": int(self.workers), "policy": self.policy, **self.model().describe(), "durations": "crn"}


#: The strategy identity the duration stream of a cell is derived under
#: (in place of a strategy's ``rng_identity``): no strategy has this name.
DURATION_STREAM_IDENTITY = "<virtual-clock durations>"


@dataclass(order=True)
class _Call:
    """One simulated evaluation call; ordered by ``(t_complete, seq)``."""

    t_complete: float
    seq: int
    task_id: str = field(compare=False)
    point: Any = field(compare=False)
    t_dispatch: float = field(compare=False)
    timed_out: bool = field(default=False, compare=False)
    result: Any = field(default=None, compare=False)
    error: Optional[str] = field(default=None, compare=False)
    #: A timeout the problem signalled (``lib.EvaluationTimedOut``, a
    #: family's failure region): delivered as the ``NaN`` placeholder
    #: result, but a spent call for the observer.
    signalled: bool = field(default=False, compare=False)

    @property
    def ok(self) -> bool:
        """Did the call produce a value (the observer's view: failures and timeouts are spent)?"""
        return self.error is None and not self.timed_out and not self.signalled


def failure_mode(problem: Any, x: np.ndarray) -> Optional[str]:
    """``problem.failure_at(x)`` (``"crash"`` / ``"timeout"`` / ``None``) if the problem can tell in advance."""
    failure_at = getattr(problem, "failure_at", None)
    if not callable(failure_at):
        return None
    mode = failure_at(np.asarray(x, dtype=float))
    return mode if isinstance(mode, str) else None


def _placeholder(point: Any) -> Any:
    from .lib import Result

    return Result(point, float("nan"), cv_vec=None, timed_out=True)


def validate_config(config: Any) -> List[str]:
    """Error messages for the ``evaluation.virtual_*`` settings (empty if they are valid)."""
    errors = []
    try:
        if int(getattr(config, "virtual_workers", 4)) < 1:
            errors.append("evaluation.virtual_workers must be >= 1, got %r" % (config.virtual_workers,))
    except (TypeError, ValueError):
        errors.append("evaluation.virtual_workers must be an integer, got %r" % (config.virtual_workers,))
    policy = getattr(config, "virtual_policy", "async")
    if policy not in POLICIES:
        errors.append("evaluation.virtual_policy must be one of %s, got %r" % (POLICIES, policy))
    try:
        _model_of(config)
    except (TypeError, ValueError) as e:
        errors.append("evaluation.virtual_duration: %s" % e)
    return errors


def _model_of(config: Any) -> DurationModel:
    mean = getattr(config, "virtual_duration_mean", None)
    return make_duration_model(
        getattr(config, "virtual_duration", "constant"),
        float(getattr(config, "virtual_duration_sigma", 0.5)),
        None if mean is None else float(mean),
    )


class VirtualClock:
    """The simulated worker pool of one strategy (``strategy._virtual_clock``).

    Settings are read from the strategy's config by :meth:`configure`,
    which the strategy calls at set-up and again when the run starts, so a
    harness may change them in between.
    """

    def __init__(self, strategy: "StrategyBase") -> None:
        self.strategy = strategy
        #: Current virtual time.
        self.now: float = 0.0
        self.workers: int = 1
        self.policy: str = "async"
        self.model: DurationModel = ConstantDuration()
        self._rng: Optional[np.random.Generator] = None
        self._seq = 0
        self._running: List[_Call] = []  # heap by (t_complete, seq)
        self._done: List[_Call] = []  # completed, not yet delivered; (t_complete, seq) order
        self._n_drawn = 0
        self._sum_drawn = 0.0
        #: Candidates :meth:`admit` had to hand back (should stay 0: a
        #: strategy that ignores its request cap).
        self.n_trimmed = 0

    # ── set-up ──

    def configure(self) -> None:
        """Read the ``evaluation.virtual_*`` settings from the config."""
        cfg = self.strategy.config
        errors = validate_config(cfg)
        if errors:
            raise ValueError("; ".join(errors))
        self.workers = int(getattr(cfg, "virtual_workers", 4))
        self.policy = getattr(cfg, "virtual_policy", "async")
        self.model = _model_of(cfg)
        if self._rng is None:
            from .core import keyed_rng

            self._rng = keyed_rng(duration_seed(cfg, self.strategy.seed), RNG_STREAM_KEY)
        # ``len(strategy.evaluators)``: the strategies size their batches by it.
        self.strategy._n_processes = self.workers

    # ── state ──

    @property
    def busy(self) -> int:
        """Workers running a call (completed-but-undelivered calls have freed theirs)."""
        return len(self._running)

    @property
    def free(self) -> int:
        """Idle workers."""
        return self.workers - self.busy

    @property
    def jobs_per_client(self) -> Optional[int]:
        """``1`` under the async policy; ``None``: the synchronous rule."""
        return 1 if self.policy == "async" else None

    def _observer(self) -> Any:
        return getattr(self.strategy, "_virtual_observer", None)

    # ── simulation ──

    def request_cap(self) -> Optional[int]:
        """Async policy: how many candidates the strategy may produce now (free workers, within the budget)."""
        if self.policy != "async":
            return None
        cap = max(0, self.free)
        max_eval = self.strategy.config.max_eval
        if max_eval:
            s = self.strategy
            used = max(s._dispatched, len(s.results) + len(s.pending))
            cap = min(cap, max(0, int(max_eval) - used))
        return cap

    def admit(self, points: List[Any]) -> List[Any]:
        """Async policy safety net: keep the first candidates the free workers can take, return the rest.

        The strategy was asked for at most :meth:`request_cap` points, so
        this normally trims nothing; when it does, the strategy's bookkeeping
        already counted the surplus (logged at DEBUG).
        """
        if self.policy != "async" or len(points) <= self.free:
            return points
        free = max(0, self.free)
        msg = "virtual clock: the strategy produced %d candidates for %d free workers; returning %d to their queues" % (
            len(points),
            free,
            len(points) - free,
        )
        self.strategy.logger.debug(msg)
        if self.n_trimmed == 0:  # once per run: the strategy ignores its request cap
            warnings.warn(msg + " (the strategy ignores StrategyBase.request_cap)", RuntimeWarning, stacklevel=2)
        self.n_trimmed += len(points) - free
        self.strategy._return_to_queues(list(points[free:]))
        return list(points[:free])

    def _dispatch(self, point: Any) -> None:
        """A free worker takes ``point`` at :attr:`now`: draw its duration, evaluate it (unless it times out)."""
        strategy = self.strategy
        assert self._rng is not None
        x = np.asarray(point.x, dtype=float)
        d = float(self.model(x, self._rng))
        if not (d > 0 and math.isfinite(d)):
            raise ValueError("duration model returned %r for %s; durations must be finite and > 0" % (d, x))
        self._n_drawn += 1
        self._sum_drawn += d
        seq = self._seq
        self._seq += 1
        timeout = strategy._eval_timeout()
        timed_out = timeout is not None and d > timeout
        # A signalled timeout (a family's failure region, known in advance
        # through ``failure_at``) takes ``evaluation.timeout`` — the time a
        # real one is cut off at — or, with no timeout set, its drawn
        # duration.  It is not evaluated, like a simulated timeout.
        until_cut = timeout if timeout is not None else d
        problem = strategy._problem
        untranslate = getattr(problem, "_untranslate", None)
        x_eval = np.asarray(untranslate(x), dtype=float) if callable(untranslate) else x
        signalled = not timed_out and failure_mode(problem, x_eval) == "timeout"
        call = _Call(
            t_complete=self.now + (until_cut if (timed_out or signalled) else d),
            seq=seq,
            task_id="virtual_task_%d" % seq,
            point=point,
            t_dispatch=self.now,
            timed_out=timed_out,
            signalled=signalled,
        )
        if signalled:
            call.result = _placeholder(point)
        elif not timed_out:
            observer = self._observer()
            if observer is not None:
                observer.begin_call(seq)
            try:
                call.result = problem(point)
            except Exception as e:  # noqa: BLE001 — a failed evaluation, booked as such
                call.error = repr(e)
            finally:
                if observer is not None:
                    observer.end_call()
            if getattr(call.result, "timed_out", False):
                # Signalled by the objective without ``failure_at``: charged
                # the cut-off time too.
                call.signalled = True
                call.t_complete = self.now + until_cut
        strategy.pending[call.task_id] = call.task_id
        heapq.heappush(self._running, call)

    def _advance(self) -> None:
        """Jump to the next completion instant; every call completing then frees its worker."""
        t = self._running[0].t_complete
        self.now = t
        observer = self._observer()
        while self._running and self._running[0].t_complete <= t:
            call = heapq.heappop(self._running)
            if observer is not None:
                observer.complete_call(call.seq, call.t_complete, call.ok)
            self._done.append(call)

    def step(self, points: List[Any]) -> None:
        """One main-loop pass: dispatch ``points``, run the clock to the next decision point, deliver."""
        strategy = self.strategy
        max_eval = strategy.config.max_eval
        budget_left = not (max_eval and strategy._dispatched >= int(max_eval))
        if self.policy == "async":
            if len(points) > self.free:  # not admitted (:meth:`admit`): never queue past the workers
                raise RuntimeError("virtual clock: %d candidates for %d free workers" % (len(points), self.free))
            for p in points:
                self._dispatch(p)
            # Workers still free and the strategy still producing: ask again
            # at this instant.  Otherwise wait for the next completion.
            if not (self.free > 0 and points and budget_left) and self._running:
                self._advance()
        else:
            queue = list(points)
            while True:
                while queue and self.free > 0:
                    self._dispatch(queue.pop(0))
                if queue:  # every worker busy, candidates waiting: wait for one to finish
                    self._advance()
                    continue
                if self.free > 0 and points and budget_left:
                    break  # decision point: a free worker, nothing waiting — ask the strategy
                if self._running:  # nothing more to dispatch now: wait for the next completion
                    self._advance()
                break
        self._deliver()

    def _deliver(self) -> None:
        """Book every completed call, in ``(time, sequence)`` order, as one result batch."""
        from .local_pool import Outcome

        strategy = self.strategy
        strategy.new_finished = []
        new_results: List[Any] = []
        failed: List[Any] = []
        done, self._done = self._done, []
        for call in done:
            o = Outcome(
                call.task_id,
                call.error is None and not call.timed_out,  # a signalled timeout arrives as its placeholder
                result=call.result,
                error=call.error or ("timed out" if call.timed_out else ""),
                started=call.t_dispatch,
                finished=call.t_complete,
                point=call.point,
                timed_out=call.timed_out,
            )
            n0 = len(new_results)
            strategy._harvest([o], new_results, failed)
            for r in new_results[n0:]:
                r.t_dispatch = call.t_dispatch
                r.t_complete = call.t_complete
        strategy._publish_failures(failed)
        strategy.results += new_results

    def observed_mean(self) -> Optional[float]:
        """Mean of the durations drawn so far (``None`` before the first call)."""
        return self._sum_drawn / self._n_drawn if self._n_drawn else None

    def finish(self) -> None:
        """End-of-run check: warn if an x-dependent model's observed mean is far from its nominal mean."""
        check_nominal_mean(self.model, self._n_drawn, self._sum_drawn, self.strategy.logger)


def check_nominal_mean(model: DurationModel, n: int, total: float, logger: Any = None) -> None:
    """Warn if an x-dependent model's observed mean (``total / n`` over ``n >= 10`` calls) is off its nominal mean."""
    if not isinstance(model, CallableDuration) or n < 10:
        return
    observed, nominal = total / n, model.mean
    if abs(observed - nominal) > MEAN_TOLERANCE * nominal:
        msg = (
            "virtual clock: observed mean duration %.3g differs from the nominal mean %.3g by more than %d%%; "
            "the virtual-time metric uses the nominal mean as its unit" % (observed, nominal, MEAN_TOLERANCE * 100)
        )
        if logger is not None:
            logger.warning(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


def duration_seed(cfg: Any, fallback: int) -> int:
    """Seed of a run's duration stream: ``cfg.virtual_duration_seed`` (per cell) if set, else ``fallback``."""
    seed = getattr(cfg, "virtual_duration_seed", None)
    return int(fallback if seed is None else seed)


def _draw(model: DurationModel, x: np.ndarray, rng: np.random.Generator) -> float:
    d = float(model(x, rng))
    if not (d > 0 and math.isfinite(d)):
        raise ValueError("duration model returned %r for %s; durations must be finite and > 0" % (d, x))
    return d


def run_ask_tell(
    ask: Callable[[int], List[Tuple[Any, np.ndarray]]],
    tell: Callable[[Any, float], None],
    evaluate: Callable[[np.ndarray], float],
    *,
    workers: int,
    model: DurationModel,
    rng: np.random.Generator,
    budget: int,
    policy: str = "async",
    timeout: Optional[float] = None,
    observer: Any = None,
    reraise: Tuple[Type[BaseException], ...] = (),
    logger: Any = None,
    failure_at: Optional[Callable[[np.ndarray], Optional[str]]] = None,
) -> None:
    """Drive an ask/tell optimizer on the virtual clock (the external baselines' driver).

    The same model as :class:`VirtualClock`, for optimizers that are not a
    :class:`~panobbgo.core.StrategyBase`: ``workers`` simulated workers,
    durations from ``model`` (drawn from ``rng``), completions in ``(time,
    dispatch sequence)`` order.  Under ``policy="async"`` the optimizer is
    asked for up to the free workers (within ``budget``) at every
    completion instant, and again at the same instant while it fills fewer
    than are free; ``"sync"`` asks only when every worker is idle (the batch
    policy — for this driver "sync" means *batches of the free workers*,
    while :class:`VirtualClock`'s "sync" means the strategy's own
    synchronous batch sizing).  ``evaluate(x)`` runs at dispatch; ``tell(key, fx)`` is called
    at completion, in completion order, with ``NaN`` for a call that raised
    or timed out (``duration > timeout``: never evaluated, completes at
    ``dispatch + timeout``).  ``observer`` (an
    :class:`~panobbgo.ioh_runner.IOHTracker`) sees ``begin_call`` /
    ``end_call`` around each evaluation and ``complete_call`` at completion.
    Exceptions in ``reraise`` (a hard budget stop) propagate.
    ``failure_at(x)`` (a family's :meth:`~panobbgo.lib.families.Family.failure_at`)
    marks a signalled timeout in advance: not evaluated, told ``NaN`` after
    ``timeout`` (or its drawn duration with no timeout), a spent call for the
    observer — as :class:`VirtualClock` treats it.
    """
    if policy not in POLICIES:
        raise ValueError("virtual policy must be one of %s, got %r" % (POLICIES, policy))
    now = 0.0
    seq = 0
    n_drawn, sum_drawn = 0, 0.0
    running: List[Tuple[float, int, Any, float, bool]] = []  # heap (t_complete, seq, key, fx, ok)
    while True:
        free = workers - len(running)
        room = budget - seq
        want = min(free, room) if (policy == "async" or not running) else 0
        batch = ask(want) if want > 0 else []
        if len(batch) > want:
            raise RuntimeError("ask(%d) returned %d candidates; an adapter returns at most n" % (want, len(batch)))
        for key, x in batch:
            x = np.asarray(x, dtype=float)
            d = _draw(model, x, rng)
            n_drawn, sum_drawn = n_drawn + 1, sum_drawn + d
            if timeout is not None and d > timeout:
                heapq.heappush(running, (now + timeout, seq, key, float("nan"), False))
            elif failure_at is not None and failure_at(x) == "timeout":
                cut = timeout if timeout is not None else d
                heapq.heappush(running, (now + cut, seq, key, float("nan"), False))
            else:
                ok, fx = True, float("nan")
                if observer is not None:
                    observer.begin_call(seq)
                try:
                    fx = float(evaluate(x))
                except reraise:
                    raise
                except Exception:  # noqa: BLE001 — a failed evaluation: told as NaN
                    ok = False
                finally:
                    if observer is not None:
                        observer.end_call()
                heapq.heappush(running, (now + d, seq, key, fx, ok))
            seq += 1
        if policy == "async" and batch and len(batch) < want:
            continue  # workers still free: ask again at this instant
        if not running:
            if seq >= budget:
                break
            raise RuntimeError("ask() proposed nothing with no evaluation pending")
        t = running[0][0]
        now = t
        while running and running[0][0] <= t:
            t_c, s_c, key, fx, ok = heapq.heappop(running)
            if observer is not None:
                observer.complete_call(s_c, t_c, ok)
            tell(key, fx)
    check_nominal_mean(model, n_drawn, sum_drawn, logger)
