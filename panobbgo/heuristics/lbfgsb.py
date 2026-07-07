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
L-BFGS-B Heuristic
==================

Multi-start, bound-constrained quasi-Newton local optimizer built on
:func:`scipy.optimize.fmin_l_bfgs_b` (finite-difference gradients).

Why a gradient-based arm matters
--------------------------------

Panobbgo's portfolio is rich in *derivative-free* generators — Differential
Evolution (L-SHADE / jSO / NL-SHADE-RSP), PSO, CMA-ES, Nelder-Mead, and the
Powell-family trust-region :class:`~panobbgo.heuristics.cobyqa.COBYQA`.  None
of them follow a smooth gradient directly.  On smooth, ill-conditioned
*valleys* — the Rosenbrock family being the canonical example — a quasi-Newton
method that builds a curvature estimate from finite-difference gradients
converges in a tiny fraction of the evaluations a population method needs.
The benchmark harness makes the gap concrete: on ``Rosenbrock_5D`` at a
200-evaluation budget every Panobbgo *strategy* scores ``0.0`` (never reaches
the optimum), while a single dedicated L-BFGS-B descent from the box centre
reaches the optimum basin (``f < 0.02``) — and ``scipy``'s ``dual_annealing``,
which wins this problem on the external-baseline column, owes that win to its
*own* L-BFGS-B local-search step.

Multi-start
-----------

A single descent only finds the local minimum of whatever basin its starting
point sits in, so on multi-modal landscapes one run is not enough.  The worker
therefore runs L-BFGS-B **repeatedly**: the first descent starts from the box
centre (deterministic and reproducible), and every subsequent descent starts
from a fresh uniform-random point in the box.  Random-restart L-BFGS-B is a
classic, surprisingly strong global strategy on smooth problems — each restart
is cheap because the local solver converges fast, and the best basin found
across restarts is kept by the strategy's ``Best`` analyzer.  The loop runs
until the strategy exhausts its evaluation budget (the parent terminates the
subprocess) or the optional ``max_starts`` cap is hit.

This fixes a long-standing defect: the previous implementation ran L-BFGS-B
**once** from the box centre and then went idle for the rest of the budget,
and it was wired into neither the default strategies nor the self-improvement
loop's structural catalog — i.e. it was effectively dead code.

Warm-started restarts (memetic mode)
------------------------------------

Uniform-random restarts are the right default when L-BFGS-B is the *only*
generator, but they are wasteful inside a **portfolio**: the rest of the
strategy (a DE variant, PSO, CMA-ES, …) is busy discovering good basins, yet a
uniform-restart L-BFGS-B ignores that intelligence and keeps sampling the whole
box blindly.  This is exactly the negative result the benchmark recorded on
2026-07-06 — bolting uniform-restart L-BFGS-B onto ``Rewarding_Diverse``
*regressed* the composite even though it halved the Rosenbrock best-distance,
because the wrong restart geometry (box centre → random) never exploited the
basin the portfolio had already found.

``warm_start=True`` switches the restart geometry to the **memetic** recipe
that ``scipy.optimize.dual_annealing`` owes its Rosenbrock win to: every restart
after the first descends from a small Gaussian perturbation of the strategy's
**best incumbent** result, so each L-BFGS-B run *polishes* the best basin the
whole portfolio has found so far rather than gambling on a fresh uniform draw.
Because L-BFGS-B builds a curvature estimate from finite-difference gradients,
a warm-started descent is intrinsically curvature-aware — the sharpest known
gap in the benchmark is precisely the curved-valley (Rosenbrock) class, where
every Panobbgo strategy scored ``0`` while stock dual annealing solved it.

The best incumbent lives *parent-side* (only the strategy's ``Best`` analyzer
knows it), so the worker cannot draw the warm-start point itself.  Instead the
worker **requests** an ``x0`` from the parent at the start of each restart over
the same pipe it already uses for ``f(x)`` round-trips (a sentinel string the
parent recognises); the parent replies with the perturbed incumbent.  When no
best exists yet (very early in the run) the parent falls back to a uniform-
random draw, so a warm-started worker degrades gracefully to classic multi-
start until the portfolio produces its first result.  ``warm_start=False``
(the default) keeps the historical uniform-restart worker byte-for-byte.

Asynchronous execution
----------------------

Like :class:`~panobbgo.heuristics.cobyqa.COBYQA`, ``fmin_l_bfgs_b`` is
*synchronous*: it calls a Python callable ``f(x)`` and blocks for the return
value.  We run it in a dedicated subprocess and bridge each ``f(x)`` request
to Panobbgo's event-driven main thread over a pipe:

1. The subprocess calls ``f(x)``, which ``pipe.send(x)`` and ``pipe.recv()``.
2. The main thread (:meth:`on_start`) polls the pipe, projects ``x`` onto the
   feasible box, emits the projected point, and when the evaluation returns in
   :meth:`on_new_results` sends the penalty value back over the pipe.
3. The subprocess uses that value to continue the quasi-Newton step.

The :attr:`Heuristic.cap` is fixed to ``1`` because L-BFGS-B can only have one
outstanding evaluation at a time — the blocking pipe naturally rate-limits the
descent against the rest of the portfolio.

Constraint handling delegates to ``strategy.constraint_handler`` exactly like
:class:`~panobbgo.heuristics.cobyqa.COBYQA`: the value piped back is
``constraint_handler.get_penalty_value(result)`` (true ``fx`` for feasible
points, a penalized value otherwise), so the solver sees a smooth penalty
objective it can descend even when raw constraints are non-smooth.

References
----------

* C. Zhu, R. H. Byrd, P. Lu & J. Nocedal (1997). "Algorithm 778: L-BFGS-B:
  Fortran subroutines for large-scale bound-constrained optimization."
  *ACM Transactions on Mathematical Software*, 23(4):550-560.
* SciPy ``scipy.optimize.fmin_l_bfgs_b``.
"""

from __future__ import annotations

import multiprocessing
from typing import Any, Optional

import numpy as np

from panobbgo.core import Heuristic


def _make_pipe_objective(pipe: Any):
    """Build an objective callable that round-trips ``x`` / ``f(x)`` over a pipe.

    Mirrors :func:`panobbgo.heuristics.cobyqa._make_pipe_objective`.  The
    callable raises ``SystemExit`` on a closed pipe so the worker can shut
    down cleanly when the parent terminates it.
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


def _safe_send(output: Any, payload: Any) -> None:
    """Send ``payload`` over the result pipe, swallowing parent-side teardowns."""
    try:
        output.send(payload)
    except Exception:
        pass


# A large per-start evaluation cap; the strategy budget (which terminates the
# subprocess) is the real limit.  ``None`` resolves to scipy's own default.
_DEFAULT_MAX_STARTS: Optional[int] = None  # unlimited until budget exhausted
_DEFAULT_MAXFUN: Optional[int] = None  # scipy default (15000) per start
_DEFAULT_EPSILON: Optional[float] = None  # scipy default finite-diff step

# Sentinel the warm-start worker sends over the request pipe to ask the parent
# for a restart ``x0`` (a perturbation of the strategy's best incumbent).  It is
# a bare string so it can never be mistaken for an ``np.ndarray`` search point.
_X0_REQUEST = "__lbfgsb_x0_request__"


class LBFGSB(Heuristic):
    """Multi-start L-BFGS-B bound-constrained quasi-Newton local optimizer.

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        max_starts: Maximum number of random-restart L-BFGS-B descents.
            ``None`` (default) means "restart until the strategy budget is
            exhausted" — the parent terminates the subprocess when the run
            ends.  Must be a positive integer when set.
        maxfun: Maximum function evaluations the underlying
            ``fmin_l_bfgs_b`` may spend on a *single* descent.  ``None``
            (default) uses scipy's own default (15000); the strategy budget
            still caps total evaluations.  A smaller value forces earlier
            restarts on functions where a single descent does not converge.
            Must be a positive integer when set.
        epsilon: Finite-difference step size for the gradient approximation.
            ``None`` (default) uses scipy's default (``1e-8``).  Must be a
            positive finite float when set.
        warm_start: When ``True``, every restart after the first box-centre
            descent starts from a small Gaussian perturbation of the
            strategy's **best incumbent** result instead of a fresh
            uniform-random point — the memetic "polish the best basin" recipe
            (see the module docstring).  Falls back to a uniform-random draw
            until the strategy produces its first result.  ``False`` (default)
            preserves the historical uniform-restart behaviour byte-for-byte.
        warm_start_sigma: Standard deviation of the warm-start perturbation as
            a fraction of each dimension's box range (default ``0.1``).  Only
            consulted when ``warm_start=True``.  A small positive value turns
            the restarts into iterated local search / basin hopping around the
            incumbent; ``0.0`` polishes the exact incumbent every restart
            (degenerate once the incumbent is itself an L-BFGS-B local
            minimum).  Must be a non-negative finite float.
        seed: Optional seed for the per-instance restart RNG (controls the
            random restart points after the first, box-centre descent, and —
            under ``warm_start`` — the parent-side perturbation RNG).
        name: Override the heuristic's display name.

    Notes:
        - The first descent always starts from the box centre (deterministic,
          reproducible).  Subsequent descents start from fresh uniform-random
          points in the box, or — under ``warm_start`` — from perturbations of
          the strategy's best incumbent.
        - The heuristic spawns one dedicated subprocess; ``on_restart``
          tears it down and relaunches it warm-started from the supplied
          restart centre (matching :class:`~panobbgo.heuristics.cobyqa.COBYQA`).
        - Out-of-bounds proposals from the subprocess are projected onto the
          feasible box by :meth:`panobbgo.lib.Problem.project` before being
          emitted; the value the subprocess sees is therefore the objective
          at the projected (feasible) point.
        - The subprocess uses the ``"spawn"`` start method to avoid the
          deadlocks that ``fork`` triggers in multi-threaded processes.
    """

    def __init__(
        self,
        strategy,
        max_starts: Optional[int] = _DEFAULT_MAX_STARTS,
        maxfun: Optional[int] = _DEFAULT_MAXFUN,
        epsilon: Optional[float] = _DEFAULT_EPSILON,
        warm_start: bool = False,
        warm_start_sigma: float = 0.1,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if max_starts is not None:
            if not isinstance(max_starts, int) or isinstance(max_starts, bool):
                raise ValueError(f"LBFGSB: max_starts must be an integer or None, got {max_starts!r}")
            if max_starts <= 0:
                raise ValueError(f"LBFGSB: max_starts must be > 0, got {max_starts}")
        if maxfun is not None:
            if not isinstance(maxfun, int) or isinstance(maxfun, bool):
                raise ValueError(f"LBFGSB: maxfun must be an integer or None, got {maxfun!r}")
            if maxfun <= 0:
                raise ValueError(f"LBFGSB: maxfun must be > 0, got {maxfun}")
        if epsilon is not None:
            if not np.isfinite(epsilon) or epsilon <= 0.0:
                raise ValueError(f"LBFGSB: epsilon must be a positive finite float or None, got {epsilon!r}")
        if not isinstance(warm_start, bool):
            raise ValueError(f"LBFGSB: warm_start must be a bool, got {warm_start!r}")
        if not np.isfinite(warm_start_sigma) or warm_start_sigma < 0.0:
            raise ValueError(f"LBFGSB: warm_start_sigma must be a non-negative finite float, got {warm_start_sigma!r}")

        Heuristic.__init__(self, strategy, name=name or "LBFGSB", cap=1)
        self.logger = self.config.get_logger("LBFGS")
        self.max_starts: Optional[int] = max_starts
        self.maxfun: Optional[int] = maxfun
        self.epsilon: Optional[float] = None if epsilon is None else float(epsilon)
        self.warm_start: bool = warm_start
        self.warm_start_sigma: float = float(warm_start_sigma)
        self.seed: Optional[int] = seed

        # Best incumbent tracked parent-side for warm-started restarts; updated
        # by :meth:`on_new_best`.  ``None`` until the strategy's first result.
        self._best_x: Optional[np.ndarray] = None
        # Parent-side RNG for the warm-start perturbation / fallback draw.  Kept
        # distinct from the worker's restart RNG (which lives in the subprocess)
        # so the two streams never share state across the process boundary.
        self._warm_rng = np.random.default_rng(seed)

        # Subprocess handles — populated by :meth:`__start__`.
        self.p1: Any = None  # parent end of the request pipe
        self.p2: Any = None  # child end of the request pipe
        self.out1: Any = None  # parent end of the status pipe
        self.out2: Any = None  # child end of the status pipe
        self.lbfgsb: Any = None  # the Process itself

    # ------------------------------------------------------------------
    # Subprocess plumbing
    # ------------------------------------------------------------------

    def _box_bounds(self) -> list:
        """Return the feasible box as a list of ``(low, high)`` tuples."""
        return [tuple(row) for row in self.problem.box.box]

    def _box_center(self, bounds: list) -> np.ndarray:
        """Midpoint of every box axis — the deterministic first start point."""
        return np.array([(low + high) / 2.0 for low, high in bounds], dtype=float)

    def _spawn(self, x0_first: np.ndarray, bounds: list) -> None:
        """Launch a fresh worker subprocess starting from ``x0_first``."""
        ctx = multiprocessing.get_context("spawn")
        self.p1, self.p2 = ctx.Pipe()
        self.out1, self.out2 = ctx.Pipe(False)

        lb = np.asarray([b[0] for b in bounds], dtype=float)
        ub = np.asarray([b[1] for b in bounds], dtype=float)

        self.lbfgsb = ctx.Process(
            target=self.worker,
            args=(
                self.p2,
                self.out2,
                np.asarray(x0_first, dtype=float),
                bounds,
                lb,
                ub,
                self.seed,
                self.max_starts,
                self.maxfun,
                self.epsilon,
                self.warm_start,
            ),
            name=f"{self.name}-LBFGS",
        )
        self.lbfgsb.daemon = True
        self.lbfgsb.start()

    def __start__(self) -> None:
        try:
            bounds = self._box_bounds()
            self._spawn(self._box_center(bounds), bounds)
        except Exception as e:
            raise RuntimeError(
                f"Failed to start LBFGSB subprocess for heuristic '{self.name}'. "
                f"This usually indicates a multiprocessing issue. "
                f"Make sure multiprocessing is supported on this system. "
                f"Original error: {e}"
            ) from e

    @staticmethod
    def worker(
        pipe: Any,
        output: Any,
        x0_first: np.ndarray,
        bounds: list,
        lb: np.ndarray,
        ub: np.ndarray,
        seed: Optional[int],
        max_starts: Optional[int],
        maxfun: Optional[int],
        epsilon: Optional[float],
        warm_start: bool = False,
    ) -> None:
        """Subprocess entry point: run multi-start L-BFGS-B over the pipe.

        When ``warm_start`` is set, restart points (after the first box-centre
        descent) are requested from the parent — a perturbation of the
        strategy's best incumbent — instead of drawn from the local uniform
        RNG.  The request is a single :data:`_X0_REQUEST` sentinel over the
        same pipe used for ``f(x)`` round-trips; a closed pipe raises
        ``SystemExit`` so the worker shuts down cleanly on parent teardown.
        """
        from scipy.optimize import fmin_l_bfgs_b

        f = _make_pipe_objective(pipe)
        rng = np.random.default_rng(seed)
        kwargs: dict = {"bounds": bounds, "approx_grad": True}
        if maxfun is not None:
            kwargs["maxfun"] = maxfun
        if epsilon is not None:
            kwargs["epsilon"] = epsilon

        def _request_warm_x0() -> np.ndarray:
            """Ask the parent for a warm-start ``x0`` (perturbed incumbent)."""
            pipe.send(_X0_REQUEST)
            try:
                x0 = pipe.recv()
            except (EOFError, OSError):
                raise SystemExit(0)
            return np.clip(np.asarray(x0, dtype=float), lb, ub)

        starts = 0
        try:
            while max_starts is None or starts < max_starts:
                if starts == 0:
                    x0 = np.asarray(x0_first, dtype=float)
                elif warm_start:
                    x0 = _request_warm_x0()
                else:
                    x0 = rng.uniform(lb, ub)
                try:
                    fmin_l_bfgs_b(f, x0, **kwargs)
                except SystemExit:
                    return
                except Exception:
                    # A single descent failed numerically (e.g. a non-finite
                    # gradient); fall through to the next restart.
                    pass
                starts += 1
        except SystemExit:
            return
        _safe_send(output, {"done": starts})

    def on_start(self) -> None:
        """Pipe x → emit → wait for fx → pipe.send loop (shared with COBYQA).

        Under ``warm_start`` the worker interleaves :data:`_X0_REQUEST`
        sentinels with its ``f(x)`` sends; those are answered inline with a
        perturbed incumbent (:meth:`_warm_start_x0`) rather than emitted for
        evaluation.
        """
        while not self._stopped:
            try:
                if self.out1.poll(0):
                    output = self.out1.recv()
                    self.logger.info(output)
                # Short poll timeout so we still notice the stop flag promptly.
                if self.p1.poll(0.1):
                    msg = self.p1.recv()
                    if isinstance(msg, str):
                        # The only string the worker sends is the warm-start
                        # x0 sentinel; answer it inline, never emit a string.
                        if msg == _X0_REQUEST:
                            self.p1.send(self._warm_start_x0())
                    else:
                        self.emit(msg)
            except (EOFError, OSError):
                break
            except Exception as e:
                self.logger.error(f"Error in LBFGSB loop: {e}")
                break

    def on_new_best(self, best) -> None:
        """Track the strategy's best incumbent for warm-started restarts.

        A no-op unless ``warm_start`` is enabled; the parent-side
        :meth:`_warm_start_x0` reads :attr:`_best_x` when the worker requests a
        restart point.  Mirrors :meth:`panobbgo.heuristics.nearby.Nearby.on_new_best`.
        """
        if best is None:
            return
        x = getattr(best, "x", None)
        if x is None:
            return
        self._best_x = np.asarray(x, dtype=float)

    def _warm_start_x0(self) -> np.ndarray:
        """Return a restart point: a perturbed best incumbent, or a uniform draw.

        Called on the parent side in :meth:`on_start` when the warm-start
        worker requests an ``x0``.  With a known incumbent the point is
        ``clip(best + N(0, sigma·range), box)``; before the first result (no
        incumbent yet) it is a uniform-random draw over the box, so a
        warm-started worker degrades to classic multi-start until the portfolio
        produces something to polish.
        """
        bounds = self._box_bounds()
        lo = np.asarray([b[0] for b in bounds], dtype=float)
        hi = np.asarray([b[1] for b in bounds], dtype=float)
        best = self._best_x
        if best is None:
            return self._warm_rng.uniform(lo, hi)
        ranges = hi - lo
        x0 = np.asarray(best, dtype=float) + self._warm_rng.normal(0.0, self.warm_start_sigma * ranges)
        return np.clip(x0, lo, hi)

    def __stop__(self) -> None:
        super(LBFGSB, self).__stop__()
        if self.lbfgsb is not None and self.lbfgsb.is_alive():
            self.lbfgsb.terminate()
            self.lbfgsb.join(timeout=1.0)
            if self.lbfgsb.is_alive():
                self.lbfgsb.kill()

    def on_new_results(self, results) -> None:
        for result in results:
            if result.who == self.name:
                # Penalty value: true fx if feasible, else penalized.
                val = self.strategy.constraint_handler.get_penalty_value(result)
                self.p1.send(val)

    def on_restart(self, center, reason: str = "") -> None:
        """Tear down and relaunch the subprocess warm-started at ``center``.

        Mirrors :meth:`panobbgo.heuristics.cobyqa.COBYQA.on_restart`: the
        first descent of the relaunched worker starts from ``center``
        (clipped into the box) when one is supplied, falling back to the box
        centre otherwise.  Subsequent descents resume random multi-start.
        """
        try:
            if self.lbfgsb is not None and self.lbfgsb.is_alive():
                self.lbfgsb.terminate()
                self.lbfgsb.join(timeout=1.0)
                if self.lbfgsb.is_alive():
                    self.lbfgsb.kill()
        except Exception as exc:
            self.logger.debug(f"LBFGSB: subprocess teardown on restart failed: {exc}")

        self.clear_output()

        if self._stopped:
            return

        try:
            bounds = self._box_bounds()
            if center is None:
                x0 = self._box_center(bounds)
            else:
                center = np.asarray(center, dtype=float)
                lo = np.asarray([b[0] for b in bounds], dtype=float)
                hi = np.asarray([b[1] for b in bounds], dtype=float)
                x0 = np.clip(center, lo, hi)
            self._spawn(x0, bounds)
        except Exception as exc:
            self.logger.warning(f"LBFGSB: subprocess restart failed: {exc}")
