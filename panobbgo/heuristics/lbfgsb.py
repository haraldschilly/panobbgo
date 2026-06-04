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
        seed: Optional seed for the per-instance restart RNG (controls the
            random restart points after the first, box-centre descent).
        name: Override the heuristic's display name.

    Notes:
        - The first descent always starts from the box centre (deterministic,
          reproducible).  Subsequent descents start from fresh uniform-random
          points in the box.
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

        Heuristic.__init__(self, strategy, name=name or "LBFGSB", cap=1)
        self.logger = self.config.get_logger("LBFGS")
        self.max_starts: Optional[int] = max_starts
        self.maxfun: Optional[int] = maxfun
        self.epsilon: Optional[float] = None if epsilon is None else float(epsilon)
        self.seed: Optional[int] = seed

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
    ) -> None:
        """Subprocess entry point: run multi-start L-BFGS-B over the pipe."""
        from scipy.optimize import fmin_l_bfgs_b

        f = _make_pipe_objective(pipe)
        rng = np.random.default_rng(seed)
        kwargs: dict = {"bounds": bounds, "approx_grad": True}
        if maxfun is not None:
            kwargs["maxfun"] = maxfun
        if epsilon is not None:
            kwargs["epsilon"] = epsilon

        starts = 0
        try:
            while max_starts is None or starts < max_starts:
                x0 = np.asarray(x0_first, dtype=float) if starts == 0 else rng.uniform(lb, ub)
                try:
                    fmin_l_bfgs_b(f, x0, **kwargs)
                except SystemExit:
                    return
                except Exception:
                    # A single descent failed numerically (e.g. a non-finite
                    # gradient); fall through to the next random restart.
                    pass
                starts += 1
        except SystemExit:
            return
        _safe_send(output, {"done": starts})

    def on_start(self) -> None:
        """Pipe x → emit → wait for fx → pipe.send loop (shared with COBYQA)."""
        while not self._stopped:
            try:
                if self.out1.poll(0):
                    output = self.out1.recv()
                    self.logger.info(output)
                # Short poll timeout so we still notice the stop flag promptly.
                if self.p1.poll(0.1):
                    x = self.p1.recv()
                    self.emit(x)
            except (EOFError, OSError):
                break
            except Exception as e:
                self.logger.error(f"Error in LBFGSB loop: {e}")
                break

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
