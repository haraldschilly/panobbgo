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
COBYQA Heuristic
================

Powell-family derivative-free trust-region local optimizer.

`COBYQA <https://www.cobyqa.com/>`_ (*Constrained Optimization BY Quadratic
Approximations*) is the modern successor — developed by Tom M. Ragonneau under
Zaikun Zhang's supervision (2023) — to Powell's BOBYQA / COBYLA / LINCOA /
NEWUOA family.  Like BOBYQA it maintains an interpolation set of ``2·n + 1``
points and fits a *quadratic model* of the objective inside an adaptive
trust-region; like LINCOA / COBYLA it natively supports bounds and linear /
nonlinear constraints.  Empirically COBYQA dominates Nelder-Mead on smooth
and near-smooth black-box objectives — Nelder-Mead's simplex updates are not
curvature-aware, so it converges slowly on ill-conditioned valleys (e.g.
Rosenbrock) where COBYQA's quadratic model directly captures the
second-order shape.

This heuristic fills a clear gap in Panobbgo's portfolio.  Before it,
:class:`~panobbgo.heuristics.nelder_mead.NelderMead` was the *only* generic
derivative-free local refinement step; the other local optimizer
(:class:`~panobbgo.heuristics.lbfgsb.LBFGSB`) requires a smooth gradient
approximation that breaks on noisy objectives.  COBYQA provides a
derivative-free *and* curvature-aware local refinement step that the
structural mutation catalog can swap into a portfolio whenever the
problem's local geometry has structure Nelder-Mead cannot exploit.

Asynchronous execution
----------------------

Like :class:`~panobbgo.heuristics.lbfgsb.LBFGSB`, COBYQA's reference
implementation in ``scipy.optimize.minimize`` is synchronous: the solver
calls a Python callable ``f(x)`` and blocks waiting for the return value.
We run it in a dedicated subprocess and pipe the request / response
between the solver and Panobbgo's event-driven main thread:

1. The subprocess invokes ``scipy.optimize.minimize(method='COBYQA')``
   with a callable that ``pipe.send(x)`` and ``pipe.recv()`` for ``f(x)``.
2. The main thread polls the pipe, projects ``x`` onto the feasible box,
   emits the projected point for evaluation, and when the result arrives
   in :meth:`on_new_results` sends the penalty value back over the pipe.
3. The subprocess uses that value to continue COBYQA's trust-region
   update and either calls ``f(x)`` again or terminates.

The :attr:`Heuristic.cap` is fixed to ``1`` because COBYQA can only have
one outstanding evaluation at a time — the subprocess blocks until the
previous return value arrives.

Constraint handling delegates to ``strategy.constraint_handler`` exactly
like :class:`~panobbgo.heuristics.lbfgsb.LBFGSB`: the value piped back
is ``constraint_handler.get_penalty_value(result)``, which yields the
true ``fx`` for feasible points and a penalized value otherwise.  This
means COBYQA "sees" a smooth penalty objective, which it can refine
locally even when raw constraints are non-smooth.

References
----------

* T. M. Ragonneau, Z. Zhang (2023).
  "An optimization method based on quadratic interpolation models for
  derivative-free unconstrained, bound-constrained, and linearly /
  nonlinearly constrained optimization."
  https://www.cobyqa.com/
* M. J. D. Powell (2009). "The BOBYQA algorithm for bound constrained
  optimization without derivatives." Cambridge NA Report DAMTP 2009/NA06.
* SciPy 1.14+ ``scipy.optimize.minimize(method='COBYQA')``.
"""

from __future__ import annotations

import multiprocessing
from typing import Any, Optional

import numpy as np

from panobbgo.core import Heuristic


def _make_pipe_objective(pipe: Any):
    """Build an objective callable that round-trips ``x`` / ``f(x)`` over a pipe.

    Extracted from :meth:`COBYQA.worker` to keep the subprocess entry
    point's cyclomatic complexity small.  The callable raises
    ``SystemExit`` on a closed pipe so the worker can shut down
    cleanly when the parent terminates it.
    """

    def f(x: np.ndarray) -> float:
        pipe.send(x)
        try:
            fx = pipe.recv()
        except (EOFError, OSError):
            raise SystemExit(0)
        if fx is None or not np.isfinite(fx):
            return float("inf")
        return float(fx)

    return f


def _build_cobyqa_options(
    initial_tr_radius: float,
    final_tr_radius: float,
    maxfev: Optional[int],
    scale: bool,
) -> dict:
    """Assemble the ``options`` dict passed to ``scipy.optimize.minimize``."""
    options: dict = {
        "initial_tr_radius": initial_tr_radius,
        "final_tr_radius": final_tr_radius,
        "scale": scale,
    }
    if maxfev is not None:
        options["maxfev"] = maxfev
    return options


def _safe_send(output: Any, payload: Any) -> None:
    """Send ``payload`` over the result pipe, swallowing parent-side teardowns."""
    try:
        output.send(payload)
    except Exception:
        # Parent already terminated us; nothing to report.
        pass


# Sensible defaults: ``1e-6`` is the COBYQA library default for final TR
# radius (essentially "machine-precision-ish" for double precision); the
# initial TR radius defaults to one tenth of the typical box width when the
# user does not override it.  Scaling is on by default — every Panobbgo
# problem has finite bounds, and scaling to ``[-1, 1]`` keeps the
# interpolation geometry well-conditioned for boxes whose axes span very
# different magnitudes.
_DEFAULT_FINAL_TR_RADIUS: float = 1e-6
_DEFAULT_INITIAL_TR_RADIUS: Optional[float] = None  # auto from box width
_DEFAULT_MAXFEV: Optional[int] = None  # let strategy budget terminate us
_DEFAULT_SCALE: bool = True


class COBYQA(Heuristic):
    """COBYQA: Powell-family derivative-free trust-region local optimizer.

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        initial_tr_radius: Initial trust-region radius.  Sensible values
            are in the order of one tenth of the greatest expected change
            to the variables.  When ``None`` (default) the heuristic uses
            ``0.1 · max(box_width)`` so the first step explores a useful
            slice of the feasible region.  Must be positive.
        final_tr_radius: Final trust-region radius — accuracy required in
            the converged variables.  COBYQA terminates once the radius
            falls below this threshold.  Default ``1e-6``.  Must be
            positive and strictly less than ``initial_tr_radius``.
        maxfev: Maximum number of function evaluations the underlying
            ``scipy.optimize.minimize`` may consume.  When ``None``
            (default) the strategy's evaluation budget is the only cap;
            the subprocess is terminated when the strategy stops.
            ``maxfev`` is useful when the user wants COBYQA to give up
            *earlier* than the global budget — e.g. when paired with
            other heuristics inside the same strategy.
        scale: When ``True`` (default), COBYQA rescales the variables to
            ``[-1, 1]`` based on the box bounds, keeping the
            interpolation geometry well-conditioned for boxes whose axes
            span very different magnitudes.  Panobbgo problems always
            have finite bounds, so this is safe.
        name: Override the heuristic's display name.

    Notes:
        - The heuristic spawns one dedicated subprocess per restart
          (Powell's interpolation set has to be rebuilt from scratch
          anyway, so a fresh process is the cleanest unit of restart).
        - Out-of-bounds proposals from the subprocess are projected onto
          the feasible box by :meth:`panobbgo.lib.Problem.project` before
          being emitted; the value the subprocess sees is therefore the
          objective at the projected (feasible) point.
        - Like :class:`~panobbgo.heuristics.lbfgsb.LBFGSB`, the
          subprocess uses the ``"spawn"`` start method to avoid the
          deadlocks that ``fork`` triggers in multi-threaded processes.
    """

    def __init__(
        self,
        strategy,
        initial_tr_radius: Optional[float] = _DEFAULT_INITIAL_TR_RADIUS,
        final_tr_radius: float = _DEFAULT_FINAL_TR_RADIUS,
        maxfev: Optional[int] = _DEFAULT_MAXFEV,
        scale: bool = _DEFAULT_SCALE,
        name: Optional[str] = None,
    ) -> None:
        if initial_tr_radius is not None:
            if not np.isfinite(initial_tr_radius) or initial_tr_radius <= 0.0:
                raise ValueError(
                    f"COBYQA: initial_tr_radius must be a positive finite float, got {initial_tr_radius!r}"
                )
        if not np.isfinite(final_tr_radius) or final_tr_radius <= 0.0:
            raise ValueError(f"COBYQA: final_tr_radius must be a positive finite float, got {final_tr_radius!r}")
        if initial_tr_radius is not None and final_tr_radius >= initial_tr_radius:
            raise ValueError(
                f"COBYQA: final_tr_radius ({final_tr_radius}) must be strictly less than "
                f"initial_tr_radius ({initial_tr_radius})"
            )
        if maxfev is not None:
            if not isinstance(maxfev, int):
                raise ValueError(f"COBYQA: maxfev must be an integer or None, got {maxfev!r}")
            if maxfev <= 0:
                raise ValueError(f"COBYQA: maxfev must be > 0, got {maxfev}")

        Heuristic.__init__(self, strategy, name=name or "COBYQA", cap=1)
        self.logger = self.config.get_logger("COBYQ")
        self.initial_tr_radius: Optional[float] = initial_tr_radius
        self.final_tr_radius: float = float(final_tr_radius)
        self.maxfev: Optional[int] = maxfev
        self.scale: bool = bool(scale)

        # Subprocess handles — populated by :meth:`__start__`.
        self.p1: Any = None  # parent end of the request pipe
        self.p2: Any = None  # child end of the request pipe
        self.out1: Any = None  # parent end of the result pipe
        self.out2: Any = None  # child end of the result pipe
        self.cobyqa: Any = None  # the Process itself

    def _resolve_initial_tr(self, box: np.ndarray) -> float:
        """Pick a sensible initial trust-region radius from the box width.

        When the user sets ``initial_tr_radius`` we use it verbatim.
        Otherwise we default to ``0.1 · max(box_width)`` so the first
        step explores ~10% of the largest axis — large enough to leave
        a degenerate starting point, small enough that COBYQA does not
        immediately stall against the bounds.
        """
        if self.initial_tr_radius is not None:
            return float(self.initial_tr_radius)
        widths = np.asarray(box)[:, 1] - np.asarray(box)[:, 0]
        max_width = float(np.max(widths)) if widths.size > 0 else 1.0
        radius = 0.1 * max_width if max_width > 0.0 else 0.1
        # Always keep the radius strictly above the convergence threshold
        # so COBYQA does not declare success on the first step.
        return max(radius, 10.0 * self.final_tr_radius)

    def __start__(self) -> None:
        try:
            ctx = multiprocessing.get_context("spawn")
            self.p1, self.p2 = ctx.Pipe()
            self.out1, self.out2 = ctx.Pipe(False)

            bounds = [tuple(row) for row in self.problem.box.box]
            x0 = np.array([(low + high) / 2.0 for low, high in bounds], dtype=float)
            initial_tr = self._resolve_initial_tr(self.problem.box.box)

            self.cobyqa = ctx.Process(
                target=self.worker,
                args=(
                    self.p2,
                    self.out2,
                    x0,
                    bounds,
                    initial_tr,
                    self.final_tr_radius,
                    self.maxfev,
                    self.scale,
                ),
                name=f"{self.name}-COBYQA",
            )
            self.cobyqa.daemon = True
            self.cobyqa.start()
        except Exception as e:
            raise RuntimeError(
                f"Failed to start COBYQA subprocess for heuristic '{self.name}'. "
                f"This usually indicates a multiprocessing issue. "
                f"Make sure multiprocessing is supported on this system. "
                f"Original error: {e}"
            ) from e

    @staticmethod
    def worker(
        pipe: Any,
        output: Any,
        x0: np.ndarray,
        bounds: list,
        initial_tr_radius: float,
        final_tr_radius: float,
        maxfev: Optional[int],
        scale: bool,
    ) -> None:
        """Subprocess entry point: drive scipy's COBYQA via pipe-based f(x)."""
        from scipy.optimize import minimize

        f = _make_pipe_objective(pipe)
        options = _build_cobyqa_options(initial_tr_radius, final_tr_radius, maxfev, scale)
        try:
            solution = minimize(f, x0, method="COBYQA", bounds=bounds, options=options)
        except SystemExit:
            return
        except Exception as exc:
            _safe_send(output, {"error": repr(exc)})
            return
        _safe_send(output, solution)

    def on_start(self) -> None:
        """Pipe x → emit → wait for fx → pipe.send pattern."""
        while not self._stopped:
            try:
                if self.out1.poll(0):
                    output = self.out1.recv()
                    self.logger.info(output)
                # Check for input with a short timeout so we still notice
                # the stop flag promptly.
                if self.p1.poll(0.1):
                    x = self.p1.recv()
                    self.emit(x)
            except (EOFError, OSError):
                break
            except Exception as e:
                self.logger.error(f"Error in COBYQA loop: {e}")
                break

    def __stop__(self) -> None:
        super(COBYQA, self).__stop__()
        if self.cobyqa is not None and self.cobyqa.is_alive():
            self.cobyqa.terminate()
            self.cobyqa.join(timeout=1.0)
            if self.cobyqa.is_alive():
                self.cobyqa.kill()

    def on_new_results(self, results) -> None:
        for result in results:
            if result.who == self.name:
                val = self.strategy.constraint_handler.get_penalty_value(result)
                self.p1.send(val)

    def on_restart(self, center, reason) -> None:
        """Tear down and restart the subprocess around ``center``.

        COBYQA's interpolation set is built from scratch on construction;
        the cleanest restart strategy is to terminate the subprocess and
        start a fresh one with ``center`` (when provided) as the new
        starting point.  When ``center`` is ``None`` we fall back to the
        box centre, matching :meth:`__start__`.
        """
        # Tear down the current subprocess.
        try:
            if self.cobyqa is not None and self.cobyqa.is_alive():
                self.cobyqa.terminate()
                self.cobyqa.join(timeout=1.0)
                if self.cobyqa.is_alive():
                    self.cobyqa.kill()
        except Exception as exc:
            self.logger.debug(f"COBYQA: subprocess teardown on restart failed: {exc}")

        self.clear_output()

        if self._stopped:
            return

        try:
            ctx = multiprocessing.get_context("spawn")
            self.p1, self.p2 = ctx.Pipe()
            self.out1, self.out2 = ctx.Pipe(False)

            bounds = [tuple(row) for row in self.problem.box.box]
            if center is None:
                x0 = np.array([(low + high) / 2.0 for low, high in bounds], dtype=float)
            else:
                # Clip the suggested center into the box — the strategy
                # may emit a center sitting on the boundary, and COBYQA
                # tolerates equality but not strict violation.
                center = np.asarray(center, dtype=float)
                lo = np.asarray([b[0] for b in bounds], dtype=float)
                hi = np.asarray([b[1] for b in bounds], dtype=float)
                x0 = np.clip(center, lo, hi)

            initial_tr = self._resolve_initial_tr(self.problem.box.box)

            self.cobyqa = ctx.Process(
                target=self.worker,
                args=(
                    self.p2,
                    self.out2,
                    x0,
                    bounds,
                    initial_tr,
                    self.final_tr_radius,
                    self.maxfev,
                    self.scale,
                ),
                name=f"{self.name}-COBYQA",
            )
            self.cobyqa.daemon = True
            self.cobyqa.start()
        except Exception as exc:
            self.logger.warning(f"COBYQA: subprocess restart failed: {exc}")
