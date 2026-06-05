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
Sobol' Quasi-Random Sequence Heuristic
======================================

Generates a low-discrepancy Sobol' sequence as a one-shot space-filling
initial design.  Compared to :class:`~panobbgo.heuristics.latin_hypercube.LatinHypercube`
or pure :class:`~panobbgo.heuristics.random.Random` sampling, Sobol' points
fill a hypercube more uniformly (lower star discrepancy), which gives
downstream model-based heuristics — Gaussian Processes, CMA-ES seed
populations, Nearby's local search around the current best — a higher-
quality grid of "first looks" at the search space for the same number of
evaluations.

Sobol' is widely used in modern Bayesian optimization libraries (BoTorch,
TuRBO, GPyOpt, scikit-optimize) as the default initial-design generator for
exactly this reason.

References
----------

* I. M. Sobol' (1967). "Distribution of points in a cube and approximate
  evaluation of integrals". Zh. Vych. Mat. Mat. Fiz. 7, pp. 784-802.
* A. B. Owen (2003). "Quasi-Monte Carlo sampling".
"""

from typing import Any, Optional

import numpy as np

from panobbgo.core import Heuristic


class Sobol(Heuristic):
    """
    Emit ``n`` Sobol'-sequence points scaled to the problem's bounding box.

    Sobol' is a low-discrepancy quasi-random sequence: the first ``2^k``
    points cover the unit hypercube with much better uniformity than the
    same number of i.i.d. uniform samples.  By default the sequence is
    *scrambled* (Owen-style) so that across reps we still get statistical
    independence — without that, every run would produce the same points
    and there would be no per-rep variance to average over.

    Note: the quick-mode ``Rewarding_Diverse`` strategy ships
    ``scramble=False`` after a 2026-05-31 ledger-evidence-driven
    codification (three independent self-improvement loop accepts with
    bootstrap-CI lower bound > 0; see
    ``planning/SELF_IMPROVEMENT_LOOP.md`` §13).  At ``n=16`` in the 2-D
    quick-mode battery the deterministic Sobol' grid is already
    optimally space-filling, and Owen scrambling adds variance the local
    heuristics then have to absorb — the per-rep "different points"
    benefit is dominated by the randomized harness's per-rep instance
    translation / rotation, which already provides the statistical
    independence scramble normally supplies.

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        n: Number of points to emit.  Sobol' uniformity is best at powers
            of two (``2^k``); the heuristic accepts any positive integer
            and warns when ``n`` is not a power of two.  Default ``16``.
        scramble: If ``True`` (default), apply Owen scrambling so different
            seeds produce statistically independent point sets while
            preserving the low-discrepancy property within each set.
        seed: Optional seed for reproducibility.  ``None`` (default) uses
            ``numpy.random``'s global state, which the harness seeds via
            ``np.random.seed``.
        name: Override the heuristic's display name.

    Notes:
        - Heuristic is "one-shot": after emitting ``n`` points it stops.
          That matches the role of an initial-design heuristic feeding a
          downstream local/model-based optimizer.
        - The output queue capacity is set to ``n`` so all points are
          enqueued at once; the strategy pulls them as evaluation budget
          allows.
        - Re-uses :func:`scipy.stats.qmc.Sobol`; no new dependency.
    """

    def __init__(
        self,
        strategy: "Any",
        n: int = 16,
        scramble: bool = True,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(n, int):
            raise ValueError(f"Sobol: n must be an integer, got {n!r}")
        if n <= 0:
            raise ValueError(f"Sobol: n must be positive, got {n}")
        if not isinstance(scramble, bool):
            raise ValueError(f"Sobol: scramble must be a bool, got {scramble!r}")

        cap = n
        Heuristic.__init__(self, strategy, cap=cap, name=name or "Sobol")
        self.n: int = n
        self.scramble: bool = scramble
        self.seed: Optional[int] = seed

        # Warn (but don't fail) when n is not a power of two — Sobol' has
        # the strongest uniformity guarantees at 2^k.  We still permit
        # arbitrary n because budgets are not always powers of two.
        if (n & (n - 1)) != 0:
            self.logger.debug(
                f"Sobol: n={n} is not a power of two; uniformity is best at the next 2^k.  Consider n=16, 32, 64, …"
            )

    def _generate(self) -> np.ndarray:
        """Sample ``self.n`` Sobol' points, scaled to ``problem.box``.

        Split out as a separate method so it's directly testable without
        starting the heuristic's emit thread.
        """
        from scipy.stats import qmc

        dim = self.problem.dim
        # qmc.Sobol(rng=...) accepts None, an int, or a Generator.  We use
        # ``rng`` (modern scipy keyword) instead of the deprecated ``seed``.
        sampler = qmc.Sobol(d=dim, scramble=self.scramble, rng=self.seed)

        # Use random_base2 when n is a power of two — preserves Sobol's
        # balance properties.  Otherwise fall back to .random(n).
        if (self.n & (self.n - 1)) == 0 and self.n > 0:
            m = int(np.log2(self.n))
            unit = sampler.random_base2(m=m)
        else:
            unit = sampler.random(n=self.n)

        # Map [0, 1)^d → box.
        lower = self.problem.box[:, 0]
        ranges = self.problem.ranges
        return unit * ranges + lower

    def on_start(self) -> None:
        """Generate and emit the full Sobol' batch once."""
        pts = self._generate()
        self.emit([p for p in pts])
