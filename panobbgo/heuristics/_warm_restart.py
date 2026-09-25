# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
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
Warm restarts
=============

A warm-started population heuristic (:class:`~.pso.PSO`,
:class:`~.lshade.LSHADE` and its variants) that receives a ``restart`` event
has two places to go: the :class:`~panobbgo.analyzers.restart.Restart`
analyzer's ``center``, or the shared archive's best points.  The archive's
best points usually lie in the basin the population has just stagnated in, so
seeding from them puts the restart straight back where it was and makes the
analyzer's ``"diverse"`` / ``"sphere"`` restart strategies ineffective.

The rule (:func:`restart_from_archive`): seed from the archive only when the
archive's best point lies **outside the stagnated basin**; otherwise restart
around ``center`` exactly as the cold path does.

The *stagnated basin* is deliberately simple: the axis-aligned bounding box of
the live population's positions at the moment of the restart, inflated on
every side by :data:`BASIN_MARGIN` times the problem box's range in that
coordinate (so a one-point or flat population still has a non-degenerate
box).  With no live positions there is no basin to avoid, and the archive is
used as before.

Warm starts outside restarts (``on_start``, ``warm_start_now``) do not go
through this test.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

#: Inflation of the stagnated-basin box, per side, as a fraction of the
#: problem box's range in each coordinate.
BASIN_MARGIN: float = 0.01


def basin_box(problem: Any, live_x: Any, margin: float = BASIN_MARGIN) -> Optional[np.ndarray]:
    """The stagnated-basin box of the live positions, shape ``(dim, 2)``.

    The axis-aligned bounding box of ``live_x`` (one position per row),
    inflated per side by ``margin`` times the problem box's range.  ``None``
    when there are no finite live positions.
    """
    pts = np.asarray(live_x, dtype=float)
    if pts.size == 0:
        return None
    pts = pts.reshape(-1, problem.dim)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return None
    ranges = problem.box[:, 1] - problem.box[:, 0]
    pad = margin * ranges
    return np.column_stack([pts.min(axis=0) - pad, pts.max(axis=0) + pad])


def restart_from_archive(h: Any, live_x: Any) -> bool:
    """Whether a warm-started ``h`` should re-seed from the archive on restart.

    ``True`` iff the archive has a best point (the ``"archive"`` selector,
    restricted to ``h.warm_start_box`` like the seeding itself) and that point
    lies outside :func:`basin_box` of ``live_x``.  ``False`` means "restart
    around the analyzer's ``center``" — including the empty-archive case,
    where the archive seeding would have fallen back to that path anyway.

    A pure query: no RNG draws, no state changes.
    """
    best = h.archive_seed(1, mode="archive", box=getattr(h, "warm_start_box", None))
    if not best:
        return False
    box = basin_box(h.problem, live_x)
    if box is None:
        return True
    x = np.asarray(best[0].x, dtype=float)
    inside = bool(np.all(x >= box[:, 0]) and np.all(x <= box[:, 1]))
    return not inside
