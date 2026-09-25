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
Tagged trials
=============

Population heuristics (:class:`~.lshade.LSHADE` and its variants,
:class:`~.pso.PSO`) tag every point they emit with a request id,
``who = "<name>:<req_id>"``, keep per-request bookkeeping in a ``pending``
dict, and match each returning result back to it.  These two helpers are that
round trip.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import numpy as np

from panobbgo.lib import Point


def emit_tagged(h, x: np.ndarray, label: str) -> Optional[Tuple[np.ndarray, str]]:
    """Project *x*, queue it under a fresh request id; ``(x_proj, req_id)`` or ``None``.

    Nothing is queued once *h* is stopped or when the projection fails
    (logged at debug level under *label*).  The request id is drawn from the
    heuristic's own RNG (not ``uuid4``/OS entropy) so ``Result.who`` tags are
    reproducible under a fixed seed.
    """
    if h._stopped:
        return None
    try:
        x_proj = h.problem.project(x)
    except Exception as exc:
        h.logger.debug(f"{label}: projection failed: {exc}")
        return None
    who = h.new_who(h._rng)
    h._put(Point(x_proj, who))
    return x_proj, who.split(":", 1)[1]


def own_results(name: str, results: Iterable[Any], pending: Dict[str, Any]) -> Iterator[Tuple[Any, Any]]:
    """Yield ``(result, meta)`` for each result tagged ``"<name>:<req_id>"`` with *req_id* pending.

    The entry is popped from *pending* as its result is reached, so the
    consumer may add new pending trials between results.  Foreign, stale and
    unknown results are skipped.
    """
    prefix = f"{name}:"
    for r in results:
        who: str = getattr(r, "who", "") or ""
        if not who.startswith(prefix):
            continue
        meta = pending.pop(who[len(prefix) :], None)
        if meta is None:
            continue  # stale or unknown trial id
        yield r, meta
