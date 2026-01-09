# -*- coding: utf8 -*-
# Copyright 2012 Harald Schilly <harald.schilly@gmail.com>
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
from panobbgo.analyzers.best import Best


from panobbgo.core import Heuristic


class Nearby(Heuristic):
    """
    This provider generates new points based
    on a cheap (i.e. fast) algorithm. For each new best point,
    it and generates ``new`` many nearby point(s).
    The @radius is scaled along each dimension's range in the search box.

    Arguments::

    - ``axes``:
       * ``one``: only desturb one axis
       * ``all``: desturb all axes

    - ``new``: number of new points to generate (default: 1)
    """

    def __init__(self, strategy, cap=3, radius=1.0 / 100, new=1, axes="one"):
        Heuristic.__init__(
            self, strategy, cap=cap, name="Nearby %.3f/%s" % (radius, axes)
        )
        self.radius = radius
        self.new = new
        self.axes = axes
        self._depends_on = [Best]

    def on_new_best(self, best):
        import numpy as np

        ret = []
        x = best.x
        if x is None:
            return
        # generate self.new many new points near best x
        for _ in range(self.new):
            new_x = x.copy()
            if self.axes == "all":
                dx = (2.0 * np.random.rand(self.problem.dim) - 1.0) * self.radius
                dx *= self.problem.ranges
                new_x += dx
            elif self.axes == "one":
                idx = np.random.randint(self.problem.dim)
                dx = (2.0 * np.random.rand() - 1.0) * self.radius
                dx *= self.problem.ranges[idx]
                new_x[idx] += dx
            else:
                raise ValueError(
                    f"Nearby heuristic received invalid 'axes' parameter: '{self.axes}'. "
                    f"Valid options are 'one' (perturb one axis) or 'all' (perturb all axes)."
                )
            ret.append(new_x)
        self.emit(ret)
