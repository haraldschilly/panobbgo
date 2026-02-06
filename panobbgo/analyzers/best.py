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
from __future__ import unicode_literals

from panobbgo.core import Analyzer


class Best(Analyzer):
    """
    Listens on all results, does accounting for "best" results,
    manages a pareto front of good points and emits the following events:

    - ``new_best``: when a new "best" point,
    - ``new_min``: a point with smallest objective function value,
    - ``new_cv``: one with a smaller
      :attr:`~panobbgo.lib.Result.cv`, or
    - ``new_pareto``: when a best point on the pareto front of the
      objective value and constraint violation

      has been found.

    .. Note::

      `Pareto`: Once the constraint
      violation is zero, only the objective function value is considered
      and future results with a positive constraint violation are ignored.

    The best point (pareto) is available via the :attr:`.best` attribute.
    There is also :attr:`.cv`, :attr:`.min` and :attr:`.pareto`.

    .. figure:: img/pareto-front.png
       :scale: 50 %

       This is an example of how several pareto fronts progress during the optimization.


    """

    def __init__(self, strategy):
        Analyzer.__init__(self, strategy)
        self.logger = self.config.get_logger("BEST")
        self._min = None
        self._cv = None
        self._pareto = None
        self._pareto_front = []  # this is a heapq, sorted by result.fx













    @property
    def best(self):
        """
        Currently best :class:`~panobbgo.lib.Result`.

        .. Note::

          At the moment, this is :attr:`.pareto` but might change.
        """
        return self.pareto

    @property
    def cv(self):
        """
        The point with currently minimal constraint violation, likely 0.0.
        If there are several points with the same minimal constraint violation,
        the value of the objective function is a secondary selection argument.
        """
        return self._cv

    @property
    def pareto(self):
        return self._pareto

    @property
    def min(self):
        """
        The point, with currently smallest objective function value.
        """
        return self._min

    @property
    def pareto_front(self):
        """
        This is the list of points building the current pareto front.

        .. Note::

          This is a shallow copy.
        """
        return self._pareto_front[:]

    def _update_pareto(self, result):
        """
        Update the pareto front with this new @result.

        Either ignore it, or add it to the front and remove
        all points from the front which are obsolete.
        """
        # Note: result.pp returns np.array([cv, fx])
        # add the new point
        pf_old = self.pareto_front

        # old code for convex front, below the one for a monotone step function
        # from utils import is_left
        # pf = self.pareto_front
        # pf needs to be sorted
        # pf.append(result)
        # pf = sorted(pf)
        # ... and re-calculate the front
        # new_front = pf[:1]
        # for p in pf[1:]:
        #  new_front.append(p)
        # next point needs to be left (smaller cv) and and above (higher fx)
        #  while len(new_front) > 1 and new_front[-1].cv >= new_front[-2].cv:
        #    del new_front[-1]
        # always a "right turn", concerning the ".pp" pareto points
        #  while len(new_front) > 2 and is_left(*map(lambda _:_.pp, new_front[-3:])):
        #    del new_front[-2]

        # stepwise monotone decreasing pareto front
        pf = self.pareto_front
        pf.append(result)
        pf = sorted(pf)
        pf_new = [pf[0]]
        for pp in pf[1:]:
            if pf_new[-1].cv > pp.cv:
                pf_new.append(pp)

        new_front = sorted(pf_new)

        self._pareto_front = new_front
        if pf_old != new_front:
            if len(self.pareto_front) > 2:
                self.logger.debug(
                    "pareto: %s" % [(x.cv, x.fx) for x in self.pareto_front]
                )
            self.eventbus.publish("new_pareto_front", front=new_front)

    def on_new_results(self, results):
        for r in results:
            if (
                (self._min is None)
                or (r.fx < self._min.fx)
                or (r.fx == self._min.fx and r.cv < self._min.cv)
            ):
                # self.logger.info(u"\u2318 %s by %s" %(r, r.who))
                self._min = r
                self.eventbus.publish("new_min", min=r)

            if (
                (self._cv is None)
                or (r.cv < self._cv.cv)
                or (r.cv == self._cv.cv and r.fx < self._cv.fx)
            ):
                self._cv = r
                self.eventbus.publish("new_cv", cv=r)

            # Use the strategy's constraint handler to determine if the new point is better
            is_better = False
            if self._pareto is None:
                is_better = True
            elif hasattr(self.strategy, 'constraint_handler') and self.strategy.constraint_handler:
                is_better = self.strategy.constraint_handler.is_better(self._pareto, r)
            else:
                # Fallback to default lexicographic behavior
                if r.cv < self._pareto.cv:
                    is_better = True
                elif r.cv == self._pareto.cv and r.fx < self._pareto.fx:
                    is_better = True

            if is_better:
                self._pareto = r
                self.eventbus.publish("new_pareto", pareto=r)
                self.eventbus.publish("new_best", best=r)

            self._update_pareto(r)

    def on_new_pareto(self, pareto):
        # self.logger.info("pareto: %s" % pareto)
        pass

    def on_new_pareto_front(self, front):
        """Report progress when a new pareto front is found."""
        self._report_progress_event("New Pareto front", "pareto_front")

    def on_new_cv(self, cv):
        """Report progress when a new constraint violation minimum is found."""
        self._report_progress_event(f"CV improved to {cv.cv:.4f}", "cv_improvement")

    def on_refresh_best(self, candidates):
        """
        Handle a refresh_best event, which suggests candidate points that might be
        better than current best due to changed criteria (e.g. updated constraint weights).
        """
        if not candidates:
            return

        for r in candidates:
            # Check if this candidate is better than current best
            is_better = False
            if self._pareto is None:
                is_better = True
            elif (
                hasattr(self.strategy, "constraint_handler")
                and self.strategy.constraint_handler
            ):
                is_better = self.strategy.constraint_handler.is_better(self._pareto, r)
            else:
                # Fallback to default lexicographic behavior
                if r.cv < self._pareto.cv:
                    is_better = True
                elif r.cv == self._pareto.cv and r.fx < self._pareto.fx:
                    is_better = True

            if is_better:
                self.logger.info(
                    f"Updated best point due to criteria change: {r.fx} (cv={r.cv})"
                )
                self._pareto = r
                self.eventbus.publish("new_pareto", pareto=r)
                self.eventbus.publish("new_best", best=r)

    def on_new_min(self, min):
        """Report progress when a new global minimum is found."""
        self._report_progress_event(f"New best: {min.fx:.6f}", "new_best")

    def _report_progress_event(self, message: str, event_type: str):
        """Report a significant optimization event via progress system."""
        # Log the event
        self.logger.info(message)

        # Get progress reporter from strategy and trigger status update
        if hasattr(self.strategy, 'panobbgo_logger') and self.strategy.panobbgo_logger:
            progress_reporter = self.strategy.panobbgo_logger.progress_reporter
            if progress_reporter and progress_reporter.enabled:
                # Trigger a status update on the strategy (this will refresh the progress display)
                if hasattr(self.strategy, '_update_progress_status'):
                    try:
                        self.strategy._update_progress_status()
                    except Exception:
                        # If status update fails, just continue
                        pass


