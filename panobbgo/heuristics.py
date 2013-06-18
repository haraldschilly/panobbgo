# -*- coding: utf8 -*-
# Copyright 2012 Harald Schilly <harald.schilly@univie.ac.at>
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

r'''
Heuristics
==========
The main idea behind all heuristics is, ...

Each heuristic needs to listen to at least one stream of
:class:`Events <panobbgo.core.Event>` from the :class:`~panobbgo.core.EventBus`.
Most likely, it is the `one-shot` event ``start``, which is
:meth:`published <panobbgo.core.EventBus.publish>` by the
:class:`~panobbgo.core.StrategyBase`.

.. inheritance-diagram:: panobbgo.heuristics

.. codeauthor:: Harald Schilly <harald.schilly@univie.ac.at>
'''
from config import get_config
from core import Heuristic, StopHeuristic
import numpy as np


class Random(Heuristic):
    '''
    always generates random points inside the box of the
    "best leaf" (see "Splitter") until the capped queue is full.
    '''
    def __init__(self, cap=None, name=None):
        name = "Random" if name is None else name
        self.leaf = None
        from threading import Event
        # used in on_start, to continue when we have a leaf.
        self.first_split = Event()
        Heuristic.__init__(self, name=name)

    def on_start(self):
        self.first_split.wait()
        splitter = self.strategy.analyzer("splitter")
        while True:
            r = self.leaf.ranges * np.random.rand(
                splitter.dim) + self.leaf.box[:, 0]
            self.emit(r)

    def on_new_split(self, box, children, dim):
        '''
        we are only interested in the (possibly new)
        leaf around the best point
        '''
        best = self.strategy.analyzer("best").best
        self.leaf = self.strategy.analyzer("splitter").get_leaf(best)
        self.clear_output()
        self.first_split.set()


class LatinHypercube(Heuristic):
    '''
    Partitions the search box into n x n x ... x n cubes.
    Selects randomly in such a way, that there is only one cube in each dimension.
    Then, it randomly selects one point from inside such a cube.

    e.g. with div=4 and dim=2::

      +---+---+---+---+
      | X |   |   |   |
      +---+---+---+---+
      |   |   |   | X |
      +---+---+---+---+
      |   | X |   |   |
      +---+---+---+---+
      |   |   | X |   |
      +---+---+---+---+
    '''
    def __init__(self, div):
        '''
        Args:
           - `div`: number of divisions, positive integer.
        '''
        cap = div
        Heuristic.__init__(self, cap=cap, name="Latin Hypercube")
        if not isinstance(div, int):
            raise Exception("LH: div needs to be an integer")
        self.div = div

    def _init_(self):
        # length of each box'es dimension
        self.lengths = self.problem.ranges / float(self.div)

    def on_start(self):
        import numpy as np
        div = self.div
        dim = self.problem.dim
        while True:
            pts = np.repeat(
                np.arange(div, dtype=np.float64), dim).reshape(div, dim)
            pts += np.random.rand(div, dim)  # add [0,1) jitter
            pts *= self.lengths             # scale with length, already divided by div
            pts += self.problem.box[:, 0]    # shift with min
            [np.random.shuffle(pts[:, i]) for i in range(dim)]
            self.emit([p for p in pts])  # needs to be a list of np.ndarrays


class NelderMead(Heuristic):
    r'''
    This heuristic is inspired by the
    `Nelder Mead Method <http://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method>`_

    Algorithm:

    * If there are enough result points available, it tries to find a
      subset of points, which are linear independent (hence, suiteable for NM)
      and have the best (so far) function values, and are
      close (in the same :class:`Box <panobbgo.analyzers.Splitter>`).

    * Then, it applies the NM heuristic in a randomized fashion, i.e. it generates
      several promising points into the same direction as
      the implied search direction. See :meth:`here <.nelder_mead>`.
    '''
    def __init__(self):
        Heuristic.__init__(self, name="Nelder Mead")
        self.logger = get_config().get_logger('H:NM')
        from threading import Event
        self.got_bb = Event()

    def gram_schmidt(self, dim, results, tol=1e-4):
        """
        Tries to calculate an orthogonal base of dimension `dim`
        with given list of :class:`Results <panobbgo_lib.lib.Result>` points.
        Retuns `None`, if not enough points or impossible.
        The actual basis is not important, only the points for it are.
        They are used in :meth:`~.nelder_mead`.
        """
        # start empty, and append in each iteration
        # sort points ascending by fx -> calc gs -> skip if <= tol
        import numpy as np
        base = []  # orthogonal system basis
        ret = []  # list of results, which will be returned
        if len(results) < dim:
            return None
        # sort the results by asc. f(x)
        results = sorted(results, key=lambda p: p.fx)
        # better? randomize results to diversify
        # from random import shuffle
        # shuffle(results)
        first = results.pop(0)
        base.append(first.x)
        ret.append(first)
        for p in results:
            w = p.x - np.sum(((v.dot(p.x) / v.dot(v)) *
                             v for v in base), axis=0)
            if np.any(np.abs(w) > tol):
                base.append(w)
                ret.append(p)
                if len(ret) >= dim:
                    return ret
            else:
                # self.logger.info("below tol: %s (base: %s)" % (np.abs(w),
                # base))
                pass
        return None

    def nelder_mead(self, base, scale=3, offset=0):
        """
        Retuns a new *randomized* search point for the given set of results (``base``),
        which are linearly independent enough to form a orthonormal base,
        using the Nelder-Mead Method.

        Optional Arguments:

        - ``scale``: Used when sampling the new points via the :func:`~numpy.random.rayleigh` method.
        - ``offset``: This is subtracted from the sample factor; i.e. negative
          values account for the "contraction".
        """
        # base = sorted(base, key = lambda r : r.fx)
        # worst = base.pop() # worst point, base is the remainder

        # get worst point and it's index (to remove it)
        worst_idx, worst = max(enumerate(base), key=lambda _: _[1].fx)
        del base[worst_idx]

        # TODO f(x) values are available and could be used for weighting
        # weights = [ np.log1p(worst.fx - r.fx) for r in base ]
        # weights = 1 + .1 * np.random.randn(len(base))
        centroid = np.average(
            [p.x for p in base], axis=0)  # , weights = weights)
        factor = np.random.rayleigh(scale=scale) - offset
        return worst.x + factor * (centroid - worst.x)

    def on_start(self):
        '''
        Algorithm Outline:

        #. Wait until a first or new best box has been found.

        #. Clear the ``got_bb`` flag, later on we use this to be notified
           about new best boxes via :meth:`.on_new_best_box`.

        #. ``bb`` is the currently used best box, it might be ``None`` if
           we have to look up the parents when searching for more result points.

        #. Inside the outer while, we try to find a suiteable base via :meth:`.gram_schmidt`.

        #. If we got such a base, we generate new search points via :meth:`.nelder_mead`
           until the queue is full (which blocks) or there is a new best box (breaks inner loop).

        #. The ``break`` exits the outer while and we start fresh with the new best box.
        '''
        dim = self.problem.dim
        while True:
            self.got_bb.wait()
            bb = self.best_box
            self.got_bb.clear()
            while bb is not None:
                base = self.gram_schmidt(dim, bb.results)
                if base:  # was able to find a base
                    if len(base) == 0:
                        break
                    # TODO split nelder_mead into a init phase, and a sampling
                    # routine
                    while not self.got_bb.is_set():
                        new_point = self.nelder_mead(base[:])
                        # self.logger.info("new point: %s" % new_point)
                        self.emit(new_point)
                    break
                else:  # not able to find base, try with parent of current best box
                    bb = bb.parent
                    if bb is None:
                        self.got_bb.clear(
                        )  # the "wait()" at the top is now active

    def on_new_best_box(self, best_box):
        '''
        When a new best box has been found by the :class:`~.analyzers.Splitter`, the
        ``got_bb`` :class:`~threading.Event` is set and the output queue is cleared.
        '''
        self.best_box = best_box
        self.got_bb.set()
        self.clear_output()  # clearing must come last


class Nearby(Heuristic):
    '''
    This provider generates new points based
    on a cheap (i.e. fast) algorithm. For each new best point,
    it and generates ``new`` many nearby point(s).
    The @radius is scaled along each dimension's range in the search box.

    Arguments::

    - ``axes``:
       * ``one``: only desturb one axis
       * ``all``: desturb all axes

    - ``new``: number of new points to generate (default: 1)
    '''
    def __init__(self, cap=3, radius=1. / 100, new=1, axes='one'):
        Heuristic.__init__(
            self, cap=cap, name="Nearby %.3f/%s" % (radius, axes))
        self.radius = radius
        self.new = new
        self.axes = axes

    def on_new_best(self, best):
        import numpy as np
        ret = []
        x = best.x
        if x is None:
            return
        # generate self.new many new points near best x
        for _ in range(self.new):
            new_x = x.copy()
            if self.axes == 'all':
                dx = (2.0 * np.random.rand(
                    self.problem.dim) - 1.0) * self.radius
                dx *= self.problem.ranges
                new_x += dx
            elif self.axes == 'one':
                idx = np.random.randint(self.problem.dim)
                dx = (2.0 * np.random.rand() - 1.0) * self.radius
                dx *= self.problem.ranges[idx]
                new_x[idx] += dx
            else:
                raise Exception("axis parameter not 'one' or 'all'")
            ret.append(new_x)
        self.emit(ret)


class Extremal(Heuristic):
    '''
    This heuristic is specifically seeking for points at the
    border of the box and around 0.
    The @where parameter takes a list or tuple, which has values
    from 0 to 1, which indicate the probability for sampling from the
    minimum, zero, center and the maximum. default = ( 1, .2, .2, 1 )
    '''
    def __init__(self, diameter=1. / 10, prob=None):
        Heuristic.__init__(self, name="Extremal")
        import numpy as np
        if prob is None:
            prob = (1, .2, .2, 1)
        for i in prob:
            if i < 0 or i > 1:
                raise Exception("entries in prob must be in [0, 1]")
        prob = np.array(prob) / float(sum(prob))
        self.probabilities = prob.cumsum()
        self.diameter = diameter  # inside the box or around zero

    def _init_(self):
        import numpy as np
        problem = self.problem
        low = problem.box[:, 0]
        high = problem.box[:, 1]
        zero = np.zeros(problem.dim)
        center = low + (high - low) / 2.0
        self.vals = np.row_stack((low, zero, center, high))

    def on_start(self):
        import numpy as np
        while True:
            ret = np.empty(self.problem.dim)
            for i in range(self.problem.dim):
                r = np.random.rand()
                for idx, val in enumerate(self.probabilities):
                    if val > r:
                        radius = self.problem.ranges[i] * self.diameter
                        # jitter = radius * (np.random.rand() - .5)
                        jitter = np.random.normal(0, radius)
                        if idx == 0:
                            ret[i] = self.vals[idx, i] + abs(jitter)
                        elif idx == len(self.probabilities) - 1:
                            ret[i] = self.vals[idx, i] - abs(jitter)
                        else:
                            ret[i] = self.vals[idx, i] + jitter
                        break  # since we found the idx, break!
            self.emit(ret)


class Zero(Heuristic):
    '''
    This heuristic only returns the 0 vector once.
    '''
    def __init__(self):
        Heuristic.__init__(self, name="Zero", cap=1)

    def on_start(self):
        from numpy import zeros
        return zeros(self.problem.dim)


class Center(Heuristic):
    '''
    This heuristic checks the point in the center of the box.
    '''
    def __init__(self):
        Heuristic.__init__(self, name="Center", cap=1)

    def on_start(self):
        box = self.problem.box
        return box[:, 0] + (box[:, 1] - box[:, 0]) / 2.0


class QuadraticOlsModel(Heuristic):
    '''
    This heuristic uses an quadratic OLS model to find an approximate new best point
    for each new best box (the latter is subject to change).

    The actual calculation is performed out of process.
    '''
    def __init__(self):
        Heuristic.__init__(self)
        self.logger = get_config().get_logger('H:QM')

        from multiprocessing import Process, Pipe
        # a pipe has two ends, parent and child.
        self.p1, self.p2 = Pipe()
        self.process = Process(
              target=self.solve_ols,
              args=(self.p2,),
              name='%s-Subprocess' % (self.name))
        self.process.daemon = True
        self.process.start()

    @staticmethod
    def solve_ols(pipe):
        def predict(xx):
            '''
            helper for the while loop: calculates the prediction
            based on the model result
            '''
            dim = len(xx)
            res = [1]
            res.extend(xx)
            for i in range(dim - 1):
                for j in range(i + 1, dim):
                    res.append(xx[i] * xx[j])
            for i in range(dim):
                res.append(xx[i] ** 2)
            return result.predct(np.array(res))


        while True:
            points, bounds = pipe.recv()
            dim = len(points[0].x)

            import numpy as np
            from pandas import DataFrame
            import statsmodels.api as sm
            data = {}
            for i in dim:
                data['x%s' % i] = [ x.x[i] for x in points]
            X = DataFrame({'Intercept' : np.ones(dim)})
            X = X.join(data)

            y = DataFrame({'y' : [ _.fx for _  in points ]})

            model = sm.OLS(y, X)
            result = model.fit()

            # optimize predict with x \in bounds
            from scipy.optimize import fmin_l_bfgs_b
            sol, fval, info = fmin_l_bfgs_b(predict, np.zeros(dim),
                              bounds=bounds, approx_grad=True)

            pipe.send((sol, fval, info)) # end while loop

    def on_new_best_box(self, best_box):
        #self.logger.info("")
        self.p1.send((best_box.points, self.problem.box))
        sol, fval, info = self.p1.recv()
        print 'solution:', sol
        print 'fval:', fval
        print 'info:', info
        self.emit(sol)


class WeightedAverage(Heuristic):
    '''
    This strategy calculates the weighted average of all points
    in the box around the best point of the :class:`~panobbgo.analyzers.Splitter`.
    '''
    def __init__(self, k=.1):
        Heuristic.__init__(self)
        self.k = k
        self.logger = get_config().get_logger('WAvg')

    def _init_(self):
        self.minstd = min(self.problem.ranges) / 1000.

    def on_new_best(self, best):
        assert best is not None and best.x is not None
        box = self.strategy.analyzer('splitter').get_leaf(best)
        if len(box.results) < 3:
            return

        # actual calculation
        import numpy as np
        xx = np.array([r.x for r in box.results])
        yy = np.array([r.fx for r in box.results])
        weights = np.log1p(yy - best.fx)
        weights = -weights + (1 + self.k) * weights.max()
        # weights = np.log1p(np.arange(len(yy) + 1, 1, -1))
        # self.logger.info("weights: %s" % zip(weights, yy))
        self.clear_output()
        ret = np.average(xx, axis=0, weights=weights)
        std = xx.std(axis=0)
        # std must be > 0
        std[std < self.minstd] = self.minstd
        # self.logger.info("std: %s" % std)
        for i in range(self.cap):
            ret = ret.copy()
            ret += (float(i) / self.cap) * np.random.normal(0, std)
            if np.linalg.norm(best.x - ret) > .01:
                # self.logger.info("out: %s" % ret)
                self.emit(ret)


class Subprocess(Heuristic):
    '''
    This is a test example for spawning a :mod:`subprocess <multiprocessing>`.
    The GIL is released and the actual work can be done in
    another process in parallel. Communication is done via a
    :func:`~multiprocessing.Pipe`.
    '''
    def __init__(self):
        Heuristic.__init__(self)
        self.logger = get_config().get_logger('SUBPR')
        from multiprocessing import Process, Pipe

        # a pipe has two ends, parent and child.
        self.p1, self.p2 = Pipe()

        self.process = Process(target=self.worker, args=(
            self.p2,), name='%s-Subprocess' % (self.name))
        self.process.daemon = True
        self.process.start()

    @staticmethod
    def worker(pipe):
        '''
        This static function is the target payload for the :class:`~multiprocessing.Process`.
        '''
        while True:
            x = pipe.recv()
            x += np.random.normal(0, 0.01, len(x))
            pipe.send(x)

    def on_new_best(self, best):
        self.p1.send(best.x)
        x = self.p1.recv()
        self.logger.debug("%s -> %s" % (best.x, x))
        return x


class LBFGSB(Heuristic):
    '''
    This uses :func:`scipy.optimize.fmin_l_bfgs_b` in a subprocess.
    '''
    def __init__(self):
        Heuristic.__init__(self, cap=1)
        self.logger = get_config().get_logger("LBFGS")

    def _init_(self):
        from multiprocessing import Process, Pipe
        self.p1, self.p2 = Pipe()
        self.out1, self.out2 = Pipe(False)
        self.lbfgsb = Process(target=self.worker, args=(self.p2, self.out2,
                              self.problem.dim), name='%s-LBFGS' % self.name)
        self.lbfgsb.daemon = True
        self.lbfgsb.start()

    @staticmethod
    def worker(pipe, output, dims):
        from scipy.optimize import fmin_l_bfgs_b
        import numpy as np

        def f(x):
            pipe.send(x)
            fx = pipe.recv()
            return fx

        solution = fmin_l_bfgs_b(f, np.zeros(dims), approx_grad=True)
        print solution
        output.send(solution)

    def on_start(self):
        while True:
            if self.out1.poll(0):
                output = self.out1.recv()
                self.logger.info(output)
            x = self.p1.recv()
            self.logger.info("x: %s" % x)
            self.emit(x)

    def on_new_results(self, results):
        for result in results:
            if result.who == self.name:
                self.p1.send(result.fx)


class Testing(Heuristic):
    '''
    just to try some ideas ...
    '''
    def __init__(self):
        Heuristic.__init__(self)
        self.i = 0
        self.j = 0

    def on_start(self):
        self.eventbus.publish('calling_testing')

    def on_calling_testing(self):
        # self.eventbus.publish('calling_testing')
        pass

    def on_new_best(self, best):
        # logger.info("TEST best: %s" % best)
        self.i += 1
        import numpy as np
        p = np.random.normal(size=self.problem.dim)
        self.emit(p)
        if self.i > 2:
            # logger.info('TEST best i = %s'%self.i)
            raise StopHeuristic()

    def on_new_results(self, results):
        # logger.info("TEST results: %s" % r)
        self.j += 1
        import numpy as np
        p = np.random.normal(size=self.problem.dim)
        self.emit(p)
        # if self.j > 5:
        #  raise StopHeuristic()
