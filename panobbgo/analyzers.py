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
Analyzers
=========

Analyzers, just like :mod:`.heuristics`, listen to events
and change their internal state based on local and
global data. They can emit :class:`events <panobbgo.core.Event>`
on their own and they are accessible via the
:meth:`~panobbgo.core.StrategyBase.analyzer` method of
the strategy.

.. inheritance-diagram:: panobbgo.analyzers

.. codeauthor:: Harald Schilly <harald.schilly@univie.ac.at>
'''
from config import get_config
from panobbgo_lib import Result
from core import Analyzer
from utils import memoize
import numpy as np


class Best(Analyzer):
    '''
    Listens on all results, does accounting for "best" results,
    manages a pareto front of good points and emits the following events:

    - ``new_best``: when a new "best" point,
    - ``new_min``: a point with smallest objective function value,
    - ``new_cv``: one with a smaller
      :attr:`~panobbgo_lib.lib.Result.constraint_violation`, or
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

    It also creates UI plots.
    '''
    def __init__(self):
        Analyzer.__init__(self)
        self.logger = get_config().get_logger("BEST")
        self._min = None
        self._cv = None
        self._pareto = None
        self._pareto_front = []  # this is a heapq, sorted by result.fx

    def _init_plot(self):
        return [self._init_plot_pareto(),
                self._init_plot_fx(),
                self._init_plot_eval()]

    def _init_plot_fx(self):
        from ui import NavigationToolbar
        import gtk
        self.fx_canvas, fig = self.ui.mk_canvas()

        # f(x) plot
        self.ax_fx = ax_fx = fig.add_subplot(1, 1, 1)
        # from matplotlib.ticker import MultipleLocator
        # ax_fx.xaxis.set_major_locator(MultipleLocator(.1))
        # ax_fx.xaxis.set_minor_locator(MultipleLocator(.01))
        ax_fx.grid(True, which="major", ls="--", color="blue")
        ax_fx.grid(True, which="minor", ls=".", color="blue")
        ax_fx.set_title(r"$f(x)$ and $\|\vec{\mathrm{cv}}\|_2$")
        ax_fx.set_xlabel("evaluation")
        ax_fx.set_ylabel(r"obj. value $f(x)$", color="blue")
        ax_fx.set_xlim([0, get_config().max_eval])
        ax_fx.set_yscale('symlog', linthreshy=0.001)
        ax_fx.set_ylim((0, 1))
        # for tl in self.ax_fx.get_yticklabels():
        #  tl.set_color('blue')
        self.min_plot, = ax_fx.plot(
            [], [], linestyle='-', marker='o', color="blue", zorder=-1)

        # cv plot
        # self.ax_cv = ax_cv = self.ax_fx.twinx()
        self.ax_cv = ax_cv = ax_fx
        # ax_cv.yaxis.tick_right()
        # ax_cv.grid(True, which="major", ls="--", color="red")
        # ax_cv.grid(True, which="minor", ls=".", color="red")
        # ax_cv.set_ylabel(r'constr. viol. $\|\vec{\mathrm{cv}}\|_2$',
        # color='red')
        ax_fx.set_ylabel(r"$f(x)-min(f(x))$ and $\|\vec{\mathrm{cv}}\|_2$")
                         #, color="blue")
        ax_cv.set_xlim([0, get_config().max_eval])
        # ax_cv.set_yscale('symlog', linthreshy=0.001)
        ax_cv.set_ylim((0, 1))
        # for tl in self.ax_cv.get_yticklabels():
        #  tl.set_color('red')
        self.cv_plot, = ax_cv.plot(
            [], [], linestyle='-', marker='o', color="red", zorder=-1)

        from matplotlib.widgets import Cursor
        self.fx_cursor = Cursor(
            self.ax_cv, useblit=True, color='black', alpha=0.5)

        vbox = gtk.VBox(False, 0)
        vbox.pack_start(self.fx_canvas, True, True)
        self.toolbar = NavigationToolbar(self.fx_canvas, self)
        vbox.pack_start(self.toolbar, False, False)
        return "f(x)", vbox

    def _init_plot_eval(self):
        from ui import NavigationToolbar
        from matplotlib import colorbar
        import gtk
        mx = self.problem.dim
        vbox = gtk.VBox(False, 0)
        if mx <= 1:
            vbox.add(gtk.Label("not enough dimensions"))
            return
        self.eval_canvas, fig = self.ui.mk_canvas()
        self.eval_ax = fig.add_subplot(111)
        self.eval_cb_ax, _ = colorbar.make_axes(self.eval_ax)

        spinner_hbox = gtk.HBox(gtk.FALSE, 5)

        def mk_cb(l):
            cb = gtk.combo_box_new_text()
            [cb.append_text('Axis %d' % i) for i in range(0, mx)]
            cb.set_active(mk_cb.i)
            mk_cb.i += 1
            spinner_hbox.add(gtk.Label(l))
            spinner_hbox.add(cb)
            return cb
        mk_cb.i = 0

        cb_0 = mk_cb("X Coord:")
        cb_1 = mk_cb("Y Coord:")

        for cb in [cb_0, cb_1]:
            cb.connect('changed', self.on_eval_spinner, cb_0, cb_1)

        self.eval_btn = btn = gtk.Button("Redraw")
        btn.connect('clicked', self.on_eval_spinner, cb_0, cb_1)
        spinner_hbox.add(btn)

        vbox.pack_start(self.eval_canvas, True, True)
        vbox.pack_start(spinner_hbox, False, False)
        self.toolbar = NavigationToolbar(self.eval_canvas, self)
        vbox.pack_start(self.toolbar, False, False)
        return "Values", vbox

    def on_finished(self):
        if hasattr(self, "eval_btn"):
            self.eval_btn.clicked()

    def on_eval_spinner(self, widget, cb0, cb1):
        from matplotlib import colorbar
        cx = cb0.get_active()
        cy = cb1.get_active()
        if cx == cy:
            self.logger.debug("eval plot: cx == cy and discarded")
            return
        if len(self.results.results) == 0:
            return

        px = 0  # self.problem.ranges[cx] / 10.
        py = 0  # self.problem.ranges[cy] / 10.
        xmin, xmax = self.problem.box[cx, :]
        ymin, ymax = self.problem.box[cy, :]
        xmin, xmax = xmin - px, xmax + px
        ymin, ymax = ymin - py, ymax + py

        rslts = zip(*[(r.x[cx], r.x[cy], r.fx) for r in self.results.results])
        x, y, z = rslts
        xi = np.linspace(xmin, xmax, 30)
        yi = np.linspace(ymin, ymax, 30)
        # grid the data.
        from matplotlib.mlab import griddata
        zi = griddata(x, y, z, xi, yi, interp='linear')
        # ci = griddata(x,y,c,xi,yi,interp='linear')

        self.eval_ax.clear()
        self.eval_cb_ax.clear()
        self.eval_ax.grid(True, which="both", ls="-", color='grey')
        # contour the gridded data
        # constraint violation
        # self.eval_ax.contourf(xi, yi, ci, 10, colors='k', zorder=5, alpha=.5, levels=[0,1])
        # f(x)
        self.eval_ax.contour(
            xi, yi, zi, 10, linewidths=0.5, colors='k', zorder=3)
        from matplotlib.pylab import cm
        cf = self.eval_ax.contourf(xi, yi, zi, 10, cmap=cm.jet, zorder=2)
        cb = colorbar.Colorbar(self.eval_cb_ax, cf)
        cf.colorbar = cb

        # plot data points
        self.eval_ax.scatter(x, y, marker='o', c='b', s=5, zorder=10)
        self.eval_ax.set_xlim((xmin, xmax))
        self.eval_ax.set_ylim((ymin, ymax))
        self.ui.redraw_canvas(self.eval_canvas)

    def _update_fx_plot(self, plt, ax, xval, yval):
        xx = np.append(plt.get_xdata(), xval)
        if xval < 0:
            xx = map(lambda _: _ - xval, xx)
        yy = np.append(plt.get_ydata(), yval)
        if yval < 0:
            yy = map(lambda _: _ - yval, yy)
        plt.set_xdata(xx)
        plt.set_ydata(yy)
        ylim = [0,  # min(ax.get_ylim()[0], yval),
                max(ax.get_ylim()[1], max(plt.get_ydata()))]
        ax.set_ylim(ylim)
        self.ui.redraw_canvas(self.fx_canvas)

    def _init_plot_pareto(self):
        from ui import NavigationToolbar
        from matplotlib.widgets import Cursor, Slider
        from matplotlib.axes import Axes
        import gtk
        # view = gtk.TextView()
        # view.set_cursor_visible(False)
        # view.set_editable(False)
        # buffer = view.get_buffer()
        # iter = buffer.get_iter_at_offset(0)
        # for i in range(100):
        #  buffer.insert(iter, " " * i + "Line %d\n" % i)
        # scrolled_window = gtk.ScrolledWindow()
        # scrolled_window.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        # scrolled_window.add(view)
        # return "Pareto", scrolled_window

        self.pf_canvas, fig = self.ui.mk_canvas()

        # self.pf_ax = pf_ax = fig.add_subplot(1,1,1)
        self.pf_ax = pf_ax = Axes(fig, [0.1, 0.2, 0.8, 0.7])
        fig.add_axes(self.pf_ax)

        pf_ax.grid(True, which="both", ls="-", color='grey')
        pf_ax.set_title("Pareto Front")
        pf_ax.set_xlabel("constr. violation")
        pf_ax.set_ylabel("obj. value")

        self.pf_plt_pnts = np.empty(shape=(0, 2))
        self.pf_plt, = pf_ax.plot(
            [], [], marker='o', ls='', alpha=.3, color='black')

        self.pf_cursor = Cursor(pf_ax, useblit=True, color='black', alpha=0.5)

        axcolor = 'lightgoldenrodyellow'
        pf_slider_ax = Axes(fig, [0.1, 0.04, 0.8, 0.04], axisbg=axcolor)
        fig.add_axes(pf_slider_ax)
        v = int(get_config().max_eval * 1.1)
        self.pf_slider = Slider(
            pf_slider_ax, '#', 0, v, valfmt="%d", valinit=v)
        self.pf_slider.on_changed(self.on_pf_slide)

        pf_vbox = gtk.VBox(False, 0)
        pf_vbox.pack_start(self.pf_canvas, True, True)
        self.toolbar = NavigationToolbar(self.pf_canvas, self)
        pf_vbox.pack_start(self.toolbar, False, False)
        return "Pareto", pf_vbox

    def on_pf_slide(self, val):
        val = int(val)
        self.pf_plt.set_xdata(self.pf_plt_pnts[:val, 0])
        self.pf_plt.set_ydata(self.pf_plt_pnts[:val, 1])
        self.ui.redraw_canvas(self.pf_canvas)

    @property
    def best(self):
        '''
        Currently best :class:`~panobbgo_lib.lib.Result`.

        .. Note::

          At the moment, this is :attr:`.pareto` but might change.
        '''
        return self.pareto

    @property
    def cv(self):
        '''
        The point with currently minimal constraint violation, likely 0.0.
        If there are several points with the same minimal constraint violation,
        the value of the objective function is a secondary selection argument.
        '''
        return self._cv

    @property
    def pareto(self):
        return self._pareto

    @property
    def min(self):
        '''
        The point, with currently smallest objective function value.
        '''
        return self._min

    @property
    def pareto_front(self):
        '''
        This is the list of points building the current pareto front.

        .. Note::

          This is a shallow copy.
        '''
        return self._pareto_front[:]

    def _update_pareto(self, result):
        '''
        Update the pareto front with this new @result.

        Either ignore it, or add it to the front and remove
        all points from the front which are obsolete.
        '''
        # Note: result.pp returns np.array([cv, fx])
        # add the new point
        pf_old = self.pareto_front

        # old code for convex front, below the one for a monotone step function
        # from utils import is_left
        # pf = self.pareto_front
        ## pf needs to be sorted
        # pf.append(result)
        # pf = sorted(pf)
        ## ... and re-calculate the front
        # new_front = pf[:1]
        # for p in pf[1:]:
        #  new_front.append(p)
        #  # next point needs to be left (smaller cv) and and above (higher fx)
        #  while len(new_front) > 1 and new_front[-1].cv >= new_front[-2].cv:
        #    del new_front[-1]
        #  # always a "right turn", concerning the ".pp" pareto points
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
            # TODO remove this check
            self._check_pareto_front()

            if len(self.pareto_front) > 2:
                self.logger.debug("pareto: %s" % map(
                    lambda x: (x.cv, x.fx), self.pareto_front))
            self.eventbus.publish("new_pareto_front", front=new_front)

    def _check_pareto_front(self):
        '''
        just used for testing
        '''
        pf = self.pareto_front
        for p1, p2 in zip(pf[:-1], pf[1:]):
            assert p1.fx <= p2.fx, u'fx > fx for %s, %s' % (p1, p2)
            assert p1.cv >= p2.cv, u'cv < cv for %s, %s' % (p1, p2)
        # if len(pf) >= 3:
        #  from utils import is_left
        #  for p1, p2, p3 in zip(pf[:-2], pf[1:-1], pf[2:]):
        #    if is_left(p1.pp, p2.pp, p3.pp):
        # self.logger.critical('is_left %s' % map(lambda _:_.pp, [p1, p2, p3]))

    def on_new_results(self, results):
        for r in results:
            if (self._min is None) or (r.fx < self._min.fx) or (r.fx == self._min.fx and r.cv < self._min.cv):
                # self.logger.info(u"\u2318 %s by %s" %(r, r.who))
                self._min = r
                self.eventbus.publish("new_min", min=r)

            if (self._cv is None) or (r.cv < self._cv.cv) or (r.cv == self._cv.cv and r.fx < self._cv.fx):
                self._cv = r
                self.eventbus.publish("new_cv", cv=r)

            # the pareto is weighted by the _min.cv and _cv.fx values
            # if pareto.cv is 0.0, then just the fx value counts
            weight = np.array([self._cv.fx, self._min.cv])
            if self._pareto is None \
                or (self._pareto.cv == 0.0 and r.cv == 0.0 and self._pareto.fx > r.fx)  \
                or self._pareto.cv > 0.0 and \
                weight.dot([self._pareto.cv, self._pareto.fx]) >\
                    weight.dot([r.cv, r.fx]):
                self._pareto = r
                self.eventbus.publish("new_pareto", pareto=r)
                self.eventbus.publish("new_best", best=r)

            self._update_pareto(r)
        self._update_pf_plot(results)

    def on_new_pareto(self, pareto):
        # self.logger.info("pareto: %s" % pareto)
        pass

    def _update_pf_plot(self, results):
        if not hasattr(self, "pf_plt"):
            return
        plt = self.pf_plt
        pnts = self.pf_plt_pnts
        a = [pnts]
        a.extend([r.pp for r in results])
        pnts = np.vstack(a)
        self.pf_plt_pnts = pnts
        plt.set_xdata(pnts[:, 0])
        plt.set_ydata(pnts[:, 1])
        self.ui.redraw_canvas(self.pf_canvas)

    def on_new_pareto_front(self, front):
        if not hasattr(self, "pf_ax"):
            return
        # self.ax1.clear()
        pnts = map(lambda x: x.pp, front)
        # insert points to make a staircase
        inserts = []
        for p1, p2 in zip(pnts[:-1], pnts[1:]):
            inserts.append((p1[0], p2[1]))
        all_pnts = []
        for i in range(len(inserts)):
            all_pnts.append(pnts[i])
            all_pnts.append(inserts[i])
        all_pnts.append(pnts[-1])
        data = zip(*all_pnts)
        self.pf_ax.plot(
            data[0], data[1], '-', alpha=.7, color="black")  # ms = ?
        self.pf_ax.autoscale()  # TODO get rid of autoscale
        self.ui.redraw_canvas(self.pf_canvas)

    def on_new_cv(self, cv):
        if hasattr(self, "cv_plot"):
            self._update_fx_plot(self.cv_plot, self.ax_cv, cv.cnt, cv.cv)

    def on_new_min(self, min):
        if hasattr(self, "min_plot"):
            self._update_fx_plot(self.min_plot, self.ax_fx, min.cnt, min.fx)


class Grid(Analyzer):
    '''
    packs nearby points into grid boxes
    '''
    def __init__(self):
        Analyzer.__init__(self)

    def _init_(self):
        # grid for storing points which are nearby.
        # maps from rounded coordinates tuple to point
        self._grid = dict()
        self._grid_div = 5.
        self._grid_lengths = self.problem.ranges / float(self._grid_div)

    def in_same_grid(self, point):
        key = tuple(self._grid_mapping(point.x))
        return self._grid.get(key, [])

    def _grid_mapping(self, x):
        from numpy import floor
        l = self._grid_lengths
        # m = self._problem.box[:,0]
        return tuple(floor(x / l) * l)

    def _grid_add(self, r):
        key = self._grid_mapping(r.x)
        box = self._grid.get(key, [])
        box.append(r)
        self._grid[key] = box
        # print ' '.join('%2s' % str(len(self._grid[k])) for k in
        # sorted(self._grid.keys()))

    def on_new_results(self, results):
        for result in results:
            self._grid_add(result)

#
# Splitter + inside is its Box class
#


class Splitter(Analyzer):
    '''
    Manages a tree of splits.
    Each split in this tree is a :class:`box <.Splitter.Box>`, which
    partitions the search space into smaller boxes and can have children.
    Boxes without children are :attr:`leafs <.Splitter.Box.leaf>`.

    The goal for this splitter is to balance between the
    depth level of splits and the number of points inside such a box.

    A heuristic can build upon this hierarchy
    to investigate interesting subregions.
    '''
    def __init__(self):
        Analyzer.__init__(self)
        # split, if there are more than this number of points in the box
        self.leafs = []
        self._id = 0  # block id
        self.logger = get_config().get_logger('SPLIT')  # , 10)
        self.max_eval = get_config().max_eval
        # _new_result used to signal get_leaf and others when there
        # are updates regarding box/split/leaf status
        from threading import Condition
        self._new_result = Condition()

    def _init_(self):
        # root box is equal to problem's box
        self.dim = self.problem.dim
        self.limit = max(20, self.max_eval / self.dim ** 2)
        self.logger.debug("limit = %s" % self.limit)
        self.root = Splitter.Box(None, self, self.problem.box.copy())
        self.leafs.append(self.root)
        # big boxes
        self.biggest_leaf = self.root
        self.big_by_depth = {}
        self.big_by_depth[self.root.depth] = self.root
        self.max_depth = self.root.depth
        # best box (with best f(x))
        self.best_box = None
        # in which box (a list!) is each point?
        from collections import defaultdict
        self.result2boxes = defaultdict(list)
        self.result2leaf = {}

    def _new_box(self, new_box):
        '''
        Called for each new box when there is a split.
        E.g. it updates the ``biggest`` box and related
        information for each depth level.
        '''
        self.max_depth = max(new_box.depth, self.max_depth)

        old_biggest_leaf = self.biggest_leaf
        self.biggest_leaf = max(self.leafs, key=lambda l: l.log_volume)
        if old_biggest_leaf is not self.biggest_leaf:
            self.eventbus.publish('new_biggest_leaf', box=new_box)

        dpth = new_box.depth
        # also consider the parent depth level
        for d in [dpth - 1, dpth]:
            old_big_by_depth = self.big_by_depth.get(d, None)
            if old_big_by_depth is None:
                self.big_by_depth[d] = new_box
            else:
                leafs_at_depth = list(
                    filter(lambda l: l.depth == d, self.leafs))
                if len(leafs_at_depth) > 0:
                    self.big_by_depth[d] = max(
                        leafs_at_depth, key=lambda l: l.log_volume)

            if self.big_by_depth[d] is not old_big_by_depth:
                self.eventbus.publish('new_biggest_by_depth',
                                      depth=d, box=self.big_by_depth[d])

    def on_new_biggest_leaf(self, box):
        self.logger.debug("biggest leaf at depth %d -> %s" % (box.depth, box))

    def on_new_biggest_by_depth(self, depth, box):
        self.logger.debug("big by depth: %d -> %s" % (depth, box))

    def get_box(self, point):
        '''
        return "leftmost" leaf box, where given point is contained in
        '''
        box = self.root
        while not box.leaf:
            box = box.get_child_boxes(point)[0]
        return box

    def get_all_boxes(self, result):
        '''
        return all boxes, where point is contained in
        '''
        assert isinstance(result, Result)
        return self.result2boxes[result]

    def get_leaf(self, result):
        '''
        returns the leaf box, where given result is currently sitting in
        '''
        assert isinstance(result, Result)
        # it might happen, that the result isn't in the result2leaf map
        # then we have to wait until on_new_results got it
        with self._new_result:
            while result not in self.result2leaf:
                # logger.info("RESULT NOT FOUND %s" % result)
                # logger.info("BOXES: %s" % self.get_all_boxes(result))
                self._new_result.wait()
        return self.result2leaf[result]

    def in_same_leaf(self, result):
        l = self.get_leaf(result)
        return l.results, l

    def on_new_results(self, results):
        with self._new_result:
            for result in results:
                self.root += result
            self._new_result.notify_all()
        # logger.info("leafs: %s" % map(lambda x:(x.depth, len(x)), self.leafs))
        # logger.info("point %s in boxes: %s" % (result.x, self.get_all_boxes(result)))
        # logger.info("point %s in leaf: %s" % (result.x, self.get_leaf(result)))
        # assert self.get_all_boxes(result)[-1] == self.get_leaf(result)

    def on_new_split(self, box, children, dim):
        self.logger.debug("Split: %s" % box)
        for i, chld in enumerate(children):
            self.logger.debug(" +ch%d: %s" % (i, chld))
        # logger.info("children: %s" % map(lambda x:(x.depth, len(x)),
        # children))

        # update self.best_box
        # check if new box contains the best point (>= because it could
        # be a child box)
        for new_box in children:
            if self.best_box is None or self.best_box.fx >= new_box.fx:
                self.best_box = new_box
        self.eventbus.publish('new_best_box', best_box=self.best_box)

    class Box(object):
        '''
        Used by :class:`.Splitter`, therefore nested.

        Most important routine is :meth:`.split`.

        .. Note::

          In the future, this might be refactored to allow different
          splitting methods.
        '''
        def __init__(self, parent, splitter, box):
            self.parent = parent
            self.logger = splitter.logger
            self.depth = parent.depth + 1 if parent else 0
            self.box = box
            self.splitter = splitter
            self.limit = splitter.limit
            self.dim = splitter.dim
            self.best = None              # best point
            self.results = []
            self.children = []
            self.split_dim = None
            self.id = splitter._id
            splitter._id += 1

        @property
        def leaf(self):
            '''
            returns ``true``, if this box is a leaf. i.e. no children
            '''
            return len(self.children) == 0

        @property
        def fx(self):
            '''
            Function value of best point in this particular box.
            '''
            return self.best.fx

        @memoize
        def __ranges(self):
            return self.box.ptp(axis=1)  # self.box[:,1] - self.box[:,0]

        @property
        def ranges(self):
            '''
            Gives back a vector with all the ranges of this box,
            i.e. upper - lower bound.
            '''
            return self.__ranges()

        @memoize
        def __log_volume(self):
            return np.sum(np.log(self.ranges))

        @property
        def log_volume(self):
            '''
            Returns the `logarithmic` volume of this box.
            '''
            return self.__log_volume()

        @memoize
        def __volume(self):
            return np.exp(self.log_volume)

        @property
        def volume(self):
            '''
            Returns the volume of the box.

            .. Note::

              Currently, the exponential of :attr:`.log_volume`
            '''
            return self.__volume()

        def _register_result(self, result):
            '''
            This updates the splitter and box specific datatypes,
            i.e. the maps from a result to the corresponding boxes or leafs.
            '''
            assert isinstance(result, Result)
            self.results.append(result)

            # new best result in box? (for best fx value, too)
            if self.best is not None:
                if self.best.fx > result.fx:
                    self.best = result
            else:
                self.best = result

            self.splitter.result2boxes[result].append(self)
            if self.leaf:
                self.splitter.result2leaf[result] = self

        def add_result(self, result):
            '''
            Registers and adds a new :class:`~panobbgo_lib.lib.Result`.
            In particular, it adds the given ``result`` to the
            current box and it's children (also all descendents).

            If the current box is a leaf and too big, the :meth:`.split`
            routine is called.

            .. Note::

              ``box += result`` is fine, too.
            '''
            self._register_result(result)
            if not self.leaf:
                for child in self.get_child_boxes(result.x):
                    child += result  # recursive
            elif self.leaf and len(self.results) >= self.limit:
                self.split()

        def __iadd__(self, result):
            '''Convenience wrapper for :meth:`.add_result`.'''
            self.add_result(result)
            return self

        def __len__(self):
            return len(self.results)

        def split(self, dim=None):
            '''
            Arguments::

            - ``dim``: Dimension, along which to split. (default: `None`, and calculated)
            '''
            assert self.leaf, 'only leaf boxes are allowed to be split'
            if dim is None:
                # scaled_coords = np.vstack(map(lambda r:r.x, self.results)) / self.ranges
                # dim = np.argmax(np.std(scaled_coords, axis=0))
                dim = np.argmax(self.ranges)
            # self.logger.debug("dim: %d" % dim)
            assert dim >= 0 and dim < self.dim, 'dimension along where to split is %d' % dim
            b1 = Splitter.Box(self, self.splitter, self.box.copy())
            b2 = Splitter.Box(self, self.splitter, self.box.copy())
            self.split_dim = dim
            # split_point = np.median(map(lambda r:r.x[dim], self.results))
            split_point = np.average(map(lambda r: r.x[dim], self.results))
            b1.box[dim, 1] = split_point
            b2.box[dim, 0] = split_point
            self.children.extend([b1, b2])
            self.splitter.leafs.remove(self)
            map(self.splitter.leafs.append, self.children)
            for c in self.children:
                self.splitter._new_box(c)
                for r in self.results:
                    if c.contains(r.x):
                        c._register_result(r)
            self.splitter.eventbus.publish('new_split',
                                           box=self, children=self.children, dim=dim)

        def contains(self, point):
            '''
            true, if given point is inside this box (including boundaries).
            '''
            l, u = self.box[:, 0], self.box[:, 1]
            return (l <= point).all() and (u >= point).all()

        def get_child_boxes(self, point):
            '''
            returns all immediate child boxes, which contain given point.
            '''
            assert not self.leaf, 'not applicable for "leaf" box'
            ret = filter(lambda c: c.contains(point), self.children)
            assert len(ret) > 0, "no child box containing %s found!" % point
            return ret

        def __repr__(self):
            v = self.volume
            l = ',leaf' if self.leaf else ''
            l = '(%d,%.3f%s) ' % (len(self), v, l)
            b = ','.join('%s' % _ for _ in self.box)
            return 'Box-%d %s[%s]' % (self.id, l, b)

# end Splitter
