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

from panobbgo.core import Analyzer

import numpy as np


class Best(Analyzer):

    """
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
    """

    def __init__(self, strategy):
        Analyzer.__init__(self, strategy)
        self.logger = self.config.get_logger("BEST")
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
        ax_fx.set_xlim([0, self.config.max_eval])
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
        ax_cv.set_xlim([0, self.config.max_eval])
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
        v = int(self.config.max_eval * 1.1)
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
        """
        Currently best :class:`~panobbgo_lib.lib.Result`.

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
            # TODO remove this check
            self._check_pareto_front()

            if len(self.pareto_front) > 2:
                self.logger.debug("pareto: %s" % map(
                    lambda x: (x.cv, x.fx), self.pareto_front))
            self.eventbus.publish("new_pareto_front", front=new_front)

    def _check_pareto_front(self):
        """
        just used for testing
        """
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
