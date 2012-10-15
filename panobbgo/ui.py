# -*- coding: utf-8 -*-
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
User Interface
--------------

This draws a window and plots graphs.

.. figure:: img/ui1.png
   :scale:  75 %
'''
from config import get_config
import numpy as np
from core import Module
from threading import Thread

import pygtk
pygtk.require('2.0')
import gtk
from gtk import gdk

import matplotlib
matplotlib.use('GTKAgg') # 'GTKAgg' or 'GTK'
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import Slider, Cursor # SpanSelector
from gtk import Label
from matplotlib.axes import Axes
from matplotlib import colorbar

class UI(Module, gtk.Window, Thread):
  r'''
  UI
  '''
  def __init__(self):
    Module.__init__(self)
    gtk.Window.__init__(self, gtk.WINDOW_TOPLEVEL)
    self.set_position(gtk.WIN_POS_CENTER)
    Thread.__init__(self)

  def show(self):
    self.dirty = False # used to indicate, if something needs to be drawn
    config = get_config()
    self.logger = config.get_logger("UI")

    self.set_default_size(900, 800)
    self.connect('destroy', self.destroy)
    self.set_title('Panobbgo %s@%s' % (config.version, config.git_head[:8]))
    self.set_border_width(3)

    self.top_hbox = gtk.HBox(False, 0)

    self.notebook = notebook = gtk.Notebook()
    notebook.set_tab_pos(gtk.POS_LEFT)
    self.top_hbox.add(notebook)
    notebook.show()

    self.pf_vbox = gtk.VBox(False, 0)
    self.add_notebook_page(self.pf_vbox, "Pareto")
    self.fx_vbox = gtk.VBox(False, 0)
    self.add_notebook_page(self.fx_vbox, "Optimal")
    self.eval_vbox = gtk.VBox(False, 0)
    self.add_notebook_page(self.eval_vbox, "Values")

    self.add(self.top_hbox)

    self.init_pareto()
    self.init_fx()
    self.init_eval()

    self.add_events(gdk.BUTTON_PRESS_MASK |
                    gdk.KEY_PRESS_MASK|
                    gdk.KEY_RELEASE_MASK)

    self.show_all()
    gdk.threads_init()
    #def run_gtk_main():
    #self.mt = Thread(target=run_gtk_main)
    #self.mt.start()
    self.start()
    self.draw()

  def add_notebook_page(self, frame, label_text):
    label = Label(label_text)
    self.notebook.append_page(frame, label)

  def run(self):
    gtk.threads_enter()
    gtk.main()
    gtk.threads_leave()

  def destroy(self, win):
    self.logger.warning("window destroyed")
    gtk.main_quit()

  def init_eval(self):
    mx = self.problem.dim
    if mx <= 1:
      self.eval_vbox.add(Label("not enoguh dimensions"))
      return
    self.eval_fig = fig = Figure(figsize=(10,10))
    self.eval_canvas = FigureCanvas(fig)
    self.eval_ax = fig.add_subplot(111)
    self.eval_cb_ax, _ = colorbar.make_axes(self.eval_ax)

    spinner_hbox = gtk.HBox(gtk.FALSE, 5)
    adj0 = gtk.Adjustment(0, 0, mx-1, 1, 1, 0)
    spinner_0 = gtk.SpinButton(adj0, 0, 0)
    spinner_hbox.add(Label("x coord:"))
    spinner_hbox.add(spinner_0)

    adj1 = gtk.Adjustment(1, 0, mx-1, 1, 1, 0)
    spinner_1 = gtk.SpinButton(adj1, 0, 0)
    spinner_hbox.add(Label('y coord:'))
    spinner_hbox.add(spinner_1)

    adj0.connect('value_changed', self.on_eval_spinner, spinner_0, spinner_1)
    adj1.connect('value_changed', self.on_eval_spinner, spinner_0, spinner_1)

    self.eval_btn = btn = gtk.Button("redraw")
    btn.connect('clicked', self.on_eval_spinner, spinner_0, spinner_1)
    spinner_hbox.add(btn)

    self.eval_vbox.pack_start(self.eval_canvas, True, True)
    self.eval_vbox.pack_start(spinner_hbox, False, False)
    self.toolbar = NavigationToolbar(self.eval_canvas, self)
    self.eval_vbox.pack_start(self.toolbar, False, False)

  def init_pareto(self):
    fig = Figure(figsize=(10,10))
    self.pf_canvas = FigureCanvas(fig) # gtk.DrawingArea

    #self.pf_ax = pf_ax = fig.add_subplot(1,1,1)
    self.pf_ax = pf_ax = Axes(fig, [0.1, 0.2, 0.8, 0.7])
    fig.add_axes(self.pf_ax)

    pf_ax.grid(True, which="both", ls="-", color='grey')
    pf_ax.set_title("Pareto Front")
    pf_ax.set_xlabel("constr. violation")
    pf_ax.set_ylabel("obj. value")

    self.pf_plt_pnts = np.empty(shape = (0, 2))
    self.pf_plt, = pf_ax.plot([], [], marker='o', ls = '', alpha=.3, color='black')

    self.pf_cursor = Cursor(pf_ax, useblit=True, color='black', alpha=0.5)

    axcolor = 'lightgoldenrodyellow'
    pf_slider_ax = Axes(fig, [0.1, 0.04, 0.8, 0.04], axisbg=axcolor)
    fig.add_axes(pf_slider_ax)
    v = int(get_config().max_eval * 1.1)
    self.pf_slider = Slider(pf_slider_ax, '#', 0, v, valfmt  ="%d", valinit = v)
    self.pf_slider.on_changed(self.on_pf_slide)

    self.pf_vbox.pack_start(self.pf_canvas, True, True)
    self.toolbar = NavigationToolbar(self.pf_canvas, self)
    self.pf_vbox.pack_start(self.toolbar, False, False)

  def init_fx(self):
    fig = Figure(figsize=(10,10))
    self.fx_canvas = FigureCanvas(fig) # gtk.DrawingArea

    # f(x) plot
    self.ax_fx = ax_fx = fig.add_subplot(1,1,1)
    #from matplotlib.ticker import MultipleLocator
    #ax_fx.xaxis.set_major_locator(MultipleLocator(.1))
    #ax_fx.xaxis.set_minor_locator(MultipleLocator(.01))
    ax_fx.grid(True, which="major", ls="--", color="blue")
    ax_fx.grid(True, which="minor", ls=".", color="blue")
    ax_fx.set_title(r"$f(x)$ and $\|\vec{\mathrm{cv}}\|_2$")
    ax_fx.set_xlabel("evaluation")
    ax_fx.set_ylabel(r"obj. value $f(x)$", color="blue")
    ax_fx.set_xlim([0, get_config().max_eval])
    ax_fx.set_yscale('symlog', linthreshy=0.001)
    ax_fx.set_ylim((0, 1))
    #for tl in self.ax_fx.get_yticklabels():
    #  tl.set_color('blue')
    self.min_plot, = ax_fx.plot([], [], linestyle='-', marker='o', color="blue", zorder=-1)

    # cv plot
    #self.ax_cv = ax_cv = self.ax_fx.twinx()
    self.ax_cv = ax_cv = ax_fx
    #ax_cv.yaxis.tick_right()
    #ax_cv.grid(True, which="major", ls="--", color="red")
    #ax_cv.grid(True, which="minor", ls=".", color="red")
    #ax_cv.set_ylabel(r'constr. viol. $\|\vec{\mathrm{cv}}\|_2$', color='red')
    ax_fx.set_ylabel(r"$f(x)-min(f(x))$ and $\|\vec{\mathrm{cv}}\|_2$") #, color="blue")
    ax_cv.set_xlim([0, get_config().max_eval])
    #ax_cv.set_yscale('symlog', linthreshy=0.001)
    ax_cv.set_ylim((0, 1))
    #for tl in self.ax_cv.get_yticklabels():
    #  tl.set_color('red')
    self.cv_plot,  = ax_cv.plot([], [], linestyle='-', marker='o', color="red", zorder=-1)

    self.fx_cursor = Cursor(self.ax_cv, useblit=True, color='black', alpha=0.5)

    self.fx_vbox.pack_start(self.fx_canvas, True, True)
    self.toolbar = NavigationToolbar(self.fx_canvas, self)
    self.fx_vbox.pack_start(self.toolbar, False, False)

  def on_new_pareto_front(self, front):
    #self.ax1.clear()
    pnts = map(lambda x : x.pp, front)
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
    self.pf_ax.plot(data[0], data[1], '-', alpha=.7, color="black") # ms = ?
    self.pf_ax.autoscale() # TODO get rid of autoscale
    self.dirty = True

  def on_pf_slide(self, val):
    val = int(val)
    self.pf_plt.set_xdata(self.pf_plt_pnts[:val,0])
    self.pf_plt.set_ydata(self.pf_plt_pnts[:val,1])
    self.dirty = True

  def on_new_results(self, results):
    plt = self.pf_plt
    pnts = self.pf_plt_pnts
    for r in results:
      pnts = np.vstack((pnts, r.pp))
      #self.pf_ax.plot(r.pp[0], r.pp[1], marker='.', alpha=.5, color='black')
      #self.ax_fx.plot(r.cnt, r.fx, marker='.', alpha=.5)
      #ylim = [min(self.ax_fx.get_ylim()[0], r.fx),
      #        max(self.ax_fx.get_ylim()[1], r.fx)]
    #self.ax_fx.set_ylim(ylim)
    self.pf_plt_pnts = pnts
    plt.set_xdata(pnts[:,0])
    plt.set_ydata(pnts[:,1])
    self.dirty = True

  def on_finished(self):
    self.eval_btn.clicked()

  def on_eval_spinner(self, widget, spinner0, spinner1):
    cx = spinner0.get_value_as_int()
    cy = spinner1.get_value_as_int()
    if cx == cy:
      self.logger.debug("eval plot: cx == cy and discarded")
      return
    if len(self.results.results) == 0:
      return

    px = 0 #self.problem.ranges[cx] / 10.
    py = 0 #self.problem.ranges[cy] / 10.
    xmin, xmax = self.problem.box[cx, :]
    ymin, ymax = self.problem.box[cy, :]
    xmin, xmax = xmin-px, xmax+px
    ymin, ymax = ymin-py, ymax+py

    rslts = zip(*[(r.x[cx], r.x[cy], r.fx) for r in self.results.results])
    x, y, z = rslts
    xi = np.linspace(xmin, xmax, 30)
    yi = np.linspace(ymin, ymax, 30)
    # grid the data.
    from matplotlib.mlab import griddata
    zi = griddata(x,y,z,xi,yi,interp='linear')
    #ci = griddata(x,y,c,xi,yi,interp='linear')

    self.eval_ax.clear()
    self.eval_cb_ax.clear()
    self.eval_ax.grid(True, which="both", ls="-", color='grey')
    # contour the gridded data
    # constraint violation
    #self.eval_ax.contourf(xi, yi, ci, 10, colors='k', zorder=5, alpha=.5, levels=[0,1])
    # f(x)
    self.eval_ax.contour(xi,yi,zi,10,linewidths=0.5,colors='k', zorder=3)
    from matplotlib.pylab import cm
    cf = self.eval_ax.contourf(xi,yi,zi,10,cmap=cm.jet, zorder=2)
    cb = colorbar.Colorbar(self.eval_cb_ax, cf)
    cf.colorbar = cb

    # plot data points.
    self.eval_ax.scatter(x,y,marker='o',c='b',s=5,zorder=10)
    self.eval_ax.set_xlim((xmin, xmax))
    self.eval_ax.set_ylim((ymin, ymax))
    self.dirty = True

  def _update_plot(self, plt, ax, xval, yval):
    xx = np.append(plt.get_xdata(), xval)
    if xval < 0: xx = map(lambda _:_ - xval, xx)
    yy = np.append(plt.get_ydata(), yval)
    if yval < 0: yy = map(lambda _:_ - yval, yy)
    plt.set_xdata(xx)
    plt.set_ydata(yy)
    ylim = [0, #min(ax.get_ylim()[0], yval),
            max(ax.get_ylim()[1], max(plt.get_ydata()))]
    ax.set_ylim(ylim)
    self.dirty = True

  def on_new_cv(self, cv):
    self._update_plot(self.cv_plot, self.ax_cv, cv.cnt, cv.cv)

  def on_new_min(self, min):
    self._update_plot(self.min_plot, self.ax_fx, min.cnt, min.fx)

  def draw(self):
    def task():
      while True:
        if self.dirty:
          gtk.threads_enter()
          try:
            self.dirty = False
            self.fx_canvas.draw()
            self.pf_canvas.draw()
            self.eval_canvas.draw()
          finally:
            gtk.threads_leave()
        from IPython.utils.timing import time
        time.sleep(get_config().ui_redraw_delay)

    self.t = Thread(target=task)
    self.t.daemon = True
    self.t.start()

  def finish(self):
    '''called by base strategy in _cleanup for shutdown'''
    #plt.ioff()
    self.join() # not necessary, since not a daemon
