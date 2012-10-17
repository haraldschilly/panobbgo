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
from matplotlib.axes import Axes
from matplotlib import colorbar

class UI(Module, gtk.Window, Thread):
  r'''
  UI
  '''
  def __init__(self):
    Module.__init__(self)
    gtk.Window.__init__(self, gtk.WINDOW_TOPLEVEL)
    # fill current window
    screen = self.get_screen()
    monitor = screen.get_monitor_at_window(self.get_root_window())
    geom = screen.get_monitor_geometry(monitor)
    self.set_resize_mode(gtk.RESIZE_QUEUE)
    s = min(map(lambda _ : int(_ * .8), [geom.width, geom.height]))
    self.resize(int(s * 4./3.), s)
    # centered
    self.set_position(gtk.WIN_POS_CENTER)
    Thread.__init__(self)

  def show(self):
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

    #self.pf_vbox = gtk.VBox(False, 0)
    #self.add_notebook_page("Pareto", self.pf_vbox)
    self.fx_vbox = gtk.VBox(False, 0)
    self.add_notebook_page("Optimal", self.fx_vbox)
    self.eval_vbox = gtk.VBox(False, 0)
    self.add_notebook_page("Values", self.eval_vbox)

    self.add(self.top_hbox)

    #self.init_pareto()
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
    #self.draw()

  def add_notebook_page(self, label_text, frame):
    if label_text is None or frame is None: return
    label = gtk.Label(label_text)
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
      self.eval_vbox.add(gtk.Label("not enoguh dimensions"))
      return
    self.eval_fig = fig = Figure(figsize=(10,10))
    self.eval_canvas = FigureCanvas(fig)
    self.eval_ax = fig.add_subplot(111)
    self.eval_cb_ax, _ = colorbar.make_axes(self.eval_ax)

    spinner_hbox = gtk.HBox(gtk.FALSE, 5)
    adj0 = gtk.Adjustment(0, 0, mx-1, 1, 1, 0)
    spinner_0 = gtk.SpinButton(adj0, 0, 0)
    spinner_hbox.add(gtk.Label("x coord:"))
    spinner_hbox.add(spinner_0)

    adj1 = gtk.Adjustment(1, 0, mx-1, 1, 1, 0)
    spinner_1 = gtk.SpinButton(adj1, 0, 0)
    spinner_hbox.add(gtk.Label('y coord:'))
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

    # plot data points
    self.eval_ax.scatter(x,y,marker='o',c='b',s=5,zorder=10)
    self.eval_ax.set_xlim((xmin, xmax))
    self.eval_ax.set_ylim((ymin, ymax))
    self.eval_canvas._need_redraw = True

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
    self.fx_canvas._need_redraw = True

  def on_new_cv(self, cv):
    self._update_plot(self.cv_plot, self.ax_cv, cv.cnt, cv.cv)

  def on_new_min(self, min):
    self._update_plot(self.min_plot, self.ax_fx, min.cnt, min.fx)

  def finish(self):
    '''called by base strategy in _cleanup for shutdown'''
    #plt.ioff()
    self.join() # not necessary, since not a daemon
