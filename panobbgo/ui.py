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

class UI(Module, gtk.Window, Thread):
  r'''
  UI
  '''
  def __init__(self):
    Module.__init__(self)
    config = get_config()
    self.logger = config.get_logger("UI")

    self.dirty = False # used to indicate, if something needs to be drawn

    gtk.Window.__init__(self)
    self.set_default_size(1000, 500)
    self.connect('destroy', self.destroy)
    self.set_title('Panobbgo %s@%s' % (config.version, config.git_head[:8]))
    self.set_border_width(3)

    self.top_hbox = gtk.HBox(False, 0)

    self.pf_vbox = gtk.VBox(False, 0)
    self.top_hbox.add(self.pf_vbox)
    self.fx_vbox = gtk.VBox(False, 0)
    self.top_hbox.add(self.fx_vbox)
    self.add(self.top_hbox)

    #label = gtk.Label("This is the UI")
    #self.pf_vbox.pack_start(label, False, False)

    self.init_pareto()
    self.init_fx()

    self.add_events(gdk.BUTTON_PRESS_MASK |
                    gdk.KEY_PRESS_MASK|
                    gdk.KEY_RELEASE_MASK)

    self.show_all()
    gdk.threads_init()
    #def run_gtk_main():
    #self.mt = Thread(target=run_gtk_main)
    #self.mt.start()
    Thread.__init__(self)
    self.start()
    self.draw()

  def run(self):
    gtk.threads_enter()
    gtk.main()
    gtk.threads_leave()

  def destroy(self, win):
    self.logger.warning("window destroyed")
    gtk.main_quit()

  def init_pareto(self):
    fig = Figure(figsize=(10,10))
    self.pf_canvas = FigureCanvas(fig) # gtk.DrawingArea

    #self.pf_ax = pf_ax = fig.add_subplot(1,1,1)
    self.pf_ax = pf_ax = Axes(fig, [0.1, 0.2, 0.8, 0.7])
    fig.add_axes(self.pf_ax)

    #from matplotlib.ticker import MultipleLocator
    #pf_ax.xaxis.set_major_locator(MultipleLocator(1))
    #pf_ax.xaxis.set_minor_locator(MultipleLocator(.1))
    #pf_ax.xaxis.grid(True,'major',linewidth=1)
    #pf_ax.yaxis.grid(True,'major',linewidth=1)
    #pf_ax.xaxis.grid(True,'minor',linewidth=.5)
    #pf_ax.yaxis.grid(True,'minor',linewidth=.5)
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
    self.ax_fx = fig.add_subplot(1,1,1)
    #from matplotlib.ticker import MultipleLocator
    #self.ax_fx.xaxis.set_major_locator(MultipleLocator(.1))
    #self.ax_fx.xaxis.set_minor_locator(MultipleLocator(.01))
    self.ax_fx.grid(True, which="both", ls="-", color="grey") # ls="-."
    self.ax_fx.set_title(r"$f(x)$ and $\|\vec{\mathrm{cv}}\|_2$")
    self.ax_fx.set_xlabel("evaluation")
    self.ax_fx.set_ylabel(r"obj. value $f(x)$", color="blue")
    self.ax_fx.set_xlim([0, get_config().max_eval])

    self.min_plot, = self.ax_fx.plot([], [], linestyle='--', marker='o', color="blue", zorder=-1)
    for tl in self.ax_fx.get_yticklabels():
      tl.set_color('blue')

    # cv plot
    self.ax_cv = self.ax_fx.twinx()
    self.ax_cv.set_ylabel(r'constr. viol. $\|\vec{\mathrm{cv}}\|_2$', color='red')
    self.cv_plot,  = self.ax_cv.plot([], [], linestyle='--', marker='o', color="red", zorder=-1)
    self.ax_cv.set_xlim([0, get_config().max_eval])
    for tl in self.ax_cv.get_yticklabels():
      tl.set_color('red')

    self.fx_cursor = Cursor(self.ax_cv, useblit=True, color='black', alpha=0.5)

    self.fx_vbox.pack_start(self.fx_canvas, True, True)
    self.toolbar = NavigationToolbar(self.fx_canvas, self)
    self.fx_vbox.pack_start(self.toolbar, False, False)

  def on_new_pareto_front(self, front):
    #self.ax1.clear()
    self.pf_ax.plot(*zip(*map(lambda x:x.pp, front)))
    self.pf_ax.autoscale()
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

  def _update_plot(self, plt, ax, xval, yval):
    plt.set_xdata(np.append(plt.get_xdata(), xval))
    plt.set_ydata(np.append(plt.get_ydata(), yval))
    ylim = [min(ax.get_ylim()[0], xval),
            max(ax.get_ylim()[1], yval)]
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
