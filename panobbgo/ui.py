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

class UI(Module, gtk.Window, Thread):
  r'''
  UI
  '''
  def __init__(self):
    Module.__init__(self)

    self.dirty = False # used to indicate, if something needs to be drawn

    gtk.Window.__init__(self)
    self.set_default_size(1000, 500)
    self.connect('destroy', lambda win: gtk.main_quit())
    self.set_title('Panobbgo')
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

  def init_pareto(self):
    fig = Figure(figsize=(10,10))
    self.pf_plt = fig.add_subplot(1,1,1)
    from matplotlib.ticker import MultipleLocator
    self.pf_plt.xaxis.set_major_locator(MultipleLocator(1))
    self.pf_plt.xaxis.set_minor_locator(MultipleLocator(.1))
    self.pf_plt.xaxis.grid(True,'major',linewidth=1)
    self.pf_plt.yaxis.grid(True,'major',linewidth=1)
    self.pf_plt.xaxis.grid(True,'minor',linewidth=.5)
    self.pf_plt.yaxis.grid(True,'minor',linewidth=.5)
    self.pf_plt.set_title("Pareto Front")
    self.pf_plt.set_xlabel("constr. violation")
    self.pf_plt.set_ylabel("obj. value")

    self.pf_canvas = FigureCanvas(fig) # gtk.DrawingArea
    self.pf_vbox.pack_start(self.pf_canvas, True, True)

    self.toolbar = NavigationToolbar(self.pf_canvas, self)
    self.pf_vbox.pack_start(self.toolbar, False, False)

  def init_fx(self):
    fig = Figure(figsize=(10,10))
    self.fx_plt = fig.add_subplot(1,1,1)
    #from matplotlib.ticker import MultipleLocator
    #self.fx_plt.xaxis.set_major_locator(MultipleLocator(.1))
    #self.fx_plt.xaxis.set_minor_locator(MultipleLocator(.01))
    self.fx_plt.grid(True, which="both", ls="-.")
    self.fx_plt.set_title("f(x)")
    self.fx_plt.set_xlabel("evaluation")
    self.fx_plt.set_ylabel("obj. value")
    self.fx_plt.set_xlim([0, get_config().max_eval])

    self.fx_canvas = FigureCanvas(fig) # gtk.DrawingArea
    self.fx_vbox.pack_start(self.fx_canvas, True, True)
    self.best_plot, = self.fx_plt.plot([], [], linestyle='--', marker='o', color="red", zorder=-1)

    self.toolbar = NavigationToolbar(self.fx_canvas, self)
    self.fx_vbox.pack_start(self.toolbar, False, False)

  def on_new_pareto_front(self, front):
    #self.ax1.clear()
    self.pf_plt.plot(*zip(*map(lambda x:x.pp, front)))
    self.pf_plt.autoscale()
    self.dirty = True

  def on_new_results(self, results):
    for r in results:
      self.fx_plt.plot(r.cnt, r.fx, marker='.', alpha=.5)
      ylim = [min(self.fx_plt.get_ylim()[0], r.fx),
              max(self.fx_plt.get_ylim()[1], r.fx)]
    self.fx_plt.set_ylim(ylim)
    self.dirty = True

  def on_new_best(self, best):
    print best.cnt
    self.best_plot.set_xdata(np.append(self.best_plot.get_xdata(), best.cnt))
    self.best_plot.set_ydata(np.append(self.best_plot.get_ydata(), best.fx))
    self.dirty = True

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
