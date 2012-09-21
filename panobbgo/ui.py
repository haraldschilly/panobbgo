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
'''
import pygtk
pygtk.require('2.0')
import gtk
from gtk import gdk

import matplotlib
matplotlib.use('GTKAgg') # 'GTKAgg' or 'GTK'
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NavigationToolbar
from matplotlib.figure import Figure

from threading import Thread

from core import Module

class UI(Module, gtk.Window, Thread):
  r'''
  UI
  '''
  def __init__(self):
    Module.__init__(self)

    gtk.Window.__init__(self)
    self.set_default_size(600, 600)
    self.connect('destroy', lambda win: gtk.main_quit())
    self.set_title('Panobbgo')
    self.set_border_width(5)

    self.vbox = gtk.VBox(False, 5)
    self.add(self.vbox)

    label = gtk.Label("This is the UI")
    self.vbox.pack_start(label, False, False)

    self.init_pareto()

    self.toolbar = NavigationToolbar(self.canvas, self)
    self.vbox.pack_start(self.toolbar, False, False)

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
    self.dirty = False # used to indicate, if something needs to be drawn

    self.fig = Figure(figsize=(10,10))
    self.pfplt = self.fig.add_subplot(1,1,1)
    from matplotlib.ticker import MultipleLocator
    self.pfplt.xaxis.set_major_locator(MultipleLocator(.1))
    self.pfplt.xaxis.set_minor_locator(MultipleLocator(.01))
    self.pfplt.xaxis.grid(True,'minor',linewidth=.5)
    self.pfplt.yaxis.grid(True,'minor',linewidth=.5)
    self.pfplt.xaxis.grid(True,'major',linewidth=1)
    self.pfplt.yaxis.grid(True,'major',linewidth=1)
    self.pfplt.set_title("Pareto Front")
    self.pfplt.set_xlabel("constr. violation")
    self.pfplt.set_ylabel("obj. value")

    self.canvas = FigureCanvas(self.fig) # gtk.DrawingArea
    self.vbox.pack_start(self.canvas, True, True)

  def on_new_pareto_front(self, front):
    #self.ax1.clear()
    self.pfplt.plot(*zip(*map(lambda x:x.pp, front)))
    self.pfplt.autoscale()
    self.dirty = True

  def draw(self):
    def task():
      while True:
        if self.dirty:
          gtk.threads_enter()
          try:
            self.canvas.draw()
            self.dirty = False
          finally:
            gtk.threads_leave()
        from IPython.utils.timing import time
        time.sleep(.1)

    self.t = Thread(target=task)
    self.t.daemon = True
    self.t.start()

  def finish(self):
    '''called by base strategy in _cleanup for shutdown'''
    #plt.ioff()
    self.join() # not necessary, since not a daemon
