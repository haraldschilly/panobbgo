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

r"""
User Interface
--------------

This draws a window and plots graphs.

.. figure:: img/ui1.png
   :scale:  75 %
"""
from config import get_config
from core import Module
from threading import Thread

import pygtk
pygtk.require('2.0')
import gtk
from gtk import gdk

import matplotlib
import os
if os.environ.get("TRAVIS") == "true":
    matplotlib.use('Agg')  # 'GTKAgg' or 'GTK', or 'Agg' ?
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.backends.backend_agg import NavigationToolbar2Agg as NavigationToolbar
else:
    matplotlib.use('GTKAgg')  # 'GTKAgg' or 'GTK', or 'Agg' ?
    from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
    from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NavigationToolbar
del os
# from matplotlib.widgets import Slider, Cursor # SpanSelector
# from matplotlib.axes import Axes


class UI(Module, gtk.Window, Thread):

    r"""
    UI
    """

    def __init__(self):
        Module.__init__(self)
        gtk.Window.__init__(self, gtk.WINDOW_TOPLEVEL)
        # fill current window
        screen = self.get_screen()
        monitor = screen.get_monitor_at_window(self.get_root_window())
        geom = screen.get_monitor_geometry(monitor)
        self.set_resize_mode(gtk.RESIZE_QUEUE)
        s = min(map(lambda _: int(_ * .8), [geom.width, geom.height]))
        self.resize(int(s * 4. / 3.), s)
        self._canvases = set()
        # centered
        self.set_position(gtk.WIN_POS_CENTER)
        Thread.__init__(self)

    @staticmethod
    def mk_canvas():
        """
        Creates a FigureCanvas, ready to be added to a gtk layout element
        """
        from matplotlib.figure import Figure
        fig = Figure(figsize=(10, 10))
        return FigureCanvas(fig), fig

    def show(self):
        config = get_config()
        self.logger = config.get_logger("UI")

        self.set_default_size(900, 800)
        self.connect('destroy', self.destroy)
        self.set_title(
            'Panobbgo %s@%s' % (config.version, config.git_head[:8]))
        self.set_border_width(0)

        self.top_hbox = gtk.HBox(False, 0)

        self.notebook = notebook = gtk.Notebook()
        notebook.set_tab_pos(gtk.POS_LEFT)
        self.top_hbox.add(notebook)
        notebook.show()

        self.add(self.top_hbox)

        self.add_events(gdk.BUTTON_PRESS_MASK |
                        gdk.KEY_PRESS_MASK |
                        gdk.KEY_RELEASE_MASK)

        self.show_all()
        gdk.threads_init()
        # def run_gtk_main():
        # self.mt = Thread(target=run_gtk_main)
        # self.mt.start()
        self.start()
        self._auto_redraw()

    def _auto_redraw(self):
        def task():
            while True:
                gtk.threads_enter()
                try:
                    [c.draw_idle() for c in self._canvases if c._need_redraw]
                finally:
                    gtk.threads_leave()
                from IPython.utils.timing import time
                time.sleep(get_config().ui_redraw_delay)

        self.t = Thread(target=task)
        self.t.daemon = True
        self.t.start()

    def redraw_canvas(self, c):
        """
        If your canvas needs to be redrawn, pass it into this function.
        """
        assert isinstance(c, FigureCanvas)
        self._canvases.add(c)
        c._need_redraw = True

    def add_notebook_page(self, label_text, frame):
        assert label_text is not None and frame is not None
        label = gtk.Label(label_text)
        self.notebook.append_page(frame, label)
        frame.show_all()
        self.notebook.show_all()

    def run(self):
        gtk.threads_enter()
        gtk.main()
        gtk.threads_leave()

    def destroy(self, win):
        self.logger.info("window destroyed")
        gtk.main_quit()

    def finish(self):
        """called by base strategy in _cleanup for shutdown'''
        # plt.ioff()
        self.join()  # not necessary, since not a daemon
