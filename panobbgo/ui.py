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
import matplotlib.pyplot as plt
from core import Module

class UI(Module):
  r'''
  UI
  '''
  def __init__(self):
    Module.__init__(self)
    plt.ion()
    self.fig = plt.figure()
    self.ax1 = self.fig.add_subplot(1,1,1)
    self.ax1.set_title("Pareto Front")
    plt.show()

  def on_new_pareto_front(self, front):
    #self.ax1.clear()
    self.ax1.plot(*zip(*map(lambda x:x.pp, front)))
    self.draw()

  def draw(self):
    self.fig.canvas.draw()

  def ioff(self):
    plt.ioff()
