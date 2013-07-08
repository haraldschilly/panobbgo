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

from panobbgo.core import Heuristic, StopHeuristic

class Center(Heuristic):
    '''
    This heuristic checks the point in the center of the box.
    '''
    def __init__(self):
        Heuristic.__init__(self, name="Center", cap=1)

    def on_start(self):
        box = self.problem.box
        return box[:, 0] + (box[:, 1] - box[:, 0]) / 2.0

