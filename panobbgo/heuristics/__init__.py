# -*- coding: utf8 -*-
# Copyright 2012 Harald Schilly <harald.schilly@gmail.com>
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
Heuristics
==========
The main idea behind all heuristics is, ...

Each heuristic needs to listen to at least one stream of
:class:`Events <panobbgo.core.Event>` from the :class:`~panobbgo.core.EventBus`.
Most likely, it is the `one-shot` event ``start``, which is
:meth:`published <panobbgo.core.EventBus.publish>` by the
:class:`~panobbgo.core.StrategyBase`.

.. inheritance-diagram:: panobbgo.heuristics

.. codeauthor:: Harald Schilly <harald.schilly@gmail.com>
"""

from __future__ import absolute_import
from __future__ import unicode_literals

from .center import Center
from .extremal import Extremal
from .gaussian_process import GaussianProcessHeuristic
from .latin_hypercube import LatinHypercube
from .lbfgsb import LBFGSB
from .nearby import Nearby
from .nelder_mead import NelderMead
from .quadratic_wls import QuadraticWlsModel
from .random import Random
from .weighted_average import WeightedAverage
from .zero import Zero
