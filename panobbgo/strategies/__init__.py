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
Strategies
==========

This part outlines the coordination between the point-producing
heuristics, the interaction with the cluster and the
:class:`DB of evaluated points <panobbgo.core.Results>`.

Basically, one or more threads produce points where to search,
and another one consumes them and dispatches tasks.
Subclass the :class:`~panobbgo.core.StrategyBase` class to implement
a new strategy.

.. inheritance-diagram:: panobbgo.strategies

.. codeauthor:: Harald Schilly <harald.schilly@gmail.com>
"""

from __future__ import absolute_import
from __future__ import unicode_literals

from .rewarding import StrategyRewarding
from .round_robin import StrategyRoundRobin
from .ucb import StrategyUCB

__all__ = ["StrategyRewarding", "StrategyRoundRobin", "StrategyUCB"]
