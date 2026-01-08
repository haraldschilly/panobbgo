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

r"""
Analyzers
=========

Analyzers, just like :mod:`.heuristics`, listen to events
and change their internal state based on local and
global data. They can emit :class:`events <panobbgo.core.Event>`
on their own and they are accessible via the
:meth:`~panobbgo.core.StrategyBase.analyzer` method of
the strategy.

.. inheritance-diagram:: panobbgo.analyzers
    :parts: 1

.. automodule:: panobbgo.analyzers.best
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: panobbgo.analyzers.grid
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: panobbgo.analyzers.dedensifyer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: panobbgo.analyzers.splitter
   :members:
   :undoc-members:
   :show-inheritance:

.. codeauthor:: Harald Schilly <harald.schilly@univie.ac.at>
"""

from __future__ import absolute_import
from __future__ import unicode_literals

from .best import Best
from .splitter import Splitter
from .grid import Grid
from .dedensifyer import Dedensifyer
