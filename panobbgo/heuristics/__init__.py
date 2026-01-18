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

"""
Heuristics
==========

The following heuristics for generating new points are implemented.
They are all derived from :class:`~panobbgo.core.Heuristic`.

.. inheritance-diagram:: panobbgo.heuristics

"""

from .center import Center
from .zero import Zero
from .random import Random
from .extremal import Extremal
from .latin_hypercube import LatinHypercube
from .nearby import Nearby
from .weighted_average import WeightedAverage
from .nelder_mead import NelderMead
from .lbfgsb import LBFGSB
from .quadratic_wls import QuadraticWlsModel
from .gaussian_process import GaussianProcessHeuristic
from .feasible_search import FeasibleSearch
from .constraint_gradient import ConstraintGradient
from .local_penalty_search import LocalPenaltySearch

__all__ = [
    "Center",
    "Zero",
    "Random",
    "Extremal",
    "LatinHypercube",
    "Nearby",
    "WeightedAverage",
    "NelderMead",
    "LBFGSB",
    "QuadraticWlsModel",
    "GaussianProcessHeuristic",
    "FeasibleSearch",
    "ConstraintGradient",
    "LocalPenaltySearch",
]
