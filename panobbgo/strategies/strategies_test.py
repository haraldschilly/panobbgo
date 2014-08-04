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
from panobbgo.core import StrategyBase

from panobbgo.utils import PanobbgoTestCase
from . import StrategyRoundRobin, StrategyRewarding
from panobbgo_lib.classic import Rosenbrock
import mock


def get_my_setup_cluster():
    def my_setup_cluster(self, nb_gens, problem):
        print nb_gens, problem
    return my_setup_cluster


class StrategiesTests(PanobbgoTestCase):

    def setUp(self):
        self.problem = Rosenbrock(3)

    @mock.patch('panobbgo.core.StrategyBase._setup_cluster', new_callable=get_my_setup_cluster)
    def test_round_robin(self, my_setup_cluster):
        rr = StrategyRoundRobin(self.problem, size=20)
        assert rr.size == 20

    @mock.patch('panobbgo.core.StrategyBase._setup_cluster', new_callable=get_my_setup_cluster)
    def test_rewarding(self, my_setup_cluster):
        rwd = StrategyRewarding(self.problem)
        assert rwd is not None


if __name__ == '__main__':
    import unittest
    unittest.main()
