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

from panobbgo.utils import PanobbgoTestCase


class AnalyzersUtils(PanobbgoTestCase):

    def setUp(self):
        from panobbgo_lib.classic import RosenbrockConstraint
        self.problem = RosenbrockConstraint(2)
        self.strategy = self.init_strategy()

    def test_best(self):
        from panobbgo.analyzers import Best
        best = Best(self.strategy)
        assert best is not None
        results = self.random_results(2, 100, pcv=.99)
        #for r in results:
        #    print unicode(r)
        best.on_new_results(results)
        best._check_pareto_front()
        print "Pareto Front:"
        for r in best.pareto_front:
            print unicode(r)


if __name__ == '__main__':
    import unittest
    unittest.main()
