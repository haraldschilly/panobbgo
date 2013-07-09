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


class QuadraticWlsModel(Heuristic):

    '''
    This heuristic uses an quadratic OLS model to find an approximate new best point
    for each new best box (the latter is subject to change).

    The actual calculation is performed out of process.
    '''

    def __init__(self):
        Heuristic.__init__(self)
        self.logger = get_config().get_logger('H:QM')

        from multiprocessing import Process, Pipe
        # a pipe has two ends, parent and child.
        self.p1, self.p2 = Pipe()
        self.process = Process(
            target=self.solve_ols,
            args=(self.p2,),
            name='%s-Subprocess' % (self.name))
        self.process.daemon = True
        self.process.start()

    @staticmethod
    def solve_ols(pipe):
        def predict(xx):
            '''
            helper for the while loop: calculates the prediction
            based on the model result
            '''
            dim = len(xx)
            res = [1]
            res.extend(xx)
            for i in range(dim - 1):
                for j in range(i + 1, dim):
                    res.append(xx[i] * xx[j])
            for i in range(dim):
                res.append(xx[i] ** 2)
            return result.predct(np.array(res))

        while True:
            points, bounds = pipe.recv()
            dim = len(points[0].x)

            import numpy as np
            from pandas import DataFrame
            import statsmodels.api as sm
            data = {}
            for i in dim:
                data['x%s' % i] = [x.x[i] for x in points]
            X = DataFrame({'Intercept': np.ones(dim)})
            X = X.join(data)

            y = DataFrame({'y': [_.fx for _ in points]})

            model = sm.OLS(y, X)  # TODO WLS
            result = model.fit()

            # optimize predict with x \in bounds
            from scipy.optimize import fmin_l_bfgs_b
            sol, fval, info = fmin_l_bfgs_b(predict, np.zeros(dim),
                                            bounds=bounds, approx_grad=True)

            pipe.send((sol, fval, info))  # end while loop

    def on_new_best_box(self, best_box):
        # self.logger.info("")
        self.p1.send((best_box.points, self.problem.box))
        sol, fval, info = self.p1.recv()
        print 'solution:', sol
        print 'fval:', fval
        print 'info:', info
        self.emit(sol)
