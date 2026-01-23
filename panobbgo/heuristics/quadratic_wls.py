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

from panobbgo.core import HeuristicSubprocess
import numpy as np
from functools import reduce
import operator


class QuadraticWlsModel(HeuristicSubprocess):
    """
    This heuristic uses an quadratic OLS model to find an approximate new best point
    for each new best box (the latter is subject to change).

    The actual calculation is performed out of process.
    """

    def __init__(self, strategy):
        HeuristicSubprocess.__init__(self, strategy)
        self.logger = self.config.get_logger("H:WLS")

    @staticmethod
    def subprocess(pipe):
        def predict(xx):
            """
            helper for the while loop:
            calculates the prediction based on the model result
            """
            dim = len(xx)
            res = [1]
            res.extend(xx)
            for i in range(dim - 1):
                for j in range(i + 1, dim):
                    res.append(xx[i] * xx[j])
            for i in range(dim):
                res.append(xx[i] ** 2)
            return result.predict(np.array(res))

        while True:
            points, bounds, best_point, fx_vals = pipe.recv()
            dim = points.shape[1]

            import numpy as np
            from pandas import DataFrame
            import statsmodels.api as sm

            # import statsmodels.formula.api as sm_formula
            data = {}
            for i in range(dim):
                data["x%s" % i] = points[:, i]
            for i in range(dim):
                for j in range(i + 1, dim):
                    data["x%s:x%s" % (i, j)] = points[:, i] * points[:, j]
            for i in range(dim):
                data["x%s^2" % i] = points[:, i] ** 2
            data.update({"Intercept": np.ones(len(points))})
            X = DataFrame(data)
            # X.columns =
            cols = ["Intercept"] + ["x%i" % i for i in range(dim)]
            mixedterms = reduce(
                operator.add,
                [["x%s:x%s" % (i, j) for j in range(i + 1, dim)] for i in range(dim)],
            )
            cols.extend(mixedterms)
            cols.extend(["x%s^2" % i for i in range(dim)])
            X.columns = cols

            y = DataFrame({"y": fx_vals})

            # Type hint for apply_along_axis is tricky, ignoring for now as linalg.norm is well-defined
            distances = np.apply_along_axis(np.linalg.norm, 1, points - best_point) # type: ignore
            weights = 1.0 / (1 + np.argsort(distances))

            model = sm.WLS(y, X, weights=weights)  # type: ignore
            result = model.fit()

            # optimize predict with x \in bounds
            from scipy.optimize import fmin_l_bfgs_b

            sol, _, _ = fmin_l_bfgs_b(
                predict, np.zeros(dim), bounds=bounds, approx_grad=True
            )

            pipe.send(sol)
            # end while loop

    def on_new_best_box(self, best_box):
        # self.logger.info("")
        # self.logger.debug("best_box.best: %s" % best_box.best)
        pointarray = np.r_[[r.x for r in best_box.results]]
        # self.logger.debug("pointarray: \n%s" % pointarray)
        get_val = self.strategy.constraint_handler.get_penalty_value
        fx_vals = np.array([get_val(r) for r in best_box.results])
        self.pipe.send((pointarray, self.problem.box, best_box.best.x, fx_vals))
        sol = self.pipe.recv()
        # print 'solution:', sol
        self.emit(sol)
