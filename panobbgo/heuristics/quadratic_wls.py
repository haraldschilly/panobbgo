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
        import numpy as np
        from pandas import DataFrame
        import statsmodels.api as sm
        from scipy.optimize import fmin_l_bfgs_b
        import traceback

        while True:
            try:
                # Wait for input
                if not pipe.poll(1.0):
                    continue

                payload = pipe.recv()
                points, bounds, best_point, fx_vals = payload
                dim = points.shape[1]

                # Build data dictionary in the correct order for statsmodels (Intercept first)
                # Use vectorized slicing instead of list comprehensions for performance
                data = {"Intercept": np.ones(len(points))}
                for i in range(dim):
                    data["x%s" % i] = points[:, i]
                for i in range(dim):
                    for j in range(i + 1, dim):
                        data["x%s:x%s" % (i, j)] = points[:, i] * points[:, j]
                for i in range(dim):
                    data["x%s^2" % i] = points[:, i] ** 2

                # DataFrame creation preserves insertion order in Python 3.7+
                X = DataFrame(data)

                # Define columns explicitly for clarity, though it now matches X's order
                cols = ["Intercept"] + ["x%i" % i for i in range(dim)]
                mixedterms = reduce(
                    operator.add,
                    [["x%s:x%s" % (i, j) for j in range(i + 1, dim)] for i in range(dim)],
                )
                cols.extend(mixedterms)
                cols.extend(["x%s^2" % i for i in range(dim)])
                X.columns = cols

                y = DataFrame({"y": fx_vals})

                # Optimized distance calculation using axis parameter instead of apply_along_axis
                distances = np.linalg.norm(points - best_point, axis=1)
                weights = 1.0 / (1 + np.argsort(distances))

                model = sm.WLS(y, X, weights=weights)  # type: ignore
                result = model.fit()

                def predict(xx):
                    """
                    helper for the while loop:
                    calculates the prediction based on the model result
                    """
                    # dim is from outer scope (loop variable)
                    # result is from outer scope (loop variable)

                    # Use outer product for mixed terms to avoid nested loops
                    outer = np.outer(xx, xx)
                    mixed = outer[np.triu_indices(dim, k=1)]
                    # Construct feature vector: [1, x0..xn, mixed, squared]
                    res = np.concatenate(([1], xx, mixed, xx**2))
                    return result.predict(res)

                # optimize predict with x \in bounds

                sol, _, _ = fmin_l_bfgs_b(
                    predict, np.zeros(dim), bounds=bounds, approx_grad=True
                )

                pipe.send(sol)
            except EOFError:
                break
            except Exception:
                # Log the error to stderr so it's visible in tests/logs
                traceback.print_exc()
                # Send None to indicate failure and prevent parent from hanging indefinitely
                try:
                    pipe.send(None)
                except:
                    pass

    def on_new_best_box(self, best_box):
        # self.logger.info("")
        # self.logger.debug("best_box.best: %s" % best_box.best)
        pointarray = np.r_[[r.x for r in best_box.results]]
        # self.logger.debug("pointarray: \n%s" % pointarray)
        get_val = self.strategy.constraint_handler.get_penalty_value
        fx_vals = np.array([get_val(r) for r in best_box.results])

        try:
            # Send bounds as list of tuples for scipy compatibility
            # self.problem.box is a BoundingBox object which might not be fully compatible
            bounds = [tuple(row) for row in self.problem.box.box]
            self.pipe.send((pointarray, bounds, best_box.best.x, fx_vals))

            # Use poll with timeout to prevent hanging if subprocess crashes/hangs
            if self.pipe.poll(10.0):
                sol = self.pipe.recv()
                if sol is not None:
                    # print 'solution:', sol
                    self.emit(sol)
                else:
                    self.logger.warning("QuadraticWlsModel subprocess returned None (error occurred).")
            else:
                self.logger.warning("QuadraticWlsModel subprocess timed out waiting for solution.")
        except Exception as e:
            self.logger.error(f"Error communicating with QuadraticWlsModel subprocess: {e}")
