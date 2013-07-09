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


class LBFGSB(Heuristic):

    '''
    This uses :func:`scipy.optimize.fmin_l_bfgs_b` in a subprocess.
    '''

    def __init__(self):
        Heuristic.__init__(self, cap=1)
        self.logger = get_config().get_logger("LBFGS")

    def _init_(self):
        from multiprocessing import Process, Pipe
        self.p1, self.p2 = Pipe()
        self.out1, self.out2 = Pipe(False)
        self.lbfgsb = Process(target=self.worker, args=(self.p2, self.out2,
                              self.problem.dim), name='%s-LBFGS' % self.name)
        self.lbfgsb.daemon = True
        self.lbfgsb.start()

    @staticmethod
    def worker(pipe, output, dims):
        from scipy.optimize import fmin_l_bfgs_b
        import numpy as np

        def f(x):
            pipe.send(x)
            fx = pipe.recv()
            return fx

        solution = fmin_l_bfgs_b(f, np.zeros(dims), approx_grad=True)
        print solution
        output.send(solution)

    def on_start(self):
        while True:
            if self.out1.poll(0):
                output = self.out1.recv()
                self.logger.info(output)
            x = self.p1.recv()
            self.logger.info("x: %s" % x)
            self.emit(x)

    def on_new_results(self, results):
        for result in results:
            if result.who == self.name:
                self.p1.send(result.fx)
