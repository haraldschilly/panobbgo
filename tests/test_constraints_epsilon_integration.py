from panobbgo.core import StrategyBase
from panobbgo.lib import Problem
from panobbgo.lib.constraints import EpsilonConstraintHandler
import numpy as np

class MockProblem(Problem):
    def __init__(self):
        super().__init__(box=[(-5.0, 5.0)]*2)
    def eval(self, x): return np.sum(x**2)

def test_strategy_epsilon_init():
    p = MockProblem()
    # Pass config overrides
    s = StrategyBase(p, constraint_handler="EpsilonConstraintHandler",
                     epsilon_start=2.0, epsilon_cp=3.0, epsilon_cutoff=50)

    assert isinstance(s.constraint_handler, EpsilonConstraintHandler)
    assert s.constraint_handler.epsilon_start == 2.0
    assert s.constraint_handler.cp == 3.0
    assert s.constraint_handler.cutoff == 50
