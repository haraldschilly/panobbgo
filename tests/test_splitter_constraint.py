from panobbgo.lib import Problem, Point, Result
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.analyzers.splitter import Splitter
from panobbgo.lib.constraints import DefaultConstraintHandler
import numpy as np

class SimpleConstrainedProblem(Problem):
    def __init__(self):
        # Pass a list of tuples for box. Length 1.
        super().__init__([(-10, 10)])

    def eval(self, x):
        return float(x[0])

    def eval_constraints(self, x):
        # x >= 0 is feasible
        # x < 0 is infeasible
        if x[0] < 0:
            return np.array([abs(x[0])])
        return np.array([0.0])

def test_splitter_respects_constraints():
    problem = SimpleConstrainedProblem()
    strategy = StrategyRewarding(problem, testing_mode=True)

    # Manually set constraint handler to Default (prioritizes feasibility)
    strategy.constraint_handler = DefaultConstraintHandler(strategy)

    # Manually add points
    # Point 1: Infeasible, very low fx (-100)
    p1 = Point(np.array([-100.0]), "manual")
    r1 = Result(p1, -100.0, cv_vec=np.array([100.0]))

    # Point 2: Feasible, higher fx (0)
    p2 = Point(np.array([0.0]), "manual")
    r2 = Result(p2, 0.0, cv_vec=np.array([0.0]))

    splitter = Splitter(strategy)
    strategy._analyzers["Splitter"] = splitter
    splitter.__start__()

    # Feed results to Splitter via on_new_results
    splitter.on_new_results([r1])
    splitter.on_new_results([r2])

    # Check root box's local best
    # Since root box contains both points, its 'best' attribute determines what Splitter thinks is best in this region.
    best_in_box = splitter.root.best

    print(f"Root box best point: fx={best_in_box.fx}, cv={best_in_box.cv}")

    # Current behavior: Box selects point with lowest fx regardless of constraints
    # Expected behavior: Box should use constraint_handler.is_better

    # DefaultConstraintHandler prefers feasible (cv=0) over infeasible (cv=100) regardless of fx
    # So it should pick r2 (fx=0, cv=0) over r1 (fx=-100, cv=100)

    if best_in_box.cv > 0:
        print("FAIL: Splitter Box selected infeasible point as best.")
        assert False, "Splitter Box selected infeasible point as best."
    else:
        print("PASS: Splitter Box selected feasible point as best.")
        assert True

if __name__ == "__main__":
    test_splitter_respects_constraints()
