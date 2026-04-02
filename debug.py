import sys
sys.path.append('.')
from panobbgo.heuristics.differential_evolution import DifferentialEvolution
from tests.test_heuristic_de import MockStrategy
from panobbgo.lib import Point, Result, Rosenbrock
import numpy as np

problem = Rosenbrock(dim=2)
strategy = MockStrategy(problem)
de = DifferentialEvolution(strategy, NP=4)

# Initialize population manually to skip random part
for i in range(4):
    x = np.ones(2) * i # [0,0], [1,1], [2,2], [3,3]
    p = Point(x, de.name)
    r = Result(p, float(i*10))
    de.population[i] = r
de.pop_size = 4

# The issue is that we manually initialized without appending to de.active_indices!
# Let's fix active_indices in test or manually append.
# Actually, the test code bypasses `on_new_results` for the initial items and manually mutates `de.population`.
print("de.active_indices:", de.active_indices)
