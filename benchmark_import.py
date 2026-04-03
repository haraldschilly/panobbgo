import timeit

setup = """
import numpy as np
def _solve_lp_direction():
    from scipy.optimize import linprog
    pass
"""

stmt = "_solve_lp_direction()"

# Run the benchmark
times = timeit.repeat(stmt, setup=setup, number=10000, repeat=5)
print(f"Repeated import: {min(times):.4f} seconds per 10000 calls")

setup_opt = """
import numpy as np
try:
    from scipy.optimize import linprog
except ImportError:
    linprog = None

def _solve_lp_direction():
    if linprog is None:
        raise ImportError
    pass
"""

stmt_opt = "_solve_lp_direction()"

times_opt = timeit.repeat(stmt_opt, setup=setup_opt, number=10000, repeat=5)
print(f"Module-level import: {min(times_opt):.4f} seconds per 10000 calls")
