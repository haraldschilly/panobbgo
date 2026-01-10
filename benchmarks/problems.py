"""
Benchmark Problems for Panobbgo Optimization Framework

This module defines a comprehensive battery of benchmark problems for evaluating
optimization strategies across various dimensions, shifted optima, and difficulty levels.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from panobbgo.lib.classic import (
    Rosenbrock, RosenbrockConstraint, RosenbrockAbs, RosenbrockAbsConstraint,
    RosenbrockStochastic, Himmelblau, Rastrigin, Ackley, Griewank,
    StyblinskiTang, Schwefel, DixonPrice, Zakharov, RosenbrockModified,
    RotatedEllipse, RotatedEllipse2, Ripple1, Ripple25, Shekel, DeJong,
    Quadruple, Powell, Trigonometric, SumDifferentPower, Step, Box, Wood,
    HelicalValley, Beale, NesterovQuadratic, Arwhead, Branin, GoldsteinPrice
)


@dataclass
class BenchmarkCase:
    """Represents a specific benchmark case with problem, dimension, and shift."""
    problem_name: str
    dimension: int
    shift_vector: np.ndarray  # Shift the global optimum
    global_optimum: Optional[np.ndarray]
    global_minimum: float
    search_bounds: Tuple[float, float]
    difficulty: str
    separable: bool
    unimodal: bool

    def create_problem(self):
        """Create the actual problem instance."""
        problem_class = PROBLEM_CLASSES[self.problem_name]
        try:
            problem = problem_class(dims=self.dimension)
        except TypeError:
             # Some problems don't accept dims in __init__ if they are fixed dim
             # or if they auto-detect from other args.
             # Check if it's one of those fixed dim problems
             try:
                 problem = problem_class()
             except Exception:
                 # Try with dim instead of dims (though most use dims or just **kwargs)
                 try:
                    problem = problem_class(dim=self.dimension)
                 except:
                    # Fallback for fixed dim problems that might ignore dims or fail
                    problem = problem_class()

        # Check if dimension matches requested
        if hasattr(problem, 'dim') and problem.dim != self.dimension:
             # This happens for fixed dimension problems like GoldsteinPrice (2D)
             # We can't force dimension, so we might need to skip or warn if mismatch
             # But for BenchmarkCase generation we should only generate valid dims.
             pass

        # Apply shift to the problem
        if np.any(self.shift_vector != 0):
            problem = ShiftedProblem(problem, self.shift_vector)

        return problem


class ShiftedProblem:
    """Wrapper to shift a problem's optimum."""

    def __init__(self, base_problem, shift_vector):
        self.base_problem = base_problem
        self.shift_vector = np.array(shift_vector)
        self.dim = base_problem.dim
        self.box = base_problem.box

    def __call__(self, point):
        # Shift the point before evaluation
        shifted_point = point.x - self.shift_vector
        shifted_point_obj = type(point)(shifted_point, point.who)
        return self.base_problem(shifted_point_obj)

    def __str__(self):
        return f"Shifted{self.base_problem} (shift={self.shift_vector})"


# Available problem classes
PROBLEM_CLASSES = {
    'Rosenbrock': Rosenbrock,
    'RosenbrockConstraint': RosenbrockConstraint,
    'RosenbrockAbs': RosenbrockAbs,
    'RosenbrockAbsConstraint': RosenbrockAbsConstraint,
    'RosenbrockStochastic': RosenbrockStochastic,
    'Himmelblau': Himmelblau,
    'Rastrigin': Rastrigin,
    'Ackley': Ackley,
    'Griewank': Griewank,
    'StyblinskiTang': StyblinskiTang,
    'Schwefel': Schwefel,
    'DixonPrice': DixonPrice,
    'Zakharov': Zakharov,
    'RosenbrockModified': RosenbrockModified,
    'RotatedEllipse': RotatedEllipse,
    'RotatedEllipse2': RotatedEllipse2,
    'Ripple1': Ripple1,
    'Ripple25': Ripple25,
    'Shekel': Shekel,
    'DeJong': DeJong,
    'Quadruple': Quadruple,
    'Powell': Powell,
    'Trigonometric': Trigonometric,
    'SumDifferentPower': SumDifferentPower,
    'Step': Step,
    'Box': Box,
    'Wood': Wood,
    'HelicalValley': HelicalValley,
    'Beale': Beale,
    'NesterovQuadratic': NesterovQuadratic,
    'Arwhead': Arwhead,
    'Branin': Branin,
    'GoldsteinPrice': GoldsteinPrice,
}


def generate_benchmark_battery() -> List[BenchmarkCase]:
    """Generate a comprehensive battery of benchmark cases."""
    cases = []

    # Standard shifts for testing
    shifts_2d = [
        np.array([0.0, 0.0]),      # Origin
        np.array([1.0, -0.5]),     # Shifted
    ]

    # --- 2D Problems ---
    # Problems that support arbitrary dimensions (we test 2D)
    variable_dim_problems = [
        'Rosenbrock', 'RosenbrockConstraint', 'RosenbrockAbs', 'RosenbrockAbsConstraint',
        'RosenbrockStochastic', 'Rastrigin', 'Ackley', 'Griewank', 'StyblinskiTang',
        'Schwefel', 'DixonPrice', 'Zakharov', 'DeJong', 'Quadruple',
        'Trigonometric', 'SumDifferentPower', 'Step', 'NesterovQuadratic', 'Arwhead'
    ]

    # Problems that are fixed dimension (mostly 2D, some 3D/4D)
    fixed_dim_problems_2d = [
        'Himmelblau', 'RosenbrockModified', 'RotatedEllipse', 'RotatedEllipse2',
        'Ripple1', 'Ripple25', 'Beale', 'Branin', 'GoldsteinPrice'
    ]

    # 3D
    fixed_dim_problems_3d = ['HelicalValley', 'Box']

    # 4D
    fixed_dim_problems_4d = ['Wood', 'Powell']

    all_2d = variable_dim_problems + fixed_dim_problems_2d

    for problem_name in all_2d:
        for shift in shifts_2d:
            # Determine global optimum and minimum
            global_min = 0.0 # Default for many
            global_opt = None

            if problem_name == 'Branin':
                global_min = 0.397887
                global_opt = np.array([-np.pi, 12.275]) + shift
            elif problem_name == 'GoldsteinPrice':
                global_min = 3.0
                global_opt = np.array([0.0, -1.0]) + shift
            elif problem_name == 'Himmelblau':
                global_min = 0.0
                global_opt = np.array([3.0, 2.0]) + shift  # One of the optima
            elif problem_name == 'StyblinskiTang':
                 # -39.16617 * 2 = -78.33234
                 global_min = -39.16617 * 2
                 global_opt = np.array([-2.903534, -2.903534]) + shift
            elif problem_name == 'Schwefel':
                 global_min = 0.0
                 global_opt = np.array([420.9687, 420.9687]) + shift
            elif problem_name == 'RosenbrockModified':
                global_min = 0.0
                global_opt = np.array([-1.0, -1.0]) + shift
            elif problem_name == 'Ackley' or problem_name == 'Rastrigin' or problem_name == 'Griewank' or \
                 problem_name == 'DeJong' or problem_name == 'Quadruple' or problem_name == 'Step' or \
                 problem_name == 'SumDifferentPower' or problem_name == 'RotatedEllipse' or \
                 problem_name == 'RotatedEllipse2' or problem_name == 'Zakharov':
                 global_min = 0.0
                 global_opt = shift.copy() # At origin (0,0) + shift
            elif problem_name == 'Rosenbrock':
                 global_min = 0.0
                 global_opt = np.array([1.0, 1.0]) + shift

            # If we don't know the exact location easily, we set global_opt to None to skip param distance check
            # For many fixed ones, we might not have exact coords handy in this script yet.

            case = BenchmarkCase(
                problem_name=problem_name,
                dimension=2,
                shift_vector=shift,
                global_optimum=global_opt,
                global_minimum=global_min,
                search_bounds=(-5.0, 5.0),
                difficulty=_get_difficulty(problem_name),
                separable=_is_separable(problem_name),
                unimodal=_is_unimodal(problem_name)
            )
            cases.append(case)

    # --- 3D Problems ---
    for problem_name in fixed_dim_problems_3d:
        dim = 3
        shift = np.zeros(dim)
        global_min = 0.0
        global_opt = None

        # HelicalValley min is at (1, 0, 0) if x1 > 0
        if problem_name == 'HelicalValley':
             global_opt = np.array([1.0, 0.0, 0.0])
        elif problem_name == 'Box':
            # Box has multiple solutions typically or specific ones based on m
            pass

        case = BenchmarkCase(
            problem_name=problem_name,
            dimension=dim,
            shift_vector=shift,
            global_optimum=global_opt,
            global_minimum=global_min,
            search_bounds=(-5.0, 5.0),
            difficulty=_get_difficulty(problem_name),
            separable=_is_separable(problem_name),
            unimodal=_is_unimodal(problem_name)
        )
        cases.append(case)

    # --- 4D Problems ---
    for problem_name in fixed_dim_problems_4d:
        dim = 4
        shift = np.zeros(dim)
        global_min = 0.0
        global_opt = None

        if problem_name == 'Wood':
            global_opt = np.ones(dim)
        elif problem_name == 'Powell':
            global_opt = np.zeros(dim)

        case = BenchmarkCase(
            problem_name=problem_name,
            dimension=dim,
            shift_vector=shift,
            global_optimum=global_opt,
            global_minimum=global_min,
            search_bounds=(-5.0, 5.0),
            difficulty=_get_difficulty(problem_name),
            separable=_is_separable(problem_name),
            unimodal=_is_unimodal(problem_name)
        )
        cases.append(case)

    # --- Higher Dimensional Problems (5D) ---
    for problem_name in ['DeJong', 'Rosenbrock', 'Rastrigin', 'Ackley', 'Griewank', 'StyblinskiTang', 'Schwefel', 'Zakharov']:
        dim = 5
        shift = np.zeros(dim)

        global_min = 0.0
        global_opt = None

        if problem_name == 'StyblinskiTang':
             global_min = -39.16617 * dim
             global_opt = np.full(dim, -2.903534)
        elif problem_name == 'Schwefel':
             global_min = 0.0
             global_opt = np.full(dim, 420.9687)
        elif problem_name == 'Rosenbrock':
             global_min = 0.0
             global_opt = np.ones(dim)
        else:
             global_opt = np.zeros(dim)

        case = BenchmarkCase(
            problem_name=problem_name,
            dimension=dim,
            shift_vector=shift,
            global_optimum=global_opt,
            global_minimum=global_min,
            search_bounds=(-5.0, 5.0),
            difficulty=_get_difficulty(problem_name),
            separable=_is_separable(problem_name),
            unimodal=_is_unimodal(problem_name)
        )
        cases.append(case)

    return cases


def _get_difficulty(problem_name: str) -> str:
    """Get difficulty level for a problem."""
    easy = ['DeJong', 'Quadruple', 'SumDifferentPower', 'Step', 'RotatedEllipse', 'RotatedEllipse2', 'NesterovQuadratic']
    medium = ['Rosenbrock', 'RosenbrockConstraint', 'RosenbrockAbs', 'RosenbrockAbsConstraint',
              'Himmelblau', 'Branin', 'DixonPrice', 'Beale', 'Wood', 'HelicalValley', 'Powell',
              'Arwhead', 'Trigonometric', 'Box']
    # Hard usually means highly multimodal

    if problem_name in easy:
        return 'easy'
    elif problem_name in medium:
        return 'medium'
    else:
        return 'hard'


def _is_separable(problem_name: str) -> bool:
    """Check if problem is separable."""
    # Separable: can be optimized dimension by dimension
    separable = ['DeJong', 'Rastrigin', 'Ackley', 'Schwefel', 'Step', 'SumDifferentPower', 'Quadruple']
    return problem_name in separable


def _is_unimodal(problem_name: str) -> bool:
    """Check if problem is unimodal."""
    unimodal = ['DeJong', 'Quadruple', 'SumDifferentPower', 'Step', 'RotatedEllipse',
                'RotatedEllipse2', 'DixonPrice', 'Powell', 'Wood', 'HelicalValley',
                'Beale', 'NesterovQuadratic']
    return problem_name in unimodal


@dataclass
class SuccessCriteria:
    """Defines what constitutes success for a benchmark."""
    tolerance: float  # Acceptable distance from global minimum
    max_evaluations: int  # Maximum evaluations allowed
    name: str

    def is_successful(self, func_distance: float, evaluations: int) -> bool:
        """Check if benchmark run meets success criteria."""
        return func_distance <= self.tolerance and evaluations <= self.max_evaluations


# Standard success criteria
SUCCESS_CRITERIA = [
    SuccessCriteria(tolerance=1e-3, max_evaluations=100, name="strict_100"),
    SuccessCriteria(tolerance=1e-2, max_evaluations=500, name="moderate_500"),
    SuccessCriteria(tolerance=1e-1, max_evaluations=1000, name="lenient_1000"),
    SuccessCriteria(tolerance=1.0, max_evaluations=5000, name="very_lenient_5000"),
]


def benchmark_result_to_dict(case: BenchmarkCase, criteria: SuccessCriteria,
                           quality: Dict[str, float], evaluations: int,
                           time_taken: float) -> Dict[str, Any]:
    """Convert benchmark result to dictionary for storage/analysis."""
    return {
        'problem': case.problem_name,
        'dimension': case.dimension,
        'shift': case.shift_vector.tolist(),
        'difficulty': case.difficulty,
        'separable': case.separable,
        'unimodal': case.unimodal,
        'criteria_name': criteria.name,
        'tolerance': criteria.tolerance,
        'max_evaluations': criteria.max_evaluations,
        'evaluations_used': evaluations,
        'time_taken': time_taken,
        'success': criteria.is_successful(quality['func_distance'], evaluations),
        **quality
    }


def calculate_solution_quality(found_x: np.ndarray, found_fx: float,
                             true_x: Optional[np.ndarray], true_fx: float) -> Dict[str, Any]:
    """Calculate solution quality metrics."""
    # Distance in parameter space
    if true_x is not None:
        param_distance = np.linalg.norm(found_x - true_x)
    else:
        param_distance = -1.0 # Unknown

    # Distance in function space
    func_distance = abs(found_fx - true_fx)

    # Relative error
    if abs(true_fx) > 1e-10:
        relative_error = abs((found_fx - true_fx) / true_fx)
    else:
        relative_error = func_distance

    return {
        'param_distance': param_distance,
        'func_distance': func_distance,
        'relative_error': relative_error,
        'found_fx': found_fx,
        'true_fx': true_fx
    }


def is_solution_found(quality_metrics: Dict[str, float],
                     tolerance: float = 1e-2) -> bool:
    """Check if solution is within acceptable tolerance."""
    return quality_metrics['func_distance'] <= tolerance
