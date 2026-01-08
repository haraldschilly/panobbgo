"""
Benchmark Problems for Panobbgo Optimization Framework

This module defines a comprehensive battery of benchmark problems for evaluating
optimization strategies across various dimensions, shifted optima, and difficulty levels.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from panobbgo.lib.classic import (
    Rosenbrock, Rastrigin, Branin, GoldsteinPrice, Himmelblau, DeJong
)


@dataclass
class BenchmarkCase:
    """Represents a specific benchmark case with problem, dimension, and shift."""
    problem_name: str
    dimension: int
    shift_vector: np.ndarray  # Shift the global optimum
    global_optimum: np.ndarray
    global_minimum: float
    search_bounds: Tuple[float, float]
    difficulty: str
    separable: bool
    unimodal: bool

    def create_problem(self):
        """Create the actual problem instance."""
        problem_class = PROBLEM_CLASSES[self.problem_name]
        problem = problem_class(self.dimension)

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
    'DeJong': DeJong,  # Sphere-like (sum of squares)
    'Rosenbrock': Rosenbrock,
    'Rastrigin': Rastrigin,
    'Branin': Branin,
    'GoldsteinPrice': GoldsteinPrice,
    'Himmelblau': Himmelblau,
}


def generate_benchmark_battery() -> List[BenchmarkCase]:
    """Generate a comprehensive battery of benchmark cases."""
    cases = []

    # Standard shifts for testing
    shifts_2d = [
        np.array([0.0, 0.0]),      # Origin
        np.array([1.0, -0.5]),     # Shifted
        np.array([-2.0, 3.0]),     # Far shifted
    ]

    shifts_hd = [
        np.zeros(5),               # Origin
        np.array([1.0, -0.5, 2.0, -1.5, 0.8]),  # Shifted 5D
    ]

    # 2D Problems
    for problem_name in ['DeJong', 'Rosenbrock', 'Rastrigin', 'Branin', 'GoldsteinPrice', 'Himmelblau']:
        for shift in shifts_2d:
            # Set appropriate global minimum for each problem
            if problem_name == 'Branin':
                global_min = 0.397887
                global_opt = np.array([-np.pi, 12.275]) + shift
            elif problem_name == 'GoldsteinPrice':
                global_min = 3.0
                global_opt = np.array([0.0, -1.0]) + shift
            elif problem_name == 'Himmelblau':
                global_min = 0.0
                global_opt = np.array([3.0, 2.0]) + shift  # One of the optima
            else:
                global_min = 0.0
                global_opt = shift.copy()

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

    # Higher dimensional problems
    for problem_name in ['DeJong', 'Rosenbrock', 'Rastrigin']:
        for dim in [5, 10]:
            for shift in [np.zeros(dim)]:  # Only origin for higher dims
                case = BenchmarkCase(
                    problem_name=problem_name,
                    dimension=dim,
                    shift_vector=shift,
                    global_optimum=shift.copy(),
                    global_minimum=0.0,
                    search_bounds=(-5.0, 5.0),
                    difficulty=_get_difficulty(problem_name),
                    separable=_is_separable(problem_name),
                    unimodal=_is_unimodal(problem_name)
                )
                cases.append(case)

    return cases


def _get_difficulty(problem_name: str) -> str:
    """Get difficulty level for a problem."""
    easy = ['DeJong']
    medium = ['Rosenbrock', 'Branin', 'Himmelblau']
    hard = ['Rastrigin', 'GoldsteinPrice']
    if problem_name in easy:
        return 'easy'
    elif problem_name in medium:
        return 'medium'
    else:
        return 'hard'


def _is_separable(problem_name: str) -> bool:
    """Check if problem is separable."""
    separable = ['DeJong']  # Sum of squares is separable
    return problem_name in separable


def _is_unimodal(problem_name: str) -> bool:
    """Check if problem is unimodal."""
    unimodal = ['DeJong', 'Rosenbrock']
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


# Success criteria for different benchmark scenarios
SUCCESS_CRITERIA = [
    SuccessCriteria(tolerance=1e-3, max_evaluations=100, name="strict_100"),
    SuccessCriteria(tolerance=1e-2, max_evaluations=500, name="moderate_500"),
    SuccessCriteria(tolerance=1e-1, max_evaluations=1000, name="lenient_1000"),
    SuccessCriteria(tolerance=1.0, max_evaluations=5000, name="very_lenient_5000"),
]


def calculate_solution_quality(found_x: np.ndarray, found_fx: float,
                             true_x: np.ndarray, true_fx: float) -> Dict[str, Any]:
    """Calculate solution quality metrics."""
    # Distance in parameter space
    param_distance = np.linalg.norm(found_x - true_x)

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