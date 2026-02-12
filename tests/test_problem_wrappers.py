# -*- coding: utf8 -*-
import pytest
import numpy as np
from panobbgo.lib import Problem, Point, Result
from panobbgo.lib.wrappers import (
    ProblemWrapper,
    NormalizedProblem,
    LogTransformProblem,
    NoisyProblem,
)


class QuadraticProblem(Problem):
    """f(x) = sum(x^2), box [-5, 5]^dim"""

    def __init__(self, dim=2):
        box = [(-5.0, 5.0)] * dim
        super().__init__(box=box)

    def eval(self, x):
        return float(np.sum(x**2))

    def eval_constraints(self, x):
        # constraint: x[0] >= 1 => violation = max(0, 1 - x[0])
        return np.array([max(0.0, 1.0 - x[0])])


class AsymmetricProblem(Problem):
    """f(x) = x[0]^2 + x[1]^2, box [0, 10] x [-100, 100]"""

    def __init__(self):
        super().__init__(box=[(0.0, 10.0), (-100.0, 100.0)])

    def eval(self, x):
        return float(np.sum(x**2))


# --- ProblemWrapper base ---


def test_wrapper_delegates():
    p = QuadraticProblem(dim=2)
    w = ProblemWrapper(p)
    assert w.dim == 2
    assert w.wrapped is p
    x = np.array([1.0, 2.0])
    assert w.eval(x) == p.eval(x)


def test_wrapper_box_matches():
    p = QuadraticProblem(dim=3)
    w = ProblemWrapper(p)
    np.testing.assert_array_equal(w.box[:, 0], p.box[:, 0])
    np.testing.assert_array_equal(w.box[:, 1], p.box[:, 1])


def test_wrapper_callable():
    p = QuadraticProblem(dim=2)
    w = ProblemWrapper(p)
    pt = Point(np.array([1.0, 2.0]), "test")
    result = w(pt)
    assert isinstance(result, Result)
    assert result.fx == pytest.approx(5.0)


# --- NormalizedProblem ---


def test_normalized_box_is_unit():
    p = AsymmetricProblem()
    n = NormalizedProblem(p)
    assert n.dim == 2
    np.testing.assert_array_equal(n.box[:, 0], [0.0, 0.0])
    np.testing.assert_array_equal(n.box[:, 1], [1.0, 1.0])


def test_normalized_midpoint():
    """Evaluating at [0.5, 0.5] in normalized space should equal eval at midpoint of original box."""
    p = AsymmetricProblem()
    n = NormalizedProblem(p)
    # midpoint of [0,10] is 5, midpoint of [-100,100] is 0
    x_norm = np.array([0.5, 0.5])
    assert n.eval(x_norm) == pytest.approx(p.eval(np.array([5.0, 0.0])))


def test_normalized_corners():
    p = AsymmetricProblem()
    n = NormalizedProblem(p)
    # [0, 0] in normalized = [0, -100] in original => 0^2 + (-100)^2 = 10000
    assert n.eval(np.array([0.0, 0.0])) == pytest.approx(10000.0)
    # [1, 1] in normalized = [10, 100] in original => 100 + 10000 = 10100
    assert n.eval(np.array([1.0, 1.0])) == pytest.approx(10100.0)


def test_normalized_constraints():
    p = QuadraticProblem(dim=2)
    n = NormalizedProblem(p)
    # x_norm=[0.5, 0.5] => x_orig=[0, 0] (midpoint of [-5,5]) => cv = max(0, 1-0) = 1.0
    cv = n.eval_constraints(np.array([0.5, 0.5]))
    assert cv is not None
    np.testing.assert_array_almost_equal(cv, [1.0])


def test_normalized_random_point_in_box():
    p = AsymmetricProblem()
    n = NormalizedProblem(p)
    for _ in range(20):
        x = n.random_point()
        assert np.all(x >= 0.0) and np.all(x <= 1.0)


def test_normalized_callable():
    p = AsymmetricProblem()
    n = NormalizedProblem(p)
    pt = Point(np.array([0.5, 0.5]), "test")
    result = n(pt)
    assert isinstance(result, Result)
    assert result.fx == pytest.approx(25.0)  # 5^2 + 0^2


# --- LogTransformProblem ---


def test_log_transform_basic():
    p = QuadraticProblem(dim=2)
    lt = LogTransformProblem(p)
    x = np.array([1.0, 2.0])
    expected = np.log1p(5.0)  # log(1 + 1 + 4)
    assert lt.eval(x) == pytest.approx(expected)


def test_log_transform_with_offset():
    p = QuadraticProblem(dim=2)
    lt = LogTransformProblem(p, offset=3.0)
    x = np.array([1.0, 2.0])
    expected = np.log1p(5.0 - 3.0)  # log(1 + 2)
    assert lt.eval(x) == pytest.approx(expected)


def test_log_transform_callable():
    p = QuadraticProblem(dim=2)
    lt = LogTransformProblem(p)
    pt = Point(np.array([1.0, 2.0]), "test")
    result = lt(pt)
    assert result.fx == pytest.approx(np.log1p(5.0))


def test_log_transform_preserves_constraints():
    p = QuadraticProblem(dim=2)
    lt = LogTransformProblem(p)
    pt = Point(np.array([0.0, 0.0]), "test")
    result = lt(pt)
    # constraints should NOT be transformed
    assert result.cv == pytest.approx(1.0)


# --- NoisyProblem ---


def test_noisy_additive():
    p = QuadraticProblem(dim=2)
    n = NoisyProblem(p, noise_std=0.1, seed=42)
    x = np.array([1.0, 2.0])
    true_val = 5.0
    # Evaluate many times and check that mean ~ true_val and spread ~ noise_std
    vals = [n.eval(x) for _ in range(1000)]
    assert abs(np.mean(vals) - true_val) < 0.05
    assert abs(np.std(vals) - 0.1) < 0.05


def test_noisy_multiplicative():
    p = QuadraticProblem(dim=2)
    n = NoisyProblem(p, noise_std=0.1, noise_type="multiplicative", seed=42)
    x = np.array([1.0, 2.0])
    true_val = 5.0
    vals = [n.eval(x) for _ in range(1000)]
    assert abs(np.mean(vals) - true_val) < 0.1
    # std should be roughly true_val * noise_std = 0.5
    assert abs(np.std(vals) - 0.5) < 0.1


def test_noisy_seed_reproducibility():
    p = QuadraticProblem(dim=2)
    x = np.array([1.0, 2.0])
    n1 = NoisyProblem(p, noise_std=1.0, seed=123)
    n2 = NoisyProblem(p, noise_std=1.0, seed=123)
    assert n1.eval(x) == n2.eval(x)


# --- Composition ---


def test_composition_normalized_noisy():
    p = AsymmetricProblem()
    wrapped = NormalizedProblem(NoisyProblem(p, noise_std=0.01, seed=0))
    assert wrapped.dim == 2
    np.testing.assert_array_equal(wrapped.box[:, 0], [0.0, 0.0])
    np.testing.assert_array_equal(wrapped.box[:, 1], [1.0, 1.0])
    # Should evaluate without error
    result = wrapped.eval(np.array([0.5, 0.5]))
    assert isinstance(result, float)


def test_composition_log_normalized():
    p = AsymmetricProblem()
    wrapped = LogTransformProblem(NormalizedProblem(p))
    assert wrapped.dim == 2
    x_norm = np.array([0.5, 0.5])
    expected = np.log1p(25.0)  # log(1 + 5^2 + 0^2)
    assert wrapped.eval(x_norm) == pytest.approx(expected)
