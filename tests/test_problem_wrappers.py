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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_noisy_additive():
    p = QuadraticProblem(dim=2)
    n = NoisyProblem(p, noise_std=0.1, seed=42)
    x = np.array([1.0, 2.0])
    true_val = 5.0
    # Evaluate many times and check that mean ~ true_val and spread ~ noise_std
    vals = [n.eval(x) for _ in range(1000)]
    assert abs(np.mean(vals) - true_val) < 0.05
    assert abs(np.std(vals) - 0.1) < 0.05


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_noisy_multiplicative():
    p = QuadraticProblem(dim=2)
    n = NoisyProblem(p, noise_std=0.1, noise_type="multiplicative", seed=42)
    x = np.array([1.0, 2.0])
    true_val = 5.0
    vals = [n.eval(x) for _ in range(1000)]
    assert abs(np.mean(vals) - true_val) < 0.1
    # std should be roughly true_val * noise_std = 0.5
    assert abs(np.std(vals) - 0.5) < 0.1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_noisy_seed_reproducibility():
    p = QuadraticProblem(dim=2)
    x = np.array([1.0, 2.0])
    n1 = NoisyProblem(p, noise_std=1.0, seed=123)
    n2 = NoisyProblem(p, noise_std=1.0, seed=123)
    assert n1.eval(x) == n2.eval(x)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_legacy_noisy_is_thread_order_independent():
    """One generator shared by all evaluator threads made the draws depend on scheduling."""
    from concurrent.futures import ThreadPoolExecutor

    xs = [np.array([i * 0.1, -i * 0.2]) for i in range(40)]

    def run(order, workers):
        n = NoisyProblem(QuadraticProblem(dim=2), noise_std=1.0, seed=7)
        with ThreadPoolExecutor(workers) as pool:
            vals = dict(zip(order, pool.map(lambda i: n.eval(xs[i]), order)))
        return [vals[i] for i in range(len(xs))]

    fwd = run(list(range(40)), 1)
    assert run(list(reversed(range(40))), 4) == fwd


def test_legacy_noisy_is_deprecated_and_lib_exports_the_deterministic_one():
    import panobbgo.lib
    import panobbgo.lib.noise

    assert panobbgo.lib.NoisyProblem is panobbgo.lib.noise.NoisyProblem
    with pytest.warns(DeprecationWarning, match="panobbgo.lib.noise.NoisyProblem"):
        n = NoisyProblem(QuadraticProblem(dim=2), noise_std=0.1, seed=1)
    assert isinstance(n, panobbgo.lib.noise.NoisyProblem)
    # Raw-value noise: a negative objective is not clamped to the f_opt floor.

    class Neg(QuadraticProblem):
        def eval(self, x):
            return -100.0

    with pytest.warns(DeprecationWarning):
        m = NoisyProblem(Neg(), noise_std=0.1, seed=1)
    assert abs(m.eval(np.zeros(2)) + 100.0) < 1.0


# --- Composition ---


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
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


# --- review of #326: dx, f_opt, pickling, per-point noise ---


def test_wrappers_evaluate_a_dx_shifted_problem_at_its_shifted_optimum():
    """Wrappers copied the shifted box but evaluated the inner problem without undoing dx."""
    from panobbgo.harness_randomized import TransformedProblem
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.lib.noise import NoisyProblem as DetNoisy, NoNoise

    inner = Rosenbrock(2, dx=[0.5, 0.5])
    x_star = np.array([1.5, 1.5])  # (1, 1) + dx
    assert inner(Point(x_star, "t")).fx == pytest.approx(0.0)
    assert LogTransformProblem(inner).eval(x_star) == pytest.approx(0.0)
    assert ProblemWrapper(inner)(Point(x_star, "t")).fx == pytest.approx(0.0)
    norm = NormalizedProblem(inner)
    assert norm.eval((x_star - inner.box[:, 0]) / inner.ranges) == pytest.approx(0.0)
    assert DetNoisy(inner, NoNoise(), seed=0).eval(x_star) == pytest.approx(0.0)
    assert TransformedProblem(inner, x_star=[0.0, 0.0], y_base_star=x_star).eval(np.zeros(2)) == pytest.approx(0.0)


def test_noisy_problem_f_opt_fallbacks_and_no_clamp_when_unknown():
    from panobbgo.lib.classic import StyblinskiTang
    from panobbgo.lib.noise import AdditiveGaussianNoise, NoisyProblem as DetNoisy

    st = StyblinskiTang(dims=2)
    n = DetNoisy(st, AdditiveGaussianNoise(sigma=1e-3), seed=1)
    assert n.f_opt == n.optimum_y == pytest.approx(st.f_opt)
    assert n.eval(np.asarray(st.x_opt)) == pytest.approx(st.f_opt, abs=0.01)  # was clamped to ~0

    class Unknown(QuadraticProblem):
        def eval(self, x):
            return -100.0 + float(np.sum(x**2))

    u = DetNoisy(Unknown(), AdditiveGaussianNoise(sigma=1e-3), seed=1)
    assert u.f_opt is None and u.optimum_y is None
    assert u.eval(np.array([1.0, 0.0])) == pytest.approx(-99.0, abs=0.01)


def test_noisy_problem_pickles_and_keeps_its_draw_counts():
    import pickle

    from panobbgo.lib.noise import GaussianNoise, NoisyProblem as DetNoisy

    n = DetNoisy(QuadraticProblem(dim=2), GaussianNoise(beta=0.5), seed=4, resample=True, f_opt=0.0)
    x = np.array([1.0, 2.0])
    n.eval(x)
    m = pickle.loads(pickle.dumps(n))
    assert m.eval(x) == n.eval(x)  # both at draw k=1


def _thread_order_values(make, xs):
    from concurrent.futures import ThreadPoolExecutor

    def run(order, workers):
        p = make()
        with ThreadPoolExecutor(workers) as pool:
            vals = dict(zip(order, pool.map(lambda i: p.eval(xs[i]), order)))
        return [vals[i] for i in range(len(xs))]

    return run(list(range(len(xs))), 1), run(list(reversed(range(len(xs)))), 4)


def test_stochastic_classics_and_transformed_noise_are_thread_order_independent():
    from panobbgo.harness_randomized import TransformedProblem
    from panobbgo.lib.classic import Rosenbrock, RosenbrockStochastic

    xs = [np.array([0.1 * i, -0.2 * i, 0.05 * i]) for i in range(30)]
    a, b = _thread_order_values(lambda: RosenbrockStochastic(dims=3, seed=5), xs)
    assert a == b
    a, b = _thread_order_values(
        lambda: TransformedProblem(Rosenbrock(dims=3), x_star=np.zeros(3), noise_sigma=0.5, noise_seed=9), xs
    )
    assert a == b


def test_unseeded_stochastic_classics_follow_the_global_numpy_seed():
    from panobbgo.lib.classic import NesterovQuadratic, RosenbrockStochastic

    x = np.array([0.3, -0.2, 0.9])
    np.random.seed(11)
    a, na = RosenbrockStochastic(dims=3), NesterovQuadratic(dim=3)
    np.random.seed(11)
    b, nb = RosenbrockStochastic(dims=3), NesterovQuadratic(dim=3)
    assert a.eval(x) == b.eval(x)
    assert np.array_equal(na.A, nb.A)


def test_bbob_noise_models_need_a_known_f_opt():
    """On a raw value UniformNoise went complex for f < 0, GaussianNoise flipped direction."""
    from panobbgo.lib.noise import (
        AdditiveGaussianNoise,
        CauchyNoise,
        GaussianNoise,
        MultiplicativeGaussianNoise,
        NoNoise,
        NoisyProblem as DetNoisy,
        UniformNoise,
    )

    for model in (GaussianNoise(), UniformNoise(), CauchyNoise()):
        with pytest.raises(ValueError, match="f_opt"):
            DetNoisy(QuadraticProblem(dim=2), model, seed=0)
        DetNoisy(QuadraticProblem(dim=2), model, seed=0, f_opt=0.0)  # explicit: fine
    for model in (NoNoise(), AdditiveGaussianNoise(), MultiplicativeGaussianNoise()):
        DetNoisy(QuadraticProblem(dim=2), model, seed=0)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_legacy_noisy_fingerprint_differs_from_the_deterministic_one():
    """Different values need different storage identities.

    The legacy wrapper corrupts the raw value, the new one the precision
    above ``f_opt``; with a multiplicative model and ``f_opt != 0`` they
    disagree at the same (model, seed, point), yet both fingerprinted as
    ``NoisyProblem(...)``, so one's database resumed under the other.
    """
    from panobbgo.lib.noise import MultiplicativeGaussianNoise, NoisyProblem as DetNoisy
    from panobbgo.storage import problem_fingerprint

    class Shifted(QuadraticProblem):
        f_opt = 5.0

        def eval(self, x):
            return 5.0 + float(np.sum(np.asarray(x) ** 2))

    legacy = NoisyProblem(Shifted(dim=2), noise_std=0.3, noise_type="multiplicative", seed=4)
    new = DetNoisy(Shifted(dim=2), MultiplicativeGaussianNoise(sigma=0.3), seed=4, resample=True)
    x = np.array([1.0, 1.0])
    assert legacy.eval(x) != new.eval(x)
    assert problem_fingerprint(legacy) != problem_fingerprint(new)
