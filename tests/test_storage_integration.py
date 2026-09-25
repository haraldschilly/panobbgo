# -*- coding: utf8 -*-
import os
import pytest
import numpy as np
from panobbgo.core import StrategyBase, Result, Point
from panobbgo.lib.classic import Rosenbrock
from panobbgo.heuristics import Random


@pytest.fixture
def storage_uri(tmp_path):
    # Per-test temp dir keeps parallel workers (pytest-xdist) from colliding
    # on a shared on-disk filename.
    uri = str(tmp_path / "test_integration.db")
    yield uri
    if os.path.exists(uri):
        os.unlink(uri)


class MockStrategy(StrategyBase):
    def execute(self):
        # We need to return points.
        # Just use Random heuristic
        h = self.heuristic("Random")
        return h.get_points(1)


def test_resume_capability(storage_uri):
    # 1. Run optimization with storage enabled
    problem = Rosenbrock(dims=2)

    # Run for 10 evals
    strategy1 = MockStrategy(problem, max_eval=10, testing_mode=True, storage_backend="sqlite", storage_uri=storage_uri)
    strategy1.add(Random)

    # Add an analyzer to ensure framework is happy
    # StrategyBase.start adds default analyzers, so we are good.

    strategy1.start()

    assert len(strategy1.results) == 10

    # 2. Verify DB has 10 results
    from panobbgo.storage import SQLiteStorage

    db = SQLiteStorage(storage_uri)
    assert db.count() == 10

    # 3. Create NEW strategy with same DB
    strategy2 = MockStrategy(
        problem,
        max_eval=20,  # Continue to 20
        testing_mode=True,
        storage_backend="sqlite",
        storage_uri=storage_uri,
    )
    strategy2.add(Random)

    # Mocking start behavior slightly since we can't easily interrupt execute loop in a clean way
    # without running it fully.
    # But StrategyBase.start() calls load_from_storage().

    strategy2.start()

    # Should have loaded 10, then ran 10 more to reach 20.
    assert len(strategy2.results) == 20

    # Verify the first 10 match somehow?
    # We can check if the loaded results are present
    # Exact count ensures no duplication happened during load
    assert db.count() == 20


def test_storage_refuses_another_problems_results(storage_uri):
    """A shared database used to resume foreign results (a dim mismatch crashed ``_flush_buffer``)."""
    from panobbgo.lib.classic import Quadruple
    from panobbgo.storage import StorageMismatchError

    s1 = MockStrategy(
        Rosenbrock(dims=2), max_eval=5, testing_mode=True, storage_backend="sqlite", storage_uri=storage_uri
    )
    s1.add(Random)
    s1.start()

    for other in (Rosenbrock(dims=3), Quadruple(dims=2)):
        with pytest.raises(StorageMismatchError, match="different problem"):
            MockStrategy(other, max_eval=5, testing_mode=True, storage_backend="sqlite", storage_uri=storage_uri)


def test_legacy_storage_without_fingerprint_is_refused(storage_uri):
    """Results written before fingerprints (and before the classic formula fixes) cannot be verified."""
    from panobbgo.storage import SQLiteStorage, StorageMismatchError, problem_fingerprint

    legacy = SQLiteStorage(storage_uri)  # no fingerprint: a pre-fingerprint database
    legacy.save([Result(Point(np.zeros(2), "R"), 1.0)])
    legacy.close()
    with pytest.raises(StorageMismatchError, match="without a problem fingerprint"):
        SQLiteStorage(storage_uri, fingerprint=problem_fingerprint(Rosenbrock(dims=2)))


def test_legacy_storage_can_be_adopted_explicitly(storage_uri):
    from panobbgo.lib.classic import Rastrigin
    from panobbgo.storage import SQLiteStorage, StorageMismatchError, problem_fingerprint

    legacy = SQLiteStorage(storage_uri)
    legacy.save([Result(Point(np.zeros(2), "R"), 1.0)])
    legacy.close()
    fp = problem_fingerprint(Rosenbrock(dims=2))
    SQLiteStorage(storage_uri, fingerprint=fp, adopt_legacy=True).close()
    assert SQLiteStorage(storage_uri, fingerprint=fp).count() == 1  # adopted: now it just matches
    with pytest.raises(StorageMismatchError):
        SQLiteStorage(storage_uri, fingerprint=problem_fingerprint(Rastrigin(dims=2)))
    SQLiteStorage(storage_uri).adopt(problem_fingerprint(Rastrigin(dims=2)))  # the method form
    assert SQLiteStorage(storage_uri, fingerprint=problem_fingerprint(Rastrigin(dims=2))).count() == 1

    # Through the strategy config.
    legacy2 = SQLiteStorage(storage_uri + "3")
    legacy2.save([Result(Point(np.zeros(2), "R"), 1.0)])
    legacy2.close()
    s2 = MockStrategy(
        Rosenbrock(dims=2),
        max_eval=3,
        testing_mode=True,
        storage_backend="sqlite",
        storage_uri=storage_uri + "3",
        storage_adopt_legacy=True,
    )
    assert s2.results.backend.count() == 1
    s2.results.close()


def test_fingerprint_distinguishes_wrapped_parametrised_and_noisy_problems():
    from panobbgo.lib.classic import Rastrigin
    from panobbgo.lib.noise import AdditiveGaussianNoise, GaussianNoise, NoisyProblem
    from panobbgo.lib.wrappers import NormalizedProblem
    from panobbgo.storage import problem_fingerprint as fp

    assert fp(NormalizedProblem(Rosenbrock(dims=2))) != fp(NormalizedProblem(Rastrigin(dims=2)))
    assert fp(Rosenbrock(dims=2)) != fp(Rosenbrock(dims=2, par1=50))
    assert fp(Rosenbrock(dims=2)) == fp(Rosenbrock(dims=2))
    base = Rosenbrock(dims=2)
    a = NoisyProblem(base, GaussianNoise(beta=0.1), seed=1)
    assert fp(a) != fp(NoisyProblem(base, GaussianNoise(beta=0.1), seed=2))
    assert fp(a) != fp(NoisyProblem(base, GaussianNoise(beta=1.0), seed=1))
    assert fp(a) != fp(NoisyProblem(base, AdditiveGaussianNoise(sigma=0.1), seed=1))
    assert fp(a) == fp(NoisyProblem(Rosenbrock(dims=2), GaussianNoise(beta=0.1), seed=1))


def test_fingerprint_distinguishes_rotations_and_formula_versions():
    from panobbgo.harness_randomized import TransformedProblem
    from panobbgo.lib.classic import Wood
    from panobbgo.storage import problem_fingerprint as fp

    q = np.array([[0.0, 1.0], [1.0, 0.0]])
    plain = TransformedProblem(Rosenbrock(dims=2), x_star=[0.5, 0.5])
    assert fp(plain) != fp(TransformedProblem(Rosenbrock(dims=2), x_star=[0.5, 0.5], Q=q))
    assert '"formula_version": 2' in fp(Wood())  # corrected formula: older databases do not match


def test_clear_keeps_the_fingerprint_of_an_open_store(storage_uri):
    from panobbgo.lib.classic import Rastrigin
    from panobbgo.storage import SQLiteStorage, StorageMismatchError, problem_fingerprint

    s = SQLiteStorage(storage_uri, fingerprint=problem_fingerprint(Rosenbrock(dims=2)))
    s.save([Result(Point(np.zeros(2), "R"), 1.0)])
    s.clear()
    with pytest.raises(StorageMismatchError):
        SQLiteStorage(storage_uri, fingerprint=problem_fingerprint(Rastrigin(dims=2)))
