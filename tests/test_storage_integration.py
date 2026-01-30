# -*- coding: utf8 -*-
import os
import pytest
import numpy as np
from panobbgo.core import StrategyBase, Result, Point
from panobbgo.lib.classic import Rosenbrock
from panobbgo.heuristics import Random

@pytest.fixture
def storage_uri():
    uri = "test_integration.db"
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
    strategy1 = MockStrategy(
        problem,
        max_eval=10,
        testing_mode=True,
        storage_backend="sqlite",
        storage_uri=storage_uri
    )
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
        max_eval=20, # Continue to 20
        testing_mode=True,
        storage_backend="sqlite",
        storage_uri=storage_uri
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
