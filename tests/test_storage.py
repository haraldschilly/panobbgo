# -*- coding: utf8 -*-
import os
import pytest
import numpy as np
from panobbgo.storage import SQLiteStorage
from panobbgo.lib import Result, Point

@pytest.fixture
def storage_uri():
    uri = "test_storage.db"
    yield uri
    if os.path.exists(uri):
        os.unlink(uri)

def test_sqlite_storage_init(storage_uri):
    storage = SQLiteStorage(storage_uri)
    assert os.path.exists(storage_uri)
    assert storage.count() == 0

def test_sqlite_storage_save_load(storage_uri):
    storage = SQLiteStorage(storage_uri)

    # Create some dummy results
    p1 = Point(np.array([1.0, 2.0]), "heuristic1")
    r1 = Result(p1, 0.5, cv_vec=np.array([0.1]))

    p2 = Point(np.array([3.0, 4.0]), "heuristic2")
    r2 = Result(p2, 1.5, cv_vec=None, error=0.01)

    storage.save([r1, r2])

    assert storage.count() == 2

    loaded = storage.load()
    assert len(loaded) == 2

    # Verify contents
    l1, l2 = loaded[0], loaded[1]

    assert np.allclose(l1.x, p1.x)
    assert l1.fx == r1.fx
    assert l1.who == r1.who
    assert np.allclose(l1.cv_vec, r1.cv_vec)

    assert np.allclose(l2.x, p2.x)
    assert l2.fx == r2.fx
    assert l2.who == r2.who
    assert l2.cv_vec is None
    assert l2.error == r2.error

def test_sqlite_storage_clear(storage_uri):
    storage = SQLiteStorage(storage_uri)
    p = Point(np.array([1.0]), "h")
    r = Result(p, 1.0)
    storage.save([r])
    assert storage.count() == 1

    storage.clear()
    assert storage.count() == 0
