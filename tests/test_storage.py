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


def test_sqlite_storage_none_conn(storage_uri):
    storage = SQLiteStorage(storage_uri)
    storage.close()  # Set _conn to None

    # Try all methods that require _conn
    storage._init_db()  # Should return silently

    p = Point(np.array([1.0]), "h")
    r = Result(p, 1.0)
    storage.save([r])  # Should return silently

    assert storage.load() == []
    assert storage.count() == 0
    storage.clear()  # Should return silently
    storage.close()  # Should return silently


def test_sqlite_storage_save_none_arrays(storage_uri):
    storage = SQLiteStorage(storage_uri)
    p = Point(None, "h")
    r = Result(p, None, cv_vec=None)
    storage.save([r])

    loaded = storage.load()
    assert len(loaded) == 1
    # Check that None values are safely round-tripped and not crashing
    assert loaded[0].x is not None
    assert loaded[0].fx is None
    assert loaded[0].cv_vec is None


def test_sqlite_storage_json_parsing_errors(storage_uri):
    storage = SQLiteStorage(storage_uri)
    p = Point(np.array([1.0]), "h")
    r = Result(p, 1.0)
    storage.save([r])

    # Manually corrupt the JSON in the database
    storage._conn.execute("UPDATE results SET x = 'invalid_json', cv_vec = 'invalid_json'")

    loaded = storage.load()
    assert len(loaded) == 1
    assert loaded[0].x.size == 0
    assert loaded[0].cv_vec is None


def test_sqlite_storage_save_empty(storage_uri):
    storage = SQLiteStorage(storage_uri)
    storage.save([])
    assert storage.count() == 0


def test_storage_backend_abstract_methods():
    from panobbgo.storage import StorageBackend

    class DummyStorage(StorageBackend):
        def save(self, results):
            super().save(results)

        def load(self):
            return super().load()

        def count(self):
            return super().count()

        def clear(self):
            super().clear()

    dummy = DummyStorage()
    dummy.save([])
    assert dummy.load() is None
    assert dummy.count() is None
    dummy.clear()
    dummy.close()
