# -*- coding: utf8 -*-
# Copyright 2025-2026 Panobbgo Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Storage Backend
===============

This module provides storage backends for persisting optimization results.
This enables features like pausing/resuming optimization and post-hoc analysis.
"""

import abc
import hashlib
import json
import sqlite3
import threading
import numpy as np
import time
from typing import Any, List, Optional
from panobbgo.lib import Result, Point


class StorageMismatchError(ValueError):
    """The storage holds results of a different problem than the one being solved."""


#: Bumped when the default fingerprint's content changes, so databases keyed
#: by an older scheme are not mistaken for matches.
FINGERPRINT_VERSION = 2


def _param_value(v: Any) -> Any:
    """A JSON-able, run-independent image of one public attribute, or ``None`` to skip it."""
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
        return "nd:" + hashlib.blake2b(np.ascontiguousarray(v, dtype=np.float64).tobytes(), digest_size=12).hexdigest()
    if isinstance(v, (list, tuple)) and all(isinstance(e, (bool, int, float, str)) for e in v):
        return list(v)
    return None  # generators, callables, handles: not part of the identity


def problem_fingerprint(problem: Any) -> str:
    """Identify a problem for storage resumption.

    A problem (or wrapper) may define ``fingerprint() -> str``; the
    wrappers, :class:`~panobbgo.lib.noise.NoisyProblem`, the randomized
    harness's ``TransformedProblem`` and the IOH / COCO problems do, and
    wrappers include their inner problem's fingerprint.  Otherwise it is the
    class, its ``formula_version`` (bumped when a classic function's formula
    was corrected), the dimension, the box and the public scalar / array
    parameters (``par1``, ``optimum``, ...).  Two problems with equal
    fingerprints are treated as the same problem, so their stored results
    are interchangeable.
    """
    hook = getattr(type(problem), "fingerprint", None)
    if callable(hook):
        return str(problem.fingerprint())
    cls = type(problem)
    box = np.asarray(problem.box.box if hasattr(problem.box, "box") else problem.box, dtype=float)
    params = {}
    for k, v in sorted(vars(problem).items()):
        if k.startswith("_") or k == "dx":  # dx is already in the box
            continue
        val = _param_value(v)
        if val is not None:
            params[k] = val
    return json.dumps(
        {
            "v": FINGERPRINT_VERSION,
            "class": "%s.%s" % (cls.__module__, cls.__qualname__),
            "formula_version": getattr(cls, "formula_version", 1),
            "dim": int(problem.dim),
            "box": [[float(lo), float(hi)] for lo, hi in box],
            "params": params,
        },
        sort_keys=True,
    )


class StorageBackend(abc.ABC):
    """
    Abstract base class for storage backends.
    """

    @abc.abstractmethod
    def save(self, results: List[Result]):
        """
        Save a list of results to the storage.
        """
        pass

    @abc.abstractmethod
    def load(self) -> List[Result]:
        """
        Load all results from the storage.
        """
        pass

    @abc.abstractmethod
    def count(self) -> int:
        """
        Return the number of results in storage.
        """
        pass

    @abc.abstractmethod
    def clear(self):
        """
        Clear all results from storage.
        """
        pass

    def close(self):
        """
        Close the storage backend connection.
        """
        pass


class SQLiteStorage(StorageBackend):
    """
    SQLite-based storage backend.

    Args:
        uri: SQLite database path.
        fingerprint: identity of the problem the results belong to (see
            :func:`problem_fingerprint`).  It is stored in a ``meta`` table
            on first use; opening the database later with a different
            fingerprint -- or one that holds results but no fingerprint --
            raises :class:`StorageMismatchError` instead of resuming another
            problem's results.  ``None`` skips the check.
    """

    def __init__(self, uri: str = "panobbgo.db", fingerprint: Optional[str] = None):
        self.uri = uri
        self._lock = threading.RLock()
        # Open connection once and keep it open.
        # check_same_thread=False allows using the connection from multiple threads,
        # provided we serialize access (which we do via self._lock).
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(self.uri, check_same_thread=False)
        self._init_db()
        if fingerprint is not None:
            try:
                self._check_fingerprint(fingerprint)
            except Exception:
                self.close()
                raise

    def _check_fingerprint(self, fingerprint: str) -> None:
        with self._lock:
            assert self._conn is not None
            with self._conn:
                row = self._conn.execute("SELECT value FROM meta WHERE key = 'problem'").fetchone()
                if row is None:
                    has_rows = self._conn.execute("SELECT 1 FROM results LIMIT 1").fetchone() is not None
                    if has_rows:
                        # Written before fingerprints existed -- and before the
                        # classic functions' formulas were corrected: nothing
                        # can show these results belong to this problem.
                        raise StorageMismatchError(
                            "Storage %r holds results without a problem fingerprint (written by an older "
                            "panobbgo); they cannot be verified to belong to this problem. Use another "
                            "storage_uri, or clear the database." % (self.uri,)
                        )
                    # Two runs opening a new database at once: the first
                    # insert wins, both then compare against what is stored.
                    self._conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('problem', ?)", (fingerprint,))
                    row = self._conn.execute("SELECT value FROM meta WHERE key = 'problem'").fetchone()
            if row[0] != fingerprint:
                raise StorageMismatchError(
                    "Storage %r holds results of a different problem:\n  stored:  %s\n  current: %s\n"
                    "Use another storage_uri, or clear the database." % (self.uri, row[0], fingerprint)
                )

    def _init_db(self):
        with self._lock:
            if self._conn is None:
                return
            with self._conn:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        x TEXT,
                        fx REAL,
                        cv_vec TEXT,
                        who TEXT,
                        error REAL,
                        timestamp REAL
                    )
                    """
                )
                self._conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
                # No explicit commit needed, context manager handles it

    def save(self, results: List[Result]):
        if not results:
            return

        data = []
        for r in results:
            x_json = json.dumps(r.x.tolist()) if r.x is not None else "[]"
            cv_vec_json = json.dumps(r.cv_vec.tolist()) if r.cv_vec is not None else "[]"
            # Use current time if timestamp not explicitly available, though Result has _time but it's private-ish?
            # Result has _time attribute.
            timestamp = getattr(r, "_time", time.time())

            data.append(
                (
                    x_json,
                    r.fx if r.fx is not None else None,
                    cv_vec_json,
                    r.who,
                    r.error,
                    timestamp,
                )
            )

        with self._lock:
            if self._conn is None:
                return
            with self._conn:
                self._conn.executemany(
                    """
                    INSERT INTO results (x, fx, cv_vec, who, error, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    data,
                )

    def load(self) -> List[Result]:
        results = []
        with self._lock:
            if self._conn is None:
                return []
            # We don't need transaction for read, but it's fine.
            # Using cursor directly.
            cursor = self._conn.execute("SELECT x, fx, cv_vec, who, error, timestamp FROM results ORDER BY id ASC")
            try:
                for row in cursor:
                    x_json, fx, cv_vec_json, who, error, timestamp = row

                    try:
                        x = np.array(json.loads(x_json), dtype=np.float64)
                    except (ValueError, TypeError):
                        x = np.array([])

                    try:
                        cv_vec_list = json.loads(cv_vec_json)
                        cv_vec = np.array(cv_vec_list, dtype=np.float64) if cv_vec_list else None
                    except (ValueError, TypeError):
                        cv_vec = None

                    point = Point(x, who)
                    result = Result(point, fx, cv_vec=cv_vec, error=error)
                    # Ideally restore timestamp too, but Result doesn't expose it in init.
                    # We can manually set it if needed, but it's internal.
                    if hasattr(result, "_time"):
                        result._time = timestamp

                    results.append(result)
            finally:
                cursor.close()
        return results

    def count(self) -> int:
        with self._lock:
            if self._conn is None:
                return 0
            cursor = self._conn.execute("SELECT COUNT(*) FROM results")
            try:
                return cursor.fetchone()[0]
            finally:
                cursor.close()

    def clear(self):
        with self._lock:
            if self._conn is None:
                return
            with self._conn:
                # The fingerprint stays: this store is still open for the same problem.
                self._conn.execute("DELETE FROM results")

    def close(self):
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
