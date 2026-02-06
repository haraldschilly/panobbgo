# -*- coding: utf8 -*-
# Copyright 2025 Panobbgo Contributors
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
import json
import sqlite3
import numpy as np
import time
from typing import List
from panobbgo.lib import Result, Point


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


class SQLiteStorage(StorageBackend):
    """
    SQLite-based storage backend.
    """

    def __init__(self, uri: str = "panobbgo.db"):
        self.uri = uri
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.uri) as conn:
            conn.execute(
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
            conn.commit()

    def save(self, results: List[Result]):
        if not results:
            return

        data = []
        for r in results:
            x_json = json.dumps(r.x.tolist()) if r.x is not None else "[]"
            cv_vec_json = (
                json.dumps(r.cv_vec.tolist()) if r.cv_vec is not None else "[]"
            )
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

        with sqlite3.connect(self.uri) as conn:
            conn.executemany(
                """
                INSERT INTO results (x, fx, cv_vec, who, error, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                data,
            )
            conn.commit()

    def load(self) -> List[Result]:
        results = []
        with sqlite3.connect(self.uri) as conn:
            cursor = conn.execute(
                "SELECT x, fx, cv_vec, who, error, timestamp FROM results ORDER BY id ASC"
            )
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
        return results

    def count(self) -> int:
        with sqlite3.connect(self.uri) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM results")
            return cursor.fetchone()[0]

    def clear(self):
        with sqlite3.connect(self.uri) as conn:
            conn.execute("DELETE FROM results")
            conn.commit()
