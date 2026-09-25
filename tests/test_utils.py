from __future__ import unicode_literals
# -*- coding: utf-8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
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

import unittest
import numpy as np
from panobbgo.utils import is_right, is_left


class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_is_left(self):
        # simplified
        p0 = np.array([5, 5])
        p1 = np.array([7, 21])
        pt = np.array([5, 5.01])
        self.assertTrue(is_left(p0, p1, pt))
        pt = p1
        self.assertFalse(is_left(p0, p1, pt))

    def test_is_right_vertical(self):
        p0 = np.array([1, 1])
        p1 = np.array([1, 3])
        testpoints = [np.array([1.1, 2.2]), np.array([1.1, -2.2]), np.array([5, 2])]

        for tp in testpoints:
            self.assertTrue(is_right(p0, p1, tp), "%s" % tp)
            tp = p1
            self.assertFalse(is_left(p0, p1, tp))

    def test_is_right_diagonal(self):
        p0 = np.array([0, 2])
        p1 = np.array([1, 3])
        testpoints = [np.array([0.5, 1]), np.array([1.2, 3.1])]

        for tp in testpoints:
            self.assertTrue(is_right(p0, p1, tp), "%s" % tp)

    def test_is_right_horizontal(self):
        p0 = np.array([2, 2])
        p1 = np.array([3, 2])
        testpoints = [np.array([2, 2.2]), np.array([0, 5])]

        for tp in testpoints:
            self.assertFalse(is_right(p0, p1, tp), "%s" % tp)


def _fresh_info(monkeypatch, run):
    import subprocess

    from panobbgo import utils

    utils._info.cache_clear()
    monkeypatch.setattr(subprocess, "run", run)
    try:
        return utils.info()
    finally:
        utils._info.cache_clear()


def test_info_without_git_binary(monkeypatch):
    def run(*a, **kw):
        raise FileNotFoundError("git")

    assert _fresh_info(monkeypatch, run)["git HEAD"] == "unknown"


def test_info_outside_a_checkout(monkeypatch):
    """Empty ``git rev-parse`` output used to raise IndexError."""
    import subprocess

    def run(args, **kw):
        return subprocess.CompletedProcess(args, 128, stdout="", stderr="fatal: not a git repository")

    assert _fresh_info(monkeypatch, run)["git HEAD"] == "unknown"


def test_info_reports_panobbgos_repo_not_the_cwd(monkeypatch, tmp_path):
    import os
    import subprocess

    import panobbgo

    seen = {}
    real = subprocess.run

    def run(args, **kw):
        seen["cwd"] = kw.get("cwd")
        return real(args, **kw)

    monkeypatch.chdir(tmp_path)
    head = _fresh_info(monkeypatch, run)["git HEAD"]
    assert seen["cwd"] == os.path.dirname(os.path.abspath(panobbgo.utils.__file__))
    assert isinstance(head, str)


def test_info_ignores_an_enclosing_foreign_repository(monkeypatch, tmp_path):
    """Installed in a venv inside another repo: rev-parse answers for that repo -> 'unknown'."""
    import subprocess

    def run(args, **kw):
        return subprocess.CompletedProcess(args, 0, stdout="%s\n%s\n" % (tmp_path, "f" * 40), stderr="")

    assert _fresh_info(monkeypatch, run)["git HEAD"] == "unknown"


def test_info_reports_the_head_of_panobbgos_own_checkout(monkeypatch):
    import os
    import subprocess

    import panobbgo

    top = os.path.dirname(os.path.dirname(os.path.abspath(panobbgo.utils.__file__)))

    def run(args, **kw):
        return subprocess.CompletedProcess(args, 0, stdout="%s\n%s\n" % (top, "a" * 40), stderr="")

    assert _fresh_info(monkeypatch, run)["git HEAD"] == "a" * 40


if __name__ == "__main__":
    unittest.main()
