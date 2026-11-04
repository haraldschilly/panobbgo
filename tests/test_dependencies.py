# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
"""Runtime dependencies that nothing else in the suite imports."""

import subprocess
import sys


def test_cocopp_imports():
    """``python -m cocopp`` (``benchmarks/coco_benchmark.py``) needs ``six``, which cocopp does not declare."""
    result = subprocess.run(
        [sys.executable, "-c", "import cocopp"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
