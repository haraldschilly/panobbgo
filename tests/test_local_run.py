# -*- coding: utf8 -*-
"""Resource hygiene helpers for long local runs."""

import argparse
import subprocess
import sys
from unittest import mock

import pytest

from panobbgo import local_run


def test_available_memory_is_positive_on_linux():
    avail = local_run.available_memory_gb()
    if sys.platform.startswith("linux"):
        assert avail is not None and avail > 0
    else:
        assert avail is None or avail > 0


def test_check_free_memory_aborts_below_floor():
    with mock.patch.object(local_run, "available_memory_gb", return_value=1.0):
        with pytest.raises(SystemExit):
            local_run.check_free_memory(2.0)
        local_run.check_free_memory(0.5)  # no exit


def test_be_nice_raises_priority_in_a_subprocess():
    # os.nice cannot be undone by an unprivileged process, so probe in a child.
    code = "import os; from panobbgo.local_run import be_nice; be_nice(15); print(os.nice(0))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert int(out.stdout.strip()) >= 15


def test_arguments_and_apply():
    p = argparse.ArgumentParser()
    local_run.add_arguments(p)
    args = p.parse_args(["--nice", "7", "--min-free-mem-gb", "0"])
    with mock.patch.object(local_run, "be_nice") as nice, mock.patch.object(local_run, "check_free_memory") as mem:
        local_run.apply(args)
    nice.assert_called_once_with(7)
    mem.assert_not_called()

    args = p.parse_args(["--no-nice", "--min-free-mem-gb", "1.5"])
    with mock.patch.object(local_run, "be_nice") as nice, mock.patch.object(local_run, "check_free_memory") as mem:
        local_run.apply(args)
    nice.assert_not_called()
    mem.assert_called_once_with(1.5)
