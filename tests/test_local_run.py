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


def test_task_pool_returns_results_in_task_order():
    tasks = [{"n": n} for n in (5, 1, 3, 0, 2)]
    for jobs in (1, 2):
        with local_run.TaskPool(jobs, niceness=None, min_free_gb=0) as pool:
            seen = []
            out = pool.map(_slow_square, tasks, on_done=lambda i, r: seen.append(i))
        assert out == [25, 1, 9, 0, 4]
        assert sorted(seen) == [0, 1, 2, 3, 4]


def test_task_pool_workers_are_niced():
    with local_run.TaskPool(2, niceness=17, min_free_gb=0) as pool:
        assert all(n >= 17 for n in pool.map(_niceness, [{}, {}]))


_GUARDED = """
from panobbgo.local_run import TaskPool


def square(x):
    return x * x


def main():
    open({marker!r}, "a").write("x")
    with TaskPool(2, niceness=None, min_free_gb=0) as pool:
        print(pool.map(square, [{{"x": 1}}, {{"x": 2}}, {{"x": 3}}]))


if __name__ == "__main__":
    main()
"""


@pytest.mark.parametrize("how", ["path", "module"])
def test_task_pool_runs_a_guarded_script_once(tmp_path, how):
    """The screens' shape: work under a main guard, the task defined in the
    script itself.  Run as a path and with ``python -m``; the workers import
    the module (as ``__mp_main__``) but must not run it."""
    marker = tmp_path / "ran"
    pkg = tmp_path / "screens"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "screen.py").write_text(_GUARDED.format(marker=str(marker)))
    cmd = [sys.executable, str(pkg / "screen.py")] if how == "path" else [sys.executable, "-m", "screens.screen"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[1, 4, 9]"
    assert marker.read_text() == "x"


def test_task_pool_propagates_a_task_error_and_stays_usable():
    with local_run.TaskPool(2, niceness=None, min_free_gb=0) as pool:
        with pytest.raises(ValueError, match="boom"):
            pool.map(_fail_on, [{"n": n} for n in range(6)])
        assert pool.map(_slow_square, [{"n": 2}]) == [4]


def test_a_dead_worker_does_not_poison_the_shared_pool():
    from concurrent.futures.process import BrokenProcessPool

    pool = local_run.shared_pool(2)
    try:
        with pytest.raises(BrokenProcessPool):
            pool.map(_die, [{}, {}])
        assert pool.broken
        fresh = local_run.shared_pool(2)
        assert fresh is not pool
        assert fresh.map(_slow_square, [{"n": 3}]) == [9]
    finally:
        local_run._close_shared()


def test_screen_jobs_parses_and_defaults():
    with mock.patch.object(local_run, "check_free_memory"), mock.patch.object(local_run, "be_nice"):
        assert local_run.screen_jobs({}) == 1
        assert local_run.screen_jobs({"jobs": "4"}) == 4
        assert local_run.screen_jobs({"jobs": "0"}) == 1


def _slow_square(n):
    import time

    time.sleep(0.01 * n)
    return n * n


def _niceness():
    import os

    return os.nice(0)


def _fail_on(n):
    if n == 1:
        raise ValueError("boom")
    return n


def _die():
    import os

    os._exit(3)
