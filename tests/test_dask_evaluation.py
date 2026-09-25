"""Unit tests for :mod:`panobbgo.dask_evaluation` with an in-process fake client."""

from types import SimpleNamespace

from panobbgo import dask_evaluation


class _Future:
    def __init__(self, fn, *args):
        self.key = "k%d" % id(self)
        try:
            self._value, self._exc = fn(*args), None
        except Exception as exc:  # what a real future would re-raise
            self._value, self._exc = None, exc

    def done(self):
        return True

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._value


class _Client:
    def submit(self, fn, *args, **kwargs):
        return _Future(fn, *args)


def _strategy(problem):
    walltimes = []
    errors = []
    failed = []
    return SimpleNamespace(
        _client=_Client(),
        _problem_future=problem,
        pending={},
        new_finished=[],
        n_finished=0,
        show_last=float("inf"),
        config=SimpleNamespace(show_interval=1e9),
        record_walltime=walltimes.append,
        logger=SimpleNamespace(error=errors.append),
        walltimes=walltimes,
        errors=errors,
        _publish_failures=failed.extend,
        failed=failed,
    )


def test_failed_task_walltime_is_booked_like_a_successful_one():
    def problem(point):
        if point == "bad":
            raise ValueError("boom")
        return "result-of-%s" % point

    s = _strategy(problem)
    out = dask_evaluation.run_evaluation(s, ["good", "bad"])

    assert out == ["result-of-good"]
    assert len(s.walltimes) == 2  # the failed evaluation counts too
    assert s.n_finished == 2 and not s.pending
    assert len(s.errors) == 1 and "boom" in s.errors[0]
    assert s.failed == ["bad"]  # published as failed_evaluations, not only logged


def test_sync_is_ignored_with_one_warning():
    """evaluation.sync was dropped silently for dask (evaluation.timeout applies since 2026-09-25)."""
    s = _strategy(lambda point: point)
    warnings = []
    s.logger = SimpleNamespace(error=s.errors.append, warning=warnings.append)
    s.config = SimpleNamespace(show_interval=1e9, evaluation_timeout=5.0, sync_evaluation=True)
    dask_evaluation.run_evaluation(s, ["a"])
    dask_evaluation.run_evaluation(s, ["b"])
    assert len(warnings) == 1
    assert "evaluation.sync" in warnings[0] and "evaluation.timeout" not in warnings[0]


def test_no_warning_without_the_options():
    s = _strategy(lambda point: point)
    s.logger = SimpleNamespace(error=s.errors.append, warning=lambda msg: (_ for _ in ()).throw(AssertionError(msg)))
    dask_evaluation.run_evaluation(s, ["a"])


class _Hanging:
    """A future that never finishes (a task stuck on the cluster)."""

    def __init__(self, fn, *args):
        self.key = "h%d" % id(self)
        self.cancelled = False

    def done(self):
        return False

    def cancel(self):
        self.cancelled = True


def test_timeout_books_an_outstanding_future_as_a_nan_placeholder():
    """evaluation.timeout for dask: measured from submission, released, booked once."""
    import time

    s = _strategy(lambda point: point)
    s._client = SimpleNamespace(submit=lambda fn, *args, **kw: _Hanging(fn, *args))
    s.config = SimpleNamespace(show_interval=1e9, evaluation_timeout=0.05, sync_evaluation=False)
    s.logger = SimpleNamespace(error=s.errors.append, warning=lambda msg: None)
    booked = []
    s._timed_out_result = lambda point, seconds=None: booked.append(point) or ("nan-result", point)
    assert dask_evaluation.run_evaluation(s, ["slow"]) == []  # not yet past the limit
    [future] = s.pending.values()
    time.sleep(0.1)
    out = dask_evaluation.run_evaluation(s, [])
    assert out == [("nan-result", "slow")]
    assert booked == ["slow"] and future.cancelled
    assert not s.pending and s.n_finished == 1 and len(s.walltimes) == 1
    assert dask_evaluation.run_evaluation(s, []) == []  # booked once
