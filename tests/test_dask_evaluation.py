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


class _Sleeper:
    """Picklable objective (module level): sleeps ``x`` seconds, raises for ``x < 0``."""

    def __call__(self, point):
        import time

        if point < 0:
            raise ValueError("negative")
        time.sleep(point)
        return "slept-%s" % point


def test_evaluate_point_enforces_the_timeout_per_call_in_a_child():
    import time

    t0 = time.time()
    result, walltime, error, timed_out = dask_evaluation.evaluate_point(_Sleeper(), 30, timeout=1.0)
    assert timed_out and result is None and error is None
    assert 1.0 <= walltime < 5.0 and time.time() - t0 < 25
    assert dask_evaluation.evaluate_point(_Sleeper(), 0, timeout=10.0)[0] == "slept-0"
    _, _, error, timed_out = dask_evaluation.evaluate_point(_Sleeper(), -1, timeout=10.0)
    assert not timed_out and "negative" in error


def test_evaluate_point_without_a_timeout_runs_in_the_worker():
    result, _, error, timed_out = dask_evaluation.evaluate_point(lambda p: p * 2, 21)  # a lambda: no pickling
    assert (result, error, timed_out) == (42, None, False)


def test_a_timed_out_task_is_booked_as_a_placeholder():
    s = _strategy(lambda point: point)
    s._client = SimpleNamespace(submit=lambda fn, *args, **kw: _Done((None, 1.5, None, True)))
    booked = []
    s._timed_out_result = lambda point, seconds=None: booked.append((point, seconds)) or ("nan-result", point)
    assert dask_evaluation.run_evaluation(s, ["slow"]) == [("nan-result", "slow")]
    assert booked == [("slow", 1.5)] and s.walltimes == [1.5] and not s.pending and s.failed == []


class _Done:
    def __init__(self, value):
        self.key = "d%d" % id(self)
        self._value = value

    def done(self):
        return True

    def result(self):
        return self._value
