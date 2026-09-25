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
