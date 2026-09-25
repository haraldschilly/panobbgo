from tests.support import attach_spawn_rng
import unittest
import unittest.mock as mock
import numpy as np

from panobbgo.heuristics.quadratic_wls import QuadraticWlsModel, rank_weights


class TestQuadraticWLSModel(unittest.TestCase):
    def test_rank_weights_favour_the_nearest_point(self):
        """Regression: argsort indices were used as ranks, so [5, 1, 3] weighted the farthest point 1."""
        np.testing.assert_allclose(rank_weights(np.array([5.0, 1.0, 3.0])), [1 / 3, 1.0, 1 / 2])

    def test_subprocess_loop(self):
        pipe_mock = mock.MagicMock()

        pipe_mock.poll.side_effect = [True, False, EOFError()]

        points = np.array([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0]])
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        best_point = np.array([0.0, 0.0])
        fx_vals = np.array([2.0, 8.0, 2.0])

        payload = (7, points, bounds, best_point, fx_vals)
        pipe_mock.recv.return_value = payload

        QuadraticWlsModel.subprocess(pipe_mock)

        pipe_mock.send.assert_called_once()
        req_id, sent_val = pipe_mock.send.call_args[0][0]
        self.assertEqual(req_id, 7)  # the reply names the request it answers
        self.assertIsNotNone(sent_val)
        self.assertEqual(len(sent_val), 2)

    def _model(self, sync=True):
        """A model on a mock strategy with its pipe replaced by a mock (the real worker idles)."""
        strategy_mock = mock.MagicMock()
        attach_spawn_rng(strategy_mock)
        strategy_mock.constraint_handler.get_penalty_value = lambda r: r.fx
        strategy_mock.problem.box.box = [(-5.0, 5.0), (-5.0, 5.0)]
        strategy_mock.config.sync_evaluation = sync
        strategy_mock.config.capacity = 20
        strategy_mock.problem.project = lambda x: np.asarray(x, dtype=float)
        patcher = mock.patch("panobbgo.core.HeuristicSubprocess.problem", new_callable=mock.PropertyMock)
        prop_mock = patcher.start()
        self.addCleanup(patcher.stop)
        prop_mock.return_value = strategy_mock.problem
        wls_model = QuadraticWlsModel(strategy_mock)
        self.addCleanup(wls_model.__stop__)
        real_pipe = wls_model.pipe
        self.addCleanup(real_pipe.close)
        wls_model.pipe = mock.MagicMock()
        wls_model._poll_slice = 0.01
        return wls_model

    @staticmethod
    def _box(*pairs):
        class ResultMock:
            def __init__(self, x, fx):
                self.x = x
                self.fx = fx

        box = mock.MagicMock()
        box.results = [ResultMock(np.array(x, dtype=float), fx) for x, fx in pairs]
        box.best = box.results[0]
        return box

    def test_on_new_best_box_only_records(self):
        """The event handler never touches the pipe: a slow fit cannot block the event bus."""
        wls_model = self._model()
        wls_model.on_new_best_box(self._box(([1.0, 1.0], 2.0), ([2.0, 2.0], 8.0)))
        wls_model.pipe.send.assert_not_called()
        wls_model.pipe.poll.assert_not_called()
        wls_model.pipe.recv.assert_not_called()
        assert wls_model._pending_fit is not None
        assert wls_model.can_produce

    def test_produce_sends_the_box_and_emits_the_fit(self):
        wls_model = self._model()
        wls_model.pipe.poll.return_value = True
        wls_model.pipe.recv.return_value = (1, np.array([0.5, 0.5]))
        wls_model.on_new_best_box(self._box(([1.0, 1.0], 2.0), ([2.0, 2.0], 8.0)))

        points = wls_model.produce(1)

        wls_model.pipe.send.assert_called_once()
        req_id, pts, bounds, best_x, fx_vals = wls_model.pipe.send.call_args[0][0]
        assert req_id == 1
        np.testing.assert_array_equal(pts, [[1.0, 1.0], [2.0, 2.0]])
        np.testing.assert_array_equal(best_x, [1.0, 1.0])
        np.testing.assert_array_equal(fx_vals, [2.0, 8.0])
        assert bounds == [(-5.0, 5.0), (-5.0, 5.0)]
        assert len(points) == 1
        np.testing.assert_array_equal(points[0].x, [0.5, 0.5])
        assert not wls_model.can_produce  # nothing recorded, nothing in flight

    def test_boxes_before_a_pull_coalesce_to_the_latest(self):
        wls_model = self._model()
        wls_model.pipe.poll.return_value = True
        wls_model.pipe.recv.return_value = (1, np.array([0.5, 0.5]))
        wls_model.on_new_best_box(self._box(([1.0, 1.0], 2.0)))
        wls_model.on_new_best_box(self._box(([3.0, 3.0], 1.0)))

        wls_model.produce(1)

        wls_model.pipe.send.assert_called_once()
        np.testing.assert_array_equal(wls_model.pipe.send.call_args[0][0][3], [3.0, 3.0])

    def test_async_pull_skips_until_the_fit_is_ready(self):
        """Without sync evaluation produce() never waits; the reply is taken on a later pull."""
        wls_model = self._model(sync=False)
        wls_model.pipe.poll.return_value = False
        wls_model.on_new_best_box(self._box(([1.0, 1.0], 2.0), ([2.0, 2.0], 8.0)))

        assert wls_model.produce(1) == []
        wls_model.pipe.send.assert_called_once()
        assert wls_model.can_produce  # the fit is on its way
        # Only non-blocking looks at the pipe, never a wait.
        assert all(c.args == (0,) for c in wls_model.pipe.poll.call_args_list)

        wls_model.pipe.poll.return_value = True
        wls_model.pipe.recv.return_value = (1, np.array([0.5, 0.5]))
        points = wls_model.produce(1)
        assert len(points) == 1
        wls_model.pipe.send.assert_called_once()  # no second request

    def test_sync_backstop_abandons_a_wedged_worker(self):
        wls_model = self._model()
        wls_model.fit_timeout = 0.05
        wls_model.pipe.poll.return_value = False
        wls_model.on_new_best_box(self._box(([1.0, 1.0], 2.0), ([2.0, 2.0], 8.0)))

        with mock.patch.object(wls_model.logger, "error") as err_mock:
            assert wls_model.produce(1) == []
            err_mock.assert_called_once()
        assert wls_model._inflight_id is None

    def test_send_exception_is_logged(self):
        wls_model = self._model()
        wls_model.pipe.send.side_effect = Exception("Test Exception")
        wls_model.on_new_best_box(self._box(([1.0, 1.0], 2.0)))

        with mock.patch.object(wls_model.logger, "error") as err_mock:
            assert wls_model.produce(1) == []
            err_mock.assert_called_once()

    def test_late_reply_to_an_abandoned_request_is_dropped(self):
        """Regression: after one timeout every later emission was the answer to the box before."""
        wls_model = self._model()
        wls_model.pipe.poll.return_value = True
        wls_model._request_id = 1  # request 1 was abandoned; its answer is still in the pipe
        stale, fresh = np.array([4.0, 4.0]), np.array([0.5, 0.5])
        wls_model.pipe.recv.side_effect = [(1, stale), (2, fresh)]
        wls_model.on_new_best_box(self._box(([1.0, 1.0], 2.0), ([2.0, 2.0], 8.0)))

        points = wls_model.produce(1)

        assert wls_model.pipe.send.call_args[0][0][0] == 2
        assert len(points) == 1
        np.testing.assert_array_equal(points[0].x, fresh)

    def test_stopped_model_does_not_touch_the_pipe(self):
        wls_model = self._model()
        wls_model.on_new_best_box(self._box(([1.0, 1.0], 2.0)))
        wls_model._stopped = True
        assert wls_model.produce(1) == []
        wls_model.pipe.send.assert_not_called()

    def test_subprocess_loop_exception(self):
        pipe_mock = mock.MagicMock()

        pipe_mock.poll.side_effect = [True, False, EOFError()]
        pipe_mock.recv.side_effect = Exception("Test Exception")

        with mock.patch("traceback.print_exc") as tb_mock:
            QuadraticWlsModel.subprocess(pipe_mock)
            tb_mock.assert_called_once()
            pipe_mock.send.assert_called_with((None, None))

    def test_fit_failure_is_logged(self):
        wls_model = self._model()
        wls_model.pipe.poll.return_value = True
        wls_model.pipe.recv.return_value = (1, None)
        wls_model.on_new_best_box(self._box(([1.0, 1.0], 2.0)))

        with mock.patch.object(wls_model.logger, "warning") as warn_mock:
            assert wls_model.produce(1) == []
            warn_mock.assert_called_once()
