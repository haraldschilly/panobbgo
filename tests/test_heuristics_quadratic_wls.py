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

    def test_on_new_best_box(self):
        strategy_mock = mock.MagicMock()
        attach_spawn_rng(strategy_mock)
        strategy_mock.constraint_handler.get_penalty_value = lambda r: r.fx
        strategy_mock.problem.box.box = [(-5.0, 5.0), (-5.0, 5.0)]

        with mock.patch("panobbgo.core.HeuristicSubprocess.problem", new_callable=mock.PropertyMock) as prop_mock:
            prop_mock.return_value = strategy_mock.problem
            wls_model = QuadraticWlsModel(strategy_mock)
            wls_model.pipe = mock.MagicMock()
            wls_model.pipe.poll.return_value = True
            wls_model.pipe.recv.return_value = (1, np.array([0.5, 0.5]))

            best_box_mock = mock.MagicMock()

            class ResultMock:
                def __init__(self, x, fx):
                    self.x = x
                    self.fx = fx

            best_box_mock.results = [ResultMock(np.array([1.0, 1.0]), 2.0), ResultMock(np.array([2.0, 2.0]), 8.0)]
            best_box_mock.best = best_box_mock.results[0]

            with mock.patch.object(wls_model, "emit") as emit_mock:
                wls_model.on_new_best_box(best_box_mock)

                wls_model.pipe.send.assert_called_once()
                emit_mock.assert_called_once()

    def test_on_new_best_box_timeout(self):
        strategy_mock = mock.MagicMock()
        attach_spawn_rng(strategy_mock)
        strategy_mock.constraint_handler.get_penalty_value = lambda r: r.fx
        strategy_mock.problem.box.box = [(-5.0, 5.0), (-5.0, 5.0)]

        with mock.patch("panobbgo.core.HeuristicSubprocess.problem", new_callable=mock.PropertyMock) as prop_mock:
            prop_mock.return_value = strategy_mock.problem
            wls_model = QuadraticWlsModel(strategy_mock)
            wls_model.pipe = mock.MagicMock()
            wls_model.pipe.poll.return_value = False

            best_box_mock = mock.MagicMock()

            class ResultMock:
                def __init__(self, x, fx):
                    self.x = x
                    self.fx = fx

            best_box_mock.results = [ResultMock(np.array([1.0, 1.0]), 2.0), ResultMock(np.array([2.0, 2.0]), 8.0)]
            best_box_mock.best = best_box_mock.results[0]

            with mock.patch.object(wls_model.logger, "warning") as warn_mock:
                wls_model.on_new_best_box(best_box_mock)
                warn_mock.assert_called_once()

    def test_on_new_best_box_exception(self):
        strategy_mock = mock.MagicMock()
        attach_spawn_rng(strategy_mock)
        strategy_mock.constraint_handler.get_penalty_value = lambda r: r.fx
        strategy_mock.problem.box.box = [(-5.0, 5.0), (-5.0, 5.0)]

        with mock.patch("panobbgo.core.HeuristicSubprocess.problem", new_callable=mock.PropertyMock) as prop_mock:
            prop_mock.return_value = strategy_mock.problem
            wls_model = QuadraticWlsModel(strategy_mock)
            wls_model.pipe = mock.MagicMock()
            wls_model.pipe.send.side_effect = Exception("Test Exception")

            best_box_mock = mock.MagicMock()

            class ResultMock:
                def __init__(self, x, fx):
                    self.x = x
                    self.fx = fx

            best_box_mock.results = [
                ResultMock(np.array([1.0, 1.0]), 2.0),
            ]
            best_box_mock.best = best_box_mock.results[0]

            with mock.patch.object(wls_model.logger, "error") as err_mock:
                wls_model.on_new_best_box(best_box_mock)
                err_mock.assert_called_once()

    def test_late_reply_to_a_timed_out_request_is_dropped(self):
        """Regression: after one timeout every later emission was the answer to the box before."""
        strategy_mock = mock.MagicMock()
        attach_spawn_rng(strategy_mock)
        strategy_mock.constraint_handler.get_penalty_value = lambda r: r.fx
        strategy_mock.problem.box.box = [(-5.0, 5.0), (-5.0, 5.0)]

        with mock.patch("panobbgo.core.HeuristicSubprocess.problem", new_callable=mock.PropertyMock) as prop_mock:
            prop_mock.return_value = strategy_mock.problem
            wls_model = QuadraticWlsModel(strategy_mock)
            wls_model.pipe = mock.MagicMock()
            wls_model.pipe.poll.return_value = True
            wls_model._request_id = 1  # request 1 timed out; its answer is still in the pipe
            stale, fresh = np.array([4.0, 4.0]), np.array([0.5, 0.5])
            wls_model.pipe.recv.side_effect = [(1, stale), (2, fresh)]

            class ResultMock:
                def __init__(self, x, fx):
                    self.x = x
                    self.fx = fx

            best_box_mock = mock.MagicMock()
            best_box_mock.results = [ResultMock(np.array([1.0, 1.0]), 2.0), ResultMock(np.array([2.0, 2.0]), 8.0)]
            best_box_mock.best = best_box_mock.results[0]

            with mock.patch.object(wls_model, "emit") as emit_mock:
                wls_model.on_new_best_box(best_box_mock)
                assert wls_model.pipe.send.call_args[0][0][0] == 2
                emit_mock.assert_called_once()
                np.testing.assert_array_equal(emit_mock.call_args[0][0], fresh)

    def test_subprocess_loop_exception(self):
        pipe_mock = mock.MagicMock()

        pipe_mock.poll.side_effect = [True, False, EOFError()]
        pipe_mock.recv.side_effect = Exception("Test Exception")

        with mock.patch("traceback.print_exc") as tb_mock:
            QuadraticWlsModel.subprocess(pipe_mock)
            tb_mock.assert_called_once()
            pipe_mock.send.assert_called_with((None, None))

    def test_on_new_best_box_none_returned(self):
        strategy_mock = mock.MagicMock()
        attach_spawn_rng(strategy_mock)
        strategy_mock.constraint_handler.get_penalty_value = lambda r: r.fx
        strategy_mock.problem.box.box = [(-5.0, 5.0), (-5.0, 5.0)]

        with mock.patch("panobbgo.core.HeuristicSubprocess.problem", new_callable=mock.PropertyMock) as prop_mock:
            prop_mock.return_value = strategy_mock.problem
            wls_model = QuadraticWlsModel(strategy_mock)
            wls_model.pipe = mock.MagicMock()
            wls_model.pipe.poll.return_value = True
            wls_model.pipe.recv.return_value = (1, None)

            best_box_mock = mock.MagicMock()

            class ResultMock:
                def __init__(self, x, fx):
                    self.x = x
                    self.fx = fx

            best_box_mock.results = [
                ResultMock(np.array([1.0, 1.0]), 2.0),
            ]
            best_box_mock.best = best_box_mock.results[0]

            with mock.patch.object(wls_model.logger, "warning") as warn_mock:
                wls_model.on_new_best_box(best_box_mock)
                warn_mock.assert_called_once()
