import numpy as np
import threading
import time
import pytest
from panobbgo.heuristics.lbfgsb import LBFGSB

class MockProblem:
    def __init__(self):
        self.dim = 2
        self.box = type('Box', (), {'box': [[-1, 1], [-1, 1]]})()

class MockConfig:
    def __init__(self):
        self.capacity = 100
    def get_logger(self, name):
        import logging
        return logging.getLogger(name)

class MockStrategy:
    def __init__(self):
        self.problem = MockProblem()
        self.name = "Mock"
        self.config = MockConfig()

def test_lbfgsb_init_fails(monkeypatch):
    strategy = MockStrategy()
    h = LBFGSB(strategy)

    # Mock multiprocessing context to raise an exception
    import multiprocessing
    def mock_get_context(method):
        raise RuntimeError("Simulated MP failure")

    monkeypatch.setattr(multiprocessing, "get_context", mock_get_context)

    with pytest.raises(RuntimeError, match="Simulated MP failure"):
        h.__start__()

def test_worker_logic():
    # Test the static worker method directly

    class MockPipe:
        def __init__(self):
            self.sent = []
            self.recv_values = [1.0, 0.5, 0.1]
            self.recv_idx = 0

        def send(self, val):
            self.sent.append(val)

        def recv(self):
            val = self.recv_values[self.recv_idx]
            self.recv_idx = (self.recv_idx + 1) % len(self.recv_values)
            return val

    pipe = MockPipe()
    output = MockPipe()
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]

    # Needs to be mocked or it will loop infinitely if it converges slowly
    # but with mock pipe it might converge quickly due to gradient approx
    LBFGSB.worker(pipe, output, 2, bounds)

    assert len(output.sent) > 0 # Should have sent the result

def test_on_start_pipe_error():
    strategy = MockStrategy()
    h = LBFGSB(strategy)
    h.out1 = type('MockPipe', (), {'poll': lambda self, t: True, 'recv': lambda self: "result"})()

    class FaultyPipe:
        def poll(self, timeout):
            raise OSError("Pipe broken")

    h.p1 = FaultyPipe()

    # Run the loop, it should break and not block forever
    h.on_start()

def test_on_start_general_error():
    strategy = MockStrategy()
    h = LBFGSB(strategy)

    class ErrorPipe:
        def poll(self, timeout):
            raise ValueError("Some other error")

    h.out1 = ErrorPipe()
    h.p1 = type('MockPipe', (), {})()

    h.on_start()

def test_on_start_emit():
    strategy = MockStrategy()
    h = LBFGSB(strategy)

    # Need to set up basic mock pipes
    class GoodPipe:
        def __init__(self, val, delay=False):
            self.val = val
            self.polled = False
            self.delay = delay
        def poll(self, timeout):
            if not self.polled:
                return True
            return False
        def recv(self):
            self.polled = True
            return self.val

    h.out1 = GoodPipe("Output MSG")
    h.p1 = GoodPipe(np.array([0., 0.]))

    # We replace self.emit to catch what's emitted.
    emitted = []
    def mock_emit(x):
        emitted.append(x)
    h.emit = mock_emit

    def background():
        time.sleep(0.1)
        h._stopped = True

    t = threading.Thread(target=background)
    t.start()

    # It should emit the value from p1
    h.on_start()
    t.join()

    assert len(emitted) == 1
    np.testing.assert_array_equal(emitted[0], np.array([0., 0.]))
