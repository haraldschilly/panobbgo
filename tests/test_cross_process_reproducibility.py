# -*- coding: utf8 -*-
"""A seeded run must be reproducible *across processes*, not just within one.

``tests/test_reproducibility.py`` and ``tests/test_harness_reproducibility.py``
compare two runs inside a single interpreter.  That is blind to two whole
classes of bug:

1.  **Per-process hash randomisation.**  Iterating a ``set`` (or a ``dict``
    built from one) of string keys — the ``who`` request ids of the DE family
    and PSO are the obvious candidates — orders differently in every
    interpreter, which reorders the RNG draws that follow.
2.  **Thread scheduling.**  Handlers run on the :class:`~panobbgo.core.EventBus`
    dispatcher thread while the main loop carries on.  Any handler that reads
    strategy-wide state the main thread is concurrently updating gets a
    scheduling-dependent answer.  This is what actually bit: until 2026-09-11
    :meth:`Results.add_results <panobbgo.core.Results.add_results>` published
    ``new_results`` *before* appending the batch to its buffer, so
    ``len(strategy.results)`` — the clock behind L-SHADE's LPSR population
    schedule, its F-schedule and its ``p_best`` annealing — counted the
    current batch or not depending on which thread won.  A single LPSR step
    landing one batch early shifts the RNG stream and therefore every point
    drawn afterwards.  Three of the 48 cells of the constrained family
    battery moved by up to ``|ΔAOCC| = 0.095`` between two runs of the same
    code, seed and ``sync_eval=True``.

So this module runs the same tiny optimisation in *separate* interpreters,
with deliberately different ``PYTHONHASHSEED`` values, and demands one
trajectory hash.  It doubles as its own worker: ``python
tests/test_cross_process_reproducibility.py <arm> <max_eval> <seed>`` prints
``OK <sha256>``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

#: Interpreter hash seeds the workers run under.  ``0`` disables
#: randomisation entirely, the other two pick two unrelated tables, so a
#: hash-ordered iteration anywhere in the run has to show up as a mismatch.
HASH_SEEDS = ("0", "1", "12345")

#: Arms to pin.  The DE family (``LSHADE``/``JSO``) carries the
#: ``len(strategy.results)``-paced schedules, ``PSO`` the other ``who``-keyed
#: bookkeeping, ``CMAES`` is the control that was already reproducible.
ARMS = ("LSHADE", "JSO", "PSO", "CMAES")

#: Small on purpose: the whole module has to stay inside a few seconds per
#: arm, and the schedules that raced are paced on *relative* progress, so a
#: short budget exercises them just as well as a long one.
MAX_EVAL = 260
SEED = 42

_REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Worker: one run, one hash.  Kept import-light so it also works standalone.
# ---------------------------------------------------------------------------


def _trajectory_hash(arm: str, max_eval: int, seed: int) -> str:
    """SHA-256 over the whole ``(x, fx, who)`` trajectory of one run."""
    import numpy as np

    from panobbgo.heuristics import CMAES, JSO, LSHADE, PSO
    from panobbgo.lib.classic import Rastrigin
    from panobbgo.strategies import StrategyRoundRobin

    factories = {
        "LSHADE": lambda st: LSHADE(st, NP_init="auto"),
        "JSO": lambda st: JSO(st, NP_init="auto"),
        "PSO": lambda st: PSO(st),
        "CMAES": lambda st: CMAES(st),
    }

    problem = Rastrigin(3)
    strategy = StrategyRoundRobin(problem, parse_args=False, seed=seed)
    strategy.config.max_eval = max_eval
    strategy.config.sync_evaluation = True
    strategy.config.stop_on_convergence = False
    strategy.add_heuristic(factories[arm](strategy))
    strategy.start()

    df = strategy.results.results
    assert df is not None and len(df) >= max_eval, f"{arm}: only {0 if df is None else len(df)} results"
    x = np.asarray(df["x"].to_numpy(dtype=float))
    fx = np.asarray(df["fx"].to_numpy(dtype=float)).ravel()
    who = "\n".join(str(w) for w in np.asarray(df["who"].to_numpy()).ravel())

    digest = hashlib.sha256()
    digest.update(x.tobytes())
    digest.update(fx.tobytes())
    digest.update(who.encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _run_worker(arm: str, hash_seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    # Keep the workers single-threaded: BLAS pools add wall-clock jitter
    # without adding coverage, and a dozen of them would oversubscribe CI.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arm, str(MAX_EVAL), str(SEED)],
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"worker {arm} (PYTHONHASHSEED={hash_seed}) failed with {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n{proc.stderr[-2000:]}"
        )
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("OK ")]
    assert line, f"worker {arm} printed no hash:\n{proc.stdout[-2000:]}"
    return line[-1].split(None, 1)[1].strip()


@pytest.mark.parametrize("arm", ARMS)
def test_trajectory_is_identical_across_processes(arm: str) -> None:
    """One seed, three interpreters, three hash seeds -> one trajectory."""
    with ThreadPoolExecutor(max_workers=len(HASH_SEEDS)) as pool:
        hashes = list(pool.map(lambda hs: _run_worker(arm, hs), HASH_SEEDS))

    assert len(set(hashes)) == 1, f"{arm}: the trajectory depends on the process it runs in — " + ", ".join(
        f"PYTHONHASHSEED={hs} -> {h[:16]}" for hs, h in zip(HASH_SEEDS, hashes)
    )


class _ResultCountProbe:
    """Event-bus subscriber that records ``len(strategy.results)`` per batch."""

    name = "ResultCountProbe"

    def __init__(self, strategy) -> None:
        self.strategy = strategy
        #: ``(batch size, len(strategy.results) as the handler saw it)``.
        self.seen: list[tuple[int, int]] = []

    def on_new_results(self, results):
        self.seen.append((len(results), len(self.strategy.results)))
        return None  # nothing to emit


def test_handlers_see_their_own_batch_in_the_result_count() -> None:
    """The deterministic half of the contract above.

    ``new_results`` is dispatched on the bus thread while the main loop runs
    on.  Everything a handler reads off the strategy therefore has to be
    settled *before* the event goes out — otherwise the handler's view
    depends on which thread wins, which is precisely the race that made the
    DE family's LPSR schedule land a batch early or late.  The count is the
    piece that actually paces the schedules, so pin it directly: at handler
    time the store already contains every result of the batch being
    announced, and nothing more.
    """
    from panobbgo.heuristics import LSHADE
    from panobbgo.lib.classic import Rastrigin
    from panobbgo.strategies import StrategyRoundRobin

    # The budget has to be big enough that the publishing thread still has
    # real work left after ``publish`` — ``add_results`` rescans and sorts
    # every ``fx`` seen so far to build its progress stats, which is what
    # gives the bus thread its window.  At 900 evaluations a publish-then-
    # store ordering is caught in tens of batches per run; at 300 it hides.
    problem = Rastrigin(2)
    strategy = StrategyRoundRobin(problem, parse_args=False, seed=SEED)
    strategy.config.max_eval = 900
    strategy.config.sync_evaluation = True
    strategy.config.stop_on_convergence = False
    strategy.add_heuristic(LSHADE(strategy, NP_init=8))

    probe = _ResultCountProbe(strategy)
    strategy.eventbus.register(probe)

    # Force the interleaving instead of hoping for it.  At the default 5 ms
    # switch interval the publishing thread almost always runs to the end of
    # ``add_results`` before the bus thread gets the GIL, so a publish-then-
    # store ordering looks fine ~11 times out of 12 — which is exactly how
    # this shipped unnoticed.  A microsecond interval hands the bus thread
    # the very next slot after ``publish``.
    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        strategy.start()
    finally:
        sys.setswitchinterval(switch_interval)

    assert probe.seen, "the probe never saw a result batch"
    total = 0
    for i, (batch, count) in enumerate(probe.seen):
        total += batch
        assert count == total, (
            f"batch {i}: handler saw len(results)={count}, expected {total} "
            f"(the {batch} results it was just handed are not in the store yet)"
        )


if __name__ == "__main__":  # worker entry point
    print("OK " + _trajectory_hash(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
