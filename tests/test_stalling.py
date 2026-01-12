#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import pytest
from panobbgo.lib.classic import Rosenbrock
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.heuristics import Random

def test_stalling_repro():
    problem = Rosenbrock(dims=2)
    strategy = StrategyRewarding(problem, parse_args=False)
    strategy.config.max_eval = 20
    strategy.config.evaluation_method = "threaded"
    strategy.config.ui_show = False

    # Add only Random heuristic for simplicity
    strategy.add(Random)

    # This should NOT hang or terminate prematurely
    strategy.start()

    print(f"Finished with {len(strategy.results)} results in {strategy.loops} loops.")
    assert len(strategy.results) >= 20

if __name__ == "__main__":
    test_stalling_repro()
