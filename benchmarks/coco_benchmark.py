#!/usr/bin/env python
# -*- coding: utf8 -*-
# Copyright 2012-2026 Panobbgo Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
COCO/BBOB Benchmark Runner
===========================

Runs panobbgo optimization strategies on the COCO/BBOB benchmark suite,
producing results compatible with the ``cocopp`` post-processing tools.

Usage::

    uv run python benchmarks/coco_benchmark.py [options]

Quick test on a small subset::

    uv run python benchmarks/coco_benchmark.py \\
        --suite-options "dimensions: 2 function_indices: 1-3 instance_indices: 1" \\
        --budget 5 --verbose

After completion, post-process results with::

    uv run python -m cocopp exdata/panobbgo-phased-on-bbob
"""

import argparse
import logging
from typing import Optional

import cocoex

from panobbgo.lib.coco_wrapper import CocoProblem
from panobbgo.strategies.phased import StrategyPhased


def create_default_phases():
    """Return the default 2-phase configuration for StrategyPhased.

    Phase 1 (30%): Exploration via RoundRobin with Random + LatinHypercube.
    Phase 2 (70%): Exploitation via Rewarding with GP + LBFGSB + ClaudeHeuristic.
    """
    from panobbgo.heuristics import (
        ClaudeHeuristic,
        GaussianProcessHeuristic,
        LatinHypercube,
        LBFGSB,
        Random,
    )
    from panobbgo.strategies.rewarding import StrategyRewarding
    from panobbgo.strategies.round_robin import StrategyRoundRobin

    return [
        {
            "pct": 30,
            "strategy": (StrategyRoundRobin, {"size": 10}),
            "heuristics": [(Random, {}), (LatinHypercube, {"div": 5})],
        },
        {
            "strategy": (StrategyRewarding, {}),
            "heuristics": [
                (GaussianProcessHeuristic, {}),
                (LBFGSB, {}),
                (ClaudeHeuristic, {}),
            ],
        },
    ]


def run_coco_benchmark(
    suite_name: str = "bbob",
    budget_multiplier: int = 10,
    result_folder: str = "panobbgo-phased-on-bbob",
    suite_options: str = "",
    phases: Optional[list] = None,
    verbose: bool = False,
):
    """Run the full COCO benchmark experiment.

    Parameters
    ----------
    suite_name
        COCO suite name (e.g. ``'bbob'``, ``'bbob-constrained'``).
    budget_multiplier
        Budget per problem = ``budget_multiplier * dimension``.
    result_folder
        Output folder name under ``exdata/``.
    suite_options
        COCO suite filter options, e.g.
        ``"dimensions: 2,5 function_indices: 1-5"``.
    phases
        Phase configuration list for :class:`StrategyPhased`.
        If ``None``, uses the default 2-phase explore/exploit setup.
    verbose
        Enable verbose logging.
    """
    if phases is None:
        phases = create_default_phases()

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(name)s | %(message)s")
    logger = logging.getLogger("coco_benchmark")

    suite = cocoex.Suite(suite_name, "", suite_options)
    observer = cocoex.Observer(suite_name, f"result_folder: {result_folder}")
    repeater = cocoex.ExperimentRepeater(budget_multiplier)
    minimal_print = cocoex.utilities.MiniPrint()

    logger.info(
        "Starting COCO benchmark: suite=%s, budget_multiplier=%d",
        suite_name,
        budget_multiplier,
    )
    logger.info("Output folder: exdata/%s", result_folder)

    problems_done = 0

    while not repeater.done():
        for coco_problem in suite:
            if repeater.done(coco_problem):
                continue

            coco_problem.observe_with(observer)

            dim = coco_problem.dimension
            budget = budget_multiplier * dim

            logger.debug(
                "%s (dim=%d, budget=%d)",
                coco_problem.id,
                dim,
                budget,
            )

            # Wrap as panobbgo Problem
            pano_problem = CocoProblem(coco_problem)

            # Create and configure StrategyPhased
            strategy = StrategyPhased(
                pano_problem,
                phases=phases,
                parse_args=False,
                testing_mode=True,
            )
            strategy.config.max_eval = budget
            strategy.config.evaluation_method = "threaded"

            # Run optimization
            try:
                strategy.start()
            except Exception:
                logger.exception("Error on %s", coco_problem.id)
                continue

            # Log result
            best = strategy.best
            if best is not None:
                logger.debug(
                    "  -> best f(x) = %.6e (COCO evals: %d)",
                    best.fx,
                    coco_problem.evaluations,
                )

            repeater.track(coco_problem)
            minimal_print(coco_problem)
            problems_done += 1

    logger.info(
        "Benchmark complete: %d problem instances processed",
        problems_done,
    )
    logger.info(
        "Post-process with: uv run python -m cocopp exdata/%s",
        result_folder,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run panobbgo on COCO/BBOB benchmark suite",
    )
    parser.add_argument(
        "--suite",
        default="bbob",
        help="COCO suite name (default: bbob)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=10,
        help="Budget multiplier (budget = multiplier * dimension, default: 10)",
    )
    parser.add_argument(
        "--output",
        default="panobbgo-phased-on-bbob",
        help="Result folder name (default: panobbgo-phased-on-bbob)",
    )
    parser.add_argument(
        "--suite-options",
        default="",
        help='COCO suite filter options (e.g., "dimensions: 2,5")',
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    run_coco_benchmark(
        suite_name=args.suite,
        budget_multiplier=args.budget,
        result_folder=args.output,
        suite_options=args.suite_options,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
