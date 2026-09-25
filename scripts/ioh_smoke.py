#!/usr/bin/env python
# -*- coding: utf8 -*-
"""Smoke test for the panobbgo -> IOH adapter.

Runs one strategy spec (default ``Rewarding_Diverse``) on a single MA-BBOB
instance at the competition budget (2000 * d) through the same code path as
the IOH benchmark (:func:`panobbgo.harness_ioh._run_one`): the budget goes
into the strategy's constructor, the run is seeded and evaluated
synchronously, so a given ``--seed`` reproduces the same AOCC.

Usage::

    uv run python scripts/ioh_smoke.py
    uv run python scripts/ioh_smoke.py --dim 5 --instance 0 --max-eval 500
    uv run python scripts/ioh_smoke.py --strategy Baseline_Random
"""

from __future__ import annotations

import argparse

from panobbgo.harness import _make_full_strategies, _make_quick_strategies, _make_standard_strategies
from panobbgo.harness_baselines import make_baseline_strategies
from panobbgo.harness_ioh import _run_one
from panobbgo.ioh_runner import AOCC_LOG_HI, AOCC_LOG_LO


def _specs():
    specs = {}
    for make in (_make_quick_strategies, _make_standard_strategies, _make_full_strategies, make_baseline_strategies):
        for spec in make():
            specs.setdefault(spec.name, spec)
    return specs


def main() -> int:
    specs = _specs()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", default="Rewarding_Diverse", choices=sorted(specs))
    parser.add_argument(
        "--max-eval",
        type=int,
        default=None,
        help="Override budget; default is 2000*d per MA-BBOB anytime rules.",
    )
    args = parser.parse_args()

    budget = args.max_eval if args.max_eval is not None else 2000 * args.dim
    rec = _run_one(
        specs[args.strategy],
        "MA-BBOB",
        args.dim,
        args.instance,
        0,
        budget,
        args.seed,
        {},
        AOCC_LOG_LO,
        AOCC_LOG_HI,
        sync_eval=True,
    )
    print(
        f"{rec.strategy_name} on MA-BBOB d={rec.dim} inst={rec.instance} seed={rec.seed}: "
        f"n_evals={rec.n_evals}/{budget}  best_fx={rec.best_fx:.6f}  f_opt={rec.f_opt:.6f}  "
        f"precision={rec.best_fx - rec.f_opt:.3e}  AOCC={rec.aocc:.4f}  elapsed={rec.elapsed_s:.1f}s"
    )
    if rec.error:
        print(f"error: {rec.error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
