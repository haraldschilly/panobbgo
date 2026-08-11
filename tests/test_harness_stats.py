#!/usr/bin/env python
# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
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
Tests for the statistical acceptance rule (bootstrap CI on composite delta).

Exercises :func:`panobbgo.harness.statistical_accept` and the
``--statistical`` CLI path on ``benchmark_harness.py compare``.  Pairs with
``planning/SELF_IMPROVEMENT_LOOP.md`` §6.2.
"""

from __future__ import annotations

import json
import pathlib
from typing import List, Optional

import numpy as np
import pytest

from panobbgo.harness import (
    ConvergencePoint,
    HarnessConfig,
    HarnessResult,
    PairCI,
    ProblemStrategyResult,
    RunRecord,
    StatisticalDecision,
    _solve_fractions,
    statistical_accept,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _run(first_hit: Optional[int], budget: int = 100) -> RunRecord:
    """Synthetic RunRecord with a controlled first-hit evaluation."""
    success = first_hit is not None
    convergence: List[ConvergencePoint] = []
    if first_hit is not None:
        convergence.append(ConvergencePoint(eval_idx=first_hit, fx=0.05, func_distance=0.05))
    return RunRecord(
        problem_name="P",
        problem_dim=2,
        strategy_name="S",
        rep=0,
        seed=0,
        budget=budget,
        evaluations_used=budget,
        best_fx=0.05 if success else 999.0,
        f_opt=0.0,
        func_distance=0.05 if success else 1.0,
        tolerance=0.1,
        success=success,
        convergence=convergence,
        heuristic_counts={},
        duration=0.1,
    )


def _psr(
    prob: str,
    strat: str,
    first_hits: List[Optional[int]],
    budget: int = 100,
) -> ProblemStrategyResult:
    """Build a ProblemStrategyResult from a list of first-hit evaluations."""
    runs = [_run(h, budget=budget) for h in first_hits]
    for r in runs:
        r.problem_name = prob
        r.strategy_name = strat
    psr = ProblemStrategyResult(
        problem_name=prob,
        problem_dim=2,
        strategy_name=strat,
        f_opt=0.0,
        tolerance=0.1,
        budget=budget,
        runs=runs,
    )
    psr.compute_metrics()
    return psr


def _harness_result(psrs: List[ProblemStrategyResult]) -> HarnessResult:
    composite = float(np.mean([p.score for p in psrs])) if psrs else 0.0
    return HarnessResult(
        config=HarnessConfig(mode="quick"),
        timestamp="2026-01-01T00:00:00+00:00",
        total_runs=sum(len(p.runs) for p in psrs),
        total_duration=0.0,
        problem_strategy_results=psrs,
        composite_score=composite,
    )


# ===========================================================================
# _solve_fractions
# ===========================================================================


class TestSolveFractions:
    def test_all_solve_at_eval_1(self):
        """Solving at evaluation 1 gives fraction 1.0."""
        psr = _psr("P", "S", [1, 1, 1], budget=100)
        fr = _solve_fractions(psr)
        assert fr.shape == (3,)
        assert np.allclose(fr, 1.0)

    def test_all_solve_at_budget(self):
        """Solving at the last evaluation gives fraction 1/budget."""
        psr = _psr("P", "S", [100, 100], budget=100)
        fr = _solve_fractions(psr)
        assert np.allclose(fr, 0.01)

    def test_failure_scores_zero(self):
        psr = _psr("P", "S", [None, None], budget=100)
        fr = _solve_fractions(psr)
        assert np.allclose(fr, 0.0)

    def test_mixed_reps(self):
        psr = _psr("P", "S", [1, 50, None], budget=100)
        fr = _solve_fractions(psr)
        assert fr[0] == pytest.approx(1.0)
        # fraction = 1 - (50-1)/100 = 0.51
        assert fr[1] == pytest.approx(0.51)
        assert fr[2] == pytest.approx(0.0)

    def test_matches_psr_score(self):
        """Mean of _solve_fractions must equal ProblemStrategyResult.score."""
        psr = _psr("P", "S", [10, 50, None, 20], budget=100)
        fr = _solve_fractions(psr)
        assert float(fr.mean()) == pytest.approx(psr.score)

    def test_empty_runs(self):
        psr = ProblemStrategyResult(
            problem_name="P",
            problem_dim=2,
            strategy_name="S",
            f_opt=0.0,
            tolerance=0.1,
            budget=100,
            runs=[],
        )
        fr = _solve_fractions(psr)
        assert fr.shape == (0,)


# ===========================================================================
# statistical_accept — accept path
# ===========================================================================


class TestStatisticalAcceptAcceptPath:
    def test_large_uniform_improvement_accepted(self):
        """All reps moved from failure to fast success → accept."""
        before = _harness_result(
            [
                _psr("P1", "S1", [None, None, None, None, None]),
                _psr("P2", "S2", [None, None, None, None, None]),
            ]
        )
        after = _harness_result(
            [
                _psr("P1", "S1", [10, 10, 10, 10, 10]),
                _psr("P2", "S2", [20, 20, 20, 20, 20]),
            ]
        )
        d = statistical_accept(before, after, eps_accept=0.005, eps_regress=0.05, n_boot=1000, seed=0)
        assert d.accept is True
        assert d.delta > 0.8
        assert d.ci_low > 0.0
        assert d.worst_pair_regression == 0.0
        assert d.worst_pair is None
        assert len(d.per_pair) == 2

    def test_small_but_stable_improvement_accepted(self):
        """Small consistent lift with negligible variance → accept."""
        before = _harness_result([_psr("P", "S", [50] * 10)])
        # 50 → 40 at budget=100 means fraction rises from 0.51 to 0.61 → +0.10.
        after = _harness_result([_psr("P", "S", [40] * 10)])
        d = statistical_accept(before, after, eps_accept=0.005, eps_regress=0.05, n_boot=2000, seed=1)
        assert d.accept is True
        assert d.delta == pytest.approx(0.10, abs=0.01)
        # Zero variance within each side → CI collapses to a point estimate.
        assert d.ci_low == pytest.approx(d.ci_high, abs=1e-9)


# ===========================================================================
# statistical_accept — reject paths
# ===========================================================================


class TestStatisticalAcceptRejectPaths:
    def test_noise_without_signal_rejected(self):
        """Identical distributions — CI straddles zero, so reject."""
        before = _harness_result([_psr("P", "S", [30, 50, 70, 90, 10])])
        after = _harness_result([_psr("P", "S", [10, 30, 50, 70, 90])])
        d = statistical_accept(before, after, eps_accept=0.005, eps_regress=0.05, n_boot=2000, seed=2)
        # The point delta may even be 0.0 here; either way the CI should
        # bracket zero and the decision should be reject.
        assert d.accept is False
        assert "CI" in " ".join(d.reasons) or "eps_accept" in " ".join(d.reasons)

    def test_regression_rejected(self):
        """Composite dropped → reject."""
        before = _harness_result([_psr("P", "S", [10, 10, 10, 10, 10])])
        after = _harness_result([_psr("P", "S", [None, None, None, None, None])])
        d = statistical_accept(before, after, eps_accept=0.005, eps_regress=0.05, n_boot=1000, seed=3)
        assert d.accept is False
        assert d.delta < 0.0

    def test_composite_up_but_one_pair_regresses_badly_rejected(self):
        """
        Composite delta positive, but one pair crashes by more than
        eps_regress → reject (the regression guard fires).
        """
        before = _harness_result(
            [
                _psr("P1", "S1", [None, None, None, None, None]),  # score 0.0
                _psr("P2", "S2", [1, 1, 1, 1, 1]),  # score 1.0
            ]
        )
        after = _harness_result(
            [
                _psr("P1", "S1", [1, 1, 1, 1, 1]),  # 0.0 → 1.0
                _psr("P2", "S2", [None, None, None, None, None]),  # 1.0 → 0.0
            ]
        )
        d = statistical_accept(before, after, eps_accept=0.005, eps_regress=0.05, n_boot=1000, seed=4)
        # Composite delta is 0.0 on average, and one pair regressed by 1.0.
        assert d.accept is False
        assert d.worst_pair_regression == pytest.approx(-1.0)
        assert d.worst_pair == ("P2", "S2")
        # Reason should mention the regression
        assert any("regressed" in r for r in d.reasons)

    def test_delta_below_eps_accept_rejected(self):
        """Tiny positive delta that doesn't clear eps_accept → reject."""
        # before: all solve at 50 → score 0.51
        # after: all solve at 49 → score 0.52 → delta = +0.01
        before = _harness_result([_psr("P", "S", [50] * 10)])
        after = _harness_result([_psr("P", "S", [49] * 10)])
        d = statistical_accept(before, after, eps_accept=0.05, eps_regress=0.05, n_boot=1000, seed=5)
        assert d.accept is False
        assert d.delta > 0.0
        assert any("eps_accept" in r for r in d.reasons)


# ===========================================================================
# statistical_accept — reproducibility
# ===========================================================================


class TestReproducibility:
    def test_same_seed_same_result(self):
        before = _harness_result([_psr("P", "S", [30, 50, 70, 10])])
        after = _harness_result([_psr("P", "S", [20, 40, 60, 5])])
        d1 = statistical_accept(before, after, n_boot=500, seed=7)
        d2 = statistical_accept(before, after, n_boot=500, seed=7)
        assert d1.delta == pytest.approx(d2.delta)
        assert d1.ci_low == pytest.approx(d2.ci_low)
        assert d1.ci_high == pytest.approx(d2.ci_high)
        assert d1.accept == d2.accept

    def test_different_seeds_give_slightly_different_cis(self):
        before = _harness_result([_psr("P", "S", [30, 50, 70, 10])])
        after = _harness_result([_psr("P", "S", [20, 40, 60, 5])])
        d1 = statistical_accept(before, after, n_boot=500, seed=7)
        d2 = statistical_accept(before, after, n_boot=500, seed=8)
        # CIs should differ (different resamples) but point deltas match.
        assert d1.delta == pytest.approx(d2.delta)
        assert (d1.ci_low, d1.ci_high) != (d2.ci_low, d2.ci_high)


# ===========================================================================
# statistical_accept — structural properties
# ===========================================================================


class TestStructural:
    def test_returns_statistical_decision(self):
        before = _harness_result([_psr("P", "S", [10, 10])])
        after = _harness_result([_psr("P", "S", [5, 5])])
        d = statistical_accept(before, after, n_boot=200, seed=9)
        assert isinstance(d, StatisticalDecision)
        assert isinstance(d.per_pair, list)
        assert all(isinstance(p, PairCI) for p in d.per_pair)

    def test_ci_low_leq_delta_leq_ci_high_typical(self):
        """For generic inputs the CI should bracket the point delta."""
        rng = np.random.default_rng(11)
        # Simulate real-ish noise with 10 reps per side.
        before_hits = rng.integers(10, 90, size=10).tolist()
        after_hits = rng.integers(10, 90, size=10).tolist()
        before = _harness_result([_psr("P", "S", before_hits)])
        after = _harness_result([_psr("P", "S", after_hits)])
        d = statistical_accept(before, after, n_boot=2000, seed=11)
        # Percentile intervals contain the mean with high probability on
        # resamples of this size; assert it holds.
        assert d.ci_low - 1e-9 <= d.delta <= d.ci_high + 1e-9

    def test_only_shared_pairs_are_bootstrapped(self):
        """Pairs present in only one side must not contribute."""
        before = _harness_result(
            [
                _psr("P1", "S1", [1, 1]),
                _psr("P2", "S2", [None, None]),  # only in before
            ]
        )
        after = _harness_result(
            [
                _psr("P1", "S1", [1, 1]),
                _psr("P3", "S3", [1, 1]),  # only in after
            ]
        )
        d = statistical_accept(before, after, n_boot=200, seed=12)
        assert len(d.per_pair) == 1
        assert d.per_pair[0].problem == "P1"
        assert d.per_pair[0].strategy == "S1"

    def test_no_shared_pairs_degenerate(self):
        """Zero shared pairs → reject, even when raw composite delta is positive."""
        before = _harness_result([_psr("P1", "S1", [None, None])])
        after = _harness_result([_psr("P2", "S2", [1, 1])])
        d = statistical_accept(before, after, n_boot=100, seed=13)
        assert d.per_pair == []
        assert d.ci_low == pytest.approx(d.ci_high)
        # No shared pairs ⇒ reject, even though raw composite shifted from
        # 0.0 → 1.0.  We have zero statistical evidence because the
        # comparisons aren't apples-to-apples.
        assert d.accept is False
        assert any("no (problem, strategy) pairs are shared" in r for r in d.reasons)

    def test_to_dict_json_serialisable(self):
        before = _harness_result([_psr("P", "S", [None])])
        after = _harness_result([_psr("P", "S", [1])])
        d = statistical_accept(before, after, n_boot=100, seed=14)
        doc = d.to_dict()
        # Should round-trip through JSON without errors.
        s = json.dumps(doc)
        assert "accept" in s
        loaded = json.loads(s)
        assert loaded["accept"] == d.accept
        assert loaded["delta"] == pytest.approx(d.delta)
        assert loaded["eps_accept"] == d.eps_accept
        assert loaded["n_boot"] == d.n_boot
        assert len(loaded["per_pair"]) == 1


# ===========================================================================
# CLI integration — benchmark_harness.py compare --statistical
# ===========================================================================


class TestCompareCLIStatistical:
    def _write_result(self, path: str, psrs: List[ProblemStrategyResult]) -> None:
        result = _harness_result(psrs)
        result.save(path)

    def test_compare_statistical_accept(self, tmp_path, capsys):
        """A large uniform improvement should print ACCEPT and exit 0."""
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from benchmark_harness import main

        before = str(tmp_path / "before.json")
        after = str(tmp_path / "after.json")
        self._write_result(
            before,
            [
                _psr("P1", "S1", [None, None, None, None, None]),
                _psr("P2", "S2", [None, None, None, None, None]),
            ],
        )
        self._write_result(
            after,
            [
                _psr("P1", "S1", [5, 5, 5, 5, 5]),
                _psr("P2", "S2", [10, 10, 10, 10, 10]),
            ],
        )
        ret = main(
            [
                "compare",
                before,
                after,
                "--statistical",
                "--n-boot",
                "300",
                "--fail-on-regression",
            ]
        )
        out = capsys.readouterr().out
        assert ret == 0
        assert "STATISTICAL DECISION: ACCEPT" in out

    def test_compare_statistical_reject_regression(self, tmp_path, capsys):
        """A one-pair crash with zero composite lift should reject and exit 2."""
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from benchmark_harness import main

        before = str(tmp_path / "before.json")
        after = str(tmp_path / "after.json")
        self._write_result(
            before,
            [
                _psr("P1", "S1", [None, None, None, None, None]),
                _psr("P2", "S2", [1, 1, 1, 1, 1]),
            ],
        )
        self._write_result(
            after,
            [
                _psr("P1", "S1", [1, 1, 1, 1, 1]),
                _psr("P2", "S2", [None, None, None, None, None]),
            ],
        )
        ret = main(
            [
                "compare",
                before,
                after,
                "--statistical",
                "--n-boot",
                "300",
                "--fail-on-regression",
            ]
        )
        out = capsys.readouterr().out
        assert ret == 2
        assert "STATISTICAL DECISION: REJECT" in out

    def test_compare_statistical_json_payload(self, tmp_path, capsys):
        """--json with --statistical should include a statistical block."""
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from benchmark_harness import main

        before = str(tmp_path / "before.json")
        after = str(tmp_path / "after.json")
        self._write_result(before, [_psr("P", "S", [50, 50, 50])])
        self._write_result(after, [_psr("P", "S", [20, 20, 20])])

        ret = main(["compare", before, after, "--statistical", "--n-boot", "200", "--json"])
        assert ret == 0
        out = capsys.readouterr().out
        # The cmd_compare prints the ASCII summary/decision, then the JSON
        # block.  The JSON block starts at a line that is exactly "{" and
        # ends at a line that is exactly "}" (indent=2 produces these).
        lines = out.splitlines()
        starts = [i for i, ln in enumerate(lines) if ln == "{"]
        ends = [i for i, ln in enumerate(lines) if ln == "}"]
        assert starts and ends
        payload = json.loads("\n".join(lines[starts[-1] : ends[-1] + 1]))
        assert "statistical" in payload
        stats = payload["statistical"]
        assert "accept" in stats
        assert "ci_low" in stats and "ci_high" in stats
        assert "per_pair" in stats and len(stats["per_pair"]) == 1


# ===========================================================================
# Paired vs unpaired bootstrap (added with the paired-bootstrap upgrade).
# ===========================================================================


class TestPairedBootstrap:
    """The randomized harness keeps reps instance-aligned by index, so the
    paired bootstrap is the statistically efficient choice and the
    historical unpaired sampler is wastefully wide.  These tests pin the
    auto-detection rule and the contract of the explicit ``paired`` knob.
    """

    def test_paired_tighter_than_unpaired_when_correlated(self):
        """Strongly correlated paired reps → paired CI strictly narrower."""
        # Same instance index gives correlated outcomes.  The per-rep delta
        # is exactly +5 evals across all reps, so the paired bootstrap CI
        # collapses to a point.  The unpaired sampler shuffles the two
        # sides independently and gets a wide CI driven by the (large)
        # within-side rep variance.
        before = _harness_result([_psr("P", "S", [50, 80, 30, 60, 20])])
        after = _harness_result([_psr("P", "S", [45, 75, 25, 55, 15])])
        d_paired = statistical_accept(before, after, n_boot=2000, seed=42, paired=True)
        d_unpaired = statistical_accept(before, after, n_boot=2000, seed=42, paired=False)
        assert d_paired.delta == pytest.approx(d_unpaired.delta)
        # Identical per-rep delta ⇒ paired CI collapses; unpaired stays wide.
        assert d_paired.ci_high - d_paired.ci_low == pytest.approx(0.0, abs=1e-9)
        assert d_unpaired.ci_high - d_unpaired.ci_low > 0.1
        assert d_paired.paired is True
        assert d_unpaired.paired is False

    def test_paired_unblocks_genuine_improvement(self):
        """A 5-eval lift on every paired rep is a real improvement that
        the unpaired bootstrap cannot distinguish from noise."""
        before = _harness_result([_psr("P", "S", [50, 80, 30, 60, 20])])
        after = _harness_result([_psr("P", "S", [45, 75, 25, 55, 15])])
        d_paired = statistical_accept(before, after, n_boot=2000, seed=42, paired=True)
        d_unpaired = statistical_accept(before, after, n_boot=2000, seed=42, paired=False)
        assert d_paired.delta > 0
        # Paired: tight CI strictly above zero ⇒ accept.
        assert d_paired.ci_low > 0.0
        assert d_paired.accept is True
        # Unpaired: CI brackets zero ⇒ reject (correct only if reps
        # were genuinely independent, which they aren't here).
        assert d_unpaired.ci_low <= 0.0
        assert d_unpaired.accept is False

    def test_auto_detect_uses_paired_when_rep_counts_match(self):
        """``paired=None`` auto-selects paired when ``n_before == n_after``."""
        before = _harness_result([_psr("P", "S", [10, 50, 90])])
        after = _harness_result([_psr("P", "S", [5, 45, 85])])
        d = statistical_accept(before, after, n_boot=500, seed=0)
        # All three pairs have matched counts → paired path should fire.
        assert d.paired is True

    def test_auto_detect_falls_back_to_unpaired_on_mismatch(self):
        """Mismatched rep counts on every shared pair → fall back to unpaired."""
        before = _harness_result([_psr("P", "S", [10, 50, 90])])  # 3 reps
        after = _harness_result([_psr("P", "S", [5, 45])])  # 2 reps
        d = statistical_accept(before, after, n_boot=500, seed=0)
        assert d.paired is False
        assert d.per_pair[0].n_before == 3
        assert d.per_pair[0].n_after == 2

    def test_force_paired_truncates_mismatched_reps(self):
        """``paired=True`` with mismatched reps truncates to the common prefix."""
        # before reps: [10, 50, 90] → after truncation [10, 50]
        # after reps:  [5, 45]
        before = _harness_result([_psr("P", "S", [10, 50, 90])])
        after = _harness_result([_psr("P", "S", [5, 45])])
        d = statistical_accept(before, after, n_boot=500, seed=0, paired=True)
        # Per-pair n_* reflect the truncated values, not the originals.
        assert d.per_pair[0].n_before == 2
        assert d.per_pair[0].n_after == 2
        assert d.paired is True

    def test_paired_field_survives_json_roundtrip(self):
        """``paired`` must round-trip through the JSON serialisation."""
        before = _harness_result([_psr("P", "S", [50, 80, 30])])
        after = _harness_result([_psr("P", "S", [45, 75, 25])])
        d = statistical_accept(before, after, n_boot=200, seed=42, paired=True)
        doc = json.loads(json.dumps(d.to_dict()))
        assert doc["paired"] is True
        d2 = statistical_accept(before, after, n_boot=200, seed=42, paired=False)
        doc2 = json.loads(json.dumps(d2.to_dict()))
        assert doc2["paired"] is False

    def test_paired_print_summary_mentions_scheme(self, capsys):
        """``print_summary`` should report which bootstrap was used."""
        before = _harness_result([_psr("P", "S", [50, 80, 30])])
        after = _harness_result([_psr("P", "S", [45, 75, 25])])
        d_paired = statistical_accept(before, after, n_boot=200, seed=42, paired=True)
        d_paired.print_summary()
        out = capsys.readouterr().out
        assert "bootstrap=paired" in out
        d_unpaired = statistical_accept(before, after, n_boot=200, seed=42, paired=False)
        d_unpaired.print_summary()
        out = capsys.readouterr().out
        assert "bootstrap=unpaired" in out

    def test_auto_detect_with_no_reps_picks_unpaired(self):
        """An empty intersection trivially leaves ``paired=False``."""
        # Both sides have zero reps.  Edge case — no per-pair bootstrap fires.
        before = _harness_result([_psr("P", "S", [])])
        after = _harness_result([_psr("P", "S", [])])
        d = statistical_accept(before, after, n_boot=200, seed=0)
        assert d.paired is False

    def test_paired_bootstrap_seed_reproducible(self):
        """Same seed + same paired flag → identical CI."""
        before = _harness_result([_psr("P", "S", [50, 80, 30, 60, 20])])
        after = _harness_result([_psr("P", "S", [45, 75, 25, 55, 15])])
        d1 = statistical_accept(before, after, n_boot=500, seed=7, paired=True)
        d2 = statistical_accept(before, after, n_boot=500, seed=7, paired=True)
        assert (d1.ci_low, d1.ci_high) == (d2.ci_low, d2.ci_high)


# ===========================================================================
# CLI: --paired / --unpaired flags on `compare --statistical`
# ===========================================================================


class TestCompareCLIPaired:
    def _write_result(self, path: str, psrs: List[ProblemStrategyResult]) -> None:
        result = _harness_result(psrs)
        result.save(path)

    def test_compare_paired_flag_unblocks_acceptance(self, tmp_path, capsys):
        """The same data that fails the unpaired gate passes when --paired
        is set, mirroring the auto-detect default for randomized runs."""
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from benchmark_harness import main

        before = str(tmp_path / "before.json")
        after = str(tmp_path / "after.json")
        self._write_result(before, [_psr("P", "S", [50, 80, 30, 60, 20])])
        self._write_result(after, [_psr("P", "S", [45, 75, 25, 55, 15])])

        # --paired ⇒ accept.
        ret_paired = main(
            [
                "compare",
                before,
                after,
                "--statistical",
                "--paired",
                "--n-boot",
                "300",
            ]
        )
        out_paired = capsys.readouterr().out
        assert ret_paired == 0
        assert "STATISTICAL DECISION: ACCEPT" in out_paired
        assert "bootstrap=paired" in out_paired

        # --unpaired ⇒ reject (CI lower bound straddles zero).
        ret_unpaired = main(
            [
                "compare",
                before,
                after,
                "--statistical",
                "--unpaired",
                "--n-boot",
                "300",
            ]
        )
        out_unpaired = capsys.readouterr().out
        assert ret_unpaired == 0
        assert "STATISTICAL DECISION: REJECT" in out_unpaired
        assert "bootstrap=unpaired" in out_unpaired

    def test_compare_paired_and_unpaired_are_mutually_exclusive(self, tmp_path, capsys):
        """argparse should reject both flags together."""
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from benchmark_harness import main

        before = str(tmp_path / "before.json")
        after = str(tmp_path / "after.json")
        self._write_result(before, [_psr("P", "S", [50])])
        self._write_result(after, [_psr("P", "S", [40])])

        with pytest.raises(SystemExit):
            main(
                [
                    "compare",
                    before,
                    after,
                    "--statistical",
                    "--paired",
                    "--unpaired",
                ]
            )


# ===========================================================================
# statistical_accept — rank (Wilcoxon) acceptance mode
# ===========================================================================


class TestHodgesLehmann:
    """The location estimator that pairs with the Wilcoxon test."""

    def test_empty(self):
        from panobbgo.harness import _hodges_lehmann

        assert _hodges_lehmann(np.array([])) == 0.0

    def test_single_value(self):
        from panobbgo.harness import _hodges_lehmann

        assert _hodges_lehmann(np.array([0.7])) == pytest.approx(0.7)

    def test_symmetric_sample_equals_the_mean(self):
        from panobbgo.harness import _hodges_lehmann

        assert _hodges_lehmann(np.array([-1.0, 0.0, 1.0])) == pytest.approx(0.0)

    def test_known_walsh_average(self):
        """Walsh averages of [1, 2, 4] (i <= j) are 1,1.5,2.5,2,3,4 → median 2.25."""
        from panobbgo.harness import _hodges_lehmann

        assert _hodges_lehmann(np.array([1.0, 2.0, 4.0])) == pytest.approx(2.25)

    def test_resists_a_single_outlier(self):
        from panobbgo.harness import _hodges_lehmann

        base = [0.01] * 9
        assert _hodges_lehmann(np.array(base)) == pytest.approx(0.01)
        # One wild pair drags the mean by ~0.1 but barely moves HL.
        with_outlier = np.array(base + [1.0])
        assert float(np.mean(with_outlier)) > 0.10
        assert _hodges_lehmann(with_outlier) < 0.10


class TestWilcoxonGreaterGuards:
    """``_wilcoxon_greater`` must never raise on the loop's accept path."""

    def test_all_zero_returns_no_evidence(self):
        from panobbgo.harness import _wilcoxon_greater

        assert _wilcoxon_greater(np.zeros(8), 0.0) == 1.0

    def test_empty_returns_no_evidence(self):
        from panobbgo.harness import _wilcoxon_greater

        assert _wilcoxon_greater(np.array([]), 0.0) == 1.0

    def test_non_finite_returns_no_evidence(self):
        from panobbgo.harness import _wilcoxon_greater

        assert _wilcoxon_greater(np.array([np.nan, np.nan]), 0.0) == 1.0

    def test_uniform_positive_shift_is_significant(self):
        from panobbgo.harness import _wilcoxon_greater

        assert _wilcoxon_greater(np.full(8, 0.02), 0.005) < 0.05

    def test_zeros_are_split_not_dropped(self):
        """``zero_method='zsplit'`` keeps inert pairs in the sample.

        With ``'wilcox'`` the zeros would be discarded and two positive
        pairs alone would read as near-significant — the exact failure
        mode of a mutation that moves only a couple of pairs.
        """
        from panobbgo.harness import _wilcoxon_greater

        d = np.array([0.02, 0.02] + [0.0] * 8)
        assert _wilcoxon_greater(d, 0.0) > 0.05


class TestStatisticalAcceptRankMode:
    """``accept_stat="rank"`` — GOAL.md §5.3.

    Mean-AOCC deltas are outlier-sensitive: one pair that happens to
    solve can carry the composite past ``eps_accept`` on its own.  The
    rank rule asks whether the change wins *typically* across pairs.
    """

    B = 1000  # fine-grained budget: solve-fraction resolution is 1/budget

    def _pairs(self, first_hits_before, first_hits_after):
        before = _harness_result([_psr(f"P{i}", "S", [h] * 3, budget=self.B) for i, h in enumerate(first_hits_before)])
        after = _harness_result([_psr(f"P{i}", "S", [h] * 3, budget=self.B) for i, h in enumerate(first_hits_after)])
        return before, after

    def test_default_is_mean_and_reports_no_rank_fields(self):
        before, after = self._pairs([500] * 4, [480] * 4)
        d = statistical_accept(before, after, n_boot=200, seed=0)
        assert d.accept_stat == "mean"
        assert d.rank_p is None
        assert d.rank_delta is None

    def test_rejects_an_unknown_statistic(self):
        before, after = self._pairs([500], [480])
        with pytest.raises(ValueError, match="accept_stat"):
            statistical_accept(before, after, n_boot=100, seed=0, accept_stat="median")

    def test_outlier_driven_mean_accept_is_rejected_by_rank(self):
        """The headline property.

        Seven pairs each drift −0.005 and one pair goes 0 → 1.0.  The
        mean composite is +0.12 with a degenerate (zero-variance) CI, so
        the mean rule accepts on the strength of a single pair.  The
        rank rule sees 7 losses against 1 win and refuses.
        """
        before, after = self._pairs([500] * 7 + [None], [505] * 7 + [1])

        mean_d = statistical_accept(before, after, eps_accept=0.005, n_boot=500, seed=0)
        assert mean_d.accept is True
        assert mean_d.delta > 0.10
        assert mean_d.ci_low > 0.0

        rank_d = statistical_accept(before, after, eps_accept=0.005, n_boot=500, seed=0, accept_stat="rank")
        assert rank_d.accept is False
        assert rank_d.rank_p is not None and rank_d.rank_p > 0.5
        # The mean is still reported unchanged — ledger continuity.
        assert rank_d.delta == pytest.approx(mean_d.delta)
        # ...but the rank location estimate is negative, as it should be.
        assert rank_d.rank_delta is not None and rank_d.rank_delta < 0.0
        assert any("Wilcoxon" in r for r in rank_d.reasons)

    def test_consistent_small_win_is_accepted_by_both(self):
        """Rank is not merely stricter — a real, consistent lift passes."""
        before, after = self._pairs([500] * 8, [480] * 8)

        for stat in ("mean", "rank"):
            d = statistical_accept(before, after, eps_accept=0.005, n_boot=500, seed=0, accept_stat=stat)
            assert d.accept is True, stat
        rank_d = statistical_accept(before, after, eps_accept=0.005, n_boot=500, seed=0, accept_stat="rank")
        assert rank_d.rank_p is not None and rank_d.rank_p < 0.05
        assert rank_d.rank_delta == pytest.approx(0.020, abs=1e-6)

    def test_rank_can_accept_where_the_mean_cannot(self):
        """A broad win dragged negative by one catastrophic pair.

        Eleven pairs gain +0.020 and one loses −0.300.  The mean is
        −0.0067 so the mean rule rejects; the rank rule sees 11 wins
        against 1 loss and accepts.  ``eps_regress`` is widened here on
        purpose — the per-pair regression guard is orthogonal to the
        statistic and would otherwise mask what is under test.
        """
        before, after = self._pairs([500] * 12, [480] * 11 + [800])

        mean_d = statistical_accept(before, after, eps_accept=0.005, eps_regress=0.5, n_boot=500, seed=0)
        assert mean_d.accept is False
        assert mean_d.delta < 0.0

        rank_d = statistical_accept(
            before, after, eps_accept=0.005, eps_regress=0.5, n_boot=500, seed=0, accept_stat="rank"
        )
        assert rank_d.accept is True
        assert rank_d.rank_p is not None and rank_d.rank_p < 0.05

    def test_regression_guard_still_applies_under_rank(self):
        """The rank test replaces the delta/CI conditions, not the guard."""
        before, after = self._pairs([500] * 12, [480] * 11 + [800])
        d = statistical_accept(
            before, after, eps_accept=0.005, eps_regress=0.05, n_boot=500, seed=0, accept_stat="rank"
        )
        assert d.accept is False
        assert any("regressed" in r for r in d.reasons)

    def test_no_op_change_is_not_significant(self):
        before, after = self._pairs([500] * 8, [500] * 8)
        d = statistical_accept(before, after, eps_accept=0.005, n_boot=200, seed=0, accept_stat="rank")
        assert d.accept is False
        assert d.rank_p == 1.0

    def test_bootstrap_ci_is_still_reported_under_rank(self):
        """The CI does not gate under rank, but it stays in the record."""
        before, after = self._pairs([500] * 8, [480] * 8)
        d = statistical_accept(before, after, eps_accept=0.005, n_boot=500, seed=0, accept_stat="rank")
        assert d.ci_low == pytest.approx(d.delta, abs=1e-6)
        assert d.n_boot == 500

    def test_serialises_the_rank_fields(self):
        before, after = self._pairs([500] * 8, [480] * 8)
        d = statistical_accept(before, after, eps_accept=0.005, n_boot=200, seed=0, accept_stat="rank")
        blob = d.to_dict()
        assert blob["accept_stat"] == "rank"
        assert blob["rank_p"] == pytest.approx(d.rank_p)
        assert blob["rank_delta"] == pytest.approx(d.rank_delta)


class TestRankModeNeedsEnoughPairs:
    """The rank rule has a hard floor on how few pairs it can work with.

    The smallest attainable one-sided Wilcoxon p-value on ``n``
    observations is ``2**-n`` (every observation positive), so with
    ``confidence=0.95`` a rank accept is *impossible* below ``n = 5``
    however large the effect.  That is a property of the test, not a
    bug, but it constrains where the mode may be switched on: the AOCC
    quick battery yields 3 instances x n_specs pairs (6 with two specs,
    12 once the d5 slice is on), which clears the floor.  A future
    caller running a narrower battery must not silently get a rule that
    can never fire.
    """

    B = 1000

    def _pairs(self, n, before_hit=500, after_hit=1):
        before = _harness_result([_psr(f"P{i}", "S", [before_hit] * 3, budget=self.B) for i in range(n)])
        after = _harness_result([_psr(f"P{i}", "S", [after_hit] * 3, budget=self.B) for i in range(n)])
        return before, after

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_below_five_pairs_cannot_accept_even_on_a_huge_effect(self, n):
        before, after = self._pairs(n)
        d = statistical_accept(before, after, eps_accept=0.005, n_boot=200, seed=0, accept_stat="rank")
        assert d.delta > 0.4, "the underlying effect really is enormous"
        assert d.accept is False
        assert d.rank_p is not None and d.rank_p >= 0.05

    @pytest.mark.parametrize("n", [5, 6, 12])
    def test_five_or_more_pairs_can_accept(self, n):
        before, after = self._pairs(n)
        d = statistical_accept(before, after, eps_accept=0.005, n_boot=200, seed=0, accept_stat="rank")
        assert d.accept is True
        assert d.rank_p is not None and d.rank_p < 0.05

    def test_the_mean_rule_has_no_such_floor(self):
        """Contrast: the bootstrap rule accepts a huge effect on 1 pair."""
        before, after = self._pairs(1)
        d = statistical_accept(before, after, eps_accept=0.005, n_boot=200, seed=0)
        assert d.accept is True
