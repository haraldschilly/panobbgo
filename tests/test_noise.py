# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
"""Tests for :mod:`panobbgo.lib.noise` and the noisy IOH batteries.

Three layers, in increasing cost:

1.  the noise models and the wrapper — pure Python, no worker;
2.  the tracker's true-value path and AOCC-on-the-true-value scoring —
    still no worker, using a local sphere;
3.  a real (tiny) battery run through ``run_ioh_harness``, gated on the
    ``ioh`` worker venv like every other IOH test.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from panobbgo.harness_ioh import (
    NOISY_PROBLEM_KINDS,
    SUPPORTED_PROBLEM_KINDS,
    _derive_noise_seed,
    _derive_seed,
    make_highdim_battery,
    make_ioh_strategies,
    make_noisy_battery,
    make_noisy_highdim_battery,
    make_standard_battery,
    resolve_problem_kind,
    run_ioh_harness,
)
from panobbgo.ioh_runner import IOHTracker, aocc
from panobbgo.lib.ioh_wrapper import worker_available
from panobbgo.lib.lib import Problem
from panobbgo.lib.noise import (
    AdditiveGaussianNoise,
    CauchyNoise,
    GaussianNoise,
    MultiplicativeGaussianNoise,
    NoNoise,
    NoisyProblem,
    UniformNoise,
    make_noise_model,
    rng_for_point,
)


requires_worker = pytest.mark.skipif(
    not worker_available(),
    reason="ioh worker venv not set up (run `cd tools/ioh_worker && uv sync`)",
)


class Sphere(Problem):
    """Local noiseless reference: ``f(x) = |x|^2``, optimum 0 at the origin."""

    def __init__(self, dim: int = 2) -> None:
        super().__init__(box=[(-5.0, 5.0)] * dim)
        self.n_calls = 0

    def eval(self, x: np.ndarray) -> float:
        self.n_calls += 1
        return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))

    @property
    def optimum_y(self) -> float:
        return 0.0


ALL_MODELS = ["gauss", "unif", "cauchy", "add", "mult"]


# ---------------------------------------------------------------------------
# 1. Models and wrapper
# ---------------------------------------------------------------------------


class TestDeterminism:
    @pytest.mark.parametrize("tag", ALL_MODELS)
    def test_noise_is_a_pure_function_of_seed_and_x(self, tag: str) -> None:
        prob = NoisyProblem(Sphere(), make_noise_model(tag, dim=2), seed=11)
        x = np.array([0.5, -0.25])
        values = {prob.eval(x) for _ in range(5)}
        assert len(values) == 1, "re-evaluating the same x must return the same noisy value"

    @pytest.mark.parametrize("tag", ALL_MODELS)
    def test_two_wrappers_same_seed_agree(self, tag: str) -> None:
        a = NoisyProblem(Sphere(), make_noise_model(tag, dim=2), seed=11)
        b = NoisyProblem(Sphere(), make_noise_model(tag, dim=2), seed=11)
        xs = [np.array([0.1 * i, -0.3 * i]) for i in range(1, 6)]
        assert [a.eval(x) for x in xs] == [b.eval(x) for x in xs]

    def test_evaluation_order_does_not_matter(self) -> None:
        a = NoisyProblem(Sphere(), GaussianNoise(beta=0.5), seed=5)
        b = NoisyProblem(Sphere(), GaussianNoise(beta=0.5), seed=5)
        xs = [np.array([0.1 * i, 0.2 * i]) for i in range(6)]
        forward = {i: a.eval(x) for i, x in enumerate(xs)}
        backward = {i: b.eval(x) for i, x in reversed(list(enumerate(xs)))}
        assert forward == backward

    def test_different_seeds_are_different_realisations(self) -> None:
        x = np.array([0.5, -0.25])
        a = NoisyProblem(Sphere(), GaussianNoise(beta=0.5), seed=1).eval(x)
        b = NoisyProblem(Sphere(), GaussianNoise(beta=0.5), seed=2).eval(x)
        assert a != b

    def test_resample_draws_fresh_noise_per_call(self) -> None:
        prob = NoisyProblem(Sphere(), GaussianNoise(beta=0.5), seed=11, resample=True)
        x = np.array([0.5, -0.25])
        values = [prob.eval(x) for _ in range(5)]
        assert len(set(values)) == 5

    def test_resample_is_reproducible_for_the_same_call_sequence(self) -> None:
        x = np.array([0.5, -0.25])

        def seq() -> list[float]:
            p = NoisyProblem(Sphere(), GaussianNoise(beta=0.5), seed=11, resample=True)
            return [p.eval(x) for _ in range(5)]

        assert seq() == seq()

    def test_resample_first_draw_matches_frozen(self) -> None:
        x = np.array([0.5, -0.25])
        frozen = NoisyProblem(Sphere(), GaussianNoise(beta=0.5), seed=11).eval(x)
        resampled = NoisyProblem(Sphere(), GaussianNoise(beta=0.5), seed=11, resample=True).eval(x)
        assert frozen == resampled

    def test_rng_for_point_is_the_documented_hash(self) -> None:
        x = np.array([1.0, 2.0])
        a = rng_for_point(3, x).standard_normal()
        b = rng_for_point(3, x.copy()).standard_normal()
        c = rng_for_point(3, x, draw=1).standard_normal()
        assert a == b and a != c


class TestModels:
    def test_no_noise_is_the_identity(self) -> None:
        prob = NoisyProblem(Sphere(), NoNoise(), seed=1)
        x = np.array([0.3, 0.4])
        assert prob.eval(x) == pytest.approx(prob.true_eval(x))

    def test_gaussian_is_multiplicative_and_positive(self) -> None:
        rng = np.random.default_rng(0)
        model = GaussianNoise(beta=1.0)
        vals = [model.apply(1.0, np.random.default_rng(int(rng.integers(1 << 30)))) for _ in range(400)]
        assert all(v > 0 for v in vals), "log-normal noise cannot make the precision negative"
        # exp(N(0,1)) has median 1, so ~half the draws sit on either side.
        assert 0.3 < sum(v > 1.0 for v in vals) / len(vals) < 0.7

    def test_gaussian_beta_zero_is_noise_free_up_to_the_bbob_floor(self) -> None:
        assert GaussianNoise(beta=0.0).apply(3.0, np.random.default_rng(0)) == pytest.approx(3.0, abs=1e-7)

    def test_uniform_only_blows_up_below_1e9(self) -> None:
        model = UniformNoise(alpha=1.0, beta=0.0)
        # f > 1e9 -> max(1, (1e9/f)^x) == 1, so the value is untouched.
        assert model.apply(1e10, np.random.default_rng(0)) == pytest.approx(1e10, rel=1e-6)
        # deep in the endgame the inflation factor is enormous
        assert model.apply(1e-6, np.random.default_rng(0)) > 1e-6

    def test_cauchy_shifts_by_1000_alpha_and_is_heavy_tailed(self) -> None:
        model = CauchyNoise(alpha=0.01, p=0.05)
        rng = np.random.default_rng(0)
        vals = [model.apply(0.0, np.random.default_rng(int(rng.integers(1 << 30)))) for _ in range(2000)]
        # the modal value is exactly alpha * 1000
        assert sum(abs(v - 10.0) < 1e-6 for v in vals) / len(vals) > 0.9
        # ...and the tail is fat: some draws are far away
        assert max(vals) > 11.0

    def test_additive_absolute_vs_relative(self) -> None:
        rng_a, rng_b = np.random.default_rng(4), np.random.default_rng(4)
        absolute = AdditiveGaussianNoise(sigma=0.1).apply(2.0, rng_a) - 2.0
        relative = AdditiveGaussianNoise(sigma=0.1, f_range=10.0).apply(2.0, rng_b) - 2.0
        assert relative == pytest.approx(10.0 * absolute)

    def test_multiplicative_scales_with_f(self) -> None:
        rng_a, rng_b = np.random.default_rng(9), np.random.default_rng(9)
        a = MultiplicativeGaussianNoise(sigma=0.2).apply(1.0, rng_a) - 1.0
        b = MultiplicativeGaussianNoise(sigma=0.2).apply(100.0, rng_b) - 100.0
        assert b == pytest.approx(100.0 * a)

    def test_uniform_alpha_scales_with_dimension(self) -> None:
        m2 = make_noise_model("unif", dim=2)
        m10 = make_noise_model("unif", dim=10)
        assert isinstance(m2, UniformNoise) and isinstance(m10, UniformNoise)
        assert m2.alpha == pytest.approx(0.01 * (0.49 + 1 / 2))
        assert m10.alpha == pytest.approx(0.01 * (0.49 + 1 / 10))

    def test_severe_is_stronger_than_moderate(self) -> None:
        moderate = make_noise_model("gauss", dim=2, level="moderate")
        severe = make_noise_model("gauss", dim=2, level="severe")
        assert isinstance(moderate, GaussianNoise) and isinstance(severe, GaussianNoise)
        assert severe.beta > moderate.beta

    def test_unknown_kind_and_level_raise(self) -> None:
        with pytest.raises(ValueError):
            make_noise_model("nope", dim=2)
        with pytest.raises(ValueError):
            make_noise_model("gauss", dim=2, level="apocalyptic")


class TestWrapper:
    def test_true_value_is_exposed(self) -> None:
        inner = Sphere()
        prob = NoisyProblem(inner, GaussianNoise(beta=1.0), seed=2)
        x = np.array([1.0, 2.0])
        assert prob.true_eval(x) == pytest.approx(5.0)
        noisy, true_fx = prob.eval_pair(x)
        assert true_fx == pytest.approx(5.0)
        assert noisy != true_fx

    def test_eval_pair_costs_one_inner_evaluation(self) -> None:
        inner = Sphere()
        prob = NoisyProblem(inner, GaussianNoise(), seed=2)
        inner.n_calls = 0
        prob.eval_pair(np.array([1.0, 1.0]))
        assert inner.n_calls == 1

    def test_box_and_dim_are_inherited(self) -> None:
        inner = Sphere(dim=3)
        prob = NoisyProblem(inner, NoNoise(), seed=0)
        assert prob.dim == 3
        assert np.allclose(np.asarray(prob.box.box), np.asarray(inner.box.box))
        assert np.allclose(prob.project(np.array([9.0, -9.0, 0.0])), [5.0, -5.0, 0.0])

    def test_optimum_and_attribute_passthrough(self) -> None:
        inner = Sphere()
        inner.ioh_name = "sphere-ish"  # type: ignore[attr-defined]
        prob = NoisyProblem(inner, NoNoise(), seed=0)
        assert prob.optimum_y == 0.0
        assert prob.ioh_name == "sphere-ish"

    def test_noise_acts_on_the_precision_not_the_raw_value(self) -> None:
        """A multiplicative model must not scale the ``f_opt`` offset."""
        inner = Sphere()
        prob = NoisyProblem(inner, MultiplicativeGaussianNoise(sigma=0.5), seed=3, f_opt=-100.0)
        x = np.array([0.0, 0.0])  # true f = 0, precision = 100
        noisy = prob.eval(x)
        # noisy = -100 + 100*(1+0.5*eps)  ->  |noisy| <= a few hundred, never ~0.
        assert noisy == pytest.approx(-100.0 + 100.0 * (1.0 + 0.5 * rng_for_point(3, x).standard_normal()))

    def test_call_protocol_returns_the_noisy_result(self) -> None:
        from panobbgo.lib.lib import Point

        prob = NoisyProblem(Sphere(), GaussianNoise(beta=1.0), seed=4)
        x = np.array([1.0, 1.0])
        res = prob(Point(x, "test"))
        assert res.fx == pytest.approx(prob.eval(x))
        assert res.cv_vec is None


# ---------------------------------------------------------------------------
# 2. Tracker: AOCC is computed on the true value
# ---------------------------------------------------------------------------


class TestTrackerTruePath:
    def _run(self, model, seed: int = 1, n: int = 60):
        prob = NoisyProblem(Sphere(), model, seed=seed)
        tracker = IOHTracker(prob, budget=n)
        rng = np.random.default_rng(0)
        for _ in range(n):
            prob.eval(rng.uniform(-5.0, 5.0, 2))
        traj = tracker.trajectory()
        tracker.restore()
        return tracker, traj

    def test_noiseless_problem_records_no_true_trace(self) -> None:
        prob = Sphere()
        tracker = IOHTracker(prob, budget=5)
        for i in range(5):
            prob.eval(np.array([float(i), 0.0]))
        traj = tracker.trajectory()
        tracker.restore()
        assert tracker.has_true is False
        assert traj.best_so_far_true is None
        # aocc_true falls back to the observed trace, so old callers are safe.
        assert traj.aocc_true() == traj.aocc()

    def test_three_traces_are_recorded_and_have_the_right_shape(self) -> None:
        tracker, traj = self._run(GaussianNoise(beta=1.0))
        assert tracker.has_true is True
        assert traj.best_so_far_true is not None and traj.best_so_far_reco is not None
        assert len(traj.best_so_far) == len(traj.best_so_far_true) == len(traj.best_so_far_reco) == 60
        # the true best-so-far is monotone; the recommendation need not be
        assert all(a >= b for a, b in zip(traj.best_so_far_true, traj.best_so_far_true[1:]))

    def test_true_trace_is_the_noiseless_sphere_minimum(self) -> None:
        """Scoring is on the *true* value: no noise realisation can change it."""
        _, a = self._run(GaussianNoise(beta=0.01), seed=1)
        _, b = self._run(GaussianNoise(beta=2.0), seed=2)
        # Both runs evaluate the identical point sequence (same rng), so the
        # set of visited points -- hence the true best-so-far -- is identical
        # even though the observed values differ wildly.
        assert a.best_so_far_true == b.best_so_far_true
        assert a.best_so_far != b.best_so_far
        assert a.aocc_true() == b.aocc_true()

    def test_observed_aocc_is_optimistically_biased(self) -> None:
        """The min of many noisy readings sits below the min of their means."""
        _, traj = self._run(GaussianNoise(beta=2.0), seed=7, n=200)
        assert traj.aocc() > traj.aocc_true()

    def test_recommendation_never_beats_the_true_best(self) -> None:
        _, traj = self._run(GaussianNoise(beta=2.0), seed=7, n=200)
        assert traj.best_so_far_reco is not None and traj.best_so_far_true is not None
        assert all(r >= t - 1e-12 for r, t in zip(traj.best_so_far_reco, traj.best_so_far_true))
        assert aocc(traj.best_so_far_reco) <= traj.aocc_true() + 1e-12

    def test_best_true_fx_reported(self) -> None:
        _, traj = self._run(GaussianNoise(beta=1.0))
        assert traj.best_so_far_true is not None
        assert traj.best_true_fx == pytest.approx(traj.best_so_far_true[-1])
        assert not math.isnan(traj.best_true_fx)


# ---------------------------------------------------------------------------
# 3. Battery shape and seed derivation (no worker)
# ---------------------------------------------------------------------------


class TestNoisyBatteries:
    def test_kinds_are_registered(self) -> None:
        for kind in NOISY_PROBLEM_KINDS:
            assert kind in SUPPORTED_PROBLEM_KINDS
            worker_kind, tag = resolve_problem_kind(kind)
            assert worker_kind == "MA-BBOB" and tag in ("gauss", "unif", "cauchy")
        assert resolve_problem_kind("MA-BBOB") == ("MA-BBOB", None)

    @pytest.mark.parametrize("noise", ["gauss", "unif", "cauchy"])
    def test_noisy_battery_matches_the_standard_cube(self, noise: str) -> None:
        std, b = make_standard_battery(), make_noisy_battery(noise)
        assert b.is_noisy and b.problem_kind == f"MA-BBOB-noisy-{noise}"
        assert (b.dims, b.instances, b.budget_multiplier) == (std.dims, std.instances, std.budget_multiplier)
        assert b.noise_level == "moderate" and b.noise_resample is False
        assert b.name != std.name, "a noisy result must not be mistakable for a §2 one"

    def test_severe_level_is_reachable_and_named(self) -> None:
        b = make_noisy_battery("gauss", level="severe")
        assert b.noise_level == "severe" and "severe" in b.name

    def test_highdim_battery(self) -> None:
        b = make_highdim_battery()
        assert b.dims == (10, 20) and b.instances == (0, 1, 2)
        assert b.budget_for(10) == 20000 and b.budget_for(20) == 40000
        assert not b.is_noisy

    def test_noisy_highdim_battery(self) -> None:
        b = make_noisy_highdim_battery()
        assert b.dims == (10,) and b.is_noisy and b.budget_for(10) == 5000

    def test_cli_flags_select_the_presets(self) -> None:
        """``scripts/ioh_benchmark.py run --noisy/--highdim`` resolve to the presets."""
        import argparse
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "scripts" / "ioh_benchmark.py"
        spec = importlib.util.spec_from_file_location("ioh_benchmark_cli", path)
        assert spec is not None and spec.loader is not None
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)

        def resolve(**kw):
            ns = argparse.Namespace(
                full=False, standard=False, noisy=None, noisy_highdim=None, highdim=False, noisy_severe=False, reps=None
            )
            for k, v in kw.items():
                setattr(ns, k, v)
            return cli._resolve_battery(ns)

        assert resolve(noisy="gauss").problem_kind == "MA-BBOB-noisy-gauss"
        assert resolve(noisy="cauchy", noisy_severe=True).noise_level == "severe"
        assert resolve(highdim=True).dims == (10, 20)
        assert resolve(noisy_highdim="unif").problem_kind == "MA-BBOB-noisy-unif"
        assert resolve(noisy_highdim="unif").dims == (10,)
        # the pre-existing flags keep their meaning
        assert resolve().name == "ioh-quick"
        assert resolve(standard=True).name == "ioh-standard"

    def test_unknown_noise_model_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_noisy_battery("laplace")
        with pytest.raises(ValueError):
            make_noisy_highdim_battery("laplace")

    def test_noiseless_seeds_are_unchanged(self) -> None:
        """The frozen contract: adding noise must not move an existing battery."""
        assert _derive_seed(42, "MA-BBOB", 2, 1, "RoundRobin_CMAES", 0) == _derive_seed(
            42, "MA-BBOB", 2, 1, "RoundRobin_CMAES", 0, None
        )

    def test_noise_seed_enters_the_run_seed(self) -> None:
        base = _derive_seed(42, "MA-BBOB-noisy-gauss", 2, 1, "s", 0)
        assert _derive_seed(42, "MA-BBOB-noisy-gauss", 2, 1, "s", 0, 777) != base

    def test_noise_seed_differs_per_instance_and_rep_but_not_per_strategy(self) -> None:
        args = (42, "MA-BBOB-noisy-gauss", 2)
        seeds = {_derive_noise_seed(*args, inst, 0) for inst in range(5)}
        assert len(seeds) == 5, "each instance is a different noise realisation"
        assert _derive_noise_seed(*args, 0, 0) != _derive_noise_seed(*args, 0, 1)
        # ...and there is no strategy argument at all: every arm on a cell
        # faces the identical noisy function, so the comparison stays paired.
        assert _derive_noise_seed(*args, 0, 0) == _derive_noise_seed(*args, 0, 0)


# ---------------------------------------------------------------------------
# 4. End to end through the real worker
# ---------------------------------------------------------------------------


@requires_worker
class TestNoisyHarnessRun:
    @staticmethod
    def _tiny(battery):
        return dataclasses.replace(battery, dims=(2,), instances=(0,), budget_multiplier=40)

    @pytest.mark.parametrize("noise", ["gauss", "unif", "cauchy"])
    def test_one_cell_completes_and_spends_its_budget(self, noise: str) -> None:
        specs = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"]
        battery = self._tiny(make_noisy_battery(noise))
        res = run_ioh_harness(specs, battery, base_seed=42, progress=False, sync_eval=True)
        assert len(res.runs) == 1
        run = res.runs[0]
        assert run.error is None
        assert run.n_evals == run.budget == 80
        assert run.noise_seed is not None
        assert run.aocc_observed is not None and run.aocc_reco is not None
        assert 0.0 <= run.aocc <= 1.0

    def test_noisy_run_is_reproducible(self) -> None:
        specs = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"]
        battery = self._tiny(make_noisy_battery("gauss"))

        def once():
            res = run_ioh_harness(specs, battery, base_seed=42, progress=False, sync_eval=True)
            return [(r.aocc, r.aocc_observed, r.n_evals, tuple(r.trace_fx)) for r in res.runs]

        assert once() == once()

    def test_noise_actually_changes_the_run(self) -> None:
        specs = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"]
        noisy = run_ioh_harness(
            specs, self._tiny(make_noisy_battery("gauss", level="severe")), base_seed=42, progress=False, sync_eval=True
        )
        clean = run_ioh_harness(
            specs, self._tiny(make_standard_battery()), base_seed=42, progress=False, sync_eval=True
        )
        assert noisy.runs[0].aocc != clean.runs[0].aocc

    def test_run_record_round_trips_through_json(self) -> None:
        from panobbgo.harness_ioh import IOHHarnessResult

        specs = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"]
        res = run_ioh_harness(
            specs, self._tiny(make_noisy_battery("gauss")), base_seed=42, progress=False, sync_eval=True
        )
        back = IOHHarnessResult.from_dict(res.to_dict())
        assert back.runs[0].aocc == res.runs[0].aocc
        assert back.runs[0].noise_seed == res.runs[0].noise_seed
        assert back.runs[0].aocc_observed == res.runs[0].aocc_observed

    def test_highdim_battery_runs_one_tiny_cell(self) -> None:
        specs = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"]
        battery = dataclasses.replace(make_highdim_battery(), dims=(10,), instances=(0,), budget_multiplier=20)
        res = run_ioh_harness(specs, battery, base_seed=42, progress=False, sync_eval=True)
        assert res.runs[0].error is None
        assert res.runs[0].n_evals == res.runs[0].budget == 200
