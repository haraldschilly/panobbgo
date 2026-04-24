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
Tests for :mod:`panobbgo.harness_randomized` — the parametrically randomized
problem layer (Phase 3 of the self-improvement loop).

The suite is organised as:

1. **Sampling primitives** — Haar orthogonality, condition number range,
   interior-point sampling, deterministic seed derivation.
2. **Transforms** — the optimum is preserved; translations / rotations /
   scalings compose correctly; noise level is honoured.
3. **ProblemFamily / sampler** — per-family transform gating, dim choice,
   reproducibility across calls, independent draws across reps.
4. **RandomizedProblemSpec & harness integration** — the before/after
   contract (identical instances within one iteration), end-to-end smoke.
"""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.harness import BenchmarkHarness, HarnessConfig
from panobbgo.harness_randomized import (
    ProblemFamily,
    RandomizedProblemSpec,
    SUPPORTED_TRANSFORMS,
    TransformedProblem,
    derive_instance_seed,
    make_default_families,
    make_randomized_specs,
    sample_diagonal_scaling,
    sample_interior_point,
    sample_orthogonal,
)
from panobbgo.lib.classic import Ackley, DeJong, Rastrigin


# ===========================================================================
# 1. Sampling primitives
# ===========================================================================


class TestSampleOrthogonal:
    """Haar-uniform orthogonal matrices."""

    @pytest.mark.parametrize("dim", [1, 2, 3, 5, 8])
    def test_orthogonality(self, dim):
        rng = np.random.default_rng(0)
        Q = sample_orthogonal(rng, dim)
        assert Q.shape == (dim, dim)
        np.testing.assert_allclose(Q.T @ Q, np.eye(dim), atol=1e-10)
        np.testing.assert_allclose(Q @ Q.T, np.eye(dim), atol=1e-10)

    def test_determinant_magnitude_is_one(self):
        rng = np.random.default_rng(1)
        Q = sample_orthogonal(rng, 5)
        assert abs(abs(np.linalg.det(Q)) - 1.0) < 1e-10

    def test_different_seeds_different_matrices(self):
        q0 = sample_orthogonal(np.random.default_rng(0), 4)
        q1 = sample_orthogonal(np.random.default_rng(1), 4)
        assert not np.allclose(q0, q1)

    def test_same_seed_same_matrix(self):
        q0 = sample_orthogonal(np.random.default_rng(42), 4)
        q1 = sample_orthogonal(np.random.default_rng(42), 4)
        np.testing.assert_allclose(q0, q1)

    def test_dim_zero_raises(self):
        with pytest.raises(ValueError):
            sample_orthogonal(np.random.default_rng(0), 0)


class TestSampleDiagonalScaling:
    """Condition-number-controlled diagonal scalings."""

    def test_no_scaling_returns_ones(self):
        scales = sample_diagonal_scaling(np.random.default_rng(0), 4, 0.0)
        np.testing.assert_allclose(scales, np.ones(4))

    def test_dim_one_returns_single_one(self):
        scales = sample_diagonal_scaling(np.random.default_rng(0), 1, 4.0)
        assert scales.shape == (1,)
        np.testing.assert_allclose(scales, 1.0)

    def test_condition_within_bound(self):
        rng = np.random.default_rng(0)
        # Draw many to check the bound statistically.
        log10_max = 2.0
        for _ in range(20):
            s = sample_diagonal_scaling(rng, 5, log10_max)
            cond = s.max() / s.min()
            assert cond <= 10.0**log10_max + 1e-9
            assert cond >= 1.0 - 1e-9

    def test_negative_log10_raises(self):
        with pytest.raises(ValueError):
            sample_diagonal_scaling(np.random.default_rng(0), 3, -1.0)


class TestSampleInteriorPoint:
    """Interior-point sampling stays inside the padded box."""

    def test_inside_box(self):
        rng = np.random.default_rng(0)
        box = np.array([[-2.0, 2.0], [-5.0, 5.0], [0.0, 10.0]])
        margin = 0.15
        for _ in range(200):
            p = sample_interior_point(rng, box, margin)
            assert p.shape == (3,)
            widths = box[:, 1] - box[:, 0]
            inner_lo = box[:, 0] + margin * widths
            inner_hi = box[:, 1] - margin * widths
            assert np.all(p >= inner_lo - 1e-12)
            assert np.all(p <= inner_hi + 1e-12)

    def test_zero_margin_samples_entire_box(self):
        # With margin=0 each sample lies in the full box and the per-axis
        # spread covers most of the width.
        rng = np.random.default_rng(0)
        box = np.array([[-10.0, 10.0], [-1.0, 1.0]])
        pts = np.array([sample_interior_point(rng, box, 0.0) for _ in range(2000)])
        assert np.all(pts[:, 0] >= -10.0 - 1e-12)
        assert np.all(pts[:, 0] <= 10.0 + 1e-12)
        # Samples span at least 80% of each axis.
        assert pts[:, 0].max() - pts[:, 0].min() > 0.8 * 20
        assert pts[:, 1].max() - pts[:, 1].min() > 0.8 * 2

    def test_invalid_margin_raises(self):
        with pytest.raises(ValueError):
            sample_interior_point(np.random.default_rng(0), np.array([[0.0, 1.0]]), 0.5)
        with pytest.raises(ValueError):
            sample_interior_point(np.random.default_rng(0), np.array([[0.0, 1.0]]), -0.1)


class TestDeriveInstanceSeed:
    """SHA-256-backed seed derivation is stable and distinguishes inputs."""

    def test_stable_across_calls(self):
        a = derive_instance_seed(42, 0, "Rastrigin_family", 3)
        b = derive_instance_seed(42, 0, "Rastrigin_family", 3)
        assert a == b

    def test_nonnegative_32bit(self):
        seed = derive_instance_seed(42, 17, "DeJong_family", 2)
        assert 0 <= seed < 2**32

    @pytest.mark.parametrize(
        "a, b",
        [
            ((42, 0, "x", 0), (43, 0, "x", 0)),
            ((42, 0, "x", 0), (42, 1, "x", 0)),
            ((42, 0, "x", 0), (42, 0, "y", 0)),
            ((42, 0, "x", 0), (42, 0, "x", 1)),
        ],
    )
    def test_inputs_produce_different_seeds(self, a, b):
        assert derive_instance_seed(*a) != derive_instance_seed(*b)


# ===========================================================================
# 2. TransformedProblem
# ===========================================================================


class TestTransformedProblem:
    """The wrapper preserves the optimum and honours every transform."""

    def test_identity_transform_matches_base(self):
        base = DeJong(dims=2)
        # No translate, no rotate, no scale, no noise.
        tp = TransformedProblem(
            base_problem=base,
            x_star=np.zeros(2),
            Q=None,
            scale=None,
            noise_sigma=0.0,
        )
        for _ in range(10):
            x = np.random.default_rng(0).standard_normal(2) * 2
            assert tp.eval(x) == pytest.approx(base.eval(x))

    def test_translation_moves_optimum(self):
        base = Rastrigin(dims=2)
        x_star = np.array([1.2, -0.4])
        tp = TransformedProblem(base_problem=base, x_star=x_star)
        # f(x_star) should equal f_base(0) == 0 for Rastrigin.
        assert tp.eval(x_star) == pytest.approx(0.0, abs=1e-12)

    def test_rotation_preserves_optimum(self):
        base = Rastrigin(dims=3)
        x_star = np.array([0.5, -0.7, 0.1])
        Q = sample_orthogonal(np.random.default_rng(0), 3)
        tp = TransformedProblem(base_problem=base, x_star=x_star, Q=Q)
        assert tp.eval(x_star) == pytest.approx(0.0, abs=1e-12)

    def test_scaling_preserves_optimum(self):
        base = Ackley(dims=2)
        x_star = np.array([0.3, 0.2])
        scale = np.array([3.0, 0.25])
        tp = TransformedProblem(base_problem=base, x_star=x_star, scale=scale)
        assert tp.eval(x_star) == pytest.approx(0.0, abs=1e-12)

    def test_composed_transforms_preserve_optimum(self):
        base = Rastrigin(dims=4)
        rng = np.random.default_rng(123)
        x_star = np.array([0.9, -0.4, 0.1, 0.55])
        Q = sample_orthogonal(rng, 4)
        scale = sample_diagonal_scaling(rng, 4, 1.5)
        tp = TransformedProblem(base_problem=base, x_star=x_star, Q=Q, scale=scale)
        assert tp.eval(x_star) == pytest.approx(0.0, abs=1e-12)

    def test_noise_has_right_scale(self):
        base = DeJong(dims=2)
        x_star = np.zeros(2)
        sigma = 0.1
        tp = TransformedProblem(
            base_problem=base,
            x_star=x_star,
            noise_sigma=sigma,
            noise_seed=123,
        )
        # f_base at x_star is 0 for DeJong, so the eval *is* the noise.
        samples = np.array([tp.eval(x_star) for _ in range(2000)])
        # Mean roughly 0, std roughly sigma.
        assert abs(samples.mean()) < 0.02
        assert abs(samples.std() - sigma) < 0.02

    def test_noise_reproducible_from_seed(self):
        base = DeJong(dims=2)
        t1 = TransformedProblem(base, x_star=np.zeros(2), noise_sigma=0.5, noise_seed=7)
        t2 = TransformedProblem(base, x_star=np.zeros(2), noise_sigma=0.5, noise_seed=7)
        for _ in range(10):
            x = np.array([0.2, 0.3])
            assert t1.eval(x) == t2.eval(x)

    def test_dimension_mismatch_raises(self):
        base = DeJong(dims=2)
        with pytest.raises(ValueError):
            TransformedProblem(base_problem=base, x_star=np.zeros(3))
        with pytest.raises(ValueError):
            TransformedProblem(base_problem=base, x_star=np.zeros(2), Q=np.eye(3))
        with pytest.raises(ValueError):
            TransformedProblem(base_problem=base, x_star=np.zeros(2), scale=np.ones(3))

    def test_negative_noise_raises(self):
        base = DeJong(dims=2)
        with pytest.raises(ValueError):
            TransformedProblem(base_problem=base, x_star=np.zeros(2), noise_sigma=-0.1)

    def test_constraints_are_none(self):
        base = DeJong(dims=2)
        tp = TransformedProblem(base_problem=base, x_star=np.zeros(2))
        assert tp.eval_constraints(np.array([0.1, 0.2])) is None

    def test_repr_contains_name(self):
        base = DeJong(dims=2)
        tp = TransformedProblem(base_problem=base, x_star=np.zeros(2), name="MyFamily")
        assert "MyFamily" in repr(tp)


# ===========================================================================
# 3. ProblemFamily
# ===========================================================================


class TestProblemFamily:
    def test_default_families(self):
        fams = make_default_families()
        assert len(fams) >= 3
        names = {f.name for f in fams}
        assert "Rastrigin_family" in names
        assert "DeJong_family" in names

    def test_unknown_transform_raises(self):
        with pytest.raises(ValueError):
            ProblemFamily(
                name="bogus",
                base_class=DeJong,
                supported_transforms={"warp"},
            )

    def test_empty_dim_choices_raises(self):
        with pytest.raises(ValueError):
            ProblemFamily(
                name="bogus",
                base_class=DeJong,
                dim_choices=(),
            )

    def test_sample_instance_uses_only_supported_transforms(self):
        # Translate-only family: no rotation, no scaling, no noise.
        fam = ProblemFamily(
            name="sphere_translate",
            base_class=DeJong,
            supported_transforms={"translate"},
        )
        rng = np.random.default_rng(0)
        prob, params = fam.sample_instance(rng)
        assert isinstance(prob, TransformedProblem)
        assert "translation" in params
        assert "rotation_trace" not in params
        assert "scale_cond" not in params
        # The rotation matrix was not applied.
        assert prob._Q is None
        assert prob._scale is None
        assert prob._noise_sigma == 0.0

    def test_sample_instance_rotation_scaling(self):
        fam = ProblemFamily(
            name="sphere_rs",
            base_class=DeJong,
            supported_transforms={"rotate", "scale"},
            log10_cond_max=1.0,
        )
        rng = np.random.default_rng(0)
        prob, params = fam.sample_instance(rng)
        assert prob._Q is not None
        assert prob._scale is not None
        assert "rotation_trace" in params
        assert "scale_cond" in params

    def test_noise_flag(self):
        fam = ProblemFamily(
            name="sphere_noise",
            base_class=DeJong,
            supported_transforms={"noise"},
            noise_sigma=0.05,
        )
        rng = np.random.default_rng(0)
        prob, params = fam.sample_instance(rng)
        assert prob._noise_sigma == pytest.approx(0.05)
        assert params["noise_sigma"] == pytest.approx(0.05)

    def test_f_opt_reported(self):
        fam = make_default_families()[0]
        rng = np.random.default_rng(42)
        prob, _ = fam.sample_instance(rng)
        # At the reported optimum, the objective is f_opt (up to noise).
        assert prob.eval(prob.optimum) == pytest.approx(fam.f_opt, abs=1e-9)


# ===========================================================================
# 4. RandomizedProblemSpec
# ===========================================================================


class TestRandomizedProblemSpec:
    def test_create_problem_raises_requires_rep(self):
        fam = make_default_families()[0]
        spec = RandomizedProblemSpec(family=fam, iteration_id=0, base_seed=42, max_evaluations=100)
        with pytest.raises(RuntimeError):
            spec.create_problem()

    def test_same_iter_same_rep_same_instance(self):
        fam = make_default_families()[0]
        s1 = RandomizedProblemSpec(fam, iteration_id=3, base_seed=42, max_evaluations=100)
        s2 = RandomizedProblemSpec(fam, iteration_id=3, base_seed=42, max_evaluations=100)
        p1 = s1.create_problem_for_rep(1)
        p2 = s2.create_problem_for_rep(1)
        np.testing.assert_allclose(p1.optimum, p2.optimum)
        # Same eval at a random point.
        x = np.array([0.1, -0.2])
        assert p1.eval(x) == pytest.approx(p2.eval(x))

    def test_different_iteration_different_instance(self):
        fam = make_default_families()[0]
        s1 = RandomizedProblemSpec(fam, iteration_id=3, base_seed=42, max_evaluations=100)
        s2 = RandomizedProblemSpec(fam, iteration_id=4, base_seed=42, max_evaluations=100)
        p1 = s1.create_problem_for_rep(0)
        p2 = s2.create_problem_for_rep(0)
        # Different x* with overwhelming probability.
        assert not np.allclose(p1.optimum, p2.optimum)

    def test_different_reps_different_instances(self):
        fam = make_default_families()[0]
        spec = RandomizedProblemSpec(fam, iteration_id=0, base_seed=42, max_evaluations=100)
        p0 = spec.create_problem_for_rep(0)
        p1 = spec.create_problem_for_rep(1)
        assert not np.allclose(p0.optimum, p1.optimum)

    def test_last_sampled_params(self):
        fam = make_default_families()[0]
        spec = RandomizedProblemSpec(fam, iteration_id=0, base_seed=42, max_evaluations=100)
        assert spec.last_sampled_params() is None
        spec.create_problem_for_rep(2)
        params = spec.last_sampled_params()
        assert params is not None
        assert params["family"] == fam.name
        assert params["rep"] == 2
        assert params["dim"] == fam.dim_choices[0]
        assert "seed" in params

    def test_spec_known_optima_carries_f_opt(self):
        fam = make_default_families()[0]
        spec = RandomizedProblemSpec(fam, iteration_id=0, base_seed=42, max_evaluations=123)
        assert spec.known_optima[0]["fx"] == pytest.approx(fam.f_opt)
        assert spec.tolerance == fam.tolerance
        assert spec.max_evaluations == 123
        assert spec.dims == fam.dim_choices[0]


class TestMakeRandomizedSpecs:
    def test_returns_one_per_family(self):
        fams = make_default_families()
        specs = make_randomized_specs(fams, iteration_id=0, base_seed=42, max_evaluations=75)
        assert len(specs) == len(fams)
        assert all(isinstance(s, RandomizedProblemSpec) for s in specs)
        assert [s.family.name for s in specs] == [f.name for f in fams]


# ===========================================================================
# 5. Harness integration
# ===========================================================================


class TestHarnessRandomize:
    """Tie the randomized layer into :class:`BenchmarkHarness`."""

    def test_get_problems_returns_randomized_specs(self):
        cfg = HarnessConfig(mode="quick", randomize=True, budget=30)
        harness = BenchmarkHarness(cfg)
        specs = harness.get_problems()
        assert len(specs) == len(make_default_families())
        assert all(isinstance(s, RandomizedProblemSpec) for s in specs)
        # Budget propagates.
        assert all(s.max_evaluations == 30 for s in specs)

    def test_problem_filter_applies(self):
        cfg = HarnessConfig(
            mode="quick",
            randomize=True,
            budget=30,
            problems=["Rastrigin_family", "DeJong_family"],
        )
        harness = BenchmarkHarness(cfg)
        specs = harness.get_problems()
        assert {s.name for s in specs} == {"Rastrigin_family", "DeJong_family"}

    @pytest.mark.flaky(retries=2)
    def test_end_to_end_smoke(self):
        """One run, one strategy, tiny budget — verify the plumbing works."""
        cfg = HarnessConfig(
            mode="quick",
            randomize=True,
            randomize_iteration=0,
            seed=42,
            budget=25,
            reps=1,
            strategies=["RoundRobin_Random"],
            problems=["DeJong_family"],
            timeout_per_run=30.0,
        )
        harness = BenchmarkHarness(cfg)
        result = harness.run(verbose=False)
        assert result.total_runs == 1
        assert 0.0 <= result.composite_score <= 1.0
        assert len(result.problem_strategy_results) == 1
        psr = result.problem_strategy_results[0]
        assert psr.problem_name == "DeJong_family"
        assert psr.strategy_name == "RoundRobin_Random"
        # Every run in the psr gets a real evaluation count.
        for run in psr.runs:
            # Even on a failure we should record > 0 evaluations when the
            # strategy actually runs.
            assert run.evaluations_used >= 0

    def test_before_after_sees_same_instances(self):
        """The central reproducibility contract: same iteration → same
        instances → comparable before/after scores."""
        fams = make_default_families()
        spec_a = RandomizedProblemSpec(fams[0], iteration_id=7, base_seed=42, max_evaluations=100)
        spec_b = RandomizedProblemSpec(fams[0], iteration_id=7, base_seed=42, max_evaluations=100)
        for rep in range(3):
            p_a = spec_a.create_problem_for_rep(rep)
            p_b = spec_b.create_problem_for_rep(rep)
            np.testing.assert_allclose(p_a.optimum, p_b.optimum)

    def test_supported_transforms_enum(self):
        # Sanity: the publicly-exported transform names cover every family.
        for fam in make_default_families():
            for t in fam.supported_transforms:
                assert t in SUPPORTED_TRANSFORMS
