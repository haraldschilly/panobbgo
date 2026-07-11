# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
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
Tests for the curvature-aware quadratic step of the Nearby heuristic.

Two layers:

* ``fit_quadratic_step`` — the pure model-fitting function. Verified against
  known quadratics (it must recover the exact minimiser), against
  ill-conditioned / coupled (Rosenbrock-like) landscapes, and for its
  robustness guards (too few points, non-finite values, trust-region clipping).
* ``Nearby(quadratic=True)`` — the wiring: buffer accumulation, the
  curvature-aware first point, and byte-identical fallback when disabled.
"""

import numpy as np
import pytest
from unittest import mock

from panobbgo.heuristics.nearby import Nearby, fit_quadratic_step
from panobbgo.lib import Problem
from panobbgo.config import Config


class QuadProblem(Problem):
    """Simple ``dim``-D problem with box ``[-5, 5]^dim``."""

    def __init__(self, dim=3):
        super().__init__(box=[(-5.0, 5.0)] * dim)

    def eval(self, x):
        return float(np.sum(x**2))


def _make_strategy(dim=3):
    problem = QuadProblem(dim=dim)
    config = Config(parse_args=False, testing_mode=True)
    strategy = mock.MagicMock()
    strategy.problem = problem
    strategy.config = config
    # penalty value == raw fx of the QuadProblem
    strategy.constraint_handler.get_penalty_value.side_effect = lambda r: problem.eval(r.x)
    return strategy, problem


# ---------------------------------------------------------------------------
# fit_quadratic_step: recovery on known quadratics
# ---------------------------------------------------------------------------


def _sample_around(center, spread, n, rng):
    return center + rng.uniform(-spread, spread, size=(n, len(center)))


def test_recovers_isotropic_quadratic_minimiser():
    """On f = ||x - m||^2 the Newton step from any center points (near) at m.

    The model is exact, so recovery is limited only by the mild ridge
    shrinkage (~1% per coefficient) — well within a small tolerance.  The
    data cloud (spread 1.0, ‖·‖ up to ~1.7 raw) comfortably contains ``m``.
    """
    rng = np.random.default_rng(0)
    dim = 3
    m = np.array([1.0, -2.0, 0.5])
    center = np.array([1.5, -1.5, 0.2])  # close enough that m is inside the cloud
    ranges = np.full(dim, 10.0)

    pts = _sample_around(center, 1.5, 60, rng)
    vals = np.array([np.sum((p - m) ** 2) for p in pts])

    step = fit_quadratic_step(pts, vals, center, ranges, trust_radius=10.0)
    assert step is not None
    np.testing.assert_allclose(center + step, m, atol=2e-2)


def test_recovers_well_curved_directions_of_anisotropic_quadratic():
    """Ill-conditioned diagonal quadratic: strong directions recovered; descent overall.

    The near-flat directions (tiny ``k``) carry almost no curvature signal and
    cannot be recovered from a local cloud — but the *strongly* curved ones
    are, and the composite step reduces the objective. This is exactly the
    regime a curvature-aware step wins over an isotropic one.
    """
    rng = np.random.default_rng(1)
    dim = 4
    k = np.array([100.0, 1.0, 0.02, 40.0])
    m = np.array([0.3, -0.4, 1.2, -0.7])
    center = np.zeros(dim)
    ranges = np.full(dim, 10.0)

    pts = _sample_around(center, 1.0, 120, rng)

    def f(x):
        return float(np.sum(k * (x - m) ** 2))

    vals = np.array([f(p) for p in pts])
    step = fit_quadratic_step(pts, vals, center, ranges, trust_radius=10.0)
    assert step is not None
    # The curvature-aware step descends, and beats the *average* isotropic
    # step of equal magnitude — the property that makes it worth the fit.
    assert f(center + step) < f(center)
    step_norm = np.linalg.norm(step)
    rng2 = np.random.default_rng(99)
    iso_vals = []
    for _ in range(200):
        d = rng2.normal(size=dim)
        d = d / np.linalg.norm(d) * step_norm
        iso_vals.append(f(center + d))
    assert f(center + step) < np.mean(iso_vals)


def test_quadratic_step_improves_rosenbrock_locally():
    """A coupled Rosenbrock-like local model: the Newton step must reduce f.

    Uses a realistic (small) trust radius so the step interpolates inside the
    fitted cloud rather than extrapolating across the quartic valley.
    """
    rng = np.random.default_rng(2)
    dim = 2

    def rosen(x):
        return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

    center = np.array([-0.5, 0.5])
    ranges = np.full(dim, 4.0)
    pts = _sample_around(center, 0.15, 60, rng)
    vals = np.array([rosen(p) for p in pts])

    step = fit_quadratic_step(pts, vals, center, ranges, trust_radius=0.1)
    assert step is not None
    assert rosen(center + step) < rosen(center)


# ---------------------------------------------------------------------------
# fit_quadratic_step: robustness guards
# ---------------------------------------------------------------------------


def test_returns_none_with_too_few_points():
    rng = np.random.default_rng(3)
    dim = 3
    center = np.zeros(dim)
    ranges = np.full(dim, 10.0)
    pts = _sample_around(center, 1.0, 2 * dim, rng)  # one short of 2*dim+1
    vals = np.sum(pts**2, axis=1)
    assert fit_quadratic_step(pts, vals, center, ranges, trust_radius=5.0) is None


def test_drops_non_finite_values():
    rng = np.random.default_rng(4)
    dim = 2
    m = np.array([0.5, -0.5])
    center = np.zeros(dim)
    ranges = np.full(dim, 10.0)
    pts = _sample_around(center, 1.0, 40, rng)
    vals = np.sum((pts - m) ** 2, axis=1)
    vals[::5] = np.inf  # penalty spikes — must be ignored, not poison the fit
    step = fit_quadratic_step(pts, vals, center, ranges, trust_radius=10.0)
    assert step is not None
    np.testing.assert_allclose(center + step, m, atol=2e-2)


def test_trust_region_clips_step_norm():
    rng = np.random.default_rng(5)
    dim = 3
    m = np.array([4.0, 4.0, 4.0])  # far from center -> large unconstrained step
    center = np.zeros(dim)
    ranges = np.full(dim, 10.0)
    pts = _sample_around(center, 1.0, 60, rng)
    vals = np.sum((pts - m) ** 2, axis=1)

    trust = 0.05
    step = fit_quadratic_step(pts, vals, center, ranges, trust_radius=trust)
    assert step is not None
    # ‖step‖ in normalised units must not exceed the trust radius (+ tiny slack).
    norm_u = np.linalg.norm(step / ranges)
    assert norm_u <= trust + 1e-9
    # and it still points toward the minimiser
    assert np.dot(step, m - center) > 0


def test_r2_gate_passes_on_clean_quadratic():
    """A well-fitting quadratic clears a strict R² gate."""
    rng = np.random.default_rng(60)
    dim = 3
    m = np.array([0.5, -0.5, 0.2])
    center = np.zeros(dim)
    ranges = np.full(dim, 10.0)
    pts = _sample_around(center, 1.0, 60, rng)
    vals = np.sum((pts - m) ** 2, axis=1)
    step = fit_quadratic_step(pts, vals, center, ranges, trust_radius=10.0, min_r2=0.95)
    assert step is not None


def test_r2_gate_rejects_multimodal_neighbourhood():
    """A neighbourhood straddling several basins fails the R² gate → None."""
    rng = np.random.default_rng(61)
    dim = 2
    center = np.zeros(dim)
    ranges = np.full(dim, 10.0)
    # Highly multimodal (egg-carton) sampled over a wide cloud: no single
    # quadratic explains it, so a strict gate must reject the fit.
    pts = _sample_around(center, 3.0, 120, rng)
    vals = np.array([float(np.sum(np.sin(3.0 * p) ** 2) + 0.01 * np.sum(p**2)) for p in pts])
    step = fit_quadratic_step(pts, vals, center, ranges, trust_radius=5.0, min_r2=0.9)
    assert step is None
    # ... and with the gate disabled a step is still produced (gate is the cause)
    assert fit_quadratic_step(pts, vals, center, ranges, trust_radius=5.0, min_r2=0.0) is not None


# ---------------------------------------------------------------------------
# fit_quadratic_step: distance-weighting scheme (weight_sigma)
# ---------------------------------------------------------------------------


def test_weight_sigma_none_reproduces_legacy_rank_weights():
    """``weight_sigma=None`` must be byte-for-byte the pre-2026-07-11 fit.

    The legacy weighting was ``w = 1 / (1 + rank)``.  We reconstruct that
    weighted-least-squares solution by hand and check the returned step
    matches the ``weight_sigma=None`` path exactly.
    """
    rng = np.random.default_rng(70)
    dim = 2
    m = np.array([0.4, -0.3])
    center = np.zeros(dim)
    ranges = np.full(dim, 8.0)
    # Fewer points than max_points (= 4·n_features = 24 here) so the fit uses
    # every sample and the hand-reconstruction below need not replicate the
    # nearest-point truncation.
    pts = _sample_around(center, 1.0, 20, rng)
    vals = np.sum((pts - m) ** 2, axis=1)

    step_legacy = fit_quadratic_step(pts, vals, center, ranges, trust_radius=10.0, weight_sigma=None)
    assert step_legacy is not None

    # Hand-recompute the legacy rank-weighted normal equations.
    safe = ranges
    u = (pts - center) / safe
    dists = np.linalg.norm(u, axis=1)
    rank = np.argsort(np.argsort(dists))
    w = 1.0 / (1.0 + rank)
    ti, tj = np.triu_indices(dim)
    quad = u[:, ti] * u[:, tj]
    phi = np.concatenate([np.ones((len(u), 1)), u, quad], axis=1)
    ata = phi.T @ (phi * w[:, None])
    aty = (phi * w[:, None]).T @ vals
    rd = np.diag(ata).copy()
    rd[0] = 0.0
    beta = np.linalg.solve(ata + 1e-2 * np.diag(rd), aty)
    g = beta[1 : 1 + dim]
    qc = beta[1 + dim :]
    H = np.zeros((dim, dim))
    for k, (i, j) in enumerate(zip(ti, tj)):
        if i == j:
            H[i, i] = 2.0 * qc[k]
        else:
            H[i, j] = qc[k]
            H[j, i] = qc[k]
    ev, evec = np.linalg.eigh(H)
    sc = max(np.max(np.abs(ev)), 1e-8)
    ev = np.maximum(ev, 1e-3 * sc)
    su = -(evec @ np.diag(1.0 / ev) @ evec.T) @ g
    su = su * safe
    np.testing.assert_allclose(step_legacy, su, rtol=1e-9, atol=1e-12)


def test_gaussian_weighting_is_the_default():
    """The default weighting differs from the legacy rank scheme.

    On an anisotropic quadratic the two weightings weight the sample
    differently, so the returned steps must not be identical — a guard that
    the new default is actually wired through.
    """
    rng = np.random.default_rng(71)
    dim = 3
    k = np.array([50.0, 1.0, 8.0])
    m = np.array([0.3, -0.2, 0.5])
    center = np.zeros(dim)
    ranges = np.full(dim, 10.0)
    pts = _sample_around(center, 1.2, 80, rng)
    vals = np.array([float(np.sum(k * (p - m) ** 2)) for p in pts])

    default_step = fit_quadratic_step(pts, vals, center, ranges, trust_radius=10.0)
    legacy_step = fit_quadratic_step(pts, vals, center, ranges, trust_radius=10.0, weight_sigma=None)
    assert default_step is not None and legacy_step is not None
    assert not np.allclose(default_step, legacy_step)


def test_gaussian_weighting_localises_fit_on_curved_valley():
    """On a curved (Rosenbrock) valley the localised fit gives a better step.

    A quadratic averaged over a wide cloud is a poor model of the quartic
    valley; the Gaussian kernel down-weights the far points so the model
    tracks the local valley floor.  A single Newton step from one cloud is
    noisy, so we compare the *median* post-step objective across many random
    clouds — the localised fit must win in aggregate, the property motivating
    the 2026-07-11 change.
    """
    dim = 5

    def rosen(x):
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

    center = np.array([0.5, 0.25, 0.06, 0.004, 1e-4])
    ranges = np.full(dim, 4.0)
    trust = 2.0 * 0.124 * np.sqrt(dim)

    gauss_f, rank_f = [], []
    for seed in range(40):
        rng = np.random.default_rng(100 + seed)
        pts = center + rng.normal(0.0, 0.12, size=(84, dim)) * ranges
        vals = np.array([rosen(p) for p in pts])
        sg = fit_quadratic_step(pts, vals, center, ranges, trust, min_r2=0.0, weight_sigma=0.35)
        sr = fit_quadratic_step(pts, vals, center, ranges, trust, min_r2=0.0, weight_sigma=None)
        if sg is not None:
            gauss_f.append(rosen(center + sg))
        if sr is not None:
            rank_f.append(rosen(center + sr))
    assert gauss_f and rank_f
    assert np.median(gauss_f) < np.median(rank_f)


def test_gaussian_weighting_handles_degenerate_coincident_cloud():
    """All points coinciding with the center → uniform-weight fallback, no crash."""
    dim = 2
    center = np.zeros(dim)
    ranges = np.full(dim, 10.0)
    pts = np.tile(center, (2 * dim + 1, 1))  # every point identical to center
    vals = np.zeros(len(pts))
    # Must not raise (median distance is 0 → uniform fallback); a flat sample
    # yields no informative step, so None is acceptable.
    step = fit_quadratic_step(pts, vals, center, ranges, trust_radius=5.0)
    assert step is None or np.all(np.isfinite(step))


def test_saddle_indefinite_hessian_still_descends():
    """Indefinite model (saddle) — PD regularisation yields a finite descent step."""
    rng = np.random.default_rng(6)
    dim = 2

    def saddle(x):
        return x[0] ** 2 - 0.5 * x[1] ** 2  # indefinite Hessian

    center = np.array([1.0, 1.0])
    ranges = np.full(dim, 10.0)
    pts = _sample_around(center, 0.5, 50, rng)
    vals = np.array([saddle(p) for p in pts])
    step = fit_quadratic_step(pts, vals, center, ranges, trust_radius=1.0)
    assert step is not None
    assert np.all(np.isfinite(step))
    assert saddle(center + step) < saddle(center)


# ---------------------------------------------------------------------------
# Nearby(quadratic=...) wiring
# ---------------------------------------------------------------------------


def test_quadratic_disabled_is_default_and_skips_buffer():
    strategy, _ = _make_strategy()
    h = Nearby(strategy)
    h.__start__()
    assert h.quadratic is False
    # on_new_results must be an inert no-op when disabled
    r = mock.MagicMock()
    r.x = np.zeros(3)
    h.on_new_results([r])
    assert h._hist_x == []


def test_quadratic_name_suffix():
    strategy, _ = _make_strategy()
    assert "+quad" in Nearby(strategy, quadratic=True).name
    assert "+quad" not in Nearby(strategy).name


def test_invalid_quadratic_params_raise():
    strategy, _ = _make_strategy()
    with pytest.raises(ValueError):
        Nearby(strategy, quadratic="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Nearby(strategy, quadratic=True, quadratic_trust=0.0)
    with pytest.raises(ValueError):
        Nearby(strategy, quadratic=True, quadratic_trust=float("nan"))
    with pytest.raises(ValueError):
        Nearby(strategy, quadratic=True, quadratic_min_r2=1.5)
    with pytest.raises(ValueError):
        Nearby(strategy, quadratic=True, quadratic_min_r2=-0.1)
    with pytest.raises(ValueError):
        Nearby(strategy, quadratic=True, quadratic_weight_sigma=0.0)
    with pytest.raises(ValueError):
        Nearby(strategy, quadratic=True, quadratic_weight_sigma=-0.5)
    with pytest.raises(ValueError):
        Nearby(strategy, quadratic=True, quadratic_weight_sigma=float("nan"))


def test_quadratic_weight_sigma_default_and_forwarded():
    """The default sigma is 0.35 and is threaded into ``fit_quadratic_step``."""
    strategy, _ = _make_strategy(dim=3)
    h = Nearby(strategy, quadratic=True)
    h.__start__()
    assert h.quadratic_weight_sigma == 0.35

    rng = np.random.default_rng(11)
    center = np.array([2.0, -1.0, 0.5])
    for _ in range(60):
        r = mock.MagicMock()
        r.x = center + rng.uniform(-0.5, 0.5, size=3)
        h.on_new_results([r])

    with mock.patch("panobbgo.heuristics.nearby.fit_quadratic_step", wraps=fit_quadratic_step) as spy:
        h._quadratic_candidate(center)
    assert spy.call_count == 1
    assert spy.call_args.kwargs["weight_sigma"] == 0.35


def test_on_new_results_accumulates_and_caps():
    strategy, problem = _make_strategy(dim=3)
    h = Nearby(strategy, quadratic=True)
    h.__start__()
    rng = np.random.default_rng(7)
    # Feed many results; buffer must cap at _hist_capacity().
    for _ in range(500):
        r = mock.MagicMock()
        r.x = rng.uniform(-1, 1, size=3)
        h.on_new_results([r])
    cap = h._hist_capacity()
    assert len(h._hist_x) == cap
    assert len(h._hist_f) == cap


def test_quadratic_candidate_none_when_buffer_thin():
    strategy, _ = _make_strategy(dim=3)
    h = Nearby(strategy, quadratic=True)
    h.__start__()
    # fewer than 2*dim+1 points
    for _ in range(5):
        r = mock.MagicMock()
        r.x = np.zeros(3)
        h.on_new_results([r])
    assert h._quadratic_candidate(np.zeros(3)) is None


def test_quadratic_candidate_points_toward_minimum():
    strategy, problem = _make_strategy(dim=3)
    h = Nearby(strategy, quadratic=True)
    h.__start__()
    rng = np.random.default_rng(8)
    center = np.array([2.0, -1.0, 0.5])
    # QuadProblem minimum is the origin; feed a local cloud around `center`.
    for _ in range(60):
        r = mock.MagicMock()
        r.x = center + rng.uniform(-0.5, 0.5, size=3)
        h.on_new_results([r])
    cand = h._quadratic_candidate(center)
    assert cand is not None
    # The step should reduce ||x||^2 relative to the center.
    assert np.sum(cand**2) < np.sum(center**2)


def test_on_new_best_emits_quadratic_first_point():
    strategy, problem = _make_strategy(dim=3)
    h = Nearby(strategy, quadratic=True, new=3, axes="all")
    h.__start__()
    rng = np.random.default_rng(9)
    center = np.array([2.0, -1.0, 0.5])
    for _ in range(60):
        r = mock.MagicMock()
        r.x = center + rng.uniform(-0.5, 0.5, size=3)
        h.on_new_results([r])

    best = mock.MagicMock()
    best.x = center
    h.on_new_best(best)
    pts = h.get_points()
    assert len(pts) == 3
    # First point is the curvature-aware step; it should sit closer to the
    # origin (the true minimum) than the center does.
    first = pts[0].x
    assert np.sum(first**2) < np.sum(center**2)


def test_on_new_best_all_random_when_no_model():
    """With an empty buffer every emitted point is an isotropic perturbation."""
    strategy, problem = _make_strategy(dim=3)
    h = Nearby(strategy, quadratic=True, new=2, axes="all", radius=0.01)
    h.__start__()
    best = mock.MagicMock()
    best.x = np.array([1.0, 1.0, 1.0])
    h.on_new_best(best)
    pts = h.get_points()
    assert len(pts) == 2
    # All within one radius*range perturbation of the center (no big Newton jump).
    for p in pts:
        assert np.max(np.abs(p.x - best.x)) <= 0.01 * 10.0 + 1e-9
