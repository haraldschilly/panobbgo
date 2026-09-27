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

"""The wide preset (:func:`panobbgo.harness_families.make_wide_battery`) and its placement knobs.

What AOCC and the selector depend on: ``f(x_opt) == f_opt`` exactly and
nothing in the box below it, instances that are a pure function of their
seed, the optimum away from the box centre, values finite everywhere in
the box — and knobs that do what they claim (a face optimum is really
constrained by the box, the neutral directions of an embedding really are
neutral), without moving any instance that does not use them.
"""

import pickle

import numpy as np
import pytest

from panobbgo.harness_families import (
    FREE_FAMILIES,
    WIDE_FAMILIES,
    WIDE_MIN_CENTRE_DIST,
    make_families_battery,
    make_wide_battery,
)
from panobbgo.lib.families import (
    BASE_FUNCTIONS,
    EMBEDDABLE_BASES,
    MAX_CENTRE_DIST_FRACTION,
    STYBLINSKI_TANG_ARGMIN,
    Family,
)

B = 5.0


@pytest.fixture(scope="module")
def wide():
    return make_wide_battery()


def _box_probes(dim: int, n: int, seed: int) -> np.ndarray:
    """Uniform points plus every corner (d <= 5) or 64 random corners: the extremes of the box."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-B, B, size=(n, dim))
    if dim <= 5:
        corners = np.array(np.meshgrid(*[[-B, B]] * dim)).reshape(dim, -1).T
    else:
        corners = rng.choice([-B, B], size=(64, dim))
    return np.vstack([pts, corners])


def test_wide_preset_shape(wide):
    labels = [cfg.name() for cfg in WIDE_FAMILIES]
    assert len(labels) == len(set(labels)) == 15
    assert len(wide) == 15 * 3 * 3
    assert [n for n, _ in wide] == [n for n, _ in make_wide_battery()]
    assert {p.dim for _, p in wide} == {2, 5, 10}
    # The free preset's families are all in it, under the same labels.
    assert {cfg.name() for cfg in FREE_FAMILIES} <= set(labels)


def test_f_opt_is_exact_and_the_minimum_over_the_box(wide):
    for i, (name, p) in enumerate(wide):
        assert p.eval(p.x_opt) == p.f_opt, name
        values = np.array([p.eval(x) for x in _box_probes(p.dim, 400, i)])
        assert np.all(np.isfinite(values)), name
        assert values.min() >= p.f_opt - 1e-9, name


def test_the_optimum_is_never_near_the_box_centre(wide):
    """Every optimum, including every point of an embedding's optimal set, is >= 0.3 B sqrt(d) from the centre."""
    for name, p in wide:
        x = p.x_opt
        bound = WIDE_MIN_CENTRE_DIST * B * np.sqrt(p.dim)
        assert np.all(np.abs(x) <= B), name
        assert np.linalg.norm(x) >= bound - 1e-12, name
        assert p.centre_distance(x) >= bound - 1e-12, name
        if p.effective_dim < 1.0:
            # The nearest point of the optimal set x_opt + span(R[k:]): the projection onto span(R[:k]).
            k = int(np.ceil(p.effective_dim * p.dim - 1e-9))
            r = p.rotation
            assert r is not None
            nearest = r[:k].T @ (r[:k] @ x)
            assert p.eval(nearest) == pytest.approx(p.f_opt, abs=1e-9), name
            assert np.linalg.norm(nearest) >= bound - 1e-12, name


def test_every_instance_is_shifted_and_rotated_or_signed_permuted(wide):
    for name, p in wide:
        r = p.rotation
        assert r is not None, name
        np.testing.assert_allclose(r @ r.T, np.eye(p.dim), atol=1e-12)
        if p.family.endswith("_sep"):
            # A signed permutation: one +-1 per row and column, separability kept.
            assert np.all(np.isin(r, (-1.0, 0.0, 1.0))) and np.all(np.abs(r).sum(axis=0) == 1), name
        else:
            assert np.count_nonzero(np.abs(r) > 1e-12) > p.dim, name
    # Instances of one family differ in their optimum and rotation.
    by_fam = {}
    for _, p in wide:
        by_fam.setdefault((p.family, p.dim), []).append(p)
    for (family, dim), ps in by_fam.items():
        assert len({tuple(np.round(p.x_opt, 12)) for p in ps}) == 3
        if not (family.endswith("_sep") and dim == 2):  # d = 2 has only 8 signed permutations
            assert len({tuple(np.round(p.rotation.ravel(), 12)) for p in ps}) == 3


def test_instances_are_deterministic(wide):
    again = dict(make_wide_battery())
    rng = np.random.default_rng(5)
    for name, p in wide:
        q = again[name]
        assert np.array_equal(p.x_opt, q.x_opt) and p.f_opt == q.f_opt
        assert np.array_equal(p.rotation, q.rotation)
        xs = rng.uniform(-B, B, size=(5, p.dim))
        assert [p.eval(x) for x in xs] == [q.eval(x) for x in xs]


def test_wide_instances_pickle(wide):
    rng = np.random.default_rng(9)
    for name, p in wide[::7]:
        q = pickle.loads(pickle.dumps(p))
        xs = rng.uniform(-B, B, size=(5, p.dim))
        assert [p.eval(x) for x in xs] == [q.eval(x) for x in xs], name


def test_shared_labels_keep_the_free_instances_unless_redrawn():
    """The free families in wide are the free instances, except where ``min_centre_dist`` redrew ``x_opt``."""
    free = dict(make_families_battery())
    wide = dict(make_wide_battery())
    kept = 0
    for name, p in free.items():
        w = wide[name]
        assert w.f_opt == p.f_opt and np.array_equal(w.rotation, p.rotation)
        if np.linalg.norm(p.x_opt) >= WIDE_MIN_CENTRE_DIST * B * np.sqrt(p.dim):
            assert np.array_equal(w.x_opt, p.x_opt)
            kept += 1
        else:
            assert not np.array_equal(w.x_opt, p.x_opt)
    assert kept >= 30  # 38 of 45 on 2026-09-27


# ---------------------------------------------------------------------------
# the knobs
# ---------------------------------------------------------------------------


def test_placement_knobs_do_not_move_the_instance_stream():
    plain = Family("rastrigin", dim=5, seed=31)
    knobs = Family("rastrigin", dim=5, seed=31, min_centre_dist=0.45, boundary_faces=0.4)
    assert plain.f_opt == knobs.f_opt
    assert np.array_equal(plain.rotation, knobs.rotation)
    # Knobs at their defaults: bit-identical to an instance built without them.
    same = Family("rastrigin", dim=5, seed=31, min_centre_dist=0.0, boundary_faces=0.0, effective_dim=1.0)
    assert np.array_equal(plain.x_opt, same.x_opt)
    xs = np.random.default_rng(0).uniform(-B, B, size=(20, 5))
    assert [plain.eval(x) for x in xs] == [same.eval(x) for x in xs]


@pytest.mark.parametrize("dim", [2, 5, 10])
def test_boundary_faces_put_an_active_bound_at_the_optimum(dim):
    p = Family("rosenbrock", dim=dim, seed=4, boundary_faces=0.5)
    x = p.x_opt
    faces = np.flatnonzero(np.abs(x) == B)
    assert len(faces) == int(np.ceil(dim / 2))
    assert p.eval(x) == p.f_opt
    for i in faces:
        out = x.copy()
        out[i] += 1e-5 * np.sign(x[i])  # just outside the box: lower than f_opt
        assert p.eval(out) < p.f_opt
        inside = x.copy()
        inside[i] -= 1e-5 * np.sign(x[i])  # just inside: higher, with a non-vanishing slope
        assert (p.eval(inside) - p.f_opt) / 1e-5 > 0.5 * p.boundary_slope


@pytest.mark.parametrize("dim", [2, 5, 10])
def test_effective_dim_leaves_the_other_directions_neutral(dim):
    p = Family("levy", dim=dim, seed=8, effective_dim=1.0 / 3.0)
    k = int(np.ceil(dim / 3))
    r = p.rotation
    assert r is not None
    for j in range(k, dim):  # rows of R past k: directions the function does not see
        y = p.x_opt + 0.7 * r[j]
        assert p.eval(y) == pytest.approx(p.f_opt, abs=1e-9)
    y = p.x_opt + 0.7 * r[0]
    assert p.eval(y) > p.f_opt + 1e-3


def test_signed_permutation_and_styblinski_tang():
    p = Family("styblinski_tang", dim=4, seed=2, rotate=False, signed_permutation=True)
    assert p.eval(p.x_opt) == p.f_opt
    # The minimiser is a root of the derivative to float precision.
    u = STYBLINSKI_TANG_ARGMIN
    assert abs(4 * u**3 - 32 * u + 5) < 1e-12
    # Separable: a change of one coordinate changes f by a function of that coordinate only.
    x0 = p.x_opt + 0.3
    e = np.eye(4)
    d01 = p.eval(x0 + e[0] + e[1]) - p.eval(x0 + e[1])
    d0 = p.eval(x0 + e[0]) - p.eval(x0)
    assert d01 == pytest.approx(d0, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("dim", [2, 5, 10])
def test_schwefel_box_maps_the_box_onto_a_window_without_the_penalty(dim):
    """The box is plain Schwefel (``80 s x + o`` inside ``[-500, 500]``): the penalty never fires in it."""
    from panobbgo.lib.classic import Schwefel

    ps = [Family("schwefel_box", dim=dim, seed=s, rotate=False, signed_permutation=True) for s in (1, 2, 3)]
    raw = Schwefel(dims=dim)
    for p in ps:
        assert p.eval(p.x_opt) == p.f_opt
        assert np.all((np.abs(p.x_opt) >= 4.0) & (np.abs(p.x_opt) <= B))
        r = p.rotation
        assert r is not None
        xs = _box_probes(dim, 300, 0)
        # u = 80 * R (x - x_opt) + 420.97 stays inside the classic domain, and f is raw Schwefel there.
        us = 80.0 * (xs - p.x_opt) @ r.T + 420.9687463319553
        assert np.all(np.abs(us) <= 500.0 + 1e-9)
        base0 = raw.eval(80.0 * (p.x_opt - p.x_opt) @ r.T + 420.9687463319553)
        for x, u in zip(xs[:20], us[:20]):
            assert p.eval(x) - p.f_opt == pytest.approx(raw.eval(u) - base0, rel=1e-9, abs=1e-6)
    # Instances with the same sign pattern still differ (the random window offset).
    assert len({tuple(np.round(p.x_opt, 9)) for p in ps}) == 3
    with pytest.raises(ValueError, match="schwefel_box"):
        Family("schwefel_box", dim=dim, seed=1)  # rotated: refused


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"signed_permutation": True}, "rotate=False"),
        ({"effective_dim": 0.0}, "effective_dim"),
        ({"min_centre_dist": 0.9}, "min_centre_dist"),
        ({"min_centre_dist": 0.49}, "min_centre_dist"),  # above 0.6 * (1 - 0.2)
        ({"boundary_faces": 1.5}, "boundary_faces"),
        ({"boundary_faces": 0.5, "shift": False}, "shift=True"),
    ],
)
def test_bad_knobs_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Family("sphere", dim=3, seed=0, **kwargs)


def test_knob_combinations_the_bbob_bases_refuse():
    with pytest.raises(ValueError, match="effective_dim"):
        Family("gallagher", dim=3, seed=0, effective_dim=0.5)
    with pytest.raises(ValueError, match="schwefel_box"):
        Family("schwefel_box", dim=3, seed=0, rotate=False, min_centre_dist=0.3)
    with pytest.raises(ValueError, match="placement='box'"):
        Family("lunacek_bi_rastrigin", dim=3, seed=0, min_centre_dist=0.3)


# ---------------------------------------------------------------------------
# review #385: placement bounds, embeddable bases, levy at k = 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frac", [1.0, 1.0 / 3.0])
@pytest.mark.parametrize("dim", [2, 5, 10, 40, 160])
def test_the_min_centre_dist_bound_keeps_the_acceptance_high(dim, frac):
    """At the largest allowed ``min_centre_dist`` a uniform draw is accepted with probability > 1 %."""
    k = max(1, int(np.ceil(frac * dim - 1e-9)))
    rho = MAX_CENTRE_DIST_FRACTION * 0.8 * np.sqrt(k / dim)
    rng = np.random.default_rng(dim)
    xs = rng.uniform(-4.0, 4.0, size=(4000, dim))
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    rows = (q * np.sign(np.diag(r))).T[:k]
    for proj in (rows, np.eye(dim)[:k]):  # a Haar subspace and a coordinate one
        acc = np.mean(np.linalg.norm(xs @ proj.T, axis=1) >= rho * B * np.sqrt(dim))
        assert acc > 0.01, (dim, frac, acc)
    # ... and an instance at that bound builds (well within MAX_PLACEMENT_TRIES).
    base = "levy" if frac < 1.0 else "sphere"
    p = Family(base, dim=dim, seed=3, effective_dim=frac, min_centre_dist=rho * (1 - 1e-9))
    assert p.centre_distance(p.x_opt) >= rho * (1 - 1e-9) * B * np.sqrt(dim)


def test_placement_gives_up_with_an_error(monkeypatch):
    """The redraw loop is bounded: with no redraws allowed, a first draw that is too close raises."""
    monkeypatch.setattr(Family, "MAX_PLACEMENT_TRIES", 0)
    raised = 0
    for seed in range(40):
        try:
            Family("sphere", dim=2, seed=seed, min_centre_dist=0.45)
        except ValueError as exc:
            assert "could not place x_opt" in str(exc)
            raised += 1
    assert raised > 0


@pytest.mark.parametrize("base", EMBEDDABLE_BASES)
@pytest.mark.parametrize("frac", [0.3, 0.5])  # k = 1 and k = 2 at d = 3
def test_embeddable_bases_keep_an_exact_nonconstant_optimum(base, frac):
    p = Family(base, dim=3, seed=5, effective_dim=frac)
    assert p.eval(p.x_opt) == p.f_opt
    values = np.array([p.eval(x) for x in _box_probes(3, 500, 1)])
    assert values.min() >= p.f_opt - 1e-9
    assert np.ptp(values) > 1e-3


@pytest.mark.parametrize("base", ["rosenbrock", "sharp_ridge", "discus", "schwefel", "attractive_sector"])
def test_effective_dim_refuses_bases_that_do_not_embed(base):
    with pytest.raises(ValueError, match="effective_dim"):
        Family(base, dim=3, seed=0, effective_dim=0.5)


def test_levy_at_dimension_one():
    f = BASE_FUNCTIONS["levy"](1)
    assert f(np.zeros(1)) == 0.0
    us = np.linspace(-30.0, 30.0, 6001)
    values = np.array([f(np.array([u])) for u in us])
    assert values.min() >= 0.0 and values.max() > 1.0
