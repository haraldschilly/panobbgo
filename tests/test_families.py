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

"""Tests for :mod:`panobbgo.lib.families` and :mod:`panobbgo.harness_families`.

The properties under test are the ones the AOCC metric *depends* on: an
instance must be reproducible from its seed, ``f(x_opt)`` must equal the
declared ``f_opt`` exactly, ``f_opt`` must actually be the minimum (a
rotation must not move it), and on a constrained instance ``x_opt`` must
be feasible with the first constraint active — otherwise the constraints
do nothing and the "constrained" battery is the unconstrained one.
"""

import numpy as np
import pytest

from panobbgo.harness_families import (
    PENALTY_RHO,
    PenaltyTracker,
    describe_instances,
    make_constrained_battery,
    make_families_battery,
    run_family_harness,
)
from panobbgo.harness_ioh import make_ioh_strategies
from panobbgo.lib import Point
from panobbgo.lib.families import (
    BASE_FUNCTIONS,
    Family,
    FamilyConfig,
    make_family_instances,
)

ALL_BASES = sorted(BASE_FUNCTIONS)


# ---------------------------------------------------------------------------
# Base functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", ALL_BASES)
@pytest.mark.parametrize("dim", [2, 5])
def test_base_minimum_is_zero_at_origin(base, dim):
    """Every base is normalised to ``f(0) == 0`` *exactly*, not to 1e-16."""
    f = BASE_FUNCTIONS[base](dim)
    assert f(np.zeros(dim)) == 0.0


@pytest.mark.parametrize("base", ALL_BASES)
def test_base_origin_is_the_global_minimum(base):
    """A random probe never dips below the base's declared minimum.

    This is the property that makes ``f_opt`` an AOCC *target* rather than
    a guess; it is also the check that catches a base whose optimum is
    only box-local (Schwefel, fenced by a boundary penalty).
    """
    dim = 3
    f = BASE_FUNCTIONS[base](dim)
    rng = np.random.default_rng(11)
    # Well beyond the reach of a family instance's affine transform.
    probes = rng.uniform(-30.0, 30.0, size=(3000, dim))
    values = np.array([f(p) for p in probes])
    assert values.min() >= -1e-9


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", ALL_BASES)
@pytest.mark.parametrize("dim", [2, 5])
def test_optimum_value_is_exact(base, dim):
    """``f(x_opt) == f_opt`` bit-for-bit, and ``f_opt`` is not trivially zero."""
    p = Family(base, dim=dim, seed=1234)
    assert p.eval(p.x_opt) == p.f_opt
    assert p.f_opt != 0.0  # the vertical offset is really drawn


@pytest.mark.parametrize("base", ALL_BASES)
def test_rotation_preserves_the_optimum(base):
    """Rotating (and conditioning) moves the landscape, never the optimum.

    ``R`` is orthogonal and ``Lambda`` invertible, so the transform is a
    bijection sending ``x_opt`` to the base's minimiser: the optimum value
    is identical with and without the rotation, and no probe beats it.
    """
    dim = 4
    plain = Family(base, dim=dim, seed=7, rotate=False)
    rotated = Family(base, dim=dim, seed=7, rotate=True)
    cond = Family(base, dim=dim, seed=7, rotate=True, condition=100.0)

    # Same seed -> same x_opt and f_opt; only the landscape between them differs.
    assert np.array_equal(plain.x_opt, rotated.x_opt)
    assert plain.f_opt == rotated.f_opt == cond.f_opt
    for p in (plain, rotated, cond):
        assert p.eval(p.x_opt) == p.f_opt

    assert rotated.rotation is not None
    r = rotated.rotation
    np.testing.assert_allclose(r @ r.T, np.eye(dim), atol=1e-12)
    assert plain.rotation is None
    assert plain.scaling is None
    assert cond.scaling is not None

    # ... and the rotation actually changed the function somewhere.
    # (The sphere is rotation-invariant by construction, so it is exempt.)
    probe = plain.x_opt + 1.0
    if base != "sphere":
        assert plain.eval(probe) != rotated.eval(probe)

    rng = np.random.default_rng(3)
    probes = rng.uniform(-5.0, 5.0, size=(2000, dim))
    for p in (plain, rotated, cond):
        values = np.array([p.eval(x) for x in probes])
        assert values.min() >= p.f_opt - 1e-9


def test_instances_are_deterministic_per_seed():
    """Same seed -> identical instance; different seed -> different instance."""
    a = Family("rastrigin", dim=5, seed=99)
    b = Family("rastrigin", dim=5, seed=99)
    c = Family("rastrigin", dim=5, seed=100)

    assert np.array_equal(a.x_opt, b.x_opt)
    assert a.f_opt == b.f_opt
    assert a.rotation is not None and b.rotation is not None
    np.testing.assert_array_equal(a.rotation, b.rotation)

    rng = np.random.default_rng(0)
    probes = rng.uniform(-5.0, 5.0, size=(50, 5))
    assert [a.eval(x) for x in probes] == [b.eval(x) for x in probes]

    assert not np.array_equal(a.x_opt, c.x_opt)


def test_make_family_instances_is_deterministic_and_shaped():
    """The battery is a pure function of ``(families, dims, n_instances, seed)``."""
    fams = ["sphere", FamilyConfig(base="ellipsoid", condition=10.0)]
    one = make_family_instances(fams, dims=(2, 5), n_instances=3, seed=5)
    two = make_family_instances(fams, dims=(2, 5), n_instances=3, seed=5)
    other = make_family_instances(fams, dims=(2, 5), n_instances=3, seed=6)

    assert len(one) == 2 * 2 * 3
    assert [n for n, _ in one] == [n for n, _ in two]
    assert all(a.f_opt == b.f_opt for (_, a), (_, b) in zip(one, two))
    assert any(a.f_opt != b.f_opt for (_, a), (_, b) in zip(one, other))

    names = [n for n, _ in one]
    assert names[0] == "sphere_d2_i0"
    assert len(set(names)) == len(names)
    # Adding a dimension does not re-draw the instances already there.
    wider = make_family_instances(fams, dims=(2, 5, 10), n_instances=3, seed=5)
    kept = {n: p.f_opt for n, p in wider}
    assert all(kept[n] == p.f_opt for n, p in one)


def test_x_opt_is_inside_the_box_and_unconstrained_has_no_constraints():
    p = Family("sphere", dim=6, seed=2)
    assert Point(p.x_opt, "test") in p.box
    assert p.eval_constraints(p.x_opt) is None
    assert p.n_constraints == 0
    assert p.name == "sphere_d6_i0"


def test_unknown_base_and_bad_dim_are_rejected():
    with pytest.raises(ValueError, match="unknown base"):
        Family("no-such-function", dim=2, seed=0)
    with pytest.raises(ValueError, match="dim must be"):
        Family("sphere", dim=1, seed=0)
    with pytest.raises(ValueError, match="unknown constraint_kind"):
        Family("sphere", dim=2, seed=0, n_constraints=1, constraint_kind="nope")


# ---------------------------------------------------------------------------
# Constrained instances
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["linear", "ball", "mixed"])
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("dim", [2, 5])
def test_optimum_is_feasible_with_an_active_constraint(kind, k, dim):
    """``x_opt`` is feasible, the first constraint is exactly active there.

    ``Result.cv`` counts only *strictly positive* entries, so an exactly
    active constraint (``g == 0``) is feasible.  Without an active one the
    constraints would be slack at the optimum and would not bite.
    """
    p = Family("rosenbrock", dim=dim, seed=17, n_constraints=k, constraint_kind=kind)
    g = p.eval_constraints(p.x_opt)
    assert g is not None
    assert g.shape == (k,)
    assert np.all(g <= 0.0), g
    assert g[0] == pytest.approx(0.0, abs=1e-12)

    # The panobbgo-level view: zero violation, i.e. feasible.
    result = p(Point(p.x_opt, "test"))
    assert result.cv == 0.0
    assert result.fx == p.f_opt

    # The constraints are a real restriction: some box points violate them.
    rng = np.random.default_rng(4)
    probes = rng.uniform(-5.0, 5.0, size=(400, dim))
    violated = [np.any(p.eval_constraints(x) > 0) for x in probes]
    assert any(violated)


def test_constraints_are_o1_and_bounded_away_from_absurd_scales():
    """Violations are normalised, the way ``classic.py``'s PressureVessel is."""
    rng = np.random.default_rng(6)
    for kind in ("linear", "ball"):
        p = Family("sphere", dim=5, seed=21, n_constraints=3, constraint_kind=kind)
        probes = rng.uniform(-5.0, 5.0, size=(500, 5))
        g = np.array([p.eval_constraints(x) for x in probes])
        assert np.abs(g).max() < 50.0


def test_constrained_instance_still_has_an_exact_known_optimum():
    """The constrained optimum equals the unconstrained one — the AOCC target."""
    free = Family("rastrigin", dim=3, seed=31, n_constraints=0)
    tied = Family("rastrigin", dim=3, seed=31, n_constraints=2, constraint_kind="ball")
    assert np.array_equal(free.x_opt, tied.x_opt)
    assert free.f_opt == tied.f_opt
    assert tied.eval(tied.x_opt) == tied.f_opt

    rng = np.random.default_rng(8)
    probes = rng.uniform(-5.0, 5.0, size=(2000, 3))
    feasible = [x for x in probes if np.all(tied.eval_constraints(x) <= 0)]
    assert feasible, "the feasible region should not be empty inside the box"
    assert min(tied.eval(x) for x in feasible) >= tied.f_opt - 1e-9


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


def test_penalty_tracker_records_the_penalty_value():
    """The metric's trace is ``f + rho*cv``; the strategy still sees ``f``."""
    p = Family("sphere", dim=2, seed=3, n_constraints=1, constraint_kind="linear")
    pristine = Family("sphere", dim=2, seed=3, n_constraints=1, constraint_kind="linear")
    # A point on the infeasible side of the active half-space.
    rng = np.random.default_rng(0)
    bad = next(x for x in rng.uniform(-5.0, 5.0, size=(200, 2)) if pristine.eval_constraints(x)[0] > 0)
    raw = pristine.eval(bad)
    g = pristine.eval_constraints(bad)
    assert g is not None and g[0] > 0
    cv = float(np.linalg.norm(g[g > 0]))

    tracker = PenaltyTracker(p, budget=10)
    try:
        returned = p.eval(bad)  # goes through the tracker
        assert tracker.best_so_far[-1] == pytest.approx(raw + PENALTY_RHO * cv)
        # ``eval`` handed the *unpenalised* objective back to the strategy.
        assert returned == raw
        # The optimum itself is feasible, so its penalty value is f_opt.
        p.eval(p.x_opt)
        assert tracker.best_fx == pytest.approx(p.f_opt)
        assert tracker.best_cv == 0.0
    finally:
        tracker.restore()
    assert p.eval(p.x_opt) == p.f_opt  # the patch was undone


def test_presets_have_the_documented_shape():
    free = make_families_battery()
    con = make_constrained_battery()

    d_free = describe_instances(free)
    assert d_free["dims"] == [2, 5, 10]
    assert len(d_free["families"]) == 5
    assert d_free["constrained"] is False
    assert d_free["n_instances"] == 5 * 3 * 3

    d_con = describe_instances(con)
    assert d_con["dims"] == [2, 5]
    assert len(d_con["families"]) == 4
    assert d_con["n_constraints"] == [1, 2, 3]
    assert d_con["n_instances"] == 4 * 2 * 3
    # Every constrained instance keeps an exactly active first constraint.
    for _n, p in con:
        g = p.eval_constraints(p.x_opt)
        assert g is not None and np.all(g <= 0.0) and g[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("preset", ["free", "constrained"])
def test_tiny_run_through_the_harness(preset):
    """One seed, one instance, a real strategy: no error, full budget spent."""
    battery = make_families_battery if preset == "free" else make_constrained_battery
    instances = [x for x in battery(dims=(2,), n_instances=1)][:1]
    specs = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"]
    assert specs

    result = run_family_harness(
        specs,
        instances,
        budget_multiplier=25,
        base_seed=42,
        progress=False,
    )
    assert len(result.runs) == 1
    run = result.runs[0]
    assert run.error is None
    assert run.n_evals == run.budget == 50
    assert 0.0 <= run.aocc <= 1.0
    assert run.precision >= -1e-9  # f_opt really is the minimum
    assert result.sync_eval is True
    assert run.problem_kind == instances[0][1].family


def test_harness_is_reproducible_and_paired_on_the_rng_identity():
    """Same base seed -> same AOCC; ``seed_name`` shares the stream."""
    import dataclasses

    instances = make_families_battery(dims=(2,), n_instances=1)[:1]
    base = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"][0]
    a = dataclasses.replace(base, name="arm_a", seed_name="shared")
    b = dataclasses.replace(base, name="arm_b", seed_name="shared")

    first = run_family_harness([a, b], instances, budget_multiplier=25, base_seed=7, progress=False)
    again = run_family_harness([a], instances, budget_multiplier=25, base_seed=7, progress=False)

    seeds = {r.strategy_name: r.seed for r in first.runs}
    assert seeds["arm_a"] == seeds["arm_b"], "a shared seed_name must share the RNG stream"
    assert first.runs[0].seed == again.runs[0].seed
    assert first.runs[0].aocc == again.runs[0].aocc
