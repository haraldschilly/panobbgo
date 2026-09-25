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

"""Every classic problem that declares ``x_opt`` / ``f_opt`` is right about it.

Wood was unbounded below, Branin's default killed its cosine term, Box had
the wrong exponent sign, Step had no floor, Trigonometric and Powell had
wrong terms, RosenbrockModified claimed ``f(-1, -1) = 0`` (it is 78; the
minimum is 34.04), Sargan lacked its factor ``D``.  Two checks per
declaration: the value at ``x_opt`` is ``f_opt``, and a seeded DE search
over the default box finds nothing lower.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from panobbgo.lib import classic


def _instances():
    out = []
    for name, cls in sorted(vars(classic).items()):
        if not (inspect.isclass(cls) and issubclass(cls, classic.Problem) and cls is not classic.Problem):
            continue
        if cls.__module__ != classic.__name__ or not hasattr(cls, "f_opt"):
            continue
        params = inspect.signature(cls.__init__).parameters
        if "dims" in params:
            out += [pytest.param(cls, {"dims": d}, id=f"{name}-{d}") for d in (2, 4)]
        else:
            out.append(pytest.param(cls, {}, id=name))
    return out


INSTANCES = _instances()


def test_declarations_cover_the_fixed_functions():
    names = {p.id.split("-")[0] for p in INSTANCES}
    fixed = {"Wood", "Branin", "Box", "Step", "Trigonometric", "Powell", "RosenbrockModified", "Sargan"}
    assert fixed <= names


@pytest.mark.parametrize("cls,kwargs", INSTANCES)
def test_f_at_x_opt_is_f_opt_and_de_cannot_beat_it(cls, kwargs):
    from scipy.optimize import differential_evolution

    prob = cls(**kwargs)
    x_opt = np.asarray(prob.x_opt if prob.x_opt is not None else prob.x_opt_example, dtype=np.float64)
    f_opt = float(prob.f_opt)
    box = np.asarray(prob.box.box)
    assert x_opt.shape == (prob.dim,)
    assert np.all(box[:, 0] <= x_opt) and np.all(x_opt <= box[:, 1]), "x_opt outside the default box"
    assert prob.eval(x_opt) == pytest.approx(f_opt, rel=1e-8, abs=1e-8)

    res = differential_evolution(
        lambda x: float(prob.eval(x)), list(map(tuple, box)), seed=1, maxiter=300, tol=1e-12, polish=True
    )
    assert res.fun >= f_opt - 1e-6 * (1.0 + abs(f_opt)), f"DE found {res.fun} at {res.x} below f_opt={f_opt}"


def test_seeded_stochastic_problems_are_reproducible():
    x = np.array([0.3, -0.7, 1.1])
    a = classic.RosenbrockStochastic(dims=3, seed=5)
    b = classic.RosenbrockStochastic(dims=3, seed=5)
    assert [a.eval(x) for _ in range(3)] == [b.eval(x) for _ in range(3)]
    n1 = classic.NesterovQuadratic(dim=5, seed=2)
    n2 = classic.NesterovQuadratic(dim=5, seed=2)
    assert n1.dim == 5 and np.array_equal(n1.A, n2.A) and np.array_equal(n1.b, n2.b)
