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

"""Dimensions 30/40, the ``bbob-largescale`` slice, and the sealed test set.

Three kinds of test:

* **Frozen presets stay bit-identical.**  The digests below were taken on
  master before the large and sealed batteries existed (2026-09-26,
  commit 306a8d3).  The exact part (names, instance seeds, battery
  fields) must match exactly; the numeric part (``x_opt``, ``R``,
  ``f(x)`` at fixed points) is rounded to 9 significant digits so a BLAS
  kernel that differs in the last bit on another machine cannot fail it.
* **The new batteries have the documented shape** and build fast.
* **The sealed set is disjoint from everything else**, and hard to
  reach by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pytest

from panobbgo import harness_families, harness_ioh
from panobbgo.lib.families import EvaluationCrashed, EvaluationTimedOut, _instance_seed, make_family_instances
from panobbgo.lib.ioh_wrapper import IOHProblem, worker_available
from panobbgo.sealed import (
    SEALED_FAMILY_SEED,
    SEALED_INSTANCE_MIN,
    SEALED_MABBOB_DIMS,
    SEALED_MABBOB_INSTANCES,
)

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Digests of the frozen presets
# ---------------------------------------------------------------------------


def _num(v: float) -> str:
    return f"{float(v):.9e}"


def _family_digest(instances) -> Dict[str, str]:
    """``{"exact": sha, "numeric": sha}`` of a family battery."""
    exact = hashlib.sha256()
    numeric = hashlib.sha256()
    for name, p in instances:
        failure = p.failure.tag() if p.failure is not None else "-"
        exact.update(f"{name}|{p.seed}|{p.family}|{p.dim}|{p.instance}|{p.n_constraints}|{failure}\n".encode())
        rng = np.random.default_rng(12345)
        points = [p.x_opt] + [rng.uniform(-5.0, 5.0, size=p.dim) for _ in range(3)]
        vals: List[str] = [_num(p.f_opt), _num(p.failure_share)]
        vals += [_num(v) for v in p.x_opt]
        rot = p.rotation
        if rot is not None:
            vals += [_num(v) for v in rot[:, 0]]
        for x in points:
            try:
                vals.append(_num(p.eval(x)))
            except EvaluationCrashed:
                vals.append("crash")
            except EvaluationTimedOut:
                vals.append("timeout")
            g = p.eval_constraints(x)
            if g is not None:
                vals += [_num(v) for v in g]
        numeric.update(("|".join(vals) + "\n").encode())
    return {"exact": exact.hexdigest()[:16], "numeric": numeric.hexdigest()[:16]}


def _ioh_digest(spec) -> str:
    fields = (
        spec.name,
        spec.problem_kind,
        spec.dims,
        spec.instances,
        spec.reps,
        spec.budget_multiplier,
        spec.extra_builder_kwargs,
        spec.noise_level,
        spec.noise_resample,
        spec.fids,
    )
    return hashlib.sha256(repr(fields).encode()).hexdigest()[:16]


FAMILY_PRESETS: Dict[str, Callable[..., Any]] = {
    "free": harness_families.make_families_battery,
    "constrained": harness_families.make_constrained_battery,
    "shapes": harness_families.make_shapes_battery,
    "failure": harness_families.make_failure_battery,
}


def _ioh_presets() -> Dict[str, Any]:
    out = {
        "quick": harness_ioh.make_quick_battery(),
        "standard": harness_ioh.make_standard_battery(),
        "full": harness_ioh.make_full_battery(),
        "highdim": harness_ioh.make_highdim_battery(),
        "bbob": harness_ioh.make_bbob_battery(),
    }
    for noise in ("gauss", "unif", "cauchy"):
        for level in ("moderate", "severe"):
            out[f"noisy-{noise}-{level}"] = harness_ioh.make_noisy_battery(noise, level=level)
            out[f"noisy-highdim-{noise}-{level}"] = harness_ioh.make_noisy_highdim_battery(noise, level=level)
    return out


def _run_seed_digest() -> str:
    """Per-run seeds of a few cells: the harness's seed derivation must not move either."""
    seeds = [
        harness_ioh._derive_seed(42, kind, d, i, "RoundRobin_CMAES", 0, None, fid)
        for kind in ("MA-BBOB", "BBOB", "ellipsoid")
        for d in (2, 5, 40)
        for i in (0, 2)
        for fid in (None, 10)
    ]
    return hashlib.sha256(repr(seeds).encode()).hexdigest()[:16]


#: Pinned on master 306a8d3, before this change (``python tests/test_large_and_sealed.py``).
PINNED_FAMILIES = {
    "free": {"exact": "aed3ccf62cae3f9a", "numeric": "2d4bd65589d9f20f"},
    "constrained": {"exact": "f8e97aea1ce7499e", "numeric": "b1149dbdc6b2f713"},
    "shapes": {"exact": "0211a5feaec8d300", "numeric": "2b84af9d670f40a8"},
    "failure": {"exact": "1b0168d7b6d2777d", "numeric": "186dfce55fdc91b8"},
}
PINNED_IOH = {
    "quick": "ae72593ae52c9862",
    "standard": "023695892dc9c24c",
    "full": "401b888e268bb281",
    "highdim": "628f8edb5c112ec1",
    "bbob": "d4364a3731004cb7",
    "noisy-gauss-moderate": "3edd5d3d9d72ad6a",
    "noisy-highdim-gauss-moderate": "bb2d8528c2c1541e",
    "noisy-gauss-severe": "5167bfbdb381dd40",
    "noisy-highdim-gauss-severe": "31ef89f78a0cef71",
    "noisy-unif-moderate": "7f75d803b4fcc740",
    "noisy-highdim-unif-moderate": "a0635a559994361c",
    "noisy-unif-severe": "acd0c4d7cb742ad3",
    "noisy-highdim-unif-severe": "3769056d2f86c51d",
    "noisy-cauchy-moderate": "fe110013452acd9a",
    "noisy-highdim-cauchy-moderate": "cd7b4748cf1b6009",
    "noisy-cauchy-severe": "f94f0d599a35b537",
    "noisy-highdim-cauchy-severe": "b36060bfcc6d3798",
}
PINNED_RUN_SEEDS = "1ee17f6be582bd98"


@pytest.mark.parametrize("preset", sorted(PINNED_FAMILIES))
def test_family_presets_are_bit_identical(preset):
    assert _family_digest(FAMILY_PRESETS[preset]()) == PINNED_FAMILIES[preset]


def test_ioh_presets_and_run_seeds_are_unchanged():
    presets = _ioh_presets()
    assert {k: _ioh_digest(v) for k, v in presets.items()} == PINNED_IOH
    assert not any(b.sealed for b in presets.values())
    assert _run_seed_digest() == PINNED_RUN_SEEDS


# ---------------------------------------------------------------------------
# Dimensions 30/40 and the largescale slice
# ---------------------------------------------------------------------------


def test_large_families_battery_is_the_presets_at_30_40():
    large = harness_families.make_large_families_battery()
    d = harness_families.describe_instances(large)
    assert d["dims"] == [30, 40] and d["n_instances"] == 10 * 2 * 3 and not d["constrained"]
    # The same instances the free / shapes presets build when asked for these dims.
    same = harness_families.make_families_battery(dims=(30, 40)) + harness_families.make_shapes_battery(dims=(30, 40))
    assert [(n, p.seed) for n, p in large] == [(n, p.seed) for n, p in same]
    for _n, p in large:
        assert p.eval(p.x_opt) == p.f_opt


@pytest.mark.parametrize("preset", ["free", "constrained", "shapes", "failure"])
def test_every_preset_builds_fast_at_40(preset):
    """Per-instance setup stays cheap at d = 40 (measured: < 0.1 s, failure regions included)."""
    t0 = time.perf_counter()
    instances = FAMILY_PRESETS[preset](dims=(40,), n_instances=1)
    per_instance = (time.perf_counter() - t0) / len(instances)
    assert per_instance < 2.0  # generous: a CI runner under load
    for _n, p in instances:
        assert p.dim == 40
        assert p.failure_at(p.x_opt) is None and p.eval(p.x_opt) == p.f_opt
        if p.failure is not None:
            assert abs(p.failure_share - p.failure.share) < 0.05


def test_ioh_large_and_largescale_batteries():
    large = harness_ioh.make_large_battery()
    assert large.problem_kind == "MA-BBOB" and large.dims == (30, 40) and large.budget_for(40) == 20000
    ls = harness_ioh.make_largescale_battery()
    assert ls.problem_kind == "BBOB" and ls.dims == (80, 160) and ls.fids == harness_ioh.LARGESCALE_FIDS
    assert [harness_ioh.bbob_class_of(f) for f in ls.fids] == list(harness_ioh.BBOB_CLASS_ORDER)
    assert not large.sealed and not ls.sealed


def _load_cli():
    path = ROOT / "scripts" / "ioh_benchmark.py"
    spec = importlib.util.spec_from_file_location("ioh_benchmark_cli_large", path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_cli_flags_select_the_new_batteries():
    cli = _load_cli()

    def ns(**kw):
        base = dict(
            full=False,
            standard=False,
            noisy=None,
            noisy_highdim=None,
            highdim=False,
            noisy_severe=False,
            reps=None,
            families_quick=False,
            families_constrained=False,
            families=False,
        )
        base.update(kw)
        return argparse.Namespace(**base)

    assert cli._resolve_battery(ns(large=True)).name == "ioh-large"
    assert cli._resolve_battery(ns(largescale=True)).name == "ioh-bbob-largescale"
    assert cli._resolve_battery(ns(sealed=True)).sealed
    assert cli._resolve_battery(ns(sealed=True, reps=2)).sealed  # dataclasses.replace keeps the mark
    name, inst, _bm = cli._resolve_family_battery(ns(families_large=True))
    assert name == "families-large" and len(inst) == 60
    assert cli._resolve_family_battery(ns()) is None


# ---------------------------------------------------------------------------
# The sealed test set
# ---------------------------------------------------------------------------


def test_sealed_mabbob_battery_shape():
    b = harness_ioh.make_sealed_battery()
    assert b.sealed and b.problem_kind == "MA-BBOB" and b.name.startswith("sealed")
    assert b.instances == SEALED_MABBOB_INSTANCES and len(set(b.instances)) == 5
    assert b.dims == SEALED_MABBOB_DIMS and {30, 40} <= set(b.dims)
    assert all(SEALED_INSTANCE_MIN <= i < 2**31 - 1 for i in b.instances)


def test_ioh_development_instances_are_disjoint_from_the_sealed_ones():
    batteries = list(_ioh_presets().values()) + [
        harness_ioh.make_large_battery(),
        harness_ioh.make_largescale_battery(),
    ]
    dev = {int(i) for b in batteries for i in b.instances}
    assert dev and max(dev) < SEALED_INSTANCE_MIN
    assert not dev & set(SEALED_MABBOB_INSTANCES)
    # Every battery builder of the module is covered above; a new one must be added here.
    builders = {
        n for n, _f in inspect.getmembers(harness_ioh, inspect.isfunction) if re.fullmatch(r"make_\w*battery", n)
    }
    assert builders == {
        "make_quick_battery",
        "make_standard_battery",
        "make_full_battery",
        "make_highdim_battery",
        "make_bbob_battery",
        "make_noisy_battery",
        "make_noisy_highdim_battery",
        "make_large_battery",
        "make_largescale_battery",
        "make_sealed_battery",
    }


def test_the_sealed_range_is_refused_outside_the_sealed_battery():
    spec = harness_ioh.IOHBatterySpec
    with pytest.raises(ValueError, match="sealed"):
        spec(name="x", problem_kind="MA-BBOB", dims=(2,), instances=(SEALED_MABBOB_INSTANCES[0],))
    with pytest.raises(ValueError, match="sealed"):
        spec(name="x", problem_kind="MA-BBOB", dims=(2,), instances=(SEALED_INSTANCE_MIN + 5,))
    with pytest.raises(ValueError, match="sealed"):
        spec(name="x", problem_kind="MA-BBOB", dims=(2,), instances=(0,), sealed=True)


def _dev_family_seeds() -> set:
    """Instance seeds of every development family preset at every dim 2..160 and index < 20."""
    labels = {
        cfg.name()
        for group in (
            harness_families.FREE_FAMILIES,
            harness_families.CONSTRAINED_FAMILIES,
            harness_families.SHAPES_FAMILIES,
            harness_families.FAILURE_FAMILIES,
        )
        for cfg in group
    }
    return {
        _instance_seed(harness_families.DEFAULT_BATTERY_SEED, label, dim, index)
        for label in labels
        for dim in range(2, 161)
        for index in range(20)
    }


def test_sealed_families_battery_is_disjoint_and_marked():
    sealed = harness_families.make_sealed_families_battery()
    d = harness_families.describe_instances(sealed)
    assert d["n_instances"] == 10 * 6 * 3 + 8 * 3 * 3
    assert d["dims"] == [2, 5, 10, 20, 30, 40] and d["constrained"] and len(d["failure"]) == 4
    assert all(p.sealed for _n, p in sealed)
    assert SEALED_FAMILY_SEED != harness_families.DEFAULT_BATTERY_SEED
    sealed_seeds = {p.seed for _n, p in sealed}
    assert len(sealed_seeds) == len(sealed)
    assert not sealed_seeds & _dev_family_seeds()
    # Development instances are never marked.
    for build in FAMILY_PRESETS.values():
        assert not any(p.sealed for _n, p in build(dims=(2,), n_instances=1))
    assert not any(p.sealed for _n, p in harness_families.make_large_families_battery(n_instances=1))


def test_the_sealed_family_seed_is_refused_elsewhere():
    with pytest.raises(ValueError, match="sealed"):
        make_family_instances(["sphere"], dims=(2,), seed=SEALED_FAMILY_SEED)
    with pytest.raises(ValueError, match="sealed"):
        harness_families.make_families_battery(seed=SEALED_FAMILY_SEED)
    with pytest.raises(ValueError, match="sealed"):
        make_family_instances(["sphere"], dims=(2,), seed=7, sealed=True)


def test_the_harnesses_print_the_banner(capsys):
    spec = [s for s in harness_ioh.make_ioh_strategies() if s.name == "RoundRobin_Random"]
    sealed = harness_families.make_sealed_families_battery()[:1]
    res = harness_families.run_family_harness(spec, sealed, budget_multiplier=3, progress=False)
    assert len(res.runs) == 1 and res.runs[0].n_evals == 6
    assert "SEALED TEST SET" in capsys.readouterr().err
    plain = harness_families.make_families_battery(dims=(2,), n_instances=1)[:1]
    harness_families.run_family_harness(spec, plain, budget_multiplier=3, progress=False)
    assert "SEALED" not in capsys.readouterr().err
    # IOH: the banner comes before any cell runs (no strategies: no worker needed).
    harness_ioh.run_ioh_harness([], harness_ioh.make_sealed_battery(), progress=False)
    assert "SEALED TEST SET: sealed-mabbob" in capsys.readouterr().err


def test_sealed_ids_appear_in_no_other_code():
    """The sealed identities live in ``panobbgo/sealed.py``; the re-baseline workflow never names the set.

    Code only: a claim's result file (``planning/results/...``) legitimately
    carries the sealed instance ids.
    """
    literals = [str(i) for i in SEALED_MABBOB_INSTANCES] + [str(SEALED_FAMILY_SEED)]
    allowed = {ROOT / "panobbgo" / "sealed.py"}
    scanned = 0
    for sub in ("panobbgo", "benchmarks", "scripts", "tests", "tools", ".github", "sketchpad"):
        for path in (ROOT / sub).rglob("*"):
            if path.suffix not in {".py", ".yml", ".yaml", ".toml", ".sh"}:
                continue
            if path in allowed or ".venv" in path.parts or not path.is_file():
                continue
            scanned += 1
            text = path.read_text(errors="replace")
            for lit in literals:
                assert lit not in text, f"sealed id {lit} appears in {path.relative_to(ROOT)}"
    assert scanned > 50
    for path in [ROOT / "scripts" / "rebaseline.py", *(ROOT / ".github" / "workflows").glob("*.y*ml")]:
        if path.exists():
            assert "sealed" not in path.read_text().lower(), f"{path.relative_to(ROOT)} names the sealed set"


@pytest.mark.skipif(not worker_available(), reason="ioh worker venv not set up")
def test_ioh_builds_sealed_and_largescale_problems():
    for kind, inst, dim, fid in (("MA-BBOB", SEALED_MABBOB_INSTANCES[0], 40, None), ("BBOB", 0, 160, 21)):
        p = IOHProblem(kind, inst, dim, fid=fid)
        try:
            fx = p.eval(np.zeros(dim))
            assert np.isfinite(fx) and fx >= p.optimum_y
        finally:
            p.close()


if __name__ == "__main__":  # pragma: no cover - prints the pinned values
    for key, fn in FAMILY_PRESETS.items():
        print(key, _family_digest(fn()))
    for key, spec in _ioh_presets().items():
        print(key, _ioh_digest(spec))
    print("run seeds", _run_seed_digest())
