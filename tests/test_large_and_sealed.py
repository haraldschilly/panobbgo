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
  ``f(x)`` at fixed points) is rounded to 10 significant digits so a BLAS
  kernel that differs in the last bit on another machine cannot fail it.
* **The new batteries have the documented shape** and build fast.
* **The sealed set is disjoint from everything else**, and hard to
  reach by accident.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pytest

from panobbgo import harness_families, harness_ioh
from panobbgo.harness_ioh import IOHHarnessResult, IOHMultiSeedResult
from panobbgo.lib.families import EvaluationCrashed, EvaluationTimedOut, _instance_seed, make_family_instances
from panobbgo.lib.ioh_wrapper import IOHProblem, worker_available
from panobbgo.local_run import BLAS_THREADS
from panobbgo.sealed import (
    DEV_INSTANCE_LIMIT,
    SEALED_FAMILY_SEED,
    SEALED_INSTANCE_MIN,
    SEALED_MABBOB_DIMS,
    SEALED_MABBOB_INSTANCES,
    alias_residues,
    is_dev_instance_id,
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
    assert ls.budget_for(160) == 500 * 160
    assert not large.sealed and not ls.sealed
    # The widened AOCC range travels with the battery into the result file.
    assert ls.log_hi == harness_ioh.LARGESCALE_LOG_HI == 6.0 and large.log_hi is None
    assert ls.aocc_bounds() == (harness_ioh.AOCC_LOG_LO, 6.0)
    assert large.aocc_bounds() == (harness_ioh.AOCC_LOG_LO, harness_ioh.AOCC_LOG_HI)
    with pytest.raises(ValueError, match="log_hi"):
        ls.aocc_bounds(log_hi=4.0)
    res = harness_ioh.run_ioh_harness([], ls, progress=False)  # no cells: no worker needed
    assert res.log_hi == 6.0 and IOHHarnessResult.from_dict(res.to_dict()).log_hi == 6.0
    multi = harness_ioh.run_ioh_harness_multi_seed([], ls, [1, 2], progress=False)
    assert multi.log_hi == 6.0 and all(r.log_hi == 6.0 for r in multi.results)


def test_compare_refuses_different_aocc_bounds(tmp_path, capsys):
    cli = _load_script("ioh_benchmark")
    wide = tmp_path / "wide.json"
    std = tmp_path / "std.json"
    wide.write_text(harness_ioh.run_ioh_harness([], harness_ioh.make_largescale_battery(), progress=False).to_json())
    std.write_text(harness_ioh.run_ioh_harness([], harness_ioh.make_large_battery(), progress=False).to_json())
    assert cli.main(["compare", str(std), str(wide)]) == 2
    assert "different target ranges" in capsys.readouterr().err


def test_cmaes_smoke_at_d40():
    """One CMA-ES run on a family instance at d = 40, 10*d evaluations: the high-d path end to end (~1 s)."""
    spec = [s for s in harness_ioh.make_ioh_strategies() if s.name == "RoundRobin_CMAES"]
    inst = harness_families.make_families_battery(dims=(40,), n_instances=1)[:1]
    res = harness_families.run_family_harness(spec, inst, budget_multiplier=10, progress=False)
    (rec,) = res.runs
    assert rec.dim == 40 and rec.n_evals == 400 and rec.error is None and np.isfinite(rec.best_fx)
    assert res.blas_threads == BLAS_THREADS == 1 and not res.sealed and not rec.sealed


def test_runs_are_blas_pinned(monkeypatch):
    """Every AOCC run goes through ``_run_tracked``, which pins BLAS to one thread and never lifts the pin.

    Restoring the previous limit after a run raised the OpenBLAS thread
    count while threads abandoned by ``evaluation.timeout`` could still be
    inside BLAS; the pin is process-wide instead (``local_run.pin_blas``).
    """
    import threadpoolctl
    from threadpoolctl import threadpool_info

    seen = []
    real = harness_ioh._run_tracked_unpinned

    def spy(*args, **kwargs):
        seen.extend(lib["num_threads"] for lib in threadpool_info() if lib.get("user_api") == "blas")
        return real(*args, **kwargs)

    restores = []

    class Limits(threadpoolctl.threadpool_limits):
        def restore_original_limits(self):
            restores.append(self)
            return super().restore_original_limits()

    monkeypatch.setattr(harness_ioh, "_run_tracked_unpinned", spy)
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", Limits)
    spec = [s for s in harness_ioh.make_ioh_strategies() if s.name == "RoundRobin_Random"]
    inst = harness_families.make_families_battery(dims=(2,), n_instances=1)[:1]
    harness_families.run_family_harness(spec, inst, budget_multiplier=3, progress=False)
    assert seen and set(seen) == {1}
    assert restores == []  # the limit is never restored (raised) behind a run
    assert set(lib["num_threads"] for lib in threadpool_info() if lib.get("user_api") == "blas") == {1}


def test_pool_workers_run_with_blas_pinned():
    """A spawned ``jobs=2`` worker has BLAS at one thread before any run pins it."""
    from panobbgo.local_run import TaskPool, blas_thread_counts

    with TaskPool(2, niceness=None, min_free_gb=0.0) as pool:
        counts = pool.map(blas_thread_counts, [{}, {}, {}])
    assert all(c and set(c) == {1} for c in counts)


def test_family_harness_in_workers_matches_in_process():
    """``jobs=2`` (pinned workers) gives the same records as in-process (pinned per run)."""
    spec = [s for s in harness_ioh.make_ioh_strategies() if s.name == "RoundRobin_CMAES"]
    inst = harness_families.make_families_battery(dims=(5,), n_instances=1)[:2]
    one = harness_families.run_family_harness(spec, inst, budget_multiplier=20, progress=False)
    two = harness_families.run_family_harness(spec, inst, budget_multiplier=20, progress=False, jobs=2)
    assert [(r.best_fx, r.aocc) for r in one.runs] == [(r.best_fx, r.aocc) for r in two.runs]
    assert two.blas_threads == 1


def test_compare_warns_on_sealed_or_blas_mismatch(tmp_path, capsys):
    cli = _load_script("ioh_benchmark")
    a = harness_ioh.run_ioh_harness([], harness_ioh.make_quick_battery(), progress=False)
    b = dataclasses.replace(a, blas_threads=None, sealed=True)
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(a.to_json())
    pb.write_text(b.to_json())
    cli.main(["compare", str(pa), str(pb)])
    err = capsys.readouterr().err
    assert "sealed mismatch" in err and "blas_threads mismatch" in err


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ns(**kw):
    base = dict(
        full=False,
        standard=False,
        noisy=None,
        noisy_highdim=None,
        highdim=False,
        noisy_severe=False,
        reps=None,
        legacy=False,
        families_quick=False,
        families_constrained=False,
        families=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_flags_select_the_new_batteries():
    cli = _load_script("ioh_benchmark")
    assert cli._resolve_battery(_ns(large=True)).name == "ioh-large"
    assert cli._resolve_battery(_ns(largescale=True)).name == "ioh-bbob-largescale"
    assert cli._resolve_battery(_ns(sealed=True)) == harness_ioh.make_sealed_battery()
    name, inst, bm = cli._resolve_family_battery(_ns(families_large=True))
    assert name == "families-large" and len(inst) == 60 and bm == 500
    name, inst, bm = cli._resolve_family_battery(_ns(families_sealed=True))
    assert name.startswith("sealed") and len(inst) == 252 and all(p.sealed for _n, p in inst) and bm == 500
    assert cli._resolve_family_battery(_ns()) is None


@pytest.mark.parametrize("flag", ["large", "largescale", "families_large", "sealed", "families_sealed"])
def test_cli_refuses_legacy_with_the_large_and_sealed_batteries(flag):
    cli = _load_script("ioh_benchmark")
    with pytest.raises(SystemExit, match="--legacy"):
        cli._check_battery_options(_ns(**{flag: True, "legacy": True}))
    cli._check_battery_options(_ns(**{flag: True}))  # alone: fine


@pytest.mark.parametrize("flag", ["sealed", "families_sealed"])
def test_cli_refuses_reps_on_the_sealed_set(flag):
    cli = _load_script("ioh_benchmark")
    with pytest.raises(SystemExit, match="--reps"):
        cli._check_battery_options(_ns(**{flag: True, "reps": 2}))
    cli._check_battery_options(_ns(large=True, reps=2))  # a development battery takes it


def test_family_screen_has_no_sealed_preset():
    text = (ROOT / "benchmarks" / "family_screen.py").read_text()
    assert "make_sealed" not in text and '"sealed"' not in text


# ---------------------------------------------------------------------------
# The sealed test set
# ---------------------------------------------------------------------------


def test_sealed_mabbob_battery_shape():
    b = harness_ioh.make_sealed_battery()
    assert b.sealed and b.problem_kind == "MA-BBOB" and b.name.startswith("sealed")
    assert b.instances == SEALED_MABBOB_INSTANCES and len(set(b.instances)) == 20
    assert b.dims == SEALED_MABBOB_DIMS and {30, 40} <= set(b.dims)
    assert all(SEALED_INSTANCE_MIN <= i < 2**31 - 1 for i in b.instances)


def test_sealed_ids_follow_the_documented_derivation():
    lo, hi = SEALED_INSTANCE_MIN, 2**31 - 1
    derived = tuple(
        lo
        + int.from_bytes(hashlib.sha256(f"panobbgo-sealed-2026-09-26|mabbob|{k}".encode()).digest()[:8], "little")
        % (hi - lo)
        for k in range(20)
    )
    assert derived == SEALED_MABBOB_INSTANCES


def test_no_development_id_aliases_a_sealed_id():
    """ioh seeds BBOB sub-transforms with ``fid + 10000*id`` in 32 bits: residues mod 2**28 must stay apart."""
    seen: set = set()
    for s in SEALED_MABBOB_INSTANCES:
        res = alias_residues(s)
        assert min(res) >= DEV_INSTANCE_LIMIT  # a development id is its own residue
        assert not res & seen
        seen |= res
    # The collision rule itself: 10000*(i - j) = fid' - fid (mod 2**32) exactly for the listed residues.
    for s in SEALED_MABBOB_INSTANCES[:3]:
        for r in alias_residues(s):
            delta = (10000 * (r - s)) % 2**32
            assert delta in (0, 16, 2**32 - 16)


def test_ioh_development_instances_are_in_the_window():
    batteries = list(_ioh_presets().values()) + [
        harness_ioh.make_large_battery(),
        harness_ioh.make_largescale_battery(),
    ]
    dev = {int(i) for b in batteries for i in b.instances}
    assert dev and all(is_dev_instance_id(i) for i in dev)
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


@pytest.mark.parametrize(
    "bad", [-1, DEV_INSTANCE_LIMIT, 211613860, SEALED_INSTANCE_MIN + 5, SEALED_MABBOB_INSTANCES[0]]
)
def test_ids_outside_the_development_window_are_refused(bad):
    with pytest.raises(ValueError, match="development window"):
        harness_ioh.IOHBatterySpec(name="x", problem_kind="MA-BBOB", dims=(2,), instances=(0, bad))


def test_the_sealed_spec_is_exactly_the_sealed_battery():
    spec = harness_ioh.IOHBatterySpec
    ok = spec(
        name="sealed-mabbob",
        problem_kind="MA-BBOB",
        dims=SEALED_MABBOB_DIMS,
        instances=SEALED_MABBOB_INSTANCES,
        budget_multiplier=500,
        sealed=True,
    )
    assert ok == harness_ioh.make_sealed_battery()
    sealed = harness_ioh.make_sealed_battery()
    for change in (
        dict(instances=SEALED_MABBOB_INSTANCES[:5]),
        dict(instances=tuple(reversed(SEALED_MABBOB_INSTANCES))),
        dict(instances=(0, 1, 2)),
        dict(dims=(2, 5)),
        dict(reps=3),
        dict(budget_multiplier=100),
        dict(name="mine"),
        dict(log_hi=6.0),
    ):
        with pytest.raises(ValueError, match="sealed"):
            dataclasses.replace(sealed, **change)


def test_a_single_run_cannot_reach_the_sealed_set():
    spec = harness_ioh.make_ioh_strategies()[0]
    run = harness_ioh._run_one
    with pytest.raises(ValueError, match="development window"):
        run(spec, "MA-BBOB", 2, SEALED_MABBOB_INSTANCES[0], 0, 10, 1, {}, -8.0, 2.0)
    with pytest.raises(ValueError, match="sealed"):
        run(spec, "MA-BBOB", 2, 0, 0, 10, 1, {}, -8.0, 2.0, sealed=True)
    smoke = _load_script("ioh_smoke")
    for inst in (SEALED_MABBOB_INSTANCES[0], -3):
        with pytest.raises(SystemExit):
            import sys

            argv = sys.argv
            sys.argv = ["ioh_smoke.py", "--instance", str(inst)]
            try:
                smoke.main()
            finally:
                sys.argv = argv


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


@pytest.fixture(scope="module")
def sealed_families():
    return harness_families.make_sealed_families_battery()


def test_sealed_families_battery_is_disjoint_and_marked(sealed_families):
    sealed = sealed_families
    d = harness_families.describe_instances(sealed)
    assert d["n_instances"] == 10 * 6 * 3 + 8 * 3 * 3
    assert d["dims"] == [2, 5, 10, 20, 30, 40] and d["constrained"] and len(d["failure"]) == 4
    assert all(p.sealed for _n, p in sealed)
    assert {(p.family, p.dim, p.instance) for _n, p in sealed} == harness_families.sealed_family_keys()
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


def test_the_sealed_family_set_runs_whole_and_is_marked(sealed_families, capsys):
    run = harness_families.run_family_harness
    plain = harness_families.make_families_battery(dims=(2,), n_instances=1)[:1]
    with pytest.raises(ValueError, match="sub-selection"):
        run([], sealed_families[:10], progress=False)
    with pytest.raises(ValueError, match="mixing"):
        run([], sealed_families + plain, progress=False)
    with pytest.raises(ValueError, match="500"):
        run([], sealed_families, budget_multiplier=100, progress=False)
    with pytest.raises(ValueError, match="reps"):
        run([], sealed_families, reps=2, progress=False)
    capsys.readouterr()
    res = run([], sealed_families, progress=False, battery_name="families")
    assert res.sealed and res.battery_name == "sealed-families"
    assert "SEALED TEST SET: sealed-families" in capsys.readouterr().err
    back = IOHHarnessResult.from_dict(res.to_dict())
    assert back.sealed and back.blas_threads == 1


def test_sealed_runs_are_marked_per_record(monkeypatch, sealed_families):
    """Each record of a sealed run says so (checked on one cell: the others take the same path)."""
    spec = [s for s in harness_ioh.make_ioh_strategies() if s.name == "RoundRobin_Random"]
    rec = harness_families._run_one(spec[0], sealed_families[0][1], 0, 6, 1, -8.0, 2.0, True)
    assert rec.sealed
    plain = harness_families.make_families_battery(dims=(2,), n_instances=1)[0][1]
    assert not harness_families._run_one(spec[0], plain, 0, 6, 1, -8.0, 2.0, True).sealed


def test_the_ioh_harness_prints_the_banner_and_marks_the_result(capsys):
    res = harness_ioh.run_ioh_harness([], harness_ioh.make_sealed_battery(), progress=False)
    assert "SEALED TEST SET: sealed-mabbob" in capsys.readouterr().err
    assert res.sealed and res.blas_threads == 1
    multi = harness_ioh.run_ioh_harness_multi_seed([], harness_ioh.make_sealed_battery(), [1], progress=False)
    assert multi.sealed and IOHMultiSeedResult.from_dict(json.loads(multi.to_json())).sealed
    assert "SEALED TEST SET" in capsys.readouterr().err
    plain = harness_ioh.run_ioh_harness([], harness_ioh.make_quick_battery(), progress=False)
    assert not plain.sealed and "SEALED" not in capsys.readouterr().err


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


# ---------------------------------------------------------------------------
# Fingerprints of the sealed MA-BBOB problems (need the ioh worker)
# ---------------------------------------------------------------------------


def _sealed_fingerprint() -> str:
    """Digest of every sealed MA-BBOB problem at d 2 and 40: ``f_opt`` and ``f`` at three fixed points.

    ``f`` at a point depends on the weights, ``x_opt`` and every
    sub-problem transform, so an ``ioh`` upgrade that redraws any of them
    changes the digest.  Rounded to 10 significant digits.
    """
    h = hashlib.sha256()
    for inst in SEALED_MABBOB_INSTANCES:
        for dim in (2, 40):
            p = IOHProblem("MA-BBOB", inst, dim)
            try:
                rng = np.random.default_rng(2026)
                vals = [p.optimum_y] + [p.eval(rng.uniform(-5.0, 5.0, size=dim)) for _ in range(3)]
            finally:
                p.close()
            h.update(("|".join(_num(v) for v in vals) + "\n").encode())
    return h.hexdigest()[:16]


#: Pinned with ioh 0.3.22 (``tools/ioh_worker/uv.lock``), 2026-09-26.
PINNED_SEALED_FINGERPRINT = "4aa1e64d20e42bfa"


@pytest.mark.skipif(not worker_available(), reason="ioh worker venv not set up")
def test_sealed_mabbob_problems_are_pinned():
    assert _sealed_fingerprint() == PINNED_SEALED_FINGERPRINT


@pytest.mark.skipif(not worker_available(), reason="ioh worker venv not set up")
def test_ioh_builds_largescale_problems():
    p = IOHProblem("BBOB", 0, 160, fid=21)
    try:
        fx = p.eval(np.zeros(160))
        assert np.isfinite(fx) and fx >= p.optimum_y
    finally:
        p.close()


if __name__ == "__main__":  # pragma: no cover - prints the pinned values
    for key, fn in FAMILY_PRESETS.items():
        print(key, _family_digest(fn()))
    for key, spec in _ioh_presets().items():
        print(key, _ioh_digest(spec))
    print("run seeds", _run_seed_digest())
    if worker_available():
        print("sealed fingerprint", _sealed_fingerprint())
