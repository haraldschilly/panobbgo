#!/usr/bin/env python
# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
"""Expensive-track measurement: panobbgo against the incumbents with q parallel workers.

The engine behind ``.github/workflows/measure.yml`` (roadmap
``planning/DESIGN_roadmap_2026-09-26.md`` §5.2; ``doc/dev/benchmarking.md``,
"Expensive-track measurement").  It runs the family track at small budgets
(``bm``·d evaluations, ``bm`` in {20, 100}) on the virtual clock (async
policy, log-normal durations, sigma 0.5) with q simulated workers, and scores
AOCC over evaluations and ``aocc_time`` over virtual time.  Subcommands:

``plan``
    Print the job matrix (JSON).  Stdlib only, so the planning job needs no
    dependencies::

        python scripts/measure.py plan --seeds 5

``run``
    Run the units of one shard and write one result file per unit (written
    as each unit finishes, so a cut job keeps what it finished) plus
    ``meta_<shard>.json`` with the host's CPU::

        uv run python scripts/measure.py run --units core.free.b20.q4.d2.s42 --shard core-01 --out-dir out

``aggregate``
    Turn downloaded shard artifacts into ``summary.md`` / ``summary.json``:
    per (preset, dim, budget, q) every strategy's mean AOCC and ``aocc_time``,
    and for every panobbgo spec the paired-seed CI of its delta against the
    best external baseline of that cell::

        gh run download RUN_ID --pattern 'measure-*' --dir measure-raw
        uv run python scripts/measure.py aggregate measure-raw --out-dir measure-summary

A **unit** is one strategy group on one cell and one base seed,
``<group>.<preset>.b<bm>.q<q>.d<dim>.s<seed>``: every instance of the preset
at that dimension.  The groups:

``core``
    The panobbgo specs (``make_ioh_strategies``), the cheap-track external
    baselines (pycma IPOP/BIPOP, NGOpt, Optuna CmaEs/TPE) and Py-BOBYQA, the
    local reference.  All of them run in one process, so their paired
    comparisons share one floating-point environment.
``qLogEI``, ``TuRBO1``, ``SMAC``
    The GP-based baselines, each in shards of its own (they cost minutes per
    run).  A comparison against them crosses jobs, so it is only as exact as
    the two jobs' FP environments agree: the summary flags it ``≈`` and every
    shard records its CPU.  SMAC has no batch acquisition and runs at q = 1
    only; qLogEI and TuRBO are left out where a run would not fit a job
    (:data:`COVERAGE`).

Parallelism: q ∈ ``--qs`` with ``q <= bm`` (at least ``dim`` full rounds of q
calls), so q = 64 runs at 100·d only.  Seeds: the first N of the decision
roster.  Every run is seeded and deterministic on one host, with no
wall-clock limit (the GP baselines are exempt from it anyway).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebaseline import resolve_seeds  # noqa: E402  (stdlib only, like this module's plan)

#: The cheap-track external baselines of the core group (the ``baselines`` extra).
CHEAP_EXTERNALS: Tuple[str, ...] = (
    "Baseline_pycma_IPOP",
    "Baseline_pycma_BIPOP",
    "Baseline_NGOpt",
    "Baseline_Optuna_CmaEs",
    "Baseline_Optuna_TPE",
)

#: The local model-based reference; cheap (under a few seconds a run), so it
#: runs with the core group, in the same process as the panobbgo specs.
LOCAL_REFERENCE = "Baseline_PyBOBYQA"

#: Group -> the strategy names it runs (``core``: the panobbgo specs too,
#: resolved at run time from ``make_ioh_strategies``).
GROUPS: Dict[str, Tuple[str, ...]] = {
    "core": CHEAP_EXTERNALS + (LOCAL_REFERENCE,),
    "qLogEI": ("Baseline_BoTorch_qLogEI",),
    "TuRBO1": ("Baseline_TuRBO1",),
    "SMAC": ("Baseline_SMAC_BB",),
}

#: The GP groups (shards of their own, comparisons flagged as cross-job).
BO_GROUPS: Tuple[str, ...] = ("qLogEI", "TuRBO1", "SMAC")

#: Presets of the family track this measurement knows, with their dimensions.
PRESET_DIMS: Dict[str, Tuple[int, ...]] = {"free": (2, 5, 10), "failure": (2, 5)}

#: Instances per (family, dim) of both presets (the preset default), and the
#: families per preset: the number of runs of a unit is their product.
N_INSTANCES = 3
PRESET_FAMILIES: Dict[str, int] = {"free": 5, "failure": 4}

#: The virtual clock of every run: async policy, log-normal durations.
DURATION = "lognormal"
SIGMA = 0.5


#: Measured seconds of ONE run of a GP baseline at q = 1, per (dim, budget = bm*dim), on a
#: 16-core laptop (2026-09-26, light load, ``nice -n 15``, one BLAS thread, one run at a
#: time, the ellipsoid family).  ``(10, 200)`` qLogEI and SMAC are the guide's numbers
#: (measured under heavy load); SMAC ``(5, 100)`` is interpolated and ``(5, 500)`` a lower
#: bound (stopped unfinished after 40 min); the rest are measured.
#: Other cells extrapolate from the same dimension (:func:`laptop_seconds`).
LAPTOP_SECONDS: Dict[str, Dict[Tuple[int, int], float]] = {
    "qLogEI": {(2, 40): 19.0, (2, 200): 87.0, (5, 100): 74.0, (5, 500): 972.0, (10, 200): 1000.0},
    "TuRBO1": {(2, 40): 9.4, (2, 200): 52.0, (5, 100): 22.0, (5, 500): 146.0, (10, 200): 56.0},
    "SMAC": {(2, 40): 9.5, (2, 200): 107.0, (5, 100): 45.0, (5, 500): 1800.0 * 1.2, (10, 200): 390.0},
}

#: A GP baseline runs on a cell only if one run is estimated at most this long on the
#: laptop (half an hour): a unit of 15 runs is then about 4 rounds x 45 min on a
#: 4-core runner, below the 330-minute step limit.
MAX_RUN_SECONDS = 1800.0

#: A GitHub runner core is taken to be this much slower than the laptop's.
RUNNER_FACTOR = 1.5

#: Growth of a GP baseline's run time with the budget at a fixed dimension (between
#: linear, a fixed cost per proposal, and quadratic, a fit that grows with the data).
BUDGET_EXPONENT = 1.6


def laptop_seconds(group: str, dim: int, budget: int, q: int = 1) -> float:
    """Estimated laptop seconds of one run of ``group`` (the core group: all its specs on one instance).

    GP groups: the measured entry of :data:`LAPTOP_SECONDS`, else the entry of
    the same (or the nearest larger) dimension with the nearest budget, scaled
    by ``(budget / b) ** BUDGET_EXPONENT``.  qLogEI and SMAC fit once per
    evaluation on the async clock at any q; TuRBO proposes once per batch of
    q, measured at about ``0.1 + 0.9 / q`` of its q = 1 time.
    """
    if group == "core":
        # All 10 specs on one instance: about 15 s at d = 10, 1000 evaluations (NGOpt, TPE, Py-BOBYQA dominate).
        return 2.0 + 0.015 * budget
    table = LAPTOP_SECONDS[group]
    if (dim, budget) in table:
        t = table[(dim, budget)]
    else:
        dims = sorted({d for d, _ in table})
        ref_dim = next((d for d in dims if d >= dim), dims[-1])
        ref_b = min((b for d, b in table if d == ref_dim), key=lambda b: abs(math.log(b / budget)))
        t = table[(ref_dim, ref_b)] * (budget / ref_b) ** BUDGET_EXPONENT
    if group == "TuRBO1":
        t *= 0.1 + 0.9 / q
    return t


def covered(group: str, bm: int, dim: int, q: int) -> bool:
    """Whether ``group`` runs on the cell ``(bm, dim, q)``.

    The core group runs everywhere.  A GP baseline runs where one run is
    estimated below :data:`MAX_RUN_SECONDS` at q = 1: with the table above,
    everything but d = 10 at 100·d for qLogEI (about 3.6 h a run) and
    SMAC, and SMAC at d = 5, 100·d (over 30 min a run).  SMAC has no batch
    acquisition, so it runs at q = 1 only (the guide reports it there).
    """
    if group == "core":
        return True
    if group not in LAPTOP_SECONDS:
        raise ValueError(f"unknown group {group!r}")
    if group == "SMAC" and q != 1:
        return False
    return laptop_seconds(group, dim, bm * dim) <= MAX_RUN_SECONDS


def run_seconds(group: str, dim: int, budget: int, q: int) -> float:
    """Estimated seconds of one run on a GitHub runner core (sizes the shards only: never changes a number)."""
    return RUNNER_FACTOR * laptop_seconds(group, dim, budget, q)


# ---------------------------------------------------------------------------
# units and the plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Unit:
    """One strategy group on one cell (preset, budget multiplier, q, dim) and one base seed."""

    group: str
    preset: str
    bm: int
    q: int
    dim: int
    seed: int

    @property
    def id(self) -> str:
        """``<group>.<preset>.b<bm>.q<q>.d<dim>.s<seed>`` (also the result file's stem)."""
        return f"{self.group}.{self.preset}.b{self.bm}.q{self.q}.d{self.dim}.s{self.seed}"

    @classmethod
    def parse(cls, text: str) -> "Unit":
        """The inverse of :attr:`id`."""
        try:
            group, preset, b, q, d, s = text.strip().split(".")
            if not (b[0] == "b" and q[0] == "q" and d[0] == "d" and s[0] == "s"):
                raise ValueError
            unit = cls(group, preset, int(b[1:]), int(q[1:]), int(d[1:]), int(s[1:]))
        except (ValueError, IndexError):
            raise ValueError(f"not a unit id: {text!r} (want <group>.<preset>.b<bm>.q<q>.d<dim>.s<seed>)") from None
        if unit.group not in GROUPS or unit.preset not in PRESET_DIMS:
            raise ValueError(f"unknown group or preset in {text!r}")
        return unit

    @property
    def n_runs(self) -> int:
        """Instances of the unit (every group runs each of them once per strategy)."""
        return PRESET_FAMILIES[self.preset] * N_INSTANCES


def int_list(spec: str) -> List[int]:
    """``"1,4,16"`` -> ``[1, 4, 16]``."""
    return [int(x) for x in spec.replace(" ", "").split(",") if x]


def make_units(
    seeds: Sequence[int],
    presets: Sequence[str],
    bms: Sequence[int],
    dims: Optional[Sequence[int]],
    qs: Sequence[int],
    groups: Sequence[str],
) -> List[Unit]:
    """Every unit of the grid that :func:`covered` admits, with ``q <= bm``."""
    units = []
    for preset in presets:
        if preset not in PRESET_DIMS:
            raise ValueError(f"unknown preset {preset!r}; known: {', '.join(PRESET_DIMS)}")
        pdims = [d for d in PRESET_DIMS[preset] if dims is None or d in dims]
        for group in groups:
            if group not in GROUPS:
                raise ValueError(f"unknown group {group!r}; known: {', '.join(GROUPS)}")
            for bm in bms:
                for q in qs:
                    if q > bm:
                        continue
                    for dim in pdims:
                        if not covered(group, bm, dim, q):
                            continue
                        units.extend(Unit(group, preset, bm, q, dim, s) for s in seeds)
    return units


def unit_minutes(unit: Unit, jobs: int) -> float:
    """Estimated wall minutes of a unit: its runs spread over ``jobs`` processes, in rounds."""
    return math.ceil(unit.n_runs / jobs) * run_seconds(unit.group, unit.dim, unit.bm * unit.dim, unit.q) / 60.0


def pack(units: Sequence[Unit], jobs: int, target_minutes: float) -> List[List[Unit]]:
    """Pack the units of one group into shards of about ``target_minutes`` wall time on ``jobs`` cores.

    First-fit decreasing on :func:`unit_minutes`; a unit longer than the
    target gets a shard of its own.
    """
    bins: List[Tuple[float, List[Unit]]] = []
    for u in sorted(units, key=lambda u: (-unit_minutes(u, jobs), u)):
        w = unit_minutes(u, jobs)
        for i, (load, items) in enumerate(bins):
            if load + w <= target_minutes:
                bins[i] = (load + w, items + [u])
                break
        else:
            bins.append((w, [u]))
    return [sorted(items) for _, items in bins]


def shard_minutes(units: Sequence[Unit], jobs: int) -> float:
    """Estimated wall minutes of a shard: its units one after the other."""
    return sum(unit_minutes(u, jobs) for u in units)


#: Wall-minute target of a core shard: smaller than the GP shards', so the
#: panobbgo side (every paired comparison's one side) is spread over a few jobs.
CORE_TARGET_MINUTES = 30.0


def plan(units: Sequence[Unit], jobs: int, target_minutes: float) -> List[Dict[str, Any]]:
    """The matrix entries: one per shard, the core group first."""
    entries = []
    for group in GROUPS:
        target = min(target_minutes, CORE_TARGET_MINUTES) if group == "core" else target_minutes
        packed = pack([u for u in units if u.group == group], jobs, target)
        for i, items in enumerate(packed, 1):
            entries.append(
                {
                    "shard": f"{group}-{i:02d}",
                    "group": group,
                    "units": ";".join(u.id for u in items),
                    "n_units": len(items),
                    "est_min": int(math.ceil(shard_minutes(items, jobs))),
                }
            )
    return entries


def cmd_plan(args: argparse.Namespace) -> int:
    groups = list(GROUPS) if args.groups in ("", "all") else [g.strip() for g in args.groups.split(",") if g.strip()]
    units = make_units(
        resolve_seeds(args.seeds),
        [p.strip() for p in args.presets.split(",") if p.strip()],
        int_list(args.budgets),
        int_list(args.dims) if args.dims else None,
        int_list(args.qs),
        groups,
    )
    if not units:
        print("error: the grid is empty", file=sys.stderr)
        return 2
    entries = plan(units, args.runner_cores, args.target_minutes)
    if len(entries) > 256:
        print(f"error: {len(entries)} shards; a GitHub matrix takes at most 256", file=sys.stderr)
        return 2
    print(json.dumps({"include": entries}, separators=(",", ":")))
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def host_info() -> Dict[str, Any]:
    """The CPU and FP-relevant environment of this host (recorded in every result file)."""
    info: Dict[str, Any] = {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "runner": os.environ.get("RUNNER_NAME"),
        "env": {
            k: os.environ.get(k)
            for k in (
                "OPENBLAS_CORETYPE",
                "NPY_DISABLE_CPU_FEATURES",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
            if os.environ.get(k) is not None
        },
    }
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
        flags = next((line.split(":", 1)[1].split() for line in cpuinfo.splitlines() if line.startswith("flags")), [])
        info["avx512f"] = "avx512f" in flags
        info["avx2"] = "avx2" in flags
    except OSError:
        info["cpu_model"] = platform.processor() or None
    try:
        info["lscpu"] = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        info["lscpu"] = None
    return info


def fp_class(host: Dict[str, Any]) -> str:
    """A short label of a host's FP class: CPU model and whether it has AVX-512."""
    return f"{host.get('cpu_model') or '?'}{' +avx512' if host.get('avx512f') else ''}"


def _strategies(group: str) -> List[Any]:
    from panobbgo.harness_baselines import make_baseline_strategies
    from panobbgo.harness_ioh import make_ioh_strategies

    names = list(GROUPS[group])
    specs = list(make_ioh_strategies()) if group == "core" else []
    by_name = {s.name: s for s in make_baseline_strategies(names)}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"unknown baseline names {missing}")
    return specs + [by_name[n] for n in names]


def _instances(preset: str, dim: int) -> List[Any]:
    from panobbgo.harness_families import make_failure_battery, make_families_battery

    make = {"free": make_families_battery, "failure": make_failure_battery}[preset]
    return list(make(dims=(dim,), n_instances=N_INSTANCES))


def run_unit(unit: Unit, jobs: int, progress: bool = False) -> Dict[str, Any]:
    """Run one unit and return its result (an ``IOHHarnessResult`` dict) with the unit's key."""
    from panobbgo.harness_families import run_family_harness
    from panobbgo.virtual_clock import VirtualSpec

    t0 = time.time()
    result = run_family_harness(
        _strategies(unit.group),
        _instances(unit.preset, unit.dim),
        budget_multiplier=unit.bm,
        base_seed=unit.seed,
        sync_eval=True,
        progress=progress,
        battery_name=f"measure-{unit.preset}-b{unit.bm}",
        jobs=jobs,
        virtual=VirtualSpec(workers=unit.q, duration=DURATION, sigma=SIGMA, policy="async"),
    )
    return {
        "unit": unit.id,
        "group": unit.group,
        "preset": unit.preset,
        "bm": unit.bm,
        "q": unit.q,
        "dim": unit.dim,
        "seed": unit.seed,
        "elapsed_s": time.time() - t0,
        "result": result.to_dict(),
    }


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GITHUB_SHA", "unknown")


def cmd_run(args: argparse.Namespace) -> int:
    try:  # the FP pin (PR #359), before numpy loads; a no-op until it exists
        import panobbgo.fp_pin  # noqa: F401  # pyright: ignore[reportMissingImports, reportUnusedImport]
    except ImportError:
        pass
    from panobbgo import local_run

    if not args.no_nice:
        local_run.be_nice()
    local_run.pin_blas_env()  # before torch loads: it reads OMP_NUM_THREADS once
    local_run.blas_limit()
    units = [Unit.parse(u) for u in args.units.split(";") if u.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs or os.cpu_count() or 1
    host = host_info()
    meta: Dict[str, Any] = {
        "shard": args.shard,
        "units": [u.id for u in units],
        "done": [],
        "failed": [],
        "git_sha": _git_sha(),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "started": datetime.now(tz=timezone.utc).isoformat(),
        "jobs": jobs,
        "host": host,
    }
    meta_path = out_dir / f"meta_{args.shard}.json"
    print(f"shard {args.shard}: {len(units)} unit(s) on {fp_class(host)}, {jobs} process(es)", flush=True)
    status = 0
    for i, unit in enumerate(units, 1):
        print(f"[{i}/{len(units)}] {unit.id}", flush=True)
        try:
            payload = run_unit(unit, jobs, progress=args.progress)
        except Exception as exc:  # noqa: BLE001 — record, go on with the next unit
            print(f"error: {unit.id}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            meta["failed"].append(unit.id)
            status = 1
        else:
            payload.update(shard=args.shard, host=host, git_sha=meta["git_sha"])
            (out_dir / f"{unit.id}.json").write_text(json.dumps(payload, default=float))
            meta["done"].append(unit.id)
            print(f"    done in {payload['elapsed_s'] / 60:.1f} min", flush=True)
        # Rewritten after every unit: a job cut by its timeout still says what it finished.
        meta["finished"] = datetime.now(tz=timezone.utc).isoformat()
        meta["status"] = status
        meta_path.write_text(json.dumps(meta, indent=2))
    return status


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

#: A cell of the summary: (preset, dim, bm, q).
Cell = Tuple[str, int, int, int]
#: A paired observation: (seed, family, instance).
Key = Tuple[int, str, int]


@dataclass
class Obs:
    """One run's scores and where it ran."""

    aocc: float
    aocc_time: Optional[float]
    shard: str
    fp: str
    error: Optional[str]
    elapsed_s: float


def load_units(src: Path) -> List[Dict[str, Any]]:
    """Every unit result file under ``src`` (the ``meta_*`` files excluded)."""
    out = []
    for p in sorted(src.rglob("*.json")):
        if p.name.startswith("meta_") or p.name.startswith("summary"):
            continue
        d = json.loads(p.read_text())
        if isinstance(d, dict) and "unit" in d and "result" in d:
            out.append(d)
    return out


def collect(payloads: Iterable[Dict[str, Any]]) -> Dict[Cell, Dict[str, Dict[Key, Obs]]]:
    """``cell -> strategy -> (seed, family, instance) -> Obs``; a unit seen twice is an error."""
    cells: Dict[Cell, Dict[str, Dict[Key, Obs]]] = {}
    seen = set()
    for d in payloads:
        if d["unit"] in seen:
            raise ValueError(f"unit {d['unit']} appears twice")
        seen.add(d["unit"])
        cell: Cell = (d["preset"], int(d["dim"]), int(d["bm"]), int(d["q"]))
        fp = fp_class(d.get("host") or {})
        for r in d["result"]["runs"]:
            key: Key = (int(d["seed"]), str(r["problem_kind"]), int(r["instance"]))
            cells.setdefault(cell, {}).setdefault(r["strategy_name"], {})[key] = Obs(
                aocc=float(r["aocc"]),
                aocc_time=None if r.get("aocc_time") is None else float(r["aocc_time"]),
                shard=str(d.get("shard")),
                fp=fp,
                error=r.get("error"),
                elapsed_s=float(r.get("elapsed_s") or 0.0),
            )
    return cells


def _value(o: Obs, metric: str) -> Optional[float]:
    return o.aocc if metric == "aocc" else o.aocc_time


def seed_means(obs: Dict[Key, Obs], metric: str, keys: Optional[Iterable[Key]] = None) -> Dict[int, float]:
    """Per base seed, the mean of ``metric`` over the instances (``keys``: only those)."""
    acc: Dict[int, List[float]] = {}
    for k in keys if keys is not None else obs:
        v = _value(obs[k], metric)
        if v is not None:
            acc.setdefault(k[0], []).append(v)
    return {s: sum(v) / len(v) for s, v in sorted(acc.items())}


def mean_over_seeds(obs: Dict[Key, Obs], metric: str) -> Optional[float]:
    """The mean over seeds of the per-seed instance means (``None`` without values)."""
    m = seed_means(obs, metric)
    return sum(m.values()) / len(m) if m else None


def is_external(name: str) -> bool:
    """External baselines are the ``Baseline_*`` specs; the rest are panobbgo's."""
    return name.startswith("Baseline_")


def paired(
    a: Dict[Key, Obs], b: Dict[Key, Obs], metric: str, t_ci: Callable[[Sequence[float]], Tuple[float, float]]
) -> Dict[str, Any]:
    """Paired-seed delta ``a - b`` of ``metric`` over their common (seed, instance) runs.

    Per seed the mean delta over the common instances, then a t-CI95 over
    seeds (``harness_ioh.t_ci``).  ``cross_job``: some pair ran in two
    different jobs, so its two sides may have seen different FP
    environments; ``cross_fp``: in two different CPU classes.
    """
    common = [k for k in a if k in b and _value(a[k], metric) is not None and _value(b[k], metric) is not None]
    per_seed: Dict[int, List[float]] = {}
    for k in common:
        per_seed.setdefault(k[0], []).append(float(_value(a[k], metric)) - float(_value(b[k], metric)))  # type: ignore[arg-type]
    deltas = [sum(v) / len(v) for _, v in sorted(per_seed.items())]
    mean, half = t_ci(deltas)
    return {
        "n_seeds": len(deltas),
        "n_pairs": len(common),
        "delta": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "wins": sum(1 for d in deltas if d > 0),
        "cross_job": any(a[k].shard != b[k].shard for k in common),
        "cross_fp": any(a[k].fp != b[k].fp for k in common),
    }


METRICS: Tuple[str, ...] = ("aocc", "aocc_time")


def summarize_cell(strats: Dict[str, Dict[Key, Obs]], t_ci: Callable[..., Tuple[float, float]]) -> Dict[str, Any]:
    """Means per strategy and, per metric, every panobbgo spec against the cell's best external baselines.

    Two references per metric: the best external overall (``vs_best_*``; a
    GP baseline ran in another job, so that comparison may cross FP
    environments) and the best external of the core group (``vs_core_*``),
    which ran in the same process as the panobbgo specs: an exact pairing.
    """
    rows: Dict[str, Any] = {}
    for name, obs in strats.items():
        rows[name] = {
            "external": is_external(name),
            "n_seeds": len({k[0] for k in obs}),
            "n_runs": len(obs),
            "errors": sum(1 for o in obs.values() if o.error),
            "seconds_per_run": sum(o.elapsed_s for o in obs.values()) / len(obs) if obs else None,
            **{m: mean_over_seeds(obs, m) for m in METRICS},
        }
    best: Dict[str, Optional[str]] = {}
    best_core: Dict[str, Optional[str]] = {}
    for m in METRICS:
        for ref, pool, key in (
            (best, [n for n in strats if is_external(n)], "vs_best"),
            (best_core, [n for n in strats if n in GROUPS["core"]], "vs_core"),
        ):
            ext = [(rows[n][m], n) for n in pool if rows[n][m] is not None]
            ref[m] = max(ext)[1] if ext else None
            for name in strats:
                if not is_external(name) and ref[m] is not None:
                    rows[name][f"{key}_{m}"] = paired(strats[name], strats[ref[m]], m, t_ci)  # type: ignore[index]
    return {"best_external": best, "best_core_external": best_core, "strategies": rows}


def aggregate(src: Path, planned_units: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """The summary of every unit result under ``src`` (``planned_units``: report the missing ones)."""
    from panobbgo.harness_ioh import t_ci

    payloads = load_units(src)
    if not payloads:
        raise ValueError(f"no unit result files under {src}")
    metas = [json.loads(p.read_text()) for p in sorted(src.rglob("meta_*.json"))]
    cells = collect(payloads)
    done = {d["unit"] for d in payloads}
    missing = sorted(set(planned_units or []) - done)
    fp_by_shard = {str(d.get("shard")): fp_class(d.get("host") or {}) for d in payloads}
    return {
        "created": datetime.now(tz=timezone.utc).isoformat(),
        "git_sha": sorted({str(d.get("git_sha")) for d in payloads}),
        "github_run_id": sorted({str(m.get("github_run_id")) for m in metas if m.get("github_run_id")}),
        "units": len(payloads),
        "missing_units": missing,
        "failed_units": sorted({u for m in metas for u in m.get("failed", [])}),
        "fp_classes": {
            fp: sorted(s for s, f in fp_by_shard.items() if f == fp) for fp in sorted(set(fp_by_shard.values()))
        },
        "virtual": {"duration": DURATION, "sigma": SIGMA, "policy": "async"},
        "cells": {
            f"{p}/d{d}/b{bm}/q{q}": {
                "preset": p,
                "dim": d,
                "bm": bm,
                "q": q,
                **summarize_cell(cells[(p, d, bm, q)], t_ci),
            }
            for (p, d, bm, q) in sorted(cells)
        },
    }


def _fmt(x: Optional[float], nd: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _fmt_delta(st: Optional[Dict[str, Any]]) -> str:
    if not st or st["n_seeds"] == 0:
        return "–"
    lo, hi = st["ci_low"], st["ci_high"]
    ci = "" if math.isnan(lo) else f" [{lo:+.3f},{hi:+.3f}]"
    mark = " ≈" if st["cross_job"] else ""
    return f"{st['delta']:+.3f}{ci} {st['wins']}/{st['n_seeds']}{mark}"


def summary_markdown(summary: Dict[str, Any]) -> str:
    """The summary as Markdown: an overview line per cell, then one table per cell."""
    lines = [
        "# Expensive-track measurement",
        "",
        f"Commit {', '.join(s[:7] for s in summary['git_sha'])}; run {', '.join(summary['github_run_id']) or 'local'}; "
        f"{summary['units']} unit(s); missing {len(summary['missing_units'])}, failed {len(summary['failed_units'])}.",
        f"Virtual clock: async policy, {summary['virtual']['duration']} durations "
        f"(sigma {summary['virtual']['sigma']}); `aocc_time` over the horizon budget/q mean durations.",
        "",
        "Δ = panobbgo spec − the cell's best external baseline (by that metric's mean): mean over seeds of the "
        "per-seed instance-mean delta, t-CI95 over seeds, wins/seeds.  **≈** marks a comparison across jobs "
        "(the GP baselines run in shards of their own): the two sides may have seen different FP environments, "
        "so read it as approximate.  *Δ same-job* is against the best external of the core group (pycma, "
        "NGOpt, Optuna, Py-BOBYQA), which ran in the same process as the panobbgo specs: exact.  The best "
        "external is a selected maximum of several baselines, which favours the baseline side.",
        "",
        "FP classes (CPU per shard): "
        + "; ".join(f"{fp}: {len(shards)} shard(s)" for fp, shards in summary["fp_classes"].items()),
        "",
        "## Overview",
        "",
        "| cell | best ext (AOCC) | best panobbgo | Δ AOCC | Δ same-job | best ext (time) | best panobbgo "
        "| Δ aocc_time | Δ same-job |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, c in summary["cells"].items():
        cols = [name]
        for m in METRICS:
            rows = c["strategies"]
            b = c["best_external"][m]
            own = [(r[m], n) for n, r in rows.items() if not r["external"] and r[m] is not None]
            top = max(own)[1] if own else None
            cols.append(f"{b.removeprefix('Baseline_')} {_fmt(rows[b][m])}" if b else "–")
            cols.append(f"{top} {_fmt(rows[top][m])}" if top else "–")
            cols.append(_fmt_delta(rows[top].get(f"vs_best_{m}")) if top else "–")
            cols.append(_fmt_delta(rows[top].get(f"vs_core_{m}")) if top else "–")
        lines.append("| " + " | ".join(cols) + " |")
    for name, c in summary["cells"].items():
        lines += [
            "",
            f"### {name}",
            "",
            f"best external: AOCC {c['best_external']['aocc']}, aocc_time {c['best_external']['aocc_time']}; "
            f"same job: AOCC {c['best_core_external']['aocc']}, aocc_time {c['best_core_external']['aocc_time']}",
            "",
            "| strategy | seeds | AOCC | Δ vs best ext | Δ same-job | aocc_time | Δ vs best ext | Δ same-job "
            "| errors | s/run |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        rows = c["strategies"]
        for n in sorted(rows, key=lambda n: -(rows[n]["aocc"] or 0.0)):
            r = rows[n]
            lines.append(
                f"| {n} | {r['n_seeds']} | {_fmt(r['aocc'])} | {_fmt_delta(r.get('vs_best_aocc'))} | "
                f"{_fmt_delta(r.get('vs_core_aocc'))} | {_fmt(r['aocc_time'])} | "
                f"{_fmt_delta(r.get('vs_best_aocc_time'))} | {_fmt_delta(r.get('vs_core_aocc_time'))} | "
                f"{r['errors']} | {_fmt(r['seconds_per_run'], 1)} |"
            )
    if summary["missing_units"] or summary["failed_units"]:
        lines += ["", "## Missing / failed units", ""]
        lines += [f"- missing: {u}" for u in summary["missing_units"]]
        lines += [f"- failed: {u}" for u in summary["failed_units"]]
    return "\n".join(lines) + "\n"


def cmd_aggregate(args: argparse.Namespace) -> int:
    planned = None
    if args.plan:
        data = json.loads(Path(args.plan).read_text())
        entries = data["include"] if isinstance(data, dict) else data
        planned = [u for e in entries for u in e["units"].split(";") if u]
    summary = aggregate(Path(args.src), planned)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    md = summary_markdown(summary)
    (out / "summary.md").write_text(md)
    print(md)
    if summary["missing_units"] or summary["failed_units"]:
        print(
            f"warning: {len(summary['missing_units'])} missing and {len(summary['failed_units'])} failed unit(s)",
            file=sys.stderr,
        )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("plan", help="Print the job matrix (JSON).")
    pl.add_argument("--seeds", default="5", help="A count (the first N of the decision roster) or a comma list.")
    pl.add_argument("--presets", default="free", help="Comma list of: free, failure.")
    pl.add_argument("--budgets", default="20,100", help="Budget multipliers (budget = bm*dim).")
    pl.add_argument("--dims", default="", help="Restrict the presets' dimensions (default: all of them).")
    pl.add_argument("--qs", default="1,4,16,64", help="Virtual worker counts; q <= bm only.")
    pl.add_argument("--groups", default="all", help=f"'all' or a comma list of: {', '.join(GROUPS)}.")
    pl.add_argument("--runner-cores", type=int, default=4, help="Processes per runner (default 4).")
    pl.add_argument("--target-minutes", type=float, default=180.0, help="Estimated wall minutes per shard.")
    pl.set_defaults(func=cmd_plan)

    rn = sub.add_parser("run", help="Run the units of one shard.")
    rn.add_argument("--units", required=True, help="';'-joined unit ids.")
    rn.add_argument("--shard", default="local", help="Shard name (the meta file's suffix).")
    rn.add_argument("--out-dir", required=True)
    rn.add_argument("--jobs", type=int, default=0, help="Worker processes (default: all cores).")
    rn.add_argument("--progress", action="store_true", help="Print a line per run.")
    rn.add_argument("--no-nice", action="store_true", help="Do not raise the niceness to 15.")
    rn.set_defaults(func=cmd_run)

    ag = sub.add_parser("aggregate", help="Summarize downloaded shard results.")
    ag.add_argument("src")
    ag.add_argument("--out-dir", default="measure-summary")
    ag.add_argument("--plan", help="The planned matrix (plan's JSON): report the units that left no result.")
    ag.set_defaults(func=cmd_aggregate)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
