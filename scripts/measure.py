#!/usr/bin/env python
# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
"""Expensive-track measurement: panobbgo against the incumbents with q parallel workers.

The engine behind ``.github/workflows/measure.yml`` (roadmap
``planning/DESIGN_roadmap_2026-09-26.md`` §5.2; ``doc/dev/benchmarking.md``,
"Expensive-track measurement").  It runs the family track at small budgets
(``bm``·d evaluations, ``bm`` in {20, 100}) on the virtual clock (async
policy, log-normal durations, sigma 0.5, common random numbers per cell) with
q simulated workers, and scores AOCC over evaluations and ``aocc_time`` over
virtual time.  Subcommands:

``plan``
    Print the job matrix (JSON).  Stdlib only, so the planning job needs no
    dependencies::

        python scripts/measure.py plan --seeds 5
        python scripts/measure.py plan --seeds 1 --dims 2 --qs 1,4 \\
            --extra-units 'qLogEI.free.b100.q16.d5.s42.i0;qLogEI.free.b100.q64.d5.s42.i0'

``run``
    Run the units of one shard and write one result file per unit (atomically,
    as each unit finishes, so a cut job keeps what it finished) plus
    ``meta_<shard>.json`` with the host's CPU::

        uv run python scripts/measure.py run --units core.free.b20.q4.d2.s42 --shard core-01 --out-dir out

``aggregate``
    Turn downloaded shard artifacts into ``summary.md`` / ``summary.json``
    (:func:`aggregate` says what is in them)::

        gh run download RUN_ID --pattern 'measure-*' --dir measure-raw
        uv run python scripts/measure.py aggregate measure-raw --plan plan.json --out-dir measure-summary

A **unit** is one strategy group on one cell and one base seed,
``<group>.<preset>.b<bm>.q<q>.d<dim>.s<seed>[.i<j>]``: every instance of the
preset at that dimension, or with ``.i<j>`` only instance ``j`` of every
family (``plan`` splits a unit that would not fit a shard that way).  The
groups:

``core``
    The panobbgo specs (``make_ioh_strategies``), the cheap-track external
    baselines (pycma IPOP/BIPOP, NGOpt, Optuna CmaEs/TPE) and Py-BOBYQA, the
    local reference.  All of them run in one process.
``qLogEI``, ``TuRBO1``, ``SMAC``
    The GP-based baselines, each in shards of its own (they cost minutes per
    run).  SMAC has no batch acquisition and runs at q = 1 only; qLogEI and
    SMAC are left out where a run would take hours (:func:`covered`).

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
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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

#: Group -> the external baselines it runs (``core``: the panobbgo specs too,
#: resolved at run time from ``make_ioh_strategies``).
GROUPS: Dict[str, Tuple[str, ...]] = {
    "core": CHEAP_EXTERNALS + (LOCAL_REFERENCE,),
    "qLogEI": ("Baseline_BoTorch_qLogEI",),
    "TuRBO1": ("Baseline_TuRBO1",),
    "SMAC": ("Baseline_SMAC_BB",),
}

#: The GP groups (shards of their own).
BO_GROUPS: Tuple[str, ...] = ("qLogEI", "TuRBO1", "SMAC")

#: The pre-declared headline panobbgo spec (the sharing portfolio accepted at
#: low budget); every other panobbgo spec is secondary.
HEADLINE_SPEC = "Blocks_warm_CMAES_JSO"

#: Labels in the tables: what a reader must know about a baseline's row.
LABELS: Dict[str, str] = {
    "Baseline_PyBOBYQA": "PyBOBYQA (sequential)",
    "Baseline_SMAC_BB": "SMAC_BB (q=1 only)",
}

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
#: 16-core laptop (2026-09-26, light load, niced, one BLAS thread, one run at a time,
#: the ellipsoid family).  ``(10, 200)`` qLogEI and SMAC are the guide's numbers
#: (measured under heavy load); SMAC ``(5, 100)`` is interpolated and ``(5, 500)`` a lower
#: bound (stopped unfinished after 40 min); the rest are measured.
#: Other cells extrapolate from the same dimension (:func:`laptop_seconds`).
LAPTOP_SECONDS: Dict[str, Dict[Tuple[int, int], float]] = {
    "qLogEI": {(2, 40): 19.0, (2, 200): 87.0, (5, 100): 74.0, (5, 500): 972.0, (10, 200): 1000.0},
    "TuRBO1": {(2, 40): 9.4, (2, 200): 52.0, (5, 100): 22.0, (5, 500): 146.0, (10, 200): 56.0},
    "SMAC": {(2, 40): 9.5, (2, 200): 107.0, (5, 100): 45.0, (5, 500): 1800.0 * 1.2, (10, 200): 390.0},
}

#: A GP baseline runs on a cell only if one run at q = 1 is estimated at most this
#: long on the laptop (half an hour).  Coverage does not depend on q (SMAC aside),
#: so the pool of baselines is the same in every q cell of a (preset, dim, budget).
MAX_RUN_SECONDS = 1800.0

#: A GitHub runner core is taken to be this much slower than the laptop's.
RUNNER_FACTOR = 1.5

#: Growth of a GP baseline's run time with the budget at a fixed dimension (between
#: linear, a fixed cost per proposal, and quadratic, a fit that grows with the data).
BUDGET_EXPONENT = 1.6

#: qLogEI's cost grows with q: up to q - 1 pending points (``X_pending``) enter a
#: joint posterior at every proposal.  Factor ``1 + QLOGEI_Q_SLOPE * (q - 1)`` over
#: its q = 1 time: a GUESS (q = 16: 1.75, q = 64: 4.15) until the smoke run's
#: q = 16 / 64 units at d = 5, 100·d measure it.
QLOGEI_Q_SLOPE = 0.05

#: The step limit of a shard (``measure.yml``) and the planner's warning level below it.
STEP_LIMIT_MINUTES = 330.0
WARN_MINUTES = 300.0


def laptop_seconds(group: str, dim: int, budget: int, q: int = 1) -> float:
    """Estimated laptop seconds of one run of ``group`` (the core group: all its specs on one instance).

    GP groups: the measured entry of :data:`LAPTOP_SECONDS`, else the entry of
    the same (or the nearest larger) dimension with the nearest budget, scaled
    by ``(budget / b) ** BUDGET_EXPONENT``.  SMAC fits once per evaluation on
    the async clock at any q; qLogEI too, with a joint posterior over the
    pending points (:data:`QLOGEI_Q_SLOPE`); TuRBO proposes once per batch of
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
    elif group == "qLogEI":
        t *= 1.0 + QLOGEI_Q_SLOPE * (q - 1)
    return t


def covered(group: str, bm: int, dim: int, q: int) -> bool:
    """Whether ``group`` runs on the cell ``(bm, dim, q)``.

    The core group runs everywhere.  A GP baseline runs where one run *at
    q = 1* is estimated below :data:`MAX_RUN_SECONDS`: with the table above,
    everything but d = 10 at 100·d for qLogEI (about 3.6 h a run) and SMAC,
    and SMAC at d = 5, 100·d (over 40 min a run).  SMAC has no batch
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
    """One strategy group on one cell (preset, budget multiplier, q, dim), one base seed, and instances.

    ``inst = -1``: every instance; ``inst = j``: instance ``j`` of every family.
    """

    group: str
    preset: str
    bm: int
    q: int
    dim: int
    seed: int
    inst: int = -1

    @property
    def id(self) -> str:
        """``<group>.<preset>.b<bm>.q<q>.d<dim>.s<seed>[.i<j>]`` (also the result file's stem)."""
        tail = f".i{self.inst}" if self.inst >= 0 else ""
        return f"{self.group}.{self.preset}.b{self.bm}.q{self.q}.d{self.dim}.s{self.seed}{tail}"

    @classmethod
    def parse(cls, text: str) -> "Unit":
        """The inverse of :attr:`id`."""
        try:
            parts = text.strip().split(".")
            if len(parts) not in (6, 7):
                raise ValueError
            group, preset, b, q, d, s = parts[:6]
            if not (b[0] == "b" and q[0] == "q" and d[0] == "d" and s[0] == "s"):
                raise ValueError
            inst = -1
            if len(parts) == 7:
                if parts[6][0] != "i":
                    raise ValueError
                inst = int(parts[6][1:])
            unit = cls(group, preset, int(b[1:]), int(q[1:]), int(d[1:]), int(s[1:]), inst)
        except (ValueError, IndexError):
            raise ValueError(
                f"not a unit id: {text!r} (want <group>.<preset>.b<bm>.q<q>.d<dim>.s<seed>[.i<j>])"
            ) from None
        if unit.group not in GROUPS or unit.preset not in PRESET_DIMS:
            raise ValueError(f"unknown group or preset in {text!r}")
        if not (-1 <= unit.inst < N_INSTANCES):
            raise ValueError(f"instance out of range in {text!r}")
        return unit

    @property
    def n_runs(self) -> int:
        """Instances of the unit (every group runs each of them once per strategy)."""
        return PRESET_FAMILIES[self.preset] * (N_INSTANCES if self.inst < 0 else 1)

    def split(self) -> List["Unit"]:
        """One unit per instance index (a unit already split stays as it is)."""
        return [self] if self.inst >= 0 else [replace(self, inst=j) for j in range(N_INSTANCES)]


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


def split_long(units: Sequence[Unit], jobs: int, target_minutes: float) -> List[Unit]:
    """Split every unit estimated above ``target_minutes`` into one unit per instance index."""
    out: List[Unit] = []
    for u in units:
        out.extend(u.split() if unit_minutes(u, jobs) > target_minutes else [u])
    return out


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


def _entry(shard: str, group: str, items: Sequence[Unit], jobs: int) -> Dict[str, Any]:
    return {
        "shard": shard,
        "group": group,
        "units": ";".join(u.id for u in items),
        "n_units": len(items),
        "est_min": int(math.ceil(shard_minutes(items, jobs))),
    }


def plan(units: Sequence[Unit], jobs: int, target_minutes: float, extra: Sequence[Unit] = ()) -> List[Dict[str, Any]]:
    """The matrix entries: one per shard, the core group first, then ``extra`` units one shard each.

    A unit estimated above the target is split per instance first
    (:func:`split_long`).  ``extra`` units (calibration runs outside the
    grid) are never packed together.
    """
    entries = []
    for group in GROUPS:
        target = min(target_minutes, CORE_TARGET_MINUTES) if group == "core" else target_minutes
        mine = split_long([u for u in units if u.group == group], jobs, target)
        for i, items in enumerate(pack(mine, jobs, target), 1):
            entries.append(_entry(f"{group}-{i:02d}", group, items, jobs))
    have = {u for e in entries for u in e["units"].split(";")}
    for i, u in enumerate((u for u in extra if u.id not in have), 1):
        entries.append(_entry(f"extra-{i:02d}", u.group, [u], jobs))
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
    extra = [Unit.parse(u) for u in (args.extra_units or "").split(";") if u.strip()]
    if not units and not extra:
        print("error: the grid is empty", file=sys.stderr)
        return 2
    entries = plan(units, args.runner_cores, args.target_minutes, extra)
    if len(entries) > 256:
        print(f"error: {len(entries)} shards; a GitHub matrix takes at most 256", file=sys.stderr)
        return 2
    for e in entries:
        if e["est_min"] > WARN_MINUTES:
            print(
                f"warning: shard {e['shard']} is estimated at {e['est_min']} min (step limit {STEP_LIMIT_MINUTES:.0f})",
                file=sys.stderr,
            )
    print(json.dumps({"include": entries}, separators=(",", ":")))
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file and a rename: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


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
                "ATEN_CPU_CAPABILITY",
                "MKL_CBWR",
                "MKL_ENABLE_INSTRUCTIONS",
                "ONEDNN_MAX_CPU_ISA",
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


def fp_label(payload: Dict[str, Any]) -> str:
    """The FP environment of a unit result: ``fp_env_id`` (#359) when recorded, else the CPU class."""
    fp_id = (payload.get("result") or {}).get("fp_env_id")
    return str(fp_id) if fp_id else fp_class(payload.get("host") or {})


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


def _instances(preset: str, dim: int, inst: int = -1) -> List[Any]:
    from panobbgo.harness_families import make_failure_battery, make_families_battery

    make = {"free": make_families_battery, "failure": make_failure_battery}[preset]
    instances = list(make(dims=(dim,), n_instances=N_INSTANCES))
    return instances if inst < 0 else [(n, p) for n, p in instances if int(p.instance) == inst]


def run_unit(unit: Unit, jobs: int, progress: bool = False) -> Dict[str, Any]:
    """Run one unit and return its result (an ``IOHHarnessResult`` dict) with the unit's key."""
    from panobbgo.harness_families import run_family_harness
    from panobbgo.virtual_clock import VirtualSpec

    t0 = time.time()
    instances = _instances(unit.preset, unit.dim, unit.inst)
    if not instances:
        raise ValueError(f"{unit.id}: no instances")
    result = run_family_harness(
        _strategies(unit.group),
        instances,
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
        "inst": unit.inst,
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
    import panobbgo.fp_pin  # noqa: F401  # pyright: ignore[reportUnusedImport]  (the FP pin, before numpy loads)
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
            write_atomic(out_dir / f"{unit.id}.json", json.dumps(payload, default=float))
            meta["done"].append(unit.id)
            print(f"    done in {payload['elapsed_s'] / 60:.1f} min", flush=True)
        # Rewritten after every unit: a job cut by its timeout still says what it finished.
        meta["finished"] = datetime.now(tz=timezone.utc).isoformat()
        meta["status"] = status
        write_atomic(meta_path, json.dumps(meta, indent=2))
    return status


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

#: A cell of the summary: (preset, dim, bm, q).
Cell = Tuple[str, int, int, int]
#: A paired observation: (seed, family, instance).
Key = Tuple[int, str, int]

METRICS: Tuple[str, ...] = ("aocc", "aocc_time")


def headline_metric(q: int) -> str:
    """The pre-declared headline metric of a cell: ``aocc`` at q = 1, ``aocc_time`` at q > 1."""
    return "aocc" if q == 1 else "aocc_time"


@dataclass
class Obs:
    """One run's scores and where it ran."""

    aocc: float
    aocc_time: Optional[float]
    shard: str
    fp: str
    error: Optional[str]
    elapsed_s: float

    @property
    def hard_error(self) -> bool:
        """A crash or a timeout (``EndedEarly`` is a scored run that stopped itself, not an error)."""
        return bool(self.error) and not str(self.error).startswith("EndedEarly")


def label(name: str) -> str:
    """The table label of a strategy (``Baseline_`` dropped, :data:`LABELS` applied)."""
    return LABELS.get(name, name.removeprefix("Baseline_"))


def is_external(name: str) -> bool:
    """External baselines are the ``Baseline_*`` specs; the rest are panobbgo's."""
    return name.startswith("Baseline_")


def group_of(name: str) -> str:
    """The group a strategy runs in (every panobbgo spec: ``core``)."""
    return next((g for g, names in GROUPS.items() if name in names), "core")


def load_units(src: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Every unit result file under ``src`` and the files that could not be read (``meta_*`` excluded)."""
    out, bad = [], []
    for p in sorted(src.rglob("*.json")):
        if p.name.startswith(("meta_", "summary", "plan")):
            continue
        try:
            d = json.loads(p.read_text())
            if not (isinstance(d, dict) and "unit" in d and "result" in d):
                raise ValueError("not a unit result")
            Unit.parse(d["unit"])
        except (OSError, ValueError) as exc:
            bad.append(f"{p}: {type(exc).__name__}: {exc}")
            continue
        out.append(d)
    return out, bad


def collect(payloads: Iterable[Dict[str, Any]]) -> Dict[Cell, Dict[str, Dict[Key, Obs]]]:
    """``cell -> strategy -> (seed, family, instance) -> Obs``; a run seen twice is an error.

    A run that crashed or timed out has ``aocc = 0`` and no ``aocc_time``;
    it is scored ``aocc_time = 0`` too, so the time metric does not average
    over the survivors only.
    """
    cells: Dict[Cell, Dict[str, Dict[Key, Obs]]] = {}
    seen: Set[str] = set()
    for d in payloads:
        if d["unit"] in seen:
            raise ValueError(f"unit {d['unit']} appears twice")
        seen.add(d["unit"])
        cell: Cell = (d["preset"], int(d["dim"]), int(d["bm"]), int(d["q"]))
        fp = fp_label(d)
        virtual = d["result"].get("virtual") is not None
        for r in d["result"]["runs"]:
            key: Key = (int(d["seed"]), str(r["problem_kind"]), int(r["instance"]))
            obs = Obs(
                aocc=float(r["aocc"]),
                aocc_time=None if r.get("aocc_time") is None else float(r["aocc_time"]),
                shard=str(d.get("shard")),
                fp=fp,
                error=r.get("error"),
                elapsed_s=float(r.get("elapsed_s") or 0.0),
            )
            if obs.aocc_time is None and obs.hard_error and virtual:
                obs.aocc_time = 0.0
            runs = cells.setdefault(cell, {}).setdefault(r["strategy_name"], {})
            if key in runs:
                raise ValueError(f"run {r['strategy_name']} {key} of cell {cell} appears twice")
            runs[key] = obs
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


def mean_over_seeds(obs: Dict[Key, Obs], metric: str, keys: Optional[Iterable[Key]] = None) -> Optional[float]:
    """The mean over seeds of the per-seed instance means (``None`` without values)."""
    m = seed_means(obs, metric, keys)
    return sum(m.values()) / len(m) if m else None


def t_test(deltas: Sequence[float]) -> Tuple[float, float, float]:
    """``(mean, CI95 half-width, two-sided p)`` of a one-sample t-test of ``deltas`` against 0."""
    from panobbgo.harness_ioh import t_ci

    mean, half = t_ci(deltas)
    n = len(deltas)
    if n < 2:
        return mean, float("nan"), float("nan")
    import numpy as np
    from scipy.stats import t as t_dist

    sd = float(np.std(np.asarray(deltas, dtype=float), ddof=1))
    if sd == 0.0:
        return mean, half, (1.0 if mean == 0.0 else 0.0)
    stat = mean / (sd / math.sqrt(n))
    return mean, half, float(2.0 * t_dist.sf(abs(stat), n - 1))


def paired(a: Dict[Key, Obs], b: Dict[Key, Obs], metric: str, keys: Optional[Iterable[Key]] = None) -> Dict[str, Any]:
    """Paired-seed delta ``a - b`` of ``metric`` over their common (seed, family, instance) runs.

    Per seed the mean delta over the common instances (``keys``: only those),
    then a t-test over seeds.  ``cross_job`` / ``cross_fp``: some pair ran in
    two different jobs / FP environments — reproducibility metadata only, a
    different host is an equally valid sample.
    """
    pool = keys if keys is not None else a
    common = [k for k in pool if k in a and k in b]
    common = [k for k in common if _value(a[k], metric) is not None and _value(b[k], metric) is not None]
    per_seed: Dict[int, List[float]] = {}
    for k in common:
        per_seed.setdefault(k[0], []).append(float(_value(a[k], metric)) - float(_value(b[k], metric)))  # type: ignore[arg-type]
    deltas = [sum(v) / len(v) for _, v in sorted(per_seed.items())]
    mean, half, p = t_test(deltas)
    return {
        "n_seeds": len(deltas),
        "n_pairs": len(common),
        "delta": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "p": p,
        "wins": sum(1 for d in deltas if d > 0),
        "cross_job": any(a[k].shard != b[k].shard for k in common),
        "cross_fp": any(a[k].fp != b[k].fp for k in common),
    }


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-adjusted p-values (NaN stays NaN and is not counted)."""
    items = sorted(((p, k) for k, p in pvalues.items() if not math.isnan(p)))
    m = len(items)
    out = {k: float("nan") for k in pvalues}
    running = 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def _expected(
    cells: Dict[Cell, Dict[str, Dict[Key, Obs]]], planned_units: Optional[Sequence[str]], core_names: Sequence[str]
) -> Dict[Cell, Dict[str, Set[int]]]:
    """``cell -> strategy -> planned seeds``: from the plan, else every strategy seen, on the cell's seeds."""
    out: Dict[Cell, Dict[str, Set[int]]] = {}
    if planned_units is not None:
        for text in planned_units:
            u = Unit.parse(text)
            names = list(GROUPS[u.group]) + (list(core_names) if u.group == "core" else [])
            per = out.setdefault((u.preset, u.dim, u.bm, u.q), {})
            for n in names:
                per.setdefault(n, set()).add(u.seed)
    for cell, strats in cells.items():
        if planned_units is None or cell not in out:
            seeds = {k[0] for obs in strats.values() for k in obs}
            per = out.setdefault(cell, {})
            for n in strats:
                per.setdefault(n, set()).update(seeds)
    return out


def _status(obs: Dict[Key, Obs], seeds: Set[int], instances: Set[Tuple[str, int]]) -> Dict[str, Any]:
    """Completeness and errors of one strategy on one cell."""
    have = {k[0] for k in obs}
    complete = all((s, f, i) in obs for s in seeds for f, i in instances)
    return {
        "n_seeds": len(have & seeds),
        "planned_seeds": len(seeds),
        "complete": complete,
        "errors": sum(1 for o in obs.values() if o.hard_error),
        "ended_early": sum(1 for o in obs.values() if o.error and not o.hard_error),
    }


def summarize(
    cells: Dict[Cell, Dict[str, Dict[Key, Obs]]],
    planned_units: Optional[Sequence[str]] = None,
    core_names: Sequence[str] = (),
) -> Dict[str, Dict[str, Any]]:
    """The per-cell analysis (see :func:`aggregate`)."""
    expected = _expected(cells, planned_units, core_names)
    # The cell's instance set: every (family, instance) any strategy ran on it.
    instances = {c: {(k[1], k[2]) for obs in s.values() for k in obs} for c, s in cells.items()}
    status = {
        c: {n: _status(obs, expected[c].get(n, set()), instances[c]) for n, obs in s.items()} for c, s in cells.items()
    }

    # The fixed pool per (preset, dim, bm): externals present, complete and
    # error-free in EVERY q cell of it, so best-of does not change with q
    # because the pool changed.
    groups: Dict[Tuple[str, int, int], List[Cell]] = {}
    for c in cells:
        groups.setdefault(c[:3], []).append(c)
    pool: Dict[Tuple[str, int, int], List[str]] = {}
    excluded: Dict[Tuple[str, int, int], Dict[str, str]] = {}
    for g, cs in groups.items():
        why: Dict[str, List[str]] = {}
        for n in sorted({n for c in cs for n in cells[c] if is_external(n)}):
            for c in sorted(cs, key=lambda c: c[3]):
                st = status[c].get(n)
                problem = (
                    "absent" if st is None else "incomplete" if not st["complete"] else "errors" if st["errors"] else ""
                )
                if problem:
                    why.setdefault(n, []).append(f"{problem} at q={c[3]}")
        pool[g] = sorted({n for c in cs for n in cells[c] if is_external(n)} - set(why))
        excluded[g] = {n: ", ".join(r) for n, r in why.items()}

    out: Dict[str, Dict[str, Any]] = {}
    for c in sorted(cells):
        preset, dim, bm, q = c
        strats = cells[c]
        hm = headline_metric(q)
        cell_pool = pool[c[:3]]
        rows: Dict[str, Any] = {}
        for name, obs in strats.items():
            fams: Dict[str, Dict[str, Optional[float]]] = {}
            for fam in sorted({k[1] for k in obs}):
                fk = [k for k in obs if k[1] == fam]
                fams[fam] = {m: mean_over_seeds(obs, m, fk) for m in METRICS}
            rows[name] = {
                "label": label(name),
                "external": is_external(name),
                "group": group_of(name),
                "in_pool": name in cell_pool,
                **status[c][name],
                "seconds_per_run": sum(o.elapsed_s for o in obs.values()) / len(obs) if obs else None,
                **{m: mean_over_seeds(obs, m) for m in METRICS},
                "per_family": fams,
            }
        best = {
            m: max(cell_pool, key=lambda n: rows[n][m] if rows[n][m] is not None else -1.0, default=None)
            for m in METRICS
        }
        # Flag every external outside the pool that scores above the pool's best (it is a reference row).
        flags = []
        for m in METRICS:
            b = best[m]
            floor = rows[b][m] if b and rows[b][m] is not None else -1.0
            for n in sorted(strats):
                if is_external(n) and n not in cell_pool and rows[n][m] is not None and rows[n][m] > floor:
                    reason = excluded[c[:3]].get(n, "?")
                    flags.append(f"{m}: {label(n)} scores above the pool's best but is outside the pool ({reason})")
        externals = [n for n in strats if is_external(n)]
        for name in strats:
            if is_external(name):
                continue
            r = rows[name]
            r["vs"] = {e: {m: paired(strats[name], strats[e], m) for m in METRICS} for e in externals}
            r["vs_pool_best"] = {}
            for m in METRICS:
                bm_name = best[m]
                r["vs_pool_best"][m] = paired(strats[name], strats[bm_name], m) if bm_name else None
            best_hm = best[hm]
            if best_hm:
                bo = strats[best_hm]
                r["per_family_vs_pool_best"] = {
                    fam: paired(strats[name], bo, hm, [k for k in strats[name] if k[1] == fam])
                    for fam in r["per_family"]
                }
        planned_names = sorted(expected[c])
        out[f"{preset}/d{dim}/b{bm}/q{q}"] = {
            "preset": preset,
            "dim": dim,
            "bm": bm,
            "q": q,
            "headline_metric": hm,
            "pool": cell_pool,
            "pool_excluded": excluded[c[:3]],
            "pool_best": best,
            "flags": flags,
            "planned": planned_names,
            "present": sorted(strats),
            "missing": sorted(set(planned_names) - set(strats)),
            "strategies": rows,
        }
    # Holm over the headline set: the headline spec vs the pool's best, per cell, on the headline metric.
    ps = {
        k: v["strategies"][HEADLINE_SPEC]["vs_pool_best"][v["headline_metric"]]["p"]
        for k, v in out.items()
        if HEADLINE_SPEC in v["strategies"] and v["strategies"][HEADLINE_SPEC]["vs_pool_best"][v["headline_metric"]]
    }
    adjusted = holm(ps)
    for k, v in out.items():
        v["headline"] = None
        if k in ps:
            st = v["strategies"][HEADLINE_SPEC]["vs_pool_best"][v["headline_metric"]]
            v["headline"] = {**st, "p_holm": adjusted[k], "vs": v["pool_best"][v["headline_metric"]]}
    return out


def aggregate(src: Path, planned_units: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """The summary of every unit result under ``src`` (``planned_units``: the plan's unit ids).

    Per cell (preset, dim, bm, q):

    * every strategy's mean AOCC and ``aocc_time`` (mean over seeds of the
      per-seed instance mean; a crashed or timed-out run scores 0 on both),
      per-family means, seeds against the plan, errors;
    * the **pool**: the externals present, complete and error-free in every q
      cell of the (preset, dim, bm) — fixed across q, so a delta does not move
      with q because the pool changed.  Others (SMAC at q = 1, a baseline
      left out of some cells, one with errors) are reference rows;
    * for every panobbgo spec the paired-seed delta against the pool's best
      (per metric) and against every external, and per family against the
      pool's best on the headline metric;
    * the headline: :data:`HEADLINE_SPEC` against the pool's best on
      :func:`headline_metric`, Holm-adjusted over the cells.
    """
    from panobbgo.harness_ioh import make_ioh_strategies

    payloads, unreadable = load_units(src)
    if not payloads:
        raise ValueError(f"no unit result files under {src}")
    metas = []
    for p in sorted(src.rglob("meta_*.json")):
        try:
            metas.append(json.loads(p.read_text()))
        except (OSError, ValueError) as exc:
            unreadable.append(f"{p}: {type(exc).__name__}: {exc}")
    cells = collect(payloads)
    done = {d["unit"] for d in payloads}
    missing = sorted(set(planned_units or []) - done)
    fp_by_shard = {str(d.get("shard")): fp_label(d) for d in payloads}
    return {
        "created": datetime.now(tz=timezone.utc).isoformat(),
        "git_sha": sorted({str(d.get("git_sha")) for d in payloads}),
        "github_run_id": sorted({str(m.get("github_run_id")) for m in metas if m.get("github_run_id")}),
        "units": len(payloads),
        "missing_units": missing,
        "failed_units": sorted({u for m in metas for u in m.get("failed", [])}),
        "unreadable_files": unreadable,
        "fp_classes": {
            fp: sorted(s for s, f in fp_by_shard.items() if f == fp) for fp in sorted(set(fp_by_shard.values()))
        },
        "virtual": {"duration": DURATION, "sigma": SIGMA, "policy": "async", "durations": "crn"},
        "headline_spec": HEADLINE_SPEC,
        "cells": summarize(cells, planned_units, [s.name for s in make_ioh_strategies()]),
    }


def _fmt(x: Optional[float], nd: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _fmt_delta(st: Optional[Dict[str, Any]]) -> str:
    if not st or st["n_seeds"] == 0:
        return "–"
    lo, hi = st["ci_low"], st["ci_high"]
    ci = "" if math.isnan(lo) else f" [{lo:+.3f},{hi:+.3f}]"
    return f"{st['delta']:+.3f}{ci} {st['wins']}/{st['n_seeds']}"


def _seeds(r: Dict[str, Any]) -> str:
    mark = "" if r["complete"] else "!"
    return f"{r['n_seeds']}/{r['planned_seeds']}{mark}"


def summary_markdown(summary: Dict[str, Any]) -> str:
    """The summary as Markdown: the headline table, the per-family table, then one table per cell."""
    hs = summary["headline_spec"]
    lines = [
        "# Expensive-track measurement",
        "",
        f"Commit {', '.join(s[:7] for s in summary['git_sha'])}; run {', '.join(summary['github_run_id']) or 'local'}; "
        f"{summary['units']} unit(s); missing {len(summary['missing_units'])}, failed "
        f"{len(summary['failed_units'])}, unreadable files {len(summary['unreadable_files'])}.",
        f"Virtual clock: async policy, {summary['virtual']['duration']} durations "
        f"(sigma {summary['virtual']['sigma']}), common random numbers per cell (the i-th dispatch takes the same "
        "time for every strategy); `aocc_time` over the horizon budget/q mean durations.",
        "",
        "**How to read this.**",
        f"- Pre-declared: the headline spec is `{hs}`; the other panobbgo specs are secondary.  The headline "
        "metric is AOCC at q = 1 and `aocc_time` at q > 1; tables are sorted by it.",
        "- Δ = panobbgo spec − reference: the mean over seeds of the per-seed instance-mean delta, t-CI95 over "
        "seeds, wins/seeds.  Crashed or timed-out runs score 0 on both metrics.",
        "- *Pool best*: the best external of a pool fixed per (preset, dim, budget) — the externals present, "
        "complete and error-free in every q cell — so the reference does not change with q because a baseline "
        "is missing at some q.  SMAC (q = 1 only) and baselines left out of some cells are reference rows.",
        "- The pool's best is a selected maximum over several baselines, which favours the baseline side.",
        f"- Multiplicity: many cells, specs and baselines are compared.  Only the headline set (`{hs}` vs the "
        "pool's best, per cell, on the headline metric) is Holm-adjusted (p_holm); every other CI is "
        "unadjusted and descriptive.  With 5 seeds, wins/5 alone cannot be significant (5/5 has p = 0.0625 "
        "two-sided in a sign test).",
        "- The CIs are over optimizer seeds and conditional on the fixed instances: they do not generalize "
        "over problem instances beyond the 3 per family.",
        "- `n/planned` counts seeds against the plan; `!` marks a strategy missing some (seed, instance) run.",
        "",
        "FP environments (per shard): "
        + "; ".join(f"{fp}: {len(shards)} shard(s)" for fp, shards in summary["fp_classes"].items())
        + ".  Comparisons across jobs are ordinary samples; the environment is reproducibility metadata.",
        "",
        "## Headline",
        "",
        f"| cell | metric | pool best | {hs} | Δ [CI95] wins | p | p_holm | best other panobbgo | errors pb/ext | flags |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, c in summary["cells"].items():
        rows, hm = c["strategies"], c["headline_metric"]
        b = c["pool_best"][hm]
        h = rows.get(hs)
        others = [(r[hm], n) for n, r in rows.items() if not r["external"] and n != hs and r[hm] is not None]
        other = max(others)[1] if others else None
        err_pb = sum(r["errors"] for r in rows.values() if not r["external"])
        err_ext = sum(r["errors"] for r in rows.values() if r["external"])
        hl = c["headline"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    hm,
                    f"{label(b)} {_fmt(rows[b][hm])} ({_seeds(rows[b])})" if b else "–",
                    f"{_fmt(h[hm])} ({_seeds(h)})" if h else "–",
                    _fmt_delta(hl),
                    _fmt(hl["p"]) if hl else "–",
                    _fmt(hl["p_holm"]) if hl else "–",
                    f"{other} {_fmt(rows[other][hm])} {_fmt_delta(rows[other]['vs_pool_best'][hm])}" if other else "–",
                    f"{err_pb}/{err_ext}",
                    "; ".join(c["flags"]) or "",
                ]
            )
            + " |"
        )
    fams = sorted({f for c in summary["cells"].values() for r in c["strategies"].values() for f in r["per_family"]})
    lines += [
        "",
        f"## Per family: {hs} − pool best (headline metric)",
        "",
        "Δ per family (paired over seeds on that family's instances); the roadmap claim is *never much worse on "
        "any class*, so read the minimum of each row.",
        "",
        "| cell | " + " | ".join(fams) + " |",
        "|---|" + "---|" * len(fams),
    ]
    for name, c in summary["cells"].items():
        h = c["strategies"].get(hs) or {}
        pf = h.get("per_family_vs_pool_best") or {}
        lines.append("| " + name + " | " + " | ".join(_fmt_delta(pf.get(f)) for f in fams) + " |")
    for name, c in summary["cells"].items():
        rows, hm = c["strategies"], c["headline_metric"]
        b = c["pool_best"][hm]
        strongest = sorted(
            (n for n in rows if rows[n]["external"] and n != b and rows[n][hm] is not None),
            key=lambda n: -rows[n][hm],
        )[:2]
        refs = ([b] if b else []) + strongest
        lines += [
            "",
            f"### {name} (headline: {hm})",
            "",
            f"Pool: {', '.join(label(n) for n in c['pool']) or '(empty)'}.  Pool best: AOCC "
            f"{label(c['pool_best']['aocc']) if c['pool_best']['aocc'] else '–'}, aocc_time "
            f"{label(c['pool_best']['aocc_time']) if c['pool_best']['aocc_time'] else '–'}.  "
            f"Planned {len(c['planned'])} strategies, present {len(c['present'])}"
            + (f"; missing: {', '.join(label(n) for n in c['missing'])}" if c["missing"] else "")
            + ".",
            "",
            "| strategy | seeds | AOCC | aocc_time | "
            + " | ".join(f"Δ {hm} vs {label(n)}{' (pool best)' if n == b else ''}" for n in refs)
            + " | errors | s/run |",
            "|---|---|---|---|" + "---|" * len(refs) + "---|---|",
        ]
        for n in sorted(rows, key=lambda n: -(rows[n][hm] if rows[n][hm] is not None else -1.0)):
            r = rows[n]
            name_cell = f"**{label(n)}**" if n == hs else label(n)
            if r["external"]:
                name_cell += " (pool)" if r["in_pool"] else " (reference)"
            deltas = [("–" if r["external"] else _fmt_delta(r["vs"][e][hm])) for e in refs]
            lines.append(
                f"| {name_cell} | {_seeds(r)} | {_fmt(r['aocc'])} | {_fmt(r['aocc_time'])} | "
                + " | ".join(deltas)
                + f" | {r['errors']} | {_fmt(r['seconds_per_run'], 1)} |"
            )
    if summary["missing_units"] or summary["failed_units"] or summary["unreadable_files"]:
        lines += ["", "## Missing / failed units, unreadable files", ""]
        lines += [f"- missing: {u}" for u in summary["missing_units"]]
        lines += [f"- failed: {u}" for u in summary["failed_units"]]
        lines += [f"- unreadable: {u}" for u in summary["unreadable_files"]]
    return "\n".join(lines) + "\n"


def cmd_aggregate(args: argparse.Namespace) -> int:
    planned = None
    if args.plan:
        data = json.loads(Path(args.plan).read_text())
        entries = data["include"] if isinstance(data, dict) else data
        planned = [u for e in entries for u in e["units"].split(";") if u]
    summary = aggregate(Path(args.src), planned)
    out = Path(args.out_dir)
    write_atomic(out / "summary.json", json.dumps(summary, indent=2, default=float) + "\n")
    md = summary_markdown(summary)
    write_atomic(out / "summary.md", md)
    print(md)
    problems = {k: len(summary[k]) for k in ("missing_units", "failed_units", "unreadable_files") if summary[k]}
    if problems:
        print("warning: " + ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in problems.items()), file=sys.stderr)
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
    pl.add_argument(
        "--extra-units", default="", help="';'-joined unit ids run in shards of their own (calibration runs)."
    )
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
    ag.add_argument("--plan", help="The planned matrix (plan's JSON): seeds against the plan, missing units.")
    ag.set_defaults(func=cmd_aggregate)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
