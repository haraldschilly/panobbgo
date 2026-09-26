#!/usr/bin/env python
# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
"""Re-measure the reference baselines, sharded over GitHub runners.

The engine behind ``.github/workflows/rebaseline.yml`` (``doc/dev/benchmarking.md``,
"Re-baselining on GitHub runners").  Subcommands:

``plan``
    Print the job matrix (JSON) for a set of suites and seeds.  Stdlib only,
    so the workflow's planning job needs no dependencies::

        python scripts/rebaseline.py plan --suites all --seeds 12

``run``
    Run one shard — one suite on a few base seeds — and write its raw result
    files plus a ``meta_<shard>.json`` into ``OUT_DIR/<suite>/``::

        uv run python scripts/rebaseline.py run ioh-standard --seeds 42 --shard 01 --out-dir out

``aggregate``
    Turn downloaded shard artifacts (``gh run download``) into the reference
    files, in the formats the comparison tools read, plus ``ref_MANIFEST.json``
    and the small ``SUMMARY.json``::

        gh run download RUN_ID --pattern 'shard-*' --dir rebaseline-raw
        uv run python scripts/rebaseline.py aggregate rebaseline-raw

``publish``
    Attach an aggregated directory to a GitHub release (the workflow does
    this; needs ``gh`` with write access).  The raw ``ref_*`` files are not
    committed to git (Harald, 2026-09-26)::

        uv run python scripts/rebaseline.py publish planning/results/2026-09-26 --target SHA

``fetch``
    Download and unpack a release's reference files, by default into
    ``planning/results/<date>/`` (``.gitignore`` keeps the raw files out of
    git; ``SUMMARY.json`` and ``ref_MANIFEST.json`` are committed)::

        uv run python scripts/rebaseline.py fetch rebaseline-2026-09-26-run36228301268

``summarize``
    (Re)write ``SUMMARY.json`` of an aggregated directory.

Release convention: tag ``rebaseline-<UTC date the run started>``
(``rebaseline-<date>-run<id>`` when that tag is taken, by another release or a
bare git tag), a *pre-release* never marked latest, the tag on the measured
commit.  A release whose notes carry the run's marker is reused, never
duplicated.  Assets: ``<tag>.tar.gz``
(every ``ref_*.json``, flat), ``ref_MANIFEST.json`` and ``SUMMARY.json``.
The repository has immutable releases: a published release never changes,
and the tag of a deleted one cannot be used again.

Suites and the files ``aggregate`` writes (default into
``planning/results/<UTC date>/``):

==========================  =================================================
suite                       reference file(s)
==========================  =================================================
``composite-quick``         ``ref_composite_quick_s<seed>.json`` — one
``composite-standard``      ``benchmark_harness.py run`` result per seed, for
                            ``benchmark_harness.py compare``
``ioh-quick``               ``ref_ioh_<key>.json`` — one multi-seed
``ioh-standard``            result (``ioh_benchmark.py run --baselines
``ioh-external``            --seeds ...``) for ``ioh_benchmark.py compare``,
                            plus ``ref_ioh_<key>_s<seed>.json`` single-seed
                            files for a plain ``run --output`` A/B;
                            ``<key>`` is ``quick``, ``standard`` and
                            ``standard_external`` (the standard battery with
                            the pycma / Nevergrad / Optuna baselines too)
``families-free``           ``ref_family_screen_<preset>.json`` — the rows
``families-constrained``    file of ``benchmarks/family_screen.py``;
``families-shapes``         re-analyse with ``family_screen.py from=FILE``
``families-failure``
==========================  =================================================

Every measurement is synchronous (``sync_eval``) and seeded, with no
wall-clock limit, so a result depends on the code and the seed only — not
on the runner's speed or the ``--jobs`` count — within one floating-point
environment: the kernels are pinned (``panobbgo/fp_env.py``), every result
file and shard meta records ``fp_env`` / ``fp_env_id``, and ``aggregate``
refuses to merge shards of one suite from different ids unless
``--allow-mixed-fp``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The 12-seed decision roster, ``panobbgo.harness_ioh.DEFAULT_DECISION_SEEDS``
#: (copied so ``plan`` runs without the package's dependencies; a test keeps
#: the two equal).
ROSTER: Tuple[int, ...] = (42, 7, 1234, 2025, 3, 11, 99, 123, 777, 2024, 31337, 555)


@dataclass(frozen=True)
class Suite:
    """One reference measurement: a harness, its battery and how many seeds one job runs."""

    name: str
    #: ``composite`` (benchmark_harness.py), ``ioh`` (scripts/ioh_benchmark.py)
    #: or ``families`` (benchmarks/family_screen.py).
    kind: str
    #: The mode (``quick`` / ``standard``) or the ``family_screen.py`` preset
    #: (``free`` / ``constrained`` / ``shapes`` / ``failure``).
    battery: str
    #: Seeds per job — sized so a job stays far below the workflow's 350-minute timeout.
    seeds_per_job: int
    #: Tells suites on the same battery apart: names the strategy set
    #: (:data:`STRATEGY_SETS`, IOH only) and goes into the reference file
    #: names (:attr:`ref_key`).  Empty: the harness' default strategies.
    variant: str = ""
    #: Optional-dependency extras the job installs besides ``dev``
    #: (``uv sync --extra dev --extra <name>``).
    extras: Tuple[str, ...] = ()

    @property
    def ref_key(self) -> str:
        """The battery, plus the variant if any: ``ref_ioh_<ref_key>.json``."""
        return f"{self.battery}_{self.variant}" if self.variant else self.battery


# Shard sizing.  The first full run (release rebaseline-2026-09-26-run36228301268,
# 4-core runners, --jobs 4) gives the job times of the older suites; the newer
# ones come from local timings (2026-09-26, a laptop under load, niced, one
# process, seed 42):
#
# *   ioh-standard: 0.3-0.5 min per seed in that run (one seed per job
#     then), so 4 seeds per job; the reference files stay the same.
# *   ioh-external: per seed 10 cells (d 2 and 5 x 5 instances, 500*d
#     evaluations) x 14 strategies.  One d = 5 cell (2500 evaluations):
#     Baseline_NGOpt 40 s, Baseline_Optuna_TPE 34 s (its cost grows about
#     quadratically with the budget: 1 / 3.6 / 8.4 / 34 s at 250 / 500 / 1000 /
#     2500), Baseline_NG_CMA 6 s, every other spec 1-4 s; ~100 s for all 14.
#     One d = 2 cell (1000 evaluations): ~25 s for all 14.  So ~10 min per
#     seed serially; 4 seeds are ~40 min per job serially, ~10-15 min at
#     --jobs 4, far below 350 min even on a runner 3x slower than the laptop.
# *   families-shapes: one seed with one instance per (family, dim) takes
#     28 CPU-s, ~1.5 CPU-min per seed at the preset's 3 instances; 4 per job.
# *   families-failure: 12 CPU-s likewise, ~0.6 CPU-min per seed; 6 per job.
#     Its failure regions raise instead of sleeping, so no wall time is lost.
SUITES: Dict[str, Suite] = {
    s.name: s
    for s in (
        Suite("composite-quick", "composite", "quick", 12),
        Suite("composite-standard", "composite", "standard", 3),
        Suite("ioh-quick", "ioh", "quick", 12),
        Suite("ioh-standard", "ioh", "standard", 4),
        Suite("ioh-external", "ioh", "standard", 4, variant="external", extras=("baselines",)),
        # The expensive-track baselines (BoTorch / SMAC / HEBO, extra
        # ``baselines-bo``) join as a suite of their own here: a variant in
        # STRATEGY_SETS that names them, extras=("baselines-bo",), a battery
        # whose budget they can afford, and seeds_per_job from a local timing.
        Suite("families-free", "families", "free", 2),
        Suite("families-constrained", "families", "constrained", 3),
        Suite("families-shapes", "families", "shapes", 4),
        Suite("families-failure", "families", "failure", 6),
    )
}


#: The extra an external baseline class needs when it names none: every
#: pycma / Nevergrad / Optuna class comes from ``baselines``.
DEFAULT_BASELINE_EXTRA = "baselines"


def baseline_extra(strategy_class: type) -> str:
    """The optional-dependency extra that provides an external baseline class.

    Its ``extra`` attribute, else :data:`DEFAULT_BASELINE_EXTRA`.
    """
    return str(getattr(strategy_class, "extra", DEFAULT_BASELINE_EXTRA))


def external_baselines_for(extras: Sequence[str]) -> List[str]:
    """The external baselines (``EXTERNAL_BASELINE_NAMES`` order) whose extra is among ``extras``.

    Selected by extra, not by taking the whole name list: a baseline from
    another extra (e.g. the expensive track's ``baselines-bo``) would make a
    shard that installs only ``baselines`` fail before its first run.
    """
    from panobbgo.harness_baselines import make_external_baseline_strategies

    return [s.name for s in make_external_baseline_strategies() if baseline_extra(s.strategy_class) in extras]


def _external_strategy_names(suite: Suite) -> List[str]:
    """The default IOH specs and baselines plus the external baselines ``suite.extras`` provide."""
    from panobbgo.harness_baselines import DEFAULT_BASELINE_NAMES
    from panobbgo.harness_ioh import make_ioh_strategies

    names = [s.name for s in make_ioh_strategies()] + list(DEFAULT_BASELINE_NAMES)
    return names + external_baselines_for(suite.extras)


#: IOH suite variant -> the ``--strategies`` names its shards run.  Resolved
#: in the shard, which has the package installed (``plan`` stays stdlib only).
#: The external baselines join a run only when ``--strategies`` names them.
STRATEGY_SETS: Dict[str, Callable[[Suite], List[str]]] = {
    "external": _external_strategy_names,
}


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def seed_list(spec: str) -> List[int]:
    """``"42,7"`` (or ``"42"``) -> exactly those distinct seeds."""
    seeds = [int(s) for s in spec.replace(" ", "").split(",") if s]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"need distinct seeds; got {spec!r}")
    return seeds


def resolve_seeds(spec: str) -> List[int]:
    """The ``plan`` input: ``"12"`` -> the first 12 roster seeds; ``"42,7"`` -> exactly those.

    A bare number is a count; one explicit seed takes a trailing comma (``"42,"``).
    """
    spec = spec.strip()
    if spec.isdigit():
        n = int(spec)
        if not 1 <= n <= len(ROSTER):
            raise ValueError(f"seed count must be 1..{len(ROSTER)} (the roster); got {n}")
        return list(ROSTER[:n])
    return seed_list(spec)


def resolve_suites(spec: str) -> List[Suite]:
    """``"all"`` or a comma list of suite names, in :data:`SUITES` order."""
    names = [n.strip() for n in spec.split(",") if n.strip()]
    if not names or names == ["all"]:
        return list(SUITES.values())
    unknown = [n for n in names if n not in SUITES]
    if unknown:
        raise ValueError(f"unknown suite(s) {unknown}; known: {', '.join(SUITES)}")
    return [s for s in SUITES.values() if s.name in names]


def plan(suites: Sequence[Suite], seeds: Sequence[int]) -> List[Dict[str, str]]:
    """One matrix entry per (suite, seed chunk), with the extras its job installs."""
    entries = []
    for suite in suites:
        k = suite.seeds_per_job
        for i in range(0, len(seeds), k):
            chunk = seeds[i : i + k]
            entries.append(
                {
                    "suite": suite.name,
                    "shard": f"{i // k + 1:02d}",
                    "seeds": ",".join(str(s) for s in chunk),
                    # '+'-joined; the workflow turns it into --extra flags.
                    "extras": "+".join(suite.extras),
                }
            )
    return entries


def cmd_plan(args: argparse.Namespace) -> int:
    entries = plan(resolve_suites(args.suites), resolve_seeds(args.seeds))
    print(json.dumps({"include": entries}, separators=(",", ":")))
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def shard_commands(
    suite: Suite, seeds: Sequence[int], shard: str, out_dir: Path, jobs: int, python: str = sys.executable
) -> List[Tuple[List[str], Path]]:
    """The ``(argv, result file)`` pairs that measure ``suite`` on ``seeds``.

    Synchronous evaluation and no wall-clock limit everywhere: the results
    depend on the code and the seeds only.  ``evaluation.timeout`` stays unset.
    """
    if suite.kind == "composite":
        # One harness run per seed: that is the file ``compare`` reads.
        commands = []
        for seed in seeds:
            out = out_dir / f"{suite.name}_s{seed}.json"
            argv = [
                python,
                str(REPO_ROOT / "benchmark_harness.py"),
                "run",
                f"--{suite.battery}",
                "--seed",
                str(seed),
                "--sync-eval",
                "--no-timeout",
                "--quiet",
                "--output",
                str(out),
            ]
            commands.append((argv, out))
        return commands
    out = out_dir / f"{suite.name}_shard{shard}.json"
    if suite.kind == "ioh":
        # DISCOVERY §1: the default IOH strategies plus the external baselines.
        argv = [
            python,
            str(REPO_ROOT / "scripts" / "ioh_benchmark.py"),
            "run",
            f"--{suite.battery}",
            "--baselines",
            *(["--strategies", *STRATEGY_SETS[suite.variant](suite)] if suite.variant else []),
            "--seeds",
            *[str(s) for s in seeds],
            "--sync-eval",
            "--jobs",
            str(jobs),
            "--quiet",
            "--output",
            str(out),
        ]
        return [(argv, out)]
    if suite.kind == "families":
        # DISCOVERY §34: the family screen is the multi-seed instrument of the
        # family track (sync_eval is built in, no timeout unless asked for).
        argv = [
            python,
            str(REPO_ROOT / "benchmarks" / "family_screen.py"),
            str(out),
            *[str(s) for s in seeds],
            f"preset={suite.battery}",
            f"jobs={jobs}",
        ]
        return [(argv, out)]
    raise ValueError(f"unknown suite kind {suite.kind!r}")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GITHUB_SHA", "unknown")


def _fp_env() -> Dict[str, Any]:
    """The FP environment of this runner (the shards' result files carry their own)."""
    from panobbgo import fp_env

    return fp_env.collect()


def cmd_run(args: argparse.Namespace) -> int:
    suite = SUITES[args.suite]
    seeds = seed_list(args.seeds)
    out_dir = Path(args.out_dir) / suite.name
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs or os.cpu_count() or 1
    commands = shard_commands(suite, seeds, args.shard, out_dir, jobs)
    meta: Dict[str, Any] = {
        "suite": suite.name,
        "shard": args.shard,
        "seeds": seeds,
        "git_sha": _git_sha(),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "started": datetime.now(tz=timezone.utc).isoformat(),
        "commands": [" ".join(argv[1:]) for argv, _ in commands],
        "jobs": jobs,
        "fp_env": _fp_env(),
    }
    status = 0
    for argv, out in commands:
        print(f"$ {' '.join(argv[1:])}", flush=True)
        rc = subprocess.run(argv, cwd=REPO_ROOT).returncode
        if rc != 0 or not out.exists():
            print(f"error: exit code {rc}, result file {'present' if out.exists() else 'missing'}", file=sys.stderr)
            status = rc or 1
            break
    meta["finished"] = datetime.now(tz=timezone.utc).isoformat()
    meta["status"] = status
    (out_dir / f"meta_{args.shard}.json").write_text(json.dumps(meta, indent=2))
    return status


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def _seed_order(seed: int) -> Tuple[int, int]:
    """Roster order first, other seeds after them by value."""
    return (ROSTER.index(seed), 0) if seed in ROSTER else (len(ROSTER), seed)


def _result_files(src: Path, suite: str) -> List[Path]:
    return sorted(p for p in src.rglob(f"{suite}/*.json") if not p.name.startswith("meta_"))


def _check_unique(suite: str, seeds: Sequence[int]) -> None:
    dup = sorted({s for s in seeds if list(seeds).count(s) > 1})
    if dup:
        raise ValueError(f"{suite}: seed(s) {dup} appear in more than one shard file")


def _fp_ids(suite: Suite, ids: Sequence[Optional[str]], allow_mixed_fp: bool) -> List[Optional[str]]:
    """The distinct ``fp_env_id`` values of a suite's shards; more than one raises unless allowed.

    A reference merged from two FP environments is not a reference: the
    same seed need not give the same numbers in both (``doc/dev/benchmarking.md``).
    ``None`` (a file without the record) counts as an id of its own.
    """
    distinct = sorted(set(ids), key=str)
    if len(distinct) > 1:
        what = f"{suite.name}: shards measured in different FP environments (fp_env_id {', '.join(map(str, distinct))})"
        if not allow_mixed_fp:
            raise ValueError(what + "; re-run them in one environment, or pass --allow-mixed-fp")
        print(f"warning: {what}; merged anyway (--allow-mixed-fp)", file=sys.stderr)
    return distinct


def _merged_fp_id(ids: Sequence[Optional[str]]) -> Optional[str]:
    """The id a merged file carries: the shared one, or ``mixed:<ids>`` (never equal to a real id)."""
    return ids[0] if len(ids) == 1 else "mixed:" + ",".join(map(str, ids))


def _aggregate_composite(
    suite: Suite, files: List[Path], out_dir: Path, allow_mixed_fp: bool = False
) -> Dict[str, Any]:
    by_seed: Dict[int, Path] = {}
    scores: Dict[int, float] = {}
    seeds = []
    ids: List[Optional[str]] = []
    envs: Dict[str, Any] = {}
    for f in files:
        data = json.loads(f.read_text())
        seed = int(data["config"]["seed"])
        if not data["config"].get("sync_eval", False):
            raise ValueError(f"{f}: not measured with sync_eval")
        seeds.append(seed)
        by_seed[seed] = f
        scores[seed] = float(data["composite_score"])
        ids.append(data.get("fp_env_id"))
        if data.get("fp_env_id"):
            envs[data["fp_env_id"]] = data.get("fp_env")
    _check_unique(suite.name, seeds)
    fp_ids = _fp_ids(suite, ids, allow_mixed_fp)
    written = []
    for seed in sorted(by_seed, key=_seed_order):
        dest = out_dir / f"ref_composite_{suite.ref_key}_s{seed}.json"
        shutil.copyfile(by_seed[seed], dest)
        written.append(dest.name)
    ordered = sorted(scores, key=_seed_order)
    mean = sum(scores.values()) / len(scores) if scores else float("nan")
    print(f"{suite.name}: {len(scores)} seed(s), composite mean {mean:.4f}")
    for s in ordered:
        print(f"    seed {s:>6d}  {scores[s]:.4f}")
    return {
        "seeds": ordered,
        "files": written,
        "composite_score": {str(s): scores[s] for s in ordered},
        "fp_env_ids": fp_ids,
        "_fp_envs": envs,
    }


def _aggregate_ioh(suite: Suite, files: List[Path], out_dir: Path, allow_mixed_fp: bool = False) -> Dict[str, Any]:
    from panobbgo.harness_ioh import IOHMultiSeedResult

    parts = [IOHMultiSeedResult.from_dict(json.loads(f.read_text())) for f in files]
    if not parts:
        return {"seeds": [], "files": []}
    names = {p.battery_name for p in parts}
    if len(names) != 1:
        raise ValueError(f"{suite.name}: shards from different batteries {sorted(names)}")
    if not all(p.sync_eval for p in parts):
        raise ValueError(f"{suite.name}: a shard was not measured with sync_eval")
    threads = {p.blas_threads for p in parts}
    if len(threads) != 1:
        raise ValueError(f"{suite.name}: shards measured with different BLAS thread counts {sorted(map(str, threads))}")
    fp_ids = _fp_ids(suite, [p.fp_env_id for p in parts], allow_mixed_fp)
    pairs = [(s, r) for p in parts for s, r in zip(p.base_seeds, p.results)]
    _check_unique(suite.name, [s for s, _ in pairs])
    pairs.sort(key=lambda sr: _seed_order(sr[0]))
    first = parts[0]
    combined = IOHMultiSeedResult(
        battery_name=first.battery_name,
        problem_kind=first.problem_kind,
        log_lo=first.log_lo,
        log_hi=first.log_hi,
        base_seeds=[s for s, _ in pairs],
        results=[r for _, r in pairs],
        sync_eval=True,
        blas_threads=first.blas_threads,
        fp_env=first.fp_env if len(fp_ids) == 1 else None,
        fp_env_id=_merged_fp_id(fp_ids),
    )
    stem = f"ref_ioh_{suite.ref_key}"
    written = [f"{stem}.json"]
    (out_dir / written[0]).write_text(combined.to_json())
    for seed, res in pairs:
        name = f"{stem}_s{seed}.json"
        (out_dir / name).write_text(res.to_json())
        written.append(name)
    print(f"{suite.name}: {len(pairs)} seed(s), mean AOCC {combined.mean_aocc:.4f}")
    for name, val in sorted(combined.per_strategy_aocc().items(), key=lambda kv: -kv[1]):
        print(f"    {name:32s}  {val:.4f}")
    return {
        "seeds": combined.base_seeds,
        "files": written,
        "mean_aocc": combined.mean_aocc,
        "fp_env_ids": fp_ids,
        "_fp_envs": {p.fp_env_id: p.fp_env for p in parts if p.fp_env_id},
    }


def _aggregate_families(suite: Suite, files: List[Path], out_dir: Path, allow_mixed_fp: bool = False) -> Dict[str, Any]:
    chunks = [json.loads(f.read_text()) for f in files]
    seeds = [s for rows in chunks for s in sorted({r["seed"] for r in rows})]
    _check_unique(suite.name, seeds)
    fp_ids = _fp_ids(suite, [r.get("fp_env_id") for rows in chunks for r in rows], allow_mixed_fp)
    rows = [r for c in chunks for r in c]
    # Stable sort: within a seed the screen's own row order is kept.
    rows.sort(key=lambda r: _seed_order(int(r["seed"])))
    name = f"ref_family_screen_{suite.ref_key}.json"
    (out_dir / name).write_text(json.dumps(rows))
    ordered = sorted(set(seeds), key=_seed_order)
    print(f"{suite.name}: {len(ordered)} seed(s), {len(rows)} rows (re-analyse: family_screen.py from={name})")
    return {"seeds": ordered, "files": [name], "rows": len(rows), "fp_env_ids": fp_ids, "_fp_envs": {}}


MANIFEST = "ref_MANIFEST.json"
SUMMARY = "SUMMARY.json"

_AGGREGATORS = {"composite": _aggregate_composite, "ioh": _aggregate_ioh, "families": _aggregate_families}


def missing_shards(planned: Sequence[Dict[str, str]], metas: Sequence[Dict[str, Any]]) -> List[str]:
    """``suite/shard`` of every planned matrix entry that left no meta file.

    A shard that dies before writing its meta (timeout, lost runner,
    cancelled job, failed install step) is otherwise invisible.
    """
    seen = {(str(m.get("suite")), str(m.get("shard"))) for m in metas}
    return [f"{e['suite']}/{e['shard']}" for e in planned if (e["suite"], str(e["shard"])) not in seen]


def aggregate(
    src: Path, out_dir: Path, planned: Optional[Sequence[Dict[str, str]]] = None, allow_mixed_fp: bool = False
) -> Dict[str, Any]:
    """Write the reference files for every suite found under ``src``; return the manifest.

    ``planned`` is the job matrix (``plan``'s ``include`` list): a planned
    shard without a meta file counts as failed.  Shards of one suite from
    different FP environments (``fp_env_id``) raise ``ValueError`` unless
    ``allow_mixed_fp``; the manifest records every id (``fp_env_ids``, per
    suite and overall) and what each stands for (``fp_env``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = [json.loads(p.read_text()) for p in sorted(src.rglob("meta_*.json"))]
    missing = missing_shards(planned or [], metas)
    manifest: Dict[str, Any] = {
        "created": datetime.now(tz=timezone.utc).isoformat(),
        "git_sha": sorted({m.get("git_sha", "unknown") for m in metas}),
        "github_run_id": sorted({str(m.get("github_run_id")) for m in metas if m.get("github_run_id")}),
        "failed_shards": [f"{m['suite']}/{m['shard']}" for m in metas if m.get("status")] + missing,
        "missing_shards": missing,
        "planned_shards": len(planned) if planned is not None else None,
        "suites": {},
        "shards": metas,
    }
    fp_envs: Dict[str, Any] = {m["fp_env"]["id"]: m["fp_env"] for m in metas if (m.get("fp_env") or {}).get("id")}
    for suite in SUITES.values():
        files = _result_files(src, suite.name)
        if files:
            info = _AGGREGATORS[suite.kind](suite, files, out_dir, allow_mixed_fp)
            fp_envs.update({k: v for k, v in info.pop("_fp_envs").items() if v})
            manifest["suites"][suite.name] = info
    if not manifest["suites"]:
        raise ValueError(f"no shard result files found under {src}")
    all_ids = sorted({i for v in manifest["suites"].values() for i in v["fp_env_ids"]}, key=str)
    manifest["fp_env_ids"] = all_ids
    manifest["mixed_fp"] = any(len(v["fp_env_ids"]) > 1 for v in manifest["suites"].values())
    manifest["fp_env"] = {k: fp_envs.get(k) for k in all_ids if k is not None}
    if len(all_ids) > 1:
        print(f"warning: the suites were measured in different FP environments: {all_ids}", file=sys.stderr)
    if len(manifest["git_sha"]) > 1:
        print(f"warning: shards come from several commits: {manifest['git_sha']}", file=sys.stderr)
    if manifest["failed_shards"]:
        print(f"warning: failed shards (their seeds are missing): {manifest['failed_shards']}", file=sys.stderr)
    seed_sets = {tuple(v["seeds"]) for v in manifest["suites"].values()}
    if len(seed_sets) > 1:
        print(f"warning: the suites cover different seed sets (see {MANIFEST})", file=sys.stderr)
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    write_summary(out_dir)
    return manifest


def cmd_aggregate(args: argparse.Namespace) -> int:
    out_dir = Path(
        args.out_dir or REPO_ROOT / "planning" / "results" / datetime.now(tz=timezone.utc).date().isoformat()
    )
    planned = None
    if args.plan:
        data = json.loads(Path(args.plan).read_text())
        planned = data["include"] if isinstance(data, dict) else data
    aggregate(Path(args.src), out_dir, planned, allow_mixed_fp=args.allow_mixed_fp)
    print(f"\nwrote the reference files to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def _stats(values: Sequence[float]) -> Dict[str, float]:
    vals = list(values)
    return {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)} if vals else {}


def _by_spec_mean(pairs: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for name, val in pairs:
        acc.setdefault(name, []).append(val)
    return dict(sorted(((k, sum(v) / len(v)) for k, v in acc.items()), key=lambda kv: -kv[1]))


def summarize(out_dir: Path) -> Dict[str, Any]:
    """The small, committable digest of an aggregated directory.

    Per suite the seeds and the numbers the DISCOVERY log cites: composite
    mean / min / max and per seed; IOH mean AOCC and per spec; families rows,
    failed rows and mean AOCC per spec.  ``release`` is ``None`` here;
    :func:`publish` fills it in.
    """
    manifest = json.loads((out_dir / MANIFEST).read_text())
    suites: Dict[str, Any] = {}
    for name, info in manifest["suites"].items():
        suite = SUITES[name]
        entry: Dict[str, Any] = {"seeds": info["seeds"], "files": info["files"]}
        if suite.kind == "composite":
            scores = {str(k): float(v) for k, v in info["composite_score"].items()}
            entry["composite_score"] = {**_stats(list(scores.values())), "per_seed": scores}
        elif suite.kind == "ioh":
            from panobbgo.harness_ioh import IOHMultiSeedResult

            data = json.loads((out_dir / f"ref_ioh_{suite.ref_key}.json").read_text())
            combined = IOHMultiSeedResult.from_dict(data)
            entry["mean_aocc"] = combined.mean_aocc
            entry["per_spec_aocc"] = dict(sorted(combined.per_strategy_aocc().items(), key=lambda kv: -kv[1]))
        else:
            rows = json.loads((out_dir / f"ref_family_screen_{suite.ref_key}.json").read_text())
            scored = [r for r in rows if r.get("aocc") is not None]
            entry["rows"] = len(rows)
            entry["failed_rows"] = sum(1 for r in rows if r.get("err"))
            entry["mean_aocc"] = sum(float(r["aocc"]) for r in scored) / len(scored) if scored else None
            entry["per_spec_aocc"] = _by_spec_mean([(str(r["s"]), float(r["aocc"])) for r in scored])
        suites[name] = entry
    return {
        "created": manifest.get("created"),
        "git_sha": manifest.get("git_sha", []),
        "github_run_id": manifest.get("github_run_id", []),
        "failed_shards": manifest.get("failed_shards", []),
        "fp_env_ids": manifest.get("fp_env_ids"),
        "release": None,
        "suites": suites,
    }


def write_summary(out_dir: Path, release: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """(Re)write ``SUMMARY.json``; a release recorded earlier is kept unless a new one is given."""
    summary = summarize(out_dir)
    path = out_dir / SUMMARY
    if release is None and path.exists():
        release = json.loads(path.read_text()).get("release")
    summary["release"] = release
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def summary_markdown(summary: Dict[str, Any]) -> str:
    """Markdown rendering of a summary: the release notes and the workflow's job summary."""
    run = run_label(summary)
    sha = ", ".join(s[:7] for s in summary.get("git_sha") or []) or "unknown"
    lines = [
        "Reference data of a re-baseline (`.github/workflows/rebaseline.yml`), not a software release.",
        "",
        f"Commit {sha}, workflow run {run}, aggregated {summary.get('created')}.",
        f"Failed shards: {', '.join(summary.get('failed_shards') or []) or 'none'}.",
        "",
        "| Suite | seeds | result |",
        "|---|---|---|",
    ]
    for name, e in summary["suites"].items():
        if "composite_score" in e:
            c = e["composite_score"]
            res = f"composite mean {c['mean']:.4f} (per seed {c['min']:.3f} … {c['max']:.3f})"
        elif "rows" in e:
            mean = "n/a" if e["mean_aocc"] is None else f"{e['mean_aocc']:.4f}"
            res = f"{e['rows']} rows ({e['failed_rows']} failed), mean AOCC {mean}"
        else:
            res = f"mean AOCC {e['mean_aocc']:.4f}"
        lines.append(f"| {name} | {len(e['seeds'])} | {res} |")
    lines += [
        "",
        "Unpack locally: `uv run python scripts/rebaseline.py fetch <tag>` (doc/dev/benchmarking.md).",
        "",
        # resolve_tag() recognises the release of its own run by this marker.
        _marker(run),
    ]
    return "\n".join(lines) + "\n"


def cmd_summarize(args: argparse.Namespace) -> int:
    if args.print:
        summary = json.loads((Path(args.dir) / SUMMARY).read_text())
    else:
        summary = write_summary(Path(args.dir))
    print(summary_markdown(summary))
    return 0


# ---------------------------------------------------------------------------
# publish / fetch (GitHub releases through the gh CLI)
# ---------------------------------------------------------------------------

TAG_PREFIX = "rebaseline-"
_TAG_DATE = re.compile(rf"^{TAG_PREFIX}(\d{{4}}-\d{{2}}-\d{{2}})")


def _gh(args: Sequence[str], check: bool = True) -> "subprocess.CompletedProcess[str]":
    """Run ``gh``: the only network access of this script (tests replace it)."""
    proc = subprocess.run(["gh", *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        proc.check_returncode()
    return proc


def _repo_args(repo: Optional[str]) -> List[str]:
    return ["--repo", repo] if repo else []


def default_tag(manifest: Dict[str, Any]) -> str:
    """``rebaseline-<UTC date the run started>``: the earliest shard start, else the aggregation time.

    Keyed to the start, not the aggregation, so a re-run of the aggregate
    job after midnight names the same release.
    """
    starts = [m["started"] for m in manifest.get("shards", []) if m.get("started")]
    stamp = min(starts, default=None) or manifest.get("created") or datetime.now(tz=timezone.utc).isoformat()
    return f"{TAG_PREFIX}{stamp[:10]}"


#: Release states :func:`resolve_tag` reports.
ABSENT, DRAFT, PUBLISHED = "absent", "draft", "published"


def run_label(summary: Dict[str, Any]) -> str:
    """The run a summary comes from, as its release notes' marker names it (``local`` without one)."""
    return ", ".join(summary.get("github_run_id") or []) or "local"


def _marker(label: str) -> str:
    return f"<!-- rebaseline-run: {label} -->"


def repo_name(repo: Optional[str]) -> str:
    """``owner/name``; default: this checkout's GitHub repository."""
    if repo:
        return repo
    return _gh(["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"]).stdout.strip()


def list_releases(repo: str) -> Dict[str, Tuple[str, str]]:
    """Every release (drafts included) as ``tag -> (state, notes)``.

    One listing instead of ``gh release view`` per tag: a failed call raises
    rather than passing for "no such release".
    """
    jq = ".[] | {tag: .tag_name, draft: .draft, body: .body}"
    out = _gh(["api", "--paginate", f"repos/{repo}/releases?per_page=100", "--jq", jq]).stdout
    releases = {}
    for line in out.splitlines():
        if line.strip():
            r = json.loads(line)
            releases[r["tag"]] = (DRAFT if r.get("draft") else PUBLISHED, r.get("body") or "")
    return releases


def tag_commit(tag: str, repo: str) -> Optional[str]:
    """The commit the git tag ``tag`` points at, or ``None`` when there is no such tag.

    Uses the exact-match endpoint ``git/ref/`` (``git/refs/`` also returns
    prefix matches).  Anything but a 404 raises.
    """
    proc = _gh(["api", f"repos/{repo}/git/ref/tags/{tag}"], check=False)
    if proc.returncode != 0:
        if "HTTP 404" in proc.stderr:
            return None
        raise RuntimeError(f"cannot look up tag {tag}: {proc.stderr.strip()}")
    obj = json.loads(proc.stdout)["object"]
    if obj["type"] == "tag":  # annotated: dereference to the commit
        obj = json.loads(_gh(["api", f"repos/{repo}/git/tags/{obj['sha']}"]).stdout)["object"]
    return obj["sha"]


def _same_commit(a: str, b: str) -> bool:
    return a.startswith(b) or b.startswith(a)


def resolve_tag(
    summary: Dict[str, Any], tag: Optional[str], repo: str, target: Optional[str], default: str
) -> Tuple[str, str]:
    """``(tag, state)`` of the release this run publishes to (``tag`` if given, else ``default``).

    *   A release whose notes carry this run's marker is this run's: it is
        reused (whatever its tag, so a re-run after midnight does not publish
        a duplicate).  Only with a single workflow run id is every release
        searched; otherwise only the explicit ``tag``.
    *   Otherwise the tag must be free: no release and no git tag.  A taken
        tag (another run's release, or a bare tag, e.g. one burned by a
        deleted immutable release) gives ``<tag>-run<run_id>``, which must
        be free in turn.
    *   Without a single run id (a local publish) ``tag`` is required and
        there is no fallback.
    *   A tag that exists must point at ``target``.
    """
    runs = summary.get("github_run_id") or []
    run_id = runs[0] if len(runs) == 1 else ""
    if not run_id and not tag:
        raise ValueError("no single workflow run id in the manifest: name the release with --tag")
    releases = list_releases(repo)
    mine = _marker(run_label(summary))
    own = [t for t, (_, body) in releases.items() if mine in body and (run_id or t == tag)]
    if len(own) > 1:
        raise ValueError(f"several releases carry this run's marker: {sorted(own)}")
    if own:
        chosen, state = own[0], releases[own[0]][0]
    else:
        chosen = tag or default
        if chosen in releases or tag_commit(chosen, repo) is not None:
            if not run_id:
                raise ValueError(f"tag {chosen} is taken (a release or a bare git tag); choose another --tag")
            chosen = f"{chosen}-run{run_id}"
            if chosen in releases or tag_commit(chosen, repo) is not None:
                raise ValueError(f"tag {chosen} is taken too; name the release with --tag")
        state = ABSENT
    if state != PUBLISHED and target:
        sha = tag_commit(chosen, repo)
        if sha is not None and not _same_commit(sha, target):
            raise ValueError(f"tag {chosen} points at {sha}, not at the measured commit {target}")
    return chosen, state


def release_url(repo: str, tag: str) -> str:
    """The release page of ``tag`` in ``repo`` (``owner/name``)."""
    return f"https://github.com/{repo}/releases/tag/{tag}"


def make_tarball(out_dir: Path, tag: str, dest: Path) -> Path:
    """``dest/<tag>.tar.gz``, holding every ``ref_*.json`` of ``out_dir`` flat (no directory)."""
    files = sorted(out_dir.glob("ref_*.json"))
    if not files:
        raise ValueError(f"no ref_*.json files in {out_dir}")
    tarball = dest / f"{tag}.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=f.name)
    return tarball


def publish(
    out_dir: Path,
    tag: Optional[str],
    repo: Optional[str],
    target: Optional[str],
    measure_result: Optional[str] = None,
) -> Dict[str, Any]:
    """Create the release of an aggregated directory and record it in ``SUMMARY.json``.

    The release is a pre-release never marked latest, so it does not pose as
    a software release; a new tag points at ``target`` (the measured commit).
    The repository has *immutable releases* enabled: assets can only be
    attached to a draft, so the release is created as a draft, filled, then
    published.  A draft left by an interrupted attempt is resumed; this run's
    release that is already published cannot change and is left as it is.
    Which tag: :func:`resolve_tag`.  The default tag is refused for an
    incomplete run — failed or missing shards, or ``measure_result`` (the
    workflow's ``needs.measure.result``) other than ``success``; an explicit
    ``tag`` publishes anyway.  The local ``SUMMARY.json`` records the release
    only once it is published; for a release published earlier, its own
    ``SUMMARY.json`` and manifest are downloaded into ``out_dir`` instead.
    """
    summary = write_summary(out_dir)
    if tag is None and summary.get("failed_shards"):
        raise ValueError(
            f"failed shards {summary['failed_shards']}: not publishing an incomplete re-baseline "
            "under the default tag (name it explicitly with --tag to publish anyway)"
        )
    if tag is None and measure_result not in (None, "success"):
        raise ValueError(
            f"the measure jobs ended '{measure_result}': not publishing under the default tag "
            "(name it explicitly with --tag to publish anyway)"
        )
    repo = repo_name(repo)
    manifest = json.loads((out_dir / MANIFEST).read_text())
    tag, state = resolve_tag(summary, tag, repo, target, default_tag(manifest))
    release = {"tag": tag, "url": release_url(repo, tag), "asset": f"{tag}.tar.gz"}
    if state == PUBLISHED:
        # Immutable: make the local copies match what the release holds.
        download = ["release", "download", tag, "--dir", str(out_dir), "--clobber"]
        _gh(download + ["--pattern", SUMMARY, "--pattern", MANIFEST, *_repo_args(repo)])
        print(f"{release['url']} is already published (immutable); nothing uploaded, its {SUMMARY} kept")
        return json.loads((out_dir / SUMMARY).read_text())
    summary["release"] = release
    with tempfile.TemporaryDirectory() as tmp:
        # The uploaded SUMMARY.json names the release; the local one only
        # does once the release is published (below).
        (Path(tmp) / SUMMARY).write_text(json.dumps(summary, indent=2) + "\n")
        notes = Path(tmp) / "notes.md"
        notes.write_text(summary_markdown(summary))
        tarball = make_tarball(out_dir, tag, Path(tmp))
        label = tag[len(TAG_PREFIX) :] if tag.startswith(TAG_PREFIX) else tag
        if state == ABSENT:
            create = ["release", "create", tag, "--title", f"Re-baseline {label} (reference data)"]
            create += ["--notes-file", str(notes), "--draft", "--prerelease", "--latest=false"]
            create += ["--target", target] if target else []
            _gh(create + _repo_args(repo))
        assets = [str(tarball), str(out_dir / MANIFEST), str(Path(tmp) / SUMMARY)]
        _gh(["release", "upload", tag, *assets, "--clobber", *_repo_args(repo)])
        finish = ["release", "edit", tag, "--notes-file", str(notes), "--draft=false", "--prerelease", "--latest=false"]
        _gh(finish + _repo_args(repo))
    summary = write_summary(out_dir, release)
    print(f"published {tarball.name}, {MANIFEST} and {SUMMARY} to {release['url']}")
    return summary


def cmd_publish(args: argparse.Namespace) -> int:
    publish(Path(args.dir), args.tag, args.repo, args.target, args.measure_result)
    return 0


def default_fetch_dir(tag: str) -> Path:
    """``planning/results/<date>/`` for ``rebaseline-<date>...``, else ``planning/results/<tag>/``."""
    m = _TAG_DATE.match(tag)
    return REPO_ROOT / "planning" / "results" / (m.group(1) if m else tag)


def fetch(tag: str, dest: Optional[Path] = None, repo: Optional[str] = None) -> List[Path]:
    """Download a re-baseline release and unpack its reference files into ``dest``; return them.

    Only the asset ``<tag>.tar.gz`` is unpacked (in a temporary directory, so
    nothing else in ``dest`` is touched).  ``ref_MANIFEST.json`` and
    ``SUMMARY.json`` are committed files: an existing copy is kept — skipped
    when identical, a warning when it differs — and written only when absent.
    """
    dest = dest or default_fetch_dir(tag)
    dest.mkdir(parents=True, exist_ok=True)
    asset = f"{tag}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        download = ["release", "download", tag, "--dir", tmp, "--pattern", asset, "--pattern", SUMMARY]
        _gh(download + _repo_args(repo))
        if not (tmp_dir / asset).exists():
            raise ValueError(f"release {tag} has no asset {asset}")
        unpacked_dir = tmp_dir / "unpacked"
        with tarfile.open(tmp_dir / asset) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            tar.extractall(unpacked_dir, members=members, filter="data")
        if (tmp_dir / SUMMARY).exists():
            shutil.copyfile(tmp_dir / SUMMARY, unpacked_dir / SUMMARY)
        files: List[Path] = []
        for src in sorted(unpacked_dir.iterdir()):
            out = dest / src.name
            if src.name in (MANIFEST, SUMMARY) and out.exists():
                if out.read_bytes() != src.read_bytes():
                    print(f"warning: {out} differs from the release's copy; kept the local file", file=sys.stderr)
            else:
                shutil.copyfile(src, out)
            if src.name != SUMMARY:
                files.append(out)
    print(f"unpacked {len(files)} reference files of {tag} into {dest}")
    return files


def cmd_fetch(args: argparse.Namespace) -> int:
    fetch(args.tag, Path(args.dir) if args.dir else None, args.repo)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    plan_p = sub.add_parser("plan", help="print the job matrix as JSON")
    plan_p.add_argument("--suites", default="all", help=f"'all' or a comma list of: {', '.join(SUITES)}")
    plan_p.add_argument(
        "--seeds", default=str(len(ROSTER)), help="a count (the first N roster seeds) or a comma list (one seed: '42,')"
    )
    plan_p.set_defaults(func=cmd_plan)

    run_p = sub.add_parser("run", help="run one shard")
    run_p.add_argument("suite", choices=list(SUITES))
    run_p.add_argument("--seeds", required=True, help="comma list of base seeds (a shard's matrix entry)")
    run_p.add_argument("--shard", default="01", help="shard label (names the result file)")
    run_p.add_argument("--out-dir", default="rebaseline-out")
    run_p.add_argument("--jobs", type=int, default=0, help="worker processes (default: all cores)")
    run_p.set_defaults(func=cmd_run)

    agg_p = sub.add_parser("aggregate", help="build the reference files from downloaded artifacts")
    agg_p.add_argument("src", help="directory holding the downloaded artifacts")
    agg_p.add_argument("--out-dir", help="default: planning/results/<UTC date>/")
    agg_p.add_argument(
        "--plan", help="the job matrix (JSON of `plan`): planned shards that left no meta file count as failed"
    )
    agg_p.add_argument(
        "--allow-mixed-fp",
        action="store_true",
        help="merge shards of one suite measured in different FP environments (fp_env_id) instead of refusing",
    )
    agg_p.set_defaults(func=cmd_aggregate)

    sum_p = sub.add_parser("summarize", help="(re)write SUMMARY.json of an aggregated directory")
    sum_p.add_argument("dir", help="directory holding ref_MANIFEST.json and the ref_* files")
    sum_p.add_argument("--print", action="store_true", help="only render the existing SUMMARY.json, do not rewrite it")
    sum_p.set_defaults(func=cmd_summarize)

    pub_p = sub.add_parser("publish", help="create or update the GitHub release of an aggregated directory")
    pub_p.add_argument("dir", help="directory holding ref_MANIFEST.json and the ref_* files")
    pub_p.add_argument(
        "--tag",
        help=f"release tag (default: {TAG_PREFIX}<UTC date the run started>; required without a single run id)",
    )
    pub_p.add_argument("--repo", help="owner/name (default: this checkout's GitHub repository)")
    pub_p.add_argument("--target", help="commit a new tag points at: the measured commit")
    pub_p.add_argument(
        "--measure-result",
        help="the workflow's needs.measure.result; anything but 'success' refuses the default tag",
    )
    pub_p.set_defaults(func=cmd_publish)

    fetch_p = sub.add_parser("fetch", help="download and unpack the reference files of a re-baseline release")
    fetch_p.add_argument("tag", help=f"release tag, e.g. {TAG_PREFIX}2026-09-26-run36228301268")
    fetch_p.add_argument("--dir", help="target directory (default: planning/results/<date of the tag>/)")
    fetch_p.add_argument("--repo", help="owner/name (default: this checkout's GitHub repository)")
    fetch_p.set_defaults(func=cmd_fetch)
    return p


def _pin_fp() -> None:
    """Pin the FP kernels for this process and its shard commands (``panobbgo.fp_env``).

    ``plan`` runs with a bare interpreter (no dependencies, panobbgo not
    importable): then there is nothing to pin, and the shard commands pin
    themselves anyway.
    """
    try:
        from panobbgo.fp_env import pin_fp_env
    except ImportError:
        return
    pin_fp_env()


def main(argv: Optional[List[str]] = None) -> int:
    _pin_fp()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
