#!/usr/bin/env python
# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
"""Re-measure the reference baselines, sharded over GitHub runners.

The engine behind ``.github/workflows/rebaseline.yml`` (TODO.md "Re-baseline
once").  Three subcommands:

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
    files, in the formats the comparison tools read::

        gh run download RUN_ID --pattern 'rebaseline-*' --dir rebaseline-raw
        uv run python scripts/rebaseline.py aggregate rebaseline-raw

Suites and the files ``aggregate`` writes (default into
``planning/results/<UTC date>/``):

==========================  =================================================
suite                       reference file(s)
==========================  =================================================
``composite-quick``         ``ref_composite_quick_s<seed>.json`` — one
``composite-standard``      ``benchmark_harness.py run`` result per seed, for
                            ``benchmark_harness.py compare``
``ioh-quick``               ``ref_ioh_<battery>.json`` — one multi-seed
``ioh-standard``            result (``ioh_benchmark.py run --baselines
                            --seeds ...``) for ``ioh_benchmark.py compare``,
                            plus ``ref_ioh_<battery>_s<seed>.json`` single-seed
                            files for a plain ``run --output`` A/B
``families-free``           ``ref_family_screen_<preset>.json`` — the rows
``families-constrained``    file of ``benchmarks/family_screen.py``;
                            re-analyse with ``family_screen.py from=FILE``
==========================  =================================================

Every measurement is synchronous (``sync_eval``) and seeded, with no
wall-clock limit, so a result depends on the code and the seed only — not
on the runner's speed or the ``--jobs`` count.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    #: The mode (``quick`` / ``standard``) or the family preset (``free`` / ``constrained``).
    battery: str
    #: Seeds per job — sized so a job stays far below GitHub's 6 h limit.
    seeds_per_job: int


SUITES: Dict[str, Suite] = {
    s.name: s
    for s in (
        Suite("composite-quick", "composite", "quick", 12),
        Suite("composite-standard", "composite", "standard", 3),
        Suite("ioh-quick", "ioh", "quick", 12),
        Suite("ioh-standard", "ioh", "standard", 1),
        Suite("families-free", "families", "free", 2),
        Suite("families-constrained", "families", "constrained", 3),
    )
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
    """One matrix entry per (suite, seed chunk)."""
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


def _aggregate_composite(suite: Suite, files: List[Path], out_dir: Path) -> Dict[str, Any]:
    by_seed: Dict[int, Path] = {}
    scores: Dict[int, float] = {}
    seeds = []
    for f in files:
        data = json.loads(f.read_text())
        seed = int(data["config"]["seed"])
        if not data["config"].get("sync_eval", False):
            raise ValueError(f"{f}: not measured with sync_eval")
        seeds.append(seed)
        by_seed[seed] = f
        scores[seed] = float(data["composite_score"])
    _check_unique(suite.name, seeds)
    written = []
    for seed in sorted(by_seed, key=_seed_order):
        dest = out_dir / f"ref_composite_{suite.battery}_s{seed}.json"
        shutil.copyfile(by_seed[seed], dest)
        written.append(dest.name)
    ordered = sorted(scores, key=_seed_order)
    mean = sum(scores.values()) / len(scores) if scores else float("nan")
    print(f"{suite.name}: {len(scores)} seed(s), composite mean {mean:.4f}")
    for s in ordered:
        print(f"    seed {s:>6d}  {scores[s]:.4f}")
    return {"seeds": ordered, "files": written, "composite_score": {str(s): scores[s] for s in ordered}}


def _aggregate_ioh(suite: Suite, files: List[Path], out_dir: Path) -> Dict[str, Any]:
    from panobbgo.harness_ioh import IOHMultiSeedResult

    parts = [IOHMultiSeedResult.from_dict(json.loads(f.read_text())) for f in files]
    if not parts:
        return {"seeds": [], "files": []}
    names = {p.battery_name for p in parts}
    if len(names) != 1:
        raise ValueError(f"{suite.name}: shards from different batteries {sorted(names)}")
    if not all(p.sync_eval for p in parts):
        raise ValueError(f"{suite.name}: a shard was not measured with sync_eval")
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
    )
    stem = f"ref_ioh_{suite.battery}"
    written = [f"{stem}.json"]
    (out_dir / written[0]).write_text(combined.to_json())
    for seed, res in pairs:
        name = f"{stem}_s{seed}.json"
        (out_dir / name).write_text(res.to_json())
        written.append(name)
    print(f"{suite.name}: {len(pairs)} seed(s), mean AOCC {combined.mean_aocc:.4f}")
    for name, val in sorted(combined.per_strategy_aocc().items(), key=lambda kv: -kv[1]):
        print(f"    {name:32s}  {val:.4f}")
    return {"seeds": combined.base_seeds, "files": written, "mean_aocc": combined.mean_aocc}


def _aggregate_families(suite: Suite, files: List[Path], out_dir: Path) -> Dict[str, Any]:
    chunks = [json.loads(f.read_text()) for f in files]
    seeds = [s for rows in chunks for s in sorted({r["seed"] for r in rows})]
    _check_unique(suite.name, seeds)
    rows = [r for c in chunks for r in c]
    # Stable sort: within a seed the screen's own row order is kept.
    rows.sort(key=lambda r: _seed_order(int(r["seed"])))
    name = f"ref_family_screen_{suite.battery}.json"
    (out_dir / name).write_text(json.dumps(rows))
    ordered = sorted(set(seeds), key=_seed_order)
    print(f"{suite.name}: {len(ordered)} seed(s), {len(rows)} rows (re-analyse: family_screen.py from={name})")
    return {"seeds": ordered, "files": [name], "rows": len(rows)}


_AGGREGATORS = {"composite": _aggregate_composite, "ioh": _aggregate_ioh, "families": _aggregate_families}


def aggregate(src: Path, out_dir: Path) -> Dict[str, Any]:
    """Write the reference files for every suite found under ``src``; return the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = [json.loads(p.read_text()) for p in sorted(src.rglob("meta_*.json"))]
    manifest: Dict[str, Any] = {
        "created": datetime.now(tz=timezone.utc).isoformat(),
        "git_sha": sorted({m.get("git_sha", "unknown") for m in metas}),
        "github_run_id": sorted({str(m.get("github_run_id")) for m in metas if m.get("github_run_id")}),
        "failed_shards": [f"{m['suite']}/{m['shard']}" for m in metas if m.get("status")],
        "suites": {},
        "shards": metas,
    }
    for suite in SUITES.values():
        files = _result_files(src, suite.name)
        if files:
            manifest["suites"][suite.name] = _AGGREGATORS[suite.kind](suite, files, out_dir)
    if not manifest["suites"]:
        raise ValueError(f"no shard result files found under {src}")
    if len(manifest["git_sha"]) > 1:
        print(f"warning: shards come from several commits: {manifest['git_sha']}", file=sys.stderr)
    if manifest["failed_shards"]:
        print(f"warning: failed shards (their seeds are missing): {manifest['failed_shards']}", file=sys.stderr)
    seed_sets = {tuple(v["seeds"]) for v in manifest["suites"].values()}
    if len(seed_sets) > 1:
        print("warning: the suites cover different seed sets (see ref_MANIFEST.json)", file=sys.stderr)
    (out_dir / "ref_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def cmd_aggregate(args: argparse.Namespace) -> int:
    out_dir = Path(
        args.out_dir or REPO_ROOT / "planning" / "results" / datetime.now(tz=timezone.utc).date().isoformat()
    )
    aggregate(Path(args.src), out_dir)
    print(f"\nwrote the reference files to {out_dir}")
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
    agg_p.set_defaults(func=cmd_aggregate)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
