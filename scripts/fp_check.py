#!/usr/bin/env python
# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
"""Check that one seeded screen gives bit-identical numbers on several machines.

The engine behind ``.github/workflows/fp-check.yml``: every matrix job runs
the same family screen (``benchmarks/family_screen.py``, one seed) and
saves its rows plus its FP environment (``python -m panobbgo.fp_env``) into
``<dir>/<job>/rows.json`` and ``<dir>/<job>/fp_env.json``.  This script
reads them all, prints one line per job (CPU, AVX-512, BLAS kernels, numpy
SIMD targets, ``fp_env_id``, a digest of the rows) and exits 1 unless every
job produced the same rows::

    uv run python scripts/fp_check.py fp-check-raw

The digest leaves ``fp_env_id`` out of the rows, so the check is about the
numbers; the ids are reported next to it.  See ``doc/dev/benchmarking.md``,
"Evaluations, not wall time".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROWS = "rows.json"
FP_ENV = "fp_env.json"


def rows_digest(rows: List[Dict[str, Any]]) -> str:
    """A hash of the rows without their ``fp_env_id`` (the numbers only)."""
    clean = [{k: v for k, v in r.items() if k != "fp_env_id"} for r in rows]
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()[:12]


def load_jobs(src: Path) -> List[Dict[str, Any]]:
    """One entry per job directory under ``src`` that holds a ``rows.json``."""
    jobs = []
    for rows_path in sorted(src.rglob(ROWS)):
        env_path = rows_path.with_name(FP_ENV)
        rows = json.loads(rows_path.read_text())
        env: Optional[Dict[str, Any]] = json.loads(env_path.read_text()) if env_path.exists() else None
        jobs.append({"job": rows_path.parent.name, "rows": rows, "fp_env": env, "digest": rows_digest(rows)})
    return jobs


def n_differing(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> int:
    """Rows that differ between two row lists (compared in order, ``fp_env_id`` ignored)."""
    strip = [[{k: v for k, v in r.items() if k != "fp_env_id"} for r in rows] for rows in (a, b)]
    return sum(x != y for x, y in zip(*strip)) + abs(len(a) - len(b))


def describe(env: Optional[Dict[str, Any]]) -> str:
    if not env:
        return "fp_env missing"
    blas = ",".join(sorted({str(b.get("architecture")) for b in env.get("blas") or []}))
    avx512 = (env.get("isa") or {}).get("avx512f")
    simd = ",".join(env.get("numpy_simd") or [])
    return f"{env.get('cpu')} | avx512f={avx512} | blas={blas} | numpy_simd={simd} | id={env.get('id')}"


def check(src: Path) -> int:
    jobs = load_jobs(src)
    if not jobs:
        print(f"no {ROWS} under {src}", file=sys.stderr)
        return 1
    ref = jobs[0]
    print(f"{len(jobs)} job(s); reference: {ref['job']}\n")
    for j in jobs:
        diff = n_differing(ref["rows"], j["rows"])
        print(f"{j['job']:>10s}  rows={len(j['rows'])}  digest={j['digest']}  differing_rows={diff}")
        print(f"{'':>10s}  {describe(j['fp_env'])}")
    digests = {j["digest"] for j in jobs}
    ids = sorted({str((j["fp_env"] or {}).get("id")) for j in jobs})
    cpus = sorted({str((j["fp_env"] or {}).get("cpu")) for j in jobs})
    print(f"\nCPUs: {cpus}")
    print(f"fp_env_ids: {ids}")
    if len(digests) == 1:
        print(f"OK: all {len(jobs)} jobs produced bit-identical rows")
        return 0
    print(f"FAIL: {len(digests)} different results across {len(jobs)} jobs")
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="directory with one sub-directory per job (rows.json, fp_env.json)")
    return check(Path(p.parse_args(argv).src))


if __name__ == "__main__":
    raise SystemExit(main())
