# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
"""The screens' shared t-interval, and their cell keys on the fid axis.

The benchmark screens used to hand-code t-critical values that disagreed
(2.5 for any n > 6 in ``arm_sweep``, 2.0 for n > 12 in ``np_accept``, 2.26
for every n > 8 elsewhere) and to key them on the number of seeds rather
than the deltas present.  They now all call :func:`harness_ioh.t_ci`.

``oracle`` / ``meta_screen`` rows carried no ``fid``, so on the BBOB axis
every function folded into one cell.  Driven like a user does: a synthetic
rows file and ``--from``.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from scipy.stats import t as t_dist

from panobbgo.harness_ioh import t_ci

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("n", [2, 3, 7, 9, 12, 13, 30])
def test_t_ci_uses_the_exact_critical_value(n):
    ds = np.random.default_rng(n).normal(0.01, 0.02, size=n)
    m, h = t_ci(ds)
    assert m == pytest.approx(float(ds.mean()))
    expected = float(t_dist.ppf(0.975, n - 1)) * float(ds.std(ddof=1)) / math.sqrt(n)
    assert h == pytest.approx(expected, rel=1e-12)


def test_t_ci_degenerate_sizes():
    m, h = t_ci([])
    assert math.isnan(m) and math.isnan(h)
    m, h = t_ci([0.25])
    assert m == 0.25 and math.isnan(h)


def _run(script: str, tmp_path: Path, rows: List[Dict[str, Any]], *extra: str) -> str:
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows))
    proc = subprocess.run(
        [sys.executable, str(REPO / "benchmarks" / script), f"from={path}", *extra],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_oracle_keeps_fids_apart(tmp_path):
    rows = [
        {"seed": seed, "arm": arm, "fid": fid, "dim": 2, "inst": 0, "rep": 0, "aocc": a, "err": None}
        for seed in (1, 2, 3)
        for fid in (1, 2, 3, 4)
        for arm, a in (("cmaes", 0.5 + 0.01 * fid), ("jso", 0.5 - 0.01 * fid))
    ]
    out = _run("oracle.py", tmp_path, rows)
    # 3 seeds × 4 fids, not 3 seeds × one merged (dim, inst) cell.
    assert "cells: 12 complete" in out
    assert "f4d2i0" in out


def test_meta_screen_keeps_fids_apart(tmp_path):
    rows = [
        {
            "seed": seed,
            "s": s,
            "fid": fid,
            "dim": 2,
            "inst": 0,
            "rep": 0,
            "aocc": 0.5 + (0.01 if s == "Meta_never" else 0.0),
            "evals": 400,
            "budget": 400,
            "err": None,
        }
        for seed in (1, 2, 3)
        for fid in (1, 2)
        for s in ("Blocks_uniform_cj_warm2", "Meta_never")
    ]
    out = _run("meta_screen.py", tmp_path, rows)
    assert "cells: 6" in out


def test_run_seeds_interrupt_keeps_the_finished_rows(tmp_path, monkeypatch):
    """Ctrl-C while the rows file is rewritten must not destroy the rows already on disk."""
    from benchmarks import _screen

    out = tmp_path / "rows.json"
    real_dump = json.dump
    calls = {"n": 0}

    def dump(obj, fp, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real_dump(obj, fp, *a, **kw)

    monkeypatch.setattr(_screen.json, "dump", dump)
    with pytest.raises(KeyboardInterrupt):
        _screen.run_seeds([1, 2], str(out), lambda seed: [[{"seed": seed}]])
    assert json.loads(out.read_text()) == [{"seed": 1}]
