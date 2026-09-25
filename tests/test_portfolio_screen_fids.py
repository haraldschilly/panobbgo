# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
"""Tests for ``benchmarks/portfolio_screen.py``'s function axis.

The screen is a script: its analysis runs in ``main()`` on
``sys.argv``.  So these tests drive it the way a user does — a synthetic
rows file and ``from=...`` — and read the printed report.  That also
covers the part a unit test of a helper could not: that a *rows file
without a fid* (every results file written before 2026-09-14) still folds
into the same cells and prints no class block at all.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO = Path(__file__).resolve().parents[1]
SCREEN = REPO / "benchmarks" / "portfolio_screen.py"

#: Two specs that exist in the screen's ``SPECS`` table; ``CMAES_alone`` is
#: one of the ``_alone`` references the delta blocks key on.
REF = "CMAES_alone"
ALT = "Blocks_uniform_cj_warm2"


def _row(
    seed: int, spec: str, aocc: float, *, fid: Optional[int] = None, dim: int = 2, inst: int = 0
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "seed": seed,
        "s": spec,
        "dim": dim,
        "inst": inst,
        "aocc": aocc,
        "obs": None,
        "evals": 400,
        "budget": 400,
        "err": None,
    }
    if fid is not None:
        row["fid"] = fid
    return row


def _run(tmp_path: Path, rows: List[Dict[str, Any]]) -> str:
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows))
    proc = subprocess.run(
        [sys.executable, str(SCREEN), f"from={path}"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def _fid_rows() -> List[Dict[str, Any]]:
    """Three seeds x 24 fids x 2 specs.

    ``ALT`` beats ``REF`` by +0.10 on the separable functions (f1-f5) and
    loses 0.02 everywhere else, so the pooled mean is a *win* while four of
    the five classes are negative — exactly the shape the per-class block
    exists to expose.
    """
    rows: List[Dict[str, Any]] = []
    for seed in (42, 7, 1234):
        for fid in range(1, 25):
            base = 0.5 + 0.001 * fid + 0.0001 * seed
            rows.append(_row(seed, REF, base, fid=fid))
            rows.append(_row(seed, ALT, base + (0.10 if fid <= 5 else -0.02), fid=fid))
    return rows


class TestFidAxisReport:
    def test_cells_fold_on_the_extended_key(self, tmp_path: Path) -> None:
        out = _run(tmp_path, _fid_rows())
        # 3 seeds x 24 fids x 1 dim x 1 instance — the fid is part of the
        # key, so the 24 functions are 24 cells and not one averaged cell.
        assert "cells: 72" in out
        # ...and the header says how wide the axis was, on a re-analysis too.
        assert "24 fids" in out

    def test_prints_the_five_classes_present(self, tmp_path: Path) -> None:
        out = _run(tmp_path, _fid_rows())
        assert "--- per COCO class" in out
        for cls in ("separable", "low-cond", "high-cond", "multimodal-global", "multimodal-weak"):
            assert cls in out

    def test_per_class_means_and_neg_count(self, tmp_path: Path) -> None:
        out = _run(tmp_path, _fid_rows())
        block = out.split(f"delta vs {REF}, per class")[1]
        line = next(ln for ln in block.splitlines() if ln.startswith(ALT))
        # pooled: (5*0.10 + 19*(-0.02)) / 24 = +0.0050
        assert "+0.0050" in line
        assert "+0.1000" in line  # separable
        assert "-0.0200" in line  # every other class
        # four of the five classes are negative — the pooled win is carried
        # by one landscape group.
        assert line.rstrip().endswith("4/5")

    def test_only_classes_present_are_printed(self, tmp_path: Path) -> None:
        rows = [_row(seed, spec, 0.5, fid=fid) for seed in (42, 7) for fid in (1, 2) for spec in (REF, ALT)]
        out = _run(tmp_path, rows)
        assert "separable" in out
        for absent in ("low-cond", "high-cond", "multimodal-global", "multimodal-weak"):
            assert absent not in out


class TestWithoutFids:
    def test_rows_without_a_fid_print_no_class_block(self, tmp_path: Path) -> None:
        rows = [
            _row(seed, spec, 0.5 + 0.01 * i, dim=dim, inst=inst)
            for seed in (42, 7)
            for i, (dim, inst) in enumerate([(2, 0), (2, 1), (5, 0)])
            for spec in (REF, ALT)
        ]
        out = _run(tmp_path, rows)
        assert "per COCO class" not in out
        assert "cells: 6" in out
        assert f"delta vs {REF}" in out  # the pooled block is untouched

    def test_mixed_rows_group_only_the_ones_with_a_fid(self, tmp_path: Path) -> None:
        rows = _fid_rows() + [_row(9, spec, 0.4) for spec in (REF, ALT)]
        out = _run(tmp_path, rows)
        assert "cells: 73" in out
        assert "--- per COCO class" in out


class TestScreenCli:
    def test_fids_option_is_rejected_on_a_kind_without_the_axis(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCREEN), "out.json", "42", "kind=standard", "fids=1,2"],
            capture_output=True,
            text=True,
            cwd=str(REPO),
        )
        assert proc.returncode != 0
        assert "BBOB problem kind" in (proc.stdout + proc.stderr)

    @pytest.mark.parametrize("bad", ["0", "25"])
    def test_fids_option_validates_the_range(self, bad: str) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCREEN), "out.json", "42", "kind=bbob", f"fids={bad}"],
            capture_output=True,
            text=True,
            cwd=str(REPO),
        )
        assert proc.returncode != 0
        assert "1..24" in (proc.stdout + proc.stderr)


def test_short_runs_name_their_function(tmp_path: Path) -> None:
    """A run below budget names its fid — with 24 functions, ``d2i0`` alone is ambiguous."""
    rows = _fid_rows()
    stalled = next(r for r in rows if r["s"] == ALT and r["fid"] == 7)
    stalled["evals"] = 100
    out = _run(tmp_path, rows)
    assert "f7d2i0 100/400" in out


def test_short_runs_without_fid_keep_the_old_label(tmp_path: Path) -> None:
    rows = [_row(seed, spec, 0.5) for seed in (42, 7) for spec in (REF, ALT)]
    rows[1]["evals"] = 100
    out = _run(tmp_path, rows)
    assert " d2i0 100/400" in out
