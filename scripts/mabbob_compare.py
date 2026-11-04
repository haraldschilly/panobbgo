#!/usr/bin/env python
# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
"""Compare Panobbgo with published MA-BBOB competition entries on the official instances.

``compare`` reads result files of ``ioh_benchmark.py run --mabbob-official``
(or ``--mabbob-official-heldout``; a re-baseline ``ref_ioh_*`` file works too)
and compares every strategy with every entry, per split, per dimension and on
the mean of the two dimensions: mean AOCC, the paired per-instance delta with a
bootstrap CI over instances, wins and losses, and a breakdown by the BBOB base
functions that dominate an instance::

    uv run python scripts/mabbob_compare.py compare tune.json --json-out cmp.json

Everything is on the competition's metric
(:func:`panobbgo.mabbob_official.official_aocc`), so a run's stored AOCC is
converted first (the repo's differs by at most ``1 / budget``).

``build-reference`` turns the entries' logged data into the small per-instance
file that ``compare`` reads (:data:`DEFAULT_REFERENCE`).  It parses downloaded
files as data and runs nothing from them; start it with ``python -I``::

    uv run python -I scripts/mabbob_compare.py build-reference \\
        --entry NeighborhoodAdaptiveDE=/path/NeighborhoodAdaptiveDE --entry ERADS=/path/ERADS \\
        --entry PSS=/path/PSS --entry ACA=/path/ACA

Details and the validation against ``iohinspector``:
``doc/dev/benchmarking.md``, "Official MA-BBOB instances".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from panobbgo.harness_ioh import (
    BBOB_CLASS_ORDER,
    IOHHarnessResult,
    IOHMultiSeedResult,
    IOHRunRecord,
    bbob_class_of,
)
from panobbgo.ioh_runner import AOCC_LOG_HI, AOCC_LOG_LO
from panobbgo.mabbob_official import (
    COMPETITION_BUDGET_MULTIPLIER,
    COMPETITION_DIMS,
    CSV_SHA256,
    DATA_URL,
    N_ROWS,
    OFFICIAL_KIND,
    load_logged_entry,
    load_table,
    official_aocc_of_points,
    official_from_repo_aocc,
    split_rows,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The per-instance scores of the published entries.
DEFAULT_REFERENCE = REPO_ROOT / "benchmarks" / "references" / "mabbob_official_entries.json"

_BASELINES = (
    "Baseline shipped with the MA-BBOB 2025 getting-started archive (MABBOB_Baselines.zip in "
    "IOHprofiler/IOHdata, which states no licence); only derived per-instance scores are stored here."
)

#: Why ERADS at d = 2 differs from what iohinspector prints.
ERADS_NOTE = (
    "iohinspector's loader drops row 452 at d=2 (a whole-file raw_y.diff filter), so the organisers' tool gives "
    "0.836957 over 999 rows; this file keeps the row, scored 1.0, giving 0.837120."
)

#: Where each entry's logged data comes from.
ENTRY_SOURCES: Dict[str, Dict[str, str]] = {
    "NeighborhoodAdaptiveDE": {
        "source": "https://zenodo.org/records/15074959 (NeighborhoodAdaptiveDE-ioh.zip, "
        "md5 5ee1a3ac7e29e45f5957b255426f42dc)",
        "licence": "CC BY 4.0",
        "attribution": "'NeighborhoodAdaptiveDE for Many-Affine Black-box optimization', Anonymous (Zenodo, 2025), "
        "doi:10.5281/zenodo.15074959; designed with the LLaMEA framework (van Stein & Baeck, IEEE TEVC 2024). "
        "Winner of the 2025 competition per van Stein et al., arXiv:2601.21511 (\u00a75.4: tuned on MA-BBOB 5d), and "
        "GECCO 2025 lists van Stein first; the Zenodo record is dated before the deadline, so it may not be the exact "
        "submitted version. The per-instance scores here are computed from its logged data.",
    },
    "ERADS": {
        "source": DATA_URL,
        "licence": "none stated",
        "attribution": _BASELINES,
        "note": ERADS_NOTE,
    },
    "PSS": {
        "source": DATA_URL,
        "licence": "none stated",
        "attribution": _BASELINES + " Not the data of Almonacid's figshare record 26026381, which ran on "
        "another instance set and another configuration.",
    },
    "ACA": {"source": DATA_URL, "licence": "none stated", "attribution": _BASELINES},
}

#: Two values this close are a tie.
TIE_EPS: float = 1e-12

#: Rounding of the stored scores.
REFERENCE_DECIMALS: int = 6


# ---------------------------------------------------------------------------
# build-reference
# ---------------------------------------------------------------------------


def build_reference(entries: Dict[str, Path]) -> Dict[str, Any]:
    """The reference document: per-row official AOCC of every entry at every competition dimension."""
    out: Dict[str, Any] = {
        "description": "Per-instance official AOCC (index = row of the official instance files) of published "
        "MA-BBOB competition entries, computed from their logged data by scripts/mabbob_compare.py build-reference. "
        "Regenerate it, do not edit it.",
        "metric": "panobbgo.mabbob_official.official_aocc: iohinspector transform_fval(df, 1e-8, 1e2) + get_aocc",
        "bounds": [10.0**AOCC_LOG_LO, 10.0**AOCC_LOG_HI],
        "budget_multiplier": COMPETITION_BUDGET_MULTIPLIER,
        "rows": N_ROWS,
        "instances": {"source": DATA_URL, "sha256": CSV_SHA256},
        "entries": {},
    }
    for name, root in entries.items():
        entry: Dict[str, Any] = {**ENTRY_SOURCES.get(name, {}), "algorithm": None, "aocc": {}, "mean": {}}
        for dim in COMPETITION_DIMS:
            logged = load_logged_entry(root, dim)
            entry["algorithm"] = logged.name
            budget = COMPETITION_BUDGET_MULTIPLIER * dim
            scores = np.zeros(N_ROWS)
            for row, points in zip(logged.rows, logged.runs):
                scores[row] = official_aocc_of_points(points, budget)
            entry["aocc"][str(dim)] = [round(float(v), REFERENCE_DECIMALS) for v in scores]
            entry["mean"][str(dim)] = round(float(scores.mean()), REFERENCE_DECIMALS)
        out["entries"][name] = entry
    return out


def dump_reference(doc: Dict[str, Any]) -> str:
    """JSON with one line per score list, so the file stays readable and its diffs small."""
    text = json.dumps(doc, indent=1)
    return re.sub(r"\[\s+([-0-9.e,\s]+?)\s+\]", lambda m: "[" + re.sub(r"\s+", "", m.group(1)) + "]", text) + "\n"


def load_reference(path: Path = DEFAULT_REFERENCE) -> Dict[str, Any]:
    """The reference document; :class:`ValueError` if a score list does not cover every row."""
    doc = json.loads(Path(path).read_text())
    for name, entry in doc["entries"].items():
        for dim, values in entry["aocc"].items():
            if len(values) != N_ROWS:
                raise ValueError(f"{path}: {name} d={dim} has {len(values)} scores, expected {N_ROWS}")
    return doc


def cmd_build_reference(args: argparse.Namespace) -> int:
    entries = {}
    for spec in args.entry:
        name, sep, root = spec.partition("=")
        if not sep:
            raise SystemExit(f"error: --entry takes NAME=DIR, got {spec!r}")
        entries[name] = Path(root)
    doc = build_reference(entries)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_reference(doc))
    for name, entry in doc["entries"].items():
        print(f"{name:24s} {entry['algorithm']:32s} " + "  ".join(f"d={d} {m:.6f}" for d, m in entry["mean"].items()))
    print(f"wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# Our runs
# ---------------------------------------------------------------------------


def official_score_of_record(rec: IOHRunRecord) -> float:
    """A run's AOCC on the competition's metric, from its stored AOCC and the ends of its trace.

    The trace always holds evaluation 1 and the last one, which is all the
    conversion needs.  A crashed run with no trace keeps its stored 0.
    """
    if not rec.trace_fx:
        return float(rec.aocc)
    if rec.trace_evals[0] != 1:
        raise ValueError(
            f"row {rec.instance} d={rec.dim} {rec.strategy_name}: the trace does not start at evaluation 1"
        )
    return official_from_repo_aocc(
        rec.aocc, rec.trace_fx[0] - rec.f_opt, rec.trace_fx[-1] - rec.f_opt, rec.budget, AOCC_LOG_LO, AOCC_LOG_HI
    )


def _check_result(path: Any, res: IOHHarnessResult) -> None:
    """Refuse anything that is not a competition-rule run on the official rows."""
    if res.problem_kind != OFFICIAL_KIND:
        raise ValueError(f"{path}: problem kind {res.problem_kind!r}, expected {OFFICIAL_KIND!r}")
    if not res.sync_eval or res.virtual is not None:
        raise ValueError(f"{path}: not a synchronous evaluation-count run (sync_eval={res.sync_eval}, {res.virtual})")
    if (res.log_lo, res.log_hi) != (AOCC_LOG_LO, AOCC_LOG_HI):
        raise ValueError(f"{path}: AOCC bounds {res.log_lo}, {res.log_hi}, not the competition's")
    for rec in res.runs:
        if rec.budget != COMPETITION_BUDGET_MULTIPLIER * rec.dim:
            raise ValueError(f"{path}: row {rec.instance} d={rec.dim} has budget {rec.budget}, not 2000*d")


def _results_of(path: Path) -> List[Tuple[Optional[int], IOHHarnessResult]]:
    data = json.loads(Path(path).read_text())
    if data.get("multi_seed"):
        multi = IOHMultiSeedResult.from_dict(data)
        return list(zip(multi.base_seeds, multi.results))
    return [(None, IOHHarnessResult.from_dict(data))]  # a plain result does not record its base seed


class OurScores:
    """Scores per ``(strategy, dim, row)``, averaged over the base seeds the files hold."""

    def __init__(self) -> None:
        self.cells: Dict[Tuple[str, int, int], List[float]] = {}
        self.seen: set = set()
        self.problems: Dict[str, Dict[str, int]] = {}
        self.fp_env_ids: set = set()
        self.seeds: set = set()

    def add(self, path: Any, seed: Optional[int], res: IOHHarnessResult) -> None:
        """Add one result; the same run twice (same run seed, cell and rep) raises."""
        _check_result(path, res)
        self.fp_env_ids.add(res.fp_env_id)
        if seed is not None:
            self.seeds.add(seed)
        for rec in res.runs:
            key = (rec.seed, rec.strategy_name, rec.dim, rec.instance, rec.rep)
            if key in self.seen:
                raise ValueError(f"{path}: duplicate run {key}")
            self.seen.add(key)
            self.cells.setdefault((rec.strategy_name, rec.dim, rec.instance), []).append(official_score_of_record(rec))
            counts = self.problems.setdefault(rec.strategy_name, {"crashed": 0, "timed_out": 0, "ended_early": 0})
            counts["crashed"] += int(rec.crashed)
            counts["timed_out"] += int(rec.timed_out)
            counts["ended_early"] += int(rec.ended_early)

    @property
    def strategies(self) -> List[str]:
        """The strategies, in the order of first appearance."""
        return list(dict.fromkeys(s for s, _d, _r in self.cells))

    def vector(self, strategy: str, dim: int) -> np.ndarray:
        """The per-row score at ``dim``; ``NaN`` where the files have no run."""
        out = np.full(N_ROWS, np.nan)
        for row in range(N_ROWS):
            vals = self.cells.get((strategy, dim, row))
            if vals:
                out[row] = float(np.mean(vals))
        return out


def load_scores(paths: Sequence[Path]) -> OurScores:
    """Read the result files."""
    scores = OurScores()
    for path in paths:
        for seed, res in _results_of(path):
            scores.add(path, seed, res)
    if not scores.cells:
        raise ValueError(f"no runs in {[str(p) for p in paths]}")
    return scores


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(
    deltas: np.ndarray, *, n_boot: int = 10000, seed: int = 20260, level: float = 0.95
) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean over instances (deterministic for a given seed)."""
    n = deltas.size
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    # One batch at a time: 10000 resamples of 500 rows would otherwise hold 40 MB per cell.
    means = np.empty(n_boot)
    for start in range(0, n_boot, 1000):
        stop = min(start + 1000, n_boot)
        means[start:stop] = deltas[rng.integers(0, n, size=(stop - start, n))].mean(axis=1)
    tail = (1.0 - level) / 2.0 * 100.0
    return float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail))


def paired(ours: np.ndarray, theirs: np.ndarray, n_boot: int, seed: int) -> Dict[str, Any]:
    """Mean AOCC of both sides, mean paired delta with CI, wins / losses / ties, on the rows both have."""
    mask = np.isfinite(ours) & np.isfinite(theirs)
    a, b = ours[mask], theirs[mask]
    delta = a - b
    lo, hi = bootstrap_ci(delta, n_boot=n_boot, seed=seed)
    return {
        "n": int(mask.sum()),
        "ours": float(a.mean()) if a.size else float("nan"),
        "entry": float(b.mean()) if b.size else float("nan"),
        "delta": float(delta.mean()) if delta.size else float("nan"),
        "ci_lo": lo,
        "ci_hi": hi,
        "wins": int((delta > TIE_EPS).sum()),
        "losses": int((delta < -TIE_EPS).sum()),
        "ties": int((np.abs(delta) <= TIE_EPS).sum()),
    }


def _entry_vector(reference: Dict[str, Any], entry: str, dim: int) -> np.ndarray:
    return np.asarray(reference["entries"][entry]["aocc"][str(dim)], dtype=np.float64)


def _dim_views(
    scores: OurScores, reference: Dict[str, Any], strategy: str, entry: str
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """``(label, ours, entry)`` per dimension, then on the per-row mean of both (NaN unless both ran)."""
    views = []
    for dim in COMPETITION_DIMS:
        views.append((f"d={dim}", scores.vector(strategy, dim), _entry_vector(reference, entry, dim)))
    ours_mean = np.mean([v[1] for v in views], axis=0)  # NaN propagates: a row needs both dims
    entry_mean = np.mean([v[2] for v in views], axis=0)
    views.append(("mean", ours_mean, entry_mean))
    return views


def dominant_fids(weights: Sequence[Sequence[float]]) -> np.ndarray:
    """The BBOB function id (1..24) with the largest weight, per row."""
    return np.argmax(np.asarray(weights, dtype=np.float64), axis=1) + 1


def compare(
    scores: OurScores,
    reference: Dict[str, Any],
    weights: Sequence[Sequence[float]],
    *,
    breakdown_entry: Optional[str] = None,
    n_boot: int = 10000,
    seed: int = 20260,
) -> Dict[str, Any]:
    """Everything the report prints, as data: per split, strategy, entry, dimension view; and the breakdowns."""
    entries = list(reference["entries"])
    target = breakdown_entry or entries[0]
    if target not in reference["entries"]:
        raise ValueError(f"unknown entry {target!r}; known: {entries}")
    w = np.asarray(weights, dtype=np.float64)
    dominant = dominant_fids(w)
    tune, held = split_rows()
    report: Dict[str, Any] = {
        "breakdown_entry": target,
        "seeds": sorted(scores.seeds),
        "fp_env_ids": sorted(map(str, scores.fp_env_ids)),
        "problems": scores.problems,
        "splits": {},
    }
    for split, rows in (("tune", tune), ("heldout", held)):
        mask = np.zeros(N_ROWS, dtype=bool)
        mask[list(rows)] = True
        cover = {
            s: {d: int(np.isfinite(scores.vector(s, d)[mask]).sum()) for d in COMPETITION_DIMS}
            for s in scores.strategies
        }
        if not any(v for by_dim in cover.values() for v in by_dim.values()):
            continue
        section: Dict[str, Any] = {"rows": len(rows), "coverage": cover, "strategies": {}, "breakdown": {}}
        for strategy in scores.strategies:
            per_entry: Dict[str, Any] = {}
            for entry in entries:
                per_entry[entry] = {}
                for label, ours, theirs in _dim_views(scores, reference, strategy, entry):
                    per_entry[entry][label] = paired(
                        np.where(mask, ours, np.nan), np.where(mask, theirs, np.nan), n_boot, seed
                    )
            section["strategies"][strategy] = per_entry
            section["breakdown"][strategy] = _breakdown(scores, reference, strategy, target, mask, dominant, w)
        report["splits"][split] = section
    return report


def _breakdown(
    scores: OurScores,
    reference: Dict[str, Any],
    strategy: str,
    entry: str,
    mask: np.ndarray,
    dominant: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, Any]:
    """Mean delta per dominant function and its COCO class, and the delta attributed by weight to each function."""
    views = {lab: ours - theirs for lab, ours, theirs in _dim_views(scores, reference, strategy, entry)}

    def row(sel: np.ndarray) -> Dict[str, Any]:
        # Rows with a run at any dimension, so a partial file shows what it covers.
        covered = np.any([np.isfinite(d) for d in views.values()], axis=0)
        out: Dict[str, Any] = {"n": int((sel & covered).sum())}
        for lab, delta in views.items():
            vals = delta[sel & np.isfinite(delta)]
            out[lab] = float(vals.mean()) if vals.size else None
        return out

    by_fid: Dict[str, Any] = {}
    for fid in range(1, 25):
        cell = row(mask & (dominant == fid))
        # All rows, by the share of the mixture the function has.
        delta = views["mean"]
        ok = mask & np.isfinite(delta)
        total = float(weights[ok, fid - 1].sum())
        cell["weighted"] = float((weights[ok, fid - 1] * delta[ok]).sum() / total) if total > 0 else None
        by_fid[str(fid)] = cell
    classes = np.array([bbob_class_of(int(f)) for f in dominant])
    by_class = {tag: row(mask & (classes == tag)) for tag in BBOB_CLASS_ORDER}
    return {"by_dominant_fid": by_fid, "by_dominant_class": by_class}


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float], plus: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "     -"
    return f"{value:+.4f}" if plus else f"{value:.4f}"


def print_report(report: Dict[str, Any], out: Any = None) -> None:
    """The tables, in plain text."""
    out = sys.stdout if out is None else out

    def say(text: str = "") -> None:
        print(text, file=out)

    say(f"Seeds: {report['seeds']}   fp_env_id: {report['fp_env_ids']}")
    if len(report["fp_env_ids"]) > 1 or any(str(i).startswith("mixed:") for i in report["fp_env_ids"]):
        say("WARNING: the runs come from several floating-point environments")
    say(
        "Metric: the competition's AOCC (d 2 and 5, 2000*d, bounds 1e-8..1e2).  A CI is a percentile bootstrap over rows."
    )
    say("Public rows: not a competition result, which is judged on new instances.")
    say(f"ERADS: {ERADS_NOTE}")
    for strategy, counts in report["problems"].items():
        if any(counts.values()):
            say(f"WARNING: {strategy}: {counts}")
    for split, section in report["splits"].items():
        say(f"\n=== split: {split} ({section['rows']} rows) ===")
        if split == "heldout":
            say("Conservative for us: the entries may have been tuned on these rows, and we did not develop on them.")
        else:
            say("Not conservative: the entries may have been tuned on these rows, and so were we.")
        for strategy, by_dim in section["coverage"].items():
            short = [d for d, n in by_dim.items() if n != section["rows"]]
            if short:
                say(f"PARTIAL: {strategy} covers {by_dim} of {section['rows']} rows; not a result")
        for strategy, per_entry in section["strategies"].items():
            say(f"\n{strategy}")
            say(f"  {'entry':24s} {'dim':5s} {'n':>4s} {'ours':>7s} {'entry':>7s} {'delta':>8s}  {'95% CI':19s} W/L/T")
            for entry, views in per_entry.items():
                for label, c in views.items():
                    say(
                        f"  {entry:24s} {label:5s} {c['n']:4d} {_fmt(c['ours']):>7s} {_fmt(c['entry']):>7s} "
                        f"{_fmt(c['delta'], True):>8s}  [{_fmt(c['ci_lo'], True)}, {_fmt(c['ci_hi'], True)}] "
                        f"{c['wins']}/{c['losses']}/{c['ties']}"
                    )
        say(f"\nWhere the rows are lost, delta against {report['breakdown_entry']}:")
        for strategy, tables in section["breakdown"].items():
            say(f"\n{strategy}, by the COCO class of the dominant function")
            say(f"  {'class':20s} {'n':>4s} {'d=2':>8s} {'d=5':>8s} {'mean':>8s}")
            for tag, cell in tables["by_dominant_class"].items():
                say(
                    f"  {tag:20s} {cell['n']:4d} {_fmt(cell['d=2'], True):>8s} {_fmt(cell['d=5'], True):>8s} "
                    f"{_fmt(cell['mean'], True):>8s}"
                )
            say(f"{strategy}, by dominant function (weighted: all rows, by the function's share of the mixture)")
            say(f"  {'fid':>4s} {'n':>4s} {'d=2':>8s} {'d=5':>8s} {'mean':>8s} {'weighted':>9s}")
            for fid, cell in tables["by_dominant_fid"].items():
                say(
                    f"  {'f' + fid:>4s} {cell['n']:4d} {_fmt(cell['d=2'], True):>8s} {_fmt(cell['d=5'], True):>8s} "
                    f"{_fmt(cell['mean'], True):>8s} {_fmt(cell['weighted'], True):>9s}"
                )


def cmd_compare(args: argparse.Namespace) -> int:
    scores = load_scores([Path(p) for p in args.results])
    report = compare(
        scores,
        load_reference(Path(args.reference)),
        load_table().weights,
        breakdown_entry=args.breakdown_entry,
        n_boot=args.bootstrap,
        seed=args.bootstrap_seed,
    )
    print_report(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1))
        print(f"\nSaved: {args.json_out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="Compare result files with the published entries.")
    c.add_argument("results", nargs="+", help="Result files of ioh_benchmark.py run --mabbob-official(-heldout).")
    c.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="The entries' per-instance scores.")
    c.add_argument("--breakdown-entry", help="The entry the breakdown tables compare with (default: the first).")
    c.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap resamples per CI.")
    c.add_argument("--bootstrap-seed", type=int, default=20260)
    c.add_argument("--json-out", help="Also save the whole report as JSON.")
    c.set_defaults(fn=cmd_compare)
    b = sub.add_parser("build-reference", help="Compute the entries' per-instance scores from their logged data.")
    b.add_argument("--entry", action="append", required=True, metavar="NAME=DIR", help="An entry's IOH data folder.")
    b.add_argument("--out", default=str(DEFAULT_REFERENCE))
    b.set_defaults(fn=cmd_build_reference)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
