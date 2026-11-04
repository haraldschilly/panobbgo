# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
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

r"""
The official MA-BBOB competition instances
==========================================

The GECCO MA-BBOB Anytime competition (2024, 2025; rules:
https://iohprofiler.github.io/competitions/mabbob25) publishes 1000 instances
as three CSV files, one row per instance.  A row builds a problem as::

    ioh.problem.ManyAffine(xopt=opt_locs[row][:dim], weights=weights[row], instances=iids[row], n_variables=dim)

These rows are **not** ``ManyAffine(instance=N)``: no id up to 1100 reproduces
any of them, so ``--full`` and the other ``"MA-BBOB"`` batteries are not the
official set.  This module holds what the head-to-head on the official rows
needs and nothing else: the data, the split, and the official way to score.

Data
----

The repository IOHprofiler/IOHdata states no licence, so the CSV files are not
vendored.  :func:`ensure_data` fetches them from a URL pinned to a commit into
a cache outside the repository and verifies each file's SHA-256
(:data:`CSV_SHA256`); ``PANOBBGO_MABBOB_DATA`` overrides the directory.  The
archive is parsed as data: only the three members are read, nothing is run.

The tune / held-out split
-------------------------

The 1000 rows split 500 / 500 by the salted SHA-256 of the row index
(:func:`split_rows`).  **Develop on tune only.**  Held-out runs once per frozen
candidate, to report that candidate; a number from it that steers a decision
burns the split the way a sealed number burns the sealed set
(``doc/dev/benchmarking.md``, "Official MA-BBOB instances").

The published entries may have been tuned on these same 1000 public rows, so
their score may be optimistic against unseen instances.  That makes a
**held-out** comparison conservative for us.  A **tune** comparison is not: we
develop on those rows too, so it can flatter us as well.  Neither is a
competition result, which is judged on new instances.

Scoring
-------

:func:`official_aocc` is the competition's convention (``iohinspector``,
``transform_fval(df, 1e-8, 1e2)`` then ``get_aocc``).  It integrates the
best-so-far as a step function over ``[0, B]`` and holds the first value over
``[0, 1)``, so it counts the first evaluation twice and the last one not at
all.  :func:`panobbgo.ioh_runner.aocc` takes the plain mean over the ``B``
evaluations; it is a frozen contract and stays.  The two differ by
``(q(1) - q(B)) / B`` where ``q`` is the normalised score
(:func:`official_from_repo_aocc`), never more than ``1 / B``.
"""

from __future__ import annotations

import csv
import hashlib
import http.client
import io
import json
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

import numpy as np

from panobbgo.ioh_runner import AOCC_LOG_HI, AOCC_LOG_LO, aocc

#: The worker kind: ``instance`` is the row index into the CSV files.
OFFICIAL_KIND: str = "MA-BBOB-official"

#: Rows in the public instance files.
N_ROWS: int = 1000

#: The competition's dimensions and budget (``budget = 2000 * d``).
COMPETITION_DIMS: Tuple[int, ...] = (2, 5)
COMPETITION_BUDGET_MULTIPLIER: int = 2000

#: The problem id the competition's notebook sets (``f_new.set_id(100)``).
OFFICIAL_PROBLEM_ID: int = 100

#: Fixed once, before any result on the official rows existed.  Changing it is
#: a new split, and every number on the old held-out half is then development data.
SPLIT_SALT: str = "panobbgo-mabbob-official-split-2026-11-04"

#: The archive, pinned to the commit that added it (2025-01-28).
DATA_URL: str = (
    "https://github.com/IOHprofiler/IOHdata/raw/a3ecaac6c06aa97190249c0c2c74b6a3aac14304/MABBOB_GettingStarted.zip"
)

#: SHA-256 of each instance file in the archive.
CSV_SHA256: Dict[str, str] = {
    "iids.csv": "90c6f0012a8301cfe4884e588e9a49309097d785f3ebe90b4734a2819218184e",
    "opt_locs.csv": "5f26909949b49ec69e40a19eab43be40f19e9047583f48fc25d5e7032f9927e4",
    "weights.csv": "3f27b6abe8e66cec6a889576c4db90cfff0ff5f790961a2ac4c24226e6e0b0b9",
}

#: The environment variable naming the data directory.
DATA_DIR_ENV: str = "PANOBBGO_MABBOB_DATA"

_MAX_ARCHIVE_BYTES: int = 64 * 1024 * 1024

HELDOUT_BANNER: str = """\
################################################################################
#  HELD-OUT SPLIT OF THE OFFICIAL MA-BBOB INSTANCES: {name}
#  Run this once per frozen candidate, to report it.  Never tune, screen, select
#  or train on it.  A number from it that steers a decision burns the split
#  (doc/dev/benchmarking.md, "Official MA-BBOB instances").
################################################################################"""


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def _split_key(row: int) -> bytes:
    return hashlib.sha256(f"{SPLIT_SALT}|{int(row)}".encode()).digest()


@lru_cache(maxsize=1)
def split_rows() -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """``(tune, heldout)``: the rows, 500 each, ascending, disjoint, together ``0..999``.

    The rows are ranked by the salted SHA-256 of the index; the first half is
    tune.  Ranking, rather than thresholding one hash byte, makes the sizes
    exact.
    """
    ranked = sorted(range(N_ROWS), key=_split_key)
    half = N_ROWS // 2
    return tuple(sorted(ranked[:half])), tuple(sorted(ranked[half:]))


def split_of(row: int) -> str:
    """``"tune"`` or ``"heldout"``; :class:`ValueError` for a row outside ``0..999``."""
    row = int(row)
    if not 0 <= row < N_ROWS:
        raise ValueError(f"not an official MA-BBOB row (expected 0..{N_ROWS - 1}): {row}")
    return "tune" if row in set(split_rows()[0]) else "heldout"


def check_official_rows(rows: Iterable[int], *, heldout: bool) -> None:
    """Raise :class:`ValueError` unless ``rows`` all lie in the tune split, or all in held-out with ``heldout=True``.

    The guard behind every entry point that runs official rows: a tune battery
    cannot reach a held-out row by a typo or a stray id, and a held-out battery
    states what it is.  As with the sealed set it stops accidents; it is not a
    lock against a caller who passes ``heldout=True`` deliberately.
    """
    rows = [int(r) for r in rows]
    tune, held = split_rows()
    home, name = (set(held), "held-out") if heldout else (set(tune), "tune")
    out_of_range = [r for r in rows if not 0 <= r < N_ROWS]
    if out_of_range:
        raise ValueError(f"not official MA-BBOB rows (expected 0..{N_ROWS - 1}): {out_of_range[:10]}")
    wrong = [r for r in rows if r not in home]
    if wrong:
        hint = (
            "the held-out split takes an explicit flag (--mabbob-official-heldout, heldout=True) and runs "
            "once per frozen candidate"
            if not heldout
            else "a held-out battery takes held-out rows only"
        )
        raise ValueError(f"rows {wrong[:10]} are not in the {name} split of the official MA-BBOB instances: {hint}")


def chunk_rows(rows: Sequence[int], index: int, count: int) -> Tuple[int, ...]:
    """The rows of chunk ``index`` of ``count`` (1-based), by stride: every chunk is a fair sample of ``rows``."""
    if not 1 <= int(index) <= int(count):
        raise ValueError(f"chunk must be 1..{count}, got {index}")
    if int(index) > len(rows):
        raise ValueError(f"chunk {index} of {count} is empty: there are only {len(rows)} rows")
    return tuple(rows[int(index) - 1 :: int(count)])


def print_heldout_banner(name: str, stream: Optional[TextIO] = None) -> None:
    """Print the held-out warning banner for battery ``name`` (to stderr by default)."""
    print(HELDOUT_BANNER.format(name=name), file=sys.stderr if stream is None else stream, flush=True)


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OfficialTable:
    """The parsed CSV files: ``xopt[row]`` (20 coordinates), ``weights[row]`` and ``iids[row]`` (24 each)."""

    xopt: Tuple[Tuple[float, ...], ...]
    weights: Tuple[Tuple[float, ...], ...]
    iids: Tuple[Tuple[int, ...], ...]


def data_dir() -> Path:
    """Where the CSV files live: :data:`DATA_DIR_ENV`, else the user cache directory."""
    explicit = os.environ.get(DATA_DIR_ENV)
    if explicit:
        return Path(explicit)
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "panobbgo" / "mabbob_official"


def _sha256_ok(path: Path, expected: str) -> bool:
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected


def _write_atomic(path: Path, data: bytes) -> None:
    """Concurrent shard processes may all fetch: each writes a temp file and renames it into place."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o644)  # mkstemp's 0600 would hide a shared data directory from everyone else
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _download(url: str) -> bytes:
    """The archive's bytes, one attempt."""
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (a pinned https URL)
        data = resp.read(_MAX_ARCHIVE_BYTES + 1)
    if len(data) > _MAX_ARCHIVE_BYTES:
        raise ValueError(f"{url}: larger than {_MAX_ARCHIVE_BYTES} bytes")
    return data


def _fetch_members(url: str, names: Sequence[str], attempts: int = 4) -> Dict[str, bytes]:
    """The named files of the archive, with retries (a shard must not die on one failed request).

    A cut connection, an error page instead of the zip and a zip without the
    files all count as a failed request.  Only the final failure raises, as
    :class:`RuntimeError`.
    """
    last: Optional[Exception] = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(2.0**attempt)
        try:
            with zipfile.ZipFile(io.BytesIO(_download(url))) as archive:
                return {n: archive.read(n) for n in names}
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            OSError,
            zipfile.BadZipFile,
            zlib.error,
            KeyError,
        ) as exc:
            last = exc
    raise RuntimeError(
        f"could not download {url}: {last}; offline, point {DATA_DIR_ENV} at a directory with the three CSV files"
    ) from last


def ensure_data(directory: Optional[Path] = None, url: str = DATA_URL) -> Path:
    """The directory holding the three verified CSV files, fetching them if needed.

    A file already there with the wrong hash is refused when the directory was
    named explicitly (:data:`DATA_DIR_ENV`), since it may be someone's work, and
    replaced in the cache.  A download whose content does not hash to
    :data:`CSV_SHA256` raises and writes nothing.
    """
    explicit = directory is not None or bool(os.environ.get(DATA_DIR_ENV))
    target = Path(directory) if directory is not None else data_dir()
    missing = [n for n, h in CSV_SHA256.items() if not _sha256_ok(target / n, h)]
    if not missing:
        return target
    if explicit:
        wrong = [n for n in missing if (target / n).exists()]
        if wrong:
            raise ValueError(f"{target}: {wrong} do not match the pinned SHA-256 (panobbgo.mabbob_official.CSV_SHA256)")
    target.mkdir(parents=True, exist_ok=True)
    contents = _fetch_members(url, missing)
    for name, data in contents.items():
        digest = hashlib.sha256(data).hexdigest()
        if digest != CSV_SHA256[name]:
            raise ValueError(f"{url}: {name} has SHA-256 {digest}, expected {CSV_SHA256[name]}")
    for name, data in contents.items():
        _write_atomic(target / name, data)
    return target


def _read_rows(path: Path, width: int) -> List[List[str]]:
    """The cells of one CSV file, checked for shape: a header, then ``row, v0, ..., v<width-1>``."""
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) != N_ROWS + 1:
        raise ValueError(f"{path}: expected {N_ROWS} rows, got {len(rows) - 1}")
    for i, row in enumerate(rows[1:]):
        if int(row[0]) != i or len(row) != width + 1:
            raise ValueError(f"{path}: malformed row {i}")
    return [row[1:] for row in rows[1:]]


@lru_cache(maxsize=1)
def _load_table(directory: str) -> OfficialTable:
    d = Path(directory)
    return OfficialTable(
        xopt=tuple(tuple(float(v) for v in r) for r in _read_rows(d / "opt_locs.csv", 20)),
        weights=tuple(tuple(float(v) for v in r) for r in _read_rows(d / "weights.csv", 24)),
        iids=tuple(tuple(int(v) for v in r) for r in _read_rows(d / "iids.csv", 24)),
    )


def load_table() -> OfficialTable:
    """The official instance table (fetched and verified on first use, then cached in this process)."""
    return _load_table(str(ensure_data()))


def row_payload(row: int, dim: int) -> Dict[str, Any]:
    """The arguments of the worker's ``create`` for ``row`` at ``dim``.

    The parent resolves the row, so the worker needs neither the files nor a
    copy of the verification and stays what it is: a thin wrapper around
    ``ioh``.  ``float`` parses a cell exactly; ``pandas.read_csv``, which the
    competition's notebook uses, can land one ULP away.  That is far below the
    ``1e-8`` the metric resolves.
    """
    row, dim = int(row), int(dim)
    if not 0 <= row < N_ROWS:
        raise ValueError(f"not an official MA-BBOB row (expected 0..{N_ROWS - 1}): {row}")
    if not 1 <= dim <= 20:
        raise ValueError(f"the official instances define dimensions 1..20, got {dim}")
    table = load_table()
    return {"xopt": list(table.xopt[row][:dim]), "weights": list(table.weights[row]), "iids": list(table.iids[row])}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _quality(values: Any, log_lo: float, log_hi: float) -> Any:
    """Normalised score ``q`` of a value: 1 at or below ``10**log_lo``, 0 at or above ``10**log_hi``."""
    arr = np.maximum(np.asarray(values, dtype=np.float64), 10.0**log_lo)
    return 1.0 - (np.clip(np.log10(arr), log_lo, log_hi) - log_lo) / (log_hi - log_lo)


def official_from_repo_aocc(
    repo_aocc: float,
    first: float,
    last: float,
    budget: int,
    log_lo: float = AOCC_LOG_LO,
    log_hi: float = AOCC_LOG_HI,
) -> float:
    """Convert :func:`panobbgo.ioh_runner.aocc` to :func:`official_aocc`, given the first and last best-so-far gaps.

    ``first`` and ``last`` are ``best_so_far - f_opt`` after the first and the
    last of the ``budget`` evaluations.  A run record keeps both (its
    downsampled trace includes both ends), so a stored AOCC converts exactly.
    """
    return float(repo_aocc) + float(_quality(first, log_lo, log_hi) - _quality(last, log_lo, log_hi)) / int(budget)


def official_aocc(
    gaps: Sequence[float], budget: int, log_lo: float = AOCC_LOG_LO, log_hi: float = AOCC_LOG_HI
) -> float:
    """The competition's AOCC of a best-so-far gap trace, one entry per evaluation.

    ``gaps[t]`` is ``best_so_far - f_opt`` after evaluation ``t + 1``.  Entries
    past ``budget`` are not scored; a short trace is held at its last value.
    """
    arr = np.asarray(gaps, dtype=np.float64)[: int(budget)]
    if arr.size == 0:
        return 0.0
    repo = aocc(arr.tolist(), f_opt=0.0, log_lo=log_lo, log_hi=log_hi, budget=int(budget))
    return official_from_repo_aocc(repo, float(arr[0]), float(arr[-1]), int(budget), log_lo, log_hi)


def expand_improvements(points: Sequence[Tuple[int, float]], budget: int) -> np.ndarray:
    """Best-so-far per evaluation ``1..budget`` from logged ``(evaluation, value)`` improvements.

    Logged data has one line per improvement.  Lines past ``budget`` are
    dropped (some entries overrun it: NeighborhoodAdaptiveDE logs up to 4050 at
    d = 2) and the last value before the budget is held to the end (others stop
    early: PSS at 369).  Evaluations before the first line are ``+inf``.
    """
    budget = int(budget)
    out = np.full(budget, np.inf, dtype=np.float64)
    best, start = np.inf, 0
    for evaluation, value in points:
        evaluation = int(evaluation)
        if evaluation > budget:
            break
        if evaluation > start + 1:
            out[start : evaluation - 1] = best
            start = evaluation - 1
        best = min(best, float(value))
    out[start:] = best
    return out


def official_aocc_of_points(
    points: Sequence[Tuple[int, float]], budget: int, log_lo: float = AOCC_LOG_LO, log_hi: float = AOCC_LOG_HI
) -> float:
    """:func:`official_aocc` of a run logged as ``(evaluation, value)`` improvements."""
    return official_aocc(expand_improvements(points, budget).tolist(), budget, log_lo, log_hi)


def repo_aocc_of_points(
    points: Sequence[Tuple[int, float]], budget: int, log_lo: float = AOCC_LOG_LO, log_hi: float = AOCC_LOG_HI
) -> float:
    """:func:`panobbgo.ioh_runner.aocc` of a run logged as ``(evaluation, value)`` improvements."""
    return aocc(
        expand_improvements(points, budget).tolist(), f_opt=0.0, log_lo=log_lo, log_hi=log_hi, budget=int(budget)
    )


# ---------------------------------------------------------------------------
# Logged IOH data (parsed as data; nothing in a downloaded archive is run)
# ---------------------------------------------------------------------------


def parse_ioh_dat(text: Iterable[str]) -> List[List[Tuple[int, float]]]:
    """The runs of an ``IOHprofiler_f*_DIM*.dat`` file, as lists of ``(evaluation, raw_y)``.

    A line starting with ``evaluations`` opens a run.  Only the first two
    columns are read, so the ``x0 x1 ...`` columns some loggers add are ignored.
    """
    runs: List[List[Tuple[int, float]]] = []
    for line in text:
        if line.startswith("evaluations"):
            runs.append([])
        elif line.strip():
            if not runs:
                raise ValueError("data line before the first header")
            evaluation, value = line.split()[:2]
            value_f = float(value)
            if math.isnan(value_f):
                raise ValueError(f"not a number in a logged value: {line!r}")
            runs[-1].append((int(evaluation), value_f))
    return runs


@dataclass(frozen=True)
class LoggedEntry:
    """One entry's runs of one dimension: ``runs[i]`` belongs to row ``rows[i]``."""

    name: str
    dim: int
    rows: Tuple[int, ...]
    runs: Tuple[Tuple[Tuple[int, float], ...], ...]
    #: The best point the logger recorded per run, from the ``.json`` metadata.
    best_y: Tuple[float, ...]
    best_x: Tuple[Tuple[float, ...], ...]
    best_evals: Tuple[int, ...]
    evals: Tuple[int, ...]


def load_logged_entry(root: Path, dim: int) -> LoggedEntry:
    """Read an entry's ``IOHprofiler_f100_ManyAffine.json`` and its ``.dat`` for ``dim``.

    Raises :class:`ValueError` unless there is one run per official row and the
    runs and the metadata agree in number.
    """
    root = Path(root)
    meta = json.loads((root / "IOHprofiler_f100_ManyAffine.json").read_text())
    scenario = next((s for s in meta["scenarios"] if int(s["dimension"]) == int(dim)), None)
    if scenario is None:
        raise ValueError(f"{root}: no scenario for d = {dim}")
    with (root / str(scenario["path"]).replace("\\", "/")).open() as f:
        runs = parse_ioh_dat(f)
    infos = scenario["runs"]
    if len(runs) != len(infos):
        raise ValueError(f"{root} d={dim}: {len(runs)} runs in the data, {len(infos)} in the metadata")
    rows = tuple(int(r["instance"]) for r in infos)
    if sorted(rows) != list(range(N_ROWS)):
        raise ValueError(f"{root} d={dim}: not one run per official row")
    return LoggedEntry(
        name=str(meta["algorithm"]["name"]),
        dim=int(dim),
        rows=rows,
        runs=tuple(tuple(r) for r in runs),
        best_y=tuple(float(r["best"]["y"]) for r in infos),
        best_x=tuple(tuple(float(v) for v in r["best"]["x"]) for r in infos),
        best_evals=tuple(int(r["best"]["evals"]) for r in infos),
        evals=tuple(int(r["evals"]) for r in infos),
    )
