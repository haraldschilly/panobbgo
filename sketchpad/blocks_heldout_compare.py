# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Pair worker-utilisation probe output with a runner's measure units (DISCOVERY §72).

The probe (``worker_utilisation_probe.py``) runs one panobbgo spec on the
``measure.py`` core path; its runs are bit-identical to the runner's for the
same code, seed and instance (§72 checked 675 + 225 of them).  So a local
candidate can be paired, per (seed, instance) under CRN, with the units of a
``measure.yml`` run: the old headline spec and the external pool.  For each
cell (dim, q) this prints the candidate minus the old spec, minus
``RegimeGate_oracle`` and minus the pool best (the best external by the cell
mean, fixed over the cell, as in ``measure.py``), with Holm over the cells.

Usage (download the run's units first)::

    gh run download 36315576900 -R haraldschilly/panobbgo --pattern 'measure-*' --dir raw12
    uv run python sketchpad/blocks_heldout_compare.py raw12 candidate.json [--by-family]

Deltas are paired over seeds (the per-seed mean over the cell's instances),
with a t-CI95 and wins/n, as in ``measure.py``'s tables.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

HEADLINE = "Blocks_warm_CMAES_JSO"
EXTERNALS = (
    "Baseline_BoTorch_qLogEI",
    "Baseline_NGOpt",
    "Baseline_Optuna_CmaEs",
    "Baseline_Optuna_TPE",
    "Baseline_PyBOBYQA",
    "Baseline_TuRBO1",
    "Baseline_pycma_BIPOP",
    "Baseline_pycma_IPOP",
)
FAMILIES = ("ackley", "ellipsoid", "rastrigin", "rosenbrock", "sharp_ridge")

#: (dim, q, seed, family, instance)
Key = Tuple[int, int, int, str, int]
#: key -> (AOCC, aocc_time)
Vals = Dict[Key, Tuple[float, float]]


def load_units(root: str, bm: int) -> Dict[str, Vals]:
    """Strategy -> values, from every unit file under ``root`` with budget multiplier ``bm``."""
    out: Dict[str, Vals] = defaultdict(dict)
    for path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        if os.path.basename(path).startswith("meta"):
            continue
        with open(path) as f:
            unit = json.load(f)
        if "result" not in unit or unit.get("bm") != bm:
            continue
        for r in unit["result"]["runs"]:
            key = (r["dim"], unit["q"], unit["seed"], r["problem_kind"], r["instance"])
            out[r["strategy_name"]][key] = (r["aocc"] or 0.0, r["aocc_time"] or 0.0)
    return out


def load_probe(path: str) -> Vals:
    with open(path) as f:
        return {(r["dim"], r["q"], r["seed"], r["fam"], r["inst"]): (r["aocc"], r["aocc_time"]) for r in json.load(f)}


def paired(a: Vals, b: Vals, keys: List[Key], metric: int) -> Tuple[float, float, float, int, int, float]:
    """Mean, CI95 low/high, wins, n, p of the per-seed paired delta ``a - b``."""
    by_a: Dict[int, List[float]] = defaultdict(list)
    by_b: Dict[int, List[float]] = defaultdict(list)
    for k in keys:
        by_a[k[2]].append(a[k][metric])
        by_b[k[2]].append(b[k][metric])
    d = np.array([np.mean(by_a[s]) - np.mean(by_b[s]) for s in sorted(by_a)])
    n, m = len(d), float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd == 0.0:
        return m, m, m, int((d > 0).sum()), n, 1.0 if m == 0.0 else 0.0
    se = sd / np.sqrt(n)
    t = float(stats.t.ppf(0.975, n - 1))
    return m, m - t * se, m + t * se, int((d > 0).sum()), n, float(2 * stats.t.sf(abs(m / se), n - 1))


def fmt(c: Tuple[float, float, float, int, int, float]) -> str:
    m, lo, hi, w, n, p = c
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] {w}/{n} p={p:.3f}"


def holm(ps: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    out, running = {}, 0.0
    for i, k in enumerate(sorted(ps, key=ps.__getitem__)):
        running = max(running, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = running
    return out


def pool_best(units: Dict[str, Vals], keys: List[Key]) -> Optional[str]:
    """The external with the best mean ``aocc_time`` over ``keys`` (only those with every key)."""
    best, best_mean = None, -1.0
    for name in EXTERNALS:
        vals = units.get(name, {})
        if not all(k in vals for k in keys):
            continue
        m = float(np.mean([vals[k][1] for k in keys]))
        if m > best_mean:
            best, best_mean = name, m
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("units", help="directory with the downloaded measure units")
    ap.add_argument("probe", help="probe output (JSON) of the candidate")
    ap.add_argument("--bm", type=int, default=100)
    ap.add_argument("--by-family", action="store_true")
    a = ap.parse_args()
    units = load_units(a.units, a.bm)
    cand = load_probe(a.probe)
    old, rg = units[HEADLINE], units.get("RegimeGate_oracle", {})
    ps_new: Dict[Tuple[int, int], float] = {}
    ps_old: Dict[Tuple[int, int], float] = {}
    for cell in sorted({k[:2] for k in cand}):
        cell_keys = [k for k in cand if k[:2] == cell and k in old]
        pb = pool_best(units, cell_keys)
        for fam in FAMILIES if a.by_family else (None,):
            keys = [k for k in cell_keys if fam is None or k[3] == fam]
            print(f"d{cell[0]} q{cell[1]} {fam or 'all'} (n={len(keys)}, pool best {pb})")
            print(f"  new - old aocc_time   {fmt(paired(cand, old, keys, 1))}")
            print(f"  new - old AOCC        {fmt(paired(cand, old, keys, 0))}")
            if all(k in rg for k in keys):
                print(f"  new - RegimeGate      {fmt(paired(cand, rg, keys, 1))}")
            if pb is not None:
                c_old, c_new = paired(old, units[pb], keys, 1), paired(cand, units[pb], keys, 1)
                print(f"  old - pool best       {fmt(c_old)}")
                print(f"  new - pool best       {fmt(c_new)}")
                if fam is None:
                    ps_old[cell], ps_new[cell] = c_old[5], c_new[5]
    if ps_new:
        h_new, h_old = holm(ps_new), holm(ps_old)
        print("Holm over the cells (vs the pool best): new / old")
        for cell in sorted(ps_new):
            print(f"  d{cell[0]} q{cell[1]}: {h_new[cell]:.3f} / {h_old[cell]:.3f}")


if __name__ == "__main__":
    main()
