# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
"""The pipeline the multi-seed screens share.

``arm_sweep``, ``np_accept``, ``oracle``, ``family_screen``, ``meta_screen``
and ``portfolio_screen`` all do the same five things around their own
question: parse ``key=value`` arguments, run a harness once per base seed
and rewrite the rows file after each, fold the rows into cells (reps into
their mean), pair two specs per seed, and print means, delta tables and a
run-health block.  Those pieces live here; each screen keeps its specs, its
battery and its own verdicts.

A **cell** is ``(seed, group, dim, inst)`` — ``group`` is the BBOB ``fid``
(``None`` without a function axis) or the problem family — mapped to
``{spec: mean AOCC over reps}``.  Every helper that selects cells takes a
``where(key)`` predicate; :func:`match` builds the usual ones.

The screens are run as scripts (``uv run python benchmarks/<screen>.py``),
so they import this module as ``_screen``.
"""

from __future__ import annotations

import dataclasses
import json
import statistics as st
import sys
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from panobbgo.harness_ioh import make_ioh_strategies, t_ci

Cell = Tuple[Any, Any, Any, Any]
Where = Optional[Callable[[Cell], bool]]

#: Row fields of a solo-arm IOH screen: ``(row key, run attribute)``.
IOH_FIELDS = (
    ("s", "strategy_name"),
    ("fid", "fid"),
    ("dim", "dim"),
    ("inst", "instance"),
    ("rep", "rep"),
    ("aocc", "aocc"),
    ("err", "error"),
)
#: :data:`IOH_FIELDS` plus the evaluation count and budget (run health).
IOH_FIELDS_BUDGET = IOH_FIELDS[:-1] + (("evals", "n_evals"), ("budget", "budget"), ("err", "error"))


# ── argv ──


def parse_argv(argv: Sequence[str], value_flags: Iterable[str] = ()) -> Tuple[Dict[str, str], set, List[str]]:
    """Split ``argv`` into ``key=value`` options, ``--flag``s and positionals.

    A flag named in *value_flags* (e.g. ``"--from"``) takes the next argument
    as its value and lands in the options without its dashes.
    """
    value_flags = set(value_flags)
    opts: Dict[str, str] = {}
    flags: set = set()
    pos: List[str] = []
    args = iter(argv)
    for a in args:
        if a in value_flags:
            opts[a[2:]] = next(args)
        elif a.startswith("--"):
            flags.add(a[2:])
        elif "=" in a:
            k, _, v = a.partition("=")
            opts[k] = v
        else:
            pos.append(a)
    return opts, flags, pos


def int_tuple(text: str) -> Tuple[int, ...]:
    """``"2,5"`` -> ``(2, 5)``."""
    return tuple(int(x) for x in text.split(","))


def csv_of(values: Iterable[Any]) -> str:
    """``(2, 5)`` -> ``"2,5"``: a battery field as the default of its option."""
    return ",".join(str(v) for v in values)


def select_names(opts: Dict[str, str], specs: Dict[str, Any]) -> List[str]:
    """The ``specs=a,b`` selection (default: every spec); exits on an unknown name."""
    names = [n for n in opts["specs"].split(",") if n] if opts.get("specs") else list(specs)
    unknown = [n for n in names if n not in specs]
    if unknown:
        sys.exit(f"unknown spec(s): {','.join(unknown)}  (known: {','.join(specs)})")
    return names


def load_rows(src: str, opts: Dict[str, str], names: List[str], specs: Dict[str, Any], key: str = "s"):
    """Rows of a finished run for ``from=``: ``(rows, seeds, names)``.

    ``names`` narrows to the specs present in the file — in the order asked
    for when ``specs=`` was given, else in the order of *specs*.
    """
    rows = json.load(open(src))
    seeds = sorted({r["seed"] for r in rows})
    have = {r[key] for r in rows}
    names = [n for n in names if n in have] if opts.get("specs") else [n for n in specs if n in have]
    print(f"read {len(rows)} rows from {src}")
    return rows, seeds, names


# ── specs ──


def base_spec():
    """The IOH reference spec every screen derives its specs from."""
    return [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"][0]


def solo_spec(base, name: str, seed_name: str, heuristic: Tuple[Any, Dict[str, Any]]):
    """One heuristic alone under ``StrategyRoundRobin``, no analyzers."""
    from panobbgo.strategies import StrategyRoundRobin

    return dataclasses.replace(
        base,
        name=name,
        seed_name=seed_name,
        strategy_class=StrategyRoundRobin,
        heuristics=[heuristic],
        analyzers=[],
    )


def table_spec(base, name: str, entry, seed_name: str = "screen"):
    """A spec from a ``name -> (strategy_class, heuristics, strategy kwargs, analyzers)`` table entry.

    One ``seed_name`` for the whole screen: every spec sees the identical
    cell seeds, so the comparison is paired on the stream.
    """
    cls, heuristics, kw, analyzers = entry
    return dataclasses.replace(
        base,
        name=name,
        seed_name=seed_name,
        strategy_class=cls,
        heuristics=[(c, dict(k)) for c, k in heuristics],
        analyzers=[(c, dict(k)) for c, k in analyzers],
        config_overrides=dict(kw),
    )


# ── running ──


def runs_to_rows(seed: int, runs, fields=IOH_FIELDS, extra: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """One JSON row per harness run: ``seed``, then *extra*, then the *fields*."""
    extra = extra or {}
    return [{"seed": seed, **extra, **{k: getattr(x, a) for k, a in fields}} for x in runs]


def run_seeds(seeds: Iterable[int], out: str, batches: Callable[[int], Iterable[List[Dict[str, Any]]]]):
    """Collect ``batches(seed)`` for every seed, rewriting *out* after each batch.

    An interrupted screen never loses finished work.  Returns all rows.
    """
    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for seed in seeds:
        for batch in batches(seed):
            rows += batch
            json.dump(rows, open(out, "w"))
        print(f"seed {seed} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    return rows


# ── folding ──


def ioh_cell(r: Dict[str, Any]) -> Cell:
    """``(seed, fid, dim, inst)``; ``fid`` is ``None`` without a function axis (and in older files)."""
    return (r["seed"], r.get("fid"), r["dim"], r["inst"])


def fold(
    rows: Iterable[Dict[str, Any]],
    names: Optional[Iterable[str]] = None,
    key: str = "s",
    cell: Callable[[Dict[str, Any]], Cell] = ioh_cell,
    short_label: Optional[Callable[[Dict[str, Any]], str]] = None,
):
    """Fold rows into ``cells``, and collect ``errs`` and ``short`` runs per spec.

    ``cells`` maps a cell key to ``{spec: mean AOCC over reps}``; rows of specs
    outside *names* are skipped.  ``errs[spec]`` lists error messages.  With
    *short_label*, ``short[spec]`` lists ``(label, evals, budget)`` of every
    run that stopped short of 98 % of its budget — a stall, an exhausted arm,
    or a strategy returning no points.
    """
    keep = None if names is None else set(names)
    raw: Dict[Cell, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    errs: Dict[str, list] = defaultdict(list)
    short: Dict[str, list] = defaultdict(list)
    for r in rows:
        if keep is not None and r[key] not in keep:
            continue
        raw[cell(r)][r[key]].append(r["aocc"])
        if r["err"]:
            errs[r[key]].append(r["err"])
        if short_label is not None and r.get("budget") and r.get("evals", 0) < 0.98 * r["budget"]:
            short[r[key]].append((short_label(r), r["evals"], r["budget"]))
    cells = {k: {s: st.mean(v) for s, v in d.items()} for k, d in raw.items()}
    return cells, errs, short


def ioh_label(r: Dict[str, Any]) -> str:
    """``f<fid>d<dim>i<inst>`` (no ``f`` part without a function axis)."""
    fid = r.get("fid")
    return (f"f{fid}" if fid is not None else "") + f"d{r['dim']}i{r['inst']}"


# ── statistics ──


def match(dim=None, group=None, inst=None) -> Callable[[Cell], bool]:
    """Cell predicate on ``dim``, ``group`` (fid / family) and ``inst``; ``None`` matches all."""

    def where(k: Cell) -> bool:
        _, g, d, i = k
        return (dim is None or d == dim) and (group is None or g == group) and (inst is None or i == inst)

    return where


def mean_of(cells: Dict[Cell, Dict[str, float]], name: str, where: Where = None) -> float:
    """Mean of spec *name* over the selected cells, ``nan`` if it has none."""
    vals = [v[name] for k, v in cells.items() if name in v and (where is None or where(k))]
    return st.mean(vals) if vals else float("nan")


def paired_by_seed(cells: Dict[Cell, Dict[str, float]], a: str, b: str, where: Where = None) -> Dict[Any, float]:
    """``{seed: mean of a − b}`` over the selected cells where both have a result."""
    ps: Dict[Any, List[float]] = defaultdict(list)
    for k, v in cells.items():
        if a in v and b in v and (where is None or where(k)):
            ps[k[0]].append(v[a] - v[b])
    return {s: st.mean(x) for s, x in ps.items()}


def paired(cells: Dict[Cell, Dict[str, float]], a: str, b: str, where: Where = None) -> List[float]:
    """The per-seed deltas of :func:`paired_by_seed`, in seed order of first appearance."""
    return list(paired_by_seed(cells, a, b, where).values())


def delta(cells: Dict[Cell, Dict[str, float]], a: str, b: str) -> float:
    """Mean paired delta of *a* over *b*, or ``nan`` if uncomparable."""
    ds = paired(cells, a, b)
    return st.mean(ds) if ds else float("nan")


def mean_or_nan(values: List[float]) -> float:
    return st.mean(values or [float("nan")])


# ── printing ──


def dim_head(dims: Iterable[Any]) -> str:
    return "".join(f"  {'d=' + str(d):>8s}" for d in dims)


def print_means(cells, order: List[str], dims: List[Any], errs, short, width: int) -> None:
    """Mean AOCC per spec, overall and per dimension, with error/short counts."""
    print(f"\n{'spec':{width}s} {'mean':>7s} " + dim_head(dims))
    for s in order:
        per = "".join(f"  {mean_of(cells, s, match(dim=d)):8.4f}" for d in dims)
        tail = f"  errors={len(errs[s])}" if errs[s] else ""
        tail += f"  short={len(short[s])}" if short[s] else ""
        print(f"{s:{width}s} {mean_of(cells, s):7.4f} " + per + tail)


def print_delta_table(cells, order: List[str], ref: str, dims: List[Any], width: int) -> None:
    """Paired deltas against *ref*: overall t-CI, seeds positive, per-dimension means."""
    print(f"\ndelta vs {ref} (paired per cell, t-CI over per-seed means)")
    print(f"{'spec':{width}s} {'delta':>8s} {'95% CI':>21s} {'seeds':>7s} " + dim_head(dims))
    for s in order:
        if s == ref:
            continue
        ds = paired(cells, s, ref)
        if not ds:
            continue
        m, h = t_ci(ds)
        band = f"[{m - h:+.4f},{m + h:+.4f}]" if h == h else "        (n<2)"
        flag = " <--" if h == h and (m - h > 0 or m + h < 0) else ""
        per = "".join(f"  {mean_or_nan(paired(cells, s, ref, match(dim=d))):+8.4f}" for d in dims)
        print(f"{s:{width}s} {m:+8.4f} {band:>21s} {sum(d > 0 for d in ds):3d}/{len(ds):<3d} " + per + flag)


def print_run_health(names: Iterable[str], errs, short) -> None:
    """What the harness itself reported: errored runs and runs short of their budget."""
    print("\n--- run health ---")
    if not any(errs.values()) and not any(short.values()):
        print("  no errored runs, every run spent its full budget")
    for s in names:
        if errs[s]:
            seen = sorted(set(errs[s]))[:3]
            print(f"  {s}: {len(errs[s])} errored run(s); first messages: {seen}")
        if short[s]:
            ex = ", ".join(f"{label} {e}/{b}" for label, e, b in short[s][:4])
            print(f"  {s}: {len(short[s])} run(s) below budget: {ex}")
