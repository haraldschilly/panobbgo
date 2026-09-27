# -*- coding: utf8 -*-
# Copyright 2026 Harald Schilly <harald.schilly@gmail.com>
"""How much budget does each arm lose in failure regions, and does the failure model win it back?

Roadmap ``planning/DESIGN_roadmap_2026-09-26.md`` §4 D, step 1
(``planning/DISCOVERY_2026-09-09.md`` §71).  The expensive-track set-up of
``scripts/measure.py`` (family preset, ``bm``·d evaluations, the virtual
clock with q workers, async policy, log-normal durations, sigma 0.5) on a
chosen list of single arms, and per run the share of spent evaluations that
failed (``IOHRunRecord.n_failed / n_evals``), AOCC and ``aocc_time``.

Arms (``ARMS``): the panobbgo single arms and the headline portfolio, the
cheap-track external baselines and Py-BOBYQA.  ``<arm>+fm`` is the arm with
the shared failure model (:class:`~panobbgo.analyzers.failure_model.FailureModel`,
``filter=True``) and its arm-specific handling (``failure_aware``) switched on,
``<arm>+filter`` / ``<arm>+aware`` one of the two (:data:`MODES`); every
variant shares the arm's RNG stream (``seed_name``), so a paired delta
carries only the change.

Usage::

    uv run python benchmarks/failure_screen.py run OUT_DIR --seeds 5 --dims 2,5 --qs 1,4 \\
        [--bm 100] [--preset failure] [--arms a,b,...] [--jobs 3]
    uv run python benchmarks/failure_screen.py summary OUT_DIR [--pairs]

``run`` writes one JSON file of records per (preset, dim, q, seed) and skips
files that exist.  ``summary`` prints per-arm means over every record;
``--pairs`` adds the paired ``<arm>+fm`` minus ``<arm>`` deltas (t-interval
over seeds).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rebaseline import resolve_seeds  # noqa: E402

#: External baselines (``harness_baselines``), as in ``measure.py``'s core group.
EXTERNALS: Tuple[str, ...] = (
    "Baseline_pycma_IPOP",
    "Baseline_pycma_BIPOP",
    "Baseline_NGOpt",
    "Baseline_Optuna_CmaEs",
    "Baseline_Optuna_TPE",
    "Baseline_PyBOBYQA",
)

#: The failure model's options for the ``+fm`` / ``+filter`` variants (the analyzer's defaults).
FM_KWARGS: Dict[str, Any] = {"filter": True}

#: Variants of a panobbgo arm: ``base``; ``fm`` = the filter and the arm's
#: ``failure_aware`` handling (CMA-ES, TRQ; the others have none); ``filter`` =
#: the filter only; ``aware`` = the arm's handling only (no model).
MODES: Tuple[str, ...] = ("base", "fm", "filter", "aware")


def _panobbgo_specs() -> Dict[str, Callable[[str], Any]]:
    """Arm name -> factory(with_model) of its :class:`StrategySpec`."""
    from panobbgo.analyzers.failure_model import FailureModel
    from panobbgo.benchmark import StrategySpec
    from panobbgo.heuristics import CMAES, COBYQA, JSO, NelderMead, Random, TrustRegionQuadratic
    from panobbgo.harness_ioh import make_ioh_strategies
    from panobbgo.strategies import StrategyRoundRobin

    blocks = next(s for s in make_ioh_strategies() if s.name == "Blocks_warm_CMAES_JSO")

    def rr(name: str, heuristics: List[Tuple[type, Dict[str, Any]]], fm_heur: Dict[str, Any]):
        def make(mode: str) -> Any:
            aware = mode in ("fm", "aware")
            heur = [(h, {**kw, **(fm_heur if aware and h is heuristics[-1][0] else {})}) for h, kw in heuristics]
            return StrategySpec(
                name=name + ("" if mode == "base" else "+" + mode),
                strategy_class=StrategyRoundRobin,
                heuristics=heur,
                analyzers=[(FailureModel, dict(FM_KWARGS))] if mode in ("fm", "filter") else [],
                seed_name=name,
            )

        return make

    def blocks_make(mode: str) -> Any:
        aware = mode in ("fm", "aware")
        heur = [(h, {**kw, **({"failure_aware": True} if aware and h is CMAES else {})}) for h, kw in blocks.heuristics]
        return StrategySpec(
            name=blocks.name + ("" if mode == "base" else "+" + mode),
            strategy_class=blocks.strategy_class,
            heuristics=heur,
            analyzers=list(blocks.analyzers) + ([(FailureModel, dict(FM_KWARGS))] if mode in ("fm", "filter") else []),
            config_overrides=dict(blocks.config_overrides),
            seed_name=blocks.name,
        )

    return {
        "RoundRobin_Random": rr("RoundRobin_Random", [(Random, {})], {}),
        "RoundRobin_CMAES": rr("RoundRobin_CMAES", [(CMAES, {})], {"failure_aware": True}),
        "RoundRobin_JSO": rr("RoundRobin_JSO", [(JSO, {"NP_init": "auto"})], {}),
        "RoundRobin_TRQ": rr("RoundRobin_TRQ", [(TrustRegionQuadratic, {})], {"failure_aware": True}),
        "RoundRobin_COBYQA": rr("RoundRobin_COBYQA", [(COBYQA, {})], {}),
        # Nelder-Mead needs results to build a simplex from: Random seeds it.
        "RoundRobin_Random_NM": rr("RoundRobin_Random_NM", [(Random, {}), (NelderMead, {})], {}),
        "Blocks_warm_CMAES_JSO": blocks_make,
    }


ARMS: Tuple[str, ...] = (
    "RoundRobin_Random",
    "RoundRobin_CMAES",
    "RoundRobin_JSO",
    "RoundRobin_TRQ",
    "RoundRobin_COBYQA",
    "RoundRobin_Random_NM",
    "Blocks_warm_CMAES_JSO",
) + EXTERNALS


def make_specs(names: Sequence[str]) -> List[Any]:
    """Specs for ``names`` (an arm, or ``<arm>+fm`` for a panobbgo arm with the failure model)."""
    from panobbgo.harness_baselines import make_baseline_strategies

    factories = _panobbgo_specs()
    ext = {s.name: s for s in make_baseline_strategies([n for n in names if n in EXTERNALS])}
    out = []
    for n in names:
        base, _, mode = n.partition("+")
        mode = mode or "base"
        if base in factories and mode in MODES:
            out.append(factories[base](mode))
        elif base in ext and mode == "base":
            out.append(ext[base])
        else:
            raise ValueError(f"unknown arm {n!r}")
    return out


def _instances(preset: str, dim: int) -> List[Any]:
    from panobbgo.harness_families import make_failure_battery, make_families_battery

    make = {"free": make_families_battery, "failure": make_failure_battery}[preset]
    return list(make(dims=(dim,), n_instances=3))


def cmd_run(args: argparse.Namespace) -> int:
    import panobbgo.fp_pin  # noqa: F401  # pyright: ignore[reportUnusedImport]
    from panobbgo import local_run
    from panobbgo.harness_families import run_family_harness
    from panobbgo.virtual_clock import VirtualSpec

    local_run.be_nice()
    local_run.pin_blas_env()
    local_run.pin_blas()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in args.arms.split(",") if n] if args.arms else list(ARMS)
    tag = args.tag or "run"
    for dim in [int(d) for d in args.dims.split(",")]:
        for q in [int(x) for x in args.qs.split(",")]:
            for seed in resolve_seeds(args.seeds):
                path = out / f"{tag}.{args.preset}.b{args.bm}.q{q}.d{dim}.s{seed}.json"
                if path.exists():
                    continue
                print(f"{path.name}: {len(names)} arms", flush=True)
                result = run_family_harness(
                    make_specs(names),
                    _instances(args.preset, dim),
                    budget_multiplier=args.bm,
                    base_seed=seed,
                    sync_eval=True,
                    progress=False,
                    battery_name=f"failure-screen-{args.preset}",
                    jobs=args.jobs,
                    virtual=VirtualSpec(workers=q, duration="lognormal", sigma=0.5, policy="async"),
                )
                rows = []
                for r in result.runs:
                    rows.append(
                        {
                            "arm": r.strategy_name,
                            "family": r.problem_kind,
                            "dim": r.dim,
                            "inst": r.instance,
                            "q": q,
                            "seed": seed,
                            "n_evals": r.n_evals,
                            "n_failed": r.n_failed,
                            "aocc": r.aocc,
                            "aocc_time": r.aocc_time,
                            "precision": r.best_fx - r.f_opt,
                            "error": r.error,
                        }
                    )
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(rows, default=float))
                tmp.replace(path)
    return 0


def load_rows(src: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(src.glob("*.json")):
        rows.extend(json.loads(p.read_text()))
    return rows


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _t_ci(ds: Sequence[float]) -> Tuple[float, float]:
    """Mean and 95 % t half-width over ``ds`` (the per-seed deltas)."""
    from scipy import stats

    n = len(ds)
    m = _mean(ds)
    if n < 2:
        return m, float("nan")
    sd = math.sqrt(sum((d - m) ** 2 for d in ds) / (n - 1))
    return m, float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)


def cmd_summary(args: argparse.Namespace) -> int:
    rows = load_rows(Path(args.out_dir))
    cells = sorted({(r["dim"], r["q"]) for r in rows})
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    fams = sorted({r["family"] for r in rows})
    print(f"{len(rows)} records; cells {cells}; families {fams}")
    print("\n## fail share (failed / spent evaluations), mean over runs; per cell and per family (all cells)\n")
    hdr = ["arm"] + [f"d{d} q{q}" for d, q in cells] + ["all"] + [f[: f.rfind("_f")] if "_f" in f else f for f in fams]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for a in arms:
        ra = [r for r in rows if r["arm"] == a and r.get("n_failed") is not None and r["n_evals"]]
        vals = [_mean([r["n_failed"] / r["n_evals"] for r in ra if (r["dim"], r["q"]) == c]) for c in cells]
        vals.append(_mean([r["n_failed"] / r["n_evals"] for r in ra]))
        vals += [_mean([r["n_failed"] / r["n_evals"] for r in ra if r["family"] == f]) for f in fams]
        print("| " + " | ".join([a] + [f"{v:.3f}" for v in vals]) + " |")
    for metric in ("aocc", "aocc_time"):
        print(f"\n## {metric}, mean over runs\n")
        hdr = ["arm"] + [f"d{d} q{q}" for d, q in cells] + ["all"]
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for a in arms:
            ra = [r for r in rows if r["arm"] == a and r.get(metric) is not None]
            vals = [_mean([r[metric] for r in ra if (r["dim"], r["q"]) == c]) for c in cells]
            vals.append(_mean([r[metric] for r in ra]))
            print("| " + " | ".join([a] + [f"{v:.4f}" for v in vals]) + " |")
    errs = [r for r in rows if r.get("error")]
    if errs:
        print(f"\n{len(errs)} records with an error, e.g. {errs[0]['arm']}: {errs[0]['error']}")
    if args.pairs:
        _pairs(rows, cells)
    return 0


def _pairs(rows: List[Dict[str, Any]], cells: List[Tuple[int, int]]) -> None:
    key = {(r["arm"], r["family"], r["dim"], r["inst"], r["q"], r["seed"]): r for r in rows}
    variants = sorted({r["arm"] for r in rows if "+" in r["arm"]})
    for metric in ("fail", "aocc", "aocc_time"):
        print(f"\n## paired delta (variant minus base) of {metric}: mean over seeds of the per-seed mean, 95 % t CI\n")
        hdr = ["arm"] + [f"d{d} q{q}" for d, q in cells] + ["all"]
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for v in variants:
            b = v.split("+")[0]
            out = [v]
            for sel in [[c] for c in cells] + [cells]:
                per_seed: Dict[int, List[float]] = {}
                for (arm, fam, dim, inst, q, seed), r in key.items():
                    if arm != v or (dim, q) not in sel:
                        continue
                    base = key.get((b, fam, dim, inst, q, seed))
                    if base is None:
                        continue
                    if metric == "fail":
                        dv = r["n_failed"] / r["n_evals"] - base["n_failed"] / base["n_evals"]
                    else:
                        if r.get(metric) is None or base.get(metric) is None:
                            continue
                        dv = r[metric] - base[metric]
                    per_seed.setdefault(seed, []).append(dv)
                m, h = _t_ci([_mean(x) for x in per_seed.values()])
                wins = sum(1 for x in per_seed.values() if _mean(x) > 0)
                out.append(f"{m:+.4f} ± {h:.4f} ({wins}/{len(per_seed)} up)")
            print("| " + " | ".join(out) + " |")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("--seeds", default="5")
    r.add_argument("--dims", default="2,5")
    r.add_argument("--qs", default="1,4")
    r.add_argument("--bm", type=int, default=100)
    r.add_argument("--preset", default="failure", choices=("failure", "free"))
    r.add_argument("--arms", default="")
    r.add_argument("--tag", default="")
    r.add_argument("--jobs", type=int, default=3)
    s = sub.add_parser("summary")
    s.add_argument("out_dir")
    s.add_argument("--pairs", action="store_true")
    args = p.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_summary(args)


if __name__ == "__main__":
    sys.exit(main())
