# -*- coding: utf8 -*-
# Copyright 2012-2026 Panobbgo Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Worker-utilisation probe for panobbgo specs on the virtual clock (DISCOVERY §63).

Runs the ``scripts/measure.py`` core configuration (free preset, log-normal
durations sigma 0.5, async policy, CRN durations, ``sync_eval``) for one spec,
one (family, instance) per run, and records per run from a patched
:class:`~panobbgo.virtual_clock.VirtualClock`:

* ``busy_frac``: busy worker-time / (q * makespan); ``busy_frac_horizon`` the
  same over [0, budget / q], the part ``aocc_time`` scores;
* ``idle_starved`` / ``idle_tail``: idle worker-time while budget was left /
  after the whole budget was dispatched (fractions of q * makespan);
* ``makespan_over_ideal``: makespan / (budget / q);
* decision points: how many were asked (``request_cap``) against returned,
  how many came back empty, and which block owner was asked;
* ``owner_busy``: mean busy workers while each block owner held the block;
* ``trimmed``: candidates the ``admit`` safety net handed back (should be 0);
* ``first_full``: the first virtual time all q workers were busy (``None``: never).

Usage (4 processes, niced; ``--no-floor`` runs CMA-ES without the
``popsize_min_workers`` floor, i.e. the pre-§63 behaviour)::

    nice -n 10 ionice -c3 uv run python sketchpad/worker_utilisation_probe.py \\
        --spec Blocks_warm_CMAES_JSO --dims 2,5 --qs 4,16,64 \\
        --seeds 3,7,42,1234,2025 --out util.json [--no-floor]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import panobbgo.virtual_clock as vc

_LOG: List[Dict[str, Any]] = []


def _patch() -> None:
    """Wrap ``VirtualClock.step`` / ``_advance`` once per process to record the schedule."""
    if getattr(vc.VirtualClock, "_probed", False):
        return
    vc.VirtualClock._probed = True  # type: ignore[attr-defined]
    orig_step = vc.VirtualClock.step
    orig_advance = vc.VirtualClock._advance

    def step(self: Any, points: List[Any], *args: Any, **kwargs: Any) -> None:
        st = self.strategy
        rec = getattr(self, "_probe", None)
        if rec is None:
            rec = self._probe = {
                "asks": [],
                "segs": [],
                "short_owner": Counter(),
                "zero_owner": Counter(),
                "by_who": Counter(),
            }
            _LOG.append(rec)
        cap = st.request_cap
        owner = getattr(st, "_owner", None) or "-"
        if cap is not None and cap > 0:
            rec["asks"].append((self.now, cap, len(points)))
            if len(points) < cap:
                rec["short_owner"][owner] += 1
                if not points:
                    rec["zero_owner"][owner] += 1
        for p in points:
            rec["by_who"][str(p.who).split(":")[0]] += 1
        orig_step(self, points, *args, **kwargs)
        rec["trimmed"] = self.n_trimmed

    def _advance(self: Any) -> None:
        rec, st = self._probe, self.strategy
        t0, busy = self.now, self.busy
        if busy == self.workers and "first_full" not in rec:
            rec["first_full"] = t0
        room = int(st.config.max_eval) - st._dispatched
        orig_advance(self)
        rec["segs"].append((t0, self.now, busy, room > 0, getattr(st, "_owner", None) or "-"))

    vc.VirtualClock.step = step  # type: ignore[method-assign]
    vc.VirtualClock._advance = _advance  # type: ignore[method-assign]


def run(job: Tuple[str, int, int, int, int, int, int, bool, str, int]) -> Dict[str, Any]:
    """One run: ``(spec, family index, instance, dim, budget multiplier, q, seed, no_floor, overrides)``.

    ``overrides`` is a JSON object merged into the spec's ``config_overrides``
    (the strategy's keyword arguments); the run keeps the spec's RNG identity,
    so a variant is paired with the spec seed by seed (DISCOVERY §72).
    """
    spec_name, fam_idx, inst, dim, bm, q, seed, no_floor, overrides, min_gen = job
    _patch()
    if min_gen:
        from panobbgo.heuristics.cma_es import CMAES

        CMAES.MIN_GENERATIONS = min_gen  # prototype: the floor's budget cap (§63.3)
    if no_floor:
        from panobbgo.heuristics.cma_es import CMAES

        CMAES._n_workers = lambda self: 1  # type: ignore[method-assign]
    from panobbgo.harness_families import make_families_battery, run_family_harness
    from panobbgo.harness_ioh import make_ioh_strategies

    spec = {s.name: s for s in make_ioh_strategies()}[spec_name]
    if overrides:
        from dataclasses import replace

        extra = json.loads(overrides)
        # ``"_heuristics": {"CMAES": {...}}`` merges into an arm's kwargs (by class name).
        per_arm = extra.pop("_heuristics", {})
        heuristics = [(cls, {**kw, **per_arm.get(cls.__name__, {})}) for cls, kw in spec.heuristics]
        spec = replace(
            spec,
            heuristics=heuristics,
            config_overrides={**spec.config_overrides, **extra},
            seed_name=spec.rng_identity,
        )
    insts = list(make_families_battery(dims=(dim,), n_instances=3))
    fams = list(dict.fromkeys(str(p.family) for _, p in insts))
    chosen = [(n, p) for n, p in insts if str(p.family) == fams[fam_idx] and int(p.instance) == inst]
    _LOG.clear()
    res = run_family_harness(
        [spec],
        chosen,
        budget_multiplier=bm,
        base_seed=seed,
        sync_eval=True,
        progress=False,
        jobs=1,
        virtual=vc.VirtualSpec(workers=q, duration="lognormal", sigma=0.5, policy="async"),
    )
    r, rec = res.runs[0], _LOG[-1]
    budget = bm * dim
    horizon = budget / q
    segs = rec["segs"]
    makespan = segs[-1][1] if segs else 0.0
    asks = rec["asks"]

    def frac(x: float) -> Any:
        return x / (q * makespan) if makespan else None

    owners = {s[4] for s in segs}
    owner_time = {o: sum(b - a for a, b, _n, room, oo in segs if oo == o and room) for o in owners}
    return {
        "spec": spec_name,
        "no_floor": no_floor,
        "overrides": overrides,
        "error": r.error,
        "fam": fams[fam_idx],
        "inst": inst,
        "dim": dim,
        "q": q,
        "seed": seed,
        "budget": budget,
        "n_evals": r.n_evals,
        "aocc": r.aocc,
        "aocc_time": r.aocc_time,
        "makespan_over_ideal": makespan / horizon,
        "busy_frac": frac(sum((b - a) * n for a, b, n, _r, _o in segs)),
        "busy_frac_horizon": sum((min(b, horizon) - a) * n for a, b, n, _r, _o in segs if a < horizon) / budget,
        "idle_starved": frac(sum((b - a) * (q - n) for a, b, n, room, _o in segs if room)),
        "idle_tail": frac(sum((b - a) * (q - n) for a, b, n, room, _o in segs if not room)),
        "asks": len(asks),
        "requested": sum(c for _, c, _ in asks),
        "returned": sum(p for _, _, p in asks),
        "zero_asks": sum(1 for _, _, p in asks if p == 0),
        "owner_busy": {
            o: sum((b - a) * n for a, b, n, room, oo in segs if oo == o and room) / max(1e-12, owner_time[o])
            for o in owners
        },
        "owner_time": owner_time,
        "short_owner": dict(rec["short_owner"]),
        "zero_owner": dict(rec["zero_owner"]),
        "trimmed": rec.get("trimmed", 0),
        "dispatched_by": dict(rec["by_who"]),
        "first_full": rec.get("first_full"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ints = lambda s: [int(x) for x in s.split(",")]  # noqa: E731
    ap.add_argument("--spec", default="Blocks_warm_CMAES_JSO")
    ap.add_argument("--dims", type=ints, default=[2, 5])
    ap.add_argument("--qs", type=ints, default=[4, 16, 64])
    ap.add_argument("--seeds", type=ints, default=[3, 7, 42, 1234, 2025])
    ap.add_argument("--fams", type=ints, default=[0, 1, 2, 3, 4])
    ap.add_argument("--insts", type=ints, default=[0, 1, 2])
    ap.add_argument("--bm", type=int, default=100)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--no-floor", action="store_true")
    ap.add_argument("--overrides", default="", help="JSON merged into the spec's config_overrides (paired seeds)")
    ap.add_argument("--cma-min-gen", type=int, default=0, help="prototype: CMAES.MIN_GENERATIONS (0: unchanged)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [
        (a.spec, fam, inst, dim, a.bm, q, seed, a.no_floor, a.overrides, a.cma_min_gen)
        for dim in a.dims
        for q in a.qs
        for seed in a.seeds
        for fam in a.fams
        for inst in a.insts
    ]
    with Pool(min(4, a.procs)) as pool:  # the laptop is shared: at most 4 processes
        out = pool.map(run, jobs, chunksize=1)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
