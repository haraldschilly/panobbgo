"""Acceptance run for the new ``NP_init="auto"`` sizing rule.

``planning/DISCOVERY_2026-09-09.md`` §17 located the population law at
``NP ≈ 3·dim`` at the reference budget of ``500·dim`` evaluations, against
the ``18·dim`` the library shipped.  This script decides whether the new
rule may become the default: it runs, per arm, three solo-arm variants
paired on the same RNG stream and reports the paired per-seed deltas.

    auto_old   the *shipped* rule, ``clip(round(min(18·dim, budget/12)), 6, 400)``,
               passed as an explicit int (that is what ``"auto"`` used to give)
    auto_new   ``NP_init="auto"`` — i.e. whatever the rule in
               :mod:`panobbgo.heuristics.lshade` currently resolves to
    fixed_cd   ``NP_init = c·dim`` with ``c`` the arm class's
               ``AUTO_DIM_COEF`` (3 for L-SHADE/jSO, 4 for NLSHADE_LBC — see
               DISCOVERY §24): the plain dimensional rule with no budget term.
               At ``bm=500`` this is ``auto_new`` by construction, so a
               non-zero delta here would mean the harness is not pairing

Because ``NP_init`` must be a concrete ``int`` per spec and the two explicit
variants depend on the dimension, each dimension of the battery is run as its
own battery (``dataclasses.replace(battery, dims=(d,))``) with per-dimension
specs.  Everything else follows ``benchmarks/arm_sweep.py``: solo arms via
``StrategyRoundRobin`` with no analyzers, a constant ``seed_name`` so all
variants share the RNG stream, ``sync_eval=True``, rows flushed after every
seed, and paired-per-seed t-CIs reported overall and per dimension.

Usage::

    uv run python benchmarks/np_accept.py ARM OUT.json SEED [SEED ...] \
        [dims=2,5] [bm=500] [jobs=N]

    ARM   one of: lshade, jso, lbc
    SEED  base seeds; the canonical decision roster is the 12 seeds in
          ``panobbgo.harness_ioh.DEFAULT_DECISION_SEEDS``
    jobs  run the (seed, cell) runs in N worker processes; results do not depend on N (default 1)
"""

import sys
import dataclasses
import statistics as st
from collections import defaultdict
from panobbgo.harness_ioh import make_standard_battery, run_ioh_harness, t_ci
from panobbgo.local_run import screen_jobs
from panobbgo.heuristics import JSO, LSHADE, NLSHADE_LBC

if __package__:  # ``python -m benchmarks.<screen>``: make ``_screen`` importable by name
    import sys

    from . import _screen as _screen_module

    sys.modules.setdefault("_screen", _screen_module)
from _screen import base_spec, fold, int_tuple, match, mean_of, paired_by_seed, parse_argv, run_seeds, runs_to_rows
from _screen import solo_spec


def main():
    BASE = base_spec()

    # The rule as it shipped before ``planning/DISCOVERY_2026-09-09.md`` §17:
    # ``min(18·dim, budget/12)`` clipped into ``[6, 400]``.  Reproduced here (not
    # imported) so the reference stays the *old* behaviour even after the module
    # constants change.
    _OLD_DIM_COEF, _OLD_GEN_TARGET, _OLD_MIN_NP, _OLD_MAX_NP = 18, 12.0, 6, 400

    def old_auto(dim: int, budget: float) -> int:
        """What ``NP_init="auto"`` resolved to before the new rule."""
        raw = min(float(_OLD_DIM_COEF * dim), budget / _OLD_GEN_TARGET)
        return int(min(max(round(raw), _OLD_MIN_NP), _OLD_MAX_NP))

    ARMS = {
        "lshade": (LSHADE, {}),
        "jso": (JSO, {}),
        "lbc": (NLSHADE_LBC, {}),
    }

    _opts, _, pos = parse_argv(sys.argv[1:])
    arm, out = pos[0], pos[1]
    seeds = [int(x) for x in pos[2:]]
    cls, base_kw = ARMS[arm]
    JOBS = screen_jobs(_opts)

    battery = make_standard_battery()
    if "dims" in _opts or "bm" in _opts:
        battery = dataclasses.replace(
            battery,
            dims=int_tuple(_opts.get("dims", "2,5")),
            budget_multiplier=int(_opts.get("bm", battery.budget_multiplier)),
        )

    def solo(name, kw):
        # ``seed_name`` is the *arm*, constant across all three variants, so each
        # of them runs the identical RNG stream on every (dim, inst, rep) cell and
        # the paired delta carries the parameter's effect, not run-to-run variance.
        return solo_spec(BASE, name, arm, (cls, {**base_kw, **kw}))

    def fixed_cd(dim: int) -> int:
        """``c·dim`` with the arm class's own per-dimension coefficient."""
        return max(6, round(cls.AUTO_DIM_COEF * dim))

    def specs_for(dim: int):
        budget = battery.budget_for(dim)
        return [
            solo("auto_old", {"NP_init": old_auto(dim, budget)}),
            solo("auto_new", {"NP_init": "auto"}),
            solo("fixed_cd", {"NP_init": fixed_cd(dim)}),
        ]

    def batches(seed):
        for d in battery.dims:
            one_dim = dataclasses.replace(battery, dims=(d,))
            r = run_ioh_harness(specs_for(d), one_dim, base_seed=seed, progress=False, sync_eval=True, jobs=JOBS)
            yield runs_to_rows(seed, r.runs)

    rows = run_seeds(seeds, out, batches)

    # Cells are (seed, fid, dim, inst); reps fold into their mean.  ``fid`` is
    # None without a function axis, so the BBOB axis cannot merge cells.
    tot = defaultdict(list)
    for r in rows:
        tot[r["s"]].append(r["aocc"])
    by, errs, _ = fold(rows)
    n = len(seeds)
    dims = sorted({r["dim"] for r in rows})

    def ci(ds):
        m, h = t_ci(ds)
        return m, m - h, m + h

    def report(name, ref):
        ds = list(paired_by_seed(by, name, ref).values())
        m, lo, hi = ci(ds)
        pos = sum(d > 0 for d in ds)
        flag = " <--" if (lo > 0 or hi < 0) else "   "
        print(f"\n{name} vs {ref}:  {m:+.4f} [{lo:+.4f},{hi:+.4f}]  {pos}/{len(ds)} seeds positive{flag}")
        for dim in dims:
            dd = list(paired_by_seed(by, name, ref, match(dim=dim)).values())
            dm, dlo, dhi = ci(dd)
            dflag = " <--" if (dlo > 0 or dhi < 0) else ""
            print(f"    d={dim:<3d} {dm:+.4f} [{dlo:+.4f},{dhi:+.4f}]  {sum(x > 0 for x in dd)}/{len(dd)}{dflag}")
        per_seed = paired_by_seed(by, name, ref)
        print("    per seed: " + "  ".join(f"{s}:{v:+.4f}" for s, v in sorted(per_seed.items())))

    print(f"\n=== {arm} ===  ({n} seeds, dims {dims}, budget {battery.budget_multiplier}*d)")
    old_np = {d: old_auto(d, battery.budget_for(d)) for d in dims}
    new_np = {d: fixed_cd(d) for d in dims}
    print(f"NP_init: auto_old={old_np}  fixed_cd={new_np} (coef {cls.AUTO_DIM_COEF})")
    head = f"{'variant':12s} {'mean':>7s}"
    print(head + "".join(f"   {'d=' + str(d):>8s}" for d in dims))
    for s in sorted(tot, key=lambda k: -st.mean(tot[k])):
        per = "".join(f"   {mean_of(by, s, match(dim=dim)):8.4f}" for dim in dims)
        err = f"  errors={len(errs[s])}" if errs[s] else ""
        print(f"{s:12s} {st.mean(tot[s]):7.4f}" + per + err)

    report("auto_new", "auto_old")
    report("fixed_cd", "auto_new")
    print("\n`<--` marks a 95% t-CI on the paired delta that excludes zero.")


if __name__ == "__main__":
    main()
