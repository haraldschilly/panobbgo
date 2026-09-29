# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
"""DISCOVERY §73: the trust-region candidates against the headline spec and the pool's best.

Descriptive tables (unadjusted, outside the Holm family) from the unit files of the two runs, on
the cell's common runs, with and without the ellipsoid family (the pool's best re-selected there,
as ``measure.py``'s ex-ellipsoid table does).  Also r05 − ``RoundRobin_TRQ`` (same RNG streams)
and r05 − ``Blocks_warm_CMAES_JSO_TRQ``.

Usage (the release ``measure-2026-09-29-confirm-3001``, each tarball unpacked into a directory)::

    uv run python planning/results/2026-09-29-measure-confirm-3001/trq_table.py MAIN_DIR Q1_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import measure as ms  # noqa: E402

HEADLINE = "Blocks_warm_CMAES_JSO"
SPECS = ("RoundRobin_TRQ", "RoundRobin_TRQ_r05", "Blocks_warm_CMAES_JSO_TRQ", "RoundRobin_COBYQA")
PAIRS = (("RoundRobin_TRQ_r05", "RoundRobin_TRQ"), ("RoundRobin_TRQ_r05", "Blocks_warm_CMAES_JSO_TRQ"))


def fmt(st: dict) -> str:
    return f"{st['delta']:+.3f} [{st['ci_low']:+.3f},{st['ci_high']:+.3f}] {st['wins']}/{st['n_seeds']}"


def main(dirs: list[str]) -> None:
    rows, pair_rows = [], []
    for d in map(Path, dirs):
        summary = json.loads((d / "measure-summary" / "summary.json").read_text())
        payloads, bad = ms.load_units(d)
        if bad:
            raise SystemExit(f"unreadable: {bad}")
        cells = ms.collect(payloads)
        for cid, c in sorted(summary["cells"].items(), key=lambda kv: (kv[1]["q"] > 1, kv[1]["dim"], kv[1]["q"])):
            strats = cells[(c["preset"], c["dim"], c["bm"], c["q"])]
            m = c["headline_metric"]
            best = c["pool_best"][m]
            common = set(strats[HEADLINE])
            for n in c["pool"]:
                common &= set(strats[n])
            ex = {k for k in common if not ms.is_ex_family(k[1])}
            best_ex = max(c["pool"], key=lambda n: ms.mean_over_seeds(strats[n], m, [k for k in strats[n] if k in ex]))
            for s in SPECS:
                a = strats[s]
                rows.append(
                    f"| {cid} | {s} | {fmt(ms.paired(a, strats[HEADLINE], m, common))} | "
                    f"{fmt(ms.paired(a, strats[best], m, common))} | {fmt(ms.paired(a, strats[HEADLINE], m, ex))} | "
                    f"{fmt(ms.paired(a, strats[best_ex], m, ex))} ({ms.label(best_ex)}) |"
                )
            for a, b in PAIRS:
                pair_rows.append(
                    f"| {cid} | {a} − {b} | {fmt(ms.paired(strats[a], strats[b], m, common))} | "
                    f"{fmt(ms.paired(strats[a], strats[b], m, ex))} |"
                )
    print("| cell | spec | − headline | − pool best | − headline, ex-ellipsoid | − pool best, ex-ellipsoid |")
    print("|---|---|---|---|---|---|")
    print("\n".join(rows))
    print()
    print("| cell | pair | Δ | Δ ex-ellipsoid |")
    print("|---|---|---|---|")
    print("\n".join(pair_rows))


if __name__ == "__main__":
    main(sys.argv[1:])
