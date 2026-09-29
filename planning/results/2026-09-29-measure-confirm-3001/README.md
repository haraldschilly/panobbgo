# Fresh-seed confirmation of the §72 dim/budget gate (seeds 3001–3012)

DISCOVERY §73.  Two `measure.yml` runs, both dispatched at commit **e86eee3**
(`e86eee34b2c5408c5f0046b4225e7adc5f7285c6`) on optimiser seeds **3001–3012**
(unused before), default free battery (battery seed 20260910), 100·d:

| Run | grid | units | shards |
|---|---|---|---|
| [36418128133](https://github.com/haraldschilly/panobbgo/actions/runs/36418128133) | d 2/5/10 × q 4/16/64; groups core, qLogEI, TuRBO1, SMAC, trq | 540, missing 0, failed 0 | 214 |
| [36418132989](https://github.com/haraldschilly/panobbgo/actions/runs/36418132989) | d 2/5/10 × q 1; groups core, trq (the q = 1 supplement) | 72, missing 0, failed 0 | 6 |

Every shard of both runs ran in one FP environment (`fp_env_id` 80ee2a0090c4).
The 20·d d10 supplement of the plan (optional) was not run.

**Raw data** (not in git): GitHub pre-release
[`measure-2026-09-29-confirm-3001`](https://github.com/haraldschilly/panobbgo/releases/tag/measure-2026-09-29-confirm-3001):
`measure-run<RUN_ID>.tar.gz` holds every artifact of a run as `gh run download
<RUN_ID>` wrote it (unit files, `meta_*.json`, `lscpu_*.txt`, the workflow's
`measure-summary/`); the summaries and plans are attached as separate assets too.

Files here:

* `summary.md`, `plan.json` — the workflow's aggregate and plan of the main run;
* `summary-q1.md`, `plan-q1.json` — the same for the q = 1 supplement;
* `tables.md` — the trust-region candidates against the headline spec and the
  pool's best, with and without the ellipsoid family, and r05 − `RoundRobin_TRQ`
  / − `Blocks_warm_CMAES_JSO_TRQ` (descriptive; made by `trq_table.py`).

Reproduce: download the tarballs, unpack each into a directory `D`, then
`uv run python scripts/measure.py aggregate D --plan D/measure-summary/plan.json --out-dir out`
(with a `measure.py` after #393 the aggregate adds the *Variants − their base spec*
table, i.e. r05 − `RoundRobin_TRQ`, and lists `RoundRobin_TRQ_fa` / `_aware` as
missing: they joined the `trq` group after the dispatch); and
`uv run python planning/results/2026-09-29-measure-confirm-3001/trq_table.py MAIN_DIR Q1_DIR`.

## The pre-declared rule for the gate (TODO.md, fixed by Harald 2026-09-28 before the dispatch)

In the main grid the gate (`Blocks_warm_CMAES_JSO_dimbudget`) binds only at
100·d d10/q4.  It becomes the default if (1) there its paired Δ against
`Blocks_warm_CMAES_JSO` on `aocc_time` has a CI95 above 0, (2) its Δ against the
pool's best has a CI95 not entirely below 0, (3) it is identical in every other
cell (*equal* + ties at 0 = n; a cell marked `(FP)` may instead show Δ ≈ 0), and
(4) no CI95 is entirely below 0 in the supplements' binding cells (100·d
d10/q1; 20·d not run).

| # | condition | measured | source | pass |
|---|---|---|---|---|
| 1 | d10/q4, gate − headline, `aocc_time` | +0.011 [+0.006, +0.016] 11/12 (AOCC +0.011 [+0.006, +0.017]) | `summary.md`, *Secondary specs − Blocks_warm_CMAES_JSO* | yes |
| 2 | d10/q4, gate − pool best (TuRBO1) | +0.004 [+0.001, +0.008] 10/12 | `summary.md`, cell free/d10/b100/q4 | yes |
| 3 | every other main cell identical | equal + ties at 0 = 180/180 in all 8 (d2/q4 180; d2/q16 180; d2/q64 177 + 3; d5/q4 168 + 12; d5/q16 152 + 28; d5/q64 146 + 34; d10/q16 144 + 36; d10/q64 144 + 36); no `(FP)` pair | same table | yes |
| 4 | supplement d10/q1, gate − headline, AOCC | +0.001 [−0.004, +0.007] 6/12 (vs pool best Optuna_CmaEs +0.005 [+0.002, +0.008] 11/12) | `summary-q1.md` | yes |

The supplement's non-binding cells are identical too (d2/q1 180/180, d5/q1
160 + 20).  **All four pass: `regime_gate="dim-budget"` is the default of
`Blocks_warm_CMAES_JSO` since §73.**  At d10/q4 the gated spec also equals
`RegimeGate_oracle` (the oracle gate picks the same row on the noiseless free
battery).

## Headline Holm family (9 cells, pre-declared)

`Blocks_warm_CMAES_JSO` (as dispatched: **ungated**) vs the pool's best on
`aocc_time`, Holm at 0.05 over the 9 cells: **4 wins, 1 loss, 4 unresolved**.

| cell | pool best | Δ [CI95] wins | p_holm | reading |
|---|---|---|---|---|
| d2/q4 | qLogEI | +0.018 [+0.006, +0.030] 10/12 | 0.039 | win |
| d2/q16 | qLogEI | −0.014 [−0.020, −0.008] 1/12 | 0.002 | **loss** |
| d2/q64 | qLogEI | +0.001 [−0.005, +0.008] 6/12 | 0.845 | unresolved |
| d5/q4 | TuRBO1 | +0.007 [+0.000, +0.013] 7/12 | 0.116 | unresolved |
| d5/q16 | qLogEI | +0.026 [+0.023, +0.029] 12/12 | 0.000 | win |
| d5/q64 | qLogEI | +0.001 [−0.002, +0.004] 9/12 | 0.845 | unresolved |
| d10/q4 | TuRBO1 | −0.007 [−0.013, −0.001] 2/12 | 0.080 | unresolved |
| d10/q16 | TuRBO1 | +0.009 [+0.005, +0.013] 11/12 | 0.004 | win |
| d10/q64 | TuRBO1 | +0.018 [+0.017, +0.019] 12/12 | 0.000 | win |

With the gate (post hoc, the new default's numbers read off the same runs):
d10/q4 becomes +0.004 [+0.001, +0.008] 10/12, p 0.025, p_holm 0.100 over the
same 9 cells — still unresolved, but no longer a loss; the other 8 cells are
identical.

## q = 1 (supplement, descriptive; its 3 cells were not a pre-declared family)

| cell | pool best | headline Δ [CI95] wins | gated Δ |
|---|---|---|---|
| d2/q1 | PyBOBYQA (sequential) 0.430 | −0.203 [−0.231, −0.175] 0/12 | the same |
| d5/q1 | PyBOBYQA (sequential) 0.140 | −0.015 [−0.032, +0.002] 3/12 | the same |
| d10/q1 | Optuna_CmaEs 0.082 | +0.004 [−0.002, +0.010] 8/12 | +0.005 [+0.002, +0.008] 11/12 |

## Trust-region candidates (descriptive; `tables.md`)

Not pre-declared for a decision.  With all families, the three TRQ specs lead
the headline spec 11–12/12 in every cell but d10/q64, and the pool's best in
every cell but `Blocks_warm_CMAES_JSO_TRQ` at d2/q1 (+0.001, 5/12).  Without
the ellipsoid family they still lead at d ≤ 5 in most cells, but **at d = 10
all three trail the headline spec in all four q cells** (e.g. d10/q4 r05
−0.021 [−0.026, −0.015] 0/12): at d = 10 the headline spec and the pool's
best score 0 on the ellipsoid, the TRQ specs 0.04–0.85, and that one family
carries their lead (ackley is most of the ex-ellipsoid gap).
