# Expensive-track measurement

Commit e86eee3; run 36418128133; 540 unit(s); missing 0, failed 0, unreadable files 0.
Virtual clock: async policy, lognormal durations (sigma 0.5), common random numbers per cell (the i-th dispatch takes the same time for every strategy); `aocc_time` over the horizon budget/q mean durations.

**How to read this.**
- Pre-declared: the headline spec is `Blocks_warm_CMAES_JSO`; the other panobbgo specs are secondary.  The headline metric is AOCC at q = 1 and `aocc_time` at q > 1; tables are sorted by it.
- Δ = panobbgo spec − reference: the mean over seeds of the per-seed instance-mean delta, t-CI95 over seeds, wins/seeds.  A crashed run scores 0 on both metrics; a run cut by a wall-clock deadline (none is set in this workflow) would keep its scores up to the cut; an `EndedEarly` run is scored on its short trace and is not an error.
- Every mean and Δ of a cell is taken on its *common runs*: the (seed, instance) keys present for the headline spec and every pool member (each pair on its intersection with them).  `n` counts them against the plan; a cell below the plan is flagged.  A missing unit shrinks n, it does not exclude a baseline.
- *Pool best*: the best external of a pool fixed per (preset, dim, budget) — the externals present, run in every q cell with no crashed run — so the reference does not change with q because a baseline is missing at some q.  SMAC (q = 1 only) and baselines left out of some cells are reference rows.
- The pool's best is a selected maximum over several baselines, which favours the baseline side.
- Multiplicity: many cells, specs and baselines are compared.  Only the headline set (`Blocks_warm_CMAES_JSO` vs the pool's best, per cell, on the headline metric) is Holm-adjusted (p_holm); every other CI is unadjusted and descriptive.  With 5 seeds, wins/5 alone cannot be significant (5/5 has p = 0.0625 two-sided in a sign test).
- The CIs are over optimizer seeds and conditional on the fixed instances: they do not generalize over problem instances beyond the 3 per family.
- `n/planned` counts seeds against the plan; `!` marks a strategy missing some (seed, instance) run.
- *Best other panobbgo*: the best secondary panobbgo spec of the cell (the opt-in candidates of the `trq` group too, when its units are aggregated here), with its Δ vs the pool best.  The table without the ellipsoid family and the table of the secondary specs against `Blocks_warm_CMAES_JSO` are descriptive: read them next to the headline, not instead of it.

FP environments (per shard): 80ee2a0090c4: 214 shard(s).  Comparisons across jobs are ordinary samples; the environment is reproducibility metadata.

## Headline

| cell | metric | n (runs/plan) | pool best | Blocks_warm_CMAES_JSO | Δ [CI95] wins | p | p_holm | best other panobbgo | errors pb/ext | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| free/d2/b100/q4 | aocc_time | 180/180 | BoTorch_qLogEI 0.196 (12/12) | 0.214 (12/12) | +0.018 [+0.006,+0.030] 10/12 | 0.008 | 0.039 | RoundRobin_TRQ_r05 0.497 +0.301 [+0.267,+0.335] 12/12 | 0/0 |  |
| free/d2/b100/q16 | aocc_time | 180/180 | BoTorch_qLogEI 0.178 (12/12) | 0.164 (12/12) | -0.014 [-0.020,-0.008] 1/12 | 0.000 | 0.002 | RoundRobin_TRQ_r05 0.382 +0.204 [+0.193,+0.216] 12/12 | 0/0 |  |
| free/d2/b100/q64 | aocc_time | 180/180 | BoTorch_qLogEI 0.113 (12/12) | 0.114 (12/12) | +0.001 [-0.005,+0.008] 6/12 | 0.696 | 0.845 | RoundRobin_TRQ_r05 0.245 +0.131 [+0.122,+0.140] 12/12 | 0/0 |  |
| free/d5/b100/q4 | aocc_time | 180/180 | TuRBO1 0.114 (12/12) | 0.121 (12/12) | +0.007 [+0.000,+0.013] 7/12 | 0.039 | 0.116 | RoundRobin_TRQ_r05 0.312 +0.198 [+0.179,+0.216] 12/12 | 0/0 |  |
| free/d5/b100/q16 | aocc_time | 180/180 | BoTorch_qLogEI 0.075 (12/12) | 0.101 (12/12) | +0.026 [+0.023,+0.029] 12/12 | 0.000 | 0.000 | RoundRobin_TRQ_r05 0.270 +0.195 [+0.182,+0.207] 12/12 | 0/0 |  |
| free/d5/b100/q64 | aocc_time | 180/180 | BoTorch_qLogEI 0.061 (12/12) | 0.062 (12/12) | +0.001 [-0.002,+0.004] 9/12 | 0.422 | 0.845 | RoundRobin_TRQ_r05 0.224 +0.163 [+0.156,+0.170] 12/12 | 0/0 |  |
| free/d10/b100/q4 | aocc_time | 180/180 | TuRBO1 0.088 (12/12) | 0.081 (12/12) | -0.007 [-0.013,-0.001] 2/12 | 0.020 | 0.080 | Blocks_warm_CMAES_JSO_TRQ 0.195 +0.107 [+0.082,+0.132] 12/12 | 0/0 |  |
| free/d10/b100/q16 | aocc_time | 180/180 | TuRBO1 0.060 (12/12) | 0.069 (12/12) | +0.009 [+0.005,+0.013] 11/12 | 0.001 | 0.004 | RoundRobin_TRQ_r05 0.123 +0.063 [+0.036,+0.090] 11/12 | 0/0 |  |
| free/d10/b100/q64 | aocc_time | 180/180 | TuRBO1 0.028 (12/12) | 0.046 (12/12) | +0.018 [+0.017,+0.019] 12/12 | 0.000 | 0.000 | RoundRobin_TRQ_r05 0.056 +0.029 [+0.010,+0.048] 12/12 | 0/0 |  |

## Without the ellipsoid family (descriptive)

The headline table on the common runs minus the ellipsoid instances: the one exactly quadratic family, which a quadratic model solves exactly and which can decide a family mean on its own (DISCOVERY §66).  The pool is the cell's; its best is re-selected on these runs.  **Descriptive**: unadjusted, not part of the Holm family above, which stays over all families.

| cell | metric | n (runs) | pool best | Blocks_warm_CMAES_JSO | Δ [CI95] wins | best other panobbgo |
|---|---|---|---|---|---|---|
| free/d2/b100/q4 | aocc_time | 144 | BoTorch_qLogEI 0.215 | 0.236 | +0.021 [+0.009,+0.034] 10/12 | RoundRobin_TRQ_r05 0.388 +0.173 [+0.134,+0.211] 12/12 |
| free/d2/b100/q16 | aocc_time | 144 | BoTorch_qLogEI 0.193 | 0.186 | -0.007 [-0.014,-0.000] 3/12 | RoundRobin_TRQ_r05 0.259 +0.067 [+0.053,+0.080] 12/12 |
| free/d2/b100/q64 | aocc_time | 144 | BoTorch_qLogEI 0.133 | 0.137 | +0.004 [-0.005,+0.013] 7/12 | RoundRobin_TRQ_r05 0.155 +0.023 [+0.014,+0.031] 12/12 |
| free/d5/b100/q4 | aocc_time | 144 | TuRBO1 0.140 | 0.148 | +0.008 [+0.001,+0.015] 9/12 | RoundRobin_TRQ_r05 0.177 +0.037 [+0.015,+0.060] 9/12 |
| free/d5/b100/q16 | aocc_time | 144 | BoTorch_qLogEI 0.094 | 0.126 | +0.032 [+0.027,+0.036] 12/12 | RoundRobin_TRQ_r05 0.133 +0.039 [+0.029,+0.048] 12/12 |
| free/d5/b100/q64 | aocc_time | 144 | BoTorch_qLogEI 0.076 | 0.077 | +0.001 [-0.002,+0.005] 9/12 | RoundRobin_TRQ_r05 0.099 +0.023 [+0.017,+0.028] 12/12 |
| free/d10/b100/q4 | aocc_time | 144 | TuRBO1 0.110 | 0.101 | -0.009 [-0.016,-0.002] 2/12 | RegimeGate_oracle 0.115 +0.005 [+0.001,+0.010] 10/12 |
| free/d10/b100/q16 | aocc_time | 144 | TuRBO1 0.075 | 0.086 | +0.011 [+0.006,+0.016] 11/12 | Blocks_warm_CMAES_JSO_dimbudget 0.086 +0.011 [+0.006,+0.016] 11/12 |
| free/d10/b100/q64 | aocc_time | 144 | TuRBO1 0.035 | 0.057 | +0.022 [+0.021,+0.024] 12/12 | Blocks_warm_CMAES_JSO_dimbudget 0.057 +0.022 [+0.021,+0.024] 12/12 |

## Secondary panobbgo specs − Blocks_warm_CMAES_JSO (descriptive)

Every other panobbgo spec (the opt-in candidates of the `trq` group too) against `Blocks_warm_CMAES_JSO`, paired over seeds on the cell's common runs, on the headline metric and on AOCC (at q = 1 they are the same).  This is where an opt-in candidate is judged against the spec it would replace or join.  *equal*: pairs with exactly equal, nonzero headline-metric values (ties at 0 are shown apart, `+k at 0`); it means something only for a variant sharing the headline's `seed_name` (`Blocks_warm_CMAES_JSO_dimbudget`, `RegimeGate_oracle`), which is identical where its option does not act — on the same FP class: `(FP)` marks pairs across two FP classes, where equal runs may differ in the last bits.  **Descriptive**: unadjusted, not part of the Holm family.  The `RoundRobin_TRQ` specs are nearly seed-invariant at q = 1 (box-centre start), so their q = 1 CIs carry little.

| cell | spec | group | n (pairs) | metric | Δ metric [CI95] wins | Δ AOCC [CI95] wins | equal |
|---|---|---|---|---|---|---|---|
| free/d2/b100/q4 | RegimeGate_oracle | core | 180 | aocc_time | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 180/180 |
| free/d2/b100/q4 | RoundRobin_CMAES | core | 180 | aocc_time | -0.034 [-0.043,-0.025] 0/12 | -0.034 [-0.044,-0.025] 0/12 | 0/180 |
| free/d2/b100/q4 | RoundRobin_Random | core | 180 | aocc_time | -0.091 [-0.101,-0.081] 0/12 | -0.092 [-0.101,-0.082] 0/12 | 0/180 |
| free/d2/b100/q4 | Blocks_warm_CMAES_JSO_TRQ | trq | 180 | aocc_time | +0.172 [+0.158,+0.186] 12/12 | +0.173 [+0.158,+0.188] 12/12 | 0/180 |
| free/d2/b100/q4 | Blocks_warm_CMAES_JSO_dimbudget | trq | 180 | aocc_time | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 180/180 |
| free/d2/b100/q4 | RoundRobin_COBYQA | trq | 180 | aocc_time | +0.050 [+0.042,+0.058] 12/12 | +0.241 [+0.233,+0.249] 12/12 | 0/180 |
| free/d2/b100/q4 | RoundRobin_TRQ | trq | 180 | aocc_time | +0.239 [+0.219,+0.258] 12/12 | +0.241 [+0.223,+0.260] 12/12 | 0/180 |
| free/d2/b100/q4 | RoundRobin_TRQ_r05 | trq | 180 | aocc_time | +0.283 [+0.255,+0.311] 12/12 | +0.287 [+0.259,+0.315] 12/12 | 0/180 |
| free/d2/b100/q16 | RegimeGate_oracle | core | 180 | aocc_time | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 180/180 |
| free/d2/b100/q16 | RoundRobin_CMAES | core | 180 | aocc_time | -0.022 [-0.028,-0.016] 0/12 | -0.018 [-0.025,-0.012] 0/12 | 0/180 |
| free/d2/b100/q16 | RoundRobin_Random | core | 180 | aocc_time | -0.047 [-0.055,-0.040] 0/12 | -0.051 [-0.059,-0.043] 0/12 | 0/180 |
| free/d2/b100/q16 | Blocks_warm_CMAES_JSO_TRQ | trq | 180 | aocc_time | +0.166 [+0.148,+0.183] 12/12 | +0.172 [+0.153,+0.192] 12/12 | 0/180 |
| free/d2/b100/q16 | Blocks_warm_CMAES_JSO_dimbudget | trq | 180 | aocc_time | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 180/180 |
| free/d2/b100/q16 | RoundRobin_COBYQA | trq | 180 | aocc_time | -0.081 [-0.088,-0.074] 0/12 | +0.285 [+0.279,+0.291] 12/12 | 0/180 |
| free/d2/b100/q16 | RoundRobin_TRQ | trq | 180 | aocc_time | +0.176 [+0.163,+0.188] 12/12 | +0.181 [+0.168,+0.194] 12/12 | 0/180 |
| free/d2/b100/q16 | RoundRobin_TRQ_r05 | trq | 180 | aocc_time | +0.218 [+0.208,+0.229] 12/12 | +0.226 [+0.214,+0.239] 12/12 | 0/180 |
| free/d2/b100/q64 | RegimeGate_oracle | core | 180 | aocc_time | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 177/180 (+3 at 0) |
| free/d2/b100/q64 | RoundRobin_CMAES | core | 180 | aocc_time | -0.033 [-0.039,-0.028] 0/12 | +0.002 [-0.004,+0.008] 8/12 | 0/180 (+1 at 0) |
| free/d2/b100/q64 | RoundRobin_Random | core | 180 | aocc_time | -0.026 [-0.031,-0.022] 0/12 | -0.029 [-0.035,-0.024] 0/12 | 0/180 (+2 at 0) |
| free/d2/b100/q64 | Blocks_warm_CMAES_JSO_TRQ | trq | 180 | aocc_time | +0.047 [+0.036,+0.058] 12/12 | +0.077 [+0.065,+0.089] 12/12 | 0/180 |
| free/d2/b100/q64 | Blocks_warm_CMAES_JSO_dimbudget | trq | 180 | aocc_time | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 177/180 (+3 at 0) |
| free/d2/b100/q64 | RoundRobin_COBYQA | trq | 180 | aocc_time | -0.084 [-0.088,-0.080] 0/12 | +0.314 [+0.311,+0.317] 12/12 | 0/180 (+3 at 0) |
| free/d2/b100/q64 | RoundRobin_TRQ | trq | 180 | aocc_time | +0.078 [+0.065,+0.092] 12/12 | +0.088 [+0.075,+0.101] 12/12 | 0/180 |
| free/d2/b100/q64 | RoundRobin_TRQ_r05 | trq | 180 | aocc_time | +0.130 [+0.122,+0.139] 12/12 | +0.150 [+0.142,+0.157] 12/12 | 0/180 |
| free/d5/b100/q4 | RegimeGate_oracle | core | 180 | aocc_time | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 168/180 (+12 at 0) |
| free/d5/b100/q4 | RoundRobin_CMAES | core | 180 | aocc_time | -0.011 [-0.019,-0.003] 2/12 | -0.011 [-0.018,-0.003] 2/12 | 0/180 (+4 at 0) |
| free/d5/b100/q4 | RoundRobin_Random | core | 180 | aocc_time | -0.077 [-0.082,-0.071] 0/12 | -0.077 [-0.083,-0.071] 0/12 | 0/180 (+11 at 0) |
| free/d5/b100/q4 | Blocks_warm_CMAES_JSO_TRQ | trq | 180 | aocc_time | +0.167 [+0.152,+0.182] 12/12 | +0.167 [+0.153,+0.182] 12/12 | 0/180 |
| free/d5/b100/q4 | Blocks_warm_CMAES_JSO_dimbudget | trq | 180 | aocc_time | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 168/180 (+12 at 0) |
| free/d5/b100/q4 | RoundRobin_COBYQA | trq | 180 | aocc_time | +0.017 [+0.012,+0.022] 12/12 | +0.157 [+0.152,+0.162] 12/12 | 0/180 (+4 at 0) |
| free/d5/b100/q4 | RoundRobin_TRQ | trq | 180 | aocc_time | +0.189 [+0.176,+0.203] 12/12 | +0.190 [+0.177,+0.204] 12/12 | 0/180 |
| free/d5/b100/q4 | RoundRobin_TRQ_r05 | trq | 180 | aocc_time | +0.191 [+0.176,+0.205] 12/12 | +0.191 [+0.176,+0.207] 12/12 | 0/180 |
| free/d5/b100/q16 | RegimeGate_oracle | core | 180 | aocc_time | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 152/180 (+28 at 0) |
| free/d5/b100/q16 | RoundRobin_CMAES | core | 180 | aocc_time | -0.025 [-0.028,-0.022] 0/12 | -0.023 [-0.027,-0.020] 0/12 | 0/180 (+24 at 0) |
| free/d5/b100/q16 | RoundRobin_Random | core | 180 | aocc_time | -0.060 [-0.063,-0.057] 0/12 | -0.063 [-0.067,-0.060] 0/12 | 0/180 (+27 at 0) |
| free/d5/b100/q16 | Blocks_warm_CMAES_JSO_TRQ | trq | 180 | aocc_time | +0.125 [+0.113,+0.137] 12/12 | +0.127 [+0.116,+0.138] 12/12 | 0/180 |
| free/d5/b100/q16 | Blocks_warm_CMAES_JSO_dimbudget | trq | 180 | aocc_time | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 152/180 (+28 at 0) |
| free/d5/b100/q16 | RoundRobin_COBYQA | trq | 180 | aocc_time | -0.061 [-0.064,-0.058] 0/12 | +0.173 [+0.170,+0.176] 12/12 | 0/180 (+27 at 0) |
| free/d5/b100/q16 | RoundRobin_TRQ | trq | 180 | aocc_time | +0.145 [+0.133,+0.157] 12/12 | +0.145 [+0.133,+0.157] 12/12 | 0/180 |
| free/d5/b100/q16 | RoundRobin_TRQ_r05 | trq | 180 | aocc_time | +0.168 [+0.154,+0.183] 12/12 | +0.169 [+0.155,+0.183] 12/12 | 0/180 |
| free/d5/b100/q64 | RegimeGate_oracle | core | 180 | aocc_time | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 146/180 (+34 at 0) |
| free/d5/b100/q64 | RoundRobin_CMAES | core | 180 | aocc_time | -0.018 [-0.020,-0.016] 0/12 | -0.020 [-0.023,-0.018] 0/12 | 0/180 (+34 at 0) |
| free/d5/b100/q64 | RoundRobin_Random | core | 180 | aocc_time | -0.027 [-0.030,-0.025] 0/12 | -0.034 [-0.037,-0.032] 0/12 | 0/180 (+34 at 0) |
| free/d5/b100/q64 | Blocks_warm_CMAES_JSO_TRQ | trq | 180 | aocc_time | +0.080 [+0.061,+0.099] 12/12 | +0.101 [+0.082,+0.121] 12/12 | 0/180 (+3 at 0) |
| free/d5/b100/q64 | Blocks_warm_CMAES_JSO_dimbudget | trq | 180 | aocc_time | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 146/180 (+34 at 0) |
| free/d5/b100/q64 | RoundRobin_COBYQA | trq | 180 | aocc_time | -0.038 [-0.040,-0.036] 0/12 | +0.207 [+0.205,+0.209] 12/12 | 0/180 (+34 at 0) |
| free/d5/b100/q64 | RoundRobin_TRQ | trq | 180 | aocc_time | +0.105 [+0.086,+0.123] 12/12 | +0.107 [+0.087,+0.127] 12/12 | 0/180 (+2 at 0) |
| free/d5/b100/q64 | RoundRobin_TRQ_r05 | trq | 180 | aocc_time | +0.162 [+0.156,+0.168] 12/12 | +0.167 [+0.161,+0.173] 12/12 | 0/180 |
| free/d10/b100/q4 | RegimeGate_oracle | core | 180 | aocc_time | +0.011 [+0.006,+0.016] 11/12 | +0.011 [+0.006,+0.017] 11/12 | 0/180 (+35 at 0) |
| free/d10/b100/q4 | RoundRobin_CMAES | core | 180 | aocc_time | +0.010 [+0.005,+0.016] 11/12 | +0.011 [+0.005,+0.016] 11/12 | 0/180 (+35 at 0) |
| free/d10/b100/q4 | RoundRobin_Random | core | 180 | aocc_time | -0.054 [-0.058,-0.049] 0/12 | -0.054 [-0.059,-0.049] 0/12 | 0/180 (+37 at 0) |
| free/d10/b100/q4 | Blocks_warm_CMAES_JSO_TRQ | trq | 180 | aocc_time | +0.114 [+0.087,+0.141] 12/12 | +0.114 [+0.086,+0.141] 12/12 | 0/180 (+6 at 0) |
| free/d10/b100/q4 | Blocks_warm_CMAES_JSO_dimbudget | trq | 180 | aocc_time | +0.011 [+0.006,+0.016] 11/12 | +0.011 [+0.006,+0.017] 11/12 | 0/180 (+35 at 0) |
| free/d10/b100/q4 | RoundRobin_COBYQA | trq | 180 | aocc_time | -0.001 [-0.005,+0.004] 5/12 | +0.053 [+0.048,+0.058] 12/12 | 0/180 (+36 at 0) |
| free/d10/b100/q4 | RoundRobin_TRQ | trq | 180 | aocc_time | +0.071 [+0.054,+0.088] 12/12 | +0.071 [+0.054,+0.088] 12/12 | 0/180 (+3 at 0) |
| free/d10/b100/q4 | RoundRobin_TRQ_r05 | trq | 180 | aocc_time | +0.073 [+0.052,+0.094] 12/12 | +0.073 [+0.051,+0.094] 12/12 | 0/180 (+5 at 0) |
| free/d10/b100/q16 | RegimeGate_oracle | core | 180 | aocc_time | -0.000 [-0.003,+0.003] 4/12 | -0.001 [-0.004,+0.003] 4/12 | 0/180 (+36 at 0) |
| free/d10/b100/q16 | RoundRobin_CMAES | core | 180 | aocc_time | -0.003 [-0.006,+0.000] 5/12 | -0.000 [-0.004,+0.003] 5/12 | 0/180 (+36 at 0) |
| free/d10/b100/q16 | RoundRobin_Random | core | 180 | aocc_time | -0.042 [-0.047,-0.038] 0/12 | -0.045 [-0.050,-0.041] 0/12 | 0/180 (+36 at 0) |
| free/d10/b100/q16 | Blocks_warm_CMAES_JSO_TRQ | trq | 180 | aocc_time | +0.049 [+0.029,+0.069] 11/12 | +0.050 [+0.029,+0.071] 11/12 | 0/180 (+14 at 0) |
| free/d10/b100/q16 | Blocks_warm_CMAES_JSO_dimbudget | trq | 180 | aocc_time | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 144/180 (+36 at 0) |
| free/d10/b100/q16 | RoundRobin_COBYQA | trq | 180 | aocc_time | -0.033 [-0.036,-0.029] 0/12 | +0.062 [+0.058,+0.066] 12/12 | 0/180 (+36 at 0) |
| free/d10/b100/q16 | RoundRobin_TRQ | trq | 180 | aocc_time | +0.053 [+0.031,+0.076] 11/12 | +0.051 [+0.029,+0.074] 11/12 | 0/180 (+8 at 0) |
| free/d10/b100/q16 | RoundRobin_TRQ_r05 | trq | 180 | aocc_time | +0.054 [+0.029,+0.079] 11/12 | +0.052 [+0.027,+0.077] 11/12 | 0/180 (+9 at 0) |
| free/d10/b100/q64 | RegimeGate_oracle | core | 180 | aocc_time | -0.002 [-0.004,-0.000] 2/12 | -0.008 [-0.010,-0.006] 0/12 | 0/180 (+36 at 0) |
| free/d10/b100/q64 | RoundRobin_CMAES | core | 180 | aocc_time | -0.017 [-0.018,-0.015] 0/12 | -0.024 [-0.026,-0.022] 0/12 | 0/180 (+36 at 0) |
| free/d10/b100/q64 | RoundRobin_Random | core | 180 | aocc_time | -0.021 [-0.022,-0.020] 0/12 | -0.029 [-0.030,-0.027] 0/12 | 0/180 (+36 at 0) |
| free/d10/b100/q64 | Blocks_warm_CMAES_JSO_TRQ | trq | 180 | aocc_time | +0.007 [-0.003,+0.018] 5/12 | +0.016 [+0.003,+0.029] 9/12 | 0/180 (+21 at 0) |
| free/d10/b100/q64 | Blocks_warm_CMAES_JSO_dimbudget | trq | 180 | aocc_time | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 144/180 (+36 at 0) |
| free/d10/b100/q64 | RoundRobin_COBYQA | trq | 180 | aocc_time | -0.025 [-0.026,-0.024] 0/12 | +0.080 [+0.078,+0.081] 12/12 | 0/180 (+36 at 0) |
| free/d10/b100/q64 | RoundRobin_TRQ | trq | 180 | aocc_time | -0.008 [-0.016,+0.001] 3/12 | -0.015 [-0.024,-0.006] 3/12 | 0/180 (+31 at 0) |
| free/d10/b100/q64 | RoundRobin_TRQ_r05 | trq | 180 | aocc_time | +0.011 [-0.009,+0.030] 6/12 | +0.004 [-0.016,+0.024] 6/12 | 0/180 (+27 at 0) |

## Per family: Blocks_warm_CMAES_JSO − pool best (headline metric)

Δ per family (paired over seeds on that family's instances); the roadmap claim is *never much worse on any class*, so read the minimum of each row.

| cell | ackley | ellipsoid | rastrigin | rosenbrock | sharp_ridge |
|---|---|---|---|---|---|
| free/d2/b100/q4 | +0.061 [+0.031,+0.090] 10/12 | +0.006 [-0.021,+0.033] 6/12 | -0.004 [-0.028,+0.021] 7/12 | +0.032 [-0.004,+0.067] 9/12 | -0.004 [-0.028,+0.019] 5/12 |
| free/d2/b100/q16 | +0.001 [-0.011,+0.013] 7/12 | -0.042 [-0.058,-0.026] 0/12 | -0.014 [-0.029,+0.001] 4/12 | -0.009 [-0.029,+0.010] 3/12 | -0.006 [-0.025,+0.013] 5/12 |
| free/d2/b100/q64 | -0.005 [-0.015,+0.006] 5/12 | -0.009 [-0.025,+0.007] 6/12 | +0.006 [-0.006,+0.018] 10/12 | +0.009 [-0.013,+0.032] 6/12 | +0.004 [-0.009,+0.017] 6/12 |
| free/d5/b100/q4 | +0.025 [+0.005,+0.045] 9/12 | +0.002 [-0.005,+0.009] 7/12 | -0.006 [-0.023,+0.012] 5/12 | +0.024 [+0.014,+0.034] 12/12 | -0.011 [-0.022,-0.001] 3/12 |
| free/d5/b100/q16 | +0.057 [+0.046,+0.068] 12/12 | +0.004 [+0.001,+0.008] 7/12 | -0.014 [-0.017,-0.011] 0/12 | +0.080 [+0.070,+0.090] 12/12 | +0.003 [-0.004,+0.011] 8/12 |
| free/d5/b100/q64 | -0.012 [-0.019,-0.005] 2/12 | +0.001 [-0.000,+0.001] 2/12 | -0.010 [-0.013,-0.006] 0/12 | +0.050 [+0.040,+0.060] 12/12 | -0.024 [-0.030,-0.017] 0/12 |
| free/d10/b100/q4 | -0.038 [-0.055,-0.020] 1/12 | +0.000 [+0.000,+0.000] 0/12 | -0.016 [-0.022,-0.011] 0/12 | +0.013 [-0.006,+0.032] 7/12 | +0.005 [-0.002,+0.013] 8/12 |
| free/d10/b100/q16 | -0.002 [-0.017,+0.012] 7/12 | +0.000 [+0.000,+0.000] 0/12 | -0.000 [-0.004,+0.003] 6/12 | +0.029 [+0.024,+0.035] 12/12 | +0.018 [+0.009,+0.026] 11/12 |
| free/d10/b100/q64 | +0.036 [+0.032,+0.039] 12/12 | +0.000 [+0.000,+0.000] 0/12 | +0.006 [+0.004,+0.008] 12/12 | +0.020 [+0.016,+0.024] 12/12 | +0.027 [+0.026,+0.029] 12/12 |

### free/d2/b100/q4 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 17 strategies, present 17.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs PyBOBYQA (sequential) | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| RoundRobin_TRQ_r05 | 12/12 | 0.502 | 0.497 | +0.301 [+0.267,+0.335] 12/12 | +0.330 [+0.300,+0.361] 12/12 | +0.331 [+0.288,+0.373] 12/12 | 0 | 0.6 |
| RoundRobin_TRQ | 12/12 | 0.457 | 0.452 | +0.257 [+0.236,+0.278] 12/12 | +0.286 [+0.267,+0.304] 12/12 | +0.286 [+0.262,+0.311] 12/12 | 0 | 0.8 |
| Blocks_warm_CMAES_JSO_TRQ | 12/12 | 0.389 | 0.386 | +0.190 [+0.172,+0.208] 12/12 | +0.219 [+0.206,+0.233] 12/12 | +0.220 [+0.195,+0.244] 12/12 | 0 | 0.4 |
| RoundRobin_COBYQA | 12/12 | 0.457 | 0.264 | +0.068 [+0.061,+0.075] 12/12 | +0.097 [+0.092,+0.102] 12/12 | +0.098 [+0.077,+0.118] 12/12 | 0 | 2.1 |
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.216 | 0.214 | +0.018 [+0.006,+0.030] 10/12 | +0.047 [+0.040,+0.055] 12/12 | +0.048 [+0.023,+0.073] 11/12 | 0 | 0.2 |
| RegimeGate_oracle | 12/12 | 0.216 | 0.214 | +0.018 [+0.006,+0.030] 10/12 | +0.047 [+0.040,+0.055] 12/12 | +0.048 [+0.023,+0.073] 11/12 | 0 | 0.2 |
| Blocks_warm_CMAES_JSO_dimbudget | 12/12 | 0.216 | 0.214 | +0.018 [+0.006,+0.030] 10/12 | +0.047 [+0.040,+0.055] 12/12 | +0.048 [+0.023,+0.073] 11/12 | 0 | 0.2 |
| BoTorch_qLogEI (pool) | 12/12 | 0.198 | 0.196 | – | – | – | 0 | 238.6 |
| RoundRobin_CMAES | 12/12 | 0.182 | 0.180 | -0.016 [-0.027,-0.005] 3/12 | +0.013 [+0.004,+0.022] 11/12 | +0.014 [-0.011,+0.039] 7/12 | 0 | 0.1 |
| Optuna_TPE (pool) | 12/12 | 0.168 | 0.167 | – | – | – | 0 | 0.5 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.430 | 0.166 | – | – | – | 0 | 0.3 |
| TuRBO1 (pool) | 12/12 | 0.194 | 0.162 | – | – | – | 0 | 27.7 |
| NGOpt (pool) | 12/12 | 0.163 | 0.160 | – | – | – | 0 | 2.0 |
| Optuna_CmaEs (pool) | 12/12 | 0.150 | 0.148 | – | – | – | 0 | 0.2 |
| pycma_IPOP (pool) | 12/12 | 0.170 | 0.138 | – | – | – | 0 | 0.1 |
| pycma_BIPOP (pool) | 12/12 | 0.168 | 0.136 | – | – | – | 0 | 0.1 |
| RoundRobin_Random | 12/12 | 0.124 | 0.123 | -0.073 [-0.082,-0.064] 0/12 | -0.044 [-0.049,-0.039] 0/12 | -0.043 [-0.065,-0.022] 2/12 | 0 | 0.1 |

### free/d2/b100/q16 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 17 strategies, present 17.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs TuRBO1 | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| RoundRobin_TRQ_r05 | 12/12 | 0.398 | 0.382 | +0.204 [+0.193,+0.216] 12/12 | +0.235 [+0.224,+0.247] 12/12 | +0.254 [+0.239,+0.268] 12/12 | 0 | 0.9 |
| RoundRobin_TRQ | 12/12 | 0.353 | 0.339 | +0.162 [+0.150,+0.173] 12/12 | +0.193 [+0.181,+0.205] 12/12 | +0.211 [+0.196,+0.226] 12/12 | 0 | 1.0 |
| Blocks_warm_CMAES_JSO_TRQ | 12/12 | 0.344 | 0.329 | +0.152 [+0.135,+0.169] 12/12 | +0.183 [+0.165,+0.201] 12/12 | +0.201 [+0.181,+0.220] 12/12 | 0 | 0.4 |
| BoTorch_qLogEI (pool) | 12/12 | 0.185 | 0.178 | – | – | – | 0 | 347.6 |
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.172 | 0.164 | -0.014 [-0.020,-0.008] 1/12 | +0.017 [+0.011,+0.023] 12/12 | +0.035 [+0.025,+0.045] 12/12 | 0 | 0.1 |
| RegimeGate_oracle | 12/12 | 0.172 | 0.164 | -0.014 [-0.020,-0.008] 1/12 | +0.017 [+0.011,+0.023] 12/12 | +0.035 [+0.025,+0.045] 12/12 | 0 | 0.1 |
| Blocks_warm_CMAES_JSO_dimbudget | 12/12 | 0.172 | 0.164 | -0.014 [-0.020,-0.008] 1/12 | +0.017 [+0.011,+0.023] 12/12 | +0.035 [+0.025,+0.045] 12/12 | 0 | 0.2 |
| Optuna_TPE (pool) | 12/12 | 0.153 | 0.147 | – | – | – | 0 | 0.5 |
| RoundRobin_CMAES | 12/12 | 0.154 | 0.142 | -0.036 [-0.041,-0.031] 0/12 | -0.005 [-0.010,+0.000] 2/12 | +0.013 [+0.004,+0.023] 10/12 | 0 | 0.1 |
| TuRBO1 (pool) | 12/12 | 0.190 | 0.129 | – | – | – | 0 | 9.1 |
| NGOpt (pool) | 12/12 | 0.130 | 0.124 | – | – | – | 0 | 1.5 |
| Optuna_CmaEs (pool) | 12/12 | 0.124 | 0.119 | – | – | – | 0 | 0.2 |
| RoundRobin_Random | 12/12 | 0.121 | 0.116 | -0.062 [-0.066,-0.057] 0/12 | -0.030 [-0.035,-0.025] 0/12 | -0.012 [-0.022,-0.003] 3/12 | 0 | 0.1 |
| pycma_IPOP (pool) | 12/12 | 0.134 | 0.094 | – | – | – | 0 | 0.1 |
| pycma_BIPOP (pool) | 12/12 | 0.133 | 0.093 | – | – | – | 0 | 0.1 |
| RoundRobin_COBYQA | 12/12 | 0.457 | 0.083 | -0.095 [-0.103,-0.086] 0/12 | -0.064 [-0.072,-0.055] 0/12 | -0.046 [-0.056,-0.035] 0/12 | 0 | 2.1 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.430 | 0.050 | – | – | – | 0 | 0.3 |

### free/d2/b100/q64 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 17 strategies, present 17.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs Optuna_CmaEs | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| RoundRobin_TRQ_r05 | 12/12 | 0.293 | 0.245 | +0.131 [+0.122,+0.140] 12/12 | +0.136 [+0.131,+0.142] 12/12 | +0.148 [+0.139,+0.156] 12/12 | 0 | 1.3 |
| RoundRobin_TRQ | 12/12 | 0.231 | 0.193 | +0.080 [+0.064,+0.095] 12/12 | +0.085 [+0.072,+0.098] 12/12 | +0.096 [+0.082,+0.110] 12/12 | 0 | 1.4 |
| Blocks_warm_CMAES_JSO_TRQ | 12/12 | 0.220 | 0.162 | +0.048 [+0.037,+0.060] 12/12 | +0.053 [+0.043,+0.064] 12/12 | +0.065 [+0.052,+0.077] 12/12 | 0 | 0.3 |
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.143 | 0.114 | +0.001 [-0.005,+0.008] 6/12 | +0.006 [+0.001,+0.011] 11/12 | +0.018 [+0.012,+0.024] 12/12 | 0 | 0.1 |
| RegimeGate_oracle | 12/12 | 0.143 | 0.114 | +0.001 [-0.005,+0.008] 6/12 | +0.006 [+0.001,+0.011] 11/12 | +0.018 [+0.012,+0.024] 12/12 | 0 | 0.1 |
| Blocks_warm_CMAES_JSO_dimbudget | 12/12 | 0.143 | 0.114 | +0.001 [-0.005,+0.008] 6/12 | +0.006 [+0.001,+0.011] 11/12 | +0.018 [+0.012,+0.024] 12/12 | 0 | 0.1 |
| BoTorch_qLogEI (pool) | 12/12 | 0.135 | 0.113 | – | – | – | 0 | 763.9 |
| Optuna_TPE (pool) | 12/12 | 0.128 | 0.108 | – | – | – | 0 | 0.4 |
| Optuna_CmaEs (pool) | 12/12 | 0.112 | 0.097 | – | – | – | 0 | 0.2 |
| TuRBO1 (pool) | 12/12 | 0.158 | 0.093 | – | – | – | 0 | 2.1 |
| RoundRobin_Random | 12/12 | 0.114 | 0.088 | -0.025 [-0.033,-0.018] 0/12 | -0.020 [-0.025,-0.015] 0/12 | -0.009 [-0.015,-0.002] 1/12 | 0 | 0.1 |
| RoundRobin_CMAES | 12/12 | 0.145 | 0.081 | -0.032 [-0.037,-0.027] 0/12 | -0.027 [-0.033,-0.022] 0/12 | -0.016 [-0.021,-0.011] 1/12 | 0 | 0.1 |
| pycma_BIPOP (pool) | 12/12 | 0.112 | 0.077 | – | – | – | 0 | 0.0 |
| pycma_IPOP (pool) | 12/12 | 0.110 | 0.077 | – | – | – | 0 | 0.0 |
| NGOpt (pool) | 12/12 | 0.069 | 0.058 | – | – | – | 0 | 0.5 |
| RoundRobin_COBYQA | 12/12 | 0.457 | 0.031 | -0.083 [-0.088,-0.078] 0/12 | -0.078 [-0.082,-0.074] 0/12 | -0.066 [-0.071,-0.062] 0/12 | 0 | 2.1 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.430 | 0.023 | – | – | – | 0 | 0.3 |

### free/d5/b100/q4 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time TuRBO1.  Planned 17 strategies, present 17.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs TuRBO1 (pool best) | Δ aocc_time vs Optuna_CmaEs | Δ aocc_time vs Optuna_TPE | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| RoundRobin_TRQ_r05 | 12/12 | 0.313 | 0.312 | +0.198 [+0.179,+0.216] 12/12 | +0.228 [+0.212,+0.245] 12/12 | +0.229 [+0.211,+0.246] 12/12 | 0 | 1.8 |
| RoundRobin_TRQ | 12/12 | 0.312 | 0.310 | +0.196 [+0.184,+0.209] 12/12 | +0.227 [+0.216,+0.239] 12/12 | +0.227 [+0.215,+0.239] 12/12 | 0 | 2.1 |
| Blocks_warm_CMAES_JSO_TRQ | 12/12 | 0.289 | 0.288 | +0.174 [+0.162,+0.185] 12/12 | +0.205 [+0.193,+0.217] 12/12 | +0.205 [+0.192,+0.218] 12/12 | 0 | 0.8 |
| RoundRobin_COBYQA | 12/12 | 0.279 | 0.138 | +0.024 [+0.020,+0.028] 12/12 | +0.055 [+0.052,+0.057] 12/12 | +0.055 [+0.052,+0.058] 12/12 | 0 | 2.8 |
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.121 | 0.121 | +0.007 [+0.000,+0.013] 7/12 | +0.038 [+0.032,+0.043] 12/12 | +0.038 [+0.033,+0.043] 12/12 | 0 | 0.5 |
| RegimeGate_oracle | 12/12 | 0.121 | 0.121 | +0.007 [+0.000,+0.013] 7/12 | +0.038 [+0.032,+0.043] 12/12 | +0.038 [+0.033,+0.043] 12/12 | 0 | 0.5 |
| Blocks_warm_CMAES_JSO_dimbudget | 12/12 | 0.121 | 0.121 | +0.007 [+0.000,+0.013] 7/12 | +0.038 [+0.032,+0.043] 12/12 | +0.038 [+0.033,+0.043] 12/12 | 0 | 0.5 |
| TuRBO1 (pool) | 12/12 | 0.132 | 0.114 | – | – | – | 0 | 69.0 |
| RoundRobin_CMAES | 12/12 | 0.111 | 0.110 | -0.004 [-0.009,+0.001] 4/12 | +0.027 [+0.023,+0.031] 12/12 | +0.027 [+0.022,+0.032] 12/12 | 0 | 0.3 |
| Optuna_CmaEs (pool) | 12/12 | 0.084 | 0.083 | – | – | – | 0 | 0.9 |
| Optuna_TPE (pool) | 12/12 | 0.083 | 0.083 | – | – | – | 0 | 4.0 |
| BoTorch_qLogEI (pool) | 12/12 | 0.081 | 0.081 | – | – | – | 0 | 1268.7 |
| pycma_IPOP (pool) | 12/12 | 0.092 | 0.073 | – | – | – | 0 | 0.2 |
| pycma_BIPOP (pool) | 12/12 | 0.092 | 0.072 | – | – | – | 0 | 0.2 |
| NGOpt (pool) | 12/12 | 0.065 | 0.065 | – | – | – | 0 | 10.1 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.140 | 0.053 | – | – | – | 0 | 2.7 |
| RoundRobin_Random | 12/12 | 0.044 | 0.044 | -0.070 [-0.073,-0.067] 0/12 | -0.039 [-0.042,-0.036] 0/12 | -0.039 [-0.042,-0.036] 0/12 | 0 | 0.3 |

### free/d5/b100/q16 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 17 strategies, present 17.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs TuRBO1 | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| RoundRobin_TRQ_r05 | 12/12 | 0.275 | 0.270 | +0.195 [+0.182,+0.207] 12/12 | +0.202 [+0.189,+0.216] 12/12 | +0.204 [+0.190,+0.218] 12/12 | 0 | 1.8 |
| RoundRobin_TRQ | 12/12 | 0.251 | 0.247 | +0.171 [+0.160,+0.183] 12/12 | +0.179 [+0.168,+0.191] 12/12 | +0.181 [+0.171,+0.191] 12/12 | 0 | 1.9 |
| Blocks_warm_CMAES_JSO_TRQ | 12/12 | 0.232 | 0.226 | +0.151 [+0.138,+0.164] 12/12 | +0.159 [+0.146,+0.172] 12/12 | +0.160 [+0.148,+0.173] 12/12 | 0 | 0.8 |
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.106 | 0.101 | +0.026 [+0.023,+0.029] 12/12 | +0.034 [+0.030,+0.038] 12/12 | +0.036 [+0.032,+0.039] 12/12 | 0 | 0.5 |
| RegimeGate_oracle | 12/12 | 0.106 | 0.101 | +0.026 [+0.023,+0.029] 12/12 | +0.034 [+0.030,+0.038] 12/12 | +0.036 [+0.032,+0.039] 12/12 | 0 | 0.5 |
| Blocks_warm_CMAES_JSO_dimbudget | 12/12 | 0.106 | 0.101 | +0.026 [+0.023,+0.029] 12/12 | +0.034 [+0.030,+0.038] 12/12 | +0.036 [+0.032,+0.039] 12/12 | 0 | 0.3 |
| RoundRobin_CMAES | 12/12 | 0.082 | 0.076 | +0.001 [-0.000,+0.002] 9/12 | +0.009 [+0.006,+0.011] 12/12 | +0.010 [+0.007,+0.014] 12/12 | 0 | 0.3 |
| BoTorch_qLogEI (pool) | 12/12 | 0.076 | 0.075 | – | – | – | 0 | 1459.3 |
| Optuna_TPE (pool) | 12/12 | 0.069 | 0.067 | – | – | – | 0 | 4.0 |
| TuRBO1 (pool) | 12/12 | 0.111 | 0.066 | – | – | – | 0 | 25.0 |
| Optuna_CmaEs (pool) | 12/12 | 0.053 | 0.052 | – | – | – | 0 | 1.0 |
| NGOpt (pool) | 12/12 | 0.046 | 0.045 | – | – | – | 0 | 10.1 |
| RoundRobin_Random | 12/12 | 0.042 | 0.042 | -0.034 [-0.035,-0.032] 0/12 | -0.026 [-0.028,-0.023] 0/12 | -0.024 [-0.027,-0.021] 0/12 | 0 | 0.3 |
| RoundRobin_COBYQA | 12/12 | 0.279 | 0.040 | -0.035 [-0.036,-0.034] 0/12 | -0.027 [-0.030,-0.025] 0/12 | -0.026 [-0.028,-0.023] 0/12 | 0 | 2.2 |
| pycma_BIPOP (pool) | 12/12 | 0.065 | 0.039 | – | – | – | 0 | 0.2 |
| pycma_IPOP (pool) | 12/12 | 0.064 | 0.038 | – | – | – | 0 | 0.2 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.140 | 0.025 | – | – | – | 0 | 2.7 |

### free/d5/b100/q64 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 17 strategies, present 17.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs TuRBO1 | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| RoundRobin_TRQ_r05 | 12/12 | 0.239 | 0.224 | +0.163 [+0.156,+0.170] 12/12 | +0.180 [+0.175,+0.186] 12/12 | +0.185 [+0.179,+0.191] 12/12 | 0 | 2.1 |
| RoundRobin_TRQ | 12/12 | 0.179 | 0.166 | +0.106 [+0.087,+0.124] 12/12 | +0.123 [+0.105,+0.141] 12/12 | +0.128 [+0.109,+0.146] 12/12 | 0 | 2.6 |
| Blocks_warm_CMAES_JSO_TRQ | 12/12 | 0.173 | 0.142 | +0.081 [+0.064,+0.098] 12/12 | +0.098 [+0.081,+0.116] 12/12 | +0.103 [+0.085,+0.120] 12/12 | 0 | 0.7 |
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.072 | 0.062 | +0.001 [-0.002,+0.004] 9/12 | +0.019 [+0.016,+0.021] 12/12 | +0.023 [+0.021,+0.025] 12/12 | 0 | 0.3 |
| RegimeGate_oracle | 12/12 | 0.072 | 0.062 | +0.001 [-0.002,+0.004] 9/12 | +0.019 [+0.016,+0.021] 12/12 | +0.023 [+0.021,+0.025] 12/12 | 0 | 0.3 |
| Blocks_warm_CMAES_JSO_dimbudget | 12/12 | 0.072 | 0.062 | +0.001 [-0.002,+0.004] 9/12 | +0.019 [+0.016,+0.021] 12/12 | +0.023 [+0.021,+0.025] 12/12 | 0 | 0.3 |
| BoTorch_qLogEI (pool) | 12/12 | 0.064 | 0.061 | – | – | – | 0 | 3159.4 |
| RoundRobin_CMAES | 12/12 | 0.052 | 0.044 | -0.017 [-0.019,-0.014] 0/12 | +0.001 [-0.001,+0.002] 7/12 | +0.005 [+0.004,+0.006] 12/12 | 0 | 0.2 |
| Optuna_TPE (pool) | 12/12 | 0.046 | 0.043 | – | – | – | 0 | 2.7 |
| TuRBO1 (pool) | 12/12 | 0.059 | 0.039 | – | – | – | 0 | 9.2 |
| Optuna_CmaEs (pool) | 12/12 | 0.038 | 0.036 | – | – | – | 0 | 0.8 |
| RoundRobin_Random | 12/12 | 0.038 | 0.034 | -0.026 [-0.028,-0.024] 0/12 | -0.009 [-0.010,-0.007] 0/12 | -0.004 [-0.006,-0.003] 0/12 | 0 | 0.2 |
| pycma_IPOP (pool) | 12/12 | 0.038 | 0.030 | – | – | – | 0 | 0.1 |
| pycma_BIPOP (pool) | 12/12 | 0.038 | 0.030 | – | – | – | 0 | 0.1 |
| NGOpt (pool) | 12/12 | 0.026 | 0.025 | – | – | – | 0 | 2.0 |
| RoundRobin_COBYQA | 12/12 | 0.279 | 0.023 | -0.037 [-0.039,-0.035] 0/12 | -0.020 [-0.021,-0.018] 0/12 | -0.016 [-0.017,-0.014] 0/12 | 0 | 2.8 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.140 | 0.018 | – | – | – | 0 | 1.6 |

### free/d10/b100/q4 (headline: aocc_time)

Pool: NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC TuRBO1, aocc_time TuRBO1.  Planned 16 strategies, present 16.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs TuRBO1 (pool best) | Δ aocc_time vs Optuna_CmaEs | Δ aocc_time vs pycma_BIPOP | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| Blocks_warm_CMAES_JSO_TRQ | 12/12 | 0.195 | 0.195 | +0.107 [+0.082,+0.132] 12/12 | +0.130 [+0.105,+0.155] 12/12 | +0.143 [+0.118,+0.168] 12/12 | 0 | 1.7 |
| RoundRobin_TRQ_r05 | 12/12 | 0.154 | 0.154 | +0.066 [+0.044,+0.088] 11/12 | +0.089 [+0.068,+0.110] 12/12 | +0.102 [+0.080,+0.123] 12/12 | 0 | 3.1 |
| RoundRobin_TRQ | 12/12 | 0.152 | 0.152 | +0.064 [+0.047,+0.081] 12/12 | +0.087 [+0.070,+0.104] 12/12 | +0.100 [+0.083,+0.117] 12/12 | 0 | 2.9 |
| RegimeGate_oracle | 12/12 | 0.093 | 0.092 | +0.004 [+0.001,+0.008] 10/12 | +0.027 [+0.025,+0.029] 12/12 | +0.040 [+0.037,+0.043] 12/12 | 0 | 0.6 |
| Blocks_warm_CMAES_JSO_dimbudget | 12/12 | 0.093 | 0.092 | +0.004 [+0.001,+0.008] 10/12 | +0.027 [+0.025,+0.029] 12/12 | +0.040 [+0.037,+0.043] 12/12 | 0 | 0.6 |
| RoundRobin_CMAES | 12/12 | 0.092 | 0.091 | +0.003 [+0.001,+0.006] 10/12 | +0.026 [+0.025,+0.028] 12/12 | +0.039 [+0.037,+0.041] 12/12 | 0 | 0.3 |
| TuRBO1 (pool) | 12/12 | 0.100 | 0.088 | – | – | – | 0 | 197.8 |
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.081 | 0.081 | -0.007 [-0.013,-0.001] 2/12 | +0.016 [+0.011,+0.021] 12/12 | +0.029 [+0.023,+0.034] 12/12 | 0 | 0.8 |
| RoundRobin_COBYQA | 12/12 | 0.134 | 0.080 | -0.008 [-0.010,-0.005] 0/12 | +0.015 [+0.014,+0.016] 12/12 | +0.028 [+0.026,+0.030] 12/12 | 0 | 3.3 |
| Optuna_CmaEs (pool) | 12/12 | 0.065 | 0.065 | – | – | – | 0 | 1.9 |
| pycma_BIPOP (pool) | 12/12 | 0.066 | 0.052 | – | – | – | 0 | 0.3 |
| pycma_IPOP (pool) | 12/12 | 0.066 | 0.052 | – | – | – | 0 | 0.3 |
| NGOpt (pool) | 12/12 | 0.050 | 0.050 | – | – | – | 0 | 1.8 |
| Optuna_TPE (pool) | 12/12 | 0.044 | 0.044 | – | – | – | 0 | 14.4 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.077 | 0.043 | – | – | – | 0 | 9.6 |
| RoundRobin_Random | 12/12 | 0.027 | 0.027 | -0.061 [-0.063,-0.058] 0/12 | -0.038 [-0.039,-0.036] 0/12 | -0.025 [-0.028,-0.022] 0/12 | 0 | 0.6 |

### free/d10/b100/q16 (headline: aocc_time)

Pool: NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC TuRBO1, aocc_time TuRBO1.  Planned 16 strategies, present 16.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs TuRBO1 (pool best) | Δ aocc_time vs NGOpt | Δ aocc_time vs Optuna_CmaEs | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| RoundRobin_TRQ_r05 | 12/12 | 0.124 | 0.123 | +0.063 [+0.036,+0.090] 11/12 | +0.069 [+0.043,+0.094] 11/12 | +0.085 [+0.059,+0.111] 12/12 | 0 | 4.7 |
| RoundRobin_TRQ | 12/12 | 0.123 | 0.122 | +0.062 [+0.040,+0.084] 12/12 | +0.068 [+0.044,+0.092] 12/12 | +0.084 [+0.062,+0.106] 12/12 | 0 | 4.8 |
| Blocks_warm_CMAES_JSO_TRQ | 12/12 | 0.122 | 0.118 | +0.058 [+0.039,+0.077] 12/12 | +0.064 [+0.045,+0.082] 12/12 | +0.080 [+0.060,+0.100] 12/12 | 0 | 2.1 |
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.072 | 0.069 | +0.009 [+0.005,+0.013] 11/12 | +0.015 [+0.010,+0.019] 12/12 | +0.031 [+0.027,+0.034] 12/12 | 0 | 0.9 |
| Blocks_warm_CMAES_JSO_dimbudget | 12/12 | 0.072 | 0.069 | +0.009 [+0.005,+0.013] 11/12 | +0.015 [+0.010,+0.019] 12/12 | +0.031 [+0.027,+0.034] 12/12 | 0 | 0.9 |
| RegimeGate_oracle | 12/12 | 0.071 | 0.069 | +0.009 [+0.004,+0.014] 10/12 | +0.015 [+0.010,+0.019] 12/12 | +0.031 [+0.026,+0.035] 12/12 | 0 | 0.7 |
| RoundRobin_CMAES | 12/12 | 0.072 | 0.066 | +0.006 [+0.003,+0.008] 11/12 | +0.012 [+0.008,+0.015] 12/12 | +0.028 [+0.026,+0.030] 12/12 | 0 | 0.4 |
| TuRBO1 (pool) | 12/12 | 0.090 | 0.060 | – | – | – | 0 | 73.2 |
| NGOpt (pool) | 12/12 | 0.055 | 0.054 | – | – | – | 0 | 1.8 |
| Optuna_CmaEs (pool) | 12/12 | 0.039 | 0.038 | – | – | – | 0 | 2.6 |
| RoundRobin_COBYQA | 12/12 | 0.134 | 0.036 | -0.024 [-0.026,-0.022] 0/12 | -0.018 [-0.021,-0.015] 0/12 | -0.002 [-0.003,-0.001] 1/12 | 0 | 3.9 |
| Optuna_TPE (pool) | 12/12 | 0.036 | 0.036 | – | – | – | 0 | 17.0 |
| pycma_BIPOP (pool) | 12/12 | 0.047 | 0.027 | – | – | – | 0 | 0.3 |
| pycma_IPOP (pool) | 12/12 | 0.048 | 0.027 | – | – | – | 0 | 0.3 |
| RoundRobin_Random | 12/12 | 0.027 | 0.027 | -0.033 [-0.036,-0.031] 0/12 | -0.028 [-0.030,-0.025] 0/12 | -0.011 [-0.013,-0.010] 0/12 | 0 | 0.6 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.077 | 0.021 | – | – | – | 0 | 12.1 |

### free/d10/b100/q64 (headline: aocc_time)

Pool: NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time TuRBO1.  Planned 16 strategies, present 16.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs TuRBO1 (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs Optuna_CmaEs | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| RoundRobin_TRQ_r05 | 12/12 | 0.058 | 0.056 | +0.029 [+0.010,+0.048] 12/12 | +0.029 [+0.010,+0.048] 12/12 | +0.029 [+0.010,+0.048] 12/12 | 0 | 3.7 |
| Blocks_warm_CMAES_JSO_TRQ | 12/12 | 0.071 | 0.053 | +0.025 [+0.014,+0.036] 12/12 | +0.025 [+0.015,+0.036] 12/12 | +0.026 [+0.015,+0.037] 12/12 | 0 | 1.1 |
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.054 | 0.046 | +0.018 [+0.017,+0.019] 12/12 | +0.018 [+0.017,+0.020] 12/12 | +0.019 [+0.017,+0.020] 12/12 | 0 | 0.7 |
| Blocks_warm_CMAES_JSO_dimbudget | 12/12 | 0.054 | 0.046 | +0.018 [+0.017,+0.019] 12/12 | +0.018 [+0.017,+0.020] 12/12 | +0.019 [+0.017,+0.020] 12/12 | 0 | 0.5 |
| RegimeGate_oracle | 12/12 | 0.046 | 0.044 | +0.016 [+0.015,+0.017] 12/12 | +0.016 [+0.015,+0.018] 12/12 | +0.017 [+0.015,+0.018] 12/12 | 0 | 0.5 |
| RoundRobin_TRQ | 12/12 | 0.039 | 0.038 | +0.010 [+0.001,+0.019] 10/12 | +0.011 [+0.002,+0.019] 10/12 | +0.011 [+0.002,+0.020] 12/12 | 0 | 3.7 |
| RoundRobin_CMAES | 12/12 | 0.030 | 0.029 | +0.001 [+0.001,+0.002] 12/12 | +0.002 [+0.001,+0.002] 12/12 | +0.002 [+0.001,+0.003] 11/12 | 0 | 0.4 |
| TuRBO1 (pool) | 12/12 | 0.046 | 0.028 | – | – | – | 0 | 75.2 |
| Optuna_TPE (pool) | 12/12 | 0.028 | 0.027 | – | – | – | 0 | 15.5 |
| Optuna_CmaEs (pool) | 12/12 | 0.028 | 0.027 | – | – | – | 0 | 3.4 |
| RoundRobin_Random | 12/12 | 0.025 | 0.025 | -0.003 [-0.004,-0.002] 0/12 | -0.003 [-0.003,-0.002] 0/12 | -0.002 [-0.003,-0.002] 1/12 | 0 | 0.5 |
| pycma_IPOP (pool) | 12/12 | 0.027 | 0.023 | – | – | – | 0 | 0.2 |
| pycma_BIPOP (pool) | 12/12 | 0.026 | 0.022 | – | – | – | 0 | 0.2 |
| NGOpt (pool) | 12/12 | 0.021 | 0.021 | – | – | – | 0 | 9.2 |
| RoundRobin_COBYQA | 12/12 | 0.134 | 0.021 | -0.007 [-0.008,-0.007] 0/12 | -0.007 [-0.008,-0.006] 0/12 | -0.006 [-0.007,-0.006] 0/12 | 0 | 3.0 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.077 | 0.018 | – | – | – | 0 | 10.9 |
