# Expensive-track measurement

Commit e86eee3; run 36418132989; 72 unit(s); missing 0, failed 0, unreadable files 0.
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

FP environments (per shard): 80ee2a0090c4: 6 shard(s).  Comparisons across jobs are ordinary samples; the environment is reproducibility metadata.

## Headline

| cell | metric | n (runs/plan) | pool best | Blocks_warm_CMAES_JSO | Δ [CI95] wins | p | p_holm | best other panobbgo | errors pb/ext | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| free/d2/b100/q1 | aocc | 180/180 | PyBOBYQA (sequential) 0.430 (12/12) | 0.227 (12/12) | -0.203 [-0.231,-0.175] 0/12 | 0.000 | 0.000 | RoundRobin_TRQ_r05 0.602 +0.172 [+0.148,+0.195] 12/12 | 0/0 |  |
| free/d5/b100/q1 | aocc | 180/180 | PyBOBYQA (sequential) 0.140 (12/12) | 0.125 (12/12) | -0.015 [-0.032,+0.002] 3/12 | 0.081 | 0.161 | RoundRobin_TRQ_r05 0.405 +0.265 [+0.249,+0.281] 12/12 | 0/0 |  |
| free/d10/b100/q1 | aocc | 180/180 | Optuna_CmaEs 0.082 (12/12) | 0.086 (12/12) | +0.004 [-0.002,+0.010] 8/12 | 0.192 | 0.192 | Blocks_warm_CMAES_JSO_TRQ 0.245 +0.163 [+0.151,+0.174] 12/12 | 0/0 |  |

## Without the ellipsoid family (descriptive)

The headline table on the common runs minus the ellipsoid instances: the one exactly quadratic family, which a quadratic model solves exactly and which can decide a family mean on its own (DISCOVERY §66).  The pool is the cell's; its best is re-selected on these runs.  **Descriptive**: unadjusted, not part of the Holm family above, which stays over all families.

| cell | metric | n (runs) | pool best | Blocks_warm_CMAES_JSO | Δ [CI95] wins | best other panobbgo |
|---|---|---|---|---|---|---|
| free/d2/b100/q1 | aocc | 144 | PyBOBYQA (sequential) 0.341 | 0.248 | -0.093 [-0.125,-0.061] 1/12 | RoundRobin_TRQ_r05 0.516 +0.175 [+0.149,+0.202] 12/12 |
| free/d5/b100/q1 | aocc | 144 | PyBOBYQA (sequential) 0.144 | 0.154 | +0.010 [-0.006,+0.027] 8/12 | RoundRobin_COBYQA 0.310 +0.166 [+0.151,+0.182] 12/12 |
| free/d10/b100/q1 | aocc | 144 | Optuna_CmaEs 0.103 | 0.107 | +0.005 [-0.003,+0.012] 8/12 | RoundRobin_COBYQA 0.166 +0.064 [+0.061,+0.067] 12/12 |

## Secondary panobbgo specs − Blocks_warm_CMAES_JSO (descriptive)

Every other panobbgo spec (the opt-in candidates of the `trq` group too) against `Blocks_warm_CMAES_JSO`, paired over seeds on the cell's common runs, on the headline metric and on AOCC (at q = 1 they are the same).  This is where an opt-in candidate is judged against the spec it would replace or join.  *equal*: pairs with exactly equal, nonzero headline-metric values (ties at 0 are shown apart, `+k at 0`); it means something only for a variant sharing the headline's `seed_name` (`Blocks_warm_CMAES_JSO_dimbudget`, `RegimeGate_oracle`), which is identical where its option does not act — on the same FP class: `(FP)` marks pairs across two FP classes, where equal runs may differ in the last bits.  **Descriptive**: unadjusted, not part of the Holm family.  The `RoundRobin_TRQ` specs are nearly seed-invariant at q = 1 (box-centre start), so their q = 1 CIs carry little.

| cell | spec | group | n (pairs) | metric | Δ metric [CI95] wins | Δ AOCC [CI95] wins | equal |
|---|---|---|---|---|---|---|---|
| free/d2/b100/q1 | RegimeGate_oracle | core | 180 | aocc | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 180/180 |
| free/d2/b100/q1 | RoundRobin_CMAES | core | 180 | aocc | -0.037 [-0.053,-0.022] 1/12 | -0.037 [-0.053,-0.022] 1/12 | 0/180 |
| free/d2/b100/q1 | RoundRobin_Random | core | 180 | aocc | -0.099 [-0.112,-0.086] 0/12 | -0.099 [-0.112,-0.086] 0/12 | 0/180 |
| free/d2/b100/q1 | Blocks_warm_CMAES_JSO_TRQ | trq | 180 | aocc | +0.204 [+0.180,+0.227] 12/12 | +0.204 [+0.180,+0.227] 12/12 | 0/180 |
| free/d2/b100/q1 | Blocks_warm_CMAES_JSO_dimbudget | trq | 180 | aocc | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 180/180 |
| free/d2/b100/q1 | RoundRobin_COBYQA | trq | 180 | aocc | +0.229 [+0.217,+0.242] 12/12 | +0.229 [+0.217,+0.242] 12/12 | 0/180 |
| free/d2/b100/q1 | RoundRobin_TRQ | trq | 180 | aocc | +0.318 [+0.306,+0.330] 12/12 | +0.318 [+0.306,+0.330] 12/12 | 0/180 |
| free/d2/b100/q1 | RoundRobin_TRQ_r05 | trq | 180 | aocc | +0.375 [+0.361,+0.388] 12/12 | +0.375 [+0.361,+0.388] 12/12 | 0/180 |
| free/d5/b100/q1 | RegimeGate_oracle | core | 180 | aocc | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 160/180 (+20 at 0) |
| free/d5/b100/q1 | RoundRobin_CMAES | core | 180 | aocc | -0.013 [-0.021,-0.006] 3/12 | -0.013 [-0.021,-0.006] 3/12 | 0/180 (+9 at 0) |
| free/d5/b100/q1 | RoundRobin_Random | core | 180 | aocc | -0.080 [-0.086,-0.073] 0/12 | -0.080 [-0.086,-0.073] 0/12 | 0/180 (+20 at 0) |
| free/d5/b100/q1 | Blocks_warm_CMAES_JSO_TRQ | trq | 180 | aocc | +0.199 [+0.182,+0.216] 12/12 | +0.199 [+0.182,+0.216] 12/12 | 0/180 |
| free/d5/b100/q1 | Blocks_warm_CMAES_JSO_dimbudget | trq | 180 | aocc | +0.000 [+0.000,+0.000] 0/12 | +0.000 [+0.000,+0.000] 0/12 | 160/180 (+20 at 0) |
| free/d5/b100/q1 | RoundRobin_COBYQA | trq | 180 | aocc | +0.153 [+0.147,+0.160] 12/12 | +0.153 [+0.147,+0.160] 12/12 | 0/180 |
| free/d5/b100/q1 | RoundRobin_TRQ | trq | 180 | aocc | +0.232 [+0.225,+0.239] 12/12 | +0.232 [+0.225,+0.239] 12/12 | 0/180 |
| free/d5/b100/q1 | RoundRobin_TRQ_r05 | trq | 180 | aocc | +0.280 [+0.273,+0.286] 12/12 | +0.280 [+0.273,+0.286] 12/12 | 0/180 |
| free/d10/b100/q1 | RegimeGate_oracle | core | 180 | aocc | +0.001 [-0.004,+0.007] 6/12 | +0.001 [-0.004,+0.007] 6/12 | 0/180 (+34 at 0) |
| free/d10/b100/q1 | RoundRobin_CMAES | core | 180 | aocc | -0.001 [-0.008,+0.005] 6/12 | -0.001 [-0.008,+0.005] 6/12 | 0/180 (+36 at 0) |
| free/d10/b100/q1 | RoundRobin_Random | core | 180 | aocc | -0.059 [-0.064,-0.054] 0/12 | -0.059 [-0.064,-0.054] 0/12 | 0/180 (+36 at 0) |
| free/d10/b100/q1 | Blocks_warm_CMAES_JSO_TRQ | trq | 180 | aocc | +0.159 [+0.148,+0.170] 12/12 | +0.159 [+0.148,+0.170] 12/12 | 0/180 (+1 at 0) |
| free/d10/b100/q1 | Blocks_warm_CMAES_JSO_dimbudget | trq | 180 | aocc | +0.001 [-0.004,+0.007] 6/12 | +0.001 [-0.004,+0.007] 6/12 | 0/180 (+34 at 0) |
| free/d10/b100/q1 | RoundRobin_COBYQA | trq | 180 | aocc | +0.048 [+0.043,+0.053] 12/12 | +0.048 [+0.043,+0.053] 12/12 | 0/180 |
| free/d10/b100/q1 | RoundRobin_TRQ | trq | 180 | aocc | +0.058 [+0.052,+0.063] 12/12 | +0.058 [+0.052,+0.063] 12/12 | 0/180 |
| free/d10/b100/q1 | RoundRobin_TRQ_r05 | trq | 180 | aocc | +0.121 [+0.116,+0.127] 12/12 | +0.121 [+0.116,+0.127] 12/12 | 0/180 |

## Per family: Blocks_warm_CMAES_JSO − pool best (headline metric)

Δ per family (paired over seeds on that family's instances); the roadmap claim is *never much worse on any class*, so read the minimum of each row.

| cell | ackley | ellipsoid | rastrigin | rosenbrock | sharp_ridge |
|---|---|---|---|---|---|
| free/d2/b100/q1 | -0.132 [-0.222,-0.042] 1/12 | -0.643 [-0.673,-0.614] 0/12 | +0.055 [-0.004,+0.114] 10/12 | -0.280 [-0.377,-0.182] 0/12 | -0.015 [-0.052,+0.023] 4/12 |
| free/d5/b100/q1 | +0.135 [+0.074,+0.196] 11/12 | -0.115 [-0.145,-0.085] 0/12 | +0.039 [+0.027,+0.050] 12/12 | -0.076 [-0.130,-0.021] 3/12 | -0.057 [-0.078,-0.037] 0/12 |
| free/d10/b100/q1 | +0.007 [-0.016,+0.030] 6/12 | -0.000 [-0.000,+0.000] 0/12 | -0.002 [-0.007,+0.002] 4/12 | +0.004 [-0.002,+0.010] 10/12 | +0.010 [+0.002,+0.018] 10/12 |

### free/d2/b100/q1 (headline: aocc)

Pool: NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time PyBOBYQA (sequential).  Planned 15 strategies, present 15.

| strategy | seeds | AOCC | aocc_time | Δ aocc vs PyBOBYQA (sequential) (pool best) | Δ aocc vs NGOpt | Δ aocc vs Optuna_CmaEs | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| RoundRobin_TRQ_r05 | 12/12 | 0.602 | 0.600 | +0.172 [+0.148,+0.195] 12/12 | +0.413 [+0.398,+0.427] 12/12 | +0.418 [+0.412,+0.425] 12/12 | 0 | 0.3 |
| RoundRobin_TRQ | 12/12 | 0.545 | 0.544 | +0.115 [+0.092,+0.138] 12/12 | +0.356 [+0.343,+0.369] 12/12 | +0.362 [+0.354,+0.370] 12/12 | 0 | 0.5 |
| RoundRobin_COBYQA | 12/12 | 0.457 | 0.456 | +0.027 [+0.004,+0.050] 9/12 | +0.267 [+0.254,+0.281] 12/12 | +0.273 [+0.265,+0.281] 12/12 | 0 | 1.7 |
| Blocks_warm_CMAES_JSO_TRQ | 12/12 | 0.431 | 0.430 | +0.001 [-0.036,+0.037] 5/12 | +0.242 [+0.220,+0.263] 12/12 | +0.247 [+0.219,+0.276] 12/12 | 0 | 0.3 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.430 | 0.429 | – | – | – | 0 | 0.3 |
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.227 | 0.227 | -0.203 [-0.231,-0.175] 0/12 | +0.038 [+0.020,+0.056] 10/12 | +0.044 [+0.028,+0.059] 12/12 | 0 | 0.2 |
| RegimeGate_oracle | 12/12 | 0.227 | 0.227 | -0.203 [-0.231,-0.175] 0/12 | +0.038 [+0.020,+0.056] 10/12 | +0.044 [+0.028,+0.059] 12/12 | 0 | 0.2 |
| Blocks_warm_CMAES_JSO_dimbudget | 12/12 | 0.227 | 0.227 | -0.203 [-0.231,-0.175] 0/12 | +0.038 [+0.020,+0.056] 10/12 | +0.044 [+0.028,+0.059] 12/12 | 0 | 0.2 |
| RoundRobin_CMAES | 12/12 | 0.190 | 0.190 | -0.240 [-0.263,-0.218] 0/12 | +0.001 [-0.015,+0.016] 6/12 | +0.006 [-0.005,+0.017] 7/12 | 0 | 0.1 |
| NGOpt (pool) | 12/12 | 0.189 | 0.189 | – | – | – | 0 | 2.1 |
| Optuna_CmaEs (pool) | 12/12 | 0.184 | 0.183 | – | – | – | 0 | 0.2 |
| pycma_IPOP (pool) | 12/12 | 0.170 | 0.170 | – | – | – | 0 | 0.1 |
| Optuna_TPE (pool) | 12/12 | 0.168 | 0.167 | – | – | – | 0 | 0.5 |
| pycma_BIPOP (pool) | 12/12 | 0.168 | 0.168 | – | – | – | 0 | 0.1 |
| RoundRobin_Random | 12/12 | 0.128 | 0.128 | -0.302 [-0.325,-0.279] 0/12 | -0.061 [-0.076,-0.046] 0/12 | -0.055 [-0.064,-0.047] 0/12 | 0 | 0.1 |

### free/d5/b100/q1 (headline: aocc)

Pool: NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time PyBOBYQA (sequential).  Planned 15 strategies, present 15.

| strategy | seeds | AOCC | aocc_time | Δ aocc vs PyBOBYQA (sequential) (pool best) | Δ aocc vs NGOpt | Δ aocc vs Optuna_CmaEs | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| RoundRobin_TRQ_r05 | 12/12 | 0.405 | 0.404 | +0.265 [+0.249,+0.281] 12/12 | +0.290 [+0.286,+0.295] 12/12 | +0.300 [+0.297,+0.303] 12/12 | 0 | 1.3 |
| RoundRobin_TRQ | 12/12 | 0.357 | 0.357 | +0.217 [+0.201,+0.233] 12/12 | +0.243 [+0.238,+0.247] 12/12 | +0.252 [+0.249,+0.255] 12/12 | 0 | 1.4 |
| Blocks_warm_CMAES_JSO_TRQ | 12/12 | 0.324 | 0.324 | +0.184 [+0.167,+0.201] 12/12 | +0.210 [+0.194,+0.226] 12/12 | +0.219 [+0.205,+0.234] 12/12 | 0 | 0.6 |
| RoundRobin_COBYQA | 12/12 | 0.279 | 0.279 | +0.139 [+0.123,+0.155] 12/12 | +0.164 [+0.160,+0.169] 12/12 | +0.174 [+0.171,+0.177] 12/12 | 0 | 2.2 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.140 | 0.140 | – | – | – | 0 | 1.3 |
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.125 | 0.125 | -0.015 [-0.032,+0.002] 3/12 | +0.011 [+0.002,+0.020] 9/12 | +0.020 [+0.015,+0.025] 12/12 | 0 | 0.3 |
| RegimeGate_oracle | 12/12 | 0.125 | 0.125 | -0.015 [-0.032,+0.002] 3/12 | +0.011 [+0.002,+0.020] 9/12 | +0.020 [+0.015,+0.025] 12/12 | 0 | 0.3 |
| Blocks_warm_CMAES_JSO_dimbudget | 12/12 | 0.125 | 0.125 | -0.015 [-0.032,+0.002] 3/12 | +0.011 [+0.002,+0.020] 9/12 | +0.020 [+0.015,+0.025] 12/12 | 0 | 0.4 |
| NGOpt (pool) | 12/12 | 0.114 | 0.114 | – | – | – | 0 | 6.1 |
| RoundRobin_CMAES | 12/12 | 0.112 | 0.112 | -0.028 [-0.044,-0.012] 2/12 | -0.002 [-0.007,+0.002] 6/12 | +0.007 [+0.003,+0.011] 10/12 | 0 | 0.1 |
| Optuna_CmaEs (pool) | 12/12 | 0.105 | 0.104 | – | – | – | 0 | 0.5 |
| pycma_IPOP (pool) | 12/12 | 0.092 | 0.092 | – | – | – | 0 | 0.1 |
| pycma_BIPOP (pool) | 12/12 | 0.092 | 0.091 | – | – | – | 0 | 0.1 |
| Optuna_TPE (pool) | 12/12 | 0.084 | 0.084 | – | – | – | 0 | 2.2 |
| RoundRobin_Random | 12/12 | 0.045 | 0.045 | -0.095 [-0.111,-0.078] 0/12 | -0.069 [-0.074,-0.065] 0/12 | -0.059 [-0.063,-0.056] 0/12 | 0 | 0.2 |

### free/d10/b100/q1 (headline: aocc)

Pool: NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), pycma_BIPOP, pycma_IPOP.  Pool best: AOCC Optuna_CmaEs, aocc_time Optuna_CmaEs.  Planned 15 strategies, present 15.

| strategy | seeds | AOCC | aocc_time | Δ aocc vs Optuna_CmaEs (pool best) | Δ aocc vs NGOpt | Δ aocc vs PyBOBYQA (sequential) | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| Blocks_warm_CMAES_JSO_TRQ | 12/12 | 0.245 | 0.245 | +0.163 [+0.151,+0.174] 12/12 | +0.164 [+0.151,+0.177] 12/12 | +0.168 [+0.156,+0.180] 12/12 | 0 | 2.0 |
| RoundRobin_TRQ_r05 | 12/12 | 0.207 | 0.207 | +0.125 [+0.123,+0.128] 12/12 | +0.127 [+0.124,+0.130] 12/12 | +0.131 [+0.124,+0.137] 12/12 | 0 | 3.9 |
| RoundRobin_TRQ | 12/12 | 0.143 | 0.143 | +0.061 [+0.059,+0.064] 12/12 | +0.063 [+0.061,+0.066] 12/12 | +0.067 [+0.060,+0.073] 12/12 | 0 | 3.8 |
| RoundRobin_COBYQA | 12/12 | 0.134 | 0.134 | +0.052 [+0.049,+0.054] 12/12 | +0.054 [+0.051,+0.056] 12/12 | +0.057 [+0.051,+0.064] 12/12 | 0 | 3.9 |
| RegimeGate_oracle | 12/12 | 0.087 | 0.087 | +0.005 [+0.002,+0.008] 11/12 | +0.007 [+0.003,+0.011] 9/12 | +0.010 [+0.004,+0.017] 10/12 | 0 | 0.7 |
| Blocks_warm_CMAES_JSO_dimbudget | 12/12 | 0.087 | 0.087 | +0.005 [+0.002,+0.008] 11/12 | +0.007 [+0.003,+0.011] 9/12 | +0.010 [+0.004,+0.017] 10/12 | 0 | 0.7 |
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.086 | 0.086 | +0.004 [-0.002,+0.010] 8/12 | +0.006 [+0.000,+0.011] 9/12 | +0.009 [+0.002,+0.017] 10/12 | 0 | 1.0 |
| RoundRobin_CMAES | 12/12 | 0.085 | 0.084 | +0.003 [-0.001,+0.006] 10/12 | +0.004 [+0.000,+0.008] 9/12 | +0.008 [+0.000,+0.016] 9/12 | 0 | 0.4 |
| Optuna_CmaEs (pool) | 12/12 | 0.082 | 0.082 | – | – | – | 0 | 2.0 |
| NGOpt (pool) | 12/12 | 0.080 | 0.080 | – | – | – | 0 | 11.3 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.077 | 0.077 | – | – | – | 0 | 12.1 |
| pycma_IPOP (pool) | 12/12 | 0.066 | 0.066 | – | – | – | 0 | 0.3 |
| pycma_BIPOP (pool) | 12/12 | 0.066 | 0.066 | – | – | – | 0 | 0.3 |
| Optuna_TPE (pool) | 12/12 | 0.047 | 0.047 | – | – | – | 0 | 16.1 |
| RoundRobin_Random | 12/12 | 0.027 | 0.027 | -0.055 [-0.058,-0.052] 0/12 | -0.053 [-0.056,-0.051] 0/12 | -0.050 [-0.057,-0.043] 0/12 | 0 | 0.7 |
