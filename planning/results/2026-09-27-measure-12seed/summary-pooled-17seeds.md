# Expensive-track measurement

Commit 1778e1b, 58cf1e7; run local; 562 unit(s); missing 0, failed 0, unreadable files 0.
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
- *Best other panobbgo*: the best secondary panobbgo spec of the cell (the opt-in `trq` group's specs too, when its units are aggregated here), with its Δ vs the pool best.  The table without the ellipsoid family is descriptive: read it next to the headline, not instead of it.

FP environments (per shard): 80ee2a0090c4: 207 shard(s).  Comparisons across jobs are ordinary samples; the environment is reproducibility metadata.

## Headline

| cell | metric | n (runs/plan) | pool best | Blocks_warm_CMAES_JSO | Δ [CI95] wins | p | p_holm | best other panobbgo | errors pb/ext | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| free/d2/b100/q4 | aocc_time | 255/255 | BoTorch_qLogEI 0.198 (17/17) | 0.212 (17/17) | +0.013 [-0.005,+0.031] 10/17 | 0.136 | 0.272 | RegimeGate_oracle 0.212 +0.013 [-0.005,+0.031] 10/17 | 0/0 |  |
| free/d2/b100/q16 | aocc_time | 255/255 | BoTorch_qLogEI 0.176 (17/17) | 0.163 (17/17) | -0.012 [-0.020,-0.004] 3/17 | 0.005 | 0.015 | RegimeGate_oracle 0.163 -0.012 [-0.020,-0.004] 3/17 | 0/0 |  |
| free/d2/b100/q64 | aocc_time | 255/255 | BoTorch_qLogEI 0.113 (17/17) | 0.098 (17/17) | -0.015 [-0.019,-0.011] 0/17 | 0.000 | 0.000 | RegimeGate_oracle 0.098 -0.015 [-0.019,-0.011] 0/17 | 0/0 |  |
| free/d5/b100/q4 | aocc_time | 255/255 | TuRBO1 0.112 (17/17) | 0.124 (17/17) | +0.011 [+0.008,+0.015] 16/17 | 0.000 | 0.000 | RegimeGate_oracle 0.124 +0.011 [+0.008,+0.015] 16/17 | 0/0 |  |
| free/d5/b100/q16 | aocc_time | 255/255 | BoTorch_qLogEI 0.073 (17/17) | 0.103 (17/17) | +0.030 [+0.026,+0.034] 17/17 | 0.000 | 0.000 | RegimeGate_oracle 0.103 +0.030 [+0.026,+0.034] 17/17 | 0/0 |  |
| free/d5/b100/q64 | aocc_time | 255/255 | BoTorch_qLogEI 0.060 (17/17) | 0.060 (17/17) | +0.000 [-0.001,+0.002] 10/17 | 0.727 | 0.727 | RegimeGate_oracle 0.060 +0.000 [-0.001,+0.002] 10/17 | 0/0 |  |
| free/d10/b100/q4 | aocc_time | 255/255 | TuRBO1 0.086 (17/17) | 0.080 (17/17) | -0.006 [-0.010,-0.002] 3/17 | 0.004 | 0.015 | RegimeGate_oracle 0.093 +0.007 [+0.004,+0.009] 14/17 | 0/0 |  |
| free/d10/b100/q16 | aocc_time | 255/255 | TuRBO1 0.059 (17/17) | 0.072 (17/17) | +0.014 [+0.010,+0.017] 17/17 | 0.000 | 0.000 | RegimeGate_oracle 0.070 +0.012 [+0.008,+0.015] 17/17 | 0/0 |  |
| free/d10/b100/q64 | aocc_time | 255/255 | TuRBO1 0.028 (17/17) | 0.046 (17/17) | +0.018 [+0.017,+0.018] 17/17 | 0.000 | 0.000 | RegimeGate_oracle 0.044 +0.016 [+0.016,+0.017] 17/17 | 0/0 |  |

## Without the ellipsoid family (descriptive)

The headline table on the common runs minus the ellipsoid instances: the one exactly quadratic family, which a quadratic model solves exactly and which can decide a family mean on its own (DISCOVERY §66).  The pool is the cell's; its best is re-selected on these runs.  **Descriptive**: unadjusted, not part of the Holm family above, which stays over all families.

| cell | metric | n (runs) | pool best | Blocks_warm_CMAES_JSO | Δ [CI95] wins | best other panobbgo |
|---|---|---|---|---|---|---|
| free/d2/b100/q4 | aocc_time | 204 | BoTorch_qLogEI 0.216 | 0.236 | +0.019 [+0.001,+0.037] 10/17 | RegimeGate_oracle 0.236 +0.019 [+0.001,+0.037] 10/17 |
| free/d2/b100/q16 | aocc_time | 204 | BoTorch_qLogEI 0.190 | 0.185 | -0.005 [-0.014,+0.004] 6/17 | RegimeGate_oracle 0.185 -0.005 [-0.014,+0.004] 6/17 |
| free/d2/b100/q64 | aocc_time | 204 | Optuna_TPE 0.134 | 0.118 | -0.016 [-0.021,-0.011] 1/17 | RegimeGate_oracle 0.118 -0.016 [-0.021,-0.011] 1/17 |
| free/d5/b100/q4 | aocc_time | 204 | TuRBO1 0.138 | 0.153 | +0.015 [+0.011,+0.018] 17/17 | RegimeGate_oracle 0.153 +0.015 [+0.011,+0.018] 17/17 |
| free/d5/b100/q16 | aocc_time | 204 | BoTorch_qLogEI 0.091 | 0.128 | +0.037 [+0.032,+0.042] 17/17 | RegimeGate_oracle 0.128 +0.037 [+0.032,+0.042] 17/17 |
| free/d5/b100/q64 | aocc_time | 204 | BoTorch_qLogEI 0.075 | 0.075 | +0.000 [-0.002,+0.002] 10/17 | RegimeGate_oracle 0.075 +0.000 [-0.002,+0.002] 10/17 |
| free/d10/b100/q4 | aocc_time | 204 | TuRBO1 0.107 | 0.100 | -0.008 [-0.013,-0.003] 3/17 | RegimeGate_oracle 0.116 +0.009 [+0.005,+0.012] 14/17 |
| free/d10/b100/q16 | aocc_time | 204 | TuRBO1 0.073 | 0.090 | +0.017 [+0.013,+0.021] 17/17 | RegimeGate_oracle 0.088 +0.015 [+0.010,+0.019] 17/17 |
| free/d10/b100/q64 | aocc_time | 204 | TuRBO1 0.035 | 0.057 | +0.022 [+0.021,+0.023] 17/17 | RegimeGate_oracle 0.055 +0.021 [+0.020,+0.022] 17/17 |

## Per family: Blocks_warm_CMAES_JSO − pool best (headline metric)

Δ per family (paired over seeds on that family's instances); the roadmap claim is *never much worse on any class*, so read the minimum of each row.

| cell | ackley | ellipsoid | rastrigin | rosenbrock | sharp_ridge |
|---|---|---|---|---|---|
| free/d2/b100/q4 | +0.065 [+0.035,+0.096] 14/17 | -0.011 [-0.038,+0.016] 5/17 | -0.004 [-0.047,+0.038] 6/17 | +0.017 [-0.023,+0.056] 9/17 | -0.001 [-0.025,+0.023] 8/17 |
| free/d2/b100/q16 | -0.005 [-0.022,+0.013] 9/17 | -0.041 [-0.054,-0.028] 0/17 | -0.007 [-0.019,+0.004] 7/17 | +0.006 [-0.014,+0.026] 9/17 | -0.015 [-0.031,+0.002] 4/17 |
| free/d2/b100/q64 | -0.030 [-0.039,-0.022] 0/17 | -0.022 [-0.032,-0.011] 3/17 | -0.004 [-0.010,+0.001] 6/17 | -0.007 [-0.020,+0.006] 6/17 | -0.011 [-0.020,-0.002] 3/17 |
| free/d5/b100/q4 | +0.028 [+0.017,+0.040] 16/17 | -0.003 [-0.011,+0.006] 7/17 | -0.000 [-0.009,+0.008] 10/17 | +0.026 [+0.014,+0.038] 14/17 | +0.006 [-0.006,+0.018] 10/17 |
| free/d5/b100/q16 | +0.051 [+0.039,+0.062] 16/17 | +0.004 [+0.002,+0.007] 14/17 | -0.007 [-0.011,-0.004] 2/17 | +0.084 [+0.077,+0.091] 17/17 | +0.019 [+0.009,+0.029] 13/17 |
| free/d5/b100/q64 | -0.018 [-0.022,-0.014] 0/17 | +0.000 [-0.000,+0.001] 2/17 | -0.010 [-0.014,-0.006] 2/17 | +0.047 [+0.043,+0.051] 17/17 | -0.017 [-0.022,-0.013] 0/17 |
| free/d10/b100/q4 | -0.036 [-0.051,-0.021] 3/17 | +0.000 [-0.000,+0.001] 1/17 | -0.014 [-0.021,-0.007] 3/17 | +0.021 [+0.014,+0.028] 16/17 | -0.002 [-0.008,+0.004] 8/17 |
| free/d10/b100/q16 | +0.017 [+0.003,+0.031] 12/17 | +0.000 [+0.000,+0.000] 0/17 | -0.005 [-0.009,+0.000] 5/17 | +0.033 [+0.026,+0.040] 16/17 | +0.023 [+0.017,+0.029] 16/17 |
| free/d10/b100/q64 | +0.032 [+0.030,+0.035] 17/17 | +0.000 [+0.000,+0.000] 0/17 | +0.007 [+0.005,+0.009] 17/17 | +0.023 [+0.019,+0.027] 17/17 | +0.026 [+0.025,+0.028] 17/17 |

### free/d2/b100/q4 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs TuRBO1 | Δ aocc_time vs Optuna_TPE | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| **Blocks_warm_CMAES_JSO** | 17/17 | 0.214 | 0.212 | +0.013 [-0.005,+0.031] 10/17 | +0.039 [+0.020,+0.059] 15/17 | +0.046 [+0.031,+0.061] 17/17 | 0 | 0.2 |
| RegimeGate_oracle | 17/17 | 0.214 | 0.212 | +0.013 [-0.005,+0.031] 10/17 | +0.039 [+0.020,+0.059] 15/17 | +0.046 [+0.031,+0.061] 17/17 | 0 | 0.2 |
| BoTorch_qLogEI (pool) | 17/17 | 0.200 | 0.198 | – | – | – | 0 | 249.0 |
| RoundRobin_CMAES | 17/17 | 0.184 | 0.183 | -0.015 [-0.023,-0.007] 4/17 | +0.011 [-0.001,+0.022] 13/17 | +0.017 [+0.012,+0.022] 16/17 | 0 | 0.1 |
| TuRBO1 (pool) | 17/17 | 0.207 | 0.172 | – | – | – | 0 | 24.4 |
| Optuna_TPE (pool) | 17/17 | 0.167 | 0.166 | – | – | – | 0 | 0.7 |
| PyBOBYQA (sequential) (pool) | 17/17 | 0.415 | 0.157 | – | – | – | 0 | 0.4 |
| NGOpt (pool) | 17/17 | 0.158 | 0.157 | – | – | – | 0 | 2.6 |
| Optuna_CmaEs (pool) | 17/17 | 0.151 | 0.150 | – | – | – | 0 | 0.3 |
| pycma_IPOP (pool) | 17/17 | 0.177 | 0.143 | – | – | – | 0 | 0.1 |
| pycma_BIPOP (pool) | 17/17 | 0.173 | 0.141 | – | – | – | 0 | 0.1 |
| RoundRobin_Random | 17/17 | 0.129 | 0.128 | -0.070 [-0.079,-0.062] 0/17 | -0.044 [-0.055,-0.034] 0/17 | -0.038 [-0.044,-0.032] 0/17 | 0 | 0.1 |

### free/d2/b100/q16 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs TuRBO1 | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| BoTorch_qLogEI (pool) | 17/17 | 0.182 | 0.176 | – | – | – | 0 | 345.5 |
| **Blocks_warm_CMAES_JSO** | 17/17 | 0.171 | 0.163 | -0.012 [-0.020,-0.004] 3/17 | +0.016 [+0.009,+0.023] 17/17 | +0.040 [+0.033,+0.047] 17/17 | 0 | 0.1 |
| RegimeGate_oracle | 17/17 | 0.171 | 0.163 | -0.012 [-0.020,-0.004] 3/17 | +0.016 [+0.009,+0.023] 17/17 | +0.040 [+0.033,+0.047] 17/17 | 0 | 0.1 |
| Optuna_TPE (pool) | 17/17 | 0.153 | 0.147 | – | – | – | 0 | 0.5 |
| RoundRobin_CMAES | 17/17 | 0.153 | 0.141 | -0.034 [-0.040,-0.028] 0/17 | -0.006 [-0.011,-0.000] 4/17 | +0.018 [+0.012,+0.025] 16/17 | 0 | 0.1 |
| TuRBO1 (pool) | 17/17 | 0.181 | 0.123 | – | – | – | 0 | 7.1 |
| NGOpt (pool) | 17/17 | 0.126 | 0.120 | – | – | – | 0 | 1.7 |
| Optuna_CmaEs (pool) | 17/17 | 0.122 | 0.118 | – | – | – | 0 | 0.3 |
| RoundRobin_Random | 17/17 | 0.121 | 0.116 | -0.059 [-0.066,-0.053] 0/17 | -0.031 [-0.037,-0.025] 0/17 | -0.007 [-0.013,-0.000] 6/17 | 0 | 0.1 |
| pycma_BIPOP (pool) | 17/17 | 0.137 | 0.097 | – | – | – | 0 | 0.1 |
| pycma_IPOP (pool) | 17/17 | 0.137 | 0.096 | – | – | – | 0 | 0.1 |
| PyBOBYQA (sequential) (pool) | 17/17 | 0.415 | 0.053 | – | – | – | 0 | 0.3 |

### free/d2/b100/q64 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs Optuna_CmaEs | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| BoTorch_qLogEI (pool) | 17/17 | 0.134 | 0.113 | – | – | – | 0 | 759.4 |
| Optuna_TPE (pool) | 17/17 | 0.129 | 0.110 | – | – | – | 0 | 0.4 |
| **Blocks_warm_CMAES_JSO** | 17/17 | 0.146 | 0.098 | -0.015 [-0.019,-0.011] 0/17 | -0.012 [-0.015,-0.008] 1/17 | +0.003 [+0.000,+0.006] 12/17 | 0 | 0.1 |
| RegimeGate_oracle | 17/17 | 0.146 | 0.098 | -0.015 [-0.019,-0.011] 0/17 | -0.012 [-0.015,-0.008] 1/17 | +0.003 [+0.000,+0.006] 12/17 | 0 | 0.1 |
| Optuna_CmaEs (pool) | 17/17 | 0.110 | 0.095 | – | – | – | 0 | 0.2 |
| TuRBO1 (pool) | 17/17 | 0.159 | 0.090 | – | – | – | 0 | 1.9 |
| RoundRobin_Random | 17/17 | 0.114 | 0.089 | -0.024 [-0.029,-0.019] 0/17 | -0.021 [-0.025,-0.017] 0/17 | -0.006 [-0.011,-0.001] 6/17 | 0 | 0.1 |
| RoundRobin_CMAES | 17/17 | 0.147 | 0.083 | -0.030 [-0.034,-0.026] 0/17 | -0.027 [-0.032,-0.023] 0/17 | -0.012 [-0.018,-0.007] 1/17 | 0 | 0.1 |
| pycma_BIPOP (pool) | 17/17 | 0.108 | 0.078 | – | – | – | 0 | 0.0 |
| pycma_IPOP (pool) | 17/17 | 0.109 | 0.077 | – | – | – | 0 | 0.0 |
| NGOpt (pool) | 17/17 | 0.073 | 0.062 | – | – | – | 0 | 0.5 |
| PyBOBYQA (sequential) (pool) | 17/17 | 0.415 | 0.023 | – | – | – | 0 | 0.3 |

### free/d5/b100/q4 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time TuRBO1.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs TuRBO1 (pool best) | Δ aocc_time vs Optuna_CmaEs | Δ aocc_time vs BoTorch_qLogEI | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| **Blocks_warm_CMAES_JSO** | 17/17 | 0.124 | 0.124 | +0.011 [+0.008,+0.015] 16/17 | +0.039 [+0.035,+0.044] 17/17 | +0.043 [+0.038,+0.048] 17/17 | 0 | 0.5 |
| RegimeGate_oracle | 17/17 | 0.124 | 0.124 | +0.011 [+0.008,+0.015] 16/17 | +0.039 [+0.035,+0.044] 17/17 | +0.043 [+0.038,+0.048] 17/17 | 0 | 0.5 |
| TuRBO1 (pool) | 17/17 | 0.130 | 0.112 | – | – | – | 0 | 64.6 |
| RoundRobin_CMAES | 17/17 | 0.111 | 0.110 | -0.002 [-0.006,+0.002] 6/17 | +0.026 [+0.022,+0.029] 17/17 | +0.029 [+0.026,+0.033] 17/17 | 0 | 0.2 |
| Optuna_CmaEs (pool) | 17/17 | 0.085 | 0.084 | – | – | – | 0 | 0.8 |
| BoTorch_qLogEI (pool) | 17/17 | 0.081 | 0.081 | – | – | – | 0 | 1451.7 |
| Optuna_TPE (pool) | 17/17 | 0.080 | 0.079 | – | – | – | 0 | 3.5 |
| pycma_BIPOP (pool) | 17/17 | 0.094 | 0.074 | – | – | – | 0 | 0.2 |
| pycma_IPOP (pool) | 17/17 | 0.091 | 0.072 | – | – | – | 0 | 0.2 |
| NGOpt (pool) | 17/17 | 0.068 | 0.067 | – | – | – | 0 | 9.2 |
| PyBOBYQA (sequential) (pool) | 17/17 | 0.145 | 0.059 | – | – | – | 0 | 2.3 |
| RoundRobin_Random | 17/17 | 0.045 | 0.045 | -0.068 [-0.071,-0.064] 0/17 | -0.040 [-0.042,-0.037] 0/17 | -0.036 [-0.040,-0.033] 0/17 | 0 | 0.3 |

### free/d5/b100/q16 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs TuRBO1 | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| **Blocks_warm_CMAES_JSO** | 17/17 | 0.108 | 0.103 | +0.030 [+0.026,+0.034] 17/17 | +0.034 [+0.030,+0.038] 17/17 | +0.038 [+0.034,+0.042] 17/17 | 0 | 0.4 |
| RegimeGate_oracle | 17/17 | 0.108 | 0.103 | +0.030 [+0.026,+0.034] 17/17 | +0.034 [+0.030,+0.038] 17/17 | +0.038 [+0.034,+0.042] 17/17 | 0 | 0.4 |
| RoundRobin_CMAES | 17/17 | 0.085 | 0.079 | +0.006 [+0.004,+0.007] 17/17 | +0.009 [+0.007,+0.012] 15/17 | +0.014 [+0.012,+0.016] 17/17 | 0 | 0.2 |
| BoTorch_qLogEI (pool) | 17/17 | 0.074 | 0.073 | – | – | – | 0 | 1627.7 |
| Optuna_TPE (pool) | 17/17 | 0.071 | 0.069 | – | – | – | 0 | 3.3 |
| TuRBO1 (pool) | 17/17 | 0.111 | 0.065 | – | – | – | 0 | 24.0 |
| Optuna_CmaEs (pool) | 17/17 | 0.055 | 0.054 | – | – | – | 0 | 0.9 |
| NGOpt (pool) | 17/17 | 0.046 | 0.045 | – | – | – | 0 | 8.4 |
| RoundRobin_Random | 17/17 | 0.042 | 0.041 | -0.032 [-0.034,-0.030] 0/17 | -0.028 [-0.031,-0.025] 0/17 | -0.024 [-0.025,-0.022] 0/17 | 0 | 0.2 |
| pycma_IPOP (pool) | 17/17 | 0.065 | 0.038 | – | – | – | 0 | 0.1 |
| pycma_BIPOP (pool) | 17/17 | 0.063 | 0.037 | – | – | – | 0 | 0.1 |
| PyBOBYQA (sequential) (pool) | 17/17 | 0.145 | 0.027 | – | – | – | 0 | 2.1 |

### free/d5/b100/q64 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs TuRBO1 | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| **Blocks_warm_CMAES_JSO** | 17/17 | 0.075 | 0.060 | +0.000 [-0.001,+0.002] 10/17 | +0.015 [+0.014,+0.017] 17/17 | +0.023 [+0.022,+0.024] 17/17 | 0 | 0.3 |
| RegimeGate_oracle | 17/17 | 0.075 | 0.060 | +0.000 [-0.001,+0.002] 10/17 | +0.015 [+0.014,+0.017] 17/17 | +0.023 [+0.022,+0.024] 17/17 | 0 | 0.3 |
| BoTorch_qLogEI (pool) | 17/17 | 0.064 | 0.060 | – | – | – | 0 | 3157.5 |
| Optuna_TPE (pool) | 17/17 | 0.048 | 0.045 | – | – | – | 0 | 3.7 |
| RoundRobin_CMAES | 17/17 | 0.053 | 0.045 | -0.015 [-0.017,-0.013] 0/17 | -0.000 [-0.001,+0.001] 8/17 | +0.007 [+0.006,+0.008] 17/17 | 0 | 0.2 |
| TuRBO1 (pool) | 17/17 | 0.058 | 0.038 | – | – | – | 0 | 7.5 |
| Optuna_CmaEs (pool) | 17/17 | 0.039 | 0.037 | – | – | – | 0 | 1.1 |
| RoundRobin_Random | 17/17 | 0.038 | 0.035 | -0.025 [-0.027,-0.023] 0/17 | -0.010 [-0.011,-0.009] 0/17 | -0.003 [-0.004,-0.002] 0/17 | 0 | 0.2 |
| pycma_BIPOP (pool) | 17/17 | 0.038 | 0.030 | – | – | – | 0 | 0.1 |
| pycma_IPOP (pool) | 17/17 | 0.038 | 0.030 | – | – | – | 0 | 0.1 |
| NGOpt (pool) | 17/17 | 0.026 | 0.025 | – | – | – | 0 | 2.7 |
| PyBOBYQA (sequential) (pool) | 17/17 | 0.145 | 0.019 | – | – | – | 0 | 2.2 |

### free/d10/b100/q4 (headline: aocc_time)

Pool: NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC TuRBO1, aocc_time TuRBO1.  Planned 11 strategies, present 11.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs TuRBO1 (pool best) | Δ aocc_time vs Optuna_CmaEs | Δ aocc_time vs pycma_IPOP | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| RegimeGate_oracle | 17/17 | 0.093 | 0.093 | +0.007 [+0.004,+0.009] 14/17 | +0.026 [+0.024,+0.028] 17/17 | +0.039 [+0.037,+0.041] 17/17 | 0 | 0.7 |
| RoundRobin_CMAES | 17/17 | 0.093 | 0.093 | +0.007 [+0.004,+0.009] 16/17 | +0.026 [+0.024,+0.028] 17/17 | +0.039 [+0.036,+0.041] 17/17 | 0 | 0.4 |
| TuRBO1 (pool) | 17/17 | 0.097 | 0.086 | – | – | – | 0 | 196.5 |
| **Blocks_warm_CMAES_JSO** | 17/17 | 0.080 | 0.080 | -0.006 [-0.010,-0.002] 3/17 | +0.013 [+0.009,+0.018] 16/17 | +0.026 [+0.023,+0.029] 17/17 | 0 | 0.9 |
| Optuna_CmaEs (pool) | 17/17 | 0.067 | 0.067 | – | – | – | 0 | 2.1 |
| pycma_IPOP (pool) | 17/17 | 0.067 | 0.054 | – | – | – | 0 | 0.3 |
| pycma_BIPOP (pool) | 17/17 | 0.066 | 0.053 | – | – | – | 0 | 0.3 |
| NGOpt (pool) | 17/17 | 0.051 | 0.051 | – | – | – | 0 | 2.0 |
| Optuna_TPE (pool) | 17/17 | 0.044 | 0.044 | – | – | – | 0 | 14.5 |
| PyBOBYQA (sequential) (pool) | 17/17 | 0.072 | 0.043 | – | – | – | 0 | 11.5 |
| RoundRobin_Random | 17/17 | 0.027 | 0.027 | -0.059 [-0.061,-0.057] 0/17 | -0.040 [-0.041,-0.039] 0/17 | -0.027 [-0.029,-0.025] 0/17 | 0 | 0.7 |

### free/d10/b100/q16 (headline: aocc_time)

Pool: NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC TuRBO1, aocc_time TuRBO1.  Planned 11 strategies, present 11.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs TuRBO1 (pool best) | Δ aocc_time vs NGOpt | Δ aocc_time vs Optuna_CmaEs | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| **Blocks_warm_CMAES_JSO** | 17/17 | 0.076 | 0.072 | +0.014 [+0.010,+0.017] 17/17 | +0.018 [+0.015,+0.021] 17/17 | +0.034 [+0.031,+0.036] 17/17 | 0 | 0.7 |
| RegimeGate_oracle | 17/17 | 0.073 | 0.070 | +0.012 [+0.008,+0.015] 17/17 | +0.016 [+0.014,+0.019] 17/17 | +0.032 [+0.028,+0.035] 17/17 | 0 | 0.5 |
| RoundRobin_CMAES | 17/17 | 0.070 | 0.064 | +0.006 [+0.003,+0.008] 15/17 | +0.010 [+0.008,+0.013] 17/17 | +0.026 [+0.024,+0.027] 17/17 | 0 | 0.3 |
| TuRBO1 (pool) | 17/17 | 0.088 | 0.059 | – | – | – | 0 | 90.6 |
| NGOpt (pool) | 17/17 | 0.054 | 0.054 | – | – | – | 0 | 1.4 |
| Optuna_CmaEs (pool) | 17/17 | 0.039 | 0.039 | – | – | – | 0 | 2.1 |
| Optuna_TPE (pool) | 17/17 | 0.036 | 0.036 | – | – | – | 0 | 12.8 |
| pycma_IPOP (pool) | 17/17 | 0.047 | 0.027 | – | – | – | 0 | 0.2 |
| pycma_BIPOP (pool) | 17/17 | 0.047 | 0.027 | – | – | – | 0 | 0.2 |
| RoundRobin_Random | 17/17 | 0.026 | 0.026 | -0.033 [-0.035,-0.031] 0/17 | -0.028 [-0.031,-0.025] 0/17 | -0.013 [-0.014,-0.012] 0/17 | 0 | 0.5 |
| PyBOBYQA (sequential) (pool) | 17/17 | 0.072 | 0.021 | – | – | – | 0 | 9.5 |

### free/d10/b100/q64 (headline: aocc_time)

Pool: NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time TuRBO1.  Planned 11 strategies, present 11.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs TuRBO1 (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs Optuna_CmaEs | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| **Blocks_warm_CMAES_JSO** | 17/17 | 0.054 | 0.046 | +0.018 [+0.017,+0.018] 17/17 | +0.018 [+0.017,+0.019] 17/17 | +0.019 [+0.018,+0.019] 17/17 | 0 | 0.8 |
| RegimeGate_oracle | 17/17 | 0.047 | 0.044 | +0.016 [+0.016,+0.017] 17/17 | +0.017 [+0.015,+0.018] 17/17 | +0.017 [+0.016,+0.018] 17/17 | 0 | 0.6 |
| RoundRobin_CMAES | 17/17 | 0.031 | 0.029 | +0.001 [+0.001,+0.002] 17/17 | +0.001 [+0.001,+0.002] 16/17 | +0.002 [+0.002,+0.003] 17/17 | 0 | 0.4 |
| TuRBO1 (pool) | 17/17 | 0.046 | 0.028 | – | – | – | 0 | 72.5 |
| Optuna_TPE (pool) | 17/17 | 0.028 | 0.028 | – | – | – | 0 | 17.9 |
| Optuna_CmaEs (pool) | 17/17 | 0.028 | 0.027 | – | – | – | 0 | 3.9 |
| RoundRobin_Random | 17/17 | 0.025 | 0.024 | -0.003 [-0.004,-0.003] 0/17 | -0.003 [-0.004,-0.003] 0/17 | -0.003 [-0.003,-0.002] 0/17 | 0 | 0.6 |
| pycma_IPOP (pool) | 17/17 | 0.027 | 0.023 | – | – | – | 0 | 0.2 |
| pycma_BIPOP (pool) | 17/17 | 0.027 | 0.023 | – | – | – | 0 | 0.2 |
| NGOpt (pool) | 17/17 | 0.021 | 0.021 | – | – | – | 0 | 10.4 |
| PyBOBYQA (sequential) (pool) | 17/17 | 0.072 | 0.018 | – | – | – | 0 | 13.0 |
