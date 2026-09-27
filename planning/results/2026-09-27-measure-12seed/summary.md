# Expensive-track measurement

Commit 58cf1e7; run 36315576900; 432 unit(s); missing 0, failed 0, unreadable files 0.
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
| free/d2/b100/q4 | aocc_time | 180/180 | BoTorch_qLogEI 0.194 (12/12) | 0.215 (12/12) | +0.021 [-0.002,+0.043] 8/12 | 0.067 | 0.183 | RegimeGate_oracle 0.215 +0.021 [-0.002,+0.043] 8/12 | 0/0 |  |
| free/d2/b100/q16 | aocc_time | 180/180 | BoTorch_qLogEI 0.176 (12/12) | 0.165 (12/12) | -0.011 [-0.022,+0.001] 3/12 | 0.061 | 0.183 | RegimeGate_oracle 0.165 -0.011 [-0.022,+0.001] 3/12 | 0/0 |  |
| free/d2/b100/q64 | aocc_time | 180/180 | BoTorch_qLogEI 0.113 (12/12) | 0.097 (12/12) | -0.015 [-0.021,-0.010] 0/12 | 0.000 | 0.001 | RegimeGate_oracle 0.097 -0.015 [-0.021,-0.010] 0/12 | 0/0 |  |
| free/d5/b100/q4 | aocc_time | 180/180 | TuRBO1 0.113 (12/12) | 0.124 (12/12) | +0.011 [+0.007,+0.016] 11/12 | 0.000 | 0.001 | RegimeGate_oracle 0.124 +0.011 [+0.007,+0.016] 11/12 | 0/0 |  |
| free/d5/b100/q16 | aocc_time | 180/180 | BoTorch_qLogEI 0.074 (12/12) | 0.103 (12/12) | +0.030 [+0.025,+0.035] 12/12 | 0.000 | 0.000 | RegimeGate_oracle 0.103 +0.030 [+0.025,+0.035] 12/12 | 0/0 |  |
| free/d5/b100/q64 | aocc_time | 180/180 | BoTorch_qLogEI 0.060 (12/12) | 0.060 (12/12) | +0.001 [-0.001,+0.002] 8/12 | 0.347 | 0.347 | RegimeGate_oracle 0.060 +0.001 [-0.001,+0.002] 8/12 | 0/0 |  |
| free/d10/b100/q4 | aocc_time | 180/180 | TuRBO1 0.086 (12/12) | 0.079 (12/12) | -0.008 [-0.012,-0.004] 1/12 | 0.002 | 0.007 | RegimeGate_oracle 0.093 +0.006 [+0.003,+0.010] 10/12 | 0/0 |  |
| free/d10/b100/q16 | aocc_time | 180/180 | TuRBO1 0.058 (12/12) | 0.072 (12/12) | +0.014 [+0.010,+0.018] 12/12 | 0.000 | 0.000 | RegimeGate_oracle 0.070 +0.012 [+0.007,+0.017] 12/12 | 0/0 |  |
| free/d10/b100/q64 | aocc_time | 180/180 | TuRBO1 0.028 (12/12) | 0.045 (12/12) | +0.017 [+0.016,+0.018] 12/12 | 0.000 | 0.000 | RegimeGate_oracle 0.044 +0.016 [+0.015,+0.018] 12/12 | 0/0 |  |

## Without the ellipsoid family (descriptive)

The headline table on the common runs minus the ellipsoid instances: the one exactly quadratic family, which a quadratic model solves exactly and which can decide a family mean on its own (DISCOVERY §66).  The pool is the cell's; its best is re-selected on these runs.  **Descriptive**: unadjusted, not part of the Holm family above, which stays over all families.

| cell | metric | n (runs) | pool best | Blocks_warm_CMAES_JSO | Δ [CI95] wins | best other panobbgo |
|---|---|---|---|---|---|---|
| free/d2/b100/q4 | aocc_time | 144 | BoTorch_qLogEI 0.211 | 0.238 | +0.027 [+0.005,+0.050] 8/12 | RegimeGate_oracle 0.238 +0.027 [+0.005,+0.050] 8/12 |
| free/d2/b100/q16 | aocc_time | 144 | BoTorch_qLogEI 0.190 | 0.186 | -0.004 [-0.016,+0.008] 5/12 | RegimeGate_oracle 0.186 -0.004 [-0.016,+0.008] 5/12 |
| free/d2/b100/q64 | aocc_time | 144 | Optuna_TPE 0.135 | 0.118 | -0.017 [-0.022,-0.011] 1/12 | RegimeGate_oracle 0.118 -0.017 [-0.022,-0.011] 1/12 |
| free/d5/b100/q4 | aocc_time | 144 | TuRBO1 0.138 | 0.153 | +0.015 [+0.010,+0.020] 12/12 | RegimeGate_oracle 0.153 +0.015 [+0.010,+0.020] 12/12 |
| free/d5/b100/q16 | aocc_time | 144 | BoTorch_qLogEI 0.092 | 0.128 | +0.036 [+0.030,+0.043] 12/12 | RegimeGate_oracle 0.128 +0.036 [+0.030,+0.043] 12/12 |
| free/d5/b100/q64 | aocc_time | 144 | BoTorch_qLogEI 0.074 | 0.075 | +0.001 [-0.001,+0.003] 8/12 | RegimeGate_oracle 0.075 +0.001 [-0.001,+0.003] 8/12 |
| free/d10/b100/q4 | aocc_time | 144 | TuRBO1 0.108 | 0.098 | -0.010 [-0.015,-0.005] 1/12 | RegimeGate_oracle 0.116 +0.008 [+0.004,+0.012] 10/12 |
| free/d10/b100/q16 | aocc_time | 144 | TuRBO1 0.073 | 0.090 | +0.017 [+0.012,+0.023] 12/12 | RegimeGate_oracle 0.088 +0.015 [+0.009,+0.021] 12/12 |
| free/d10/b100/q64 | aocc_time | 144 | TuRBO1 0.035 | 0.057 | +0.022 [+0.021,+0.023] 12/12 | RegimeGate_oracle 0.055 +0.021 [+0.019,+0.022] 12/12 |

## Per family: Blocks_warm_CMAES_JSO − pool best (headline metric)

Δ per family (paired over seeds on that family's instances); the roadmap claim is *never much worse on any class*, so read the minimum of each row.

| cell | ackley | ellipsoid | rastrigin | rosenbrock | sharp_ridge |
|---|---|---|---|---|---|
| free/d2/b100/q4 | +0.077 [+0.047,+0.107] 11/12 | -0.004 [-0.042,+0.034] 4/12 | +0.022 [-0.033,+0.076] 6/12 | +0.011 [-0.034,+0.057] 6/12 | -0.001 [-0.030,+0.028] 7/12 |
| free/d2/b100/q16 | -0.006 [-0.029,+0.017] 6/12 | -0.037 [-0.050,-0.023] 0/12 | -0.002 [-0.014,+0.010] 6/12 | +0.005 [-0.020,+0.030] 6/12 | -0.014 [-0.038,+0.011] 4/12 |
| free/d2/b100/q64 | -0.029 [-0.040,-0.019] 0/12 | -0.025 [-0.037,-0.012] 1/12 | -0.004 [-0.010,+0.003] 4/12 | -0.009 [-0.023,+0.005] 3/12 | -0.011 [-0.023,+0.001] 2/12 |
| free/d5/b100/q4 | +0.030 [+0.017,+0.044] 11/12 | -0.002 [-0.014,+0.010] 6/12 | +0.003 [-0.008,+0.013] 9/12 | +0.027 [+0.010,+0.044] 10/12 | -0.000 [-0.015,+0.014] 6/12 |
| free/d5/b100/q16 | +0.050 [+0.035,+0.065] 11/12 | +0.005 [+0.002,+0.007] 11/12 | -0.006 [-0.010,-0.003] 1/12 | +0.083 [+0.075,+0.091] 12/12 | +0.017 [+0.006,+0.029] 9/12 |
| free/d5/b100/q64 | -0.019 [-0.024,-0.014] 0/12 | +0.000 [-0.000,+0.001] 1/12 | -0.012 [-0.016,-0.007] 1/12 | +0.048 [+0.044,+0.052] 12/12 | -0.014 [-0.018,-0.011] 0/12 |
| free/d10/b100/q4 | -0.042 [-0.058,-0.027] 1/12 | +0.000 [+0.000,+0.000] 0/12 | -0.014 [-0.023,-0.005] 2/12 | +0.019 [+0.010,+0.027] 11/12 | -0.002 [-0.010,+0.007] 6/12 |
| free/d10/b100/q16 | +0.016 [-0.004,+0.036] 8/12 | +0.000 [+0.000,+0.000] 0/12 | -0.004 [-0.010,+0.002] 4/12 | +0.035 [+0.025,+0.044] 11/12 | +0.023 [+0.016,+0.031] 11/12 |
| free/d10/b100/q64 | +0.032 [+0.029,+0.035] 12/12 | +0.000 [+0.000,+0.000] 0/12 | +0.006 [+0.004,+0.008] 12/12 | +0.023 [+0.018,+0.029] 12/12 | +0.026 [+0.024,+0.028] 12/12 |

### free/d2/b100/q4 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs TuRBO1 | Δ aocc_time vs Optuna_TPE | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.217 | 0.215 | +0.021 [-0.002,+0.043] 8/12 | +0.043 [+0.018,+0.067] 11/12 | +0.050 [+0.030,+0.070] 12/12 | 0 | 0.2 |
| RegimeGate_oracle | 12/12 | 0.217 | 0.215 | +0.021 [-0.002,+0.043] 8/12 | +0.043 [+0.018,+0.067] 11/12 | +0.050 [+0.030,+0.070] 12/12 | 0 | 0.2 |
| BoTorch_qLogEI (pool) | 12/12 | 0.196 | 0.194 | – | – | – | 0 | 246.2 |
| RoundRobin_CMAES | 12/12 | 0.185 | 0.184 | -0.011 [-0.020,-0.001] 4/12 | +0.011 [-0.003,+0.026] 9/12 | +0.018 [+0.012,+0.024] 12/12 | 0 | 0.1 |
| TuRBO1 (pool) | 12/12 | 0.206 | 0.172 | – | – | – | 0 | 26.3 |
| Optuna_TPE (pool) | 12/12 | 0.166 | 0.165 | – | – | – | 0 | 0.7 |
| NGOpt (pool) | 12/12 | 0.155 | 0.154 | – | – | – | 0 | 2.5 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.405 | 0.153 | – | – | – | 0 | 0.4 |
| Optuna_CmaEs (pool) | 12/12 | 0.152 | 0.151 | – | – | – | 0 | 0.3 |
| pycma_IPOP (pool) | 12/12 | 0.181 | 0.147 | – | – | – | 0 | 0.1 |
| pycma_BIPOP (pool) | 12/12 | 0.173 | 0.141 | – | – | – | 0 | 0.1 |
| RoundRobin_Random | 12/12 | 0.131 | 0.130 | -0.064 [-0.073,-0.054] 0/12 | -0.042 [-0.055,-0.029] 0/12 | -0.035 [-0.042,-0.028] 0/12 | 0 | 0.1 |

### free/d2/b100/q16 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs TuRBO1 | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| BoTorch_qLogEI (pool) | 12/12 | 0.182 | 0.176 | – | – | – | 0 | 341.0 |
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.172 | 0.165 | -0.011 [-0.022,+0.001] 3/12 | +0.018 [+0.009,+0.027] 12/12 | +0.042 [+0.034,+0.049] 12/12 | 0 | 0.1 |
| RegimeGate_oracle | 12/12 | 0.172 | 0.165 | -0.011 [-0.022,+0.001] 3/12 | +0.018 [+0.009,+0.027] 12/12 | +0.042 [+0.034,+0.049] 12/12 | 0 | 0.1 |
| Optuna_TPE (pool) | 12/12 | 0.152 | 0.147 | – | – | – | 0 | 0.5 |
| RoundRobin_CMAES | 12/12 | 0.150 | 0.138 | -0.037 [-0.045,-0.030] 0/12 | -0.009 [-0.014,-0.003] 2/12 | +0.015 [+0.007,+0.022] 11/12 | 0 | 0.1 |
| TuRBO1 (pool) | 12/12 | 0.181 | 0.123 | – | – | – | 0 | 7.2 |
| Optuna_CmaEs (pool) | 12/12 | 0.123 | 0.119 | – | – | – | 0 | 0.2 |
| NGOpt (pool) | 12/12 | 0.125 | 0.119 | – | – | – | 0 | 1.5 |
| RoundRobin_Random | 12/12 | 0.121 | 0.117 | -0.059 [-0.068,-0.050] 0/12 | -0.030 [-0.038,-0.022] 0/12 | -0.007 [-0.014,+0.001] 4/12 | 0 | 0.1 |
| pycma_BIPOP (pool) | 12/12 | 0.140 | 0.099 | – | – | – | 0 | 0.0 |
| pycma_IPOP (pool) | 12/12 | 0.138 | 0.098 | – | – | – | 0 | 0.1 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.405 | 0.055 | – | – | – | 0 | 0.3 |

### free/d2/b100/q64 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs Optuna_CmaEs | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| BoTorch_qLogEI (pool) | 12/12 | 0.133 | 0.113 | – | – | – | 0 | 761.3 |
| Optuna_TPE (pool) | 12/12 | 0.129 | 0.110 | – | – | – | 0 | 0.3 |
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.145 | 0.097 | -0.015 [-0.021,-0.010] 0/12 | -0.013 [-0.018,-0.008] 1/12 | +0.003 [+0.000,+0.006] 10/12 | 0 | 0.1 |
| RegimeGate_oracle | 12/12 | 0.145 | 0.097 | -0.015 [-0.021,-0.010] 0/12 | -0.013 [-0.018,-0.008] 1/12 | +0.003 [+0.000,+0.006] 10/12 | 0 | 0.1 |
| Optuna_CmaEs (pool) | 12/12 | 0.108 | 0.094 | – | – | – | 0 | 0.2 |
| RoundRobin_Random | 12/12 | 0.114 | 0.089 | -0.023 [-0.030,-0.016] 0/12 | -0.021 [-0.026,-0.015] 0/12 | -0.005 [-0.012,+0.002] 5/12 | 0 | 0.1 |
| TuRBO1 (pool) | 12/12 | 0.159 | 0.089 | – | – | – | 0 | 1.8 |
| RoundRobin_CMAES | 12/12 | 0.146 | 0.082 | -0.031 [-0.037,-0.025] 0/12 | -0.028 [-0.035,-0.022] 0/12 | -0.012 [-0.019,-0.005] 1/12 | 0 | 0.1 |
| pycma_IPOP (pool) | 12/12 | 0.111 | 0.079 | – | – | – | 0 | 0.0 |
| pycma_BIPOP (pool) | 12/12 | 0.109 | 0.079 | – | – | – | 0 | 0.0 |
| NGOpt (pool) | 12/12 | 0.071 | 0.059 | – | – | – | 0 | 0.4 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.405 | 0.024 | – | – | – | 0 | 0.2 |

### free/d5/b100/q4 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time TuRBO1.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs TuRBO1 (pool best) | Δ aocc_time vs Optuna_CmaEs | Δ aocc_time vs BoTorch_qLogEI | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.125 | 0.124 | +0.011 [+0.007,+0.016] 11/12 | +0.039 [+0.033,+0.046] 12/12 | +0.044 [+0.036,+0.051] 12/12 | 0 | 0.5 |
| RegimeGate_oracle | 12/12 | 0.125 | 0.124 | +0.011 [+0.007,+0.016] 11/12 | +0.039 [+0.033,+0.046] 12/12 | +0.044 [+0.036,+0.051] 12/12 | 0 | 0.5 |
| TuRBO1 (pool) | 12/12 | 0.131 | 0.113 | – | – | – | 0 | 68.3 |
| RoundRobin_CMAES | 12/12 | 0.111 | 0.110 | -0.002 [-0.006,+0.002] 5/12 | +0.026 [+0.021,+0.030] 12/12 | +0.030 [+0.026,+0.034] 12/12 | 0 | 0.2 |
| Optuna_CmaEs (pool) | 12/12 | 0.085 | 0.085 | – | – | – | 0 | 0.8 |
| BoTorch_qLogEI (pool) | 12/12 | 0.081 | 0.081 | – | – | – | 0 | 1383.5 |
| Optuna_TPE (pool) | 12/12 | 0.080 | 0.080 | – | – | – | 0 | 3.3 |
| pycma_BIPOP (pool) | 12/12 | 0.094 | 0.074 | – | – | – | 0 | 0.2 |
| pycma_IPOP (pool) | 12/12 | 0.091 | 0.071 | – | – | – | 0 | 0.2 |
| NGOpt (pool) | 12/12 | 0.066 | 0.066 | – | – | – | 0 | 8.9 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.144 | 0.059 | – | – | – | 0 | 2.2 |
| RoundRobin_Random | 12/12 | 0.046 | 0.046 | -0.067 [-0.070,-0.064] 0/12 | -0.039 [-0.043,-0.036] 0/12 | -0.035 [-0.039,-0.031] 0/12 | 0 | 0.3 |

### free/d5/b100/q16 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs TuRBO1 | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.108 | 0.103 | +0.030 [+0.025,+0.035] 12/12 | +0.034 [+0.029,+0.038] 12/12 | +0.039 [+0.034,+0.043] 12/12 | 0 | 0.4 |
| RegimeGate_oracle | 12/12 | 0.108 | 0.103 | +0.030 [+0.025,+0.035] 12/12 | +0.034 [+0.029,+0.038] 12/12 | +0.039 [+0.034,+0.043] 12/12 | 0 | 0.4 |
| RoundRobin_CMAES | 12/12 | 0.085 | 0.079 | +0.005 [+0.003,+0.008] 12/12 | +0.009 [+0.006,+0.012] 11/12 | +0.014 [+0.012,+0.017] 12/12 | 0 | 0.2 |
| BoTorch_qLogEI (pool) | 12/12 | 0.075 | 0.074 | – | – | – | 0 | 1530.0 |
| Optuna_TPE (pool) | 12/12 | 0.071 | 0.070 | – | – | – | 0 | 3.0 |
| TuRBO1 (pool) | 12/12 | 0.111 | 0.065 | – | – | – | 0 | 25.5 |
| Optuna_CmaEs (pool) | 12/12 | 0.055 | 0.054 | – | – | – | 0 | 0.8 |
| NGOpt (pool) | 12/12 | 0.047 | 0.046 | – | – | – | 0 | 7.8 |
| RoundRobin_Random | 12/12 | 0.042 | 0.041 | -0.032 [-0.035,-0.030] 0/12 | -0.029 [-0.032,-0.025] 0/12 | -0.023 [-0.025,-0.022] 0/12 | 0 | 0.2 |
| pycma_BIPOP (pool) | 12/12 | 0.063 | 0.037 | – | – | – | 0 | 0.1 |
| pycma_IPOP (pool) | 12/12 | 0.064 | 0.037 | – | – | – | 0 | 0.1 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.144 | 0.027 | – | – | – | 0 | 1.9 |

### free/d5/b100/q64 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs TuRBO1 | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.075 | 0.060 | +0.001 [-0.001,+0.002] 8/12 | +0.015 [+0.013,+0.017] 12/12 | +0.023 [+0.022,+0.024] 12/12 | 0 | 0.3 |
| RegimeGate_oracle | 12/12 | 0.075 | 0.060 | +0.001 [-0.001,+0.002] 8/12 | +0.015 [+0.013,+0.017] 12/12 | +0.023 [+0.022,+0.024] 12/12 | 0 | 0.3 |
| BoTorch_qLogEI (pool) | 12/12 | 0.063 | 0.060 | – | – | – | 0 | 3132.2 |
| Optuna_TPE (pool) | 12/12 | 0.048 | 0.045 | – | – | – | 0 | 3.7 |
| RoundRobin_CMAES | 12/12 | 0.052 | 0.045 | -0.015 [-0.017,-0.013] 0/12 | -0.000 [-0.002,+0.001] 5/12 | +0.007 [+0.006,+0.008] 12/12 | 0 | 0.2 |
| TuRBO1 (pool) | 12/12 | 0.058 | 0.037 | – | – | – | 0 | 7.4 |
| Optuna_CmaEs (pool) | 12/12 | 0.039 | 0.037 | – | – | – | 0 | 1.1 |
| RoundRobin_Random | 12/12 | 0.038 | 0.035 | -0.025 [-0.027,-0.023] 0/12 | -0.011 [-0.012,-0.009] 0/12 | -0.003 [-0.004,-0.002] 0/12 | 0 | 0.2 |
| pycma_BIPOP (pool) | 12/12 | 0.038 | 0.030 | – | – | – | 0 | 0.1 |
| pycma_IPOP (pool) | 12/12 | 0.037 | 0.029 | – | – | – | 0 | 0.1 |
| NGOpt (pool) | 12/12 | 0.026 | 0.024 | – | – | – | 0 | 2.6 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.144 | 0.018 | – | – | – | 0 | 2.1 |

### free/d10/b100/q4 (headline: aocc_time)

Pool: NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC TuRBO1, aocc_time TuRBO1.  Planned 11 strategies, present 11.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs TuRBO1 (pool best) | Δ aocc_time vs Optuna_CmaEs | Δ aocc_time vs pycma_IPOP | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| RegimeGate_oracle | 12/12 | 0.093 | 0.093 | +0.006 [+0.003,+0.010] 10/12 | +0.026 [+0.024,+0.028] 12/12 | +0.040 [+0.036,+0.043] 12/12 | 0 | 0.7 |
| RoundRobin_CMAES | 12/12 | 0.093 | 0.093 | +0.006 [+0.003,+0.010] 11/12 | +0.026 [+0.024,+0.028] 12/12 | +0.040 [+0.036,+0.043] 12/12 | 0 | 0.5 |
| TuRBO1 (pool) | 12/12 | 0.098 | 0.086 | – | – | – | 0 | 193.3 |
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.079 | 0.079 | -0.008 [-0.012,-0.004] 1/12 | +0.012 [+0.007,+0.017] 11/12 | +0.025 [+0.021,+0.029] 12/12 | 0 | 1.0 |
| Optuna_CmaEs (pool) | 12/12 | 0.067 | 0.067 | – | – | – | 0 | 2.3 |
| pycma_IPOP (pool) | 12/12 | 0.066 | 0.053 | – | – | – | 0 | 0.3 |
| pycma_BIPOP (pool) | 12/12 | 0.064 | 0.051 | – | – | – | 0 | 0.3 |
| NGOpt (pool) | 12/12 | 0.048 | 0.048 | – | – | – | 0 | 2.2 |
| Optuna_TPE (pool) | 12/12 | 0.044 | 0.044 | – | – | – | 0 | 15.9 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.071 | 0.044 | – | – | – | 0 | 13.0 |
| RoundRobin_Random | 12/12 | 0.027 | 0.027 | -0.060 [-0.063,-0.057] 0/12 | -0.040 [-0.041,-0.039] 0/12 | -0.026 [-0.029,-0.024] 0/12 | 0 | 0.7 |

### free/d10/b100/q16 (headline: aocc_time)

Pool: NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC TuRBO1, aocc_time TuRBO1.  Planned 11 strategies, present 11.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs TuRBO1 (pool best) | Δ aocc_time vs NGOpt | Δ aocc_time vs Optuna_CmaEs | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.075 | 0.072 | +0.014 [+0.010,+0.018] 12/12 | +0.019 [+0.014,+0.023] 12/12 | +0.033 [+0.030,+0.037] 12/12 | 0 | 0.7 |
| RegimeGate_oracle | 12/12 | 0.073 | 0.070 | +0.012 [+0.007,+0.017] 12/12 | +0.017 [+0.015,+0.020] 12/12 | +0.032 [+0.028,+0.036] 12/12 | 0 | 0.6 |
| RoundRobin_CMAES | 12/12 | 0.070 | 0.064 | +0.006 [+0.004,+0.009] 12/12 | +0.011 [+0.008,+0.014] 12/12 | +0.026 [+0.024,+0.027] 12/12 | 0 | 0.4 |
| TuRBO1 (pool) | 12/12 | 0.087 | 0.058 | – | – | – | 0 | 87.7 |
| NGOpt (pool) | 12/12 | 0.054 | 0.053 | – | – | – | 0 | 1.5 |
| Optuna_CmaEs (pool) | 12/12 | 0.039 | 0.038 | – | – | – | 0 | 2.2 |
| Optuna_TPE (pool) | 12/12 | 0.036 | 0.036 | – | – | – | 0 | 13.4 |
| pycma_IPOP (pool) | 12/12 | 0.047 | 0.027 | – | – | – | 0 | 0.2 |
| pycma_BIPOP (pool) | 12/12 | 0.047 | 0.027 | – | – | – | 0 | 0.2 |
| RoundRobin_Random | 12/12 | 0.026 | 0.026 | -0.032 [-0.034,-0.030] 0/12 | -0.027 [-0.031,-0.024] 0/12 | -0.012 [-0.014,-0.011] 0/12 | 0 | 0.5 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.071 | 0.022 | – | – | – | 0 | 10.1 |

### free/d10/b100/q64 (headline: aocc_time)

Pool: NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time TuRBO1.  Planned 11 strategies, present 11.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs TuRBO1 (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs Optuna_CmaEs | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| **Blocks_warm_CMAES_JSO** | 12/12 | 0.054 | 0.045 | +0.017 [+0.016,+0.018] 12/12 | +0.018 [+0.017,+0.019] 12/12 | +0.018 [+0.017,+0.019] 12/12 | 0 | 0.8 |
| RegimeGate_oracle | 12/12 | 0.047 | 0.044 | +0.016 [+0.015,+0.018] 12/12 | +0.017 [+0.015,+0.018] 12/12 | +0.017 [+0.016,+0.019] 12/12 | 0 | 0.6 |
| RoundRobin_CMAES | 12/12 | 0.031 | 0.029 | +0.001 [+0.001,+0.002] 12/12 | +0.001 [+0.001,+0.002] 11/12 | +0.002 [+0.002,+0.003] 12/12 | 0 | 0.4 |
| TuRBO1 (pool) | 12/12 | 0.045 | 0.028 | – | – | – | 0 | 71.5 |
| Optuna_TPE (pool) | 12/12 | 0.028 | 0.028 | – | – | – | 0 | 17.7 |
| Optuna_CmaEs (pool) | 12/12 | 0.028 | 0.027 | – | – | – | 0 | 3.9 |
| RoundRobin_Random | 12/12 | 0.025 | 0.024 | -0.004 [-0.004,-0.003] 0/12 | -0.003 [-0.004,-0.003] 0/12 | -0.003 [-0.003,-0.002] 0/12 | 0 | 0.6 |
| pycma_IPOP (pool) | 12/12 | 0.027 | 0.023 | – | – | – | 0 | 0.2 |
| pycma_BIPOP (pool) | 12/12 | 0.027 | 0.023 | – | – | – | 0 | 0.2 |
| NGOpt (pool) | 12/12 | 0.021 | 0.021 | – | – | – | 0 | 10.4 |
| PyBOBYQA (sequential) (pool) | 12/12 | 0.071 | 0.018 | – | – | – | 0 | 13.0 |
