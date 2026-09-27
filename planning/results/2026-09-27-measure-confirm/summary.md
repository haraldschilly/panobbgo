# Expensive-track measurement

Commit 1778e1b, 58cf1e7; run 36274781342, 36305967615, 36313485264; 323 unit(s); missing 0, failed 0, unreadable files 0.
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

FP environments (per shard): 80ee2a0090c4: 57 shard(s).  Comparisons across jobs are ordinary samples; the environment is reproducibility metadata.

## Headline

| cell | metric | n (runs/plan) | pool best | Blocks_warm_CMAES_JSO | Δ [CI95] wins | p | p_holm | best other panobbgo | errors pb/ext | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| free/d2/b20/q1 | aocc | 75/75 | TuRBO1 0.145 (5/5) | 0.095 (5/5) | -0.051 [-0.073,-0.028] 0/5 | 0.003 | 0.043 | RegimeGate_oracle 0.095 -0.051 [-0.073,-0.028] 0/5 | 0/0 |  |
| free/d2/b20/q4 | aocc_time | 75/75 | BoTorch_qLogEI 0.100 (5/5) | 0.090 (5/5) | -0.011 [-0.022,+0.000] 1/5 | 0.055 | 0.328 | RegimeGate_oracle 0.090 -0.011 [-0.022,+0.000] 1/5 | 0/0 |  |
| free/d2/b20/q16 | aocc_time | 75/75 | BoTorch_qLogEI 0.067 (5/5) | 0.063 (5/5) | -0.003 [-0.010,+0.003] 1/5 | 0.245 | 1.000 | RegimeGate_oracle 0.063 -0.003 [-0.010,+0.003] 1/5 | 0/0 |  |
| free/d2/b100/q1 | aocc | 75/75 | PyBOBYQA (sequential) 0.440 (5/5) | 0.226 (5/5) | -0.214 [-0.262,-0.166] 0/5 | 0.000 | 0.005 | RegimeGate_oracle 0.226 -0.214 [-0.262,-0.166] 0/5 | 0/0 |  |
| free/d2/b100/q4 | aocc_time | 75/75 | BoTorch_qLogEI 0.209 (5/5) | 0.203 (5/5) | -0.006 [-0.037,+0.026] 2/5 | 0.656 | 1.000 | RegimeGate_oracle 0.203 -0.006 [-0.037,+0.026] 2/5 | 0/0 |  |
| free/d2/b100/q16 | aocc_time | 75/75 | BoTorch_qLogEI 0.175 (5/5) | 0.159 (5/5) | -0.016 [-0.028,-0.005] 0/5 | 0.017 | 0.136 | RegimeGate_oracle 0.159 -0.016 [-0.028,-0.005] 0/5 | 0/0 |  |
| free/d2/b100/q64 | aocc_time | 75/75 | BoTorch_qLogEI 0.113 (5/5) | 0.099 (5/5) | -0.013 [-0.021,-0.006] 0/5 | 0.009 | 0.079 | RegimeGate_oracle 0.099 -0.013 [-0.021,-0.006] 0/5 | 0/0 |  |
| free/d5/b20/q1 | aocc | 75/75 | NGOpt 0.073 (5/5) | 0.044 (5/5) | -0.029 [-0.045,-0.013] 0/5 | 0.007 | 0.068 | RegimeGate_oracle 0.044 -0.029 [-0.045,-0.013] 0/5 | 0/0 |  |
| free/d5/b20/q4 | aocc_time | 75/75 | BoTorch_qLogEI 0.049 (5/5) | 0.044 (5/5) | -0.005 [-0.009,-0.001] 0/5 | 0.023 | 0.159 | RegimeGate_oracle 0.044 -0.005 [-0.009,-0.001] 0/5 | 0/0 |  |
| free/d5/b20/q16 | aocc_time | 75/75 | BoTorch_qLogEI 0.040 (5/5) | 0.035 (5/5) | -0.005 [-0.007,-0.004] 0/5 | 0.001 | 0.020 | RegimeGate_oracle 0.035 -0.005 [-0.007,-0.004] 0/5 | 0/0 |  |
| free/d5/b100/q1 | aocc | 75/75 | PyBOBYQA (sequential) 0.145 (5/5) | 0.131 (5/5) | -0.015 [-0.047,+0.017] 2/5 | 0.269 | 1.000 | RegimeGate_oracle 0.131 -0.015 [-0.047,+0.017] 2/5 | 0/0 |  |
| free/d5/b100/q4 | aocc_time | 75/75 | TuRBO1 0.112 (5/5) | 0.123 (5/5) | +0.011 [+0.006,+0.017] 5/5 | 0.005 | 0.060 | RegimeGate_oracle 0.123 +0.011 [+0.006,+0.017] 5/5 | 0/0 |  |
| free/d5/b100/q16 | aocc_time | 75/75 | BoTorch_qLogEI 0.072 (5/5) | 0.103 (5/5) | +0.031 [+0.021,+0.041] 5/5 | 0.001 | 0.015 | RegimeGate_oracle 0.103 +0.031 [+0.021,+0.041] 5/5 | 0/0 |  |
| free/d5/b100/q64 | aocc_time | 75/75 | BoTorch_qLogEI 0.062 (5/5) | 0.061 (5/5) | -0.001 [-0.006,+0.005] 2/5 | 0.715 | 1.000 | RegimeGate_oracle 0.061 -0.001 [-0.006,+0.005] 2/5 | 0/0 |  |
| free/d10/b20/q1 | aocc | 75/75 | NGOpt 0.051 (5/5) | 0.028 (5/5) | -0.023 [-0.030,-0.017] 0/5 | 0.001 | 0.010 | RegimeGate_oracle 0.029 -0.023 [-0.028,-0.017] 0/5 | 0/0 |  |
| free/d10/b20/q4 | aocc_time | 75/75 | BoTorch_qLogEI 0.034 (5/5) | 0.028 (5/5) | -0.007 [-0.009,-0.005] 0/5 | 0.001 | 0.010 | RoundRobin_CMAES 0.029 -0.006 [-0.007,-0.004] 0/5 | 0/0 |  |
| free/d10/b20/q16 | aocc_time | 75/75 | BoTorch_qLogEI 0.031 (5/5) | 0.026 (5/5) | -0.005 [-0.007,-0.003] 0/5 | 0.002 | 0.023 | RoundRobin_CMAES 0.026 -0.006 [-0.007,-0.004] 0/5 | 0/0 |  |
| free/d10/b100/q1 | aocc | 75/75 | TuRBO1 0.096 (5/5) | 0.086 (5/5) | -0.010 [-0.015,-0.006] 0/5 | 0.003 | 0.042 | RegimeGate_oracle 0.089 -0.007 [-0.011,-0.004] 0/5 | 0/0 |  |
| free/d10/b100/q4 | aocc_time | 75/75 | TuRBO1 0.085 (5/5) | 0.083 (5/5) | -0.002 [-0.012,+0.009] 2/5 | 0.640 | 1.000 | RegimeGate_oracle 0.093 +0.008 [+0.001,+0.015] 4/5 | 0/0 |  |
| free/d10/b100/q16 | aocc_time | 75/75 | TuRBO1 0.060 (5/5) | 0.073 (5/5) | +0.013 [+0.007,+0.019] 5/5 | 0.004 | 0.049 | RegimeGate_oracle 0.070 +0.010 [+0.002,+0.019] 5/5 | 0/0 |  |
| free/d10/b100/q64 | aocc_time | 75/75 | Optuna_TPE 0.028 (5/5) | 0.046 (5/5) | +0.018 [+0.016,+0.021] 5/5 | 0.000 | 0.001 | RegimeGate_oracle 0.044 +0.017 [+0.014,+0.019] 5/5 | 0/0 |  |

## Per family: Blocks_warm_CMAES_JSO − pool best (headline metric)

Δ per family (paired over seeds on that family's instances); the roadmap claim is *never much worse on any class*, so read the minimum of each row.

| cell | ackley | ellipsoid | rastrigin | rosenbrock | sharp_ridge |
|---|---|---|---|---|---|
| free/d2/b20/q1 | -0.070 [-0.111,-0.029] 0/5 | -0.064 [-0.089,-0.039] 0/5 | -0.019 [-0.067,+0.029] 2/5 | -0.039 [-0.114,+0.036] 2/5 | -0.061 [-0.089,-0.034] 0/5 |
| free/d2/b20/q4 | -0.030 [-0.062,+0.002] 1/5 | -0.018 [-0.051,+0.014] 1/5 | -0.000 [-0.014,+0.014] 4/5 | +0.017 [-0.028,+0.061] 3/5 | -0.022 [-0.031,-0.012] 0/5 |
| free/d2/b20/q16 | -0.010 [-0.022,+0.001] 1/5 | -0.007 [-0.030,+0.016] 1/5 | +0.002 [-0.011,+0.015] 4/5 | +0.007 [-0.030,+0.045] 2/5 | -0.008 [-0.028,+0.012] 1/5 |
| free/d2/b100/q1 | -0.198 [-0.284,-0.112] 0/5 | -0.655 [-0.722,-0.589] 0/5 | +0.015 [-0.030,+0.061] 4/5 | -0.247 [-0.462,-0.032] 0/5 | +0.017 [-0.082,+0.116] 1/5 |
| free/d2/b100/q4 | +0.038 [-0.061,+0.138] 3/5 | -0.028 [-0.055,-0.000] 1/5 | -0.066 [-0.098,-0.035] 0/5 | +0.030 [-0.091,+0.151] 3/5 | -0.002 [-0.069,+0.066] 1/5 |
| free/d2/b100/q16 | -0.002 [-0.042,+0.039] 3/5 | -0.051 [-0.093,-0.008] 0/5 | -0.020 [-0.055,+0.014] 1/5 | +0.009 [-0.046,+0.064] 3/5 | -0.018 [-0.024,-0.012] 0/5 |
| free/d2/b100/q64 | -0.034 [-0.054,-0.013] 0/5 | -0.015 [-0.045,+0.016] 2/5 | -0.005 [-0.020,+0.010] 2/5 | -0.002 [-0.044,+0.039] 3/5 | -0.011 [-0.026,+0.004] 1/5 |
| free/d5/b20/q1 | +0.002 [-0.034,+0.039] 3/5 | +0.000 [+0.000,+0.000] 0/5 | +0.004 [-0.025,+0.033] 4/5 | -0.063 [-0.090,-0.036] 0/5 | -0.088 [-0.140,-0.037] 0/5 |
| free/d5/b20/q4 | -0.021 [-0.034,-0.007] 0/5 | +0.000 [+0.000,+0.000] 0/5 | -0.001 [-0.016,+0.014] 2/5 | +0.018 [+0.014,+0.022] 5/5 | -0.021 [-0.033,-0.008] 0/5 |
| free/d5/b20/q16 | -0.008 [-0.014,-0.002] 0/5 | +0.000 [+0.000,+0.000] 0/5 | -0.003 [-0.010,+0.005] 2/5 | +0.005 [-0.002,+0.013] 4/5 | -0.022 [-0.030,-0.014] 0/5 |
| free/d5/b100/q1 | +0.101 [-0.010,+0.213] 4/5 | -0.116 [-0.208,-0.024] 0/5 | +0.042 [+0.027,+0.057] 5/5 | -0.064 [-0.149,+0.021] 1/5 | -0.037 [-0.085,+0.011] 1/5 |
| free/d5/b100/q4 | +0.025 [-0.008,+0.057] 5/5 | -0.004 [-0.018,+0.010] 1/5 | -0.008 [-0.023,+0.006] 1/5 | +0.023 [+0.005,+0.040] 4/5 | +0.021 [-0.000,+0.043] 4/5 |
| free/d5/b100/q16 | +0.052 [+0.027,+0.077] 5/5 | +0.004 [-0.001,+0.009] 3/5 | -0.010 [-0.019,-0.000] 1/5 | +0.086 [+0.063,+0.110] 5/5 | +0.023 [-0.007,+0.053] 4/5 |
| free/d5/b100/q64 | -0.016 [-0.024,-0.008] 0/5 | +0.000 [-0.000,+0.000] 1/5 | -0.007 [-0.017,+0.002] 1/5 | +0.044 [+0.031,+0.056] 5/5 | -0.025 [-0.035,-0.014] 0/5 |
| free/d10/b20/q1 | +0.013 [-0.007,+0.033] 4/5 | +0.000 [+0.000,+0.000] 0/5 | +0.002 [-0.003,+0.007] 3/5 | -0.045 [-0.063,-0.027] 0/5 | -0.086 [-0.117,-0.055] 0/5 |
| free/d10/b20/q4 | -0.020 [-0.026,-0.013] 0/5 | +0.000 [+0.000,+0.000] 0/5 | -0.004 [-0.009,+0.000] 0/5 | +0.001 [-0.001,+0.002] 1/5 | -0.011 [-0.014,-0.008] 0/5 |
| free/d10/b20/q16 | -0.012 [-0.017,-0.007] 0/5 | +0.000 [+0.000,+0.000] 0/5 | -0.003 [-0.007,+0.001] 1/5 | +0.000 [-0.000,+0.000] 2/5 | -0.011 [-0.016,-0.005] 0/5 |
| free/d10/b100/q1 | -0.034 [-0.074,+0.006] 1/5 | +0.000 [-0.000,+0.000] 1/5 | -0.029 [-0.044,-0.014] 0/5 | +0.011 [-0.004,+0.027] 4/5 | -0.000 [-0.013,+0.013] 2/5 |
| free/d10/b100/q4 | -0.022 [-0.066,+0.023] 2/5 | +0.001 [-0.001,+0.003] 1/5 | -0.014 [-0.028,+0.000] 1/5 | +0.027 [+0.017,+0.037] 5/5 | -0.001 [-0.009,+0.006] 2/5 |
| free/d10/b100/q16 | +0.019 [-0.000,+0.039] 4/5 | +0.000 [+0.000,+0.000] 0/5 | -0.006 [-0.019,+0.006] 1/5 | +0.029 [+0.022,+0.037] 5/5 | +0.021 [+0.007,+0.036] 5/5 |
| free/d10/b100/q64 | +0.036 [+0.030,+0.042] 5/5 | +0.000 [+0.000,+0.000] 0/5 | +0.006 [+0.002,+0.010] 5/5 | +0.024 [+0.017,+0.031] 5/5 | +0.026 [+0.023,+0.029] 5/5 |

### free/d2/b20/q1 (headline: aocc)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC TuRBO1, aocc_time TuRBO1.  Planned 13 strategies, present 13.

| strategy | seeds | AOCC | aocc_time | Δ aocc vs TuRBO1 (pool best) | Δ aocc vs PyBOBYQA (sequential) | Δ aocc vs NGOpt | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| TuRBO1 (pool) | 5/5 | 0.145 | 0.143 | – | – | – | 0 | 15.3 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.142 | 0.139 | – | – | – | 0 | 0.1 |
| NGOpt (pool) | 5/5 | 0.125 | 0.124 | – | – | – | 0 | 0.2 |
| SMAC_BB (q=1 only) (reference) | 5/5 | 0.115 | 0.113 | – | – | – | 0 | 26.9 |
| BoTorch_qLogEI (pool) | 5/5 | 0.114 | 0.113 | – | – | – | 0 | 21.9 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.095 | 0.093 | -0.051 [-0.073,-0.028] 0/5 | -0.047 [-0.070,-0.024] 0/5 | -0.031 [-0.059,-0.002] 0/5 | 0 | 0.1 |
| RegimeGate_oracle | 5/5 | 0.095 | 0.093 | -0.051 [-0.073,-0.028] 0/5 | -0.047 [-0.070,-0.024] 0/5 | -0.031 [-0.059,-0.002] 0/5 | 0 | 0.1 |
| RoundRobin_CMAES | 5/5 | 0.092 | 0.091 | -0.053 [-0.071,-0.034] 0/5 | -0.049 [-0.073,-0.026] 0/5 | -0.033 [-0.061,-0.004] 0/5 | 0 | 0.0 |
| Optuna_TPE (pool) | 5/5 | 0.089 | 0.087 | – | – | – | 0 | 0.1 |
| Optuna_CmaEs (pool) | 5/5 | 0.078 | 0.077 | – | – | – | 0 | 0.1 |
| pycma_BIPOP (pool) | 5/5 | 0.077 | 0.076 | – | – | – | 0 | 0.1 |
| RoundRobin_Random | 5/5 | 0.073 | 0.073 | -0.072 [-0.088,-0.056] 0/5 | -0.068 [-0.091,-0.045] 0/5 | -0.052 [-0.083,-0.020] 0/5 | 0 | 0.0 |
| pycma_IPOP (pool) | 5/5 | 0.071 | 0.070 | – | – | – | 0 | 0.1 |

### free/d2/b20/q4 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs TuRBO1 | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| BoTorch_qLogEI (pool) | 5/5 | 0.106 | 0.100 | – | – | – | 0 | 31.9 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.095 | 0.090 | -0.011 [-0.022,+0.000] 1/5 | +0.010 [-0.002,+0.023] 5/5 | +0.015 [-0.012,+0.041] 4/5 | 0 | 0.0 |
| RegimeGate_oracle | 5/5 | 0.095 | 0.090 | -0.011 [-0.022,+0.000] 1/5 | +0.010 [-0.002,+0.023] 5/5 | +0.015 [-0.012,+0.041] 4/5 | 0 | 0.0 |
| RoundRobin_CMAES | 5/5 | 0.090 | 0.084 | -0.016 [-0.026,-0.006] 0/5 | +0.005 [-0.003,+0.013] 4/5 | +0.010 [-0.007,+0.027] 4/5 | 0 | 0.0 |
| Optuna_TPE (pool) | 5/5 | 0.083 | 0.079 | – | – | – | 0 | 0.1 |
| TuRBO1 (pool) | 5/5 | 0.102 | 0.075 | – | – | – | 0 | 3.9 |
| RoundRobin_Random | 5/5 | 0.073 | 0.070 | -0.030 [-0.041,-0.019] 0/5 | -0.009 [-0.022,+0.003] 1/5 | -0.005 [-0.028,+0.019] 2/5 | 0 | 0.0 |
| NGOpt (pool) | 5/5 | 0.070 | 0.066 | – | – | – | 0 | 0.5 |
| Optuna_CmaEs (pool) | 5/5 | 0.068 | 0.066 | – | – | – | 0 | 0.1 |
| pycma_BIPOP (pool) | 5/5 | 0.077 | 0.064 | – | – | – | 0 | 0.0 |
| pycma_IPOP (pool) | 5/5 | 0.072 | 0.060 | – | – | – | 0 | 0.0 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.142 | 0.040 | – | – | – | 0 | 0.1 |

### free/d2/b20/q16 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs Optuna_CmaEs | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| BoTorch_qLogEI (pool) | 5/5 | 0.084 | 0.067 | – | – | – | 0 | 37.1 |
| Optuna_TPE (pool) | 5/5 | 0.080 | 0.065 | – | – | – | 0 | 0.1 |
| Optuna_CmaEs (pool) | 5/5 | 0.077 | 0.064 | – | – | – | 0 | 0.0 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.086 | 0.063 | -0.003 [-0.010,+0.003] 1/5 | -0.002 [-0.015,+0.012] 2/5 | -0.000 [-0.012,+0.011] 3/5 | 0 | 0.0 |
| RegimeGate_oracle | 5/5 | 0.086 | 0.063 | -0.003 [-0.010,+0.003] 1/5 | -0.002 [-0.015,+0.012] 2/5 | -0.000 [-0.012,+0.011] 3/5 | 0 | 0.0 |
| RoundRobin_Random | 5/5 | 0.075 | 0.061 | -0.005 [-0.008,-0.002] 0/5 | -0.004 [-0.017,+0.009] 1/5 | -0.002 [-0.013,+0.008] 2/5 | 0 | 0.0 |
| TuRBO1 (pool) | 5/5 | 0.085 | 0.060 | – | – | – | 0 | 0.9 |
| RoundRobin_CMAES | 5/5 | 0.095 | 0.057 | -0.010 [-0.023,+0.003] 1/5 | -0.009 [-0.018,+0.000] 0/5 | -0.007 [-0.015,+0.001] 0/5 | 0 | 0.0 |
| pycma_BIPOP (pool) | 5/5 | 0.069 | 0.050 | – | – | – | 0 | 0.0 |
| NGOpt (pool) | 5/5 | 0.061 | 0.050 | – | – | – | 0 | 0.2 |
| pycma_IPOP (pool) | 5/5 | 0.063 | 0.047 | – | – | – | 0 | 0.0 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.142 | 0.018 | – | – | – | 0 | 0.1 |

### free/d2/b100/q1 (headline: aocc)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time PyBOBYQA (sequential).  Planned 13 strategies, present 13.

| strategy | seeds | AOCC | aocc_time | Δ aocc vs PyBOBYQA (sequential) (pool best) | Δ aocc vs TuRBO1 | Δ aocc vs SMAC_BB (q=1 only) | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| PyBOBYQA (sequential) (pool) | 5/5 | 0.440 | 0.439 | – | – | – | 0 | 0.5 |
| TuRBO1 (pool) | 5/5 | 0.272 | 0.271 | – | – | – | 0 | 96.4 |
| SMAC_BB (q=1 only) (reference) | 3/5! | 0.244 | 0.243 | – | – | – | 0 | 157.0 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.226 | 0.226 | -0.214 [-0.262,-0.166] 0/5 | -0.046 [-0.058,-0.034] 0/5 | -0.016 [-0.088,+0.056] 1/3 | 0 | 0.2 |
| RegimeGate_oracle | 5/5 | 0.226 | 0.226 | -0.214 [-0.262,-0.166] 0/5 | -0.046 [-0.058,-0.034] 0/5 | -0.016 [-0.088,+0.056] 1/3 | 0 | 0.2 |
| BoTorch_qLogEI (pool) | 5/5 | 0.206 | 0.206 | – | – | – | 0 | 223.6 |
| RoundRobin_CMAES | 5/5 | 0.192 | 0.191 | -0.248 [-0.284,-0.213] 0/5 | -0.080 [-0.116,-0.045] 0/5 | -0.050 [-0.070,-0.031] 0/3 | 0 | 0.1 |
| Optuna_CmaEs (pool) | 5/5 | 0.182 | 0.182 | – | – | – | 0 | 0.3 |
| NGOpt (pool) | 5/5 | 0.181 | 0.180 | – | – | – | 0 | 2.8 |
| Optuna_TPE (pool) | 5/5 | 0.172 | 0.172 | – | – | – | 0 | 0.7 |
| pycma_BIPOP (pool) | 5/5 | 0.172 | 0.172 | – | – | – | 0 | 0.1 |
| pycma_IPOP (pool) | 5/5 | 0.167 | 0.167 | – | – | – | 0 | 0.1 |
| RoundRobin_Random | 5/5 | 0.128 | 0.129 | -0.311 [-0.344,-0.279] 0/5 | -0.143 [-0.169,-0.117] 0/5 | -0.115 [-0.145,-0.085] 0/3 | 0 | 0.1 |

### free/d2/b100/q4 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs TuRBO1 | Δ aocc_time vs Optuna_TPE | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| BoTorch_qLogEI (pool) | 5/5 | 0.211 | 0.209 | – | – | – | 0 | 255.8 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.206 | 0.203 | -0.006 [-0.037,+0.026] 2/5 | +0.030 [-0.019,+0.079] 4/5 | +0.036 [+0.009,+0.062] 5/5 | 0 | 0.2 |
| RegimeGate_oracle | 5/5 | 0.206 | 0.203 | -0.006 [-0.037,+0.026] 2/5 | +0.030 [-0.019,+0.079] 4/5 | +0.036 [+0.009,+0.062] 5/5 | 0 | 0.2 |
| RoundRobin_CMAES | 5/5 | 0.183 | 0.182 | -0.027 [-0.036,-0.018] 0/5 | +0.008 [-0.017,+0.033] 4/5 | +0.014 [-0.001,+0.029] 4/5 | 0 | 0.1 |
| TuRBO1 (pool) | 5/5 | 0.208 | 0.173 | – | – | – | 0 | 19.9 |
| Optuna_TPE (pool) | 5/5 | 0.170 | 0.168 | – | – | – | 0 | 0.7 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.440 | 0.167 | – | – | – | 0 | 0.5 |
| NGOpt (pool) | 5/5 | 0.165 | 0.163 | – | – | – | 0 | 2.9 |
| Optuna_CmaEs (pool) | 5/5 | 0.148 | 0.147 | – | – | – | 0 | 0.3 |
| pycma_BIPOP (pool) | 5/5 | 0.172 | 0.140 | – | – | – | 0 | 0.1 |
| pycma_IPOP (pool) | 5/5 | 0.168 | 0.132 | – | – | – | 0 | 0.1 |
| RoundRobin_Random | 5/5 | 0.124 | 0.123 | -0.086 [-0.099,-0.073] 0/5 | -0.050 [-0.080,-0.020] 0/5 | -0.045 [-0.059,-0.030] 0/5 | 0 | 0.1 |

### free/d2/b100/q16 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs NGOpt | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| BoTorch_qLogEI (pool) | 5/5 | 0.182 | 0.175 | – | – | – | 0 | 356.2 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.166 | 0.159 | -0.016 [-0.028,-0.005] 0/5 | +0.011 [-0.004,+0.026] 5/5 | +0.036 [+0.011,+0.060] 5/5 | 0 | 0.2 |
| RegimeGate_oracle | 5/5 | 0.166 | 0.159 | -0.016 [-0.028,-0.005] 0/5 | +0.011 [-0.004,+0.026] 5/5 | +0.036 [+0.011,+0.060] 5/5 | 0 | 0.2 |
| RoundRobin_CMAES | 5/5 | 0.162 | 0.149 | -0.026 [-0.038,-0.014] 0/5 | +0.001 [-0.014,+0.017] 2/5 | +0.026 [+0.008,+0.044] 5/5 | 0 | 0.1 |
| Optuna_TPE (pool) | 5/5 | 0.154 | 0.148 | – | – | – | 0 | 0.7 |
| NGOpt (pool) | 5/5 | 0.129 | 0.123 | – | – | – | 0 | 2.2 |
| TuRBO1 (pool) | 5/5 | 0.182 | 0.122 | – | – | – | 0 | 6.8 |
| RoundRobin_Random | 5/5 | 0.120 | 0.115 | -0.060 [-0.067,-0.053] 0/5 | -0.033 [-0.044,-0.021] 0/5 | -0.008 [-0.028,+0.012] 3/5 | 0 | 0.1 |
| Optuna_CmaEs (pool) | 5/5 | 0.121 | 0.115 | – | – | – | 0 | 0.3 |
| pycma_BIPOP (pool) | 5/5 | 0.131 | 0.094 | – | – | – | 0 | 0.1 |
| pycma_IPOP (pool) | 5/5 | 0.134 | 0.090 | – | – | – | 0 | 0.1 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.440 | 0.047 | – | – | – | 0 | 0.4 |

### free/d2/b100/q64 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs Optuna_CmaEs | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| BoTorch_qLogEI (pool) | 5/5 | 0.135 | 0.113 | – | – | – | 0 | 755.0 |
| Optuna_TPE (pool) | 5/5 | 0.130 | 0.109 | – | – | – | 0 | 0.5 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.149 | 0.099 | -0.013 [-0.021,-0.006] 0/5 | -0.010 [-0.017,-0.003] 0/5 | +0.003 [-0.007,+0.012] 2/5 | 0 | 0.1 |
| RegimeGate_oracle | 5/5 | 0.149 | 0.099 | -0.013 [-0.021,-0.006] 0/5 | -0.010 [-0.017,-0.003] 0/5 | +0.003 [-0.007,+0.012] 2/5 | 0 | 0.1 |
| Optuna_CmaEs (pool) | 5/5 | 0.112 | 0.097 | – | – | – | 0 | 0.3 |
| TuRBO1 (pool) | 5/5 | 0.159 | 0.092 | – | – | – | 0 | 2.0 |
| RoundRobin_Random | 5/5 | 0.114 | 0.088 | -0.025 [-0.034,-0.016] 0/5 | -0.021 [-0.029,-0.013] 0/5 | -0.009 [-0.018,+0.000] 1/5 | 0 | 0.1 |
| RoundRobin_CMAES | 5/5 | 0.149 | 0.084 | -0.029 [-0.031,-0.026] 0/5 | -0.025 [-0.028,-0.021] 0/5 | -0.013 [-0.023,-0.002] 0/5 | 0 | 0.1 |
| pycma_BIPOP (pool) | 5/5 | 0.106 | 0.076 | – | – | – | 0 | 0.0 |
| pycma_IPOP (pool) | 5/5 | 0.104 | 0.070 | – | – | – | 0 | 0.0 |
| NGOpt (pool) | 5/5 | 0.079 | 0.068 | – | – | – | 0 | 0.6 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.440 | 0.020 | – | – | – | 0 | 0.4 |

### free/d5/b20/q1 (headline: aocc)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC NGOpt, aocc_time NGOpt.  Planned 13 strategies, present 13.

| strategy | seeds | AOCC | aocc_time | Δ aocc vs NGOpt (pool best) | Δ aocc vs TuRBO1 | Δ aocc vs PyBOBYQA (sequential) | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| NGOpt (pool) | 5/5 | 0.073 | 0.072 | – | – | – | 0 | 0.3 |
| TuRBO1 (pool) | 5/5 | 0.071 | 0.070 | – | – | – | 0 | 32.5 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.054 | 0.053 | – | – | – | 0 | 0.4 |
| SMAC_BB (q=1 only) (reference) | 5/5 | 0.053 | 0.053 | – | – | – | 0 | 94.1 |
| BoTorch_qLogEI (pool) | 5/5 | 0.049 | 0.049 | – | – | – | 0 | 134.4 |
| Optuna_TPE (pool) | 5/5 | 0.046 | 0.045 | – | – | – | 0 | 0.4 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.044 | 0.044 | -0.029 [-0.045,-0.013] 0/5 | -0.026 [-0.034,-0.018] 0/5 | -0.009 [-0.014,-0.005] 0/5 | 0 | 0.1 |
| RegimeGate_oracle | 5/5 | 0.044 | 0.044 | -0.029 [-0.045,-0.013] 0/5 | -0.026 [-0.034,-0.018] 0/5 | -0.009 [-0.014,-0.005] 0/5 | 0 | 0.1 |
| RoundRobin_CMAES | 5/5 | 0.042 | 0.041 | -0.032 [-0.047,-0.017] 0/5 | -0.029 [-0.037,-0.022] 0/5 | -0.012 [-0.016,-0.008] 0/5 | 0 | 0.1 |
| Optuna_CmaEs (pool) | 5/5 | 0.036 | 0.035 | – | – | – | 0 | 0.2 |
| pycma_IPOP (pool) | 5/5 | 0.035 | 0.035 | – | – | – | 0 | 0.1 |
| pycma_BIPOP (pool) | 5/5 | 0.034 | 0.033 | – | – | – | 0 | 0.1 |
| RoundRobin_Random | 5/5 | 0.030 | 0.030 | -0.043 [-0.059,-0.027] 0/5 | -0.040 [-0.050,-0.030] 0/5 | -0.023 [-0.029,-0.017] 0/5 | 0 | 0.1 |

### free/d5/b20/q4 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC TuRBO1, aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs TuRBO1 | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| BoTorch_qLogEI (pool) | 5/5 | 0.050 | 0.049 | – | – | – | 0 | 196.4 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.045 | 0.044 | -0.005 [-0.009,-0.001] 0/5 | +0.004 [+0.002,+0.006] 5/5 | +0.004 [+0.001,+0.007] 5/5 | 0 | 0.1 |
| RegimeGate_oracle | 5/5 | 0.045 | 0.044 | -0.005 [-0.009,-0.001] 0/5 | +0.004 [+0.002,+0.006] 5/5 | +0.004 [+0.001,+0.007] 5/5 | 0 | 0.1 |
| Optuna_TPE (pool) | 5/5 | 0.041 | 0.040 | – | – | – | 0 | 0.4 |
| TuRBO1 (pool) | 5/5 | 0.054 | 0.040 | – | – | – | 0 | 9.5 |
| RoundRobin_CMAES | 5/5 | 0.040 | 0.039 | -0.009 [-0.014,-0.004] 0/5 | -0.001 [-0.004,+0.003] 3/5 | -0.000 [-0.002,+0.002] 3/5 | 0 | 0.1 |
| Optuna_CmaEs (pool) | 5/5 | 0.033 | 0.033 | – | – | – | 0 | 0.2 |
| pycma_IPOP (pool) | 5/5 | 0.035 | 0.032 | – | – | – | 0 | 0.0 |
| pycma_BIPOP (pool) | 5/5 | 0.034 | 0.031 | – | – | – | 0 | 0.0 |
| RoundRobin_Random | 5/5 | 0.030 | 0.030 | -0.019 [-0.023,-0.015] 0/5 | -0.010 [-0.013,-0.007] 0/5 | -0.010 [-0.012,-0.007] 0/5 | 0 | 0.1 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.054 | 0.025 | – | – | – | 0 | 0.4 |
| NGOpt (pool) | 5/5 | 0.025 | 0.024 | – | – | – | 0 | 1.4 |

### free/d5/b20/q16 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs TuRBO1 | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| BoTorch_qLogEI (pool) | 5/5 | 0.044 | 0.040 | – | – | – | 0 | 210.3 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.039 | 0.035 | -0.005 [-0.007,-0.004] 0/5 | +0.002 [-0.001,+0.005] 4/5 | +0.005 [+0.004,+0.007] 5/5 | 0 | 0.1 |
| RegimeGate_oracle | 5/5 | 0.039 | 0.035 | -0.005 [-0.007,-0.004] 0/5 | +0.002 [-0.001,+0.005] 4/5 | +0.005 [+0.004,+0.007] 5/5 | 0 | 0.1 |
| Optuna_TPE (pool) | 5/5 | 0.036 | 0.033 | – | – | – | 0 | 0.4 |
| RoundRobin_CMAES | 5/5 | 0.040 | 0.033 | -0.007 [-0.009,-0.005] 0/5 | -0.000 [-0.004,+0.004] 3/5 | +0.004 [+0.001,+0.007] 5/5 | 0 | 0.1 |
| TuRBO1 (pool) | 5/5 | 0.041 | 0.029 | – | – | – | 0 | 3.4 |
| Optuna_CmaEs (pool) | 5/5 | 0.031 | 0.029 | – | – | – | 0 | 0.2 |
| RoundRobin_Random | 5/5 | 0.030 | 0.028 | -0.012 [-0.015,-0.010] 0/5 | -0.005 [-0.008,-0.001] 0/5 | -0.001 [-0.005,+0.003] 1/5 | 0 | 0.1 |
| pycma_BIPOP (pool) | 5/5 | 0.031 | 0.026 | – | – | – | 0 | 0.0 |
| pycma_IPOP (pool) | 5/5 | 0.032 | 0.026 | – | – | – | 0 | 0.0 |
| NGOpt (pool) | 5/5 | 0.024 | 0.022 | – | – | – | 0 | 1.4 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.054 | 0.018 | – | – | – | 0 | 0.4 |

### free/d5/b100/q1 (headline: aocc)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time PyBOBYQA (sequential).  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc vs PyBOBYQA (sequential) (pool best) | Δ aocc vs TuRBO1 | Δ aocc vs NGOpt | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| PyBOBYQA (sequential) (pool) | 5/5 | 0.145 | 0.146 | – | – | – | 0 | 2.7 |
| TuRBO1 (pool) | 5/5 | 0.134 | 0.134 | – | – | – | 0 | 230.0 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.131 | 0.130 | -0.015 [-0.047,+0.017] 2/5 | -0.004 [-0.026,+0.018] 2/5 | +0.016 [-0.003,+0.036] 4/5 | 0 | 0.6 |
| RegimeGate_oracle | 5/5 | 0.131 | 0.130 | -0.015 [-0.047,+0.017] 2/5 | -0.004 [-0.026,+0.018] 2/5 | +0.016 [-0.003,+0.036] 4/5 | 0 | 0.6 |
| NGOpt (pool) | 5/5 | 0.115 | 0.114 | – | – | – | 0 | 10.2 |
| RoundRobin_CMAES | 5/5 | 0.111 | 0.111 | -0.034 [-0.063,-0.006] 0/5 | -0.023 [-0.027,-0.020] 0/5 | -0.003 [-0.006,-0.001] 1/5 | 0 | 0.3 |
| Optuna_CmaEs (pool) | 5/5 | 0.108 | 0.107 | – | – | – | 0 | 0.8 |
| pycma_IPOP (pool) | 5/5 | 0.094 | 0.093 | – | – | – | 0 | 0.2 |
| pycma_BIPOP (pool) | 5/5 | 0.093 | 0.093 | – | – | – | 0 | 0.2 |
| Optuna_TPE (pool) | 5/5 | 0.088 | 0.088 | – | – | – | 0 | 3.9 |
| BoTorch_qLogEI (pool) | 5/5 | 0.086 | 0.085 | – | – | – | 0 | 1599.3 |
| RoundRobin_Random | 5/5 | 0.045 | 0.045 | -0.100 [-0.131,-0.069] 0/5 | -0.089 [-0.092,-0.086] 0/5 | -0.069 [-0.075,-0.064] 0/5 | 0 | 0.3 |

### free/d5/b100/q4 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time TuRBO1.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs TuRBO1 (pool best) | Δ aocc_time vs Optuna_CmaEs | Δ aocc_time vs BoTorch_qLogEI | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.124 | 0.123 | +0.011 [+0.006,+0.017] 5/5 | +0.040 [+0.030,+0.049] 5/5 | +0.041 [+0.036,+0.045] 5/5 | 0 | 0.5 |
| RegimeGate_oracle | 5/5 | 0.124 | 0.123 | +0.011 [+0.006,+0.017] 5/5 | +0.040 [+0.030,+0.049] 5/5 | +0.041 [+0.036,+0.045] 5/5 | 0 | 0.5 |
| TuRBO1 (pool) | 5/5 | 0.128 | 0.112 | – | – | – | 0 | 55.6 |
| RoundRobin_CMAES | 5/5 | 0.111 | 0.110 | -0.002 [-0.014,+0.011] 1/5 | +0.027 [+0.017,+0.036] 5/5 | +0.028 [+0.019,+0.037] 5/5 | 0 | 0.2 |
| Optuna_CmaEs (pool) | 5/5 | 0.084 | 0.083 | – | – | – | 0 | 0.9 |
| BoTorch_qLogEI (pool) | 5/5 | 0.083 | 0.082 | – | – | – | 0 | 1615.4 |
| Optuna_TPE (pool) | 5/5 | 0.078 | 0.078 | – | – | – | 0 | 3.8 |
| pycma_BIPOP (pool) | 5/5 | 0.094 | 0.074 | – | – | – | 0 | 0.2 |
| pycma_IPOP (pool) | 5/5 | 0.094 | 0.074 | – | – | – | 0 | 0.2 |
| NGOpt (pool) | 5/5 | 0.072 | 0.071 | – | – | – | 0 | 10.0 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.145 | 0.059 | – | – | – | 0 | 2.7 |
| RoundRobin_Random | 5/5 | 0.043 | 0.043 | -0.069 [-0.079,-0.059] 0/5 | -0.041 [-0.047,-0.034] 0/5 | -0.040 [-0.048,-0.031] 0/5 | 0 | 0.3 |

### free/d5/b100/q16 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs TuRBO1 | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.108 | 0.103 | +0.031 [+0.021,+0.041] 5/5 | +0.034 [+0.022,+0.047] 5/5 | +0.037 [+0.026,+0.049] 5/5 | 0 | 0.5 |
| RegimeGate_oracle | 5/5 | 0.108 | 0.103 | +0.031 [+0.021,+0.041] 5/5 | +0.034 [+0.022,+0.047] 5/5 | +0.037 [+0.026,+0.049] 5/5 | 0 | 0.5 |
| RoundRobin_CMAES | 5/5 | 0.085 | 0.078 | +0.006 [+0.003,+0.009] 5/5 | +0.010 [+0.002,+0.018] 4/5 | +0.012 [+0.007,+0.018] 5/5 | 0 | 0.2 |
| BoTorch_qLogEI (pool) | 5/5 | 0.073 | 0.072 | – | – | – | 0 | 1862.4 |
| Optuna_TPE (pool) | 5/5 | 0.070 | 0.069 | – | – | – | 0 | 3.8 |
| TuRBO1 (pool) | 5/5 | 0.110 | 0.066 | – | – | – | 0 | 20.4 |
| Optuna_CmaEs (pool) | 5/5 | 0.054 | 0.053 | – | – | – | 0 | 1.0 |
| NGOpt (pool) | 5/5 | 0.043 | 0.042 | – | – | – | 0 | 9.8 |
| RoundRobin_Random | 5/5 | 0.042 | 0.042 | -0.030 [-0.035,-0.026] 0/5 | -0.027 [-0.035,-0.019] 0/5 | -0.024 [-0.030,-0.019] 0/5 | 0 | 0.3 |
| pycma_IPOP (pool) | 5/5 | 0.066 | 0.039 | – | – | – | 0 | 0.1 |
| pycma_BIPOP (pool) | 5/5 | 0.063 | 0.038 | – | – | – | 0 | 0.1 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.145 | 0.027 | – | – | – | 0 | 2.7 |

### free/d5/b100/q64 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs TuRBO1 | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| BoTorch_qLogEI (pool) | 5/5 | 0.065 | 0.062 | – | – | – | 0 | 3218.3 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.076 | 0.061 | -0.001 [-0.006,+0.005] 2/5 | +0.016 [+0.013,+0.018] 5/5 | +0.022 [+0.019,+0.025] 5/5 | 0 | 0.4 |
| RegimeGate_oracle | 5/5 | 0.076 | 0.061 | -0.001 [-0.006,+0.005] 2/5 | +0.016 [+0.013,+0.018] 5/5 | +0.022 [+0.019,+0.025] 5/5 | 0 | 0.4 |
| RoundRobin_CMAES | 5/5 | 0.053 | 0.046 | -0.016 [-0.021,-0.011] 0/5 | +0.000 [-0.002,+0.003] 3/5 | +0.007 [+0.005,+0.009] 5/5 | 0 | 0.2 |
| Optuna_TPE (pool) | 5/5 | 0.048 | 0.045 | – | – | – | 0 | 3.8 |
| TuRBO1 (pool) | 5/5 | 0.059 | 0.039 | – | – | – | 0 | 7.8 |
| Optuna_CmaEs (pool) | 5/5 | 0.039 | 0.037 | – | – | – | 0 | 1.2 |
| RoundRobin_Random | 5/5 | 0.039 | 0.036 | -0.026 [-0.031,-0.020] 0/5 | -0.009 [-0.011,-0.008] 0/5 | -0.003 [-0.006,-0.000] 0/5 | 0 | 0.3 |
| pycma_IPOP (pool) | 5/5 | 0.038 | 0.030 | – | – | – | 0 | 0.1 |
| pycma_BIPOP (pool) | 5/5 | 0.038 | 0.030 | – | – | – | 0 | 0.1 |
| NGOpt (pool) | 5/5 | 0.027 | 0.026 | – | – | – | 0 | 2.8 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.145 | 0.019 | – | – | – | 0 | 2.4 |

### free/d10/b20/q1 (headline: aocc)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC NGOpt, aocc_time NGOpt.  Planned 13 strategies, present 13.

| strategy | seeds | AOCC | aocc_time | Δ aocc vs NGOpt (pool best) | Δ aocc vs TuRBO1 | Δ aocc vs SMAC_BB (q=1 only) | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| NGOpt (pool) | 5/5 | 0.051 | 0.051 | – | – | – | 0 | 0.7 |
| TuRBO1 (pool) | 5/5 | 0.050 | 0.050 | – | – | – | 0 | 106.7 |
| SMAC_BB (q=1 only) (reference) | 5/5 | 0.040 | 0.040 | – | – | – | 0 | 1060.7 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.039 | 0.039 | – | – | – | 0 | 2.5 |
| BoTorch_qLogEI (pool) | 5/5 | 0.037 | 0.037 | – | – | – | 0 | 468.0 |
| Optuna_TPE (pool) | 5/5 | 0.029 | 0.029 | – | – | – | 0 | 1.6 |
| RegimeGate_oracle | 5/5 | 0.029 | 0.029 | -0.023 [-0.028,-0.017] 0/5 | -0.021 [-0.024,-0.018] 0/5 | -0.011 [-0.014,-0.009] 0/5 | 0 | 0.1 |
| RoundRobin_CMAES | 5/5 | 0.029 | 0.029 | -0.023 [-0.028,-0.017] 0/5 | -0.021 [-0.024,-0.018] 0/5 | -0.011 [-0.013,-0.010] 0/5 | 0 | 0.1 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.028 | 0.028 | -0.023 [-0.030,-0.017] 0/5 | -0.022 [-0.024,-0.019] 0/5 | -0.012 [-0.014,-0.010] 0/5 | 0 | 0.2 |
| Optuna_CmaEs (pool) | 5/5 | 0.027 | 0.027 | – | – | – | 0 | 0.4 |
| pycma_BIPOP (pool) | 5/5 | 0.025 | 0.025 | – | – | – | 0 | 0.1 |
| pycma_IPOP (pool) | 5/5 | 0.025 | 0.025 | – | – | – | 0 | 0.1 |
| RoundRobin_Random | 5/5 | 0.022 | 0.022 | -0.029 [-0.035,-0.024] 0/5 | -0.028 [-0.031,-0.025] 0/5 | -0.018 [-0.020,-0.016] 0/5 | 0 | 0.1 |

### free/d10/b20/q4 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC TuRBO1, aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs TuRBO1 | Δ aocc_time vs Optuna_TPE | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| BoTorch_qLogEI (pool) | 5/5 | 0.035 | 0.034 | – | – | – | 0 | 526.9 |
| TuRBO1 (pool) | 5/5 | 0.042 | 0.031 | – | – | – | 0 | 22.0 |
| RoundRobin_CMAES | 5/5 | 0.029 | 0.029 | -0.006 [-0.007,-0.004] 0/5 | -0.002 [-0.003,-0.000] 0/5 | +0.002 [+0.001,+0.002] 5/5 | 0 | 0.1 |
| RegimeGate_oracle | 5/5 | 0.029 | 0.029 | -0.006 [-0.008,-0.003] 0/5 | -0.002 [-0.004,+0.000] 1/5 | +0.002 [-0.000,+0.003] 4/5 | 0 | 0.2 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.028 | 0.028 | -0.007 [-0.009,-0.005] 0/5 | -0.003 [-0.005,-0.001] 0/5 | +0.000 [-0.001,+0.002] 4/5 | 0 | 0.2 |
| Optuna_TPE (pool) | 5/5 | 0.027 | 0.027 | – | – | – | 0 | 1.7 |
| Optuna_CmaEs (pool) | 5/5 | 0.026 | 0.026 | – | – | – | 0 | 0.5 |
| pycma_BIPOP (pool) | 5/5 | 0.025 | 0.024 | – | – | – | 0 | 0.1 |
| pycma_IPOP (pool) | 5/5 | 0.025 | 0.024 | – | – | – | 0 | 0.1 |
| RoundRobin_Random | 5/5 | 0.022 | 0.022 | -0.012 [-0.014,-0.011] 0/5 | -0.009 [-0.010,-0.007] 0/5 | -0.005 [-0.007,-0.004] 0/5 | 0 | 0.1 |
| NGOpt (pool) | 5/5 | 0.020 | 0.020 | – | – | – | 0 | 1.9 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.039 | 0.020 | – | – | – | 0 | 2.5 |

### free/d10/b20/q16 (headline: aocc_time)

Pool: BoTorch_qLogEI, NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time BoTorch_qLogEI.  Planned 12 strategies, present 12.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs BoTorch_qLogEI (pool best) | Δ aocc_time vs Optuna_TPE | Δ aocc_time vs TuRBO1 | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| BoTorch_qLogEI (pool) | 5/5 | 0.033 | 0.031 | – | – | – | 0 | 645.5 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.028 | 0.026 | -0.005 [-0.007,-0.003] 0/5 | +0.002 [+0.001,+0.003] 5/5 | +0.003 [+0.002,+0.003] 5/5 | 0 | 0.2 |
| RoundRobin_CMAES | 5/5 | 0.027 | 0.026 | -0.006 [-0.007,-0.004] 0/5 | +0.001 [+0.001,+0.002] 5/5 | +0.002 [+0.001,+0.002] 5/5 | 0 | 0.1 |
| RegimeGate_oracle | 5/5 | 0.026 | 0.025 | -0.006 [-0.007,-0.006] 0/5 | +0.001 [-0.000,+0.002] 4/5 | +0.001 [+0.000,+0.003] 4/5 | 0 | 0.1 |
| Optuna_TPE (pool) | 5/5 | 0.025 | 0.024 | – | – | – | 0 | 1.7 |
| TuRBO1 (pool) | 5/5 | 0.031 | 0.024 | – | – | – | 0 | 6.1 |
| Optuna_CmaEs (pool) | 5/5 | 0.024 | 0.023 | – | – | – | 0 | 0.5 |
| pycma_IPOP (pool) | 5/5 | 0.024 | 0.021 | – | – | – | 0 | 0.1 |
| pycma_BIPOP (pool) | 5/5 | 0.024 | 0.021 | – | – | – | 0 | 0.1 |
| RoundRobin_Random | 5/5 | 0.022 | 0.021 | -0.010 [-0.011,-0.009] 0/5 | -0.003 [-0.004,-0.002] 0/5 | -0.003 [-0.003,-0.002] 0/5 | 0 | 0.1 |
| NGOpt (pool) | 5/5 | 0.020 | 0.019 | – | – | – | 0 | 1.8 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.039 | 0.018 | – | – | – | 0 | 2.5 |

### free/d10/b100/q1 (headline: aocc)

Pool: NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC TuRBO1, aocc_time TuRBO1.  Planned 11 strategies, present 11.

| strategy | seeds | AOCC | aocc_time | Δ aocc vs TuRBO1 (pool best) | Δ aocc vs Optuna_CmaEs | Δ aocc vs NGOpt | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| TuRBO1 (pool) | 5/5 | 0.096 | 0.096 | – | – | – | 0 | 575.8 |
| RegimeGate_oracle | 5/5 | 0.089 | 0.089 | -0.007 [-0.011,-0.004] 0/5 | +0.005 [-0.001,+0.011] 4/5 | +0.007 [+0.000,+0.014] 4/5 | 0 | 0.6 |
| RoundRobin_CMAES | 5/5 | 0.087 | 0.087 | -0.009 [-0.014,-0.005] 0/5 | +0.003 [-0.003,+0.008] 4/5 | +0.005 [-0.004,+0.014] 4/5 | 0 | 0.4 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.086 | 0.086 | -0.010 [-0.015,-0.006] 0/5 | +0.002 [-0.005,+0.008] 3/5 | +0.004 [-0.006,+0.014] 4/5 | 0 | 0.9 |
| Optuna_CmaEs (pool) | 5/5 | 0.084 | 0.084 | – | – | – | 0 | 1.9 |
| NGOpt (pool) | 5/5 | 0.082 | 0.082 | – | – | – | 0 | 10.7 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.074 | 0.074 | – | – | – | 0 | 11.3 |
| pycma_BIPOP (pool) | 5/5 | 0.070 | 0.070 | – | – | – | 0 | 0.3 |
| pycma_IPOP (pool) | 5/5 | 0.068 | 0.068 | – | – | – | 0 | 0.3 |
| Optuna_TPE (pool) | 5/5 | 0.046 | 0.046 | – | – | – | 0 | 14.4 |
| RoundRobin_Random | 5/5 | 0.026 | 0.026 | -0.070 [-0.073,-0.068] 0/5 | -0.058 [-0.061,-0.055] 0/5 | -0.056 [-0.062,-0.050] 0/5 | 0 | 0.7 |

### free/d10/b100/q4 (headline: aocc_time)

Pool: NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC TuRBO1, aocc_time TuRBO1.  Planned 11 strategies, present 11.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs TuRBO1 (pool best) | Δ aocc_time vs Optuna_CmaEs | Δ aocc_time vs NGOpt | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| RegimeGate_oracle | 5/5 | 0.093 | 0.093 | +0.008 [+0.001,+0.015] 4/5 | +0.026 [+0.019,+0.034] 5/5 | +0.035 [+0.030,+0.040] 5/5 | 0 | 0.5 |
| RoundRobin_CMAES | 5/5 | 0.092 | 0.092 | +0.007 [+0.003,+0.012] 5/5 | +0.026 [+0.020,+0.032] 5/5 | +0.034 [+0.026,+0.043] 5/5 | 0 | 0.3 |
| TuRBO1 (pool) | 5/5 | 0.096 | 0.085 | – | – | – | 0 | 204.1 |
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.083 | 0.083 | -0.002 [-0.012,+0.009] 2/5 | +0.017 [+0.003,+0.030] 5/5 | +0.025 [+0.011,+0.039] 5/5 | 0 | 0.7 |
| Optuna_CmaEs (pool) | 5/5 | 0.067 | 0.066 | – | – | – | 0 | 1.6 |
| NGOpt (pool) | 5/5 | 0.058 | 0.058 | – | – | – | 0 | 1.4 |
| pycma_BIPOP (pool) | 5/5 | 0.070 | 0.056 | – | – | – | 0 | 0.2 |
| pycma_IPOP (pool) | 5/5 | 0.068 | 0.055 | – | – | – | 0 | 0.2 |
| Optuna_TPE (pool) | 5/5 | 0.042 | 0.042 | – | – | – | 0 | 11.2 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.074 | 0.042 | – | – | – | 0 | 8.1 |
| RoundRobin_Random | 5/5 | 0.027 | 0.027 | -0.058 [-0.061,-0.055] 0/5 | -0.040 [-0.044,-0.035] 0/5 | -0.031 [-0.038,-0.024] 0/5 | 0 | 0.5 |

### free/d10/b100/q16 (headline: aocc_time)

Pool: NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC TuRBO1, aocc_time TuRBO1.  Planned 11 strategies, present 11.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs TuRBO1 (pool best) | Δ aocc_time vs NGOpt | Δ aocc_time vs Optuna_CmaEs | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.076 | 0.073 | +0.013 [+0.007,+0.019] 5/5 | +0.017 [+0.014,+0.021] 5/5 | +0.034 [+0.030,+0.037] 5/5 | 0 | 0.6 |
| RegimeGate_oracle | 5/5 | 0.073 | 0.070 | +0.010 [+0.002,+0.019] 5/5 | +0.015 [+0.006,+0.023] 5/5 | +0.031 [+0.020,+0.042] 5/5 | 0 | 0.5 |
| RoundRobin_CMAES | 5/5 | 0.070 | 0.064 | +0.004 [-0.003,+0.012] 3/5 | +0.009 [+0.005,+0.012] 5/5 | +0.025 [+0.022,+0.028] 5/5 | 0 | 0.3 |
| TuRBO1 (pool) | 5/5 | 0.088 | 0.060 | – | – | – | 0 | 97.6 |
| NGOpt (pool) | 5/5 | 0.056 | 0.056 | – | – | – | 0 | 1.2 |
| Optuna_CmaEs (pool) | 5/5 | 0.040 | 0.039 | – | – | – | 0 | 1.8 |
| Optuna_TPE (pool) | 5/5 | 0.036 | 0.036 | – | – | – | 0 | 11.5 |
| pycma_IPOP (pool) | 5/5 | 0.049 | 0.028 | – | – | – | 0 | 0.2 |
| pycma_BIPOP (pool) | 5/5 | 0.047 | 0.027 | – | – | – | 0 | 0.2 |
| RoundRobin_Random | 5/5 | 0.026 | 0.026 | -0.034 [-0.040,-0.028] 0/5 | -0.030 [-0.034,-0.026] 0/5 | -0.013 [-0.015,-0.012] 0/5 | 0 | 0.4 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.074 | 0.021 | – | – | – | 0 | 8.1 |

### free/d10/b100/q64 (headline: aocc_time)

Pool: NGOpt, Optuna_CmaEs, Optuna_TPE, PyBOBYQA (sequential), TuRBO1, pycma_BIPOP, pycma_IPOP.  Pool best: AOCC PyBOBYQA (sequential), aocc_time Optuna_TPE.  Planned 11 strategies, present 11.

| strategy | seeds | AOCC | aocc_time | Δ aocc_time vs Optuna_TPE (pool best) | Δ aocc_time vs TuRBO1 | Δ aocc_time vs Optuna_CmaEs | errors | s/run |
|---|---|---|---|---|---|---|---|---|
| **Blocks_warm_CMAES_JSO** | 5/5 | 0.055 | 0.046 | +0.018 [+0.016,+0.021] 5/5 | +0.018 [+0.017,+0.020] 5/5 | +0.019 [+0.018,+0.020] 5/5 | 0 | 0.8 |
| RegimeGate_oracle | 5/5 | 0.047 | 0.044 | +0.017 [+0.014,+0.019] 5/5 | +0.017 [+0.015,+0.018] 5/5 | +0.017 [+0.015,+0.019] 5/5 | 0 | 0.6 |
| RoundRobin_CMAES | 5/5 | 0.031 | 0.029 | +0.002 [+0.000,+0.003] 5/5 | +0.002 [+0.001,+0.003] 5/5 | +0.002 [+0.001,+0.003] 5/5 | 0 | 0.4 |
| Optuna_TPE (pool) | 5/5 | 0.028 | 0.028 | – | – | – | 0 | 18.4 |
| TuRBO1 (pool) | 5/5 | 0.046 | 0.028 | – | – | – | 0 | 74.8 |
| Optuna_CmaEs (pool) | 5/5 | 0.028 | 0.027 | – | – | – | 0 | 4.0 |
| RoundRobin_Random | 5/5 | 0.025 | 0.024 | -0.003 [-0.005,-0.002] 0/5 | -0.003 [-0.004,-0.002] 0/5 | -0.003 [-0.004,-0.002] 0/5 | 0 | 0.6 |
| pycma_BIPOP (pool) | 5/5 | 0.027 | 0.023 | – | – | – | 0 | 0.2 |
| pycma_IPOP (pool) | 5/5 | 0.027 | 0.023 | – | – | – | 0 | 0.2 |
| NGOpt (pool) | 5/5 | 0.021 | 0.021 | – | – | – | 0 | 10.4 |
| PyBOBYQA (sequential) (pool) | 5/5 | 0.074 | 0.018 | – | – | – | 0 | 13.1 |

## Calibration

Extra units run to calibrate the cost model (`--extra-units`): not part of the grid, not in any comparison above.

| unit | strategy | runs | unit min | s/run | AOCC | aocc_time | errors | FP |
|---|---|---|---|---|---|---|---|---|
| SMAC.free.b100.q1.d2.s3 | SMAC_BB (q=1 only) | 15 | 11.0 | 163.9 | 0.249 | 0.250 | 0 | 80ee2a0090c4 |
| SMAC.free.b100.q1.d2.s7 | SMAC_BB (q=1 only) | 15 | 10.6 | 159.8 | 0.249 | 0.248 | 0 | 80ee2a0090c4 |
