540 labelled tasks, 90 instances, cells: ['d2 q1', 'd2 q4', 'd5 q1', 'd5 q4']

17 tasks tie at the best score (several arms at the same final value); their win is split equally between the tied arms in every win share below.

### Label distribution (regret vs the best arm of the task; score: AOCC at q = 1, aocc_time at q = 4)

| arm | mean score | mean regret | median | p90 | max | wins (ties split) | regret < 0.01 | EndedEarly |
|---|---|---|---|---|---|---|---|---|
| Blocks | 0.247 | 0.231 | 0.136 | 0.647 | 0.974 | 11% | 16% | 0% |
| RR_CMAES | 0.208 | 0.270 | 0.172 | 0.719 | 0.962 | 8% | 14% | 0% |
| RR_TRQ | 0.400 | 0.078 | 0.000 | 0.246 | 0.904 | 48% | 57% | 0% |
| RR_COBYQA | 0.283 | 0.195 | 0.104 | 0.586 | 0.962 | 17% | 22% | 83% |
| Blocks_TRQ | 0.349 | 0.129 | 0.068 | 0.367 | 0.904 | 15% | 22% | 0% |

Per cell, mean regret (wins):

| cell | Blocks | RR_CMAES | RR_TRQ | RR_COBYQA | Blocks_TRQ |
|---|---|---|---|---|---|
| d2 q1 | 0.303 (11%) | 0.355 (7%) | 0.106 (52%) | 0.171 (18%) | 0.171 (11%) |
| d2 q4 | 0.232 (13%) | 0.297 (8%) | 0.073 (57%) | 0.292 (4%) | 0.124 (17%) |
| d5 q1 | 0.221 (6%) | 0.245 (7%) | 0.081 (37%) | 0.131 (37%) | 0.127 (12%) |
| d5 q4 | 0.170 (13%) | 0.182 (11%) | 0.050 (47%) | 0.186 (8%) | 0.094 (20%) |

### Headroom: oracle vs single best arm

SBS = the arm with the best mean score over the tasks in scope (chosen in hindsight on the same tasks); gap = oracle mean − SBS mean = the SBS's mean regret.  CIs: cluster bootstrap over instances (2000 resamples); the second CI re-chooses the SBS in every resample.  Instance / family oracle *in sample*: the arm with the best mean over the instance's seeds (the family's tasks), the task's own score included.  *LOO*: the same pick without the task's own score — the other seeds of the instance (a mean over 2 seeds with 3), the other instances of the family (same d, q).  Only the LOO numbers are headroom a selector could reach.

| scope | tasks | SBS | SBS mean | task oracle | gap (task oracle) [CI] [CI, SBS re-chosen] | instance oracle gap: in sample / LOO | family oracle gap: in sample / LOO |
|---|---|---|---|---|---|---|---|
| all | 540 | RR_TRQ | 0.400 | 0.478 | **0.078** [0.057, 0.104] [0.057, 0.104] | 0.051 / **0.015** | 0.032 / **-0.003** |
| all, ex-ellipsoid | 504 | RR_TRQ | 0.364 | 0.447 | **0.083** [0.060, 0.108] [0.060, 0.108] | 0.055 / **0.016** | 0.034 / **-0.003** |
| d2 q1 | 135 | RR_TRQ | 0.511 | 0.617 | **0.106** [0.061, 0.161] [0.061, 0.156] | 0.072 / **0.022** | 0.039 / **-0.027** |
| d2 q4 | 135 | RR_TRQ | 0.476 | 0.549 | **0.073** [0.044, 0.111] [0.044, 0.110] | 0.041 / **-0.001** | 0.022 / **-0.005** |
| d5 q1 | 135 | RR_TRQ | 0.321 | 0.402 | **0.081** [0.052, 0.113] [0.052, 0.110] | 0.056 / **0.021** | 0.045 / **0.025** |
| d5 q4 | 135 | RR_TRQ | 0.294 | 0.344 | **0.050** [0.031, 0.070] [0.031, 0.070] | 0.036 / **0.018** | 0.022 / **-0.004** |

Context-only selector (the best arm per (d, q) cell, in hindsight): mean score 0.400, 0.000 above the global SBS; the task oracle is 0.078 above it.

### Best arm per family (mean score over dims, seeds, q; wins = share of tasks)

| family | best arm (mean score) | runner-up | spread best − worst | wins |
|---|---|---|---|---|
| ackley | RR_COBYQA (0.451) | Blocks_TRQ (0.394) | 0.163 | RR_TRQ 39%, RR_COBYQA 39%, Blocks_TRQ 11%, Blocks 11% |
| attractive_sector | Blocks_TRQ (0.264) | Blocks (0.229) | 0.082 | RR_TRQ 28%, Blocks_TRQ 28%, Blocks 19%, RR_CMAES 14%, RR_COBYQA 11% |
| bent_cigar | RR_TRQ (0.360) | Blocks_TRQ (0.242) | 0.302 | RR_TRQ 56%, Blocks_TRQ 31%, RR_COBYQA 14% |
| different_powers | RR_TRQ (0.496) | Blocks_TRQ (0.445) | 0.120 | RR_TRQ 58%, RR_COBYQA 28%, Blocks_TRQ 8%, Blocks 6% |
| ellipsoid | RR_TRQ (0.914) | Blocks_TRQ (0.802) | 0.871 | RR_TRQ 92%, Blocks_TRQ 8% |
| gallagher21 | RR_TRQ (0.323) | Blocks (0.266) | 0.135 | RR_TRQ 53%, RR_COBYQA 17%, Blocks 14%, RR_CMAES 11%, Blocks_TRQ 6% |
| levy_embed | RR_TRQ (0.948) | Blocks_TRQ (0.853) | 0.277 | RR_TRQ 92%, RR_COBYQA 6%, Blocks_TRQ 3% |
| lunacek_box | Blocks_TRQ (0.106) | RR_TRQ (0.098) | 0.016 | Blocks_TRQ 39%, RR_TRQ 28%, RR_COBYQA 17%, Blocks 11%, RR_CMAES 6% |
| rastrigin | Blocks_TRQ (0.209) | Blocks (0.149) | 0.092 | Blocks_TRQ 39%, RR_CMAES 19%, RR_TRQ 17%, RR_COBYQA 14%, Blocks 11% |
| rosenbrock | RR_TRQ (0.366) | RR_COBYQA (0.304) | 0.207 | RR_TRQ 58%, RR_COBYQA 28%, Blocks 8%, RR_CMAES 6% |
| rosenbrock_edge | RR_TRQ (0.631) | Blocks_TRQ (0.422) | 0.450 | RR_TRQ 72%, RR_COBYQA 19%, Blocks_TRQ 6%, RR_CMAES 3% |
| schwefel_sep | RR_COBYQA (0.166) | RR_TRQ (0.073) | 0.145 | RR_COBYQA 32%, Blocks 26%, RR_TRQ 21%, Blocks_TRQ 12%, RR_CMAES 9% |
| sharp_ridge | RR_TRQ (0.183) | Blocks (0.166) | 0.059 | RR_TRQ 31%, Blocks 28%, RR_COBYQA 22%, RR_CMAES 11%, Blocks_TRQ 8% |
| step_ellipsoid | Blocks (0.328) | Blocks_TRQ (0.307) | 0.175 | RR_CMAES 47%, Blocks 25%, RR_TRQ 17%, Blocks_TRQ 11% |
| styblinski_tang_sep | RR_TRQ (0.595) | Blocks_TRQ (0.503) | 0.383 | RR_TRQ 67%, Blocks_TRQ 19%, RR_COBYQA 8%, Blocks 6% |

### Features vs arms

Spearman ρ between a probe feature and an arm's regret over all tasks (negative = the arm does better where the feature is high); the three strongest |ρ| per arm, and per feature the arm whose regret it tracks most.

| arm | strongest features (ρ with regret) |
|---|---|
| Blocks | fr2_quad +0.30, flog_quad_gap -0.30, y_skew +0.28 |
| RR_CMAES | fr2_quad +0.28, flog_quad_gap -0.28, y_skew +0.27 |
| RR_TRQ | y_skew -0.13, hess_pos +0.12, flog10_cond -0.11 |
| RR_COBYQA | fr2_quad +0.28, flog_quad_gap -0.28, q +0.27 |
| Blocks_TRQ | r2_quad +0.16, r2_add +0.16, fdc +0.15 |

| feature | defined | Blocks | RR_CMAES | RR_TRQ | RR_COBYQA | Blocks_TRQ |
|---|---|---|---|---|---|---|
| dim | 100% | -0.15 | -0.22 | 0.05 | -0.23 | -0.13 |
| q | 100% | -0.13 | -0.12 | -0.08 | 0.27 | -0.11 |
| fdc | 100% | 0.14 | 0.10 | 0.01 | 0.03 | 0.15 |
| nbc_mean_ratio | 100% | -0.03 | -0.07 | 0.03 | -0.10 | -0.02 |
| nbc_sd_ratio | 100% | -0.06 | -0.10 | 0.08 | -0.06 | -0.03 |
| nbc_nn_nb_cor | 100% | -0.02 | -0.06 | 0.05 | -0.07 | 0.03 |
| nbc_dist_ratio_cv | 100% | 0.03 | 0.08 | -0.02 | 0.12 | 0.02 |
| nbc_nb_fitness_cor | 100% | 0.22 | 0.18 | -0.05 | 0.18 | 0.11 |
| disp_10 | 100% | 0.01 | 0.03 | 0.00 | -0.07 | -0.08 |
| disp_25 | 100% | -0.13 | -0.14 | -0.02 | -0.12 | -0.11 |
| r2_lin | 100% | 0.21 | 0.23 | -0.03 | 0.14 | 0.12 |
| r2_add | 100% | 0.19 | 0.20 | 0.04 | 0.05 | 0.16 |
| r2_quad | 100% | 0.18 | 0.17 | 0.05 | 0.05 | 0.16 |
| sep_ratio | 97% | 0.07 | 0.11 | 0.01 | -0.07 | 0.05 |
| log10_cond | 97% | -0.02 | -0.06 | -0.02 | -0.00 | -0.08 |
| hess_pos | 97% | -0.13 | -0.13 | 0.12 | -0.05 | -0.05 |
| fr2_lin | 100% | 0.23 | 0.25 | -0.00 | 0.19 | 0.10 |
| fr2_add | 100% | 0.23 | 0.24 | 0.01 | 0.13 | 0.14 |
| fr2_quad | 100% | 0.30 | 0.28 | -0.01 | 0.28 | 0.14 |
| flog_quad_gap | 100% | -0.30 | -0.28 | 0.01 | -0.28 | -0.14 |
| fsep_ratio | 98% | 0.02 | 0.01 | 0.07 | -0.11 | -0.02 |
| flog10_cond | 98% | 0.27 | 0.22 | -0.11 | 0.18 | 0.08 |
| fhess_pos | 98% | 0.03 | -0.01 | 0.11 | 0.09 | -0.08 |
| y_skew | 100% | 0.28 | 0.27 | -0.13 | 0.25 | 0.12 |
| y_kurt | 100% | 0.18 | 0.18 | -0.10 | 0.13 | 0.12 |
| y_ties | 100% | -0.00 | -0.06 | -0.00 | 0.03 | 0.01 |

Mean feature value by winning arm (tasks it wins, weighted by its win share):

| winner | tasks | flog_quad_gap | fr2_quad | r2_quad | fdc | nbc_nb_fitness_cor | disp_10 | y_ties | fsep_ratio | y_skew |
|---|---|---|---|---|---|---|---|---|---|---|
| Blocks | 59.4 | -1.11 | 0.71 | 0.66 | 0.45 | -0.53 | 0.72 | 0.00 | 0.83 | 0.61 |
| RR_CMAES | 45.4 | -1.39 | 0.80 | 0.72 | 0.54 | -0.50 | 0.68 | 0.00 | 0.84 | 0.86 |
| RR_TRQ | 261.4 | -2.41 | 0.82 | 0.76 | 0.55 | -0.50 | 0.72 | 0.00 | 0.85 | 1.18 |
| RR_COBYQA | 91.4 | -1.07 | 0.79 | 0.76 | 0.53 | -0.52 | 0.77 | 0.00 | 0.91 | 0.63 |
| Blocks_TRQ | 82.4 | -1.38 | 0.75 | 0.70 | 0.47 | -0.53 | 0.78 | 0.00 | 0.87 | 1.13 |

### Warm continuation vs the registry's cold start (diagnostic arms, not labelled)

| arm | cell | warm | cold | warm − cold [CI] | warm better |
|---|---|---|---|---|---|
| RR_CMAES | d2 q1 | 0.262 | 0.234 | +0.028 [+0.010, +0.045] | 61% |
| RR_CMAES | d2 q4 | 0.252 | 0.219 | +0.033 [+0.020, +0.046] | 63% |
| RR_CMAES | d5 q1 | 0.158 | 0.143 | +0.014 [+0.004, +0.026] | 60% |
| RR_CMAES | d5 q4 | 0.162 | 0.138 | +0.024 [+0.013, +0.036] | 73% |
| RR_COBYQA | d2 q1 | 0.446 | 0.414 | +0.031 [-0.008, +0.079] | 41% |
| RR_COBYQA | d2 q4 | 0.258 | 0.267 | -0.010 [-0.051, +0.021] | 36% |
| RR_COBYQA | d5 q1 | 0.271 | 0.268 | +0.003 [-0.032, +0.032] | 49% |
| RR_COBYQA | d5 q4 | 0.157 | 0.148 | +0.009 [-0.008, +0.024] | 49% |

