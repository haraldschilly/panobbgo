# Wide preset: local smoke and landscape characterisation (DISCOVERY §68)

Smoke: `scripts/measure.py run`, groups core + trq, preset `wide`, 100·d,
d 2/5, q 1/4, seeds 42/7/1234/2025/3, 4 processes, niced, laptop.  Metric:
AOCC at q = 1, `aocc_time` at q = 4; mean over 5 seeds × 3 instances;
**bold** = best arm of the row (unpaired, descriptive, in sample).  The
`schwefel_sep` rows come from a re-run of that family alone (`.f9` units)
after its base became `schwefel_box`; every other family was checked
bit-identical after that change (60/60 runs of
`core.wide.b100.q1.d2.s42.f0` and `.f14`).  Produced by `perfam.py`.

Short names: RR_CMA = RoundRobin_CMAES, Blocks = Blocks_warm_CMAES_JSO,
RR_TRQ = RoundRobin_TRQ, Bl3_TRQ = Blocks_warm_CMAES_JSO_TRQ, COBYQA =
RoundRobin_COBYQA (seed-invariant: one run per instance), IPOP =
Baseline_pycma_IPOP, NGOpt, OptCMA = Baseline_Optuna_CmaEs, TPE =
Baseline_Optuna_TPE, PyBOBYQA (sequential).

## Smoke results

#### d = 2, 100·d, q = 1 (AOCC, mean over 5 seeds x 3 instances)

| family | RR_CMA | Blocks | RR_TRQ | Bl3_TRQ | COBYQA | IPOP | NGOpt | OptCMA | TPE | PyBOBYQA | best | spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ellipsoid | 0.070 | 0.096 | **0.963** | 0.879 | 0.914 | 0.051 | 0.276 | 0.081 | 0.027 | 0.803 | RR_TRQ | 0.94 |
| different_powers | 0.369 | 0.445 | 0.538 | 0.494 | 0.545 | 0.354 | 0.385 | 0.365 | 0.355 | **0.551** | PyBOBYQA | 0.20 |
| bent_cigar | 0.075 | 0.120 | 0.495 | 0.415 | 0.303 | 0.084 | 0.144 | 0.090 | 0.064 | **0.524** | PyBOBYQA | 0.46 |
| rosenbrock | 0.188 | 0.240 | **0.620** | 0.361 | 0.418 | 0.200 | 0.194 | 0.214 | 0.250 | 0.464 | RR_TRQ | 0.43 |
| sharp_ridge | 0.199 | 0.166 | **0.293** | 0.203 | 0.065 | 0.165 | 0.174 | 0.170 | 0.166 | 0.163 | RR_TRQ | 0.23 |
| attractive_sector | 0.289 | 0.321 | 0.342 | **0.372** | 0.265 | 0.243 | 0.273 | 0.250 | 0.248 | 0.253 | Bl3_TRQ | 0.13 |
| step_ellipsoid | 0.388 | 0.435 | **0.698** | 0.467 | 0.176 | 0.349 | 0.342 | 0.376 | 0.313 | 0.252 | RR_TRQ | 0.52 |
| rastrigin | **0.178** | 0.155 | 0.091 | 0.138 | 0.170 | 0.154 | 0.114 | 0.160 | 0.161 | 0.140 | RR_CMA | 0.09 |
| ackley | 0.260 | 0.355 | 0.524 | 0.489 | 0.495 | 0.230 | 0.255 | 0.270 | 0.255 | **0.590** | PyBOBYQA | 0.36 |
| schwefel_sep | 0.026 | 0.044 | 0.000 | 0.100 | **0.304** | 0.050 | 0.000 | 0.013 | 0.118 | 0.126 | COBYQA | 0.30 |
| styblinski_tang_sep | 0.232 | 0.444 | 0.084 | 0.559 | **0.631** | 0.254 | 0.303 | 0.244 | 0.303 | 0.600 | COBYQA | 0.55 |
| lunacek_box | 0.125 | **0.138** | 0.126 | 0.133 | 0.117 | 0.132 | 0.112 | 0.127 | 0.126 | 0.111 | Blocks | 0.03 |
| gallagher21 | 0.212 | 0.301 | 0.304 | 0.293 | 0.169 | 0.200 | 0.211 | 0.177 | 0.299 | **0.345** | PyBOBYQA | 0.18 |
| levy_embed | 0.704 | 0.805 | 0.831 | 0.902 | **0.948** | 0.667 | 0.507 | 0.711 | 0.586 | 0.815 | COBYQA | 0.44 |
| rosenbrock_edge | 0.242 | 0.211 | **0.737** | 0.551 | 0.672 | 0.214 | 0.180 | 0.172 | 0.205 | 0.572 | RR_TRQ | 0.57 |
| **mean** | 0.237 | 0.285 | 0.443 | 0.424 | 0.413 | 0.223 | 0.231 | 0.228 | 0.232 | 0.421 | RR_TRQ | |
| mean ex-ellipsoid | 0.249 | 0.299 | 0.406 | 0.391 | 0.377 | 0.235 | 0.228 | 0.238 | 0.246 | 0.393 | RR_TRQ | |

families won: RR_TRQ 5, PyBOBYQA 4, COBYQA 3, Bl3_TRQ 1, RR_CMA 1, Blocks 1

#### d = 2, 100·d, q = 4 (aocc_time, mean over 5 seeds x 3 instances)

| family | RR_CMA | Blocks | RR_TRQ | Bl3_TRQ | COBYQA | IPOP | NGOpt | OptCMA | TPE | PyBOBYQA | best | spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ellipsoid | 0.070 | 0.063 | **0.937** | 0.820 | 0.637 | 0.030 | 0.188 | 0.049 | 0.029 | 0.276 | RR_TRQ | 0.91 |
| different_powers | 0.341 | 0.426 | **0.483** | 0.468 | 0.348 | 0.298 | 0.331 | 0.327 | 0.347 | 0.264 | RR_TRQ | 0.22 |
| bent_cigar | 0.081 | 0.142 | **0.453** | 0.300 | 0.117 | 0.056 | 0.052 | 0.047 | 0.056 | 0.347 | RR_TRQ | 0.41 |
| rosenbrock | 0.206 | 0.204 | **0.428** | 0.275 | 0.153 | 0.171 | 0.185 | 0.184 | 0.221 | 0.174 | RR_TRQ | 0.28 |
| sharp_ridge | 0.164 | 0.174 | 0.136 | **0.195** | 0.055 | 0.123 | 0.125 | 0.127 | 0.162 | 0.103 | Bl3_TRQ | 0.14 |
| attractive_sector | 0.242 | 0.250 | 0.286 | **0.377** | 0.115 | 0.188 | 0.200 | 0.197 | 0.232 | 0.095 | Bl3_TRQ | 0.28 |
| step_ellipsoid | 0.306 | 0.380 | 0.285 | **0.529** | 0.156 | 0.252 | 0.325 | 0.328 | 0.305 | 0.168 | Bl3_TRQ | 0.37 |
| rastrigin | 0.150 | 0.135 | 0.145 | **0.180** | 0.145 | 0.136 | 0.111 | 0.149 | 0.172 | 0.080 | Bl3_TRQ | 0.10 |
| ackley | 0.240 | 0.340 | **0.592** | 0.455 | 0.320 | 0.184 | 0.251 | 0.217 | 0.259 | 0.232 | RR_TRQ | 0.41 |
| schwefel_sep | 0.000 | 0.046 | 0.000 | 0.097 | **0.216** | 0.033 | 0.000 | 0.011 | 0.116 | 0.056 | COBYQA | 0.22 |
| styblinski_tang_sep | 0.190 | 0.388 | **0.593** | 0.484 | 0.427 | 0.196 | 0.267 | 0.205 | 0.263 | 0.311 | RR_TRQ | 0.40 |
| lunacek_box | 0.128 | 0.136 | 0.124 | **0.138** | 0.102 | 0.125 | 0.108 | 0.135 | 0.133 | 0.090 | Bl3_TRQ | 0.05 |
| gallagher21 | 0.194 | 0.251 | 0.275 | **0.288** | 0.159 | 0.190 | 0.193 | 0.203 | 0.285 | 0.180 | Bl3_TRQ | 0.13 |
| levy_embed | 0.663 | 0.761 | 0.706 | **0.883** | 0.769 | 0.554 | 0.501 | 0.577 | 0.601 | 0.582 | Bl3_TRQ | 0.38 |
| rosenbrock_edge | 0.218 | 0.162 | **0.495** | 0.370 | 0.124 | 0.178 | 0.108 | 0.139 | 0.202 | 0.117 | RR_TRQ | 0.39 |
| **mean** | 0.213 | 0.257 | 0.396 | 0.391 | 0.256 | 0.181 | 0.196 | 0.193 | 0.226 | 0.205 | RR_TRQ | |
| mean ex-ellipsoid | 0.223 | 0.271 | 0.357 | 0.360 | 0.229 | 0.192 | 0.197 | 0.203 | 0.240 | 0.200 | Bl3_TRQ | |

families won: RR_TRQ 7, Bl3_TRQ 7, COBYQA 1

#### d = 5, 100·d, q = 1 (AOCC, mean over 5 seeds x 3 instances)

| family | RR_CMA | Blocks | RR_TRQ | Bl3_TRQ | COBYQA | IPOP | NGOpt | OptCMA | TPE | PyBOBYQA | best | spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ellipsoid | 0.011 | 0.004 | **0.910** | 0.846 | 0.153 | 0.004 | 0.014 | 0.015 | 0.000 | 0.120 | RR_TRQ | 0.91 |
| different_powers | 0.338 | 0.368 | 0.509 | 0.405 | **0.512** | 0.306 | 0.306 | 0.323 | 0.301 | 0.491 | COBYQA | 0.21 |
| bent_cigar | 0.012 | 0.057 | **0.208** | 0.132 | 0.084 | 0.004 | 0.038 | 0.015 | 0.000 | 0.134 | RR_TRQ | 0.21 |
| rosenbrock | 0.103 | 0.124 | 0.367 | 0.182 | **0.435** | 0.069 | 0.097 | 0.099 | 0.095 | 0.189 | COBYQA | 0.37 |
| sharp_ridge | 0.095 | 0.136 | **0.227** | 0.140 | 0.181 | 0.083 | 0.115 | 0.090 | 0.063 | 0.173 | RR_TRQ | 0.16 |
| attractive_sector | 0.154 | 0.163 | 0.066 | 0.175 | **0.215** | 0.106 | 0.137 | 0.155 | 0.142 | 0.108 | COBYQA | 0.15 |
| step_ellipsoid | **0.219** | 0.164 | 0.111 | 0.129 | 0.101 | 0.185 | 0.202 | 0.214 | 0.182 | 0.053 | RR_CMA | 0.17 |
| rastrigin | 0.075 | **0.087** | 0.065 | 0.081 | 0.056 | 0.076 | 0.076 | 0.081 | 0.084 | 0.045 | Blocks | 0.04 |
| ackley | 0.261 | 0.298 | 0.153 | 0.266 | **0.655** | 0.230 | 0.269 | 0.250 | 0.203 | 0.233 | COBYQA | 0.50 |
| schwefel_sep | 0.000 | **0.010** | 0.000 | 0.000 | 0.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.000 | Blocks | 0.01 |
| styblinski_tang_sep | 0.176 | 0.206 | **0.535** | 0.435 | 0.066 | 0.109 | 0.250 | 0.166 | 0.126 | 0.419 | RR_TRQ | 0.47 |
| lunacek_box | 0.062 | 0.063 | 0.058 | 0.071 | **0.084** | 0.055 | 0.060 | 0.058 | 0.060 | 0.031 | COBYQA | 0.05 |
| gallagher21 | 0.133 | 0.148 | 0.136 | 0.158 | **0.244** | 0.133 | 0.151 | 0.181 | 0.142 | 0.153 | COBYQA | 0.11 |
| levy_embed | 0.564 | 0.672 | 0.177 | 0.811 | **0.934** | 0.559 | 0.692 | 0.567 | 0.521 | 0.381 | COBYQA | 0.76 |
| rosenbrock_edge | 0.088 | 0.095 | **0.540** | 0.392 | 0.472 | 0.111 | 0.128 | 0.050 | 0.059 | 0.334 | RR_TRQ | 0.49 |
| **mean** | 0.153 | 0.173 | 0.271 | 0.282 | 0.279 | 0.135 | 0.170 | 0.151 | 0.132 | 0.191 | Bl3_TRQ | |
| mean ex-ellipsoid | 0.163 | 0.185 | 0.225 | 0.241 | 0.289 | 0.145 | 0.181 | 0.161 | 0.141 | 0.196 | COBYQA | |

families won: COBYQA 7, RR_TRQ 5, Blocks 2, RR_CMA 1

#### d = 5, 100·d, q = 4 (aocc_time, mean over 5 seeds x 3 instances)

| family | RR_CMA | Blocks | RR_TRQ | Bl3_TRQ | COBYQA | IPOP | NGOpt | OptCMA | TPE | PyBOBYQA | best | spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ellipsoid | 0.006 | 0.003 | **0.851** | 0.828 | 0.011 | 0.001 | 0.009 | 0.004 | 0.000 | 0.006 | RR_TRQ | 0.85 |
| different_powers | 0.324 | 0.347 | **0.431** | 0.384 | 0.358 | 0.260 | 0.226 | 0.276 | 0.297 | 0.287 | RR_TRQ | 0.20 |
| bent_cigar | 0.012 | 0.040 | 0.123 | **0.141** | 0.024 | 0.000 | 0.012 | 0.001 | 0.000 | 0.041 | Bl3_TRQ | 0.14 |
| rosenbrock | 0.097 | 0.104 | **0.274** | 0.133 | 0.083 | 0.049 | 0.057 | 0.081 | 0.074 | 0.028 | RR_TRQ | 0.25 |
| sharp_ridge | 0.103 | 0.133 | **0.136** | 0.136 | 0.136 | 0.055 | 0.063 | 0.064 | 0.055 | 0.097 | RR_TRQ | 0.08 |
| attractive_sector | **0.148** | 0.145 | 0.127 | 0.145 | 0.092 | 0.076 | 0.092 | 0.105 | 0.129 | 0.035 | RR_CMA | 0.11 |
| step_ellipsoid | **0.193** | 0.181 | 0.077 | 0.169 | 0.088 | 0.146 | 0.122 | 0.179 | 0.175 | 0.022 | RR_CMA | 0.17 |
| rastrigin | 0.084 | 0.076 | 0.067 | **0.093** | 0.051 | 0.069 | 0.035 | 0.067 | 0.069 | 0.027 | Bl3_TRQ | 0.07 |
| ackley | 0.251 | 0.294 | 0.265 | 0.308 | **0.390** | 0.192 | 0.180 | 0.198 | 0.198 | 0.124 | COBYQA | 0.27 |
| schwefel_sep | **0.011** | 0.000 | 0.000 | 0.009 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | RR_CMA | 0.01 |
| styblinski_tang_sep | 0.158 | 0.201 | **0.523** | 0.237 | 0.056 | 0.077 | 0.146 | 0.079 | 0.111 | 0.128 | RR_TRQ | 0.47 |
| lunacek_box | 0.059 | 0.062 | 0.058 | **0.067** | 0.067 | 0.050 | 0.041 | 0.057 | 0.059 | 0.020 | Bl3_TRQ | 0.05 |
| gallagher21 | 0.132 | 0.162 | 0.122 | **0.172** | 0.158 | 0.127 | 0.118 | 0.145 | 0.113 | 0.082 | Bl3_TRQ | 0.09 |
| levy_embed | 0.573 | 0.661 | 0.176 | **0.784** | 0.729 | 0.489 | 0.305 | 0.524 | 0.504 | 0.205 | Bl3_TRQ | 0.61 |
| rosenbrock_edge | 0.088 | 0.105 | **0.388** | 0.276 | 0.124 | 0.064 | 0.019 | 0.022 | 0.041 | 0.011 | RR_TRQ | 0.38 |
| **mean** | 0.149 | 0.168 | 0.241 | 0.259 | 0.158 | 0.110 | 0.095 | 0.120 | 0.122 | 0.074 | Bl3_TRQ | |
| mean ex-ellipsoid | 0.160 | 0.179 | 0.198 | 0.218 | 0.168 | 0.118 | 0.101 | 0.128 | 0.130 | 0.079 | Bl3_TRQ | |

families won: RR_TRQ 6, Bl3_TRQ 5, RR_CMA 3, COBYQA 1

## Landscape characterisation (`charact.py 5 wide`)

Per family at d = 5, means over the 3 instances.  *1-R² quad*: residual
share of a full quadratic least-squares fit in f-scale, on max(1000, 40d²)
uniform box points, and on 400 points within ±1 of x_opt.  *FDC*: Spearman
correlation of f with the distance to x_opt (uniform points).
*Interaction*: median |f(x+h_i+h_j) − f(x+h_i) − f(x+h_j) + f(x)| /
(|Δ_i| + |Δ_j|), unit steps, 200 random pairs (0 = additive).  *Neutral*:
share of 0.05-sd Gaussian steps that leave f exactly unchanged.  *f(centre)
quantile*: share of uniform points better than the box centre (low = a
centre start is a head start).  *Local-search hits*: share of 20 L-BFGS-B
runs from uniform starts that reach f_opt.

| family | 1-R² quad (box) | 1-R² quad (r=1 at opt) | FDC | interaction | neutral | f(centre) quantile | local-search hits | ‖x_opt‖/(B√d) | f range |
|---|---|---|---|---|---|---|---|---|---|
| ellipsoid | 0.00 | 5.0e-28 | +0.43 | 0.06 | 0.00 | 0.40 | 1.00 | 0.44 | 9.4e+07 |
| different_powers | 0.09 | 2.1e-01 | +0.68 | 0.08 | 0.00 | 0.22 | 1.00 | 0.53 | 7.0e+02 |
| bent_cigar | 0.33 | 2.0e-02 | +0.70 | 0.11 | 0.00 | 0.26 | 1.00 | 0.54 | 2.8e+10 |
| rosenbrock | 0.13 | 1.9e-01 | +0.85 | 0.09 | 0.00 | 0.15 | 0.85 | 0.40 | 1.6e+06 |
| sharp_ridge | 0.03 | 2.6e-02 | +0.88 | 0.03 | 0.00 | 0.06 | 0.05 | 0.35 | 1.1e+03 |
| attractive_sector | 0.05 | 7.6e-02 | +0.47 | 0.11 | 0.00 | 0.19 | 0.92 | 0.40 | 1.1e+06 |
| step_ellipsoid | 0.00 | 1.0e-01 | +0.58 | 0.23 | 0.57 | 0.26 | 0.00 | 0.44 | 5.7e+03 |
| rastrigin | 0.16 | 9.3e-01 | +0.91 | 0.66 | 0.00 | 0.09 | 0.00 | 0.50 | 2.4e+02 |
| ackley | 0.07 | 4.0e-01 | +0.98 | 0.56 | 0.00 | 0.13 | 0.00 | 0.46 | 1.1e+01 |
| schwefel_sep | 0.74 | 1.0e-02 | -0.03 | 0.00 | 0.00 | 0.53 | 0.00 | 0.88 | 2.3e+03 |
| styblinski_tang_sep | 0.04 | 2.3e-02 | +0.62 | 0.00 | 0.00 | 0.18 | 0.17 | 0.46 | 8.8e+03 |
| lunacek_box | 0.14 | 9.3e-01 | +0.35 | 0.67 | 0.00 | 0.05 | 0.00 | 0.49 | 3.1e+02 |
| gallagher21 | 0.75 | 2.6e-01 | +0.35 | 0.23 | 0.00 | 0.36 | 0.07 | 0.46 | 8.3e+01 |
| levy_embed | 0.39 | 4.0e-02 | +0.41 | 0.30 | 0.00 | 0.45 | 0.50 | 0.57 | 6.5e+01 |
| rosenbrock_edge | 0.03 | 7.0e-02 | +0.83 | 0.05 | 0.00 | 0.35 | 0.58 | 0.80 | 5.7e+06 |
