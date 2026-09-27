# TRQ collapses on the wide preset: diagnosis and the tabu-catchment fix (DISCOVERY §69)

**All in sample** (the wide/free presets and roster seeds 42/7/1234/2025/3
that exposed the problem), except the fresh-battery section.  Local,
`nice -n 10 ionice -c3`, at most 4 processes, `run_family_harness` as in
`scripts/measure.py run` (virtual clock, async, log-normal σ 0.5, CRN per
cell).  Metric: AOCC at q = 1, `aocc_time` at q = 4; mean over seeds ×
3 instances.  "before" = origin/master 8d4408f (`trust_region.py` loaded
from that commit), "after" = this branch.  Δ is paired over seeds, t-CI95,
unadjusted.

**RoundRobin_TRQ at q = 1 is seed-invariant** (box-centre start, the only
randomness is a restart point when the archive has no non-tabu point and
the space-filling fallback): its 5 "seeds" at q = 1 are 3 distinct runs
per family.  CIs at q = 1 only carry that residual randomness and are not
evidence of anything; the q = 1 rows are 3 instances per family.

Short names: RR_TRQ = RoundRobin_TRQ, Bl3_TRQ = Blocks_warm_CMAES_JSO_TRQ.

## Before / after, both presets (compact)

Wide preset, 100·d:

| spec | cell | before | after | Δ [CI95 over seeds] | wins | Δ ex-ell | families moved (> 0.01) |
|---|---|---|---|---|---|---|---|
| RR_TRQ | d2 100·d q1 | 0.443 | 0.475 | +0.032 [+0.019, +0.046] | 5/5 | +0.035 | sharp_ridge 0.293→0.213; step_ellipsoid 0.698→0.557; rastrigin 0.091→0.253; ackley 0.524→0.571; styblinski_tang_sep 0.084→0.546; gallagher21 0.304→0.277; levy_embed 0.830→0.889 |
| RR_TRQ | d2 100·d q4 | 0.396 | 0.410 | +0.015 [+0.004, +0.025] | 5/5 | +0.016 | step_ellipsoid 0.285→0.327; rastrigin 0.145→0.129; styblinski_tang_sep 0.593→0.711; gallagher21 0.275→0.301; levy_embed 0.704→0.749 |
| RR_TRQ | d5 100·d q1 | 0.274 | 0.276 | +0.002 [+0.002, +0.002] | 5/5 | +0.002 | sharp_ridge 0.227→0.150; ackley 0.153→0.192; gallagher21 0.136→0.239; levy_embed 0.225→0.173 |
| RR_TRQ | d5 100·d q4 | 0.241 | 0.243 | +0.003 [+0.001, +0.004] | 5/5 | +0.003 | styblinski_tang_sep 0.523→0.535; gallagher21 0.122→0.138 |
| Bl3_TRQ | d2 100·d q1 | 0.424 | 0.424 | +0.000 [-0.000, +0.001] | 2/3 | +0.000 | — |
| Bl3_TRQ | d2 100·d q4 | 0.390 | 0.390 | +0.000 [-0.000, +0.000] | 2/2 | +0.000 | — |
| Bl3_TRQ | d5 100·d q1 | 0.279 | 0.280 | +0.001 [-0.001, +0.002] | 4/5 | +0.001 | — |
| Bl3_TRQ | d5 100·d q4 | 0.257 | 0.257 | +0.000 [-0.000, +0.001] | 3/3 | +0.000 | — |

Free preset (§66 cells):

| spec | cell | before | after | Δ [CI95 over seeds] | wins | Δ ex-ell | families moved (> 0.01) |
|---|---|---|---|---|---|---|---|
| RR_TRQ | d2 20·d q1 | 0.303 | 0.303 | +0.000 [+0.000, +0.000] | 0/0 | +0.000 | — |
| RR_TRQ | d2 20·d q4 | 0.227 | 0.227 | +0.000 [+0.000, +0.000] | 0/0 | +0.000 | — |
| RR_TRQ | d2 100·d q1 | 0.529 | 0.545 | +0.016 [+0.016, +0.017] | 5/5 | +0.021 | rastrigin 0.091→0.253; sharp_ridge 0.285→0.206 |
| RR_TRQ | d2 100·d q4 | 0.453 | 0.451 | -0.003 [-0.012, +0.006] | 2/4 | -0.004 | rastrigin 0.145→0.129 |
| RR_TRQ | d5 20·d q1 | 0.183 | 0.183 | +0.000 [+0.000, +0.000] | 0/0 | +0.000 | — |
| RR_TRQ | d5 20·d q4 | 0.138 | 0.138 | +0.000 [+0.000, +0.000] | 0/0 | +0.000 | — |
| RR_TRQ | d5 100·d q1 | 0.365 | 0.357 | -0.008 [-0.008, -0.008] | 0/5 | -0.010 | ackley 0.156→0.193; sharp_ridge 0.227→0.150 |
| RR_TRQ | d5 100·d q4 | 0.331 | 0.329 | -0.001 [-0.004, +0.002] | 1/5 | -0.001 | — |
| RR_TRQ | d10 20·d q1 | 0.035 | 0.035 | +0.000 [+0.000, +0.000] | 0/0 | +0.000 | — |
| RR_TRQ | d10 20·d q4 | 0.031 | 0.031 | +0.000 [+0.000, +0.000] | 0/0 | +0.000 | — |
| RR_TRQ | d10 100·d q1 | 0.143 | 0.143 | +0.000 [+0.000, +0.000] | 5/5 | +0.000 | — |
| RR_TRQ | d10 100·d q4 | 0.155 | 0.155 | +0.000 [-0.000, +0.000] | 1/1 | +0.000 | — |
| Bl3_TRQ | d2 20·d q1 | 0.215 | 0.215 | +0.000 [+0.000, +0.000] | 0/0 | +0.000 | — |
| Bl3_TRQ | d2 20·d q4 | 0.134 | 0.134 | +0.000 [+0.000, +0.000] | 0/0 | +0.000 | — |
| Bl3_TRQ | d2 100·d q1 | 0.437 | 0.437 | -0.000 [-0.000, +0.000] | 0/1 | -0.000 | — |
| Bl3_TRQ | d2 100·d q4 | 0.408 | 0.408 | +0.000 [+0.000, +0.000] | 0/0 | +0.000 | — |
| Bl3_TRQ | d5 20·d q1 | 0.124 | 0.124 | +0.000 [+0.000, +0.000] | 0/0 | +0.000 | — |
| Bl3_TRQ | d5 20·d q4 | 0.091 | 0.091 | +0.000 [+0.000, +0.000] | 0/0 | +0.000 | — |
| Bl3_TRQ | d5 100·d q1 | 0.308 | 0.308 | +0.000 [-0.000, +0.000] | 1/2 | +0.000 | — |
| Bl3_TRQ | d5 100·d q4 | 0.294 | 0.294 | +0.000 [+0.000, +0.000] | 0/0 | +0.000 | — |
| Bl3_TRQ | d10 20·d q1 | 0.120 | 0.120 | +0.000 [+0.000, +0.000] | 0/0 | +0.000 | — |
| Bl3_TRQ | d10 20·d q4 | 0.058 | 0.058 | +0.000 [+0.000, +0.000] | 0/0 | +0.000 | — |
| Bl3_TRQ | d10 100·d q1 | 0.242 | 0.242 | +0.000 [-0.000, +0.000] | 1/4 | +0.000 | — |
| Bl3_TRQ | d10 100·d q4 | 0.201 | 0.201 | +0.000 [+0.000, +0.000] | 0/0 | +0.000 | — |

## Wide preset per family: RR_TRQ (before = oldW, after = newW)

## wide 100*d d=2 q=1  seeds=[3, 7, 42, 1234, 2025]
| family | oldW | newW |
|---|---|---|
| ellipsoid | 0.963 | 0.963 |
| different_powers | 0.538 | 0.538 |
| bent_cigar | 0.495 | 0.495 |
| rosenbrock | 0.620 | 0.620 |
| sharp_ridge | 0.293 | 0.213 |
| attractive_sector | 0.342 | 0.342 |
| step_ellipsoid | 0.698 | 0.557 |
| rastrigin | 0.091 | 0.253 |
| ackley | 0.524 | 0.571 |
| schwefel_sep | 0.000 | 0.010 |
| styblinski_tang_sep | 0.084 | 0.546 |
| lunacek_box | 0.126 | 0.120 |
| gallagher21 | 0.304 | 0.277 |
| levy_embed | 0.830 | 0.889 |
| rosenbrock_edge | 0.737 | 0.737 |
| **mean** | 0.443 | 0.475 |
| **mean ex-ellipsoid** | 0.406 | 0.441 |

## wide 100*d d=2 q=4  seeds=[3, 7, 42, 1234, 2025]
| family | oldW | newW |
|---|---|---|
| ellipsoid | 0.937 | 0.937 |
| different_powers | 0.483 | 0.483 |
| bent_cigar | 0.453 | 0.453 |
| rosenbrock | 0.428 | 0.428 |
| sharp_ridge | 0.136 | 0.138 |
| attractive_sector | 0.286 | 0.286 |
| step_ellipsoid | 0.285 | 0.327 |
| rastrigin | 0.145 | 0.129 |
| ackley | 0.592 | 0.589 |
| schwefel_sep | 0.000 | 0.006 |
| styblinski_tang_sep | 0.593 | 0.711 |
| lunacek_box | 0.124 | 0.124 |
| gallagher21 | 0.275 | 0.301 |
| levy_embed | 0.704 | 0.749 |
| rosenbrock_edge | 0.495 | 0.495 |
| **mean** | 0.396 | 0.410 |
| **mean ex-ellipsoid** | 0.357 | 0.373 |

## wide 100*d d=5 q=1  seeds=[3, 7, 42, 1234, 2025]
| family | oldW | newW |
|---|---|---|
| ellipsoid | 0.910 | 0.910 |
| different_powers | 0.509 | 0.509 |
| bent_cigar | 0.208 | 0.208 |
| rosenbrock | 0.367 | 0.367 |
| sharp_ridge | 0.227 | 0.150 |
| attractive_sector | 0.066 | 0.066 |
| step_ellipsoid | 0.111 | 0.117 |
| rastrigin | 0.065 | 0.065 |
| ackley | 0.153 | 0.192 |
| schwefel_sep | 0.000 | 0.000 |
| styblinski_tang_sep | 0.535 | 0.537 |
| lunacek_box | 0.058 | 0.063 |
| gallagher21 | 0.136 | 0.239 |
| levy_embed | 0.225 | 0.173 |
| rosenbrock_edge | 0.540 | 0.540 |
| **mean** | 0.274 | 0.276 |
| **mean ex-ellipsoid** | 0.229 | 0.230 |

## wide 100*d d=5 q=4  seeds=[3, 7, 42, 1234, 2025]
| family | oldW | newW |
|---|---|---|
| ellipsoid | 0.851 | 0.851 |
| different_powers | 0.431 | 0.431 |
| bent_cigar | 0.123 | 0.123 |
| rosenbrock | 0.274 | 0.274 |
| sharp_ridge | 0.136 | 0.136 |
| attractive_sector | 0.127 | 0.127 |
| step_ellipsoid | 0.077 | 0.081 |
| rastrigin | 0.067 | 0.069 |
| ackley | 0.265 | 0.271 |
| schwefel_sep | 0.000 | 0.000 |
| styblinski_tang_sep | 0.523 | 0.535 |
| lunacek_box | 0.058 | 0.058 |
| gallagher21 | 0.122 | 0.138 |
| levy_embed | 0.167 | 0.168 |
| rosenbrock_edge | 0.388 | 0.388 |
| **mean** | 0.241 | 0.243 |
| **mean ex-ellipsoid** | 0.197 | 0.200 |

## Wide preset per family: Bl3_TRQ (before = oldW, after = newW)

## wide 100*d d=2 q=1  seeds=[3, 7, 42, 1234, 2025]
| family | oldW | newW |
|---|---|---|
| ellipsoid | 0.879 | 0.879 |
| different_powers | 0.494 | 0.494 |
| bent_cigar | 0.415 | 0.415 |
| rosenbrock | 0.361 | 0.361 |
| sharp_ridge | 0.203 | 0.203 |
| attractive_sector | 0.372 | 0.372 |
| step_ellipsoid | 0.467 | 0.467 |
| rastrigin | 0.138 | 0.138 |
| ackley | 0.489 | 0.489 |
| schwefel_sep | 0.100 | 0.100 |
| styblinski_tang_sep | 0.559 | 0.565 |
| lunacek_box | 0.133 | 0.133 |
| gallagher21 | 0.293 | 0.293 |
| levy_embed | 0.904 | 0.904 |
| rosenbrock_edge | 0.551 | 0.551 |
| **mean** | 0.424 | 0.424 |
| **mean ex-ellipsoid** | 0.391 | 0.392 |

## wide 100*d d=2 q=4  seeds=[3, 7, 42, 1234, 2025]
| family | oldW | newW |
|---|---|---|
| ellipsoid | 0.820 | 0.820 |
| different_powers | 0.468 | 0.468 |
| bent_cigar | 0.300 | 0.300 |
| rosenbrock | 0.275 | 0.275 |
| sharp_ridge | 0.195 | 0.195 |
| attractive_sector | 0.377 | 0.377 |
| step_ellipsoid | 0.529 | 0.529 |
| rastrigin | 0.180 | 0.180 |
| ackley | 0.455 | 0.455 |
| schwefel_sep | 0.097 | 0.097 |
| styblinski_tang_sep | 0.484 | 0.484 |
| lunacek_box | 0.138 | 0.138 |
| gallagher21 | 0.288 | 0.288 |
| levy_embed | 0.877 | 0.877 |
| rosenbrock_edge | 0.370 | 0.370 |
| **mean** | 0.390 | 0.390 |
| **mean ex-ellipsoid** | 0.360 | 0.360 |

## wide 100*d d=5 q=1  seeds=[3, 7, 42, 1234, 2025]
| family | oldW | newW |
|---|---|---|
| ellipsoid | 0.846 | 0.846 |
| different_powers | 0.405 | 0.405 |
| bent_cigar | 0.132 | 0.132 |
| rosenbrock | 0.182 | 0.182 |
| sharp_ridge | 0.140 | 0.140 |
| attractive_sector | 0.175 | 0.175 |
| step_ellipsoid | 0.129 | 0.129 |
| rastrigin | 0.081 | 0.081 |
| ackley | 0.266 | 0.266 |
| schwefel_sep | 0.000 | 0.000 |
| styblinski_tang_sep | 0.435 | 0.441 |
| lunacek_box | 0.071 | 0.071 |
| gallagher21 | 0.158 | 0.160 |
| levy_embed | 0.774 | 0.774 |
| rosenbrock_edge | 0.392 | 0.392 |
| **mean** | 0.279 | 0.280 |
| **mean ex-ellipsoid** | 0.239 | 0.239 |

## wide 100*d d=5 q=4  seeds=[3, 7, 42, 1234, 2025]
| family | oldW | newW |
|---|---|---|
| ellipsoid | 0.828 | 0.828 |
| different_powers | 0.384 | 0.384 |
| bent_cigar | 0.141 | 0.141 |
| rosenbrock | 0.133 | 0.133 |
| sharp_ridge | 0.136 | 0.136 |
| attractive_sector | 0.145 | 0.145 |
| step_ellipsoid | 0.169 | 0.169 |
| rastrigin | 0.093 | 0.093 |
| ackley | 0.308 | 0.308 |
| schwefel_sep | 0.009 | 0.009 |
| styblinski_tang_sep | 0.237 | 0.240 |
| lunacek_box | 0.067 | 0.067 |
| gallagher21 | 0.172 | 0.172 |
| levy_embed | 0.762 | 0.762 |
| rosenbrock_edge | 0.276 | 0.276 |
| **mean** | 0.257 | 0.257 |
| **mean ex-ellipsoid** | 0.217 | 0.217 |

## Fresh battery (out of sample for the fix): wide, battery seed 20260927, RR_TRQ

q = 1: seed 42 only (seed-invariant); q = 4: seeds 42/7/1234.  oldO = before, newO = after.

## wide 100*d d=2 q=1  seeds=[42]
| family | oldO | newO |
|---|---|---|
| ellipsoid | 0.964 | 0.964 |
| different_powers | 0.555 | 0.555 |
| bent_cigar | 0.966 | 0.966 |
| rosenbrock | 0.745 | 0.745 |
| sharp_ridge | 0.161 | 0.150 |
| attractive_sector | 0.434 | 0.434 |
| step_ellipsoid | 0.337 | 0.211 |
| rastrigin | 0.392 | 0.393 |
| ackley | 0.497 | 0.554 |
| schwefel_sep | 0.000 | 0.000 |
| styblinski_tang_sep | 0.082 | 0.613 |
| lunacek_box | 0.112 | 0.120 |
| gallagher21 | 0.128 | 0.242 |
| levy_embed | 0.899 | 0.899 |
| rosenbrock_edge | 0.533 | 0.533 |
| **mean** | 0.454 | 0.492 |
| **mean ex-ellipsoid** | 0.417 | 0.458 |

## wide 100*d d=2 q=4  seeds=[7, 42, 1234]
| family | oldO | newO |
|---|---|---|
| ellipsoid | 0.928 | 0.928 |
| different_powers | 0.485 | 0.485 |
| bent_cigar | 0.846 | 0.846 |
| rosenbrock | 0.580 | 0.580 |
| sharp_ridge | 0.209 | 0.209 |
| attractive_sector | 0.420 | 0.420 |
| step_ellipsoid | 0.210 | 0.223 |
| rastrigin | 0.275 | 0.232 |
| ackley | 0.556 | 0.534 |
| schwefel_sep | 0.000 | 0.000 |
| styblinski_tang_sep | 0.163 | 0.387 |
| lunacek_box | 0.118 | 0.123 |
| gallagher21 | 0.291 | 0.314 |
| levy_embed | 0.707 | 0.713 |
| rosenbrock_edge | 0.399 | 0.399 |
| **mean** | 0.412 | 0.426 |
| **mean ex-ellipsoid** | 0.376 | 0.390 |

## wide 100*d d=5 q=1  seeds=[42]
| family | oldO | newO |
|---|---|---|
| ellipsoid | 0.868 | 0.868 |
| different_powers | 0.469 | 0.469 |
| bent_cigar | 0.145 | 0.145 |
| rosenbrock | 0.584 | 0.584 |
| sharp_ridge | 0.130 | 0.128 |
| attractive_sector | 0.145 | 0.145 |
| step_ellipsoid | 0.096 | 0.097 |
| rastrigin | 0.065 | 0.069 |
| ackley | 0.155 | 0.156 |
| schwefel_sep | 0.000 | 0.000 |
| styblinski_tang_sep | 0.339 | 0.638 |
| lunacek_box | 0.065 | 0.066 |
| gallagher21 | 0.085 | 0.108 |
| levy_embed | 0.425 | 0.441 |
| rosenbrock_edge | 0.254 | 0.254 |
| **mean** | 0.255 | 0.278 |
| **mean ex-ellipsoid** | 0.211 | 0.236 |

## wide 100*d d=5 q=4  seeds=[7, 42, 1234]
| family | oldO | newO |
|---|---|---|
| ellipsoid | 0.851 | 0.851 |
| different_powers | 0.421 | 0.421 |
| bent_cigar | 0.151 | 0.150 |
| rosenbrock | 0.311 | 0.315 |
| sharp_ridge | 0.140 | 0.138 |
| attractive_sector | 0.094 | 0.094 |
| step_ellipsoid | 0.101 | 0.102 |
| rastrigin | 0.060 | 0.061 |
| ackley | 0.185 | 0.183 |
| schwefel_sep | 0.000 | 0.000 |
| styblinski_tang_sep | 0.406 | 0.635 |
| lunacek_box | 0.067 | 0.067 |
| gallagher21 | 0.122 | 0.125 |
| levy_embed | 0.219 | 0.220 |
| rosenbrock_edge | 0.261 | 0.261 |
| **mean** | 0.226 | 0.242 |
| **mean ex-ellipsoid** | 0.181 | 0.198 |

## Diagnostic, not adopted: RR_TRQ with radius_init = 0.5 (after = newW/newF, r05 = the same with radius_init 0.5)

q = 1: one seed (seed-invariant), q = 4: 5 seeds.  The seeds line lists the after-run's seeds.
COBYQA's effective start radius is 0.5 of the box (§69.1).  In sample, a constant choice: a candidate only.

## wide 100*d d=2 q=1  seeds=[3, 7, 42, 1234, 2025]
| family | newW | r05W |
|---|---|---|
| ellipsoid | 0.963 | 0.970 |
| different_powers | 0.538 | 0.536 |
| bent_cigar | 0.495 | 0.515 |
| rosenbrock | 0.620 | 0.742 |
| sharp_ridge | 0.213 | 0.264 |
| attractive_sector | 0.342 | 0.305 |
| step_ellipsoid | 0.557 | 0.192 |
| rastrigin | 0.253 | 0.407 |
| ackley | 0.571 | 0.680 |
| schwefel_sep | 0.010 | 0.228 |
| styblinski_tang_sep | 0.546 | 0.625 |
| lunacek_box | 0.120 | 0.105 |
| gallagher21 | 0.277 | 0.209 |
| levy_embed | 0.889 | 0.958 |
| rosenbrock_edge | 0.737 | 0.770 |
| **mean** | 0.475 | 0.500 |
| **mean ex-ellipsoid** | 0.441 | 0.467 |

## wide 100*d d=2 q=4  seeds=[3, 7, 42, 1234, 2025]
| family | newW | r05W |
|---|---|---|
| ellipsoid | 0.937 | 0.941 |
| different_powers | 0.483 | 0.494 |
| bent_cigar | 0.453 | 0.585 |
| rosenbrock | 0.428 | 0.402 |
| sharp_ridge | 0.138 | 0.107 |
| attractive_sector | 0.286 | 0.260 |
| step_ellipsoid | 0.327 | 0.506 |
| rastrigin | 0.129 | 0.298 |
| ackley | 0.589 | 0.501 |
| schwefel_sep | 0.006 | 0.361 |
| styblinski_tang_sep | 0.711 | 0.636 |
| lunacek_box | 0.124 | 0.114 |
| gallagher21 | 0.301 | 0.228 |
| levy_embed | 0.749 | 0.917 |
| rosenbrock_edge | 0.495 | 0.688 |
| **mean** | 0.410 | 0.469 |
| **mean ex-ellipsoid** | 0.373 | 0.436 |

## wide 100*d d=5 q=1  seeds=[3, 7, 42, 1234, 2025]
| family | newW | r05W |
|---|---|---|
| ellipsoid | 0.910 | 0.931 |
| different_powers | 0.509 | 0.481 |
| bent_cigar | 0.208 | 0.091 |
| rosenbrock | 0.367 | 0.559 |
| sharp_ridge | 0.150 | 0.121 |
| attractive_sector | 0.066 | 0.086 |
| step_ellipsoid | 0.117 | 0.137 |
| rastrigin | 0.065 | 0.072 |
| ackley | 0.192 | 0.462 |
| schwefel_sep | 0.000 | 0.000 |
| styblinski_tang_sep | 0.537 | 0.332 |
| lunacek_box | 0.063 | 0.056 |
| gallagher21 | 0.239 | 0.151 |
| levy_embed | 0.173 | 0.919 |
| rosenbrock_edge | 0.540 | 0.526 |
| **mean** | 0.276 | 0.328 |
| **mean ex-ellipsoid** | 0.230 | 0.285 |

## wide 100*d d=5 q=4  seeds=[3, 7, 42, 1234, 2025]
| family | newW | r05W |
|---|---|---|
| ellipsoid | 0.851 | 0.818 |
| different_powers | 0.431 | 0.437 |
| bent_cigar | 0.123 | 0.137 |
| rosenbrock | 0.274 | 0.220 |
| sharp_ridge | 0.136 | 0.155 |
| attractive_sector | 0.127 | 0.105 |
| step_ellipsoid | 0.081 | 0.128 |
| rastrigin | 0.069 | 0.078 |
| ackley | 0.271 | 0.225 |
| schwefel_sep | 0.000 | 0.000 |
| styblinski_tang_sep | 0.535 | 0.474 |
| lunacek_box | 0.058 | 0.058 |
| gallagher21 | 0.138 | 0.171 |
| levy_embed | 0.168 | 0.874 |
| rosenbrock_edge | 0.388 | 0.494 |
| **mean** | 0.243 | 0.292 |
| **mean ex-ellipsoid** | 0.200 | 0.254 |

## free 20*d d=2 q=1  seeds=[3, 7, 42, 1234, 2025]
| family | newF | r05F |
|---|---|---|
| ellipsoid | 0.806 | 0.696 |
| rosenbrock | 0.170 | 0.181 |
| rastrigin | 0.077 | 0.253 |
| ackley | 0.324 | 0.265 |
| sharp_ridge | 0.136 | 0.149 |
| **mean** | 0.303 | 0.309 |
| **mean ex-ellipsoid** | 0.177 | 0.212 |

## free 20*d d=2 q=4  seeds=[3, 7, 42, 1234, 2025]
| family | newF | r05F |
|---|---|---|
| ellipsoid | 0.606 | 0.714 |
| rosenbrock | 0.124 | 0.120 |
| rastrigin | 0.093 | 0.122 |
| ackley | 0.212 | 0.170 |
| sharp_ridge | 0.099 | 0.073 |
| **mean** | 0.227 | 0.240 |
| **mean ex-ellipsoid** | 0.132 | 0.121 |

## free 20*d d=5 q=1  seeds=[3, 7, 42, 1234, 2025]
| family | newF | r05F |
|---|---|---|
| ellipsoid | 0.552 | 0.655 |
| rosenbrock | 0.083 | 0.091 |
| rastrigin | 0.049 | 0.058 |
| ackley | 0.141 | 0.169 |
| sharp_ridge | 0.092 | 0.076 |
| **mean** | 0.183 | 0.210 |
| **mean ex-ellipsoid** | 0.091 | 0.099 |

## free 20*d d=5 q=4  seeds=[3, 7, 42, 1234, 2025]
| family | newF | r05F |
|---|---|---|
| ellipsoid | 0.392 | 0.218 |
| rosenbrock | 0.059 | 0.034 |
| rastrigin | 0.046 | 0.053 |
| ackley | 0.141 | 0.149 |
| sharp_ridge | 0.050 | 0.047 |
| **mean** | 0.138 | 0.100 |
| **mean ex-ellipsoid** | 0.074 | 0.071 |

## free 20*d d=10 q=1  seeds=[3, 7, 42, 1234, 2025]
| family | newF | r05F |
|---|---|---|
| ellipsoid | 0.000 | 0.000 |
| rosenbrock | 0.005 | 0.000 |
| rastrigin | 0.019 | 0.026 |
| ackley | 0.136 | 0.155 |
| sharp_ridge | 0.017 | 0.036 |
| **mean** | 0.035 | 0.043 |
| **mean ex-ellipsoid** | 0.044 | 0.054 |

## free 20*d d=10 q=4  seeds=[3, 7, 42, 1234, 2025]
| family | newF | r05F |
|---|---|---|
| ellipsoid | 0.018 | 0.009 |
| rosenbrock | 0.000 | 0.004 |
| rastrigin | 0.014 | 0.014 |
| ackley | 0.118 | 0.136 |
| sharp_ridge | 0.004 | 0.012 |
| **mean** | 0.031 | 0.035 |
| **mean ex-ellipsoid** | 0.034 | 0.042 |

## free 100*d d=2 q=1  seeds=[3, 7, 42, 1234, 2025]
| family | newF | r05F |
|---|---|---|
| ellipsoid | 0.961 | 0.939 |
| rosenbrock | 0.633 | 0.700 |
| rastrigin | 0.253 | 0.407 |
| ackley | 0.675 | 0.666 |
| sharp_ridge | 0.206 | 0.290 |
| **mean** | 0.545 | 0.601 |
| **mean ex-ellipsoid** | 0.442 | 0.516 |

## free 100*d d=2 q=4  seeds=[3, 7, 42, 1234, 2025]
| family | newF | r05F |
|---|---|---|
| ellipsoid | 0.921 | 0.943 |
| rosenbrock | 0.444 | 0.509 |
| rastrigin | 0.129 | 0.298 |
| ackley | 0.605 | 0.524 |
| sharp_ridge | 0.153 | 0.146 |
| **mean** | 0.451 | 0.484 |
| **mean ex-ellipsoid** | 0.333 | 0.369 |

## free 100*d d=5 q=1  seeds=[3, 7, 42, 1234, 2025]
| family | newF | r05F |
|---|---|---|
| ellipsoid | 0.910 | 0.931 |
| rosenbrock | 0.467 | 0.597 |
| rastrigin | 0.065 | 0.072 |
| ackley | 0.193 | 0.302 |
| sharp_ridge | 0.150 | 0.121 |
| **mean** | 0.357 | 0.405 |
| **mean ex-ellipsoid** | 0.219 | 0.273 |

## free 100*d d=5 q=4  seeds=[3, 7, 42, 1234, 2025]
| family | newF | r05F |
|---|---|---|
| ellipsoid | 0.851 | 0.818 |
| rosenbrock | 0.330 | 0.279 |
| rastrigin | 0.069 | 0.078 |
| ackley | 0.262 | 0.303 |
| sharp_ridge | 0.136 | 0.155 |
| **mean** | 0.329 | 0.327 |
| **mean ex-ellipsoid** | 0.199 | 0.204 |

## free 100*d d=10 q=1  seeds=[3, 7, 42, 1234, 2025]
| family | newF | r05F |
|---|---|---|
| ellipsoid | 0.315 | 0.634 |
| rosenbrock | 0.060 | 0.070 |
| rastrigin | 0.026 | 0.043 |
| ackley | 0.144 | 0.168 |
| sharp_ridge | 0.173 | 0.120 |
| **mean** | 0.143 | 0.207 |
| **mean ex-ellipsoid** | 0.101 | 0.100 |

## free 100*d d=10 q=4  seeds=[3, 7, 42, 1234, 2025]
| family | newF | r05F |
|---|---|---|
| ellipsoid | 0.515 | 0.386 |
| rosenbrock | 0.025 | 0.037 |
| rastrigin | 0.023 | 0.030 |
| ackley | 0.124 | 0.158 |
| sharp_ridge | 0.086 | 0.083 |
| **mean** | 0.155 | 0.139 |
| **mean ex-ellipsoid** | 0.064 | 0.077 |
