# Regime table: which arm wins where, and is regime gating a better default?

2026-09-11.  Read-only re-analysis of every battery measured this week.
Question (DISCOVERY §38): the best arm flips with the regime — quantify it,
and decide whether *selection by regime* (dimension, budget per dimension,
noise class, constrained-or-not — all known before or in the first
evaluations) beats any fixed arm and beats the online portfolio.

**Short answer.** The flip is real and large: across 17 regime cells the
winner is CMA-ES in 6, jSO in 5, L-SHADE in 3, the warm portfolio in 3,
and the spread between best and worst arm inside one cell reaches 0.17
AOCC.  A regime-gated selector fitted on the data scores **0.5771** against
**0.5381** for the best fixed arm (CMA-ES) — +0.039.  Honestly estimated
(leave-one-seed-out) the gain shrinks to **+0.026 [−0.006, +0.057], 3/3
seeds** — *inside* the ±0.05 three-seed null floor and with a CI that
includes zero.  And on the **only** regime pair with twelve seeds behind
it, leave-one-seed-out gating is **negative** (−0.017 against the best
fixed arm, 5/12).  So: gating is the right *shape* of answer, the per-cell
flips that carry it are mostly 3-seed, and the headline gain is not yet
evidence.  Gating clearly beats the portfolio (+0.036 [+0.011, +0.062],
3/3) and clearly beats L-SHADE; against CMA-ES it does not clear the floor.

---

## 1. The long table, and what is paired with what

One row per (battery, dim, budget/dim, constrained, noise class, seed,
instance-or-family/instance, arm) → AOCC.  1932 unique measurement rows
after de-duplication, from these files:

| file | battery | dims | bpd | seeds | cells | arms | paired? |
|---|---|---|---|---|---|---|---|
| `2026-09-10/screen12.json` | MA-BBOB, noiseless | 2, 5 | 500 | 12 | 5 inst | CMA-ES, jSO, L-SHADE, portfolio (+ducb) | **yes** (`seed_name="screen"`) |
| `2026-09-10/roster2.json` | same stream, subset | 2, 5 | 500 | 12 | 5 | CMA-ES, jSO | yes — *byte-identical* to `screen12` where they overlap |
| `2026-09-11/noiseless.json` | same stream, subset | 2, 5 | 500 | 3 | 5 | 4 arms | yes — byte-identical to `screen12` |
| `2026-09-11/noisy_{gauss,unif,cauchy}.json` | MA-BBOB + BBOB noise | 2, 5 | 500 | 3 | 5 | 4 arms | yes, within file |
| `2026-09-11/hd10_bm500.json` | MA-BBOB | 10 | 500 | 3 | 3 | 4 arms | yes |
| `2026-09-11/hd10.json` | MA-BBOB | 10 | 2000 | 3 | 3 | 4 arms | yes |
| `2026-09-11/hd20.json` | MA-BBOB | 20 | 2000 | 3 | 3 | 4 arms | yes |
| `2026-09-11/noisy_hd.json` | MA-BBOB + gauss | 10 | 500 | 3 | 3 | 4 arms | yes |
| `2026-09-10/family_screen_free_s42-7-1234.json` | families, free | 2, 5, 10 | 500 | 3 | 5 fam × 3 inst | 4 arms | yes |
| `2026-09-10/family_screen_constrained_s42-7-1234.json` | families, constrained | 2, 5 | 500 | 3 | 4 fam × 3 inst | 4 arms | yes |

Excluded, with reasons:

* **`2026-09-10/oracle12b.json` is not paired across arms.**  `benchmarks/oracle.py`
  passes `seed_name=<arm key>`, so *each arm runs its own RNG stream*
  (`StrategySpec.rng_identity`).  The disagreement is not cosmetic: at
  (seed 42, d=2, inst 2) CMA-ES scores **0.3227** in `oracle12b` and
  **0.8427** on the screen stream.  It is kept out of the long table and
  used below only as an independent-stream cross-check and as the sole
  source for the LBC arm.
* **`2026-09-11/highdim.json`** is a different, older harness (arms
  `cmaes_only` / `lbc_only` / `portfolio` / `Baseline_SciPyDE`, budgets
  1000/1953/2000, git HEAD `42209c7`, numpy 2.4).  Not commensurate with
  the `*_alone` specs; excluded.
* `family_repro_*.json` — reproducibility evidence, not measurements
  (per the coordinator).
* `screen_p*.json`, `p[2-5]_*.json`, `sw_*.json`, `null_*.json`,
  `accept12*.json` — parameter sweeps and null runs over strategy
  variants, not arm-vs-arm-in-a-regime measurements.

**Pairing across batteries does not exist.**  Every battery with a
different `problem_kind` (noiseless vs each noise class) draws a different
RNG stream, so `noisy-gauss − noiseless` carries run-to-run variance;
only *within-file* arm deltas are paired.  Cross-regime aggregates below
therefore average *levels*, not paired deltas, and the LOSO comparison is
paired only through the shared seed.

**Null floor rule used throughout** (from §18a / §30):

* 12 seeds → **±0.02**;
* 3 seeds → **±0.05** if CMA-ES *or the portfolio* (a CMA-ES-containing
  spec) is on either side of the comparison, **±0.03** for DE-vs-DE.

A margin inside its floor is marked `INSIDE` and is a direction, not
evidence.

---

## 2. Per-regime table

Mean AOCC per arm over the seeds available in that regime (12 for the two
MA-BBOB noiseless 500·d cells, 3 everywhere else), averaged over the
regime's instance cells.

| regime (kind / noise / d / budget·d) | n seeds | cells | CMA-ES | jSO | L-SHADE | portfolio | winner | margin to runner-up | seed wins | vs floor |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| MA-BBOB / noiseless / d2 / 500 | 12 | 5 | 0.7205 | 0.7241 | **0.7326** | 0.7272 | L-SHADE | +0.0054 | 2/12 | INSIDE ±0.02 |
| MA-BBOB / noiseless / d5 / 500 | 12 | 5 | 0.6119 | 0.5739 | 0.5593 | **0.6436** | portfolio | +0.0317 | 8/12 | clear ±0.02 |
| MA-BBOB / noiseless / d10 / 500 | 3 | 3 | **0.4034** | 0.3750 | 0.3570 | 0.3971 | CMA-ES | +0.0063 | 1/3 | INSIDE ±0.05 |
| MA-BBOB / noiseless / d10 / 2000 | 3 | 3 | 0.5189 | 0.6003 | **0.6522** | 0.6229 | L-SHADE | +0.0293 | 0/3 | INSIDE ±0.05 |
| MA-BBOB / noiseless / d20 / 2000 | 3 | 3 | 0.4773 | 0.4138 | 0.3882 | **0.4803** | portfolio | +0.0030 | 1/3 | INSIDE ±0.05 |
| MA-BBOB / cauchy / d2 / 500 | 3 | 5 | **0.7195** | 0.5684 | 0.5782 | 0.6345 | CMA-ES | +0.0850 | 2/3 | clear ±0.05 |
| MA-BBOB / cauchy / d5 / 500 | 3 | 5 | **0.6018** | 0.4122 | 0.4318 | 0.4869 | CMA-ES | +0.1149 | 3/3 | clear ±0.05 |
| MA-BBOB / gauss / d2 / 500 | 3 | 5 | 0.6938 | **0.7665** | 0.7255 | 0.6613 | jSO | +0.0410 | 2/3 | clear ±0.03 |
| MA-BBOB / gauss / d5 / 500 | 3 | 5 | 0.5671 | 0.6264 | 0.5955 | **0.6970** | portfolio | +0.0706 | 3/3 | clear ±0.05 |
| MA-BBOB / gauss / d10 / 500 | 3 | 3 | **0.5000** | 0.3625 | 0.3502 | 0.4009 | CMA-ES | +0.0992 | 3/3 | clear ±0.05 |
| MA-BBOB / unif / d2 / 500 | 3 | 5 | 0.7393 | **0.7692** | 0.7132 | 0.7154 | jSO | +0.0299 | 3/3 | INSIDE ±0.05 |
| MA-BBOB / unif / d5 / 500 | 3 | 5 | 0.5359 | 0.5477 | 0.5666 | **0.6171** | portfolio | +0.0505 | 2/3 | clear ±0.05 |
| families-free / noiseless / d2 / 500 | 3 | 15 | 0.5536 | **0.6459** | 0.6276 | 0.5593 | jSO | +0.0183 | 2/3 | INSIDE ±0.03 |
| families-free / noiseless / d5 / 500 | 3 | 15 | **0.3755** | 0.3137 | 0.2944 | 0.2874 | CMA-ES | +0.0618 | 3/3 | clear ±0.05 |
| families-free / noiseless / d10 / 500 | 3 | 15 | **0.2550** | 0.1715 | 0.1577 | 0.1984 | CMA-ES | +0.0566 | 3/3 | clear ±0.05 |
| families-constr / noiseless / d2 / 500 | 3 | 12 | 0.5474 | 0.5620 | **0.5797** | 0.5542 | L-SHADE | +0.0177 | 1/3 | INSIDE ±0.03 |
| families-constr / noiseless / d5 / 500 | 3 | 12 | 0.3162 | **0.3677** | 0.3420 | 0.2895 | jSO | +0.0257 | 2/3 | INSIDE ±0.03 |

**Winner counts: CMA-ES 6, jSO 5, portfolio 3, L-SHADE 3.**  No arm owns
more than a third of the table — §38's "the best arm is a function of the
regime" holds on the full week's data, not only on the noise batteries.

Nine of seventeen margins clear their floor; eight are inside it.  The
cells whose flip is *established* (clear margin **and** a mechanism):

* **cauchy → CMA-ES**, at both dimensions (+0.085 / +0.115, 3/3 at d=5) —
  §38's rank-recombination-survives-outliers mechanism.
* **families, d ≥ 5 → CMA-ES** (+0.062 / +0.057, 3/3 each; the d=10 delta
  vs the portfolio is +0.0566 [+0.0379, +0.0754], the tightest CI in the
  table) — ill-conditioned, rotated, non-MA-BBOB landscapes.
* **gauss, d=10 → CMA-ES** (+0.099, 3/3).
* **gauss, d ≤ 5 → jSO or the warm portfolio** (+0.041 / +0.071, 3/3 at d=5).

Single-arm-only view (portfolio excluded), for the cells where the
portfolio was the winner above: d5/noiseless → **CMA-ES** 0.6119 (+0.038
over jSO, clear at 12 seeds); d5/gauss → **jSO** 0.6264 (+0.031, clear
±0.03); d5/unif → **L-SHADE** 0.5666 (+0.019, INSIDE); d20/2000 →
**CMA-ES** 0.4773 (+0.064, clear).

**Cross-check on an independent stream.**  `oracle12b.json` (12 seeds,
one stream *per arm*, LBC included) puts the same four DE/ES arms within
0.015 at d=2 (jSO 0.7328, LBC 0.7311, L-SHADE 0.7270, CMA-ES 0.7029) and
within 0.023 at d=5 (CMA-ES 0.5808, jSO 0.5788, LBC 0.5627, L-SHADE
0.5577) — i.e. it reproduces §22/§28 and it **disagrees with the paired
screen stream at d=5**, where CMA-ES leads jSO by +0.038.  Two 12-seed
measurements of the same regime differ by more than the 12-seed floor,
purely through the RNG-stream convention.  That is the single largest
methodological caveat on this whole table.

---

## 3. The regime oracle vs fixed arms and vs the portfolio

All cross-regime numbers use the **three seeds common to every regime**
(7, 42, 1234) and weight each of the 17 regimes equally.  (Run-weighted
pooling gives the same ordering: CMA-ES 0.4944, jSO 0.4840, portfolio
0.4745, L-SHADE 0.4715, regime-gated fitted 0.5321.)

| selector | mean AOCC | vs CMA-ES |
|---|---:|---:|
| fixed L-SHADE | 0.5094 | −0.0287 |
| fixed jSO | 0.5206 | −0.0175 |
| fixed portfolio `Blocks_uniform_cj_warm2` | 0.5277 | −0.0103 |
| **fixed CMA-ES (shipped flagship, best fixed arm)** | **0.5381** | — |
| regime oracle, 3 single arms — **fitted on these seeds** | 0.5681 | +0.0301 |
| regime oracle, 4 arms incl. portfolio — **fitted** | **0.5771** | **+0.0391** |
| per-instance oracle within regime, 3 arms (fitted) | 0.5870 | +0.0490 |
| per-instance oracle within regime, 4 arms (fitted) | 0.5987 | +0.0606 |
| per-(seed, instance) oracle, 4 arms (absolute ceiling) | 0.6253 | +0.0873 |

The fitted numbers are **optimistically biased** — the winner of each cell
was chosen using the same three seeds it is scored on, with three or four
arms competing.  §18 and §30 are exactly about that bias.  The honest
version:

### Leave-one-seed-out (3 folds, t(2)-based 95 % CI)

The selection *and* the "best fixed arm" are both re-fitted on the two
training seeds and scored on the held-out seed.

| gated selector | vs best fixed | vs CMA-ES | vs jSO | vs L-SHADE | vs portfolio |
|---|---|---|---|---|---|
| 3 single arms | +0.0193 [−0.0052, +0.0439] 3/3 | +0.0193 [−0.0052, +0.0439] | +0.0368 [−0.0061, +0.0797] | +0.0480 [+0.0091, +0.0869] | +0.0297 [+0.0114, +0.0480] |
| 3 arms + portfolio | +0.0258 [−0.0056, +0.0572] 3/3 | +0.0258 [−0.0056, +0.0572] | +0.0433 [−0.0035, +0.0901] | +0.0545 [+0.0117, +0.0973] | +0.0361 [+0.0106, +0.0617] |

Per held-out seed (4-arm gating): seed 7 → 0.5720 gated vs 0.5550 CMA-ES;
seed 42 → 0.5518 vs 0.5216; seed 1234 → 0.5484 vs 0.5376.  Positive in
3/3 folds, but n = 3 and the interval crosses zero.

**Read this against the floor.**  +0.026 is inside the ±0.05 three-seed
floor for a CMA-ES-containing comparison.  The 3/3 sign and the fact that
the mechanism (§38) predicted the direction before the measurement are
worth something; the CI is not.

### The 12-seed control, and why it is the most important row here

The one regime *pair* with twelve seeds is MA-BBOB noiseless 500·d,
d ∈ {2, 5}.  Gating there means "pick the arm by dimension".  Twelve-fold
LOSO, four arms:

| | mean | 95 % t-CI | folds won |
|---|---:|---|---|
| gated − best fixed (= portfolio, 0.6854) | **−0.0168** | [−0.0482, +0.0147] | 5/12 |
| gated − CMA-ES (0.6662) | +0.0025 | [−0.0374, +0.0423] | 7/12 |
| gated (3 single arms) − best fixed single arm (CMA-ES) | −0.0077 | [−0.0386, +0.0231] | 6/12 |

**Regime gating loses here.**  Why is visible in the per-regime table: at
d=2 the four arms sit within 0.012, so the fitted "winner" is noise, and
the fold-to-fold choice flips (L-SHADE in 9 folds, jSO in 2, portfolio in
1).  Gating pays a selection cost wherever the arms are within the floor.
Everything the cross-regime LOSO gain is made of comes from the 3-seed
regimes; the moment a regime pair has enough seeds to test the gate, the
gate does not clear.

### Where the +0.026 actually comes from

Per-regime contribution to the 4-arm LOSO gain over a fixed CMA-ES default
(each regime carries 1/17 of the total):

| regime | gate picks | Δ vs CMA-ES | contribution |
|---|---|---:|---:|
| MA-BBOB / gauss / d5 | portfolio | +0.1299 | +0.0076 |
| MA-BBOB / noiseless / d10 / 2000 | L-SHADE, portfolio | +0.1004 | +0.0059 |
| families-free / d2 | jSO | +0.0923 | +0.0054 |
| MA-BBOB / unif / d5 | portfolio | +0.0812 | +0.0048 |
| MA-BBOB / gauss / d2 | jSO | +0.0728 | +0.0043 |
| families-constr / d5 | jSO | +0.0515 | +0.0030 |
| MA-BBOB / unif / d2 | jSO | +0.0299 | +0.0018 |
| **cauchy d2, cauchy d5, gauss d10, fam-free d5, fam-free d10** | CMA-ES | 0.0000 | 0.0000 |
| families-constr / d2 | L-SHADE/jSO/portfolio (unstable) | −0.0154 | −0.0009 |
| MA-BBOB / noiseless / d5 | jSO, portfolio | −0.0138 | −0.0008 |
| MA-BBOB / noiseless / d2 | CMA-ES/L-SHADE/jSO (unstable) | −0.0288 | −0.0017 |
| MA-BBOB / noiseless / d20 / 2000 | CMA-ES, portfolio | −0.0244 | −0.0014 |
| MA-BBOB / noiseless / d10 / 500 | CMA-ES, portfolio | −0.0371 | −0.0022 |
| | | | **+0.0258** |

The uncomfortable line: **the five regimes where the flip is established
with a mechanism contribute exactly zero**, because in all five the right
answer is CMA-ES and CMA-ES is already the default.  The gain is made
entirely of *switches away from* CMA-ES at d ≤ 5 and at d=10/2000·d, and
every one of those rests on three seeds.  Against a **DE** default the
picture reverses — gating beats fixed L-SHADE by +0.055 [+0.012, +0.097],
because it buys the cauchy insurance.  Regime gating's value is therefore
mostly *insurance for a non-CMA-ES default*, not *upside for the shipped
one*.

---

## 4. Per-instance oracle: how much headroom does gating capture?

| level of selection | mean | headroom over fixed CMA-ES | share captured |
|---|---:|---:|---:|
| fixed CMA-ES | 0.5381 | — | — |
| regime-gated, 3 arms (fitted) | 0.5681 | +0.0301 | **61 %** of per-instance |
| regime-gated, 4 arms (fitted) | 0.5771 | +0.0391 | **64 %** of per-instance, 45 % of ceiling |
| per-instance oracle within regime, 3 arms | 0.5870 | +0.0490 | 100 % (of its own) |
| per-instance oracle within regime, 4 arms | 0.5987 | +0.0606 | |
| per-(seed, instance) oracle, 4 arms | 0.6253 | +0.0873 | absolute ceiling |

So the regime features capture roughly **two thirds of the arm-choice
headroom that a per-instance oracle would have**, and a bit under half of
the unattainable per-run ceiling.  Honestly (LOSO, per-instance gating
re-fitted per fold): per-instance gating scores +0.0210 [+0.0071, +0.0348]
over the best fixed arm, i.e. **statistically indistinguishable from
regime gating's +0.0193** on three seeds.  Going finer than the regime
buys nothing measurable — consistent with §22/§28, where the per-instance
oracle headroom (+0.074) was large but no pairing of arms recovered more
than ~64 % of it either.

---

## 5. Which regime features matter

Fitted vs honest (LOSO) scores for gating on a feature subset.  Each
subset pools regimes that share the feature values and picks one arm per
group, from the four arms.

| gate on | groups | fitted | LOSO | LOSO vs CMA-ES |
|---|---:|---:|---:|---|
| nothing (fixed arm) | 1 | 0.5381 | 0.5381 | 0.0000 |
| dimension alone | 4 | 0.5428 | 0.5225 | **−0.0156** [−0.0307, −0.0004] |
| budget/dim alone | 2 | 0.5444 | 0.5444 | +0.0063 [−0.0027, +0.0153] |
| noise class alone | 4 | 0.5419 | 0.5353 | −0.0028 [−0.0118, +0.0063] |
| constrained alone | 2 | 0.5419 | 0.5419 | +0.0039 [+0.0016, +0.0062] |
| battery kind alone | 2 | 0.5388 | 0.5273 | −0.0108 [−0.0147, −0.0068] |
| dim + budget/dim | 5 | 0.5507 | 0.5354 | −0.0027 [−0.0195, +0.0141] |
| dim + noise | 11 | 0.5659 | 0.5518 | +0.0138 [+0.0051, +0.0225] |
| dim + budget + noise | 12 | 0.5713 | 0.5632 | +0.0252 [+0.0127, +0.0376] |
| **dim + budget + noise + constrained** | 14 | 0.5754 | **0.5691** | **+0.0310** [+0.0176, +0.0444] |
| the above + battery kind (= full regime) | 17 | 0.5771 | 0.5639 | +0.0258 [+0.0115, +0.0401] |

* **Dimension alone is the worst feature** — negative out of sample.  Its
  fitted gain (+0.005) is entirely winner's curse; the d=2 and d=5 cells
  it splits are the ones where the arms are within the floor.
* **Noise class is the feature with the mechanism** but alone it is also
  ≈ 0, for the same reason as in §3: it only tells you to keep CMA-ES.
* **Dim × noise is where the interaction lives** (+0.014): jSO at d ≤ 5
  under gauss/unif, CMA-ES at d ≥ 10 *and* under cauchy.
* Adding budget/dim on top is worth another +0.011 — the d=10 flip
  (CMA-ES at 500·d → L-SHADE at 2000·d) is the biggest single noiseless
  contribution in §3.  **Caveat: budget/dim is fully confounded with
  dimension in this data** — every 2000·d cell is d ≥ 10 and every
  d ≤ 5 cell is 500·d.  "Budget matters" is untested as an independent
  claim (see gaps).
* Adding battery kind *hurts* out of sample (17 groups, 3 seeds): the
  extra resolution is overfitting.

### Hand-written rules (no fitted parameter, so the score is honest)

| rule | mean | vs CMA-ES | vs jSO | vs portfolio |
|---|---:|---|---|---|
| `cauchy or d≥10 → CMA-ES, else jSO` | 0.5542 | +0.0162 [−0.0120, +0.0444] 3/3 | +0.0337 [−0.0160, +0.0833] | +0.0265 [−0.0050, +0.0580] |
| `cauchy → CMA-ES; d≥10 → (L-SHADE if bpd≥2000 else CMA-ES); non-MA-BBOB & d≥5 → CMA-ES; MA-BBOB noiseless d=5 → portfolio; else jSO` | **0.5624** | **+0.0244 [−0.0220, +0.0708] 3/3** | +0.0419 [+0.0311, +0.0526] 3/3 | +0.0347 [+0.0155, +0.0539] 3/3 |
| fitted 4-arm regime oracle (upper bound) | 0.5771 | +0.0391 | | |

The five-clause rule agrees with the fitted oracle in **13 of 17** cells
and recovers 62 % of the fitted oracle's gain over CMA-ES while having no
free parameters.  The one-line rule recovers 41 %.  Neither clears the
±0.05 floor against CMA-ES.

**Recommended rule, if one is to be written down today** (the simplest
form whose every clause has either a mechanism or a 12-seed number behind
it):

```
if outliers detected (heavy-tailed re-evaluation spread):   CMA-ES
elif dim >= 10 and budget_per_dim < 1000:                    CMA-ES
elif problem is not MA-BBOB-like and dim >= 5:               CMA-ES
else:                                                        jSO      # or the warm portfolio at d=5
```

with the explicit note that the `else` branch is where the 12-seed data
says the arms are within the floor, so the branch is a coin toss and the
value of the rule is entirely in its first three clauses.

---

## 6. Verdict

1. **The flip is real.**  Four different arms win at least three of the 17
   regime cells; the best-minus-worst spread inside a cell reaches 0.17.
   §38's claim survives the full week's data.
2. **A regime-gated default is better than any fixed arm on this data —
   but not by a margin that clears the floor.**  Honest (LOSO) gain over
   fixed CMA-ES: **+0.026 [−0.006, +0.057], 3/3 seeds**, against a
   three-seed floor of ±0.05.  Say it plainly: **inside the floor.**
3. **It beats the online portfolio clearly**: +0.036 [+0.011, +0.062],
   3/3.  The portfolio as a *fixed* default (0.5277) is worse than fixed
   CMA-ES (0.5381) across the 17 regimes — it wins only 3 cells, all at
   d=5.  §31's conclusion (the *selection policy* is worth nothing, the
   *sharing* is worth everything) is unchanged; what this table adds is
   that a *context* gate, unlike an *online* gate, at least points the
   right way in every regime where a mechanism is known.
4. **The gate's value depends on the default it replaces.**  Against
   CMA-ES it is +0.026 (inside the floor); against L-SHADE it is +0.055
   with a CI clear of zero.  The shipped flagship already sits on the
   right answer in five of the six regimes where the evidence is
   strongest.
5. **The 12-seed control is negative.**  Gating on dimension over
   d ∈ {2, 5}, MA-BBOB noiseless: −0.017 [−0.048, +0.015], 5/12.  Do not
   gate where the arms are within the floor.
6. **Finer than the regime buys nothing**: per-instance gating scores
   +0.021 LOSO against regime gating's +0.019.  Regime gating captures
   ~64 % of the per-instance headroom and 45 % of the per-run ceiling.

The one design consequence that does not depend on any of the contested
numbers: **an outlier detector plus a CMA-ES fallback is worth more than
any bandit measured this week**, and it is cheap — a handful of
re-evaluations at the start of a run.  Everything else in the gate should
wait for seeds.

---

## 7. Data gaps, and the runs that would settle the table

### Cells that exist but have only 3 seeds

Fifteen of the seventeen regimes.  Only MA-BBOB/noiseless/d2/500 and
/d5/500 have twelve.

### Cells that do not exist at all

| missing | why it matters |
|---|---|
| **MA-BBOB noiseless, d = 2 and 5, at 2000·d** | *The* confound: budget/dim never varies at d ≤ 5, so "budget/dim matters" cannot be separated from "dimension matters". This is the highest-value missing cell in the table. |
| MA-BBOB noise (gauss/unif/cauchy) at 2000·d, any d | the noise × budget interaction is untested |
| unif and cauchy at d = 10 and 20 | "outliers → CMA-ES" is measured only at d ≤ 5; the gate's strongest clause has no high-dimensional support |
| gauss at d = 20 | — |
| MA-BBOB noiseless d = 20 at 500·d | the d=20 winner is only known at one budget |
| families-free at d = 20; families-constrained at d = 10, 20 | constrained is measured only at d ≤ 5 |
| families under any noise | the two "off-MA-BBOB" axes are never crossed |
| LBC (and PSO) outside the standard battery | LBC wins 4 of 5 d=2 instances in §28 but has never been run on noise, high dim, families or constraints, and its only data (`oracle12b`) is on an unpaired stream |
| d = 10 has 3 instances, d ≤ 5 has 5 | the high-dim cells are thinner as well as fewer |

### Runs that would settle it, in priority order

12-seed roster: `42 7 1234 2025 3 11 99 123 777 2024 31337 555`.
Specs: `specs=CMAES_alone,JSO_alone,LSHADE_alone,Blocks_uniform_cj_warm2`
(abbreviated `$S` below).  Timings extrapolated from the 3-seed logs.

```sh
S=CMAES_alone,JSO_alone,LSHADE_alone,Blocks_uniform_cj_warm2
R="42 7 1234 2025 3 11 99 123 777 2024 31337 555"

# 1. Break the budget/dimension confound (NEW cell, ~30 min) -- highest value.
uv run python benchmarks/portfolio_screen.py planning/results/2026-09-12/bm2000_d2d5.json $R \
    kind=standard dims=2,5 bm=2000 specs=$S

# 2. The cauchy clause on 12 seeds (~10 min) -- the one mechanism the gate leans on.
uv run python benchmarks/portfolio_screen.py planning/results/2026-09-12/cauchy12.json $R \
    kind=noisy-cauchy specs=$S

# 3. The two switch-away-from-CMA-ES regimes that carry most of the LOSO gain (~10 min each).
uv run python benchmarks/portfolio_screen.py planning/results/2026-09-12/gauss12.json $R \
    kind=noisy-gauss specs=$S
uv run python benchmarks/portfolio_screen.py planning/results/2026-09-12/unif12.json $R \
    kind=noisy-unif specs=$S

# 4. The budget flip at d=10: CMA-ES at 500*d vs L-SHADE at 2000*d (~15 min + ~35 min).
uv run python benchmarks/portfolio_screen.py planning/results/2026-09-12/hd10_bm500_12.json $R \
    kind=highdim dims=10 bm=500 specs=$S
uv run python benchmarks/portfolio_screen.py planning/results/2026-09-12/hd10_bm2000_12.json $R \
    kind=highdim dims=10 bm=2000 specs=$S

# 5. Outliers at high dimension (NEW cells, ~15 min each) -- the gate's weakest support.
uv run python benchmarks/portfolio_screen.py planning/results/2026-09-12/cauchy_d10.json $R \
    kind=noisy-cauchy dims=10 bm=500 insts=0,1,2 specs=$S
uv run python benchmarks/portfolio_screen.py planning/results/2026-09-12/unif_d10.json $R \
    kind=noisy-unif dims=10 bm=500 insts=0,1,2 specs=$S

# 6. Noise x high dim, 12 seeds (~15 min).
uv run python benchmarks/portfolio_screen.py planning/results/2026-09-12/noisy_hd12.json $R \
    kind=noisy-highdim specs=$S

# 7. Off-MA-BBOB, 12 seeds (~1.5 h free, ~1.2 h constrained).
uv run python benchmarks/family_screen.py planning/results/2026-09-12/fam_free12.json $R \
    preset=free dims=2,5,10 bm=500 specs=$S
uv run python benchmarks/family_screen.py planning/results/2026-09-12/fam_con12.json $R \
    preset=constrained dims=2,5 bm=500 specs=$S

# 8. d=20 at both budgets, 12 seeds (~2.2 h at 2000*d, ~35 min at 500*d) -- run last.
uv run python benchmarks/portfolio_screen.py planning/results/2026-09-12/hd20_12.json $R \
    kind=highdim dims=20 bm=2000 specs=$S
uv run python benchmarks/portfolio_screen.py planning/results/2026-09-12/hd20_bm500_12.json $R \
    kind=highdim dims=20 bm=500 specs=$S

# 9. LBC and PSO on the standard battery at the missing budget, paired-per-arm stream (~40 min).
uv run python benchmarks/oracle.py planning/results/2026-09-12/oracle_bm2000.json $R \
    dims=2,5 bm=2000 arms=cmaes,jso,lshade,lbc
```

Runs 1–4 (≈ 1.5 h in total) are what decide whether the regime gate is a
default or a footnote.  Run 1 alone decides whether "budget per dimension"
belongs in the gate at all.

### One methodological gap that no amount of seeds fixes

`benchmarks/oracle.py` and `benchmarks/portfolio_screen.py` disagree about
RNG pairing (`seed_name=<arm>` vs `seed_name="screen"`), and on the same
12-seed standard battery they disagree about whether CMA-ES or jSO leads
at d=5 by more than the 12-seed floor.  Until the two harnesses are put on
the same convention — or the disagreement is shown to be the §34
cross-process DE nondeterminism — every arm ranking in this file carries
that term.

---

*Sources: `planning/results/2026-09-10/{screen12,roster2,oracle12b,
family_screen_free_s42-7-1234,family_screen_constrained_s42-7-1234}.json`,
`planning/results/2026-09-11/{noiseless,noisy_gauss,noisy_unif,
noisy_cauchy,noisy_hd,hd10,hd10_bm500,hd20}.json`.  Analysis scripts are
scratch-only; every number above is reproducible from those rows files.*
