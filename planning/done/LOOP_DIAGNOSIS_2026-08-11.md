# Why the self-improvement loop stopped improving — a 34-night audit

**Date:** 2026-08-11 · **Data:** `planning/self_improve_ledger_aocc.jsonl`,
952 records / 680 proposals / 34 nights (2026-07-09 .. 2026-08-11), archived
at rotation to `planning/done/self_improve_ledger_aocc_2026-08-11.jsonl`.

This document exists so that no future session has to re-derive any of it.
It records what the loop's first 34 nights actually produced, why, and what
was changed in response (PRs #299–#302). Read it alongside `GOAL.md` §2.

---

## 1. The headline

**Over 34 nights and 680 proposals the loop produced one measurable
improvement, worth +0.005 mean AOCC.**

| quantity | value |
|---|---|
| first-of-night baseline, 34 nights | mean 0.3398, sd 0.0061 |
| linear trend | +0.00015/night, t = **+1.39** (not significant) |
| first 5 nights → last 5 nights | 0.3433 → **0.3420** |
| proposals | 680 |
| accepts | 72 (10.6%) |
| codify slots that reached a 12-seed A/B | 5 |
| codify slots that survived it | **0** |

The one improvement — dropping the `Restart` analyzer on 2026-07-30 —
claimed **+0.0384** in its local A/B and delivered **+0.0049** in the nightly
baseline (n=22 before vs 12 after, SE 0.0018; real, but ~8× smaller than
claimed). That +0.0049 is essentially the *entire* 34-night drift. Treat
every future A/B headline number as inflated several-fold; this is textbook
winner's curse and it is now a measured local constant, not a worry.

---

## 2. The accept rate was exactly what pure noise predicts

```
proposal deltas:      n=680   mean -0.00057   sd 0.01011
eps_accept:           0.005                 <- 0.5 sd of the noise
P(delta > 0.005):     0.301
screening accepts:    210/680 = 0.309       <- matches
confirm survival:     72/210  = 0.343       (independent redraw predicts 0.30)
joint:                0.30 x 0.34 = 0.106   vs observed accept rate 0.106
```

The screening bar sat at **half a standard deviation** of the measurement
noise, so roughly 30% of *zero-effect* proposals cleared it, and the
confirmation gate re-rolled at the same bar against a fresh
`randomize_iteration`. There is no room left in the observed accept rate for
a real signal. The loop was not finding weak improvements; it was sampling
its own noise twice.

---

## 3. Four defects, all in the instrument

### 3.1 The hold-out leg measured a different metric entirely

`SelfImprover._measure()` branches on `metric == "aocc"`.
`_measure_holdout()` did not — it went straight to
`holdout_harness_config()` → the **composite** harness.

| | mean |
|---|---|
| `seed_training_score` (AOCC) | 0.3402 |
| `seed_holdout_score` (actually `composite_score`) | 0.0339 |
| composite ledger's own baseline mean | **0.0452** ← same scale |

So the 8.5× "instance-family generalization gap" that headed `GOAL.md` §5.1
as the project's number-one research priority from 2026-07-30 was **a unit
mismatch**. After the fix a live quick-battery run measures **0.3383
training vs 0.3342 hold-out** — a 0.004 gap. It also explains
`overfit=False` on all 66 hold-out records: the gate was differencing two
incommensurable quantities.

The ratio evidence had pointed the right way before the cause was found —
the *untuned* seed spec dropped by the same factor as the tuned one
(0.0996 vs 0.1166), which already ruled out overfitting. The wrong
conclusion drawn from it at first was "the hold-out battery is just harder";
the real answer was in the code.

### 3.2 No accept decision ever crossed an instance-family boundary

The §6.4 confirmation gate's hold-out leg was guarded by
`and self.config.metric != "aocc"` — added precisely because of §3.1.

```
confirm_holdout_seed across 138 confirm records:  {None: 138}
accepts citing a hold-out base seed:              0 / 72
```

The confirmation only ever drew a fresh `randomize_iteration` *inside* base
seed 42.

### 3.3 The nightly never passed `--base-seed`

```
base_seed across all 952 records:  {42: 952}
```

`--base-seed` existed and defaulted to 42; the workflow simply never set it.
Codify-scan's "k ≥ 2 distinct nights" gate — and the `--min-fresh-nights`
resurrection gate added 2026-08-08 — were therefore counting **k
re-measurements of one instance draw** as k independent confirmations.

This is the mechanical cause of the 0/5 codify hit rate. Five slots
(`Sensitivity.update_interval`, `Sensitivity drop_analyzer`, `NelderMead
drop_heuristic`, `JSO drop_heuristic`, `NLSHADE_LBC add_heuristic`) each had
a tight pooled CI excluding zero, and each died at 12 seeds. The transfer
coefficient is measurable directly:

```
corr(training_delta, holdout_delta) = +0.175   (n = 66)
positive on training 32/66   positive on holdout 33/66
```

r = 0.175 — the nightly measurement explained 3% of the variance in what
happened on a different seed.

### 3.4 `--sync-eval` was unreachable from the loop

Shipped 2026-08-09 on `scripts/ioh_benchmark.py`, where it cut measurement
noise 1.6×, but never plumbed into `scripts/self_improve.py run`. The
process generating *all* the evidence was the one not using it. Measured
cost when finally wired: **2.6 s vs 2.7 s** on the quick battery.

---

## 4. The deeper problem: a scalar objective over heterogeneous regimes

Fixing precision is necessary but not sufficient. On 2026-08-11 (PR #298)
the NL-SHADE-LBC arm produced the first per-dimension split where both CIs
exclude zero, in opposite directions:

| dim | Δmean | CI95 |
|---|---|---|
| d2 | **−0.0241** | [−0.0401, −0.0080] |
| d5 | **+0.0080** | [+0.0007, +0.0154] |
| scalar composite | −0.0080 | "lean-negative" |

And the same arm leaned *positive* at 2-D×200 evals while losing at
2-D×1000 — a budget effect on top of the dimension effect.

If effects are routinely that heterogeneous, the population-mean objective
has a **flat optimum by construction**. No amount of extra measurement
precision helps, because the surface really is level; the loop was climbing
an average of two hills that point in opposite directions.

The power arithmetic underlines it: between-seed paired delta sd is ~0.025,
so detecting a +0.005 *population* effect at 80% power needs **~190 seeds
per proposal** — about 45 minutes per decision at 6.8 s/iteration. Chasing
sub-0.01 scalar effects was never going to work at this budget. The way out
is not more seeds; it is asking a better-posed question.

---

## 5. What was changed (2026-08-11)

| PR | change |
|---|---|
| **#299** | hold-out routes through the AOCC path (§3.1); confirm gate crosses a base seed (§3.2); nightly rotates `--base-seed` over 7 values (§3.3); `--sync-eval` reachable and on (§3.4); `eps_accept` 0.005 → 0.0125 (2σ of the post-sync-eval floor), relax floor 0.001 → 0.006 |
| **#300** | `--aocc-extra-dims`; nightly battery `dims=(2,)` → `(2, 5)`; cost 1.34× |
| **#301** | `--accept-stat rank` — Wilcoxon signed-rank + Hodges-Lehmann. **Available, not enabled** |
| **#302** | `--cell-by dim` — per-dim deltas/CIs reported and gated; on in the nightly |

Under the null, the joint screen+confirm false-positive rate goes from ~9%
to ~0.05%; power for a true +0.02 effect stays ~77%. Effects below ~0.01
were never shippable — five A/Bs proved that — so the loop should stop
spending nights on them.

`#301` is deliberately **not** enabled. Tonight already changes the accept
regime three ways; a fourth simultaneous change to the accept *rule* would
make the next few weeks unattributable. That discipline is the whole lesson
of §3.

---

## 6. Reading the ledger across this boundary

Records before and after 2026-08-11 are **not comparable**, in three
independent ways: single vs rotated base seed, async vs sync evaluation, and
d2-only vs (d2, d5) battery — the last of which moves the score *level*
(d2 alone ~0.369, (d2, d5) ~0.309), not just its noise.

Each iteration record now carries `base_seed`, `sync_eval`,
`aocc_extra_dims`, `accept_stat`, and `per_cell`, precisely so a consumer can
group rather than blindly pool. **Codify-scan does not yet honour these** —
that is queued in `TODO.md` and is the highest-value remaining item, because
until it does, cross-night evidence still mixes regimes.

The pre-boundary ledger was rotated into `planning/done/` at this point. The
bandit posterior is preserved: the nightly passes `--prime-include-archives`,
which replays archived ledgers. What resets is codify-scan's evidence base,
which is the intent — 34 nights of single-draw evidence should count as one
night's worth, not 34.

---

## 7. What to do next

1. **Teach codify-scan to read `per_cell` and to group by
   `base_seed` / `sync_eval` / `aocc_extra_dims`.** Without this the new
   instrumentation is recorded but not actionable.
2. **Let the rotated-seed ledger accumulate**, then A/B `--accept-stat rank`
   against the mean rule on accept rate and codify survival.
3. **Budget-phase cells** — needs AOCC recomputed on trajectory slices;
   `trace_evals` / `trace_fx` are already recorded.
4. **Dimension/budget-gated arm activation** — the shippable form of the d5
   gains that keep showing up (NLSHADE_LBC, and probably CMA-ES; re-measure
   that one at d5 first).
5. **Re-earn the codify backlog.** Every slot rejected on pre-2026-08-11
   evidence was rejected for the right reason (it did not replicate), but on
   an instrument that could not have told the difference. They are open
   questions again, not settled ones — though the burden of proof is on
   anyone who wants to re-litigate them.
