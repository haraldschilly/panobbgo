# Self-Improvement Loop: Design & Operating Instructions

**Status:** living document — compact instructions; revised 2026-06-09 (V2).
**Owner:** Panobbgo maintainers + coding agents.
**Related:** `panobbgo/harness.py`, `panobbgo/self_improve.py`,
`scripts/self_improve.py`, `doc/source/guide_benchmarking.rst`,
`AGENTS.md` (Benchmark Harness section).
**History:** dated shipping entries and the idea backlog live in
`planning/SELF_IMPROVEMENT_LOG.md` (formerly §13 of this file).

## 1. Vision

Run Panobbgo in a loop that *improves itself*: measure → propose a
change → apply → measure → accept if better, revert otherwise — against
a battery of **standardized but parametrically randomized** tests.
Trustworthy enough to run unattended; honest enough that a sustained
positive trend means the framework really got better.

Three hard properties, each mapped to a component:

1. **Over-fitting** ➜ parametric randomization of problems (§4).
2. **Noise** ➜ statistical acceptance rule (§6).
3. **No absolute anchor** ➜ external baselines in the same harness (§5).

## 2. State of the loop (diagnosis, 2026-06-09)

All V1 infrastructure is **shipped and mechanically solid**: randomized
battery (§4), baselines (§5), paired-bootstrap acceptance (§6.2),
anti-cherry-pick guard (§6.3), hold-out validation, Thompson-sampling
mutation bandit, structural mutations (§7.2), nightly cron (§9/§12),
JSONL ledger.  ~30 mutation rules, ~16 heuristics.  Dated details:
`SELF_IMPROVEMENT_LOG.md`.

But measured against §11, the loop **does not yet improve anything
durably**.  Evidence from the ledger (80 iterations, 2026-06-06 →
2026-06-09) and `gh run list`:

1. **No metric resolution where the loop operates.**  Median
   `baseline_score` ≈ 0.03 on the randomized quick battery; **34% of
   mutations measure Δ = exactly 0.0000** (zero information).  Accepts
   are upward noise spikes (winner's curse).
2. **Accept → rollback churn.**  15/16 guard checks rolled the ladder
   back; all hold-out records ran on an empty ladder (`top_iter=-1`,
   vacuous `drift=0.0000` reported as OK).
3. **Nothing persists between nights.**  The ladder is in-memory; the
   only durable channel is manual codification (used once:
   `Sobol.scramble=False`, 2026-05-31 — which *worked*).
4. **Catalog ≫ registry mismatch.**  The nightly runs quick mode, whose
   registry is 2 simple strategies; most kwarg rules shipped since
   mid-May target heuristics that never appear in the nightly run.
5. **Compute ~94% idle** (20 quick iterations ≈ 5 min of a 90-min
   budget) while statistical power is the binding constraint.
6. **Bandit starved**: binary accept reward at ~2.5% base rate, and
   priming reads only the current ledger — archives in `planning/done/`
   are invisible.

Every symptom is downstream of (1).  Hence the V2 priorities in §9:
**fix the metric, exercise the catalog, confirm before accept, persist
via codify PRs** — in that order, before adding any new arms (§7.3).

## 3. Architecture

```
LOOP DRIVER (scripts/self_improve.py run)
  for i in range(iterations):
      baseline  = measure(current specs)          # randomized battery, §4
      proposal  = bandit.pick_mutation(history)   # catalog, §7
      candidate = measure(specs + proposal)
      accept    = statistical_accept(...)         # §6.2, paired bootstrap
      [V2] confirm on fresh instances before recording accept   # §6.4
      if accept: advance ladder  else: revert, bias bandit away
      every Kth iter: guard re-validates ladder top              # §6.3
  end-of-run: hold-out re-measure on independent base_seed(s)
  [V2] codify-scan: cross-night evidence → codify PR             # §9.3
```

## 4. Parametric problem battery

Each rep draws a transformed instance `P̃ = T(P; θ)`: translation
(kills center exploitation), Haar-random rotation (kills axis-aligned
advantage), log-uniform diagonal scaling (ill-conditioning), additive
Gaussian noise, and cyclically stratified dimensions (rep `i` →
`dim_choices[i % k]`, so any `k` contiguous reps cover every dim — see
§10).  Instance seeds derive from
`sha256(base_seed, iteration_id, family, rep)`: within one iteration
`before`/`after` see the **same** instances (paired comparison);
across iterations instances differ (anti-overfit).  The composite
score becomes a Monte-Carlo estimate of expected performance on the
family.  Implementation: `panobbgo/harness_randomized.py`; CLI
`--randomize`, `--randomize-iteration N`.

## 5. External absolute baselines

`Baseline_Random` (floor), `Baseline_SciPyDE`, `Baseline_SciPyAnneal`
(competitive references) as `StrategySpec` adapters with hard budget
enforcement — `panobbgo/harness_baselines.py`, CLI `--baselines`.
They anchor the score absolutely ("Panobbgo vs the field"), not just
relatively ("vs its previous self").  Run them at `--standard`/`--full`.

## 6. Statistical acceptance rule

### 6.1 Paired bootstrap CI

Per `(problem, strategy)` pair, bootstrap a 95% CI on the composite
delta.  Under `--randomize`, reps are instance-aligned, so the
**paired** sampler (one shared resample index, default since
2026-05-14) applies — typically 3–10× narrower than unpaired.  Force
with `--paired`/`--unpaired`; use unpaired only when reps are not
instance-aligned (e.g. different `base_seed` ledgers).

### 6.2 Decision rule

Let `Δ = composite_after − composite_before`, `r_i` the per-pair deltas.
**Accept** iff all of:

- `Δ > ε_accept` (default `0.005`),
- lower bound of the bootstrap 95% CI on `Δ` is `> 0`,
- `min_i r_i > −ε_regress` (default `−0.05`) — no catastrophic pair
  regression.

Otherwise reject and revert.  API:
`panobbgo.harness.statistical_accept`; CLI `compare --statistical
--fail-on-regression`.  An inactivity-guarded `eps_accept` relaxation
(geometric decay during accept droughts, floored, re-tightened on
accept) is available via `--inactivity-relax-after` and records the
effective threshold per iteration for auditability.

### 6.3 Anti-cherry-pick guard

Every Kth iteration, re-measure the ladder top on a *fresh* randomized
seed (`iteration + guard_iteration_offset`); pop entries whose score
drifts more than `guard_eps_ladder` below their stored
`last_validated_score`, down to the never-popped seed.  CLI
`--guard-interval`, `--guard-eps-ladder`.  V2 note: with the §6.4
confirm gate in place, a guard rollback of a *confirmed* accept is an
anomaly worth surfacing, not routine cleanup.

### 6.4 Same-night confirmation gate (V2 — open)

The V1 flow recorded accepts straight from the screening measurement;
the guard then rolled back ~all of them (§2.2).  V2 inverts this:
**promotion requires confirmation before the accept is recorded.**

- Re-measure every screening-accepted candidate on a fresh
  `randomize_iteration` *and* on the hold-out base_seed, same night.
- Promote iff the pooled paired CI (screen + confirm) stays > 0.
- Failed confirmations are recorded (`record_type="confirm_reject"`)
  and count as bandit reward 0.
- Hold-out records on an empty ladder must report `status="vacuous"`,
  never `OK drift=0.0`.

## 7. Change catalog

The mutation space, in rough order of safety:

1. **Hyperparameter retunes** — bounded numeric/categorical kwarg
   perturbations (`default_catalog()`); fire only on specs that set
   the kwarg explicitly.
2. **Strategy portfolio composition (§7.2)** — structural ops
   `add_heuristic` / `drop_heuristic` / `add_analyzer` /
   `drop_analyzer` from curated pools
   (`default_structural_catalog()`, CLI `--structural`); per-class
   bandit arms via `--structural-per-class-arms`, hierarchical
   strength-borrowing via `--structural-borrow-alpha`.
3. **Analyzer parameters** — covered by 1.
4. **Heuristic code edits / new scaffolds** — delegated to a coding
   agent with human review; the loop proposes, never commits these.

Every mutation must have a trivial rollback (revert the in-memory spec).

### 7.3 Catalog policy: freeze-and-exercise (V2)

**No new mutation rules, structural candidates, or heuristics until
the nightly loop can resolve them**: two consecutive nightly runs each
producing ≥ 1 *confirmed* accept, or a no-op (Δ=0) rate < 10%.
Rationale: §2.1/§2.4 — adding arms the loop cannot measure is motion,
not progress.  Weekly agent priority order: (a) merge/close open
codify PRs, (b) metric & registry work (§9), (c) only then new rules.

### 7.4 Bandit reward shaping (V2 — open)

Replace binary accept-reward with a graded reward in [0, 1]:
`0` for no-ops and confirm-rejects; `0.5 + clip(ci_low/eps_scale)` for
confirmed accepts; `clip(0.5 + Δ/eps_scale, 0, 0.5)` for honest
rejections (real signal, wrong sign / too small), `eps_scale ≈
4·eps_accept`.  Beta arms update as `alpha += r, beta += 1−r`.  Arms
that consistently produce small-positive deltas become distinguishable
from harmful arms at realistic per-night iteration counts.

## 8. Safety rails

- The cron never edits `panobbgo/` source; source changes go through
  reviewed PRs (§9.3).
- One atomic ledger record per decision; ledger is append-only.
- Hard per-run timeout (`HarnessConfig.timeout_per_run`) and wall-clock
  budget per iteration.
- A mutation that breaks tests is auto-reverted regardless of score.
- Human escape hatch: audit the ledger; `STOP` sentinel halts the loop.

## 9. Nightly pipeline (V2 target)

Three stages, one workflow run (`self_improve_nightly.yml`, 03:00 UTC,
90-min cap; manual `workflow_dispatch` accepts `iterations`/`mode`).
Spend ~60 of the 90 minutes — V1 used ~5.

### 9.1 Stage 1 — Screening (~35 min) [partially open]

- **Metric: AOCC** (`--metric aocc`, shipped) instead of
  composite_score.  AOCC is anytime and continuous — every evaluation
  moves it, eliminating the Δ=0 dead zone and the floor effect of §2.1.
  composite_score stays as a reporting metric.  *Fallback* if the IOH
  worker is unwanted in CI: re-base the composite battery (larger
  budgets, easier family mix, or relaxed tolerance) until the median
  score sits in 0.3–0.6 and <10% of mutations measure Δ=0.
- **Registry: a dedicated loop registry** — **shipped 2026-06-10** as
  :func:`panobbgo.harness._make_loop_strategies`, opt in via
  ``scripts/self_improve.py run --registry loop``.  Returns the two
  quick specs plus five compact family specs (``Loop_DE_Family``,
  ``Loop_PSO``, ``Loop_RegionUCB``, ``Loop_LocalSearch``,
  ``Loop_Restart``) — every tunable kwarg of LSHADE / JSO /
  NLSHADE_RSP / NLSHADE_LBC / LSHADE_EpSin / PSO / RegionUCB / COBYQA /
  LBFGSB / Restart is explicit at the constructor default.  Lifts
  catalog kwarg-rule activation from **4 / 44** under the quick
  registry to **44 / 44** under the loop registry.  See the
  2026-06-10 entry in `SELF_IMPROVEMENT_LOG.md`.
- 30–40 iterations, `--registry loop --structural --adaptive
  --structural-per-class-arms --adaptive-prime-from-ledger`, ≥5 reps,
  paired bootstrap.

### 9.2 Stage 2 — Confirmation (~15 min) [open]

§6.4: confirm screening accepts on a fresh iteration + hold-out seed
before recording.  Guard keeps running as a backstop
(`--guard-interval 10`).

### 9.3 Stage 3 — Cross-night codification [open]

`scripts/self_improve.py codify-scan --open-pr`:

- Scan the live ledger **plus archives in `planning/done/`** for
  directionally consistent confirmed accepts: same
  `(class, param, direction)` or structural arm, `k ≥ 2` confirmed
  accepts on distinct nights, pooled CI > 0.
- For each hit, open **one** codify PR editing the seed spec /
  constructor default, with the ledger evidence
  (`base_seed`, `randomize_iteration`, iterations, deltas, CIs) in the
  PR body.  Dedup first: `gh pr list --state open` — skip if a codify
  PR for the same `(class, param)` exists (§12.3 step 0 lesson,
  enforced in code).
- **Merged codify PRs are the persistence mechanism**: the next night
  reads the improved defaults from source.  No in-memory ladder
  serialization needed.

### 9.4 Target invocation

```yaml
uv run python scripts/self_improve.py run \
  --iterations 35 --mode quick --metric aocc \
  --registry loop \
  --adaptive --adaptive-prime-from-ledger --prime-include-archives \
  --structural --structural-per-class-arms \
  --confirm-accepts \
  --holdout-base-seeds 7,1234 \
  --guard-interval 10 \
  --ledger planning/self_improve_ledger.jsonl
uv run python scripts/self_improve.py codify-scan --open-pr
```

(`permissions: pull-requests: write` in the workflow.)  ``--registry
loop`` shipped 2026-06-10 (§9.5 step 1); the other open flags above
remain queued.  Until they exist, the V1 invocation in the workflow
file stands.

### 9.5 Implementation order (one PR each)

1. ~`make_loop_strategies()` + `--registry loop`~ — **shipped
   2026-06-10**.  See §9.1 above and the dated entry in
   `SELF_IMPROVEMENT_LOG.md`.  Catalog kwarg-rule coverage on the
   seed lifted from 4 / 44 to 44 / 44 — the nightly cron can now
   pick `--registry loop` so the dormant catalog actually fires.
2. Nightly to `--metric aocc` (or battery re-base), after one manual
   `workflow_dispatch` A/B comparing signal quality.
3. `--confirm-accepts` (§6.4) + graded bandit reward (§7.4) + no-op
   detection (§12.4).
4. `codify-scan --open-pr` + `--prime-include-archives` +
   vacuous-holdout fix + summary trend block.
5. Flip the workflow to §9.4; enforce the catalog freeze (§7.3).

## 10. Open questions / known constraints

- **Compute**: `--standard` ≈ 240 runs/iteration; GitHub-hosted runner
  has 2 cores.  Standard-mode nightly needs a self-hosted runner; the
  V2 answer is to fix metric resolution first so quick-class budgets
  carry signal.
- **Dimension stability**: resolved 2026-05-02 via stratified dimension
  sampling (`ProblemFamily.stratify_dims`, cyclic rep→dim assignment).
- **Accept rate**: V1's answer (bandit + eps relaxation) treated the
  symptom; V2 treats the cause (metric resolution, §2.1).
- **Coordination with human PRs**: the loop never touches source, so
  races are limited to ledger commits (push-retry in the workflow
  handles them).  Codify PRs (§9.3) go through normal review.

## 11. Success criteria (V2 — over the first 30 nights)

1. **Resolution**: median per-night seed score in the metric's
   responsive range; exactly-zero-delta iterations < 10%.
2. **Throughput**: ≥ 3 codify PRs opened from ledger evidence; ≥ 2
   merged.
3. **Durability**: merged codify changes re-confirmed by the next
   night's seed measurement; zero guard rollbacks of *confirmed*
   accepts.
4. **Honesty**: zero vacuous hold-outs reported as OK; every codify PR
   body carries reproducible evidence.

If criterion 2 is still 0 after 30 nights, the kwarg-mutation space
around the current seeds is exhausted at this budget — switch to
strategy-level search (strategy-class swap, `StrategyPhased`
composition; see the backlog in `SELF_IMPROVEMENT_LOG.md`), not more
kwarg arms.

## 12. Nightly cron and the ledger feedback path

Two loops feed the same ledger: the **nightly cron** (measures,
appends, commits — never edits source) and the **daily coding routine**
(reads trends, codifies persistent wins via PRs).

### 12.1 Persisted artifacts

| Artifact | Purpose | Consumed by |
|---|---|---|
| `planning/self_improve_ledger.jsonl` | Append-only record of every iteration / guard / hold-out / confirm decision | next night's bandit priming; codify-scan; drill-down |
| `planning/done/self_improve_ledger_*.jsonl` | Rotated archives (rotate at >2000 records, only via `planning/done/`) | bandit priming (`--prime-include-archives`, open); codify-scan |
| `planning/self_improve_summary.txt` | Latest `summary` output, overwritten nightly | daily routine at-a-glance |
| GH Actions artifact (30 d) | same files per-run | debugging a specific night |

### 12.2 What the cron does *not* do

It never commits changes under `panobbgo/` and (until §9.3 ships)
never opens PRs.  Accepted mutations live in the in-memory ladder for
the run; durable improvement happens only through codification.

### 12.3 Daily routine checklist

0. **Deduplicate before picking a task**: `gh pr list --state open`
   (drafts included).  The nightly routine branches from `master` and
   cannot see unmerged work — skipping this check produced four
   duplicate NL-SHADE-RSP PRs (#227–#230).  Open PRs, not the backlog
   list, are the source of truth for in-flight work.
1. **Skim `planning/self_improve_summary.txt`**: rules with positive
   confirmed-accept history, guard anomalies, hold-out drift.
2. **Codify persistent wins**: a rule with repeated confirmed accepts
   in one direction → PR changing the default kwarg (and re-centering
   the catalog bounds), citing ledger evidence.  Structural winners →
   promote into the registry factories in `panobbgo/harness.py`.
3. **Treat hold-out overfit flags as bugs** (§11.3 is failing), not
   noise.
4. **Never hand-edit the ledger.**  To start fresh, archive it to
   `planning/done/` and let the cron create a new one.
5. **Respect the catalog freeze (§7.3)** — prefer merging and
   measurement work over new arms.

### 12.4 Telemetry rules (V2 — open)

- **No-op detection**: an iteration whose per-pair results are
  bit-identical to baseline records `reason_skipped="no_op"` and does
  not count as a bandit pull.
- **Vacuous hold-outs** (`top_iter == -1`) report `status="vacuous"`
  and are excluded from drift aggregation.
- **Summary trend block**: per-night seed score (both metrics),
  accept / confirm / no-op rates, top-10 and bottom-5 bandit
  posteriors.  This — not raw JSONL — is what the daily routine reads.

## 13. Iteration log

Moved to `planning/SELF_IMPROVEMENT_LOG.md` (2026-06-09), including
the "Next iteration ideas" backlog.  External references to "§13" of
this document point there.
