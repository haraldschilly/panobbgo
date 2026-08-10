# GOAL — Converge on a world-class black-box optimizer

**Audience**: any agent (or human) picking up this repository with the standing
instruction "improve panobbgo". This file is the durable goal contract; read it
first, then act through the operating loop below. Everything referenced here
already exists and is tested — no setup beyond `uv sync --extra dev` and
`cd tools/ioh_worker && uv sync`.

---

## 1. North star (the measurable goal)

Maximize **mean AOCC** (Area Over the Convergence Curve, the IOHprofiler /
MA-BBOB Anytime competition metric) of the best panobbgo strategy on the
MA-BBOB battery, at competition-style budgets, without regressing the frozen
`composite_score` contract on the internal battery.

Concretely, in priority order:

1. **Beat the internal floor**: `Rewarding_Restart` (the competition candidate
   spec in `panobbgo/harness_ioh.py::make_ioh_strategies`) must dominate
   `RoundRobin_Random` on every battery tier (quick / standard / full).
2. **Beat the external baselines**: `Baseline_SciPyDE` and
   `Baseline_SciPyAnneal` (`panobbgo/harness_baselines.py`) on mean AOCC at
   the standard battery. Random search is the hard floor — never lose to it.
3. **Approach competition level**: the MA-BBOB Anytime competition regime is
   budget `2000·d`, dims 2 and 5, ~1000 affine instances. World-class 2024/2025
   entries were LLaMEA-generated hybrids (LLM-designed metaheuristics — i.e.
   structurally the same loop this repo runs on itself). The long-run target is
   a strategy competitive with tuned modular CMA-ES / L-SHADE-lineage hybrids
   with warm-started local search at that regime.

**Metric of record**: mean AOCC from `scripts/ioh_benchmark.py` /
`--metric aocc` self-improvement runs. `composite_score` is the frozen
legacy contract — keep it green, don't optimize for it.

## 2. State snapshot (2026-07-30 — update when it materially changes)

* Nightly cron (`self_improve_nightly.yml`, 03:00 UTC) runs 20 mutation
  iterations on `--metric aocc`, quick IOH battery, with confirmation gate,
  guard, multi-seed hold-out; appends to
  `planning/self_improve_ledger_aocc.jsonl` and commits summary + codify-scan
  reports.
* Seed score plateau: quick-battery mean AOCC ≈ 0.33–0.36 for weeks — the
  loop found improvements nightly but the codify last-mile was broken for the
  aocc regime until 2026-07-30 (metric-aware routing fix + first aocc codify:
  dropped the `Restart` analyzer from `Rewarding_Restart`, 17 nights of
  evidence, local A/B `0.3538 → 0.3922`).
* Known competitive gap: hold-out base seeds (7, 1234) score far below the
  training seed (0.04 vs 0.33 on 2026-07-30) — instance-family sensitivity
  is the biggest open weakness. Higher-dim (5-D) rotated valleys are the
  second (see `--extra-highdim` and the 2026-07-06..13 log entries).

## 3. Operating loop (one agent session ≈ one iteration)

Run this every session; it is deliberately mechanical. The nightly cron
produces evidence while you're away — your job is to bank it and then push the
frontier.

```bash
# 0. Sync + sanity (both venvs; IOH tests skip without the worker venv)
uv sync --extra dev && (cd tools/ioh_worker && uv sync)

# 1. Read the overnight state (committed nightly by the cron)
cat planning/self_improve_summary.txt          # trend, bandit posteriors
cat planning/self_improve_codify_scan.txt      # actionable evidence

# 2. Bank the evidence: apply the top cross-night candidate
uv run python scripts/self_improve.py codify-scan --metric aocc \
    --apply-top --apply-dry-run                # inspect first
uv run python scripts/self_improve.py codify-scan --metric aocc \
    --apply-top --apply-format --apply-run-tests

# 3. Verify with a local A/B on the metric of record
uv run python scripts/ioh_benchmark.py run --quick --output /tmp/before.json   # on master
#   ...apply change...
uv run python scripts/ioh_benchmark.py run --quick --output /tmp/after.json
uv run python scripts/ioh_benchmark.py compare /tmp/before.json /tmp/after.json

# 4. Full gate before PR
uv run pytest -q && uv run ruff check && uv run ruff format --check

# 5. One PR per change, evidence in the body (see AGENTS.md
#    "Agent-driven improve X PRs" for the evidence-form rules).
```

Before opening a PR, check open PRs for duplicates (the nightly branches from
master and cannot see unmerged work).

## 4. Multi-day convergence protocol

A multi-day run is the loop above plus an escalation ladder. Each day:

1. **Bank** (steps 1–5 above) — codify accumulated ledger evidence. This is
   always first: unbanked evidence is re-discovered and wasted every night.
2. **Diagnose** — find the sharpest measured gap. Sources, in order:
   hold-out drift records (instance-family generalization), per-problem AOCC
   breakdown from a `--standard` run, `--extra-highdim` families, the
   baseline comparison (`--baselines`).
3. **Attack one gap** with a *measured* algorithmic change (new heuristic
   kwarg, warm-start, schedule, or structural mix). The 2026-07-05..12 log
   entries (NP_init="auto", warm-started L-BFGS-B, quadratic Nearby) are the
   template: isolate the mechanism, measure at the dimension/regime where the
   effect lives, ship §7.3-freeze-compliant.
4. **Widen the evidence base** when the quick battery saturates: promote
   validation runs to `--standard` (locally or via `workflow_dispatch` with
   `mode=standard`), and check the plain-BBOB / higher-dim regimes so gains
   aren't MA-BBOB-quick-battery overfit.
5. **Log** — dated entry in `planning/SELF_IMPROVEMENT_LOG.md` (what was
   measured, what shipped, next ideas). The log is the loop's long-term
   memory; unlogged negative results get retried by future sessions.

Cadence guardrails:

* Never ship an optimizer-behavior change without a paired A/B on the metric
  of record (nightly ledger evidence counts; see AGENTS.md evidence forms).
* One codify slot per PR. Independent evidence ≠ joint evidence — don't batch
  three "individually positive" drops into one unmeasured combination.
* The composite-score formula and the default randomized battery are frozen
  contracts. Extend via opt-in flags (`--extra-highdim` pattern), never edit.
* If the quick-battery score stalls for >1 week with evidence banked, the
  bottleneck is the *measurement regime*, not the mutations — move up the
  ladder (standard battery, more instances, higher dims) before inventing
  new mutations.

## 5. Research backlog (SOTA-informed, 2026-07)

Ordered by expected value; each item should enter through the loop above.

1. **Instance-family generalization** — close the training-seed vs hold-out
   gap (0.33 vs 0.04). Suspects: Sobol-heavy initial design overfit to the
   training instances' scale; missing restart diversity after the Restart
   analyzer drop. Measure per-instance AOCC spread first.
2. **CMA-ES arm** — *shipped 2026-08-06*: the existing `CMAES` heuristic
   (hand-rolled (μ/μ_w, λ)-ES with IPOP/BIPOP restart) is now a structural
   catalog candidate, so the bandit measures it against the DE family
   nightly.  Direct addition to `Rewarding_Restart` was flat on a 12-seed
   paired quick-2-D A/B (CI95 [-0.0113, +0.0123]) — the open question is
   whether the arm earns pulls at 5-D / standard regimes where covariance
   adaptation should pay; watch the `add_heuristic` posterior and ledger.
3. **Rank-based acceptance stats** — mean-AOCC deltas are outlier-sensitive;
   competition practice is Wilcoxon / Friedman over (function, instance)
   pairs. Add as an alternative `statistical_accept` mode.
4. **Plain-BBOB cross-validation battery** — 24 BBOB functions, dims
   {2, 3, 5, 10}, as an opt-in hold-out suite (the `ioh` package already
   provides them through the same worker protocol).
5. **Anytime-aware strategy scheduling** — AOCC rewards early descent;
   panobbgo's rewarding strategy re-weights on "new best" events only.
   Explore time-decayed rewards / explicit budget-phase schedules.
6. **Behavior-space diagnostics** (LLaMEA-SAGE direction) — log per-run
   trajectory features (dispersion, basin-jump counts) into the ledger so
   codify-scan can correlate *why* an arm wins, not just that it does.
7. **Learned "intuition layer" (Dynamic Algorithm Configuration)** — a small
   dense policy network that watches run-time progress and re-weights /
   switches heuristics mid-run.  `StrategyRewarding`'s bandit is the
   memoryless special case; the generalization is a stateful policy:

   * **Observations** (per batch of results): budget fraction consumed,
     best-so-far improvement rate over the last k batches, stagnation
     length, per-heuristic reward distribution, point-cloud dispersion,
     Splitter depth stats — cheap features already derivable from the
     eventbus (`on_new_results` / `on_new_best`).
   * **Action**: a weight vector over the active heuristics (drop-in for
     the rewarding strategy's bandit weights), optionally a restart /
     phase-switch signal.
   * **Training**: evolution strategies (CMA-ES / OpenAI-ES) over the
     policy weights — no backprop through the optimization run needed.
     Fitness = mean AOCC over a stratified batch of randomized instances
     (`harness_randomized` families / MA-BBOB instances); hold-out
     base seeds catch policy overfit exactly as they do for the mutation
     loop.  A few thousand policy evaluations × quick-battery cost is
     feasible on the existing parallel harness.
   * **Deliverable path**: (a) feature extractor as an analyzer publishing
     a `progress_features` event, (b) `StrategyLearned` consuming it with
     a hand-set linear policy (sanity baseline), (c) ES meta-training
     script writing the trained weights as a versioned artifact, (d) the
     trained policy enters the nightly battery as one more spec —
     measured, not trusted.  Item 6's diagnostics are the natural feature
     source, so build 6 first.

   Literature anchors: Dynamic Algorithm Configuration (Biedenkapp et al.,
   2020+), adaptive operator selection, learning-to-optimize; Nevergrad's
   NGOpt is a hand-crafted (non-learned) version of the same switching
   idea.

References: MA-BBOB generator (Vermetten et al., ACM TELO 2024);
IOHprofiler competitions (iohprofiler.github.io/competitions); LLaMEA
(arXiv:2405.20132); CEC winners longitudinal analysis (arXiv:2603.24140);
benchmarking best practice (arXiv:2007.03488).

## 6. Done criteria (revisit quarterly)

The goal is met for a given quarter when:

* mean AOCC of the best spec on the **standard** IOH battery improves
  quarter-over-quarter with a bootstrap CI excluding zero, and
* the best spec beats `Baseline_SciPyDE` and `Baseline_SciPyAnneal` on the
  same battery, and
* hold-out drift is non-negative (no overfit verdicts), and
* every shipped change is traceable to ledger or A/B evidence.
