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
   are upward noise spikes (winner's curse).  *2026-06-12 update:* the
   §12.4 no-op detection now flags those zero-information iterations
   and excludes them from bandit pulls so the posterior is no longer
   mis-trained on them — but the underlying metric resolution problem
   stands until the §9.5 step 2 (`--metric aocc` or battery re-base)
   ships.
2. **Accept → rollback churn.**  15/16 guard checks rolled the ladder
   back; all hold-out records ran on an empty ladder (`top_iter=-1`,
   vacuous `drift=0.0000` reported as OK).  *Honesty bug fixed
   2026-06-11 (§6.4 / §12.4): vacuous hold-outs now surface as
   ``status="vacuous"`` / ``VACUOUS`` in the CLI and are excluded from
   the drift bootstrap.*  *Structural fix shipped 2026-06-14 (§6.4):
   the same-night confirmation gate (``LoopConfig.confirm_accepts`` /
   ``--confirm-accepts``) re-measures every screening-accepted
   candidate on a fresh ``randomize_iteration`` (plus the first
   hold-out base_seed when configured) and re-runs
   ``statistical_accept`` on the pooled sample — a screening noise
   spike can no longer drive a promotion because the confirmation
   batch is independent and the pooled CI rules it out.  Failed
   confirmations land as ``LoopConfirmRecord`` (``record_type=
   "confirm_reject"``) and the bandit consumes the post-confirmation
   reward.  *2026-06-21 update: the §9.5 step 5 partial flip puts
   every no-cost V2 flag in the live cron (``--registry loop`` /
   ``--prime-include-archives`` / ``--structural-per-class-arms`` /
   ``--bandit-reward graded`` / multi-seed hold-out / etc.) but the
   ``--confirm-accepts`` lever stays off pending a manual
   ``workflow_dispatch`` A/B because it's the only V2 flag with
   meaningful per-iteration cost.  Symptom (1)-(5) of §2 should
   measurably improve on the next 2-3 nights from the bandit
   activations alone; symptom (2) "Accept → rollback churn" stays
   open until the confirm-gate flip lands.*
3. **Nothing persists between nights.**  The ladder is in-memory; the
   only durable channel is manual codification (used once:
   `Sobol.scramble=False`, 2026-05-31 — which *worked*).
4. **Catalog ≫ registry mismatch.**  The nightly runs quick mode, whose
   registry is 2 simple strategies; most kwarg rules shipped since
   mid-May target heuristics that never appear in the nightly run.
   *2026-06-21 update: closed.  The nightly cron now passes
   ``--registry loop`` (loop registry shipped 2026-06-10), lifting
   catalog kwarg-rule activation from 4 / 44 to 44 / 44.  See the
   2026-06-21 entry in ``SELF_IMPROVEMENT_LOG.md``.*
5. **Compute ~94% idle** (20 quick iterations ≈ 5 min of a 90-min
   budget) while statistical power is the binding constraint.
6. **Bandit starved**: binary accept reward at ~2.5% base rate, and
   priming reads only the current ledger — archives in `planning/done/`
   are invisible.  *2026-06-15 update:* graded reward shaping
   (§7.4, shipped 2026-06-13) addresses the first half; the
   `--prime-include-archives` flag and matching
   `LoopConfig.adaptive_prime_include_archives` /
   `adaptive_prime_archive_dir` fields shipped 2026-06-15 (this entry)
   address the second half — the bandit now accumulates evidence
   across every retained nightly run rather than just the current one.
   See §9.5 step 4 and the dated entry in `SELF_IMPROVEMENT_LOG.md`.
   *2026-06-21 update: both fixes are now active in the live cron
   (the §9.5 step 5 partial flip wires ``--bandit-reward graded``
   and ``--prime-include-archives`` into the nightly invocation).*

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
confirm gate now in place (shipped 2026-06-14), a guard rollback of a
*confirmed* accept is an anomaly worth surfacing, not routine cleanup.

### 6.4 Same-night confirmation gate (V2 — shipped 2026-06-14)

The V1 flow recorded accepts straight from the screening measurement;
the guard then rolled back ~all of them (§2.2).  V2 inverts this:
**promotion requires confirmation before the accept is recorded.**

- ~Re-measure every screening-accepted candidate on a fresh
  `randomize_iteration` *and* on the hold-out base_seed, same night.~
  — **shipped 2026-06-14** as :attr:`LoopConfig.confirm_accepts` +
  ``--confirm-accepts``.  Every screening-accepted candidate is
  re-measured on ``iteration + LoopConfig.confirm_iteration_offset``
  (default ``500_000``, distinct from the guard's ``1_000_000`` so the
  two fresh-seed streams never collide), and the
  :func:`~panobbgo.harness.statistical_accept` rule is re-run on the
  *pooled* (screen + confirm) sample.  When at least one hold-out
  base_seed is configured the confirmation step additionally
  re-measures on the *first* hold-out seed and pools that too — per-
  iteration cost bounded at ``≤ 3×`` screening regardless of how many
  hold-out seeds the end-of-loop drift check walks.
- ~Promote iff the pooled paired CI (screen + confirm) stays > 0.~ —
  **shipped 2026-06-14**.  Same gate logic as the screening step
  (Δ > eps_accept, ci_low > 0, no catastrophic per-pair regression);
  see the dated entry in `SELF_IMPROVEMENT_LOG.md`.
- ~Failed confirmations are recorded (`record_type="confirm_reject"`)
  and count as bandit reward 0.~ — **shipped 2026-06-14** as
  :class:`LoopConfirmRecord`.  The bandit reward path consumes the
  *post-confirmation* pooled decision, so a screening noise-spike that
  the gate overturned collects the reject-regime reward
  (binary: ``0``; graded: ``clip(0.5 + pooled_Δ/(4·eps), 0, 0.5)``)
  rather than the accept reward the screening would have produced.
  The companion :class:`LoopIterationRecord` carries the new
  ``confirmed: Optional[bool]`` field — ``None`` when no confirmation
  ran, ``True`` on promotion, ``False`` on confirm-reject — so codify-
  scan can distinguish "confirmed accept" (durable signal) from
  "screening accept overturned by the gate" (noise spike) without
  re-deriving the verdict from per-record fields.
- ~Hold-out records on an empty ladder must report `status="vacuous"`,
  never `OK drift=0.0`.~ — **shipped 2026-06-11**.
  :class:`panobbgo.self_improve.LoopHoldoutRecord` gains a ``status``
  field ``("ok" | "overfit" | "vacuous")`` and
  :func:`aggregate_holdout_drift` filters vacuous records out of the
  bootstrap so the CI is no longer pulled toward zero by the empty
  ladder.  CLI prints surface ``VACUOUS`` / ``VACUOUS_CI`` instead of
  ``OK`` / ``OK_CI``.  Legacy ledger lines (no ``status`` field)
  classify correctly via :meth:`LoopHoldoutRecord.effective_status`.
  See the dated entry in `planning/SELF_IMPROVEMENT_LOG.md`.

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

### 7.4 Bandit reward shaping (V2 — shipped 2026-06-13)

Replace binary accept-reward with a graded reward in [0, 1]:
`0` for no-ops (gated by :meth:`AdaptiveMutationSampler.discard_outcome`
since 2026-06-12); `0.5 + clip(ci_low/eps_scale, 0, 0.5)` for
accepts; `clip(0.5 + Δ/eps_scale, 0, 0.5)` for honest
rejections (real signal, wrong sign / too small), `eps_scale =
4·eps_accept`.  Beta arms update as `alpha += r, beta += 1−r` — the
:class:`MutationRuleStats` ``reward_sum`` field accumulates the graded
reward and :meth:`AdaptiveMutationSampler.sample` swaps it in for
``n_accepts`` in the Beta posterior, so an arm that consistently
produces small-positive deltas (``r ≈ 0.5``) becomes distinguishable
from an arm that produces clearly-harmful deltas (``r ≈ 0``).  Opt in
via ``LoopConfig.bandit_reward_shaping = "graded"`` /
``--bandit-reward graded`` (CLI default ``binary`` preserved so
existing invocations are byte-identical).  When V2 §6.4 ships, the
confirm-reject branch will route through the same code path — same
shape, just an extra terminal state.  See the 2026-06-13 entry in
`SELF_IMPROVEMENT_LOG.md`.

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

### 9.3 Stage 3 — Cross-night codification [detection shipped 2026-06-17, --open-pr still open]

`scripts/self_improve.py codify-scan` (detection): **shipped
2026-06-17** — the scanner reads the live ledger plus every archive
under `planning/done/`, groups every accepted iteration by
`(class_name, param_name, direction)` (numeric: `"up"` / `"down"`;
categorical: `repr(new_value)`; structural: the op name), and
surfaces every group with at least `--min-nights` (default `2`)
distinct accept dates **and** every contributing record's `ci_low > 0`
(toggle with `--no-require-positive-min-ci`).  Each candidate carries
a pooled point-delta CI (percentile bootstrap on the per-record
deltas), the per-record evidence, and the
:attr:`panobbgo.self_improve.CodifyCandidate.slot_key` tuple a
follow-up `--open-pr` driver will dedup against `gh pr list --state
open`.  Public library API:
:class:`panobbgo.self_improve.CodifyCandidate`,
:func:`panobbgo.self_improve.aggregate_codify_candidates`,
:func:`panobbgo.self_improve.load_ledgers_for_codify_scan`.
`--confirmed-only` restricts to post-V2-§6.4 records; `--json`
emits one `to_dict()` JSON per line; `--top N` truncates.  See the
2026-06-17 entry in `SELF_IMPROVEMENT_LOG.md`.

`scripts/self_improve.py codify-scan --open-pr` (still open) is the
follow-up that translates each surfaced candidate into a concrete
source edit + PR:

- For each hit, open **one** codify PR editing the seed spec /
  constructor default, with the ledger evidence
  (`base_seed`, `randomize_iteration`, iterations, deltas, CIs) in the
  PR body.  Dedup first: `gh pr list --state open` — skip if a codify
  PR for the same `(class, param)` exists (§12.3 step 0 lesson,
  enforced in code via the
  :attr:`CodifyCandidate.slot_key` tuple).
- **Merged codify PRs are the persistence mechanism**: the next night
  reads the improved defaults from source.  No in-memory ladder
  serialization needed.

`scripts/self_improve.py codify-scan --widen-bounds` (**shipped
2026-06-19**) is the sibling detection mode for *bidirectional*
patterns — slots whose codify-scan surfaces both `"up"` and `"down"`
direction candidates on the same `(class_name, param_name)`.  These
two are contradictory under the default-shift interpretation (which
direction wins?) but become a clean *catalog bound update* under the
right interpretation: focus the bandit's exploration on the observed
range, with a fixed multiplicative widen factor (default `1.5`) for
headroom outside the observed window.  See
:class:`panobbgo.self_improve.WideningCandidate`,
:func:`panobbgo.self_improve.detect_widening_candidates`, and the
2026-06-19 entry in `SELF_IMPROVEMENT_LOG.md` for the per-rule-kind
bound arithmetic (multiplicative for log_uniform_perturb /
float_uniform; outward-rounded for integer_add).  On the live ledger
today the detector surfaces `Nearby.radius` and `Sobol.n` — both
*tightening* candidates because the bandit consistently picks values
in a window 5-10× narrower than the catalog admits.  Pairs naturally
with the queued `--open-pr` driver: the
:attr:`WideningCandidate.slot_key` tuple matches
:attr:`CodifyCandidate.slot_key` so a future driver dedups uniformly
across both candidate kinds.  The 2026-06-22 *auto-tune widen factor*
follow-up (CLI ``--widen-auto-tune`` plus ``--widen-factor-min`` /
``--widen-factor-max``) sizes the widen factor per-candidate from the
observed-spread / catalog-bound ratio in the rule's natural scale
(log for ``log_uniform_perturb``, linear for ``integer_add`` /
``float_uniform``).  Narrow observed spread (high agreement) →
larger factor for exploration headroom; wide spread (low agreement)
→ smaller factor focused on the consensus.  Lifts the live
``Nearby.radius`` factor from a fixed 1.5 to ~2.31, opening the
proposed bound to ``[0.032, 0.313]`` instead of ``[0.049, 0.203]``.
*2026-06-26 update:* the ``Nearby.radius`` auto-tuned proposal has
been **manually codified** into :func:`default_catalog` (bounds
``(0.005, 0.5) → (0.032, 0.313)``) — the first widening-detector
output to land as a catalog change.  Re-running the detector against
the same ledger after the codify shows the auto-tune converges on
``[0.0345, 0.287]`` (the now-narrower catalog yields a smaller
spread ratio so the per-candidate factor settles near 2.12), which
sits effectively at the new bounds — the detector is
self-stabilising.  The ``Sobol.n`` bidirectional candidate is *not*
codified in the same change because the auto-tune classifies it as
``"widens current"`` rather than ``"tightens"`` (mixed signal); see
the 2026-06-26 entry in ``SELF_IMPROVEMENT_LOG.md``.

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
3. ~`--confirm-accepts` (§6.4)~ + ~graded bandit reward (§7.4)~ + ~no-op
   detection (§12.4)~ — **no-op detection shipped 2026-06-12**;
   **graded bandit reward shipped 2026-06-13**; **same-night
   confirmation gate shipped 2026-06-14**.  All three V2 sub-tasks
   closed; see the dated entries in `SELF_IMPROVEMENT_LOG.md`.
4. `codify-scan --open-pr` + ~`--prime-include-archives`~
   (**shipped 2026-06-15**) + ~vacuous-holdout fix~
   (**shipped 2026-06-11**) + summary trend block.  *Detection half of
   `codify-scan` shipped 2026-06-17 (no `--open-pr` yet — that's the
   queued follow-up); `--prime-include-archives` shipped 2026-06-15
   (closes the §2.6 second half); summary trend block shipped
   2026-06-16.*
5. Flip the workflow to §9.4 — **partially shipped 2026-06-21**
   (see ``SELF_IMPROVEMENT_LOG.md`` entry).  All zero-cost V2 flags are
   now in the nightly cron: ``--registry loop``,
   ``--prime-include-archives``, ``--structural-per-class-arms``,
   ``--bandit-reward graded``, ``--inactivity-relax-after 10``,
   ``--holdout-base-seeds 7,1234``, ``--guard-interval 10``.  The
   remaining toggle is ``--confirm-accepts`` (held back pending a
   manual ``workflow_dispatch`` A/B because it carries ~2-3× per-
   iteration compute cost) and ``--metric aocc`` (queued at step 2,
   needs the IOH worker on the runner).  Enforce the catalog freeze
   (§7.3).

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
   merged.  *2026-06-28 update: ``3 codify PRs opened`` —
   ``Sobol.scramble=False`` (2026-05-31, merged), ``Nearby.radius``
   catalog-bound tightening (2026-06-26, merged), and
   ``Nearby.radius`` seed shift (2026-06-28, this entry's companion
   PR).  Two merged so far; the third is in flight.*
3. **Durability**: merged codify changes re-confirmed by the next
   night's seed measurement; zero guard rollbacks of *confirmed*
   accepts.
4. **Honesty**: zero vacuous hold-outs reported as OK; every codify PR
   body carries reproducible evidence.  *Vacuous-hold-out telemetry
   shipped 2026-06-11 — the "reported as OK" half of this criterion is
   structurally closed; what remains is operator vigilance that the
   evidence in codify PRs continues to be reproducible.*

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

### 12.4 Telemetry rules

- **No-op detection** — **shipped 2026-06-12**.  Every
  :class:`LoopIterationRecord` carries a ``no_op: bool`` field;
  iterations whose per-(problem, strategy) candidate scores are
  bit-identical to baseline set ``no_op=True``,
  ``reason_skipped="no_op"``, and ``accepted=False``.  The bandit's
  ``record_outcome`` is *not* called on these iterations (the new
  :meth:`AdaptiveMutationSampler.discard_outcome` clears the pending
  arm without an update), and ``prime_from_ledger`` skips them on
  resume.  Directly addresses §2.1 ("34% of mutations measure Δ =
  exactly 0.0000") — those iterations no longer mis-train the
  posterior.  See the §13 entry.
- ~**Vacuous hold-outs** (`top_iter == -1`) report `status="vacuous"`
  and are excluded from drift aggregation.~ — **shipped 2026-06-11**
  on :class:`panobbgo.self_improve.LoopHoldoutRecord` (new ``status``
  field, ``effective_status`` helper for legacy records) and
  :func:`aggregate_holdout_drift` (``vacuous_count`` / ``all_vacuous``
  reductions; vacuous records filtered out of the bootstrap so the CI
  is no longer biased toward zero by empty ladders).  CLI prints
  surface ``VACUOUS`` / ``VACUOUS_CI`` in both ``run`` and ``summary``
  modes.  See the dated entry in `planning/SELF_IMPROVEMENT_LOG.md`.
- ~**Summary trend block**: per-night seed score (both metrics),
  accept / confirm / no-op rates, top-10 and bottom-5 bandit
  posteriors.~ — **shipped 2026-06-16**.  ``scripts/self_improve.py
  summary`` now renders three additive sub-blocks after the existing
  per-record sections: (1) a **Trend** table with one row per loop run
  (oldest first) carrying date / base_seed / mode / iters / decided /
  accepts / no-op / best Δ / seed score; (2) **Bandit posteriors**
  ranked by graded ``mean_reward`` with configurable ``--top-n``
  (default 10) / ``--bottom-n`` (default 5) / ``--min-attempts``
  (default 3); (3) **Inactivity** telemetry surfacing the longest
  accept drought, the relaxed-accept count, and the mean decay factor
  at the moment of accept.  The trend block is what the §12.3 daily
  routine reads — not raw JSONL.  Implementation: three new helpers
  in ``scripts/self_improve.py`` (``_group_runs``, ``_print_trend_block``,
  ``_replay_bandit_posteriors`` / ``_print_bandit_block``,
  ``_print_inactivity_block``) plus three CLI flags on the ``summary``
  subcommand.  20 new tests in ``TestSummaryTrendBlock``.  See the
  2026-06-16 entry in `planning/SELF_IMPROVEMENT_LOG.md`.

## 13. Iteration log

Moved to `planning/SELF_IMPROVEMENT_LOG.md` (2026-06-09), including
the "Next iteration ideas" backlog.  External references to "§13" of
this document point there.
