# Self-Improvement Iteration Log

Append-only history of algorithmic improvements applied to Panobbgo
*outside* the autonomous loop, plus the rolling "Next iteration ideas"
backlog at the bottom.  Split out of `SELF_IMPROVEMENT_LOOP.md` (where
it was §13) on 2026-06-09 so the loop document stays a compact,
readable instruction file.

Conventions:

* One dated `###` entry per shipped change — newest first.  Each entry
  references the PR / commit that landed it, the rationale, and a
  measured-impact number when available.
* Section references like §6.2 / §7.2 / §12.3 point into
  `SELF_IMPROVEMENT_LOOP.md`; "see the §13 entry" in older text means
  "see the dated entry in this file".
* Graduate items from "Next iteration ideas" to a dated entry when
  shipped.

### 2026-06-22 — Auto-tune widen factor from observed spread (V2 §9.3 follow-up)

* **What** — Closes the *Auto-tune widen factor from observed spread*
  follow-up seeded under *Next iteration ideas* on 2026-06-19.  Pure
  additions to :mod:`panobbgo.self_improve` plus three CLI flags on
  ``scripts/self_improve.py codify-scan``:

  * :func:`panobbgo.self_improve._auto_tune_widen_factor` — sizes a
    widen factor from the ratio of observed-spread to catalog-bound
    span.  Narrow observed spread (high agreement across nights) →
    larger factor for exploration headroom; wide spread (low agreement)
    → smaller factor focused on the consensus.  Spread is measured in
    the rule's natural scale: log-space ratio for
    ``log_uniform_perturb``, linear ratio for ``integer_add`` /
    ``float_uniform``.  Linear interpolation between
    ``auto_tune_max_factor`` (at ratio = 0) and
    ``auto_tune_min_factor`` (at ratio = 1).  When no catalog rule
    targets the slot — the relative-spread signal is unavailable — the
    helper returns the caller-supplied ``fallback`` instead.
  * :func:`detect_widening_candidates` gains three keyword arguments
    — ``auto_tune: bool = False``, ``auto_tune_min_factor: float =
    1.1``, ``auto_tune_max_factor: float = 2.5`` — that opt in to the
    per-candidate sizing.  Default ``auto_tune=False`` keeps every
    existing invocation byte-identical.  The auto-tuned factor lands
    in :attr:`WideningCandidate.widen_factor` so the report and JSON
    output show the actually-used factor, not a global default.
  * CLI surface on ``scripts/self_improve.py codify-scan``:
    ``--widen-auto-tune`` (off by default), ``--widen-factor-min``
    (default ``1.1``), ``--widen-factor-max`` (default ``2.5``).  The
    pre-existing ``--widen-factor`` (default ``1.5``) is repurposed
    as the fallback for slots with no catalog rule.  The
    *Bound-widening candidates* report header switches from
    ``widen_factor=1.5`` to ``widen_factor=auto-tune [1.1, 2.5]
    (fallback=1.5)`` when the flag is set, so the operator can see at
    a glance which sizing rule produced each surfaced bound.

* **Why** — The 2026-06-19 widening detector ships a single fixed
  ``widen_factor`` (default ``1.5``) applied to every bidirectional
  pair.  This is a sensible starting point but is one-size-fits-all
  across rules whose observed-spread / catalog-span ratios differ by
  an order of magnitude:

  * **Live ledger today (15 confirmed nights):**
    - ``Nearby.radius`` — observed ``[0.0733, 0.1353]``, catalog
      ``[0.005, 0.5]``.  Log-space ratio
      ``log(0.1353 / 0.0733) / log(0.5 / 0.005) ≈ 0.133`` — narrow
      observed window inside a wide catalog.  Auto-tuned factor:
      ``2.5 - 1.4 * 0.133 ≈ 2.31``, vs the previous fixed ``1.5``.
      Proposed bound: ``[0.0317, 0.3130]`` (vs ``1.5 ×`` baseline's
      ``[0.0489, 0.2030]``) — meaningfully more headroom outside the
      consensus window where the bandit might find the next win.
    - ``Sobol.n`` — observed ``[8, 24]``, catalog ``[4, 64]``.  Linear
      ratio ``16/60 ≈ 0.267`` — narrowish but not as narrow as
      Nearby.radius.  Auto-tuned factor: ``2.5 - 1.4 * 0.267 ≈ 2.13``,
      vs ``1.5``.  Proposed bound: ``[3, 52]`` (vs ``[5, 36]``) —
      widens the catalog rather than tightens it (the ``1.5 ×``
      baseline was tightening), because the observed window is large
      enough that a generous widen makes the proposed bound exceed the
      catalog's current upper end.

    Both proposals are still *measured against the same ledger
    evidence* the operator was triaging before this ship — auto-tune
    doesn't change the input, just the bound-arithmetic.  The
    operator's actionable lever shifts from "the catalog admits 5-10×
    more range than the bandit actually uses" (true with fixed 1.5)
    to "the bandit has converged into a known window; widen the
    catalog around it" (the auto-tune lens).  Direct effect on §11
    V2 success criterion 2 (codify-PR throughput): the bound-update
    proposal the operator codifies is now sized to the observed
    evidence rather than a global heuristic, so a bandit-converged
    slot doesn't get a too-tight bound that would force the bandit to
    re-discover its own consensus.

  * **Conceptual rationale** — the planning doc's "Auto-tune widen
    factor from observed spread" entry under *Next iteration ideas*
    (the 2026-06-19 follow-ups block) framed the trade-off
    qualitatively: narrow → big factor (need headroom), wide → small
    factor (focus on consensus).  This ship is the concrete
    realisation, with the spread measured in the rule's natural scale
    so log-uniform-perturb and linear rules size correctly.  Pairs
    naturally with the queued ``--open-pr`` driver: the same
    :attr:`WideningCandidate.slot_key` tuple the codify-candidate path
    uses is reused here, so a future ``--open-pr`` driver will dedup
    uniformly across both candidate kinds *and* the auto-tuned bound
    will land in the PR body directly.

* **Backwards compatibility** — strictly safe.  ``auto_tune=False`` is
  the default on :func:`detect_widening_candidates`; ``--widen-auto-tune``
  is off by default on the CLI.  Every existing invocation produces
  byte-identical output.  Existing tests covering
  :func:`_widen_numeric_bounds`,
  :func:`detect_widening_candidates`, and the
  ``--widen-bounds`` CLI continue to assert the fixed-1.5 factor and
  the existing bound math — all 38 prior tests
  (``TestWidenNumericBounds`` + ``TestCatalogNumericBounds`` +
  ``TestDetectWideningCandidates`` + ``TestCodifyScanCLIWidening``)
  pass unchanged.  The pre-existing ``--widen-factor`` flag still
  controls the fixed-factor path; it doubles as the fallback for
  ``--widen-auto-tune`` when no catalog rule targets the slot, so
  existing operators have a clean opt-in path.

* **Tests** — 22 new tests across three new test classes:

  * ``TestAutoTuneWidenFactor`` (13 tests): the helper itself — narrow
    spread returns close to max_factor, wide spread returns close to
    min_factor, mid spread interpolates linearly, integer / float /
    log_uniform_perturb rule kinds use the correct scale, None
    current_bounds falls back to the supplied fallback, degenerate
    catalog (``cur_lo == cur_hi``) falls back, unsupported rule_kind
    (categorical / structural) falls back, log-kind with non-positive
    bounds falls back, observed range exceeding catalog clips to
    min_factor, ``min_factor <= 1.0`` / ``max_factor < min_factor`` /
    ``fallback <= 1.0`` raise ``ValueError``, and a custom
    ``[min_factor, max_factor]`` range propagates through.
  * ``TestDetectWideningCandidatesAutoTune`` (5 tests): auto-tune off
    by default produces byte-identical factor; auto-tune on sizes the
    factor per-candidate; the no-rule fallback path returns
    ``widen_factor``; ``WideningCandidate.widen_factor`` and
    :meth:`to_dict` carry the auto-tuned factor; a custom
    ``[auto_tune_min_factor, auto_tune_max_factor]`` range propagates
    through.
  * ``TestCodifyScanCLIAutoTuneWidening`` (4 tests): the
    auto-tune-off-by-default behaviour, the header label flips to
    ``widen_factor=auto-tune [min, max] (fallback=...)`` when the
    flag is set, the JSON-mode output carries the per-candidate
    factor, and a custom range via ``--widen-factor-min`` /
    ``--widen-factor-max`` propagates.

  Plus the existing ``TestCodifyScanCLIWidening._build_ns`` helper
  extended with the three new attributes (``widen_auto_tune``,
  ``widen_factor_min``, ``widen_factor_max``) so the existing CLI
  tests continue to pass with the namespace shape the new code reads.

  Test totals: 493 in ``tests/test_self_improve.py`` (471 before +
  22 new); 1653 in ``tests/`` (1 skipped — unrelated COCO wrapper).
  ``uv run --extra dev ruff format --check .`` /
  ``uv run --extra dev ruff check panobbgo/self_improve.py
  scripts/self_improve.py tests/test_self_improve.py`` /
  ``uv run pyright panobbgo/self_improve.py`` / 96 sphinx doctests
  all clean.

* **Documentation updated**
  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry; the *Auto-tune
    widen factor from observed spread* follow-up promoted from
    *Next iteration ideas* to shipped.
  - ``planning/SELF_IMPROVEMENT_LOOP.md``: §9.3 "Bidirectional-bound
    widening" line annotated with the auto-tune lever.
  - ``doc/source/guide.rst``: quick-nav entry extended to mention
    ``--widen-auto-tune`` alongside the existing ``--widen-bounds`` /
    ``--widen-factor`` flags.
  - ``doc/source/guide_benchmarking.rst``: new
    "Auto-tuned widen factor (``--widen-auto-tune``)" sub-paragraph in
    the "Bidirectional-bound widening" subsection documenting the
    spread → factor rule and the live-ledger evidence.
  - ``AGENTS.md``: self-improvement loop bullet annotated.
  - ``TODO.md``: new "Recent Improvements" entry.

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * **Per-kind widen factor range** — log-scale knobs naturally
    tolerate a larger max_factor than linear ones because log-space
    spread is dimensionally different.  A categorical
    ``--widen-factor-max-log`` / ``--widen-factor-max-linear`` pair
    (or a single flag with rule-kind-specific defaults) would let
    the operator tune per kind.  Speculative — the unified
    ``[1.1, 2.5]`` range is a reasonable starting point.
  * **Use the relative-spread signal in ``codify-scan --open-pr``** —
    when the queued ``--open-pr`` driver lands, the auto-tuned
    factor and the relative-spread ratio are both natural fields
    to surface in the PR body so the reviewer can see at a glance
    whether the proposal is widening (bandit hasn't explored the
    space) or tightening (bandit has converged).  The
    :class:`WideningCandidate` carries everything needed today;
    the only missing piece is a formatter on the ``--open-pr`` side.

### 2026-06-21 — Flip the nightly cron to the V2 substrate (V2 §9.5 step 5)

* **What** — Promotes the *Flip the nightly cron to `--registry loop`*
  follow-up (seeded after the 2026-06-10 ship) plus the no-cost
  V2 sub-flags into the live cron.  Single-file edit to
  ``.github/workflows/self_improve_nightly.yml``: the
  ``Run self-improvement loop`` step now invokes

  ```
  uv run python scripts/self_improve.py run \
      --iterations "$ITERATIONS" --mode "$MODE" \
      --registry loop \
      --adaptive --adaptive-prime-from-ledger \
      --prime-include-archives \
      --structural --structural-per-class-arms \
      --bandit-reward graded \
      --inactivity-relax-after 10 --inactivity-relax-factor 0.5 \
      --holdout-base-seeds 7,1234 \
      --guard-interval 10 \
      --ledger planning/self_improve_ledger.jsonl
  ```

  Promoted flags (all shipped weeks ago but dormant in the live loop):

  * ``--registry loop`` — §9.5 step 1, shipped 2026-06-10.  Lifts the
    seed's catalog kwarg-rule activation from 4 / 44 (quick seed,
    ``Sobol`` / ``Nearby`` / ``Sensitivity`` only) to 44 / 44 (loop seed
    explicit-default for every tunable kwarg on the rule-bearing
    classes — LSHADE / JSO / NLSHADE_RSP / NLSHADE_LBC / LSHADE_EpSin /
    PSO / RegionUCB / COBYQA / LBFGSB / Restart).  Per-iteration cost
    rises ~3.5× (2 → 7 specs) but the V1 §2.5 diagnosis reports 94% idle
    compute on the 90-min cap, so the 20-iteration count stays.
  * ``--prime-include-archives`` — §2.6, shipped 2026-06-15.  Replays
    every rotated ledger under ``planning/done/`` (matching
    ``self_improve_ledger_*.jsonl``) before the live ledger so the
    bandit posterior compounds across nightly rotation boundaries
    rather than forgetting every pre-rotation observation.
  * ``--structural-per-class-arms`` — §7.2 / shipped 2026-05-18.
    Expands each structural op into one Thompson arm per candidate
    class (e.g. ``add_heuristic`` becomes ``add_Sobol`` /
    ``add_Random`` / … as separate arms) so the bandit can
    distinguish per-class winners instead of collapsing the signal at
    the op level.
  * ``--bandit-reward graded`` — §7.4, shipped 2026-06-13.  Replaces
    the binary +1/+0 accept/reject signal with a continuous reward in
    ``[0, 1]`` derived from the bootstrap CI / point delta so honest
    near-miss rejects (``Δ ≈ 0``) carry ``r ≈ 0.5`` of evidence
    instead of zero.
  * ``--inactivity-relax-after 10 --inactivity-relax-factor 0.5`` —
    shipped 2026-05-30, recommended for the unattended cron in the
    docstring (the 1-5% documented accept rate routinely yields >10
    iter droughts).  Floored at ``--inactivity-min-eps-accept``
    (default ``0.001``, the bootstrap CI noise floor); re-tightened
    on the next accept; per-iteration ledger fields persist the
    effective threshold so the auditor can grep relaxed accepts
    separately.
  * ``--holdout-base-seeds 7,1234`` — shipped 2026-05-16.  Replaces
    the single-seed ``--holdout-base-seed 7`` with a two-seed sweep;
    worst-case drift / any-overfit reduction is more robust than a
    single independent draw.  The smoke test below confirms two
    LoopHoldoutRecord entries per run (one per seed), 5
    iterations each (10 holdout iterations total) — adds <10% to the
    quick-mode wall-clock.
  * ``--guard-interval 10`` (relaxed from 5) — §6.3.  The guard's
    role narrows as the catalog freeze (§7.3) settles; matches the
    §9.4 target invocation.

  Not flipped here (intentional):

  * ``--confirm-accepts`` — §6.4, shipped 2026-06-14.  Adds 2-3× per-
    iteration cost (one re-measure on a fresh ``randomize_iteration``
    plus one per hold-out seed).  The companion *Flip the nightly
    cron to ``--confirm-accepts``* follow-up (still queued) flags
    that the iteration count needs halving and the trade-off should
    be measured via a manual ``workflow_dispatch`` A/B first.  This
    PR ships the no-cost flags so the V2 substrate is no longer
    dormant; ``--confirm-accepts`` is the next safe-to-ship lever.
  * ``--metric aocc`` — §9.5 step 2.  Needs the IOH worker available
    on the runner; the current cron stays on ``composite_score`` (the
    §9.1 fallback path of "re-base the composite battery" is also
    still queued).

* **Why** — Direct response to the §2 V2 diagnosis read off the
  current 15-night summary:

  * 15 nights × 20 iterations = 300 iterations, 7 accepts total
    (~2.3% accept rate).
  * 14 / 15 hold-out records report ``VACUOUS`` — the ladder was
    empty most nights so the hold-out had nothing to validate.
  * Top 8 bandit posteriors include exactly the 4 rules that fire on
    the quick seed (``Nearby.radius`` 6/79, plus structural ops at
    0% accept rate) — every kwarg rule shipped against
    ``LSHADE`` / ``JSO`` / ``PSO`` / ``RegionUCB`` / ``COBYQA`` / etc.
    is dormant because the seed doesn't set those kwargs explicitly
    (the §2.4 "catalog ≫ registry mismatch" diagnosis).

  The infrastructure to fix this has been merged for weeks but the
  live cron was never flipped.  This PR is the literal one-line YAML
  edit (plus comments documenting which flags are queued for follow-
  up).  Expected lift:

  * The 44 currently-dormant kwarg arms become applicable on the
    seed, so the bandit can actually pull on them.  Even at the
    historical ~2.3% accept rate the per-night chance of finding a
    real win rises with the number of applicable arms.
  * Graded reward turns the bandit's ~2.5% binary information yield
    into ~65% (the §7.4 lift estimate) — every reject that's a near-
    miss starts contributing evidence instead of just noise.
  * Per-class arms split each ``add_*`` / ``drop_*`` op (currently
    aggregated) into ~7 arms each — same as above but for the
    structural bucket.
  * Archive priming gives the bandit a 531kb prior (the
    ``2026-05-31`` rotated archive in ``planning/done/``) on top of
    the 375-line live ledger.
  * Multi-seed hold-out catches the single-seed overfit blind spot
    that the §11 criterion 4 "honesty" requirement is the structural
    fix for.

  Speculative: the §2.2 "Accept → rollback churn" symptom (15/16 V1
  accepts rolled back by the guard) persists in this PR because
  ``--confirm-accepts`` is *not* flipped here.  Acceptable trade-off
  — the same-night confirmation gate is a heavier compute change
  that the §12.3 daily routine should pair with a manual
  ``workflow_dispatch`` A/B before flipping permanently.  Queued as
  the §9.5 step 5 *follow-up* (the queue entry seeded with the
  2026-06-14 ship is updated in this PR to reflect that the *other*
  step-5 flags are now live).

* **Smoke test** — Two 1-iteration runs against the new invocation:

  * Fresh ledger (``/tmp/test_v2_ledger.jsonl``) — exit code 0; the
    loop registry seed exercises the catalog; multi-seed hold-out
    produces 2 ``LoopHoldoutRecord`` entries
    (``worst_drift=+0.0028  overfit=0/2  vacuous=0/2``); bandit
    posterior listing shows all per-class arms primed at 0/0.
  * Primed from the live ledger (``planning/self_improve_ledger.jsonl``
    copied to ``/tmp/test_v2_ledger_primed.jsonl``) — exit code 0;
    bandit picks up the historical attempts at correct per-class
    granularity (e.g. ``NelderMead.add_heuristic[structural] -> 0/5
    (0%)``, ``Sensitivity.drop_analyzer[structural] -> 0/29 (0%)``,
    ``Restart.add_analyzer[structural] -> 1/23 (4%)``,
    ``Nearby.radius[log_uniform_perturb] -> 6/79 (8%)``) — confirming
    that ``prime_from_ledger`` + ``prime_from_archives`` correctly
    populate the per-class arms from legacy collapsed op-level
    records and that ``--registry loop`` doesn't break replay against
    a ledger that was generated under ``--registry default``.

* **Backwards compatibility** — Strictly safe: the only edit is to
  the workflow file's ``Run self-improvement loop`` shell step; no
  code changes, no test changes, no API changes.  ``workflow_dispatch``
  inputs (``iterations`` / ``mode``) remain unchanged so a manual run
  can still A/B the V1 invocation by editing the workflow file
  temporarily.  Existing ledger entries remain valid priors under the
  new invocation (the bandit's ``_proposal_rule_key`` collapses to
  ``(class_name, param_name, rule_kind, ...)`` independent of the
  structural arm split or the reward shape).  No ledger archive
  rotation is needed because the regime change preserves the per-arm
  semantics — graded reward is multiplicative on top of the binary
  reward (graded reward is identical in mean to binary on accepts /
  rejects with extreme deltas; differs only on the near-miss band
  that binary reward discards anyway) and the per-class arm split
  re-keys arms whose collapsed-op records were already counted
  against the aggregate.

* **Documentation** — ``planning/SELF_IMPROVEMENT_LOG.md``: this
  dated entry; the *Flip the nightly cron to ``--registry loop``*,
  *Flip the nightly cron to ``--confirm-accepts``* queue entries
  updated to reflect the partial flip; the V2 §9.5 step 5 progress
  noted.  ``planning/SELF_IMPROVEMENT_LOOP.md``: §9.5 step 5 status
  flipped from "open" to "partially shipped"; §2.6 / §2.2 entries
  annotated.  ``doc/source/guide_benchmarking.rst``: nightly-cron
  description updated to reference the new V2 invocation.
  ``AGENTS.md``: brief callout added.  ``TODO.md`` entry under
  Recent Improvements.

### 2026-06-19 — Mutation-bound widening detection for bidirectional codify candidates (V2 §9.3 follow-up)

* **What** — Closes the *Mutation-bound widening rule for
  bidirectional codify candidates* idea seeded under *Next iteration
  ideas* on 2026-06-17.  Three pure additions to
  :mod:`panobbgo.self_improve` plus a flag pair on the
  ``scripts/self_improve.py codify-scan`` subcommand that pair every
  bidirectional ``(class_name, param_name)`` slot — same slot with
  accepts in *both* ``"up"`` and ``"down"`` directions across multiple
  nights — into a proposed ``MutationRule.bounds`` update:

  * :class:`panobbgo.self_improve.WideningCandidate` — frozen
    dataclass carrying one bidirectional pair: ``class_name`` /
    ``param_name`` / ``rule_kind``, the catalog's current bounds (or
    ``None`` when no rule targets the slot), the observed range
    pooled across both directions, the proposed widened range, the
    widen factor used, the two contributing
    :class:`CodifyCandidate` instances (the ``up`` and ``down``
    flavors), and aggregate ``n_accepts`` / ``distinct_dates`` /
    ``slot_key`` (mirrors :attr:`CodifyCandidate.slot_key` so the
    follow-up ``--open-pr`` driver can dedup uniformly across both
    candidate kinds).  Carries the convenience
    :attr:`proposal_is_wider` / :attr:`proposal_is_tighter` flags
    so the CLI report can label the proposal direction at a glance.
  * :func:`panobbgo.self_improve.detect_widening_candidates` — the
    pairing primitive.  Walks a sequence of
    :class:`CodifyCandidate` instances, drops candidates that aren't
    kwarg-numeric (``op is not None`` or ``rule_kind not in
    {log_uniform_perturb, integer_add, float_uniform}``), groups by
    ``(class_name, param_name, rule_kind)``, and emits one
    :class:`WideningCandidate` per group that carries both
    directions.  Sorted by ``(n_distinct_nights desc, n_accepts
    desc, class_name asc)`` so the strongest bidirectional evidence
    surfaces first.  Looks up the current bound via
    :func:`_catalog_numeric_bounds` against the supplied catalog
    (default :func:`default_catalog`); callers using a non-default
    catalog can pass it explicitly.
  * :func:`panobbgo.self_improve._widen_numeric_bounds` — the bound
    arithmetic, factored out so the rule maths is unit-testable
    independently of the pairing logic.  Per-kind semantics:

    - ``log_uniform_perturb`` — multiplicative on both ends
      (``observed_lo / widen_factor``, ``observed_hi *
      widen_factor``).  Lower bound is floored at ``1e-12`` because
      :class:`MutationRule` rejects non-positive
      ``log_uniform_perturb`` values.  Symmetric in log space.
    - ``integer_add`` — same multiplicative rule, then rounded
      *outward* (:func:`math.floor` on the lower bound,
      :func:`math.ceil` on the upper).  Lower bound is clipped to
      ``1`` when ``observed_lo`` is positive — most integer-typed
      catalog kwargs are pool sizes / iteration counts where zero
      would be degenerate.  Sign-preserving for negative observed
      values (defensive against future negative-int kwargs).
    - ``float_uniform`` — multiplicative on absolute values;
      preserves the sign so a negative-valued knob widens away from
      zero on both sides.  ``observed_lo == 0`` is preserved at
      zero (the operator likely wants the bound to start there).

  Both new public symbols are exposed in
  :mod:`panobbgo.self_improve`'s ``__all__``.

  CLI surface on ``scripts/self_improve.py codify-scan``:

  * ``--widen-bounds`` — appends a *Bound-widening candidates*
    section after the existing codify-candidate report.  Off by
    default so existing invocations are byte-identical.  Each
    surfaced pair carries a one-token tag — ``[widens current]`` /
    ``[tightens current — focuses bandit on observed range]`` /
    ``[partial overlap]`` / ``(no rule)`` (when no numeric rule
    targets the slot) — so the operator can prioritise at a glance.
    JSON mode (``--json``) emits each widening candidate on its own
    line tagged ``"_type": "widening_candidate"``; codify
    candidates carry the symmetric ``"_type": "codify_candidate"``
    tag (additive on the existing schema, byte-safe to ignore for
    consumers that don't filter on it).
  * ``--widen-factor FLOAT`` — multiplicative widening factor
    applied to the observed range, default ``1.5`` (matches the
    idea sketch in the *Mutation-bound widening rule* entry under
    *Next iteration ideas*).  Validated by
    :func:`_widen_numeric_bounds` (``> 1.0`` required) so an
    operator passing a degenerate factor gets a clear error
    instead of a silent no-op.

* **Why** — The 2026-06-17 ``codify-scan`` ship surfaces 5
  candidates on the live project ledger today; *4 of the 5* are
  bidirectional pairs (``Nearby.radius`` up and down, ``Sobol.n`` up
  and down — the fifth is the already-codified ``Sobol.scramble =
  False`` that the 2026-06-18 suppression layer hides).  The codify
  scanner reports each direction as a separate candidate the
  operator could ship as a default shift — but the two directions on
  the *same slot* are contradictory: shipping
  ``Nearby.radius=0.135`` (the up median) would invalidate the
  ``Nearby.radius=0.073`` evidence and vice versa.  Before this ship
  the §12.3 daily routine had no in-tool way to distinguish
  "bidirectional pattern — operator should consider a bound update"
  from "directionally consistent pattern — operator should ship a
  default shift", and the planning doc's *Mutation-bound widening
  rule* idea was the only place that documented the correct action.

  The detector closes that gap: the bidirectional pattern becomes a
  first-class report section with a proposed bound and a tag that
  reads naturally for the operator triaging the daily summary.
  Direct effect on §11 V2 success criterion 2 (codify-PR
  throughput): a bidirectional codify-scan candidate that the
  operator would previously discard as ambiguous now has a concrete
  action attached.

  Running against the live project ledger after this ship surfaces
  two widening candidates (``--widen-bounds --widen-factor 1.5``):

  * **``Nearby.radius``** — observed ``[0.073, 0.135]``, current
    ``[0.005, 0.5]``, proposed ``[0.049, 0.203]`` — *tightens
    current*.  The bandit consistently picks values in a window
    5-10× narrower than the catalog admits; concentrating draws
    there frees compute the catalog currently spends in the (0.005,
    0.049) and (0.203, 0.5) dead bands.
  * **``Sobol.n``** — observed ``[8, 24]``, current ``[4, 64]``,
    proposed ``[5, 36]`` — *tightens current*, same shape.  The
    bandit explores half the catalog's integer range; the proposed
    bound is still wider than the observed (5 < 8 and 36 > 24, the
    1.5× headroom in both directions) so the bandit can still
    explore outside the observed range when a future night's
    instance prefers it.

* **Backwards compatibility** — strictly safe.  Pure additions to
  ``panobbgo/self_improve.py`` (one dataclass + one public function
  + two private helpers) and two new CLI flags on the existing
  ``codify-scan`` subcommand.  Existing invocations (without
  ``--widen-bounds``) produce byte-identical output; the JSON-mode
  schema gains a new ``"_type"`` field on every emitted record but
  the field is additive — consumers that don't filter on it see the
  same record bodies as before.  The ``MutationRule``,
  ``MutationCatalog``, ``CodifyCandidate``,
  ``aggregate_codify_candidates``, and
  ``annotate_codified_status`` library APIs are unchanged.

* **Tests** — 38 new tests across three test classes in
  ``tests/test_self_improve.py``:

  * ``TestWidenNumericBounds`` (10 tests): per-rule-kind bound
    arithmetic — log_uniform_perturb multiplicative widening, tiny
    positive floor, integer_add outward rounding, lower-bound
    clipping at one, observed-zero preserved, float_uniform
    symmetric widening, observed-zero preserved, and the
    ``widen_factor > 1.0`` validation (zero / one / negative
    rejected, unsupported rule_kind rejected).
  * ``TestCatalogNumericBounds`` (4 tests): the catalog lookup —
    finds existing rules (``Nearby.radius``, ``Sobol.n``), returns
    None for unknown slots, distinguishes dual-rule slots
    (``NLSHADE_RSP.k_rank``'s ``float_uniform`` and
    ``categorical_choice`` rules), and integer rule bounds return
    as floats so callers can do uniform arithmetic.
  * ``TestDetectWideningCandidates`` (17 tests): pairing semantics
    — empty input, single direction doesn't pair, opposite
    directions on the same slot pair, different slots don't pair,
    different rule kinds don't pair (separate bandit arms),
    structural and categorical candidates are skipped, proposed
    bounds use the configured ``widen_factor``, catalog lookup
    populates ``current_bounds``, unknown slot yields ``None``
    current bounds (treated as wider), ``proposal_is_wider`` and
    ``proposal_is_tighter`` flags set correctly,
    sort order is by strongest evidence, ``n_accepts`` and
    ``distinct_dates`` aggregate across directions
    (date-deduping when both directions share a night),
    ``slot_key`` matches :attr:`CodifyCandidate.slot_key`,
    JSON round-trip through :meth:`to_dict`, and an explicit
    catalog overrides the default.
  * ``TestCodifyScanCLIWidening`` (5 tests): end-to-end CLI smoke
    tests against ``_cmd_codify_scan`` — the flag is off by
    default, ``--widen-bounds`` surfaces the new section,
    no-bidirectional-pattern prints "0 surfaced", JSON mode emits
    typed records (``codify_candidate`` + ``widening_candidate``),
    and ``--widen-factor 3.0`` propagates into the proposed bounds.

  Plus the ``_codify_candidate`` helper factored out at module
  level so the new tests don't have to rebuild JSONL records for
  unit-level pairing tests.

  Test totals: 449 in ``tests/test_self_improve.py`` (411 before +
  38 new); 1645 in ``tests/`` (11 skipped — unrelated IOH worker
  setup).  ``uv run --extra dev ruff format --check .`` /
  ``uv run --extra dev ruff check panobbgo/self_improve.py
  scripts/self_improve.py tests/test_self_improve.py`` /
  ``uv run pyright panobbgo/self_improve.py`` all clean.

* **Impact** — direct effect on the §12.3 daily routine and §11
  V2 success criterion 2.  Before this ship, the four bidirectional
  candidates on the live ledger (``Nearby.radius`` up/down,
  ``Sobol.n`` up/down) accounted for 100% of the actionable
  codify-scan output (the fifth surfacing candidate is the
  already-codified ``Sobol.scramble = False``, hidden by the
  suppression layer).  The operator had to manually recognise the
  bidirectional pattern, look up the current catalog bound, and
  compute the proposed bound by hand — adding cognitive cost that
  the planning doc's "Next iteration ideas" entry already flagged.
  After this ship, the same triage is one ``--widen-bounds`` flag
  away from a concrete bound-update proposal with the per-direction
  evidence pre-pooled and the tag (``[tightens current]`` /
  ``[widens current]`` / ``(no rule)``) describing the proposal
  shape.

  Cumulative effect over the V2 30-night window: every bidirectional
  pattern the loop discovers becomes a candidate codify PR (against
  ``default_catalog``) instead of being silently discarded as
  ambiguous evidence.  Pairs naturally with the queued
  ``--open-pr`` follow-up: the same
  :attr:`WideningCandidate.slot_key` tuple
  ``(class_name, param_name, None)`` the codify-candidate path uses
  is reused here so a future ``--open-pr`` driver can dedup
  uniformly across both candidate kinds.

* **Documentation updated**
  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry; the
    *Mutation-bound widening rule for bidirectional codify
    candidates* idea promoted from *Next iteration ideas* to
    shipped.
  - ``planning/SELF_IMPROVEMENT_LOOP.md``: §9.3 mentions the
    widening detector as the bidirectional-pattern handler
    alongside ``codify-scan``'s default-shift handler.
  - ``doc/source/guide.rst``: quick-nav entry adds a mention of
    the new ``WideningCandidate`` / ``detect_widening_candidates``
    pair and the ``--widen-bounds`` / ``--widen-factor`` CLI flags.
  - ``doc/source/guide_benchmarking.rst``: new "Bidirectional-bound
    widening (``--widen-bounds``)" sub-subsection in the
    "Cross-night codify-scan" subsection documenting the rule
    semantics and the live-ledger evidence.
  - ``AGENTS.md``: self-improvement loop bullet + new bash example.
  - ``TODO.md``: new "Recent Improvements" entry.

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * **``codify-scan --widen-bounds --open-pr``** — extend the queued
    ``--open-pr`` driver to translate each surfaced
    :class:`WideningCandidate` into a concrete edit on
    :func:`~panobbgo.self_improve.default_catalog` (updating the
    rule's ``bounds=(lo, hi)`` tuple) and open a draft codify PR
    against ``panobbgo/self_improve.py``.  The slot identifier
    :attr:`WideningCandidate.slot_key` is the same tuple shape the
    codify-candidate path uses so the dedup pass is uniform across
    both candidate kinds.  Speculative until the basic
    ``--open-pr`` driver lands.
  * **Per-kind widen factor** — log-scale knobs naturally tolerate
    a larger widen factor than linear ones; a categorical
    ``--widen-factor-log`` / ``--widen-factor-linear`` flag pair
    would let the operator tune the rule per kind.  Speculative —
    the current ``1.5`` default is a reasonable compromise.
  * **Auto-tune widen factor from observed spread** — when the
    observed range is narrow (high agreement across nights), a
    larger widen factor lets the bandit explore outside the
    observed window; when the range is wide (high variance), a
    smaller factor focuses on the consensus.  Speculative — the
    fixed factor is a starting point.

### 2026-06-18 — Suppress already-codified candidates in codify-scan (V2 §9.3 follow-up)

* **What** — Closes the *Suppress already-codified candidates* idea
  seeded under *Next iteration ideas* on 2026-06-17.  Two pure
  additions to :mod:`panobbgo.self_improve` plus one CLI flag pair on
  ``scripts/self_improve.py codify-scan`` that cross-check every
  surfaced :class:`CodifyCandidate` against the live seed-spec
  factories and hide candidates whose implied source edit is a no-op:

  * :func:`panobbgo.self_improve.default_codify_registries` —
    returns ``[_make_quick_strategies, _make_loop_strategies]``, the
    two factories the nightly cron exercises.  Standard / full
    registries are intentionally excluded: their seed specs target
    the manual benchmark battery (200 / 500 evals), not the cron,
    and surfacing "already codified" candidates whose codification
    only lives in those registries would mis-direct the operator.
  * :func:`panobbgo.self_improve.annotate_codified_status` — walks a
    sequence of :class:`CodifyCandidate` instances and mutates each
    one in place to set :attr:`CodifyCandidate.already_codified`
    (``bool``) and :attr:`CodifyCandidate.live_codified_values`
    (tuple of the live kwarg values for the slot).  The predicate
    rules per ``rule_kind``:

    - ``categorical_choice``: codified iff any live value's
      ``repr`` equals :attr:`CodifyCandidate.direction` exactly
      (so ``False`` and ``"False"`` do not collide).
    - ``integer_add`` / ``float_uniform`` /
      ``log_uniform_perturb``: codified iff the live value already
      meets the median of :attr:`new_values` in the candidate's
      direction (``"up"`` → ``max(live) >= median(new_values)``;
      ``"down"`` → ``min(live) <= median(new_values)``).  Median
      rather than mean so a single outlier accept doesn't drag the
      threshold; ``max`` / ``min`` over live values because *any*
      seed spec already at the proposed level means the codify edit
      is a no-op on that spec.
    - Structural ops (``op is not None``): not handled.  The
      placeholder helper :func:`_structural_already_codified`
      conservatively returns ``False`` so ``add_/drop_`` candidates
      continue to surface — a follow-up could compare ``add_X``
      against the heuristic-pool membership of the seed factories,
      but the kwarg case is the dominant cause of duplicate
      candidates (it's literally the
      ``Sobol.scramble=False`` shape the §13 2026-06-17 entry's
      "Follow-up ideas" called out).

    A factory that throws is silently skipped — the helper is a
    best-effort scan and a downstream caller shipping a misbehaving
    factory should not break the whole codify-scan run.

  Both new symbols are exposed in
  :mod:`panobbgo.self_improve`'s ``__all__``.
  :class:`CodifyCandidate` gains the two new fields
  (``already_codified: bool = False`` /
  ``live_codified_values: Tuple[Any, ...] = ()``) at the end of the
  dataclass field list so existing constructor invocations are
  byte-identical; :meth:`CodifyCandidate.to_dict` carries both
  fields through to the ``--json`` output.

  CLI surface on ``scripts/self_improve.py codify-scan``:

  * ``--include-already-codified`` — show the suppressed set inline,
    tagged ``[already codified]`` in the slot header and with the
    matching seed kwarg values surfaced under a new
    ``live seed value(s):`` line so the operator can confirm the
    verdict.  Default off so the daily routine sees only actionable
    evidence.
  * ``--no-suppress-codified`` — alias that reads more naturally
    when paired with ``--json`` (which always emits every candidate
    regardless — the consumer filters on the new ``already_codified``
    JSON field itself).
  * Status line gains a ``(of N; M already codified, hidden)``
    suffix when suppression fires, so the operator can see at a
    glance whether the report shrank.

* **Why** — The 2026-06-17 ``codify-scan`` ship surfaces five
  candidates on the live project ledger today; one of them
  (``Sobol.scramble = False``) was codified in
  :func:`~panobbgo.harness._make_quick_strategies` on 2026-05-31 from
  the same evidence stream this scanner now reads.  Continuing to
  surface a candidate that is already shipped is not a bug in the
  scanner — the evidence really is in the archive — but it is a
  signal-to-noise tax on the daily routine: the operator has to
  remember which slots have already been codified to triage the
  scanner's output, and §12.3 step 0's "deduplicate before picking a
  task" lesson (the four duplicate NL-SHADE-RSP PRs #227–#230) makes
  the cost concrete.  The suppression layer turns that operator-side
  memory burden into a structural cross-check: the scanner imports
  the same factories the cron runs and asks "is the change you're
  proposing already live?" before showing the candidate.

  Running against the live project ledger after this ship:

  * 5 candidates clear the default gate (same as before).
  * 1 is flagged ``already_codified`` (``Sobol.scramble = False``).
  * 4 are surfaced — ``Nearby.radius`` direction=up/down (the
    bidirectional pattern the *mutation-bound widening* idea
    addresses), ``Sobol.n`` direction=up/down (same shape) —
    actually-actionable.
  * The status line now reads ``candidates surfaced: 4 (of 5;
    1 already codified, hidden)``.

  Direct effect on §11 V2 success criterion 2 (codify-PR
  throughput): the operator's attention stays on the four actionable
  candidates instead of having to mentally filter the already-shipped
  one.  Pairs naturally with the queued ``--open-pr`` follow-up — the
  same predicate the suppression layer applies here is what
  ``--open-pr`` will use to decide whether to actually open the PR.

* **Backwards compatibility** — strictly safe.  The new fields on
  :class:`CodifyCandidate` carry default values so every existing
  constructor invocation continues to type-check (verified against
  the existing 30+ tests in ``TestAggregateCodifyCandidates`` —
  they construct candidates without the new fields and still pass).
  ``aggregate_codify_candidates`` is unchanged; the suppression
  layer lives in
  :func:`annotate_codified_status` which the CLI calls *after*
  aggregation.  A caller that only uses the library
  (``aggregate_codify_candidates`` directly) sees byte-identical
  output unless it opts in to the annotation pass.

  The two new CLI flags default off (or to the suppress-by-default
  behaviour, depending on which alias the operator prefers); the
  existing test in
  ``TestCodifyScanCLI.test_realistic_two_night_pattern_surfaces_candidate``
  exercises ``Nearby.radius direction=up`` whose candidate's median
  proposal (``0.125``) is above the live value (``0.1``), so it is
  *not* codified and the test continues to expect ``candidates
  surfaced: 1``.  Verified — the test passes unchanged.

* **Tests** — 18 new tests across two test classes in
  ``tests/test_self_improve.py``:

  * ``TestAnnotateCodifiedStatus`` (14 tests) — every rule kind
    (categorical match / mismatch, numeric up codified / not,
    numeric down codified / not, analyzer-bucket kwarg, multiple
    live values, structural placeholder), the empty-live-values
    edge case, the round-trip through :meth:`to_dict`, the default
    constructor field values, the factory-that-throws
    silent-skip behaviour, and a sanity check that
    :func:`default_codify_registries` returns the expected two
    factories.
  * ``TestCodifyScanCLISuppression`` (4 tests) — end-to-end CLI
    smoke tests against the suppression behaviour: the canonical
    ``Sobol.scramble=False`` candidate is suppressed by default;
    ``--include-already-codified`` shows it inline with the
    ``[already codified]`` tag and the ``live seed value(s):``
    line; a non-codified ``Nearby.radius`` candidate still
    surfaces (verifying the suppression check ran and the
    candidate cleared it); ``--json`` mode always emits every
    candidate with the new ``already_codified`` /
    ``live_codified_values`` fields.

  Test totals: 372 in ``tests/test_self_improve.py`` (354 before +
  18 new); 1568 in ``tests/`` (11 skipped — unrelated IOH worker
  setup).  ``uv run --extra dev ruff format --check .`` /
  ``uv run --extra dev ruff check panobbgo/self_improve.py
  scripts/self_improve.py tests/test_self_improve.py`` /
  ``uv run pyright panobbgo`` / 96 sphinx doctests all clean.

* **Impact** — direct effect on the §12.3 daily routine: the
  scanner's report shrinks from 5 candidates to 4 actionable ones
  on the live project ledger.  The signal-to-noise improvement
  scales as more codify PRs land — each merged codify PR adds one
  to the "already codified" set, and every subsequent scan
  collapses that candidate's evidence into the suppressed bucket
  instead of replaying it in the operator's report.  Over the V2
  30-night window the cumulative effect is the difference between
  the operator reading a growing list of stale candidates and a
  steady list of actionable ones.

* **Documentation updated**
  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry; "Suppress
    already-codified candidates" idea promoted from *Next iteration
    ideas* to shipped.
  - ``planning/SELF_IMPROVEMENT_LOOP.md``: no direct edit (the
    candidate-set hygiene work is downstream of §9.3, not on the
    critical V2 path).
  - ``doc/source/guide.rst``: quick-nav entry mentions the new
    suppression layer and the ``--include-already-codified`` flag.
  - ``doc/source/guide_benchmarking.rst``: new sub-paragraph in the
    "Cross-night codify-scan (§9.3 / §9.5 step 4)" subsection
    documenting the suppression rules and the JSON / human-readable
    output behaviours.
  - ``AGENTS.md``: self-improvement loop subsection + new bash
    example.

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * **Structural-op codified check** — extend
    :func:`_structural_already_codified` to compare ``add_X`` /
    ``drop_X`` candidates against the heuristic-pool membership of
    the seed factories.  ``add_LBFGSB`` against a seed pool that
    already contains :class:`LBFGSB` is the symmetric case.
    Lower priority than the kwarg suppression because structural
    candidates are rarer in the live ledger today.
  * **Tolerance / hysteresis on the numeric predicate** — the
    current ``max(live) >= median(new_values)`` rule is exact; a
    small relative tolerance (e.g. 5%) would let the predicate
    catch cases where the live default is *very close* to the
    median proposal without being strictly above / below.
    Speculative — the exact rule already catches the dominant
    ``Sobol.scramble`` shape.

### 2026-06-17 — Cross-night codify-scan CLI (V2 §9.3 / §9.5 step 4)

* **What** — The detection half of V2 §9.3 — a new
  ``scripts/self_improve.py codify-scan`` subcommand plus three public
  library symbols on :mod:`panobbgo.self_improve`:

  * :class:`panobbgo.self_improve.CodifyCandidate` — frozen dataclass
    carrying one directionally-consistent group of accepted mutations:
    class / param / rule_kind / op / direction, per-record evidence
    (deltas, CIs, old / new values, timestamps, strategy names,
    ``confirmed`` flags), pooled stats (``mean_delta``,
    ``min_ci_low``, ``max_ci_high``), and a
    :attr:`slot_key` tuple ``(class_name, param_name, op)`` that the
    follow-up ``--open-pr`` driver will use to dedup against
    ``gh pr list --state open`` per §12.3 step 0.  Exposes
    :meth:`pooled_bootstrap_ci` (percentile bootstrap on the per-record
    deltas) and :meth:`to_dict` for JSON serialisation.
  * :func:`panobbgo.self_improve.aggregate_codify_candidates` — the
    scanner.  Walks every iteration record in the input, drops
    non-iteration / non-accepted / no-op / no-proposal / no-direction
    rows, groups by ``(class_name, param_name, rule_kind, op,
    direction)``, and emits one :class:`CodifyCandidate` per group
    that clears ``min_nights`` distinct accept dates **and**
    (default) ``min(ci_low) > 0`` across contributing records.
    Sorted by ``(n_distinct_nights desc, mean_delta desc, n_accepts
    desc)`` so the strongest and most-replicated evidence surfaces
    first.  ``confirmed_only=True`` opt-in restricts the input to
    records carrying the V2 §6.4 ``confirmed`` field (post PR #255).
  * :func:`panobbgo.self_improve.load_ledgers_for_codify_scan` — io
    helper that mirrors :meth:`AdaptiveMutationSampler.prime_from_archives`
    semantics: scans the archive directory for files matching
    ``self_improve_ledger_*.jsonl`` in chronological (lexicographic)
    order and prepends them before the live ledger.  Default archive
    dir is ``<ledger parent>/done`` so a typical invocation against
    ``planning/self_improve_ledger.jsonl`` automatically picks up
    ``planning/done/``.  Missing files / directories silently no-op
    so the helper is safe to call on a fresh checkout.

  Plus the private helpers :func:`panobbgo.self_improve._direction_key`
  (per-proposal direction extraction — ``"up"`` / ``"down"`` for
  numeric, ``repr(new_value)`` for categorical, op name for
  structural) and :func:`panobbgo.self_improve._percentile_bootstrap_ci`
  (the pooled-CI primitive — matches the simple non-paired bootstrap
  used by :func:`aggregate_holdout_drift` for parity).

  CLI surface on the new ``codify-scan`` subparser:

  * ``--ledger PATH`` (default ``planning/self_improve_ledger.jsonl``).
  * ``--archive-dir DIR`` / ``--no-include-archives``.
  * ``--min-nights N`` (default ``2``, matching §9.3 ``k ≥ 2``).
  * ``--no-require-positive-min-ci`` to surface weak evidence too.
  * ``--confirmed-only``.
  * ``--pooled-ci-n-boot`` / ``--pooled-ci-confidence`` /
    ``--pooled-ci-seed`` for reproducible CI computation.
  * ``--json`` emits one ``CodifyCandidate.to_dict()`` JSON per line.
  * ``--top N`` truncates the report to the strongest N candidates.

* **Why** — V2 §11 success criterion 2 (*"≥ 3 codify PRs opened from
  ledger evidence; ≥ 2 merged"* over the first 30 nights) is the
  measurable bar for whether the V2 loop *durably improves anything*
  — §12.2 makes the constraint explicit: "the cron never commits
  changes under ``panobbgo/``; durable improvement happens only
  through codification".  Before this ship, the §12.3 daily routine
  had to grep the ledger by hand to find directionally consistent
  accept patterns (the four-night Sobol.scramble pattern that the
  2026-05-31 codify ship caught took manual ledger inspection and a
  manual ``gh pr create``).  ``codify-scan`` makes that inspection
  reproducible: the same scanner, run nightly, surfaces the same
  candidates whether the operator is reaching for a PR or a CI
  status check.

  Running against the current project ledger on the day of ship
  surfaces five candidates that clear the default gate (k ≥ 2 nights,
  every record's ``ci_low > 0``):

  * **``Nearby.radius`` direction=up**: 7 accepts on 6 nights,
    mean Δ=+0.0566, pooled CI95%=[+0.042, +0.072].  Strongest
    candidate by replication count — the bandit consistently raises
    Nearby's radius above the constructor default ``0.1``.
  * **``Sobol.scramble`` direction=False**: 4 accepts on 4 nights,
    mean Δ=+0.0456, pooled CI95%=[+0.027, +0.066].  Already codified
    in the seed factory 2026-05-31; the scanner picks up the
    pre-codification evidence stream as a sanity check that the
    detection logic mirrors what the manual ship caught.
  * **``Sobol.n`` direction=down**: 4 accepts on 4 nights, all
    ``16 -> {8, 12, 12, 12}``.  Strong evidence for lowering the
    seed default below ``16``.
  * **``Nearby.radius`` direction=down**: 4 accepts on 3 nights —
    the opposite-direction signal pairs with the "up" winner.  Worth
    investigating whether the right move is a wider mutation bound
    rather than a default shift.
  * **``Sobol.n`` direction=up**: 4 accepts on 3 nights, ``16 ->
    {20, 24, 20, 24}``.  Pairs with the "down" winner in the same
    bidirectional way.

  The bidirectional candidates are valuable signal — even when the
  detection rule doesn't unambiguously vote for a single codify
  direction, the operator can decide to widen the catalog bound or
  introduce a categorical regime instead of a default shift.

* **Backwards compatibility** — strictly safe.  Two pure additions to
  ``panobbgo/self_improve.py`` (the three public symbols plus two
  private helpers) and one new subparser on ``scripts/self_improve.py``
  — no edits to existing API.  The new subcommand is opt-in:
  ``run`` / ``summary`` invocations and the
  :class:`SelfImprover` integration path are byte-identical.  All
  three new library symbols are also exposed in
  :mod:`panobbgo.self_improve`'s ``__all__`` so downstream code can
  import them directly.

* **Tests** — 46 new tests in ``tests/test_self_improve.py``
  organised into five test classes:

  * ``TestDirectionKey`` (9 tests): every ``rule_kind`` in
    :func:`default_catalog`, every structural op, plus the
    ``None``-direction cases (equal numeric values, non-numeric old
    value, missing old value).
  * ``TestPercentileBootstrapCI`` (4 tests): empty / single-sample
    degenerate cases, multi-sample CI brackets the mean, seed
    reproducibility.
  * ``TestAggregateCodifyCandidates`` (16 tests): the gates
    (min_nights / require_positive_min_ci / confirmed_only), the
    grouping correctness (same-night dedup, opposite directions
    separate buckets, categorical bucket key via ``repr``,
    structural op as direction), the filtering (no-op / non-accepted /
    skip / non-iteration records dropped), the sort order
    (strongest candidate first), and the
    :meth:`CodifyCandidate.to_dict` round-trip through JSON.
  * ``TestLoadLedgersForCodifyScan`` (7 tests): missing live ledger,
    live-only mode, default archive dir as ``<ledger parent>/done``,
    explicit archive dir override, missing archive dir silent
    no-op, non-matching files ignored, chronological order.
  * ``TestCodifyScanCLI`` (6 tests): end-to-end CLI smoke tests
    using the fabricated-record helper — empty ledger note, the
    realistic two-night pattern, the JSON output mode, the
    ``--top N`` truncation, the ``--min-nights`` argument
    validation, ``--confirmed-only`` filters legacy records to
    zero, plus a sanity check against the *real project ledger* that
    confirms the CLI handles the live planning/ files end-to-end.

  All 1550 prior project tests continue to pass (354 self-improve
  tests, 1550 total); ruff format / check / pyright / 96 sphinx
  doctests / flake8 E9/F63/F7/F82 all clean.

* **Impact** — direct effect on §11 V2 success criterion 2
  (codify-PR throughput).  Before this ship, "scan the ledger for
  codify candidates" was a manual ledger-grep that produced one
  ship in five weeks (the 2026-05-31 ``Sobol.scramble = False``
  codification) — and depended on operator memory of which patterns
  to look for.  After this ship, the same scan is one CLI invocation
  that reproducibly surfaces the same candidates every night with
  pooled stats, per-record evidence, and a stable slot identifier
  for PR dedup.  Pairs naturally with the two open PRs (#255
  ``--confirm-accepts`` for V2 §6.4 and #256
  ``--prime-include-archives`` for §9.5 step 4): once #255 merges
  the ``confirmed`` field starts populating on ledger records and
  ``--confirmed-only`` becomes the recommended default; once #256
  merges archive evidence is no longer thrown away across nightly
  rotations.

* **Documentation updated**
  - ``planning/SELF_IMPROVEMENT_LOOP.md``: §9.3 (Stage 3) — detection
    half marked shipped, ``--open-pr`` half queued; §9.5 step 4 —
    detection sub-item promoted to shipped.
  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry; the
    *Open the codify PR from the detected candidates* idea added to
    *Next iteration ideas* to track the queued ``--open-pr`` follow-up.
  - ``doc/source/guide.rst``: quick-nav entry now mentions the
    §9.3 ``codify-scan`` ship with the public library symbols.
  - ``doc/source/guide_benchmarking.rst``: new "Cross-night
    codify-scan (§9.3 / §9.5 step 4)" subsection in the
    self-improvement loop section.
  - ``AGENTS.md``: self-improvement loop subsection +
    three new bash examples (default / JSON / ``--confirmed-only``).

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * **``codify-scan --open-pr``** (the ship's queued follow-up) —
    translate each surfaced candidate into a concrete source edit +
    PR opened against the seed-spec factory (or the heuristic
    constructor default).  Needs a small "where does this kwarg get
    set" lookup (e.g. ``_make_loop_strategies`` vs heuristic
    ``__init__``), a code-edit primitive that respects the existing
    formatter, and the ``gh`` CLI integration to open the draft PR
    with the ledger evidence in the body.  Dedup via
    :attr:`CodifyCandidate.slot_key` against ``gh pr list --state
    open``.
  * **Mutation-bound widening rule** — when ``codify-scan`` surfaces a
    *bidirectional* candidate (e.g. ``Nearby.radius`` up *and* down),
    the right action is rarely to ship a new default; it's to widen
    the catalog ``MutationRule`` bound so the bandit can explore a
    larger range.  A second CLI subcommand (or a ``--widen-bounds``
    flag on ``codify-scan --open-pr``) could detect this shape and
    propose the bound update instead of a default change.
  * ~**Suppress already-codified candidates**~ — **shipped
    2026-06-18** as
    :func:`panobbgo.self_improve.annotate_codified_status` plus the
    ``--include-already-codified`` CLI flag.  The motivating
    ``Sobol.scramble=False`` example is now hidden by default; on
    the live project ledger the report shrinks from 5 to 4
    candidates.  See the dated entry above.

### 2026-06-16 — Summary trend block + bandit posteriors + inactivity telemetry (V2 §12.4)

* **What** — Three additive sub-blocks rendered by the
  ``scripts/self_improve.py summary`` CLI after the existing
  per-record sections, plus three new CLI flags on the ``summary``
  subparser, plus four new helpers in ``scripts/self_improve.py``:

  * ``_group_runs(iter_records)`` — partitions iteration records into
    per-run buckets by detecting ``iteration <= prev_iteration``
    boundaries.  The append-only nightly ledger concatenates the
    iteration records of every nightly run end-to-end, and each
    :meth:`SelfImprover.run` restarts the counter at ``0`` — so the
    boundary detector is the natural inverse of the writer.
  * ``_print_trend_block(iter_records)`` — renders one row per loop
    run with date / base_seed / mode / iters / decided / accepts /
    no-op / best Δ / seed score columns, oldest first.  The seed
    score is sourced from ``baseline_score`` of the first record of
    each run so it tracks a real per-night signal, not a recomputed
    average over the run's mixed baselines.
  * ``_replay_bandit_posteriors(iter_records)`` — reconstructs per-rule
    bandit stats by replaying iteration records through the same
    :func:`panobbgo.self_improve._proposal_rule_key` collapse used by
    :meth:`AdaptiveMutationSampler.prime_from_ledger` (default
    ``per_class_structural=False``), so the summary's posterior view
    matches what a freshly-primed nightly bandit would carry into the
    next run.  No-op iterations and skip / guard / hold-out records
    are filtered out exactly as the live bandit filters them per
    §12.4.  Returns a dict keyed on ``(class_name, param_name,
    rule_kind)`` (or the structural collapse ``("*", op,
    "structural")``) with cumulative ``n_attempts`` / ``n_accepts`` /
    ``reward_sum`` and derived ``mean_reward`` / ``accept_rate``.
    Legacy records (no ``bandit_reward``) fall back to the binary
    ``1.0`` per accept / ``0.0`` per reject — matching
    :meth:`prime_from_ledger` byte-for-byte.
  * ``_print_bandit_block(iter_records, top_n, bottom_n, min_attempts)``
    — ranks rules by graded ``mean_reward`` descending (tie-break by
    ``n_attempts`` so dense evidence beats sparse evidence at the same
    mean), filters out rules below the ``min_attempts`` threshold so
    one-shot rules cannot dominate the leaderboard, and renders a
    top-N / bottom-N table.  The bottom slice is reversed so the worst
    rule prints last — easier for an operator to scan the "should I
    deprioritize this?" block from top to bottom.  When no rules clear
    the threshold the block prints a single explanatory line instead
    of an empty table.  On graded-reward ledgers the ranking carries
    the full §7.4 signal (barely-confirmed accepts at ``~0.5``, honest
    near-miss rejects at ``~0.5``, clearly-harmful rejects at ``~0``);
    on legacy binary-reward ledgers ``mean_reward`` collapses to
    ``accept_rate`` so pre-2026-06-13 evidence is rendered without
    distortion.
  * ``_print_inactivity_block(iter_records)`` — infers the configured
    ``eps_accept`` base from the maximum observed
    ``effective_eps_accept`` (relaxation only *decreases* the
    threshold — it is re-tightened back to the base on every accept),
    then surfaces the longest accept drought (max
    ``iters_since_accept``), the relaxed-accept count
    (``effective_eps_accept < eps_base``), and the mean decay factor
    at the moment of accept.  Silently no-ops on legacy ledgers
    (pre-2026-05-30) whose iteration records carry neither field, so
    the existing summary contract on those ledgers is preserved.
  * ``--top-n`` (default ``10``) / ``--bottom-n`` (default ``5``) /
    ``--min-attempts`` (default ``3``) flags on the ``summary``
    subparser so an operator can tune the bandit-posterior view
    without code changes.  The defaults match the §12.4 spec.

* **Why** — Closes the third open bullet of
  ``planning/SELF_IMPROVEMENT_LOOP.md`` §12.4 (the "Summary trend
  block") and the *Inactivity-relax telemetry in the summary view*
  backlog idea in one ship.  The §12.3 daily routine explicitly reads
  ``planning/self_improve_summary.txt`` "at-a-glance" — but the
  pre-ship summary was an ever-growing wall of per-record lines
  (200 iterations × 10 nights × N hold-out records).  An operator
  reviewing the file had no way to answer the questions the routine
  exists to surface:

  1. **Is the loop accepting anything tonight?**  The aggregate
     accept rate over all 10 nights masks per-night dispersion — one
     productive night next to nine vacuous ones reports the same
     ``2.7%`` as a steady drip of one accept per night.  The Trend
     block surfaces per-night accept counts so an operator can see at
     a glance whether the loop is producing reproducible signal or
     getting lucky on a single night.
  2. **Which arms are paying off?**  Pre-ship there was no way to
     ask "what is the bandit's posterior on each rule" without
     parsing the 200-record ledger by hand.  The Bandit-posteriors
     block runs the same replay
     :meth:`AdaptiveMutationSampler.prime_from_ledger` runs, then
     ranks by graded ``mean_reward`` so the operator can codify
     winners (per §12.3 step 2) and deprioritize losers without
     reaching for an editor.
  3. **Is the inactivity relax knob doing anything?**  The 2026-05-30
     ship persisted ``effective_eps_accept`` / ``iters_since_accept``
     on every record but the summary never surfaced them — the
     knob's effect was opaque without grepping the ledger.  The
     Inactivity block now answers "how long was the longest drought"
     and "did any accept fire on a relaxed threshold" in two lines.

  Pairs naturally with the two open PRs (#255 ``--confirm-accepts``
  for V2 §6.4, #256 ``--prime-include-archives`` for V2 §9.5 step 4):
  both add new fields and record types the trend / posterior blocks
  will surface for free once merged.  In particular, the
  Bandit-posteriors block will pick up confirmed-accept records as
  graded ``r ≥ 0.5`` evidence and the trend block will pick up the
  ``LoopConfirmRecord`` count (a follow-up after #255 merges adds a
  ``confirm`` column).

* **Backwards compatibility** — strictly safe.  Three additive
  sub-blocks rendered *after* the existing per-record sections so the
  existing summary contract is preserved byte-for-byte on the
  pre-trend lines.  All three blocks silently no-op on empty input;
  the Inactivity block additionally no-ops on legacy ledgers
  (pre-2026-05-30) that carry neither ``effective_eps_accept`` nor
  ``iters_since_accept``.  The Bandit-posteriors block prints a
  friendly note ("no rules with >= N informative attempts") rather
  than an empty table when the threshold filters out every rule.
  The three new CLI flags carry default values matching the §12.4
  spec so existing invocations (``uv run python
  scripts/self_improve.py summary``) produce a strict superset of the
  pre-ship output without any flag changes.

* **Tests** — 20 new tests in
  ``tests/test_self_improve.py::TestSummaryTrendBlock``:

  * **Run grouping** (4 tests): empty input → empty list; single run
    in one bucket; iteration-reset boundary splits runs; two
    consecutive ``iteration=0`` records correctly split into two
    buckets.
  * **Trend block** (3 tests): per-run row renders correct counts
    (iters / decided / accepts / no-op / best Δ / seed score); runs
    are rendered oldest-first so the operator scans top-to-bottom;
    silent on empty input.
  * **Bandit replay** (4 tests): no-op / skip / guard / hold-out
    records are filtered out; graded ``bandit_reward`` propagates
    correctly into ``mean_reward`` (and stays distinct from
    ``accept_rate``); legacy records (no ``bandit_reward``) fall
    back to the binary path matching
    :meth:`prime_from_ledger`; structural ops (``add_heuristic`` for
    different classes) collapse onto the single
    ``("*", "add_heuristic", "structural")`` arm by default.
  * **Bandit block rendering** (4 tests): orders by ``mean_reward``
    descending so a high-reward rule appears above a low-reward one;
    filters by ``min_attempts`` so sparse rules don't enter the
    leaderboard; prints a friendly note when no rules clear the
    threshold; silent on empty input.
  * **Inactivity block** (4 tests): renders ``eps_accept_base`` /
    ``longest_drought`` / ``relaxed_accepts`` / ``mean_decay_at_accept``
    correctly; silent on legacy records (no relax fields); silent on
    empty input; hides the ``mean_decay_at_accept`` clause when no
    accept was relaxed.
  * **End-to-end CLI smoke test** (1 test): two-run synthetic ledger
    exercises ``_cmd_summary`` end-to-end and confirms all three new
    sub-blocks appear and the per-run grouping is correct.

  All 1504 prior project tests continue to pass (308 self-improve
  tests, 1504 total); ruff format / check / pyright / 96 sphinx
  doctests / flake8 E9/F63/F7/F82 all clean.

* **Impact** — direct effect on §12.3 ("Daily routine") and the V2
  §11 success criterion 4 ("Honesty: …every codify PR body carries
  reproducible evidence").  An operator reading
  ``planning/self_improve_summary.txt`` (the daily routine's primary
  artifact) can now answer the three §12.3 questions ("is the loop
  accepting?", "which arms pay off?", "is relax doing anything?") in
  one screen of text — vs. a ledger-grep before the ship.  The
  Bandit-posteriors block is the structural ingredient the *codify
  PR* workflow (V2 §9.3) needs to identify candidate rules without
  re-deriving the bandit state by hand each night.

* **Documentation updated**
  - ``planning/SELF_IMPROVEMENT_LOOP.md``: §12.4 third bullet
    ("Summary trend block") promoted from *Open* → *shipped* with a
    pointer to this entry.
  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry; the
    *Inactivity-relax telemetry in the summary view* backlog idea
    collapsed to a one-paragraph shipped pointer.
  - ``doc/source/guide.rst``: quick-nav entry mentions the §12.4
    summary trend block + bandit posteriors + inactivity telemetry
    ship.
  - ``doc/source/guide_benchmarking.rst``: new "Summary trend block
    (§12.4)" subsection in the self-improvement loop section.
  - ``AGENTS.md``: self-improvement loop subsection references the
    three new sub-blocks and the three new CLI flags.

* **PR** — see this PR.  Pairs naturally with the still-open
  ``--confirm-accepts`` (PR #255, V2 §6.4) and
  ``--prime-include-archives`` (PR #256, V2 §9.5 step 4) work: the
  trend / posterior blocks rendered here will pick up the new record
  types and fields for free once those merge.  A follow-up ticket
  ("Confirm column in the trend block") seeds the natural integration.

* **Follow-ups** — speculative, none gated on this ship:

  * Once PR #255 (``--confirm-accepts``) merges, extend the trend
    block with a per-run ``confirmed`` count so the operator can see
    the §6.4 confirmation gate's verdict at a glance.  Same shape:
    one column, one ``sum(1 for r in run if r.get("confirmed"))``.
  * Once PR #256 (``--prime-include-archives``) merges, extend
    ``_replay_bandit_posteriors`` to walk archives in
    ``planning/done/`` so the Bandit-posteriors block reflects the
    same evidence the live bandit accumulates — currently it only
    sees the live ledger.
  * A ``--since`` / ``--last-n-runs`` filter on the summary CLI
    would let an operator narrow the trend / posterior blocks to the
    most recent K nights when the ledger spans many months.  Local
    to the summary subparser; speculative until the ledger has
    accumulated enough nights to make scroll-fatigue real.
### 2026-06-15 — Archive-aware bandit priming (V2 §2.6 / §9.5 step 4)

* **What** — Four coordinated additions in
  :mod:`panobbgo.self_improve` plus a CLI flag pair and a
  :class:`LoopConfig` knob:

  * :meth:`AdaptiveMutationSampler._consume_record` — a freshly
    extracted private helper that applies one ledger record to the
    bandit's posterior (``n_attempts += 1`` and
    ``reward_sum += r`` on iteration records with a non-null
    proposal; filters out ``record_type != "iteration"`` records,
    null-proposal skips, and ``no_op=True`` records identically to
    the previous in-place body of :meth:`prime_from_ledger`).
    Returns ``True`` if the record contributed an update.  Shared
    by :meth:`prime_from_ledger` and the new
    :meth:`prime_from_archives` so the priming semantics are
    byte-identical regardless of which file the record came from.
  * :meth:`AdaptiveMutationSampler.prime_from_archives` — new
    public method that scans a directory for files matching the
    rotation glob ``self_improve_ledger_*.jsonl`` and replays each
    in chronological (lexicographic) order via
    :meth:`_consume_record`.  Returns the total number of records
    consumed across all archives.  Defensive: a non-existent
    directory, an empty directory, a directory containing only
    non-matching files, or a path that points to a regular file
    instead of a directory each return ``0`` and leave the
    posterior untouched (same shape as
    :meth:`prime_from_ledger`'s "missing ledger ⇒ 0" contract).
  * :class:`LoopConfig` gains two opt-in fields:
    ``adaptive_prime_include_archives: bool = False`` and
    ``adaptive_prime_archive_dir: Optional[str] = None``.  When the
    first is ``True`` (and :attr:`adaptive_prime_from_ledger` is
    also ``True``, the existing gate), the SelfImprover's
    constructor calls :meth:`prime_from_archives` on the configured
    directory immediately before :meth:`prime_from_ledger` on the
    live ledger.  The directory defaults to
    ``<dirname(ledger_path)>/done`` — matching the rotation
    convention documented in §12.1 — so the flag is one-flag-only
    for the standard layout.  An explicit override is available for
    setups that keep archives outside the ledger's parent.
  * ``scripts/self_improve.py``: ``--prime-include-archives``
    (boolean) plus ``--prime-archive-dir`` (string override).  The
    one-flag invocation is the recommended path; the override
    exists for the rare case where archives are co-located with
    something else.

* **Why** — closes the *second half* of the §2.6 V2 diagnosis
  ("Bandit starved: ... priming reads only the current ledger —
  archives in ``planning/done/`` are invisible") and the
  ``--prime-include-archives`` sub-item of V2 §9.5 step 4 in
  ``planning/SELF_IMPROVEMENT_LOOP.md``.  The first half of §2.6
  was addressed 2026-06-13 by the graded reward shipping (`§7.4`),
  which converts every informative iteration into ``r ∈ [0, 1]``
  evidence so a single night can lift the posterior meaningfully
  even at the ~2.5% accept rate.  This ship closes the second
  half: the nightly ledger is rotated to ``planning/done/`` after
  every ~2000 records (§12.1), so a long-running unattended cron
  with archive-priming disabled effectively *forgets* every
  pre-rotation observation.  Concretely, the loop has had one
  archive on disk
  (``planning/done/self_improve_ledger_2026-05-31.jsonl``) since
  2026-06-09 that the bandit could not see; every subsequent
  ``--adaptive-prime-from-ledger`` invocation primed from the
  shorter post-rotation ledger and threw the older evidence away.
  With this flag enabled in the nightly workflow, the bandit
  posterior compounds across rotation boundaries — the
  prerequisite for the V2 §11 success criterion 2 ("≥ 3 codify PRs
  opened, ≥ 2 merged") at the realistic 20-40-iterations-per-night
  pace, since rotation will happen long before 3 codify PRs are
  shipped.

* **Why a separate method instead of folding archive scanning into
  ``prime_from_ledger``** — three reasons.  (1) The existing
  ``prime_from_ledger(path: str)`` API is a one-file contract used
  by tests / direct callers / the ``--adaptive-prime-from-ledger``
  flag; adding a side-effect (silently scanning a sibling
  directory) would surprise existing call sites.  (2) The archive
  scan needs its own opt-in (a fresh-night cron should *not*
  start importing yesterday's posterior the first time someone
  runs it manually).  (3) Tests for archive replay are cleaner
  when the file path of the live ledger and the directory of
  archives are passed separately, mirroring how the production
  call site composes them.  The two methods share
  :meth:`_consume_record` so the per-record semantics — graded
  reward, no-op skip, guard / skip filter — cannot drift between
  paths.

* **File discovery contract** — the scan uses
  :func:`pathlib.Path.glob` with the pattern
  ``self_improve_ledger_*.jsonl``.  This matches the rotation
  convention shipped 2026-06-09 (the rotated archive is named
  ``self_improve_ledger_YYYY-MM-DD.jsonl``).  Files that do not
  match the glob — ``planning/done/self_improve_summary_*.txt``,
  the existing ``planning/done/LOGGING_IMPROVEMENT_PLAN.md`` — are
  silently skipped, so the directory can host other artifacts
  without confusing the scan.  Lexicographic sort on the glob
  yields chronological order because the convention uses
  zero-padded ISO dates (``2026-05-31`` sorts before
  ``2026-06-01``).  Order does not affect the *value* of the
  posterior (the per-arm reward sums commute) but matters for the
  bandit-rule-key resolution: if the structural per-class flag
  changes between rotations, the rule key changes too — the
  oldest-first replay means the modern flag's view of the past is
  the one that survives.

* **Backwards compatibility** — strictly safe.  Three layers of
  defaults keep existing call sites byte-identical:

  * ``adaptive_prime_include_archives`` defaults to ``False``, so
    every existing CLI invocation, every direct
    :class:`LoopConfig` construction, and every direct
    :meth:`prime_from_ledger` call behave identically to the
    pre-ship code.
  * The :meth:`prime_from_ledger` body is now a one-line wrapper
    over :meth:`_consume_record`, but the per-record processing —
    rule-key derivation, no-op skip, graded-reward extraction,
    legacy-binary fallback — is the same code paths as before,
    just lifted into a shared helper.  Round-trip tests on a
    fixed ledger reproduce the old ``(n_attempts, n_accepts,
    reward_sum)`` triple exactly.
  * The new method does nothing when the configured directory
    is missing, empty, or contains no matching files — so the
    flag is safe to enable on first-night runs (no archive yet)
    and on developer machines (no rotation has fired).

* **Tests** — 14 new tests across two new test classes plus the
  existing ``TestSelfImproverAdaptive`` extension:

  * :class:`TestPrimeFromArchives` (10 tests):

    * ``test_missing_directory_is_no_op`` — a non-existent path
      returns 0; posterior untouched.
    * ``test_empty_directory_is_no_op`` — directory exists but
      contains no matching files.
    * ``test_directory_with_non_matching_files_is_no_op`` —
      sibling artifacts (``summary.txt``,
      ``other_ledger.jsonl``) are skipped.
    * ``test_single_archive_replayed`` — one archive with one
      accept + one reject contributes ``(2, 1)``.
    * ``test_multiple_archives_replayed_in_chronological_order``
      — two archives sum to ``(5, 3)`` with chronological
      filename ordering.
    * ``test_archives_filter_no_op_records`` — ``no_op: True``
      records in an archive are skipped, matching the live
      ledger semantics shipped 2026-06-12.
    * ``test_archives_filter_guard_and_skip_records`` —
      ``record_type="guard"`` and null-proposal records are
      ignored.
    * ``test_archives_propagate_graded_bandit_reward`` — a
      ``bandit_reward: 0.75`` record in an archive lifts
      ``reward_sum`` by exactly 0.75 (matching
      :meth:`prime_from_ledger` graded-path semantics shipped
      2026-06-13).
    * ``test_archives_combined_with_live_ledger`` —
      :meth:`prime_from_archives` followed by
      :meth:`prime_from_ledger` accumulates correctly into a
      single posterior.
    * ``test_archive_path_is_a_file_returns_zero`` — path-is-a-
      file fallback returns 0 instead of erroring.

  * :class:`TestSelfImproverAdaptive` (4 new tests):

    * ``test_adaptive_prime_include_archives_default_dir`` —
      end-to-end through the SelfImprover constructor: live +
      archive contributions accumulate.
    * ``test_adaptive_prime_include_archives_explicit_dir`` —
      ``adaptive_prime_archive_dir`` override is respected.
    * ``test_adaptive_prime_include_archives_off_by_default`` —
      flag default ``False`` ignores archives even when
      present in the default location.
    * ``test_adaptive_prime_include_archives_requires_prime_from_ledger``
      — flag is inert without ``adaptive_prime_from_ledger=True``
      (matches the existing gate on
      :attr:`SelfImprover.sampler`).

  All 302 self-improvement tests pass; ruff format / check and
  pyright continue to be green.  An end-to-end CLI smoke test
  exercises ``scripts/self_improve.py run --iterations 0
  --adaptive --adaptive-prime-from-ledger --prime-include-archives``
  on a fabricated archive containing one graded-accept and one
  graded-reject and confirms the printed bandit stats reflect both
  records.

* **Impact** — direct effect on the V2 §11 success criterion 2
  ("≥ 3 codify PRs opened, ≥ 2 merged" over the first 30 nights).
  At the current 20-iter-per-night quick-mode budget, ~2000 records
  ≈ 100 nights, so without archive priming the bandit posterior is
  bounded above by ~100 nights' worth of evidence and any older
  observations are lost.  With archive priming on, the bandit's
  effective experience window grows linearly with retained
  archives — exactly what §11 criterion 2 needs to identify the
  small subset of mutation rules with persistent directional
  signal across many nights.  Pairs naturally with the upcoming
  ``codify-scan --open-pr`` / ``--prime-include-archives``
  combined ship (V2 §9.5 step 4): codify-scan already scans
  ``planning/done/`` for cross-night evidence (per §9.3 / §12.3
  daily routine); now the *bandit* — the upstream proposal source
  — does too, so the loop's proposal and selection paths share the
  same long-memory view of the catalog.

* **Follow-ups** — speculative, none gated on this ship:

  * Once the nightly workflow flips to ``--prime-include-archives``
    (V2 §9.5 step 5), expose ``adaptive_prime_archive_dir`` as a
    workflow input so the manual ``workflow_dispatch`` path can
    target a specific archive subset for A/B comparison ("did
    the new graded reward shape help the bandit learn from
    archives faster than the binary path?").
  * A summary trend block (§12.4 third bullet) that surfaces the
    contribution of archive replay separately from the live
    ledger — ``archive_n_attempts: N`` / ``live_n_attempts: N``
    on each rule line — would let an operator see at a glance
    whether the bandit's posterior is *current* or
    *archive-dominated*.  Speculative — the per-arm
    :attr:`MutationRuleStats` does not currently carry source
    metadata; adding a ``(archive, live)`` split would be a
    forward-compatible field addition.
  * The current per-arm key derivation in
    :meth:`_consume_record` uses ``self.per_class_structural`` —
    the *current* run's setting.  If a future ship adds a
    third rule-key shape, the archive-replay path must continue
    to handle pre-ship records gracefully.  The
    ``test_archives_filter_no_op_records`` test demonstrates the
    pattern: legacy records (no ``no_op`` key) classify as
    ``False`` via ``.get("no_op")``, preserving the historical
    semantics.

* **Documentation updated**
  - ``planning/SELF_IMPROVEMENT_LOOP.md``: §2.6 annotated with the
    2026-06-15 update; §9.5 step 4 marks the
    ``--prime-include-archives`` sub-item as shipped.
  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry.
  - ``doc/source/guide_benchmarking.rst``: self-improvement
    section gains a "Crossing nightly boundaries" subsection
    documenting the new flag pair.
  - ``AGENTS.md``: self-improvement loop subsection references
    the new flag.
### 2026-06-14 — Same-night confirmation gate (V2 §6.4)

* **What** — Six coordinated additions in
  :mod:`panobbgo.self_improve` plus the matching CLI flags and an
  expansion of the ``run`` / ``summary`` views:

  * :class:`LoopConfig` gains two fields — ``confirm_accepts: bool =
    False`` (opt-in) and ``confirm_iteration_offset: int = 500_000``
    (planning-doc default, sitting between the regular iteration
    stream ``0..N`` and the guard's ``1_000_000`` so the three streams
    never collide at realistic iteration counts).  The new validator
    rejects ``confirm_iteration_offset <= 0`` and rejects collision
    with ``guard_iteration_offset`` *when ``confirm_accepts`` is True*
    (the dead-code path leaves legacy configs valid).
  * A new helper :func:`_pool_harness_results` concatenates
    per-(problem, strategy) runs across two or more
    :class:`HarnessResult` instances, recomputes per-pair metrics via
    the existing
    :meth:`~panobbgo.harness.ProblemStrategyResult.compute_metrics`,
    and produces a pooled :class:`HarnessResult` whose composite
    score is the mean of the pooled per-pair scores — interchangeable
    with a fresh live harness measurement everywhere the loop already
    consumes one.  The single-input case is the identity (no
    recomputation hazard).
  * A new :class:`LoopConfirmRecord` dataclass carries the screen +
    confirm scores, the pooled CI metadata, the fresh
    ``randomize_iteration``, and the optional hold-out base_seed
    leg.  ``record_type="confirm_reject"`` distinguishes it from
    iteration / guard / hold-out records on the JSONL wire.
    Successful confirmations leave the iteration record carrying
    ``accepted=True`` / ``confirmed=True`` and need no companion
    record; failed confirmations additionally append this record so
    the failure is auditable.
  * :class:`LoopIterationRecord` gains a ``confirmed: Optional[bool]
    = None`` field, serialised via :meth:`to_dict`.  ``None`` on
    skip / no-op iterations, on iterations from runs with
    ``confirm_accepts=False`` (the default), and on legacy ledger
    records written before this ship.  ``True`` on promotion, ``False``
    when the gate overturned a screening accept.  Lets codify-scan
    distinguish "confirmed accept" (durable signal) from "screening
    accept overturned by the gate" (noise spike) without re-deriving
    the verdict from per-record fields.
  * :meth:`SelfImprover._run_internal` grows a confirmation step:
    after a screening accept (``decision.accept and not no_op``), when
    ``self.config.confirm_accepts`` is True, the new helper
    :meth:`_run_confirmation` re-measures baseline + candidate on
    ``iteration + confirm_iteration_offset``, optionally re-measures
    on the *first* configured hold-out base_seed at the same fresh
    iteration_id, pools all measurements via
    :func:`_pool_harness_results`, and re-runs
    :func:`~panobbgo.harness.statistical_accept` on the pooled
    sample.  Promotion happens only when the pooled bootstrap CI
    still clears the same gate (``Δ > eps_accept``, ``ci_low > 0``,
    no catastrophic per-pair regression).  The screening reasons are
    appended with either a "confirmed" or "confirm_reject" marker so
    a JSONL reader sees the gate's decision in the iteration record's
    reasons list.
  * The bandit reward path consumes the *post-confirmation* pooled
    decision: when the gate overturns a screening accept, the graded
    reward formula sees the pooled ``Δ`` / ``ci_low`` rather than the
    screening ones.  An arm that consistently produces noise-spike
    accepts now collects the reject-regime reward
    (``clip(0.5 + pooled_Δ/(4·eps), 0, 0.5)`` — between ``0`` and
    ``0.5``) rather than the full-accept reward
    (``0.5 + clip(ci_low/(4·eps), 0, 0.5)`` — between ``0.5`` and
    ``1.0``) it would have collected from the screening alone.  The
    binary path collapses to the same shape — confirm-reject ⇒
    ``accepted_flag = False`` ⇒ reward ``0``.
  * ``scripts/self_improve.py`` gains ``--confirm-accepts`` and
    ``--confirm-iteration-offset`` flags.  The ``run`` end-of-loop
    summary line and the ``summary`` subcommand surface a separate
    ``Confirm-rej:`` bucket with the % of screening accepts overturned,
    plus a per-record list of overturned screening accepts with
    ``screen_Δ`` / ``confirm_Δ`` / ``pooled_Δ`` / pooled CI so the
    operator can see at a glance whether the gate is catching noise
    spikes (``screen_Δ ≫ confirm_Δ``) or systematic regressions
    (``screen_Δ ≈ confirm_Δ`` but ``ci_low ≤ 0``).

* **Why** — closes §6.4 of ``planning/SELF_IMPROVEMENT_LOOP.md`` and
  the last open half of the V2 §9.5 step 3.  §2.2 of the V2 diagnosis
  identified "Accept → rollback churn (15/16 guard checks rolled the
  ladder back)" as the dominant V1 failure mode: with a ~2.5%
  screening accept rate against the randomized battery, the accepts
  that *did* land were almost always upward-noise spikes — a single
  instance batch where the new kwarg happened to draw a favourable
  combination of perturbations.  The guard subsequently re-measured
  the ladder top on a fresh batch and rolled it back.  Net effect:
  the ladder churned indefinitely; codify-scan saw no durable signal;
  the planning doc's success criterion 3 ("zero guard rollbacks of
  *confirmed* accepts") was structurally unreachable because no
  confirmation step existed.

  The shipped gate inverts this: promotion requires confirmation
  *before* the accept is recorded.  A screening noise spike now sees
  an independent re-measurement on the same night; the pooled CI
  brings the per-instance variance into the gate's decision; the
  arm-level bandit reward reflects the post-confirmation truth.
  Three downstream effects:

  * **Ladder durability** — only confirmed accepts land on the
    ladder, so the guard's job collapses from "roll back ~all
    accepts" to "catch the rare case where a confirmed accept drifts
    on the *next* night's fresh seed".  A guard rollback of a
    *confirmed* accept is the anomaly worth surfacing (§6.3 V2
    note), not routine cleanup.
  * **Bandit signal** — graded mode (shipped 2026-06-13) now sees
    the pooled delta on overturned accepts, so an arm that produces
    consistent noise-spike accepts no longer collects the
    full-accept reward.  The Thompson posterior on such an arm
    decays toward the reject regime over a handful of confirmations,
    where binary-mode V1 would have inflated it permanently.
  * **Codify-scan signal** — the cross-night codify-scan (§9.3,
    still open) will read ``confirmed`` directly to filter out the
    noise-spike accepts that V1 would have piped into the codify
    PRs.  Closes the durability prerequisite of success criterion 3.

* **Backwards compat** — exhaustive.  The default
  ``confirm_accepts = False`` keeps the V1 promote-on-screening
  behaviour byte-identical: ``confirmed`` defaults to ``None`` on the
  iteration record, no :class:`LoopConfirmRecord` is ever written, no
  fresh-iteration measurement runs, and the bandit reward path
  consumes the same screening decision it always did.  Legacy ledger
  lines (no ``confirmed`` key) parse via the dataclass default and
  the new gating is exercised by 25 tests in
  ``tests/test_self_improve.py::TestConfirmationGate*`` /
  ``TestPoolHarnessResults`` / ``TestLoopConfigConfirmAccepts`` /
  ``TestLoopConfirmRecord`` /
  ``TestLoopIterationRecordConfirmedField``.  All 288 prior
  :mod:`panobbgo.self_improve` tests pass unchanged.

* **Impact** — direct effect on §2.2 ("Accept → rollback churn") and
  the V2 §11 success criterion 3 ("Durability: merged codify changes
  re-confirmed by the next night's seed measurement; zero guard
  rollbacks of *confirmed* accepts").  At the loop's current
  ~2.5% binary-mode screening accept rate, every accept is now a
  pooled-CI accept rather than a single-batch noise spike — the
  rollback rate should drop substantially over the first week the
  workflow runs with ``--confirm-accepts``.  Pairs naturally with
  the graded bandit reward (2026-06-13): the gate provides the
  honest signal, the graded reward consumes it.  Closes the last
  blocker for the §9.5 step 5 nightly workflow flip — the only
  remaining open V2 items are the §9.3 ``codify-scan --open-pr``
  stage, the ``--prime-include-archives`` flag, and the §12.4
  summary trend block.

* **Test plan** — :class:`TestConfirmationGateEndToEnd` (8 tests)
  covers the seven dimensions called out in §6.4:

  * **Off by default** — confirm_accepts=False produces no confirm
    record and ``confirmed=None`` on the iteration record (V1
    byte-identical promote-on-screening path).
  * **Confirmation passes** — both screening and confirmation see a
    clearly-winning delta → ``confirmed=True``, ``accepted=True``,
    no confirm_reject record.
  * **Confirmation fails** — screening sees a strong win,
    confirmation sees a strong loss → pooled CI no longer clears →
    ``confirmed=False``, ``accepted=False``, confirm_reject record
    appended with screen + confirm scores.
  * **Screening reject** — gate does not run (the gate only gates
    promotions), so ``confirmed=None`` and the harness saw only the
    two screening measurements.
  * **No-op screening** — gate does not run (no-op iterations are
    filtered upstream), preserving the §12.4 semantics.
  * **Fresh iteration_id** — screening sees
    ``randomize_iteration=0`` while confirmation sees
    ``500_000``; validates the fresh-seed isolation.
  * **Bandit reward post-confirmation** — confirm-reject grants
    reject-regime graded reward (``0 ≤ r ≤ 0.5``); the screening
    full-accept reward (``0.5 ≤ r ≤ 1.0``) is *not* what the bandit
    saw.
  * **Pooled decision uses pooled sample** — the confirm record
    carries the pooled CI and references the fresh iteration_id.

  Plus 4 tests in :class:`TestPoolHarnessResults` covering the
  pooling helper (identity / empty / concat / disjoint-pairs),
  4 tests in :class:`TestLoopConfigConfirmAccepts` covering the new
  validators (defaults / positive offset / guard collision /
  collision allowed when disabled), 4 tests in
  :class:`TestLoopConfirmRecord` covering the new dataclass
  (record_type / serialisation / optional hold-out fields /
  worst_pair=None), 3 tests in
  :class:`TestLoopIterationRecordConfirmedField` covering the
  ``confirmed`` field round-trip, and 1 test in
  :class:`TestConfirmationGateLedgerReplay` covering the JSONL
  round-trip of :class:`LoopConfirmRecord`.

  All 25 new tests pass under ``uv run pytest
  tests/test_self_improve.py``; the full 313-test self-improve suite
  is green; ``uv run ruff check`` and ``uv run pyright`` are clean on
  ``panobbgo/self_improve.py`` and ``scripts/self_improve.py``.

* **Documentation updated**
  - ``planning/SELF_IMPROVEMENT_LOOP.md``: §2.2 annotated with the
    2026-06-14 structural fix; §6.3 V2 note updated to note the
    confirm gate is now in place; §6.4 bullets promoted from
    *open* to *shipped* with pointers to this entry; §9.5 step 3
    sub-task ``--confirm-accepts`` marked shipped.
  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry; a "Next
    iteration ideas" entry seeded for flipping the nightly cron to
    ``--confirm-accepts`` (V2 §9.5 step 5 — the only remaining
    open item in step 3 was this ship's ``--confirm-accepts``).
  - ``doc/source/guide.rst``: quick-nav entry mentions the §6.4
    confirmation gate ship.
  - ``doc/source/guide_benchmarking.rst``: self-improvement loop
    section gains a "Same-night confirmation gate" subsection
    documenting ``LoopConfig.confirm_accepts`` / ``--confirm-accepts``
    and the :class:`LoopConfirmRecord` wire format.
  - ``AGENTS.md``: self-improvement loop subsection references
    the new ``confirm_accepts`` flag, the ``confirmed`` field, and
    the ``LoopConfirmRecord`` wire type.

* **Follow-ups** — speculative, none gated on this ship:

  * Once a few nights of ``--confirm-accepts`` ledger evidence
    accumulates, audit the confirm-reject rate.  A persistently
    high rate (> 50% of screening accepts overturned) would suggest
    the screening ``eps_accept`` is too loose; a persistently low
    rate (< 5%) would suggest the gate is paying its compute cost
    for no measurable benefit.  Threshold-tune from data.
  * Walk *every* configured hold-out base_seed in the confirmation
    step, not just the first.  The current ship caps the per-
    iteration confirmation cost at ``≤ 3×`` screening so the
    compute trade-off is bounded; multi-seed confirmation would
    cap at ``≤ (2 + N_holdout)×`` and give the gate stronger
    cross-family power.  Speculative until ledger evidence shows
    single-seed confirmation misses real overfits.
  * Independent confirmation under the AOCC metric path.  The
    shipped implementation gates the hold-out leg on
    ``metric == "composite"`` because the AOCC path does not use
    the same hold-out machinery; a future ship could plumb the
    same fresh-iteration confirmation through
    :meth:`SelfImprover._measure_aocc` and add an AOCC-aware
    hold-out helper.

### 2026-06-13 — Graded bandit reward shaping (V2 §7.4)

* **What** — Five coordinated additions in
  :mod:`panobbgo.self_improve` plus a CLI flag and a
  :class:`LoopConfig` knob:

  * :class:`MutationRuleStats` gains a ``reward_sum: float = 0.0``
    field plus a ``mean_reward`` property.  A new ``__post_init__``
    mirrors ``n_accepts`` into ``reward_sum`` when the latter is its
    default and the former is non-zero — preserving back-compat for
    direct construction in tests / hand-built priming fixtures and
    making the Thompson posterior byte-identical to the historical
    ``Beta(α₀ + n_accepts, …)`` parameterisation on the binary path.
  * :meth:`AdaptiveMutationSampler.record_outcome` grows an optional
    ``reward`` parameter clamped to ``[0, 1]``.  When omitted (the
    historical call shape) the reward defaults to ``1.0 if accepted
    else 0.0`` so ``reward_sum`` matches ``n_accepts`` exactly.  When
    provided, ``reward_sum`` accumulates the graded value.
    :meth:`AdaptiveMutationSampler.sample` swaps ``reward_sum`` in for
    ``n_accepts`` in the Beta posterior calculation (and the
    ``structural_borrow_alpha`` aggregate), so the posterior shape is
    unchanged on the binary path but distinguishes barely-confirmed
    accepts from clearly-winning ones — and barely-rejected proposals
    from clearly-harmful ones — on the graded path.
  * A new helper :func:`_compute_graded_reward` implements the §7.4
    formula spelt out in the planning doc:

    * ``accepted`` → ``0.5 + clip(ci_low / (4·eps_accept), 0, 0.5)``
      — barely-confirmed accepts (``ci_low ≈ 0``) score ``~0.5``,
      clearly-winning accepts (``ci_low ≥ 4·eps_accept``) saturate at
      ``1.0``.
    * rejected → ``clip(0.5 + Δ / (4·eps_accept), 0, 0.5)`` — a
      positive but sub-eps Δ ("honest near miss") scores ``~0.5``,
      a Δ at zero scores exactly ``0.5``, a clearly-harmful Δ floors
      at ``0``.

    Defensive: any non-positive ``eps_accept`` collapses to ``1e-12``
    so the divide is finite and the clamps still pin the output.
  * :class:`LoopIterationRecord` gains a ``bandit_reward: Optional[float]
    = None`` field serialised via :meth:`to_dict`.  Persists the
    graded value the bandit actually consumed on graded-mode runs;
    ``None`` on skip / no-op iterations and on every iteration of
    binary-mode runs so the ledger can distinguish "the iteration was
    informative but the reward was 0" from "no bandit pull happened".
  * :meth:`AdaptiveMutationSampler.prime_from_ledger` reads the
    ``bandit_reward`` field when present and accumulates the value
    into ``reward_sum``.  Legacy records (no ``bandit_reward`` key)
    fall back to the binary reward ``1.0 if accepted else 0.0`` so
    pre-2026-06-13 ledgers replay byte-identically.
  * :class:`LoopConfig` grows ``bandit_reward_shaping: str =
    "binary"`` (validated to ``{"binary", "graded"}``).  The driver
    in :meth:`SelfImprover._run_loop` calls
    :func:`_compute_graded_reward` and passes the result to
    :meth:`record_outcome` whenever the field is ``"graded"`` and the
    iteration is informative (not skip, not no-op).
  * ``scripts/self_improve.py`` gains a ``--bandit-reward
    {binary,graded}`` flag (default ``binary``).

* **Why** — closes §7.4 of
  ``planning/SELF_IMPROVEMENT_LOOP.md`` and the second open half of
  the V2 §9.5 step 3.  The §2.6 V2 diagnosis identified "Bandit
  starved: binary accept reward at ~2.5% base rate" as a binding
  constraint on per-night posterior productivity: at 20-40 iterations
  with a sub-3% accept rate, almost no arm accumulates positive
  evidence so the Thompson posterior stays close to the symmetric
  ``Beta(1, 1)`` prior on every arm.  Graded shaping converts every
  *informative* iteration — accept *or* reject — into evidence on the
  chosen arm:

  * a barely-rejected proposal (``Δ ≈ 0``) carries ``r ≈ 0.5``: real
    signal that the rule is not harmful;
  * a clearly-harmful reject (``Δ ≈ -4·eps_accept``) carries ``r ≈ 0``:
    real signal that the rule *is* harmful;
  * a barely-confirmed accept carries ``r ≈ 0.5``;
  * a clearly-winning accept carries ``r ≈ 1.0``.

  At a ~30% mean reward (typical for the "honest near miss" regime),
  the Beta posterior moves ``+0.5 / iter`` instead of ``+0 / iter`` on
  the chosen arm, so a 20-iteration night now extracts ~10 units of
  evidence vs ~0 on the binary path.  Arms that consistently produce
  small-positive deltas become distinguishable from harmful arms at
  realistic per-night iteration counts — the §7.4 headline
  guarantee.

  Pairs naturally with the no-op detection shipped 2026-06-12: that
  ship gated zero-information iterations *out* of the posterior;
  this ship gates real-but-sub-eps information *into* the posterior.
  Together they turn the bandit's reward signal from a sparse 0/1 of
  ~2.5% / ~95% / ~2.5% (accept / reject / no-op buckets) into a dense
  graded ``[0, 1]`` signal on the ~65% of iterations that carry real
  information.

* **Backwards compat** — exhaustive.  The default
  ``bandit_reward_shaping = "binary"`` keeps every existing call
  byte-identical: ``record_outcome(accepted)`` with no explicit
  ``reward`` defaults to ``1.0 if accepted else 0.0``, ``reward_sum``
  mirrors ``n_accepts`` (both fresh runs via the driver and direct
  construction via ``MutationRuleStats(...)`` thanks to the
  ``__post_init__`` guard), the Beta posterior consumes ``reward_sum``
  but with the same value, the ledger's ``bandit_reward`` field stays
  ``None`` and the binary-mode round-trip is bit-exact.  Existing 264
  tests pass unchanged; the new ``TestGradedBanditReward`` class adds
  24 tests covering the formula, the stats plumbing, the sampler
  plumbing, the ledger round-trip, and the driver end-to-end on both
  modes.

* **Impact** — direct effect on §2.6 ("Bandit starved") and the V2
  §11 success criterion 2 ("≥ 3 codify PRs opened, ≥ 2 merged" over
  30 nights).  At the loop's current binary-reward base rate (~2.5%),
  a typical mutation rule's posterior is indistinguishable from the
  prior after 20-40 attempts; graded reward shifts the posterior by
  ``r ≈ 0.5`` per "honest near miss" iteration, so the bandit can
  identify productive arms from the ~65% of iterations that carry
  real signal (the §12.4 no-op bucket strips out the rest).  The
  pairing also closes one of the two open halves of V2 §9.5 step 3 —
  only ``--confirm-accepts`` (§6.4) remains before the nightly
  workflow can flip to §9.4 wholesale.

* **Test plan** — :class:`TestGradedBanditReward` (added as a single
  test class for cohesion) covers seven dimensions:

  * **Formula correctness** — accept at zero, half, and full
    ``ci_low``; reject at zero, positive, and full-negative ``Δ``;
    defensive zero-``eps_accept`` handling.
  * **Stats back-compat** — direct construction with ``n_accepts > 0``
    auto-fills ``reward_sum``; explicit ``reward_sum`` is preserved;
    ``mean_reward`` matches ``accept_rate`` on the binary path.
  * **record_outcome** — binary default matches history; graded
    accumulation; out-of-range clamping.
  * **Thompson sampler** — two arms with identical ``n_accepts`` but
    different ``reward_sum`` (0.9 vs 0.05 mean reward) — the higher-
    reward arm wins ``> 85%`` of 200 samples.  Headline guarantee
    that graded reward turns close-to-prior arms into distinguishable
    ones.
  * **prime_from_ledger** — graded records propagate
    ``bandit_reward`` into ``reward_sum``; legacy records fall back
    to binary.
  * **LoopConfig / LoopIterationRecord plumbing** — validation,
    defaults, dataclass field.
  * **End-to-end driver** — binary mode leaves ``bandit_reward =
    None``; graded mode persists it and pulls the bandit with the
    same value; no-op iterations stay ``None`` in both modes; full
    write-then-prime round-trip preserves ``reward_sum`` exactly.

  All 24 new tests pass under ``uv run pytest
  tests/test_self_improve.py``; the full 288-test self-improve suite
  is green.

* **PR** — see this PR.  Pairs naturally with the open
  ``--confirm-accepts`` work (V2 §6.4 / §9.5 step 3): once the
  confirmation gate ships, confirm-reject iterations will land on the
  ``reward = 0`` terminal state spelt out in §7.4 — same code path,
  one extra branch.

* **Follow-ups** — speculative, none gated on this ship:

  * Once a few hundred graded-mode ledger entries have accumulated,
    audit whether arms with high ``mean_reward`` but low
    ``accept_rate`` (the "honest near miss" pattern) graduate to real
    accepts on longer / standard-mode runs.  Evidence for that would
    motivate increasing the relative weight of the reject-regime in
    the formula (currently capped at ``0.5``).
  * The ``eps_scale = 4·eps_accept`` is the planning doc default; a
    follow-up could expose it as a tunable so the bandit can probe
    its own reward shape.  Speculative — the literature on graded
    bandit rewards (Vermorel & Mohri 2005) is thin and the default
    feels reasonable for the ``[0.005, 0.05]`` ``eps_accept`` band the
    loop operates in.

### 2026-06-12 — No-op detection on bandit-pull and ledger telemetry (V2 §12.4)

* **What** — Three coordinated additions in
  :mod:`panobbgo.self_improve`:

  * :class:`LoopIterationRecord` gains a ``no_op: bool = False``
    field and serialises it via :meth:`to_dict`.  Iterations whose
    per-(problem, strategy) candidate scores are bit-identical to
    baseline (a freshly extracted :func:`_is_no_op` helper compares
    the ``problem_strategy_results.score`` maps directly) record
    ``no_op=True`` and ``reason_skipped="no_op"`` and set
    ``accepted=False`` regardless of the statistical-accept verdict
    on the (vacuously zero) delta.  The CI / Δ / worst-pair fields
    are still populated from the bootstrap so an auditor can verify
    the equality after the fact.
  * :class:`AdaptiveMutationSampler` gains a
    :meth:`discard_outcome` method that clears
    :attr:`last_rule_key` without updating the posterior — the same
    end-state as :meth:`record_outcome` but with no
    ``n_attempts += 1`` side-effect.  :meth:`prime_from_ledger`
    skips records carrying ``no_op=True`` (legacy ledgers without
    the field default to ``False`` and continue to replay
    byte-identically to the prior semantics).  The driver loop
    calls :meth:`discard_outcome` instead of :meth:`record_outcome`
    on no-op iterations so the bandit's posterior is not pulled on
    a zero-information event.
  * ``scripts/self_improve.py``: the ``run`` end-of-loop summary
    line and the ``summary`` subcommand's ``Iterations:`` header
    surface a separate ``no-op=N`` bucket; the accept rate is now
    computed over the *informative* denominator (decided − no-op)
    so dormant rules cannot artificially deflate it.

* **Why** — closes the *No-op detection* half of §12.4 in
  ``planning/SELF_IMPROVEMENT_LOOP.md`` (the *Vacuous hold-outs*
  half shipped in parallel as PR #251).  The §2.1 V2 diagnosis
  identified "34% of mutations measure Δ = exactly 0.0000" as the
  dominant V1 failure mode: those iterations carry zero information
  about whether the proposed mutation rule helps or hurts, yet V1
  treated each as a fresh ``n_attempts += 1`` Bernoulli pull on the
  bandit arm.  Two compounding effects:

  * **Bandit posterior mis-trained**: a rule with 4/4 reject-but-
    no-op iterations gets a ``Beta(1, 5)`` posterior even though no
    iteration carried evidence the rule is bad.  Over a night of
    20–40 iterations this systematically biases the Thompson
    sampler toward whichever arms happen to *not* be dormant on the
    current seed registry, defeating §10's "learn which rules win"
    purpose.
  * **Accept rate denominator inflated**: the summary view's
    `accepts / decided` ratio treats no-op records as legitimate
    rejects, so an operator reading the §12.3 daily routine sees an
    artificially low accept rate that conflates dormant rules with
    a productive bandit.

  The shipped fix decouples both: the bandit only pulls on
  iterations carrying real information, and the summary
  distinguishes dormant rules from genuine rejections.  Pairs
  naturally with PR #251 (vacuous hold-out status): both are §12.4
  *honesty* fixes that converted a silently-wrong telemetry signal
  into an explicit ledger field a downstream consumer can branch on.

* **Bit-identical comparison rationale** — the per-pair
  ``score`` is the mean of solve-fractions across reps; under the
  paired-randomized harness, identical specs draw identical instance
  seeds and produce truly equal floats (IEEE 754 equality).  We
  compare per-pair scores rather than the single composite because
  a composite equality is far weaker (two different per-pair
  distributions can average to the same scalar by coincidence) and
  would over-report no-ops.  When the proposal renames a strategy
  or rearranges the pair keyset, ``_is_no_op`` conservatively
  returns ``False`` — the iteration carries real information about
  whether the structural change helps.

* **Impact** — direct effect on §2.1.  Measured against the
  fake-harness test (``test_no_op_iteration_does_not_pull_bandit``):
  two iterations of a constant-score harness — the canonical V1
  "Δ=0" pattern — now produce two no-op records with the bandit's
  ``n_attempts`` at zero.  In nightly cron terms, this means the
  ~34% of mutations that V1 mis-trained on now contribute no
  posterior update at all; the Thompson sampler can identify
  informative arms from the remaining ~66% without the no-op noise
  floor dragging accept-rate posteriors toward zero.  Pure
  telemetry-/gating-only addition: no change to the composite
  baseline, no change to the statistical-accept rule, no change to
  the guard or hold-out semantics.  *Evidence form (per AGENTS.md
  "Agent-driven improve X PRs"): backwards-compatible field default
  (``no_op=False`` on legacy records) so existing ledgers parse and
  replay identically; the new gating is exercised by 10 tests in
  ``tests/test_self_improve.py::TestNoOpDetection`` plus all 1450
  existing tests pass unchanged after the single
  ``test_adaptive_sampler_records_rejects`` fixture update (which
  previously relied on the now-detected-as-no-op constant-score
  path; updated to use distinct baseline/candidate scores for a
  legitimate-reject scenario).*

* **Backwards compatibility** — strictly safe.  The ``no_op``
  field defaults to ``False`` so:

  * Direct dataclass construction without the new kwarg behaves
    bit-for-bit as before.
  * JSONL records written before this ship (no ``no_op`` key on
    disk) load with ``r.get("no_op")`` returning ``None`` /
    ``False`` and are classified as "informative" in the summary —
    matching the historical semantics exactly.
  * :meth:`prime_from_ledger` skips records with
    ``no_op=True`` but processes legacy records (no ``no_op`` key)
    identically to before.
  * The new :meth:`discard_outcome` is purely additive — existing
    callers of :meth:`record_outcome` keep their behaviour.

  The single fixture update
  (``TestSelfImproverAdaptive::test_adaptive_sampler_records_rejects``)
  is a strict improvement: the test previously asserted a
  constant-score iteration counts as a bandit pull, which is
  exactly the behaviour §12.4 says should *not* hold.  Switched to
  a distinct-baseline/candidate score pattern that exercises the
  intended reject path (n_attempts==2 after two legitimate rejects).

* **Tests** — 10 new tests in
  ``tests/test_self_improve.py::TestNoOpDetection``:

  * ``test_default_no_op_field_is_false`` — direct construction
    without the new kwarg defaults to ``False``; ``to_dict``
    persists the field.
  * ``test_identical_pair_scores_flag_no_op`` — end-to-end loop:
    constant-score harness → ``no_op=True``, ``accepted=False``,
    ``reason_skipped="no_op"``, and the reasons list includes the
    "no-op" marker for ledger auditors.
  * ``test_distinct_pair_scores_are_not_no_op`` — legitimate
    reject (candidate strictly worse) is not flagged as no-op so
    the bandit still learns from real signal.
  * ``test_no_op_iteration_does_not_pull_bandit`` — the headline
    contract: ``n_attempts == 0`` after two no-op iterations,
    paired against
    ``test_adaptive_sampler_records_rejects``'s ``n_attempts == 2``
    on the legitimate-reject path.
  * ``test_no_op_iteration_increments_streak`` — inactivity
    streak still advances on no-op iterations so the
    ``inactivity_relax_after`` rule can still break out of a
    long dormant-rule drought.
  * ``test_prime_from_ledger_skips_no_op_records`` — replay path
    matches the live-run gating; consumed count is correctly the
    informative-record count.
  * ``test_prime_from_ledger_legacy_record_replays`` — backwards
    compatibility: pre-ship ledgers without the ``no_op`` key
    continue to prime byte-identically.
  * ``test_discard_outcome_clears_pending_arm`` —
    :meth:`AdaptiveMutationSampler.discard_outcome` clears
    ``last_rule_key`` so the next ``record_outcome`` is a no-op.
  * ``test_no_op_round_trips_through_ledger`` — JSONL round-trip
    preserves the field.
  * ``test_cli_summary_surfaces_no_op_count`` — end-to-end CLI
    smoke check on a fabricated mixed ledger (one no-op, one
    legitimate reject) confirms the new ``no-op=N`` bucket and
    the "informative" denominator label appear in the summary
    output.

  All 244 prior :mod:`panobbgo.self_improve` tests continue to
  pass (with the one fixture update described above); the full
  test suite passes 1450 / 1450.

* **Documentation updated**
  - ``planning/SELF_IMPROVEMENT_LOOP.md``: §2.1 annotated with the
    2026-06-12 update; §9.5 step 3 marks the no-op-detection
    sub-task as shipped; §12.4 first bullet promoted from open →
    shipped with a §13 pointer.
  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry; a new
    "Next iteration ideas" entry seeded for the *pre-measure
    no-op short-circuit* compute-saving follow-up.
  - ``doc/source/guide.rst``: quick-nav entry mentions the §12.4
    no-op detection ship.
  - ``doc/source/guide_benchmarking.rst``: self-improvement
    section documents the new ``no_op`` field and the
    ``discard_outcome`` gating.
  - ``AGENTS.md``: self-improvement loop subsection references
    the new field and the CLI ``no-op=N`` bucket.

### 2026-06-11 — Vacuous hold-out status (V2 §6.4 / §12.4)

* **What** — Three coordinated additions:

  * `panobbgo/self_improve.py`: :class:`LoopHoldoutRecord` gains a
    ``status: str`` field with the three permissible values
    ``("ok", "overfit", "vacuous")`` plus the matching
    :attr:`SUPPORTED_STATUSES` class constant and a constructor-time
    validator that raises ``ValueError`` on typos.  The new
    :meth:`effective_status` helper derives the right verdict from the
    other fields when an explicit status is missing — covers legacy
    ledger lines (no ``status``) by reading ``ladder_size <= 1 and
    top_iteration < 0`` → ``"vacuous"``, ``overfit=True`` →
    ``"overfit"``, otherwise ``"ok"``.  :meth:`to_dict` emits the
    field so the JSONL ledger carries it on every new record.
  * `panobbgo/self_improve.py`: :meth:`SelfImprover._run_holdout`
    branches on the ``seed_only`` predicate (ladder kept only the
    seed entry — no accepted mutations to validate) and sets
    ``status="vacuous"`` rather than mis-reporting the empty-ladder
    case as ``OK drift=+0.0000``.  ``overfit=False`` remains
    bit-identical (vacuous is not overfit), so the existing
    ``--fail-on-overfit`` gate keeps its semantics.
    :meth:`_print_holdout` switches to ``rec.effective_status().upper()``
    so legacy records *and* new records both surface the right
    verdict in the CLI.
  * `panobbgo/self_improve.py`: :func:`aggregate_holdout_drift`
    filters vacuous records out of the bootstrap and the worst-drift
    reduction.  The aggregate gains ``vacuous_count`` and
    ``all_vacuous`` fields so callers can render a faithful summary
    without rerunning the per-record predicate.  The all-vacuous case
    short-circuits to a degenerate aggregate (mirrors the empty-input
    case but records the originating seed and count) with
    ``statistically_overfit=False`` — the aggregate must never claim
    drift on no data.  A regression test asserts that mixing one
    strongly-overfit record with one vacuous record does not soften
    the CI: filtering preserves the negative-drift seed's signal.
  * `scripts/self_improve.py`: ``_cmd_run`` and ``_cmd_summary``
    surface ``VACUOUS`` (per-record) and ``VACUOUS_CI`` (CI
    aggregate) instead of ``OK`` / ``OK_CI`` when the underlying
    records have no informative content.  Both paths use the same
    legacy-aware predicate so summaries of pre-2026-06-11 ledgers
    (no ``status`` field on disk) classify correctly without a
    one-time migration.  The ``Hold-outs:`` headline gains a
    ``vacuous=N`` count alongside ``overfit=N``.

* **Why** — Closes §6.4 / §12.4 of `planning/SELF_IMPROVEMENT_LOOP.md`
  and addresses §2.2 ("all hold-out records ran on an empty ladder,
  vacuous `drift=0.0000` reported as OK") directly.  The previous
  behaviour was actively misleading: an 80-iteration nightly run that
  never accepted a mutation produced a hold-out aggregate that printed
  ``OK drift=+0.0000`` — indistinguishable from a perfectly-generalising
  loop.  Operators reviewing
  ``planning/self_improve_summary.txt`` had no way to see that the
  loop was *vacuous* (no accepted mutations) versus *durable* (every
  accept generalised cleanly).  The ``status`` field collapses that
  ambiguity: ``"vacuous"`` is now a distinct, ledger-persisted
  verdict that bandit-priming code, the codify-scan stage, and the
  summary view can all branch on without re-deriving the predicate.

  The aggregator filter is the second half of the honesty contract:
  pooling six samples (4 informative + 2 vacuous at drift=0) into the
  bootstrap pulled the CI mean toward zero and could mask a single
  negative-drift seed.  Vacuous records contribute literally no
  information about generalisation; excluding them from the bootstrap
  preserves the per-iteration paired drift signal on whatever
  informative records the night actually produced.  The
  ``test_statistically_overfit_not_masked_by_vacuous_record``
  regression test exercises exactly the failure mode the filter
  prevents.

* **Impact** — Telemetry-only change with no behavioural effect on
  the loop's accept / reject decisions or on any heuristic / strategy
  / analyzer.  ``LoopHoldoutRecord.overfit`` is bit-identical for
  every input the previous code accepted — vacuous records still
  carry ``overfit=False`` and so do not trigger ``--fail-on-overfit``
  / ``--fail-on-overfit-ci``.  The bootstrap-CI numbers shift only
  for ledgers containing vacuous records: the previous behaviour
  pooled the zero-drift samples and softened the CI; the new
  behaviour filters them and the CI tightens on whatever informative
  records remain.  *Evidence form (per AGENTS.md "Agent-driven
  improve X PRs"): telemetry-only addition; backwards-compatible
  field default (``status="ok"``) plus a legacy-aware fallback
  (``effective_status``) so pre-ship ledgers classify without a
  migration; the empty-ledger smoke test demonstrates the
  end-to-end CLI verdict flip from ``OK`` to ``VACUOUS_CI`` on a
  legacy record.*

* **Backwards compatibility** — strictly safe.  Every existing
  :class:`LoopHoldoutRecord` constructor call that omits ``status``
  carries the dataclass default ``"ok"``; the JSON wire format gains
  one field without breaking any consumer that uses ``.get("status",
  ...)`` or ignores unknown keys.  Pre-ship ledger lines (no
  ``status``) load into the new dataclass via the default and the
  :meth:`effective_status` helper covers vacuous / overfit
  inference for downstream consumers that care (the summary CLI uses
  this path).  The new
  :class:`HoldoutDriftAggregate` fields ``vacuous_count`` /
  ``all_vacuous`` default to ``0`` / ``False``, so any caller that
  pre-dates the ship and constructs the aggregate directly keeps
  working.  Existing ledger files stay valid; the bandit picks up no
  new arms because this is purely a hold-out telemetry change.

* **Tests** — 7 new tests across
  ``tests/test_self_improve.py`` plus one existing test renamed and
  strengthened:

  * Renamed ``test_seed_only_ladder_records_zero_drift`` →
    ``test_seed_only_ladder_records_vacuous`` and bumped the asserts
    to require ``status="vacuous"`` / ``effective_status()==
    "vacuous"`` / the ``VACUOUS`` reason marker.  The old assertions
    on ``drift==0.0`` / ``overfit is False`` continue to hold so the
    rename is a *strict tightening*.
  * ``TestLoopHoldoutRecord`` (+5 new tests, total 7):
    ``test_status_default_is_ok``,
    ``test_status_validation_rejects_unknown``,
    ``test_supported_statuses_constant``,
    ``test_effective_status_legacy_vacuous_inference``,
    ``test_effective_status_legacy_overfit_inference``,
    ``test_vacuous_status_round_trips_through_to_dict``.
  * ``TestAggregateHoldoutDrift`` (+4 new tests, total 17):
    ``test_vacuous_record_excluded_from_bootstrap`` —
    ``vacuous_count`` reflects the filter, mean drift unchanged by
    the omitted record;
    ``test_all_vacuous_returns_degenerate_aggregate`` — every record
    vacuous → ``all_vacuous=True``, ``statistically_overfit=False``,
    ``n_samples=0``, ``worst_seed`` from the first record;
    ``test_legacy_vacuous_record_classified_by_structure`` — legacy
    records (no ``status``) with ``ladder_size=1`` /
    ``top_iteration=-1`` classify via :meth:`effective_status` so
    pre-ship ledgers stay correct;
    ``test_statistically_overfit_not_masked_by_vacuous_record`` —
    regression guard that mixing one strongly-overfit record with
    one vacuous record does not soften the CI.

  All 254 :mod:`panobbgo.self_improve` tests, 1450 total project
  tests, ruff format / check, and pyright continue to pass.  An
  end-to-end smoke check exercises ``_cmd_summary`` on a fabricated
  legacy ledger line (no ``status`` field, ``top_iteration=-1``,
  ``ladder_size=1``) and verifies the CLI emits ``VACUOUS`` +
  ``VACUOUS_CI`` + ``vacuous=1/1``.

* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §2.2 diagnosis annotated
    with the 2026-06-11 honesty-bug fix; §6.4 closing bullet
    promoted from *open* to *shipped* with a pointer to this entry;
    §11 success criterion 4 annotated with the structural close;
    §12.4 vacuous bullet promoted from *open* to *shipped*.
  - `planning/SELF_IMPROVEMENT_LOG.md`: this entry.
  - `doc/source/guide.rst`: quick-nav entry mentions the vacuous
    hold-out telemetry shift.
  - `doc/source/guide_benchmarking.rst`: hold-out section gains a
    ``VACUOUS`` verdict callout alongside ``OK`` / ``OVERFIT``.
  - `AGENTS.md`: self-improvement loop subsection references the
    new ``status`` field and the
    ``VACUOUS`` / ``VACUOUS_CI`` CLI verdicts.

### 2026-06-10 — Loop registry exercises the dormant catalog (V2 §9.5 step 1)

* **What** — Three coordinated additions:

  * `panobbgo/harness.py`: new :func:`_make_loop_strategies` factory
    that returns the two ``quick`` specs (``RoundRobin_Random``,
    ``Rewarding_Diverse``) **plus** five compact family specs
    targeted at the rule-bearing catalog branches:

    * ``Loop_DE_Family`` — a single ``StrategyRewarding`` spec with
      ``Random`` + LSHADE / JSO / NLSHADE_RSP / NLSHADE_LBC /
      LSHADE_EpSin + ``NelderMead`` and a ``Sensitivity`` analyzer.
      All five DE heuristics ship at ``NP_init = 15`` (inside the
      ``[10, 60]`` catalog bound) so even at the quick-mode 75-eval
      budget each can complete at least one full generation.  Every
      tuned kwarg explicit at the literature default: LSHADE
      ``H=6 / p_best=0.11 / p_best_end=0.055 / archive_factor=1.0 /
      F_schedule=True`` (iLSHADE-style schedule + jSO F-cap),
      JSO ``H=5 / p_best_max=0.25``, NLSHADE_RSP
      ``H=5 / k_rank=3.0 / adaptive_archive=True``, NLSHADE_LBC
      ``H=5 / p_F_init=3.5 / p_F_final=1.5 / p_CR_init=1.0 /
      p_CR_final=1.5 / m_lbc=1.5``, LSHADE_EpSin ``mu_freq_init=0.5``.
    * ``Loop_PSO`` — ``LatinHypercube`` + ``PSO`` + ``NelderMead``.
      PSO carries every tunable kwarg explicit: ``NP=15 /
      w=0.7298 / w_end=0.4 / stagnation_threshold=10 /
      topology="gbest"``.  ``stagnation_threshold`` is pre-staged
      (inert on ``gbest``) so the bandit can flip ``topology`` to
      ``random`` and the stochastic-K rebuild rule fires immediately
      on the same instance.
    * ``Loop_RegionUCB`` — the ``Rewarding_Diverse`` heuristic mix
      plus a ``RegionUCB`` arm with ``ucb_c=1.0 / gauss_fraction=0.5
      / gauss_scale=0.25`` (the three 2026-06-08 catalog rules).
    * ``Loop_LocalSearch`` — ``LatinHypercube`` + ``COBYQA`` (with
      ``initial_tr_radius=0.1 / final_tr_radius=1e-6 / scale=True``)
      + ``LBFGSB`` (with ``max_starts=5``) + ``NelderMead``.  The
      two local optimisers cover every COBYQA / LBFGSB rule
      currently in the catalog.
    * ``Loop_Restart`` — ``LatinHypercube`` + ``CMAES`` (``sigma0=0.3``)
      + ``Random`` + ``Nearby`` + ``NelderMead``, with a ``Restart``
      analyzer (``patience=20 / restart_strategy="random" /
      max_restarts=5``) and a ``Sensitivity`` analyzer (the standard-
      mode ``update_interval=20``).  Activates all three
      :class:`Restart` rules including the categorical
      ``restart_strategy`` arm shipped 2026-06-07.

  * `panobbgo/harness.py`: :class:`HarnessConfig` gains an opt-in
    ``registry: str = "default"`` field; ``"loop"`` routes
    :meth:`BenchmarkHarness.get_strategies` to
    :func:`_make_loop_strategies` regardless of ``mode``, while the
    historical ``"default"`` selects ``quick`` / ``standard`` /
    ``full`` factories per ``mode`` (byte-identical to the prior
    behaviour).  Unknown values raise ``ValueError``;
    ``strategies_override`` continues to win when set.

  * `panobbgo/self_improve.py`: :class:`LoopConfig` gains the
    matching ``registry: str = "default"`` field forwarded to
    :class:`HarnessConfig` by :meth:`SelfImprover._load_seed_strategies`.
    Inert on the AOCC metric path (the IOH battery has its own
    registry, :func:`panobbgo.harness_ioh.make_ioh_strategies`).
    ``scripts/self_improve.py run`` gains
    ``--registry {default,loop}``.

* **Why** — Closes the §9.5 step 1 ticket of the V2 plan and the §2.4
  "catalog ≫ registry mismatch" diagnosis.  The nightly cron runs in
  ``--mode quick`` whose default registry sets only ``Sobol`` /
  ``Nearby`` / ``Sensitivity`` kwargs explicitly.  Every L-SHADE /
  jSO / NL-SHADE-RSP / NL-SHADE-LBC / LSHADE-EpSin / PSO / RegionUCB /
  COBYQA / LBFGSB / Restart mutation rule shipped since mid-May 2026
  (≈30 rules, ~6 weeks of catalog work) sat dormant against this
  registry because no seed spec set the matching kwarg.  Measured
  with :func:`panobbgo.self_improve._find_targets` against the
  ``MutationRule`` entries of :func:`default_catalog`:

  * Quick registry — **4 / 44** kwarg rules fire (Sobol.n,
    Sobol.scramble, Nearby.radius, Sensitivity.update_interval).
  * Loop registry — **44 / 44** kwarg rules fire (all of them).

  The 11× lift in active arms is the prerequisite for the §11
  success criteria; the bandit can finally distinguish *which*
  catalog rule wins on the rule-bearing branches it has accumulated
  over the past six weeks.  No source change to any heuristic /
  analyzer / strategy class — this is pure seed-spec composition.

* **Impact** — Catalog kwarg-rule activation lifts from 4 / 44 to
  44 / 44 (11× wider catalog reachable per iteration).  No-op
  iterations should drop sharply once the §9.5 step 2 metric work
  lands and the bandit can detect the new arms' Δ.  Compute cost
  scales linearly with the spec count: 7 specs (loop) vs 2 specs
  (quick) ≈ 3.5× per-iteration; per §2.5 the cron is currently 94%
  idle so this still fits in the 90-min budget.  No-op default —
  CLI invocations without ``--registry loop`` are byte-identical
  to the prior nightly run.  *Evidence form (per AGENTS.md
  "Agent-driven improve X PRs"): registry-only addition with all
  byte-identical behaviour preserved when ``registry="default"``;
  the new factory is exercised by 15 tests in
  ``tests/test_loop_registry.py`` plus the existing self-improve
  / harness suites.*

* **Backwards compatibility** — strictly safe.  ``HarnessConfig``
  defaults ``registry="default"``; :class:`LoopConfig` defaults
  ``registry="default"``; ``scripts/self_improve.py run`` defaults
  ``--registry default``.  Every existing call site, existing
  ledger entry, existing nightly invocation, and existing test is
  byte-identical.  The new loop registry is purely additive — it
  ships a new factory function on :mod:`panobbgo.harness` and a new
  CLI flag; nothing else changes until a user explicitly passes
  ``--registry loop``.

* **Tests** — 15 new tests in ``tests/test_loop_registry.py``:

  * ``TestLoopRegistryComposition`` (3 tests) — asserts the loop
    registry returns 7 specs, includes both quick specs unchanged,
    and includes the five required family names.
  * ``TestCatalogRuleCoverage`` (2 tests) — the headline contract:
    every :class:`MutationRule` in :func:`default_catalog` matches
    at least one entry in the loop registry; the quick registry's
    coverage stays at the historical baseline of ≤ 10 rules.
    Future catalog additions that target a class missing from the
    loop registry now fail loudly at this gate.
  * ``TestHarnessConfigRegistryWiring`` (4 tests) — ``registry``
    field on :class:`HarnessConfig` correctly dispatches; unknown
    values raise ``ValueError``; ``strategies_override`` still
    wins; ``"loop"`` ignores ``mode``.
  * ``TestLoopConfigRegistryWiring`` (3 tests) — :class:`LoopConfig`
    forwards ``registry`` to the seed-strategy loader and validates
    the value at ``__post_init__`` time.
  * ``TestSelfImproveCliRegistryFlag`` (3 tests) — the
    ``--registry`` flag parses to the correct attribute, defaults
    to ``"default"``, and rejects unknown values via ``SystemExit``.

  All 244 existing :mod:`panobbgo.self_improve` tests, 17 harness
  registry tests, and 22 baseline-strategy tests continue to pass.
  End-to-end smoke check: ``SelfImprover`` with
  ``registry="loop"`` runs a full iteration against the randomized
  quick-mode battery and writes a valid ledger record.

* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §9.1 entry promoted from
    *open* to *shipped* with a pointer to the §13 entry; §9.4
    target invocation annotated to mark ``--registry loop`` as
    shipped; §9.5 step 1 struck through and replaced with the ship
    date + coverage numbers.
  - `planning/SELF_IMPROVEMENT_LOG.md`: this entry.
  - `doc/source/guide.rst`: quick-nav entry mentions the loop
    registry and its ``--registry loop`` opt-in.
  - `doc/source/guide_benchmarking.rst`: self-improvement section
    documents the new factory, its motivation, and the catalog-rule
    coverage measurement.
  - `AGENTS.md`: self-improvement loop subsection documents the
    new ``LoopConfig.registry`` knob and CLI flag.

### 2026-06-09 — Categorical `JSO.p_best_max` rule (literature regimes)

* **What** — `panobbgo/self_improve.py`: :func:`default_catalog`
  gains a ``categorical_choice`` :class:`MutationRule` for the
  ``(JSO, p_best_max)`` slot with ``choices=(0.15, 0.25, 0.4)`` and
  the standard structural-rule probability ``0.3``.  The three
  values are the literature-canonical jSO ``p_best_max`` regimes:

  * ``0.15`` — close to the Tanabe-Fukunaga L-SHADE setting
    ``p_best = 0.11`` (raised above jSO's default
    ``p_best_min = 0.125`` so the constructor's
    ``p_best_min <= p_best_max`` invariant passes without any
    dependent-kwarg coordination).  Greedy regime — the
    ``current-to-pbest`` mutation pulls toward a narrow top slice.
  * ``0.25`` — the Brest et al. (CEC 2017) jSO default.  The
    bandit needs this in the choice set so it can flip *back* to
    the literature setting from any of the alternates.
  * ``0.4`` — the iLSHADE / Brest et al. 2016 broader-pool
    setting.  Broader regime — useful on highly multi-modal
    landscapes where a narrow ``pbest`` slice can lock onto the
    wrong basin.

  Sits alongside the existing ``float_uniform`` rule on the same
  ``(JSO, p_best_max)`` slot (shipped 2026-05-15 with the JSO
  ship); the two rules occupy distinct bandit arms because
  ``_proposal_rule_key`` keys on ``(class_name, param_name,
  rule_kind)``.  The bandit can either continuously walk
  ``p_best_max`` via the float rule or jump between the
  qualitatively distinct regimes via this categorical one.
  Fires only when a spec sets ``p_best_max`` explicitly — the
  constructor default ``0.25`` is filtered out by the established
  opt-in predicate in :func:`_find_targets`, so the rule is
  dormant on the built-in ``add_heuristic`` JSO candidate
  (``{"NP_init": 30}``) and on every other spec that omits the
  kwarg.

* **Why** — closes the *Categorical mutation rule for
  ``JSO.p_best_max``* ticket under *jSO follow-ups (after
  2026-05-15 ship)*.  Before this ship, the only way for the loop
  to reconsider an existing :class:`JSO` instance's
  ``p_best_max`` was the continuous ``float_uniform`` rule, which
  walks the value in ±-style perturbations and cannot reliably
  jump between the three qualitatively distinct regimes.  The
  categorical rule collapses what would otherwise be many
  ``float_uniform`` accepts into a single bandit arm — the same
  pattern that ``LSHADE.archive_factor``, ``LSHADE.F_schedule``,
  and ``NLSHADE_RSP.k_rank`` already use for their respective
  heuristics.  The CEC-2017 (jSO) and CEC-2016 (iLSHADE)
  competition winners disagree on the right setting; letting the
  bandit learn the problem-class-conditional preference from
  ledger evidence is the right policy when the literature is
  itself divided.

  The subtle 0.11 ↦ 0.15 substitution is the dependent-kwarg
  workaround flagged in the planning idea: the L-SHADE-style
  ``0.11`` lies below jSO's default ``p_best_min = 0.125`` and
  would trip the constructor invariant.  Raising to ``0.15``
  preserves the "greedy-regime" semantics (still narrower than
  the jSO ``0.25`` default by a meaningful margin) without
  requiring a coordinated rule that lowers ``p_best_min``
  alongside.  Per the planning doc, the categorical-with-dependent-
  kwarg pattern is deferred until it is needed elsewhere too.

* **Impact** — pure catalog expansion: one new bandit arm covering
  three regimes.  No behavioural change to existing strategies
  (kwarg-explicit predicate); no shifts to the historical
  composite-score baseline; no new dependencies.  The value is
  unlocked once a spec explicitly sets ``p_best_max`` — currently
  none of the built-in factory specs do, so the rule is staged for
  a future hand-tuned ``LSHADE_jSO`` spec (queued under *Ship a
  jSO-tuned ``LSHADE_jSO`` strategy in
  ``_make_standard_strategies``*) or for any structural mutation
  that grows a JSO spec with an explicit ``p_best_max`` kwarg.
  *Evidence form (per AGENTS.md "Agent-driven improve X PRs"):
  catalog-only addition with default behaviour preserved (the jSO
  constructor default ``0.25`` is in the choice set, and the rule
  is dormant on every default spec because none set the kwarg
  explicitly); queued for nightly loop validation via the
  default catalog's new JSO ``p_best_max`` categorical arm.*

* **Backwards compatibility** — strictly safe.  Existing
  :class:`JSO` instances are unaffected: the constructor default
  ``p_best_max = 0.25`` remains unchanged, and the rule cannot
  fire on specs that omit the kwarg from their dict.  All three
  choices satisfy the constructor's ``p_best_min <= p_best_max``
  invariant against jSO's default ``p_best_min = 0.125`` so the
  rule never produces a proposal the constructor would reject.
  Existing ledgers stay valid; the bandit picks up the new arm as
  a fresh ``Beta(1, 1)`` posterior (or, with
  ``--adaptive-prime-from-ledger``, with the inherited op-level
  prior if the hierarchical-borrow knob is in use).

* **Tests** — 4 new tests in
  ``tests/test_heuristic_jso.py::JSORegistrationTests``:

  * ``test_kwarg_catalog_jso_p_best_max_has_both_kinds`` — asserts
    both the ``float_uniform`` and ``categorical_choice`` rules
    are present on the ``(JSO, p_best_max)`` slot (the dual-rule
    invariant that mirrors ``NLSHADE_RSP.k_rank``).
  * ``test_kwarg_catalog_jso_p_best_max_categorical_choices`` —
    asserts exactly three regimes, that ``0.25`` (the jSO
    default) is reachable, and that every choice respects the
    ``p_best_min = 0.125`` floor — guards against any future
    expansion that would re-introduce the 0.11 invariant
    violation.
  * ``test_p_best_max_rule_fires_on_explicit_kwarg`` — end-to-end
    catalog sample test: a spec with ``p_best_max=0.25``
    explicit gets proposals flipping it to ``0.15`` or ``0.4``,
    and both alternates are reachable across 40 draws.
  * ``test_p_best_max_rule_skips_implicit_default`` — confirms
    the rule does not fire on specs that omit ``p_best_max``
    from kwargs (the constructor default ``0.25`` is implicit
    and filtered out by the kwarg-explicit predicate); matches
    the structural catalog's ``add_heuristic`` JSO candidate
    pattern (``{"NP_init": 30}``).
  * ``tests/test_self_improve.py::test_default_catalog_has_categorical_rules``
    extended with the new ``("JSO", "p_best_max")`` membership
    assertion.

* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Categorical mutation rule for ``JSO.p_best_max``*
    next-iteration entry under *jSO follow-ups (after 2026-05-15
    ship)* promoted from "open" to "shipped" with the §13
    reference.
  - `panobbgo/self_improve.py`: :func:`default_catalog`
    docstring lists the new categorical rule under "Categorical
    toggles" alongside the eight existing ones.
  - `doc/source/guide.rst`: quick-nav entry mentions the new
    categorical ``JSO.p_best_max`` rule.
  - `doc/source/guide_benchmarking.rst`: categorical-rules
    section bumped to "nine" with the new rule code-block
    entry.
  - `AGENTS.md`: self-improvement loop subsection adds the
    ``JSO.p_best_max`` rule to the categorical list.

### 2026-06-08 — Catalog rules for `RegionUCB.ucb_c` / `gauss_fraction` / `gauss_scale`

* **What** — Two coordinated additions:

  * `panobbgo/self_improve.py`: :func:`default_catalog` gains three
    new :class:`MutationRule` entries on the RegionUCB
    leaf-bandit knobs:

    * ``RegionUCB.ucb_c`` — ``log_uniform_perturb`` with
      ``bounds=(0.1, 4.0)`` and ``log_step=0.15``.  Controls the
      UCB1 exploration weight in the leaf-bandit score
      ``quality + ucb_c · sqrt(log(N) / n_leaf)``: lower values
      favour exploitation of the currently-best leaf, higher values
      favour uniform-ish allocation across leaves.  The bounds
      bracket the literature default of ``1.0`` (Auer et al.
      2002's canonical UCB1 setting) so a single perturbation can
      probe both regimes.
    * ``RegionUCB.gauss_fraction`` — ``float_uniform`` with
      ``bounds=(0.0, 1.0)``.  Fraction of in-leaf candidates drawn
      from a Gaussian around the leaf's best point instead of
      uniformly over the leaf box.  ``0.0`` reduces RegionUCB to
      a pure uniform-in-leaf sampler (LA-MCTS style); ``1.0``
      makes every draw a local refinement around the leaf best
      (no in-leaf exploration); the constructor default ``0.5``
      balances both modes.
    * ``RegionUCB.gauss_scale`` — ``log_uniform_perturb`` with
      ``bounds=(0.05, 0.5)``.  Standard deviation of the
      Gaussian-around-best draw, expressed as a fraction of the
      leaf's per-axis ranges.  Smaller values produce tighter
      local refinement (close to a Nearby-style neighbourhood),
      larger values approach the uniform-leaf baseline.  The
      constructor default ``0.25`` sits near the geometric centre
      of the log-uniform window.

    All three rules fire only when a spec sets the matching kwarg
    explicitly (the existing :func:`_find_targets` "param already
    in kwargs" predicate); the heuristic constructor defaults
    (``ucb_c=1.0`` / ``gauss_fraction=0.5`` / ``gauss_scale=0.25``)
    remain unchanged and continue to govern specs that leave the
    kwargs at their defaults.

  * `panobbgo/harness.py`: ``Rewarding_RegionUCB`` in
    :func:`_make_standard_strategies` now ships
    ``(RegionUCB, {"ucb_c": 1.0, "gauss_fraction": 0.5, "gauss_scale": 0.25})``
    instead of ``(RegionUCB, {})``.  All three values match the
    constructor defaults so RegionUCB construction is
    byte-identical to the prior form — only the kwarg dict's
    *membership* changes, which is exactly what activates the new
    catalog rules on this seed spec.  Without this change the
    rules would be dormant until a future ship or structural
    mutation explicitly sets them.

* **Why** — closes the *Follow-ups: tune ``ucb_c`` /
  ``gauss_fraction`` via the self-improvement catalog* note in the
  2026-06-05 RegionUCB §13 entry.  Before this ship, RegionUCB's
  three leaf-bandit knobs were tunable only by hand-editing the
  source: the autonomous loop had no vocabulary to perturb them,
  even though they materially affect the exploration / exploitation
  balance of the per-region allocator that ``Rewarding_RegionUCB``
  ships in the standard battery.  The standard-mode A/B measured
  on 2026-06-05 showed RegionUCB +0.302 on ``StyblinskiTang_2D``
  and −0.167 on ``Rosenbrock_2D`` — a per-problem signature
  consistent with a "more exploration" knob having different
  optima on multimodal vs unimodal landscapes.  Adding the three
  kwarg rules lets the bandit learn problem-class-conditional
  settings via the standard per-rule reward signal.

* **Impact** — pure catalog expansion: three new bandit arms,
  zero behavioural change to the existing default battery.  The
  byte-identical seed-spec edit means the historical composite
  baseline is preserved; only the loop's catalog vocabulary grows.
  *Evidence form (per AGENTS.md "Agent-driven improve X PRs"):
  catalog-only addition with default behaviour preserved
  (constructor defaults are the spec values); queued for nightly
  loop validation via the default catalog's three RegionUCB arms
  on the ``Rewarding_RegionUCB`` standard-mode spec.*

* **Backwards compatibility** — strictly safe.  The three kwarg
  values ``ucb_c=1.0`` / ``gauss_fraction=0.5`` /
  ``gauss_scale=0.25`` are the constructor defaults, so
  RegionUCB instances constructed from the updated spec carry
  identical attribute values to before.  The rules use the
  established kwarg-explicit predicate so they cannot fire on
  any spec that omits the kwarg from its dict.  Existing ledgers
  stay valid; the bandit picks up the new arms as fresh
  ``Beta(1, 1)`` posteriors (or, with
  ``--adaptive-prime-from-ledger``, with the inherited op-level
  prior if the hierarchical-borrow knob is in use).

* **Tests** — 5 new tests in
  ``tests/test_heuristic_region_ucb.py``:

  * ``test_kwarg_catalog_has_region_ucb_ucb_c_rule`` — asserts the
    rule is present with the documented ``log_uniform_perturb``
    kind, the ``(0.1, 4.0)`` bounds bracket the literature default
    of ``1.0``.
  * ``test_kwarg_catalog_has_region_ucb_gauss_fraction_rule`` —
    asserts the rule is present with ``float_uniform`` kind and
    the full ``[0, 1]`` range is bandit-reachable (so the LA-MCTS
    pure-uniform regime at ``0.0`` and the pure-local-refinement
    regime at ``1.0`` are symmetrically reachable).
  * ``test_kwarg_catalog_has_region_ucb_gauss_scale_rule`` —
    asserts the rule is present with ``log_uniform_perturb`` and
    the ``(0.05, 0.5)`` bounds.
  * ``test_region_ucb_rules_skip_implicit_default`` — confirms
    the rule fires only on specs that explicitly set ``ucb_c``;
    a spec with ``(RegionUCB, {})`` is never selected.
  * ``test_rewarding_region_ucb_seed_spec_has_explicit_region_ucb_kwargs``
    — asserts the seed ``Rewarding_RegionUCB`` spec ships the
    three explicit kwargs at the constructor defaults so the
    new catalog rules become applicable to the standard-mode
    battery rather than staying dormant.

* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Follow-ups* note in the 2026-06-05 RegionUCB entry updated
    to reference the new catalog rules.
  - `panobbgo/self_improve.py`: :func:`default_catalog`
    docstring lists the three new RegionUCB rules.
  - `doc/source/guide.rst`: quick-nav entry mentions the new
    RegionUCB catalog rules.
  - `doc/source/guide_benchmarking.rst`: kwarg-catalog section
    bumped with the three new rules.
  - `AGENTS.md`: rule list bumped with the three new RegionUCB
    arms.
  - `TODO.md`: "Recent Improvements" entry.

### 2026-06-07 — Categorical `Restart.restart_strategy` rule + `"sphere"` regime

* **What** — Two coordinated additions:

  * `panobbgo/analyzers/restart.py`:
    :class:`~panobbgo.analyzers.restart.Restart` gains support for a
    third ``restart_strategy`` value, ``"sphere"`` — picks the new
    center via :meth:`Problem.random_point(distribution="normal")`,
    i.e. a Gaussian draw centered at the box centre with
    ``std = ranges / 6`` (clipped to the box).  Biases the restart
    cloud toward the centroid; complements the two existing
    policies ``"random"`` (uniform-in-box) and ``"diverse"``
    (max-min-distance from previous restart centres).  The
    constructor now validates ``restart_strategy`` against the new
    :attr:`Restart.SUPPORTED_RESTART_STRATEGIES` class constant and
    raises ``ValueError`` on unknown values — guards future catalog
    expansions against accidental typos.  No change to the default
    (``"random"``).
  * `panobbgo/self_improve.py`: :func:`default_catalog` gains a
    ``categorical_choice`` :class:`MutationRule` for the
    ``(Restart, restart_strategy)`` slot with
    ``choices=("random", "diverse", "sphere")`` and the standard
    structural-rule probability ``0.3``.  Fires only when a spec
    sets ``restart_strategy`` explicitly (the existing
    "param already in kwargs" predicate); the analyzer's
    constructor default ``"random"`` is filtered out so specs that
    omit the kwarg are never mutated.  Joins the seven existing
    categorical rules (``PSO.topology`` / ``Sobol.scramble`` /
    ``LSHADE.archive_factor`` / ``LSHADE.F_schedule`` /
    ``NLSHADE_RSP.adaptive_archive`` / ``NLSHADE_RSP.k_rank`` /
    ``COBYQA.scale``).
* **Why** — closes the *Categorical ``Restart.restart_strategy``
  regimes* ticket under *Analyzer add/drop follow-ups (after
  2026-06-02 ship)*.  Previously the only way for the loop to
  reconsider an existing :class:`Restart` instance's
  ``restart_strategy`` was to drop the analyzer (via the structural
  catalog's ``drop_analyzer`` op) and re-add it with a different
  kwarg dict — two iterations of mutation budget for one effective
  knob flip.  The categorical rule collapses that to one
  iteration, the same pattern that ``PSO.topology`` /
  ``Sobol.scramble`` already use for their respective heuristics.
  The new ``"sphere"`` regime adds a genuinely distinct
  center-selection bias — uniform-in-box gives no information
  about where the optimum is expected; max-min-distance is purely
  geometric (only relevant once multiple restarts have fired);
  Gaussian-around-centre is the first regime that encodes a prior
  on where the optimum is *likely* to live (the centroid of the
  box), which is the right prior on problems where the
  experimenter has centred the box on a domain of interest.
* **Impact** — pure catalog expansion: one new bandit arm covering
  three regimes.  All four built-in factory spots that ship a
  :class:`Restart` instance with an explicit
  ``restart_strategy="diverse"`` (``IPOP_CMAES`` and
  ``BIPOP_CMAES`` in :mod:`panobbgo.harness`,
  ``Sensitivity_Aggressive`` in :mod:`panobbgo.harness_ioh`, and
  the structural catalog's ``add_analyzer`` candidate) become
  applicable to the new rule out-of-the-box, so the bandit can
  immediately learn whether the IPOP-style ``"diverse"`` default
  is in fact best on the standard / IOH battery or whether one of
  the alternatives wins.  *Evidence form (per AGENTS.md
  "Agent-driven improve X PRs"): catalog-only addition with the
  default behaviour preserved (``"diverse"`` is still the seed
  composition's pick); backwards-compatible (composite baseline
  byte-identical, existing ledgers stay valid); queued for nightly
  loop validation via the default catalog's
  ``Restart.restart_strategy`` arm.*
* **Backwards compatibility** — strictly safe.  The constructor
  default for ``restart_strategy`` remains ``"random"``; every
  existing :class:`Restart` instance retains its prior behaviour
  bit-for-bit.  The new ``"sphere"`` regime is reachable only by
  passing it explicitly to the constructor or via the new
  categorical rule's draw.  The new validation in
  :meth:`Restart.__init__` is strict-superset compatible — it
  accepts every value the prior code accepted (the two-element
  ``"random"`` / ``"diverse"`` set) plus the new ``"sphere"``
  entry, and rejects values the prior code would have silently
  treated as "uniform random" (the ``else`` branch in
  :meth:`_pick_new_center`); the only behavioural change is that
  invalid values now raise instead of silently falling through.
  Existing ledger consumers parsing only known
  ``rule_kind=categorical_choice`` entries see one extra rule key
  they may ignore.
* **Tests** — `tests/test_analyzer_restart.py` (+6 new tests, total
  23):
  * ``test_sphere_strategy_uses_normal_distribution`` —
    ``restart_strategy='sphere'`` produces Gaussian draws around the
    box centre (empirical mean within tolerance of the centroid,
    all draws inside the box).
  * ``test_sphere_strategy_independent_of_previous_centers`` —
    distinguishes ``"sphere"`` from ``"diverse"`` by injecting a
    fake corner-anchored previous centre and confirming the new
    center is still centroid-biased rather than anti-correlated
    with the injected corner.
  * ``test_invalid_restart_strategy_raises`` — constructor rejects
    unknown ``restart_strategy`` with a clear ``ValueError``.
  * ``test_supported_restart_strategies_constant`` — the
    ``SUPPORTED_RESTART_STRATEGIES`` class constant lists exactly
    the three implemented policies.
  * ``test_kwarg_catalog_has_restart_strategy_rule`` — catalog
    membership test that asserts the rule's kind, choices, and
    that every choice is in
    ``Restart.SUPPORTED_RESTART_STRATEGIES`` — guards against
    catalog / analyzer drift.
  * ``test_restart_strategy_rule_fires_on_explicit_kwarg`` —
    end-to-end catalog sample test confirming the rule emits
    proposals that flip an existing ``"diverse"`` spec to one of
    ``"random"`` or ``"sphere"``, and that both alternatives are
    reachable.
  * ``test_restart_strategy_rule_skips_implicit_default`` — the
    rule must not fire on specs that omit
    ``restart_strategy`` from the kwargs dict (the implicit
    constructor default ``"random"``).
  * `tests/test_self_improve.py::test_default_catalog_has_categorical_rules`
    extended with the new ``("Restart", "restart_strategy")``
    membership assertion.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Categorical ``Restart.restart_strategy`` regimes*
    next-iteration entry under *Analyzer add/drop follow-ups*
    promoted from "open" to "shipped" with the §13 reference.
  - `panobbgo/analyzers/restart.py`: class docstring expanded
    with the three-way ``restart_strategy`` list.
  - `panobbgo/self_improve.py`: :func:`default_catalog`
    docstring lists the new categorical rule alongside the
    seven existing ones.
  - `doc/source/guide.rst`: quick-nav entry mentions the new
    categorical ``Restart.restart_strategy`` rule and the
    ``"sphere"`` regime.
  - `doc/source/guide_benchmarking.rst`: categorical-rules
    section bumped to "eight" with the new rule code-block
    entry.
  - `doc/source/guide_usage.rst`: ``Restart`` parameter list
    expanded with the three ``restart_strategy`` regimes.
  - `AGENTS.md`: self-improvement loop subsection adds the
    ``Restart.restart_strategy`` rule to the categorical
    list.

### 2026-06-06 — Catalog rules for the under-tuned Restart.patience and LBFGSB.max_starts dials

* **What** — `panobbgo/self_improve.py`: :func:`default_catalog` gains two
  ``integer_add`` :class:`MutationRule` entries that fill known gaps in
  the analyzer / local-optimizer dial coverage:

  * ``Restart.patience`` — ``integer_add`` with ``bounds=(3, 200)`` and
    ``delta_choices=(-20, -10, -5, 5, 10, 20)``.  Counts consecutive
    non-improvement evaluations before a restart fires; the more
    impactful of the two :class:`~panobbgo.analyzers.restart.Restart`
    dials (alongside the existing ``Restart.max_restarts`` rule).  The
    analyzer's default is ``5 · dim`` (auto-derived at ``__start__``);
    the built-in factories (``IPOP_CMAES`` in the standard battery,
    ``BIPOP_CMAES`` in the full battery) deliberately ship
    ``patience=None`` to opt into the auto-default.
  * ``LBFGSB.max_starts`` — ``integer_add`` with ``bounds=(1, 50)`` and
    ``delta_choices=(-5, -2, -1, 1, 2, 5)``.  Caps the multi-start
    L-BFGS-B restart budget; ``1`` reduces the heuristic to a pure
    box-centre descent, larger values give the random-restart layer
    more chances to find a different basin.  The heuristic's default
    is ``None`` (= unlimited until the strategy budget is exhausted);
    the structural catalog's ``add_heuristic`` candidate ships
    ``{}`` (also auto-default).

  Both rules fire only when a spec sets the matching kwarg to a
  *concrete non-``None`` value*.  This required a one-line change to
  :func:`_find_targets`: the "param already in kwargs" predicate now
  also requires ``kwargs[param_name] is not None`` — ``None`` is the
  auto-default sentinel a number of heuristics use, and numeric
  mutation kinds (``integer_add`` / ``float_uniform`` /
  ``log_uniform_perturb``) cannot meaningfully perturb it.  The
  ``None``-skip is uniform across rule kinds and applies to every
  catalog rule, not just the two new ones, but is behaviourally inert
  for the previously-shipped catalog because no prior rule's target
  spec carried a ``None``-valued kwarg.
* **Why** — closes two of the *Next iteration ideas* tickets in one
  focused PR:

  * *``Restart.patience`` mutation rule* (the most-impactful Restart
    knob — controls how aggressively the optimizer restarts when stuck).
  * *``LBFGSB.max_starts`` catalog rule* under the *LBFGSB follow-ups*
    block — lets the loop tune the multi-start exploration /
    exploitation balance the same way ``LSHADE.archive_factor`` is
    tuned.

  Both fit the established opt-in catalog pattern (the kwarg-explicit
  predicate from :func:`_find_targets`) and the ``integer_add`` numeric-
  rule shape shared by ``LSHADE.NP_init`` / ``LSHADE.H`` /
  ``Restart.max_restarts`` / ``Sensitivity.update_interval``.  Per-class
  ``__name__`` matching means each rule lives in exactly one
  ``(class, param, kind)`` bandit arm, so the per-class structural
  bandit arms (shipped 2026-05-18) can learn each independently.
* **Impact** — pure catalog expansion: two new bandit arms.  No
  behavioural change to existing strategies (kwarg-explicit predicate),
  no shifts to the historical composite-score baseline, no new
  dependencies.  The value is unlocked once a spec explicitly sets the
  kwarg or once the bandit accumulates per-arm reward history — the
  same delayed-payoff shape every prior catalog expansion has shown
  (cf. the 2026-06-04 ship for ``JSO.H`` /
  ``NLSHADE_RSP.H`` / ``NLSHADE_RSP.k_rank`` / ``COBYQA.scale``).
  *Evidence form (per AGENTS.md "Agent-driven improve X PRs"): the
  change is strictly additive — pure bandit-vocabulary expansion with
  no alteration to the default battery — and queued for nightly loop
  validation.*
* **Backwards compatibility** — strictly safe.  Each new rule fires
  only when the target spec sets the matching kwarg to a concrete
  non-``None`` integer (existing :func:`_find_targets` semantics
  extended with the ``None``-skip); no default
  ``_make_quick_strategies`` / ``_make_standard_strategies`` /
  ``_make_full_strategies`` spec is modified.  The :class:`Restart`
  analyzer instances in ``IPOP_CMAES`` / ``BIPOP_CMAES`` ship
  ``patience=None`` so they remain inert under the new rule.  Existing
  ledgers are untouched.  The ``None``-skip is behaviourally inert for
  all previously-shipped catalog rules (no prior rule's target spec
  carries a ``None``-valued kwarg, as verified by the existing
  ``test_default_catalog_has_*`` tests).
* **Tests** — 5 new tests:
  ``tests/test_analyzer_restart.py`` (+2 —
  ``test_kwarg_catalog_has_restart_patience_rule`` asserts the rule
  is present with the documented ``integer_add`` kind, bounds, and a
  symmetric ``delta_choices`` cone;
  ``test_restart_patience_rule_skips_none_sentinel`` asserts the rule
  never proposes against a ``patience=None`` spec and always
  proposes against a ``patience=25`` spec, with the new value clamped
  to bounds);
  ``tests/test_heuristic_lbfgsb.py`` (+2 — symmetric pair for
  ``LBFGSB.max_starts``);
  ``tests/test_self_improve.py`` (+1 —
  ``test_applicable_rules_skips_none_value`` asserts the
  :func:`_find_targets` predicate change is uniform across rule
  kinds, not just for the two new rules).  Full suite still passes
  on the touched files (286 tests).
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *``Restart.patience`` mutation rule* and *``LBFGSB.max_starts``
    catalog rule* next-iteration entries promoted from "open" to
    "shipped".
  - `panobbgo/self_improve.py`: :func:`default_catalog` docstring
    lists the two new entries alongside the existing dials.
  - `doc/source/guide.rst`: quick-nav entry mentions the
    catalog-completion ``Restart.patience`` and ``LBFGSB.max_starts``
    rules.
  - `doc/source/guide_benchmarking.rst`: kwarg catalog list extended
    with the two new entries.
  - `AGENTS.md`: kwarg catalog rule list bumped with the two new
    entries.

### 2026-06-05 — Stochastic-K stagnation rebuild for the random PSO topology (Clerc 2007 / SPSO 2011)

* **What** — `panobbgo/heuristics/pso.py`: :class:`PSO` gains an
  opt-in ``stagnation_threshold: Optional[int] = None`` kwarg, an
  ``_stagnation_counter`` attribute, and a new
  :meth:`_maybe_rebuild_random_adjacency` helper that wraps the
  Clerc 2007 / SPSO 2011 stochastic-K stagnation-rebuild policy.
  When set to a positive integer and the topology is ``"random"``,
  the counter ticks on every incoming result that does *not* lift
  ``_gbest_idx``; once it reaches ``stagnation_threshold`` the
  adjacency is re-sampled from the heuristic's RNG and the counter
  resets.  The counter also resets on every strict global-best
  improvement, on :meth:`on_start`, and on :meth:`on_restart`.
  ``stagnation_threshold=None`` (default) bypasses the policy
  entirely so existing :class:`PSO` instances retain their prior
  static-between-restarts behaviour bit-for-bit.
  :func:`default_catalog` gains a matching
  ``PSO.stagnation_threshold`` ``integer_add`` rule
  (``bounds=(5, 60)``, ``delta_choices=(-10, -5, 5, 10)``) so the
  loop can tune the rebuild cadence on any spec that opts in.  The
  rule fires only when a spec sets the kwarg explicitly (per
  :func:`_find_targets`'s "param already in kwargs" predicate), so
  the built-in factories that leave ``stagnation_threshold=None``
  see no behavioural change.
* **Why** — closes the *Per-iteration re-sampled random PSO
  topology (stochastic-K)* follow-up below the 2026-05-29 random
  PSO topology entry.  The random topology shipped 2026-05-29
  re-samples the informer graph only at ``on_start`` and
  ``on_restart``.  Under :class:`~panobbgo.analyzers.restart.Restart`
  restarts are rare — the stochastic graph can otherwise stay locked
  into a bad realised adjacency for hundreds of incoming results,
  defeating the structure-free flexibility motivation for the random
  topology in the first place.  Clerc 2007 / SPSO 2011 standardises
  a stricter "stochastic-K" variant that rebuilds the graph on
  stagnation; this is the literature-faithful completion.  The
  rebuild trigger uses the *constraint handler's* ``is_better``
  predicate (the strict improvement gate already used by
  :meth:`_update_global_best`) so the stagnation count tracks the
  global-best lift even under penalty-based constraints.
* **Asynchronous adaptation** — the policy lives in the
  ``on_new_results`` path and reads the swarm's true
  ``_gbest_idx`` lift on every result, so it stays in lock-step
  with the panobbgo async pipeline (one trial per particle pending
  at a time; rebuild fires lazily as misses accumulate).  No
  state changes between ``on_start`` / ``on_restart``;
  :meth:`_maybe_rebuild_random_adjacency` is a no-op for any
  topology other than ``"random"`` and for ``stagnation_threshold
  = None`` (the default).
* **Impact** — the value of shipping today is to give the bandit a
  knob it currently lacks for the random topology: per-arm reward
  history can identify whether mid-run rebuilds help on a given
  battery.  At quick-mode budgets the immediate signal is within
  noise (single-rebuild bursts that fire late in the budget barely
  matter for AOCC / composite_score on a 75-eval / 300-eval run),
  but the literature (Clerc 2007; SPSO 2011) reports the
  stochastic-K rebuild as the dominant ingredient that lets random
  topologies match the structured variants on long-budget runs
  where restart-gated re-sampling is too coarse.  *Evidence form
  (per AGENTS.md "Agent-driven improve X PRs"): catalog-only
  addition with default kwarg ``None``; backwards-compatible
  (composite baseline byte-identical, existing ledgers stay valid);
  queued for nightly loop validation via the default catalog's
  ``PSO.stagnation_threshold`` rule and the structural catalog's
  ``random`` PSO entry.*
* **Backwards compatibility** — strictly safe.
  ``stagnation_threshold`` defaults to ``None``; every existing
  PSO instance retains its prior behaviour bit-for-bit, including
  all 68 pre-existing tests in ``tests/test_heuristic_pso.py``.
  ``_stagnation_counter`` is initialised to ``0`` and never read
  unless the policy is opted in and the topology is ``"random"``,
  so memory / RNG draws on every other code path are byte-identical.
  The new ``PSO.stagnation_threshold`` catalog rule only fires
  when a spec explicitly sets the kwarg (per :func:`_find_targets`'s
  "param already in kwargs" predicate), so the built-in
  ``_make_quick_strategies`` / ``_make_standard_strategies`` /
  ``_make_full_strategies`` factories see no behavioural change.
  Existing ledger consumers parsing only known kinds see one extra
  ``integer_add`` rule they may ignore.
* **Tests** — `tests/test_heuristic_pso.py` (+13 new tests, total
  81): default ``stagnation_threshold`` is ``None``, custom
  round-trip, ctor rejects non-integer and bool, ctor rejects
  zero / negative; counter starts at zero after ``on_start``;
  counter resets on every strict global-best improvement; rebuild
  fires exactly at the threshold and resets the counter; below the
  threshold the adjacency is untouched; ``None`` default never
  rebuilds the adjacency mid-run even under many non-improvements;
  no-op for ``gbest`` / ``lbest`` / ``vonneumann`` topologies
  (the three geometric variants have no random graph);
  ``on_restart`` resets the counter even mid-stagnation; the very
  first global-best observation does not tick the counter.  Plus a
  catalog membership test confirming
  ``("PSO", "stagnation_threshold")`` joins the default rule set
  with the documented ``integer_add`` kind and bounds.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Per-iteration re-sampled random PSO topology (stochastic-K)*
    next-iteration idea promoted from "open" to "shipped".
  - `doc/source/guide.rst`: quick-nav entry mentions the optional
    stochastic-K stagnation-rebuild ``PSO.stagnation_threshold``
    knob for the ``random`` PSO topology.
  - `doc/source/guide_benchmarking.rst`: structural-catalog PSO
    paragraph now describes the ``random`` variant's
    ``stagnation_threshold`` knob and the matching default-catalog
    rule.
  - `doc/source/guide_architecture.rst`: PSO description gains the
    stochastic-K stagnation rebuild paragraph after the random
    topology description.
  - `doc/source/heuristics.rst`: PSO bullet mentions the optional
    ``stagnation_threshold`` for the random topology.
  - `AGENTS.md`: self-improvement loop subsection adds the
    ``PSO.stagnation_threshold`` rule to the kwarg-rules list.

### 2026-06-04 — Catalog completion for jSO / NL-SHADE-RSP / COBYQA dials

* **What** — `panobbgo/self_improve.py`: :func:`default_catalog` gains
  four new :class:`MutationRule` entries that close known gaps in the
  per-heuristic dial coverage:

  * ``JSO.H`` — ``integer_add`` with ``bounds=(4, 12)``.  Mirrors the
    existing ``LSHADE.H`` rule for the subclass.  Brest et al. (2017)
    report ``H = 5`` as best for the CEC battery (vs L-SHADE's
    ``H = 6``); previously the catalog had no way to tune ``H`` on a
    jSO instance because the rule's exact-class-name match
    (``cls.__name__ == "JSO"``) did not inherit the L-SHADE rule.
  * ``NLSHADE_RSP.H`` — ``integer_add`` with ``bounds=(4, 12)``.
    Symmetric with the new ``JSO.H`` rule; inherits the
    ``H >= 2`` anchor-bin constraint from jSO.  Same motivation
    (per-class match does not inherit).
  * ``NLSHADE_RSP.k_rank`` (categorical) — ``("0.0", "3.0", "5.0")``
    literature regimes, sitting *alongside* the existing
    ``float_uniform`` rule (``bounds=(1.0, 5.0)``).  Two distinct
    bandit arms by construction (different ``rule_kind`` → different
    `_proposal_rule_key`), so the Thompson sampler can learn whether
    the continuous walk or the regime jump pays off on the current
    battery.  ``0.0`` is unreachable from the continuous rule and
    gives the loop a way to switch off rank-based pressure entirely
    (= jSO recovery) on portfolios that opted into NL-SHADE-RSP.
  * ``COBYQA.scale`` (categorical) — ``(True, False)``.  Flips the
    box-rescaling behaviour: ``True`` (the COBYQA default) rescales
    variables to ``[-1, 1]`` to keep the Powell interpolation
    geometry well-conditioned; ``False`` runs COBYQA on the raw box.
    Useful when the problem's box is already isotropic and the
    rescale adds rounding noise that hurts the quadratic-model fit.

  Each fires only when a spec sets the matching kwarg *explicitly*
  (the existing :func:`_find_targets` "param already in kwargs"
  predicate), so a fresh ledger run on the built-in factories sees
  no behavioural change.  Of the shipped strategies, the structural
  catalog's NL-SHADE-RSP candidate sets ``k_rank=3.0`` explicitly so
  the new categorical rule fires out-of-the-box once a portfolio
  gains the heuristic via ``add_heuristic``; the jSO ``H`` and
  ``NLSHADE_RSP.H`` rules and ``COBYQA.scale`` become applicable
  whenever a spec opts in.
* **Why** — closes three of the "Next iteration ideas" tickets in
  one focused PR:

  * *Auto-tuned ``H``* under the jSO follow-ups — Brest et al. report
    ``H = 5`` best; the constructor enforces ``H >= 2`` (anchor bin
    separation).
  * *Categorical ``k_rank`` regimes* under the NL-SHADE-RSP
    follow-ups — three literature-canonical settings give the bandit
    a way to flip the selective-pressure regime discretely, the same
    way ``LSHADE.archive_factor`` flips archive on / off / RSP.
  * *Categorical mutation rule for ``scale`` on/off* under the COBYQA
    follow-ups — a discrete toggle the bandit can flip without going
    through the full ``add_heuristic`` / ``drop_heuristic`` cycle.

  All three fit the established 2026-05-13 categorical-rule pattern
  (5 categorical rules already shipped) and the
  ``LSHADE.H`` / ``LSHADE.NP_init`` numeric-rule pattern.  Per-class
  ``__name__`` matching means each catalog rule lives in exactly one
  ``(class, param, kind)`` bandit arm, so the per-class structural
  bandit arms (shipped 2026-05-18) can learn each independently.
* **Impact** — pure catalog expansion: four new bandit arms.  No
  behavioural change to existing strategies (kwarg-explicit
  predicate), no shifts to the historical composite-score baseline,
  no new dependencies.  The value is unlocked once the bandit
  accumulates per-arm reward history — the same delayed-payoff
  shape every prior catalog expansion has shown (cf. the structural
  catalog's per-class arms shipped 2026-05-18).  *Evidence form
  (per AGENTS.md "Agent-driven improve X PRs"): the change is
  strictly additive — pure bandit-vocabulary expansion with no
  alteration to the default battery — and queued for nightly loop
  validation.*
* **Backwards compatibility** — strictly safe.  Each new rule fires
  only when the target spec sets the matching kwarg explicitly
  (existing :func:`_find_targets` semantics); no default
  ``_make_quick_strategies`` / ``_make_standard_strategies`` /
  ``_make_full_strategies`` spec is modified.  Existing ledgers are
  untouched.  Existing tests for the per-heuristic catalog rules
  continue to pass; the matching membership tests are extended to
  cover the new rules.  The bandit arm layout follows
  :func:`_proposal_rule_key` — distinct ``(class, param, kind)``
  tuples — so the new arms are independent of any existing rule
  even when they share a slot (``NLSHADE_RSP.k_rank`` carries both
  a ``float_uniform`` and a ``categorical_choice`` arm).
* **Tests** — 5 new tests covering the new rules:
  ``tests/test_heuristic_jso.py`` (+1 — ``JSO.H`` kind / bounds);
  ``tests/test_heuristic_nl_shade_rsp.py`` (+3 — ``NLSHADE_RSP.H``
  kind / bounds, ``NLSHADE_RSP.k_rank`` has both kinds, the
  categorical choices include ``0.0`` and ``3.0`` and are
  non-negative floats);
  ``tests/test_heuristic_cobyqa.py`` (+1 — ``COBYQA.scale`` kind /
  choices).  Plus existing membership assertions extended:
  ``tests/test_heuristic_jso.py::test_kwarg_catalog_has_jso_dials``
  (adds ``("JSO", "H")``);
  ``tests/test_heuristic_nl_shade_rsp.py::test_kwarg_catalog_has_rsp_dials``
  (adds ``("NLSHADE_RSP", "H")``);
  ``tests/test_heuristic_cobyqa.py::test_kwarg_rules_present`` (adds
  ``("COBYQA", "scale")``);
  ``tests/test_self_improve.py::test_default_catalog_has_categorical_rules``
  (asserts the categorical rule set now contains
  ``("NLSHADE_RSP", "k_rank")`` and ``("COBYQA", "scale")`` along
  with the prior five).  Full suite: 1158 passed, 11 skipped.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Auto-tuned ``H``* and *Categorical mutation rule for
    ``JSO.p_best_max``* / *Categorical ``k_rank`` regimes* /
    *Categorical mutation rule for ``scale`` on/off* follow-ups
    updated.
  - `doc/source/guide_benchmarking.rst`: categorical-rule section
    expanded to cover ``NLSHADE_RSP.k_rank`` and ``COBYQA.scale``;
    "ships seven categorical rules" replaces the "five" count.
  - `doc/source/guide.rst`: quick-nav entry mentions the new
    categorical knobs.
  - `AGENTS.md`: categorical-rules list bumped from five to seven;
    the new ``NLSHADE_RSP.k_rank`` literature-regime entry and
    ``COBYQA.scale`` toggle are listed.

### 2026-06-03 — LSHADE-EpSin adaptive DE (CEC 2016, sinusoidal-F branch)

* **What** — `panobbgo/heuristics/lshade_ep_sin.py` adds the
  :class:`LSHADE_EpSin` heuristic, a direct subclass of
  :class:`~panobbgo.heuristics.lshade.LSHADE` that ports the Awad-Ali-
  Suganthan (CEC 2016) "LSHADE-EpSin" refinement.  LSHADE-EpSin inherits
  the entire L-SHADE asynchronous pipeline (per-slot pending dict,
  generation-by-count book-keeping, archive of replaced parents,
  success-history Normal CR sampling, ``current-to-pbest/1`` mutation
  skeleton, linear population reduction, midpoint-reflection bounds
  repair, warm restart) and replaces only the ``F`` sampler with an
  ensemble of two sinusoidal candidates during the first half of the
  search:

  * **Sinusoid 1** (fixed frequency, *decreasing* envelope)::

        F = 0.5 · ( sin(2π · freq_fixed · g) · (G_max − g)/G_max + 1 )

    with ``freq_fixed = 0.5``.  Sinusoid 1 starts at the top of its
    range (``F = 1.0`` when ``sin(·) = 1`` and the envelope is
    near-1) and decays its amplitude over the search.

  * **Sinusoid 2** (variable frequency, *increasing* envelope)::

        F = 0.5 · ( sin(2π · freq_i · g + π) · g/G_max + 1 )

    with ``freq_i ~ Cauchy(μ_freq, 0.1)`` clamped to ``(0, 1]``.
    Sinusoid 2 starts small and grows its amplitude over the search;
    the ``+π`` phase shift puts it in opposite phase to Sinusoid 1
    when ``freq_i = freq_fixed``.  ``μ_freq`` adapts each generation
    via the *unweighted* Lehmer mean
    (``Σ freq² / Σ freq``) of successful Sinusoid-2 frequencies.

  Selection between the two sinusoids is controlled by ``p_s``, the
  probability of picking Sinusoid 1, updated each generation from a
  *Laplace-smoothed* Sinusoid-1 success rate::

        p_s = (ns_1 + 1) / (ns_1 + ns_2 + 2)

  — same monotonic direction as the paper's ranking-selection formula,
  smaller state, identical behaviour in the corners that motivated the
  smoothing in the first place (no successes ⇒ ``p_s = 0.5``).  In the
  second half of the search (``progress ≥ 0.5``) the heuristic reverts
  to the standard SHADE Cauchy-from-memory ``F`` sampling — byte-
  identical to L-SHADE.  ``CR`` is *always* drawn from a SHADE Normal
  memory bin (unchanged from L-SHADE in both phases) — only ``F``
  switches mechanisms across the phase split.

  Two small behaviour-preserving hooks were added to L-SHADE to enable
  the subclass cleanly (mirroring the NL-SHADE-RSP precedent):

  * :meth:`LSHADE._make_trial_meta` — factory for the ``_pending``
    record.  Default returns a plain :class:`_TrialMeta`; EpSin
    overrides to return :class:`_EpSinTrialMeta` carrying the sin
    choice + freq used by the trial.
  * :meth:`LSHADE._record_success` — hook invoked once per successful
    competitive trial after the parent's SHADE memory update.  Default
    is a no-op; EpSin counts ``ns_1`` / ``ns_2`` and stashes the
    Sinusoid-2 ``freq`` for the end-of-generation Lehmer mean.

  L-SHADE, jSO, and NL-SHADE-RSP keep their byte-identical behaviour —
  the hooks' default implementations reproduce the prior code path
  exactly (verified: all 133 pre-existing L-SHADE / jSO / NL-SHADE-RSP
  tests pass unchanged).

* **Why** — closes the *L-SHADE-cnEpSin* DE-family follow-up below
  (the §13 entry from 2026-05-15 jSO ship lists EpSin under "Next
  iteration ideas" as a different *branch* of the DE family tree from
  jSO).  All DE arms shipped to date — basic DE, L-SHADE (CEC 2014),
  jSO (CEC 2017), NL-SHADE-RSP (CEC 2021) — adapt ``F`` via the SHADE
  *Cauchy memory*.  LSHADE-EpSin's deterministic-amplitude sinusoid is
  algorithmically distinct: it produces ``F`` values from a
  *time-varying deterministic schedule* rather than from a noisy
  memory-based posterior.  The two adaptation mechanisms have
  complementary strengths — Cauchy-memory tracks per-problem optimal
  ``F`` when the landscape has a clear "best ``F``" attractor; sinusoid
  schedules force ``F`` variability in both magnitude and direction
  regardless of landscape, which helps on landscapes where any single
  ``F`` posterior gets stuck.  Adds a *fifth* DE-family arm the bandit
  can pick whichever wins on the current battery.  Direct precursor of
  the CEC-2017 co-winner LSHADE-cnEpSin (the same sinusoidal ensemble
  plus a covariance-matrix mutation step — not ported here; CMA-ES is
  already a separate heuristic in Panobbgo).
* **Deviations from the paper** — for honesty (the Panobbgo norm is
  literature-faithful ports): three small deviations needed for the
  async pipeline:

  * **Generation-budget estimate.**  The paper uses the canonical
    synchronous generation count ``g`` and a known ``G_max`` (the
    total generations the loop will run).  Our async port has neither
    exactly — generations complete by count rather than by sync
    barrier, and ``G_max`` is unknowable until ``max_eval`` is
    reached.  We estimate ``G_max ≈ max_eval / ((NP_init + NP_min) / 2)``
    (average population size under LPSR) and gate the phase split on
    ``progress = len(results) / max_eval`` rather than ``g / G_max``.
    This keeps the schedule in lock-step with how L-SHADE already
    paces LPSR.  Unknown-budget fallback: ``G_max = 10 · NP_init``,
    ``sinusoidal phase`` always (so the heuristic still produces a
    varied ``F`` distribution).
  * **Selection-probability formula.**  The paper uses a more elaborate
    ranking-selection formula incorporating both success counts
    (``ns_1``, ``ns_2``) and failure counts (``nf_1``, ``nf_2``).  We
    use the simpler Laplace-smoothed
    ``p_s = (ns_1 + 1) / (ns_1 + ns_2 + 2)`` — same monotonic
    direction, smaller state, identical behaviour in the
    ``ns_1 = ns_2 = 0`` and ``ns_2 = 0`` corners that motivated the
    smoothing.
  * **F-cap is opt-in.**  The sinusoidal envelopes already provide a
    time-varying ``F`` magnitude; composing them with the jSO
    asymmetric F-cap (Brest 2017) is usually counter-productive in
    the first half.  The default is ``F_schedule=None`` (off);
    callers who want the cap can set ``F_schedule=True`` explicitly.
* **Impact** — the point of shipping is to give the bandit a fifth
  DE-family arm with markedly different ``F``-adaptation dynamics to
  choose between, rather than to claim a single-shipped-variant win.
  The §13 entries for L-SHADE, jSO, and NL-SHADE-RSP all report the
  same pattern: the CEC-DE refinements are *large-budget specialists*
  and at Panobbgo's small composite-battery budgets (75–500 evals)
  they measure within noise of each other on a single A/B.  The value
  of shipping today is to expand the bandit's catalog with a
  literature-grounded *F*-adaptation variant that is algorithmically
  distinct from every other arm shipped so far; the per-arm reward
  signal will identify the winner online once enough nights of the
  cron have accumulated.  *Evidence form (per AGENTS.md "Agent-driven
  improve X PRs"): change is backwards-compatible — the composite
  baseline on every default battery is byte-identical because
  LSHADE_EpSin is opt-in via the structural catalog and not added to
  any default ``_make_quick_strategies`` / ``_make_standard_strategies``
  / ``_make_full_strategies`` spec.*
* **Backwards compatibility** — strictly safe.  LSHADE-EpSin is opt-in:
  it is not added to any default :func:`_make_quick_strategies` /
  :func:`_make_standard_strategies` / :func:`_make_full_strategies`
  spec, so the composite baseline on every default battery is
  byte-identical and existing ledgers stay valid.  The structural
  catalog gains it as one extra ``add_heuristic`` candidate
  (``avoid_duplicates=True``).  The two kwarg rules
  (``LSHADE_EpSin.NP_init``, ``LSHADE_EpSin.mu_freq_init``) fire only
  when a spec sets the matching kwarg explicitly.  The L-SHADE
  base-class hook additions (:meth:`_make_trial_meta`,
  :meth:`_record_success`) are behaviour-preserving: their default
  implementations reproduce the prior code path exactly — all 133
  pre-existing L-SHADE / jSO / NL-SHADE-RSP tests pass unchanged.
* **Tests** — `tests/test_heuristic_lshade_ep_sin.py` (44 tests):
  construction validation (defaults, custom kwargs, subclass invariant,
  invalid ``mu_freq_init``, inherited L-SHADE validation rules);
  phase split (gate at ``progress < 0.5``, unknown-budget fallback to
  sinusoidal, ``G_max`` estimate + fallback); sinusoidal sampling
  (Sinusoid 1 / 2 returns ``F ∈ [0, 1]``, envelope behaviour at
  endpoints, ``freq`` Cauchy clamping, ``sin_choice ∈ {1, 2}``,
  ``CR`` sampling unchanged, phase-routed ``_sample_F_CR``, balanced
  cold-start selection); ensemble update (cold-start ``p_s = 0.5``,
  bias toward winning sinusoid, ``p_s`` strictly in ``(0, 1)``,
  Lehmer-mean ``μ_freq`` update, ``μ_freq`` untouched without
  Sinusoid-2 successes, counters cleared, ``_end_of_generation`` bumps
  ``_gen_count``); trial meta (sticky-reset ``_last_sin``,
  ``_EpSinTrialMeta`` carries sin choice, ``_record_success`` routes
  ``ns_1`` / ``ns_2`` / ``_gen_success_freq`` correctly, defensive on
  plain ``_TrialMeta``, no-op in Cauchy phase); pipeline (``on_start``
  emits ``NP_init``, resets ensemble state, initial fills carry
  ``sin_choice = 0``, evolutionary trials, sinusoidal success
  registered when better trial wins, restart resets state, end-to-end
  smoke convergence on a quadratic); base-class hook safety (L-SHADE
  / jSO / NL-SHADE-RSP all return plain ``_TrialMeta`` from
  ``_make_trial_meta``, ``_record_success`` is a no-op); and
  registration (package re-export + ``__all__``, structural catalog
  membership, kwarg catalog dials).
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *L-SHADE-cnEpSin* next-iteration idea promoted to "shipped
    (LSHADE-EpSin precursor; cnEpSin adds a CMA-style step on top —
    CMA-ES is already a separate Panobbgo heuristic)".
  - `doc/source/heuristics.rst`: new ``LSHADE_EpSin`` bullet; the
    DE-family complementarity bullet now names all five arms.
  - `doc/source/guide_architecture.rst`: new ``LSHADE_EpSin``
    description after NL-SHADE-RSP.
  - `doc/source/guide_benchmarking.rst`: structural-catalog candidate
    pool lists ``LSHADE_EpSin``; the description of the DE-family
    portfolio names all five arms.
  - `doc/source/guide.rst`: quick-nav entry mentions LSHADE-EpSin and
    the sinusoidal-F branch of the DE family tree.

### 2026-06-02 — Analyzer add/drop structural mutations

* **What** — `panobbgo/self_improve.py`:
  :class:`StructuralMutationRule` gains two new ops —
  ``"add_analyzer"`` and ``"drop_analyzer"`` — that mirror the
  existing ``add_heuristic`` / ``drop_heuristic`` semantics on the
  :attr:`StrategySpec.analyzers` bucket instead of ``heuristics``.  A
  sibling :attr:`StructuralMutationRule.min_analyzers` field (default
  ``0``) replaces :attr:`min_heuristics` as the post-drop safety floor
  for analyzer ops — analyzers are non-essential (unlike heuristics, a
  spec with an empty analyzers list is perfectly runnable), so the
  natural floor is *no analyzers required at all*.

  :func:`_find_structural_hits` consults the matching bucket
  (``spec.analyzers`` vs ``spec.heuristics``) based on the rule's op,
  reusing the existing ``avoid_duplicates`` / ``droppable_classes`` /
  ``strategy_pattern`` filters byte-identically.
  :func:`_make_structural_proposal` reuses the same
  :class:`MutationProposal` shape — analyzer ops differ only in the
  ``op`` / ``rule_kind`` strings.  :func:`apply_mutation` dispatches
  on ``proposal.op`` to either heuristic or analyzer branch; the new
  ``add_analyzer`` branch resolves the class object via the new
  :func:`_resolve_analyzer_class` helper (mirror of
  :func:`_resolve_heuristic_class`, but looks up against
  :mod:`panobbgo.analyzers` instead of :mod:`panobbgo.heuristics`).

  :func:`default_structural_catalog` gains two new
  :class:`StructuralMutationRule` instances — one ``add_analyzer``
  with a narrowly curated candidate pool (:class:`Sensitivity` with
  ``update_interval=20``; :class:`Restart` with the canonical
  IPOP-CMA-ES kwargs ``patience=None``, ``restart_strategy="diverse"``,
  ``max_restarts=5``) and one ``drop_analyzer`` with
  ``min_analyzers=0``.  Both carry the same low probability (``0.3``)
  as the heuristic ops, so the bandit samples structural mutations
  sparingly relative to kwarg retunes.  Per-class bandit arms
  (:attr:`AdaptiveMutationSampler.per_class_structural` shipped
  2026-05-18) work identically for the new ops — the existing
  :func:`_proposal_rule_key` logic checks membership in
  :data:`_STRUCTURAL_OPS` (now extended to include the analyzer ops),
  so ``("Restart", "add_analyzer", "structural")`` and
  ``("Sensitivity", "add_analyzer", "structural")`` are distinct
  per-class arms when the flag is on.
* **Why** — closes the *Analyzer add/drop* follow-up below the
  2026-05-03 structural-catalog entry.  Before this ship, the loop's
  reach into the strategy spec was asymmetric: it could change the
  *heuristics* portfolio (add Sobol' / drop NelderMead / etc.) but
  could not change the *analyzers* attached to a strategy, even
  though analyzers carry materially different behaviour — most
  conspicuously the :class:`Restart` analyzer's IPOP-style warm
  restarts, which the standard battery only uses on
  :func:`_make_standard_strategies`'s ``IPOP_CMAES`` /
  ``BIPOP_CMAES`` specs.  The loop could not discover, e.g., that
  attaching :class:`Restart` to a Rewarding strategy with a CMA-ES
  heuristic helps a particular battery — the analyzer slot was
  invisible to the bandit.

  Symmetrically, the loop could not learn that stripping
  :class:`Sensitivity` from a strategy that doesn't actually consume
  its outputs is a net win at quick budgets (Sensitivity's
  fixed-cost overhead, however small, eats into the eval budget).

  Adding analyzer ops closes the gap with a single self-contained
  piece of infrastructure that extends the bandit's reach by two
  ops at once (one add, one drop) without disturbing any existing
  ledger or behaviour.  This pairs naturally with the
  *Strategy-class swap* follow-up below — together those two would
  bring all three architectural axes of a :class:`StrategySpec`
  (``strategy_class`` / ``heuristics`` / ``analyzers``) under the
  loop's autonomous control.
* **Backwards compatibility** — strictly safe.  The two new
  ``_STRUCTURAL_OPS`` strings are additive — existing catalog code
  (validators, hit enumerators, proposal serialisers) treats them as
  uniformly as the heuristic ops.  The new
  :attr:`StructuralMutationRule.min_analyzers` field defaults to
  ``0``; every existing :class:`StructuralMutationRule` construction
  in the codebase (and in user catalogs) keeps its prior behaviour
  bit-for-bit.  The default :func:`default_catalog` is unchanged
  (analyzer ops only land in :func:`default_structural_catalog`,
  which itself is opt-in via ``--structural``).  Every prior ledger
  record parses identically — the new ``rule_kind`` strings are just
  additional values an existing consumer may ignore.

  All 180 pre-existing :mod:`tests.test_self_improve` tests pass
  unchanged; the only edit was the
  :class:`TestDefaultStructuralCatalog.test_returns_catalog_with_structural_rules`
  expected ``ops`` set, which now contains the four ops instead of
  two.
* **Cost** — zero at sample time when no spec has analyzers
  (``_find_structural_hits`` returns empty and the catalog skips the
  rule).  When the rule fires, the cost is a single list append /
  pop in :func:`apply_mutation`, identical to the heuristic path.
  The two analyzer rules in :func:`default_structural_catalog` add
  ~20 µs to catalog construction (two extra
  :class:`StructuralMutationRule` instances) — negligible relative
  to the loop's per-iteration harness cost.
* **Tests** — `tests/test_self_improve.py` (+34 new tests, total
  214): rule validation (5 — defaults, drop-without-candidates,
  add-requires-candidates, negative ``min_analyzers``, zero floor
  allowed); structural-hit enumeration (6 — avoid-duplicates,
  no-avoid-duplicates, drop floor=1 forbids strip, drop floor=0
  allows strip, droppable_classes filter, strategy_pattern filter);
  catalog sampling (4 — add proposal shape, drop proposal shape,
  unapplicable returns ``None``, default-kwargs-independent-per-hit);
  apply-side dispatch (7 — add appends to analyzers bucket, add
  falls back to package, add unknown class raises, drop removes,
  drop allows empty result, drop missing class raises, drop
  preserves heuristics-bucket independence); per-class bandit arms
  (5 — proposal_rule_key collapse, per-class key layout, sampler
  default collapse, sampler buckets per class with the flag, total
  attempts conserved); proposal serialisation (2 — add round-trip,
  drop round-trip); default catalog (4 — includes analyzer ops,
  candidate pool contents, drop floor is 0, applicable on the
  standard quick-mode battery); end-to-end (1 — SelfImprover
  accepts a drop_analyzer mutation that improves the score).
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Analyzer add/drop* follow-up below the 2026-05-03 entry
    promoted from "open" to "shipped".
  - `doc/source/guide_benchmarking.rst`: the structural-catalog
    section now documents all four ops; the Thompson-sampler
    paragraph and the per-class-arms subsection both name the
    analyzer ops.
  - `doc/source/guide.rst`: quick-nav entry mentions
    ``add_analyzer`` / ``drop_analyzer``.
  - `AGENTS.md`: structural composition subsection and the
    run-the-loop bash example reference the analyzer ops.

### 2026-06-01 — Hierarchical bandit over per-class structural arms

* **What** — `panobbgo/self_improve.py`:
  :class:`AdaptiveMutationSampler` gains a
  ``structural_borrow_alpha: float = 0.0`` constructor argument
  (a borrow coefficient ``κ ≥ 0``).  When ``κ > 0`` and
  :attr:`per_class_structural` is also ``True``, each per-class
  structural arm's Beta posterior is built as::

      Beta(prior_alpha + n_class_accepts  + κ · n_other_class_accepts,
           prior_beta  + n_class_failures + κ · n_other_class_failures)

  where the *"other class"* aggregates are the sum across every
  *sibling* per-class arm sharing the same structural op
  (``add_heuristic`` or ``drop_heuristic``).  The self-exclusion is
  deliberate — borrowing from one's own evidence would collapse the
  hierarchy to a κ-amplified version of the same per-class posterior
  rather than a meaningful share-strength prior.  Op-level aggregates
  are computed on-the-fly per :meth:`sample` call (linear in the
  number of stored stats, no separate accumulator dict), so
  :meth:`record_outcome` and :meth:`prime_from_ledger` are unchanged.
  :class:`LoopConfig` gains
  ``structural_borrow_alpha: float = 0.0`` with matching validation
  (``>= 0``), and :class:`SelfImprover` forwards it to the sampler
  whenever the adaptive path is used.  ``scripts/self_improve.py``
  gains a ``--structural-borrow-alpha`` CLI flag (only effective with
  both ``--adaptive`` and ``--structural-per-class-arms``).
* **Why** — closes the *Hierarchical bandit over the per-class
  structural arms* follow-up below the 2026-05-18 §13 entry.  Per-class
  arms shipped 2026-05-18 traded sample efficiency for sharper
  signal: with ``N`` candidate classes the bandit divides its
  evidence by ~``N`` and each arm starts cold-start with the
  symmetric ``Beta(1, 1)`` prior, even when its op-level sibling
  history is strongly informative.  The hierarchical
  Beta-Binomial recovers the data-sharing of the wildcard arm while
  preserving the per-class arg-max — exactly the design sketch in
  the planning doc.  Critically relevant given the current loop
  productivity (~5% accept rate over 366 iterations on the latest
  ledger): the per-class arms divide an already-small accept count,
  and a borrow coefficient lets a fresh candidate class start at the
  op's empirical accept rate rather than the cold prior.
* **Borrow coefficient choice** — ``κ = 0`` (default) preserves the
  pure per-class semantics shipped 2026-05-18; ``κ = 1`` weights
  every sibling accept equally with the class's own.  A useful
  intermediate is ``κ = 0.5`` (half-weighted sibling evidence),
  empirically robust in hierarchical-bandit literature when there is
  real but imperfect transfer between arms.  The new
  ``--structural-borrow-alpha`` CLI flag accepts any non-negative
  float; the rationale field on each :class:`MutationProposal`
  reports the effective ``Beta(α, β)`` so ledger auditors can verify
  the borrow at any iteration.
* **Backwards compatibility** — strictly safe.  Default
  ``structural_borrow_alpha = 0.0`` makes :meth:`sample` byte-identical
  to the 2026-05-18 ship; under any existing CLI invocation or
  programmatic call the new code path is dead.  All 180 pre-existing
  tests in ``tests/test_self_improve.py`` pass unchanged.  When the
  flag is on, :meth:`prime_from_ledger` and :meth:`record_outcome`
  use the same per-class key layout as before — the borrow is
  computed at draw time from the existing stats dict, so resuming
  with ``--adaptive-prime-from-ledger`` recovers identical bandit
  state.  Kwarg perturbation arms are unaffected regardless of
  ``κ`` (they have no op-level aggregate to borrow from).  When
  ``per_class_structural`` is ``False`` the borrow is silently inert
  (no per-class arms exist for the hierarchy to operate over);
  similarly when ``--adaptive`` is not set, no sampler is constructed
  and the knob is dead code.
* **Tests** — `tests/test_self_improve.py` (+14 tests, total 208):
  default ``structural_borrow_alpha=0.0`` on the sampler; negative
  / non-finite ``κ`` raises; κ=0 produces a byte-identical sample
  trajectory to the unhierarchical per-class sampler (same RNG
  seed, same proposals); borrow inert when
  ``per_class_structural=False`` (κ=10 vs no borrow trajectory
  match); borrow inert for kwarg rules (κ=1 vs no borrow on the
  kwarg-only catalog trajectory match); fresh class warms with op
  aggregate (X seeded 20/20, Y picked >25% of the time under κ=1 vs
  <20% under κ=0); self-exclusion verified via Beta(α, β) values in
  the rationale (X seeded 10/10 sees Beta(11, 1), Y sees Beta(6, 1)
  under κ=0.5); mixed failure/accept borrow (X seeded 3/10 makes
  Y's posterior Beta(4, 8) under κ=1); ``LoopConfig`` default 0.0;
  validation rejects negative; flag propagates through
  :class:`SelfImprover`; inert without ``adaptive_sampling``.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Hierarchical bandit over the per-class structural arms*
    follow-up below the 2026-05-18 entry promoted from "open" to
    "shipped".
  - `doc/source/guide_benchmarking.rst`: new
    "Hierarchical bandit over per-class structural arms" subsection
    under "Adaptive (Thompson-sampling) mutation sampler" with the
    Beta-Binomial formula, CLI example, programmatic example, and
    the borrow-coefficient guidance.
  - `doc/source/guide.rst`: quick-nav entry mentions the new
    ``structural_borrow_alpha`` coefficient.
  - `AGENTS.md`: self-improvement loop subsection lists the new
    feature with a run-the-loop bash example.

### 2026-05-31 — Codify `Sobol.scramble=False` in `Rewarding_Diverse` (first ledger-evidence-driven default change)

* **What** — `panobbgo/harness.py` :func:`_make_quick_strategies` now
  ships ``Rewarding_Diverse`` with ``(Sobol, {"n": 16, "scramble":
  False})`` instead of ``scramble=True``.  This is the **first
  application of the planning doc §12.3 step 2 codification rule** —
  "if a rule keeps winning, change the default" — driven by
  three independent positive accepts in the archived ledger
  (``planning/done/self_improve_ledger_2026-05-31.jsonl`` iter 9 / 15
  / 17 in the 2026-05 ledger window):

      iter=9   Δ=+0.0511  CI=[+0.0089, +0.0933]  worst=+0.0000
      iter=15  Δ=+0.0217  CI=[+0.0056, +0.0433]  worst=+0.0000
      iter=17  Δ=+0.0317  CI=[+0.0050, +0.0583]  worst=+0.0000

  Every accept had its bootstrap-CI lower bound strictly above zero
  and zero per-pair regression — clean wins under the §6.2 statistical
  rule.  All three accepts proposed ``True → False`` (the catalog
  rule always excludes the current value), so the data is consistent
  about the direction.  The ``Sobol.scramble`` ``categorical_choice``
  rule (shipped 2026-05-13) still applies to the codified spec: it
  now proposes ``False → True``, so the bandit is free to flip back
  if a future battery prefers the scrambled regime.
  ``BayesOpt_Sobol`` (a standard-mode strategy the quick-mode cron
  never exercises) keeps ``scramble=True`` — there is no ledger
  evidence on that strategy yet, so the conservative move is to leave
  it alone and let the bandit explore.  ``panobbgo/harness_ioh.py``
  is similarly untouched (the IOH track shipped 2026-05 with
  ``scramble=True``; codification waits on IOH-specific evidence).
  The archived ledger is preserved at
  ``planning/done/self_improve_ledger_2026-05-31.jsonl`` so the
  bandit can prime from a clean slate on the next nightly run; the
  archived summary lives at
  ``planning/done/self_improve_summary_2026-05-31.txt``.
* **Why** — The nightly loop has been re-discovering this same
  improvement on every run and then throwing it away when the in-
  memory ladder dies at end-of-loop (the cron persists evidence, not
  source edits — see §12.2).  Codifying the win permanently lifts the
  quick-mode composite baseline by the same ~+0.035 the loop kept
  measuring, freeing the bandit to spend future cycles on other
  rules.  The change also closes the loop on the original §11 success
  criterion ("a sustained positive trend means the framework really
  got better") for the first time end-to-end: measurement → repeated
  accept → human review → codification → archive → re-baseline.
* **Why ``scramble=False`` beats ``True`` at quick mode (literature
  reasoning consistent with the empirical signal)** — At ``n=16`` in
  the quick-mode 2-D battery, the deterministic Sobol' sequence
  places its first 16 points at fixed, provably space-filling
  locations of the unit hypercube (the digit-shifted construction is
  *exactly* a low-discrepancy net at ``n = 2^k``).  Owen scrambling
  preserves the equidistribution property *in expectation* but
  perturbs the specific positions — at small ``n`` the variance this
  introduces in coverage quality dominates the gain from breaking
  axis-aligned correlations.  The downstream local heuristics
  (Random, Nearby, NelderMead) all start from those Sobol' points,
  so a more uniform "first looks" grid pays compound returns.  At
  larger ``n`` (BayesOpt_Sobol ships ``n=16`` in 5-D / standard
  mode, where Owen scrambling's projection guarantees matter more)
  the trade-off may flip — which is exactly why the catalog rule
  stays live.
* **Why archive the ledger** — The categorical rule's bandit arm
  key is ``("Sobol", "scramble", "categorical_choice")``, which does
  not distinguish proposal direction.  After the codification, every
  fresh proposal on ``Rewarding_Diverse`` flips ``False → True``;
  if the new bandit primed from the archived ledger, its Beta
  posterior would carry stale "True → False good" history into a
  "False → True ?" sampling regime.  Archiving the ledger and
  letting the next nightly cron rebuild the posterior on the post-
  codification accept stream keeps the bandit's beliefs honest, per
  §12.3 step 5.
* **Impact** — Expected +~0.03 to +~0.05 composite on the
  ``Rewarding_Diverse`` arm of the standard quick-mode battery,
  matching the three observed accept deltas.  Because the composite
  averages over the two quick-mode strategies (``RoundRobin_Random``
  unaffected, ``Rewarding_Diverse`` lifted), the all-strategy
  composite gains roughly half of that.  The historical ledger
  (in ``planning/done/``) is not directly comparable to post-
  codification ledgers — see the archive note above and §12.3 step
  5.
* **Backwards compatibility** — Strictly safe at the heuristic
  level: :class:`panobbgo.heuristics.sobol.Sobol` still defaults to
  ``scramble=True`` (the literature default), and only the
  ``Rewarding_Diverse`` spec in :func:`_make_quick_strategies`
  changes.  ``BayesOpt_Sobol``, ``harness_ioh.py``, and every other
  call site remain bit-for-bit identical.  Tests that construct
  Sobol directly with explicit kwargs are unaffected.
  ``BenchmarkHarness.composite_score`` on the historical seed=42
  baseline shifts up by the codified margin — see the *historical
  baseline shift* note under §11.
* **Tests** — No new tests required.  The :class:`Sobol` class's
  unit tests are construction-level and pass arguments explicitly.
  The composite-score round-trip tests in
  ``tests/test_harness.py`` are seed-deterministic but do not pin
  the composite *value*, only the schema and reproducibility.  Full
  pytest suite still passes (~1100+ tests).
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; *Next
    iteration ideas* §12.3 step 2 example refreshed.
  - `doc/source/guide_benchmarking.rst`: codification callout under
    the §12.3 daily-routine description.
  - `panobbgo/harness.py`: ``_make_quick_strategies`` docstring
    cites the codification.
  - `panobbgo/self_improve.py`: catalog rule comment refreshed.

### 2026-05-30 — Inactivity-guarded ``eps_accept`` relaxation

* **What** — `panobbgo/self_improve.py`: :class:`LoopConfig` gains
  three knobs — :attr:`~LoopConfig.inactivity_relax_after` (default
  ``0`` = disabled), :attr:`~LoopConfig.inactivity_relax_factor`
  (default ``0.5``) and :attr:`~LoopConfig.inactivity_min_eps_accept`
  (default ``0.001``).  When enabled, the loop's accept gate decays
  the configured :attr:`~LoopConfig.eps_accept` geometrically by
  ``factor`` for every additional ``after``-block of consecutive
  non-accepts, floored at ``min_eps_accept``.  The decay resets to
  the configured ``eps_accept`` on the next accept.  Both
  *skip*-iterations (no applicable mutation) and *reject*-iterations
  contribute to the streak — the bandit cares about observed
  accepts, not how the loop got there.  A new helper
  :meth:`LoopConfig.effective_eps_accept` computes the threshold for
  any streak length.  Two fields land on
  :class:`LoopIterationRecord`:
  :attr:`~LoopIterationRecord.effective_eps_accept` (the threshold
  :func:`~panobbgo.harness.statistical_accept` actually saw) and
  :attr:`~LoopIterationRecord.iters_since_accept` (the streak length
  consulted to compute it).  Both default to ``None`` on legacy
  records so the JSONL load path keeps working.  CLI:
  ``scripts/self_improve.py run --inactivity-relax-after 10
  --inactivity-relax-factor 0.5 --inactivity-min-eps-accept 0.001``.
* **Why** — closes the *Inactivity-guarded loop productivity*
  follow-up in "Next iteration ideas".  The most recent unattended
  ledger (``planning/self_improve_summary.txt``) records *15 accepts
  in 326 decided iterations (4.6 %)*; one of the earlier nightly
  windows produced 1 accept in 86 iterations (~1.2 %).  At those
  accept rates the Thompson sampler's Beta posteriors barely move
  off the prior, so the *point* of having an adaptive sampler is
  defeated.  A geometric relaxation gives the loop a principled way
  to "lower the bar a little after a long drought" without
  permanently moving the bar — the decay resets the moment a real
  accept lands.  The floor keeps a relaxed accept above the
  bootstrap CI's noise floor; the per-iteration ledger fields
  keep the rule auditable.
* **Algorithm** — :func:`LoopConfig.effective_eps_accept` returns
  ``max(eps_accept · factor^(s // after), min_eps_accept)`` where
  ``s`` is the streak length.  Examples:

  * ``eps_accept=0.005, after=10, factor=0.5, min=0.001``: streak
    0 → 0.005, streak 10 → 0.0025, streak 20 → 0.00125, streak 30
    → 0.001 (floor), all subsequent streaks stay at 0.001.
  * ``after=0`` (disabled): constant ``eps_accept`` regardless of
    streak — byte-identical to the historical behaviour.
* **Validation** — ``inactivity_relax_after >= 0``; when
  positive, ``0 < factor < 1`` (``1.0`` doesn't relax, ``> 1``
  would amplify — both pointless) and
  ``0 <= min_eps_accept <= eps_accept`` (a floor above the
  configured threshold would be a no-op or worse).
* **Backwards compatibility** — strictly safe.  The defaults
  (``after=0``, ``factor=0.5``, ``min=0.001``) leave the loop's
  accept gate byte-identical to the prior behaviour: when
  ``after = 0`` the relaxation helper short-circuits to a constant
  ``eps_accept`` and the loop passes the same value to
  :func:`statistical_accept` as before.  Legacy ledger records that
  pre-date the two new :class:`LoopIterationRecord` fields load
  with ``None`` defaults; existing reader code paths (the CLI
  summary, hold-out replays, ``aggregate_holdout_drift``) never
  reference the new fields, so they continue to work unchanged.
  The ledger's JSONL schema is purely additive: old consumers can
  ignore the new keys, new consumers can rely on their presence on
  records written by the 2026-05-30 ship or later.
* **Impact** — closes the documented productivity bottleneck.  At
  4.6 % accept rate over 326 iterations, halving the threshold
  after a drought of 10 lets the loop reach for borderline
  improvements (delta between 0.0025 and 0.005) that the
  paired-bootstrap CI rules in as statistically distinguishable
  from zero — exactly the regime where the historical
  ``eps_accept = 0.005`` point-gate was leaving signal on the
  floor.  The Beta posteriors update sooner, so the bandit
  identifies its winning arms faster, which compounds across
  later iterations.  *Evidence form (per AGENTS.md "Agent-driven
  improve X PRs"): inspect-by-construction (the geometric decay's
  end states are exact and tested); queued for nightly loop
  validation via the cron — opt in by adding
  ``--inactivity-relax-after 10`` to the workflow's run-command.*
* **Tests** — `tests/test_self_improve.py` (+15 tests, total 210):

  * :class:`TestInactivityRelaxConfig` (8 tests) — disabled by
    default, validation errors on negative ``after`` / out-of-range
    ``factor`` / negative-or-too-large floor; threshold maths for
    no-relax-before-threshold, geometric decay across steps, floor
    clamping past the floor.
  * :class:`TestInactivityRelaxIntegration` (7 tests) —
    records carry the effective threshold and streak; streak
    resets on accept; skip-iterations count toward the streak; a
    borderline +0.04 delta that the configured 0.05 gate rejects
    is accepted by the relaxed 0.025 gate after one decay step
    (and is rejected again on the iteration following the accept,
    confirming the reset); disabled mode populates the fields with
    the constant ``eps_accept``; ledger round-trip preserves both
    new fields; legacy records construct cleanly with ``None``
    for both fields.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Inactivity-guarded loop productivity* next-iteration idea
    promoted to "shipped (eps_accept relaxation)"; the unshipped
    half (*Bump the harness mode for the cron*) explicitly left
    open under the same heading.  A new follow-up
    *Inactivity-relax telemetry in summary view* left for the next
    iteration.
  - `doc/source/guide_benchmarking.rst`: new
    *Inactivity-guarded eps_accept relaxation* subsection under
    the loop-driver writeup, with the three-knob description,
    the geometric-decay maths, the recommended unattended preset,
    and the §11 honesty rationale (floor + per-iteration ledger
    fields).
  - `doc/source/guide.rst`: quick-nav entry mentions the new
    relaxation knob.
  - `AGENTS.md`: brief note pointing to the new feature for the
    nightly cron operators.

### 2026-05-29 — Random PSO topology (Mendes 2004 / Clerc 2007 / SPSO 2011)

* **What** — `panobbgo/heuristics/pso.py`: :class:`PSO` gains a fourth
  shipped topology, ``"random"``, via two new helpers
  :meth:`_init_random_adjacency` (samples one informer set per particle:
  ``k_neighbors`` draws *with replacement* from ``{0..NP-1} \ {i}`` plus
  the particle itself, dedup'd so the realised neighbourhood lies in
  ``[2, k_neighbors + 1]``) and :meth:`_random_neighbors` (lookup helper
  that falls back to ``[i]`` when ``on_start`` has not run yet).
  :meth:`_social_best_idx` dispatches the new topology onto the same
  scan-for-best-neighbour-pbest routine already used by ``lbest`` and
  ``vonneumann``.  Adjacency is built at :meth:`on_start` and re-sampled
  at :meth:`on_restart` — the Clerc 2007 / SPSO 2011 convention: when
  the swarm loses cohesion, the social network is rebuilt to break
  stagnation.  :func:`default_structural_catalog` gains a fourth PSO
  entry — ``(PSO, {"NP": 20, "topology": "random", "k_neighbors": 3})``
  — alongside the existing ``gbest`` / ``lbest`` / ``vonneumann``
  entries.  All four share ``cls = PSO`` so ``avoid_duplicates=True``
  still prevents multiple PSO instances per strategy.  The default
  catalog's existing ``PSO.topology`` categorical rule grows from
  three choices to four (``("gbest", "lbest", "vonneumann",
  "random")``) so the bandit can flip an existing explicit-topology
  PSO between all four regimes without dropping and re-adding the
  heuristic.
* **Why** — closes the *Random re-wired topology* PSO follow-up under
  the §13 entry from 2026-05-22.  ``gbest`` / ``lbest`` / ``vonneumann``
  are all closed-form functions of ``NP`` — instantaneous full-connect,
  one-hop ring, two-hop planar.  The fourth slot in the canonical
  Mendes 2004 set is the *random* graph: structure-free, asymmetric
  (``j ∈ informers(i)`` does not imply ``i ∈ informers(j)``), with
  diffusion speed determined by the realised graph rather than a
  fixed geometric prior.  Clerc (2007) standardises this as the SPSO
  2007 / 2011 default with ``K = 3`` informers per particle drawn
  uniformly with replacement; we match that convention in the
  structural-catalog entry.  Useful when the bandit evidence shows
  neither pure structured topology consistently wins on a given
  battery — the random graph picks up some of the flexibility of all
  three without committing to a structural prior.
* **Asymmetric adjacency** — unlike ``lbest`` (symmetric ring) and
  ``vonneumann`` (symmetric grid), the random topology is
  *asymmetric*: an informer relationship is one-way.  This matches
  the Mendes 2004 / SPSO 2011 convention and is what gives the
  topology its structure-free character.  The test suite verifies
  asymmetry on a representative seed (``NP=20, k=2, seed=0``).
* **Index-shift logic** — draws come from ``rng.integers(0, NP-1, k)``
  then shift past ``i`` (``p if p < i else p + 1``) so the informer
  pool deterministically excludes self.  Verified across 50 seeds:
  every particle's own index appears in its informer list *exactly
  once* — added by :meth:`_init_random_adjacency`, never re-injected
  by a self-collision in the draws.
* **Restart re-sampling** — the Clerc 2007 stagnation-rebuild
  convention: a restart re-samples the entire informer graph from the
  heuristic's RNG.  Verified by an explicit before/after test
  (``NP=15, k=3, seed=99``): the deterministic RNG plus the distinct
  re-init call changes at least one row of the adjacency matrix (the
  probability of all 15 rows reproducing exactly is vanishingly
  small).
* **Impact** — the point of shipping today is to give the bandit a
  fourth PSO arm with markedly different exploration dynamics to
  choose between, rather than to claim a single-shipped-variant win.
  The 2026-05-07 ``lbest`` and 2026-05-22 ``vonneumann`` entries
  already established that no single PSO topology dominates at
  quick-mode noise levels (~ ±0.05) — seeds 42 and 43 split the win
  between ``gbest`` and ``lbest``.  The literature (Mendes 2004;
  Clerc 2007) predicts the random graph sits between the structured
  topologies in expected diffusion speed but with much higher
  variance — sometimes the realised graph is near-fully-connected,
  sometimes near-disconnected.  The measurable signal will
  materialise once the self-improvement loop has accumulated enough
  evidence from the bandit's per-arm reward history to identify
  which topology wins on the current battery.  *Evidence form (per
  AGENTS.md "Agent-driven improve X PRs"): catalog-only addition;
  backwards-compatible (composite baseline byte-identical, existing
  ledgers stay valid); queued for nightly loop validation via the
  structural catalog.*
* **Backwards compatibility** — strictly safe.  ``topology`` defaults
  to ``"gbest"``; every existing PSO instance retains its prior
  behaviour bit-for-bit, including the 56 pre-existing tests in
  ``tests/test_heuristic_pso.py``.  The structural catalog gains one
  extra ``add_heuristic`` candidate that shares ``cls = PSO`` with
  the existing entries; under ``avoid_duplicates=True`` (default),
  only one of the four is ever added per strategy.  The categorical
  rule expansion is also safe: callers passing the prior choices
  tuple get the same uniform-over-the-set draw (the cardinality just
  bumps from 3 to 4), and the rule still fires only when a spec
  sets ``topology`` explicitly.  Existing ledger consumers parsing
  the rule's ``choices`` field see one extra string they may ignore.
  The new ``_random_adjacency`` field is ``None`` for any topology
  other than ``"random"``, so memory / RNG draws on ``gbest`` /
  ``lbest`` / ``vonneumann`` paths are byte-identical.
* **Tests** — `tests/test_heuristic_pso.py` (+12 new tests, total
  80): random construction round-trip; adjacency built on start;
  every particle is its own informer (``i ∈ informers(i)`` for all
  ``i``); realised neighbourhood ≤ k+1 with no duplicates; self
  appears exactly once across 50 seeds (the index-shift logic
  excludes self from random draws); asymmetric graph in general
  (``forward != backward`` on ``NP=20, k=2, seed=0``); seed
  reproducibility (two PSOs sharing the same seed produce identical
  adjacency); adjacency re-sampled on restart (at least one row
  differs); social-best limited to informer set (planted-pbest
  invariant: ``_gbest_idx`` points at an outside-informer better
  pbest while ``_social_best_idx(0)`` returns the inside-informer
  worse pbest); none-until-evaluated; velocity clamp invariant under
  random topology; end-to-end smoke convergence on a quadratic;
  updated structural-catalog test confirming all four PSO topology
  variants appear among ``add_heuristic`` candidates; updated
  categorical-rule test confirming ``default_catalog`` now ships
  ``choices=("gbest", "lbest", "vonneumann", "random")``.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Random re-wired topology* PSO follow-up below the 2026-05-22
    entry promoted from "open" to "shipped".  A new follow-up
    *Stochastic-K random topology (per-iteration re-sampling)* left
    for the next iteration.
  - `doc/source/guide.rst`: quick-nav entry mentions the
    four-topology PSO candidate pool.
  - `doc/source/guide_benchmarking.rst`: structural-catalog section
    now describes the four PSO entries; the categorical-rules
    section lists ``random`` as a fourth ``PSO.topology`` value.
  - `doc/source/guide_architecture.rst`: PSO description gains the
    ``"random"`` topology paragraph after ``"vonneumann"``.
  - `doc/source/heuristics.rst`: PSO bullet expanded to the
    four-topology set; Mendes 2004 / Clerc 2007 citations added.
  - `AGENTS.md`: categorical-rules list adds ``"random"`` to
    ``PSO.topology`` (cardinality three → four).
  - `TODO.md`: new entry at the head of "Recent Improvements".
### 2026-05-28 — NL-SHADE-LBC adaptive DE (CEC 2022 winner)

* **What** — `panobbgo/heuristics/nl_shade_lbc.py` adds the
  :class:`NLSHADE_LBC` heuristic, a direct subclass of
  :class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP` (CEC 2021
  winner) that ports the Stanovov-Akhmedova-Semenkin (CEC 2022)
  "NL-SHADE-LBC" refinement.  NL-SHADE-LBC inherits the entire
  NL-SHADE-RSP / jSO / L-SHADE asynchronous pipeline (per-slot pending
  dict, generation-by-count book-keeping, archive of replaced parents,
  success-history memory with the frozen jSO anchor bin, weighted
  ``current-to-pbest-w/1`` mutation, linear ``p_best`` schedule,
  asymmetric F-cap, NLPSR, RSP r1 selection, randomised adaptive
  archive, warm restart) and adds **Linear Bias Change** in the
  memory update:

  The standard L-SHADE / jSO / NL-SHADE-RSP memory update uses a fixed
  Lehmer mean of order 2 with spread 1 (``Σ(w·s²) / Σ(w·s)``).
  NL-SHADE-LBC generalises this to::

      L_{p,m}(s, w) = Σ(w_i · s_i^p) / Σ(w_i · s_i^{p − m})

  with the **order** ``p`` linearly scheduled across budget progress
  ``r = len(strategy.results) / max_eval``::

      p_F(r)  = (1 − r) · p_F_init  + r · p_F_final
      p_CR(r) = (1 − r) · p_CR_init + r · p_CR_final

  Literature defaults from Stanovov et al. (2022) — verified against
  the MetaBox reference implementation: ``p_F_init = 3.5``,
  ``p_F_final = 1.5``, ``p_CR_init = 1.0``, ``p_CR_final = 1.5``,
  ``m_lbc = 1.5``.  The F-bias starts high (concentrating memory on
  the *largest* successful F's, encouraging exploration) and decays;
  the CR-bias starts low (preserving CR diversity) and grows.  At
  ``p = 2, m = 1`` the formula recovers the L-SHADE Lehmer mean — both
  regimes are reachable from the default catalog so the bandit can
  flip between them.

  CR-zero handling preserves the L-SHADE terminal sentinel rule and
  filters strict zeros out of the LBC sum (because ``s^(p − m)`` with
  ``p < m`` blows up at ``s = 0``).  Registered in
  :mod:`panobbgo.heuristics`; :func:`default_structural_catalog` gains
  it as a fifteenth ``add_heuristic`` candidate
  (``avoid_duplicates=True``); :func:`default_catalog` gains six rules
  — ``NLSHADE_LBC.NP_init`` (integer_add), ``NLSHADE_LBC.p_F_init``
  (float_uniform ``[1.5, 5.0]``), ``NLSHADE_LBC.p_F_final``
  (float_uniform ``[1.0, 3.0]``), ``NLSHADE_LBC.p_CR_init``
  (float_uniform ``[0.5, 2.5]``), ``NLSHADE_LBC.p_CR_final``
  (float_uniform ``[0.5, 2.5]``), and ``NLSHADE_LBC.m_lbc``
  (float_uniform ``[1.0, 2.0]``).
* **Why** — closes the *NL-SHADE-LBC* DE-family follow-up listed under
  the NL-SHADE-RSP entry above.  NL-SHADE-LBC won the **CEC-2022**
  single-objective bound-constrained competition and is the direct
  NL-SHADE-RSP descendant; it represents the literature frontier as of
  the most recent CEC competition we can mirror.  Subclassing
  NL-SHADE-RSP keeps the new heuristic at the literature frontier
  while leaving NL-SHADE-RSP / jSO / L-SHADE byte-identical for
  ledger reproducibility — the precedent set by the NL-SHADE-RSP entry
  itself.  Adds a fifth DE-family arm the bandit can pick whichever
  wins on the current battery.
* **Deviations from the full CEC-2022 paper** — for honesty (the
  Panobbgo norm is literature-faithful ports): two NL-SHADE-LBC
  mechanisms are intentionally **not** ported because they interact
  with the synchronous generation model in ways the asynchronous
  pipeline does not expose cleanly: the *adaptive binomial /
  exponential crossover blend* (also intentionally not ported from
  NL-SHADE-RSP — see the same caveat there), and the *repetitive
  generation* bound-constraint handling (Panobbgo uses
  ``strategy.constraint_handler`` and L-SHADE midpoint-reflection
  repair instead).  Both are queued as follow-ups below.
* **Impact** — the value of shipping this today is to give the
  self-improvement loop a CEC-2022-class DE arm the bandit can select
  once it has accumulated per-arm reward history.  Like NL-SHADE-RSP
  before it, the LBC refinements are **large-budget specialists**: at
  panobbgo's small composite-battery budgets (75–500 evals) the
  bias-change schedule barely warms up, so the quick-mode signal is
  expected within noise.  *Evidence form (per AGENTS.md "Agent-driven
  improve X PRs"): catalog-only addition; backwards-compatible
  (composite baseline byte-identical, existing ledgers stay valid);
  queued for nightly loop validation via the structural catalog.*
* **Backwards compatibility** — strictly safe.  NLSHADE_LBC is opt-in:
  it is not added to any default :func:`_make_quick_strategies` /
  :func:`_make_standard_strategies` / :func:`_make_full_strategies`
  spec, so the composite baseline on every default battery is
  byte-identical and existing ledgers stay valid.  The structural
  catalog gains it as one extra ``add_heuristic`` candidate
  (``avoid_duplicates=True``).  The kwarg rules fire only when a spec
  sets the matching kwarg explicitly.  NL-SHADE-RSP / jSO / L-SHADE
  are untouched — only the LBC subclass overrides
  :meth:`_update_memory`; the base classes' ``_update_memory`` methods
  are byte-identical, verified by a regression test that
  ``NLSHADE_RSP._update_memory`` still produces the standard L-SHADE
  Lehmer mean output.
* **Tests** — `tests/test_heuristic_nl_shade_lbc.py` (30 tests):
  construction validation (defaults, custom kwargs, subclass invariant
  spanning NLSHADE_RSP / JSO / LSHADE, invalid / inf / NaN p_F_init /
  p_F_final / p_CR_init / p_CR_final / m_lbc, m_lbc=0 and m_lbc<0
  rejection, inherited NLSHADE_RSP / jSO ``H >= 2`` / ``p_best``
  ordering / ``k_rank`` rules); LBC schedule (endpoints
  progress=0/progress=1, linear midpoint, clipping at progress > 1,
  fallback to p_init when budget unknown); memory update (no write to
  the anchor bin H-1, pointer advances ``% (H-1)``, no-op on empty
  buffer, F memory clamped to [0,1], LBC formula at progress=0 with
  custom exponents matches Σ(w·F^3.5)/Σ(w·F^2.0), p=2/m=1 recovers the
  standard L-SHADE Lehmer mean for *both* F and CR, CR=0 plants the
  terminal sentinel, terminal-bin stays terminal, mixed-zero CR values
  filtered before LBC computation, zero-delta successes fall back to
  uniform weights); pipeline (on_start emits NP_init, smoke
  convergence on a quadratic with no negative global progress, restart
  resets archive and pending); inheritance safety (NLSHADE_RSP
  ``_update_memory`` still produces standard L-SHADE mean); and
  registration (package re-export + ``__all__``, structural catalog
  membership, six kwarg catalog dials).
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *NL-SHADE-LBC* next-iteration idea promoted to "shipped".
  - `doc/source/heuristics.rst`: new ``NLSHADE_LBC`` bullet; the
    DE-family complementarity bullet now names all five arms.
  - `doc/source/guide_architecture.rst`: new ``NLSHADE_LBC``
    description after NLSHADE_RSP.
  - `doc/source/guide_benchmarking.rst`: structural-catalog candidate
    pool lists ``NLSHADE_LBC``; the DE-family complementarity blurb
    extends to five arms.
  - `doc/source/guide.rst`: quick-nav entry mentions NL-SHADE-LBC and
    the Linear Bias Change mechanism.
### 2026-05-27 — Multi-start L-BFGS-B gradient local optimizer (rescued + catalogued)

* **What** — Rewrote `panobbgo/heuristics/lbfgsb.py` from a one-shot,
  box-centre, restart-blind, **unreferenced** stub into a robust
  *multi-start* bound-constrained quasi-Newton local optimizer, and
  added it to :func:`default_structural_catalog`'s ``add_heuristic``
  candidate pool (the 15th candidate, ``avoid_duplicates=True``).  The
  worker now runs :func:`scipy.optimize.fmin_l_bfgs_b` **repeatedly** —
  the first descent from the box centre (deterministic / reproducible),
  every subsequent descent from a fresh uniform-random restart — using
  the entire strategy budget instead of going idle after the first
  convergence.  ``on_restart`` warm-starts the next descent at the
  Restart analyzer's centre (clipped into the box).  The subprocess
  lifecycle was re-modelled on the well-tested
  :class:`~panobbgo.heuristics.cobyqa.COBYQA` adapter (shared
  ``_make_pipe_objective`` / ``_safe_send`` shape, ``spawn`` context,
  ``cap=1``, graceful ``SystemExit``-on-closed-pipe shutdown).  New
  ctor kwargs ``max_starts`` / ``maxfun`` / ``epsilon`` / ``seed`` are
  all validated.
* **Why** — LBFGSB is the *only* gradient-based arm in a portfolio that
  is otherwise entirely derivative-free (DE family, PSO, CMA-ES,
  Nelder-Mead, COBYQA).  On smooth, ill-conditioned *valleys* a
  finite-difference quasi-Newton method converges in a fraction of the
  evaluations a population method needs.  The harness made the gap
  unmistakable: on a fresh ``--standard --baselines`` run, **every
  Panobbgo strategy scores 0.0 on ``Rosenbrock_5D``** (composite 0.26),
  while ``scipy``'s ``dual_annealing`` solves it (its win owes to its
  *own* L-BFGS-B local-search step).  The pre-existing LBFGSB could
  have closed this gap but was wired into neither the default
  strategies nor the structural catalog *and* ran only a single descent
  from the box centre — effectively dead code.
* **Impact** — A/B with the harness (`_run_single`, base_seed 42,
  budget 200):
  * A *dedicated* LBFGSB strategy (RoundRobin, single LBFGSB arm) solves
    **Rosenbrock_2D and Rosenbrock_5D to ``func_distance ≈ 3e-11``,
    SR 5/5** — where every default strategy scores 0.0.  A standalone
    ``scipy`` check confirms a single centre descent reaches
    ``Rosenbrock_5D`` ``f < 0.02`` in ~210 evals.
  * **Negative result worth recording:** simply *adding* LBFGSB (or
    COBYQA) to the existing 5-heuristic ``Rewarding_Diverse`` portfolio
    does **not** crack Rosenbrock_5D and can *regress* other problems
    (e.g. StyblinskiTang) — the bandit splits the 200-eval budget across
    6 arms, so no single gradient descent gets enough evaluations.  The
    value is in *dedicated* / loop-discovered portfolios where the
    gradient arm carries enough budget, which is exactly what the
    structural catalog lets the loop search for.  *This is why the
    change is catalog-only and does not touch the default battery —
    adding a gradient arm to a budget-split portfolio is not an
    unconditional win, and the loop's accept/reject + bootstrap-CI
    guard is the right place to decide it per battery.*
  * *Evidence form (per AGENTS.md "Agent-driven improve X PRs"): local
    A/B with the harness; backwards-compatible (no default battery
    change — composite baseline byte-identical, existing ledgers stay
    valid); queued for nightly loop validation via the structural
    catalog.*
* **Backwards compatibility** — strictly safe.  LBFGSB is opt-in (not in
  any ``_make_quick`` / ``_make_standard`` / ``_make_full`` strategy),
  so the composite baseline on every default battery is byte-identical.
  The first descent still starts from the box centre exactly as before,
  so the existing integration tests (`test_lbfgsb_integration`,
  `test_lbfgsb_constrained_integration`) and the ``on_new_results``
  penalty-value contract (`test_heuristics_lbfgsb_constraints.py`) pass
  unchanged.  The structural catalog gains one extra ``add_heuristic``
  candidate.
* **Tests** — Rewrote `tests/test_heuristic_lbfgsb.py` (29 tests) and
  `tests/test_heuristic_lbfgsb_robustness.py` (9 tests) on the COBYQA
  template: ctor validation (defaults, custom kwargs, invalid /
  bool-rejected ``max_starts`` / ``maxfun`` / ``epsilon``), subprocess
  lifecycle (spawn / stop / force-kill), pipe wiring (penalty routing,
  foreign-who ignore, pipe-closed exit, status logging, emit-on-poll),
  restart (relaunch, ``center=None`` box centre, out-of-box clip,
  stopped no-op, teardown-failure swallowed), worker behaviour through a
  fake pipe (completes all ``max_starts``, first start is box centre,
  clean ``SystemExit`` on closed pipe, seed-reproducible restarts,
  minimises a quadratic, survives a degenerate first descent), the
  ``_make_pipe_objective`` contract (NaN / None → ``inf``,
  passthrough, ``SystemExit``), registration (package re-export,
  structural-catalog membership), and an end-to-end ``scipy`` smoke
  proving a single descent cracks ``Rosenbrock_5D``.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; a new
    *LBFGSB follow-ups* block under "Next iteration ideas" (dedicated
    gradient-local-search default strategy — needs ADR; warm-start
    restarts from the portfolio best; ``LBFGSB.max_starts`` catalog
    rule).
  - `doc/source/heuristics.rst`: rewrote the ``LBFGSB`` bullet
    (multi-start, gradient-based, valley specialist, catalog opt-in).
  - `doc/source/guide_architecture.rst`: expanded the ``LBFGSB``
    classical-optimizer description and added the missing ``COBYQA``
    line beside it.
  - `doc/source/guide_benchmarking.rst`: structural-catalog candidate
    pool lists ``LBFGSB`` with its gradient-arm rationale.
  - `doc/source/guide.rst`: quick-nav entry mentions the multi-start
    L-BFGS-B candidate.
  - `AGENTS.md`: structural-catalog ``add_heuristic`` pool description
    now enumerates the DE family + COBYQA + LBFGSB.

### 2026-05-26 — Loop deduplication guard (in-flight PR awareness)

* **What** — Added §12.3 step 0 and a callout at the head of "Next
  iteration ideas" instructing the daily routine to run
  `gh pr list --state open` (drafts included) and consult §13 *before*
  picking a task. No source code changed — this is a process fix to the
  loop's own playbook.
* **Why** — The nightly routine branches from `master` and has no memory
  of unmerged work. NL-SHADE-RSP was listed under "Next iteration ideas"
  as a high-priority candidate (the natural step after jSO, which shipped
  2026-05-15). Each night 2026-05-23 … 2026-05-26 the routine branched
  from `master`, saw the idea still unshipped (the prior night's PR was
  open/draft, so it never updated `master`), and re-implemented it —
  producing four near-identical PRs (#227, #228, #229, #230). Each burned
  a full CI run (~21 min for the test job alone).
* **Resolution** — #229 (the most complete: non-linear LPSR + rank-based
  selective pressure + adaptive archive, clean base-class hooks, full
  bandit integration) was merged; #227/#228/#230 were closed as
  duplicates with their unique ideas captured as follow-ups (RSP on the
  `r2` donor; `archive_factor=2.6` default; 3-arg `_select_r1` hook).
* **Open / draft PRs are the source of truth for in-flight work** — the
  candidate list on `master` is not, because it does not reflect unmerged
  branches. The matching fix on the cron / routine side is to make the
  routine *finish or close* its PR each run rather than leave drafts to
  accumulate.

### 2026-05-25 — NL-SHADE-RSP adaptive DE (CEC 2021 winner)

* **What** — `panobbgo/heuristics/nl_shade_rsp.py` adds the
  :class:`NLSHADE_RSP` heuristic, a direct subclass of
  :class:`~panobbgo.heuristics.jso.JSO` that ports the
  Stanovov-Akhmedova-Semenkin (CEC 2021) "NL-SHADE-RSP" refinement.
  NL-SHADE-RSP inherits the entire jSO / L-SHADE asynchronous pipeline
  (per-slot pending dict, generation-by-count book-keeping, archive of
  replaced parents, success-history memory with the frozen jSO anchor
  bin, weighted ``current-to-pbest-w/1`` mutation, linear ``p_best``
  schedule, asymmetric F-cap, warm restart) and adds the three
  refinements the asynchronous model can carry cleanly:

  * **Non-Linear Population Size Reduction (NLPSR)**.  Replaces
    L-SHADE's linear schedule with
    ``NP(r) = round((NP_min − NP_init) · r^(1 − r) + NP_init)`` where
    ``r = len(results) / max_eval``.  Since ``r^(1−r) > r`` on
    ``(0, 1)`` (``0.5^0.5 ≈ 0.707``), the population drops *faster*
    early — concentrating the late-search budget on a small
    exploitative population sooner.  ``r^(1−r)`` is monotone increasing
    on ``[0, 1]``, so the population is monotone non-increasing.
  * **Rank-based Selective Pressure (RSP)** (LSHADE-RSP, Stanovov et
    al. 2018).  The differential ``r1`` index is drawn with probability
    proportional to a fitness rank weight ``w_i = k_rank·(n−i)/n + 1``
    (best first), biasing the mutation toward better individuals.
    ``k_rank`` default ``3`` (literature); ``k_rank = 0`` recovers
    jSO's uniform selection.
  * **Randomised adaptive archive**.  The archive cap is resampled per
    generation uniformly in ``[0, round(archive_factor·NP)]`` instead
    of the fixed jSO / L-SHADE cap.  Set ``adaptive_archive=False`` to
    recover the fixed cap.

  The implementation is enabled by a small, behaviour-preserving
  refactor of the L-SHADE base class into three override hooks —
  :meth:`LSHADE._select_r1` (r1 selection), :meth:`LSHADE._lpsr_target`
  (population-reduction schedule), and :meth:`LSHADE._archive_cap`
  (archive cap) — that L-SHADE and jSO consume with their *exact* prior
  RNG-draw sequence, so both stay byte-identical (verified: all 99
  pre-existing L-SHADE / jSO tests pass unchanged).
  :class:`NLSHADE_RSP` overrides only those three hooks plus
  :meth:`_end_of_generation` (resample the archive cap) and the
  start/restart resets.  Registered in :mod:`panobbgo.heuristics`;
  :func:`default_structural_catalog` gains it as a fourteenth
  ``add_heuristic`` candidate (``avoid_duplicates=True``);
  :func:`default_catalog` gains three rules — ``NLSHADE_RSP.NP_init``
  (integer_add), ``NLSHADE_RSP.k_rank`` (float_uniform ``[1, 5]``,
  live out-of-the-box because the catalog candidate sets ``k_rank``
  explicitly), and ``NLSHADE_RSP.adaptive_archive``
  (categorical ``True``/``False``).
* **Why** — closes the *NL-SHADE-RSP / NL-SHADE-LBC* DE-family
  follow-up below.  The DE arms shipped to date — basic DE
  (``DE/rand/1/bin``), L-SHADE (CEC 2014), jSO (CEC 2017) — cover the
  high-water mark up to ~2017.  NL-SHADE-RSP won the **CEC-2021**
  single-objective bound-constrained competition and is the direct
  jSO descendant; every later CEC winner (NL-SHADE-LBC, etc.) refines
  it.  Subclassing jSO keeps the new heuristic at the literature
  frontier while leaving jSO / L-SHADE byte-identical for ledger
  reproducibility — the precedent set by the jSO entry itself.  Adds a
  fourth DE-family arm the bandit can pick whichever wins on the
  current battery.
* **Deviations from the full CEC-2021 paper** — for honesty (the
  Panobbgo norm is literature-faithful ports): two NL-SHADE-RSP
  mechanisms are **not** ported because they interact with the
  synchronous generation model in ways the asynchronous pipeline does
  not expose cleanly — the *adaptive binomial / exponential crossover
  blend* and the exact *success-ratio archive-probability (pA)
  adaptation*.  Binomial crossover (inherited from jSO) and the
  randomised-cap variant from the *Next iteration ideas* sketch are
  used instead.  Both are queued as follow-ups below.
* **Impact** — A/B against jSO in the same Rewarding strategy (Random +
  Nearby + Center + NelderMead + DE-arm), fixed battery, **12 reps ×
  3 problems × 1000 evaluations** (12 reps to average out the
  bimodal basin-flipping noise that ±0.06 single-run swings exhibit at
  5 reps):

  * Seed 42 — ``jSO`` **0.874** / ``NLSHADE_RSP`` 0.798 (-0.076)
  * Seed 43 — ``jSO`` 0.848 / ``NLSHADE_RSP`` **0.874** (+0.026)
  * Seed 44 — ``jSO`` 0.771 / ``NLSHADE_RSP`` **0.822** (+0.051)
  * **Mean composite delta +0.0004** — a statistical tie.

  Each variant wins on different seeds — exactly the *complementarity*
  that motivates carrying both in the structural catalog (the jSO and
  COBYQA entries report the same pattern).  A component decomposition
  (RSP-only / NLPSR-only / archive-only vs jSO) confirmed there is no
  bug: every variant lands on the *same* basin attractors as jSO, the
  differences are basin-flipping noise.  The CEC-DE refinements are
  **large-budget specialists** — at panobbgo's small composite-battery
  budgets (75–500 evals) they barely warm up, so the quick-mode signal
  is within noise.  The value of shipping this today is to give the
  self-improvement loop a CEC-2021-class DE arm the bandit can select
  once it has accumulated per-arm reward history.  *Evidence form
  (per AGENTS.md "Agent-driven improve X PRs"): local A/B, within
  noise; the change is backwards-compatible (composite baseline
  unchanged — see below) and queued for nightly loop validation.*
* **Backwards compatibility** — strictly safe.  NL-SHADE-RSP is opt-in:
  it is not added to any default :func:`_make_quick_strategies` /
  :func:`_make_standard_strategies` / :func:`_make_full_strategies`
  spec, so the composite baseline on every default battery is
  byte-identical and existing ledgers stay valid.  The structural
  catalog gains it as one extra ``add_heuristic`` candidate
  (``avoid_duplicates=True``).  The kwarg rules fire only when a spec
  sets the matching kwarg explicitly.  The L-SHADE / jSO base-class
  refactor is behaviour-preserving: :meth:`_select_r1`,
  :meth:`_lpsr_target`, and :meth:`_archive_cap` reproduce the exact
  prior logic (same RNG draws) for the base classes — all 99
  pre-existing L-SHADE / jSO tests pass unchanged.
* **Tests** — `tests/test_heuristic_nl_shade_rsp.py` (34 tests):
  construction validation (defaults, custom kwargs, subclass invariant,
  invalid / zero-allowed ``k_rank``, invalid ``adaptive_archive`` type,
  inherited jSO ``H >= 2`` / ``p_best`` ordering rules); NLPSR
  (endpoints, monotonicity, faster-than-linear midrun with the concrete
  17 → 12 check, ``_apply_lpsr`` shrink + worst-dropped, no-op without
  budget); RSP (excludes target, returns ``None`` on empty pool, better
  individuals selected ≥ 2× more than worst at ``k_rank=3``, ``k_rank=0``
  ≈ uniform); adaptive archive (fixed cap when off, within-bounds sample,
  clip to shrunk ``A_max``, lazy single sample, ``_end_of_generation``
  resample, never exceeds cap); pipeline (on_start emits ``NP_init``,
  archive-cap reset, evolutionary trials, better-trial-wins-and-archives,
  restart reset, end-to-end smoke convergence on a quadratic);
  base-class hook safety (L-SHADE ``_select_r1`` uniform-excludes-target,
  ``_lpsr_target`` linear, ``_archive_cap`` fixed); and registration
  (package re-export + ``__all__``, structural catalog membership, kwarg
  catalog dials).
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *NL-SHADE-RSP / NL-SHADE-LBC heuristic* next-iteration idea promoted
    to "shipped (NL-SHADE-RSP)"; a new *adaptive crossover blend +
    pA archive adaptation* follow-up left for the next iteration.
  - `doc/source/heuristics.rst`: new ``NLSHADE_RSP`` bullet; the
    DE-family complementarity bullet now names all four arms.
  - `doc/source/guide_architecture.rst`: new ``NLSHADE_RSP``
    description after jSO.
  - `doc/source/guide_benchmarking.rst`: structural-catalog candidate
    pool lists ``NLSHADE_RSP``; categorical-rules section gains the
    ``NLSHADE_RSP.adaptive_archive`` rule (count three → five).
  - `doc/source/guide.rst`: quick-nav entry mentions NL-SHADE-RSP and
    the new categorical knob.
  - `AGENTS.md`: categorical-rules list adds
    ``NLSHADE_RSP.adaptive_archive`` (count four → five).

### 2026-05-22 — Von Neumann (4-connected 2-D toroidal grid) PSO topology

* **What** — `panobbgo/heuristics/pso.py`: :class:`PSO` gains a third
  shipped topology, ``"vonneumann"``, via two new helpers
  :meth:`_vonneumann_grid` (factors ``NP`` into ``R × C >= NP`` with
  ``R ≈ √NP``) and :meth:`_vonneumann_neighbors` (returns the
  4-connected wrap-around N/S/E/W indices plus the particle itself,
  skipping phantom slots whose index is ``>= NP`` when the grid is
  not a perfect rectangle).  :meth:`_social_best_idx` dispatches the
  new topology onto the same scan-for-best-neighbour-pbest routine
  already used by ``lbest``.  :func:`default_structural_catalog`
  gains a third PSO entry — ``(PSO, {"NP": 20, "topology":
  "vonneumann"})`` — alongside the existing ``gbest`` and ``lbest``
  entries.  All three share ``cls = PSO`` so ``avoid_duplicates=True``
  still prevents multiple PSO instances per strategy.  The default
  catalog's existing ``PSO.topology`` categorical rule grows from
  two choices to three (``("gbest", "lbest", "vonneumann")``) so the
  bandit can flip an existing explicit-topology PSO between all three
  regimes without dropping and re-adding the heuristic.
* **Why** — closes the *Random / Von Neumann topologies* PSO follow-up
  under the §13 entry from 2026-05-07.  ``gbest`` and ``lbest`` cover
  the two extremes of the diffusion-speed spectrum (instantaneous
  full-connect vs one-hop ring); Von Neumann's 4-connected grid sits
  between them — two-dimensional information diffusion that gives
  multiple sub-swarms room to probe distinct basins without the slow
  linear chain of ``lbest``.  Mendes (2004) PhD thesis identifies Von
  Neumann as a strong default across a wide problem battery; the
  literature consensus (Kennedy & Mendes 2002, 2003) is that the
  three topologies are *complementary* and the best choice depends
  on the problem landscape.  Shipping all three in the structural
  catalog gives the self-improvement loop a third PSO arm the bandit
  can pick whichever wins on the current battery.
* **Grid factoring** — ``rows = round(√NP)``, ``cols = ceil(NP/rows)``
  so ``rows · cols >= NP`` and ``rows ≈ √NP``.  Perfect rectangles in
  this scheme (``NP ∈ {4, 6, 9, 12, 16, 20, 25, …}``) leave no phantom
  cells; non-square NPs (``NP ∈ {7, 8, 10, 11, 13, 17, 19, 23, …}``)
  leave 1–3 phantom slots that :meth:`_vonneumann_neighbors` skips —
  edge particles on the trailing partial row then have 3 or 4 real
  neighbours instead of 5.  Wrap-around on very small swarms
  (``NP=4``) collapses N/S to the same cell; :meth:`_vonneumann_neighbors`
  de-duplicates so the caller always sees a *set*.
* **Asynchronous adaptation** — Von Neumann is a *static* topology
  (the grid layout is fixed at construction time, just like the ring
  for ``lbest``).  No state changes between ``on_start`` /
  ``on_new_results`` / ``on_restart``; the social-attractor lookup
  is read-only.  PSO's per-particle pbest update path is unchanged,
  so the existing IPOP-style warm restart works without modification.
* **Impact** — the point of shipping today is to give the bandit a
  third PSO arm with markedly different exploration dynamics to
  choose between, rather than to claim a single-shipped-variant win.
  The 2026-05-07 ``lbest`` entry's A/B benchmark already established
  that no single PSO topology dominates at quick-mode noise levels
  (~ ±0.05) — seeds 42 and 43 split the win between ``gbest`` and
  ``lbest``.  The literature (Kennedy & Mendes 2002, 2003; Mendes
  2004) predicts Von Neumann's two-hop planar diffusion sits between
  gbest's instantaneous diffusion and lbest's one-hop linear
  diffusion, and Mendes' PhD thesis identifies it as a stable
  default across a broader battery than either extreme.  The
  measurable signal will materialise once the self-improvement loop
  has accumulated enough evidence from the bandit's per-arm reward
  history to identify which topology wins on the current battery.
* **Backwards compatibility** — strictly safe.  ``topology`` defaults
  to ``"gbest"``; every existing PSO instance retains its prior
  behaviour bit-for-bit, including the 56 pre-existing tests in
  ``tests/test_heuristic_pso.py``.  The structural catalog gains one
  extra ``add_heuristic`` candidate that shares ``cls = PSO`` with the
  existing entries; under ``avoid_duplicates=True`` (default), only
  one of the three is ever added per strategy.  The categorical
  rule expansion is also safe: callers passing the prior choices
  tuple get the same uniform-over-the-set draw (the cardinality just
  bumps from 2 to 3), and the rule still fires only when a spec
  sets ``topology`` explicitly.  Existing ledger consumers parsing
  the rule's ``choices`` field see one extra string they may ignore.
* **Tests** — `tests/test_heuristic_pso.py` (+11 new tests, total
  67): vonneumann construction round-trip; grid factoring for
  perfect rectangles (``NP ∈ {4, 9, 12, 16, 20, 25}`` — rows·cols
  exactly equals ``NP``); grid factoring for primes / near-primes
  (``NP ∈ {7, 11, 13, 17, 19, 23}`` — rows·cols > NP, rows ≈ √NP);
  4-connected wrap-around correctness on a 4×5 grid (corner
  particles 0, 12, 19 verified); phantom-cell skipping on a 3×4
  grid with NP=10 (particles 7 and 2 each have 4 real neighbours
  instead of 5); duplicate elimination on a 2×2 swarm (NP=4);
  social attractor uses the 2-D neighbourhood, *not* the global
  best, when a better pbest exists outside the N/S/E/W set;
  social attractor returns ``None`` until at least one neighbour
  has a pbest; velocity clamp invariant under vonneumann; an
  end-to-end smoke run confirming the swarm strictly improves
  on a quadratic; a categorical-rule membership test confirming
  the default catalog now ships
  ``choices=("gbest", "lbest", "vonneumann")``; an updated
  structural-catalog test confirming all three PSO topology
  variants appear among the ``add_heuristic`` candidates.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Random / Von Neumann topologies* PSO follow-up below the
    2026-05-07 entry promoted from "open" to "shipped" for
    Von Neumann.
  - `doc/source/guide.rst`: quick-nav entry mentions the
    tri-topology PSO candidate pool.
  - `doc/source/guide_benchmarking.rst`: structural-catalog
    section now describes the three PSO entries; the categorical
    rule section lists ``vonneumann`` as a third PSO.topology
    choice.
  - `doc/source/guide_architecture.rst`: PSO description now
    enumerates all three topologies.
  - `doc/source/heuristics.rst`: PSO bullet updated to the
    three-topology description.
  - `AGENTS.md`: PSO.topology categorical rule entry updated.

### 2026-05-21 — jSO asymmetric F-cap (three-phase, Brest 2017)

* **What** — `panobbgo/heuristics/lshade.py`:
  :class:`LSHADE` gains an opt-in ``F_schedule: Optional[bool] = None``
  kwarg, a new :meth:`_progress` helper (returns ``None`` when the
  budget is unknown so each schedule picks its own fall-back), and a
  new :meth:`_apply_F_cap` helper that implements the three-phase
  asymmetric cap.  The cap is keyed on
  ``progress = len(strategy.results) / strategy.config.max_eval``::

      F ≤ 0.7   if  progress < 0.6
      F ≤ 0.8   if  progress < 0.9
      F ≤ 1.0   otherwise (unclamped — sampler already enforces ≤ 1)

  When ``F_schedule`` is ``None`` (default) or ``False`` the cap is
  bypassed and :class:`LSHADE` reproduces the byte-identical
  Tanabe-Fukunaga 2014 behaviour shipped 2026-05-10.
  ``_sample_F_CR()`` consults ``_apply_F_cap()`` once on every draw so
  the cap is shared infrastructure rather than per-subclass code.
  :class:`~panobbgo.heuristics.jso.JSO` opts into the cap by
  construction (passes ``F_schedule=True`` to ``super().__init__``)
  and drops its own ``_progress`` / ``_sample_F_CR`` overrides in
  favour of the inherited versions.  :func:`default_catalog` gains
  one new :class:`MutationRule` (``LSHADE.F_schedule``,
  ``categorical_choice`` over ``(True, False)``) so the loop driver
  can flip an existing :class:`LSHADE` instance between the
  Tanabe-Fukunaga and jSO regimes without dropping and re-adding the
  heuristic.
* **Why** — closes the *jSO asymmetric F-cap during early
  generations* follow-up under the 2026-05-19 iLSHADE / jSO ``p_best``
  entry.  jSO (Brest et al. 2017) ships with a **three-phase**
  asymmetric F-cap as part of its winning CEC-2017 spec; the
  2026-05-15 :class:`JSO` ship implemented only the *first* phase
  (``F ≤ 0.7`` while ``progress < 0.6``) and left the middle phase
  (``F ≤ 0.8`` while ``0.6 ≤ progress < 0.9``) absent — a literature
  drift that this entry fixes.  Adding the same cap as an opt-in on
  :class:`LSHADE` also gives the structural-mutation-free regime a
  way to access the jSO refinement without dropping and re-adding
  the heuristic: a single ``F_schedule`` flip lets the bandit move
  L-SHADE between the Tanabe-Fukunaga and Brest regimes.  The cap is
  Brest et al. (2017, §III-D) verbatim.
* **Asynchronous adaptation** — the cap reads
  ``progress = len(strategy.results) / max_eval`` — the same idiom
  L-SHADE already uses for LPSR pacing — so the F-cap stays in
  lock-step with the population shrink.  When the budget is unknown
  (no ``max_eval``, zero, or non-numeric) the cap is bypassed,
  matching the LPSR fallback: an unmeasured environment keeps the
  heuristic in the unclamped Tanabe-Fukunaga regime rather than
  guessing a horizon.
* **Impact** — micro-benchmark on a single-LSHADE Rewarding strategy
  (3 problems × 5 reps × 150 evaluations), comparing
  ``F_schedule=False`` (legacy L-SHADE) vs ``F_schedule=True`` (jSO
  F-cap) across three seeds:

  * Seed 42 — 0.811 → **0.828** (+0.017)
  * Seed 43 — **0.835** → 0.726 (-0.109)
  * Seed 44 — 0.688 → **0.827** (+0.138)

  Mean delta +0.015 across seeds, with high per-seed variance at
  quick budgets — exactly the regime where the literature reports
  L-SHADE's success-history adaptation is still warming up.  The
  point of shipping this today is not the quick-mode delta (within
  noise) but the *literature-faithful* completion of jSO: the
  2026-05-15 :class:`JSO` ship was missing the second phase of the
  asymmetric cap that won CEC-2017, and the structural-mutation
  catalog now exposes the same opt-in on plain :class:`LSHADE`.
* **Backwards compatibility** — strictly safe on L-SHADE.
  ``F_schedule=None`` (default) bypasses the cap, so every existing
  L-SHADE instance retains its prior behaviour bit-for-bit, including
  all pre-existing tests in ``tests/test_heuristic_lshade.py``.  The
  new ``LSHADE.F_schedule`` catalog rule only fires when a spec
  explicitly sets the kwarg (per :func:`_find_targets`'s "param
  already in kwargs" predicate), so a fresh ledger run on the
  built-in factories sees no behavioural change.  Existing ledger
  consumers parsing only numeric ``rule_kind`` strings see one extra
  categorical rule they may ignore.  **jSO behaviour changes**: the
  middle-phase cap (``F ≤ 0.8`` while ``0.6 ≤ progress < 0.9``) was
  not active before this entry, so jSO instances will draw slightly
  smaller ``F`` values in roughly 30% of the budget.  The change is
  a literature-faithful completion rather than a behaviour
  regression; the unit tests have been updated to reflect the
  three-phase contract.
* **Tests** — `tests/test_heuristic_lshade.py` (+15 tests, total 97):
  default ``F_schedule`` is ``None``, custom construction with
  ``True`` / ``False``, invalid type rejection, ``_apply_F_cap``
  disabled-when-off paths (None and False), three-phase clamping
  (phase 1 ≤ 0.7, phase 2 ≤ 0.8 and admits values > 0.7, phase 3
  unclamped), phase-boundary inclusivity (progress = 0.6 → phase 2;
  progress = 0.9 → phase 3), bypass when budget unknown, end-to-end
  ``_sample_F_CR`` respects the cap across phases, ``_progress``
  returns ``None`` without budget, ``_progress`` clipping, and a
  catalog membership test confirming ``("LSHADE", "F_schedule")``
  joins the default rule set.  `tests/test_heuristic_jso.py` (+3
  tests, total 36): jSO opts into ``F_schedule=True`` by
  construction; jSO ``_progress()`` returns ``None`` (not 0.0)
  without budget; jSO ``_current_p_best`` / ``_current_F_weight``
  fall back to the early-phase value when the budget is unknown.
  Plus updated tests for the *three-phase* clamp on jSO (replacing
  the old two-phase tests).
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *jSO asymmetric F-cap during early generations* follow-up
    promoted from "open" to "shipped".
  - `doc/source/guide_benchmarking.rst`: the L-SHADE / jSO entries
    under the structural-catalog "Algorithms in the candidate pool"
    section now mention the opt-in jSO F-cap on L-SHADE and the
    literature-faithful three-phase cap on jSO.
  - `AGENTS.md`: self-improvement loop subsection lists the new
    catalog rule.

### 2026-05-19 — iLSHADE / jSO adaptive ``p_best`` schedule

* **What** — `panobbgo/heuristics/lshade.py`:
  :class:`LSHADE` gains an opt-in ``p_best_end: Optional[float] = None``
  keyword argument and a new :meth:`_current_p_best` helper.  When
  ``p_best_end`` is set, the effective greediness at evaluation count
  ``e`` (out of ``E = strategy.config.max_eval``) becomes
  ``p_eff(e) = p_best − (p_best − p_best_end) · min(e/E, 1)`` — the
  iLSHADE (Brest et al. 2016) / jSO (Brest et al. 2017) linearly-
  decreasing schedule that shrinks the ``current-to-pbest/1``
  greediness as the population shrinks under LPSR.  When
  ``p_best_end is None`` (default), :meth:`_current_p_best` returns
  ``self.p_best`` unchanged — byte-identical to the 2026-05-10 ship.
  When the strategy budget is unknown (no ``max_eval``, zero, or
  non-numeric) the heuristic falls back to constant ``self.p_best``
  rather than guessing a horizon, matching the
  :class:`~panobbgo.heuristics.pso.PSO` ``w_end`` pattern shipped
  2026-05-07.  ``_generate_trial`` now consults
  ``_current_p_best()`` exactly where it used ``self.p_best`` before,
  so the mutation / crossover / bounds-reflection paths are shared.
  :func:`default_catalog` gains one new :class:`MutationRule`
  (``LSHADE.p_best_end``, ``float_uniform`` over the literature
  range ``[0.025, 0.15]``) so the loop driver can tune the
  adaptive-greediness schedule once a spec opts in by setting the
  kwarg explicitly.
* **Why** — closes the *iLSHADE / jSO* follow-up under the L-SHADE
  entry below.  L-SHADE shipped 2026-05-10 with the fixed
  Tanabe-Fukunaga 2014 ``p_best = 0.11``; the iLSHADE refinement
  (Brest et al. 2016) showed that linearly shrinking ``p_best`` over
  the run pairs naturally with LPSR — when the population is large
  (early), exploration benefits from a broader top-p slice; when the
  population is small (late), exploitation benefits from pulling
  toward a tighter top-p slice.  jSO (Brest et al. 2017) builds on
  iLSHADE and won the CEC-2017 single-objective competition,
  establishing the schedule as the literature-best refinement on
  top of L-SHADE.  The extension is *opt-in* — the default
  constructor preserves the shipped behaviour exactly — so the
  loop driver can discover whether any given strategy benefits
  without disturbing existing ledgers.
* **Impact** — measured A/B at ``--quick`` (3 problems × 3 reps ×
  75 evaluations, seed 42), comparing a single L-SHADE-backed
  Rewarding strategy with and without the schedule:

  * ``LSHADE (fixed)``        — DeJong / Rosenbrock / Rastrigin,
    constant ``p_best=0.25``.
  * ``LSHADE (jSO schedule)`` — same, plus ``p_best_end=0.125``
    (canonical jSO half-greediness annealing).

  The schedule contributes most when the late-search pressure
  needs to be sharper — exactly the regime where the literature
  reports the largest jSO-over-L-SHADE gains.  At ``--quick``
  budgets the cost is mostly noise; the value of shipping this
  today is to give the bandit a *literature-best DE arm* it can
  pick whichever wins on the current battery once enough loop
  iterations have run.
* **Backwards compatibility** — strictly safe.  ``p_best_end``
  defaults to ``None``; every existing :class:`LSHADE` instance
  retains its prior behaviour bit-for-bit, including all 39
  pre-existing tests in ``tests/test_heuristic_lshade.py``.  The
  new ``LSHADE.p_best_end`` catalog rule only fires when a spec
  explicitly sets the kwarg (per :func:`_find_targets`'s "param
  already in kwargs" predicate), so a fresh ledger run on the
  built-in factories sees no behavioural change.  Existing ledger
  consumers parsing only ``rule_kind`` strings are unaffected —
  ``p_best_end`` uses the existing ``float_uniform`` kind.
* **Tests** — `tests/test_heuristic_lshade.py` (10 new tests,
  total 49): construction validation (default ``p_best_end``
  is ``None``; opt-in construction round-trips; invalid
  ``p_best_end`` rejected — zero / negative / too-large / NaN /
  inf), schedule semantics
  (:meth:`LSHADEAdaptivePBestTests.test_constant_when_p_best_end_is_none`,
  ``test_linear_decrease_when_p_best_end_set``,
  ``test_clipped_above_full_budget``,
  ``test_linear_increase_when_p_best_end_above_p_best``,
  ``test_constant_when_budget_unknown``,
  ``test_p_best_end_equal_to_p_best_is_constant``), end-to-end
  pool sizing (``test_generate_trial_uses_scheduled_p_best``),
  and a catalog membership test confirming
  ``LSHADE.p_best_end`` joins ``NP_init`` / ``H`` / ``p_best``
  in :func:`default_catalog`.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    iLSHADE / jSO follow-up below the L-SHADE entry promoted from
    "open" to "shipped".
  - `doc/source/guide_benchmarking.rst`: the L-SHADE bullet
    under the structural-catalog "Algorithms in the candidate
    pool" section now names the opt-in iLSHADE / jSO
    ``p_best_end`` schedule alongside L-SHADE's success-history
    adaptive DE / LPSR description.

### 2026-05-18 — Per-class bandit arms for structural mutations

* **What** — `panobbgo/self_improve.py`:
  :class:`AdaptiveMutationSampler` gains a
  ``per_class_structural: bool = False`` constructor argument.
  When ``True``, each :class:`StructuralMutationRule` is expanded at
  :meth:`sample` time into one bandit arm per candidate class
  (``add_heuristic`` ``Sobol`` is now distinct from ``add_heuristic``
  ``Random``), Thompson-sampled directly so the bandit can learn
  *which class* wins or loses inside a structural op.
  :func:`_proposal_rule_key` gains a matching
  ``per_class_structural`` keyword so :meth:`prime_from_ledger`
  recovers the same arm layout as live sampling — without the
  flag, structural records still collapse onto the legacy
  ``("*", op, "structural")`` wildcard.  :class:`LoopConfig` gains
  ``structural_per_class_arms: bool = False`` and ``SelfImprover``
  passes it through to the sampler whenever the adaptive path is
  used.  ``scripts/self_improve.py`` gains a
  ``--structural-per-class-arms`` CLI flag (only effective with
  ``--adaptive``).  A new helper
  :meth:`AdaptiveMutationSampler._structural_arm_key` centralises
  the "per-class vs collapsed" decision so :meth:`sample` and
  :meth:`prime_from_ledger` cannot drift out of sync.
* **Why** — closes the *Per-class arms in the bandit* follow-up
  below the §13 entry from 2026-05-03.  The structural catalog
  shipped 2026-05-03 collapses every ``add_heuristic`` proposal —
  regardless of which class is added — into the single
  ``("*", "add_heuristic", "structural")`` bandit arm.  That makes
  cold-start variance small (one arm = lots of evidence per draw)
  but is conceptually wrong once enough evidence accumulates: if
  ``add_heuristic Sobol`` is consistently accepted and
  ``add_heuristic Random`` is consistently rejected, the bandit
  cannot learn the difference; the wildcard arm's posterior is a
  weighted average of two regimes the sampler still mixes uniformly.
  Per-class arms split the posterior so the bandit can concentrate
  probability on the *winning class* (Thompson sampling's headline
  guarantee).  This pairs naturally with the next-iteration
  *contextual / hierarchical bandit* idea: per-class arms are the
  leaf nodes a hierarchical Beta-Binomial would share strength
  across.
* **Backwards compatibility** — strictly safe.  Default is ``False``
  for the new constructor argument and the new ``LoopConfig`` field;
  existing CLI invocations and existing ledger consumers see the
  same arm layout they always have.  When the flag is on, live
  sampling and :meth:`prime_from_ledger` use *the same* key layout
  (delegated through :func:`_proposal_rule_key`'s new
  ``per_class_structural`` keyword), so resuming with
  ``--adaptive-prime-from-ledger`` works identically to a fresh
  run.  Kwarg perturbations are unaffected regardless of the flag —
  their ``(class_name, param_name, kind)`` arms are already
  per-class.  When ``--adaptive`` is *not* set the flag is inert
  (no :class:`AdaptiveMutationSampler` is constructed); we tolerate
  the combination rather than reject it so a caller can safely set
  the flag in a config that may toggle ``adaptive_sampling`` later.
* **Tests** — `tests/test_self_improve.py` (11 new tests, total
  158):
  :func:`_proposal_rule_key` per-class round-trip (per-class flag
  adds the class name, off-mode collapses, kwarg keys unaffected);
  default ``per_class_structural=False`` on the sampler; structural
  arms split per candidate class (both X and Y observed, total
  attempts conserved, wildcard key absent); Thompson sampling
  concentrates probability on the winning class
  (4x ratio threshold over 500 post-training samples); drop ops
  also produce per-class arms (both A and B observed across hits);
  kwarg arms untouched by the flag; :meth:`prime_from_ledger`
  uses per-class keys when flag is on; off-flag priming still
  collapses to the wildcard arm; ``LoopConfig`` default is
  ``False``; flag propagates to sampler via :class:`SelfImprover`;
  flag is inert without adaptive sampling.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Per-class arms in the bandit* follow-up below the 2026-05-03
    entry promoted from "open" to "shipped".
  - `doc/source/guide_benchmarking.rst`: new "Per-class
    structural bandit arms" subsection under the adaptive
    sampler.
  - `doc/source/guide.rst`: quick-nav entry mentions per-class
    structural arms.
  - `AGENTS.md`: self-improvement loop subsection lists the
    feature with a run-the-loop bash example.

### 2026-05-17 — Bootstrap CI on multi-seed hold-out drift

* **What** — `panobbgo/self_improve.py`:
  :class:`LoopHoldoutRecord` gains two list-typed fields
  (`seed_iteration_scores`, `top_iteration_scores`) that persist the
  per-iteration paired composite scores of the seed and top ladder
  entries on the hold-out instances.  Both default to empty lists so
  every legacy ledger record reads back unchanged.
  :class:`HoldoutDriftAggregate` (new dataclass) and
  :func:`aggregate_holdout_drift` (new module-level helper) pool the
  per-iteration paired drifts across every input record and
  bootstrap-resample the mean using the same
  :func:`statistical_accept`-style machinery already in
  :mod:`panobbgo.harness`.  A record's drift contribution at iteration
  ``k`` is ``(top_k − seed_k) − training_delta_r``; pooling across
  ``r`` (records / hold-out seeds) and ``k`` (iterations within a
  record) turns the previous worst-case point reduction into a real
  CI.  `aggregate_holdout_drift` falls back to one-sample-per-record
  on legacy records that lack the per-iteration lists, so mixed
  ledgers work transparently.  ``scripts/self_improve.py`` prints the
  CI on both `run` and `summary` and gains a `--fail-on-overfit-ci`
  flag plus tunable `--holdout-ci-confidence` /
  `--holdout-ci-n-boot` knobs.  The CI verdict ``OVERFIT_CI`` fires
  iff ``ci_high < -holdout_eps_overfit`` — i.e. the bootstrap rules
  out a drift better than the tolerance at the configured confidence
  level.
* **Why** — closes the *Bootstrap CI on the drift estimate*
  follow-up listed under the 2026-05-16 multi-seed hold-out entry.
  The shipped multi-seed reduction (``min`` over drifts, ``any`` over
  overfit flags) is conservative — one bad seed flags the entire
  ladder — but gives no sense of whether the worst-case drift is
  typical or a lucky tail of a small sample.  A single recent ledger
  run reported ``drift=-0.0074`` (well within the default
  ``eps_overfit=0.05``); the new aggregate places the same data at
  ``mean=-0.0012, CI95%=[-0.0037, +0.0000]`` — i.e. the data does
  **not** rule out zero drift.  That re-interpretation matters: the
  loop is not silently overfitting, it is just noisy at quick-mode
  budgets.  The CI also gives unattended cron-driven loops a
  principled exit rule that does not over-react to single-seed
  noise.  Pairs naturally with the existing
  :func:`statistical_accept` rule.
* **Backwards compatibility** — strictly safe.  The two new fields on
  :class:`LoopHoldoutRecord` default to empty lists; existing
  callers (including all 147 prior tests) construct records without
  the kwargs.  Reading a legacy JSONL ledger works through the
  empty-list defaults, and `aggregate_holdout_drift` treats records
  without per-iteration lists as one-sample legacy contributions.
  The new CLI flags (`--fail-on-overfit-ci`, `--holdout-ci-confidence`,
  `--holdout-ci-n-boot`) are all opt-in.  Existing
  `--fail-on-overfit` behaviour is unchanged.
* **Cost** — `aggregate_holdout_drift` is a vectorised numpy bootstrap
  that runs in well under a second at ``n_boot=10000`` for the typical
  multi-seed × multi-iteration sample size (≤ 50 paired drifts);
  negligible relative to the hold-out's harness cost.  The two list
  fields on the record add at most ``holdout_iterations`` floats per
  hold-out record per seed — typically 5–10 floats per seed.
* **Tests** — `tests/test_self_improve.py` (+20 tests, total 167):
  the empty-input degenerate path, the per-iteration pooling path
  (records × iterations), legacy-record fallback to one sample per
  record, mixed legacy + modern aggregation, worst-drift / worst-seed
  reductions, any-overfit semantics, ``statistically_overfit`` true
  on constant-negative samples and false on mixed-sign samples, CI
  width vs confidence level, reproducibility under fixed seed,
  distinct seeds give distinct CIs on non-degenerate samples,
  explicit ``eps_overfit`` override, defensive handling of
  unequal-length per-iteration lists, JSON round-trip of the
  aggregate, default empty lists on :class:`LoopHoldoutRecord`,
  per-iteration scores reach ``to_dict``, end-to-end
  :class:`SelfImprover` runs persist per-iteration scores both
  single-seed and multi-seed, and JSONL round-trip preserves the
  new fields.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Bootstrap CI on the drift estimate* follow-up promoted from
    "open" to "shipped"; §2 missing-pieces list refreshed.
  - `doc/source/guide_benchmarking.rst`: new
    "Bootstrap CI on the aggregated drift" subsection under
    "Hold-out validation set" with the bootstrap formula, the
    CLI example, programmatic example, and the legacy-fallback
    note.
  - `doc/source/guide.rst`: quick-nav entry mentions the
    bootstrap-CI aggregation.
  - `AGENTS.md`: self-improvement loop subsection updated.

### 2026-05-16 — Multi-seed hold-out for robust drift estimation

* **What** — `panobbgo/self_improve.py`:
  :class:`LoopConfig` gains a list-typed
  ``holdout_base_seeds: Tuple[int, ...]`` field (default ``()``)
  that sits alongside the scalar ``holdout_base_seed`` shipped
  2026-05-08.  A new helper :meth:`LoopConfig.resolved_holdout_seeds`
  returns the effective seed tuple: the list when non-empty, else
  the scalar promoted to a 1-tuple, else ``()`` (= disabled).
  :meth:`LoopConfig.holdout_harness_config` gains an optional
  ``base_seed`` argument so the multi-seed loop can drive the
  ``HarnessConfig.seed`` per call rather than reading it from the
  config attribute.  :class:`SelfImprover._run_holdout` similarly
  takes ``base_seed`` as a parameter (formerly read from
  ``self.config.holdout_base_seed``) and :class:`SelfImprover._run_internal`
  iterates over the resolved tuple, writing one
  :class:`LoopHoldoutRecord` per seed to the ledger.  The
  ``record_type='holdout'`` tag is unchanged, so existing ledger
  consumers see N records back-to-back per loop run instead of one.
  ``scripts/self_improve.py`` gains a ``--holdout-base-seeds``
  flag that accepts a comma-separated list (e.g.
  ``--holdout-base-seeds 1234,5678,9012``); the parser tolerates
  whitespace and trailing commas and rejects non-integer tokens
  with a clear error.  The CLI's end-of-run summary line and the
  ``summary`` subcommand both report the aggregated verdict:
  ``OVERFIT`` if *any* per-seed record flagged overfit, with the
  *worst* (most negative) drift across seeds.
* **Why** — closes the *Multi-seed hold-out for robust drift
  estimation* follow-up below.  The single-seed hold-out shipped
  2026-05-08 reduces the entire generalisation question to one
  independent SHA-256 draw, and a single recent ledger run
  produced ``drift=-0.0074`` (well within the default
  ``eps_overfit=0.05``, but on a single draw it is hard to know
  whether ``-0.0074`` is the typical drift or the lucky tail of a
  larger one).  Multi-seed aggregation gives a worst-case
  estimate over several independent draws — strictly more
  conservative — at a cost that scales linearly with the seed
  list and stays small relative to the loop's training budget.
  The reduction matches the planning doc's request: ``min`` over
  drifts, ``any`` over overfit flags.
* **Backwards compatibility** — strictly safe.  The default for
  ``holdout_base_seeds`` is the empty tuple; existing callers that
  set only ``holdout_base_seed`` see exactly one
  :class:`LoopHoldoutRecord` as before.  ``resolved_holdout_seeds()``
  promotes a scalar to a 1-tuple, so the multi-seed code path
  handles both cases through one branch.  When both are set, the
  list takes precedence (the explicit "do exactly this" override)
  and the scalar is silently ignored.  No existing ledger or
  ledger consumer is affected; the new records share the same
  schema as the single-seed record and the same ``record_type``
  tag.
* **Validation** — three rules at config construction time, with
  distinct error messages: no zero entries (``0`` is the disable
  sentinel), no collision with ``base_seed``, no duplicates.  The
  CLI parser also tolerates ``"1234, 5678 , 9012"`` and trailing
  commas so common copy/paste inputs don't trip the user.
* **Cost** — fixed at ``2 × holdout_iterations × len(seeds)``
  harness runs at the end of the loop (or
  ``holdout_iterations × len(seeds)`` when the ladder has only
  the seed entry — both endpoints are the same spec list).  At
  the standard ``holdout_iterations=5`` with 3 seeds that is 30
  extra harness runs, small relative to the ``2 × iterations``
  cost of a typical 50-iteration loop.
* **Tests** — `tests/test_self_improve.py` (25 new tests, total 147):
  config validation (default empty tuple, list/tuple normalization,
  zero entry rejected, collision with base_seed rejected,
  duplicates rejected), :meth:`resolved_holdout_seeds` (list
  precedence, scalar fallback, empty fallback),
  :meth:`holdout_harness_config` explicit-seed override and
  default-to-scalar paths, end-to-end behaviour (one record per
  seed in configured order, per-seed harness seeds reach the
  factory, overfit flagged independently per seed, list-wins-over-
  scalar precedence, all records written to JSONL ledger, scalar
  back-compat path unaffected, disable when both knobs unset), and
  the CLI parser (empty / whitespace / single / multiple /
  whitespace-tolerant / negative-accepted / non-integer-rejected /
  trailing-comma-skipped paths — 8 tests).
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Multi-seed hold-out for robust drift estimation* follow-up
    promoted from "open" to "shipped".
  - `doc/source/guide_benchmarking.rst`: new "Multi-seed hold-out"
    subsection under "Hold-out validation set" with the
    aggregation rule, validation rules, CLI example, and
    programmatic example.
  - `doc/source/guide.rst`: quick-nav entry mentions the multi-seed
    hold-out.
  - `AGENTS.md`: self-improvement loop subsection lists the
    multi-seed feature with a run-the-loop bash example.

### 2026-05-14 — Paired bootstrap for `statistical_accept`

* **What** — `panobbgo/harness.py`:
  :func:`statistical_accept` gains a ``paired: Optional[bool] = None``
  parameter and :class:`StatisticalDecision` gains a ``paired: bool``
  field.  When ``paired=True`` (or auto-selected when
  ``n_before == n_after`` on at least one shared pair), the per-pair
  bootstrap draws **one shared resample index** and applies it to both
  sides — mathematically equivalent to bootstrapping the per-rep delta
  vector ``d = a_frac − b_frac``.  ``paired=False`` (or the auto
  fallback for asymmetric-rep pairs) preserves the historical
  independent-resample sampler.  ``paired=True`` with mismatched rep
  counts truncates to the common prefix so index alignment stays valid;
  ``paired=False`` is the safe choice when reps are *not*
  instance-aligned (e.g. comparing ledgers built with different
  ``base_seed`` values).  The CLI gains ``--paired`` /
  ``--unpaired`` mutually-exclusive flags on
  ``benchmark_harness.py compare --statistical`` and on
  ``scripts/self_improve.py run``.
  :class:`~panobbgo.self_improve.LoopConfig` gains a matching
  ``paired: Optional[bool] = None`` field that is forwarded through to
  ``statistical_accept`` for every iteration's accept/reject decision.
  ``StatisticalDecision.print_summary()`` reports
  ``bootstrap=paired|unpaired``; the JSON payload from
  ``--json --statistical`` carries the new ``paired`` boolean.
* **Why** — closes the measurement gap §6.1 implicitly assumed.  Under
  ``--randomize`` (the recommended setting for the autonomous loop) the
  harness keeps reps instance-aligned by index — rep ``i`` on the
  ``before`` side and rep ``i`` on the ``after`` side are evaluated on
  the *same* sampled problem instance because
  ``derive_instance_seed(base_seed, iteration_id, family, rep)`` is
  deterministic.  The per-rep deltas are therefore strongly positively
  correlated and the historical independent-resample bootstrap throws
  that signal away, inflating the CI proportionally to the within-side
  rep variance and leaving the loop unable to clear ``ci_low > 0`` on
  genuinely improving but moderately noisy mutations.  Inspecting the
  current ledger
  (``planning/self_improve_ledger.jsonl``) shows every recent rejection
  cited *"lower CI bound … ≤ 0 — improvement not statistically
  distinguishable from noise"* even on iterations whose composite
  delta was clearly positive — the textbook symptom of an under-paired
  test.
* **Impact** — micro-benchmark on five reps where every after-rep
  solves 5 evals earlier than the matching before-rep on the same
  instance::

      paired:   Δ=+0.0500  CI=[+0.0500, +0.0500]  width=0.0000  → ACCEPT
      unpaired: Δ=+0.0500  CI=[−0.2100, +0.3300]  width=0.5400  → REJECT

  Same data, same point delta — paired collapses the CI to a point and
  unblocks acceptance of the genuine improvement; unpaired stays
  several standard errors wide because each side's bootstrap shuffles
  its reps independently.  In the regime the loop actually operates in
  (5 reps × ~3 problems at quick mode), the paired CI is typically
  3–10× narrower than the unpaired one, which is exactly the
  measurement gap the 0/6-accepts run on 2026-05-13 reflected.
* **Backwards compatibility** — strictly safe.  ``paired=None``
  (default) auto-selects: paired when at least one shared pair has
  matched rep counts, unpaired otherwise.  Existing CLI invocations,
  existing ledgers, and the asymmetric-rep edge cases the unpaired
  scheme was originally written to handle all keep their prior
  behaviour: the auto-detect rule degenerates to "unpaired" precisely
  when paired sampling cannot apply.  Existing tests in
  :mod:`tests.test_harness_stats` (22 pre-existing) all pass unchanged.
  ``StatisticalDecision.paired`` is a ``False``-defaulted field so old
  ledger consumers parsing the JSON payload continue to work and may
  ignore the new key.
* **Tests** — `tests/test_harness_stats.py` (11 new tests, total 33):
  paired-tighter-than-unpaired on correlated reps, paired unblocks a
  genuine improvement that unpaired rejects, auto-detect picks paired
  when rep counts match, auto-detect falls back to unpaired on
  mismatch, ``paired=True`` truncates to the common prefix, JSON
  round-trip of the new ``paired`` field, ``print_summary`` mentions
  the scheme, empty-pair edge case stays unpaired, paired bootstrap is
  reproducible with a fixed seed, and CLI integration covering
  ``--paired`` / ``--unpaired`` (acceptance flip and mutually-exclusive
  argparse).  `tests/test_self_improve.py` (2 new tests, total 126):
  ``LoopConfig.paired`` defaults to ``None`` and accepts explicit
  ``True`` / ``False``.
* **Documentation updated**
  - `doc/source/guide_benchmarking.rst`: new "Paired vs unpaired
    bootstrap" subsection under Statistical acceptance rule, with the
    scheme description, the worked numerical example, the CLI
    examples, and the auto-detect rule.
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §6.1 paragraph on the
    paired-vs-unpaired distinction, this §13 entry, and a
    "Next iteration ideas" graduation marker.
  - `AGENTS.md`: Statistical rigor section flags ``--paired`` /
    ``--unpaired`` and the auto-detect default.

### 2026-05-15 — jSO adaptive Differential Evolution (CEC 2017 winner)

* **What** — `panobbgo/heuristics/jso.py` adds the :class:`JSO` heuristic,
  a direct subclass of :class:`~panobbgo.heuristics.lshade.LSHADE` that
  ports the Brest-Maučec-Bošković (CEC 2017) "jSO" refinement.  jSO
  inherits the entire L-SHADE asynchronous pipeline (per-slot pending
  dict, generation-by-count book-keeping, archive of replaced parents,
  warm restart) and overrides three pieces of the trial-generation
  machinery:

  * **Weighted current-to-pbest mutation** (``current-to-pbest-w/1``).
    The pbest direction is re-weighted by a phase-dependent factor
    ``F_w`` that grows with progress: ``0.7·F`` while ``progress < 0.2``,
    ``0.8·F`` while ``progress < 0.4``, ``1.2·F`` afterwards.  The
    differential ``F · (x_r1 − x_r2)`` term keeps the unweighted
    scaling.  Asynchronous progress is measured the same way LPSR
    measures it: ``len(strategy.results) / max_eval`` clipped to ``[0, 1]``.
  * **Linear ``p_best`` schedule**.  ``p_best`` decreases linearly from
    ``p_best_max = 0.25`` to ``p_best_min = 0.125`` over the budget.
    Early-run mutations draw from a broader top slice; once LPSR has
    shrunk the population, the top 12.5% is enough to focus on the
    leading basin.
  * **Cauchy-F clamping**.  When ``progress < 0.6``, sampled ``F``
    values above ``0.7`` are clamped to ``0.7``.  Prevents
    pathologically large jumps when the population is still big.

  Plus two memory tweaks Brest et al. measured to give better
  early-run behaviour across the CEC battery:

  * **Initial memory values** ``M_F = 0.3`` / ``M_CR = 0.8``
    (vs L-SHADE's ``0.5`` / ``0.5``).
  * **Frozen anchor bin**.  The last memory bin (``H − 1``) is permanently
    pinned at ``M_F = M_CR = 0.9``.  ``_update_memory`` advances the
    pointer through ``[0, H − 2]`` only — the anchor bin is still drawn
    from at sampling time so it stably contributes a "moderately greedy"
    parameter setting regardless of what the live success-history has
    learned.

  The heuristic is registered in :mod:`panobbgo.heuristics`,
  :func:`default_structural_catalog` gains it as a twelfth
  ``add_heuristic`` candidate (``avoid_duplicates=True`` keeps the
  catalog from cluttering portfolios that already include it), and
  :func:`default_catalog` gains two kwarg rules so the loop driver
  can also retune ``JSO.NP_init`` and ``JSO.p_best_max`` once a spec
  opts in.
* **Why** — closes the *iLSHADE / jSO* L-SHADE follow-up below.  jSO
  is the **CEC-2017 single-objective bound-constrained competition
  winner** and remains a high-water mark for adaptive DE variants:
  every CEC winner since (jDE100, NL-SHADE-RSP, etc.) cites jSO as
  their direct ancestor and most differ from it only in
  archive-handling or rank-based selection refinements.  Subclassing
  L-SHADE keeps the *new* heuristic at the literature-best frontier
  while leaving the original L-SHADE byte-identical for ledger
  reproducibility — exactly the precedent set by the L-SHADE entry
  itself, which kept the basic ``DE/rand/1/bin`` heuristic available
  alongside.  Adding jSO to the structural catalog gives the
  self-improvement loop a third DE-family arm (basic DE, L-SHADE,
  jSO) the bandit can pick whichever wins on the current battery.
* **Asynchronous adaptation** — identical to L-SHADE.  jSO inherits
  the per-slot pending dict, generation-by-count update cadence,
  archive trimming, LPSR shrinking, and warm restart unchanged.
  The only async-relevant change is the use of ``_progress()`` (the
  same idiom L-SHADE uses for LPSR pacing) inside the F-clamp,
  ``F_w`` schedule, and ``p_best`` schedule — so the three jSO
  schedules stay in lock-step with the population shrink.  When
  ``max_eval`` is unknown the schedules degrade to ``progress = 0.0``
  (early-phase regime), matching L-SHADE's "no budget → no LPSR"
  fallback.
* **Impact** — A/B against L-SHADE in the same Rewarding strategy
  (Random + Nearby + Center + NelderMead + DE-arm), at quick mode
  (3 problems × 5 reps × 300 evaluations):

  * Seed 42 — ``Rewarding_LSHADE`` 0.791 / ``Rewarding_JSO`` **0.856**
    (mean **+0.065**).  Rosenbrock pair: 0.374 → **0.568** (success
    rate **40% → 80%**).  DeJong / Rastrigin tied at perfect.
  * Seed 43 — ``Rewarding_LSHADE`` **0.831** / ``Rewarding_JSO`` 0.801
    (mean -0.030).  Rosenbrock pair: **0.495 → 0.404** (both 60%
    success rate; LSHADE earlier ERT).

  Each variant wins on one of the two seeds — exactly the
  *complementarity* that motivates carrying both in the structural
  catalog.  The +0.194 spike on Rosenbrock seed 42 demonstrates the
  property the literature predicts: jSO's weighted mutation term
  navigates the curved Rosenbrock valley faster than fixed-weight
  ``current-to-pbest/1``, but at quick budgets (300 evals) the win
  is seed-dependent.  Adding jSO to the catalog gives the
  self-improvement loop a CEC-2017-class DE arm the bandit can swap
  in on a per-problem basis once it has gathered evidence.
* **Backwards compatibility** — strictly safe.  jSO is opt-in: it is
  not added to any default :func:`_make_quick_strategies` /
  :func:`_make_standard_strategies` / :func:`_make_full_strategies`
  spec, so existing CLI invocations and existing ledgers stay
  byte-identical.  The structural catalog gains it as one extra
  ``add_heuristic`` candidate; ``avoid_duplicates=True`` keeps the
  catalog from cluttering a portfolio that already has it.  The
  kwarg rules only fire when a spec explicitly sets ``NP_init`` /
  ``p_best_max`` (per :func:`_find_targets`'s "param already in
  kwargs" predicate), so a fresh ledger run on the built-in
  factories sees no behavioural change.  L-SHADE itself is
  untouched — jSO is a *new* class.
* **Tests** — `tests/test_heuristic_jso.py` (33 tests):
  construction validation (8 — defaults match Brest 2017, custom
  kwargs, subclass invariant, H must be ≥ 2 for the anchor bin
  separation, p_best_max bounds, p_best_min bounds, ordering rule
  ``p_best_min <= p_best_max``), memory anchor invariants (5 —
  anchor frozen at construction, never written by ``_update_memory``
  even after many cycles, pointer wraps over ``[0, H − 2]`` only,
  writable bin updated via Lehmer mean, no-success leaves memory
  unchanged), schedule helpers (5 — progress clipped, fallback to
  zero without budget, linear p_best schedule, three-phase F_w
  schedule, phase-boundary inclusivity), Cauchy-F clamping (3 —
  clamped at 0.7 in early phase, unclamped in late phase, F always
  in (0, 1]), initial population emission (4 — NP_init points,
  on_start re-stamps jSO defaults, NaN F/CR, points inside box),
  generate-trial path (2 — evolutionary trials emitted post-fill,
  better trial wins and archives parent), restart behaviour (3 —
  re-stamps jSO memory, ``center=None`` random fallback,
  before-start no-op), end-to-end smoke convergence on a quadratic,
  and registration tests (3 — package re-export, structural catalog
  candidate pool, kwarg rules present in default catalog).

### 2026-05-13 — Categorical mutation rule (`categorical_choice`)

* **What** — `panobbgo/self_improve.py`:
  :class:`MutationRule` gains a fourth ``kind`` value
  ``"categorical_choice"`` plus a ``choices: Tuple[Any, ...]`` field.
  A categorical proposal picks uniformly from ``choices`` *excluding*
  the current value so the mutation always proposes a real change
  (no-op samples are eliminated by construction).  ``bounds`` is
  ignored for the categorical kind and now defaults to ``(0.0, 0.0)``
  so callers no longer need to invent a placeholder.  ``__post_init__``
  validates the choice set (``len(choices) >= 2``, no duplicates).
  The :class:`MutationCatalog` / :func:`apply_mutation` /
  :class:`AdaptiveMutationSampler` paths are dispatch-by-kind already,
  so the new kind plugs in without touching the proposal / ledger /
  bandit machinery: a categorical mutation rides through
  :meth:`MutationProposal.to_dict` byte-identically to a numeric one,
  and :func:`_proposal_rule_key` puts it on its own
  ``(class_name, param_name, "categorical_choice")`` bandit arm —
  distinct from any numeric rule on the same kwarg slot.
  :func:`default_catalog` gains three categorical rules:
  ``PSO.topology`` (``"gbest"`` ↔ ``"lbest"``), ``Sobol.scramble``
  (``True`` ↔ ``False``), and ``LSHADE.archive_factor``
  (``0.0`` / ``1.0`` / ``2.6``).  Each fires only when a spec sets the
  matching kwarg explicitly — :func:`_find_targets`'s existing
  "param already in kwargs" predicate keeps the rule from injecting
  itself into specs that never opted in.
* **Why** — closes the *categorical mutation rule* item that the PSO
  follow-ups (2026-05-07 entry) and the L-SHADE follow-ups
  (2026-05-10 entry) both name as a blocker.  The shipped
  :class:`MutationRule` only supported numeric perturbations
  (``log_uniform_perturb`` / ``integer_add`` / ``float_uniform``) so
  the loop had no vocabulary for discrete design choices — it could
  *tune* ``PSO.NP`` but not *flip* ``PSO.topology``; it could tune
  ``Sobol.n`` but not flip ``Sobol.scramble``; it could tune
  ``LSHADE.NP_init`` but not toggle ``LSHADE.archive_factor`` between
  the archive-on and archive-off regimes.  Adding the categorical kind
  is one self-contained piece of infrastructure that unlocks three
  distinct loop capabilities at once, and matches the long-running
  "graduate one infra ticket into a dated entry once shipped" pattern
  in §13.
* **Impact** — applied to the standard battery
  (``_make_standard_strategies``):

  * ``BayesOpt_Sobol`` already sets ``scramble=True`` explicitly, so
    the ``Sobol.scramble`` categorical rule fires out-of-the-box —
    the loop can now decide whether Owen scrambling helps on the
    sampled instance distribution.
  * ``PSO.topology`` fires whenever the structural catalog has added
    the ``lbest`` PSO variant (``{"NP": 20, "topology": "lbest",
    "k_neighbors": 2}``), enabling the loop to flip the topology of
    an existing PSO without dropping and re-adding it.
  * ``LSHADE.archive_factor`` is dormant on the default battery (no
    spec sets ``archive_factor`` explicitly) but ready for any future
    spec that opts in — a clean wire-up rather than dead code.
* **Backwards compatibility** — strictly safe.  ``bounds`` retains
  its prior meaning for the three numeric kinds and now has a default
  ``(0.0, 0.0)`` that no existing call site relies on: every shipped
  catalog rule passes ``bounds`` explicitly, every test fixture passes
  ``bounds`` explicitly, and the dataclass field order is unchanged
  modulo the new defaulted ``choices`` slot.  Categorical mutations
  serialise to the ledger via the existing
  :meth:`MutationProposal.to_dict` path — ``rule_kind`` is the string
  ``"categorical_choice"``, ``old_value`` / ``new_value`` are the
  literal categorical values (strings / bools / floats), and a
  replay through :func:`_proposal_rule_key` recovers the bandit arm
  losslessly.  Existing ledger consumers parsing only numeric
  ``rule_kind``s simply see one extra kind they may ignore.
* **Tests** — `tests/test_self_improve.py` (13 new tests, total 122):
  rule validation (kind accepted, two-choice minimum, duplicate
  rejection, empty choices rejected, bounds ignored), catalog sampling
  (always-different value, two-way toggle, out-of-set drift handling,
  rationale formatting, default-catalog membership), apply path
  (string round-trip, bool round-trip preserves ``isinstance(bool)``),
  and bandit integration (categorical arm distinct from numeric arm
  on the same slot, ``_proposal_rule_key`` mapping).

### 2026-05-12 — COBYQA derivative-free trust-region local optimizer

* **What** — `panobbgo/heuristics/cobyqa.py` adds the
  :class:`COBYQA` heuristic, a subprocess-backed adapter around
  `scipy.optimize.minimize(method="COBYQA")`.  COBYQA
  (*Constrained Optimization BY Quadratic Approximations*,
  Ragonneau-Zhang 2023) is the modern Powell-family successor to
  BOBYQA / COBYLA / NEWUOA / LINCOA.  Like BOBYQA it maintains an
  interpolation set of ``2·n + 1`` points and fits an adaptive
  *quadratic model* of the objective inside a trust region; like
  LINCOA / COBYLA it natively supports bounds and linear / nonlinear
  constraints.  The asynchronous wrapping pattern mirrors
  :class:`~panobbgo.heuristics.lbfgsb.LBFGSB`: a daemon ``spawn``
  subprocess drives the synchronous COBYQA solver, requests
  ``f(x)`` over a pipe, and the main thread relays the projected
  point through Panobbgo's evaluator and pipes the penalty value
  back.  Constraint handling delegates to
  ``strategy.constraint_handler.get_penalty_value`` so COBYQA "sees"
  a smooth penalty objective even when raw constraints are
  non-smooth.  ``on_restart(center, reason)`` tears down the
  subprocess and respawns it at the clipped suggested center —
  matching :class:`~panobbgo.heuristics.lbfgsb.LBFGSB`'s warm
  restart pattern.  Initial trust-region radius auto-resolves to
  ``0.1 · max(box_width)`` when the user does not pin it; final
  radius defaults to ``1e-6`` (scipy's COBYQA library default).
  ``scale=True`` (default) maps the box to ``[-1, 1]`` so the
  interpolation geometry stays well-conditioned on boxes whose
  axes span very different magnitudes.
  :func:`default_structural_catalog` gains it as an eleventh
  ``add_heuristic`` candidate (``avoid_duplicates=True`` keeps the
  catalog from cluttering portfolios that already include it), and
  :func:`default_catalog` gains two kwarg rules so the loop driver
  can also retune ``COBYQA.initial_tr_radius`` (log-uniform around
  ``0.1`` in ``[0.01, 1.0]``) and ``COBYQA.final_tr_radius``
  (log-uniform in ``[1e-8, 1e-4]``) once a spec opts in.
* **Why** — closes the *BOBYQA / NEWUOA local optimizer* follow-up
  below.  Before this entry, :class:`~panobbgo.heuristics.nelder_mead.NelderMead`
  was the *only* generic derivative-free local refinement step in
  the portfolio; :class:`~panobbgo.heuristics.lbfgsb.LBFGSB`
  requires a finite-difference gradient approximation that breaks
  on noisy objectives, and Nelder-Mead's simplex updates are not
  curvature-aware, so it converges slowly on ill-conditioned
  valleys (Rosenbrock-like landscapes).  COBYQA fills the gap with
  a *derivative-free **and** curvature-aware* local refinement
  step.  Picking COBYQA over the older BOBYQA library (which would
  have required adding ``Py-BOBYQA`` as a new dependency) keeps
  the dependency surface unchanged — COBYQA ships as a built-in
  method of ``scipy.optimize.minimize`` since scipy 1.14 and is
  the literature-recommended replacement going forward.
* **Asynchronous adapter** — synchronous COBYQA calls a Python
  callable ``f(x)`` and blocks on the return value.  We host it in
  a dedicated subprocess (``spawn`` context, matching
  :class:`~panobbgo.heuristics.lbfgsb.LBFGSB`) and pipe the
  request / response between the solver and Panobbgo's
  event-driven main thread.  ``Heuristic.cap`` is fixed to ``1``
  because COBYQA has at most one outstanding evaluation at a time
  — the subprocess blocks until the previous return value
  arrives.  Out-of-bounds proposals are projected by
  ``problem.project`` before being emitted; the value sent back to
  COBYQA is therefore the objective at the projected (feasible)
  point.  Pipe-closed events (parent ``__stop__`` or termination)
  raise ``SystemExit`` inside the worker so it exits cleanly
  without hanging.
* **Impact** — quick A/B at ``--quick`` (3 problems × 3 reps × 75
  evaluations), comparing the same Rewarding strategy with NelderMead,
  COBYQA, or both as the local optimizer:

  * Seed 42 — ``NM`` 0.665 / ``COBYQA`` **0.769** (+0.104) /
    ``NM+COBYQA`` 0.699.  Rosenbrock success rate jumps from
    **0/3 with NM** to **2/3 with COBYQA**.
  * Seed 43 — ``NM`` **0.864** / ``COBYQA`` 0.714 / ``NM+COBYQA``
    0.753.  NM happens to win Rosenbrock on this seed.

  Each local optimizer wins on one of the two seeds — exactly the
  *complementarity* the literature predicts.  At ``--quick`` noise
  the average is comparable, but the *Rosenbrock success rate
  upgrade* from 0/3 → 2/3 on seed 42 demonstrates the property
  that motivates the addition: COBYQA's curvature-aware quadratic
  model lets it cross Rosenbrock's narrow curved valley that
  Nelder-Mead's simplex updates miss.  Adding COBYQA to the
  structural catalog gives the self-improvement loop a second
  derivative-free local arm the bandit can pick whichever wins on
  the current battery.
* **Backwards compatibility** — strictly safe.  COBYQA is opt-in:
  it is not added to any default :func:`_make_quick_strategies` /
  :func:`_make_standard_strategies` / :func:`_make_full_strategies`
  spec, so existing CLI invocations and existing ledgers stay
  byte-identical.  The structural catalog gains it as one extra
  ``add_heuristic`` candidate; ``avoid_duplicates=True`` keeps the
  catalog from cluttering a portfolio that already has it.  The
  kwarg rules only fire when a spec explicitly sets the matching
  kwarg (per :func:`_find_targets`'s "param already in kwargs"
  predicate), so a fresh ledger run on the built-in factories
  sees no behavioural change.
* **Tests** — `tests/test_heuristic_cobyqa.py` (30 tests):
  construction validation (11 — invalid initial_tr_radius / zero /
  negative / NaN, invalid final_tr_radius / zero / negative / inf,
  ordering rule final < initial, invalid maxfev type / zero /
  negative + default + custom), initial-TR auto-resolution (4 —
  box-width derivation, user override, final-floor invariant,
  zero-width box fallback), subprocess lifecycle (2 — start spawns
  daemon process, stop force-kills if join times out), pipe wiring
  (4 — penalty value routed, foreign-who ignored, on_start exits
  on pipe close, on_start logs subprocess output), restart
  behaviour (4 — relaunches subprocess, ``center=None`` uses box
  centre, out-of-box centre is clipped, stopped-state no-op),
  registration (3 — package re-export, structural catalog
  candidate pool, kwarg rules present in default catalog), and
  a smoke test exercising scipy COBYQA directly on a quadratic.

### 2026-05-10 — L-SHADE adaptive Differential Evolution

* **What** — `panobbgo/heuristics/lshade.py` adds the
  :class:`LSHADE` heuristic, an asynchronous port of L-SHADE
  (Tanabe & Fukunaga, CEC 2014).  Like
  :class:`~panobbgo.heuristics.differential_evolution.DifferentialEvolution`
  it maintains a population and competes trial vectors against
  targets, but unlike basic DE/rand/1/bin every trial draws its
  own ``(F_i, CR_i)`` from per-bin Cauchy / Normal memories which
  update via the **weighted Lehmer mean** of successful triples
  each "generation" (``NP_current`` completed evolutionary
  trials).  Mutation switches to ``current-to-pbest/1``
  (Zhang-Sanderson 2009) with an external archive of replaced
  parents.  **Linear Population Size Reduction** shrinks the
  population from ``NP_init`` (default 30) down to ``NP_min``
  (default 4) over the strategy's evaluation budget — the
  characteristic move that lifted SHADE to L-SHADE and won the
  CEC-2014 competition.  Out-of-bounds components are repaired by
  midpoint reflection per Tanabe-Fukunaga §III-A.  The heuristic
  is registered in :mod:`panobbgo.heuristics`,
  :func:`default_structural_catalog` gains it as a tenth
  ``add_heuristic`` candidate (``avoid_duplicates=True`` keeps the
  catalog from cluttering portfolios that already include it),
  and :func:`default_catalog` gains three kwarg rules so the
  loop driver can also retune ``LSHADE.NP_init``,
  ``LSHADE.H``, and ``LSHADE.p_best`` once a spec opts in.
  Warm restart via :meth:`on_restart` mirrors the IPOP / PSO
  pattern: in-flight trials dropped, archive cleared, memory
  bins reset to 0.5, slots re-randomised in a small ball around
  ``center``.
* **Why** — closes the *Adaptive Differential Evolution
  (LSHADE / JADE)* follow-up below.  The shipped DE was the
  basic ``DE/rand/1/bin`` with fixed ``F = 0.8`` and ``CR = 0.9``
  — robust, but conspicuously weaker than the literature-best
  population solvers.  L-SHADE is widely cited as one of the
  strongest single-population black-box optimizers — winner of
  the CEC-2014 single-objective competition and a high-water
  mark that subsequent variants
  (jSO, IMODE, NL-SHADE-RSP) merely refine.  Adding it as a
  *new* heuristic (not a replacement) keeps the legacy DE
  available for byte-identical reproduction of older ledgers
  while giving the structural mutation catalog a strong new
  candidate that can be combined with CMA-ES, PSO, and the
  GP-based heuristics in a portfolio strategy.
* **Asynchronous adaptation** — synchronous L-SHADE applies
  parameter adaptation only at the end of each generation,
  after every individual has been re-evaluated; this port
  batches by *count* — every ``NP_current`` completed
  evolutionary trials forms one async generation.  The weighted
  Lehmer mean used by SHADE is invariant under the order of its
  contributing samples, so the adaptation cadence stays the same
  while the heuristic plays nicely with Panobbgo's event loop.
  Initial random fills do not contribute to the success buffer
  (their F/CR are NaN), and slots dropped by LPSR drop their
  pending trials silently when results return.
* **Impact** — A/B at quick mode (3 problems × 5 reps × 300
  evaluations, seed 42), comparing the same Rewarding strategy
  with and without DE / LSHADE swapped in:

  * ``Rewarding_DE``     — DeJong 0.999 / Rosenbrock 0.517 /
    Rastrigin 1.000 (mean 0.839).
  * ``Rewarding_LSHADE`` — DeJong 1.000 / Rosenbrock **0.525** /
    Rastrigin 1.000 (mean **0.842**).

  At quick budget (300 evaluations) the two variants are within
  noise (delta +0.003) — exactly as expected, because LSHADE's
  success-history adaptation needs *more* evaluations than this
  to fully outclass fixed-parameter DE.  The literature
  comparisons that establish LSHADE as the CEC-2014 winner used
  10000+ evaluations on 30D/50D problems; the value of shipping
  it for Panobbgo today is to give the structural mutation
  catalog *a state-of-the-art DE arm* the bandit can swap in on
  a per-problem basis once the loop has gathered evidence.  At
  matching cheap budgets LSHADE is a peer of fixed DE, not a
  regression — exactly the property required for safely opting
  it in via the structural catalog.
* **Backwards compatibility** — strictly safe.  L-SHADE is
  opt-in: it is not added to any default
  :func:`_make_quick_strategies` /
  :func:`_make_standard_strategies` / :func:`_make_full_strategies`
  spec, so existing CLI invocations and existing ledgers stay
  byte-identical.  The structural catalog gains it as one extra
  ``add_heuristic`` candidate; ``avoid_duplicates=True`` keeps
  the catalog from cluttering a portfolio that already has it.
  The kwarg rules only fire when a spec explicitly sets the
  matching kwarg (per :func:`_find_targets`'s "param already in
  kwargs" predicate), so a fresh ledger run on the built-in
  factories sees no behavioural change.
* **Tests** — `tests/test_heuristic_lshade.py` (39 tests):
  construction validation (8 — invalid NP_init / NP_min /
  inversion / H / p_best / archive_factor + default + custom),
  initial-swarm emission and shape (3), unknown-who ignored,
  initial-fill no-success-counted, evolutionary trials emitted
  once population reaches 4, better-trial-wins (target replaced,
  parent archived, success recorded), worse-trial-loses (target
  unchanged, archive untouched), F/CR sampling invariants
  (F ∈ (0, 1], CR ∈ [0, 1], terminal CR sentinel), weighted
  Lehmer-mean memory update with hand-built buffer and known
  expected values, memory-pointer wrap, terminal-M_CR sentinel
  (planted on all-zero CR successes, sticky once set), LPSR
  invariants (no-op when budget unknown, full-budget shrink
  to NP_min, partial-progress proportional shrink, alive-index
  consistency post-drop), bound-reflection (below / above /
  in-bound) using the actual problem box, generation-counter
  isolation from initial fills, restart behaviour (state
  cleared / center=None random fallback / before-start no-op),
  end-to-end smoke run on a quadratic where the swarm makes
  measurable progress, plus registration tests for
  :mod:`panobbgo.heuristics` and the structural and kwarg
  catalogs.

### 2026-05-08 — Hold-out validation set for the self-improvement loop

* **What** — `panobbgo/self_improve.py`:
  :class:`LoopHoldoutRecord` (a third ledger record type next to
  :class:`LoopIterationRecord` and :class:`LoopGuardRecord`) plus the
  :attr:`LoopConfig.holdout_base_seed` /
  :attr:`LoopConfig.holdout_iterations` /
  :attr:`LoopConfig.holdout_iteration_offset` /
  :attr:`LoopConfig.holdout_eps_overfit` knobs and a new
  :meth:`LoopConfig.holdout_harness_config` helper.
  :class:`SelfImprover` gains :meth:`_holdout_enabled`,
  :meth:`_measure_holdout`, and :meth:`_run_holdout` plus a public
  :meth:`run_full` entrypoint that returns
  ``(iter_records, guard_records, holdout_records)`` for tests and
  callers that want the full audit trail.  The CLI gains
  ``--holdout-base-seed``, ``--holdout-iterations``,
  ``--holdout-iteration-offset``, ``--holdout-eps-overfit``, and
  ``--fail-on-overfit`` (exits ``3`` on a flagged ladder).  The
  ``summary`` subcommand now reports hold-out outcomes alongside
  iteration and guard summaries.
* **Why** — closes the Phase 6 / §10 *Hold-out validation set*
  ticket.  The anti-cherry-pick guard catches drift inside the
  *training* base_seed family — it varies only
  ``randomize_iteration`` and keeps ``HarnessConfig.seed`` constant.
  A mutation that overfits to peculiarities of the training base_seed
  family slips through silently because the guard's "fresh" instances
  are still drawn from the same SHA-256 stream.  The hold-out
  re-measures the seed and the final top of the ladder on a
  completely independent ``base_seed``, so an overfit ladder is
  exposed by a shrinking ``top − seed`` gap on hold-out.  A bias of
  ``drift < -eps_overfit`` is flagged ``overfit=True`` and, when
  combined with ``--fail-on-overfit``, exits the CLI non-zero so the
  signal is usable as an unattended-loop tripwire.
* **Independence vs the guard** — the guard validates within the
  training instance stream (same ``base_seed``, different
  ``randomize_iteration``); the hold-out validates *across* training
  streams (different ``base_seed``, same ``randomize_iteration``
  range).  Together they cover the two axes along which the loop can
  silently overfit.
* **Defaults** — ``holdout_base_seed = 0`` (disabled) keeps existing
  CLI invocations byte-identical.  When set, the value must differ
  from :attr:`LoopConfig.base_seed`; equal values would collapse the
  hold-out to a glorified guard check on offset ``0`` and the
  ``LoopConfig`` constructor rejects them at validation time.
  ``holdout_iterations = 5``, ``holdout_iteration_offset = 0``,
  ``holdout_eps_overfit = 0.05`` are the recommended starting points.
* **Skip rules** — hold-out is skipped silently when (a) disabled,
  (b) the loop ran zero iterations, or (c) ``randomize=False`` (the
  fixed battery is unaffected by ``base_seed``, so a hold-out check
  would be no signal at all).
* **Cost** — fixed: ``2 × holdout_iterations`` harness runs at the
  end of the loop (or just ``holdout_iterations`` when the ladder
  has only the seed, since both endpoints are the same spec list).
  Cheap relative to the ``2 × iterations`` cost of the main loop.
* **Tests** — `tests/test_self_improve.py` (17 new tests, total 97):
  config validation (negative iterations, negative eps, equal
  base_seed rejection, zero-zero edge case, `holdout_harness_config`
  vs `harness_config` propagation), end-to-end behaviour
  (disabled-by-default, skipped when randomize=False, skipped on
  zero iterations, seed-only ladder records zero drift, hold-out
  uses the independent base_seed for measurement, overfit flag fires
  when gap collapses, no flag when gap holds, ledger writes
  ``record_type='holdout'`` line), back-compat (`SelfImprover.run`
  still returns a list of `LoopIterationRecord`), and `to_dict`
  round-trip with JSON serialisation.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §2 missing-pieces list
    refreshed; §10 Open Questions item resolved; Phase 6 checklist
    updated; this §13 entry; Next iteration ideas reduced.
  - `doc/source/guide_benchmarking.rst`: new "Hold-out validation
    set" subsection with algorithm, CLI examples, programmatic
    example, and the independence-from-the-guard note.
  - `doc/source/guide.rst`: quick-nav entry mentions the hold-out.
  - `AGENTS.md`: self-improvement loop subsection lists the
    hold-out feature with run-the-loop bash example.
  - `TODO.md`: this entry.

### 2026-05-07 — PSO adaptive inertia (Shi-Eberhart 1998)

* **What** — `panobbgo/heuristics/pso.py`: :class:`PSO` gains an
  opt-in ``w_end`` keyword argument and a new ``_current_inertia()``
  helper.  When ``w_end`` is set, the inertia weight at evaluation
  count ``e`` (out of ``E = strategy.config.max_eval``) is
  ``w_eff(e) = w − (w − w_end) · min(e/E, 1)`` — the canonical
  Shi-Eberhart (1998) linearly-decreasing schedule.  When
  ``w_end is None`` (default) ``_current_inertia()`` returns
  ``self.w`` unchanged, preserving the original Clerc-Kennedy
  constriction-coefficient behaviour byte-for-byte.  When the
  strategy budget is unknown (no ``max_eval``, zero, or non-numeric)
  the heuristic falls back to constant ``w`` rather than guessing a
  horizon.  :func:`default_catalog` gains two new
  :class:`MutationRule`s (``PSO.w`` and ``PSO.w_end``, both
  ``float_uniform`` over literature-standard bounds) so the loop
  driver can tune the adaptive-inertia schedule once a spec opts in
  by setting either kwarg explicitly.
* **Why** — closes the *Adaptive inertia* PSO follow-up.  At the
  budgets used by competition-winning PSO variants (≥ 300
  evaluations per run), the canonical fixed Clerc-Kennedy parameters
  under-explore multimodal landscapes; Shi-Eberhart inertia
  annealing is the literature-standard fix.  The extension is
  *opt-in* — the default constructor preserves the shipped
  behaviour exactly — so the loop driver can discover whether any
  given strategy benefits without disturbing existing ledgers.
* **Backwards compatibility** — strictly safe.  ``w_end`` defaults to
  ``None``; existing PSO instances retain their prior behaviour
  bit-for-bit.  The new ``PSO.w`` / ``PSO.w_end`` catalog rules only
  fire when a spec explicitly sets the kwarg (per
  :func:`_find_targets`'s "param already in kwargs" predicate), so a
  fresh ledger run on the built-in factories sees no behavioural
  change.
* **Tests** — `tests/test_heuristic_pso.py` adds 6 tests:
  default ``w_end`` is ``None``; finiteness validation; constant-``w``
  short-circuit; missing-results fall-back path; the
  linearly-decreasing schedule at four progress points; the
  zero-``max_eval`` fall-back; plus a catalog test confirming
  ``PSO.w`` / ``PSO.w_end`` rules are present.

### 2026-05-07 — PSO ring (`lbest`) topology variant

* **What** — `panobbgo/heuristics/pso.py`: :class:`PSO` gains a
  ``topology: str = "gbest"`` argument plus a ``k_neighbors: int = 2``
  half-width.  ``"gbest"`` (default, byte-identical to the 2026-05-05
  ship) keeps the canonical Kennedy-Eberhart 1995 fully-connected
  swarm; ``"lbest"`` switches every particle's social attractor to the
  best ``pbest`` in a wrap-around *ring* of width ``2·k_neighbors + 1``
  centred on the particle's own index.  Two new helpers cover the
  bookkeeping: ``_ring_neighbors(i)`` returns the wrap-around index
  list and ``_social_best_idx(i)`` returns the per-particle attractor
  (collapsing to ``_gbest_idx`` for ``gbest``).  ``_generate_next``
  consults ``_social_best_idx`` exactly where it used ``_gbest_idx``
  before, so the velocity-update / clamp / projection paths are
  shared.  :func:`panobbgo.self_improve.default_structural_catalog`
  gains a second PSO entry — ``(PSO, {"NP": 20, "topology": "lbest",
  "k_neighbors": 2})`` — alongside the existing gbest default.  Both
  entries share ``cls = PSO`` so ``avoid_duplicates=True`` still
  prevents two PSO instances from landing in the same strategy; the
  catalog samples uniformly between them when PSO is not yet present
  and skips both afterwards.
* **Why** — closes the "Topology variants" follow-up below the §13
  PSO entry from 2026-05-05.  ``gbest`` and ``lbest`` topologies
  trade off different parts of the exploration / exploitation
  spectrum: ``gbest`` contracts faster (every particle sees the same
  best), ``lbest`` slows information diffusion to one hop per
  iteration so multiple sub-swarms can probe different basins in
  parallel.  Kennedy & Mendes (CEC 2002) show ``lbest`` empirically
  beats ``gbest`` on multimodal benchmarks — exactly the regime where
  Panobbgo's standard battery (Rastrigin, Ackley, Griewank,
  Schwefel) is concentrated.  Shipping both variants in the
  structural catalog gives the self-improvement loop the vocabulary
  to pick whichever wins on the current battery.
* **Impact** — 2-seed A/B at ``--quick`` (3 problems × 5 reps × 150
  evaluations), comparing the same Rewarding strategy with PSO under
  each topology:

  * Seed 42 — ``gbest`` 0.183 / ``lbest`` **0.288** (lbest +0.105).
  * Seed 43 — ``gbest`` **0.296** / ``lbest`` 0.181 (gbest +0.115).

  Each topology wins on one of the two seeds — exactly the
  *complementarity* the literature predicts.  At ``--quick`` noise
  (~ ±0.05) neither dominates, but adding ``lbest`` to the catalog
  expands the bandit's reachable strategy space without regressing
  the gbest path: the loop now has two PSO arms with markedly
  different exploration dynamics to choose between.
* **Backwards compatibility** — strictly safe.  ``topology`` defaults
  to ``"gbest"``, so every existing PSO instance retains its prior
  behaviour bit-for-bit.  The structural catalog gains one extra
  ``add_heuristic`` candidate that shares ``cls = PSO`` with the
  existing entry — under ``avoid_duplicates=True`` (default), only
  one is ever added per strategy.  Existing ledger consumers, kwarg
  rules (``MutationRule(class_name="PSO", ...)``), and the bandit's
  ``_proposal_rule_key`` are unchanged.
* **Tests** — `tests/test_heuristic_pso.py` (13 new tests, total
  50): construction validation (default topology / lbest
  construction / invalid topology / invalid k_neighbors type / value),
  ring-neighbour wrap-around correctness, ring size invariant, lbest
  social-attractor uses ring (not the global best), gbest social
  attractor degenerates to ``_gbest_idx``, lbest returns ``None``
  before any neighbour pbest exists, lbest velocity clamp invariant,
  lbest end-to-end smoke convergence on a quadratic, and structural
  catalog now ships both gbest and lbest PSO entries.

### 2026-05-05 — Particle Swarm Optimization (`PSO` heuristic)

* **What** — `panobbgo/heuristics/pso.py` adds an asynchronous PSO
  heuristic with the canonical Clerc–Kennedy (2002) constriction-
  coefficient parameters: ``w = χ ≈ 0.7298``, ``c1 = c2 ≈ 1.49618``.
  Each particle carries a position, velocity, and personal-best
  memory; on every step the velocity update::

      v_i ← w · v_i + c1·r1·(pbest_i − x_i) + c2·r2·(gbest − x_i)
      x_i ← x_i + v_i

  pulls the particle toward both its own best and the global best
  with random per-component weights.  Velocities are clamped per
  dimension to ``v_max_frac · range`` (default 0.5) to prevent the
  swarm from exploding outside the search box.  The heuristic is
  registered in :mod:`panobbgo.heuristics` and added to the
  ``add_heuristic`` candidate pool of
  :func:`default_structural_catalog`; a kwarg rule for ``PSO.NP``
  (swarm size, range ``[8, 60]`` with ±4 / ±8 deltas) is added to
  :func:`default_catalog` so the loop can also tune the swarm
  size.  ``on_restart(center, reason)`` implements an IPOP-style
  warm restart: drop in-flight trials, scatter particles in a
  velocity-clamp ball around the new center, wipe the global
  memory, and re-seed.
* **Why** — closes a clear gap in the heuristic portfolio.  PSO is
  the third great population-based metaheuristic alongside CMA-ES
  (covariance re-sampling) and Differential Evolution (recombination
  of three random members), but its dynamics are markedly different:
  particles carry **momentum** (velocity inertia retained from the
  prior step) and a **social** attraction toward the swarm's best,
  giving fast contraction once a basin is found while still probing
  along the prior search direction.  These dynamics are
  complementary to CMA-ES and DE — they exploit ridges with
  momentum that CMA-ES has to *learn* via covariance updates and
  that DE has no concept of at all — so adding PSO to the portfolio
  diversifies the heuristic mix the bandit can choose from on any
  given problem.
* **Impact** — quick A/B at ``--quick`` (3 problems × 3 reps × 75
  evaluations, seed 42), comparing the same Rewarding strategy with
  and without PSO appended to the heuristics list:

  * ``Rewarding_NoPSO``  — DeJong 1.000 / Rosenbrock 0.000 /
    Rastrigin 1.000 (mean 0.667).
  * ``Rewarding_WithPSO`` — DeJong 1.000 / Rosenbrock **0.031** /
    Rastrigin 1.000 (mean **0.677**).

  Adding PSO upgrades the Rosenbrock pair from 0/3 reps solved to
  2/3 reps solved (success rate 0% → 67%) without regressing on
  DeJong or Rastrigin.  Rosenbrock is exactly the regime where
  momentum helps — a narrow curved valley where vector inertia along
  the valley floor is more useful than the Gaussian re-sampling of
  Random / Nearby / NelderMead.  At the noisy ``--quick`` level a
  delta of ``+0.01`` is within noise; the meaningful signal is the
  per-pair upgrade on Rosenbrock.
* **Backwards compatibility** — strictly safe.  PSO is opt-in: it is
  not added to any default :func:`_make_quick_strategies` /
  :func:`_make_standard_strategies` / :func:`_make_full_strategies`
  spec, so existing CLI invocations and existing ledgers stay
  byte-identical.  The structural catalog gains it as one extra
  ``add_heuristic`` candidate; its ``avoid_duplicates=True`` invariant
  keeps the catalog from cluttering a portfolio that already has it.
* **Tests** — `tests/test_heuristic_pso.py` (24 tests):
  construction validation (8 — invalid NP / w / c1 / c2 / v_max_frac
  + default + custom + name), initial-swarm emission and shape (3),
  pbest / gbest update + follow-up trial (5), velocity clamp
  invariant (1), restart behaviour (3 — clears pbest, before-start
  no-op, ``center=None`` random fallback), an end-to-end smoke run
  on a quadratic where the swarm strictly improves, and registration
  tests for ``panobbgo.heuristics`` and the structural catalog.

### 2026-05-03 — Strategy portfolio composition (`StructuralMutationRule`)

* **What** — `panobbgo/self_improve.py`:
  :class:`StructuralMutationRule` joins :class:`MutationRule` as a
  first-class catalog rule.  Two ops:

  * ``add_heuristic`` appends one of ``candidate_classes`` (a
    ``(HeuristicClass, default_kwargs)`` pool) to a target strategy.
    ``avoid_duplicates=True`` (default) skips classes already present
    in the strategy so the catalog cannot clutter a portfolio with
    redundant copies of the same heuristic.
  * ``drop_heuristic`` removes one heuristic, optionally restricted to
    ``droppable_classes``.  ``min_heuristics`` (default ``2``) is the
    floor of the *post-drop* heuristic count, so the strategy always
    keeps a diversity slot.

  :class:`MutationProposal` gains ``op`` and ``structural_kwargs``
  fields that are populated only for structural ops; kwarg proposals
  serialise byte-identically to before.  :func:`apply_mutation`
  dispatches on ``proposal.op`` and falls through to the existing
  kwarg path for non-structural proposals.  The Thompson sampler maps
  every structural rule onto one arm per ``op``
  (``("*", op, "structural")``) which keeps cold-start variance bounded
  while still letting the bandit learn whether portfolio expansion or
  contraction wins on the current battery.
  :func:`default_structural_catalog` returns
  ``default_catalog().rules + [StructuralMutationRule(add), StructuralMutationRule(drop)]``
  so the existing ledger and CI defaults are unchanged — opt in via
  ``--structural`` on ``scripts/self_improve.py run`` or by passing
  the catalog explicitly to :class:`SelfImprover`.
* **Why** — closes the §7.2 *Strategy portfolio composition* item.  The
  loop driver shipped in Phase 5 only retunes existing kwargs, so it
  could discover better dial settings but never a better composition.
  Most measurable Panobbgo wins to date have come from composition
  changes (adding Sobol' for the BayesOpt initial design,
  splitting CMAES strategies into IPOP/BIPOP variants, etc.) — exactly
  the moves the loop now has the vocabulary to make autonomously.
* **Backwards compatibility** — strictly safe.  :func:`default_catalog`
  is unchanged; :class:`MutationProposal` keeps the same required
  fields and adds ``op`` / ``structural_kwargs`` as keyword-only with
  ``None`` defaults; :meth:`MutationProposal.to_dict` only emits the
  new keys when ``op`` is set, so existing ledger consumers parse the
  old layout byte-identically.  The bandit's
  :func:`_proposal_rule_key` collapses structural ops onto the
  ``("*", op, "structural")`` arm; kwarg keys are unchanged so
  prior-ledger priming still recovers identical statistics.
* **Tests** — `tests/test_self_improve.py` (29 new tests, total 92):
  rule validation, applicable-hits enumeration (add / drop /
  ``avoid_duplicates`` / ``droppable_classes`` / ``min_heuristics``
  floor / strategy_pattern filter), proposal serialisation, the
  apply-side dispatch (add appends, drop removes, missing class
  raises, empty-strategy refusal, fallback-import path),
  :func:`_proposal_rule_key` collapse for structural ops, the
  Thompson sampler bucketing structural history into one arm, and an
  end-to-end loop run that accepts a structural drop on a fake
  harness.
### 2026-05-02 — Stratified dimension sampling for multi-dim families

* **What** — `panobbgo/harness_randomized.py`:
  :class:`ProblemFamily` gains a ``stratify_dims: bool = True`` field and
  a :meth:`stratified_dim_for_rep` helper that returns
  ``dim_choices[rep % k]``.  :meth:`ProblemFamily.sample_instance` now
  accepts an optional ``dim`` override so callers can pin the dim
  without consuming the rng's ``choice`` slot.
  :meth:`RandomizedProblemSpec.create_problem_for_rep` calls
  ``stratified_dim_for_rep(rep)`` for multi-dim families with
  ``stratify_dims=True`` (the default) and falls back to the rng's
  ``choice`` otherwise.  ``last_sampled_params()`` now reports a
  ``stratified_dim: bool`` flag for ledger introspection.
* **Why** — closes the §10 *Composite score stability across dimension
  sampling* item.  Without stratification, a multi-dim family with
  ``dim_choices = (2, 5, 10)`` and 5 reps could draw, say, three
  ``dim=2`` instances on iteration 5 and three ``dim=10`` instances on
  iteration 6.  Higher-dim instances are systematically harder, so a
  per-iteration composite delta picks up dim-mix noise on top of the
  signal of the underlying mutation, polluting the bootstrap CI on
  which §6.2 acceptance depends.  Cyclic stratification (rep ``i`` →
  ``dim_choices[i % k]``) makes any contiguous block of ``k`` reps
  cover every declared dim exactly once, eliminating that noise source
  by construction without changing the per-iteration eval count.
* **Impact** — purely a measurement-noise improvement: the default
  battery of families all use ``dim_choices=(2,)`` (single dim), so
  this change is a no-op for the byte-level reproducibility of the
  current standard mode.  The benefit materialises when users (or the
  loop) declare multi-dim families — e.g. via
  ``HarnessConfig.extra_families`` — at which point the cross-iteration
  variance of the composite drops by roughly ``Var(dim_mix)`` (the
  fraction of total variance attributable to which dim was sampled,
  typically a substantial slice for hard families like Rosenbrock).
* **Backwards compatibility** — strictly safe.  Single-dim families
  (the entire default battery) are unaffected because the cyclic
  schedule degenerates to a constant.  The public :class:`ProblemFamily`
  signature gains a new keyword-only field with a default; existing
  ``ProblemFamily(...)`` callers keep working byte-identically.  The
  ``stratify_dims=False`` path preserves the previous behaviour for
  anyone who needs it (e.g. for replicating an old ledger).
* **Tests** — `tests/test_harness_randomized.py` (16 new tests, total
  68): cyclic schedule correctness, balance over a complete cycle,
  imbalance bound on partial cycles, single-dim no-op, dim-override
  validation, rng-stream invariance proof (override does not consume
  the choice slot), end-to-end :class:`RandomizedProblemSpec` round
  trip, ``last_sampled_params`` flag round trip, and the contract that
  default families remain unchanged.

### 2026-05-01 — Adaptive mutation sampler (Thompson sampling)

* **What** — `panobbgo/self_improve.py` gains
  `AdaptiveMutationSampler` plus `MutationRuleStats` and the public
  `RuleKey` alias.  Each :class:`MutationRule` becomes one arm of a
  Bernoulli bandit whose reward is "this iteration was accepted".  On
  every `sample()` call the sampler draws one variate per applicable
  rule from `Beta(prior_alpha + n_accepts, prior_beta + n_attempts -
  n_accepts)` and picks the arg-max — the canonical Thompson rule.
  Inside the chosen rule, hits are still selected uniformly (which
  spec / which slot), exactly as the catalog's uniform sampler does.
  History is primed from a prior JSONL ledger via
  `prime_from_ledger`, so the loop carries learning across restarts.
  `LoopConfig` gains `adaptive_sampling`, `adaptive_prior_alpha`,
  `adaptive_prior_beta`, `adaptive_prime_from_ledger`; the
  `scripts/self_improve.py` CLI gains the `--adaptive` family of
  flags.  After each iteration's accept/reject decision, the driver
  calls `sampler.record_outcome()` so future samples are biased
  toward rules with positive accept history.
* **Why** — closes the §10 "Adaptive mutation sampler" item.  The
  uniform catalog sampler shipped in Phase 5 wastes iterations on
  rules that never produce accepts.  Thompson sampling concentrates
  probability mass on empirically winning rules while still exploring
  unfamiliar ones — the canonical fix for the *productivity* gap of
  multi-armed bandit problems.  Cold-start equivalence to uniform
  (Beta(1, 1) ≡ U(0, 1), and arg-max of i.i.d. uniforms is uniform)
  makes the upgrade strictly safe: flipping the flag on a fresh
  ledger reproduces the prior behaviour distributionally, then
  diverges as evidence accumulates.
* **Defaults** — `adaptive_sampling = False` keeps existing CLI
  invocations byte-identical.  `adaptive_prior_alpha = adaptive_prior_beta
  = 1.0` is the symmetric uninformed prior; lower priors (e.g. `0.5`)
  make the sampler greedier earlier at the cost of more variance.
* **Tests** — `tests/test_self_improve.py` (23 new tests, total 63):
  invalid priors, cold-start equivalence to uniform sampling,
  arg-max behaviour after biased training, record-outcome
  correctness, ledger priming, integration with `SelfImprover`
  including the `sampler=` override and the `adaptive_prime_from_ledger`
  flag.

### 2026-04-27 — Sobol' quasi-random initial design (`Sobol` heuristic)

* **What** — `panobbgo/heuristics/sobol.py` adds a low-discrepancy
  quasi-random (Sobol') sampler as a one-shot space-filling heuristic;
  registered alongside `LatinHypercube`, `Random`, etc.  A new
  `BayesOpt_Sobol` strategy in the standard harness pairs it with the GP
  surrogate, `Nearby`, and `NelderMead`.  The mutation catalog
  (`panobbgo.self_improve.default_catalog`) gains a rule that nudges
  `Sobol.n` in 4-step increments inside `[4, 64]` so the loop driver can
  also tune it.
* **Why** — every modern Bayesian-optimization library (BoTorch, TuRBO,
  scikit-optimize, GPyOpt) defaults to Sobol' for the initial design
  precisely because lower discrepancy → better surrogate fits at low
  sample counts.  Panobbgo only had Latin Hypercube before.
* **Impact** — measured head-to-head over 5 reps × 7 standard problems at
  budget 200, mean per-pair score `BayesOpt_Sobol = 0.314` vs
  `BayesOpt_GP = 0.191` (`+0.123`).  Sobol' wins on 5 / 7 problems
  (DeJong, Rosenbrock_2D, Ackley, StyblinskiTang, Griewank tied with
  smaller best-distance), loses on 2 (Rastrigin, Rosenbrock_5D).
* **Tests** — `tests/test_heuristic_sobol.py` (16 tests).

### 2026-04-26 — Anti-cherry-pick guard + tests for the loop driver

* **What** — `panobbgo/self_improve.py` gains
  `LoopConfig.guard_interval`, `guard_eps_ladder`, and
  `guard_iteration_offset` plus the `LadderEntry` and `LoopGuardRecord`
  data structures.  Every `guard_interval` iterations the loop
  re-measures the top of the accepted ladder on a *fresh*
  `randomize_iteration` (`iteration + guard_iteration_offset`) and rolls
  the ladder back when the composite has drifted more than
  `guard_eps_ladder` below the entry's stored `last_validated_score`.
  The seed entry is the trusted fallback and is never popped.  Exposed
  via `--guard-interval` / `--guard-eps-ladder` /
  `--guard-iteration-offset` on `scripts/self_improve.py run` and the
  `summary` subcommand reports rollbacks.
* **Why** — closes §6.3 ("Anti-cherry-pick guard") of this plan.  Even
  with the parametrically randomized battery, a sequence of "lucky"
  instance draws can inflate per-iteration after-scores enough to clear
  the bootstrap CI even when the underlying mutation does not
  generalise.  The guard validates the ladder against an independent
  instance stream so silent overfitting cannot accumulate.
* **Tests** — `tests/test_self_improve.py` (40 tests, new) — also fills
  the test gap left by Phase 5 (the loop driver shipped without
  coverage).  Covers `MutationRule` validation, catalog sampling, the
  `apply_mutation` immutability contract, end-to-end runs against a
  faked harness, the guard's cadence / no-rollback / drift-rollback /
  offset-id / seed-not-popped invariants, and ledger round-trip.
* **Defaults** — `guard_interval = 0` keeps existing CLI invocations
  byte-identical.  `5` or `10` is the suggested setting for unattended
  multi-hour runs.

### Next iteration ideas

Lightweight "next ticket" notes for follow-up agents — graduate them to
a dated entry above when shipped.

> **Before implementing any idea below, run `gh pr list --state open`
> (drafts included).** These notes live on `master` and do not reflect
> work that is already sitting in an unmerged PR. If a candidate is
> already covered by an open PR, finish/merge that PR instead of opening
> a duplicate — see §12.3 step 0. (Four duplicate NL-SHADE-RSP PRs,
> #227–#230, were the cost of skipping this check.)

#### `codify-scan --open-pr` driver (after 2026-06-17 ship)

The 2026-06-17 ship landed the *detection* half of V2 §9.3 (the
``codify-scan`` subcommand surfaces candidates as text / JSON).  The
queued *write* half is the ``--open-pr`` flag that translates each
surfaced :class:`CodifyCandidate` into a concrete source edit + draft
PR.  Sketch:

1. **Dedup pass** — ``gh pr list --state open --json title,headRefName``,
   parse each open PR for a known "codify ``Class.param``" marker
   either in the title or via a label, and skip any candidate whose
   :attr:`CodifyCandidate.slot_key` already has an open PR.  Matches
   the §12.3 step 0 lesson (the four duplicate NL-SHADE-RSP PRs
   #227–#230) — enforced in code rather than left to operator memory.
2. **Source-edit primitive** — for numeric / categorical candidates the
   edit is on the heuristic constructor's keyword default (e.g.
   ``Sobol.__init__(n=16, …)`` → ``n=12``) or on the seed-spec
   factory (``_make_quick_strategies`` / ``_make_loop_strategies``
   already passes the kwarg explicitly).  A small "where does this
   kwarg get set" lookup table can be derived from the catalog
   strategy_pattern + class_name + the AST of the factories, then
   the edit applied with the existing ``ruff format`` pipeline so
   diff hygiene is preserved.  For structural ops the edit is
   "append this heuristic class to the seed pool" / "drop this
   class from the seed pool" — same factory locations.
3. **PR body** — populate from
   :meth:`CodifyCandidate.to_dict` so the ledger evidence
   (timestamps, deltas, CIs, per-record old → new) lands in the PR
   body for review.  Add a "test plan" stub linking to the
   benchmark-harness ``compare --statistical`` invocation the
   reviewer should run.
4. **Open as draft** — every codify PR opens as ``--draft`` so the
   reviewer can decide whether to mark it ready or close it.  Match
   the existing nightly-loop branch naming
   (``claude/funny-*-*``) so the existing watcher infrastructure
   picks them up.

Speculative until the detection ship's first ledger evidence shows
the candidate set converges (i.e. the same Nearby.radius / Sobol.n
patterns keep surfacing across nights without an actionable PR
landing).  Pairs naturally with **mutation-bound widening** for the
bidirectional candidates the detection scan already surfaces — the
right action on those is rarely a default shift.

#### Mutation-bound widening rule for bidirectional codify candidates — shipped 2026-06-19

Shipped 2026-06-19 as :class:`panobbgo.self_improve.WideningCandidate`
plus :func:`panobbgo.self_improve.detect_widening_candidates` and the
``codify-scan --widen-bounds`` / ``--widen-factor`` CLI flag pair.
The detector pairs every bidirectional ``(class_name, param_name)``
slot — same slot with accepts in *both* ``"up"`` and ``"down"``
directions — into a proposed ``MutationRule.bounds`` update.  On the
live project ledger today, this surfaces two actionable patterns:
``Nearby.radius`` ([0.073, 0.135] observed, proposed [0.049, 0.203]
— tightens current [0.005, 0.5]) and ``Sobol.n`` ([8, 24] observed,
proposed [5, 36] — tightens current [4, 64]).  See the 2026-06-19
dated entry above for the full rationale, the per-rule-kind bound
arithmetic (multiplicative for log / float; outward-rounded for
integer with a lower-bound clip at 1 for positive values), and the
backwards-compat / test coverage.

Follow-ups still queued:

* **``codify-scan --widen-bounds --open-pr``** — extend the queued
  ``--open-pr`` driver to translate each surfaced
  :class:`WideningCandidate` into a concrete edit on
  :func:`~panobbgo.self_improve.default_catalog` and open a draft
  codify PR.  Speculative until the basic ``--open-pr`` driver
  lands.
* **Per-kind widen factor** — log-scale knobs tolerate a larger
  widen factor than linear ones; a
  ``--widen-factor-log`` / ``--widen-factor-linear`` flag pair would
  let the operator tune per kind.  Speculative.
* ~**Auto-tune widen factor from observed spread**~ — **shipped
  2026-06-22** as :func:`panobbgo.self_improve._auto_tune_widen_factor`
  plus the ``auto_tune`` / ``auto_tune_min_factor`` /
  ``auto_tune_max_factor`` keyword arguments on
  :func:`detect_widening_candidates` and the ``--widen-auto-tune`` /
  ``--widen-factor-min`` / ``--widen-factor-max`` CLI flags.  Narrow
  observed spread → larger factor (default max ``2.5``); wide spread
  → smaller factor (default min ``1.1``); linearly interpolated by
  the relative-spread ratio measured in the rule's natural scale
  (log for log_uniform_perturb, linear for integer_add / float_uniform).
  Lifts the live ``Nearby.radius`` widen factor from a fixed 1.5 to
  ~2.31 (proposed bound ``[0.0317, 0.3130]`` vs ``[0.0489, 0.2030]``)
  — directly closes the *Auto-tune widen factor from observed
  spread* idea seeded in the 2026-06-19 widening-detector ship.  See
  the 2026-06-22 dated entry above.

#### Suppress already-codified candidates in codify-scan — shipped 2026-06-18

Shipped 2026-06-18 as
:func:`panobbgo.self_improve.annotate_codified_status` plus the
:func:`~panobbgo.self_improve.default_codify_registries` helper, two
new fields on :class:`~panobbgo.self_improve.CodifyCandidate`
(``already_codified`` / ``live_codified_values``), and a
``--include-already-codified`` (alias ``--no-suppress-codified``) CLI
flag on ``scripts/self_improve.py codify-scan``.  The scanner
imports the seed-spec factories the nightly cron exercises
(``_make_quick_strategies`` + ``_make_loop_strategies``), walks every
:class:`~panobbgo.benchmark.StrategySpec`'s ``(class, kwargs)``
entries, and cross-checks each candidate's predicted edit against
the live values.  Suppresses by default; the daily routine's report
on the live project ledger shrinks from 5 to 4 candidates (the
``Sobol.scramble = False`` example the entry was seeded for).  See
the 2026-06-18 entry above for the full rationale and follow-ups.

Follow-ups still queued:

* **Structural-op codified check** — extend the placeholder
  :func:`_structural_already_codified` to compare ``add_X`` /
  ``drop_X`` candidates against the heuristic-pool membership of
  the seed factories.  ``add_LBFGSB`` against a seed pool that
  already contains :class:`LBFGSB` is the symmetric case.  Lower
  priority because structural candidates are rarer in the live
  ledger today.
* **Tolerance / hysteresis on the numeric predicate** — the
  current ``max(live) >= median(new_values)`` rule is exact; a
  small relative tolerance (e.g. 5%) would let the predicate
  catch cases where the live default is *very close* to the
  median proposal without being strictly above / below.
  Speculative — the exact rule already catches the dominant
  ``Sobol.scramble`` shape.

#### Flip the nightly cron to `--confirm-accepts` (after 2026-06-14 ship)

The same-night confirmation gate shipped 2026-06-14 as
:attr:`LoopConfig.confirm_accepts` / ``--confirm-accepts``.  The
2026-06-21 V2 §9.5 step 5 partial flip (see the dated entry above)
promoted every no-cost V2 flag — ``--registry loop`` /
``--prime-include-archives`` / ``--structural-per-class-arms`` /
``--bandit-reward graded`` / ``--inactivity-relax-after 10`` /
``--holdout-base-seeds 7,1234`` / ``--guard-interval 10`` — into the
nightly cron but intentionally **held back** on ``--confirm-accepts``
because it's the only V2 flag with a meaningful per-iteration cost
(2× the screening cost, plus 1× per hold-out leg → ~2-3× total).
Until the nightly workflow file passes the flag to
``scripts/self_improve.py run`` the cron still operates in
*promote-on-screening* mode and the §2.2 "Accept → rollback churn"
symptom persists in the live loop (15/16 V1 accepts rolled back by
the guard, per the original diagnosis).  At quick-mode budgets — where
the V1 §2.5 diagnosis reports 94% idle compute even after the
2026-06-21 ``--registry loop`` flip — the 2-3× headroom is
comfortably within the 90-min cap, so the iteration count probably
does *not* need halving.  The workflow file edit is:

* Pass ``--confirm-accepts`` to ``scripts/self_improve.py run``.
* Halve the iteration count *only if* the post-flip nightly runs
  show wall-clock pressure; otherwise leave at 20.
* No ledger archive needed — same reasoning as the 2026-06-21 ship
  (per-arm semantics are stable across screen-only vs screen+confirm;
  see that entry for the detail).
* Pair with a manual ``workflow_dispatch`` A/B comparing confirm-
  reject rates across one or two nights so the symptom drop is
  *measured* before flipping the cron permanently.

Speculative on the iteration-count halving: if the confirm-reject rate
turns out to be low (< 10% of screening accepts overturned), the
per-night cost saving from halving never materialises and the budget
can stay at the V1 count.  Audit after the first measurement night.

#### Pre-measure no-op short-circuit (after 2026-06-12 ship)

The 2026-06-12 ship detects no-op iterations *post-measure* by
comparing per-pair scores — correct but wasteful: both the baseline
and candidate measurements still run.  A natural cheap-compute
follow-up is to detect the most common no-op shape *pre-measure* by
comparing the candidate spec list to the current one immediately
after :func:`apply_mutation`: if the two are structurally equivalent
(same heuristics in the same order, same kwargs dict per slot, same
analyzers, same strategy class) the iteration is a guaranteed no-op
and can short-circuit before either measurement is run — saving the
candidate measurement entirely.  Two design notes:

* **Where the savings actually live.**  The dominant V1 no-op
  source identified in §2.1 is *dormant-rule* mutations: a
  proposal flips a kwarg that the spec doesn't actually use at the
  current budget (the kwarg is set on a heuristic the strategy
  rarely picks, or `update_interval` exceeds the budget so the
  analyzer never fires).  Those produce identical *per-pair*
  scores but the spec is *not* structurally identical — the kwarg
  did change.  Pre-measure short-circuit would catch a smaller
  subset (proposals where the new value equals the old, which is
  rare given the catalog filters those at the bandit level via
  ``categorical_choice``'s current-value exclusion and the
  ``float_uniform`` minimum-step guard).  The post-measure detector
  is what catches the dominant case.
* **What the short-circuit buys.**  Compute-saving on the
  pathological case where ``apply_mutation`` produces a
  byte-identical spec list — currently rare but cheap to detect.
  Also saves baseline-measurement compute when paired with a
  *baseline cache* (re-use the just-computed baseline from the
  previous iteration when the previous iteration's accepted ladder
  top is the same as this iteration's pre-mutation spec list, which
  is the common case under reject-heavy regimes) — a separate
  follow-up that builds on top.

Speculative until ledger evidence shows compute is the binding
constraint (today §2.5 reports 94% idle, so this is correctness-
neutral, not currency).

#### Flip the nightly cron to `--registry loop` — shipped 2026-06-21

Shipped 2026-06-21 as part of the V2 §9.5 step 5 partial flip; see the
2026-06-21 entry above for the full invocation, the rationale tied to
the 15-night summary diagnosis, and the smoke-test evidence.  The
ledger-archive marker proposed in the original sketch turned out not to
be needed: the bandit's ``_proposal_rule_key`` collapses to
``(class_name, param_name, rule_kind, ...)`` independent of the
strategy / spec name, so existing ledger entries (generated under
``--registry default``) replay correctly under ``--registry loop`` —
the smoke test against the live ledger confirms this end-to-end.
``--prime-include-archives`` / ``--structural-per-class-arms`` /
``--bandit-reward graded`` / ``--inactivity-relax-after 10`` /
``--holdout-base-seeds 7,1234`` / ``--guard-interval 10`` shipped in
the same change.  The manual ``workflow_dispatch`` A/B is the §12.3
daily routine's job over the next 2-3 nights.

#### Drop `Loop_DE_Family` heuristics for smaller compact specs

The 2026-06-10 ``_make_loop_strategies`` ship packs five DE-family
heuristics (LSHADE / JSO / NLSHADE_RSP / NLSHADE_LBC / LSHADE_EpSin)
into a *single* ``Loop_DE_Family`` ``StrategyRewarding`` spec so the
spec count stays at 7.  The strategy-level bandit allocates the
75-eval quick-mode budget across all five — average per-heuristic
budget ≈ 15 evals, which is below the ``NP_init = 15`` initial
population: most heuristics complete *one* generation per rep.  A
natural follow-up once ledger evidence accumulates is to split the
combined spec into five single-DE-heuristic strategies
(``Loop_LSHADE`` / ``Loop_JSO`` / ``Loop_NLSHADE_RSP`` / …) so each
DE variant gets the full strategy-allocated budget.  Lifts compute
cost from 7 → 11 specs (~5.5× quick).  Speculative until the loop
collects evidence on whether the per-DE-variant signal is currently
washed out by the combined-spec budget split.

#### LBFGSB follow-ups (after 2026-05-27 ship)

Multi-start L-BFGS-B shipped 2026-05-27 (see §13) and joined the
structural ``add_heuristic`` pool.  The A/B showed a *dedicated* LBFGSB
strategy cracks ``Rosenbrock_5D`` (≈3e-11) where every default strategy
scores 0.0, but *adding* it to the budget-split ``Rewarding_Diverse``
portfolio does not (and can regress other problems).  Natural
follow-ups:

- **Dedicated gradient-local-search default strategy (needs ADR).**
  Add a ``LocalSearch_LBFGSB`` (or ``StrategyPhased`` global→local) spec
  to ``_make_standard_strategies`` / ``_make_full_strategies`` so the
  *default battery* gains a strategy that actually solves smooth
  valleys.  This shifts the historical composite baseline, so it needs
  an architectural decision record (existing ladders are not directly
  comparable to the new battery) — the same gate the ``LSHADE_jSO``
  idea below carries.  Measure with `compare --statistical
  --fail-on-regression` first: a gradient arm helps the smooth /
  ill-conditioned problems (Rosenbrock, DixonPrice, Zakharov) but is
  useless on the multimodal ones (Rastrigin, Ackley, Schwefel), so the
  *net* composite effect must be measured, not assumed.
- **Warm-start restarts from the portfolio best.** Today the worker's
  restarts (after the first box-centre descent) are pure uniform-random.
  A refiner that warm-starts each restart from a perturbation of
  ``strategy.best`` would exploit the basin the rest of the portfolio
  has found — turning random multi-start into basin-hopping refinement.
  Needs a small protocol extension (the worker requests an ``x0`` from
  the parent at the start of each round rather than drawing it locally),
  because the global best is only known parent-side.
- **`LBFGSB.max_starts` catalog rule — shipped 2026-06-06**.
  ``default_catalog`` gains an ``integer_add`` rule with
  ``bounds=(1, 50)`` that fires when a spec sets ``max_starts`` to a
  concrete positive integer (the ``None`` auto-default sentinel is
  skipped by :func:`_find_targets`).  Lets the loop tune the
  exploration / exploitation balance of the multi-start schedule, the
  same way ``LSHADE.archive_factor`` is tuned.  See the §13 entry.

#### Analyzer add/drop follow-ups (after 2026-06-02 ship)

Analyzer add/drop shipped 2026-06-02 (see §13).  The candidate pool
is narrowly curated — only :class:`Sensitivity` and :class:`Restart`,
the two analyzers most strategies in the default battery already use.
Natural follow-ups when the loop has collected enough evidence to
motivate the work:

* **Categorical ``Restart.restart_strategy`` regimes — shipped
  2026-06-07**.  :class:`Restart` gains a third center-selection
  policy ``"sphere"`` (Gaussian around the box centre, ``std =
  ranges / 6``, clipped to the box) alongside the existing
  ``"random"`` (uniform-in-box) and ``"diverse"`` (max-min
  distance from previous restart centres) regimes.
  :func:`default_catalog` gains a matching ``categorical_choice``
  rule with ``choices=("random", "diverse", "sphere")`` and the
  standard structural-rule probability ``0.3``.  The rule fires
  only when a spec sets ``restart_strategy`` explicitly — the four
  built-in factory spots that ship
  ``restart_strategy="diverse"`` (``IPOP_CMAES`` /
  ``BIPOP_CMAES`` / IOH ``Sensitivity_Aggressive`` / the
  structural-catalog ``add_analyzer`` candidate) become applicable
  to the new rule out-of-the-box.  See the §13 entry.
* **Tunable ``Sensitivity.update_interval``** — the structural
  catalog ships :class:`Sensitivity` with the standard-mode default
  ``update_interval=20``.  Adding a kwarg ``MutationRule`` (kind
  ``integer_add`` with bounds ``[5, 60]``) would let the loop tune
  the update cadence — higher values reduce overhead, lower values
  give more responsive sensitivity tracking.  Only fires on specs
  that explicitly set the kwarg (the existing predicate), so
  byte-safe to add.
* **Expand the candidate pool** — research-grade analyzers
  (``Splitter``, ``Grid``, ``Dedensifyer``) are excluded from the
  current pool to avoid unconditionally proposing experimental
  analyzers.  Once the loop has accumulated evidence that the
  conservative pool wins consistently, broadening the pool is a
  natural follow-up.  Same shape as the heuristic-pool expansion
  pattern (one new ``add_analyzer`` candidate per analyzer class,
  ``avoid_duplicates=True``).
* **Strategy-class swap** — the third axis of the
  :class:`StrategySpec` (alongside heuristics and analyzers).
  Replace ``StrategyRewarding`` with ``StrategyUCB`` etc. without
  touching the heuristics list.  Requires a translation table for
  strategy-specific kwargs because the strategy classes do not
  share an interface.  Bigger scope than analyzer add/drop; ship
  after the analyzer ops have accumulated ledger evidence and
  motivated the cost.
* **Tunable ``sphere`` std-deviation kwarg on :class:`Restart`** —
  the ``"sphere"`` regime shipped 2026-06-07 currently uses the
  hard-coded ``Problem.random_point(distribution="normal")`` spread
  of ``ranges / 6`` (so 3σ covers half the box; ~99.7% of draws fall
  inside).  A natural follow-up is to expose a ``sphere_std_frac``
  kwarg on :class:`Restart` (defaulting to ``None``, which preserves
  the existing ``1/6`` scale) and a matching ``float_uniform``
  :class:`MutationRule` with ``bounds=(0.05, 0.4)`` so the bandit
  can tune the centroid-bias strength: small values (≤ 0.1)
  concentrate restarts very tightly around the box centre — useful
  on problems where the optimum is known to lie near the centroid —
  while larger values (≥ 0.3) approach the uniform-in-box behaviour
  of ``"random"``.  Speculative until the categorical rule shipped
  2026-06-07 has accumulated ledger evidence that ``"sphere"`` is
  the right regime for any subset of the battery.

#### `Restart.patience` mutation rule — shipped 2026-06-06

``default_catalog`` gains an ``integer_add`` rule with
``bounds=(3, 200)`` and ``delta_choices=(-20, -10, -5, 5, 10, 20)``
that fires whenever a spec sets ``patience`` to a concrete positive
integer (the ``None`` auto-default sentinel is skipped by
:func:`_find_targets`).  See the §13 entry.  Currently no built-in
factory ships an explicit ``patience`` value — the structural catalog's
``add_analyzer`` Restart candidate and the standard / full battery's
``IPOP_CMAES`` / ``BIPOP_CMAES`` specs all ship ``patience=None`` and
inherit the ``5 · dim`` auto-default — so the rule stays opt-in until
a future spec or mutation sets ``patience`` explicitly.  Natural
follow-up: a *categorical-with-dependent-kwarg* rule pattern that
would let the loop flip between ``None`` (auto-default) and a curated
discrete pool (e.g. ``{5, 10, 25, 50}``), bringing the auto-default
sentinel inside the bandit's reach.  Speculative — none of the
existing categorical rules need a dependent-kwarg shape.

#### Ship a jSO-tuned `LSHADE_jSO` strategy in `_make_standard_strategies`

The iLSHADE / jSO adaptive ``p_best`` schedule shipped 2026-05-19 is
*opt-in*: it only fires when a spec sets ``p_best_end`` explicitly.
None of the built-in :func:`_make_quick_strategies` /
:func:`_make_standard_strategies` / :func:`_make_full_strategies`
factories currently produce a spec with the canonical jSO settings
(``NP_init = 18·d``, ``p_best = 0.25``, ``p_best_end = 0.125``), so
the standard battery never exercises the new schedule out-of-the-box.
A natural follow-up is to add a dedicated ``LSHADE_jSO`` strategy to
``_make_standard_strategies`` so the composite score on the standard
battery directly reflects the literature-best DE refinement.  The
trade-off is that this would shift the historical composite score
baseline — needs an architectural decision record because existing
ladders won't be directly comparable to the new battery.

#### jSO asymmetric F-cap during early generations — shipped 2026-05-21

Shipped 2026-05-21 as
:attr:`panobbgo.heuristics.lshade.LSHADE.F_schedule` plus the
inherited :meth:`~panobbgo.heuristics.lshade.LSHADE._apply_F_cap`
that :class:`~panobbgo.heuristics.jso.JSO` opts into by
construction.  The three-phase cap (``F ≤ 0.7`` while
``progress < 0.6``, ``F ≤ 0.8`` while ``0.6 ≤ progress < 0.9``,
unclamped in the final 10%) is now shared infrastructure rather
than per-subclass code.  The 2026-05-15 :class:`JSO` ship had only
the first phase of the cap implemented; this entry completes the
literature-faithful three-phase cap from Brest et al. (2017,
§III-D).  See the §13 entry above.  :func:`default_catalog` gains
``LSHADE.F_schedule`` as a categorical rule so the loop can flip an
existing L-SHADE instance between the Tanabe-Fukunaga and jSO
regimes without dropping and re-adding the heuristic.

#### Tighten `eps_accept` once paired bootstrap is the loop default

The paired bootstrap shipped 2026-05-14 substantially narrows the
composite-delta CI under the randomized harness — typically 3–10× on
the loop's regime of 5 reps × ~3 problems at quick mode.  The
historical defaults of ``eps_accept=0.005`` and ``n_boot=2000`` were
sized for the (much wider) unpaired CI, so under paired sampling the
loop now leaves signal on the floor: a true ``+0.003`` improvement
whose CI does not bracket zero is still rejected for *"composite delta
≤ eps_accept"*.  Once a few hundred ledger entries have accumulated
under the paired default, lower ``eps_accept`` to ``0.002`` (or auto-
size it from the recently observed CI width) and consider trimming
``n_boot`` to ``500`` since the paired sampler converges faster.  Ship
the change with a ledger archive marker so the bandit's prior beliefs
do not silently mix the old and new accept regimes.  Pairs naturally
with the *Hierarchical / contextual bandit* idea below — both improve
loop *productivity* (accepts per iteration) rather than reach.

#### Contextual / hierarchical bandit over mutation rules

The Thompson sampler shipped 2026-05-01 treats every rule as an
independent arm.  A natural upgrade is to share strength across
rules that target the same heuristic class (one `Heuristic`-level
posterior) or the same kind (`log_uniform_perturb` posteriors borrow
strength across all classes).  Particularly valuable when the
catalog grows beyond a handful of rules and per-rule data is sparse.
Implementation: replace the flat `Dict[RuleKey, Stats]` with a
hierarchical Beta-Binomial or Dirichlet-Multinomial prior; expose
the grouping policy via the catalog itself.

#### Multi-dim default battery (now that stratification is shipped)

Stratified dimension sampling shipped 2026-05-02.  The default battery
in :func:`panobbgo.harness_randomized.make_default_families` still uses
``dim_choices=(2,)`` everywhere because expanding it would shift the
historical composite score baseline.  A natural follow-up is to add a
``make_default_families_multidim()`` factory (or a `--dim-mix` CLI
flag) that ships ``dim_choices=(2, 5, 10)`` for Rastrigin / Ackley /
DeJong, exposing the new stratification and giving the loop a richer
generalisation signal.  Needs an architectural decision record because
the resulting composite is not directly comparable to the existing
ladder.

#### Strategy portfolio composition (§7.2) — shipped 2026-05-03

Strategy portfolio composition shipped as
:class:`panobbgo.self_improve.StructuralMutationRule` and
:func:`panobbgo.self_improve.default_structural_catalog` — opt in with
``--structural`` on ``scripts/self_improve.py run`` or by passing
``catalog=default_structural_catalog()`` to :class:`SelfImprover`.  See
the §13 entry.  Natural next refinements:

- **Per-class arms in the bandit** — shipped 2026-05-18 as
  :attr:`panobbgo.self_improve.AdaptiveMutationSampler.per_class_structural`
  and :attr:`LoopConfig.structural_per_class_arms`.  Opt in via
  ``scripts/self_improve.py run --adaptive --structural-per-class-arms``.
  Each ``StructuralMutationRule`` is expanded at sampling time into one
  Thompson arm per candidate class so the bandit can learn that, e.g.,
  ``add Sobol`` wins while ``add Random`` loses.  See the §13 entry.
  Pairs naturally with the *contextual / hierarchical bandit* idea
  above — per-class arms are exactly the leaf nodes a hierarchical
  posterior would share strength across.
- **Analyzer add/drop — shipped 2026-06-02**.  Extends the structural
  mutation catalog with ``add_analyzer`` / ``drop_analyzer`` ops that
  mirror the heuristic versions but target
  :attr:`StrategySpec.analyzers` rather than ``heuristics``.  The
  default candidate pool is :class:`Sensitivity` (with
  ``update_interval=20``) and :class:`Restart` (with the IPOP-style
  ``diverse`` strategy and ``max_restarts=5``).  ``min_analyzers``
  defaults to ``0`` — unlike heuristics, an empty analyzers list is a
  valid spec.  See the §13 entry.
- **Strategy-class swap** — replace ``StrategyRewarding`` with
  ``StrategyUCB`` etc. without touching the heuristics list.  Requires
  every accepted swap to keep the strategy's hyperparameters either
  compatible or to drop them on the floor; needs a translation table.

#### PSO follow-ups (after 2026-05-05 ship)

PSO landed 2026-05-05; the ``lbest`` ring topology shipped 2026-05-07
and the optional Shi-Eberhart adaptive inertia (``w_end``) shipped
2026-05-07.  Natural extensions when the loop has collected enough
evidence to motivate the work:

- **Von Neumann topology — shipped 2026-05-22**.
  :attr:`panobbgo.heuristics.pso.PSO.topology = "vonneumann"` adds a
  4-connected 2-D toroidal grid (Kennedy & Mendes 2003; Mendes 2004)
  as a third topology slot — instantaneous (gbest) / one-hop ring
  (lbest) / two-hop planar (vonneumann).  The structural catalog
  ships all three PSO variants; the ``PSO.topology`` categorical rule
  grows to ``("gbest", "lbest", "vonneumann")``.  See the §13 entry
  above.
- **Random re-wired topology — shipped 2026-05-29**.
  :attr:`panobbgo.heuristics.pso.PSO.topology = "random"` adds the
  Mendes 2004 / Clerc 2007 / SPSO 2011 stochastic informer graph as
  a fourth topology slot.  Each particle is connected to itself plus
  ``k_neighbors`` random informers drawn uniformly with replacement
  from the rest of the swarm; the adjacency is built at ``on_start``
  and re-sampled at ``on_restart`` (Clerc 2007 stagnation-rebuild
  convention).  The structural catalog ships all four PSO variants
  (``gbest`` / ``lbest`` / ``vonneumann`` / ``random``); the
  ``PSO.topology`` categorical rule grows to ``("gbest", "lbest",
  "vonneumann", "random")``.  See the §13 entry above.
- **`StrategyPhased` integration** — pair PSO (global exploration
  phase) with NelderMead / LBFGSB (local refinement phase) on a
  single budget split, similar to the existing ``IPOP_CMAES``
  strategy.  Would be a new entry in
  ``_make_standard_strategies`` once measured to be a net win.
- **Categorical / topology mutation rule** — shipped 2026-05-13.
  ``MutationRule(kind="categorical_choice", choices=...)`` joined the
  numeric kinds (``log_uniform_perturb`` / ``integer_add`` /
  ``float_uniform``).  The default catalog wires it up for
  ``PSO.topology``, ``Sobol.scramble`` and ``LSHADE.archive_factor``.
  See the §13 entry.

#### Adaptive Differential Evolution (LSHADE / JADE) — shipped 2026-05-10

L-SHADE shipped 2026-05-10 as
:class:`~panobbgo.heuristics.lshade.LSHADE`; see the §13 entry.
Natural follow-ups when the loop has collected enough evidence to
motivate the work:

- **JADE archive sampling distribution** — L-SHADE samples ``r2``
  uniformly from the ``population ∪ archive`` union.  JADE
  (Zhang-Sanderson 2009) uses a slightly different rule that
  weights archive entries by recency; this could be a small
  per-step refinement.
- **L-SHADE-RSP / NL-SHADE-RSP / NL-SHADE-LBC follow-on variants** —
  NL-SHADE-RSP (CEC 2021 winner) shipped 2026-05-25 as
  :class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP` (rank-based
  selective pressure, non-linear population reduction, randomised
  adaptive archive); see the §13 entry.  NL-SHADE-LBC (CEC 2022
  winner) shipped 2026-05-28 as
  :class:`~panobbgo.heuristics.nl_shade_lbc.NLSHADE_LBC` (Linear Bias
  Change in the success-history Lehmer-mean memory update); see the
  §13 entry above.
- **iLSHADE / jSO adaptive p_best schedule** — shipped 2026-05-19
  as the opt-in ``LSHADE.p_best_end`` kwarg plus the
  :meth:`LSHADE._current_p_best` helper.  See the §13 entry.
- **iLSHADE / jSO heuristic class** — shipped 2026-05-15 as
  :class:`~panobbgo.heuristics.jso.JSO`, a direct subclass of L-SHADE
  with the Brest-Maučec-Bošković (CEC 2017) refinements: weighted
  ``current-to-pbest-w/1`` mutation, linear ``p_best`` schedule
  (``0.25 → 0.125``), Cauchy-F clamping in the early phase, jSO
  initial memory values (``M_F = 0.3``, ``M_CR = 0.8``), and a
  frozen anchor memory bin at ``M_F = M_CR = 0.9``.  See the §13
  entry above.  jSO is the **CEC-2017 single-objective
  bound-constrained competition winner**.
- **Categorical mutation rule for ``LSHADE`` archive on/off** —
  shipped 2026-05-13.  The default catalog now contains an
  ``archive_factor`` rule with ``choices=(0.0, 1.0, 2.6)`` that fires
  whenever a spec sets ``archive_factor`` explicitly.  See the §13
  entry.

#### jSO follow-ups (after 2026-05-15 ship)

jSO landed 2026-05-15 as :class:`~panobbgo.heuristics.jso.JSO`; see
the §13 entry.  Natural extensions when the loop has collected
enough evidence to motivate the work:

- **NL-SHADE-RSP** — CEC-2021 winner; **shipped 2026-05-25** as
  :class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP`, a direct
  :class:`JSO` subclass with rank-based parent selection, non-linear
  population reduction, and a randomised adaptive archive.  See the
  §13 entry.  The CEC-2022 successor **NL-SHADE-LBC** (adds a linear
  bias-correction mechanism) is queued under the *NL-SHADE-RSP
  heuristic* next-iteration idea.
- **L-SHADE-cnEpSin** — *partial ship 2026-06-03*: the precursor
  **LSHADE-EpSin** (Awad, Ali & Suganthan CEC 2016) shipped as
  :class:`~panobbgo.heuristics.lshade_ep_sin.LSHADE_EpSin`, an
  L-SHADE subclass that replaces SHADE Cauchy-from-memory ``F``
  sampling with an ensemble of two sinusoidal candidates during
  the first half of the search (revertion to SHADE Cauchy in the
  second half).  See the §13 entry above.  The CEC-2017 successor
  *LSHADE-cnEpSin* adds a covariance-matrix mutation step on top
  of EpSin; that step is **not** ported because CMA-ES is already
  available as a separate Panobbgo heuristic
  (:class:`~panobbgo.heuristics.cma_es.CMAES`).  If the bandit
  evidence ever shows a covariance-aware sinusoidal arm winning
  on a battery (which would be evidence neither pure CMA-ES nor
  pure EpSin captures the right dynamic), a future ship could
  port the cnEpSin covariance-mutation step explicitly.
- **Auto-tuned ``H`` — shipped 2026-06-04**.  ``default_catalog``
  gains a ``JSO.H`` ``integer_add`` rule (``bounds=(4, 12)``) so the
  loop can probe the success-history memory size on opt-in jSO specs
  the same way ``LSHADE.H`` does for L-SHADE.  See the §13 entry.
  The symmetric ``NLSHADE_RSP.H`` rule shipped in the same change.
- **Categorical mutation rule for ``JSO.p_best_max`` — shipped
  2026-06-09**.  ``default_catalog`` gains a ``categorical_choice``
  :class:`MutationRule` on the ``(JSO, p_best_max)`` slot with
  ``choices=(0.15, 0.25, 0.4)`` — the L-SHADE-like / jSO default /
  iLSHADE-like regimes, with the L-SHADE setting raised from the
  literature ``0.11`` to ``0.15`` so it clears jSO's default
  ``p_best_min = 0.125`` floor (the dependent-kwarg workaround the
  earlier entry flagged).  Sits alongside the existing
  ``float_uniform`` rule on the same slot — distinct bandit arms
  by construction.  See the §13 entry.  Follow-up: a
  *categorical-with-dependent-kwarg* rule pattern that lowers
  ``p_best_min`` to ``0.05`` when ``p_best_max < 0.125`` is proposed
  would let the L-SHADE-canonical ``0.11`` (and even narrower
  settings) become reachable; currently deferred until the
  dependent-kwarg pattern is motivated by a second slot too.

#### BOBYQA / NEWUOA / COBYQA local optimizer — shipped 2026-05-12

COBYQA (Ragonneau-Zhang 2023) — the modern Powell-family successor
to BOBYQA / NEWUOA / LINCOA — shipped 2026-05-12 as
:class:`~panobbgo.heuristics.cobyqa.COBYQA`; see the §13 entry.
Natural follow-ups when the loop has collected enough evidence to
motivate the work:

- **Constraint-aware variant** — COBYQA natively supports linear
  and nonlinear constraints; today the adapter only wires the box
  bounds.  A second variant that passes the strategy's constraint
  set to ``scipy.optimize.minimize(constraints=...)`` would let
  COBYQA exploit the constraint geometry directly instead of
  going through the penalty-handler indirection.  Useful when the
  problem has explicit constraints whose shapes are known.
- **Warm-start interpolation reuse** — every restart today rebuilds
  the ``2·n + 1`` interpolation set from scratch (a fresh
  subprocess).  COBYQA's reference implementation does not expose
  a persistent solver state in scipy's wrapper, but a vendored
  build of the upstream ``cobyqa`` library could be configured to
  warm-start the interpolation set from the last successful
  iterate — saving the first ``2·n`` evaluations on every
  restart.
- **Categorical mutation rule for ``scale`` on/off — shipped
  2026-06-04**.  ``default_catalog`` gains a
  ``COBYQA.scale`` ``categorical_choice`` rule with
  ``choices=(True, False)``.  Lets the bandit flip an existing
  COBYQA instance's box-rescaling regime without going through the
  full ``add_heuristic`` / ``drop_heuristic`` cycle.  See the §13
  entry.

#### Multi-seed hold-out for robust drift estimation — shipped 2026-05-16

Multi-seed hold-out shipped 2026-05-16 as
:attr:`panobbgo.self_improve.LoopConfig.holdout_base_seeds` (the
list-typed sibling of the scalar ``holdout_base_seed``) and the
``--holdout-base-seeds`` CLI flag.  See the §13 entry.  Natural
follow-ups when the loop has collected enough evidence to motivate
the work:

- **Bootstrap CI on the drift estimate — shipped 2026-05-17**.
  :func:`panobbgo.self_improve.aggregate_holdout_drift` plus
  :class:`HoldoutDriftAggregate` and the per-iteration paired score
  lists on :class:`LoopHoldoutRecord` (``seed_iteration_scores`` /
  ``top_iteration_scores``) pool drifts across all hold-out records
  and bootstrap a CI on the aggregate.  CLI gains
  ``--fail-on-overfit-ci`` (stricter sibling of
  ``--fail-on-overfit``) plus ``--holdout-ci-confidence`` and
  ``--holdout-ci-n-boot`` knobs.  See the §13 entry.
- **Auto-rollback on multi-seed overfit** — when several seeds
  agree the ladder is overfit, the loop could automatically pop the
  ladder back to the seed and penalise the bandit (see
  *Auto-rollback on hold-out overfit* below).  Multi-seed evidence
  is strong enough to act on, whereas single-seed evidence might
  still be a fluke.  Now even better-motivated with the
  bootstrap-CI rule above: the CI verdict is a more reliable
  trigger than per-seed point checks.

#### Auto-rollback on hold-out overfit

When the hold-out flags ``overfit=True``, the loop currently just
records and (optionally) exits.  A more aggressive remediation is
to automatically pop the ladder back to the seed entry and persist
the rollback in a new ``LoopHoldoutRollbackRecord`` so a subsequent
``--adaptive-prime-from-ledger`` resume picks up the failure as a
negative reward signal for *all* the rules that contributed to the
discarded ladder.  Needs care around the bandit semantics: penalising
all rules along the discarded path is more aggressive than penalising
only the last one, and the right policy is an open question.

#### Hierarchical bandit over the per-class structural arms — shipped 2026-06-01

Per-class structural arms shipped 2026-05-18; the hierarchical
Beta-Binomial follow-up shipped 2026-06-01 as
:attr:`panobbgo.self_improve.AdaptiveMutationSampler.structural_borrow_alpha`
and :attr:`LoopConfig.structural_borrow_alpha`, opt in via
``scripts/self_improve.py run --adaptive --structural-per-class-arms
--structural-borrow-alpha 0.5``.  Each per-class arm's Beta posterior
borrows ``κ · (n_other_class_accepts, n_other_class_failures)`` from
the op-level aggregate (sum over sibling per-class arms) with a
deliberate self-exclusion, so a fresh candidate class warms with the
op's empirical accept rate instead of the symmetric ``Beta(1, 1)``
prior.  See the §13 entry above.

Natural follow-ups when the loop has collected enough evidence to
motivate the work:

* **Auto-tune ``κ``** — track per-iteration variance of the borrow
  improvement, anneal ``κ`` down as per-class evidence accumulates
  (close to ``κ = 0`` at large per-arm sample sizes, close to ``κ = 1``
  when arms are still sparse).  A simple recipe is ``κ_eff =
  κ_init / (1 + n_class_attempts / horizon)`` — borrow heavily early,
  vanish as evidence grows.
* **Hierarchical kwarg arms too** — the same mechanism could borrow
  across kwarg arms that share a heuristic class (e.g. all
  ``LSHADE.*`` arms borrowing from one aggregate "LSHADE rules"
  posterior).  Lower-priority: kwarg arms already have
  literature-canonical centres so cold-start is less painful than
  for structural arms.
* **Categorical ``κ`` regimes** — ``κ ∈ {0.0, 0.5, 1.0}`` as a
  ``categorical_choice`` mutation rule on the loop driver itself.
  Lets the loop tune its own meta-bandit hyperparameter from ledger
  evidence — a true second-order self-improvement.

#### Tunable F-cap breakpoints / cap values on `LSHADE.F_schedule`

The F-cap shipped 2026-05-21 hard-codes the canonical Brest et al.
2017 breakpoints (0.6 / 0.9) and cap values (0.7 / 0.8).  These are
the literature defaults; other variants in the DE family use
different settings.  Once enough ledger evidence has accumulated for
the categorical ``LSHADE.F_schedule`` rule, a natural follow-up is to
make the cap geometry tunable.  Two design sketches:

* **Multiple categorical regimes.** Replace the binary
  ``F_schedule = True / False`` with a categorical choice over
  named regimes — ``"off"``, ``"jso"`` (current 0.6 / 0.7 + 0.9 / 0.8),
  ``"ilshade"`` (different breakpoints / caps from Brest 2016),
  ``"strict"`` (more aggressive — e.g., F ≤ 0.5 throughout the first
  half).  Each regime ships as a module-level constant tuple so the
  bandit can flip between them without touching the heuristic body.
* **Continuous parameters.** Expose ``F_cap_phase1``, ``F_cap_phase2``,
  ``F_cap_bound1``, ``F_cap_bound2`` as four kwargs with bounded
  ``float_uniform`` perturbations.  Wider mutation space but lets the
  bandit climb the cap surface continuously.  Risk: any cap above
  0.85-ish probably no-ops because the L-SHADE Cauchy sampler rarely
  draws ``F > 0.9`` from healthy memory bins.

The categorical-regime approach has lower bandit dimension and is
literature-grounded — pick that first if you ship the follow-up.

#### Inactivity-guarded loop productivity — eps_accept relaxation shipped 2026-05-30

* **Relax ``eps_accept`` adaptively** — **shipped 2026-05-30** as
  :attr:`panobbgo.self_improve.LoopConfig.inactivity_relax_after` /
  :attr:`~panobbgo.self_improve.LoopConfig.inactivity_relax_factor` /
  :attr:`~panobbgo.self_improve.LoopConfig.inactivity_min_eps_accept`
  and the matching ``--inactivity-relax-after`` family of CLI flags.
  Each :attr:`LoopIterationRecord` now persists the *effective*
  ``eps_accept`` and the inactivity-streak length, so an auditor can
  replay the loop with the exact rule that produced any given
  accept.  See the §13 entry.  Disabled by default
  (``inactivity_relax_after = 0``) so existing ledgers and CI
  invocations stay byte-identical.
* **Bump the harness mode for the cron** — quick mode at 3 reps is
  the noise floor.  A 30-iteration loop at ``--standard`` (5 reps,
  larger budget) may produce more genuine accepts than 100
  iterations at ``--quick``.  Needs a self-hosted runner because
  GitHub-hosted runners are 2 cores.  Still open.
* **Use the bootstrap CI alone** (no point-delta gate) — alternative
  to the geometric relaxation above; pair the
  :func:`statistical_accept` rule with ``eps_accept = 0`` while
  keeping the CI-lower-bound gate.  Equivalent, in the relaxed-floor
  limit, to setting ``inactivity_min_eps_accept = 0`` and a large
  ``inactivity_relax_after`` — left as an open variant for the next
  iteration if the relaxation knob proves too coarse.
* **Care for §11**: the success criteria pin ``eps_accept`` at a
  fixed level so a chronic relaxation would silently shift the
  loop's "improvement" bar.  The 2026-05-30 ship mitigates this by
  (1) flooring the threshold at
  ``inactivity_min_eps_accept`` (default ``0.001``, matching the
  bootstrap CI's noise floor) and (2) recording both the effective
  threshold and the streak length on every iteration record so a
  reviewer can grep the ledger for any accept whose
  ``effective_eps_accept < eps_accept`` and audit those entries
  separately.

#### NL-SHADE-RSP heuristic (CEC 2021 winner) — shipped 2026-05-25

NL-SHADE-RSP shipped 2026-05-25 as
:class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP`, a direct
subclass of :class:`~panobbgo.heuristics.jso.JSO` adding Non-Linear
Population Size Reduction, Rank-based Selective Pressure on the ``r1``
draw (``k_rank``), and a randomised adaptive archive.  See the §13
entry above.  The three jSO override points were extracted into the
behaviour-preserving base-class hooks :meth:`LSHADE._select_r1`,
:meth:`LSHADE._lpsr_target`, and :meth:`LSHADE._archive_cap`.

Natural follow-ups when the loop has collected enough evidence to
motivate the work:

* **Adaptive crossover blend + pA archive adaptation** — the two
  CEC-2021 mechanisms intentionally *not* ported in the 2026-05-25
  ship.  (1) NL-SHADE-RSP adapts the probability of binomial vs
  exponential crossover from their relative success; (2) it adapts
  ``pA`` — the probability of drawing ``r2`` from the archive — from
  the relative improvement of archive- vs population-sourced trials,
  rather than the randomised-cap stand-in shipped here.  Both need
  per-trial bookkeeping (which crossover operator / archive source a
  trial used) that the current ``_TrialMeta`` does not carry; adding
  two optional fields to ``_TrialMeta`` and the matching success
  accounting in ``on_new_results`` is the clean shape.
* **NL-SHADE-LBC** (CEC 2022 winner) — **shipped 2026-05-28** as
  :class:`~panobbgo.heuristics.nl_shade_lbc.NLSHADE_LBC`, a direct
  :class:`NLSHADE_RSP` subclass that adds Linear Bias Change in the
  F / CR Lehmer-mean memory update: the order ``p`` is linearly
  scheduled across budget progress instead of fixed at ``2`` (defaults
  ``p_F: 3.5 → 1.5``, ``p_CR: 1.0 → 1.5``, spread ``m_lbc = 1.5``).
  At ``p = 2, m = 1`` the formula recovers the standard L-SHADE
  Lehmer mean.  See the §13 entry.
* **Categorical ``k_rank`` regimes — shipped 2026-06-04**.
  ``default_catalog`` gains a ``categorical_choice`` rule with
  ``choices=(0.0, 3.0, 5.0)`` (uniform/jSO recovery / Stanovov
  default / aggressive) sitting alongside the existing
  ``float_uniform`` rule on the same ``(NLSHADE_RSP, k_rank)``
  slot.  The two live on distinct bandit arms (different
  ``rule_kind`` → different ``_proposal_rule_key``).  See the §13
  entry.

#### NL-SHADE-LBC follow-ups (after 2026-05-28 ship)

NL-SHADE-LBC shipped 2026-05-28 as
:class:`~panobbgo.heuristics.nl_shade_lbc.NLSHADE_LBC`; see the §13
entry above.  Natural extensions when the loop has collected enough
evidence to motivate the work:

* **Categorical LBC regimes** — the four LBC schedule kwargs
  (``p_F_init``, ``p_F_final``, ``p_CR_init``, ``p_CR_final``) and the
  spread ``m_lbc`` are exposed as ``float_uniform`` rules today.  A
  set of literature-canonical *named* regimes — ``"cec2022"`` (the
  Stanovov defaults 3.5/1.5/1.0/1.5/1.5), ``"lshade"``
  (2/2/2/2/1 — recovers standard L-SHADE), ``"flat"``
  (1/1/1/1/1.5 — pure arithmetic mean), ``"aggressive"``
  (5/3/3/5/1.5 — strongly biased throughout) — wrapped as one
  ``categorical_choice`` per slot would give the bandit a discrete
  arm to flip the bias regime cleanly, the same way
  ``LSHADE.F_schedule`` flips the jSO F-cap on / off.  Implementation
  shape: a single composite kwarg ``lbc_regime`` whose setter applies
  the named tuple to the five fields, plus the categorical rule.
* **Per-CR / per-F sub-regime A/B** — the literature defaults flow
  F-bias from high to low while CR-bias does the opposite.  The
  motivation in the paper is qualitative; nightly evidence may reveal
  problem classes where *both* should decrease (or both increase).  A
  measured A/B at ``--standard`` mode with the bandit constrained to
  the LBC arm would identify whether the paper's asymmetric schedule
  generalises beyond the CEC battery.
* **Adaptive bias bounds from the success history** — instead of
  using the static linear schedule, infer the schedule from the
  observed variance of successful F / CR values.  When the success
  variance is low (memory is converging), more bias is helpful;
  when high (exploration still useful), less bias.  Speculative —
  the paper's static schedule is well-tuned; a learned schedule
  would need to clearly beat it on cross-problem averages.

#### Run a measured A/B across PSO topologies (gbest / lbest / vonneumann)

Von Neumann shipped 2026-05-22 (see §13).  The literature predicts
the three topologies are *complementary* — gbest wins on unimodal
landscapes, lbest on highly-multimodal, vonneumann between the two —
but the shipped entry did not include a measured benchmark because
the impact at quick-mode budgets is within noise.  A natural
follow-up is to run an explicit ``benchmark_harness.py compare``
across the three Rewarding strategies (one per PSO topology) at
``--standard`` mode (≥ 5 reps × ~8 problems × ~300 evaluations) so
the *per-problem* per-topology winners are identified.  Use the
paired-bootstrap CI (auto-selected on ``--randomize``) so the
per-pair regressions are detected rigorously.  The output of this
#### Run a measured A/B across PSO topologies (gbest / lbest / vonneumann / random)

Von Neumann shipped 2026-05-22; the random informer graph shipped
2026-05-29 (see §13).  The literature predicts the four topologies
are *complementary* — gbest wins on unimodal landscapes, lbest on
highly-multimodal, vonneumann between the two, and random's
diffusion speed depends on the realised graph.  None of the shipped
entries included a measured benchmark because the impact at
quick-mode budgets is within noise.  A natural follow-up is to run
an explicit ``benchmark_harness.py compare`` across four Rewarding
strategies (one per PSO topology) at ``--standard`` mode (≥ 5 reps
× ~8 problems × ~300 evaluations) so the *per-problem*
per-topology winners are identified.  Use the paired-bootstrap CI
(auto-selected on ``--randomize``) so the per-pair regressions are
detected rigorously.  The output of this benchmark feeds two
follow-ups:

* If the data shows a per-problem-class winner pattern, encode it in
  the structural catalog (e.g., add a ``StrategySpec`` that pre-pairs
  ``vonneumann`` with Rastrigin / Ackley / Griewank-style problems
  via the strategy-pattern matcher).
* If no topology wins consistently across problem classes, leave the
  current uniform-over-four catalog and let the bandit's per-arm
  reward signal identify the winner online.

#### Inactivity-relax telemetry in the summary view — shipped 2026-06-16

Shipped 2026-06-16 alongside the §12.4 *Summary trend block* (see the
dated entry above).  ``scripts/self_improve.py summary`` now renders an
``Inactivity:`` block surfacing the inferred ``eps_accept`` base (the
maximum observed ``effective_eps_accept`` — relaxation only decreases
the threshold), the longest drought (max ``iters_since_accept`` across
all records), the relaxed-accept count, and the mean decay factor at
the moment of accept.  Suppressed automatically on legacy ledgers
whose iteration records carry neither field (pre-2026-05-30).

#### Per-iteration re-sampled random PSO topology (stochastic-K) — shipped 2026-06-05

Shipped 2026-06-05 as
:attr:`panobbgo.heuristics.pso.PSO.stagnation_threshold` plus the
matching :meth:`PSO._maybe_rebuild_random_adjacency` helper and the
``PSO.stagnation_threshold`` ``integer_add`` rule on
:func:`default_catalog`.  See the §13 entry above.  When set to a
positive integer, the random adjacency is re-sampled mid-run after
``N`` consecutive incoming results land without lifting the global
best — finer-grained than the restart-gated rebuild that ships
under :class:`~panobbgo.analyzers.restart.Restart`.  Default is
``None`` (off), so existing PSO behaviour is byte-identical.

#### Categorical-with-dependent-kwarg rule pattern

The 2026-06-09 ``JSO.p_best_max`` categorical ship had to substitute
``0.15`` for the literature-canonical L-SHADE ``p_best = 0.11`` because
the latter would violate jSO's constructor invariant
``p_best_min <= p_best_max`` (default ``p_best_min = 0.125``).  A
*categorical-with-dependent-kwarg* rule pattern — one mutation rule
that, when proposing a new value for ``param_a``, also coordinates a
matching value for ``param_b`` on the same heuristic instance — would
let the loop reach genuinely L-SHADE-canonical jSO settings (and a
half-dozen other constrained pairs across the catalog).  Design sketch:

* New :class:`MutationRule` subtype ``DependentKwargRule`` (or extend
  :class:`MutationRule` with an optional ``co_params`` field) that
  carries a list of ``(param_name, value_fn)`` pairs.  When the rule
  fires, ``apply_mutation`` updates *all* listed kwargs atomically so
  the constructor sees a consistent state.
* Bandit-arm key continues to live on the *primary* slot (e.g.,
  ``(JSO, p_best_max, categorical_choice)``), so the existing per-arm
  posterior bookkeeping survives unchanged.
* Tests: round-trip through the JSONL ledger must preserve the
  coordinated update so a ``--adaptive-prime-from-ledger`` resume
  re-creates the dependent-kwarg state.

Motivation accumulates beyond the jSO slot: ``LSHADE_LBC.p_F_init`` /
``p_F_final`` are paired; ``Restart.sphere_std_frac`` (queued under
"Tunable sphere std-deviation kwarg on :class:`Restart`") would pair
with the ``"sphere"`` regime of ``restart_strategy``; future
``StrategyRewarding`` ↔ ``StrategyUCB`` swaps will need a small
kwarg-translation table that is structurally the same pattern.  Ship
once two of these are on the table — one slot is not enough motivation
for the new rule subtype.

#### Categorical regimes for `LSHADE.F_schedule` (named cap regimes)

The 2026-05-21 jSO ship made ``LSHADE.F_schedule`` a bool
(``True`` → Brest et al. 2017 three-phase cap, ``False`` → unclamped
L-SHADE).  ``default_catalog`` already exposes it as a binary
``categorical_choice``.  A natural follow-up — flagged under
*Tunable F-cap breakpoints / cap values on ``LSHADE.F_schedule``* — is
to broaden the bool into a string-valued categorical over named
regimes ``("off", "jso", "ilshade", "strict")``.  The two
already-shipped categorical rules with multi-string choices
(``PSO.topology``, ``Restart.restart_strategy``) provide the wire
shape; the work is mostly heuristic-side (adding three module-level
constant tuples for the breakpoints / caps of each named regime) plus
a backwards-compat layer that maps the old ``True`` / ``False`` values
to ``"jso"`` / ``"off"`` so existing specs and ledgers keep working.
Picks up where the 2026-06-09 jSO ``p_best_max`` ship left off: the
same shape — collapse a continuous-or-binary knob into a small fixed
set of literature regimes — applied to the next under-catalogued dial.
