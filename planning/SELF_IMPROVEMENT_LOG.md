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

### 2026-07-08 — Curvature-aware quadratic local step for the `Nearby` refinement heuristic

* **What** — Added an in-process curvature-aware quadratic step to
  :class:`panobbgo.heuristics.nearby.Nearby` (constructor kwargs
  ``quadratic: bool = False`` / ``quadratic_trust: float = 2.0`` /
  ``quadratic_min_r2: float = 0.8``).  When ``quadratic=True`` the heuristic
  keeps a rolling buffer of recently evaluated ``(x, f(x))`` pairs (populated
  by a new ``on_new_results`` handler, guarded by a lock because the EventBus
  dispatches ``on_new_results`` and ``on_new_best`` on separate threads) and,
  on each new best, fits a **distance-weighted ridge quadratic** to the
  nearest points in box-normalised coordinates via the new module-level
  :func:`panobbgo.heuristics.nearby.fit_quadratic_step` and emits its
  **trust-region-constrained Newton minimiser** as the *first* of the
  heuristic's ``new`` points (the remaining ``new − 1`` stay isotropic
  perturbations so exploration is preserved).  ``quadratic=False`` (the
  default) is byte-for-byte the historical heuristic — no buffer, no model.

  The fit is hardened for the portfolio setting: per-column
  (standardization-equivalent) ridge so the poorly-scaled quadratic columns
  are not swamped by the intercept; positive-definite regularisation of the
  Hessian so the Newton step is always a descent direction even at saddles;
  a **data-support trust region** (never step beyond the radius of the local
  data cloud — a quadratic is only trustworthy where it interpolates, not
  where it extrapolates); and a **weighted-R² fit-quality gate**
  (``min_r2 = 0.8``) that returns ``None`` when a single quadratic does not
  explain the local sample, so the step fires only on genuinely
  quadratic-like neighbourhoods (smooth valleys) and falls back to isotropic
  exploration on multimodal neighbourhoods spanning several basins.

  The six Rewarding-family seed specs whose ``Nearby`` plays the standard
  ``radius=0.124, axes="all", new=3`` local-refinement role now ship
  ``quadratic=True`` in :mod:`panobbgo.harness`: ``Rewarding_Diverse``
  (quick), ``Rewarding_RegionUCB`` / ``UCB_Diverse`` / ``Thompson_Diverse``
  (standard / full), and ``Loop_RegionUCB`` / ``Loop_Restart`` (loop
  registry — measured nightly by the cron).  The tighter GP-specialised
  ``Nearby(radius=0.05)`` entries are left untouched.

* **Why** — Isotropic perturbation is an *un*-informed local move: on
  ill-conditioned valleys most random perturbations of the best point step
  *across* the narrow valley (uphill) rather than *along* it.  The randomized
  battery — the self-improvement loop's own optimization + anti-overfit
  metric — injects log-uniform diagonal scaling and Haar rotation into
  *every* problem, so it systematically stresses exactly the regime where a
  curvature-aware (per-model Newton) step wins.  Unlike the L-BFGS-B warm
  restart (2026-07-07), the quadratic model is fitted from points the *rest
  of the portfolio* already evaluated, so it spends **zero** extra objective
  evaluations building its curvature estimate.  Graduates the top-priority
  §7.3-freeze-compliant idea from *Next iteration ideas* (*improve an
  existing heuristic*; better default kwargs for existing specs — no new
  catalog arms).

* **Measured impact** — paired ``--randomize`` A/B on the exact
  ``Rewarding_Diverse`` quick-registry spec:

  | metric | value |
  |---|---|
  | composite (reps 12, iter 0) | 0.0339 → 0.0612 |
  | ``statistical_accept`` | **ACCEPT** Δ=+0.0274, 95% CI ``[+0.0057, +0.0521]`` |
  | worst-pair regression | −0.0178 (> −eps_regress 0.05) |
  | mean Δ over 20 iters × 2 base_seeds | ≈ **+0.075** (18/20 positive) |

  Per randomized family (iter 0): DeJong +0.167, Rosenbrock +0.070, Ackley
  −0.018 (tiny), Rastrigin unchanged.  On the *fixed* natural-conditioning
  battery (Rosenbrock/Styblinski at default conditioning) the effect is
  net-neutral within noise and ``Rosenbrock_5D`` stays at composite 0 (the
  binary-success metric does not register "getting closer") — the win is
  specific to the ill-conditioned regime, which is what the loop optimizes
  against.  Standard / full / loop siblings carry the identical codify and
  are queued for nightly re-validation (the anti-cherry-pick guard + §6.4
  confirm gate protect the loop specs).

* **Tests** — 21 new tests in ``tests/test_heuristic_nearby_quadratic.py``:
  ``fit_quadratic_step`` recovery on isotropic / anisotropic / Rosenbrock-like
  local models, robustness guards (too-few-points, non-finite values,
  trust-region clipping, indefinite-Hessian descent), the R²-gate accept /
  reject behaviour, and the ``Nearby(quadratic=…)`` wiring (buffer
  accumulation + cap, curvature-aware first point, byte-identical disabled
  path).  Full ``Nearby`` suite (32) + harness suites (130) + loop-registry
  suite green; ruff + pyright clean.

* **Follow-up ideas** seeded under *Next iteration ideas*: a full-quadratic
  (vs the current ridge-regularised) fit needs ``O(d²)`` local points, so in
  high dimensions a *diagonal-plus-low-rank* Hessian model may recover the
  valley curvature from fewer samples; wiring ``quadratic=True`` into the
  ``Loop_LocalSearch`` seed once the loop confirms the Rewarding-family
  siblings; and a categorical ``Nearby.quadratic`` catalog rule (blocked by
  the §7.3 freeze until the loop resolves its current arms).

### 2026-07-07 — Warm-started memetic restarts for the L-BFGS-B local polish (curved-valley class)

* **What** — Added a ``warm_start`` mode to
  :class:`panobbgo.heuristics.lbfgsb.LBFGSB` (constructor kwargs
  ``warm_start: bool = False`` / ``warm_start_sigma: float = 0.1``).  When
  enabled, every restart *after* the first box-centre descent starts from a
  small Gaussian perturbation of the strategy's **best incumbent** result
  instead of a fresh uniform-random point in the box — the memetic recipe
  scipy ``dual_annealing`` owes its Rosenbrock win to.  Because the incumbent
  lives parent-side (only the ``Best`` analyzer knows it), the subprocess
  worker cannot draw the warm point itself: a small protocol extension has the
  worker **request** an ``x0`` from the parent over the existing request pipe
  (a bare ``_X0_REQUEST`` sentinel string), and :meth:`LBFGSB.on_start`
  answers inline with :meth:`LBFGSB._warm_start_x0` (``clip(best + N(0,
  sigma·range), box)``, or a uniform draw before the first result so a
  warm-started worker degrades gracefully to classic multi-start).  A new
  :meth:`LBFGSB.on_new_best` tracks the incumbent (mirrors
  :meth:`panobbgo.heuristics.nearby.Nearby.on_new_best`).  ``warm_start=False``
  (the default) keeps the historical uniform-restart worker byte-for-byte.

  The **structural-catalog** LBFGSB candidate in
  :func:`panobbgo.self_improve.default_structural_catalog` was flipped from
  ``(LBFGSB, {})`` to ``(LBFGSB, {"warm_start": True})`` so the loop's
  ``add_heuristic`` op inserts the warm variant.  Seed specs are left as-is;
  the loop's structural bandit + statistical-accept gate will now measure
  warm-LBFGSB additions live and can codify a seed flip if the evidence lands.

* **Why** — This targets the sharpest measured competitive gap (the 2026-07-06
  flagship finding): every Panobbgo strategy scored ``0`` on ``Rosenbrock_5D``
  while stock ``dual_annealing`` solved it.  The 2026-07-06 A/B recorded a
  *negative* result — bolting **cold** (uniform-restart) LBFGSB onto
  ``Rewarding_Diverse`` *regressed* the composite — and diagnosed the root
  cause as the wrong restart geometry: a local optimizer inside a portfolio
  should polish the basins the *rest* of the portfolio is discovering, not
  gamble on fresh uniform draws.  Warm restarts fix exactly that geometry and
  make the descent intrinsically curvature-aware (L-BFGS-B builds a
  finite-difference curvature estimate from the incumbent).  This is the
  §7.3-freeze-compliant path the backlog's top-priority idea called for:
  *improve an existing heuristic*, not add a new one.

* **Measured impact** (≥3-seed aggregate per the measurement discipline):

  | portfolio | budget | cold | warm |
  |---|---|---|---|
  | ``[Sobol, Random, Nearby, LBFGSB]`` (curved-valley battery) | full (500) | 0.156 | **0.198** |
  | ``[Sobol, LBFGSB, NelderMead]`` (Rosenbrock 2D/5D + Styblinski) | standard (200) | 0.583 | 0.583 |

  At the full budget warm wins clearly (+0.042, driven by ``Rosenbrock_2D``
  and ``StyblinskiTang_2D``); at the tighter standard budget the composite is a
  tie (both 2D problems saturate, both fail 5D) but warm's ``Rosenbrock_5D``
  best-distance is consistently lower (11.7 vs 15.8) — it approaches the valley
  optimum faster without ever regressing the composite.  Fully *crossing* the
  ``Rosenbrock_5D`` tolerance still needs more budget or a dedicated
  local-search strategy (which shifts the historical composite baseline and
  needs an ADR — out of scope; left as a next idea).

* **Tests** — 17 new tests in ``tests/test_heuristic_lbfgsb.py``
  (``LBFGSBConstructionTests`` warm-start validation,
  ``LBFGSBWarmStartTests`` for ``on_new_best`` / ``_warm_start_x0`` /
  ``on_start`` sentinel handling, ``LBFGSBWarmStartWorkerTests`` for the
  worker's x0-request protocol and clean-exit-on-closed-pipe).  Full
  LBFGSB suite (59) + structural-catalog suite (145) green; ruff + pyright
  clean.

* **Follow-up ideas** seeded under *Next iteration ideas*: a curvature-aware
  quadratic/trust-region warm step; wiring ``warm_start`` into the
  ``Loop_LocalSearch`` seed once the loop confirms it; and a dedicated
  warm-started local-search strategy (needs an ADR).

### 2026-07-06 — Codify: drop the `LatinHypercube` seeder from `Loop_LocalSearch` (structural, ledger-evidenced)

* **What** — Removed the ``(LatinHypercube, {"div": 4})`` seeder entry
  from the ``Loop_LocalSearch`` seed spec in
  :func:`panobbgo.harness._make_loop_strategies`.  The strategy now runs
  ``COBYQA`` (derivative-free trust region) + ``LBFGSB`` (multi-start
  quasi-Newton) + ``NelderMead`` (cheap simplex fallback) with no
  low-discrepancy seeder.  Applied end-to-end through the sanctioned
  ``scripts/self_improve.py codify-scan --apply-top --apply-format``
  pipeline (V2 §9.3 / §12.3 step 2) — the first *structural* codify to
  land via the automated apply driver (the 2026-07-01 ship built the
  structural-edit primitive; this exercises it on real ledger evidence).

* **Why** — The self-improvement ledger accumulated **two independent
  ``drop_heuristic`` accepts** for ``LatinHypercube`` on
  ``Loop_LocalSearch`` across two distinct nights, each clearing its own
  bootstrap-CI accept gate:

  | night | Δ | CI95% |
  |---|---|---|
  | 2026-06-24 | +0.0511 | [+0.0352, +0.0670] |
  | 2026-06-29 | +0.0471 | [+0.0368, +0.0617] |

  (pooled per-record CI ``[+0.0471, +0.0511]``, ``min_record_ci_low``
  ``+0.0352``.)  Mechanistically the drop makes sense: both local
  optimizers already seed their *first* descent from the box centre
  (``COBYQA`` / ``LBFGSB._box_center``) and multi-start from fresh
  points thereafter, so the LHC "first looks" only diluted the tight
  quick-mode 75-eval budget without giving the refiners a better
  anchor.  The 2026-07-01 entry explicitly flagged this as the
  live-ledger's top structural candidate, awaiting one more night of
  evidence — which the 2026-06-29 accept supplied.  Advances V2 §11
  success criterion 2 (codify-PR throughput; this is the fourth
  ledger-evidence-driven codify after ``Sobol.scramble=False``,
  ``Nearby.radius`` catalog-bound tightening, and the ``Nearby.radius``
  seed shift).  Respects the §7.3 catalog freeze — no new arms, no new
  heuristics; a seed-spec composition change backed by measurement.

* **Tests** — ``tests/test_self_improve.py`` + ``tests/test_harness.py``
  (646 passed).  The apply is idempotent: a re-run of ``--apply-top``
  derives 0 edits (the "missing class not re-dropped" safety guard), so
  the queued ``--open-pr`` driver returns early without opening an
  empty PR.

* **Known cosmetic gap (seeded as a next idea)** — the structural
  already-codified suppression predicate
  (:func:`_structural_already_codified`) is *global*: a ``drop_heuristic``
  candidate is suppressed only when **no** seed spec carries the class.
  ``LatinHypercube`` still lives in ``Loop_Restart`` (and the ``quick`` /
  ``standard`` seeders), so this candidate keeps surfacing in the scan
  report even though the apply targets — and has already edited — only
  the evidenced ``Loop_LocalSearch``.  Harmless (apply idempotent, no
  empty PRs) but the report is misleading.  The fix is to make the
  structural suppression *spec-scoped* — mirror the apply's
  ``strategy_names`` narrowing — so the candidate is suppressed once its
  own evidenced spec(s) no longer carry the class.  See the
  "Membership-vs-coverage rule for structural ops" note under
  *Next iteration ideas*.

* **Flagship competitive-gap finding (measured this iteration; seeded as
  the top next idea)** — a standard-mode ``--baselines`` run
  (8 problems × 5 reps, seed 42) shows Panobbgo's best strategy
  (``Rewarding_Diverse``, composite **0.736**) beats stock scipy dual
  annealing (``Baseline_SciPyAnneal``, **0.552**) overall — **but** on
  three problems *every* Panobbgo strategy loses to that stock baseline:

  | problem | best Panobbgo | Baseline_SciPyAnneal |
  |---|---|---|
  | StyblinskiTang_2D | 0.48 (BayesOpt_GP) | **0.73** |
  | Rosenbrock_2D | 0.72 | **0.88** |
  | Rosenbrock_5D | **0.00 (all)** | **0.49** |

  The curved-valley class (Rosenbrock) is the sharpest gap: **every
  Panobbgo strategy scores exactly 0 on Rosenbrock_5D** (tolerance 1.0,
  200 evals) while dual annealing solves it.  **Measured negative
  result:** bolting ``LBFGSB`` (max_starts ∈ {2, 5}) onto
  ``Rewarding_Diverse`` and re-measuring over **three seeds** (42/43/44)
  is a *net regression* (composite 0.657 → 0.652 / 0.643); it roughly
  halves the Rosenbrock_5D best-distance (14.7 → 7.4) but never crosses
  the tolerance, and the apparent single-seed StyblinskiTang win
  (0.17 → 0.54) evaporated under the seed sweep (aggregate 0.219 →
  0.166).  Documented so the next iteration does not re-spend the
  effort: the multi-start-from-box-centre / random policy is the wrong
  restart geometry for a curved valley.  The next thing to try is a
  **warm-started, curvature-aware local polish** — a quasi-Newton /
  trust-region descent that always restarts from the strategy's *best
  incumbent* (the memetic recipe scipy dual annealing itself uses),
  rather than from the box centre.  See *Next iteration ideas*.

### 2026-07-05 — Budget-adaptive `NP_init="auto"` for the DE family + structural-catalog adoption

* **What** — The L-SHADE family (``LSHADE`` and its subclasses ``JSO`` /
  ``NLSHADE_RSP`` / ``NLSHADE_LBC`` / ``LSHADE_EpSin``) now accepts
  ``NP_init="auto"`` for **budget-adaptive population sizing**.  ``"auto"``
  resolves at construction from the strategy's evaluation budget and the
  problem dimension::

      NP = clip( round( min(18·dim, budget / 12) ), max(NP_min, 6), 400 )

  The ``18·dim`` term is the CEC-2014 upper bound (Tanabe-Fukunaga); the
  ``budget / 12`` term dominates at the tight budgets Panobbgo actually
  runs, keeping ~12 generations available for the SHADE success-history
  adaptation to pay off.  The resolution happens in the base
  :meth:`LSHADE.__init__` (via :func:`panobbgo.heuristics.lshade._resolve_auto_np_init`)
  so every downstream code path — validation, ``on_start``, LPSR, the
  ``LSHADE_EpSin`` ``G_max`` estimate, and every subclass — sees a normal
  ``int`` and needs no further branching.  Falls back to the fixed
  ``NP_init=30`` when the budget is unknown.  The fixed ``int`` default is
  **unchanged** (byte-identical) — ``"auto"`` is strictly opt-in.

  The four DE candidate classes in :func:`default_structural_catalog`
  now ship ``NP_init="auto"`` instead of the fixed ``NP_init=30`` so a
  structurally-added DE arm is sized for the strategy budget rather than
  hobbled by an oversized initial swarm.  To keep the ``LSHADE.NP_init``
  ``integer_add`` catalog rule from crashing on the string sentinel,
  :func:`_find_targets` gained a ``rule_kind`` argument: numeric rule
  kinds now skip non-numeric values (``"auto"`` is ignored, never
  ``int("auto")``-crashed), while categorical rules still see strings so
  e.g. an ``F_schedule`` regime flip keeps working.

* **Why** — The single fixed ``NP_init=30`` (and even the loop specs'
  hand-pinned ``15``) is badly mistuned for Panobbgo's budgets.  A lone
  ``LSHADE`` on ``Rosenbrock_2D`` measures (6 reps, seed 42):

  | budget | ``NP_init=30`` | ``NP_init="auto"`` |
  |---|---|---|
  | 75 (quick — the nightly loop budget) | **0.036** | **0.604** |
  | 200 (standard) | 0.46 | 0.43 (within run-to-run noise) |

  At the quick-mode budget the loop actually runs, an oversized swarm
  spends nearly the whole budget on the initial random fill and never
  runs enough generations for parameter adaptation — a **~16×** score
  collapse that ``"auto"`` (which sizes to ``NP=6`` there) fixes.  At
  budget 200 the two sit within the (large) run-to-run variance of the
  single-strategy Rosenbrock cell (per-seed scores span 0.19–0.61); a
  companion 3-seed sweep of ``NP_init ∈ {15, 17, 30}`` measured
  0.42 / 0.43 / 0.46 respectively — statistically indistinguishable, i.e.
  no systematic standard-budget regression.  This directly serves the "best black box optimizer in the
  world" goal: Panobbgo's strongest single optimizers (jSO / L-SHADE were
  measured *best* on ``Rosenbrock_2D`` standalone, 0.62) were previously
  crippled at the loop's own operating budget whenever the structural
  bandit tried to add one.

  Respects the §7.3 catalog freeze — **no new mutation rules, structural
  candidates, or heuristics**.  Same candidate classes, same ops; only the
  default kwargs of existing structural candidates improved, plus a
  robustness fix to a shared heuristic.  Registry work (§9 priority (b)).

* **Scope** — Deliberately did **not** change the loop seed specs
  (``Loop_DE_Family`` pins ``NP_init=15``): those DE arms run inside a
  6-way portfolio where each optimizer gets only a fraction of the budget,
  so ``strategy.config.max_eval`` over-estimates each arm's effective
  budget and ``"auto"`` would still over-size.  The structural-catalog
  adoption is the clean win (a structurally-added DE is typically the
  dominant point-generator of its strategy, so the full budget estimate is
  right); wiring ``"auto"`` into the portfolio seed specs is left as a
  *Next iteration idea* pending a per-heuristic budget-share estimate.

* **Tests** — 8 new ``LSHADEAutoNPInitTests`` in
  ``tests/test_heuristic_lshade.py`` (resolution across budgets, floor at
  6, custom-``NP_min`` floor, unknown-budget fallback, ``on_start`` emits
  the resolved count, invalid-string / bool rejection, subclass
  inheritance) and 6 new tests in ``tests/test_self_improve.py``
  (``TestNumericRuleSkipsStringSentinel`` +
  ``TestStructuralCatalogDEAutoSizing``: the numeric-rule skip guard,
  ``_is_numeric_value``, end-to-end catalog sampling against an
  ``NP_init="auto"`` spec never crashes, and the DE candidates ship
  ``"auto"``).  Full affected suites: 802 passed.

* **Documentation updated** — ``doc/source/heuristics.rst`` (LSHADE entry
  gains the ``"auto"`` formula + measured motivation),
  ``panobbgo/heuristics/lshade.py`` (module comment + arg docstring +
  ``_resolve_auto_np_init`` docstring), the four DE subclass ``NP_init``
  docstrings, ``AGENTS.md``, ``TODO.md``, and this entry.

* **Rejected this iteration (measured negative result)** — A
  Hooke-Jeeves **pattern-move / directional momentum** for the ``Nearby``
  local search was implemented and measured across three designs (global
  bias on all points; one directed probe + isotropic majority;
  provenance-gated to Nearby's own improvements) at momentum ∈ {0.5, 1.0}
  over 5 seeds on a Rosenbrock/StyblinskiTang + ceiling-check battery.
  Every variant came in **null-to-negative** on composite (0.581–0.588 vs
  control 0.592; ``Rosenbrock_2D`` consistently *degraded* because a linear
  extrapolation overshoots the curved valley, and the resolvable battery
  ceilings 3/6 problems leaving little to measure).  Reverted per the
  AGENTS.md evidence rule.  Documented here so a future iteration does not
  re-spend the effort — the local-search direction estimate from
  mixed-heuristic ``on_new_best`` events is too incoherent to pay off; a
  *curvature-aware* step (trust-region / quadratic model) would be the
  next thing to try, not a straight pattern move.

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * **Per-heuristic budget-share for ``"auto"`` in portfolios** — size
    ``"auto"`` from ``max_eval · (arm_share)`` rather than the full
    ``max_eval`` when a DE arm shares a strategy with N other
    point-generators, then wire ``"auto"`` into the ``Loop_DE_Family``
    seed specs.  Needs a cheap estimate of each arm's realised evaluation
    share (the Rewarding bandit already tracks per-heuristic pulls).
### 2026-07-04 — `--metric aocc` workflow_dispatch A/B mechanism in the nightly cron (V2 §9.5 step 2)

* **What** — Adds a ``metric: choice[composite, aocc]``
  ``workflow_dispatch`` input to
  ``.github/workflows/self_improve_nightly.yml`` plus the matching
  ioh_worker venv sync / cache steps so the GitHub-hosted runner has
  the IOH C++ backend available at loop start.  Scheduled runs
  continue on ``composite`` (byte-identical to the pre-2026-07-04
  cron so ledger comparability is preserved); manual dispatch with
  ``metric=aocc`` invokes the loop with ``--metric aocc``, routing
  the accept/reject decision through :meth:`SelfImprover._measure_aocc`
  (:mod:`panobbgo.self_improve`) which delegates to
  :func:`panobbgo.harness_ioh.run_ioh_harness` against the mode-mapped
  IOH battery (quick / standard / full) and converts the per-instance
  AOCC values into a :class:`~panobbgo.harness.HarnessResult` the rest
  of the loop (statistical_accept, ledger writer, guard, hold-out)
  consumes unchanged.

  Workflow surface changes (single file):

  * New ``inputs.metric`` under ``workflow_dispatch`` with
    ``options: [composite, aocc]`` and ``default: composite``.  Preserves
    the pre-existing ``inputs.iterations`` / ``inputs.mode`` /
    ``inputs.confirm_accepts`` shape — the operator opt-in is a single
    dropdown next to the ``mode`` dropdown they already use.
  * New "Cache IOH worker venv" step (id ``cache-ioh-worker``) mirroring
    the tests.yml equivalent — key derived from
    ``tools/ioh_worker/pyproject.toml`` + ``tools/ioh_worker/uv.lock``,
    restore-keys degrade gracefully on lockfile bumps so the cp312
    manylinux wheel (~8 MiB) survives normal lockfile refreshes.
  * New "Sync IOH worker venv" step gated on the cache miss.  Kept
    **eager** (not conditional on ``METRIC == aocc``) so an operator
    who flips the dropdown gets a warm venv immediately — the ~2 s
    cold-cache tax is amortised across every scheduled run instead of
    surfacing as a spike on the first aocc dispatch.
  * New ``METRIC`` env variable derived from the input default; two-way
    conditional append at the end of the ``CMD`` array (mirroring the
    ``CONFIRM_ACCEPTS`` shape shipped 2026-06-27):
    ``if [ "$METRIC" = "aocc" ]; then CMD+=(--metric aocc); fi``.
  * Commit-message tag: aocc-regime dispatch runs get
    ``mode=$MODE, metric=aocc`` in the commit subject so an auditor
    grepping ``git log`` can identify A/B nights; the composite
    default preserves the historical commit-message shape.

* **Why this closes the last V2 lever** — Direct effect on the V2
  §11 success criteria:

  * **Unblocks §9.5 step 2** — the last remaining implementation-order
    task before "Enforce the catalog freeze" is the operational bar
    for any new work.  Every other V2 flag is already in the cron
    (see §9.5 step 5 dated notes 2026-06-21 / 2026-06-27).  The
    mechanism ships today; the manual A/B nights that gate the
    *default* flip (from ``composite`` to ``aocc``) are the operator's
    next lever — one line of workflow edit once the aocc regime has
    demonstrated meaningfully-more resolution than composite.
  * **Directly addresses §2.1 "no metric resolution where the loop
    operates"** — the root-cause diagnosis every other V2 symptom
    is downstream of.  AOCC is anytime and continuous — every
    evaluation moves the metric, eliminating the composite Δ=0 dead
    zone (34% of V1 mutations measured Δ = exactly 0.0000).  Local
    smoke-test evidence: a one-iteration ``--metric aocc`` run
    against the quick IOH battery on ``Rewarding_Restart / Sobol.n``
    produced ``Δ = +0.0033``, ``CI = [+0.0033, +0.0033]`` where the
    same slot has historically produced Δ = 0 exactly on the
    composite path.  The A/B nights will quantify the resolution
    delta across the full 20-iteration sweep.
  * **Preserves the§7.3 catalog freeze policy** — pure operational
    plumbing, no new mutation rules / heuristics / structural
    candidates.  Directly aligns with the freeze's *"weekly agent
    priority order: (a) merge/close open codify PRs, (b) metric &
    registry work (§9), (c) only then new rules"* — this ship is
    the **(b)** work, unblocking the freeze exit criterion.

* **Why the default stays composite** — Three practical reasons:

  1. **Ladder comparability across the pre/post transition night.**
     If the scheduled run silently switched metric the very next
     night, the summary trend block's ``seed_score`` column becomes
     apples-to-oranges (composite in ``0.02-0.08``; AOCC in
     ``0.5-0.9``) — an auditor scrolling the trend would see a
     phantom 10× improvement.  The manual A/B nights let the
     operator explicitly compare and codify the flip in a dated
     entry when the evidence supports it.
  2. **§9.5 step 2 explicitly calls for a manual A/B** — the loop
     doc's language is precise: "after one manual
     ``workflow_dispatch`` A/B comparing signal quality".  The ship
     provides the A/B *mechanism*; the A/B itself is the operator's
     job.
  3. **Iteration-cost hedge remains reversible.**  A scheduled aocc
     run costs one ioh_worker subprocess spawn per iteration (~50-100
     ms overhead) plus the per-instance AOCC computation.  On a
     20-iteration quick-mode night the aggregate overhead is bounded
     but not zero; the composite default lets the operator revert
     with zero risk if the aocc regime surfaces any unexpected
     failure mode (worker crash, IOH battery timeout, etc.) that
     wasn't caught in local smoke tests.

* **Live-test evidence** — Ran
  ``uv run python scripts/self_improve.py run --iterations 1
  --mode quick --metric aocc --base-seed 42 --ledger /tmp/aocc_smoke.jsonl``
  locally (after ``cd tools/ioh_worker && uv sync``).  The IOH
  worker spawned successfully, the AOCC harness ran end-to-end
  against the quick IOH battery, and the loop produced a single
  reject iteration (``Rewarding_Restart / Sobol.n: 32 → 36``) with a
  non-zero Δ (``+0.0033``) — smoke-test only; not a signal-quality
  claim, but confirms the wiring is intact.

* **Follow-ups seeded** (see the "Next iteration ideas" section below):

  * *A/B night: dispatch ``metric=aocc`` beside next scheduled
    composite night* — the immediate next step for the daily
    routine.  Compare no-op-rate, accept-rate, and median-Δ CI
    width across the two regimes; when aocc's <10% Δ=0 rate is
    demonstrated, flip the scheduled default in a follow-up
    single-line workflow edit.
  * *Composite battery re-base as the fallback* — the §2.1 fallback
    if the aocc A/B evidence surfaces reasons to prefer composite
    (worker instability, per-iteration cost, etc.): larger budgets,
    easier family mix, or relaxed tolerance until the composite
    median score sits in 0.3–0.6.  Not primary path; retained as an
    escape hatch.
### 2026-07-03 — `codify-scan --apply-top --apply-format` / `--apply-run-tests` hygiene flags

* **What** — Two optional flags on
  ``scripts/self_improve.py codify-scan --apply-top`` that chain the
  daily codify routine's last two manual steps into the same command:

  * ``--apply-format`` — after the write, runs ``uv run ruff format``
    on the modified files (the ``sorted(modified_files)`` list the
    driver reports in its ``Wrote N file(s):`` line).
  * ``--apply-run-tests`` — after the (optional) format step, runs
    ``uv run pytest tests/test_self_improve.py`` so the operator
    gets immediate feedback that the codify edit did not break the
    codify plumbing itself.

  Both are **inert under** ``--apply-dry-run`` (no edits landed,
  nothing to format or test) and **inert when no site needed
  editing** (the per-site direction guard skipped every candidate).
  Non-zero rc from either subprocess propagates back to the CLI
  caller so a CI wrapper surfaces the failure.  When
  ``--apply-run-tests`` succeeds, the driver's final "Next: …" line
  drops the ``uv run pytest`` clause (already done) — otherwise the
  existing message is preserved verbatim, matching the pre-flag
  operator workflow.

  New module surface in ``scripts/self_improve.py``:

  * ``_run_subprocess(cmd: Sequence[str])`` — module-level indirection
    over :func:`subprocess.run` so tests can monkeypatch a capture-
    only fake without shelling out to the real ``uv`` / ``ruff`` /
    ``pytest`` binaries.  Matches the same dependency-injection
    pattern the queued ``--open-pr`` driver in PR #275 uses for its
    ``gh`` / ``git`` sequence.
  * ``_apply_top_codify_candidate(...)`` gains two keyword-only
    parameters ``run_format`` and ``run_tests`` (both default
    ``False`` so existing callers stay byte-identical).  The
    subprocess dispatch is a straight-line if-chain matching the
    documented dry-run / no-edit / success-then-format-then-tests
    sequence.

  New CLI surface on ``codify-scan``:

  * ``--apply-format`` — bool flag, default False.
  * ``--apply-run-tests`` — bool flag, default False.

  Both parse cleanly independent of ``--apply-top`` (harmless
  no-op when the parent isn't set); the ``_cmd_codify_scan`` handler
  reads them via ``getattr(args, "apply_format", False)`` so
  hand-rolled ``argparse.Namespace``-shaped test callers continue
  to work without the two fields.

* **Why it improves Panobbgo** — three direct effects, each tied
  to the §12.3 daily routine:

  * **Closes the "run ruff, then run pytest, then commit" gap.**
    The 2026-06-30 ``--apply-top`` ship reduced the manual codify
    routine from ~30 min to ~30 s of "run one command, review the
    diff, then remember to run ``uv run ruff format`` + ``uv run
    pytest tests/test_self_improve.py`` before committing".  The
    two flags fold both of those into the same command — one
    line, one review pass, one commit.
  * **Prevents "landed but broke tests" codify PRs.**  The
    ``--apply-run-tests`` gate makes the driver fail fast on any
    edit that ships a value the seed factories can't consume
    (constructor-invariant violation, catalog-bound mismatch,
    silent import cycle).  Directly the same safety the 2026-06-30
    per-site direction guard applies at the AST layer, now
    extended to runtime semantics.
  * **Advances the §11 success criteria without adding new arms.**
    Respects the §7.3 catalog freeze (no new mutation rules /
    heuristics / structural candidates) — pure operator-usability
    plumbing.  Speeds the codify-PR cadence without changing what
    the loop can measure.

* **Documentation** —
  ``planning/SELF_IMPROVEMENT_LOG.md`` (this dated entry + the
  2026-06-30 entry's ``--apply-top --auto-format`` / ``--run-tests``
  follow-ups graduated from queued to shipped);
  ``planning/SELF_IMPROVEMENT_LOOP.md`` (§9.3 paragraph extended
  with the hygiene-flag mention);
  ``doc/source/guide_benchmarking.rst`` (new *Hygiene flags*
  sub-block under *Apply the top candidate to the working tree
  (--apply-top)*, plus the recommended-one-liner code sample);
  ``doc/source/guide.rst`` (Benchmarking summary line extended
  with the 2026-07-03 entry); ``AGENTS.md`` (new bullet under the
  V2 ship list); ``TODO.md`` (new *Recent Improvements* entry).

* **Tests** — 8 new tests in
  ``tests/test_self_improve.py::TestApplyTopHygieneFlags`` cover:

  * ``--apply-format`` alone → single ``ruff format`` subprocess
    on the modified files + `Formatting: uv run ruff format` line
    in the output.
  * ``--apply-run-tests`` alone → single ``pytest`` subprocess +
    "Running tests: …" line + the trailing "Next: …" message
    drops the "run pytest" clause.
  * Both together → format runs before tests (verified via
    subprocess call order + output substring order).
  * ``--apply-format`` failure (rc=3) → CLI returns rc=3, pytest
    subprocess is skipped (short-circuit on format failure).
  * ``--apply-run-tests`` failure (rc=2) → CLI returns rc=2 after
    format has already succeeded.
  * ``--apply-dry-run`` with both flags → zero subprocesses
    spawned + "inert under --apply-dry-run: --apply-format,
    --apply-run-tests skipped" line.
  * No-site-needed path (per-site guard finds nothing to edit)
    with both flags → zero subprocesses spawned.
  * Argparse round-trip: both flags default False + parse as True
    when passed.

  Full ``tests/test_self_improve.py`` suite: 541 → 549 tests
  (+8), all pass; ``uv run pytest`` (no ignores) reports 1762
  passed / 11 skipped IOH workers; ``uv run ruff check
  scripts/self_improve.py tests/test_self_improve.py`` clean;
  ``uv run ruff format --check ...`` clean; ``uv run pyright
  scripts/self_improve.py`` reports 0 errors.

* **Live-ledger smoke test** — Against the live ledger today
  (``planning/self_improve_ledger.jsonl``): ``uv run python
  scripts/self_improve.py codify-scan --apply-top --apply-dry-run
  --apply-format --apply-run-tests`` reports every visible
  candidate is skipped (1 structural + 3 bidirectional — the
  correct outcome per the 2026-06-30 driver's safety guards).
  Because no edits landed, the two hygiene flags don't fire even
  though they were requested — matches the "inert when no site
  needed editing" contract.

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * **--apply-open-pr hygiene composition** (once PR #275 lands
    the ``--open-pr`` driver): a single ``--apply-format
    --apply-run-tests --open-pr`` chain runs format + tests +
    ``gh pr create`` from one command.  Speculative until #275
    merges.
  * **Custom pytest scope** — ``--apply-run-tests-scope=STR`` to
    let the operator swap in a broader test path when the codify
    slot touches something outside ``test_self_improve.py``
    (e.g. a ``Restart.patience`` change should also run
    ``tests/test_analyzer_restart.py``).  Speculative — the
    codify plumbing itself lives in the fast test module and the
    default is where the real risk is.
### 2026-07-02 — `codify-scan --open-pr` driver — the final layer of V2 §9.5 step 4

* **What** — Adds a ``--open-pr`` flag on ``scripts/self_improve.py
  codify-scan`` that, after applying the top actionable kwarg
  candidate (implies ``--apply-top``), creates a git branch, commits
  the codify diff, pushes it, and opens a draft PR via ``gh pr
  create``.  Closes the last open piece of V2 §9.5 step 4 (the
  detection → value derivation → source edit → **PR** pipeline the
  three prior 2026-06-17 / 2026-06-29 / 2026-06-30 entries stood up).

  New library surface in ``panobbgo/self_improve.py`` — every helper
  is a pure function so the test suite exercises the shape without
  shelling out:

  * :func:`~panobbgo.self_improve.codify_pr_marker(candidate)` —
    machine-readable dedup marker.  Format:
    ``codify-slot: <slot_key_string>`` where ``slot_key_string``
    renders ``(class_name, param_name, op)`` as ``Class.param`` for
    kwarg candidates or ``Class::structural::op`` for structural
    candidates.  The direction is intentionally excluded so a
    same-slot opposite-direction signal is treated as "an existing
    PR already covers this slot, supersede it in review" rather than
    "open a duplicate" — matches the §12.3 step 0 lesson and the
    docstring on :attr:`CodifyCandidate.slot_key`.
  * :func:`~panobbgo.self_improve.codify_pr_title(candidate)` —
    one-line PR title.  Format:
    ``codify(<Class>.<param>): shift default <old_repr> -> <new_repr>
    (<direction>, ledger evidence)`` for kwarg candidates;
    ``codify(<Class>): <op_name> (ledger evidence)`` for structural
    candidates.
  * :func:`~panobbgo.self_improve.codify_pr_branch_name(candidate, *,
    prefix)` — stable branch name.  Format:
    ``<prefix>-<class_snake>-<param_snake>-<direction>``, non-ASCII
    collapsed to ``_``.  Default prefix ``claude/codify`` keys on the
    watcher-infrastructure ``claude/`` namespace.
  * :func:`~panobbgo.self_improve.codify_pr_body(candidate, edits, *,
    marker, base_branch)` — draft PR body in Markdown with four
    sections: **Codify slot** (marker + direction + rule kind +
    proposed value + live seed values), **Ledger evidence** (per-
    record table with date / strategy / Δ / CI / old→new /
    confirmed?), **Proposed source edit** (per-``CodifyEdit`` bullet
    list citing ``source_path:lineno``), **Test plan** (``uv run
    pytest`` + ``benchmark_harness.py compare --statistical``
    checklist).  Marker embedded in an HTML comment at the top so the
    dedup layer picks it up without polluting the human-readable
    rendering.
  * :func:`~panobbgo.self_improve.find_open_pr_for_slot(candidate,
    open_prs)` — dedup helper.  Consumes the parsed JSON output of
    ``gh pr list --state open --json number,title,body,headRefName``,
    returns the first PR whose title or body contains the candidate's
    marker (``None`` when no match).  Defensive against missing keys
    in the JSON payload — a partial gh response doesn't raise.

  New CLI surface on ``scripts/self_improve.py codify-scan``:

  * ``--open-pr`` — the driver flag.  Implies ``--apply-top`` when
    not set explicitly (opening a PR without an apply would produce
    an empty commit).  Composes with ``--apply-dry-run`` — a dry-run
    invocation prints the ``gh`` / ``git`` command sequence the
    driver *would* run without invoking any subprocess.  Skipped
    (with a note) when :func:`find_open_pr_for_slot` finds a
    matching open PR.
  * ``--pr-branch-prefix`` (default ``claude/codify``),
    ``--pr-base`` (default ``master``), ``--pr-gh-bin`` (default
    ``gh``), ``--pr-git-bin`` (default ``git``) — knobs the operator
    can tweak per-invocation without editing source.

  New CLI driver ``_open_pr_for_candidate`` in
  ``scripts/self_improve.py``:

  * Presence-checks ``gh`` and ``git`` via
    :func:`shutil.which` before touching subprocess; missing binary
    yields a clean rc=4 with an actionable diagnostic instead of a
    :class:`FileNotFoundError` deep in the git call.
  * Runs the dedup ``gh pr list`` step; parses the JSON output; skips
    (rc=0 with a ``PR #N already covers this slot`` note) on match.
  * Sequences ``git checkout -b`` → ``git add <edited files>`` →
    ``git commit -m <title>`` → ``git push -u origin <branch>`` →
    ``gh pr create --draft --base <base_branch> --head <branch>
    --title <title> --body-file <tmpfile>``.  Any step's non-zero
    return code aborts the sequence and propagates the rc so the
    workflow logs surface the failing step.
  * Accepts a ``runner`` dependency-injection hook (defaults to
    :func:`subprocess.run` with ``capture_output=True``,
    ``text=True``, ``check=False``) so the test suite intercepts
    every subprocess call without touching the real ``gh`` / ``git``
    binaries.  Same shape as the ``sources`` DI in
    :func:`derive_codify_edits` from the 2026-06-30 ship.

* **Why** — Two direct effects, each tied to a V2 §11 success
  criterion:

  * **Automates the last manual step in the codify pipeline.**  The
    prior four codify PRs (``Sobol.scramble=False`` 2026-05-31;
    ``Nearby.radius`` catalog tightening 2026-06-26;
    ``Nearby.radius`` seed shift 2026-06-28;
    :meth:`CodifyCandidate.proposed_codify_value` plumbing
    2026-06-29) each required the operator to hand-run ``gh pr
    list`` for dedup, hand-craft the branch name, hand-copy the
    ledger evidence into a Markdown body, and hand-invoke
    ``gh pr create``.  The 2026-06-30 ``--apply-top`` driver
    mechanised the source-edit step but stopped at the working-tree
    diff — the operator still had to commit + push + open the PR.
    The ``--open-pr`` driver closes that gap so the daily-routine
    codify step drops from ~30 minutes of manual GitHub work to one
    command that runs unattended in the nightly cron.
  * **Advances the §11.2 throughput criterion.**  Three codify PRs
    have shipped so far (§11.2 bar: ≥ 3 opened, ≥ 2 merged over 30
    nights — currently 3 / 2).  With the ``--open-pr`` driver, the
    marginal cost of opening the *fourth* codify PR drops to
    zero — the nightly cron can invoke ``codify-scan --open-pr`` as
    its final step and the driver's dedup guard makes the invocation
    idempotent (a re-run against the same ledger evidence surfaces
    ``PR #N already covers this slot`` and exits ``0``).  The
    cadence ceiling lifts to "whenever the live ledger surfaces a
    non-bidirectional kwarg candidate that isn't already codified".

* **Why the runner DI hook** — The nightly workflow runs on a
  GitHub-hosted runner with ``gh`` pre-installed and authenticated
  via ``GH_TOKEN``, but the CI job that runs ``pytest`` on every PR
  does not — and neither does an operator's local dev machine when
  they run the tests interactively.  Sub-processing to ``gh`` from
  inside a test would either need the binary + auth (fragile in CI)
  or a mock at the ``subprocess`` module level (leaky, and
  monkeypatching ``subprocess.run`` globally affects unrelated code
  in the same test file).  The ``runner`` argument is the clean
  answer: production code passes :func:`subprocess.run`, tests pass
  a :class:`_StubRunner` that records the exact command list and
  returns queued stubs.  Every code path — happy path, dedup match,
  missing binary, dedup rc≠0, git step rc≠0 — is exercised without
  ever invoking a real subprocess.

* **Why the marker in an HTML comment** — GitHub renders the PR body
  as Markdown, which strips ``<!-- ... -->`` from the visible
  rendering but keeps it in the raw body served by ``gh pr list
  --json body``.  The marker therefore stays greppable by the dedup
  layer *and* invisible to a human reviewer.  Alternative shapes
  considered:

  * Marker in the title — pollutes the PR list rendering (``gh pr
    list`` shows the title on every line) with a machine-readable
    identifier the reviewer doesn't need.  Rejected.
  * Marker as a distinct GitHub label — needs label management on
    the repo (which the operator's non-admin PATs can't do) and
    doesn't survive PR-body-only ``gh pr view --json`` responses.
    Rejected.
  * Marker in a hidden Markdown link ``[](#codify-slot-Class.param)``
    — GitHub renders empty-text links as invisible in the diff view,
    but the raw source still shows the marker.  Same effect as the
    HTML comment with worse operator ergonomics.  Rejected.
  * Marker in an HTML comment — ships (this entry).

* **Live-ledger smoke test** — Running
  ``uv run python scripts/self_improve.py codify-scan --open-pr
  --apply-dry-run`` against the live ledger today reports the same
  candidate-list the 2026-06-30 ``--apply-top`` smoke test surfaced
  (four visible candidates, all skipped) plus an ``Open-PR:`` block
  showing the git / gh command sequence that *would* run for the
  top actionable candidate (currently ``LatinHypercube``
  ``drop_heuristic`` from ``Loop_LocalSearch`` — surfaced as a
  structural skip note by ``--apply-top`` before the PR block even
  fires).  Once a non-bidirectional kwarg candidate clears the
  gates on a future nightly run, the driver will open a real draft
  PR without further operator intervention.

* **Test coverage** — 19 new tests in ``tests/test_self_improve.py``
  (``TestCodifyPrPrimitives`` + ``TestOpenPRCLIDriver``):

  * Pure-function primitives: marker is slot-scoped +
    direction-agnostic; structural encoding contains
    ``::structural::``; branch name slug obeys git ref rules for
    the identifiers Panobbgo ships; title / body / evidence table
    round-trip; dedup helper handles missing JSON keys defensively.
  * Driver flow: dry-run prints commands and skips the runner
    entirely; dedup match short-circuits to rc=0 with a clean note;
    happy path emits the full ``gh pr list`` → ``git checkout -b``
    → ``git add`` → ``git commit`` → ``git push`` → ``gh pr create``
    sequence in order; missing gh binary yields rc=4; dedup step rc
    propagates; git step rc aborts the sequence at the failing step.
  * CLI integration: ``--open-pr`` alone implies ``--apply-top``
    (else the resulting PR would carry an empty commit).

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * **Structural-op ``--open-pr`` support** — today the driver
    inherits the ``--apply-top`` skip for structural candidates
    (``op is not None``).  Once :func:`derive_codify_edits` grows
    an structural-edit path (the 2026-06-30 follow-up seed:
    inserting / removing a tuple entry in the ``heuristics`` /
    ``analyzers`` list literal), the ``--open-pr`` layer inherits
    it for free — the driver already threads ``candidate.op``
    through the title and branch name.
  * **``--open-pr --auto-rebase``** — a follow-up run against a
    stale branch (someone else merged a change that touched the
    same file) currently fails on ``git push`` with a non-fast-
    forward.  A ``--auto-rebase`` flag would ``git pull --rebase
    origin <base_branch>`` after the ``git checkout -b`` and re-
    push; deferred until the daily cron shows a failure caused by
    this exact interaction (which the ``--apply-top`` per-site
    guard makes rare because the guard skips sites that already
    match the proposal).
  * **``--open-pr --retry-network N``** — the cron's per-request
    retry logic (documented in the top-level system prompt) lives
    outside the driver; a ``--retry-network N`` flag would push the
    same exponential-backoff loop inside the driver so a single
    ``uv run`` invocation is resilient without a wrapper script.
    Deferred until network flakes actually surface in the nightly
    logs.
  * **Structured-JSON ``--open-pr --json``** — emits one JSON
    object per attempted (or skipped) PR so a downstream dashboard
    can compute the codify-PR cadence directly.  Speculative until
    the operator surfaces a need for cross-night trend analysis
    the ``codify-scan --json`` output doesn't already cover.
### 2026-07-01 — Structural-edit primitive for the `codify-scan --apply-top` driver (V2 §9.5 step 4 follow-up)

* **What** — Extends the 2026-06-30 ``--apply-top`` driver to handle
  the four structural codify ops (``add_heuristic`` / ``drop_heuristic``
  / ``add_analyzer`` / ``drop_analyzer``) in addition to kwarg-value
  edits.  The primitive parses the target source file, locates the
  ``(ClassName, {...})`` tuple entries inside each ``StrategySpec``'s
  ``heuristics`` / ``analyzers`` list literal, and emits
  :class:`~panobbgo.self_improve.CodifyEdit` objects that either add
  a new entry or remove an existing one — scoped to the specs listed
  in the candidate's :attr:`~CodifyCandidate.strategy_names`.  Closes
  the queued *Structural-edit primitive for the apply driver* seed
  under the 2026-06-30 entry's *Next iteration ideas* list.

  New / extended library surface in :mod:`panobbgo.self_improve`:

  * :func:`~panobbgo.self_improve._scan_source_for_structural_edits`
    — sibling of :func:`~panobbgo.self_improve._scan_source_for_kwarg_edits`
    that handles list-entry insertion / removal.  Reused by
    :func:`~panobbgo.self_improve.derive_codify_edits` when the
    candidate's ``op`` is not ``None``.
  * :func:`~panobbgo.self_improve._byte_to_lineno_col` — small helper
    to invert the ``line_starts[lineno-1] + col_offset`` convention
    used by :func:`~panobbgo.self_improve._apply_edits_to_text` so
    the new structural code can compute :class:`CodifyEdit`
    coordinates from expanded byte offsets.
  * Module-level constant
    :data:`~panobbgo.self_improve._STRUCTURAL_OPS_TO_BUCKET` mapping
    each op to its target bucket (``heuristics`` or ``analyzers``).
  * :func:`~panobbgo.self_improve.derive_codify_edits` — now
    dispatches structural candidates to the new scanner instead of
    returning an empty list.  When the candidate carries no recorded
    ``strategy_names`` the function still returns ``[]`` — refuse to
    guess which spec to modify.

  Behaviour by op:

  * **``drop_heuristic`` / ``drop_analyzer``** — one
    :class:`CodifyEdit` per matching ``(ClassName, {...})`` tuple.
    The removal span covers the tuple plus its trailing comma and
    the inter-entry whitespace so the surviving literal is
    well-formatted.  When the entry is the last one in the bucket
    (next non-whitespace is ``]``) the span extends *backwards*
    through the entry's leading newline + indent so the closing
    bracket inherits the pre-entry indentation instead of the
    entry's inner indent — a regression-guarded corner case.
  * **``add_heuristic`` / ``add_analyzer``** — one zero-width
    insertion at the position immediately after the last existing
    entry's trailing comma; the new entry ships as
    ``(ClassName, {})`` (constructor defaults — a follow-up will
    consume the candidate's :attr:`structural_kwargs` when the
    ledger's per-record kwargs converge on one shape).  For a
    completely empty bucket (e.g. ``analyzers=[]``) the insertion is
    inline: ``analyzers=[(ClassName, {})]``.

  Three safety guards keep the primitive conservative:

  * ``drop_*`` skips specs whose bucket has only one entry (else
    the surviving spec has no way to generate points / observe
    events).
  * ``drop_*`` skips specs where the target class is not in the
    bucket (nothing to drop — matches
    :func:`~panobbgo.self_improve._structural_already_codified`'s
    drop rule so re-runs are idempotent).
  * ``add_*`` skips specs where the class is already in the bucket
    (matches :func:`~panobbgo.self_improve._structural_already_codified`'s
    add rule so re-runs are idempotent).
  * A ``target_spec_names`` filter (populated from the candidate's
    :attr:`CodifyCandidate.strategy_names`) restricts edits to the
    specs the ledger accumulated evidence against, unlike kwarg
    edits which safely propagate across every matching spec.

* **Why** — Three direct effects:

  * **Unblocks the live-ledger's top structural candidate.**  As of
    the 2026-06-30 entry, the top structural candidate on the live
    ledger was ``LatinHypercube`` ``drop_heuristic`` from
    ``Loop_LocalSearch`` (``n_nights=2``, ``mean_Δ=+0.0491``) — one
    night away from clearing the daily-routine threshold.  With the
    2026-06-30 kwarg-only apply driver, the operator would have had
    to hand-remove the ``(LatinHypercube, {"div": 4}),`` tuple from
    the Loop_LocalSearch heuristics list once the evidence
    accumulated.  The structural primitive shipped here mechanises
    that step — one command instead of manual AST search + edit +
    format.
  * **Closes the structural codify gap in the daily routine
    (§12.3).**  The kwarg / structural split was the last
    remaining reason a codify iteration had to fall back to manual
    editing.  With this ship the daily routine can codify every
    surfaced candidate — kwarg tunes, categorical flips, structural
    add / drop — via ``codify-scan --apply-top`` alone.  The queued
    ``--open-pr`` driver (V2 §9.5 step 4 final layer) now gains the
    structural primitive as a bundled capability rather than a
    kwarg-only stopgap.
  * **Advances the §11.2 throughput criterion.**  The V2 bar is
    ``≥ 3`` codify PRs opened and ``≥ 2`` merged over the first 30
    nights (currently 3 / 2).  Structural codification lifts the
    kwarg-only cadence ceiling: analyzer add / heuristic drop
    candidates that surface from the ``--structural`` mutation
    catalog can now translate to source edits directly, unblocking
    a new class of throughput.

* **Live-ledger smoke test** — Running::

      uv run python scripts/self_improve.py codify-scan --apply-top
        --apply-dry-run

  against the live ledger today no longer emits a "skipped 1
  structural candidate" line.  The ``LatinHypercube``
  ``drop_heuristic`` candidate still doesn't clear the ``min_nights``
  threshold on the current 20-iter ledger, but once it does the
  driver picks it up automatically instead of asking the operator
  to hand-apply.

* **Backwards compatibility** — strictly safe:

  * All four registry factories in ``panobbgo/harness.py`` are
    byte-identical (the change is confined to
    :mod:`panobbgo.self_improve` + :mod:`scripts.self_improve`).
    No seed-spec values change.
  * The ``codify-scan`` text + JSON output for kwarg candidates is
    byte-identical.
  * For structural candidates the CLI's ``Apply-top`` block now
    prints ``selected: X [op]`` and ``target spec(s): ...`` lines
    (2026-06-30 behaviour: ``skipped N structural candidate(s)``).
    Only affects ``--apply-top`` invocations on ledgers containing
    structural candidates.
  * :func:`~panobbgo.self_improve.derive_codify_edits` returns an
    empty list for a structural candidate whose
    :attr:`CodifyCandidate.strategy_names` is empty — defensive
    against a corrupt / synthetic input rather than guessing.
  * Existing kwarg-edit tests are unchanged.
  * The bidirectional-slot safety guard in the CLI applies only to
    kwarg candidates (numeric ``"up"``/``"down"`` directions); it
    does not affect structural candidates (structural directions
    are the op name).

* **Tests** — 7 new tests in
  ``tests/test_self_improve.py``:

  * ``TestApplyCodifyEdits`` gains 6 new tests: drop-missing-class
    is no-op, no-strategy-names refuse, drop-actually-removes,
    add-actually-inserts, add-already-present is no-op, single-entry
    bucket drop is guarded, strategy_names filter honoured,
    round-trip idempotency on structural apply.
  * ``TestApplyTopCLI`` gains 3 new tests: no-matching-site
    graceful-exit, drop_heuristic actually removes, add_analyzer
    actually inserts, drop-last-entry preserves closing bracket
    alignment (regression guard for the backwards-expansion path).
  * The pre-existing
    ``test_apply_top_skips_structural_with_note`` was retired and
    replaced with
    ``test_apply_top_structural_no_matching_site_leaves_source_unchanged``
    matching the new "structural is handled" semantics.
  * The pre-existing
    ``test_derive_edits_structural_returns_empty_list`` was retired
    and replaced with
    ``test_derive_edits_structural_drop_missing_class_returns_empty``
    matching the new semantics.
  * Full ``tests/test_self_improve.py`` suite: 551 passed
    (was 544 before — net +7 tests).  ``uv run ruff check`` /
    ``uv run ruff format --check`` clean.

* **Documentation updated**

  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry.  The
    *Structural-edit primitive for the apply driver* follow-up
    seeded under the 2026-06-30 entry graduates from queued to
    shipped.
  - ``planning/SELF_IMPROVEMENT_LOOP.md``: §9.3 paragraph extended
    noting the structural source-edit layer is shipped alongside
    the kwarg layer.
  - ``doc/source/guide_benchmarking.rst``: the *Apply the top
    candidate to the working tree (--apply-top)* sub-section under
    *Cross-night codify-scan* extended with the structural-op
    behaviour (drop-removes, add-inserts, safety guards) and the
    strategy_names filter rationale.
  - ``doc/source/guide.rst``: Benchmarking summary line extended
    with the 2026-07-01 entry alongside the existing 2026-06-30
    apply-driver entry.
  - ``AGENTS.md``: new bullet under the V2 ship list referencing
    this entry.
  - ``TODO.md``: new *Recent Improvements* entry below.

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * **`--open-pr` driver with structural PR bodies**: the queued
    ``codify-scan --open-pr`` driver (V2 §9.5 step 4 final layer)
    can now populate the PR body from structural candidates using
    the ``strategy_names`` list ("this PR drops
    ``LatinHypercube`` from ``Loop_LocalSearch``, evidence:
    N_nights, mean_Δ, ...").  No new primitive needed — the
    :meth:`CodifyCandidate.to_dict` already carries the fields.
  * **``add_heuristic`` with recorded ``structural_kwargs``**: today
    the add-primitive ships ``(ClassName, {})`` — constructor
    defaults.  A follow-up could inspect each contributing record's
    ``structural_kwargs`` and, when they converge on the same
    values across all accepts, ship those instead.  Speculative
    until the live ledger surfaces a converged-kwargs add
    candidate; the empty-dict form matches the "add class to pool"
    semantic naturally.
  * **``--apply-top --auto-format`` flag**: after applying, run
    ``uv run ruff format`` on the modified file automatically.
    The current structural primitive preserves indentation for
    typical multi-line lists but the empty-bucket inline shape
    (``analyzers=[(NewClass, {})]``) would benefit from an
    automatic re-flow when the bucket subsequently gains more
    entries.
  * **Line-wrap heuristic for long constructor arguments**: today
    the add path always ships ``(ClassName, {})`` on one line.  A
    future :class:`CodifyEdit` shape variant could ship
    multi-line ``(\\n<indent + 4>ClassName,\\n<indent + 4>{...},\\n<indent>)``
    when the argument dict has enough entries to overflow the
    88-column ruff line-length limit.  Motivated once a live
    add candidate ships with enough ``structural_kwargs`` to
    warrant the reflow.

### 2026-06-30 — `codify-scan --apply-top` driver — mechanise the manual codify edit (V2 §9.5 step 4 plumbing)

* **What** — Translates the top actionable kwarg :class:`CodifyCandidate`
  into concrete AST-located source edits on every matching
  ``(ClassName, {param_name: value, ...})`` heuristic / analyzer literal
  across the four registry factories in ``panobbgo/harness.py``
  (``_make_quick_strategies`` / ``_make_standard_strategies`` /
  ``_make_full_strategies`` / ``_make_loop_strategies``).  Strictly
  additive — the existing ``codify-scan`` report is unchanged on
  default invocations.

  New library surface:

  * :class:`panobbgo.self_improve.CodifyEdit` (frozen dataclass) —
    one concrete source edit derived from a :class:`CodifyCandidate`.
    Carries the source path, factory + spec name (for traceability),
    rule kind / direction (so consumers don't re-aggregate), AST
    coordinates (``lineno`` / ``col_offset`` / ``end_lineno`` /
    ``end_col_offset``), and both the old + new source text.
    :meth:`CodifyEdit.to_dict` round-trips through ``json.dumps`` for
    the JSON-mode / log surface.
  * :func:`panobbgo.self_improve.derive_codify_edits` — given a
    candidate, walks every named factory function via
    :func:`ast.parse` and returns the list of edits without writing
    to disk.  Numeric candidates: replaces the kwarg literal with
    :meth:`CodifyCandidate.proposed_codify_value` formatted via
    :func:`repr` (boundary case: integers render without a trailing
    ``.0`` so ``Sobol(n=12)`` reads naturally).  Categorical
    candidates: same shape with the chosen literal verbatim
    (preserves bool/str types).  Structural candidates: returns an
    empty list (out of scope — see "Why" below).
  * :func:`panobbgo.self_improve.apply_codify_edits` — applies the
    edits to disk (or simulates via ``dry_run=True``), in
    reverse byte-offset order so earlier edits don't invalidate
    later coordinates.  Returns ``{source_path: new_text}`` for the
    diff / preview shape.
  * :func:`panobbgo.self_improve.apply_codify_candidate` — convenience
    wrapper that combines the two above into the single call the
    ``--apply-top`` CLI driver makes.  Returns
    ``(edits, modified_files)``.
  * :func:`panobbgo.self_improve.default_codify_apply_sources` —
    returns the default source-file + factory-name pairs the driver
    scans.  Broader than :func:`default_codify_registries` (the
    suppression scope, quick + loop only): the apply driver covers
    ``standard`` / ``full`` too so a single apply propagates to
    every sibling spec sharing the same heuristic mix — matches the
    2026-06-28 manual codify pattern (one PR shifted six sibling
    spec literals).

  New CLI surface on ``scripts/self_improve.py codify-scan``:

  * ``--apply-top`` — after printing the candidate report, take the
    top actionable kwarg candidate and apply its implied source
    edits to ``panobbgo/harness.py`` in place.  Skipped candidates
    (structural / bidirectional) are reported with a one-line note
    so the operator knows the driver isn't quietly ignoring
    evidence.
  * ``--apply-dry-run`` — print the edits the driver would apply
    but don't write to disk.  Useful for previewing.  Inert without
    ``--apply-top``.
  * ``--apply-include-bidirectional`` — override the default
    skip-on-bidirectional safety guard (for the rare case where the
    operator has a specific reason to force a default shift on a
    slot whose ``"up"`` and ``"down"`` directions are both active).

  Two **safety guards** are built into the apply driver:

  * **Per-site direction guard** (in :func:`_should_apply_at_site`):
    sites whose current value already sits at-or-beyond the
    proposal in the candidate's direction are skipped.  So
    ``BayesOpt_GP``'s deliberately-tighter ``Nearby(radius=0.05)``
    is preserved when the proposal is ``radius=0.08`` — same shape
    as the manual 2026-06-28 codify explicitly left smaller-radius
    specs alone.  The guard makes the apply **idempotent**: a
    second pass against the now-codified source derives an empty
    edit list (every site already satisfies the direction
    predicate).
  * **Bidirectional-slot skip** (in the CLI's apply-top dispatcher,
    on by default): if the same ``(class_name, param_name)`` slot
    appears with both ``"up"`` and ``"down"`` directions anywhere
    in the full candidate list — including already-codified ones,
    so a freshly-codified ``"up"`` whose ``"down"`` sibling is
    still active doesn't trigger a re-shift — the candidate is
    skipped.  The right action for bidirectional slots is a
    *catalog bound update* via ``--widen-bounds``; applying either
    direction's default shift would guess against contradictory
    ledger evidence.

* **Why** — Three direct effects, each tied to a V2 §11 success
  criterion:

  * **Closes the manual-edit gap in the daily routine (§12.3).**
    The four ledger-evidence-driven codify PRs to date
    (``Sobol.scramble=False`` 2026-05-31; ``Nearby.radius`` catalog
    tightening 2026-06-26; ``Nearby.radius`` seed shift 2026-06-28;
    :meth:`CodifyCandidate.proposed_codify_value` plumbing
    2026-06-29) each required the operator to hand-find every
    sibling spec literal, edit each one, re-format, and re-test —
    the 2026-06-28 entry alone touched six sibling specs across
    four registry tiers.  The driver mechanises that step: one
    command (``--apply-top``) produces the same file edits the
    manual routine would have, with the same per-site direction
    guard the manual routine applied implicitly.
  * **Unblocks the queued ``--open-pr`` driver (V2 §9.5 step 4).**
    The full automation has three layers — detection
    (``aggregate_codify_candidates``, shipped 2026-06-17), value
    derivation (:meth:`CodifyCandidate.proposed_codify_value`,
    shipped 2026-06-29), and source editing (this entry).  All
    three are now library primitives; the queued ``--open-pr``
    driver wraps them with a ``gh pr create`` call and a PR body
    populated from :meth:`CodifyCandidate.to_dict`.  The remaining
    work is the ``gh`` integration and the dedup-against-open-PRs
    pass, not any new primitive.
  * **Advances the §11.2 throughput criterion.**  Three codify PRs
    have shipped so far (the V2 bar is ≥ 3 opened, ≥ 2 merged
    over the first 30 nights — currently 3 / 2).  With the apply
    driver, opening the *fourth* codify PR (whenever the live
    ledger surfaces a non-bidirectional kwarg candidate that isn't
    already codified) drops from ~30 minutes of careful manual
    editing to ~30 seconds of running one command and reviewing
    its diff.  The cadence ceiling lifts proportionally.

* **Why structural candidates are out of scope (initial ship)** —
  The kwarg-edit path is a pure value substitution in a dict
  literal; the AST coordinates uniquely identify the value node and
  the replacement is a literal text swap.  Structural edits
  (``add_heuristic`` / ``drop_heuristic`` / ``add_/drop_analyzer``)
  require *inserting* or *removing* a tuple element in the
  ``heuristics`` / ``analyzers`` list — which the AST can represent
  cleanly but the source-text edit is more invasive (need to
  re-flow the list literal, preserve trailing commas, decide
  formatting).  Out of scope for the initial ship; the structural
  case stays manual, with the apply driver printing
  ``skipped N structural candidate(s) — apply manually for now``
  so the operator knows the driver isn't quietly ignoring evidence.

  Live-ledger relevance: the current top candidate today is
  ``LatinHypercube`` ``drop_heuristic`` from ``Loop_LocalSearch``
  (n_nights=2, mean_Δ=+0.0491).  The driver surfaces it in the
  skip note; a follow-up structural-apply driver can pick this up
  whenever the bandit accumulates one more night of evidence.

* **Why bidirectional is skipped by default** — On the live
  ledger today, three of the four visible candidates are
  bidirectional: ``Sobol.n`` (both up and down accepts across
  3-4 nights each), ``Nearby.radius`` (the up direction was just
  codified 2026-06-28 → ``0.124``; the down direction has 4
  accepts at lower values).  The widening detector
  (``--widen-bounds``) correctly classifies these as "should be a
  catalog bound update, not a default shift" — applying either
  direction's default shift would oscillate the bandit between
  contradictory signals.  The safety guard makes the apply driver
  defer to ``--widen-bounds`` (which has its own dedicated codify
  path the 2026-06-26 ``Nearby.radius`` tightening exercised) for
  these cases.

  The override (``--apply-include-bidirectional``) exists for the
  edge case where the operator has a specific reason to force the
  shift — e.g. the up evidence is stale (predates a relevant
  source change) but the down evidence is fresh, or one direction
  has 10× the night count of the other.  Not the recommended
  path; the daily routine should prefer ``--widen-bounds`` for
  bidirectional slots.

* **Live-ledger smoke test** — Running
  ``uv run python scripts/self_improve.py codify-scan --apply-top
  --apply-dry-run`` against the live ledger today reports::

      Apply-top:
        skipped 1 structural candidate(s) — the apply driver
        currently handles kwarg edits only.  Apply structural
        candidates manually for now (see V2 §9.5 step 4 in
        planning/SELF_IMPROVEMENT_LOG.md).
        skipped 3 bidirectional candidate(s) — same (class, param)
        slot fired in both 'up' and 'down' directions.  Use
        --widen-bounds for catalog bound updates (the recommended
        action), or pass --apply-include-bidirectional to override.
        (every visible candidate was skipped — nothing to apply)

  Exactly the *correct* outcome on the live ledger — the four
  visible candidates today are all either structural or
  bidirectional, so the driver refuses to ship a questionable
  change.  The next time the bandit surfaces a clean
  unidirectional kwarg candidate (e.g. a new ``COBYQA.scale``
  ``False`` accept or a fresh ``LSHADE.archive_factor`` shift),
  the driver will apply it in one command.

* **Backwards compatibility** — strictly safe:

  * All four registry factories in ``panobbgo/harness.py`` are
    byte-identical (the change is in :mod:`panobbgo.self_improve`
    + :mod:`scripts.self_improve` only).  No seed-spec values
    change.
  * The ``codify-scan`` text + JSON output is byte-identical
    without ``--apply-top`` set.
  * Library additions only — no signature changes to existing
    public functions.  :func:`derive_codify_edits` returns an
    empty list for every existing-style structural candidate, so
    a caller that mistakenly passes a structural candidate gets
    a graceful no-op rather than a crash.
  * The CLI dispatcher uses ``getattr(args, "apply_top", False)``
    so existing test invocations that pass a hand-rolled
    namespace without the new attributes continue to work
    byte-identically.

* **Tests** — 25 new tests in
  ``tests/test_self_improve.py``:

  * ``TestApplyCodifyEdits`` (18 tests) — library-level tests
    covering numeric / categorical / structural candidates,
    per-site direction guard (sites already at-or-beyond proposal
    skipped), dry-run preserves source, idempotent re-apply,
    missing source / invalid Python / unknown factory return
    empty list gracefully, ``to_dict`` JSON round-trip,
    ``default_codify_apply_sources`` shape.
  * ``TestApplyTopCLI`` (7 tests) — end-to-end CLI tests using a
    synthetic harness snippet in ``tmp_path``: dry-run writes
    nothing, real apply writes the file, bidirectional skip
    on by default, override flag works, structural-only ledger
    skipped with note, no-candidates graceful exit, already-
    codified yields no edits.

  Smoke-checked the full suite: ``uv run pytest`` reports 1779
  passed / 11 skipped (the IOH worker-dependent set).
  ``uv run ruff check panobbgo/self_improve.py
  scripts/self_improve.py`` and ``uv run ruff format --check
  panobbgo/self_improve.py scripts/self_improve.py`` both clean.

* **Documentation updated**

  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry; the
    ``codify-scan --apply-top driver`` follow-up seeded under
    the 2026-06-29 entry graduates from queued to shipped.
    Next iteration ideas seeds the ``--open-pr`` follow-up that
    builds on top.
  - ``planning/SELF_IMPROVEMENT_LOOP.md``: §9.3 follow-up
    paragraph noting the driver lands as the third layer
    (detection → value derivation → source editing → ``--open-pr``)
    of the V2 §9.5 step 4 stack.
  - ``doc/source/guide_benchmarking.rst``: new *Apply the top
    candidate to the working tree (--apply-top)* sub-section under
    *Cross-night codify-scan* with the safety-guard rationale and
    the operator workflow (preview → apply → test → commit → PR).
  - ``doc/source/guide.rst``: Benchmarking summary line extended
    with the 2026-06-30 entry alongside the existing 2026-05-31 /
    2026-06-26 / 2026-06-28 / 2026-06-29 codify entries.
  - ``AGENTS.md``: new bullet under the V2 ship list referencing
    this entry.
  - ``panobbgo/self_improve.py``: full docstrings on every new
    function / dataclass explaining the per-rule-kind branching,
    the direction-guard policy, the dry-run semantics, and the
    self-stability invariant the queued ``--open-pr`` driver
    depends on.
  - ``TODO.md``: new "Recent Improvements" entry below.

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * ~**`codify-scan --open-pr` driver (V2 §9.5 step 4 final
    layer)**~ — **shipped 2026-07-02**.  See the dated entry
    above.  All three queued sub-pieces landed together: dedup via
    :func:`~panobbgo.self_improve.find_open_pr_for_slot` /
    :func:`~panobbgo.self_improve.codify_pr_marker`; PR body
    template as :func:`~panobbgo.self_improve.codify_pr_body`;
    branch naming as
    :func:`~panobbgo.self_improve.codify_pr_branch_name`
    (default ``claude/codify-*`` matches the watcher
    infrastructure).
  * **Structural-edit primitive for the apply driver** — extend
    :func:`derive_codify_edits` to support ``add_/drop_heuristic`` /
    ``add_/drop_analyzer`` candidates by emitting a richer
    :class:`CodifyEdit` shape that targets a list literal
    insertion / removal rather than a single value substitution.
    Live-ledger motivation: the current top kwarg candidate
    today is the ``LatinHypercube`` ``drop_heuristic`` from
    ``Loop_LocalSearch`` (n_nights=2, mean_Δ=+0.0491), one more
    night away from clearing the daily-routine threshold.  Once
    structural candidates start surfacing as the *top* actionable
    evidence, the structural-edit primitive moves from
    speculative to motivated.
  * **`--apply-top --auto-format` flag** — after applying the
    edits, run ``uv run ruff format`` on the modified file
    automatically.  Today the apply driver doesn't touch
    formatting (the AST coordinates already preserve indentation
    / surrounding whitespace).  Speculative until ledger evidence
    surfaces a slot whose edit needs re-flowing.
  * **`--apply-top --run-tests` flag** — after applying, run
    ``uv run pytest tests/test_self_improve.py`` (the most
    relevant suite) so the operator gets immediate feedback on
    whether the apply broke anything.  Trivial addition; deferred
    because the manual operator already runs tests before
    committing.

### 2026-06-29 — `CodifyCandidate.proposed_codify_value()` — centralise the median+round-outward policy (V2 §9.5 step 4 plumbing)

* **What** — Adds a new method
  :meth:`panobbgo.self_improve.CodifyCandidate.proposed_codify_value`
  that computes the seed value a codify edit would ship, and surfaces
  it in the ``codify-scan`` CLI report (new
  ``proposed codify value:`` line on every actionable candidate) and
  the JSON payload (new ``proposed_codify_value`` field).  Pure
  additions to ``panobbgo/self_improve.py`` and the
  ``_print_codify_candidate`` helper in ``scripts/self_improve.py`` —
  no behaviour change for the existing ``codify-scan`` consumers,
  byte-identical for every existing JSONL ledger and test invocation
  that does not read the new field.

  The rounding policy:

  * **Numeric kwarg, direction='up'/'down'** —
    ``median(new_values)`` rounded outward in ``direction`` to 3
    significant digits (floats) or :func:`math.ceil` / :func:`math.floor`
    (``integer_add``).  Captured in the new
    :func:`panobbgo.self_improve._round_outward_to_significant` helper
    so the same policy applies anywhere a codify value is derived.
  * **Categorical** — the chosen literal (``False`` for the
    ``Sobol.scramble=False`` direction etc.), preserved as the
    original Python type (no string-coercion that would break a
    bool-typed constructor argument).
  * **Structural ops** — returns ``None``.  Structural codification
    adds / drops a class, it has no kwarg value; callers consult
    :attr:`CodifyCandidate.class_name` and :attr:`CodifyCandidate.op`
    directly.

  Self-stability invariant — the rounded value satisfies
  :func:`_candidate_already_codified` on the next scan:

  * ``direction="up"``: ``proposed >= median(new_values)`` →
    ``max(live) >= median(new_values)`` after codify → suppressed.
  * ``direction="down"``: ``proposed <= median(new_values)`` →
    ``min(live) <= median(new_values)`` after codify → suppressed.

  This is the property the queued ``--open-pr`` driver needs: the
  source edit it ships must cleanly remove the candidate from the
  work-list, otherwise it would re-open the same PR every night.

* **Why** — Three direct effects:

  * **Closes the manual computation gap** — every prior codify PR
    (PR #271's ``Nearby.radius`` seed shift, the 2026-06-26 catalog
    tightening, the 2026-05-31 ``Sobol.scramble=False`` codify) had to
    hand-compute the median of the accepted ``new_values`` and pick
    a rounding policy.  PR #271's description spells it out:
    "Median ``0.123105``; exact mode ``0.123105`` (three accepts at
    the same value).  The shipped seed is rounded slightly outward to
    ``0.124`` so the ``_candidate_already_codified`` predicate
    ``max(live) >= median(new_values)`` cleanly suppresses the
    candidate next night."  This entry codifies that policy into the
    library so every future codify PR (manual or automated) uses
    the same rule.
  * **Unblocks the queued ``--open-pr`` driver (V2 §9.5 step 4)** —
    the driver's core "what value to ship?" question is answered by
    this method.  When the driver lands it consumes the new field
    directly; until then the manual codify routine reads the value
    from the report header instead of computing it by hand.
  * **Pairs with the existing suppression machinery** — the
    self-stability invariant ties the new method to the existing
    :func:`_candidate_already_codified` predicate (shipped 2026-06-18
    in the §9.3 already-codified detection).  Together they form a
    closed loop: the method proposes a value, the predicate confirms
    it suppresses the candidate next night, the codify scan no longer
    re-surfaces it.

* **Live ledger verification** — running
  ``uv run python scripts/self_improve.py codify-scan`` against the
  live ledger surfaces four actionable candidates with the proposed
  values:

  * ``Nearby.radius`` direction=up: **0.124** (matches PR #271 exactly).
  * ``Sobol.n`` direction=down: **12** (the floor of median([8, 12, 12, 12]) = 12,
    matches the deferred-codify note in
    ``planning/SELF_IMPROVEMENT_LOG.md`` for the manual companion).
  * ``Nearby.radius`` direction=down: **0.0809** (down-rounded median).
  * ``Sobol.n`` direction=up: **22** (ceil of median([20, 24, 20, 24]) = 22).

* **Backwards compatibility** — strictly safe:

  * All five fields on the existing :func:`CodifyCandidate.to_dict`
    return value remain unchanged; the new
    ``proposed_codify_value`` key is appended, so a JSON consumer
    that filters by known keys (no ``**`` spread) sees the same data.
  * The ``codify-scan`` text report gains one optional line per
    candidate (``proposed codify value: <value>``); existing CLI
    tests that assert specific text fragments pass unchanged.
  * No mutation rule changes, no catalog modifications, no new
    bandit arms — strictly compatible with the §7.3 catalog freeze
    (the V2 priority pre-empts new arms until the loop resolves
    them; this entry is loop infrastructure, not a new arm).

* **Tests** — 19 new tests in the new
  ``tests/test_self_improve.py::TestProposedCodifyValue`` class
  covering:

  * Direct unit tests of
    :func:`_round_outward_to_significant`: matches PR #271's
    ``0.123105 → 0.124``; rounds ``0.080996535 → 0.0809``;
    handles ``0.0`` (returned unchanged); negative-value
    sign-flip direction handling; invalid direction raises;
    non-finite inputs pass through.
  * Numeric ``direction="up"``: ``Nearby.radius`` end-to-end
    against synthetic records returns ``0.124``.
  * Numeric ``direction="down"``: integer rule
    (``Sobol.n``) returns ``12``; the result is an ``int`` not a
    ``float`` (preserves heuristic constructor signature).
  * Numeric ``direction="up"``: integer rule returns ``22``.
  * Categorical: returns the chosen value (``False``) verbatim
    with the boolean type preserved.
  * Structural: returns ``None`` (no kwarg value).
  * Empty ``new_values``: returns ``None`` (defensive).
  * Self-stability: ``_candidate_already_codified(c, [proposed])``
    is ``True`` for ``up`` direction, ``down`` direction, and
    integer rules.  This is the invariant the queued
    ``--open-pr`` driver depends on.
  * ``to_dict()``: the new field is present, JSON-serialisable,
    and preserves boolean type for categorical candidates.

  Plus one new assertion line on
  ``TestCodifyScanCLI.test_realistic_two_night_pattern_surfaces_candidate``
  (verifies the ``proposed codify value: 0.125`` line appears) and
  one on
  ``TestCodifyScanCLI.test_json_mode_emits_one_object_per_candidate``
  (verifies the JSON payload carries ``proposed_codify_value: 0.125``).

  All 19 new tests pass.  The full ``tests/test_self_improve.py``
  suite (516 tests) passes; ``uv run ruff check`` /
  ``uv run ruff format --check`` /
  ``uv run pyright panobbgo/self_improve.py scripts/self_improve.py``
  all clean.

* **Documentation updated**

  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry.
  - ``planning/SELF_IMPROVEMENT_LOOP.md``: §9.3 ``--open-pr`` paragraph
    extended with a bullet pointing at the new method and its
    self-stability invariant.
  - ``doc/source/guide_benchmarking.rst``: the codify-scan
    ``proposed codify value:`` line documented alongside the
    suppression rule.
  - ``doc/source/guide.rst``: Benchmarking summary line extended with
    the 2026-06-29 entry alongside the existing 2026-05-31 /
    2026-06-26 / 2026-06-27 / 2026-06-28 codify entries.
  - ``AGENTS.md``: new bullet under the V2 ship list.
  - ``panobbgo/self_improve.py``: full docstring on the new method
    explaining the per-rule-kind branching, the self-stability
    invariant, and the manual-codify policy lineage.
  - ``TODO.md``: new "Recent Improvements" entry below.

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * **`codify-scan --apply-top` driver (working-tree edit only)** —
    one step toward the full ``--open-pr`` driver: read the top
    actionable candidate, derive the source edit (find every
    matching spec factory in ``panobbgo/harness.py`` and replace
    the literal old value with :meth:`proposed_codify_value`), and
    write the change to the working tree.  Operator runs tests and
    opens the PR manually.  Skips the PR-creation and ``gh pr
    list`` deduplication that the full ``--open-pr`` driver still
    needs, but mechanises the next-most-tedious manual step.
  * **`--open-pr` driver itself (V2 §9.5 step 4)** — translates
    each surfaced candidate into a draft PR with the ledger
    evidence in the PR body.  The new
    :meth:`CodifyCandidate.proposed_codify_value` is the value the
    driver edits the source to; the
    :func:`_candidate_already_codified` predicate verifies the
    same value would suppress the candidate next night before the
    PR is opened (sanity-check on the rounding).
  * **Log-space rounding for `log_uniform_perturb`** — the current
    helper rounds in linear space (3 sig digits).  For
    ``log_uniform_perturb`` rules the natural metric space is
    *log* — a future refinement could compute the median in
    log-space and round outward in log-space, then exp-transform
    back.  Speculative until live evidence shows the linear
    rounding produces values that the codified rule then needs to
    re-tune.

### 2026-06-28 — Codify `Nearby.radius` seed shift from `0.1` to `0.124` (manual codify-scan output)

* **What** — Shifts the ``Nearby.radius`` seed value from ``0.1`` to
  ``0.124`` in five sibling :class:`StrategySpec` factories that share
  the same heuristic mix:

  * :func:`panobbgo.harness._make_quick_strategies` — the
    ``Rewarding_Diverse`` spec (the seed every codify-scan
    contributing record was generated against).
  * :func:`panobbgo.harness._make_standard_strategies` — the
    ``Rewarding_RegionUCB`` spec (identical heuristic mix + the
    RegionUCB arm) and the ``UCB_Diverse`` spec (same heuristic mix
    under the UCB bandit).
  * :func:`panobbgo.harness._make_full_strategies` — the
    ``Thompson_Diverse`` spec (Thompson-sampling counterpart of the
    diverse mix).
  * :func:`panobbgo.harness._make_loop_strategies` — both
    ``Loop_RegionUCB`` and ``Loop_Restart`` (the catalog-exercising
    loop registry's two specs that include :class:`Nearby`).

  Pure seed-spec value update — no new arms, no catalog edit, no
  constructor change.  Same shape as the 2026-05-31
  ``Sobol.scramble=False`` codify (the first ledger-evidence-driven
  default change) and the 2026-06-26 ``Nearby.radius`` catalog
  tightening (the second).

* **Why** — :command:`uv run python scripts/self_improve.py codify-scan`
  against the live ledger surfaces ``Nearby.radius [log_uniform_perturb]
  direction=up`` as the strongest current codify candidate (9 accepts
  across 8 distinct nights, 2026-05-26 → 2026-06-18, every contributing
  record's per-record ``ci_low > 0``, pooled per-record CI
  ``[+0.0365, +0.0658]``).  The accepted ``new_value`` distribution
  (chronological)::

      0.100885, 0.108882, 0.105456, 0.129437, 0.123105, 0.135257,
      0.134681, 0.123105, 0.123105

  has median ``0.123105`` and an exact mode of ``0.123105`` (three
  accepts at the same value).  The shipped seed is rounded slightly
  outward to ``0.124`` so the
  :func:`panobbgo.self_improve._candidate_already_codified` predicate
  cleanly suppresses the candidate from future codify-scan reports
  (the ``"up"`` predicate ``max(live) >= median(new_values)`` requires
  ``max(live) >= 0.123105``; ``0.124 >= 0.123105`` is True, so the
  candidate is correctly marked ``already_codified=True`` next night).

  Three direct effects:

  * **Bandit proposes from the consensus, not the prior** — the
    ``log_uniform_perturb`` rule samples symmetric perturbations
    around the *current* spec value.  Pre-codify the bandit had to
    pull itself out of the ``0.1`` neighbourhood every night through a
    chain of accept events; post-codify it starts at ``0.124`` and
    explores around the consensus.  Tighter per-night exploration
    means more proposals land in the productive region and the
    bandit's per-arm Beta posterior updates on values drawn from a
    meaningfully narrower distribution — directly improving the §11.1
    *resolution* criterion on the same nightly budget.
  * **Persists the loop's accumulated knowledge** — the V1 ladder
    discards every accept at the end of each night because the only
    durable channel is manual codification (§2.3 in the V2 loop
    diagnosis).  Eight nights of consistent ``"up"`` accepts on a
    single arm is the canonical "manual codify" pattern — exactly the
    cross-night-evidence shape that the
    :command:`codify-scan --open-pr` driver will automate once it
    lands (§9.5 step 4).  Until then, manual codification is the
    persistence mechanism.
  * **Advances the §11.2 *throughput* criterion** — the V2 success
    bar is "≥ 3 codify PRs opened from ledger evidence, ≥ 2 merged
    over the first 30 nights".  Today's ledger count (this entry
    inclusive) is **3 ledger-evidence-driven codify PRs**:
    ``Sobol.scramble=False`` (2026-05-31, merged),
    ``Nearby.radius`` catalog tightening (2026-06-26, merged), and
    ``Nearby.radius`` seed shift (this entry, open).  The codify
    cadence is on-track relative to the V2 target.

* **Pairs cleanly with 2026-06-26** — the catalog tightening
  ``(0.005, 0.5) → (0.032, 0.313)`` and this seed shift act on
  different layers of the same parameter:

  * Catalog bound (2026-06-26): defines the *range* the bandit can
    propose values from for the ``Nearby.radius`` arm.  ``0.124`` sits
    comfortably inside the new ``[0.032, 0.313]`` window with ~3.9×
    headroom upward and ~3.9× headroom downward.
  * Seed value (2026-06-28, this entry): defines the *centre* the
    ``log_uniform_perturb`` rule perturbs around.  Combined effect:
    the bandit now explores a ~2.5× window in log-space centred on the
    consensus value, instead of a ~16× window centred on a value the
    bandit had consistently moved away from.

  The two changes were intentionally split into separate PRs because
  they have different reversal characteristics — a catalog change is a
  pure-policy update, while a seed change shifts the constructor
  invocation in user-visible benchmark output.

* **Backwards compatibility** — strictly safe:

  * The :class:`~panobbgo.heuristics.nearby.Nearby` constructor
    default (``radius=1.0 / 100 = 0.01``) is unchanged.  Every direct
    caller that does not pass ``radius=`` continues to receive ``0.01``.
  * The catalog rule key
    ``(Nearby, radius, log_uniform_perturb)`` is unchanged, so the
    pre-codify Beta posterior accumulates seamlessly across the
    seed change — the bandit only sees a different starting point.
    Ledger replay through
    :meth:`AdaptiveMutationSampler.prime_from_ledger` continues to
    map every historical proposal onto the same arm.
  * The :func:`panobbgo.self_improve._find_targets` "param already in
    kwargs" predicate still fires (``0.124 != 0.01`` constructor
    default), so the catalog rule continues to fire on every affected
    spec — the codify is a value shift, not a deactivation.
  * Four built-in factory locations updated; the IOH-battery seed
    (``Rewarding_Restart`` in :mod:`panobbgo.harness_ioh`), the legacy
    :func:`panobbgo.benchmark.create_standard_strategies` factory, and
    the smaller-radius BayesOpt / CMAES specs (``0.05``) are
    intentionally left at their existing values — different evidence
    contexts, separate codify decisions.

* **Tests** — no test assertion changes required.

  * The ``TestCodifyScanCLI`` /
    ``TestAlreadyCodifiedAnnotation`` suites construct their own mock
    factories and ``new_values`` arrays so they are insulated from
    the live registry change.
  * ``TestLoopRegistry.test_quick_registry_covers_few_rules`` /
    ``test_loop_registry_exercises_full_catalog`` count rule
    activations rather than asserting specific kwarg values — the
    ``Nearby.radius`` rule continues to fire (``0.124 != 0.01``
    constructor default), so the active-rule count stays at 4 in
    quick mode and 44 in the loop registry.
  * Smoke-checked the full test suite: 1127 tests + 583 specific
    self_improve/harness tests pass byte-identically; no test depended
    on the literal ``0.1`` seed value.
  * Smoke-checked the codify-scan output: the live ledger now
    surfaces ``3 candidates surfaced (of 5; 2 already codified,
    hidden)`` — the ``Nearby.radius direction=up`` candidate is
    correctly suppressed by the
    :func:`_candidate_already_codified` predicate (the
    ``Sobol.scramble=False`` historical suppression is preserved).

* **Documentation updated**

  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry; the *Codify
    persistent wins* idea from §12.3 (a rule with repeated confirmed
    accepts → a PR changing the default kwarg) graduated from
    standing-instruction-with-no-PR-yet to shipped on the
    ``Nearby.radius`` slot.  The ledger evidence is now ~9 accepts
    deep; the next manual codify candidate is open for the
    ``Sobol.n`` slot once the auto-tune classifies it as a clear
    tightening rather than ``"widens current"``.
  - ``planning/SELF_IMPROVEMENT_LOOP.md``: §11.2 throughput-count
    note bumped to ``3 codify PRs opened`` (this entry inclusive).
  - ``doc/source/guide_benchmarking.rst``: *Codify-scan suppression*
    sub-section extended with a paragraph noting that the
    ``Nearby.radius`` seed shift to ``0.124`` is the second
    ledger-evidence-driven default change to land via manual codify
    (after ``Sobol.scramble=False``), and is the canonical example of
    a numeric ``"up"`` direction suppression because the live ledger
    now exhibits it.
  - ``doc/source/guide.rst``: Benchmarking summary line extended with
    the 2026-06-28 entry alongside the existing 2026-05-31 /
    2026-06-26 codify entries.
  - ``AGENTS.md``: new bullet under the V2 ship list pointing at
    this entry.

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * **``Sobol.n`` seed shift (manual companion to 2026-06-28)** —
    accumulating evidence (4 accepts at ``new_value ∈ {8, 12, 12, 12}``
    across 4 nights for ``direction=down``) suggests a future codify
    from ``Sobol.n = 16`` to ``Sobol.n = 12``.  Deferred today
    because the widening detector classifies the bidirectional
    pattern as ``"widens current"`` (mixed signal: 4 accepts also in
    ``direction=up`` at ``new_value ∈ {20, 24, 20, 24}``).  Re-visit
    once more nights of evidence cluster the observed range more
    tightly in one direction.
  * **Re-run the widening detector on post-shift ledger evidence** —
    the 2026-06-26 catalog tightening + this 2026-06-28 seed shift
    together meaningfully change the bandit's exploration regime.
    A few nights from now, re-running ``codify-scan --widen-bounds
    --widen-auto-tune`` will show whether the bandit converges on a
    tighter consensus window or starts exploring outward — both
    outcomes inform whether a second-round catalog adjustment is
    warranted.

### 2026-06-27 — Flip the nightly cron to `--confirm-accepts` (V2 §6.4 / §9.5 step 5 completion)

* **What** — Closes the *Flip the nightly cron to `--confirm-accepts`*
  follow-up seeded under the 2026-06-14 same-night confirmation gate
  ship and the 2026-06-21 V2 §9.5 step 5 partial flip.  Pure
  ``.github/workflows/self_improve_nightly.yml`` edit — no Python /
  test / documentation-mode changes outside the loop docs themselves:

  * The ``Run self-improvement loop`` step now appends
    ``--confirm-accepts`` to the ``scripts/self_improve.py run``
    invocation by default.  The command is constructed as a bash
    array (``CMD=(…)``) so the conditional ``--confirm-accepts``
    append composes cleanly under ``set -euo pipefail``; the prior
    long backslash-continued single-call form would have required
    fragile quoting to make the toggle conditional.
  * A new ``workflow_dispatch.inputs.confirm_accepts`` boolean
    input is exposed with ``default: true``.  Scheduled (cron) runs
    do not consume ``workflow_dispatch`` inputs, so the run step's
    ``CONFIRM_ACCEPTS: ${{ github.event.inputs.confirm_accepts ||
    'true' }}`` fall-through promotes the gate to the default
    everywhere — scheduled *and* manual.  The operator can opt back
    into the screen-only regime for the explicit A/B comparison by
    setting ``confirm_accepts=false`` in the manual dispatch UI.
  * The in-workflow comment block is updated: the
    ``--confirm-accepts`` paragraph moves out of the "NOT enabled"
    bullet list and into the active-flags list with a one-line
    rationale (closes §2.2 *Accept → rollback churn*); the
    workflow_dispatch toggle is called out as the A/B escape hatch.

* **Why** — The §2.2 V1 diagnosis (15 / 16 V1 guard checks rolled
  the ladder back) is the *single* open V1 symptom in the loop's
  diagnosis after the 2026-06-21 §9.5 step 5 partial flip.  The
  structural fix is the same-night confirmation gate (§6.4, shipped
  2026-06-14): every screening-accepted candidate is re-measured on
  a fresh ``randomize_iteration`` (plus the first hold-out base_seed
  when configured), and promotion only happens when the *pooled*
  (screen + confirm) bootstrap CI still clears ``eps_accept``.  A
  screening noise-spike cannot drive a permanent promotion because
  the confirmation batch is independent and the pooled CI rules it
  out.

  Three direct effects:

  * **Closes §2.2** — guard rollbacks of *screening* accepts have
    been ~94 % of guard activations in the live ledger (44 rollbacks
    / 72 guard checks across the most recent 420 iterations).  Under
    the gate, those rollbacks now happen *pre-record* as
    ``confirm_reject`` entries; the ladder no longer churns through
    them and the bandit consumes the post-confirmation reward (per
    §7.4 graded shaping, ``r ≈ 0.5 + Δ/eps_scale`` for honest
    near-miss rejects).
  * **Unblocks §11.3** — the success criterion "zero guard rollbacks
    of *confirmed* accepts" is structurally measurable only when the
    gate is on (otherwise every accept is a screening accept by
    definition and the criterion is vacuous).  Future guard
    rollbacks of confirmed accepts now qualify as anomalies worth
    surfacing per the §6.3 update language.
  * **Unblocks §11.2** — codify-scan currently produces no
    ``confirmed=True`` records (all evidence is "legacy: no
    confirmation gate").  Once the gate runs nightly, future
    candidates accumulate confirmation flags and the
    ``--confirmed-only`` codify-scan filter becomes meaningful;
    operators can route higher-confidence codify PRs by reading
    only the confirmed-evidence subset.

* **Why now (the held-back hedge re-evaluated)** — The 2026-06-21
  V2 §9.5 step 5 partial-flip note explicitly held this lever back
  pending a manual ``workflow_dispatch`` A/B because
  ``--confirm-accepts`` is the only V2 flag with meaningful
  per-iteration cost (~2-3× screening at worst).  Re-measuring that
  cost against the live ledger:

  * At quick mode each iteration runs ~3 problems × 3 reps × 75
    evaluations on 2 strategies — wall-clock ~15 s.  20 iterations
    is ~5 min, well inside the V1 §2.5 ~94 % idle compute slack
    that the 2026-06-21 ``--registry loop`` flip did not erase
    (the loop registry has 7 specs vs the quick registry's 2, so
    per-iteration cost is now ~10-15 min — still ≤ 20 % of the
    90-min cap).
  * The gate only fires on *accepts*.  The live ledger reports a
    3.6 % accept rate (14 / 420 informative iterations after
    no-op exclusion); 20 iterations average ~0.7 accept events
    per night.  Worst-case confirm cost: 0.7 × 2 × 15 s ≈ 30-60 s
    per night.
  * Combined: the cron's per-night cost rises from ~10-15 min to
    ~10-16 min, comfortably within the 90-min cap.  The original
    "needs manual A/B" hedge anticipated worst-case 2-3× *across
    every iteration*, but the gate's per-accept gating makes the
    worst case much smaller.

  The A/B escape hatch is preserved as the ``confirm_accepts``
  ``workflow_dispatch`` input — operators can still flip it off
  for one or two nights and compare confirm-reject rates against
  the historical screening-only ledger (the change is
  bidirectionally reversible without a code edit).

* **Backwards compatibility** — strictly safe at the script level:

  * ``LoopConfig.confirm_accepts`` defaults to ``False`` and the
    ``--confirm-accepts`` CLI flag is opt-in (set_defaults
    ``False``); the change is purely on the workflow side that
    *invokes* the script.
  * Pre-§6.4 ledger entries (no ``confirmed`` field) replay
    correctly through ``aggregate_codify_candidates`` —
    ``LoopIterationRecord.confirmed`` defaults to ``None`` so the
    codify-scan ``--confirmed-only`` filter naturally excludes
    legacy entries without crashing.
  * The bandit's ``_proposal_rule_key`` is independent of whether
    the rule fired on a screening-only iteration or a screen +
    confirm iteration, so ``prime_from_ledger`` (with or without
    ``--prime-include-archives``) mixes pre-flip and post-flip
    evidence onto the same arm posteriors without a re-key step.
  * Symmetric: an operator who flips the ``confirm_accepts``
    workflow_dispatch input to ``false`` and re-runs the cron
    immediately gets a pre-flip iteration, no warmup state needed.
  * The ``Commit ledger + summary`` step's commit-message uses
    ``ITER`` / ``MODE`` only; no message change needed (the gate
    state is visible in the ledger itself via
    ``LoopIterationRecord.confirmed`` and ``LoopConfirmRecord``
    entries).
  * No ledger-archive marker needed — same reasoning as the
    2026-06-21 ``--registry loop`` flip (the bandit's per-arm key
    is stable under the change, so existing posterior state
    composes seamlessly with post-flip iterations).

* **Tests** — none added.  The change is a workflow-file edit; the
  ``--confirm-accepts`` CLI flag and ``LoopConfig.confirm_accepts``
  behaviour are exercised by the existing 2026-06-14 ship's test
  suite (``TestConfirmAccepts*`` in ``tests/test_self_improve.py``)
  which continues to pass byte-identically.  Local sanity check:
  ``CONFIRM_ACCEPTS={true,false}`` bash-array expansion of the
  inlined command produces the expected argv with and without
  ``--confirm-accepts``.

* **Documentation updated**

  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry; the *Flip the
    nightly cron to ``--confirm-accepts``* idea promoted from
    queued-with-manual-A/B-hedge to shipped (the A/B escape hatch
    is preserved as the ``workflow_dispatch`` toggle so the hedge
    semantics are still available without a code edit).
  - ``planning/SELF_IMPROVEMENT_LOOP.md``: §2.2 V1 symptom block
    extended with the 2026-06-27 update noting the cron flip;
    §9.5 step 5 paragraph rewritten so ``--confirm-accepts``
    moves from the "remaining toggle" list to the active flags
    list, and the only outstanding lever becomes ``--metric aocc``
    (step 2).

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * **A/B audit on the first confirm-gate nightly** — read the
    first 2-3 nightly summaries after this flip and verify
    (a) the guard-rollback rate dropped (confirm-rejects now
    catch screening noise spikes before record), (b) the
    confirm-reject rate sits in the expected 50-70 % range (the
    rate the §2.2 evidence implies), and (c) the
    accept→codify-candidate funnel produces at least one
    ``confirmed=True`` candidate on the live ledger so the
    codify-scan ``--confirmed-only`` filter starts surfacing
    evidence.  Speculative on the exact rates — the §2.2 V1
    counts predict ~95 % rollback, but the V2 substrate (loop
    registry, graded reward, structural-per-class arms) may have
    already reduced screening noise.
  * **Halve iteration count if confirm-reject rate is low** — if
    the audit shows confirm-rejects are rare (< 10 % of screening
    accepts), the per-night cost saving from halving never
    materialises and the budget stays at 20.  But if confirm
    activates frequently and the wall-clock rises uncomfortably
    close to the 90-min cap, halving to 10 iterations preserves
    quality at half the cost (the bandit's per-arm posterior is
    cumulative across nights via ``--prime-include-archives``).
    Speculative until the first measurement night.

### 2026-06-26 — Codify auto-tuned `Nearby.radius` catalog tightening (manual widening-detector codify)

* **What** — Tightens the :func:`panobbgo.self_improve.default_catalog`
  ``Nearby.radius`` :class:`MutationRule` ``bounds`` from
  ``(0.005, 0.5)`` to ``(0.032, 0.313)`` based on cross-night ledger
  evidence surfaced by the 2026-06-19 widening detector and sized by
  the 2026-06-22 auto-tune.  Pure catalog-bound update — no new arms,
  no constructor changes, no behaviour change for the heuristic itself.
  Codifies the manual companion to the auto-tuned widening
  ``codify-scan --widen-bounds --widen-auto-tune`` proposal — the same
  shape as the 2026-05-31 ``Sobol.scramble=False`` codify (also a
  ledger-evidence-driven default change) but on a continuous numeric
  slot instead of a categorical.

* **Why** — Two direct effects:

  * **Bandit productivity** — the live ledger today carries 13 accepts
    on the ``(Nearby, radius, log_uniform_perturb)`` arm across 9
    distinct nights, with every accepted ``new_value`` falling inside
    the observed window ``[0.073, 0.135]``.  The pre-tightening
    catalog bounds ``[0.005, 0.5]`` admit values 6.25× below and 1.6×
    above that window — a sizable fraction of each ``log_uniform_perturb``
    pull lands outside the productive range and the rule was the
    catalog's dominant no-op generator.  The auto-tuned widening
    detector (2026-06-22) recommends a ~2.31× headroom factor around
    the observed range, i.e. proposed catalog
    ``[0.0317, 0.3130]`` ≈ ``[0.032, 0.313]`` (rounded for readability
    in the literal).  Tightening focuses every per-iteration pull onto
    the productive region without removing exploration headroom on
    either side.
  * **Per-arm posterior resolution** — concentrating proposals into the
    productive window means each accept / reject feedback updates the
    arm's Beta posterior on a value drawn from a meaningfully narrower
    distribution.  Same nightly budget → tighter per-arm posterior →
    the §11.1 "resolution" criterion improves on the same compute.

* **Backwards compatibility** — strictly safe:

  * The :class:`~panobbgo.heuristics.nearby.Nearby` constructor default
    (``radius=0.01``) is unchanged.  The catalog rule only fires when a
    spec sets ``radius`` explicitly (via :func:`_find_targets`), and the
    four registry seed specs that ship explicit
    ``radius={0.1, 0.05}`` are well inside the new bounds.
  * Bandit arm key ``(Nearby, radius, log_uniform_perturb)`` is
    unchanged, so the pre-tightening Beta posterior accumulates
    seamlessly across the bound change — the bandit only sees a
    narrower proposal distribution.  Ledger replay through
    :meth:`AdaptiveMutationSampler.prime_from_ledger` continues to map
    every historical proposal onto the same arm.
  * Every observed ``new_value`` from the live ledger (range
    ``[0.073, 0.135]``) sits comfortably inside the new bounds, so
    the bandit's accepted-region knowledge survives the change.
  * The widening detector's own ``current_bounds`` cache (looked up
    via :func:`_catalog_numeric_bounds`) is the single source of
    truth — re-running ``codify-scan --widen-bounds --widen-auto-tune``
    against the same ledger after this codify shows the auto-tuned
    proposal converges on ``[0.0345, 0.287]`` (essentially the new
    catalog bounds, modulo the ~2.12 factor the now-narrower catalog
    yields for the same observed spread).  The detector is
    self-stabilising.

* **Tests** — two existing tests updated (assertion-only):

  * ``TestCatalogNumericBounds.test_finds_existing_rule`` —
    expected value flipped from ``(0.005, 0.5)`` to ``(0.032, 0.313)``
    with a comment pointing back to this dated entry.
  * ``TestDetectWideningCandidates.test_looks_up_current_bounds_from_default_catalog`` —
    same flip; the test exercises the
    :func:`_catalog_numeric_bounds` lookup path the widening detector
    uses to size its proposal.
  * ``TestDetectWideningCandidatesAutoTune.test_auto_tune_sizes_factor_per_candidate``
    (and the JSON-mode CLI sibling
    ``TestCodifyScanCLIAutoTuneWidening.test_auto_tune_json_mode_emits_per_candidate_factor``)
    relaxed from ``2.2 < factor < 2.5`` to ``2.0 < factor < 2.3`` —
    the observed-spread ratio is now ~0.27 (vs ~0.13 under the prior
    wider bounds) so the auto-tuned factor sits near 2.12 rather than
    2.31.  Both still verify the spirit ("factor sits in the upper
    half of the [1.1, 2.5] range").
  * ``TestDetectWideningCandidatesAutoTune.test_auto_tune_with_custom_range``
    (and the JSON-mode CLI sibling
    ``TestCodifyScanCLIAutoTuneWidening.test_auto_tune_custom_range_propagates``)
    relaxed from ``> 3.5`` to ``> 3.0`` for the custom ``[1.2, 4.0]``
    range — same reason as above: tighter catalog → larger
    spread ratio → smaller per-candidate factor for the same
    observed range.

  All 1681 pre-existing tests (incl. the four updated assertions) pass.
  ``ruff check`` / ``ruff format`` / ``pyright`` are clean on every
  touched file.

* **Documentation updated**

  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry; the *Codify
    widening-detector proposals as a catalog bound update* idea
    promoted from speculative-because-no-driver-yet to shipped
    (manual codify, no driver needed — the §9.5 step 4 ``--open-pr``
    driver remains queued for the automation).
  - ``planning/SELF_IMPROVEMENT_LOOP.md``: §9.3 widening-detector
    paragraph extended with a note that the live ``Nearby.radius``
    candidate has been manually codified; the proposed
    ``Sobol.n`` candidate is left for a future iteration because the
    auto-tune classifies it as ``"widens current"`` (mixed signal,
    less clear than ``Nearby.radius``'s clean tightening).
  - ``panobbgo/self_improve.py``: in-rule comment in
    :func:`default_catalog` cites this dated entry and the ledger
    evidence count.
  - ``tests/test_self_improve.py``: two new-bounds assertions, two
    relaxed assertions; comments cite this entry for the rationale.

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * **Codify-scan `--widen-bounds --open-pr` driver** — the queued
    automation layer that turns a :class:`WideningCandidate` into a
    draft codify PR.  Speculative until the basic
    ``codify-scan --open-pr`` driver lands.  Sketch in the existing
    *Mutation-bound widening rule for bidirectional codify
    candidates* follow-up.
  * **``Sobol.n`` widening codify (manual companion)** — the
    auto-tune output today classifies the ``Sobol.n`` bidirectional
    candidate as ``"widens current"`` (proposed ``[3, 52]`` vs
    current ``[4, 64]``: expands the lower bound from 4 to 3 but
    contracts the upper from 64 to 52).  Less clear-cut than the
    ``Nearby.radius`` tightening; defer until either (a) more
    nights of evidence cluster the observed range more tightly or
    (b) the ``--open-pr`` driver decides what to do with the mixed
    signal.

### 2026-06-25 — Auto-tune κ for hierarchical structural bandit (V2 follow-up)

* **What** — Closes the *Auto-tune ``κ``* follow-up seeded under the
  2026-06-01 hierarchical-borrow ship.  Pure additions to
  :class:`panobbgo.self_improve.AdaptiveMutationSampler` and
  :class:`panobbgo.self_improve.LoopConfig` plus one CLI flag on
  ``scripts/self_improve.py run``:

  * :attr:`AdaptiveMutationSampler.structural_borrow_horizon` —
    new ``float ≥ 0`` constructor kwarg.  When ``> 0`` (and the two
    borrow preconditions are met: ``structural_borrow_alpha > 0``
    and ``per_class_structural = True``), the per-class arm's
    effective borrow is annealed by::

        κ_eff = κ / (1 + n_class_attempts / h)

    Cold arm (``n_class_attempts = 0``) borrows the full configured
    ``κ`` — same as the fixed-``κ`` path.  At
    ``n_class_attempts = h`` the borrow halves exactly.  Saturated
    arm (``n_class_attempts >> h``) effectively trusts its own
    per-class posterior.
  * New helper :meth:`AdaptiveMutationSampler._effective_borrow`
    centralises the annealing math so the same rule is consulted
    from the sample-path code and from tests.
  * :attr:`LoopConfig.structural_borrow_horizon` mirrors the
    constructor kwarg with the same default (``0.0`` → disabled).
    Validated in ``__post_init__`` (raises ``ValueError`` on
    negative).
  * CLI surface on ``scripts/self_improve.py run``:
    ``--structural-borrow-horizon`` (default ``0.0``).  Off by
    default so the nightly cron's behaviour is byte-identical
    until the flag is explicitly set.

* **Why** — The 2026-06-01 hierarchical-borrow ship closed the
  cold-start gap for per-class structural arms (a fresh candidate
  class no longer starts at the symmetric ``Beta(1, 1)`` prior; it
  warms with the op's empirical accept rate).  But the borrow
  *never lets go*: an arm with hundreds of attempts of its own
  still pays the ``κ ·`` op-aggregate cost, indefinitely pulling
  its leaf posterior toward the op-level mean.  Empirically the
  right behaviour is "borrow heavily early, vanish as evidence
  grows" — every hierarchical-bandit textbook puts this knob
  behind an annealing rule.

  Three direct effects:

  * **Convergence** — once an arm has accumulated enough evidence,
    the leaf posterior dominates.  An arm that the op-aggregate
    *underestimates* (say a niche heuristic that wins on a narrow
    subset of problems) stops being held down by sibling failures.
  * **Stability** — when sibling arms' posteriors are noisy (the
    op-aggregate jitters around the truth), the annealing reduces
    the noise contagion across siblings as each settles.
  * **No regression** — the ``h = 0`` default keeps the
    2026-06-01 fixed-``κ`` behaviour byte-identical.  Existing
    ledgers replay correctly; the bandit's stored arm keys are
    unchanged.

  Recommended values for an unattended cron: ``h = 5`` to ``10``.
  The per-arm posteriors warm up over a couple of nights at the
  catalog's typical per-iteration cardinality.

* **Backwards compatibility** — strictly safe:

  * Default ``structural_borrow_horizon = 0.0`` on both
    :class:`AdaptiveMutationSampler` and :class:`LoopConfig`
    keeps every existing invocation byte-identical.
  * The :meth:`_effective_borrow` helper returns the configured
    :attr:`structural_borrow_alpha` unchanged whenever
    annealing is off, when ``κ = 0`` (no borrow to anneal), or
    when the arm has no attempts (cold-start case).
  * The constructor's input validation matches the
    :attr:`structural_borrow_alpha` validation shape (raises on
    negative / non-finite).
  * Ledger replay: the bandit's arm key
    ``(class_name, op, "structural")`` is unchanged, so existing
    archives replay onto the same per-class arms regardless of
    whether the consumer enables the annealing knob.

* **Tests** — 16 new tests in the new
  ``tests/test_self_improve.py::TestStructuralBorrowAnneal`` class:

  * ``test_default_horizon_is_zero`` — default constructor.
  * ``test_negative_horizon_raises`` /
    ``test_non_finite_horizon_raises`` — validation paths.
  * ``test_effective_borrow_horizon_zero_returns_kappa`` —
    ``h = 0`` disables annealing.
  * ``test_effective_borrow_kappa_zero_returns_zero`` — ``κ = 0``
    means no borrow at all.
  * ``test_effective_borrow_cold_arm_returns_full_kappa`` —
    cold-start path.
  * ``test_effective_borrow_halved_at_horizon`` —
    ``n_class_attempts == h`` halves exactly.
  * ``test_effective_borrow_vanishes_at_saturation`` —
    ``n_class_attempts = 10_000`` shrinks to ``< 0.01``.
  * ``test_effective_borrow_monotonic_decreasing`` — strictly
    decreasing in attempts.
  * ``test_horizon_zero_byte_identical_to_no_annealing`` —
    backwards-compat sampling trajectory.
  * ``test_annealed_borrow_reduces_with_evidence`` — verifies the
    cold sibling sees the full borrow even with annealing on.
  * ``test_annealed_borrow_saturated_arm_drops`` — three-class
    catalog with ``y_attempts = 20, h = 5`` produces
    ``κ_eff = 0.2``; the rationale carries the exact
    ``Beta(6.6, 17.4)`` parameters.
  * ``test_annealed_borrow_inert_without_per_class`` — the knob
    requires per-class arms (same precondition chain).
  * LoopConfig integration: default, validation, propagation
    through :class:`SelfImprover`.

  All 16 new tests pass.  Full project test suite (1697 tests)
  green; ``uv run ruff check`` / ``uv run ruff format --check`` /
  ``uv run pyright panobbgo/self_improve.py scripts/self_improve.py``
  all clean.

* **Documentation updated**
  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry; the
    *Auto-tune ``κ``* follow-up promoted from *Next iteration
    ideas* to shipped.
  - ``doc/source/guide_benchmarking.rst``: new "Auto-tune ``κ``
    from observed evidence (``structural_borrow_horizon``)"
    sub-section documenting the annealing rule, the recommended
    horizon range, and the inert-preconditions matrix.
  - ``panobbgo/self_improve.py``: ``AdaptiveMutationSampler``
    docstring extended with the new ``structural_borrow_horizon``
    parameter; ``LoopConfig`` docstring similarly extended;
    ``_effective_borrow`` carries the full annealing-rule
    explanation.
  - ``TODO.md``: new "Recent Improvements" entry below.

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * **Hierarchical kwarg arms with annealing** — the same
    ``κ / (1 + n / h)`` rule could be applied to a hypothetical
    hierarchical kwarg-arm posterior that borrows across rules
    sharing the same heuristic class.  Lower priority than the
    structural version because kwarg arms already have
    literature-canonical centres (so cold-start is less painful),
    but the annealing knob would naturally apply once that
    hierarchy lands.
  * **Categorical horizon regimes** —
    ``structural_borrow_horizon ∈ {0, 5, 10, 25}`` as a meta-
    bandit choice on the loop driver itself (mirrors the
    *Categorical ``κ`` regimes* idea below).  Lets the loop tune
    its own annealing horizon from ledger evidence — a true
    second-order self-improvement.

### 2026-06-24 — Named LBC regimes for `NLSHADE_LBC.lbc_regime` (composite categorical arm)

* **What** — Closes the *Categorical LBC regimes* follow-up seeded
  under the *NL-SHADE-LBC follow-ups* backlog entry (2026-05-28
  ship).  Introduces a single composite ``lbc_regime`` constructor
  kwarg on
  :class:`panobbgo.heuristics.nl_shade_lbc.NLSHADE_LBC` that wraps
  the five LBC Lehmer-mean schedule fields
  (``p_F_init`` / ``p_F_final`` / ``p_CR_init`` / ``p_CR_final`` /
  ``m_lbc``) under four literature-motivated named regimes:

  * ``"cec2022"`` — Stanovov, Akhmedova & Semenkin 2022 defaults
    (``3.5, 1.5, 1.0, 1.5, 1.5``); byte-identical to opting into
    the constructor defaults.  The CEC-2022 winning configuration.
  * ``"lshade"`` — recovers the standard L-SHADE / jSO / NL-SHADE-RSP
    weighted Lehmer mean (``p = 2, m = 1`` for both F and CR, with
    no LBC schedule).  Useful as a degenerate baseline arm — turns
    the LBC mechanism itself off without dropping the heuristic so
    the bandit can A/B "LBC on" vs "LBC off" on the same population
    seed.
  * ``"flat"`` — pure arithmetic mean (``p = 1`` throughout, default
    spread ``m_lbc = 1.5``).  Drops all bias toward larger successful
    F / CR values; the success-history memory tracks the centre of
    mass of recent successes.
  * ``"aggressive"`` — strong bias throughout the run (``p_F`` decays
    ``5 → 3``, ``p_CR`` grows ``3 → 5``, default spread).
    Counterpart to ``"flat"`` — sharper concentration on the largest
    successes than the literature default.

  Each regime is stored as a ``(p_F_init, p_F_final, p_CR_init,
  p_CR_final, m_lbc)`` 5-tuple in the new module-level dict
  :data:`panobbgo.heuristics.nl_shade_lbc._LBC_REGIMES`.  The
  :func:`panobbgo.heuristics.nl_shade_lbc._normalize_lbc_regime`
  helper validates the constructor input and raises
  :class:`ValueError` for unknown strings or non-string values.

  Constructor semantics — the regime is **mutually exclusive** with
  the five per-field LBC float kwargs (uses a sentinel ``_UNSET``
  default on the five floats to detect explicit override; passing
  both raises ``ValueError`` rather than silently overriding).
  Existing call sites that set per-field kwargs explicitly continue
  to work; new specs that opt into the regime get the named preset.

  Catalog — :func:`panobbgo.self_improve.default_catalog` gains a
  new ``categorical_choice`` :class:`MutationRule` on the
  ``(NLSHADE_LBC, lbc_regime)`` slot with
  ``choices=("cec2022", "lshade", "flat", "aggressive")``.  The
  five previous per-field LBC ``float_uniform`` rules — shipped
  2026-05-28 — are **retired** in the same change.  Net catalog
  change: −5 rules + 1 = **−4** kwarg rules; the loop registry's
  catalog reach drops from ``44 / 44`` to ``40 / 40`` arms (same
  100% activation rate; the count is just lower because the five
  dormant per-field rules disappear).

  Loop registry —
  :func:`panobbgo.harness._make_loop_strategies`'s
  ``Loop_DE_Family`` spec now passes
  ``NLSHADE_LBC(NP_init=15, H=5, lbc_regime="cec2022")`` instead
  of the five explicit per-field LBC kwargs.  Behaviour is
  byte-identical (``"cec2022"`` resolves to the same defaults) but
  the new categorical arm fires on the seed registry the next
  nightly run.

* **Why** — Three direct effects:

  * **Catalog cardinality reduction** — five independent
    cold-started float arms replaced by one well-curated discrete
    arm.  Strict simplification: the five per-field rules were
    ``float_uniform`` over bounded ranges with no joint coupling;
    the bandit had to find a *correlated* set of optimal values
    across all five.  The composite arm collapses that
    five-dimensional search into four literature-tested points,
    each a coherent joint configuration.  Mirrors the §7.3 freeze
    spirit — the freeze permits broadening / consolidation of
    existing rule slots; this entry is a strict-strict
    consolidation, *removing* arms net of the new composite.
  * **Algorithmic reach** — the four named regimes span the
    qualitatively distinct LBC operating points.  ``"cec2022"`` is
    one well-tested point; the *shape* of the bias schedule (does
    F bias decrease, stay flat, or stay sharp?  same for CR?)
    varies meaningfully across DE applications.  ``"flat"`` and
    ``"aggressive"`` extend the Stanovov regime to two adjacent
    operating points whose geometry is differentiated in both the
    *F* axis (initial vs final exponent magnitude) and the *CR*
    axis (does CR bias grow or stay flat?).  ``"lshade"`` is the
    "LBC off" degenerate that lets the bandit test whether the LBC
    mechanism itself helps on the current battery — without
    dropping the heuristic.  Pattern-matches the 2026-06-23
    F_schedule broadening on L-SHADE.
  * **Persistence proximity** — the ``--open-pr`` codify driver
    (still queued under *Next iteration ideas*) reads the
    catalog's ``choices`` vocabulary directly.  A future ledger
    pattern where ``"flat"`` or ``"aggressive"`` consistently
    beats ``"cec2022"`` on the loop registry's DE family spec
    would surface as a regular kwarg-default codify candidate via
    :func:`aggregate_codify_candidates` — exactly the same path
    that surfaced ``Sobol.scramble=False`` in 2026-05-31.  The
    composite arm makes the regime *codify-able* as a single
    well-defined preset name.

* **Backwards compatibility** — strictly safe on every existing
  call path:

  * The default ``lbc_regime=None`` is unchanged.  Every existing
    spec — including the prior ``Loop_DE_Family`` spec that
    explicitly set the five LBC fields — would behave identically;
    the new ``Loop_DE_Family`` spec is byte-identical at the
    constructor-output level because ``"cec2022"`` maps to the same
    five defaults.
  * Existing call sites that pass per-field LBC kwargs continue to
    work (``lbc_regime`` stays ``None``); the constructor still
    validates the per-field values and reports the same
    ``ValueError`` messages on invalid input.
  * Ledger replay: the prior five per-field rules emitted
    ``new_value`` of floats.  Records replayed against the live
    catalog now skip those (the rules are retired) — but the
    bandit's :func:`panobbgo.self_improve._proposal_rule_key`
    keying ignores the value (only ``class_name / param_name /
    rule_kind`` matter), so any Beta posterior entry on the five
    retired ``(NLSHADE_LBC, p_F_init/...)`` arms cleanly drops
    out of the active prior on resume.  Existing ledgers stay
    parseable; the bandit just no longer pulls those arms.
  * Documentation: the canonical CEC-2022 module-level constants
    (``_DEFAULT_P_F_INIT`` etc.) stay as module-level floats and
    are referenced from the ``"cec2022"`` regime tuple, so any
    external introspection code keeps working.

* **Tests** — 13 new tests in
  ``tests/test_heuristic_nl_shade_lbc.py``:

  * ``test_default_regime_is_none`` — the bare constructor stores
    ``lbc_regime = None`` (no regime applied; per-field defaults
    take effect).
  * ``test_regime_cec2022_matches_defaults`` — the ``"cec2022"``
    regime is bit-identical to the per-field defaults
    (back-compat invariant).
  * ``test_regime_lshade_recovers_standard_lehmer`` — the
    ``"lshade"`` regime sets ``p = 2, m = 1`` for both F and CR.
  * ``test_regime_flat_is_constant_arithmetic`` — the ``"flat"``
    regime sets ``p = 1`` throughout (pure arithmetic mean).
  * ``test_regime_aggressive_is_high_biased`` — the ``"aggressive"``
    regime sets the largest exponents (``F: 5 → 3``, ``CR: 3 → 5``).
  * ``test_regime_dict_has_expected_keys`` — sanity-checks
    :data:`_LBC_REGIMES` membership and tuple well-formedness.
  * ``test_invalid_regime_string_raises`` — unknown / empty strings
    raise ``ValueError``.
  * ``test_invalid_regime_type_raises`` — non-string non-None
    inputs (ints, bools) raise ``ValueError``.
  * ``test_regime_with_explicit_kwargs_raises`` — passing
    ``lbc_regime`` together with any per-field LBC float raises
    ``ValueError`` ("mutually exclusive") — for each of the five
    fields independently.
  * ``test_regime_with_unrelated_kwargs_ok`` — regime composes
    cleanly with non-LBC kwargs (``NP_init`` / ``H``).
  * ``test_normalize_helper_collapses_none`` — direct test of the
    ``_normalize_lbc_regime`` helper.
  * ``test_lshade_regime_memory_update_matches_standard_lehmer``
    — analytic equivalence: ``lbc_regime="lshade"`` reproduces
    the standard ``s^2 / s^1`` weighted Lehmer mean
    bit-identically on a hand-rolled test vector.
  * ``test_kwarg_catalog_lbc_regime_is_categorical`` — verifies
    the new ``categorical_choice`` rule lives on the
    ``(NLSHADE_LBC, lbc_regime)`` slot with the four named
    regime choices.

  Existing ``test_kwarg_catalog_has_lbc_dials`` updated to assert
  the consolidated rule set (``NP_init`` + ``lbc_regime``) and
  explicitly forbid the retired five per-field rules from
  reappearing without a deliberate evidence-driven PR.  All 1694
  pre-existing tests in the project test suite pass unchanged.

* **Documentation updated**
  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry; the
    *Categorical LBC regimes* idea promoted from *Next iteration
    ideas* to shipped.
  - ``doc/source/heuristics.rst``: the ``NLSHADE_LBC`` bullet
    now mentions the named regimes.
  - ``doc/source/guide_benchmarking.rst``: the L-SHADE-derived
    family description names the regime presets; the categorical
    rule list adds the ``NLSHADE_LBC.lbc_regime`` entry (count
    nine → ten); the categorical-knobs bullet gains the LBC
    regime entry; the ``Loop_DE_Family`` description names the
    regime arm.
  - ``AGENTS.md``: categorical-rules list adds
    ``NLSHADE_LBC.lbc_regime`` (count nine → ten).
  - ``panobbgo/heuristics/nl_shade_lbc.py``: module docstring +
    constructor docstring updated for the regime dict and
    mutual-exclusion semantics.
  - ``panobbgo/self_improve.py``: ``default_catalog`` docstring
    + per-rule comment block updated for the consolidation.
  - ``panobbgo/harness.py``: ``_make_loop_strategies`` comment
    block updated for the per-heuristic rule count and the
    consolidation rationale.

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * **Per-CR / per-F sub-regime broadening** — the four named
    regimes intentionally share spread ``m_lbc`` across regimes
    (the literature default ``1.5`` except for ``"lshade"`` at
    ``1.0``).  A future broadening could expose a separate
    ``spread`` axis in a new categorical rule
    (``"narrow"`` / ``"default"`` / ``"wide"``) so the bandit can
    pick the spread independently of the bias regime.
    Speculative until ledger evidence shows the named regimes
    don't span the right space.
  * **``lbc_regime`` codify evidence** — once the cron has
    accumulated 2-3 nights of consistent regime preference (say
    ``"lshade"`` winning on the DE-family loop spec, indicating
    LBC mechanism itself is harmful at quick-mode budgets) the
    ``codify-scan`` step shipped 2026-06-17 will surface the
    slot as a candidate.  No code change needed — the queued
    ``--open-pr`` driver picks the candidate up automatically.
  * **Same broadening on the ``LSHADE``-base ``p_best_end``
    schedule** — the ``p_best_end`` kwarg currently ships as a
    ``float_uniform`` rule.  A future categorical broadening
    pattern: replace the bare-float choices with named regimes
    (``"fixed"`` / ``"half"`` / ``"quarter"``) so the catalog's
    expressive vocabulary stays uniform across DE schedule
    kwargs.  Speculative — the bare floats are already plenty
    discoverable.

### 2026-06-23 — Named regimes for `LSHADE.F_schedule` (categorical broadening)

* **What** — Closes the *Categorical regimes for ``LSHADE.F_schedule``
  (named cap regimes)* follow-up seeded under *Next iteration ideas*.
  Promotes the binary toggle shipped 2026-05-21 (``True`` / ``False``)
  into a four-way categorical over named cap regimes:

  * ``"off"`` — no cap (byte-identical Tanabe-Fukunaga 2014; replaces
    ``False``).
  * ``"jso"`` — Brest et al. 2017 §III-D: clamp ``F ≤ 0.7`` while
    ``progress < 0.6``, ``F ≤ 0.8`` while ``progress < 0.9``,
    unclamped in the final 10% (replaces ``True``).
  * ``"early"`` — earlier and tighter kick-in: clamp ``F ≤ 0.6``
    while ``progress < 0.4``, ``F ≤ 0.8`` while ``progress < 0.7``,
    unclamped after that.
  * ``"strict"`` — most aggressive: clamp ``F ≤ 0.5`` while
    ``progress < 0.5``, ``F ≤ 0.7`` while ``progress < 0.85``,
    unclamped in the final 15%.

  Each regime is stored as a ``(phase1_bound, phase2_bound,
  phase1_cap, phase2_cap)`` 4-tuple in the new module-level dict
  :data:`panobbgo.heuristics.lshade._F_SCHEDULE_REGIMES`.
  The :func:`panobbgo.heuristics.lshade._normalize_F_schedule` helper
  validates the constructor argument and maps the legacy bool inputs
  to the new strings (``True`` → ``"jso"``, ``False`` → ``"off"`` →
  ``None``) so ledger replay against the binary toggle and any spec
  that still passes the boolean form keep working.
  :meth:`panobbgo.heuristics.lshade.LSHADE._apply_F_cap` is
  rewritten to look up the per-regime tuple instead of branching on
  hard-coded module-level constants — the canonical Brest 2017
  constants (``_F_SCHEDULE_PHASE1_BOUND`` etc.) stay as aliases for
  the ``"jso"`` regime tuple for backwards-compat with any
  external introspection.
  The ``default_catalog`` rule for ``LSHADE.F_schedule`` flips its
  ``choices`` from ``(True, False)`` to
  ``("off", "jso", "early", "strict")`` so the Thompson bandit can
  search the broader cap geometry.  The bandit arm key
  ``(LSHADE, F_schedule, categorical_choice)`` is unchanged, so the
  pre-2026-06-23 ledger Beta posterior accumulates seamlessly across
  the regime broadening — only the proposed value vocabulary expands.

* **Why** — Three direct effects:

  * **Algorithmic reach** — the bandit now searches across
    qualitatively distinct asymmetric F-cap geometries rather than
    just toggling Brest 2017 on/off.  The literature canonical jSO
    cap (``0.6 / 0.9`` breakpoints, ``0.7 / 0.8`` caps) is one
    well-tested operating point; the *shape* of the cap (when it
    kicks in, how tight the early-phase cap is) varies meaningfully
    across DE applications.  ``"early"`` and ``"strict"`` extend the
    Brest 2017 regime to two adjacent operating points whose
    geometry is differentiated in both the *progress* axis (when the
    cap kicks in) and the *cap* axis (how tight it is) — see the
    per-regime tuples in :data:`_F_SCHEDULE_REGIMES`.  On
    ill-conditioned basins (Rosenbrock, DixonPrice) where large F
    can blow the population apart, ``"strict"`` is a natural fit; on
    smooth, well-conditioned landscapes where the population already
    converges quickly, ``"early"`` lets the cap kick in before the
    population shrinks under LPSR.
  * **Catalog activation discipline** — the §7.3 freeze policy
    permits broadening an existing rule's *choices vocabulary*
    when the underlying signal exhausts the current dimension (the
    binary toggle had been on the books since 2026-05-21 but
    nightly evidence so far hasn't shown either of ``True`` /
    ``False`` clearly dominating, suggesting the right answer lives
    *between* them).  This entry stays within the freeze — no new
    bandit arm, no new heuristic, just a wider value set on an
    existing arm.
  * **Persistence proximity** — the ``--open-pr`` codify driver
    (still queued under *Next iteration ideas*) reads the catalog's
    `choices` vocabulary directly.  A future ledger pattern where
    ``"early"`` consistently beats ``"jso"`` on the loop registry's
    DE family spec would surface as a regular kwarg-default codify
    candidate via :func:`aggregate_codify_candidates` — exactly the
    same path that surfaced ``Sobol.scramble=False`` in 2026-05-31.
    The categorical broadening unblocks codify evidence on a slot
    that previously had no expressive room for a non-trivial fix.

* **Backwards compatibility** — strictly safe:

  * The default ``F_schedule=None`` (cap disabled) is unchanged.
  * Existing call sites that pass the bool form still work:
    :class:`~panobbgo.heuristics.jso.JSO` now passes
    ``F_schedule="jso"`` (canonical) but the constructor still
    accepts ``True`` (normalized to ``"jso"``) so any user-facing
    subclass that re-passes the legacy form keeps working.
  * Ledger replay: the prior binary categorical rule emitted
    ``new_value`` of ``True`` / ``False``.  Both are accepted by
    the new constructor and normalize correctly; the bandit's
    :func:`panobbgo.self_improve._proposal_rule_key` ignores the
    value (only ``class_name / param_name / rule_kind`` matter), so
    every pre-broadening Beta posterior entry on
    ``(LSHADE, F_schedule, categorical_choice)`` is replayed onto
    the same arm under the broader vocabulary.
  * Documentation: the canonical Brest 2017 module-level constants
    (``_F_SCHEDULE_PHASE1_BOUND`` etc.) stay as aliases into the
    ``"jso"`` regime tuple, so any external introspection code that
    references them by name keeps working.

* **Tests** — 4 new tests in ``tests/test_heuristic_lshade.py``:

  * ``test_apply_F_cap_early_regime`` — exercises the three phases
    of the ``"early"`` regime against the regime tuple
    ``(0.4, 0.7, 0.6, 0.8)``.
  * ``test_apply_F_cap_strict_regime`` — exercises the three phases
    of the ``"strict"`` regime against the regime tuple
    ``(0.5, 0.85, 0.5, 0.7)``.
  * ``test_apply_F_cap_regime_dict_is_complete`` — sanity-checks
    the keys of :data:`_F_SCHEDULE_REGIMES` and the well-formedness
    of every regime tuple.
  * ``test_custom_F_schedule_construction_named_regimes`` —
    verifies every named regime survives construction with the
    canonical name *and* the explicit ``"off"`` collapses onto
    ``None``.

  The existing ``test_custom_F_schedule_construction`` test is
  renamed to ``test_custom_F_schedule_construction_bool_compat``
  and updated to assert the normalized form (``True`` → ``"jso"``,
  ``False`` → ``None``) — verifying the back-compat path that
  preserves ledger replay.  Existing ``_apply_F_cap`` test cases
  switched from ``F_schedule=True`` / ``False`` to the equivalent
  ``"jso"`` / ``"off"`` strings; behaviour is byte-identical.
  Catalog test ``test_default_catalog_has_categorical_rules``
  unchanged (asserts the slot exists, not the choices).  Tests in
  ``test_heuristic_jso.py``, ``test_heuristic_nl_shade_rsp.py``,
  ``test_heuristic_nl_shade_lbc.py`` that asserted
  ``h.F_schedule is True`` are updated to assert
  ``h.F_schedule == "jso"``.

* **Documentation updated**
  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry; the *Categorical
    regimes for ``LSHADE.F_schedule``* follow-up promoted from
    *Next iteration ideas* to shipped.
  - ``doc/source/guide_benchmarking.rst``: the ``F_schedule``
    categorical bullet, the example ``MutationRule`` literal, and
    the L-SHADE prose paragraph all rewritten for the four regimes
    plus the bool back-compat note.
  - ``panobbgo/heuristics/lshade.py``: module docstring + the
    ``F_schedule`` constructor docstring + the ``_apply_F_cap``
    docstring all updated for the regime dict.
  - ``panobbgo/heuristics/jso.py``: the ``F_schedule=True`` call
    site rewritten to ``F_schedule="jso"`` for clarity; the
    docstring reference updated.
  - ``panobbgo/heuristics/lshade_ep_sin.py``: docstring reference
    updated to mention named regimes.

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * **Tunable F-cap breakpoints / cap values on ``LSHADE.F_schedule``**
    — the open follow-up below is still relevant.  The
    *categorical-regimes* shipped here covers the discrete operating
    points; a future *continuous* refinement would expose the four
    cap-geometry parameters directly so the bandit could climb the
    cap surface.  Speculative until ledger evidence shows the named
    regimes don't span the right space.
  * **``F_schedule`` codify evidence** — once the cron has
    accumulated 2-3 nights of consistent regime preference (say
    ``"early"`` winning on the DE-family loop spec) the
    ``codify-scan`` step shipped 2026-06-17 will surface the slot
    as a candidate.  No code change needed — the queued
    ``--open-pr`` driver picks the candidate up automatically.
  * **Same broadening on ``LSHADE.archive_factor``** — the
    archive_factor categorical currently ships three discrete
    values ``(0.0, 1.0, 2.6)``.  A future categorical broadening
    pattern: replace the bare-float choices with named regimes
    (``"off"`` / ``"vanilla"`` / ``"rsp"``) so the catalog's
    expressive vocabulary stays uniform.  Speculative — the bare
    floats are already plenty discoverable.

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
### 2026-06-19 — Structural-op already-codified check (V2 §9.3 follow-up)

* **What** — Closes the *Structural-op codified check* idea seeded
  under *Next iteration ideas* on 2026-06-18.  Replaces the
  :func:`panobbgo.self_improve._structural_already_codified`
  placeholder (which always returned ``False``) with a real
  class-membership predicate, plus a new helper
  :func:`panobbgo.self_improve._live_class_membership` that walks the
  seed-spec factories to find which specs already carry the
  candidate's class in their ``heuristics`` / ``analyzers`` bucket.
  :func:`panobbgo.self_improve.annotate_codified_status` now branches
  on the candidate shape: structural candidates take the membership
  path; numeric / categorical kwarg candidates take the existing
  :func:`_live_kwarg_values` / :func:`_candidate_already_codified`
  path unchanged.

  Predicate rules — symmetric to the kwarg case's
  ``max(live) >= median(new_values)`` / ``min(live) <= median``
  shape:

  * ``add_heuristic`` of class ``X``: codified iff at least one seed
    spec already lists ``X`` under ``heuristics``.  The codify edit
    "append ``X`` to the seed pool" would be partially redundant —
    at least one spec already carries the heuristic.  Matches the
    existing kwarg rule: *suppress when the proposed change is
    already partially live*.
  * ``drop_heuristic`` of class ``X``: codified iff no seed spec
    lists ``X``.  The codify edit "remove ``X``" cannot remove
    anything that is not already there.
  * ``add_analyzer`` / ``drop_analyzer``: same shape, against the
    ``analyzers`` bucket.

  The CLI surfaces ``live_codified_values`` for structural
  candidates as the *spec names* that carry the class (instead of
  the kwarg-value list for kwarg candidates) so the
  ``--include-already-codified`` audit trail still tells the
  operator where the membership lives.  An unknown op (defensive —
  the catalog ships exactly the four ops above) classifies as
  *not* codified so the candidate continues to surface.

* **Why** — V2 §11 success criterion 2 ("≥ 3 codify PRs opened from
  ledger evidence; ≥ 2 merged" over the first 30 nights) is gated on
  the signal-to-noise of the daily codify-scan report.  The
  2026-06-18 ship already suppresses already-codified *kwarg*
  candidates; this ship closes the symmetric structural gap so the
  scanner's "already codified" predicate behaves consistently across
  rule kinds.  The live ledger doesn't yet carry structural codify
  candidates that exercise the gap (the V1 catalog freeze in §7.3
  means structural ops are rarely confirmed across multiple nights),
  but the predicate is in place for when the V2 catalog flow
  starts surfacing them — there's no signal-to-noise tax to wait
  out before the next operator session benefits.

* **Backwards compatibility** — strictly safe.  Pure additions to
  :mod:`panobbgo.self_improve`: one new private helper
  (:func:`_live_class_membership`), a real implementation for an
  existing helper (:func:`_structural_already_codified`), and a
  branch in :func:`annotate_codified_status` that routes structural
  candidates to the new path.  The dead structural branch in
  :func:`_candidate_already_codified` is replaced with a defensive
  ``return False`` (the function is now only reached for kwarg
  candidates; a mis-routed call would otherwise silently mis-classify).
  Every existing public API call site is unchanged; the live ledger's
  4-of-5 surfaced-candidate count is identical to before the ship.

* **Tests** — 8 new tests + 1 renamed test in
  ``tests/test_self_improve.py::TestAnnotateCodifiedStatus``
  (``test_structural_op_is_never_codified`` →
  ``test_structural_add_heuristic_not_codified_when_class_absent``)
  cover every structural op in both codified / not-codified
  directions, the heuristic-vs-analyzer bucket distinction (one
  class registered under analyzers should NOT codify a structural
  ``add_heuristic`` for the same class), the multi-spec membership
  recording, and the defensive unknown-op fallback.  Plus 2 new
  end-to-end CLI smoke tests in ``TestCodifyScanCLISuppression``
  (``test_structural_add_heuristic_suppressed_when_already_in_pool``,
  ``test_structural_drop_heuristic_surfaces_when_class_in_pool``)
  exercising the suppression behaviour against synthetic ledger
  records for ``Nearby`` (which the live quick + loop registries
  already ship under heuristics).

  Test totals: 421 in ``tests/test_self_improve.py`` (was 412 — 8
  added + 1 renamed).  ``uv run --extra dev ruff format --check .``
  / ``uv run --extra dev ruff check panobbgo/self_improve.py
  tests/test_self_improve.py`` / ``uv run pyright panobbgo`` /
  sphinx doctests all clean.

* **Impact** — minor signal-to-noise improvement on the daily
  routine.  Today's live ledger doesn't carry any structural
  candidates that clear the codify-scan gate, so the four-of-five
  surfaced-candidate count is byte-identical.  The win is forward-
  looking: once the V2 flow (``--confirm-accepts`` + ``--registry
  loop`` + ``--bandit-reward graded``) lands in the nightly workflow
  and the bandit accumulates confirmed structural-op accepts across
  multiple nights, this predicate will keep the operator's attention
  on actionable structural evidence rather than re-surfacing
  ``add_LBFGSB`` against a seed pool that already carries
  ``LBFGSB``.

* **Documentation updated**
  - ``planning/SELF_IMPROVEMENT_LOG.md``: this entry; "Structural-op
    codified check" follow-up promoted from queued to shipped.
  - ``planning/SELF_IMPROVEMENT_LOOP.md``: no direct edit (the
    candidate-set hygiene work is downstream of §9.3, not on the
    critical V2 path).
  - ``doc/source/guide.rst``: quick-nav entry mentions the new
    structural-op predicate alongside the 2026-06-18 kwarg
    suppression entry.
  - ``doc/source/guide_benchmarking.rst``: extended the existing
    "Cross-night codify-scan" subsection's predicate description so
    the structural rules are documented alongside the kwarg rules.
  - ``AGENTS.md``: self-improvement loop bullet annotated with the
    structural extension to the suppression predicate.
  - ``TODO.md``: new dated entry under "Recent Improvements" for the
    structural-op codified check.

* **Follow-up ideas** seeded under *Next iteration ideas*:

  * **Tolerance / hysteresis on the numeric predicate** —
    still queued (from the 2026-06-18 ship).
  * **Membership-vs-coverage rule** — the current rule suppresses
    when *at least one* spec carries the class (the symmetric
    "partially redundant" rule).  A *stricter* alternative would
    suppress only when *every* spec carries the class — closer to
    "the codify edit is a complete no-op everywhere".  Speculative
    until the loop produces structural codify candidates that
    differentiate the two rules.

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

#### Warm-started curvature-aware local polish for the curved-valley class (first layer shipped 2026-07-07)

**Status: the warm-started restart geometry shipped 2026-07-07** (see that
dated entry).  ``LBFGSB.warm_start`` polishes a perturbation of the best
incumbent on every restart after the first; the structural-catalog LBFGSB
candidate now ships ``warm_start=True``.  Measured warm 0.198 vs cold 0.156
(+0.042) at full budget on the curved-valley battery; a tie at the tighter
standard budget with consistently lower ``Rosenbrock_5D`` best-distance
(11.7 vs 15.8) and no regression anywhere.  The negative result the original
note warned about (cold LBFGSB bolted onto ``Rewarding_Diverse`` *regresses*)
was the direct motivation: warm restarts fix the wrong restart geometry.

**Still open — fully closing the ``Rosenbrock_5D`` gap:**

* ~**Curvature-aware quadratic warm step**~ — **shipped 2026-07-08** for the
  derivative-free ``Nearby`` refinement half of this note (see that dated
  entry): ``Nearby.quadratic`` fits a distance-weighted ridge quadratic to the
  recent evaluated points and emits its trust-region Newton minimiser, gated on
  a weighted-R² fit-quality check, at zero extra objective evaluations.
  Statistical-accept ACCEPT on the randomized battery (roughly doubles the
  composite there; +0.0274 Δ, 95% CI ``[+0.0057, +0.0521]`` on
  ``Rewarding_Diverse``).  **Still open** — the *5D valley* half: the current
  fit is ridge-regularised (cross terms shrunk toward zero) so it recovers
  per-axis curvature cheaply but not the strong coordinate *coupling* of a 5D
  Rosenbrock valley; a full quadratic needs ``O(d²)`` local points.  A
  **diagonal-plus-low-rank** Hessian model (capture the dominant valley
  direction with a rank-1 or rank-2 correction) may recover the coupling from
  far fewer samples than a full quadratic, and remains the recommended
  successor for the Rosenbrock_5D gap.
* **Wire ``warm_start`` into the ``Loop_LocalSearch`` seed** — the 2026-07-07
  ship only flipped the *structural-catalog* candidate (which the loop's
  bandit measures live).  The seed spec's own LBFGSB stays cold pending a
  clean A/B on the exact COBYQA+LBFGSB+NelderMead composition (the COBYQA
  subprocess made a fast local A/B impractical this iteration).  Let the
  nightly loop's ``add_heuristic``/codify path confirm it, or measure it
  directly once compute allows.
* **Dedicated warm-started local-search strategy (needs an ADR)** — a
  ``LocalSearch_LBFGSB`` (or ``StrategyPhased`` global→local) spec in the
  default battery would give it a strategy that *solves* smooth valleys, but
  shifts the historical composite baseline — the same ADR gate the older
  LBFGSB follow-ups carry.
* **Measurement discipline** — decide on a **≥3-seed** aggregate composite,
  never a single seed (the 2026-07-06 sweep caught a single-seed
  StyblinskiTang "win" that was pure noise).  Note: COBYQA/LBFGSB portfolio
  runs are subprocess-heavy (~15 s each), so a full standard-battery A/B of a
  COBYQA-bearing spec can exceed a 10-min budget — prefer a lean
  COBYQA-free portfolio or the curved-valley subset to isolate the effect.

#### Membership-vs-coverage rule for structural codify suppression (cosmetic; seeded 2026-07-06)

The structural already-codified predicate
(:func:`panobbgo.self_improve._structural_already_codified`) is
*global*: a ``drop_heuristic`` candidate is suppressed only when **no**
seed spec carries the class, and an ``add_heuristic`` candidate when
**at least one** does.  But the apply driver
(:func:`derive_codify_edits`) is *spec-scoped* — it edits only the
candidate's :attr:`~CodifyCandidate.strategy_names`.  The mismatch means
a codify that drops a class from its evidenced spec keeps re-surfacing
in the scan report if any *other* spec still carries it (observed
2026-07-06: the ``LatinHypercube`` drop from ``Loop_LocalSearch``
re-surfaces because ``Loop_Restart`` still seeds it).  Harmless — the
apply is idempotent and the 0-edit path returns before ``--open-pr`` so
no empty PR is opened — but the report misleads the daily routine.
**Fix**: make the structural suppression consult the candidate's
``strategy_names`` and suppress once *those* specs no longer carry the
class (drop) / already carry it (add) — mirroring the apply's narrowing
and the numeric ``_candidate_already_codified`` self-stability
invariant.  This supersedes the older, vaguer "membership-vs-coverage"
note in the 2026-06-18 suppression follow-ups.

#### Budget-adaptive `NP_init="auto"` follow-ups (after 2026-07-05 ship)

* **Per-heuristic budget-share for `"auto"` in portfolios** — the
  2026-07-05 ship wired `NP_init="auto"` into the structural catalog (where
  a structurally-added DE is typically the dominant point-generator, so
  the full `max_eval` estimate is right) but *not* into the
  `Loop_DE_Family` seed spec (a 6-way portfolio where each DE arm gets only
  a fraction of the budget, so `"auto"` still over-sizes).  Size `"auto"`
  from `max_eval · arm_share` when a DE arm shares a strategy with N other
  point-generators.  The Rewarding bandit already tracks per-heuristic
  pulls, so a cheap realised-share estimate is available; once it exists,
  wire `"auto"` into `Loop_DE_Family` and re-measure.
* **Curvature-aware local step (replaces the rejected pattern move)** —
  the 2026-07-05 momentum experiment showed a *straight* directional
  extrapolation in `Nearby` overshoots curved valleys.  The next thing to
  try for the Rosenbrock-class weakness is a **quadratic / trust-region
  local model** fit to the recent local best points (a mini-`QuadraticWLS`
  around the best), proposing the model's minimiser instead of an
  isotropic perturbation — a curvature-aware step rather than a
  first-order momentum one.

#### `codify-scan --apply-top` follow-ups (after 2026-06-30 ship)

The 2026-06-30 ship landed the *source-edit* layer of the
``--apply-top`` driver — every kwarg :class:`CodifyCandidate` can now
be applied to ``panobbgo/harness.py`` in one CLI command.  Three
follow-ups are natural next tickets:

* **Structural-edit primitive** — extend
  :func:`derive_codify_edits` to support ``add_/drop_heuristic`` /
  ``add_/drop_analyzer`` candidates by emitting a richer
  :class:`CodifyEdit` shape that targets a list literal insertion /
  removal rather than a single value substitution.  The AST can
  represent the change cleanly (``ast.List.elts`` slice + a
  re-formatting pass via :mod:`black` or :func:`ast.unparse`).
  Live-ledger motivation: the current top kwarg-or-structural
  candidate today is the ``LatinHypercube`` ``drop_heuristic`` from
  ``Loop_LocalSearch`` (n_nights=2, mean_Δ=+0.0491, min_record_ci_low
  +0.0352).  Once structural candidates start surfacing as the *top*
  actionable evidence repeatedly, the structural-edit primitive
  moves from speculative to motivated.
* ~**`--apply-top --auto-format` flag**~ — **shipped 2026-07-03**
  as the ``--apply-format`` flag on ``codify-scan --apply-top``
  (renamed for CLI parity with the sibling ``--apply-run-tests``
  flag).  Runs ``uv run ruff format`` on the modified files
  after the write.  See the 2026-07-03 dated entry above.
* ~**`--apply-top --run-tests` flag**~ — **shipped 2026-07-03**
  as the ``--apply-run-tests`` flag on ``codify-scan --apply-top``.
  Runs ``uv run pytest tests/test_self_improve.py`` after the
  (optional) format step; non-zero rc propagates.  See the
  2026-07-03 dated entry above.

#### `codify-scan --open-pr` driver (after 2026-06-17 / 2026-06-29 / 2026-06-30 ships)

The 2026-06-17 ship landed the *detection* half of V2 §9.3 (the
``codify-scan`` subcommand surfaces candidates as text / JSON), the
2026-06-29 ship landed the *value derivation* layer
(:meth:`CodifyCandidate.proposed_codify_value`), and the 2026-06-30
ship landed the *source-edit* layer
(:func:`derive_codify_edits` / :func:`apply_codify_candidate`).  The
remaining queued *publish* layer is the ``--open-pr`` flag that
opens a draft PR for each apply.  Sketch:

1. **Dedup pass** — ``gh pr list --state open --json title,headRefName``,
   parse each open PR for a known "codify ``Class.param``" marker
   either in the title or via a label, and skip any candidate whose
   :attr:`CodifyCandidate.slot_key` already has an open PR.  Matches
   the §12.3 step 0 lesson (the four duplicate NL-SHADE-RSP PRs
   #227–#230) — enforced in code rather than left to operator memory.
2. **Source-edit primitive** — *shipped 2026-06-30* as
   :func:`derive_codify_edits` + :func:`apply_codify_candidate`.  For
   numeric / categorical candidates the edit is on the seed-spec
   factory ``(ClassName, {"param": value, ...})`` literal across
   every registered factory function.  Per-site direction guard
   preserves deliberately-tighter sibling specs.  The ``--open-pr``
   driver consumes :func:`apply_codify_candidate` directly — the
   queued work is the PR-creation wrapper, not any new edit
   primitive.  For structural ops the source-edit primitive is
   still queued (the 2026-06-30 ship intentionally scopes to kwarg
   edits; structural list-entry insertion / removal is a future
   extension flagged under *Next iteration ideas*).
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
  lands.  *2026-06-26 update:* the ``Nearby.radius`` candidate has
  been manually codified (see the 2026-06-26 dated entry above) —
  this is the first widening-detector candidate to land as a
  catalog change.  The pattern (manual codify before automation) is
  the same shape as 2026-05-31's ``Sobol.scramble=False`` codify;
  the driver remains queued for the automation.
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
* **``Sobol.n`` widening codify (manual companion to 2026-06-26)** —
  the auto-tune output today classifies the ``Sobol.n`` bidirectional
  candidate as ``"widens current"`` (proposed ``[3, 52]`` vs current
  ``[4, 64]``: expands the lower bound from 4 to 3 but contracts the
  upper from 64 to 52).  Mixed signal — defer the manual codify until
  either (a) more nights of evidence cluster the observed range more
  tightly or (b) the ``--open-pr`` driver decides on a tie-breaking
  rule for ``"widens"`` candidates.  See the 2026-06-26 entry above
  for the manual-codify shape the queued driver should emulate.

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

* ~**Structural-op codified check**~ — **shipped 2026-06-19**.
  :func:`_structural_already_codified` now implements a real
  class-membership predicate against the seed pool's
  ``heuristics`` / ``analyzers`` buckets, symmetric to the kwarg
  rule's "at least one spec already meets the proposal"
  semantics.  See the 2026-06-19 entry above.
* **Tolerance / hysteresis on the numeric predicate** — the
  current ``max(live) >= median(new_values)`` rule is exact; a
  small relative tolerance (e.g. 5%) would let the predicate
  catch cases where the live default is *very close* to the
  median proposal without being strictly above / below.
  Speculative — the exact rule already catches the dominant
  ``Sobol.scramble`` shape.
* **Membership-vs-coverage rule for structural ops** — the
  2026-06-19 ship suppresses when *at least one* spec carries the
  class (the "partially redundant" semantic, symmetric to the
  numeric ``max(live) >= median`` rule).  A *stricter* alternative
  would suppress only when *every* spec carries the class — closer
  to "the codify edit is a complete no-op everywhere".
  Speculative until the loop produces structural codify candidates
  that differentiate the two rules.

#### Flip the nightly cron to `--confirm-accepts` — shipped 2026-06-27

Shipped 2026-06-27; see the 2026-06-27 dated entry above for the
workflow edit, the per-night compute audit that re-evaluated the
"2-3× per-iteration cost" hedge against the actual ~3.6 % accept
rate, and the ``confirm_accepts`` ``workflow_dispatch`` toggle that
preserves the A/B escape hatch without requiring a code edit.

Follow-ups still queued (graduated from the original sketch):

* **A/B audit on the first confirm-gate nightly** — speculative.
  Read the first 2-3 nightly summaries after the flip and verify
  (a) the guard-rollback rate dropped, (b) the confirm-reject rate
  sits in the expected range, and (c) the
  accept→codify-candidate funnel produces at least one
  ``confirmed=True`` candidate so the codify-scan
  ``--confirmed-only`` filter starts surfacing evidence.
* **Halve iteration count if wall-clock pressure appears** — only
  triggered if the confirm-gate's per-accept activations push the
  cron close to the 90-min cap.  Per the cost audit, the gate's
  worst-case per-night overhead is ~30-60 s (~0.7 accept events ×
  2 × 15 s), so halving is unlikely to be needed; revisit only if
  measurement shows otherwise.

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

* ~**Auto-tune ``κ``**~ — **shipped 2026-06-25** as
  :attr:`AdaptiveMutationSampler.structural_borrow_horizon`
  (``h ≥ 0``) plus the matching
  :attr:`LoopConfig.structural_borrow_horizon` field and
  ``--structural-borrow-horizon`` CLI flag.  When ``h > 0`` (and the
  two borrow preconditions are met) each per-class arm's effective
  borrow shrinks toward zero as its own attempts accumulate:
  ``κ_eff = κ / (1 + n_class_attempts / h)``.  Cold arms still
  borrow the full configured ``κ``; at ``n_class_attempts = h`` the
  borrow halves exactly; saturated arms effectively stop borrowing
  and trust the leaf posterior.  Default ``h = 0`` disables annealing
  (byte-identical to the 2026-06-01 fixed-``κ`` ship).  Recommended
  values for an unattended cron: ``h = 5`` to ``10`` (per-arm
  posteriors warm up over a couple of nights).  See the
  2026-06-25 dated entry above.
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

* **Categorical LBC regimes — shipped 2026-06-24** as the
  :data:`panobbgo.heuristics.nl_shade_lbc._LBC_REGIMES` dict +
  :func:`panobbgo.heuristics.nl_shade_lbc._normalize_lbc_regime`
  helper plus the ``lbc_regime`` constructor kwarg on
  :class:`NLSHADE_LBC` and the matching
  ``(NLSHADE_LBC, lbc_regime, categorical_choice)``
  :class:`MutationRule` on :func:`default_catalog`.  The four named
  regimes (``"cec2022"`` / ``"lshade"`` / ``"flat"`` /
  ``"aggressive"``) wrap the five LBC schedule fields under
  literature-motivated joint configurations.  The five previous
  per-field ``float_uniform`` rules (shipped 2026-05-28) are
  retired in the same change — net catalog cardinality reduction:
  five cold-started independent dial arms replaced by one
  well-curated composite arm.  Mutually exclusive with the
  per-field LBC kwargs (constructor raises ``ValueError`` if both
  are passed).  See the 2026-06-24 entry above.
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

#### Categorical regimes for `LSHADE.F_schedule` — shipped 2026-06-23

Shipped 2026-06-23 as the :data:`panobbgo.heuristics.lshade._F_SCHEDULE_REGIMES`
dict + :func:`panobbgo.heuristics.lshade._normalize_F_schedule` plus the
broadened ``default_catalog`` ``LSHADE.F_schedule`` rule.  The 2026-05-21
binary toggle (``True`` / ``False``) is promoted to a four-way
categorical (``"off"`` / ``"jso"`` / ``"early"`` / ``"strict"``) so the
bandit can search across qualitatively distinct cap geometries instead
of just toggling Brest 2017 on/off.  ``True`` / ``False`` continue to
work as backwards-compatible synonyms for ``"jso"`` / ``"off"``
(preserving ledger replay and any spec that still passes the boolean
form).  See the 2026-06-23 dated entry above.

#### Named LBC regimes for `NLSHADE_LBC.lbc_regime` — shipped 2026-06-24

Shipped 2026-06-24 as the
:data:`panobbgo.heuristics.nl_shade_lbc._LBC_REGIMES` dict +
:func:`panobbgo.heuristics.nl_shade_lbc._normalize_lbc_regime`
helper plus the ``lbc_regime`` constructor kwarg on
:class:`NLSHADE_LBC` and the matching ``categorical_choice``
:class:`MutationRule` on :func:`default_catalog`.  The four named
regimes (``"cec2022"`` / ``"lshade"`` / ``"flat"`` /
``"aggressive"``) wrap the five LBC schedule fields under
literature-motivated joint configurations; the five per-field LBC
``float_uniform`` rules previously on the catalog (shipped
2026-05-28) are retired in the same change.  Net catalog
cardinality reduction: −4 kwarg rules.  See the 2026-06-24 dated
entry above for the regime tuples, mutual-exclusion semantics, and
the loop-registry wiring update.

#### Tunable spread axis on `NLSHADE_LBC.lbc_regime`

The four named regimes shipped 2026-06-24 share spread ``m_lbc``
across regimes (``1.5`` everywhere except ``"lshade"`` at ``1.0`` —
the L-SHADE recovery point).  A future broadening could split the
*spread* axis into a separate categorical rule
(``"narrow"`` / ``"default"`` / ``"wide"``) so the bandit can pick
the spread independently of the bias regime.  Pairs with the
*Categorical-with-dependent-kwarg rule pattern* idea above
(``lbc_regime + spread_regime`` would be the second motivating slot
the dependent-kwarg pattern needs).  Speculative until ledger
evidence shows the four named regimes don't span the right joint
space.
