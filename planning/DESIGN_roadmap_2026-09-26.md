# Roadmap: beat the field on expensive parallel black-box problems (2026-09-26)

Harald's decisions of 2026-09-26, and the design they commit to.
`GOAL.md` §1/§2c point here; `TODO.md` carries the work items.

## 1. Battlefield

* **Primary: expensive, parallel evaluations at small budgets** —
  10…200·dim, q parallel workers, evaluations that take long, vary in
  duration, crash or time out.  This is what panobbgo's architecture
  (event bus, shared archive, async dispatch, dask, per-call timeout)
  was built for, and where the incumbents (Bayesian optimisation tools)
  are weakest: they scale poorly with q, assume every call returns, and
  are sequential at heart.
* **Floor: the cheap-evaluation regime stays measured** — MA-BBOB /
  BBOB anytime at 500…2000·dim, the crowded field (CMA-ES variants,
  L-SHADE lineage, NGOpt).  Every change reports both tracks; a win on
  the primary track that loses the floor is not a win.

## 2. The claim we want to be able to make

> Never much worse than the best single solver on any problem class,
> clearly better on average — on problems nobody tuned for, and with
> q > 1 workers.

That is a regret bound against the best arm plus a gain from sharing.
It is falsifiable only with the instrument of §3.

## 3. Instrument (step 1 — before any new mechanism)

**Status 2026-09-26:** done — cheap-track external baselines (pycma
IPOP/BIPOP, Nevergrad NGOpt, Optuna CMA/TPE; #344), the virtual-clock
simulator with `aocc_time` (#345), the failure-region families and BBOB
shapes (`family_screen.py preset=failure|shapes`, #346), dims 30/40 with a
`bbob-largescale`-style slice and the sealed test set (MA-BBOB + families,
`panobbgo/sealed.py`), the expensive-track baselines (BoTorch qLogEI,
TuRBO-1, SMAC3, Py-BOBYQA in the `baselines-bo` extra; HEBO and PDFO do not
install under numpy 2.5 / Python 3.14).  Open — the real-world set (and its
share of the sealed set),
feature logging (`TODO.md` §1).

1. **External baselines** as harness arms, batch-capable (ask/tell with q):
   * cheap track: pycma (IPOP/BIPOP), Nevergrad `NGOpt` (the direct rival —
     a hand-ruled selector over a portfolio), Optuna CMA;
   * expensive track: Nevergrad `NGOpt` with batches, Optuna TPE,
     BoTorch/Ax (qEI, TuRBO), SMAC3, HEBO, PDFO/BOBYQA as the local
     reference.
   Heavy dependencies (torch) go into an optional extra; runs happen on
   GitHub runners.
2. **Virtual-clock parallel simulator.**  A discrete-event simulation of
   q workers with a duration model (constant, log-normal, x-dependent),
   so async behaviour is measured deterministically without real waiting.
   Metric: AOCC over evaluations **and** over virtual time, at
   q ∈ {1, 4, 16, 64}.
3. **Suite extensions** (on top of `DESIGN_suite_2026-09-14.md`):
   * dims 30/40 (and a `bbob-largescale` slice);
   * **failure regions** as a family knob — half-space, ball, random
     boxes, share of volume; failure mode crash vs timeout (§5 D);
   * real-world set: CEC2020 real-world constrained, COCO
     `bbob-constrained` / `bbob-mixint`, ESA GTOP, HPO surrogates
     (YAHPO / HPOBench);
   * a **sealed test set** (fresh MA-BBOB seeds + part of the real-world
     set) never used for tuning or training, touched only for claims.
4. **Feature logging.**  At fixed checkpoints each run records landscape
   features and per-arm trajectory statistics — the training data for
   A and B.

## 4. Mechanisms

### A. Probe → select → unleash, in cycles

Not a one-shot pick on dimension and budget (that is what NGOpt does),
but a loop that keeps learning about the problem:

1. **Probe** with a small budget (space-filling points plus short runs
   of two or three cheap arms).
2. **Featurise**: ELA features (y-distribution, linear/quadratic
   meta-model fit, level sets, dispersion, nearest-better clustering),
   plus what only a run can see — each arm's progress rate, restarts,
   noise estimate from repeats, failure share, evaluation duration.
3. **Select** arm(s), configuration and the next budget chunk.
4. **Run** the chunk, re-featurise (now with trajectory data), keep /
   switch / split, repeat.

The selector is learned: gradient-boosted trees first (small tabular
data, interpretable, cheap), a neural net once the data warrants it.
Target: predicted final quality per arm given features so far.  Training
data comes from our own harness on the development suite; validation on
the sealed set; a fallback to the default portfolio when the model is
unsure.  Headroom measured so far: the per-cell oracle beats the
portfolio by 0.015…0.039 AOCC (§53).  The oracle gate (§45) is the seam
it plugs into.

#### A in detail: what the selector learns (Harald, 2026-09-26: xgboost first)

* **Label = counterfactual normalised regret per arm.**  At each
  checkpoint the run state is frozen and *every* arm continues from it
  to the end of the budget (deterministic runs make this a true branch).
  Arm i's label is its shortfall against the best arm on that instance,
  in rank or AOCC-difference units — relative, so d = 2 and d = 40,
  easy and hard instances share one scale.  xgboost regresses regret per
  arm; the pick is the argmin (cost-sensitive algorithm selection, as in
  ASlib).  Every cycle is a new checkpoint, so the same model serves B
  (keep / switch / split).
* **Context as features, one model**: dim, remaining evaluations / dim,
  q, noise level.  Generalisation across ranges is tested, not assumed.
* **Invariant features** — invariant where the arms are:
  - to f → a·f + b and monotone transforms (rank-based arms): Spearman
    fitness–distance correlation, nearest-better clustering ratios,
    dispersion of the top k % vs all;
  - affine in f: R² of linear / quadratic / additive meta-models;
  - to rotation of x: condition estimate from the fitted Hessian's
    eigenvalues, distance statistics;
  - deliberately *not* rotation-invariant: separability (additive vs
    full quadratic R²), because DE and coordinate methods depend on it;
  - to dimension: evaluations / d, distances / √d, progress rates × d;
  - in-run only: per-arm progress rate, step-size trend, restarts, noise
    estimate from repeats, failure share, evaluation durations.
  - **"stuck locally"** (Harald): an arm that makes no progress *and* no
    longer explores is unlikely to find anything new.  Measured relative to
    the region the arm works in, not the whole box: spread of its recent
    samples (geometric-mean std per axis / √d, and its trend — contracting
    or not), novelty (distance of new points to the nearest earlier ones,
    relative to that spread), revisit rate, the share of the box the region
    covers, and how much of the box is still unsampled (global dispersion).
    Low progress + contracting spread + low novelty = stuck; whether
    switching pays depends on what the rest of the box still promises
    (meta-model / global-structure features).  A stuck arm is the trigger
    for B's reallocation and for a hand-over or basin-leaving restart (C).
  Never raw f values or raw coordinates.
* **Invariance is tested**: besides the sealed set, leave-one-COCO-class-out
  and leave-dimension-out (train d ∈ {2, 5, 10}, test d ∈ {20, 40}).
  Feature importances show which feature leaks scale when it fails.
* **Target metric**: share of the single-best-solver → oracle gap closed
  (§53 ceiling: 0.015…0.039 AOCC).
* **Risks**: small-probe features are noisy — train at the probe sizes
  used at run time, bootstrap features to expose their variance; the
  branched labels cost arms × instances × checkpoints — deterministic,
  so it parallelises on GitHub runners; MA-BBOB mixtures and the
  extended families supply unlimited training instances.  If the
  problem set lacks diversity for this, extend it (Harald: "do it").

### B. Budget allocation by forecast

The bandit gave nothing (§31) because its credit was last-step
improvement — noisy and myopic.  Instead fit each arm's convergence
curve (log-linear rate, restart-aware) and give the next block to the
arm with the best *predicted value at the end of the budget* —
Hyperband-style reasoning over optimisers.  Worst case ≈ best arm plus
the exploration cost.  A and B share one model: "value of continuing
arm i, given features and its trajectory".

### C. Sharing as a ratchet

The warm hand-over (§48/§50) is the one measured positive of sharing;
it carries m and σ.  Next payloads: a surrogate passed between arms,
surrogate pre-selection (lq-CMA-ES: a cheap model screens candidates
before an expensive evaluation), and the hand-over into a warm-started
local polish (L-BFGS-B / BOBYQA).

### D. Failure regions ("poison")

Evaluations that crash or time out (NaN placeholders since #337) are
information: in real simulators they cluster — a region, often a
half-space along one variable (a solver diverges above some pressure,
a mesh breaks below some thickness).  The literature calls these hidden
or unknown constraints (Le Digabel & Wild's QRAK taxonomy; GP
classifiers for crash constraints in BO, Gelbart et al. 2014; the
extreme barrier in MADS).

Design:

* **One shared failure model** on the event bus: a classifier
  P(fail | x), trained on every result, separate for crash and timeout
  (a timeout may be about cost, not validity).  Models in order of
  cost: axis-aligned trees / forests (they find the half-space directly
  and stay interpretable), RBF / kNN density, GP classification at
  small budgets.
* **Conservative**: unexplored is not poisoned.  A zone counts only
  where failures are dense and the model is confident.
* **Integration reuses the constraint machinery**: P(fail) enters as a
  learned constraint through the constraint handler's `rank_key`, so
  CMA-ES / DE / PSO rank points in poisoned zones last without
  per-algorithm code; a strategy-level candidate filter additionally
  rejects / resamples candidates above a threshold before they are
  dispatched — which is where the saving is, because a failed expensive
  call costs its full duration.
* Measured on the failure-region families (§3.3); failed calls count as
  spent budget.

A duration model (predicted cost of x) is the natural sibling: with q
workers and uneven durations, scheduling cheap-and-promising points
first matters.

## 5. Order of work

1. Instrument §3 (engineering; parallelisable; no research decisions).
2. Measure where we stand: panobbgo vs the baselines, per COCO class,
   per budget, per q, both tracks.  GitHub runners suffice — only
   evaluation counts and virtual time matter.
3. Build D first if step 2 confirms the incumbents break on failures
   (cheap to build, directly on the primary track), then A+B (one
   model), then C's new payloads.
