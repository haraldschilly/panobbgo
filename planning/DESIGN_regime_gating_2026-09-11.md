# Design: regime gating — a noise probe and a four-branch table (2026-09-11)

Read-only design against `planning/REGIME_TABLE_2026-09-11.md` and
`planning/DISCOVERY_2026-09-09.md` §38/§41/§42/§43.  Numbers in §1 are
from a Monte-Carlo probe run against `panobbgo/lib/noise.py` today;
everything else is cited.  Status: **design, not implemented.**

## 0. The finding that makes this cheap

The evidence names exactly **one** branch that switches away from the
shipped default, and exactly one that must never fire by mistake:

| regime | best arm | margin, 12 seeds | source |
|---|---|---|---|
| cauchy outliers | **CMA-ES alone** | +0.135 vs DE arms (0/12), +0.101 vs portfolio [−0.124, −0.078] | §42 |
| unif noise, d ≤ 5 | **`Blocks_warm_CMAES_JSO`** | +0.026 vs L-SHADE (9/12), +0.037 vs CMA-ES (10/12) | §42 |
| gauss noise, d ≤ 5 | portfolio, but *not* by the rule | +0.005 vs jSO (5/12) | §42 |
| d = 10 @ 500·dim | **CMA-ES** | portfolio −0.023 (6/12), DE −0.056/−0.069 | §42 |
| noiseless d ≤ 5 | toss-up | arms within ±0.02 | REGIME_TABLE §2 |

So the gate is **CMA-ES everywhere, except: bounded noise at d ≤ 5 →
the sharing portfolio.**  That single switch is what the probe has to
earn, and the only expensive mistake is taking *cauchy* for bounded
noise (−0.101).  Both facts drive everything below.

Two supporting facts that bound the ambition.  Gating by **dimension
alone** is negative on the only 12-seed control (−0.017 [−0.048,
+0.015], 5/12; REGIME_TABLE §3) — where arms are level, selection
costs.  And the paired 12-seed oracle (§43) puts total arm-choice
headroom at **+0.081**, of which CMA-ES + jSO capture 78 % — so the arm
*pair* is settled and the gate is arguing over ≲ 0.02–0.04, not 0.08.

---

## 1. The noise probe

### 1.1 Exact re-evaluation is useless here

`NoisyProblem` with `resample=False` (the default,
`noise.py:423-441,467-479`) makes the noise a pure function of
`(seed, x)`: `apply_noise` hashes the float64 bytes of `x`
(`noise.py:125-143`) and re-evaluating the *same* `x` returns the
*same* value (`noise.py:63-69`).  A probe built on exact repeats
measures **zero** and classifies every battery as noiseless.

Two ways out: **(a)** probe at *nearby* points `x + δ` — distinct byte
patterns, fresh noise draw, true value moved by only O(δ‖∇f‖); or
**(b)** run the noisy batteries with `resample=True`
(`harness_ioh.py:307` `noise_resample`, plumbed at
`harness_ioh.py:1071,1111`).

**Recommend (a).**  (b) changes the *battery*, so every 12-seed number
in §42 would need re-measuring, and it buys nothing: (a) works under
both settings and is the harder case, so a detector validated on it
transfers to a genuinely stochastic simulator (Monte-Carlo model,
measurement rig), where the same statistic with δ = 0 is strictly
easier.  A deterministic simulator with discretisation error — the
engineering default, and what `resample=False` models
(`noise.py:63-69`) — offers *only* (a).

### 1.2 Statistic

k pairs.  Each pair is (an incumbent-quality point the arm evaluated
anyway, a δ-neighbour of it).  Per pair, from the **observed** values:

    l_i = log f_obs(x_i + δ) − log f_obs(x_i)        # relative channel
    a_i = f_obs(x_i + δ) − f_obs(x_i)                # absolute channel

    bounded-noise scale   m = median |l_i|
    outlier tail          t = max |a_i| / median |a_i|

    if t > t_tail:  "outlier"      # tested first — its miss is the costly one
    elif m > t_noise: "bounded"
    else:           "clean"

Thresholds `t_noise = 3e-3`, `t_tail = 10`.  Two channels, not one,
because the BBOB models differ in *kind*: gauss/unif are multiplicative
in the precision (`noise.py:198-199,229-233`) and show in the log
channel; cauchy is **additive** — `f + α·max(0, 1000 + 1{U<p}·C)`,
`noise.py:263-266` — so at α = 0.01 it is a constant +10 shift that
cancels in *both* channels except on the ~5 % of draws that hit, which
show only in the absolute channel.

### 1.3 Misclassification, measured

Synthetic quadratic, d = 5, BBOB **moderate** levels from
`make_noise_model` (`noise.py:339-383`): gauss β = 0.01, unif
α = 0.01·(0.49 + 1/d), cauchy α = 0.01, p = 0.05.  2000 trials per
cell, pooled over four incumbent precisions f_raw ∈ {1e2, 1, 1e-2,
1e-4}; pair separation 1e-6 relative.  Cell = fraction classified
(c)lean / (b)ounded / (o)utlier.

| k pairs | evals | noiseless | gauss | unif | cauchy |
|---:|---:|---|---|---|---|
| 3 | 3 | **c 1.00** b 0.00 o 0.00 | c 0.06 **b 0.93** o 0.02 | c 0.00 **b 0.98** o 0.02 | c 0.76 b 0.00 **o 0.23** |
| 5 | 5 | **c 1.00** b 0.00 o 0.00 | c 0.03 **b 0.96** o 0.01 | c 0.00 **b 0.99** o 0.01 | c 0.60 b 0.00 **o 0.40** |
| 10 | 10 | **c 1.00** b 0.00 o 0.00 | c 0.00 **b 0.99** o 0.00 | c 0.00 **b 1.00** o 0.00 | c 0.37 b 0.00 **o 0.63** |

Four statements:

1. **noiseless vs bounded separates perfectly**, already at k = 3, at
   every stage of the descent: 0 false alarms in 8000 noiseless probes.
2. **gauss and unif do not separate at all** — both read "bounded".
   Fine: §42 sends both to the same arm at d ≤ 5 and the gauss/unif
   *split* does not clear the rule anyway.  The table must therefore
   speak of **bounded noise**, one class.
3. **cauchy is detected 23 / 40 / 63 % at k = 3 / 5 / 10** — weak, and
   bounded by the hit rate itself: P(≥1 hit in 2k draws) =
   1 − 0.95^{2k} = 0.26 / 0.40 / 0.64.  The probe is within a point of
   that ceiling; cleverer statistics cannot help, only more
   evaluations.
4. **The miss is benign.**  Cauchy is *never* called "bounded" (0.00 in
   24 000 probes) — it falls back to "clean", which routes to CMA-ES,
   the *right* arm under cauchy (§42).  The branch that would cost
   −0.101 is the one the probe never takes.

Threshold knee (k = 5, pooled), `t_tail` → (cauchy detected, gauss/unif
false alarm): 5 → (0.28, 0.08); **10 → (0.26, 0.01)**; 30 → (0.23,
0.00).  A false alarm costs −0.037 (unif), 3× less than the miss it
guards, so erring sensitive is right — but below 10 the alarm rate
rises faster than the detection rate.

### 1.4 How close the δ-neighbour must be

Same setup, k = 5, varying the pair's *true* relative gap:

| true Δf/f of the pair | 1e-5 | 1e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 |
|---|---|---|---|---|---|---|
| noiseless → "bounded" | 0.00 | 0.00 | 0.00 | 0.00 | **1.00** | **1.00** |

**Hard constraint: the pair's true objective gap must stay under ~3e-3
relative**, or the local slope is mistaken for noise and every
noiseless run is gated into the portfolio.  This kills the tempting
"free" variant — reusing two points of the *same generation*, whose f
values differ by O(1) relative.  A δ-neighbour of the incumbent is
required; set δ from the box scale, `δ = 1e-6 · problem.box.ranges`,
one coordinate at a time.  A tight gap also makes cauchy detection
*stage-independent*: at a 1e-3 gap it collapses to 0.00 at f_raw = 1e2
(the true gap 0.1 swamps the α = 0.01 jump) and reaches 0.39 only near
the optimum; at 1e-6 it is 0.40 at every stage.

### 1.5 Cost

k = 5 extra evaluations (one per pair; the other member is a point the
arm produced anyway).  AOCC is a **uniform** mean over the padded
best-so-far trace (`ioh_runner.py:82-91`), so spending k evaluations
that cannot improve the incumbent shifts the whole trace right by k and
costs at most `k / max_eval` — ≤ 0.005 at d = 2 / 500·dim (1000 evals)
and ≤ 0.002 at d = 5 (2500 evals).  Inside the 12-seed ±0.02 floor by
a factor of four.  This is *not* the §20 situation (5 % of budget on an
initial design cost −0.38): a δ-neighbour of the incumbent is itself an
incumbent-quality point, so it loses only the delay, not the quality.
**Hard rule: k·(probe cost) ≤ 1 % of `config.max_eval`, clamp k
accordingly; skip the probe entirely if `max_eval < 200`.**

---

## 2. The gate

### 2.1 Where it lives — recommendation

* **(a) a `StrategyRegimeGate` wrapper — rejected**, and the reason is
  worth stating plainly: *one strategy owns the run.*  `start()` →
  `initialize()` → `_run()` (`core.py:1621-1678`) builds the event bus,
  the four mandatory analyzers and the evaluator once and publishes
  `start` exactly once (`core.py:1660-1666`); a wrapper cannot hold two
  built strategies and pick one after 5 evaluations without discarding
  a live evaluator or double-counting the probe.
* **(b) a `StrategySpec`-level `gate_noise`**, extending
  `gate_min_dim` / `gate_max_dim` (`benchmark.py:110-119`, applied at
  `benchmark.py:211-220`).  Dimension is available there
  (`problem.dim`); the noise class is **not** — a benchmark can cheat
  (it built the `NoisyProblem`), a library user cannot.  **Rejected as
  the product, kept as the oracle** (§4).
* **(c) a "regime prior" inside `StrategyBlockBandit`**
  (`blocks.py:526` `_open_block`).  The portfolio spec already *is* one
  (`harness_ioh.py:598-624`).

**Recommendation: (c′) — a flag on `StrategyBlockBandit`,
`regime_gate="table-v1"`, with an `enabled` mask over its own arms.**
The spec already carries both arms; gating *disables* one rather than
choosing between two strategies.

### 2.2 The RNG-order contract, and how to keep byte-identity

`StrategyBase.initialize` adds heuristics **sorted by name**
(`core.py:1624-1625`), and each `add_heuristic` → `init_module` →
`spawn_rng` draws the module's stream from the master generator in that
order (`core.py:1687-1692`, `core.py:499-509`).  Constructing an arm
*lazily*, after the probe, shifts every later stream and changes the
run bit-for-bit.

**So: construct every arm up front, always.**  The gate flips a boolean
afterwards; arm set, construction order and spawned streams are
identical whether it fires or not.  Then `regime_gate=None` is
byte-identical to today's `Blocks_warm_CMAES_JSO` — provable by a test
in the style of `tests/test_reproducibility.py`.

Caveat to document: a **disabled** arm still receives `on_new_results`
and still tops up its queue, so its own stream advances (the §4 note in
`DESIGN_block_bandit_2026-09-10.md`).  Deterministic under
`sync_evaluation`, but it means "CMA-ES alone via the gate" is **not**
byte-identical to `CMAES_alone` — a different, measurably
equal-or-better (§42) thing.  §4 measures both.

### 2.3 Interface

```python
class StrategyBlockBandit(StrategyBase):
    def __init__(self, problem, *, regime_gate: str | None = None,
                 probe_k: int = 5, probe_rel: float = 1e-6,
                 probe_t_noise: float = 3e-3, probe_t_tail: float = 10.0,
                 **kw): ...
```

* `regime_gate=None` (default) — today's behaviour, no probe, no mask.
* `regime_gate="table-v1"` — run the probe once, in the **first block**
  (after the arms' `on_start` points exist, so there is an incumbent to
  perturb), then apply `REGIME_TABLE_V1`.
* `regime_gate="oracle:<class>"` — skip the probe, take the class as
  given.  This is how the harness passes the known regime (§4).

Mechanism in the scheduler: a per-arm `self._enabled: dict[str, bool]`,
honoured in exactly one place — `_select`'s `ready` list
(`blocks.py:764`) becomes

```python
ready = [h for h in self.heuristics
         if self._enabled.get(h.name, True) and (h.can_produce or self._can_warm_start(h))]
```

Nothing else changes: `execute` (`blocks.py:794-817`) already handles
`_select` returning one arm forever, and the block-bandit test #8
(single arm matches `StrategyRoundRobin`) pins that a one-arm mask
behaves as `RoundRobin_CMAES`.  The prologue list (`blocks.py:795-796`)
must be filtered at the moment the mask is set, or the disabled arm
still gets its prologue block.

### 2.4 The table

Stored as a module-level dict in `panobbgo/strategies/blocks.py`, next
to the class, versioned, with the section each row came from — the
`GATE_MIN_DIM_KEY` docstring (`benchmark.py:110-117`) is the precedent
for documenting the evidence beside the gate.

```python
#: Regime -> arm set.  Version it: every row is an empirical claim with a
#: date and a section, and a row without 12-seed evidence does not belong.
REGIME_TABLE_V1 = {
    #  (noise class, dim predicate, budget predicate) -> enabled arms
    ("outlier", None,       None):          ("CMAES",),          # §42, 12 seeds, 0/12 against
    ("bounded", "dim <= 5", None):          ("CMAES", "JSO"),    # §42 unif +0.037 vs CMA-ES, 10/12
    (None,      "dim >= 10", "bpd <= 500"): ("CMAES",),          # §42, portfolio -0.023, DE clear
    # default, and every cell not named above:
    (None,      None,       None):          ("CMAES",),          # REGIME_TABLE §6
}
```

Four branches, first match wins.  Deliberately absent: `d = 10 @
2000·dim` (L-SHADE, 3 seeds), **constrained** (DE arms, 3 seeds), any
gauss/unif split (the probe cannot make it and §42 does not support
it), and any branch on dimension alone at d ≤ 5 (the 12-seed control is
negative, REGIME_TABLE §3).  Constrained-or-not *is* free to read —
`problem.eval_constraints` returns `None` when unconstrained
(`lib/lib.py:404-411`), and a handler is a `ConstraintHandler` subclass
(`lib/constraints.py:29`) — it is left out for lack of seeds, not lack
of access.  Add rows as the in-flight 2000·dim / 200·dim / constrained
12-seed runs land, bumping to `_V2`.

### 2.5 How a wrong probe degrades

Enumerate all four true classes against all three probe verdicts, at
d ≤ 5 where the only switch lives (Δ AOCC vs a fixed CMA-ES default,
from §42):

| truth \ verdict | clean → CMA-ES | bounded → portfolio | outlier → CMA-ES |
|---|---|---|---|
| noiseless | 0 | **+0.03** (12-seed d5) / ≈0 (d2) | 0 |
| gauss | 0 | +0.022 | 0 |
| unif | 0 | **+0.037** | 0 |
| cauchy | 0 | **−0.101** ← never observed (§1.3) | 0 |

Every reachable cell is ≥ 0; the one negative cell has measured
probability 0.00 in 24 000 trials, because cauchy's signature is absent
from the relative channel by construction (`noise.py:263-266`).
**Worst case over the reachable cells is the probe's own cost**,
≤ 0.005.

---

## 3. What "activate the portfolio" means at runtime

Both arms are constructed at `initialize()`; the second arm's queue is
never drained while its mask bit is false, because `_select` never
returns it.  It keeps receiving `on_new_results`, keeps its `active`
flag (`core.py:913-925`), keeps filtering foreign results on its own
`who` prefix — the "a paused arm needs no pause/resume API" argument of
`DESIGN_block_bandit_2026-09-10.md` §4, one step further.  The
alternative ("the gate picks between two pre-built strategies") is not
implementable (§2.1); say so in the docstring so it is not re-proposed.

---

## 4. Experiment

`benchmarks/portfolio_screen.py`, four specs, `kind=` selecting the
regime (`portfolio_screen.py:481-499`), 12-seed roster
`42 7 1234 2025 3 11 99 123 777 2024 31337 555`, `seed_name="screen"`
so all four share the stream (`portfolio_screen.py:510-522`).

| spec | what it is |
|---|---|
| `CMAES_alone` | the bar (`portfolio_screen.py:141`) |
| `Blocks_warm_CMAES_JSO` | the thing being gated (`harness_ioh.py:598-624`) |
| `RegimeGate` | `regime_gate="table-v1"` — probe + table |
| `RegimeGate_oracle` | `regime_gate="oracle:<class>"`, class injected by the battery — **the upper bound** |

Cells: `kind=noisy-cauchy`, `noisy-gauss`, `noisy-unif` (d 2/5,
500·dim), `kind=standard` (the must-not-lose control), `kind=highdim
dims=10 bm=500`.  ~1 h total at the §42 timings.

Acceptance rule, as everywhere: 12 seeds, paired CI excluding zero on
the positive side, ≥ 9/12 seeds, no dimension negative; ±0.02 floor.

Expected sizes, from §42: **cauchy** +0.101 over the portfolio minus the
probe cost — and, because a missed cauchy falls back to "clean" →
CMA-ES (the same arm), that recovery is expected *regardless* of the
0.40 detection rate; if it is not there, the mask is wrong, not the
probe.  **unif** ≈ +0.03 over `CMAES_alone` (detection 0.99).
**gauss** ≈ +0.02, not expected to clear the rule.  **noiseless** ≈ 0,
must not lose more than the floor (−0.02) — the run that can kill the
design.  **d = 10 @ 500·dim** ≈ 0 vs `CMAES_alone`, +0.02 vs the
portfolio.

**Falsifier.**  If `RegimeGate` < `RegimeGate_oracle` by **more than
the probe cost (0.005)**, the probe is the problem and the table is
fine — raise k, or move the probe later in the run.  If
`RegimeGate_oracle` itself fails to beat `CMAES_alone` by the
acceptance rule on cauchy/unif, **the table is wrong and no probe can
save it** — stop, and record that regime gating stays a footnote
(REGIME_TABLE §6).

---

## 5. Risks

1. **Probe cost in the AOCC integral.**  AOCC is a uniform mean
   (`ioh_runner.py:82-91`), so the probe does not "weigh more" early —
   it delays the *whole* trace by k, hence the `k/max_eval` bound, and
   hence it must never be a random design (§20: 5 % on a design cost
   −0.38).  Mitigation: δ-neighbours of the incumbent, k ≤ 1 % of
   budget, skip below 200 evaluations.
2. **Misclassification at small k** — §1.3; benign in every reachable
   cell (§2.5), but the 0.40 cauchy detection rate at k = 5 is *not*
   the safety margin people will assume.  The safety comes from the
   fallback, not the detector; document it that way.
3. **`resample` semantics.**  Switching the batteries to
   `resample=True` (`harness_ioh.py:307`) makes the probe easier and
   re-opens every §42 number.  Keep `resample=False`; the probe must
   not depend on it either way.
4. **`_derive_noise_seed` (`harness_ioh.py:1009-1020`) hashes
   `noise|seed|kind|dim|instance|rep`, deliberately *not* the strategy
   name (`harness_ioh.py:1231-1233`)**, so every arm in a cell sees the
   identical noise field — good (the `RegimeGate` vs oracle comparison
   is paired on the realisation), and the same determinism, one level
   down, that makes exact re-evaluation noise-free.  One fact, two
   consequences.
5. **Library path vs benchmark path.**  The probe is the only way a
   library user learns the noise class; `gate_noise` on a
   `StrategySpec` works only in benchmarks.  Hence the gate lives in
   the strategy and the spec-level form exists only as the oracle.
6. **The gate argues over a small number.**  §43: headroom +0.081, of
   which CMA-ES + jSO already capture 78 %.  If `RegimeGate_oracle`
   scores below `CMAES_alone` + 0.02 on the noisy cells, the honest
   conclusion is that the flagship is already right almost everywhere
   (REGIME_TABLE §6.4) and this file is a footnote.

---

## 6. Implementation plan, smallest first

| # | step | effort | gate to the next step |
|---|---|---|---|
| 1 | **Oracle gate via the harness.** `regime_gate="oracle:<class>"` + the `_enabled` mask in `_select` (`blocks.py:764`) + prologue filter; two specs in `portfolio_screen.py` wiring the battery's known noise tag (`harness_ioh.py:239-241,248`). No probe. | ~60 lines + 2 specs, half a day | run the §4 battery. If the oracle does not clear the rule on cauchy **and** unif, **stop**. |
| 2 | **Probe, offline.** `panobbgo/lib/noise_probe.py`: `classify(pairs) -> {"clean","bounded","outlier"}`, pure function of the observed values, plus unit tests replaying §1.3's Monte Carlo at fixed seeds (assert the confusion cells within ±0.03). | ~80 lines + tests, half a day | confusion table reproduces. |
| 3 | **Probe, in-run.** First-block hook in `StrategyBlockBandit`: emit k δ-neighbours of the incumbent, collect their results in `on_new_results`, classify, set `_enabled`, filter the prologue. Reproducibility test: `regime_gate=None` byte-identical to today (`tests/test_reproducibility.py:19-49` pattern); gated path deterministic under `sync_evaluation`. | ~80 lines + tests, one day | byte-identity holds. |
| 4 | **The 12-seed battery** of §4, `RegimeGate` vs the oracle vs both fixed arms. | ~1 h compute | falsifier in §4. |
| 5 | **Ship or shelve.** If it clears: `regime_gate="table-v1"` becomes the default of `Blocks_warm_CMAES_JSO` only — `RoundRobin_CMAES` stays the flagship until the gate beats it by the rule. Record in DISCOVERY; bump the table to `_V2` when the in-flight 2000·dim / 200·dim / constrained runs land. | a day | — |

Steps 1 and 4 alone answer the question.  Steps 2–3 are only worth
building if step 1's oracle clears.
