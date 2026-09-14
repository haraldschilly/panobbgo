# Extending the benchmark suite (2026-09-14)

Harald, 2026-09-14: extend the suite with *existing problems from large
standard benchmarks* **and** *own parametrised synthetic problems, new or
building on those*.  This note fixes the scope against what already
exists, so neither half is rebuilt.

## What already exists

* **IOH track** (`panobbgo/harness_ioh.py`, worker subprocess): MA-BBOB
  instances, dims (2, 5) and (10, 20), noise variants (gauss/unif/cauchy,
  two levels), budgets 100…2000·dim.  `tools/ioh_worker` already accepts
  `kind="BBOB"` with `fid=1..24`, and `panobbgo.lib.ioh_wrapper.IOHProblem`
  already takes `fid`.
* **Families track** (`panobbgo/lib/families.py`,
  `panobbgo/harness_families.py`, `benchmarks/family_screen.py`):
  BBOB-style parametrised instances with a known optimum —
  `f(x) = f_base(Λ R (x − x_opt)) + f_opt` — over nine bases (sphere,
  rosenbrock, rastrigin, ackley, griewank, schwefel, ellipsoid, discus,
  sharp_ridge), with knobs `condition`, `rotate`, `shift`,
  `n_constraints`, `constraint_kind`, and the *same* AOCC measurement as
  the IOH track (same tracker, same records, same seed derivation), so
  `portfolio_screen.py`'s analysis works on its rows unchanged.

So both halves of Harald's sentence have a home already.  What is missing
is an **axis** on each.

## Gap 1 — the standard-benchmark function axis

`IOHBatterySpec` has no `fids` field: a battery is one problem family
over (dim × instance × rep).  The 24 BBOB functions are therefore
reachable only one battery at a time, and results cannot be grouped by
the COCO classes (separable / low conditioning / high conditioning
unimodal / multimodal with global structure / multimodal weak structure).

**Why it matters now.**  Every result of §44–§51 rests on MA-BBOB
*mixtures*, which are affine combinations of two BBOB functions — broad
by construction but not attributable to a class.  The regime table
(`DESIGN_regime_gating_2026-09-11.md`) currently keys on noise, dim, bpd
and constrainedness because those are the only axes measured.  A class
axis is the one that could replace `bpd <= 200` with something about the
*landscape* rather than the wallet.

**Design.**  Add `fids: Tuple[int, ...] = ()` to `IOHBatterySpec` (empty =
today's behaviour).  The run cell becomes `(fid, dim, instance, rep)`;
`fid` joins the `_derive_seed` and `_derive_noise_seed` payloads and the
`IOHRunRecord`.  Group labels come from a `BBOB_CLASS_OF_FID` mapping
(the five COCO groups, f1–f24) so the screen can print per-class blocks.
`benchmarks/portfolio_screen.py` gets `fids=` and folds cells on the
extended key; a `kind=bbob` battery preset carries all 24.

## Gap 2 — parametrised synthetic problems as *sweeps*, and the shapes still missing

The family machinery supports one-off configs, but the preset battery
uses five fixed families at default knobs.  Two extensions:

1. **Regime sweeps.**  A battery that varies one knob over a range while
   holding everything else fixed — first and most useful, `condition` on
   `ellipsoid` over 1e0, 1e2, 1e4, 1e6.  That is a *continuous* version of
   the question the regime table asks discretely ("is there a covariance
   model worth having here?"), and the campaign's central claim — sharing
   pays where a single arm cannot converge inside the budget — predicts a
   monotone response along it.
2. **The shapes the nine bases do not cover**, all standard and all
   parametrisable in the same construction:
   * `lunacek_bi_rastrigin` — a **double funnel**: a deceptive global
     structure where the second funnel is broader.  This is the canonical
     class where restarts and portfolios should pay most, and nothing in
     the current battery has it.  Knob: funnel depth/width ratio.
   * `gallagher` — many random peaks, **weak global structure**.  Knob:
     number of peaks.
   * `attractive_sector` — strongly asymmetric around the optimum.
   * `step_ellipsoid` — plateaus: the case where ranking-based methods
     see ties and a smooth surrogate is systematically wrong.
   * `bent_cigar` — one soft direction, curved.

## Staging (cost)

144 cells per spec and seed on the full BBOB axis (24 fids × 3 instances
× 2 dims) against today's 10.  At the measured ~0.7 s per run that is
~40 min for 6 seeds × 4 specs at 200·dim, ~100 min for 12 × 5.  So:

* **screens** stay on the small MA-BBOB cube (cheap, many specs),
* **decisions** run on the wide axis with fewer seeds — the paired
  variance comes mostly from cells, not seeds,
* every accepted result of §44–§51 gets re-tested once on the wide axis
  before it becomes a default.  That is the gate Harald set on
  2026-09-13.

## Order of work

1. Gap 1 (fid axis, class labels, screen support) — the mechanical half,
   unblocks the re-test.
2. Re-test the campaign's three standing claims on the BBOB axis: sharing
   wins at low budget; block length by budget; the payload is m and σ.
   Report **per class**, which is the point of the axis.
3. Gap 2 (new bases + the conditioning sweep), then the same re-test on
   the families track.
