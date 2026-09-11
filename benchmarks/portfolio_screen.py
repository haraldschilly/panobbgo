"""Step 2 of the block-bandit experiment plan: does a *portfolio* beat one arm?

``planning/DESIGN_block_bandit_2026-09-10.md`` §6 asks for a three-seed
screen before any twelve-seed roster is spent: run the tuned Phase A arms
under :class:`~panobbgo.strategies.StrategyBlockBandit` and compare them,
paired per (seed, dim, instance) cell, against the two references that a
portfolio has to beat — CMA-ES alone (the current competition candidate,
"the bar") and L-SHADE alone (the best single arm in the provisional
oracle).

The screen separates the two things a bandit portfolio does:

* ``Blocks_uniform_2`` blocks the budget but *learns nothing* (round-robin
  over blocks).  Its delta to the references is the price of blocking —
  the transient every switch costs, and the budget CMA-ES no longer has
  for its covariance adaptation.
* ``Blocks_ducb_2`` adds the discounted-UCB rule on the same arms.  Its
  delta to ``Blocks_uniform_2`` is what the AOCC-area reward *learns*.
  If that is zero, the reward carries no signal and the policy is noise.
* ``Rewarding_ema_2`` is the interleaving control: the same two arms under
  the existing probability-matching strategy, which hands every ready arm
  points on every pass instead of giving one arm a contiguous block.
* The ``*_warm`` specs turn the shared :class:`~panobbgo.analyzers.Archive`
  on and let L-SHADE re-seed from it whenever the scheduler hands it a
  block back (``warm_start_on_resume=True``).  Their delta to the cold
  specs is the direct test of the thesis a portfolio stands on: arms are
  only worth their switching transients if they *share* the evaluations
  they paid for.  CMA-ES has no warm-start hook yet, so the sharing is
  one-directional — CMA-ES simply resumes its own paused state.

Every spec shares ``seed_name="screen"``, so all of them run the identical
RNG stream on each (dim, instance, rep) cell and a delta carries only the
strategy's own effect, not the run-to-run variance (see
:attr:`panobbgo.benchmark.StrategySpec.seed_name`).

Usage::

    uv run python benchmarks/portfolio_screen.py OUT.json SEED [SEED ...] \
        [kind=standard] [dims=2,5] [bm=500] [insts=0,1,2] [specs=name,name]

    OUT.json  rows file, rewritten after every seed
    SEED      base seeds; three screens, twelve decides
    kind      battery preset (default ``standard``); see ``BATTERIES``
    dims      battery dimensions (default: the preset's)
    bm        budget multiplier; the budget per run is ``bm * dim``
    insts     instance ids (default: the preset's)
    specs     subset of ``SPECS`` to run (default: all of them)

Re-analysis of a finished run is free::

    uv run python benchmarks/portfolio_screen.py from=OUT.json [specs=a,b]

The screening gates of §6 are printed at the end with PASS/FAIL.  Read
them against the measured null floor: on three seeds a CMA-ES-containing
spec moves by up to ±0.05 for no reason at all, so only deltas larger
than that are evidence.
"""

import dataclasses
import json
import statistics as st
import sys
import time
from collections import defaultdict

from panobbgo.analyzers import Archive
from panobbgo.harness_ioh import (
    make_highdim_battery,
    make_ioh_strategies,
    make_noisy_battery,
    make_noisy_highdim_battery,
    make_standard_battery,
    run_ioh_harness,
)
from panobbgo.heuristics import CMAES, JSO, LSHADE, NLSHADE_LBC, PSO
from panobbgo.strategies import StrategyBlockBandit, StrategyRewarding, StrategyRoundRobin

BASE = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"][0]

#: The tuned Phase A arms.  One dict, so "which arm with which knobs" is a
#: single obvious edit and every spec below quotes the same settings.
ARM = {
    "cmaes": (CMAES, {}),
    # ``NP_init="auto"`` is now the accepted rule (12/12), so every DE arm
    # runs on the shipped default rather than a hand-picked constant.
    "lshade": (LSHADE, {"NP_init": "auto"}),
    "jso": (JSO, {"NP_init": "auto"}),
    "lbc": (NLSHADE_LBC, {"NP_init": "auto", "k_rank": 3.0}),
    # PSO won zero cells in the provisional oracle, so it only appears in
    # the deliberately over-armed five-arm spec.
    "pso": (PSO, {"NP": 6}),
}


def arms(*keys):
    return [ARM[k] for k in keys]


def warm(key, mode):
    """The tuned arm ``key``, re-seeding from the shared ``Archive`` on re-acquisition."""
    cls, kw = ARM[key]
    return (cls, {**kw, "warm_start": mode})


def warm_lshade(mode):
    """``warm("lshade", mode)`` — kept because the §21 specs below read better with it."""
    return warm("lshade", mode)


#: ``warm_start_only_if_foreign=False``: re-seed on *every* re-acquisition.
#: §21 measured the ``_any`` variants above the foreign-only default
#: (``Blocks_ducb_2_warm_any`` +0.005 over ``Blocks_ducb_2_warm``), so every
#: spec added afterwards uses it.
WARM_ON = {"warm_start_on_resume": True, "warm_start_only_if_foreign": False}

#: The §22 pair, both arms warm — the arms every §24/§27 spec is built on.
CJ = [warm("cmaes", "archive"), warm("jso", "archive")]

#: D-UCB with the commitment taken out (§26's leader): heavy exploration, no
#: incumbent bonus, a short memory.  The hard default is ucb_c 0.5,
#: hysteresis 1.2, gamma 0.9 and relays 8x less often.
SOFT = {"policy": "ducb", "ucb_c": 2.0, "hysteresis": 1.0, "gamma": 0.7}

#: ``warm_start_only_if_better`` became a default (True) in c13748d, *after*
#: screen #6 was measured.  Every spec added since pins it explicitly, so
#: ``False`` reproduces what #6 actually ran and ``True`` is the new guard.
NOB = {**WARM_ON, "warm_start_only_if_better": False}


#: Analyzer list of every warm spec.  ``Splitter`` is **not** listed: it is
#: one of the four analyzers ``StrategyBase.initialize`` always installs
#: (``core.py:1327``), so ``archive_leaf`` finds it without help.  ``Archive``
#: is the opt-in one, and it must be present or ``archive_seed`` silently
#: falls back to the Splitter root.
ARCHIVE = [(Archive, {})]

#: name -> (strategy_class, heuristics, strategy kwargs, analyzers).  Strategy
#: kwargs travel through ``config_overrides``: ``create_strategy`` passes them
#: to the constructor for a ``StrategyBase`` subclass (benchmark.py:204).
SPECS = {
    # -- references ------------------------------------------------------
    "CMAES_alone": (StrategyRoundRobin, arms("cmaes"), {}, []),
    "LSHADE_alone": (StrategyRoundRobin, arms("lshade"), {}, []),
    # -- blocking, with and without learning ------------------------------
    "Blocks_uniform_2": (StrategyBlockBandit, arms("cmaes", "lshade"), {"policy": "uniform"}, []),
    "Blocks_ducb_2": (StrategyBlockBandit, arms("cmaes", "lshade"), {"policy": "ducb"}, []),
    "Blocks_ducb_4": (StrategyBlockBandit, arms("cmaes", "lshade", "jso", "lbc"), {"policy": "ducb"}, []),
    "Blocks_ducb_5": (StrategyBlockBandit, arms("cmaes", "lshade", "jso", "lbc", "pso"), {"policy": "ducb"}, []),
    # -- the interleaving control -----------------------------------------
    "Rewarding_ema_2": (StrategyRewarding, arms("cmaes", "lshade"), {}, []),
    # -- the same portfolios, but the arms now SHARE their evaluations ----
    #
    # The thesis a portfolio stands or falls on: an arm that resumes from
    # the best points the *other* arm paid for does not re-buy them.  Only
    # L-SHADE warm-starts here — CMA-ES has no ``warm_start`` hook yet, so
    # it simply resumes its own paused state (its covariance, step size and
    # mean survive the pause untouched).  The asymmetry is the honest
    # measurement of what exists today, not a handicap: the point is whether
    # sharing in *one* direction already moves the number.
    "Blocks_uniform_2_warm": (
        StrategyBlockBandit,
        [ARM["cmaes"], warm_lshade("archive")],
        {"policy": "uniform", "warm_start_on_resume": True},
        ARCHIVE,
    ),
    "Blocks_ducb_2_warm": (
        StrategyBlockBandit,
        [ARM["cmaes"], warm_lshade("archive")],
        {"policy": "ducb", "warm_start_on_resume": True},
        ARCHIVE,
    ),
    # ``archive_leaf`` takes the best point of each of the k best Splitter
    # leaves: k *different basins* rather than k neighbours of one incumbent.
    "Blocks_ducb_2_warm_leaf": (
        StrategyBlockBandit,
        [ARM["cmaes"], warm_lshade("archive_leaf")],
        {"policy": "ducb", "warm_start_on_resume": True},
        ARCHIVE,
    ),
    "Blocks_ducb_2_warm_div": (
        StrategyBlockBandit,
        [ARM["cmaes"], warm_lshade("archive_diverse")],
        {"policy": "ducb", "warm_start_on_resume": True},
        ARCHIVE,
    ),
    # ``warm_start_only_if_foreign=False``: re-seed on *every* re-acquisition,
    # even when the top-k are all the arm's own points.  The default skips
    # that case because re-seeding an arm from itself is a no-op that still
    # throws away its live generation; these two specs measure whether the
    # skip is worth its condition or is just suppressing warm starts.
    "Blocks_uniform_2_warm_any": (
        StrategyBlockBandit,
        [ARM["cmaes"], warm_lshade("archive")],
        {"policy": "uniform", "warm_start_on_resume": True, "warm_start_only_if_foreign": False},
        ARCHIVE,
    ),
    "Blocks_ducb_2_warm_any": (
        StrategyBlockBandit,
        [ARM["cmaes"], warm_lshade("archive")],
        {"policy": "ducb", "warm_start_on_resume": True, "warm_start_only_if_foreign": False},
        ARCHIVE,
    ),
    # -- §22: the pair the 12-seed oracle actually points at ---------------
    #
    # No arm is a champion any more (jSO 0.656, L-SHADE 0.642, CMA-ES 0.642,
    # winning different cells) and the best two-arm oracle is CMA-ES + jSO,
    # capturing 62% of a +0.076 headroom.  So the portfolio worth screening
    # is that pair — and now that CMA-ES has a ``warm_start`` of its own
    # (4ef8b35) the sharing can finally run in *both* directions.
    "JSO_alone": (StrategyRoundRobin, arms("jso"), {}, []),
    "Blocks_ducb_cj": (StrategyBlockBandit, arms("cmaes", "jso"), {"policy": "ducb"}, []),
    # One-directional: only jSO re-seeds, CMA-ES resumes its own paused
    # distribution.  The control that isolates what CMA-ES's own warm start
    # adds in ``Blocks_ducb_cj_warm2``.
    "Blocks_ducb_cj_warmJ": (
        StrategyBlockBandit,
        [ARM["cmaes"], warm("jso", "archive")],
        {"policy": "ducb", **WARM_ON},
        ARCHIVE,
    ),
    "Blocks_ducb_cj_warm2": (
        StrategyBlockBandit,
        [warm("cmaes", "archive"), warm("jso", "archive")],
        {"policy": "ducb", **WARM_ON},
        ARCHIVE,
    ),
    # ``archive_cov`` additionally seeds C with the covariance of the seed
    # cloud; ``archive`` only moves the mean and sigma and leaves C = I.
    "Blocks_ducb_cj_warm2cov": (
        StrategyBlockBandit,
        [warm("cmaes", "archive_cov"), warm("jso", "archive")],
        {"policy": "ducb", **WARM_ON},
        ARCHIVE,
    ),
    # Blocking without learning, on the fully warm pair: separates what the
    # d-UCB rule contributes from what sharing contributes.
    "Blocks_uniform_cj_warm2": (
        StrategyBlockBandit,
        [warm("cmaes", "archive"), warm("jso", "archive")],
        {"policy": "uniform", **WARM_ON},
        ARCHIVE,
    ),
    # The §21 pair, both arms warm, for continuity across the two screens.
    "Blocks_ducb_cl_warm2": (
        StrategyBlockBandit,
        [warm("cmaes", "archive"), warm("lshade", "archive")],
        {"policy": "ducb", **WARM_ON},
        ARCHIVE,
    ),
    # -- §24: is the mechanism "switch often, relay often"? ----------------
    #
    # §22 measured 5 warm starts in 40 blocks under d-UCB against 39 in 41
    # under uniform, and the uniform spec won.  If the relay is what pays,
    # then (a) finer blocks should keep paying — every block boundary is one
    # hand-off — and (b) a third and fourth arm should now *help* rather than
    # dilute, because an arm that is not running still feeds the archive the
    # others start from.  Both are falsifiable here.
    #
    # Block size is ``max(round(max_eval / n_blocks), 2 * dim)`` and a draw
    # asks for ``size = 10`` points, so the grid bottoms out:
    #   d=2 (1000 evals): nb25 -> 40, nb50 -> 20, nb100 -> 10, nb200 -> 5
    #   d=5 (2500 evals): nb25 -> 100, nb50 -> 50, nb100 -> 25, nb200 -> 12
    # At nb200/d=2 the block is 5 evaluations while one draw asks for 10, so
    # the hard cap (``2 * block_size`` = 10) closes the block after a *single*
    # draw: nb200 is not really "200 blocks of 5", it is "one generation per
    # block", the finest granularity the scheduler can express.  That is a
    # meaningful end of the grid — it is the relay taken to its limit — but
    # it is not the nominal number, so read it as such.
    "Blocks_uniform_cj_warm2_nb25": (
        StrategyBlockBandit,
        [warm("cmaes", "archive"), warm("jso", "archive")],
        {"policy": "uniform", "n_blocks": 25, **WARM_ON},
        ARCHIVE,
    ),
    "Blocks_uniform_cj_warm2_nb100": (
        StrategyBlockBandit,
        [warm("cmaes", "archive"), warm("jso", "archive")],
        {"policy": "uniform", "n_blocks": 100, **WARM_ON},
        ARCHIVE,
    ),
    "Blocks_uniform_cj_warm2_nb200": (
        StrategyBlockBandit,
        [warm("cmaes", "archive"), warm("jso", "archive")],
        {"policy": "uniform", "n_blocks": 200, **WARM_ON},
        ARCHIVE,
    ),
    # Three and four arms, all warm.  Cold portfolios lost monotonically with
    # every added arm (§21: 2 arms -0.090, 4 arms -0.187, 5 arms -0.210).
    # If sharing is what makes a portfolio work, that ordering should break.
    "Blocks_uniform_cjl_warm3": (
        StrategyBlockBandit,
        [warm("cmaes", "archive"), warm("jso", "archive"), warm("lbc", "archive")],
        {"policy": "uniform", **WARM_ON},
        ARCHIVE,
    ),
    "Blocks_uniform_cjls_warm4": (
        StrategyBlockBandit,
        [
            warm("cmaes", "archive"),
            warm("jso", "archive"),
            warm("lbc", "archive"),
            warm("lshade", "archive"),
        ],
        {"policy": "uniform", **WARM_ON},
        ARCHIVE,
    ),
    # Can learning be made to beat rotation once sharing is on?  The d-UCB
    # default commits hard (ucb_c 0.5, hysteresis 1.2, gamma 0.9) and so
    # relays rarely.  This is the same rule with the commitment taken out:
    # heavy exploration, no incumbent bonus, a short memory.
    "Blocks_ducb_cj_warm2_soft": (StrategyBlockBandit, CJ, {**SOFT, **WARM_ON}, ARCHIVE),
    # -- §27 A: block length in ABSOLUTE evaluations ----------------------
    #
    # §26 read an interior optimum in ``n_blocks`` (nb50 best, nb25 and
    # nb100 worse) and noted that the d=2 and d=5 optima disagreed at nb100,
    # which is exactly what a *relative* knob looks like when the truth is
    # absolute: ``n_blocks`` fixes the number of decisions, so the block
    # length it implies scales with ``500 * dim``.  ``block_evals`` sets the
    # length directly and identically at every dimension, so if ~20-25
    # evaluations really is the crossing point between "enough consecutive
    # budget to adapt" and "relay often", these five specs should peak in
    # the middle at *both* dimensions.
    #
    # ``block_evals`` bypasses the ``2 * dim`` floor (``__init__`` stores it
    # verbatim), but the draw granularity still rounds up: a block closes on
    # ``block_n >= block_size and drained``, and a draw asks for ``size = 10``
    # points, so a nominal 12 cannot be realised as 12.  The probe reports
    # the realised median per dimension.
    "Blocks_uniform_cj_warm2_be12": (
        StrategyBlockBandit,
        CJ,
        {"policy": "uniform", "block_evals": 12, **WARM_ON},
        ARCHIVE,
    ),
    "Blocks_uniform_cj_warm2_be20": (
        StrategyBlockBandit,
        CJ,
        {"policy": "uniform", "block_evals": 20, **WARM_ON},
        ARCHIVE,
    ),
    "Blocks_uniform_cj_warm2_be30": (
        StrategyBlockBandit,
        CJ,
        {"policy": "uniform", "block_evals": 30, **WARM_ON},
        ARCHIVE,
    ),
    "Blocks_uniform_cj_warm2_be50": (
        StrategyBlockBandit,
        CJ,
        {"policy": "uniform", "block_evals": 50, **WARM_ON},
        ARCHIVE,
    ),
    "Blocks_uniform_cj_warm2_be80": (
        StrategyBlockBandit,
        CJ,
        {"policy": "uniform", "block_evals": 80, **WARM_ON},
        ARCHIVE,
    ),
    # -- §27 B: which soft-D-UCB knob carries the gain? -------------------
    #
    # ``_soft`` (ucb_c 2.0, hysteresis 1.0, gamma 0.7) led §26 at 0.6921 by
    # relaying 29 times in 40 blocks — between the hard default's 5 and plain
    # rotation's 39.  One knob at a time from that point: ``ucb_c`` sets how
    # much exploration outweighs the estimate, ``gamma`` how fast the arm
    # statistics forget.  ``hysteresis`` is already 1.0 (off) and has nowhere
    # softer to go, so it is not on the grid.
    "Blocks_ducb_cj_warm2_soft_c1": (StrategyBlockBandit, CJ, {**SOFT, "ucb_c": 1.0, **WARM_ON}, ARCHIVE),
    "Blocks_ducb_cj_warm2_soft_c4": (StrategyBlockBandit, CJ, {**SOFT, "ucb_c": 4.0, **WARM_ON}, ARCHIVE),
    "Blocks_ducb_cj_warm2_soft_g05": (StrategyBlockBandit, CJ, {**SOFT, "gamma": 0.5, **WARM_ON}, ARCHIVE),
    "Blocks_ducb_cj_warm2_soft_g09": (StrategyBlockBandit, CJ, {**SOFT, "gamma": 0.9, **WARM_ON}, ARCHIVE),
    # The two findings crossed: soft learning on an absolute block length.
    "Blocks_ducb_cj_warm2_soft_be25": (StrategyBlockBandit, CJ, {**SOFT, "block_evals": 25, **WARM_ON}, ARCHIVE),
    # The two warm-start guards are now defaults (c13748d); pin both variants so
    # the measured configuration (no only_if_better) stays reproducible.
    "Blocks_ducb_cj_warm2_soft_be25_nob": (
        StrategyBlockBandit,
        CJ,
        {**SOFT, "block_evals": 25, **WARM_ON, "warm_start_only_if_better": False},
        ARCHIVE,
    ),
    "Blocks_ducb_cj_warm2_soft_be25_oib": (
        StrategyBlockBandit,
        CJ,
        {**SOFT, "block_evals": 25, **WARM_ON, "warm_start_only_if_better": True},
        ARCHIVE,
    ),
    # -- §28 (1): is the tail the whole of "soft"? -------------------------
    #
    # §27 found ucb_c and gamma inert (ucb_c 4 was bit-identical to 2 in
    # 30/30 cells) and read the leader as "round-robin plus a greedy tail":
    # with hysteresis 1.0 and ucb_c >= 2 the exploration bonus swamps the
    # value estimate, so the policy alternates until ``_exploration_c``
    # switches the bonus off in the last ``tail_frac`` of the budget and the
    # incumbent then keeps every remaining block.  ``tail_frac=0`` is the
    # falsifying control: if it lands on ``Blocks_uniform_cj_warm2_be25``,
    # "soft d-UCB" is nothing but the tail and the bandit can be deleted.
    "Blocks_ducb_cj_warm2_soft_be25_t0": (
        StrategyBlockBandit,
        CJ,
        {**SOFT, "block_evals": 25, "tail_frac": 0.0, **NOB},
        ARCHIVE,
    ),
    "Blocks_ducb_cj_warm2_soft_be25_t10": (
        StrategyBlockBandit,
        CJ,
        {**SOFT, "block_evals": 25, "tail_frac": 0.10, **NOB},
        ARCHIVE,
    ),
    "Blocks_ducb_cj_warm2_soft_be25_t40": (
        StrategyBlockBandit,
        CJ,
        {**SOFT, "block_evals": 25, "tail_frac": 0.40, **NOB},
        ARCHIVE,
    ),
    "Blocks_ducb_cj_warm2_soft_be25_t60": (
        StrategyBlockBandit,
        CJ,
        {**SOFT, "block_evals": 25, "tail_frac": 0.60, **NOB},
        ARCHIVE,
    ),
    # -- §28 (2): block length *under* the tail ----------------------------
    #
    # Uniform peaked at ~50 absolute evaluations, the soft policy at ~25.
    # If what matters is evaluations between relays rather than block length
    # as such, that is exactly the interaction to expect — soft switches less
    # often per block — and this row of the grid measures it at fixed
    # tail_frac = 0.25.
    "Blocks_ducb_cj_warm2_soft_be20": (StrategyBlockBandit, CJ, {**SOFT, "block_evals": 20, **NOB}, ARCHIVE),
    "Blocks_ducb_cj_warm2_soft_be35": (StrategyBlockBandit, CJ, {**SOFT, "block_evals": 35, **NOB}, ARCHIVE),
    "Blocks_ducb_cj_warm2_soft_be50": (StrategyBlockBandit, CJ, {**SOFT, "block_evals": 50, **NOB}, ARCHIVE),
    # The uniform-at-25 reference the ``tail_frac=0`` control has to be read
    # against: same arms, same block length, no bandit at all.
    "Blocks_uniform_cj_warm2_be25": (
        StrategyBlockBandit,
        CJ,
        {"policy": "uniform", "block_evals": 25, **NOB},
        ARCHIVE,
    ),
    # -- §28 (4): the other pair the oracle likes --------------------------
    #
    # ``cl`` here is CMA-ES + **NLSHADE_LBC** (not L-SHADE, unlike the older
    # ``Blocks_ducb_cl_warm2``), now that ``NP_init="auto"`` gives it the
    # 4*dim coefficient.  Same winning configuration as the cj pair, so the
    # only difference is the second arm.
    "Blocks_ducb_cl_warm2_soft_be25": (
        StrategyBlockBandit,
        [warm("cmaes", "archive"), warm("lbc", "archive")],
        {**SOFT, "block_evals": 25, **NOB},
        ARCHIVE,
    ),
    # No ``Phased_cma60_lshade_warm``: ``StrategyPhased`` never calls
    # ``warm_start_now`` at a phase boundary (the §12 defect), and the arm's
    # own ``on_start`` warm path runs at t = 0 against an empty archive.  The
    # spec would therefore be a *cold* hand-off wearing a warm label, which
    # is worse than not measuring it.  Adding the boundary call means editing
    # ``phased.py``, which this screen does not own.
}

#: Every single-arm reference.  §22 killed the idea of one champion — jSO,
#: L-SHADE and CMA-ES sit within 0.014 of each other and win different cells —
#: so "the bar" is the best of the three, computed per run rather than named.
REFS = tuple(n for n in SPECS if n.endswith("_alone"))
#: Specs whose arms share evaluations, in the order the gates prefer them.
WARM = [n for n in SPECS if "_warm" in n]

# --- argv: `key=value` options, then positionals ---------------------------
opts, pos = {}, []
for a in sys.argv[1:]:
    if "=" in a:
        k, _, v = a.partition("=")
        opts[k] = v
    else:
        pos.append(a)

src = opts.get("from")
names = [n for n in opts["specs"].split(",") if n] if opts.get("specs") else list(SPECS)
unknown = [n for n in names if n not in SPECS]
if unknown:
    sys.exit(f"unknown spec(s): {','.join(unknown)}  (known: {','.join(SPECS)})")

#: ``kind=`` picks the regime the screen runs on.  The screen's whole
#: question — does a *sharing* portfolio beat the best single arm? — was
#: answered "level" on the standard battery (§27/§30/§31), and
#: ``planning/GOAL.md`` §2c says the places left to look are noise and
#: dimension.  Same specs, same gates, different regime.
BATTERIES = {
    "standard": make_standard_battery,
    "noisy-gauss": lambda: make_noisy_battery("gauss"),
    "noisy-unif": lambda: make_noisy_battery("unif"),
    "noisy-cauchy": lambda: make_noisy_battery("cauchy"),
    "noisy-gauss-severe": lambda: make_noisy_battery("gauss", level="severe"),
    "highdim": make_highdim_battery,
    "noisy-highdim": make_noisy_highdim_battery,
}

kind = opts.get("kind", "standard")
if kind not in BATTERIES:
    sys.exit(f"unknown kind {kind!r}  (known: {','.join(BATTERIES)})")
battery = BATTERIES[kind]()
if "dims" in opts or "bm" in opts or "insts" in opts:
    battery = dataclasses.replace(
        battery,
        dims=tuple(int(d) for d in opts.get("dims", ",".join(str(d) for d in battery.dims)).split(",")),
        budget_multiplier=int(opts.get("bm", battery.budget_multiplier)),
        instances=tuple(int(i) for i in opts.get("insts", ",".join(str(i) for i in battery.instances)).split(",")),
    )


def spec(name):
    cls, heuristics, kw, analyzers = SPECS[name]
    return dataclasses.replace(
        BASE,
        name=name,
        # One stream for the whole screen: every spec sees the identical
        # (dim, inst, rep) seeds, so the comparison is paired on the stream.
        seed_name="screen",
        strategy_class=cls,
        heuristics=[(c, dict(k)) for c, k in heuristics],
        analyzers=[(c, dict(k)) for c, k in analyzers],
        config_overrides=dict(kw),
    )


if src:
    rows = json.load(open(src))
    seeds = sorted({r["seed"] for r in rows})
    have = {r["s"] for r in rows}
    names = [n for n in names if n in have] if opts.get("specs") else [n for n in SPECS if n in have]
    print(f"read {len(rows)} rows from {src}")
else:
    if len(pos) < 2:
        sys.exit(__doc__)
    out = pos[0]
    seeds = [int(x) for x in pos[1:]]
    specs = [spec(n) for n in names]
    rows, t0 = [], time.perf_counter()
    for seed in seeds:
        r = run_ioh_harness(specs, battery, base_seed=seed, progress=False, sync_eval=True)
        rows += [
            {
                "seed": seed,
                "s": x.strategy_name,
                "dim": x.dim,
                "inst": x.instance,
                "aocc": x.aocc,
                # noisy batteries only: AOCC is scored on the TRUE value
                # above; this is what the optimizer's own observations
                # would have said.  Kept so a re-analysis can see both.
                "obs": x.aocc_observed,
                "evals": x.n_evals,
                "budget": x.budget,
                "err": x.error,
            }
            for x in r.runs
        ]
        json.dump(rows, open(out, "w"))
        print(f"seed {seed} done ({time.perf_counter() - t0:.0f}s)", flush=True)

# --- fold rows into cells: (seed, dim, inst) -> {spec: mean AOCC over reps} --
raw = defaultdict(lambda: defaultdict(list))
errs, short = defaultdict(list), defaultdict(list)
for r in rows:
    if r["s"] not in names:
        continue
    raw[(r["seed"], r["dim"], r["inst"])][r["s"]].append(r["aocc"])
    if r["err"]:
        errs[r["s"]].append(r["err"])
    # A run that stops short of its budget did not spend what it was given
    # -- a stall, an exhausted arm, or the strategy returning no points.
    if r.get("budget") and r.get("evals", 0) < 0.98 * r["budget"]:
        short[r["s"]].append((r["dim"], r["inst"], r["evals"], r["budget"]))
cells = {k: {s: st.mean(v) for s, v in d.items()} for k, d in raw.items()}
dims = sorted({d for _, d, _ in cells})
n = len(seeds)
tc = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}.get(n, 2.26 if n > 8 else 2.5)

if not cells:
    sys.exit("no rows for the selected specs — nothing to compare")


def mean_of(name, dim=None):
    vals = [v[name] for (_, d, _), v in cells.items() if name in v and (dim is None or d == dim)]
    return st.mean(vals) if vals else float("nan")


def paired(a, b, dim=None):
    """Per-seed mean of ``a - b`` over the cells where both have a result."""
    ps = defaultdict(list)
    for (seed, d, _), v in cells.items():
        if a in v and b in v and (dim is None or d == dim):
            ps[seed].append(v[a] - v[b])
    return [st.mean(x) for x in ps.values()]


def ci(ds):
    """``(mean, halfwidth)`` of a 95% t-CI over the per-seed deltas."""
    if len(ds) < 2:
        return (st.mean(ds) if ds else float("nan")), float("nan")
    m = st.mean(ds)
    return m, tc * st.stdev(ds) / len(ds) ** 0.5


def delta(a, b):
    """Mean paired delta of ``a`` over ``b``, or ``nan`` if uncomparable."""
    ds = paired(a, b)
    return st.mean(ds) if ds else float("nan")


setup = (
    f"from {src}" if src else f"{battery.name}, budget {battery.budget_multiplier}*d, insts {list(battery.instances)}"
)
print(f"\n=== portfolio screen ===  ({n} seeds, dims {dims}, {setup})")
print(f"specs: {', '.join(names)}   cells: {len(cells)}")

# (a) means, overall and per dimension.
print(f"\n{'spec':36s} {'mean':>7s} " + "".join(f"  {'d=' + str(d):>8s}" for d in dims))
order = sorted(names, key=lambda s: -mean_of(s))
for s in order:
    per = "".join(f"  {mean_of(s, d):8.4f}" for d in dims)
    tail = f"  errors={len(errs[s])}" if errs[s] else ""
    tail += f"  short={len(short[s])}" if short[s] else ""
    print(f"{s:36s} {mean_of(s):7.4f} " + per + tail)

# (b) paired deltas against each reference, overall CI + per-dimension means.
for ref in REFS:
    if ref not in names:
        continue
    print(f"\ndelta vs {ref} (paired per cell, t-CI over per-seed means)")
    print(
        f"{'spec':36s} {'delta':>8s} {'95% CI':>21s} {'seeds':>7s} " + "".join(f"  {'d=' + str(d):>8s}" for d in dims)
    )
    for s in order:
        if s == ref:
            continue
        ds = paired(s, ref)
        if not ds:
            continue
        m, h = ci(ds)
        band = f"[{m - h:+.4f},{m + h:+.4f}]" if h == h else "        (n<2)"
        flag = " <--" if h == h and (m - h > 0 or m + h < 0) else ""
        per = "".join(f"  {st.mean(paired(s, ref, d) or [float('nan')]):+8.4f}" for d in dims)
        print(f"{s:36s} {m:+8.4f} {band:>21s} {sum(d > 0 for d in ds):3d}/{len(ds):<3d} " + per + flag)

# (c) the §6 screening gates.
#
# The gates are stated over roles, not over spec names, so the same three
# questions survive a change of arm pair (§21 screened CMA-ES + L-SHADE,
# §22 moved to CMA-ES + jSO).  ``best single`` is the best of the ``_alone``
# specs *in this run*, which after §22 is a measured question rather than a
# constant.
best_ref = max((r for r in REFS if r in names), key=mean_of, default=None)
#: (uniform, ducb) pairs on the same arms, most-preferred first.
PAIRS = [
    ("Blocks_uniform_cj", "Blocks_ducb_cj"),
    ("Blocks_uniform_cj_warm2", "Blocks_ducb_cj_warm2"),
    ("Blocks_uniform_2", "Blocks_ducb_2"),
    ("Blocks_uniform_2_warm_any", "Blocks_ducb_2_warm_any"),
]
pair = next(((u, d) for u, d in PAIRS if u in names and d in names), (None, None))
uni, duc = pair
duc = duc or next((d for _, d in PAIRS if d in names), None)
warm_here = [s for s in WARM if s in names]
best_warm = max(warm_here, key=mean_of) if warm_here else None
#: The cold counterpart G4 measures the warm spec against.
cold_ref = next((d for _, d in PAIRS if d in names and "_warm" not in d), None)

print("\n--- screening gates (design §6) ---")
gates = []
if uni and best_ref:
    gates.append(("G1", f"{uni} - best single ({best_ref})", delta(uni, best_ref), -0.02, ">="))
if uni and duc:
    gates.append(("G2", f"{duc} - {uni}", delta(duc, uni), 0.005, ">="))
g3 = cold_ref or duc  # G3 is about the *cold* bandit: no sharing, just learning
if g3 and best_ref:
    gates.append(("G3", f"{g3} - best single ({best_ref})", delta(g3, best_ref), -0.01, ">="))
# G4/G5 test the sharing thesis: a portfolio is only worth its transients if
# the arms hand each other the evaluations they already paid for.  G4 asks
# whether sharing moves the number *at all* beyond the +-0.05 null floor;
# G5 asks the only question that decides the phase — does it rescue the
# portfolio past the best single arm.
if best_warm and cold_ref:
    gates.append(("G4", f"{best_warm} - {cold_ref} (cold)", delta(best_warm, cold_ref), 0.03, ">="))
if best_warm and best_ref:
    gates.append(("G5", f"best warm ({best_warm}) - best single ({best_ref})", delta(best_warm, best_ref), 0.0, ">="))
if not gates:
    print("  (no gate is computable from the selected specs)")
for tag, what, val, thr, _ in gates:
    ok = val == val and val >= thr
    print(f"  {tag} {'PASS' if ok else 'FAIL'}  {what:58s} {val:+.4f}  (need >= {thr:+.3f})")

# (d) what the harness itself reported about the runs.
print("\n--- run health ---")
if not any(errs.values()) and not any(short.values()):
    print("  no errored runs, every run spent its full budget")
for s in names:
    if errs[s]:
        seen = sorted(set(errs[s]))[:3]
        print(f"  {s}: {len(errs[s])} errored run(s); first messages: {seen}")
    if short[s]:
        ex = ", ".join(f"d{d}i{i} {e}/{b}" for d, i, e, b in short[s][:4])
        print(f"  {s}: {len(short[s])} run(s) below budget: {ex}")

print(
    "\nNull floor: on 3 seeds a CMA-ES-containing spec drifts by up to +-0.05 for no\n"
    "reason at all, so a gate verdict inside that band is a direction, not evidence.\n"
    "Only deltas larger than +-0.05 (or a CI that excludes zero, marked `<--`) count;\n"
    "the 12-seed roster of §6 step 3 is what decides."
)
