# -*- coding: utf8 -*-
"""Structural invariants every heuristic has to satisfy.

These are *probes*, not benchmarks.  Every real defect found on this code
base so far surfaced by asking a mechanical question about a heuristic --
"is this constructor argument read at all?", "does this run spend its
budget?", "does emitting a whole generation lose points?" -- rather than by
measuring quality (``planning/DISCOVERY_2026-09-09.md`` §5, §16, §18).  The
tests below encode those questions once, parametrised over every heuristic
in :mod:`panobbgo.heuristics`, so a regression of the same shape fails loudly.

Six families:

#. :func:`test_constructor_kwarg_is_read` -- the dead-parameter detector.
   Two seeded solo runs, one at the default and one with the argument
   perturbed; identical trajectories mean the argument changed nothing on
   the path a solo run takes.  This is exactly the ``ipop_factor`` class of
   bug (§16): a knob whose only reader sits behind an event no solo spec
   ever publishes, which then "won" a sweep on pure RNG noise.
#. :func:`test_spends_its_budget` / :func:`test_points_are_finite_and_in_box`
   -- an arm alone must use the evaluations it was given and emit nothing
   that is NaN, infinite or outside the box.
#. :func:`test_beats_random` -- a paired comparison against
   :class:`~panobbgo.heuristics.Random` on three problems and two seeds.
#. :func:`test_generation_survives_a_small_queue` -- the output-queue
   contract of §5: a generation larger than ``config.capacity`` must not be
   silently truncated.
#. :func:`test_construction_advances_master_rng_exactly_once` -- module RNG
   streams are a function of *construction order*; a constructor that draws
   from ``strategy.rng`` beyond its own :meth:`spawn_rng` shifts every later
   module's stream.
#. :func:`test_event_handler_signature_matches_publisher` -- every ``on_*``
   handler must bind the payload its publisher sends.  A mismatch is
   invisible until the rare event fires, and
   :meth:`panobbgo.core.EventBus._dispatch` even swallows the ``TypeError``
   outright for the one-shot ``start`` / ``finished`` events.

Findings that are *pinned rather than fixed* are marked ``xfail`` with the
reason in the marker; see the accompanying findings report.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import pkgutil
from typing import Any, Callable, Dict, FrozenSet, List, Sequence, Set, Tuple

import numpy as np
import pytest

import panobbgo.heuristics as heuristics_module
from panobbgo.core import Heuristic, Module, StrategyBase
from panobbgo.lib.classic import DeJong, Rastrigin, Rosenbrock
from panobbgo.strategies import StrategyRoundRobin

# --------------------------------------------------------------------------
# the heuristic registry
# --------------------------------------------------------------------------

#: Every concrete heuristic exported by :mod:`panobbgo.heuristics`.  Derived
#: from ``__all__`` so a newly added heuristic is covered automatically.
HEURISTICS: Dict[str, type] = {
    name: obj
    for name in heuristics_module.__all__
    for obj in [getattr(heuristics_module, name)]
    if inspect.isclass(obj) and issubclass(obj, Heuristic)
}

#: Constructor arguments without a default, so the class can be built at all.
EXTRA_ARGS: Dict[str, Dict[str, Any]] = {"LatinHypercube": {"div": 8}}

#: Heuristics that emit a fixed, small number of points and then stop.  They
#: cannot spend a budget by construction, so the budget/beats-Random probes
#: do not apply to them.
ONE_SHOT: Dict[str, int] = {"Center": 1, "Zero": 1, "Sobol": 16}

#: Reactive heuristics: they only produce points in reaction to *results*, so
#: alone they emit nothing.  They are run next to a ``Random`` seeder and
#: compared against ``Random`` alone.
NEEDS_PRIMING: Set[str] = {
    "Nearby",
    "WeightedAverage",
    "NelderMead",
    "QuadraticWlsModel",
    "ClaudeHeuristic",
    "RegionUCB",
    "GaussianProcessHeuristic",
}

#: Heuristics that only act on a *constrained* problem; on the unconstrained
#: probe problems they legitimately contribute nothing.
CONSTRAINT_ONLY: Set[str] = {
    "FeasibleSearch",
    "ConstraintGradient",
    "ConstraintRepair",
    "LocalPenaltySearch",
}

#: Heuristics excluded from the run-based probes because a single 300-eval
#: run does not finish in seconds.  Each entry is a finding in its own right
#: -- see :func:`test_slow_heuristic_still_spends_its_budget`.
TOO_SLOW: Dict[str, str] = {
    "GaussianProcessHeuristic": "≈0.4 s per evaluation; a 300-eval solo run takes ~60 s",
    "COBYQA": "the subprocess bridge emits so slowly that the stall guard ends the run first",
}

#: Heuristics that are not search arms and therefore emit nothing alone.
#: That is their documented contract, not a defect; they stay in the static
#: probes (RNG stream, handler signatures).
NOT_A_SEARCH_ARM: Dict[str, str] = {
    "MetaAnalyst": "a trigger-driven helper (panobbgo/heuristics/meta.py), not a search arm: alone it emits nothing",
}

#: ``(heuristic, kwarg) -> reason`` for constructor arguments the
#: dead-parameter detector is *not allowed* to flag, because the code path
#: that reads them is provably out of reach for the probe run: it needs an
#: analyzer the solo run has no reason to add, an event nothing publishes, a
#: sibling argument at a non-default value, or a population large enough for
#: the argument to survive an ``int``/``ceil`` quantisation.  Every entry
#: names that mechanism; anything *not* on this list must change the run.
DEAD_PARAM_ALLOWLIST: Dict[Tuple[str, str], str] = {
    # --- reachable only through the Restart analyzer's `restart` event -----
    ("CMAES", "ipop_factor"): (
        "read only in _restart_ipop, reached only from on_restart, published only by the "
        "Restart analyzer — DISCOVERY_2026-09-09.md §16"
    ),
    ("CMAES", "restart_mode"): "selects between _restart_ipop/_restart_bipop, both reached only from on_restart",
    # --- Hansen termination criteria that never trip on the probe ---------
    ("CMAES", "tolx"): "termination criterion; sigma/D never fall below it within the probe budget",
    ("CMAES", "tolfunhist"): "termination criterion; the fitness history never flattens that far on the probe",
    ("CMAES", "conditioncov"): "termination criterion; the covariance stays far better conditioned than 1e14",
    ("CMAES", "noeffectaxis"): "termination criterion; the probe never reaches a no-effect axis",
    ("CMAES", "noeffectcoord"): "termination criterion; the probe never reaches a no-effect coordinate",
    ("CMAES", "stagnation_rel_tol"): "only read when stagnation_frac is set, and that defaults to None",
    ("CMAES", "sigma_divergence"): (
        "the sigma-divergence restart only fires once sigma exceeds sigma_max_frac of the box range; "
        "on a 3-d Rastrigin sigma only shrinks, so the criterion accepted in DISCOVERY §23 is never "
        "exercised by this probe"
    ),
    ("CMAES", "sigma_max_frac"): "threshold of the sigma-divergence criterion, which never fires here",
    ("CMAES", "sigma_divergence_gens"): "patience of the sigma-divergence criterion, which never fires here",
    # --- warm start needs a non-empty archive, which on_start never has ----
    ("LBFGSB", "warm_start_sigma"): "only read in _warm_start_x0, i.e. only when warm_start=True",
    # --- gated by a sibling argument --------------------------------------
    ("PSO", "k_neighbors"): 'documented as ignored unless topology is "lbest"/"vonneumann"/"random"',
    ("PSO", "stagnation_threshold"): (
        'pso.py:548 returns early unless topology == "random"; under the default "gbest" the '
        "stochastic-K rebuild is unreachable, yet the constructor validates the argument without warning"
    ),
    ("Nearby", "sensitivity_scale"): "scales the per-axis radius from the Sensitivity analyzer's on_new_sensitivity",
    ("Nearby", "quadratic_trust"): "only read when quadratic=True",
    ("Nearby", "quadratic_min_r2"): "only read when quadratic=True",
    ("Nearby", "quadratic_weight_sigma"): "only read when quadratic=True",
    ("Nearby", "quadratic_hessian_rank"): "only read when quadratic=True",
    ("GaussianProcessHeuristic", "enable_eic"): "constraint-aware acquisition; no effect on an unconstrained problem",
    # --- swallowed by integer quantisation at the probe's problem size -----
    ("LSHADE", "p_best_end"): (
        "p_count = ceil(p_eff * NP) (lshade.py:773); at the budget-adaptive NP of a 3-d probe both "
        "0.11 and 0.05 quantise to the same count, so the iLSHADE schedule is a no-op at small NP"
    ),
    ("ClaudeHeuristic", "max_clusters"): (
        "k = min(max_clusters, n_elite // (2*dim)) (claude_heuristic.py:153); the second term binds "
        "for every elite set the probe produces"
    ),
    ("ClaudeHeuristic", "min_points"): (
        "activation threshold, 2*dim+1 = 7 by default; results arrive in batches of 20, so lowering "
        "it to 5 does not move the activation batch"
    ),
    ("LBFGSB", "maxfun"): "cap on the worker's evaluations per descent; a 3-d descent never reaches 25",
    ("Nearby", "cap"): (
        "cap only sizes the output queue, and Heuristic._put grows it on demand since the §5 fix, so "
        "for a heuristic that emits via on_new_best (rather than fill_queue) it has no effect at all"
    ),
}

#: Allowlist entries that hold for *every* heuristic carrying the argument.
DEAD_PARAM_ALLOWLIST_BY_NAME: Dict[str, str] = {
    "warm_start": (
        "the shared archive is empty at on_start, so every mode falls back to the cold path; "
        "the real contract is pinned by test_warm_start_with_empty_archive_is_the_cold_path"
    ),
}

#: Non-numeric / sentinel constructor defaults, and the value the detector
#: perturbs them to.  ``None`` here means "leave this argument alone".
PERTURBATIONS: Dict[Tuple[str, str], Any] = {
    ("CMAES", "restart_mode"): "bipop",
    ("CMAES", "restart_from"): "best",
    ("CMAES", "popsize"): 12,
    ("CMAES", "stagnation"): 4,
    ("CMAES", "stagnation_frac"): 0.1,
    ("CMAES", "warm_start"): "archive",
    ("PSO", "topology"): "lbest",
    ("PSO", "w_end"): 0.4,
    ("PSO", "stagnation_threshold"): 4,
    ("PSO", "warm_start"): "archive",
    ("Nearby", "axes"): "all",
    ("Nearby", "quadratic_min_r2"): 0.4,
    ("Nearby", "quadratic_hessian_rank"): None,
    ("LocalPenaltySearch", "method"): "Powell",
    ("LBFGSB", "warm_start"): True,
    ("LBFGSB", "max_starts"): 3,
    ("LBFGSB", "maxfun"): 25,
    ("LBFGSB", "epsilon"): 1e-4,
    ("COBYQA", "initial_tr_radius"): 0.5,
    ("COBYQA", "maxfev"): 40,
    ("Extremal", "prob"): 0.5,
    ("ClaudeHeuristic", "min_points"): 5,
    ("LSHADE", "p_best_end"): 0.05,
    ("LSHADE", "F_schedule"): "jso",
    ("NLSHADE_LBC", "lbc_regime"): "aggressive",
    ("NLSHADE_LBC", "p_F_init"): 2.0,
    ("NLSHADE_LBC", "p_F_final"): 2.5,
    ("NLSHADE_LBC", "p_CR_init"): 2.0,
    ("NLSHADE_LBC", "p_CR_final"): 2.5,
    ("NLSHADE_LBC", "m_lbc"): 1.0,
}

#: Perturbations keyed by argument name alone, applied to every heuristic
#: that has the argument.
PERTURBATIONS_BY_NAME: Dict[str, Any] = {
    "NP_init": 12,
    "warm_start": "archive",
    "seed": 20260910,
}

#: Constructor arguments that carry no behaviour and are skipped outright.
IGNORED_KWARGS: Set[str] = {"name"}

#: Population heuristics and the argument that sets their generation size,
#: for the output-queue contract of §5.
POPULATION_SIZE_ARG: Dict[str, str] = {
    "DifferentialEvolution": "NP",
    "LSHADE": "NP_init",
    "LSHADE_EpSin": "NP_init",
    "JSO": "NP_init",
    "NLSHADE_RSP": "NP_init",
    "NLSHADE_LBC": "NP_init",
    "PSO": "NP",
    "CMAES": "popsize",
    "LatinHypercube": "div",
    "Sobol": "n",
}

#: ``max_stall_seconds`` for the probe runs.  The default 30 s turns every
#: starved run into a half-minute wait.
STALL = 3.0

#: ``heuristic -> (problem kind, evaluations)`` for the dead-parameter probe.
#: The default is a cheap, easy problem; a heuristic whose interesting code
#: paths (restarts, adaptive schedules) only open later gets a longer, harder
#: one, so "the argument is dead" is not just "the budget was too short".
#: CMA-ES needs ~1500 evaluations on a multimodal problem before its
#: self-restart machinery fires at all.
PROBE_SETTINGS: Dict[str, Tuple[str, int]] = {"CMAES": ("rastrigin", 1500)}
DEFAULT_PROBE: Tuple[str, int] = ("dejong", 150)


def _runnable(name: str) -> bool:
    """Can ``name`` be exercised by a short run-based probe at all?"""
    return not (name in ONE_SHOT or name in CONSTRAINT_ONLY or name in TOO_SLOW or name in NOT_A_SEARCH_ARM)


RUNNABLE = sorted(n for n in HEURISTICS if _runnable(n))


# --------------------------------------------------------------------------
# running one deterministic probe
# --------------------------------------------------------------------------

Trajectory = Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]


def _make(name: str, **kwargs: Any) -> Callable[[StrategyBase], Heuristic]:
    """Factory for heuristic ``name``, defaults plus ``kwargs``."""
    cls = HEURISTICS[name]
    args = dict(EXTRA_ARGS.get(name, {}))
    args.update(kwargs)
    return lambda strategy: cls(strategy, **args)


def _strategy(problem: Any, seed: int, max_eval: int, capacity: int | None = None) -> StrategyBase:
    strategy = StrategyRoundRobin(problem, parse_args=False, testing_mode=True, seed=seed)
    cfg = strategy.config
    cfg.max_eval = max_eval
    cfg.sync_evaluation = True
    cfg.stop_on_convergence = False
    cfg.ui_show = False
    cfg.evaluation_method = "threaded"
    cfg.max_stall_seconds = STALL
    if capacity is not None:
        cfg.capacity = capacity
    return strategy


def _trajectory(strategy: StrategyBase) -> Trajectory:
    """``(x, fx, who)`` of a finished run, as plain arrays."""
    df = strategy.results.results
    if df is None or len(df) == 0:
        return np.zeros((0, strategy.problem.dim)), np.zeros(0), ()
    x = np.asarray([np.asarray(v, dtype=float) for v in df["x"].to_numpy()])
    fx = df["fx"].to_numpy(dtype=float).ravel()
    who = tuple(str(w) for w in df["who"].to_numpy().ravel())
    return x, fx, who


def run(
    factories: Sequence[Callable[[StrategyBase], Heuristic]],
    problem: Any,
    seed: int = 42,
    max_eval: int = 150,
    capacity: int | None = None,
) -> Trajectory:
    """One seeded, synchronous :class:`StrategyRoundRobin` run."""
    strategy = _strategy(problem, seed=seed, max_eval=max_eval, capacity=capacity)
    for factory in factories:
        strategy.add_heuristic(factory(strategy))
    try:
        strategy.start()
    except Exception:
        # ``StrategyBase.start`` only cleans up after a KeyboardInterrupt, so
        # anything else would leak the event-bus and evaluator threads.
        strategy._cleanup()
        raise
    return _trajectory(strategy)


def solo(name: str, problem: Any, seed: int = 42, max_eval: int = 150, **kwargs: Any) -> Trajectory:
    """Run ``name`` on its own -- next to a ``Random`` seeder if it is reactive."""
    factories: List[Callable[[StrategyBase], Heuristic]] = []
    if name in NEEDS_PRIMING:
        factories.append(_make("Random"))
    factories.append(_make(name, **kwargs))
    return run(factories, problem, seed=seed, max_eval=max_eval)


def own_points(name: str, who: Sequence[str]) -> int:
    """How many results the heuristic under test actually produced."""
    if name in NEEDS_PRIMING:
        return sum(1 for w in who if not w.startswith("Random"))
    return len(who)


def make_problem(kind: str, dim: int = 3) -> Any:
    """One of the three cheap probe problems, at ``dim`` dimensions."""
    if kind == "dejong":
        return DeJong(dims=dim)
    if kind == "rastrigin":
        return Rastrigin(dims=dim)
    if kind == "rosenbrock":
        return Rosenbrock(dim=dim)
    raise ValueError(kind)


def probe_problem(dim: int = 3) -> Any:
    return DeJong(dims=dim)


# --------------------------------------------------------------------------
# 1. dead-parameter detector
# --------------------------------------------------------------------------


def _perturb(name: str, param: inspect.Parameter) -> Any:
    """The value ``param`` is moved to, or :data:`_SKIP` if we cannot.

    Numbers double (or become 1 when the default is 0), bools flip, and
    everything else -- strings, enums, ``None`` defaults, module-private
    sentinels -- comes from the explicit tables above.
    """
    key = (name, param.name)
    if key in PERTURBATIONS:
        return PERTURBATIONS[key]
    if param.name in PERTURBATIONS_BY_NAME:
        return PERTURBATIONS_BY_NAME[param.name]
    default = param.default
    if isinstance(default, bool):
        return not default
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        return type(default)(default * 2) if default else type(default)(1)
    return _SKIP


class _Skip:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<no perturbation>"


_SKIP = _Skip()


def _kwargs_of(name: str) -> List[inspect.Parameter]:
    """Constructor arguments of ``name`` that the detector should probe."""
    signature = inspect.signature(HEURISTICS[name].__init__)
    out = []
    for param in list(signature.parameters.values())[2:]:  # skip self, strategy
        if param.name in IGNORED_KWARGS or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty:
            continue  # required: covered by EXTRA_ARGS, cannot be "dead"
        if isinstance(_perturb(name, param), _Skip):
            continue
        out.append(param)
    return out


DEAD_PARAM_CASES = [(name, param.name) for name in RUNNABLE for param in _kwargs_of(name)]


@pytest.fixture(scope="module")
def default_trajectories() -> Dict[str, Trajectory]:
    """Cache of the at-default run per heuristic (one run, many kwargs)."""
    return {}


@pytest.mark.parametrize("name,kwarg", DEAD_PARAM_CASES, ids=["%s.%s" % c for c in DEAD_PARAM_CASES])
def test_constructor_kwarg_is_read(name: str, kwarg: str, default_trajectories: Dict[str, Trajectory]) -> None:
    """Perturbing a constructor argument must change the run.

    A kwarg that leaves the ``(x, fx, who)`` trajectory bit-identical is
    either dead code or reachable only through a path a solo run never
    takes; both are reported rather than tolerated, because a sweep over
    such a knob measures nothing but RNG noise
    (``planning/DISCOVERY_2026-09-09.md`` §18).
    """
    reason = DEAD_PARAM_ALLOWLIST.get((name, kwarg)) or DEAD_PARAM_ALLOWLIST_BY_NAME.get(kwarg)
    if reason is not None:
        pytest.skip("%s.%s is allowlisted: %s" % (name, kwarg, reason))
    kind, budget = PROBE_SETTINGS.get(name, DEFAULT_PROBE)
    problem = make_problem(kind)
    if name not in default_trajectories:
        default_trajectories[name] = solo(name, problem, max_eval=budget)
    _, fx_default, who_default = default_trajectories[name]
    assert own_points(name, who_default) > 0, "%s produced no points at all -- probe is blind" % name

    param = inspect.signature(HEURISTICS[name].__init__).parameters[kwarg]
    _, fx_perturbed, who_perturbed = solo(name, problem, max_eval=budget, **{kwarg: _perturb(name, param)})

    same = (
        len(fx_default) == len(fx_perturbed)
        and np.array_equal(fx_default, fx_perturbed)
        and who_default == who_perturbed
    )
    assert not same, (
        "%s(%s=%r) produced a bit-identical trajectory to the default %r — the argument is "
        "not read on any path a solo run takes (the `ipop_factor` failure mode)."
        % (name, kwarg, _perturb(name, param), param.default)
    )


WARM_START_CLASSES = sorted(
    n for n in HEURISTICS if "warm_start" in inspect.signature(HEURISTICS[n].__init__).parameters
)


@pytest.mark.parametrize("name", WARM_START_CLASSES)
def test_warm_start_none_equals_omitted(name: str) -> None:
    """``warm_start=None`` must be *exactly* the default (cold start).

    The reproducibility contract of ``DESIGN_warm_start_2026-09-10.md`` §2:
    passing the documented "off" value may not consume a different amount of
    randomness than not passing it at all.
    """
    if name in TOO_SLOW:
        pytest.skip(TOO_SLOW[name])
    problem = probe_problem()
    default = inspect.signature(HEURISTICS[name].__init__).parameters["warm_start"].default
    if default is not None:
        pytest.skip("%s.warm_start defaults to %r, not None" % (name, default))
    _, fx_omitted, who_omitted = solo(name, problem)
    _, fx_explicit, who_explicit = solo(name, problem, warm_start=None)
    np.testing.assert_array_equal(fx_omitted, fx_explicit)
    assert who_omitted == who_explicit


#: The classes whose cold path an *empty* archive does not restore.
COLD_PATH_XFAIL: Dict[str, str] = {
    "NLSHADE_RSP": (
        "LSHADE._warm_start_population calls self._archive_cap() at lshade.py:939 — one line "
        "*before* the `if not pool: return False` bail-out at lshade.py:942.  In NLSHADE_RSP that "
        "override draws from self._rng (nl_shade_rsp.py:231, adaptive_archive=True by default), so "
        "merely setting warm_start= consumes an RNG draw even when the warm start is abandoned and "
        "shifts the entire initial population.  Every paired 'warm vs cold' A/B on these two arms "
        "therefore compares two different RNG streams — the DISCOVERY §18 failure mode"
    ),
    "NLSHADE_LBC": "same as NLSHADE_RSP, which it subclasses",
}


@pytest.mark.parametrize("name", WARM_START_CLASSES)
def test_warm_start_with_empty_archive_is_the_cold_path(name: str) -> None:
    """An archive with nothing in it must leave the run bit-identical.

    ``LSHADE.on_start`` promises "``warm_start=None`` — or an archive that
    has nothing to give — this is the cold start, statement for statement as
    before" (lshade.py:978).  Nothing has been evaluated when ``on_start``
    fires, so *every* warm-start mode has to hit that fallback.  A heuristic
    that nevertheless moves makes the warm/cold comparison meaningless.
    """
    if name in TOO_SLOW:
        pytest.skip(TOO_SLOW[name])
    default = inspect.signature(HEURISTICS[name].__init__).parameters["warm_start"].default
    if default is not None:
        pytest.skip(
            "%s.warm_start is a %s (default %r), not an archive selector: it switches which "
            "generator produces x0, so it legitimately moves the run even without an incumbent"
            % (name, type(default).__name__, default)
        )
    if name in COLD_PATH_XFAIL:
        pytest.xfail(COLD_PATH_XFAIL[name])
    mode = PERTURBATIONS.get((name, "warm_start"), PERTURBATIONS_BY_NAME["warm_start"])
    problem = probe_problem()
    _, fx_cold, who_cold = solo(name, problem)
    _, fx_warm, who_warm = solo(name, problem, warm_start=mode)
    np.testing.assert_array_equal(fx_cold, fx_warm)
    assert who_cold == who_warm


# --------------------------------------------------------------------------
# 2. budget, finiteness, box
# --------------------------------------------------------------------------

BUDGET = 300


@pytest.mark.parametrize("name", RUNNABLE)
def test_spends_its_budget(name: str) -> None:
    """A heuristic alone must use every evaluation it was given.

    Stopping short means the strategy's stall guard fired: the heuristic
    starved the main loop.  Batch heuristics may overshoot by one
    generation, never undershoot.
    """
    _, fx, who = solo(name, probe_problem(), max_eval=BUDGET)
    assert own_points(name, who) > 0, "%s emitted nothing" % name
    assert len(fx) >= BUDGET, "%s stopped after %d of %d evaluations" % (name, len(fx), BUDGET)


@pytest.mark.parametrize("name", RUNNABLE)
def test_points_are_finite_and_in_box(name: str) -> None:
    """No NaN/inf value, and every evaluated point inside the box.

    ``Heuristic.emit`` projects, so a point outside the box means something
    bypassed ``emit``.
    """
    x, fx, _ = solo(name, probe_problem(), max_eval=BUDGET)
    assert np.all(np.isfinite(fx)), "%s produced %d non-finite values" % (name, int(np.sum(~np.isfinite(fx))))
    assert np.all(np.isfinite(x)), "%s produced non-finite coordinates" % name
    box = probe_problem().box
    assert np.all(x >= box[:, 0] - 1e-9), "%s emitted a point below the box" % name
    assert np.all(x <= box[:, 1] + 1e-9), "%s emitted a point above the box" % name


# --------------------------------------------------------------------------
# 3. beats Random
# --------------------------------------------------------------------------

PAIRED_CELLS: List[Tuple[str, int]] = [(p, s) for p in ("dejong", "rastrigin", "rosenbrock") for s in (42, 7)]


@pytest.fixture(scope="module")
def random_baseline() -> Dict[Tuple[str, int], float]:
    """Best ``fx`` of a solo ``Random`` run per (problem, seed) cell."""
    out: Dict[Tuple[str, int], float] = {}
    for kind, seed in PAIRED_CELLS:
        _, fx, _ = run([_make("Random")], make_problem(kind), seed=seed, max_eval=BUDGET)
        out[(kind, seed)] = float(np.min(fx))
    return out


#: Arms that lose this paired comparison.  Pinned, not weakened: a search
#: heuristic that cannot beat uniform sampling on DeJong / Rastrigin /
#: Rosenbrock at dim 3 is a finding about the heuristic.
BEATS_RANDOM_XFAIL: Dict[str, str] = {
    "Extremal": "a box-corner sampler, not an optimizer — it never converges anywhere",
    "LatinHypercube": "a space-filling design; it stratifies the box but never exploits",
    "DifferentialEvolution": (
        "the plain DE/rand/1/bin arm with its default NP=20 loses to uniform sampling at 300 "
        "evaluations on all three probe problems — the population never gets enough generations"
    ),
    "WeightedAverage": "convex combinations of existing results cannot leave their convex hull",
    "NelderMead": "the simplex restarts from scratch too often to accumulate progress in 300 evaluations",
    "RegionUCB": "the region bandit spends its budget exploring boxes instead of refining the incumbent",
}


@pytest.mark.parametrize("name", [n for n in RUNNABLE if n != "Random"])
def test_beats_random(name: str, random_baseline: Dict[Tuple[str, int], float]) -> None:
    """An arm must beat uniform sampling on a majority of the paired cells.

    Three problems × two seeds, same budget, best ``fx`` compared.  A
    heuristic that cannot clear this floor is not a search strategy; the
    failure is a finding, not a reason to weaken the threshold.
    """
    if name in BEATS_RANDOM_XFAIL:
        pytest.xfail(BEATS_RANDOM_XFAIL[name])
    wins = 0
    losses = []
    for kind, seed in PAIRED_CELLS:
        _, fx, _ = solo(name, make_problem(kind), seed=seed, max_eval=BUDGET)
        mine = float(np.min(fx))
        theirs = random_baseline[(kind, seed)]
        if mine < theirs:
            wins += 1
        else:
            losses.append("%s/%d: %.6g vs Random %.6g" % (kind, seed, mine, theirs))
    assert wins >= 4, "%s beat Random in only %d of %d cells (%s)" % (name, wins, len(PAIRED_CELLS), "; ".join(losses))


# --------------------------------------------------------------------------
# 4. the output-queue contract
# --------------------------------------------------------------------------

GENERATION = 30
TINY_CAPACITY = 5


@pytest.mark.parametrize("name", sorted(POPULATION_SIZE_ARG))
def test_generation_survives_a_small_queue(name: str) -> None:
    """A generation bigger than ``config.capacity`` must not be truncated.

    Until 2026-09 ``Heuristic.emit`` did ``put_nowait`` on a
    ``Queue(config.capacity)`` (default 20) and swallowed ``Full``, so the
    whole DE family ran a 20-member population no matter what ``NP_init``
    said (``planning/DISCOVERY_2026-09-09.md`` §5).
    """
    arg = POPULATION_SIZE_ARG[name]
    strategy = _strategy(probe_problem(), seed=42, max_eval=BUDGET, capacity=TINY_CAPACITY)
    heuristic = _make(name, **{arg: GENERATION})(strategy)
    strategy.add_heuristic(heuristic)
    try:
        strategy.initialize()  # publishes ``start`` and waits for the bus
        queued = heuristic._output.qsize()
    finally:
        strategy._cleanup()
    assert queued >= GENERATION, "%s(%s=%d) kept only %d points at capacity=%d — the rest was dropped" % (
        name,
        arg,
        GENERATION,
        queued,
        TINY_CAPACITY,
    )


# --------------------------------------------------------------------------
# 5. the constructor must not disturb the master RNG stream
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(HEURISTICS))
def test_construction_advances_master_rng_exactly_once(name: str) -> None:
    """Building a heuristic may consume exactly one :meth:`spawn_rng` draw.

    Module streams are derived from ``strategy.rng`` in *construction
    order*; a constructor that draws again from the master generator shifts
    every module built after it, so an unrelated heuristic's run changes
    when this one gains a feature.  Randomness a constructor needs belongs
    to its own :attr:`Module.rng`.
    """
    reference = _strategy(probe_problem(), seed=42, max_eval=10)
    probe = _strategy(probe_problem(), seed=42, max_eval=10)
    try:
        reference.spawn_rng()
        _make(name)(probe)
        assert probe.rng.bit_generator.state == reference.rng.bit_generator.state, (
            "%s.__init__ drew from strategy.rng beyond its own spawn_rng(); every module "
            "constructed after it would get a different stream" % name
        )
    finally:
        reference._cleanup()
        probe._cleanup()


# --------------------------------------------------------------------------
# 6. event handler hygiene
# --------------------------------------------------------------------------

PACKAGE_ROOT = pathlib.Path(inspect.getfile(StrategyBase)).parent


def _published_payloads() -> Dict[str, Set[FrozenSet[str]]]:
    """``event key -> {payload kwarg sets}`` for every literal ``publish(...)``."""
    payloads: Dict[str, Set[FrozenSet[str]]] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "publish":
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            key = node.args[0].value
            names = frozenset(kw.arg for kw in node.keywords if kw.arg and kw.arg not in ("terminate", "event"))
            payloads.setdefault(str(key), set()).add(names)
    return payloads


def _event_aware_classes() -> Dict[type, None]:
    """Every :class:`Module` / strategy / constraint-handler class in the package."""
    found: Dict[type, None] = {}
    for info in pkgutil.walk_packages([str(PACKAGE_ROOT)], "panobbgo."):
        try:
            module = importlib.import_module(info.name)
        except Exception:  # optional backends (dask, ioh, ...) may be absent
            continue
        for obj in vars(module).values():
            if not inspect.isclass(obj) or not getattr(obj, "__module__", "").startswith("panobbgo"):
                continue
            if issubclass(obj, (Module, StrategyBase)) or "ConstraintHandler" in obj.__name__:
                found[obj] = None
    return found


HANDLER_CASES = [
    (cls, attr)
    for cls in _event_aware_classes()
    for attr, value in sorted(vars(cls).items())
    if attr.startswith("on_") and inspect.isfunction(value)
]


@pytest.mark.parametrize(
    "cls,attr", HANDLER_CASES, ids=["%s.%s.%s" % (c.__module__.split(".")[-1], c.__name__, a) for c, a in HANDLER_CASES]
)
def test_event_handler_signature_matches_publisher(cls: type, attr: str) -> None:
    """Every ``on_<key>`` must bind the payload every ``publish("<key>", ...)`` sends.

    :meth:`panobbgo.core.EventBus._dispatch` calls handlers as
    ``on_<key>(**event._kwargs)``.  A signature mismatch therefore raises a
    ``TypeError`` only when that event actually fires -- and for the one-shot
    ``start`` / ``finished`` events the dispatcher swallows the ``TypeError``
    silently, so the handler simply never runs.  This is a static check:
    no run is needed to see the mismatch.
    """
    payloads = _published_payloads()
    key = attr[3:]
    assert key in payloads, (
        "%s.%s listens to '%s', but nothing in the package publishes it — either a typo "
        "or dead code (the EventBus subscribes to it regardless)" % (cls.__name__, attr, key)
    )
    handler = inspect.Signature(list(inspect.signature(getattr(cls, attr)).parameters.values())[1:])
    for payload in sorted(payloads[key], key=sorted):
        try:
            handler.bind(**dict.fromkeys(payload))
        except TypeError as exc:
            pytest.fail(
                "%s.%s.%s%s cannot accept publish('%s', %s): %s"
                % (cls.__module__, cls.__name__, attr, handler, key, sorted(payload), exc)
            )


# --------------------------------------------------------------------------
# pinned findings
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            "Center",
            marks=pytest.mark.xfail(
                raises=ZeroDivisionError,
                strict=True,
                reason=(
                    "StrategyRoundRobin.execute does `% len(hs)` on the *active* heuristics; when the "
                    "last arm goes inactive the run dies with ZeroDivisionError instead of stopping, "
                    "and StrategyBase.start() skips _cleanup() so the threads leak "
                    "(panobbgo/strategies/round_robin.py:41)"
                ),
            ),
        ),
        pytest.param("Zero", marks=pytest.mark.xfail(raises=ZeroDivisionError, strict=True, reason="see Center")),
        pytest.param("Sobol", marks=pytest.mark.xfail(raises=ZeroDivisionError, strict=True, reason="see Center")),
    ],
)
def test_exhausted_heuristic_ends_the_run_cleanly(name: str) -> None:
    """A run whose only heuristic is finished must terminate, not crash."""
    _, fx, _ = solo(name, probe_problem(), max_eval=BUDGET)
    assert len(fx) == ONE_SHOT[name]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "subprocess-bridge heuristics (LBFGSB, COBYQA) emit from a pump thread, so they never have "
        "a point ready when StrategyRoundRobin polls them; any competitor that does starves them "
        "completely (panobbgo/heuristics/lbfgsb.py:419)"
    ),
)
def test_subprocess_bridge_contributes_next_to_a_competitor() -> None:
    """``LBFGSB`` alone spends the whole budget; next to ``Random`` it emits nothing."""
    _, _, who = run([_make("Random"), _make("LBFGSB")], probe_problem(), max_eval=150)
    assert sum(1 for w in who if w.startswith("LBFGSB")) > 0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "COBYQA's subprocess bridge emits so slowly that the wall-clock stall guard "
        "(panobbgo/core.py:1739) ends the run before the budget is spent — which also makes a "
        "seeded, sync_evaluation=True run depend on machine speed, breaking the reproducibility "
        "contract of tests/test_reproducibility.py.  GaussianProcessHeuristic shows the same "
        "shape at ~0.4 s/evaluation but is too slow to pin here; see the findings report."
    ),
)
def test_slow_heuristic_still_spends_its_budget() -> None:
    """Pinned: a slow arm loses part of its budget to the wall-clock stall guard."""
    _, fx, _ = solo("COBYQA", probe_problem(), max_eval=BUDGET)
    assert len(fx) >= BUDGET


@pytest.mark.parametrize("name", sorted(CONSTRAINT_ONLY))
def test_constraint_heuristic_is_inert_when_unconstrained(name: str) -> None:
    """The constraint-only arms emit nothing on an unconstrained problem.

    Documented on purpose: it means a portfolio that carries them on an
    unconstrained battery pays a scheduler slot for a heuristic that can
    never produce a point.
    """
    _, fx, who = run([_make("Random"), _make(name)], probe_problem(), max_eval=150)
    assert len(fx) >= 150
    assert sum(1 for w in who if not w.startswith("Random")) == 0
