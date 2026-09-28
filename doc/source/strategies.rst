Strategies Module
=================

Available optimization strategies:

- **StrategyBase**: Abstract base class for all strategies
- **StrategyRoundRobin**: Simple round-robin point evaluation
- **StrategyRewarding**: Adaptive heuristic selection based on performance (softmax multi-armed bandit)
- **StrategyUCB**: Upper Confidence Bound (UCB1) algorithm for principled exploration/exploitation
- **StrategyThompsonSampling**: Probabilistic selection using Thompson Sampling (Beta-Bernoulli bandit)
- **StrategyLinUCB**: Contextual bandit using disjoint linear UCB models over budget-progress / success-rate features
- **StrategyPhased**: Budget-phased meta-strategy composing different sub-strategies and heuristic portfolios across phases
- **StrategyBlockBandit**: Hands a whole *block* of evaluations to one arm and scores it by the AOCC area it bought (experimental)

Regime gating in ``StrategyBlockBandit``
----------------------------------------

``StrategyBlockBandit(regime_gate=...)`` switches arms *off* by regime.
The regime is four things — the noise class (``clean`` / ``bounded`` /
``outlier``), the dimension, the budget per dimension and whether the
problem is constrained — looked up, first match wins, in the versioned
:data:`~panobbgo.strategies.blocks.REGIME_TABLE_V1`; every row cites the
12-seed section of ``planning/DISCOVERY_2026-09-09.md`` behind it.  The
rows say: outliers → CMA-ES alone; constrained → jSO alone; bounded noise
at ``d ≤ 5`` and budgets ``≤ 200·dim`` → the CMA-ES + jSO sharing
portfolio; ``d ≥ 10`` and everything else → CMA-ES alone.

``regime_gate="oracle:<class>"`` takes the noise class as given (the
benchmark harness knows it; see :doc:`guide_benchmarking`).
``regime_gate="dim-budget"`` needs no oracle: it applies only the rows
keyed on the dimension and the budget, and only while the parallel
workers do not exceed the kept arms' serial generation size (λ_default for
CMA-ES).  On the current table that is: unconstrained, ``d ≥ 10``,
``≤ 500·dim``, at most λ_default workers → CMA-ES alone; everything else is
left ungated.  The headline harness spec ``Blocks_warm_CMAES_JSO`` uses it
(``planning/DISCOVERY_2026-09-09.md`` §72).  ``"table-v1"``, the in-run
noise probe of ``planning/DESIGN_regime_gating_2026-09-11.md`` §1, raises
``NotImplementedError`` until the oracle has cleared its battery.

``StrategyBlockBandit(first_round_fill=True)`` fills, before the first
result arrives, the workers the owning arm leaves idle: first with the
other arms' queued points, then with one Latin hypercube over the box.  It
only acts under a request cap (the virtual clock's async policy or a real
pull-mode pool) and only while the arms' first generations are smaller than
the worker count, e.g. at ``d = 2``, 200 evaluations, 64 workers (§72).

Two contracts worth knowing.  **Every arm is always constructed**, gate or
no gate, so the run holds the same modules in the same event-bus order
either way (module RNG streams are keyed by name, so building an arm or not
would not shift another module's stream).  The gate is a per-arm *enabled* bit read at one
place, the ``ready`` list of block selection, plus a filtered prologue —
``regime_gate=None`` is byte-identical to a run without the feature, and a
gate that enables every arm is byte-identical to ``None``.  And a disabled
arm is a *paused* arm: it keeps receiving results and topping its queue
up, so "CMA-ES alone via the gate" is deterministic but not *guaranteed*
byte-identical to ``StrategyRoundRobin`` with CMA-ES (measured, it was:
all 276 runs the gate reduced to CMA-ES on the 12-seed cauchy, standard
and d = 10 batteries matched ``CMAES_alone`` bit for bit).  The alternative — a gate that
picks between two pre-built strategies — is not implementable: one
strategy owns the run (the event bus, the analyzers and the evaluator are
built once and ``start`` is published once), so it is not re-proposed.

.. automodule:: panobbgo.strategies
   :members:
   :undoc-members:
   :show-inheritance:

