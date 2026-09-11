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

Only the oracle form exists so far: ``regime_gate="oracle:<class>"`` takes
the noise class as given (the benchmark harness knows it; see
:doc:`guide_benchmarking`).  ``"table-v1"``, the in-run noise probe of
``planning/DESIGN_regime_gating_2026-09-11.md`` §1, raises
``NotImplementedError`` until the oracle has cleared its battery.

Two contracts worth knowing.  **Every arm is always constructed**, gate or
no gate: ``StrategyBase`` spawns each module's RNG stream in construction
order, so building an arm lazily would shift every later stream and change
the run bit-for-bit.  The gate is a per-arm *enabled* bit read at one
place, the ``ready`` list of block selection, plus a filtered prologue —
``regime_gate=None`` is byte-identical to a run without the feature, and a
gate that enables every arm is byte-identical to ``None``.  And a disabled
arm is a *paused* arm: it keeps receiving results and topping its queue
up, so "CMA-ES alone via the gate" is deterministic but not byte-identical
to ``StrategyRoundRobin`` with CMA-ES.  The alternative — a gate that
picks between two pre-built strategies — is not implementable: one
strategy owns the run (the event bus, the analyzers and the evaluator are
built once and ``start`` is published once), so it is not re-proposed.

.. automodule:: panobbgo.strategies
   :members:
   :undoc-members:
   :show-inheritance:

