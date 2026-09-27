Analyzers Module
================

Result analysis and monitoring components:

- **Analyzer**: Base class for analysis components
- **Best**: Tracks best solutions found (feasible best, infeasible best, Pareto front of ``(f, CV)``)
- **Convergence**: Monitors optimization progress; publishes a ``converged`` event using ``std`` or ``improv`` mode
- **Sensitivity**: Estimates per-dimension importance via rank correlation; enables sensitivity-aware perturbations in ``Nearby``
- **Restart**: Detects stagnation (no improvement within ``patience`` evaluations) and triggers multi-start restarts; pairs with CMAES for IPOP-CMA-ES
- **Splitter**: Adaptive hierarchical box-decomposition of the search space; publishes ``new_split`` and identifies the best leaf box.  Resolution scales with the evaluation budget (``leaf_size`` / ``min_leaf_size`` / ``max_leaves``); ``split_rule`` and ``cut_rule`` select the cut, ``legacy=True`` restores the fixed-resolution tree
- **Archive**: Opt-in bounded top-K of the *shared* result stream, ranked by penalty and unfiltered by ``who``; the query layer heuristics warm-start from
- **FailureModel**: Opt-in shared model of failure regions ("poison zones": crashes, ``NaN``, timeouts) — ``p_fail(x)`` / ``in_poison(x)`` from a kernel classifier with an adaptive bandwidth and a success prior; with ``filter=True`` the strategy answers candidates in a zone as failed without evaluating them (no budget), so each arm's own failure handling applies.  ``CMAES(failure_aware=True)`` and ``TrustRegionQuadratic(failure_aware=True)`` add arm-specific handling (DISCOVERY §71)

.. automodule:: panobbgo.analyzers
   :members:
   :undoc-members:
   :show-inheritance:
