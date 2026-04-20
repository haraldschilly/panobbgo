Analyzers Module
================

Result analysis and monitoring components:

- **Analyzer**: Base class for analysis components
- **Best**: Tracks best solutions found (feasible best, infeasible best, Pareto front of ``(f, CV)``)
- **Convergence**: Monitors optimization progress; publishes a ``converged`` event using ``std`` or ``improv`` mode
- **Sensitivity**: Estimates per-dimension importance via rank correlation; enables sensitivity-aware perturbations in ``Nearby``
- **Restart**: Detects stagnation (no improvement within ``patience`` evaluations) and triggers multi-start restarts; pairs with CMAES for IPOP-CMA-ES
- **Grid**: Simple spatial grid grouping of nearby points
- **Dedensifyer**: Hierarchical grid that avoids point clustering by keeping only min/max representatives per region
- **Splitter**: Adaptive hierarchical box-decomposition of the search space; publishes ``new_split`` and identifies the best leaf box

.. automodule:: panobbgo.analyzers
   :members:
   :undoc-members:
   :show-inheritance:
