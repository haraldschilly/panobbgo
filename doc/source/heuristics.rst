Heuristics Module
=================

Point generation algorithms (heuristics):

- **Center**: Generates center point of search space.
- **Zero**: Generates the zero vector point.
- **Random**: Uniform random sampling within the best leaf box.
- **LatinHypercube**: Stratified sampling (Space-filling).
- **Extremal**: Samples from the boundaries of the search space.
- **Nearby**: Gaussian perturbations around current best points (sensitivity-aware when paired with the Sensitivity analyzer).
- **WeightedAverage**: Averages points in the best region.
- **QuadraticWlsModel**: Fits a weighted least-squares quadratic surrogate model.
- **GaussianProcessHeuristic**: Gaussian Process surrogate with EI / UCB / PI acquisition functions; uses Expected Improvement with Constraints (EIC) when constraints are active.
- **NelderMead**: Simplex optimization method.
- **LBFGSB**: Local optimization using the L-BFGS-B algorithm.
- **CMAES**: Covariance Matrix Adaptation Evolution Strategy with IPOP-CMA-ES restart support (paired with the Restart analyzer) — gold standard for derivative-free continuous optimization, excellent on ill-conditioned/ridge-following problems like Rosenbrock.
- **DifferentialEvolution**: DE mutation/crossover/selection applied to the accumulated result database — strong on multimodal landscapes (Rastrigin, Schwefel).
- **FeasibleSearch**: Actively searches for and repairs feasible solutions via line search towards feasibility.
- **ConstraintGradient**: Uses estimated gradients of constraint violations to find feasible regions.
- **LocalPenaltySearch**: Optimizes the scalarized penalty function directly with a local scipy optimizer.
- **ConstraintRepair**: Projects infeasible best points onto the feasible region using SLSQP.
- **ClaudeHeuristic**: Cluster-based adaptive search using Mixture of Gaussians over elite points.

.. automodule:: panobbgo.heuristics
   :members:
   :undoc-members:
   :show-inheritance:

