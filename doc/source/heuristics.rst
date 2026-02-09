Heuristics Module
=================

Point generation algorithms (heuristics):

- **Center**: Generates center point of search space.
- **Zero**: Generates the zero vector point.
- **Random**: Uniform random sampling within the best leaf box.
- **LatinHypercube**: Stratified sampling (Space-filling).
- **Extremal**: Samples from the boundaries of the search space.
- **Nearby**: Gaussian perturbations around current best points.
- **WeightedAverage**: Averages points in the best region.
- **QuadraticWLS**: Fits a weighted least-squares quadratic surrogate model.
- **GaussianProcess**: Uses Gaussian Process models and Expected Improvement with Constraints (EIC).
- **NelderMead**: Simplex optimization method.
- **LBFGSB**: Local optimization using the L-BFGS-B algorithm.
- **FeasibleSearch**: Actively searches for and repairs feasible solutions.
- **ConstraintGradient**: Uses estimated gradients of constraint violations to find feasible regions.
- **LocalPenaltySearch**: Optimizes the scalarized penalty function directly.

.. automodule:: panobbgo.heuristics
   :members:
   :undoc-members:
   :show-inheritance:

