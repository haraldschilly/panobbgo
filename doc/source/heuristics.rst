Heuristics Module
=================

Point generation algorithms (heuristics):

- **Center**: Generates center point of search space.
- **Zero**: Generates the zero vector point.
- **Random**: Uniform random sampling within the best leaf box.
- **LatinHypercube**: Stratified sampling (Space-filling).
- **Sobol**: Low-discrepancy quasi-random sequence (Sobol') for one-shot space-filling initial designs — better uniformity than random or Latin Hypercube at the same sample count, scrambled for per-rep variance.
- **Extremal**: Samples from the boundaries of the search space.
- **Nearby**: Gaussian perturbations around current best points (sensitivity-aware when paired with the Sensitivity analyzer).
- **WeightedAverage**: Averages points in the best region.
- **QuadraticWlsModel**: Fits a weighted least-squares quadratic surrogate model.
- **GaussianProcessHeuristic**: Gaussian Process surrogate with EI / UCB / PI acquisition functions; uses Expected Improvement with Constraints (EIC) when constraints are active.
- **NelderMead**: Simplex optimization method.
- **LBFGSB**: Local optimization using the L-BFGS-B algorithm.
- **CMAES**: Covariance Matrix Adaptation Evolution Strategy with IPOP-CMA-ES restart support (paired with the Restart analyzer) — gold standard for derivative-free continuous optimization, excellent on ill-conditioned/ridge-following problems like Rosenbrock.
- **DifferentialEvolution**: Basic ``DE/rand/1/bin`` (fixed ``F = 0.8``, ``CR = 0.9``) — the canonical recombination-based mutation/crossover/selection loop, strong on multimodal landscapes (Rastrigin, Schwefel) when budget is generous.
- **LSHADE**: Linear Population Size Reduction Success-History Adaptive DE (Tanabe & Fukunaga, CEC 2014) — ``DE/current-to-pbest/1`` mutation with an external archive (JADE; Zhang–Sanderson 2009), per-trial ``(F, CR)`` sampled from a success-history memory of size ``H`` and updated via the weighted Lehmer mean of successful trials, plus a population that **shrinks linearly** from ``NP_init`` to ``NP_min`` as the evaluation budget is consumed.  L-SHADE was the CEC-2014 winner and remains one of the strongest single-population black-box solvers in the literature; complementary to the basic ``DifferentialEvolution`` (which it does **not** replace) — DE/rand/1 with fixed parameters wins under generous budgets where the population can cover the space, L-SHADE wins under the kind of budget-constrained, expensive-evaluation regime Panobbgo targets.
- **PSO**: Particle Swarm Optimization with the canonical Clerc–Kennedy (2002) constriction-coefficient parameters — a swarm of particles tracks personal and global bests with momentum, providing exploration dynamics distinct from CMA-ES (covariance) and DE (recombination); supports IPOP-style warm restarts via the Restart analyzer.  Two swarm topologies are available via the ``topology`` argument: the default fully-connected ``"gbest"`` (Kennedy-Eberhart 1995) and the wrap-around ring ``"lbest"`` of width ``2·k_neighbors+1`` (Kennedy & Mendes 2002), which trades slower information diffusion for stronger multimodal exploration.  Optional ``w_end`` enables the Shi–Eberhart (1998) linearly-decreasing inertia schedule (off by default; constant ``w`` reproduces the prior behaviour).
- **FeasibleSearch**: Actively searches for and repairs feasible solutions via line search towards feasibility.
- **ConstraintGradient**: Uses estimated gradients of constraint violations to find feasible regions.
- **LocalPenaltySearch**: Optimizes the scalarized penalty function directly with a local scipy optimizer.
- **ConstraintRepair**: Projects infeasible best points onto the feasible region using SLSQP.
- **ClaudeHeuristic**: Cluster-based adaptive search using Mixture of Gaussians over elite points.

.. automodule:: panobbgo.heuristics
   :members:
   :undoc-members:
   :show-inheritance:

