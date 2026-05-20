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
- **COBYQA**: Powell-family derivative-free trust-region local optimizer (Ragonneau & Zhang 2023) — the modern successor to BOBYQA / COBYLA / NEWUOA / LINCOA.  Maintains an interpolation set of ``2·n + 1`` points and fits an adaptive *quadratic model* of the objective inside a trust region; natively supports bounds.  Curvature-aware and derivative-free at once, dominant on smooth / near-smooth local refinement (e.g. Rosenbrock-like ill-conditioned valleys where Nelder-Mead converges slowly).  Subprocess-backed adapter around ``scipy.optimize.minimize(method="COBYQA")`` (scipy 1.14+).
- **CMAES**: Covariance Matrix Adaptation Evolution Strategy with IPOP-CMA-ES restart support (paired with the Restart analyzer) — gold standard for derivative-free continuous optimization, excellent on ill-conditioned/ridge-following problems like Rosenbrock.
- **DifferentialEvolution**: DE mutation/crossover/selection applied to the accumulated result database — strong on multimodal landscapes (Rastrigin, Schwefel).
- **LSHADE**: Linear-population-reduction Success-History Adaptive DE (Tanabe & Fukunaga, CEC 2014 winner).  Adapts ``F`` and ``CR`` per-trial via per-bin Cauchy / Normal memories that update each generation by the weighted Lehmer mean of successful triples; uses the ``current-to-pbest/1`` mutation with an external archive of replaced parents.  The population shrinks linearly from ``NP_init`` (default 30) down to ``NP_min`` (default 4) over the strategy's evaluation budget — broad exploration early, focused exploitation late.  Out-of-bounds components are repaired by midpoint reflection.  Strictly stronger than the basic ``DE/rand/1/bin`` on multimodal benchmarks; opt-in via the structural mutation catalog.
- **JSO**: jSO (Brest, Maučec & Bošković, CEC 2017 winner) — direct subclass of :class:`~panobbgo.heuristics.lshade.LSHADE` that adds three literature-best refinements to the success-history adaptive DE machinery: a **weighted current-to-pbest-w/1 mutation** (the pbest direction is re-weighted by a phase-dependent ``F_w`` factor — ``0.7·F`` early, ``0.8·F`` mid, ``1.2·F`` late), a **linear ``p_best`` schedule** that decreases from ``0.25`` to ``0.125`` over the budget, and the **asymmetric three-phase Cauchy-F cap** that limits sampled ``F`` to ``0.7`` for the first 60% of the budget, ``0.8`` for the next 30%, and ``1.0`` (effectively unbounded) for the final 10%.  The history memory uses ``M_F = 0.3`` / ``M_CR = 0.8`` initial values and reserves the last bin (``H − 1``) as a frozen anchor at ``0.9 / 0.9`` that ``_update_memory`` never overwrites — a stable "moderately greedy" parameter source independent of live success history.  Inherits LSHADE's asynchronous pipeline (per-slot pending dict, generation-by-count book-keeping, archive trimming, LPSR shrinking, warm restart) unchanged.  Opt-in via the structural mutation catalog.
- **DifferentialEvolution / LSHADE / JSO complementarity**: Panobbgo carries all three DE-family arms in the structural catalog so the bandit can pick whichever wins on the current battery — basic DE for byte-identical legacy reproduction, L-SHADE for the established CEC-2014 high-water mark, and jSO for the CEC-2017 refinement.  Each variant occupies a distinct exploration / exploitation regime and the literature shows real per-problem complementarity.
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

