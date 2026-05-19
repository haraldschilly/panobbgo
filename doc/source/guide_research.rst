Research Context and Future Directions
======================================

This section places Panobbgo in the broader context of optimization research and outlines future research directions.

Inspiration: SNOBFIT
--------------------

Panobbgo is inspired by `SNOBFIT <http://www.mat.univie.ac.at/~neum/software/snobfit/>`_
(Stable Noisy Optimization by Branch and Fit) [Huyer2008]_.

SNOBFIT's Key Ideas
~~~~~~~~~~~~~~~~~~~

1. **Quadratic local models** with uncertainty quantification
2. **Branch-and-bound** spatial decomposition of search space
3. **Noise handling** through repeated evaluations and safeguards
4. **Adaptive sampling** balancing exploration and exploitation

Panobbgo's Innovations
~~~~~~~~~~~~~~~~~~~~~~

While inspired by SNOBFIT, Panobbgo introduces several innovations:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - SNOBFIT
     - Panobbgo
   * - Architecture
     - Monolithic algorithm
     - **Modular components** (heuristics, analyzers)
   * - Search strategy
     - Single algorithm
     - **Multiple heuristics** working together
   * - Adaptation
     - Fixed strategy
     - **Multi-armed bandit** learns heuristic performance
   * - Extensibility
     - Closed system
     - **Event-driven** plug-in architecture
   * - Parallelism
     - Sequential
     - **Dask distributed** integration from start

Related Algorithms
------------------

Black-box optimization is a rich research area with many approaches.

Bayesian Optimization
~~~~~~~~~~~~~~~~~~~~~

**Approach**: Fit Gaussian process (GP) surrogate model, optimize acquisition function

**Panobbgo implementation**: :class:`~panobbgo.heuristics.gaussian_process.GaussianProcessHeuristic`
fits a Matérn-5/2 GP via scikit-learn and optimises EI / UCB / PI acquisition functions.  Uses
Constrained Expected Improvement (EIC) automatically when the problem has active constraint
violations.  :class:`~panobbgo.heuristics.quadratic_wls_model.QuadraticWlsModel` is retained as a
faster, less accurate quadratic surrogate alternative.

**Trade-offs**:

- GP (GaussianProcessHeuristic): Better uncertainty quantification, higher computational cost — gold standard for expensive BO
- Quadratic (QuadraticWlsModel): Faster, scales better, but less accurate

**References**: [Mockus1989]_, [Brochu2010]_

CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Approach**: Evolutionary algorithm with adaptive covariance matrix

**Panobbgo implementation**: :class:`~panobbgo.heuristics.cma_es.CMAES` provides a pure-NumPy
CMA-ES with asynchronous generation tracking that fits panobbgo's threaded evaluation model.
When paired with :class:`~panobbgo.analyzers.restart.Restart` it becomes **IPOP-CMA-ES**
(Increasing Population CMA-ES, Auger & Hansen CEC 2005) — the competition-winning approach on
BBOB/COCO: each restart doubles the population λ → 2λ, resets the covariance to identity, and
moves the mean to a diverse new center, enabling systematic escape from local optima while the
accumulated result history is preserved.

**Trade-offs**:

- CMA-ES: Powerful for smooth, ill-conditioned, and ridge-following problems (e.g., Rosenbrock); invariant under orthogonal search-space transformations
- IPOP-CMA-ES: Adds multimodal robustness at the cost of extra restarts
- NelderMead: Deterministic simplex, good for final local refinement

**References**: [Hansen2001]_

Differential Evolution
~~~~~~~~~~~~~~~~~~~~~~

**Approach**: Population-based mutation/crossover/selection on a population of solution vectors

**Panobbgo implementation**: :class:`~panobbgo.heuristics.differential_evolution.DifferentialEvolution`
runs the basic ``DE/rand/1/bin`` operators against the accumulated result database.  Strong
complement to CMA-ES on multimodal landscapes such as Rastrigin and Schwefel where purely
local methods get trapped.

For the literature-best adaptive variant, see :class:`~panobbgo.heuristics.lshade.LSHADE`,
which adapts ``F`` and ``CR`` per-trial via success-history memories and shrinks the
population linearly with the budget (Tanabe & Fukunaga 2014, CEC 2014 winner).  L-SHADE
uses the ``current-to-pbest/1`` mutation (Zhang & Sanderson 2009) with an external
archive of replaced parents and is widely cited as one of the strongest single-population
black-box optimizers — the high-water mark from which subsequent variants
(jSO, IMODE, NL-SHADE-RSP) refine.

For the **CEC-2017 refinement**, see :class:`~panobbgo.heuristics.jso.JSO`
(Brest, Maučec & Bošković 2017 — winner of the CEC-2017 single-objective
bound-constrained competition).  jSO is a direct subclass of L-SHADE that adds:
a *weighted* ``current-to-pbest-w/1`` mutation (the pbest direction is re-weighted by a
phase-dependent ``F_w`` factor — ``0.7·F`` early, ``0.8·F`` mid, ``1.2·F`` late);
a *linear* ``p_best`` schedule (``0.25 → 0.125``) that contracts greediness as the
population shrinks; *Cauchy-F clamping* that limits ``F`` to ``0.7`` while
``progress < 0.6``; and a frozen anchor memory bin at ``M_F = M_CR = 0.9`` that
``_update_memory`` never overwrites.  Subsequent CEC winners (jDE100,
NL-SHADE-RSP, NL-SHADE-LBC) all cite jSO as their direct ancestor.

**References**: [Storn1997]_; Zhang & Sanderson (2009); Tanabe & Fukunaga (2013, 2014);
Brest, Maučec & Bošković (CEC 2017).

Particle Swarm Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Approach**: A population (swarm) of particles, each carrying a position, velocity, and
memory of its personal best.  Velocities are pulled toward both the personal best and the
global best with random per-component weights, giving the swarm both *momentum* (velocity
inertia retained from the prior step) and *social* attraction toward the swarm's leader.

**Panobbgo implementation**: :class:`~panobbgo.heuristics.pso.PSO` uses the canonical
Clerc–Kennedy (2002) constriction-coefficient parameters
(``w = χ ≈ 0.7298``, ``c1 = c2 ≈ 1.49618``) which provably guarantee convergence.  The
heuristic runs asynchronously in the panobbgo event loop following the same pattern as
:class:`DifferentialEvolution`, and supports IPOP-style warm restarts via the
:class:`~panobbgo.analyzers.restart.Restart` analyzer.  PSO's dynamics are markedly
different from CMA-ES (which adapts a covariance matrix and re-samples) and DE (which
recombines three random members) — they exploit narrow-valley problems where vector
inertia along the valley floor is more useful than the alternatives.

The ``topology`` argument selects how the *social* attractor for each particle is
chosen.  ``"gbest"`` (default) uses the single global best — fast contraction once a
basin is found.  ``"lbest"`` switches to a wrap-around *ring* of width
``2·k_neighbors + 1`` so information about a new best diffuses through the swarm at
one hop per iteration; this is empirically stronger on highly-multimodal landscapes
(Kennedy & Mendes 2002).  Panobbgo's structural mutation catalog ships *both* variants
so the self-improvement loop can pick whichever helps on a given problem family.

**References**: Kennedy & Eberhart (1995); Clerc & Kennedy (2002); Kennedy & Mendes
(2002); Poli, Kennedy & Blackwell (2007).

DIRECT (Dividing Rectangles)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Approach**: Lipschitz-based partitioning, optimistic sampling

**Panobbgo equivalent**: :class:`~panobbgo.analyzers.splitter.Splitter`
provides hierarchical box decomposition, but without Lipschitz assumptions

**Trade-offs**:

- DIRECT: Rigorous convergence guarantees with Lipschitz constant
- Splitter: No Lipschitz assumption, adapts to actual problem

**References**: [Jones1993]_

Simulated Annealing
~~~~~~~~~~~~~~~~~~~

**Approach**: Probabilistic acceptance with cooling schedule

**Panobbgo equivalent**: :class:`~panobbgo.strategies.rewarding.StrategyRewarding`
uses probabilistic selection, but without explicit temperature

**Trade-offs**:

- SA: Single-point trajectory, cooling schedule critical
- StrategyRewarding: Multi-heuristic, adapts automatically

**References**: [Kirkpatrick1983]_

Comparison Summary
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Algorithm
     - Strengths
     - Weaknesses
     - Best For
   * - Bayesian Opt (GP)
     - Uncertainty quantification
     - O(n³) cost, scales poorly
     - Low-dim (≤20), expensive evaluations
   * - CMA-ES
     - Robust, widely applicable
     - Population-based, many evaluations
     - Medium-dim (≤100), smooth
   * - DIRECT
     - Convergence guarantees
     - Needs Lipschitz constant
     - Lipschitz functions
   * - SNOBFIT
     - Noise handling
     - Single algorithm
     - Noisy, box-constrained
   * - **Panobbgo**
     - **Modular, extensible**
     - **Requires tuning**
     - **Research, experimentation**

Theoretical Properties
----------------------

Convergence Guarantees
~~~~~~~~~~~~~~~~~~~~~~

**Theorem** (Informal): Under mild assumptions, Panobbgo converges to the global optimum almost surely as the number of evaluations :math:`N \to \infty`.

**Assumptions**:

1. :math:`f` is locally Lipschitz continuous almost everywhere
2. At least one heuristic performs dense sampling (e.g., :class:`~panobbgo.heuristics.random.Random`)
3. Multi-armed bandit maintains non-zero selection probability for all heuristics (via additive smoothing)

**Proof Sketch**:

1. Dense sampling ensures every region eventually explored
2. Local heuristics (NelderMead, Nearby) ensure refinement
3. Adaptive selection focuses on promising regions
4. Lexicographic ordering handles constraints correctly

**Convergence Rate**: Problem-dependent, but typically:

- **Easy problems** (unimodal, smooth): :math:`O(\log N)` evaluations to :math:`\epsilon`-optimum
- **Hard problems** (highly multimodal): :math:`O(N^{\alpha})` for :math:`\alpha < 1`

Complexity Analysis
~~~~~~~~~~~~~~~~~~~

**Per-iteration complexity**:

- Sorting results: :math:`O(N \log N)`
- Pareto updates: :math:`O(N \cdot |\text{Front}|) = O(N \log N)` typically
- Event dispatching: :math:`O(|H| + |A|)` where :math:`|H|` = heuristics, :math:`|A|` = analyzers
- **Total**: :math:`O(N \log N)` per iteration

**Overall complexity**:

- Function evaluations: :math:`O(N_{max})` (the actual bottleneck)
- Framework overhead: :math:`O(N_{max}^2 \log N_{max})` worst case
- Practical overhead: :math:`O(N_{max} \log N_{max})` (negligible compared to evaluations)

**Parallelism**:

- Up to :math:`P` simultaneous evaluations (cluster size)
- Wall-clock time: :math:`O(N_{max} / P)` assuming balanced load
- Communication overhead: Minimal (Dask handles efficiently)

Current Research Challenges
---------------------------

High-Dimensional Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Challenge**: Curse of dimensionality—exponential growth of search space

**Current approaches**:

- Random embeddings [Wang2013]_
- Coordinate descent
- Additive models

**Panobbgo future work**:

- Implement random embedding heuristic
- Coordinate-wise optimization strategy
- Dimension reduction analyzers

Multi-Fidelity Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Challenge**: Leverage cheap low-fidelity evaluations to guide expensive high-fidelity ones

**Example**: CFD simulation at coarse vs. fine mesh

**Approach**:

- Low-fidelity: Fast, less accurate
- High-fidelity: Slow, accurate
- Learn correlation, allocate budget optimally

**Panobbgo future work**:

- Multi-fidelity Problem class
- Fidelity-aware heuristics
- Budget allocation strategy

Constraint Handling
~~~~~~~~~~~~~~~~~~~

**Challenge**: Efficiently find feasible solutions

**Current Panobbgo**: Pluggable :class:`~panobbgo.lib.constraints.ConstraintHandler` implementations:

- ``DefaultConstraintHandler`` — lexicographic ordering (feasibility first)
- ``PenaltyConstraintHandler`` — static penalty :math:`f(x) + \rho \cdot cv(x)^{e}`
- ``DynamicPenaltyConstraintHandler`` — penalty coefficient increases over time
- ``AugmentedLagrangianConstraintHandler`` — adaptive multipliers :math:`\lambda` and penalty :math:`\mu`
- ``EpsilonConstraintHandler`` — tolerance :math:`\epsilon(t)` shrinks towards zero
- ``FilterConstraintHandler`` — Pareto-dominance filter on :math:`(f, cv)`

Complemented by constraint-focused heuristics: FeasibleSearch (line search to feasibility),
ConstraintGradient (finite-difference constraint-gradient descent), LocalPenaltySearch
(scipy local optimiser on the scalarised penalty), and ConstraintRepair (SLSQP projection).

**Future work**:

- Feasibility pump analyzer
- Adaptive penalty parameters guided by the Sensitivity analyzer
- Constraint approximation / surrogate-assisted repair

Robust Optimization
~~~~~~~~~~~~~~~~~~~

**Challenge**: Optimize under uncertainty in parameters or noise

**Formulation**:

.. math::

   \min_{x} \mathbb{E}[f(x, \xi)] \quad \text{or} \quad \min_x \max_\xi f(x, \xi)

where :math:`\xi` represents uncertainty

**Panobbgo future work**:

- Stochastic evaluation with confidence intervals
- Worst-case optimization heuristics
- Robust best analyzer

Transfer Learning
~~~~~~~~~~~~~~~~~

**Challenge**: Learn from previous optimization runs on related problems

**Approach**:

- Meta-learning: Learn heuristic performance across problem classes
- Warm-start: Initialize from previous solutions
- Feature extraction: Characterize problem properties

**Panobbgo future work**:

- Problem feature extraction
- Heuristic performance database
- Meta-strategy that initializes bandit weights

Future Directions
-----------------

Recently Completed
~~~~~~~~~~~~~~~~~~

The following items from earlier roadmaps have been implemented:

1. **Gaussian Process Surrogate** — see :class:`~panobbgo.heuristics.gaussian_process.GaussianProcessHeuristic`
   (scikit-learn, Matérn-5/2 kernel, EI / UCB / PI acquisition, constrained EI).
2. **Persistent Storage** — SQLite backend for :class:`~panobbgo.core.Results` with automatic
   resume-from-checkpoint (see :mod:`panobbgo.storage`).
3. **Convergence Detection** — :class:`~panobbgo.analyzers.convergence.Convergence` analyzer
   with ``std`` and ``improv`` modes publishing a ``converged`` event.
4. **Better Constraint Handling** — multiple constraint handlers (Default, Penalty,
   DynamicPenalty, AugmentedLagrangian, Epsilon, Filter) plus constraint-focused heuristics
   (FeasibleSearch, ConstraintGradient, LocalPenaltySearch, ConstraintRepair).
5. **Advanced Bandit Strategies** — :class:`~panobbgo.strategies.ucb.StrategyUCB`,
   :class:`~panobbgo.strategies.thompson.StrategyThompsonSampling`, and contextual
   :class:`~panobbgo.strategies.contextual.StrategyLinUCB`.
6. **Budget-Phased Meta-Strategy** — :class:`~panobbgo.strategies.phased.StrategyPhased`
   composes different sub-strategies and heuristic portfolios across budget phases.
7. **Population-Based Global Search** — :class:`~panobbgo.heuristics.cma_es.CMAES` with
   IPOP restart support via :class:`~panobbgo.analyzers.restart.Restart`, and
   :class:`~panobbgo.heuristics.differential_evolution.DifferentialEvolution`.
8. **Sensitivity Analysis** — :class:`~panobbgo.analyzers.sensitivity.Sensitivity` with a
   sensitivity-aware :class:`~panobbgo.heuristics.nearby.Nearby` heuristic.
9. **Reproducible Benchmark Harness** — see :mod:`panobbgo.harness` and ``benchmark_harness.py``
   CLI for measuring the impact of code changes on optimisation performance.

Short-Term Improvements
~~~~~~~~~~~~~~~~~~~~~~~

1. **Enhanced Visualization**

   - Real-time optimization plots
   - Convergence dashboards
   - Pareto front visualization
   - Heuristic performance tracking

2. **Parallel Batch Generation (q-acquisition)**

   - Generate batches considering pending evaluations
   - Avoid redundant sampling (q-EI, q-UCB)
   - Optimize batch diversity

Medium-Term Research
~~~~~~~~~~~~~~~~~~~~

1. **Multi-Fidelity Support**

   - Multi-fidelity Problem class
   - Fidelity allocation strategy
   - Low/high-fidelity coordination

2. **Dimension Reduction**

   - Random embeddings
   - Active subspace detection
   - Coordinate importance ranking (building on the Sensitivity analyzer)

3. **Benchmarking Suite**

   - Standardized test problems
   - Performance profiles [Dolan2002]_
   - Comparison with state-of-the-art

Long-Term Vision (1-2 years)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Transfer Learning Framework**

   - Meta-learning across problem families
   - Heuristic performance prediction
   - Automatic portfolio construction

2. **AutoML Integration**

   - Hyperparameter optimization for ML models
   - Neural architecture search
   - Ensemble model optimization

3. **Distributed Computing**

   - Cloud-native deployment
   - Container orchestration (Kubernetes)
   - Serverless function evaluation

4. **Interactive Optimization**

   - User-in-the-loop feedback
   - Preference learning
   - Multi-objective trade-off exploration

5. **Symbolic Regression Integration**

   - Learn interpretable models of :math:`f`
   - Symbolic surrogate optimization
   - Physical constraint discovery

 Bandit Strategy Benchmarking
----------------------------

Panobbgo includes a comprehensive benchmarking suite for comparing different
adaptive strategies (multi-armed bandits).

Benchmark Script
~~~~~~~~~~~~~~~~

The script ``benchmarks/run_bandit_comparison.py`` allows you to empirically
compare the performance of:

- **RoundRobin**: Deterministic baseline
- **Rewarding**: Softmax-based adaptive strategy
- **UCB**: Upper Confidence Bound (UCB1)
- **Thompson**: Thompson Sampling (Beta-Bernoulli)
- **LinUCB**: Contextual bandit using problem features

Usage:

.. code-block:: bash

   uv run python benchmarks/run_bandit_comparison.py

This generates convergence plots and a summary CSV in ``benchmarks/results_bandits/``.

Strategies Compared
~~~~~~~~~~~~~~~~~~~

1. **StrategyRoundRobin**:
   Cycles through heuristics in a fixed order. Serves as a robust baseline
   that ensures all heuristics get equal resources.

2. **StrategyRewarding**:
   Uses a weighted probability distribution (Softmax) based on recent performance.
   Rewards are discounted over time to adapt to changing landscapes.

3. **StrategyUCB**:
   Balances exploration (uncertainty) and exploitation (average reward) using
   the UCB1 algorithm. Effective for stationary distributions but adapts slowly.

4. **StrategyThompsonSampling**:
   Maintains a Beta distribution for each heuristic's success probability.
   Samples from these distributions to select heuristics, naturally handling
   exploration-exploitation trade-offs.

5. **StrategyLinUCB**:
   Learns a linear mapping from context features (e.g., budget progress,
   global success rate) to expected reward for each heuristic.
   Allows the strategy to learn *when* to use specific heuristics (e.g.,
   exploration early, local search late).

Contributing to Research
------------------------

Academic Usage
~~~~~~~~~~~~~~

If you use Panobbgo in academic research, please cite:

.. code-block:: bibtex

   @software{panobbgo,
     author = {Schilly, Harald},
     title = {Panobbgo: Parallel Noisy Black-Box Global Optimization},
     year = {2012-2025},
     url = {https://github.com/haraldschilly/panobbgo},
     version = {0.0.1}
   }

Research Collaboration
~~~~~~~~~~~~~~~~~~~~~~

We welcome research collaborations! Areas of interest:

- Novel heuristic designs
- Adaptive strategy algorithms
- Application domains (materials, aerospace, etc.)
- Theoretical analysis (convergence, complexity)
- Benchmarking studies

Contact: See https://github.com/haraldschilly/panobbgo

Open Problems
~~~~~~~~~~~~~

Interesting open questions:

1. **Optimal heuristic portfolio**: What's the minimal set of heuristics for broad effectiveness?
2. **Dynamic portfolio**: Can we add/remove heuristics during optimization?
3. **Problem fingerprinting**: What features predict which heuristics will work?
4. **Parallel efficiency**: How to minimize redundant evaluations in parallel batches?
5. **Noisy constraints**: How to handle stochastic constraint violations?

Experimental Validation
-----------------------

Benchmark Problems
~~~~~~~~~~~~~~~~~~

Standard test suites for validation:

- **CEC Benchmark Functions** [Liang2013]_
- **BBOB (Black-Box Optimization Benchmarking)** [Hansen2009]_
- **Engineering design problems** [Mezura2006]_

Performance Metrics
~~~~~~~~~~~~~~~~~~~

Common metrics for comparison:

1. **Best value found**: :math:`f(x^*)`
2. **Success rate**: Fraction reaching :math:`\epsilon`-optimum
3. **Convergence speed**: Evaluations to reach target
4. **Robustness**: Performance variance across runs
5. **Pareto front quality**: For constrained/multi-objective

Experimental Design
~~~~~~~~~~~~~~~~~~~

Rigorous benchmarking requires:

- **Multiple runs**: Account for stochasticity (typically 20-50 runs)
- **Statistical tests**: Wilcoxon, Friedman, etc.
- **Performance profiles**: Visualize relative performance
- **Fair comparison**: Equal budgets, same problems
- **Reproducibility**: Fixed seeds, documented setup

References
----------

.. [Auer2002] Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).
   Finite-time analysis of the multiarmed bandit problem.
   *Machine Learning*, 47(2), 235-256.

.. [Brochu2010] Brochu, E., Cora, V. M., & De Freitas, N. (2010).
   A tutorial on Bayesian optimization of expensive cost functions.
   *arXiv preprint arXiv:1012.2599*.

.. [Dolan2002] Dolan, E. D., & Moré, J. J. (2002).
   Benchmarking optimization software with performance profiles.
   *Mathematical Programming*, 91(2), 201-213.

.. [Hansen2001] Hansen, N., & Ostermeier, A. (2001).
   Completely derandomized self-adaptation in evolution strategies.
   *Evolutionary Computation*, 9(2), 159-195.

.. [Hansen2009] Hansen, N., Auger, A., Ros, R., Finck, S., & Pošík, P. (2009).
   Comparing results of 31 algorithms from the black-box optimization benchmarking BBOB-2009.
   *ACM SIGEVO*, 1(1), 1-25.

.. [Huyer2008] Huyer, W., & Neumaier, A. (2008).
   SNOBFIT–stable noisy optimization by branch and fit.
   *ACM Trans. Math. Softw.*, 35(2), 1-25.

.. [Jones1993] Jones, D. R., Perttunen, C. D., & Stuckman, B. E. (1993).
   Lipschitzian optimization without the Lipschitz constant.
   *Journal of Optimization Theory and Applications*, 79(1), 157-181.

.. [Kirkpatrick1983] Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983).
   Optimization by simulated annealing.
   *Science*, 220(4598), 671-680.

.. [Liang2013] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013).
   Problem definitions and evaluation criteria for the CEC 2014 special session
   and competition on single objective real-parameter numerical optimization.
   *Technical Report*.

.. [Mezura2006] Mezura-Montes, E., & Coello, C. A. C. (2006).
   A simple multimembered evolution strategy to solve constrained optimization problems.
   *IEEE Transactions on Evolutionary Computation*, 9(1), 1-17.

.. [Mockus1989] Mockus, J. (1989).
   *Bayesian Approach to Global Optimization: Theory and Applications*.
   Kluwer Academic Publishers.

.. [Storn1997] Storn, R., & Price, K. (1997).
   Differential evolution – A simple and efficient heuristic for global optimization over continuous spaces.
   *Journal of Global Optimization*, 11(4), 341-359.

.. [Thompson1933] Thompson, W. R. (1933).
   On the likelihood that one unknown probability exceeds another in view of the evidence of two samples.
   *Biometrika*, 25(3-4), 285-294.

.. [Wang2013] Wang, Z., Zoghi, M., Hutter, F., Matheson, D., & De Freitas, N. (2013).
   Bayesian optimization in high dimensions via random embeddings.
   *IJCAI*, 1778-1784.
