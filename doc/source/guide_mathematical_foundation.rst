Mathematical Foundation
=======================

This section provides the mathematical foundation underlying Panobbgo's approach to black-box global optimization.

Problem Formulation
-------------------

Objective Function
~~~~~~~~~~~~~~~~~~

The basic optimization problem is:

.. math::

   \min_{x \in B \subset \mathbb{R}^n} f(x)

where:

- :math:`B = \{x \in \mathbb{R}^n : l_i \leq x_i \leq u_i, \; i=1,\ldots,n\}` is the search box
- :math:`f: B \rightarrow \mathbb{R}` is a potentially noisy, expensive-to-evaluate function

Constraint Handling
~~~~~~~~~~~~~~~~~~~

If constraint functions :math:`g_i(x) \leq 0` for :math:`i = 1, \ldots, m` exist, we define the constraint violation vector:

.. math::

   cv(x) = [g_1(x)_+, \ldots, g_m(x)_+] \quad \text{where} \quad (a)_+ = \max(0, a)

The total constraint violation is the L2 norm:

.. math::

   CV(x) = \|cv(x)\|_2 = \sqrt{\sum_{i=1}^m g_i(x)_+^2}

Improvement Calculation
~~~~~~~~~~~~~~~~~~~~~~~

When comparing two results (e.g., the current best vs. a new result) to calculate rewards for heuristics, Panobbgo uses a dedicated :class:`~panobbgo.lib.constraints.ConstraintHandler`.

Default Constraint Handler
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~panobbgo.lib.constraints.DefaultConstraintHandler` prioritizes feasibility using a lexicographic approach:

1. **Both Feasible**: Standard improvement in objective function.

   .. math::
      I = \max(0, f(x_{old}) - f(x_{new}))

2. **Infeasible to Feasible**: Significant improvement, rewarding the transition to feasibility.

   .. math::
      I = C + \rho \cdot CV(x_{old})

   where :math:`C` is a base constant (e.g., 10.0) and :math:`\rho` is a penalty factor (e.g., 100.0).

3. **Both Infeasible**: Improvement based on reduction of constraint violation.

   .. math::
      I = \rho \cdot \max(0, CV(x_{old}) - CV(x_{new}))

4. **Feasible to Infeasible**: No improvement (:math:`I=0`).

Penalty Constraint Handlers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, Panobbgo supports penalty-based handlers which combine the objective and constraint violation into a single scalar metric :math:`P(x)`.

**PenaltyConstraintHandler**:
Uses a static penalty function:

.. math::
   P(x) = f(x) + \rho \cdot CV(x)^\gamma

where :math:`\gamma` is an exponent (typically 1 or 2). Improvement is defined as reduction in :math:`P(x)`.

**DynamicPenaltyConstraintHandler**:
Uses a penalty factor that increases over time to gradually enforce constraints:

.. math::
   P(x, t) = f(x) + \rho(t) \cdot CV(x)^\gamma

where :math:`\rho(t)` grows with the number of evaluations.

Heuristic-Specific Approaches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some heuristics implement specialized constraint handling logic beyond the general handlers.

**Expected Improvement with Constraints (EIC)**:
The :class:`~panobbgo.heuristics.gaussian_process.GaussianProcessHeuristic` supports EIC. It models the objective function :math:`f(x)` and constraint violation :math:`cv(x)` separately using Gaussian Processes.

.. math::
   EIC(x) = EI_f(x) \times P(cv(x) \le 0)

where :math:`EI_f(x)` is the standard Expected Improvement of the objective, and :math:`P(cv(x) \le 0)` is the probability that the point is feasible, derived from the constraint GP model. This effectively penalizes infeasible regions probabilistically rather than using a hard penalty factor.

Lexicographic Ordering
~~~~~~~~~~~~~~~~~~~~~~

For maintaining the global "best" point, point :math:`x` is considered better than :math:`y` if:

1. :math:`CV(x) < CV(y)` (less constraint violation), **OR**
2. :math:`CV(x) = CV(y) = 0` **AND** :math:`f(x) < f(y)` (better objective value)

This ensures that:

- Feasible points (:math:`CV(x) = 0`) are always preferred over infeasible ones
- Among feasible points, we minimize the objective function
- Among infeasible points, we minimize constraint violations

The Exploration-Exploitation Tradeoff
--------------------------------------

Black-box optimization requires balancing two competing objectives:

- **Exploration**: Sample broadly to discover the global structure and avoid missing the global optimum
- **Exploitation**: Sample intensively in promising regions to refine the current best solution

Traditional Approaches
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Method
     - Exploration Strategy
     - Exploitation Strategy
   * - Pure Random Search
     - ✓ Complete coverage
     - ✗ No refinement
   * - Local Optimization
     - ✗ Gets stuck in local minima
     - ✓ Fast convergence locally
   * - Simulated Annealing
     - Initially high, decreases
     - Increases over time
   * - Bayesian Optimization
     - Via acquisition function
     - Via acquisition function

Panobbgo's Approach
~~~~~~~~~~~~~~~~~~~

Panobbgo addresses the tradeoff through:

1. **Diverse heuristics**: Each with different exploration/exploitation profiles
2. **Adaptive selection**: Learn which heuristics work for this problem (multi-armed bandit)
3. **Hierarchical decomposition**: Partition space based on results (Splitter analyzer)

Multi-Armed Bandit Strategy
----------------------------

The Core Idea
~~~~~~~~~~~~~

Selecting which heuristic to use next is modeled as a **multi-armed bandit problem**:

- Each heuristic :math:`h \in H` is an "arm" of a slot machine
- "Pulling arm :math:`h`" = requesting a point from heuristic :math:`h`
- "Reward" :math:`R` = improvement in objective value
- **Goal**: Maximize cumulative reward (find best points quickly)

The challenge is the **exploration-exploitation dilemma**:

- Should we use heuristics that have worked well (exploitation)?
- Or try other heuristics that might work better (exploration)?

Implementation in StrategyRewarding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each heuristic :math:`h` maintains a performance score :math:`p_h(t)` updated as:

.. math::

   p_h(t+1) = \begin{cases}
   p_h(t) + R(x) & \text{if } h \text{ generated } x \text{ and } f(x) < f_{best} \\
   p_h(t) \cdot d & \text{if } h \text{ generated a point} \\
   p_h(t) & \text{otherwise}
   \end{cases}

where:

- :math:`R(x) = 1 - e^{-(f_{best} - f(x))}` is the reward (saturates to 1)
- :math:`d \in (0, 1)` is a discount factor (default: 0.95)

Selection Probability
~~~~~~~~~~~~~~~~~~~~~

Heuristic :math:`h` is selected with probability:

.. math::

   P(h) = \frac{p_h + s}{\sum_{h' \in H} p_{h'} + s \cdot |H|}

where :math:`s > 0` is an **additive smoothing** parameter (default: 0.1).

**Properties:**

- Better-performing heuristics get selected more often
- Discount factor :math:`d` causes old successes to fade, preventing lock-in
- Smoothing :math:`s` ensures all heuristics tried occasionally (exploration)
- As :math:`s \to 0`: more exploitation; as :math:`s \to \infty`: uniform random

Why This Works
~~~~~~~~~~~~~~

1. **Adaptivity**: Performance tracked dynamically, strategy adapts to problem
2. **Diversification**: Multiple heuristics with different search characteristics
3. **Robustness**: No single heuristic needs to solve everything
4. **Learning**: Better heuristics discovered automatically

Pareto Optimality for Constrained Problems
-------------------------------------------

Pareto Front Definition
~~~~~~~~~~~~~~~~~~~~~~~

For constrained problems, we track the **Pareto front** in :math:`(f, CV)` space.

A point :math:`(f, cv)` is **Pareto optimal** if there exists no other point :math:`(f', cv')` such that:

.. math::

   f' \leq f \land cv' \leq cv \land (f' < f \lor cv' < cv)

In other words: no other point is better in both objectives simultaneously.

Pareto Update Algorithm
~~~~~~~~~~~~~~~~~~~~~~~

When a new result :math:`(f_{new}, cv_{new})` arrives:

1. Check if dominated by existing Pareto points:

   .. math::

      \text{If } \exists (f_p, cv_p) \in \text{Front}: f_p \leq f_{new} \land cv_p \leq cv_{new}

   Then :math:`(f_{new}, cv_{new})` is **not** Pareto optimal → reject

2. Otherwise, add :math:`(f_{new}, cv_{new})` to Front

3. Remove existing points dominated by :math:`(f_{new}, cv_{new})`:

   .. math::

      \text{Remove } (f_p, cv_p) \text{ if } f_{new} \leq f_p \land cv_{new} \leq cv_p

Practical Implications
~~~~~~~~~~~~~~~~~~~~~~

The Pareto front reveals:

- **Trade-off curve**: How much objective improvement costs in constraint violations
- **Feasibility boundary**: Where constraints become satisfiable
- **Solution diversity**: Multiple viable solutions with different characteristics

Hierarchical Spatial Decomposition
-----------------------------------

Motivation
~~~~~~~~~~

As optimization progresses, we learn which regions of the search space are promising.
The :class:`~panobbgo.analyzers.splitter.Splitter` maintains a hierarchical decomposition
to focus search appropriately.

Box Tree Algorithm
~~~~~~~~~~~~~~~~~~

1. **Initialization**: Root node = entire bounding box :math:`B`

2. **Split Criterion**: When a box :math:`B_i` contains sufficient points:

   .. math::

      n(B_i) > \alpha \cdot \dim(B)

   where :math:`\alpha` is a split factor (default: 5) and :math:`n(B_i)` is the number of evaluated points in :math:`B_i`

3. **Split Operation**:

   - Find dimension :math:`d^*` with largest range: :math:`d^* = \arg\max_d (u_d - l_d)`
   - Compute midpoint: :math:`m = (l_{d^*} + u_{d^*}) / 2`
   - Create two child boxes splitting at :math:`m` along dimension :math:`d^*`

4. **Best Leaf Identification**:

   The "best leaf" is the leaf box containing the current best point (by lexicographic ordering)

Usage by Heuristics
~~~~~~~~~~~~~~~~~~~

Heuristics can adapt their sampling based on the decomposition:

- **Random**: Samples uniformly from the best leaf box
- **Latin Hypercube**: Initial space-filling over entire :math:`B`
- **Nearby**: Local search around best point (independent of boxes)

This provides a **dynamic balance**:

- Early optimization: Few splits → broad exploration
- Late optimization: Many splits → focused exploitation in promising regions

Convergence Properties
-----------------------

Theoretical Guarantees
~~~~~~~~~~~~~~~~~~~~~~

Under mild assumptions, Panobbgo converges to the global optimum almost surely:

**Assumptions:**

1. :math:`f` is locally Lipschitz continuous almost everywhere
2. Evaluation budget :math:`N \to \infty`
3. At least one heuristic performs uniform sampling (e.g., Random)

**Why Convergence Holds:**

1. **Dense Coverage**: Latin Hypercube + Random ensure dense initial sampling
2. **Persistent Exploration**: Random heuristic never stops, smoothing ensures it's selected
3. **Exploitation**: Local heuristics (NelderMead, Nearby) refine promising regions
4. **Adaptivity**: Bandit focuses on heuristics that work for this problem

Practical Convergence Rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In practice, convergence depends on:

- **Problem difficulty**: Multimodality, noise level, dimension
- **Budget allocation**: How many evaluations available
- **Heuristic portfolio**: Quality and diversity of heuristics
- **Parallelism**: Number of simultaneous evaluations

Complexity Analysis
~~~~~~~~~~~~~~~~~~~

**Per iteration:**

- Sorting results: :math:`O(N \log N)` where :math:`N` = evaluations so far
- Pareto updates: :math:`O(N \cdot |\text{Front}|)` worst case, typically :math:`O(\log N)`
- Event dispatching: :math:`O(|H| + |A|)` where :math:`|H|` = heuristics, :math:`|A|` = analyzers

**Total:**

- Function evaluations: :math:`O(N_{max})` (the actual bottleneck)
- Overhead: :math:`O(N_{max}^2 \log N_{max})` worst case, typically :math:`O(N_{max} \log N_{max})`

**Parallelism:**

- Up to :math:`P` evaluations simultaneously (Dask cluster size)
- Wall-clock time: :math:`O(N_{max} / P)` assuming load balance

References
----------

.. [Auer2002] Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).
   Finite-time analysis of the multiarmed bandit problem.
   *Machine Learning*, 47(2), 235-256.

.. [Huyer2008] Huyer, W., & Neumaier, A. (2008).
   SNOBFIT–stable noisy optimization by branch and fit.
   *ACM Trans. Math. Softw.*, 35(2), 1-25.

.. [Jones1993] Jones, D. R., Perttunen, C. D., & Stuckman, B. E. (1993).
   Lipschitzian optimization without the Lipschitz constant.
   *J. Opt. Theory Appl.*, 79(1), 157-181.

.. [Mockus1989] Mockus, J. (1989).
   *Bayesian Approach to Global Optimization*. Springer.
