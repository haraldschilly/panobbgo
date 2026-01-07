# Panobbgo: A Comprehensive Guide to Black-Box Global Optimization

**Version:** 0.0.1 (Alpha)
**Author:** Harald Schilly
**License:** Apache 2.0

---

## Table of Contents

1. [What is Panobbgo?](#what-is-panobbgo)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Core Concepts](#core-concepts)
4. [Architecture Overview](#architecture-overview)
5. [Key Components](#key-components)
6. [Usage Guide](#usage-guide)
7. [Extending Panobbgo](#extending-panobbgo)
8. [Research Context](#research-context)
9. [Future Directions](#future-directions)

---

## What is Panobbgo?

**Panobbgo** (Parallel Noisy Black-Box Global Optimization) is a Python framework for optimizing black-box functions where:

- **Black-box**: You can only evaluate the function, not analyze its structure
- **Global**: You seek the global optimum, not just local minima
- **Noisy**: Function evaluations may be stochastic or have errors
- **Constrained**: You can handle constraint violations
- **Parallel**: Evaluations can run simultaneously on distributed clusters

### The Black-Box Optimization Problem

Given:
- An unknown function $f: \mathbb{R}^n \rightarrow \mathbb{R}$
- A bounding box $B = [x_1^{min}, x_1^{max}] \times \cdots \times [x_n^{min}, x_n^{max}]$
- Optional constraint functions $g_i(x) \leq 0$ for $i = 1, \ldots, m$
- A limited evaluation budget $N_{max}$

Find:
$$
x^* = \arg\min_{x \in B} f(x) \quad \text{subject to} \quad g_i(x) \leq 0, \; \forall i
$$

### Real-World Challenges

1. **Expensive evaluations**: Each $f(x)$ might take minutes to hours
2. **No gradients**: Cannot use derivative-based methods
3. **Noise**: $f(x + \epsilon) \approx f(x)$ but not exactly
4. **Unknown smoothness**: $f$ might be discontinuous or highly multimodal
5. **Limited budget**: Can only afford 100-10,000 evaluations

### Panobbgo's Solution

Panobbgo addresses these challenges through:

1. **Multiple search heuristics** working in parallel
2. **Adaptive strategy selection** (multi-armed bandit approach)
3. **Result database** to avoid re-evaluations and learn from history
4. **Event-driven architecture** for modular, extensible design
5. **IPython cluster integration** for distributed parallel evaluation

---

## Mathematical Foundation

### Problem Formulation

**Objective Function:**
$$
\min_{x \in B \subset \mathbb{R}^n} f(x)
$$

where:
- $B = \{x \in \mathbb{R}^n : l_i \leq x_i \leq u_i, \; i=1,\ldots,n\}$ is the search box
- $f: B \rightarrow \mathbb{R}$ is a potentially noisy, expensive-to-evaluate function

**Constraint Handling:**

If constraints exist, define violation vector:
$$
cv(x) = [g_1(x)_+, \ldots, g_m(x)_+] \quad \text{where} \quad (a)_+ = \max(0, a)
$$

Total constraint violation:
$$
CV(x) = \|cv(x)\|_2 = \sqrt{\sum_{i=1}^m g_i(x)_+^2}
$$

**Lexicographic Ordering:**

Point $x$ is better than $y$ if:
1. $CV(x) < CV(y)$ (less constraint violation), OR
2. $CV(x) = CV(y) = 0$ AND $f(x) < f(y)$ (better objective value)

### The Exploration-Exploitation Tradeoff

Black-box optimization requires balancing:

- **Exploration**: Sample broadly to discover global structure
- **Exploitation**: Sample intensively in promising regions

Panobbgo addresses this through:

1. **Diverse heuristics**: Different exploration/exploitation profiles
2. **Adaptive selection**: Reward heuristics that find improvements
3. **Hierarchical decomposition**: Split search space based on results

### Multi-Armed Bandit Strategy

The selection of which heuristic to use next is modeled as a **multi-armed bandit problem**:

- Each heuristic is an "arm" of a bandit machine
- "Pulling an arm" = requesting a point from that heuristic
- "Reward" = improvement in objective value

**Implementation (StrategyRewarding):**

Each heuristic $h$ maintains a performance score $p_h(t)$ updated as:

$$
p_h(t+1) = \begin{cases}
p_h(t) + R(x) & \text{if } h \text{ generated } x \text{ and } f(x) < f_{best} \\
p_h(t) \cdot d & \text{if } h \text{ generated a point} \\
p_h(t) & \text{otherwise}
\end{cases}
$$

where:
- $R(x) = 1 - e^{-(f_{best} - f(x))}$ is the reward (saturates to 1)
- $d \in (0, 1)$ is a discount factor (default: 0.95)

**Selection Probability:**

Heuristic $h$ is selected with probability:
$$
P(h) = \frac{p_h + s}{\sum_{h'} p_{h'} + s \cdot |H|}
$$

where $s$ is an additive smoothing parameter ensuring exploration.

---

## Core Concepts

### 1. Points and Results

**Point**: A candidate solution to evaluate
- Contains: $x \in \mathbb{R}^n$ (location) and `who` (heuristic name)
- Created by heuristics, queued for evaluation

**Result**: Outcome of evaluating a Point
- Contains: $x$, $f(x)$, $cv(x)$, error estimate, timestamp
- Stored in Results database
- Triggers events for analyzers and heuristics

### 2. Heuristics (Point Generators)

Heuristics generate candidate points using different strategies:

| Type | Heuristics | Strategy |
|------|-----------|----------|
| **Initialization** | Center, Zero | Start at known points |
| **Random Sampling** | Random, Extremal, Latin Hypercube | Explore broadly |
| **Local Search** | Nearby, Weighted Average | Refine around best |
| **Model-Based** | Quadratic WLS | Fit surrogate, optimize |
| **Classical Optimizers** | Nelder-Mead, L-BFGS-B | Local optimization |

**Key Properties:**
- Each maintains an output queue of Points
- React to events (new_best, new_results, new_split, etc.)
- Can be parameterized (e.g., `LatinHypercube(div=5)`)
- Performance tracked for adaptive selection

### 3. Analyzers (Result Processors)

Analyzers process results and maintain derived information:

| Analyzer | Maintains | Publishes |
|----------|-----------|-----------|
| **Best** | Best points, Pareto front | `new_best`, `new_min`, `new_cv`, `new_pareto` |
| **Splitter** | Hierarchical box decomposition | `new_split`, identifies best leaf box |
| **Grid** | Spatial grid grouping | Grid-based neighborhoods |
| **Dedensifyer** | Hierarchical grid | Min/max representatives per region |

### 4. Strategies (Orchestration)

Strategies coordinate the optimization process:

**StrategyRoundRobin**:
- Cycles through heuristics in fixed order
- Simple, predictable, no adaptation

**StrategyRewarding** (Recommended):
- Implements multi-armed bandit
- Rewards heuristics that find improvements
- Adapts selection based on performance

### 5. EventBus (Communication)

The EventBus enables decoupled communication:

```
Strategy evaluates points → Results added to database
                          ↓
                 EventBus publishes "new_results"
                          ↓
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
    Analyzer         Analyzer          Heuristic
    (Best)          (Splitter)         (Random)
        ↓                 ↓
    publishes         publishes
   "new_best"        "new_split"
        ↓                 ↓
    Heuristic         Heuristic
  (NelderMead)       (Nearby)
```

**Event Types:**
- `start`: Optimization begins
- `new_results`: New evaluations available
- `new_best`: New best point found
- `new_split`: Box split occurred
- `finished`: Budget exhausted

---

## Architecture Overview

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                         Strategy                            │
│  ┌─────────────┐      ┌──────────────┐     ┌─────────────┐│
│  │  Heuristic  │─────▶│ Point Queue  │────▶│  Evaluator  ││
│  │ (generates) │      │              │     │ (IPython    ││
│  └─────────────┘      └──────────────┘     │  cluster)   ││
│         ▲                                   └──────┬──────┘│
│         │                                          │       │
│         │                                          ▼       │
│  ┌──────┴──────┐       ┌──────────────┐    ┌──────────┐  │
│  │  Analyzer   │◀──────│  EventBus    │◀───│ Results  │  │
│  │ (processes) │       │ (publishes)  │    │ Database │  │
│  └─────────────┘       └──────────────┘    └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Class Hierarchy

```
Module (base class)
├── Heuristic (point generators)
│   ├── Center
│   ├── Random
│   ├── LatinHypercube
│   ├── Nearby
│   ├── NelderMead
│   └── QuadraticWlsModel
└── Analyzer (result processors)
    ├── Best
    ├── Splitter
    ├── Grid
    └── Dedensifyer

StrategyBase (orchestrator)
├── StrategyRoundRobin
└── StrategyRewarding

Results (database)
EventBus (communication)
```

### Execution Flow

1. **Initialization**
   ```python
   strategy = StrategyRewarding(problem, size=10)
   strategy.add(LatinHypercube, div=3)
   strategy.add(Random)
   strategy.add(NelderMead)
   ```

2. **Start**
   - Strategy calls `__start__()` on all modules
   - EventBus publishes `start` event
   - Heuristics initialize (e.g., Latin Hypercube generates initial grid)

3. **Main Loop**
   ```
   while budget_remaining:
       points = strategy.execute()  # Get points from heuristics
       results = evaluate_parallel(points)  # Send to IPython cluster
       results_db.add_results(results)  # Store in database
       eventbus.publish("new_results", results)  # Notify modules
   ```

4. **Event Cascade**
   - `new_results` → Analyzers update (Best checks for improvements, Splitter updates boxes)
   - Analyzers publish derived events (`new_best`, `new_split`)
   - Heuristics respond to events (NelderMead starts on `new_best`, Random samples from best box)

5. **Termination**
   - Budget exhausted (`len(results) >= max_evaluations`)
   - EventBus publishes `finished`
   - Strategy returns best result

---

## Key Components

### 1. Problem Definition

**Abstract Base Class:**
```python
class Problem:
    dim: int                    # Dimensionality
    box: BoundingBox           # Search space

    def eval(self, x: np.ndarray) -> float:
        """Evaluate objective function"""
        raise NotImplementedError

    def eval_constraints(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Evaluate constraint violations (optional)"""
        return None

    def __call__(self, point: Point) -> Result:
        """Evaluate and return Result object"""
        fx = self.eval(point.x)
        cv_vec = self.eval_constraints(point.x)
        return Result(point, fx, cv_vec)
```

**22 Benchmark Problems Included:**
- Classic: Rosenbrock, Rastrigin, Himmelblau, Beale, Branin
- Constrained: RosenbrockConstraint, RosenbrockAbsConstraint
- Stochastic: RosenbrockStochastic
- High-dimensional: Rastrigin (arbitrary dim), Quadruple, Powell

### 2. Heuristics in Detail

#### Random
- Samples uniformly from best leaf box (identified by Splitter)
- Pure exploration
- Always active (infinite queue)

#### Latin Hypercube
- Stratified sampling: divides each dimension into `div` intervals
- Ensures good coverage across all dimensions
- Runs once at start, emits all points

#### Nearby
- Perturbs best point with random noise
- Gaussian perturbation scaled by box ranges
- Local exploitation

#### Nelder-Mead
- Randomized simplex method
- Triggered by `new_best` event
- Runs in separate thread, emits simplex vertices
- Adaptive: reflects, expands, contracts

#### Quadratic WLS Model
- Fits weighted least-squares quadratic model to recent results
- Optimizes surrogate to find next point
- Model-based exploitation

#### L-BFGS-B
- Limited-memory BFGS with box constraints
- Uses SciPy's implementation in subprocess
- Local gradient-free optimization

### 3. Splitter (Box Decomposition)

**Purpose**: Hierarchically partition search space to focus on promising regions.

**Algorithm**:
1. Start with entire bounding box as root
2. When box has sufficient points (threshold: `split_factor * dim`):
   - Split along dimension with largest range
   - Create two child boxes
3. Identify "best leaf box" = leaf with best point
4. Publish `new_split` event
5. Heuristics like Random sample from best leaf

**Tree Structure**:
```
        [0, 10] × [0, 10]
              |
    ┌─────────┴─────────┐
    |                   |
[0,5]×[0,10]      [5,10]×[0,10]
    |                   |
  (leaf)           ┌────┴────┐
                   |         |
              [5,7.5]×[0,10] [7.5,10]×[0,10]
```

### 4. Best Analyzer (Pareto Front)

**Responsibilities**:
- Track best feasible point (min $f(x)$ with $CV(x) = 0$)
- Track best infeasible point (min $CV(x)$)
- Maintain Pareto front of $(f(x), CV(x))$ pairs

**Pareto Update**:
A new result $(f, cv)$ joins the Pareto front if:
$$
\nexists (f', cv') \in \text{Front}: f' \leq f \land cv' \leq cv \land (f' < f \lor cv' < cv)
$$

**Events Published**:
- `new_best`: New best point (considering constraints)
- `new_min`: New minimum $f(x)$ (feasible only)
- `new_cv`: New minimum $CV(x)$
- `new_pareto`: Pareto front updated

### 5. Results Database

**Storage**: Pandas DataFrame with MultiIndex columns:
- $(x_0, x_1, \ldots, x_{n-1})$: Coordinates
- $f(x)$: Objective value
- $(cv_0, cv_1, \ldots, cv_{m-1})$: Individual constraint violations
- $CV$: Total constraint violation (L2 norm)
- `who`: Heuristic name
- `error`: Estimated error margin

**Queries**:
```python
results.results                 # Full DataFrame
results.results['fx']          # Objective values
results.results[results.results[('cv', 0)] == 0]  # Feasible points
len(results)                   # Number of evaluations
```

---

## Usage Guide

### Installation

```bash
# Using UV (recommended)
git clone https://github.com/haraldschilly/panobbgo.git
cd panobbgo
uv sync --extra dev

# Using pip
pip install -e ".[dev]"
```

### Basic Example

```python
from panobbgo.lib.classic import Rosenbrock
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.heuristics import (
    Center, LatinHypercube, Random, Nearby, NelderMead
)

# 1. Define problem
problem = Rosenbrock(dim=5)  # 5-dimensional Rosenbrock

# 2. Create strategy
strategy = StrategyRewarding(
    problem,
    size=10,  # Jobs per client
    max_evaluations=1000
)

# 3. Add heuristics
strategy.add(Center)                    # Start at center
strategy.add(LatinHypercube, div=5)    # Initial space-filling
strategy.add(Random)                    # Ongoing exploration
strategy.add(Nearby)                    # Local refinement
strategy.add(NelderMead)               # Local optimization

# 4. Start IPython cluster
# Terminal: ipcluster start -n 4

# 5. Run optimization
strategy.start()

# 6. Get results
print("Best found:", strategy.best)
print("Best x:", strategy.best.x)
print("Best f(x):", strategy.best.fx)
print("Total evaluations:", len(strategy.results))
```

### Defining Custom Problems

```python
import numpy as np
from panobbgo.lib.lib import Problem, BoundingBox

class MyProblem(Problem):
    def __init__(self):
        dim = 3
        box = BoundingBox(np.array([
            [-5.0, 5.0],   # x1 ∈ [-5, 5]
            [-10.0, 10.0], # x2 ∈ [-10, 10]
            [0.0, 1.0]     # x3 ∈ [0, 1]
        ]))
        super().__init__(dim, box)

    def eval(self, x):
        """Your expensive black-box function"""
        return np.sum(x**2) + 0.1 * np.random.randn()  # Noisy sphere

    def eval_constraints(self, x):
        """Optional: return array of constraint violations"""
        g1 = x[0] + x[1] - 1.0  # x1 + x2 ≤ 1
        g2 = -x[2]              # x3 ≥ 0
        return np.array([max(0, g1), max(0, g2)])
```

### Custom Heuristic

```python
from panobbgo.core import Heuristic
from panobbgo.lib.lib import Point
import numpy as np

class MyHeuristic(Heuristic):
    def on_start(self):
        """Generate initial points"""
        for _ in range(10):
            x = self.problem.random_point()
            self.emit(Point(x, self.name))

    def on_new_best(self, best):
        """React to new best point"""
        # Generate points near the new best
        for _ in range(5):
            x = best.x + 0.1 * np.random.randn(self.problem.dim)
            x = self.problem.project(x)  # Ensure in box
            self.emit(Point(x, self.name))
```

### Configuration

Edit `~/.panobbgo/config.ini`:

```ini
[ipython]
profile = default
max_wait_for_job = 10

[optimization]
max_evaluations = 1000
queue_capacity = 20

[strategy]
smooth = 0.1          # Additive smoothing
discount = 0.95       # Performance decay
jobs_per_client = 5   # Batch size

[logging]
level = INFO
```

---

## Extending Panobbgo

### Adding a New Heuristic

1. **Subclass Heuristic**:
```python
class GradientSampling(Heuristic):
    def __init__(self, strategy, epsilon=1e-5):
        super().__init__(strategy)
        self.epsilon = epsilon
```

2. **Implement Event Handlers**:
```python
    def on_new_best(self, best):
        # Finite difference gradient approximation
        x0 = best.x
        for i in range(self.problem.dim):
            ei = np.zeros(self.problem.dim)
            ei[i] = self.epsilon
            self.emit(Point(x0 + ei, self.name))
            self.emit(Point(x0 - ei, self.name))
```

3. **Register**:
```python
strategy.add(GradientSampling, epsilon=1e-4)
```

### Adding a New Analyzer

1. **Subclass Analyzer**:
```python
from panobbgo.core import Analyzer

class ConvergenceDetector(Analyzer):
    def __init__(self, strategy):
        super().__init__(strategy)
        self.window_size = 50
        self.tolerance = 1e-6
```

2. **Subscribe to Events**:
```python
    def on_new_results(self, results):
        # Check if improvement rate has stalled
        recent = self.results.results['fx'].tail(self.window_size)
        if len(recent) == self.window_size:
            std = recent.std()
            if std < self.tolerance:
                self.eventbus.publish("converged")
```

3. **Use in Strategy**:
```python
from panobbgo.analyzers import Best, Splitter, ConvergenceDetector

strategy.add_analyzer(Best)
strategy.add_analyzer(Splitter)
strategy.add_analyzer(ConvergenceDetector)
```

---

## Research Context

### Inspiration: SNOBFIT

Panobbgo is inspired by [SNOBFIT](http://www.mat.univie.ac.at/~neum/software/snobfit/) (Stable Noisy Optimization by Branch and Fit):

**SNOBFIT Contributions:**
- Quadratic local models with uncertainty quantification
- Branch-and-bound spatial decomposition
- Noise handling through repeated evaluations

**Panobbgo Innovations:**
- **Modular heuristics**: Not tied to single algorithm
- **Multi-armed bandit**: Adaptive heuristic selection
- **Event-driven**: Decoupled, extensible architecture
- **Parallel-first**: IPython cluster integration from the start

### Related Algorithms

| Algorithm | Approach | Panobbgo Equivalent |
|-----------|----------|-------------------|
| **Bayesian Optimization** | Gaussian process surrogate | QuadraticWlsModel (simpler surrogate) |
| **CMA-ES** | Evolution strategy | NelderMead (similar adaptive sampling) |
| **DIRECT** | Lipschitz-based | Splitter (box decomposition) |
| **Random Search** | Pure exploration | Random heuristic |
| **Simulated Annealing** | Probabilistic acceptance | StrategyRewarding (probabilistic selection) |

### Theoretical Properties

**Convergence**: Under mild assumptions (local Lipschitz continuity, sufficient budget), panobbgo converges to global optimum almost surely due to:
1. Latin Hypercube ensures dense initial coverage
2. Random heuristic maintains exploration
3. Splitter focuses on promising regions
4. Bandit strategy adapts to problem structure

**Complexity**:
- Per iteration: $O(N \log N)$ where $N$ = evaluations so far (sorting, Pareto updates)
- Parallelism: Up to $P$ evaluations simultaneously (IPython cluster size)
- Total: $O(N_{max})$ evaluations

---

## Future Directions

### Short-Term Improvements

1. **Better Surrogate Models**
   - Gaussian Processes (GP)
   - Random Forests
   - Neural networks for high dimensions

2. **Advanced Constraint Handling**
   - Penalty methods
   - Augmented Lagrangian
   - Constraint approximation

3. **Persistent Storage**
   - SQLite or PostgreSQL backend
   - Resume optimization from checkpoint
   - Share results across runs

4. **Enhanced Bandit Strategy**
   - UCB (Upper Confidence Bound)
   - Thompson Sampling
   - Contextual bandits (problem features)

### Long-Term Research

1. **Transfer Learning**
   - Learn heuristic performance across problems
   - Meta-learning for strategy initialization

2. **Parallel Batch Generation**
   - Generate batches considering pending evaluations
   - Avoid redundant sampling in parallel

3. **Multi-Fidelity Optimization**
   - Use cheap approximations to guide expensive evaluations
   - Hierarchical accuracy levels

4. **High-Dimensional Scaling**
   - Random embeddings
   - Coordinate descent
   - Dimension reduction

---

## Conclusion

Panobbgo is a **flexible, modular, and extensible framework** for black-box global optimization. Its key strengths are:

✓ **Adaptive**: Multi-armed bandit learns which heuristics work
✓ **Parallel**: Built for distributed evaluation from the start
✓ **Modular**: Easy to add new heuristics, analyzers, strategies
✓ **Robust**: Handles noise, constraints, expensive evaluations
✓ **Principled**: Based on solid mathematical optimization theory

Whether you're optimizing hyperparameters, calibrating simulations, or exploring complex design spaces, panobbgo provides the tools to efficiently find global optima with limited budget.

---

## References

1. Huyer, W., & Neumaier, A. (2008). SNOBFIT–stable noisy optimization by branch and fit. *ACM Trans. Math. Softw.*, 35(2), 1-25.

2. Jones, D. R., Perttunen, C. D., & Stuckman, B. E. (1993). Lipschitzian optimization without the Lipschitz constant. *J. Opt. Theory Appl.*, 79(1), 157-181.

3. Mockus, J. (1989). *Bayesian Approach to Global Optimization*. Springer.

4. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2), 235-256.

5. Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. *Evol. Comput.*, 9(2), 159-195.

---

**For more details, see:**
- API Documentation: http://haraldschilly.github.com/panobbgo/html/
- Source Code: https://github.com/haraldschilly/panobbgo
- Issue Tracker: https://github.com/haraldschilly/panobbgo/issues
