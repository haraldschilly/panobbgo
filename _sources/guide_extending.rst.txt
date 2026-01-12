Extending Panobbgo
==================

This guide shows how to extend Panobbgo with custom components.

Adding a Custom Heuristic
--------------------------

Basic Template
~~~~~~~~~~~~~~

.. code-block:: python

   from panobbgo.core import Heuristic
   from panobbgo.lib import Point
   import numpy as np

   class MyHeuristic(Heuristic):
       """Brief description of what this heuristic does."""

       def __init__(self, strategy, my_param=1.0):
           """Initialize with parameters.

           Args:
               strategy: The strategy instance
               my_param: Custom parameter
           """
           super().__init__(strategy)
           self.my_param = my_param

       def on_start(self):
           """Called when optimization starts.

           Generate initial points here.
           """
           for i in range(10):
               x = self.problem.random_point()
               self.emit(Point(x, self.name))

       def on_new_best(self, best):
           """Called when a new best point is found.

           Args:
               best: The new best Result object
           """
           # Generate points near the new best
           for i in range(5):
               x = best.x + 0.1 * np.random.randn(self.problem.dim)
               x = self.problem.project(x)  # Ensure in bounding box
               self.emit(Point(x, self.name))

Example: Gradient Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A heuristic that approximates gradients using finite differences:

.. code-block:: python

   class GradientSampling(Heuristic):
       """Samples along estimated gradient directions."""

       def __init__(self, strategy, epsilon=1e-5, num_samples=5):
           super().__init__(strategy)
           self.epsilon = epsilon
           self.num_samples = num_samples
           self.last_best_x = None

       def on_start(self):
           """No initial points."""
           pass

       def on_new_best(self, best):
           """Generate points for finite difference gradient."""
           # Avoid re-processing same point
           if self.last_best_x is not None and \
              np.allclose(best.x, self.last_best_x):
               return

           self.last_best_x = best.x.copy()

           # Finite difference in each dimension
           for i in range(self.problem.dim):
               ei = np.zeros(self.problem.dim)
               ei[i] = self.epsilon

               # Forward and backward
               self.emit(Point(best.x + ei, self.name))
               self.emit(Point(best.x - ei, self.name))

       def on_new_results(self, results):
           """Approximate gradient and sample along it."""
           # Only process if we have gradient estimates
           if self.last_best_x is None:
               return

           # Find results near last_best_x
           nearby = [r for r in results
                    if np.linalg.norm(r.x - self.last_best_x) < 2*self.epsilon]

           if len(nearby) < self.problem.dim * 2:
               return  # Not enough points yet

           # Approximate gradient
           grad = np.zeros(self.problem.dim)
           for i in range(self.problem.dim):
               ei = np.zeros(self.problem.dim)
               ei[i] = self.epsilon

               # Find forward and backward points
               fwd = [r for r in nearby
                     if np.allclose(r.x, self.last_best_x + ei)]
               bwd = [r for r in nearby
                     if np.allclose(r.x, self.last_best_x - ei)]

               if fwd and bwd:
                   grad[i] = (fwd[0].fx - bwd[0].fx) / (2 * self.epsilon)

           # Sample along negative gradient direction
           step_size = self.problem.ranges.mean() * 0.1
           for alpha in np.linspace(0.1, 2.0, self.num_samples):
               x_new = self.last_best_x - alpha * step_size * grad
               x_new = self.problem.project(x_new)
               self.emit(Point(x_new, self.name))

Example: Particle Swarm Component
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ParticleSwarmHeuristic(Heuristic):
       """Generates points using particle swarm dynamics."""

       def __init__(self, strategy, n_particles=10, inertia=0.7,
                   cognitive=1.5, social=1.5):
           super().__init__(strategy)
           self.n_particles = n_particles
           self.inertia = inertia
           self.cognitive = cognitive
           self.social = social

           self.particles = None
           self.velocities = None
           self.personal_best = None
           self.global_best = None

       def on_start(self):
           """Initialize particle swarm."""
           # Random initial positions
           self.particles = np.array([
               self.problem.random_point()
               for _ in range(self.n_particles)
           ])

           # Random initial velocities
           ranges = self.problem.ranges
           self.velocities = np.array([
               np.random.uniform(-ranges, ranges) * 0.1
               for _ in range(self.n_particles)
           ])

           # Personal best = initial positions
           self.personal_best = self.particles.copy()

           # Emit initial positions for evaluation
           for x in self.particles:
               self.emit(Point(x, self.name))

       def on_new_results(self, results):
           """Update particles based on new results."""
           if self.particles is None:
               return

           # Update global best from all results
           best_result = self.strategy.best
           if best_result is not None:
               self.global_best = best_result.x

           # Update velocities and positions
           for i in range(self.n_particles):
               # Random factors
               r_cognitive = np.random.rand(self.problem.dim)
               r_social = np.random.rand(self.problem.dim)

               # Update velocity
               self.velocities[i] = (
                   self.inertia * self.velocities[i] +
                   self.cognitive * r_cognitive * (self.personal_best[i] - self.particles[i]) +
                   self.social * r_social * (self.global_best - self.particles[i])
               )

               # Update position
               self.particles[i] = self.particles[i] + self.velocities[i]
               self.particles[i] = self.problem.project(self.particles[i])

               # Emit new position
               self.emit(Point(self.particles[i], self.name))

Registering Your Heuristic
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # In your optimization script
   from my_module import GradientSampling

   strategy = StrategyRewarding(problem, max_evaluations=1000)
   strategy.add(GradientSampling, epsilon=1e-4, num_samples=10)
   strategy.add(Random)  # Include other heuristics
   strategy.start()

Adding a Custom Analyzer
-------------------------

Basic Template
~~~~~~~~~~~~~~

.. code-block:: python

   from panobbgo.core import Analyzer

   class MyAnalyzer(Analyzer):
       """Brief description of what this analyzer does."""

       def __init__(self, strategy):
           super().__init__(strategy)
           # Initialize internal state
           self.state = {}

       def __start__(self):
           """Called when optimization starts."""
           self.state = {'count': 0}

       def on_new_results(self, results):
           """Process new results.

           Args:
               results: List of Result objects
           """
           self.state['count'] += len(results)

           # Analyze results...
           # Optionally publish events
           if self.state['count'] > 100:
               self.eventbus.publish('threshold_reached')

Example: Convergence Detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detect when optimization has stagnated. Note that a built-in `Convergence` analyzer
is now available in `panobbgo.analyzers.convergence`, but here is how you could implement
a simple version yourself:

.. code-block:: python

   class ConvergenceDetector(Analyzer):
       """Detects convergence and publishes converged event."""

       def __init__(self, strategy, window_size=50, tolerance=1e-6):
           super().__init__(strategy)
           self.window_size = window_size
           self.tolerance = tolerance
           self.converged = False

       def on_new_results(self, results):
           """Check for convergence."""
           if self.converged:
               return

           df = self.results.results
           if len(df) < self.window_size:
               return

           # Get recent objective values
           recent_fx = df[('fx', 0)].tail(self.window_size)

           # Check standard deviation
           std = recent_fx.std()
           if std < self.tolerance:
               self.converged = True
               self.logger.info(f"Convergence detected! std={std:.2e}")
               self.eventbus.publish('converged')

       def on_converged(self):
           """React to convergence (even if detected by another instance)."""
           self.logger.info("Optimization has converged")

The built-in `Convergence` analyzer supports two modes: 'std' (standard deviation) and
'improv' (relative improvement). It can be added to your strategy like this:

.. code-block:: python

   from panobbgo.analyzers.convergence import Convergence
   strategy.add_analyzer(Convergence, window_size=50, threshold=1e-6, mode='std')

Example: Clustering Analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Identify clusters of good points:

.. code-block:: python

   from sklearn.cluster import DBSCAN

   class ClusterAnalyzer(Analyzer):
       """Identifies clusters of good solutions."""

       def __init__(self, strategy, eps=0.5, min_samples=5,
                   top_fraction=0.1):
           super().__init__(strategy)
           self.eps = eps
           self.min_samples = min_samples
           self.top_fraction = top_fraction
           self.clusters = []

       def on_new_results(self, results):
           """Cluster the best points."""
           df = self.results.results

           if len(df) < 50:
               return  # Need minimum data

           # Get top fraction of results
           n_top = max(10, int(len(df) * self.top_fraction))
           top_indices = df[('fx', 0)].nsmallest(n_top).index

           # Extract coordinates
           X = df.loc[top_indices, [('x', i) for i in range(self.problem.dim)]].values

           # Cluster
           clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
           labels = clustering.fit_predict(X)

           # Identify cluster centers
           n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
           self.clusters = []

           for cluster_id in range(n_clusters):
               mask = labels == cluster_id
               cluster_points = X[mask]
               center = cluster_points.mean(axis=0)
               self.clusters.append(center)

           # Publish event
           if n_clusters > 0:
               self.eventbus.publish('clusters_found',
                                    centers=self.clusters,
                                    n_clusters=n_clusters)

Registering Your Analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from my_module import ConvergenceDetector, ClusterAnalyzer

   strategy = StrategyRewarding(problem, max_evaluations=1000)
   strategy.add_analyzer(ConvergenceDetector, window_size=50, tolerance=1e-6)
   strategy.add_analyzer(ClusterAnalyzer, eps=0.5)
   # ... add heuristics ...
   strategy.start()

Adding a Custom Strategy
-------------------------

Basic Template
~~~~~~~~~~~~~~

.. code-block:: python

   from panobbgo.core import StrategyBase

   class MyStrategy(StrategyBase):
       """Brief description of strategy."""

       def __init__(self, problem, **kwargs):
           super().__init__(problem, **kwargs)
           # Initialize strategy-specific state

       def execute(self):
           """Generate next batch of points to evaluate.

           Returns:
               List of Point objects
           """
           points = []

           # Your logic to select heuristics and get points
           for h in self.heuristics:
               pts = h.get_points(limit=10)
               points.extend(pts)

           return points

Example: Epsilon-Greedy Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import random

   class StrategyEpsilonGreedy(StrategyBase):
       """Epsilon-greedy heuristic selection."""

       def __init__(self, problem, epsilon=0.1, **kwargs):
           super().__init__(problem, **kwargs)
           self.epsilon = epsilon

       def __start__(self):
           """Initialize performance tracking."""
           for h in self.heuristics:
               h.performance = 1.0
               h.n_selected = 0

       def execute(self):
           """Select heuristics epsilon-greedily."""
           points = []
           target = self.jobs_per_client * len(self.evaluators)

           while len(points) < target:
               if random.random() < self.epsilon:
                   # Explore: random heuristic
                   h = random.choice(self.heuristics)
               else:
                   # Exploit: best performing heuristic
                   h = max(self.heuristics, key=lambda h: h.performance)

               h.n_selected += 1
               pts = h.get_points(limit=max(1, target // len(self.heuristics)))
               points.extend(pts)

           return points

       def on_new_best(self, best):
           """Reward the heuristic that found the best."""
           h = self.heuristic(best.who)
           h.performance += 1.0

Defining a Custom Problem
--------------------------

Problem with External Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import subprocess
   import tempfile
   import os
   from panobbgo.lib import Problem, BoundingBox

   class CFDSimulation(Problem):
       """Aerodynamic optimization using CFD simulation."""

       def __init__(self):
           # 5 design parameters
           dim = 5
           box = BoundingBox(np.array([
               [0.1, 1.0],   # Parameter 1
               [0.1, 1.0],   # Parameter 2
               [0.0, 90.0],  # Parameter 3 (angle)
               [10.0, 100.0],  # Parameter 4
               [0.5, 2.0]    # Parameter 5
           ]))
           super().__init__(dim, box)

       def eval(self, x):
           """Run CFD simulation and return drag coefficient."""
           # Create temporary directory for simulation
           with tempfile.TemporaryDirectory() as tmpdir:
               # Write input file
               input_file = os.path.join(tmpdir, 'input.dat')
               with open(input_file, 'w') as f:
                   f.write(f"{x[0]} {x[1]} {x[2]} {x[3]} {x[4]}\n")

               # Run simulation
               try:
                   result = subprocess.run(
                       ['./cfd_solver', input_file],
                       cwd=tmpdir,
                       capture_output=True,
                       text=True,
                       timeout=600  # 10 minute timeout
                   )

                   # Parse output
                   output_file = os.path.join(tmpdir, 'output.dat')
                   with open(output_file, 'r') as f:
                       drag = float(f.readline().strip())

                   return drag

               except subprocess.TimeoutExpired:
                   self.logger.warning(f"Simulation timeout at x={x}")
                   return 1e10  # Penalty value

               except Exception as e:
                   self.logger.error(f"Simulation failed: {e}")
                   return 1e10

Problem with Complex Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class StructuralDesign(Problem):
       """Structural optimization with stress and deflection constraints."""

       def __init__(self, max_stress=100.0, max_deflection=0.01):
           dim = 8  # 8 structural members
           box = BoundingBox(np.array([[0.001, 0.1]] * dim))  # Cross-sections
           super().__init__(dim, box)

           self.max_stress = max_stress
           self.max_deflection = max_deflection

       def eval(self, x):
           """Minimize weight of structure."""
           density = 7850  # kg/m^3 (steel)
           length = 1.0    # m

           # Weight = sum of (area * length * density)
           weight = sum(x) * length * density
           return weight

       def eval_constraints(self, x):
           """Compute stress and deflection violations."""
           # Simplified structural analysis (replace with FEA)
           stress = self._compute_stress(x)
           deflection = self._compute_deflection(x)

           # Constraint violations
           g1 = stress - self.max_stress
           g2 = deflection - self.max_deflection

           return np.array([max(0, g1), max(0, g2)])

       def _compute_stress(self, x):
           """Compute maximum stress (simplified)."""
           force = 10000  # N
           min_area = x.min()
           stress = force / min_area if min_area > 0 else 1e10
           return stress

       def _compute_deflection(self, x):
           """Compute maximum deflection (simplified)."""
           E = 200e9  # Pa (Young's modulus)
           force = 10000
           I = sum(x**4) / 12  # Moment of inertia
           deflection = force / (E * I) if I > 0 else 1e10
           return deflection

Custom Events and Communication
--------------------------------

Publishing Custom Events
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MyHeuristic(Heuristic):
       def on_new_results(self, results):
           # Detect interesting pattern
           if len(results) > 100:
               best_10 = sorted(results, key=lambda r: r.fx)[:10]
               self.eventbus.publish('top_ten_found', results=best_10)

Subscribing to Custom Events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MyAnalyzer(Analyzer):
       def on_top_ten_found(self, results):
           """React to custom event."""
           self.logger.info(f"Received top 10 results")
           # Process results...

Integration with External Tools
--------------------------------

Exporting Results to CSV
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # After optimization
   df = strategy.results.results
   df.to_csv('results.csv', index=False)

Loading Previous Results
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from panobbgo.lib import Result, Point

   # Load CSV
   df = pd.read_csv('previous_results.csv')

   # Convert to Result objects
   results = []
   for _, row in df.iterrows():
       x = row[[f'x_{i}' for i in range(dim)]].values
       fx = row['fx']
       point = Point(x, who='loaded')
       result = Result(point, fx, cv_vec=None)
       results.append(result)

   # Add to new optimization
   strategy.results.add_results(results)

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   from mpl_toolkits.mplot3d import Axes3D

   def plot_2d_results(strategy):
       """Plot results for 2D problem."""
       df = strategy.results.results

       x0 = df[('x', 0)].values
       x1 = df[('x', 1)].values
       fx = df[('fx', 0)].values

       plt.figure(figsize=(10, 8))
       scatter = plt.scatter(x0, x1, c=fx, cmap='viridis',
                           s=50, alpha=0.6)
       plt.colorbar(scatter, label='f(x)')

       # Mark best point
       best = strategy.best
       plt.scatter(best.x[0], best.x[1], c='red',
                  marker='*', s=500, edgecolors='black',
                  label='Best')

       plt.xlabel('x0')
       plt.ylabel('x1')
       plt.title('Optimization Results')
       plt.legend()
       plt.grid(True, alpha=0.3)
       plt.show()

Best Practices
--------------

For Heuristics
~~~~~~~~~~~~~~

1. **Respect the queue capacity**: Don't emit too many points at once
2. **Use event handlers appropriately**: ``on_start`` for initialization, ``on_new_best`` for exploitation
3. **Handle edge cases**: Empty queues, degenerate geometries
4. **Log informatively**: Help users understand what the heuristic is doing
5. **Make it configurable**: Parameters should be adjustable

For Analyzers
~~~~~~~~~~~~~

1. **Be efficient**: Analyzers run on every result update
2. **Maintain consistent state**: Thread-safe if needed
3. **Publish meaningful events**: Help heuristics make better decisions
4. **Document events**: Describe parameters and semantics

For Strategies
~~~~~~~~~~~~~~

1. **Balance heuristics**: Ensure all get chances to contribute
2. **Respect budget**: Don't request more points than needed
3. **Handle edge cases**: No heuristics, empty queues
4. **Track performance**: Enable adaptive behavior

Testing Your Extensions
-----------------------

.. code-block:: python

   import pytest
   from panobbgo.lib.classic import Rosenbrock
   from panobbgo.strategies.round_robin import StrategyRoundRobin
   from my_module import MyHeuristic

   def test_my_heuristic():
       """Test custom heuristic."""
       problem = Rosenbrock(dim=3)
       strategy = StrategyRoundRobin(problem, max_evaluations=100)

       # Add heuristic
       strategy.add(MyHeuristic, my_param=2.0)

       # Get heuristic instance
       h = strategy.heuristics[0]
       assert h.my_param == 2.0

       # Trigger start
       h.__start__()

       # Check point generation
       points = h.get_points(10)
       assert len(points) <= 10
       assert all(isinstance(p, Point) for p in points)

       # Check bounds
       for p in points:
           assert problem.box.contains(p.x)

Contributing Back
-----------------

If you develop useful extensions, consider contributing them to Panobbgo:

1. Fork the repository: https://github.com/haraldschilly/panobbgo
2. Create a feature branch
3. Add your component with tests and documentation
4. Submit a pull request

See the project's CONTRIBUTING guide for details.
