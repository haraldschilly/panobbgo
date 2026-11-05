# NeighborhoodAdaptiveDE, from "NeighborhoodAdaptiveDE for Many-Affine Black-box
# optimization", Anonymous (Zenodo, 2025), doi:10.5281/zenodo.15074959, file
# NeighborhoodAdaptiveDE.py (md5 8802791b458e1f6931bf87b70bd580b1).  CC BY 4.0:
# https://creativecommons.org/licenses/by/4.0/.  Designed with the LLaMEA framework
# (van Stein & Baeck, IEEE TEVC 2024).
#
# Changed: this header, and np.Inf became np.inf (NumPy 2 removed np.Inf).  Nothing
# else; tests/test_mabbob_calibration.py checks that against the md5 above.
#
# Here to calibrate the harness (doc/dev/benchmarking.md, "Official MA-BBOB
# instances"), through harness_baselines.NeighborhoodAdaptiveDEStrategy.
# --- end of header ---
import numpy as np

class NeighborhoodAdaptiveDE:
    def __init__(self, budget_factor = 2000, dim=10, pop_size=50, adapt_freq=50, stagnation_threshold=1000, learning_rate=0.1, neighborhood_size=5):
        self.budget_factor = budget_factor
        self.dim = dim
        self.pop_size = pop_size
        self.adapt_freq = adapt_freq
        self.stagnation_threshold = stagnation_threshold
        self.learning_rate = learning_rate
        self.neighborhood_size = neighborhood_size
        self.f = 0.5
        self.cr = 0.9
        self.success_f = []
        self.success_cr = []
        self.best_fitness_history = []

    def __call__(self, func, seed=0):
        np.random.seed(seed)
        self.f_opt = np.inf
        self.x_opt = None
        self.budget = self.budget_factor * func.meta_data.n_variables
        self.dim = func.meta_data.n_variables #take dim from IOH problem
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.evals = self.pop_size
        self.last_improvement = 0

        best_idx = np.argmin(self.fitness)
        self.f_opt = self.fitness[best_idx]
        self.x_opt = self.population[best_idx]
        self.best_fitness_history.append(self.f_opt)

        while self.evals < self.budget:
            for i in range(self.pop_size):
                # Neighborhood-based Mutation
                neighbors = np.random.choice(np.arange(self.pop_size), self.neighborhood_size, replace=False)
                best_neighbor_idx = neighbors[np.argmin(self.fitness[neighbors])]
                
                idxs = np.random.choice(np.arange(self.pop_size), 2, replace=False)
                x1, x2 = self.population[idxs]

                v = self.population[best_neighbor_idx] + self.f * (x1 - x2)
                v = np.clip(v, func.bounds.lb, func.bounds.ub)
                
                # Orthogonal Crossover
                u = np.copy(self.population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == j_rand:
                        u[j] = v[j]
                
                # Evaluation
                f_new = func(u)
                self.evals += 1
                
                # Selection
                if f_new < self.fitness[i]:
                    self.success_f.append(self.f)
                    self.success_cr.append(self.cr)
                    self.fitness[i] = f_new
                    self.population[i] = u

                    if f_new < self.f_opt:
                        self.f_opt = f_new
                        self.x_opt = u
                        self.best_fitness_history.append(self.f_opt)
                        self.last_improvement = self.evals

            # Adaptive Parameter Control
            if self.evals % self.adapt_freq == 0:
                if self.success_f:
                    self.f = (1 - self.learning_rate) * self.f + self.learning_rate * np.mean(self.success_f)
                    self.cr = (1 - self.learning_rate) * self.cr + self.learning_rate * np.mean(self.success_cr)
                self.f = np.clip(self.f, 0.1, 0.9)
                self.cr = np.clip(self.cr, 0.1, 1.0)
                self.success_f = []
                self.success_cr = []
            
            # Stagnation Check and Restart
            if self.evals - self.last_improvement > self.stagnation_threshold:
                self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                self.fitness = np.array([func(x) for x in self.population])
                self.evals += self.pop_size
                best_idx = np.argmin(self.fitness)
                self.f_opt = self.fitness[best_idx]
                self.x_opt = self.population[best_idx]
                self.last_improvement = self.evals
                self.best_fitness_history.append(self.f_opt)


            if self.evals >= self.budget:
                break
                    
        return self.f_opt, self.x_opt