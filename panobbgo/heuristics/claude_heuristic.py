# -*- coding: utf8 -*-
# Copyright 2026 Panobbgo Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Claude Heuristic — Cluster-Based Adaptive Search
=================================================

This heuristic identifies clusters of elite (top-performing) evaluated points,
fits a local Gaussian distribution to each cluster, and samples new candidates
from the resulting mixture model.

The intuition: near groups of good points, even better points may be hiding.
By capturing each cluster's local geometry (elongated valleys, ridges) via
the empirical covariance, we sample efficiently in promising directions.

Related to the Cross-Entropy Method and Estimation of Distribution Algorithms.
"""

from panobbgo.core import Heuristic
import numpy as np


class ClaudeHeuristic(Heuristic):
    """
    Cluster-based adaptive search heuristic (written by Claude, hence the name).

    Fits a Mixture of Gaussians to the top-performing evaluated points,
    then samples new candidates from the mixture. Mixture weights favor
    clusters with lower penalty values (better objective + feasibility).

    Args:
        strategy: The optimization strategy instance.
        elite_fraction: Fraction of all results to consider "elite" (default 0.2).
        max_clusters: Maximum number of clusters (default 5).
        n_candidates: Number of points to emit per trigger (default 5).
        min_points: Minimum total results before activating (default 2*dim+1).
        regularization: Covariance regularization strength (default 1e-3).
    """

    def __init__(
        self,
        strategy,
        elite_fraction=0.2,
        max_clusters=5,
        n_candidates=5,
        min_points=None,
        regularization=1e-3,
    ):
        super().__init__(strategy, name="ClaudeHeuristic")
        self.logger = self.config.get_logger("H:CL")
        self.elite_fraction = elite_fraction
        self.max_clusters = max_clusters
        self.n_candidates = n_candidates
        self.regularization = regularization
        self._min_points_cfg = min_points
        self._min_points = 5  # will be set properly in on_start

        # Incremental data storage
        self.X_all: np.ndarray | None = None
        self.y_all: np.ndarray | None = None

    def on_start(self):
        """Reset state at start of optimization."""
        self.X_all = None
        self.y_all = None
        if self._min_points_cfg is not None:
            self._min_points = self._min_points_cfg
        else:
            self._min_points = 2 * self.problem.dim + 1

    def on_new_results(self, results):
        """Accumulate results, rebuild clusters, sample new candidates."""
        if not results:
            return

        if not self._accumulate(results):
            return

        assert self.y_all is not None
        if len(self.y_all) < self._min_points:
            return

        X_elite, y_elite = self._select_elite()
        if len(X_elite) < 2:
            return

        components = self._compute_clusters(X_elite, y_elite)
        if not components:
            return

        candidates = self._sample_from_mixture(components)

        self.clear_output()
        self.emit(candidates)

    def _accumulate(self, results) -> bool:
        """Append new results to stored arrays. Returns True if any added."""
        get_val = self.strategy.constraint_handler.get_penalty_value
        new_X = []
        new_y = []
        for r in results:
            if r.fx is not None and np.isfinite(r.fx):
                new_X.append(r.x)
                new_y.append(get_val(r))

        if not new_X:
            return False

        new_X_arr = np.array(new_X)
        new_y_arr = np.array(new_y)

        if self.X_all is None:
            self.X_all = new_X_arr
            self.y_all = new_y_arr
        else:
            self.X_all = np.vstack([self.X_all, new_X_arr])
            assert self.y_all is not None
            self.y_all = np.concatenate([self.y_all, new_y_arr])

        return True

    def _select_elite(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X_elite, y_elite) — top elite_fraction by penalty value."""
        assert self.y_all is not None and self.X_all is not None
        n_elite = max(2, int(len(self.y_all) * self.elite_fraction))
        n_elite = min(n_elite, len(self.y_all))
        idx = np.argpartition(self.y_all, n_elite)[:n_elite]
        return self.X_all[idx], self.y_all[idx]

    def _compute_clusters(self, X_elite, y_elite):
        """
        Cluster elite points via k-means, fit a Gaussian per cluster,
        and assign mixture weights inversely proportional to mean penalty.

        Returns list of dicts: [{"mean": ndarray, "cov": ndarray, "weight": float}, ...]
        """
        dim = self.problem.dim
        n_elite = len(X_elite)

        # Determine number of clusters
        k = min(self.max_clusters, max(1, n_elite // max(1, 2 * dim)))

        if k <= 1 or n_elite < 2 * dim:
            # Single Gaussian fallback
            mean, cov = self._fit_cluster_gaussian(X_elite)
            return [{"mean": mean, "cov": cov, "weight": 1.0}]

        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(X_elite)

        components = []
        raw_penalties = []

        for i in range(k):
            mask = labels == i
            X_c = X_elite[mask]
            y_c = y_elite[mask]

            if len(X_c) < 2:
                # Degenerate cluster: use point as mean, regularization-only cov
                mean = X_c[0] if len(X_c) == 1 else kmeans.cluster_centers_[i]
                cov = self.regularization * np.diag(self.problem.ranges**2)
            else:
                mean, cov = self._fit_cluster_gaussian(X_c)

            components.append({"mean": mean, "cov": cov})
            raw_penalties.append(float(np.mean(y_c)) if len(y_c) > 0 else np.inf)

        # Compute mixture weights: lower penalty → higher weight
        raw_penalties_arr = np.array(raw_penalties)
        shifted = raw_penalties_arr - raw_penalties_arr.min() + 1e-8
        inv_weights = 1.0 / shifted
        total = inv_weights.sum()
        for i, comp in enumerate(components):
            comp["weight"] = float(inv_weights[i] / total)

        return components

    def _fit_cluster_gaussian(self, X_cluster) -> tuple[np.ndarray, np.ndarray]:
        """Fit a regularized Gaussian to a cluster of points."""
        mean = np.mean(X_cluster, axis=0)

        if len(X_cluster) < 2:
            cov = self.regularization * np.diag(self.problem.ranges**2)
            return mean, cov

        cov = np.cov(X_cluster, rowvar=False, ddof=0)

        # np.cov returns scalar for 1D — reshape to (1,1)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

        # Regularization for numerical stability
        cov = cov + self.regularization * np.diag(self.problem.ranges**2)

        return mean, cov

    def _sample_from_mixture(self, components) -> list[np.ndarray]:
        """Draw n_candidates points from the mixture model, projected to box."""
        candidates: list[np.ndarray] = []
        weights = np.array([c["weight"] for c in components])

        # Multinomial allocation of samples across components
        counts = np.random.multinomial(self.n_candidates, weights)

        for comp, count in zip(components, counts):
            if count == 0:
                continue
            try:
                samples = np.random.multivariate_normal(
                    comp["mean"], comp["cov"], size=int(count)
                )
            except np.linalg.LinAlgError:
                # Fallback to isotropic sampling at mean
                self.logger.debug("Singular covariance, falling back to isotropic")
                scale = np.sqrt(self.regularization) * self.problem.ranges
                samples = comp["mean"] + scale * np.random.randn(int(count), len(comp["mean"]))

            for s in samples:
                candidates.append(self.problem.project(s))

        return candidates
