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
Sensitivity Analyzer
====================

Estimates which input dimensions have the most impact on the objective function,
using the accumulated evaluation history.

Publishes ``new_sensitivity(importance=ndarray)`` events, where ``importance``
is a ``(dim,)``-array with values in ``[0, 1]`` — higher means more important.
"""

from __future__ import unicode_literals

import numpy as np
from panobbgo.core import Analyzer
from scipy.stats import spearmanr


class Sensitivity(Analyzer):
    """
    Estimates per-dimension sensitivity from evaluation history.

    Listens to ``new_results`` events, accumulates X and y (penalty values),
    and periodically computes importance scores for each dimension.

    Configuration parameters:

    - ``min_samples`` (int): Minimum evaluations before first analysis (default: 10 * dim).
    - ``update_interval`` (int): Recompute every N new results (default: 50).
    - ``method`` (str): ``"spearman"`` (rank correlation) or ``"partial"`` (partial correlation).

    Events published:

    - ``new_sensitivity(importance=ndarray)``: Array of shape ``(dim,)``, values in ``[0, 1]``.
    """

    def __init__(
        self, strategy, min_samples=None, update_interval=50, method="spearman"
    ):
        super().__init__(strategy)
        self.logger = self.config.get_logger("SENSI")

        self._min_samples_cfg = min_samples
        self._min_samples: int = 0  # resolved in __start__
        self._update_interval = update_interval
        self._method = method

        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._count_since_update = 0
        self._importance: np.ndarray | None = None

    def __start__(self):
        dim = self.problem.dim
        self._min_samples = (
            self._min_samples_cfg if self._min_samples_cfg is not None else 10 * dim
        )

    @property
    def importance(self) -> np.ndarray | None:
        """Current importance scores, or None if not yet computed."""
        return self._importance

    def on_new_results(self, results):
        self._accumulate(results)

        if self._X is None or len(self._X) < self._min_samples:
            return

        self._count_since_update += len(results)
        if self._count_since_update >= self._update_interval:
            self._count_since_update = 0
            self._compute_and_publish()

    def _accumulate(self, results):
        handler = getattr(self.strategy, "constraint_handler", None)

        new_X = []
        new_y = []
        for r in results:
            if r.x is None or r.fx is None:
                continue
            if handler:
                y_val = handler.get_penalty_value(r)
            else:
                y_val = r.fx
            new_X.append(r.x)
            new_y.append(y_val)

        if not new_X:
            return

        batch_X = np.array(new_X)
        batch_y = np.array(new_y)

        if self._X is None or self._y is None:
            self._X = batch_X
            self._y = batch_y
        else:
            self._X = np.vstack([self._X, batch_X])
            self._y = np.concatenate([self._y, batch_y])

    def _compute_and_publish(self):
        assert self._X is not None and self._y is not None
        dim = self._X.shape[1]

        if self._method == "partial":
            importance = self._partial_correlation(self._X, self._y)
        else:
            importance = self._spearman_correlation(self._X, self._y, dim)

        self._importance = importance
        self.logger.info(f"Sensitivity: {importance}")
        self.eventbus.publish("new_sensitivity", importance=importance)

    @staticmethod
    def _spearman_correlation(X: np.ndarray, y: np.ndarray, dim: int) -> np.ndarray:
        raw = np.zeros(dim)
        for i in range(dim):
            result = spearmanr(X[:, i], y)
            corr = float(result.statistic)  # type: ignore[union-attr]
            raw[i] = abs(corr) if np.isfinite(corr) else 0.0

        max_val = raw.max()
        if max_val > 0:
            return raw / max_val
        return raw

    @staticmethod
    def _partial_correlation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Partial correlation: correlation of x_i with y after removing linear effect of other dims."""
        dim = X.shape[1]
        raw = np.zeros(dim)

        for i in range(dim):
            # Regress y on all dims except i, take residual
            others = np.delete(X, i, axis=1)
            if others.shape[1] == 0:
                # 1D case: just use regular correlation
                result = spearmanr(X[:, i], y)
                corr = float(result.statistic)  # type: ignore[union-attr]
                raw[i] = abs(corr) if np.isfinite(corr) else 0.0
                continue

            # Least squares: y = others @ beta + residual_y
            beta_y, _, _, _ = np.linalg.lstsq(others, y, rcond=None)
            resid_y = y - others @ beta_y

            # Regress x_i on other dims, take residual
            beta_x, _, _, _ = np.linalg.lstsq(others, X[:, i], rcond=None)
            resid_x = X[:, i] - others @ beta_x

            result = spearmanr(resid_x, resid_y)
            corr = float(result.statistic)  # type: ignore[union-attr]
            raw[i] = abs(corr) if np.isfinite(corr) else 0.0

        max_val = raw.max()
        if max_val > 0:
            return raw / max_val
        return raw
