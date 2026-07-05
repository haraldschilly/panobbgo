# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
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
LSHADE-EpSin Heuristic
======================

LSHADE-EpSin (Awad, Ali, Suganthan, CEC 2016) — *Ensemble Sinusoidal*
parameter adaptation layered on top of L-SHADE.  Direct ancestor of the
CEC-2017 co-winner LSHADE-cnEpSin (the same ensemble of sinusoids plus a
covariance-matrix step that we do not port here).  EpSin is a different
branch of the DE family tree from the jSO / NL-SHADE-RSP line: all the
arms shipped so far adapt ``F`` via the SHADE *Cauchy memory*; EpSin
draws ``F`` from a deterministic-amplitude *sinusoid* during the first
half of the search.

The shipped variant inherits the entire
:class:`~panobbgo.heuristics.lshade.LSHADE` asynchronous pipeline —
per-slot pending dict, generation-by-count book-keeping, archive of
replaced parents, linear population reduction, success-history Normal
CR sampling, ``current-to-pbest/1`` mutation skeleton, and warm
restart.  The only change is **how ``F`` is sampled** and the bandit
state that drives it:

1. **First half of generations** (``progress < 0.5`` where
   ``progress = len(results) / max_eval``).  Each trial picks one of
   two sinusoidal candidates with probability ``p_s`` (initially
   ``0.5``):

   * **Sinusoid 1** (fixed frequency, *decreasing* envelope)::

         F = 0.5 · ( sin(2π · freq_fixed · g) · (G_max − g)/G_max + 1 )

     with ``freq_fixed = 0.5``.  Large ``F`` early, shrinking late.

   * **Sinusoid 2** (variable frequency, *increasing* envelope)::

         F = 0.5 · ( sin(2π · freq_i · g + π) · g/G_max + 1 )

     with ``freq_i ~ Cauchy(μ_freq, 0.1)`` clamped to ``(0, 1]``.
     Small ``F`` early, growing late.  ``μ_freq`` adapts each
     generation via Lehmer mean of successful ``freq_i`` values from
     Sinusoid 2 trials.

   ``g`` is the integer generation count (incremented each
   :meth:`_end_of_generation`); ``G_max`` is approximated from the
   budget as ``max_eval / ((NP_init + NP_min) / 2)``.

   The selection probability ``p_s`` is updated each generation from
   the relative success of the two sinusoids::

         p_s = (ns_1 + 1) / (ns_1 + ns_2 + 2)

   — a Laplace-smoothed estimate.  No successes ⇒ ``p_s = 0.5``.

2. **Second half of generations** (``progress ≥ 0.5``).  Reverts to
   the SHADE Cauchy-from-memory ``F`` sampling — byte-identical to
   :class:`~panobbgo.heuristics.lshade.LSHADE`.  The first-half
   sinusoidal phase has populated the memory with successes from a
   diverse ``F`` distribution; the second half exploits that memory.

``CR`` sampling is unchanged from L-SHADE (Normal around ``M_CR[r]``
for a random bin ``r``) in *both* phases — only ``F`` switches
mechanisms.  The asymmetric F-cap (:meth:`LSHADE._apply_F_cap`) is
applied to the sinusoidal-phase ``F`` only if the user opts in via
``F_schedule="jso"`` (or any other named regime); the default is
``F_schedule=None`` because the sinusoidal envelope already provides
its own time-varying cap.

Asynchronous adaptation
-----------------------

EpSin's per-generation bookkeeping is tied to the L-SHADE generation
counter (``_end_of_generation`` fires every ``NP_current`` completed
evolutionary trials).  At each generation boundary the heuristic:

* updates ``p_s`` from ``ns_1 / (ns_1 + ns_2)``;
* updates ``μ_freq`` via the Lehmer mean of successful Sinusoid-2 ``freq``
  values from the just-finished generation;
* resets ``ns_1 / ns_2 / _gen_success_freq`` for the next generation;
* increments ``_gen_count``.

The hooks needed from the parent class
(:meth:`LSHADE._make_trial_meta` to attach the sin choice + freq, and
:meth:`LSHADE._record_success` to count successes) were added as
behaviour-preserving extension points alongside this ship.  L-SHADE,
jSO, and NL-SHADE-RSP keep their byte-identical behaviour — their
hooks default to plain ``_TrialMeta`` construction and no-op success
recording.

Deviations from the paper
-------------------------

* **Generation-budget estimate.**  The paper uses the canonical
  synchronous generation count ``g`` and a known ``G_max`` (the total
  number of generations the loop will run).  Our async port has neither
  exactly — generations complete by count rather than by sync barrier,
  and ``G_max`` is unknowable until ``max_eval`` is reached.  We
  estimate ``G_max ≈ max_eval / ((NP_init + NP_min) / 2)`` (average
  population size under LPSR) and gate the phase split on
  ``progress = len(results) / max_eval`` rather than ``g / G_max``.
  This keeps the schedule in lock-step with how L-SHADE already
  paces LPSR.
* **Selection-probability formula.**  The paper uses a more elaborate
  ranking-selection formula incorporating both success counts
  ``ns_1, ns_2`` and failure counts ``nf_1, nf_2``.  We use the
  simpler Laplace-smoothed ``p_s = (ns_1 + 1) / (ns_1 + ns_2 + 2)``
  — same monotonic direction, smaller state, identical behaviour
  in the ``ns_2 = 0`` and ``ns_1 = 0`` corners that motivated the
  smoothing in the first place.
* **No CMA-style covariance step.**  The CEC-2017 successor
  LSHADE-cnEpSin layers a covariance-matrix mutation step on top of
  EpSin.  We ship only the sinusoidal-F ensemble; CMA-ES is already
  available in Panobbgo as a separate heuristic
  (:class:`~panobbgo.heuristics.cma_es.CMAES`).

References
----------

* N. H. Awad, M. Z. Ali, P. N. Suganthan (2016).  "An Ensemble
  Sinusoidal Parameter Adaptation incorporated with L-SHADE for
  Solving CEC 2014 Benchmark Problems."  *Proceedings of CEC 2016*,
  pp. 2958-2965.  Introduces the LSHADE-EpSin algorithm.
* N. H. Awad, M. Z. Ali, P. N. Suganthan (2017).  "Ensemble Sinusoidal
  Differential Covariance Matrix Adaptation with Euclidean
  Neighborhood for Solving CEC2017 Benchmark Problems."  *Proceedings
  of CEC 2017*.  The cnEpSin descendant — co-winner with jSO of the
  CEC-2017 single-objective competition.
* R. Tanabe & A. Fukunaga (2014).  "Improving the Search Performance
  of SHADE Using Linear Population Size Reduction."  *Proceedings of
  CEC 2014*.  The L-SHADE foundation EpSin refines.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from panobbgo.heuristics.lshade import (
    _DEFAULT_ARCHIVE_FACTOR,
    _DEFAULT_H,
    _DEFAULT_NP_INIT,
    _DEFAULT_NP_MIN,
    _DEFAULT_P_BEST,
    _F_MAX_REDRAWS,
    _PARAM_SCALE,
    LSHADE,
    _TrialMeta,
)


# Default mean frequency for Sinusoid 2 (variable-freq, increasing envelope).
# Awad et al. (2016) report 0.5 as the best initialisation.
_DEFAULT_MU_FREQ: float = 0.5
# Fixed frequency for Sinusoid 1 (decreasing-envelope).
_FIXED_FREQ: float = 0.5
# Cauchy scale for variable-frequency sampling (same as SHADE F/CR scale).
_FREQ_SCALE: float = 0.1
# Initial probability of selecting Sinusoid 1.
_DEFAULT_PS: float = 0.5
# Phase split between sinusoidal-F and Cauchy-from-memory F sampling.
_PHASE_SPLIT: float = 0.5
# Sentinel for "this trial did not use a sinusoid" (i.e. was sampled
# from the SHADE memory).  Sinusoid 1 uses ``1``; Sinusoid 2 uses ``2``.
_SIN_NONE: int = 0
_SIN_1: int = 1
_SIN_2: int = 2
# Fallback G_max when ``max_eval`` is unknown — pure async; gates the
# sinusoidal envelope.  Picked large enough that ``g/G_max ≪ 1``
# during the unknown-budget early phase, which keeps Sinusoid 2's
# envelope small (matching its early-low-F intent) and Sinusoid 1's
# envelope close to 1 (matching its early-high-F intent).
_GMAX_FALLBACK_FACTOR: int = 10


class _EpSinTrialMeta(_TrialMeta):
    """Per-trial bookkeeping that also records the sinusoid + freq used."""

    __slots__ = ("sin_choice", "freq")

    def __init__(
        self,
        slot_idx: int,
        F: float,
        CR: float,
        sin_choice: int = _SIN_NONE,
        freq: float = 0.0,
    ) -> None:
        super().__init__(slot_idx=slot_idx, F=F, CR=CR)
        self.sin_choice = sin_choice
        self.freq = freq


class LSHADE_EpSin(LSHADE):
    """LSHADE-EpSin: ensemble-of-sinusoids F adaptation on top of L-SHADE.

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        NP_init: Initial population size, or ``"auto"`` for budget-adaptive
            sizing.  Default ``30``.  See
            :class:`~panobbgo.heuristics.lshade.LSHADE` for the ``"auto"``
            sizing formula and budget notes.
        NP_min: Minimum population size after LPSR shrinking.
            Default ``4``.
        H: History memory size (used by the second-half Cauchy-from-memory
            ``F`` sampling and the Normal-from-memory ``CR`` sampling at
            all times).  Default ``6`` (inherited from
            :class:`~panobbgo.heuristics.lshade.LSHADE`).  Awad et al.
            report results across a range of ``H`` values; the L-SHADE
            default keeps EpSin in lock-step with its parent's defaults.
        p_best: Greediness factor for ``current-to-pbest/1``.  Default
            ``0.11`` — the L-SHADE default; EpSin inherits the
            ``current-to-pbest/1`` mutation skeleton unchanged.
        archive_factor: Multiplier for the external archive size.
            Default ``1.0``.
        mu_freq_init: Initial mean frequency for the variable-freq
            sinusoid (Sinusoid 2).  Default ``0.5``.  Must lie in
            ``(0, 1]``.
        seed: Optional seed for the per-instance RNG.
        name: Override the heuristic's display name.

    Notes:
        - All numeric arguments are validated; bad values raise
          :class:`ValueError`.
        - Like every Panobbgo heuristic, all state is per-instance.
        - **F_schedule is intentionally opt-in.**  The sinusoidal
          envelope already provides a time-varying ``F`` magnitude;
          composing it with the jSO asymmetric F-cap (Brest 2017) is
          usually counter-productive in the first half.  The default
          (``F_schedule=None``) reproduces the literature behaviour.
          Callers who want the cap can set ``F_schedule="jso"``
          (or another named regime) explicitly — it is applied to
          *both* phases' ``F`` then.
        - When the strategy budget is unknown (no ``max_eval``, zero,
          or non-numeric), the phase split defaults to "sinusoidal" so
          the heuristic still produces a varied ``F`` distribution.
          The ``G_max`` estimate falls back to
          ``_GMAX_FALLBACK_FACTOR · NP_init``.
    """

    def __init__(
        self,
        strategy,
        NP_init: Union[int, str] = _DEFAULT_NP_INIT,
        NP_min: int = _DEFAULT_NP_MIN,
        H: int = _DEFAULT_H,
        p_best: float = _DEFAULT_P_BEST,
        archive_factor: float = _DEFAULT_ARCHIVE_FACTOR,
        mu_freq_init: float = _DEFAULT_MU_FREQ,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if not np.isfinite(mu_freq_init) or not (0.0 < mu_freq_init <= 1.0):
            raise ValueError(f"LSHADE_EpSin: mu_freq_init must be in (0, 1], got {mu_freq_init}")

        super().__init__(
            strategy,
            NP_init=NP_init,
            NP_min=NP_min,
            H=H,
            p_best=p_best,
            archive_factor=archive_factor,
            F_schedule=None,
            seed=seed,
            name=name or "LSHADE_EpSin",
        )
        self.mu_freq_init: float = float(mu_freq_init)

        # Ensemble bandit state.
        self._mu_freq: float = float(mu_freq_init)
        self._p_s: float = _DEFAULT_PS
        # Per-generation success counters for the two sinusoids.
        self._ns1: int = 0
        self._ns2: int = 0
        # Successful Sinusoid-2 frequencies for the in-flight generation.
        self._gen_success_freq: List[float] = []
        # Integer generation counter.  Incremented at ``_end_of_generation``.
        self._gen_count: int = 0
        # Transient sin choice + freq, set by :meth:`_sample_F_CR` and
        # consumed by :meth:`_make_trial_meta`.  Always reset to
        # ``(_SIN_NONE, 0.0)`` between trials.
        self._last_sin: Tuple[int, float] = (_SIN_NONE, 0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _in_sinusoidal_phase(self) -> bool:
        """``True`` when the next trial should draw ``F`` from a sinusoid.

        Maps to ``progress < _PHASE_SPLIT`` when the budget is known;
        falls back to ``True`` (sinusoidal) when the budget is unknown
        so the heuristic still produces a varied ``F`` distribution
        instead of accidentally sampling from the cold memory.
        """
        progress = self._progress()
        if progress is None:
            return True
        return progress < _PHASE_SPLIT

    def _estimate_gmax(self) -> float:
        """Approximate the total generation count when budget exhausts.

        Used by the sinusoidal envelopes (``(G_max − g)/G_max`` and
        ``g/G_max``).  When ``max_eval`` is known we estimate it as
        ``max_eval / avg_NP`` with ``avg_NP = (NP_init + NP_min) / 2``;
        when unknown we use ``_GMAX_FALLBACK_FACTOR · NP_init``.
        Always returns at least ``1.0`` so the envelope ratios stay
        finite.
        """
        max_eval = self._max_eval()
        if max_eval is None:
            return float(max(_GMAX_FALLBACK_FACTOR * self.NP_init, 1))
        avg_NP = max((self.NP_init + self.NP_min) / 2.0, 1.0)
        return max(float(max_eval) / avg_NP, 1.0)

    def _sample_freq_cauchy(self) -> float:
        """Draw a frequency for Sinusoid 2 from ``Cauchy(μ_freq, 0.1)``.

        Clamped to ``(0, 1]`` via redraws on non-positive samples, with
        bounded redraws to prevent worst-case loops.  Falls back to
        ``μ_freq`` when every redraw fails (vanishingly unlikely with
        the heavy-tailed Cauchy + 0.1 scale).
        """
        for _ in range(_F_MAX_REDRAWS):
            f = self._mu_freq + _FREQ_SCALE * float(self._rng.standard_cauchy())
            if f > 0.0:
                return float(min(f, 1.0))
        return float(min(self._mu_freq, 1.0))

    def _sinusoid1_F(self, g: float, gmax: float) -> float:
        """Sinusoid 1: fixed-frequency, decreasing-envelope ``F``.

        ``F = 0.5 · (sin(2π · freq_fixed · g) · (G_max − g)/G_max + 1)``,
        clamped to ``[0, 1]`` to guard against floating-point edge cases.
        """
        envelope = max(0.0, (gmax - g) / gmax)
        F = 0.5 * (np.sin(2.0 * np.pi * _FIXED_FREQ * g) * envelope + 1.0)
        return float(np.clip(F, 0.0, 1.0))

    def _sinusoid2_F(self, g: float, gmax: float, freq: float) -> float:
        """Sinusoid 2: variable-frequency, increasing-envelope ``F``.

        ``F = 0.5 · (sin(2π · freq · g + π) · g/G_max + 1)``, clamped to
        ``[0, 1]``.  The ``+ π`` phase shift means Sinusoid 2 starts at
        the centre of its range while Sinusoid 1 starts at the top of
        its range.
        """
        envelope = max(0.0, min(1.0, g / gmax))
        F = 0.5 * (np.sin(2.0 * np.pi * freq * g + np.pi) * envelope + 1.0)
        return float(np.clip(F, 0.0, 1.0))

    def _sample_F_sinusoidal(self) -> Tuple[float, int, float]:
        """Draw ``F`` from one of the two sinusoids per current ``p_s``.

        Returns ``(F, sin_choice, freq)`` where ``sin_choice`` is
        ``_SIN_1`` or ``_SIN_2`` and ``freq`` is the literal frequency
        used (Sinusoid 1 uses the fixed ``_FIXED_FREQ``; Sinusoid 2
        carries its drawn frequency so end-of-generation can Lehmer-mean
        the successful ones).
        """
        g = float(self._gen_count + 1)  # 1-indexed for the sinusoid argument
        gmax = self._estimate_gmax()
        if self._rng.random() < self._p_s:
            return self._sinusoid1_F(g, gmax), _SIN_1, _FIXED_FREQ
        freq = self._sample_freq_cauchy()
        return self._sinusoid2_F(g, gmax, freq), _SIN_2, freq

    def _sample_CR_from_bin(self) -> float:
        """Draw ``CR`` from a random SHADE memory bin (unchanged from L-SHADE)."""
        r = int(self._rng.integers(0, self.H))
        m_cr = float(self._M_CR[r])
        if m_cr < 0:
            return 0.0
        return float(np.clip(self._rng.normal(m_cr, _PARAM_SCALE), 0.0, 1.0))

    def _sample_F_cauchy_from_bin(self) -> float:
        """Draw ``F`` from a random SHADE memory bin (Cauchy from M_F)."""
        r = int(self._rng.integers(0, self.H))
        m_f = float(self._M_F[r])
        for _ in range(_F_MAX_REDRAWS):
            f = m_f + _PARAM_SCALE * float(self._rng.standard_cauchy())
            if f > 0.0:
                return float(min(f, 1.0))
        return 0.5

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _sample_F_CR(self) -> Tuple[float, float]:
        """Sinusoidal-ensemble ``F`` in the first half, Cauchy-from-memory in the second.

        ``CR`` is always sampled from a random SHADE bin (Normal centred
        on ``M_CR[r]``, clamped to ``[0, 1]``) — identical to L-SHADE in
        both phases.

        Sets :attr:`_last_sin` to the chosen sinusoid (``_SIN_1`` /
        ``_SIN_2``) and its frequency in the first half, or to
        ``(_SIN_NONE, 0.0)`` in the second half.  Consumed by
        :meth:`_make_trial_meta` and reset to ``(_SIN_NONE, 0.0)`` on
        each call so an unused stash from a prior trial does not leak.
        """
        CR = self._sample_CR_from_bin()
        if self._in_sinusoidal_phase():
            F, sin_choice, freq = self._sample_F_sinusoidal()
            self._last_sin = (sin_choice, freq)
        else:
            F = self._sample_F_cauchy_from_bin()
            self._last_sin = (_SIN_NONE, 0.0)
        return self._apply_F_cap(F), CR

    def _make_trial_meta(self, slot_idx: int, F: float, CR: float) -> _TrialMeta:
        """Attach the most-recent sinusoid + freq to the trial record.

        Reads :attr:`_last_sin` set by :meth:`_sample_F_CR` and resets
        it to the no-op sentinel so the next ``_make_trial_meta`` call
        (e.g. one triggered by the random-fill ``on_start`` /
        ``on_restart`` path where ``F = NaN``) does not inherit a stale
        sinusoid choice.
        """
        sin_choice, freq = self._last_sin
        self._last_sin = (_SIN_NONE, 0.0)
        return _EpSinTrialMeta(slot_idx=slot_idx, F=F, CR=CR, sin_choice=sin_choice, freq=freq)

    def _record_success(self, meta: _TrialMeta, delta: float) -> None:
        """Count the successful sinusoid and stash its frequency for Lehmer averaging.

        Random-fill trials (``F = CR = NaN``, ``sin_choice = _SIN_NONE``)
        are skipped — the parent's NaN guard in :meth:`on_new_results`
        already short-circuits, but we re-check here so subclassing
        is hermetic.  Cauchy-phase successes (``sin_choice = _SIN_NONE``,
        non-NaN F/CR) are tracked by the inherited SHADE memory update
        and do not feed the ensemble bandit.
        """
        if not isinstance(meta, _EpSinTrialMeta):
            return
        if meta.sin_choice == _SIN_1:
            self._ns1 += 1
        elif meta.sin_choice == _SIN_2:
            self._ns2 += 1
            self._gen_success_freq.append(float(meta.freq))

    def _update_ensemble_state(self) -> None:
        """End-of-generation update for ``p_s``, ``μ_freq``, and counters.

        ``p_s`` is the Laplace-smoothed Sinusoid-1 success rate
        ``(ns_1 + 1) / (ns_1 + ns_2 + 2)``.  With no recorded
        successes (``ns_1 = ns_2 = 0``) this collapses to the
        cold-start ``0.5`` so the next generation samples the two
        sinusoids equally.

        ``μ_freq`` is the *weighted Lehmer mean* of successful
        Sinusoid-2 frequencies in the just-finished generation, mirroring
        the SHADE memory update for ``F``.  We do not have a per-trial
        ``delta`` weight here (the L-SHADE parent already consumed it for
        the F/CR memory bins; carrying a parallel weighted list of freqs
        is fine but the Sinusoid-2 success counts are typically small),
        so we use the unweighted Lehmer mean
        ``Σ freq² / Σ freq`` — same monotonic direction, no per-call
        bookkeeping.  When no Sinusoid-2 successes occurred this
        generation, ``μ_freq`` is left untouched.

        All counters are cleared after the update so the next generation
        starts from a fresh per-generation slate.
        """
        if self._ns1 + self._ns2 > 0:
            self._p_s = float((self._ns1 + 1) / (self._ns1 + self._ns2 + 2))
        if self._gen_success_freq:
            arr = np.asarray(self._gen_success_freq, dtype=float)
            num = float(np.sum(arr * arr))
            den = float(np.sum(arr))
            if den > 0.0:
                self._mu_freq = float(np.clip(num / den, 1e-6, 1.0))
        self._ns1 = 0
        self._ns2 = 0
        self._gen_success_freq.clear()

    def _end_of_generation(self) -> None:
        """Apply the EpSin ensemble update on top of the L-SHADE memory update."""
        self._update_ensemble_state()
        super()._end_of_generation()
        self._gen_count += 1

    # ------------------------------------------------------------------
    # Heuristic interface
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Allocate state and reset the ensemble bandit + frequency memory."""
        super().on_start()
        self._mu_freq = float(self.mu_freq_init)
        self._p_s = _DEFAULT_PS
        self._ns1 = 0
        self._ns2 = 0
        self._gen_success_freq.clear()
        self._gen_count = 0
        self._last_sin = (_SIN_NONE, 0.0)

    def on_restart(self, center, reason: str = "") -> None:
        """Reseed the population and reset the ensemble bandit + frequency memory."""
        super().on_restart(center, reason)
        self._mu_freq = float(self.mu_freq_init)
        self._p_s = _DEFAULT_PS
        self._ns1 = 0
        self._ns2 = 0
        self._gen_success_freq.clear()
        self._gen_count = 0
        self._last_sin = (_SIN_NONE, 0.0)
