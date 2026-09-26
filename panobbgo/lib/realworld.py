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

r"""
Real-world constrained problems (CEC 2020)
==========================================

A first real-world set for the AOCC harness
(``planning/DESIGN_roadmap_2026-09-26.md`` §3.3): a subset of the CEC 2020
real-world single-objective constrained suite,

    A. Kumar, G. Wu, M. Z. Ali, R. Mallipeddi, P. N. Suganthan, S. Das,
    "A test-suite of non-convex constrained optimization problems from the
    real-world and some baseline results", *Swarm and Evolutionary
    Computation* 56 (2020) 100693, doi:10.1016/j.swevo.2020.100693.
    Preprint and competition material:
    https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation
    (``Problem-Definitions.pdf``, ``Guidelines_Real_World_Constrained.pdf``).

Each problem here is transcribed from the formulas in the paper (§2 there;
"Kumar et al., eq. (n)" below); no code is copied.  Where the paper's
typesetting is ambiguous or differs from the suite's reference MATLAB code
(``cec20_func.m``, which defines the benchmark the paper's best-known values
were measured on), the problem follows the reference definition and says so
in :attr:`RealWorldSpec.notes`.

Which problems, and why these
-----------------------------

Eighteen of the 57, all with :math:`D \le 14` so a 500·D budget stays
cheap: the industrial chemical processes RC01–RC05 (equality-heavy), two of
the small process-synthesis problems (RC09, RC10: one binary variable each)
and eleven mechanical design problems (RC15–RC21, RC23, RC25, RC29, RC32) —
among them the classic engineering designs (speed reducer, spring, pressure
vessel, welded beam, three-bar truss) in their CEC 2020 form.  Left out:
RC06/RC07 (38/48 variables, 32/38 equalities), RC11–RC14 and
RC22/RC26/RC30/RC31 (mostly integer), RC24 (robot gripper: an inner
optimisation per evaluation), RC27/RC33 (finite-element models), RC28
(rolling element bearing: Table 3's 14614.1357 could not be reproduced —
the minimum found on the published definition is 16958.2), and the
power-system, power-electronics and livestock problems RC34–RC57 (74–158
variables, or input data files).

How each problem is verified
----------------------------

Every :class:`RealWorldSpec` carries a feasible reference point ``x_ref``
and the value ``f_ref`` of the model there; ``tests/test_realworld.py``
checks the value, the feasibility (under the suite's rule, below) and that
``f_ref`` agrees with the best-known value ``f_best`` to ``best_rtol``
(1e-9 relative for most).  The best-known values are Kumar et al.,
Table 3, except two, which say so: RC18 (Table 3 lists the
continuous-thickness optimum; the paper defines integer thicknesses) and
RC25 (a point 0.57 % better than Table 3 exists).  Kumar et al. publish no
optimal points, so most ``x_ref`` were computed here, by a search on this
transcription, and reproducing Table 3's eleven-digit value on an
independent search is the check of the transcription.  RC32 is
problem g04 of CEC 2006, whose published optimum is ``x_ref``; the tests
also evaluate points published for the classic problems (Haverly's pooling
optimum, the integer pressure-vessel optimum, the three-bar truss, the
spring, the gas compressor).

Constraints and feasibility
---------------------------

:meth:`RealWorldProblem.eval_constraints` returns the inequalities
:math:`g_i(x) \le 0` followed by every equality as
:math:`|h_j(x)| - \varepsilon \le 0` with :math:`\varepsilon = 10^{-4}`,
the suite's prescription (Kumar et al., eq. (2); guidelines, eq. (2)).  So
panobbgo's own constraint handling sees ordinary inequality constraints,
and a point is *feasible* exactly when every entry is :math:`\le 0` — the
CEC 2020 definition.  :meth:`RealWorldProblem.violation` is the suite's
mean violation :math:`\nu(x)`.

Integer variables are *relaxed and rounded inside the model*, as the suite
does (a variable with the bounds ``[-0.51, 1.49]`` is rounded to 0 or 1):
the optimiser searches a continuous box and sees plateaus.

Failure regions
---------------

Where a formula is undefined — the logarithm of a negative temperature
difference in the heat-exchanger networks RC01 and RC02, a division by zero
on the boundary of RC20 — the evaluation **fails**: :meth:`RealWorldProblem.eval`
raises :class:`~panobbgo.lib.lib.EvaluationCrashed`, which the evaluation
paths book as a failed evaluation and the AOCC trackers count as a spent
call without progress (roadmap §4 D).  The reference MATLAB code guards some
of these spots (``log(abs(.) + 1e-8)``); this module deliberately does not,
because a real simulator in those regions returns no number either.
:attr:`RealWorldSpec.failure` names each problem's failure region.  The guards never matter at a
feasible optimum.

Public surface
--------------

* :data:`REALWORLD_SPECS` — the registry, keyed by name (``"rc17_spring"``).
* :class:`RealWorldProblem` — one problem, built from a spec.
* :func:`make_realworld_instances` — ``(name, problem)`` pairs for the
  harness (:mod:`panobbgo.harness_realworld`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lib import EvaluationCrashed, Problem

#: Tolerance of an equality constraint, :math:`|h(x)| - 10^{-4} \le 0`
#: (Kumar et al. 2020, eq. (2); the CEC 2020 guidelines).
EQ_TOL: float = 1e-4

#: ``(f, g, h)``: objective, inequality values (``<= 0`` feasible) and
#: equality values (``== 0`` feasible).
ModelValue = Tuple[float, np.ndarray, np.ndarray]
Model = Callable[[np.ndarray], ModelValue]

_EMPTY = np.zeros(0)


def _round(v: float) -> float:
    """MATLAB ``round``: halves away from zero (numpy rounds them to even)."""
    return float(np.sign(v) * np.floor(abs(v) + 0.5))


def _arr(*vals: float) -> np.ndarray:
    return np.array(vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# Industrial chemical processes (Kumar et al. §2.1)
# ---------------------------------------------------------------------------


def _rc01(x: np.ndarray) -> ModelValue:
    """Heat exchanger network design, case 1 (Kumar et al., eq. (3))."""
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
    f = 35.0 * x1**0.6 + 35.0 * x2**0.6
    h = _arr(
        200.0 * x1 * x4 - x3,
        200.0 * x2 * x6 - x5,
        x3 - 10000.0 * (x7 - 100.0),
        x5 - 10000.0 * (300.0 - x7),
        x3 - 10000.0 * (600.0 - x8),
        x5 - 10000.0 * (900.0 - x9),
        x4 * np.log(x8 - 100.0) - x4 * np.log(600.0 - x7) - x8 + x7 + 500.0,
        x6 * np.log(x9 - x7) - x6 * np.log(600.0) - x9 + x7 + 600.0,
    )
    return f, _EMPTY, h


def _rc02(x: np.ndarray) -> ModelValue:
    """Heat exchanger network design, case 2 (Kumar et al., eq. (4))."""
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x
    f = (x1 / (120.0 * x4)) ** 0.6 + (x2 / (80.0 * x5)) ** 0.6 + (x3 / (40.0 * x6)) ** 0.6
    h = _arr(
        x1 - 1e4 * (x7 - 100.0),
        x2 - 1e4 * (x8 - x7),
        x3 - 1e4 * (500.0 - x8),
        x1 - 1e4 * (300.0 - x9),
        x2 - 1e4 * (400.0 - x10),
        x3 - 1e4 * (600.0 - x11),
        x4 * np.log(x9 - 100.0) - x4 * np.log(300.0 - x7) - x9 - x7 + 400.0,
        x5 * np.log(x10 - x7) - x5 * np.log(400.0 - x8) - x10 + x7 - x8 + 400.0,
        x6 * np.log(x11 - x8) - x6 * np.log(100.0) - x11 + x8 + 100.0,
    )
    return f, _EMPTY, h


def _rc03(x: np.ndarray) -> ModelValue:
    """Optimal operation of an alkylation unit (Kumar et al., eq. (8)), as a minimisation."""
    x1, x2, x3, x4, x5, x6, x7 = x
    f = -(0.035 * x1 * x6 + 1.715 * x1 + 10.0 * x2 + 4.0565 * x3 - 0.063 * x3 * x5)
    g = _arr(
        0.0059553571 * x6**2 * x1 + 0.88392857 * x3 - 0.1175625 * x6 * x1 - x1,
        1.1088 * x1 + 0.1303533 * x1 * x6 - 0.0066033 * x1 * x6**2 - x3,
        6.66173269 * x6**2 - 56.596669 * x4 + 172.39878 * x5 - 10000.0 - 191.20592 * x6,
        1.08702 * x6 - 0.03762 * x6**2 + 0.32175 * x4 + 56.85075 - x5,
        0.006198 * x7 * x4 * x3 + 2462.3121 * x2 - 25.125634 * x2 * x4 - x3 * x4,
        161.18996 * x3 * x4 + 5000.0 * x2 * x4 - 489510.0 * x2 - x3 * x4 * x7,
        0.33 * x7 + 44.333333 - x5,
        0.022556 * x5 - 1.0 - 0.007595 * x7,
        0.00061 * x3 - 1.0 - 0.0005 * x1,
        0.819672 * x1 - x3 + 0.819672,
        24500.0 * x2 - 250.0 * x2 * x4 - x3 * x4,
        1020.4082 * x4 * x2 + 1.2244898 * x3 * x4 - 100000.0 * x2,
        6.25 * x1 * x6 + 6.25 * x1 - 7.625 * x3 - 100000.0,
        1.22 * x3 - x6 * x1 - x1 + 1.0,
    )
    return f, g, _EMPTY


_RC04_K1 = 0.09755988
_RC04_K2 = 0.99 * _RC04_K1
_RC04_K3 = 0.0391908
_RC04_K4 = 0.9 * _RC04_K3


def _rc04(x: np.ndarray) -> ModelValue:
    """Reactor network design (Kumar et al., eq. (9)), as a minimisation."""
    x1, x2, x3, x4, x5, x6 = x
    f = -x4
    h = _arr(
        _RC04_K1 * x5 * x2 + x1 - 1.0,
        _RC04_K3 * x5 * x3 + x3 + x1 - 1.0,
        _RC04_K2 * x6 * x2 - x1 + x2,
        _RC04_K4 * x6 * x4 + x2 - x1 + x4 - x3,
    )
    g = _arr(np.sqrt(x5) + np.sqrt(x6) - 4.0)
    return f, g, h


def _rc05(x: np.ndarray) -> ModelValue:
    """Haverly's pooling problem (Kumar et al., eq. (5)), as a minimisation."""
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
    f = -(9.0 * x1 + 15.0 * x2 - 6.0 * x3 - 16.0 * x4 - 10.0 * (x5 + x6))
    h = _arr(
        x7 + x8 - x4 - x3,
        x1 - x5 - x7,
        x2 - x6 - x8,
        x9 * x7 + x9 * x8 - 3.0 * x3 - x4,
    )
    g = _arr(
        x9 * x7 + 2.0 * x5 - 2.5 * x1,
        x9 * x8 + 2.0 * x6 - 1.5 * x2,
    )
    return f, g, h


# ---------------------------------------------------------------------------
# Process synthesis (Kumar et al. §2.2)
# ---------------------------------------------------------------------------


def _rc09(x: np.ndarray) -> ModelValue:
    """Process synthesis and design problem (Kumar et al., eq. (11)); ``x3`` binary."""
    x1, x2 = x[0], x[1]
    x3 = _round(x[2])
    f = -x3 + x2 + 2.0 * x1
    h = _arr(-2.0 * np.exp(-x2) + x1)
    g = _arr(x2 - x1 + x3)
    return f, g, h


def _rc10(x: np.ndarray) -> ModelValue:
    """Process flow sheeting problem (Kumar et al., eq. (12)); ``x3`` binary."""
    x1, x2 = x[0], x[1]
    x3 = _round(x[2])
    f = -0.7 * x3 + 0.8 + 5.0 * (0.5 - x1) ** 2
    g = _arr(
        -np.exp(x1 - 0.2) - x2,
        x2 + 1.1 * x3 + 1.0,
        x1 - x3 - 0.2,
    )
    return f, g, _EMPTY


# ---------------------------------------------------------------------------
# Mechanical design (Kumar et al. §2.3)
# ---------------------------------------------------------------------------


def _rc15(x: np.ndarray) -> ModelValue:
    """Weight minimisation of a speed reducer (Kumar et al., eq. (21))."""
    x1, x2, x3, x4, x5, x6, x7 = x
    f = (
        0.7854 * x2**2 * x1 * (14.9334 * x3 - 43.0934 + 3.3333 * x3**2)
        + 0.7854 * (x5 * x7**2 + x4 * x6**2)
        - 1.508 * x1 * (x7**2 + x6**2)
        + 7.477 * (x7**3 + x6**3)
    )
    g = _arr(
        -x1 * x2**2 * x3 + 27.0,
        -x1 * x2**2 * x3**2 + 397.5,
        -x2 * x6**4 * x3 / x4**3 + 1.93,
        -x2 * x7**4 * x3 / x5**3 + 1.93,
        10.0 / x6**3 * np.sqrt(16.91e6 + (745.0 * x4 / (x2 * x3)) ** 2) - 1100.0,
        10.0 / x7**3 * np.sqrt(157.5e6 + (745.0 * x5 / (x2 * x3)) ** 2) - 850.0,
        x2 * x3 - 40.0,
        -x1 / x2 + 5.0,
        x1 / x2 - 12.0,
        1.5 * x6 - x4 + 1.9,
        1.1 * x7 - x5 + 1.9,
    )
    return f, g, _EMPTY


def _rc16(x: np.ndarray) -> ModelValue:
    """Optimal design of an industrial refrigeration system (Kumar et al., eq. (22))."""
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14 = x
    f = (
        63098.88 * x2 * x4 * x12
        + 5441.5 * x2**2 * x12
        + 115055.5 * x2**1.664 * x6
        + 6172.27 * x2**2 * x6
        + 63098.88 * x1 * x3 * x11
        + 5441.5 * x1**2 * x11
        + 115055.5 * x1**1.664 * x5
        + 6172.27 * x1**2 * x5
        + 140.53 * x1 * x11
        + 281.29 * x3 * x11
        + 70.26 * x1**2
        + 281.29 * x1 * x3
        + 281.29 * x3**2
        + 14437.0 * x8**1.8812 * x12**0.3424 * x10 / x14 * x1**2 * x7 / x9
        + 20470.2 * x7**2.893 * x11**0.316 * x1**2
    )
    g = _arr(
        1.524 / x7 - 1.0,
        1.524 / x8 - 1.0,
        0.07789 * x1 - 2.0 / x7 * x9 - 1.0,
        7.05305 / x9 * x1**2 * x10 / x8 / x2 / x14 - 1.0,
        0.0833 / x13 * x14 - 1.0,
        0.04771 * x10 * x8**1.8812 * x12**0.3424 - 1.0,
        0.0488 * x9 * x7**1.893 * x11**0.316 - 1.0,
        0.0099 * x1 / x3 - 1.0,
        0.0193 * x2 / x4 - 1.0,
        0.0298 * x1 / x5 - 1.0,
        47.136 * x2**0.333 / x10 * x12 - 1.333 * x8 * x13**2.1195 + 62.08 * x13**2.1195 * x8**0.2 / (x12 * x10) - 1.0,
        0.056 * x2 / x6 - 1.0,
        2.0 / x9 - 1.0,
        2.0 / x10 - 1.0,
        x12 / x11 - 1.0,
    )
    return f, g, _EMPTY


def _rc17(x: np.ndarray) -> ModelValue:
    """Tension/compression spring design, case 1 (Kumar et al., eq. (23))."""
    x1, x2, x3 = x
    f = x1**2 * x2 * (2.0 + x3)
    g = _arr(
        1.0 - x2**3 * x3 / (71785.0 * x1**4),
        (4.0 * x2**2 - x1 * x2) / (12566.0 * (x2 * x1**3 - x1**4)) + 1.0 / (5108.0 * x1**2) - 1.0,
        1.0 - 140.45 * x1 / (x2**2 * x3),
        (x1 + x2) / 1.5 - 1.0,
    )
    return f, g, _EMPTY


def _rc18(x: np.ndarray) -> ModelValue:
    """Pressure vessel design (Kumar et al., eq. (24)); thicknesses in steps of 1/16 inch."""
    z1 = 0.0625 * _round(x[0])
    z2 = 0.0625 * _round(x[1])
    x3, x4 = x[2], x[3]
    f = 0.6224 * z1 * x3 * x4 + 1.7781 * z2 * x3**2 + 3.1661 * z1**2 * x4 + 19.84 * z1**2 * x3
    g = _arr(
        -z1 + 0.0193 * x3,
        -z2 + 0.00954 * x3,
        -np.pi * x3**2 * x4 - 4.0 / 3.0 * np.pi * x3**3 + 1296000.0,
        x4 - 240.0,
    )
    return f, g, _EMPTY


def _rc19(x: np.ndarray) -> ModelValue:
    """Welded beam design (Kumar et al., eq. (25))."""
    x1, x2, x3, x4 = x
    P, L, delta_max, E, G = 6000.0, 14.0, 0.25, 30e6, 12e6
    tau_max, sigma_max = 13600.0, 30000.0
    f = 1.10471 * x1**2 * x2 + 0.04811 * x3 * x4 * (14.0 + x2)
    Pc = 4.013 * E * np.sqrt(x3**2 * x4**6 / 30.0) / L**2 * (1.0 - x3 / (2.0 * L) * np.sqrt(E / (4.0 * G)))
    sigma = 6.0 * P * L / (x4 * x3**2)
    delta = 6.0 * P * L**3 / (E * x3**2 * x4)
    J = 2.0 * (np.sqrt(2.0) * x1 * x2 * (x2**2 / 4.0 + (x1 + x3) ** 2 / 4.0))
    R = np.sqrt(x2**2 / 4.0 + (x1 + x3) ** 2 / 4.0)
    M = P * (L + x2 / 2.0)
    tau2 = M * R / J
    tau1 = P / (np.sqrt(2.0) * x1 * x2)
    tau = np.sqrt(tau1**2 + 2.0 * tau1 * tau2 * x2 / (2.0 * R) + tau2**2)
    g = _arr(tau - tau_max, sigma - sigma_max, x1 - x4, delta - delta_max, P - Pc)
    return f, g, _EMPTY


def _rc20(x: np.ndarray) -> ModelValue:
    """Three-bar truss design (Kumar et al., eq. (26))."""
    x1, x2 = x
    f = 100.0 * (2.0 * np.sqrt(2.0) * x1 + x2)
    den = np.sqrt(2.0) * x1**2 + 2.0 * x1 * x2
    g = _arr(
        2.0 * (np.sqrt(2.0) * x1 + x2) / den - 2.0,
        2.0 * x2 / den - 2.0,
        2.0 / (np.sqrt(2.0) * x2 + x1) - 2.0,
    )
    return f, g, _EMPTY


def _rc21(x: np.ndarray) -> ModelValue:
    """Multiple disk clutch brake design (Kumar et al., eq. (27))."""
    ri, ro, t, F, Z = x
    Mf, Ms, Iz, n, Tmax, s, delta = 3.0, 40.0, 55.0, 250.0, 15.0, 1.5, 0.5
    Vsrmax, rho, pmax, mu, Lmax, delR = 10.0, 0.0000078, 1.0, 0.6, 30.0, 20.0
    Rsr = 2.0 / 3.0 * (ro**3 - ri**3) / (ro**2 * ri**2)
    Vsr = np.pi * Rsr * n / 30.0
    A = np.pi * (ro**2 - ri**2)
    Prz = F / A
    w = np.pi * n / 30.0
    Mh = 2.0 / 3.0 * mu * F * Z * (ro**3 - ri**3) / (ro**2 - ri**2)
    T = Iz * w / (Mh + Mf)
    f = np.pi * (ro**2 - ri**2) * t * (Z + 1.0) * rho
    g = _arr(
        -ro + ri + delR,
        (Z + 1.0) * (t + delta) - Lmax,
        Prz - pmax,
        Prz * Vsr - pmax * Vsrmax,
        Vsr - Vsrmax,
        T - Tmax,
        s * Ms - Mh,
        -T,
    )
    return f, g, _EMPTY


def _rc23(x: np.ndarray) -> ModelValue:
    """Step-cone pulley (Kumar et al., eq. (29)); diameters and width in mm."""
    d = x[:4] * 1e-3
    w = x[4] * 1e-3
    N = 350.0
    Ni = np.array([750.0, 450.0, 250.0, 150.0])
    rho, a, mu, s, t = 7200.0, 3.0, 0.35, 1.75e6, 8e-3
    ratio = Ni / N
    f = rho * w * np.pi / 4.0 * float(np.sum(d**2 * (1.0 + ratio**2)))
    C = np.pi * d / 2.0 * (1.0 + ratio) + (ratio - 1.0) ** 2 * d**2 / (4.0 * a) + 2.0 * a
    wrap = np.pi - 2.0 * np.arcsin((ratio - 1.0) * d / (2.0 * a))
    R = np.exp(mu * wrap)
    P = s * t * w * (1.0 - np.exp(-mu * wrap)) * np.pi * d * Ni / 60.0
    g = np.concatenate([2.0 - R, 0.75 * 745.6998 - P])
    h = _arr(C[0] - C[1], C[0] - C[2], C[0] - C[3])
    return f, g, h


def _rc25(x: np.ndarray) -> ModelValue:
    """Hydrostatic thrust bearing design (Kumar et al., eq. (31))."""
    R, Ro, mu, Q = x
    gamma, C, n, C1 = 0.0307, 0.5, -3.55, 10.04
    Ws, Pmax, delTmax, hmin = 101000.0, 1000.0, 50.0, 0.001
    gg, N = 386.4, 750.0
    P = (np.log10(np.log10(8.122e6 * mu + 0.8)) - C1) / n
    delT = 2.0 * (10.0**P - 560.0)
    Ef = 9336.0 * Q * gamma * C * delT
    hh = (2.0 * np.pi * N / 60.0) ** 2 * 2.0 * np.pi * mu / Ef * (R**4 / 4.0 - Ro**4 / 4.0) - 1e-5
    Po = (6.0 * mu * Q / (np.pi * hh**3)) * np.log(R / Ro)
    W = np.pi * Po / 2.0 * (R**2 - Ro**2) / (np.log(R / Ro) - 1e-5)
    f = (Q * Po / 0.7 + Ef) / 12.0
    g = _arr(
        Ws - W,
        Po - Pmax,
        delT - delTmax,
        hmin - hh,
        Ro - R,
        gamma / (gg * Po) * (Q / (2.0 * np.pi * R * hh)) - 0.001,
        W / (np.pi * (R**2 - Ro**2) + 1e-5) - 5000.0,
    )
    return f, g, _EMPTY


def _rc29(x: np.ndarray) -> ModelValue:
    """Gas transmission compressor design (Kumar et al., eq. (35))."""
    x1, x2, x3, x4 = x
    f = (
        8.61e5 * x1**0.5 * x2 * x3 ** (-2.0 / 3.0) * x4 ** (-0.5)
        + 3.69e4 * x3
        + 7.72e8 / x1 * x2**0.219
        - 765.43e6 / x1
    )
    g = _arr(x4 / x2**2 + 1.0 / x2**2 - 1.0)
    return f, g, _EMPTY


def _rc32(x: np.ndarray) -> ModelValue:
    """Himmelblau's function (Kumar et al., eq. (39))."""
    x1, x2, x3, x4, x5 = x
    f = 5.3578547 * x3**2 + 0.8356891 * x1 * x5 + 37.293239 * x1 - 40792.141
    G1 = 85.334407 + 0.0056858 * x2 * x5 + 0.0006262 * x1 * x4 - 0.0022053 * x3 * x5
    G2 = 80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3**2
    G3 = 9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4
    g = _arr(G1 - 92.0, -G1, G2 - 110.0, -G2 + 90.0, G3 - 25.0, -G3 + 20.0)
    return f, g, _EMPTY


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealWorldSpec:
    """One real-world problem: model, box, best-known value and a verified reference point.

    Attributes:
        name: Registry key and the harness's problem label, e.g. ``"rc17_spring"``.
        cec_id: The suite's id, ``"RC17"``.
        title: Human-readable title.
        model: ``x -> (f, g, h)``.
        lower, upper: The search box (Kumar et al. / the suite's ``Cal_par.m``).
        f_best: Best-known value, the AOCC target (``f_best_source`` says where it is from).
        f_best_source: Citation of ``f_best``.
        x_ref: A feasible point at which the model gives ``f_ref``.
        f_ref: The model's value at ``x_ref``; within ``best_rtol`` of ``f_best``.
        ref_source: Where ``x_ref`` is from (a publication, or computed here and why).
        best_rtol: Relative distance allowed between ``f_ref`` and ``f_best`` (the equality
            tolerance lets the suite's best-known values sit slightly below an exactly
            feasible optimum).
        integer: Indices of the variables the model rounds.
        failure: The model's failure region (where :meth:`RealWorldProblem.eval` raises
            :class:`~panobbgo.lib.lib.EvaluationCrashed`); empty if there is none in the box.
        notes: Deviations from the paper's printed formula.
    """

    name: str
    cec_id: str
    title: str
    model: Model
    lower: Tuple[float, ...]
    upper: Tuple[float, ...]
    f_best: float
    f_best_source: str
    x_ref: Tuple[float, ...]
    f_ref: float
    ref_source: str
    best_rtol: float = 1e-9
    integer: Tuple[int, ...] = ()
    failure: str = ""
    notes: str = ""

    @property
    def dim(self) -> int:
        """Number of variables."""
        return len(self.lower)


_TABLE3 = "Kumar et al. 2020, Table 3"


class RealWorldProblem(Problem):
    r"""A real-world constrained problem from :data:`REALWORLD_SPECS`.

    :meth:`eval` returns the objective; :meth:`eval_constraints` the
    inequalities followed by :math:`|h_j| - 10^{-4}` for each equality, all
    feasible at ``<= 0``.  Where the model is undefined (:attr:`RealWorldSpec.failure`)
    :meth:`eval` raises :class:`~panobbgo.lib.lib.EvaluationCrashed` and
    :meth:`eval_constraints` returns ``NaN`` entries (an unknown violation,
    i.e. infeasible, see :attr:`Result.cv <panobbgo.lib.lib.Result.cv>`).

    The attributes ``family`` (the name), ``instance`` (always 0) and
    ``f_best`` make it a drop-in problem for :mod:`panobbgo.harness_realworld`.
    """

    def __init__(self, spec: RealWorldSpec) -> None:
        super().__init__(list(zip(spec.lower, spec.upper)))
        self.spec = spec
        self.family: str = spec.name
        self.instance: int = 0
        self.f_best: float = float(spec.f_best)
        self.n_ineq: int = int(len(spec.model(np.asarray(spec.x_ref, dtype=np.float64))[1]))
        self.n_eq: int = int(len(spec.model(np.asarray(spec.x_ref, dtype=np.float64))[2]))
        self.n_constraints: int = self.n_ineq + self.n_eq

    def evaluate(self, x: np.ndarray) -> ModelValue:
        """``(f, g, h)`` at ``x``; raises :class:`EvaluationCrashed` where the model is undefined."""
        x = np.asarray(x, dtype=np.float64)
        with np.errstate(all="ignore"):
            f, g, h = self.spec.model(x)
        f = float(f)
        if not (np.isfinite(f) and np.all(np.isfinite(g)) and np.all(np.isfinite(h))):
            raise EvaluationCrashed(f"{self.spec.cec_id}: model undefined at x={np.array2string(x, precision=6)}")
        return f, g, h

    def eval(self, x: np.ndarray) -> float:
        """Objective value; raises :class:`~panobbgo.lib.lib.EvaluationCrashed` in the failure region."""
        return self.evaluate(x)[0]

    def eval_constraints(self, x: np.ndarray) -> np.ndarray:
        r"""``[g_1 .. g_p, |h_1| - 1e-4 .. |h_q| - 1e-4]``; feasible iff every entry is ``<= 0``.

        ``NaN`` entries in the failure region (never raises).
        """
        try:
            _f, g, h = self.evaluate(x)
        except EvaluationCrashed:
            return np.full(self.n_constraints, np.nan)
        return np.concatenate([g, np.abs(h) - EQ_TOL])

    def is_feasible(self, x: np.ndarray) -> bool:
        """The CEC 2020 rule: every :math:`g_i \\le 0` and every :math:`|h_j| \\le 10^{-4}`."""
        return bool(np.all(self.eval_constraints(x) <= 0.0))

    def violation(self, x: np.ndarray) -> float:
        r"""The suite's mean constraint violation :math:`\nu(x)` (CEC 2020 guidelines, eq. (2)).

        :math:`\nu = (\sum_i \max(g_i, 0) + \sum_j |h_j| [|h_j| > 10^{-4}]) / m`; ``inf`` in the
        failure region.
        """
        try:
            _f, g, h = self.evaluate(x)
        except EvaluationCrashed:
            return float("inf")
        ah = np.abs(h)
        total = float(np.sum(np.maximum(g, 0.0)) + np.sum(ah[ah > EQ_TOL]))
        return total / max(1, self.n_constraints)

    def relative_gap(self, fx: float) -> float:
        """``(fx - f_best) / |f_best|`` — the quantity the real-world AOCC scores."""
        return (float(fx) - self.f_best) / abs(self.f_best)


_FOUND = (
    "computed here: a multistart SLSQP / differential-evolution search on this transcription, polished; "
    "it reproduces the paper's best-known value"
)

#: The real-world problems, keyed by name, in suite order.
REALWORLD_SPECS: Dict[str, RealWorldSpec] = {
    s.name: s
    for s in [
        RealWorldSpec(
            name="rc01_heat_exchanger_1",
            cec_id="RC01",
            title="Heat exchanger network design (case 1)",
            model=_rc01,
            lower=(0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 100.0, 100.0, 100.0),
            upper=(10.0, 200.0, 100.0, 200.0, 2e6, 600.0, 600.0, 600.0, 900.0),
            f_best=1.8931162966e02,
            f_best_source=_TABLE3,
            x_ref=(
                7.8184860265665568e-15,
                16.666666764635092,
                1.9095836017729341e-10,
                121.93825560987264,
                1999999.9999999995,
                599.99999647178856,
                100.00000000000001,
                600.0,
                700.0,
            ),
            f_ref=189.31163047450704,
            ref_source=_FOUND,
            best_rtol=1e-8,
            failure=(
                "h8 takes log(x9 - x7): undefined for x9 <= x7, about 31 % of the box; "
                "h7 at x8 = 100 or x7 = 600 (log 0) on the boundary"
            ),
            notes="The reference code guards the logarithms with log(abs(.) + 1e-8); this transcription does not.",
        ),
        RealWorldSpec(
            name="rc02_heat_exchanger_2",
            cec_id="RC02",
            title="Heat exchanger network design (case 2)",
            model=_rc02,
            lower=(1e4, 1e4, 1e4, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0),
            upper=(0.819e6, 1.131e6, 2.05e6, 0.05074, 0.05074, 0.05074, 200.0, 300.0, 300.0, 300.0, 400.0),
            f_best=7.0490369540e03,
            f_best_source=_TABLE3,
            x_ref=(819000.0, 1131000.0, 2050000.0, 0.05074, 0.05074, 0.05074, 181.9, 295.0, 218.1, 286.9, 395.0),
            f_ref=7049.036954314099,
            ref_source=(
                "derived here: h1-h6 fix x1..x3, x9..x11 given (x7, x8); on that set h7-h9 vanish for any "
                "x4..x6, which then sit at their upper bound, and x7, x8 at the bounds of x1, x2"
            ),
            failure=(
                "h8 takes log(x10 - x7) and h9 log(x11 - x8): undefined for x10 <= x7 or x11 <= x8, about "
                "half the box; x4, x5 or x6 = 0 (division by zero) and x9 = 100 (log 0) on the boundary"
            ),
            notes="The reference code guards the logarithms with log(abs(.) + 1e-8); this transcription does not.",
        ),
        RealWorldSpec(
            name="rc03_alkylation",
            cec_id="RC03",
            title="Optimal operation of an alkylation unit",
            model=_rc03,
            lower=(1000.0, 0.0, 2000.0, 0.0, 0.0, 0.0, 0.0),
            upper=(2000.0, 100.0, 4000.0, 100.0, 100.0, 20.0, 200.0),
            f_best=-4.5291197395e03,
            f_best_source=_TABLE3,
            x_ref=(2000.0, 0.0, 2576.3043112221167, 0.0, 58.160287217572574, 1.2596145353174055, 41.875650894071342),
            f_ref=-4529.119737810161,
            ref_source=_FOUND,
            notes="Maximisation in the paper; minimised here as -f, as the suite does.",
        ),
        RealWorldSpec(
            name="rc04_reactor_network",
            cec_id="RC04",
            title="Reactor network design",
            model=_rc04,
            lower=(0.0, 0.0, 0.0, 0.0, 1e-5, 1e-5),
            upper=(1.0, 1.0, 1.0, 1.0, 16.0, 16.0),
            f_best=-3.8826043623e-01,
            f_best_source=_TABLE3,
            x_ref=(
                0.99989961642271330,
                0.39317206180096043,
                2.0038349783478511e-04,
                0.38826043623131240,
                1.0000000000331242e-05,
                15.974711778718234,
            ),
            f_ref=-0.3882604362313124,
            ref_source=(
                _FOUND + "; the best-known value uses the 1e-4 equality tolerance (with exact equalities "
                "this local optimum is -0.3880822)"
            ),
            notes="Maximisation of x4 in the paper; minimised here as -x4.",
        ),
        RealWorldSpec(
            name="rc05_haverly_pooling",
            cec_id="RC05",
            title="Haverly's pooling problem",
            model=_rc05,
            lower=(0.0,) * 9,
            upper=(100.0, 200.0, 100.0, 100.0, 100.0, 100.0, 200.0, 100.0, 200.0),
            f_best=-4.0000560000e02,
            f_best_source=_TABLE3,
            x_ref=(
                9.9990001395037131e-05,
                200.0,
                1.4998500064053607e-04,
                99.999650034996236,
                1.3947465750454258e-14,
                99.999999999999375,
                2.5061461146101134e-14,
                99.999900009999166,
                1.0000009999010053,
            ),
            f_ref=-400.00559944007546,
            ref_source=(
                _FOUND + "; it perturbs Haverly's exact optimum x = (0, 200, 0, 100, 0, 100, 0, 100, 1), "
                "f = -400, within the 1e-4 equality tolerance"
            ),
            best_rtol=1e-8,  # Table 3 prints -4.0000560000E+02: seven significant digits
            notes="Maximisation in the paper; minimised here as -f.",
        ),
        RealWorldSpec(
            name="rc09_process_synthesis_design",
            cec_id="RC09",
            title="Process synthesis and design problem",
            model=_rc09,
            lower=(0.5, 0.5, -0.51),
            upper=(1.4, 1.4, 1.49),
            f_best=2.5576545740e00,
            f_best_source=_TABLE3,
            x_ref=(0.8525515246516739, 0.8525515246516739, -0.4533606577090741),
            f_ref=2.5576545739550216,
            ref_source=_FOUND,
            integer=(2,),
        ),
        RealWorldSpec(
            name="rc10_process_flow_sheeting",
            cec_id="RC10",
            title="Process flow sheeting problem",
            model=_rc10,
            lower=(0.2, -2.22554, -0.51),
            upper=(1.0, -1.0, 1.49),
            f_best=1.0765430833e00,
            f_best_source=_TABLE3,
            x_ref=(0.9419373447293773, -2.1, 1.484419871578422),
            f_ref=1.0765430833322625,
            ref_source=_FOUND,
            integer=(2,),
        ),
        RealWorldSpec(
            name="rc15_speed_reducer",
            cec_id="RC15",
            title="Weight minimisation of a speed reducer",
            model=_rc15,
            lower=(2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0),
            upper=(3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5),
            f_best=2.9944244658e03,
            f_best_source=_TABLE3,
            x_ref=(
                3.500000000000981,
                0.7000000000000541,
                17.000000000002224,
                7.300000000050687,
                7.715319911491498,
                3.3505409491068536,
                5.286654464980233,
            ),
            f_ref=2994.4244657587406,
            ref_source=_FOUND,
        ),
        RealWorldSpec(
            name="rc16_refrigeration",
            cec_id="RC16",
            title="Optimal design of an industrial refrigeration system",
            model=_rc16,
            lower=(0.001,) * 14,
            upper=(5.0,) * 14,
            f_best=3.2213000814e-02,
            f_best_source=_TABLE3,
            x_ref=(
                1e-3,
                1e-3,
                1e-3,
                1e-3,
                1e-3,
                1e-3,
                1.524,
                1.5240000001524001,
                5.0,
                2.0000000002,
                1e-3,
                1e-3,
                7.2934007809940917e-03,
                8.7555831695918698e-02,
            ),
            f_ref=0.0322130008920229,
            ref_source=_FOUND,
            best_rtol=1e-8,
            notes="The order of the inequalities follows the reference code (the paper lists g6, g7, g11 differently).",
        ),
        RealWorldSpec(
            name="rc17_spring",
            cec_id="RC17",
            title="Tension/compression spring design (case 1)",
            model=_rc17,
            lower=(0.05, 0.25, 2.0),
            upper=(2.0, 1.3, 15.0),
            f_best=1.2665232788e-02,
            f_best_source=_TABLE3,
            x_ref=(0.0516890527256411, 0.3567175387490875, 11.288977538579212),
            f_ref=0.012665232788320725,
            ref_source=_FOUND,
        ),
        RealWorldSpec(
            name="rc18_pressure_vessel",
            cec_id="RC18",
            title="Pressure vessel design",
            model=_rc18,
            lower=(0.51, 0.51, 10.0, 10.0),
            upper=(99.49, 99.49, 200.0, 200.0),
            f_best=6059.714335048431,
            f_best_source=(
                "computed here: the optimum over the integer thicknesses z = 0.0625 x (enumerated) with x3, x4 "
                "on the active constraints, at x = (13, 7, 42.0984456, 176.6365958); Table 3's 5885.3328 is the "
                "continuous-thickness optimum, out of reach with the integer z the paper defines"
            ),
            x_ref=(13.0, 7.0, 42.0984455958549, 176.63659585),
            f_ref=6059.714335225192,
            ref_source="the optimum of f_best_source, x4 rounded up so the volume constraint holds",
            integer=(0, 1),
        ),
        RealWorldSpec(
            name="rc19_welded_beam",
            cec_id="RC19",
            title="Welded beam design",
            model=_rc19,
            lower=(0.125, 0.1, 0.1, 0.1),
            upper=(2.0, 10.0, 10.0, 2.0),
            f_best=1.6702177263e00,
            f_best_source=_TABLE3,
            x_ref=(0.1988323072243747, 3.337365298646237, 9.19202432248124, 0.19883230722443496),
            f_ref=1.6702177262807858,
            ref_source=_FOUND,
            notes=(
                "The suite's variant: J uses x2^2/4 (the classic problem has x2^2/12), hence 1.6702 "
                "rather than the classic 1.7249."
            ),
        ),
        RealWorldSpec(
            name="rc20_three_bar_truss",
            cec_id="RC20",
            title="Three-bar truss design",
            model=_rc20,
            lower=(0.0, 0.0),
            upper=(1.0, 1.0),
            f_best=2.6389584338e02,
            f_best_source=_TABLE3,
            x_ref=(0.7886751352071408, 0.4082482887319383),
            f_ref=263.8958433764684,
            ref_source=_FOUND,
            failure="division by zero at x1 = 0 (the lower bound)",
        ),
        RealWorldSpec(
            name="rc21_disk_clutch_brake",
            cec_id="RC21",
            title="Multiple disk clutch brake design",
            model=_rc21,
            lower=(60.0, 90.0, 1.0, 0.0, 2.0),
            upper=(80.0, 110.0, 3.0, 1000.0, 9.0),
            f_best=2.3524245790e-01,
            f_best_source=_TABLE3,
            x_ref=(70.0, 90.0, 1.0, 626.4595599930869, 2.0),
            f_ref=0.2352424579008037,
            ref_source=_FOUND,
            notes="Eight inequalities as in the reference code (Table 3 counts seven; the eighth is T >= 0).",
        ),
        RealWorldSpec(
            name="rc23_step_cone_pulley",
            cec_id="RC23",
            title="Step-cone pulley",
            model=_rc23,
            lower=(0.0,) * 5,
            upper=(60.0, 60.0, 90.0, 90.0, 90.0),
            f_best=1.6069868725e01,
            f_best_source=_TABLE3,
            x_ref=(38.393739976978935, 52.802944686969965, 70.3984498267989, 84.49571606860209, 89.99999999999736),
            f_ref=16.06986872512538,
            ref_source=_FOUND,
        ),
        RealWorldSpec(
            name="rc25_hydrostatic_thrust_bearing",
            cec_id="RC25",
            title="Hydrostatic thrust bearing design",
            model=_rc25,
            lower=(1.0, 1.0, 1e-6, 1.0),
            upper=(16.0, 16.0, 16e-6, 16.0),
            f_best=1616.1197650512402,
            f_best_source=(
                "computed here (x_ref); 0.57 % below Table 3's 1625.4428, which the paper's three algorithms "
                "did not improve on"
            ),
            x_ref=(5.955511371494067, 5.388715911808754, 5.358697265746318e-06, 2.256637978677851),
            f_ref=1616.1197650512402,
            ref_source="computed here: a differential-evolution search on this transcription, polished",
            notes="As in the reference code, f = (Q P0 / 0.7 + Ef) / 12 (the paper prints it without the 1/12).",
        ),
        RealWorldSpec(
            name="rc29_gas_compressor",
            cec_id="RC29",
            title="Gas transmission compressor design",
            model=_rc29,
            lower=(20.0, 1.0, 20.0, 0.1),
            upper=(50.0, 10.0, 50.0, 60.0),
            f_best=2.9648954173e06,
            f_best_source=_TABLE3,
            x_ref=(49.99999999882101, 1.1782839263919476, 24.592594637640346, 0.38835301119360466),
            f_ref=2964895.417339593,
            ref_source=_FOUND,
        ),
        RealWorldSpec(
            name="rc32_himmelblau",
            cec_id="RC32",
            title="Himmelblau's function",
            model=_rc32,
            lower=(78.0, 33.0, 27.0, 27.0, 27.0),
            upper=(102.0, 45.0, 45.0, 45.0, 45.0),
            f_best=-3.0665538672e04,
            f_best_source=_TABLE3,
            x_ref=(78.0, 33.0, 29.9952560256815985, 45.0, 36.7758129057882073),
            f_ref=-30665.538671783317,
            ref_source=(
                "published: problem g04 of the CEC 2006 constrained suite (J. J. Liang et al., 'Problem "
                "definitions and evaluation criteria for the CEC 2006 special session on constrained "
                "real-parameter optimization', 2006), f* = -30665.5386717834"
            ),
        ),
    ]
}


def make_realworld_instances(names: Optional[Sequence[str]] = None) -> List[Tuple[str, RealWorldProblem]]:
    """``(name, problem)`` pairs for the harness: all of :data:`REALWORLD_SPECS`, or the named ones in that order.

    Names are the registry keys (``"rc17_spring"``) or the suite ids (``"RC17"``, case-insensitive).
    """
    if names is None:
        keys = list(REALWORLD_SPECS)
    else:
        by_id = {s.cec_id.lower(): s.name for s in REALWORLD_SPECS.values()}
        keys = []
        for n in names:
            key = n if n in REALWORLD_SPECS else by_id.get(n.lower())
            if key is None:
                raise KeyError(f"unknown real-world problem {n!r}; known: {', '.join(REALWORLD_SPECS)}")
            keys.append(key)
    return [(k, RealWorldProblem(REALWORLD_SPECS[k])) for k in keys]
