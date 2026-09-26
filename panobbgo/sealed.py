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
The sealed test set
===================

A held-out battery that is **never** used for tuning, screening, training
or selecting anything — only to report a result or back a claim
(``planning/DESIGN_roadmap_2026-09-26.md`` §3.3, ``doc/dev/benchmarking.md``
"The sealed test set").  Once a number from it has steered a decision, it
is a development battery and a new sealed set has to be drawn.

This module holds only the identities of the sealed problems and the
warning banner; the batteries themselves are
:func:`panobbgo.harness_ioh.make_sealed_battery` (MA-BBOB) and
:func:`panobbgo.harness_families.make_sealed_families_battery` (generated
families).  Both harnesses print :func:`print_sealed_banner` whenever they
run a sealed battery.

Disjointness, by construction
-----------------------------

* **MA-BBOB instance ids.**  Development ids live in a small window,
  ``0 <= id < DEV_INSTANCE_LIMIT`` (``2**20``; the presets use ``0..9``),
  and :class:`~panobbgo.harness_ioh.IOHBatterySpec` refuses any other id
  unless it is the sealed battery itself, which takes exactly
  :data:`SEALED_MABBOB_INSTANCES` (in ``[SEALED_INSTANCE_MIN, 2**31 - 1)``)
  and nothing else.  ``harness_ioh._run_one`` applies the same rule to a
  single run.

  The window matters because ``ioh`` does not use the id as-is everywhere:
  an MA-BBOB instance ``i`` mixes BBOB sub-problems whose transforms are
  seeded with ``fid + 10000 * i`` in 32-bit arithmetic, so ids equal
  modulo ``2**28`` share their sub-problem transforms (and a fid ``f`` at
  ``i`` meets fid ``f + 16`` at ``i + 625**-1 mod 2**28``).  Every sealed id
  keeps all three residues ``i``, ``i +- 625**-1`` (mod ``2**28``) at or
  above ``2**20`` and apart from the other sealed ids'
  (:func:`alias_residues`, checked in ``tests/test_large_and_sealed.py``),
  so no id in the development window can reproduce a sealed component.
* **Family instance seeds.**  A family instance's seed is
  ``SHA-256(battery seed, family label, dim, index)``.  The sealed set uses
  its own battery seed, :data:`SEALED_FAMILY_SEED`, and
  :func:`~panobbgo.lib.families.make_family_instances` refuses that seed
  unless called with ``sealed=True``.  ``tests/test_large_and_sealed.py``
  also checks the drawn instance seeds against every development preset
  at every dimension.

The values were drawn once (2026-09-26) from
``SHA-256("panobbgo-sealed-2026-09-26|mabbob|k")`` for ``k = 0..19``
(the first 8 bytes, little-endian, folded into the reserved range; every
``k`` passed the residue check, none was skipped) and
``SHA-256("panobbgo-sealed-2026-09-26|families")`` (mod ``2**32``); no
harness, screen, test or result file used them before.
"""

from __future__ import annotations

import sys
from typing import Optional, TextIO

#: Lower end of the reserved MA-BBOB instance-id range.  ``ioh`` takes the
#: instance id as a signed 32-bit seed, so the range ends at ``2**31 - 1``.
SEALED_INSTANCE_MIN: int = 1_000_000_000

#: Development MA-BBOB / BBOB instance ids are ``0 <= id < DEV_INSTANCE_LIMIT``.
DEV_INSTANCE_LIMIT: int = 2**20

#: The twenty sealed MA-BBOB instance ids.
SEALED_MABBOB_INSTANCES: tuple = (
    1553791140,
    1983209043,
    2058859109,
    1654171961,
    1553796816,
    1495437080,
    1964370456,
    1201132381,
    1234808549,
    1924739434,
    1736584409,
    2142743780,
    1735647550,
    1890314892,
    1499167996,
    1884231440,
    1518184701,
    1577731663,
    1501495242,
    2015483677,
)

#: Dimensions of the sealed MA-BBOB battery: the development range plus 30/40.
SEALED_MABBOB_DIMS: tuple = (2, 5, 10, 20, 30, 40)

#: Battery seed of the sealed family instances (the development presets use
#: :data:`panobbgo.harness_families.DEFAULT_BATTERY_SEED`).
SEALED_FAMILY_SEED: int = 569725740

SEALED_BANNER: str = """\
################################################################################
#  SEALED TEST SET: {name}
#  Run this only to report a result or back a claim.  Never tune, screen,
#  select or train on it.  A number from it that steers a decision burns the
#  set (doc/dev/benchmarking.md, "The sealed test set").
################################################################################"""


def is_sealed_instance_id(instance: int) -> bool:
    """``True`` for an MA-BBOB instance id in the reserved sealed range."""
    return int(instance) >= SEALED_INSTANCE_MIN


def is_dev_instance_id(instance: int) -> bool:
    """``True`` for an id in the development window ``[0, DEV_INSTANCE_LIMIT)``."""
    return 0 <= int(instance) < DEV_INSTANCE_LIMIT


def alias_residues(instance: int) -> frozenset:
    """Residues mod ``2**28`` whose ids share a BBOB sub-problem transform seed with ``instance``.

    ``ioh`` seeds a BBOB transform with ``fid + 10000 * i`` in 32-bit
    arithmetic.  Two seeds coincide iff ``10000 (i - j) = f' - f (mod 2**32)``;
    ``10000 = 16 * 625`` and ``|f' - f| <= 23``, so ``f' - f`` is ``0`` or
    ``+-16`` and ``j = i`` or ``i +- 625**-1`` modulo ``2**28``.
    """
    m = 2**28
    inv = pow(625, -1, m)
    i = int(instance)
    return frozenset({i % m, (i + inv) % m, (i - inv) % m})


def check_sealed_mabbob_instances(instances) -> None:
    """Raise ``ValueError`` unless ``instances`` is exactly :data:`SEALED_MABBOB_INSTANCES`."""
    if tuple(int(i) for i in instances) != SEALED_MABBOB_INSTANCES:
        raise ValueError(
            "a sealed battery takes exactly panobbgo.sealed.SEALED_MABBOB_INSTANCES, in order "
            "(no sub-selection, no other ids)"
        )


def print_sealed_banner(name: str, stream: Optional[TextIO] = None) -> None:
    """Print the sealed-set warning banner for battery ``name`` (to stderr by default)."""
    out = sys.stderr if stream is None else stream
    print(SEALED_BANNER.format(name=name), file=out, flush=True)
