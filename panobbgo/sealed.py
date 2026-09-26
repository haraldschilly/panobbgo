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

* **MA-BBOB instance ids.**  Every development battery uses small ids
  (``0..9``); the sealed ids lie in a reserved range
  ``[SEALED_INSTANCE_MIN, 2**31 - 1)``, and
  :class:`~panobbgo.harness_ioh.IOHBatterySpec` refuses an id from that
  range unless the spec is marked ``sealed=True`` — and refuses a
  non-reserved id in a sealed spec.  A development battery therefore
  cannot contain a sealed instance, whatever ids it asks for.
* **Family instance seeds.**  A family instance's seed is
  ``SHA-256(battery seed, family label, dim, index)``.  The sealed set uses
  its own battery seed, :data:`SEALED_FAMILY_SEED`, and
  :func:`~panobbgo.lib.families.make_family_instances` refuses that seed
  unless called with ``sealed=True``.  ``tests/test_large_and_sealed.py``
  also checks the drawn instance seeds against every development preset
  at every dimension.

The values were drawn once (2026-09-26) from
``SHA-256("panobbgo-sealed-2026-09-26|mabbob|k")`` for ``k = 0..4``
(folded into the reserved range) and
``SHA-256("panobbgo-sealed-2026-09-26|families")`` (mod ``2**32``); no
harness, screen, test or result file used them before.
"""

from __future__ import annotations

import sys
from typing import Optional, TextIO

#: Lower end of the reserved MA-BBOB instance-id range.  ``ioh`` takes the
#: instance id as a signed 32-bit seed, so the range ends at ``2**31 - 1``.
SEALED_INSTANCE_MIN: int = 1_000_000_000

#: The five sealed MA-BBOB instance ids.
SEALED_MABBOB_INSTANCES: tuple = (1553791140, 1983209043, 2058859109, 1654171961, 1553796816)

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


def print_sealed_banner(name: str, stream: Optional[TextIO] = None) -> None:
    """Print the sealed-set warning banner for battery ``name`` (to stderr by default)."""
    out = sys.stderr if stream is None else stream
    print(SEALED_BANNER.format(name=name), file=out, flush=True)
