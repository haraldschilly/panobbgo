# -*- coding: utf8 -*-
# Copyright 2012 - 2026 Harald Schilly <harald.schilly@univie.ac.at>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Resource hygiene for long local runs (benchmarks, self-improvement loops).

Optimizer progress is measured in objective evaluations, never in wall
time, so throttling the CPU never changes a result.  Every long-running
entry point therefore

* lowers its own scheduling priority (``nice``) so the machine stays
  usable while a battery runs, and
* refuses to start when the free memory is below a floor, instead of
  pushing the desktop into swap.

Use :func:`add_arguments` on the script's ``ArgumentParser`` and
:func:`apply` on the parsed namespace.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

DEFAULT_NICENESS = 15
DEFAULT_MIN_FREE_GB = 2.0


def be_nice(niceness: int = DEFAULT_NICENESS) -> int:
    """Raise the process niceness to at least ``niceness`` (never lower it).

    Child processes (evaluation workers, the IOH worker) inherit it.
    Returns the effective niceness afterwards.
    """
    try:
        current = os.nice(0)
        if current < niceness:
            return os.nice(niceness - current)
        return current
    except AttributeError, OSError:  # not POSIX, or not permitted
        return 0


def available_memory_gb() -> Optional[float]:
    """``MemAvailable`` from ``/proc/meminfo`` in GiB, or ``None`` if unknown."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0**2)
    except OSError, ValueError, IndexError:
        pass
    return None


def check_free_memory(min_free_gb: float = DEFAULT_MIN_FREE_GB) -> None:
    """Abort with a clear message when less than ``min_free_gb`` GiB is available."""
    avail = available_memory_gb()
    if avail is not None and avail < min_free_gb:
        sys.exit(
            "refusing to start: %.1f GiB memory available, floor is %.1f GiB "
            "(override with --min-free-mem-gb)" % (avail, min_free_gb)
        )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--nice`` / ``--no-nice`` / ``--min-free-mem-gb`` to ``parser``.

    A parser with subcommands gets the flags on each subparser instead, so
    they can be given after the subcommand name.
    """
    subparsers = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    if subparsers:
        for action in subparsers:
            for sub in action.choices.values():
                add_arguments(sub)
        return
    g = parser.add_argument_group("local-run hygiene")
    g.add_argument(
        "--nice",
        type=int,
        default=DEFAULT_NICENESS,
        metavar="N",
        help="run at this niceness so the machine stays responsive (default: %(default)s)",
    )
    g.add_argument("--no-nice", action="store_true", help="keep the normal scheduling priority")
    g.add_argument(
        "--min-free-mem-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        metavar="GB",
        help="refuse to start below this much available memory; 0 disables (default: %(default)s)",
    )


def apply(args: argparse.Namespace) -> None:
    """Apply the flags added by :func:`add_arguments`."""
    if args.min_free_mem_gb > 0:
        check_free_memory(args.min_free_mem_gb)
    if not args.no_nice:
        be_nice(args.nice)
