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
"""The floating-point environment of a measurement: record it, and pin it.

A seeded ``sync_eval`` run is bit-reproducible on one machine, but not
across machines whose OpenBLAS and numpy pick different SIMD kernels at
run time.  GitHub runners come in (at least) two classes: AVX2 hosts and
AVX-512 hosts, where OpenBLAS selects its SkylakeX / Zen4 kernels and numpy
its ``X86_V4`` loops.  The last-bit differences are amplified along a
trajectory and move a cell by up to ~0.08 AOCC (re-baseline run
36228301268, 2026-09-26).

**The pin.**  :func:`pin_fp_env` sets :data:`PIN_ENV` in ``os.environ``:
``OPENBLAS_CORETYPE=Haswell`` (the AVX2 kernels),
``NPY_DISABLE_CPU_FEATURES`` = numpy's AVX-512 dispatch targets, and the
AVX2 caps of torch (``ATEN_CPU_CAPABILITY``), MKL (``MKL_CBWR``,
``MKL_ENABLE_INSTRUCTIONS``) and oneDNN (``ONEDNN_MAX_CPU_ISA``), which are
harmless without torch.  They are read once, when the libraries load, so
the pin must happen **before numpy is imported**.  ``import panobbgo`` does
it (a documented import-time side effect of the package), and the entry
points (``benchmark_harness.py``, ``scripts/ioh_benchmark.py``,
``scripts/ioh_smoke.py``, the ``benchmarks/`` screens) also import
:mod:`panobbgo.fp_pin` first, explicitly; ``scripts/rebaseline.py`` pins in
``main()``.  Child processes inherit ``os.environ``: the spawned
:class:`~panobbgo.local_run.TaskPool` workers, the heuristics' solver
subprocesses and the IOH worker (which gets :func:`child_env` explicitly).
If numpy is already loaded, :func:`pin_fp_env` leaves the environment
alone, so a parent and its workers never run different kernels.

The pin applies only on Linux x86-64 hosts with AVX2 and FMA (forcing the
Haswell kernels on an older CPU would crash).  ``PANOBBGO_FP_PIN=0`` turns
it off.

**The record.**  :func:`collect` describes the environment of this process
(CPU model, ISA flags, BLAS kernels, numpy SIMD targets, versions);
:func:`fp_env_id` hashes the fields that decide the numbers.  Every result
file carries both as ``fp_env`` / ``fp_env_id``; the ``compare`` tools warn
when the ids differ, and ``rebaseline.py aggregate`` refuses to merge
shards from different ids.  See ``doc/dev/benchmarking.md``, "Evaluations,
not wall time".

This module imports nothing heavy at module level: it must be importable
before numpy.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from typing import Any, Dict, List, Mapping, Optional

#: The opt-out: ``PANOBBGO_FP_PIN=0`` (or ``false`` / ``no`` / ``off``) disables the pin.
OPT_OUT_VAR = "PANOBBGO_FP_PIN"

#: numpy's AVX-512 dispatch targets (numpy 2.4 / 2.5 ``__cpu_dispatch__``:
#: ``X86_V3 X86_V4 AVX512_ICL AVX512_SPR``).  An unknown name makes numpy
#: emit an ``ImportWarning`` at import (an error under ``-W error``), which
#: ``tests/test_fp_env.py`` checks.
NUMPY_DISABLED_FEATURES = "X86_V4 AVX512_ICL AVX512_SPR"

#: The environment the pin sets.  The last four cap torch / MKL / oneDNN at
#: AVX2 (the BO baselines); without torch they are ignored.  Bit-identity of
#: the torch path across runner classes is not verified yet (``TODO.md``).
PIN_ENV: Dict[str, str] = {
    "OPENBLAS_CORETYPE": "Haswell",
    "NPY_DISABLE_CPU_FEATURES": NUMPY_DISABLED_FEATURES,
    "ATEN_CPU_CAPABILITY": "avx2",
    "MKL_CBWR": "AVX2",
    "MKL_ENABLE_INSTRUCTIONS": "AVX2",
    "ONEDNN_MAX_CPU_ISA": "AVX2",
}

#: Fields of :func:`collect` that decide the numbers (hashed by :func:`fp_env_id`).
#: Not the CPU model or ISA flags: two CPUs running the same kernels give the
#: same bits, which is the point of the pin.  ``libc`` (its libm) and the
#: Python ``major.minor`` are in: the laptop and the runners differ there,
#: and the id says so honestly even when the numbers happen to agree.
ID_FIELDS = ("machine", "blas", "numpy", "scipy", "numpy_simd", "libc", "python")


def pin_requested(environ: Optional[Mapping[str, str]] = None) -> bool:
    """``False`` iff ``PANOBBGO_FP_PIN`` is set to ``0`` / ``false`` / ``no`` / ``off``."""
    env = os.environ if environ is None else environ
    return env.get(OPT_OUT_VAR, "1").strip().lower() not in ("0", "false", "no", "off")


def cpu_info() -> Dict[str, Any]:
    """``{"model": str, "flags": set}`` from ``/proc/cpuinfo``; an empty flag set when unknown."""
    model = ""
    flags: set = set()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                key = key.strip()
                if key == "model name" and not model:
                    model = value.strip()
                elif key == "flags" and not flags:
                    flags = set(value.split())
                if model and flags:
                    break
    except OSError:
        pass
    return {"model": model or platform.processor() or "unknown", "flags": flags}


def pin_supported() -> bool:
    """Whether the Haswell kernels can run here: Linux x86-64 with AVX2 and FMA."""
    if not sys.platform.startswith("linux") or platform.machine() not in ("x86_64", "AMD64"):
        return False
    flags = cpu_info()["flags"]
    return {"avx2", "fma"} <= flags


def pin_fp_env() -> bool:
    """Set :data:`PIN_ENV` in ``os.environ`` unless opted out, unsupported or too late.

    Returns ``True`` when the pin is (now) in effect for this process: the
    variables are set and numpy was not loaded before.  If numpy is
    already imported, nothing is changed (its kernels are chosen), so the
    workers this process spawns run what it runs.
    """
    if not pin_requested() or not pin_supported():
        return False
    if "numpy" in sys.modules:
        return all(os.environ.get(k) == v for k, v in PIN_ENV.items())
    os.environ.update(PIN_ENV)
    return True


def child_env(base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """An environment for a child process: ``base`` (default ``os.environ``) plus the pin.

    For a child whose numbers do not feed back into this process's kernels
    (the IOH worker), so it is pinned even when this process could not be.
    """
    env = dict(os.environ if base is None else base)
    if pin_requested(env) and pin_supported():
        env.update(PIN_ENV)
    return env


def _blas_libs() -> List[Dict[str, Any]]:
    import numpy  # noqa: F401  # pyright: ignore[reportUnusedImport]  (loads numpy's OpenBLAS)
    from threadpoolctl import threadpool_info

    try:
        import scipy.linalg  # noqa: F401  # pyright: ignore[reportUnusedImport]  (loads scipy's OpenBLAS)
    except ImportError:  # pragma: no cover - scipy is a dependency
        pass
    libs = {
        (str(lib.get("internal_api")), str(lib.get("version")), str(lib.get("architecture")))
        for lib in threadpool_info()
        if lib.get("user_api") == "blas"
    }
    return [{"internal_api": a, "version": v, "architecture": c} for a, v, c in sorted(libs)]


def _numpy_simd() -> Optional[List[str]]:
    """numpy's dispatch targets active in this process (e.g. ``["X86_V3"]``); ``None`` if unknown."""
    try:
        from numpy._core import _multiarray_umath as mu  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

        features = mu.__cpu_features__
        return [f for f in mu.__cpu_dispatch__ if features.get(f)]
    except Exception:  # pragma: no cover - numpy internals moved
        return None


def _torch_info() -> Optional[Dict[str, Any]]:
    """torch's version and CPU capability, only when torch is already loaded (never imports it)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    info: Dict[str, Any] = {"version": str(getattr(torch, "__version__", "unknown"))}
    try:
        info["cpu_capability"] = str(torch.backends.cpu.get_cpu_capability())
    except Exception:
        info["cpu_capability"] = None
    return info


def collect() -> Dict[str, Any]:
    """The floating-point environment of this process (imports numpy, scipy and threadpoolctl).

    ``cpu`` / ``isa`` describe the host, ``blas`` the loaded BLAS libraries
    (``internal_api``, ``version``, ``architecture`` = the kernel set in use),
    ``numpy_simd`` numpy's active dispatch targets, ``pin`` the pin
    variables as this process sees them, ``torch`` (version, CPU
    capability) when torch is loaded, and ``id`` = :func:`fp_env_id`.
    Collected at call time, not cached: a library loaded later (another
    BLAS) shows up.
    """
    import numpy
    import scipy

    info = cpu_info()
    flags = info["flags"]
    env: Dict[str, Any] = {
        "cpu": info["model"],
        "machine": platform.machine(),
        "isa": {f: (f in flags) if flags else None for f in ("avx2", "fma", "avx512f")},
        "blas": _blas_libs(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "numpy_simd": _numpy_simd(),
        "libc": " ".join(platform.libc_ver()).strip() or None,
        "python": "%d.%d" % sys.version_info[:2],
        "pin": {k: os.environ.get(k) for k in (OPT_OUT_VAR, *PIN_ENV)},
    }
    torch = _torch_info()
    if torch is not None:
        env["torch"] = torch
    env["id"] = fp_env_id(env)
    return env


def fp_env_id(env: Optional[Mapping[str, Any]]) -> Optional[str]:
    """A short hash of the :data:`ID_FIELDS` of ``env``; ``None`` for ``None``."""
    if env is None:
        return None
    key = {k: env.get(k) for k in ID_FIELDS}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def current() -> Dict[str, Any]:
    """``{"fp_env": collect(), "fp_env_id": ...}``: the two keys every result file carries.

    Never raises: the record is metadata, and a failure to collect it must
    not lose a finished run's results (both keys are then ``None``, with a
    warning).
    """
    try:
        env = collect()
        return {"fp_env": env, "fp_env_id": env["id"]}
    except Exception as exc:
        import warnings

        warnings.warn(f"could not record the FP environment: {exc!r}", RuntimeWarning, stacklevel=2)
        return {"fp_env": None, "fp_env_id": None}


def current_id() -> str:
    """The ``fp_env_id`` of this process (picklable: a pool worker can report its own)."""
    return collect()["id"]


def mismatch(before: Optional[str], after: Optional[str], label_before: str, label_after: str) -> Optional[str]:
    """A warning text when two ``fp_env_id`` values differ (or one is unknown), else ``None``."""
    if before == after:
        return None
    if before is None or after is None:
        return (
            f"FP environment unknown on one side ({label_before} fp_env_id={before}, {label_after} "
            f"fp_env_id={after}): the file predates the record; a seeded run is bit-reproducible only "
            "within one FP environment."
        )
    return (
        f"FP environment mismatch ({label_before} fp_env_id={before}, {label_after} fp_env_id={after}): "
        "different BLAS / numpy kernels or versions, so the same seed need not give the same numbers; "
        "deltas are not decision-grade."
    )


def known_mismatch(before: Optional[str], after: Optional[str]) -> bool:
    """Both ids are recorded and differ (the case ``--fail-on-regression`` refuses to gate)."""
    return before is not None and after is not None and before != after
