#!/usr/bin/env python
# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
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

"""Digests of a fixed set of BLAS / LAPACK results, pinned like the entry points (``fp-check.yml``).

The family screen of the FP check runs at d = 2 / 5, where every GEMM is
small.  This adds large ones: GEMMs whose K crosses the Haswell kernels'
blocking (K = 321, 512, 1024) and a 1002 x 496 x 496 GEMM, ``lstsq`` and
``eigh``, the sizes at which OpenBLAS 0.3.34's Zen 4 override crashed or
changed bits under the pin (``panobbgo/fp_env.py``, ``OPENBLAS_L2_SIZE``).
Prints one ``name digest`` line per result; the FP check requires the same
output on every runner.  Single-threaded BLAS, as in every measurement.
"""

import hashlib

import panobbgo.fp_pin  # noqa: F401  # pyright: ignore[reportUnusedImport]  (the FP pin, before numpy loads)
from panobbgo.local_run import pin_blas


def main() -> None:
    """Print the digests."""
    import numpy as np

    pin_blas()
    r = np.random.default_rng(7)
    out = []
    for m, k, n in [(64, 321, 64), (64, 512, 64), (64, 1024, 64), (200, 400, 150), (1002, 496, 496), (30, 700, 30)]:
        a = r.random((m, k))
        b = r.random((k, n))
        out.append((f"gemm_{m}x{k}x{n}", a @ b))
    a = r.random((1002, 496))
    out.append(("lstsq_1002x496", np.linalg.lstsq(a, r.random(1002), rcond=None)[0]))
    s = r.random((160, 160))
    out.append(("eigh_160", np.linalg.eigh(s @ s.T)[0]))
    out.append(("gram_496", a.T @ a))
    for name, value in out:
        print(name, hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
