"""Temporary probe (experiment branch only): digests of BLAS/LAPACK results, for cross-host FP identity."""

import hashlib
import sys

import numpy as np

r = np.random.default_rng(7)
out = []
for m, k, n in [(64, 321, 64), (64, 512, 64), (64, 1024, 64), (200, 400, 150), (1002, 496, 496), (30, 700, 30)]:
    a = r.random((m, k))
    b = r.random((k, n))
    out.append((f"gemm{m}x{k}x{n}", (a @ b).tobytes()))
A = r.random((1002, 496))
y = r.random(1002)
out.append(("lstsq1002x496", np.linalg.lstsq(A, y, rcond=None)[0].tobytes()))
S = r.random((160, 160))
S = S @ S.T
out.append(("eigh160", np.linalg.eigh(S)[0].tobytes()))
G = A.T @ A
out.append(("gram", G.tobytes()))
allh = hashlib.sha256()
for name, data in out:
    h = hashlib.sha256(data).hexdigest()[:10]
    allh.update(data)
    print(name, h)
print("ALL", sys.argv[1] if len(sys.argv) > 1 else "", allh.hexdigest()[:16])
