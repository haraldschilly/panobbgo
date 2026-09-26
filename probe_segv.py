"""Temporary probe (experiment branch only): which BLAS/LAPACK call crashes, per variant."""

import sys

import numpy as np

which = sys.argv[1]
r = np.random.default_rng(2)
A = r.random((1002, 496))
b = r.random(1002)
if which == "lstsq":
    np.linalg.lstsq(A, b, rcond=None)
elif which == "gemm_AtA":
    A.T @ A
elif which == "gemm_AB":
    A @ r.random((496, 496))
elif which == "gemm_small":
    r.random((64, 64)) @ r.random((64, 64))
elif which == "svd":
    np.linalg.svd(A, full_matrices=False)
elif which == "qr":
    np.linalg.qr(A)
elif which == "scipy_gelsy":
    import scipy.linalg

    scipy.linalg.lstsq(A, b, lapack_driver="gelsy")
elif which == "scipy_gelsd":
    import scipy.linalg

    scipy.linalg.lstsq(A, b, lapack_driver="gelsd")
elif which == "lstsq_300":
    B = r.random((600, 300))
    np.linalg.lstsq(B, r.random(600), rcond=None)
elif which == "lstsq_100":
    B = r.random((200, 100))
    np.linalg.lstsq(B, r.random(200), rcond=None)
elif which == "config":
    from threadpoolctl import threadpool_info

    print(threadpool_info())
    import ctypes

    for lib in threadpool_info():
        if lib.get("internal_api") == "openblas":
            h = ctypes.CDLL(lib["filepath"])
            for name in ("scipy_openblas_get_config64_", "openblas_get_config64_", "openblas_get_config"):
                try:
                    f = getattr(h, name)
                    f.restype = ctypes.c_char_p
                    print(name, f())
                    break
                except AttributeError:
                    pass
print("ok", which)
