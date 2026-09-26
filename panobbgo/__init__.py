"""
This is the main part of Panobbgo.

Importing ``panobbgo`` has one side effect: when numpy is not loaded yet,
it pins the floating-point kernels (OpenBLAS core type, numpy's SIMD
targets, torch / MKL / oneDNN ISA) through environment variables, so that a
seeded run gives the same bits on every x86-64 host
(:func:`panobbgo.fp_env.pin_fp_env`, ``doc/dev/benchmarking.md``).  Set
``PANOBBGO_FP_PIN=0`` to leave the environment alone.

.. moduleauthor:: Harald Schilly <harald.schilly@gmail.com>
"""

from __future__ import unicode_literals

from panobbgo.fp_env import pin_fp_env as _pin_fp_env

__version__ = "0.0.1pre"

_pin_fp_env()
