"""Panobbgo IOH worker.

Stateful child process that hosts a single ``ioh.problem.RealSingleObjective``
instance and proxies ``eval(x) -> fx`` calls over a JSON-Lines stdin/stdout
protocol.  See ``__main__.py`` for the protocol.
"""

__version__ = "0.1.0"
