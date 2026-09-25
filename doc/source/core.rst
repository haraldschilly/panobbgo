Core Module
===========

The core module provides the fundamental building blocks:

- **Analyzer**: Base class for result analysis components
- **EventBus**: Communication system between optimization components
- **StrategyBase**: Base class for optimization strategies
- **Heuristic**: Base class for point generation algorithms
- **Module**: Base class for all Panobbgo components

.. automodule:: panobbgo.core
   :members:
   :undoc-members:
   :show-inheritance:


Local Evaluation Pool
---------------------

Backend of ``evaluation_method = "threaded"`` and ``"processes"``.

.. automodule:: panobbgo.local_pool
   :members: LocalPool, ProcessPool, Outcome

Dask Evaluation Backend
-----------------------

Optional distributed-evaluation backend (``evaluation_method = "dask"``),
imported lazily so the core carries no Dask dependency. Install the
``dask`` extra to use it.

.. automodule:: panobbgo.dask_evaluation
   :members:
   :undoc-members:

Per-call timeout in a child process
-----------------------------------

How ``evaluation.timeout`` is enforced on a dask worker.

.. automodule:: panobbgo.timeout_call
   :members: call_with_timeout, TimedCall
