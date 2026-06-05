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


Dask Evaluation Backend
-----------------------

Optional distributed-evaluation backend (``evaluation_method = "dask"``),
imported lazily so the core carries no Dask dependency. Install the
``dask`` extra to use it.

.. automodule:: panobbgo.dask_evaluation
   :members:
   :undoc-members:
