.. panobbgo documentation master file, created by
   sphinx-quickstart on Thu Jan 19 23:37:51 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Panobbgo: Parallel Noisy Black-Box Global Optimization
===========================================================

..
  .. image:: img/panobbgo-graphic.png
     :align: center

Overview
--------

..
  Date: |today|
  Version: |version|



Panobbgo is an open-source framework for parallel noisy black-box global optimization.
The primary aim is to experiment with new ideas and algorithms.
A couple of functional building blocks build the solver and
exchange information via an :class:`~panobbgo.core.EventBus` among each other.
This allows to rapidly prototype new modules and to combine them with existing parts.
There are three basic types of parts that work together:

* the :mod:`Strategy <panobbgo.strategies>`
* several :mod:`Heuristics <panobbgo.heuristics>`
* and :mod:`Analyzers <panobbgo.analyzers>`

Various tools for extracting statistical data and inspecting the optimization process are included.
Additionally, parallel evaluation of the objective black-box function can be
achieved as SMP or on a cluster via Dask distributed.

In the background, there are additional utility features for the configuration and dependency management available.

This software package is licensed under the
`Apache 2.0 License <http://www.apache.org/licenses/LICENSE-2.0.html>`_.

.. image:: https://github.com/haraldschilly/panobbgo/actions/workflows/tests.yml/badge.svg
   :alt: CI Status
   :target: https://github.com/haraldschilly/panobbgo/actions

User Guide
----------

Comprehensive guide covering concepts, usage, and extension.

.. toctree::
   :maxdepth: 2

   guide

API Reference
-------------

Complete API documentation organized by component.

Core Components
~~~~~~~~~~~~~~~

The fundamental building blocks of Panobbgo.

.. toctree::
   :maxdepth: 1

   core
   config
   utils

Optimization Components
~~~~~~~~~~~~~~~~~~~~~~~

Strategies, heuristics, and analyzers for optimization.

.. toctree::
   :maxdepth: 1

   strategies
   heuristics
   analyzers

Problem Library
~~~~~~~~~~~~~~~

Built-in test problems and problem definition utilities.

.lib

.. toctree::
   :maxdepth: 1

   classic
   lib

User Interface
~~~~~~~~~~~~~~

Graphical and command-line interfaces.

.. toctree::
   :maxdepth: 1

   ui



History
========

This project was revived in 2026 with the help of coding agents like Jules and Claude Code.

Links
=====

* `source repository <http://github.com/haraldschilly/panobbgo>`_
* short `introduction talk <https://docs.google.com/presentation/pub?id=10fyYYtti5B-rdVE9gaJ-H4LWLMCOlBCVUC8B_gk3wXo&start=false&loop=false&delayms=3000>`_
* Indices and Tables

  * :ref:`genindex`
  * :ref:`modindex`
  * :ref:`search`

References
==========

.. [Dask] https://dask.org/
.. [HB] http://en.wikipedia.org/wiki/Himmelblau%27s_function
.. [SH] http://en.wikipedia.org/wiki/Shekel_function
.. [QuadF] Hewlett, Joel D., Bogdan M. Wilamowski, and Gunhan Dundar.
    "Optimization using a modified second-order approach with evolutionary enhancement."
    Industrial Electronics, IEEE Transactions on 55.9 (2008): 3374-3380.
.. [UncTest] Moré, Jorge J., Burton S. Garbow, and Kenneth E. Hillstrom.
    "Testing unconstrained optimization software."
    ACM Transactions on Mathematical Software (TOMS) 7.1 (1981): 17-41.
.. [CompStudy] Pham, Nam, A. Malinowski, and T. Bartczak.
    "Comparative study of derivative free optimization algorithms."
    Industrial Informatics, IEEE Transactions on 7.4 (2011): 592-600.
.. [NQuad] Nesterov, Yurii.
    "Gradient methods for minimizing composite objective function." (2007).
.. [Conn] Conn, A. R., et al.
    "Performance of a multifrontal scheme for partially separable optimization."
    Springer Netherlands, 1994.
.. [branin] LCW Dixon, GP Szegö.
    "Towards global optimisation 2."
    Amsterdam: North-Holland, 1978.
