.. panobbgo documentation master file, created by
   sphinx-quickstart on Thu Jan 19 23:37:51 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|name|: `parallel noisy black-box global optimization`
========================================================

..
  .. image:: img/panobbgo-graphic.png
     :align: center

Introduction
------------

..
  Date: |today|
  Version: |version|

.. Warning ::

  It is currently work in progress and definitely not ready for any kind of usage.

|name| is an open-source framework for parallel noisy black-box global optimization.
The primary aim is to experiment with new ideas and algorithms.
A couple of functional building blocks build the solver and
exchange information via an :class:`~panobbgo.core.EventBus` among each other.
This allows to rapidly prototype new modules and to combine them with existing parts.
There are three basic types of parts that work together:

* the :mod:`Strategy <panobbgo.strategies>`
* several :mod:`Heuristics <panobbgo.heuristics>`
* and :mod:`Analyzers <panobbgo.analyzers>`

Various tools for extracting statistical data and inspecting the optimization process are included (`planned`).
Additionally, parallel evaluation of the objective black-box function can be
archived as SMP or on a cluster via IPython [IP]_.

In the background, there are additional utility features for the configuration and dependency management (`planned`) available.

This software package is licensed under the 
`Apache 2.0 License <http://www.apache.org/licenses/LICENSE-2.0.html>`_.

.. image:: https://secure.travis-ci.org/haraldschilly/panobbgo.png?branch=master
   :alt: Build Status
   :target: https://travis-ci.org/haraldschilly/panobbgo

Main
----

This component is the main part of the program.

.. automodule:: panobbgo

.. toctree::
   :maxdepth: 2

   core
   strategies
   heuristics
   analyzers
   config
   ui
   utils

Library
-------

The library component contains the code for constructing the problem instances
and running them on the computational nodes.

.. automodule:: panobbgo_lib

.. toctree::
   :maxdepth: 2

   classic
   lib

.. include:: examples.rst

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

.. [IP] http://www.ipython.org
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
