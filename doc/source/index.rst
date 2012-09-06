.. panobbgo documentation master file, created by
   sphinx-quickstart on Thu Jan 19 23:37:51 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|name|: `parallel noisy black-box global optimization`
========================================================

..
  .. image:: panobbgo-graphic.png
     :align: center

Introduction
------------

..
  Date: |today|
  Version: |version|

|name| is an open-source framework for parallel noisy black-box global optimization.
The basic idea is to combine a couple of functional building blocks via an
:class:`~panobbgo.core.EventBus`.
Additionally, parallel evaluation of the objective black-box function can be
archived as SMP or on a cluster via IPython [IP]_.
It is licensed under the `Apache 2.0 License <http://www.apache.org/licenses/LICENSE-2.0.html>`_.

.. Warning ::

  It is currently work in progress and definitely not ready for any kind of usage.

Main
----

.. automodule:: panobbgo

.. toctree::
   :maxdepth: 2

   core
   strategies
   heuristics
   analyzers
   config

Library
-------
.. automodule:: panobbgo_lib

.. toctree::
   :maxdepth: 2

   classic
   lib

Links
-----

* `source repository <http://github.com/haraldschilly/panobbgo>`_
* short `introduction talk <https://docs.google.com/presentation/pub?id=10fyYYtti5B-rdVE9gaJ-H4LWLMCOlBCVUC8B_gk3wXo&start=false&loop=false&delayms=3000>`_
* Indices and Tables

  * :ref:`genindex`
  * :ref:`modindex`
  * :ref:`search`

.. [IP] http://www.ipython.org
