.. panobbgo documentation master file, created by
   sphinx-quickstart on Thu Jan 19 23:37:51 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |NAME|   replace:: Panobbgo
.. |AUTHOR| replace:: Harald Schilly

Panobbgo: `parallel noisy black-box global optimization`
========================================================

..
  .. image:: panobbgo-graphic.png
     :align: center

..
  Date: |today|
  Version: |version|

Introduction
------------

Panobbgo is an open-source framework for parallel noisy black-box global optimization.
The basic idea is to combine a couple of functional building blocks via an
:class:`~panobbgo.core.EventBus`.
Additionally, parallel evaluation of the objective black-box function can be
archived as SMP or on a cluster via IPython [IP]_.


* `source repository <http://github.com/haraldschilly/panobbgo>`_
* short `introduction talk <https://docs.google.com/presentation/pub?id=10fyYYtti5B-rdVE9gaJ-H4LWLMCOlBCVUC8B_gk3wXo&start=false&loop=false&delayms=3000>`_


.. toctree::
   :maxdepth: 2

   panobbgo
   library

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [HB] http://en.wikipedia.org/wiki/Himmelblau%27s_function
.. [IP] http://www.ipython.org
