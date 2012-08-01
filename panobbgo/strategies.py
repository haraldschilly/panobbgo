# -*- coding: utf8 -*-
'''
This part outlines the coordination between the point-producing
heuristics, the interaction with the cluster and the global
DB of evaluated points.

Basically, one or more threads produce points where to search,
and another one consumes them and dispatches tasks.
'''
import threading
import config
from utils import logger

class Collector(threading.Thread):
  '''
  This Collector receives result lists and passes them along to the statistics and UI.
  It needs to be terminated via a `None` element.
  '''
  def __init__(self, results):
    from Queue import Queue
    self._tasklist = Queue()
    self.results = results
    threading.Thread.__init__(self, name=self.__class__.__name__)
    self.start()

  def run(self):
    while True:
      tlist = self.tasklist.get()
      if tlist is None: return
      for t in tlist:
        self.results += t
        logger.debug("%s by %s" % (t, t.who))

  @property
  def tasklist(self): return self._tasklist

class Strategy0(threading.Thread):
  '''
  Very basic strategy, mainly for testing purposes.
  '''

  def __init__(self, problem, results):
    self.problem = problem
    self.results = results
    self._setup_cluster(1, problem)
    self.collector = Collector(results)
    self.tasklist = self.collector.tasklist
    threading.Thread.__init__(self, name=self.__class__.__name__)

  def _setup_cluster(self, nb_gens, problem):
    from IPython.parallel import Client
    c = Client(profile=config.ipy_profile)
    c.clear() # clears remote engines
    c.purge_results('all') # all results are memorized in the hub

    if len(c.ids) < nb_gens + 1:
      raise Exception('I need at least %d clients.' % (nb_gens + 1))
    dv_evaluators = c[nb_gens:]
    dv_evaluators['problem'] = problem
    self.generators = c.load_balanced_view(c.ids[:nb_gens])
    self.evaluators = c.load_balanced_view(c.ids[nb_gens:])

    # import some packages  (also locally)
    #with c[:].sync_imports():
    #  from IPython.utils.timing import time
    #  import numpy
    #  import math

  def run(self):
    from IPython.parallel import Reference
    from heuristics import Heuristic
    prob_ref = Reference("problem") # see _setup_cluster
    nb_points = 0
    #perf_sum = sum(h.perf for h in heurs)
    Heuristic.normalize_performances()
    perf_sum = 1.0
    while True:
      points = []
      target = 10
      heurs = Heuristic.heuristics()
      while len(points) < target:
        for h in heurs:
          # calc probability based on performance with additive smoothing
          delta = .5
          prob = (h.performance + delta)/(perf_sum + delta * len(heurs))
          np_h = int(target * prob) + 1
          logger.debug("  %s -> %s" % (h, np_h))
          points.extend(h.get_points(np_h))

        if len(points) == 0:
          from IPython.utils.timing import time
          time.sleep(1e-3)

      nb_points += len(points)
      new_tasks = self.evaluators.map_async(prob_ref, points, chunksize = 5, ordered=False)
      new_tasks.wait()
      self.tasklist.put(new_tasks)

      # show heuristic performances after each round
      logger.info('  '.join(('%s:%.3f' % (h, h.performance) for h in heurs)))

      # stopping criteria
      if nb_points > config.max_eval: break

    # signal to end
    self.tasklist.put(None)
    self.collector.join()
    logger.info("Strategy0 finished")

