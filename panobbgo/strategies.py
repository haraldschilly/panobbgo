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
from config import loggers
logger = loggers['strategy']
from statistics import Statistics
from core import Results
from heuristics import Heuristic

#constant
PROBLEM_KEY = "problem"

class Collector(threading.Thread):
  '''
  This Collector receives result lists and passes them along to the statistics and UI.
  It needs to be terminated via a `None` element.
  '''
  def __init__(self, results):
    threading.Thread.__init__(self, name=self.__class__.__name__)
    from Queue import Queue
    self._tasklist = Queue()
    self.results = results
    self.start()

  def run(self):
    while True:
      tlist = self.tasklist.get()
      if tlist is None: return
      for t in tlist:
        self.results += t

  @property
  def tasklist(self): return self._tasklist

class Strategy0(threading.Thread):
  '''
  Very basic strategy, mainly for testing purposes.
  '''

  def __init__(self, problem, heurs):
    self._name = name = self.__class__.__name__
    threading.Thread.__init__(self, name=name)
    self.problem = problem
    self._statistics = Statistics()
    self.results = Results(problem, self.stats)
    Heuristic.register_heuristics(heurs, problem, self.results)
    logger.info("Init of strategy: '%s' w/ %d heuristics." % (name, len(heurs)))
    logger.info("%s" % problem)
    self._setup_cluster(1, problem)
    self.collector = Collector(self.results)
    self.tasklist = self.collector.tasklist
    self.start()

  def _setup_cluster(self, nb_gens, problem):
    from IPython.parallel import Client
    c = Client(profile=config.ipy_profile)
    c.clear() # clears remote engines
    c.purge_results('all') # all results are memorized in the hub

    if len(c.ids) < nb_gens + 1:
      raise Exception('I need at least %d clients.' % (nb_gens + 1))
    dv_evaluators = c[nb_gens:]
    dv_evaluators[PROBLEM_KEY] = problem
    self.generators = c.load_balanced_view(c.ids[:nb_gens])
    self.evaluators = c.load_balanced_view(c.ids[nb_gens:])
    # TODO remove this hack. "problem" wasn't pushed to all clients
    #from IPython.utils.timing import time
    #time.sleep(1e-1)

    # import some packages  (also locally)
    #with c[:].sync_imports():
    #  from IPython.utils.timing import time
    #  import numpy
    #  import math

  @property
  def stats(self): return self._statistics

  def run(self):
    from IPython.parallel import Reference
    from IPython.utils.timing import time
    from heuristics import Heuristic
    prob_ref = Reference(PROBLEM_KEY) # see _setup_cluster
    self._start = time.time()
    logger.info("Strategy '%s' started" % self._name)
    while True:
      points = []
      target = 10
      #Heuristic.normalize_performances()
      heurs = Heuristic.heuristics()
      perf_sum = sum(h.performance for h in heurs)
      while len(points) < target:
        for h in heurs:
          # calc probability based on performance with additive smoothing
          s = config.smooth
          prob = (h.performance + s)/(perf_sum + s * len(heurs))
          np_h = int(target * prob) + 1
          logger.debug("  %s -> %s" % (h, np_h))
          points.extend(h.get_points(np_h))

        if len(points) == 0:
          from IPython.utils.timing import time
          time.sleep(1e-3)

      new_tasks = self.evaluators.map_async(prob_ref, points, chunksize = 5, ordered=False)
      self.stats.add_tasks(new_tasks)
      new_tasks.wait()
      self.tasklist.put(new_tasks)

      # show heuristic performances after each round
      #logger.info('  '.join(('%s:%.3f' % (h, h.performance) for h in heurs)))

      # stopping criteria
      if self.stats.cnt > config.max_eval: break

    # signal to end
    self.tasklist.put(None)
    self.collector.join()
    self._end = time.time()
    logger.info("Strategy '%s' finished after %.3f [s]" % (self._name, self._end - self._start))
    #logger.info("distance matrix:\n%s" % self.results._distance)
    self.stats.info()

