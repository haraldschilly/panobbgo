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
from core import Results, EventBus
from analyzers import Best, Rewarder

#constant
PROBLEM_KEY = "problem"

class Strategy0(threading.Thread):
  '''
  Very basic strategy, mainly for testing purposes.
  '''

  def __init__(self, problem, heurs):
    self._name = name = self.__class__.__name__
    threading.Thread.__init__(self, name=name)
    logger.info("Init of '%s' w/ %d heuristics." % (name, len(heurs)))
    logger.debug("Heuristics %s" % heurs)
    logger.info("%s" % problem)
    self._setup_cluster(0, problem)
    self.problem = problem
    self._eventbus = EventBus()
    self.results = Results(self)
    self._statistics = Statistics(self.evaluators, self.results)
    self._init_heuristics(heurs)
    map(self._eventbus.register, heurs)

    self.best = Best()
    analyzers = [ self.best, Rewarder() ]
    self._init_analyzers(analyzers)
    map(self._eventbus.register, analyzers)

    self.start()
    self._eventbus.publish('start')

  @property
  def heuristics(self):
    return filter(lambda h : not h.stopped, self._heurs.values())

  def _init_analyzers(self, alyz):
    for a in alyz:
      a._strategy = self

  def _init_heuristics(self, heurs):
    import collections
    self._heurs = collections.OrderedDict()
    for h in sorted(heurs, key = lambda h : h.name):
      name = h.name
      assert name not in self._heurs, "Names of heuristics need to be unique. '%s' is already used." % name
      self._heurs[name] = h
      h._strategy = self
      h._init_()

  def _setup_cluster(self, nb_gens, problem):
    from IPython.parallel import Client
    c = self._client = Client(profile=config.ipy_profile)
    c.clear() # clears remote engines
    c.purge_results('all') # all results are memorized in the hub

    if len(c.ids) < nb_gens + 1:
      raise Exception('I need at least %d clients.' % (nb_gens + 1))
    dv_evaluators = c[nb_gens:]
    dv_evaluators[PROBLEM_KEY] = problem
    self.generators = c.load_balanced_view(c.ids[:nb_gens])
    self.evaluators = c.load_balanced_view(c.ids[nb_gens:])
    self.direct_view = c.ids[:]
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

  @property
  def eventbus(self): return self._eventbus

  def run(self):
    from IPython.parallel import Reference
    from IPython.utils.timing import time
    prob_ref = Reference(PROBLEM_KEY) # see _setup_cluster
    self._start = time.time()
    logger.info("Strategy '%s' started" % self._name)
    loops = 0
    while True:
      loops += 1
      points = []
      per_client = 20
      target = per_client * len(self.evaluators)
      new_tasks = None
      if len(self.evaluators.outstanding) < target:
        #Heuristic.normalize_performances()
        heurs = self.heuristics
        perf_sum = sum(h.performance for h in heurs)
        s = config.smooth
        while True:
          for h in heurs:
            # calc probability based on performance with additive smoothing
            prob = (h.performance + s)/(perf_sum + s * len(heurs))
            nb_h = int(target * prob) + 1
            new_points = h.get_points(nb_h)
            points.extend(new_points)

          # stopping criteria
          if len(points) > target: break

          # wait a bit, and loop
          from IPython.utils.timing import time
          time.sleep(1e-3)

        new_tasks = self.evaluators.map_async(prob_ref, points, chunksize = per_client, ordered=False)

      # don't forget, this updates the statistics - new_tasks's default is "None"
      self.stats.add_tasks(new_tasks)

      # collect new results for each finished task, hand over to result DB
      for msg_id in self.stats.new_results:
        for r in self.evaluators.get_result(msg_id).result:
          self.results += r

      # show heuristic performances after each round
      #logger.info('  '.join(('%s:%.3f' % (h, h.performance) for h in heurs)))

      # stopping criteria
      if len(self.results) > config.max_eval: break

      # limit loop speed
      self.evaluators.wait(None, 1e-3)

    # cleanup + shutdown
    self._end = time.time()
    for msg_id in self.evaluators.outstanding:
      try:
        self.evaluators.get_result(msg_id).abort()
      except:
        pass
    logger.info("Strategy '%s' finished after %.3f [s] and %d loops." % (self._name, self._end - self._start, loops))
    #logger.info("distance matrix:\n%s" % self.results._distance)
    self.stats.info()
    self.results.info()

