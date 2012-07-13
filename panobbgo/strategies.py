# -*- coding: utf8 -*-
import config
import threading
from utils import stats, logger
from IPython.parallel import Client, Reference #, require, Reference
from IPython.utils.timing import time
#import numpy as np

class Collector(threading.Thread):
  '''
  This Collector receives result lists and passes them along to the statistics and UI.
  It needs to be terminated via a `None` element.
  '''
  def __init__(self):
    from Queue import Queue
    self._tasklist = Queue()
    threading.Thread.__init__(self, name=self.__class__.__name__)
    self.start()

  def run(self):
    while True:
      tlist = self.tasklist.get()
      if tlist is None: return
      for t in tlist:
        logger.info("%s by %s" % (t, t.point.who))

  @property
  def tasklist(self): return self._tasklist

class Strategy0(threading.Thread):
  def __init__(self, problem, results, heurs):
    self.problem = problem
    self.results = results
    self.heurs = heurs
    self._setup_cluster(1, problem)
    self.collector = Collector()
    self.tasklist = self.collector.tasklist
    threading.Thread.__init__(self, name=self.__class__.__name__)

  def _setup_cluster(self, nb_gens, problem):
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
    fff = Reference("problem")
    for run in range(10):
      points = []
      while len(points) < 10:
        for h in self.heurs:
          points.extend(h.get_points(1))
        if len(points) == 0:
          time.sleep(1e-3)

      new_tasks = self.evaluators.map_async(fff, points, chunksize = 5, ordered=False)
      new_tasks.wait()
      self.tasklist.put(new_tasks)

    # signal to end
    self.tasklist.put(None)
    self.collector.join()
    print "Strategy0 finished"


def strategy_bare(problem, results, heurs, nb_gens=1):
  #for h in heurs:
  #  if not isinstance(h, Heuristic):
  #    raise Exception('List must contain Heuristics')
  generators, evaluators = _setup_cluster(nb_gens, problem)
  #from heuristics import Point
  #  points = [ Point(np.random.rand(2), "test"), Point(np.random.rand(2), "test2") ]
  points = []
  for _ in range(10):
    time.sleep(1e-3)
    for h in heurs:
      points.extend(h.get_points(2))
  fff = Reference("problem")
  #fff = problem
  #fff = lambda x : x.x.dot(x.x)
  new_tasks = evaluators.map_async(fff, points, chunksize = 3, ordered=False)
  new_tasks.wait()
  for t in new_tasks:
    print "%s by %s" % (t, t.point.who)

  return "strategy1 finished"


def strategy0(problem, results, heurs, nb_gens=1):
  while stats.cnt < config.max_eval:
    while True:
      evaluators.spin() # check outstanding tasks
      stats.update_finished(evaluators.outstanding)
      if stats.pending < 10:
        break
      time.sleep(1e-3)

    new_points = []

    target = 10 #target 10 new points
    perf_sum = sum(h.perf for h in heurs)
    while True:
      for h in heurs:
        # calc probability based on perfomance with additive smoothing
        delta = .5
        prob = (h.perf + delta)/(perf_sum + delta * len(heurs)) 
        np_h = int(target * prob) + 1
        #logger.info("  %s -> %s" % (h, np_h))
        if rounds < 5:
          np_h = int(float(target) / len(heurs)) + 1
        new_points.extend(h.get_points(np_h))

      # TODO make this more intelligent
      if len(new_points) == 0:
        time.sleep(1e-3)
      else:
        break


    ##for point in new_points:
    ##  stats.add_cnt()
    ##  #logger.info(" new point: %s" % p)
    ##  # TODO later there is a separat thread collecting
    ##  # results and adding them to the @result list
    ##  # and notifying all generating threads
    ##  #time.sleep(1e-3)
    ##  res = Result(point, fx=self.problem(point))
    ##  self.results += res

    #fff = lambda x,y : x(y)
    #ref = [ Reference("problem") ] * len(new_points)
    #fff = lambda x,y : x(y)
    new_tasks = evaluators.map_async(fff, new_points)#, chunksize = 3, ordered=False)
    #new_tasks.wait()
    #print new_tasks.result
    ##import pdb; pdb.set_trace()
    for t in new_tasks:
     print t
    #stats.add_tasks(new_tasks)
    #stats.update_finished(self.evaluators.outstanding)

    # discount all heuristics after each round
    for h in heurs:
      h.discount()

    logger.debug('  '.join(('%s:%.3f' % (h, h.perf) for h in heurs)))
    # TODO remove this hack
    stats._cnt += 1
    rounds += 1
