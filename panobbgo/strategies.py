# -*- coding: utf8 -*-
#import config
#from utils import stats, logger
#from core import Heuristic
from IPython.parallel import Client, Reference #, require, Reference
from IPython.utils.timing import time
#import numpy as np

def _setup_cluster(nb_gens, problem):
  c = Client(profile="default") #config.ipy_profile)
  c.clear() # clears remote engines
  c.purge_results('all') # all results are memorized in the hub

  if len(c.ids) < nb_gens + 1:
    raise Exception('I need at least %d clients.' % (nb_gens + 1))
  dv_evaluators = c[nb_gens:]
  dv_evaluators['problem'] = problem
  generators = c.load_balanced_view(c.ids[:nb_gens])
  evaluators = c.load_balanced_view(c.ids[nb_gens:])

  # import some packages  (also locally)
  #with c[:].sync_imports():
  #  from IPython.utils.timing import time
  #  import numpy
  #  import math
  return generators, evaluators


def strategy1(problem, results, heurs, nb_gens=1):
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
