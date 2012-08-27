# -*- coding: utf8 -*-
'''
This part outlines the coordination between the point-producing
heuristics, the interaction with the cluster and the global
DB of evaluated points.

Basically, one or more threads produce points where to search,
and another one consumes them and dispatches tasks.
'''
#import threading
import config
logger = config.get_logger('STRA')
slogger = config.get_logger('STAT')
from core import Results, EventBus
from IPython.utils.timing import time
import numpy as np

### constants
# reference id for sending the evaluation code to workers
PROBLEM_KEY = "problem"

class Strategy0(object):
  '''
  Very basic strategy, mainly for testing purposes.
  '''

  def __init__(self, problem, heurs):
    self._name = name = self.__class__.__name__
    #threading.Thread.__init__(self, name=name)
    logger.info("Init of '%s' w/ %d heuristics." % (name, len(heurs)))
    logger.debug("Heuristics %s" % heurs)
    logger.info("%s" % problem)
    # statistics
    self.cnt         = 0 # show info about evaluated points
    self.show_last  = 0 # for printing the info line in _add_tasks()
    self.time_start = time.time()
    self.tasks_walltimes = {}
    # task accounting (tasks != points !!!)
    self.pending     = set([])
    self.new_results = []
    self.finished    = []
    # init & start everything
    self._setup_cluster(0, problem)
    self.problem = problem
    self.eventbus = EventBus()
    self.results = Results(self)

    # heuristics
    import collections
    self._heuristics = collections.OrderedDict()
    for h in sorted(heurs, key = lambda h : h.name):
      self.add_heuristic(h)

    # analyzers
    from analyzers import Best, Rewarder, Grid, Splitter
    self._analyzers = {
        'best' :     Best(),
        'rewarder' : Rewarder(),
        'grid':      Grid(),
        'splitter':  Splitter()
    }
    for a in self._analyzers.values(): self._init_module(a)

    logger.debug("Eventbus keys: %s" % self.eventbus.keys)

    try:
      self.run() # CHECK if strategy is a thread, then change this to start()
    except KeyboardInterrupt:
      logger.critical("KeyboardInterrupt recieved, e.g. via Ctrl-C")
      self._cleanup()

  @property
  def heuristics(self):
    return filter(lambda h : h.active, self._heuristics.values())

  def heuristic(self, who):
    return self._heuristics[who]

  def analyzer(self, who):
    return self._analyzers[who]

  def add_heuristic(self, h):
    name = h.name
    assert name not in self._heuristics, \
      "Names of heuristics need to be unique. '%s' is already used." % name
    self._heuristics[name] = h
    self._init_module(h)

  def add_analyzer(self, key, a):
    assert key not in self._analyzers, \
        "Names of analyzers need to be unique. '%s' is already used." % key
    self._analyzers[key] = a
    self._init_module(a)

  def _init_module(self, m):
    '''
    do this *after* the specialized init's
    '''
    m.strategy = self
    m.eventbus = self.eventbus
    m.problem  = self.problem
    m.results  = self.results
    m._init_()
    # only after _init_ it is ready to recieve events
    self.eventbus.register(m)

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
    #time.sleep(1e-1)

    # import some packages  (also locally)
    #with c[:].sync_imports():
    #  import numpy
    #  import math

  @property
  def best(self): return self._analyzers['best'].best

  def run(self):
    self.eventbus.publish('start', terminate=True)
    from IPython.parallel import Reference
    prob_ref = Reference(PROBLEM_KEY) # see _setup_cluster
    self._start = time.time()
    logger.info("Strategy '%s' started" % self._name)
    self.loops = 0
    while True:
      self.loops += 1
      points = []
      per_client = max(1, int(min(config.max_eval / 50, 1.0 / self.avg_time_per_task)))
      target = per_client * len(self.evaluators)
      logger.debug("per_client = %s | target = %s" % (per_client, target))
      new_tasks = None
      if len(self.evaluators.outstanding) < target:
        s = config.smooth
        while True:
          heurs = self.heuristics
          perf_sum = sum(h.performance for h in heurs)
          for h in heurs:
            # calc probability based on performance with additive smoothing
            prob = (h.performance + s)/(perf_sum + s * len(heurs))
            nb_h = max(1, round(target * prob))
            points.extend(h.get_points(nb_h))
            #print "  %16s -> %s" % (h, nb_h)

          # stopping criteria
          if len(points) >= target: break

          # wait a bit, and loop
          time.sleep(1e-3)

        new_tasks = self.evaluators.map_async(prob_ref, points, chunksize = per_client, ordered=False)

      # don't forget, this updates the statistics - new_tasks's default is "None"
      self._add_tasks(new_tasks)

      # collect new results for each finished task, hand over to result DB
      for msg_id in self.new_results:
        for r in self.evaluators.get_result(msg_id).result:
          self.results += r

      # show heuristic performances after each round
      #logger.info('  '.join(('%s:%.3f' % (h, h.performance) for h in heurs)))

      # stopping criteria
      if len(self.results) > config.max_eval: break

      # limit loop speed
      self.evaluators.wait(None, 1e-3)

    self._cleanup()

  def _cleanup(self):
    '''
    cleanup + shutdown
    '''
    self._end = time.time()
    for msg_id in self.evaluators.outstanding:
      try:
        self.evaluators.get_result(msg_id).abort()
      except:
        pass
    logger.info("Strategy '%s' finished after %.3f [s] and %d loops." \
             % (self._name, self._end - self._start, self.loops))
    #logger.info("distance matrix:\n%s" % self.results._distance)
    self.info()
    self.results.info()

  def _add_tasks(self, new_tasks):
    if new_tasks != None:
      for mid in new_tasks.msg_ids:
        self.pending.add(mid)
        self.cnt += len(self.evaluators.get_result(mid).result)
    self.new_results = self.pending.difference(self.evaluators.outstanding)
    self.pending = self.pending.difference(self.new_results)
    for tid in self.new_results:
       self.finished.append(tid)
       self.tasks_walltimes[tid] = self.evaluators.get_result(tid).elapsed

    self.cnt += len(self.new_results)
    #if self.cnt / 100 > self.show_last / 100:
    if time.time() - self.show_last > config.show_interval:
      self.info()
      self.show_last = time.time() #self.cnt

  def info(self):
    avg   = self.avg_time_per_task
    pend  = len(self.pending)
    fini  = len(self.finished)
    peval = len(self.results)
    slogger.info("%4d (%4d) pnts | Tasks: %3d pend, %3d finished | %6.3f [s] cpu, %6.3f [s] wall, %6.3f [s/task]" %
               (peval, self.cnt, pend, fini, self.time_cpu, self.time_wall, avg))

  @property
  def avg_time_per_task(self):
    if len(self.tasks_walltimes) > 1:
      return np.average(self.tasks_walltimes.values())
    slogger.warning("avg time per task for 0 tasks!")
    return 0.01

  @property
  def time_wall(self):
    '''
    wall time in seconds
    '''
    return time.time() - self.time_start

  @property
  def time_cpu(self):
    '''
    effective cpu time in seconds
    '''
    return time.clock()

  @property
  def time_start_str(self):
    return time.ctime(self.time_start)
