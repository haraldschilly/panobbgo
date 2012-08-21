# -*- coding: utf8 -*-
'''
global statistics and reporting
'''
import config
logger = config.loggers['statistic']
from IPython.utils.timing import time
import numpy as np

class Statistics(object):
  def __init__(self, evaluators, results):
    self._cnt      = 0 # show info about evaluated points
    self._cnt_last = 0 # for printing the info line in add_tasks()
    self._time_start = time.time()

    self._evaluators = evaluators
    self._results    = results

    self._tasks_walltimes = {}

    # task stats (tasks != points !!!)
    self._pending  = set([])
    self._new      = []
    self._finished = []

  def add_tasks(self, new_tasks):
    if new_tasks != None:
      for mid in new_tasks.msg_ids:
        self._pending.add(mid)
        self._cnt += len(self._evaluators.get_result(mid).result)
    self._new     = self._pending.difference(self._evaluators.outstanding)
    self._pending = self._pending.difference(self._new)
    for tid in self._new:
       self._finished.append(tid)
       self._tasks_walltimes[tid] = self._evaluators.get_result(tid).elapsed

    self._cnt += len(self._new)
    if self._cnt / 100 > self._cnt_last / 100:
      self.info()
      self._cnt_last = self._cnt

  @property
  def avg_time_per_task(self):
    return np.average(self._tasks_walltimes.values())

  def info(self):
    avg_tasks = self.avg_time_per_task
    pend = len(self.pending)
    fini = len(self.finished)
    peval = len(self._results)
    logger.info("%4d (%4d) pnts | Tasks: %3d pend, %3d finished | %6.3f [s] cpu, %6.3f [s] wall, %6.3f [s/task]" %
               (peval, self.cnt, pend, fini, self.time_cpu, self.time_wall, avg_tasks))

  @property
  def cnt(self): return self._cnt

  @property
  def finished(self): return self._finished

  @property
  def new_results(self): return self._new

  @property
  def pending(self): return self._pending

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
  def time_start(self):
    return self._time_start

  @property
  def time_start_str(self):
    return time.ctime(self.time_start())
