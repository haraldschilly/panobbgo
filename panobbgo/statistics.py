# -*- coding: utf8 -*-
'''
global statistics and reporting
'''
from config import loggers
logger = loggers['statistic']
from IPython.utils.timing import time

class Statistics(object):
  def __init__(self, evaluators):
    self._cnt      = 0 # show info about evaluated points
    self._cnt_last = 0 # for printing the info line in add_tasks()
    self._time_start = time.time()

    self._evaluators = evaluators

    # task stats
    self._pending  = set([])
    self._new      = []
    self._finished = []

  def add_tasks(self, new_tasks, outstanding):
    if new_tasks != None: 
      map(self._pending.add, new_tasks.msg_ids)
      for mid in new_tasks.msg_ids:
        self._cnt += len(self._evaluators.get_result(mid).result)
    self._new     = self._pending.difference(outstanding)
    self._pending = self._pending.difference(self._new)
    map(self._finished.append, self._new)

    self._cnt += len(self._new)
    if self._cnt / 100 > self._cnt_last / 100:
      self.info()
      self._cnt_last = self._cnt


  def info(self):
    pend = len(self.pending)
    fini = len(self.finished)
    logger.info("Points: %4d | %4d | %4d. Time: %6.3f [s] cpu, %6.3f [s] wall" %
               (self.cnt, pend, fini, self.time_cpu, self.time_wall))

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
