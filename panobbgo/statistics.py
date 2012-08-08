# -*- coding: utf8 -*-
'''
global statistics and reporting
'''
from config import loggers
logger = loggers['statistic']
from IPython.utils.timing import time

class Statistics(object):
  def __init__(self):
    self._cnt = 0 # evaluated points
    self._last = 0 # for printing the info line in add_tasks()
    self._time_start = time.time()

    # task stats
    self._pending = set([])
    self._finished = set([])

  def add_tasks(self, new_tasks):
    self._cnt += len(new_tasks.msg_ids)
    map(self._pending.add, new_tasks.msg_ids)
    if self._cnt / 100 > self._last / 100:
      self.info()
      self._last = self._cnt

  def update_finished(self, outstanding):
    self._finished = self._pending.difference(outstanding)
    self._pending = self._pending.difference(self._finished)

  def info(self):
    pend = self.pending
    fini = self.finished
    logger.info("Points: %4d | %4d | %4d. Time: %6.3f [s] cpu, %6.3f [s] wall" %
               (self.cnt, pend, fini, self.time_cpu, self.time_wall))

  @property
  def cnt(self): return self._cnt

  @property
  def finished(self): return len(self._finished)

  def get_finished(self): return self._finished

  @property
  def pending(self): return len(self._pending)

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
