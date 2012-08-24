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
    self.cnt      = 0 # show info about evaluated points
    self.show_last = 0 # for printing the info line in add_tasks()
    self.time_start = time.time()

    self.evaluators = evaluators
    self.results    = results

    self.tasks_walltimes = {}

    # task stats (tasks != points !!!)
    self.pending     = set([])
    self.new_results = []
    self.finished    = []

  def add_tasks(self, new_tasks):
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
    if time.time() - self.show_last > .5:
      self.info()
      self.show_last = time.time() #self.cnt

  def info(self):
    avg   = self.avg_time_per_task
    pend  = len(self.pending)
    fini  = len(self.finished)
    peval = len(self.results)
    logger.info("%4d (%4d) pnts | Tasks: %3d pend, %3d finished | %6.3f [s] cpu, %6.3f [s] wall, %6.3f [s/task]" %
               (peval, self.cnt, pend, fini, self.time_cpu, self.time_wall, avg))

  @property
  def avg_time_per_task(self):
    return np.average(self.tasks_walltimes.values())

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
