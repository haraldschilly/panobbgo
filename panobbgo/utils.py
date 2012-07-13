# -*- coding: utf-8 -*-
import config
import logging
from IPython.utils.timing import time

def create_logger(name, level):
  '''
  creates logger with @name and @level logging level
  '''
  logger = logging.getLogger('psnobfit')
  logger.setLevel(logging.DEBUG)
  log_stream_handler = logging.StreamHandler()
  log_stream_handler.setLevel(level)
  log_formatter = logging.Formatter('%(asctime)s %(name)s/%(levelname)-7s %(message)s')
  log_stream_handler.setFormatter(log_formatter)
  logger.addHandler(log_stream_handler)
  return logger

logger = create_logger('psnobfit', config.loglevel)

#
# global statistics and reporting
#

class Statistics(object):
  def __init__(self):
    self._cnt = 0 # evaluated points
    self._time_start = time.time()

    # task stats
    self._pending = set([])
    self._finished = set([])

  def add_tasks(self, new_tasks):
    self._cnt += len(new_tasks.msg_ids)
    map(self._pending.add, new_tasks.msg_ids)
    if self._cnt % 100 == 0:
      logger.info("%5d points evaluated. time: %5.2f [s] cpu, %5.2f [s] wall" % (self.cnt, self.time_cpu, self.time_wall))

  def update_finished(self, outstanding):
    self._finished = self._pending.difference(outstanding)
    self._pending = self._pending.difference(self._finished)

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
    return time.clock()

  @property
  def time_start(self):
    return self._time_start

  @property
  def time_start_str(self):
    return time.ctime(self.time_start())

stats = Statistics()
