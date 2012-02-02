#!/usr/bin/env python
# just testing basic parallelization that will be used in the actual project
import time
import numpy as np

# setup proper logging
import logging
logger = logging.getLogger('psnobfit')
logger.setLevel(logging.DEBUG)
log_stream_handler = logging.StreamHandler()
log_stream_handler.setLevel(logging.DEBUG)
log_formatter = logging.Formatter('%(asctime)s %(name)s/%(levelname)-9s %(message)s')
log_stream_handler.setFormatter(log_formatter)
logger.addHandler(log_stream_handler)
del logging, log_stream_handler, log_formatter

# START
logger.info("init")
from IPython.parallel import Client
c = Client(profile='unicluster')
c.clear() # clears remote engines
c.purge_results('all') # all results are memorized in the hub
lb = c.load_balanced_view()
N = len(c.ids)
    
# import everything
with c[:].sync_imports():
  #from IPython.utils.timing import time
  import time
  from random import random
  from numpy import pi, sum
  import math


# the actual function
def func(x, data):
  'x is either a number or a list/vector of numbers'
  time.sleep(math.log(2 + x * random()))
  return sum(x), len(data)

added = 0
queue_size = 0
added = 0

def status():
  logger.debug("queue_size = %3d | adding %2d tasks | total: %3d" % (queue_size, new, added))

logger.info("start")

pending = set([])
results = []

while pending or added < 100:
  lb.spin() # check outstanding tasks

  # finished is the set of msg_ids that are complete
  finished = pending.difference(lb.outstanding)
  # update pending to exclude those that just finished
  pending = pending.difference(finished)

  # check, if we have to create new tasks
  queue_size = len(lb.outstanding)
  if queue_size < N and added < 100:
    new = N - queue_size + 0
    added += new
    status()
    # create new tasks
    newt = lb.map_async(func, range(new), [ np.random.rand(100000) for _ in range(new) ], chunksize=1)
    map(pending.add, newt.msg_ids)

  # collect results from finished tasks
  for msg_id in finished:
      # we know these are done, so don't worry about blocking
      res = lb.get_result(msg_id)
      #print "job id %s finished on engine %i" % (msg_id, res.engine_id)
      #print "with stdout:"
      #print '    ' + ar.stdout.replace('\n', '\n    ').rstrip()
      #print "and results:"
      
      # note that each job in a map always returns a list of length chunksize
      # even if chunksize == 1
      t = res.result[0]
      logger.debug("result '%s' = %s" % (msg_id, t))
      results.append(t)

  lb.wait(pending, 1e-3)

status()
logger.debug("queues:")
for k,v in lb.queue_status().iteritems():
  logger.debug("%5s: %s" % (k, v)) 

logger.info("results: %s" % results)
logger.info("#results = %s" % len(results))
logger.info("finished")
