#!/usr/bin/env python
# just testing basic parallelization that will be used in the actual project
import time
import numpy as np
import itertools as it

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

# MAX number of tasks in total
MAX = 1000
# length of test data, sent over the wire
DSIZE = 1000
    
# import everything
with c[:].sync_imports():
  #from IPython.utils.timing import time
  import time
  from random import random
  from numpy import pi, sum
  import math


# the actual function
def func(tid, data):
  'x is either a number or a list/vector of numbers'
  #time.sleep(math.log(2 + random()))
  return tid, sum(data)

added = 0
queue_size = 0
added = 0
nb_finished = 0

def status():
  logger.info("pending %3d | adding %2d tasks | added: %3d | finished: %3d" % (queue_size, new, added, nb_finished))

logger.info("start")

pending = set([])
results = []

while pending or added < MAX:
  lb.spin() # check outstanding tasks

  # check, if we have to create new tasks
  queue_size = len(pending)
  if queue_size < len(c.ids) and added < MAX:
    now = added
    new = len(c.ids) - queue_size + 0
    # at the end, make sure to not add more tasks then MAX
    new = min(new, MAX - added)
    # update the counter
    added += new
    status()
    # create new tasks
    tids, vals = range(now, now+new), [ np.random.rand(DSIZE) for _ in range(new) ]
    newt = lb.map_async(func, tids, vals, chunksize=1)
    map(pending.add, newt.msg_ids)
  else:
    new = 0

  # finished is the set of msg_ids that are complete
  finished = pending.difference(lb.outstanding)
  # update pending to exclude those that just finished
  pending = pending.difference(finished)

  # collect results from finished tasks
  for msg_id in finished:
      nb_finished += 1
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

logger.info("pending: %s" % pending)
logger.info("added:   %s" % added)

#logger.info("results: %s" % sorted([r[0] for r in results]))
logger.info("#results = %s" % len(results))
logger.info("finished")
