#!/usr/bin/env python
# just testing basic parallelization that will be used in the actual project
import time
import numpy as np
import itertools as it

# setup proper logging
import logging
logger = logging.getLogger('psnobfit')
logger.setLevel(logging.INFO) #DEBUG)
log_stream_handler = logging.StreamHandler()
log_stream_handler.setLevel(logging.DEBUG)
log_formatter = logging.Formatter('%(asctime)s %(name)s/%(levelname)-9s %(message)s')
log_stream_handler.setFormatter(log_formatter)
logger.addHandler(log_stream_handler)
del logging, log_stream_handler, log_formatter

# read cmd line options
from optparse import OptionParser
opt_parser = OptionParser()
opt_parser.add_option("-p", "--profile", dest="client_profile", default="unissh", action="store_const",
                      help="the profile to use for ipython.parallel")
options, args = opt_parser.parse_args()

# START
logger.info("init")
from IPython.parallel import Client
c = Client(profile=options.client_profile)
c.clear() # clears remote engines
c.purge_results('all') # all results are memorized in the hub
lb = c.load_balanced_view()

# MAX number of tasks in total
MAX = 100
# length of test data, sent over the wire
DSIZE = 100
# when adding machines, this is the number of additional tasks
# beyond the number of free machines
new_extra = 2
    
# import some packages  (also locally)
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

# some stats
added       = 0
queue_size  = 0
added       = 0
nb_finished = 0
cum_sum     = 0
loops       = 0
tasks_added = 0

def status():
  logger.info("pending %4d | + %2d tasks | total added: %4d | finished: %4d | cum_sum: %.2f" % (queue_size, new, added, nb_finished, cum_sum))

logger.info("start")

# pending is the set of jobs we are expecting in each loop
pending = set([])
pending_ts = []
# collects all returns
results = []

while pending or added < MAX: #cum_sum < MAX *  DSIZE / 2:
  lb.spin() # check outstanding tasks
  loops += 1

  # check, if we have to create new tasks
  queue_size = len(pending)
  if queue_size <= len(c.ids) and added < MAX:
    tasks_added += 1
    now = added
    new = len(c.ids) - queue_size + new_extra
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
  pending_ts.append(len(pending))

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
      cum_sum += t[1]
      results.append(t)

  lb.wait(pending, 1e-3)

status()
logger.debug("queues:")
for k,v in sorted(lb.queue_status().iteritems()):
  logger.debug("%5s: %s" % (k, v)) 

logger.debug("pending: %s" % pending)
logger.info("added in total:   %s" % added)

#logger.info("results: %s" % sorted([r[0] for r in results]))
logger.info("nb of machines = %s" % len(c.ids))
logger.info("nb of results = %s | cum_sum = %.2f" % (len(results), cum_sum))
logger.info("nb of total loops %s | of that, %s times tasks were added | %.4f%%" % (loops, tasks_added, tasks_added / float(loops) * 100.))
logger.info("finished")

#logger.info("pending_ts: %s" % pending_ts)

