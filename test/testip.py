#!/usr/bin/env python
# just testing basic parallelization that will be used in the actual project
import numpy as np
import itertools as it

# setup proper logging
import logging
logger = logging.getLogger('psnobfit')
logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)
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
from IPython.parallel import Client, require
c = Client(profile=options.client_profile)
c.clear() # clears remote engines
c.purge_results('all') # all results are memorized in the hub
lb = c.load_balanced_view()

# MAX number of tasks in total
MAX = 20000
# length of test data, sent over the wire
DIMSIZE = 50
# when adding machines, this is the number of additional tasks
# beyond the number of free machines
new_extra = DIMSIZE / 2
    
# import some packages  (also locally)
with c[:].sync_imports():
  from IPython.utils.timing import time ## time.time & time.clock for cpu time
  #import time
  from random import random
  from numpy import pi, sum
  import numpy
  import math

# the actual function
def func_one(tid, data):
  return tid, 1

def func_sum(tid, data):
  'x is either a number or a list/vector of numbers'
  time.sleep(math.log(1 + random()))
  return tid, numpy.sum(data)

def func_eval(tid, data):
  np = numpy
  v = np.dot(np.cos(numpy.pi + data), np.sin(data + numpy.pi/2))
  v = np.exp(np.abs(v) / len(data))
  #s = np.sin(data[::2] + numpy.pi / 2)
  #c = np.cos(data[1::2])
  #v += np.sum(s) + np.sum(c) #np.append(s,c))
  #time.sleep(1e-3)
  #time.sleep(1e-2 + math.log(1 + random()))
  return tid, v

func = func_eval

# some stats
added       = 0
queue_size  = 0
added       = 0
nb_finished = 0
loops       = 0
tasks_added = 0
cum_sum     = 0
best_x      = None
best_obj    = numpy.infty
last_best   = best_obj

def status():
  global last_best
  s = '*' if last_best != best_obj else ' '
  logger.info("pending %4d | + %2d tasks | total: %4d | finished: %4d | best_obj: %.10f %s" % (queue_size, new, added, nb_finished, best_obj, s))
  last_best = best_obj

logger.info("start")
start_time = time.time()

# pending is the set of jobs we are expecting in each loop
pending = set([])
pending_ts = []
# collects all returns
results = []
allx = dict() # store all x vectors

def gen_points(nb):
  '''
  generates @nb new points, depends on results and allx
  '''
  global results, allx

  def rpoint():
    return 2 * (np.random.rand(DIMSIZE)) # - .5)

  if results:
    cur_best_res = min(results, key = lambda _:_[1])
    cur_best_x = allx[cur_best_res[0]]

  def general_case():
    if np.random.random() < .2:
      return rpoint()
    else:
      rv = (np.random.rand(DIMSIZE) - .5) / 2
      # make it sparse
      sp = np.random.rand(DIMSIZE) < (np.random.random() / 2)  + .5
      rv[sp] = 0
      #import scipy
      #rv = scipy.sparse.rand(DIMSIZE, 1, 0.1)
      return np.minimum(2, np.maximum(0, rv + cur_best_x))

  new_point = general_case if results else rpoint
  return np.array([ new_point() for _ in range(new) ])



while pending or added < MAX: 
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
    tids, vals = range(now, now+new), gen_points(new)
    chunksize = new 
    newt = lb.map_async(func, tids, vals, chunksize=chunksize, ordered=False)
    allx.update(zip(tids, vals))
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
    # we know these are done, so don't worry about blocking
    res = lb.get_result(msg_id)
    nb_finished += len(res.result)
    
    # each job returns a list of length chunksize
    for t in res.result:
      logger.debug("result '%s' = %s" % (msg_id, t))
      results.append(t)
      cum_sum += 1 #t[1] ## just to test how many results come back
      if t[1] < best_obj:
        best_obj = t[1]
        best_x   = allx[t[0]]

  # wait for 'pending' jobs or 1/1000s
  lb.wait(pending, 1e-3)

status()
logger.debug("queues:")
for k,v in sorted(lb.queue_status().iteritems()):
  logger.debug("%5s: %s" % (k, v)) 

logger.info("pending: %s" % pending)
logger.info("added in total:   %s" % added)

#logger.info("results: %s" % sorted([r[0] for r in results]))
logger.info("# machines = %s" % len(c.ids))
logger.info("# results = %s" % len(results))
logger.info("cum_sum = %s" % cum_sum)
logger.info("# total loops %s | of that, %s times tasks were added | %.4f%%" % (loops, tasks_added, tasks_added / float(loops) * 100.))
ttime = time.time() - start_time
evalspersec = added / ttime
logger.info("total time: %s [s] | %.5f [feval/s]" % (ttime, evalspersec))
logger.info("best:")
logger.info(" obj. value: %f" % best_obj)
logger.info(" x:\n%s" % best_x)
logger.info("finished")

#logger.info("pending_ts: %s" % pending_ts)

