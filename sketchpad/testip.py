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

# START: create remote evaluators and a few (or one) special one for #
# generating new points
logger.info("init")
from IPython.parallel import Client, require
c = Client(profile=options.client_profile)
c.clear() # clears remote engines
c.purge_results('all') # all results are memorized in the hub

if len(c.ids) < 2:
  raise Exception('I need at least 2 clients.')
nbGens = min(1, len(c.ids) - 1)
generators = c.load_balanced_view(c.ids[:nbGens])
evaluators = c.load_balanced_view(c.ids[nbGens:])

# MAX number of tasks in total
MAX = 5000
# length of test data, sent over the wire
DIMSIZE = 10
# when adding machines, this is the number of additional tasks
# beyond the number of free machines
new_extra = DIMSIZE 

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
  data = data * numpy.pi  / 2
  v = np.multiply(np.cos(numpy.pi + data), np.sin(data + numpy.pi/2))
  v = np.exp(np.linalg.norm(v - 1, 1) / len(data))
  #s = np.sin(data[::2] + numpy.pi / 2)
  #c = np.cos(data[1::2])
  #v += np.sum(s) + np.sum(c) #np.append(s,c))
  #time.sleep(1e-3)
  #time.sleep(1e-2 + math.log(1 + random()))
  return tid, v

func = func_eval

# some stats
added        = 0
queue_size   = 0
added        = 0
nb_finished  = 0
nb_generated = 0
loops        = 0
tasks_added  = 0
cum_sum      = 0
best_x       = None
best_obj     = numpy.infty
last_best    = best_obj

def status():
  global last_best
  s = '*' if last_best != best_obj else ' '
  logger.info("pend %4d | + %2d | tot: %4d | finished: %4d | gen: %3d | best_obj: %.10f %s" % (queue_size, new, added, nb_finished, nb_generated, best_obj, s))
  last_best = best_obj

logger.info("start")
start_time = time.time()

# pending is the set of jobs we are expecting in each loop
pending = set([])
pending_generators = set([])
new_points = []
# collects all returns
results = []
allx = dict() # store all x vectors

def gen_points(new, DIMSIZE, cur_best_res = None, cur_best_x = None):
  '''
  generates @new new points, depends on results and allx
  '''
  np = numpy
  #lambda rp : 10 * (np.random.rand(DIMSIZE) )
  FACT = 3
  OFF  = 0

  if np.random.random() < .2 or not cur_best_res:
    return np.array([FACT * (np.random.rand(DIMSIZE) + OFF) for _ in range(new)])

  # better local value new best point
  ret = []
  for i in range(new):
    rv = (np.random.rand(DIMSIZE) - .5) / 5
    # make it sparse
    sp = np.random.rand(DIMSIZE) < .9 
    rv[sp] = 0
    #import scipy
    #rv = scipy.sparse.rand(DIMSIZE, 1, 0.1)
    ret.append(np.minimum(2, np.maximum(0, rv + cur_best_x)))
  return np.array(ret)


# itertools counter for successive task ID numbers
tid_counter = it.count(0)

while pending or added < MAX: 
  evaluators.spin() # check outstanding tasks
  loops += 1

  # get new points if they have arrived

  # check if we have to generate new points
  if not new_points:
    if results:
      cur_best_res = min(results, key = lambda _:_[1])
      cur_best_x = allx[cur_best_res[0]]
    else:
      cur_best_res, cur_best_x = None, None
    new = len(c.ids) - queue_size + new_extra
    # at the end, make sure to not add more tasks then MAX
    new = min(new, MAX - added)
    # update the counter
    added += new
    new_points_tasks = generators.map_async(gen_points, [new], [DIMSIZE], [cur_best_res], [cur_best_x], ordered=False)
    #print ">>>", new_points_tasks.msg_ids
    map(pending_generators.add, new_points_tasks.msg_ids)

  finished_generators = pending_generators.difference(generators.outstanding)
  pending_generators  = pending_generators.difference(finished_generators)

  # if we have generated points in the queue, eval the function
  for msg_id in finished_generators:
    res = generators.get_result(msg_id)
    nb_generated += len(res.result)
    for g in res.result:
      #logger.info('new points "%s" = %s' % (msg_id, g))
      cs = max(1, min(5, len(res.result)))
      newt = evaluators.map_async(func, tids, vals, chunksize = cs, ordered=False)
      cum_sum += 1

  # check, if we have to create new tasks
  queue_size = len(pending)
  if queue_size <= len(c.ids) + new_extra and added < MAX:
    tasks_added += 1
    new = len(c.ids) - queue_size + new_extra
    # at the end, make sure to not add more tasks then MAX
    new = min(new, MAX - added)
    # update the counter
    added += new
    status()
    # create new tasks
    tids, vals = list(it.islice(tid_counter, new)), gen_points(new, DIMSIZE)
    chunksize = max(1, min(new, len(c.ids)))
    newt = evaluators.map_async(func, tids, vals, chunksize=chunksize, ordered=False)
    allx.update(zip(tids, vals))
    map(pending.add, newt.msg_ids)
  else:
    new = 0

  # finished is the set of msg_ids that are complete
  finished = pending.difference(evaluators.outstanding)
  # update pending to exclude those that just finished
  pending = pending.difference(finished)

  # collect results from finished tasks
  for msg_id in finished:
    # we know these are done, so don't worry about blocking
    res = evaluators.get_result(msg_id)
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
  evaluators.wait(None, 1e-3) #pending.union(pending_generators), 1e-3)

status()
logger.debug("queues:")
for k,v in sorted(evaluators.queue_status().iteritems()):
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


