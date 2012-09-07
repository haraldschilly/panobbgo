# -*- coding: utf8 -*-
# Copyright 2012 Harald Schilly <harald.schilly@univie.ac.at>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Core
====

This is the core part containing:

- :class:`.Results`: DB of all results
- :class:`.EventBus`
- abstract classes for modules

  - :mod:`.heuristics`
  - :mod:`.analyzers`.

- and most importantly, the :class:`.StrategyBase` which holds everything together and
  subclasses in :mod:`.strategies` implement the strategies.

.. inheritance-diagram:: panobbgo.core

.. codeauthor:: Harald Schilly <harald.schilly@univie.ac.at>
'''
from config import get_config
from panobbgo_lib import Result, Point
from IPython.utils.timing import time
import numpy as np

#
# Result DB
#

class Results(object):
  '''
  List of results w/ notificaton for new results.

  .. Note::

    Later on, maybe this will be a cool database which allows to
    persistenly store past evaluations for a given problem,
    to allow resuming and so on.
  '''
  def __init__(self, strategy):
    self.strategy = strategy
    self.eventbus = strategy.eventbus
    self.problem  = strategy.problem
    self.results = []
    self._last_nb = 0 #for logging
    self.fx_delta_last = None

  def add_results(self, new_results):
    '''
    add one single or a list of new @Result objects.
    * calc some statistics
    * send out new_results & new_result events
    '''
    import heapq
    if isinstance(new_results, Result):
      new_results = [ new_results ]
    assert all(map(lambda _ : isinstance(_, Result), new_results))
    # notification for all recieved results at once
    self.eventbus.publish("new_results", results = new_results)
    for r in new_results:
      heapq.heappush(self.results, r)
      self.eventbus.publish("new_result", result = r)
    if len(self.results) / 100 > self._last_nb / 100:
      #self.info()
      self._last_nb = len(self.results)

  def info(self):
    logger = get_config().get_logger('CORE')
    logger.info("%d results in DB" % len(self.results))

  def __iadd__(self, results):
    self.add_results(results)
    return self

  def __len__(self):
    return len(self.results)

  def n_best(self, n):
    import heapq
    return heapq.nsmallest(n, self.results)

class Module(object):
  '''
  "abstract" parent class for various panobbgo modules, e.g. Heuristic and Analyzer.
  '''
  def __init__(self, name = None):
    name = name if name else self.__class__.__name__
    self._name = name
    self.strategy = None
    self._threads = []

  @property
  def name(self):
    '''
    The module's name.
    '''
    return self._name

  def _init_module(self, strategy):
    '''
    :class:`~panobbgo.strategies.StrategyBase` calls this method.
    '''
    self.strategy = strategy
    self.eventbus = strategy.eventbus
    self.problem  = strategy.problem
    self.results  = strategy.results
    self._init_()
    # only after _init_ it is ready to recieve events
    self.eventbus.register(self)

  def _init_(self):
    '''
    2nd initialization, after registering and hooking up the heuristic.
    e.g. self._problem is available.
    '''
    pass

  def __repr__(self):
    return '%s' % self.name

#
# Heuristic
#

from Queue import Empty, LifoQueue # PriorityQueue

class StopHeuristic(Exception):
  '''
  Used to indicate, that the heuristic has finished and should be ignored.
  '''
  def __init__(self, msg = "stopped"):
    Exception.__init__(self, msg)

class Heuristic(Module):
  '''
  abstract parent class for all types of point generating classes
  '''
  def __init__(self, name = None, q = None, cap = None):
    Module.__init__(self, name)
    self.config = get_config()
    self.logger = self.config.get_logger('HEUR')
    self.cap = cap if cap else get_config().capacity
    self._q = q if q else LifoQueue(self.cap)

    # statistics; performance
    self.performance = 0.0

  def clear_queue(self):
    with self._q.mutex:
      del self._q.queue[:]

  def emit(self, points):
    '''
    this is used in the heuristic's thread.
    '''
    try:
      if points is None: raise StopHeuristic()
      if not isinstance(points, list): points = [ points ]
      for point in points:
        x = self.problem.project(point)
        point = Point(x, self.name)
        self.discount()
        self._q.put(point)
    except StopHeuristic:
      self._stopped = True
      self.logger.info("'%s' heuristic stopped." % self.name)

  def reward(self, reward):
    '''
    Give this heuristic a reward (e.g. when it finds a new point)
    '''
    #logger.debug("Reward of %s for '%s'" % (reward, self.name))
    self.performance += reward

  def discount(self, discount = None):
    '''
    Discount the heuristic's reward. Default is set in the configuration.
    '''
    self.performance *= discount if discount else self.config.discount

  def get_points(self, limit=None):
    '''
    this drains the self._q Queue until @limit
    elements are removed or the Queue is empty.
    for each actually emitted point,
    the performance value is discounted (i.e. "punishment" or "energy
    consumption")
    '''
    new_points = []
    try:
      while limit is None or len(new_points) < limit:
        new_points.append(self._q.get(block=False))
        self.discount()
    except Empty:
      pass
    return new_points

  @property
  def active(self):
    '''
    This is queried by the strategy to determine, if it should still consider
    this "module". This is the case, iff there is still something in it's output queue
    or if there is a chance that there will be something in the future (a thread is running).
    '''
    t = any(t.isAlive() for t in self._threads)
    q = self._q.qsize() > 0
    return t or q

#
# Analyzer
#

class Analyzer(Module):
  '''
  Abstract parent class for all types of analyzers.
  '''
  def __init__(self, name = None):
    Module.__init__(self, name)

#
# EventBus
#

class Event(object):
  '''
  This class holds the data for one single :class:`~.EventBus` event.
  '''
  def __init__(self, **kwargs):
    self._when   = time.time()
    self._kwargs = kwargs
    for k, v in kwargs.iteritems():
      setattr(self, k, v)

  def __repr__(self):
    return "Event[%s]" % self._kwargs

class EventBus(object):
  '''
  This event bus is used to publish and send events.
  E.g. it is used to send information like "new best point"
  to all subscribing heuristics.
  '''
  # pattern for a valid key
  import re
  _re_key = re.compile(r'^[a-z_]+$')

  def __init__(self):
    self._subs = {}
    self.logger = get_config().get_logger('EVBUS')

  @property
  def keys(self):
    '''
    List of all keys where you can send an :class:`Event` to.
    '''
    return self._subs.keys()

  def register(self, target):
    '''
    Registers a given ``target`` for this EventBus instance.
    It needs to have suitable ``on_<key>`` methods.
    For each of them, a :class:`~threading.Thread` is spawn as a daemon.
    '''
    from heuristics import StopHeuristic
    from Queue import Empty, LifoQueue
    from threading import Thread

    # important: this decouples the dispatcher's thread from the actual target
    def run(key, target, drain = False):
      if drain: # not using draining for now, doesn't make much sense
        isfirst = True
        while True:
          # draining the queue... otherwise it might get really huge
          # it's up to the heuristics to only work with the most important points
          events = []
          terminate = False
          try:
            while True:
              event = target._eventbus_events[key].get(block=isfirst)
              terminate |= event._terminate
              events.append(event)
              isfirst = False
          except Empty:
            isfirst = True

          try:
            new_points = getattr(target, 'on_%s' % key)(events)
            # heuristics might call self.emit and/or return a list
            if new_points != None: target.emit(new_points)
            if terminate: raise StopHeuristic("terminated")
          except StopHeuristic, e:
            self.logger.debug("'%s/on_%s' %s -> unsubscribing." % (target.name, key, e.message))
            self.unsubscribe(key, target)
            return

      else: # not draining (default)
        while True:
          try:
            event = target._eventbus_events[key].get(block=True)
            try:
              new_points = getattr(target, 'on_%s' % key)(**event._kwargs)
              # heuristics might call self.emit and/or return a list
              if new_points != None: target.emit(new_points)
              if event._terminate: raise StopHeuristic("terminated")
            except StopHeuristic, e:
              self.logger.debug("'%s/on_%s' %s -> unsubscribing." % (target.name, key, e.message))
              self.unsubscribe(key, target)
              return
          except Exception, e:
            # usually, they only happen during shutdown
            self.logger.critical("Exception: %s in %s: %s" % (key, target, e))
            return

    target._eventbus_events = {}
    # bind all 'on_<key>' methods to events in the eventbus
    import inspect
    for name, _ in inspect.getmembers(target, predicate=inspect.ismethod):
      if not name.startswith("on_"): continue
      key = name[3:]
      self._check_key(key)
      target._eventbus_events[key] = LifoQueue()
      t = Thread(target = run, args = (key, target,),
          name='EventBus::%s/%s'%(target.name, key))
      t.daemon = True
      t.start()
      target._threads.append(t)
      # thread running, now subscribe to events
      self.subscribe(key, target)
      #logger.debug("%s subscribed and running." % t.name)

  def _check_key(self, key):
    if not EventBus._re_key.match(key):
      raise Exception('"%s" key not allowed' % key)

  def subscribe(self, key, target):
    '''
    Called by :meth:`.register`.

    .. Note:: counterpart is :func:`unsubscribe`.
    '''
    self._check_key(key)
    if not key in self._subs:
      self._subs[key] = []

    assert target not in self._subs[key]
    self._subs[key].append(target)

  def unsubscribe(self, key, target):
    '''
    Args:

    - if ``key`` is ``None``, the target is removed from all keys.

    '''
    if key is None:
      for k, v in self._subs.iteritems():
        for t in v:
          if t is target:
            self.unsubscribe(k, t)
      return

    self._check_key(key)
    if not key in self._subs:
      self.logger.critical("cannot unsubscribe unknown key '%s'" % key)
      return

    if target in self._subs[key]:
      self._subs[key].remove(target)

  def publish(self, key, event = None, terminate = False, **kwargs):
    '''
    This is used to send 

    Args:

    - ``terminate``: if True, the associated thread will end.
                     (use it for ``on_start`` and similar).
    - ``event``: if set, this given :class:`.Event` is sent (and not a new one created).
    - ``**kwargs``: any additional keyword arguments are stored inside the Event
                    if ``event`` is ``None``.
    '''
    if key not in self._subs:
      self.logger.warning("key '%s' unknown." % key)
      return

    for target in self._subs[key]:
      event = Event(**kwargs) if event == None else event
      event._terminate = terminate
      #logger.info("EventBus: publishing %s -> %s" % (key, event))
      target._eventbus_events[key].put(event)

#
# Strategy
#

class StrategyBase(object):
  '''
  This abstract BaseStrategy is the parent class of all Strategies.

  Use it this way:

  #. Subclass it, write your optional initializer, *afterwards* call the initializer
     of this class (it will start its the main loop).

  #. Overwrite the :meth:`.execute`, which returns a list of new search points
     (by requesting them from the :mod:`~panobbgo.heuristics` via the
     :meth:`~panobbgo.core.Heuristic.get_points` method) and might
     also emit :class:`Events <panobbgo.core.Event>`.

  This ``execute`` method will be called repeatedly as long as there are less than the
  given maximum number of search points evaluated.
  '''
  # constant reference id for sending the evaluation code to workers
  PROBLEM_KEY = "problem"

  def __init__(self, problem, heurs):
    self._name = name = self.__class__.__name__
    #threading.Thread.__init__(self, name=name)
    self.config = config = get_config()
    self.logger = logger = config.get_logger('STRAT')
    self.slogger = config.get_logger('STATS')
    logger.info("Init of '%s' w/ %d heuristics." % (name, len(heurs)))
    logger.debug("Heuristics %s" % heurs)
    logger.info("%s" % problem)

    # statistics
    self.cnt         = 0 # show info about evaluated points
    self.show_last   = 0 # for printing the info line in _add_tasks()
    self.time_start  = time.time()
    self.tasks_walltimes = {}

    # task accounting (tasks != points !!!)
    self.per_client   = 1 # #tasks per client in 'chunksize'
    self.pending      = set([])
    self.new_finished = []
    self.finished     = []

    # init & start everything
    self._setup_cluster(0, problem)
    self.problem     = problem
    self.eventbus    = EventBus()
    self.results     = Results(self)

    # heuristics
    import collections
    self._heuristics = collections.OrderedDict()
    map(self.add_heuristic, sorted(heurs, key = lambda h : h.name))

    # analyzers
    from analyzers import Best, Rewarder, Grid, Splitter
    best = Best()
    self._analyzers = {
        'best' :     best,
        'rewarder' : Rewarder(),
        'grid':      Grid(),
        'splitter':  Splitter()
    }
    map(lambda a : a._init_module(self), self._analyzers.values())

    logger.debug("Eventbus keys: %s" % self.eventbus.keys)

    try:
      import threading
      if isinstance(self, threading.Thread):
        raise Exception("change run() to start()")
      self.run()
    except KeyboardInterrupt:
      logger.critical("KeyboardInterrupt recieved, e.g. via Ctrl-C")
      self._cleanup()

  @property
  def heuristics(self):
    return filter(lambda h : h.active, self._heuristics.values())

  def heuristic(self, who):
    return self._heuristics[who]

  def analyzer(self, who):
    return self._analyzers[who]

  def add_heuristic(self, h):
    name = h.name
    assert name not in self._heuristics, \
      "Names of heuristics need to be unique. '%s' is already used." % name
    self._heuristics[name] = h
    h._init_module(self)

  def add_analyzer(self, a):
    name = a.name
    assert key not in self._analyzers, \
        "Names of analyzers need to be unique. '%s' is already used." % key
    self._analyzers[name] = a
    a._init_module(self)

  def _setup_cluster(self, nb_gens, problem):
    from IPython.parallel import Client
    c = self._client = Client(profile=self.config.ipy_profile)
    c.clear() # clears remote engines
    c.purge_results('all') # all results are memorized in the hub

    if len(c.ids) < nb_gens + 1:
      raise Exception('I need at least %d clients.' % (nb_gens + 1))
    dv_evaluators = c[nb_gens:]
    dv_evaluators[StrategyBase.PROBLEM_KEY] = problem
    self.generators = c.load_balanced_view(c.ids[:nb_gens])
    self.evaluators = c.load_balanced_view(c.ids[nb_gens:])
    self.direct_view = c.ids[:]
    # TODO remove this hack. "problem" wasn't pushed to all clients
    #time.sleep(1e-1)

    # import some packages (also locally)
    #with c[:].sync_imports():
    #  import numpy
    #  import math

  @property
  def best(self): return self._analyzers['best'].best

  def run(self):
    self.eventbus.publish('start', terminate=True)
    from IPython.parallel import Reference
    prob_ref = Reference(StrategyBase.PROBLEM_KEY) # see _setup_cluster
    self._start = time.time()
    self.logger.info("Strategy '%s' started" % self._name)
    self.loops = 0
    while True:
      self.loops += 1

      points = self.execute()

      # distribute work
      new_tasks = self.evaluators.map_async(prob_ref, points, \
                  chunksize = self.per_client, ordered=False)

      # don't forget, this updates the statistics - new_tasks's default is "None"
      self._add_tasks(new_tasks)

      # collect new results for each finished task, hand them over to result DB
      for msg_id in self.new_finished:
        for r in self.evaluators.get_result(msg_id).result:
          self.results += r

      self.per_client = max(1, int(min(self.config.max_eval / 50, 1.0 / self.avg_time_per_task)))

      # show heuristic performances after each round
      #logger.info('  '.join(('%s:%.3f' % (h, h.performance) for h in heurs)))

      # stopping criteria
      if len(self.results) > self.config.max_eval: break

      # limit loop speed
      self.evaluators.wait(None, 1e-3)

    self._cleanup()

  def execute(self):
    '''
    Overwrite this method when you extend this base strategy.
    '''
    raise Exception('You need to extend the class StrategyBase and overwrite this execute method.')

  def _cleanup(self):
    '''
    cleanup + shutdown
    '''
    self._end = time.time()
    for msg_id in self.evaluators.outstanding:
      try:
        self.evaluators.get_result(msg_id).abort()
      except:
        pass
    self.logger.info("Strategy '%s' finished after %.3f [s] and %d loops." \
             % (self._name, self._end - self._start, self.loops))

    self.info()
    self.results.info()

  def _add_tasks(self, new_tasks):
    '''
    Accounting routine for the parallel tasks, only used by :meth:`.run`.
    '''
    if new_tasks != None:
      for mid in new_tasks.msg_ids:
        self.pending.add(mid)
        self.cnt += len(self.evaluators.get_result(mid).result)
    self.new_finished = self.pending.difference(self.evaluators.outstanding)
    self.pending = self.pending.difference(self.new_finished)
    for tid in self.new_finished:
       self.finished.append(tid)
       self.tasks_walltimes[tid] = self.evaluators.get_result(tid).elapsed

    #if self.cnt / 100 > self.show_last / 100:
    if time.time() - self.show_last > self.config.show_interval:
      self.info()
      self.show_last = time.time() #self.cnt

  def info(self):
    avg   = self.avg_time_per_task
    pend  = len(self.pending)
    fini  = len(self.finished)
    peval = len(self.results)
    self.slogger.info("%4d (%4d) pnts | Tasks: %3d pend, %3d finished | %6.3f [s] cpu, %6.3f [s] wall, %6.3f [s/task]" %
               (peval, self.cnt, pend, fini, self.time_cpu, self.time_wall, avg))

  @property
  def avg_time_per_task(self):
    if len(self.tasks_walltimes) > 1:
      return np.average(self.tasks_walltimes.values())
    self.slogger.warning("avg time per task for 0 tasks!")
    return 0.01

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

