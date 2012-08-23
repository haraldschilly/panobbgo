# -*- coding: utf8 -*-

'''
This is the core part, currently only managing the global
DB of point evaluations. For more, look into the strategy.py file.
'''
import config
logger = config.loggers['core']
from panobbgo_problems import Result

#
# Result DB
#

class Results(object):
  '''
  List of results w/ notificaton for new results.
  Later on, this will be a cool database.
  '''
  def __init__(self, strategy):
    import numpy as np
    self._strategy = strategy
    self._results = []
    self._last_nb = 0 #for logging
    self.fx_delta_last = None
    self._best = Result(None, np.infty)

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
      heapq.heappush(self._results, r)
      self.eventbus.publish("new_result", result = r)
    if len(self._results) / 100 > self._last_nb / 100:
      #self.info()
      self._last_nb = len(self._results)

  def info(self):
    logger.info("%d results in DB" % len(self._results))

  def __iadd__(self, results):
    self.add_results(results)
    return self

  def __len__(self):
    return len(self._results)

  @property
  def best(self): return self._best

  @property
  def strategy(self): return self._strategy

  @property
  def eventbus(self): return self._strategy.eventbus

  @property
  def problem(self): return self._strategy.problem

  def n_best(self, n):
    import heapq
    return heapq.nsmallest(n, self._results)

class Module(object):
  '''
  "abstract" parent class for various panobbgo modules, e.g. Heuristic and Analyzer.
  '''
  def __init__(self, name = None):
    name = name if name else self.__class__.__name__
    self._name = name
    self._strategy = None
    self._threads = []

  def _init_(self):
    '''
    2nd initialization, after registering and hooking up the heuristic.
    e.g. self._problem is available.
    '''
    pass

  def __repr__(self):
    return '%s' % self.name

  @property
  def strategy(self): return self._strategy

  @property
  def eventbus(self): return self._strategy.eventbus

  @property
  def problem(self): return self._strategy.problem

  @property
  def results(self): return self._strategy.results

  @property
  def name(self): return self._name

#
# Heuristic
#
from Queue import Empty, LifoQueue # PriorityQueue
from panobbgo_problems import Point

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
    self._q = q if q else LifoQueue(cap)

    # statistics; performance
    self._performance = 0.0

  def emit(self, points):
    '''
    this is used in the heuristic's thread.
    '''
    try:
      if points == None: raise StopHeuristic()
      if not isinstance(points, list): points = [ points ]
      for point in points:
        x = self.problem.project(point)
        point = Point(x, self.name)
        self.discount()
        self._q.put(point)
    except StopHeuristic:
      self._stopped = True
      logger.info("'%s' heuristic stopped." % self.name)

  def reward(self, reward):
    '''
    Give this heuristic a reward (e.g. when it finds a new point)
    '''
    #logger.debug("Reward of %s for '%s'" % (reward, self.name))
    self._performance += reward

  def discount(self, discount = config.discount):
    '''
    Discount the heuristic's reward. Default is set in the configuration.
    '''
    self._performance *= discount

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
  def performance(self): return self._performance

  @property
  def active(self):
    '''
    This is queried by the strategy to determine, if it should still consider
    this "module". This is the case, ifthere is still something in it's output queue
    or f there is a chance that there will be something in the future (a thread is running).
    '''
    t = any(t.isAlive() for t in self._threads)
    q = self._q.qsize() > 0
    return t or q

#
# Analyzer
#

class Analyzer(Module):
  '''
  abstract parent class for all types of analyzers
  '''
  def __init__(self, name = None):
    Module.__init__(self, name)

#
# EventBus
#

class Event(object):
  '''
  This class holds the data for one single @EventBus event.
  '''
  def __init__(self, **kwargs):
    from IPython.utils.timing import time
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

  @property
  def keys(self): return self._subs.keys()

  def register(self, target):
    '''
    registers a target for this event bus instance. it needs to have
    "on_<key>" methods.
    '''
    from heuristics import StopHeuristic
    from Queue import Empty, LifoQueue
    from threading import Thread

    # important: this decouples the dispatcher's thread from the actual target
    def run(key, target):
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
          logger.info("'%s/on_%s' %s -> unsubscribing." % (target.name, key, e.message))
          self.unsubscribe(key, target)
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
          name='EventBus: %s/%s'%(target.name, key))
      t.daemon = True
      t.start()
      target._threads.append(t)
      # thread running, now subscribe to events
      self.subscribe(key, target)
      #logger.debug("%s subscribed and running." % t.name)

  def _check_key(self, key):
    if not EventBus._re_key.match(key):
      raise Exception('EventBus: "%s" key not allowed' % key)

  def subscribe(self, key, target):
    self._check_key(key)
    if not key in self._subs:
      self._subs[key] = []

    assert target not in self._subs[key]
    self._subs[key].append(target)

  def unsubscribe(self, key, target):
    '''
      - if @key is None, the target is removed from all keys
        (used in Heuristic.stop_me(...))
    '''
    if key is None:
      for k, v in self._subs.iteritems():
        for t in v:
          if t is target:
            self.unsubscribe(k, t)
      return

    self._check_key(key)
    if not key in self._subs:
      logger.critical("cannot unsubscribe unknown key '%s'" % key)
      return

    if target in self._subs[key]:
      self._subs[key].remove(target)

  def publish(self, key, e = None, terminate = False, **kwargs):
    '''
     - terminate: if True, the associated thread will end.
    '''
    if key not in self._subs:
      logger.warning("EventBus: key '%s' unknown." % key)
      return

    for target in self._subs[key]:
      event = e if e else Event(**kwargs)
      event._terminate = terminate
      #logger.info("EventBus: publishing %s -> %s" % (key, event))
      target._eventbus_events[key].put(event)
