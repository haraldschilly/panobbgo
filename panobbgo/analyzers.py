# -*- coding: utf8 -*-
from config import loggers
logger = loggers['analyzers']
from panobbgo_problems import Result
import numpy as np

class Analyzer(object):
  '''
  abstract parent class for all types of analyzers
  '''
  def __init__(self, name = None, q = None, cap = None, start = True):
    name = name if name else self.__class__.__name__
    self._name = name
    self._strategy = None

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

class Best(Analyzer):
  '''
  listens on all results and emits a "new_best" event,
  if a new best point has been found
  '''
  def __init__(self):
    Analyzer.__init__(self)
    self.best = Result(None, np.infty)

  def on_new_result(self, events):
    for event in events:
      r = event.result
      if r.fx < self.best.fx:
        logger.info(u"\u2318 %s | \u0394 %.7f %s" %(r, 0.0, r.who))
        self.best = r
        self.eventbus.publish("new_bestt", best = r)

class Rewarder(Analyzer):
  '''
  listens for new points and rewards the heuristic, e.g. if it is better
  than the currently known best point
  '''
  def __init__(self):
    Analyzer.__init__(self)
    self.best = Result(None, np.infty)

  def on_new_result(self, events):
    for event in events:
      if event.result.fx < self.best.fx:
        pass

  def on_new_bestt(self, events):
    for event in events[::-1]:
      logger.info("rewarder: new best: %s" % self.best)
      self.best = event.best
