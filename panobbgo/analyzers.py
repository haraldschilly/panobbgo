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

  def _reward_heuristic(self, result):
    '''
    for each new result, decide if there is a reward and also
    calculate its amount.
    '''
    import numpy as np
    # currently, only reward if better point found.
    # TODO in the future also reward if near the best value (but
    # e.g. not in the proximity of the best x)
    fx_delta, reward = 0.0, 0.0
    if result.fx < self.best.fx:
      # ATTN: always take care of self.best.fx == np.infty
      #fx_delta = np.log1p(self.best.fx - r.fx) # log1p ok?
      fx_delta = 1.0 - np.exp(-1.0 * (self.best.fx - result.fx)) # saturates to 1
      #if self.fx_delta_last == None: self.fx_delta_last = fx_delta
      reward = fx_delta #/ self.fx_delta_last
      self.strategy._heurs[result.who].reward(reward)
      #self.fx_delta_last = fx_delta
    return reward

  def on_new_result(self, events):
    for event in events:
      if event.result.fx < self.best.fx:
        reward = self._reward_heuristic(event.result)
        logger.info("rewarder: %s for %s" % (reward, event.result.who))

  def on_new_bestt(self, events):
    for event in events[::-1]:
      #logger.info("rewarder: new best: %s" % self.best)
      self.best = event.best
