# -*- coding: utf-8 -*-
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

r'''
Utilities
---------

Some utility functions, will move eventually.
'''

import logging


class ColoredFormatter(logging.Formatter):

    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[0;%dm"
    COLOR_SEQ_BOLD = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    COLORS = {
        'DEBUG': BLUE,
        #'INFO': WHITE,
        'WARNING': YELLOW,
        'CRITICAL': MAGENTA,
        'ERROR': RED
    }

    def __init__(self):
        msg = '%(runtime)f %(where)-15s $BOLD%(name)-5s$RESET %(levelname)-9s %(message)s'
        msg = msg.replace("$RESET", ColoredFormatter.RESET_SEQ).replace(
            "$BOLD", ColoredFormatter.BOLD_SEQ)
        logging.Formatter.__init__(self, fmt=msg)

    @staticmethod
    def colorize(string, color, bold=False):
        cs = ColoredFormatter.COLOR_SEQ_BOLD if bold else ColoredFormatter.COLOR_SEQ
        string = '%s%s%s' % (
            cs % (30 + color), string, ColoredFormatter.RESET_SEQ)
        string = "%-20s" % string
        return string

    def format(self, record):
        levelname = record.levelname
        if levelname in ColoredFormatter.COLORS:
            col = ColoredFormatter.COLORS[levelname]
            record.name = self.colorize(record.name, col, True)
            record.lineno = self.colorize(record.lineno, col, True)
            record.levelname = self.colorize(levelname, col, True)
            record.msg = self.colorize(record.msg, col)
        return logging.Formatter.format(self, record)


class PanobbgoContext(logging.Filter):

    def __init__(self):
        logging.Filter.__init__(self)
        from IPython.utils.timing import time
        self._start = time.time()

    def filter(self, record):
        from IPython.utils.timing import time
        record.runtime = time.time() - self._start
        record.where = "%s:%s" % (record.filename[:-3], record.lineno)
        return True


def create_logger(name, level=logging.INFO):
    '''
    Creates logger with ``name`` and given ``level`` logging level.
    '''
    logger = logging.getLogger(name)
    logger.addFilter(PanobbgoContext())
    logger.setLevel(logging.DEBUG)
    log_stream_handler = logging.StreamHandler()
    log_stream_handler.setLevel(level)
    log_formatter = ColoredFormatter()
    log_stream_handler.setFormatter(log_formatter)
    logger.addHandler(log_stream_handler)
    return logger


def info():
    '''
    Shows a bit of info about the libraries and other environment information.
    '''
    import subprocess
    git = subprocess.Popen(
        ["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
    v = {}

    def version(what):
        m = __import__(what)
        v[what] = m.__version__

    version("numpy")
    version("scipy")
    version("IPython")
    try:
        version("matplotlib")
    except:
        print "matplotlib broken :("
    v['git HEAD'] = git.communicate()[0].splitlines()[0]
    return v


def is_left(p0, p1, ptest):
    return is_right(p1, p0, ptest)


def is_right(p0, p1, ptest):
    '''
    p0->p1 existing results, ptest other point
    return true, if ptest is on the right of p0->p1

    Args::

      - p0, p1, ptest: :class:`numpy.ndarray`.
    '''
    import numpy as np
    v1 = p1 - p0
    v2 = ptest - p0
    return np.linalg.det(np.vstack([v1, v2])) < 0


class memoize(object):

    """
    Caches the return value of a method inside the instance's function!

    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.

    Usage::

      class Obj(object):
        @memoize
        def method(self, hashable):
          result = do_calc(...)
          return result

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method::

      class Obj(object):
        @memoize
        def add_to(self, arg):
          return self + arg
      Obj.add_to(1) # not enough arguments
      Obj.add_to(1, 2) # returns 3, result is not cached

    Derived from `ActiveState 577432 <http://code.activestate.com/recipes/577452-a-memoize-decorator-for-instance-methods/>`_.
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, tpe):
        if obj is None:
            return self.func
        from functools import partial  # , update_wrapper
        p = partial(self, obj)
        # update_wrapper(p, self.func)
        return p

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res

# Testing

import unittest
import mock


class MockupEventBus(object):

    def __init__(self):
        self.targets = []

    def register(self, who):
        self.targets.append(who)


# class MockupStrategy(object):
#
#    def __init__(self, problem):
#        self.problem = problem
#        self._eventbus = MockupEventBus()
#
#    @property
#    def eventbus(self):
#        return self._eventbus

class PanobbgoTestCase(unittest.TestCase):

    def __init__(self, name):
        import panobbgo.config
        panobbgo.config.PARSE_ARGS = False
        unittest.TestCase.__init__(self, name)

    def setUp(self):
        from panobbgo_lib.classic import Rosenbrock
        self.problem = Rosenbrock(2)
        # self.strategy = MockupStrategy(self.problem)

    @mock.patch('panobbgo.core.StrategyBase')
    def init_strategy(self, heur, StrategyBaseMock):
        strategy = StrategyBaseMock(self.problem, [heur])
        return strategy
