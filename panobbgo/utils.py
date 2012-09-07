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

def create_logger(name, level = logging.INFO):
  '''
  Creates logger with ``name`` and given ``level`` logging level.
  '''
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  log_stream_handler = logging.StreamHandler()
  log_stream_handler.setLevel(level)
  log_formatter = logging.Formatter('%(asctime)s %(name)-4s/%(levelname)-8s %(message)s')
  log_stream_handler.setFormatter(log_formatter)
  logger.addHandler(log_stream_handler)
  return logger

def info():
  '''
  Shows a bit of info about the libraries and other environment information.
  '''
  import subprocess
  git = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
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

  def __get__(self, obj):
    if obj is None:
      return self.func
    from functools import partial
    return partial(self, obj)

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
