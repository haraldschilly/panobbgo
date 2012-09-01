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

import logging

def create_logger(name, level = logging.INFO):
  '''
  creates logger with @name and @level logging level
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
  show a bit of info
  '''
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
  return v

