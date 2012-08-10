# -*- coding: utf-8 -*-
import logging

def create_logger(name, level = logging.INFO):
  '''
  creates logger with @name and @level logging level
  '''
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  log_stream_handler = logging.StreamHandler()
  log_stream_handler.setLevel(level)
  log_formatter = logging.Formatter('%(asctime)s %(name)-5s/%(levelname)-8s %(message)s')
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
  version("matplotlib")
  return v

