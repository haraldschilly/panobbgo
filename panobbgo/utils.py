# -*- coding: utf-8 -*-

def create_logger(name, level):
  '''
  creates logger with @name and @level logging level
  '''
  import logging
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  log_stream_handler = logging.StreamHandler()
  log_stream_handler.setLevel(level)
  log_formatter = logging.Formatter('%(asctime)s %(name)-5s/%(levelname)-7s %(message)s')
  log_stream_handler.setFormatter(log_formatter)
  logger.addHandler(log_stream_handler)
  return logger

def info():
  '''
  show a bit of info
  '''
  def version(what):
    m = __import__(what)
    print "%s: %s" % (what, m.__version__)

  version("numpy")
  version("scipy")
  version("IPython")
  version("matplotlib")

