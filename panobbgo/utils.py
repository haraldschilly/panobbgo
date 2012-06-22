# -*- coding: utf-8 -*-
import config
import logging

def create_logger(name, level):
  '''
  creates logger with @name and @level logging level
  '''
  logger = logging.getLogger('psnobfit')
  logger.setLevel(logging.DEBUG)
  log_stream_handler = logging.StreamHandler()
  log_stream_handler.setLevel(level)
  log_formatter = logging.Formatter('%(asctime)s %(name)s/%(levelname)-9s %(message)s')
  log_stream_handler.setFormatter(log_formatter)
  logger.addHandler(log_stream_handler)
  return logger

logger = create_logger('psnobfit', config.loglevel)

