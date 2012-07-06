# -*- coding: utf8 -*-
"""
This file must not depend on other files from this project.
It's purpose is to parse a config file (create a default one if none
is present) and replace values stored within it with those given
via optional command-line arguments.
"""

default_config_fn = 'config.ini'

# 1: parsing command-line arguments
from optparse import OptionParser
_parser = OptionParser()
_parser.add_option('-c', '--config-file', dest="config_file",
                  help='configuration file [default: %default]', default=default_config_fn)
_parser.add_option('-p', '--profile', dest='ipy_profile', 
                  help='IPython profile for the ipcluster configuration')
_parser.add_option('--max', dest='max_eval', help="maximum number of evaluations", type="int")
_parser.add_option("-v", action="count", dest="verbosity", help="verbosity level: -v, -vv, or -vvv")

_options, _args = _parser.parse_args()

print 'cmdln options: %s' % _options

import os
from ConfigParser import ConfigParser

# 2/1: does config file exist?
if not os.path.exists(_options.config_file):
  _cfgp = ConfigParser()
  # create them in the reverse order

  _cfgp.add_section('db') # database config
  #_cfgp.set('db', 'port', '37010')
  #_cfgp.set('db', 'host', 'localhost')

  _cfgp.add_section('ipython')
  _cfgp.set('ipython', 'profile', 'default')

  _cfgp.add_section('core') # core configuration
  _cfgp.set('core', 'loglevel', '40') # default: no debug mode
  _cfgp.set('core', 'max_eval', '1000')

  with open(_options.config_file, 'wb') as configfile:
    _cfgp.write(configfile)

# 2/2: reading the config file
_cfgp = ConfigParser()
_cfgp.read(_options.config_file)

# 3: override specific settings
_cur_verb = _cfgp.getint('core', 'loglevel')
if _options.verbosity:   _cfgp.set('core', 'loglevel', str(_cur_verb - 10*_options.verbosity))
if _options.max_eval:    _cfgp.set('core', 'max_eval', str(_options.max_eval))
if _options.ipy_profile: _cfgp.set('ipython', 'profile', _options.ipy_profile)

## some generic function
def get_config(section, key):
  return _cfgp.get(section, key)

def all(sep = '::'):
  ret = {}
  for s in _cfgp.sections():
    for k, v in _cfgp.items(s):
      ret['%s%s%s' % (s, sep, k)] = v
  return ret

print 'config.ini: %s' % all()

## specific data
loglevel    = _cfgp.getint('core', 'loglevel')
max_eval    = _cfgp.getint('core', 'max_eval')
ipy_profile = _cfgp.get('ipython', 'profile')

print 'loglevel: %s' % loglevel
print 'ipython profile: %s' % ipy_profile
