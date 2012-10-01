# -*- coding: utf8 -*-
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

r"""
Configuration
=============

It's purpose is to parse a config file (create a default one if none
is present) and replace values stored within it with those given
via optional command-line arguments.

.. Note::

  This will also hold a class for configuring the Panobbgo framework
  in general. I.e. modules declare other modules as dependencies, etc...

.. inheritance-diagram:: panobbgo.configuration
"""

_config = None

class Config(object):

  def __init__(self):
    self._loggers = {}
    self._create()

  def _create(self):
    from utils import info, create_logger
    logger = create_logger("CONFG")

    defaultself_fn = 'config.ini'

    # 1: parsing command-line arguments
    from optparse import OptionParser
    _parser = OptionParser()
    _parser.add_option('-c', '--config-file',
        dest="config_file",
        help='configuration file [default: %default]',
        default=defaultself_fn)

    _parser.add_option('-p', '--profile',
        dest='ipy_profile',
        help='IPython profile for the ipcluster configuration')

    _parser.add_option('--max',
        dest='max_eval',
        help="maximum number of evaluations",
        type="int")

    _parser.add_option('--smooth',
        dest='smooth',
        help="smoothing parameter for (additive or other) smoothing",
        type="float")

    _parser.add_option('--cap',
        dest='capacity',
        help="capacity for each queue in each heuristic",
        type="int")

    _parser.add_option("-v",
        action="count",
        dest="verbosity",
        help="verbosity level: -v, -vv, or -vvv")

    _parser.add_option('--ui',
        dest='ui',
        action='store_true',
        default=False,
        help='If specified, the GTK+/matplotlib based UI is opened. It helps understanding the progress.')

    _parser.add_option("-d", '--debug',
        dest="debug",
        action="store_true",
        default=False,
        help="If set, changes behavior in some cases to be more verbose or to break earlier.")

    _parser.add_option('--lf', '--log-focus',
        dest="logger_focus",
        action="append",
        default = [],
        help=' '.join(["List names of loggers, which should be shown verbosely.",
                       "You can specify this option multiple times!",
                       "e.g. --lf=CORE --lf=SPLIT"]))

    _options, _args = _parser.parse_args()

    logger.info('cmdln options: %s' % _options)

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

      _cfgp.add_section('heuristic')
      _cfgp.set('heuristic', 'capacity', '20')

      _cfgp.add_section('core') # core configuration
      _cfgp.set('core', 'loglevel', '40') # default: no debug mode
      _cfgp.set('core', 'show_interval', '1.0')
      _cfgp.set('core', 'max_eval', '1000')
      _cfgp.set('core', 'discount', '0.95')
      _cfgp.set('core', 'smooth', 0.5)

      _cfgp.add_section('ui')
      _cfgp.set('ui', 'show', False)

      with open(_options.config_file, 'wb') as configfile:
        _cfgp.write(configfile)

    # 2/2: reading the config file
    _cfgp = ConfigParser()
    _cfgp.read(_options.config_file)

    # 3: override specific settings
    _cur_verb = _cfgp.getint('core', 'loglevel')
    if _options.verbosity:   _cfgp.set('core',    'loglevel', str(_cur_verb - 10*_options.verbosity))
    if _options.max_eval:    _cfgp.set('core',    'max_eval', str(_options.max_eval))
    if _options.smooth:      _cfgp.set('core',    'smooth',   str(_options.smooth))
    if _options.capacity:    _cfgp.set('heuristic', 'capacity', str(_options.capacity))
    if _options.ipy_profile: _cfgp.set('ipython', 'profile',  _options.ipy_profile)
    if _options.ui:          _cfgp.set('ui',      'show',    "True")

    ## some generic function
    def getself(section, key):
      return _cfgp.get(section, key)

    def all_cfgp(sep = '.'):
      ret = {}
      for s in _cfgp.sections():
        for k, v in _cfgp.items(s):
          ret['%s%s%s' % (s, sep, str(k))] = v
      return ret

    logger.info('config.ini: %s' % all_cfgp())
    self.environment = info()
    from panobbgo import __version__

    ## specific data
    self.loglevel        = _cfgp.getint  ('core', 'loglevel')
    self.debug           = _options.debug
    self.show_interval   = _cfgp.getfloat('core', 'show_interval')
    self.max_eval        = _cfgp.getint  ('core', 'max_eval')
    self.discount        = _cfgp.getfloat('core', 'discount')
    self.smooth          = _cfgp.getfloat('core', 'smooth')
    self.capacity        = _cfgp.getint  ('heuristic', 'capacity')
    self.ipy_profile     = _cfgp.get     ('ipython', 'profile')
    self.ui_show         = _cfgp.getboolean('ui', 'show')
    self.logger_focus    = _options.logger_focus
    self.ui_redraw_delay = 0.5
    self.version         = __version__
    self.git_head        = self.environment['git HEAD']

    logger.info('IPython profile: %s' % self.ipy_profile)
    logger.info("Environment: %s" % self.environment)

  def get_logger(self, name, loglevel = None):
    assert len(name) <= 5, 'Lenght of logger name > 5: "%s"' % name
    name = "%-5s" % name
    loglevel = loglevel or self.loglevel
    # logger focus
    lf = map(lambda _:_.upper(), ["%-5s" % _ for _ in self.logger_focus])
    if name in lf: loglevel = 0
    # cache
    key = '%s::%s' % (name, loglevel)
    if key in self._loggers:
      return self._loggers[key]
    from utils import create_logger
    l = create_logger(name, loglevel)
    self._loggers[key] = l
    return l

def get_config():
  global _config
  if _config != None:
    return _config
  _config = Config()
  return _config

