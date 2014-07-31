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
from __future__ import absolute_import
from __future__ import unicode_literals
from future.builtins import str
from future import standard_library
standard_library.install_hooks()
from future.builtins import object

_config = None

_EPILOG = """\
Note: By default, the 'debug' mode is enabled automatically.
Disable it via the '-O' flag of the python interpreter, e.g. 'python -O run.py'.
Website: http://dev.harald.schil.ly/panobbgo/
Sources: https://github.com/haraldschilly/panobbgo
"""


class Config(object):

    def __init__(self, parse_args=False, testing_mode=False):
        """

        :param boolean parse_args:
        :param boolean testing_mode: if True, signals that it is run by the unittests
        """
        import os
        self.parse_args = parse_args
        self.testing_mode = testing_mode
        self._appdata_dir = os.path.expanduser("~/.panobbgo")
        self.config_fn = os.path.join(self._appdata_dir, 'config.ini')
        self._loggers = {}
        self._create()

    def _create(self):
        from .utils import info, create_logger
        logger = create_logger("CONFG")

        # create application data dir if necessary
        import os
        if not os.path.exists(self._appdata_dir):
            os.mkdir(self._appdata_dir)

        # 1: parsing command-line arguments
        from argparse import ArgumentParser

        descr = 'Panobbgo - Parallel Noisy Black-Box Global Optimizer.'

        epilog = _EPILOG

        parser = ArgumentParser(description=descr, epilog=epilog)

        parser.add_argument('-c', '--config-file',
                            dest="config_file",
                            help='configuration file [default: %(default)s]',
                            default=self.config_fn)

        from panobbgo import __version__
        parser.add_argument('--version', action='version', version=__version__)

        parser.add_argument('-p', '--profile',
                            dest='ipy_profile',
                            help='IPython profile for the ipcluster configuration')

        parser.add_argument('--max',
                            dest='max_eval',
                            help="maximum number of evaluations",
                            type=int)

        parser.add_argument('--smooth',
                            dest='smooth',
                            help="smoothing parameter for (additive or other) smoothing",
                            type=float)

        parser.add_argument('--cap',
                            dest='capacity',
                            help="capacity for each queue in each heuristic",
                            type=int)

        parser.add_argument("-v",
                            action="count",
                            dest="verbosity",
                            help="verbosity level: -v, -vv, or -vvv")

        parser.add_argument('--ui',
                            dest='ui',
                            action='store_true',
                            default=False,
                            help='If specified, the GTK+/matplotlib based UI is opened. It helps understanding the progress.')

        parser.add_argument('--lf', '--log-focus',
                            dest="logger_focus",
                            action="append",
                            default=[],
                            help=' '.join(["List names of loggers, which should be shown verbosely.",
                                           "You can specify this option multiple times!",
                                           "e.g. --lf=CORE --lf=SPLIT"]))

        if self.parse_args:
            args = parser.parse_args()
            logger.info('cmdln options: %s' % args)
            self.config_fn = args.config_file
        else:
            # logger.info("Parsing command-line arguments is disabled.")
            args = None

        import os
        from configparser import ConfigParser

        # 2/1: does config file exist?
        if not os.path.exists(self.config_fn):
            cfgp = ConfigParser()
            # create them in the reverse order

            cfgp.add_section('db')  # database config
            # cfgp.set('db', 'port', '37010')
            # cfgp.set('db', 'host', 'localhost')

            cfgp.add_section('ipython')
            cfgp.set('ipython', 'profile', 'default')

            cfgp.add_section('heuristic')
            cfgp.set('heuristic', 'capacity', '20')

            cfgp.add_section('core')  # core configuration
            cfgp.set('core', 'loglevel', '40')  # default: no debug mode
            cfgp.set('core', 'show_interval', '1.0')
            cfgp.set('core', 'max_eval', '1000')
            cfgp.set('core', 'discount', '0.95')
            cfgp.set('core', 'smooth', 0.5)

            cfgp.add_section('ui')
            cfgp.set('ui', 'show', False)

            with open(self.config_fn, 'wb') as configfile:
                cfgp.write(configfile)

        # 2/2: reading the config file
        cfgp = ConfigParser()
        cfgp.read(self.config_fn)

        # 3: override specific settings
        _cur_verb = cfgp.getint('core', 'loglevel')
        if args is not None:
            if args.verbosity:
                cfgp.set(
                    'core', 'loglevel', str(_cur_verb - 10 * args.verbosity))
            if args.max_eval:
                cfgp.set('core', 'max_eval', str(args.max_eval))
            if args.smooth:
                cfgp.set('core', 'smooth', str(args.smooth))
            if args.capacity:
                cfgp.set('heuristic', 'capacity', str(args.capacity))
            if args.ipy_profile:
                cfgp.set('ipython', 'profile', args.ipy_profile)
            if args.ui:
                cfgp.set('ui', 'show', "True")

        # some generic function
        def getself(section, key):
            return cfgp.get(section, key)

        def allcfgp(sep='.'):
            ret = {}
            for s in cfgp.sections():
                for k, v in cfgp.items(s):
                    ret['%s%s%s' % (s, sep, str(k))] = v
            return ret

        logger.info('config.ini: %s' % allcfgp())
        self.environment = info()
        from panobbgo import __version__

        # specific data
        self.loglevel = cfgp.getint('core', 'loglevel')
        self.show_interval = cfgp.getfloat('core', 'show_interval')
        self.max_eval = cfgp.getint('core', 'max_eval')
        self.discount = cfgp.getfloat('core', 'discount')
        self.smooth = cfgp.getfloat('core', 'smooth')
        self.capacity = cfgp.getint('heuristic', 'capacity')
        self.ipy_profile = cfgp.get('ipython', 'profile')
        self.ui_show = cfgp.getboolean('ui', 'show')
        self.logger_focus = [] if args is None else args.logger_focus
        self.ui_redraw_delay = 0.5
        self.version = __version__
        self.git_head = self.environment['git HEAD']

        logger.info('IPython profile: %s' % self.ipy_profile)
        logger.info("Environment: %s" % self.environment)

    @property
    def debug(self):
        return __debug__

    def get_logger(self, name, loglevel=None):
        assert len(name) <= 5, 'Length of logger name > 5: "%s"' % name
        name = "%-5s" % name
        loglevel = loglevel or self.loglevel
        # logger focus
        lf = [_.upper() for _ in ["%-5s" % _ for _ in self.logger_focus]]
        if name in lf:
            loglevel = 0
        # cache
        key = '%s::%s' % (name, loglevel)
        if key in self._loggers:
            return self._loggers[key]
        from .utils import create_logger
        l = create_logger(name, loglevel)
        self._loggers[key] = l
        return l
