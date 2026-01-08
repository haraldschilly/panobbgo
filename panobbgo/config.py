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

_EPILOG = """\
Note: By default, the 'debug' mode is enabled automatically.
Disable it via the '-O' flag of the python interpreter, e.g. 'python -O run.py'.
Website: http://dev.harald.schil.ly/panobbgo/
Sources: https://github.com/haraldschilly/panobbgo
"""


class Config:
    # Singleton instance
    _instance = None
    # Class variable to track if config info has been logged
    _config_logged = False

    def __new__(cls, parse_args=False, testing_mode=False):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self, parse_args=False, testing_mode=False):
        """
        Initialize Config singleton.

        :param boolean parse_args:
        :param boolean testing_mode: if True, signals that it is run by the unittests
        """
        # Allow reinitialization for testing or if parameters changed
        current_parse_args = getattr(self, "parse_args", None)
        current_testing_mode = getattr(self, "testing_mode", None)

        if (
            current_parse_args != parse_args
            or current_testing_mode != testing_mode
            or not hasattr(self, "_initialized")
        ):
            import os

            self.parse_args = parse_args
            self.testing_mode = testing_mode
            self._appdata_dir = os.path.expanduser("~/.panobbgo")
            self.config_fn = os.path.join(self._appdata_dir, "config.ini")
            self.config_yaml = "config.yaml"  # YAML config in current directory
            self._loggers = {}
            self._create()
            self._initialized = True

    def _create(self):
        import os
        from .utils import info, create_logger

        logger = create_logger("CONFG")

        # create application data dir if necessary
        if not os.path.exists(self._appdata_dir):
            os.mkdir(self._appdata_dir)

        # 1: parsing command-line arguments
        from argparse import ArgumentParser

        descr = "Panobbgo - Parallel Noisy Black-Box Global Optimizer."

        epilog = _EPILOG

        parser = ArgumentParser(description=descr, epilog=epilog)

        parser.add_argument(
            "-c",
            "--config-file",
            dest="config_file",
            help="configuration file [default: %(default)s]",
            default=self.config_fn,
        )

        from panobbgo import __version__

        parser.add_argument("--version", action="version", version=__version__)

        parser.add_argument(
            "-p",
            "--profile",
            dest="ipy_profile",
            help="IPython profile for the ipcluster configuration",
        )

        parser.add_argument(
            "--max", dest="max_eval", help="maximum number of evaluations", type=int
        )

        parser.add_argument(
            "--smooth",
            dest="smooth",
            help="smoothing parameter for (additive or other) smoothing",
            type=float,
        )

        parser.add_argument(
            "--cap",
            dest="capacity",
            help="capacity for each queue in each heuristic",
            type=int,
        )

        parser.add_argument(
            "-v",
            action="count",
            dest="verbosity",
            help="verbosity level: -v, -vv, or -vvv",
        )

        parser.add_argument(
            "--ui",
            dest="ui",
            action="store_true",
            default=False,
            help="If specified, the GTK+/matplotlib based UI is opened. It helps understanding the progress.",
        )

        parser.add_argument(
            "--lf",
            "--log-focus",
            dest="logger_focus",
            action="append",
            default=[],
            help=" ".join(
                [
                    "List names of loggers, which should be shown verbosely.",
                    "You can specify this option multiple times!",
                    "e.g. --lf=CORE --lf=SPLIT",
                ]
            ),
        )

        if self.parse_args:
            args = parser.parse_args()
            logger.info("cmdln options: %s" % args)
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

            cfgp.add_section("db")  # database config
            # cfgp.set('db', 'port', '37010')
            # cfgp.set('db', 'host', 'localhost')

            cfgp.add_section("ipython")
            cfgp.set("ipython", "profile", "default")

            cfgp.add_section("heuristic")
            cfgp.set("heuristic", "capacity", "20")

            cfgp.add_section("core")  # core configuration
            cfgp.set("core", "loglevel", "40")  # default: no debug mode
            cfgp.set("core", "show_interval", "1.0")
            cfgp.set("core", "max_eval", "1000")
            cfgp.set("core", "discount", "0.95")
            cfgp.set("core", "smooth", "0.5")

            cfgp.add_section("ui")
            cfgp.set("ui", "show", "False")

            with open(self.config_fn, "w") as configfile:
                cfgp.write(configfile)

        # 2/2: reading the config file
        cfgp = ConfigParser()
        cfgp.read(self.config_fn)

        # 2/3: reading YAML config if it exists (takes precedence)
        import yaml

        self.yaml_config = {}
        if os.path.exists(self.config_yaml):
            with open(self.config_yaml, "r") as f:
                self.yaml_config = yaml.safe_load(f) or {}
            if not Config._config_logged:
                logger.info("config.yaml loaded from: %s" % self.config_yaml)

        # 3: override specific settings
        _cur_verb = cfgp.getint("core", "loglevel")
        if args is not None:
            if args.verbosity:
                cfgp.set("core", "loglevel", str(_cur_verb - 10 * args.verbosity))
            if args.max_eval:
                cfgp.set("core", "max_eval", str(args.max_eval))
            if args.smooth:
                cfgp.set("core", "smooth", str(args.smooth))
            if args.capacity:
                cfgp.set("heuristic", "capacity", str(args.capacity))
            if args.ipy_profile:
                cfgp.set("ipython", "profile", args.ipy_profile)
            if args.ui:
                cfgp.set("ui", "show", "True")

        # some generic function
        def getself(section, key):
            return cfgp.get(section, key)

        def allcfgp(sep="."):
            ret = {}
            for s in cfgp.sections():
                for k, v in cfgp.items(s):
                    ret["%s%s%s" % (s, sep, str(k))] = v
            return ret

        if not Config._config_logged:
            logger.info("config.ini: %s" % allcfgp())
        self.environment = info()
        from panobbgo import __version__

        # Helper function to get config value (YAML takes precedence over INI)
        def get_config(yaml_path, ini_section, ini_key, default=None, type_cast=str):
            """Get config value from YAML first, then INI, then default"""
            # Check YAML first
            if yaml_path and self.yaml_config:
                val = self.yaml_config
                for part in yaml_path.split("."):
                    if isinstance(val, dict) and part in val:
                        val = val[part]
                    else:
                        val = None
                        break
                if val is not None:
                    return type_cast(val) if type_cast != bool else bool(val)

            # Fall back to INI
            if ini_section and ini_key and cfgp.has_option(ini_section, ini_key):
                if type_cast == int:
                    return cfgp.getint(ini_section, ini_key)
                elif type_cast == float:
                    return cfgp.getfloat(ini_section, ini_key)
                elif type_cast == bool:
                    return cfgp.getboolean(ini_section, ini_key)
                else:
                    return cfgp.get(ini_section, ini_key)

            return default

        # specific data
        self.loglevel = get_config("core.loglevel", "core", "loglevel", 40, int)
        self.show_interval = get_config(
            "core.show_interval", "core", "show_interval", 1.0, float
        )
        self.max_eval = get_config("core.max_eval", "core", "max_eval", 1000, int)
        self.discount = get_config("core.discount", "core", "discount", 0.95, float)
        self.smooth = get_config("core.smooth", "core", "smooth", 0.5, float)
        self.capacity = get_config(
            "heuristic.capacity", "heuristic", "capacity", 20, int
        )
        self.ui_show = get_config("ui.show", "ui", "show", False, bool)
        self.ui_redraw_delay = get_config(
            "ui.redraw_delay", "ui", "redraw_delay", 0.5, float
        )

        # Evaluation method configuration (YAML only)
        self.evaluation_method = get_config(
            "evaluation.method", None, None, "direct", str
        )

        # Dask cluster configuration (YAML only, only used when evaluation_method is 'dask')
        self.dask_cluster_type = get_config(
            "dask.cluster_type", None, None, "local", str
        )
        self.dask_n_workers = get_config("dask.local.n_workers", None, None, 2, int)
        self.dask_threads_per_worker = get_config(
            "dask.local.threads_per_worker", None, None, 1, int
        )
        self.dask_memory_limit = get_config(
            "dask.local.memory_limit", None, None, "2GB", str
        )
        self.dask_dashboard_address = get_config(
            "dask.local.dashboard_address", None, None, ":8787", str
        )
        self.dask_scheduler_address = get_config(
            "dask.remote.scheduler_address", None, None, "tcp://localhost:8786", str
        )

        # Legacy IPython support (kept for backward compatibility)
        self.ipy_profile = get_config(None, "ipython", "profile", "default", str)

        self.logger_focus = [] if args is None else args.logger_focus
        self.version = __version__
        self.git_head = self.environment["git HEAD"]

        # Only log configuration info once per session to avoid spam
        if not Config._config_logged:
            logger.info("Evaluation method: %s" % self.evaluation_method)
            if self.evaluation_method == "dask":
                logger.info("Dask cluster type: %s" % self.dask_cluster_type)
            logger.info("Environment: %s" % self.environment)
            Config._config_logged = True

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
        key = "%s::%s" % (name, loglevel)
        if key in self._loggers:
            return self._loggers[key]
        from .utils import create_logger

        l = create_logger(name, loglevel)
        self._loggers[key] = l
        return l
