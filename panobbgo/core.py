# -*- coding: utf8 -*-
# Copyright 2012 -- 2013 Harald Schilly <harald.schilly@univie.ac.at>
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
Core
====

This is the core part. It contains the essential components
and base-classes for the modules:

- :class:`.Results`: Database of all results, with some rudimentary queries and statistics.
- :class:`.EventBus`: This is the backbone for communicating between the strategy,
  the heuristics and the analyzers.
- "abstract" base-classes for the modules

  - :mod:`.heuristics`
  - :mod:`.analyzers`.

- and most importantly, the :class:`.StrategyBase` which holds everything together and
  subclasses in :mod:`.strategies` implement the actual strategies.

.. inheritance-diagram:: panobbgo.core

.. codeauthor:: Harald Schilly <harald.schilly@univie.ac.at>
"""

from .config import Config
from panobbgo.lib.lib import Result, Point
from panobbgo.lib.constraints import (
    DefaultConstraintHandler,
    PenaltyConstraintHandler,
    DynamicPenaltyConstraintHandler,
)
from .logging import PanobbgoLogger
import time as time_module
import numpy as np


class Results:
    """
    A very simple database of results with a notification for new results.
    The new results are fed directly by the :class:`.StrategyBase`, outside of the
    :class:`.EventBus`.

    .. Note::

      Later on, maybe this will be a cool actual database which allows to
      persistently store past evaluations for a given problem.
      This would allow resuming and further a-posteriory analysis.
      In the meantime, this is a pandas DataFrame.
    """

    def __init__(self, strategy):
        self.logger = strategy.config.get_logger("RSLTS")
        self.strategy = strategy
        self.eventbus = strategy.eventbus
        self.problem = strategy.problem
        self.results = None
        self._last_nb = 0  # for logging

    def add_results(self, new_results):
        """
        Add one single or a list of new @Result objects.
        Then, publish a ``new_result`` event.
        """
        from pandas import DataFrame, MultiIndex, concat

        if self.results is None:
            if len(new_results) == 0:
                return
            r = new_results[0]
            midx_x = [("x", _) for _ in range(len(r.x))]
            len_cv_vec = 0 if r.cv_vec is None else len(r.cv_vec)
            midx_cv = [("cv", _) for _ in range(len_cv_vec)]
            midx = MultiIndex.from_tuples(
                midx_x + [("fx", 0)] + midx_cv + [("cv", 0), ("who", 0), ("error", 0)]
            )
            self.results = DataFrame(columns=midx)

        assert all([isinstance(_, Result) for _ in new_results])
        # notification for all received results at once
        self.eventbus.publish("new_results", results=new_results)

        # Report progress for each result
        for result in new_results:
            self._report_evaluation_progress(result)

        new_rows = []
        for r in new_results:
            new_rows.append(
                np.r_[
                    r.x,
                    r.fx,
                    [] if r.cv_vec is None else r.cv_vec,
                    [r.cv, r.who, r.error],
                ]
            )
        results_new = DataFrame(new_rows, columns=self.results.columns)
        self.results = concat([self.results, results_new], ignore_index=True)

        if len(self.results) / 100 > self._last_nb / 100:
            self.info()
            self._last_nb = len(self.results)

    def info(self):
        self.logger.info("%d results in DB" % len(self))
        if self.results is not None:
            self.logger.debug("Dataframe Results:\n%s" % self.results.tail(3))

    def __iadd__(self, results):
        self.add_results(results)
        return self

    def __len__(self):
        return len(self.results) if self.results is not None else 0

    def _report_evaluation_progress(self, result: Result):
        """
        Report progress for a single evaluation result.

        Args:
            result: The evaluation result to report
        """
        from .logging.progress import ProgressContext

        # Get strategy's logger for progress reporting
        if not hasattr(self.strategy, 'panobbgo_logger'):
            return  # No logger configured, skip progress reporting

        progress_reporter = self.strategy.panobbgo_logger.progress_reporter

        # Determine progress context
        context = ProgressContext()

        # Only analyze if we have valid fx
        if result.fx is not None:
            try:
                # Check if this is a new global best
                if self.results is not None and len(self.results) > 0:
                    current_best_fx = float(self.results['fx', 0].min())
                    if result.fx < current_best_fx:
                        context.is_global_best = True

                # Check for significant improvement (top 10% of results)
                if self.results is not None and len(self.results) > 10:
                    sorted_fx = sorted([float(x) for x in self.results['fx', 0].dropna()])
                    threshold_idx = int(len(sorted_fx) * 0.1)  # top 10%
                    if threshold_idx > 0:
                        threshold = sorted_fx[threshold_idx]
                        if result.fx < threshold:
                            context.is_significant_improvement = True

                # Check for general improvement over previous results
                if self.results is not None and len(self.results) > 1:
                    prev_best = float(self.results['fx', 0].iloc[:-1].min())  # exclude current result
                    if result.fx < prev_best:
                        context.is_improvement = True
            except (ValueError, TypeError, KeyError):
                # If we can't determine improvement status, skip it
                pass

        # Check for failure (fx is None or has error)
        if result.fx is None or (result.error and result.error > 0):
            context.evaluation_failed = True

        # Check for warnings (constraint violations, etc.)
        if result.cv is not None and result.cv > 0:
            context.has_warnings = True

        # Report the evaluation
        progress_reporter.report_evaluation(result, context)


class Module:
    """
    "Abstract" parent class for various panobbgo modules, e.g.
    :class:`.Heuristic` and :class:`.Analyzer`.
    """

    def __init__(self, strategy, name=None):
        """
        :param StrategyBase strategy:
        :param str name:
        @type strategy: StrategyBase
        """
        name = name if name else self.__class__.__name__
        self._strategy = strategy
        self.config = strategy.config
        self._name = name
        self._threads = []
        # implicit dependency check (only class references)
        self._depends_on = []

    @property
    def name(self):
        """
        The module's name.

        .. Note::

          It should be unique, which is important for
          parameterized heuristics or analyzers!
        """
        return self._name

    @property
    def strategy(self):
        return self._strategy

    @property
    def ui(self):
        return self._strategy.ui

    @property
    def eventbus(self):
        return self._strategy.eventbus

    @property
    def problem(self):
        return self._strategy.problem

    @property
    def results(self):
        return self._strategy.results

    def check_dependencies(self, analyzers, heuristics):
        """
        This method is called by the core initialization to assess,
        if the dependencies for the given module are met.

        By default, it returns true. Return false if there is
        a problem.

        The arguments are the list of pre-initialized analyzers
        and heuristics.
        """
        return True

    def __start__(self):
        """
        This method should be overwritten by the respective subclass.
        It is called in the 2nd initialization phase, inside :meth:`._init_module`.
        Now, the strategy and all its components (e.g. :class:`panobbgo.lib.lib.Problem`, ...)
        are available.
        """
        pass

    def __stop__(self):
        """
        Called right at the end after the strategy has finished.
        """
        for t in self._threads:
            if t.is_alive():
                try:
                    t._Thread__stop()
                except:
                    pass
            t.join()

    def _init_plot(self):
        """
        This plot initializer is called right after the :meth:`._init` method.
        It could be used to tell the (optionally enabled) :module:`user interface <.ui>` that
        this module wants to have a tab for displaying and visualizing some data.

        It has to return a tuple consisting of a string as the label of the tab,
        and a gtk container (e.g. :class:`gtk.VBox`)

        To trigger a redraw after an update, call the ``.draw_idle()`` method
        of the :class:`~matplotlib.backends.backend_gtkagg.FigureCnavasGTKAgg`.
        """
        return None, None

    def __repr__(self):
        return "Module %s" % self.name


class StopHeuristic(Exception):
    """
    Indicates the heuristic has finished and should be ignored/removed.
    """

    def __init__(self, msg="stopped"):
        """
        Args:

        - ``msg``: a custom message, will be visible in the log. (default: "stopped")
        """
        Exception.__init__(self, msg)


class Heuristic(Module):
    """
    This is the "abstract" parent class for all types of point generating classes,
    which we call collectively ":mod:`Heuristics <.heuristics>`".

    Such a heuristic is capable of the following:

    #. They can be parameterized by passing in optional arguments in the constructor.
       This should be reflected in the :attr:`~.Module.name`!
    #. The :class:`.EventBus` spawns a thread for each ``on_*`` method
       and calls them when a corresponding :class:`.Event` occurs.
    #. Of course, they are capable of storing their state in the instance.
       This is also the way of how information is shared between those threads.
    #. The `main purpose` of a heuristic is to emit new search points
       by calling either :meth:`.emit` or returning a list of points.
       The datatype must be :class:`numpy.ndarray` of
       `floats <http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html>`_.
    #. Additionally, the can get hold of other heuristics or anayzers via the strategy instance.
    #. The :class:`.EventBus` inside this strategy instance allows them to publish their
       own events, too. This can be used to signal related heuristics something
       or to queue up tasks for itself.
    """

    def __init__(self, strategy, name=None, cap=None):
        Module.__init__(self, strategy, name)
        self.config = strategy.config
        self.logger = self.config.get_logger("HEUR")
        self.cap = cap if cap is not None else self.config.capacity
        self._stopped = False
        from queue import Queue

        self._output = Queue(self.cap)

        # statistics; performance
        self.performance = 0.0

    def clear_output(self):
        q = self._output
        with q.not_full:
            # with q.mutex:
            # del self._output.queue[:]  # LifoQueue
            q.queue.clear()  # Queue
            q.not_full.notify()  # to wakeup "put()"

    def emit(self, points):
        """
        This is used to send out new search points for evaluation.
        Args:

        - ``points``: Either a :class:`numpy.ndarray` of ``float64`` or preferrably a list of them.
        """
        try:
            if points is None:
                raise StopHeuristic()
            if not isinstance(points, (list, tuple)):
                points = [points]
            for point in points:
                if not isinstance(point, np.ndarray):
                    raise Exception("point is not a numpy ndarray")
                x = self.problem.project(point)
                point = Point(x, self.name)
                self._output.put(point)
        except StopHeuristic:
            self._stopped = True
            self.logger.info("'%s' heuristic stopped." % self.name)

    def get_points(self, limit=None):
        """
        this drains the output Queue until ``limit``
        elements are removed or the Queue is empty.
        For each actually emitted point,
        the performance value is discounted (i.e. "punishment" or "energy
        consumption")
        """
        from queue import Empty

        new_points = []
        try:
            while limit is None or len(new_points) < limit:
                new_points.append(self._output.get(block=False))
        except Empty:
            pass
        return new_points

    @property
    def active(self):
        """
        This is queried by the strategy to determine, if it should still consider it.
        This is the case, iff there is still something in its output queue
        or if there is a chance that there will be something in the future (at least
        one thread is running).
        """
        t = any(t.is_alive() for t in self._threads)
        q = self._output.qsize() > 0
        return t or q


class HeuristicSubprocess(Heuristic):
    r"""
    This Heuristic is a subclass of :class:`.Heuristic`, which is additionally starting
    a subprocess, which communicates with the main thread via a pipe in a blocking
    communication scheme.
    """

    def __init__(self, strategy, name=None, cap=None):
        Heuristic.__init__(self, strategy, name=name, cap=cap)

        from multiprocessing import Process, Pipe

        # a pipe has two ends, parent and child.
        self.pipe, self.pipe_child = Pipe()
        self.__subprocess = Process(
            target=self.subprocess,
            args=(self.pipe_child,),
            name="%s-subprocess" % (self.name),
        )
        self.__subprocess.daemon = True
        self.__subprocess.start()

    @staticmethod
    def subprocess(pipe):
        """
        overwrite this pipe.recv() & pipe.send() loop and
        compute something in between.
        """
        while True:
            payload = pipe.recv()
            pipe.send("subprocess received: %s" % payload)


#
# Analyzer
#
class Analyzer(Module):
    """
    Abstract parent class for all types of analyzers.
    """

    def __init__(self, strategy, name=None):
        Module.__init__(self, strategy, name)


#
# EventBus
#


class Event:
    """
    This class holds the data for one single :class:`~.EventBus` event.
    """

    def __init__(self, **kwargs):
        self._when = time_module.time()
        self._kwargs = kwargs
        for k, v in list(kwargs.items()):
            setattr(self, k, v)

    def __repr__(self):
        return "Event[%s]" % self._kwargs


class EventBus:
    """
    This event bus is used to publish and send events.
    E.g. it is used to send information like "new best point"
    to all subscribing heuristics.
    """

    # pattern for a valid key
    import re

    _re_key = re.compile(r"^[a-z_]+$")

    def __init__(self, config):
        self._subs = {}
        self.config = config
        self.logger = config.get_logger("EVBUS")

    @property
    def keys(self):
        """
        List of all keys where you can send an :class:`Event` to.
        """
        return list(self._subs.keys())

    def register(self, target):
        """
        Registers a given ``target`` for this EventBus instance.
        It needs to have suitable ``on_<key>`` methods.
        For each of them, a :class:`~threading.Thread` is spawn as a daemon.

        :param Module target:
        """
        from queue import Empty, Queue  # LifoQueue
        from threading import Thread

        # important: this decouples the dispatcher's thread from the actual
        # target
        def run(key, target, drain=False):
            if drain:  # not using draining for now, doesn't make much sense
                isfirst = True
                while True:
                    # draining the queue... otherwise it might get really huge
                    # it's up to the heuristics to only work with the most
                    # important points
                    events = []
                    terminate = False
                    try:
                        while True:
                            event = target.eventbus_events[key].get(block=isfirst)
                            terminate |= event.terminate
                            events.append(event)
                            isfirst = False
                    except Empty:
                        isfirst = True

                    try:
                        new_points = getattr(target, "on_%s" % key)(events)
                        # heuristics might call self.emit and/or return a list
                        if new_points is not None:
                            target.emit(new_points)
                        if terminate:
                            raise StopHeuristic("%s terminated" % target.name)
                    except StopHeuristic as e:
                        self.logger.debug(
                            "'%s/on_%s' %s -> unsubscribing."
                            % (target.name, key, e.message)
                        )
                        self.unsubscribe(key, target)
                        return

            else:  # not draining (default)
                while True:
                    try:
                        event = target.eventbus_events[key].get(block=True)
                        assert isinstance(event, Event)
                        try:
                            new_points = getattr(target, "on_%s" % key)(**event._kwargs)
                            # heuristics might call self.emit and/or return a
                            # list
                            if new_points is not None:
                                target.emit(new_points)
                            if event.terminate:
                                raise StopHeuristic("%s terminated" % target.name)
                        except StopHeuristic as e:
                            self.logger.debug(
                                "'%s/on_%s' %s -> unsubscribing."
                                % (target.name, key, str(e))
                            )
                            self.unsubscribe(key, target)
                            return
                    except Exception as e:
                        # usually, they only happen during shutdown
                        if self.config.debug:
                            # sys.exc_info() -> re-create original exception
                            # (otherwise we don't know the actual cause!)
                            import sys

                            ex = sys.exc_info()
                            raise (ex[1], None, ex[2])
                        else:  # just issue a critical warning
                            self.logger.critical(
                                "Exception: %s in %s: %s" % (key, target, e)
                            )
                        return

        target.eventbus_events = {}
        # bind all 'on_<key>' methods to events in the eventbus
        import inspect

        for name, _ in inspect.getmembers(target, predicate=inspect.ismethod):
            if not name.startswith("on_"):
                continue
            key = self._check_key(name[3:])
            target.eventbus_events[key] = Queue()
            t = Thread(
                target=run,
                args=(
                    key,
                    target,
                ),
                name="EventBus::%s/%s" % (target.name, key),
            )
            t.daemon = True
            t.start()
            target._threads.append(t)
            # thread running, now subscribe to events
            self.subscribe(key, target)
            # logger.debug("%s subscribed and running." % t.name)

    @staticmethod
    def _check_key(key):
        if not EventBus._re_key.match(key):
            raise ValueError('"%s" key not allowed' % key)
        return key

    def subscribe(self, key, target):
        """
        Called by :meth:`.register`.

        .. Note:: counterpart is :func:`unsubscribe`.
        """
        self._check_key(key)
        if key not in self._subs:
            self._subs[key] = []

        assert target not in self._subs[key]
        self._subs[key].append(target)

    def unsubscribe(self, key, target):
        """
        Args:

        - if ``key`` is ``None``, the target is removed from all keys.

        """
        if key is None:
            for k, v in list(self._subs.items()):
                for t in v:
                    if t is target:
                        self.unsubscribe(k, t)
            return

        self._check_key(key)
        if key not in self._subs:
            self.logger.critical("cannot unsubscribe unknown key '%s'" % key)
            return

        if target in self._subs[key]:
            self._subs[key].remove(target)

    def publish(self, key, event=None, terminate=False, **kwargs):
        """
        Publishes a new :class:`.Event` to all subscribers,
        who listen to the given ``key``.
        It is either possible to send an existing event or to create an event
        object on the fly with the given ``**kwargs``.

        Args:

        - ``event``: if set, this given :class:`.Event` is sent (and not a new one created).
        - ``terminate``: if True, the associated thread will end.
                         (use it for ``on_start`` and similar).
        - ``**kwargs``: any additional keyword arguments are stored inside the Event
                        if ``event`` is ``None``.
        """
        if key not in self._subs:
            if self.config.debug:
                self.logger.warning("key '%s' unknown." % key)
            return

        for target in self._subs[key]:
            event = Event(**kwargs) if event is None else event
            event.terminate = terminate
            # logger.info("EventBus: publishing %s -> %s" % (key, event))
            target.eventbus_events[key].put(event)


class StrategyBase:
    """
    This abstract BaseStrategy is the parent class of all Strategies.

    Use it this way:

    #. Subclass it, write your optional initializer, *afterwards* call the initializer
       of this class (it will start its the main loop).

    #. Overwrite the :meth:`.execute`, which returns a list of new search points
       (by requesting them from the :mod:`~panobbgo.heuristics` via the
       :meth:`~panobbgo.core.Heuristic.get_points` method) and might
       also emit :class:`Events <panobbgo.core.Event>`.

    This ``execute`` method will be called repeatedly as long as there are less than the
    given maximum number of search points evaluated.
    """

    # constant reference id for sending the evaluation code to workers
    PROBLEM_KEY = "problem"

    def __init__(self, problem, parse_args=False):
        """


        @type problem: panobbgo.lib.lib.Problem
        @param problem:
        @param parse_args:
        """
        self._name = name = self.__class__.__name__
        self.config = config = Config(parse_args)
        self.logger = logger = config.get_logger("STRAT")
        self.slogger = config.get_logger("STATS")
        logger.info("Init of '%s'" % (name))
        logger.info("%s" % problem)

        # aux configuration
        import pandas as pd

        # determine width based on console info
        pd.set_option("display.width", None)
        pd.set_option("display.precision", 2)  # default 7

        # statistics
        self.show_last = 0  # for printing the info line in _add_tasks()
        self.time_start = time_module.time()
        self.tasks_walltimes = {}

        # task accounting (tasks != points !!!)
        self.jobs_per_client = 1  # number of tasks per client in 'chunksize'
        self.pending = {}  # dict mapping future id to future object
        self.new_finished = []
        self.finished = []

        # init & start everything
        self._setup_cluster(problem)
        self._threads = []
        self._hs = []
        import collections

        self._heuristics = collections.OrderedDict()
        self._analyzers = collections.OrderedDict()
        self.problem = problem

        # Configure Constraint Handler
        rho = float(config.rho) if hasattr(config, "rho") else 100.0
        exponent = float(config.constraint_exponent) if hasattr(config, "constraint_exponent") else 1.0
        rate = float(config.dynamic_penalty_rate) if hasattr(config, "dynamic_penalty_rate") else 0.01
        handler_name = getattr(config, "constraint_handler", "DefaultConstraintHandler")

        if handler_name == "PenaltyConstraintHandler":
            self.constraint_handler = PenaltyConstraintHandler(
                strategy=self, rho=rho, exponent=exponent
            )
        elif handler_name == "DynamicPenaltyConstraintHandler":
            self.constraint_handler = DynamicPenaltyConstraintHandler(
                strategy=self, rho_start=rho, rate=rate, exponent=exponent
            )
        else:
            self.constraint_handler = DefaultConstraintHandler(
                strategy=self, rho=rho
            )

        self.eventbus = EventBus(config)
        self.results = Results(self)

        # Initialize new logging system
        self.panobbgo_logger = PanobbgoLogger(config.logging if hasattr(config, 'logging') else {})

        # UI
        if config.ui_show:
            from .ui import UI

            self.ui = UI()
            self.ui._init_module(self)
            self.ui.show()

    def add(self, Heur, **kwargs):
        self.logger.debug("init: %s %s" % (Heur.__name__, kwargs))
        self._hs.append(Heur(self, **kwargs))

    def start(self):
        # heuristics
        for h in sorted(self._hs, key=lambda h: h.name):
            self.add_heuristic(h)

        # analyzers
        from .analyzers import Best, Grid, Splitter

        new_analyzers = [Best(self), Grid(self), Splitter(self)]
        for a in new_analyzers:
            self.add_analyzer(a)

        self.check_dependencies()

        self.logger.debug("EventBus keys: %s" % self.eventbus.keys)

        try:
            import threading

            if isinstance(self, threading.Thread):
                raise Exception("change run() to start()")
            self._run()
        except KeyboardInterrupt:
            self.logger.critical("KeyboardInterrupt received, e.g. via Ctrl-C")
            self._cleanup()

    @property
    def heuristics(self):
        return [h for h in list(self._heuristics.values()) if h.active]

    @property
    def analyzers(self):
        return list(self._analyzers.values())

    def heuristic(self, who):
        return self._heuristics[who]

    def analyzer(self, who):
        return self._analyzers[who]

    def add_heuristic(self, h):
        """

        :param Heuristic h:
        """
        name = h.name
        assert name not in self._heuristics, (
            "Names of heuristics need to be unique. '%s' is already used." % name
        )
        self._heuristics[name] = h
        self.init_module(h)

    def add_analyzer(self, a):
        """

        :param Analyzer a:
        """
        name = a.name
        assert name not in self._analyzers, (
            "Names of analyzers need to be unique. '%s' is already used." % name
        )
        self._analyzers[name] = a
        self.init_module(a)

    def check_dependencies(self):
        """
        This method is called in self.start() (and only there)
        for checking all the dependencies of all modules.
        """
        heuristics = list(self._heuristics.values())
        analyzers = list(self._analyzers.values())
        all_mods = [m.__class__ for m in heuristics]
        all_mods.extend([m.__class__ for m in analyzers])
        all_mods = set(all_mods)
        for module in analyzers + heuristics:
            # explicit
            if not module.check_dependencies(analyzers, heuristics):
                raise Exception("%s does not satisfy dependencies. #1" % module)
            # implicit (just list of respective classes)
            for mod_class in module._depends_on:
                if mod_class not in all_mods:
                    raise Exception(
                        "%s depends on %s, but missing." % (module, mod_class)
                    )

    def init_module(self, module):
        """
        :class:`.StrategyBase` calls this method.

        :param Module module:
        """
        module.__start__()
        if self.config.ui_show:
            plt = module._init_plot()
            if not isinstance(plt, list):
                plt = [plt]
            [self.ui.add_notebook_page(*p) for p in plt]
        # only after _init_ it is ready to receive events
        module.eventbus.register(module)

    def _setup_cluster(self, problem):
        """
        Set up evaluation infrastructure based on configuration.
        Supports both direct subprocess evaluation and Dask distributed evaluation.
        """
        if self.config.evaluation_method == "dask":
            self._setup_dask_cluster(problem)
        elif self.config.evaluation_method == "direct":
            self._setup_direct_evaluation(problem)
        else:
            raise ValueError(
                f"Unknown evaluation method: {self.config.evaluation_method}"
            )

    def _setup_dask_cluster(self, problem):
        """
        Set up a Dask cluster based on configuration.
        Supports both local and remote clusters.
        """
        from dask.distributed import Client, LocalCluster

        if self.config.dask_cluster_type == "local":
            # Create a local cluster
            self.logger.info(
                "Setting up local Dask cluster with %d workers"
                % self.config.dask_n_workers
            )
            cluster = LocalCluster(
                n_workers=self.config.dask_n_workers,
                threads_per_worker=self.config.dask_threads_per_worker,
                memory_limit=self.config.dask_memory_limit,
                dashboard_address=self.config.dask_dashboard_address,
                silence_logs=False,
            )
            self._client = Client(cluster)
        else:
            # Connect to remote cluster
            self.logger.info(
                "Connecting to remote Dask cluster at %s"
                % self.config.dask_scheduler_address
            )
            self._client = Client(self.config.dask_scheduler_address)

        # Scatter the problem to all workers (similar to IPython Reference)
        self._problem_future = self._client.scatter(problem, broadcast=True)

        self.logger.info(
            "Dask cluster ready with %d workers"
            % len(self._client.scheduler_info()["workers"])
        )
        if self.config.dask_cluster_type == "local":
            self.logger.info(
                "Dashboard available at: http://localhost%s"
                % self.config.dask_dashboard_address
            )

    def _setup_direct_evaluation(self, problem):
        """
        Set up direct subprocess evaluation.
        """
        import multiprocessing
        import pickle
        import os
        import tempfile

        # Store problem for subprocess evaluation
        self._problem = problem

        # Determine number of processes (default to number of CPU cores)
        self._n_processes = (
            self.config.dask_n_workers
            if hasattr(self.config, "dask_n_workers")
            else multiprocessing.cpu_count()
        )
        self.logger.info(
            "Setting up direct evaluation with %d subprocesses" % self._n_processes
        )

        # Create a temporary file to store the problem for subprocesses
        self._problem_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        try:
            pickle.dump(problem, self._problem_file)
            self._problem_file.close()
        except Exception as e:
            self._problem_file.close()
            os.unlink(self._problem_file.name)
            raise e

        self.logger.info("Direct evaluation ready")

    @property
    def best(self):
        best_analyzer = self._analyzers.get("Best")
        return best_analyzer.best if best_analyzer else None

    @property
    def name(self):
        return self._name

    @property
    def evaluators(self):
        """
        Compatibility property for legacy code that references evaluators.
        Returns a mock object with attributes needed by strategies.
        """
        if self.config.evaluation_method == "dask":

            class DaskEvaluatorsMock:
                def __init__(self, strategy):
                    self.strategy = strategy

                @property
                def outstanding(self):
                    # Return list of pending task keys
                    return list(self.strategy.pending.keys())

                def __len__(self):
                    # Return number of Dask workers
                    if hasattr(self.strategy, "_client"):
                        return len(self.strategy._client.scheduler_info()["workers"])
                    return 0

            return DaskEvaluatorsMock(self)
        else:  # direct evaluation

            class DirectEvaluatorsMock:
                def __init__(self, strategy):
                    self.strategy = strategy

                @property
                def outstanding(self):
                    # Return list of pending task keys
                    return list(self.strategy.pending.keys())

                def __len__(self):
                    # Return number of subprocesses
                    return getattr(self.strategy, "_n_processes", 1)

            return DirectEvaluatorsMock(self)

    def _run(self):
        self.eventbus.publish("start", terminate=True)
        self._start = time_module.time()
        self.eventbus.register(self)
        self.logger.info("Strategy '%s' started" % self._name)
        self.loops = 0

        while True:
            self.loops += 1

            # execute the actual strategy
            points = self.execute()

            # Update progress status
            self._update_progress_status()

            if self.config.evaluation_method == "dask":
                self._run_dask_evaluation(points)
            elif self.config.evaluation_method == "direct":
                self._run_direct_evaluation(points)

            self.jobs_per_client = max(
                1, int(min(self.config.max_eval / 50.0, 1.0 / self.avg_time_per_task))
            )

            # show heuristic performances after each round
            # logger.info('  '.join(('%s:%.3f' % (h, h.performance) for h in
            # heurs)))

            # stopping criteria
            if len(self.results) > self.config.max_eval:
                break

            # limit loop speed - sleep briefly
            time_module.sleep(1e-3)

        self._cleanup()

    def _run_dask_evaluation(self, points):
        """Run evaluation using Dask distributed computing."""

        # Helper function to evaluate a point using the problem
        def evaluate_point(problem, point):
            """Evaluate a single point using the problem instance"""
            return problem(point)

        # distribute work using Dask futures
        # Submit each point as a separate task
        new_futures = []
        for point in points:
            future = self._client.submit(
                evaluate_point,
                self._problem_future,
                point,
                pure=False,  # Function may have side effects
            )
            new_futures.append(future)

        # and don't forget, this updates the statistics
        self._add_tasks(new_futures)

        # collect new results for each finished task, hand them over to result DB
        new_results = []
        for future_id in self.new_finished:
            future = self.pending.pop(future_id, None)
            if future is not None:
                try:
                    result = future.result()
                    if isinstance(result, list):
                        new_results.extend(result)
                    else:
                        new_results.append(result)
                except Exception as e:
                    self.logger.error("Task failed with error: %s" % e)

        self.results += new_results

    def _run_direct_evaluation(self, points):
        """Run evaluation using direct subprocess calls."""
        import subprocess
        import pickle
        import tempfile
        import os
        import tempfile
        import os

        # Submit each point as a separate subprocess task
        new_processes = []
        temp_files = []
        task_ids = []

        for i, point in enumerate(points):
            # Create task ID
            task_id = f"direct_task_{self.loops}_{i}"
            task_ids.append(task_id)

            # Create temporary file for the point
            point_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
            try:
                pickle.dump(point, point_file)
                point_file.close()
                temp_files.append(point_file.name)

                # Create temporary file for the result
                result_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
                result_file.close()
                temp_files.append(result_file.name)

                # Launch subprocess
                cmd = [
                    "python3",
                    "-c",
                    f"""
import pickle
import sys
sys.path.insert(0, '{os.path.dirname(os.path.abspath(__file__))}')
from panobbgo.utils import evaluate_point_subprocess

# Load problem and point
with open('{self._problem_file.name}', 'rb') as f:
    problem = pickle.load(f)
with open('{point_file.name}', 'rb') as f:
    point = pickle.load(f)

# Evaluate and save result
result = evaluate_point_subprocess(problem, point)
with open('{result_file.name}', 'wb') as f:
    pickle.dump(result, f)
""",
                ]

                process = subprocess.Popen(cmd)
                new_processes.append((process, result_file.name, task_id))

            except Exception as e:
                point_file.close()
                os.unlink(point_file.name)
                if "result_file" in locals():
                    result_file.close()
                    os.unlink(result_file.name)
                raise e

        # Add tasks to pending (for compatibility with existing code)
        for task_id in task_ids:
            self.pending[task_id] = task_id  # Mock pending entry

        # Wait for all processes to complete and collect results
        self.new_finished = []
        new_results = []
        for process, result_file, task_id in new_processes:
            try:
                process.wait(timeout=30)  # 30 second timeout per evaluation
                if process.returncode == 0:
                    with open(result_file, "rb") as f:
                        result = pickle.load(f)
                        if isinstance(result, list):
                            new_results.extend(result)
                        else:
                            new_results.append(result)
                    self.new_finished.append(task_id)
                    self.finished.append(task_id)
                else:
                    self.logger.error(
                        "Subprocess evaluation failed with return code: %d"
                        % process.returncode
                    )
            except subprocess.TimeoutExpired:
                process.kill()
                self.logger.error("Subprocess evaluation timed out")
            except Exception as e:
                self.logger.error("Task failed with error: %s" % e)

        # Remove completed tasks from pending
        for task_id in self.new_finished:
            self.pending.pop(task_id, None)

        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

        self.results += new_results

    def _update_progress_status(self):
        """
        Update the progress status line with current optimization information.
        """
        if not hasattr(self, 'panobbgo_logger'):
            return

        progress_reporter = self.panobbgo_logger.progress_reporter
        if not progress_reporter.status_enabled:
            return

        # Gather status information
        current_evals = len(self.results)
        try:
            max_evals = int(self.config.max_eval)
        except (TypeError, ValueError):
            max_evals = 1000
        budget_pct = (current_evals / max_evals) * 100 if max_evals > 0 else 0

        # Calculate ETA
        if hasattr(self, '_start') and self._start is not None:
            elapsed_time = time_module.time() - self._start
            if current_evals > 0:
                avg_time_per_eval = elapsed_time / current_evals
                remaining_evals = max(0, max_evals - current_evals)
                eta_seconds = int(avg_time_per_eval * remaining_evals)
            else:
                eta_seconds = 0
        else:
            eta_seconds = 0

        # Get convergence estimate (simplified: distance to best known)
        convergence = 0.0
        if hasattr(self, 'best') and self.best is not None:
            # Simple convergence measure: could be improved
            convergence = min(100.0, (current_evals / max_evals) * 100)

        # Get best function value
        best_value = None
        if hasattr(self, 'best') and self.best is not None:
            best_value = self.best.fx

        # Get strategy-specific information (for bandit strategies)
        extra_fields = {}
        if hasattr(self, '_get_status_info'):
            extra_fields = self._get_status_info()

        # Update status
        progress_reporter.update_status(
            budget_pct=budget_pct,
            eta_seconds=eta_seconds,
            convergence=convergence,
            best_value=best_value,
            current_evals=current_evals,
            max_evals=max_evals,
            extra_fields=extra_fields
        )

    def execute(self):
        """
        Overwrite this method when you extend this base strategy.
        """
        raise Exception(
            "You need to extend the class StrategyBase and overwrite this execute method."
        )

    def _cleanup(self):
        """
        cleanup + shutdown
        """
        self.eventbus.publish("finished")
        self._end = time_module.time()

        if self.config.evaluation_method == "dask":
            # Cancel any outstanding Dask futures
            for future_id, future in list(self.pending.items()):
                try:
                    future.cancel()
                except:
                    pass

            # Close Dask client
            if hasattr(self, "_client"):
                self._client.close()
        elif self.config.evaluation_method == "direct":
            # Clean up problem file for direct evaluation
            if hasattr(self, "_problem_file"):
                import os

                try:
                    os.unlink(self._problem_file.name)
                except:
                    pass

        # Finalize progress reporting
        if hasattr(self, 'panobbgo_logger'):
            self.panobbgo_logger.progress_reporter.finalize()

        self.logger.info(
            "Strategy '%s' finished after %.3f [s] and %d loops."
            % (self._name, self._end - self._start, self.loops)
        )

        self.info()
        self.results.info()
        [m.__stop__() for m in self.analyzers + self.heuristics]

        # Close Dask client
        if hasattr(self, "_client"):
            self._client.close()

        if self.config.ui_show:
            self.ui.finish()  # blocks figure window

    def _add_tasks(self, new_futures):
        """
        Accounting routine for the parallel tasks, only used by :meth:`.run`.
        Adapted for Dask futures instead of IPython message IDs.
        """
        if new_futures is not None:
            for future in new_futures:
                # Use future.key as identifier
                self.pending[future.key] = future

        # Find completed futures
        self.new_finished = []
        completed_keys = []

        for future_id, future in list(self.pending.items()):
            if future.done():
                self.new_finished.append(future_id)
                completed_keys.append(future_id)
                self.finished.append(future_id)

                # Calculate elapsed time if available
                if hasattr(future, "done") and future.done():
                    try:
                        # Get task duration from Dask
                        # Note: This is approximate, based on completion time
                        self.tasks_walltimes[future_id] = 0.1  # placeholder
                    except:
                        pass

        if time_module.time() - self.show_last > float(self.config.show_interval):
            self.info()
            self.show_last = time_module.time()

    def info(self):
        """ """
        avg = self.avg_time_per_task
        pend = len(self.pending)
        fini = len(self.finished)
        peval = len(self.results)
        s = (
            "{0:4d} ({1:4d}) pnts | Tasks: {2:3d} pend, {3:3d} finished | "
            "{4:6.3f} [s] cpu, {5:6.3f} [s] wall, {6:6.3f} [s/task]".format(
                peval, len(self.results), pend, fini, self.time_cpu, self.time_wall, avg
            )
        )
        self.slogger.info(s)

    @property
    def avg_time_per_task(self):
        """
        :return float: average time per task
        """
        if len(self.tasks_walltimes) > 1:
            return np.average(list(self.tasks_walltimes.values()))
        self.slogger.warning("avg time per task for 0 tasks! -> returning NaN")
        return np.nan

    @property
    def time_wall(self):
        """
        wall time in seconds
        """
        return time_module.time() - self.time_start

    @property
    def time_cpu(self):
        """
        effective cpu time in seconds
        """
        return time_module.process_time()

    @property
    def time_start_str(self):
        return time_module.ctime(self.time_start)
