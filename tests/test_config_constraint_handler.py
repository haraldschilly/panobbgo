# -*- coding: utf8 -*-
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.lib.classic import Rosenbrock
from panobbgo.lib.constraints import (
    DefaultConstraintHandler,
    PenaltyConstraintHandler,
    AugmentedLagrangianConstraintHandler,
    DynamicPenaltyConstraintHandler,
)
from tests.support import PanobbgoTestCase
import numpy as np
import pytest


class TestConstraintHandlerConfiguration(PanobbgoTestCase):
    def setUp(self):
        super().setUp()
        self.problem = Rosenbrock(dims=2)
        # Reset config pollution from singleton
        from panobbgo.config import Config

        config = Config(testing_mode=True)
        config.constraint_handler = "DefaultConstraintHandler"

    def test_default_handler(self):
        """Test that DefaultConstraintHandler is used by default."""
        strategy = StrategyRewarding(self.problem, testing_mode=True)
        assert isinstance(strategy.constraint_handler, DefaultConstraintHandler)

    def test_augmented_lagrangian_handler(self):
        """Test configuring AugmentedLagrangianConstraintHandler via kwargs."""
        strategy = StrategyRewarding(
            self.problem, testing_mode=True, constraint_handler="AugmentedLagrangianConstraintHandler", rho=5.0
        )
        assert isinstance(strategy.constraint_handler, AugmentedLagrangianConstraintHandler)
        assert strategy.constraint_handler.mu == 5.0

    def test_penalty_handler(self):
        """Test configuring PenaltyConstraintHandler via kwargs."""
        strategy = StrategyRewarding(
            self.problem,
            testing_mode=True,
            constraint_handler="PenaltyConstraintHandler",
            rho=10.0,
            constraint_exponent=2,
        )
        assert isinstance(strategy.constraint_handler, PenaltyConstraintHandler)
        assert strategy.constraint_handler.rho == 10.0
        assert strategy.constraint_handler.exponent == 2.0

    def test_dynamic_penalty_handler(self):
        """Test configuring DynamicPenaltyConstraintHandler via kwargs."""
        strategy = StrategyRewarding(
            self.problem,
            testing_mode=True,
            constraint_handler="DynamicPenaltyConstraintHandler",
            dynamic_penalty_rate=0.05,
        )
        assert isinstance(strategy.constraint_handler, DynamicPenaltyConstraintHandler)
        assert strategy.constraint_handler.rate == 0.05

    def test_default_handler_uses_the_documented_rho(self):
        """The default handler's penalty is ``fx + 100·cv`` (config used to pass rho=1.0)."""
        from panobbgo.lib import Point, Result

        strategy = StrategyRewarding(self.problem, testing_mode=True)
        assert strategy.constraint_handler.rho == 100.0
        r = Result(Point(np.zeros(2), "t"), 1.0, cv_vec=np.array([2.0]))
        assert strategy.constraint_handler.get_penalty_value(r) == pytest.approx(201.0)

    def test_unset_settings_keep_the_class_defaults(self):
        """Every handler gets its own defaults for rho, exponent and rate when the config is silent."""
        s = StrategyRewarding(self.problem, testing_mode=True, constraint_handler="PenaltyConstraintHandler")
        assert (s.constraint_handler.rho, s.constraint_handler.exponent) == (100.0, 1.0)
        s = StrategyRewarding(self.problem, testing_mode=True, constraint_handler="DynamicPenaltyConstraintHandler")
        h = s.constraint_handler
        assert (h.rho_start, h.rate, h.exponent) == (10.0, 0.01, 2.0)
        s = StrategyRewarding(
            self.problem, testing_mode=True, constraint_handler="AugmentedLagrangianConstraintHandler"
        )
        assert (s.constraint_handler.mu, s.constraint_handler.rate) == (10.0, 2.0)

    def test_dynamic_penalty_rate_does_not_reach_the_augmented_lagrangian(self):
        """The ALM mu multiplier has its own key: a 0.05 growth rate made mu shrink every update."""
        s = StrategyRewarding(
            self.problem,
            testing_mode=True,
            constraint_handler="AugmentedLagrangianConstraintHandler",
            dynamic_penalty_rate=0.05,
        )
        assert s.constraint_handler.rate == 2.0
        s = StrategyRewarding(
            self.problem, testing_mode=True, constraint_handler="AugmentedLagrangianConstraintHandler", alm_rate=3.0
        )
        assert s.constraint_handler.rate == 3.0
