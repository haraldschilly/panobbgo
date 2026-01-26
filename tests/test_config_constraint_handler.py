# -*- coding: utf8 -*-
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.lib.classic import Rosenbrock
from panobbgo.lib.constraints import (
    DefaultConstraintHandler,
    PenaltyConstraintHandler,
    AugmentedLagrangianConstraintHandler,
    DynamicPenaltyConstraintHandler,
)
from panobbgo.utils import PanobbgoTestCase
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
            self.problem,
            testing_mode=True,
            constraint_handler="AugmentedLagrangianConstraintHandler",
            rho=5.0
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
            constraint_exponent=2
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
            dynamic_penalty_rate=0.05
        )
        assert isinstance(strategy.constraint_handler, DynamicPenaltyConstraintHandler)
        assert strategy.constraint_handler.rate == 0.05
