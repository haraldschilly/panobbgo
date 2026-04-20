Strategies Module
=================

Available optimization strategies:

- **StrategyBase**: Abstract base class for all strategies
- **StrategyRoundRobin**: Simple round-robin point evaluation
- **StrategyRewarding**: Adaptive heuristic selection based on performance (softmax multi-armed bandit)
- **StrategyUCB**: Upper Confidence Bound (UCB1) algorithm for principled exploration/exploitation
- **StrategyThompsonSampling**: Probabilistic selection using Thompson Sampling (Beta-Bernoulli bandit)
- **StrategyLinUCB**: Contextual bandit using disjoint linear UCB models over budget-progress / success-rate features
- **StrategyPhased**: Budget-phased meta-strategy composing different sub-strategies and heuristic portfolios across phases

.. automodule:: panobbgo.strategies
   :members:
   :undoc-members:
   :show-inheritance:

