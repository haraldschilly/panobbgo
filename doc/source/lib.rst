Library Module
==============

Core problem definition classes:

- **Problem**: Base class for optimization problems
- **Point**: Represents a point in parameter space
- **Result**: Contains evaluation results
- **BoundingBox**: Defines parameter bounds

Problem Wrappers:

- **ProblemWrapper**: Base class for composable problem decorators
- **NormalizedProblem**: Scales all dimensions to [0, 1]
- **LogTransformProblem**: Applies log transform to objective
- **NoisyProblem** (:mod:`panobbgo.lib.noise`): deterministic, seeded noise models (the
  ``panobbgo.lib.wrappers.NoisyProblem`` adapter is deprecated)

.. automodule:: panobbgo.lib
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: panobbgo.lib.wrappers
   :members:
   :undoc-members:
   :show-inheritance:

