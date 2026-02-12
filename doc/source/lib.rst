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
- **NoisyProblem**: Adds controlled Gaussian noise

.. automodule:: panobbgo.lib
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: panobbgo.lib.wrappers
   :members:
   :undoc-members:
   :show-inheritance:

