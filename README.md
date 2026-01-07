# Panobbgo: Parallel Noisy Black-Box Global Optimization

[![Tests](https://github.com/haraldschilly/panobbgo/actions/workflows/tests.yml/badge.svg)](https://github.com/haraldschilly/panobbgo/actions/workflows/tests.yml)

Panobbgo minimizes a function over a box in $R^n$ (n = dimension of the problem)
while respecting a vector of constraint violations.

## Documentation

* [HTML](http://haraldschilly.github.com/panobbgo/html/)
* [PDF](http://haraldschilly.github.com/panobbgo/pdf/panobbgo.pdf)

## Installation

### Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package manager. Install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and install panobbgo:

```bash
git clone https://github.com/haraldschilly/panobbgo.git
cd panobbgo
uv sync --extra dev
```

### Using pip

```bash
git clone https://github.com/haraldschilly/panobbgo.git
cd panobbgo
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Dependencies

* Python &ge; 3.8
* NumPy &ge; 2.0
* SciPy &ge; 1.16
* matplotlib &ge; 3.0
* pandas &ge; 2.0
* statsmodels &ge; 0.14
* IPython &ge; 9.0

Development dependencies:
* pytest &ge; 9.0
* pytest-cov &ge; 7.0
* black, flake8, mypy

## Running Tests

```bash
# With UV
uv run pytest

# With pip/virtualenv
pytest

# Run with coverage
pytest --cov=panobbgo
```

All 27 tests should pass.

## Type Checking

This project uses Pyright for static type checking:

```bash
# Run type checker
uv run pyright panobbgo

# Or with pip/virtualenv
pyright panobbgo
```

## Usage

### One-time Setup

1. Setup your IPython cluster according to the [IPython parallel documentation](https://ipyparallel.readthedocs.io/)
2. `panobbgo.lib` contains the problem definitions (Rosenbrock, HelicalValley, etc.)
3. After running it the first time, it will create a `config.ini` file
4. Configure your IPython profile name if it's not `default`

### Running Optimization

1. Start your IPython cluster: `ipcluster start`
2. Run your optimization script (see examples in the repository)

Example:
```python
from panobbgo.lib.classic import Rosenbrock
from panobbgo.core import StrategyRoundRobin

# Define the problem
problem = Rosenbrock(dim=5)

# Setup and run optimization
strategy = StrategyRoundRobin(problem)
# ... configure heuristics and run
```

## License

<a href="http://www.apache.org/licenses/LICENSE-2.0">Apache 2.0</a>

## Credits

Based on ideas of Snobfit:

* http://reflectometry.org/danse/docs/snobfit/

* http://www.mat.univie.ac.at/~neum/software/snobfit/

