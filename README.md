# Panobbgo: Parallel Noisy Black-Box Global Optimization

[![Tests](https://github.com/haraldschilly/panobbgo/actions/workflows/tests.yml/badge.svg)](https://github.com/haraldschilly/panobbgo/actions/workflows/tests.yml)

Panobbgo minimizes a function over a box in $R^n$ (n = dimension of the problem)
while respecting a vector of constraint violations.

Panobbgo is a **framework for black-box optimization** that includes **out-of-the-box runnable examples** for testing and demonstration. Use the example scripts in `sketchpad/` to see complete optimization runs, or import components directly to build custom optimization pipelines.

## Documentation

* [📚 Documentation](https://haraldschilly.github.io/panobbgo/) - Complete user guide with setup instructions
* [Guide](doc/source/guide.rst) - Source documentation files (reStructuredText)

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
* Dask &ge; 2023.0

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

1. **For testing/development**: No cluster setup needed - Panobbgo automatically starts a local 2-worker Dask cluster
2. **For production/custom clusters**: Setup your Dask cluster according to the [Dask distributed documentation](https://docs.dask.org/en/stable/deploying.html)
3. `panobbgo.lib` contains the problem definitions (Rosenbrock, HelicalValley, etc.)
4. After running it the first time, it will create a `config.ini` file
5. Configure your Dask cluster settings if needed

### Running Optimization

1. **Default (automatic)**: Just run your optimization script - a local cluster starts automatically
2. **Custom cluster**: Start your Dask cluster manually: `dask scheduler & dask worker localhost:8786 --nprocs 4 &`

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

## Authors

* Harald Schilly <harald.schilly@gmail.com>

## History

This project was revived in 2026 with the help of coding agents like Jules and Claude Code.
