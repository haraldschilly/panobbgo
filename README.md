# Panobbgo: Parallel Noisy Black-Box Global Optimization

[![Tests](https://github.com/haraldschilly/panobbgo/actions/workflows/tests.yml/badge.svg)](https://github.com/haraldschilly/panobbgo/actions/workflows/tests.yml)

Panobbgo minimizes a function over a box in $R^n$ (n = dimension of the problem)
while respecting a vector of constraint violations.

It is a **framework**: you compose a *strategy* (a multi-armed bandit over
point generators), a portfolio of *heuristics* (random/space-filling designs,
local search, DE/CMA-ES/PSO, surrogate models, ...) and *analyzers* into an
optimization run in a few lines of Python. Evaluations are dispatched in
parallel — local threads by default, optionally a Dask cluster.

## Documentation

* [Documentation](https://haraldschilly.github.io/panobbgo/) — user guide and API reference
* [User guide sources](doc/source/guide.rst) — reStructuredText, built with Sphinx
* [Benchmarking guide](doc/source/guide_benchmarking.rst) — how quality is measured (composite score, statistical acceptance)
* `AGENTS.md` — rules and commands for contributors and coding agents; `TODO.md` — current status

## Installation

Panobbgo requires Python 3.11 or later. Core dependencies: NumPy, SciPy,
pandas, matplotlib, statsmodels, scikit-learn (see
[pyproject.toml](pyproject.toml) for the exact list). Dask is an optional
extra (`dask`) for distributed evaluation.

### Using UV (recommended)

Install [UV](https://github.com/astral-sh/uv), then:

```bash
git clone https://github.com/haraldschilly/panobbgo.git
cd panobbgo
uv sync --extra dev
```

### Using pip

```bash
git clone https://github.com/haraldschilly/panobbgo.git
cd panobbgo
pip install -e ".[dev]"
```

## Running tests

```bash
uv run pytest -q -n 4          # full suite, ~2000 tests, about a minute
uv run pytest --cov=panobbgo   # with coverage
uv run pyright panobbgo        # type checking
uv run ruff format --check .   # formatting (the CI gate)
```

Serial `uv run pytest` also works; `-n 4` uses pytest-xdist.

## Usage

Threaded local evaluation needs no setup. A minimal run:

```python
from panobbgo.lib.classic import Rosenbrock
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.heuristics import Center, Random, NelderMead

problem = Rosenbrock(dims=5)
strategy = StrategyRewarding(problem, max_evaluations=500)
strategy.add(Center)
strategy.add(Random)
strategy.add(NelderMead)
strategy.start()

print(strategy.best)          # best result found
df = strategy.results.results # pandas DataFrame of all evaluations
```

`panobbgo.lib.classic` contains the built-in test problems (Rosenbrock,
Rastrigin, Himmelblau, Shekel, ...). To define your own, subclass
`panobbgo.lib.Problem` and implement `eval(x)` (and optionally
`eval_constraints(x)`). Configuration (evaluation backend, budgets, logging)
lives in `config.yaml` / `~/.panobbgo/config.ini`; see the
[usage guide](doc/source/guide_usage.rst) for Dask setup, constrained
problems, persistent storage and more examples.

## Repository layout

* `panobbgo/` — the library (`core`, `strategies/`, `heuristics/`, `analyzers/`, `lib/` problems, benchmark harness)
* `tests/` — pytest suite; `benchmarks/` — micro-benchmarks and comparison scripts
* `benchmark_harness.py`, `scripts/` — the composite-score and IOH/MA-BBOB benchmark CLIs
* `doc/` — Sphinx documentation; `planning/` — goals, design notes and history
* `sketchpad/` — unpolished scratch scripts, not maintained

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
