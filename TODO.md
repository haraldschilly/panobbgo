# TODO

## Setup & Modernization (Completed)
- [x] Restructure repository: Move `panobbgo.lib` to `panobbgo/lib`.
- [x] Modernize `setup.py` / Create `pyproject.toml`.
- [x] Update dependencies in `requirements.txt`.
- [x] Replace `nose` with `pytest`.
- [x] Update imports after restructuring.
- [x] Run and fix existing tests.
- [x] Add type hinting where possible.
- [x] Update `README.md` with new installation and usage instructions.
- [x] Setup CI/CD (GitHub Actions) - *optional but recommended*.

## Recent Improvements

### BIPOP-CMA-ES Restart Mode (2026-04-20)
- [x] **Added BIPOP-CMA-ES restart support to `CMAES` heuristic** (`panobbgo/heuristics/cma_es.py`)
  - New `restart_mode` parameter: ``"ipop"`` (default, existing) or ``"bipop"`` (new)
  - BIPOP alternates two restart regimes following Hansen (2009):
    * **Large regime**: geometric population growth ``λ_l = 2^k · λ_default``
      (where ``k`` is the number of large-regime selections so far), σ resets to default
    * **Small regime**: random small population
      ``λ_s = ⌊λ_default · (½ · λ_l/λ_default)^(U[0,1]²)⌋`` and random small step size
      ``σ_s = σ_default · 10^(-2·U[0,1])``
  - Regime selection: after each restart, the regime that has accumulated *fewer*
    cumulative evaluations is selected next (ties → large)
  - New properties: `bipop_regime`, `bipop_evals_large`, `bipop_evals_small`
  - Refactored common restart bookkeeping into shared `_apply_restart()` helper
  - Reference: N. Hansen (2009). "Benchmarking a BI-Population CMA-ES on the
    BBOB-2009 Function Testbed." GECCO Workshop on BBOB.
- [x] **Updated `BIPOP_CMAES` strategy in full benchmark harness** (`panobbgo/harness.py`)
  - Now uses real BIPOP via `restart_mode="bipop"` (previously was just IPOP with more restarts)
  - Pairs `CMAES(sigma0=0.3, restart_mode="bipop")` with diverse Restart analyzer (max 10 restarts)
- [x] **18 new tests** (`tests/test_heuristic_cmaes.py::TestCMAESBIPOP` + integration test)
  - Parameter validation: default mode is "ipop"; invalid modes raise ValueError
  - Initial state: large regime, zero evals tracked
  - Regime alternation: balances cumulative budget within one delta
  - Large regime: geometric population growth `λ_l = 2^k · λ_default`
  - Small regime: λ ≥ base, σ ≤ default
  - Distribution state resets correctly (paths, covariance, eigendecomposition)
  - Box-clamped emission post-restart, base_lam preserved
  - IPOP path unchanged when `restart_mode="ipop"` (no BIPOP attribution)
  - Integration test: BIPOP-CMA-ES on Rastrigin reaches < 20 within 80 evals
- [x] **Documentation updated**
  - `doc/source/guide_architecture.rst`: CMAES section now documents IPOP and BIPOP
    schemes with mathematical formulas and selection rule
  - `doc/source/guide_usage.rst`: New "Highly multimodal problems with BIPOP-CMA-ES"
    section with worked example and IPOP-vs-BIPOP guidance; portfolio table updated
  - `TODO.md`: this entry

### IPOP-CMA-ES Restart Support (2026-04-19)
- [x] **Added IPOP restart to `CMAES` heuristic** (`panobbgo/heuristics/cma_es.py`)
  - `on_restart(center, reason)` handler: moves search mean to new center, doubles λ (IPOP)
  - Resets covariance matrix C, evolution paths p_c/p_σ, and step size σ to initial values
  - Flushes stale pending/in-flight generation results on restart
  - Recomputes all CMA-ES adaptation constants (c_σ, d_σ, c_c, c_1, c_μ) for new population
  - `ipop_factor` parameter (default 2.0) controls per-restart population growth multiplier
  - `restart_count` property tracks total number of IPOP restarts triggered
  - `_base_lam` records the initial population size (preserved across restarts)
  - Reference: Auger & Hansen (2005). "A restart CMA evolution strategy with increasing
    population size." CEC 2005.
- [x] **Added `IPOP_CMAES` strategy to standard benchmark harness** (`panobbgo/harness.py`)
  - Pairs `CMAES(sigma0=0.3, ipop_factor=2.0)` with `Restart(patience=None, restart_strategy="diverse", max_restarts=5)`
  - `Sensitivity` analyzer included for adaptive Nearby perturbations
- [x] **Added `BIPOP_CMAES` strategy to full benchmark harness**
  - Same as IPOP_CMAES but with `max_restarts=10` for the larger 500-eval budget
- [x] **25 comprehensive tests** (`tests/test_heuristic_cmaes.py`)
  - Unit tests for: default/custom ipop_factor, restart_count tracking, population doubling
  - Correctness tests: mean moves to center, sigma resets, paths reset, covariance resets
  - Behavioral tests: pending queue flushed, new generation emitted, box-constraint preservation
  - Weight renormalization, base_lam preservation, multiple restarts
  - Integration tests: IPOP on Rastrigin 2D (200 evals), restart triggered with short patience
- [x] **Documentation updated**
  - `doc/source/guide_architecture.rst`: CMAES entry now documents IPOP restart capability
  - `doc/source/guide_usage.rst`: New "Multimodal problems with IPOP-CMA-ES" section with
    worked example, parameter guide, and comparison to plain CMA-ES

### CMA-ES Heuristic & Core Reward Fix (2026-04-18)
- [x] **Implemented `CMAES` heuristic** (`panobbgo/heuristics/cma_es.py`)
  - Pure-NumPy implementation of the canonical CMA-ES algorithm (Hansen 2016)
  - Async-compatible: tracks results per generation via `who = "CMAES:g<gen>:i<idx>"` tags
  - Adapts covariance matrix C and step size σ from evaluated offspring
  - Lazy eigendecomposition with condition-number guard (resets to spherical if > 1e7)
  - Parameters: `sigma0` (initial step-size fraction), `popsize` (overrides λ=4+3 ln n),
    `min_results_fraction` (fraction of λ before update trigger, default 0.5 = μ)
  - Gold standard for smooth/ridge-following problems (Rosenbrock, ill-conditioned quadratics)
- [x] **Added `CMAES` to standard and full harness strategies** (`panobbgo/harness.py`)
  - `CMAES_Portfolio` strategy in standard mode: LatinHypercube + CMAES + Nearby + NelderMead
  - `CMAES_GP` strategy in full mode: LatinHypercube + CMAES + GaussianProcessHeuristic + NelderMead
  - Intentionally excluded from quick mode (75 evals) — CMA-ES needs ≥ 100 evals to converge
- [x] **Fixed `StrategyBase.heuristic()` lookup for compound `who` strings** (`panobbgo/core.py`)
  - Heuristics like CMAES and DifferentialEvolution embed generation/UUID info in `who`
    (e.g., `"CMAES:g3:i0"`, `"DifferentialEvolution:abc123"`)
  - `heuristic(who)` now falls back to the prefix before `:` if the full key is not found
  - Prevents spurious `KeyError` in `StrategyRewarding.on_new_best` and `_reward_near_best`
- [x] **20 comprehensive tests** (`tests/test_heuristic_cmaes.py`)
  - Initialisation, parameter validation, point emission within bounds
  - Update triggering from partial results, mean convergence, sigma adaptation
  - Covariance positive-definiteness, weight normalisation, foreign-result handling
  - End-to-end integration test on Rosenbrock 2D (150 evals, fx < 5.0)
- [x] **Documentation updated**
  - `doc/source/guide_architecture.rst`: CMAES added to "Population-based" heuristics section
  - `doc/source/guide_usage.rst`: Added CMA-ES portfolio table entry and usage example

### Bayesian Optimization Harness Integration & UCB Bug Fix (2026-04-17)
- [x] **BayesOpt_GP strategy added to standard harness** (`panobbgo/harness.py`)
  - New `BayesOpt_GP` strategy spec added to `_make_standard_strategies()` (200-eval budget)
  - Uses `GaussianProcessHeuristic(n_restarts=5)` + `LatinHypercube(div=4)` + `Nearby` + `NelderMead`
  - Demonstrates GP-based Bayesian optimization within the reproducible harness
- [x] **BayesOpt_Enhanced added to full harness** (`panobbgo/harness.py`)
  - New `BayesOpt_Enhanced` strategy spec added to `_make_full_strategies()` (500-eval budget)
  - Combines `GaussianProcessHeuristic(n_restarts=10)` + `DifferentialEvolution` + `NelderMead`
  - DifferentialEvolution provides global search; GP provides surrogate-guided exploitation
- [x] **Fixed UCB acquisition function bug** (`panobbgo/heuristics/gaussian_process.py`)
  - `_upper_confidence_bound` was maximising LCB instead of minimising it (wrong for minimisation)
  - Fixed: method now returns `-(μ - κσ)` so the outer maximiser correctly minimises LCB
  - Acquisition functions EI and PI were already correct; only UCB was affected
- [x] **Documentation updated** (`doc/source/guide_architecture.rst`, `doc/source/guide_usage.rst`)
  - `guide_architecture.rst`: Added `GaussianProcessHeuristic`, `DifferentialEvolution`,
    `FeasibleSearch`, `ConstraintGradient`, `LocalPenaltySearch`, `ConstraintRepair` to heuristics
  - `guide_architecture.rst`: Added `StrategyUCB`, `StrategyThompsonSampling`, `StrategyLinUCB`,
    `StrategyPhased` with mathematical descriptions
  - `guide_usage.rst`: Added "Bayesian Optimization with Gaussian Process" section with
    acquisition function details, EIC description, and two-phase BO workflow example
  - `guide_usage.rst`: Updated heuristic portfolio table and recommended configurations

### Sensitivity-Aware Nearby Heuristic & StrategySpec Analyzers (2026-04-15)
- [x] **Sensitivity-Aware `Nearby` Heuristic** (`panobbgo/heuristics/nearby.py`)
  - Added `on_new_sensitivity(importance)` event handler to `Nearby`
  - When `Sensitivity` analyzer is active and has published importance scores,
    `Nearby` scales per-dimension perturbations by importance (normalised so overall
    magnitude is preserved)
  - New `sensitivity_scale` constructor parameter controls contrast sharpness (default 1.0)
  - For `axes="all"`: each dimension's step is multiplied by its (normalised) weight
  - For `axes="one"`: dimension is sampled proportionally to importance weights
  - Both `on_new_best` and `on_restart` use the sensitivity-aware `_make_perturbation` helper
  - Improves local search in high-dimensional problems where only a subset of dimensions matter
  - Added `_perturbation_weights()` helper returning normalised weights (mean = 1)
- [x] **`StrategySpec.analyzers` field** (`panobbgo/benchmark.py`)
  - Added optional `analyzers: List[Tuple[type, dict]]` field to `StrategySpec`
  - `create_strategy()` adds extra analyzers (e.g. `Sensitivity`, `Restart`) alongside heuristics
  - Four required analyzers (Best, Grid, Splitter, Convergence) still added in `initialize()`
- [x] **Sensitivity in Benchmark Strategies** (`panobbgo/harness.py`)
  - Added `Sensitivity(update_interval=20)` to `Rewarding_Diverse`, `UCB_Diverse`, and `Thompson_Diverse`
  - Enables adaptive Nearby perturbations in all adaptive benchmark strategies
- [x] **15 new tests** (`tests/test_heuristic_nearby_sensitivity.py`)
  - Verifies `_perturbation_weights()` normalisation and ordering
  - Confirms sensitivity-aware perturbations statistically bias important dimensions
  - Tests both `axes="all"` and `axes="one"` modes
  - Tests `on_restart` with/without sensitivity and with None center
  - Tests that sensitivity updates are immediately effective
  - Tests `StrategySpec.analyzers` round-trip and creation
- [x] **Documentation updated**
  - `doc/source/guide_architecture.rst`: updated event table and Nearby description
  - `doc/source/guide_usage.rst`: added Sensitivity-Aware Nearby section with example

### Benchmark Harness for Agent Feedback Loops (2026-02-23)
- [x] **Implemented `panobbgo/harness.py` – Reproducible Benchmark Harness**
  - `BenchmarkHarness` class: runs seeded, reproducible benchmark suites
  - Three modes: `quick` (3 problems × 2 strategies × 3 reps, 75 evals), `standard`, `full`
  - Per-run seed derivation for best-effort reproducibility across runs
  - Convergence trace extraction directly from the MultiIndex results DataFrame
  - ERT (Expected Running Time) and per-pair performance score in [0, 1]
  - Composite score = mean of per-pair scores; single scalar for before/after comparison
  - Full JSON serialisation / deserialisation (`save()` / `load()`)
  - `compare()` helper: diff two `HarnessResult` files, flag regressions/improvements
- [x] **`benchmark_harness.py` – CLI for Agent Loop**
  - `run`: execute benchmarks and save a timestamped JSON file
  - `score`: print human-readable summary + optional machine-readable JSON
  - `compare`: side-by-side diff with `--fail-on-regression` exit-code support
  - `list`: enumerate available problems and strategies per mode
- [x] **`tests/test_harness.py` – 60 tests covering all harness components**
  - Unit tests for metrics, serialisation, comparison, seed derivation
  - Smoke integration tests for end-to-end runs (single problem, 30 evals)
  - CLI tests via `main()` invocation

### Contextual Bandit Strategy (2025-01-13)
- [x] **Implemented StrategyLinUCB (Contextual Bandits)**
  - Implemented `StrategyLinUCB` with disjoint linear models for each heuristic.
  - Features include Bias, Budget Progress, and Recent Success Rate.
  - Added unit/integration test `tests/test_strategy_contextual.py`.
  - Updated `panobbgo/lib/classic.py` to support `Rosenbrock(dim=2)` kwargs.

### Thompson Sampling Strategy (2025-01-13)
- [x] **Implemented StrategyThompsonSampling**
  - Added new strategy using Beta-Bernoulli bandit logic
  - Implemented `reward` based on improvement magnitude
  - Implemented `execute` with randomized selection based on Beta samples
  - Added unit tests in `tests/test_strategy_thompson.py`

### PR #43 - Dask Memory Leak Fix & Test Suite Cleanup (2025-01-13)
- [x] **Fixed Critical Memory Leak in Dask Cleanup**
  - Added proper `LocalCluster` cleanup in `_setup_dask_cluster()` and shutdown code
  - Store cluster reference (`self._cluster`) to ensure worker processes are terminated
  - Call both `self._client.close()` AND `self._cluster.close()` during cleanup
  - Prevents memory blowup when running multiple tests that use Dask evaluation
- [x] **Deferred Dask Testing (Future Work - Weeks)**
  - Disabled all Dask-related tests (`test_config_init.py`, `test_dask_evaluation_integration()`)
  - Default test execution model is now "threaded" only
  - Dask evaluation still works in production, just not tested in test suite
  - TODO: Proper Dask test isolation and cleanup testing in future sprint

### PR #42 - FeasibleSearch & Test Warnings (2025-01-13)
- [x] **Test Suite Warnings Resolved**
  - Fixed NumPy RuntimeWarnings in convergence analyzer using `warnings.catch_warnings()`
  - Suppressed warnings for edge cases (identical values, small samples) in std deviation calculations
  - Skipped Dask evaluation integration test (focusing on threaded evaluation for now)
  - All 143 tests now pass with 1 skipped, 0 warnings
- [x] **FeasibleSearch Heuristic Enhanced**
  - Implemented biased line search using Beta(2,1) distribution for more efficient boundary finding
  - Improved comments explaining the line search strategy between feasible/infeasible points
  - Updated copyright year to 2012-2025 per project guidelines
  - All FeasibleSearch tests passing

## Framework Quality Assurance & Completion

### 🔴 CRITICAL: TDD Bug Fixes & Quality Validation (Priority 1)
**TDD Approach**: Write failing tests first, then implement fixes
- [x] **Optimization Loop Stability** - Major hanging issues resolved
  - [x] **FIXED**: Random heuristic infinite wait (main hang cause)
  - [x] **FIXED**: abs() errors in convergence analyzer and progress reporting
  - [x] Basic optimization now completes successfully
  - [ ] Full optimization loop robustness (complex threading - lower priority)
- [x] **Heuristic Functionality** - Core issues resolved
  - [x] **FIXED**: Random heuristic infinite wait (main hang cause)
  - [x] **VALIDATED**: Nearby heuristic generates correct points
  - [x] Added TDD tests for heuristic point generation
  - [ ] Full event system integration (lower priority)
- [x] **Dedensifyer Analyzer** - Fix critical implementation bugs
  - [x] Write TDD tests for proper initialization and grid management
  - [x] Fix constructor (missing strategy parameter)
  - [x] Fix undefined variables and wrong method signatures
  - [x] Validate hierarchical grid functionality
- [x] **Optimization Correctness Validation** - Add tests proving algorithms work
  - [x] Write tests validating convergence to known optima
  - [x] Compare optimization vs random baseline performance
  - [x] Add statistical significance testing

### 🟡 MEDIUM: Coverage Expansion on Validated Code (Priority 2)
**Revised Goal**: 75% coverage on components proven to work correctly
- [x] Expand UCB strategy tests (currently 91% - add edge cases)
- [x] Complete Best analyzer test coverage (currently 34%)
- [x] Add Grid analyzer comprehensive tests (currently 56%)
- [x] Test remaining heuristics: LBFGSB (30%), Nelder-Mead (51%)
- [x] Add integration tests for constrained optimization scenarios

### 🟢 LOW: Documentation & Polish (Priority 3)
- [x] Update documentation references from IPython parallel to Dask
- [x] Review and fix minor naming inconsistencies in guide documentation
- [x] Remove remaining IPython parallel references from code and documentation
- [ ] Review and potentially simplify UI components
- [ ] Add performance benchmarks comparing different strategies
- [ ] Review and optimize threading/event handling

### 🔵 DEFERRED: Dask Testing & Validation (Future Work - Weeks)
**Status**: Completed! Dask tests are isolated, pass locally, and memory leak fix is verified.
- [x] **Dask Test Isolation**: Properly isolate Dask tests to avoid port conflicts
  - Use pytest fixtures to ensure clean Dask cluster setup/teardown
  - Ensure each test gets a fresh LocalCluster with unique dashboard port
  - Test that cluster cleanup properly terminates all worker processes
- [x] **Re-enable Dask Tests**: Currently skipped tests
  - `tests/test_config_init.py` - testing_mode and dashboard configuration
  - `tests/test_integration.py::test_dask_evaluation_integration` - Dask evaluation
- [x] **Verify Memory Leak Fix**: Test that the LocalCluster cleanup fix prevents memory leaks
  - Run repeated Dask evaluations and monitor memory usage
  - Verify worker processes are terminated after cleanup
- [x] **Dask Production Usage**: While tests are disabled, Dask evaluation still works
  - Document current Dask usage patterns for production
  - Consider adding example scripts demonstrating Dask evaluation

## Known Issues & Technical Debt

### Strategy Lifecycle Management (Systemic Issue)
**Problem**: Real strategy instances (StrategyRoundRobin, StrategyRewarding) start background processes (via Dask) that don't clean up properly when tests complete. This causes:
- Test hangs when multiple tests use real strategies (PR #35, PR #32)
- Resource leaks in test suites
- Unreliable benchmark tests

- `strategy.start()` initializes background threads/processes
- [x] **FIXED**: Strategy lifecycle methods (`__stop__`, `_cleanup`) implemented.
- [x] **FIXED**: Context manager support (`__enter__`, `__exit__`) implemented.
- Tests can now properly tear down strategies using `strategy.stop()` or `with` blocks.

**Current Workarounds**:
- Unit tests: Use `@mock.patch("panobbgo.core.StrategyBase")` to avoid real strategies
- Integration tests: Skip tests that hang (e.g., `test_heuristic_tracking` in benchmarks)
- Set `evaluation_method="threaded"` helps but doesn't fully solve cleanup issues

**Proper Solution Needed**:
- [x] **FIXED**: Cleanly terminate background processes.
- [x] **FIXED**: Implementation of `strategy.cleanup()` methods.
- [x] **FIXED**: Context manager support (`__enter__`/`__exit__`).
- [x] Implement pytest fixtures for automatic strategy setup/teardown in tests.
- [ ] Review all Dask distributed usage for best practice cleanup patterns.

**Affected Files**:
- `panobbgo/core.py` - StrategyBase class needs lifecycle methods
- `tests/test_heuristic_feasible.py` - Fixed by using mocked strategies (PR #35)
- `benchmarks/test_benchmarks.py` - Skipped hanging test (PR #32)

### Benchmark Heuristic Tracking Issues (PR #32)
**Bug in convergence_trace logic** (`benchmarks/test_benchmarks.py:88-93`) - **FIXED**:
- ~~When `best_fx == float('inf')` (first evaluation), `old_best_fx` is set to `result.fx`~~
- ~~This causes `improvement = result.fx - result.fx = 0`, which is incorrect~~
- **Fixed**: First improvement now correctly recorded as `result.fx` (function value from baseline)
- **Fixed**: Subsequent improvements correctly calculated as `best_fx - result.fx`

### 🎯 TARGET: 75% Coverage on Validated Components
**Prerequisites**: All Priority 1 items completed with TDD validation
**Quality Metrics**: Correctness + Coverage (not just coverage)
**Status**: Core issues resolved, coverage stands at ~71%.

## Known Issues & Technical Debt

### Strategy.start() Hang Bug (FIXED)
**CRITICAL**: `strategy.start()` doesn't return after reaching `max_eval` evaluations
- **Status**: FIXED by addressing result collection deadlocks and improving cleanup in [PR #38](https://github.com/haraldschilly/panobbgo/pull/38).

### PR #36 Bug Fixes (Merged)
**Fixed Issues** - All good fixes:
- [x] **Splitter.Box.__ranges** - Fixed `.ptp()` call to work with BoundingBox objects (`panobbgo/analyzers/splitter.py:215-220`)
- [x] **memoize decorator** - Added handling for unhashable NumPy arrays by converting to bytes (`panobbgo/utils.py:205-230`)
- [x] **Analyzer name consistency** - Changed "splitter"/"best" to "Splitter"/"Best" (Random, WeightedAverage heuristics)
- [x] **Random heuristic initialization** - Added logic to get root leaf from Splitter on start (`panobbgo/heuristics/random.py:38-48`)
