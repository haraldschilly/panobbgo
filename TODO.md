# TODO

## Setup & Modernization (Completed)
- [x] Restructure repository: Move `panobbgo.lib` to `panobbgo/lib`.
- [x] Modernize `setup.py` / Create `pyproject.toml`.
- [x] Update dependencies in `requirements.txt`.
- [x] Replace `nose` with `pytest`.
- [x] Update imports after restructuring.
- [x] Run and fix existing tests.
- [ ] Add type hinting where possible.
- [x] Update `README.md` with new installation and usage instructions.
- [x] Setup CI/CD (GitHub Actions) - *optional but recommended*.

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
  - [ ] Add statistical significance testing

### 🟡 MEDIUM: Coverage Expansion on Validated Code (Priority 2)
**Revised Goal**: 75% coverage on components proven to work correctly
- [ ] Expand UCB strategy tests (currently 91% - add edge cases)
- [ ] Complete Best analyzer test coverage (currently 34%)
- [ ] Add Grid analyzer comprehensive tests (currently 56%)
- [ ] Test remaining heuristics: LBFGSB (30%), Nelder-Mead (51%)
- [ ] Add integration tests for constrained optimization scenarios

### 🟢 LOW: Documentation & Polish (Priority 3)
- [ ] Update documentation references from IPython parallel to Dask
- [ ] Review and fix minor naming inconsistencies in guide documentation
- [ ] Remove remaining IPython parallel references from code and documentation
- [ ] Review and potentially simplify UI components
- [ ] Add performance benchmarks comparing different strategies
- [ ] Review and optimize threading/event handling

## Known Issues & Technical Debt

### Strategy Lifecycle Management (Systemic Issue)
**Problem**: Real strategy instances (StrategyRoundRobin, StrategyRewarding) start background processes (via Dask) that don't clean up properly when tests complete. This causes:
- Test hangs when multiple tests use real strategies (PR #35, PR #32)
- Resource leaks in test suites
- Unreliable benchmark tests

**Root Cause**: No proper lifecycle management (start/stop/cleanup methods) for strategies
- `strategy.start()` initializes background threads/processes
- No corresponding `strategy.stop()` or `strategy.cleanup()` method
- Tests have no way to properly tear down strategies

**Current Workarounds**:
- Unit tests: Use `@mock.patch("panobbgo.core.StrategyBase")` to avoid real strategies
- Integration tests: Skip tests that hang (e.g., `test_heuristic_tracking` in benchmarks)
- Set `evaluation_method="threaded"` helps but doesn't fully solve cleanup issues

**Proper Solution Needed**:
- [ ] Add `strategy.stop()` method to cleanly terminate background processes
- [ ] Add `strategy.cleanup()` method to release resources
- [ ] Implement `__enter__`/`__exit__` for context manager support
- [ ] Add pytest fixtures that properly setup/teardown strategies
- [ ] Review all Dask distributed usage for proper cleanup patterns

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
**Blocker**: Strategy.start() hang bug must be fixed before proper validation testing

## Known Issues & Technical Debt

### Strategy.start() Hang Bug (FIXED)
**CRITICAL**: `strategy.start()` doesn't return after reaching `max_eval` evaluations
- **Status**: FIXED by addressing result collection deadlocks and improving cleanup.
- **Needs Investigation**:
  - [ ] Why doesn't the main loop exit after max_eval?
  - [ ] Are results being counted correctly?
  - [ ] Is there a race condition with threaded evaluation?
  - [ ] Do heuristic threads prevent loop termination?

### PR #36 Bug Fixes (Merged)
**Fixed Issues** - All good fixes:
- [x] **Splitter.Box.__ranges** - Fixed `.ptp()` call to work with BoundingBox objects (`panobbgo/analyzers/splitter.py:215-220`)
- [x] **memoize decorator** - Added handling for unhashable NumPy arrays by converting to bytes (`panobbgo/utils.py:205-230`)
- [x] **Analyzer name consistency** - Changed "splitter"/"best" to "Splitter"/"Best" (Random, WeightedAverage heuristics)
- [x] **Random heuristic initialization** - Added logic to get root leaf from Splitter on start (`panobbgo/heuristics/random.py:38-48`)
