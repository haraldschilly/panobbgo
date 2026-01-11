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
- [ ] **Optimization Correctness Validation** - Add tests proving algorithms work
  - [ ] Write tests validating convergence to known optima
  - [ ] Compare optimization vs random baseline performance
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

### 🎯 TARGET: 75% Coverage on Validated Components
**Prerequisites**: All Priority 1 items completed with TDD validation
**Quality Metrics**: Correctness + Coverage (not just coverage)
**Blocker**: Strategy.start() hang bug must be fixed before proper validation testing

## Known Issues & Technical Debt

### Strategy.start() Hang Bug (PR #36 - Pre-existing Critical Bug)
**CRITICAL**: `strategy.start()` doesn't return after reaching `max_eval` evaluations
- **Affects**: All validation tests in `tests/test_validation.py`
- **Root Cause**: `_run()` method in `panobbgo/core.py` has termination logic bugs
  - Loop should break when `len(self.results) >= self.config.max_eval` (line ~1233)
  - Something prevents this termination condition from being reached
  - Even simple tests with Center heuristic hang (not just Random heuristic issue)
  - Happens with `evaluation_method="threaded"` (not just Dask)
- **Status**: Pre-existing on master (tested commit 677f54b), not introduced by PR #36
- **Current Workaround**: Skip all validation tests with `@pytest.mark.skip`
- **Impact**: Cannot run end-to-end validation tests, limits framework testing capability
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
