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

## Framework Completion Tasks

### 🎯 Coverage Goal: 75% (Revised - Quality First)
**Current**: 58% coverage with 96 tests
**Revised Goal**: 75% coverage, but only after validating core component correctness

### 🚨 CRITICAL: Code Quality Assessment Needed
**Finding**: Current tests may be covering buggy implementations rather than robust code.
- Dedensifyer analyzer has clear bugs (undefined variables, wrong constructors)
- Heuristics fail basic functionality tests (point generation)
- Full optimization loops may hang/crash
- Integration tests validate "doesn't crash" but not "actually optimizes"

**Action Required**: Before increasing coverage to 75%, validate that core components actually work.

### High Priority - Code Quality & Bug Fixes
- [ ] **URGENT**: Fix Dedensifyer analyzer bugs (undefined box_dims, wrong constructor calls)
- [ ] **URGENT**: Validate heuristic point generation works correctly
- [ ] **URGENT**: Fix optimization loop hangs/crashes
- [ ] Complete test coverage for UCB strategy implementation
- [ ] Add tests for convergence analyzer edge cases and event handling
- [ ] Add tests for grid and dedensifyer analyzers (currently 56% and 29% coverage)
- [ ] Add tests for remaining low-coverage heuristics (lbfgsb 30%, nelder_mead 51%, etc.)

### Medium Priority - Integration & Validation
- [ ] Add more integration tests for end-to-end optimization scenarios with different strategies
- [ ] Improve error handling and edge case testing for all components
- [ ] Add tests for constraint handling integration with different strategies
- [ ] Add validation for framework configuration and component compatibility

### Low Priority - Documentation & Polish
- [ ] Update documentation references from IPython parallel to Dask throughout codebase
- [ ] Review and fix minor naming inconsistencies in guide documentation (e.g., QuadraticWlsModel naming)
- [ ] Remove remaining IPython parallel references from code and documentation, replace with Dask references
- [ ] Review and potentially simplify UI components or ensure they work with current framework
- [ ] Add performance benchmarks comparing different strategies on benchmark problems
- [ ] Review and optimize threading/event handling for better performance
