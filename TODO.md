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

### High Priority - Core Robustness
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
