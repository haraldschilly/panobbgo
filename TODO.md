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
**Order**: Optimization hangs → Heuristics → Dedensifyer
- [ ] **Optimization Loop Stability** - Fix hangs/crashes in full runs
  - [ ] Write TDD tests for loop termination behavior
  - [ ] Identify root causes (threading, infinite loops, missing timeouts)
  - [ ] Implement fixes and validate with tests
- [ ] **Heuristic Functionality** - Fix point generation failures
  - [ ] Write TDD tests requiring valid point generation from Random/Nearby
  - [ ] Debug event handling and queue management issues
  - [ ] Validate all heuristics work in framework context
- [ ] **Dedensifyer Analyzer** - Fix critical implementation bugs
  - [ ] Write TDD tests for proper initialization and grid management
  - [ ] Fix constructor (missing strategy parameter)
  - [ ] Fix undefined variables and wrong method signatures
  - [ ] Validate hierarchical grid functionality
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
