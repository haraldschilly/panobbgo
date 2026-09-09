# Next: Missing Modular Building Blocks

These are the three highest-priority gaps identified in the panobbgo framework,
beyond heuristics (which are already well-covered with 15 implementations).

---

## 1. Sensitivity Analyzer

**What**: An analyzer that estimates which input dimensions have the most impact
on the objective function, using the accumulated evaluation history.

**Why**: In high-dimensional problems (dim > 10), many dimensions may be irrelevant.
If heuristics knew which dimensions matter, they could focus their search and converge faster.
Currently no component provides this information.

**Where it fits**: New file `panobbgo/analyzers/sensitivity.py`.
Analyzers inherit from `Analyzer` (in `panobbgo/core.py`), subscribe to events via `on_*` methods,
and publish derived events. Register in `panobbgo/analyzers/__init__.py`.

**Algorithm — Morris-like screening from evaluation history**:

1. Subscribe to `on_new_results`.
2. Accumulate X (n_samples × dim) and y (penalty values) from results.
3. After a configurable `min_samples` (default: 10 × dim), compute sensitivity:
   - For each dimension i, compute the partial correlation between x_i and y,
     controlling for other dimensions (using residuals from linear regression).
   - Alternatively, simpler: compute Spearman rank correlation |corr(x_i, y)| per dimension.
   - Normalize to [0, 1] range → `importance[i]`.
4. Publish `new_sensitivity(importance=np.ndarray)` event.
5. Recompute periodically (every `update_interval` new results, default: 50).

**Events published**:
- `new_sensitivity(importance=ndarray)` — array of shape (dim,), values in [0, 1], higher = more important.

**How heuristics could use it** (future work, not part of this PR):
- ClaudeHeuristic could weight covariance by importance (shrink unimportant dims).
- Nearby could perturb only important dimensions.
- GP could use ARD (Automatic Relevance Determination) kernel initialized from importance.

**Parameters**:
- `min_samples` (int, default=None → 10*dim): minimum evaluations before first analysis.
- `update_interval` (int, default=50): recompute every N new results.
- `method` (str, default="spearman"): "spearman" (rank correlation) or "partial" (partial correlation).

**Key code patterns to follow**:
- `panobbgo/analyzers/convergence.py` — similar structure: subscribes to `on_new_results`,
  accumulates history, publishes events when thresholds are met.
- `panobbgo/analyzers/__init__.py` — add import + `__all__` entry.
- Use `self.strategy.constraint_handler.get_penalty_value(r)` for the target variable,
  same as ClaudeHeuristic and GP do.

**Test file**: `tests/test_analyzer_sensitivity.py`
- Test that no event fires before min_samples.
- Test with a problem where dim 0 matters and dim 1 doesn't (e.g., f(x) = x[0]^2).
  Verify importance[0] >> importance[1].
- Test update_interval triggers recomputation.
- Follow mock pattern from `tests/test_heuristic_feasible.py` (MockProblem, mock strategy).

**Docs to update**:
- `doc/source/analyzers.rst` — add entry.
- `doc/source/guide_architecture.rst` — add to "Implemented Analyzers" section.

---

## 2. Restart Analyzer

**What**: An analyzer that detects when the optimizer is stuck in a local optimum
and publishes a `restart` event, allowing heuristics and the strategy to reset
their search to a different region — without losing accumulated results.

**Why**: The current `Convergence` analyzer only detects when to *stop*.
For multimodal problems, getting stuck in a local basin is inevitable.
A restart mechanism is the standard solution (multi-start optimization),
but panobbgo has no way to trigger it.

**Where it fits**: New file `panobbgo/analyzers/restart.py`.
Register in `panobbgo/analyzers/__init__.py`.

**Algorithm**:

1. Subscribe to `on_new_results`.
2. Track a sliding window of the best penalty value seen (like Convergence does).
3. If no improvement greater than `improvement_threshold` in the last `patience` evaluations:
   - Generate a new random starting region (or pick from under-explored areas using the Splitter).
   - Publish `restart(center=ndarray, reason=str)` event.
   - Reset the patience counter.
4. Track number of restarts; stop publishing after `max_restarts`.

**Events published**:
- `restart(center=ndarray, reason=str)` — suggested new center point for search, plus reason string.

**How heuristics should respond** (convention to document):
- Heuristics with `on_restart(center)` should:
  - `clear_output()` their queue.
  - Reset their internal model/state.
  - Begin generating points around the new `center`.
- Heuristics without `on_restart` simply continue as before (graceful degradation).
- Random heuristic: could switch leaf to the box containing `center`.
- NelderMead: reset `got_bb` flag, wait for new best box.
- GP: clear training data, start fresh.
- ClaudeHeuristic: clear accumulated X_all/y_all, rebuild from scratch.

**Parameters**:
- `patience` (int, default=None → 5*dim): evaluations without improvement before restart.
- `improvement_threshold` (float, default=1e-6): minimum relative improvement to reset counter.
- `max_restarts` (int, default=10): stop restarting after this many.
- `strategy` (str, default="random"): how to pick new center — "random" (random_point) or
  "diverse" (maximize distance from previous restart centers).

**Key code patterns to follow**:
- `panobbgo/analyzers/convergence.py` — very similar sliding-window logic. That analyzer tracks
  `_history` deque, checks std or improvement, publishes `converged`. The restart analyzer
  would track `_best_in_window` and check for stagnation, but publish `restart` instead.
- The `Convergence` analyzer (convergence.py lines ~30-130) has configurable `window`, `threshold`,
  and `mode`. Mirror this pattern.

**Important interaction with Convergence**: If both Convergence and Restart are active,
Restart should fire *before* Convergence declares the optimization done.
The Restart analyzer should have a lower patience than Convergence's window.
Document this in the docstring.

**Test file**: `tests/test_analyzer_restart.py`
- Test that restart fires after `patience` evaluations with no improvement.
- Test that restart does NOT fire if improvements keep coming.
- Test max_restarts limit.
- Test that `center` is a valid point in the box.

**Docs to update**:
- `doc/source/analyzers.rst`
- `doc/source/guide_architecture.rst`
- `doc/source/guide_usage.rst` — add a "Multi-start Optimization" section.

---

## 3. Problem Wrappers (Composable Transforms)

**What**: Decorator classes that wrap a `Problem` instance and transform inputs/outputs,
without modifying the original problem. Composable: `NormalizedProblem(LogTransformProblem(MyProblem()))`.

**Why**: Many heuristics implicitly assume similar scales across dimensions.
Problems with wildly different ranges (e.g., [0, 1] × [0, 10000]) perform poorly.
Currently there's no way to normalize without modifying each Problem subclass.
Also useful: logging, noise injection, eval counting.

**Where it fits**: New file `panobbgo/lib/wrappers.py`.
These are `Problem` subclasses that delegate to a wrapped problem.

**Base pattern**:

```python
class ProblemWrapper(Problem):
    """Base class for problem wrappers. Delegates to wrapped problem."""

    def __init__(self, problem: Problem):
        # Copy box from wrapped problem (subclasses may transform it)
        self._wrapped = problem
        super().__init__(box=problem.box_tuples)  # need to check exact API

    def eval(self, x):
        return self._wrapped.eval(x)

    def eval_constraints(self, x):
        return self._wrapped.eval_constraints(x)

    # Delegate other properties
    @property
    def dim(self):
        return self._wrapped.dim
```

**Three concrete wrappers**:

### 3a. NormalizedProblem

Scales all dimensions to [0, 1]. Heuristics see a unit hypercube;
the wrapper maps back to original coordinates for evaluation.

```
x_original = x_normalized * ranges + lower_bounds
```

- `__init__(problem)`: compute ranges and offsets from `problem.box`.
- `eval(x_normalized)`: denormalize → call `_wrapped.eval(x_original)`.
- `eval_constraints(x_normalized)`: denormalize → call `_wrapped.eval_constraints(x_original)`.
- Box becomes `[(0, 1)] * dim`.

### 3b. LogTransformProblem

Applies log transform to the objective: `log(1 + f(x) - f_offset)`.
Useful when objective spans orders of magnitude. The penalty/reward calculations
become more balanced.

- `__init__(problem, offset=0.0)`: store offset.
- `eval(x)`: `return np.log1p(self._wrapped.eval(x) - self.offset)`.
- Constraints are NOT transformed (they have their own scale via cv).

### 3c. NoisyProblem

Adds controlled Gaussian noise to evaluations. Useful for robustness testing.

- `__init__(problem, noise_std=0.1, noise_type="additive")`.
- `eval(x)`: `return self._wrapped.eval(x) + noise_std * np.random.randn()`.
- `noise_type="multiplicative"`: `return self._wrapped.eval(x) * (1 + noise_std * np.random.randn())`.

**Key challenge — the Problem API**:
- `Problem.__init__` in `panobbgo/lib/lib.py` takes `box` (list of tuples) and computes
  `self.dim`, `self.box` (BoundingBox), `self.ranges`, etc.
- The wrapper must produce a valid Problem that the rest of the framework accepts.
- Need to check: does `Problem.__init__` do anything besides store box/dim?
  Read `panobbgo/lib/lib.py` lines ~1-80 to confirm.
- The wrapper's `eval()` receives the *transformed* x and must map back.

**Test file**: `tests/test_problem_wrappers.py`
- NormalizedProblem: verify box is [0,1]^dim, verify eval at [0.5, 0.5, ...] equals
  original eval at midpoint, verify round-trip.
- LogTransformProblem: verify log1p transform on known values.
- NoisyProblem: verify mean over many evals ≈ true value, verify std ≈ noise_std.
- Composition: `NormalizedProblem(NoisyProblem(Rosenbrock(dims=2)))` should work.

**Docs to update**:
- `doc/source/guide_usage.rst` — add "Problem Wrappers" section with examples.
- `doc/source/lib.rst` — add automodule for wrappers.

---

## Implementation Order

Recommended sequence (each as a separate branch + PR off master):

1. **Problem Wrappers** — smallest scope, no event system changes, foundational for testing the others.
   Branch: `feat/problem-wrappers`

2. **Sensitivity Analyzer** — self-contained analyzer, publishes new event type.
   Branch: `feat/sensitivity-analyzer`

3. **Restart Analyzer** — depends on understanding how heuristics should respond;
   may want to add `on_restart` handlers to existing heuristics in a follow-up PR.
   Branch: `feat/restart-analyzer`

## Key Files Reference

| File | Role | Relevant for |
|------|------|-------------|
| `panobbgo/core.py:545-644` | `Heuristic` base class (emit, clear_output, get_points) | All |
| `panobbgo/core.py:455-540` | `Analyzer` base class | Analyzers |
| `panobbgo/core.py:716-927` | `EventBus` (register, publish, subscribe) | Analyzers |
| `panobbgo/analyzers/convergence.py` | Reference pattern for sliding-window analyzer | Restart, Sensitivity |
| `panobbgo/analyzers/best.py` | Reference pattern for event-publishing analyzer | All analyzers |
| `panobbgo/analyzers/__init__.py` | Registration point for analyzers | Analyzers |
| `panobbgo/lib/lib.py:1-100` | `Problem`, `BoundingBox`, `Point`, `Result` | Wrappers |
| `panobbgo/lib/constraints.py` | `ConstraintHandler.get_penalty_value()` | Sensitivity |
| `panobbgo/heuristics/claude_heuristic.py` | Reference for accumulation + penalty pattern | Sensitivity |
| `tests/test_heuristic_feasible.py` | Reference for mock setup (MockProblem, Config) | All tests |

## Verification (for each PR)

```bash
uv run ruff check <new_files>
uv run pyright <new_files>
uv run pytest tests/<new_test_file> -v
uv run pytest tests/ -x  # full suite
```
