# GitHub Actions CI Workflows

This directory contains the GitHub Actions workflows for continuous integration and testing of the Panobbgo project.

## Overview

The CI pipeline consists of multiple jobs that run in parallel to ensure code quality, functionality, and performance:

- **test**: Runs the full pytest test suite with coverage reporting
- **lint**: Code quality checks with flake8
- **typecheck**: Type checking with Pyright
- **format**: Code formatting validation with ruff
- **benchmark**: Performance benchmarking with pytest-benchmark

## CI Optimization Strategy

### Environment Setup Optimization

**Problem**: Each CI job was performing identical setup work:
- Installing UV package manager
- Installing Python dependencies
- Setting up the environment

This resulted in ~30-60 seconds of redundant work per job.

**Solution**: Strategic caching with conditional execution:

```yaml
- name: Cache UV and dependencies
  id: cache-uv
  uses: actions/cache@v4
  with:
    path: |
      ~/.cargo/bin/uv
      .venv
    key: uv-${{ runner.os }}-python-${{ env.PYTHON }}-${{ hashFiles('pyproject.toml', 'uv.lock') }}
    restore-keys: |
      uv-${{ runner.os }}-python-${{ env.PYTHON }}-

- name: Install UV
  if: steps.cache-uv.outputs.cache-hit != 'true'
  run: |
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "$HOME/.cargo/bin" >> $GITHUB_PATH

- name: Install dependencies
  if: steps.cache-uv.outputs.cache-hit != 'true'
  run: |
    uv sync --extra dev
```

**Benefits**:
- First run: Full setup (cache miss)
- Subsequent runs: Instant cache restoration (cache hit)
- ~50-70% reduction in setup time
- Consistent environment across runs

### Python Version Management

**Problem**: Python version scattered across workflow files, hard to maintain.

**Solution**: Centralized environment variable:

```yaml
env:
  PYTHON: "3.12"

# Used throughout:
python-version: ${{ env.PYTHON }}
```

**Benefits**:
- Single place to change Python version
- Consistent across all jobs
- Easy version upgrades

### Job Parallelization

All CI jobs run in parallel (no dependencies), maximizing concurrency:

```
[test]     [lint]     [typecheck]     [format]     [benchmark]
   |         |            |            |            |
   +---------+------------+------------+------------+
                    Parallel Execution
```

**Benefits**:
- Fastest possible execution time
- Independent failure isolation
- Optimal resource utilization

## Workflow Files

### `tests.yml`
Main CI workflow triggered on pushes and PRs to `main`/`master` branches.

**Jobs**:
- `test`: Core functionality testing
- `lint`: Code quality enforcement
- `typecheck`: Type safety validation
- `format`: Code style consistency
- `benchmark`: Performance regression detection

**Configuration**:
- Python 3.12 (configurable via `env.PYTHON`)
- Ubuntu latest runners
- Comprehensive test matrix (previously included multiple Python versions)

## Alternative: Reusable Workflows (Not Currently Used)

For future consideration, reusable workflows could provide even better optimization:

```yaml
# .github/workflows/setup.yml (reusable)
jobs:
  setup:
    runs-on: ubuntu-latest
    outputs:
      cache-hit: ${{ steps.cache-uv.outputs.cache-hit }}
    steps:
      # Setup steps...

# .github/workflows/tests.yml
jobs:
  setup:
    uses: ./.github/workflows/setup.yml
    with:
      python-version: ${{ env.PYTHON }}

  test:
    needs: setup
    # Test steps...
```

**Pros**: Single setup job, shared across workflows
**Cons**: More complex, requires careful output handling

## Performance Metrics

**Typical CI Run Times** (with caching):
- `lint`: 10-15 seconds
- `typecheck`: 15-20 seconds
- `format`: 10-15 seconds
- `benchmark`: 10-15 seconds
- `test`: 15-25 seconds

**Total**: ~60-90 seconds (parallel execution)

## Maintenance

### Changing Python Version
1. Update `env.PYTHON` in `tests.yml`
2. Commit and push
3. CI will use new version on next run

### Adding New Jobs
1. Copy existing job structure
2. Add to `jobs` section
3. Include caching for consistency
4. Update this README

### Troubleshooting Cache Issues
If cache becomes corrupted:
1. Go to GitHub Actions → Caches
2. Delete problematic cache entries
3. Next CI run will recreate cache

## Dependencies

- **UV**: Fast Python package manager
- **pytest**: Testing framework
- **pytest-benchmark**: Performance testing
- **flake8**: Code linting
- **ruff**: Code formatting
- **pyright**: Type checking
- **codecov**: Coverage reporting

## Security Notes

- All third-party actions pinned to specific versions (e.g., `actions/cache@v4`)
- Cache keys include dependency hash for security
- No sensitive data in workflow files
- Public repository with standard CI permissions