# benchmarks/

Micro-benchmarks and comparison scripts, separate from the composite-score harness (`benchmark_harness.py`, see `doc/source/guide_benchmarking.rst`).

- `test_benchmarks.py` — pytest-benchmark suite run by CI (`uv run pytest benchmarks/ --benchmark-min-rounds=1 --benchmark-max-time=0.1 -q`); `problems.py` and `strategies.py` define its problem battery and strategy configurations.
- `bench_gp_fit.py`, `bench_results_add.py` — standalone timing scripts for GP fitting and `Results` insertion.
- `run_bandit_comparison.py`, `run_constrained_benchmark.py`, `coco_benchmark.py` — comparison drivers (bandit strategies, constraint handlers, COCO/BBOB); `results_constrained/` holds saved plots and a CSV summary from the constrained run.
