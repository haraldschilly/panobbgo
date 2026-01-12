#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
Panobbgo Benchmark Runner
========================

Comprehensive benchmarking script for evaluating optimization algorithms
on standard test problems with validation against known optima.
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panobbgo.benchmark import (
    BenchmarkSuite, create_standard_problems, create_standard_strategies
)


def main():
    parser = argparse.ArgumentParser(description="Run Panobbgo benchmarks")
    parser.add_argument("--problems", nargs="+",
                       help="Specific problems to run (default: all standard problems)")
    parser.add_argument("--strategies", nargs="+",
                       help="Specific strategies to run (default: all standard strategies)")
    parser.add_argument("--repetitions", type=int, default=3,
                       help="Number of repetitions per problem-strategy combination")
    parser.add_argument("--max-evaluations", type=int,
                       help="Maximum evaluations per run (overrides problem defaults)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Directory to save results")
    parser.add_argument("--output-format", choices=["csv", "json", "both"], default="both",
                       help="Output format for results")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark with reduced evaluations")

    args = parser.parse_args()

    print("🚀 Starting Panobbgo Benchmark Suite")
    print("=" * 50)

    # Create benchmark suite
    suite = BenchmarkSuite("panobbgo_comprehensive_benchmark")

    # Add problems
    all_problems = create_standard_problems()
    if args.problems:
        problems_to_run = [p for p in all_problems if p.name in args.problems]
        if not problems_to_run:
            print(f"❌ No matching problems found. Available: {[p.name for p in all_problems]}")
            return 1
    else:
        problems_to_run = all_problems

    for problem in problems_to_run:
        suite.add_problem(problem)
        print(f"📋 Added problem: {problem.name}")

    # Add strategies
    all_strategies = create_standard_strategies()
    if args.strategies:
        strategies_to_run = [s for s in all_strategies if s.name in args.strategies]
        if not strategies_to_run:
            print(f"❌ No matching strategies found. Available: {[s.name for s in all_strategies]}")
            return 1
    else:
        strategies_to_run = all_strategies

    for strategy in strategies_to_run:
        suite.add_strategy(strategy)
        print(f"🎯 Added strategy: {strategy.name}")

    # Configure for quick run if requested
    max_evals = args.max_evaluations
    if args.quick:
        max_evals = max_evals or 100  # More evaluations for meaningful results
        args.repetitions = 1  # Single repetition for quick runs
        print("⚡ Running quick benchmark (reduced evaluations)")

    print("\n🔬 Benchmark Configuration:")
    print(f"  Problems: {len(problems_to_run)}")
    print(f"  Strategies: {len(strategies_to_run)}")
    print(f"  Total combinations: {len(problems_to_run) * len(strategies_to_run)}")
    print(f"  Repetitions: {args.repetitions}")
    print(f"  Max evaluations: {max_evals or 'problem defaults'}")
    print(f"  Total runs: {len(problems_to_run) * len(strategies_to_run) * args.repetitions}")

    # Run benchmarks
    print("\n🏃 Running benchmarks...")
    try:
        runs = suite.run_all(repetitions=args.repetitions, max_evaluations=max_evals)
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        runs = suite.runs

    # Print summary
    suite.print_summary()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.output_format in ["csv", "both"]:
        csv_path = output_dir / "benchmark_results.csv"
        df = suite.get_summary_dataframe()
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to: {csv_path}")

    if args.output_format in ["json", "both"]:
        json_path = output_dir / "benchmark_results.json"

        # Convert runs to serializable format
        results_data = []
        for run in runs:
            run_data = {
                'problem': run.problem_spec.name,
                'strategy': run.strategy_spec.name,
                'run_id': run.run_id,
                'duration': run.duration,
                'success': run.success,
                'error': run.error,
            }

            if run.best_result:
                run_data.update({
                    'best_fx': run.best_result.fx,
                    'best_x': list(run.best_result.x) if run.best_result.x is not None else None,
                })

            if run.validation:
                run_data['validation'] = run.validation

            results_data.append(run_data)

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        serializable_data = [convert_numpy_types(run_data) for run_data in results_data]

        with open(json_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

        print(f"💾 Detailed results saved to: {json_path}")

    # Final statistics
    successful_runs = [r for r in runs if r.success]
    if successful_runs:
        avg_duration = sum(r.duration for r in successful_runs) / len(successful_runs)
        success_rate = len(successful_runs) / len(runs) * 100

        print("\n📊 Final Statistics:")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Average duration: {avg_duration:.2f}s per successful run")
        print(f"  Total successful runs: {len(successful_runs)}/{len(runs)}")
    else:
        print("\n⚠️  No successful runs completed")

    print("\n✅ Benchmark completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())