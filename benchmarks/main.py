import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark_runner import BenchmarkRunner
from .results_analyzer import ResultsAnalyzer
from .plots import BenchmarkPlotter


def run_benchmarks(results_dir="results", random_state=42):
    print("=" * 70)
    print("ADE-FCM Benchmark Suite")
    print("=" * 70)

    runner = BenchmarkRunner(random_state=random_state, results_dir=results_dir)
    analyzer = ResultsAnalyzer(results_dir=results_dir)
    plotter = BenchmarkPlotter(output_dir=Path(results_dir) / "plots")

    print("\n[1/4] Loading algorithms...")
    print(f"  Registered: {', '.join(runner.algorithms.keys())}")

    print("\n[2/4] Loading datasets...")
    for ds_name, (df_data, n_classes) in runner.datasets.items():
        print(f"  {ds_name}: {df_data.shape[0]} samples, "
              f"{df_data.shape[1] - 1} features, {n_classes} classes")

    print("\n[3/4] Running benchmarks...")
    results_df = runner.run_all()
    print(f"  Completed {len(results_df)} runs across "
          f"{results_df['algorithm'].nunique()} algorithms and "
          f"{results_df['dataset'].nunique()} datasets")

    error_count = results_df.get("error", pd.Series()).notna().sum()
    if error_count:
        print(f"  WARNING: {error_count} runs had errors")

    print("\n[4/4] Generating outputs...")

    print("\n  --- Comparison Table ---")
    comparison = runner.compare_algorithms()
    print(comparison.to_string())

    print("\n  --- Rankings (by Silhouette) ---")
    try:
        ranking = analyzer.rank_algorithms(results_df, "silhouette_score", ascending=False)
        print(ranking.to_string())
    except Exception as e:
        print(f"  (Skipped: {e})")

    print("\n  --- Statistical Significance ---")
    try:
        sig = analyzer.statistical_significance(
            results_df, "silhouette_score", method="wilcoxon"
        )
        significant = sig[sig["significant"]].index.tolist()
        print(f"  Significant pairs (p<0.05): {len(significant)}")
        for pair in significant:
            print(f"    {pair}: p={sig.loc[pair, 'p_value']:.6f}, "
                  f"better={sig.loc[pair, 'better']}")
    except Exception as e:
        print(f"  (Skipped: {e})")

    print("\n  --- LaTeX Table ---")
    try:
        latex = analyzer.generate_latex_table(comparison)
        latex_path = Path(results_dir) / "comparison_table.tex"
        latex_path.write_text(latex, encoding="utf-8")
        print(f"  Saved to {latex_path}")
        print(latex[:500] + "...")
    except Exception as e:
        print(f"  (Skipped: {e})")

    print("\n  --- Markdown Report ---")
    try:
        report_path = Path(results_dir) / "benchmark_report.md"
        analyzer.generate_report(results_df, output_path=str(report_path))
        print(f"  Saved to {report_path}")
    except Exception as e:
        print(f"  (Skipped: {e})")

    print("\n  --- Generating Plots ---")
    plot_paths = plotter.generate_all_plots(results_df)
    for name, path in plot_paths.items():
        if path and path.exists():
            print(f"  {name}: {path}")

    print("\n" + "=" * 70)
    print("Benchmark complete. Results saved to:", Path(results_dir).resolve())
    print("=" * 70)

    return results_df


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    return run_benchmarks(results_dir=results_dir)


if __name__ == "__main__":
    main()
