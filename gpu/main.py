import time
import warnings
import numpy as np

try:
    import cupy as cp
    _CUDA_AVAILABLE = True
except ImportError:
    cp = None
    _CUDA_AVAILABLE = False

try:
    from pyspark.sql import SparkSession
    _SPARK_AVAILABLE = True
except ImportError:
    SparkSession = None
    _SPARK_AVAILABLE = False

from .gpu_fcm import GPUFCMManager
from .rapids_fcm import RAPIDSFCM
from .spark_gpu_hybrid import SparkGPUHybridEngine


def generate_synthetic_data(n_samples, n_features=10, n_clusters=5, noise=0.3, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, n_features) * 3
    labels = rng.choice(n_clusters, n_samples)
    data = centers[labels] + noise * rng.randn(n_samples, n_features)
    return data.astype(np.float64)


def run_benchmark(data_sizes=None, n_features=10, n_clusters=5,
                  max_iter=50, m=2.0, n_runs=2):
    if data_sizes is None:
        data_sizes = [1000, 10000, 100000]

    all_results = []

    for n_samples in data_sizes:
        print(f"\n{'='*60}")
        print(f"Benchmark: n_samples={n_samples}, n_features={n_features}")
        print(f"{'='*60}")

        X = generate_synthetic_data(n_samples, n_features, n_clusters)

        row = {"n_samples": n_samples, "n_features": n_features, "n_clusters": n_clusters}

        # CPU
        try:
            cpu_times = []
            for _ in range(n_runs):
                model = GPUFCMManager(use_gpu=False, n_clusters=n_clusters,
                                      max_iter=max_iter, m=m)
                t0 = time.perf_counter()
                model.fit(X)
                cpu_times.append(time.perf_counter() - t0)
            row["cpu_time"] = np.mean(cpu_times)
            row["cpu_std"] = np.std(cpu_times)
            print(f"  CPU:      {row['cpu_time']:.4f}s")
        except Exception as e:
            row["cpu_time"] = None
            print(f"  CPU:      FAILED ({e})")

        # GPU (CuPy)
        if _CUDA_AVAILABLE:
            try:
                gpu_times = []
                for _ in range(n_runs):
                    model = GPUFCMManager(use_gpu=True, n_clusters=n_clusters,
                                          max_iter=max_iter, m=m)
                    t0 = time.perf_counter()
                    model.fit(X)
                    gpu_times.append(time.perf_counter() - t0)
                row["gpu_time"] = np.mean(gpu_times)
                row["gpu_std"] = np.std(gpu_times)
                if row.get("cpu_time") and row["gpu_time"] > 0:
                    row["gpu_speedup"] = row["cpu_time"] / row["gpu_time"]
                else:
                    row["gpu_speedup"] = None
                print(f"  GPU:      {row['gpu_time']:.4f}s "
                      f"(x{row.get('gpu_speedup', 'N/A')})")
            except Exception as e:
                row["gpu_time"] = None
                row["gpu_speedup"] = None
                print(f"  GPU:      FAILED ({e})")
        else:
            row["gpu_time"] = None
            row["gpu_speedup"] = None
            print(f"  GPU:      SKIPPED (no CUDA)")

        # RAPIDS
        if _CUDA_AVAILABLE:
            try:
                rapids_times = []
                for _ in range(n_runs):
                    model = RAPIDSFCM(n_clusters=n_clusters, max_iter=max_iter, m=m)
                    t0 = time.perf_counter()
                    model.fit_fuzzy(X)
                    rapids_times.append(time.perf_counter() - t0)
                row["rapids_time"] = np.mean(rapids_times)
                row["rapids_std"] = np.std(rapids_times)
                print(f"  RAPIDS:   {row['rapids_time']:.4f}s")
            except Exception as e:
                row["rapids_time"] = None
                print(f"  RAPIDS:   FAILED ({e})")
        else:
            row["rapids_time"] = None

        # Spark modes
        if _SPARK_AVAILABLE:
            for mode_name, mode_key in [("Spark CPU", "spark_cpu"),
                                        ("Spark GPU", "spark_gpu")]:
                try:
                    engine = SparkGPUHybridEngine(
                        n_clusters=n_clusters, max_iter=max_iter, m=m
                    )
                    t0 = time.perf_counter()
                    if mode_key == "spark_cpu":
                        engine.fit_spark_cpu(X)
                    else:
                        if _CUDA_AVAILABLE:
                            engine.fit_spark_gpu(X)
                        else:
                            raise RuntimeError("CUDA not available")
                    elapsed = time.perf_counter() - t0
                    row[mode_key + "_time"] = elapsed
                    print(f"  {mode_name}: {elapsed:.4f}s")
                    engine.stop_spark()
                except Exception as e:
                    row[mode_key + "_time"] = None
                    print(f"  {mode_name}: FAILED ({e})")
        else:
            row["spark_cpu_time"] = None
            row["spark_gpu_time"] = None

        all_results.append(row)

    import pandas as pd
    return pd.DataFrame(all_results)


def plot_results(results_df, output_dir="."):
    try:
        import matplotlib.pyplot as plt
        import os
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    df = results_df.copy()
    modes = [
        ("cpu_time", "CPU"),
        ("gpu_time", "GPU (CuPy)"),
        ("rapids_time", "GPU (RAPIDS)"),
        ("spark_cpu_time", "Spark CPU"),
        ("spark_gpu_time", "Spark GPU"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for col, label in modes:
        if col in df.columns and df[col].notna().any():
            ax.plot(df["n_samples"], df[col], marker="o", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Execution Time vs Dataset Size")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    ax = axes[0, 1]
    if "gpu_speedup" in df.columns and df["gpu_speedup"].notna().any():
        ax.plot(df["n_samples"], df["gpu_speedup"], marker="o", color="green")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Speedup (CPU / GPU)")
    ax.set_title("GPU Speedup vs CPU")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    ax = axes[1, 0]
    cols_present = [(c, l) for c, l in modes if c in df.columns and df[c].notna().any()]
    x = np.arange(len(df))
    width = 0.8 / len(cols_present)
    for i, (col, label) in enumerate(cols_present):
        offset = (i - len(cols_present) / 2 + 0.5) * width
        ax.bar(x + offset, df[col], width * 0.9, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s)}" for s in df["n_samples"]])
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Mode Comparison (bar)")
    ax.legend()
    ax.grid(True, axis="y", ls="--", alpha=0.5)

    ax = axes[1, 1]
    ax.axis("off")
    col_data = []
    for col, label in modes:
        if col in df.columns:
            vals = df[col].tolist()
            col_data.append([f"{v:.4f}" if v is not None else "N/A" for v in vals])
    col_labels = [l for c, l in modes if c in df.columns]
    row_labels = [f"{int(s)}" for s in df["n_samples"]]
    tbl_data = list(zip(*col_data)) if col_data else []
    if tbl_data:
        table = ax.table(cellText=tbl_data, colLabels=col_labels,
                         rowLabels=row_labels, loc="center",
                         cellLoc="center", colColours=["#f2f2f2"] * len(col_labels))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title("Timing Summary (seconds)")

    plt.tight_layout()
    path = os.path.join(output_dir, "gpu_benchmark_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {path}")
    plt.close()


def export_results(results_df, output_dir="."):
    import os
    csv_path = os.path.join(output_dir, "gpu_benchmark_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results exported to: {csv_path}")

    md_path = os.path.join(output_dir, "gpu_benchmark_results.md")
    with open(md_path, "w") as f:
        f.write("# ADE-FCM GPU Benchmark Results\n\n")
        f.write(results_df.to_markdown(index=False) + "\n")
    print(f"Markdown exported to: {md_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ADE-FCM GPU Benchmark Pipeline")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[1000, 10000, 100000],
                        help="Dataset sizes to benchmark")
    parser.add_argument("--features", type=int, default=10,
                        help="Number of features")
    parser.add_argument("--clusters", type=int, default=5,
                        help="Number of clusters")
    parser.add_argument("--max-iter", type=int, default=50,
                        help="Maximum iterations")
    parser.add_argument("--m", type=float, default=2.0,
                        help="Fuzziness exponent")
    parser.add_argument("--runs", type=int, default=2,
                        help="Number of runs per configuration")
    parser.add_argument("--output", type=str, default=".",
                        help="Output directory for plots and CSV")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip exporting results")

    args = parser.parse_args()

    print("=" * 60)
    print("ADE-FCM GPU Benchmark Pipeline")
    print("=" * 60)
    print(f"CUDA available:    {_CUDA_AVAILABLE}")
    print(f"PySpark available: {_SPARK_AVAILABLE}")
    print(f"Dataset sizes:     {args.sizes}")
    print(f"Features:          {args.features}")
    print(f"Clusters:          {args.clusters}")
    print(f"Max iter:          {args.max_iter}")
    print(f"Fuzziness (m):     {args.m}")
    print(f"Runs per config:   {args.runs}")
    print("=" * 60)

    results = run_benchmark(
        data_sizes=args.sizes,
        n_features=args.features,
        n_clusters=args.clusters,
        max_iter=args.max_iter,
        m=args.m,
        n_runs=args.runs,
    )

    print(f"\n{'='*60}")
    print("Summary:")
    print(results.to_string(index=False))

    if not args.no_plot:
        try:
            plot_results(results, args.output)
        except Exception as e:
            print(f"Plotting failed: {e}")

    if not args.no_export:
        export_results(results, args.output)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
