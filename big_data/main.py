import time
import gc
import numpy as np
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

from .large_scale_fcm import LargeScaleFCM


def generate_synthetic_data(n_samples, n_features=8, n_clusters=5, random_state=42):
    rng = np.random.RandomState(random_state)
    samples_per_cluster = n_samples // n_clusters
    X_list = []
    for i in range(n_clusters):
        center = rng.uniform(-10, 10, size=n_features)
        cov = rng.uniform(0.1, 0.5, size=n_features)
        cov_matrix = np.diag(cov)
        cluster_X = rng.multivariate_normal(
            center, cov_matrix, size=samples_per_cluster
        )
        X_list.append(cluster_X)

    remainder = n_samples - samples_per_cluster * n_clusters
    if remainder > 0:
        X_list.append(
            rng.uniform(-10, 10, size=(remainder, n_features))
        )

    X = np.vstack(X_list)
    rng.shuffle(X)
    return X


def generate_dataframe(spark, X):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    schema_fields = []
    for j in range(n_features):
        schema_fields.append(StructField(f"f{j}", DoubleType(), False))
    schema_fields.append(StructField("id", IntegerType(), False))
    schema = StructType(schema_fields)

    rows = []
    for i in range(n_samples):
        row = [float(X[i, j]) for j in range(n_features)] + [i]
        rows.append(tuple(row))

    df = spark.createDataFrame(rows, schema=schema)
    return df


def benchmark_method(
    method_name,
    fcm,
    X,
    spark=None,
    df=None,
    n_clusters=5,
    max_iter=20,
):
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {method_name}")
    logger.info(f"{'='*60}")

    gc.collect()
    start_time = time.time()
    start_mem = __import__("psutil").Process().memory_info().rss / 1024 ** 3

    try:
        if method_name == "sequential":
            centers, U, J_hist = fcm.fit_sequential(X)
        elif method_name == "rdd":
            centers, U, J_hist = fcm.fit_rdd(X)
        elif method_name == "dataframe":
            centers, U, J_hist = fcm.fit_dataframe(df)
        elif method_name == "sql":
            centers, U, J_hist = fcm.fit_sql(df)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        elapsed = time.time() - start_time
        end_mem = __import__("psutil").Process().memory_info().rss / 1024 ** 3
        mem_used = max(0, end_mem - start_mem)
        n_iter = len(J_hist)
        final_J = J_hist[-1] if J_hist else None

        result = {
            "method": method_name,
            "samples": X.shape[0],
            "features": X.shape[1],
            "clusters": n_clusters,
            "max_iter": max_iter,
            "time_seconds": round(elapsed, 3),
            "memory_gb": round(mem_used, 4),
            "iterations": n_iter,
            "converged": fcm._converged,
            "final_objective": round(final_J, 4) if final_J else None,
        }
        logger.info(f"{method_name.upper():>12s}: {elapsed:8.3f}s, "
                     f"{mem_used:.4f} GB, {n_iter} iters, "
                     f"converged={fcm._converged}")
        return result

    except Exception as e:
        logger.error(f"{method_name} failed: {e}")
        return {
            "method": method_name,
            "samples": X.shape[0],
            "features": X.shape[1],
            "clusters": n_clusters,
            "max_iter": max_iter,
            "time_seconds": None,
            "memory_gb": None,
            "iterations": None,
            "converged": None,
            "final_objective": None,
            "error": str(e),
        }


def run_single_benchmark(spark, n_samples, n_features, n_clusters, max_iter, chunk_size):
    logger.info(f"\n{'#'*60}")
    logger.info(f"Benchmark: n_samples={n_samples}, n_features={n_features}, "
                 f"n_clusters={n_clusters}, max_iter={max_iter}")
    logger.info(f"{'#'*60}")

    X = generate_synthetic_data(n_samples, n_features, n_clusters)
    df = generate_dataframe(spark, X)

    fcm = LargeScaleFCM(
        spark_session=spark,
        n_clusters=n_clusters,
        max_iter=max_iter,
        m=2.0,
        epsilon=1e-4,
        chunk_size=chunk_size,
        verbose=False,
    )

    results = []
    methods = ["sequential", "rdd", "dataframe", "sql"]
    for method in methods:
        result = benchmark_method(method, fcm, X, spark, df, n_clusters, max_iter)
        results.append(result)
        gc.collect()

    results_df = pd.DataFrame(results)
    logger.info(f"\nResults for {n_samples} samples:")
    print(results_df.to_string(index=False))
    return results_df


def run_full_benchmark():
    import psutil

    logger.info("Starting Big Data ADE-FCM Benchmark Pipeline")
    logger.info(f"System: {psutil.cpu_count()} cores, "
                 f"{psutil.virtual_memory().total / 1024**3:.1f} GB RAM")

    spark = (
        SparkSession.builder.appName("ADE-FCM-BigData-Benchmark")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.minExecutors", "1")
        .config("spark.dynamicAllocation.maxExecutors", "4")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "2g")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"Spark UI: {spark.sparkContext.uiWebUrl}")

    configs = [
        {"n_samples": 100000,  "n_features": 4, "n_clusters": 5, "max_iter": 15, "chunk_size": 25000},
        {"n_samples": 500000,  "n_features": 4, "n_clusters": 5, "max_iter": 15, "chunk_size": 50000},
        {"n_samples": 1000000, "n_features": 4, "n_clusters": 5, "max_iter": 10, "chunk_size": 100000},
        {"n_samples": 5000000, "n_features": 4, "n_clusters": 5, "max_iter": 5,  "chunk_size": 250000},
    ]

    all_results = []
    for cfg in configs:
        res = run_single_benchmark(
            spark,
            cfg["n_samples"],
            cfg["n_features"],
            cfg["n_clusters"],
            cfg["max_iter"],
            cfg["chunk_size"],
        )
        all_results.append(res)

    comparison = pd.concat(all_results, ignore_index=True)
    final_table = comparison.pivot_table(
        index="method",
        columns="samples",
        values="time_seconds",
        aggfunc="first",
    )
    logger.info("\n" + "=" * 80)
    logger.info("SCALABILITY COMPARISON TABLE (execution time in seconds)")
    logger.info("=" * 80)
    print("\nExecution Time (seconds):")
    print(final_table.to_string(float_format="%.3f"))

    memory_table = comparison.pivot_table(
        index="method",
        columns="samples",
        values="memory_gb",
        aggfunc="first",
    )
    print("\nMemory Usage (GB):")
    print(memory_table.to_string(float_format="%.4f"))

    try:
        _generate_scalability_plots(comparison)
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")

    spark.stop()
    logger.info("Benchmark pipeline complete")
    return comparison


def _generate_scalability_plots(results_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = results_df["method"].unique()
    sizes = sorted(results_df["samples"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for method in methods:
        mask = results_df["method"] == method
        subset = results_df[mask].sort_values("samples")
        axes[0].plot(
            subset["samples"],
            subset["time_seconds"],
            marker="o",
            label=method.upper(),
            linewidth=2,
        )
        axes[1].plot(
            subset["samples"],
            subset["memory_gb"],
            marker="s",
            label=method.upper(),
            linewidth=2,
        )

    axes[0].set_xlabel("Number of Samples")
    axes[0].set_ylabel("Execution Time (seconds)")
    axes[0].set_title("Scalability: Execution Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")

    axes[1].set_xlabel("Number of Samples")
    axes[1].set_ylabel("Memory Usage (GB)")
    axes[1].set_title("Scalability: Memory Usage")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale("log")

    plt.tight_layout()
    plt.savefig("big_data_scalability.png", dpi=150)
    logger.info("Saved scalability plot to big_data_scalability.png")

    speedup = _compute_speedup(results_df)
    if speedup is not None:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for method in methods:
            if method == "sequential":
                continue
            mask = speedup["method"] == method
            subset = speedup[mask]
            ax2.plot(
                subset["samples"],
                subset["speedup"],
                marker="^",
                label=method.upper(),
                linewidth=2,
            )

        ax2.set_xlabel("Number of Samples")
        ax2.set_ylabel("Speedup vs Sequential")
        ax2.set_title("Distributed Speedup")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")
        plt.tight_layout()
        plt.savefig("big_data_speedup.png", dpi=150)
        logger.info("Saved speedup plot to big_data_speedup.png")


def _compute_speedup(results_df):
    seq_times = results_df[results_df["method"] == "sequential"][
        ["samples", "time_seconds"]
    ].set_index("samples")["time_seconds"]
    if seq_times.empty:
        return None

    speedup_rows = []
    for _, row in results_df.iterrows():
        if row["method"] == "sequential":
            continue
        base = seq_times.get(row["samples"])
        if base and base > 0 and row["time_seconds"] and row["time_seconds"] > 0:
            speedup_rows.append(
                {
                    "method": row["method"],
                    "samples": row["samples"],
                    "speedup": round(base / row["time_seconds"], 3),
                }
            )

    return pd.DataFrame(speedup_rows)


if __name__ == "__main__":
    run_full_benchmark()
