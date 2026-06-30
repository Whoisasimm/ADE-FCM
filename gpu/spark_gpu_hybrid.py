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
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.types import StructType, StructField, DoubleType
    from pyspark.ml.clustering import KMeans as SparkKMeans
    from pyspark.ml.feature import VectorAssembler
    _SPARK_AVAILABLE = True
except ImportError:
    SparkSession = None
    DataFrame = None
    _SPARK_AVAILABLE = False

from .gpu_fcm import GPUFCMManager
from .cuda_kernels import (
    compute_membership_gpu,
    compute_centers_gpu,
    compute_distances_gpu,
    compute_objective_gpu,
)


def _fcm_cpu_single(X, n_clusters, max_iter, m, tol, seed):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.choice(n, n_clusters, replace=False)
    centers = X[idx].copy()
    for _ in range(max_iter):
        d = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        d = np.maximum(d, 1e-15)
        e = -2.0 / (m - 1.0)
        inv = d ** e
        U = inv / np.sum(inv, axis=1, keepdims=True)
        Um = U ** m
        nc = (Um.T @ X) / np.sum(Um, axis=0)[:, None]
        if np.linalg.norm(nc - centers) < tol:
            centers = nc
            break
        centers = nc
    return centers, U


class SparkGPUHybridEngine:
    def __init__(self, spark_session=None, n_clusters=5, max_iter=100, m=2.0,
                 tol=1e-4, seed=42, spark_mode="spark_gpu"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.tol = tol
        self.seed = seed
        self.spark_mode = spark_mode
        self._centers = None
        self._U = None
        self._J_history = []
        self._results = {}

        if spark_session is not None:
            self.spark = spark_session
        elif _SPARK_AVAILABLE:
            self.spark = SparkSession.builder \
                .appName("ADE-FCM-GPU-Hybrid") \
                .config("spark.sql.shuffle.partitions", "8") \
                .config("spark.executor.memory", "4g") \
                .getOrCreate()
        else:
            self.spark = None

        if spark_mode == "spark_gpu" and not _CUDA_AVAILABLE:
            warnings.warn("CUDA not available. Falling back to spark_cpu mode.")
            self.spark_mode = "spark_cpu"

    def _to_spark_df(self, df):
        if isinstance(df, np.ndarray):
            if not _SPARK_AVAILABLE:
                raise RuntimeError("PySpark not available")
            rows = df.tolist()
            n_features = df.shape[1]
            fields = [StructField(f"f{i}", DoubleType(), False) for i in range(n_features)]
            schema = StructType(fields)
            return self.spark.createDataFrame(rows, schema=schema)
        return df

    def _df_to_numpy(self, df):
        if isinstance(df, np.ndarray):
            return df
        return np.array(df.select("*").collect(), dtype=np.float64)

    def fit_spark_gpu(self, df):
        if self.spark is None:
            raise RuntimeError("PySpark session not available")
        self.spark_mode = "spark_gpu"
        t0 = time.perf_counter()

        sdf = self._to_spark_df(df)
        n_features = len(sdf.columns)
        n_partitions = sdf.rdd.getNumPartitions()

        def process_partition(iterator):
            rows = list(iterator)
            if not rows:
                return iter([])
            X_local = np.array(rows, dtype=np.float64)
            X_gpu = cp.asarray(X_local, dtype=cp.float64)

            rng = cp.random.RandomState(self.seed)
            n = X_gpu.shape[0]
            idx = rng.choice(n, min(self.n_clusters, n), replace=False)
            centers = X_gpu[idx].copy()

            for _ in range(self.max_iter):
                d = compute_distances_gpu(X_gpu, centers)
                d = cp.maximum(d, 1e-15)
                U = compute_membership_gpu(d, self.m)
                nc = compute_centers_gpu(X_gpu, U, self.m)
                if cp.linalg.norm(nc - centers) < self.tol:
                    centers = nc
                    break
                centers = nc

            centers_cpu = cp.asnumpy(centers)
            yield (0, centers_cpu.tobytes())

        partition_centers = sdf.rdd.mapPartitions(process_partition).collect()

        all_centers_list = [np.frombuffer(c, dtype=np.float64).reshape(-1, n_features)
                            for _, c in partition_centers]
        if not all_centers_list:
            raise RuntimeError("No partition centers collected")
        all_centers = np.vstack(all_centers_list)

        X_full = self._df_to_numpy(sdf)
        distances = np.sum(
            (X_full[:, None, :] - all_centers[None, :, :]) ** 2, axis=2
        )
        distances = np.maximum(distances, 1e-15)
        inv_exponent = -2.0 / (self.m - 1.0)
        inv_dist = distances ** inv_exponent
        U = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
        Um = U ** self.m
        centers_final = (Um.T @ X_full) / np.sum(Um, axis=0)[:, None]

        self._centers = centers_final
        self._U = U
        elapsed = time.perf_counter() - t0
        self._results["spark_gpu"] = elapsed
        return centers_final, U

    def fit_spark_cpu(self, df):
        if self.spark is None:
            raise RuntimeError("PySpark session not available")
        self.spark_mode = "spark_cpu"
        t0 = time.perf_counter()

        sdf = self._to_spark_df(df)
        n_features = len(sdf.columns)

        def process_partition(iterator):
            rows = list(iterator)
            if not rows:
                return iter([])
            X_local = np.array(rows, dtype=np.float64)
            n = X_local.shape[0]
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(n, min(self.n_clusters, n), replace=False)
            centers = X_local[idx].copy()
            for _ in range(self.max_iter):
                d = np.sum((X_local[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                d = np.maximum(d, 1e-15)
                e = -2.0 / (self.m - 1.0)
                inv = d ** e
                U = inv / np.sum(inv, axis=1, keepdims=True)
                Um = U ** self.m
                nc = (Um.T @ X_local) / np.sum(Um, axis=0)[:, None]
                if np.linalg.norm(nc - centers) < self.tol:
                    centers = nc
                    break
                centers = nc
            yield (0, centers.tobytes())

        partition_centers = sdf.rdd.mapPartitions(process_partition).collect()
        all_centers_list = [np.frombuffer(c, dtype=np.float64).reshape(-1, n_features)
                            for _, c in partition_centers]
        all_centers = np.vstack(all_centers_list)

        X_full = self._df_to_numpy(sdf)
        distances = np.sum(
            (X_full[:, None, :] - all_centers[None, :, :]) ** 2, axis=2
        )
        distances = np.maximum(distances, 1e-15)
        inv_exponent = -2.0 / (self.m - 1.0)
        inv_dist = distances ** inv_exponent
        U = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
        Um = U ** self.m
        centers_final = (Um.T @ X_full) / np.sum(Um, axis=0)[:, None]

        self._centers = centers_final
        self._U = U
        elapsed = time.perf_counter() - t0
        self._results["spark_cpu"] = elapsed
        return centers_final, U

    def fit_gpu(self, df):
        self.spark_mode = "gpu"
        if not _CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available for GPU mode")
        t0 = time.perf_counter()
        X = self._df_to_numpy(df)
        model = GPUFCMManager(use_gpu=True, n_clusters=self.n_clusters,
                              max_iter=self.max_iter, m=self.m, tol=self.tol)
        centers, U, _ = model.fit(X)
        self._centers = centers
        self._U = U
        elapsed = time.perf_counter() - t0
        self._results["gpu"] = elapsed
        return centers, U

    def fit_cpu(self, df):
        self.spark_mode = "cpu"
        t0 = time.perf_counter()
        X = self._df_to_numpy(df)
        model = GPUFCMManager(use_gpu=False, n_clusters=self.n_clusters,
                              max_iter=self.max_iter, m=self.m, tol=self.tol)
        centers, U, _ = model.fit(X)
        self._centers = centers
        self._U = U
        elapsed = time.perf_counter() - t0
        self._results["cpu"] = elapsed
        return centers, U

    def compare_modes(self, df, include_cpu=True, include_spark_cpu=True,
                      include_gpu=True, include_spark_gpu=True):
        results = {}

        if include_spark_gpu and _CUDA_AVAILABLE:
            t0 = time.perf_counter()
            self.fit_spark_gpu(df)
            results["spark_gpu"] = time.perf_counter() - t0

        if include_spark_cpu:
            t0 = time.perf_counter()
            self.fit_spark_cpu(df)
            results["spark_cpu"] = time.perf_counter() - t0

        if include_gpu and _CUDA_AVAILABLE:
            t0 = time.perf_counter()
            self.fit_gpu(df)
            results["gpu"] = time.perf_counter() - t0

        if include_cpu:
            t0 = time.perf_counter()
            self.fit_cpu(df)
            results["cpu"] = time.perf_counter() - t0

        import pandas as pd
        comparison = pd.DataFrame([
            {"mode": k, "time_seconds": v}
            for k, v in results.items()
        ])
        if not comparison.empty and comparison["time_seconds"].min() > 0:
            baseline = comparison["time_seconds"].max()
            comparison["speedup_vs_slowest"] = baseline / comparison["time_seconds"]
        return comparison

    @property
    def centers(self):
        return self._centers

    @property
    def U(self):
        return self._U

    @property
    def results(self):
        return dict(self._results)

    def stop_spark(self):
        if self.spark is not None:
            self.spark.stop()

    def __repr__(self):
        return (
            f"SparkGPUHybridEngine(mode={self.spark_mode}, "
            f"n_clusters={self.n_clusters}, max_iter={self.max_iter})"
        )
