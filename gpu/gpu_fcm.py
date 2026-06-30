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
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

from .cuda_kernels import (
    check_cuda,
    compute_membership_gpu,
    compute_centers_gpu,
    compute_distances_gpu,
    compute_objective_gpu,
)


def _fcm_cpu(X, n_clusters, max_iter, m, tol, seed):
    rng = np.random.RandomState(seed)
    n_samples = X.shape[0]

    centers_idx = rng.choice(n_samples, n_clusters, replace=False)
    centers = X[centers_idx].copy()

    J_history = []
    for iteration in range(max_iter):
        distances = np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)

        distances = np.maximum(distances, 1e-15)
        inv_exponent = -2.0 / (m - 1.0)
        inv_dist = distances ** inv_exponent
        U = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)

        Um = U ** m
        new_centers = (Um.T @ X) / np.sum(Um, axis=0)[:, np.newaxis]

        J = np.sum(Um * distances)
        J_history.append(J)

        if np.linalg.norm(new_centers - centers) < tol:
            centers = new_centers
            break
        centers = new_centers

    return centers, U, J_history


class GPUFCMManager:
    def __init__(self, use_gpu=True, n_clusters=5, max_iter=100, m=2.0, tol=1e-4, seed=42):
        self.use_gpu = use_gpu and _CUDA_AVAILABLE
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.tol = tol
        self.seed = seed
        self._centers = None
        self._U = None
        self._J_history = []
        self._fit_time = 0.0
        self._memory_usage = {}

        if self.use_gpu:
            check_cuda()

    def to_gpu(self, X):
        if not self.use_gpu:
            return X
        return cp.asarray(X, dtype=cp.float64)

    def to_cpu(self, X):
        if cp is not None and isinstance(X, cp.ndarray):
            return cp.asnumpy(X)
        return np.asarray(X)

    @property
    def centers(self):
        return self._centers

    @property
    def U(self):
        return self._U

    @property
    def J_history(self):
        return self._J_history

    @property
    def fit_time(self):
        return self._fit_time

    def _track_memory(self, stage):
        if not _PSUTIL_AVAILABLE:
            return
        proc = psutil.Process()
        mem = proc.memory_info()
        self._memory_usage[stage] = {
            "rss_mb": mem.rss / 1e6,
            "vms_mb": mem.vms / 1e6,
        }

    def fit(self, X):
        self._track_memory("before_fit")
        t_start = time.perf_counter()

        if self.use_gpu:
            result = self._fit_gpu(X)
        else:
            result = _fcm_cpu(
                np.asarray(X), self.n_clusters, self.max_iter,
                self.m, self.tol, self.seed
            )

        self._centers, self._U, self._J_history = result
        self._fit_time = time.perf_counter() - t_start
        self._track_memory("after_fit")

        return self._centers, self._U, self._J_history

    def _fit_gpu(self, X):
        data = cp.asarray(X, dtype=cp.float64)
        n_samples, n_features = data.shape
        cp.random.seed(self.seed)

        idx = cp.random.choice(n_samples, self.n_clusters, replace=False)
        centers = data[idx].copy()

        J_history = []
        for iteration in range(self.max_iter):
            distances = compute_distances_gpu(data, centers)
            distances = cp.maximum(distances, 1e-15)

            U = compute_membership_gpu(distances, self.m)

            new_centers = compute_centers_gpu(data, U, self.m)

            J = compute_objective_gpu(U, distances, self.m)
            J_history.append(J)

            diff = cp.linalg.norm(new_centers - centers)
            centers = new_centers

            if diff < self.tol:
                break

        return self.to_cpu(centers), self.to_cpu(U), J_history

    def predict(self, X):
        if self._centers is None:
            raise RuntimeError("Must call fit() before predict()")

        X_arr = np.asarray(X)
        centers = np.asarray(self._centers)

        if self.use_gpu:
            X_gpu = cp.asarray(X_arr, dtype=cp.float64)
            centers_gpu = cp.asarray(centers, dtype=cp.float64)
            distances = compute_distances_gpu(X_gpu, centers_gpu)
            U = compute_membership_gpu(distances, self.m)
            labels = cp.argmax(U, axis=1)
            return cp.asnumpy(labels)
        else:
            distances = np.sum(
                (X_arr[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2
            )
            U = self._softmax(-distances)
            return np.argmax(U, axis=1)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def benchmark_cpu_vs_gpu(self, X, n_runs=3):
        results = {}

        cpu_manager = GPUFCMManager(use_gpu=False, n_clusters=self.n_clusters,
                                    max_iter=self.max_iter, m=self.m, tol=self.tol)
        cpu_times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            cpu_manager.fit(X)
            cpu_times.append(time.perf_counter() - t0)
        results["cpu"] = {
            "mean": np.mean(cpu_times),
            "std": np.std(cpu_times),
            "times": cpu_times,
        }

        if _CUDA_AVAILABLE:
            gpu_times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                self.fit(X)
                gpu_times.append(time.perf_counter() - t0)
            results["gpu"] = {
                "mean": np.mean(gpu_times),
                "std": np.std(gpu_times),
                "times": gpu_times,
            }
            if results["gpu"]["mean"] > 0:
                results["speedup"] = results["cpu"]["mean"] / results["gpu"]["mean"]
            else:
                results["speedup"] = float("inf")
        else:
            results["gpu"] = None
            results["speedup"] = 1.0

        return results

    def get_memory_report(self):
        return dict(self._memory_usage)

    def __repr__(self):
        gpu_str = "enabled" if self.use_gpu else "disabled"
        return (
            f"GPUFCMManager(use_gpu={gpu_str}, n_clusters={self.n_clusters}, "
            f"max_iter={self.max_iter}, m={self.m})"
        )
