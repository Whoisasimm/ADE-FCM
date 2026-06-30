import warnings
import numpy as np

try:
    import cupy as cp
    _CUDA_AVAILABLE = True
except ImportError:
    cp = None
    _CUDA_AVAILABLE = False

try:
    from cuml.cluster import KMeans as cuKMeans
    from cuml.metrics.cluster import silhouette_score as cu_silhouette
    from cuml.metrics import pairwise_distances as cu_pairwise
    _RAPIDS_AVAILABLE = True
except ImportError:
    cuKMeans = None
    cu_silhouette = None
    cu_pairwise = None
    _RAPIDS_AVAILABLE = False

from .cuda_kernels import (
    check_cuda,
    compute_membership_gpu,
    compute_centers_gpu,
    compute_distances_gpu,
    compute_objective_gpu,
)


class RAPIDSFCM:
    def __init__(self, n_clusters=5, max_iter=100, m=2.0, tol=1e-4, seed=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.tol = tol
        self.seed = seed
        self._centers = None
        self._U = None
        self._J_history = []
        self._use_rapids = _RAPIDS_AVAILABLE and _CUDA_AVAILABLE

        if not self._use_rapids:
            warnings.warn("RAPIDS cuML not available. Falling back to CuPy-only mode.")

    def _validate(self):
        if not _CUDA_AVAILABLE:
            raise RuntimeError("CUDA/CuPy is required for RAPIDSFCM. Install cupy.")
        check_cuda()

    def fit_rapids_kmeans(self, X):
        self._validate()
        data = cp.asarray(X, dtype=cp.float64)

        km = cuKMeans(n_clusters=self.n_clusters, max_iter=self.max_iter,
                      random_state=self.seed, tol=self.tol, verbose=0)
        km.fit(data)

        self._centers = km.cluster_centers_
        self._U = self._compute_fuzzy_membership(data, self._centers)
        self._J_history = [float(km.inertia_)]
        return self._centers, self._U, self._J_history

    def fit_fuzzy(self, X):
        self._validate()
        data = cp.asarray(X, dtype=cp.float64)
        n_samples = data.shape[0]
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

            if cp.linalg.norm(new_centers - centers) < self.tol:
                centers = new_centers
                break
            centers = new_centers

        self._centers = centers
        self._U = U
        self._J_history = J_history
        return self.to_cpu(centers), self.to_cpu(U), J_history

    def _compute_fuzzy_membership(self, data, centers):
        distances = compute_distances_gpu(data, centers)
        distances = cp.maximum(distances, 1e-15)
        return compute_membership_gpu(distances, self.m)

    def predict(self, X, use_rapids=False):
        if self._centers is None:
            raise RuntimeError("Must call fit() before predict()")

        if use_rapids and _RAPIDS_AVAILABLE:
            data = cp.asarray(X, dtype=cp.float64)
            centers = cp.asarray(self._centers, dtype=cp.float64)
            distances = cu_pairwise(data, centers, metric="euclidean") ** 2
        else:
            data = cp.asarray(X, dtype=cp.float64)
            centers = cp.asarray(self._centers, dtype=cp.float64)
            distances = compute_distances_gpu(data, centers)

        distances = cp.maximum(distances, 1e-15)
        U = compute_membership_gpu(distances, self.m)
        return cp.argmax(U, axis=1)

    @staticmethod
    def silhouette_score(X, labels, use_rapids=True):
        if use_rapids and cu_silhouette is not None:
            data = cp.asarray(X, dtype=cp.float64)
            labels_gpu = cp.asarray(labels, dtype=cp.int32)
            return float(cu_silhouette(data, labels_gpu))
        else:
            from sklearn.metrics import silhouette_score as sk_silhouette
            return sk_silhouette(np.asarray(X), np.asarray(labels))

    @staticmethod
    def pairwise_distances_gpu(X, Y=None, metric="euclidean"):
        if cu_pairwise is not None:
            x_gpu = cp.asarray(X, dtype=cp.float64)
            y_gpu = cp.asarray(Y, dtype=cp.float64) if Y is not None else None
            return cu_pairwise(x_gpu, y_gpu, metric=metric)
        check_cuda()
        x_gpu = cp.asarray(X, dtype=cp.float64)
        y_gpu = cp.asarray(Y, dtype=cp.float64) if Y is not None else None
        from cupyx.scipy.spatial.distance import cdist
        return cdist(x_gpu, y_gpu if y_gpu is not None else x_gpu, metric=metric)

    @staticmethod
    def to_gpu(X):
        return cp.asarray(X, dtype=cp.float64)

    @staticmethod
    def to_cpu(X):
        if cp is not None and isinstance(X, cp.ndarray):
            return cp.asnumpy(X)
        return np.asarray(X)

    @property
    def centers(self):
        return self.to_cpu(self._centers) if self._centers is not None else None

    @property
    def U(self):
        return self.to_cpu(self._U) if self._U is not None else None

    @property
    def J_history(self):
        return self._J_history

    def __repr__(self):
        mode = "RAPIDS+CuPy" if self._use_rapids else "CuPy-only"
        return (
            f"RAPIDSFCM(mode={mode}, n_clusters={self.n_clusters}, "
            f"max_iter={self.max_iter}, m={self.m})"
        )
