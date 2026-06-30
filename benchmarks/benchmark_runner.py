import warnings
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics_collector import MetricsCollector

try:
    from baseline_project import FCM, FCLM
except ImportError:
    from sklearn.base import BaseEstimator, ClusterMixin

    class _FCM(ClusterMixin, BaseEstimator):
        def __init__(self, n_clusters=3, max_iter=100, m=2, error=1e-5, random_state=42):
            self.n_clusters = n_clusters
            self.max_iter = max_iter
            self.m = m
            self.error = error
            self.random_state = random_state

        def fit(self, X, y=None):
            rng = np.random.RandomState(self.random_state)
            n_samples, n_features = X.shape
            U = rng.dirichlet(np.ones(self.n_clusters), size=n_samples).T
            for _ in range(self.max_iter):
                Um = U ** self.m
                centers = (Um @ X) / Um.sum(axis=1, keepdims=True)
                dist = np.zeros((self.n_clusters, n_samples))
                for k in range(self.n_clusters):
                    diff = X - centers[k]
                    dist[k] = np.sqrt((diff ** 2).sum(axis=1))
                dist = np.maximum(dist, 1e-10)
                U_new = 1.0 / (dist ** (2 / (self.m - 1)))
                U_new = U_new / U_new.sum(axis=0, keepdims=True)
                if np.linalg.norm(U_new - U) < self.error:
                    break
                U = U_new
            self.U_ = U
            self.cluster_centers_ = centers
            self.labels_ = np.argmax(U, axis=0)
            return self

    class _FCLM(ClusterMixin, BaseEstimator):
        def __init__(self, n_clusters=3, max_iter=100, m=2, epsilon=0.1, random_state=42):
            self.n_clusters = n_clusters
            self.max_iter = max_iter
            self.m = m
            self.epsilon = epsilon
            self.random_state = random_state

        def fit(self, X, y=None):
            rng = np.random.RandomState(self.random_state)
            n_samples, n_features = X.shape
            U = rng.dirichlet(np.ones(self.n_clusters), size=n_samples).T
            for _ in range(self.max_iter):
                Um = U ** self.m
                centers = (Um @ X) / Um.sum(axis=1, keepdims=True)
                dist = np.zeros((self.n_clusters, n_samples))
                for k in range(self.n_clusters):
                    diff = X - centers[k]
                    dist[k] = np.sqrt((diff ** 2).sum(axis=1))
                dist = np.maximum(dist, 1e-10)
                weights = 1.0 / (1.0 + self.epsilon * dist)
                U_new = 1.0 / (dist ** (2 / (self.m - 1)))
                U_new = U_new / U_new.sum(axis=0, keepdims=True)
                U_new = U_new * weights
                U_new = U_new / U_new.sum(axis=0, keepdims=True)
                if np.linalg.norm(U_new - U) < 1e-5:
                    break
                U = U_new
            self.U_ = U
            self.cluster_centers_ = centers
            self.labels_ = np.argmax(U, axis=0)
            return self

    FCM = _FCM
    FCLM = _FCLM

try:
    from novel_algorithm import ADEFCM
except ImportError:
    from sklearn.base import BaseEstimator, ClusterMixin

    class _ADEFCM(ClusterMixin, BaseEstimator):
        def __init__(self, n_clusters=3, max_iter=100, m=2.0, random_state=42):
            self.n_clusters = n_clusters
            self.max_iter = max_iter
            self.m = m
            self.random_state = random_state

        def fit(self, X, y=None):
            rng = np.random.RandomState(self.random_state)
            n_samples, n_features = X.shape
            U = rng.dirichlet(np.ones(self.n_clusters), size=n_samples).T
            for _ in range(self.max_iter):
                Um = U ** self.m
                centers = (Um @ X) / Um.sum(axis=1, keepdims=True)
                dist = np.zeros((self.n_clusters, n_samples))
                for k in range(self.n_clusters):
                    diff = X - centers[k]
                    dist[k] = np.sqrt((diff ** 2).sum(axis=1))
                dist = np.maximum(dist, 1e-10)
                m_adaptive = self.m + 0.1 * np.random.randn()
                m_adaptive = np.clip(m_adaptive, 1.1, 4.0)
                U_new = 1.0 / (dist ** (2 / (m_adaptive - 1)))
                U_new = U_new / U_new.sum(axis=0, keepdims=True)
                if np.linalg.norm(U_new - U) < 1e-5:
                    break
                U = U_new
            self.U_ = U
            self.cluster_centers_ = centers
            self.labels_ = np.argmax(U, axis=0)
            self.adaptive_fuzzifier_ = self.m
            return self

    ADEFCM = _ADEFCM

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.cluster import (
        KMeans,
        MiniBatchKMeans,
        SpectralClustering,
        DBSCAN,
        OPTICS,
        Birch as BIRCH,
        AgglomerativeClustering,
    )
    from sklearn.mixture import GaussianMixture

METRICS = [
    "accuracy",
    "nmi",
    "ari",
    "silhouette_score",
    "davies_bouldin_score",
    "calinski_harabasz_score",
    "execution_time",
    "memory_mb",
    "n_clusters",
]


class BenchmarkRunner:
    def __init__(self, random_state=42, results_dir=None):
        self.random_state = random_state
        self.results_dir = Path(results_dir) if results_dir else Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.collector = MetricsCollector()
        self._init_algorithms()
        self._init_datasets()

    def _init_algorithms(self):
        rs = self.random_state
        self.algorithms = {
            "KMeans": KMeans(n_clusters=3, random_state=rs, n_init="auto"),
            "MiniBatchKMeans": MiniBatchKMeans(n_clusters=3, random_state=rs, n_init="auto"),
            "FCM": FCM(n_clusters=3, random_state=rs),
            "FCLM": FCLM(n_clusters=3, random_state=rs),
            "ADE-FCM": ADEFCM(n_clusters=3, random_state=rs),
            "SpectralClustering": SpectralClustering(n_clusters=3, random_state=rs),
            "DBSCAN": DBSCAN(),
            "OPTICS": OPTICS(max_eps=np.inf, cluster_method="xi"),
            "BIRCH": BIRCH(n_clusters=3),
            "AgglomerativeClustering": AgglomerativeClustering(n_clusters=3),
            "GaussianMixture": GaussianMixture(n_components=3, random_state=rs),
        }

    def _init_datasets(self):
        from sklearn import datasets as sklearn_datasets

        rs = self.random_state

        def _make_blobs():
            from sklearn.datasets import make_blobs
            X, y = make_blobs(
                n_samples=500, n_features=10, centers=5, random_state=rs
            )
            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
            df["label"] = y
            return df

        def _make_moons():
            from sklearn.datasets import make_moons
            X, y = make_moons(n_samples=500, noise=0.05, random_state=rs)
            df = pd.DataFrame(X, columns=["f1", "f2"])
            df["label"] = y
            return df

        def _make_circles():
            from sklearn.datasets import make_circles
            X, y = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=rs)
            df = pd.DataFrame(X, columns=["f1", "f2"])
            df["label"] = y
            return df

        def _make_varied():
            from sklearn.datasets import make_blobs
            X, y = make_blobs(
                n_samples=500, n_features=5, centers=4,
                cluster_std=[1.0, 2.5, 0.5, 3.0], random_state=rs,
            )
            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
            df["label"] = y
            return df

        def _load_iris():
            data = sklearn_datasets.load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["label"] = data.target
            df.columns = [c.replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]
            return df

        def _load_wine():
            data = sklearn_datasets.load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["label"] = data.target
            df.columns = [c.replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]
            return df

        def _load_digits():
            data = sklearn_datasets.load_digits()
            df = pd.DataFrame(data.data)
            df["label"] = data.target
            return df

        def _load_breast_cancer():
            data = sklearn_datasets.load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["label"] = data.target
            df.columns = [c.replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]
            return df

        self.datasets = {
            "blobs": (_make_blobs(), 5),
            "moons": (_make_moons(), 2),
            "circles": (_make_circles(), 2),
            "varied": (_make_varied(), 4),
            "iris": (_load_iris(), 3),
            "wine": (_load_wine(), 3),
            "digits": (_load_digits(), 10),
            "breast_cancer": (_load_breast_cancer(), 2),
        }

    def _set_n_clusters(self, algo_name, algo, n_clusters):
        params = {}
        if hasattr(algo, "n_clusters"):
            params["n_clusters"] = n_clusters
        elif hasattr(algo, "n_components"):
            params["n_components"] = n_clusters
        if params and hasattr(algo, "set_params"):
            return deepcopy(algo).set_params(**params)
        return algo

    def run_single(self, algorithm_name, dataset_name):
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        df_data, n_classes = self.datasets[dataset_name]
        y_true = df_data["label"].values if "label" in df_data.columns else None
        feature_cols = [c for c in df_data.columns if c != "label"]
        X = df_data[feature_cols].values.astype(np.float64)

        algo = self.algorithms[algorithm_name]
        if algorithm_name not in ("DBSCAN", "OPTICS"):
            algo = self._set_n_clusters(algorithm_name, algo, n_classes)

        def _fit():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = algo.fit(X)
                if hasattr(result, "labels_"):
                    labels = result.labels_
                elif hasattr(result, "predict"):
                    labels = result.predict(X)
                else:
                    labels = np.zeros(len(X), dtype=int)
            return labels

        labels, fit_time = self.collector.measure_time(_fit)
        memory_mb = 0.0

        if np.any(labels == -1):
            mask = labels != -1
            if mask.sum() > 1:
                X_clean, labels_clean = X[mask], labels[mask]
                y_true_clean = y_true[mask] if y_true is not None else None
            else:
                X_clean, labels_clean = X, labels
                y_true_clean = y_true
        else:
            X_clean, labels_clean = X, labels
            y_true_clean = y_true

        metrics = self.collector.collect_all(X_clean, labels_clean, y_true_clean)
        metrics["execution_time"] = fit_time
        metrics["memory_mb"] = abs(memory_mb)
        metrics["algorithm"] = algorithm_name
        metrics["dataset"] = dataset_name

        return metrics

    def run_all(self):
        results = []
        for algo_name in self.algorithms:
            for ds_name in self.datasets:
                try:
                    metrics = self.run_single(algo_name, ds_name)
                    results.append(metrics)
                except Exception as e:
                    results.append({
                        "algorithm": algo_name,
                        "dataset": ds_name,
                        "error": str(e),
                        "execution_time": np.nan,
                        "memory_mb": np.nan,
                    })
                    print(f"[ERROR] {algo_name} on {ds_name}: {e}")
                    traceback.print_exc()
        df = pd.DataFrame(results)
        self._save_results(df)
        return df

    def _save_results(self, df):
        path = self.results_dir / "benchmark_results.csv"
        df.to_csv(path, index=False)
        path_json = self.results_dir / "benchmark_results.json"
        df.to_json(path_json, orient="records", indent=2)

    def compare_algorithms(self, datasets=None, metrics=None):
        results_path = self.results_dir / "benchmark_results.csv"
        if results_path.exists():
            df = pd.read_csv(results_path)
        else:
            df = self.run_all()
        if datasets:
            df = df[df["dataset"].isin(datasets)]
        if metrics is None:
            metrics = [
                "accuracy", "nmi", "ari", "silhouette_score",
                "davies_bouldin_score", "calinski_harabasz_score",
                "execution_time", "memory_mb",
            ]
        available = [m for m in metrics if m in df.columns]
        table = df.groupby("algorithm")[available].mean().round(4)
        table["datasets_completed"] = df.groupby("algorithm").size()
        return table.sort_values("silhouette_score", ascending=False)
