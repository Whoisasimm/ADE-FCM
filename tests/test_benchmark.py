import numpy as np
import pandas as pd
import pytest

from benchmarks.metrics_collector import MetricsCollector
from benchmarks.benchmark_runner import BenchmarkRunner


class TestMetricsCollector:
    def test_clustering_metrics_silhouette(self, iris_X, iris_y):
        collector = MetricsCollector()
        metrics = collector.collect_clustering_metrics(iris_X, iris_y)
        assert "silhouette_score" in metrics
        assert isinstance(metrics["silhouette_score"], float)
        assert -1.0 <= metrics["silhouette_score"] <= 1.0

    def test_clustering_metrics_davies_bouldin(self, iris_X, iris_y):
        collector = MetricsCollector()
        metrics = collector.collect_clustering_metrics(iris_X, iris_y)
        assert "davies_bouldin_score" in metrics
        assert metrics["davies_bouldin_score"] >= 0

    def test_clustering_metrics_calinski_harabasz(self, iris_X, iris_y):
        collector = MetricsCollector()
        metrics = collector.collect_clustering_metrics(iris_X, iris_y)
        assert "calinski_harabasz_score" in metrics
        assert metrics["calinski_harabasz_score"] > 0

    def test_clustering_metrics_n_clusters(self, iris_X, iris_y):
        collector = MetricsCollector()
        metrics = collector.collect_clustering_metrics(iris_X, iris_y)
        assert metrics["n_clusters"] == 3

    def test_clustering_metrics_single_cluster(self):
        collector = MetricsCollector()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = np.array([0, 0])
        metrics = collector.collect_clustering_metrics(X, labels)
        assert np.isnan(metrics["silhouette_score"])

    def test_classification_metrics(self, iris_X, iris_y):
        collector = MetricsCollector()
        metrics = collector.collect_classification_metrics(iris_y, iris_y)
        assert metrics["accuracy"] == 1.0
        assert metrics["nmi"] == 1.0
        assert metrics["ari"] == 1.0

    def test_collect_all_no_y(self, iris_X, iris_y):
        collector = MetricsCollector()
        metrics = collector.collect_all(iris_X, iris_y)
        assert "silhouette_score" in metrics
        assert "n_clusters" in metrics

    def test_collect_all_with_y(self, iris_X, iris_y):
        collector = MetricsCollector()
        metrics = collector.collect_all(iris_X, iris_y, y_true=iris_y)
        assert "accuracy" in metrics
        assert "nmi" in metrics
        assert "ari" in metrics

    def test_collect_all_with_centers(self, iris_X, iris_y):
        collector = MetricsCollector()
        centers = np.array([iris_X[iris_y == k].mean(axis=0) for k in range(3)])
        metrics = collector.collect_all(iris_X, iris_y, centers=centers)
        assert "center_recovery" in metrics

    def test_measure_time(self):
        collector = MetricsCollector()

        def dummy():
            return 42

        result, elapsed = collector.measure_time(dummy)
        assert result == 42
        assert elapsed >= 0

    def test_measure_memory(self):
        collector = MetricsCollector()

        def dummy():
            return np.zeros(1000)

        result, mem_mb = collector.measure_memory(dummy)
        assert result.shape == (1000,)
        assert isinstance(mem_mb, float)


class TestBenchmarkRunner:
    def test_initialization(self):
        runner = BenchmarkRunner(random_state=42)
        assert hasattr(runner, "algorithms")
        assert hasattr(runner, "datasets")
        assert "KMeans" in runner.algorithms
        assert "ADE-FCM" in runner.algorithms
        assert "FCM" in runner.algorithms
        assert "iris" in runner.datasets

    def test_run_single_kmeans_iris(self):
        runner = BenchmarkRunner(random_state=42)
        metrics = runner.run_single("KMeans", "iris")
        assert metrics["algorithm"] == "KMeans"
        assert metrics["dataset"] == "iris"
        assert "silhouette_score" in metrics
        assert "execution_time" in metrics
        assert metrics["n_clusters"] == 3

    def test_run_single_ade_fcm_iris(self):
        runner = BenchmarkRunner(random_state=42)
        metrics = runner.run_single("ADE-FCM", "iris")
        assert metrics["algorithm"] == "ADE-FCM"
        assert metrics["dataset"] == "iris"
        assert "silhouette_score" in metrics

    def test_run_single_fcm_iris(self):
        runner = BenchmarkRunner(random_state=42)
        metrics = runner.run_single("FCM", "iris")
        assert metrics["algorithm"] == "FCM"
        assert metrics["dataset"] == "iris"

    def test_run_single_fclm_iris(self):
        runner = BenchmarkRunner(random_state=42)
        metrics = runner.run_single("FCLM", "iris")
        assert metrics["algorithm"] == "FCLM"

    def test_run_single_dbscan_iris(self):
        runner = BenchmarkRunner(random_state=42)
        metrics = runner.run_single("DBSCAN", "iris")
        assert metrics["algorithm"] == "DBSCAN"

    def test_run_single_birch_iris(self):
        runner = BenchmarkRunner(random_state=42)
        metrics = runner.run_single("BIRCH", "iris")
        assert metrics["algorithm"] == "BIRCH"

    def test_run_single_gaussian_mixture_iris(self):
        runner = BenchmarkRunner(random_state=42)
        metrics = runner.run_single("GaussianMixture", "iris")
        assert metrics["algorithm"] == "GaussianMixture"

    def test_run_single_unknown_algorithm(self):
        runner = BenchmarkRunner(random_state=42)
        with pytest.raises(ValueError, match="Unknown algorithm"):
            runner.run_single("NonExistent", "iris")

    def test_run_single_unknown_dataset(self):
        runner = BenchmarkRunner(random_state=42)
        with pytest.raises(ValueError, match="Unknown dataset"):
            runner.run_single("KMeans", "nonexistent")
