import time
import warnings
import tracemalloc

import numpy as np
import psutil
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    accuracy_score,
    f1_score,
)


class MetricsCollector:
    def collect_all(self, X, labels, y_true=None, centers=None):
        metrics = self.collect_clustering_metrics(X, labels)
        if y_true is not None:
            metrics.update(self.collect_classification_metrics(y_true, labels))
        if centers is not None:
            metrics["center_recovery"] = self._center_recovery_error(centers, labels, X)
        return metrics

    def collect_clustering_metrics(self, X, labels):
        unique = set(labels)
        n_labels = len(unique)
        if n_labels < 2 or n_labels >= len(labels):
            metrics = {
                "silhouette_score": np.nan,
                "davies_bouldin_score": np.nan,
                "calinski_harabasz_score": np.nan,
                "n_clusters": n_labels,
            }
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = {
                    "silhouette_score": silhouette_score(X, labels),
                    "davies_bouldin_score": davies_bouldin_score(X, labels),
                    "calinski_harabasz_score": calinski_harabasz_score(X, labels),
                    "n_clusters": n_labels,
                }
        return metrics

    def collect_classification_metrics(self, y_true, y_pred):
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics.cluster import contingency_matrix

        cm = contingency_matrix(y_true, y_pred)
        row_ind, col_ind = linear_sum_assignment(-cm)
        y_pred_mapped = self._map_labels(y_true, y_pred)
        return {
            "accuracy": accuracy_score(y_true, y_pred_mapped),
            "f1_score": f1_score(y_true, y_pred_mapped, average="weighted"),
            "nmi": normalized_mutual_info_score(y_true, y_pred),
            "ari": adjusted_rand_score(y_true, y_pred),
        }

    def _map_labels(self, y_true, y_pred):
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics.cluster import contingency_matrix

        cm = contingency_matrix(y_true, y_pred)
        row_ind, col_ind = linear_sum_assignment(-cm)
        mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
        return np.array([mapping.get(p, -1) for p in y_pred])

    def _center_recovery_error(self, true_centers, labels, X):
        from scipy.optimize import linear_sum_assignment

        unique_labels = np.unique(labels)
        n_clusters = min(len(true_centers), len(unique_labels))
        if n_clusters < 2:
            return np.nan
        computed_centers = np.array([
            X[labels == c].mean(axis=0) for c in unique_labels[:n_clusters]
        ])
        tc = np.asarray(true_centers)[:n_clusters]
        cost_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                cost_matrix[i, j] = np.linalg.norm(tc[i] - computed_centers[j])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return float(cost_matrix[row_ind, col_ind].mean())

    def measure_time(self, func, *args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    def measure_memory(self, func, *args, **kwargs):
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
        result = func(*args, **kwargs)
        end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        stats = end_snapshot.compare_to(start_snapshot, "lineno")
        memory_bytes = sum(s.size_diff for s in stats)
        memory_mb = memory_bytes / (1024 * 1024)
        return result, memory_mb
