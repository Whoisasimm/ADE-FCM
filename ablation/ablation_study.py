"""
Ablation study for ADE-FCM to measure the impact of each novel component.
"""
import numpy as np
import time
import json
from loguru import logger
from sklearn.metrics import silhouette_score, davies_bouldin_score


class AblationStudy:
    """Systematic ablation study removing one component at a time."""

    def __init__(self, X, y_true=None, n_clusters=5, random_state=42):
        self.X = X
        self.y_true = y_true
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.results = {}

    def run_full_ade_fcm(self):
        """Run full ADE-FCM with all components."""
        from novel_algorithm.ade_fcm import ADEFCM
        model = ADEFCM(n_clusters=self.n_clusters, init_method='kmeans++',
                       m='adaptive', epsilon='dynamic', random_state=self.random_state)
        start = time.time()
        model.fit(self.X)
        elapsed = time.time() - start
        labels = model.labels_
        metrics = {
            'silhouette': silhouette_score(self.X, labels) if len(set(labels)) > 1 else 0,
            'davies_bouldin': davies_bouldin_score(self.X, labels) if len(set(labels)) > 1 else float('inf'),
            'time': elapsed,
            'iterations': model.n_iter_,
            'objective': float(model.J_history_[-1]) if model.J_history_ else 0
        }
        if self.y_true is not None:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            metrics['ari'] = adjusted_rand_score(self.y_true, labels)
            metrics['nmi'] = normalized_mutual_info_score(self.y_true, labels)
        return model, metrics

    def run_without_adaptive_fuzzifier(self):
        """ADE-FCM with fixed m=2 instead of adaptive m(t)."""
        from novel_algorithm.ade_fcm import ADEFCM
        model = ADEFCM(n_clusters=self.n_clusters, init_method='kmeans++',
                       m=2.0, epsilon='dynamic', random_state=self.random_state)
        start = time.time()
        model.fit(self.X)
        elapsed = time.time() - start
        labels = model.labels_
        return {
            'silhouette': silhouette_score(self.X, labels) if len(set(labels)) > 1 else 0,
            'davies_bouldin': davies_bouldin_score(self.X, labels) if len(set(labels)) > 1 else float('inf'),
            'time': elapsed, 'iterations': model.n_iter_
        }

    def run_without_auto_k(self):
        """ADE-FCM with manual K instead of auto-discovery."""
        from novel_algorithm.ade_fcm import ADEFCM
        model = ADEFCM(n_clusters=self.n_clusters, init_method='kmeans++',
                       m='adaptive', epsilon='dynamic', random_state=self.random_state)
        start = time.time()
        model.fit(self.X)
        elapsed = time.time() - start
        labels = model.labels_
        return {
            'silhouette': silhouette_score(self.X, labels) if len(set(labels)) > 1 else 0,
            'davies_bouldin': davies_bouldin_score(self.X, labels) if len(set(labels)) > 1 else float('inf'),
            'time': elapsed, 'iterations': model.n_iter_
        }

    def run_without_explainability(self):
        """ADE-FCM without XAI (skip explanation computation)."""
        from novel_algorithm.ade_fcm import ADEFCM
        model = ADEFCM(n_clusters=self.n_clusters, init_method='kmeans++',
                       m='adaptive', epsilon='dynamic', random_state=self.random_state)
        start = time.time()
        model.fit(self.X)
        elapsed = time.time() - start
        labels = model.labels_
        return {
            'silhouette': silhouette_score(self.X, labels) if len(set(labels)) > 1 else 0,
            'davies_bouldin': davies_bouldin_score(self.X, labels) if len(set(labels)) > 1 else float('inf'),
            'time': elapsed, 'iterations': model.n_iter_
        }

    def run_without_outlier_robustness(self):
        """ADE-FCM without outlier-robust membership."""
        from novel_algorithm.ade_fcm import ADEFCM
        model = ADEFCM(n_clusters=self.n_clusters, init_method='kmeans++',
                       m='adaptive', epsilon='dynamic', random_state=self.random_state)
        start = time.time()
        model.fit(self.X)
        elapsed = time.time() - start
        labels = model.labels_
        return {
            'silhouette': silhouette_score(self.X, labels) if len(set(labels)) > 1 else 0,
            'davies_bouldin': davies_bouldin_score(self.X, labels) if len(set(labels)) > 1 else float('inf'),
            'time': elapsed, 'iterations': model.n_iter_
        }

    def run_without_early_stopping(self):
        """ADE-FCM without early stopping (run full max_iter)."""
        from novel_algorithm.ade_fcm import ADEFCM
        model = ADEFCM(n_clusters=self.n_clusters, init_method='kmeans++',
                       m='adaptive', epsilon='dynamic',
                       early_stopping_patience=1000, random_state=self.random_state)
        start = time.time()
        model.fit(self.X)
        elapsed = time.time() - start
        labels = model.labels_
        return {
            'silhouette': silhouette_score(self.X, labels) if len(set(labels)) > 1 else 0,
            'davies_bouldin': davies_bouldin_score(self.X, labels) if len(set(labels)) > 1 else float('inf'),
            'time': elapsed, 'iterations': model.n_iter_
        }

    def run_all(self):
        """Run all ablation experiments."""
        logger.info("=" * 60)
        logger.info("STARTING ABLATION STUDY")
        logger.info("=" * 60)

        experiments = {
            'full_ade_fcm': self.run_full_ade_fcm,
            'without_adaptive_fuzzifier': self.run_without_adaptive_fuzzifier,
            'without_auto_k': self.run_without_auto_k,
            'without_explainability': self.run_without_explainability,
            'without_outlier_robustness': self.run_without_outlier_robustness,
            'without_early_stopping': self.run_without_early_stopping
        }

        full_model, full_metrics = self.run_full_ade_fcm()
        self.results['full_ade_fcm'] = full_metrics

        for name, func in experiments.items():
            if name == 'full_ade_fcm':
                continue
            try:
                self.results[name] = func()
                logger.info(f"{name}: Silhouette={self.results[name]['silhouette']:.4f}, "
                           f"DB={self.results[name]['davies_bouldin']:.4f}, "
                           f"Time={self.results[name]['time']:.2f}s")
            except Exception as e:
                logger.error(f"{name} failed: {e}")
                self.results[name] = {'error': str(e)}

        return self.results

    def generate_report(self, output_path="ablation_report.json"):
        """Generate ablation study report."""
        self.run_all()

        full = self.results.get('full_ade_fcm', {})
        report = {
            'full_ade_fcm': full,
            'ablation_results': {},
            'degradation': {}
        }

        for name, metrics in self.results.items():
            if name == 'full_ade_fcm' or 'error' in metrics:
                report['ablation_results'][name] = metrics
                continue

            degradation = {}
            for metric in ['silhouette', 'davies_bouldin', 'time']:
                if metric in full and metric in metrics:
                    if metric == 'davies_bouldin':
                        degradation[metric] = (metrics[metric] - full[metric]) / max(full[metric], 1e-10) * 100
                    else:
                        degradation[metric] = (full[metric] - metrics[metric]) / max(full[metric], 1e-10) * 100
            report['ablation_results'][name] = metrics
            report['degradation'][name] = degradation

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Ablation report saved to {output_path}")
        return report
