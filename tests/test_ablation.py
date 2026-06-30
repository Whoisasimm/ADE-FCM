import numpy as np
import pytest

from ablation.ablation_study import AblationStudy


class TestAblation:
    def test_initialization(self, small_blobs):
        X, y = small_blobs
        study = AblationStudy(X, y_true=y, n_clusters=3, random_state=42)
        assert study.X.shape == X.shape
        assert study.y_true is y
        assert study.n_clusters == 3

    def test_run_full_ade_fcm(self, small_blobs):
        X, y = small_blobs
        study = AblationStudy(X, y_true=y, n_clusters=3, random_state=42)
        model, metrics = study.run_full_ade_fcm()
        assert model is not None
        assert model.labels_ is not None
        assert "silhouette" in metrics
        assert "davies_bouldin" in metrics
        assert "time" in metrics
        assert "iterations" in metrics
        assert "objective" in metrics
        assert metrics["iterations"] >= 1

    def test_run_without_adaptive_fuzzifier(self, small_blobs):
        X, _ = small_blobs
        study = AblationStudy(X, n_clusters=3, random_state=42)
        metrics = study.run_without_adaptive_fuzzifier()
        assert "silhouette" in metrics
        assert "davies_bouldin" in metrics
        assert metrics["iterations"] >= 1

    def test_run_without_auto_k(self, small_blobs):
        X, _ = small_blobs
        study = AblationStudy(X, n_clusters=3, random_state=42)
        metrics = study.run_without_auto_k()
        assert "silhouette" in metrics
        assert metrics["iterations"] >= 1

    def test_run_without_explainability(self, small_blobs):
        X, _ = small_blobs
        study = AblationStudy(X, n_clusters=3, random_state=42)
        metrics = study.run_without_explainability()
        assert "silhouette" in metrics
        assert metrics["iterations"] >= 1

    def test_run_without_outlier_robustness(self, small_blobs):
        X, _ = small_blobs
        study = AblationStudy(X, n_clusters=3, random_state=42)
        metrics = study.run_without_outlier_robustness()
        assert "silhouette" in metrics
        assert metrics["iterations"] >= 1

    def test_run_without_early_stopping(self, small_blobs):
        X, _ = small_blobs
        study = AblationStudy(X, n_clusters=3, random_state=42)
        metrics = study.run_without_early_stopping()
        assert "silhouette" in metrics
        assert metrics["iterations"] >= 1

    def test_run_all_completes(self, small_blobs):
        X, y = small_blobs
        study = AblationStudy(X, y_true=y, n_clusters=3, random_state=42)
        results = study.run_all()
        assert isinstance(results, dict)
        assert "full_ade_fcm" in results
        assert "without_adaptive_fuzzifier" in results
        assert "without_auto_k" in results
        assert "without_explainability" in results
        assert "without_outlier_robustness" in results
        assert "without_early_stopping" in results

    def test_full_ade_fcm_ari_computed(self, small_blobs):
        X, y = small_blobs
        study = AblationStudy(X, y_true=y, n_clusters=3, random_state=42)
        model, metrics = study.run_full_ade_fcm()
        assert "ari" in metrics
        assert "nmi" in metrics
        assert isinstance(metrics["ari"], float)

    def test_run_all_results_have_silhouette(self, small_blobs):
        X, _ = small_blobs
        study = AblationStudy(X, n_clusters=3, random_state=42)
        results = study.run_all()
        for name, metrics in results.items():
            if isinstance(metrics, dict) and "error" not in metrics:
                assert "silhouette" in metrics, f"{name} missing silhouette"
                assert "davies_bouldin" in metrics, f"{name} missing davies_bouldin"

    def test_generate_report(self, small_blobs, tmp_path):
        X, y = small_blobs
        study = AblationStudy(X, y_true=y, n_clusters=3, random_state=42)
        output = str(tmp_path / "report.json")
        report = study.generate_report(output_path=output)
        assert "full_ade_fcm" in report
        assert "ablation_results" in report
        assert "degradation" in report

    def test_reproducible_results(self, small_blobs):
        X, _ = small_blobs
        study1 = AblationStudy(X, n_clusters=3, random_state=42)
        study2 = AblationStudy(X, n_clusters=3, random_state=42)
        r1 = study1.run_all()
        r2 = study2.run_all()
        for key in r1:
            if isinstance(r1[key], dict) and "error" not in r1[key]:
                assert np.isclose(r1[key]["silhouette"], r2[key]["silhouette"], atol=1e-4), key
