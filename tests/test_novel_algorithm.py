import numpy as np
import pytest

from novel_algorithm.ade_fcm import ADEFCM
from novel_algorithm.density_init import DensityInitializer, KMeansPlusPlusInitializer, RandomInitializer
from novel_algorithm.adaptive_params import AdaptiveFuzzifier, DynamicThreshold, EarlyStopping
from novel_algorithm.auto_cluster import AutomaticClusterDiscovery, ClusterEvaluator
from novel_algorithm.outlier_detector import OutlierDetector
from novel_algorithm.xai import (
    feature_importance,
    cluster_summary,
    describe_cluster_natural,
    explain_clusters,
    shap_explain,
)


class TestADEFCMInit:
    def test_kmeans_pp_init_shape(self, tiny_data):
        model = ADEFCM(n_clusters=2, init_method="kmeans++", random_state=42, verbose=False)
        centers = model._kmeans_pp_init(tiny_data)
        assert centers.shape == (2, 2)

    def test_kmeans_pp_init_selects_from_data(self, tiny_data):
        model = ADEFCM(n_clusters=2, init_method="kmeans++", random_state=42, verbose=False)
        centers = model._kmeans_pp_init(tiny_data)
        for c in centers:
            assert any(np.allclose(c, row) for row in tiny_data)

    def test_density_init_shape(self, tiny_data):
        model = ADEFCM(n_clusters=2, init_method="density", random_state=42, verbose=False)
        centers = model._density_init(tiny_data)
        assert centers.shape == (2, 2)

    def test_random_init_from_ade_fcm(self, tiny_data):
        model = ADEFCM(n_clusters=2, init_method="random", random_state=42, verbose=False)
        centers = model._initialize_centers(tiny_data)
        assert centers.shape == (2, 2)

    def test_density_initializer_standalone(self, tiny_data):
        di = DensityInitializer(n_clusters=2, random_state=42)
        centers = di.initialize(tiny_data)
        assert centers.shape == (2, 2)

    def test_kmeans_plus_plus_initializer_standalone(self, tiny_data):
        kppi = KMeansPlusPlusInitializer(n_clusters=2, random_state=42)
        centers = kppi.initialize(tiny_data)
        assert centers.shape == (2, 2)

    def test_random_initializer_standalone(self, tiny_data):
        ri = RandomInitializer(n_clusters=2, random_state=42)
        centers = ri.initialize(tiny_data)
        assert centers.shape == (2, 2)
        for c in centers:
            assert any(np.allclose(c, row) for row in tiny_data)


class TestADEFCM:
    def test_fit_returns_self(self, tiny_data):
        model = ADEFCM(n_clusters=2, max_iter=10, random_state=42, verbose=False)
        result = model.fit(tiny_data)
        assert result is model

    def test_fit_sets_centers(self, tiny_data):
        model = ADEFCM(n_clusters=2, max_iter=10, random_state=42, verbose=False)
        model.fit(tiny_data)
        assert model.centers_ is not None
        assert model.centers_.shape == (2, 2)

    def test_fit_sets_labels(self, tiny_data):
        model = ADEFCM(n_clusters=2, max_iter=10, random_state=42, verbose=False)
        model.fit(tiny_data)
        assert model.labels_ is not None
        assert model.labels_.shape == (len(tiny_data),)
        assert model.labels_.dtype == np.int_ or model.labels_.dtype == np.int64

    def test_fit_sets_U(self, tiny_data):
        model = ADEFCM(n_clusters=2, max_iter=10, random_state=42, verbose=False)
        model.fit(tiny_data)
        assert model.U_ is not None
        assert model.U_.shape == (len(tiny_data), 2)
        assert np.allclose(model.U_.sum(axis=1), 1.0)

    def test_fit_J_history(self, tiny_data):
        model = ADEFCM(n_clusters=2, max_iter=10, random_state=42, verbose=False)
        model.fit(tiny_data)
        assert len(model.J_history_) > 0

    def test_predict_shape(self, tiny_data):
        model = ADEFCM(n_clusters=2, max_iter=10, random_state=42, verbose=False)
        model.fit(tiny_data)
        labels = model.predict(tiny_data)
        assert labels.shape == (len(tiny_data),)

    def test_fit_predict(self, tiny_data):
        model = ADEFCM(n_clusters=2, max_iter=10, random_state=42, verbose=False)
        labels = model.fit_predict(tiny_data)
        assert labels.shape == (len(tiny_data),)
        assert np.array_equal(labels, model.labels_)

    def test_outlier_detection(self, tiny_data):
        model = ADEFCM(n_clusters=2, max_iter=10, random_state=42, verbose=False)
        model.fit(tiny_data)
        assert model.outlier_mask_ is not None
        assert model.outlier_mask_.shape == (len(tiny_data),)
        assert model.outlier_mask_.dtype == bool
        assert model.outlier_scores_ is not None
        assert model.outlier_scores_.shape == (len(tiny_data),)

    def test_fit_on_blobs(self, blobs_X):
        model = ADEFCM(n_clusters=3, max_iter=30, random_state=42, verbose=False)
        model.fit(blobs_X)
        assert len(np.unique(model.labels_)) == 3

    def test_confidence_weighted_membership(self, tiny_data):
        model = ADEFCM(n_clusters=2, max_iter=5, random_state=42, verbose=False)
        U = np.array([[0.9, 0.1], [0.6, 0.4], [0.8, 0.2], [0.3, 0.7], [0.5, 0.5], [0.2, 0.8]])
        U_conf = model._confidence_weighted_membership(U)
        assert U_conf.shape == U.shape
        assert np.all(U_conf >= 0)

    def test_adaptive_fuzzifier_values(self):
        model = ADEFCM(n_clusters=2, max_iter=100, random_state=42, verbose=False)
        m0 = model._adaptive_fuzzifier(0)
        m99 = model._adaptive_fuzzifier(99)
        assert m0 > m99
        assert m0 > 1.0
        assert m99 >= 1.1

    def test_dynamic_threshold_values(self):
        model = ADEFCM(n_clusters=2, max_iter=100, random_state=42, verbose=False)
        eps0 = model._dynamic_threshold(0)
        eps99 = model._dynamic_threshold(99)
        assert eps0 > eps99
        assert eps99 >= 1e-8

    def test_fixed_m_and_epsilon(self, tiny_data):
        model = ADEFCM(n_clusters=2, m=2.0, epsilon=1e-3, max_iter=10, random_state=42, verbose=False)
        model.fit(tiny_data)
        assert model._adaptive_fuzzifier(0) == 2.0
        assert model._dynamic_threshold(0) == 1e-3


class TestAdaptiveParams:
    def test_adaptive_fuzzifier_decay(self):
        af = AdaptiveFuzzifier(m_min=1.1, m_max=2.5, alpha=3.0, max_iter=100)
        m0 = af(0)
        m50 = af(50)
        m99 = af(99)
        assert m0 == pytest.approx(2.5, abs=0.01)
        expected_end = 1.1 + (2.5 - 1.1) * np.exp(-3.0)
        assert m99 == pytest.approx(expected_end, abs=0.01)
        assert m0 > m50 > m99

    def test_adaptive_fuzzifier_schedule_length(self):
        af = AdaptiveFuzzifier(max_iter=50)
        sched = af.schedule()
        assert len(sched) == 50

    def test_adaptive_fuzzifier_summary(self):
        af = AdaptiveFuzzifier(max_iter=100)
        summary = af.summary()
        assert "m_start" in summary
        assert "m_end" in summary
        assert summary["m_start"] > summary["m_end"]

    def test_dynamic_threshold_decay(self):
        dt = DynamicThreshold(eps_0=1e-3, beta=5.0, max_iter=100)
        e0 = dt(0)
        e99 = dt(99)
        assert e0 == pytest.approx(1e-3, abs=1e-6)
        assert e99 == pytest.approx(1e-8, abs=1e-4)
        assert e0 > e99

    def test_dynamic_threshold_schedule_length(self):
        dt = DynamicThreshold(max_iter=50)
        sched = dt.schedule()
        assert len(sched) == 50

    def test_dynamic_threshold_summary(self):
        dt = DynamicThreshold(max_iter=100)
        summary = dt.summary()
        assert "eps_start" in summary
        assert "eps_end" in summary

    def test_early_stopping_not_triggered_initially(self):
        es = EarlyStopping(patience=3, verbose=False)
        assert not es.check(0.5, 0.1, 1)
        assert not es.stopped

    def test_early_stopping_triggers_after_patience(self):
        es = EarlyStopping(patience=3, verbose=False)
        assert not es.check(0.01, 0.1, 1)
        assert not es.check(0.01, 0.1, 2)
        assert es.check(0.02, 0.1, 3)
        assert es.check(0.05, 0.1, 4)
        assert es.stopped

    def test_early_stopping_resets_on_large_change(self):
        es = EarlyStopping(patience=3, verbose=False)
        es.check(0.01, 0.1, 1)
        es.check(0.5, 0.1, 2)
        es.check(0.01, 0.1, 3)
        es.check(0.01, 0.1, 4)
        assert not es.stopped

    def test_early_stopping_summary(self):
        es = EarlyStopping(patience=2, verbose=False)
        es.check(0.01, 0.1, 1)
        es.check(0.01, 0.1, 2)
        summary = es.summary()
        assert summary["patience"] == 2

    def test_early_stopping_reset(self):
        es = EarlyStopping(patience=2, verbose=False)
        es.check(0.01, 0.1, 1)
        es.check(0.01, 0.1, 2)
        assert es.stopped
        es.reset()
        assert not es.stopped
        assert es._counter == 0


class TestAutoCluster:
    def test_cluster_evaluator_silhouette(self, blobs_X, blobs_y):
        from sklearn.metrics import silhouette_score as sk_sil
        labels = blobs_y
        centers = np.array([blobs_X[blobs_y == k].mean(axis=0) for k in range(3)])
        evaluator = ClusterEvaluator(blobs_X, labels, centers)
        score = evaluator.silhouette_score()
        expected = sk_sil(blobs_X, labels)
        assert np.isclose(score, expected, atol=1e-4)

    def test_cluster_evaluator_davies_bouldin(self, blobs_X, blobs_y):
        labels = blobs_y
        centers = np.array([blobs_X[blobs_y == k].mean(axis=0) for k in range(3)])
        evaluator = ClusterEvaluator(blobs_X, labels, centers)
        score = evaluator.davies_bouldin_score()
        assert np.isfinite(score)
        assert score > 0

    def test_cluster_evaluator_bic(self, blobs_X, blobs_y):
        labels = blobs_y
        centers = np.array([blobs_X[blobs_y == k].mean(axis=0) for k in range(3)])
        evaluator = ClusterEvaluator(blobs_X, labels, centers)
        bic = evaluator.bic_score()
        assert np.isfinite(bic)

    def test_cluster_evaluator_gap(self, blobs_X, blobs_y):
        labels = blobs_y
        centers = np.array([blobs_X[blobs_y == k].mean(axis=0) for k in range(3)])
        evaluator = ClusterEvaluator(blobs_X, labels, centers)
        gap = evaluator.gap_statistic(n_reference=5)
        assert np.isfinite(gap)

    def test_cluster_evaluator_evaluate_all(self, blobs_X, blobs_y):
        labels = blobs_y
        centers = np.array([blobs_X[blobs_y == k].mean(axis=0) for k in range(3)])
        evaluator = ClusterEvaluator(blobs_X, labels, centers)
        metrics = evaluator.evaluate_all()
        assert "silhouette" in metrics
        assert "davies_bouldin" in metrics
        assert "bic" in metrics
        assert "gap" in metrics
        assert metrics["n_clusters"] == 3

    def test_consensus_search_returns_int(self, iris_X):
        discover = AutomaticClusterDiscovery(random_state=42)
        best_k = discover.consensus_search(iris_X, range(2, 6))
        assert isinstance(best_k, int)
        assert 2 <= best_k <= 5

    def test_search_returns_dict(self, iris_X):
        discover = AutomaticClusterDiscovery(random_state=42)
        results = discover.search(iris_X, [2, 3])
        assert isinstance(results, dict)
        assert 2 in results
        assert 3 in results


class TestXAI:
    def test_feature_importance_shift_shape(self, blobs_X, blobs_y):
        n_clusters = len(np.unique(blobs_y))
        centers = np.array([blobs_X[blobs_y == k].mean(axis=0) for k in range(n_clusters)])
        imp = feature_importance(blobs_X, blobs_y, centers, method="shift")
        assert imp.shape == (n_clusters, blobs_X.shape[1])

    def test_feature_importance_fisher_shape(self, blobs_X, blobs_y):
        n_clusters = len(np.unique(blobs_y))
        centers = np.array([blobs_X[blobs_y == k].mean(axis=0) for k in range(n_clusters)])
        imp = feature_importance(blobs_X, blobs_y, centers, method="fisher")
        assert imp.shape == (n_clusters, blobs_X.shape[1])

    def test_feature_importance_permutation_shape(self, blobs_X, blobs_y):
        n_clusters = len(np.unique(blobs_y))
        centers = np.array([blobs_X[blobs_y == k].mean(axis=0) for k in range(n_clusters)])
        imp = feature_importance(blobs_X, blobs_y, centers, method="permutation")
        assert imp.shape == (n_clusters, blobs_X.shape[1])

    def test_feature_importance_invalid_method(self, blobs_X, blobs_y):
        n_clusters = len(np.unique(blobs_y))
        centers = np.array([blobs_X[blobs_y == k].mean(axis=0) for k in range(n_clusters)])
        with pytest.raises(ValueError):
            feature_importance(blobs_X, blobs_y, centers, method="invalid")

    def test_cluster_summary_structure(self, blobs_X, blobs_y):
        n_clusters = len(np.unique(blobs_y))
        centers = np.array([blobs_X[blobs_y == k].mean(axis=0) for k in range(n_clusters)])
        summaries = cluster_summary(blobs_X, blobs_y, centers)
        assert len(summaries) == n_clusters
        for s in summaries:
            assert "cluster_id" in s
            assert "size" in s
            assert "proportion" in s
            assert "radius" in s
            assert "top_features" in s

    def test_cluster_summary_empty_cluster(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]])
        labels = np.array([0, 0, 1])
        centers = np.array([[0.5, 0.5], [10.0, 10.0], [5.0, 5.0]])
        summaries = cluster_summary(X, labels, centers)
        assert len(summaries) == 3
        assert summaries[2]["size"] == 0

    def test_describe_cluster_natural(self):
        summary = {
            "cluster_id": 0,
            "size": 50,
            "proportion": 0.5,
            "radius": 1.234,
            "top_features": [("f0", 0.9), ("f1", 0.5)],
        }
        desc = describe_cluster_natural(summary)
        assert "Cluster 0" in desc
        assert "50 points" in desc

    def test_shap_explain(self, tiny_data):
        labels = np.array([0, 0, 1, 1, 0, 1])
        centers = np.array([[1.2, 1.5], [7.0, 9.0]])
        shap_values, explanation = shap_explain(tiny_data, labels, centers, sample_idx=0)
        assert len(shap_values) == 2
        assert "Explanation for point 0" in explanation

    def test_explain_clusters(self, blobs_X, blobs_y):
        n_clusters = len(np.unique(blobs_y))
        centers = np.array([blobs_X[blobs_y == k].mean(axis=0) for k in range(n_clusters)])
        outlier_mask = np.zeros(len(blobs_X), dtype=bool)
        outlier_mask[:2] = True
        expl = explain_clusters(blobs_X, blobs_y, centers, outlier_mask=outlier_mask)
        assert expl["n_clusters"] == n_clusters
        assert expl["n_outliers"] == 2
        assert len(expl["summaries"]) == n_clusters
        assert len(expl["descriptions"]) == n_clusters


class TestOutlierDetector:
    def test_weighted_distance_scores(self, tiny_data):
        X = tiny_data
        U = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        centers = np.array([[1.5, 1.8], [7.0, 9.0]])
        detector = OutlierDetector(method="weighted_distance", threshold_multiplier=2.0)
        detector.fit(X, U, centers, m=2.0)
        assert detector.outlier_scores_ is not None
        assert detector.outlier_mask_ is not None
        assert len(detector.outlier_scores_) == len(X)
        assert len(detector.outlier_mask_) == len(X)

    def test_entropy_method(self, tiny_data):
        X = tiny_data
        U = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        centers = np.array([[1.5, 1.8], [7.0, 9.0]])
        detector = OutlierDetector(method="entropy", threshold_multiplier=2.0)
        detector.fit(X, U, centers, m=2.0)
        assert detector.outlier_scores_ is not None
        assert np.all(detector.outlier_scores_ >= 0)

    def test_lof_method(self, tiny_data):
        X = tiny_data
        U = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        centers = np.array([[1.5, 1.8], [7.0, 9.0]])
        detector = OutlierDetector(method="lof", threshold_multiplier=2.0)
        detector.fit(X, U, centers, m=2.0)
        assert detector.outlier_scores_ is not None

    def test_contamination_based(self, tiny_data):
        X = tiny_data
        U = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        centers = np.array([[1.5, 1.8], [7.0, 9.0]])
        detector = OutlierDetector(method="weighted_distance", contamination=0.2)
        detector.fit(X, U, centers, m=2.0)
        assert detector.outlier_mask_.sum() >= 1

    def test_predict_with_scores(self):
        detector = OutlierDetector(method="weighted_distance", threshold_multiplier=2.0)
        scores = np.array([1.0, 1.1, 10.0, 1.2, 12.0])
        mask, out_scores = detector.predict(scores=scores)
        assert len(mask) == 5
        assert np.array_equal(out_scores, scores)

    def test_predict_raises_without_args(self):
        detector = OutlierDetector()
        with pytest.raises(ValueError):
            detector.predict()

    def test_invalid_method(self, tiny_data):
        X = tiny_data
        U = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        centers = np.array([[1.5, 1.8], [7.0, 9.0]])
        detector = OutlierDetector(method="invalid")
        with pytest.raises(ValueError):
            detector.fit(X, U, centers)

    def test_outlier_summary(self, tiny_data):
        X = tiny_data
        U = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        centers = np.array([[1.5, 1.8], [7.0, 9.0]])
        detector = OutlierDetector(method="weighted_distance")
        detector.fit(X, U, centers)
        summary = detector.outlier_summary()
        assert summary["method"] == "weighted_distance"
        assert summary["n_total"] == len(X)
