import numpy as np
import pandas as pd
import pytest

from baseline_project.data_loader import DataLoader
from baseline_project.preprocessing import Preprocessor
from baseline_project.distance_metrics import (
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    cosine_similarity,
    squared_euclidean,
    minkowski_distance,
    pairwise_distance_matrix,
    point_to_centers_distance,
)
from baseline_project.membership_update import MembershipUpdater
from baseline_project.cluster_update import ClusterUpdater
from baseline_project.objective_function import ObjectiveFunction
from baseline_project.convergence import ConvergenceChecker


class TestDataLoader:
    def test_load_benchmark_dataset_iris(self):
        loader = DataLoader()
        X, y = loader.load_benchmark_dataset("iris")
        assert X.shape == (150, 4)
        assert y.shape == (150,)
        assert len(np.unique(y)) == 3

    def test_load_benchmark_dataset_wine(self):
        loader = DataLoader()
        X, y = loader.load_benchmark_dataset("wine")
        assert X.shape == (178, 13)
        assert y.shape == (178,)

    def test_load_benchmark_dataset_invalid(self):
        loader = DataLoader()
        with pytest.raises(ValueError, match="Unknown dataset"):
            loader.load_benchmark_dataset("nonexistent")

    def test_load_synthetic_data_default(self):
        loader = DataLoader()
        X, y = loader.load_synthetic_data(n_samples=100, n_features=5, n_clusters=3)
        assert X.shape == (100, 5)
        assert y.shape == (100,)
        assert len(np.unique(y)) == 3

    def test_load_synthetic_data_small(self):
        loader = DataLoader()
        X, y = loader.load_synthetic_data(n_samples=10, n_features=2, n_clusters=2, noise=0.01)
        assert X.shape == (10, 2)
        assert y.shape == (10,)


class TestPreprocessor:
    def test_clean_weblog_data_removes_robots(self, sample_weblog_df):
        pre = Preprocessor()
        cleaned = pre.clean_weblog_data(sample_weblog_df)
        assert "python-requests" not in cleaned["user_agent"].values
        assert "curl" not in cleaned["user_agent"].values

    def test_clean_weblog_data_removes_4xx_status(self):
        pre = Preprocessor()
        df = pd.DataFrame({
            "url": ["/a", "/b"],
            "status": [200, 404],
            "method": ["GET", "GET"],
            "user_agent": ["Mozilla/5.0", "Mozilla/5.0"],
        })
        cleaned = pre.clean_weblog_data(df)
        assert len(cleaned) == 1

    def test_identify_users_by_ip(self, sample_weblog_df):
        pre = Preprocessor()
        df = pre.identify_users(sample_weblog_df)
        assert "user_id" in df.columns
        assert df["user_id"].nunique() == 3

    def test_identify_users_fallback(self):
        pre = Preprocessor()
        df = pd.DataFrame({"a": [1, 2]})
        result = pre.identify_users(df)
        assert result["user_id"].nunique() == 1

    def test_identify_sessions_with_timeout(self, sample_weblog_df):
        pre = Preprocessor()
        df = pre.identify_users(sample_weblog_df)
        df = pre.identify_sessions(df, timeout_minutes=30)
        assert "session_id" in df.columns
        assert df["session_id"].nunique() >= 2

    def test_identify_sessions_no_timestamp(self):
        pre = Preprocessor()
        df = pd.DataFrame({"user_id": [0, 0]})
        result = pre.identify_sessions(df)
        assert "session_id" in result.columns

    def test_reduce_dimensions(self):
        pre = Preprocessor()
        df = pd.DataFrame({"url": ["/a", "/a", "/b", "/c"], "user_id": [0, 0, 0, 0]})
        result = pre.reduce_dimensions(df, min_support=2)
        assert "/b" not in result["url"].values
        assert "/c" not in result["url"].values
        assert len(result) == 2

    def test_build_session_matrix(self, sample_weblog_df):
        pre = Preprocessor()
        df = pre.identify_users(sample_weblog_df)
        df = pre.identify_sessions(df)
        mat, feature_names = pre.build_session_matrix(df)
        assert mat.shape[0] == df["session_id"].nunique()
        assert len(feature_names) > 0
        assert hasattr(mat, "toarray")

    def test_standardize(self):
        pre = Preprocessor()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = pre.standardize(X)
        assert np.allclose(result.mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(result.std(axis=0), 1.0, atol=1e-10)

    def test_min_max_scale(self):
        pre = Preprocessor()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = pre.min_max_scale(X)
        assert np.allclose(result.min(axis=0), 0.0)
        assert np.allclose(result.max(axis=0), 1.0)


class TestDistanceMetrics:
    def test_euclidean_1d(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        d = euclidean_distance(a, b)
        assert np.isclose(d, np.sqrt(27))

    def test_euclidean_2d(self):
        a = np.array([[0.0, 0.0], [0.0, 0.0]])
        b = np.array([[3.0, 4.0], [6.0, 8.0]])
        d = euclidean_distance(a, b)
        assert d.shape == (2, 2)
        assert np.isclose(d[0, 0], 5.0)
        assert np.isclose(d[0, 1], 10.0)

    def test_manhattan_1d(self):
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 6.0])
        d = manhattan_distance(a, b)
        assert np.isclose(d, 7.0)

    def test_chebyshev_1d(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 10.0])
        d = chebyshev_distance(a, b)
        assert np.isclose(d, 7.0)

    def test_cosine_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        d = cosine_similarity(a, a)
        assert np.isclose(d, 0.0)

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        d = cosine_similarity(a, b)
        assert np.isclose(d, 1.0)

    def test_squared_euclidean(self):
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 6.0])
        d = squared_euclidean(a, b)
        assert np.isclose(d, 25.0)

    def test_minkowski_p1(self):
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 6.0])
        d = minkowski_distance(a, b, p=1)
        assert np.isclose(d, 7.0)

    def test_minkowski_p2(self):
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 6.0])
        d = minkowski_distance(a, b, p=2)
        assert np.isclose(d, 5.0)

    def test_minkowski_pinf(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 10.0])
        d = minkowski_distance(a, b, p=np.inf)
        assert np.isclose(d, 7.0)

    def test_pairwise_distance_matrix(self):
        X = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        D = pairwise_distance_matrix(X, metric="euclidean")
        assert D.shape == (3, 3)
        assert np.isclose(D[0, 0], 0.0)
        assert np.isclose(D[0, 1], 5.0)

    def test_point_to_centers_distance(self):
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        centers = np.array([[0.0, 0.0]])
        D = point_to_centers_distance(X, centers, metric="euclidean")
        assert D.shape == (2, 1)
        assert np.isclose(D[0, 0], 0.0)
        assert np.isclose(D[1, 0], 5.0)


class TestMembershipUpdate:
    def test_update_fcm_shape(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        centers = np.array([[2.0, 3.0], [6.0, 7.0]])
        U = MembershipUpdater.update_fcm(X, centers, m=2.0)
        assert U.shape == (4, 2)

    def test_update_fcm_row_sums_one(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        centers = np.array([[2.0, 3.0], [5.0, 6.0]])
        U = MembershipUpdater.update_fcm(X, centers, m=2.0)
        assert np.allclose(U.sum(axis=1), 1.0)

    def test_update_fclm_shape(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        centers = np.array([[2.0, 3.0], [4.0, 5.0]])
        U = MembershipUpdater.update_fclm(X, centers, m=2.0)
        assert U.shape == (2, 2)

    def test_update_fclm_row_sums_one(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        U = MembershipUpdater.update_fclm(X, centers, m=2.0)
        assert np.allclose(U.sum(axis=1), 1.0)

    def test_initialize_random_shape(self):
        U = MembershipUpdater.initialize_random(10, 3, random_state=42)
        assert U.shape == (10, 3)

    def test_initialize_random_row_sums_one(self):
        U = MembershipUpdater.initialize_random(10, 3, random_state=42)
        assert np.allclose(U.sum(axis=1), 1.0)

    def test_initialize_random_reproducible(self):
        U1 = MembershipUpdater.initialize_random(5, 2, random_state=42)
        U2 = MembershipUpdater.initialize_random(5, 2, random_state=42)
        assert np.allclose(U1, U2)

    def test_initialize_uniform_shape(self):
        U = MembershipUpdater.initialize_uniform(10, 4)
        assert U.shape == (10, 4)

    def test_initialize_uniform_values(self):
        U = MembershipUpdater.initialize_uniform(10, 4)
        assert np.allclose(U, 0.25)
        assert np.allclose(U.sum(axis=1), 1.0)


class TestClusterUpdate:
    def test_update_centers_fcm_shape(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        U = MembershipUpdater.initialize_uniform(3, 2)
        centers = ClusterUpdater.update_centers_fcm(X, U, m=2.0)
        assert centers.shape == (2, 2)

    def test_update_centers_fcm_values(self):
        X = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0]])
        U = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
        centers = ClusterUpdater.update_centers_fcm(X, U, m=2.0)
        assert centers[0, 0] < 5.0
        assert centers[1, 0] > 5.0

    def test_update_centers_fclm_shape(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        U = MembershipUpdater.initialize_uniform(3, 2)
        centers = ClusterUpdater.update_centers_fclm(X, U, m=2.0)
        assert centers.shape == (2, 2)

    def test_update_centers_fclm_selects_from_data(self):
        X = np.array([[1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0]])
        U = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
        centers = ClusterUpdater.update_centers_fclm(X, U, m=2.0)
        for c in centers:
            assert any(np.allclose(c, row) for row in X)

    def test_initialize_kmeans_plus_plus_shape(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        centers = ClusterUpdater.initialize_kmeans_plus_plus(X, 3, random_state=42)
        assert centers.shape == (3, 2)

    def test_initialize_kmeans_plus_plus_selects_from_data(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        centers = ClusterUpdater.initialize_kmeans_plus_plus(X, 2, random_state=42)
        for c in centers:
            assert any(np.allclose(c, row) for row in X)


class TestObjectiveFunction:
    def test_compute_fcm_positive(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]])
        centers = np.array([[0.5, 0.5], [9.0, 9.0]])
        U = MembershipUpdater.initialize_uniform(3, 2)
        J = ObjectiveFunction.compute_fcm(X, U, centers, m=2.0)
        assert J > 0.0

    def test_compute_fcm_zero_for_perfect_fit(self):
        X = np.array([[0.0, 0.0], [0.0, 0.0]])
        centers = np.array([[0.0, 0.0]])
        U = np.array([[1.0], [1.0]])
        J = ObjectiveFunction.compute_fcm(X, U, centers, m=2.0)
        assert np.isclose(J, 0.0)

    def test_compute_fclm_delegates(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        centers = np.array([[0.5, 0.5]])
        U = np.array([[1.0], [1.0]])
        J1 = ObjectiveFunction.compute_fcm(X, U, centers, m=2.0)
        J2 = ObjectiveFunction.compute_fclm(X, U, centers, m=2.0)
        assert np.isclose(J1, J2)

    def test_compute_partition_coefficient(self):
        U = np.array([[1.0, 0.0], [0.0, 1.0]])
        pc = ObjectiveFunction.compute_partition_coefficient(U)
        assert np.isclose(pc, 1.0)

    def test_compute_partition_entropy(self):
        U = np.array([[1.0, 0.0], [0.5, 0.5]])
        pe = ObjectiveFunction.compute_partition_entropy(U)
        assert pe > 0.0

    def test_compute_sse(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]])
        labels = np.array([0, 0, 1])
        centers = np.array([[0.5, 0.5], [10.0, 10.0]])
        sse = ObjectiveFunction.compute_sse(X, labels, centers)
        assert np.isclose(sse, 0.5 + 0.5 + 0.0)


class TestConvergenceChecker:
    def test_check_converged(self):
        U_new = np.array([[0.8, 0.2], [0.3, 0.7]])
        U_old = np.array([[0.8, 0.2], [0.3, 0.7]])
        assert ConvergenceChecker.check(U_new, U_old, epsilon=1e-5)

    def test_check_not_converged(self):
        U_new = np.array([[1.0, 0.0], [0.0, 1.0]])
        U_old = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert not ConvergenceChecker.check(U_new, U_old, epsilon=1e-5)

    def test_compute_change_zero(self):
        U = np.array([[0.5, 0.5], [0.5, 0.5]])
        change = ConvergenceChecker.compute_change(U, U)
        assert np.isclose(change, 0.0)

    def test_compute_change_nonzero(self):
        U_new = np.array([[1.0, 0.0], [0.0, 1.0]])
        U_old = np.array([[0.5, 0.5], [0.5, 0.5]])
        change = ConvergenceChecker.compute_change(U_new, U_old)
        assert change > 0.0

    def test_check_objective_converged(self):
        assert ConvergenceChecker.check_objective(100.0, 100.0, epsilon=1e-6)

    def test_check_objective_not_converged(self):
        assert not ConvergenceChecker.check_objective(100.0, 0.0, epsilon=1e-6)

    def test_early_stopping_not_enough_history(self):
        assert not ConvergenceChecker.early_stopping([1.0, 2.0], patience=5)

    def test_early_stopping_triggered(self):
        J = [10.0, 9.9, 9.8, 9.7, 9.6, 9.5]
        assert ConvergenceChecker.early_stopping(J, patience=3, min_delta=1.0)

    def test_early_stopping_not_triggered(self):
        J = [10.0, 5.0, 9.0, 3.0, 8.0]
        assert not ConvergenceChecker.early_stopping(J, patience=3, min_delta=0.5)
