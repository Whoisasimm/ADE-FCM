from streaming.online_clustering import OnlineFCM

import numpy as np
import pytest


class TestOnlineFCM:
    def test_initialization(self):
        model = OnlineFCM(n_clusters=3, m=2.0, learning_rate=0.3, random_state=42)
        assert model.n_clusters == 3
        assert model.m == 2.0
        assert model.lr == 0.3
        assert model.centers_ is None
        assert model.n_samples_ == 0

    def test_partial_fit_initializes_centers(self):
        model = OnlineFCM(n_clusters=2, random_state=42)
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        model.partial_fit(X)
        assert model.centers_ is not None
        assert model.centers_.shape == (2, 2)
        assert model.n_samples_ == 3

    def test_partial_fit_incremental(self):
        model = OnlineFCM(n_clusters=2, random_state=42)
        X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        model.partial_fit(X1)
        centers_after_first = model.centers_.copy()
        model.partial_fit(X2)
        assert model.n_samples_ == 4
        assert not np.allclose(model.centers_, centers_after_first)

    def test_partial_fit_1d_input(self):
        model = OnlineFCM(n_clusters=2, random_state=42)
        X = np.array([1.0, 2.0, 3.0, 4.0])
        model.partial_fit(X)
        assert model.centers_ is not None

    def test_partial_fit_empty(self):
        model = OnlineFCM(n_clusters=2, random_state=42)
        model.partial_fit(np.array([]))
        assert model.centers_ is None

    def test_predict_after_partial_fit(self):
        model = OnlineFCM(n_clusters=2, random_state=42)
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 11.0], [12.0, 13.0]])
        model.partial_fit(X_train)
        X_test = np.array([[1.5, 2.5], [11.0, 12.0]])
        labels = model.predict(X_test)
        assert labels.shape == (2,)
        assert labels[0] != labels[1]

    def test_predict_without_fit(self):
        model = OnlineFCM(n_clusters=2)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = model.predict(X)
        assert np.array_equal(labels, np.zeros(2))

    def test_get_membership(self):
        model = OnlineFCM(n_clusters=2, random_state=42)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        model.partial_fit(X)
        U = model.get_membership(X)
        assert U is not None
        assert U.shape == (2, 2)
        assert np.allclose(U.sum(axis=1), 1.0)

    def test_get_membership_without_fit(self):
        model = OnlineFCM(n_clusters=2)
        X = np.array([[1.0, 2.0]])
        U = model.get_membership(X)
        assert U is None

    def test_fit_full_dataset(self):
        model = OnlineFCM(n_clusters=3, random_state=42)
        X = np.random.RandomState(42).randn(250, 4)
        model.fit(X)
        assert model.centers_ is not None
        assert model.n_samples_ == 250

    def test_reset(self):
        model = OnlineFCM(n_clusters=2, random_state=42)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        model.partial_fit(X)
        assert model.centers_ is not None
        model.reset()
        assert model.centers_ is None
        assert model.n_samples_ == 0
        assert model.U_history_ == []

    def test_center_update_convergence(self):
        model = OnlineFCM(n_clusters=2, learning_rate=0.5, random_state=42)
        X = np.array([[1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0]])
        model.partial_fit(X[:2])
        initial = model.centers_.copy()
        for _ in range(5):
            model.partial_fit(X)
        assert not np.allclose(model.centers_, initial)
