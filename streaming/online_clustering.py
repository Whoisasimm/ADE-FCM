import numpy as np
from loguru import logger


class OnlineFCM:
    """Online Fuzzy C-Means for streaming data with incremental updates.

    Supports partial_fit() for incremental learning over mini-batches,
    making it suitable for real-time clustering of unbounded data streams.
    """

    def __init__(self, n_clusters=5, m=2.0, learning_rate=0.3, random_state=42):
        self.n_clusters = n_clusters
        self.m = m
        self.lr = learning_rate
        self.random_state = random_state
        self.centers_ = None
        self.n_samples_ = 0
        self.U_history_ = []

    def partial_fit(self, X, y=None):
        """Update model incrementally with a single batch of data."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if len(X) == 0:
            return self

        if self.centers_ is None:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X), min(self.n_clusters, len(X)), replace=False)
            self.centers_ = X[idx].copy().astype(np.float64)
            self.n_samples_ = len(X)
            return self

        n, c = X.shape[0], self.centers_.shape[0]
        dist = np.zeros((n, c))
        for j in range(c):
            diff = X - self.centers_[j]
            dist[:, j] = np.sqrt(np.sum(diff ** 2, axis=1))
        dist = np.maximum(dist, 1e-10)

        inv = 2.0 / (self.m - 1)
        U = np.zeros((n, c))
        for i in range(n):
            for j in range(c):
                ratio = dist[i, j] / dist[i, :]
                denom = np.sum(ratio ** inv)
                U[i, j] = 1.0 / denom if denom > 0 else 0.0

        Um = U ** self.m
        for j in range(c):
            weighted_sum = np.sum(Um[:, j, np.newaxis] * X, axis=0)
            weight = np.sum(Um[:, j])
            if weight > 1e-10:
                batch_center = weighted_sum / weight
                self.centers_[j] = (1 - self.lr) * self.centers_[j] + self.lr * batch_center

        self.n_samples_ += len(X)
        self.U_history_.append(U)
        return self

    def fit(self, X, y=None):
        """Fit on full dataset by iterating mini-batches."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        for i in range(0, len(X), 100):
            self.partial_fit(X[i:i + 100])
        return self

    def predict(self, X):
        """Predict cluster assignments for new data points."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.centers_ is None:
            return np.zeros(len(X))
        dist = np.zeros((X.shape[0], self.centers_.shape[0]))
        for j in range(self.centers_.shape[0]):
            diff = X - self.centers_[j]
            dist[:, j] = np.sqrt(np.sum(diff ** 2, axis=1))
        return np.argmin(dist, axis=1)

    def get_membership(self, X):
        """Compute membership matrix for new data against current centers."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.centers_ is None:
            return None
        n, c = X.shape[0], self.centers_.shape[0]
        dist = np.zeros((n, c))
        for j in range(c):
            diff = X - self.centers_[j]
            dist[:, j] = np.sqrt(np.sum(diff ** 2, axis=1))
        dist = np.maximum(dist, 1e-10)
        inv = 2.0 / (self.m - 1)
        U = np.zeros((n, c))
        for i in range(n):
            for j in range(c):
                ratio = dist[i, j] / dist[i, :]
                denom = np.sum(ratio ** inv)
                U[i, j] = 1.0 / denom if denom > 0 else 0.0
        return U

    def reset(self):
        """Reset the model state for a fresh stream."""
        self.centers_ = None
        self.n_samples_ = 0
        self.U_history_ = []
        logger.info("OnlineFCM model reset")
