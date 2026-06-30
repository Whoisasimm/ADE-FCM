import numpy as np
from loguru import logger

class MembershipUpdater:
    """Membership matrix update for FCM and FCLM."""
    
    @staticmethod
    def update_fcm(X, centers, m=2.0):
        """
        Update membership matrix U using FCM formula (paper Eq 3).
        u_ij = 1 / sum_k (d_ij/d_ik)^(2/(m-1))
        """
        n = X.shape[0]
        c = centers.shape[0]
        # Compute pairwise distances: n x c
        dist = np.zeros((n, c))
        for j in range(c):
            diff = X - centers[j]
            dist[:, j] = np.sqrt(np.sum(diff**2, axis=1))
        # Avoid division by zero
        dist = np.maximum(dist, 1e-10)
        # Compute membership
        inv = 2.0 / (m - 1)
        U = np.zeros((n, c))
        for i in range(n):
            for j in range(c):
                denom = np.sum((dist[i, j] / np.maximum(dist[i, :], 1e-10)) ** inv)
                U[i, j] = 1.0 / denom if denom > 0 else 0.0
        return U
    
    @staticmethod
    def update_fclm(X, centers, m=2.0):
        """Same as FCM membership (paper uses same formula)."""
        return MembershipUpdater.update_fcm(X, centers, m)
    
    @staticmethod
    def initialize_random(n_samples, n_clusters, random_state=42):
        """Random membership initialization."""
        rng = np.random.RandomState(random_state)
        U = rng.rand(n_samples, n_clusters)
        U = U / U.sum(axis=1, keepdims=True)
        return U
    
    @staticmethod
    def initialize_uniform(n_samples, n_clusters):
        """Uniform membership initialization."""
        return np.ones((n_samples, n_clusters)) / n_clusters
