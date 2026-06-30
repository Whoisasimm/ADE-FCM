import numpy as np
from loguru import logger

class ClusterUpdater:
    """Cluster center update for FCM and FCLM."""
    
    @staticmethod
    def update_centers_fcm(X, U, m=2.0):
        """
        Update centers using FCM formula (paper Eq 5).
        v_j = sum_i u_ij^m * x_i / sum_i u_ij^m
        """
        Um = U ** m
        numerator = X.T @ Um  # d x c
        denominator = Um.sum(axis=0)  # c
        denominator = np.maximum(denominator, 1e-10)
        centers = (numerator / denominator).T  # c x d
        return centers
    
    @staticmethod
    def update_centers_fclm(X, U, m=2.0):
        """
        FCLM center update with median-based selection (paper Eq 4, 6).
        D_i = Median(D_ij(S_k - S_i) * u_ij)
        p = Argmin(D_i : n)
        """
        n, d = X.shape
        c = U.shape[1]
        # First compute weighted mean centers (same as FCM)
        Um = U ** m
        numerator = X.T @ Um
        denominator = np.maximum(Um.sum(axis=0), 1e-10)
        weighted_centers = (numerator / denominator).T
        
        # For FCLM, select median-based representative points
        centers = np.zeros_like(weighted_centers)
        for j in range(c):
            # Compute weighted distances from all points to weighted center
            diff = X - weighted_centers[j]
            dists = np.sqrt(np.sum(diff**2, axis=1))
            weighted_dists = dists * U[:, j]
            # Find median index
            median_idx = np.argsort(weighted_dists)[len(weighted_dists) // 2]
            centers[j] = X[median_idx]
        
        return centers
    
    @staticmethod
    def initialize_kmeans_plus_plus(X, n_clusters, random_state=42):
        """KMeans++ initialization."""
        rng = np.random.RandomState(random_state)
        n = X.shape[0]
        centers = [X[rng.randint(n)]]
        for _ in range(1, n_clusters):
            dists = np.array([min(np.linalg.norm(x - c) for c in centers) for x in X])
            probs = dists ** 2 / np.sum(dists ** 2)
            next_idx = rng.choice(n, p=probs)
            centers.append(X[next_idx])
        return np.array(centers)
    
    @staticmethod
    def initialize_random(X, n_clusters, random_state=42):
        """Random center initialization."""
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), n_clusters, replace=False)
        return X[idx].copy()
