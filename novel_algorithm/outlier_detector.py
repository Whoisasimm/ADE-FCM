"""
Outlier Detector for ADE-FCM
=============================
Implements Contribution 6: Outlier Robust Membership.

Provides a standalone OutlierDetector class that identifies outliers
based on weighted fuzzy membership-distance scores and statistical
thresholding. Integrates with ADE-FCM's _outlier_detection method.

Methods:
  - Weighted distance score (primary)
  - Local outlier factor (LOF) approximation
  - Membership entropy based
"""

import numpy as np
from scipy.spatial.distance import cdist
from loguru import logger


class OutlierDetector:
    """Detect outliers in fuzzy clustering results.

    Parameters
    ----------
    method : {'weighted_distance', 'entropy', 'lof'}, default='weighted_distance'
        Detection method.
    threshold_multiplier : float, default=2.0
        Standard deviation multiplier for thresholding.
    contamination : float or None, default=None
        If set (0 < contamination <= 0.5), flags the top
        contamination fraction as outliers instead of using
        the statistical threshold.
    """

    def __init__(
        self,
        method="weighted_distance",
        threshold_multiplier=2.0,
        contamination=None,
    ):
        self.method = method
        self.threshold_multiplier = threshold_multiplier
        self.contamination = contamination
        self.outlier_mask_ = None
        self.outlier_scores_ = None
        self.threshold_ = None

    # ----- Weighted Distance Score ------------------------------------------
    def _weighted_distance_scores(self, X, U, centers, m):
        """Compute outlier scores as sum_j u_ij^m * ||x_i - c_j||.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        U : ndarray of shape (n_samples, n_clusters)
        centers : ndarray of shape (n_clusters, n_features)
        m : float

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        Um = U ** m
        n_clusters = centers.shape[0]
        scores = np.zeros(X.shape[0])
        for j in range(n_clusters):
            diff = X - centers[j]
            dists_j = np.sqrt(np.sum(diff ** 2, axis=1))
            scores += Um[:, j] * dists_j
        return scores

    # ----- Membership Entropy Score -----------------------------------------
    def _entropy_scores(self, U):
        """Compute outlier scores based on membership entropy.

        High entropy -> uncertain assignment -> potential outlier.

        Parameters
        ----------
        U : ndarray of shape (n_samples, n_clusters)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        U_safe = np.maximum(U, 1e-10)
        entropy = -np.sum(U_safe * np.log(U_safe), axis=1)
        # Normalize by log(n_clusters) to get [0, 1] range
        entropy = entropy / np.log(U.shape[1] + 1)
        return entropy

    # ----- LOF Approximation -------------------------------------------------
    def _lof_scores(self, X, k=20):
        """Approximate Local Outlier Factor using k-nearest neighbors.

        A simplified LOF: ratio of average k-NN distance to a point's
        own average distance to its k-NN.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        k : int, default=20

        Returns
        -------
        lof : ndarray of shape (n_samples,)
        """
        n = X.shape[0]
        k = min(k, n - 1)
        # Pairwise distances
        dists = cdist(X, X, metric="euclidean")
        np.fill_diagonal(dists, np.inf)

        lof = np.zeros(n)
        for i in range(n):
            # k-nearest neighbors of i
            nn_idx = np.argpartition(dists[i], k)[:k]
            nn_dists = dists[i, nn_idx]
            # Average distance of i to its NN
            d_i = np.mean(nn_dists) if k > 0 else 0.0
            # Average k-NN distance of each neighbor
            d_nn_avg = 0.0
            for j in nn_idx:
                j_nn = np.argpartition(dists[j], k)[:k]
                d_nn_avg += np.mean(dists[j, j_nn])
            d_nn_avg = d_nn_avg / k if k > 0 else 1.0
            lof[i] = d_i / max(d_nn_avg, 1e-10)
        return lof

    # -------------------------------------------------------------------------
    def _determine_threshold(self, scores):
        """Determine the cut-off threshold.

        Uses either statistical (mean + k*std) or contamination-based.

        Parameters
        ----------
        scores : ndarray

        Returns
        -------
        threshold : float
        mask : ndarray of bool
        """
        if self.contamination is not None:
            # Top contamination fraction
            n_out = max(1, int(len(scores) * self.contamination))
            threshold = np.sort(scores)[-n_out]
            mask = scores >= threshold
        else:
            mu = np.mean(scores)
            sigma = np.std(scores)
            threshold = mu + self.threshold_multiplier * sigma
            mask = scores > threshold
        return threshold, mask

    # -------------------------------------------------------------------------
    def fit(self, X, U, centers, m=2.0):
        """Fit outlier detector and flag outliers.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        U : ndarray of shape (n_samples, n_clusters)
        centers : ndarray of shape (n_clusters, n_features)
        m : float, default=2.0
            Fuzzifier used during clustering.

        Returns
        -------
        self : OutlierDetector
        """
        n = X.shape[0]

        if self.method == "weighted_distance":
            scores = self._weighted_distance_scores(X, U, centers, m)
        elif self.method == "entropy":
            scores = self._entropy_scores(U)
        elif self.method == "lof":
            scores = self._lof_scores(X)
        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")

        self.outlier_scores_ = scores
        self.threshold_, self.outlier_mask_ = self._determine_threshold(scores)

        n_out = int(self.outlier_mask_.sum())
        logger.info(
            f"OutlierDetector ({self.method}): "
            f"{n_out}/{n} points flagged, threshold={self.threshold_:.4f}"
        )
        return self

    def predict(self, X=None, U=None, centers=None, scores=None):
        """Return outlier flags for given scores or new data.

        Parameters
        ----------
        X : ndarray or None
        U : ndarray or None
        centers : ndarray or None
        scores : ndarray or None
            Direct outlier scores (bypasses computation).

        Returns
        -------
        mask : ndarray of bool
        scores : ndarray
        """
        if scores is not None:
            _, mask = self._determine_threshold(scores)
            return mask, scores
        if X is not None and U is not None and centers is not None:
            # Use already known parameters
            return self.fit(X, U, centers).outlier_mask_, self.outlier_scores_
        raise ValueError(
            "Provide either 'scores' or (X, U, centers) to predict."
        )

    def outlier_summary(self):
        """Return a summary of the detected outliers.

        Returns
        -------
        summary : dict
        """
        if self.outlier_mask_ is None:
            return {"error": "Not fitted yet."}
        return {
            "method": self.method,
            "threshold_multiplier": self.threshold_multiplier,
            "threshold": float(self.threshold_),
            "n_outliers": int(self.outlier_mask_.sum()),
            "n_total": len(self.outlier_mask_),
            "outlier_ratio": float(self.outlier_mask_.mean()),
            "score_mean": float(np.mean(self.outlier_scores_)),
            "score_std": float(np.std(self.outlier_scores_)),
            "score_max": float(np.max(self.outlier_scores_)),
        }
