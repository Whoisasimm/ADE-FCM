"""
Automatic Cluster Discovery for ADE-FCM
=========================================
Implements Contribution 5: Automatic Cluster Discovery.

Provides multiple internal validation indices and a consensus mechanism
to automatically determine the optimal number of clusters.

Indices implemented:
  - Silhouette Score
  - Gap Statistic (Tibshirani et al. 2001)
  - Davies-Bouldin Index
  - Bayesian Information Criterion (BIC) for GMM
  - Consensus search that aggregates all indices
"""

import numpy as np
from scipy.spatial.distance import cdist
from loguru import logger


class ClusterEvaluator:
    """Compute multiple cluster validity indices for a given data and labels.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    labels : ndarray of shape (n_samples,)
    centers : ndarray of shape (n_clusters, n_features)
    """

    def __init__(self, X, labels, centers):
        self.X = X
        self.labels = labels
        self.centers = centers
        self.n_clusters = len(centers)
        self.n = X.shape[0]

    # ----- Silhouette Score -------------------------------------------------
    def silhouette_score(self):
        """Compute the mean Silhouette Coefficient.

        Returns
        -------
        s : float
            Mean silhouette score (-1 to 1, higher is better).
        """
        if self.n_clusters < 2 or self.n_clusters >= self.n:
            return -1.0
        dists = cdist(self.X, self.X, metric="euclidean")
        np.fill_diagonal(dists, 0.0)
        sil_values = np.zeros(self.n)

        for i in range(self.n):
            same_cluster = self.labels == self.labels[i]
            same_count = np.sum(same_cluster)
            if same_count <= 1:
                sil_values[i] = 0.0
                continue
            # Mean intra-cluster distance
            a_i = np.sum(dists[i, same_cluster]) / (same_count - 1)
            # Mean nearest-cluster distance
            b_i = float("inf")
            for k in range(self.n_clusters):
                if k == self.labels[i]:
                    continue
                other = self.labels == k
                if np.sum(other) == 0:
                    continue
                mean_dist = np.mean(dists[i, other])
                b_i = min(b_i, mean_dist)
            sil_values[i] = (b_i - a_i) / max(a_i, b_i, 1e-10)

        return float(np.mean(sil_values))

    # ----- Davies-Bouldin Index ---------------------------------------------
    def davies_bouldin_score(self):
        """Compute the Davies-Bouldin Index.

        Returns
        -------
        db : float
            Davies-Bouldin score (lower is better).
        """
        if self.n_clusters < 2:
            return float("inf")
        # Cluster dispersions
        sigmas = np.zeros(self.n_clusters)
        for k in range(self.n_clusters):
            mask = self.labels == k
            if np.sum(mask) == 0:
                sigmas[k] = 0.0
            else:
                sigmas[k] = np.mean(
                    np.sqrt(np.sum((self.X[mask] - self.centers[k]) ** 2, axis=1))
                )
        # Pairwise center distances
        center_dists = cdist(self.centers, self.centers, metric="euclidean")
        np.fill_diagonal(center_dists, 1.0)  # avoid div-by-zero

        R = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            R[i] = np.max(
                (sigmas[i] + sigmas) / np.maximum(center_dists[i], 1e-10)
            )
        return float(np.mean(R))

    # ----- Gap Statistic -----------------------------------------------------
    def gap_statistic(self, n_reference=50):
        """Compute the Gap Statistic (Tibshirani et al. 2001).

        Parameters
        ----------
        n_reference : int, default=50
            Number of reference null reference datasets.

        Returns
        -------
        gap : float
            Gap value (higher gap -> better K).
        """
        # Observed dispersion
        Wk = self._log_dispersion(self.X, self.labels, self.centers)

        # Reference dispersion (uniform over bounding box)
        X_min, X_max = self.X.min(axis=0), self.X.max(axis=0)
        ref_dispersions = np.zeros(n_reference)
        for b in range(n_reference):
            X_ref = np.random.uniform(X_min, X_max, size=self.X.shape)
            # Assign to nearest center
            ref_dists = cdist(X_ref, self.centers, metric="euclidean")
            ref_labels = np.argmin(ref_dists, axis=1)

            # Compute ref centers
            ref_centers = np.zeros_like(self.centers)
            for k in range(self.n_clusters):
                mask = ref_labels == k
                if np.sum(mask) > 0:
                    ref_centers[k] = X_ref[mask].mean(axis=0)
                else:
                    ref_centers[k] = self.centers[k]
            ref_dispersions[b] = self._log_dispersion(X_ref, ref_labels, ref_centers)

        mean_ref = np.mean(ref_dispersions)
        gap = mean_ref - Wk
        return float(gap)

    @staticmethod
    def _log_dispersion(X, labels, centers):
        """Compute log of the within-cluster dispersion W_k."""
        n_clusters = len(centers)
        Wk = 0.0
        for k in range(n_clusters):
            mask = labels == k
            if np.sum(mask) <= 1:
                continue
            diff = X[mask] - centers[k]
            dists = np.sqrt(np.sum(diff ** 2, axis=1))
            Wk += 0.5 * np.sum(dists ** 2) / np.sum(mask)
        return np.log(Wk + 1e-10)

    # ----- BIC (for Gaussian clusters) --------------------------------------
    def bic_score(self):
        """Compute Bayesian Information Criterion (approximate).

        BIC = n * ln(W_k / n) + k * ln(n) * (d + 1) / 2

        Lower BIC is better (more negative).

        Returns
        -------
        bic : float
        """
        n, d = self.X.shape
        k = self.n_clusters
        if k < 1:
            return float("inf")
        # Within-cluster sum of squared errors
        wss = 0.0
        for j in range(k):
            mask = self.labels == j
            if np.sum(mask) > 0:
                diff = self.X[mask] - self.centers[j]
                wss += np.sum(diff ** 2)
        if wss < 1e-10:
            wss = 1e-10
        n_params = k * d + (k - 1)
        bic = n * np.log(wss / n) + n_params * np.log(n)
        return float(bic)

    # ----- All metrics in one call -------------------------------------------
    def evaluate_all(self):
        """Compute all validity indices.

        Returns
        -------
        metrics : dict
        """
        sil = self.silhouette_score()
        db = self.davies_bouldin_score()
        bic = self.bic_score()
        try:
            gap = self.gap_statistic(n_reference=20)
        except Exception:
            gap = None
        return {
            "silhouette": sil,
            "davies_bouldin": db,
            "bic": bic,
            "gap": gap,
            "n_clusters": self.n_clusters,
            "n_samples": self.n,
        }


class AutomaticClusterDiscovery:
    """Automatically determine the optimal number of clusters using
    consensus across multiple internal validation indices.

    Parameters
    ----------
    k_range : iterable of int
        Candidate values of K to evaluate.
    random_state : int, default=42
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results_ = {}

    def _fit_for_k(self, X, k, base_estimator=None):
        """Fit ADE-FCM for a specific K and return labels + centers.

        Parameters
        ----------
        X : ndarray
        k : int
        base_estimator : ADEFCM or None
            If provided, uses its configuration.

        Returns
        -------
        labels : ndarray
        centers : ndarray
        """
        if base_estimator is not None:
            from .ade_fcm import ADEFCM

            model = ADEFCM(
                n_clusters=k,
                max_iter=base_estimator.max_iter,
                m=base_estimator.m,
                epsilon=base_estimator.epsilon,
                init_method=base_estimator.init_method,
                early_stopping_patience=base_estimator.early_stopping_patience,
                outlier_threshold=base_estimator.outlier_threshold,
                random_state=self.random_state,
                verbose=False,
            )
        else:
            from .ade_fcm import ADEFCM

            model = ADEFCM(
                n_clusters=k,
                max_iter=200,
                random_state=self.random_state,
                verbose=False,
            )
        model.fit(X)
        return model.labels_, model.centers_

    def evaluate_k(self, X, k, base_estimator=None):
        """Evaluate a single K using all validity indices.

        Parameters
        ----------
        X : ndarray
        k : int
        base_estimator : ADEFCM or None

        Returns
        -------
        metrics : dict
        """
        labels, centers = self._fit_for_k(X, k, base_estimator)
        evaluator = ClusterEvaluator(X, labels, centers)
        metrics = evaluator.evaluate_all()
        metrics["k"] = k
        return metrics

    def search(self, X, k_range, base_estimator=None):
        """Evaluate all K in k_range and return the full results table.

        Parameters
        ----------
        X : ndarray
        k_range : iterable of int
        base_estimator : ADEFCM or None

        Returns
        -------
        results : dict of {k: metrics}
        """
        results = {}
        for k in k_range:
            logger.info(f"AutoCluster: evaluating K = {k}")
            try:
                metrics = self.evaluate_k(X, k, base_estimator)
                results[k] = metrics
            except Exception as exc:
                logger.warning(f"AutoCluster: K={k} failed: {exc}")
                continue
        self.results_ = results
        return results

    def consensus_search(self, X, k_range, base_estimator=None, complexity_penalty=0.02):
        """Determine optimal K via consensus voting across indices.

        Applies a complexity penalty to silhouette to prevent over-selection
        of large K values in dense latent spaces.

        For each index, computes a score and picks the best K for that
        index. The final K is the mode (most voted) across indices.

        Parameters
        ----------
        X : ndarray
        k_range : iterable of int
        base_estimator : ADEFCM or None
        complexity_penalty : float, default=0.02
            Penalty per additional cluster applied to silhouette before voting.
            Higher values penalize large K more aggressively.

        Returns
        -------
        best_k : int
        """
        results = self.search(X, k_range, base_estimator)
        if not results:
            logger.warning("AutoCluster: no valid results, default K=2")
            return 2

        votes = []

        # 1. Silhouette (penalized) -> maximize
        sil_values = {k: v["silhouette"] - complexity_penalty * k for k, v in results.items()}
        if sil_values:
            best_sil = max(sil_values, key=sil_values.get)
            votes.append(best_sil)

        # 2. Davies-Bouldin -> minimize
        db_values = {k: v["davies_bouldin"] for k, v in results.items()}
        if db_values:
            best_db = min(db_values, key=db_values.get)
            votes.append(best_db)

        # 3. BIC -> minimize (BIC naturally penalizes complexity)
        bic_values = {k: v["bic"] for k, v in results.items()}
        if bic_values:
            best_bic = min(bic_values, key=bic_values.get)
            votes.append(best_bic)

        # 4. Gap -> maximize (complexity-penalized)
        gap_values = {k: v["gap"] for k, v in results.items() if v.get("gap") is not None}
        if gap_values and len(gap_values) > 1:
            # Apply penalty: prefer fewer clusters if gaps are similar
            penalized_gap = {k: g - 0.01 * k for k, g in gap_values.items()}
            best_gap = max(penalized_gap, key=penalized_gap.get)
            votes.append(best_gap)

        if not votes:
            logger.warning("AutoCluster: no consensus, default K=2")
            return 2

        best_k = max(set(votes), key=votes.count)
        logger.info(
            f"AutoCluster consensus: votes={votes}, "
            f"sil={best_sil}, db={best_db}, bic={best_bic}"
        )
        return best_k

    def elbow_curve(self, X, k_range, base_estimator=None):
        """Return the within-cluster SSE curve for elbow method.

        Parameters
        ----------
        X : ndarray
        k_range : iterable of int
        base_estimator : ADEFCM or None

        Returns
        -------
        wss : dict {k: sse}
        """
        wss = {}
        for k in k_range:
            labels, centers = self._fit_for_k(X, k, base_estimator)
            sse = 0.0
            for j in range(k):
                mask = labels == j
                if np.sum(mask) > 0:
                    diff = X[mask] - centers[j]
                    sse += np.sum(diff ** 2)
            wss[k] = sse
        return wss
