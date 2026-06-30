"""
ADE-FCM: Adaptive Distributed Explainable Fuzzy C-Means
========================================================
Core implementation of the ADE-FCM algorithm integrating all 10 novel contributions.

Contributions:
1. KMeans++ Initialization          - ade_fcm._kmeans_pp_init()
2. Density-Based Initialization     - ade_fcm._density_init()
3. Adaptive Fuzzifier m(t)          - ade_fcm._adaptive_fuzzifier()
4. Confidence Weighted Membership   - ade_fcm._confidence_weighted_membership()
5. Automatic Cluster Discovery      - auto_cluster.AutomaticClusterDiscovery
6. Outlier Robust Membership        - ade_fcm._robust_center_update()
7. Early Stopping                   - ade_fcm.fit() patience_counter
8. Dynamic Convergence Threshold    - ade_fcm._dynamic_threshold()
9. Explainable Clustering           - xai module functions
10. Distributed Spark Optimization  - spark_ade_fcm.SparkADEFCM
"""
import numpy as np
from scipy.spatial.distance import cdist
from loguru import logger


class ADEFCM:
    """Adaptive Distributed Explainable Fuzzy C-Means clustering algorithm.

    Parameters
    ----------
    n_clusters : int or 'auto', default='auto'
        Number of clusters. If 'auto', uses AutomaticClusterDiscovery.
    max_iter : int, default=300
        Maximum number of iterations.
    m : float or 'adaptive', default='adaptive'
        Fuzzifier exponent. If 'adaptive', uses m(t) schedule.
    epsilon : float or 'dynamic', default='dynamic'
        Convergence threshold. If 'dynamic', uses epsilon(t) schedule.
    init_method : {'kmeans++', 'density', 'random'}, default='kmeans++'
        Center initialization method.
    early_stopping_patience : int, default=10
        Number of consecutive converged iterations before early stop.
    outlier_contamination : float, default=0.05
        Fraction of points trimmed as outliers during robust center update.
        Set to 0 to disable robust updating (standard FCM).
    outlier_threshold : float, default=2.0
        Standard deviation multiplier for post-hoc outlier flagging.
    m_min : float, default=1.1
        Minimum fuzzifier value in adaptive m(t) schedule. Only used
        when m='adaptive'.
    m_max : float, default=2.5
        Maximum fuzzifier value in adaptive m(t) schedule. Only used
        when m='adaptive'.
    alpha : float, default=3.0
        Decay rate in adaptive m(t) schedule. Only used when m='adaptive'.
    eps_0 : float, default=1e-3
        Initial convergence threshold in epsilon(t) schedule. Only used
        when epsilon='dynamic'.
    beta : float, default=5.0
        Decay rate in dynamic threshold schedule. Only used when
        epsilon='dynamic'.
    center_reinit_threshold : float, default=1.0
        Minimum membership sum for a cluster. Centers below this are
        re-initialized to the point farthest from all existing centers.
        Set to 0 to disable.
    metric : {'euclidean', 'manhattan', 'cosine', 'mahalanobis'}, default='euclidean'
        Distance metric for membership computation.
    compute_xai : bool, default=True
        If True, compute feature importance and cluster summaries after fit.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to log iteration details.
    """
    def __init__(
        self,
        n_clusters="auto",
        max_iter=300,
        m="adaptive",
        epsilon="dynamic",
        init_method="kmeans++",
        early_stopping_patience=10,
        outlier_contamination=0.05,
        outlier_threshold=2.0,
        m_min=1.1,
        m_max=2.5,
        alpha=3.0,
        eps_0=1e-3,
        beta=5.0,
        center_reinit_threshold=1.0,
        metric="euclidean",
        compute_xai=True,
        random_state=42,
        verbose=True,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.epsilon = epsilon
        self.init_method = init_method
        self.early_stopping_patience = early_stopping_patience
        self.outlier_contamination = outlier_contamination
        self.outlier_threshold = outlier_threshold
        self.m_min = m_min
        self.m_max = m_max
        self.alpha = alpha
        self.eps_0 = eps_0
        self.beta = beta
        self.center_reinit_threshold = center_reinit_threshold
        self.metric = metric
        self.compute_xai = compute_xai
        self.random_state = random_state
        self.verbose = verbose
        self.centers_ = None
        self.U_ = None
        self.labels_ = None
        self.J_history_ = []
        self.n_iter_ = 0
        self.outlier_mask_ = None
        self.outlier_scores_ = None
        self.feature_importances_ = None
        self.cluster_summaries_ = None
        self.convergence_history_ = []
        self._metric_kwargs = {}

    def _set_mahalanobis_VI(self, X):
        """Pre-compute inverse covariance for Mahalanobis distance."""
        cov = np.cov(X, rowvar=False)
        try:
            self._metric_kwargs['VI'] = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            self._metric_kwargs['VI'] = np.eye(cov.shape[0])

    METRIC_ALIASES = {
        'manhattan': 'cityblock', 'l1': 'cityblock',
        'euclidean': 'euclidean', 'l2': 'euclidean',
        'cosine': 'cosine',
        'mahalanobis': 'mahalanobis',
    }

    def _compute_distances(self, X, centers):
        """Compute distance matrix using the configured metric.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        centers : ndarray of shape (n_clusters, n_features)

        Returns
        -------
        dist : ndarray of shape (n_samples, n_clusters)
        """
        metric = self.METRIC_ALIASES.get(self.metric, self.metric)
        kwargs = self._metric_kwargs if self.metric == 'mahalanobis' else {}
        return cdist(X, centers, metric=metric, **kwargs)

    def _adaptive_fuzzifier(self, t):
        """Compute time-varying fuzzifier: m(t) = m_min + (m_max - m_min) * exp(-alpha * t / T)."""
        if not isinstance(self.m, str):
            return float(self.m)
        ratio = t / max(self.max_iter - 1, 1)
        return self.m_min + (self.m_max - self.m_min) * np.exp(-self.alpha * ratio)

    def _dynamic_threshold(self, t):
        """Compute time-varying threshold: epsilon(t) = eps_0 * exp(-beta * t / T)."""
        if not isinstance(self.epsilon, str):
            return float(self.epsilon)
        ratio = t / max(self.max_iter - 1, 1)
        return max(self.eps_0 * np.exp(-self.beta * ratio), 1e-8)

    def _kmeans_pp_init(self, X):
        """KMeans++ initialization (Arthur & Vassilvitskii 2007)."""
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]
        if self.n_clusters > n:
            self.n_clusters = n
        centers = [X[rng.randint(n)]]
        for _ in range(1, self.n_clusters):
            dists = np.array([min(np.sum((x - c) ** 2) for c in centers) for x in X])
            probs = dists / np.sum(dists)
            next_idx = rng.choice(n, p=probs)
            centers.append(X[next_idx])
        return np.array(centers)

    def _density_init(self, X):
        """Density-based initialization from high-density regions."""
        n = X.shape[0]
        rng = np.random.RandomState(self.random_state)
        if n > 1000:
            idx = rng.choice(n, 1000, replace=False)
            sample = X[idx]
        else:
            sample = X
        dists = []
        max_pairs = min(len(sample), 200)
        for i in range(max_pairs):
            for j in range(i + 1, max_pairs):
                dists.append(np.sqrt(np.sum((sample[i] - sample[j]) ** 2)))
        d_c = max(np.percentile(dists, 2) if dists else 1.0, 1e-10)
        density = np.zeros(n)
        for i in range(n):
            diff = X - X[i]
            dists_i = np.sqrt(np.sum(diff ** 2, axis=1))
            density[i] = np.sum(np.exp(-(dists_i / d_c) ** 2))
        threshold = np.percentile(density, 90)
        candidates = np.where(density >= threshold)[0]
        if len(candidates) < self.n_clusters:
            candidates = np.argsort(density)[-self.n_clusters:]
        selected = rng.choice(candidates, self.n_clusters, replace=False)
        return X[selected].copy()

    def _confidence_weighted_membership(self, U):
        """Weight membership values by confidence score per point.
        conf_i = 1 - (2/pi) * arctan(sigma_i / mu_i)."""
        U_safe = np.maximum(U, 1e-10)
        mu = np.mean(U_safe, axis=1)
        sigma = np.std(U_safe, axis=1)
        ratio = np.where(mu > 1e-10, sigma / mu, 1e10)
        ratio = np.clip(ratio, 0, 1e10)
        conf = 1.0 - (2.0 / np.pi) * np.arctan(ratio)
        conf = np.maximum(conf, 1e-10)
        return U_safe * conf[:, np.newaxis]

    def _update_membership(self, X, centers, m):
        """Vectorized FCM membership update via scipy cdist.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        centers : ndarray of shape (n_clusters, n_features)
        m : float

        Returns
        -------
        U : ndarray of shape (n_samples, n_clusters)
        """
        n, c = X.shape[0], centers.shape[0]
        dist = self._compute_distances(X, centers)
        dist = np.maximum(dist, 1e-10)
        inv = 2.0 / max(m - 1, 0.1)
        ratios = dist[:, :, np.newaxis] / dist[:, np.newaxis, :]
        denom = np.sum(ratios ** inv, axis=2)
        U = np.where(denom > 1e-10, 1.0 / denom, 0.0)
        return U

    def _update_centers(self, X, U, m):
        """Weighted mean center update.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        U : ndarray of shape (n_samples, n_clusters)
        m : float

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        """
        Um = U ** m
        numerator = X.T @ Um
        denominator = np.maximum(Um.sum(axis=0), 1e-10)
        return (numerator / denominator).T

    def _robust_center_update(self, X, U, m, contamination=0.05):
        """Outlier-robust center update using trimmed FCM.

        Computes weighted distances per point, flags the top `contamination`
        fraction as outliers, and updates centers using only the inlier points.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        U : ndarray of shape (n_samples, n_clusters)
        m : float
        contamination : float
            Fraction of points to trim (0 = disabled, uses standard update).

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        outlier_mask : ndarray of bool, shape (n_samples,)
        outlier_scores : ndarray of shape (n_samples,)
        """
        c = U.shape[1]
        Um = U ** m
        scores = np.zeros(X.shape[0])

        # Compute weighted distance scores
        dist = np.maximum(self._compute_distances(X, self.centers_), 1e-10)

        # Temporarily skip confidence weighting for distance computation
        tdist = np.maximum(self._compute_distances(X, self.centers_), 1e-10)
        for j in range(c):
            scores += Um[:, j] * tdist[:, j]

        n_out = max(1, int(len(X) * contamination))
        threshold = np.sort(scores)[-n_out] if n_out < len(scores) else np.max(scores)
        outlier_mask = scores >= threshold

        # Trimmed update: set membership to 0 for outliers, renormalize
        U_trimmed = U.copy()
        U_trimmed[outlier_mask] = 0.0
        row_sum = U_trimmed.sum(axis=1, keepdims=True)
        row_sum = np.maximum(row_sum, 1e-10)
        U_trimmed = U_trimmed / row_sum

        centers = self._update_centers(X, U_trimmed, m)
        return centers, outlier_mask, scores

    def _reinitialize_centers(self, X, centers, U, threshold=1.0):
        """Re-initialize degenerate centers whose membership sum < threshold.

        For each degenerate center, replace it with the point farthest from
        all surviving centers (furthest-first heuristic).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        centers : ndarray of shape (n_clusters, n_features)
        U : ndarray of shape (n_samples, n_clusters)
        threshold : float
            Minimum membership sum. Centers below this are reinitialized.

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        n_reinit : int
            Number of centers reinitialized.
        """
        membership_sum = U.sum(axis=0)
        weak = np.where(membership_sum < threshold)[0]
        n_reinit = len(weak)
        if n_reinit == 0:
            return centers, 0

        strong_idx = np.where(membership_sum >= threshold)[0]
        if len(strong_idx) == 0:
            strong_idx = np.arange(len(centers))

        # Compute distance of each point to nearest strong center
        dist_to_strong = cdist(X, centers[strong_idx], metric='euclidean')
        min_dist = dist_to_strong.min(axis=1)
        # Farthest points get re-initialized centers
        reinit_idx = np.argsort(min_dist)[-n_reinit:]
        rng = np.random.RandomState(self.random_state)

        for i, k in enumerate(weak):
            if i < len(reinit_idx):
                centers[k] = X[reinit_idx[i]].copy()
            else:
                centers[k] = X[rng.randint(len(X))].copy()

        if self.verbose:
            logger.info(f"Re-initialized {n_reinit} degenerate centers")
        return centers, n_reinit

    def _compute_objective(self, X, U, centers, m):
        """Compute fuzzy clustering objective J with entropy regularization.

        J = sum_j sum_i u_ij^m * d_ij + 0.01 * sum_i sum_j u_ij * log(u_ij)
        """
        dist = np.maximum(self._compute_distances(X, centers), 1e-10)
        J = np.sum(U ** m * dist)
        U_safe = np.maximum(U, 1e-10)
        entropy = -np.sum(U * np.log(U_safe))
        return J + 0.01 * entropy

    def _outlier_detection(self, X, centers, U, m):
        """Post-hoc outlier flagging (diagnostic, not used in clustering)."""
        scores = np.zeros(X.shape[0])
        dist = np.maximum(self._compute_distances(X, centers), 1e-10)
        Um = U ** m
        for j in range(centers.shape[0]):
            scores += Um[:, j] * dist[:, j]
        mu = np.mean(scores)
        sigma = np.std(scores)
        threshold = mu + self.outlier_threshold * sigma
        return scores > threshold, scores

    def fit(self, X, y=None):
        """Fit ADE-FCM to data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self : ADEFCM
        """
        n, d = X.shape
        logger.info(f"ADE-FCM fitting: {n} samples, {d} features, metric={self.metric}")

        # Pre-compute Mahalanobis inverse covariance
        if self.metric == 'mahalanobis':
            self._set_mahalanobis_VI(X)

        if self.n_clusters == "auto":
            from .auto_cluster import AutomaticClusterDiscovery
            discover = AutomaticClusterDiscovery(random_state=self.random_state)
            k_range = range(2, min(15, int(np.sqrt(n)) + 2))
            if len(k_range) < 2:
                k_range = range(2, min(10, max(3, n // 2)))
            self.n_clusters = discover.consensus_search(X, k_range, base_estimator=self, complexity_penalty=0.02)
            logger.info(f"Auto-discovered K = {self.n_clusters}")

        if self.n_clusters is None or self.n_clusters < 2:
            self.n_clusters = 2

        self.centers_ = self._initialize_centers(X)

        rng = np.random.RandomState(self.random_state)
        self.U_ = rng.rand(n, self.n_clusters)
        self.U_ = self.U_ / self.U_.sum(axis=1, keepdims=True)

        self.J_history_ = []
        self.convergence_history_ = []
        patience_counter = 0
        iteration = 0

        for iteration in range(self.max_iter):
            m_t = self._adaptive_fuzzifier(iteration)
            eps_t = self._dynamic_threshold(iteration)

            # Update membership
            U_new = self._update_membership(X, self.centers_, m_t)

            # Confidence weighting
            U_new = self._confidence_weighted_membership(U_new)
            row_sum = U_new.sum(axis=1, keepdims=True)
            row_sum = np.maximum(row_sum, 1e-10)
            U_new = U_new / row_sum

            # Convergence check
            change = np.linalg.norm(U_new - self.U_, "fro")
            self.convergence_history_.append(change)
            self.U_ = U_new

            # Robust center update (outlier-aware)
            if self.outlier_contamination > 0:
                self.centers_, _, _ = self._robust_center_update(
                    X, self.U_, m_t, contamination=self.outlier_contamination
                )
            else:
                self.centers_ = self._update_centers(X, self.U_, m_t)

            # Center reinitialization for degenerate clusters
            if self.center_reinit_threshold > 0:
                self.centers_, n_reinit = self._reinitialize_centers(
                    X, self.centers_, self.U_, threshold=self.center_reinit_threshold
                )
                if n_reinit > 0:
                    # After reinit, recompute membership and also apply a
                    # membership floor to prevent immediate re-collapse
                    U_new = self._update_membership(X, self.centers_, m_t)
                    c = U_new.shape[1]
                    min_per_center = max(1.0, n * 0.005)
                    for k in range(c):
                        col_sum = U_new[:, k].sum()
                        if col_sum < min_per_center:
                            shortfall = min_per_center - col_sum
                            U_new[:, k] += shortfall / n
                    U_new = self._confidence_weighted_membership(U_new)
                    row_sum = U_new.sum(axis=1, keepdims=True)
                    row_sum = np.maximum(row_sum, 1e-10)
                    self.U_ = U_new / row_sum

            # Objective
            J = self._compute_objective(X, self.U_, self.centers_, m_t)
            self.J_history_.append(J)

            # Early stopping
            if change < eps_t:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        logger.info(
                            f"Early stopping at iteration {iteration + 1} "
                            f"(change={change:.6f} < eps={eps_t:.6f})"
                        )
                    break
            else:
                patience_counter = 0

            if self.verbose and iteration % 20 == 0:
                logger.info(
                    f"Iter {iteration + 1:3d} | J={J:.4f} | m={m_t:.3f} | "
                    f"eps={eps_t:.2e} | change={change:.2e}"
                )

        self.n_iter_ = iteration + 1
        self.labels_ = np.argmax(self.U_, axis=1)

        # Post-hoc outlier flagging
        self.outlier_mask_, self.outlier_scores_ = self._outlier_detection(
            X, self.centers_, self.U_, m_t
        )

        n_outliers = int(self.outlier_mask_.sum())
        logger.info(
            f"ADE-FCM converged in {self.n_iter_}/{self.max_iter} iterations, "
            f"J*={self.J_history_[-1]:.4f}, outliers={n_outliers}/{n}, "
            f"metric={self.metric}"
        )

        if self.compute_xai:
            try:
                from .xai import feature_importance, cluster_summary
                self.feature_importances_ = feature_importance(
                    X, self.labels_, self.centers_, method="permutation", U=self.U_
                )
                self.cluster_summaries_ = cluster_summary(
                    X, self.labels_, self.centers_
                )
            except Exception as exc:
                logger.warning(f"XAI summarization skipped: {exc}")

        return self

    def _initialize_centers(self, X):
        """Dispatch to the selected initialization method."""
        if self.init_method == "kmeans++":
            return self._kmeans_pp_init(X)
        elif self.init_method == "density":
            return self._density_init(X)
        else:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X), self.n_clusters, replace=False)
            return X[idx].copy()

    def predict(self, X):
        """Predict cluster labels for new data."""
        U = self._update_membership(X, self.centers_, self._adaptive_fuzzifier(self.n_iter_))
        return np.argmax(U, axis=1)

    def fit_predict(self, X, y=None):
        """Fit and return labels."""
        self.fit(X, y)
        return self.labels_

    def explain(self, X, feature_names=None):
        """Return a human-readable explanation of the clustering result."""
        from .xai import explain_clusters
        return explain_clusters(
            X, self.labels_, self.centers_,
            feature_names=feature_names, outlier_mask=self.outlier_mask_,
        )
